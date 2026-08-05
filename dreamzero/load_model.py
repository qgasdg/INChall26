"""SO-101 DreamZero 실제 가중치 적재 — 베이스 1303 + 액션모듈 14 + LoRA 병합.

키 대조는 통과했다(probe_lora_keys.py: 베이스 1303/1303, LoRA 814/814). 이제 shape 까지
맞는지, 그리고 32GB V100 에 실제로 올라가는지를 본다.

LoRA 는 peft 로 주입하지 않고 **베이스 가중치에 직접 병합**한다(W += B@A * alpha/rank).
추론만 할 거라 병합이 더 단순하고 메모리도 덜 쓴다. rank=alpha=4 라 스케일은 1.0.

메모리 전략(V100 32GB): DiT 14B fp16 = 28GB 를 GPU 에, T5(10.6GB)·CLIP(4.4GB)은 CPU 에
둔다. 이 스크립트는 DiT 만 다룬다.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

DZ = Path("/home/jovyan/work/model_poc_20260726/DreamZero-SO101")
BASE = Path("/home/jovyan/work/pocd/checkpoints/Wan2.1-I2V-14B-480P")
LORA = DZ / "checkpoints/dreamzero-so101-lora"
sys.path.insert(0, str(DZ / "dreamzero-official"))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


def mem(tag: str) -> None:
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    g = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    print("    [%-22s] CPU 최대 %.1fGB · GPU %.1fGB" % (tag, rss, g), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to-gpu", action="store_true", help="병합 후 GPU 로 올려본다")
    ap.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    args = ap.parse_args()
    dt = getattr(torch, args.dtype)

    print("=== 시스템 ===")
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemTotal", "MemAvailable")):
                print("   ", line.strip())
    if torch.cuda.is_available():
        print("    GPU:", torch.cuda.get_device_name(0),
              "%.1fGB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))

    cfg = json.loads((LORA / "config.json").read_text())["action_head_cfg"]["config"]
    dm = {k: v for k, v in cfg["diffusion_model_cfg"].items() if not k.startswith("_")}
    dm["diffusion_model_pretrained_path"] = None

    print("\n=== 1. 모델 구성 (%s, CPU) ===" % args.dtype)
    t0 = time.time()
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel
    torch.set_default_dtype(dt)
    model = CausalWanModel(**dm)
    model = model.to(dt).eval()
    n = sum(p.numel() for p in model.parameters())
    print("    파라미터 %.2fB · %.1f초" % (n / 1e9, time.time() - t0))
    mem("모델 구성")

    print("\n=== 2. 베이스 DiT 적재 (샤드 7개) ===")
    idx = json.loads((BASE / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(idx.values()))
    sd = {}
    for i, s in enumerate(shards, 1):
        sd.update(load_file(str(BASE / s)))
        print("    [%d/%d] %s" % (i, len(shards), s), flush=True)
    sd = {k: v.to(dt) for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    print("    적재 %d · 미적재(모델에만) %d · 남음(ckpt에만) %d"
          % (len(sd) - len(res.unexpected_keys), len(res.missing_keys), len(res.unexpected_keys)))
    if res.unexpected_keys:
        print("    ★남은 키 예시:", res.unexpected_keys[:5])
    del sd
    mem("베이스 적재")

    print("\n=== 3. SO-101 액션 모듈 14개 + LoRA 800개 ===")
    lw = load_file(str(LORA / "model.safetensors"))
    PREF = "action_head.model.base_model.model."
    lw = {k[len(PREF):]: v for k, v in lw.items() if k.startswith(PREF)}

    plain = {k: v for k, v in lw.items() if ".lora_" not in k}
    msd = model.state_dict()
    ok = bad = 0
    for k, v in plain.items():
        if k not in msd:
            print("    ✗ 모델에 없음:", k); bad += 1; continue
        if tuple(msd[k].shape) != tuple(v.shape):
            print("    ✗ shape 불일치 %s: 모델 %s vs 파일 %s" % (k, tuple(msd[k].shape), tuple(v.shape)))
            bad += 1; continue
        ok += 1
    print("    액션·상태 모듈: 일치 %d / 불일치 %d" % (ok, bad))
    if bad == 0:
        model.load_state_dict({k: v.to(dt) for k, v in plain.items()}, strict=False)

    # LoRA 병합: W += (alpha/rank) * B @ A
    rank, alpha = cfg["lora_rank"], cfg["lora_alpha"]
    scale = alpha / rank
    pairs = {}
    for k, v in lw.items():
        m = re.match(r"^(.*?)\.lora_(A|B)(?:\.[^.]+)?\.weight$", k)
        if m:
            pairs.setdefault(m.group(1), {})[m.group(2)] = v
    merged = skipped = 0
    with torch.no_grad():
        for base_name, ab in pairs.items():
            wk = base_name + ".weight"
            if "A" not in ab or "B" not in ab or wk not in msd:
                skipped += 1; continue
            delta = (ab["B"].float() @ ab["A"].float()) * scale
            if tuple(delta.shape) != tuple(msd[wk].shape):
                print("    ✗ delta shape %s: %s vs %s" % (wk, tuple(delta.shape), tuple(msd[wk].shape)))
                skipped += 1; continue
            msd[wk] += delta.to(dt)
            merged += 1
    print("    LoRA 병합 %d개 모듈 (rank=%d alpha=%d scale=%.1f) · 건너뜀 %d"
          % (merged, rank, alpha, scale, skipped))
    mem("LoRA 병합")

    print("\n=== 4. 최종 확인 ===")
    bad_stat = [k for k, v in model.state_dict().items() if torch.isnan(v).any() or torch.isinf(v).any()]
    print("    NaN/Inf 텐서: %d개" % len(bad_stat))
    if bad_stat:
        print("    예시:", bad_stat[:5])

    if args.to_gpu:
        print("\n=== 5. GPU 적재 ===")
        t0 = time.time()
        try:
            model = model.cuda()
            torch.cuda.synchronize()
            print("    ★성공 · %.1f초" % (time.time() - t0))
            print("    GPU 사용 %.1fGB / %.1fGB"
                  % (torch.cuda.memory_allocated() / 2**30,
                     torch.cuda.get_device_properties(0).total_memory / 2**30))
        except RuntimeError as e:
            print("    ★실패:", str(e)[:200])
    print("\n완료")


if __name__ == "__main__":
    main()
