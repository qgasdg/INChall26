"""DreamZero-SO101 적재 — 스트리밍 + fp8. 32GB GPU / 62GB RAM 에서 돌아가게.

왜 다시 쓰나: 앞선 load_model.py 는 모델 33GB 와 state_dict 33GB 를 동시에 들어 66GB 를
썼고, V100 컨테이너(한도 44.7GB)에서 OOM 으로 죽었다(oom_kill 11). 5090 도 RAM 62GB 라
같은 방식으로는 죽는다.

두 가지를 고친다.
  1) **스트리밍 적재** — 모델을 meta 로 만들고 샤드를 하나씩 읽어 즉시 assign 하고 버린다.
     피크가 (모델) + (샤드 1개) 로 떨어진다.
  2) **fp8 상주** — 저장소의 vram_management 가 이미 fp8 을 지원한다(AutoWrappedLinear.
     enable_fp8, per-row 스케일링). 가중치는 fp8 로 GPU 에 두고 계산만 bf16 으로 올린다.
     16.48B x 1바이트 = 16.5GB 라 32GB 에 여유 있게 들어간다.

fp8 은 Blackwell(sm_120, 5090)에서 지원된다. V100(sm_70)은 fp8 이 없으므로 --dtype bfloat16
과 --max-gpu-params 로 일부만 GPU 에 올리는 오프로딩 경로를 쓴다.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path


def human(n: float) -> str:
    return "%.1fGB" % (n / 2**30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dreamzero 저장소 경로")
    ap.add_argument("--base", required=True, help="Wan2.1-I2V-14B-480P 경로")
    ap.add_argument("--lora", required=True, help="dreamzero-so101-lora 경로")
    ap.add_argument("--store-dtype", default="float8_e4m3fn",
                    help="GPU 상주 dtype (fp8 미지원 GPU 는 bfloat16)")
    ap.add_argument("--compute-dtype", default="bfloat16")
    ap.add_argument("--max-gpu-params", type=float, default=None,
                    help="GPU 에 상주시킬 최대 파라미터 수(억 단위 아님, 개수). 초과분은 CPU 오프로드")
    args = ap.parse_args()

    sys.path.insert(0, args.root)
    import torch
    from safetensors.torch import load_file

    store_dt = getattr(torch, args.store_dtype)
    comp_dt = getattr(torch, args.compute_dtype)
    BASE, LORA = Path(args.base), Path(args.lora)

    print("=== 환경 ===")
    cc = torch.cuda.get_device_capability(0)
    print("  GPU:", torch.cuda.get_device_name(0), "sm_%d%d" % cc,
          "· VRAM", human(torch.cuda.get_device_properties(0).total_memory))
    if args.store_dtype.startswith("float8") and cc[0] < 8:
        raise SystemExit("★fp8 은 sm_89+ 에서만 쓸 수 있다. --store-dtype bfloat16 으로 실행할 것")

    cfg = json.loads((LORA / "config.json").read_text())["action_head_cfg"]["config"]
    dm = {k: v for k, v in cfg["diffusion_model_cfg"].items() if not k.startswith("_")}
    dm["diffusion_model_pretrained_path"] = None

    from groot.vla.model.dreamzero.modules.utils import init_weights_on_device
    from groot.vla.model.dreamzero.modules.vram_management import (
        AutoWrappedLinear, AutoWrappedModule, enable_vram_management)
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

    print("\n=== 1. 모델 뼈대 (meta — 메모리 0) ===")
    t0 = time.time()
    with init_weights_on_device(device=torch.device("meta")):
        model = CausalWanModel(**dm)
    n_param = sum(p.numel() for p in model.parameters())
    print("  파라미터 %.2fB · %.1f초 · 메모리 사용 0" % (n_param / 1e9, time.time() - t0))

    print("\n=== 2. 베이스 DiT 스트리밍 적재 (샤드 하나씩) ===")
    idx = json.loads((BASE / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(idx.values()))
    loaded = 0
    for i, s in enumerate(shards, 1):
        sd = load_file(str(BASE / s))
        sd = {k: v.to(comp_dt) for k, v in sd.items()}
        res = model.load_state_dict(sd, strict=False, assign=True)
        loaded += len(sd) - len(res.unexpected_keys)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print("  [%d/%d] %-46s 누적 %4d개 · GPU %.1fGB" % (i, len(shards), s, loaded, peak), flush=True)
        del sd
    print("  베이스 적재 %d개" % loaded)

    print("\n=== 3. SO-101 액션 모듈 + LoRA 병합 ===")
    lw = load_file(str(LORA / "model.safetensors"))
    P = "action_head.model.base_model.model."
    lw = {k[len(P):]: v for k, v in lw.items() if k.startswith(P)}
    plain = {k: v.to(comp_dt) for k, v in lw.items() if ".lora_" not in k}
    res = model.load_state_dict(plain, strict=False, assign=True)
    print("  액션·상태 모듈 %d개 적재 (남음 %d)" % (len(plain) - len(res.unexpected_keys), len(res.unexpected_keys)))

    rank, alpha = cfg["lora_rank"], cfg["lora_alpha"]
    scale = alpha / rank
    pairs = {}
    for k, v in lw.items():
        m = re.match(r"^(.*?)\.lora_(A|B)(?:\.[^.]+)?\.weight$", k)
        if m:
            pairs.setdefault(m.group(1), {})[m.group(2)] = v
    msd = model.state_dict()
    merged = 0
    with torch.no_grad():
        for name, ab in pairs.items():
            wk = name + ".weight"
            if "A" in ab and "B" in ab and wk in msd and not msd[wk].is_meta:
                msd[wk] += ((ab["B"].float() @ ab["A"].float()) * scale).to(msd[wk].dtype)
                merged += 1
    print("  LoRA 병합 %d개 (rank=%d alpha=%d scale=%.1f)" % (merged, rank, alpha, scale))

    still_meta = [k for k, v in model.state_dict().items() if v.is_meta]
    print("  ★남은 meta 텐서: %d개%s" % (len(still_meta), (" -> " + str(still_meta[:4])) if still_meta else " (전부 실체화됨)"))

    print("\n=== 4. GPU 배치 (%s 상주 / %s 계산) ===" % (args.store_dtype, args.compute_dtype))
    t0 = time.time()
    enable_vram_management(
        model,
        module_map={torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                    torch.nn.Conv3d: AutoWrappedModule},
        module_config=dict(offload_dtype=store_dt, offload_device="cuda",
                           onload_dtype=store_dt, onload_device="cuda",
                           computation_dtype=comp_dt, computation_device="cuda"),
        max_num_param=args.max_gpu_params,
        overflow_module_config=dict(offload_dtype=store_dt, offload_device="cpu",
                                    onload_dtype=store_dt, onload_device="cpu",
                                    computation_dtype=comp_dt, computation_device="cuda"),
    )
    # enable_vram_management 는 래퍼만 씌운다. 실제 배치는 onload() 를 불러야 일어난다
    # (저장소의 load_models_to_device 가 같은 일을 한다).
    n_on = 0
    for m in model.modules():
        if hasattr(m, "onload"):
            m.onload()
            n_on += 1
    torch.cuda.synchronize()
    print("  onload 호출 %d개 모듈 · %.1f초" % (n_on, time.time() - t0))
    print("  ★GPU 사용 %s / %s"
          % (human(torch.cuda.memory_allocated()), human(torch.cuda.get_device_properties(0).total_memory)))
    import resource
    print("  CPU 최대 %.1fGB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20))
    print("\n★적재 성공")


if __name__ == "__main__":
    main()
