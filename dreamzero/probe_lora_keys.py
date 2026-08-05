"""SO-101 LoRA 가 공개 dreamzero 코드 위에 실리는가 — 키 대조만으로 판정.

so101 통합 코드(patches/so101_embodiment.patch, infer_demo.py)가 공개되지 않았다(2026-08-05 확인:
Vizuara-AI-Lab/dreamzero-so101 은 프로젝트 웹사이트라 코드가 없다). 그래서 LoRA 를 공개
dreamzero 저장소의 CausalWanModel 위에 직접 얹을 수 있는지 먼저 확인해야 한다.

★메타 디바이스로 모듈 트리만 세운다. 14B 를 실제로 할당하지 않으므로 메모리·시간이 거의 안 든다.
판정: LoRA 대상 모듈이 전부 모델에 존재하면 -> 패치 없이 추론 경로를 만들 수 있다.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

DZ = Path("/home/jovyan/work/model_poc_20260726/DreamZero-SO101")
REPO = DZ / "dreamzero-official"
LORA = DZ / "checkpoints/dreamzero-so101-lora"
BASE = Path("/home/jovyan/work/pocd/checkpoints/Wan2.1-I2V-14B-480P")

sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402


def main() -> None:
    cfg = json.loads((LORA / "config.json").read_text())["action_head_cfg"]["config"]
    dm = dict(cfg["diffusion_model_cfg"])
    dm.pop("_target_", None)
    dm.pop("_convert_", None)
    # 가중치는 안 싣는다 — 뼈대만 세워 키를 본다
    dm["diffusion_model_pretrained_path"] = None

    print("=== CausalWanModel 인자 ===")
    for k, v in sorted(dm.items()):
        print("  %-32s = %s" % (k, v))

    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

    print("\n=== 메타 디바이스로 모듈 트리 생성 ===")
    with torch.device("meta"):
        model = CausalWanModel(**dm)
    msd = dict(model.state_dict())
    print("  모델 텐서 %d개" % len(msd))
    mods = {k.rsplit(".", 1)[0] for k in msd}

    with safe_open(str(LORA / "model.safetensors"), "pt") as f:
        lk = list(f.keys())
    print("  LoRA 파일 텐서 %d개" % len(lk))

    PREF = "action_head.model.base_model.model."
    hit = miss = 0
    missing = []
    lora_targets = Counter()
    for k in lk:
        if not k.startswith(PREF):
            missing.append(("접두사 불일치", k))
            miss += 1
            continue
        rest = k[len(PREF):]
        # peft 형식: <모듈경로>.lora_A[.<adapter>].weight  /  비-LoRA 는 그대로
        m = re.match(r"^(.*?)\.lora_(A|B)(\.[^.]+)?\.weight$", rest)
        base = m.group(1) if m else rest.rsplit(".", 1)[0]
        if m:
            lora_targets[base.rsplit(".", 1)[-1]] += 1
        if base in mods:
            hit += 1
        else:
            miss += 1
            missing.append(("모듈 없음", rest))

    print("\n=== 대조 결과 ===")
    print("  ★맞음 %d / %d  (%.1f%%)" % (hit, len(lk), 100 * hit / len(lk)))
    print("  안 맞음 %d" % miss)
    print("\n  LoRA 가 붙은 모듈 종류:", dict(lora_targets.most_common(10)))
    if missing:
        print("\n  안 맞는 것 예시:")
        for why, k in missing[:12]:
            print("    [%s] %s" % (why, k))

    print("\n=== 베이스 체크포인트 존재 확인 ===")
    for name in ("diffusion_pytorch_model.safetensors.index.json",
                 "models_t5_umt5-xxl-enc-bf16.pth",
                 "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                 "Wan2.1_VAE.pth"):
        p = BASE / name
        print("  %-56s %s" % (name, "✓ %.1fGB" % (p.stat().st_size / 2**30) if p.exists() else "✗"))

    print("\n=== 판정 ===")
    if miss == 0:
        print("  ★전부 일치 — 없어진 패치 없이 공개 코드 위에 LoRA 를 얹을 수 있다.")
        print("    다음: 실제 가중치 적재 -> 액션 조건 생성 스크립트 작성")
    else:
        print("  ★불일치 %d건 — 위 목록을 보고 어느 계층이 어긋나는지 판단해야 한다." % miss)


if __name__ == "__main__":
    main()
