"""학습 체크포인트에서 **바뀐 부분만** 뽑아낸다.

`save_only_unet: True` 라도 체크포인트는 UNet 전체(5.4GB)를 담는다. 그런데 우리 학습은
`action_embed` 1.66M 만 건드렸고 나머지 1438.86M 은 동결이라 `backbone.ckpt` 와 같다.
그래서 액션 관련 텐서만 남기면 **5.4GB → 약 7MB** 가 된다.

지우기 전에 **동결이 실제로 지켜졌는지 검증**한다 — 무작위로 고른 동결 텐서들이
backbone 과 비트 단위로 같아야 한다. 하나라도 다르면 중단한다(그 체크포인트는
"액션만 학습"이 아니므로 지우면 안 된다).

사용:
    python extract_action_ckpt.py <ckpt> <출력.pt> [--backbone <backbone.ckpt>]
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch

ACTION_KEYS = ("action_embed", "null_action_emb")


def load_sd(path: str) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("state_dict", "module"):
        if isinstance(raw, dict) and k in raw:
            return raw[k]
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--backbone", default=os.path.expanduser("~/ft/checkpoints/backbone.ckpt"))
    ap.add_argument("--verify-n", type=int, default=40, help="대조할 동결 텐서 개수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sd = load_sd(args.ckpt)
    act = {k: v for k, v in sd.items() if any(a in k for a in ACTION_KEYS)}
    if not act:
        raise SystemExit(f"액션 텐서가 없다: {args.ckpt}")

    # ── 동결 검증 ────────────────────────────────────────────────────────
    base = load_sd(args.backbone)
    common = [k for k in sd if k in base and not any(a in k for a in ACTION_KEYS)]
    random.seed(args.seed)
    sample = random.sample(common, min(args.verify_n, len(common)))
    bad = []
    for k in sample:
        a, b = sd[k], base[k]
        if a.shape != b.shape or not torch.equal(a.float(), b.float()):
            bad.append(k)
    if bad:
        raise SystemExit(f"★동결 위반 — backbone 과 다른 텐서 {len(bad)}개: {bad[:3]} … 지우지 말 것")
    print(f"  동결 검증 통과 — 표본 {len(sample)}개가 backbone 과 완전히 일치")

    torch.save(act, args.out)
    size = Path(args.out).stat().st_size / 1e6
    total = sum(v.numel() for v in act.values())
    print(f"  액션 텐서 {len(act)}개 · {total/1e6:.2f}M 파라미터 · {size:.1f}MB → {args.out}")
    for k, v in act.items():
        print(f"    {k}  {tuple(v.shape)}  |w| 평균 {v.float().abs().mean():.3e}")


if __name__ == "__main__":
    main()
