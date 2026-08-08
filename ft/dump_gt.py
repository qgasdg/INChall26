"""train 데이터의 정답 클립을 우리 생성물과 **같은 형식**으로 뽑는다.

같은 전처리(320×512, 패딩)·같은 8×2 그리드·같은 mp4 인코딩을 쓰므로 나란히 놓고 비교할 수 있다.
"정답 모션이 실제로 어느 정도 크기인가"를 눈과 숫자로 확인하는 용도다.

주의: train 은 128개 데이터셋이라 장면이 eval 과 다르다. 움직임의 **크기·성격**을 보는 것이지
장면을 비교하는 것이 아니다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

BASELINE = Path(os.environ.get("FT_BASELINE", os.path.expanduser("~/ft/kit/baseline")))
KIT = BASELINE / "challenge_kit"
for p in (KIT / "libs/dynamicrafter", KIT / "src", BASELINE / "shared_libs/video_utils", KIT,
          Path(os.path.expanduser("~/ft"))):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from lvdm.utils.utils import instantiate_from_config  # noqa: E402
from scripts.eval.feature_csv_utils import save_video_tensor  # noqa: E402
from gen_step0 import save_grid  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser("~/ft/configs/ft-20-residual.yaml"))
    ap.add_argument("--out", default=os.path.expanduser("~/ft/out/gt"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = OmegaConf.load(args.config)
    data = instantiate_from_config(cfg.data)
    data.setup()
    ds = data.val_dataset          # 학습에 안 쓴 홀드아웃에서 뽑는다

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = np.random.RandomState(args.seed).choice(len(ds), args.n, replace=False)

    print(f"{'파일':<12}{'프레임간 평균변화':>18}{'첫↔끝 변화':>14}{'액션 |Δ| 평균':>16}")
    for k, i in enumerate(idx):
        item = ds[int(i)]
        v = item["video"]                                   # [c, t, h, w] in [-1,1]
        act = item["act"]                                   # [t, 12] (절대6 + Δ6)
        name = f"gt_{k:02d}"
        save_video_tensor(v, out / f"{name}.mp4", 6)
        save_grid(v, out / f"{name}.png")

        px = ((v + 1) * 127.5)
        inter = (px[:, 1:] - px[:, :-1]).abs().mean().item()
        endpt = (px[:, -1] - px[:, 0]).abs().mean().item()
        adel = act[:, 6:].abs().mean().item()               # 정규화된 Δ 액션 크기
        print(f"{name:<12}{inter:>18.2f}{endpt:>14.2f}{adel:>16.3f}")

    print(f"\n저장 위치: {out}")


if __name__ == "__main__":
    main()
