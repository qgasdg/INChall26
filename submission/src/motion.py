"""생성 영상의 움직임 크기를 GT 기준과 견준다 — "정지 복사"인지 판별하는 수치.

**왜.** 눈으로 "정지에 가까워 보인다"는 판단은 흔들린다. 태양님이 exp22 에서 IRASim 을
기각할 때 쓴 기준이 "생성 모션 = GT 의 0.57%" 였다. 같은 잣대를 우리 것에도 대야 비교가 된다.

**주의.** eval 에는 GT 영상이 없어 같은 장면끼리 비교할 수 없다. 그래서 기준선을 두 개 쓴다.
  ① train GT 클립 N 개의 프레임간 변화량 — "이 정도는 움직여야 한다"는 절대 척도
  ② 우리 생성물 — 첫 프레임을 그대로 복사하면 정확히 0 이 된다
장면이 달라 배경 질감 차이가 섞이므로 **비율은 대략치**다. 0 에 가까운지 아닌지를 보는 용도.

프레임간 변화 외에 **누적 변위**(첫 프레임 대비)도 잰다. 프레임간만 보면 제자리 떨림도
움직임으로 세어지지만, 팔이 실제로 이동했다면 누적 변위가 시간에 따라 커져야 한다.
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
          Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def stats(v: np.ndarray) -> tuple[float, float]:
    """v: [T,H,W,C] 0~255 → (프레임간 평균변화, 마지막 프레임의 첫 프레임 대비 변위)"""
    f = v.astype(np.float32)
    inter = np.abs(f[1:] - f[:-1]).mean()
    disp = np.abs(f[-1] - f[0]).mean()
    return float(inter), float(disp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True, help="생성 mp4 디렉터리")
    ap.add_argument("--config", default=os.path.expanduser("~/base/base-01.yaml"))
    ap.add_argument("--train-n", type=int, default=20, help="기준선으로 쓸 train GT 클립 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.eval.feature_csv_utils import read_video_uint8  # noqa: E402

    # ── 우리 생성물 ────────────────────────────────────────────────────
    files = sorted(Path(args.pred_root).glob("*.mp4"))
    rows = []
    for f in files:
        v = np.asarray(read_video_uint8(f))
        rows.append(stats(v))
    a = np.array(rows)
    print(f"\n=== 생성물 {len(files)}개 ({args.pred_root}) ===")
    print(f"프레임간 평균변화 : {a[:,0].mean():7.3f}   (표준편차 {a[:,0].std():.3f})")
    print(f"마지막 프레임 변위: {a[:,1].mean():7.3f}   (표준편차 {a[:,1].std():.3f})")
    lo = a[a[:, 0].argsort()][:3]
    print(f"가장 정적인 3개   : {[round(x,3) for x in lo[:,0]]}")

    # ── train GT 기준선 ────────────────────────────────────────────────
    from lvdm.utils.utils import instantiate_from_config  # noqa: E402
    cfg = OmegaConf.load(args.config)
    data = instantiate_from_config(cfg.data); data.setup()
    ds = data.val_dataset
    idx = np.random.RandomState(args.seed).choice(len(ds), args.train_n, replace=False)
    g = []
    for i in idx:
        v = ds[int(i)]["video"]                                   # [c,T,h,w] in [-1,1]
        v = ((v.clamp(-1, 1) + 1) * 127.5).permute(1, 2, 3, 0).numpy()
        g.append(stats(v))
    b = np.array(g)
    print(f"\n=== train GT {args.train_n}개 (기준선) ===")
    print(f"프레임간 평균변화 : {b[:,0].mean():7.3f}   (표준편차 {b[:,0].std():.3f})")
    print(f"마지막 프레임 변위: {b[:,1].mean():7.3f}   (표준편차 {b[:,1].std():.3f})")

    print(f"\n=== 비율 (생성 / GT) ===")
    print(f"프레임간   : {100*a[:,0].mean()/b[:,0].mean():6.1f}%")
    print(f"누적 변위  : {100*a[:,1].mean()/b[:,1].mean():6.1f}%")
    print("\n※ eval 에는 GT 영상이 없어 장면이 다른 train GT 와 견준 대략치다.")
    print("   0%에 붙으면 정지 복사, 100% 부근이면 크기는 맞는 것(방향은 별개).")


if __name__ == "__main__":
    main()
