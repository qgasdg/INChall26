"""시점 분류 육안 스팟체크 — 자동 분류 결과를 사람이 확인할 수 있는 대조표로 만든다.

자동 추정은 두 장면(eval 셋업 A·B)으로만 보정됐다. 즉 "216장으로 검증"처럼 보여도
**실제로 본 장면은 2종뿐**이라 일반화 근거가 약하다. 그래서 train 128개에 대해서는
육안 확인이 검증의 본체다. 이 스크립트는 그 확인을 체계적으로 하기 위한 도구다.

만드는 대조표 3종:
  1. spectrum  — depth_break 값 순서대로 전 데이터셋 1장씩. 경계가 어디서 넘어가는지
                 눈으로 훑어 임계값의 타당성을 본다.
  2. flagged   — 자동 분류가 불확실하다고 스스로 표시한 데이터셋(보조신호 불일치·
                 경계 근접·mixed). 오분류가 있다면 여기 몰려 있어야 한다.
  3. by_label  — 라벨별 무작위 표본. 라벨 안이 실제로 균질한지 본다.

각 칸 위에 데이터셋 이름·라벨·depth_break를 적어 넣어 판정과 근거를 함께 본다.

사용: uv run python scripts/eda17_viewpoint_eyeball.py --out local_runs/eyeball
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "local_eval"))

RESULTS = ROOT / "results"
TW, TH = 256, 192          # 칸 크기
BAR = 26                   # 이름표 높이


def thumb(ds: str, label: str, val: float, seed: int = 0) -> Image.Image:
    """데이터셋 대표 프레임 1장 + 이름표."""
    from eda15_viewpoint import to_std, train_frames  # noqa: PLC0415

    frames = train_frames(ds, 3)
    if not frames:
        im = Image.new("RGB", (TW, TH), (40, 40, 40))
    else:
        rgb = frames[len(frames) // 2][1]
        im = Image.fromarray(rgb).resize((TW, TH), Image.BILINEAR)
    tile = Image.new("RGB", (TW, TH + BAR), (255, 255, 255))
    tile.paste(im, (0, BAR))
    d = ImageDraw.Draw(tile)
    color = {"low_angle": (170, 0, 0), "top_down": (0, 90, 0)}.get(label, (150, 90, 0))
    name = ds if len(ds) <= 34 else ds[:31] + "..."
    d.text((3, 2), name, fill=(0, 0, 0))
    d.text((3, 13), f"{label}  break={val:.3f}", fill=color)
    return tile


def sheet(rows: pd.DataFrame, out: Path, cols: int = 6) -> None:
    tiles = [thumb(r.dataset, r.viewpoint, r.depth_break_med) for r in rows.itertuples()]
    if not tiles:
        return
    n = len(tiles)
    h = (n + cols - 1) // cols
    canvas = Image.new("RGB", (TW * cols, (TH + BAR) * h), (255, 255, 255))
    for k, t in enumerate(tiles):
        canvas.paste(t, (TW * (k % cols), (TH + BAR) * (k // cols)))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"  {out}  ({n}칸)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="local_runs/eyeball")
    ap.add_argument("--per-page", type=int, default=24)
    args = ap.parse_args()
    outdir = ROOT / args.out

    ds = pd.read_csv(RESULTS / "train_viewpoint.csv")

    print("1) spectrum — depth_break 내림차순 전수")
    s = ds.sort_values("depth_break_med", ascending=False).reset_index(drop=True)
    for p in range(0, len(s), args.per_page):
        sheet(s.iloc[p : p + args.per_page], outdir / f"spectrum_{p // args.per_page:02d}.png")

    print("2) flagged — 자동 분류가 불확실하다고 표시한 것")
    f = ds[ds.needs_eyeball].sort_values("depth_break_med", ascending=False)
    for p in range(0, len(f), args.per_page):
        sheet(f.iloc[p : p + args.per_page], outdir / f"flagged_{p // args.per_page:02d}.png")

    print("3) by_label — 라벨별 무작위 표본")
    rng = np.random.default_rng(0)
    for v, sub in ds.groupby("viewpoint"):
        take = sub.sample(min(len(sub), args.per_page), random_state=int(rng.integers(1 << 30)))
        sheet(take.sort_values("depth_break_med", ascending=False), outdir / f"label_{v}.png")


if __name__ == "__main__":
    main()
