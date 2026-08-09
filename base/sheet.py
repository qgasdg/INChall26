"""체크포인트별 생성물을 한 장으로 묶는다 — 스텝이 늘면서 무엇이 달라지는지 한눈에 보려고.

`peek.py` 가 남기는 그리드(16프레임을 8×2 로 편 것)를 다시 16프레임으로 잘라
**한 줄에 16프레임씩** 눕히고, 줄을 GT · 300 · 900 · 1500 순으로 쌓는다.
같은 프레임끼리 세로로 정렬되므로 스텝 간 차이가 같은 위치에서 비교된다.

사용:
  python sheet.py --dirs 300=<디렉터리> 900=<디렉터리> 1500=<디렉터리> --out <출력디렉터리>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CELL_W, CELL_H = 256, 160          # 512×320 프레임을 절반으로. 16장이면 4096px 폭
LABEL_W = 110


def frames(path: Path) -> np.ndarray:
    """8×2 그리드 PNG → [16, h, w, 3]"""
    a = np.asarray(Image.open(path).convert("RGB"))
    h, w = a.shape[0] // 2, a.shape[1] // 8
    return np.stack([a[r * h:(r + 1) * h, c * w:(c + 1) * w] for r in range(2) for c in range(8)])


def row(path: Path) -> np.ndarray:
    fs = [np.asarray(Image.fromarray(f).resize((CELL_W, CELL_H), Image.LANCZOS)) for f in frames(path)]
    return np.concatenate(fs, axis=1)


def sheet(rows: list[tuple[str, Path]], out: Path, title: str) -> None:
    imgs = [row(p) for _, p in rows]
    W = imgs[0].shape[1] + LABEL_W
    canvas = np.full((CELL_H * len(imgs) + 26, W, 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        canvas[26 + i * CELL_H: 26 + (i + 1) * CELL_H, LABEL_W:] = im
    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    d.text((6, 8), title, fill=(0, 0, 0))
    for i, (name, _) in enumerate(rows):
        d.text((8, 26 + i * CELL_H + CELL_H // 2 - 4), name, fill=(0, 0, 0))
    img.save(out)
    print(f"{out}  ({len(rows)}줄 × 16프레임)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="라벨=디렉터리 (예: 300=/path 900=/path)")
    ap.add_argument("--gt-dir", default=None, help="train GT 를 가져올 디렉터리 (없으면 첫 번째 dirs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.dirs]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.gt_dir) if args.gt_dir else pairs[0][1]

    # 어떤 클립이 있는지는 첫 번째 디렉터리에서 알아낸다
    first = pairs[0][1]
    evals = sorted({re.match(r"(eval_sample_\d+)_", f.name).group(1)
                    for f in first.glob("eval_sample_*_*.png") if "_input" not in f.name})
    trains = sorted({re.match(r"(train\d+)_", f.name).group(1)
                     for f in first.glob("train*_*.png") if not f.name.endswith("_gt.png")})

    for sid in evals:
        rows = []
        inp = first / f"{sid}_input.png"
        if inp.exists():
            rows.append(("입력(정지)", inp))
        for label, d in pairs:
            g = next(d.glob(f"{sid}_*.png"), None)
            g = next((x for x in d.glob(f"{sid}_*.png") if "_input" not in x.name), None)
            if g:
                rows.append((label, g))
        if len(rows) > 1:
            sheet(rows, out / f"{sid}.png", f"{sid}  (윗줄부터: " + " · ".join(n for n, _ in rows) + ")")

    for tid in trains:
        rows = []
        gt = gt_dir / f"{tid}_gt.png"
        if gt.exists():
            rows.append(("GT", gt))
        for label, d in pairs:
            g = next((x for x in d.glob(f"{tid}_*.png") if not x.name.endswith("_gt.png")), None)
            if g:
                rows.append((label, g))
        if len(rows) > 1:
            sheet(rows, out / f"{tid}.png", f"{tid}  (윗줄부터: " + " · ".join(n for n, _ in rows) + ")")


if __name__ == "__main__":
    main()
