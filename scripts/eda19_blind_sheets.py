"""육안 검증용 '무라벨' 대조표 — 자동 판정을 가린 채 사람이 먼저 라벨을 매기기 위한 도구.

왜 무라벨인가: 자동 판정을 보면서 눈으로 확인하면 **확증 편향**(맞다고 먼저 믿고
그렇게 보이는 것)이 생긴다. 자동 분류의 정확도를 재려면 사람 라벨이 자동 라벨과
독립이어야 한다. 그래서 이 대조표에는 데이터셋 번호만 적고 판정값은 넣지 않는다.

절차: 이 대조표로 사람이 128개에 라벨을 매겨 파일로 남긴 뒤(`--emit-template`으로
      빈 서식 생성), eda20에서 자동 판정과 대조해 정확도를 계산한다.

용어: 확증 편향(confirmation bias, 기대한 결과를 뒷받침하는 쪽으로 보게 되는 경향) ·
      무라벨/블라인드(판정 결과를 가리고 평가하는 방식).

사용: uv run python scripts/eda19_blind_sheets.py --out local_runs/blind
산출: local_runs/blind/blind_NN.png, local_runs/blind/index.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "local_eval"))

TW, TH, BAR = 300, 225, 18


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="local_runs/blind")
    ap.add_argument("--per-page", type=int, default=16)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    from eda15_viewpoint import train_datasets, train_frames  # noqa: PLC0415

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    dss = train_datasets()          # 이름순 = 자동 판정값과 무관한 순서
    rows = []

    page, tiles = 0, []
    for i, ds in enumerate(dss):
        fr = train_frames(ds, 3)
        im = (Image.fromarray(fr[len(fr) // 2][1]).resize((TW, TH), Image.BILINEAR)
              if fr else Image.new("RGB", (TW, TH), (40, 40, 40)))
        tile = Image.new("RGB", (TW, TH + BAR), (255, 255, 255))
        tile.paste(im, (0, BAR))
        ImageDraw.Draw(tile).text((3, 4), f"#{i:03d}", fill=(0, 0, 0))
        tiles.append(tile)
        rows.append({"idx": i, "dataset": ds})
        if len(tiles) == args.per_page or i == len(dss) - 1:
            h = (len(tiles) + args.cols - 1) // args.cols
            canvas = Image.new("RGB", (TW * args.cols, (TH + BAR) * h), (255, 255, 255))
            for k, t in enumerate(tiles):
                canvas.paste(t, (TW * (k % args.cols), (TH + BAR) * (k // args.cols)))
            canvas.save(outdir / f"blind_{page:02d}.png")
            print(f"  blind_{page:02d}.png ({len(tiles)}칸)", flush=True)
            page, tiles = page + 1, []

    pd.DataFrame(rows).to_csv(outdir / "index.csv", index=False)
    print(f"→ {outdir}/index.csv ({len(rows)}개)")


if __name__ == "__main__":
    main()
