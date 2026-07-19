"""holdout_v2 unseen 샘플을 challenge 입력 포맷으로 변환.

challenge 포맷 = images/<sample_id>.png (윈도우 첫 프레임, 원본 해상도 RGB)
             + actions/<sample_id>.npy (start부터 16프레임의 raw deg 행동, float32 (16,6)).
→ generate_baseline_videos.py를 무수정 재사용하기 위함 (docs/16 P0).

사용: .venv/bin/python local_eval/make_challenge_inputs_v2.py --out <dir> [--tier unseen]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from episode_io import read_actions, read_frames

HOLDOUT_V2 = Path(__file__).resolve().parent / "holdout_v2.json"
WINDOW = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", default="unseen", choices=["unseen", "all"],
                    help="unseen = unseen_cousin + unseen_general (기본)")
    args = ap.parse_args()

    holdout = json.loads(HOLDOUT_V2.read_text())
    samples = [s for s in holdout["samples"]
               if args.tier == "all" or s["tier"] != "indomain"]

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "actions").mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(samples):
        sid = s["sample_id"]
        frame0 = read_frames(s["dataset"], s["episode_index"], s["start"], 1)[0]
        actions = read_actions(s["dataset"], s["episode_index"], s["start"], WINDOW)
        assert actions.shape == (WINDOW, 6), (sid, actions.shape)
        Image.fromarray(frame0).save(out / "images" / f"{sid}.png")
        np.save(out / "actions" / f"{sid}.npy", actions)
        if (i + 1) % 40 == 0 or i + 1 == len(samples):
            print(f"{i + 1}/{len(samples)}", flush=True)

    print(f"완료: {len(samples)}개 -> {out}")


if __name__ == "__main__":
    main()
