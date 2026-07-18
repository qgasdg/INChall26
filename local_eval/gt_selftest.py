"""GT self-test: 정답 영상을 채점기에 넣어 '도달 가능한 점수 바닥'을 실측한다.

측정 항목 (홀드아웃 표본 K개 에피소드, 윈도우 [0..15]):
1. Action floor  = MAE(extractor(GT영상), GT행동)  ← 완벽한 영상을 내도 남는 점수
   (DINO/Video는 GT vs GT 비교 = 정확히 0이므로 계산 생략)
2. Static 앵커   = 첫 프레임을 16번 반복한 영상의 DINO/Video/Action
   ← "아무것도 안 움직이는 영상"의 점수. 리더보드 해석용 기준점.

사용: .venv/bin/python local_eval/gt_selftest.py [--n 12] [--tier in_domain|unseen_scene] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from episode_io import read_actions, read_frames
from kit_bridge import (
    KitScorer,
    TEMPORAL_LENGTH,
    action_mae,
    cosine_distance,
    dino_distance_both,
    get_device,
    total_score,
)

HOLDOUT = Path(__file__).resolve().parent / "holdout.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--tier", default="in_domain", choices=["in_domain", "unseen_scene"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device}")
    scorer = KitScorer(device)

    samples = json.loads(HOLDOUT.read_text())[args.tier]
    rng = random.Random(0)
    picks = rng.sample(samples, min(args.n, len(samples)))

    rows = []
    for s in picks:
        frames = read_frames(s["dataset"], s["episode_index"], 0, TEMPORAL_LENGTH)
        actions = read_actions(s["dataset"], s["episode_index"], 0, TEMPORAL_LENGTH)
        target = scorer.normalize_actions(actions)

        gt = scorer.to_eval_video(frames).unsqueeze(0)          # (1,16,320,512,3) uint8
        static = scorer.to_eval_video(
            np.repeat(frames[:1], TEMPORAL_LENGTH, axis=0)
        ).unsqueeze(0)

        # 1) Action floor (GT 영상)
        floor = action_mae(scorer.action_pred(gt)[0], target)

        # 2) static 앵커
        gt_dino = scorer.dino_features(gt)[0]
        st_dino = scorer.dino_features(static)[0]
        dino_pf, dino_fl = dino_distance_both(st_dino, gt_dino)
        video_d = cosine_distance(scorer.video_features(static)[0], scorer.video_features(gt)[0])
        st_action = action_mae(scorer.action_pred(static)[0], target)

        rows.append({
            "dataset": s["dataset"], "ep": s["episode_index"],
            "action_floor": floor,
            "static_dino_pf": dino_pf, "static_dino_flat": dino_fl,
            "static_video": video_d, "static_action": st_action,
        })
        print(f"  {s['dataset'][:40]:42s} ep{s['episode_index']:4d}  floor {floor:.4f} | "
              f"static: dino {dino_pf:.4f}/{dino_fl:.4f} video {video_d:.4f} act {st_action:.4f}")

    def mean(k): return float(np.mean([r[k] for r in rows]))

    print(f"\n=== 요약 ({args.tier}, n={len(rows)}) ===")
    print(f"Action floor (GT 영상의 MAE):       {mean('action_floor'):.4f}"
          f"  -> 총점 기여 하한 {0.4 * mean('action_floor'):.4f}")
    print(f"Static 앵커: DINO {mean('static_dino_pf'):.4f}(프레임평균)/{mean('static_dino_flat'):.4f}(flatten)"
          f"  Video {mean('static_video'):.4f}  Action {mean('static_action'):.4f}")
    print(f"Static 총점(프레임평균 기준):        "
          f"{total_score(mean('static_dino_pf'), mean('static_video'), mean('static_action')):.4f}")
    print(f"이론 최저 총점 = 0.4 x floor =       {0.4 * mean('action_floor'):.4f}")


if __name__ == "__main__":
    main()
