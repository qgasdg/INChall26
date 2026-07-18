"""로컬 리더보드 채점기: 생성 영상 디렉토리를 홀드아웃 GT와 대조해 대회 산식으로 채점.

입력 규약:
  생성 영상 파일명 = "<dataset의 '/'를 '__'로 치환>__ep<episode_index>.mp4", 16프레임.
  예: local_runs/exp01/aaa__bbb__ep3.mp4  (dataset "aaa/bbb", episode 3, 윈도우 [0..15])

출력: 샘플별 성분 점수 + 요약 (DINO는 프레임평균/flatten 병산).

사용:
  .venv/bin/python local_eval/score.py --videos local_runs/exp01 --tier in_domain [--device cpu] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

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

# submission_kit과 동일한 로더로 생성 영상을 읽는다 (프레임 수 16 강제 포함)
import kit_bridge  # noqa: E402
kit = kit_bridge.kit


def sample_key(dataset: str, episode_index: int) -> str:
    return f"{dataset.replace('/', '__')}__ep{episode_index}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="생성 영상 디렉토리")
    ap.add_argument("--tier", default="in_domain", choices=["in_domain", "unseen_scene", "both"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--csv", default=None, help="샘플별 결과 CSV 저장 경로")
    args = ap.parse_args()

    device = get_device(args.device)
    scorer = KitScorer(device)
    holdout = json.loads(HOLDOUT.read_text())
    tiers = ["in_domain", "unseen_scene"] if args.tier == "both" else [args.tier]

    video_root = Path(args.videos)
    rows = []
    for tier in tiers:
        for s in holdout[tier]:
            key = sample_key(s["dataset"], s["episode_index"])
            path = video_root / f"{key}.mp4"
            if not path.exists():
                continue

            gen_np = kit.read_video_uint8(path, expected_frames=TEMPORAL_LENGTH).numpy()
            gen = scorer.to_eval_video(gen_np).unsqueeze(0)

            gt_frames = read_frames(s["dataset"], s["episode_index"], 0, TEMPORAL_LENGTH)
            gt = scorer.to_eval_video(gt_frames).unsqueeze(0)
            target = scorer.normalize_actions(
                read_actions(s["dataset"], s["episode_index"], 0, TEMPORAL_LENGTH)
            )

            dino_pf, dino_fl = dino_distance_both(
                scorer.dino_features(gen)[0], scorer.dino_features(gt)[0]
            )
            video_d = cosine_distance(
                scorer.video_features(gen)[0], scorer.video_features(gt)[0]
            )
            act = action_mae(scorer.action_pred(gen)[0], target)

            rows.append({
                "tier": tier, "sample": key,
                "dino_pf": dino_pf, "dino_flat": dino_fl,
                "video": video_d, "action": act,
                "total_pf": total_score(dino_pf, video_d, act),
                "total_flat": total_score(dino_fl, video_d, act),
            })
            print(f"  [{tier}] {key[:52]:54s} dino {dino_pf:.4f} video {video_d:.4f} "
                  f"act {act:.4f} -> {rows[-1]['total_pf']:.4f}")

    if not rows:
        raise SystemExit(f"{video_root}에서 홀드아웃과 매칭되는 mp4를 찾지 못함 (파일명 규약 확인)")

    print(f"\n=== 요약 (n={len(rows)}) ===")
    for tier in tiers:
        tr = [r for r in rows if r["tier"] == tier]
        if not tr:
            continue
        m = {k: float(np.mean([r[k] for r in tr])) for k in
             ("dino_pf", "dino_flat", "video", "action", "total_pf", "total_flat")}
        print(f"[{tier}] n={len(tr)}  DINO {m['dino_pf']:.4f}(pf)/{m['dino_flat']:.4f}(flat)  "
              f"Video {m['video']:.4f}  Action {m['action']:.4f}")
        print(f"         총점: {m['total_pf']:.4f} (프레임평균) / {m['total_flat']:.4f} (flatten)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV 저장: {args.csv}")


if __name__ == "__main__":
    main()
