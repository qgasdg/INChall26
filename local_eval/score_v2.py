"""홀드아웃 v2 채점기: score.py의 v2 대응판 (v1 score.py는 앵커 이력 보존 위해 동결).

v1 대비 변경:
  - --holdout 인자 (기본 holdout_v2.json), samples[] + tier(indomain/unseen_cousin/unseen_general) 구조.
  - start 윈도우 지원 (GT = start부터 16프레임).
  - 영상 파일명 = <sample_id>.mp4 (challenge 변환 포맷과 동일 키).
  - --static: 영상 없이 GT 첫 프레임 16장 반복(정적 베이스라인)을 즉석 채점.

사용:
  python local_eval/score_v2.py --videos <dir> [--tiers unseen] [--csv out.csv]
  python local_eval/score_v2.py --static --tiers unseen --csv static.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
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
import kit_bridge

kit = kit_bridge.kit

DEFAULT_HOLDOUT = Path(__file__).resolve().parent / "holdout_v2.json"
TIER_SETS = {
    "unseen": ["unseen_cousin", "unseen_general"],
    "indomain": ["indomain"],
    "all": ["indomain", "unseen_cousin", "unseen_general"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None, help="생성 영상 디렉토리 (<sample_id>.mp4)")
    ap.add_argument("--static", action="store_true", help="GT 첫 프레임 반복(정적)을 채점")
    ap.add_argument("--holdout", default=str(DEFAULT_HOLDOUT))
    ap.add_argument("--tiers", default="unseen", choices=list(TIER_SETS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    if not args.static and not args.videos:
        raise SystemExit("--videos 또는 --static 필요")

    device = get_device(args.device)
    scorer = KitScorer(device)
    holdout = json.loads(Path(args.holdout).read_text())
    tiers = TIER_SETS[args.tiers]
    samples = [s for s in holdout["samples"] if s["tier"] in tiers]
    video_root = Path(args.videos) if args.videos else None

    rows = []
    t0 = time.time()
    for s in samples:
        sid = s["sample_id"]
        gt_frames = read_frames(s["dataset"], s["episode_index"], s["start"], TEMPORAL_LENGTH)
        if args.static:
            gen_np = np.repeat(gt_frames[:1], TEMPORAL_LENGTH, axis=0)
        else:
            path = video_root / f"{sid}.mp4"
            if not path.exists():
                continue
            gen_np = kit.read_video_uint8(path, expected_frames=TEMPORAL_LENGTH).numpy()

        gen = scorer.to_eval_video(gen_np).unsqueeze(0)
        gt = scorer.to_eval_video(gt_frames).unsqueeze(0)
        target = scorer.normalize_actions(
            read_actions(s["dataset"], s["episode_index"], s["start"], TEMPORAL_LENGTH)
        )

        dino_pf, dino_fl = dino_distance_both(
            scorer.dino_features(gen)[0], scorer.dino_features(gt)[0]
        )
        video_d = cosine_distance(
            scorer.video_features(gen)[0], scorer.video_features(gt)[0]
        )
        act = action_mae(scorer.action_pred(gen)[0], target)

        rows.append({
            "tier": s["tier"], "sample": sid,
            "dino_pf": dino_pf, "dino_flat": dino_fl,
            "video": video_d, "action": act,
            "total_pf": total_score(dino_pf, video_d, act),
            "total_flat": total_score(dino_fl, video_d, act),
        })
        if len(rows) % 20 == 0:
            print(f"  {len(rows)}/{len(samples)}  ({time.time() - t0:.0f}s)", flush=True)

    if not rows:
        raise SystemExit("채점된 샘플 0개 (경로/파일명 규약 확인)")

    print(f"\n=== 요약 (n={len(rows)}, {time.time() - t0:.0f}s) ===")
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
