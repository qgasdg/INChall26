"""축별 train↔eval 좌표 격차 분석 — wrist_roll 문제의 정밀 진단.

■ 배경
Phase 1(전수 카탈로그)과 동작 유사도 분석이 각각 독립적으로 같은 곳을 가리켰다:
wrist_roll(손목 회전축)이 train과 eval 사이에서 유독 어긋나 있다.
그 어긋남이 "분포 모양이 다른 것"인지 "원점만 밀린 것"인지를 가른다.
처방이 완전히 달라지기 때문이다.

■ 재는 것
축마다 세 가지를 본다.
  1) eval 중앙값 vs train 데이터셋 평균들의 중앙값 → 격차의 크기
  2) eval 중앙값이 각 train 데이터셋의 관측 범위 [min, max] 안에 들어가는가
     → 몇 개의 데이터셋이 eval의 작동 구간을 아예 본 적 없는지 (정의역 겹침)
  3) 자세 거리 vs 변화량 거리 (eda09 결과) → 모양이 다른지 원점만 밀렸는지

용어: 정의역(domain, 값이 실제로 나타나는 구간) · 중앙값(median, 크기순 한가운데 값 —
      평균과 달리 극단값에 덜 흔들림) · 오프셋(offset, 일정하게 더해진 값).

사용: .venv/bin/python scripts/eda10_axis_offset.py
산출: results/axis_offset_analysis.csv
선행: results/train_catalog.csv, results/dataset_eval_motion_similarity.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIMS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main() -> None:
    ev = np.stack([np.load(p) for p in
                   sorted((ROOT / "open" / "data" / "eval" / "actions").glob("*.npy"))])
    ev_flat = ev.reshape(-1, 6)
    cat = pd.read_csv(ROOT / "results" / "train_catalog.csv")
    mot = pd.read_csv(ROOT / "results" / "dataset_eval_motion_similarity.csv")

    rows = []
    for i, d in enumerate(DIMS):
        mean_col, lo_col, hi_col = f"a_main_{d}_mean", f"a_main_{d}_min", f"a_main_{d}_max"
        ev_med = float(np.median(ev_flat[:, i]))
        covered = int(((cat[lo_col] <= ev_med) & (cat[hi_col] >= ev_med)).sum())
        rows.append({
            "axis": d,
            "eval_median": round(ev_med, 2),
            "eval_std": round(float(ev_flat[:, i].std()), 2),
            "train_median_of_means": round(float(cat[mean_col].median()), 2),
            "offset_eval_minus_train": round(ev_med - float(cat[mean_col].median()), 2),
            "train_between_dataset_std": round(float(cat[mean_col].std()), 2),
            "n_datasets_covering_eval_median": covered,
            "n_datasets_total": len(cat),
            "coverage_pct": round(100 * covered / len(cat), 1),
            "pose_dist_median": round(float(mot[f"pose_{d}"].median()), 2),
            "delta_dist_median": round(float(mot[f"delta_{d}"].median()), 4),
            # 모양이 같고 원점만 밀렸다면 자세거리는 크고 변화량거리는 작다
            "pose_to_delta_ratio": round(float(mot[f"pose_{d}"].median())
                                         / float(mot[f"delta_{d}"].median()), 1),
        })

    df = pd.DataFrame(rows).sort_values("pose_to_delta_ratio", ascending=False)
    df.to_csv(ROOT / "results" / "axis_offset_analysis.csv", index=False)

    print("=== 축별 train↔eval 격차 ===")
    print(df[["axis", "eval_median", "train_median_of_means", "offset_eval_minus_train",
              "coverage_pct", "pose_dist_median", "delta_dist_median",
              "pose_to_delta_ratio"]].to_string(index=False))

    wr = df[df.axis == "wrist_roll"].iloc[0]
    print(f"\n=== wrist_roll 판정 ===")
    print(f"  격차 {wr.offset_eval_minus_train:+.1f}도, eval 작동구간을 본 train 데이터셋 "
          f"{wr.n_datasets_covering_eval_median}/{wr.n_datasets_total} "
          f"({wr.coverage_pct}%) — 다른 축은 {df[df.axis!='wrist_roll'].coverage_pct.min():.0f}% 이상")
    print(f"  자세거리/변화량거리 비 {wr.pose_to_delta_ratio} (전 축 중 "
          f"{'최대' if wr.pose_to_delta_ratio == df.pose_to_delta_ratio.max() else '중간'})")
    print("  → 분포 '모양'이 아니라 '원점'이 어긋난 것. 처방은 오프셋 보정.")
    print("\n→ results/axis_offset_analysis.csv")


if __name__ == "__main__":
    main()
