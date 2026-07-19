"""train × eval 동작(action) 유사도 랭킹 — Phase 2 한계 보완.

■ 왜 하는가
Phase 2(scripts/eda05_eval_similarity.py)는 DINO로 **장면의 겉모습**이 얼마나 닮았는지만
쟀다. 문서에도 한계로 적어뒀다 — "로봇의 동작이 얼마나 닮았는지는 재지 못한다".
그런데 대회 점수의 40%가 Action 성분(로봇이 어떻게 움직였는지 맞히기)이므로,
큐레이션(학습 데이터 선별)을 겉모습만 보고 정하면 위험하다. 동작 축의 랭킹을 따로 만든다.

■ 무엇을 비교하는가
eval은 정답 영상이 없지만 **행동 시퀀스(actions/*.npy)는 주어진다**. 즉 eval이 요구하는
동작의 분포를 우리는 정확히 안다. 이걸 train 각 데이터셋의 동작 분포와 비교한다.

두 가지를 본다 (둘 다 16프레임 창 기준 = 대회 규격):
  1) 자세(pose) 분포 — 관절 6축의 절대 각도가 어느 값 근처에 머무는가
  2) 변화량(delta) 분포 — 프레임 사이에 각 축이 얼마나 움직이는가
     (자세는 조립·영점조정 차이에 크게 흔들리지만(Phase 1 §4), 변화량은 상대적으로
      그 영향이 적어 "동작의 성격"을 더 잘 드러낸다)

거리 척도는 축별 1차원 분포 간 **와서스테인 거리**(Wasserstein distance, 두 분포를 겹치게
만들 때 옮겨야 하는 '흙의 양' — 값이 작을수록 닮음)를 쓰고, 6축 평균으로 데이터셋 1점을 낸다.
단위가 도(degree)라 해석이 직관적이다.

용어: 행동 시퀀스(로봇 관절 6개의 목표 각도를 시간순으로 적은 표) · 분위수(quantile,
      분포를 크기순으로 줄 세웠을 때의 위치별 값) · 와서스테인 거리(분포 간 거리).

사용: .venv/bin/python scripts/eda09_motion_similarity.py [--workers 32]
산출: results/dataset_eval_motion_similarity.csv
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "open" / "data" / "train"
EVAL = ROOT / "open" / "data" / "eval"
WINDOW = 16
DIMS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# 분포를 통째로 들고 다니면 무거우므로 분위수 격자로 요약한다 (와서스테인 거리는
# 두 분포의 분위수 함수 차이의 적분과 같으므로, 균등 분위수 격자면 정확도 손실이 거의 없다)
QS = np.linspace(0.005, 0.995, 200)


def wasserstein_from_quantiles(qa: np.ndarray, qb: np.ndarray) -> float:
    """균등 분위수 격자에서의 1차원 와서스테인-1 거리 근사."""
    return float(np.mean(np.abs(qa - qb)))


def dataset_quantiles(rel: str) -> dict | None:
    """데이터셋의 (자세, 변화량) 분위수 요약. 16프레임 미만 에피소드는 제외."""
    d = TRAIN / rel
    poses, deltas = [], []
    for p in sorted((d / "data" / "chunk-000").glob("*.parquet")):
        try:
            a = np.stack(pd.read_parquet(p, columns=["action"])["action"].to_numpy())
        except Exception:
            continue
        if a.shape[0] < WINDOW:
            continue
        poses.append(a)
        deltas.append(np.diff(a, axis=0))
    if not poses:
        return None
    pose = np.concatenate(poses).astype(np.float64)
    delta = np.concatenate(deltas).astype(np.float64)
    return {
        "dataset": rel,
        "n_rows": int(pose.shape[0]),
        "pose_q": np.quantile(pose, QS, axis=0),    # (200, 6)
        "delta_q": np.quantile(delta, QS, axis=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    # --- eval 기준 분포 ---
    ev = np.stack([np.load(p) for p in sorted((EVAL / "actions").glob("*.npy"))])  # (216,16,6)
    print(f"eval 행동: {ev.shape}", flush=True)
    ev_pose = ev.reshape(-1, 6).astype(np.float64)
    ev_delta = np.diff(ev, axis=1).reshape(-1, 6).astype(np.float64)
    ev_pose_q = np.quantile(ev_pose, QS, axis=0)
    ev_delta_q = np.quantile(ev_delta, QS, axis=0)

    datasets = sorted("/".join(p.parts[-4:-2]) for p in TRAIN.glob("*/*/meta/info.json"))
    print(f"데이터셋 {len(datasets)}개, 워커 {args.workers}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(dataset_quantiles, datasets), 1):
            if r is None:
                continue
            pose_d = [wasserstein_from_quantiles(r["pose_q"][:, k], ev_pose_q[:, k])
                      for k in range(6)]
            delta_d = [wasserstein_from_quantiles(r["delta_q"][:, k], ev_delta_q[:, k])
                       for k in range(6)]
            row = {"dataset": r["dataset"], "n_rows": r["n_rows"],
                   "pose_dist_mean": round(float(np.mean(pose_d)), 3),
                   "delta_dist_mean": round(float(np.mean(delta_d)), 4)}
            row.update({f"pose_{n}": round(v, 2) for n, v in zip(DIMS, pose_d)})
            row.update({f"delta_{n}": round(v, 4) for n, v in zip(DIMS, delta_d)})
            rows.append(row)
            if i % 20 == 0 or i == len(datasets):
                print(f"  {i}/{len(datasets)}", flush=True)

    df = pd.DataFrame(rows).sort_values("delta_dist_mean").reset_index(drop=True)
    df.insert(0, "motion_rank", df.index + 1)
    df.to_csv(ROOT / "results" / "dataset_eval_motion_similarity.csv", index=False)

    print("\n=== 동작(변화량) 기준 eval과 가장 닮은 15개 — 값이 작을수록 닮음 ===")
    print(df.head(15)[["motion_rank", "dataset", "delta_dist_mean", "pose_dist_mean"]]
          .to_string(index=False))
    print("\n=== 가장 안 닮은 10개 ===")
    print(df.tail(10)[["motion_rank", "dataset", "delta_dist_mean", "pose_dist_mean"]]
          .to_string(index=False))
    print("\n=== 축별 거리 (전 데이터셋 중앙값) ===")
    for n in DIMS:
        print(f"  {n:16s} 자세 {df[f'pose_{n}'].median():8.2f}도   변화량 {df[f'delta_{n}'].median():7.4f}도")

    # 겉모습 랭킹과의 관계 — 둘이 같은 것을 재는지 확인
    sim_path = ROOT / "results" / "dataset_eval_similarity.csv"
    if sim_path.exists():
        vis = pd.read_csv(sim_path)[["dataset", "rank", "sim_p95"]]
        m = df.merge(vis, on="dataset")
        # 스피어만 순위상관 = 순위로 바꾼 뒤의 피어슨 상관. scipy 의존을 피해 직접 계산한다
        # (scipy는 uv.lock에 없고, 이 한 줄 때문에 의존성을 늘릴 이유가 없다)
        def spearman(a: pd.Series, b: pd.Series) -> float:
            return float(a.rank().corr(b.rank()))

        rho_d = spearman(m["delta_dist_mean"], m["sim_p95"])
        rho_p = spearman(m["pose_dist_mean"], m["sim_p95"])
        print(f"\n겉모습(DINO p95) vs 동작 거리 순위상관: 변화량 {rho_d:+.3f} / 자세 {rho_p:+.3f}")
        print("  (0에 가까우면 두 랭킹이 서로 다른 것을 재고 있다는 뜻 = 둘 다 봐야 함)")
        both = m[(m["rank"] <= 40) & (m["motion_rank"] <= 40)]
        print(f"  겉모습·동작 모두 상위 40위 안: {len(both)}개 → {sorted(both.dataset)[:10]}")
    print("\n→ results/dataset_eval_motion_similarity.csv")


if __name__ == "__main__":
    main()
