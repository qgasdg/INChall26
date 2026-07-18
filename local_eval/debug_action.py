"""Action floor 1.49 이상치 디버깅.

가설:
H1. 정렬 문제 (video[t] vs action[t±k])
H2. extractor가 action이 아니라 state를 예측
H3. 정규화 불일치 (z-score vs deg)
H4. extractor가 상수(평균) 예측에 가까움 → 도메인 갭/약한 모델
H5. mp4 인코딩 유무 차이 (내 브리지는 무손실 경로)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from episode_io import episode_paths, read_frames
from kit_bridge import KitScorer, TEMPORAL_LENGTH, get_device

SAMPLES = [
    ("pranavsaroha/so100_carrot_5", 9),
    ("vladfatu/so100_above", 5),
    ("sihyun77/sihyun_3_17_2", 16),
]

scorer = KitScorer(get_device("cpu"))
np.set_printoptions(precision=2, suppress=True, linewidth=140)

for ds, ep in SAMPLES:
    _, pq = episode_paths(ds, ep)
    df = pd.read_parquet(pq, columns=["action", "observation.state"])
    A = np.stack(df["action"].to_numpy()).astype(np.float32)
    S = np.stack(df["observation.state"].to_numpy()).astype(np.float32)

    frames = read_frames(ds, ep, 0, TEMPORAL_LENGTH)
    video = scorer.to_eval_video(frames).unsqueeze(0)
    pred = scorer.action_pred(video)[0].numpy()  # (16,6) z-공간 추정

    z = lambda x: (torch.from_numpy(x) - scorer.action_mean).div(scorer.action_std).numpy()

    print(f"\n=== {ds} ep{ep} ===")
    tgt = z(A[:16])
    print("target z[0]:", tgt[0], " pred[0]:", pred[0])
    print("target z[8]:", tgt[8], " pred[8]:", pred[8])
    print("pred 시간표준편차(차원별):", pred.std(axis=0), " target:", tgt.std(axis=0))

    # H1: 정렬 시프트
    for k in range(-2, 3):
        if 0 + k < 0 or 16 + k > len(A):
            continue
        mae = np.abs(pred - z(A[k : k + 16])).mean()
        print(f"  H1 action shift {k:+d}: MAE {mae:.4f}")
    # H2: state 비교
    mae_s = np.abs(pred - z(S[:16])).mean()
    print(f"  H2 vs state:        MAE {mae_s:.4f}")
    # H3: deg 그대로 비교
    mae_deg = np.abs(pred - A[:16]).mean()
    print(f"  H3 vs deg(비정규):  MAE {mae_deg:.4f}")
    # H4: 상수 예측 대비 — pred가 데이터셋 평균/전역0과 얼마나 다른가
    mae_zero = np.abs(tgt).mean()  # pred=0(전역 평균) 가정 시 MAE
    print(f"  H4 pred=0 가정 MAE {mae_zero:.4f}  (실제 pred MAE {np.abs(pred - tgt).mean():.4f})")
    # 차원별 분해
    print("  차원별 |pred-target| :", np.abs(pred - tgt).mean(axis=0))
