"""C안용 — 프레임 간 관절 변화량(Δ)의 통계를 train 데이터에서 낸다.

절대 관절값은 so100_action_statistics.json 으로 정규화되지만 Δ 는 스케일이 전혀 달라서
(전역 std 의 8% 수준) 같은 통계로 나누면 값이 0 근처에 눌린다. Δ 전용 std 가 필요하다.

출력: <root>/so100_delta_statistics.json  {"count":…, "mean":[6], "std":[6]}
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/ft/data/train"))
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

random.seed(0)
files = sorted(ROOT.rglob("*.parquet"))
print(f"parquet {len(files)}개 중 {min(LIMIT, len(files))}개 표본")
if len(files) > LIMIT:
    files = random.sample(files, LIMIT)

n = 0
total = np.zeros(6, dtype=np.float64)
total_sq = np.zeros(6, dtype=np.float64)
for f in files:
    try:
        a = np.stack(pd.read_parquet(f, columns=["action"])["action"].to_numpy()).astype(np.float64)
    except Exception:
        continue
    if a.shape[0] < 2 or a.shape[-1] != 6:
        continue
    d = np.diff(a, axis=0)
    n += d.shape[0]
    total += d.sum(axis=0)
    total_sq += np.square(d).sum(axis=0)

mean = total / n
std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 1e-12))
out = {"count": int(n), "mean": mean.tolist(), "std": std.tolist()}
(ROOT / "so100_delta_statistics.json").write_text(json.dumps(out, indent=2))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
print(f"\n프레임 전이 {n:,}개")
print(f"  {'관절':<14} {'Δ mean':>10} {'Δ std':>10}")
for j, name in enumerate(JOINTS):
    print(f"  {name:<14} {mean[j]:>10.4f} {std[j]:>10.4f}")
print(f"\n저장: {ROOT / 'so100_delta_statistics.json'}")
