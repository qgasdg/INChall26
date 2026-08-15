"""데이터셋마다 `shoulder_pan` 의 화면 원점과 부호를 데이터에서 직접 잰다.

**왜 필요한가.** 좌우 반전 증강을 하려면 pan 을 "정면" 기준으로 부호 반전해야 하는데,
**관절 원점이 데이터셋마다 다르다**([docs/17](../docs/17_train_전수카탈로그.md): `wrist_roll` 87도 편차).
원점을 모르면 엉뚱한 각도로 반전되고, 그러면 **같은 액션에 반대 움직임을 가르치게 된다** —
우리가 고치려는 바로 그 능력을 오염시킨다. 7/28 에 자체 IDM 을 폐기시킨 원인도 같은 문제였다.

**재는 법.** 팔은 화면에서 움직이는 거의 유일한 것이므로, 에피소드 안에서 **시간 중앙값 프레임
대비 변화량**이 큰 곳이 팔이다. 그 변화량의 **가로 무게중심** x 와 그 프레임의 pan 각도를
모아 회귀하면 `x = a·pan + b` 가 나온다.

- **부호** = a 의 부호 (pan 이 커질 때 팔이 오른쪽으로 가는가)
- **원점** = x 가 화면 중앙일 때의 pan = (0.5 − b) / a  (x 는 0~1 로 정규화)
- **상관** = 낮으면 pan 이 화면에 안 드러나는 데이터셋 → 반전 대상에서 제외

eval 을 전혀 참조하지 않는다 — 학습 데이터 안에서만 닫힌 측정이다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np


def arm_centroid_x(frames: np.ndarray) -> np.ndarray:
    """프레임별 '움직이는 것'의 가로 무게중심(0~1). frames [T,H,W,3] uint8."""
    f = frames.astype(np.float32).mean(-1)              # 흑백 [T,H,W]
    ref = np.median(f, axis=0)                          # 시간 중앙값 = 배경
    d = np.abs(f - ref)                                 # [T,H,W]
    col = d.sum(1)                                      # 세로로 합 → [T,W]
    thr = np.percentile(col, 70, axis=1, keepdims=True)  # 상위 30%만 (잡음 억제)
    w = np.clip(col - thr, 0, None)
    x = np.arange(col.shape[1], dtype=np.float32)
    tot = w.sum(1)
    cx = np.where(tot > 0, (w * x).sum(1) / np.maximum(tot, 1e-6), np.nan)
    return cx / max(col.shape[1] - 1, 1)


def episode_pairs(vid: Path, pq: Path, max_frames: int) -> tuple[np.ndarray, np.ndarray] | None:
    import imageio.v3 as iio
    import pandas as pd
    try:
        fr = np.asarray(iio.imread(vid, plugin="pyav"))
        df = pd.read_parquet(pq)
    except Exception:
        return None
    st = df["observation.state"].values
    st = np.stack([np.asarray(s, dtype=np.float32) for s in st])
    n = min(len(fr), len(st))
    if n < 8:
        return None
    if n > max_frames:                                   # 균등 표본
        idx = np.linspace(0, n - 1, max_frames).astype(int)
    else:
        idx = np.arange(n)
    cx = arm_centroid_x(fr[idx])
    pan = st[idx, 0]
    ok = np.isfinite(cx)
    return (pan[ok], cx[ok]) if ok.sum() >= 8 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=6, help="데이터셋당 표본 에피소드")
    ap.add_argument("--max-frames", type=int, default=40, help="에피소드당 표본 프레임")
    args = ap.parse_args()

    root = Path(args.root)
    dss = sorted({p.parent for p in root.glob("*/*/data")})   # .../<계정>/<데이터셋>/data → 데이터셋 폴더
    print(f">>> 데이터셋 {len(dss)}개", flush=True)
    out = {}
    for i, ds in enumerate(dss, 1):
        name = f"{ds.parent.name}/{ds.name}"
        vids = sorted(ds.glob("videos/*/*/*.mp4"))
        pans, cxs, used = [], [], 0
        for v in vids[:: max(1, len(vids) // args.episodes)][: args.episodes]:
            pq = ds / "data" / v.parent.parent.name / (v.stem + ".parquet")
            if not pq.exists():
                cand = list(ds.glob(f"data/*/{v.stem}.parquet"))
                if not cand:
                    continue
                pq = cand[0]
            r = episode_pairs(v, pq, args.max_frames)
            if r is None:
                continue
            pans.append(r[0]); cxs.append(r[1]); used += 1
        if used == 0:
            out[name] = {"n_ep": 0, "note": "표본 없음"}
            print(f"[{i}/{len(dss)}] {name}: 표본 없음", flush=True)
            continue
        p = np.concatenate(pans); c = np.concatenate(cxs)
        if p.std() < 1e-3:
            out[name] = {"n_ep": used, "note": "pan 변화 없음", "pan_std": float(p.std())}
            print(f"[{i}/{len(dss)}] {name}: pan 변화 없음", flush=True)
            continue
        a, b = np.polyfit(p, c, 1)
        r = float(np.corrcoef(p, c)[0, 1])
        origin = float((0.5 - b) / a) if abs(a) > 1e-9 else float("nan")
        out[name] = {"n_ep": used, "n_pt": int(len(p)), "slope": float(a), "intercept": float(b),
                     "corr": r, "pan_origin_deg": origin,
                     "pan_min": float(p.min()), "pan_max": float(p.max()), "pan_std": float(p.std())}
        print(f"[{i}/{len(dss)}] {name}: 상관 {r:+.3f} · 기울기 {a:+.5f} · 원점 {origin:+.1f}도 "
              f"· pan {p.min():.0f}~{p.max():.0f}", flush=True)
        if i % 10 == 0:
            json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    good = [v for v in out.values() if abs(v.get("corr", 0)) >= 0.5]
    print(f"\n>>> 완료 · 상관 |r|>=0.5 인 데이터셋 {len(good)}/{len(out)}개 · {args.out}")


if __name__ == "__main__":
    main()
