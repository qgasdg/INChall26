"""시점각이 '뭉개짐'을 설명하는가 — 셋업 안/밖을 나눠 본 상관 분석.

가설([17 §1.3](../docs/17_E2_학습전략_초안.md))은 "낯선 시점이라 시간적 일관성이 빨리
무너진다"고 말한다. 이 말이 맞다면 **시점이 비스듬할수록 더 뭉개져야** 한다.

핵심 검정 설계 — 셋업 사이(between)와 셋업 안(within)을 반드시 나눈다:
  eval에는 장면이 A·B **딱 2종**뿐이다. 그래서 A와 B가 다르기만 하면 어떤 변수든
  전체 상관이 크게 나온다(집단이 둘뿐이라 '두 점을 잇는 직선'이 되기 때문).
  시점각이 진짜 원인이라면 **같은 셋업 안에서도** 각도가 클수록 더 뭉개져야 한다.
  안에서 관계가 사라지면, 전체 상관은 시점각이 아니라 '셋업이 다르다'는 사실을 잰 것이다.

용어: 상관계수 r(두 값이 함께 움직이는 정도, -1~+1, 0이면 무관) · p값(우연히 이만큼
      나올 확률, 작을수록 우연이 아님) · 교란(confounding, 실제 원인이 아닌 것이
      원인처럼 보이는 현상) · 검정력(power, 효과가 있을 때 그것을 잡아낼 능력).

사용: uv run python scripts/eda21_pitch_vs_blur.py
산출: results/pitch_vs_blur.csv (셋업별 상관 표)
"""
from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def pearson(x, y) -> tuple[float, float, int]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(x, y)[0, 1])
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, float(erfc(abs(t) / sqrt(2))), n      # 정규근사 양측 p값


def main() -> None:
    s = pd.read_csv(RESULTS / "generated_sharpness.csv")
    p = pd.read_csv(RESULTS / "eval_camera_pitch_frames.csv")
    d = s.merge(p, on="sample_id", suffixes=("", "_p"))
    assert (d.setup == d.setup_p).all(), "셋업 라벨 불일치"
    # 장면 안에 깊이 변화가 얼마나 있나 (평면 하나로 설명되는 정도의 반대편 지표)
    d["depth_rng"] = (d.depth_p95 - d.depth_p5) / d.depth_p50

    targets = [("sharp_ratio", "선명도 유지율"), ("mae", "Action 오차")]
    drivers = [("pitch_deg", "시점각"), ("depth_rng", "깊이 폭"),
               ("plane_inlier", "평면 설명력")]
    rows = []
    for tcol, tname in targets:
        for dcol, dname in drivers:
            for gname, sub in [("전체(A+B)", d), ("셋업A 안", d[d.setup == "A"]),
                               ("셋업B 안", d[d.setup == "B"])]:
                r, pv, n = pearson(sub[dcol], sub[tcol])
                rows.append({"대상": tname, "설명변수": dname, "범위": gname,
                             "n": n, "r": round(r, 3), "p": round(pv, 4)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "pitch_vs_blur.csv", index=False)

    print(out.to_string(index=False))
    print("\n셋업별 평균:")
    print(d.groupby("setup")[["pitch_deg", "plane_inlier", "depth_rng",
                              "sharp_ratio", "mae"]].mean().round(3).to_string())
    print("\n셋업 안 시점각 범위 (검정력 확인 — 범위가 좁으면 효과를 잡기 어렵다):")
    for st, sub in d.groupby("setup"):
        print(f"  셋업 {st}: p5 {sub.pitch_deg.quantile(.05):.1f}도 ~ "
              f"p95 {sub.pitch_deg.quantile(.95):.1f}도 "
              f"(폭 {sub.pitch_deg.quantile(.95) - sub.pitch_deg.quantile(.05):.1f}도)")


if __name__ == "__main__":
    main()
