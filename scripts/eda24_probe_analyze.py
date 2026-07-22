"""프로브셋 분석 — 시점각이 뭉개짐의 원인인가에 대한 최종 판정.

## 분석 단위는 '샘플'이 아니라 '장면(데이터셋)'이다

한 데이터셋에서 뽑은 4개 에피소드는 **서로 독립이 아니다**(같은 카메라·같은 방·같은 과제).
이를 독립 표본 168개로 세면 통계가 실제보다 강해 보인다(유사반복). 그래서 데이터셋
단위로 평균 낸 뒤 **장면 42개**를 표본으로 쓴다.

## 세 가지를 순서대로 본다

1. **구간 간 차이** — 구간별 선명도 유지율 평균. 가설이 옳다면
   top_down > mid > side 순으로 유지율이 떨어져야 한다(비스듬할수록 더 뭉개짐).
2. **연속 관계** — 시점각을 구간이 아닌 연속값으로 두고 상관을 본다. 구간 경계를
   어디로 잡든 무관하게 관계가 있는지 확인.
3. **교란 확인** — 움직임 크기·입력 선명도·정답 영상 유지율이 구간별로 갈리는지.
   갈린다면 그것이 시점각을 가장한 진짜 원인일 수 있다.

또한 eval 셋업 A·B의 실측치(66.6% / 40.5%)를 같은 축에 놓고 비교한다 — 프로브셋에서
같은 크기의 격차가 재현되는지가 핵심이다.

용어: 유사반복(pseudo-replication, 독립이 아닌 표본을 독립처럼 세는 오류) ·
      효과크기 Cohen's d(두 집단 평균차를 표준편차로 나눈 값. 0.2 작음/0.5 중간/0.8 큼) ·
      상관계수 r · p값(우연히 이 정도가 나올 확률).

사용: uv run python scripts/eda24_probe_analyze.py
산출: results/viewpoint_probe_by_band.csv, results/viewpoint_probe_by_scene.csv
"""
from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ORDER = ["top_down", "mid", "side"]


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(x, y)[0, 1])
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, float(erfc(abs(t) / sqrt(2))), n


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (s + 1e-12))


def main() -> None:
    s = pd.read_csv(RESULTS / "viewpoint_probe_samples.csv")

    # --- 장면(데이터셋) 단위로 축약 — 유사반복 방지 ------------------------
    scene = s.groupby(["dataset", "band", "uploader"], as_index=False).agg(
        n_samples=("sample_id", "size"), pitch=("pitch_med", "first"),
        gen_ratio=("gen_ratio", "mean"), gt_ratio=("gt_ratio", "mean"),
        gen_mae=("gen_mae", "mean"), gt_floor=("gt_floor_mae", "mean"),
        mae_above_floor=("mae_above_floor", "mean"),
        motion=("motion_deg", "mean"), input_sharp=("input_sharpness", "mean"),
    )
    scene.to_csv(RESULTS / "viewpoint_probe_by_scene.csv", index=False)

    # --- 1) 구간별 집계 ----------------------------------------------------
    band = scene.groupby("band").agg(
        장면수=("dataset", "size"),
        시점각중앙=("pitch", "median"),
        생성_선명도유지율=("gen_ratio", "mean"),
        유지율_표준편차=("gen_ratio", "std"),
        정답영상_유지율=("gt_ratio", "mean"),
        생성MAE=("gen_mae", "mean"),
        GT바닥MAE=("gt_floor", "mean"),
        바닥위초과=("mae_above_floor", "mean"),
        움직임=("motion", "mean"),
        입력선명도=("input_sharp", "mean"),
    ).reindex(ORDER).round(3)
    band.to_csv(RESULTS / "viewpoint_probe_by_band.csv")

    print("=" * 78)
    print("구간별 결과 (장면 단위 평균, n = 구간당 장면 수)")
    print("=" * 78)
    print(band.to_string())

    td = scene[scene.band == "top_down"]
    sd = scene[scene.band == "side"]
    md = scene[scene.band == "mid"]

    print("\n" + "=" * 78)
    print("가설 검정: 비스듬할수록 더 뭉개지는가?")
    print("=" * 78)
    print(f"top_down 유지율 {td.gen_ratio.mean():.3f} vs side 유지율 {sd.gen_ratio.mean():.3f}"
          f"  차이 {td.gen_ratio.mean() - sd.gen_ratio.mean():+.3f}")
    print(f"  Cohen's d = {cohen_d(td.gen_ratio, sd.gen_ratio):+.3f}  "
          f"(가설이 옳다면 top_down이 더 커야 → d가 양수)")
    print(f"top_down vs mid: {cohen_d(td.gen_ratio, md.gen_ratio):+.3f} / "
          f"mid vs side: {cohen_d(md.gen_ratio, sd.gen_ratio):+.3f}")

    print("\n연속 관계 (장면 42개, 구간 경계와 무관):")
    for lab, col in [("선명도 유지율", "gen_ratio"), ("생성 MAE", "gen_mae"),
                     ("바닥 위 초과 MAE", "mae_above_floor")]:
        r, p, n = pearson(scene.pitch, scene[col])
        print(f"  시점각 ~ {lab:16s} r={r:+.3f} (p={p:.3f}, n={n})")

    print("\n" + "=" * 78)
    print("교란 확인 — 구간 간에 다른 것이 함께 갈리는가")
    print("=" * 78)
    for lab, col in [("정답영상 유지율", "gt_ratio"), ("움직임(도)", "motion"),
                     ("입력 선명도", "input_sharp"), ("GT 바닥 MAE", "gt_floor")]:
        vals = [scene[scene.band == b][col].mean() for b in ORDER]
        d = cohen_d(td[col], sd[col])
        print(f"  {lab:14s} top_down {vals[0]:8.3f} / mid {vals[1]:8.3f} / "
              f"side {vals[2]:8.3f}   d(td-side)={d:+.2f}")

    # --- 2) ★ 진짜 원인 후보 경쟁 — 무엇이 유지율을 예측하나 -----------------
    # 정답영상 대비로 정규화하면 '원본 영상 자체가 흐려지는 정도'를 뺄 수 있다.
    scene["norm_ratio"] = scene.gen_ratio / scene.gt_ratio
    scene["log_sharp"] = np.log(scene.input_sharp)
    print("\n" + "=" * 78)
    print("무엇이 뭉개짐을 예측하나 (장면 42개, 정규화 유지율 = 생성/정답)")
    print("=" * 78)
    for lab, c in [("시점각", "pitch"), ("log(입력 선명도)", "log_sharp"),
                   ("움직임", "motion"), ("정답영상 유지율", "gt_ratio")]:
        r, p, n = pearson(scene[c], scene.norm_ratio)
        print(f"  {lab:16s} r={r:+.3f} p={p:.3f}")

    def partial(x, y, z):
        """z를 통제한 x~y 편상관 = 각각 z로 회귀한 잔차끼리의 상관."""
        X, Y, Z = (scene[c].to_numpy(float) for c in (x, y, z))
        m = ~(np.isnan(X) | np.isnan(Y) | np.isnan(Z))
        X, Y, Z = X[m], Y[m], Z[m]
        rx = X - np.polyval(np.polyfit(Z, X, 1), Z)
        ry = Y - np.polyval(np.polyfit(Z, Y, 1), Z)
        r = float(np.corrcoef(rx, ry)[0, 1])
        n = len(X)
        t = r * np.sqrt((n - 3) / max(1e-12, 1 - r * r))
        return r, float(erfc(abs(t) / sqrt(2))), n

    rp, pp, n = partial("pitch", "norm_ratio", "log_sharp")
    rs, ps, _ = partial("log_sharp", "norm_ratio", "pitch")
    print(f"\n  ★ 입력선명도 통제 후 시점각 편상관   r={rp:+.3f} p={pp:.3f}")
    print(f"  ★ 시점각 통제 후 입력선명도 편상관   r={rs:+.3f} p={ps:.3f}")
    print(f"  → n={n}에서 p<0.05가 되려면 |r|>={1.96 / np.sqrt(n - 3):.3f} 필요"
          f" (그보다 작은 효과는 이 표본으로 못 가른다)")

    # --- 3) eval 격차를 시점각으로 얼마나 설명할 수 있나 --------------------
    EV_A_PITCH, EV_B_PITCH, EV_GAP = 11.7, 33.3, -0.261
    d = EV_B_PITCH - EV_A_PITCH
    b_simple = np.polyfit(scene.pitch, scene.gen_ratio, 1)[0]
    X = np.column_stack([np.ones(len(scene)), scene.pitch, scene.log_sharp])
    b_ctrl = np.linalg.lstsq(X, scene.gen_ratio.to_numpy(float), rcond=None)[0][1]

    print("\n" + "=" * 78)
    print("★ 최종 — eval 셋업 A→B 격차를 시점각이 설명하는가 (docs/17 §1.3)")
    print("=" * 78)
    print(f"  eval 실측       A {EV_A_PITCH}도 0.666 → B {EV_B_PITCH}도 0.405   변화 {EV_GAP:+.3f}")
    print(f"  프로브 직접 측정  {td.pitch.median():.1f}도 {td.gen_ratio.mean():.3f} → "
          f"{md.pitch.median():.1f}도 {md.gen_ratio.mean():.3f}   변화 "
          f"{md.gen_ratio.mean() - td.gen_ratio.mean():+.3f}  ← 부호 반대")
    print(f"  회귀 예측(단순)   {b_simple * d:+.3f}  = eval 격차의 {100 * abs(b_simple * d / EV_GAP):.0f}%")
    print(f"  회귀 예측(통제후) {b_ctrl * d:+.3f}  = eval 격차의 {100 * abs(b_ctrl * d / EV_GAP):.0f}%")

    summary = {
        "n_scenes": int(len(scene)), "n_samples": int(len(s)),
        "band_means_gen_ratio": {b: round(float(scene[scene.band == b].gen_ratio.mean()), 4)
                                 for b in ORDER},
        "band_means_norm_ratio": {b: round(float(scene[scene.band == b].norm_ratio.mean()), 4)
                                  for b in ORDER},
        "monotonic_as_hypothesised": bool(
            scene[scene.band == "top_down"].gen_ratio.mean()
            > scene[scene.band == "mid"].gen_ratio.mean()
            > scene[scene.band == "side"].gen_ratio.mean()),
        "pitch_vs_norm_ratio": {"r": round(pearson(scene.pitch, scene.norm_ratio)[0], 4),
                                "p": round(pearson(scene.pitch, scene.norm_ratio)[1], 4)},
        "partial_pitch_ctrl_sharp": {"r": round(rp, 4), "p": round(pp, 4)},
        "partial_sharp_ctrl_pitch": {"r": round(rs, 4), "p": round(ps, 4)},
        "eval_gap": EV_GAP,
        "probe_same_transition": round(float(md.gen_ratio.mean() - td.gen_ratio.mean()), 4),
        "predicted_gap_simple": round(float(b_simple * d), 4),
        "predicted_gap_controlled": round(float(b_ctrl * d), 4),
        "detectable_r_at_n": round(float(1.96 / np.sqrt(len(scene) - 3)), 4),
    }
    import json
    (RESULTS / "viewpoint_probe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    scene.to_csv(RESULTS / "viewpoint_probe_by_scene.csv", index=False)


if __name__ == "__main__":
    main()
