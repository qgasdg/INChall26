"""시점 분포 최종 집계 — 자동 추정(pitch) × 육안 라벨 대조 + train/eval 분포 비교.

산출:
  results/train_viewpoint.csv          데이터셋 1행 (시점각·라벨·육안라벨·규모)
  results/train_viewpoint_summary.json 분포 집계 + 자동/육안 일치도

가설 검증의 핵심 질문 두 가지를 나눠 답한다:
  Q1. train은 정말 'top-down 다수'인가?  → 시점각 분포로 답한다.
  Q2. eval 셋업 B와 **같은 각도대**의 데이터가 train에 몇 %인가? → B 범위 점유율로 답한다.
      (Q1과 Q2는 다른 질문이다. train이 top-down이 아니어도, B와 같은 각도대가
       비어 있을 수 있다.)

용어: 시점각(카메라가 작업면을 얼마나 비스듬히 보는가, 0도=수직으로 내려다봄) ·
      사분위(quartile, 자료를 4등분하는 값) · 일치도(자동 판정과 사람 판정이 같은 비율).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def main() -> None:
    ev = pd.read_csv(RESULTS / "eval_camera_pitch_frames.csv")
    tr = pd.read_csv(RESULTS / "train_camera_pitch_frames.csv")
    cat = pd.read_csv(RESULTS / "train_catalog.csv")[["dataset", "n_episodes", "n_frames"]]
    eye = pd.read_csv(ROOT / "local_eval" / "viewpoint_eye_labels.csv")[["dataset", "eye"]]

    a = ev[ev.setup == "A"].pitch_deg
    b = ev[ev.setup == "B"].pitch_deg
    # eval 두 셋업의 각도대 (프레임 p5~p95). 이 구간이 'A류/B류'의 정의가 된다.
    A_LO, A_HI = float(a.quantile(0.05)), float(a.quantile(0.95))
    B_LO, B_HI = float(b.quantile(0.05)), float(b.quantile(0.95))
    # A 최대와 B 최소 사이 = 두 셋업을 가르는 경계
    THR = (float(a.max()) + float(b.min())) / 2

    g = tr.groupby("dataset")
    ds = pd.DataFrame({
        "n_frames_probed": g.size(),
        "pitch_med": g["pitch_deg"].median(),
        "pitch_p25": g["pitch_deg"].quantile(0.25),
        "pitch_p75": g["pitch_deg"].quantile(0.75),
        "pitch_iqr": g["pitch_deg"].quantile(0.75) - g["pitch_deg"].quantile(0.25),
        "pitch_med_fov50": g["pitch_deg_fov50"].median(),
        "pitch_med_fov70": g["pitch_deg_fov70"].median(),
        "plane_inlier_med": g["plane_inlier"].median(),
        "depth_p50_med": g["depth_p50"].median(),
        "frac_in_B_band": g["pitch_deg"].apply(lambda s: float(((s >= B_LO) & (s <= B_HI)).mean())),
    }).reset_index().merge(cat, on="dataset", how="left").merge(eye, on="dataset", how="left")

    ds["auto"] = np.where(ds.pitch_med >= THR, "low_angle", "top_down")
    ds["like_setup"] = np.select(
        [ds.pitch_med <= A_HI, (ds.pitch_med >= B_LO) & (ds.pitch_med <= B_HI), ds.pitch_med > B_HI],
        ["A류(내려다봄)", "B류(비스듬)", "B보다 더 비스듬"], default="A와 B 사이")
    ds = ds.sort_values("pitch_med").reset_index(drop=True)
    ds.to_csv(RESULTS / "train_viewpoint.csv", index=False)

    # --- 자동 vs 육안 일치도 (?는 제외하고 계산, 제외 수를 함께 보고) -------
    j = ds[ds.eye.isin(["T", "L"])]
    eye_low = j.eye.eq("L")
    auto_low = j.auto.eq("low_angle")
    acc = float((eye_low == auto_low).mean())
    tp = int((eye_low & auto_low).sum()); tn = int((~eye_low & ~auto_low).sum())
    fp = int((~eye_low & auto_low).sum()); fn = int((eye_low & ~auto_low).sum())

    tot_ds, tot_ep, tot_fr = len(ds), ds.n_episodes.sum(), ds.n_frames.sum()

    def share(mask):
        s = ds[mask]
        return {"datasets": int(len(s)), "datasets_pct": round(100 * len(s) / tot_ds, 1),
                "episodes_pct": round(100 * s.n_episodes.sum() / tot_ep, 1),
                "frames_pct": round(100 * s.n_frames.sum() / tot_fr, 1)}

    summary = {
        "eval_anchor": {
            "setup_A_pitch_median": round(float(a.median()), 1),
            "setup_A_band_p5_p95": [round(A_LO, 1), round(A_HI, 1)],
            "setup_B_pitch_median": round(float(b.median()), 1),
            "setup_B_band_p5_p95": [round(B_LO, 1), round(B_HI, 1)],
            "threshold": round(THR, 1),
        },
        "train_pitch_distribution": {
            k: round(float(ds.pitch_med.quantile(q)), 1)
            for k, q in [("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9)]
        },
        "totals": {"datasets": int(tot_ds), "episodes": int(tot_ep), "frames": int(tot_fr)},
        "share_by_band": {
            "A류(<=%.0f도)" % A_HI: share(ds.pitch_med <= A_HI),
            "A와B사이": share((ds.pitch_med > A_HI) & (ds.pitch_med < B_LO)),
            "B류(%.0f~%.0f도)" % (B_LO, B_HI): share((ds.pitch_med >= B_LO) & (ds.pitch_med <= B_HI)),
            "B보다더비스듬(>%.0f도)" % B_HI: share(ds.pitch_med > B_HI),
        },
        "share_auto_low_angle": share(ds.auto.eq("low_angle")),
        "eye_vs_auto": {
            "labelled": int(len(j)), "excluded_ambiguous": int(ds.eye.eq("?").sum()),
            "accuracy": round(acc, 3),
            "eye_low_auto_low": tp, "eye_top_auto_top": tn,
            "eye_top_auto_low": fp, "eye_low_auto_top": fn,
        },
        "eye_label_counts": {k: int(v) for k, v in ds.eye.value_counts().items()},
        "fov_sensitivity_median_pitch": {
            "fov50": round(float(ds.pitch_med_fov50.median()), 1),
            "fov60": round(float(ds.pitch_med.median()), 1),
            "fov70": round(float(ds.pitch_med_fov70.median()), 1),
        },
    }
    (RESULTS / "train_viewpoint_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"eval 기준: A {a.median():.1f}도 (p5~p95 {A_LO:.1f}~{A_HI:.1f}) | "
          f"B {b.median():.1f}도 ({B_LO:.1f}~{B_HI:.1f}) | 경계 {THR:.1f}도\n")
    print("train 시점각 분포(데이터셋 중앙값 기준):")
    for k, v in summary["train_pitch_distribution"].items():
        print(f"  {k:7s} {v:5.1f}도")
    print("\n각도대별 점유율:")
    for k, v in summary["share_by_band"].items():
        print(f"  {k:22s} 데이터셋 {v['datasets']:3d}개 ({v['datasets_pct']:5.1f}%)  "
              f"에피소드 {v['episodes_pct']:5.1f}%  프레임 {v['frames_pct']:5.1f}%")
    e = summary["eye_vs_auto"]
    print(f"\n자동 vs 육안: {e['labelled']}개 중 일치 {e['accuracy']:.1%} "
          f"(애매 {e['excluded_ambiguous']}개 제외)")
    print(f"  육안L·자동L {e['eye_low_auto_low']} / 육안T·자동T {e['eye_top_auto_top']} / "
          f"육안T·자동L {e['eye_top_auto_low']} / 육안L·자동T {e['eye_low_auto_top']}")
    print(f"\n화각 가정 민감도(train 중앙값): 50도 {summary['fov_sensitivity_median_pitch']['fov50']} / "
          f"60도 {summary['fov_sensitivity_median_pitch']['fov60']} / "
          f"70도 {summary['fov_sensitivity_median_pitch']['fov70']}")


if __name__ == "__main__":
    main()
