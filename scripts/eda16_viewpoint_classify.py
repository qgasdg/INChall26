"""시점 분류 + 분포 집계 — eda15가 뽑은 신호를 데이터셋 라벨로 바꾼다.

eda15(무거운 특징 추출)와 분리한 이유: 임계값·분류 규칙을 손볼 때마다 영상을
다시 디코딩하지 않기 위해서다. 이 스크립트는 CSV만 읽으므로 1초면 끝난다.

분류 규칙 (eval 셋업 A/B를 정답 기준으로 보정):
  주 신호 = depth_break(깊이 불연속) — eval에서 A 최대 0.035 / B 최소 0.109으로
            겹침 없이 갈렸다. 물리적 의미는 "작업면 너머로 깊이가 뚝 끊기는가",
            즉 **작업대 먼 쪽 모서리 너머가 화면에 잡히는가**이다.
  보조 신호 = tex_ratio_top_bot(위/아래 질감비), horiz_edge_strength(수평 경계 세기)
            — 원리가 다른 픽셀 통계. 주 신호와 어긋나면 '육안 확인 대상'으로 표시한다.

라벨 3종:
  top_down  : 내려다보는 시점. 작업면이 화면을 채우고 배경이 없다. (eval 셋업 A류)
  low_angle : 낮은 각도에서 가로질러 보는 시점. 작업대 너머 배경이 보인다. (셋업 B류)
  mixed     : 데이터셋 안에서 프레임마다 갈린다(장면·카메라가 여러 개). 별도 취급.

용어: 임계값(threshold, 이쪽/저쪽을 가르는 경계 숫자) · 중앙값(median, 크기순 한가운데
      값 — 평균과 달리 극단값에 덜 흔들린다) · 기하평균(두 수를 곱해 제곱근 — 배율
      척도에서 '가운데'를 잡을 때 산술평균보다 적절하다).

사용: uv run python scripts/eda16_viewpoint_classify.py
산출: results/train_viewpoint.csv (데이터셋 1행), results/train_viewpoint_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# eval 보정으로 얻은 경계. A 최대 0.035 / B 최소 0.109의 기하평균 ≈ 0.062.
# 배율 척도라 산술평균(0.072)보다 기하평균이 양쪽에 공평하다.
BREAK_THR = 0.062
# 프레임 단위 라벨이 이 비율 이상 한쪽이어야 데이터셋 라벨을 확정한다.
DOMINANCE = 0.70


def frame_labels(df: pd.DataFrame) -> pd.Series:
    return np.where(df["depth_break"] >= BREAK_THR, "low_angle", "top_down")


def main() -> None:
    ev = pd.read_csv(RESULTS / "eval_viewpoint_frames.csv")
    tr = pd.read_csv(RESULTS / "train_viewpoint_frames.csv")
    cat = pd.read_csv(RESULTS / "train_catalog.csv")[["dataset", "n_episodes", "n_frames"]]

    # --- eval 보정 재확인 (라벨이 정답과 얼마나 맞는지) --------------------
    ev["label"] = frame_labels(ev)
    ev_acc = float((ev["label"].eq("low_angle") == ev["setup"].eq("B")).mean())

    # --- train 프레임 라벨 → 데이터셋 집계 --------------------------------
    tr["label"] = frame_labels(tr)
    g = tr.groupby("dataset")
    ds = pd.DataFrame({
        "n_frames_probed": g.size(),
        "low_angle_frac": g["label"].apply(lambda s: float(s.eq("low_angle").mean())),
        "depth_break_med": g["depth_break"].median(),
        "depth_break_p10": g["depth_break"].quantile(0.10),
        "depth_break_p90": g["depth_break"].quantile(0.90),
        "tex_ratio_med": g["tex_ratio_top_bot"].median(),
        "horiz_edge_str_med": g["horiz_edge_strength"].median(),
        "split_row_med": g["split_row"].median(),
        "n_horiz_upper_med": g["n_horiz_upper"].median(),
        "table_edge_slope_med": g["table_edge_slope_deg"].median(),
    }).reset_index()

    def lab(f: float) -> str:
        if f >= DOMINANCE:
            return "low_angle"
        if f <= 1 - DOMINANCE:
            return "top_down"
        return "mixed"

    ds["viewpoint"] = ds["low_angle_frac"].apply(lab)

    # --- 보조 신호와의 일치 여부 (육안 확인 우선순위) ----------------------
    # eval에서 셋업 B의 질감비 중앙값은 4.6, A는 1.79였다. 2.5를 보조 경계로 둔다.
    ds["aux_says_low"] = ds["tex_ratio_med"] >= 2.5
    ds["agree"] = ds["aux_says_low"] == ds["viewpoint"].eq("low_angle")
    # 경계에서 얼마나 떨어져 있나 (작을수록 불확실 → 육안 확인 우선)
    ds["margin"] = (np.log(ds["depth_break_med"] + 1e-6) - np.log(BREAK_THR)).abs()
    ds["needs_eyeball"] = (~ds["agree"]) | ds["viewpoint"].eq("mixed") | (ds["margin"] < 0.35)

    ds = ds.merge(cat, on="dataset", how="left")
    ds = ds.sort_values("depth_break_med", ascending=False).reset_index(drop=True)
    out = RESULTS / "train_viewpoint.csv"
    ds.to_csv(out, index=False)

    # --- 분포 집계 --------------------------------------------------------
    tot_ds, tot_ep, tot_fr = len(ds), ds.n_episodes.sum(), ds.n_frames.sum()
    summary = {
        "eval_calibration_accuracy": ev_acc,
        "break_threshold": BREAK_THR,
        "dominance": DOMINANCE,
        "totals": {"datasets": int(tot_ds), "episodes": int(tot_ep), "frames": int(tot_fr)},
        "by_viewpoint": {},
        "frame_level_low_angle_share": float(tr["label"].eq("low_angle").mean()),
        "needs_eyeball": int(ds.needs_eyeball.sum()),
        "aux_disagree": int((~ds.agree).sum()),
    }
    for v, sub in ds.groupby("viewpoint"):
        summary["by_viewpoint"][v] = {
            "datasets": int(len(sub)), "datasets_pct": round(100 * len(sub) / tot_ds, 2),
            "episodes": int(sub.n_episodes.sum()),
            "episodes_pct": round(100 * sub.n_episodes.sum() / tot_ep, 2),
            "frames": int(sub.n_frames.sum()),
            "frames_pct": round(100 * sub.n_frames.sum() / tot_fr, 2),
        }
    (RESULTS / "train_viewpoint_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(f"eval 보정 정확도: {ev_acc:.4f}  (임계 depth_break={BREAK_THR})")
    print(f"→ {out}\n")
    print(f"{'라벨':<10} {'데이터셋':>10} {'에피소드':>12} {'프레임':>14}")
    for v in ("top_down", "mixed", "low_angle"):
        s = summary["by_viewpoint"].get(v)
        if s:
            print(f"{v:<10} {s['datasets']:5d} ({s['datasets_pct']:5.1f}%) "
                  f"{s['episodes']:6d} ({s['episodes_pct']:5.1f}%) "
                  f"{s['frames']:8d} ({s['frames_pct']:5.1f}%)")
    print(f"\n육안 확인 대상 {summary['needs_eyeball']}개 (보조신호 불일치 {summary['aux_disagree']}개 포함)")


if __name__ == "__main__":
    main()
