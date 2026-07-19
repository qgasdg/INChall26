"""Action 오차의 축별 분해 — 오차가 실제로 어디에 있는지 찾는다.

■ 왜 하는가
대회 점수의 40%가 Action 성분이고, 베이스라인의 Action MAE는 0.58로 가장 큰 실점 항목이다.
그런데 이 0.58이 6개 관절축에 어떻게 나뉘어 있는지는 아무도 재본 적이 없다.
EDA에서 wrist_roll(손목 회전축)이 train과 eval 사이에 +87도 어긋나 있고
train 128개 중 100개가 eval의 작동 구간을 본 적조차 없음을 확인했으므로
(results/axis_offset_analysis.csv), 이 축에 오차가 몰려 있을 것으로 예상된다.
사실이면 여기가 최대 개선 지점이고, 아니면 그 가설을 버려야 한다.

■ 어떻게 재는가
킷의 행동 추출기(IDM, 영상을 보고 로봇이 어떻게 움직였는지 되짚는 모델)로
생성 영상에서 행동을 뽑아내고, 정답 행동(eval의 actions/*.npy)과 축별로 비교한다.
킷과 동일하게 z-점수 공간(전역 평균·표준편차로 정규화한 공간)에서 잰다 —
그래야 대회 점수와 같은 척도가 된다.

  Action MAE = 6축 × 16프레임에 대한 |예측 - 정답|의 평균
  → 축별로 쪼개면 각 축이 전체 MAE에 얼마나 기여하는지 나온다.

편향(bias)과 산포(spread)도 나눠 본다.
  - 편향 = 부호 있는 평균 오차. 한쪽으로 일관되게 치우쳤다는 뜻 → 보정으로 고칠 수 있다.
  - 산포 = 편향을 뺀 나머지. 예측이 들쭉날쭉하다는 뜻 → 모델 성능 문제.
이 구분이 중요한 이유: 편향이 크면 값싼 후처리로 잡히지만, 산포는 학습으로만 줄어든다.

용어: IDM(Inverse Dynamics Model, 영상에서 행동을 역추정하는 모델) · z-점수(평균을 빼고
      표준편차로 나눠 축마다 다른 단위를 맞춘 값) · 편향(bias, 일관된 치우침) ·
      MAE(Mean Absolute Error, 평균 절대 오차).

사용: .venv/bin/python scripts/eda11_action_axis_error.py --videos local_runs/pred_v100_s50
산출: results/action_axis_error.csv, results/action_axis_error_per_sample.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_eval"))

from kit_bridge import KitScorer, TEMPORAL_LENGTH, get_device  # noqa: E402
import kit_bridge  # noqa: E402

kit = kit_bridge.kit
DIMS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="생성 영상 디렉토리 (<sample_id>.mp4)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None, help="산출 파일명 접미사")
    args = ap.parse_args()

    device = get_device(args.device)
    scorer = KitScorer(device)
    vdir = Path(args.videos)
    act_root = ROOT / "open" / "data" / "eval" / "actions"

    rows = []
    all_err = []  # 축별 편향 제거 후 오차를 정확히 내려면 원시 오차가 필요하다 (216×16×6 = 작음)
    paths = sorted(vdir.glob("*.mp4"))
    print(f"영상 {len(paths)}개 채점 (device={device})", flush=True)
    for n, p in enumerate(paths, 1):
        sid = p.stem
        gt_deg = np.load(act_root / f"{sid}.npy")            # (16,6) 원시 각도
        gen = kit.read_video_uint8(p, expected_frames=TEMPORAL_LENGTH).numpy()
        video = scorer.to_eval_video(gen).unsqueeze(0)
        pred_z = scorer.action_pred(video)[0]                 # (16,6) z-점수 공간
        gt_z = scorer.normalize_actions(gt_deg)               # (16,6) 동일 공간
        err = (pred_z - gt_z).numpy()                         # 부호 있는 오차
        row = {"sample_id": sid}
        for i, d in enumerate(DIMS):
            row[f"mae_{d}"] = float(np.abs(err[:, i]).mean())
            row[f"bias_{d}"] = float(err[:, i].mean())
        row["mae_all"] = float(np.abs(err).mean())
        rows.append(row)
        all_err.append(err)
        if n % 40 == 0 or n == len(paths):
            print(f"  {n}/{len(paths)}", flush=True)

    per = pd.DataFrame(rows)
    tag = f"_{args.tag}" if args.tag else ""
    per.to_csv(ROOT / "results" / f"action_axis_error_per_sample{tag}.csv", index=False)

    E = np.stack(all_err)  # (n, 16, 6)
    total = per["mae_all"].mean()
    out = []
    for d in DIMS:
        mae = per[f"mae_{d}"].mean()
        bias = per[f"bias_{d}"].mean()
        # 축 전체 편향을 완벽히 제거했을 때 남는 오차 = 산포 성분
        col = E[:, :, DIMS.index(d)]
        resid = float(np.abs(col - col.mean()).mean())
        out.append({
            "axis": d,
            "mae": round(mae, 4),
            "share_of_total_pct": round(100 * mae / (total * 6), 1),
            "bias": round(bias, 4),
            "abs_bias_over_mae_pct": round(100 * abs(bias) / mae, 1) if mae else 0.0,
            "mae_if_bias_removed": round(resid, 4),
        })
    df = pd.DataFrame(out).sort_values("mae", ascending=False)
    df.to_csv(ROOT / "results" / f"action_axis_error{tag}.csv", index=False)

    print(f"\n=== Action MAE 축별 분해 (전체 {total:.4f}, n={len(per)}) ===")
    print(df.to_string(index=False))
    print("\n해석 도움말:")
    print("  share_of_total_pct = 6축 합에서 이 축이 차지하는 비중 (균등하면 16.7%)")
    print("  abs_bias_over_mae_pct = 오차 중 '일관된 치우침'의 비율. 높을수록 보정으로 고치기 쉽다")
    worst = df.iloc[0]
    print(f"\n최대 실점 축: {worst.axis} (MAE {worst.mae}, 전체의 {worst.share_of_total_pct}%, "
          f"이 중 편향분 {worst.abs_bias_over_mae_pct}%)")
    print(f"→ results/action_axis_error{tag}.csv")


if __name__ == "__main__":
    main()
