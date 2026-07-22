"""프로브셋 측정 — 생성 영상의 뭉개짐(선명도 유지율) + 행동 오차 + GT 바닥값.

## 재는 것 3종

1. **선명도 유지율** (주 지표) = 생성 영상 마지막 프레임 선명도 / 첫 프레임 선명도.
   선명도 = 라플라시안 분산(영상이 얼마나 또렷한지 재는 값). docs/17 §1.3과 같은 정의.
   **비율을 쓰는 이유**: 절대 선명도는 장면 질감에 좌우된다(체크무늬 바닥은 원래 높다).
   비율은 그 영향을 상쇄한다. 프로브셋에서도 구간별 입력 선명도가 155~389으로 갈리므로
   비율이 아니면 비교가 성립하지 않는다.

2. **정답 영상의 선명도 유지율** (대조군) — 진짜 영상은 뭉개지지 않으니 1 근처여야 한다.
   이 값이 구간별로 갈리면 "생성이 뭉갠 것"이 아니라 원본 영상 자체의 성질이므로
   반드시 빼고 봐야 한다.

3. **행동 오차(Action MAE)와 그 바닥값**.
   ★ [docs/08](../docs/08_로컬리더보드_구축.md)에 따르면 **킷의 행동 추출기는 train
   도메인에서 신뢰할 수 없다**(정답 영상을 넣어도 MAE 1.49). 그래서 생성 영상의 MAE만
   보면 안 되고, **같은 장면의 정답 영상을 추출기에 넣은 값(=바닥값)** 을 함께 재서
   그 차이(생성 − 바닥)를 봐야 한다. 바닥값이 구간별로 다르면 MAE 원값 비교는 무효다.

용어: 라플라시안 분산(영상의 또렷함 지표, 클수록 선명) · MAE(Mean Absolute Error, 평균
      절대 오차, 낮을수록 좋음) · z-점수 공간(평균을 빼고 표준편차로 나눈 값. 킷이 이
      공간에서 채점한다) · 바닥값(floor, 완벽한 입력을 줘도 남는 오차) · 대조군(control,
      비교 기준이 되는 조건).

사용: uv run python scripts/eda23_probe_measure.py --tag vp50
산출: results/viewpoint_probe_samples.csv (샘플 1행)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_eval"))

from episode_io import read_actions, read_frames  # noqa: E402
from kit_bridge import KitScorer, action_mae, get_device  # noqa: E402

RESULTS = ROOT / "results"
WINDOW = 16


def read_mp4(path: Path) -> np.ndarray:
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            frames.append(f.to_ndarray(format="rgb24"))
    return np.stack(frames)


def sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def sharp_stats(vid: np.ndarray, prefix: str) -> dict:
    s = [sharpness(f) for f in vid]
    return {
        f"{prefix}_f0": s[0],
        f"{prefix}_flast": s[-1],
        f"{prefix}_ratio": s[-1] / (s[0] + 1e-9),
        f"{prefix}_mean": float(np.mean(s)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="vp50")
    ap.add_argument("--pred-root", default=None)
    ap.add_argument("--workdir", default="local_runs")
    args = ap.parse_args()

    man = pd.read_csv(RESULTS / "viewpoint_probe_manifest.csv")
    pred = Path(args.pred_root or f"local_runs/pred_{args.tag}")

    # Action MAE는 여기서 직접 계산한다.
    # 제출 킷(make_submission_csv.py)은 eval 216개 sample_id를 하드코딩하고 있어
    # 임의 프로브셋에는 쓸 수 없다. 대신 kit_bridge(킷 추출기·통계를 그대로 로드,
    # docs/08에서 킷 재현 검증됨)로 같은 산식을 적용한다:
    #   MAE = mean|extractor(영상) − z정규화(정답행동)|  (z-점수 공간)
    scorer = KitScorer(get_device())
    rows = []
    for i, r in enumerate(man.itertuples(), 1):
        mp4 = pred / f"{r.sample_id}.mp4"
        if not mp4.exists():
            print(f"  !! {r.sample_id}: mp4 없음", flush=True)
            continue
        gen = read_mp4(mp4)
        gt = read_frames(r.dataset, r.episode_index, r.start, WINDOW)
        acts = read_actions(r.dataset, r.episode_index, r.start, WINDOW)

        row = {"sample_id": r.sample_id, "band": r.band, "dataset": r.dataset,
               "uploader": r.uploader, "pitch_med": r.pitch_med,
               "motion_deg": r.motion_deg, "input_sharpness": r.input_sharpness,
               "n_gen_frames": len(gen)}
        row.update(sharp_stats(gen, "gen"))
        row.update(sharp_stats(gt, "gt"))

        target = scorer.normalize_actions(acts)
        with torch.no_grad():
            # 생성 영상의 행동 오차
            vg = scorer.to_eval_video(gen).unsqueeze(0)          # (1,16,320,512,3) uint8
            row["gen_mae"] = action_mae(scorer.action_pred(vg)[0], target)
            # GT 바닥값: 정답 영상을 추출기에 넣었을 때의 MAE (도달 가능한 최소)
            vt = scorer.to_eval_video(gt).unsqueeze(0)
            row["gt_floor_mae"] = action_mae(scorer.action_pred(vt)[0], target)
        row["mae_above_floor"] = row["gen_mae"] - row["gt_floor_mae"]
        rows.append(row)
        if i % 20 == 0 or i == len(man):
            print(f"  {i}/{len(man)}", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS / "viewpoint_probe_samples.csv"
    df.to_csv(out, index=False)
    print(f"\n{len(df)}샘플 → {out}")
    print(df.groupby("band").agg(
        n=("sample_id", "size"), 시점각=("pitch_med", "median"),
        생성유지율=("gen_ratio", "mean"), 정답유지율=("gt_ratio", "mean"),
        생성MAE=("gen_mae", "mean"), GT바닥=("gt_floor_mae", "mean"),
        바닥위=("mae_above_floor", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
