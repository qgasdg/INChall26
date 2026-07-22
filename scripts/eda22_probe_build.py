"""시점 통제 프로브셋 구축 — "시점각이 뭉개짐의 원인인가"를 직접 가르기 위한 검증셋.

## 왜 필요한가

eval에는 장면이 셋업 A·B **2종뿐**이라, A와 B가 동시에 다른 모든 것(시점각·깊이 구조·
재질·조명·로봇 색·과제) 중 무엇이 뭉개짐의 원인인지 **원리적으로 구분할 수 없다**
([21 §6](../docs/21_시점각_분포_검증.md)). 표본이 2개면 어떤 통계도 이 교란을 못 푼다.

→ 그래서 **train에서 시점각만 달리하고 나머지는 최대한 맞춘 장면들**을 직접 뽑아
같은 조건으로 생성·채점한다. 장면이 40개 넘게 확보되면 시점각과 뭉개짐의 관계를
교란 없이 볼 수 있다.

## 통제 설계

**구간 3종** (시점각 = 카메라 광축과 작업면이 이루는 각. 0도=수직으로 내려다봄):
  · `top_down` ≤15도  — eval 셋업 A류
  · `mid`   25~40도  — eval 셋업 B와 같은 각도대 (B는 33.3도)
  · `side`  ≥55도    — B보다 훨씬 비스듬 (가설이 옳다면 여기서 가장 심해야 함)

**맞춘 것(통제 변수)** — 구간 간에 체계적 차이가 생기지 않도록 고정:
  · 해상도 640×480 · 6fps 고정 (다른 값은 제외)
  · 데이터셋당 에피소드 4개, 구간당 데이터셋 14개로 **개수 균형**
  · **업로더 계정당 최대 2개** — `sihyun77` 7개처럼 한 연구실이 한 구간을 지배하면
    "시점 차이"가 아니라 "그 연구실 차이"를 재게 된다 (유사반복, pseudo-replication)
  · 창(window)은 에피소드에서 **움직임이 가장 큰 16프레임**으로 선택 — eval이 움직임
    구간만 담고 있어([10](../docs/10_조사_eval도메인.md)) 이에 맞춘다. 정지 구간이
    섞이면 "안 움직여서 안 뭉개진 것"과 구분이 안 된다.

**제외한 것**:
  · `Chojins` 계열 — 어안(fisheye) 렌즈라 시점각 추정이 무효 ([21 §9](../docs/21_시점각_분포_검증.md))
  · 데이터셋 내 시점 편차(IQR)가 15도 초과 — 한 데이터셋에 카메라 배치가 여러 개라 구간 배정 불가
  · 극단 구간(top_down·side)은 **자동 추정과 육안 라벨이 일치하는 것만** 사용.
    구간 배정이 확실해야 인과 해석이 성립한다. (대표성을 일부 희생하고 내적 타당도를 취함)

**측정할 교란 후보** — 통제가 아니라 사후 확인용. 구간 간에 이것들이 갈리면 해석에 반영한다:
  움직임 크기 · 입력 프레임 선명도 · 에피소드 길이 · 코덱.

용어: 프로브셋(probe set, 특정 가설을 겨냥해 일부러 구성한 검증용 표본) · 교란(confounding,
      진짜 원인이 아닌 것이 원인처럼 보이는 현상) · 유사반복(pseudo-replication, 서로
      독립이 아닌 표본을 독립인 것처럼 세는 오류) · 창/윈도우(연속한 16프레임 구간) ·
      라플라시안 분산(영상이 얼마나 또렷한지 재는 값. 클수록 선명).

사용: uv run python scripts/eda22_probe_build.py --out local_runs/probe_inputs
산출: results/viewpoint_probe_manifest.csv + <out>/images/*.png + <out>/actions/*.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_eval"))

from episode_io import TRAIN_ROOT, read_actions, read_frames  # noqa: E402

RESULTS = ROOT / "results"
WINDOW = 16
PER_BAND = 14          # 구간당 데이터셋 수
EPS_PER_DS = 4         # 데이터셋당 에피소드 수
MAX_PER_UPLOADER = 2

BANDS = {           # (최소, 최대, 육안라벨 일치 요구 여부)
    "top_down": (0.0, 15.0, "T"),
    "mid": (25.0, 40.0, None),
    "side": (55.0, 90.0, "L"),
}


def select_datasets() -> pd.DataFrame:
    d = pd.read_csv(RESULTS / "train_viewpoint.csv")
    c = pd.read_csv(RESULTS / "train_catalog.csv")[
        ["dataset", "fps", "height", "width", "codec", "len_median"]]
    d = d.merge(c, on="dataset")
    d["uploader"] = d.dataset.str.split("/").str[0]

    keep = ~(
        d.dataset.str.startswith("Chojins")     # 어안 → 시점각 무효
        | (d.fps != 6) | (d.height != 480) | (d.width != 640)   # 규격 통제
        | (d.pitch_iqr > 15)                    # 데이터셋 내 시점 이질
        | (d.n_episodes < EPS_PER_DS) | (d.len_median < 20)
    )
    d = d[keep]

    out = []
    for band, (lo, hi, eye_req) in BANDS.items():
        m = (d.pitch_med >= lo) & (d.pitch_med <= hi)
        if eye_req:
            m &= d.eye.eq(eye_req)
        s = d[m].sort_values("pitch_med")
        s = s.groupby("uploader").head(MAX_PER_UPLOADER)     # 업로더 편중 차단
        if len(s) > PER_BAND:
            # 구간의 각도 범위를 고르게 덮도록 균등 간격 추출 (앞쪽만 쓰면 편향)
            idx = np.linspace(0, len(s) - 1, PER_BAND).round().astype(int)
            s = s.iloc[sorted(set(idx.tolist()))]
        s = s.copy()
        s["band"] = band
        out.append(s)
        print(f"[{band}] {len(s)}개 (업로더 {s.uploader.nunique()}) "
              f"시점각 {s.pitch_med.min():.1f}~{s.pitch_med.max():.1f}도")
    return pd.concat(out, ignore_index=True)


def episode_list(ds: str) -> list[tuple[int, int]]:
    eps = []
    with (TRAIN_ROOT / ds / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("length", 0) >= WINDOW + 2:
                    eps.append((e["episode_index"], e["length"]))
            except json.JSONDecodeError:
                pass
    return eps


def best_window(actions: np.ndarray) -> tuple[int, float]:
    """움직임이 가장 큰 16프레임 창의 (시작 인덱스, 총 이동량[도]).

    프레임 간 관절 변화량의 절대값 합을 창 단위로 더해 최대인 곳을 고른다.
    """
    step = np.abs(np.diff(actions, axis=0)).sum(axis=1)      # (T-1,)
    if len(step) < WINDOW - 1:
        return 0, float(step.sum())
    csum = np.concatenate([[0.0], np.cumsum(step)])
    tot = csum[WINDOW - 1 :] - csum[: len(csum) - WINDOW + 1]  # 각 시작점의 창 합
    s = int(np.argmax(tot))
    return s, float(tot[s])


def sharpness(img: np.ndarray) -> float:
    """라플라시안 분산 = 선명도. docs/17 §1.3과 동일 정의."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="local_runs/probe_inputs")
    args = ap.parse_args()

    sel = select_datasets()
    out = ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "actions").mkdir(parents=True, exist_ok=True)

    rows, n = [], 0
    for r in sel.itertuples():
        eps = episode_list(r.dataset)
        if not eps:
            print(f"  !! {r.dataset}: 유효 에피소드 없음", flush=True)
            continue
        idx = np.linspace(0, len(eps) - 1, min(EPS_PER_DS, len(eps))).round().astype(int)
        for i in sorted(set(idx.tolist())):
            ep, ln = eps[i]
            try:
                acts_all = read_actions(r.dataset, ep, 0, ln)
                start, motion = best_window(acts_all)
                start = min(start, ln - WINDOW)
                acts = acts_all[start : start + WINDOW]
                frame0 = read_frames(r.dataset, ep, start, 1)[0]
            except Exception as e:                      # noqa: BLE001
                print(f"  !! {r.dataset} ep{ep}: {e}", flush=True)
                continue
            if acts.shape != (WINDOW, 6):
                continue
            sid = f"vp_{n:05d}"
            Image.fromarray(frame0).save(out / "images" / f"{sid}.png")
            np.save(out / "actions" / f"{sid}.npy", acts.astype(np.float32))
            rows.append({
                "sample_id": sid, "band": r.band, "dataset": r.dataset,
                "uploader": r.uploader, "pitch_med": r.pitch_med,
                "episode_index": ep, "start": start, "ep_length": ln,
                "motion_deg": motion, "input_sharpness": sharpness(frame0),
                "codec": r.codec, "eye": r.eye,
            })
            n += 1
        print(f"  {r.band:8s} {r.dataset:52s} {r.pitch_med:5.1f}도 → 누적 {n}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "viewpoint_probe_manifest.csv", index=False)
    print(f"\n총 {len(df)}샘플 / {df.dataset.nunique()}장면 → {out}")
    print(df.groupby("band").agg(
        장면=("dataset", "nunique"), 샘플=("sample_id", "size"),
        시점각=("pitch_med", "median"), 움직임=("motion_deg", "median"),
        입력선명도=("input_sharpness", "median")).round(1).to_string())


if __name__ == "__main__":
    main()
