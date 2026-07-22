"""카메라 시점각(pitch) 직접 추정 — 작업면 평면을 맞춰 광축과의 각도를 잰다.

## 왜 다시 만들었나 (eda15의 실패)

eda15는 "깊이가 뚝 끊기는 곳이 있는가"(depth_break)로 시점을 갈랐고 eval 셋업 A/B를
100% 맞혔다. 그러나 train 육안 확인에서 **대량 오분류**가 드러났다 — `pierfabre/sheep`,
`lirislab/so100_demo`처럼 명백한 측면 뷰가 top_down으로 분류됐다.

원인: depth_break가 잰 것은 시점각이 아니라 **"책상 먼 쪽 모서리 너머로 깊이가 급락하는
구성"** 이었다. eval 셋업 B가 우연히 그 구성이라(책상 끝 너머로 바닥이 뚝 떨어짐)
100%가 나왔을 뿐이다. 벽을 등진 측면 뷰는 깊이가 매끄럽게 이어져 급락이 없다.
→ **eval 2개 장면만으로 보정한 지표는 그 2개 장면에만 맞는다**는 교훈. (§한계 참조)

## 이번 방법 — 물리적으로 정의된 양을 잰다

시점각의 정의 자체로 돌아간다: **카메라 광축(렌즈가 향하는 방향)과 작업면(책상·매트)이
이루는 각도**. 0도면 작업면을 정면으로(수직으로) 내려다보는 것 = top-down.
90도에 가까우면 작업면을 스치듯 옆에서 보는 것 = 측면 뷰.

절차:
  1. 단안 **미터 깊이** 추정(Depth Anything V2 Metric Indoor) — 픽셀마다 실제 거리(m).
     상대 깊이가 아니라 미터 값이어야 3차원 복원이 성립한다.
  2. 가정한 초점거리로 각 픽셀을 3차원 점으로 역투영 → 점구름(point cloud).
  3. RANSAC(무작위 표본 합의 — 이상치에 강한 맞춤법)으로 **가장 넓은 평면** 하나를 찾는다.
     로봇 팔·물체는 이상치로 걸러지고 작업면만 남는다.
  4. 그 평면의 법선(수직 방향)과 광축이 이루는 각도 = **시점각(pitch)**.

가정하는 것과 그 영향:
  · 초점거리는 알 수 없다(메타데이터에 없음). 수평화각 60도를 기본으로 두고
    50/70도로도 계산해 **결론이 이 가정에 흔들리는지** 함께 보고한다(--fov-sweep).
  · 미터 깊이 모델은 실내 데이터로 학습됐다. 절대값은 틀릴 수 있으나 시점각은
    깊이의 **상대적 기울기**에서 나오므로 전역 배율 오차에는 둔감하다.

용어: 광축(optical axis, 렌즈가 정면으로 향하는 방향) · 법선(normal, 면에 수직인 방향) ·
      역투영(back-projection, 2차원 픽셀을 3차원 위치로 되돌리는 계산) ·
      화각(FOV, Field of View — 렌즈가 담는 시야의 각도) ·
      인라이어(inlier, 맞춘 모형에 잘 들어맞는 점) · RANSAC(이상치에 강한 맞춤 알고리즘).

사용:
  uv run python scripts/eda18_camera_pitch.py eval
  uv run python scripts/eda18_camera_pitch.py train --frames-per-ds 24
산출: results/eval_camera_pitch_frames.csv, results/train_camera_pitch_frames.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from eda15_viewpoint import (  # noqa: E402
    EVAL_IMAGES,
    RESULTS,
    SETUP_B_START,
    STD_H,
    STD_W,
    load_std,
    to_std,
    train_datasets,
    train_frames,
)

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEFAULT_HFOV = 60.0          # 웹캠 통상값. --fov-sweep로 민감도 확인.


class MetricDepth:
    """미터 단위 깊이(각 픽셀까지의 거리, m)를 배치로 추정."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.proc = AutoImageProcessor.from_pretrained(MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, rgbs: list[np.ndarray]) -> list[np.ndarray]:
        inp = self.proc(images=[Image.fromarray(r) for r in rgbs], return_tensors="pt")
        out = self.model(**inp.to(self.device)).predicted_depth
        out = torch.nn.functional.interpolate(
            out[:, None], size=(STD_H, STD_W), mode="bilinear", align_corners=False
        )[:, 0]
        return [d.float().cpu().numpy() for d in out]


def focal_px(hfov_deg: float, width: int = STD_W) -> float:
    """수평화각(도) → 픽셀 단위 초점거리."""
    return (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)


def backproject(depth: np.ndarray, f: float) -> np.ndarray:
    """깊이맵 → (N, 3) 카메라 좌표계 점구름. 광축 = +Z."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    u = u - (W - 1) / 2.0
    v = v - (H - 1) / 2.0
    z = depth
    return np.stack([z * u / f, z * v / f, z], axis=-1).reshape(-1, 3)


def fit_plane_ransac(
    pts: np.ndarray, iters: int = 300, thr_frac: float = 0.01, rng=None
) -> tuple[np.ndarray, float]:
    """가장 많은 점이 속한 평면의 단위 법선과 인라이어 비율.

    thr_frac: 허용 오차를 '장면 깊이 중앙값의 몇 배'로 준다 (절대 미터 대신 상대값 —
              가까운 책상 장면과 먼 바닥 장면에 같은 기준을 적용하기 위해).
    """
    rng = rng or np.random.default_rng(0)
    n = len(pts)
    if n < 100:
        return np.array([0.0, 0.0, 1.0]), 0.0
    thr = thr_frac * float(np.median(pts[:, 2]))

    # 가설 iters개를 한꺼번에 세우고 한 번의 행렬곱으로 채점한다 (파이썬 반복문 제거).
    tri = rng.integers(0, n, size=(iters, 3))
    p0 = pts[tri[:, 0]]                                   # (I, 3)
    nv = np.cross(pts[tri[:, 1]] - p0, pts[tri[:, 2]] - p0)
    nn = np.linalg.norm(nv, axis=1, keepdims=True)
    ok = nn[:, 0] > 1e-8
    if not ok.any():
        return np.array([0.0, 0.0, 1.0]), 0.0
    nv, p0 = nv[ok] / nn[ok], p0[ok]
    # 점 p가 평면 위에 있을 조건: (p - p0)·n = 0  →  p·n - p0·n
    dist = np.abs(pts @ nv.T - np.sum(p0 * nv, axis=1))    # (N, I)
    counts = (dist < thr).sum(axis=0)
    b = int(np.argmax(counts))
    best_cnt, best_n = int(counts[b]), nv[b]
    if best_cnt == 0:
        return np.array([0.0, 0.0, 1.0]), 0.0
    # 인라이어로 최소제곱 재맞춤 (RANSAC 표본 3점의 잡음을 없앤다)
    inl = pts[dist[:, b] < thr]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    nv = vt[-1]
    return nv / (np.linalg.norm(nv) + 1e-12), best_cnt / n


# 작업면이 있을 것으로 보는 화면 영역 (세로 50~95%, 가로 15~85%).
# 로봇이 물건을 다루는 곳은 거의 항상 화면 아래-가운데이기 때문.
WORK_REGION = (0.50, 0.95, 0.15, 0.85)


def pitch_from_depth(depth: np.ndarray, hfov: float, rng=None) -> tuple[float, float]:
    """(시점각 도, 평면 인라이어 비율).

    0도 = 작업면을 정면으로 내려다봄(top-down), 90도 = 스치듯 옆에서 봄.

    ★ 평면은 **화면 전체가 아니라 작업면 영역에서만** 맞춘다.
    전체에서 '가장 넓은 평면'을 찾으면 뒷벽·바닥처럼 더 넓은 면에 끌려간다.
    실측 실패 사례: `lirislab/lemon_into_bowl`(명백한 수평 측면 뷰)이 뒷벽에 평면이
    맞아 4.6도(=내려다보기)로 나왔다. 재는 대상은 작업면이므로 영역을 고정한다.
    """
    H, W = depth.shape
    f = focal_px(hfov)
    r0, r1, c0, c1 = WORK_REGION
    sub = depth[int(r0 * H) : int(r1 * H), int(c0 * W) : int(c1 * W)]
    # 역투영은 픽셀 좌표가 화면 중심 기준이어야 하므로, 잘라낸 영역의 오프셋을 반영한다.
    hh, ww = sub.shape
    v, u = np.mgrid[0:hh, 0:ww].astype(np.float32)
    u = u + int(c0 * W) - (W - 1) / 2.0
    v = v + int(r0 * H) - (H - 1) / 2.0
    pts = np.stack([sub * u / f, sub * v / f, sub], axis=-1).reshape(-1, 3)[::3]
    nv, inl = fit_plane_ransac(pts, rng=rng)
    cos = abs(float(nv[2]))              # 광축 (0,0,1)과의 각
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))), inl


def run(items, probe, hfovs: list[float], batch: int = 24) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for s in range(0, len(items), batch):
        chunk = items[s : s + batch]
        deps = probe([c[-1] for c in chunk])
        for (k1, k2, _), dep in zip(chunk, deps):
            r = {"key": k1, "group": k2,
                 "depth_p5": float(np.percentile(dep, 5)),
                 "depth_p50": float(np.percentile(dep, 50)),
                 "depth_p95": float(np.percentile(dep, 95))}
            for hf in hfovs:
                p, inl = pitch_from_depth(dep, hf, rng)
                tag = "" if hf == DEFAULT_HFOV else f"_fov{int(hf)}"
                r[f"pitch_deg{tag}"] = p
                r[f"plane_inlier{tag}"] = inl
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["eval", "train"])
    ap.add_argument("--frames-per-ds", type=int, default=24)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--fov-sweep", action="store_true",
                    help="50/60/70도로 함께 계산해 초점거리 가정의 민감도를 본다")
    args = ap.parse_args()

    hfovs = [DEFAULT_HFOV] + ([50.0, 70.0] if args.fov_sweep else [])
    probe = MetricDepth("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    if args.target == "eval":
        items = []
        for p in sorted(EVAL_IMAGES.glob("sample_*.png")):
            idx = int(p.stem.split("_")[1])
            items.append((p.stem, "B" if idx >= SETUP_B_START else "A", load_std(p)))
        df = run(items, probe, hfovs, args.batch).rename(
            columns={"key": "sample_id", "group": "setup"})
        out = RESULTS / "eval_camera_pitch_frames.csv"
        df.to_csv(out, index=False)
        print(f"[eval] {len(df)}장 → {out} ({time.time() - t0:.0f}s)")
        for st, sub in df.groupby("setup"):
            print(f"  셋업 {st}: pitch 중앙값 {sub.pitch_deg.median():.1f}도 "
                  f"(p10 {sub.pitch_deg.quantile(.1):.1f} / p90 {sub.pitch_deg.quantile(.9):.1f}), "
                  f"평면 인라이어 {sub.plane_inlier.median():.2f}")
        return

    dss = train_datasets()
    print(f"[train] 데이터셋 {len(dss)}개 × {args.frames_per_ds}장", flush=True)
    parts = []
    for n, ds in enumerate(dss, 1):
        fr = train_frames(ds, args.frames_per_ds)
        if not fr:
            print(f"  !! {ds}: 프레임 0장", flush=True)
            continue
        d = run([(ds, str(e), r) for e, r in fr], probe, hfovs, args.batch)
        parts.append(d)
        print(f"  [{n}/{len(dss)}] {ds}: {len(d)}장 pitch중앙 {d.pitch_deg.median():5.1f}도 "
              f"({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(parts, ignore_index=True).rename(
        columns={"key": "dataset", "group": "episode_index"})
    out = RESULTS / "train_camera_pitch_frames.csv"
    df.to_csv(out, index=False)
    print(f"[train] {len(df)}장 / {df.dataset.nunique()}개 → {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
