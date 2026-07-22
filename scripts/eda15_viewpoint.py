"""카메라 시점(각도) 정량화 — train 128 데이터셋 vs eval 셋업 A/B (Phase 5).

목적: "셋업 B(낮은 각도 측면 뷰)가 train에서 과소대표되어 생성 영상이 더 심하게
      뭉개진다"는 가설([17 §1.3](../docs/17_E2_학습전략_초안.md))을 검증한다.
      카메라가 작업대를 **위에서 내려다보는가(top-down)** 아니면 **책상 높이에서
      가로질러 보는가(low-angle)** 를 이미지 기하로 자동 분류한다.

핵심 아이디어 — 시점이 다르면 화면 구성이 달라진다:
  · top-down(내려다보기): 작업대 표면이 화면을 가득 채운다. 배경(멀리 있는 것)이 없다.
    따라서 화면 위/아래의 성질이 비슷하고, 깊이(카메라로부터의 거리)가 거의 균일하다.
  · low-angle(가로질러 보기): 작업대의 **먼 쪽 모서리**가 화면을 가로지르는 경계선으로
    보이고, 그 위쪽에는 배경(바닥·의자·벽)이 보인다. 화면 위는 멀고 아래는 가깝다.

그래서 두 갈래의 독립 신호를 각각 재고 서로 교차 검증한다:
  (A) 기하 신호 — 화면을 위/아래로 가르는 **수평 경계선**의 존재·위치·세기,
      위/아래 영역의 질감 차이, 수평 직선 성분의 분포. (고전 영상처리, 모델 불필요)
  (B) 깊이 신호 — 단안 깊이 추정 모델(Depth Anything V2 Small)로 상대 깊이를 뽑아
      **화면 세로 방향 깊이 기울기**를 잰다. 카메라 각도의 직접적 대리 지표다.

두 갈래는 원리가 완전히 다르므로(픽셀 통계 vs 학습된 깊이) 일치하면 신뢰도가 높고,
어긋나는 데이터셋은 육안 확인 대상으로 따로 표시한다.

용어: 단안 깊이 추정(사진 한 장으로 각 픽셀까지의 거리를 추정하는 기법) ·
      시차(disparity, 깊이의 역수 — 값이 클수록 가까움) · 소벨(Sobel, 밝기가 급변하는
      가장자리를 찾는 필터) · 백분위수(percentile, 값을 크기순으로 늘어놨을 때의 위치) ·
      AUC(Area Under Curve, 두 집단을 얼마나 잘 가르는지 0.5=무작위 1.0=완벽).

사용:
  uv run python scripts/eda15_viewpoint.py eval          # eval 216장 (보정용 정답 기준)
  uv run python scripts/eda15_viewpoint.py train         # train 128 데이터셋
산출: results/eval_viewpoint_frames.csv, results/train_viewpoint_frames.csv,
      results/train_viewpoint.csv (데이터셋 1행)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_eval"))

RESULTS = ROOT / "results"
EVAL_IMAGES = ROOT / "open" / "data" / "eval" / "images"
TRAIN_ROOT = ROOT / "open" / "data" / "train"

# docs/10 확정 경계: sample_000000~000153 = 셋업 A, 000154~000215 = 셋업 B
SETUP_B_START = 154

# 모든 프레임을 이 크기로 맞춰서 잰다 (원본 해상도 3종의 영향을 제거).
# 640x480(4:3)이 train 128개 중 123개 · eval 전량이므로 이를 기준으로 삼는다.
STD_W, STD_H = 320, 240


# ---------------------------------------------------------------- (A) 기하 신호


def geometric_features(rgb: np.ndarray) -> dict[str, float]:
    """모델 없이 픽셀 통계만으로 '수평 경계선 + 위아래 이질성'을 잰다.

    rgb: (H, W, 3) uint8, STD_H x STD_W로 이미 리사이즈된 것.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = gray.shape

    # --- 1) 행별 수평 가장자리 에너지 -------------------------------------
    # 세로 방향 소벨 = 가로로 뻗은 경계선(책상 먼 쪽 모서리 등)에 크게 반응한다.
    sob_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
    sob_y = cv2.GaussianBlur(sob_y.reshape(-1, 1), (1, 9), 0).ravel()
    # 화면 맨 위/아래 10%는 레터박스·비네팅 영향이 있어 후보에서 제외
    lo, hi = int(0.10 * H), int(0.90 * H)
    band = sob_y[lo:hi]
    peak_i = int(np.argmax(band)) + lo
    med = float(np.median(sob_y)) + 1e-6
    horiz_edge_row = peak_i / H                      # 0=맨 위, 1=맨 아래
    horiz_edge_strength = float(sob_y[peak_i]) / med  # 중앙값 대비 몇 배로 튀는가

    # --- 2) 최적 가로 분할 (위/아래가 서로 다른 장면인가) -----------------
    # 각 후보 행 y에서 위 영역과 아래 영역의 '색+질감' 평균 차이를 재고,
    # 그 차이가 최대가 되는 y를 찾는다. top-down은 위아래가 같은 작업대라 차이가 작다.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # 지역 질감(3x3 표준편차)을 4번째 채널로 붙인다
    tex = cv2.GaussianBlur(gray * gray, (5, 5), 0) - cv2.GaussianBlur(gray, (5, 5), 0) ** 2
    tex = np.sqrt(np.clip(tex, 0, None))
    feat = np.concatenate([lab, tex[..., None]], axis=2)   # (H, W, 4)
    rows = feat.mean(axis=1)                                # (H, 4) 행별 평균 벡터
    csum = np.cumsum(rows, axis=0)
    total = csum[-1]
    ys = np.arange(lo, hi)
    up_mean = csum[ys - 1] / ys[:, None]                    # 위 영역 평균
    dn_mean = (total - csum[ys - 1]) / (H - ys)[:, None]    # 아래 영역 평균
    # 채널별 전역 표준편차로 나눠 스케일을 맞춘 뒤 거리
    scale = rows.std(axis=0) + 1e-6
    sep = np.linalg.norm((up_mean - dn_mean) / scale, axis=1)
    best = int(np.argmax(sep))
    split_row = ys[best] / H
    split_score = float(sep[best])

    # --- 3) 위쪽 25% vs 아래쪽 25%의 질감비 -------------------------------
    # low-angle은 위쪽에 배경(의자·타일·벽)이 있어 질감이 복잡하다.
    top_tex = float(tex[: H // 4].mean())
    bot_tex = float(tex[-H // 4 :].mean())
    tex_ratio_top_bot = top_tex / (bot_tex + 1e-6)

    # --- 4) 긴 수평 직선 성분 (책상 먼 쪽 모서리) --------------------------
    edges = cv2.Canny(rgb, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=int(0.35 * W), maxLineGap=12)
    n_horiz_upper, best_slope = 0, np.nan
    best_len = 0.0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            ang = (ang + 90) % 180 - 90          # -90..90
            if abs(ang) <= 20:                    # 거의 수평인 선분만
                ymid = (y1 + y2) / 2
                if ymid < 0.7 * H:                # 화면 위쪽 70% 안
                    n_horiz_upper += 1
                    ln = float(np.hypot(x2 - x1, y2 - y1))
                    if ln > best_len:
                        best_len, best_slope = ln, float(ang)

    return {
        "horiz_edge_row": horiz_edge_row,
        "horiz_edge_strength": horiz_edge_strength,
        "split_row": split_row,
        "split_score": split_score,
        "tex_ratio_top_bot": tex_ratio_top_bot,
        "n_horiz_upper": float(n_horiz_upper),
        "table_edge_slope_deg": best_slope,
    }


# ---------------------------------------------------------------- (B) 깊이 신호


class DepthProbe:
    """Depth Anything V2 Small — 상대 시차(가까울수록 큰 값)를 배치로 뽑는다."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        name = "depth-anything/Depth-Anything-V2-Small-hf"
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, rgbs: list[np.ndarray]) -> list[np.ndarray]:
        ims = [Image.fromarray(r) for r in rgbs]
        inp = self.proc(images=ims, return_tensors="pt").to(self.device)
        out = self.model(**inp).predicted_depth          # (B, h, w) 시차
        out = torch.nn.functional.interpolate(
            out[:, None], size=(STD_H, STD_W), mode="bilinear", align_corners=False
        )[:, 0]
        return [d.float().cpu().numpy() for d in out]


def depth_features(disp: np.ndarray) -> dict[str, float]:
    """시차 맵에서 '세로 방향 깊이 기울기'와 '깊이 폭'을 뽑는다.

    시차는 절대 스케일이 없으므로 프레임 내부에서 정규화해 상대량만 쓴다.
    """
    H, W = disp.shape
    p5, p50, p95 = np.percentile(disp, [5, 50, 95])
    span = float((p95 - p5) / (p50 + 1e-6))    # 장면 안에 깊이 변화가 얼마나 있나

    # 0..1로 정규화한 뒤 행별 중앙값의 기울기를 잰다.
    d = (disp - p5) / (p95 - p5 + 1e-6)
    row_med = np.median(d, axis=1)
    y = np.linspace(0.0, 1.0, H)
    # 위/아래 10%는 레터박스 영향 배제
    m = (y >= 0.10) & (y <= 0.90)
    slope = float(np.polyfit(y[m], row_med[m], 1)[0])   # +면 아래가 가까움(=low-angle)

    # 위 25% vs 아래 25% 시차 차이 (기울기의 비모수 버전)
    top = float(np.median(d[: H // 4]))
    bot = float(np.median(d[-H // 4 :]))

    # --- 작업대 너머가 보이는가 (핵심) ------------------------------------
    # 화면 아래 1/3은 어느 시점이든 '가까운 작업면'이다. 그 시차 분포를 기준으로
    # 삼아, 그보다 **뚜렷하게 먼** 픽셀이 얼마나 되는지를 센다. top-down은 작업면
    # 하나뿐이라 거의 0에 가깝고, low-angle은 작업대 너머 배경이 잡혀 크게 나온다.
    near = disp[-H // 3 :]
    n_med = float(np.median(near))
    n_mad = float(np.median(np.abs(near - n_med))) + 1e-6
    far_mask = disp < (n_med - 4.0 * n_mad)      # 시차가 작다 = 멀다
    depth_bg_frac = float(far_mask.mean())
    # 그 '먼' 픽셀들이 화면 어느 높이에 몰려 있나 (배경이면 위쪽에 몰린다)
    yy = np.repeat(np.linspace(0, 1, H)[:, None], W, axis=1)
    depth_bg_row = float(yy[far_mask].mean()) if far_mask.any() else np.nan

    # 행별 깊이의 최대 불연속 (작업대 먼 쪽 모서리에서 깊이가 끊긴다)
    rm = cv2.GaussianBlur(np.median(disp, axis=1).reshape(-1, 1), (1, 9), 0).ravel()
    drop = np.diff(rm)[int(0.10 * H) : int(0.90 * H)]
    depth_break = float(np.max(drop) / (np.std(rm) + 1e-6)) if drop.size else np.nan

    return {
        "depth_slope": slope,
        "depth_span": span,
        "depth_top_bot_gap": bot - top,
        "depth_bg_frac": depth_bg_frac,
        "depth_bg_row": depth_bg_row,
        "depth_break": depth_break,
    }


# ---------------------------------------------------------------- 프레임 수집


def to_std(rgb: np.ndarray) -> np.ndarray:
    """4:3으로 중앙 크롭한 뒤 STD 크기로 축소.

    train에는 4:3(640x480, 123개)과 16:9(1280x720·1920x1080, 5개)가 섞여 있다.
    그냥 늘려 맞추면 **기울기·각도가 왜곡**되므로(16:9를 4:3에 욱여넣으면 수평선이
    실제보다 완만해진다) 중앙 크롭으로 기하를 보존한다. 레터박스(여백 채우기)를
    쓰지 않는 이유는 여백 경계가 '가짜 수평선'으로 잡히기 때문이다.
    """
    h, w = rgb.shape[:2]
    want = 4 / 3
    if w / h > want:                      # 너무 넓다 → 좌우를 자른다
        nw = int(round(h * want))
        x0 = (w - nw) // 2
        rgb = rgb[:, x0 : x0 + nw]
    elif w / h < want:                    # 너무 높다 → 위아래를 자른다
        nh = int(round(w / want))
        y0 = (h - nh) // 2
        rgb = rgb[y0 : y0 + nh]
    return np.asarray(Image.fromarray(rgb).resize((STD_W, STD_H), Image.BILINEAR))


def load_std(path: Path) -> np.ndarray:
    return to_std(np.asarray(Image.open(path).convert("RGB")))


def eval_frames() -> list[tuple[str, str, np.ndarray]]:
    out = []
    for p in sorted(EVAL_IMAGES.glob("sample_*.png")):
        idx = int(p.stem.split("_")[1])
        setup = "B" if idx >= SETUP_B_START else "A"
        out.append((p.stem, setup, load_std(p)))
    return out


def train_datasets() -> list[str]:
    dss = []
    for user in sorted(TRAIN_ROOT.iterdir()):
        if user.is_dir():
            for ds in sorted(user.iterdir()):
                if (ds / "meta" / "info.json").exists():
                    dss.append(f"{user.name}/{ds.name}")
    return dss


def train_frames(ds: str, k: int) -> list[tuple[int, np.ndarray]]:
    """데이터셋당 k장 — 에피소드를 고르게 훑고 각 에피소드의 중간 프레임을 쓴다.

    (eda05와 같은 표집 규칙. 첫 프레임은 로봇 초기 자세라 장면 다양성이 낮다.)
    """
    from episode_io import read_frames  # noqa: PLC0415

    eps = []
    with (TRAIN_ROOT / ds / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("length", 0) >= 1:
                    eps.append((e["episode_index"], e["length"]))
            except json.JSONDecodeError:
                pass
    if not eps:
        return []
    idx = np.linspace(0, len(eps) - 1, min(k, len(eps))).astype(int)
    out = []
    for i in sorted(set(idx.tolist())):
        ep, ln = eps[i]
        try:
            fr = read_frames(ds, ep, max(ln // 2, 0), 1)[0]
        except Exception:
            continue
        out.append((ep, to_std(fr)))
    return out


# ---------------------------------------------------------------- 실행


def run_batch(items, probe, batch: int = 32) -> pd.DataFrame:
    """items: [(key1, key2, rgb), ...] → 프레임 1행 DataFrame."""
    rows = []
    for s in range(0, len(items), batch):
        chunk = items[s : s + batch]
        disps = probe([c[-1] for c in chunk])
        for (k1, k2, rgb), disp in zip(chunk, disps):
            r = {"key": k1, "group": k2}
            r.update(geometric_features(rgb))
            r.update(depth_features(disp))
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["eval", "train"])
    ap.add_argument("--frames-per-ds", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    probe = DepthProbe("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    if args.target == "eval":
        items = [(k, g, r) for k, g, r in eval_frames()]
        df = run_batch(items, probe, args.batch)
        df = df.rename(columns={"key": "sample_id", "group": "setup"})
        out = RESULTS / "eval_viewpoint_frames.csv"
        df.to_csv(out, index=False)
        print(f"[eval] {len(df)}장 → {out}  ({time.time() - t0:.0f}s)")
        return

    dss = train_datasets()
    print(f"[train] 데이터셋 {len(dss)}개, 데이터셋당 {args.frames_per_ds}장", flush=True)
    all_rows = []
    for n, ds in enumerate(dss, 1):
        frames = train_frames(ds, args.frames_per_ds)
        if not frames:
            print(f"  !! {ds}: 프레임 0장", flush=True)
            continue
        items = [(ds, str(ep), rgb) for ep, rgb in frames]
        d = run_batch(items, probe, args.batch)
        all_rows.append(d)
        print(f"  [{n}/{len(dss)}] {ds}: {len(d)}장  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(all_rows, ignore_index=True).rename(
        columns={"key": "dataset", "group": "episode_index"}
    )
    out = RESULTS / "train_viewpoint_frames.csv"
    df.to_csv(out, index=False)
    print(f"[train] {len(df)}장 / {df.dataset.nunique()}개 데이터셋 → {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
