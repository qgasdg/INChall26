"""전처리 — 320×512 킷 letterbox + z-score (이식: impl/w2/resolution.py · action_norm.py).

설계 결정(docs/03·04 S1):
  - 공통 전처리 = 킷과 동일 **320×512 letterbox**(canvas_aligned) → 모델이 이 캔버스를 직접 생성(왕복 손실 제거).
    킷 기하: 640×480 기준 scale=min(320/480,512/640)=0.6667 → 콘텐츠 320×427, 좌우 pad 42/43(검은 바).
  - 픽셀 [-1,1](생성 입력). action은 so100 z-score(open/data/train/so100_action_statistics.json, 전역 mean/std).
  - 증강: 수평 flip 금지(액션-기하 불일치)·hue 금지, 약한 밝기/대비만.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 킷 상수 (open/submission_kit feature_csv_utils.preprocess_video, pad=True 와 정합)
KIT_H, KIT_W = 320, 512
FINAL_H, FINAL_W = 480, 640          # 킷 기준 입력 해상도(impl/w2 정합)
RES_A, RES_B = "canvas_aligned", "native_restore"
GEN_SIZE = {RES_A: (320, 512), RES_B: (288, 512)}   # (H, W) 모델 생성 해상도

_STATS = Path(__file__).resolve().parents[2] / "open" / "data" / "train" / "so100_action_statistics.json"


def _interp(frames: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """(T,H,W,3) uint8 → (T,out_h,out_w,3) uint8, bilinear(킷과 동일 규약)."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    t = (t.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return t.permute(0, 2, 3, 1).contiguous().numpy()


def _kit_geom() -> tuple[int, int, int, int]:
    """640×480 → 320×512 letterbox 콘텐츠 영역/pad (impl/w2 KitLetterboxGeom.for_input 고정 기하)."""
    scale = min(KIT_H / FINAL_H, KIT_W / FINAL_W)
    ch, cw = max(1, round(FINAL_H * scale)), max(1, round(FINAL_W * scale))
    return ch, cw, (KIT_H - ch) // 2, (KIT_W - cw) // 2


def kit_letterbox(frames: np.ndarray) -> np.ndarray:
    """(T,H,W,3) uint8 → (T,320,512,3) uint8 zero-pad letterbox(킷 preprocess_video pad=True 재현)."""
    ch, cw, pt, pl = _kit_geom()
    content = _interp(frames, (ch, cw))
    out = np.zeros((frames.shape[0], KIT_H, KIT_W, 3), np.uint8)
    out[:, pt:pt + ch, pl:pl + cw] = content
    return out


def final_to_gen_target(frames: np.ndarray, mode: str = RES_A) -> np.ndarray:
    """GT 프레임(T,H,W,3) uint8 → 생성 해상도 픽셀공간.
    canvas_aligned: 320×512 킷 letterbox / native_restore: 288×512 resize(aspect 무시)."""
    if mode == RES_A:
        return kit_letterbox(frames)
    return _interp(frames, GEN_SIZE[RES_B])


def to_model_input(gen_frames: np.ndarray):
    """(T,gh,gw,3) uint8 → (T,3,gh,gw) float [-1,1] torch."""
    import torch

    return torch.from_numpy(gen_frames).float().div(127.5).sub(1.0).permute(0, 3, 1, 2)


def load_action_stats(path: Path | str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """so100_action_statistics.json → (mean(6,), std(6,)) float32.
    ★킷 채점기와 동일한 전역 mean/std 사용(코드경로 검증 필수, docs/17 §2.3)."""
    d = json.loads(Path(path or _STATS).read_text(encoding="utf-8"))
    return np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32)


def normalize_action(action_deg, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(...,6) deg → z-score ((a-mean)/std)."""
    return (np.asarray(action_deg, np.float32) - mean) / std


def train_augment(frames, cfg: dict):
    """학습 증강 — flip/hue 금지, 약한 밝기/대비만. TODO(선택 — 기본 로딩엔 불필요)."""
    raise NotImplementedError("train_augment: 밝기/대비 지터만 구현(flip/hue 금지)")
