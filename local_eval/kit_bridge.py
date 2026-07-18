"""submission_kit 코드를 수정 없이 import해 채점 로직을 100% 동일하게 재사용하는 브리지.

킷 파일은 절대 수정하지 않는다 (규정: 제출킷 변경 금지). sys.path 주입으로 import만 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPEN_ROOT = PROJECT_ROOT / "open"
KIT_ROOT = OPEN_ROOT / "submission_kit"
TRAIN_ROOT = OPEN_ROOT / "data" / "train"
EVAL_ROOT = OPEN_ROOT / "data" / "eval"
ACTION_STATS = TRAIN_ROOT / "so100_action_statistics.json"
ACTION_CKPT = KIT_ROOT / "checkpoints" / "action_extractor.ckpt"

sys.path.insert(0, str(KIT_ROOT))

import torch  # noqa: E402
import numpy as np  # noqa: E402

import feature_csv_utils as kit  # noqa: E402  # 킷 원본 모듈

# 킷 make_submission_csv.py의 기본 규격 (변경 금지 값들)
TEMPORAL_LENGTH = 16
TARGET_H, TARGET_W = 320, 512
PAD = True
DINO_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class KitScorer:
    """킷과 동일한 추출기 3종을 로드해 feature/점수를 계산한다."""

    def __init__(self, device: torch.device):
        self.device = device
        self.video_model = kit.load_video_feature_model(device, pretrained=True)
        self.dino_model = kit.load_dino_model(device, DINO_MODEL_NAME, pretrained=True)
        self.dino_size = kit.resolve_dino_image_size(self.dino_model, requested_size=0)
        self.action_model = kit.load_action_extractor(str(ACTION_CKPT), device)
        mean, std = kit.load_action_stats(str(ACTION_STATS))
        self.action_mean, self.action_std = mean, std

    # --- 전처리: 킷과 동일 (uint8 RGB (T,H,W,C) -> letterbox 320x512 uint8) ---
    def to_eval_video(self, frames_uint8: np.ndarray) -> torch.Tensor:
        """(T,H,W,C) uint8 -> 킷 규격 (T,320,512,C) uint8"""
        return kit.to_eval_uint8(frames_uint8, TARGET_H, TARGET_W, PAD)

    # --- feature 추출: 킷 함수 그대로 ---
    def dino_features(self, videos: torch.Tensor) -> torch.Tensor:
        """videos: (B,T,H,W,C) uint8 -> (B,16,384)"""
        return kit.extract_dino_features(videos, self.dino_model, self.device, self.dino_size)

    def video_features(self, videos: torch.Tensor) -> torch.Tensor:
        """videos: (B,T,H,W,C) uint8 -> (B,512)"""
        return kit.extract_video_features(videos, self.video_model, self.device)

    def action_pred(self, videos: torch.Tensor) -> torch.Tensor:
        """videos: (B,T,H,W,C) uint8 -> 추정 행동 (B,16,6) (z-score 공간)"""
        frames = kit.preprocess_images(videos.to(self.device))
        with torch.no_grad():
            return self.action_model(frames).float().cpu()

    def normalize_actions(self, actions_deg: np.ndarray) -> torch.Tensor:
        a = torch.from_numpy(actions_deg.astype(np.float32))
        return (a - self.action_mean) / self.action_std


# --- 거리 계산 (산식) ---
def cosine_distance(u: torch.Tensor, v: torch.Tensor) -> float:
    """1 - cos. u, v: 1-D 벡터"""
    u = u.flatten().double()
    v = v.flatten().double()
    return float(1.0 - torch.dot(u, v) / (u.norm() * v.norm() + 1e-12))


def dino_distance_both(gen: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    """(16,384) 쌍에 대해 (프레임별 cosine 평균, flatten 1회) 두 방식 병산.

    서버 집계 방식이 미공개라 둘 다 계산한다 (04_파이프라인 '잔여 미확정' 참고).
    """
    per_frame = float(np.mean([cosine_distance(gen[t], gt[t]) for t in range(gen.shape[0])]))
    flat = cosine_distance(gen, gt)
    return per_frame, flat


def action_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """킷과 동일: mean over (T, A) of |pred - target| (z-score 공간)"""
    return float(torch.mean(torch.abs(pred - target)))


def total_score(dino: float, video: float, action: float) -> float:
    return 0.3 * dino + 0.3 * video + 0.4 * action
