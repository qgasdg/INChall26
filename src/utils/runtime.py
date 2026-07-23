"""런타임 유틸 — 결정론 seed 고정 · logger · device/프로파일 (동작).

결정론(docs/04 S5): 추론 η=0 + seed 고정으로 재현. 학습도 재현 위해 seed 고정 헬퍼 제공.
"""
from __future__ import annotations

import logging
import os
import sys


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """python/numpy/torch 전역 seed 고정. deterministic=True면 cudnn 결정론 모드."""
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_logger(name: str = "inchall26", level: int = logging.INFO) -> logging.Logger:
    """단일 stdout 핸들러 logger(중복 핸들러 방지)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def pick_device(prefer: str = "auto") -> str:
    """학습/추론 device 선택. Mac은 로직 검증(mps/cpu), 서버는 cuda(docs/07 2-머신)."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
