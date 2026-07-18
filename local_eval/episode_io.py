"""train 에피소드에서 (프레임, 행동) 윈도우를 읽는 헬퍼."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import pandas as pd

TRAIN_ROOT = Path(__file__).resolve().parent.parent / "open" / "data" / "train"


@lru_cache(maxsize=256)
def camera_key(dataset: str) -> str:
    info = json.loads((TRAIN_ROOT / dataset / "meta" / "info.json").read_text())
    keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    assert len(keys) == 1, f"{dataset}: 카메라 키 {keys}"
    return keys[0]


def episode_paths(dataset: str, episode_index: int) -> tuple[Path, Path]:
    ep = f"episode_{episode_index:06d}"
    video = TRAIN_ROOT / dataset / "videos" / "chunk-000" / camera_key(dataset) / f"{ep}.mp4"
    parquet = TRAIN_ROOT / dataset / "data" / "chunk-000" / f"{ep}.parquet"
    return video, parquet


def read_frames(dataset: str, episode_index: int, start: int, length: int) -> np.ndarray:
    """(length, H, W, C) uint8 RGB. start부터 length개 프레임."""
    video_path, _ = episode_paths(dataset, episode_index)
    frames = []
    with av.open(str(video_path)) as container:
        for i, frame in enumerate(container.decode(container.streams.video[0])):
            if i < start:
                continue
            if i >= start + length:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != length:
        raise ValueError(f"{video_path}: {start}+{length} 요청, {len(frames)}개만 디코드")
    return np.stack(frames)


def read_actions(dataset: str, episode_index: int, start: int, length: int) -> np.ndarray:
    """(length, 6) float32, deg 단위 (비정규화)."""
    _, parquet_path = episode_paths(dataset, episode_index)
    df = pd.read_parquet(parquet_path, columns=["action"])
    a = np.stack(df["action"].to_numpy())[start : start + length]
    if a.shape[0] != length:
        raise ValueError(f"{parquet_path}: 행동 {a.shape[0]}개 < {length}")
    return a.astype(np.float32)
