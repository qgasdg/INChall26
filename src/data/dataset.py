"""SO100 윈도우 데이터셋 — IRASim (이식: impl/w2/so100_data.py → INChall26 유틸 배선).

impl/w2 의 로직(윈도우·제외·커리큘럼·정규화·정렬)을 INChall26 자체 데이터 생태계에 배선한 것:
  - IO     = local_eval/episode_io.read_frames/read_actions            (impl/w2 gt_clips 대체)
  - 열거   = open/data/train/<user>/<dataset>/meta/episodes.jsonl       (impl/w2 registry 대체)
  - 제외   = local_eval/holdout.json (unseen_datasets·in_domain·unseen_scene) + docs/17 hard_exclude
  - 정규화·해상도·정렬 = src/data/transforms.py, src/models/action_adapter.py

윈도우(docs/03·04): frames[t..t+15] ↔ actions[t..t+15] 동일 인덱스(action[t]≈state[t+1]). stride 로 다중 시작.
두 클래스:
  - SO100ClipDataset   : 영상 디코드 → 픽셀. **지금 바로 동작**(open/data/train 존재 시).
  - SO100LatentDataset : SDXL VAE latent 캐시 슬라이스(pre_encode, 처리량↑). 캐시 필요 → build_latent_cache 이식 후 활성.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

from src.data import transforms as T
from src.models.action_adapter import adapt_action_seq

_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = _ROOT / "open" / "data" / "train"
HOLDOUT = _ROOT / "local_eval" / "holdout.json"
SEQ_LEN = 16


@dataclass(frozen=True)
class ClipRef:
    """윈도우 1개 참조: ds_id="user/dataset", 에피소드, 시작 프레임, 커리큘럼 가중치."""
    ds_id: str
    episode_index: int
    start: int
    weight: float = 1.0


def _iter_datasets() -> list[str]:
    """open/data/train/<user>/<dataset> 중 meta/info.json 있는 것 → "user/dataset" 목록(정렬)."""
    return sorted(f"{p.parent.name}/{p.name}" for p in TRAIN_ROOT.glob("*/*")
                  if (p / "meta" / "info.json").exists())


def _episode_lengths(ds_id: str) -> dict[int, int]:
    """meta/episodes.jsonl → {episode_index: length} (parquet 없이 빠름)."""
    out: dict[int, int] = {}
    f = TRAIN_ROOT / ds_id / "meta" / "episodes.jsonl"
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            out[int(e["episode_index"])] = int(e["length"])
    return out


def _load_exclusions(cfg: dict) -> tuple[set, set]:
    """holdout.json + docs/17 hard_exclude → (제외 데이터셋 set, 제외 (ds,ep) set)."""
    ex_ds: set = set()
    ex_ep: set = set()
    if HOLDOUT.exists():
        h = json.loads(HOLDOUT.read_text(encoding="utf-8"))
        ex_ds |= set(h.get("unseen_datasets", []))
        for e in list(h.get("in_domain", [])) + list(h.get("unseen_scene", [])):
            ex_ep.add((e["dataset"], int(e["episode_index"])))
    hard = (cfg or {}).get("curriculum", {}).get("hard_exclude", {}) or {}
    ex_ds |= set(hard.get("datasets", [])) | set(hard.get("holdout_v2_unseen", []))
    return ex_ds, ex_ep


def _weight_for(ds_id: str, cfg: dict) -> float:
    """docs/17 커리큘럼 weight_down(glob 패턴, 예 "Chojins/*":0.33) → 데이터셋 가중치(기본 1.0)."""
    wd = (cfg or {}).get("curriculum", {}).get("weight_down", {}) or {}
    for pat, w in wd.items():
        if fnmatch(ds_id, pat):
            return float(w)
    return 1.0


def build_window_index(cfg: dict | None = None) -> list[ClipRef]:
    """열거 + 제외 + stride 윈도우 + 커리큘럼 가중 → ClipRef 목록(결정론 정렬).

    이식(impl/w2 build_clip_index): registry→episodes.jsonl, excluded→holdout.json 로 대체.
    """
    cfg = cfg or {}
    seq_len = int(cfg.get("task", {}).get("frames", SEQ_LEN))
    stride = int(cfg.get("data", {}).get("stride", 8))
    ex_ds, ex_ep = _load_exclusions(cfg)

    clips: list[ClipRef] = []
    for ds_id in _iter_datasets():
        if ds_id in ex_ds:
            continue
        w = _weight_for(ds_id, cfg)
        for ep, length in _episode_lengths(ds_id).items():
            if length < seq_len or (ds_id, ep) in ex_ep:
                continue
            for st in range(0, length - seq_len + 1, stride):
                clips.append(ClipRef(ds_id, ep, st, w))
    clips.sort(key=lambda c: (c.ds_id, c.episode_index, c.start))
    return clips


def build_sampler(clips: list[ClipRef], cfg: dict | None = None):
    """커리큘럼 가중 WeightedRandomSampler(replacement) — docs/17 weight_down 반영."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    w = torch.tensor([c.weight for c in clips], dtype=torch.double)
    return WeightedRandomSampler(w, len(clips), replacement=True)


class SO100ClipDataset:
    """on-the-fly: 영상 디코드 → 320×512 letterbox 픽셀 + z-score 액션. (open/data/train 존재 시 즉시 동작)

    반환:
      video(16,3,gh,gw)[-1,1] · action(16,6) z-score · action_cond(15,6) · cond_frame(3,gh,gw) · meta
    """

    def __init__(self, clips: list[ClipRef], cfg: dict | None = None):
        self.clips = clips
        self.cfg = cfg or {}
        self.seq_len = int(self.cfg.get("task", {}).get("frames", SEQ_LEN))
        self.mode = self.cfg.get("model", {}).get("resolution_mode", T.RES_A)
        self.align_mode = self.cfg.get("data", {}).get("align_mode", "shifted")
        self.mean, self.std = T.load_action_stats()

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> dict:
        import torch

        from local_eval import episode_io   # 지연 import(av·pandas) — 열거는 이거 없이도 됨

        c = self.clips[i]
        frames = episode_io.read_frames(c.ds_id, c.episode_index, c.start, self.seq_len)      # (16,H,W,3) uint8
        actions_deg = episode_io.read_actions(c.ds_id, c.episode_index, c.start, self.seq_len)  # (16,6) deg
        actions = T.normalize_action(actions_deg, self.mean, self.std)                          # (16,6) z-score
        gen = T.final_to_gen_target(frames, self.mode)                                          # (16,gh,gw,3) uint8
        video = T.to_model_input(gen)                                                           # (16,3,gh,gw) [-1,1]
        cond = np.ascontiguousarray(adapt_action_seq(actions, self.seq_len, self.align_mode))   # (15,6)
        return {
            "video": video,
            "action": torch.from_numpy(actions),
            "action_cond": torch.from_numpy(cond),
            "cond_frame": video[0].clone(),
            "meta": {"ds_id": c.ds_id, "episode_index": c.episode_index, "start": c.start},
        }


class SO100LatentDataset:
    """pre-encode: SDXL VAE latent npz 캐시 슬라이스(VAE 병목 제거).

    ★캐시 필요 — build_latent_cache(impl/w2) 이식 후 활성. 반환:
      latent(16,4,h,w) · action(16,6) · action_cond(15,6) · cond_latent(4,h,w) · meta
    """

    def __init__(self, clips: list[ClipRef], cache_dir: Path | str, cfg: dict | None = None):
        self.clips = clips
        self.cache_dir = Path(cache_dir)
        self.cfg = cfg or {}
        self.seq_len = int(self.cfg.get("task", {}).get("frames", SEQ_LEN))
        self.align_mode = self.cfg.get("data", {}).get("align_mode", "shifted")

    def __len__(self) -> int:
        return len(self.clips)

    def _key(self, ds_id: str) -> str:
        return ds_id.replace("/", "__")

    def __getitem__(self, i: int) -> dict:
        import torch

        c = self.clips[i]
        npz = self.cache_dir / "latents" / self._key(c.ds_id) / f"{c.episode_index:06d}.npz"
        with np.load(npz) as z:
            lat = z["latents"][c.start:c.start + self.seq_len]                          # (16,4,h,w) fp16
            act = z["actions_norm"][c.start:c.start + self.seq_len].astype(np.float32)  # (16,6) z-score
        latent = torch.from_numpy(lat.astype(np.float32))
        cond = np.ascontiguousarray(adapt_action_seq(act, self.seq_len, self.align_mode))
        return {
            "latent": latent,
            "action": torch.from_numpy(act),
            "action_cond": torch.from_numpy(cond),
            "cond_latent": latent[0].clone(),
            "meta": {"ds_id": c.ds_id, "episode_index": c.episode_index, "start": c.start},
        }
