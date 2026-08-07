"""액션 12차원(절대 6 + Δ 6) 데이터모듈.

킷의 `SO100DataModule` 은 6차원만 낸다. 킷 코드는 건드리지 않고 여기서 감싼다.

Δ 정규화: 데이터셋이 내는 act 는 이미 절대값을 전역 std 로 나눈 z 다.
a[t]-a[t-1] = (z[t]-z[t-1])*action_std 이므로, Δ 전용 std 로 다시 나눈다.
`make_delta_stats.py` 가 만든 so100_delta_statistics.json 을 쓴다.
t=0 의 Δ 는 0.

추론(`gen_step0.py:to_12dim`)과 **같은 식**이어야 한다 — 다르면 학습·추론 분포가 어긋난다.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ldwma.lightning.data_modules.lerobot_so100 import SO100DataModule


class Act12Dataset(Dataset):
    """[T,6] 정규화 절대값 → [T,12] (절대 6 + 정규화된 Δ 6)."""

    def __init__(self, base: Dataset, action_std: torch.Tensor, delta_std: torch.Tensor):
        self.base = base
        self.scale = (action_std / delta_std).float()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        act = item["act"]                                   # [T, 6]
        dz = torch.diff(act, dim=0, prepend=act[:1])
        item["act"] = torch.cat([act, dz * self.scale], dim=-1)
        return item

    def __getattr__(self, name):
        # dataset_paths 등 원본 속성 접근을 그대로 넘긴다
        return getattr(self.__dict__["base"], name)


class SO100Act12DataModule(SO100DataModule):
    def __init__(self, *args, delta_stats_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_stats_path = delta_stats_path

    def _delta_std(self) -> torch.Tensor:
        p = self.delta_stats_path or (Path(self.root) / "so100_delta_statistics.json")
        stats = json.loads(Path(p).read_text())
        return torch.tensor(stats["std"], dtype=torch.float32)

    def setup(self, stage=None):
        super().setup(stage)
        action_std = self.train_dataset.action_std
        if action_std is None:
            raise ValueError("normalize_actions: True 여야 한다 — 12차원 Δ 스케일이 action_std 에 의존한다.")
        delta_std = self._delta_std()
        self.train_dataset = Act12Dataset(self.train_dataset, action_std, delta_std)
        self.val_dataset = Act12Dataset(self.val_dataset, action_std, delta_std)
