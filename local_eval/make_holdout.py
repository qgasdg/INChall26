"""2단 홀드아웃 분할 생성 (04_파이프라인 S0 설계).

- in-domain: 각 데이터셋 에피소드의 5% (최소 1개, 16프레임 이상만)
- unseen-scene: 데이터셋 N개 통째 제외 (중간 규모, 480x640 위주에서 선택)

출력: local_eval/holdout.json
  { "in_domain": [{dataset, episode_index, length}], "unseen_scene": [...], "unseen_datasets": [...] }

결정론: seed 0 고정. 이 분할은 한 번 만들면 대회 끝까지 불변으로 쓴다.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parent.parent / "open" / "data" / "train"
OUT_PATH = Path(__file__).resolve().parent / "holdout.json"

MIN_LEN = 16          # 16프레임 윈도우 최소 길이 (EDA: <16 에피소드 34개 제외)
IN_DOMAIN_FRAC = 0.05
N_UNSEEN_DATASETS = 7  # 6~8 범위의 중앙값


def list_datasets() -> list[tuple[str, list[dict]]]:
    out = []
    for meta in sorted(TRAIN_ROOT.glob("*/*/meta/episodes.jsonl")):
        ds_id = "/".join(meta.parts[-4:-2])
        eps = []
        with meta.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("length", 0) >= MIN_LEN:
                    eps.append({"episode_index": e["episode_index"], "length": e["length"]})
        if eps:
            out.append((ds_id, eps))
    return out


def main() -> None:
    rng = random.Random(0)
    datasets = list_datasets()

    # unseen-scene 후보: 에피소드 수 기준 중간 규모(전체 중앙값 부근)에서 무작위 선택.
    # 너무 큰 데이터셋을 빼면 학습 손실이 크고, 너무 작으면 홀드아웃 통계가 빈약하다.
    sized = sorted(datasets, key=lambda kv: len(kv[1]))
    mid = sized[len(sized) // 4 : 3 * len(sized) // 4]
    unseen_ids = sorted(ds for ds, _ in rng.sample(mid, N_UNSEEN_DATASETS))

    in_domain, unseen = [], []
    for ds_id, eps in datasets:
        if ds_id in unseen_ids:
            unseen.extend({"dataset": ds_id, **e} for e in eps)
            continue
        k = max(1, round(len(eps) * IN_DOMAIN_FRAC))
        for e in rng.sample(eps, k):
            in_domain.append({"dataset": ds_id, **e})

    out = {
        "seed": 0,
        "min_len": MIN_LEN,
        "unseen_datasets": unseen_ids,
        "in_domain": sorted(in_domain, key=lambda x: (x["dataset"], x["episode_index"])),
        "unseen_scene": sorted(unseen, key=lambda x: (x["dataset"], x["episode_index"])),
    }
    OUT_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"in-domain 홀드아웃 에피소드: {len(in_domain)} (전 데이터셋 5%)")
    print(f"unseen-scene 에피소드: {len(unseen)} ({N_UNSEEN_DATASETS}개 데이터셋 통째)")
    print("unseen 데이터셋:", *unseen_ids, sep="\n  ")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
