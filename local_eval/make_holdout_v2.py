"""홀드아웃 v2 생성 (docs/14 확정 절차, 룰북 §2).

- in-domain: T의 holdout_t488.json indomain 328을 그대로 보존하되,
  unseen으로 승격되는 frk2/so100large의 2샘플만 제거 → 326.
  (같은 데이터셋이 in-domain과 unseen에 동시에 있으면 "학습 제외" 규칙과 모순)
- unseen_cousin: aractingi/push_cube_offline_data(셋업A) + frk2/so100large(셋업B) 전량.
- unseen_general: CSCSXX/pick_place_cube_1.18 + ZGGZZG/so100_drop0 전량
  (티어 합계 80~120 합의 범위를 만족하는 소형 조합).
- 신규 unseen 샘플: 에피소드당 1클립, start = seed 0 균등 난수 [0, length-16].
- T-unseen 8개 중 나머지 5개(AndrejOrsula/Gano007/LemonadeDai/jpata/therarelab)는 학습 복귀.

출력: local_eval/holdout_v2.json (holdout_t488.json과 동일 샘플 스키마)
결정론: seed 0. 확정 후 대회 끝까지 불변 (룰북 §2: 수치 혼용 금지).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_ROOT = HERE.parent / "open" / "data" / "train"
T488_PATH = HERE / "holdout_t488.json"
OUT_PATH = HERE / "holdout_v2.json"

MIN_LEN = 16
WINDOW = 16
SEED = 0

UNSEEN_TIERS = {
    "unseen_cousin": ["aractingi/push_cube_offline_data", "frk2/so100large"],
    "unseen_general": ["CSCSXX/pick_place_cube_1.18", "ZGGZZG/so100_drop0"],
}
RETURNED_TO_TRAIN = [
    "AndrejOrsula/lerobot_double_ball_stacking_random",
    "Gano007/so100_medic",
    "LemonadeDai/so100_coca",
    "jpata/so100_pick_place_tangerine",
    "therarelab/so100_pick_place_2",
]


def eligible_episodes(ds_id: str) -> list[dict]:
    path = TRAIN_ROOT / ds_id / "meta" / "episodes.jsonl"
    eps = []
    for line in path.open():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("length", 0) >= MIN_LEN:
            eps.append(e)
    return sorted(eps, key=lambda e: e["episode_index"])


def sample_id(tier: str, ds_id: str, ep: int, start: int) -> str:
    user, name = ds_id.split("/")
    return f"{tier}__{user}__{name}__ep{ep:06d}__t{start:04d}"


def main() -> None:
    rng = random.Random(SEED)
    t488 = json.loads(T488_PATH.read_text())

    all_unseen = [ds for tier in UNSEEN_TIERS.values() for ds in tier]
    in_domain = [
        s for s in t488["samples"]
        if s["tier"] == "indomain" and s["dataset"] not in all_unseen
    ]
    dropped = [s["sample_id"] for s in t488["samples"]
               if s["tier"] == "indomain" and s["dataset"] in all_unseen]

    samples = list(in_domain)
    tier_counts = {"indomain": len(in_domain)}
    for tier, ds_ids in UNSEEN_TIERS.items():
        n_tier = 0
        for ds_id in sorted(ds_ids):
            for e in eligible_episodes(ds_id):
                start = rng.randint(0, e["length"] - WINDOW)
                samples.append({
                    "tier": tier,
                    "dataset": ds_id,
                    "episode_index": e["episode_index"],
                    "start": start,
                    "length": e["length"],
                    "sample_id": sample_id(tier, ds_id, e["episode_index"], start),
                })
                n_tier += 1
        tier_counts[tier] = n_tier

    out = {
        "meta": {
            "source": "docs/14 확정 절차 — T488 indomain 보존 + unseen 교체 (make_holdout_v2.py, seed 0)",
            "base": "local_eval/holdout_t488.json (T, seed 42)",
            "indomain_deviation": f"frk2/so100large 승격으로 indomain 2샘플 제거(328→326): {dropped}",
            "returned_to_train": RETURNED_TO_TRAIN,
            "train_exclusion": all_unseen,  # 룰북 §2: 학습 제외, E-final만 복귀
            "counts": tier_counts | {"total": len(samples)},
        },
        "unseen_datasets": sorted(all_unseen),
        "samples": samples,
    }
    OUT_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["meta"], indent=2, ensure_ascii=False))
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
