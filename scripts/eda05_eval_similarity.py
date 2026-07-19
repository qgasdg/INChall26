"""train 128개 데이터셋 × eval 유사도 랭킹 (48h 계획 Phase 2).

목적: "어떤 train 데이터가 eval(평가 데이터)과 닮았는가"를 수치로 세워 큐레이션(학습 데이터
      선별) 우선순위의 근거를 만든다. [10_조사_eval도메인](../docs/10_조사_eval도메인.md)의
      확장판.

10과의 차이:
  - 대표 프레임을 데이터셋당 4장 → 24장으로 확대. 10이 스스로 적어둔 한계
    ("한 데이터셋 안에 여러 장면이 있으면 놓칠 수 있음")를 해소한다.
    에피소드를 고르게 훑고, 각 에피소드의 중간 지점 프레임을 쓴다(첫 프레임은
    로봇이 정지한 초기 자세라 장면 다양성이 낮음).
  - eval을 셋업 A(검은 매트형, sample 000~153)/B(나무 책상형, 154~215)로 나눠 따로 집계.
    사촌(닮은 데이터셋)이 어느 셋업 쪽인지가 큐레이션에 필요하기 때문.
  - 데이터셋 1행짜리 랭킹 CSV를 남겨 이후 단계가 그대로 조인해 쓸 수 있게 한다.

용어: DINO(자기지도학습으로 훈련된 이미지 특징 추출 모델) · CLS 토큰(이미지 전체를 요약하는
      벡터) · 코사인 유사도(두 벡터가 이루는 각도로 재는 닮음 정도, 1이면 동일 방향).

사용: uv run python scripts/eda05_eval_similarity.py [--frames-per-ds 24]
산출: results/dataset_eval_similarity.csv, results/eval_similarity_probe.npz(특징 캐시)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_eval"))

from episode_io import TRAIN_ROOT, read_frames  # noqa: E402
from kit_bridge import EVAL_ROOT, KitScorer, get_device  # noqa: E402

# docs/10에서 확정된 eval 셋업 경계 (sample_000000~000153 = A, 000154~000215 = B)
SETUP_B_START = 154


def episode_plan(ds: str, k: int) -> list[tuple[int, int]]:
    """(episode_index, start_frame) k개 — 에피소드를 고르게 훑고 각 중간 지점을 취한다."""
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
    return [(eps[i][0], max(eps[i][1] // 2, 0)) for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-per-ds", type=int, default=24)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device={device}", flush=True)
    scorer = KitScorer(device)

    def features(imgs: list[np.ndarray]) -> np.ndarray:
        """킷과 동일한 전처리(320x512 레터박스) 후 DINO CLS 벡터 (N,384)."""
        out = []
        for i in range(0, len(imgs), args.batch):
            batch = imgs[i:i + args.batch]
            vids = torch.stack([scorer.to_eval_video(np.stack([im])) for im in batch])
            out.append(scorer.dino_features(vids)[:, 0].cpu())
        return torch.cat(out).numpy()

    # --- eval 216장 ---
    eval_paths = sorted((EVAL_ROOT / "images").glob("*.png"))
    eval_ids = [p.stem for p in eval_paths]
    eval_feat = features([np.asarray(Image.open(p).convert("RGB")) for p in eval_paths])
    setup = np.array([("B" if int(s.split("_")[1]) >= SETUP_B_START else "A") for s in eval_ids])
    print(f"eval {len(eval_ids)}장 (셋업 A {(setup=='A').sum()} / B {(setup=='B').sum()})", flush=True)

    # --- train 대표 프레임 ---
    datasets = sorted({"/".join(p.parts[-4:-2]) for p in TRAIN_ROOT.glob("*/*/meta/info.json")})
    rows, feats = [], []
    for n, ds in enumerate(datasets, 1):
        imgs, keys = [], []
        for ep, start in episode_plan(ds, args.frames_per_ds):
            try:
                imgs.append(read_frames(ds, ep, start, 1)[0])
                keys.append((ds, ep, start))
            except Exception as e:
                print(f"    skip {ds} ep{ep}: {type(e).__name__}", flush=True)
        if not imgs:
            print(f"    !! {ds}: 프레임 0장", flush=True)
            continue
        feats.append(features(imgs))
        rows.extend(keys)
        if n % 20 == 0 or n == len(datasets):
            print(f"  train {n}/{len(datasets)} 데이터셋", flush=True)

    train_feat = np.concatenate(feats)
    train_ds = np.array([r[0] for r in rows])
    print(f"train 대표 프레임 {len(train_feat)}장 / {len(datasets)} 데이터셋", flush=True)

    def norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    sim = norm(train_feat) @ norm(eval_feat).T  # (N_train, 216)

    out = []
    for ds in datasets:
        m = train_ds == ds
        if not m.any():
            continue
        s = sim[m]                       # (프레임 수, 216)
        sa, sb = s[:, setup == "A"], s[:, setup == "B"]
        best_eval = eval_ids[int(s.max(axis=0).argmax())]
        out.append({
            "dataset": ds,
            "n_frames_probed": int(m.sum()),
            "sim_max": round(float(s.max()), 4),
            "sim_p95": round(float(np.percentile(s, 95)), 4),
            "sim_mean": round(float(s.mean()), 4),
            "sim_max_setupA": round(float(sa.max()), 4),
            "sim_mean_setupA": round(float(sa.mean()), 4),
            "sim_max_setupB": round(float(sb.max()), 4),
            "sim_mean_setupB": round(float(sb.mean()), 4),
            # 프레임별 최고 유사도의 중앙값 — "이 데이터셋의 전형적인 장면"이 eval과 얼마나 닮았나
            "sim_perframe_best_median": round(float(np.median(s.max(axis=1))), 4),
            "closer_to": "A" if sa.mean() >= sb.mean() else "B",
            "nearest_eval_sample": best_eval,
        })

    df = pd.DataFrame(out).sort_values("sim_p95", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df.to_csv(ROOT / "results" / "dataset_eval_similarity.csv", index=False)
    np.savez_compressed(ROOT / "results" / "eval_similarity_probe.npz",
                        train_feat=train_feat.astype(np.float16),
                        train_ds=train_ds, eval_feat=eval_feat.astype(np.float16),
                        eval_ids=np.array(eval_ids), setup=setup)

    print("\n=== eval 유사도 상위 15 데이터셋 (p95 기준) ===")
    print(df.head(15)[["rank", "dataset", "sim_max", "sim_p95", "sim_mean",
                       "sim_mean_setupA", "sim_mean_setupB", "closer_to"]].to_string(index=False))
    print("\n=== 하위 10 (큐레이션 제외 후보) ===")
    print(df.tail(10)[["rank", "dataset", "sim_max", "sim_p95", "sim_mean"]].to_string(index=False))
    print(f"\n전역: 최대 {df.sim_max.max():.4f} / p95 중앙값 {df.sim_p95.median():.4f}")
    print("동일장면 판정선 0.85 이상 데이터셋:",
          int((df.sim_max >= 0.85).sum()), "개 (docs/10 결론 = 0)")
    print("→ results/dataset_eval_similarity.csv")


if __name__ == "__main__":
    main()
