"""후속 조사 ⑤: eval 216장 vs train 128 데이터셋 장면 대조 + eval 내부 클러스터링.

방법: 킷과 동일한 DINOv2 CLS feature로
  (a) train 각 데이터셋 대표 프레임(에피소드 최대 4개의 첫 프레임) 추출
  (b) eval 216장 추출
  (c) eval×train 코사인 유사도 → eval별 최근접 데이터셋
  (d) eval 내부 유사도 → 연결요소 클러스터링(장면 그룹 수)

출력: local_eval/eval_domain_probe.npz (feature 캐시), 콘솔 리포트
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from episode_io import TRAIN_ROOT, camera_key, read_frames
from kit_bridge import EVAL_ROOT, KitScorer, get_device

OUT = Path(__file__).resolve().parent / "eval_domain_probe.npz"
FRAMES_PER_DS = 4


def dataset_episode_indices(ds: str, k: int) -> list[int]:
    eps = []
    with (TRAIN_ROOT / ds / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("length", 0) >= 1:
                    eps.append(e["episode_index"])
            except json.JSONDecodeError:
                pass
    if not eps:
        return []
    idx = np.linspace(0, len(eps) - 1, min(k, len(eps))).astype(int)
    return [eps[i] for i in idx]


def main() -> None:
    device = get_device("cpu")
    scorer = KitScorer(device)

    def img_features(imgs: list[np.ndarray]) -> np.ndarray:
        """이미지 리스트 -> 킷 전처리(320x512 letterbox) 후 DINO CLS (N,384)"""
        feats = []
        for i in range(0, len(imgs), 16):
            batch = imgs[i : i + 16]
            vids = torch.stack([scorer.to_eval_video(np.stack([im])) for im in batch])  # (B,1,320,512,3)
            feats.append(scorer.dino_features(vids)[:, 0].cpu())
            print(f"    feature {i + len(batch)}/{len(imgs)}")
        return torch.cat(feats).numpy()

    # (a) train 대표 프레임
    datasets = sorted({"/".join(p.parts[-4:-2]) for p in TRAIN_ROOT.glob("*/*/meta/info.json")})
    train_imgs, train_keys = [], []
    for ds in datasets:
        for ep in dataset_episode_indices(ds, FRAMES_PER_DS):
            try:
                train_imgs.append(read_frames(ds, ep, 0, 1)[0])
                train_keys.append(f"{ds}|ep{ep}")
            except Exception as e:  # 손상 에피소드는 건너뜀
                print(f"    skip {ds} ep{ep}: {e}")
    print(f"train 대표 프레임: {len(train_imgs)}장 ({len(datasets)} 데이터셋)")
    train_feat = img_features(train_imgs)

    # (b) eval 216장
    eval_paths = sorted((EVAL_ROOT / "images").glob("*.png"))
    eval_imgs = [np.asarray(Image.open(p).convert("RGB")) for p in eval_paths]
    eval_ids = [p.stem for p in eval_paths]
    eval_feat = img_features(eval_imgs)

    np.savez_compressed(OUT, train_feat=train_feat, train_keys=train_keys,
                        eval_feat=eval_feat, eval_ids=eval_ids)

    # (c) eval -> train 최근접
    def norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sim = norm(eval_feat) @ norm(train_feat).T  # (216, N_train)

    ds_of = [k.split("|")[0] for k in train_keys]
    print("\n=== eval별 최근접 train 데이터셋 (top-1) ===")
    best_idx = sim.argmax(axis=1)
    best_sim = sim.max(axis=1)
    order = np.argsort(-best_sim)
    for i in order[:15]:
        print(f"  {eval_ids[i]}  sim {best_sim[i]:.3f}  <- {train_keys[best_idx[i]]}")
    print(f"  ... (최저) {eval_ids[order[-1]]}  sim {best_sim[order[-1]]:.3f}")

    print("\n최근접 sim 분포: "
          f"p10 {np.percentile(best_sim,10):.3f}  중앙값 {np.median(best_sim):.3f}  p90 {np.percentile(best_sim,90):.3f}")
    for th in (0.85, 0.9, 0.95):
        n = int((best_sim >= th).sum())
        top_ds = {}
        for i in np.where(best_sim >= th)[0]:
            top_ds[ds_of[best_idx[i]]] = top_ds.get(ds_of[best_idx[i]], 0) + 1
        print(f"  sim>={th}: eval {n}장  대상 데이터셋: {sorted(top_ds.items(), key=lambda x:-x[1])[:5]}")

    # (d) eval 내부 클러스터링 (연결요소, threshold 0.85)
    es = norm(eval_feat) @ norm(eval_feat).T
    adj = es >= 0.85
    seen, clusters = set(), []
    for i in range(len(eval_ids)):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j); comp.append(j)
            stack.extend(np.where(adj[j])[0].tolist())
        clusters.append(sorted(comp))
    clusters.sort(key=len, reverse=True)
    print(f"\n=== eval 내부 장면 클러스터 (sim>=0.85 연결요소): {len(clusters)}개 ===")
    for c in clusters[:10]:
        print(f"  크기 {len(c):3d}: {eval_ids[c[0]]} ~ {eval_ids[c[-1]]} (예: {[eval_ids[j] for j in c[:3]]})")


if __name__ == "__main__":
    main()
