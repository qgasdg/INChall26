"""SO100 → SDXL VAE latent 사전 인코딩 캐시 (이식: impl/w2/build_latent_cache.py → INChall26).

pre_encode 학습(SO100LatentDataset)이 읽는 캐시를 **1회** 생성한다. 에피소드 전체 프레임을 SDXL VAE로
인코딩해 npz로 저장 → 학습은 [start:start+16] 슬라이스로 사용(매 스텝 VAE 인코딩 생략, 처리량↑).

레이아웃 (★ SO100LatentDataset 와 정확히 정합):
  <out>/latents/<user__dataset>/<episode:06d>.npz   # latents(L,4,h,w fp16) + actions_norm(L,6 z-score) + meta
  <out>/manifest.json                               # 인덱스 + 생성 파라미터(원자적 저장)

배선(dataset.py 와 동일): 열거=episodes.jsonl, IO=episode_io, 정규화/해상도=transforms.
재개 안전: 존재+로드+shape 검증(_valid_npz)으로 손상 npz 재생성, 원자적 저장(_atomic_savez).
GPU 전용(맥 금지). SDXL VAE = stabilityai/stable-diffusion-xl-base-1.0.

사용:
  python -m src.data.build_latent_cache                         # dry: 산정만
  python -m src.data.build_latent_cache --out latent_cache --resolution-mode canvas_aligned --run   # GPU
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from src.data import transforms as T
from src.data.dataset import _episode_lengths, _iter_datasets, _load_exclusions

VAE_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def _valid_npz(npz: Path, L: int, mode: str) -> bool:
    """재개 검증: 존재 + np.load 성공 + 기대 shape. 손상(EOFError·BadZipFile)이면 False → 재생성."""
    if not npz.exists():
        return False
    gh, gw = T.GEN_SIZE[mode]
    try:
        with np.load(npz) as z:
            if "latents" not in z.files or "actions_norm" not in z.files:
                return False
            return (z["latents"].shape == (L, 4, gh // 8, gw // 8)
                    and z["actions_norm"].shape == (L, 6))
    except Exception:
        return False


def _atomic_savez(npz: Path, **arrays) -> None:
    """부분 파일 방지: temp에 쓰고 원자적 rename(같은 FS). 크래시 시 깨진 최종 파일 안 남음."""
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(npz.name + ".tmp")
    try:
        with open(tmp, "wb") as f:                     # 파일객체 전달 → .npz 자동접미 방지
            np.savez_compressed(f, **arrays)
        os.replace(tmp, npz)                            # 원자적 교체(손상 파일 덮어쓰기 포함)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_vae(device, dtype):
    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(VAE_ID, subfolder="vae").to(device, dtype)
    vae.requires_grad_(False)
    return vae


def usable_episodes(cfg: dict) -> list[tuple[str, int, int]]:
    """제외 규칙 통과 + length>=seq_len 인 (ds_id, episode, length) 목록 (dataset.py 로직 재사용)."""
    seq_len = int((cfg or {}).get("task", {}).get("frames", 16))
    ex_ds, ex_ep = _load_exclusions(cfg or {})
    out: list[tuple[str, int, int]] = []
    for ds_id in _iter_datasets():
        if ds_id in ex_ds:
            continue
        for ep, length in sorted(_episode_lengths(ds_id).items()):
            if length >= seq_len and (ds_id, ep) not in ex_ep:
                out.append((ds_id, ep, length))
    return out


def estimate(cfg: dict, mode: str) -> dict:
    eps = usable_episodes(cfg)
    frames = sum(L for *_, L in eps)
    gh, gw = T.GEN_SIZE[mode]
    lat_elem = 4 * (gh // 8) * (gw // 8)
    return {"episodes": len(eps), "frames": frames, "gen_hw": [gh, gw],
            "latent_hw": [gh // 8, gw // 8], "disk_gb_fp16": round(frames * lat_elem * 2 / 1e9, 1)}


def run(mode: str, out: Path, batch: int, cfg: dict, logger) -> None:
    import torch

    from local_eval import episode_io

    if sys.platform == "darwin":
        logger.error("맥 실행 금지 — GPU 머신에서."); sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.error("CUDA 미탐지 — GPU 노드 필요."); sys.exit(1)

    mean, std = T.load_action_stats()
    eps = usable_episodes(cfg)
    vae = _load_vae(device, torch.bfloat16)
    scaling = vae.config.scaling_factor
    (out / "latents").mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    t0 = time.time()
    for idx, (ds_id, ep, L) in enumerate(eps):
        key = ds_id.replace("/", "__")
        npz = out / "latents" / key / f"{ep:06d}.npz"
        rel = f"latents/{key}/{ep:06d}.npz"
        if _valid_npz(npz, L, mode):                                   # 존재+로드+shape 검증(손상이면 재생성)
            manifest.append({"ds_id": ds_id, "episode": ep, "latent_path": rel, "num_frames": L})
            continue
        frames = episode_io.read_frames(ds_id, ep, 0, L)               # (L,H,W,3) uint8
        gen = T.final_to_gen_target(frames, mode)                      # (L,gh,gw,3) uint8
        x = torch.from_numpy(gen).to(device).float().div(127.5).sub(1.0).permute(0, 3, 1, 2).to(torch.bfloat16)
        lat = []
        with torch.no_grad():
            for i in range(0, x.shape[0], batch):
                z = vae.encode(x[i:i + batch]).latent_dist.sample().mul_(scaling)
                lat.append(z.to(torch.float16).cpu())
        latents = torch.cat(lat).numpy()                              # (L,4,h,w) fp16
        actions = T.normalize_action(episode_io.read_actions(ds_id, ep, 0, L), mean, std)  # (L,6) z-score
        _atomic_savez(npz, latents=latents, actions_norm=actions,     # 원자적 저장(부분파일 방지)
                      dataset=ds_id, episode=ep, num_frames=L, mode=mode)
        manifest.append({"ds_id": ds_id, "episode": ep, "latent_path": rel, "num_frames": L})
        if idx % 200 == 0:
            logger.info("[%d/%d] %s ep%d L=%d (%.1f min)", idx, len(eps), ds_id, ep, L, (time.time() - t0) / 60)

    mtmp = out / "manifest.json.tmp"                                  # manifest도 원자적
    mtmp.write_text(json.dumps(
        {"mode": mode, "vae": VAE_ID, "episodes": len(manifest), "clips": manifest}, ensure_ascii=False), encoding="utf-8")
    os.replace(mtmp, out / "manifest.json")
    logger.info("완료: %d 에피소드, %.1f분 → %s", len(manifest), (time.time() - t0) / 60, out)


def main() -> None:
    from src.utils.config import load_config
    from src.utils.runtime import get_logger

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("latent_cache"))
    ap.add_argument("--resolution-mode", default=None, choices=["canvas_aligned", "native_restore"],
                    help="미지정 시 base.yaml model.resolution_mode")
    ap.add_argument("--batch", type=int, default=8, help="VAE 인코딩 프레임 청크(OOM 방지)")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--server", default=None)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    logger = get_logger("build_latent_cache")

    cfg = load_config(exp=args.exp, server=args.server)
    mode = args.resolution_mode or cfg.get("model", {}).get("resolution_mode", "canvas_aligned")
    est = estimate(cfg, mode)
    logger.info("산정(%s): %s", mode, json.dumps(est, ensure_ascii=False))
    if not args.run:
        logger.info("DRY — 실제 생성은 --run (GPU). 예상 디스크 %.1fGB", est["disk_gb_fp16"])
        return
    run(mode, args.out, args.batch, cfg, logger)


if __name__ == "__main__":
    main()
