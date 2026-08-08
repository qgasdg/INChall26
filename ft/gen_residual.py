"""잔차 모델 생성 — 디노이징 반복 없이 **한 번의 forward**.

    출력 = decode( z_정적 + Δ ),   z_정적 = 첫 프레임 잠재를 16번 반복

학습 0스텝(Δ=0)이면 출력이 정적과 정확히 같아야 한다. `--verify-static` 이 그것을 검사한다:
디코드 결과의 모든 프레임이 첫 프레임과 픽셀 단위로 같은지 본다.
(VAE 재구성 오차는 모든 프레임에 똑같이 들어가므로 프레임 간 차이는 0 이어야 한다.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

BASELINE = Path(os.environ.get("FT_BASELINE", os.path.expanduser("~/ft/kit/baseline")))
KIT = BASELINE / "challenge_kit"
for p in (KIT / "libs/dynamicrafter", KIT / "src", BASELINE / "shared_libs/video_utils", KIT,
          Path(os.path.expanduser("~/ft"))):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from lvdm.utils.train import get_model  # noqa: E402
from scripts.eval.feature_csv_utils import (  # noqa: E402
    build_inference_batch,
    list_challenge_sample_ids,
    load_action_stats,
    save_video_tensor,
)
from ft_residual import static_latent  # noqa: E402
from gen_step0 import save_grid, to_12dim  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.expanduser("~/ft/configs/ft-20-residual.yaml"))
    p.add_argument("--challenge-root", default=os.path.expanduser("~/ft/data/eval"))
    p.add_argument("--action-stats-path",
                   default=os.path.expanduser("~/ft/data/train/so100_action_statistics.json"))
    p.add_argument("--delta-stats", default=os.path.expanduser("~/ft/data/train/so100_delta_statistics.json"))
    p.add_argument("--out", default=os.path.expanduser("~/ft/out/ft-20"))
    p.add_argument("--ckpt", default=None, help="학습본(.pt 또는 .ckpt). 없으면 0스텝(=정적)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--action-dims", type=int, default=12)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--precision", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify-static", action="store_true", help="프레임 간 차이가 0 인지 검사")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    cfg = OmegaConf.load(args.config)
    cfg.model.params.use_ema = False
    print(">>> 모델 구성 (backbone 적재)")
    model = get_model(cfg.model)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f">>> 학습본 적재: {args.ckpt} (안 실린 키 {len(unexpected)}개)")
    else:
        model.zero_init()   # backbone 적재 뒤에 해야 Δ=0 이 된다
    model.to(device).eval()

    action_mean, action_std = load_action_stats(args.action_stats_path)
    action_mean, action_std = action_mean.to(device), action_std.to(device)
    delta_std = None
    if args.action_dims == 12:
        delta_std = torch.tensor(json.loads(Path(args.delta_stats).read_text())["std"],
                                 dtype=torch.float32, device=device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = list_challenge_sample_ids(Path(args.challenge_root))[args.start: args.start + args.limit]
    print(f">>> 샘플 {len(ids)}개")

    tp = cfg.data.params
    amp = torch.autocast("cuda", dtype=torch.float16) if args.precision == 16 else torch.autocast("cuda", enabled=False)
    t0all = time.time()
    for i, sid in enumerate(ids, 1):
        t0 = time.time()
        batch = build_inference_batch(Path(args.challenge_root), [sid],
                                      tp.target_height, tp.target_width, tp.pad, args.fps,
                                      action_mean, action_std, device)
        if args.action_dims == 12:
            batch["act"] = to_12dim(batch["act"], action_std, delta_std)

        with torch.no_grad(), amp:
            z, c, fs = model.get_batch_input(batch, random_uncond=False, return_fs=True)
            z_static = static_latent(z)
            delta = model.predict_residual(z_static, c, fs=fs.long())
            video = model.decode_first_stage((z_static + delta).float())

        if args.verify_static:
            v = video[0]                                   # [c, t, h, w]
            d = (v - v[:, :1]).abs()
            print(f"    프레임 간 최대차 {d.max().item():.6f} · 평균차 {d.mean().item():.6f} "
                  f"(0 이면 완전 정적) · |Δ| rms {delta.float().pow(2).mean().sqrt().item():.6f}")

        save_video_tensor(video[0], out / f"{sid}.mp4", args.fps)
        save_grid(video[0], out / f"{sid}.png")
        print(f"[{i}/{len(ids)}] {sid} · {time.time()-t0:.2f}초 (누적 {(time.time()-t0all)/60:.1f}분)", flush=True)

    print(f">>> 저장 위치: {out}")


if __name__ == "__main__":
    main()
