"""eval 영상 생성 — 대회 baseline 확산 파이프라인.

킷 코드(scripts/inference/generate_baseline_videos.py)는 고치지 않고 거기 함수만 빌린다.
더한 것은 ① 샘플 범위 지정 ② 12차원 액션 변환 ③ 프레임 그리드 PNG ④ 학습 체크포인트 적재.

`--ckpt` 없이 돌리면 사전학습 백본만으로 생성한다(0스텝). 그때는 **`--no-ema` 가 필수**다 —
LitEma 는 백본 적재 이전의 무작위 가중치를 들고 있고 backbone.ckpt 에 model_ema.* 가 없어
갱신되지 않으므로, 켜두면 전 프레임이 노이즈로 나온다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

BASELINE = Path(os.environ.get("FT_BASELINE", os.path.expanduser("~/ft/kit/baseline")))
KIT = BASELINE / "challenge_kit"
sys.path.insert(0, str(KIT / "libs/dynamicrafter"))
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(BASELINE / "shared_libs/video_utils"))
sys.path.insert(0, str(KIT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from lvdm.models.samplers.ddim import DDIMSampler  # noqa: E402
from lvdm.utils.train import get_model  # noqa: E402
from scripts.eval.feature_csv_utils import (  # noqa: E402
    build_inference_batch,
    list_challenge_sample_ids,
    load_action_stats,
    save_video_tensor,
)


def save_grid(video: torch.Tensor, path: Path, cols: int = 8) -> None:
    """video: [c, t, h, w], 값 범위 [-1, 1] → 프레임 그리드 PNG."""
    v = video.detach().float().cpu().clamp(-1, 1).add(1).div(2)
    t = v.shape[1]
    rows = (t + cols - 1) // cols
    h, w = v.shape[2], v.shape[3]
    canvas = np.ones((rows * h, cols * w, 3), dtype=np.float32)
    for i in range(t):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = v[:, i].permute(1, 2, 0).numpy()
    Image.fromarray((canvas * 255).astype(np.uint8)).save(path)



def to_12dim(act: torch.Tensor, action_std: torch.Tensor, delta_std: torch.Tensor) -> torch.Tensor:
    """act: [b, t, 6] (절대값을 전역 std 로 정규화한 것) → [b, t, 12] (절대6 + 정규화된 Δ6).

    z[t] = (a[t]-mean)/std 이므로 a[t]-a[t-1] = (z[t]-z[t-1])*std.
    Δ 전용 std 로 다시 나눈다. t=0 의 Δ 는 0.
    """
    dz = torch.diff(act, dim=1, prepend=act[:, :1])
    d_norm = dz * (action_std / delta_std).view(1, 1, -1)
    return torch.cat([act, d_norm], dim=-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.expanduser("~/base/base-01.yaml"))
    p.add_argument("--challenge-root", default=os.path.expanduser("~/ft/data/eval"))
    p.add_argument("--action-stats-path",
                   default=os.path.expanduser("~/ft/data/train/so100_action_statistics.json"))
    p.add_argument("--out", default=os.path.expanduser("~/ft/out/ft-00"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--ddim-steps", type=int, default=None)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--precision", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-dims", type=int, default=6, help="12 이면 절대6 + Δ6")
    p.add_argument("--ckpt", default=None,
                   help="학습 체크포인트(.ckpt). 백본 위에 덮어씌운다. save_only_unet 이라 model* 만 들어 있다")
    p.add_argument("--no-ema", action="store_true",
                   help="EMA 비활성화 — 0스텝에서는 EMA 가 사전학습 이전의 랜덤 가중치를 들고 있다")
    p.add_argument("--delta-stats", default=os.path.expanduser("~/ft/data/train/so100_delta_statistics.json"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config)
    if args.no_ema:
        cfg.model.params.use_ema = False
        print(">>> use_ema=False — 로드된 가중치를 그대로 쓴다")
    print(">>> 모델 구성 중 (backbone 적재)")
    model = get_model(cfg.model)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f">>> 학습 체크포인트 적재: {args.ckpt} (안 실린 키 {len(unexpected)}개)")

    model.to(device).eval()
    sampler = DDIMSampler(model)

    ddim_kwargs = OmegaConf.to_container(cfg.ddim_kwargs, resolve=True)
    if args.ddim_steps is not None:
        ddim_kwargs["ddim_steps"] = args.ddim_steps

    action_mean, action_std = load_action_stats(args.action_stats_path)
    if action_mean is not None:
        action_mean, action_std = action_mean.to(device), action_std.to(device)

    delta_std = None
    if args.action_dims == 12:
        ds = json.loads(Path(args.delta_stats).read_text())
        delta_std = torch.tensor(ds["std"], dtype=torch.float32, device=device)
        print(f">>> Δ std = {[round(v, 3) for v in ds['std']]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample_ids = list_challenge_sample_ids(Path(args.challenge_root))[args.start : args.start + args.limit]
    print(f">>> 샘플 {len(sample_ids)}개: {sample_ids}")

    amp = torch.cuda.amp.autocast() if args.precision == 16 and device.type == "cuda" else nullcontext()
    tp = cfg.data.params
    import time
    t_start = time.time()
    for i, sid in enumerate(sample_ids, 1):
        if (out / f"{sid}.mp4").exists():
            continue
        t0 = time.time()
        batch = build_inference_batch(
            Path(args.challenge_root), [sid],
            tp.target_height, tp.target_width, tp.pad, args.fps,
            action_mean, action_std, device,
        )
        if args.action_dims == 12:
            batch["act"] = to_12dim(batch["act"], action_std, delta_std)
        z, c, uc, cond_mask, _logs, kwargs = model.prepare_batch_for_inference(batch)
        sk = dict(ddim_kwargs)
        sk.update(kwargs)
        steps = sk.pop("ddim_steps")
        shape = (model.channels, model.temporal_length, *model.image_size)
        with torch.no_grad(), (nullcontext() if args.no_ema else model.ema_scope("ft-00")):
            with amp:
                samples, _ = sampler.sample(steps, batch_size=z.shape[0], shape=shape,
                                            conditioning=c, unconditional_conditioning=uc,
                                            mask=cond_mask, x0=z, **sk)
            gen = model.decode_first_stage(samples)
        save_video_tensor(gen[0], out / f"{sid}.mp4", args.fps)
        save_grid(gen[0], out / f"{sid}.png")
        el = time.time() - t0
        print(f"[{i}/{len(sample_ids)}] {sid} · {el:.1f}초 (누적 {(time.time()-t_start)/60:.1f}분)", flush=True)

    print(f">>> 저장 위치: {out}")


if __name__ == "__main__":
    main()
