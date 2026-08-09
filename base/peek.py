"""train 1개 · eval 1개를 같은 조건으로 생성해 프레임 그리드로 뽑는다.

**왜 둘을 같이.** train 에는 GT 가 있어서 '무엇이 나와야 했는지'와 나란히 볼 수 있고,
eval 은 실제 채점 대상이다. 두 도메인에서 백본이 같은 품질을 내는지 눈으로 확인하는 용도다.

**공정성**: train 도 eval 과 똑같이 **첫 프레임만 남기고 나머지 15프레임을 0으로 지운 뒤** 넣는다.
DDIM 이 `mask`(0번 프레임)로 x0 을 덮어쓰므로 안 지우면 GT 가 새어들 여지가 생긴다.

`--ckpt` 로 학습 체크포인트를 얹을 수 있다. 없으면 0스텝(사전학습 백본)이고, 그때는 `--no-ema` 가 필수다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

BASELINE = Path(os.environ.get("FT_BASELINE", os.path.expanduser("~/ft/kit/baseline")))
KIT = BASELINE / "challenge_kit"
for p in (KIT / "libs/dynamicrafter", KIT / "src", BASELINE / "shared_libs/video_utils", KIT,
          Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from lvdm.models.samplers.ddim import DDIMSampler  # noqa: E402
from lvdm.utils.train import get_model  # noqa: E402
from lvdm.utils.utils import instantiate_from_config  # noqa: E402
from scripts.eval.feature_csv_utils import build_inference_batch, load_action_stats  # noqa: E402

from generate import save_grid, to_12dim  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.expanduser("~/base/base-01.yaml"))
    p.add_argument("--challenge-root", default=os.path.expanduser("~/ft/data/eval"))
    p.add_argument("--action-stats-path",
                   default=os.path.expanduser("~/ft/data/train/so100_action_statistics.json"))
    p.add_argument("--delta-stats", default=os.path.expanduser("~/ft/data/train/so100_delta_statistics.json"))
    p.add_argument("--out", default=os.path.expanduser("~/ft/out/peek"))
    p.add_argument("--eval-id", default="sample_000000")
    p.add_argument("--train-idx", type=int, default=-1, help="-1 이면 val_dataset 에서 무작위 1개")
    p.add_argument("--ddim-steps", type=int, default=None)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--tag", default="step0")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda:0")
    cfg = OmegaConf.load(args.config)
    if args.no_ema:
        cfg.model.params.use_ema = False
        print(">>> use_ema=False", flush=True)

    print(">>> 모델 구성 중 (backbone 적재)", flush=True)
    model = get_model(cfg.model)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        _, unexpected = model.load_state_dict(sd, strict=False)
        print(f">>> 학습본 적재: {args.ckpt} (안 실린 키 {len(unexpected)}개)", flush=True)
    model.to(device).eval()
    sampler = DDIMSampler(model)

    ddim_kwargs = OmegaConf.to_container(cfg.ddim_kwargs, resolve=True)
    if args.ddim_steps is not None:
        ddim_kwargs["ddim_steps"] = args.ddim_steps

    action_mean, action_std = load_action_stats(args.action_stats_path)
    action_mean, action_std = action_mean.to(device), action_std.to(device)
    delta_std = torch.tensor(json.loads(Path(args.delta_stats).read_text())["std"],
                             dtype=torch.float32, device=device)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tp = cfg.data.params

    def run(batch, name):
        t0 = time.time()
        z, c, uc, cond_mask, _, kw = model.prepare_batch_for_inference(batch)
        sk = dict(ddim_kwargs); sk.update(kw)
        steps = sk.pop("ddim_steps")
        shape = (model.channels, model.temporal_length, *model.image_size)
        with torch.no_grad(), (nullcontext() if args.no_ema else model.ema_scope("peek")):
            with torch.cuda.amp.autocast():
                s, _ = sampler.sample(steps, batch_size=z.shape[0], shape=shape,
                                      conditioning=c, unconditional_conditioning=uc,
                                      mask=cond_mask, x0=z, **sk)
            gen = model.decode_first_stage(s)
        save_grid(gen[0], out / f"{name}.png")
        print(f"[{name}] {time.time()-t0:.1f}초 → {out/f'{name}.png'}", flush=True)
        return gen[0]

    # ── eval ──────────────────────────────────────────────────────────
    b = build_inference_batch(Path(args.challenge_root), [args.eval_id],
                              tp.target_height, tp.target_width, tp.pad, args.fps,
                              action_mean, action_std, device)
    b["act"] = to_12dim(b["act"], action_std, delta_std)
    save_grid(b["video"][0, :, :1].expand(-1, 16, -1, -1), out / f"eval_{args.eval_id}_input.png")
    run(b, f"eval_{args.eval_id}_{args.tag}")

    # ── train ─────────────────────────────────────────────────────────
    data = instantiate_from_config(cfg.data); data.setup()
    ds = data.val_dataset
    idx = args.train_idx if args.train_idx >= 0 else int(np.random.RandomState(args.seed).randint(len(ds)))
    item = ds[idx]
    gt = item["video"].unsqueeze(0).to(device)                 # [1,c,T,h,w] — 정답
    save_grid(gt[0], out / f"train{idx}_gt.png")

    vid = torch.zeros_like(gt)
    vid[:, :, 0] = gt[:, :, 0]                                  # eval 과 동일: 첫 프레임만
    tb = {"video": vid,
          "act": item["act"].unsqueeze(0).to(device),           # data12 가 이미 12차원으로 준다
          "caption": [""],
          "fps": torch.full((1,), args.fps, dtype=torch.long, device=device),
          "frame_stride": torch.full((1,), args.fps, dtype=torch.long, device=device),
          "start_idx": torch.zeros(1, dtype=torch.long, device=device)}
    print(f">>> train 샘플 {idx} (act shape {tuple(item['act'].shape)})", flush=True)
    run(tb, f"train{idx}_{args.tag}")

    print(f"\n저장 위치: {out}")


if __name__ == "__main__":
    main()
