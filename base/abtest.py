"""A/B 액션 테스트 — 액션이 출력을 실제로 좌우하는지, 방향까지 맞는지 가른다.

**설계.** 같은 클립으로 두 판을 만든다.

  A: 조건 = GT 프레임 0  · 액션 = 원래 순서    → 정답 = GT
  B: 조건 = GT 프레임 15 · 액션 = 되감기       → 정답 = **되감은 GT**

B 는 조건 이미지와 액션이 서로 맞고(둘 다 마지막 자세에서 출발), 관절 범위도 벗어나지 않으며,
**비교할 정답 영상이 존재한다**. 액션만 뒤집었는데 출력이 되감은 GT 를 따라가면 액션이 작동하는 것이고,
A 와 똑같이 나오면 액션을 안 보는 것이다.

대조군으로 C 를 둔다 — 조건은 프레임 0 인데 액션만 되감기다. 이건 학습에서 본 적 없는
**모순된 입력**이라, B 와 C 가 다르면 모델이 조건 이미지와 액션의 정합을 쓰고 있다는 뜻이다.

Δ 재계산 주의: act 는 [절대 6 + Δ 6] 이다. 되감을 때 **절대값만 뒤집고 Δ 는 다시 계산**해야 한다
(Δ 열을 그냥 뒤집으면 부호와 정렬이 어긋난다).
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

from generate import save_grid  # noqa: E402


def redo_delta(abs6: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """abs6: [T,6] 정규화 절대값 → [T,12]. data12.Act12Dataset 과 같은 식."""
    dz = torch.diff(abs6, dim=0, prepend=abs6[:1])
    return torch.cat([abs6, dz * scale], dim=-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.expanduser("~/base/base-03.yaml"))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--idx", type=int, default=0, help="val_dataset 인덱스")
    p.add_argument("--out", default=os.path.expanduser("~/ft/out/abtest"))
    p.add_argument("--cfg", type=float, default=None, help="액션 CFG 배율")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda:0")
    cfg = OmegaConf.load(args.config)
    cfg.model.params.use_ema = False
    cfg.model.params.resume_unet = None          # 학습본은 --ckpt 로만 싣는다
    print(">>> 모델 구성 중", flush=True)
    model = get_model(cfg.model)
    sd = torch.load(args.ckpt, map_location="cpu"); sd = sd.get("state_dict", sd)
    _, unexpected = model.load_state_dict(sd, strict=False)
    print(f">>> 학습본 적재: {args.ckpt} (안 실린 키 {len(unexpected)}개)", flush=True)
    model.to(device).eval()
    sampler = DDIMSampler(model)

    ddim_kwargs = OmegaConf.to_container(cfg.ddim_kwargs, resolve=True)
    if args.cfg is not None:
        ddim_kwargs["unconditional_guidance_scale"] = args.cfg

    data = instantiate_from_config(cfg.data); data.setup()
    item = data.val_dataset[args.idx]
    gt = item["video"].unsqueeze(0).to(device)                   # [1,c,16,h,w]
    act = item["act"].to(device)                                 # [16,12]
    st = json.loads(Path(cfg.data.params.action_stats_path).read_text())
    ds = json.loads(Path(cfg.data.params.delta_stats_path).read_text())
    scale = (torch.tensor(st["std"]) / torch.tensor(ds["std"])).float().to(device)

    act_rev = redo_delta(act[:, :6].flip(0), scale)               # 절대값 뒤집고 Δ 재계산
    gt_rev = gt.flip(2)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save_grid(gt[0], out / f"v{args.idx}_GT.png")
    save_grid(gt_rev[0], out / f"v{args.idx}_GT_되감기.png")

    def make_batch(cond_frame, a):
        v = torch.zeros_like(gt)
        v[:, :, 0] = cond_frame
        return {"video": v, "act": a.unsqueeze(0),
                "caption": [""],
                "fps": torch.full((1,), args.fps, dtype=torch.long, device=device),
                "frame_stride": torch.full((1,), args.fps, dtype=torch.long, device=device),
                "start_idx": torch.zeros(1, dtype=torch.long, device=device)}

    cases = [
        ("A_정방향",       make_batch(gt[:, :, 0],  act),     gt),
        ("B_되감기",       make_batch(gt[:, :, -1], act_rev), gt_rev),
        ("C_모순(0+되감기)", make_batch(gt[:, :, 0],  act_rev), None),
    ]

    def to255(x):
        return (x.clamp(-1, 1) + 1) * 127.5

    print(f"\n{'경우':<18}{'정답MAE':>10}{'프레임간':>10}{'누적변위':>10}{'초':>7}")
    print("-" * 56)
    for gtn, ref in (("GT", gt), ("GT_되감기", gt_rev)):
        f = to255(ref[0]).permute(1, 0, 2, 3)
        print(f"{gtn:<18}{'—':>10}{(f[1:]-f[:-1]).abs().mean():10.3f}"
              f"{(f[-1]-f[0]).abs().mean():10.3f}{'—':>7}")
    print("-" * 56)

    outs = {}
    for name, batch, ref in cases:
        t0 = time.time()
        torch.manual_seed(args.seed)
        z, c, uc, cond_mask, _, kw = model.prepare_batch_for_inference(batch)
        sk = dict(ddim_kwargs); sk.update(kw)
        steps = sk.pop("ddim_steps")
        shape = (model.channels, model.temporal_length, *model.image_size)
        with torch.no_grad(), torch.cuda.amp.autocast():
            s, _ = sampler.sample(steps, batch_size=1, shape=shape, conditioning=c,
                                  unconditional_conditioning=uc, mask=cond_mask, x0=z, **sk)
        gen = model.decode_first_stage(s)
        save_grid(gen[0], out / f"v{args.idx}_{name}.png")
        outs[name] = gen
        g = to255(gen[0]).permute(1, 0, 2, 3)
        mae = (to255(gen) - to255(ref)).abs().mean().item() if ref is not None else float("nan")
        print(f"{name:<18}{mae:10.3f}{(g[1:]-g[:-1]).abs().mean():10.3f}"
              f"{(g[-1]-g[0]).abs().mean():10.3f}{time.time()-t0:7.1f}")

    print(f"\n=== 서로 얼마나 다른가 (MAE, 0~255) ===")
    ks = list(outs)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = (to255(outs[ks[i]]) - to255(outs[ks[j]])).abs().mean().item()
            print(f"{ks[i]} vs {ks[j]}: {d:.3f}")
    print(f"\n그림: {out}")


if __name__ == "__main__":
    main()
