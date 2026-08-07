"""액션 체크포인트의 **대조 격차(gap)** 를 홀드아웃에서 잰다 — 학습 없이.

gap = (mse_shuffled − mse_true) / mse_shuffled

0 이면 모델이 액션을 무시하는 것, >0 이면 정답 액션이 실제로 도움이 되는 것.
학습 목적함수가 아니라 **지표로** 쓰는 것이라, ft-10(평범한 손실)과 ft-11(대조 손실)을
같은 자로 비교할 수 있다. 이 비교가 "대조 손실이 필요했나 vs 액션 학습이면 무엇이든 되나"를 가른다.

같은 배치·같은 t·같은 노이즈를 모든 체크포인트에 재사용하므로(seed 고정) 짝지은 비교다.
"""
from __future__ import annotations

import argparse
import os
import sys
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
from lvdm.utils.utils import instantiate_from_config  # noqa: E402
from ft_loss import shuffle_time  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser("~/ft/configs/ft-10-actiononly.yaml"))
    ap.add_argument("--action-ckpts", nargs="*", default=[], help="비교할 .pt 들. 없으면 zero-init 만")
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.model.params.contrast_weight = 0.0        # 지표로만 쓴다
    print(">>> 모델 구성")
    model = get_model(cfg.model).cuda().eval()

    data = instantiate_from_config(cfg.data)
    data.setup()
    loader = data.val_dataloader()

    # 배치·t·노이즈를 미리 고정해 모든 체크포인트에 같은 조건을 준다
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    batches = []
    for i, b in enumerate(loader):
        if i >= args.batches:
            break
        batches.append({k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()})
    print(f">>> 홀드아웃 배치 {len(batches)}개 고정")

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    fixed = [(torch.randint(0, model.num_timesteps, (1,), generator=g).cuda(), None) for _ in batches]

    def gap_for(ckpt):
        if ckpt:
            sd = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(sd, strict=False)
        else:  # zero-init = 학습 전
            ae = model.model.diffusion_model.action_embed
            last = [m for m in ae if isinstance(m, torch.nn.Linear)][-1]
            torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)

        gaps = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for (t, _), batch in zip(fixed, batches):
                x, c, fs = model.get_batch_input(batch, random_uncond=False, return_fs=True)
                torch.manual_seed(args.seed)          # 같은 노이즈
                noise = torch.randn_like(x)
                x_noisy = model.q_sample(x_start=x, t=t, noise=noise)
                target = model.get_v(x, noise, t)

                out_t, _ = model.apply_model(x_noisy, t, c, fs=fs.long())
                mse_t = model.get_loss(out_t, target, mean=False).mean([1, 2, 3, 4])

                c_w = dict(c); c_w["act"] = shuffle_time(c["act"])
                out_w, _ = model.apply_model(x_noisy, t, c_w, fs=fs.long())
                mse_w = model.get_loss(out_w, target, mean=False).mean([1, 2, 3, 4])

                gaps.append(((mse_w - mse_t) / (mse_w + 1e-8)).float().cpu().numpy())
        return np.concatenate(gaps)

    print(f"\n{'체크포인트':<24} {'평균 gap':>12} {'표준오차':>10} {'>0 비율':>9}")
    print("-" * 60)
    for ckpt in [None] + args.action_ckpts:
        gaps = gap_for(ckpt)
        name = "zero-init (학습 전)" if ckpt is None else Path(ckpt).stem
        se = gaps.std() / np.sqrt(len(gaps))
        print(f"{name:<24} {gaps.mean():>+12.5f} {se:>10.5f} {(gaps>0).mean()*100:>8.1f}%")


if __name__ == "__main__":
    main()
