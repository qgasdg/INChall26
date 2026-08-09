"""VAE 왕복(encode→decode)에서 정보가 얼마나 살아남는지 잰다.

**왜.** 우리 방식은 전부 잠재 공간에서 일어난다(출력 = z_정적 + Δ). VAE 가 잃는 것은
어떤 모델을 써도 복구할 수 없으므로, 이 값이 **잠재 공간 접근의 상한선**이다.

특히 **움직임이 살아남는가**가 핵심이다. 점수의 40%가 Action 항인데, 팔은 화면의 몇 %뿐이라
VAE 가 배경만 잘 복원하고 팔의 미세한 변화를 뭉개면 그쪽은 시작부터 불가능하다.

측정:
  ① 픽셀 재구성 오차 (전체 / 움직이는 영역만)
  ② 움직임 보존 — 프레임 간 차분의 크기와 상관계수
  ③ 곁들여 볼 그림 (원본 / 복원 / 차이)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

BASELINE = Path(os.environ.get("FT_BASELINE", os.path.expanduser("~/ft/kit/baseline")))
KIT = BASELINE / "challenge_kit"
for p in (KIT / "libs/dynamicrafter", KIT / "src", BASELINE / "shared_libs/video_utils", KIT,
          Path(os.path.expanduser("~/ft"))):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from lvdm.utils.train import get_model  # noqa: E402
from lvdm.utils.utils import instantiate_from_config  # noqa: E402


def to255(x):
    return ((x.clamp(-1, 1) + 1) * 127.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser("~/ft/configs/ft-20-residual.yaml"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=os.path.expanduser("~/ft/out/vae_probe"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = OmegaConf.load(args.config)
    cfg.model.params.use_ema = False
    model = get_model(cfg.model).cuda().eval()

    data = instantiate_from_config(cfg.data)
    data.setup()
    ds = data.val_dataset
    idx = np.random.RandomState(args.seed).choice(len(ds), args.n, replace=False)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'샘플':<8}{'전체MAE':>9}{'PSNR':>8}{'움직임부MAE':>13}"
          f"{'원본 모션':>11}{'복원 모션':>11}{'모션보존':>10}{'모션상관':>10}")
    print("-" * 82)
    rows = []
    for k, i in enumerate(idx):
        v = ds[int(i)]["video"].unsqueeze(0).cuda()          # [1,c,t,h,w] in [-1,1]
        with torch.no_grad():
            rec = model.decode_first_stage(model.encode_first_stage(v))

        o, r = to255(v[0]), to255(rec[0])                    # [c,t,h,w]
        err = (o - r).abs()
        mae = err.mean().item()
        psnr = (20 * torch.log10(255.0 / (err.pow(2).mean().sqrt() + 1e-8))).item()

        # 움직이는 영역: 원본에서 프레임 간 변화가 큰 픽셀
        do = (o[:, 1:] - o[:, :-1]).abs().mean(0)            # [t-1,h,w]
        dr = (r[:, 1:] - r[:, :-1]).abs().mean(0)
        mv = do > do.mean() + 2 * do.std()                   # 상위 꼬리 = 팔 근처
        mae_mv = err.mean(0)[1:][mv].mean().item() if mv.any() else float("nan")

        m_o, m_r = do.mean().item(), dr.mean().item()
        keep = m_r / (m_o + 1e-8)
        corr = torch.corrcoef(torch.stack([do.flatten(), dr.flatten()]))[0, 1].item()
        rows.append((mae, psnr, mae_mv, m_o, m_r, keep, corr))
        print(f"{k:<8}{mae:>9.2f}{psnr:>8.2f}{mae_mv:>13.2f}{m_o:>11.2f}{m_r:>11.2f}{keep:>10.3f}{corr:>10.3f}")

        if k < 3:      # 앞 3개만 그림으로
            t_idx = np.linspace(0, o.shape[1] - 1, 6).astype(int)
            strip = []
            for row in (o, r, (o - r).abs() * 4):            # 차이는 4배 강조
                strip.append(np.concatenate(
                    [row[:, t].permute(1, 2, 0).cpu().numpy() for t in t_idx], axis=1))
            img = np.concatenate(strip, axis=0).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(out / f"vae_{k}.png")

    a = np.array(rows)
    print("-" * 82)
    print(f"{'평균':<8}{a[:,0].mean():>9.2f}{a[:,1].mean():>8.2f}{a[:,2].mean():>13.2f}"
          f"{a[:,3].mean():>11.2f}{a[:,4].mean():>11.2f}{a[:,5].mean():>10.3f}{a[:,6].mean():>10.3f}")
    print(f"\n그림: {out}  (위=원본 · 가운데=복원 · 아래=차이×4)")


if __name__ == "__main__":
    main()
