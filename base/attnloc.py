"""Δ어텐션이 **어디를** 보는지 찍는다 — attnmap.py 의 ②를 위치 관점으로 확장.

**왜 따로 만드나.** attnmap.py 의 히트맵은 `m / m.amax()` 로 **전체 최댓값 하나**로 정규화한다.
한 칸이 나머지의 10배 이상이면 그 칸만 빨갛고 나머지 구조는 전부 파랗게 눌려 사라진다
(b10-7200 idx3 중간층: 최대 0.756 vs 평균 0.011). 그래서 여기서는
① 최댓값의 좌표를 픽셀로 환산해 찍고 ② 분위수로 분포를 보고
③ **프레임별 정규화**와 **p99 클립** 두 가지로 다시 그려 눌린 구조를 드러낸다.

나머지(모델 구성·어텐션 기록 패치·한 스텝 통과)는 attnmap.py 와 같다.
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
          Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from lvdm.modules import attention as _attn  # noqa: E402
from lvdm.utils.train import get_model  # noqa: E402
from lvdm.utils.utils import instantiate_from_config  # noqa: E402


def heat(strip: np.ndarray, base: np.ndarray) -> np.ndarray:
    """0~1 히트맵을 프레임 위에 얹는다 (attnmap.py 와 같은 배색: 빨강=높음, 파랑=낮음)."""
    s = (strip * 255).astype(np.uint8)
    h = np.stack([s, np.zeros_like(s), 255 - s], -1)
    return (0.55 * base + 0.45 * h).clip(0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timestep", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5, help="표시할 상위 지점 개수")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    cfg = OmegaConf.load(args.config)
    cfg.model.params.use_ema = False
    cfg.model.params.resume_unet = None
    print(">>> 모델 구성 중", flush=True)
    model = get_model(cfg.model)
    sd = torch.load(args.ckpt, map_location="cpu"); sd = sd.get("state_dict", sd)
    _, unexpected = model.load_state_dict(sd, strict=False)
    print(f">>> 학습본 적재: {args.ckpt} (안 실린 키 {len(unexpected)}개)", flush=True)
    model.to(device).eval()
    unet = model.model.diffusion_model

    data = instantiate_from_config(cfg.data); data.setup()
    item = data.val_dataset[args.idx]
    batch = {"video": item["video"].unsqueeze(0).to(device),
             "act": item["act"].unsqueeze(0).to(device),
             "caption": [""],
             "fps": torch.full((1,), 6, dtype=torch.long, device=device),
             "frame_stride": torch.full((1,), 6, dtype=torch.long, device=device),
             "start_idx": torch.zeros(1, dtype=torch.long, device=device)}

    torch.manual_seed(args.seed)
    z, cond, *_ = model.get_batch_input(batch, random_uncond=False)
    b, _, T, H, W = z.shape
    ctx = cond["c_crossattn"][0]
    C = ctx.shape[-1]
    L = (ctx.shape[1] - 77) // T
    img_pf = ctx[:, 77:].view(b, T, L, C)
    act_tok = unet.action_tokens(cond["act"])
    img_only = img_pf - act_tok

    grabbed: dict[str, torch.Tensor] = {}
    orig_fwd = _attn.CrossAttention.forward

    def rec_forward(self, x, context=None, mask=None):
        out = orig_fwd(self, x, context, mask)
        if context is not None and getattr(self, "_rec_name", None):
            h = self.heads
            q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
            ctx_img = context[:, self.text_context_len:, :]
            k = rearrange(self.to_k_ip(ctx_img), "b n (h d) -> b h n d", h=h)
            a = (q @ k.transpose(-1, -2) * self.scale).softmax(-1)
            grabbed[self._rec_name] = a.mean(1).detach().float().cpu()
        return out

    _attn.CrossAttention.forward = rec_forward
    targets = [(n, m) for n, m in unet.named_modules()
               if isinstance(m, _attn.CrossAttention) and hasattr(m, "to_k_ip") and "output_blocks" in n]
    picks = [targets[0], targets[len(targets) // 2], targets[-1]] if len(targets) >= 3 else targets
    for name, m in picks:
        m._rec_name = name

    t = torch.full((b,), args.timestep, dtype=torch.long, device=device)
    torch.manual_seed(args.seed)
    noise = torch.randn_like(z)
    x_noisy = model.q_sample(x_start=z, t=t, noise=noise)

    def run(c):
        grabbed.clear()
        with torch.no_grad(), torch.cuda.amp.autocast():
            model.apply_model(x_noisy, t, c, dropout_actions=False)
        return {k: v.clone() for k, v in grabbed.items()}

    att_true = run(cond)
    cond_zero = dict(cond)
    cond_zero["c_crossattn"] = [torch.cat([ctx[:, :77], img_only.reshape(b, T * L, C)], dim=1)]
    att_zero = run(cond_zero)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    vid = ((item["video"].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 3, 0).numpy().astype(np.uint8)  # [T,ph,pw,3]
    ph, pw = vid.shape[1:3]

    for name in att_true:
        d = (att_true[name] - att_zero[name]).abs().sum(-1)      # [b*t, hw]
        n_bt, hw = d.shape
        side = int(round(hw ** 0.5))
        h = side if side * side == hw else int(round((hw * H / W) ** 0.5))
        w = hw // max(h, 1)
        if h * w != hw:
            print(f"{name}: 격자 추정 실패 (hw={hw}) — 건너뜀")
            continue
        m = d.view(n_bt, h, w)[:T]                               # [T,h,w]
        a = m.numpy()

        flat = a.reshape(-1)
        p50, p90, p99, p999 = np.percentile(flat, [50, 90, 99, 99.9])
        print(f"\n===== {name}  격자 {h}x{w} =====")
        print(f"  분위수  p50 {p50:.5f} · p90 {p90:.5f} · p99 {p99:.5f} · p99.9 {p999:.5f} · 최대 {flat.max():.5f}")
        print(f"  최대/중앙값 배율: {flat.max()/max(p50,1e-9):.1f}배")
        idxs = np.argsort(flat)[::-1][:args.topk]
        print(f"  상위 {args.topk}개 지점 (프레임, 격자 y/x → 픽셀 y/x, 값):")
        pts = []
        for r in idxs:
            f, y, x = np.unravel_index(r, a.shape)
            py, px = int((y + 0.5) * ph / h), int((x + 0.5) * pw / w)
            pts.append((int(f), int(y), int(x), py, px, float(a[f, y, x])))
            print(f"    프레임 {f:2d} · 격자({y:2d},{x:2d}) · 픽셀({py:3d},{px:3d}) · {a[f,y,x]:.5f}")

        # (a) 전체 최댓값 정규화 — 기존 attnmap.py 와 같은 그림
        g = a / (a.max() + 1e-8)
        # (b) 프레임별 정규화 — 프레임마다 자기 최댓값으로
        pf = a / (a.max(axis=(1, 2), keepdims=True) + 1e-8)
        # (c) p99 클립 — 이상점을 잘라 나머지 구조를 살린다
        cl = np.clip(a / (p99 + 1e-8), 0, 1)

        for tag, arr in (("global", g), ("perframe", pf), ("p99clip", cl)):
            rows = []
            for r0 in (0, 8):
                strip = np.concatenate([np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize((pw, ph)))
                                        for x in arr[r0:r0 + 8]], axis=1) / 255.0
                base = np.concatenate([vid[i] for i in range(r0, min(r0 + 8, T))], axis=1)
                rows.append(heat(strip, base))
            img = Image.fromarray(np.concatenate(rows, axis=0))
            dr = ImageDraw.Draw(img)
            for (f, _y, _x, py, px, v) in pts:                   # 상위 지점에 동그라미
                col, row = f % 8, f // 8
                cx, cy = col * pw + px, row * ph + py
                dr.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], outline=(255, 255, 0), width=3)
            img.save(out_dir / f"{name.replace('.', '_')}__{tag}.png")
        print(f"  그림 3종 저장: {out_dir}")


if __name__ == "__main__":
    main()
