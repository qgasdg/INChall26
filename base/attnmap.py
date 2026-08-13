"""액션이 실제로 실렸는지 세 각도로 잰다 — 크기 · Δ어텐션 맵 · 경로별 출력 민감도.

**왜 어텐션 맵만으로는 안 되나.** 액션 토큰은 이미지 토큰에 **더해져** 들어간다
(`context_image = img_emb + act_tok`). 슬롯별 어텐션 가중치는 둘을 구분하지 못하므로
"이 토큰에 주의가 갔다"가 "액션에 주의가 갔다"를 뜻하지 않는다. 그래서 **진짜 액션과
액션 0 의 차이**를 본다.

**왜 경로를 나눠 재나.** 액션은 cross-attention 말고 `emb = time_embed + act_emb` 로도
들어간다. cross-attention 이 아무것도 안 날라도 옛 경로로 영향이 갈 수 있으므로,
한 경로씩 꺼서 어느 쪽이 나르는지 본다.

**한 스텝만 잰다.** DDIM 50스텝을 다 돌리면 작은 차이가 혼돈으로 증폭돼 무엇이 원인인지
흐려진다. 고정된 노이즈·타임스텝에서 UNet 을 한 번만 통과시킨다.
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
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from lvdm.modules import attention as _attn  # noqa: E402
from lvdm.utils.train import get_model  # noqa: E402
from lvdm.utils.utils import instantiate_from_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser("~/base/base-06-duck.yaml"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/ft/out/attnmap"))
    ap.add_argument("--timestep", type=int, default=500, help="확산 타임스텝(0~999). 중간이 기본")
    ap.add_argument("--seed", type=int, default=0)
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

    # ── ① 크기 비교 ────────────────────────────────────────────────────
    ctx = cond["c_crossattn"][0]                       # [b, 77 + T*L, C]
    C = ctx.shape[-1]
    L = (ctx.shape[1] - 77) // T                       # 프레임당 이미지 토큰 (Resampler 16)
    img_pf = ctx[:, 77:].view(b, T, L, C)              # 이미지 토큰 + 액션 토큰
    act_tok = unet.action_tokens(cond["act"])          # [b, T, L, C]
    img_only = img_pf - act_tok
    print(f">>> context {tuple(ctx.shape)} · 프레임 {T} · 프레임당 이미지 토큰 {L}")
    print("\n=== ① 액션 토큰이 K/V 를 얼마나 흔드나 ===")
    print(f"‖이미지 토큰‖ 평균 : {img_only.norm(dim=-1).mean():.4f}")
    print(f"‖액션 토큰‖  평균 : {act_tok.norm(dim=-1).mean():.4f}")
    print(f"비율(액션/이미지)  : {(act_tok.norm(dim=-1).mean()/img_only.norm(dim=-1).mean()):.4f}")
    print(f"프레임 간 액션토큰 변화(있어야 프레임별 정보): "
          f"{(act_tok[:,1:]-act_tok[:,:-1]).abs().mean():.6f}")

    # ── 어텐션 기록용 패치 ─────────────────────────────────────────────
    grabbed: dict[str, torch.Tensor] = {}
    orig_fwd = _attn.CrossAttention.forward

    def rec_forward(self, x, context=None, mask=None):
        out = orig_fwd(self, x, context, mask)
        if context is not None and getattr(self, "_rec_name", None):
            h = self.heads
            q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
            ctx_img = context[:, self.text_context_len:, :]
            k = rearrange(self.to_k_ip(ctx_img), "b n (h d) -> b h n d", h=h)
            a = (q @ k.transpose(-1, -2) * self.scale).softmax(-1)     # [b,h,hw,K]
            grabbed[self._rec_name] = a.mean(1).detach().float().cpu()  # 헤드 평균 [b,hw,K]
        return out

    _attn.CrossAttention.forward = rec_forward
    targets = []
    for name, m in unet.named_modules():
        if isinstance(m, _attn.CrossAttention) and hasattr(m, "to_k_ip") and "output_blocks" in name:
            targets.append((name, m))
    picks = [targets[0], targets[len(targets) // 2], targets[-1]] if len(targets) >= 3 else targets
    for name, m in picks:
        m._rec_name = name
    print(f"\n어텐션 기록 대상 {len(picks)}개: " + " · ".join(n for n, _ in picks))

    # ── 한 스텝 통과 ──────────────────────────────────────────────────
    t = torch.full((b,), args.timestep, dtype=torch.long, device=device)
    torch.manual_seed(args.seed)
    noise = torch.randn_like(z)
    x_noisy = model.q_sample(x_start=z, t=t, noise=noise)

    def run(c, tag):
        grabbed.clear()
        with torch.no_grad(), torch.cuda.amp.autocast():
            out, _ = model.apply_model(x_noisy, t, c, dropout_actions=False)
        return out.float(), {k: v.clone() for k, v in grabbed.items()}

    cond_zero = dict(cond); cond_zero["c_crossattn"] = [
        torch.cat([ctx[:, :77], img_only.reshape(b, T * L, C)], dim=1)]   # 액션 토큰만 제거

    out_true, att_true = run(cond, "true")
    out_noact, att_zero = run(cond_zero, "zero")

    # ── ② Δ어텐션 맵 ──────────────────────────────────────────────────
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    frame0 = ((item["video"][:, 0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy()
    print("\n=== ② 액션이 어텐션을 얼마나 바꾸나 (층별) ===")
    for name in att_true:
        a1, a0 = att_true[name], att_zero[name]
        d = (a1 - a0).abs().sum(-1)                     # [b*t, hw] — 슬롯 합
        n_bt, hw = d.shape
        side = int(round(hw ** 0.5))
        h = side if side * side == hw else int(round((hw * H / W) ** 0.5))
        w = hw // max(h, 1)
        print(f"{name:<52} |Δ어텐션| 평균 {d.mean():.5f}  최대 {d.max():.5f}")
        if h * w != hw:
            continue
        m = d.view(n_bt, h, w)[:T]                      # 프레임별
        m = (m / (m.amax() + 1e-8)).numpy()
        ph, pw = frame0.shape[:2]                       # 히트맵을 픽셀 해상도로 키운다
        strip = np.concatenate([np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize((pw, ph)))
                                for x in m[:8]], axis=1)
        base = np.concatenate([frame0.astype(np.uint8)] * 8, axis=1)
        heat = np.stack([strip, np.zeros_like(strip), 255 - strip], -1)
        blend = (0.55 * base + 0.45 * heat).clip(0, 255).astype(np.uint8)
        Image.fromarray(blend).save(out_dir / f"dattn_{name.replace('.', '_')}.png")

    # ── ③ 경로별 출력 민감도 ──────────────────────────────────────────
    print("\n=== ③ UNet 출력이 액션에 얼마나 반응하나 (한 스텝, 상대 크기) ===")
    ref = out_true.abs().mean()

    def rel(o):
        return ((o - out_true).abs().mean() / ref).item()

    print(f"{'cross-attn 액션만 제거':<28}{rel(out_noact):.5f}")

    # 덧셈 경로를 끄는 방법: action_embed 의 마지막 Linear 를 잠시 0 으로 만든다
    last = [m for m in unet.action_embed.modules() if isinstance(m, torch.nn.Linear)][-1]
    w, bs = last.weight.data.clone(), (last.bias.data.clone() if last.bias is not None else None)
    last.weight.data.zero_()
    if last.bias is not None:
        last.bias.data.zero_()
    out_noadd, _ = run(cond, "noadd")
    print(f"{'덧셈(time_embed) 액션만 제거':<28}{rel(out_noadd):.5f}")
    out_none, _ = run(cond_zero, "none")
    print(f"{'둘 다 제거':<28}{rel(out_none):.5f}")
    last.weight.data.copy_(w)
    if bs is not None:
        last.bias.data.copy_(bs)

    # 액션을 시간축으로 섞으면? (크기는 같고 순서만 틀림)
    perm = torch.randperm(T, device=device)
    cond_shuf = dict(cond)
    act_s = cond["act"][:, perm]
    tok_s = unet.action_tokens(act_s)
    cond_shuf["act"] = act_s
    cond_shuf["c_crossattn"] = [torch.cat([ctx[:, :77], (img_only + tok_s).reshape(b, T * L, C)], dim=1)]
    out_shuf, _ = run(cond_shuf, "shuf")
    print(f"{'액션 시간축 섞기':<28}{rel(out_shuf):.5f}")

    _attn.CrossAttention.forward = orig_fwd
    print(f"\n그림: {out_dir}")


if __name__ == "__main__":
    main()
