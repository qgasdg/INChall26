"""킷 모델을 감싸는 얇은 서브클래스 — 킷 코드는 고치지 않는다.

두 가지를 한다.

**① 어텐션을 SDPA 로 교체.** `lvdm` 은 xformers 가 있으면 memory-efficient 경로를 쓰지만
설치돼 있지 않아 `sim = softmax(q@k^T)` 로 **어텐션 행렬을 통째로 만든다**. 공간 어텐션은
토큰이 40×64=2560 이라 한 층에서만 2GB 가까이 잡아 1.44B 전체 학습이 즉시 OOM 났다.
torch 2.8 내장 `scaled_dot_product_attention` 은 같은 계산을 행렬 없이 한다(flash/mem-efficient 커널).
`use_relative_position: false`·`use_causal_attention: False` 라 상대위치·마스크 분기가 없어 그대로 대체된다.

**② 학습 범위 선택(`freeze_scope`).** 31.4GB GPU 에서 파라미터+그래디언트+AdamW 상태만
전체 21.5GB / 디코더+middle 14.6GB / 어텐션계열 3.4GB 다. 전체가 안 들어가면 여기서 줄인다.
동결은 `__init__` 에서 한다 — 콜백으로 걸었더니 실행되지 않아 1.4B 를 통째로 학습한 적이 있다.
(`requires_grad` 는 이후의 체크포인트 적재로 바뀌지 않으므로 `__init__` 시점이 안전하다.)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from lvdm.models.ddpm3d import LatentVisualDiffusion
from lvdm.modules import attention as _attn

_PATCHED = False


def _sdpa_forward(self, x, context=None, mask=None):
    """CrossAttention.forward 를 SDPA 로. 상대위치/마스크가 있으면 원본으로 되돌린다."""
    if self.relative_position or mask is not None:
        return _sdpa_forward._orig(self, x, context, mask)

    spatial_self_attn = context is None
    k_ip = v_ip = out_ip = None
    h = self.heads
    q = self.to_q(x)
    context = x if context is None else context

    if self.image_cross_attention and not spatial_self_attn:
        context, context_image = context[:, : self.text_context_len, :], context[:, self.text_context_len :, :]
        k, v = self.to_k(context), self.to_v(context)
        k_ip, v_ip = self.to_k_ip(context_image), self.to_v_ip(context_image)
    else:
        if not spatial_self_attn:
            context = context[:, : self.text_context_len, :]
        k, v = self.to_k(context), self.to_v(context)

    q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v)          # scale=dim_head**-0.5 가 기본값
    out = rearrange(out, "b h n d -> b n (h d)")

    if k_ip is not None:
        k_ip, v_ip = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (k_ip, v_ip))
        out_ip = rearrange(F.scaled_dot_product_attention(q, k_ip, v_ip), "b h n d -> b n (h d)")
        if self.image_cross_attention_scale_learnable:
            out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha) + 1)
        else:
            out = out + self.image_cross_attention_scale * out_ip

    return self.to_out(out)


def enable_sdpa_attention() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _sdpa_forward._orig = _attn.CrossAttention.forward
    _attn.CrossAttention.forward = _sdpa_forward
    _PATCHED = True
    print(">>> [base] 어텐션을 torch SDPA 로 교체", flush=True)


# freeze_scope → 동결할 UNet 하위 모듈 이름
_SCOPES = {
    "none": (),                                  # 전체 학습 (1440M)
    "encoder": ("input_blocks",),                # 디코더+middle+기타만 (994M)
    "encoder_middle": ("input_blocks", "middle_block"),   # 디코더+기타만 (818M)
}


class BaseLatentVisualDiffusion(LatentVisualDiffusion):
    def __init__(self, *args, freeze_scope: str = "none",
                 motion_weight: float = 0.0, resume_unet: str | None = None, **kwargs):
        enable_sdpa_attention()
        super().__init__(*args, **kwargs)
        if freeze_scope not in _SCOPES:
            raise ValueError(f"freeze_scope={freeze_scope!r} — {list(_SCOPES)} 중 하나여야 한다")
        self.freeze_scope = freeze_scope
        self.motion_weight = float(motion_weight)
        self.resume_unet = resume_unet

        unet = self.model.diffusion_model
        for name in _SCOPES[freeze_scope]:
            for p in getattr(unet, name).parameters():
                p.requires_grad_(False)

        tr = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        fr = sum(p.numel() for p in unet.parameters() if not p.requires_grad)
        print(f">>> [base] freeze_scope={freeze_scope} · 학습 {tr/1e6:.1f}M / 동결 {fr/1e6:.1f}M "
              f"· 옵티마이저 예상 {tr*16/2**30:.1f}GB", flush=True)

    def get_param_list(self):
        """동결된 파라미터는 옵티마이저에 넣지 않는다 — 넣으면 상태 메모리를 그만큼 더 쓴다."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cond_stage_trainable:
            params += [p for p in self.cond_stage_model.parameters() if p.requires_grad]
        if self.image_proj_model_trainable:
            params += list(self.image_proj_model.parameters())
        if self.learn_logvar:
            params.append(self.logvar)
        return params

    def setup(self, stage=None):
        """중간 체크포인트에서 이어받기.

        `get_model` 이 `pretrained_checkpoint`(backbone)를 **`__init__` 뒤에** 싣기 때문에
        `__init__` 에서 실으면 덮어써진다. `setup` 은 그 뒤·옵티마이저 생성 앞이라 안전하다.
        체크포인트는 `save_only_unet: True` 라 `model.*` 만 들어 있고, VAE·CLIP 은 backbone 것을 쓴다.
        옵티마이저 상태는 저장돼 있지 않으므로 **Adam 은 새로 시작**한다(목적함수가 바뀌었으니 오히려 낫다).
        """
        super().setup(stage)
        if not self.resume_unet or getattr(self, "_resumed", False):
            return
        sd = torch.load(self.resume_unet, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f">>> [base] 이어받기: {self.resume_unet} · 실림 {loaded}/{len(sd)} "
              f"(안 실린 키 {len(unexpected)}개)", flush=True)
        self._resumed = True

    def _motion_weight_map(self, x_start: torch.Tensor) -> torch.Tensor:
        """x_start: [b,c,T,h,w] (GT 잠재) → 같은 shape 로 브로드캐스트되는 가중치 [b,1,T,h,w].

        **왜.** 팔은 화면의 2~5% 뿐이라 지워버려도 손실이 거의 안 는다. 반대로 엉뚱한 자리에
        그리면 두 배로 틀린다(있어야 할 곳에 없고, 없어야 할 곳에 있음) — 그래서 학습이
        진행될수록 **팔을 지우는 것이 이득**이 되고, 실제로 600스텝에서 팔이 사라졌다.
        GT 에서 시간에 따라 변하는 자리에 가중치를 줘서 지우는 선택을 비싸게 만든다.

        가중치 평균을 1 로 맞춰 전체 손실 크기를 보존한다 — 안 그러면 유효 학습률이 같이 바뀐다.
        """
        d = (x_start - x_start[:, :, :1]).abs().mean(1, keepdim=True)      # [b,1,T,h,w]
        d = d / (d.amax(dim=(2, 3, 4), keepdim=True) + 1e-8)               # 샘플별 0~1
        w = 1.0 + self.motion_weight * d
        return w / w.mean(dim=(1, 2, 3, 4), keepdim=True)

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        """킷 원본과 동일하되 손실을 움직임 가중치로 재가중한다(`motion_weight` 0 이면 원본 그대로)."""
        if self.motion_weight <= 0:
            return super().p_losses(x_start, cond, t, noise=noise, **kwargs)

        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = noise if noise is not None else torch.randn_like(x_start) + self.noise_strength * offset_noise
        else:
            noise = noise if noise is not None else torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output, info = self.apply_model(x_noisy, t, cond, **kwargs)

        loss_dict = {}
        prefix = "train" if self.training else "val"
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        w = self._motion_weight_map(x_start)
        per_elem = self.get_loss(model_output, target, mean=False) * w
        loss_simple = per_elem.mean([1, 2, 3, 4])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        if self.logvar.device is not self.device:
            self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})
        loss = self.l_simple_weight * loss.mean()

        loss_vlb = (self.lvlb_weights[t] * per_elem.mean(dim=(1, 2, 3, 4))).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})
        return loss, loss_dict, info
