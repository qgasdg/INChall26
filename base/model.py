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

from torch import nn

from lvdm.models.ddpm3d import LatentVisualDiffusion
from lvdm.modules import attention as _attn
from lvdm.modules.networks.openaimodel3d import UNetModel

_PATCHED = False

# 킷이 하드코딩한 텍스트 토큰 수 (openaimodel3d.py:719 의 `77 + t*16` 검사와 같은 값)
TEXT_CONTEXT_LEN = 77


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


class ActionCtxUNet(UNetModel):
    """액션을 **cross-attention 으로도** 넣기 위한 토큰 생성기를 UNet 에 붙인다.

    **왜.** 기존 경로는 `emb = time_emb + act_emb` 라 프레임당 벡터 하나가 공간 전체에
    균일하게 퍼진다 — "얼마나"는 말해도 **"어디를"은 표현할 자리가 없다**(dacon_submission ⑦).
    cross-attention 은 공간 위치마다 자기 주의 가중치를 계산하므로 지목이 가능하다.

    **어디에 싣나.** Resampler 가 `num_queries × video_length = 16 × 16 = 256` 개의
    **이미 프레임별인** 이미지 토큰을 낸다(`resampler.py:118`, 주석 `B (T L) C`). UNet 은
    `context` 길이가 `77 + t*16` 이면 이를 프레임별로 쪼갠다(`openaimodel3d.py:719`).
    그 슬롯에 액션 토큰을 **더한다** — 길이가 그대로라 킷 코드를 고칠 필요가 없고,
    무조건부 분기(`uc`)에는 액션이 안 들어가므로 **액션 CFG 가 그대로 성립**한다.

    ★2026-08-10 에 여기서 틀렸다. 이미지 토큰이 16개인 줄 알고 `ctx[:, -16:]` 만 이미지로,
    앞 317개를 텍스트로 잘랐다 — context 길이가 333 → 573 으로 망가진 채 base-05·06·07 이
    학습됐다. 텍스트는 항상 앞 77 개이고 나머지가 전부 이미지다.

    **왜 UNet 에 두나.** `save_only_unet: True` 라 `self.model` 아래가 아니면 체크포인트에
    저장되지 않는다. 확산 모델 쪽에 두면 학습해놓고 잃어버린다.

    마지막 층은 zero-init 이라 **학습 시작 시점의 출력이 기존과 수치적으로 동일**하다.
    """

    def __init__(self, *args, action_ctx_tokens: int = 16, action_ctx_dim: int = 1024,
                 action_ctx_hidden: int = 512, action_txt_tokens: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        d_in = kwargs.get("action_dims", 6)
        self.action_ctx_tokens = action_ctx_tokens
        self.action_ctx_dim = action_ctx_dim
        self.action_txt_tokens = action_txt_tokens
        self.action_ctx = nn.Sequential(
            nn.Linear(d_in, action_ctx_hidden),
            nn.SiLU(),
            nn.Linear(action_ctx_hidden, action_ctx_tokens * action_ctx_dim),
        )
        nn.init.zeros_(self.action_ctx[-1].weight)
        nn.init.zeros_(self.action_ctx[-1].bias)
        n = sum(p.numel() for p in self.action_ctx.parameters())
        msg = f">>> [base] 프레임별 액션 토큰 {action_ctx_tokens}개/프레임 ({n/1e6:.1f}M, zero-init)"

        if action_txt_tokens:
            # 궤적 전체를 통째로 눌러 토큰 몇 개로. 프레임 공용인 텍스트 슬롯에 들어간다 —
            # "이 클립이 어떤 동작인가"는 프레임마다 달라질 필요가 없다.
            # LazyLinear 은 첫 forward 전까지 파라미터가 없어 configure_optimizers 에서 터진다.
            t_len = kwargs.get("temporal_length", 16)
            self.action_txt = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(t_len * d_in, action_ctx_hidden),
                nn.SiLU(),
                nn.Linear(action_ctx_hidden, action_txt_tokens * action_ctx_dim),
            )
            nn.init.zeros_(self.action_txt[-1].weight)
            nn.init.zeros_(self.action_txt[-1].bias)
            msg += f" · 궤적요약 토큰 {action_txt_tokens}개(텍스트 슬롯, zero-init)"
        print(msg, flush=True)

    def action_tokens(self, act):
        """act: [b, t, d] → [b, t, K, C] — 프레임별"""
        b, t, _ = act.shape
        return self.action_ctx(act).view(b, t, self.action_ctx_tokens, self.action_ctx_dim)

    def action_summary_tokens(self, act):
        """act: [b, t, d] → [b, M, C] — 클립 전체를 요약한 프레임 공용 토큰"""
        b = act.shape[0]
        return self.action_txt(act).view(b, self.action_txt_tokens, self.action_ctx_dim)


# freeze_scope → 동결할 UNet 하위 모듈 이름
_SCOPES = {
    "none": (),                                  # 전체 학습 (1440M)
    "encoder": ("input_blocks",),                # 디코더+middle+기타만 (994M)
    "encoder_middle": ("input_blocks", "middle_block"),   # 디코더+기타만 (818M)
}


class BaseLatentVisualDiffusion(LatentVisualDiffusion):
    def __init__(self, *args, freeze_scope: str = "none",
                 motion_weight: float = 0.0, resume_unet: str | None = None,
                 optimizer: str = "adamw", **kwargs):
        enable_sdpa_attention()
        super().__init__(*args, **kwargs)
        if freeze_scope not in _SCOPES:
            raise ValueError(f"freeze_scope={freeze_scope!r} — {list(_SCOPES)} 중 하나여야 한다")
        self.freeze_scope = freeze_scope
        self.motion_weight = float(motion_weight)
        self.resume_unet = resume_unet
        self.optimizer_name = optimizer

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

    def configure_optimizers(self):
        """`optimizer: adamw8bit` 이면 8비트 Adam 을 쓴다.

        **왜.** AdamW 는 파라미터당 상태 2개를 fp32 로 들고 있어 1.44B 전체를 학습하면
        그것만 10.7GB 다. 32GB 에서 전체 미세조정이 **50MB 모자라** 죽었던 주범이다.
        8비트로 두면 2.7GB 로 줄어 인코더까지 풀 수 있다. 확산모델 미세조정에서
        널리 쓰이는 방식이고 품질 손해는 거의 보고되지 않는다.

        스케줄러(선형 워밍업)는 킷 원본과 같게 유지한다.
        """
        if self.optimizer_name != "adamw8bit":
            return super().configure_optimizers()

        import bitsandbytes as bnb
        params = self.get_param_list()
        n = sum(p.numel() for p in params)
        opt = bnb.optim.AdamW8bit(params, lr=self.learning_rate)
        print(f">>> [base] AdamW8bit · 학습 {n/1e6:.1f}M · 옵티마이저 상태 {n*2/2**30:.1f}GB "
              f"(fp32 였다면 {n*8/2**30:.1f}GB)", flush=True)

        def lr_lambda(step: int) -> float:
            return min(1.0, step / self.linear_warmup_steps)

        return [opt], [{"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
                        "interval": "step"}]

    def get_batch_input(self, batch, random_uncond, **kwargs):
        """킷이 만든 `c_crossattn` 을 프레임별로 펼치고 액션 토큰을 더한다.

        원본은 `[b, 77+16, C]`(텍스트 77 + 이미지 16, 프레임 공용)이다. 이를
        `[b, 77 + t*16, C]` 로 바꾸면 UNet 이 이미지 토큰을 프레임별로 쪼개주므로
        (`openaimodel3d.py:719`), 그 자리에 프레임별 액션 토큰을 실을 수 있다.

        zero-init 이라 학습 시작 시점에는 원본과 값이 같다 — 프레임마다 같은 이미지 토큰이
        복제될 뿐이고, 그건 원본의 `repeat_interleave` 와 동일하다.
        """
        out = super().get_batch_input(batch, random_uncond, **kwargs)
        unet = self.model.diffusion_model
        if not getattr(unet, "action_ctx_tokens", 0):
            return out

        z, cond = out[0], out[1]
        ctx = cond["c_crossattn"][0]                       # [b, 77 + t*L, C]
        b, _, C = ctx.shape
        t = z.shape[2]
        text, img = ctx[:, :TEXT_CONTEXT_LEN], ctx[:, TEXT_CONTEXT_LEN:]
        L = img.shape[1] // t                              # 프레임당 이미지 토큰 수 (Resampler 16)
        if L * t != img.shape[1]:
            raise RuntimeError(f"이미지 토큰 {img.shape[1]}개가 프레임 {t}개로 안 나뉜다")
        if L != unet.action_ctx_tokens:
            raise RuntimeError(f"action_ctx_tokens={unet.action_ctx_tokens} 인데 "
                               f"프레임당 이미지 토큰은 {L}개다 — 같아야 더할 수 있다")

        tok = unet.action_tokens(cond["act"])              # [b, t, L, C]
        keep = None
        if self.training and unet.action_dropout_prob > 0:
            keep = (torch.rand(b, device=tok.device) >= unet.action_dropout_prob)
            tok = tok * keep.view(b, 1, 1, 1)              # 떨어뜨린 표본은 액션 없음과 같아진다
        img = img.view(b, t, L, C) + tok

        if unet.action_txt_tokens:
            # 궤적 요약은 텍스트 슬롯 **뒤쪽**에 더한다. 빈 캡션이라 어차피 의미가 없는 자리이고,
            # train·eval 모두 빈 문자열이라 분포가 어긋나지 않는다.
            M = unet.action_txt_tokens
            summ = unet.action_summary_tokens(cond["act"])  # [b, M, C]
            if keep is not None:
                summ = summ * keep.view(b, 1, 1)            # 두 경로를 같이 떨어뜨려야 uncond 가 깨끗하다
            text = torch.cat([text[:, :-M], text[:, -M:] + summ], dim=1)

        cond["c_crossattn"] = [torch.cat([text, img.reshape(b, t * L, C)], dim=1)]
        return out

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
