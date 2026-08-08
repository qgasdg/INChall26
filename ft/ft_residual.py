"""잔차 모델 — **학습 0스텝에서 출력이 정확히 정적**이고, 학습이 거기에 움직임을 더한다.

    출력 잠재 = z_정적 + Δ,      z_정적 = z[:, :, 0:1] 을 16번 반복
    Δ = UNet(z_정적, t=0, 조건, 액션)   ← 마지막 conv 를 0 으로 초기화

`perframe_ae: True` 라 VAE 가 프레임을 하나씩 인코딩하므로 `z[:, :, 0:1]` 반복이
정적 영상의 잠재와 **정확히 일치**한다(근사가 아니다).

**왜 이렇게 바꾸나.** 세 가지가 한꺼번에 풀린다.

1. **출발점이 측정된 최고 점수다.** 정적은 0.3019 로 우리가 가진 어떤 결과보다 좋다.
   Δ=0 이면 출력이 그것과 동일하므로, 학습은 0.30 을 바닥에 깔고 시작한다.
   지금까지는 실험할 때마다 0.39~0.49 로 굴러떨어져 개선 여부가 안 보였다.
2. **배경이 목표에서 사라진다.** 목표가 `Δ = z_정답 − z_정적` 인데 배경은 원래 안 움직이니
   거의 0 이다. "손실이 잠재 전체 163,840개의 평균이라 배경이 기울기를 독식한다"는
   문제가 우회가 아니라 정면으로 없어진다. motion-weighted(가중치)·anchor(벌점) 가 못 푼 지점이다.
3. **모든 움직임의 출처가 하나다.** 백본의 상상이 아니라 Δ 뿐이고, Δ 는 액션을 본다.

**성격이 확산에서 회귀로 바뀐다.** 노이즈에서 뽑는 생성이 아니라 "첫 프레임 + 액션 → 움직임"을
맞히는 예측이 된다. 이 과제는 정답이 사실상 하나로 결정돼 있어 생성 다양성이 이득이 아니다.
부작용은 회귀 특유의 뿌연 평균인데, 관측 ⑥(선명할수록 점수가 나빴다)을 보면 이 판에서는 불리하지 않다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lvdm.models.ddpm3d import LatentVisualDiffusion


def static_latent(z: torch.Tensor) -> torch.Tensor:
    """[b, c, t, h, w] → 첫 프레임을 t 번 반복한 잠재. perframe_ae 라 정적 영상의 잠재와 일치."""
    return z[:, :, :1].expand_as(z).contiguous()


class ActionResidualDiffusion(LatentVisualDiffusion):
    """UNet 을 잔차 예측기로 쓴다. 마지막 conv zero-init → 0스텝 출력 = 정적.

    train_patterns 에 걸린 것만 학습한다. 기본값은 액션 임베딩 + 출력 conv 다
    (출력 conv 는 zero-init 했으므로 반드시 학습 대상이어야 한다).
    """

    def __init__(self, *args,
                 train_patterns=("action_embed", "diffusion_model.out."),
                 freeze_backbone: bool = True,
                 temporal_diff_loss: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.train_patterns = tuple(train_patterns)
        self.temporal_diff_loss = temporal_diff_loss
        if freeze_backbone:
            self._freeze()
        # ★ zero-init 은 여기서 하면 안 된다 — get_model 이 이 뒤에 backbone.ckpt 를 실으면서
        #   out.* 를 사전학습 값으로 덮어쓴다(action_embed 는 체크포인트에 없어 살아남지만).
        #   반드시 적재 **이후에** zero_init() 을 호출해야 한다.

    # ── 초기화 ────────────────────────────────────────────────────────────
    def zero_init(self):
        """UNet 출력 conv 를 0 으로 → Δ ≡ 0 → 출력이 정확히 정적."""
        out = self.model.diffusion_model.out
        convs = [m for m in out.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d)]
        if not convs:
            raise ValueError("UNet out 에서 conv 를 못 찾았다.")
        last = convs[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)
        print(f"[ft] 출력 conv 0 초기화 {tuple(last.weight.shape)} — Δ=0, 0스텝 출력 = 정적", flush=True)

        # 액션 임베딩도 0 에서 출발시킨다(있으면)
        ae = getattr(self.model.diffusion_model, "action_embed", None)
        if ae is not None:
            lin = [m for m in ae if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            print("[ft] action_embed 마지막 층 0 초기화", flush=True)

    def _freeze(self):
        kept, frozen, names = 0, 0, []
        for name, p in self.model.named_parameters():
            if any(pat in name for pat in self.train_patterns):
                p.requires_grad = True
                kept += p.numel()
                names.append(name)
            else:
                p.requires_grad = False
                frozen += p.numel()
        total = kept + frozen
        print(f"[ft] 동결 완료 — 학습 {kept/1e6:.3f}M / 전체 {total/1e6:.2f}M "
              f"({kept/total*100:.4f}%)", flush=True)
        for n in names:
            print(f"[ft]   학습 대상: {n}", flush=True)
        if kept == 0:
            raise ValueError(f"패턴 {self.train_patterns} 에 걸린 파라미터가 없다.")

    def on_fit_start(self):
        """체크포인트 적재가 끝난 뒤에 0 초기화한다 (순서가 중요하다)."""
        self.zero_init()

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[ft] 옵티마이저 대상 텐서 {len(params)}개 · lr={self.learning_rate}", flush=True)
        opt = torch.optim.AdamW(params, lr=self.learning_rate)

        def lr_lambda(step: int) -> float:
            return min(1.0, step / self.linear_warmup_steps)

        sched = {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda), "interval": "step"}
        return [opt], [sched]

    # ── 예측 ──────────────────────────────────────────────────────────────
    def predict_residual(self, z_static, cond, **kwargs):
        """Δ 를 한 번의 forward 로 얻는다 (디노이징 반복 없음)."""
        t = torch.zeros(z_static.shape[0], dtype=torch.long, device=z_static.device)
        delta, _ = self.apply_model(z_static, t, cond, **kwargs)
        return delta

    # ── 손실 ──────────────────────────────────────────────────────────────
    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        """확산 손실이 아니라 잔차 회귀 손실. t·noise 는 쓰지 않는다.

        temporal_diff_loss=True 면 **시간 차분**에 손실을 건다.

        왜: 목표 `Δ = z_정답 − z_정적` 의 시간 평균은 0 이 아니다(조명·모션블러·팔의 평균 위치).
        그래서 **시간에 대해 상수인 값 하나만 맞혀도 MSE 가 10% 준다.** ft-20 600스텝이
        정확히 그렇게 됐다 — 전역 오프셋 3.2/255 vs 프레임 간 변화 0.8/255 로,
        Δ 의 대부분이 "영상 전체를 살짝 미는 것"이었고 팔은 안 움직였다.
        차분을 쓰면 상수 성분이 소거되므로 **오프셋으로는 손실을 1도 못 줄인다.**
        점수를 따려면 실제로 프레임이 달라져야 한다.
        """
        z_static = static_latent(x_start)
        target = x_start - z_static                      # 배경은 거의 0
        delta = self.predict_residual(z_static, cond, **kwargs)

        if self.temporal_diff_loss:
            # 시간 차분: [b,c,t,h,w] → [b,c,t-1,h,w]. 상수 성분은 여기서 사라진다
            d_pred = delta[:, :, 1:] - delta[:, :, :-1]
            d_true = target[:, :, 1:] - target[:, :, :-1]
            loss_flat = self.get_loss(d_pred, d_true, mean=False).mean([1, 2, 3, 4])
            static_ref = d_true.pow(2).mean([1, 2, 3, 4])      # 정적(=차분 0)일 때의 오차
        else:
            loss_flat = self.get_loss(delta, target, mean=False).mean([1, 2, 3, 4])
            static_ref = target.pow(2).mean([1, 2, 3, 4])

        loss = loss_flat.mean()

        prefix = "train" if self.training else "val"
        with torch.no_grad():
            static_err = static_ref.mean()
            rel = loss / (static_err + 1e-8)
            # Δ 를 시간 상수 성분과 변동 성분으로 쪼개 본다 — 착시를 막는 진단값
            const_part = delta.mean(dim=2, keepdim=True)
            vary_part = delta - const_part
            const_rms = const_part.pow(2).mean().sqrt()
            vary_rms = vary_part.pow(2).mean().sqrt()
        loss_dict = {
            f"{prefix}/loss": loss,
            f"{prefix}/loss_simple": loss,
            f"{prefix}/static_err": static_err,     # 정적 기준선
            f"{prefix}/rel_to_static": rel,         # ★ <1 이면 정적보다 낫다
            f"{prefix}/delta_rms": delta.pow(2).mean().sqrt(),
            f"{prefix}/delta_const_rms": const_rms,  # 시간 상수 성분 (오프셋)
            f"{prefix}/delta_vary_rms": vary_rms,    # ★ 시간 변동 성분 = 진짜 움직임
        }
        return loss, loss_dict, {}
