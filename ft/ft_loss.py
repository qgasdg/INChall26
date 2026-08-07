"""액션 전용 학습 모델 — 백본 동결 + (선택) 액션 대조 손실.

동결·zero-init 을 **모델 `__init__` 안에서** 한다. 콜백으로 걸었더니 실행되지 않아
1.4B 전체가 학습되는 사고가 났다(2026-08-08). 여기서는 모델이 만들어지는 순간
반드시 실행되고, 로그도 flush 해서 남긴다.

두 클래스:
  ActionOnlyDiffusion        기본 손실 + 백본 동결 + action_embed zero-init
  ActionContrastiveDiffusion 위에 액션 대조 손실을 더한 것

**왜 동결인가.** 제출 이력은 예외 없이 "학습하면 나빠진다"를 가리키는데, 그것이
① 파인튜닝이 사전학습 prior 를 망가뜨려서인지 ② 액션 경로 자체가 무력해서인지
분리된 적이 없다. 백본이 동결이면 ①이 원천 차단되므로 ②만 남는다.

**왜 zero-init 인가.** 킷 원본 UNet 은 `action_embed` 를 무작위로 초기화한다. 그러면
학습 0스텝부터 의미 없는 신호가 시간 임베딩에 더해진다(ft-01 = 0.3729 가 그 상태였다).
마지막 층을 0 으로 두면 0스텝이 검증된 제로샷과 **수치적으로 동일**해지고,
액션 영향력이 0 에서부터 자란다.

**왜 대조 손실인가.** 기본 손실은 잠재 전체(4×16×40×64 = 163,840개)의 MSE 평균이라
배경 복원이 기울기를 독식한다. 팔은 화면의 몇 %뿐이다. motion-weighted(v2, 0.37619)와
anchor(v3, 개선 없음)는 "어디가 변하나"만 다뤘지 "액션과 그 변화가 맞물리나"는 안 물었다.
대조 손실은 화질을 올려서는 못 줄인다 — 두 예측이 같이 좋아지면 차이가 그대로다.
오직 출력이 액션에 **올바르게 의존해야만** 줄어든다. 배경은 두 예측에서 상쇄된다.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lvdm.models.ddpm3d import LatentVisualDiffusion


def shuffle_time(act: torch.Tensor) -> torch.Tensor:
    """[b, t, d] 의 시간 순서를 섞는다. 항등 순열이면 한 칸 굴려 항상 달라지게 한다.

    배치 축이 아니라 **시간 축**을 섞는다 — batch_size 1 에서도 동작해야 하고,
    "같은 액션 집합, 다른 순서"라야 대조가 순서 정보만 겨냥한다.
    """
    b, t, _ = act.shape
    out = torch.empty_like(act)
    ar = torch.arange(t, device=act.device)
    for i in range(b):
        perm = torch.randperm(t, device=act.device)
        if torch.equal(perm, ar):
            perm = torch.roll(perm, 1)
        out[i] = act[i, perm]
    return out


class ActionOnlyDiffusion(LatentVisualDiffusion):
    def __init__(self, *args,
                 train_patterns=("action_embed",),
                 zero_init_action: bool = True,
                 freeze_backbone: bool = True,
                 contrast_weight: float = 0.0,
                 contrast_margin: float = 0.05,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.train_patterns = tuple(train_patterns)
        self.contrast_weight = contrast_weight
        self.contrast_margin = contrast_margin
        if zero_init_action:
            self._zero_init_action()
        if freeze_backbone:
            self._freeze_backbone()

    # ── 초기화 ────────────────────────────────────────────────────────────
    def _zero_init_action(self):
        ae = getattr(self.model.diffusion_model, "action_embed", None)
        if ae is None:
            raise ValueError("action_embed 가 없다 — action_conditioned: True 인지 확인하라.")
        last = [m for m in ae if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        print("[ft] action_embed 마지막 층 0 초기화 — 0스텝 = 제로샷과 수치적으로 동일", flush=True)

    def _freeze_backbone(self):
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
        print(f"[ft] 동결 완료 — 학습 {kept/1e6:.2f}M / 전체 {total/1e6:.2f}M "
              f"({kept/total*100:.3f}%) · 동결 {frozen/1e6:.2f}M", flush=True)
        for n in names:
            print(f"[ft]   학습 대상: {n}", flush=True)
        if kept == 0:
            raise ValueError(f"패턴 {self.train_patterns} 에 걸린 파라미터가 없다.")

    def configure_optimizers(self):
        """동결된 것을 옵티마이저에서 아예 뺀다 (AdamW 상태 절약 + 실수 방지)."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[ft] 옵티마이저 대상 텐서 {len(params)}개 · lr={self.learning_rate}", flush=True)
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)

        def lr_lambda(step: int) -> float:
            return min(1.0, step / self.linear_warmup_steps)

        scheduler = {"scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "interval": "step"}
        return [optimizer], [scheduler]

    # ── 손실 ──────────────────────────────────────────────────────────────
    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        # 아래 기본 손실부는 부모(ddpm3d.LatentDiffusion.p_losses)와 같다.
        # super() 를 부르면 forward 가 한 번 더 돌아 3회가 되므로 여기서 펼친다.
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

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        if self.logvar.device is not self.device:
            self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t]
        loss = self.l_simple_weight * (loss_simple / torch.exp(logvar_t) + logvar_t).mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss = loss + self.original_elbo_weight * loss_vlb

        # ── 액션 대조 항 ──────────────────────────────────────────────────
        act = cond.get("act", None) if isinstance(cond, dict) else None
        if self.contrast_weight > 0 and act is not None:
            cond_wrong = dict(cond)
            cond_wrong["act"] = shuffle_time(act)
            # 같은 x_noisy · 같은 t · 같은 노이즈 — 오직 액션 순서만 다르다
            out_wrong, _ = self.apply_model(x_noisy, t, cond_wrong, **kwargs)
            mse_wrong = self.get_loss(out_wrong, target, mean=False).mean([1, 2, 3, 4])

            # 상대 격차: 0 이면 액션 무시, >0 이면 정답 액션이 유리
            gap = (mse_wrong - loss_simple) / (mse_wrong.detach() + 1e-8)
            l_contrast = torch.relu(self.contrast_margin - gap).mean()
            loss = loss + self.contrast_weight * l_contrast
            loss_dict.update({
                f"{prefix}/contrast_gap": gap.mean(),        # ★ 액션이 먹히는 정도
                f"{prefix}/contrast_loss": l_contrast,
                f"{prefix}/mse_wrong": mse_wrong.mean(),
            })

        loss_dict.update({f"{prefix}/loss": loss})
        return loss, loss_dict, info


class ActionContrastiveDiffusion(ActionOnlyDiffusion):
    """대조 항을 기본으로 켠 판. config 에서 contrast_weight 를 주면 그 값이 쓰인다."""

    def __init__(self, *args, contrast_weight: float = 1.0, **kwargs):
        super().__init__(*args, contrast_weight=contrast_weight, **kwargs)
