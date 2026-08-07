"""지정한 이름 패턴만 남기고 전부 동결하는 콜백.

**왜 필요한가.** 지금까지의 제출 이력은 예외 없이 "학습하면 나빠진다"를 가리킨다
(0스텝 0.3031 이 500·1250·1750스텝 전부보다 낫다). 그런데 그것이
① 파인튜닝이 사전학습 prior 를 망가뜨려서인지 ② 액션 신호 자체가 손해라서인지
**아직 분리된 적이 없다.**

백본을 물리적으로 동결하고 `action_embed`(zero-init, 사전학습에 없던 신규 모듈)만
학습하면 ①이 원천 차단되므로 ②만 남는다.

메모리도 이롭다. AdamW 는 `p.grad is None` 인 파라미터를 건너뛰므로
동결된 1.4B 에 대해서는 옵티마이저 상태를 만들지 않는다.
"""
from __future__ import annotations

import pytorch_lightning as pl


class FreezeExceptAction(pl.Callback):
    def __init__(self, patterns=("action_embed",), verbose: bool = True):
        super().__init__()
        self.patterns = tuple(patterns)
        self.verbose = verbose

    def setup(self, trainer, pl_module, stage=None):
        # configure_optimizers 보다 먼저 불린다 — 그래야 옵티마이저가 동결 상태를 본다
        kept, frozen = 0, 0
        kept_names = []
        for name, p in pl_module.model.named_parameters():
            if any(pat in name for pat in self.patterns):
                p.requires_grad = True
                kept += p.numel()
                kept_names.append(name)
            else:
                p.requires_grad = False
                frozen += p.numel()
        if self.verbose:
            total = kept + frozen
            print(f"[freeze] 학습 {kept/1e6:.2f}M / 전체 {total/1e6:.2f}M "
                  f"({kept/total*100:.3f}%) · 동결 {frozen/1e6:.2f}M")
            for n in kept_names:
                print(f"[freeze]   학습 대상: {n}")
        if kept == 0:
            raise ValueError(f"패턴 {self.patterns} 에 걸린 파라미터가 없다 — 이름을 확인하라.")
