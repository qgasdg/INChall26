"""생성 백본 로드 — IRASim (이식: impl/w2/irasim_runtime.py → src/models/irasim_runtime.py 위임).

설계 결정(docs/04):
  - 백본 = IRASim(벤더 third_party/irasim, DiT). VAE = SDXL. 액션 조건 = SO100 임베더 교체(zero-init).
  - 실제 모델 구성/생성/학습손실은 `src/models/irasim_runtime.py`(Runtime). 이 파일은 스캐폴드 진입점(얇은 어댑터).

★실행 요건: diffusers(INChall26 기본 .venv 미포함 → 설치 필요) + GPU + IRASim 슬림 ckpt.
"""
from __future__ import annotations


def load_backbone(cfg: dict, device: str = "cuda", slim_ckpt: str | None = None, logger=None):
    """cfg → IRASim DiT + SDXL VAE + 디퓨전 Runtime. irasim_runtime.build 위임.

    반환: irasim_runtime.Runtime(model, vae, diffusion, cfg, device, dtype, ...).
    """
    from src.models.irasim_runtime import build

    return build(cfg, slim_ckpt=slim_ckpt, logger=logger)


def apply_finetune_mode(rt, cfg: dict, logger=None):
    """FT 범위 적용 — full(전체 학습)·partial(일부 언프리즈)·lora(어댑터). rt.model in-place.

    full 은 즉시 적용; partial/lora 는 대상 모듈 선택 로직 필요(TODO).
    """
    mode = (cfg or {}).get("model", {}).get("finetune", {}).get("mode", "partial")
    if mode == "full":
        for p in rt.model.parameters():
            p.requires_grad_(True)
        if logger:
            logger.info("finetune=full: 전체 파라미터 학습")
        return rt
    raise NotImplementedError(f"apply_finetune_mode: mode={mode!r} 미구현 (partial=일부 언프리즈 / lora=어댑터 부착)")
