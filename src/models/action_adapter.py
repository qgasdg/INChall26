"""액션 조건 주입 — IRASim SO100 임베더 교체 + 정렬 (이식: impl/w2/action_adapter.py).

설계 결정(docs/04):
  - IRASim 원본 액션 임베더(EE 7-DoF)를 **SO100 6관절**용으로 교체. 출력층 **zero-init**(초기 기여 0 → prior 보존).
  - frame-level 조건(16액션 ↔ 16프레임 1:1, IRASim 근거). shifted 정렬(frames1..15 ← act0..14).
  - 조건 프레임(t=0)엔 액션 없음(mask_emb 조건). mask_frame_num(cfg.model)으로 조건 프레임 고정.
"""
from __future__ import annotations

ACTION_DIM = 6  # SO100 관절 수
ALIGN_SHIFTED, ALIGN_SAME = "shifted", "same_index"


def adapt_action_seq(actions, num_frames: int = 16, align_mode: str = ALIGN_SHIFTED):
    """index-aligned 16액션 → 프레임레벨 조건용 15액션. 입력(...,16,6) → 출력(...,15,6). (impl/w2 이식)

    - shifted   : frames 1..15 ← actions[0..14]  (물리 가설 action[t]≈state[t+1] 정합, 기본)
    - same_index: frames 1..15 ← actions[1..15]  (extractor 페어링 가설, 대조군)
    """
    if align_mode == ALIGN_SHIFTED:
        return actions[..., : num_frames - 1, :]      # 0..14
    if align_mode == ALIGN_SAME:
        return actions[..., 1:num_frames, :]          # 1..15
    raise ValueError(f"알 수 없는 align_mode: {align_mode} (shifted|same_index)")


def build_so100_state_embedder(hidden_size: int, state_dim: int = ACTION_DIM, zero_init: bool = True):
    """IRASim embed_state 와 동일 구조(timm Mlp)의 SO100 임베더. 출력층 zero-init. (impl/w2 이식)"""
    import torch.nn as nn
    from timm.models.vision_transformer import Mlp

    approx_gelu = lambda: nn.GELU(approximate="tanh")
    mlp = Mlp(in_features=state_dim, hidden_features=hidden_size * 4,
              out_features=hidden_size, act_layer=approx_gelu, drop=0)
    if zero_init:
        nn.init.zeros_(mlp.fc2.weight)
        nn.init.zeros_(mlp.fc2.bias)
    return mlp


def swap_action_embedder(model, state_dim: int = ACTION_DIM, zero_init: bool = True, logger=None):
    """구성된 IRASim 모델의 embed_state 를 SO100 임베더로 교체(in-place). model 반환. (impl/w2 이식)

    - model.hidden_size 로 임베더 폭 산출. 사전학습 ckpt는 이 교체 **이전에** strict=False 로 로드
      (7-dim embed_state 는 shape 불일치라 무시 → 여기서 6-dim 재초기화). 벤더 무수정 — 인스턴스 모듈만 교체.
    """
    if not hasattr(model, "embed_state"):
        raise AttributeError("model.embed_state 없음 — extras==3 조건 모델인지 확인")
    hidden = getattr(model, "hidden_size", None)
    if hidden is None:
        raise AttributeError("model.hidden_size 없음")
    model.embed_state = build_so100_state_embedder(hidden, state_dim, zero_init)
    if hasattr(model, "state_dim"):
        model.state_dim = state_dim
    if logger:
        logger.info("embed_state 교체: in=%d hidden=%d zero_init=%s", state_dim, hidden, zero_init)
    return model
