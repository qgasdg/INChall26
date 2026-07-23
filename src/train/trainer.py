"""학습 루프 — IRASim 파인튜닝 (이식: impl/w2/train_w2.py → INChall26 config/데이터 배선).

- FT: full | partial(embed_state + final_layer + 상위 K블록) | lora(미이식 — impl/w2/lora.py)
- 손실: IRASim 디퓨전 손실(pre_encode latent 또는 on-the-fly video). 후반프레임 가중 옵션(exp06)
- ckpt: 원자적 저장 + keep-last N 회전 + best 1(det_loss 최저), 재개
- eval: eval-every 마다 결정론 det_loss(고정 subset·t격자·seed) → local_lb.csv, best 판정
  (전체 DV_w 생성채점은 별도 서브시스템 — infer/generate + eval/local_score 연동은 후속)

★실행 요건: diffusers + GPU. pre_encode 학습은 build_latent_cache 산출 필요(없으면 on-the-fly ClipDataset 폴백).
"""
from __future__ import annotations

import time
from pathlib import Path

from src.data.dataset import (SO100ClipDataset, SO100LatentDataset, build_sampler,
                              build_window_index)

_EVAL_CACHE: dict = {}


def _g(cfg: dict, path: str, default=None):
    node = cfg
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def select_trainable(model, ft_mode: str, top_blocks: int = 4, logger=None):
    """ft_mode별 requires_grad. partial: embed_state+final_layer+상위K블록. (impl/w2 이식)"""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []

    def unfreeze(mod):
        for p in mod.parameters():
            p.requires_grad_(True); trainable.append(p)

    if ft_mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = list(model.parameters())
    elif ft_mode == "partial":
        unfreeze(model.embed_state)                          # 교체된 SO100 임베더(zero-init → 반드시 학습)
        if hasattr(model, "final_layer"):
            unfreeze(model.final_layer)
        for b in list(model.blocks)[-top_blocks:]:           # 상위 K 블록(도메인 적응)
            unfreeze(b)
    elif ft_mode == "lora":
        raise NotImplementedError("lora: impl/w2/lora.py(inject_lora) 이식 후 활성")
    else:
        raise ValueError(ft_mode)
    if logger:
        logger.info("FT=%s 학습 텐서 %d, params %d", ft_mode, len(trainable), sum(p.numel() for p in trainable))
    return trainable


def frame_weights(num_frames: int, w_last: float, device):
    """후반프레임 가중(exp06): frame0=1 → 선형 증가 → 마지막=w_last (평균 1 정규화)."""
    import torch

    w = torch.linspace(1.0, w_last, num_frames, device=device)
    return w / w.mean()


def _step_loss(rt, batch, post_w: float):
    """latent(pre_encode) 또는 video(on-the-fly) → IRASim 디퓨전 손실(후반프레임 가중 옵션)."""
    import torch

    from src.models.irasim_runtime import encode_video

    latents = batch["latent"].to(rt.device) if "latent" in batch else encode_video(rt, batch["video"])
    mk = dict(actions=batch["action_cond"].to(rt.device), mask_frame_num=rt.cfg.mask_frame_num)
    t = torch.randint(0, rt.diffusion.num_timesteps, (latents.shape[0],), device=rt.device)
    out = rt.diffusion.training_losses(rt.model, latents, t, mk)
    if post_w > 1.0 and "loss_perframe" in out:
        return (out["loss_perframe"] * frame_weights(latents.shape[1], post_w, rt.device)).mean()
    return out["loss"].mean()


def atomic_save(state: dict, path: Path):
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _make_dataset(cfg: dict, clips, logger=None):
    """pre_encode + 캐시 존재 → LatentDataset(빠름), 아니면 ClipDataset(on-the-fly)."""
    cache = _g(cfg, "paths.latent_cache", "latent_cache")
    if _g(cfg, "data.pre_encode", True) and (Path(cache) / "manifest.json").exists():
        if logger:
            logger.info("데이터: SO100LatentDataset (pre_encode 캐시 %s)", cache)
        return SO100LatentDataset(clips, cache, cfg)
    if logger:
        logger.info("데이터: SO100ClipDataset (on-the-fly — latent 캐시 없음)")
    return SO100ClipDataset(clips, cfg)


def _det_eval(rt, cfg: dict, logger=None) -> tuple:
    """결정론 train-loss 진단 — 고정 subset(seed) + eval_loss_grid. 실패해도 주 루프 안 죽게 호출부에서 격리."""
    import random

    import torch

    from src.models.irasim_runtime import encode_video, eval_loss_grid

    if "det_batch" not in _EVAL_CACHE:
        clips = build_window_index(cfg)
        n = min(int(_g(cfg, "train.det_n", 48)), len(clips))
        sub = [clips[i] for i in random.Random(0).sample(range(len(clips)), n)]
        sds = _make_dataset(cfg, sub, logger)
        b = torch.utils.data.default_collate([sds[i] for i in range(len(sub))])
        latents = b["latent"].to(rt.device) if "latent" in b else encode_video(rt, b["video"])
        _EVAL_CACHE["det_batch"] = (latents, b["action_cond"], len(sub))
    latents, action_cond, n = _EVAL_CACHE["det_batch"]
    det, _grid = eval_loss_grid(rt, latents, action_cond, seed=0)
    return det, n


def train(cfg: dict, device: str = "cuda", logger=None) -> None:
    """학습 진입점. cfg(base+exp+server)에서 하이퍼파라미터 파생."""
    import torch
    from torch.utils.data import DataLoader

    from src.models.backbone import apply_finetune_mode, load_backbone

    out = Path(_g(cfg, "paths.out_dir", "local_runs"))
    out.mkdir(parents=True, exist_ok=True)
    ft = _g(cfg, "model.finetune.mode", "partial")
    lr = float(_g(cfg, "train.lr", 1e-4))
    grad_accum = int(_g(cfg, "train.grad_accum", 4) or 1)
    batch = int(_g(cfg, "train.batch_size", 1) or 1)
    post_w = float(_g(cfg, "train.post_frame_weight", 1.0))
    max_steps = int(_g(cfg, "train.max_steps", 200000) or 200000)
    eval_every = int(_g(cfg, "train.eval_every", 2000))
    ckpt_every = int(_g(cfg, "train.ckpt_every", 2000))
    keep_last = int(_g(cfg, "train.keep_last", 2))
    max_hours = float(_g(cfg, "train.max_wall_hours", 96))

    rt = load_backbone(cfg, device=device, slim_ckpt=_g(cfg, "model.init_ckpt"), logger=logger)
    if ft == "full":
        apply_finetune_mode(rt, cfg, logger)
        trainable = [p for p in rt.model.parameters()]
    else:
        trainable = select_trainable(rt.model, ft, top_blocks=int(_g(cfg, "train.top_blocks", 4)), logger=logger)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=float(_g(cfg, "train.optimizer.weight_decay", 0.0)))

    clips = build_window_index(cfg)
    ds = _make_dataset(cfg, clips, logger)
    dl = DataLoader(ds, batch_size=batch, sampler=build_sampler(clips, cfg),
                    num_workers=int(_g(cfg, "runtime.num_workers", 4)), drop_last=True)

    step = 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        rt.model.load_state_dict(st["model"], strict=False); opt.load_state_dict(st["opt"]); step = st["step"]
        logger and logger.info("재개: step %d", step)

    saved: list[Path] = []
    best = {"score": float("inf")}

    def _state():
        return {"model": rt.model.state_dict(), "opt": opt.state_dict(), "step": step}

    def save_rotating():
        p = out / f"ckpt_{step:07d}.pt"
        atomic_save(_state(), p); saved.append(p)
        while len(saved) > max(1, keep_last):
            try: saved.pop(0).unlink()
            except FileNotFoundError: pass
        atomic_save(_state(), ckpt_path)
        logger and logger.info("ckpt 저장 step %d (보관 %d)", step, len(saved))

    def maybe_best(score):
        if score is not None and score < best["score"]:
            best["score"] = score
            atomic_save({**_state(), "score": score}, out / "best.pt")
            logger and logger.info("★best 갱신 step %d score %.5f", step, score)

    t0 = time.time(); budget_s = max_hours * 3600
    logger and logger.info("학습 시작 FT=%s clips=%d", ft, len(clips))
    rt.model.train(); opt.zero_grad()
    while step < max_steps:
        for i, b in enumerate(dl):
            loss = _step_loss(rt, b, post_w) / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, getattr(rt.cfg, "clip_max_norm", 0.1))
                opt.step(); opt.zero_grad(); step += 1

                if step % 100 == 0:
                    lv = float(loss) * grad_accum
                    logger and logger.info("step %d loss %.4f (%.1fh)", step, lv, (time.time() - t0) / 3600)
                    lc = out / "loss.csv"
                    if not lc.exists():
                        lc.open("w").write("step,loss\n")
                    lc.open("a").write(f"{step},{lv:.6f}\n")
                if step % ckpt_every == 0:
                    save_rotating()
                if step % eval_every == 0:
                    try:                                         # 진단 실패 격리(본 학습 안 죽게)
                        det, det_n = _det_eval(rt, cfg, logger)
                        lb = out / "local_lb.csv"
                        if not lb.exists():
                            lb.open("w").write("step,dvw,pred_total,dino,video,det_loss\n")
                        lb.open("a").write(f"{step},,,,,{det:.5f}\n")
                        logger and logger.info("[eval step %d] det_loss=%.5f(n=%d)", step, det, det_n)
                        maybe_best(det)
                    except Exception as e:
                        logger and logger.warning("[eval step %d] det 스킵(%s): %s", step, type(e).__name__, str(e)[:80])
                if time.time() - t0 > budget_s:
                    logger and logger.warning("예산 도달 — 종료."); save_rotating(); return
                if step >= max_steps:
                    break
    save_rotating()
    logger and logger.info("완료 step %d — best det_loss=%.5f", step, best["score"])
