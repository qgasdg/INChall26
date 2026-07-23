"""IRASim 런타임 래퍼 (벤더 무수정) — 모델·VAE·디퓨전 구성 + 생성/학습손실. (이식: impl/w2/irasim_runtime.py)

벤더 API(third_party/irasim):
  - 모델   : models.get_models(cfg) → IRASim DiT (extras=3 frame-level AdaLN)
  - VAE    : diffusers AutoencoderKL.from_pretrained(vae, subfolder='vae')  (SDXL)
  - 디퓨전 : diffusion.create_mask_diffusion(timestep_respacing='', learn_sigma)
  - 손실   : diffusion.training_losses(model, latents, t, dict(actions=(b,15,6), mask_frame_num=1))
  - 생성   : PNDMScheduler + sample.Trajectory2VideoGenPipeline(vae, scheduler, transformer=model)

우리 개조는 전부 여기서: state_dim 6 embed_state zero-init 교체(action_adapter), dtype/attention(cfg.runtime),
slim ema ckpt strict=False 로드.

★환경: 이 모듈의 build()/생성은 **diffusers** 필요(INChall26 기본 .venv 미포함 → 설치 필요) + GPU + IRASim 슬림 ckpt.
   config 조립(build_vendor_cfg/selfcheck)만 diffusers 없이 검증 가능(omegaconf + 벤더 yaml).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "irasim"
LATENT_DIV = 8   # SDXL VAE f8: (H,W) 픽셀 → (H/8, W/8) 잠재


def _add_vendor_path() -> None:
    p = str(VENDOR)
    if p not in sys.path:
        sys.path.insert(0, p)


def build_vendor_cfg(dataset: str = "bridge", variant: str = "frame_ada", mode: str = "val",
                     gen_hw: tuple[int, int] = (320, 512), num_frames: int = 16,
                     attention_mode: str = "math", vae_model_path: str | None = None):
    """벤더 base(data+diffusion) + evaluation/<dataset>/<variant> 병합 + 우리 오버라이드 → omegaconf cfg.
    (impl/w2 load_config — INChall26 src.utils.config.load_config 와 이름 충돌 피해 rename)."""
    _add_vendor_path()
    from omegaconf import OmegaConf

    base_data = OmegaConf.load(VENDOR / "configs" / "base" / "data.yaml")
    base_diff = OmegaConf.load(VENDOR / "configs" / "base" / "diffusion.yaml")
    exp = OmegaConf.load(VENDOR / "configs" / "evaluation" / dataset / f"{variant}.yaml")
    cfg = OmegaConf.merge(base_data, base_diff, exp)

    h, w = gen_hw
    cfg.dataset = dataset
    cfg.mode = mode
    cfg.num_frames = num_frames
    cfg.pre_encode = False                      # 우리 로더는 픽셀 video 제공 → 런타임 VAE 인코딩(on-the-fly 경로)
    cfg.video_size = [h, w]
    cfg.latent_size = [h // LATENT_DIV, w // LATENT_DIV]
    cfg.attention_mode = attention_mode
    cfg.sample_method = "PNDM"
    cfg.scheduler_path = str(VENDOR / "pretrained_models" / "scheduler")
    cfg.vae_model_path = vae_model_path or "stabilityai/stable-diffusion-xl-base-1.0"
    return cfg


# 해상도 차로 shape가 달라져도 안전하게 드롭 가능한 키(고정 sincos, 모델 자체 생성).
FIXED_EMBED_KEYS = ("pos_embed", "temp_embed")


def _robust_load(model, sd: dict, logger=None) -> dict:
    """ckpt↔model state_dict shape diff 전수 비교 후 키별 로드/드롭(strict=False가 조용히 넘기지 않게 로그)."""
    msd = model.state_dict()
    to_load, drop_shape, unexpected = {}, [], []
    for k, v in sd.items():
        if k not in msd:
            unexpected.append(k)
        elif tuple(v.shape) != tuple(msd[k].shape):
            drop_shape.append((k, tuple(v.shape), tuple(msd[k].shape)))
        else:
            to_load[k] = v
    missing = [k for k in msd if k not in sd]
    model.load_state_dict(to_load, strict=False)
    if logger:
        for k, cs, ms in drop_shape:
            if any(k == fe or k.startswith(fe) for fe in FIXED_EMBED_KEYS):
                logger.info("  drop(shape·고정sincos): %s ckpt%s vs model%s → 모델 자체값", k, cs, ms)
            else:
                logger.warning("  ⚠️ drop(shape·비고정!): %s ckpt%s vs model%s — 확인 필요", k, cs, ms)
        logger.info("ckpt 로드: 적재=%d shape드롭=%d ckpt여분=%d 모델미로드=%d",
                    len(to_load), len(drop_shape), len(unexpected), len(missing))
    return {"loaded": len(to_load), "drop_shape": drop_shape, "unexpected": unexpected, "missing": missing}


def enable_grad_checkpointing(model, logger=None) -> int:
    """벤더 무수정 grad checkpointing — model.blocks 각 forward(x,c)를 checkpoint 로 감쌈(인스턴스 몽키패치)."""
    from torch.utils.checkpoint import checkpoint

    n = 0
    for block in getattr(model, "blocks", []):
        if getattr(block, "_ckpt_wrapped", False):
            continue
        orig = block.forward
        def wrapped(x, c, _orig=orig, _blk=block):
            import torch
            if _blk.training and torch.is_grad_enabled():
                return checkpoint(_orig, x, c, use_reentrant=False)
            return _orig(x, c)
        block.forward = wrapped
        block._ckpt_wrapped = True
        n += 1
    if logger:
        logger.info("grad checkpointing 적용: %d 블록(벤더 무수정 래퍼)", n)
    return n


def _apply_runtime(icfg: dict, logger=None):
    """INChall26 cfg.runtime → tf32 설정 + dtype 반환 (impl/w2 profiles.apply_runtime 대체)."""
    import torch

    rt = (icfg or {}).get("runtime", {})
    if rt.get("matmul_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    prec = (icfg.get("train", {}).get("precision") or rt.get("precision") or "bf16")
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(prec, torch.float32)


def _resolve_attention(icfg: dict, logger=None) -> str:
    """cfg.runtime.attention (flash 미탐지 시 math 강등). (impl/w2 resolve_attention_mode 대체)."""
    want = (icfg or {}).get("runtime", {}).get("attention", "math")
    if want == "flash":
        try:
            import flash_attn  # noqa: F401
            return "flash"
        except Exception:
            if logger:
                logger.warning("flash_attn 미탐지 → attention_mode=math 강등")
            return "math"
    return want


@dataclass
class Runtime:
    model: object
    vae: object
    diffusion: object
    cfg: object          # 벤더 omegaconf cfg (mask_frame_num·beta_*·num_frames·learn_sigma·scheduler_path 보유)
    device: object
    dtype: object
    vae_chunk: int = 4
    micro_batch: int = 1
    gen_batch: int = 2


def build(icfg: dict, slim_ckpt: str | Path | None = None, logger=None) -> Runtime:
    """IRASim 모델(+SO100 임베더 교체)·SDXL VAE·디퓨전 구성. icfg = INChall26 병합 config.

    ★diffusers·GPU·slim ckpt 필요. gen_hw/frames/attention/precision/batch 는 icfg 에서 파생.
    """
    _add_vendor_path()
    import torch
    from diffusers.models import AutoencoderKL

    from diffusion import create_mask_diffusion
    from models import get_models

    from src.models.action_adapter import swap_action_embedder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _apply_runtime(icfg, logger)
    gh, gw = (icfg.get("task", {}).get("gen_hw", [320, 512]))
    vcfg = build_vendor_cfg(gen_hw=(int(gh), int(gw)),
                            num_frames=int(icfg.get("task", {}).get("frames", 16)),
                            attention_mode=_resolve_attention(icfg, logger),
                            vae_model_path=icfg.get("model", {}).get("vae"))

    model = get_models(vcfg)
    if slim_ckpt:
        ckpt = torch.load(slim_ckpt, map_location="cpu", weights_only=False)   # 신뢰 소스(슬림/공식 ckpt)
        sd = ckpt["ema"] if isinstance(ckpt, dict) and "ema" in ckpt else ckpt
        _robust_load(model, sd, logger)
    swap_action_embedder(model, state_dim=int(icfg.get("data", {}).get("action_dim", 6)),
                         zero_init=bool(icfg.get("model", {}).get("action_adapter", {}).get("zero_init", True)), logger=logger)

    if icfg.get("train", {}).get("grad_checkpoint", False):
        enable_grad_checkpointing(model, logger)

    model = model.to(device)
    vae = AutoencoderKL.from_pretrained(vcfg.vae_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    diffusion = create_mask_diffusion(timestep_respacing="", learn_sigma=vcfg.learn_sigma)
    srv = icfg.get("train", {})
    return Runtime(model=model, vae=vae, diffusion=diffusion, cfg=vcfg, device=device, dtype=dtype,
                   vae_chunk=int(icfg.get("runtime", {}).get("vae_encode_chunk", 4)),
                   micro_batch=int(srv.get("grad_accum", 1) or 1),
                   gen_batch=int(icfg.get("runtime", {}).get("gen_batch", 2)))


def encode_video(rt: Runtime, video_bf_c_h_w, chunk: int | None = None):
    """(b,f,3,H,W) 픽셀[-1,1] → (b,f,4,h,w) SDXL 잠재. OOM 방지: 프레임 청크·no_grad. fp32 유지."""
    import torch
    from einops import rearrange

    b, f = video_bf_c_h_w.shape[:2]
    x = rearrange(video_bf_c_h_w, "b f c h w -> (b f) c h w").to(rt.device)
    ck = chunk or getattr(rt, "vae_chunk", 4) or 4
    sf = rt.vae.config.scaling_factor
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], ck):
            z = rt.vae.encode(x[i:i + ck]).latent_dist.sample().mul_(sf)
            outs.append(z)
    z = torch.cat(outs, dim=0)
    return rearrange(z, "(b f) c h w -> b f c h w", b=b, f=f)


def training_loss(rt: Runtime, batch) -> "object":
    """배치(SO100ClipDataset) → diffusion.training_losses loss. actions=action_cond(b,15,6)."""
    import torch

    latents = encode_video(rt, batch["video"])
    actions = batch["action_cond"].to(rt.device)
    model_kwargs = dict(actions=actions, mask_frame_num=rt.cfg.mask_frame_num)
    t = torch.randint(0, rt.diffusion.num_timesteps, (latents.shape[0],), device=rt.device)
    return rt.diffusion.training_losses(rt.model, latents, t, model_kwargs)["loss"].mean()


def eval_loss_grid(rt: Runtime, latents, actions_cond, t_grid=(100, 300, 500, 700, 900), seed: int = 0,
                   batch: int | None = None):
    """결정론 평가 loss — 고정 latents + 고정 t격자 + 시드고정 noise + model.eval(). 반환 (평균, 격자별).
    예외에도 학습 모드·전역 RNG 원복(진단이 학습 오염 안 하게)."""
    import torch

    was_training = rt.model.training
    rng_state = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rt.model.eval()
    N = rt.diffusion.num_timesteps
    bs = batch or getattr(rt, "gen_batch", 2) or 2
    per = {}
    try:
        with torch.no_grad():
            for tv in t_grid:
                tvc = min(int(tv), N - 1)
                torch.manual_seed(seed)
                tot, cnt = 0.0, 0
                for i0 in range(0, latents.shape[0], bs):
                    lat = latents[i0:i0 + bs].to(rt.device)
                    mk = dict(actions=actions_cond[i0:i0 + bs].to(rt.device), mask_frame_num=rt.cfg.mask_frame_num)
                    t = torch.full((lat.shape[0],), tvc, device=rt.device, dtype=torch.long)
                    l = float(rt.diffusion.training_losses(rt.model, lat, t, mk)["loss"].mean())
                    tot += l * lat.shape[0]; cnt += lat.shape[0]
                per[tvc] = tot / cnt
    finally:
        if was_training:
            rt.model.train()
        torch.set_rng_state(rng_state)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return sum(per.values()) / len(per), per


def build_sampler(rt: Runtime, steps: int, eta: float = 0.0, method: str = "PNDM"):
    """샘플러 반환. PNDM(기본) | DDIM(eta) | DPM++ | SDE-DPM++."""
    _add_vendor_path()
    common = dict(beta_start=rt.cfg.beta_start, beta_end=rt.cfg.beta_end, beta_schedule=rt.cfg.beta_schedule)
    if method == "DDIM":
        from diffusers.schedulers import DDIMScheduler
        sch = DDIMScheduler.from_pretrained(rt.cfg.scheduler_path, **common)
        sch.set_timesteps(steps)
        return sch, {"eta": eta}
    if method in ("DPM++", "SDE-DPM++"):
        from diffusers.schedulers import DPMSolverMultistepScheduler
        algo = "sde-dpmsolver++" if method == "SDE-DPM++" else "dpmsolver++"
        sch = DPMSolverMultistepScheduler.from_pretrained(rt.cfg.scheduler_path, algorithm_type=algo, **common)
        sch.set_timesteps(steps)
        return sch, {}
    from diffusers.schedulers import PNDMScheduler
    sch = PNDMScheduler.from_pretrained(rt.cfg.scheduler_path, variance_type=rt.cfg.variance_type, **common)
    sch.set_timesteps(steps)
    return sch, {}


def make_pipeline(rt: Runtime, steps: int, eta: float, method: str):
    _add_vendor_path()
    from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline

    scheduler, _ = build_sampler(rt, steps, eta, method)
    return Trajectory2VideoGenPipeline(vae=rt.vae, transformer=rt.model, scheduler=scheduler)


def generate(rt: Runtime, mask_x, actions15, steps: int = 20, eta: float = 0.0,
             method: str = "PNDM", seed: int = 0, gen_hw: tuple[int, int] = (320, 512),
             batch: int | None = None):
    """조건프레임 잠재(mask_x: b,1,4,h,w) + 15액션(b,15,6) → 생성 프레임 (b,16,gh,gw,3) uint8.
    CFG 봉인(guidance_scale=1.0), 추론 model.eval(). (FreeAction truncation 은 미이식 — 필요 시 impl/w2/freeaction.py)."""
    import numpy as np
    import torch

    was_training = rt.model.training
    rt.model.eval()
    try:
        pipe = make_pipeline(rt, steps, eta, method)
        h, w = gen_hw
        gb = batch or getattr(rt, "gen_batch", 2) or 2
        g = torch.Generator(device=str(rt.device)).manual_seed(seed)
        outs = []
        for i0 in range(0, mask_x.shape[0], gb):
            mx = mask_x[i0:i0 + gb].to(rt.device)
            act = actions15[i0:i0 + gb].to(rt.device)
            videos, _ = pipe(
                action=act, mask_x=mx, num_inference_steps=steps, guidance_scale=1.0,
                video_length=rt.cfg.num_frames, height=h, width=w, eta=eta, generator=g,
                latents=None, output_type="both", device=rt.device,
            )
            px = ((videos / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
            outs.append(px.permute(0, 1, 3, 4, 2).cpu().numpy())
    finally:
        if was_training:
            rt.model.train()
    return np.concatenate(outs, axis=0)


def selfcheck() -> dict:
    """torch/diffusers 없이 벤더 config 조립만 확인(omegaconf + 벤더 yaml)."""
    cfg = build_vendor_cfg(gen_hw=(320, 512))
    return {
        "dataset": cfg.dataset, "num_frames": cfg.num_frames, "pre_encode": cfg.pre_encode,
        "video_size": list(cfg.video_size), "latent_size": list(cfg.latent_size),
        "sample_method": cfg.sample_method, "mask_frame_num": cfg.mask_frame_num,
        "extras": cfg.extras, "vae_model_path": cfg.vae_model_path,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(selfcheck(), ensure_ascii=False, indent=1))
