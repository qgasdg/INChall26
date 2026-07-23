"""결정론 생성 — IRASim (이식: impl/w2/gen_eval.py + irasim_runtime.generate → INChall26 배선).

흐름(docs/04 S5, docs/08):
  홀드아웃 unseen 샘플 → 조건프레임(GT frame0) gen-space 인코딩(mask_x) + 15액션
  → irasim_runtime.generate(PNDM, η=0) → (b,16,gh,gw,3) uint8 → mp4 저장.

★ 결정론 η=0 + seed 고정. 320×512 gen-space 저장(최종 640×480 제출 포맷팅·킷 CSV는 후속 — gen_to_final_mp4/insert_condition_frame).
★실행 요건: diffusers + GPU + 학습 ckpt.
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _g(cfg: dict, path: str, default=None):
    node = cfg
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _build_conditions(cfg: dict, n: int, logger=None) -> list[dict]:
    """holdout.json unseen_scene → {id, cond_gen(gh,gw,3), actions15(15,6)}. CPU만(디코드+정렬)."""
    import numpy as np

    from local_eval import episode_io
    from src.data import transforms as T
    from src.models.action_adapter import adapt_action_seq

    holdout = json.loads((_ROOT / "local_eval" / (_g(cfg, "eval.holdout", "local_eval/holdout.json").split("/")[-1])).read_text(encoding="utf-8"))
    unseen = holdout.get("unseen_scene", [])[:n]
    mean, std = T.load_action_stats()
    mode = _g(cfg, "model.resolution_mode", T.RES_A)
    align = _g(cfg, "data.align_mode", "shifted")
    seq = int(_g(cfg, "task.frames", 16))
    out = []
    for s in unseen:
        ds_id, ep = s["dataset"], int(s["episode_index"])
        f0 = episode_io.read_frames(ds_id, ep, 0, 1)                                    # (1,H,W,3) uint8
        actions = T.normalize_action(episode_io.read_actions(ds_id, ep, 0, seq), mean, std)  # (16,6) z
        cond_gen = T.final_to_gen_target(f0, mode)[0]                                   # (gh,gw,3) uint8
        out.append({"id": f"{ds_id.replace('/', '__')}__ep{ep}",   # score.py sample_key 규약 정합
                    "cond_gen": cond_gen,
                    "actions15": np.ascontiguousarray(adapt_action_seq(actions, seq, align))})
    if logger:
        logger.info("생성 조건 %d개(unseen_scene)", len(out))
    return out


def generate(cfg: dict, ckpt: str, out: str, device: str = "cuda", logger=None) -> None:
    """ckpt 로드 → 홀드아웃 조건으로 16프레임 mp4 생성 → out/. (결정론 η=0)"""
    import imageio
    import torch

    from src.data import transforms as T
    from src.models.backbone import load_backbone
    from src.models.irasim_runtime import encode_video
    from src.models.irasim_runtime import generate as rt_generate

    rt = load_backbone(cfg, device=device, slim_ckpt=ckpt, logger=logger)   # ckpt = 학습된 가중치
    n = int(_g(cfg, "infer.gen_n", 16))
    samples = _build_conditions(cfg, n, logger)
    if not samples:
        logger and logger.warning("생성 조건 없음(holdout unseen_scene 비어있음)"); return

    gh, gw = _g(cfg, "task.gen_hw", [320, 512])
    cond = torch.stack([torch.from_numpy(s["cond_gen"]).float().div(127.5).sub(1.0).permute(2, 0, 1)
                        for s in samples]).unsqueeze(1)                       # (b,1,3,gh,gw)
    mask_x = encode_video(rt, cond)                                           # (b,1,4,h,w)
    actions15 = torch.stack([torch.from_numpy(s["actions15"]).float() for s in samples])  # (b,15,6)

    pred = rt_generate(rt, mask_x, actions15,
                       steps=int(_g(cfg, "infer.steps", 20)), eta=float(_g(cfg, "infer.eta", 0.0)),
                       method=_g(cfg, "infer.method", "PNDM"), seed=int(_g(cfg, "seed", 0)),
                       gen_hw=(int(gh), int(gw)))                             # (b,16,gh,gw,3) uint8

    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    fps = int(_g(cfg, "infer.fps", 8))
    for i, s in enumerate(samples):
        imageio.mimwrite(outp / f"{s['id']}.mp4", list(pred[i]), fps=fps, codec="libx264",
                         output_params=["-crf", str(int(_g(cfg, "infer.video_codec.crf", 10)))])
    logger and logger.info("생성 %d개 → %s (gen-space %dx%d, η=0)", len(samples), outp, gh, gw)
