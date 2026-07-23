"""로컬 리더보드 채점 — INChall26 local_eval 킷 브리지 재사용 (thin adapter).

★채점 수학을 재구현하지 않는다 — `local_eval/kit_bridge`(킷 무수정)의 원시함수(KitScorer·dino_distance_both·
cosine_distance·action_mae·total_score)를 그대로 호출. 생성 영상 디렉토리를 홀드아웃 GT와 대조해
대회 산식(0.3·DINO + 0.3·Video + 0.4·Action, 낮을수록↑) 산출. (local_eval/score.py 의 파이프라인 호출판)

★영상 파일명 규약(local_eval/score.py 와 동일): "<dataset '/'→'__'>__ep<episode_index>.mp4" (16프레임).
  infer/generate.py 산출이 이 규약을 따라야 매칭됨(sample_key).

★실행 요건: 킷 추출기 3종 로드(open/submission_kit/checkpoints, DINOv2·R3D-18·action_extractor) + torch + 생성 영상.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_EVAL = _ROOT / "local_eval"


def _g(cfg: dict, path: str, default=None):
    node = cfg
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def sample_key(dataset: str, episode_index: int) -> str:
    """생성 영상 파일명 키 — local_eval/score.py 와 동일 규약."""
    return f"{dataset.replace('/', '__')}__ep{episode_index}"


def _load_holdout(cfg: dict, holdout) -> dict:
    import json

    if isinstance(holdout, dict):
        return holdout
    p = Path(holdout) if holdout else (_ROOT / _g(cfg, "eval.holdout", "local_eval/holdout.json"))
    if not p.is_absolute():
        p = _ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def score(cfg: dict, videos: str, holdout=None, tier: str = "unseen_scene", device=None, logger=None) -> dict:
    """생성 영상 디렉토리 → 대회 산식 채점(dict). local_eval 킷 브리지 재사용.

    반환: {"n", "dino", "video", "action", "total", "dvw"(0.3D+0.3V), "per_sample"}.
    """
    import numpy as np

    if str(_LOCAL_EVAL) not in sys.path:
        sys.path.insert(0, str(_LOCAL_EVAL))
    import kit_bridge
    from kit_bridge import (TEMPORAL_LENGTH, KitScorer, action_mae, cosine_distance,
                            dino_distance_both, get_device, total_score)
    from episode_io import read_actions, read_frames
    kit = kit_bridge.kit

    hv = _load_holdout(cfg, holdout)
    tiers = ["in_domain", "unseen_scene"] if tier == "both" else [tier]
    scorer = KitScorer(get_device(device))
    vroot = Path(videos)

    rows = []
    for t in tiers:
        for s in hv.get(t, []):
            key = sample_key(s["dataset"], int(s["episode_index"]))
            path = vroot / f"{key}.mp4"
            if not path.exists():
                continue
            gen_np = kit.read_video_uint8(path, expected_frames=TEMPORAL_LENGTH).numpy()
            gen = scorer.to_eval_video(gen_np).unsqueeze(0)
            gt = scorer.to_eval_video(
                read_frames(s["dataset"], int(s["episode_index"]), 0, TEMPORAL_LENGTH)).unsqueeze(0)
            target = scorer.normalize_actions(
                read_actions(s["dataset"], int(s["episode_index"]), 0, TEMPORAL_LENGTH))
            dino_pf, _flat = dino_distance_both(scorer.dino_features(gen)[0], scorer.dino_features(gt)[0])
            video_d = cosine_distance(scorer.video_features(gen)[0], scorer.video_features(gt)[0])
            act = action_mae(scorer.action_pred(gen)[0], target)
            rows.append({"tier": t, "sample": key, "dino": dino_pf, "video": video_d,
                         "action": act, "total": total_score(dino_pf, video_d, act)})

    if not rows:
        logger and logger.warning("%s 에서 홀드아웃 매칭 mp4 없음 — 파일명 규약(sample_key) 확인", vroot)
        return {"n": 0}
    agg = lambda k: float(np.mean([r[k] for r in rows]))
    res = {"n": len(rows), "dino": agg("dino"), "video": agg("video"), "action": agg("action"),
           "total": agg("total"), "dvw": 0.3 * agg("dino") + 0.3 * agg("video"), "per_sample": rows}
    logger and logger.info("채점 n=%d DINO %.4f Video %.4f Action %.4f → total %.4f",
                           res["n"], res["dino"], res["video"], res["action"], res["total"])
    return res


def append_local_lb(step: int, res: dict, csv_path: str = "results/local_lb.csv") -> None:
    """채점 결과 1행 append (정본 ledger, plot_training 호환 헤더)."""
    p = Path(csv_path)
    if not p.is_absolute():
        p = _ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = lambda v: f"{v:.5f}" if isinstance(v, (int, float)) else ""
    new = not p.exists()
    with p.open("a") as f:
        if new:                                             # dvw / pred_total(=total) / dino / video / det_loss(공란)
            f.write("step,dvw,pred_total,dino,video,det_loss\n")
        f.write(",".join([str(step), fmt(res.get("dvw")), fmt(res.get("total")),
                          fmt(res.get("dino")), fmt(res.get("video")), ""]) + "\n")
