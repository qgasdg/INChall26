"""INChall26 통합 엔트리포인트 — train / generate / score / config 서브커맨드.

실험 1개 = configs/exp/<expNN_이름>.yaml + 커밋 해시(docs/13 §1). 실행 시 커밋 해시를 로그에 남겨
results/local_lb.csv 기록과 짝지을 것(재현성).

사용:
  python -m src.cli config   --exp configs/exp/exp-template.yaml --server configs/server/server-template.yaml
  python -m src.cli train    --exp configs/exp/exp-01_e3_lora.yaml --server configs/server/a6000.yaml
  python -m src.cli generate --ckpt ckpts/exp-01/best.pt --out local_runs/exp-01
  python -m src.cli score    --videos local_runs/exp-01 --holdout local_eval/holdout.json
"""
from __future__ import annotations

import argparse
import json

from src.utils.config import load_config
from src.utils.runtime import get_logger, pick_device, set_seed


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--exp", default=None, help="configs/exp/<expNN_이름>.yaml (실험 1개 = config 1개)")
    p.add_argument("--server", default=None, help="configs/server/<이름>.yaml (하드웨어 종속값만)")
    p.add_argument("--set", dest="overrides", nargs="*", default=[], help="dotted override 예: train.lr=1e-4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cfg = sub.add_parser("config", help="병합된 최종 config 출력(검증용)")
    _add_common(p_cfg)

    p_tr = sub.add_parser("train", help="학습 (train/trainer.py)")
    _add_common(p_tr)

    p_gen = sub.add_parser("generate", help="결정론 생성 (infer/generate.py)")
    _add_common(p_gen)
    p_gen.add_argument("--ckpt", required=True)
    p_gen.add_argument("--out", required=True)

    p_sc = sub.add_parser("score", help="로컬 리더보드 채점 (eval/local_score.py)")
    _add_common(p_sc)
    p_sc.add_argument("--videos", required=True)
    p_sc.add_argument("--holdout", default="local_eval/holdout.json")

    args = ap.parse_args()
    logger = get_logger()
    set_seed(args.seed)
    cfg = load_config(exp=args.exp, server=args.server, overrides=args.overrides)
    device = pick_device(args.device)
    logger.info("cmd=%s device=%s exp=%s server=%s", args.cmd, device, args.exp, args.server)

    if args.cmd == "config":
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return
    if args.cmd == "train":
        from src.train.trainer import train
        train(cfg, device=device, logger=logger)
    elif args.cmd == "generate":
        from src.infer.generate import generate
        generate(cfg, ckpt=args.ckpt, out=args.out, device=device, logger=logger)
    elif args.cmd == "score":
        from src.eval.local_score import score
        score(cfg, videos=args.videos, holdout=args.holdout, logger=logger)


if __name__ == "__main__":
    main()
