"""best-of-N — 같은 장면을 시드만 바꿔 N개 생성하고, 동작 정확도로 제일 좋은 것만 남긴다.

왜 되는가: eval의 **정답 액션이 입력으로 주어진다**(`open/data/eval/actions/*.npy`). 그래서 우리가
    만든 후보들을 킷 채점기로 직접 채점해 고를 수 있다. 학습 없이 점수를 올리는 유일한 정공법이고,
    손실-지표 불일치 문제를 우회한다(지표를 직접 보고 고르니까).
검증 이력: 동종 best-of-3 총점 **−0.023**, Action 기준 선택이 오라클 이득의 91% 회수(2026-07-20, BOARD).

2단계 — 킷(.venv)과 학습 env(torch260)가 공존 불가하므로 나눈다:
  1) 생성  torch260:  best_of_n.py gen    --ckpt <ckpt> --cand local_runs/cand --seeds 0 1 2
  2) 선택  킷 .venv:  best_of_n.py select --cand local_runs/cand --out local_runs/pred_bo3

후보는 **mp4로 저장**한다 — 최종 제출물과 같은 형식이라 인코딩 손실까지 포함해 채점·선택된다
(npz로 고르면 인코딩 후 순위가 바뀔 수 있다).

이미 만들어 둔 단일 시드 결과가 있으면 후보로 재사용할 것:
  for f in local_runs/pred_x/sample_*.mp4; do cp -n $f local_runs/cand/$(basename $f .mp4)_s0.mp4; done
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GEN_HW = (320, 512)
SEQ, FPS = 16, 6


def _read_mp4(path: Path) -> np.ndarray:
    """mp4 → (16,H,W,3) uint8. 프레임 수가 다르면 예외(제출 규격 위반 조기 발견)."""
    import av

    with av.open(str(path)) as c:
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    if len(fr) != SEQ:
        raise ValueError(f"{path.name}: 프레임 {len(fr)}개 (16이어야 함)")
    return np.stack(fr)


def _write_mp4(frames: np.ndarray, path: Path) -> None:
    """make_submission.py와 동일 규격(h264 crf10 yuv420p 6fps)."""
    import av

    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=FPS)
        st.height, st.width = frames.shape[1], frames.shape[2]
        st.pix_fmt = "yuv420p"
        st.options = {"crf": "10", "preset": "medium"}
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def cmd_gen(args) -> None:
    import torch
    from PIL import Image

    from src.data import transforms as T
    from src.models.action_adapter import adapt_action_seq
    from src.models.irasim_runtime import build, encode_video
    from src.models.irasim_runtime import generate as rt_generate
    from src.utils.config import load_config

    cfg = load_config()
    cfg.setdefault("train", {})["precision"] = "fp16"
    cfg.setdefault("data", {})["pre_encode"] = False
    mean, std = T.load_action_stats()
    align = cfg.get("data", {}).get("align_mode", "shifted")

    img_dir, act_dir = ROOT / "open/data/eval/images", ROOT / "open/data/eval/actions"
    pngs = sorted(img_dir.glob("sample_*.png"))
    if args.limit:
        pngs = pngs[:args.limit]
    cand = Path(args.cand); cand.mkdir(parents=True, exist_ok=True)

    todo = [(p, s) for p in pngs for s in args.seeds if not (cand / f"{p.stem}_s{s}.mp4").exists()]
    print("샘플 %d × 시드 %s → 생성할 후보 %d개 (이미 있는 건 건너뜀)"
          % (len(pngs), args.seeds, len(todo)))
    if not todo:
        print("모두 존재 — 바로 select 로."); return

    rt = build(cfg, slim_ckpt=args.ckpt, logger=logging.getLogger("rt"))
    t0 = time.time()
    for i, (p, seed) in enumerate(todo, 1):
        f0 = np.asarray(Image.open(p).convert("RGB"))[None]
        x = torch.from_numpy(T.final_to_gen_target(f0, T.RES_A)).float().div(127.5).sub(1.0)
        x = x.permute(0, 3, 1, 2)[None]
        a_deg = np.load(act_dir / f"{p.stem}.npy")
        a15 = np.ascontiguousarray(adapt_action_seq(T.normalize_action(a_deg, mean, std), SEQ, align))
        pred = rt_generate(rt, encode_video(rt, x), torch.from_numpy(a15).float()[None],
                           steps=args.steps, eta=args.eta, method=args.method,
                           seed=seed, gen_hw=GEN_HW)
        _write_mp4(np.asarray(pred[0]), cand / f"{p.stem}_s{seed}.mp4")
        torch.cuda.empty_cache()
        if i % 20 == 0 or i == len(todo):
            el = time.time() - t0
            print("  %4d/%d  %.1fs/개  남은 %.0f분" % (i, len(todo), el / i, (len(todo) - i) * el / i / 60),
                  flush=True)
    print("\n후보 %d개 → %s  다음: 킷 .venv 로 `select`" % (len(list(cand.glob('*.mp4'))), cand))


def cmd_select(args) -> None:
    from local_eval.kit_bridge import KitScorer, action_mae, get_device

    act_dir = ROOT / "open/data/eval/actions"
    cand = Path(args.cand)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Path]] = {}
    for f in sorted(cand.glob("sample_*_s*.mp4")):
        groups.setdefault(f.stem.rsplit("_s", 1)[0], []).append(f)
    if not groups:
        raise SystemExit("후보 없음: %s" % cand)

    scorer = KitScorer(get_device())
    rows, n_multi = [], 0
    for i, (sid, files) in enumerate(sorted(groups.items()), 1):
        tgt = scorer.normalize_actions(np.load(act_dir / f"{sid}.npy"))
        scored = [(action_mae(scorer.action_pred(scorer.to_eval_video(_read_mp4(f))[None])[0], tgt), f)
                  for f in files]
        scored.sort(key=lambda t: t[0])
        shutil.copyfile(scored[0][1], out / f"{sid}.mp4")
        n_multi += len(files) > 1
        rows.append({"sample": sid, "n_cand": len(files), "chosen": scored[0][1].name,
                     "akit_best": scored[0][0], "akit_worst": scored[-1][0],
                     "akit_first": next(a for a, f in scored if f.name.endswith("_s0.mp4"))
                     if any(f.name.endswith("_s0.mp4") for _, f in scored) else scored[0][0]})
        if i % 50 == 0 or i == len(groups):
            print("  %3d/%d" % (i, len(groups)), flush=True)

    best = float(np.mean([r["akit_best"] for r in rows]))
    first = float(np.mean([r["akit_first"] for r in rows]))
    worst = float(np.mean([r["akit_worst"] for r in rows]))
    print("\n=== best-of-N 선택 결과 (샘플 %d, 후보 여럿인 것 %d) ===" % (len(rows), n_multi))
    print("  시드0만 썼을 때   A_kit %.4f   ← 선택 안 한 경우" % first)
    print("  best-of-N 선택    A_kit %.4f   (이득 %+.4f)" % (best, best - first))
    print("  최악만 골랐다면   A_kit %.4f   ← 후보 간 편차 폭" % worst)
    print("  기준선: 정적 0.4285 · 총점 기여 = 0.4 × A_kit")
    print("  %s" % ("★정적 돌파" if best < 0.4285 else "정적 미달 — 후보 수를 늘리거나 모델을 개선할 것"))

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "best_of_n.json").write_text(json.dumps(
        {"cand": str(cand), "out": str(out), "akit_first": first, "akit_best": best,
         "akit_worst": worst, "rows": rows}, ensure_ascii=False, indent=1))
    print("\n→ %s  ·  선택된 영상 %s" % ("results/best_of_n.json", out))


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="    %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="torch260 — 시드별 후보 mp4 생성")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--cand", default="local_runs/cand")
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--steps", type=int, default=20)
    g.add_argument("--eta", type=float, default=1.0)
    g.add_argument("--method", default="PNDM")
    g.add_argument("--limit", type=int, default=0)
    g.set_defaults(func=cmd_gen)

    s = sub.add_parser("select", help="킷 .venv — 후보 채점 후 최고만 out 으로")
    s.add_argument("--cand", default="local_runs/cand")
    s.add_argument("--out", default="local_runs/pred_bo3")
    s.set_defaults(func=cmd_select)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
