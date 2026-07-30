"""체크포인트 사다리를 A_kit(동작 정확도)으로 줄 세운다 — 개선 곡선 + 제출 후보 선정.

왜 필요한가: 트레이너의 in-loop best는 **확산 손실(det_loss)** 기준이라 우리 병목인 동작 정확도와
    무관하다. 대회 점수의 40%가 Action이고, 그 A_kit은 eval216의 정답 액션이 주어져 있어
    **홀드아웃 없이 킷으로 직접 측정**된다. 그래서 학습이 끝나면 저장된 ckpt들을 이걸로 다시 줄 세운다.

기준선(실측): 제로샷 0.5933 · E3 step14000 0.4730 · 정적 0.4285 · 리더보드 1위 추정 ≤0.139
★단 제로샷 0.5933은 E1 스텝 스윕(exp-05) 값이고 그때의 eta가 원장에 안 남아 있다. 태양님 생성은
  eta 1.0이었다(로그 확인). 눈금을 맞추려면 **원본 ckpt도 사다리에 함께 넣어** 같은 설정으로 재측정할 것:
    --ckpts ckpts/bridge_frame_ada_0300000.pt ckpts/e3r4_s14000_best.pt "local_runs/e4-continue/ckpt_*.pt"

★2단계로 나눠 실행한다 — 킷(pytorch_lightning)과 학습 env가 공존할 수 없기 때문:
  1) 생성  conda torch260:  scripts/rank_ckpts_akit.py gen --ckpts A.pt B.pt "runs/ckpt_*.pt"
  2) 채점  킷 .venv     :  scripts/rank_ckpts_akit.py score --npz-dir local_runs/akit_rank

샘플 수: 기본 48개(전량 216의 22%). 순위만 보면 충분하고 ckpt당 ~20분이면 끝난다.
        최종 후보 1개는 --n 216으로 다시 재서 제출 판단에 쓸 것.

용어: A_kit(영상에서 되읽은 동작이 정답 동작과 얼마나 다른가, 낮을수록 좋음) · ckpt(학습 중간 저장본).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GEN_HW = (320, 512)
DEFAULT_OUT = ROOT / "local_runs" / "akit_rank"


def _eval_samples(n: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """eval216에서 균등 간격 n개 → (id, 조건프레임(1,H,W,3), 정답액션(16,6) deg)."""
    from PIL import Image

    img_dir, act_dir = ROOT / "open/data/eval/images", ROOT / "open/data/eval/actions"
    pngs = sorted(img_dir.glob("sample_*.png"))
    idx = np.linspace(0, len(pngs) - 1, min(n, len(pngs))).round().astype(int)
    out = []
    for i in sorted(set(idx.tolist())):
        p = pngs[i]
        out.append((p.stem, np.asarray(Image.open(p).convert("RGB"))[None],
                    np.load(act_dir / f"{p.stem}.npy")))
    return out


# ------------------------------------------------------------------ 1) 생성
def cmd_gen(args) -> None:
    import torch

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

    samples = _eval_samples(args.n)
    ckpts = sorted({p for pat in args.ckpts for p in glob.glob(pat)})
    if not ckpts:
        raise SystemExit("ckpt 없음: %s" % " ".join(args.ckpts))
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    print("ckpt %d개 × 샘플 %d개 · steps=%d eta=%.1f seed=%d"
          % (len(ckpts), len(samples), args.steps, args.eta, args.seed))

    for ck in ckpts:
        dst = outdir / (Path(ck).stem + ".npz")
        if dst.exists() and not args.overwrite:
            print("  건너뜀(이미 있음): %s" % dst.name); continue
        rt = build(cfg, slim_ckpt=ck, logger=logging.getLogger("rt"))
        vids, ids, acts = [], [], []
        for sid, f0, a_deg in samples:
            gen0 = T.final_to_gen_target(f0, T.RES_A)                       # (1,320,512,3)
            x = torch.from_numpy(gen0).float().div(127.5).sub(1.0).permute(0, 3, 1, 2)[None]
            a15 = np.ascontiguousarray(adapt_action_seq(T.normalize_action(a_deg, mean, std), 16, align))
            pred = rt_generate(rt, encode_video(rt, x), torch.from_numpy(a15).float()[None],
                               steps=args.steps, eta=args.eta, method=args.method,
                               seed=args.seed, gen_hw=GEN_HW)
            vids.append(np.asarray(pred[0])); ids.append(sid); acts.append(a_deg)
            torch.cuda.empty_cache()
        np.savez_compressed(dst, videos=np.stack(vids), ids=np.array(ids), actions=np.stack(acts))
        print("  저장 %s (%d샘플)" % (dst.name, len(vids)), flush=True)
        del rt
        torch.cuda.empty_cache()
    print("\n→ %s  다음: 킷 .venv 로 `score` 실행" % outdir)


# ------------------------------------------------------------------ 2) 채점
def cmd_score(args) -> None:
    from local_eval.kit_bridge import KitScorer, action_mae, get_device

    scorer = KitScorer(get_device())
    rows = []
    for f in sorted(Path(args.npz_dir).glob("*.npz")):
        z = np.load(f)
        vids, acts = z["videos"], z["actions"]
        vals = [action_mae(scorer.action_pred(scorer.to_eval_video(vids[i])[None])[0],
                           scorer.normalize_actions(acts[i])) for i in range(len(vids))]
        step = int("".join(c for c in f.stem if c.isdigit()) or -1)
        rows.append({"ckpt": f.stem, "step": step, "n": len(vals), "akit": float(np.mean(vals))})
        print("  %-28s step %6d  A_kit %.4f (n=%d)" % (f.stem, step, rows[-1]["akit"], len(vals)), flush=True)

    rows.sort(key=lambda r: r["step"])
    print("\n=== A_kit 추이 (낮을수록 좋음) ===")
    print("  기준선: 제로샷 0.5933 · E3 step14000 0.4730 · 정적 0.4285 · 1위 추정 ≤0.139")
    for r in rows:
        bar = "#" * max(0, int((0.62 - r["akit"]) / 0.005))
        print("  step %6d  %.4f  %s" % (r["step"], r["akit"], bar))
    if rows:
        best = min(rows, key=lambda r: r["akit"])
        print("\n  최저 = %s (A_kit %.4f)" % (best["ckpt"], best["akit"]))
        print("  %s" % ("→ 정적(0.4285) 돌파. 제출 후보 — --n 216으로 재측정 후 판단"
                        if best["akit"] < 0.4285 else
                        "→ 아직 정적 미달. 학습 지속 또는 다른 지렛대 필요"))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "akit_ckpt_rank.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n→ results/akit_ckpt_rank.json")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="    %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="conda torch260 — ckpt별 eval 샘플 생성 → npz")
    g.add_argument("--ckpts", required=True, nargs="+",
                   help='경로/글롭 여러 개 (셸 중괄호는 파이썬 glob이 못 읽으니 공백으로 나열). '
                        '예: ckpts/base.pt "local_runs/e4-continue/ckpt_*.pt"')
    g.add_argument("--n", type=int, default=48, help="eval216 중 사용할 샘플 수(균등 간격)")
    g.add_argument("--steps", type=int, default=20)
    g.add_argument("--eta", type=float, default=1.0, help="E1 실측상 1.0이 결정론(0)을 이김")
    g.add_argument("--method", default="PNDM")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default=str(DEFAULT_OUT))
    g.add_argument("--overwrite", action="store_true")
    g.set_defaults(func=cmd_gen)

    s = sub.add_parser("score", help="킷 .venv — npz들을 A_kit으로 채점·정렬")
    s.add_argument("--npz-dir", default=str(DEFAULT_OUT))
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
