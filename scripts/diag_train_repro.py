"""학습 데이터 재현 진단 — "학습이 부족한가" vs "모델 자체 한계인가"를 가른다.

배경: 뭉개짐 원인 후보 5종 중 VAE 상한(A)·해상도/pos_embed(B)·노이즈 스케줄(C)은 기각됐고
      (`diag_blur.py` · `diag_res_ab.py`), 남은 건 **학습 부족**과 **모델 한계**다. 이 둘은
      *처음 보는 샘플*만 봐서는 구별되지 않는다 — 둘 다 똑같이 뭉개진 결과를 낸다.

가르는 법: **모델이 학습에 실제로 쓴 클립**을 그대로 다시 생성시킨다.
  - 학습 클립은 또렷한데 처음 보는 클립만 뭉개진다 → 모델은 맞출 능력이 있다. **학습량·일반화 문제**
    (→ EMA 도입·더 긴 학습·데이터 확대가 먹힌다)
  - 학습 클립도 똑같이 뭉개진다 → 자기가 본 것조차 못 맞춘다. **모델·설정 한계**
    (→ 더 학습해도 소용없음. 모델 교체 또는 손실·샘플링 재설계)

학습 풀은 `src.data.dataset.build_window_index(cfg)` 를 그대로 써서 뽑는다 — 학습 때와 **동일한
제외 규칙**(홀드아웃·hard_exclude)이 적용되므로 "정말 학습에 쓴 클립"이 보장된다.

**덤으로 붙은 천장 검사**(`ceiling_check`): 유지율은 `마지막÷첫`이라, 모델이 입력과 무관하게
늘 비슷한 선명도만 뱉으면 **또렷한 장면일수록 유지율이 낮게** 나온다 — 모델은 똑같이 행동하는데도.
입력 선명도 대비 **출력 절대 선명도**의 로그-로그 기울기가 0에 가까우면 천장(모델 한계)이고,
그동안 재온 유지율 격차의 상당 부분이 장면 선택이 만든 착시라는 뜻이다.

지표 3종:
  - 유지율   = 라플라시안분산(마지막 프레임) / 라플라시안분산(첫=조건 프레임). 100%면 안 뭉개짐.
  - GT 유지율 = 실제 정답 영상의 같은 값. **천장** 역할(실제 영상도 100%는 아님).
  - PSNR(마지막) = 생성 마지막 프레임이 정답 마지막 프레임과 얼마나 같은지(높을수록 같음).
               유지율이 높아도 엉뚱한 그림을 또렷하게 그렸을 수 있어 함께 본다.

용어: 라플라시안 분산(선명도 수치) · PSNR(정답과의 일치도, dB) · 클립(연속 16프레임 한 토막).

사용(GPU):
  /opt/conda/envs/torch260/bin/python scripts/diag_train_repro.py --ckpt <FT ckpt> --n 6
  # 제로샷 기준선도 같이 보려면 --ckpt ckpts/bridge_frame_ada_0300000.pt 로 한 번 더
산출: 표준출력 + results/diag_train_repro.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GEN_HW = (320, 512)


def _gray(img: np.ndarray) -> np.ndarray:
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)


def _lap_var(g: np.ndarray) -> float:
    d = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(d.var())


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def _pack(sid: str, frames: np.ndarray, actions_deg: np.ndarray, cfg: dict) -> dict:
    """원본 프레임·액션 → 생성 규격(320×512 레터박스 GT 16장 + z-score 액션 15스텝)."""
    from src.data import transforms as T
    from src.models.action_adapter import adapt_action_seq

    seq = int(cfg.get("task", {}).get("frames", 16))
    align = cfg.get("data", {}).get("align_mode", "shifted")
    mean, std = T.load_action_stats()
    gt = T.final_to_gen_target(frames, T.RES_A)                       # (16,320,512,3) uint8
    acts = T.normalize_action(actions_deg, mean, std)
    return {"id": sid, "gt": gt,
            "actions15": np.ascontiguousarray(adapt_action_seq(acts, seq, align))}


def train_samples(cfg: dict, n: int) -> list[dict]:
    """학습 풀(build_window_index)에서 데이터셋이 겹치지 않게 n개 — 균등 간격 결정론 선택."""
    from local_eval import episode_io

    from src.data.dataset import build_window_index

    clips = build_window_index(cfg)
    seen: set[str] = set()
    picked = []
    for i in range(0, len(clips), max(1, len(clips) // (n * 8) or 1)):  # 앞쪽 한 데이터셋에 쏠리지 않게 훑기
        c = clips[i]
        if c.ds_id in seen:
            continue
        seen.add(c.ds_id)
        picked.append(c)
        if len(picked) >= n:
            break
    seq = int(cfg.get("task", {}).get("frames", 16))
    out = []
    for c in picked:
        sid = "train__%s__ep%06d__t%04d" % (c.ds_id.replace("/", "_"), c.episode_index, c.start)
        out.append(_pack(sid, episode_io.read_frames(c.ds_id, c.episode_index, c.start, seq),
                         episode_io.read_actions(c.ds_id, c.episode_index, c.start, seq), cfg))
    return out


def unseen_samples(cfg: dict, n: int) -> list[dict]:
    """대조군 — 홀드아웃 unseen에서 **데이터셋 겹치지 않게** n개.

    ★앞에서 n개를 그냥 자르면 전부 한 데이터셋(aractingi/push_cube)으로 쏠린다 —
      학습 쪽은 6개 서로 다른 데이터셋이라 두 묶음이 비교 불가능해진다. 데이터셋을
      돌아가며(round-robin) 뽑아 구성 편향을 없앤다.
    """
    from local_eval import episode_io

    seq = int(cfg.get("task", {}).get("frames", 16))
    hp = ROOT / "local_eval" / "holdout_v2.json"
    pool = [s for s in json.loads(hp.read_text(encoding="utf-8"))["samples"]
            if str(s.get("tier", "")).startswith("unseen")]
    by_ds: dict[str, list] = {}
    for s in pool:
        by_ds.setdefault(s["dataset"], []).append(s)
    picked, rnd = [], 0
    while len(picked) < n and any(len(v) > rnd for v in by_ds.values()):
        for ds in sorted(by_ds):
            if len(by_ds[ds]) > rnd and len(picked) < n:
                picked.append(by_ds[ds][rnd])
        rnd += 1
    out = []
    for s in picked:
        ds, ep, st = s["dataset"], int(s["episode_index"]), int(s.get("start", 0))
        out.append(_pack(s["sample_id"], episode_io.read_frames(ds, ep, st, seq),
                         episode_io.read_actions(ds, ep, st, seq), cfg))
    return out


def run_group(rt, name: str, samples: list[dict], steps: int) -> dict:
    """한 묶음(학습/미학습) 생성 → 유지율·GT 유지율·PSNR."""
    import torch

    from src.models.irasim_runtime import encode_video
    from src.models.irasim_runtime import generate as rt_generate

    print("\n=== %s (%d개) ===" % (name, len(samples)), flush=True)
    rows = []
    for s in samples:
        gt = s["gt"]
        x = torch.from_numpy(gt[:1]).float().div(127.5).sub(1.0).permute(0, 3, 1, 2)[None]  # (1,1,3,H,W)
        mask_x = encode_video(rt, x)
        a15 = torch.from_numpy(s["actions15"]).float()[None]
        pred = np.asarray(rt_generate(rt, mask_x, a15, steps=steps, eta=0.0, method="PNDM",
                                      seed=0, gen_hw=GEN_HW)[0])                            # (16,H,W,3)
        g0, g1 = _lap_var(_gray(pred[0])), _lap_var(_gray(pred[-1]))
        t0, t1 = _lap_var(_gray(gt[0])), _lap_var(_gray(gt[-1]))
        r = {"sample": s["id"], "group": name,
             "sharp_in": t0, "sharp_out_last": g1, "sharp_out_first": g0, "sharp_gt_last": t1,
             "keep_pct": 100.0 * g1 / max(g0, 1e-9), "gt_keep_pct": 100.0 * t1 / max(t0, 1e-9),
             "psnr_last": _psnr(pred[-1], gt[-1]), "psnr_first": _psnr(pred[0], gt[0])}
        rows.append(r)
        print("  %-44s 입력 %6.0f → 출력 %6.0f  유지 %5.1f%%  PSNR %4.1f dB"
              % (s["id"][-44:], t0, g1, r["keep_pct"], r["psnr_last"]), flush=True)
        del mask_x, pred
        torch.cuda.empty_cache()

    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("sharp_in", "sharp_out_last", "keep_pct", "gt_keep_pct", "psnr_last", "psnr_first")}
    print("  → 평균 입력 %.0f → 출력 %.0f · 유지 %.1f%% · PSNR %.1f dB"
          % (agg["sharp_in"], agg["sharp_out_last"], agg["keep_pct"], agg["psnr_last"]), flush=True)
    return {"rows": rows, "agg": agg}


def ceiling_check(rows: list[dict]) -> dict:
    """입력 선명도가 올라갈 때 **출력 절대 선명도**가 따라 오르는지 — 천장 유무 판정.

    유지율(마지막÷첫)은 모델이 입력과 무관하게 늘 같은 선명도를 뱉어도 입력이 또렷할수록
    낮게 나온다. 그래서 유지율만 보면 "이 장면이 더 뭉개진다"는 착시가 생긴다.
    로그-로그 기울기로 가른다: 1이면 입력만큼 따라 오름, 0이면 완전한 천장.
    """
    xi = np.log(np.array([max(r["sharp_in"], 1e-9) for r in rows]))
    yo = np.log(np.array([max(r["sharp_out_last"], 1e-9) for r in rows]))
    slope, _ = np.polyfit(xi, yo, 1) if len(rows) >= 3 else (float("nan"), 0.0)
    corr = float(np.corrcoef(xi, yo)[0, 1]) if len(rows) >= 3 else float("nan")
    keep = np.array([r["keep_pct"] for r in rows])
    corr_keep = float(np.corrcoef(xi, keep)[0, 1]) if len(rows) >= 3 else float("nan")
    out = {"n": len(rows), "loglog_slope": float(slope), "corr_in_out": corr,
           "corr_in_keep": corr_keep,
           "out_sharp_cv": float(np.std([r["sharp_out_last"] for r in rows])
                                 / max(np.mean([r["sharp_out_last"] for r in rows]), 1e-9))}
    print("\n=== 선명도 천장 검사 (전체 %d개) ===" % out["n"])
    print("  입력↑ 대비 출력↑ 기울기 %.2f (1=입력 따라감, 0=천장) · 상관 %.2f" % (slope, corr))
    print("  입력 선명도 ↔ 유지율 상관 %.2f (음수면 '또렷한 장면일수록 유지율 낮음' = 착시 성분)"
          % corr_keep)
    print("  출력 선명도 변동계수 %.2f (작을수록 입력 무관하게 일정 = 천장)" % out["out_sharp_cv"])
    print("  %s" % ("→ 천장 있음. 유지율 차이의 상당 부분은 입력 선명도 차이가 만든 착시"
                    if slope < 0.35 else
                    "→ 천장 아님. 출력이 입력을 따라감 — 유지율 차이는 실제 차이"))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="    %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpts/bridge_frame_ada_0300000.pt")
    ap.add_argument("--n", type=int, default=6, help="묶음당 샘플 수")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--tag", default="", help="결과 파일 접미사 (예: --tag ft14000)")
    args = ap.parse_args()

    from src.models.irasim_runtime import build
    from src.utils.config import load_config

    cfg = load_config()
    cfg.setdefault("train", {})["precision"] = "fp16"     # V100
    cfg.setdefault("data", {})["pre_encode"] = False

    tr, un = train_samples(cfg, args.n), unseen_samples(cfg, args.n)
    print("ckpt: %s" % args.ckpt)
    rt = build(cfg, slim_ckpt=args.ckpt, logger=logging.getLogger("rt"))
    res = {"ckpt": args.ckpt, "steps": args.steps,
           "train": run_group(rt, "학습에 쓴 클립", tr, args.steps),
           "unseen": run_group(rt, "처음 보는 클립(대조군)", un, args.steps)}

    res["ceiling"] = ceiling_check(res["train"]["rows"] + res["unseen"]["rows"])

    d = res["train"]["agg"]["keep_pct"] - res["unseen"]["agg"]["keep_pct"]
    dp = res["train"]["agg"]["psnr_last"] - res["unseen"]["agg"]["psnr_last"]
    res["delta_keep_pp"], res["delta_psnr_db"] = d, dp
    print("\n=== 학습 vs 미학습 ===")
    print("  유지율   학습 %.1f%%  vs  미학습 %.1f%%  → 차이 %+.1f%%p" %
          (res["train"]["agg"]["keep_pct"], res["unseen"]["agg"]["keep_pct"], d))
    print("  PSNR     학습 %.1f dB vs  미학습 %.1f dB → 차이 %+.1f dB" %
          (res["train"]["agg"]["psnr_last"], res["unseen"]["agg"]["psnr_last"], dp))
    print("  입력 선명도 학습 %.0f / 미학습 %.0f  (크게 다르면 아래 판정은 무의미 — 장면 난이도 교란)" %
          (res["train"]["agg"]["sharp_in"], res["unseen"]["agg"]["sharp_in"]))
    if "bridge_frame_ada" in str(args.ckpt):
        print("  ★이 ckpt는 IRASim 사전학습 원본 — 우리 학습 풀을 본 적이 없다."
              "  '학습 클립'도 미학습이므로 이 비교는 **장면 난이도 대조군**으로만 읽을 것")
    elif abs(res["train"]["agg"]["sharp_in"] - res["unseen"]["agg"]["sharp_in"]) > \
            0.4 * res["unseen"]["agg"]["sharp_in"]:
        print("  → 두 묶음의 입력 선명도가 40% 넘게 달라 판정 보류(장면 난이도 통제 실패)")
    else:
        print("  %s" % ("→ 학습한 건 재현함. 원인은 **학습량·일반화** (EMA·추가 학습이 유효)"
                        if d > 15 or dp > 2 else
                        "→ 학습한 클립조차 못 맞춤. 원인은 **모델·설정 한계** (추가 학습 무익, 교체 검토)"))

    (ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results" / ("diag_train_repro%s.json" % (("_" + args.tag) if args.tag else ""))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("\n→ %s" % out.relative_to(ROOT))


if __name__ == "__main__":
    main()
