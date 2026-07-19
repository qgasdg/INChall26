"""동종 best-of-N 분산 EDA (48h 계획 Phase 3).

배경·목적:
  exp-01에서 "이종(異種) best-of-2는 무익"이 확정됐다 — 정적 예측과 베이스라인은 서로
  잘하는 샘플이 겹치지 않지만(상관 0.22), 우세 폭이 손익분기에 못 미쳤다. 남은 길은
  **동종(同種) best-of-N**, 즉 같은 모델을 seed(난수 씨앗)만 바꿔 여러 번 돌리고 그중
  가장 좋은 것을 고르는 방식이다. 이게 값어치가 있으려면 **같은 샘플이 seed에 따라
  충분히 흔들려야** 한다. 그 흔들림(분산)을 처음으로 실측한다.

  전제: 생성은 seed가 같으면 완전히 동일하다(exp-05에서 mp4 해시 216/216 일치로 실증).
        따라서 seed를 바꾸는 것이 유일한 변동 요인이다.

방법: eval 소표본 N샘플 × seed 8개를 step 20으로 생성 → 킷으로 Action MAE 실측 →
      ① 샘플 내 분산 ② best-of-N 기대 이득 곡선 ③ 시간 예산(16.7초/샘플) 환산.

용어: seed(난수 씨앗, 같은 값이면 같은 결과가 나오게 하는 초기값) · MAE(Mean Absolute
      Error, 평균 절대 오차 — 낮을수록 좋음) · best-of-N(N번 시도 후 최선 하나 채택).

주의: best-of-N을 실제로 쓰려면 "어느 것이 최선인지" 제출 전에 알아야 한다. Action 성분은
      킷으로 로컬 실측이 가능하지만(룰북 §3) DINO/Video 성분은 정답 영상이 없으면 못 잰다.
      본 실험은 **이득의 상한**을 재는 것이며, 실전 선택 규칙은 별도 설계가 필요하다.

사용: uv run python scripts/eda06_bestofn_variance.py [--n-samples 30] [--seeds 8] [--steps 20]
산출: results/bestofn_per_sample.csv (샘플×seed Action MAE), results/bestofn_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_eval"))
CK = ROOT / "open" / "baseline" / "challenge_kit"
KIT = ROOT / "open" / "submission_kit"
EVAL = ROOT / "open" / "data" / "eval"

from kit_bridge import KitScorer, TEMPORAL_LENGTH, get_device  # noqa: E402
import kit_bridge  # noqa: E402

kit = kit_bridge.kit


def build_subset(sample_ids: list[str], out: Path) -> None:
    """eval 216개 중 지정 샘플만 담은 challenge 루트를 만든다 (하드링크 — 복사 비용 0)."""
    for sub in ("images", "actions"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for sid in sample_ids:
        for sub, ext in (("images", ".png"), ("actions", ".npy")):
            dst = out / sub / f"{sid}{ext}"
            if not dst.exists():
                try:
                    dst.hardlink_to(EVAL / sub / f"{sid}{ext}")
                except OSError:
                    shutil.copy(EVAL / sub / f"{sid}{ext}", dst)


def generate(root: Path, pred: Path, steps: int, seed: int, log: Path) -> float:
    pred.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/inference/generate_baseline_videos.py",
           "--challenge-root", str(root.resolve()), "--prediction-root", str(pred.resolve()),
           "--ddim-steps", str(steps), "--seed", str(seed), "--overwrite"]
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=CK, check=True, stdout=f, stderr=subprocess.STDOUT)
    return time.time() - t0


def action_mae_direct(pred: Path, scorer, act_root: Path) -> dict[str, float]:
    """킷 추출기로 직접 Action MAE 계산.

    킷의 make_submission_csv.py는 eval 216개 전량을 요구하므로 소표본에 쓸 수 없다.
    대신 킷과 동일한 추출기·정규화를 쓰는 kit_bridge 경로를 쓴다.
    이 경로가 킷 공식 산출값과 일치함은 exp-08에서 검증됐다(216개 평균 0.5877 일치,
    scripts/eda11_action_axis_error.py).
    """
    import numpy as np
    out = {}
    for p_mp4 in sorted(pred.glob("*.mp4")):
        sid = p_mp4.stem
        gt_deg = np.load(act_root / f"{sid}.npy")
        gen = kit.read_video_uint8(p_mp4, expected_frames=TEMPORAL_LENGTH).numpy()
        video = scorer.to_eval_video(gen).unsqueeze(0)
        pred_z = scorer.action_pred(video)[0]
        gt_z = scorer.normalize_actions(gt_deg)
        out[sid] = float((pred_z - gt_z).abs().mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--workdir", default="local_runs/bestofn")
    args = ap.parse_args()

    work = ROOT / args.workdir
    work.mkdir(parents=True, exist_ok=True)

    # 표본: eval 216에서 고르게 뽑아 셋업 A/B가 모두 들어가게 한다 (결정론적)
    all_ids = sorted(p.stem for p in (EVAL / "images").glob("sample_*.png"))
    idx = np.linspace(0, len(all_ids) - 1, args.n_samples).astype(int)
    sample_ids = [all_ids[i] for i in idx]
    n_b = sum(int(s.split("_")[1]) >= 154 for s in sample_ids)
    print(f"표본 {len(sample_ids)}개 (셋업 A {len(sample_ids) - n_b} / B {n_b}), "
          f"seed {args.seeds}개, step {args.steps}", flush=True)

    subset = work / "challenge_subset"
    build_subset(sample_ids, subset)

    scorer = KitScorer(get_device(None))
    act_root = EVAL / "actions"
    records, timings = [], []
    for seed in range(args.seeds):
        pred = work / f"pred_seed{seed}"
        gen_s = generate(subset, pred, args.steps, seed, work / f"gen_seed{seed}.log")
        maes = action_mae_direct(pred, scorer, act_root)
        for sid, mae in maes.items():
            records.append({"sample_id": sid, "seed": seed, "action_mae": mae})
        timings.append({"seed": seed, "gen_total_s": round(gen_s, 1),
                        "per_sample_s": round(gen_s / len(sample_ids), 2)})
        print(f"seed {seed}: {gen_s:.0f}s ({gen_s / len(sample_ids):.1f}s/샘플), "
              f"Action MAE 평균 {np.mean(list(maes.values())):.4f}", flush=True)
        # 영상 원본은 용량이 크므로 채점 후 즉시 삭제 (룰북 §5 — 대용량 산출물 비커밋)
        shutil.rmtree(pred, ignore_errors=True)

    df = pd.DataFrame(records)
    df.to_csv(ROOT / "results" / "bestofn_per_sample.csv", index=False)

    piv = df.pivot(index="sample_id", columns="seed", values="action_mae")
    within = piv.std(axis=1)
    rng = piv.max(axis=1) - piv.min(axis=1)

    # best-of-N 기대 이득: seed 조합을 전부 세지 않고, 각 N에 대해 무작위 N개 뽑기를
    # 반복 추정한다 (조합 수가 작으면 전수와 사실상 같음). 결정론 위해 시드 고정.
    rs = np.random.RandomState(0)
    curve = []
    vals = piv.to_numpy()
    for n in range(1, args.seeds + 1):
        picks = [vals[:, rs.choice(args.seeds, n, replace=False)].min(axis=1).mean()
                 for _ in range(200)]
        curve.append({"N": n, "expected_action_mae": round(float(np.mean(picks)), 5),
                      "gain_vs_N1": None})
    base = curve[0]["expected_action_mae"]
    for c in curve:
        c["gain_vs_N1"] = round(base - c["expected_action_mae"], 5)

    per_sample_s = float(np.mean([t["per_sample_s"] for t in timings]))
    summary = pd.DataFrame(curve)
    summary["time_per_sample_s"] = (summary.N * per_sample_s).round(2)
    summary["within_budget_16.7s"] = summary.time_per_sample_s <= 16.7
    summary.to_csv(ROOT / "results" / "bestofn_summary.csv", index=False)
    (ROOT / "results" / "bestofn_timing.json").write_text(json.dumps(timings, indent=1))

    print(f"\n=== 샘플 내 seed 변동 (n={len(piv)}) ===")
    print(f"샘플별 표준편차: 중앙값 {within.median():.4f} / 최대 {within.max():.4f}")
    print(f"샘플별 최대-최소 폭: 중앙값 {rng.median():.4f} / 최대 {rng.max():.4f}")
    print(f"seed 평균 MAE의 seed간 표준편차: {piv.mean(axis=0).std():.5f}")
    print(f"\n=== best-of-N 곡선 (생성 {per_sample_s:.1f}초/샘플/회 실측) ===")
    print(summary.to_string(index=False))
    print("\n주의: Action 성분만 로컬 실측 가능. 실전 선택 규칙은 별도 설계 필요(문서 참조).")


if __name__ == "__main__":
    main()
