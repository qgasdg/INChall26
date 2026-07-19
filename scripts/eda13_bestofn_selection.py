"""best-of-N 선택 규칙 검증 — Action만 보고 고르면 총점이 오르는가?

■ 왜 하는가 (Phase 3의 미해결 질문)
Phase 3에서 best-of-N(같은 문제를 여러 번 풀고 최선을 고르기)의 이득이 크다는 것을
확인했다(N=3에서 Action 오차 -0.090). 그런데 실전에서 쓰려면 **어느 것이 최선인지
제출 전에 알아야** 한다.

  - Action 성분(총점의 40%): eval의 행동 시퀀스가 우리에게 주어진 입력이므로
    로컬에서 **정확히** 계산할 수 있다. → 고를 수 있다.
  - DINO·Video 성분(총점의 60%): 정답 영상이 없으면 못 잰다. → 고를 수 없다.

따라서 실전 규칙은 "Action이 가장 좋은 것을 고른다"가 된다. 문제는 이게 **총점에도
좋은가**이다. Action만 좇다가 화질·움직임이 엉망인 영상을 골라 DINO/Video를 잃으면
전체로는 손해일 수 있다.

■ 어떻게 검증하는가
홀드아웃 v2(우리가 떼어둔 모의고사)는 **정답 영상이 있으므로 총점을 실제로 잴 수 있다.**
같은 샘플을 seed만 바꿔 N번 생성하고, 세 가지를 비교한다.

  (A) N=1                  : 아무 것도 안 고름 (기준)
  (B) Action 기준 선택      : 실전에서 실제로 할 수 있는 것
  (C) 총점 기준 선택(오라클) : 정답을 알아야만 가능. 이득의 상한

(B)가 (A)보다 좋으면 best-of-N은 쓸 수 있다. (B)가 (C)에 가까울수록 Action이 좋은
대리 지표라는 뜻이다. (B)가 (A)보다 나쁘면 이 전략은 폐기한다.

■ 한계 (중요)
베이스라인 모델은 주최측이 train 데이터로 미리 학습한 것이고, 홀드아웃 v2도 train에서
떼어낸 것이므로 **모델이 이 장면들을 이미 봤을 수 있다**. 따라서 절대 점수는 eval보다
낙관적이다. 다만 여기서 묻는 것은 절대 점수가 아니라 **"Action 기준 선택과 총점 기준
선택이 같은 방향인가"**이므로, 이 편향이 결론을 뒤집을 가능성은 낮다(추정).

용어: 오라클(oracle, 정답을 아는 가상의 선택자 — 실전에선 불가능하지만 이득의 상한을 준다) ·
      상관(두 값이 함께 움직이는 정도) · D+V(DINO와 Video 성분).

사용: .venv/bin/python scripts/eda13_bestofn_selection.py [--seeds 4] [--steps 20]
산출: results/bestofn_selection.csv, results/bestofn_selection_per_sample.csv
"""
from __future__ import annotations

import argparse
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
INPUTS = ROOT / "local_runs" / "v2_unseen_inputs"   # make_challenge_inputs_v2.py 산출


def generate(pred: Path, steps: int, seed: int, log: Path) -> float:
    pred.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/inference/generate_baseline_videos.py",
           "--challenge-root", str(INPUTS.resolve()), "--prediction-root", str(pred.resolve()),
           "--ddim-steps", str(steps), "--seed", str(seed), "--overwrite"]
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=CK, check=True, stdout=f, stderr=subprocess.STDOUT)
    return time.time() - t0


def score(videos: Path, out_csv: Path) -> pd.DataFrame:
    """정본 채점기(score_v2.py)를 그대로 호출 — 채점 로직 중복 구현 금지."""
    subprocess.run([sys.executable, str(ROOT / "local_eval" / "score_v2.py"),
                    "--videos", str(videos), "--tiers", "unseen", "--csv", str(out_csv)],
                   check=True, cwd=ROOT / "local_eval", stdout=subprocess.DEVNULL)
    return pd.read_csv(out_csv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--workdir", default="local_runs/bon_sel")
    args = ap.parse_args()

    if not INPUTS.exists():
        raise SystemExit(f"{INPUTS} 없음 — 먼저 local_eval/make_challenge_inputs_v2.py 실행")

    work = ROOT / args.workdir
    work.mkdir(parents=True, exist_ok=True)
    frames = []
    for seed in range(args.seeds):
        pred = work / f"pred_seed{seed}"
        gen_s = generate(pred, args.steps, seed, work / f"gen_seed{seed}.log")
        df = score(pred, work / f"scores_seed{seed}.csv")
        df["seed"] = seed
        frames.append(df)
        print(f"seed {seed}: 생성 {gen_s:.0f}s, n={len(df)}, "
              f"총점(프레임평균) {df.total_pf.mean():.4f} "
              f"[D {df.dino_pf.mean():.4f} V {df.video.mean():.4f} A {df.action.mean():.4f}]",
              flush=True)
        shutil.rmtree(pred, ignore_errors=True)   # 영상은 용량 커서 즉시 삭제

    per = pd.concat(frames, ignore_index=True)
    per.to_csv(ROOT / "results" / "bestofn_selection_per_sample.csv", index=False)

    tot = per.pivot(index="sample", columns="seed", values="total_pf")
    act = per.pivot(index="sample", columns="seed", values="action")
    dino = per.pivot(index="sample", columns="seed", values="dino_pf")
    vid = per.pivot(index="sample", columns="seed", values="video")

    rows = []
    for n in range(1, args.seeds + 1):
        # 앞의 n개 seed만 쓸 수 있다고 가정 (결정론적, 조합 추출 없이)
        cols = list(range(n))
        a_pick = act[cols].to_numpy().argmin(axis=1)            # Action 기준 선택
        t_pick = tot[cols].to_numpy().argmin(axis=1)            # 총점 기준 선택(오라클)
        idx = np.arange(len(tot))
        rows.append({
            "N": n,
            "total_N1": round(float(tot[0].mean()), 5),
            "total_action_select": round(float(tot[cols].to_numpy()[idx, a_pick].mean()), 5),
            "total_oracle": round(float(tot[cols].to_numpy()[idx, t_pick].mean()), 5),
            "action_action_select": round(float(act[cols].to_numpy()[idx, a_pick].mean()), 5),
            "dino_action_select": round(float(dino[cols].to_numpy()[idx, a_pick].mean()), 5),
            "video_action_select": round(float(vid[cols].to_numpy()[idx, a_pick].mean()), 5),
            "pick_agreement_pct": round(100 * float((a_pick == t_pick).mean()), 1),
        })
    df = pd.DataFrame(rows)
    df["gain_action_select"] = (df.total_N1 - df.total_action_select).round(5)
    df["gain_oracle"] = (df.total_N1 - df.total_oracle).round(5)
    df["capture_of_oracle_pct"] = (100 * df.gain_action_select
                                   / df.gain_oracle.replace(0, np.nan)).round(1)
    df.to_csv(ROOT / "results" / "bestofn_selection.csv", index=False)

    # Action과 총점이 같은 방향인지 — 샘플 안에서 seed끼리 비교한 순위상관
    rhos = []
    for s in tot.index:
        a, t = act.loc[s], tot.loc[s]
        if a.std() > 0 and t.std() > 0:
            rhos.append(float(a.rank().corr(t.rank())))
    print(f"\n=== 샘플 내 seed 간 Action↔총점 순위상관: 중앙값 {np.median(rhos):+.3f} "
          f"(양수면 Action이 좋을수록 총점도 좋다는 뜻) ===")
    print("\n=== 선택 규칙 비교 (홀드아웃 v2 unseen, 총점은 낮을수록 좋음) ===")
    print(df.to_string(index=False))
    best = df.iloc[-1]
    print(f"\n판정: N={int(best.N)}에서 Action 기준 선택의 총점 이득 {best.gain_action_select:+.5f} "
          f"(오라클 {best.gain_oracle:+.5f}의 {best.capture_of_oracle_pct}%)")
    print("  → 이득이 양수면 실전 적용 가능. 음수면 Action 기준 선택은 폐기.")
    print("→ results/bestofn_selection.csv")


if __name__ == "__main__":
    main()
