"""베이스라인 생성 실행기 — 샘플당 타이밍 실측 + (선택) 킷 Action 채점.

Kaggle e1-sweep/e1-reanchor 커널의 generate()/kit_score()를 로컬 서버용으로 옮긴 것.
커널은 Kaggle 경로 가정이 박혀 있어 재사용 불가라, 같은 로직을 repo 자산으로 정착시킨다.

사용:
  # eval 216 (Action 정본, 룰북 §3)
  uv run python scripts/run_generation.py --tag s50 --ddim-steps 50 --score

  # 홀드아웃 v2 unseen 162 (D+V 재앵커용, --challenge-root로 변환 입력 지정)
  uv run python scripts/run_generation.py --tag v2s50 --ddim-steps 50 \
      --challenge-root local_runs/v2_unseen_inputs

출력: <workdir>/timing_<tag>.json, features_<tag>.csv, actions_<tag>.csv
      생성 mp4는 --prediction-root (기본 local_runs/pred_<tag>) — 용량 커서 커밋 금지.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CK = ROOT / "open" / "baseline" / "challenge_kit"
KIT = ROOT / "open" / "submission_kit"


def generate(tag: str, ddim_steps: int, pred: Path, challenge_root: str | None,
             workdir: Path, extra: list[str]) -> dict:
    """생성 실행. '[generate] wrote predictions' 줄에 타임스탬프를 찍어 샘플당 시간을 실측."""
    pred.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/inference/generate_baseline_videos.py",
           "--prediction-root", str(pred.resolve()),
           "--ddim-steps", str(ddim_steps), "--seed", "0", *extra]
    if challenge_root:
        cmd += ["--challenge-root", str(Path(challenge_root).resolve())]
    print(f"+ {' '.join(cmd)}  (cwd={CK})", flush=True)

    t0 = time.time()
    stderr_path = workdir / f"gen_{tag}.stderr"
    stamps: list[float] = []
    with open(stderr_path, "w") as errf:
        proc = subprocess.Popen(cmd, cwd=CK, stdout=subprocess.PIPE, stderr=errf,
                                text=True, bufsize=1,
                                env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
        passthrough = 0
        for line in proc.stdout:
            if "[generate] wrote predictions" in line:
                stamps.append(time.time())
                if len(stamps) % 20 == 0:
                    print(f"  {tag}: {len(stamps)}개, {time.time() - t0:.0f}s", flush=True)
            elif line.strip() and passthrough < 60:
                passthrough += 1
                print("   |", line.rstrip()[:200], flush=True)
        proc.wait()
    if proc.returncode != 0:
        print(stderr_path.read_text()[-4000:], flush=True)
        raise SystemExit(f"생성 실패 rc={proc.returncode} — {stderr_path}")

    # 첫 샘플은 모델 로드가 섞이므로 이후 간격만으로 샘플당 시간을 낸다 (커널과 동일 규약)
    deltas = [b - a for a, b in zip(stamps, stamps[1:])]
    timing = {
        "tag": tag, "ddim_steps": ddim_steps, "n": len(stamps),
        "total_s": round(time.time() - t0, 1),
        "per_sample_median_s": round(statistics.median(deltas), 3) if deltas else None,
        "per_sample_mean_s": round(sum(deltas) / len(deltas), 3) if deltas else None,
        "first_sample_incl_load_s": round(stamps[0] - t0, 1) if stamps else None,
        "server": "inha-v100",
    }
    (workdir / f"timing_{tag}.json").write_text(json.dumps(
        {**timing, "line_offsets_s": [round(t - t0, 2) for t in stamps]}, indent=1))
    print("타이밍:", timing, flush=True)
    return timing


def kit_score(tag: str, pred: Path, workdir: Path) -> float:
    """공식 킷으로 feature CSV 생성 → Action Component 평균 MAE."""
    out_csv = workdir / f"features_{tag}.csv"
    t = time.time()
    subprocess.run([sys.executable, "make_submission_csv.py",
                    "--prediction-root", str(pred.resolve()),
                    "--output-csv", str(out_csv.resolve())],
                   cwd=KIT, check=True)
    maes: dict[str, float] = {}
    with open(out_csv) as f:
        for row in csv.DictReader(f):
            if row["feature_component"] == "Action Component":
                v = json.loads(row["feature_json"])
                while isinstance(v, list):
                    v = v[0]
                maes[row["sample_id"]] = float(v)
    with open(workdir / f"actions_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "action_mae"])
        w.writerows(sorted(maes.items()))
    mean = sum(maes.values()) / len(maes)
    print(f"킷 채점 {len(maes)}개, {time.time() - t:.0f}s → Action MAE 평균 {mean:.4f}", flush=True)
    return mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ddim-steps", type=int, required=True)
    ap.add_argument("--challenge-root", default=None,
                    help="생략 시 config 기본값 = eval 216")
    ap.add_argument("--prediction-root", default=None)
    ap.add_argument("--workdir", default="local_runs")
    ap.add_argument("--score", action="store_true", help="킷 Action 채점까지 수행")
    ap.add_argument("extra", nargs="*", help="generate_baseline_videos.py 추가 인자")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pred = Path(args.prediction_root or workdir / f"pred_{args.tag}")

    timing = generate(args.tag, args.ddim_steps, pred, args.challenge_root, workdir, args.extra)
    result = {"timing": timing}
    if args.score:
        result["action_mae"] = kit_score(args.tag, pred, workdir)
    (workdir / f"result_{args.tag}.json").write_text(json.dumps(result, indent=1))
    print("완료:", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
