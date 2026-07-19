"""배치 크기가 생성 결과를 바꾸는지 검증 + 처리량 실측.

■ 왜 하는가
E1 결론은 "추론 시간은 스텝 수가 아니라 고정 오버헤드가 지배하므로, 속도를 줄일 레버는
배치 병렬(한 번에 여러 샘플을 묶어 처리)"이었다. 그런데 생성 코드를 읽어보니
난수(무작위 값) 처리 방식 때문에 **배치 크기를 바꾸면 결과 영상 자체가 달라질 가능성**이
있다. 그렇다면 배치 병렬은 "공짜 속도 향상"이 아니라 결과가 바뀌는 변경이 된다.

■ 근거 (generate_baseline_videos.py 코드 추적)
  153행: torch.manual_seed(seed)  ← 루프 시작 전 딱 한 번
   86행: for start in range(0, len(sample_ids), args.batch_size)
  107행: sampler.sample(batch_size=z.shape[0], ...)  ← 초기 잡음을 배치 단위로 한 번에 추출
  난수 상태가 샘플들 사이에서 순차적으로 소비되므로, 배치 4로 randn(4,...)를 한 번 뽑는 것과
  배치 1로 randn(1,...)를 네 번 뽑는 것은 서로 다른 값이 나온다.

■ 부수 검증: 재개(resume) 안전성
   89행: if not args.overwrite and all(path.exists()): continue
  이미 만들어진 파일은 건너뛰는데 **난수도 함께 건너뛴다**. 즉 중단된 실행을 이어서 돌리면
  남은 샘플들이 처음부터 통으로 돌렸을 때와 다른 값을 받는다.

■ 측정
  같은 샘플 8개를 배치 1/2/4로 각각 생성해 ① mp4 해시 일치 여부 ② 샘플당 처리 시간 비교.
  해시가 다르면 "배치는 결과를 바꾼다"가 실증되고, 시간이 줄면 배치 이득의 크기가 나온다.

용어: 배치(batch, 한 번에 묶어 처리하는 단위) · 해시(hash, 파일 내용을 요약한 지문 —
      한 글자만 달라도 완전히 바뀜) · 난수 씨앗(seed, 무작위 값 생성의 초기값).

사용: .venv/bin/python scripts/eda08_batch_determinism.py [--n 8] [--batches 1 2 4]
      (주의: uv run 금지 — editable 설치가 지워짐. setup_generation.sh 주석 참조)
산출: results/batch_determinism.csv
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CK = ROOT / "open" / "baseline" / "challenge_kit"
EVAL = ROOT / "open" / "data" / "eval"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def build_subset(sample_ids: list[str], out: Path) -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--workdir", default="local_runs/batch_test")
    args = ap.parse_args()

    work = ROOT / args.workdir
    work.mkdir(parents=True, exist_ok=True)
    sample_ids = sorted(p.stem for p in (EVAL / "images").glob("sample_*.png"))[: args.n]
    subset = work / "subset"
    build_subset(sample_ids, subset)
    print(f"샘플 {len(sample_ids)}개, 배치 {args.batches}, step {args.steps}", flush=True)

    hashes, rows = {}, []
    for bs in args.batches:
        pred = work / f"pred_bs{bs}"
        shutil.rmtree(pred, ignore_errors=True)
        pred.mkdir(parents=True)
        cmd = [sys.executable, "scripts/inference/generate_baseline_videos.py",
               "--challenge-root", str(subset.resolve()),
               "--prediction-root", str(pred.resolve()),
               "--ddim-steps", str(args.steps), "--seed", "0",
               "--batch-size", str(bs), "--overwrite"]
        t0 = time.time()
        with open(work / f"gen_bs{bs}.log", "w") as f:
            subprocess.run(cmd, cwd=CK, check=True, stdout=f, stderr=subprocess.STDOUT)
        wall = time.time() - t0
        hashes[bs] = {sid: md5(pred / f"{sid}.mp4") for sid in sample_ids}
        rows.append({"batch_size": bs, "wall_s": round(wall, 1),
                     "per_sample_s": round(wall / len(sample_ids), 2)})
        print(f"배치 {bs}: {wall:.0f}s ({wall / len(sample_ids):.1f}s/샘플)", flush=True)
        shutil.rmtree(pred, ignore_errors=True)

    ref = args.batches[0]
    for r in rows:
        bs = r["batch_size"]
        same = sum(hashes[bs][s] == hashes[ref][s] for s in sample_ids)
        r["identical_to_batch%d" % ref] = f"{same}/{len(sample_ids)}"
        r["speedup_vs_batch%d" % ref] = round(
            rows[0]["per_sample_s"] / r["per_sample_s"], 2)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "batch_determinism.csv", index=False)
    print("\n=== 결과 ===")
    print(df.to_string(index=False))
    allsame = all(r[f"identical_to_batch{ref}"] == f"{len(sample_ids)}/{len(sample_ids)}"
                  for r in rows)
    print(f"\n판정: 배치 크기를 바꿔도 결과가 {'동일 — 배치 병렬은 공짜 속도 향상' if allsame else '달라짐 — 배치 병렬은 결과를 바꾸는 변경 (최종 제출 설정 고정 필요)'}")
    print("→ results/batch_determinism.csv")


if __name__ == "__main__":
    main()
