"""CRF(Constant Rate Factor, 영상 압축 품질 조절값) 스윕 — 인코딩이 점수에 주는 영향 정량화.

배경: S003 제출에서 로컬 예측과 서버 실측이 D+V(DINO+Video 성분) 기준 +0.002 어긋났고,
      원인을 "mp4 인코딩 차이"로 추정했다(docs/LOG 7/18). 그 추정을 실측으로 확인한다.
      낮은 CRF = 고품질·큰 용량, 높은 CRF = 저품질·작은 용량 (0=무손실에 가까움, 51=최악).

방법: 홀드아웃 v2 unseen 162샘플의 정적 예측(GT 첫 프레임 16회 반복)을 CRF별로 인코딩 →
      다시 디코딩해 채점 → CRF와 D+V 점수의 공선(共線, 함께 움직이는 관계)을 본다.
      Action 성분은 정적 예측이면 인코딩과 무관하지 않으므로 함께 기록한다.

기준점: score_v2 --static 은 인코딩을 전혀 거치지 않은 메모리 상 배열을 채점한다.
        따라서 그 값이 "인코딩 손실 0"의 이론적 상한이고, CRF별 값은 여기서 얼마나
        멀어지는지를 나타낸다.

사용: uv run python scripts/eda04_crf_sweep.py [--crfs 0 10 18 23 28] [--limit N]
산출: results/crf_sweep.csv (CRF별 성분 평균), results/crf_sweep_per_sample.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_eval"))

FPS = 6
N_FRAMES = 16


def write_static_mp4(frame_rgb: np.ndarray, out_path: Path, crf: int) -> None:
    """단일 프레임을 16회 반복해 mp4로 인코딩 (make_static_videos.py와 동일 규약, crf만 가변)."""
    with av.open(str(out_path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.height, stream.width = frame_rgb.shape[0], frame_rgb.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "medium"}
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for _ in range(N_FRAMES):
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crfs", type=int, nargs="+", default=[0, 10, 18, 23, 28])
    ap.add_argument("--limit", type=int, default=None, help="샘플 수 제한(디버그용)")
    ap.add_argument("--workdir", default="local_runs/crf_sweep")
    args = ap.parse_args()

    from episode_io import read_frames  # noqa: E402

    holdout = json.loads((ROOT / "local_eval" / "holdout_v2.json").read_text())
    samples = [s for s in holdout["samples"] if s["tier"] != "indomain"]
    if args.limit:
        samples = samples[: args.limit]
    print(f"샘플 {len(samples)}개 × CRF {args.crfs}", flush=True)

    work = ROOT / args.workdir
    rows = []
    for crf in args.crfs:
        vdir = work / f"crf{crf}"
        if vdir.exists():
            shutil.rmtree(vdir)
        vdir.mkdir(parents=True)

        t0 = time.time()
        nbytes = 0
        for s in samples:
            frame0 = read_frames(s["dataset"], s["episode_index"], s["start"], 1)[0]
            out = vdir / f"{s['sample_id']}.mp4"
            write_static_mp4(frame0, out, crf)
            nbytes += out.stat().st_size
        enc_s = time.time() - t0

        # 채점은 정본 채점기(score_v2.py)를 그대로 호출 — 채점 로직 중복 구현 금지
        csv_path = work / f"scores_crf{crf}.csv"
        subprocess.run([sys.executable, str(ROOT / "local_eval" / "score_v2.py"),
                        "--videos", str(vdir), "--tiers", "unseen",
                        "--csv", str(csv_path)], check=True, cwd=ROOT / "local_eval")
        df = pd.read_csv(csv_path)
        df["crf"] = crf
        rows.append(df)
        print(f"CRF {crf:2d}: 인코딩 {enc_s:.0f}s, 평균 {nbytes / len(samples) / 1024:.1f} KB/파일, "
              f"DINO {df.dino_pf.mean():.4f} Video {df.video.mean():.4f} Action {df.action.mean():.4f}",
              flush=True)

    per_sample = pd.concat(rows, ignore_index=True)
    per_sample.to_csv(ROOT / "results" / "crf_sweep_per_sample.csv", index=False)

    agg = (per_sample.groupby("crf")[["dino_pf", "dino_flat", "video", "action",
                                      "total_pf", "total_flat"]].mean().round(6))
    # 파일 크기도 함께 (품질-용량 트레이드오프)
    agg["kb_per_file"] = [
        round(sum(p.stat().st_size for p in (work / f"crf{c}").glob("*.mp4"))
              / len(samples) / 1024, 1) for c in agg.index]
    agg.to_csv(ROOT / "results" / "crf_sweep.csv")
    print("\n=== CRF별 평균 (홀드아웃 v2 unseen 162, 정적 예측) ===")
    print(agg.to_string())
    print("\n인코딩 무손실 기준선(score_v2 --static, 인코딩 미경유): "
          "DINO 0.1487 / Video 0.0848 / Action 1.1195  ← exp-02·exp-07")
    print(f"→ results/crf_sweep.csv, crf_sweep_per_sample.csv")


if __name__ == "__main__":
    main()
