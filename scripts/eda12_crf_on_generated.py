"""인코딩 손실 재측정 — 실제 움직이는 생성 영상 기준.

■ 왜 다시 재는가
앞선 CRF 스윕(scripts/eda04_crf_sweep.py)은 **정적 예측**(첫 프레임을 16번 반복)으로 쟀다.
정적 영상은 16장이 전부 같아서 프레임 간 압축이 거의 무손실이 되는 퇴화 사례라,
인코딩 손실을 과소평가한다. 실제 제출물은 움직이는 영상이므로 그것으로 다시 잰다.

■ 방법
exp-08이 생성한 eval 216개 영상을 원본으로 삼아, 프레임을 디코딩한 뒤 CRF·픽셀 포맷을
바꿔 재인코딩하고 킷 추출기로 Action 성분을 다시 잰다.

한계: 원본 mp4 자체가 이미 한 번 인코딩된 것이라 "무인코딩 기준선"을 만들 수 없다.
      따라서 절대 손실이 아니라 **설정 간 상대 차이**를 본다. 다만 재인코딩 손실은
      원본 인코딩 손실과 같은 성격이므로 방향과 크기 규모는 읽을 수 있다.

용어: CRF(Constant Rate Factor, 압축 품질 조절값 — 낮을수록 고품질) ·
      yuv420p(색 정보를 가로·세로 절반으로 줄여 저장) · yuv444p(색 정보 그대로 저장) ·
      MAE(Mean Absolute Error, 평균 절대 오차).

사용: .venv/bin/python scripts/eda12_crf_on_generated.py --videos local_runs/pred_v100_s50
산출: results/crf_on_generated.csv
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_eval"))

from kit_bridge import KitScorer, TEMPORAL_LENGTH, get_device  # noqa: E402
import kit_bridge  # noqa: E402

kit = kit_bridge.kit
FPS = 6


def reencode(frames: np.ndarray, out: Path, crf: int, pix_fmt: str) -> None:
    with av.open(str(out), "w") as c:
        st = c.add_stream("libx264", rate=FPS)
        st.height, st.width = frames.shape[1:3]
        st.pix_fmt = pix_fmt
        st.options = {"crf": str(crf), "preset": "medium"}
        for f in frames:
            for p in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="local_runs/pred_v100_s50")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    scorer = KitScorer(get_device(None))
    act_root = ROOT / "open" / "data" / "eval" / "actions"
    src = sorted(Path(args.videos).glob("*.mp4"))
    if args.limit:
        src = src[: args.limit]
    print(f"원본 영상 {len(src)}개", flush=True)

    # 원본 프레임을 한 번만 디코딩해 메모리에 올린다 (216 x 16 x 480 x 640 x 3 ≈ 3GB이므로
    # 샘플 단위로 처리하며 설정별로 즉시 재인코딩·채점한다)
    configs = [(0, "yuv420p"), (10, "yuv420p"), (23, "yuv420p"), (28, "yuv420p"),
               (10, "yuv444p"), (0, "yuv444p")]
    work = ROOT / "local_runs" / "crf_gen"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    acc = {c: [] for c in configs}
    base = []
    for n, p in enumerate(src, 1):
        frames = kit.read_video_uint8(p, expected_frames=TEMPORAL_LENGTH).numpy()
        gt_z = scorer.normalize_actions(np.load(act_root / f"{p.stem}.npy"))

        # 원본(재인코딩 없음) 기준값
        pred_z = scorer.action_pred(scorer.to_eval_video(frames).unsqueeze(0))[0]
        base.append(float((pred_z - gt_z).abs().mean()))

        for crf, pix in configs:
            tmp = work / f"{p.stem}_{crf}_{pix}.mp4"
            reencode(frames, tmp, crf, pix)
            fr2 = kit.read_video_uint8(tmp, expected_frames=TEMPORAL_LENGTH).numpy()
            pz = scorer.action_pred(scorer.to_eval_video(fr2).unsqueeze(0))[0]
            acc[(crf, pix)].append({
                "mae": float((pz - gt_z).abs().mean()),
                "kb": tmp.stat().st_size / 1024,
                # 픽셀 자체가 얼마나 바뀌었는지 (점수와 무관한 직접 지표)
                "px_mae": float(np.abs(fr2.astype(np.int16) - frames.astype(np.int16)).mean()),
            })
            tmp.unlink()
        if n % 40 == 0 or n == len(src):
            print(f"  {n}/{len(src)}", flush=True)

    base_mae = float(np.mean(base))
    rows = [{"crf": "원본(재인코딩 없음)", "pix_fmt": "-", "action_mae": round(base_mae, 5),
             "delta_vs_base": 0.0, "kb_per_file": None, "pixel_mae": 0.0}]
    for (crf, pix), vals in acc.items():
        mae = float(np.mean([v["mae"] for v in vals]))
        rows.append({
            "crf": crf, "pix_fmt": pix,
            "action_mae": round(mae, 5),
            "delta_vs_base": round(mae - base_mae, 5),
            "kb_per_file": round(float(np.mean([v["kb"] for v in vals])), 1),
            "pixel_mae": round(float(np.mean([v["px_mae"] for v in vals])), 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "crf_on_generated.csv", index=False)
    shutil.rmtree(work, ignore_errors=True)

    print(f"\n=== 실제 생성 영상 재인코딩 (n={len(src)}) ===")
    print(df.to_string(index=False))
    print("\n읽는 법: delta_vs_base가 0에 가까울수록 재인코딩으로 잃는 것이 없다는 뜻.")
    print("         pixel_mae는 화소값이 평균 몇 단계(0~255) 변했는지 — 점수와 무관한 직접 지표.")
    print("→ results/crf_on_generated.csv")


if __name__ == "__main__":
    main()
