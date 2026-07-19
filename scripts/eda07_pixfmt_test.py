"""픽셀 포맷(pix_fmt) 가설 검증 — 인코딩 손실의 진짜 주범 찾기.

배경: CRF(Constant Rate Factor, 압축 품질 조절값) 스윕(scripts/eda04_crf_sweep.py) 결과,
      "인코딩을 아예 안 함 → CRF 0으로 인코딩"의 점수 손실(+0.0007)이
      "CRF 0 → CRF 23"의 손실(+0.0003)보다 컸다. 즉 품질 설정이 아니라 인코딩 행위
      자체가 주범이다.

가설: 범인은 yuv420p다. 이 픽셀 포맷은 밝기는 그대로 두고 **색상 정보만 가로·세로 각각
      절반으로 줄인다**(크로마 서브샘플링). CRF와 무관하게 항상 적용되므로 CRF를 아무리
      낮춰도 이 손실은 남는다. yuv444p(색상 정보 원본 유지)로 바꾸면 사라져야 한다.

검증: 같은 정적 예측을 yuv420p / yuv444p 두 포맷으로 각각 CRF 10에 인코딩해 채점 비교.
      기준선은 인코딩을 거치지 않은 값(exp-02·exp-07: DINO 0.1487 / Video 0.0848).

주의: 대회 제출물의 픽셀 포맷 요구사항은 별도 확인이 필요하다. 본 실험은 원인 규명이지
      제출 규격 변경 제안이 아니다.

사용: uv run python scripts/eda07_pixfmt_test.py
산출: results/pixfmt_test.csv
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import av
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_eval"))

CRF = 10
FPS = 6
N_FRAMES = 16
BASELINE = {"dino_pf": 0.1487, "video": 0.0848, "action": 1.1195}  # 인코딩 미경유 (exp-07)


def write_mp4(frame_rgb, out_path: Path, pix_fmt: str, crf: int) -> None:
    with av.open(str(out_path), "w") as c:
        st = c.add_stream("libx264", rate=FPS)
        st.height, st.width = frame_rgb.shape[:2]
        st.pix_fmt = pix_fmt
        st.options = {"crf": str(crf), "preset": "medium"}
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for _ in range(N_FRAMES):
            for p in st.encode(frame):
                c.mux(p)
        for p in st.encode():
            c.mux(p)


def main() -> None:
    from episode_io import read_frames  # noqa: E402

    holdout = json.loads((ROOT / "local_eval" / "holdout_v2.json").read_text())
    samples = [s for s in holdout["samples"] if s["tier"] != "indomain"]
    print(f"샘플 {len(samples)}개, CRF {CRF} 고정, 픽셀 포맷만 변경", flush=True)

    rows = []
    for pix in ("yuv420p", "yuv444p"):
        vdir = ROOT / "local_runs" / f"pixfmt_{pix}"
        shutil.rmtree(vdir, ignore_errors=True)
        vdir.mkdir(parents=True)
        for s in samples:
            frame0 = read_frames(s["dataset"], s["episode_index"], s["start"], 1)[0]
            write_mp4(frame0, vdir / f"{s['sample_id']}.mp4", pix, CRF)

        csv_path = ROOT / "local_runs" / f"pixfmt_{pix}.csv"
        subprocess.run([sys.executable, str(ROOT / "local_eval" / "score_v2.py"),
                        "--videos", str(vdir), "--tiers", "unseen", "--csv", str(csv_path)],
                       check=True, cwd=ROOT / "local_eval", stdout=subprocess.DEVNULL)
        df = pd.read_csv(csv_path)
        kb = sum(p.stat().st_size for p in vdir.glob("*.mp4")) / len(samples) / 1024
        rows.append({
            "pix_fmt": pix, "crf": CRF, "kb_per_file": round(kb, 1),
            "dino_pf": round(df.dino_pf.mean(), 6),
            "video": round(df.video.mean(), 6),
            "action": round(df.action.mean(), 6),
            "total_pf": round(df.total_pf.mean(), 6),
            "dino_gap_vs_unencoded": round(df.dino_pf.mean() - BASELINE["dino_pf"], 6),
        })
        print(f"{pix}: DINO {rows[-1]['dino_pf']:.5f}  Video {rows[-1]['video']:.5f}  "
              f"Action {rows[-1]['action']:.5f}  {kb:.1f} KB  "
              f"(무인코딩 대비 DINO {rows[-1]['dino_gap_vs_unencoded']:+.5f})", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "pixfmt_test.csv", index=False)
    print(f"\n무인코딩 기준선: DINO {BASELINE['dino_pf']} / Video {BASELINE['video']} "
          f"/ Action {BASELINE['action']}")
    d420 = out.loc[out.pix_fmt == "yuv420p", "dino_gap_vs_unencoded"].iloc[0]
    d444 = out.loc[out.pix_fmt == "yuv444p", "dino_gap_vs_unencoded"].iloc[0]
    print(f"판정: yuv420p 손실 {d420:+.5f} vs yuv444p 손실 {d444:+.5f} → "
          f"크로마 서브샘플링이 {'주범 맞음' if abs(d444) < abs(d420) * 0.5 else '주범 아님'}")
    print("→ results/pixfmt_test.csv")


if __name__ == "__main__":
    main()
