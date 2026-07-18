"""eval 조건 프레임 216장 → 정적 mp4(동일 프레임 x16) 생성.

용도: 계정 제출 요건 + 로컬 체인 서버 정합 검증 (docs/BOARD.md Y 항목).
출력: open/submission_kit/input_videos/sample_XXXXXX.mp4 (h264 crf10 yuv420p, 6fps)
"""
from pathlib import Path

import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "open/data/eval/images"
OUT = ROOT / "open/submission_kit/input_videos"
N_FRAMES = 16
FPS = 6


def write_static_mp4(png_path: Path, out_path: Path) -> None:
    frame_rgb = np.asarray(Image.open(png_path).convert("RGB"))
    with av.open(str(out_path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.height, stream.width = frame_rgb.shape[0], frame_rgb.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "10", "preset": "medium"}
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for _ in range(N_FRAMES):
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pngs = sorted(IMAGES.glob("sample_*.png"))
    assert len(pngs) == 216, f"eval 이미지 216장이 아님: {len(pngs)}"
    for png in pngs:
        write_static_mp4(png, OUT / f"{png.stem}.mp4")
    # 검증: 첫 파일 왕복 — 프레임 수와 픽셀 오차
    with av.open(str(OUT / pngs[0].stem) + ".mp4") as container:
        decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    src = np.asarray(Image.open(pngs[0]).convert("RGB")).astype(np.int16)
    err = np.abs(decoded[0].astype(np.int16) - src).mean()
    print(f"wrote {len(pngs)} mp4s | frames={len(decoded)} | mean |err|={err:.3f}")
    assert len(decoded) == N_FRAMES


if __name__ == "__main__":
    main()
