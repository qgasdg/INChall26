"""생성 저장부를 yuv444p로 패치 (docs/17 §6 자유개선 +0.0008).

open/ 트리는 .gitignore 대상(open.zip에서 복원)이라 git으로 전달되지 않는다. 이 스크립트가
그 수정을 **재현 가능한 형태로 버전관리**한다 — 새 서버/컨테이너에서 한 번 실행하면 적용된다.

무엇을: open/baseline/challenge_kit/scripts/eval/feature_csv_utils.py의 save_video_tensor()가
torchvision.io.write_video(yuv420p 하드코딩)로 저장하던 것을 PyAV(yuv444p)로 교체.
yuv420p는 색정보를 절반 해상도로 줄여(크로마 서브샘플링) 채점 특징에 손실을 남긴다.
제출용 킷(open/submission_kit)은 건드리지 않는다(SHA 무변경). 멱등 — 이미 적용됐으면 no-op.

사용:
  .venv/bin/python scripts/apply_yuv444p_fix.py            # 적용
  .venv/bin/python scripts/apply_yuv444p_fix.py --check    # 적용 여부만 확인
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "open" / "baseline" / "challenge_kit" / "scripts" / "eval" / "feature_csv_utils.py"

OLD = '''    frames = frames.permute(1, 2, 3, 0).contiguous()
    try:
        from torchvision.io import write_video

        write_video(str(path), frames, fps=fps, video_codec="libx264")
    except Exception as error:
        raise RuntimeError(f"Failed to save generated video to {path}.") from error'''

NEW = '''    frames = frames.permute(1, 2, 3, 0).contiguous()  # (T, H, W, C) uint8 RGB
    try:
        # 색정보 보존을 위해 yuv444p로 인코딩 (torchvision write_video는 yuv420p 하드코딩 →
        # 크로마 서브샘플링 손실이 채점 특징에 잔존, +0.0008 실측 — docs/17 §6).
        import av

        frames_np = frames.numpy()
        n, height, width, _ = frames_np.shape
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=int(fps))
            stream.height, stream.width = height, width
            stream.pix_fmt = "yuv444p"
            stream.options = {"crf": "10", "preset": "medium"}
            for i in range(n):
                frame = av.VideoFrame.from_ndarray(frames_np[i], format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as error:
        raise RuntimeError(f"Failed to save generated video to {path}.") from error'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="적용 여부만 확인(수정 안 함)")
    args = ap.parse_args()

    if not TARGET.exists():
        raise SystemExit(f"대상 없음: {TARGET}  (open.zip 복원 필요)")
    text = TARGET.read_text()

    if 'stream.pix_fmt = "yuv444p"' in text:
        print("이미 적용됨 (yuv444p) — no-op")
        return
    if args.check:
        print("미적용 (yuv420p) — 패치 필요")
        return
    if OLD not in text:
        raise SystemExit("예상 코드 블록을 못 찾음 — 베이스라인 버전이 다를 수 있음(수동 검토 필요)")
    TARGET.write_text(text.replace(OLD, NEW, 1))
    print(f"패치 적용 완료: {TARGET}")
    print("검증: .venv/bin/python scripts/apply_yuv444p_fix.py --check")


if __name__ == "__main__":
    main()
