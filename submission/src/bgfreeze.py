"""배경 고정 합성 — 생성 영상의 저모션 영역을 조건 프레임으로 되돌린다.

**왜.** 우리 실패는 두 갈래다. Action 은 액션 CFG 로 벌 수 있는 게 확인됐지만(0.2884 → 0.2695),
그 대가로 Video 성분이 깎이는 것으로 보인다. 배경 드리프트·질감 변형은 **팔의 행선지와 무관한
순손해**이므로, 움직이지 않아야 할 픽셀을 조건 프레임(프레임 0)으로 되돌리면 D+V 를 지키면서
Action 은 건드리지 않는다.

**왜 이번엔 다른가.** `step0_rff`(첫 프레임만 원본으로 교체)가 +0.093 으로 크게 실패했다.
원인은 **시간적 불연속** — 교체된 프레임 0 과 생성된 프레임 1 사이가 튀었다. 여기서는
① 16프레임 **전부**에 같은 규칙을 적용하고 ② 마스크를 **부드럽게**(페더링) 섞으며
③ 프레임별로 마스크를 따로 만들지 않고 **시간축 최대값**으로 하나를 만들어 프레임 간
마스크 깜빡임을 없앤다.

**마스크.** 생성 영상에서 프레임 0 대비 변화량을 재고(모든 프레임의 최대), 그 값이 큰 곳이
움직이는 곳(팔·물체)이다. 임계값은 분위수로 잡아 장면마다 자동 적응한다. 팔을 지우면
그 순간 Action 이 무너지므로 **넉넉히 살리는 쪽**(팽창 + 낮은 임계값)으로 편향시킨다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_video(path: Path) -> np.ndarray:
    import imageio.v3 as iio
    return np.asarray(iio.imread(path, plugin="pyav"))          # [T,H,W,3] uint8


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    import imageio.v3 as iio
    iio.imwrite(path, frames, fps=fps, codec="libx264",
                plugin="pyav", in_pixel_format="rgb24", out_pixel_format="yuv420p")


def box_blur(x: np.ndarray, r: int) -> np.ndarray:
    """가장자리를 부드럽게 만드는 용도의 정사각 평균 필터 (의존성 없이 누적합으로)."""
    if r <= 0:
        return x
    p = np.pad(x, r, mode="edge")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    H, W = x.shape
    s = c[k:k + H, k:k + W] - c[:H, k:k + W] - c[k:k + H, :W] + c[:H, :W]
    return s / (k * k)


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    """최대값 팽창 — 마스크를 r 픽셀 넓힌다(팔 주변을 넉넉히 살리려고)."""
    if r <= 0:
        return m
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out = np.maximum(out, np.roll(np.roll(m, dy, 0), dx, 1))
    return out


def compose(frames: np.ndarray, q: float, dil: int, feather: int,
            floor: float) -> tuple[np.ndarray, float]:
    """frames[0] 을 배경 원본으로 삼아 저모션 영역을 되돌린다. (합성본, 동적비율) 반환."""
    f = frames.astype(np.float32)
    base = f[0]
    d = np.abs(f - base[None]).mean(-1).max(0)                  # [H,W] 시간축 최대 변화량
    thr = np.quantile(d, q)
    m = (d > thr).astype(np.float32)
    m = dilate(m, dil)
    m = box_blur(m, feather)                                    # 경계 페더링
    m = np.clip(m, 0.0, 1.0)
    m = floor + (1.0 - floor) * m                               # 배경도 floor 만큼은 생성본을 남긴다
    out = f * m[None, :, :, None] + base[None] * (1.0 - m[None, :, :, None])
    out[0] = f[0]                                               # 프레임 0 은 원본 그대로
    return np.clip(out, 0, 255).astype(np.uint8), float(m.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="생성된 mp4 폴더")
    ap.add_argument("--dst", required=True, help="합성 결과를 쓸 폴더")
    ap.add_argument("--quantile", type=float, default=0.90,
                    help="이 분위수를 넘는 변화량만 '움직이는 곳'. 낮출수록 더 많이 살린다")
    ap.add_argument("--dilate", type=int, default=6, help="마스크 팽창 반경(픽셀)")
    ap.add_argument("--feather", type=int, default=8, help="경계 페더링 반경(픽셀)")
    ap.add_argument("--floor", type=float, default=0.15,
                    help="배경에도 남길 생성본 비율. 0 이면 배경이 완전 정지")
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    vids = sorted(src.glob("sample_*.mp4"))
    print(f">>> {len(vids)}개 · q={args.quantile} dil={args.dilate} feather={args.feather} floor={args.floor}")
    ratios = []
    for i, v in enumerate(vids):
        fr = read_video(v)
        out, r = compose(fr, args.quantile, args.dilate, args.feather, args.floor)
        write_video(dst / v.name, out, args.fps)
        ratios.append(r)
        if (i + 1) % 40 == 0 or i + 1 == len(vids):
            print(f"  [{i+1}/{len(vids)}] 평균 동적비율 {np.mean(ratios):.3f}", flush=True)
    print(f">>> 완료: {dst}")


if __name__ == "__main__":
    main()
