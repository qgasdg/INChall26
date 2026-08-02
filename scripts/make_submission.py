"""제출용 eval216 영상 생성 — 우리 IRASim 체크포인트로 mp4 216개를 만든다.

기존 `run_generation.py`는 **공식 베이스라인 모델**을 부르는 실행기라 우리 파인튜닝 ckpt를 못 쓴다.
이 스크립트가 그 자리를 메운다: 조건 프레임 + 정답 액션 15스텝 → 16프레임 생성 → mp4 저장.

인코딩은 `make_static_videos.py`와 동일 규격(h264 · crf 10 · yuv420p · 6fps)으로 맞춘다 —
정적 제출과 같은 조건이어야 점수 비교가 성립한다(코덱이 D+V에 ~0.002 영향).

이 다음은 킷 원본이 처리한다(★킷 전용 .venv 필요):
  .venv/bin/python open/submission_kit/make_submission_csv.py \
      --prediction-root <이 스크립트의 --out> \
      --challenge-root open/data/eval \
      --output-csv local_runs/submission_<태그>.csv \
      --action-stats-path open/data/train/so100_action_statistics.json \
      --action-extractor-ckpt open/submission_kit/checkpoints/action_extractor.ckpt

사용(conda torch260):
  python scripts/make_submission.py --ckpt ckpts/e5_keep/ckpt_0010000.pt --out local_runs/pred_e5s10000
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GEN_HW = (320, 512)
SEQ, FPS = 16, 6


def write_mp4(frames: np.ndarray, path: Path) -> None:
    """(16,H,W,3) uint8 → h264 crf10 yuv420p 6fps mp4. make_static_videos.py와 동일 규격."""
    import av

    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=FPS)
        st.height, st.width = frames.shape[1], frames.shape[2]
        st.pix_fmt = "yuv420p"
        st.options = {"crf": "10", "preset": "medium"}
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="    %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="mp4 저장 폴더 (킷의 --prediction-root)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eta", type=float, default=1.0, help="E1 실측 확정값")
    ap.add_argument("--method", default="PNDM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="앞 N개만(디버그). 0=전량 216")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from src.data import transforms as T
    from src.models.action_adapter import adapt_action_seq
    from src.models.irasim_runtime import build, encode_video
    from src.models.irasim_runtime import generate as rt_generate
    from src.utils.config import load_config

    cfg = load_config()
    cfg.setdefault("train", {})["precision"] = "fp16"
    cfg.setdefault("data", {})["pre_encode"] = False
    mean, std = T.load_action_stats()
    align = cfg.get("data", {}).get("align_mode", "shifted")

    img_dir, act_dir = ROOT / "open/data/eval/images", ROOT / "open/data/eval/actions"
    pngs = sorted(img_dir.glob("sample_*.png"))
    if args.limit:
        pngs = pngs[:args.limit]
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    print("샘플 %d개 · ckpt %s · steps=%d eta=%.1f seed=%d → %s"
          % (len(pngs), args.ckpt, args.steps, args.eta, args.seed, outdir))

    rt = build(cfg, slim_ckpt=args.ckpt, logger=logging.getLogger("rt"))
    t0 = time.time()
    for i, p in enumerate(pngs, 1):
        dst = outdir / f"{p.stem}.mp4"
        if dst.exists():
            continue
        f0 = np.asarray(Image.open(p).convert("RGB"))[None]
        gen0 = T.final_to_gen_target(f0, T.RES_A)                       # (1,320,512,3)
        x = torch.from_numpy(gen0).float().div(127.5).sub(1.0).permute(0, 3, 1, 2)[None]
        a_deg = np.load(act_dir / f"{p.stem}.npy")                      # (16,6) raw deg
        a15 = np.ascontiguousarray(adapt_action_seq(T.normalize_action(a_deg, mean, std), SEQ, align))
        pred = rt_generate(rt, encode_video(rt, x), torch.from_numpy(a15).float()[None],
                           steps=args.steps, eta=args.eta, method=args.method,
                           seed=args.seed, gen_hw=GEN_HW)
        write_mp4(np.asarray(pred[0]), dst)
        torch.cuda.empty_cache()
        if i % 10 == 0 or i == len(pngs):
            el = time.time() - t0
            print("  %3d/%d  %.1fs/샘플  남은 %.0f분" % (i, len(pngs), el / i, (len(pngs) - i) * el / i / 60),
                  flush=True)

    n = len(list(outdir.glob("sample_*.mp4")))
    print("\n완료: mp4 %d개 → %s" % (n, outdir))
    if n != 216 and not args.limit:
        print("★216개가 아니다 — 제출 전 반드시 확인할 것")
    print("\n다음(킷 .venv):")
    print("  .venv/bin/python open/submission_kit/make_submission_csv.py \\\n"
          "      --prediction-root %s --challenge-root open/data/eval \\\n"
          "      --output-csv local_runs/submission.csv \\\n"
          "      --action-stats-path open/data/train/so100_action_statistics.json \\\n"
          "      --action-extractor-ckpt open/submission_kit/checkpoints/action_extractor.ckpt" % outdir)


if __name__ == "__main__":
    main()
