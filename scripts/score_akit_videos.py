"""생성된 mp4 폴더 → A_kit(동작 정확도) 실측. 제출 전 판단용.

`rank_ckpts_akit.py`는 ckpt에서 새로 생성해 채점하지만, 제출용으로 이미 216개 mp4를 만들었다면
다시 생성할 이유가 없다. 이 스크립트는 **그 mp4를 그대로 읽어** 킷 채점기로 A_kit을 낸다.
제출 CSV와 완전히 같은 파일을 채점하므로 서버가 볼 값과 눈금이 같다(인코딩 손실까지 포함).

A_kit = 평균 |추정행동 − 정답행동| (z-score 공간, 낮을수록 좋음)
  추정행동 = 킷 action_extractor(생성 영상) · 정답행동 = open/data/eval/actions/sample_XXXXXX.npy

기준선: 정적 0.4285 · E3 step14000 0.4730 · 제로샷 0.5933 · 리더보드 1위 추정 ≤0.139
총점 환산: `총점 = 0.3×DINO + 0.3×Video + 0.4×A_kit` (킷 원본 산식, 상수항 없음)

★킷 전용 .venv로 실행:
  .venv/bin/python scripts/score_akit_videos.py --videos local_runs/pred_e5s10000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEQ = 16


def read_mp4(path: Path) -> np.ndarray:
    """mp4 → (T,H,W,3) uint8. 프레임 수가 16이 아니면 예외(제출 규격 위반 조기 발견)."""
    import av

    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    if len(frames) != SEQ:
        raise ValueError(f"{path.name}: 프레임 {len(frames)}개 (16이어야 함)")
    return np.stack(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="sample_XXXXXX.mp4 폴더")
    ap.add_argument("--tag", default="", help="results 파일 접미사")
    args = ap.parse_args()

    from local_eval.kit_bridge import KitScorer, action_mae, get_device

    act_dir = ROOT / "open/data/eval/actions"
    mp4s = sorted(Path(args.videos).glob("sample_*.mp4"))
    if not mp4s:
        raise SystemExit("mp4 없음: %s" % args.videos)
    print("영상 %d개 · %s" % (len(mp4s), args.videos))
    if len(mp4s) != 216:
        print("★216개가 아니다 — 제출 불가 상태일 수 있음")

    scorer = KitScorer(get_device())
    rows = []
    for i, p in enumerate(mp4s, 1):
        vid = scorer.to_eval_video(read_mp4(p))[None]
        tgt = scorer.normalize_actions(np.load(act_dir / f"{p.stem}.npy"))
        rows.append({"sample": p.stem, "akit": action_mae(scorer.action_pred(vid)[0], tgt)})
        if i % 50 == 0 or i == len(mp4s):
            print("  %3d/%d  진행 평균 %.4f" % (i, len(mp4s), np.mean([r["akit"] for r in rows])), flush=True)

    m = float(np.mean([r["akit"] for r in rows]))
    print("\n=== A_kit = %.4f (n=%d) ===" % (m, len(rows)))
    print("  정적 0.4285 · E3 step14000 0.4730 · 제로샷 0.5933")
    print("  %s" % ("★정적 돌파 — 제출 가치 있음(D+V가 정적보다 크게 나쁘지만 않다면)"
                    if m < 0.4285 else
                    "정적 미달 — 동작 항목만으로는 정적을 못 이긴다"))
    print("  참고: A_kit이 정적보다 %+.4f → 총점에 %+.4f 기여(가중치 0.4)"
          % (m - 0.4285, 0.4 * (m - 0.4285)))

    out = ROOT / "results" / ("akit_videos%s.json" % (("_" + args.tag) if args.tag else ""))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"videos": args.videos, "n": len(rows), "akit": m, "rows": rows},
                              ensure_ascii=False, indent=1))
    print("\n→ %s" % out.relative_to(ROOT))


if __name__ == "__main__":
    main()
