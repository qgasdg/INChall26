"""동작 점수(A_kit)의 천장 측정 — "완벽한 영상"은 몇 점인가.

왜 필요한가: 대회 점수의 40%인 A_kit은 **영상을 보고 로봇이 어떻게 움직였는지 거꾸로 추정**해
    정답 행동과 비교한다. 그 추정기가 완벽하지 않아서 **정답 영상을 넣어도 0점이 아니다**.
    그 값이 이 지표의 천장이고, 천장을 모르면 지금 점수가 갈 길이 먼 건지 거의 한계인 건지
    구분할 수 없다. (실측: 정적 제출 0.4285 / E3 FT step14000 0.4730 — 둘 다 정답이 아닌 영상)

읽는 법 (낮을수록 좋음):
  - 정답 영상이 **정적보다 뚜렷이 낮다**  → 잘 만들면 이길 여지가 있다. **학습 계속이 맞다.**
  - 정답 영상이 **정적과 비슷하다**       → 아무리 잘 만들어도 정적을 못 이긴다. 지표가 움직임을
                                          제대로 못 읽는다는 뜻이라 **접근을 다시 짜야 한다.**

비교군 3종을 같은 클립·같은 채점기로 나란히 잰다:
  gt       = 정답 영상 그대로       → 천장
  static   = 첫 프레임을 16번 복사  → 우리 최고 제출과 같은 방식, 기준선
  reversed = 정답을 거꾸로 재생     → "움직임은 있는데 틀린" 경우. 이게 정답보다 점수가 좋으면
             지표가 움직임의 방향조차 못 읽는다는 뜻이라, 생성 품질을 올릴 이유가 사라진다.

★킷 전용 venv로 돌려야 한다(공용 conda엔 pytorch_lightning 없음):
  .venv/bin/python scripts/diag_akit_ceiling.py --n 24
산출: 표준출력 + results/diag_akit_ceiling.json
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


def pick_clips(n: int, tier: str = "unseen") -> list[dict]:
    """홀드아웃에서 데이터셋 겹치지 않게 n개 — 한 데이터셋 쏠림 방지.

    tier — 어느 계층에서 뽑을지. **eval과 성격이 비슷한 쪽으로 재야 천장이 의미가 있다.**
      unseen 클립들은 정적 점수가 0.91인데 실제 eval216은 0.43이라 성격이 많이 다르다
      → indomain(eval과 같은 계열)으로도 재서 두 값이 같은 이야기를 하는지 확인할 것.
    """
    pool = [s for s in json.loads((ROOT / "local_eval" / "holdout_v2.json").read_text(encoding="utf-8"))["samples"]
            if tier == "all" or str(s.get("tier", "")).startswith(tier)]
    by_ds: dict[str, list] = {}
    for s in pool:
        by_ds.setdefault(s["dataset"], []).append(s)
    picked, rnd = [], 0
    while len(picked) < n and any(len(v) > rnd for v in by_ds.values()):
        for ds in sorted(by_ds):
            if len(by_ds[ds]) > rnd and len(picked) < n:
                picked.append(by_ds[ds][rnd])
        rnd += 1
    return picked


def control_eval216(scorer, n: int) -> float:
    """★대조 검증 — 같은 코드 경로로 eval216 정적을 재서 기존 실측 0.4285가 재현되는지 본다.

    재현되면 이 스크립트의 프레임 투입·정답 대조가 옳다는 뜻이고, 크게 어긋나면 천장 측정값을
    믿을 수 없다(채점기가 영상을 제대로 못 받고 있는 것). 결론을 내기 전에 반드시 통과해야 한다.
    """
    from PIL import Image

    from local_eval.kit_bridge import action_mae

    img_dir, act_dir = ROOT / "open/data/eval/images", ROOT / "open/data/eval/actions"
    pngs = sorted(img_dir.glob("sample_*.png"))[:n]
    vals = []
    for p in pngs:
        f0 = np.asarray(Image.open(p).convert("RGB"))
        vid = scorer.to_eval_video(np.repeat(f0[None], SEQ, axis=0))[None]
        acts = np.load(act_dir / f"{p.stem}.npy")                        # (16,6) raw deg
        vals.append(action_mae(scorer.action_pred(vid)[0], scorer.normalize_actions(acts)))
    m = float(np.mean(vals))
    print("\n=== 대조 검증: eval216 정적 %d개 ===" % len(vals))
    print("  이 코드 경로 %.4f  vs  기존 실측 0.4285  (차이 %+.4f)" % (m, m - 0.4285))
    print("  %s" % ("→ 재현됨. 위 천장 측정 신뢰 가능" if abs(m - 0.4285) < 0.03 else
                    "→ ★재현 실패. 프레임 투입이나 정답 대조가 잘못됐다 — 천장 결론 보류"))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24, help="측정할 클립 수(많을수록 안정적, 24면 충분)")
    ap.add_argument("--tier", default="unseen", choices=("unseen", "indomain", "all"),
                    help="어느 홀드아웃 계층에서 뽑을지 — indomain이 eval과 성격이 가깝다")
    ap.add_argument("--control", type=int, default=12,
                    help="대조 검증에 쓸 eval216 샘플 수 (0이면 생략)")
    args = ap.parse_args()

    import torch

    from local_eval import episode_io
    from local_eval.kit_bridge import KitScorer, action_mae, get_device

    scorer = KitScorer(get_device())
    clips = pick_clips(args.n, args.tier)
    if not clips:
        raise SystemExit("해당 tier 샘플 없음: %s" % args.tier)
    print("tier=%s · 클립 %d개 (데이터셋 %d종)" % (args.tier, len(clips), len({c["dataset"] for c in clips})))

    variants = ("gt", "static", "reversed")
    rows = []
    for c in clips:
        ds, ep, st = c["dataset"], int(c["episode_index"]), int(c.get("start", 0))
        frames = episode_io.read_frames(ds, ep, st, SEQ)                 # (16,H,W,3) uint8
        acts = episode_io.read_actions(ds, ep, st, SEQ)                  # (16,6) deg
        target = scorer.normalize_actions(acts)

        r = {"sample": c["sample_id"], "dataset": ds}
        for v in variants:
            f = {"gt": frames,
                 "static": np.repeat(frames[:1], SEQ, axis=0),
                 "reversed": frames[::-1].copy()}[v]
            vid = scorer.to_eval_video(f)[None]                          # (1,16,320,512,3) uint8
            r[v] = action_mae(scorer.action_pred(vid)[0], target)
        rows.append(r)
        print("  %-52s 정답 %.4f · 정적 %.4f · 역재생 %.4f"
              % (r["sample"][-52:], r["gt"], r["static"], r["reversed"]), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    agg = {v: float(np.mean([r[v] for r in rows])) for v in variants}
    gap = agg["static"] - agg["gt"]
    print("\n=== 결과 (낮을수록 좋음) ===")
    for v, label in (("gt", "정답 영상 (천장)"), ("static", "정적 (첫 프레임 반복)"),
                     ("reversed", "역재생 (움직이나 틀림)")):
        print("  %-22s %.4f" % (label, agg[v]))
    print("  정적 − 정답 = %+.4f  (이만큼이 '잘 만들어서 벌 수 있는 최대치')" % gap)
    print("\n  참고 실측: eval216 정적 0.4285 · E3 FT step14000 0.4730")
    print("  %s" % ("→ 여유 있음. 생성 품질을 올리면 A_kit이 내려간다. **학습 계속이 타당**"
                    if gap > 0.10 else
                    "→ ★정답 영상조차 정적과 비슷하다. 이 지표는 움직임을 제대로 못 읽는다 — "
                    "생성을 아무리 잘해도 A_kit으로는 정적을 못 이긴다. 접근 재설계 필요"))
    if agg["reversed"] < agg["gt"]:
        print("  ★역재생이 정답보다 좋다 — 지표가 움직임의 방향조차 구분 못 한다는 뜻")

    res = {"n": len(rows), "tier": args.tier, "agg": agg, "static_minus_gt": gap, "rows": rows}
    if args.control:
        res["control_eval216_static"] = control_eval216(scorer, args.control)
        res["control_ok"] = abs(res["control_eval216_static"] - 0.4285) < 0.03
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / ("diag_akit_ceiling_%s.json" % args.tier)).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("\n→ results/diag_akit_ceiling_%s.json" % args.tier)


if __name__ == "__main__":
    main()
