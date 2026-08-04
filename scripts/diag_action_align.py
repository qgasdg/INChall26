"""1교시 — 액션 정렬 자기검증: 킷 추출기는 action[t]를 어느 프레임에서 읽는가.

왜 필요한가: 정답 액션으로 만든 영상이 셔플 액션으로 만든 영상보다 점수가 나빴다
    (태양님 7/25 probe: true 0.4935 / shuffled 0.3838). "정답이 오히려 해롭다"는 건
    조건화가 약한 것만으로는 설명되지 않는다 — **정확한 동작을 틀린 시점에 적용**하고
    있을 때 나오는 증상이다. 이 스크립트는 **모델을 한 번도 돌리지 않고** 그 가설을 가른다.

방법: 정답 영상을 추출기에 넣어 얻은 pred(16,6)를 정답 액션 target(16,6)과 시간축으로
    -3..+3 밀어가며 **상관**을 잰다. 상관은 각 계열의 평균을 빼고 재므로 데이터셋마다
    다른 관절 0점(docs/17 wrist_roll 87도)에 영향받지 않는다 — 7/28 천장 측정을 무너뜨린
    바로 그 문제를 통과한다. MAE로는 못 하는 측정이다.

읽는 법 (pred[t] 와 target[t+s] 의 상관이 최대가 되는 s):
  s = 0  → 추출기는 action[t] ↔ frame[t]. 생성 시 미래 frame 1..15 는 act[1..15] 를 받아야
           하므로 어댑터는 **same_index** 가 맞다. (현재 기본값은 shifted = act[0..14])
  s = +1 → pred[t] ↔ target[t+1]. 프레임보다 액션 인덱스가 하나 앞선다 → **shifted** 가 맞다.
  s ≠ 0 인데 위 둘도 아님 → train parquet 의 액션·프레임 인덱싱이 우리 가정과 다르다.
  어느 s 에서도 |상관| 이 0 근처 → 추출기가 이 도메인에서 동작을 못 읽는다. 홀드아웃으로
           Action 을 개선한다는 계획 전체가 성립하지 않는다 → 설계를 다시 짜야 한다.

대조군 2종 (같은 클립·같은 채점기):
  static   = 첫 프레임 16번 복사 → 움직임이 없으니 상관이 0 근처여야 한다.
             여기서 상관이 나오면 지표가 헛것을 읽고 있다는 뜻 (측정 무효).
  reversed = 정답 역재생 → 상관 **부호가 뒤집혀야** 한다. 뒤집히면 추출기가 움직임의
             방향까지 읽는다는 강한 증거다 (7/28 에 역재생만 갈렸던 것과 같은 이야기).

★leakage 안전: eval 정답 액션(open/data/eval/actions)을 일절 읽지 않는다. 홀드아웃의
  GT 영상·GT 액션만 쓴다. 2026-08-04 킷 사용 공지 기준으로 안전한 경로.

★킷 전용 venv 로 돌릴 것 (공용 conda 엔 pytorch_lightning 이 없다):
  .venv/bin/python scripts/diag_action_align.py --n 40 --tier all
산출: 표준출력 + results/diag_action_align_<tier>.json
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
DIMS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def pick_clips(n: int, tier: str) -> list[dict]:
    """홀드아웃에서 데이터셋이 겹치지 않게 n개 — 한 데이터셋 쏠림 방지 (diag_akit_ceiling 과 동일 규칙)."""
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


def corr_at_shift(pred: np.ndarray, target: np.ndarray, s: int, min_std: float) -> np.ndarray:
    """pred[t] 와 target[t+s] 의 **관절별** 피어슨 상관 (6,). 움직임이 없는 관절은 nan.

    상관은 평균을 빼고 표준편차로 나누므로 **관절 0점 차이와 단위 차이에 불변**이다.
    16프레임 중 |s| 만큼이 잘려 나가므로 s=±3 이면 13개 시점으로 잰다.
    """
    a, b = (pred[: len(pred) - s], target[s:]) if s > 0 else \
           (pred[-s:], target[: len(target) + s]) if s < 0 else (pred, target)
    a, b = a - a.mean(0), b - b.mean(0)
    sa, sb = a.std(0), b.std(0)
    c = (a * b).mean(0) / (sa * sb + 1e-12)
    c[(sa < min_std) | (sb < min_std)] = np.nan          # 정지한 관절은 상관이 정의되지 않는다
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="측정할 클립 수 (7/28 실측상 24개 이상 필요)")
    ap.add_argument("--tier", default="all", choices=("unseen", "indomain", "all"))
    ap.add_argument("--max-shift", type=int, default=3, help="±몇 프레임까지 밀어볼지")
    ap.add_argument("--min-std", type=float, default=0.01,
                    help="이 값보다 덜 움직인 관절은 상관 계산에서 제외 (z-score 단위)")
    args = ap.parse_args()

    import torch

    from local_eval import episode_io
    from local_eval.kit_bridge import KitScorer, get_device

    shifts = list(range(-args.max_shift, args.max_shift + 1))
    scorer = KitScorer(get_device())
    clips = pick_clips(args.n, args.tier)
    if not clips:
        raise SystemExit("해당 tier 샘플 없음: %s" % args.tier)
    print("tier=%s · 클립 %d개 (데이터셋 %d종) · shift %s"
          % (args.tier, len(clips), len({c["dataset"] for c in clips}), shifts))

    variants = ("gt", "static", "reversed")
    # acc[variant][kind][shift] = 클립×관절 상관 리스트
    acc = {v: {k: {s: [] for s in shifts} for k in ("raw", "delta")} for v in variants}
    rows = []

    for i, c in enumerate(clips, 1):
        ds, ep, st = c["dataset"], int(c["episode_index"]), int(c.get("start", 0))
        frames = episode_io.read_frames(ds, ep, st, SEQ)                  # (16,H,W,3) uint8
        acts = episode_io.read_actions(ds, ep, st, SEQ)                   # (16,6) deg
        target = scorer.normalize_actions(acts).numpy()                   # (16,6) z-score

        row = {"sample": c["sample_id"], "dataset": ds}
        for v in variants:
            f = {"gt": frames,
                 "static": np.repeat(frames[:1], SEQ, axis=0),
                 "reversed": frames[::-1].copy()}[v]
            pred = scorer.action_pred(scorer.to_eval_video(f)[None])[0].numpy()   # (16,6)
            for kind, (p, t) in (("raw", (pred, target)),
                                 ("delta", (np.diff(pred, axis=0), np.diff(target, axis=0)))):
                for s in shifts:
                    cc = corr_at_shift(p, t, s, args.min_std)
                    acc[v][kind][s].append(cc)
                    if s == 0:
                        row["%s_%s_r0" % (v, kind)] = float(np.nanmean(cc))
        rows.append(row)
        print("  [%3d/%3d] %-40s gt r0 raw %+.3f · delta %+.3f"
              % (i, len(clips), row["sample"][-40:], row["gt_raw_r0"], row["gt_delta_r0"]), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def agg(v, kind, s):
        return float(np.nanmean(np.stack(acc[v][kind][s])))

    print("\n=== shift 별 평균 상관 (pred[t] vs target[t+s], 1에 가까울수록 정렬됨) ===")
    print("  %-6s | %-24s | %-24s" % ("shift", "raw (절대값)", "delta (차분)"))
    print("  %-6s | %-7s %-7s %-7s | %-7s %-7s %-7s" % ("", "정답", "정적", "역재생", "정답", "정적", "역재생"))
    for s in shifts:
        print("  %-6d | %+7.3f %+7.3f %+7.3f | %+7.3f %+7.3f %+7.3f"
              % (s, agg("gt", "raw", s), agg("static", "raw", s), agg("reversed", "raw", s),
                 agg("gt", "delta", s), agg("static", "delta", s), agg("reversed", "delta", s)))

    best = {k: max(shifts, key=lambda s: agg("gt", k, s)) for k in ("raw", "delta")}
    print("\n=== 판정 ===")
    for kind in ("raw", "delta"):
        b, r0, rb = best[kind], agg("gt", kind, 0), agg("gt", kind, best[kind])
        # 같은 클립에서 잰 값끼리 비교하므로 짝지은 차이의 표준오차로 본다
        per = [np.nanmean(x) - np.nanmean(y) for x, y in zip(acc["gt"][kind][b], acc["gt"][kind][0])]
        se = float(np.nanstd(per, ddof=1) / np.sqrt(len(per)))
        print("  [%s] 최대 상관 shift = %+d (r=%.3f) · shift 0 은 r=%.3f · 차이 %+.3f ± %.3f"
              % (kind, b, rb, r0, rb - r0, se))
        if b == 0 or (rb - r0) < 2 * se:
            print("        → 정렬은 shift 0. 추출기는 action[t] ↔ frame[t]. "
                  "어댑터는 **same_index** 가 맞다 (현재 기본값 shifted 는 한 칸 어긋남)")
        else:
            print("        → ★shift %+d 가 유의하게 낫다. 인덱싱이 어긋나 있다 — "
                  "생성 전에 이것부터 고쳐야 한다" % b)

    r_gt, r_st, r_rv = (agg("gt", "delta", 0), agg("static", "delta", 0), agg("reversed", "delta", 0))
    print("\n  대조군 (delta, shift 0):  정답 %+.3f · 정적 %+.3f · 역재생 %+.3f" % (r_gt, r_st, r_rv))
    if abs(r_st) > 0.15:
        print("        → ★정적 영상에서도 상관이 나온다. 지표가 움직임이 아닌 것을 읽고 있다 — 측정 무효")
    if r_gt < 0.15:
        print("        → ★정답 영상조차 상관이 없다. 추출기가 이 도메인에서 동작을 못 읽는다 — "
              "홀드아웃으로 Action 을 개선하는 계획은 성립하지 않는다")
    elif r_rv < r_gt - 0.15:
        print("        → 역재생에서 상관이 떨어진다. 추출기가 움직임의 **방향**까지 읽는다 (측정 유효)")

    print("\n=== shift 0 관절별 상관 (정답, delta) ===")
    per_joint = np.nanmean(np.stack(acc["gt"]["delta"][0]), axis=0)
    valid = np.mean(~np.isnan(np.stack(acc["gt"]["delta"][0])), axis=0)
    for j, name in enumerate(DIMS):
        print("  %-16s r=%+.3f  (유효 클립 %.0f%%)" % (name, per_joint[j], 100 * valid[j]))

    res = {"n": len(rows), "tier": args.tier, "shifts": shifts, "min_std": args.min_std,
           "best_shift": best,
           "agg": {v: {k: {str(s): agg(v, k, s) for s in shifts} for k in ("raw", "delta")} for v in variants},
           "per_joint_delta_shift0": {DIMS[j]: float(per_joint[j]) for j in range(6)},
           "rows": rows}
    (ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results" / ("diag_action_align_%s.json" % args.tier)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("\n→ %s" % out.relative_to(ROOT))


if __name__ == "__main__":
    main()
