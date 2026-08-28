#!/bin/bash
# 추론 — eval 216문제 영상 생성 → 배경 고정 → 제출 CSV.
#
# 사용: infer.sh [체크포인트] [출력폴더]
#   체크포인트 생략 시 학습 4단계의 step=6600(=누적 10,800)을 쓴다 — 제출 점수를 낸 모델이다.
#   LIMIT=2 로 두면 2문제만 돌려 배선을 빨리 확인할 수 있다(기본 216).
#
# 소요 — 2026-08-28 재학습 검증 때 **RTX 5090 1장에서 실측**(제한 1시간):
#   생성 216개  57분 40초  (샘플당 15.80초 · DDIM 50스텝 · 액션 CFG 2.0 이라 순전파가 2배)
#   배경 고정      41초    (GPU 불필요)
#   CSV(-bg)    1분 33초
#   ─────────── 제출 경로 합계 59분 54초  ← 한계선에 붙어 있다
#   (선택인 -raw CSV 까지 만들면 61분 30초)

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CKPT="${1:-$(ckpt_at stage4 6600)}"
OUT="${2:-$FT_ROOT/out/final}"
DATA="$FT_ROOT/data"

echo "[$(date +%H:%M:%S)] === 생성 시작 · $CKPT ==="
"$PY" "$HERE/src/generate.py" \
  --config "$HERE/configs/infer.yaml" \
  --challenge-root "$DATA/eval" \
  --action-stats-path "$DATA/train/so100_action_statistics.json" \
  --delta-stats "$DATA/train/so100_delta_statistics.json" \
  --action-dims 12 --no-ema --start 0 --limit "${LIMIT:-216}" \
  --ckpt "$CKPT" --out "$OUT-eval216"

echo "[$(date +%H:%M:%S)] === 배경 고정 ==="
"$PY" "$HERE/src/bgfreeze.py" --src "$OUT-eval216" --dst "$OUT-bg"

# 두 벌 다 만든다 — 배경 고정(후처리)이 규정상 불가하다는 판단이 나오면 -raw 를 쓴다
for v in eval216 bg; do
  tag=$([ "$v" = bg ] && echo "-bg" || echo "-raw")
  echo "[$(date +%H:%M:%S)] === CSV$tag ==="
  ( cd "$SUBKIT" && "$PY" make_submission_csv.py \
      --prediction-root "$OUT-$v" \
      --challenge-root "$DATA/eval" \
      --action-stats-path "$DATA/train/so100_action_statistics.json" \
      --output-csv "$FT_ROOT/submission$tag.csv" )
done

ls -la "$FT_ROOT"/submission-raw.csv "$FT_ROOT"/submission-bg.csv
echo ">>> 제출본: $FT_ROOT/submission-bg.csv"
