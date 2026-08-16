#!/bin/bash
# 추론 — eval 216문제 영상 생성 → 배경 고정 → 제출 CSV.
#
# 사용: infer.sh [체크포인트] [출력폴더]
#   체크포인트 생략 시 학습 4단계의 step=4800(=누적 9,000)을 쓴다.
#
# 소요(RTX PRO 6000 96GB 1장 기준, 제한 1시간):
#   생성 216개  약 49분   (DDIM 50스텝 · 액션 CFG 2.0 이라 순전파가 2배)
#   배경 고정   약 1.5분  (GPU 불필요)
#   CSV 2벌     약 1분
#   ─────────── 합계 약 52분

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CKPT="${1:-$(ckpt_at stage4 4800)}"
OUT="${2:-$FT_ROOT/out/final}"
DATA="$FT_ROOT/data"

echo "[$(date +%H:%M:%S)] === 생성 시작 · $CKPT ==="
"$PY" "$HERE/src/generate.py" \
  --config "$HERE/configs/infer.yaml" \
  --challenge-root "$DATA/eval" \
  --action-stats-path "$DATA/train/so100_action_statistics.json" \
  --delta-stats "$DATA/train/so100_delta_statistics.json" \
  --action-dims 12 --no-ema --start 0 --limit 216 \
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
