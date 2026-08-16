#!/bin/bash
# 학습 — 4단계 이어달리기, 합계 9,000스텝. GPU 1장.
#
# 왜 한 번에 안 돌리고 4단계인가: 실제로 그렇게 학습했기 때문이다. 단계를 넘을 때
# **UNet 가중치만** 물려받고 옵티마이저 상태는 새로 시작하며 학습률 워밍업도 다시 돈다.
# 그래서 9,000스텝을 한 번에 도는 것과 결과가 같지 않다 — 재현하려면 이 순서 그대로여야 한다.
#
#   1단계  300스텝  인코더 동결 · 액션 조건 없음 (백본 적응)
#   2단계 2,100스텝 인코더 해제 + 8비트 Adam + 액션 cross-attention 투입
#   3단계 1,800스텝 2단계와 설정 동일
#   4단계 4,800스텝 2단계와 설정 동일
#
# 소요: RTX PRO 6000 96GB 1장 기준 스텝당 약 10.3초 → 합계 약 26시간 (제한 4일)

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

run () {   # run <단계> [이어받을단계 이어받을스텝]
  local stage=$1
  if [ $# -gt 1 ]; then
    RESUME_CKPT="$(ckpt_at "$2" "$3")"
    export RESUME_CKPT
    echo ">>> $stage ← $RESUME_CKPT"
  fi
  echo "[$(date +%H:%M:%S)] === $stage 시작 ==="
  ( cd "$KIT" && "$PY" scripts/train_diffusion.py --base "$HERE/configs/$stage.yaml" --train --devices 1 )
  echo "[$(date +%H:%M:%S)] === $stage 끝 ==="
}

run stage1
run stage2 stage1 300
run stage3 stage2 2100
run stage4 stage3 1800

echo ">>> 최종 가중치: $(ckpt_at stage4 4800)"
