#!/bin/bash
# 모든 스크립트가 먼저 읽는 공통 환경. 직접 실행하지 않는다.
#
# 필요한 것은 두 가지뿐이다.
#   FT_ROOT  — 데이터·백본·산출물이 놓이는 작업 루트 (아래 배치도 참고)
#   KIT_ROOT — 대회 킷 루트. 기본값 $FT_ROOT/kit
#
# $FT_ROOT/
#   checkpoints/backbone.ckpt          대회 제공 사전학습 가중치
#   data/train/                        학습 데이터 + so100_{action,delta}_statistics.json
#   data/eval/                         평가 216문제
#   outputs/                           학습 산출 (스크립트가 만든다)
#   out/                               생성 영상 (스크립트가 만든다)
# $KIT_ROOT/
#   baseline/challenge_kit/            대회 킷 — **원본 그대로, 한 줄도 고치지 않았다**
#   baseline/shared_libs/video_utils/
#   submission_kit/make_submission_csv.py

set -euo pipefail

: "${FT_ROOT:?FT_ROOT 를 export 하세요 (예: export FT_ROOT=\$HOME/ft)}"
KIT_ROOT="${KIT_ROOT:-$FT_ROOT/kit}"
KIT="$KIT_ROOT/baseline/challenge_kit"
SUBKIT="$KIT_ROOT/submission_kit"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 백본·데이터·킷이 없으면 받아서 배치한다. 로직은 main.py 한 곳에만 둔다.
"${PY:-python}" "$HERE/main.py" prepare || exit 1

for p in "$FT_ROOT/checkpoints/backbone.ckpt" "$FT_ROOT/data/train" "$FT_ROOT/data/eval" \
         "$KIT/scripts/train_diffusion.py" "$SUBKIT/make_submission_csv.py"; do
  [ -e "$p" ] || { echo "★없음: $p — env.sh 위쪽 배치도를 확인하세요"; exit 1; }
done

# 우리 코드($HERE/src)가 킷보다 앞에 온다 — 설정의 model./data12. 대상이 여기서 풀린다
export PYTHONPATH="$HERE/src:$KIT/src:$KIT/libs/dynamicrafter:$KIT_ROOT/baseline/shared_libs/video_utils:$KIT"
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29733
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-python}"

# Δ 통계는 **대회 제공 파일이 아니라 우리가 학습 데이터에서 만드는 것**이다.
# 액션 12차원의 뒤 6칸(프레임 간 변화량)을 정규화하는 데 쓰는데, 절대값 통계로 나누면
# Δ 가 전역 std 의 8% 수준이라 0 근처로 눌린다. 없으면 여기서 만든다(1분 남짓).
DELTA="$FT_ROOT/data/train/so100_delta_statistics.json"
if [ ! -s "$DELTA" ]; then
  echo ">>> Δ 통계가 없어 새로 만듭니다: $DELTA"
  "$PY" "$HERE/src/make_delta_stats.py" "$FT_ROOT/data/train"
fi

ckpt_at () {   # ckpt_at <단계> <스텝> → 그 스텝의 체크포인트 경로 (epoch 번호는 데이터 크기에 따라 달라진다)
  local f
  f=$(ls "$FT_ROOT/outputs/$1/checkpoints/"*step="$2".ckpt 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "★$1 의 step=$2 체크포인트가 없습니다" >&2; return 1; }
  echo "$f"
}
