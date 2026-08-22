#!/bin/bash
# 한 판: 생성 → 배경고정 → CSV 2개 → 완료 표식. $1=태그 $2=체크포인트파일명 $3=설정yaml
set -e
TAG=$1; CK=$2; CFG=$3
B=/home/jovyan/work/base
export FT_BASELINE=/home/jovyan/work/INChall26/open/baseline
KIT=$FT_BASELINE/challenge_kit
export PYTHONPATH=$B:$KIT/src:$KIT/libs/dynamicrafter:$FT_BASELINE/shared_libs/video_utils:$KIT
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0
PY=/home/jovyan/work/INChall26/.venv/bin/python
D=/home/jovyan/work/INChall26/open/data
rm -f $B/DONE_$TAG
echo "[$(date +%H:%M:%S)] === $TAG 생성 시작 ($CK · $CFG) ==="
cd $B
$PY generate.py --config $B/$CFG --challenge-root $D/eval \
  --action-stats-path $D/train/so100_action_statistics.json \
  --delta-stats $D/train/so100_delta_statistics.json \
  --action-dims 12 --no-ema --limit 216 --ckpt $B/ckpt/$CK \
  --out $B/out/$TAG-eval216 2>&1 | tail -2
echo "[$(date +%H:%M:%S)] 배경 고정"
$PY bgfreeze.py --src $B/out/$TAG-eval216 --dst $B/out/$TAG-bg 2>&1 | tail -2
for V in "" "-bg"; do
  SRC=$B/out/$TAG$([ -n "$V" ] && echo "-bg" || echo "-eval216")
  echo "[$(date +%H:%M:%S)] CSV $TAG$V"
  cd /home/jovyan/work/INChall26/open/submission_kit
  $PY make_submission_csv.py --prediction-root $SRC --challenge-root $D/eval \
    --action-stats-path $D/train/so100_action_statistics.json \
    --output-csv /home/jovyan/work/sub_$TAG$V.csv 2>&1 | tail -1
  cd $B
done
ls -la /home/jovyan/work/sub_$TAG.csv /home/jovyan/work/sub_$TAG-bg.csv
touch $B/DONE_$TAG
echo "[$(date +%H:%M:%S)] === $TAG 끝 ==="
