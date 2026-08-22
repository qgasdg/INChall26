#!/bin/bash
# b10y 체크포인트 5개(누적 18,000~22,800)를 CFG 2.0 + 배경 고정으로 뽑는다.
R=/runpod
export FT_BASELINE=$R/kit/baseline
KIT=$FT_BASELINE/challenge_kit
export PYTHONPATH=$R/repo/submission/src:$KIT/src:$KIT/libs/dynamicrafter:$FT_BASELINE/shared_libs/video_utils:$KIT
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0
PY=$R/venv/bin/python; D=$R/data
sed -e "s|^\( *resume_unet:\).*|\1 null|" -e "s|^\( *unconditional_guidance_scale:\).*|\1 2.0|" \
    $R/pod-b10y.yaml > $R/infer-b10y-cfg2.yaml
grep -n "unconditional_guidance_scale" $R/infer-b10y-cfg2.yaml
for C in $(ls $R/outputs/b10y/checkpoints/*.ckpt | sort -t= -k3 -n); do
  S=$(echo "$C" | sed 's/.*step=//; s/\.ckpt//'); N=$((16800 + S)); T=b10y-$N-cfg2
  [ -s $R/sub_$T-bg.csv ] && { echo "[$(date +%H:%M:%S)] $T 이미 있음"; continue; }
  echo "[$(date +%H:%M:%S)] === 누적 $N · CFG 2.0 시작 ==="
  $PY $R/repo/submission/src/generate.py --config $R/infer-b10y-cfg2.yaml --challenge-root $D/eval \
    --action-stats-path $D/train/so100_action_statistics.json \
    --delta-stats $D/train/so100_delta_statistics.json \
    --action-dims 12 --no-ema --limit 216 --ckpt "$C" --out $R/out/$T-eval216 2>&1 | tail -1
  $PY $R/repo/submission/src/bgfreeze.py --src $R/out/$T-eval216 --dst $R/out/$T-bg 2>&1 | tail -1
  ( cd $R/kit/submission_kit && $PY make_submission_csv.py --prediction-root $R/out/$T-bg \
      --challenge-root $D/eval --action-stats-path $D/train/so100_action_statistics.json \
      --output-csv $R/sub_$T-bg.csv 2>&1 | tail -1 )
  rm -rf $R/out/$T-eval216 $R/out/$T-bg
  echo "[$(date +%H:%M:%S)] === 누적 $N 끝 ==="
done
echo "[$(date +%H:%M:%S)] b10y CFG 2.0 전부 끝"
