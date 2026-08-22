#!/bin/bash
# 밤새 한 번에 — 6,000스텝 학습 후 다섯 지점을 전부 216문제로 뽑는다.
#
# **노리는 것.** 관측된 주기: 첫 골짜기 9,000 → 꼭대기 13,200 → 두 번째 골짜기 15,600
# (골짜기→꼭대기 4,200 · 꼭대기→골짜기 2,400 · 주기 약 6,600). 같은 주기면
# 세 번째 꼭대기 ≈ 19,800, **세 번째 골짜기 ≈ 22,200**. 22,800 까지 1,200 간격으로 훑는다.
#
# **추론은 CFG 1.0.** 비교 대상인 스텝 곡선이 전부 CFG 1.0 원본이고, 한 판이 30분이라 다섯 점을
# 2.5시간에 끝낸다. 가장 좋은 점이 정해지면 아침에 그것만 CFG 2.0(54분)으로 다시 뽑는다.
set -e
R=/runpod
KIT=$R/kit/baseline/challenge_kit
export FT_BASELINE=$R/kit/baseline
export PYTHONPATH=$R/repo/submission/src:$KIT/src:$KIT/libs/dynamicrafter:$R/kit/baseline/shared_libs/video_utils:$KIT
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29735
export FT_ROOT=$R
PY=$R/venv/bin/python
D=$R/data
say () { echo "[$(date +%H:%M:%S)] $*"; }

# ── 학습: 누적 16,800 → 22,800
sed -e "1s|.*|name: b10y|" -e "s|^\( *max_steps:\).*|\1 6000|" $R/pod-b10x.yaml > $R/pod-b10y.yaml
grep -nE "^name:|max_steps:|every_n_train_steps:" $R/pod-b10y.yaml
export RESUME_CKPT=$R/outputs/b10x/checkpoints/epoch=6-step=3600.ckpt
say "학습 시작 (누적 16,800 → 22,800 · 6,000스텝 · 약 20시간)"
( cd $KIT && $PY scripts/train_diffusion.py --base $R/pod-b10y.yaml --train --devices 1 ) \
  > $R/b10y_train.log 2>&1
say "학습 끝"

# ── 추론: 나온 체크포인트를 스텝 순서대로 전부
sed "s|^\( *resume_unet:\).*|\1 null|" $R/pod-b10y.yaml > $R/infer-b10y.yaml
for C in $(ls $R/outputs/b10y/checkpoints/*.ckpt 2>/dev/null | sort -t= -k3 -n); do
  S=$(echo "$C" | sed 's/.*step=//; s/\.ckpt//')
  N=$((16800 + S))
  T=b10y-$N
  say "=== 누적 $N 생성 시작 ==="
  $PY $R/repo/submission/src/generate.py --config $R/infer-b10y.yaml --challenge-root $D/eval \
    --action-stats-path $D/train/so100_action_statistics.json \
    --delta-stats $D/train/so100_delta_statistics.json \
    --action-dims 12 --no-ema --limit 216 --ckpt "$C" --out $R/out/$T-eval216 2>&1 | tail -1
  $PY $R/repo/submission/src/bgfreeze.py --src $R/out/$T-eval216 --dst $R/out/$T-bg 2>&1 | tail -1
  ( cd $R/kit/submission_kit && $PY make_submission_csv.py --prediction-root $R/out/$T-bg \
      --challenge-root $D/eval --action-stats-path $D/train/so100_action_statistics.json \
      --output-csv $R/sub_$T-bg.csv 2>&1 | tail -1 )
  rm -rf $R/out/$T-eval216 $R/out/$T-bg      # 디스크가 빠듯하다 — CSV 만 남긴다
  say "=== 누적 $N 끝 ==="
done
ls -la $R/sub_b10y-*.csv
say "밤샘 작업 전부 끝 — GPU 가 비면 감시가 pod 을 반납한다"
