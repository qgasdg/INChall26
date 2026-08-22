#!/bin/bash
# 누적 10,800 주변을 **최종 레시피(CFG 2.0 + 배경 고정)로** 훑는다.
#
# **왜.** 8/19 밤에 누적 10,800 · CFG 2.0 · 배경 고정 = **0.1810** 이 나왔다(직전 최고 0.2070).
# 그런데 이 구간은 CFG 1.0 곡선에서 10,200 = 0.2794 로 "최악"이라 통째로 버렸던 곳이다.
# 즉 **CFG 이득의 크기가 체크포인트마다 극단적으로 달라서, CFG 1.0 점수로 스텝을 고른 것이 틀렸다.**
# 최종 레시피 기준으로 이 구간을 다시 재야 한다.
#
# **순서의 근거.** 8/19 에 갈래내 수프(9,000 × 9,600)가 두 부모를 다 이겼다(0.2042).
# 같은 궤적 위 이웃끼리 섞으면 이득이 나므로, 10,800 을 중심에 둔 수프를 이웃 측정보다 앞에 둔다.
set -e
B=/home/jovyan/work/base
export FT_BASELINE=/home/jovyan/work/INChall26/open/baseline
KIT=$FT_BASELINE/challenge_kit
export PYTHONPATH=$B:$KIT/src:$KIT/libs/dynamicrafter:$FT_BASELINE/shared_libs/video_utils:$KIT
export USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0
PY=/home/jovyan/work/INChall26/.venv/bin/python
say () { echo "[$(date +%H:%M:%S)] $*"; }
cd $B

# 10,800 을 중심에 둔 수프 두 개를 먼저 만들어 둔다(가중치 평균은 1분, GPU 불필요)
[ -s ckpt/soup-10200x10800.ckpt ] || $PY soup.py ckpt/base10-6000.ckpt ckpt/base10-6600.ckpt ckpt/soup-10200x10800.ckpt 0.5 | tail -2
[ -s ckpt/soup-10800x12600.ckpt ] || $PY soup.py ckpt/base10-6600.ckpt ckpt/base10-8400.ckpt ckpt/soup-10800x12600.ckpt 0.5 | tail -2

# 한 판 2시간 46분(CFG 2.0). 값어치 큰 순서로.
for E in "soup-10200x10800:soup-10200x10800.ckpt" \
         "b10-10200:base10-6000.ckpt" \
         "soup-10800x12600:soup-10800x12600.ckpt" \
         "b10-12600:base10-8400.ckpt" \
         "b10-9600:base10-5400.ckpt" \
         "b10-13200:base10-9000.ckpt"; do
  T=${E%%:*}-cfg2; C=${E##*:}
  [ -s /home/jovyan/work/sub_$T-bg.csv ] && { say "$T 이미 있음 — 건너뜀"; continue; }
  say "=== $T 시작 ==="
  bash $B/run_one.sh "$T" "$C" base-10-cfg2.0.yaml
  rm -rf $B/out/$T-eval216 $B/out/$T-bg     # CSV 만 남긴다
  say "=== $T 끝 ==="
done
say "V100 야간 작업 전부 끝"
