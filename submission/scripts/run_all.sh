#!/bin/bash
# 백본에서 제출 CSV까지 통째로. GPU 1장 · 약 27시간.
#   export FT_ROOT=$HOME/ft && bash scripts/run_all.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
bash "$HERE/scripts/train.sh"
bash "$HERE/scripts/infer.sh"
