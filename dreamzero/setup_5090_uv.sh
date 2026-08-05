#!/usr/bin/env bash
# DreamZero 환경 구성 — RTX 5090 (sm_120) / Ubuntu 22.04 / sudo 없음
#
# ★ ~/ctrl-world 와 ~/ckpt/ctrl-world 는 건드리지 않는다. 전부 ~/dreamzero-work 아래에만 만든다.
#
# 시스템에 python3.10 뿐인데 dreamzero 는 3.11~3.12 를 요구한다 → uv 로 3.11 을 별도 설치한다
# (sudo 불필요, 시스템 파이썬 무변경).
set -uo pipefail

ROOT="$HOME/dreamzero-work"
say() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\n★실패: %s\n' "$*"; }

say "0/6 환경 확인"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h "$HOME" | tail -1
echo "  ctrl-world 는 건드리지 않음: $(du -sh "$HOME/ctrl-world" 2>/dev/null | cut -f1) 유지"

say "1/6 uv 설치 (없으면)"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || { fail "uv 설치"; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "2/6 python 3.11 준비"
uv python install 3.11 || { fail "python 3.11"; exit 1; }

say "3/6 저장소"
mkdir -p "$ROOT" && cd "$ROOT"
[ -d dreamzero ] || git clone https://github.com/dreamzero0/dreamzero.git || { fail "clone"; exit 1; }
# so101 저장소는 웹사이트라 코드가 없다(2026-08-05 확인). 통합 코드는 우리가 작성한다.

say "4/6 가상환경 + torch (sm_120 이라 cu129 휠이 필요하다)"
cd "$ROOT"
[ -d .venv ] || uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --quiet torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129 || { fail "torch"; exit 1; }
python - <<'PY'
import torch
print("  torch", torch.__version__, "| cuda", torch.version.cuda)
print("  arch :", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    print("  GPU  :", torch.cuda.get_device_name(0), "sm_%d%d" % torch.cuda.get_device_capability(0))
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    print("  bf16 행렬곱:", float((a @ a).float().sum()))
PY

say "5/6 dreamzero 의존성"
cd "$ROOT/dreamzero"
uv pip install --quiet -e . --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -5
uv pip install --quiet huggingface_hub safetensors
# flash-attn 은 sm_120 지원이 버전을 타므로 실패해도 넘어간다(SDPA 폴백)
MAX_JOBS=8 uv pip install --no-build-isolation flash-attn 2>&1 | tail -3 || \
  echo "  flash-attn 실패 — torch SDPA 폴백으로 진행(속도만 손해)"

say "6/6 가중치 다운로드 (베이스 ~79GB + LoRA 217MB)"
cd "$ROOT"
mkdir -p checkpoints
export HF_HUB_ENABLE_HF_TRANSFER=0
hf download Vizuara/dreamzero-so101-lora --local-dir checkpoints/dreamzero-so101-lora \
  || python -m huggingface_hub.commands.huggingface_cli download Vizuara/dreamzero-so101-lora \
       --local-dir checkpoints/dreamzero-so101-lora
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir checkpoints/Wan2.1-I2V-14B-480P \
  || python -m huggingface_hub.commands.huggingface_cli download Wan-AI/Wan2.1-I2V-14B-480P \
       --local-dir checkpoints/Wan2.1-I2V-14B-480P

say "완료 — 용량"
du -sh "$ROOT"/checkpoints/* 2>/dev/null
df -h "$HOME" | tail -1
echo "===EXIT=0"
