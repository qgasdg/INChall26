#!/usr/bin/env bash
# DreamZero 환경 구성 — Runpod / RTX PRO 6000 Blackwell 96GB (sm_120) / Ubuntu 24.04
#
# setup_5090_uv.sh 의 Runpod 판. 다른 점은 딱 하나, **경로**다.
# pod 의 $HOME(/root)은 컨테이너 임시 디스크라 pod 을 끄면 날아간다.
# 네트워크 볼륨만 /workspace 에 붙어 살아남으므로 venv·python·캐시·가중치를
# 전부 /workspace 아래로 몰아넣는다. 안 그러면 켤 때마다 79GB 를 다시 받는다.
set -uo pipefail

ROOT="/workspace/dreamzero-work"
say() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '\n★실패: %s\n' "$*"; }

# uv 가 기본으로 쓰는 자리는 전부 $HOME 아래라 임시 디스크다 → 볼륨으로 돌린다
export UV_INSTALL_DIR="/workspace/bin"
export UV_PYTHON_INSTALL_DIR="/workspace/uv-python"
export UV_CACHE_DIR="/workspace/uv-cache"
export HF_HOME="/workspace/hf"
export PATH="$UV_INSTALL_DIR:$PATH"
mkdir -p "$UV_INSTALL_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$HF_HOME"

say "0/6 환경 확인"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h /workspace | tail -1

say "1/6 uv 설치 (없으면)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || { fail "uv 설치"; exit 1; }
fi
uv --version

say "2/6 python 3.11 준비"
uv python install 3.11 || { fail "python 3.11"; exit 1; }

say "3/6 저장소"
mkdir -p "$ROOT" && cd "$ROOT"
[ -d dreamzero ] || git clone https://github.com/dreamzero0/dreamzero.git || { fail "clone"; exit 1; }

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
uv pip install --quiet huggingface_hub safetensors hf_transfer
# flash-attn 은 sm_120 지원이 버전을 타므로 실패해도 넘어간다(SDPA 폴백)
MAX_JOBS=8 uv pip install --no-build-isolation flash-attn 2>&1 | tail -3 || \
  echo "  flash-attn 실패 — torch SDPA 폴백으로 진행(속도만 손해)"

say "6/6 가중치 다운로드 (베이스 ~79GB + LoRA 217MB)"
cd "$ROOT"
mkdir -p checkpoints
# 5090 판은 hf_transfer 를 껐지만 여기선 켠다. pod 은 시간당 과금이라 79GB 를
# 기본 속도로 받는 대기시간이 그대로 돈이다.
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Vizuara/dreamzero-so101-lora --local-dir checkpoints/dreamzero-so101-lora \
  || { fail "LoRA 다운로드"; exit 1; }
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir checkpoints/Wan2.1-I2V-14B-480P \
  || { fail "베이스 다운로드"; exit 1; }

say "완료 — 용량"
du -sh "$ROOT"/checkpoints/* 2>/dev/null
df -h /workspace | tail -1
echo "===EXIT=0"
