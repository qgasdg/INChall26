#!/usr/bin/env bash
# DreamZero-SO101 추론 환경 구축 (RTX 5090 / Blackwell sm_120 기준)
#
# 받는 것 ~70GB:
#   Wan-AI/Wan2.1-I2V-14B-480P     베이스 비디오 확산 모델 (Apache-2.0)
#   Vizuara/dreamzero-so101-lora   rank-4 LoRA + 액션 헤드 217MB (Apache-2.0)
#   dreamzero0/dreamzero           DreamZero 본체 코드 (Apache-2.0)
#   vizuara/dreamzero-so101        SO-101 임베디먼트 패치 + 데모 (Apache-2.0)
#
# 사용법:  bash dreamzero/setup_5090.sh [설치경로]   (기본값 ~/dreamzero-work)
set -euo pipefail

ROOT="${1:-$HOME/dreamzero-work}"
ENV_NAME="dreamzero"
PY_VER="3.11"          # pyproject 가 ~=3.11,<3.13 을 요구한다

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "0. 하드웨어·드라이버 확인"
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi 없음 — GPU 드라이버부터 설치"; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 30000 ]; then
  echo "★VRAM ${VRAM}MB — 14B bf16 가중치만 28GB다. 32GB 미만이면 추론도 어렵다."
fi
# 5090 은 sm_120 이라 CUDA 12.8+ 가 필요하다. cu129 휠을 쓴다.

say "1. 파이썬 환경 (conda ${ENV_NAME}, python ${PY_VER})"
if ! command -v conda >/dev/null; then echo "conda 없음 — miniconda 설치 후 재실행"; exit 1; fi
conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "$ENV_NAME" "python=${PY_VER}"
# 이 스크립트 안에서는 conda run 으로 호출한다 (activate 는 셸에 따라 안 먹는다)
PY="conda run -n ${ENV_NAME} --no-capture-output python"
PIP="conda run -n ${ENV_NAME} --no-capture-output pip"

say "2. 저장소 (본체 + SO-101 패치)"
mkdir -p "$ROOT" && cd "$ROOT"
[ -d dreamzero ]        || git clone https://github.com/dreamzero0/dreamzero.git
[ -d dreamzero-so101 ]  || git clone https://github.com/vizuara/dreamzero-so101.git
cd dreamzero
# 패치는 재적용하면 실패하므로 이미 적용됐는지 먼저 본다
if git apply --check ../dreamzero-so101/patches/so101_embodiment.patch 2>/dev/null; then
  git apply ../dreamzero-so101/patches/so101_embodiment.patch
  echo "  so101_embodiment.patch 적용됨"
else
  echo "  패치 이미 적용됨(또는 적용 불가) — git status 로 확인할 것"
fi

say "3. 의존성"
# torch 를 먼저 cu129 로 못박는다. pyproject 가 2.8.0 을 요구하지만 sm_120 지원은 cu129 휠에 있다.
$PIP install --upgrade pip
$PIP install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
$PIP install -e . --extra-index-url https://download.pytorch.org/whl/cu129
$PIP install huggingface_hub safetensors

say "4. flash-attention (선택 — 실패해도 진행)"
# sm_120(Blackwell 소비자용)은 flash-attn 지원이 버전을 탄다. 빌드가 깨지면 torch SDPA 로 돌린다.
if ! MAX_JOBS=8 $PIP install --no-build-isolation flash-attn; then
  echo "  ★flash-attn 설치 실패 — torch SDPA 폴백으로 진행한다. 속도만 손해고 결과는 같다."
fi

say "5. 가중치 (~70GB, 시간 걸린다)"
mkdir -p "$ROOT/checkpoints"
conda run -n "$ENV_NAME" huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir "$ROOT/checkpoints/Wan2.1-I2V-14B-480P"
conda run -n "$ENV_NAME" huggingface-cli download Vizuara/dreamzero-so101-lora \
  --local-dir "$ROOT/checkpoints/dreamzero-so101-lora"

say "6. 검증"
$PY - <<'PYEOF'
import torch, os
print("  torch      :", torch.__version__, "| CUDA", torch.version.cuda)
print("  GPU        :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info()
    print("  compute cap: sm_%d%d" % cc)
    print("  VRAM       : %.1f / %.1f GB 여유" % (free / 2**30, total / 2**30))
    print("  bf16       :", torch.cuda.is_bf16_supported(including_emulation=False))
try:
    import flash_attn; print("  flash-attn :", flash_attn.__version__)
except Exception as e:
    print("  flash-attn : 없음 (%s) — SDPA 폴백" % type(e).__name__)
PYEOF

say "7. 체크포인트 내용"
$PY - <<PYEOF
from pathlib import Path
for d in ("$ROOT/checkpoints/Wan2.1-I2V-14B-480P", "$ROOT/checkpoints/dreamzero-so101-lora"):
    p = Path(d)
    n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**30 if p.exists() else 0
    print("  %-52s %6.1f GB" % (p.name, n))
PYEOF

cat <<EOS

완료. 설치 위치: $ROOT

다음:
  conda activate $ENV_NAME
  cd $ROOT/dreamzero
  # 데모(액션을 '예측'하는 원래 경로)로 파이프라인이 사는지 먼저 확인
  python ../dreamzero-so101/scripts/infer_demo.py \\
    --model-path   $ROOT/checkpoints/dreamzero-so101-lora \\
    --base-model-path $ROOT/checkpoints/Wan2.1-I2V-14B-480P \\
    --image <프레임.jpg> --prompt "pick up the object"

★대회용은 반대 방향(주어진 액션 → 영상)이라 별도 스크립트가 필요하다.
  데모가 도는 것을 확인한 뒤 그걸 작성한다. dreamzero/README.md 참고.
EOS
