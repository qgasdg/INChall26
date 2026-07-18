#!/usr/bin/env bash
# GPU 서버 부트스트랩 — 어느 서버든 이 스크립트 하나로 동일 환경 (docs/07_운영전략.md §1.1)
# 사용: repo 루트에 open.zip을 미리 가져다 둔 뒤  bash scripts/setup_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OPEN_SHA256="66552169b6d037a22a91b1c11497dff0ff1e8595b7a8c7aab957da999d128e8b"

# 1) uv — 유저 공간 설치, sudo 불필요 (연구센터 공유 서버 호환)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2) 파이썬 환경 (uv.lock 기준 완전 재현)
uv sync

# 3) 데이터 — 데이콘 CDN이 인증 없이 접근 가능 확인됨 (2026-07-18, HEAD 200·크기 일치)
OPEN_URL="https://cfiles.dacon.co.kr/competitions/236736/open.zip"
if [ ! -f open.zip ]; then
    echo "open.zip 다운로드 (8.6GB, 데이콘 CDN)"
    curl -fL --retry 3 -C - -o open.zip "$OPEN_URL"
fi
echo "${OPEN_SHA256}  open.zip" | sha256sum -c -
if [ ! -d open/submission_kit ]; then
    unzip -q open.zip -d open
fi

# 4) GPU·킷 임포트 검증
uv run python - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f" | {p.name} {p.total_memory / 2**30:.0f}GB")
else:
    print(" | !! GPU 미인식 — 드라이버/인스턴스 확인 !!")
import sys
sys.path.insert(0, "open/submission_kit")
import action_extractor, feature_csv_utils  # noqa: F401
import pytorch_lightning, timm, omegaconf   # noqa: F401
print("kit imports OK")
PY

echo "== setup 완료. 다음: docs/09_E0_런북.md =="
