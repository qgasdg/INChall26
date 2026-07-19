#!/usr/bin/env bash
# 베이스라인 생성 환경 부트스트랩 — setup_server.sh 이후 1회 실행 (docs/09 §2 스크립트판)
# 사용: bash scripts/setup_generation.sh
#
# 하는 일: ① challenge_kit 3종 editable 설치 ② hf_hub>=1.0 호환 패치 ③ backbone.ckpt 확보
# 멱등(idempotent) — 재실행해도 안전. open/을 다시 푼 뒤에는 반드시 재실행할 것.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="open/baseline"
CK="$BASE/challenge_kit"

# 1) 킷 3종 editable 설치 — --no-deps 필수 (킷 requirements를 풀면 uv.lock 정본이 깨짐, 룰북 §5)
uv pip install --no-deps -e "$CK" -e "$CK/libs/dynamicrafter" -e "$BASE/shared_libs/video_utils"

# 2) huggingface_hub>=1.0 호환 — HfFolder/hf_hub_url/list_repo_files가 제거됨.
#    셋 다 원격 데이터셋 경로에서만 쓰이고 우리는 data root가 로컬이라 그 경로를 타지 않는다.
#    (Kaggle e1-sweep 커널에서 확립한 패치와 동일. 로컬 동작 불변)
uv run python - <<'PY'
from pathlib import Path

p = Path("open/baseline/challenge_kit/src/ldwma/datasets/lerobot_so100.py")
s = p.read_text()
old = "from huggingface_hub import HfFolder, hf_hub_download, hf_hub_url, list_repo_files"
new = (
    "from huggingface_hub import hf_hub_download\n"
    "try:\n"
    "    from huggingface_hub import hf_hub_url, list_repo_files\n"
    "except ImportError:  # hf_hub>=1.0\n"
    "    hf_hub_url = list_repo_files = None\n"
    "try:\n"
    "    from huggingface_hub import HfFolder\n"
    "except ImportError:  # hf_hub>=1.0\n"
    "    class HfFolder:\n"
    "        @staticmethod\n"
    "        def get_token():\n"
    "            return None\n"
)
if old in s:
    p.write_text(s.replace(old, new))
    print("patched lerobot_so100.py (hf_hub>=1.0 호환)")
else:
    assert "except ImportError:  # hf_hub>=1.0" in s, f"예상치 못한 import 형태: {p}"
    print("이미 패치됨 — 건너뜀")
PY

# 3) backbone.ckpt — DynamiCrafter_512 (공개 모델, 토큰 불필요). 하드링크로 캐시 재사용
uv run python - <<'PY'
import os
from huggingface_hub import hf_hub_download

dst = "open/baseline/checkpoints/backbone.ckpt"
if os.path.exists(dst):
    print(f"backbone.ckpt 이미 존재 ({os.path.getsize(dst) // 2**20} MB) — 건너뜀")
else:
    src = os.path.realpath(hf_hub_download("Doubiiu/DynamiCrafter_512", "model.ckpt"))
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy(src, dst)
    print(f"backbone.ckpt 준비 완료 ({os.path.getsize(dst) // 2**20} MB)")
PY

echo "== 생성 환경 준비 완료. 다음: scripts/run_generation.py =="
