#!/usr/bin/env bash
# 베이스라인 생성 환경 부트스트랩 — setup_server.sh 이후 1회 실행 (docs/09 §2 스크립트판)
# 사용: bash scripts/setup_generation.sh
#
# 하는 일: ① hf_hub>=1.0 호환 패치 ② backbone.ckpt 확보 ③ challenge_kit 3종 editable 설치
# 멱등(idempotent) — 재실행해도 안전. open/을 다시 푼 뒤에는 반드시 재실행할 것.
#
# ★ 함정 (2026-07-20 실측): `uv run`은 실행할 때마다 환경을 uv.lock 기준으로 재동기화하며
#   lock에 없는 editable 설치(lvdm·ldwma·video-utils)를 **삭제한다**. 그래서
#   ① editable 설치는 반드시 맨 마지막에 하고
#   ② 이후 생성·추론은 `uv run`이 아니라 `.venv/bin/python`으로 직접 실행해야 한다.
#   (`uv run`을 한 번이라도 타면 lvdm이 사라져 ModuleNotFoundError가 난다)
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="open/baseline"
CK="$BASE/challenge_kit"
PY_BIN=".venv/bin/python"

# 1) huggingface_hub>=1.0 호환 — HfFolder/hf_hub_url/list_repo_files가 제거됨.
#    셋 다 원격 데이터셋 경로에서만 쓰이고 우리는 data root가 로컬이라 그 경로를 타지 않는다.
#    (Kaggle e1-sweep 커널에서 확립한 패치와 동일. 로컬 동작 불변)
"$PY_BIN" - <<'PY'
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

# 2) backbone.ckpt — DynamiCrafter_512 (공개 모델, 토큰 불필요). 하드링크로 캐시 재사용
"$PY_BIN" - <<'PY'
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

# 3) 킷 3종 editable 설치 — 반드시 마지막 (위 주석의 uv run 함정 참조).
#    --no-deps 필수: 킷 requirements를 풀면 uv.lock 정본이 깨짐 (룰북 §5)
#    --python 필수: 이 서버는 CONDA_PREFIX(~/.yennefer/envs/project-env)가 잡혀 있어
#      지정하지 않으면 uv pip이 프로젝트 .venv가 아니라 conda 환경에 설치한다.
#      (설치는 성공하는데 .venv에서는 ModuleNotFoundError가 나는 형태로 드러남)
uv pip install --python "$PY_BIN" --no-deps \
    -e "$CK" -e "$CK/libs/dynamicrafter" -e "$BASE/shared_libs/video_utils"

echo "== 생성 환경 준비 완료. 다음: scripts/run_generation.py =="
