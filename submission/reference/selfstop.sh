#!/bin/bash
# pod 이 스스로 **CSV 를 HF 에 올리고** 반납한다.
#
# 왜: 감시가 맥에서 돌면 맥이 잠들거나 배터리가 나갈 때 같이 죽는다(오늘 배터리 4시간 남음).
# 그리고 반납만 하면 CSV 가 정지된 pod 볼륨에 갇혀서, 점수를 내려면 pod 을 다시 켜야 한다.
# HF 쓰기는 방금 복구를 확인했다(어제는 용량 초과로 막혀 있었다).
ID=1tan99biej6oxm
L=/runpod/selfstop.log
TOK=${HF_TOKEN}
PY=/runpod/venv/bin/python
say () { echo "[$(date -u +'%m-%d %H:%M UTC')] $*" >> "$L"; }

upload () {
  # CSV 는 한 장 27MB — gzip 으로 줄여서 올린다. 실패해도 반납은 계속한다.
  cd /runpod || return 1
  for f in sub_b10y-*.csv; do
    [ -s "$f" ] || continue
    [ -s "$f.gz" ] || gzip -c "$f" > "$f.gz"
  done
  ls -la /runpod/sub_b10y-*.csv.gz >> "$L" 2>&1
  HF_TOKEN=$TOK $PY - >> "$L" 2>&1 <<PYEOF
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/runpod/sub_b10y-*.csv.gz")):
    n = os.path.basename(p)
    try:
        api.upload_file(path_or_fileobj=p, path_in_repo="csv/"+n, repo_id="YunTTak/inchall-relay")
        print(">>> 올림:", n)
    except Exception as e:
        print(">>> 실패:", n, str(e)[:200])
PYEOF
}

say "자기반납 감시 시작 (5분 주기) — 끝나면 HF 업로드 후 반납"
GONE=0
while true; do
  sleep 300
  N=$(ps -eo cmd | grep -c "[b]10y_cfg2.sh")
  C=$(ls /runpod/sub_b10y-*.csv 2>/dev/null | wc -l)
  if [ "$N" -gt 0 ]; then GONE=0
  else GONE=$((GONE+1)); say "끝난 듯 ($GONE/2 · CSV $C장)"; fi
  if [ "$GONE" -ge 2 ]; then
    say "★업로드 시작 (CSV $C장)"
    upload
    say "★★자기반납 실행"
    runpodctl stop pod "$ID" >> "$L" 2>&1
    sleep 120
    runpodctl stop pod "$ID" >> "$L" 2>&1
    exit 0
  fi
done
