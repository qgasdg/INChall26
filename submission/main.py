#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""백본에서 제출 CSV 까지 — 파이썬 진입점.

규정이 코드 파일 확장자를 .py / .ipynb 로 정하고 있어, scripts/*.sh 와 **같은 일을 하는**
파이썬 드라이버를 둔다. 셸 스크립트는 편의용이고 이 파일 하나로 전 구간이 돈다.

    export FT_ROOT=~/ft            # 배치도는 README 4절
    export KIT_ROOT=$FT_ROOT/kit   # 생략 시 $FT_ROOT/kit
    python main.py all             # 학습 31시간 + 추론 52분
    python main.py train           # 학습만  (백본 → 누적 10,800스텝)
    python main.py infer           # 추론만  (마지막 체크포인트 → 제출 CSV)
    python main.py infer --ckpt <경로> --limit 2      # 배선 확인용

경로는 전부 FT_ROOT / KIT_ROOT 기준 상대 위치다 — 소스에 절대 경로가 없다.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# 킷의 baseline.ipynb 7번 셀이 쓰는 것과 같은 주소다. 대회가 별도로 배포하는 파일이 아니라
# 공개된 DynamiCrafter 512 가중치이며, 없으면 여기서 받는다.
BACKBONE_URL = "https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt"

# 대회 배포 꾸러미(킷 + 학습/평가 데이터). 인증 없이 받을 수 있다.
OPEN_URL = "https://cfiles.dacon.co.kr/competitions/236736/open.zip"
OPEN_SIZE = 9209529683


def fetch(url: str, dst: Path, expect: int | None = None, what: str = "") -> None:
    """큰 파일을 받는다. curl 이 있으면 그쪽을 쓴다 — 이어받기가 되고,
    파이썬의 SSL 인증서 설정에 의존하지 않는다(맥 기본 파이썬에서 자주 막힌다).

    받는 동안에는 .part 로 두고, 크기를 확인한 뒤에만 제자리로 옮긴다.
    반쪽짜리 파일로 학습이 시작되면 36시간을 헛돈다.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    log(f"{what} 내려받습니다: {url}")
    if shutil.which("curl"):
        r = subprocess.run(["curl", "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(tmp), url])
        if r.returncode != 0:
            sys.exit(f"★내려받기 실패(curl {r.returncode}): {url}")
    else:
        urllib.request.urlretrieve(url, tmp)
    n = tmp.stat().st_size
    if expect is not None and n != expect:
        sys.exit(f"★크기가 다릅니다({n} != {expect}) — {tmp} 를 지우고 다시 실행하세요")
    if expect is None and n < 5 * 2**30:
        tmp.unlink(missing_ok=True)
        sys.exit(f"★내려받기가 중간에 끊겼습니다({n/2**30:.2f}GB) — 다시 실행하세요")
    tmp.rename(dst)
    log(f"{what} 준비 완료: {dst} ({n/2**30:.2f}GB)")


def ensure_backbone(ft: Path) -> None:
    """백본이 없으면 내려받는다(약 9.7GB).

    이것이 없으면 학습이 시작되지 않는데, 킷에서는 baseline.ipynb 를 실행해야만
    받아진다. 우리 파이프라인은 그 노트북을 거치지 않으므로 여기서 챙긴다.
    """
    dst = ft / "checkpoints" / "backbone.ckpt"
    if dst.exists() and dst.stat().st_size > 5 * 2**30:
        return
    fetch(BACKBONE_URL, dst, what="백본(약 9.7GB)을")


def ensure_dataset(ft: Path, kit_root: Path) -> None:
    """대회 데이터와 킷이 없으면 open.zip 을 받아 배치한다(약 8.6GB).

    경로를 하드코딩하는 이유: 처음에 `find -name train` 으로 찾게 했더니 킷의
    설정 폴더(challenge_kit/configs/train)를 먼저 잡아서, 학습 데이터가 0건인 채로
    학습이 시작된 적이 있다. 꾸러미 안의 구조는 고정이므로 그대로 적는다.
    """
    need = [
        ft / "data" / "train", ft / "data" / "eval",
        kit_root / "baseline" / "challenge_kit" / "scripts" / "train_diffusion.py",
        kit_root / "submission_kit" / "make_submission_csv.py",
    ]
    if all(p.exists() for p in need):
        return

    pkg = ft / "openpkg"
    if not (pkg / "data" / "train").exists():
        zp = ft / "open.zip"
        if not (zp.exists() and zp.stat().st_size == OPEN_SIZE):
            fetch(OPEN_URL, zp, expect=OPEN_SIZE, what="대회 데이터·킷(약 8.6GB)을")
        log("압축을 풉니다")
        pkg.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(pkg)
        zp.unlink(missing_ok=True)   # 8.6GB — 풀고 나면 필요 없다

    def link(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)

    link(pkg / "data" / "train", ft / "data" / "train")
    link(pkg / "data" / "eval", ft / "data" / "eval")
    link(pkg / "baseline", kit_root / "baseline")
    link(pkg / "submission_kit", kit_root / "submission_kit")
    log(f"데이터·킷 배치 완료: {pkg}")


def roots() -> tuple[Path, Path]:
    ft = os.environ.get("FT_ROOT")
    if not ft:
        sys.exit("★FT_ROOT 를 export 하세요 (예: export FT_ROOT=$HOME/ft) — 배치도는 README 4절")
    ft = Path(ft).expanduser().resolve()
    kit_root = Path(os.environ.get("KIT_ROOT", ft / "kit")).expanduser().resolve()
    ensure_backbone(ft)
    ensure_dataset(ft, kit_root)
    need = [
        ft / "checkpoints" / "backbone.ckpt",
        ft / "data" / "train",
        ft / "data" / "eval",
        kit_root / "baseline" / "challenge_kit" / "scripts" / "train_diffusion.py",
        kit_root / "submission_kit" / "make_submission_csv.py",
    ]
    missing = [p for p in need if not p.exists()]
    if missing:
        sys.exit("★없음:\n  " + "\n  ".join(str(p) for p in missing) + "\n배치도는 README 4절을 보세요")
    return ft, kit_root


def environ(ft: Path, kit_root: Path) -> dict:
    kit = kit_root / "baseline" / "challenge_kit"
    env = dict(os.environ)
    # 우리 코드(src)가 킷보다 앞에 온다 — 설정의 model./data12. 대상이 여기서 풀린다
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(HERE / "src"),
            str(kit / "src"),
            str(kit / "libs" / "dynamicrafter"),
            str(kit_root / "baseline" / "shared_libs" / "video_utils"),
            str(kit),
        ]
    )
    env.update(
        FT_ROOT=str(ft),
        LOCAL_RANK="0", RANK="0", WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29733",
        USE_TF="0", TRANSFORMERS_NO_TF="1", USE_FLAX="0",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    return env


def run(cmd: list, env: dict, cwd: Path | None = None) -> None:
    r = subprocess.run([str(c) for c in cmd], env=env, cwd=None if cwd is None else str(cwd))
    if r.returncode != 0:
        sys.exit(f"★실패({r.returncode}): {' '.join(str(c) for c in cmd)}")


def ensure_delta_stats(ft: Path, env: dict) -> None:
    """Δ 통계는 대회 제공 파일이 아니라 학습 데이터에서 우리가 만드는 것이다.

    액션 12차원의 뒤 6칸(프레임 간 변화량)을 정규화하는 데 쓴다. 절대값 통계로 나누면
    Δ 가 전역 std 의 8% 수준이라 0 근처로 눌린다. 없으면 여기서 만든다(1분 남짓).
    """
    delta = ft / "data" / "train" / "so100_delta_statistics.json"
    if delta.exists() and delta.stat().st_size > 0:
        return
    log(f"Δ 통계가 없어 새로 만듭니다: {delta}")
    run([sys.executable, HERE / "src" / "make_delta_stats.py", ft / "data" / "train"], env)


def ckpt_at(ft: Path, stage: str, step: int) -> Path:
    """그 스텝의 체크포인트. epoch 번호는 데이터 크기에 따라 달라져서 글롭으로 찾는다."""
    hits = sorted(glob.glob(str(ft / "outputs" / stage / "checkpoints" / f"*step={step}.ckpt")))
    if not hits:
        sys.exit(f"★{stage} 의 step={step} 체크포인트가 없습니다")
    return Path(hits[0])


# 실제로 학습한 순서 그대로다. 단계를 넘을 때 UNet 가중치만 물려받고 옵티마이저 상태는
# 새로 시작하며 학습률 워밍업도 다시 돈다 — 한 번에 10,800스텝을 도는 것과 결과가 다르다.
STAGES = [
    ("stage1", None,               300),    # 인코더 동결 · 액션 조건 없음
    ("stage2", ("stage1",   300), 2100),    # 인코더 해제 + 8비트 AdamW + 액션 cross-attention
    ("stage3", ("stage2",  2100), 1800),
    ("stage4", ("stage3",  1800), 6600),    # → 누적 10,800 · 제출 점수를 낸 모델
]


def do_train(ft: Path, kit_root: Path, env: dict) -> Path:
    kit = kit_root / "baseline" / "challenge_kit"
    for stage, resume, steps in STAGES:
        e = dict(env)
        if resume is not None:
            src = ckpt_at(ft, *resume)
            e["RESUME_CKPT"] = str(src)
            log(f"{stage} ← {src}")
        log(f"=== {stage} 시작 ({steps}스텝) ===")
        run([sys.executable, "scripts/train_diffusion.py",
             "--base", HERE / "configs" / f"{stage}.yaml", "--train", "--devices", "1"], e, cwd=kit)
        log(f"=== {stage} 끝 ===")
    final = ckpt_at(ft, "stage4", 6600)
    log(f">>> 최종 가중치: {final}")
    return final


def do_infer(ft: Path, kit_root: Path, env: dict, ckpt: Path, out: Path, limit: int) -> None:
    data = ft / "data"
    astats = data / "train" / "so100_action_statistics.json"
    dstats = data / "train" / "so100_delta_statistics.json"

    log(f"=== 생성 시작 · {ckpt} ===")
    run([sys.executable, HERE / "src" / "generate.py",
         "--config", HERE / "configs" / "infer.yaml",
         "--challenge-root", data / "eval",
         "--action-stats-path", astats, "--delta-stats", dstats,
         "--action-dims", "12", "--no-ema", "--start", "0", "--limit", limit,
         "--ckpt", ckpt, "--out", f"{out}-eval216"], env)

    log("=== 배경 고정 ===")
    run([sys.executable, HERE / "src" / "bgfreeze.py",
         "--src", f"{out}-eval216", "--dst", f"{out}-bg"], env)

    # 두 벌 다 만든다 — 영상 후처리가 규정상 불가하다는 판단이 나오면 -raw 를 쓴다
    subkit = kit_root / "submission_kit"
    for src, tag in ((f"{out}-eval216", "-raw"), (f"{out}-bg", "-bg")):
        log(f"=== CSV{tag} ===")
        run([sys.executable, "make_submission_csv.py",
             "--prediction-root", src, "--challenge-root", data / "eval",
             "--action-stats-path", astats,
             "--output-csv", ft / f"submission{tag}.csv"], env, cwd=subkit)
    log(f">>> 제출본: {ft / 'submission-bg.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="백본 → 학습 → 추론 → 제출 CSV")
    ap.add_argument("step", choices=["all", "train", "infer", "prepare"],
                    nargs="?", default="all",
                    help="prepare = 백본·데이터·킷·Δ통계만 준비하고 끝낸다")
    ap.add_argument("--ckpt", default=None, help="추론에 쓸 가중치. 생략 시 stage4 step=6600(=누적 10,800)")
    ap.add_argument("--limit", type=int, default=216, help="문제 수. 배선 확인은 2 정도로")
    ap.add_argument("--out", default=None, help="생성 영상 폴더. 기본 $FT_ROOT/out/final")
    args = ap.parse_args()

    ft, kit_root = roots()
    env = environ(ft, kit_root)
    ensure_delta_stats(ft, env)

    if args.step == "prepare":
        log("준비 완료 — 학습은 `python main.py train`")
        return

    ckpt = None
    if args.step in ("all", "train"):
        ckpt = do_train(ft, kit_root, env)
    if args.step in ("all", "infer"):
        if args.ckpt:
            ckpt = Path(args.ckpt).expanduser().resolve()
        elif ckpt is None:
            ckpt = ckpt_at(ft, "stage4", 6600)
        out = Path(args.out).expanduser() if args.out else ft / "out" / "final"
        do_infer(ft, kit_root, env, ckpt, out, args.limit)


if __name__ == "__main__":
    main()
