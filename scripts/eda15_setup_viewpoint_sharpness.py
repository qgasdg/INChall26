"""셋업 A/B 뭉개짐 원인 분리 — 학습 데이터로 '세팅 자체 vs 미학습' 판별.

■ 왜 하는가
exp-08에서 생성 영상의 '뭉개짐'(프레임이 진행될수록 로봇 팔이 흐릿한 얼룩으로 녹음)이
셋업 B에서 1.6배 심했다 — 마지막 프레임 선명도 유지율 A 66.6% / B 40.5%
(results/generated_sharpness.csv). 남은 질문: 이 뭉개짐이
  (1) '처음 보는 데이터라서(미학습)'인가, 아니면
  (2) '그 세팅(시점·배경) 자체가 어려워서'인가?
eval만으로는 둘을 못 가른다(eval은 모델이 학습에 쓰지 않은 것).

■ 어떻게
베이스라인 모델이 **학습에 이미 쓴** train 에피소드를 겉모습 기준으로 두 무리로 나눠
똑같이 생성·측정한다.
  - A-닮음 무리: 겉모습이 셋업 A(위에서 내려다봄)에 가까운 train 데이터셋
  - B-닮음 무리: 겉모습이 셋업 B(옆 책상 상향)에 가까운 train 데이터셋
분류는 results/dataset_eval_similarity.csv의 setupA/B 유사도(eda05 DINO 유사도)로 한다.
모델이 이미 본 데이터인데도 B-닮음이 더 뭉개지면 → 원인은 '미학습'이 아니라 '세팅 자체'다.

■ 측정
생성 mp4의 선명도(라플라시안 분산)를 frame0 / frame15에서 재고 유지율(f15/f0)을 무리별로
비교. 참고로 D+V+Action도 정본 채점기(score_v2.py)로 같이 산출(채점 로직 중복 구현 금지).

■ 한계
겉모습(A/B 유사도)은 '시점 각도'와 완전히 같지 않다(docs/17 §1.3). 따라서 이 실험은
'세팅 유형'을 가르는 1차 근사다. 순수 '각도'만 분리하려면 시점 라벨링이 별도로 필요.

사용:
  .venv/bin/python scripts/eda15_setup_viewpoint_sharpness.py \
      [--datasets-per-bin 6] [--episodes-per-dataset 3] [--ddim-steps 50] [--seed 0]
산출:
  results/viewpoint_sharpness_per_sample.csv (에피소드별)
  results/viewpoint_sharpness_summary.csv    (무리별 집계)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "local_eval"))
from episode_io import TRAIN_ROOT, episode_paths, read_actions, read_frames  # noqa: E402

CK = ROOT / "open" / "baseline" / "challenge_kit"
SIM_CSV = ROOT / "results" / "dataset_eval_similarity.csv"
WINDOW = 16

# 학습에서 빼는 데이터셋 (룰북 §2 홀드아웃 unseen 4종 + docs/17 §1.1 10fps)
EXCLUDE = {
    "aractingi/push_cube_offline_data", "frk2/so100large",
    "CSCSXX/pick_place_cube_1.18", "ZGGZZG/so100_drop0",
    "dragon-95/so100_sorting",
}


def list_episodes(dataset: str) -> list[int]:
    d = TRAIN_ROOT / dataset / "data" / "chunk-000"
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("episode_*.parquet"))


def ep_length(dataset: str, ep: int) -> int:
    _, pq = episode_paths(dataset, ep)
    return int(pd.read_parquet(pq, columns=["action"]).shape[0])


def pick_samples(bin_label: str, datasets: list[str], eps_per_ds: int) -> list[dict]:
    """무리별 데이터셋 목록에서 유효 에피소드(≥16프레임)를 골라 샘플 스펙 생성."""
    out = []
    for ds in datasets:
        got = 0
        for ep in list_episodes(ds):
            if got >= eps_per_ds:
                break
            try:
                n = ep_length(ds, ep)
            except Exception:
                continue
            if n < WINDOW:
                continue
            start = max(0, (n - WINDOW) // 2)          # 중간 윈도우(움직임 잡기 쉬움)
            sid = f"vp{bin_label}__{ds.replace('/', '__')}__ep{ep:06d}__t{start:04d}"
            out.append({"tier": "indomain", "dataset": ds, "episode_index": ep,
                        "start": start, "length": WINDOW, "sample_id": sid,
                        "bin": bin_label})
            got += 1
    return out


def build_inputs(samples: list[dict], inputs: Path) -> None:
    """challenge 입력 포맷: images/<sid>.png(첫 프레임) + actions/<sid>.npy(16×6 raw deg)."""
    from PIL import Image
    (inputs / "images").mkdir(parents=True, exist_ok=True)
    (inputs / "actions").mkdir(parents=True, exist_ok=True)
    for s in samples:
        f0 = read_frames(s["dataset"], s["episode_index"], s["start"], 1)[0]
        acts = read_actions(s["dataset"], s["episode_index"], s["start"], WINDOW)
        assert acts.shape == (WINDOW, 6), (s["sample_id"], acts.shape)
        Image.fromarray(f0).save(inputs / "images" / f"{s['sample_id']}.png")
        np.save(inputs / "actions" / f"{s['sample_id']}.npy", acts)


def generate(inputs: Path, pred: Path, steps: int, seed: int, log: Path) -> float:
    """베이스라인 생성기를 무수정 재사용 (eda13과 동일 규약, cwd=challenge_kit)."""
    pred.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/inference/generate_baseline_videos.py",
           "--challenge-root", str(inputs.resolve()),
           "--prediction-root", str(pred.resolve()),
           "--ddim-steps", str(steps), "--seed", str(seed), "--overwrite"]
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=CK, check=True, stdout=f, stderr=subprocess.STDOUT)
    return time.time() - t0


def decode_mp4(path: Path, n: int = WINDOW) -> np.ndarray:
    frames = []
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i >= n:
                break
            frames.append(fr.to_ndarray(format="rgb24"))
    return np.stack(frames)


def lap_var(frame_rgb: np.ndarray) -> float:
    """라플라시안 분산 = 선명도. 4-이웃 이산 라플라시안(순수 numpy, cv2 의존 없음)."""
    g = frame_rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def score(pred: Path, holdout_json: Path, out_csv: Path) -> pd.DataFrame:
    """정본 채점기 score_v2.py 재사용 (indomain tier로 미니 홀드아웃 채점)."""
    subprocess.run([sys.executable, str(ROOT / "local_eval" / "score_v2.py"),
                    "--videos", str(pred.resolve()),
                    "--holdout", str(holdout_json.resolve()),
                    "--tiers", "indomain", "--csv", str(out_csv.resolve())],
                   check=True, cwd=ROOT / "local_eval", stdout=subprocess.DEVNULL)
    return pd.read_csv(out_csv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-per-bin", type=int, default=6)
    ap.add_argument("--episodes-per-dataset", type=int, default=3)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workdir", default="local_runs/vp_diag")
    args = ap.parse_args()

    df = pd.read_csv(SIM_CSV)
    df = df[~df["dataset"].isin(EXCLUDE)].copy()
    df["ab"] = df["sim_mean_setupA"] - df["sim_mean_setupB"]   # +면 A-닮음, -면 B-닮음
    a_ds = df.sort_values("ab", ascending=False)["dataset"].head(args.datasets_per_bin).tolist()
    b_ds = df.sort_values("ab", ascending=True)["dataset"].head(args.datasets_per_bin).tolist()
    print(f"A-닮음 데이터셋({len(a_ds)}): {a_ds}", flush=True)
    print(f"B-닮음 데이터셋({len(b_ds)}): {b_ds}", flush=True)

    samples = (pick_samples("A", a_ds, args.episodes_per_dataset)
               + pick_samples("B", b_ds, args.episodes_per_dataset))
    n_a = sum(s["bin"] == "A" for s in samples)
    n_b = sum(s["bin"] == "B" for s in samples)
    print(f"샘플 {len(samples)}개 (A {n_a} / B {n_b})", flush=True)

    work = ROOT / args.workdir
    work.mkdir(parents=True, exist_ok=True)
    holdout_json = work / "holdout_vp.json"
    holdout_json.write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=1))

    inputs, pred = work / "inputs", work / "pred"
    print("입력 변환 중...", flush=True)
    build_inputs(samples, inputs)
    print("생성 중...", flush=True)
    gen_s = generate(inputs, pred, args.ddim_steps, args.seed, work / "gen.log")
    print(f"생성 완료 {gen_s:.0f}s", flush=True)

    # 선명도(라플라시안 분산) frame0 / frame15
    sharp = []
    for s in samples:
        mp4 = pred / f"{s['sample_id']}.mp4"
        if not mp4.exists():
            continue
        fr = decode_mp4(mp4)
        v0, v15 = lap_var(fr[0]), lap_var(fr[-1])
        sharp.append({"sample": s["sample_id"], "bin": s["bin"],
                      "sharp_f0": v0, "sharp_f15": v15,
                      "sharp_ratio": v15 / v0 if v0 > 0 else np.nan})
    sh = pd.DataFrame(sharp)

    # D+V+Action (정본 채점기)
    print("채점 중...", flush=True)
    sc = score(pred, holdout_json, work / "scores.csv")
    per = sh.merge(sc[["sample", "dino_pf", "video", "action", "total_pf"]],
                   on="sample", how="left")
    per.to_csv(ROOT / "results" / "viewpoint_sharpness_per_sample.csv", index=False)

    # 무리별 집계
    agg = (per.groupby("bin")
           .agg(n=("sample", "size"),
                sharp_ratio=("sharp_ratio", "mean"),
                sharp_f0=("sharp_f0", "mean"),
                sharp_f15=("sharp_f15", "mean"),
                action=("action", "mean"),
                dino_pf=("dino_pf", "mean"),
                video=("video", "mean"),
                total_pf=("total_pf", "mean"))
           .reset_index())
    agg.to_csv(ROOT / "results" / "viewpoint_sharpness_summary.csv", index=False)

    rho = per[["sharp_ratio", "action"]].corr().iloc[0, 1]
    print("\n=== 무리별 집계 (sharp_ratio = 마지막 프레임 선명도 유지율, 높을수록 덜 뭉개짐) ===")
    print(agg.to_string(index=False))
    print(f"\n선명도 유지율 ↔ Action 오차 상관: {rho:+.3f} (음수면 덜 뭉갤수록 점수 좋음)")
    a_r = agg.loc[agg["bin"] == "A", "sharp_ratio"]
    b_r = agg.loc[agg["bin"] == "B", "sharp_ratio"]
    if len(a_r) and len(b_r):
        ar, br = float(a_r.iloc[0]), float(b_r.iloc[0])
        print(f"판정: A-닮음 유지율 {ar:.3f} vs B-닮음 {br:.3f} "
              f"→ B가 {'더' if br < ar else '덜'} 뭉개짐. "
              "(eval 66.6%/40.5%와 같은 방향이면 '세팅 자체가 원인' 쪽 근거)")
    print("→ results/viewpoint_sharpness_summary.csv")


if __name__ == "__main__":
    main()
