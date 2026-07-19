"""train 128개 데이터셋 전수 카탈로그 (48h 계획 Phase 1).

데이터셋별로 메타(fps·해상도·코덱·카메라 키·액션 스키마) + 에피소드 길이 분포 +
액션 6차원 통계를 한 행으로 요약한다. 부산물로 조사② <16프레임 에피소드 전수 목록.

에피소드 길이는 meta/episodes.jsonl에서 읽으므로 parquet 없이도 빠르다.
액션 통계는 parquet를 읽어야 하므로 데이터셋 단위 프로세스 병렬 (기본 32 워커).

사용: uv run python scripts/eda03_train_catalog.py [--workers 32] [--no-actions]
산출: results/train_catalog.csv, results/train_short_episodes.csv,
      results/train_catalog_meta.json (전역 집계·이상치 요약)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "open" / "data" / "train"
RESULTS = ROOT / "results"
WINDOW = 16  # 대회 규격 (조건 프레임 포함 16프레임)


def dataset_dirs() -> list[Path]:
    return sorted(p for p in TRAIN.glob("*/*") if (p / "meta" / "info.json").exists())


def summarize(args: tuple[str, bool]) -> dict:
    rel, with_actions = args
    d = TRAIN / rel
    info = json.loads((d / "meta" / "info.json").read_text())
    feats = info["features"]

    video_keys = [k for k, v in feats.items() if v.get("dtype") == "video"]
    vk = video_keys[0] if video_keys else None
    vinfo = feats[vk]["info"] if vk else {}

    eps = [json.loads(l) for l in (d / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    lengths = np.array([e["length"] for e in eps], dtype=int)
    short = [(rel, e["episode_index"], e["length"]) for e in eps if e["length"] < WINDOW]
    tasks = Counter(t for e in eps for t in e.get("tasks", []))

    row = {
        "dataset": rel,
        "n_episodes": len(eps),
        "n_frames": int(lengths.sum()),
        "total_episodes_meta": info.get("total_episodes"),
        "total_frames_meta": info.get("total_frames"),
        "fps": info.get("fps"),
        "original_fps": info.get("original_fps"),
        "downsample_stride": info.get("downsample_stride"),
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "n_video_keys": len(video_keys),
        "camera_key": vk,
        "height": vinfo.get("video.height"),
        "width": vinfo.get("video.width"),
        "codec": vinfo.get("video.codec"),
        "pix_fmt": vinfo.get("video.pix_fmt"),
        "len_min": int(lengths.min()) if len(lengths) else None,
        "len_p25": float(np.percentile(lengths, 25)) if len(lengths) else None,
        "len_median": float(np.median(lengths)) if len(lengths) else None,
        "len_p75": float(np.percentile(lengths, 75)) if len(lengths) else None,
        "len_max": int(lengths.max()) if len(lengths) else None,
        "len_mean": round(float(lengths.mean()), 2) if len(lengths) else None,
        "n_short_lt16": len(short),
        "n_windows_lt16": int((lengths < WINDOW).sum()),
        "n_tasks": len(tasks),
        "task_sample": next(iter(tasks), "")[:120],
        "action_dim": feats.get("action", {}).get("shape", [None])[0],
        "action_names": "|".join(feats.get("action", {}).get("names", []) or []),
        "state_dim": feats.get("observation.state", {}).get("shape", [None])[0],
    }

    if with_actions:
        parquets = sorted((d / "data" / "chunk-000").glob("*.parquet"))
        acts = []
        for p in parquets:
            try:
                acts.append(np.stack(pd.read_parquet(p, columns=["action"])["action"].to_numpy()))
            except Exception as e:  # 손상 파일은 건너뛰되 기록
                row["action_read_error"] = f"{p.name}: {type(e).__name__}"
        if acts:
            a = np.concatenate(acts).astype(np.float64)
            row["action_rows"] = int(a.shape[0])
            row["action_nan"] = int(np.isnan(a).sum())
            names = feats["action"]["names"]
            for i, nm in enumerate(names):
                col = a[:, i]
                row[f"a_{nm}_mean"] = round(float(col.mean()), 3)
                row[f"a_{nm}_std"] = round(float(col.std()), 3)
                row[f"a_{nm}_min"] = round(float(col.min()), 2)
                row[f"a_{nm}_max"] = round(float(col.max()), 2)
    return {"row": row, "short": short}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-actions", action="store_true", help="parquet 액션 통계 생략(빠름)")
    args = ap.parse_args()

    dirs = [str(p.relative_to(TRAIN)) for p in dataset_dirs()]
    print(f"데이터셋 {len(dirs)}개, 워커 {args.workers}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(summarize, [(d, not args.no_actions) for d in dirs]), 1):
            results.append(r)
            if i % 20 == 0 or i == len(dirs):
                print(f"  {i}/{len(dirs)}", flush=True)

    rows = [r["row"] for r in results]
    df = pd.DataFrame(rows).sort_values("dataset")
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "train_catalog.csv", index=False)

    shorts = [s for r in results for s in r["short"]]
    pd.DataFrame(shorts, columns=["dataset", "episode_index", "length"]).to_csv(
        RESULTS / "train_short_episodes.csv", index=False)

    # 전역 집계 + 이상치 (문서화 대상)
    meta = {
        "n_datasets": len(df),
        "n_episodes": int(df["n_episodes"].sum()),
        "n_frames": int(df["n_frames"].sum()),
        "short_episodes_lt16": len(shorts),
        "short_episode_share_pct": round(100 * len(shorts) / max(int(df["n_episodes"].sum()), 1), 3),
        "fps_values": df["fps"].value_counts().to_dict(),
        "original_fps_values": df["original_fps"].value_counts(dropna=False).to_dict(),
        "downsample_stride_values": df["downsample_stride"].value_counts(dropna=False).to_dict(),
        "resolutions": (df["width"].astype("Int64").astype(str) + "x"
                        + df["height"].astype("Int64").astype(str)).value_counts().to_dict(),
        "codecs": df["codec"].value_counts(dropna=False).to_dict(),
        "camera_keys": df["camera_key"].value_counts().to_dict(),
        "multi_camera_datasets": df.loc[df["n_video_keys"] != 1, ["dataset", "n_video_keys"]]
                                  .to_dict("records"),
        "action_dims": df["action_dim"].value_counts(dropna=False).to_dict(),
        "robot_types": df["robot_type"].value_counts(dropna=False).to_dict(),
        "meta_count_mismatch": df.loc[
            (df["n_episodes"] != df["total_episodes_meta"])
            | (df["n_frames"] != df["total_frames_meta"]),
            ["dataset", "n_episodes", "total_episodes_meta", "n_frames", "total_frames_meta"],
        ].to_dict("records"),
    }
    if "action_nan" in df:
        meta["datasets_with_action_nan"] = df.loc[df["action_nan"] > 0,
                                                  ["dataset", "action_nan"]].to_dict("records")
    (RESULTS / "train_catalog_meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False, default=str))

    print(f"\n데이터셋 {meta['n_datasets']} / 에피소드 {meta['n_episodes']:,} / 프레임 {meta['n_frames']:,}")
    print(f"<16프레임 에피소드 {meta['short_episodes_lt16']}개 ({meta['short_episode_share_pct']}%)")
    print(f"해상도 {meta['resolutions']}")
    print(f"fps {meta['fps_values']} / 코덱 {meta['codecs']}")
    print(f"카메라 키 종류 {len(meta['camera_keys'])}: {meta['camera_keys']}")
    print(f"메타 개수 불일치 데이터셋 {len(meta['meta_count_mismatch'])}개")
    print(f"→ {RESULTS}/train_catalog.csv, train_short_episodes.csv, train_catalog_meta.json")


if __name__ == "__main__":
    main()
