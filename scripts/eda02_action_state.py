"""EDA 검증 #2: action↔state 관계 + parquet 스키마 + 영상 메타 검증.

검증 대상 (docs/03_데이터분석.md §3, §4):
- parquet 스키마: action(6,), observation.state(6,), timestamp 1/6s, index 5간격
- action[t] vs state[t]/state[t+1]: corr 0.99+, |a[t]-s[t+1]|=1.65 < |a[t]-s[t]|=2.25
- 관절 커플링: shoulder_lift↔elbow_flex +0.89
- 영상: h264, 6fps, 16프레임 이상
"""
import glob, os, subprocess, json, random
import numpy as np
import pandas as pd

ROOT = "/Users/yyt/Workspace/INChall26/open"
TRAIN = os.path.join(ROOT, "data/train")
random.seed(0)

all_pq = sorted(glob.glob(os.path.join(TRAIN, "*", "*", "data", "chunk-000", "episode_000000.parquet")))
print(f"첫 에피소드 parquet 수: {len(all_pq)}")
sample = random.sample(all_pq, 30)

# ---------- 1. 스키마 확인 (첫 파일) ----------
df0 = pd.read_parquet(sample[0])
print(f"\n[1] 스키마 ({os.path.relpath(sample[0], TRAIN).split('/data/')[0]}):")
print("    columns:", list(df0.columns))
a0 = np.stack(df0["action"].to_numpy())
print("    action shape/dtype:", a0.shape, a0.dtype)
ts = df0["timestamp"].to_numpy()
print(f"    timestamp 간격: {np.diff(ts)[:5]} (1/6={1/6:.4f})")
if "index" in df0:
    print(f"    index 간격: {np.diff(df0['index'].to_numpy())[:5]} (노션: 5)")

# ---------- 2. action↔state (30개 표본) ----------
names = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
A, S = [], []
for p in sample:
    df = pd.read_parquet(p, columns=["action", "observation.state"])
    if len(df) < 2: continue
    A.append(np.stack(df["action"].to_numpy()))
    S.append(np.stack(df["observation.state"].to_numpy()))
n_frames = sum(len(a) for a in A)
print(f"\n[2] 표본: 30개 데이터셋 첫 에피소드, {n_frames:,}프레임  (노션: 4,408)")

same_t, next_t, corr_same, corr_next = [], [], [], []
for a, s in zip(A, S):
    same_t.append(np.abs(a - s))                    # |a[t]-s[t]|
    next_t.append(np.abs(a[:-1] - s[1:]))           # |a[t]-s[t+1]|
same_t = np.concatenate(same_t); next_t = np.concatenate(next_t)
print(f"    평균 |action[t]-state[t]|   = {same_t.mean():.2f}°  (노션: 2.25°)")
print(f"    평균 |action[t]-state[t+1]| = {next_t.mean():.2f}°  (노션: 1.65°)")

Acat = np.concatenate(A); Scat = np.concatenate(S)
Acat_n = np.concatenate([a[:-1] for a in A]); Snext = np.concatenate([s[1:] for s in S])
print("    관절별 corr(a[t],s[t]) / corr(a[t],s[t+1]) / MAE(t+1):")
for d in range(6):
    c1 = np.corrcoef(Acat[:, d], Scat[:, d])[0, 1]
    c2 = np.corrcoef(Acat_n[:, d], Snext[:, d])[0, 1]
    print(f"      {names[d]:14s} {c1:.4f} / {c2:.4f} / {next_t[:, d].mean():5.2f}°")

cc = np.corrcoef(Acat.T)
print(f"    shoulder_lift↔elbow_flex corr: {cc[1,2]:+.3f}  (노션: +0.89)")
print(f"    wrist_flex↔gripper corr:      {cc[3,5]:+.3f}  (노션: -0.33)")
print(f"    action 차원별 std: {np.round(Acat.std(axis=0),1)}  (노션: 14~62)")

# ---------- 3. 영상 메타 (ffprobe, 8개 표본) ----------
print("\n[3] 영상 메타 (8개 표본):")
vids = random.sample(sorted(glob.glob(os.path.join(TRAIN, "*", "*", "videos", "chunk-000", "*", "episode_000000.mp4"))), 8)
for v in vids:
    r = subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_streams","-select_streams","v:0", v],
                       capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    ds_id = os.path.relpath(v, TRAIN).split("/videos/")[0]
    print(f"    {ds_id:45s} {s['codec_name']} {s['width']}x{s['height']} "
          f"fps={s['avg_frame_rate']} frames={s.get('nb_frames','?')} pixfmt={s['pix_fmt']}")

# ---------- 4. 짧은 에피소드(<16프레임) 비율 — 윈도우 생성 불가 데이터 ----------
short = 0; total = 0
for ep_path in glob.glob(os.path.join(TRAIN, "*", "*", "meta", "episodes.jsonl")):
    with open(ep_path) as f:
        for line in f:
            try:
                l = json.loads(line).get("length")
                if l is not None:
                    total += 1
                    if l < 16: short += 1
            except json.JSONDecodeError:
                pass
print(f"\n[4] 16프레임 미만 에피소드: {short}/{total} ({100*short/total:.2f}%) — 학습 윈도우 생성 불가분")
