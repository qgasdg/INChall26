"""EDA 검증 #1: 노션 데이터 분석 리포트 v2의 수치를 실제 데이터로 검증.

검증 대상 (docs/03_데이터분석.md):
- 데이터셋 수 128, 에피소드 11,132, 프레임 1,025,666
- fps 6(127) / 10(1), 해상도 분포, 로봇 타입, 카메라 키 1개
- eval: 216샘플, 640x480 PNG, (16,6) float32 deg npy
- so100_action_statistics.json count ~974,661
"""
import json, glob, os, sys
from collections import Counter
import numpy as np
from PIL import Image

ROOT = "/Users/yyt/Workspace/INChall26/open"
TRAIN = os.path.join(ROOT, "data/train")

# ---------- 1. train 데이터셋 전수 집계 ----------
datasets = []  # (user/name, info)
for info_path in sorted(glob.glob(os.path.join(TRAIN, "*", "*", "meta", "info.json"))):
    rel = os.path.relpath(info_path, TRAIN)
    ds_id = "/".join(rel.split(os.sep)[:2])
    with open(info_path) as f:
        info = json.load(f)
    datasets.append((ds_id, info))

print(f"[1] 데이터셋 수: {len(datasets)}  (노션: 128)")

total_ep = sum(i.get("total_episodes", 0) for _, i in datasets)
total_fr = sum(i.get("total_frames", 0) for _, i in datasets)
print(f"    총 에피소드: {total_ep:,}  (노션: 11,132)")
print(f"    총 프레임:   {total_fr:,}  (노션: 1,025,666)")

fps_c = Counter(i.get("fps") for _, i in datasets)
robot_c = Counter(i.get("robot_type") for _, i in datasets)
print(f"    fps 분포: {dict(fps_c)}  (노션: 6fps 127 / 10fps 1)")
print(f"    로봇: {dict(robot_c)}  (노션: so100 125, blue 2, red 1)")

# 해상도 + 카메라 키
res_c, ncam_c, cam_keys = Counter(), Counter(), Counter()
extra_keys = Counter()
for ds_id, i in datasets:
    feats = i.get("features", {})
    cams = [k for k, v in feats.items() if v.get("dtype") == "video"]
    ncam_c[len(cams)] += 1
    for c in cams:
        cam_keys[c] += 1
        sh = feats[c].get("shape")
        res_c[tuple(sh[:2])] += 1
    if "original_fps" in i or "downsample_stride" in i:
        extra_keys[(i.get("original_fps"), i.get("downsample_stride"))] += 1
print(f"    해상도: {dict(res_c)}  (노션: 480x640 123, 1080x1920 3, 720x1280 2)")
print(f"    데이터셋당 카메라 수: {dict(ncam_c)}  (노션: 1개)")
print(f"    카메라 키 상위: {cam_keys.most_common(8)}")
print(f"    (original_fps, stride) 분포: {dict(extra_keys)}")

# 에피소드 길이 분포 (episodes.jsonl 전수)
ep_lens = []
for ep_path in glob.glob(os.path.join(TRAIN, "*", "*", "meta", "episodes.jsonl")):
    with open(ep_path) as f:
        for line in f:
            try:
                ep_lens.append(json.loads(line).get("length"))
            except json.JSONDecodeError:
                pass
ep_lens = np.array([l for l in ep_lens if l is not None])
print(f"    에피소드 길이(전수 {len(ep_lens):,}개): min {ep_lens.min()}, median {np.median(ep_lens):.0f}, "
      f"max {ep_lens.max()}  (노션: 17~1,072, 중앙값 ~104 — 단 노션은 첫 에피소드 기준)")

# ---------- 2. eval 검증 ----------
img_paths = sorted(glob.glob(os.path.join(ROOT, "data/eval/images/*.png")))
act_paths = sorted(glob.glob(os.path.join(ROOT, "data/eval/actions/*.npy")))
print(f"\n[2] eval: 이미지 {len(img_paths)} / 액션 {len(act_paths)}  (노션: 216/216)")

sizes = Counter(Image.open(p).size for p in img_paths)
print(f"    이미지 크기(W,H): {dict(sizes)}  (노션: 640x480)")

shapes, dtypes = Counter(), Counter()
all_a = []
for p in act_paths:
    a = np.load(p)
    shapes[a.shape] += 1; dtypes[str(a.dtype)] += 1
    all_a.append(a)
all_a = np.stack(all_a)  # (216,16,6)
print(f"    action shape: {dict(shapes)}, dtype: {dict(dtypes)}  (노션: (16,6) float32)")
print(f"    action 값 범위(차원별 min~max, deg 추정):")
names = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
for d in range(6):
    v = all_a[..., d]
    print(f"      {names[d]:14s} {v.min():8.2f} ~ {v.max():8.2f}  (mean {v.mean():7.2f}, std {v.std():6.2f})")

# ---------- 3. action 정규화 통계 ----------
stat_path = os.path.join(TRAIN, "so100_action_statistics.json")
if os.path.exists(stat_path):
    st = json.load(open(stat_path))
    print(f"\n[3] so100_action_statistics.json 키: {list(st.keys())}")
    def flat(x): return np.array(x).ravel()
    for k in st:
        v = st[k]
        if isinstance(v, (int, float)):
            print(f"    {k}: {v:,}  (노션 count: 974,661)")
        else:
            print(f"    {k}: {np.round(flat(v), 2)}")
else:
    print("\n[3] so100_action_statistics.json 없음! 경로 확인 필요")
    print("    후보:", glob.glob(os.path.join(TRAIN, "*.json")))
