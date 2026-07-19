# E1 inference sweep: ddim_steps 50/30/20/12 x eval 216 on T4.
# Per config: generate videos (per-sample timing) -> kit CSV -> Action MAE vs GT.
# Optional probe: step 20 guidance_rescale 0.0 vs 0.7 (no-op check, CFG scale=1.0).
# Outputs (in /kaggle/working): features_s*.csv, actions_s*.csv, timing_s*.json,
# sweep_summary.csv, rescale_probe.json. Work tree + videos live in /tmp (not retrieved).
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
W = "/kaggle/working"
BASE = "/tmp/e1"
STEPS = [50, 30, 20, 12]

# torch>=2.6 defaults weights_only=True; lvdm/backbone ckpt loads predate that.
# All checkpoints here are trusted (organizers / DynamiCrafter / laion).
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def log(*a):
    print(f"[e1-sweep {time.time() - T0:7.0f}s]", *a, flush=True)


def run(cmd, **kw):
    log("+", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, **kw)


# ---------- 1. pip (recipe from inchall26-static-csv: numpy<2 forced last) ----------
pip = [sys.executable, "-m", "pip", "install", "-q"]

r = subprocess.run([sys.executable, "-c", "import pkg_resources"], capture_output=True)
if r.returncode != 0:
    run(pip + ["setuptools<81"])

need = []
for mod, spec in [
    ("timm", "timm>=0.9,<2"),
    ("pytorch_lightning", "pytorch-lightning==1.9.3"),
    ("einops", "einops"),
    ("av", "av>=10"),
    ("imageio", "imageio>=2.9.0,<3"),
    ("omegaconf", "omegaconf>=2.1"),
    ("open_clip", "open_clip_torch==2.22.0"),
    ("kornia", "kornia>=0.6,<1"),
]:
    r = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True)
    if r.returncode != 0:
        need.append(spec)
log("missing packages:", need)
if need:
    run(pip + need)

run(pip + ["--force-reinstall", "numpy>=1.26.4,<2"])
run([
    sys.executable, "-c",
    "import numpy, torch, torchvision, timm, av, imageio, cv2, transformers, "
    "pytorch_lightning as pl, omegaconf, open_clip, kornia, einops; "
    "assert numpy.__version__.startswith('1.'), numpy.__version__; "
    "print('numpy', numpy.__version__, '| torch', torch.__version__, '| pl', pl.__version__, "
    "'| omegaconf', omegaconf.__version__)",
])
run([sys.executable, "-c",
     "import torch; p = torch.cuda.get_device_properties(0); "
     "print('GPU:', p.name, round(p.total_memory / 2**30), 'GB')"])

# ---------- 2. locate inputs (recursive glob, mount layout varies) ----------
def find_dir(marker):
    cands = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    if not cands:
        raise SystemExit(f"{marker} not found under /kaggle/input")
    return cands[0]

kit_src = find_dir("submission_kit")            # static-bundle
ck_src = find_dir("challenge_kit")              # e1-bundle
log("submission_kit at", kit_src, "| challenge_kit at", ck_src)
static_root = os.path.dirname(kit_src)          # has submission_kit + data/eval
e1_baseline = os.path.dirname(ck_src)           # has challenge_kit, shared_libs, checkpoints
e1_root = os.path.dirname(e1_baseline)

# repo-relative layout under BASE so all config default paths resolve
os.makedirs(BASE, exist_ok=True)
shutil.copytree(kit_src, f"{BASE}/submission_kit")
shutil.copytree(os.path.join(static_root, "data"), f"{BASE}/data")
shutil.copytree(e1_baseline, f"{BASE}/baseline")
os.makedirs(f"{BASE}/data/train", exist_ok=True)
shutil.copy(os.path.join(e1_root, "data", "train", "so100_action_statistics.json"),
            f"{BASE}/data/train/so100_action_statistics.json")

# hf_hub 1.x compat: HfFolder/top-level helpers removed; only used by the
# remote-dataset path which we never take (data root is local). Patch the copy only.
_p = f"{BASE}/baseline/challenge_kit/src/ldwma/datasets/lerobot_so100.py"
_s = open(_p).read()
_old = "from huggingface_hub import HfFolder, hf_hub_download, hf_hub_url, list_repo_files"
_new = (
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
assert _old in _s
open(_p, "w").write(_s.replace(_old, _new))
log("patched lerobot_so100.py hf_hub imports (remote path only; local behavior unchanged)")

run(pip + ["--no-deps", "-e", f"{BASE}/baseline/challenge_kit",
           "-e", f"{BASE}/baseline/challenge_kit/libs/dynamicrafter",
           "-e", f"{BASE}/baseline/shared_libs/video_utils"])

# ---------- 3. backbone.ckpt (DynamiCrafter_512) ----------
from huggingface_hub import hf_hub_download  # noqa: E402

bb = os.path.realpath(hf_hub_download("Doubiiu/DynamiCrafter_512", "model.ckpt"))
dst = f"{BASE}/baseline/checkpoints/backbone.ckpt"
try:
    os.link(bb, dst)
except OSError:
    shutil.copy(bb, dst)
log("backbone.ckpt ready:", os.path.getsize(dst) // 2**20, "MB")

CK = f"{BASE}/baseline/challenge_kit"
N_EVAL = len(glob.glob(f"{BASE}/data/eval/images/*.png"))
log("eval samples:", N_EVAL)


# ---------- 4. helpers ----------
def generate(tag, ddim_steps, config="configs/eval/inha_submission_eval_11M.yaml"):
    """Run generation, timestamp each per-sample progress line. Returns timing dict."""
    pred = f"/tmp/pred_{tag}"
    os.makedirs(pred, exist_ok=True)
    cmd = [sys.executable, "scripts/inference/generate_baseline_videos.py",
           "--config", config, "--prediction-root", pred,
           "--ddim-steps", str(ddim_steps), "--seed", "0"]
    log("+", " ".join(cmd), f"(cwd={CK})")
    t_start = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    stderr_path = f"/tmp/gen_{tag}.stderr"
    with open(stderr_path, "w") as errf:
        proc = subprocess.Popen(cmd, cwd=CK, env=env, stdout=subprocess.PIPE,
                                stderr=errf, text=True, bufsize=1)
        line_times = []
        passthrough = 0
        for line in proc.stdout:
            if "[generate] wrote predictions" in line:
                line_times.append(time.time())
                n = len(line_times)
                if n % 40 == 0 or n == N_EVAL:
                    log(f"  {tag}: {n} samples, {time.time() - t_start:.0f}s elapsed")
            elif line.strip() and passthrough < 100:
                passthrough += 1
                print("   |", line.rstrip()[:200], flush=True)
        proc.wait()
    if proc.returncode != 0:
        print(open(stderr_path).read()[-4000:], flush=True)
        raise SystemExit(f"generation {tag} failed rc={proc.returncode}")
    total = time.time() - t_start
    deltas = [b - a for a, b in zip(line_times, line_times[1:])]
    deltas_sorted = sorted(deltas)
    med = deltas_sorted[len(deltas_sorted) // 2] if deltas else None
    timing = {
        "tag": tag, "ddim_steps": ddim_steps, "n": len(line_times),
        "total_s": round(total, 1),
        "per_sample_median_s": round(med, 3) if med else None,
        "per_sample_mean_tail_s": round(sum(deltas) / len(deltas), 3) if deltas else None,
        "first_sample_incl_load_s": round(line_times[0] - t_start, 1) if line_times else None,
    }
    with open(f"{W}/timing_{tag}.json", "w") as f:
        json.dump({**timing, "line_offsets_s": [round(t - t_start, 2) for t in line_times]}, f)
    log("timing:", timing)
    return pred, timing


def kit_score(tag, pred):
    """Kit CSV on generated videos -> per-sample Action MAE. Returns mean."""
    out_csv = f"{W}/features_{tag}.csv"
    t = time.time()
    run([sys.executable, "make_submission_csv.py",
         "--prediction-root", pred, "--output-csv", out_csv], cwd=f"{BASE}/submission_kit")
    maes = {}
    with open(out_csv) as f:
        for row in csv.DictReader(f):
            if row["feature_component"] == "Action Component":
                v = json.loads(row["feature_json"])
                while isinstance(v, list):
                    v = v[0]
                maes[row["sample_id"]] = float(v)
    with open(f"{W}/actions_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "mae"])
        for sid in sorted(maes):
            w.writerow([sid, maes[sid]])
    mean = sum(maes.values()) / len(maes)
    log(f"kit score {tag}: n={len(maes)} action_mae={mean:.4f} ({time.time() - t:.0f}s)")
    return mean, len(maes)


def md5s(pred):
    return {os.path.basename(p): hashlib.md5(open(p, "rb").read()).hexdigest()
            for p in sorted(glob.glob(f"{pred}/*.mp4"))}


# ---------- 5. sweep ----------
summary = []
hash_s20 = None
for k in STEPS:
    tag = f"s{k}"
    pred, timing = generate(tag, k)
    if k == 20:
        hash_s20 = md5s(pred)
    mae, n = kit_score(tag, pred)
    summary.append({"config": tag, "ddim_steps": k, "n": n,
                    "action_mae": round(mae, 6),
                    "gen_total_s": timing["total_s"],
                    "per_sample_median_s": timing["per_sample_median_s"]})
    with open(f"{W}/sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    shutil.rmtree(pred)
    log("summary so far:", json.dumps(summary))

# ---------- 6. guidance_rescale probe (step 20, 0.7 vs 0.0; CFG scale 1.0 -> expect no-op) ----------
try:
    base_cfg = open(f"{CK}/configs/eval/inha_submission_eval_11M.yaml").read()
    assert "guidance_rescale: 0.7" in base_cfg
    with open(f"{CK}/configs/eval/e1_rescale0.yaml", "w") as f:
        f.write(base_cfg.replace("guidance_rescale: 0.7", "guidance_rescale: 0.0"))
    pred, timing = generate("s20r0", 20, config="configs/eval/e1_rescale0.yaml")
    hash_r0 = md5s(pred)
    identical = hash_s20 == hash_r0
    probe = {"identical_md5": identical,
             "n_match": sum(1 for k2 in hash_s20 if hash_r0.get(k2) == hash_s20[k2]),
             "n": len(hash_s20 or {})}
    if not identical:
        mae, n = kit_score("s20r0", pred)
        probe["action_mae"] = round(mae, 6)
    with open(f"{W}/rescale_probe.json", "w") as f:
        json.dump(probe, f)
    log("rescale probe:", probe)
    shutil.rmtree(pred)
except Exception as e:  # probe is optional — never kill the sweep results
    log("rescale probe FAILED (non-fatal):", repr(e))

log("DONE. summary:", json.dumps(summary))
