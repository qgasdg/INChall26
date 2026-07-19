# E1 holdout-v2 re-anchor: v2 unseen 162 -> static + baseline(step 50) D+V+Action on T4.
# Order: pip -> layout -> 4-sample static smoke (validates scorer early) ->
#        baseline generation (162, step 50, seed 0) -> full static scoring -> baseline scoring.
# Outputs (in /kaggle/working): mini_static.csv, static_v2_unseen.csv,
# baseline_s50_v2_unseen.csv, timing_v2_s50.json, reanchor_summary.json.
# Work tree + videos live in /tmp (not retrieved).
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
W = "/kaggle/working"
BASE = "/tmp/e1"
PROJ = f"{BASE}/proj"

# torch>=2.6 defaults weights_only=True; lvdm/backbone ckpt loads predate that.
# All checkpoints here are trusted (organizers / DynamiCrafter / laion).
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def log(*a):
    print(f"[e1-reanchor {time.time() - T0:7.0f}s]", *a, flush=True)


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
    ("pandas", "pandas>=1.5"),
    ("pyarrow", "pyarrow"),
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
    "import numpy, torch, torchvision, timm, av, imageio, pandas, pyarrow, "
    "pytorch_lightning as pl, omegaconf, open_clip, kornia, einops; "
    "assert numpy.__version__.startswith('1.'), numpy.__version__; "
    "print('numpy', numpy.__version__, '| torch', torch.__version__, '| pl', pl.__version__)",
])
run([sys.executable, "-c",
     "import torch; p = torch.cuda.get_device_properties(0); "
     "print('GPU:', p.name, round(p.total_memory / 2**30), 'GB')"])

# ---------- 2. locate inputs ----------
def find_dir(marker):
    cands = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    if not cands:
        raise SystemExit(f"{marker} not found under /kaggle/input")
    return cands[0]

kit_src = find_dir("submission_kit")            # static-bundle
ck_src = find_dir("challenge_kit")              # e1-bundle
e1_baseline = os.path.dirname(ck_src)
e1_root = os.path.dirname(e1_baseline)
challenge_v2 = os.path.join(e1_root, "challenge_v2")
assert os.path.isdir(challenge_v2), challenge_v2
N_V2 = len(glob.glob(f"{challenge_v2}/images/*.png"))
log("challenge_v2 at", challenge_v2, "| n =", N_V2)

# generation tree (repo-relative defaults; challenge root passed explicitly)
os.makedirs(BASE, exist_ok=True)
shutil.copytree(e1_baseline, f"{BASE}/baseline")
os.makedirs(f"{BASE}/data/train", exist_ok=True)
shutil.copy(os.path.join(e1_root, "data", "train", "so100_action_statistics.json"),
            f"{BASE}/data/train/so100_action_statistics.json")

# scorer tree: local_eval expects ../open/{submission_kit, data/train}
os.makedirs(f"{PROJ}/open/data/train", exist_ok=True)
shutil.copytree(os.path.join(e1_root, "local_eval"), f"{PROJ}/local_eval")
shutil.copytree(kit_src, f"{PROJ}/open/submission_kit")
shutil.copy(os.path.join(e1_root, "data", "train", "so100_action_statistics.json"),
            f"{PROJ}/open/data/train/so100_action_statistics.json")
for user in os.listdir(os.path.join(e1_root, "train_v2_unseen")):
    shutil.copytree(os.path.join(e1_root, "train_v2_unseen", user),
                    f"{PROJ}/open/data/train/{user}")
log("train datasets:", os.listdir(f"{PROJ}/open/data/train"))

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


# ---------- 3. scorer smoke test (4 samples, static) — fail fast before GPU-heavy gen ----------
def score(args_list, out_csv):
    t = time.time()
    run([sys.executable, "local_eval/score_v2.py", *args_list, "--csv", out_csv], cwd=PROJ)
    rows = list(csv.DictReader(open(out_csv)))
    log(f"scored {len(rows)} rows -> {out_csv} ({time.time() - t:.0f}s)")
    return rows


score(["--static", "--holdout", "local_eval/holdout_mini.json", "--tiers", "unseen"],
      f"{W}/mini_static.csv")
log("SMOKE OK")

# ---------- 4. backbone + generation (162, step 50, seed 0) ----------
from huggingface_hub import hf_hub_download  # noqa: E402

bb = os.path.realpath(hf_hub_download("Doubiiu/DynamiCrafter_512", "model.ckpt"))
dst = f"{BASE}/baseline/checkpoints/backbone.ckpt"
try:
    os.link(bb, dst)
except OSError:
    shutil.copy(bb, dst)
log("backbone.ckpt ready:", os.path.getsize(dst) // 2**20, "MB")

CK = f"{BASE}/baseline/challenge_kit"
pred = "/tmp/pred_v2_s50"
os.makedirs(pred, exist_ok=True)
cmd = [sys.executable, "scripts/inference/generate_baseline_videos.py",
       "--challenge-root", challenge_v2, "--prediction-root", pred,
       "--action-stats-path", f"{BASE}/data/train/so100_action_statistics.json",
       "--ddim-steps", "50", "--seed", "0"]
log("+", " ".join(cmd), f"(cwd={CK})")
t_start = time.time()
env = {**os.environ, "PYTHONUNBUFFERED": "1"}
with open("/tmp/gen_v2.stderr", "w") as errf:
    proc = subprocess.Popen(cmd, cwd=CK, env=env, stdout=subprocess.PIPE,
                            stderr=errf, text=True, bufsize=1)
    line_times = []
    passthrough = 0
    for line in proc.stdout:
        if "[generate] wrote predictions" in line:
            line_times.append(time.time())
            if len(line_times) % 40 == 0 or len(line_times) == N_V2:
                log(f"  gen: {len(line_times)}/{N_V2}, {time.time() - t_start:.0f}s")
        elif line.strip() and passthrough < 100:
            passthrough += 1
            print("   |", line.rstrip()[:200], flush=True)
    proc.wait()
if proc.returncode != 0:
    print(open("/tmp/gen_v2.stderr").read()[-4000:], flush=True)
    raise SystemExit(f"generation failed rc={proc.returncode}")
deltas = [b - a for a, b in zip(line_times, line_times[1:])]
timing = {"n": len(line_times), "total_s": round(time.time() - t_start, 1),
          "per_sample_median_s": round(sorted(deltas)[len(deltas) // 2], 3) if deltas else None}
with open(f"{W}/timing_v2_s50.json", "w") as f:
    json.dump(timing, f)
log("gen timing:", timing)

# ---------- 5. full scoring: static + baseline ----------
static_rows = score(["--static", "--tiers", "unseen"], f"{W}/static_v2_unseen.csv")
base_rows = score(["--videos", pred, "--tiers", "unseen"], f"{W}/baseline_s50_v2_unseen.csv")


def agg(rows, tier=None):
    sel = [r for r in rows if tier is None or r["tier"] == tier]
    if not sel:
        return None
    m = {k: round(sum(float(r[k]) for r in sel) / len(sel), 6)
         for k in ("dino_pf", "dino_flat", "video", "action", "total_pf", "total_flat")}
    return {"n": len(sel), **m}


summary = {}
for name, rows in [("static", static_rows), ("baseline_s50", base_rows)]:
    summary[name] = {
        "unseen_all": agg(rows),
        "unseen_cousin": agg(rows, "unseen_cousin"),
        "unseen_general": agg(rows, "unseen_general"),
    }
summary["timing_v2_s50"] = timing
with open(f"{W}/reanchor_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
log("SUMMARY:", json.dumps(summary))
shutil.rmtree(pred)
log("DONE")
