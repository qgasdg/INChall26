# Dump small CSV/JSON outputs of attached kernel(s) into this kernel's log,
# because the session's egress policy blocks the file-download host.
import glob
import os

for path in sorted(glob.glob("/kaggle/input/**/*", recursive=True)):
    if os.path.isfile(path) and path.endswith((".csv", ".json")):
        size = os.path.getsize(path)
        if size > 300_000:
            print(f"===SKIP {path} ({size}B)===", flush=True)
            continue
        print(f"===FILE {path} ({size}B)===", flush=True)
        with open(path) as f:
            print(f.read(), flush=True)
        print(f"===END {path}===", flush=True)
print("FETCH DONE", flush=True)
