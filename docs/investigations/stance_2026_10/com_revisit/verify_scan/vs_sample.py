"""Re-run a sample of the scan's spawns with the verifier's own harness (vs_lib.py).  Resumable.
Usage: python vs_sample.py <scan.jsonl> <out.jsonl>
Sample per cell (model x noise): every scan-flagged spawn, plus 30 unflagged ones: the 6 largest by each of
com_speed_end_rms, foot_travel_after1s, load_sd_tail_bw_max, the 4 largest by com_travel_window_mm, then random
(numpy seed 12345) to 30.  Plus both statues, and robot controls at 0.02: the 10 known creepers (111, 143, 218, 249,
251, 271, 304, 330, 388, 390) and resting seeds 0-29, 102, 141.
"""
import json
import os
import sys
import time

import numpy as np

import vs_lib as L

scan_path, out_path = sys.argv[1], sys.argv[2]
rows = [json.loads(x) for x in open(scan_path)]
cells = [("compy_anat", 0.02), ("compy_anat", 0.01), ("velo", 0.02), ("velo", 0.05)]
rng = np.random.default_rng(12345)
plan = []
for model, noise in cells:
    rs = [r for r in rows if r["model"] == model and r["noise"] == noise]
    flagged = sorted(r["seed"] for r in rs if r["any_flag"])
    unf = [r for r in rs if not r["any_flag"]]
    pick = []
    for key, n in (("com_speed_end_rms_mps", 6), ("foot_travel_after1s_m", 6), ("load_sd_tail_bw_max", 6),
                   ("com_travel_window_mm", 4)):
        for r in sorted(unf, key=lambda r: -r[key])[:n]:
            if r["seed"] not in pick:
                pick.append(r["seed"])
    rest = [r["seed"] for r in unf if r["seed"] not in pick]
    pick += list(rng.choice(rest, size=30 - len(pick), replace=False))
    plan += [(model, int(s), noise, "flagged") for s in flagged]
    plan += [(model, int(s), noise, "unflagged") for s in pick]
plan += [("compy_anat", 0, None, "statue"), ("velo", 0, None, "statue")]
plan += [("robot", s, 0.02, "known_creeper") for s in (111, 143, 218, 249, 251, 271, 304, 330, 388, 390)]
plan += [("robot", s, 0.02, "resting") for s in list(range(30)) + [102, 141]]

done = set()
if os.path.exists(out_path):
    for x in open(out_path):
        r = json.loads(x)
        done.add((r["model"], r["seed"], r["noise"]))
out = open(out_path, "a")
t0 = time.time()
for i, (model, seed, noise, role) in enumerate(plan):
    if (model, seed, noise) in done:
        continue
    r = L.run(model, seed, noise)
    r["role"] = role
    out.write(json.dumps(r) + "\n")
    out.flush()
    if i % 20 == 0:
        print(f"{i}/{len(plan)} {time.time() - t0:.0f} s", flush=True)
print(f"done {len(plan)} in {time.time() - t0:.0f} s", flush=True)
