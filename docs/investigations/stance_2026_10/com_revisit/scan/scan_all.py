"""The scan: both models, the statue, the CoM tests' reset noise 0.02 on seeds 0-29, 100-149, 200-399, and each model's
training reset noise on seeds 0-199, neutral action, base plant.  Resumable (rows already in the output are skipped).
Any spawn with a flag gets its per-substep load and root vz saved to traces/ (npz).
Usage: python scan_all.py scan.jsonl [model ...]"""

import json
import os
import sys
import time

import numpy as np

import scanlib as S

out_path = sys.argv[1]
models = sys.argv[2:] or ["compy_anat", "velo"]
SEEDS_TEST = list(range(0, 30)) + list(range(100, 150)) + list(range(200, 400))
SEEDS_TRAIN = list(range(0, 200))

done = set()
if os.path.exists(out_path):
    for line in open(out_path):
        r = json.loads(line)
        done.add((r["model"], r["seed"], r["noise"]))
os.makedirs("traces", exist_ok=True)
out = open(out_path, "a")
for model in models:
    blocks = [("statue", None, [0]), ("test", 0.02, SEEDS_TEST), ("train", S.TRAIN_NOISE[model], SEEDS_TRAIN)]
    for name, noise, seeds in blocks:
        t0 = time.time()
        flagged, fell, nrun = [], [], 0
        for seed in seeds:
            if (model, seed, noise) in done:
                continue
            row, ep, metrics, rec = S.run(model, seed, noise, keep_recorder=True)
            row["block"] = name
            nrun += 1
            if not row["stood"]:
                fell.append(seed)
            elif row["any_flag"]:
                flagged.append(seed)
                t, load, other, vz, ncon = rec.arrays()
                np.savez_compressed(f"traces/{model}_n{noise}_s{seed}.npz", t=t, load=load, other=other, vz=vz,
                                    ncon=ncon)
            out.write(json.dumps(row) + "\n")
            out.flush()
        print(f"{model} {name} noise {noise}: ran {nrun} of {len(seeds)}; any flag {flagged}; fell {fell}; "
              f"{time.time() - t0:.0f} s", flush=True)
