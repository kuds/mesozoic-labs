"""The start pair and the parent records: sha256 of the F2 .copy1 files against the fetch manifest's COPY rows."""
import csv
import hashlib
import json
import sys

FETCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch"
RUN = "stance6_r10LTSB_s42"
RELS = ["01_stance/models/stage1_1048576_steps.zip", "01_stance/models/stage1_vecnormalize_1048576_steps.pkl",
        "01_stance/eval_episodes.csv", "01_stance/evaluations.npz", "01_stance/gate_progress.npz",
        "01_stance/train_episodes.csv", "01_stance/policy_probe.npz", "01_stance/stage_config.json"]
rows = list(csv.DictReader(open(f"{FETCH}/manifest.csv", newline="")))
out, ok = {}, True
for rel in RELS:
    path = f"{FETCH}/F2/{RUN}/copies/{rel}.copy1"
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    man = sorted({r["sha256"] for r in rows if r["fetch"] == "F2" and r["run_id"] == RUN and r["path"] == rel
                  and r["status"] == "COPY" and r["copy"] == "1"})
    eq = man == [got]
    ok &= eq
    out[rel] = {"sha256": got, "manifest_copy1": man, "equal": eq}
print(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
