"""Pick, per RUN_ID and C7 label, the F2 COPY whose sha256 equals the run's artifact_manifest.json entry.

Reads only: the two agreeing artifact_manifest.json copies (parsed as JSON data) and the copies' bytes (hashed).
Executes nothing it reads; writes one plan JSON to the path given.
"""
import hashlib
import json
import sys
from pathlib import Path

STORE = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
RUNS = ["stance6_r10LTSB_s60", "stance6_r10LTSB_s42", "stance6_r10LTSB_s80"]
LABELS = ["4194304", "5242880", "6291456", "8388608", "10485760", "final"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest(run):
    c = STORE / run / "copies"
    a, b = (c / "artifact_manifest.json.copy1").read_bytes(), (c / "artifact_manifest.json.copy2").read_bytes()
    if a != b:
        raise SystemExit(f"{run}: manifest copies 1 and 2 differ")
    m = json.loads(a)
    out = {}
    for e in m["files"]:
        s = e["sha256"]
        out[e["path"]] = (s.split(":", 1)[1] if s.startswith("sha256:") else s, e["size_bytes"])
    return out


def pick(run, rel, man):
    want, size = man[rel]
    cands = sorted((STORE / run / "copies").glob(rel + ".copy*"))
    rows = []
    for p in cands:
        h = sha(p)
        rows.append({"copy": str(p), "sha256": h, "size": p.stat().st_size, "matches": h == want and p.stat().st_size == size})
    ok = [r for r in rows if r["matches"]]
    return {"path": rel, "manifest_sha256": want, "manifest_size": size, "copies": rows,
            "chosen": ok[0]["copy"] if ok else None}


def main(out):
    plan = {}
    for run in RUNS:
        man = manifest(run)
        sc = pick(run, "01_stance/stage_config.json", man)
        items = {}
        for lab in LABELS:
            if lab == "final":
                z, p = "01_stance/models/stage1_final.zip", "01_stance/models/stage1_final_vecnorm.pkl"
            else:
                z, p = f"01_stance/models/stage1_{lab}_steps.zip", f"01_stance/models/stage1_vecnormalize_{lab}_steps.pkl"
            items[lab] = {"zip": pick(run, z, man), "pkl": pick(run, p, man)}
        plan[run] = {"stage_config": sc, "labels": items}
    Path(out).write_text(json.dumps(plan, indent=1))
    for run, d in plan.items():
        print(run, "stage_config", d["stage_config"]["chosen"] is not None)
        for lab, it in d["labels"].items():
            print(" ", lab, "zip", it["zip"]["chosen"] is not None, len(it["zip"]["copies"]),
                  "pkl", it["pkl"]["chosen"] is not None, len(it["pkl"]["copies"]))


if __name__ == "__main__":
    main(sys.argv[1])
