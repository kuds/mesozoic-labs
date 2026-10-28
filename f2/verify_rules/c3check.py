import csv, hashlib, json, os, sys
S = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
fm = list(csv.DictReader(open(f"{S}/fetch/manifest.csv", newline="")))
C3 = ["provenance.json", "01_stance/stage_config.json", "01_stance/progress.json", "01_stance/eval_episodes.csv", "01_stance/gate_verdict.json"]
for st in ["4194304", "5242880", "6291456", "8388608", "10485760"]:
    C3 += [f"01_stance/models/stage1_{st}_steps.zip", f"01_stance/models/stage1_vecnormalize_{st}_steps.pkl"]
C3 += ["01_stance/models/stage1_final.zip", "01_stance/models/stage1_final_vecnorm.pkl"]
for r in ["s60", "s42", "s80"]:
    rid = f"stance6_r10LTSB_{r}"
    base = f"{S}/fetch/F2/{rid}/copies"
    man = json.load(open(f"{base}/artifact_manifest.json.copy1"))
    assert sha(f"{base}/artifact_manifest.json.copy1") == sha(f"{base}/artifact_manifest.json.copy2")
    mf = {e["path"]: e["sha256"].split(":", 1)[-1] for e in man["files"]}
    hw = "01_stance/hop_watch.json" in mf
    bad = []
    for p in C3:
        c = f"{base}/{p}.copy1"
        ok_copy = os.path.exists(c) and sha(c) == mf.get(p)
        rec = [x for x in fm if x["fetch"] == "F2" and x["run_id"] == rid and x["path"] == p and x["sha256"].split(":")[-1] == mf.get(p)]
        if not ok_copy or not rec:
            bad.append((p, ok_copy, len(rec)))
    print(r, "status", man.get("status"), "run_id", man.get("run_id"), "entries", len(man["files"]), "hop_watch listed", hw, "C3 bad", bad)
