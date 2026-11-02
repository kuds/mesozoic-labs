# Own read-only check: sha256 of the F2 .copy files this measurement uses against the run's
# artifact_manifest.json entries (copy1 and copy2 of the manifest must be byte-equal). Hashes bytes only; loads nothing.
import hashlib, json, os, sys
S = "/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBoff_s80/copies"
m1 = open(os.path.join(S, "artifact_manifest.json.copy1"), "rb").read()
m2 = open(os.path.join(S, "artifact_manifest.json.copy2"), "rb").read()
print("manifest copy1 sha256", hashlib.sha256(m1).hexdigest(), "copy2 equal:", m1 == m2)
man = {e["path"]: e["sha256"].split(":", 1)[1] for e in json.loads(m1)["files"]}
labels = ["4194304", "5242880", "6291456", "8388608", "10485760", "final"]
paths = ["01_stance/stage_config.json"]
for L in labels:
    if L == "final":
        paths += ["01_stance/models/stage1_final.zip", "01_stance/models/stage1_final_vecnorm.pkl"]
    else:
        paths += ["01_stance/models/stage1_%s_steps.zip" % L, "01_stance/models/stage1_vecnormalize_%s_steps.pkl" % L]
nm = nx = 0
for p in paths:
    for k in (1, 2):
        f = os.path.join(S, p + ".copy%d" % k)
        if not os.path.exists(f):
            if k == 1:
                print("ABSENT  ", p + ".copy1"); nx += 1
            continue
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        ok = man.get(p) == h
        nm += ok; nx += (not ok)
        print("MATCH   " if ok else "MISMATCH", p + ".copy%d" % k, h, "manifest", man.get(p))
print("MATCH", nm, "MISMATCH/ABSENT", nx)
sys.exit(0 if nx == 0 else 1)
