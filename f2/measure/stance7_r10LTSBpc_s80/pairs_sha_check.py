# Compare sha256 of the F2 copies of the measured pairs and stage_config.json with artifact_manifest.json.copy1/copy2 entries.
import hashlib, json, sys, os
F = sys.argv[1]
C = os.path.join(F, "copies")
m1 = json.load(open(os.path.join(C, "artifact_manifest.json.copy1")))
m2b = open(os.path.join(C, "artifact_manifest.json.copy2"), "rb").read()
m1b = open(os.path.join(C, "artifact_manifest.json.copy1"), "rb").read()
print("manifest copy1 == copy2 bytes:", m1b == m2b, hashlib.sha256(m1b).hexdigest())
ent = {e["path"]: e["sha256"] for e in m1["files"]}
paths = ["01_stance/stage_config.json"]
for s in ["4194304", "5242880", "6291456", "8388608", "10485760"]:
    paths += ["01_stance/models/stage1_%s_steps.zip" % s, "01_stance/models/stage1_vecnormalize_%s_steps.pkl" % s]
paths += ["01_stance/models/stage1_final.zip", "01_stance/models/stage1_final_vecnorm.pkl"]
nm = nx = 0
for p in paths:
    for k in (1, 2):
        f = os.path.join(C, p + ".copy%d" % k)
        if not os.path.exists(f):
            continue
        h = "sha256:" + hashlib.sha256(open(f, "rb").read()).hexdigest()
        ok = ent.get(p) == h
        nm += ok; nx += (not ok)
        print("MATCH" if ok else "MISMATCH", p + ".copy%d" % k, h, "manifest", ent.get(p))
print("MATCH", nm, "MISMATCH", nx)
