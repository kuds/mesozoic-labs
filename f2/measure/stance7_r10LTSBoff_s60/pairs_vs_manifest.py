import hashlib, json, sys
from pathlib import Path
C = Path("/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBoff_s60/copies")
man = json.loads((C / "artifact_manifest.json.copy1").read_text())
m2 = (C / "artifact_manifest.json.copy2").read_bytes() == (C / "artifact_manifest.json.copy1").read_bytes()
print("manifest copy1 == copy2:", m2, " top keys:", sorted(man.keys())[:20])
ents = man.get("files") or man.get("entries") or man.get("artifacts")
if isinstance(ents, dict):
    ents = [dict(path=k, **v) if isinstance(v, dict) else dict(path=k, sha256=v) for k, v in ents.items()]
idx = {}
for e in ents:
    p = e.get("path") or e.get("relpath") or e.get("name")
    idx[p] = e
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
labels = ["4194304", "5242880", "6291456", "8388608", "10485760", "final"]
rel = ["01_stance/stage_config.json"]
for L in labels:
    if L == "final":
        rel += ["01_stance/models/stage1_final.zip", "01_stance/models/stage1_final_vecnorm.pkl"]
    else:
        rel += [f"01_stance/models/stage1_{L}_steps.zip", f"01_stance/models/stage1_vecnormalize_{L}_steps.pkl"]
for r in rel:
    e = idx.get(r)
    want = (e or {}).get("sha256")
    if want and want.startswith("sha256:"):
        want = want[7:]
    for k in (1, 2):
        p = C / (r + f".copy{k}")
        if p.exists():
            got = sha(p)
            print(("MATCH   " if got == want else "MISMATCH"), p, got, "manifest", want, "size", p.stat().st_size, "manifest_size", (e or {}).get("size", (e or {}).get("bytes")))
