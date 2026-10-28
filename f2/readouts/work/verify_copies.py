# Pick, for each run and each run-relative path, the F2 copy whose sha256 equals the run's artifact_manifest.json entry.
import hashlib, json, sys, pathlib
F2 = pathlib.Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
out = {}
for rid in ["stance6_r10LTSB_s60", "stance6_r10LTSB_s42", "stance6_r10LTSB_s80"]:
    cp = F2 / rid / "copies"
    m1 = (cp / "artifact_manifest.json.copy1").read_bytes(); m2 = (cp / "artifact_manifest.json.copy2").read_bytes()
    assert hashlib.sha256(m1).hexdigest() == hashlib.sha256(m2).hexdigest()
    man = {f["path"]: f["sha256"].split(":")[-1] for f in json.loads(m1)["files"]}
    res = {}
    for f in sorted(cp.rglob("*.copy*")):
        rel = str(f.relative_to(cp)); path, k = rel.rsplit(".copy", 1)
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        ent = man.get(path)
        st = "MATCH" if ent == h else ("NO-ENTRY" if ent is None else "MISMATCH")
        res.setdefault(path, []).append(dict(copy=int(k), file=str(f), sha256=h, status=st))
    chosen = {}
    for p, lst in res.items():
        ok = [x for x in lst if x["status"] == "MATCH"]
        chosen[p] = dict(verified=ok[0]["file"] if ok else None, sha256=(ok[0]["sha256"] if ok else None),
                         copies=[(x["copy"], x["status"], x["sha256"][:12]) for x in lst])
    out[rid] = dict(manifest_sha256=hashlib.sha256(m1).hexdigest(), n_manifest_entries=len(man), paths=chosen)
json.dump(out, open(sys.argv[1], "w"), indent=1)
for rid, d in out.items():
    bad = [p for p, v in d["paths"].items() if v["verified"] is None]
    print(rid, "paths", len(d["paths"]), "unverified:", bad)
