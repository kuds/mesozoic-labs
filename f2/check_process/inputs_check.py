"""Read-only check (3): every F2 copy vs its run's artifact_manifest.json, and every measurement record's inputs.

argv: round_dir
Reads round/fetch/manifest.csv (F2 COPY rows), the F2 store's copies (sha256 only; JSON parse of the
manifest copies only; nothing else is opened as anything but bytes), and the panel summaries (JSON).
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROUND = Path(sys.argv[1])
RUNS = ["stance7_r10LTSBpc_s60", "stance7_r10LTSBoff_s60", "stance7_r10LTSBpc_s42", "stance7_r10LTSBoff_s42",
        "stance7_r10LTSBpc_s80", "stance7_r10LTSBoff_s80"]
J = ["4194304", "5242880", "6291456", "8388608", "10485760", "final"]
SEAL_TREE = "9417b7659076f415359e13c4730c5bf15e487392c67e26dd85d53e578dcc6580"
SEAL_ARCH = "babc6812fded51cb5b35ad37d7e8c103d0bfcaff9d1ce68a2757fbac11690dc5"
TOOL_SHA = {"measure_r10round.py": "75cb4a9b9028a4c4b2eb22b641627b1eac7bb01be59c90931af9d732511d04e5",
            "manip_check.py": "da8b4be6089923b5c5e540760003dd68c2205d0cb880707f22711d9beb6a8e22"}
APPROVED_SHA = "ce8865205a218541a742d7acf0cee479e87d63c7b53cbc52ae47a8892569873a"
_cache = {}


def sha(p):
    p = str(p)
    if p not in _cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        _cache[p] = h.hexdigest()
    return _cache[p]


problems = []
info = []
rows = [r for r in csv.DictReader(open(ROUND / "fetch" / "manifest.csv", newline="", encoding="utf-8"))
        if r["fetch"] == "F2" and r["status"] == "COPY"]
by_run = defaultdict(list)
for r in rows:
    by_run[r["run_id"]].append(r)

manifests = {}
for run in RUNS:
    base = ROUND / "fetch" / "F2" / run / "copies"
    c1, c2 = base / "artifact_manifest.json.copy1", base / "artifact_manifest.json.copy2"
    s1, s2 = sha(c1), sha(c2)
    if s1 != s2:
        problems.append(f"{run}: manifest copies differ {s1} {s2}")
    man = json.loads(c1.read_text(encoding="utf-8"))
    files = man.get("files") or man.get("entries") or man.get("artifacts")
    ent = {}
    if isinstance(files, list):
        for e in files:
            ent[e.get("path") or e.get("relpath")] = (str(e.get("sha256", "")).replace("sha256:", ""),
                                                     e.get("size_bytes"))
    elif isinstance(files, dict):
        for k, e in files.items():
            ent[k] = (str(e.get("sha256", "")).replace("sha256:", ""), e.get("size_bytes"))
    manifests[run] = ent
    info.append(f"{run}: manifest sha256 {s1} (2 copies equal: {s1 == s2}); keys {sorted(man)[:8]}; "
                f"{len(ent)} entries; status {man.get('status')}; run_id {man.get('run_id')}")
    if man.get("run_id") != run:
        problems.append(f"{run}: manifest run_id {man.get('run_id')}")

# every F2 COPY row: stored file present, its bytes' sha256 == the row's sha256 == the manifest entry
nrow = 0
copy_ok = defaultdict(int)
for run, rs in by_run.items():
    ent = manifests[run]
    for r in rs:
        nrow += 1
        p = Path(r["stored_as"])
        exp_prefix = ROUND / "fetch" / "F2" / run / "copies"
        if exp_prefix not in p.parents:
            problems.append(f"{run}: stored_as outside its F2 copies folder: {p}")
        if not p.is_file():
            problems.append(f"{run}: stored file missing {p}")
            continue
        s = sha(p)
        if s != r["sha256"]:
            problems.append(f"{run}: {p} bytes sha256 {s} != row {r['sha256']}")
        if r["path"] == "artifact_manifest.json":
            copy_ok[run] += 1
            continue
        if r["path"] not in ent:
            problems.append(f"{run}: {r['path']} not in its manifest")
            continue
        if ent[r["path"]][0] != s:
            problems.append(f"{run}: {r['path']} copy {r['copy']} sha256 {s} != manifest {ent[r['path']][0]}")
        else:
            copy_ok[run] += 1
info.append(f"F2 COPY rows checked: {nrow}; matching their manifest entry (manifest copies counted): {dict(copy_ok)}")

# store files not named by any COPY row (stray files in the copies folders)
for run in RUNS:
    base = ROUND / "fetch" / "F2" / run
    named = {r["stored_as"] for r in by_run[run]}
    for p in base.rglob("*"):
        if p.is_file() and str(p) not in named and "/copies/" in str(p):
            problems.append(f"{run}: file in copies/ not named by a COPY row: {p}")


def check_input(run, label, kind, path, recorded, what):
    p = Path(path)
    base = ROUND / "fetch" / "F2" / run / "copies"
    if base not in p.parents:
        problems.append(f"{run}/{label}/{kind}: {what} not an F2 copy of the run: {path}")
        return
    rel = str(p.relative_to(base))
    rel = rel.rsplit(".copy", 1)[0]
    s = sha(p)
    if s != recorded:
        problems.append(f"{run}/{label}/{kind}: {what} recorded {recorded} but bytes now {s}")
    m = manifests[run].get(rel)
    if m is None:
        problems.append(f"{run}/{label}/{kind}: {what} {rel} not in the manifest")
    elif m[0] != s:
        problems.append(f"{run}/{label}/{kind}: {what} {rel} sha256 {s} != manifest {m[0]}")
    return rel


def pair_of(label):
    if label == "final":
        return "01_stance/models/stage1_final.zip", "01_stance/models/stage1_final_vecnorm.pkl"
    return (f"01_stance/models/stage1_{label}_steps.zip", f"01_stance/models/stage1_vecnormalize_{label}_steps.pkl")


measured = defaultdict(lambda: defaultdict(list))
for run in RUNS:
    pdir = ROUND / "panels" / run
    for lab in sorted(x.name for x in pdir.iterdir()):
        for kind, fn in (("c8", "manip_check.json"), ("m1", "m1_summary.json"), ("m3", "m3_summary.json")):
            f = pdir / lab / kind / fn
            others = sorted(x.name for x in (pdir / lab / kind).glob("*refused*")) if (pdir / lab / kind).is_dir() else []
            if others:
                problems.append(f"{run}/{lab}/{kind}: refusal files {others}")
            if not f.is_file():
                if (pdir / lab / kind).exists():
                    problems.append(f"{run}/{lab}/{kind}: folder without {fn}: {sorted(x.name for x in (pdir/lab/kind).iterdir())}")
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            measured[run][lab].append(kind)
            ck = d.get("checkpoint", {})
            rr = d.get("run_records", {})
            fl = d.get("filed", {})
            zp, vp = pair_of(lab)
            rz = check_input(run, lab, kind, ck.get("path"), ck.get("sha256"), "checkpoint")
            rv = check_input(run, lab, kind, ck.get("vecnorm"), ck.get("vecnorm_sha256"), "vecnorm")
            rs = check_input(run, lab, kind, rr.get("stage_config"), rr.get("stage_config_sha256"), "stage_config")
            if rz != zp or rv != vp:
                problems.append(f"{run}/{lab}/{kind}: pair read {rz}, {rv} is not the label's pair {zp}, {vp}")
            if rs != "01_stance/stage_config.json":
                problems.append(f"{run}/{lab}/{kind}: stage_config read {rs}")
            if fl.get("run_id") != run or str(fl.get("label")) != lab:
                problems.append(f"{run}/{lab}/{kind}: filed as {fl.get('run_id')}/{fl.get('label')}")
            exp_ts = 12001280 if lab == "final" else int(lab)
            if lab != "final" and ck.get("num_timesteps") != exp_ts:
                problems.append(f"{run}/{lab}/{kind}: num_timesteps {ck.get('num_timesteps')}")
            seal = d.get("copy_seal", {})
            if seal.get("tree_digest") != SEAL_TREE or seal.get("archive_sha256") != SEAL_ARCH or \
                    d.get("copy_tree_digest_after") != SEAL_TREE:
                problems.append(f"{run}/{lab}/{kind}: seal {seal.get('tree_digest')} / after {d.get('copy_tree_digest_after')}")
            tool = Path(d.get("tool", "")).name
            if d.get("tool") not in (f"/home/user/work/pc_study/tools/pc/{tool}",) or TOOL_SHA.get(tool) != d.get("tool_sha256"):
                problems.append(f"{run}/{lab}/{kind}: tool {d.get('tool')} sha {d.get('tool_sha256')}")
            ap = d.get("copy_commit_basis", {}).get("approval", {})
            if ap.get("approved_sha256") != APPROVED_SHA:
                problems.append(f"{run}/{lab}/{kind}: approved_sha256 {ap.get('approved_sha256')}")
            if d.get("physics_revision") != 10:
                problems.append(f"{run}/{lab}/{kind}: physics_revision {d.get('physics_revision')}")
            info.append(f"{run}/{lab}/{kind}: utc {d.get('utc')} ts {ck.get('num_timesteps')} pair OK={rz == zp and rv == vp}")

for run in RUNS:
    labs = measured[run]
    got = {lab: sorted(k) for lab, k in labs.items()}
    info.append(f"{run}: measured labels {got}")
    for lab in J:
        need = ["c8", "m1"] + (["m3"] if lab == "final" else [])
        if sorted(labs.get(lab, [])) != sorted(need):
            problems.append(f"{run}: label {lab} measured {labs.get(lab)} (need {need})")
    for lab in labs:
        if lab not in J:
            problems.append(f"{run}: extra label measured {lab}")

print("\n".join(info))
print("PROBLEMS:", len(problems))
print("\n".join(problems))
