"""Compare the stage records the e2e lane's set-up path wrote at 0e6a57b with the setup check's records at 0bcb0cf
(build/check/judge/fake_runs, read only): every JSON leaf that differs, by key path, per seed and file.
Scratch helper; reads JSON only.

usage: python -I -B cmp_records.py LANE_SETUP_DIR SETUP_CHECK_FAKE_RUNS OUT.json
"""

import json
import sys
from pathlib import Path


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


setup, fake, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
res = {}
for seed in (42, 52, 60):
    for name in ("stage_config.json", "plant_identity.json", "task_fingerprint.json"):
        a = json.loads((setup / f"s{seed}" / "stage" / name).read_text())
        bp = fake / f"pilot4_walk_s{seed}" / "03_locomotion" / name
        if not bp.is_file():
            res[f"s{seed}/{name}"] = "no setup-check record"
            continue
        b = json.loads(bp.read_text())
        la, lb = dict(leaves(a)), dict(leaves(b))
        diff = {k: {"e2e_0e6a57b": la.get(k, "<absent>"), "setup_check_0bcb0cf": lb.get(k, "<absent>")}
                for k in sorted(set(la) | set(lb)) if la.get(k, "<absent>") != lb.get(k, "<absent>")
                or type(la.get(k)) is not type(lb.get(k))}
        res[f"s{seed}/{name}"] = {"n_leaves": [len(la), len(lb)], "differing": diff}
out.write_text(json.dumps(res, indent=1) + "\n")
for k, v in res.items():
    if isinstance(v, str):
        print(k, v)
        continue
    print(k, "leaves", v["n_leaves"], "differing", len(v["differing"]))
    for kk, vv in v["differing"].items():
        print("   ", kk, "|", str(vv["e2e_0e6a57b"])[:110], "|", str(vv["setup_check_0bcb0cf"])[:110])
