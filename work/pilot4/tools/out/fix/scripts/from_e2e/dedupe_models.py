"""e2e lane: confirm that the stand-in runs' pairs are byte copies of the pairs the set-up path saved (sha256 against
edge_pairs.json's own record and against the files), then list the set-up copies for removal (disk); prints JSON.
Scratch helper; reads files only.

usage: python -I -B dedupe_models.py LANE OUT.json
"""

import hashlib
import json
import sys
from pathlib import Path

lane, out = Path(sys.argv[1]), Path(sys.argv[2])
res = {"seeds": {}, "all_equal": True}
for seed in (42, 52, 60):
    rec = json.loads((lane / "setup" / f"s{seed}" / "edge_pairs.json").read_text())["saved"]
    rows = {}
    for steps, v in rec.items():
        for kind, name in (("zip", f"stage2_{steps}_steps.zip"), ("vecnorm", f"stage2_vecnormalize_{steps}_steps.pkl")):
            a = lane / "setup" / f"s{seed}" / "models" / name
            b = lane / "runs" / f"pilot4_walk_s{seed}" / "03_locomotion" / "models" / name
            ha = "sha256:" + hashlib.sha256(a.read_bytes()).hexdigest()
            hb = "sha256:" + hashlib.sha256(b.read_bytes()).hexdigest()
            ok = ha == hb == v[kind]
            rows[name] = {"recorded": v[kind], "setup_copy": ha, "run_copy": hb, "equal": ok}
            res["all_equal"] &= ok
    res["seeds"][seed] = rows
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({"all_equal": res["all_equal"]}))
sys.exit(0 if res["all_equal"] else 1)
