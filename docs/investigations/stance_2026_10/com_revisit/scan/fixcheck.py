"""The robot fix's analogues on the Velociraptor, in memory only: armature x3 on the light joint (both claws, the DOF that
carries the largest h M^-1 D, 1.963) and a 1 ms step (frame_skip doubled, control period kept).  Run on the 46 spawns
whose foot-travel flag tripped at reset noise 0.05, and on the 30 calibration spawns (seeds 0-29) at 0.02 to check that
resting CoM numbers do not move.  Base rows are scan.jsonl's (the runs are deterministic: velo_ext.txt reproduces them).
Usage: python fixcheck.py out.jsonl"""

import json
import sys

import scanlib as S

VARIANTS = {"claw_arm_x3": {"arm_joints": ["l_claw", "r_claw"], "arm_factor": 3.0}, "ts_1ms": {"hdiv": 2}}
base = {}
for line in open("scan.jsonl"):
    r = json.loads(line)
    if r["model"] == "velo" and r["noise"] is not None:
        base[(r["noise"], r["seed"])] = r
cases = [(0.05, s) for (n, s), r in sorted(base.items()) if n == 0.05 and r["flag_travel"]]
cases += [(0.02, s) for s in range(30)]
out = open(sys.argv[1], "a")
for name, spec in VARIANTS.items():
    for noise, seed in cases:
        row, ep, metrics, _ = S.run("velo", seed, noise, spec)
        row["variant"] = name
        out.write(json.dumps(row) + "\n")
        out.flush()
    print(f"{name}: {len(cases)} spawns done", flush=True)
