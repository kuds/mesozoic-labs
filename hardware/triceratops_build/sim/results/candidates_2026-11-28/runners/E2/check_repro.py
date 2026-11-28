"""Cheap consistency checks (no simulation): (1) does the B1 split reproduce each demo's stored tau_drive_sq_mean;
(2) are this runner's E1-equivalent episodes identical to the E1 session's own episodes (same code path, seeds)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/user/mesozoic-labs/hardware/triceratops_build/sim")
import candidate_checks as C
E2 = C.CAND_SCRATCH / "E2"
E1 = C.CAND_SCRATCH / "E1" / "episodes"
eo = np.array([C.ETA_OLD[C.axis_of(a)] for a in C.LEG_ACTUATORS])
for pol in C.E1_POLICIES:
    for blk, s0 in (("A", 700000), ("B", 710000)):
        demo = C.stored_demo(pol) if blk == "A" else json.loads((E2 / "demo" / pol / f"demo_course_seed{s0}.json").read_text())
        for seg, s in (("walk_0.25", s0), ("stand", s0 + 2)):
            e = json.loads((E2 / "episodes" / "B1_dt0.001" / pol / f"{seg}__{s}.json").read_text())
            st = np.array(demo["segments"][seg]["tau_drive_sq_mean"])
            rep = np.array(e["sq_motor"]) / eo**2 + np.array(e["sq_brake"]) * eo**2
            print(pol, blk, seg, s, "rel max", f"{np.max(np.abs(rep - st) / np.abs(st)):.3e}",
                  "speed demo", round(demo["segments"][seg]["speed_mps"], 6), "sweep", round(e["speed"], 6))
same = diff = missing = 0
worst = 0.0
for p in sorted((E2 / "episodes" / "B1_dt0.001").glob("*/*.json")):
    pol = p.parent.name
    cond, seed = p.stem.split("__")
    q = E1 / f"{pol}__{cond}__{seed}.npz"
    if not q.exists():
        missing += 1
        continue
    a, b = json.loads(p.read_text()), np.load(q)
    za = np.load(p.with_suffix(".npz"))
    eq = (np.array_equal(np.array(a["sq_motor"]), b["sq_motor"]) and np.array_equal(np.array(a["sq_brake"]), b["sq_brake"])
          and np.array_equal(za["rpm_hist"], b["rpm_hist"]) and np.array_equal(za["q"], b["q"]))
    worst = max(worst, float(np.max(np.abs(np.array(a["drive_sq"]) - b["drive_sq"]))))
    same += eq
    diff += not eq
print(f"E1 cross-check: identical {same}, different {diff}, not yet in E1's folder {missing}; max |drive_sq diff| {worst:.3e}")
