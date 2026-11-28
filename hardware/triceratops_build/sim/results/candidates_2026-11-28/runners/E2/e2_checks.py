"""E2's checks of its inputs, written as a result file (no simulation): (1) the B1 split reproduces each demo's stored
tau_drive_sq_mean (prereg section 3 item 3's 1e-6 test); (2) this runner's 1 ms episodes against the E1 session's own
episodes of the same policies, conditions and seeds (bit equality of sq_motor, sq_brake, rpm_hist and the poses);
(3) the harness check's records (seed 12345, 3 s, not an evaluation seed); (4) the provenance of every E2 script."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/user/mesozoic-labs/hardware/triceratops_build/sim")
import candidate_checks as C
E2 = C.CAND_SCRATCH / "E2"
E1 = C.CAND_SCRATCH / "E1" / "episodes"
RES = C.RESULTS / "E2"
eo = np.array([C.ETA_OLD[C.axis_of(a)] for a in C.LEG_ACTUATORS])
out = {"reproduction": [], "e1_cross_check": {}, "harness_check": {}, "scripts": {}}
for pol in C.E1_POLICIES:
    for blk, s0 in (("A", 700000), ("B", 710000)):
        demo = C.stored_demo(pol) if blk == "A" else json.loads((E2 / "demo" / pol / f"demo_course_seed{s0}.json").read_text())
        for seg, s in (("walk_0.25", s0), ("stand", s0 + 2)):
            e = json.loads((E2 / "episodes" / "B1_dt0.001" / pol / f"{seg}__{s}.json").read_text())
            st = np.array(demo["segments"][seg]["tau_drive_sq_mean"])
            rep = np.array(e["sq_motor"]) / eo**2 + np.array(e["sq_brake"]) * eo**2
            out["reproduction"].append({"policy": pol, "block": blk, "segment": seg, "seed": s,
                                        "rel_max": float(np.max(np.abs(rep - st) / np.abs(st)))})
same, diff, missing = [], [], []
for p in sorted((E2 / "episodes" / "B1_dt0.001").glob("*/*.json")):
    pol = p.parent.name
    cond, seed = p.stem.split("__")
    q = E1 / f"{pol}__{cond}__{seed}.npz"
    if not q.exists():
        missing.append(p.stem)
        continue
    a, b, za = json.loads(p.read_text()), np.load(q), np.load(p.with_suffix(".npz"))
    eq = (np.array_equal(np.array(a["sq_motor"]), b["sq_motor"]) and np.array_equal(np.array(a["sq_brake"]), b["sq_brake"])
          and np.array_equal(za["rpm_hist"], b["rpm_hist"]) and np.array_equal(za["q"], b["q"]))
    (same if eq else diff).append(f"{pol} {cond} {seed}")
out["e1_cross_check"] = {"identical": len(same), "different": diff, "missing_in_e1": missing,
                         "e1_folder": str(E1), "conditions": sorted({x.split()[1] for x in same})}
for p in sorted((E2 / "harness_check").glob("*/*/*.json")):
    r = json.loads(p.read_text())
    out["harness_check"][r["run"]] = {k: r[k] for k in ("seed", "condition", "fell", "termination", "delay_steps",
                                                         "delay_s", "physics_dt", "control_dt", "window_samples")}
code = E2 / "code"
for name in ("e2_run.py", "e2a_x6_beside.py", "e2_checks.py", "e2_tables.py", "check_repro.py"):
    out["scripts"][name] = C.provenance(code / name)["sha256"]["runner:" + str((code / name).resolve())]
out["provenance"] = C.provenance(Path(__file__))
(RES / "e2_checks.json").write_text(json.dumps(out, indent=1) + "\n")
print("reproduction rel max over 16:", max(x["rel_max"] for x in out["reproduction"]))
print("E1 cross-check:", {k: (v if not isinstance(v, list) else len(v)) for k, v in out["e1_cross_check"].items()})
print("harness:", {k: (v["fell"], v["delay_steps"], v["delay_s"]) for k, v in out["harness_check"].items()})
print("commit", out["provenance"]["commit"], "candidate_checks", out["provenance"]["sha256"]["sim/candidate_checks.py"][:16])
