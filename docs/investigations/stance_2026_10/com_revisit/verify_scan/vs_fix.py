"""Scan F6 spot check with my own harness: the robot fix's analogues on the Velociraptor (claw armature x3; 1 ms step,
control period kept), on 12 of the 46 travel-flagged spawns at 0.05 and calibration seeds 0-9 at 0.02.  Base values
from scan.jsonl (same seed; my base re-runs in vs_sample.jsonl reproduce them).
Usage: python vs_fix.py <scan.jsonl> > out/vs_fix.txt"""
import json
import sys

import numpy as np

import vs_lib as L


def claw3(u):
    m = u.model
    for j in ("l_claw", "r_claw"):
        m.dof_armature[m.jnt_dofadr[m.joint(j).id]] *= 3.0


def ms1(u):
    dt0 = u.dt
    u.model.opt.timestep /= 2
    u.frame_skip *= 2
    assert abs(u.dt - dt0) < 1e-12


scan = {}
for x in open(sys.argv[1]):
    r = json.loads(x)
    if r["model"] == "velo":
        scan[(r["seed"], r["noise"])] = r
flagged = sorted(s for (s, n), r in scan.items() if n == 0.05 and r["flag_travel"])
cases = [(s, 0.05) for s in flagged[::4]] + [(s, 0.02) for s in range(10)]
KEYS = ("com_fore_aft_mean_m", "com_lateral_mean_m", "com_height_mean_m", "root_pitch_mean_deg", "com_margin_lateral_min_m",
        "foot_travel_from_spawn_max_m", "foot_travel_max_m", "com_speed_end_rms_mps")
for vname, fn in (("claw_arm_x3", claw3), ("ts_1ms", ms1)):
    for noise in (0.05, 0.02):
        cs = [c for c in cases if c[1] == noise]
        dtrav, flags, keyd = [], [], {k: [] for k in KEYS}
        for seed, nz in cs:
            r = L.run("velo", seed, nz, perturb=fn)
            b = scan[(seed, nz)]
            dtrav.append(1e3 * (r["my_foot_travel_after1s_m"] - b["foot_travel_after1s_m"]))
            flags.append((r["flag_speed_test"], r["flag_travel"], r["flag_oscill"]))
            for k in KEYS:
                sc = 1.0 if k.endswith("_deg") else 1e3
                keyd[k].append(abs(sc * (r["test_metrics"][k] - b["metrics"][k])))
        print(f"{vname} noise {noise} seeds {[c[0] for c in cs]}: travel flag {sum(f[1] for f in flags)}/{len(cs)}, speed flag "
              f"{sum(f[0] for f in flags)}, oscill flag {sum(f[2] for f in flags)}; foot travel after 1 s, variant - base: "
              f"{min(dtrav):+.4f} to {max(dtrav):+.4f} mm")
        print("   max |variant - base| (mm, deg, mm/s): " + ", ".join(f"{k} {max(v):.4f}" for k, v in keyd.items()), flush=True)
