"""The Velociraptor's slow sideways relaxation (scan F5/F10): 60 s runs, base and in-memory variants, with my own
harness (vs_lib).  Fits log|mean lateral floor force| per foot over seconds 10-58 (both feet), reports the fit's
worst log residual, the CoM speed per second, foot travel 10-60 s.  Variants discriminate what sets the 14 s time
constant: noslip (no soft-friction slip), contact solref timeconst x2, impratio 10, servo kp x2, joint damping x2.
Usage: python vs_ext.py out.jsonl"""
import json
import sys

import mujoco
import numpy as np

import vs_lib as L


def noslip(u):
    u.model.opt.noslip_iterations = 10


def solref2(u):
    u.model.geom_solref[:, 0] *= 2.0


def impratio10(u):
    u.model.opt.impratio = 10.0


def kp2(u):
    m = u.model
    for a in range(m.nu):
        if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE:
            m.actuator_gainprm[a, 0] *= 2.0
            m.actuator_biasprm[a, 1] *= 2.0


def damp2(u):
    u.model.dof_damping[6:] *= 2.0


VARIANTS = {"base": None, "noslip10": noslip, "solref_tc_x2": solref2, "impratio10": impratio10, "kp_x2": kp2,
            "damping_x2": damp2}
CASES = [(v, None, 0) for v in VARIANTS] + [(v, 0.05, 126) for v in VARIANTS] + [("base", 0.05, 1), ("base", 0.05, 20),
                                                                                   ("base", 0.02, 274)]
import os
done = set()
if os.path.exists(sys.argv[1]):
    for x in open(sys.argv[1]):
        q = json.loads(x)
        done.add((q["variant"], q["noise"], q["seed"]))
out = open(sys.argv[1], "a")
for vname, noise, seed in CASES:
    if (vname, noise, seed) in done:
        continue
    r = L.run("velo", seed, noise, steps=6000, perturb=VARIANTS[vname])
    if not r["stood"] or len(r["my_lateral_per_s_n"]) < 60:
        row = {"variant": vname, "noise": noise, "seed": seed, "stood": r["stood"], "steps": r["steps"], "T": r["T"],
               "lat_per_s": r["my_lateral_per_s_n"]}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(f"{vname:13s} noise {noise} seed {seed}: stood {r['stood']} steps {r['steps']} (t = {r['T']:.2f} s); not fitted", flush=True)
        continue
    lat = np.array(r["my_lateral_per_s_n"])  # (60, 2)
    s = np.arange(10, 59)
    fits = []
    for k in range(2):
        y = np.log(np.abs(lat[10:59, k]) + 1e-15)
        b, c = np.polyfit(s, y, 1)
        res = y - (b * s + c)
        fits.append((-1.0 / b if b < 0 else float("inf"), float(np.abs(res).max())))
    cs = r["my_com_speed_per_s_mmps"]
    row = {"variant": vname, "noise": noise, "seed": seed, "stood": r["stood"], "tau_s": [f[0] for f in fits],
           "logfit_max_resid": [f[1] for f in fits], "lat_s1": lat[1].tolist(), "lat_s5": lat[5].tolist(),
           "lat_s9": lat[9].tolist(), "lat_s29": lat[29].tolist(), "lat_s59": lat[59].tolist(),
           "com_speed_s3": cs[3], "com_speed_s9": cs[9], "com_speed_s29": cs[29], "com_speed_s59": cs[59],
           "foot_travel_after1s_mm": 1e3 * r["my_foot_travel_after1s_m"], "load_sd_tail_bw": r["my_load_sd_bw_max"],
           "floor_geoms": sorted(r["floor_geoms_whole"]), "other_floor_max_n": r["my_other_floor_max_n_whole"]}
    out.write(json.dumps(row) + "\n")
    out.flush()
    print(f"{vname:13s} noise {noise} seed {seed}: stood {r['stood']}; tau r/l {fits[0][0]:.2f}/{fits[1][0]:.2f} s "
          f"(max log resid {fits[0][1]:.3f}/{fits[1][1]:.3f}); lateral N s1 {np.round(lat[1], 3).tolist()} s9 "
          f"{np.round(lat[9], 3).tolist()} s59 {np.round(lat[59], 4).tolist()}; CoM speed mm/s s3 {cs[3]:.4f} s9 {cs[9]:.4f} "
          f"s59 {cs[59]:.5f}; foot travel after 1 s {1e3 * r['my_foot_travel_after1s_m']:.2f} mm; load sd {r['my_load_sd_bw_max']:.1e} BW",
          flush=True)
