"""Test 2 addendum (written on resume, after reading sweep.txt): transplants that sweep.py did not make.

PYTHONPATH=<policies/code_copy>:<this folder> python -B sweep_extra.py run/base sweep_extra.json > sweep_extra.txt

Network evaluation only (no simulation, no training; checkpoints opened read only).  Same method as sweep.py section D
(verify_loop.py's tail_3 sweep 0..6 deg at each policy's own 120 settled states, after replacing an input group by a
"typical" value), with two changes:
 1. donors: each START checkpoint also gets the OTHER START's typical values (median over that checkpoint's own 120
    settled states), not only the other lineage's ENDS' (sweep.py used the ends' medians for everybody);
 2. groups that keep the neck/head angles out (sweep.py's POSTURE group moved neck_pitch by about 4 normaliser SDs for
    s42 recipients, far outside what those policies saw): orientation = gravity + accelerometer + prey direction
    (every channel a pelvis pitch changes), legs = the 14 leg joint angles, ORIENT+LEGS (+ foot contacts).
Controls: the same transplant from the recipient's own donor (own lineage ends' median; for a start, its own median).
Readouts per recipient: offset c0 (cmd14 at tail 0), loop gain G (slope x 5.183 deg/unit), linear fixed point
c0/(1-G), cmd14 at the recipient's own tail angle.  Medians over the 120 states.  Observation-only, correlational in the
sense that it reads the network's map, not the closed loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pp_common import I_ACC, I_FOOT, I_GRAV, I_HEAD, I_LEGS, I_PREY, NAMES, OBS_NAMES, Pol, make_env, \
    pitch_deg_from_obs, pitch_illusion, stance_ctor
from sweep import cmd, summarize_loop, tail_family

GROUPS = {
    "orient(grav+acc+prey)": I_GRAV + I_ACC + I_PREY,
    "legs_q": I_LEGS,
    "ORIENT+LEGS+FEET": I_GRAV + I_ACC + I_PREY + I_LEGS + I_FOOT,
    "head_neck_q": I_HEAD,
    "ORIENT+LEGS+FEET+HEAD": I_GRAV + I_ACC + I_PREY + I_LEGS + I_FOOT + I_HEAD,
}


def row(pol, X, Y):
    sl, c0 = tail_family(pol, Y)
    return {**summarize_loop(sl, c0), "cmd14_own_tail": float(np.median(cmd(pol, Y)))}


def main() -> int:
    base = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    ctor = stance_ctor()
    factory = lambda: make_env(ctor)  # noqa: E731
    whos = ["start_s42", "start_s80"] + NAMES
    states = {}
    for who in whos:
        X = np.load(base / "arrays" / f"{who}.npz")["obs_snaps"].reshape(-1, 63)
        states[who] = X[np.all(np.isfinite(X), axis=1)]
    donors = {
        "ends_s42": np.median(np.concatenate([states[w] for w in NAMES if w.startswith("s42")]), axis=0),
        "ends_s80": np.median(np.concatenate([states[w] for w in NAMES if w.startswith("s80")]), axis=0),
        "start_s42": np.median(states["start_s42"], axis=0),
        "start_s80": np.median(states["start_s80"], axis=0),
    }
    dpitch = {k: float(pitch_deg_from_obs(v)) for k, v in donors.items()}
    res = {"donor_seen_pitch_deg": dpitch, "groups": {k: [OBS_NAMES[i] for i in v] for k, v in GROUPS.items()},
           "policies": {}}
    print("donor typical seen pitch (deg):", {k: round(v, 3) for k, v in dpitch.items()}, flush=True)
    for who in whos:
        pol = Pol(who, factory)
        X = states[who]
        lin = who[-3:] if who.startswith("start_") else who[:3]
        oth = "s80" if lin == "s42" else "s42"
        plan = {"other_ends": donors[f"ends_{oth}"], "own_ends": donors[f"ends_{lin}"]}
        if who.startswith("start_"):
            plan["other_start"] = donors[f"start_{oth}"]
            plan["own_start"] = donors[who]
        r = {"none": row(pol, X, X)}
        for dn, typ in plan.items():
            for gname, idx in GROUPS.items():
                Y = np.array(X, copy=True)
                Y[:, idx] = typ[idx]
                z = np.abs((Y[:, idx] - X[:, idx]) / pol.std[idx])
                r[f"{dn}:{gname}"] = {**row(pol, X, Y), "shift_median_max_z": float(np.median(z.max(axis=1)))}
            gap = float(pitch_deg_from_obs(typ)) - float(np.median(pitch_deg_from_obs(X)))
            r[f"{dn}:pitch_rotation"] = {**row(pol, X, pitch_illusion(X, gap)), "rotation_deg": gap}
        res["policies"][who] = r
        print(who, "none", {k: round(v, 4) for k, v in r["none"].items() if not isinstance(v, bool)}, flush=True)
    out_path.write_text(json.dumps(res, indent=1))
    # summary
    print()
    print("CHANGE AGAINST NO TRANSPLANT: d offset c0 / d gain G / d fixed point (clipped at 9.99) / d cmd14 at own tail;"
          " railed count by c0/(1-G) >= 1")
    P = res["policies"]

    def cell(v):
        v = np.asarray(v, float)
        if len(v) == 1:
            return f"{v[0]:+.3f}"
        return f"{np.median(v):+.3f} [{v.min():+.3f},{v.max():+.3f}] +{int((v > 0).sum())}/-{int((v < 0).sum())}"

    for g, ws in (("start_s42", ["start_s42"]), ("start_s80", ["start_s80"]),
                  ("s42 ends", [w for w in NAMES if w.startswith("s42")]),
                  ("s80 ends", [w for w in NAMES if w.startswith("s80")])):
        keys = [k for k in P[ws[0]] if k != "none"]
        n0 = sum(P[w]["none"]["railed_pred"] for w in ws)
        print(f"  {g}: none: c0 {cell([P[w]['none']['offset_c0'] for w in ws])}, G {cell([P[w]['none']['gain'] for w in ws])},"
              f" fixed point {cell([min(P[w]['none']['fixed_point_cmd'], 9.99) for w in ws])}, railed {n0}/{len(ws)}")
        for k in keys:
            d = {m: [min(P[w][k][m], 9.99) - min(P[w]["none"][m], 9.99) for w in ws]
                 for m in ("offset_c0", "gain", "fixed_point_cmd", "cmd14_own_tail")}
            n1 = sum(P[w][k]["railed_pred"] for w in ws)
            extra = (f" shift max z {cell([P[w][k]['shift_median_max_z'] for w in ws])}" if "shift_median_max_z" in P[ws[0]][k]
                     else f" rotation {cell([P[w][k]['rotation_deg'] for w in ws])} deg")
            print(f"    {k:36s} dc0 {cell(d['offset_c0'])} | dG {cell(d['gain'])} | dFP {cell(d['fixed_point_cmd'])}"
                  f" | dcmd {cell(d['cmd14_own_tail'])} | railed {n0}->{n1} |{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
