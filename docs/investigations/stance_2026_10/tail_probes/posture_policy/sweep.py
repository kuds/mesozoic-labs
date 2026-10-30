"""Test 2, observation only: does the policy's tail_3 command respond to posture inputs?

PYTHONPATH=<policies/code_copy>:<this folder> python -B sweep.py BASE_RUN_DIR OUT.json

Network evaluation only (no simulation, no training; checkpoints opened read only).  The method is the verifier's
verify_loop.py (copied here as verify_loop_copy.py): take settled observations, change only the named inputs, read the
policy's raw Gaussian mean (pre-clip) of entry 14 (tail_3_pitch_act).  Here the settled observations are all of the
baseline rollouts' snapshots (run_base.py: raw env observations after steps 500, 600, ..., 1000, seeds 3042-3061,
so 120 states per policy).  Slopes are least squares per state, then the median over states.

Per policy (start_s42, start_s80, 12 s42 ends, 12 s80 ends):
 A. every one of the 63 inputs: five points at -2h..+2h, h = 0.25 x the policy's own normaliser scale for that input;
    slope of cmd14 in raw units per unit input, and per normaliser scale.
 B. physical sweeps: pitch illusion (pp_common.pitch_illusion; all channels, and gravity / accelerometer / prey only)
    -2..+2 deg; roll illusion -2..+2 deg; tail_3 angle 0..6 deg (the loop, as verify_loop.py); neck_pitch, head_pitch,
    tail_1_pitch, tail_2_pitch +-2 deg; both hips / knees / ankles pitch +-1 deg together; both foot contacts +-40 N
    together; pitch rate (gyro_y) +-0.05 rad/s.  Also head_pitch and neck_pitch commands under the pitch illusion.
 C. the tail loop under the pitch illusion: the tail_3 sweep (0..6 deg) repeated at illusions -2.2, -1.1, 0, +1.1,
    +2.2 deg: offset (cmd14 at tail 0), slope per deg and loop gain (slope x the plant's 5.183 deg per unit command,
    verify_clamp/loop_summary.txt), and the implied fixed point c0 / (1 - G) (railed if >= 1 or G >= 1).
 D. transplant: each s42 end gets, in place of its own values, the s80 ends' typical (median over 12 policies x 120
    states) values of an input group, and each s80 end the s42 ends' (the reverse); the starts likewise (start_s42 gets
    s80's, start_s80 gets s42's).  Control: the same transplant from the policy's OWN lineage median.  Groups: gravity +
    accelerometer, head/neck angles, leg angles, foot contacts, tail_1/tail_2 angles, rates (gyro, heading velocity,
    joint velocities), POSTURE (gravity + accelerometer + head/neck + legs + feet), and everything but tail_3's angle and
    the prey/command block; plus the pitch rotation by the lineage gap (pitch_illusion, all channels).  Readouts: cmd14
    at the policy's own tail angle, offset at tail 0, loop gain, implied fixed point.
 E. within each policy, across its 120 settled states: Spearman of the observed pitch against cmd14's offset (tail
    angle set to 0), i.e. whether states that happen to be more nose-down carry a higher offset (correlational).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from pp_common import (I_ACC, I_GRAV, I_GYRO, I_HEAD, I_HVEL, I_JVEL, I_LEGS, I_FOOT, I_TAIL12, I_TAIL3, NAMES,
                       OBS_NAMES, TAIL3, Pol, make_env, pitch_deg_from_obs, pitch_illusion, roll_illusion, stance_ctor)

PLANT_DEG_PER_UNIT = 5.183  # statue tail_3 at command +1 minus 0 (verify_clamp/loop_summary.txt)
TAIL_DEG = np.arange(0.0, 6.01, 1.0)
PITCH_DEG = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
LOOP_PITCH = [-2.2, -1.1, 0.0, 1.1, 2.2]


def slope_per_state(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """ys (k, n): least-squares slope against xs (k,) for each of n states."""
    xc = xs - xs.mean()
    return (xc[:, None] * (ys - ys.mean(axis=0))).sum(axis=0) / (xc ** 2).sum()


def cmd(pol: Pol, X: np.ndarray, j: int = TAIL3) -> np.ndarray:
    return pol.raw_mean(X)[:, j]


def family(pol: Pol, X: np.ndarray, xs, make, outs=(TAIL3,)):
    """Apply make(X, x) for each x; return {j: slope per state} and {j: (k, n) curves}."""
    ys = np.stack([pol.raw_mean(make(X, x)) for x in xs])  # (k, n, 15)
    return {j: slope_per_state(np.asarray(xs, float), ys[:, :, j]) for j in outs}, ys


def set_idx(idx, delta_fn):
    def make(X, x):
        Y = np.array(X, dtype=np.float64, copy=True)
        for i in idx:
            Y[:, i] = delta_fn(Y[:, i], x)
        return Y.astype(np.float32)
    return make


def tail_family(pol: Pol, X: np.ndarray):
    """The loop: tail_3 angle set to 0..6 deg.  Returns per-state slope (per deg), offset at 0 deg."""
    ys = np.stack([cmd(pol, set_idx([I_TAIL3], lambda v, x: np.full_like(v, math.radians(x)))(X, t)) for t in TAIL_DEG])
    return slope_per_state(TAIL_DEG, ys), ys[0]


def summarize_loop(slope, c0):
    g = float(np.median(slope)) * PLANT_DEG_PER_UNIT
    c = float(np.median(c0))
    fp = c / (1.0 - g) if g < 1.0 else float("inf")
    return {"slope_per_deg": float(np.median(slope)), "gain": g, "offset_c0": c, "fixed_point_cmd": fp,
            "railed_pred": bool(fp >= 1.0 or g >= 1.0)}


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    base = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    ctor = stance_ctor()
    factory = lambda: make_env(ctor)  # noqa: E731
    test = len(sys.argv) > 3  # code test only: the starts stand in for the lineages
    who_list = sys.argv[3].split(",") if test else ["start_s42", "start_s80"] + NAMES
    states = {}
    for who in who_list:
        z = np.load(base / "arrays" / f"{who}.npz")
        X = z["obs_snaps"].reshape(-1, 63)
        states[who] = X[np.all(np.isfinite(X), axis=1)]
    pool = {lin: ([states[f"start_{lin}"]] if test else [states[w] for w in NAMES if w.startswith(lin)])
            for lin in ("s42", "s80")}
    typ = {lin: np.median(np.concatenate(pool[lin]), axis=0) for lin in ("s42", "s80")}
    seen_pitch = {lin: float(np.median(pitch_deg_from_obs(np.concatenate(pool[lin])))) for lin in ("s42", "s80")}
    result = {"typical": {k: v.tolist() for k, v in typ.items()}, "seen_pitch_lineage_median_deg": seen_pitch,
              "obs_names": OBS_NAMES, "policies": {}}
    print("lineage median seen pitch (deg):", seen_pitch, flush=True)
    for who in who_list:
        pol = Pol(who, factory)
        X = states[who]
        n = len(X)
        r: dict = {"n_states": n, "seen_pitch_deg_median": float(np.median(pitch_deg_from_obs(X))),
                   "tail3_deg_median": float(np.degrees(np.median(X[:, I_TAIL3]))),
                   "cmd14_at_states_median": float(np.median(cmd(pol, X)))}
        # A. every input
        A = {}
        for i in range(63):
            h = 0.25 * pol.std[i]
            ks = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * h
            sl, _ = family(pol, X, ks, set_idx([i], lambda v, x: v + x))
            s = float(np.median(sl[TAIL3]))
            A[OBS_NAMES[i]] = {"slope_raw": s, "slope_per_normstd": s * float(pol.std[i]), "normstd": float(pol.std[i])}
        r["A_per_input"] = A
        # B. physical families
        B = {}
        outs = (TAIL3, 2, 0)
        fams = {
            "pitch_illusion_all(per deg nose-up)": (PITCH_DEG, lambda X_, x: pitch_illusion(X_, x)),
            "pitch_illusion_grav_only": (PITCH_DEG, lambda X_, x: pitch_illusion(X_, x, ("grav",))),
            "pitch_illusion_acc_only": (PITCH_DEG, lambda X_, x: pitch_illusion(X_, x, ("acc",))),
            "pitch_illusion_prey_only": (PITCH_DEG, lambda X_, x: pitch_illusion(X_, x, ("prey",))),
            "roll_illusion_all(per deg +x)": (PITCH_DEG, lambda X_, x: roll_illusion(X_, x)),
            "neck_pitch(per deg)": (PITCH_DEG, set_idx([0], lambda v, x: v + math.radians(x))),
            "head_pitch(per deg)": (PITCH_DEG, set_idx([2], lambda v, x: v + math.radians(x))),
            "tail_1_pitch(per deg)": (PITCH_DEG, set_idx([3], lambda v, x: v + math.radians(x))),
            "tail_2_pitch(per deg)": (PITCH_DEG, set_idx([5], lambda v, x: v + math.radians(x))),
            "both_hip_pitch(per deg)": (PITCH_DEG / 2, set_idx([7, 14], lambda v, x: v + math.radians(x))),
            "both_knee(per deg)": (PITCH_DEG / 2, set_idx([9, 16], lambda v, x: v + math.radians(x))),
            "both_ankle(per deg)": (PITCH_DEG / 2, set_idx([10, 17], lambda v, x: v + math.radians(x))),
            "both_foot_contact(per 10 N)": (PITCH_DEG * 2, set_idx(I_FOOT, lambda v, x: v + 10.0 * x)),
            "pitch_rate_gyro_y(per 0.01 rad/s)": (PITCH_DEG * 2.5, set_idx([46], lambda v, x: v + 0.01 * x)),
        }
        for name, (xs, make) in fams.items():
            sl, ys = family(pol, X, xs, make, outs)
            B[name] = {"cmd14_slope": float(np.median(sl[TAIL3])), "head_cmd_slope": float(np.median(sl[2])),
                       "neck_cmd_slope": float(np.median(sl[0])),
                       "cmd14_curve_median": np.median(ys[:, :, TAIL3], axis=1).round(5).tolist(), "xs": list(map(float, xs))}
        sl, c0 = tail_family(pol, X)
        B["tail_3_angle(per deg) [THE LOOP]"] = {"cmd14_slope": float(np.median(sl)), "offset_c0": float(np.median(c0)),
                                                 "gain": float(np.median(sl)) * PLANT_DEG_PER_UNIT,
                                                 "slope_pos_share": float((sl > 0).mean())}
        r["B_families"] = B
        # C. loop under the pitch illusion
        C = {}
        for p in LOOP_PITCH:
            sl, c0 = tail_family(pol, pitch_illusion(X, p) if p else X)
            C[f"{p:+.1f}"] = summarize_loop(sl, c0)
        r["C_loop_vs_pitch"] = C
        # D. transplant
        lin = who[-3:] if who.startswith("start_") else who[:3]
        other = "s80" if lin == "s42" else "s42"
        groups = {
            "grav+acc": I_GRAV + I_ACC,
            "head_neck_q": I_HEAD,
            "legs_q": I_LEGS,
            "foot_contact": I_FOOT,
            "tail12_q": I_TAIL12,
            "rates(gyro+hvel+jvel)": I_GYRO + I_HVEL + I_JVEL,
            "POSTURE(grav+acc+head+legs+feet)": I_GRAV + I_ACC + I_HEAD + I_LEGS + I_FOOT,
            "ALL_but_tail3_prey_cmd": [i for i in range(56) if i != I_TAIL3],
        }
        D = {"none": {}}
        sl, c0 = tail_family(pol, X)
        D["none"] = {**summarize_loop(sl, c0), "cmd14_own_tail": float(np.median(cmd(pol, X)))}
        for src_name, src in (("other", other), ("own", lin)):
            for gname, idx in groups.items():
                Y = np.array(X, copy=True)
                Y[:, idx] = typ[src][idx]
                sl, c0 = tail_family(pol, Y)
                z = np.abs((Y[:, idx] - X[:, idx]) / pol.std[idx])
                D[f"{src_name}:{gname}"] = {**summarize_loop(sl, c0), "cmd14_own_tail": float(np.median(cmd(pol, Y))),
                                            "shift_max_z": float(np.median(z.max(axis=1))),
                                            "shift_max_z_input": OBS_NAMES[idx[int(np.argmax(np.median(z, axis=0)))]]}
            gap = seen_pitch[src] - float(np.median(pitch_deg_from_obs(X)))
            Y = pitch_illusion(X, gap)
            sl, c0 = tail_family(pol, Y)
            D[f"{src_name}:pitch_rotation_to_lineage_median"] = {**summarize_loop(sl, c0), "rotation_deg": gap,
                                                                 "cmd14_own_tail": float(np.median(cmd(pol, Y)))}
        r["D_transplant"] = D
        # E. across-state association (correlational)
        _, c0 = tail_family(pol, X)
        r["E_spearman_pitch_vs_offset"] = spearman(pitch_deg_from_obs(X), c0)
        r["E_pitch_sd_deg"] = float(np.std(pitch_deg_from_obs(X)))
        result["policies"][who] = r
        print(who, "n", n, "loop", D["none"], flush=True)
    out_path.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
