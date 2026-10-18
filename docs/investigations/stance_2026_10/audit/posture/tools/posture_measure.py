"""Reduce posture_roll.py's per-step arrays to quiet-stance posture measures (pure numpy + the home keyframe).

    python -P posture_measure.py --rolls ROLLS_DIR --out OUT_DIR

Windows (control steps, state after the step): w40 = [40, T) (a 40-step settle), w200 = [200, T) (the stance
gate's own settle), last40 = [T-40, T), first40 = [200, 240).  Per episode the window mean of each per-step
quantity; per target the mean, sd, min and max over the episodes (seeds).

Conventions
  h          the pelvis heading (horizontal projection of the pelvis x axis); "fore-aft" = along h
  femur/tibia/metatarsus angle from vertical: signed in the sagittal plane (h, z), + = the distal end ahead of
             the proximal one; femur = hip_pitch anchor -> knee anchor, tibia = knee -> ankle anchor,
             metatarsus = ankle -> digit-III MTP anchor
  included angles (3D): knee = angle(hip-knee, ankle-knee), ankle = angle(knee-ankle, MTP-ankle),
             MTP = angle(ankle-MTP, digit-III axis end - MTP); 180 = straight
  pad statistic: (point - pad centre) . pad forward (horizontal) / 0.07 m; + toward the toe
  footprint fraction: (point - pad rear edge) . pad forward / (digit-III tip - pad rear edge) . pad forward
  CoP: recorder's signed pad CoP (the gate's normalisation); whole-foot CoP = vertical-force-weighted floor
             contact point of every contact of the limb (boundary sample), and the recorder's foot_cop
  knee vs ground-force line: (knee - whole-foot CoP) . h, per foot (vertical line), and the signed sagittal
             distance from the knee to the foot's actual reaction line (through the CoP along F)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

PAD_HALF = 0.07
WINDOWS = ("w40", "w200", "last40", "first40")


def win(name: str, T: int) -> slice:
    return {"w40": slice(40, T), "w200": slice(200, T), "last40": slice(T - 40, T), "first40": slice(200, 240)}[name]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def signed_from_vertical(prox, dist, h):
    d = dist - prox
    fa = np.einsum("ij,ij->i", d, h)
    vz = d[:, 2]
    return np.degrees(np.arctan2(fa, -vz))


def included(a, vertex, b):
    u, v = unit(a - vertex), unit(b - vertex)
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u, v), -1, 1)))


def per_step(z) -> dict[str, np.ndarray]:
    h = z["heading"]
    out: dict[str, np.ndarray] = {}
    hip_mid = 0.5 * (z["r_hip"] + z["l_hip"])
    com = z["com"]
    out["pelvis_pitch_deg"] = z["pelvis_pitch_deg"]
    out["pelvis_z_m"] = z["pelvis_pos"][:, 2]
    Lf = np.linalg.norm(z["r_knee"] - z["r_hip"], axis=1)  # femur length (rigid; checked below)
    out["femur_len_m"] = Lf
    out["com_ahead_of_hip_m"] = np.einsum("ij,ij->i", com - hip_mid, h)
    out["com_ahead_of_hip_pct_femur"] = 100 * out["com_ahead_of_hip_m"] / Lf
    out["com_below_hip_pct_femur"] = 100 * (hip_mid[:, 2] - com[:, 2]) / Lf
    out["com_height_m"] = com[:, 2]
    fz_tot = np.zeros(len(h))
    cop_net = np.zeros((len(h), 3))
    for s in "rl":
        hip, knee, ank, mtp = z[f"{s}_hip"], z[f"{s}_knee"], z[f"{s}_ankle"], z[f"{s}_mtp"]
        d3end, d3tip = z[f"{s}_d3_axis_end"], z[f"{s}_d3_tip"]
        out[f"{s}_femur_from_vertical_deg"] = signed_from_vertical(hip, knee, h)
        out[f"{s}_tibia_from_vertical_deg"] = signed_from_vertical(knee, ank, h)
        out[f"{s}_metatarsus_from_vertical_deg"] = signed_from_vertical(ank, mtp, h)
        out[f"{s}_knee_included_deg"] = included(hip, knee, ank)
        out[f"{s}_ankle_included_deg"] = included(knee, ank, mtp)
        out[f"{s}_mtp_included_deg"] = included(ank, mtp, d3end)
        out[f"{s}_digit3_from_horizontal_deg"] = np.degrees(np.arcsin(np.clip(unit(d3end - mtp)[:, 2], -1, 1)))
        for q in ("hip_pitch", "knee", "ankle", "toe_d3_joint"):
            out[f"{s}_q_{q}_deg"] = z[f"{s}_q_{q}_deg"]
        pf = z[f"{s}_pad_fwd"].copy()
        pf[:, 2] = 0
        pf = unit(pf)
        padc = z[f"{s}_pad_c"]
        rear = padc - PAD_HALF * pf
        foot_len = np.einsum("ij,ij->i", d3tip - rear, pf)
        out[f"{s}_footprint_len_m"] = foot_len
        out[f"{s}_hip_ahead_of_ankle_m"] = np.einsum("ij,ij->i", hip - ank, h)
        # COM projection over this foot
        out[f"{s}_com_padstat"] = np.einsum("ij,ij->i", com - padc, pf) / PAD_HALF
        out[f"{s}_com_footfrac"] = np.einsum("ij,ij->i", com - rear, pf) / foot_len
        # whole-foot CoP (boundary sample) and the pad CoP (recorder)
        cop = z[f"{s}_cop_xyz"]
        out[f"{s}_wholefoot_cop_padstat"] = np.einsum("ij,ij->i", cop - padc, pf) / PAD_HALF
        out[f"{s}_wholefoot_cop_footfrac"] = np.einsum("ij,ij->i", cop - rear, pf) / foot_len
        out[f"{s}_wholefoot_cop_ahead_of_mtp_m"] = np.einsum("ij,ij->i", cop - mtp, pf)
        out[f"{s}_rec_pad_cop_signed"] = z[f"rec_{s}_sole_cop_fore"]
        out[f"{s}_rec_foot_cop_m_ahead_ankle"] = z[f"rec_{s}_foot_cop_m"]
        # knee / hip / ankle relative to the foot's vertical ground-force line (through the whole-foot CoP)
        for jn, jp in (("knee", knee), ("hip", hip), ("ankle", ank)):
            out[f"{s}_{jn}_ahead_of_vGRF_m"] = np.einsum("ij,ij->i", jp - cop, h)
        out[f"{s}_knee_ahead_of_vGRF_pct_femur"] = 100 * out[f"{s}_knee_ahead_of_vGRF_m"] / Lf
        F = z[f"{s}_F"]
        Fh = np.einsum("ij,ij->i", F, h)
        Fz = F[:, 2]
        dk = knee - cop
        dkh = np.einsum("ij,ij->i", dk, h)
        out[f"{s}_knee_to_GRFline_m"] = (dkh * Fz - dk[:, 2] * Fh) / np.sqrt(Fh**2 + Fz**2)
        out[f"{s}_GRF_lean_deg"] = np.degrees(np.arctan2(Fh, Fz))  # + = reaction tilted forward (pushes animal forward)
        out[f"{s}_Fz_N"] = Fz
        # load shares (recorder floor normal forces)
        leg = z[f"rec_{s}_leg_floor_N"]
        out[f"{s}_rec_leg_N"] = leg
        out[f"{s}_rec_pad_N"] = z[f"rec_support_{s}_plantar_geom_N"]
        out[f"{s}_rec_digits_N"] = sum(z[f"rec_support_{s}_toe_d{k}_geom_N"] for k in (2, 3, 4))
        for k in (2, 3, 4):
            out[f"{s}_rec_d{k}_N"] = z[f"rec_support_{s}_toe_d{k}_geom_N"]
        out[f"{s}_probe_pad_N"] = z[f"{s}_fz_pad"]
        out[f"{s}_probe_toes_N"] = z[f"{s}_fz_toes"]
        out[f"{s}_probe_other_N"] = z[f"{s}_fz_other"]
        fz_tot = fz_tot + np.where(np.isfinite(cop[:, 0]), Fz, 0)
        cop_net = cop_net + np.where(np.isfinite(cop), cop, 0) * Fz[:, None]
    cop_net = cop_net / fz_tot[:, None]
    padmid = 0.5 * (z["r_pad_c"] + z["l_pad_c"])
    out["net_cop_minus_com_m"] = np.einsum("ij,ij->i", cop_net - com, h)
    out["com_ahead_of_padmid_m"] = np.einsum("ij,ij->i", com - padmid, h)
    out["com_ahead_of_padmid_padstat"] = out["com_ahead_of_padmid_m"] / PAD_HALF
    v = np.diff(com[:, :2], axis=0) / 0.01  # control dt = 0.002 * 5
    out["com_hspeed_mps"] = np.concatenate([[np.nan], np.linalg.norm(v, axis=1)])
    return out


RATIO_KEYS = {  # window quantities that are load-weighted ratios rather than plain means
    "pad_share": ("rec_pad_N", "rec_leg_N"), "digit_share": ("rec_digits_N", "rec_leg_N"),
    "d2_share": ("rec_d2_N", "rec_leg_N"), "d3_share": ("rec_d3_N", "rec_leg_N"), "d4_share": ("rec_d4_N", "rec_leg_N"),
    "probe_pad_share": ("probe_pad_N", None), "probe_toes_share": ("probe_toes_N", None),
}


def reduce_episode(ps: dict[str, np.ndarray], T: int) -> dict[str, dict[str, float]]:
    res: dict[str, dict[str, float]] = {}
    for w in WINDOWS:
        sl = win(w, T)
        r: dict[str, float] = {}
        for k, v in ps.items():
            if k.startswith(("r_rec_", "l_rec_", "r_probe_", "l_probe_")) and not k.endswith(("pad_cop_signed", "foot_cop_m_ahead_ankle")):
                continue
            x = v[sl]
            r[k] = float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
        for s in "rl":
            leg = np.nansum(ps[f"{s}_rec_leg_N"][sl])
            for name, (num, den) in RATIO_KEYS.items():
                if den is None:
                    tot = np.nansum(ps[f"{s}_probe_pad_N"][sl] + ps[f"{s}_probe_toes_N"][sl] + ps[f"{s}_probe_other_N"][sl])
                    r[f"{s}_{name}"] = float(np.nansum(ps[f"{s}_{num}"][sl]) / tot) if tot > 0 else float("nan")
                else:
                    r[f"{s}_{name}"] = float(np.nansum(ps[f"{s}_{num}"][sl]) / leg) if leg > 0 else float("nan")
            r[f"{s}_load_share_of_both"] = float(leg / (np.nansum(ps["r_rec_leg_N"][sl]) + np.nansum(ps["l_rec_leg_N"][sl])))
            r[f"{s}_rec_pad_cop_gate"] = float(np.mean(np.where(np.isfinite(ps[f"{s}_rec_pad_cop_signed"][sl]),
                                                                 np.abs(ps[f"{s}_rec_pad_cop_signed"][sl]), 1.0)))
        r["com_hspeed_rms_mps"] = float(np.sqrt(np.nanmean(ps["com_hspeed_mps"][sl] ** 2)))
        # leg means (both feet)
        for base in ("femur_from_vertical_deg", "tibia_from_vertical_deg", "metatarsus_from_vertical_deg",
                     "knee_included_deg", "ankle_included_deg", "mtp_included_deg", "digit3_from_horizontal_deg",
                     "com_padstat", "com_footfrac", "wholefoot_cop_padstat", "wholefoot_cop_footfrac",
                     "wholefoot_cop_ahead_of_mtp_m", "rec_pad_cop_signed", "rec_pad_cop_gate", "knee_ahead_of_vGRF_m",
                     "knee_ahead_of_vGRF_pct_femur", "hip_ahead_of_vGRF_m", "ankle_ahead_of_vGRF_m",
                     "knee_to_GRFline_m", "GRF_lean_deg", "pad_share", "digit_share", "d2_share", "d3_share",
                     "d4_share", "footprint_len_m", "hip_ahead_of_ankle_m", "q_hip_pitch_deg", "q_knee_deg",
                     "q_ankle_deg", "q_toe_d3_joint_deg", "probe_pad_share", "probe_toes_share"):
            r[f"mean_{base}"] = 0.5 * (r[f"r_{base}"] + r[f"l_{base}"])
        r["max_feet_rec_pad_cop_gate"] = max(r["r_rec_pad_cop_gate"], r["l_rec_pad_cop_gate"])
        res[w] = r
    return res


def home_reference(mjcf: str) -> dict[str, float]:
    """The same geometry at the home keyframe (mj_forward, no contacts solved)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(mjcf)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)
    J = lambda n: d.xanchor[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)].copy()  # noqa: E731
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    G = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    pel = B("pelvis")
    R = d.xmat[pel].reshape(3, 3)
    z: dict[str, np.ndarray] = {"heading": np.array([[R[0, 0], R[1, 0], 0.0]]) / np.hypot(R[0, 0], R[1, 0]),
                                "com": d.subtree_com[pel][None].copy(),
                                "pelvis_pitch_deg": np.array([math.degrees(math.asin(-R[2, 0]))]),
                                "pelvis_pos": d.xpos[pel][None].copy()}
    for s in "rl":
        z[f"{s}_hip"], z[f"{s}_knee"] = J(f"{s}_hip_pitch")[None], J(f"{s}_knee")[None]
        z[f"{s}_ankle"], z[f"{s}_mtp"] = J(f"{s}_ankle")[None], J(f"{s}_toe_d3_joint")[None]
        Rt = d.xmat[B(f"{s}_toe_d3")].reshape(3, 3)
        z[f"{s}_d3_axis_end"] = (d.xpos[B(f"{s}_toe_d3")] + Rt[:, 0] * 0.19)[None]
        z[f"{s}_d3_tip"] = (d.xpos[B(f"{s}_toe_d3")] + Rt[:, 0] * 0.218)[None]
        z[f"{s}_pad_c"] = d.geom_xpos[G(f"{s}_plantar_geom")][None].copy()
        z[f"{s}_pad_fwd"] = d.geom_xmat[G(f"{s}_plantar_geom")].reshape(3, 3)[:, 0][None].copy()
    h = z["heading"]
    hip_mid = 0.5 * (z["r_hip"] + z["l_hip"])
    Lf = float(np.linalg.norm(z["r_knee"] - z["r_hip"]))
    out = {"femur_len_m": Lf, "tibia_len_m": float(np.linalg.norm(z["r_ankle"] - z["r_knee"])),
           "metatarsus_len_m": float(np.linalg.norm(z["r_mtp"] - z["r_ankle"])),
           "pelvis_pitch_deg": float(z["pelvis_pitch_deg"][0]),
           "com_ahead_of_hip_pct_femur": float(100 * np.dot(z["com"][0] - hip_mid[0], h[0]) / Lf),
           "com_below_hip_pct_femur": float(100 * (hip_mid[0, 2] - z["com"][0, 2]) / Lf),
           "total_mass_kg": float(m.body_subtreemass[pel])}
    s = "r"
    out["femur_from_vertical_deg"] = float(signed_from_vertical(z["r_hip"], z["r_knee"], h)[0])
    out["tibia_from_vertical_deg"] = float(signed_from_vertical(z["r_knee"], z["r_ankle"], h)[0])
    out["metatarsus_from_vertical_deg"] = float(signed_from_vertical(z["r_ankle"], z["r_mtp"], h)[0])
    out["knee_included_deg"] = float(included(z["r_hip"], z["r_knee"], z["r_ankle"])[0])
    out["ankle_included_deg"] = float(included(z["r_knee"], z["r_ankle"], z["r_mtp"])[0])
    out["mtp_included_deg"] = float(included(z["r_ankle"], z["r_mtp"], z["r_d3_axis_end"])[0])
    pf = z[f"{s}_pad_fwd"][0].copy()
    pf[2] = 0
    pf /= np.linalg.norm(pf)
    padc = z[f"{s}_pad_c"][0]
    rear = padc - PAD_HALF * pf
    flen = float(np.dot(z[f"{s}_d3_tip"][0] - rear, pf))
    out["footprint_len_m"] = flen
    out["pad_centre_footfrac"] = float(PAD_HALF / flen)
    out["mtp_footfrac"] = float(np.dot(z["r_mtp"][0] - rear, pf) / flen)
    out["mtp_padstat"] = float(np.dot(z["r_mtp"][0] - padc, pf) / PAD_HALF)
    padmid = 0.5 * (z["r_pad_c"][0] + z["l_pad_c"][0])
    out["com_ahead_of_padmid_padstat"] = float(np.dot(z["com"][0] - padmid, h[0]) / PAD_HALF)
    out["com_footfrac"] = float(np.dot(z["com"][0] - rear, pf) / flen)
    out["knee_ahead_of_padcentre_m"] = float(np.dot(z["r_knee"][0] - padc, h[0]))
    out["knee_ahead_of_padcentre_pct_femur"] = 100 * out["knee_ahead_of_padcentre_m"] / Lf
    out["hip_ahead_of_ankle_m"] = float(np.dot(z["r_hip"][0] - z["r_ankle"][0], h[0]))
    out["hip_height_m"] = float(hip_mid[0, 2])
    out["com_height_m"] = float(z["com"][0, 2])
    return out


def agg(vals):
    a = np.array([v for v in vals if v is not None and math.isfinite(v)], dtype=float)
    if not a.size:
        return {"mean": None, "sd": None, "min": None, "max": None, "n": 0}
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rolls", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.rolls, "meta.json")))
    episodes = [json.loads(line) for line in open(os.path.join(args.rolls, "episodes.jsonl"))]
    per_ep = []
    femur_lens = []
    for ep in episodes:
        z = dict(np.load(os.path.join(args.rolls, f"steps_{ep['target']}_{ep['seed']}.npz")))
        ps = per_step(z)
        femur_lens.append((float(ps["femur_len_m"].min()), float(ps["femur_len_m"].max())))
        T = len(ps["pelvis_pitch_deg"])
        red = reduce_episode(ps, T)
        per_ep.append({"target": ep["target"], "seed": ep["seed"], "length": T,
                       "clean_nonstatue_criteria": ep["clean_nonstatue_criteria"],
                       "max_sole_cop_fore_aft": ep["max_sole_cop_fore_aft"], "windows": red,
                       "repo_pad_telemetry": ep["pad_telemetry_gate_window"]})
    # consistency checks: the recorder's gate statistic per foot vs pad_telemetry (w200 is the gate window)
    checks = {"femur_len_range_m": [min(a for a, _ in femur_lens), max(b for _, b in femur_lens)]}
    worst = 0.0
    for e in per_ep:
        for s in "rl":
            worst = max(worst, abs(e["windows"]["w200"][f"{s}_rec_pad_cop_gate"] - e["repo_pad_telemetry"][f"{s}_sole_cop_fore_aft_gate"]))
    checks["max_abs_diff_gate_stat_vs_repo_pad_telemetry"] = worst
    worst_ds = 0.0
    for e in per_ep:
        for s in "rl":
            worst_ds = max(worst_ds, abs(e["windows"]["w200"][f"{s}_digit_share"] - e["repo_pad_telemetry"][f"{s}_digit_share"]))
    checks["max_abs_diff_digit_share_vs_repo_pad_telemetry"] = worst_ds
    # whole-foot CoP: probe (boundary) vs recorder foot_cop (substep pooled), as pad statistic
    diffs = []
    for e in per_ep:
        for s in "rl":
            rec = (e["windows"]["w200"][f"{s}_rec_foot_cop_m_ahead_ankle"] - 0.14) / PAD_HALF
            diffs.append(abs(rec - e["windows"]["w200"][f"{s}_wholefoot_cop_padstat"]))
    checks["max_abs_diff_wholefoot_cop_probe_vs_recorder_padstat"] = float(max(diffs))
    targets = sorted({e["target"] for e in per_ep}, key=["statue", "s60", "s52"].index)
    summary = {"meta": meta, "checks": checks, "home_keyframe": home_reference(meta["mjcf"]), "targets": {}}
    keys = sorted(per_ep[0]["windows"]["w200"].keys())
    for t in targets:
        eps = [e for e in per_ep if e["target"] == t]
        summary["targets"][t] = {
            "n_episodes": len(eps), "seeds": [e["seed"] for e in eps],
            "full_horizon": sum(e["length"] >= meta["horizon"] for e in eps),
            "clean_nonstatue_criteria": sum(bool(e["clean_nonstatue_criteria"]) for e in eps),
            "windows": {w: {k: agg([e["windows"][w][k] for e in eps]) for k in keys} for w in WINDOWS},
        }
    os.makedirs(args.out, exist_ok=True)
    json.dump(per_ep, open(os.path.join(args.out, "posture_episodes.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(args.out, "posture_summary.json"), "w"), indent=1)
    print(json.dumps({"checks": checks, "home": summary["home_keyframe"]}, indent=1))


if __name__ == "__main__":
    main()
