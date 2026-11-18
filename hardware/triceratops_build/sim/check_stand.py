"""Four-foot standing check for the Triceratops build models (task 3). Rules: PREREG_STAND.md.

Usage: python check_stand.py [--out results/stand_<date>]
Writes results.json and RESULTS.md. Simulation only; nothing here is measured.
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
GEN = HERE / "generated"
CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
TOTALS = {("T20", "listed"): 18.83, ("T15", "listed"): 14.61, ("T20", "closed"): 20.00, ("T15", "closed"): 15.00}
PARTS_FRONT = {"T20": 0.552, "T15": 0.536}
STAND_REF = {"T20": {"front_flexion": 8.1, "roll": 4.3}, "T15": {"front_flexion": 6.1, "roll": 3.2}}
REF_MASS = {"T20": 20.0, "T15": 15.0}
SEEDS_JUDGED = (1, 2)
SEEDS = (0,) + SEEDS_JUDGED
T_RUN, T_WIN = 10.0, 1.0


def run(build, variant, seed):
    path = GEN / f"trike_{build}_{variant}.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    if seed:
        rng = np.random.default_rng(seed)
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                data.qpos[model.jnt_qposadr[j]] += rng.uniform(-0.02, 0.02)
        data.qpos[2] += 0.005
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    weight = -model.opt.gravity[2] * model.body_mass.sum()
    root = model.body("chassis").id
    h0 = data.xpos[root, 2]
    n = round(T_RUN / model.opt.timestep)
    nw = round(T_WIN / model.opt.timestep)
    touch = np.flatnonzero(model.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH)
    foot_of = [model.body(model.site_bodyid[model.sensor_objid[t]]).name.replace("_foot", "") for t in touch]
    min_h, max_tilt, finite = 1e9, 0.0, True
    acc_load, acc_frc, acc_v2 = np.zeros(len(touch)), np.zeros(model.nu), 0.0
    for k in range(n):
        mujoco.mj_step(model, data)
        h = data.xpos[root, 2]
        min_h = min(min_h, h / h0)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, data.xmat[root, 8]))))
        max_tilt = max(max_tilt, tilt)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            finite = False
            break
        if k >= n - nw:
            # sensordata and actuator_force come from the same forward pass inside mj_step
            acc_load += data.sensordata[model.sensor_adr[touch]]
            acc_frc += data.actuator_force
            acc_v2 += float(np.sum(data.qvel[:3] ** 2))
    loads = acc_load / nw
    frc = acc_frc / nw
    vrms = math.sqrt(acc_v2 / nw)
    lim = model.actuator_forcerange[:, 1]
    names = [model.actuator(a).name for a in range(model.nu)]
    com = data.subtree_com[root]
    return {
        "build": build, "variant": variant, "seed": seed, "mass_kg": float(model.body_mass.sum()),
        "weight_n": float(weight), "min_height_ratio": float(min_h), "max_tilt_deg": float(max_tilt),
        "finite": finite, "warnings": int(np.sum(data.warning.number)), "root_speed_rms_mps": vrms,
        "foot_load_n": {f: float(l) for f, l in zip(foot_of, loads)},
        "com_m": [float(x) for x in com],
        "actuator_force_nm": {nm: float(f) for nm, f in zip(names, frc)},
        "actuator_limit_nm": {nm: float(l) for nm, l in zip(names, lim)},
    }


def judge(r, ledger):
    b, v = r["build"], r["variant"]
    w = r["weight_n"]
    L = r["foot_load_n"]
    tot = sum(L.values())
    front = (L["left_fore"] + L["right_fore"]) / tot
    left = (L["left_fore"] + L["left_hind"]) / tot
    f = r["actuator_force_nm"]
    out = {}
    cats_ok = True
    if v == "listed":
        cats_ok = all(abs(ledger["category_kg"][k] - ledger["budget_category_kg"][k]) <= 0.01
                      for k in ledger["budget_category_kg"])
    out["R1"] = abs(r["mass_kg"] - TOTALS[(b, v)]) <= 0.01 and cats_ok
    out["R2"] = (r["min_height_ratio"] >= 0.90 and r["max_tilt_deg"] <= 5.0 and r["finite"]
                 and r["warnings"] == 0 and r["root_speed_rms_mps"] < 0.005)
    out["R3"] = all(l >= 0.10 * w for l in L.values()) and abs(tot - w) <= 0.02 * w
    out["R4a"] = 0.50 <= front <= 0.60
    out["R4b_flag_ok"] = abs(front - PARTS_FRONT[b]) <= 0.010
    out["R4c"] = abs(left - 0.50) <= 0.02
    over = []
    for c in CORNERS:
        for axis in ("roll", "pitch", "flexion"):
            nm = f"{c}_{axis}"
            if abs(f[nm]) > ledger["drives"][nm]["joint_thermal_eq_nm"]:
                over.append(nm)
    out["R5a"] = not over
    scale = 1.0 if v == "closed" else r["mass_kg"] / REF_MASS[b]
    ref = STAND_REF[b]
    ff = max(abs(f["left_fore_flexion"]), abs(f["right_fore_flexion"]))
    rr = max(abs(f[f"{c}_roll"]) for c in CORNERS)
    out["R5b_flag_ok"] = ff <= 1.15 * ref["front_flexion"] * scale and rr <= 1.15 * ref["roll"] * scale
    sat = [nm for nm in f if abs(f[nm]) >= 0.99 * r["actuator_limit_nm"][nm]]
    out["R6"] = not sat
    out["R7"] = abs(f["neck_pitch"]) <= 3.11 and (b != "T20" or abs(f["tail_pitch"]) <= 1.20)
    derived = {"front_share": front, "left_share": left, "foot_sum_over_weight": tot / w,
               "front_flexion_max_nm": ff, "roll_max_nm": rr,
               "front_flexion_ref_nm": ref["front_flexion"] * scale, "roll_ref_nm": ref["roll"] * scale,
               "hind_flexion_max_nm": max(abs(f["left_hind_flexion"]), abs(f["right_hind_flexion"])),
               "pitch_max_nm": max(abs(f[f"{c}_pitch"]) for c in CORNERS),
               "over_thermal": over, "saturated": sat}
    return out, derived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "stand_2026-11-18"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, verdicts = [], {}
    for b in ("T20", "T15"):
        for v in ("listed", "closed"):
            ledger = json.loads((GEN / f"mass_ledger_{b}_{v}.json").read_text())
            ok_all = True
            for s in SEEDS:
                r = run(b, v, s)
                rules, der = judge(r, ledger)
                r["rules"], r["derived"] = rules, der
                results.append(r)
                hard = {k: x for k, x in rules.items() if not k.endswith("flag_ok")}
                if s in SEEDS_JUDGED:
                    ok_all = ok_all and all(hard.values())
            verdicts[f"{b}_{v}"] = "PASS" if ok_all else "FAIL"
    (out / "results.json").write_text(json.dumps({"mujoco": mujoco.__version__, "verdicts": verdicts,
                                                  "runs": results}, indent=1) + "\n")
    lines = ["# Standing check results (simulation)", "",
             f"Generated by `check_stand.py`, mujoco {mujoco.__version__}. Rules: `PREREG_STAND.md`. "
             "Seeds 1 and 2 are judged; seed 0 is the exact neutral pose. Torques are joint-side, N·m, means over "
             "the last 1 s of 10 s.", "",
             "| Build | Variant | Seed | Mass kg | Front % | Left % | Feet LF / RF / LH / RH, N | Front elbow | Hind knee | Roll max | Pitch max | Neck | Rules failed | Flags |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        d, ru, L, f = r["derived"], r["rules"], r["foot_load_n"], r["actuator_force_nm"]
        failed = [k for k, x in ru.items() if not k.endswith("flag_ok") and not x]
        flags = [k.replace("_flag_ok", "") for k, x in ru.items() if k.endswith("flag_ok") and not x]
        lines.append(f"| {r['build']} | {r['variant']} | {r['seed']} | {r['mass_kg']:.2f} | {100 * d['front_share']:.1f} | "
                     f"{100 * d['left_share']:.1f} | {L['left_fore']:.1f} / {L['right_fore']:.1f} / {L['left_hind']:.1f} / "
                     f"{L['right_hind']:.1f} | {d['front_flexion_max_nm']:.2f} (ref {d['front_flexion_ref_nm']:.2f}) | "
                     f"{d['hind_flexion_max_nm']:.2f} | {d['roll_max_nm']:.2f} (ref {d['roll_ref_nm']:.2f}) | "
                     f"{d['pitch_max_nm']:.2f} | {abs(f['neck_pitch']):.2f} | {', '.join(failed) or 'none'} | "
                     f"{', '.join(flags) or 'none'} |")
    lines += ["", "Verdicts (pre-registered rule set, seeds 1 and 2):", ""]
    lines += [f"- {k}: **{v}**" for k, v in verdicts.items()]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
