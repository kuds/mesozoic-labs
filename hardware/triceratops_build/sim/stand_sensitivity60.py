"""Standing balance and design sensitivity, 60 cm model (simulation; reported, not judged).

How the front/rear load share and the standing drive loads move with (A) the battery's fore-aft position and
(B) the total mass (the listed and closed variants, and the closed model with the 0.49 kg it is over the budget's
12.2 kg removed or added again as payload). Each point is the model built by build_model60.py (revision 3) with one
change, held at the neutral pose with ctrl = 0 for 10 s at the 1 ms step (seed 0, as check_stand.py; the judged seeds
settle to the same pose); values are means over the last second. Positions that would put the battery into another
part (the packaging check's box gaps) are marked, not left out.

Also writes an ESTIMATE, labelled so: the first learned gait's worst-joint heat (rl/analysis/REPORT_GAIT_PIPE1.md,
RMS torque / continuous rating, mean of episodes) scaled by mass ratio to each mass point, on the assumption that
walking torque scales with weight. That is a scaling, not a simulation of a gait at that mass.

Usage: python stand_sensitivity60.py [--out results/stand_sensitivity60_<date>]
"""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import build_model as bm
import build_model60 as b60
from check_layout60 import body_aabb, box_gap

HERE = Path(__file__).resolve().parent
CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
S = b60.S
BATTERY_X = (-0.200, -0.180, -0.160, -0.140, -0.120, -0.100, -0.080)   # chassis frame, m; -0.160 is revision 3's
BUDGET_KG = 12.2
PIPE1_HEAT = {"left_hind_flexion": 1.10, "fore_pitch": 1.02}   # REPORT_GAIT_PIPE1.md, mean of 5 episodes, at 12.694 kg
GEN_MASS = 12.694


def model_text(variant, battery_x=-0.200 * S, payload_kg=0.0):
    """build_model60.build plus main()'s neck-spring step, as text; optional payload over the chassis origin."""
    m, _, _ = b60.build(variant, False, battery_x)
    if payload_kg:
        chassis = m.body("chassis")
        m.add_geom(chassis, name="sensitivity_payload", type="box", pos=[0.0, 0.0, 0.04], size="0.03 0.03 0.01",
                   material="power", mass=f"{payload_kg:.4f}")
    ET.indent(m.tree, space="  ")
    text = ET.tostring(m.root, encoding="unicode")
    ref, _ = bm.set_neck_spring(text)
    nj = next(j for j in m.root.iter("joint") if j.get("name") == "neck_pitch")
    nj.set("stiffness", "0.40")
    nj.set("springref", f"{math.degrees(ref * 0.70 / 0.40):.3f}")
    return ET.tostring(m.root, encoding="unicode")


def scale_mass(text, factor):
    """Every geom mass x factor (a uniform lighter or heavier build; the centre of mass does not move)."""
    root = ET.fromstring(text)
    for g in root.iter("geom"):
        if g.get("mass") and float(g.get("mass")) > 0:
            g.set("mass", f"{float(g.get('mass')) * factor:.6f}")
    return ET.tostring(root, encoding="unicode")


def stand(text):
    model = mujoco.MjModel.from_xml_string(text)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    touch = np.flatnonzero(model.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH)
    foot = [model.body(model.site_bodyid[model.sensor_objid[t]]).name.replace("_foot", "") for t in touch]
    n, nw = round(10.0 / model.opt.timestep), round(1.0 / model.opt.timestep)
    acc_l, acc_f = np.zeros(len(touch)), np.zeros(model.nu)
    for k in range(n):
        mujoco.mj_step(model, data)
        if k >= n - nw:
            acc_l += data.sensordata[model.sensor_adr[touch]]
            acc_f += data.actuator_force
    L = dict(zip(foot, acc_l / nw))
    F = {model.actuator(i).name: float(f) for i, f in enumerate(acc_f / nw)}
    tot = sum(L.values())
    ch = model.body("chassis").id
    # packaging: the battery boxes against every other solid part on the chassis (frame-to-frame skipped)
    mujoco.mj_forward(model, data)
    bats = [g for g in range(model.ngeom) if model.geom(g).name.startswith("gtech_6s_3300")]
    skip = ("PA11_rib", "shell_seam", "loom", "oak_d_usb_c_lock_cable", "sensitivity_payload")
    others = [g for g in range(model.ngeom) if model.geom_bodyid[g] == ch and g not in bats
              and not model.geom(g).name.endswith(("_contact", "_collision"))
              and not any(k in model.geom(g).name for k in skip)]

    def gap(b, o):   # as check_layout60.py: exact box gaps for box pairs, mj_geomDistance otherwise
        if model.geom_type[o] == mujoco.mjtGeom.mjGEOM_BOX:
            return box_gap(body_aabb(model, b), body_aabb(model, o))
        return mujoco.mj_geomDistance(model, data, b, o, 0.2, None)
    gaps = [(gap(b, o), model.geom(o).name) for b in bats for o in others]
    worst = min(gaps)
    return {"mass_kg": float(model.body_mass.sum()), "front_share": (L["left_fore"] + L["right_fore"]) / tot,
            "left_share": (L["left_fore"] + L["left_hind"]) / tot,
            "foot_load_n": {k: float(v) for k, v in L.items()},
            "front_elbow_nm": max(abs(F["left_fore_flexion"]), abs(F["right_fore_flexion"])),
            "hind_knee_nm": max(abs(F["left_hind_flexion"]), abs(F["right_hind_flexion"])),
            "pitch_max_nm": max(abs(F[f"{c}_pitch"]) for c in CORNERS),
            "roll_max_nm": max(abs(F[f"{c}_roll"]) for c in CORNERS),
            "battery_min_gap_mm": 1000 * worst[0], "battery_tightest_part": worst[1],
            "battery_intersects": bool(worst[0] < -1e-4)}


def chart(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf, ink, ink2, blue, orange = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "text.color": ink})
    xs = [1000 * r["battery_x_m"] for r in rows]
    bad = [r["battery_intersects"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=110, facecolor=surf)
    for ax in axes:
        ax.set_facecolor(surf)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axvline(-160, color=ink2, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xlabel("Battery centre, fore-aft (mm, chassis frame; + is forward)")
    ax = axes[0]
    ax.axhspan(50, 60, color="#e9f1fb", zorder=0)
    ys = [100 * r["front_share"] for r in rows]
    ax.plot(xs, ys, color=blue, linewidth=2, marker="o", markersize=6, markeredgecolor=surf, markeredgewidth=2)
    for x, y, b in zip(xs, ys, bad):
        if b:
            ax.plot([x], [y], marker="x", color=ink, markersize=8, markeredgewidth=2)
    ax.set_ylabel("Weight on the front feet (%)")
    ax.set_title("Front share of the weight, standing", loc="left", fontsize=11, color=ink)
    ax.text(-158, ax.get_ylim()[0] + 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "revision 3", color=ink2, fontsize=9)
    ax.text(xs[0], 59.3, "design range 50-60 %", color=ink2, fontsize=9)
    ax = axes[1]
    e = [r["front_elbow_nm"] for r in rows]
    k = [r["hind_knee_nm"] for r in rows]
    ax.plot(xs, e, color=blue, linewidth=2, marker="o", markersize=6, markeredgecolor=surf, markeredgewidth=2,
            label="front elbow (X6)")
    ax.plot(xs, k, color=orange, linewidth=2, marker="o", markersize=6, markeredgecolor=surf, markeredgewidth=2,
            label="hind knee (X4)")
    ax.text(xs[-1] + 3, e[-1], "front elbow", color=ink, fontsize=9, va="center")
    ax.text(xs[-1] + 3, k[-1], "hind knee", color=ink, fontsize=9, va="center")
    ax.set_ylabel("Standing torque, worst side (N·m)")
    ax.set_title("Standing load on the flexion drives", loc="left", fontsize=11, color=ink)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_xlim(xs[0] - 8, xs[-1] + 40)
    ax.set_ylim(0, 1.2 * max(e))   # from zero: the change across the sweep is small and the chart should show that
    fig.text(0.01, 0.01, "SIMULATION  ·  60 cm model, revision 3, closed variant (12.69 kg)  ·  × = battery would "
             "intersect another part", color=ink2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, facecolor=surf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "stand_sensitivity60_2026-11-19"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"model": "build_model60.py revision 3", "battery_sweep": [], "mass": []}
    for bx in BATTERY_X:
        r = stand(model_text("closed", battery_x=bx))
        r["battery_x_m"] = bx
        res["battery_sweep"].append(r)
        print(f"battery x {1000 * bx:+.0f} mm: front {100 * r['front_share']:.1f} %, elbow {r['front_elbow_nm']:.2f}, "
              f"knee {r['hind_knee_nm']:.2f} N·m, battery gap {r['battery_min_gap_mm']:+.1f} mm "
              f"({r['battery_tightest_part']})")
    closed = model_text("closed")
    cases = [("listed (no reserve)", model_text("listed")), ("closed", closed),
             ("closed scaled to the budget's 12.2 kg", scale_mass(closed, BUDGET_KG / GEN_MASS)),
             ("closed + 0.5 kg payload at the chassis", model_text("closed", payload_kg=0.5))]
    for label, text in cases:
        r = stand(text)
        r["case"] = label
        ratio = r["mass_kg"] / GEN_MASS
        r["ESTIMATE_pipe1_heat_scaled_by_mass"] = {k: round(v * ratio, 3) for k, v in PIPE1_HEAT.items()}
        res["mass"].append(r)
        print(f"{label}: {r['mass_kg']:.3f} kg, front {100 * r['front_share']:.1f} %, elbow {r['front_elbow_nm']:.2f}, "
              f"knee {r['hind_knee_nm']:.2f} N·m; estimate (pipe1 gait x mass ratio): "
              f"{r['ESTIMATE_pipe1_heat_scaled_by_mass']}")
    (out / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    chart(res["battery_sweep"], out / "battery_position.png")
    md = ["# Standing balance and design sensitivity, 60 cm (simulation)", "",
          "Written by `stand_sensitivity60.py`. Standing at the neutral pose, 10 s, means over the last second. "
          "Reported, not judged.", "",
          "## Battery position (closed variant)", "",
          "| Battery centre x | Front share | Left share | Front elbow | Hind knee | Pitch max | Roll max | Battery's tightest gap |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["battery_sweep"]:
        md.append(f"| {1000 * r['battery_x_m']:+.0f} mm | {100 * r['front_share']:.1f} % | {100 * r['left_share']:.1f} % | "
                  f"{r['front_elbow_nm']:.2f} N·m | {r['hind_knee_nm']:.2f} N·m | {r['pitch_max_nm']:.2f} N·m | "
                  f"{r['roll_max_nm']:.2f} N·m | {r['battery_min_gap_mm']:+.1f} mm ({r['battery_tightest_part']})"
                  f"{' INTERSECTS' if r['battery_intersects'] else ''} |")
    md += ["", "## Mass", "",
           "| Case | Mass | Front share | Front elbow | Hind knee | ESTIMATE: pipe1 gait's heat x mass ratio (left hind knee / fore pitch) |",
           "|---|---|---|---|---|---|"]
    for r in res["mass"]:
        est = r["ESTIMATE_pipe1_heat_scaled_by_mass"]
        md.append(f"| {r['case']} | {r['mass_kg']:.3f} kg | {100 * r['front_share']:.1f} % | {r['front_elbow_nm']:.2f} N·m | "
                  f"{r['hind_knee_nm']:.2f} N·m | {est['left_hind_flexion']:.2f} / {est['fore_pitch']:.2f} |")
    md += ["", "The last column scales the first learned gait's measured heat (mean of episodes) by mass; it assumes walking "
           "torque scales with weight and is not a simulation of a gait at that mass.", "",
           "Chart: `battery_position.png`."]
    (out / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
