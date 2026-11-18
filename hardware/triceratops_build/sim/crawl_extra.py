"""Report-only quantities for the judged crawl runs (PREREG_CRAWL.md, "Reported but not judged"): re-runs a model
and seed (deterministic; the rule results must match results/crawl_*/), then reports the support margin with the soles'
width counted, the lateral centre of pressure on each sole, the flexion angle range against the 1.3:1 pushrod's
constant-ratio band (-29 to +24 deg; assumption A5), and joint peak / RMS torque and peak speed.

Usage: python crawl_extra.py --out results/crawl_2026-11-18/extra.json   Simulation only.
"""
import argparse, json, math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import mujoco, numpy as np
import crawl

HERE = Path(__file__).resolve().parent
MODELS = ("trike_T20_closed", "trike_T15_closed", "trike_T20_closed_ankleroll", "trike_T15_closed_ankleroll")


def hull_margin(p, pts):
    pts = sorted(map(tuple, pts))
    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and crawl.cross2(np.subtract(h[-1], h[-2]), np.subtract(q, h[-2])) <= 0:
                h.pop()
            h.append(q)
        return h
    poly = np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
    m = 1e9
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        t = b - a
        m = min(m, crawl.cross2(t, p - a) / np.linalg.norm(t))
    return m


def one(job):
    name, seed = job
    P = crawl.PARAMS
    model = mujoco.MjModel.from_xml_path(str(HERE / "generated" / f"{name}.xml"))
    kin, steps = crawl.plan(model, P)
    times, targets, phases = crawl.trajectory(kin, steps, P)
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    crawl.apply_seed(model, d, seed)
    mujoco.mj_forward(model, d)
    act = [model.actuator(f"{c}_{a}").id for c in crawl.CORNERS for a in crawl.AXES]
    sole = {c: model.geom(f"{c}_sole").id for c in crawl.CORNERS}
    foot = {c: model.body(f"{c}_foot").id for c in crawl.CORNERS}
    flex = {c: model.jnt_qposadr[model.joint(f"{c}_flexion").id] for c in crawl.CORNERS}
    ch = model.body("chassis").id
    swing_of = lambda t: next((p["leg"] for p in phases if p["swing_start"] <= t <= p["swing_end"]), None)
    every = round(P["control_dt"] / model.opt.timestep)
    sole_margin, cop = [], {c: [] for c in crawl.CORNERS}
    fl = {c: [1e9, -1e9] for c in crawl.CORNERS}
    for k in range(round(times[-1] / model.opt.timestep)):
        t = k * model.opt.timestep
        d.ctrl[act] = [np.interp(t, times, targets[:, i]) for i in range(12)]
        mujoco.mj_step(model, d)
        if k % every:
            continue
        for c in crawl.CORNERS:
            q = math.degrees(d.qpos[flex[c]])
            fl[c] = [min(fl[c][0], q), max(fl[c][1], q)]
        # lateral centre of pressure on each loaded sole, in the sole's frame (mm, + = outboard)
        for c in crawl.CORNERS:
            F, M = 0.0, 0.0
            for i in range(d.ncon):
                con = d.contact[i]
                if sole[c] not in (con.geom1, con.geom2):
                    continue
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, d, i, f6)
                R = d.geom_xmat[sole[c]].reshape(3, 3)
                local = R.T @ (con.pos - d.geom_xpos[sole[c]])
                F += f6[0]
                M += f6[0] * local[1]
            if F > 5.0:
                side = 1 if c.startswith("left") else -1
                cop[c].append(1000 * side * M / F)
        L = swing_of(t)
        if L is not None:
            pts = []
            for c in crawl.CORNERS:
                if c == L:
                    continue
                R = d.geom_xmat[sole[c]].reshape(3, 3)
                h = model.geom_size[sole[c]]
                for sy in (-1, 1):
                    pts.append((d.xpos[foot[c]] + R @ np.array([0, sy * h[1], 0]))[:2])
            sole_margin.append(hull_margin(d.subtree_com[ch][:2], pts))
    r = crawl.run(HERE / "generated" / f"{name}.xml", seed)
    rules, info = crawl.judge(r)
    ratio13 = {c: name.startswith("trike_T15") or "hind" in c for c in crawl.CORNERS}
    return {"model": name, "seed": seed, "rules": rules,
            "min_margin_sole_width_basis_mm": round(1000 * min(sole_margin), 1),
            "cop_outboard_mm": {c: [round(float(np.percentile(v, 5)), 1), round(float(np.median(v)), 1),
                                    round(float(np.percentile(v, 95)), 1)] for c, v in cop.items()},
            "flexion_range_deg": {c: [round(a, 1), round(b, 1)] for c, (a, b) in fl.items()},
            "flexion_in_constant_ratio_band": all((not ratio13[c]) or (fl[c][0] >= -29 and fl[c][1] <= 24)
                                                  for c in crawl.CORNERS),
            "joint_peak_nm": info["joint_peak_nm"], "joint_rms_nm": info["joint_rms_nm"],
            "joint_peak_rpm": info["joint_peak_rpm"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "crawl_2026-11-18" / "extra.json"))
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(one, [(m, s) for m in MODELS for s in (1, 2)]))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    for r in res:
        print(r["model"], r["seed"], "rules", "all pass" if all(r["rules"].values()) else [k for k, v in r["rules"].items() if not v],
              "| sole-width margin", r["min_margin_sole_width_basis_mm"], "mm | flexion band ok", r["flexion_in_constant_ratio_band"],
              "| flex range", r["flexion_range_deg"], "| CoP p5/med/p95 LF", r["cop_outboard_mm"]["left_fore"], "LH", r["cop_outboard_mm"]["left_hind"])


if __name__ == "__main__":
    main()
