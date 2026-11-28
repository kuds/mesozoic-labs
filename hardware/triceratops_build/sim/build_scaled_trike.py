"""Scaled copies of the adopted 60 cm Triceratops model, and the runs that test the form-factor study's torque scaling
rule on them (the study's Simulate phase, 28 Nov 2026). SIMULATION ONLY: nothing here is measured.

THE MODEL. ``build`` writes a copy of ``generated/trike60_C_closed_capsule_fore128.xml`` (read, never written) at a
crown height H, with s = H / 0.60 m:
- every length and position x s: body, geom, site and camera positions, geom and site ``fromto`` and ``size`` (the floor
  plane excepted), the meshes' ``scale``, the ``statistic``; angles, joint ranges and control ranges unchanged;
- masses from the study's mass model for that configuration (``design/form_factor_study.py`` ``mass_model`` with no
  layout additions and no battery cut-off: the model as built), distributed as the 60 cm ledger is: each geom's mass x
  (its ledger row's mass at H / at 60 cm); inertia from the scaled geoms (``inertiafromgeom``, as the source model);
- leg drives from the chosen drive set: force limit 0.70 x maker peak (x 0.90 at the flexion joints' 1:1 linkages),
  armature the catalog's (``data/drives.json``), continuous rating the equilibrium torque (x 0.90 at flexion), the
  source model's rules (``build_model60.py`` ``AXIS_DRIVE``, ``rl/trike_env.py`` ``cont_torque_*``);
- leg servo gains scaled so the stance stiffness per weight matches in the dynamically similar sense: kp / (m g L)
  kept, i.e. kp x k_kp with k_kp = (m / 12.7506 kg) x s, the joint's sag under its share of the weight then the same
  angle at every size (the build's own similarity rule for the ankle springs, ``build_model60.py:62-66``: torque per
  radian proportional to weight x length); kv x k_kp x sqrt(s) (the same damping ratio on a link inertia ~ m L^2).
  ``--kp-rule literal`` instead keeps the foot stiffness per weight (kp / L^2) / (m g): kp x (m / m60) x s^2, kv by
  the same factor (a sensitivity case);
- the passive ankle springs by the build's rule (stiffness x (m / m60) x s, damping also x sqrt(s));
- the neck spring and the neck and tail servo gains x the ratio of the head's (tail's) gravity moment at the neutral
  pose, scaled to source (the spring keeps cancelling 70 % of it, ``build_model.set_neck_spring``; kv also x
  sqrt(s)); servo force limits by the budget's size class (XC430 / XL430 up to 67.5 cm, XM540-W270 at neck pitch and
  XM430-W350 elsewhere above it: ``budget/size_budget.py:124-135`` as the study's ``servo_set``; 0.70 x the listed
  peak, ``data/drives.json`` servos);
- unchanged: the 1 ms timestep and solver, contact parameters, joint friction damping, keyframes (empty: the scaled
  qpos0), sensors.
``--kp-rule adopted`` keeps the adopted A2 gains (kp 598.5 N m/rad and its kv, ``DECISION_LOG.md` (F28)`) unchanged at
every size: what a real robot of another size with today's drives would carry unless the owner decides otherwise
(added 28 Nov 2026 after review, F3). ``--layout L1`` / ``L2`` (added 28 Nov 2026 after review, MECH-2) moves each
leg's roll drive, yoke, pitch drive, upper link and knee drive to the study's desk layout
(``design/form_factor_study.py`` ``make_layout``: the lateral offsets of the roll drive and of the link plane, and the
knee drive's place, at L2 on the pitch axis outboard of the link), masses unchanged (the model as built), so that the
torque rule's missing layout term can be measured; no layout additions (offset mounts, longer rods) are added.
What does NOT scale, as on a real robot of another size with the same parts: the drives' rotor inertia (armature) and
force limits, the servo delay, the 100 Hz control rate, the contact model, the sensor noise. A smaller or larger
drive's geometry is not redrawn: its geoms are the source's x s (its mass is the catalog's).

THE RUNS (pre-registered by the study's lead before any run; the implementation choices are in
``results/form_factor_2026-11-27/scaled_trike_plan.json``, written before the first run):
- S1 stand: 10 s at zero action (targets at the neutral pose) on a flat floor; every foot above 15 % of the weight.
- S2 trot: ``walk/walk60.py`` ``TrotController`` (the committed model-based trot) with Froude-scaled settings (times x
  sqrt(s), lengths x s, so the period goes as sqrt(s) and the stride as s), commanded 0.25 x sqrt(s) m/s, on
  ``walk60.plant_config`` with this model (10 ms delay, noise on, nominal physics factors, friction 0.8), 3 seeds x 20
  s; each leg joint's RMS (2-20 s) against k x the 0.60 m run's, k = (m / 12.7506 kg) x s (the study's rule); the rule
  holds at a size if every leg joint is within +-25 % on at least 2 of 3 seeds.

Usage (from hardware/triceratops_build):
  python sim/build_scaled_trike.py build --size 0.50 [--drives D0] [--kp-rule similar] --out DIR
  python sim/build_scaled_trike.py run --models DIR --out RESULTS_DIR [--cases H0.50 H0.60 ...] [--workers 2]
  python sim/build_scaled_trike.py compare --out RESULTS_DIR
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
GEN = HERE / "generated"
SRC = GEN / "trike60_C_closed_capsule_fore128.xml"
LEDGER = GEN / "mass_ledger_trike60_C_closed_capsule_fore128.json"
DRIVES_JSON = HERE / "data" / "drives.json"
RESULTS = HERE / "results" / "form_factor_2026-11-27"
for _p in (BUILD / "design", BUILD / "rl", BUILD / "walk"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import form_factor_study as FF  # noqa: E402  (read-only: the rule under test, its mass model and drive figures)

S60 = FF.S60
M60 = FF.M60_SIM_KG          # 12.7506 kg, the simulated model's mass (the rule's reference)
CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
LEG_AXES = ("roll", "pitch", "flexion")
LEG_JOINTS = tuple(f"{c}_{a}" for c in CORNERS for a in LEG_AXES)
PEAK_SHARE = 0.70            # design peak = 0.70 x maker peak (data/drives.json _about; build_model60.py)
ETA = FF.ETA                 # 0.90 at the flexion joints' 1:1 linkages
NECK_SPRING_FRACTION = 0.70  # build_model.set_neck_spring
SERVO_LIMIT_SHARE = 0.70     # build_model60.py SERVO60: position-servo force limit = 0.70 x the servo's torque figure
SOURCE_SERVO_LIMITS = {"neck_pitch": 1.33, "neck_yaw": 1.33, "tail_yaw": 1.05, "tail_pitch": 1.05}  # the source XML
BIG_SERVO = {"neck_pitch": "XM540-W270-R", "neck_yaw": "XM430-W350-R", "tail_yaw": "XM430-W350-R",
             "tail_pitch": "XM430-W350-R"}  # FF.servo_set's step above 67.5 cm (budget/size_budget.py:124-135)
COMMAND60 = 0.25             # m/s, the trot's judged command at 60 cm (walk/PREREG_WALK60.md)
FRICTION = 0.8               # S2's floor
DELAY_MS = 10                # D9 (walk/PREREG_WALK60.md), physics steps at 1 ms
VERDICT_SEEDS = (101, 102, 103)  # S1 and S2 are judged on these: the trot's judged seeds (walk/PREREG_WALK60.md)
SEEDS = tuple(range(101, 111))   # 104-110 also run, as supporting scatter only (results/.../scaled_trike_plan.json)
STAND_S = 10.0
S2_BAND = 0.25
S1_LOAD = 0.15


def _fmt(x: float) -> str:
    """Shortest round-trip text of a float (x 1.0 gives back the source's own value exactly)."""
    return repr(float(x))


def _scale_text(text: str, f: float) -> str:
    return " ".join(_fmt(float(v) * f) for v in text.split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def drive_catalog() -> dict:
    return json.loads(DRIVES_JSON.read_text())


# --------------------------------------------------------------------------------------------------------- masses
def part_masses(S: float, dkey: str = "D0") -> dict:
    """Each 60 cm ledger part's mass at crown height S with drive set ``dkey``: the rules of
    ``form_factor_study.mass_model`` with ``layout=None, bms=False`` (the model as built), row by row (the tests check
    the total and the by-class sums against ``mass_model`` itself). Returns {geom: {kg, kg60, cls, body}}."""
    led = FF.load_ledger60()
    dset = FF.DRIVE_SETS[dkey]
    s = S / S60
    dr = FF.drives_of(dset)
    d60 = FF.drives_of(FF.DRIVE_SETS["D0"])
    servos = FF.servo_set(S)
    out = {}
    for row in led["parts"]:
        g, kg0 = row["geom"], row["mass_kg"]
        cls, p = FF.part_class(row)
        end = "fore" if "_fore_" in g else ("hind" if "_hind_" in g else None)
        kg = kg0
        if cls == "drive":
            ax = "roll" if "roll" in g else ("pitch" if "pitch" in g else f"{end}_knee")
            kg = dr[ax]["kg"]
            cls = f"drive: {ax}"
        elif cls == "linkage":
            ax = f"{end}_knee"
            kg = kg0 * dr[ax]["peak"] / d60[ax]["peak"]
        elif cls == "knee drive mount":
            kg = kg0 * dr["fore_knee"]["kg"] / d60["fore_knee"]["kg"]
        elif cls == "roll drive bracket":
            kg = kg0 * dr["roll"]["kg"] / d60["roll"]["kg"]
        elif cls == "servo":
            kg = servos.get(g, kg0)
        elif p:
            kg = kg0 * s ** p
        out[g] = {"kg": kg, "kg60": kg0, "cls": cls, "body": row["body"]}
    return out


def leg_drive_table(dkey: str = "D0") -> dict:
    """Per leg joint: drive name, force limit (joint side), continuous rating (joint side), armature."""
    dset = FF.DRIVE_SETS[dkey]
    cat = drive_catalog()["leg_drives"]
    out = {}
    for j in LEG_JOINTS:
        end = "fore" if "_fore_" in j else "hind"
        ax = "roll" if j.endswith("_roll") else ("pitch" if j.endswith("_pitch") else f"{end}_knee")
        name = dset[ax]
        if name not in cat:
            raise ValueError(f"drive {name} has no armature in {DRIVES_JSON.name}; the builder takes MyActuator drives")
        eta = ETA if j.endswith("_flexion") else 1.0
        d = FF.DRIVES[name]
        out[j] = {"drive": name, "limit_nm": PEAK_SHARE * d["peak"] * eta, "continuous_nm": d["eq"] * eta,
                  "armature_kgm2": float(cat[name]["armature_kgm2"])}
    return out


def factors(S: float, dkey: str = "D0", kp_rule: str = "similar") -> dict:
    """The scale factors of a configuration (see the module docstring)."""
    s = S / S60
    m = FF.mass_model(S, FF.DRIVE_SETS[dkey], layout=None, bms=False)["total_kg"]
    # gains are scaled against the same mass model's 60 cm figure (12.7504 kg, the ledger rows' sum), so the 0.60 m
    # copy is the source exactly; the rule's k uses the study's 12.7506 kg (0.002 % apart)
    m_ref = FF.mass_model(S60, FF.DRIVE_SETS["D0"], layout=None, bms=False)["total_kg"]
    k = FF.torque_factor(m, S)
    q = m / m_ref
    if kp_rule == "similar":
        k_kp, k_kv = q * s, q * s * math.sqrt(s)
    elif kp_rule == "literal":
        k_kp = k_kv = q * s * s
    elif kp_rule == "adopted":
        k_kp = k_kv = 1.0
    else:
        raise ValueError(f"kp_rule must be similar, literal or adopted, got {kp_rule!r}")
    return {"S": S, "s": s, "drives": dkey, "kp_rule": kp_rule, "mass_model_kg": m, "mass_model_60_kg": m_ref,
            "k": k, "k_kp": k_kp, "k_kv": k_kv, "k_ankle_stiffness": q * s, "k_ankle_damping": q * s * math.sqrt(s)}


# --------------------------------------------------------------------------------------------------------- build
def _scale_geometry(root: ET.Element, s: float) -> None:
    wb = root.find("worldbody")
    for el in wb.iter():
        if el.tag in ("body", "geom", "site", "camera", "light") and el.get("pos") is not None:
            el.set("pos", _scale_text(el.get("pos"), s))
        if el.tag in ("geom", "site"):
            if el.get("fromto") is not None:
                el.set("fromto", _scale_text(el.get("fromto"), s))
            if el.get("size") is not None and el.get("type") != "plane":
                el.set("size", _scale_text(el.get("size"), s))
    for mesh in root.find("asset").iter("mesh"):
        mesh.set("scale", _scale_text(mesh.get("scale", "1 1 1"), s))
    st = root.find("statistic")
    if st is not None:
        for a in ("center", "extent", "meansize"):
            if st.get(a) is not None:
                st.set(a, _scale_text(st.get(a), s))


def _set_masses(root: ET.Element, parts: dict) -> dict:
    seen = {}
    for b in root.iter("body"):
        for g in b.findall("geom"):
            if g.get("mass") is None or float(g.get("mass")) <= 0:
                continue
            name = g.get("name")
            if name not in parts:
                raise ValueError(f"massed geom {name} has no row in {LEDGER.name}")
            row = parts[name]
            r = row["kg"] / row["kg60"]
            old = float(g.get("mass"))
            if r != 1.0:
                g.set("mass", _fmt(old * r))
            seen[name] = {"xml60_kg": old, "kg": old * r, "ratio": r}
    missing = set(parts) - set(seen)
    if missing:
        raise ValueError(f"ledger rows with no massed geom in the model: {sorted(missing)}")
    return seen


def _gravity_moment(xml_text: str, joint: str) -> float:
    """Gravity torque about ``joint`` at the neutral pose (qpos0, at rest), as build_model.set_neck_spring takes it."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml_text)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return float(d.qfrc_bias[m.jnt_dofadr[m.joint(joint).id]])


def layout_places(S: float, dkey: str, layout: str) -> dict:
    """The study's desk layout for (S, drive set, layout): per end, the roll drive's lateral place y_r, the pitch drive's
    centre e_p (from the roll yoke), the link plane y_link (from the yoke) and the knee drive's centre in the upper
    link's frame (x, y outboard, z), m, from ``form_factor_study.make_layout`` (L1 runs its knee-drive search)."""
    lay = FF.make_layout(S, FF.DRIVE_SETS[dkey], layout)
    info = lay["info"]
    dr = FF.drives_of(FF.DRIVE_SETS[dkey])
    s = S / S60
    out = {"y_roll_m": info["y_roll_m"], "ends": {}}
    for end in ("fore", "hind"):
        e = info["ends"][end]
        Lk = dr[f"{end}_knee"]["length"]
        P = e["axis_pt"]
        yc = (0.0112 * s + 0.002 + Lk / 2) if layout == "L2" else 0.0
        out["ends"][end] = {"e_p_m": e["e_p"], "y_link_m": e["y_link"], "knee_centre_m": (P[0], yc, P[1])}
    return out


def _vec(t):
    return np.array([float(v) for v in t.split()])


def _set_vec(el, attr, v):
    el.set(attr, " ".join(_fmt(x) for x in v))


def _apply_layout(root: ET.Element, places: dict) -> dict:
    """Move each leg's parts to the desk layout's places (``layout_places``); masses and shapes unchanged."""
    bodies = {b.get("name"): b for b in root.iter("body")}
    geoms = {g.get("name"): g for g in root.iter("geom")}
    rec = {}
    for side, sg in (("left", 1.0), ("right", -1.0)):
        for end in ("fore", "hind"):
            c = f"{side}_{end}"
            pe = places["ends"][end]
            yoke = bodies[f"{c}_roll_yoke"]
            p = _vec(yoke.get("pos"))
            dy_roll = sg * places["y_roll_m"] - p[1]
            p[1] = sg * places["y_roll_m"]
            _set_vec(yoke, "pos", p)
            for gn in (f"{c}_x4_roll_stator", f"{c}_x4_roll_stator_collision"):
                ft = _vec(geoms[gn].get("fromto"))
                ft[[1, 4]] += dy_roll
                _set_vec(geoms[gn], "fromto", ft)
            _set_vec(geoms[f"{c}_shoulder_yoke"], "fromto", [0, 0, 0, 0, sg * pe["e_p_m"], 0])
            for gn in (f"{c}_x4_pitch_stator", f"{c}_x4_pitch_stator_collision"):
                ft = _vec(geoms[gn].get("fromto"))
                ft[[1, 4]] += sg * pe["e_p_m"] - 0.5 * (ft[1] + ft[4])
                _set_vec(geoms[gn], "fromto", ft)
            up = bodies[f"{c}_upper_link"]
            pu = _vec(up.get("pos"))
            dy_link = sg * pe["y_link_m"] - pu[1]
            pu[1] = sg * pe["y_link_m"]
            _set_vec(up, "pos", pu)
            kn = [g for g in up.findall("geom") if "flexion_stator" in g.get("name")]
            main = [g for g in kn if not g.get("name").endswith("_collision")][0]
            ft = _vec(main.get("fromto"))
            cen = 0.5 * (ft[:3] + ft[3:])
            kc = pe["knee_centre_m"]
            delta = np.array([kc[0], sg * kc[1], kc[2]]) - cen
            for g in kn + [geoms[f"{c}_drive_crank"]]:
                f2 = _vec(g.get("fromto"))
                f2[:3] += delta
                f2[3:] += delta
                _set_vec(g, "fromto", f2)
            rec[c] = {"roll_drive_dy_m": float(dy_roll), "link_plane_dy_m": float(dy_link),
                      "knee_drive_moved_m": [float(x) for x in delta]}
    return rec


def build_xml(S: float, dkey: str = "D0", kp_rule: str = "similar", src: Path = SRC,
              layout: str | None = None) -> tuple[str, dict]:
    """The scaled model's XML text and its record (``layout`` None: the model as built, scaled)."""
    f = factors(S, dkey, kp_rule)
    s = f["s"]
    src_text = src.read_text()
    root = ET.fromstring(src_text)
    root.set("model", f"{root.get('model')}_scaled_H{S:.2f}_{dkey}_{kp_rule}" + (f"_{layout}" if layout else ""))
    if s != 1.0:
        _scale_geometry(root, s)
    parts = part_masses(S, dkey)
    mass_rows = _set_masses(root, parts)
    layout_rec = None
    if layout:
        layout_rec = {"layout": layout, "places": layout_places(S, dkey, layout)}
        layout_rec["moves"] = _apply_layout(root, layout_rec["places"])
    drives = leg_drive_table(dkey)
    joints = {j.get("name"): j for j in root.iter("joint")}
    acts = {a.get("name"): a for a in root.find("actuator")}
    for jn, dd in drives.items():
        a = acts[jn]
        if f["k_kp"] != 1.0:
            a.set("kp", _fmt(float(a.get("kp")) * f["k_kp"]))
        if f["k_kv"] != 1.0:
            a.set("kv", _fmt(float(a.get("kv")) * f["k_kv"]))
        lim = dd["limit_nm"]
        if abs(lim - float(a.get("forcerange").split()[1])) > 1e-9:
            a.set("forcerange", f"{_fmt(-lim)} {_fmt(lim)}")
        if abs(dd["armature_kgm2"] - float(joints[jn].get("armature"))) > 1e-12:
            joints[jn].set("armature", _fmt(dd["armature_kgm2"]))
    for c in CORNERS:
        j = joints[f"{c}_ankle"]
        if f["k_ankle_stiffness"] != 1.0:
            j.set("stiffness", _fmt(float(j.get("stiffness")) * f["k_ankle_stiffness"]))
            j.set("damping", _fmt(float(j.get("damping")) * f["k_ankle_damping"]))
    # head and tail: gravity moments at the neutral pose, before any neck or tail change (the spring is not in
    # qfrc_bias), against the source's
    text1 = ET.tostring(root, encoding="unicode")
    g_neck = _gravity_moment(text1, "neck_pitch") / _gravity_moment(src_text, "neck_pitch")
    g_tail = _gravity_moment(text1, "tail_pitch") / _gravity_moment(src_text, "tail_pitch")
    big = FF.servo_set(S) != {}
    servo_cat = drive_catalog()["servos"]
    servo_rec = {}
    for name, ratio in (("neck_pitch", g_neck), ("neck_yaw", g_neck), ("tail_yaw", g_tail), ("tail_pitch", g_tail)):
        a = acts[name]
        if abs(ratio - 1.0) > 1e-12:
            a.set("kp", _fmt(float(a.get("kp")) * ratio))
            a.set("kv", _fmt(float(a.get("kv")) * ratio * math.sqrt(s)))
        lim = SOURCE_SERVO_LIMITS[name]
        if big:
            lim = SERVO_LIMIT_SHARE * servo_cat[BIG_SERVO[name]]["peak_nm"]
            a.set("forcerange", f"{_fmt(-lim)} {_fmt(lim)}")
        servo_rec[name] = {"moment_ratio": ratio, "limit_nm": lim, "servo": BIG_SERVO[name] if big else "as source"}
    if abs(g_neck - 1.0) > 1e-12:
        joints["neck_pitch"].set("stiffness", _fmt(float(joints["neck_pitch"].get("stiffness")) * g_neck))
    text = ET.tostring(root, encoding="unicode") + "\n"
    import mujoco
    m = mujoco.MjModel.from_xml_string(text)
    info = {**f, "source": str(src.relative_to(BUILD)), "source_sha256": sha256_text(src_text),
            "xml_sha256": sha256_text(text), "compiled_mass_kg": float(m.body_mass.sum()),
            "ledger_sum_kg": sum(r["kg"] for r in parts.values()),
            "leg_drives": drives, "head_tail": servo_rec, "neck_spring_stiffness": float(
                joints["neck_pitch"].get("stiffness")), "big_servos": big,
            "mass_by_class_kg": _by_class(parts), "geoms_with_mass": len(mass_rows), "layout": layout_rec}
    return text, info


def _by_class(parts: dict) -> dict:
    out: dict = {}
    for r in parts.values():
        out[r["cls"]] = out.get(r["cls"], 0.0) + r["kg"]
    return out


def write_model(S: float, out_dir: Path, dkey: str = "D0", kp_rule: str = "similar",
                layout: str | None = None) -> tuple[Path, dict]:
    text, info = build_xml(S, dkey, kp_rule, layout=layout)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (f"trike_scaled_H{S:.2f}_{dkey}_{kp_rule}" + (f"_{layout}" if layout else "") + ".xml")
    path.write_text(text)
    info["path"] = str(path)
    return path, info


# --------------------------------------------------------------------------------------------------------- the trot
LENGTH_FIELDS = ("leg_drop_m", "lift_m", "press_m", "max_reach_m")
TIME_FIELDS = ("period_s", "velocity_tau_s", "stand_s", "ramp_s", "attitude_tau_s", "land_ramp_s")
SECONDS_GAINS = ("k_speed", "k_side")       # m of touchdown shift per m/s of speed: a time
RATE_GAINS = ("k_heading", "k_drift")       # 1/s
SHARE_FIELDS = ("duty", "k_roll", "k_pitch", "stance_load", "swing_land_frac", "early_load")  # dimensionless
FLAG_FIELDS = ("frame", "swing_ground_from_attitude", "early_touchdown")


def scaled_walk_params(s: float):
    """``walk60.WalkParams()`` Froude-scaled: lengths x s, times x sqrt(s), rates / sqrt(s), shares and flags kept."""
    import walk60
    P = walk60.WalkParams()
    ch = {}
    for k in LENGTH_FIELDS:
        ch[k] = getattr(P, k) * s
    for k in TIME_FIELDS + SECONDS_GAINS:
        ch[k] = getattr(P, k) * math.sqrt(s)
    for k in RATE_GAINS:
        ch[k] = getattr(P, k) / math.sqrt(s)
    return P.replace(**ch) if s != 1.0 else P


def plant(model_path: str, info: dict | None, **over):
    """``walk60.plant_config`` on a model, with S2's floor and the nominal physics factors (randomisation kept on so the
    env sets the floor and sole friction, with every other factor fixed at 1), the drive set's continuous ratings."""
    import walk60
    cfg = dict(model_path=str(model_path), friction_range=(FRICTION, FRICTION), mass_scale_range=(1.0, 1.0),
               kp_scale_range=(1.0, 1.0), force_limit_scale_range=(1.0, 1.0), observation_noise=True)
    if info is not None:
        d = info["leg_drives"]
        cfg.update(cont_torque_roll=d["left_fore_roll"]["continuous_nm"],
                   cont_torque_pitch=d["left_fore_pitch"]["continuous_nm"],
                   cont_torque_fore_flexion=d["left_fore_flexion"]["continuous_nm"],
                   cont_torque_hind_flexion=d["left_hind_flexion"]["continuous_nm"])
    cfg.update(over)
    return walk60.plant_config(**cfg)


def _r(x, n=5):
    return [round(float(v), n) for v in x]


def stand_episode(model_path: str, info: dict | None, seed: int, seconds: float = STAND_S,
                  delay: int = DELAY_MS) -> dict:
    """S1: zero action for ``seconds``; per physics step the leg torques and the foot loads (touch / weight)."""
    from trike_env import TrikeEnv
    env = TrikeEnv(plant(model_path, info, episode_seconds=max(20.0, seconds)))
    d = env.data
    idx = np.asarray(env._leg_act)
    taus, loads = [], []

    def hook(e):
        taus.append(d.actuator_force[idx].copy())
        loads.append(d.sensordata[e._touch_adr] / e.nominal_weight)

    env.reset(seed=seed, options={"command_speed": 0.0, "delay_steps": delay})
    env.substep_hook = hook
    fell, term_reason = False, None
    n = int(round(seconds / env.dt))
    for _ in range(n):
        _, _, term, trunc, inf = env.step(np.zeros(12))
        if term:
            fell, term_reason = True, inf.get("termination")
            break
    env.substep_hook = None
    tau, ld = np.asarray(taus), np.asarray(loads)
    t = (np.arange(len(tau)) + 1) * env.physics_dt
    w = t > seconds / 2.0  # 5-10 s in S1 (shorter calls, e.g. the tests: the second half)
    settled = t > min(1.0, seconds / 2.0)
    ld_s = ld[settled]
    nwin = len(ld_s) // 100
    win = ld_s[:nwin * 100].reshape(nwin, 100, 4).mean(1) if nwin else ld_s
    return {"seed": seed, "fell": fell, "termination": term_reason, "length_s": round(float(t[-1]), 3),
            "torque_window_s": [seconds / 2.0, seconds], "mass_kg": float(env.nominal_mass), "standing_height_m": float(env.nominal_height),
            "joints": [env.model.actuator(int(i)).name for i in idx],
            "tau_mean_5_10s_nm": _r(tau[w].mean(0)), "tau_abs_max_5_10s_nm": _r(np.abs(tau[w]).max(0)),
            "load_share_mean_5_10s": _r(ld[w].mean(0)), "load_share_min_100ms_1_10s": _r(win.min(0)),
            "load_share_sum_mean_5_10s": round(float(ld[w].sum(1).mean()), 5)}


def trot_episode(model_path: str, info: dict | None, seed: int, s: float, delay: int = DELAY_MS) -> dict:
    """S2: the committed trot, Froude-scaled, one episode; gait_metrics' measures plus peaks and the late speed."""
    import gait_metrics
    import walk60
    from trike_env import TrikeEnv
    env = TrikeEnv(plant(model_path, info))
    P = scaled_walk_params(s)
    ctrl = walk60.TrotController(env, P)
    command = COMMAND60 * math.sqrt(s)
    log = gait_metrics.record_episode(env, ctrl.act, seed=seed, command=command, delay_steps=delay)
    ms = gait_metrics.measures(log, command)
    t = log["t"]
    w = t >= gait_metrics.WINDOW_S
    tau, om = log["tau"][w], log["omega"][w]
    lim = log["limit_nominal"]
    late = t >= 10.0
    late_speed = None
    if late.sum() > 1000:
        h = np.array([math.cos(log["heading0"]), math.sin(log["heading0"])])
        p = log["pos"][late]
        late_speed = float((p[-1, :2] - p[0, :2]) @ h / (t[late][-1] - t[late][0]))
    leg_len = 0.284 * s  # ankle pin below the chassis origin at the neutral pose (source: 0.284 m, fore and hind)
    return {"seed": seed, "command_mps": command, "delay_steps": delay, "fell": bool(log["fell"]),
            "termination": log["termination"], "length_s": float(log["time_s"]), "mass_kg": float(log["mass"]),
            "joints": list(log["names"]), "speed_mps_2_20s": ms["forward_speed_mps"], "speed_mps_10_20s": late_speed,
            "froude_speed_2_20s": ms["forward_speed_mps"] / math.sqrt(9.81 * leg_len),
            "rms_nm": _r([ms["drives"][j]["rms_nm"] for j in log["names"]]),
            "rms_over_continuous": _r([ms["drives"][j]["rms_over_continuous"] for j in log["names"]]),
            "mean_nm": _r(tau.mean(0)), "peak_nm": _r(np.abs(tau).max(0)),
            "p999_nm": _r(np.quantile(np.abs(tau), 0.999, axis=0)), "limit_nm": _r(lim),
            "share_at_limit": _r((np.abs(tau) >= 0.99 * lim).mean(0)),
            "peak_rpm": _r(np.abs(om).max(0) * 60 / (2 * math.pi), 3),
            "rms_rpm": _r(np.sqrt((om ** 2).mean(0)) * 60 / (2 * math.pi), 3),
            "power_w": ms["power_w"], "slip_ratio": ms["slip_ratio"], "touchdowns_per_s": ms["touchdowns_per_s"],
            "heading_change_deg": ms["heading_change_deg"], "sideways_drift_m": ms["sideways_drift_m"],
            "ik_err_max_m": float(ctrl.ik_err_max), "walk_params": dataclasses.asdict(P)}


# --------------------------------------------------------------------------------------------------------- cases
# name: (crown height m, drive set, kp rule, delay rule, role)
CASES = {
    "H0.60_source": (0.60, None, None, "fixed", "control: the source XML itself through the same harness"),
    "H0.50": (0.50, "D0", "similar", "fixed", "pre-registered"),
    "H0.60": (0.60, "D0", "similar", "fixed", "pre-registered (control: must reproduce the source)"),
    "H0.70": (0.70, "D0", "similar", "fixed", "pre-registered"),
    "H0.80": (0.80, "D0", "similar", "fixed", "pre-registered"),
    "H0.55": (0.55, "D0", "similar", "fixed", "supporting (planned before the runs)"),
    "H0.65": (0.65, "D0", "similar", "fixed", "supporting (planned before the runs)"),
    "H0.50_kp_literal": (0.50, "D0", "literal", "fixed", "supporting: the literal kp reading"),
    "H0.80_kp_literal": (0.80, "D0", "literal", "fixed", "supporting: the literal kp reading"),
    "H0.50_delay_scaled": (0.50, "D0", "similar", "scaled", "supporting: delay x sqrt(s)"),
    "H0.80_delay_scaled": (0.80, "D0", "similar", "scaled", "supporting: delay x sqrt(s)"),
    "H0.80_D8": (0.80, "D8", "similar", "fixed",
                 "supporting: X8 everywhere at 80 cm, as-built layout; NOT a passing configuration (none passes C1-C5 "
                 "at 0.70 or 0.80 m in design/form_factor_out.json)"),
}
# Added 28 Nov 2026 AFTER the results above and after review (findings PH-1, CTRL-7, F3, MECH-2), written into
# results/form_factor_2026-11-27/scaled_trike_plan_addendum.json before any of them ran. Supporting, post hoc: the
# same harness and the same S1 / S2 rules; none is a pre-registered size.
POST_HOC = "post hoc (after review), supporting"
_DRIVE_CASES = {0.50: ("D1", "D2", "D4", "D5", "D6", "D8"), 0.55: ("D2", "D6", "D8"), 0.60: ("D2", "D6", "D8"),
                0.65: ("D6", "D8")}
for _S, _ds in _DRIVE_CASES.items():
    for _dk in _ds:
        CASES[f"H{_S:.2f}_{_dk}"] = (_S, _dk, "similar", "fixed",
                                     f"{POST_HOC}: drive set {_dk}, as-built layout (the scaling rule on a drive set "
                                     "the desk study ranks as passing)")
for _S in (0.50, 0.55):
    CASES[f"H{_S:.2f}_kp_adopted"] = (_S, "D0", "adopted", "fixed",
                                      f"{POST_HOC}: the adopted A2 leg gains unchanged (no gain rescaling)")
for _S, _dk, _lo in ((0.60, "D0", "L1"), (0.60, "D0", "L2"), (0.50, "D0", "L1"), (0.50, "D0", "L2"),
                     (0.50, "D2", "L2"), (0.50, "D6", "L2"), (0.55, "D6", "L2"), (0.60, "D6", "L1")):
    CASES[f"H{_S:.2f}_{_dk}_{_lo}"] = (_S, _dk, "similar", "fixed",
                                       f"{POST_HOC}: drive set {_dk} in the study's desk layout {_lo} (lateral offsets "
                                       "and knee-drive place; masses as built)", _lo)


def case_layout(name: str) -> str | None:
    c = CASES[name]
    return c[5] if len(c) > 5 else None


def case_model(name: str, models_dir: Path) -> tuple[str, dict | None, float, int]:
    S, dkey, rule, delay_rule = CASES[name][:4]
    s = S / S60
    if dkey is None:
        return str(SRC), None, s, DELAY_MS
    path, info = write_model(S, models_dir, dkey, rule, layout=case_layout(name))
    delay = DELAY_MS if delay_rule == "fixed" else int(round(DELAY_MS * math.sqrt(s)))
    return str(path), info, s, delay


def _job(job):
    kind, name, path, info, s, delay, seed = job
    t0 = time.time()
    if kind == "stand":
        r = stand_episode(path, info, seed, delay=delay)
    else:
        r = trot_episode(path, info, seed, s, delay=delay)
    r["wall_s"] = round(time.time() - t0, 1)
    return kind, name, r


def run(cases: list[str], models_dir: Path, out: Path, workers: int, seeds=SEEDS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    jobs = []
    for n in cases:
        path, info, s, delay = case_model(n, models_dir)
        models[n] = {"path": path, "info": info, "s": s, "delay_steps": delay, "role": CASES[n][4],
                     "xml_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "layout": case_layout(n)}
        for sd in seeds:
            jobs.append(("stand", n, path, info, s, delay, sd))
            jobs.append(("trot", n, path, info, s, delay, sd))
    t0 = time.time()
    res = {n: {"model": models[n], "stand": [], "trot": []} for n in cases}
    with mp.get_context("spawn").Pool(workers) as pool:
        for kind, n, r in pool.imap_unordered(_job, jobs, chunksize=1):
            res[n][kind].append(r)
            print(f"{n:20s} {kind:5s} seed {r['seed']} fell {r['fell']} "
                  + (f"speed {r['speed_mps_2_20s']:.3f} m/s (10-20 s {r['speed_mps_10_20s'] or float('nan'):.3f})"
                     if kind == "trot" else f"min foot {min(r['load_share_min_100ms_1_10s']):.3f}")
                  + f"  [{r['wall_s']} s, {time.time() - t0:.0f} s total]", flush=True)
    for n in cases:
        for kind in ("stand", "trot"):
            res[n][kind].sort(key=lambda r: r["seed"])
        path = out / f"scaled_raw_{n}.json"
        rec = {"simulation_only": True, "tool": "sim/build_scaled_trike.py run", "case": n, **res[n],
               "seeds": list(seeds), "date": "2026-11-28"}
        rec["model"]["path"] = Path(rec["model"]["path"]).name
        if rec["model"]["info"]:
            rec["model"]["info"]["path"] = Path(rec["model"]["info"]["path"]).name
        path.write_text(json.dumps(rec, indent=0, default=float) + "\n")
    return res


# --------------------------------------------------------------------------------------------------------- compare
def group_of(j: str) -> str:
    end = "fore" if "_fore_" in j else "hind"
    if j.endswith("_roll"):
        return "roll"
    if j.endswith("_pitch"):
        return f"{end} pitch"
    return "front elbow" if end == "fore" else "hind knee"


def load_raw(out: Path) -> dict:
    return {p.stem.replace("scaled_raw_", ""): json.loads(p.read_text()) for p in sorted(out.glob("scaled_raw_*.json"))}


def _err(sim, pred):
    return (np.asarray(sim, dtype=float) / np.asarray(pred, dtype=float) - 1.0).tolist()


def _slow(r: dict) -> bool:
    """The plan's slow-mode flag: the 10-20 s forward speed under 0.85 x the command."""
    return r["speed_mps_10_20s"] is not None and r["speed_mps_10_20s"] < 0.85 * r["command_mps"]


def _post_hoc(tr: dict, ref_trot: dict, k: float, joints: list) -> dict:
    """ADDED AFTER SEEING THE RESULTS (descriptive, not a rule): the rule's error on the seeds where neither this case
    nor H0.60 is in the slow mode, and what the slow mode itself adds at this size."""
    sds = sorted(sd for sd in tr if sd in ref_trot and not _slow(tr[sd]) and not _slow(ref_trot[sd]))
    slow_sd = sorted(sd for sd in tr if _slow(tr[sd]))
    norm_sd = sorted(sd for sd in tr if not _slow(tr[sd]))
    out = {"label": "added after seeing the results; descriptive, not a rule", "slow_seeds": slow_sd,
           "hind_touchdowns_per_s_by_seed": {sd: [tr[sd]["touchdowns_per_s"][c] for c in ("left_hind", "right_hind")]
                                             for sd in sorted(tr)},
           "normal_pair_seeds": sds}
    if sds:
        e = np.asarray([_err(tr[sd]["rms_nm"], k * np.asarray(ref_trot[sd]["rms_nm"])) for sd in sds])
        out.update({"normal_pair_rms_error_median": np.median(e, 0).tolist(), "normal_pair_rms_error_min": e.min(0).tolist(),
                    "normal_pair_rms_error_max": e.max(0).tolist(),
                    "normal_pair_share_within_25pct": float((np.abs(e) <= S2_BAND).mean())})
        for key in ("peak_nm", "p999_nm"):
            sim = np.mean([tr[sd][key] for sd in sds], 0)
            pred = k * np.mean([ref_trot[sd][key] for sd in sds], 0)
            out[f"normal_pair_{key}_sim"] = sim.tolist()
            out[f"normal_pair_{key}_error"] = _err(sim, pred)
        out["normal_pair_share_at_limit_max"] = np.max([tr[sd]["share_at_limit"] for sd in sds], 0).tolist()
    if slow_sd and norm_sd:
        out["slow_over_normal_rms_minus_1"] = _err(np.mean([tr[sd]["rms_nm"] for sd in slow_sd], 0),
                                                   np.mean([tr[sd]["rms_nm"] for sd in norm_sd], 0))
    for name, group in (("normal", norm_sd), ("slow", slow_sd)):
        if group:
            out[f"worst_rms_over_continuous_{name}"] = float(max(max(tr[sd]["rms_over_continuous"]) for sd in group))
    return out


def _study_check(S: float, tr: dict, seeds) -> dict | None:
    """The study's stored trot predictions (design/form_factor_out.json simulate_predictions: k x its own 60 cm trot,
    seeds 810001-2, the model's sole friction 0.85) against this run's group figures (the group's worst joint, mean
    of the verdict seeds)."""
    p = BUILD / "design" / "form_factor_out.json"
    if not p.exists():
        return None
    pred = json.loads(p.read_text())["simulate_predictions"].get(f"{S:.2f}")
    if not pred or "scripted_trot_walk_rms_joint_nm_at_froude_uncapped" not in pred:
        return None
    sim: dict = {}
    for sd in seeds:
        for j, v in zip(tr[sd]["joints"], tr[sd]["rms_nm"]):
            g = group_of(j)
            sim.setdefault(sd, {})[g] = max(sim.get(sd, {}).get(g, 0.0), v)
    g_sim = {g: float(np.mean([sim[sd][g] for sd in seeds])) for g in sim[seeds[0]]}
    g_pred = pred["scripted_trot_walk_rms_joint_nm_at_froude_uncapped"]
    return {"predicted_nm": g_pred, "simulated_nm": g_sim, "error": {g: g_sim[g] / g_pred[g] - 1 for g in g_pred},
            "stand_predicted_nm": pred.get("stand_by_group_nm")}


def compare(out: Path) -> dict:
    """Predicted (k x the H0.60 run, same seed) against simulated, per leg joint, per case; S1 and S2 judged on
    ``VERDICT_SEEDS``; the other seeds and the comparison against the H0.60 seed mean are supporting."""
    raw = load_raw(out)
    ref = raw["H0.60"]
    joints = ref["trot"][0]["joints"]
    ref_trot = {r["seed"]: r for r in ref["trot"]}
    ref_stand = {r["seed"]: r for r in ref["stand"]}
    res = {"simulation_only": True, "tool": "sim/build_scaled_trike.py compare", "date": "2026-11-28",
           "rule": "k = (m / 12.7506 kg) x (H / 0.60 m) (design/form_factor_study.py torque_factor, m from its "
                   "mass_model with no layout additions or cut-off); prediction = k x the H0.60 run's figure on the "
                   "same seed; S2 holds at a size when every leg joint's RMS is within +-25 % of it on at least 2 of "
                   "seeds 101-103; S1 holds when no seed 101-103 falls in 10 s at zero action and every foot's 100 ms "
                   "load share stays above 0.15 from 1 s to 10 s",
           "verdict_seeds": list(VERDICT_SEEDS), "joints": joints, "cases": {}}
    rows = []
    for n, rec in raw.items():
        info = rec["model"]["info"]
        S = CASES[n][0]
        m = info["mass_model_kg"] if info else M60
        k = FF.torque_factor(m, S)
        s = S / S60
        c = {"role": rec["model"]["role"], "S": S, "s": s, "mass_kg": m,
             "compiled_mass_kg": info["compiled_mass_kg"] if info else None, "k": k,
             "delay_steps": rec["model"]["delay_steps"], "kp_rule": info["kp_rule"] if info else "source",
             "drives": info["drives"] if info else "D0", "layout": rec["model"].get("layout"), "stand": {}, "trot": {}}
        # ---- S1
        st = {r["seed"]: r for r in rec["stand"]}
        vs = [st[sd] for sd in VERDICT_SEEDS]
        s1 = [(not r["fell"]) and min(r["load_share_min_100ms_1_10s"]) > S1_LOAD for r in vs]
        c["stand"] = {"S1_pass_by_seed": s1, "S1_pass": bool(all(s1)),
                      "falls_all_seeds": sum(r["fell"] for r in st.values()),
                      "min_foot_share_100ms_1_10s_verdict_seeds": min(min(r["load_share_min_100ms_1_10s"])
                                                                      for r in vs),
                      "min_foot_share_100ms_1_10s_all_seeds": min(min(r["load_share_min_100ms_1_10s"])
                                                                  for r in st.values()),
                      "foot_share_mean_5_10s": np.mean([r["load_share_mean_5_10s"] for r in vs], axis=0).tolist(),
                      "standing_height_m": vs[0]["standing_height_m"]}
        sim_st = np.mean([np.abs(st[sd]["tau_mean_5_10s_nm"]) for sd in VERDICT_SEEDS], axis=0)
        ref_st = np.mean([np.abs(ref_stand[sd]["tau_mean_5_10s_nm"]) for sd in VERDICT_SEEDS], axis=0)
        c["stand"]["tau_abs_mean_nm"] = sim_st.tolist()
        c["stand"]["pred_nm"] = (k * ref_st).tolist()
        c["stand"]["error"] = _err(sim_st, k * ref_st)
        c["stand"]["error_by_seed"] = {sd: _err(np.abs(st[sd]["tau_mean_5_10s_nm"]),
                                                k * np.abs(ref_stand[sd]["tau_mean_5_10s_nm"])) for sd in VERDICT_SEEDS}
        # ---- S2
        tr = {r["seed"]: r for r in rec["trot"]}
        within, e_rms = {}, {}
        for sd in VERDICT_SEEDS:
            e_rms[sd] = _err(tr[sd]["rms_nm"], k * np.asarray(ref_trot[sd]["rms_nm"]))
            within[sd] = (np.abs(np.asarray(e_rms[sd])) <= S2_BAND).tolist()
        wa = np.asarray([within[sd] for sd in VERDICT_SEEDS])
        per_joint_ok = (wa.sum(0) >= 2).tolist()
        all_sd = sorted(set(tr) & set(ref_trot))
        e_all = np.asarray([_err(tr[sd]["rms_nm"], k * np.asarray(ref_trot[sd]["rms_nm"])) for sd in all_sd])
        mean_sim = np.mean([tr[sd]["rms_nm"] for sd in all_sd], axis=0)
        mean_ref = np.mean([ref_trot[sd]["rms_nm"] for sd in all_sd], axis=0)
        sim3 = np.mean([tr[sd]["rms_nm"] for sd in VERDICT_SEEDS], axis=0)
        ref3 = np.mean([ref_trot[sd]["rms_nm"] for sd in VERDICT_SEEDS], axis=0)
        pk = lambda key, sds: np.mean([tr[sd][key] for sd in sds], axis=0)  # noqa: E731
        pk60 = lambda key, sds: np.mean([ref_trot[sd][key] for sd in sds], axis=0)  # noqa: E731
        cmd = tr[VERDICT_SEEDS[0]]["command_mps"]
        walked = {sd: (not tr[sd]["fell"]) and tr[sd]["speed_mps_2_20s"] >= 0.5 * cmd for sd in tr}
        slow = {sd: _slow(tr[sd]) for sd in tr}
        c["trot"] = {
            "command_mps": cmd,
            "falls_verdict_seeds": sum(tr[sd]["fell"] for sd in VERDICT_SEEDS),
            "falls_all_seeds": sum(r["fell"] for r in tr.values()),
            "walked_verdict_seeds": [walked[sd] for sd in VERDICT_SEEDS],
            "slow_mode_10_20s_verdict_seeds": [slow[sd] for sd in VERDICT_SEEDS],
            "slow_mode_10_20s_all_seeds": sum(slow.values()),
            "speed_mps_verdict_seeds": [tr[sd]["speed_mps_2_20s"] for sd in VERDICT_SEEDS],
            "speed_mps_10_20s_verdict_seeds": [tr[sd]["speed_mps_10_20s"] for sd in VERDICT_SEEDS],
            "speed_over_command_mean_all_seeds": float(np.mean([tr[sd]["speed_mps_2_20s"] for sd in all_sd]) / cmd),
            "froude_speed_mean_all_seeds": float(np.mean([tr[sd]["froude_speed_2_20s"] for sd in all_sd])),
            "rms_nm_verdict_seeds": {sd: tr[sd]["rms_nm"] for sd in VERDICT_SEEDS},
            "rms_pred_nm_verdict_seeds": {sd: (k * np.asarray(ref_trot[sd]["rms_nm"])).tolist() for sd in VERDICT_SEEDS},
            "rms_error_verdict_seeds": e_rms, "within_25pct_verdict_seeds": within,
            "joint_ok_2_of_3": per_joint_ok, "S2_holds": bool(all(per_joint_ok)),
            "joints_failing": [joints[i] for i, ok in enumerate(per_joint_ok) if not ok],
            "rms_error_3seed_mean": _err(sim3, k * ref3),
            "rms_error_all_seeds_median": np.median(e_all, axis=0).tolist(),
            "rms_error_all_seeds_min": e_all.min(0).tolist(), "rms_error_all_seeds_max": e_all.max(0).tolist(),
            "rms_error_vs_60cm_mean_all_seeds": _err(mean_sim, k * mean_ref),
            "share_within_25pct_all_seeds": (np.abs(e_all) <= S2_BAND).mean(0).tolist(),
            "peak_nm_3seed_mean": pk("peak_nm", VERDICT_SEEDS).tolist(),
            "peak_pred_nm_3seed_mean": (k * pk60("peak_nm", VERDICT_SEEDS)).tolist(),
            "peak_error_3seed_mean": _err(pk("peak_nm", VERDICT_SEEDS), k * pk60("peak_nm", VERDICT_SEEDS)),
            "p999_error_3seed_mean": _err(pk("p999_nm", VERDICT_SEEDS), k * pk60("p999_nm", VERDICT_SEEDS)),
            "limit_nm": tr[VERDICT_SEEDS[0]]["limit_nm"],
            "share_at_limit_worst_all_seeds": np.max([tr[sd]["share_at_limit"] for sd in all_sd], axis=0).tolist(),
            "peak_rpm_3seed_mean": pk("peak_rpm", VERDICT_SEEDS).tolist(),
            "peak_rpm_pred_3seed_mean": (pk60("peak_rpm", VERDICT_SEEDS) / math.sqrt(s)).tolist(),
            "rms_rpm_error_3seed_mean": _err(pk("rms_rpm", VERDICT_SEEDS), pk60("rms_rpm", VERDICT_SEEDS) / math.sqrt(s)),
            "rms_over_continuous_worst_verdict_seeds": np.max([tr[sd]["rms_over_continuous"] for sd in VERDICT_SEEDS],
                                                              axis=0).tolist(),
            "power_w_mean_all_seeds": float(np.mean([tr[sd]["power_w"] for sd in all_sd])),
            "power_pred_w": float(np.mean([ref_trot[sd]["power_w"] for sd in all_sd])) * (m / M60) * math.sqrt(s),
        }
        c["post_hoc"] = _post_hoc(tr, ref_trot, k, joints)
        c["study_prediction_check"] = _study_check(S, tr, VERDICT_SEEDS) if info and info["drives"] == "D0" else None
        res["cases"][n] = c
        for i, j in enumerate(joints):
            row = {"case": n, "S_m": S, "drives": c["drives"], "layout": c["layout"] or "as built",
                   "kp_rule": c["kp_rule"], "delay_ms": c["delay_steps"],
                   "k": round(k, 5), "joint": j, "group": group_of(j),
                   "stand_pred_nm": round(c["stand"]["pred_nm"][i], 4),
                   "stand_sim_nm": round(c["stand"]["tau_abs_mean_nm"][i], 4),
                   "stand_err": round(c["stand"]["error"][i], 4)}
            for sd in VERDICT_SEEDS:
                row[f"rms_pred_nm_s{sd}"] = round(c["trot"]["rms_pred_nm_verdict_seeds"][sd][i], 4)
                row[f"rms_sim_nm_s{sd}"] = round(tr[sd]["rms_nm"][i], 4)
                row[f"rms_err_s{sd}"] = round(e_rms[sd][i], 4)
            row.update({"joint_ok_2_of_3": per_joint_ok[i],
                        "rms_err_median_10seeds": round(c["trot"]["rms_error_all_seeds_median"][i], 4),
                        "peak_pred_nm": round(c["trot"]["peak_pred_nm_3seed_mean"][i], 3),
                        "peak_sim_nm": round(c["trot"]["peak_nm_3seed_mean"][i], 3),
                        "limit_nm": tr[VERDICT_SEEDS[0]]["limit_nm"][i],
                        "share_at_limit_worst": round(c["trot"]["share_at_limit_worst_all_seeds"][i], 4)})
            rows.append(row)
    (out / "scaled_trike_compare.json").write_text(json.dumps(res, indent=0, default=float) + "\n")
    with open(out / "scaled_trike_compare.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return res


# --------------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--size", type=float, required=True)
    b.add_argument("--drives", default="D0")
    b.add_argument("--kp-rule", default="similar", choices=("similar", "literal", "adopted"))
    b.add_argument("--layout", default=None, choices=("L1", "L2"))
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", required=True, help="folder for the built XML files (not the build tree)")
    r.add_argument("--out", default=str(RESULTS))
    r.add_argument("--cases", nargs="*", default=list(CASES))
    r.add_argument("--workers", type=int, default=2)
    c = sub.add_parser("compare")
    c.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args(argv)
    if a.cmd == "build":
        path, info = write_model(a.size, Path(a.out), a.drives, a.kp_rule, layout=a.layout)
        print(json.dumps({k: v for k, v in info.items() if k not in ("leg_drives", "mass_by_class_kg")}, indent=1))
        print(path)
    elif a.cmd == "run":
        run(a.cases, Path(a.models), Path(a.out), a.workers)
    else:
        res = compare(Path(a.out))
        for n, cc in res["cases"].items():
            tt = cc["trot"]
            print(f"{n:20s} k {cc['k']:.4f} S1 {cc['stand']['S1_pass']} S2 {tt['S2_holds']} "
                  f"(failing: {', '.join(tt['joints_failing']) or 'none'}) falls {tt['falls_verdict_seeds']}/3 "
                  f"({tt['falls_all_seeds']}/10) speed {', '.join(f'{v:.3f}' for v in tt['speed_mps_verdict_seeds'])}"
                  f" of {tt['command_mps']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
