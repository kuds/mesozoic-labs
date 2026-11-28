"""Packaging check of the adopted 60 cm model: all four legs, the chassis bay, the head and the tail. A DESK geometry
check on the model's parts (signed distances between its shapes), not a simulation run and not a judged verdict.

Model: ``generated/trike60_C_closed_capsule_fore128.xml`` (D25, D41; 12.751 kg) unless ``--model`` says otherwise.
Drive housings are the cylinders the model carries, checked here against ``data/drives.json`` (X4 55 x 61 mm, X6
80 x 67.5 mm): the bodies only, without output flanges, connectors or cable exits, which no record sizes.

Distances are ``mj_geomDistance`` (mujoco 3.10.0; negative = depth of overlap, the shortest translation that
separates the pair), with build_foreleg60's retry of an exact 0 between parts on different bodies (``_dist``). Two
axis-aligned boxes on one body are measured by their exact gap in the body frame (``check_layout60.box_gap``).
Distances are capped at 50 mm (shown as ">50").

1. **Legs.** For each leg (roll yoke, upper link, lower link, foot): every pair of its own parts, and every part
   against every part of the chassis, head and tail, over a grid of the leg's roll x pitch x flexion ranges (``--steps``
   values each, 9 by default; ankle at 0), and the foot against the rest of its leg over flexion x ankle. Pairs nearer
   than 10 mm anywhere are kept, with the pose of their least distance and their distance at neutral (all joints 0).
2. **The moving pushrod.** The model's rod is a fixed capsule on the upper link; here a probe copy of it is placed
   where the 1:1 parallelogram puts it (both cranks turning with the lower link) at every grid pose, against every part
   of its leg (its own cranks and the fixed rod excepted) and of the chassis, head and tail. Also its pin-to-pin length
   and its transmission angle (between crank and rod) over the flexion range.
3. **The linkage's axial gaps** (same body, constant): elbow axle to output crank, drive crank to its drive housing.
4. **The chassis bay** at neutral: every pair of chassis parts (shell ribs, seams and cable looms aside), and the head
   and the tail over their joint ranges against the chassis. Also the room around the bay's tightest pairs: how far
   each part can move before it meets another (``bay_room``). And the hind linkage against the rails over inward roll.
5. **Trial fixes**: candidate crossmember, rail and tail-servo positions, each moved part swept against the legs (as
   in 1) and the head and the tail over their ranges, and at neutral against the chassis parts left in place and
   every other moved part (as in 4), to show what a fix would clear and what it would meet next; then three combined
   layouts, A, B and C (shortened, inset rails with the drives on brackets and the crossmembers re-placed; in C also
   the two bay blocks in the crossmembers' way and the tray stays), since the single trials interact; and, for each
   moved frame member, whether it still meets what it must carry (``MOUNTS``). A trial lists the moved parts' pairs
   only: the parts it leaves in place keep the gaps of 1 and 4. Proposals for CAD.
6. **The legs against each other** (``legs_to_each_other``): every pair of legs over all eight of their joints (roll,
   pitch, flexion and ankle of each), ``--pair-steps`` values per joint (9 by default), every pose of one leg against
   every pose of the other, then a bounded local search between grid points from the deepest grid local minima of
   every pair of parts that comes within 50 mm (the moving rod in place of the fixed one); the share of the grid's poses
   where solid parts overlap and the joints' grid values that take part (each joint's bounds on the grid, not a region
   where the legs always meet: also the share of the poses inside those bounds that overlap, and the largest
   difference between the two legs' roll values among the overlapping poses); at the deepest pose of two soles, the overlap
   of their lower halves (the real half-capsule rocker, F17; ``rocker_overlap``); and candidate joint limits that
   would keep the legs apart (``leg_limits``).

Each pair is classed (``classify``): ``joint`` (the two parts meet at a joint or a mount by design; reason given;
not printed), ``touch`` (within 0.1 mm: resting on a tray or face to face), ``frame-drive`` (a frame member runs into a
drive housing: the frame needs a bracket or another path), ``part`` (two solid parts: a clash if negative),
``cover`` (a limb cover against its own leg: the schematic cover is a capsule along the strut; a real cover is shaped
around the drives; counted, not printed), ``shell`` (a shell rib, seam, limb cover or head or tail shell against
another body's part), ``harness`` (a cable or loom), ``foot shape`` (the simulated capsule sole is a full shape through
the lower strut, F17; the real part is a half-capsule rocker).

Usage: python check_packaging60.py [--model ...] [--steps 9] [--pair-steps 9] [--no-trials] [--json out.json]
Prints a report; writes only the --json file. Needs mujoco 3.10.0, numpy and scipy (``scipy.optimize.minimize``, the
local search of section 6): ``python -m pip install mujoco==3.10.0 numpy scipy``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_foreleg60 import ZERO_RETRIES, _dist  # noqa: E402
from check_layout60 import body_aabb, box_gap  # noqa: E402

MODEL = HERE / "generated" / "trike60_C_closed_capsule_fore128.xml"
DRIVES = HERE / "data" / "drives.json"
CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
LEG_BODIES = ("roll_yoke", "upper_link", "lower_link", "foot")
REPORT_M = 0.010      # pairs nearer than this somewhere are kept
CAP_M = 0.05          # mj_geomDistance limit
TOUCH_M = 1e-4        # |distance| within this: touching
REMEASURED = []       # (pair, pose, the two readings in mm) where an exact 0 was re-measured at +-0.06 deg
SHELL = ("PA11_rib", "shell_seam", "limb_cover", "frill_pa11_panel", "hollow_skull", "brow_horn", "nasal_horn",
         "tail_shell", "_eye")
HARNESS = ("loom", "lock_cable")
FRAME = ("6061_", "crossmember", "service_tray", "tray_stay", "neck_cradle", "drop_bracket", "tof_mount")
DRIVE = ("_stator",)
# Parts that meet at a joint or a mount by design (corner and side prefixes removed; same corner or side).
JOINED = {
    frozenset(("shoulder_yoke", "x4_pitch_stator")): "the yoke carries the pitch drive (coaxial mount)",
    frozenset(("shoulder_yoke", "x4_roll_stator")): "the yoke is bolted to the roll drive's output",
    frozenset(("upper_strut", "x4_pitch_stator")): "the upper link is bolted to the pitch drive's output",
    frozenset(("upper_strut", "shoulder_yoke")): "they meet at the pitch joint",
    frozenset(("upper_strut", "lower_strut")): "they meet at the elbow / knee joint",
    frozenset(("elbow_axle", "upper_strut")): "the axle passes through the link end",
    frozenset(("elbow_axle", "lower_strut")): "the axle passes through the link end",
    frozenset(("ankle_hub", "lower_strut")): "they meet at the ankle joint",
    frozenset(("ankle_hub", "sole")): "the hub carries the sole",
    frozenset(("drive_crank", "pushrod")): "pin joint of the linkage",
    frozenset(("output_crank", "pushrod")): "pin joint of the linkage",
    frozenset(("offset_motor_mount", "x6_flexion_stator")): "PR #697's motor plate carries the X6",
    frozenset(("mount_standoff", "offset_motor_mount")): "PR #697's standoff carries the motor plate",
    frozenset(("mount_standoff", "upper_strut")): "PR #697's standoff is fixed to the strut",
    frozenset(("drop_bracket", "x4_roll_stator")): "PR #697's drop bracket holds the roll drive",
    frozenset(("tof_mount_arm", "vl53l8cx")): "PR #697's arm carries the ToF sensor",
    frozenset(("neck_pitch_case", "neck_cradle")): "the cradle carries the neck pitch servo at the head pivot",
    frozenset(("neck_hub_and_skull_carrier", "neck_cradle")): "the cradle carries the neck hub at the head pivot",
}


def corner(name: str):
    return next((c for c in CORNERS if name.startswith(c + "_")), None)


def side(name: str):
    return next((s for s in ("left", "right") if name.startswith(s + "_")), None)


def base(name: str) -> str:
    c = corner(name)
    n = name[len(c) + 1:] if c else name
    s = side(n)
    n = n[len(s) + 1:] if s else n
    for prefix in ("xc430_w240_", "xl430_w250_"):
        n = n[len(prefix):] if n.startswith(prefix) else n
    return n


def has(name, keys) -> bool:
    return any(k in name for k in keys)


def classify(a: str, b: str, dist: float | None = None) -> tuple[str, str]:
    """(class, reason) of a pair of geom names; see the module docstring."""
    ca, cb = corner(a), corner(b)
    same = not (ca and cb and ca != cb) and not (side(a) and side(b) and side(a) != side(b))
    if same and frozenset((base(a), base(b))) in JOINED:
        return "joint", JOINED[frozenset((base(a), base(b)))]
    if has(a, FRAME) and has(b, FRAME):
        return "joint", "frame members joined to each other"
    if has(a, HARNESS) or has(b, HARNESS):
        return "harness", "a cable or loom, routed in CAD"
    if ca and ca == cb and (has(a, ("limb_cover",)) or has(b, ("limb_cover",))):
        return "cover", "the schematic limb cover against its own leg"
    if has(a, SHELL) or has(b, SHELL):
        return "shell", "a shell part: the shell is shaped around the parts in CAD"
    if "sole" in (base(a), base(b)):
        return "foot shape", "the simulated capsule is a full shape; the real rocker is a half capsule (F17)"
    if dist is not None and abs(dist) <= TOUCH_M:
        return "touch", "within 0.1 mm: resting or face to face"
    if (has(a, FRAME) and has(b, DRIVE)) or (has(b, FRAME) and has(a, DRIVE)):
        return "frame-drive", "a frame member runs into a drive housing"
    return "part", "two solid parts"


def solid_geoms(m) -> list[int]:
    """Parts with substance: not the floor, not mass-less collision proxies, not the probes."""
    out = []
    for g in range(m.ngeom):
        n = m.geom(g).name
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or not n:
            continue
        if n.endswith(("_collision", "_contact")) or n.startswith("probe_"):
            continue
        out.append(g)
    return out


def leg_body_ids(m, c) -> set[int]:
    return {m.body(f"{c}_{b}").id for b in LEG_BODIES}


def with_probe_rods(model_path: Path, root=None) -> mujoco.MjModel:
    """The model with one mocap probe per leg: a copy of the leg's pushrod capsule, contacts off, mass-less."""
    root = ET.parse(model_path).getroot() if root is None else root
    wb = root.find("worldbody")
    for c in CORNERS:
        rod = root.find(f".//geom[@name='{c}_pushrod']")
        ft = np.array([float(x) for x in rod.get("fromto").split()])
        half = float(np.linalg.norm(ft[3:] - ft[:3])) / 2
        b = ET.SubElement(wb, "body", {"name": f"probe_rod_{c}", "mocap": "true"})
        ET.SubElement(b, "geom", {"name": f"probe_rod_{c}", "type": "capsule", "size": rod.get("size"),
                                  "fromto": f"0 0 {-half:.9f} 0 0 {half:.9f}", "contype": "0",
                                  "conaffinity": "0", "mass": "0", "group": "5"})
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def linkage(m, c) -> dict:
    """The parallelogram in the upper link's frame: the drive axis and the elbow axis at the crank lane, the crank
    vector at neutral (from the model's rod ends), the rod's pin-to-pin length."""
    upper, lower = m.body(f"{c}_upper_link").id, m.body(f"{c}_lower_link").id
    rod, dc = m.geom(f"{c}_pushrod").id, m.geom(f"{c}_drive_crank").id
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[rod])
    axis, half = R.reshape(3, 3)[:, 2], m.geom_size[rod][1]
    ends = [m.geom_pos[rod] + s * half * axis for s in (+1, -1)]
    drive_axis = m.geom_pos[dc].copy()
    elbow = m.body_pos[lower] + np.array([0.0, drive_axis[1], 0.0])
    top = min(ends, key=lambda e: np.linalg.norm(e - drive_axis))
    bottom = min(ends, key=lambda e: np.linalg.norm(e - elbow))
    c_top, c_bot = top - drive_axis, bottom - elbow
    j = m.joint(f"{c}_flexion").id
    return {"upper": upper, "lower": lower, "drive_axis": drive_axis, "elbow": elbow, "crank": c_top,
            "crank_mismatch_m": float(np.linalg.norm(c_top - c_bot)), "crank_radius_m": float(np.linalg.norm(c_top)),
            "rod_length_m": float(np.linalg.norm(top - bottom)), "range_deg": np.degrees(m.jnt_range[j]).tolist()}


def place_rod(m, d, c, lk):
    """Put the probe where the parallelogram puts the rod; return (transmission angle deg, pin-to-pin length m)."""
    Ru = d.xmat[lk["upper"]].reshape(3, 3)
    Rl = d.xmat[lk["lower"]].reshape(3, 3)
    crank_w = Rl @ lk["crank"]                       # both cranks turn with the lower link (1:1, parallel axes)
    top = d.xpos[lk["upper"]] + Ru @ lk["drive_axis"] + crank_w
    bottom = d.xpos[lk["upper"]] + Ru @ lk["elbow"] + crank_w
    v = top - bottom
    L = float(np.linalg.norm(v))
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, v / L)
    mb = m.body(f"probe_rod_{c}").mocapid[0]
    d.mocap_pos[mb] = 0.5 * (top + bottom)
    d.mocap_quat[mb] = q
    cosang = abs(float(np.dot(crank_w, v))) / (np.linalg.norm(crank_w) * L)
    mujoco.mj_kinematics(m, d)
    return math.degrees(math.acos(min(1.0, cosang))), L


def near(m, d, g, h) -> bool:
    return np.linalg.norm(d.geom_xpos[g] - d.geom_xpos[h]) - m.geom_rbound[g] - m.geom_rbound[h] < REPORT_M


def measure(m, d, g, h, joints=None, pose=None) -> float:
    if m.geom_bodyid[g] == m.geom_bodyid[h]:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_type[h] == mujoco.mjtGeom.mjGEOM_BOX:
            return box_gap(body_aabb(m, g), body_aabb(m, h))
        return float(mujoco.mj_geomDistance(m, d, g, h, CAP_M, np.zeros(6)))   # no relative motion to nudge
    x = _dist(mujoco, m, d, g, h, CAP_M, np.zeros(6), joints, pose)
    if x == 0.0 and joints:
        # a second try at +-1e-3 rad (0.06 deg) on every joint of the sweep; kept if either reads non-zero
        tries = []
        for s in (-1e-3, 1e-3):
            for j in joints:
                d.qpos[j.qposadr[0]] += s
            mujoco.mj_kinematics(m, d)
            tries.append(float(mujoco.mj_geomDistance(m, d, g, h, CAP_M, np.zeros(6))))
            for j in joints:
                d.qpos[j.qposadr[0]] -= s
            mujoco.mj_kinematics(m, d)
        good = [t for t in tries if t != 0.0]
        if good:
            ZERO_RETRIES.pop()
            REMEASURED.append((m.geom(g).name, m.geom(h).name, pose, [round(1000 * t, 3) for t in tries]))
            x = min(good)
    return x


def set_pose(m, d, joints, q):
    d.qpos[:] = m.qpos0
    for j, v in zip(joints, q):
        d.qpos[m.jnt_qposadr[j]] = v
    mujoco.mj_kinematics(m, d)


def key(g, h):
    return (g, h) if g < h else (h, g)


def record(store, m, k, dist, pose_deg):
    cur = store.get(k)
    if cur is None:
        a, b = m.geom(k[0]).name, m.geom(k[1]).name
        cur = store[k] = {"a": a, "b": b, "min_mm": math.inf, "at_deg": None, "neutral_mm": None}
    if 1000 * dist < cur["min_mm"]:
        cur["min_mm"], cur["at_deg"] = 1000 * dist, pose_deg


def finish(store, m, neutral):
    out = []
    for k, p in store.items():
        if k in neutral:
            p["neutral_mm"] = 1000 * neutral[k]
        p["class"], p["reason"] = classify(p["a"], p["b"], p["min_mm"] / 1000)
        out.append(p)
    return sorted(out, key=lambda p: p["min_mm"])


def sweep(m, d, joints, grids, pairs, probe=None):
    """Least distance of each pair over the grid (pose 0 first, every pair measured there). ``probe`` = (corner,
    linkage, targets) also places that leg's probe rod at every pose and measures it against ``targets``."""
    store, neutral, rod = {}, {}, {"angle_min": 180.0, "at_flexion_deg": None, "length": [math.inf, 0.0]}
    jo = [m.joint(j) for j in joints]
    for q in itertools.chain([tuple(0.0 for _ in joints)], itertools.product(*grids)):
        set_pose(m, d, joints, q)
        zero = all(v == 0.0 for v in q)
        pose = [round(math.degrees(v), 1) for v in q]
        for g, h in pairs:
            if not zero and (m.geom_bodyid[g] == m.geom_bodyid[h] or not near(m, d, g, h)):
                continue
            x = measure(m, d, g, h, jo, pose)
            k = key(g, h)
            if zero:
                neutral[k] = x
            if x < REPORT_M or k in store:
                record(store, m, k, x, pose)
        if probe is not None:
            c, lk, targets = probe
            ang, L = place_rod(m, d, c, lk)
            if ang < rod["angle_min"]:
                rod["angle_min"], rod["at_flexion_deg"] = ang, pose[-1]
            rod["length"] = [min(rod["length"][0], L), max(rod["length"][1], L)]
            pg = m.geom(f"probe_rod_{c}").id
            for h in targets:
                if not zero and not near(m, d, pg, h):
                    continue
                x = float(mujoco.mj_geomDistance(m, d, pg, h, CAP_M, np.zeros(6)))
                k = (pg, h)
                if zero:
                    neutral[k] = x
                if x < REPORT_M or k in store:
                    record(store, m, k, x, pose)
    return store, neutral, rod


def leg(m, d, c, others, steps):
    """Sections 1 and 2 for one leg."""
    solid = solid_geoms(m)
    mine = [g for g in solid if m.geom_bodyid[g] in leg_body_ids(m, c)]
    joints = [m.joint(f"{c}_{a}").id for a in ("roll", "pitch", "flexion")]
    grids = [np.linspace(*m.jnt_range[j], steps) for j in joints]
    lk = linkage(m, c)
    own = {m.geom(f"{c}_{p}").id for p in ("drive_crank", "output_crank", "pushrod")}
    targets = [h for h in mine if h not in own] + list(others)
    pairs = [(g, h) for g in mine for h in others] + list(itertools.combinations(mine, 2))
    store, neutral, rod = sweep(m, d, joints, grids, pairs, (c, lk, targets))
    # the foot against the rest of its leg over flexion x ankle
    foot = m.body(f"{c}_foot").id
    fpairs = [(g, h) for g in mine if m.geom_bodyid[g] == foot for h in mine if m.geom_bodyid[h] != foot]
    fj = [joints[2], m.joint(f"{c}_ankle").id]
    fstore, _, _ = sweep(m, d, fj, [np.linspace(*m.jnt_range[j], steps) for j in fj], fpairs)
    for k, p in fstore.items():
        if k not in store or p["min_mm"] < store[k]["min_mm"]:
            p["at_deg"] = {"flexion, ankle": p["at_deg"]}
            store[k] = p
    probe = m.geom(f"probe_rod_{c}").id
    pairs_out = finish({k: v for k, v in store.items() if probe not in k}, m, neutral)
    rod_out = finish({k: v for k, v in store.items() if probe in k}, m, neutral)
    set_pose(m, d, [], [])
    gaps = {name: 1000 * measure(m, d, m.geom(f"{c}_{a}").id, m.geom(f"{c}_{b}").id) for name, (a, b) in {
        "elbow_axle_to_output_crank": ("elbow_axle", "output_crank"),
        "drive_crank_to_drive_housing": ("drive_crank", ("x6" if "fore" in c else "x4") + "_flexion_stator")}.items()}
    link = {"crank_radius_mm": 1000 * lk["crank_radius_m"], "crank_ends_differ_mm": 1000 * lk["crank_mismatch_m"],
            "rod_pin_to_pin_mm": 1000 * lk["rod_length_m"], "rod_length_over_range_mm": [1000 * x for x in rod["length"]],
            "flexion_range_deg": lk["range_deg"], "transmission_angle_min_deg": rod["angle_min"],
            "at_flexion_deg": rod["at_flexion_deg"], "axial_gaps_mm": gaps, "moving_rod": rod_out}
    return pairs_out, link


def body_sweep(m, d, bodies, joint_names, targets, steps):
    ids = {m.body(b).id for b in bodies}
    mine = [g for g in solid_geoms(m) if m.geom_bodyid[g] in ids]
    joints = [m.joint(j).id for j in joint_names]
    store, neutral, _ = sweep(m, d, joints, [np.linspace(*m.jnt_range[j], steps) for j in joints],
                              [(g, h) for g in mine for h in targets])
    return finish(store, m, neutral)


def bay(m, d) -> list:
    set_pose(m, d, [], [])
    ch = m.body("chassis").id
    parts = [g for g in solid_geoms(m) if m.geom_bodyid[g] == ch
             and not has(m.geom(g).name, ("PA11_rib", "shell_seam") + HARNESS)]
    store, neutral = {}, {}
    for g, h in itertools.combinations(parts, 2):
        x = measure(m, d, g, h)
        neutral[key(g, h)] = x
        if x < REPORT_M:
            record(store, m, key(g, h), x, [])
    return finish(store, m, neutral)


# The bay's tightest pairs (21 Nov 2026 review, item 5; 0.5-1.0 mm): how far each part, or a part with what it
# carries, can move in each direction before it meets another chassis part (the shell's ribs and seams included;
# cables and looms left out: routed in CAD). The legs are outboard of the rails and are not in this check.
ROOM_MOVERS = {
    "NVMe tray": ("nvme_and_vibration_tray",),
    "Jetson with its NVMe tray": ("jetson_orin_nano_super_kit", "nvme_and_vibration_tray"),
    "Pilz block": ("pilz_relay_contactor_watchdog_radio_rx",),
    "power board": ("power_board_diode_or_precharge_clamp",),
    "pack 1 (+y)": ("gtech_6s_3300_1",),
    "pack 2 (-y)": ("gtech_6s_3300_2",),
    "Teensy carriers": ("teensy41_can_carriers_x2",),
    "IMU": ("trunk_imu",),
    "blowers": ("drive_bay_blowers_and_ducts",),
}
ROOM_MAX_M, ROOM_STEP_M = 0.030, 0.00025


def bay_room(m, d) -> dict:
    """Free travel (mm, up to 30, in 0.25 mm steps) of each ROOM_MOVERS group along +-x, +-y, +-z of the chassis frame,
    and the part met first: a move stops where a pair that was clear (or touching) would overlap."""
    set_pose(m, d, [], [])
    ch = m.body("chassis").id
    pool = [g for g in solid_geoms(m) if m.geom_bodyid[g] == ch and not has(m.geom(g).name, HARNESS)]
    out = {}
    for label, names in ROOM_MOVERS.items():
        mv = [m.geom(n).id for n in names]
        others = [h for h in pool if h not in mv]
        pos0 = {g: m.geom_pos[g].copy() for g in mv}
        d0 = {(g, h): float(mujoco.mj_geomDistance(m, d, g, h, CAP_M, np.zeros(6))) for g in mv for h in others}
        row = {}
        for ax, sign in itertools.product(range(3), (+1, -1)):
            u = np.zeros(3)
            u[ax] = sign
            free, met = ROOM_MAX_M, None
            for t in np.arange(ROOM_STEP_M, ROOM_MAX_M + 1e-9, ROOM_STEP_M):
                for g in mv:
                    m.geom_pos[g] = pos0[g] + t * u
                mujoco.mj_kinematics(m, d)
                hit = [(g, h) for g in mv for h in others if near(m, d, g, h) and float(
                    mujoco.mj_geomDistance(m, d, g, h, CAP_M, np.zeros(6))) < min(d0[(g, h)], 0.0) - 1e-6]
                if hit:
                    free, met = t - ROOM_STEP_M, sorted({m.geom(h).name for _, h in hit})
                    break
            for g in mv:
                m.geom_pos[g] = pos0[g]
            mujoco.mj_kinematics(m, d)
            row[("+" if sign > 0 else "-") + "xyz"[ax]] = {"free_mm": round(1000 * free, 2), "meets": met}
        out[label] = row
    set_pose(m, d, [], [])
    return out


# The legs against each other. A leg's parts move with its own four joints only, so each leg is posed on its own grid
# and every pose of one leg is measured against every pose of the other: PAIR_STEPS values per joint over all eight
# joints of the pair (roll, pitch, flexion and ankle of each; 9 values give 43 million poses for two feet). A part that
# does not move with its own body's joint (a capsule or cylinder on that joint's axis: the elbow axle, the output
# crank, the ankle hub) is posed on the joints before it only. Every part is held in a capsule that contains it (the
# part itself for a capsule, exact; a cylinder's capsule, a box's capsule along its longest side) and the capsules'
# distance is computed in closed form at every grid pose; for a part that is not a capsule, the grid poses whose
# bound comes within SEARCH_M are then measured with mj_geomDistance. From the N_STARTS deepest grid local minima of
# every pair of parts that comes within SEARCH_M, a bounded local search (Nelder-Mead over the joints the two parts
# move with, mj_geomDistance at every step) finds the closest approach between grid points. The moving rod (the
# probe) stands in for the model's fixed rod, which does not follow the linkage.
CHAIN = ("roll", "pitch", "flexion", "ankle")
LEG_PAIRS = tuple(itertools.combinations(CORNERS, 2))
PAIR_STEPS = 9        # grid values per joint for the legs against each other
SEARCH_M = CAP_M      # a pair of parts that comes this near at some grid pose is searched between grid points
N_STARTS = 8          # local searches per pair of parts, from its deepest grid local minima
LEAST_KINDS = {       # which pairs count for each "least" figure
    "any part": lambda a, b: True,
    "solid parts (no covers)": lambda a, b: "limb_cover" not in a + b,
    "solid parts, soles aside": lambda a, b: "limb_cover" not in a + b and not a.endswith("_sole")
    and not b.endswith("_sole"),
}
MAP_KINDS = ("solid parts (no covers)", "solid parts, soles aside")


def seg_dist(p0, p1, q0, q1):
    """Least distance between the segments p0-p1 and q0-q1 (broadcast over leading axes; zero-length segments
    allowed), in closed form (Ericson, Real-Time Collision Detection, 5.1.9)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = np.einsum("...i,...i->...", d1, d1)
    e = np.einsum("...i,...i->...", d2, d2)
    f = np.einsum("...i,...i->...", d2, r)
    c = np.einsum("...i,...i->...", d1, r)
    b = np.einsum("...i,...i->...", d1, d2)
    eps = 1e-18
    with np.errstate(divide="ignore", invalid="ignore"):
        den = a * e - b * b
        s = np.where(den > eps, np.clip((b * f - c * e) / np.where(den > eps, den, 1.0), 0.0, 1.0), 0.0)
        t = np.where(e > eps, (b * s + f) / np.where(e > eps, e, 1.0), 0.0)
        sa = np.where(a > eps, a, 1.0)
        s = np.where(t < 0.0, np.clip(-c / sa, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / sa, 0.0, 1.0), s))
        t = np.clip(t, 0.0, 1.0)
        s = np.where(a > eps, s, 0.0)
        t = np.where(a > eps, t, np.where(e > eps, np.clip(f / np.where(e > eps, e, 1.0), 0.0, 1.0), 0.0))
    diff = (p0 + d1 * s[..., None]) - (q0 + d2 * t[..., None])
    return np.sqrt(np.einsum("...i,...i->...", diff, diff))


def bound_capsule(m, g):
    """(half-axis vector in the geom frame, radius) of a capsule that contains the geom; exact for a capsule."""
    t, s = m.geom_type[g], m.geom_size[g]
    G = mujoco.mjtGeom
    if t in (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER):
        return np.array([0.0, 0.0, s[1]]), float(s[0])
    if t == G.mjGEOM_SPHERE:
        return np.zeros(3), float(s[0])
    if t == G.mjGEOM_BOX:
        k = int(np.argmax(s[:3]))
        u = np.zeros(3)
        u[k] = s[k]
        return u, float(np.linalg.norm(np.delete(s[:3], k)))
    return np.zeros(3), float(m.geom_rbound[g])


def leg_grid(m, d, c, steps, ranges=None, only=None) -> dict:
    """The leg's parts (solid, the moving rod in place of the fixed one) posed at every grid pose of the joints they
    move with: world position, orientation and bounding segment. ``ranges`` = {joint: (lo, hi) rad} narrows a joint
    (a candidate limit); ``only`` keeps the parts whose names contain it."""
    joints = [m.joint(f"{c}_{a}").id for a in CHAIN]
    rng = [tuple((ranges or {}).get(a, m.jnt_range[j])) for a, j in zip(CHAIN, joints)]
    grids = [np.linspace(lo, hi, steps) for lo, hi in rng]
    probe = m.geom(f"probe_rod_{c}").id
    parts = [g for g in solid_geoms(m) if m.geom_bodyid[g] in leg_body_ids(m, c) and m.geom(g).name != f"{c}_pushrod"]
    parts += [probe]
    if only:
        parts = [g for g in parts if only in m.geom(g).name]
    info = {}
    for g in parts:
        k = 3 if g == probe else LEG_BODIES.index(m.body(m.geom_bodyid[g]).name[len(c) + 1:]) + 1
        half, rad = bound_capsule(m, g)
        info[g] = {"depth": k, "half": half, "r": rad, "name": m.geom(g).name,
                   "capsule": m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE)}
    for g, p in info.items():       # a part whose axis lies on its own body's joint axis does not move with it
        if g == probe or p["depth"] == 1 or m.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_CAPSULE,
                                                                    mujoco.mjtGeom.mjGEOM_CYLINDER):
            continue
        j = joints[p["depth"] - 1]
        ends = []
        for v in m.jnt_range[j]:
            set_pose(m, d, [j], [v])
            R = d.geom_xmat[g].reshape(3, 3)
            ends.append(np.array(sorted(tuple(d.geom_xpos[g] + s * R @ p["half"]) for s in (-1, 1))))
        if np.abs(ends[0] - ends[1]).max() < 1e-9:
            p["depth"] -= 1
    anc = m.jnt_pos[joints[3]]
    for g, p in info.items():
        n = steps ** p["depth"]
        p.update(x=np.zeros((n, 3)), mat=np.zeros((n, 9)), p0=np.zeros((n, 3)), p1=np.zeros((n, 3)))
        if p["depth"] == 4:          # a sphere about the ankle that holds the part at any ankle angle (pruning)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            p["anchor_r"] = max(np.linalg.norm(m.geom_pos[g] + s * R.reshape(3, 3) @ p["half"] - anc)
                                for s in (-1, 1)) + p["r"]
            p["anchor"] = np.zeros((steps ** 3, 3))
    lk = linkage(m, c)
    for k in range(1, 5):
        mine = [g for g, p in info.items() if p["depth"] == k]
        anchor = k == 3 and any(p["depth"] == 4 for p in info.values())
        if not mine and not anchor:
            continue
        for i, idx in enumerate(itertools.product(range(steps), repeat=k)):
            set_pose(m, d, joints[:k], [grids[a][b] for a, b in enumerate(idx)])
            if probe in mine:
                place_rod(m, d, c, lk)
            for g in mine:
                p = info[g]
                R = d.geom_xmat[g].reshape(3, 3)
                p["x"][i], p["mat"][i] = d.geom_xpos[g], d.geom_xmat[g]
                p["p0"][i], p["p1"][i] = d.geom_xpos[g] - R @ p["half"], d.geom_xpos[g] + R @ p["half"]
            if anchor:
                for p in info.values():
                    if p["depth"] == 4:
                        p["anchor"][i] = d.xanchor[joints[3]]
    set_pose(m, d, [], [])
    return {"corner": c, "joints": joints, "grids": grids, "ranges": rng, "parts": info, "steps": steps, "lk": lk}


def _pose(LA, LB, ka, kb, flat):
    """Joint values (rad) of a flat index into a pair's (steps,)*(ka+kb) grid."""
    idx = np.unravel_index(int(flat), (LA["steps"],) * (ka + kb))
    return np.array([LA["grids"][k][idx[k]] for k in range(ka)] + [LB["grids"][k][idx[ka + k]] for k in range(kb)])


def pair_exact(m, d, LA, LB, ga, gb, q):
    """mj_geomDistance between part ga of leg A and gb of leg B at joint values q (A's then B's, as many as the parts
    move with); an exact 0 is re-measured 0.06 deg away (``measure``)."""
    ka, kb = LA["parts"][ga]["depth"], LB["parts"][gb]["depth"]
    js = LA["joints"][:ka] + LB["joints"][:kb]
    set_pose(m, d, js, q)
    for L, g in ((LA, ga), (LB, gb)):
        if m.geom(g).name.startswith("probe_rod_"):
            place_rod(m, d, L["corner"], L["lk"])
    x = float(mujoco.mj_geomDistance(m, d, ga, gb, CAP_M, np.zeros(6)))
    if x == 0.0:
        x = measure(m, d, ga, gb, [m.joint(j) for j in js], [round(math.degrees(v), 2) for v in q])
    return x


def pair_field(m, d, LA, LB, ga, gb, thr=SEARCH_M, chunk=2_000_000):
    """The distance of part ga (leg A) and gb (leg B) at every pose of their joint grids, (steps**ka, steps**kb),
    float32, inf where it is thr or more; None if the two never come within thr."""
    pa, pb = LA["parts"][ga], LB["parts"][gb]
    S, ka, kb = LA["steps"], pa["depth"], pb["depth"]
    box = [(np.minimum(p["p0"], p["p1"]).min(0) - p["r"], np.maximum(p["p0"], p["p1"]).max(0) + p["r"]) for p in (pa, pb)]
    if (np.maximum(box[0][0] - box[1][1], box[1][0] - box[0][1]) >= thr).any():
        return None
    lv = [(p["anchor"], p["anchor"], p["anchor_r"], 3) if p["depth"] == 4 else (p["p0"], p["p1"], p["r"], p["depth"])
          for p in (pa, pb)]
    (A0, A1, rA, la), (B0, B1, rB, lb) = lv
    V = np.full((S ** ka, S ** kb), np.inf, np.float32)
    rows = max(1, chunk // len(B0))
    survivors = []
    for s0 in range(0, len(A0), rows):
        sl = slice(s0, min(len(A0), s0 + rows))
        v = seg_dist(A0[sl, None], A1[sl, None], B0[None], B1[None]) - rA - rB
        if la == ka and lb == kb:
            V[sl] = np.where(v < thr, v, np.inf)
        else:
            ia, ib = np.nonzero(v < thr)
            survivors.append((ia + s0, ib))
    if survivors:                  # a foot part: expand the grid poses that the ankle's sphere did not rule out
        ia1 = np.concatenate([s[0] for s in survivors])
        ib1 = np.concatenate([s[1] for s in survivors])
        ea, eb = (S if ka == 4 else 1), (S if kb == 4 else 1)
        rows = max(1, chunk // (ea * eb))
        for s0 in range(0, len(ia1), rows):
            a1, b1 = ia1[s0:s0 + rows], ib1[s0:s0 + rows]
            ia = (a1[:, None, None] * ea + np.arange(ea)[None, :, None] + 0 * np.arange(eb)[None, None, :]).ravel()
            ib = (b1[:, None, None] * eb + np.arange(eb)[None, None, :] + 0 * np.arange(ea)[None, :, None]).ravel()
            v = seg_dist(pa["p0"][ia], pa["p1"][ia], pb["p0"][ib], pb["p1"][ib]) - pa["r"] - pb["r"]
            V[ia, ib] = np.where(v < thr, v, np.inf)
    if not (pa["capsule"] and pb["capsule"]):      # the bound is not the distance: measure the poses it lets through
        for ia, ib in zip(*np.nonzero(np.isfinite(V))):
            d.geom_xpos[ga], d.geom_xmat[ga] = pa["x"][ia], pa["mat"][ia]
            d.geom_xpos[gb], d.geom_xmat[gb] = pb["x"][ib], pb["mat"][ib]
            x = float(mujoco.mj_geomDistance(m, d, ga, gb, CAP_M, np.zeros(6)))
            if x == 0.0:
                x = pair_exact(m, d, LA, LB, ga, gb, _pose(LA, LB, ka, kb, ia * S ** kb + ib))
            V[ia, ib] = x if x < thr else np.inf
        set_pose(m, d, [], [])
    return V if np.isfinite(V).any() else None


def grid_minima(V, dims, steps):
    """Flat indices and values of the grid's local minima (no axis neighbour lower), finite entries only."""
    W = V.reshape((steps,) * dims)
    ok = np.isfinite(W)
    for ax in range(dims):
        lo, hi = [slice(None)] * dims, [slice(None)] * dims
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        a, b = W[tuple(lo)], W[tuple(hi)]
        ok[tuple(lo)] &= a <= b
        ok[tuple(hi)] &= b <= a
    flat = np.flatnonzero(ok)
    return flat, W.ravel()[flat]


def _starts(flat, val, dims, steps, n=N_STARTS):
    """The n deepest local minima, skipping any within one grid step (every joint) of one already taken."""
    out = []
    for i in np.argsort(val, kind="stable")[:5000]:
        idx = np.array(np.unravel_index(int(flat[i]), (steps,) * dims))
        if all(np.abs(idx - o).max() > 1 for _, o in out):
            out.append((int(flat[i]), idx))
        if len(out) == n:
            break
    return [f for f, _ in out]


def pair_search(m, d, LA, LB, ga, gb, starts):
    """Bounded Nelder-Mead from each start (joint values, rad), first with a simplex of half a grid step per joint,
    then a quarter; returns (least distance m, its joint values)."""
    ka, kb = LA["parts"][ga]["depth"], LB["parts"][gb]["depth"]
    rng = LA["ranges"][:ka] + LB["ranges"][:kb]
    lo, hi = np.array([r[0] for r in rng]), np.array([r[1] for r in rng])
    h = (hi - lo) / (LA["steps"] - 1) / 2

    def f(x):
        return pair_exact(m, d, LA, LB, ga, gb, np.clip(x, lo, hi))

    best = (math.inf, None)
    for x0 in starts:
        x, v = np.array(x0, float), f(x0)
        for scale in (1.0, 0.25):
            sim = [x]
            for k in range(len(x)):
                e = x.copy()
                e[k] = e[k] + scale * h[k] if e[k] + scale * h[k] <= hi[k] else e[k] - scale * h[k]
                sim.append(e)
            r = minimize(f, x, method="Nelder-Mead", bounds=list(zip(lo, hi)),
                         options={"initial_simplex": np.array(sim), "xatol": 1e-5, "fatol": 1e-7, "maxfev": 3000})
            if r.fun < v:
                x, v = np.clip(r.x, lo, hi), float(r.fun)
        if v < best[0]:
            best = (v, x.copy())
    set_pose(m, d, [], [])
    return best


def _named(LA, LB, ka, kb, q) -> dict:
    names = [f"{LA['corner']}_{a}" for a in CHAIN[:ka]] + [f"{LB['corner']}_{a}" for a in CHAIN[:kb]]
    return {n: round(math.degrees(v), 2) for n, v in zip(names, q)}


def legs_to_each_other(m, d, steps=PAIR_STEPS, leg_pairs=LEG_PAIRS, ranges=None, only=None) -> dict:
    """For each pair of legs: every pair of their parts that comes within REPORT_M anywhere in the joints' ranges, with
    its least distance on the grid and after the local search, the pose, and the share of grid poses where the two
    overlap; the least distance of each LEAST_KINDS; and where the solid parts overlap on the grid (MAP_KINDS: the
    share of the grid's poses, and each joint's grid values that take part). ``ranges`` = {corner: {joint: (lo, hi)
    rad}} narrows joints (``leg_limits``)."""
    L = {c: leg_grid(m, d, c, steps, (ranges or {}).get(c), only) for c in sorted({c for p in leg_pairs for c in p})}
    out = {}
    for a, b in leg_pairs:
        LA, LB = L[a], L[b]
        rows, maps = [], {}
        least = {k: {"min_mm": 1000 * CAP_M, "pair": None, "class": None, "at_deg": None} for k in LEAST_KINDS}
        for ga, pa in LA["parts"].items():
            for gb, pb in LB["parts"].items():
                V = pair_field(m, d, LA, LB, ga, gb)
                if V is None:
                    continue
                ka, kb = pa["depth"], pb["depth"]
                flat, val = grid_minima(V, ka + kb, steps)
                starts = [_pose(LA, LB, ka, kb, f) for f in _starts(flat, val, ka + kb, steps)]
                v, q = pair_search(m, d, LA, LB, ga, gb, starts)
                na, nb = pa["name"], pb["name"]
                row = {"a": na, "b": nb, "class": classify(na, nb, v)[0], "grid_min_mm": 1000 * float(V.min()),
                       "min_mm": 1000 * v, "at_deg": _named(LA, LB, ka, kb, q),
                       "grid_share_overlapping": float((V < 0).mean())}
                for k, keep in LEAST_KINDS.items():
                    if keep(na, nb) and 1000 * v < least[k]["min_mm"]:
                        least[k] = {"min_mm": 1000 * v, "pair": [na, nb], "class": row["class"], "at_deg": row["at_deg"]}
                if v < REPORT_M:
                    rows.append(row)
                over = V < 0
                if over.any():
                    shape = (steps,) * ka + (1,) * (4 - ka) + (steps,) * kb + (1,) * (4 - kb)
                    for k in MAP_KINDS:
                        if LEAST_KINDS[k](na, nb):
                            if k not in maps:
                                maps[k] = np.zeros((steps,) * 8, bool)
                            maps[k] |= over.reshape(shape)
        names = [f"{c}_{x}" for c in (a, b) for x in CHAIN]
        grid_deg = [np.degrees(g) for g in LA["grids"]] + [np.degrees(g) for g in LB["grids"]]
        region = {}
        for k, mp in maps.items():
            used, inside = {}, []
            for ax, n in enumerate(names):
                vals = grid_deg[ax][mp.any(axis=tuple(i for i in range(8) if i != ax))]
                used[n] = [round(float(vals.min()), 2), round(float(vals.max()), 2)]
                inside.append((grid_deg[ax] >= vals.min()) & (grid_deg[ax] <= vals.max()))
            rolls = mp.any(axis=(1, 2, 3, 5, 6, 7))          # (roll of leg A, roll of leg B) pairs that overlap
            ia, ib = np.nonzero(rolls)
            region[k] = {"share_of_grid_poses": float(mp.mean()), "joint_grid_values_taking_part_deg": used,
                         "share_of_the_poses_inside_these_bounds": float(mp[np.ix_(*inside)].mean()),
                         "largest_roll_difference_deg": round(float(np.abs(grid_deg[0][ia] - grid_deg[4][ib]).max()), 2)}
        out[f"{a} / {b}"] = {"values_per_joint": steps, "least": least, "region_overlapping_on_the_grid": region,
                             "pairs_nearer_than_10mm": sorted(rows, key=lambda r: r["min_mm"])}
    set_pose(m, d, [], [])
    return out


def rocker_overlap(m, d, poses: dict, step=0.001) -> float:
    """Overlap volume (cm^3, 1 mm voxels) of two soles' lower halves at ``poses`` ({joint name: deg}): F17's real foot
    is a half-capsule rocker, the half of the simulated capsule below its axis in the foot's frame."""
    js = [m.joint(n).id for n in poses]
    set_pose(m, d, js, np.radians(list(poses.values())))
    corners = sorted({n.rsplit("_", 1)[0] for n in poses})
    soles = [m.geom(f"{c}_sole").id for c in corners]
    half, r = bound_capsule(m, soles[0])
    ax = np.arange(-r, r + 1e-9, step)
    z = np.arange(-half[2] - r, half[2] + r + 1e-9, step)
    P = np.stack(np.meshgrid(ax, ax, z, indexing="ij"), -1).reshape(-1, 3)

    def inside(Q, g):                       # Q in g's geom frame: inside the capsule and below its axis (foot frame)
        down = d.geom_xmat[g].reshape(3, 3).T @ d.xmat[m.geom_bodyid[g]].reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
        zc = np.clip(Q[:, 2], -half[2], half[2])
        return (np.linalg.norm(Q - np.c_[np.zeros((len(Q), 2)), zc], axis=1) <= r) & (Q @ down >= 0)

    A, B = soles
    PA = P[inside(P, A)]
    W = PA @ d.geom_xmat[A].reshape(3, 3).T + d.geom_xpos[A]
    QB = (W - d.geom_xpos[B]) @ d.geom_xmat[B].reshape(3, 3)
    vol = float(inside(QB, B).sum()) * step ** 3 * 1e6
    set_pose(m, d, [], [])
    return vol


# Candidate joint limits that would keep the legs apart (proposals for the firmware or the CAD, not designs): each
# narrows the named joints on the named legs and reruns the search for the pairs of legs they affect.
LIMITS = {   # deg; the left side stands for both (the model is mirror-symmetric, and so are the results above)
    "hind inward roll stop, both hind legs (today -18)": (
        ("left_hind", "right_hind"), [(("left_hind", "right_hind"), {"roll": (lo, None)}) for lo in (-17, -16, -14)]),
    "fore pitch aft stop, the fore leg only (today +38; + swings the foot aft)": (
        ("left_fore", "left_hind"), [(("left_fore",), {"pitch": (None, p)}) for p in (20, 10, 5, 0)]),
    "hind pitch forward stop, the hind leg only (today -38; - swings the foot forward)": (
        ("left_fore", "left_hind"), [(("left_hind",), {"pitch": (p, None)}) for p in (-30, -20, -15, -10)]),
    "the same pitch stop on both, fore aft and hind forward (today 38 each)": (
        ("left_fore", "left_hind"), [(("left_fore",), {"pitch": (None, p)}, ("left_hind",), {"pitch": (-p, None)})
                                     for p in (20, 15, 10)]),
    "elbow and knee flexion stop, both legs (today -47)": (
        ("left_fore", "left_hind"), [(("left_fore", "left_hind"), {"flexion": (lo, None)}) for lo in (-30, -20, -10)]),
}


def leg_limits(m, d, steps=PAIR_STEPS) -> dict:
    """The legs against each other under each candidate limit in LIMITS: the least of each LEAST_KINDS."""
    out = {}
    for label, (pair, cases) in LIMITS.items():
        rows = []
        for case in cases:
            ranges = {}
            for legs, joints in zip(case[::2], case[1::2]):
                for c in legs:
                    for jn, (lo, hi) in joints.items():
                        r = m.jnt_range[m.joint(f"{c}_{jn}").id]
                        ranges.setdefault(c, {})[jn] = (r[0] if lo is None else math.radians(lo),
                                                        r[1] if hi is None else math.radians(hi))
            res = legs_to_each_other(m, d, steps, (pair,), ranges)[f"{pair[0]} / {pair[1]}"]
            rows.append({"ranges_deg": {c: {j: [round(math.degrees(v), 2) for v in r] for j, r in jr.items()}
                                        for c, jr in ranges.items()},
                         "least": res["least"]})
        out[label] = rows
    return out


def hind_roll_limit(m, d, steps, wants=(0.0, 0.003)) -> dict:
    """The most inward hind roll at which the hind linkage (drive crank, output crank, the moving rod) clears the
    rails by each wanted margin over the pitch x flexion grid: roll stepped by 0.5 deg from the inward stop."""
    out = {}
    for c in ("left_hind", "right_hind"):
        side = c.split("_")[0]
        rail = m.geom(f"6061_{side}_rail").id
        parts = [m.geom(f"{c}_{p}").id for p in ("drive_crank", "output_crank")]
        lk = linkage(m, c)
        jr, jp, jf = (m.joint(f"{c}_{a}").id for a in ("roll", "pitch", "flexion"))
        grid = list(itertools.product(np.linspace(*m.jnt_range[jp], steps), np.linspace(*m.jnt_range[jf], steps)))
        res, at = {}, {}
        for roll in np.arange(m.jnt_range[jr][0], 0.0 + 1e-9, math.radians(0.5)):
            worst = CAP_M
            for qp, qf in grid:
                set_pose(m, d, [jr, jp, jf], [roll, qp, qf])
                place_rod(m, d, c, lk)
                for g in parts + [m.geom(f"probe_rod_{c}").id]:
                    worst = min(worst, float(mujoco.mj_geomDistance(m, d, g, rail, CAP_M, np.zeros(6))))
            at[round(math.degrees(roll), 1)] = 1000 * worst
            for w in wants:
                if w not in res and worst >= w:
                    res[w] = round(math.degrees(roll), 1)
        out[c] = {f"clear by {1000 * w:.0f} mm from roll (deg)": res.get(w) for w in wants}
        out[c]["least_mm_at_inward_stop"] = at[min(at)]
        out[c]["least_mm_at_roll_0"] = at[max(at)]
    set_pose(m, d, [], [])
    return out


def check_envelopes(m) -> list[str]:
    drv = json.loads(DRIVES.read_text())["leg_drives"]
    out = []
    for g in solid_geoms(m):
        n = m.geom(g).name
        if n.endswith("_stator"):
            kind = "X6-60" if "x6_" in n else "X4-36-L"
            r, half = m.geom_size[g][0], m.geom_size[g][1]
            if abs(2 * r - drv[kind]["diameter_m"]) > 1e-6 or abs(2 * half - drv[kind]["length_m"]) > 1e-6:
                out.append(f"{n}: {2000 * r:.1f} x {2000 * half:.1f} mm against drives.json {kind}")
    return out


# Candidate positions put to the trial (chassis frame, m). Each is a proposal to test, not a design.
TRIALS = {
    "shoulder crossmember between the rails only (|y| <= 83.2 mm)": (
        "legs", {"shoulder_crossmember": {"fromto": "0.26 -0.0832 0 0.26 0.0832 0"}}),
    "shoulder crossmember between the rails, raised 6 mm": (
        "legs", {"shoulder_crossmember": {"fromto": "0.26 -0.0832 0.006 0.26 0.0832 0.006"}}),
    "shoulder crossmember between the rails, raised 10 mm": (
        "legs", {"shoulder_crossmember": {"fromto": "0.26 -0.0832 0.010 0.26 0.0832 0.010"}}),
    "shoulder crossmember between the rails, 48 mm aft (x 212 mm)": (
        "legs", {"shoulder_crossmember": {"fromto": "0.212 -0.0832 0 0.212 0.0832 0"}}),
    "pelvic crossmember between the rails only": (
        "legs", {"pelvic_crossmember": {"fromto": "-0.26 -0.0832 0 -0.26 0.0832 0"}}),
    "pelvic crossmember between the rails, 45 mm up (z +45 mm)": (
        "legs", {"pelvic_crossmember": {"fromto": "-0.26 -0.0832 0.045 -0.26 0.0832 0.045"}}),
    "pelvic crossmember between the rails, 48 mm forward (x -212 mm), 4 mm up": (
        "legs", {"pelvic_crossmember": {"fromto": "-0.212 -0.0832 0.004 -0.212 0.0832 0.004"}}),
    "rails ending 15 mm inside the roll drives' faces (x -215 to +215 mm)": (
        "legs", {"6061_left_rail": {"fromto": "-0.215 0.0832 -0.012 0.215 0.0832 -0.012"},
                 "6061_right_rail": {"fromto": "-0.215 -0.0832 -0.012 0.215 -0.0832 -0.012"}}),
    "rails ending at x -215 / +215 mm and 8 mm further in (y +-75.2 mm)": (
        "legs", {"6061_left_rail": {"fromto": "-0.215 0.0752 -0.012 0.215 0.0752 -0.012"},
                 "6061_right_rail": {"fromto": "-0.215 -0.0752 -0.012 0.215 -0.0752 -0.012"}}),
    "tail pitch servo 7 mm aft (x -31 mm in the tail frame)": (
        "tail", {"xl430_w250_tail_pitch_case": {"pos": "-0.031 0 0"}}),
    "tail pitch servo 16 mm aft (x -40 mm in the tail frame)": (
        "tail", {"xl430_w250_tail_pitch_case": {"pos": "-0.040 0 0"}}),
}
# The single trials above each move one part against today's layout; the rail and crossmember fixes interact (rails
# ending at x +-215 mm leave crossmembers at x +-260 mm 45 mm past their ends). Three combined layouts (A, B, C), all on
# the shortened, inset rails, with every roll drive carried on a bracket from the rail end to its inboard face (the
# front drop brackets moved there, two hind ones added: the rails no longer pass through the hind drives):
_RAILS_6 = {"6061_left_rail": {"fromto": "-0.215 0.0752 -0.012 0.215 0.0752 -0.012"},
            "6061_right_rail": {"fromto": "-0.215 -0.0752 -0.012 0.215 -0.0752 -0.012"}}
_BRACKET = {"type": "capsule", "size": "0.006", "material": "bracket", "mass": "0.030"}
_BRACKETS = {"left_fore_drop_bracket": {"fromto": "0.212 0.0752 -0.012 0.228 0.076 -0.04"},
             "right_fore_drop_bracket": {"fromto": "0.212 -0.0752 -0.012 0.228 -0.076 -0.04"},
             "left_hind_drop_bracket": dict(_BRACKET, fromto="-0.212 0.0752 -0.012 -0.228 0.076 0"),
             "right_hind_drop_bracket": dict(_BRACKET, fromto="-0.212 -0.0752 -0.012 -0.228 -0.076 0")}
# In C the four tray stays are re-angled to end on the inset rails' axis, their tray ends kept. As drawn they end on the
# old rail line (y +-83.2 mm, fore128.xml:71-74), which an inset rail would leave them standing 3.2 mm outboard of
# (88.0 mm against the rail's outer surface at 84.8 mm), where the rear pair passes the hind cranks at 1.89 mm.
_STAYS_INSET = {"tray_stay_-0.255_1": {"fromto": "-0.204 0.0544 -0.06 -0.204 0.0752 -0.012"},
                "tray_stay_-0.255_-1": {"fromto": "-0.204 -0.0544 -0.06 -0.204 -0.0752 -0.012"},
                "tray_stay_0.095_1": {"fromto": "0.076 0.0544 -0.06 0.076 0.0752 -0.012"},
                "tray_stay_0.095_-1": {"fromto": "0.076 -0.0544 -0.06 0.076 -0.0752 -0.012"}}
TRIALS.update({
    "combined A: rails as in the last rail trial, crossmembers kept at x +-260 mm between the rails' lines (shoulder "
    "10 mm up, pelvic 45 mm up), so their ends sit over the roll drives and need brackets from the drives' mounts": (
        "legs", dict(_RAILS_6, **_BRACKETS,
                     shoulder_crossmember={"fromto": "0.26 -0.0752 0.010 0.26 0.0752 0.010"},
                     pelvic_crossmember={"fromto": "-0.26 -0.0752 0.045 -0.26 0.0752 0.045"})),
    "combined B: rails as in the last rail trial, crossmembers moved onto the rails' ends (x +-212 mm, a ladder frame; "
    "shoulder at z 0, pelvic 4 mm up)": (
        "legs", dict(_RAILS_6, **_BRACKETS,
                     shoulder_crossmember={"fromto": "0.212 -0.0752 0 0.212 0.0752 0"},
                     pelvic_crossmember={"fromto": "-0.212 -0.0752 0.004 -0.212 0.0752 0.004"})),
    "combined C: as B, with the DC-DC block 20 mm aft and 3 mm up and the I/O carrier 20 mm forward, out of the "
    "crossmembers' way, and the four tray stays re-angled to end on the inset rails": (
        "legs", dict(_RAILS_6, **_BRACKETS, **_STAYS_INSET,
                     shoulder_crossmember={"fromto": "0.212 -0.0752 0 0.212 0.0752 0"},
                     pelvic_crossmember={"fromto": "-0.212 -0.0752 0.004 -0.212 0.0752 0.004"},
                     dc_dc_rails_and_pack_monitor={"pos": "0.165 0 0.003"},
                     io_and_sensor_carriers={"pos": "-0.17 0 -0.001"})),
})
# What each frame member must still meet after a trial moves it (signed distance at neutral, mm; <= 0 means the two
# meet and can be joined): the rails' ends, the parts the crossmembers carry (the neck cradle and the ToF arms' roots
# at the front, the tail's pivot at the back) and the roll drives' housings (a bracket or saddle bridges a gap there);
# a tray stay, its rail and the tray it carries.
MOUNTS = {
    "shoulder_crossmember": ("6061_left_rail", "6061_right_rail", "neck_cradle", "left_tof_mount_root",
                             "right_tof_mount_root", "left_fore_x4_roll_stator", "right_fore_x4_roll_stator"),
    "pelvic_crossmember": ("6061_left_rail", "6061_right_rail", "tail pivot", "left_hind_x4_roll_stator",
                           "right_hind_x4_roll_stator"),
    "left_fore_drop_bracket": ("6061_left_rail", "left_fore_x4_roll_stator"),
    "right_fore_drop_bracket": ("6061_right_rail", "right_fore_x4_roll_stator"),
    "left_hind_drop_bracket": ("6061_left_rail", "left_hind_x4_roll_stator"),
    "right_hind_drop_bracket": ("6061_right_rail", "right_hind_x4_roll_stator"),
    "tray_stay_-0.255_1": ("6061_left_rail", "removable_service_tray"),
    "tray_stay_-0.255_-1": ("6061_right_rail", "removable_service_tray"),
    "tray_stay_0.095_1": ("6061_left_rail", "removable_service_tray"),
    "tray_stay_0.095_-1": ("6061_right_rail", "removable_service_tray"),
}


def mounts(m, d, names) -> dict:
    """For each moved frame member in MOUNTS, its signed distance (mm, neutral pose) to each part it must meet."""
    set_pose(m, d, [], [])
    out = {}
    for n in names:
        if n not in MOUNTS:
            continue
        g = m.geom(n).id
        row = {}
        for t in MOUNTS[n]:
            if t == "tail pivot":     # the tail body's origin: a point; distance to the member's surface
                p = d.xpos[m.body("tail").id]
                a, b = (d.geom_xpos[g] + s * m.geom_size[g][1] * d.geom_xmat[g].reshape(3, 3)[:, 2] for s in (-1, 1))
                u = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
                row[t] = 1000 * (float(np.linalg.norm(p - (a + u * (b - a)))) - m.geom_size[g][0])
            else:
                row[t] = 1000 * float(mujoco.mj_geomDistance(m, d, g, m.geom(t).id, CAP_M, np.zeros(6)))
        out[n] = row
    return out


def trial(model_path: Path, edits: dict, steps: int, kind: str) -> dict:
    root = ET.parse(model_path).getroot()
    chassis = root.find(".//body[@name='chassis']")
    for name, attrs in edits.items():
        el = root.find(f".//geom[@name='{name}']")
        if el is None:            # a part the trial adds (a bracket the fix needs), on the chassis
            el = ET.SubElement(chassis, "geom", {"name": name})
        for k, v in attrs.items():
            el.set(k, v)
    m = with_probe_rods(model_path, root)
    d = mujoco.MjData(m)
    moved = [m.geom(n).id for n in edits]
    ch = [h for h in solid_geoms(m) if m.geom_bodyid[h] == m.body("chassis").id and h not in moved]
    # every moved part against the chassis parts left in place, and against every other moved part (a trial that
    # moves two parts out of each other's way must show they clear each other, not only the parts left in place)
    chassis_pairs = [(g, h) for g in moved for h in ch] + list(itertools.combinations(moved, 2))
    store, neutral = {}, {}
    if kind == "legs":
        for c in CORNERS:
            mine = [g for g in solid_geoms(m) if m.geom_bodyid[g] in leg_body_ids(m, c)]
            joints = [m.joint(f"{c}_{a}").id for a in ("roll", "pitch", "flexion")]
            s, n, _ = sweep(m, d, joints, [np.linspace(*m.jnt_range[j], steps) for j in joints],
                            [(g, h) for g in moved for h in mine],
                            (c, linkage(m, c), moved))
            store.update(s)
            neutral.update(n)
        # a moved frame member against the tail and the head over their joint ranges, too
        for bodies, jn in ((("tail",), ("tail_yaw", "tail_pitch")), (("head", "head_yaw"), ("neck_pitch", "neck_yaw"))):
            ids = {m.body(b).id for b in bodies}
            joints = [m.joint(j).id for j in jn]
            s, n, _ = sweep(m, d, joints, [np.linspace(*m.jnt_range[j], steps) for j in joints],
                            [(g, h) for g in moved for h in solid_geoms(m) if m.geom_bodyid[h] in ids])
            store.update(s)
            neutral.update(n)
        s, n, _ = sweep(m, d, [], [], chassis_pairs)
    else:
        joints = [m.joint(j).id for j in ("tail_yaw", "tail_pitch")]
        s, n, _ = sweep(m, d, joints, [np.linspace(*m.jnt_range[j], steps) for j in joints], chassis_pairs)
    store.update(s)
    neutral.update(n)
    return {"near_or_overlapping": [p for p in finish(store, m, neutral) if p["min_mm"] < 1000 * REPORT_M],
            "mounts_mm": mounts(m, d, list(edits))}


def run(model_path: Path = MODEL, steps: int = 9, trials: bool = True, pair_steps: int = PAIR_STEPS) -> dict:
    ZERO_RETRIES.clear()
    REMEASURED.clear()
    m = with_probe_rods(model_path)
    d = mujoco.MjData(m)
    solid = solid_geoms(m)
    leg_ids = set().union(*(leg_body_ids(m, c) for c in CORNERS))
    others = [g for g in solid if m.geom_bodyid[g] not in leg_ids]   # chassis, head, head_yaw, tail
    res = {"model": model_path.name, "mujoco": mujoco.__version__, "grid_steps": steps, "pair_grid_steps": pair_steps,
           "evidence": "desk: geometry of the simulation model's parts", "envelope_mismatches": check_envelopes(m),
           "legs": {}, "linkage": {}}
    for c in CORNERS:
        res["legs"][c], res["linkage"][c] = leg(m, d, c, others, steps)
    ch = [g for g in solid if m.geom_bodyid[g] == m.body("chassis").id]
    res["bay"] = bay(m, d)
    res["bay_room"] = bay_room(m, d)
    res["head"] = body_sweep(m, d, ("head", "head_yaw"), ("neck_pitch", "neck_yaw"), ch, steps)
    res["tail"] = body_sweep(m, d, ("tail",), ("tail_yaw", "tail_pitch"), ch, steps)
    res["legs_to_each_other"] = legs_to_each_other(m, d, pair_steps)
    for v in res["legs_to_each_other"].values():
        for p in v["pairs_nearer_than_10mm"]:
            if p["a"].endswith("_sole") and p["b"].endswith("_sole") and p["min_mm"] < 0:
                p["rocker_halves_overlap_cm3_at_this_pose"] = rocker_overlap(m, d, p["at_deg"])
    res["leg_limits_against_each_other"] = leg_limits(m, d, pair_steps)
    res["hind_roll_limit_for_rails"] = hind_roll_limit(m, d, steps)
    res["trials"] = {label: dict({"edits": edits}, **trial(model_path, edits, steps, kind))
                     for label, (kind, edits) in TRIALS.items()} if trials else {}
    res["zero_retries"] = [list(map(str, z)) for z in ZERO_RETRIES]
    res["zeros_remeasured"] = [list(map(str, z)) for z in REMEASURED]
    return res


def _mm(x):
    return ">50" if x is not None and x >= 1000 * CAP_M - 1e-6 else ("n/a" if x is None else f"{x:+.2f}")


def _fmt(p) -> str:
    at = f" at {p['at_deg']}" if p.get("at_deg") else ""
    return f"{_mm(p['min_mm'])} mm{at} (neutral {_mm(p['neutral_mm'])})  {p['a']} / {p['b']}  [{p['class']}]"


QUIET = ("joint", "cover")


def report(res: dict) -> None:
    print(f"Packaging check, {res['model']} (mujoco {res['mujoco']}; {res['grid_steps']} values per joint; DESK)")
    print("Drive housings against data/drives.json:", res["envelope_mismatches"] or "all match")
    print("1. Legs: pairs nearer than 10 mm, least distance and its pose (deg: roll, pitch, flexion), neutral "
          "distance; joint-by-design and own-cover pairs counted, not listed")
    for c, pairs in res["legs"].items():
        n = {k: sum(p["class"] == k for p in pairs) for k in QUIET}
        print(f"  {c} ({n['joint']} joint pairs, {n['cover']} own-cover pairs not listed)")
        for p in pairs:
            if p["class"] not in QUIET:
                print("    " + _fmt(p))
    print("2./3. Linkage")
    for c, lk in res["linkage"].items():
        print(f"  {c}: crank {lk['crank_radius_mm']:.2f} mm (ends differ {lk['crank_ends_differ_mm']:.3f} mm), rod pin "
              f"to pin {lk['rod_pin_to_pin_mm']:.2f} mm ({lk['rod_length_over_range_mm'][0]:.2f}-"
              f"{lk['rod_length_over_range_mm'][1]:.2f} over the grid), flexion {lk['flexion_range_deg'][0]:.0f} to "
              f"{lk['flexion_range_deg'][1]:.0f} deg, least transmission angle {lk['transmission_angle_min_deg']:.2f} "
              f"deg at flexion {lk['at_flexion_deg']} deg; axial gaps: " +
              ", ".join(f"{k} {v:.2f} mm" for k, v in lk["axial_gaps_mm"].items()))
        for p in lk["moving_rod"]:
            print("      moving rod " + _fmt(p))
    print("4. Chassis bay at neutral (frame-to-frame joints not listed); head and tail over their ranges")
    for p in res["bay"]:
        if p["class"] not in QUIET:
            print("    " + _fmt(p))
    for k in ("head", "tail"):
        for p in res[k]:
            if p["class"] not in QUIET:
                print(f"    {k}: " + _fmt(p))
    print("   room around the bay's tightest pairs: free travel, mm (part met first; '>30' = clear for 30 mm)")
    for label, row in res["bay_room"].items():
        print(f"    {label}: " + "; ".join(
            f"{k} " + (">30" if v["meets"] is None else f"{v['free_mm']:.2f} ({', '.join(v['meets'])})")
            for k, v in row.items()))
    print("   hind linkage against the rails, inward roll limit (roll range -18 to +24 deg):",
          res["hind_roll_limit_for_rails"])
    print("5. Trial fixes (the moved parts against everything and each other; pairs nearer than 10 mm; joints not "
          "listed; parts a trial leaves in place keep the gaps above)")
    for label, t in res["trials"].items():
        rows = [p for p in t["near_or_overlapping"] if p["class"] not in QUIET]
        print(f"  {label}: " + ("clear of every part by 10 mm or more" if not rows else ""))
        for p in rows:
            print("      " + _fmt(p))
        for n, row in t.get("mounts_mm", {}).items():
            print(f"      {n} meets (<= 0) or misses: " + ", ".join(f"{k} {_mm(v)}" for k, v in row.items()))
    print(f"6. Legs against each other ({res['pair_grid_steps']} values per joint over all eight joints of a pair, then a "
          "local search between grid points; least distance and its pose, deg):")
    for k, v in res["legs_to_each_other"].items():
        print(f"    {k}: " + "; ".join(f"{kind} {_mm(x['min_mm'])} mm" + (f" ({x['pair'][0]} / {x['pair'][1]} [{x['class']}] "
                                                                         f"at {x['at_deg']})" if x["pair"] else "")
                                       for kind, x in v["least"].items()))
        for kind, r in v["region_overlapping_on_the_grid"].items():
            print(f"      {kind} overlap at {100 * r['share_of_grid_poses']:.3g} % of the grid's poses; grid values taking "
                  f"part (each joint's bounds, not a region where the legs always meet): "
                  f"{r['joint_grid_values_taking_part_deg']}; inside these bounds "
                  f"{100 * r['share_of_the_poses_inside_these_bounds']:.3g} % of the poses overlap; the two legs' roll "
                  f"values differ by at most {r['largest_roll_difference_deg']} deg at an overlapping pose")
        for p in v["pairs_nearer_than_10mm"]:
            extra = (f"; the soles' lower halves (the rocker) overlap by {p['rocker_halves_overlap_cm3_at_this_pose']:.2f} "
                     "cm3 at this pose" if "rocker_halves_overlap_cm3_at_this_pose" in p else "")
            print(f"      {_mm(p['min_mm'])} mm (grid {_mm(p['grid_min_mm'])}; overlapping at "
                  f"{100 * p['grid_share_overlapping']:.3g} % of the pair's grid poses) {p['a']} / {p['b']} [{p['class']}] "
                  f"at {p['at_deg']}{extra}")
    print("   candidate joint limits, legs against each other (least distance, mm):")
    for label, rows in res["leg_limits_against_each_other"].items():
        print(f"    {label}:")
        for r in rows:
            print(f"      {r['ranges_deg']}: " + "; ".join(
                f"{kind} {_mm(x['min_mm'])}" + (f" ({x['pair'][0]} / {x['pair'][1]})" if x["pair"] else "")
                for kind, x in r["least"].items()))
    print(f"mj_geomDistance exact zeros re-measured at +-0.06 deg (failed solves): {len(res['zeros_remeasured'])}"
          + "".join(f"\n    {z}" for z in res["zeros_remeasured"]))
    print("exact zeros that no retry changed (a failed solve or a true touch, not a measurement):",
          res["zero_retries"] or "none")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--no-trials", action="store_true")
    ap.add_argument("--pair-steps", type=int, default=PAIR_STEPS, help="grid values per joint, legs against each other")
    ap.add_argument("--json", default=None, help="write the full result here")
    a = ap.parse_args(argv)
    res = run(Path(a.model), a.steps, not a.no_trials, a.pair_steps)
    report(res)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
