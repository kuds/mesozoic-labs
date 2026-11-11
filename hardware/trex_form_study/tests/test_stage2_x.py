"""Tests for Stage 2's expressive head, the ``-X`` bodies (``STAGE2_PLAN.md`` S2-2, DECISION_LOG P29): a neck yaw at
the neck base and a hinged jaw on the two picks, in simulation only.

* Every model the Stage 1 registry used before (the probe's three bodies and arm K's two) regenerates from its record
  with the record's ``xml_sha256`` and the plan's prefix: the options default to today's bodies.
* With its two new joints locked, an ``-X`` body is today's body plus the added parts (two STS3215 servos with their
  horns and brackets): the ledger, every body's inertial and the whole model's mass, centre of mass and inertia.
* The joint ranges (neck yaw +-0.6 rad, cut to the clearance screen's usable range; jaw 0 to 0.6 rad) and the motor
  map (each new joint driven by an STS3215: the selection, the sizing's motor map, the plant's drives, the MJCF's
  actuators).
* The refine cases are their picks' records with the head added, and the registry and the carpet swing follow.
* Stage 2's run records (``stage2/x/<body>/``), where present: the plant, the references, the transitions and G1-G3.

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_x.py -q
About a minute (each body is built a few times, 4 s each).  The run-record checks are skipped without the records.
"""

from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
import mujoco  # noqa: E402
from designlab import build, catalog, clearance, mjcf, refine, round2, sizing  # noqa: E402
from designlab.geometry import combine  # noqa: E402
from designlab.pipeline import build_candidate  # noqa: E402
from designlab.stage1 import (  # noqa: E402
    BODIES,
    CANDIDATE_IDS,
    HEAD_X_BODIES,
    PROBE_BODIES,
    SHA256_PREFIX,
    STUDY,
    candidate_id,
    case_dir,
    drive,
    plant,
    refs,
)

#: The -X bodies: body -> (its refine case, its pick, the pick's case).
HEAD_X = {"60-N-T-10-X": ("head_x_60", "60-N-T-10", "narrow_carpet_60"),
          "75-M-H-10-X": ("head_x_75", "75-M-H-10", "feet_75")}
#: Every body the registry held before Stage 2, with its record's case.
EARLIER = {"60-N-T-10": "narrow_carpet_60", "60-N-H-10": "narrow_carpet_60", "75-M-H-10": "feet_75",
           "60-N-T-10-K": "narrow_knee_60", "60-N-H-10-K": "narrow_knee_60"}
STS3215 = "feetech_sts3215_c018"
RUN = ROOT / "stage2" / "x"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build(rec: dict, tmp: Path) -> dict:
    """``pipeline.build_candidate`` from a sized record (the plant's and the references' rebuild), into ``tmp``."""
    res = build_candidate(round2.candidate(rec["id"], rec.get("overrides", ())), tmp, selection=dict(rec["selection"]),
                          B=rec["B"], write=False)
    res["xml"] = tmp / f"{rec['id']}.xml"
    return res


def _pick_record(body: str) -> dict:
    case, cid, base = HEAD_X[body]
    return json.loads((ROOT / "round2" / "refine" / base / "candidates" / f"{cid}.json").read_text())


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    """Each -X body and its pick, built from their records once for the module: body -> (pick, -X)."""
    out = {}
    for body in HEAD_X:
        tmp = tmp_path_factory.mktemp(body)
        (tmp / "pick").mkdir()
        (tmp / "x").mkdir()
        out[body] = (_build(_pick_record(body), tmp / "pick"), _build(plant.record(body), tmp / "x"))
    return out


# ------------------------------------------------------------------------------------------------
# Existing bodies are unchanged
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body", sorted(EARLIER))
def test_an_earlier_registry_model_regenerates_with_its_recorded_sha256(body, tmp_path):
    """The code with the expressive head's options rebuilds every model Stage 1 used from its record, byte for byte:
    the record's ``xml_sha256`` and the plan's prefix of it (``SHA256_PREFIX``).  The head's options are off."""
    rec = plant.record(body)
    assert BODIES[body] == f"round2/refine/{EARLIER[body]}" and rec["refine_case"] == EARLIER[body]
    res = _build(rec, tmp_path)
    assert _sha(res["xml"]) == rec["xml_sha256"] and rec["xml_sha256"].startswith(SHA256_PREFIX[body])
    d = res["design"]
    assert d.head_x_groups == () and not d.neck_yaw and not d.jaw and "jaw" not in d.bodies
    assert all(not b.pre_joints for b in d.bodies.values()) and "expressive_head" not in d.geometry_checks
    assert set(rec["selection"]).isdisjoint(sizing.HEAD_X_GROUPS) and set(rec["margins"]) == set(sizing.GROUPS)
    assert res["model"].names[:2] == ["neck_pitch", "tail_yaw"] and len(res["model"].names) == 14
    assert not any(mjcf._head_x(r) for r in res["rows"])


def test_the_options_default_to_todays_bodies():
    """No override, no expressive head: the sizing's groups and motor map, the roles and the clearance sweeps of a
    14-hinge model are round two's."""
    names = ["neck_pitch", "tail_yaw"] + [f"{s}_{j}" for s in ("left", "right") for j in
                                          ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    assert sizing.groups(names) == sizing.GROUPS
    tau, w = np.arange(28.0).reshape(2, 14), -np.arange(28.0).reshape(2, 14)
    assert list(sizing._motor_space(names, tau, w, "tandem")) == list(sizing.GROUPS)
    assert sizing.groups(names + ["neck_yaw", "jaw"]) == sizing.GROUPS + ("neck_yaw", "jaw")
    assert {j: mjcf.joint_role(j) for j in ("neck_pitch", "tail_yaw", "neck_yaw", "jaw", "left_knee")} == {
        "neck_pitch": "neck", "tail_yaw": "tail", "neck_yaw": "neck_yaw", "jaw": "jaw", "left_knee": "knee"}
    assert clearance.REST_ON_STOP == {"jaw": "min"}
    margins = {"left_knee": dict(margin_min=0.2, margin_max=0.3), "jaw": dict(margin_min=0.0, margin_max=0.6,
                                                                               rests_on_stop="min")}
    assert clearance.gated_margins(margins) == [("left_knee", "min", 0.2), ("left_knee", "max", 0.3),
                                                ("jaw", "max", 0.6)]


# ------------------------------------------------------------------------------------------------
# The -X models and their records
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body", sorted(HEAD_X))
def test_a_head_x_model_regenerates_with_its_recorded_sha256(body, built):
    rec = plant.record(body)
    _, x = built[body]
    assert _sha(x["xml"]) == rec["xml_sha256"] and rec["xml_sha256"].startswith(SHA256_PREFIX[body])
    assert len(SHA256_PREFIX[body]) == 8
    model = case_dir(body) / "models" / f"{candidate_id(body)}.xml"
    if model.exists():  # git-ignored: the build's own file, or one copied in
        assert _sha(model) == rec["xml_sha256"]


@pytest.mark.parametrize("body", sorted(HEAD_X))
def test_a_head_x_case_is_its_pick_with_the_head_added(body):
    """``refine.CASES``: the pick's case's settings (sizing walks, headroom, turn gap), its id and overrides with the
    head's; the record is the pick's record (unchanged since: the same sha256) with the head's overrides and an
    STS3215 for each new joint, built once and not re-sized."""
    case, cid, base = HEAD_X[body]
    c, b = refine.CASES[case], refine.CASES[base]
    assert c["head_x"] and c["base_case"] == base and refine.HEAD_X_BASE[case] == (base, cid) and c["ids"] == (cid,)
    assert c["overrides"] == tuple(b["overrides"]) + refine.HEAD_X_OVERRIDES
    assert {k: v for k, v in c.items() if k not in ("ids", "overrides", "label", "base_case", "head_x")} == {
        k: v for k, v in b.items() if k not in ("ids", "overrides", "label")}
    assert refine.HEAD_X_OVERRIDES == (("neck_yaw_joint", 1), ("jaw_joint", 1)) and refine.HEAD_X_PART == STS3215
    rec, pick = plant.record(body), _pick_record(body)
    hx = rec["head_x"]
    base_path = ROOT / hx["base_record"]
    assert hx["base_case"] == base and base_path == ROOT / "round2" / "refine" / base / "candidates" / f"{cid}.json"
    assert hx["base_record_sha256"] == _sha(base_path) and hx["base_xml_sha256"] == pick["xml_sha256"]
    assert rec["ok"] and rec["id"] == cid and rec["refine_case"] == case and rec["B"] == pick["B"]
    assert rec["overrides"] == pick["overrides"] + [list(o) for o in refine.HEAD_X_OVERRIDES]
    assert rec["selection"] == dict(pick["selection"], neck_yaw=STS3215, jaw=STS3215)
    assert rec["packaging_search"] == [] and [h["iteration"] for h in rec["sizing_history"]] == ["head_x"]
    assert set(rec["margins"]) == set(sizing.GROUPS + sizing.HEAD_X_GROUPS) and set(rec["gaits"]) == set(pick["gaits"])
    assert hx["added_mass_kg"] == pytest.approx(0.150, abs=1e-12)
    assert rec["metrics"]["mass"] == pytest.approx(pick["metrics"]["mass"] + 0.150, abs=1e-6)


# ------------------------------------------------------------------------------------------------
# Locked, an -X body is today's body plus the added parts
# ------------------------------------------------------------------------------------------------
def _key(r: dict) -> tuple:
    return (r["name"], r["category"], r["mass_kg"], tuple(r["com_body"]), tuple(map(tuple, r["inertia_body"])))


def _added(rows: list) -> list:
    return [r for r in rows if mjcf._head_x(r)]


@pytest.mark.parametrize("body", sorted(HEAD_X))
def test_the_head_x_ledger_is_todays_plus_the_added_parts(body, built):
    """The ledger: today's rows exactly, in today's order (the jaw cover's rows on the jaw body, which keeps the head's
    frame), plus the four added parts (two 0.055 kg STS3215 servos, two 0.020 kg horns and brackets); the integration
    reserve is today's to the last bit."""
    pick, x = built[body]
    r0, rx = pick["rows"], x["rows"]
    added = _added(rx)
    assert [(r["name"], r["body"], r["mass_kg"]) for r in added] == [
        ("neck_yaw_motor", "trunk", 0.055), ("neck_yaw_bracket", "trunk", 0.020),
        ("jaw_motor", "head", 0.055), ("jaw_bracket", "head", 0.020)]
    assert {r["meta"][build.HEAD_X_META] for r in added} == {"neck_yaw", "jaw"}
    kept = [r for r in rx if not mjcf._head_x(r)]
    assert [_key(r) for r in kept] == [_key(r) for r in r0]
    moved = [(a["name"], a["body"], b["body"]) for a, b in zip(r0, kept, strict=True) if a["body"] != b["body"]]
    assert moved == [(n, "head", "jaw") for n in ("cover_jaw", "cover_jaw_support", "cover_jaw_fasteners")]
    res0 = next(r for r in r0 if r["name"] == "integration_reserve")
    resx = next(r for r in rx if r["name"] == "integration_reserve")
    assert _key(res0) == _key(resx)


@pytest.mark.parametrize("body", sorted(HEAD_X))
def test_every_body_inertial_is_todays_or_todays_plus_the_added_parts(body, built):
    """Per body: every body but the trunk, the head and the new jaw has today's inertial exactly; the trunk is today's
    with the neck-yaw servo and bracket, and the head with the jaw is today's head with the jaw servo and bracket."""
    pick, x = built[body]
    i0 = mjcf.body_inertials(pick["design"], [dict(r) for r in pick["rows"] if r["name"] != "integration_reserve"])
    ix = mjcf.body_inertials(x["design"], [dict(r) for r in x["rows"] if r["name"] != "integration_reserve"])
    assert set(ix) == set(i0) | {"jaw"}
    for name in set(i0) - {"trunk", "head"}:
        m0, c0, I0 = i0[name]
        mx, cx, Ix = ix[name]
        assert mx == m0 and np.array_equal(c0, cx) and np.array_equal(I0, Ix), name

    def parts(rows, body_):
        return [(r["mass_kg"], np.array(r["com_body"]), np.array(r["inertia_body"])) for r in _added(rows)
                if r["body"] == body_]

    for got, want in ((ix["trunk"], combine([i0["trunk"]] + parts(x["rows"], "trunk"))),
                      (combine([ix["head"], ix["jaw"]]), combine([i0["head"]] + parts(x["rows"], "head")))):
        assert got[0] == pytest.approx(want[0], rel=1e-13)
        assert np.allclose(got[1], want[1], rtol=0, atol=1e-14)
        assert np.allclose(got[2], want[2], rtol=0, atol=1e-15)


def _composite(m, d) -> tuple:
    """Mass, centre of mass and inertia about it (world axes) of every body of a compiled model at ``d``'s pose."""
    mass = m.body_mass[1:]
    com = (mass[:, None] * d.xipos[1:]).sum(0) / mass.sum()
    inertia = np.zeros((3, 3))
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3)
        r = d.xipos[b] - com
        inertia += R @ np.diag(m.body_inertia[b]) @ R.T + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
    return float(mass.sum()), com, inertia


@pytest.mark.parametrize("body", sorted(HEAD_X))
def test_the_locked_head_x_model_has_todays_mass_properties_plus_the_added_parts(body, built):
    """The compiled models, both at today's home pose with the neck yaw and the jaw at 0 (locked): the -X model's
    mass, centre of mass and inertia are today's plus the four added parts (the MJCF writes six significant digits,
    hence the tolerances).  The added parts move the centre of mass 5-7 mm forward."""
    pick, x = built[body]
    m0 = mujoco.MjModel.from_xml_path(str(pick["xml"]))
    mx = mujoco.MjModel.from_xml_path(str(x["xml"]))
    d0, dx = mujoco.MjData(m0), mujoco.MjData(mx)
    d0.qpos[:] = m0.key_qpos[0]
    for j in range(m0.njnt):
        name = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)
        jx = mujoco.mj_name2id(mx, mujoco.mjtObj.mjOBJ_JOINT, name)
        n = 7 if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        dx.qpos[mx.jnt_qposadr[jx] : mx.jnt_qposadr[jx] + n] = d0.qpos[m0.jnt_qposadr[j] : m0.jnt_qposadr[j] + n]
    for name in ("neck_yaw", "jaw"):
        assert dx.qpos[mx.jnt_qposadr[mujoco.mj_name2id(mx, mujoco.mjtObj.mjOBJ_JOINT, name)]] == 0.0
    mujoco.mj_forward(m0, d0)
    mujoco.mj_forward(mx, dx)
    parts = []
    for r in _added(x["rows"]):
        b = mujoco.mj_name2id(mx, mujoco.mjtObj.mjOBJ_BODY, r["body"])
        R = dx.xmat[b].reshape(3, 3)
        parts.append((r["mass_kg"], dx.xpos[b] + R @ np.array(r["com_body"]), R @ np.array(r["inertia_body"]) @ R.T))
    mt, ct, It = _composite(m0, d0)
    want = combine([(mt, ct, It)] + parts)
    got = _composite(mx, dx)
    assert got[0] == pytest.approx(want[0], rel=2e-6) and want[0] - mt == pytest.approx(0.150, abs=1e-12)
    assert np.allclose(got[1], want[1], rtol=0, atol=2e-6)
    assert np.allclose(got[2], want[2], rtol=0, atol=2e-6 * np.abs(want[2]).max())
    # forward, about as [look] section 5.2 estimates (5.7 / 5.1 mm); at its own home pose the stance puts the feet
    # (and the legs' mass) under the new centre of mass, which moves it a little further (the record's com_ahead_of_hip)
    assert 0.004 < got[1][0] - ct[0] < 0.007


def _bodies(xml: Path) -> dict:
    return {b.get("name"): b for b in ET.parse(xml).getroot().iter("body")}


def _canon(el) -> tuple:
    """An element as written, without the indentation: its tag, attributes and children."""
    return el.tag, tuple(sorted(el.attrib.items())), tuple(_canon(c) for c in el)


@pytest.mark.parametrize("body", sorted(HEAD_X))
def test_the_head_x_mjcf_is_todays_with_the_head_added(body, built):
    """The MJCF: every body but the trunk and the head is written exactly as today's (inertial, joint, geoms, sites;
    the legs and the tail); the head holds today's geoms but the jaw cover's, plus the jaw servo, the neck yaw
    before the neck pitch and a ``jaw`` body; the trunk holds today's geoms plus the neck-yaw servo; the jaw body has
    today's jaw cover and its proxies, in the head's frame."""
    pick, x = built[body]
    b0, bx = _bodies(pick["xml"]), _bodies(x["xml"])
    assert set(bx) == set(b0) | {"jaw"}

    for name in set(b0) - {"trunk", "head"}:
        assert _canon(b0[name]) == _canon(bx[name]), name
    geoms = {k: {g.get("name"): _canon(g) for g in v.findall("geom")} for k, v in (("t0", b0["trunk"]),
             ("tx", bx["trunk"]), ("h0", b0["head"]), ("hx", bx["head"]), ("jx", bx["jaw"]))}
    assert set(geoms["tx"]) == set(geoms["t0"]) | {"hw_neck_yaw_motor"}
    assert all(geoms["tx"][g] == v for g, v in geoms["t0"].items())
    jaw = {g for g in geoms["h0"] if g == "cover_jaw" or g.startswith("cl_jaw_")}
    assert jaw and set(geoms["jx"]) == jaw and all(geoms["jx"][g] == geoms["h0"][g] for g in jaw)
    assert set(geoms["hx"]) == (set(geoms["h0"]) - jaw) | {"hw_jaw_motor"}
    assert all(geoms["hx"][g] == geoms["h0"][g] for g in set(geoms["h0"]) - jaw)
    assert [j.get("name") for j in bx["head"].findall("joint")] == ["neck_yaw", "neck_pitch"]
    assert _canon(bx["head"].findall("joint")[1]) == _canon(b0["head"].find("joint"))
    for name in ("trunk", "head"):  # the trunk's and the head's sites are today's
        assert [_canon(e) for e in b0[name].findall("site")] == [_canon(e) for e in bx[name].findall("site")]
    assert bx["jaw"].get("pos") == "0 0 0" and bx["jaw"].get("quat") == "1 0 0 0"
    acts = [a.get("joint") for a in ET.parse(x["xml"]).getroot().find("actuator")]
    assert acts == x["model"].names and acts[:4] == ["neck_yaw", "neck_pitch", "jaw", "tail_yaw"]
    r0, rx = ET.parse(pick["xml"]).getroot(), ET.parse(x["xml"]).getroot()
    for tag in ("compiler", "option", "visual", "default", "asset", "sensor"):  # the meshes too, in today's order
        assert _canon(r0.find(tag)) == _canon(rx.find(tag)), tag


# ------------------------------------------------------------------------------------------------
# Joint ranges and the motor map
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body", sorted(HEAD_X))
def test_the_new_joints_ranges_axes_and_anchors(body, built):
    """Neck yaw: +-0.6 rad about the world vertical at home, through the neck base (the neck pitch's anchor), ahead of
    the neck pitch in the chain (yaw, then pitch); jaw: 0 to 0.6 rad about a lateral axis at the hinge, on the jaw
    cover's upper edge 0.10 / 0.12 m ahead of the neck-pitch axis.  The plant cuts both to the clearance screen's
    usable range, read as the screens read it; the jaw rests closed on its stop, which the screen's range-margin gate
    leaves out."""
    _, x = built[body]
    m = mujoco.MjModel.from_xml_path(str(x["xml"]))
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    mujoco.mj_forward(m, d)
    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("neck_yaw", "neck_pitch", "jaw")}
    assert np.allclose(m.jnt_range[jid["neck_yaw"]], build.NECK_YAW_RANGE) and build.NECK_YAW_RANGE == (-0.6, 0.6)
    assert np.allclose(m.jnt_range[jid["jaw"]], build.JAW_RANGE) and build.JAW_RANGE == (0.0, 0.6)
    assert np.allclose(d.xaxis[jid["neck_yaw"]], [0, 0, 1], atol=1e-12)
    assert np.allclose(d.xanchor[jid["neck_yaw"]], d.xanchor[jid["neck_pitch"]], atol=1e-12)
    assert jid["neck_yaw"] < jid["neck_pitch"] < jid["jaw"]
    assert np.allclose(d.xaxis[jid["jaw"]], [0, 1, 0], atol=1e-12)
    design = x["design"]
    hinge = design.jaw_station["hinge"]
    assert np.allclose(d.xanchor[jid["jaw"]], design.bodies["head"].pos_world + hinge, atol=1e-5)
    assert hinge[0] == pytest.approx(build.JAW_HINGE_AHEAD_M[int(body[:2])], abs=0.001)
    rec = plant.record(body)
    cl = json.loads((case_dir(body) / "review" / "screening" / f"{candidate_id(body)}_clearance.json").read_text())
    usable = plant.usable_ranges(body)
    for name, rng in (("neck_yaw", build.NECK_YAW_RANGE), ("jaw", build.JAW_RANGE)):
        lo, hi = usable[name]
        assert rng[0] <= lo < hi <= rng[1] and cl["summary"]["joint_range_margins"][name]["usable"] == [lo, hi]
    assert cl["summary"]["joint_range_margins"]["jaw"]["rests_on_stop"] == "min"
    assert "rests_on_stop" not in cl["summary"]["joint_range_margins"]["neck_yaw"]
    assert rec["geometry_checks"]["expressive_head"]["neck_yaw"]["inside_trunk_profile"]


@pytest.mark.parametrize("body", sorted(HEAD_X))
def test_each_new_joint_is_driven_by_an_sts3215(body, built):
    """The motor map: the records' selection names an STS3215 for the neck yaw and the jaw; the sizing's motor map,
    the plant's drives (with the STS3215's planning limits: sustained 0.491, peak 0.862 N m, 2.83 rad/s), the
    sizing values the plant is checked against and the MJCF's actuators (force range +-peak, the STS3215's armature)
    all map each new joint to its own drive, as the neck and the tail are."""
    _, x = built[body]
    rec = plant.record(body)
    a = catalog.actuator(STS3215)
    assert (a.sustained, a.peak, a.speed_at(None)) == pytest.approx((0.491, 0.862, 2.827), abs=1e-3)
    assert (a.mass, a.price) == (0.055, 22)
    assert rec["selection"]["neck_yaw"] == rec["selection"]["jaw"] == STS3215
    assert {g: rec["margins"][g]["actuator"] for g in sizing.HEAD_X_GROUPS} == {"neck_yaw": STS3215, "jaw": STS3215}
    M = x["model"]
    names = M.names
    rng = np.random.default_rng(0)
    tau, w = rng.normal(size=(3, len(names))), rng.normal(size=(3, len(names)))
    ms = sizing._motor_space(names, tau, w, "tandem")
    for g in sizing.HEAD_X_GROUPS:
        j = names.index(sizing.HEAD_X_JOINT[g])
        assert np.array_equal(ms[g][0], tau[:, j]) and np.array_equal(ms[g][1], w[:, j])
    drives = drive.drives(rec["selection"], names, layout=rec["candidate"]["leg_layout"])
    by = {dr.coordinate: dr for dr in drives}
    for g in sizing.HEAD_X_GROUPS:
        dr = by[g]
        assert (dr.group, dr.part, dr.actuator, dr.bus_v) == (g, STS3215, f"{g}_servo", None)
        assert (dr.sustained, dr.peak, dr.rated_speed) == (a.sustained, a.peak, a.speed_at(None))
        assert dr.armature == a.armature
    f, sp = drive.sizing_motor_values(names, tau[0], w[0], rec["candidate"]["leg_layout"])
    for g in sizing.HEAD_X_GROUPS:
        j = names.index(g)
        assert (f[j], sp[j]) == (tau[0, j], w[0, j])
    m = M.m
    for g in sizing.HEAD_X_GROUPS:
        act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{g}_servo")
        jn = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, g)
        assert m.actuator_trnid[act, 0] == jn and np.allclose(m.actuator_forcerange[act], [-a.peak, a.peak], rtol=1e-5)
        assert m.dof_armature[m.jnt_dofadr[jn]] == pytest.approx(a.armature, rel=1e-4)
    items = {it.name: it for b in x["design"].bodies.values() for it in b.items}
    assert items["neck_yaw_motor"].body == "trunk" and items["jaw_motor"].body == "head"
    assert np.allclose(x["design"].sk.R_p @ items["neck_yaw_motor"].rot[:, 2], [0, 0, 1])  # on the yaw axis
    assert {items[n].meta["actuator"] for n in ("neck_yaw_motor", "jaw_motor")} == {STS3215}


# ------------------------------------------------------------------------------------------------
# The registry and the references
# ------------------------------------------------------------------------------------------------
def test_the_head_x_bodies_are_registered_beside_the_earlier_ones():
    assert HEAD_X_BODIES == tuple(HEAD_X) and PROBE_BODIES == ("60-N-T-10", "60-N-H-10", "75-M-H-10")
    assert set(BODIES) == set(EARLIER) | set(HEAD_X) and set(SHA256_PREFIX) == set(BODIES)
    for body, (case, cid, _) in HEAD_X.items():
        assert BODIES[body] == f"round2/refine/{case}" and CANDIDATE_IDS[body] == cid == candidate_id(body)
        assert case_dir(body) == STUDY / "round2" / "refine" / case
    assert {k: v for k, v in CANDIDATE_IDS.items() if k not in HEAD_X} == {"60-N-T-10-K": "60-N-T-10",
                                                                            "60-N-H-10-K": "60-N-H-10"}


@pytest.mark.parametrize("body", sorted(HEAD_X))
def test_a_head_x_carpet_swing_is_its_picks(body):
    """``refs.CARPET_LIFT``: an -X body takes its pick's lift, which the carpet-lift pass on its own case (the pick's
    lifts and walks) gives by the pick's rule: the lowest lift that meets the target on every pile with the drives
    within their limits and the walking clearance gates kept."""
    case, cid, _ = HEAD_X[body]
    lift, path = refs.CARPET_LIFT[body], refs.CARPET_LIFT_RECORD[body]
    assert lift == refs.CARPET_LIFT[cid] and case in path
    if not (STUDY / path).exists():
        pytest.skip(f"{path}: the carpet-lift pass has not run (stage2/x/run_x.sh build)")
    pick_doc = json.loads((STUDY / refs.CARPET_LIFT_RECORD[cid]).read_text())
    row = next(r for r in json.loads((STUDY / path).read_text()) if r["id"] == cid)
    pick = next(r for r in pick_doc if r["id"] == cid)
    # (the 75 cm pick's pass predates the "lifted" field: it lifted both walks, as the pass did then)
    assert row["source"] == f"round2/refine/{case}" and row["lifted"] == pick.get("lifted", "nominal and brisk walks")
    assert set(row["lifts"]) == set(pick["lifts"])
    walk = "nominal" if row["lifted"] == "nominal walk only" else None
    ok = {float(k): v["drives_within_limits"] and v["walking_gates"] and all(
        all(t["meets_carpet_target"].values()) for n, t in v["toe"].items() if walk is None or n == walk)
        for k, v in row["lifts"].items()}
    assert lift == min(k for k, v in ok.items() if v)
    assert refs.reference_names(body) == refs.reference_names(cid)


# ------------------------------------------------------------------------------------------------
# Stage 2's run records (stage2/x/<body>), where present
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body", sorted(HEAD_X))
def test_the_head_x_prepare_records(body):
    """plant.json, refs.json and transitions.json of an -X body: its own model (its record's), the decided plant
    (decisions 11-12) with 16 hinges, the new joints at the clearance screen's ranges and on STS3215 drives, every
    reference matching the records it names, every transition there."""
    from designlab.stage1 import transitions

    if not (RUN / body / "plant.json").exists():
        pytest.skip("stage2/x/run_x.sh run has not made this body's records")
    pj = json.loads((RUN / body / "plant.json").read_text())
    rec = plant.record(body)
    assert pj["body"] == body and pj["model"]["matches_record"] and pj["model"]["matches_plan"]
    assert pj["model"]["record_sha256"] == rec["xml_sha256"] and pj["model"]["plan_prefix"] == SHA256_PREFIX[body]
    assert (pj["options"]["ankle_gains"], pj["options"]["ankle_armature"]) == ("joint", "linkage")
    assert pj["mass_kg"] == pytest.approx(rec["metrics"]["mass"])
    ranges = pj["joint_ranges"]
    assert len(ranges["plant"]) == 16 and ranges["plant"]["neck_yaw"] == list(plant.usable_ranges(body)["neck_yaw"])
    assert ranges["mjcf"]["neck_yaw"] == [-0.6, 0.6] and ranges["mjcf"]["jaw"] == [0.0, 0.6]
    parts = {dr["actuator"]: dr["part"] for dr in pj["drives"]}
    assert parts["neck_yaw_servo"] == parts["jaw_servo"] == STS3215 and len(parts) == 16
    if (RUN / body / "refs.json").exists():
        rj = json.loads((RUN / body / "refs.json").read_text())
        assert rj["body"] == body and rj["matches_records"] and rj["model"]["record_sha256"] == rec["xml_sha256"]
        assert set(rj["references"]) == set(refs.reference_names(body)) and len(rj["joints"]) == 16
        for e in rj["references"].values():
            assert e["record_checks"] and all(c["match"] for c in e["record_checks"])
    if (RUN / body / "transitions.json").exists():
        tj = json.loads((RUN / body / "transitions.json").read_text())
        assert tj["body"] == body and set(tj["transitions"]) == set(transitions.TRANSITIONS)


@pytest.mark.parametrize("body", sorted(HEAD_X))
def test_the_head_x_gate_records(body):
    """gates.json of an -X body: G1-G3 as Stage 1 ran them (the decided plant and G2 rule, decisions 11-13), each
    verdict as its rule reads it; G4 (the tracker's gate) does not apply and is not run."""
    path = RUN / body / "gates.json"
    if not path.exists():
        pytest.skip("stage2/x/run_x.sh run has not made this body's gates")
    doc = json.loads(path.read_text())
    assert doc["body"] == body and "G4" not in doc
    if "G1" in doc:
        g1 = doc["G1"]
        assert all(h["matches_record"] and h["matches_plan"] for h in g1["hashes"].values())
        assert g1["pass"] == bool(all(s["stands"] for s in g1["stand"].values()) and g1["tendon_map"]["matches"]
                                  and g1["repeatability"]["ok"] and g1["step_rate"]["ok"])
    if "G2" in doc:
        g2 = doc["G2"]
        assert g2["rule"] == "panel" and g2["decided"]["follows_all"] and g2["design_study_strides_screened_model"] is None
        assert g2["pass"] == (g2["all_match"] and g2["panel_pass"])
    if "G3" in doc:
        assert doc["G3"]["pass_"] == (not doc["G3"]["failed"] and not doc["G3"]["missing"])
    for g in ("G1", "G2", "G3"):
        if g in doc:
            assert doc[g]["provenance"]["stage1_sources"]["unchanged_during_run"]
