"""Tests for Stage 2's Jetson body, ``75-M-H-10-J`` (``JETSON_PLEM_REVIEW.md`` section 5 task 6, DECISION_LOG P30): the
75 cm pick with the Jetson Orin Nano Super dev kit as its compute part and a 6S-to-12 V regulator for the dev kit's
carrier, in simulation only, built once and not re-sized (refine case ``jetson_75``; ``stage2/j/JETSON_BODY.md``).

* Every model the Stage 1 registry held before (the probe's three bodies, arm K's two, the two expressive heads) and
  round two's own records of the two picks regenerate with their recorded ``xml_sha256``: the regulator's option
  defaults to today's bodies, and at its default the keys it reads change nothing.
* The Jetson model regenerates to its recorded ``xml_sha256``; with the compute swap undone it is the pick's.
* Its ledger is the pick's but for the compute board, its tray's envelope, the regulator and the integration reserve;
  its mass and centre of mass are the pick's plus those rows' sums, in the ledger, in the compiled models and in the
  record.
* The regulator is the catalog part the record's rule picks, and it sits on the power box clear of everything.
* The case is its pick's with the compute changed, the Teensy control stack kept and no Jetson-direct part placed.
* The power rows: ``power.COMPUTE_BUDGET`` and every call without a mode are as they were (the pick's recorded power
  and runtime recompute), and the two mode rows give the record's ``jetson.power_modes``.
* The registry, the carpet swing, the screens' records, Stage 2's run records (``stage2/j/<body>/``) and the list of
  the git-ignored inputs (``j_inputs.json``), where present; the report's tables regenerate from the records.

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_j.py
About three minutes (each body is built a few times, 4 s each, and two bodies' sizing walks are run once, about 30 s
each).  The run-record checks are skipped without the records.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
import mujoco  # noqa: E402
from designlab import build, catalog, compute_options, gait, mjcf, power, refine, round2, sizing  # noqa: E402
from designlab.pipeline import build_candidate  # noqa: E402
from designlab.stage1 import (  # noqa: E402
    BODIES,
    CANDIDATE_IDS,
    JETSON_BODIES,
    SHA256_PREFIX,
    STUDY,
    candidate_id,
    case_dir,
    plant,
    refs,
)

BODY, CASE, CID, BASE = "75-M-H-10-J", "jetson_75", "75-M-H-10", "feet_75"
DEVKIT, PI, REGULATOR, SMALLER = ("nvidia_orin_nano_super_devkit", "rpi5_8gb_ai_hat_plus_13tops", "pololu_d42v110f12",
                                  "pololu_d36v50f12")
#: Every body the registry held before the Jetson body, with its record's case.
EARLIER = {"60-N-T-10": "narrow_carpet_60", "60-N-H-10": "narrow_carpet_60", "75-M-H-10": "feet_75",
           "60-N-T-10-K": "narrow_knee_60", "60-N-H-10-K": "narrow_knee_60", "60-N-T-10-X": "head_x_60",
           "75-M-H-10-X": "head_x_75"}
RUN = ROOT / "stage2" / "j"
#: The ledger rows that differ between the Jetson body and its pick, and the one it adds.
CHANGED, ADDED = ("compute", "compute_tray", "integration_reserve"), "compute_regulator"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build(rec: dict, tmp: Path, selection: dict | None = None, overrides=None) -> dict:
    """``pipeline.build_candidate`` from a sized record (the plant's and the references' rebuild), into ``tmp``."""
    res = build_candidate(round2.candidate(rec["id"], rec.get("overrides", ()) if overrides is None else overrides), tmp,
                          selection=dict(rec["selection"] if selection is None else selection), B=rec["B"], write=False)
    res["xml"] = tmp / f"{rec['id']}.xml"
    return res


def _pick_record() -> dict:
    return json.loads((ROOT / "round2" / "refine" / BASE / "candidates" / f"{CID}.json").read_text())


def _helper():
    spec = importlib.util.spec_from_file_location("j_inputs", RUN / "j_inputs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tables():
    spec = importlib.util.spec_from_file_location("j_tables", RUN / "j_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple:
    """The pick and the Jetson body, built from their records once for the module."""
    tmp = tmp_path_factory.mktemp(BODY)
    (tmp / "pick").mkdir()
    (tmp / "j").mkdir()
    return _build(_pick_record(), tmp / "pick"), _build(plant.record(BODY), tmp / "j")


def _regulator_rows(rows: list) -> list:
    return [r for r in rows if build.COMPUTE_REGULATOR_META in (r.get("meta") or {})]


# ------------------------------------------------------------------------------------------------
# Existing bodies are unchanged
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body", sorted(EARLIER))
def test_an_earlier_registry_model_regenerates_with_its_recorded_sha256(body, tmp_path):
    """The code with the regulator's option rebuilds every model the registry held before from its record, byte for
    byte: the record's ``xml_sha256`` and the plan's prefix of it.  The option is off and nothing of it is there."""
    rec = plant.record(body)
    assert BODIES[body] == f"round2/refine/{EARLIER[body]}" and rec["refine_case"] == EARLIER[body]
    res = _build(rec, tmp_path)
    assert _sha(res["xml"]) == rec["xml_sha256"] and rec["xml_sha256"].startswith(SHA256_PREFIX[body])
    d = res["design"]
    assert not d.compute_regulator and "compute_regulator" not in d.geometry_checks
    assert "compute_regulator" not in rec["selection"] and not _regulator_rows(res["rows"])
    assert not any(r["name"] == ADDED for r in res["rows"])
    assert rec["selection"]["compute"] == PI and "jetson" not in rec


@pytest.mark.parametrize("cid", ("60-N-T-10", "75-M-H-10"))
def test_a_round_two_record_regenerates_with_its_recorded_sha256(cid, tmp_path):
    """Round two's own sized records of the two picks' candidates (``round2/candidates``), as the screens rebuilt
    them."""
    path = ROOT / "round2" / "candidates" / f"{cid}.json"
    if not path.exists():
        pytest.skip("round two's records are not here")
    rec = json.loads(path.read_text())
    res = _build(rec, tmp_path)
    assert _sha(res["xml"]) == rec["xml_sha256"] and not res["design"].compute_regulator


def test_the_option_at_its_default_is_todays_body(tmp_path):
    """The override at 0 and the selection's regulator key present, the Pi set as the compute part: the pick's model,
    byte for byte.  The option alone decides whether the regulator is placed."""
    pick = _pick_record()
    res = _build(pick, tmp_path, selection=dict(pick["selection"], compute_regulator=REGULATOR),
                 overrides=list(pick["overrides"]) + [["compute_regulator", 0]])
    assert _sha(res["xml"]) == pick["xml_sha256"] and not _regulator_rows(res["rows"])


# ------------------------------------------------------------------------------------------------
# The Jetson model and its record
# ------------------------------------------------------------------------------------------------
def test_the_jetson_model_regenerates_with_its_recorded_sha256(built):
    rec = plant.record(BODY)
    _, j = built
    assert _sha(j["xml"]) == rec["xml_sha256"] and rec["xml_sha256"].startswith(SHA256_PREFIX[BODY])
    assert len(SHA256_PREFIX[BODY]) == 8 and rec["xml_sha256"] != _pick_record()["xml_sha256"]
    model = case_dir(BODY) / "models" / f"{CID}.xml"
    if model.exists():  # git-ignored: the build's own file, or one copied in
        assert _sha(model) == rec["xml_sha256"]


def test_with_the_compute_swap_undone_the_model_is_the_picks(tmp_path):
    """The Jetson record's selection and overrides are the pick's but for three entries (the compute part, the
    regulator's part and its override); with those put back the build is the pick's model, byte for byte."""
    rec, pick = plant.record(BODY), _pick_record()
    sel = {k: v for k, v in rec["selection"].items() if k != "compute_regulator"}
    assert sel.pop("compute") == DEVKIT and rec["selection"]["compute_regulator"] == REGULATOR
    assert sel == {k: v for k, v in pick["selection"].items() if k != "compute"}
    ov = [o for o in rec["overrides"] if o[0] != "compute_regulator"]
    assert ov == pick["overrides"] and rec["overrides"] == pick["overrides"] + [["compute_regulator", 1]]
    assert rec["B"] == pick["B"] and rec["H"] == pick["H"]
    res = _build(rec, tmp_path, selection=dict(sel, compute=pick["selection"]["compute"]), overrides=ov)
    assert _sha(res["xml"]) == pick["xml_sha256"]


def test_the_jetson_case_is_its_pick_with_the_compute_changed():
    """``refine.CASES``: the pick's case's settings (sizing walks, headroom, turn gap), its id and overrides with the
    regulator's; the record is the pick's record (unchanged since: the same sha256) with the compute part changed
    and the regulator placed, built once and not re-sized."""
    c, b = refine.CASES[CASE], refine.CASES[BASE]
    assert c["jetson"] and c["base_case"] == BASE and refine.JETSON_BASE[CASE] == (BASE, CID) and c["ids"] == (CID,)
    assert c["overrides"] == tuple(b["overrides"]) + refine.JETSON_OVERRIDES == tuple(b["overrides"]) + (
        ("compute_regulator", 1),)
    assert {k: v for k, v in c.items() if k not in ("ids", "overrides", "label", "base_case", "jetson")} == {
        k: v for k, v in b.items() if k not in ("ids", "overrides", "label")}
    assert (refine.JETSON_COMPUTE, refine.JETSON_REGULATOR) == (DEVKIT, REGULATOR)
    rec, pick = plant.record(BODY), _pick_record()
    jr = rec["jetson"]
    base_path = ROOT / jr["base_record"]
    assert jr["base_case"] == BASE and base_path == ROOT / "round2" / "refine" / BASE / "candidates" / f"{CID}.json"
    assert jr["base_record_sha256"] == _sha(base_path) and jr["base_xml_sha256"] == pick["xml_sha256"]
    assert jr["rule"] == refine.JETSON_RULE and jr["home_pose"]["pick_xml_sha256"] == pick["xml_sha256"]
    assert rec["ok"] and rec["id"] == CID and rec["refine_case"] == CASE
    assert rec["selection"] == dict(pick["selection"], compute=DEVKIT, compute_regulator=REGULATOR)
    assert rec["packaging_search"] == [] and [h["iteration"] for h in rec["sizing_history"]] == ["jetson"]
    assert set(rec["margins"]) == set(sizing.GROUPS) == set(pick["margins"]) and set(rec["gaits"]) == set(pick["gaits"])
    assert {g: m["actuator"] for g, m in rec["margins"].items()} == {g: m["actuator"] for g, m in pick["margins"].items()}
    assert jr["selection_changed"] == dict(compute=[PI, DEVKIT], compute_regulator=[None, REGULATOR])
    # same drives within their planning limits, and one selection step on the heavier body's loads re-selects nothing
    assert jr["groups_over_planning_limit"] == [] and all(jr["meets_case_headroom"].values())
    sr = jr["selection_if_resized"]
    assert sr["differs"] == {} and not sr["battery_differs"] and not sr["bus_differs"]
    assert all(not v["differs"] and v["meets_target"] for v in sr["battery_by_mode"].values())
    assert jr["heat"]["modelled"] is False


def test_the_control_layer_is_the_owners_baseline(built):
    """Jetson + microcontroller (P30): the Teensy control stack and the IMU are the pick's rows, the Pi set's one
    part is gone, and neither a USB-CAN adapter nor a supervisor is placed (their masses are 'would add' estimates in
    the record, in no mass of the model)."""
    pick, j = built
    r0, rj = ({r["name"]: r for r in x["rows"]} for x in (pick, j))
    for name in ("control_stack", "imu", "power_distribution", "battery", "battery_cassette"):
        assert rj[name] == r0[name], name
    assert rj["control_stack"]["mass_kg"] == 0.18 and "Teensy 4.1" in rj["control_stack"]["source"]
    assert (r0["compute"]["source"], rj["compute"]["source"]) == (PI, DEVKIT)
    assert not any(t in f"{r['name']} {r['source']}".lower() for r in j["rows"] for t in ("pcan", "usb-can", "supervisor"))
    jr = plant.record(BODY)["jetson"]
    ps = jr["pi_set"]
    assert ps["decision"] == refine.JETSON_PI_SET_DECISION and ps["compute_part"] == PI
    assert [(r["name"], r["mass_kg"]) for r in ps["compute_rows"]] == [("compute", 0.1), ("compute_tray", 0.12)]
    assert ps["compute_power_rows_w"] == power.COMPUTE_BUDGET[PI] == {"pi5_ai_hat_plus": 8.0, "usb_ssd": 1.5}
    # the record holds no 5 V supply and no SSD of its own: the only other row naming a converter is the power box
    assert [r["name"] for r in ps["other_rows_naming_a_supply_converter_or_storage"]] == ["power_distribution"]
    npl = jr["not_placed"]
    adapter = catalog.part("pcan_usb_fd")
    assert npl["kind"] == "estimate" and npl["rule"] == refine.JETSON_NOT_PLACED_RULE
    assert npl["usb_can_adapter"]["part"] == "pcan_usb_fd" and adapter.mass == 0.068
    assert npl["usb_can_adapter"]["would_add_kg"] == {"1": 0.068, "2": 0.136}
    assert npl["usb_can_adapter"]["with_reserve_kg"]["2"] == pytest.approx(0.136 / 0.94, rel=1e-12)
    assert npl["supervisor"] == dict(part=None, would_add_kg=None, note="no catalog part")


# ------------------------------------------------------------------------------------------------
# The ledger, the mass and the centre of mass: the pick's plus the parts' sums
# ------------------------------------------------------------------------------------------------
def _key(r: dict) -> tuple:
    return (r["name"], r["body"], r["category"], r["mass_kg"], tuple(r["com_body"]),
            tuple(map(tuple, r["inertia_body"])), tuple(r["size"]))


def test_the_jetson_ledger_is_the_picks_but_for_the_compute_rows(built):
    """The ledger: the pick's rows exactly, in the pick's order, but for the compute board (the dev kit's catalog mass
    and box), its tray's envelope (the same 0.12 kg in a box 17 x 20 x 30 mm larger than the board) and the
    integration reserve (6 % of the installed mass, the regulator included); plus the regulator, last before the
    reserve.  The covers are the pick's: the trunk profile is not widened for the dev kit or the regulator."""
    pick, j = built
    r0, rj = pick["rows"], j["rows"]
    assert [r["name"] for r in rj if r["name"] != ADDED] == [r["name"] for r in r0]
    assert [r["name"] for r in rj][-2:] == [ADDED, "integration_reserve"]
    same = [(a, b) for a, b in zip(r0, [r for r in rj if r["name"] != ADDED], strict=True) if a["name"] not in CHANGED]
    assert all(_key(a) == _key(b) for a, b in same) and len(same) == len(r0) - 3
    by0, byj = {r["name"]: r for r in r0}, {r["name"]: r for r in rj}
    kit, pi, reg = catalog.part(DEVKIT), catalog.part(PI), catalog.part(REGULATOR)
    assert (kit.mass, kit.dims, kit.price) == (0.174, (0.103, 0.0905, 0.03477), 399)
    assert (pi.mass, pi.dims, pi.price) == (0.1, (0.085, 0.056, 0.03), 290.0)
    assert (by0["compute"]["mass_kg"], byj["compute"]["mass_kg"]) == (pi.mass, kit.mass)
    assert tuple(byj["compute"]["size"]) == kit.dims and byj["compute"]["com_body"] == by0["compute"]["com_body"]
    assert byj["compute_tray"]["mass_kg"] == by0["compute_tray"]["mass_kg"] == 0.12
    assert byj["compute_tray"]["com_body"] == by0["compute_tray"]["com_body"] == by0["compute"]["com_body"]
    assert np.allclose(byj["compute_tray"]["size"], np.array(kit.dims) + (0.017, 0.020, 0.030), rtol=0, atol=1e-15)
    added = _regulator_rows(rj)
    assert [(r["name"], r["body"], r["category"], r["source"]) for r in added] == [(ADDED, "trunk", "power", REGULATOR)]
    assert added[0]["mass_kg"] == reg.mass + build.COMPUTE_REGULATOR_LEADS_KG == 0.025 and reg.mass == 0.015
    assert tuple(added[0]["size"]) == reg.dims == (0.0318, 0.0432, 0.009) and added[0]["meta"] == {ADDED: REGULATOR}
    f = mjcf.INTEGRATION_RESERVE_FRACTION / (1 - mjcf.INTEGRATION_RESERVE_FRACTION)
    installed = (kit.mass - pi.mass) + added[0]["mass_kg"]
    assert installed == pytest.approx(0.099, abs=1e-15)
    assert byj["integration_reserve"]["mass_kg"] - by0["integration_reserve"]["mass_kg"] == pytest.approx(
        f * installed, abs=1e-12)
    total = sum(r["mass_kg"] for r in rj) - sum(r["mass_kg"] for r in r0)
    assert total == pytest.approx(installed / 0.94, abs=1e-12) and total == pytest.approx(0.105319, abs=5e-7)
    # the record's list of the changed rows is this one, and its sum is the ledgers' difference
    jr = plant.record(BODY)["jetson"]
    assert [(p["name"], p["status"]) for p in jr["parts"]] == [("compute", "changed"), ("compute_tray", "changed"),
                                                                (ADDED, "added"), ("integration_reserve", "changed")]
    assert jr["parts_mass_change_kg"] == pytest.approx(total, abs=1e-12)
    assert jr["ledger_mass_kg"]["change"] == pytest.approx(total, abs=1e-12)
    assert [p["mass_change_kg"] for p in jr["parts"][:3]] == pytest.approx([0.074, 0.0, 0.025], abs=1e-15)


def test_the_exterior_is_the_picks(built):
    """The covers, the crown and the widths are the pick's (the record's exterior block; the thigh width moves by the
    re-solved stance alone, about a micrometre)."""
    pick, j = built
    assert j["design"].geometry_checks["trunk_widened_for_hardware"] == pick["design"].geometry_checks[
        "trunk_widened_for_hardware"]
    ext = plant.record(BODY)["jetson"]["exterior_m"]
    assert ext["crown"]["change"] == ext["torso_width"]["change"] == ext["length"]["change"] == 0.0
    assert all(abs(v["change"]) < 5e-6 for v in ext.values())


def _composite(m, d) -> tuple:
    """Mass, first moment and inertia about the world origin (world axes) of a compiled model at ``d``'s pose."""
    mass, moment, inertia = 0.0, np.zeros(3), np.zeros((3, 3))
    for b in range(1, m.nbody):
        R, r, mb = d.ximat[b].reshape(3, 3), d.xipos[b], m.body_mass[b]
        mass += mb
        moment += mb * r
        inertia += R @ np.diag(m.body_inertia[b]) @ R.T + mb * (r @ r * np.eye(3) - np.outer(r, r))
    return float(mass), moment, inertia


def test_mass_and_centre_of_mass_deltas_equal_the_parts_sums(built):
    """The compiled models, both at the pick's home pose (the same joints in the same order): the Jetson model's
    mass, first moment and inertia about the origin are the pick's with the changed rows' exchanged, each row a rigid
    part of the trunk (the MJCF writes six significant digits, hence the tolerances).  The centre of mass moves
    forward and up, by the record's figures."""
    pick, j = built
    m0, mj = mujoco.MjModel.from_xml_path(str(pick["xml"])), mujoco.MjModel.from_xml_path(str(j["xml"]))
    assert m0.nq == mj.nq and [mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, k) for k in range(m0.njnt)] == [
        mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, k) for k in range(mj.njnt)]
    d0, dj = mujoco.MjData(m0), mujoco.MjData(mj)
    d0.qpos[:] = dj.qpos[:] = m0.key_qpos[0]
    mujoco.mj_forward(m0, d0)
    mujoco.mj_forward(mj, dj)
    t0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    R, p = d0.xmat[t0].reshape(3, 3), d0.xpos[t0]
    mass, moment, inertia = _composite(m0, d0)
    for sign, rows in ((-1.0, pick["rows"]), (1.0, j["rows"])):
        for r in rows:
            if r["name"] in CHANGED + (ADDED,):
                assert r["body"] == "trunk"
                c = p + R @ np.array(r["com_body"])
                mass += sign * r["mass_kg"]
                moment += sign * r["mass_kg"] * c
                inertia += sign * (R @ np.array(r["inertia_body"]) @ R.T
                                   + r["mass_kg"] * (c @ c * np.eye(3) - np.outer(c, c)))
    got = _composite(mj, dj)
    assert got[0] == pytest.approx(mass, rel=2e-6)
    assert np.allclose(got[1] / got[0], moment / mass, rtol=0, atol=2e-6)
    assert np.allclose(got[2], inertia, rtol=0, atol=2e-6 * np.abs(inertia).max())
    m_pick = _composite(m0, d0)
    shift = got[1] / got[0] - m_pick[1] / m_pick[0]
    assert got[0] - m_pick[0] == pytest.approx(0.10532, abs=2e-5)
    assert 0.0015 < shift[0] < 0.0018 and abs(shift[1]) < 1e-6 and 0.0004 < shift[2] < 0.0006
    jr = plant.record(BODY)["jetson"]
    com = jr["com_at_pick_pose_m"]
    assert np.allclose(com["change"], shift, rtol=0, atol=2e-6)
    assert np.allclose(com["change"], com["from_parts_change"], rtol=0, atol=1e-6)
    assert jr["mass_kg"]["change"] == pytest.approx(0.10532, abs=2e-6)
    assert jr["home_pose"]["trunk_pose_change_max"] == 0.0
    # at its own home the stance puts the feet under the new centre of mass, which moves it a little further
    ch = jr["com_ahead_of_hip_m"]
    assert ch["change_at_pick_pose"] == pytest.approx(com["change"][0], abs=1e-12)
    assert 0.0015 < ch["change_at_pick_pose"] < ch["change"] < 0.0022


# ------------------------------------------------------------------------------------------------
# The regulator: which, and where
# ------------------------------------------------------------------------------------------------
def test_the_regulator_is_the_catalog_part_that_covers_both_modes():
    """``refine.JETSON_REGULATOR_RULE`` on the catalog: the 4.5 A part's rated 54 W covers the 15 W mode's worst
    case at the jack (18 W of rail budget plus the carrier's 30 W USB limit) and not the 25 W mode's (58 W); the 9 A
    part's 108 W covers both, so it is the one placed, 8 g and $30 more."""
    rr = refine.jetson_regulators()
    recorded = {k: v for k, v in plant.record(BODY)["jetson"]["regulator"].items() if k != "geometry"}
    assert recorded == json.loads(json.dumps(rr)) and rr["rule"] == refine.JETSON_REGULATOR_RULE
    assert refine.JETSON_REGULATOR_OPTIONS == (SMALLER, REGULATOR) and refine.JETSON_USB_VBUS_W == 30.0
    assert rr["rail_budget_w"] == {"15w": 18.0, "25w": 28.0} and rr["worst_case_at_jack_w"] == {"15w": 48.0, "25w": 58.0}
    small, big = (catalog.part(x) for x in (SMALLER, REGULATOR))
    assert all(p.category == "dcdc" and "12V" in p.model for p in (small, big))
    assert (small.power_max, small.max_current, small.voltage, small.mass, small.price) == (54, 4.5, 50.0, 0.007, 39.95)
    assert (big.power_max, big.max_current, big.voltage, big.mass, big.price) == (108, 9, 60.0, 0.015, 69.95)
    assert rr["options"][SMALLER]["covers"] == {"15w": True, "25w": False}
    assert rr["options"][REGULATOR]["covers"] == {"15w": True, "25w": True}
    assert rr["choice"] == rr["placed"] == REGULATOR == plant.record(BODY)["selection"]["compute_regulator"]
    assert min(small.voltage, big.voltage) > rr["pack_full_v"] == 25.2  # both take a full 6S pack (E7 is about 12S)


def test_the_regulator_sits_on_the_power_box_clear_of_everything(built):
    """Flat on the power box's top face, centred, 5 mm above it on its standoffs, in the power box's frame; inside
    the as-built trunk profile with the build's allowance; no component overlap on the trunk, and every gap to the
    hardware near it positive, in the build's own check and in the record's reading (which a rebuild repeats)."""
    pick, j = built
    d = j["design"]
    items = {it.name: it for it in d.bodies["trunk"].items}
    reg, box = items[ADDED], items["power_distribution"]
    lift = 0.5 * box.size[2] + build.COMPUTE_REGULATOR_STANDOFF_M + 0.5 * reg.size[2]
    assert build.COMPUTE_REGULATOR_STANDOFF_M == 0.005 and np.array_equal(reg.rot, box.rot)
    assert np.allclose(reg.pos, box.pos + box.rot @ np.array([0.0, 0.0, lift]), rtol=0, atol=1e-15)
    assert reg.volume_class == "A" and reg.category == "power" and reg.mass == 0.025
    gc = d.geometry_checks["compute_regulator"]
    assert gc["part"] == REGULATOR and gc["on"] == "power_distribution" and gc["inside_trunk_profile"]
    assert gc["trunk_profile_margin_m"] == d.profile_margin(reg) and min(gc["trunk_profile_margin_m"].values()) > 0.03
    assert (gc["part_mass_kg"], gc["leads_and_mount_kg"], gc["mass_kg"]) == (0.015, 0.010, 0.025)
    assert "compute_regulator" not in pick["design"].geometry_checks
    rec = plant.record(BODY)
    assert json.loads(json.dumps(gc)) == rec["geometry_checks"]["compute_regulator"] == rec["jetson"]["regulator"]["geometry"]
    chest = refine.jetson_chest(j["xml"], j["q_home"], d)
    got = json.loads(json.dumps(chest))
    want = rec["jetson"]["chest"]["jetson"]
    assert got["same_body_component_overlaps"] == want["same_body_component_overlaps"]
    assert set(got["gaps_mm"]) == set(want["gaps_mm"])
    assert all(got["gaps_mm"][k] == pytest.approx(v, abs=1e-6) for k, v in want["gaps_mm"].items())
    assert want["same_body_component_overlaps"]["overlaps"] == []
    g = want["gaps_mm"]
    assert g["compute_regulator|power_distribution"] == pytest.approx(5.0, abs=1e-6)
    assert all(v > 50.0 for k, v in g.items() if k.startswith("compute_regulator|") and "power_distribution" not in k)


def test_the_chest_readings_beside_the_picks(built):
    """The dev kit in the chest tray against the Pi set: the tray's envelope is 18 mm longer, so its gap to the
    power box (which does not move on this body: its rule's other bound holds) is 9 mm less and still open; the
    hip-roll motors are far behind; the pick's own reading repeats on its rebuilt model."""
    pick, j = built
    ch = plant.record(BODY)["jetson"]["chest"]
    gj, g0 = ch["jetson"]["gaps_mm"], ch["base"]["gaps_mm"]
    assert set(g0) == {k for k in gj if not k.startswith("compute_regulator|")}
    got0 = json.loads(json.dumps(refine.jetson_chest(pick["xml"], pick["q_home"], pick["design"])))
    assert all(got0["gaps_mm"][k] == pytest.approx(v, abs=1e-6) for k, v in g0.items())
    assert ch["jetson"]["centre_trunk_mm"]["power_distribution"] == ch["base"]["centre_trunk_mm"]["power_distribution"]
    assert ch["jetson"]["centre_trunk_mm"]["compute"] == ch["base"]["centre_trunk_mm"]["compute"]
    for k in g0:  # the power box's own gaps are the pick's; the dev kit's are smaller, and all are open
        if k.startswith("power_distribution|"):
            assert gj[k] == pytest.approx(g0[k], abs=1e-9)
        elif k.startswith("compute_tray|"):
            assert 0.0 < gj[k] < g0[k]
    assert gj["compute_tray|power_distribution"] == pytest.approx(20.70, abs=0.01)
    assert g0["compute_tray|power_distribution"] - gj["compute_tray|power_distribution"] == pytest.approx(9.0, abs=0.01)
    assert min(gj["compute_tray|left_hip_roll_motor"], gj["compute_tray|right_hip_roll_motor"]) > 170.0
    assert gj["power_distribution|left_hip_roll_motor"] > 60.0
    assert ch["base"]["same_body_component_overlaps"]["overlaps"] == []
    pm = ch["jetson"]["trunk_profile_margin_mm"]
    assert all(min(v.values()) > 0.0 for v in pm.values()) and set(pm) == set(refine.JETSON_CHEST_ITEMS)


# ------------------------------------------------------------------------------------------------
# Power: the existing budgets are unchanged, and the two mode rows
# ------------------------------------------------------------------------------------------------
def test_the_power_rows_leave_existing_budgets_unchanged(built):
    """``power.COMPUTE_BUDGET`` is what it was (the dev kit's 31 W allowance of round one, A13, which
    ``compute_options`` reads, and the Pi set's 9.5 W, A14); a call without a mode books those rows; the dev kit's two
    mode rows stand beside them (draft A19) and the Pi set has none."""
    assert power.COMPUTE_BUDGET == {DEVKIT: {"jetson_module": 25.0, "carrier_storage_cooling": 6.0},
                                    PI: {"pi5_ai_hat_plus": 8.0, "usb_ssd": 1.5}}
    assert power.AUX_FIXED == {"mcu_comms": 4.0} and (power.ETA_DCDC, power.USABLE, power.TARGET_MIN) == (0.88, 0.75, 20.0)
    assert compute_options.JETSON_W == 31.0
    assert power.COMPUTE_MODE_BUDGET == {DEVKIT: {"15w": {"jetson_module": 15.0, "carrier_storage_cooling": 3.0},
                                                  "25w": {"jetson_module": 25.0, "carrier_storage_cooling": 3.0}}}
    assert power.compute_loads(DEVKIT) == power.COMPUTE_BUDGET[DEVKIT] and power.compute_loads(PI) == power.COMPUTE_BUDGET[PI]
    assert power.compute_loads(DEVKIT) is not power.COMPUTE_BUDGET[DEVKIT]
    assert {m: sum(power.compute_loads(DEVKIT, m).values()) for m in ("15w", "25w")} == {"15w": 18.0, "25w": 28.0}
    kit = catalog.part(DEVKIT)
    assert (kit.power_typ, kit.power_max) == (15.0, 25) == tuple(
        power.COMPUTE_MODE_BUDGET[DEVKIT][m]["jetson_module"] for m in ("15w", "25w"))
    for bad in ((PI, "15w"), (DEVKIT, "7w")):
        with pytest.raises(KeyError):
            power.compute_loads(*bad)
    pick, j = built
    rec0, recj = _pick_record(), plant.record(BODY)
    assert power.aux_power(pick["design"]) == rec0["power"]["aux"]  # the Pi body's auxiliaries, to the last bit
    assert power.aux_power(j["design"]) == recj["power"]["aux"]
    assert power.aux_power(j["design"])["loads"] == {"jetson_module": 25.0, "carrier_storage_cooling": 6.0,
                                                    "mcu_comms": 4.0, "sensors": rec0["power"]["aux"]["loads"]["sensors"]}
    with pytest.raises(KeyError):
        power.aux_power(pick["design"], "15w")
    pack = catalog.part(rec0["selection"]["battery"])
    assert power.runtime(pack, rec0["power"]["total_w"]) == rec0["runtime"]
    assert rec0["power"]["total_w"] == rec0["power"]["drives_w"] + rec0["power"]["aux"]["input_w"]


def _walking(res: dict, case: str) -> tuple:
    """The sizing's requirements of a built body under its case's settings (``refine._init``, which sets the
    process's sizing settings and walk accuracy: put back afterwards), and the nominal walk's motor series."""
    before = (gait.ACCURACY, sizing.HEADROOM, sizing.HEADROOM_BY_GROUP, sizing.EXTRA_WALKS, round2.TURN_SEARCH_MIN_GAP_MM)
    try:
        refine._init(case)
        design = res["design"]
        req = sizing.requirements(res["model"], res["q_home"], design.sk.H, design.cand.leg_layout)
    finally:
        gait.set_accuracy(before[0])
        sizing.HEADROOM, sizing.HEADROOM_BY_GROUP, sizing.EXTRA_WALKS, round2.TURN_SEARCH_MIN_GAP_MM = before[1:]
    nom = req["runs"]["nominal"]["idr"]
    return req, sizing._motor_space(res["model"].names, nom["tau"], nom["omega"], design.cand.leg_layout)


def test_an_existing_records_power_and_runtime_recompute(built):
    """The pick's recorded walking power and runtime, recomputed with today's ``power.py`` on its rebuilt model: the
    same to the solver's noise (the record was sized with several BLAS threads, which moves the last digits of the
    walks' inverse dynamics), and the same runtime."""
    pick, _ = built
    rec = _pick_record()
    req, ms = _walking(pick, BASE)
    pw = power.walking_power(pick["design"], req, ms)
    assert pw["aux"] == rec["power"]["aux"] and pw["drives_w"] == pytest.approx(rec["power"]["drives_w"], rel=1e-9)
    assert pw["total_w"] == pytest.approx(rec["power"]["total_w"], rel=1e-9)
    rt = power.runtime(catalog.part(rec["selection"]["battery"]), pw["total_w"])
    assert rt["minutes"] == pytest.approx(rec["runtime"]["minutes"], rel=1e-9) and rt["pack"] == rec["runtime"]["pack"]
    assert rec["runtime"]["minutes"] == pytest.approx(36.7257, abs=1e-4)


def test_the_jetson_runtimes_in_both_modes(built):
    """The Jetson body's power and runtime by the records' own method on its own walk: the record's ``power`` and
    ``runtime`` book the dev kit at ``COMPUTE_BUDGET``'s 31 W, and ``jetson.power_modes`` at each mode's row; the
    drives' power is the heavier body's own (above the pick's), the same in every mode; both modes meet the 20 minute
    target on the pick's pack, below the Pi body's runtime."""
    _, j = built
    rec, rec0 = plant.record(BODY), _pick_record()
    req, ms = _walking(j, CASE)
    pack = catalog.part(rec["selection"]["battery"])
    assert pack.id == rec0["selection"]["battery"] == "lipo_6s_3300" and pack.capacity_wh == 73.26
    pw = power.walking_power(j["design"], req, ms)
    assert pw["total_w"] == pytest.approx(rec["power"]["total_w"], rel=1e-12) and pw["aux"] == rec["power"]["aux"]
    assert power.runtime(pack, pw["total_w"])["minutes"] == pytest.approx(rec["runtime"]["minutes"], rel=1e-12)
    pm = rec["jetson"]["power_modes"]
    assert set(pm["modes"]) == {"15w", "25w"} and pm["no_mode_chosen"] == "DECISION_LOG P30"
    sensors = rec0["power"]["aux"]["loads"]["sensors"]
    for mode, rail in (("15w", 18.0), ("25w", 28.0)):
        got = power.walking_power(j["design"], req, ms, compute_mode=mode)
        m = pm["modes"][mode]
        assert got["aux"] == m["aux"] and got["drives_w"] == pw["drives_w"] == pytest.approx(m["drives_w"], rel=1e-12)
        assert m["compute_rail_w"] == rail and m["aux"]["load_w"] == pytest.approx(rail + 4.0 + sensors, abs=1e-12)
        assert m["aux"]["input_w"] == pytest.approx((rail + 4.0 + sensors) / 0.88, rel=1e-12)
        assert m["total_w"] == pytest.approx(m["drives_w"] + m["aux"]["input_w"], rel=1e-12)
        assert m["total_w"] == pytest.approx(got["total_w"], rel=1e-12)
        assert m["runtime"] == power.runtime(pack, m["total_w"])
        assert m["runtime"]["minutes"] == pytest.approx(60.0 * 73.26 * 0.75 / m["total_w"], rel=1e-12)
        assert m["meets_target"] and m["runtime"]["minutes"] > 20.0
        assert m["runtime_change_against_base_min"] == pytest.approx(m["runtime"]["minutes"] - rec0["runtime"]["minutes"])
    r15, r25 = (pm["modes"][m]["runtime"]["minutes"] for m in ("15w", "25w"))
    assert rec["runtime"]["minutes"] < r25 < r15 < pm["survey_typical_15w"]["runtime_min"] < rec0["runtime"]["minutes"]
    assert (r15, r25) == pytest.approx((32.88, 29.53), abs=0.01)
    assert rec["power"]["drives_w"] > rec0["power"]["drives_w"] == pm["base"]["drives_w"]
    assert pm["base"]["runtime_min"] == rec0["runtime"]["minutes"] and pm["base"]["compute_rail_w"] == 9.5
    assert pm["record_default"]["compute_rail_w"] == 31.0 and pm["survey_typical_15w"]["compute_rail_w"] == 15.0


# ------------------------------------------------------------------------------------------------
# The registry, the references and the screens
# ------------------------------------------------------------------------------------------------
def test_the_jetson_body_is_registered_beside_the_earlier_ones():
    assert JETSON_BODIES == (BODY,) and set(BODIES) == set(EARLIER) | {BODY} and set(SHA256_PREFIX) == set(BODIES)
    assert BODIES[BODY] == f"round2/refine/{CASE}" and CANDIDATE_IDS[BODY] == CID == candidate_id(BODY)
    assert case_dir(BODY) == STUDY / "round2" / "refine" / CASE
    assert {k: v for k, v in BODIES.items() if k != BODY} == {b: f"round2/refine/{c}" for b, c in EARLIER.items()}
    assert refs.reference_names(BODY) == refs.reference_names(CID)


def test_the_jetson_carpet_swing_is_its_picks():
    """``refs.CARPET_LIFT``: the Jetson body takes its pick's lift, which the carpet-lift pass on its own case (the
    pick's lifts and walks) gives by the pick's rule: the lowest lift that meets the target on every pile with the
    drives within their limits and the walking clearance gates kept."""
    lift, path = refs.CARPET_LIFT[BODY], refs.CARPET_LIFT_RECORD[BODY]
    assert lift == refs.CARPET_LIFT[CID] and CASE in path
    if not (STUDY / path).exists():
        pytest.skip(f"{path}: the carpet-lift pass has not run (stage2/j/run_j.sh build)")
    pick = next(r for r in json.loads((STUDY / refs.CARPET_LIFT_RECORD[CID]).read_text()) if r["id"] == CID)
    row = next(r for r in json.loads((STUDY / path).read_text()) if r["id"] == CID)
    assert row["source"] == f"round2/refine/{CASE}" and set(row["lifts"]) == set(pick["lifts"])
    ok = {float(k): v["drives_within_limits"] and v["walking_gates"] and all(
        all(t["meets_carpet_target"].values()) for t in v["toe"].values()) for k, v in row["lifts"].items()}
    assert lift == min(k for k, v in ok.items() if v)


def _screens() -> tuple:
    idx = case_dir(BODY) / "review" / "review_index.json"
    if not idx.exists():
        pytest.skip("stage2/j/run_j.sh build has not screened the case")
    return (json.loads(idx.read_text())[CID],
            json.loads((ROOT / "round2" / "refine" / BASE / "review" / "review_index.json").read_text())[CID])


def test_every_round_two_screen_ran_on_the_jetson_body():
    """The five screens of ``round2.screen_one`` on the sized model (the same sha256), each with its record, and each
    gate as its own screen reads it: every clearance gate, the turning targets and the stretch case, the carpet gate
    and the envelope's walk."""
    j, _ = _screens()
    rec = plant.record(BODY)
    assert j["matches_sized"] and j["xml_sha256"] == j["sized_sha256"] == rec["xml_sha256"]
    assert not [k for k in j if k.endswith("_error")]
    for name in ("clearance", "sensors", "turning", "carpet", "envelope"):
        assert name in j and (case_dir(BODY) / "review" / "screening" / f"{CID}_{name}.json").exists(), name
    cl = j["clearance"]
    assert cl["gate_set"] == "r2" and cl["pass"] == {k: True for k in (
        "interference_free_home", "interference_free_nominal_gait", "swing_clearance", "tail_clearance", "battery_path",
        "interference_free_brisk_gait", "component_overlap_free", "joint_range_margin", "all")}
    assert all(cl["pass"][g] for g in round2.WALK_GATES) and cl["same_body_component_overlaps"] == []
    assert cl["hardware_interference_at_home"] == [] and cl["worst_pairs"] == []
    t = j["turning"]
    assert t["targets_pass"] and t["stretch_pass"] and t["xml_sha256_matches_record"] and t["ik_ok_all"]
    assert j["carpet"]["pass"] and j["carpet"]["mass_kg"] == pytest.approx(rec["metrics"]["mass"])
    e = j["envelope"]
    assert e["fast_walk_drives_ok"] and e["fast_walk_reference_cop_on_pad"] and e["fast_walk_ik_ok"]


def test_the_screens_beside_the_picks():
    """What the screens read differently on the Jetson body, and what they read the same: the battery's removal path
    and the sensors' views are the pick's (the covers and the sensors do not move; the home pose moves the feet by
    millimetres), the walks' margins and the loads move a little with the mass."""
    j, p = _screens()
    cj, cp = j["clearance"], p["clearance"]
    assert cj["battery_clear_directions"] == cp["battery_clear_directions"] == ["-z"]
    assert cj["battery_overlapping_at_home"] == cp["battery_overlapping_at_home"] == []
    assert {k: v["usable"] for k, v in cj["joint_range_margins"].items()} == pytest.approx(
        {k: v["usable"] for k, v in cp["joint_range_margins"].items()}, abs=2e-3)
    sj, sp = j["sensors"], p["sensors"]
    assert sj["camera_blocked"] == sp["camera_blocked"] and sj["camera_height_m"] == sp["camera_height_m"]
    for side in ("tof_l", "tof_r"):
        assert sj["tof"][side]["floor_zones"] == sp["tof"][side]["floor_zones"] == 64
        assert sj["tof"][side]["self_zones"] == sp["tof"][side]["self_zones"] == {}
        assert sj["tof"][side]["floor_ahead_of_toes_m"] == pytest.approx(sp["tof"][side]["floor_ahead_of_toes_m"], abs=3e-3)
    assert sj["person_visible_fraction_home"] == sp["person_visible_fraction_home"]
    assert sj["torso_follow"]["home"] == sp["torso_follow"]["home"]
    assert j["envelope"]["fast_walk_worst"][0] == p["envelope"]["fast_walk_worst"][0] == "knee"
    assert p["envelope"]["fast_walk_worst"][1] < j["envelope"]["fast_walk_worst"][1] < 1.0
    assert j["turning"]["worst_ratio_targets"] < 1.0 and j["turning"]["hip_yaw_range_used_max"] < 1.0
    # the hip pitch holds more of the heavier, further-forward body, within its drive's sustained limit
    ab = plant.record(BODY)["jetson"]["against_base"]["hip_pitch"]
    assert ab["base_hold_Nm"] < ab["hold_Nm"] and ab["base_hold_over_sustained"] < ab["hold_over_sustained"] < 0.85


# ------------------------------------------------------------------------------------------------
# Stage 2's run records (stage2/j/<body>), where present
# ------------------------------------------------------------------------------------------------
def test_the_jetson_prepare_records():
    """plant.json, refs.json and transitions.json: its own model (its record's), the decided plant (decisions 11-12)
    with the pick's 14 hinges and drives, every reference matching the records it names, every transition there."""
    from designlab.stage1 import transitions

    if not (RUN / BODY / "plant.json").exists():
        pytest.skip("stage2/j/run_j.sh run has not made this body's records")
    pj = json.loads((RUN / BODY / "plant.json").read_text())
    rec = plant.record(BODY)
    assert pj["body"] == BODY and pj["model"]["matches_record"] and pj["model"]["matches_plan"]
    assert pj["model"]["record_sha256"] == rec["xml_sha256"] and pj["model"]["plan_prefix"] == SHA256_PREFIX[BODY]
    assert (pj["options"]["ankle_gains"], pj["options"]["ankle_armature"]) == ("joint", "linkage")
    assert pj["mass_kg"] == pytest.approx(rec["metrics"]["mass"])
    assert len(pj["joint_ranges"]["plant"]) == 14 and len(pj["drives"]) == 14
    pick_plant = ROOT / "stage1" / "runs" / "2026-11-08" / CID / "plant.json"
    if pick_plant.exists():
        assert {d["actuator"]: d["part"] for d in pj["drives"]} == {
            d["actuator"]: d["part"] for d in json.loads(pick_plant.read_text())["drives"]}
    if (RUN / BODY / "refs.json").exists():
        rj = json.loads((RUN / BODY / "refs.json").read_text())
        assert rj["body"] == BODY and rj["matches_records"] and rj["model"]["record_sha256"] == rec["xml_sha256"]
        assert set(rj["references"]) == set(refs.reference_names(BODY)) and len(rj["joints"]) == 14
        for e in rj["references"].values():
            assert e["record_checks"] and all(c["match"] for c in e["record_checks"])
    if (RUN / BODY / "transitions.json").exists():
        tj = json.loads((RUN / BODY / "transitions.json").read_text())
        assert tj["body"] == BODY and set(tj["transitions"]) == set(transitions.TRANSITIONS)


def test_the_jetson_gate_records():
    """gates.json: G1-G3 as Stage 1 ran them (the decided plant and G2 rule, decisions 11-13), each verdict as its rule
    reads it.  G1's step rate records the machine's cores: timed on fewer processes than cores (a shared machine), it
    is not judged, and G1 and the record's pass are provisional (None) until ``run_j.sh g1`` times it with one
    process per core.  G4 is not run."""
    from designlab.stage1 import gates
    from designlab.stage1.__main__ import summary

    path = RUN / BODY / "gates.json"
    if not path.exists():
        pytest.skip("stage2/j/run_j.sh run has not made this body's gates")
    doc = json.loads(path.read_text())
    assert doc["body"] == BODY and "G4" not in doc
    g1 = doc.get("G1")
    if g1:
        sr = g1["step_rate"]
        assert all(h["matches_record"] and h["matches_plan"] for h in g1["hashes"].values())
        assert all(s["stands"] for s in g1["stand"].values()) and g1["tendon_map"]["matches"] and g1["repeatability"]["ok"]
        assert g1["pass"] == gates.combine([all(s["stands"] for s in g1["stand"].values()), g1["tendon_map"]["matches"],
                                            g1["repeatability"]["ok"], sr["ok"]])
        on_rate = min(np.median(sr["single_process_samples"]), np.median(sr["per_process_samples"])) >= 20_000
        if sr["processes"] == sr["machine"]["cores"]:
            assert sr["ok"] == on_rate
        else:  # timed on fewer processes than cores: provisional, never a pass
            assert sr["ok"] is None and sr["ok_on_these_processes"] == on_rate and g1["pass"] in (None, False)
            assert sr["provisional"].startswith(f"step rate timed on {sr['processes']} of {sr['machine']['cores']} cores")
            if g1["pass"] is None:
                assert g1["provisional"] == sr["provisional"] and "G1 provisional (step rate timed on" in summary(doc)
    g2 = doc.get("G2")
    if g2:
        assert g2["rule"] == "panel" and g2["decided"]["follows_all"]
        assert g2["pass"] == (g2["all_match"] and g2["panel_pass"])
    if "G3" in doc:
        assert doc["G3"]["pass_"] == (not doc["G3"]["failed"] and not doc["G3"]["missing"])
    if g1 and g2 and "G3" in doc:
        passes = [g1["pass"], g2["pass"], doc["G3"]["pass_"]]
        assert doc["pass"] == (gates.combine(passes) if doc["same_plant_options"] else False)
        assert (doc["pass"] is None) == ("provisional" in doc)
    for g in ("G1", "G2", "G3"):
        if g in doc:
            assert doc[g]["provenance"]["stage1_sources"]["unchanged_during_run"]


def test_the_list_of_git_ignored_inputs():
    """``j_inputs.json`` lists the case's two model files and the body's 10 reference and 18 transition arrays with
    their sha256 (the XML's is the record's), under the helper's commands a restore group needs; where the files are
    here, the helper's check passes."""
    from designlab.stage1 import transitions

    mod = _helper()
    assert mod.J_BODIES == JETSON_BODIES and mod.case_of(BODY) == CASE
    xml, ledger = mod.model_paths(BODY)
    assert (xml, ledger) == (f"round2/refine/{CASE}/models/{CID}.xml", f"round2/refine/{CASE}/models/{CID}.json")
    assert all(callable(getattr(mod, f)) for f in ("check", "copy_from", "models", "arrays", "records", "manifest"))
    if not mod.MANIFEST.exists():
        pytest.skip("stage2/j/run_j.sh run has not written j_inputs.json")
    doc = json.loads(mod.MANIFEST.read_text())
    files = doc["files"]
    assert doc["bodies"] == [BODY] and doc["cases"] == {BODY: CASE} and doc["rule"] == mod.RULE
    assert files[xml] == plant.record(BODY)["xml_sha256"] and len(files[ledger]) == 64
    names = set(refs.reference_names(BODY))
    assert {p for p in files if "/refs/" in p} == {f"stage2/j/{BODY}/refs/{n}.npz" for n in names} and len(names) == 10
    assert {p for p in files if "/transitions/" in p} == {f"stage2/j/{BODY}/transitions/{n}.npz"
                                                          for n in transitions.TRANSITIONS}
    assert len(files) == 2 + 10 + len(transitions.TRANSITIONS) == 30
    assert set(mod.recorded(BODY)) == set(files)
    if all((ROOT / p).exists() for p in files):
        assert mod.check(quiet=True) == 0


def test_the_reports_tables_regenerate_from_the_records():
    """``JETSON_BODY.md``'s tables are ``j_tables.py``'s, written from the committed records (the case's record and
    screens beside the pick's, the carpet-lift pass, the run's gates): the block in the document is the script's output
    on today's records, so no figure in it is typed by hand."""
    doc = RUN / "JETSON_BODY.md"
    if not doc.exists() or not (RUN / BODY / "gates.json").exists():
        pytest.skip("the report or the run's records are not here")
    mod = _tables()
    text = doc.read_text()
    assert text.count(mod.BEGIN) == 1 and text.count(mod.END) == 1
    assert text.split(mod.BEGIN)[1].split(mod.END)[0] == mod.tables()
    for kind in ("(a)", "(c)", "(d)"):
        assert kind in text
