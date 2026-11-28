"""Tests for the form-factor study (design/form_factor_study.py): the torque scaling rule, the mass model against the
60 cm ledger, the build's thermal model called through the study against stored build figures, the packaging code
against the packaging report, the cost rule against the budget's output, and the criteria logic. Run by hand:
    python -m pytest hardware/triceratops_build/design/tests/test_form_factor_study.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

DESIGN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESIGN))

import form_factor_study as F  # noqa: E402

D0 = F.DRIVE_SETS["D0"]


@pytest.fixture(scope="module")
def ledger():
    return F.load_ledger60()


@pytest.fixture(scope="module")
def demos():
    return F.load_demos()


# ---------------------------------------------------------------- the scaling rule
def test_torque_factor_is_one_at_the_simulated_60_cm_model():
    assert F.torque_factor(F.M60_SIM_KG, 0.60) == pytest.approx(1.0)


def test_torque_factor_is_m_g_L_and_matches_the_budget_rule():
    # budget/size_budget.py:55: k = (mass / 15 kg) x s75; the study's k is the same rule normalised at 60 cm
    import size_budget as sb

    for m, S in ((11.8, 0.50), (15.3, 0.75), (20.0, 0.80)):
        k_budget = (m / sb.REF["mass"]) * (S / 0.75)
        k_budget60 = (F.M60_SIM_KG / sb.REF["mass"]) * (0.60 / 0.75)
        assert F.torque_factor(m, S) == pytest.approx(k_budget / k_budget60)
        # m g L: doubling mass doubles k; doubling length doubles k
    assert F.torque_factor(2 * F.M60_SIM_KG, 0.60) == pytest.approx(2.0)
    assert F.torque_factor(F.M60_SIM_KG, 1.20) == pytest.approx(2.0)


def test_walk_speed_is_froude_matched_and_capped_at_the_owners_slow_walk():
    v, v60 = F.walk_speed(0.50)
    assert v == pytest.approx(0.25 * math.sqrt(0.5 / 0.6)) and v60 == pytest.approx(0.25)
    assert F.walk_speed(0.60) == pytest.approx((0.25, 0.25))
    v, v60 = F.walk_speed(0.80)
    assert v == pytest.approx(0.25) and v60 == pytest.approx(0.25 / math.sqrt(0.8 / 0.6))
    assert 0.15 <= v60 <= 0.25  # inside the stored segments: interpolation, not extrapolation


def test_rule_reproduces_the_75_cm_standing_torques_within_5_percent():
    r = F.rule_check_75()
    assert r["k"] == pytest.approx(15.0 / 12.69 * 0.75 / 0.60)
    for g, e in r["error"].items():
        assert abs(e) < 0.05, (g, e)


def test_profile_interpolates_between_the_stored_segments(demos):
    d = demos["s2"]
    a = np.asarray(d["segments"]["walk_0.15"]["tau_drive_sq_mean"])
    b = np.asarray(d["segments"]["walk_0.25"]["tau_drive_sq_mean"])
    assert np.allclose(F.seg_at(d, "tau_drive_sq_mean", 0.25), b)
    assert np.allclose(F.seg_at(d, "tau_drive_sq_mean", 0.20), 0.5 * (a + b))
    p = F.config_profile(d, 0.60, 1.0, D0)
    assert np.allclose(p["walk_sq"], b)


# ---------------------------------------------------------------- the mass model
def test_mass_model_reproduces_the_60_cm_ledger_within_1_percent(ledger):
    m = F.mass_model(0.60, D0, layout=None, bms=False, ledger=ledger)["total_kg"]
    assert m == pytest.approx(ledger["total_mass_kg"], rel=0.01)
    assert m == pytest.approx(12.7506, abs=1e-3)


def test_mass_split_matches_the_fact_sheet(ledger):
    sp = F.mass_split_60(ledger)
    assert sp["drive"] == pytest.approx(5.24, abs=1e-3)
    assert sp["battery"] == pytest.approx(1.012, abs=1e-3)
    assert sp["scales: structure (s^2.5)"] == pytest.approx(1.690, abs=2e-3)
    assert sp["scales: skin and harness (s^2)"] == pytest.approx(1.169, abs=2e-3)
    assert sp["scales: head and tail (s^3)"] == pytest.approx(0.718, abs=2e-3)


def test_mass_rules_scale_each_class_by_its_power(ledger):
    a = F.mass_model(0.60, D0, layout=None, bms=False, ledger=ledger)["by_class_kg"]
    b = F.mass_model(0.75, D0, layout=None, bms=False, ledger=ledger)["by_class_kg"]
    s = 0.75 / 0.60
    assert b["scales: structure"] == pytest.approx(a["scales: structure"] * s ** 2.5)
    assert b["scales: skin and harness"] == pytest.approx(a["scales: skin and harness"] * s ** 2)
    assert b["scales: head and tail"] == pytest.approx(a["scales: head and tail"] * s ** 3)
    assert b["battery"] == pytest.approx(a["battery"])
    # the budget's servo step above 67.5 cm: XM540 / XM430 replace the XC430 / XL430 cases
    assert b["servo"] > a["servo"]


def test_drive_swap_changes_mass_by_the_catalog_difference(ledger):
    a = F.mass_model(0.60, D0, layout=None, bms=False, ledger=ledger)
    b = F.mass_model(0.60, F.DRIVE_SETS["D2"], layout=None, bms=False, ledger=ledger)
    drives = 6 * (0.82 - 0.36)  # 4 pitch + 2 hind knees, X4 -> X6
    linkage = 2 * 0.0825 * 1.2088 * (60 / 30 - 1)  # the hind linkages (closed) sized by the knee drive's peak
    assert b["total_kg"] - a["total_kg"] == pytest.approx(drives + linkage, abs=0.01)


# ---------------------------------------------------------------- the build's thermal model through the study
@pytest.mark.parametrize("sd", F.SEEDS)
def test_heat_call_reproduces_the_stored_35_c_figures(demos, sd):
    """k = 1, the build's drives and the stored demo length: the study's call must give the 35 C method A peaks that
    rl/analysis/thermal_ambient.py stored (results/heat35/thermal_ambient_2026-11-27.json)."""
    d = demos[sd]
    stored = json.loads(F.HEAT35_JSON.read_text())["sets"]["demo"]["runs"][f"fore128-{sd}"]["rooms"]["35"]["A"]
    p = F.config_profile(d, 0.60, 1.0, D0)
    h = F.heat(d, p["walk_sq"], p["stand_sq"], D0, d["minutes"])
    for key, v in stored["hottest"].items():
        assert h["by_tc"][key]["peak_c"] == pytest.approx(v["peak_c"], abs=1e-6)
        assert h["by_tc"][key]["joint"] == v["joint"]


def test_heat_call_reproduces_the_stored_25_c_demo_temperatures(demos):
    d = demos["s3"]
    p = F.config_profile(d, 0.60, 1.0, D0)
    h = F.heat(d, p["walk_sq"], p["stand_sq"], D0, d["minutes"], room_c=25.0)
    for jn, v in h["per_joint_peak_c_8min"].items():
        assert v == pytest.approx(d["temperatures"][jn]["tau_480s"]["peak_c"], abs=1e-9)


def test_temperature_depends_on_load_ratio_not_on_kt_or_resistance(demos):
    """Each drive is fitted to its own equilibrium torque, so at the same load ratio an X8 and an X4 heat alike."""
    d = demos["s0"]
    p = F.config_profile(d, 0.60, 1.0, D0)
    allx8 = {a: "X8-150" for a in F.AXES} | {"name": "x"}
    r = F.DRIVES["X8-150"]["eq"] / F.DRIVES["X4-36-L"]["eq"]
    eq0 = np.array([F.drives_of(D0)[F.axis_of(j)]["eq"] for j in p["joints"]])
    scale = (F.DRIVES["X8-150"]["eq"] / eq0) ** 2  # the same ratio of load to each joint's new drive
    h0 = F.heat(d, p["walk_sq"], p["stand_sq"], D0, 60.0)
    h8 = F.heat(d, p["walk_sq"] * scale, p["stand_sq"] * scale, allx8, 60.0)
    assert r > 4
    for jn in h0["per_joint_peak_c_8min"]:
        assert h8["per_joint_peak_c_8min"][jn] == pytest.approx(h0["per_joint_peak_c_8min"][jn], abs=0.02)


def test_energy_reproduces_the_stored_runtime(demos):
    for sd, d in demos.items():
        e = F.energy(d, 0.60, 1.0, F.M60_SIM_KG, D0)
        assert e["runtime_min"] == pytest.approx(d["battery"]["runtime_minutes"], rel=0.005), sd


# ---------------------------------------------------------------- packaging code and cost rule
def test_packaging_code_reproduces_the_as_drawn_clashes():
    v = F.validate_as_drawn()
    assert v["max_abs_err_mm"] < 0.1


def test_as_drawn_layout_fails_c4_and_shows_clash_1():
    lay = F.layout_parts(0.60, D0, "L1", as_drawn=True)
    geo = F.Geometry(lay)
    geo.set_pose({})
    assert geo.dist("hind_knee_drive", "hind_upper_strut") == pytest.approx(-0.0387, abs=1e-4)


def test_cost_reproduces_the_budgets_chosen_build():
    """budget/size_budget_out.txt:96-98: option C, frame by area, full expression, radio stop: hardware $16,776,
    landed $20,105 / $21,240 / $22,149."""
    c = F.cost(0.60, D0, "L1")
    assert c["hardware_usd"] == pytest.approx(16776, abs=1)
    got = sorted(c["landed_usd"].values())
    for g, want in zip(got, (20105, 21240, 22149)):
        assert g == pytest.approx(want, abs=1)


# ---------------------------------------------------------------- criteria logic
def test_c3_clamp_bound_is_0_70_times_k():
    pk = F.peaks(1.0, D0, {"available": False})
    for v in pk["ratio_by_axis"].values():
        assert v == pytest.approx(0.70)
    pk = F.peaks(1.2, F.DRIVE_SETS["D3"], {"available": True, "worst_policy_factor": 0.5})
    assert pk["ratio_by_axis"]["pitch"] == pytest.approx(1.2 * 0.70 * 30 / 120)
    assert pk["supporting_need_ratio_by_axis"]["pitch"] == pytest.approx(0.5 * 1.2 * 0.70 * 30 / 120)


def test_c2_loop_rms_at_60_cm_matches_the_stored_segments(demos):
    d = demos["s3"]
    p = F.config_profile(d, 0.60, 1.0, D0)
    j = p["joints"].index("left_hind_pitch")
    w = d["segments"]["walk_0.25"]["tau_drive_sq_mean"][j]
    st = d["segments"]["stand"]["tau_drive_sq_mean"][j]
    assert p["loop_ratio"][j] == pytest.approx(math.sqrt((90 * w + 30 * st) / 120) / 7.5)


def test_c7_published_values_follow_the_pre_registered_levers():
    pub = F.published_c7()
    mc = pub["Mini Cheetah"]
    assert mc["hip_cont"] == pytest.approx(6.9 / (9 * 9.81 * (0.209 + 0.195)))
    assert mc["knee_cont"] == pytest.approx(10.7 / (9 * 9.81 * 0.195))


def test_peak_need_factor_logic():
    probe = {"force_factors": [1.0, 0.7, 0.5], "push": {"dv_mps": 0.5}, "walkers": {
        "policy a": {"cases": {"x1": {"factor": 1.0, "falls": 0}, "x0.7": {"factor": 0.7, "falls": 0},
                               "x0.5": {"factor": 0.5, "falls": 1}}},
        "policy b": {"cases": {"x1": {"factor": 1.0, "falls": 0}, "x0.7": {"factor": 0.7, "falls": 0},
                               "x0.5": {"factor": 0.5, "falls": 0}}}}}
    n = F.peak_need_factor(probe)
    assert n["per_walker"] == {"policy a": 0.7, "policy b": 0.5}
    assert n["worst_policy_factor"] == 0.7


def test_stored_output_is_consistent_with_the_pass_rule():
    """The committed output: a configuration passes when C1-C5 all pass, and each criterion's pass follows its
    pre-registered line."""
    out = json.loads(F.OUT_JSON.read_text())
    for c in out["configs"]:
        cr = c["criteria"]
        assert cr["C1"]["pass"] == (cr["C1"]["value_c"] <= 90.0)
        assert cr["C2"]["pass"] == (cr["C2"]["value"] <= 0.80)
        assert cr["C3"]["pass"] == (cr["C3"]["value"] <= 0.80)
        assert cr["C5"]["pass"] == (cr["C5"]["value_min"] >= 30.0)
        assert c["passes_C1_C5"] == all(cr[x]["pass"] for x in ("C1", "C2", "C3", "C4", "C5"))
        if c["drive_set"] == "D7":
            assert not c["passes_C1_C5_like_for_like"]
    sup = "passes_C1_C3_C5_and_C4_except_9_11"
    assert set(out["ranked_passing"]) == {f"{c['size_m']:.2f} {c['drive_set']} {c['layout']}"
                                          for c in out["configs"] if c[sup + "_like_for_like"]}
    assert not any(x.split()[1] == "D7" for x in out["ranked_passing"])
    for c in out["configs"]:
        cr = c["criteria"]
        assert c[sup] == (all(cr[x]["pass"] for x in ("C1", "C2", "C3", "C5"))
                          and cr["C4"]["pass_except_9_and_11_changed_after_a_result"])


# ---------------------------------------------------------------- added 28 Nov 2026 after review
def test_no_configuration_passes_c4_as_written():
    """C4 as pre-registered includes removing every known clash; no layout removes #9, so nothing passes C1-C5."""
    out = json.loads(F.OUT_JSON.read_text())
    assert out["passing_as_written"] == []
    for c in out["configs"]:
        assert c["criteria"]["C4"]["pass"] is False
        assert c["criteria"]["C4"]["known_clashes"]["9"][1] is False
        assert c["passes_C1_C5"] is False


def test_known_clash_list_follows_the_packaging_record():
    """Twelve items as the record's table (design/PACKAGING_PUSHROD_60CM.md:33-44): #10 the two hind feet, #11 the neck,
    #12 the axle gap."""
    assert sorted(F.KNOWN_CLASHES) == list(range(1, 13))
    assert "hind feet" in F.KNOWN_CLASHES[10][0] and "neck" in F.KNOWN_CLASHES[11][0]
    assert "axles" in F.KNOWN_CLASHES[12][0]
    assert all(a.startswith("hind_") and b.startswith("Rhind_") for a, b in F.KNOWN_CLASHES[10][1])


@pytest.fixture(scope="module")
def sweep60():
    lay = F.make_layout(0.60, D0, "L1")
    return lay, F.sweep_layout(lay)


def test_legs_against_each_other_reproduce_the_packaging_record(sweep60):
    """At 60 cm (the L1 offsets equal the drawn ones there) the left-against-right and front-against-hind sweeps, with
    the model's foot body and ankle, reproduce the record: the hind soles overlap 8.49 mm at roll -18, pitch +22.4,
    knee -47, ankle -22 (#10); the front soles stay 26.3 mm apart; the front and hind soles overlap 48.0 mm (#9)."""
    _, sw = sweep60
    h = sw["left_right"]["hind"]
    assert h["least_m"] == pytest.approx(-0.00849, abs=1e-4)
    assert h["least_pair"] == "hind_sole | Rhind_sole"
    q = h["least_pose_deg"]
    assert q["hind_roll"] == pytest.approx(-18.0) and q["hind_flexion"] == pytest.approx(-47.0)
    assert q["hind_ankle"] == pytest.approx(-22.0) and q["hind_pitch"] == pytest.approx(22.4, abs=0.5)
    assert sw["left_right"]["fore"]["least_m"] == pytest.approx(0.0263, abs=2e-4)
    assert sw["leg_to_leg"]["least_m"] == pytest.approx(-0.0480, abs=5e-4)


def test_c4_scores_the_record_clash_10_at_60_cm(sweep60):
    lay, sw = sweep60
    c4 = F.score_packaging(lay, sw)
    assert c4["known_clashes"][10]["removed"] is False
    assert c4["pass"] is False and c4["pass_except_9_and_11"] is False


def test_linkage_sizing_reproduces_the_record_and_flags_the_x8():
    """The record's static rod forces (3,142 N X6, 1,885 N X4) and M8 class 8.8 passing; the X8's 150 N m stall gives
    7,855 N, where M8 fails on its rod end (SF 1.64) and M10 is the smallest that passes."""
    x6 = F.size_linkage(F.DRIVES["X6-60"], 1.0, 0.0815)
    x4 = F.size_linkage(F.DRIVES["X4-36-L"], 1.0, 0.098)
    x8 = F.size_linkage(F.DRIVES["X8-150"], 1.0, 0.1318)
    assert x6["rod_force_n"] == pytest.approx(3142, abs=2) and x6["thread"] == "M8"
    assert x4["rod_force_n"] == pytest.approx(1885, abs=2) and x4["thread"] == "M8"
    assert x8["rod_force_n"] == pytest.approx(7855, abs=3)
    assert x8["by_thread"]["M8"]["sf_rod_end"] == pytest.approx(1.64, abs=0.01)
    assert x8["by_thread"]["M8"]["sf_compression"] == pytest.approx(1.52, abs=0.01)
    assert not x8["m8_passes"] and x8["thread"] == "M10"
    assert x6["rod_radius_m"] == pytest.approx(0.0045)


def test_as_drawn_scores_and_the_x4_maker_rows(demos):
    """The 60 cm design as drawn: the stored 98.3 C (s3) and C2 0.914; the X4 refitted on its 10 and 30 rpm maker
    rows gives 84.2 / 79.2 C and 0.857 / 0.806; the even split 0.740."""
    a = F.score_as_drawn(demos)
    assert a["C1_worst_c"] == pytest.approx(98.28, abs=0.01)
    assert a["C2_worst"] == pytest.approx(0.9137, abs=1e-4)
    assert a["x4_rows"]["10 rpm"]["C1_c_worst"] == pytest.approx(84.19, abs=0.05)
    assert a["x4_rows"]["10 rpm"]["C2_worst"] == pytest.approx(0.8566, abs=1e-4)
    assert a["x4_rows"]["30 rpm"]["C1_c_worst"] == pytest.approx(79.24, abs=0.05)
    assert a["even_split_C2_worst"] == pytest.approx(0.7396, abs=1e-4)
    dm = a["deciding_and_mirror"]["s3"]
    assert dm["joint"] == "left_hind_pitch" and dm["mirror_ratio"] == pytest.approx(0.509, abs=1e-3)
    assert F.DRIVES["X4-36-L"]["eq"] == 7.5 and "fit_end_c" not in F.DRIVES["X4-36-L"]


def test_even_split_keeps_each_pairs_copper(demos):
    p = F.config_profile(demos["s3"], 0.60, 1.0, D0)
    ev = F.even_split(p)
    js = p["joints"]
    for j in js:
        i, im = js.index(j), js.index(F.mirror_of(j))
        assert ev["walk_sq"][i] + ev["walk_sq"][im] == pytest.approx(p["walk_sq"][i] + p["walk_sq"][im])
        assert ev["walk_sq"][i] == pytest.approx(ev["walk_sq"][im])


def test_bay_fit_packer():
    box = F.layout_parts(0.60, D0, "L1")["info"]["bay_box_m"]
    assert F.bay_fit(box)["fits"]
    small = F.layout_parts(0.50, D0, "L1")["info"]["bay_box_m"]
    assert not F.bay_fit(small)["fits"]
    assert len(F.bay_item_boxes()) == len(F.BAY_ITEMS) - 1   # the Jetson and its tray are one stack
    assert F.bay_fit((10.0, 10.0, 10.0), allowance=0.0)["fits"]


def test_structure_by_load_leaves_d0_unchanged(ledger):
    for S in (0.50, 0.60, 0.80):
        a = F.mass_model(S, D0, "L1", ledger=ledger)["total_kg"]
        b = F.mass_model(S, D0, "L1", ledger=ledger, struct_by_load=True)["total_kg"]
        assert b == pytest.approx(a, abs=1e-9)
    a = F.mass_model(0.60, F.DRIVE_SETS["D6"], "L1", ledger=ledger)
    b = F.mass_model(0.60, F.DRIVE_SETS["D6"], "L1", ledger=ledger, struct_by_load=True)
    assert b["total_kg"] > a["total_kg"] and b["struct_load_factor"] > 1.0


def test_c7_levers_run_to_the_sole_and_anymal_in_plane():
    assert F.knee_lever_m("hind", 1.0) == pytest.approx(0.1933, abs=2e-4)
    assert F.knee_lever_m("fore", 1.0) == pytest.approx(0.1944, abs=2e-4)
    pub = F.published_c7()
    assert pub["ANYmal B"]["hip_cont"] == pytest.approx(15.0 / (30 * 9.81 * (0.25 + 0.315)))
    assert F.operating_eq(F.DRIVES["X4-36-L"]) == pytest.approx(7.5 * math.sqrt(55 / 70))
    assert F.mirror_of("left_hind_pitch") == "right_hind_pitch" and F.mirror_of("right_fore_roll") == "left_fore_roll"


# ---------------------------------------------------------------- added 28 Nov 2026 after the third review (R3-1)
def test_knee_axles_are_drawn_at_their_sf2_diameter():
    """With sized linkages each knee axle is drawn at the larger of the model's 12.8 mm and its own solid diameter for
    SF 2.0 at the static rod case, so C4 is never scored on an axle that would yield; as drawn it keeps 12.8 mm."""
    for S, key, layout in ((0.50, "D0", "L1"), (0.55, "D8", "L1"), (0.50, "D2", "L2")):
        lay = F.layout_parts(S, F.DRIVE_SETS[key], layout)
        for end, e in lay["info"]["ends"].items():
            ax = e["axle"]
            part = next(p for p in lay["parts"] if p.name == f"{end}_axle")
            assert 2 * part.r == pytest.approx(max(F.AXLE_D_M, ax["axle_d_for_sf2_m"]))
            assert ax["drawn_d_m"] == pytest.approx(2 * part.r)
            assert ax["bending_static_drawn_mpa"] <= F.ROD_PROOF_MPA / F.ROD_SF + 1e-6
    big = F.layout_parts(0.55, F.DRIVE_SETS["D8"], "L1")["info"]["ends"]["hind"]["axle"]
    assert big["drawn_d_m"] > 0.020   # the X8 knee needs about 22 mm
    lay = F.layout_parts(0.60, D0, "L1", as_drawn=True)
    assert all(p.r == pytest.approx(F.AXLE_D_M / 2) for p in lay["parts"] if p.name.endswith("_axle"))


def test_supporting_passes_clear_their_sized_axles():
    """Every supporting pass in the committed JSON has its rods sized and clears 5 mm on its own leg with the axles
    drawn at their SF 2.0 diameter (the rerun after R3-1)."""
    out = json.loads(F.OUT_JSON.read_text())
    cfg = {f"{c['size_m']:.2f} {c['drive_set']} {c['layout']}": c for c in out["configs"]}
    for key in out["ranked_passing"]:
        c4 = cfg[key]["criteria"]["C4"]
        assert c4["linkage_sized"] and c4["value_m"] >= F.C4_GAP_M
        assert c4["pass_except_9_and_11_changed_after_a_result"]
