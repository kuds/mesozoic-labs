"""Tests for the Stage 2 look ladder's Stage 0 screen (designlab/stage2/screen.py; STAGE2_PLAN.md section 4 A.2-A.3,
research_look.md 4.1 item 2).

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_screen.py -q

1. The readings on known inputs: the passive neck and tail loads equal ``look._passive_ratios``; the accelerations of a
   known periodic CoM path; Stage 0's sway reproduces Stage 1's G7 figures; round two's straight-run friction reading
   (``straight_run``) reproduces the turning screen's recorded straight run of today's nominal walk on both bodies.
2. The committed screen (``stage2/look/screen.json``): every rung of the ladder screened from its own arrays with its
   price equal to the recorded one; G2 bit-identical on the code that ran it; every drive ratio over 1 listed with
   the plan's rule and the rung's consequence, near and passive lists consistent with the ratios; the tail-phase
   check's findings and controls; every block under a named source; SCREEN.md regenerates from screen.json.

Gate G2 itself (all ten Stage 1 references of both bodies, bit for bit) is ``test_stage2_look.py``'s.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco is imported)
from designlab import sizing, statics, turning  # noqa: E402
from designlab.stage2 import look, screen  # noqa: E402

LOOK = ROOT / "stage2" / "look"
LADDER = json.loads((LOOK / "look_refs.json").read_text()) if (LOOK / "look_refs.json").exists() else None
SCREEN = json.loads((LOOK / screen.SCREEN_JSON).read_text()) if (LOOK / screen.SCREEN_JSON).exists() else None
needs_screen = pytest.mark.skipif(SCREEN is None or LADDER is None, reason="stage2/look/screen.json not generated")
#: Stage 1's G7 figures (STAGE1_PLAN section 3 step 8): Stage 0's sway of the nominal walk, H, 3 decimals.
G7_SWAY = {"60-N-T-10": 0.168, "75-M-H-10": 0.164}


@pytest.fixture(scope="module")
def src60():
    return look.source("60-N-T-10")


# ------------------------------------------------------------------------------------------------
# 1. The readings
# ------------------------------------------------------------------------------------------------
def test_passive_loads_equal_look_and_add_damping_and_friction(src60):
    """tau + b omega + f sign(omega) on the neck and tail: the ratios are ``look._passive_ratios``'; with the joint at
    rest the passive terms vanish; RMS of the passive terms alone is b times the RMS speed when f is zero."""
    M, design = src60["M"], src60["design"]
    n, na = 400, len(M.names)
    rng = np.random.default_rng(3)
    tau, omega = rng.normal(0, 0.3, (n, na)), rng.normal(0, 1.0, (n, na))
    p = screen.passive_loads(M, design, tau, omega)
    ref = look._passive_ratios(M, design, tau, omega)
    for g in ("neck", "tail"):
        assert p[g]["cont_ratio_total"] == pytest.approx(ref[g]["cont_ratio"], rel=1e-12)
        assert p[g]["peak_ratio_total"] == pytest.approx(ref[g]["peak_ratio"], rel=1e-12)
        assert p[g]["damping"] == 0.2 and p[g]["frictionloss"] == 0.15  # the MJCF default (research Bottom line 3)
        i = M.names.index(p[g]["joint"])
        assert p[g]["rms_id_nm"] == pytest.approx(float(np.sqrt(np.mean(tau[:, i] ** 2))))
        assert p[g]["cont_ratio_id"] == pytest.approx(p[g]["rms_id_nm"] / design.act(g).sustained)
    still = screen.passive_loads(M, design, tau, np.zeros_like(omega))
    for g in ("neck", "tail"):
        assert still[g]["rms_total_nm"] == pytest.approx(still[g]["rms_id_nm"]) and still[g]["rms_passive_alone_nm"] == 0


def test_accelerations_of_a_known_periodic_path():
    """A CoM path x = v t + a sin(w t), y = b sin(w t), z = z0 + c sin(2 w t) over one period: the second differences'
    peaks, the forward part's 1 Hz low pass of a 3 Hz wobble, and still feet."""
    n, T = 400, 1.0 / 3.0
    dt = T / n
    t = np.arange(n) * dt
    w = 2 * np.pi / T
    v, a, b, c = 0.3, 0.002, 0.01, 0.003
    com = np.column_stack([v * t + a * np.sin(w * t), b * np.sin(w * t), 0.4 + c * np.sin(2 * w * t)])
    arrs = dict(dt=np.array(dt), com=com, stride_motion=np.array([0, 0, 0, v * T, 0]), feet=np.zeros((n, 2, 3)))
    acc = screen.accelerations(arrs)
    assert acc["com_forward_peak"] == pytest.approx(a * w * w, rel=1e-3)
    assert acc["com_lateral_peak"] == pytest.approx(b * w * w, rel=1e-3)
    assert acc["com_horizontal_peak"] == pytest.approx(math.hypot(a, b) * w * w, rel=1e-3)
    assert acc["com_vertical_max"] == pytest.approx(c * 4 * w * w, rel=1e-3)
    assert acc["com_forward_lp1hz_peak"] < 0.05 * acc["com_forward_peak"]  # 3 Hz through a 1 Hz 2nd-order filtfilt
    assert acc["foot_vertical_peak"] == 0.0


@needs_screen
def test_stage0_sway_reproduces_stage1_g7():
    """Stage 0's sway (2 x sway_amp / H) of V0 at the nominal walk is Stage 1's G7 figure on both bodies."""
    for b, sway in G7_SWAY.items():
        assert round(SCREEN["rungs"][b]["nominal"]["V0"]["sway"]["stage0_pp_H"], 3) == sway


@pytest.mark.skipif(LADDER is None, reason="stage2/look/look_refs.json not generated")
def test_straight_run_reproduces_the_turning_screens_record(src60):
    """``straight_run`` on today's nominal walk (V0 regenerated) is the turning screen's recorded straight run, field by
    field (round two's ``turning.screen_model``: ``turning.generate`` at yaw 0, ``turning.summarize``)."""
    from designlab.stage1 import candidate_id, case_dir

    spec = look.base_spec(src60["H"], "nominal")
    g = look.generate(src60, spec)
    r = screen.straight_run(src60, g, g["look"]["rolling"], g["look"]["phase"])
    rec = json.loads((case_dir("60-N-T-10") / "review" / "screening" / f"{candidate_id('60-N-T-10')}_turning.json")
                     .read_text())["runs"]["straight"]
    cmp = screen._record_compare(r, rec)
    assert cmp["fields"] > 20 and cmp["all_max_rel_diff"] < 1e-9 and cmp["friction_max_rel_diff"] < 1e-9
    assert not cmp["differing"] and cmp["frame_rule_note"] is None
    ss = r["single_support"]
    assert ss["mu_req_max"] == pytest.approx(max(rec["feet"][s]["single_support"]["mu_req_max"] for s in statics.SIDES))
    assert r["double_support"]["read"] == r["double_support"]["frames"] > 0  # no foot rolls in today's walk
    assert r["grips"] == {"0.9": ss["mu_req_max"] <= 0.9, "0.54": ss["mu_req_max"] <= 0.54}


# ------------------------------------------------------------------------------------------------
# 2. The committed screen
# ------------------------------------------------------------------------------------------------
ENTRIES = [] if LADDER is None else [(b, s, v) for b, d in LADDER["bodies"].items() for s, e in d.items() for v in e]


@needs_screen
def test_every_rung_is_screened_from_its_own_arrays():
    assert {(b, s, v) for b, d in SCREEN["rungs"].items() for s, e in d.items() for v in e} == set(ENTRIES)
    assert SCREEN["inputs"]["arrays_checked"] == len(ENTRIES)
    for b, s, v in ENTRIES:
        r, e = SCREEN["rungs"][b][s][v], LADDER["bodies"][b][s][v]
        assert r["arrays"]["sha256"] == e["sha256"] and r["arrays"]["file"] == e["file"], (b, s, v)
        assert r["reprice"]["equal_to_recorded"], (b, s, v, r["reprice"])
        assert r["drives"] == e["price"]["drives"] and r["over"] == e["price"]["over"], (b, s, v)
        assert r["features"] == e["features"] and r["flags"] == e["flags"]


@needs_screen
def test_g2_on_the_code_that_ran_the_screen():
    """G2: Stage 1's nominal, brisk and fr05 references of both bodies regenerate bit for bit (screen.g2)."""
    for b in look.BODIES:
        g = SCREEN["g2"][b]
        assert g["passes"] and g["accuracy"] == "r2" and set(g["references"]) == set(screen.G2_NAMES), b
        for nm, x in g["references"].items():
            assert x["bit_identical"] and x["sha256"] == x["recorded"] and x["record_leaves_match"], (b, nm)


@needs_screen
def test_over_limit_rungs_follow_the_plans_rule():
    """Every drive ratio over 1 is listed; every listed rung states the plan's consequence (it keeps its features and
    loses that speed rung unless the owner decides otherwise) and, for a speed limit, its cause; near and passive
    lists follow the ratios."""
    listed = {(o["body"], o["speed"], o["variant"]) for o in SCREEN["over_limit"]}
    for b, s, v in ENTRIES:
        r = SCREEN["rungs"][b][s][v]
        over = {f"{g} {k.replace('_ratio', '')}" for g, x in r["drives"].items() for k in screen.KINDS if x[k] > 1.0}
        assert over == {o["limit"] for o in r["over"] if o["limit"].split()[0] in sizing.GROUPS}, (b, s, v)
        assert ((b, s, v) in listed) == (not r["within_limits"]) == bool(r["over"])
        if r["over"]:
            assert "loses" in r["rung"] and screen.SPEED_LABEL[s] in r["rung"] and "unless the owner" in r["rung"]
            assert all(f in r["rung"] for f in r["features"])
            for o in r["over"]:
                g, kind = o["limit"].split()[:2]
                if kind == "speed" and g in sizing.JOINT_OF:
                    c = r["cause"][o["limit"]]
                    assert c["peak_rad_s"] == pytest.approx(max(c["peak_rad_s_range"]))
        else:
            assert r["rung"].startswith("within")
        near = {(o["limit"], o["value"]) for o in r["near"]}
        assert near == {(f"{g} {k.replace('_ratio', '')}", x[k]) for g, x in r["drives"].items() for k in screen.KINDS
                        if 1.0 - screen.NEAR_BAND < x[k] <= 1.0}
        po = {o["value"] for o in r["over_with_passive"]}
        assert po == {r["passive"][g][k] for g in ("neck", "tail") for k in ("cont_ratio_total", "peak_ratio_total")
                      if r["passive"][g][k] > 1.0}
        assert r["passive"]["tail"]["cont_ratio_id"] == pytest.approx(r["drives"]["tail"]["cont_ratio"], rel=1e-12)
        assert r["passive"]["neck"]["cont_ratio_id"] == pytest.approx(r["drives"]["neck"]["cont_ratio"], rel=1e-12)


@needs_screen
def test_tail_phase_check():
    """Research Bottom line 4 on round two's straight-run reading: today's tail raises the stance pad's peak yaw moment
    over the tail-less V0, F9's phase lowers it; the record check and the look-path control hold; the rung readings
    carry the ladder's F9 phase."""
    tdoc = json.loads((LOOK / "tail_phase.json").read_text())
    for b in look.BODIES:
        for s in look.SPEEDS:
            x = SCREEN["tail_phase"][b][s]
            assert x["f9_phase_rad"] == tdoc["bodies"][b][s]["tail_phase_F9_rad"]
            assert x["today_adds_to_leg_yaw_moment"] and x["f9_phase_lowers_peak_yaw_moment"], (b, s)
            w = x["walks"]
            assert w["V0"]["tail"] == dict(amp_rad=0.12, phase_rad=math.pi / 2)
            assert w["V5"]["tail"]["phase_rad"] == x["f9_phase_rad"] and w["V0_no_tail"]["tail"]["amp_rad"] == 0.0
            assert x["look_path_control"]["joint_max_diff_rad"] < 1e-9 and x["look_path_control"]["mz_abs_max_diff_nm"] < 1e-9
            c = x["changes"]["V0_phase_today_to_F9"]["mz_abs_max_nm"]
            assert c["rel"] < 0 and c["abs"] == pytest.approx(w["V0_F9_phase"]["single_support"]["mz_abs_max_nm"]
                                                               - w["V0"]["single_support"]["mz_abs_max_nm"])
            # the rung readings are the screen's own friction readings of those rungs
            for v in ("V0", "V4", "V5", "V6"):
                assert w[v]["single_support"] == SCREEN["rungs"][b][s][v]["friction"]["single_support"], (b, s, v)
        rc = SCREEN["tail_phase"][b]["nominal"]["record_check"]
        for key in ("straight", "straight_brisk"):
            c = rc[key]
            assert c["friction_max_rel_diff"] < 1e-9, (b, key)  # the friction need, as round two recorded it
            # gait.generate and turning.generate may classify one touchdown frame differently (turning, Events): the
            # single-support counts then differ by one, and only fields that frame enters differ
            assert all(abs(m - r) <= 1 for m, r in c["single_support_frames"].values()), (b, key)
            assert (c["frame_rule_note"] is None) == all(m == r for m, r in c["single_support_frames"].values())
            assert all(x["field"] not in screen.FRICTION_FIELDS for x in c["differing"]), (b, key, c["differing"])
            assert not c["differing"] or c["frame_rule_note"], (b, key)


@needs_screen
def test_readings_are_consistent():
    """Grips follow the friction need; no rolling foot in single support; V0's changes against itself are zero; the
    double-support reading counts the rolling frames it leaves out."""
    for b, s, v in ENTRIES:
        r = SCREEN["rungs"][b][s][v]
        f = r["friction"]
        mu = f["single_support"]["mu_req_max"]
        assert f["grips"] == {f"{g:g}": mu <= g for g in screen.GRIPS}
        ds = f["double_support"]
        assert ds["read"] + ds["rolling_frames_not_read"] == ds["frames"]
        assert (ds["rolling_frames_not_read"] > 0) == bool(LADDER["bodies"][b][s][v]["spec"]["toe_off"])
        assert (r["plain_gait_id"] is not None) == bool(LADDER["bodies"][b][s][v]["spec"]["toe_off"])
        assert r["sway"]["com_pp_per_stride_max_H"] <= r["sway"]["com_pp_H"] + 1e-12
        assert r["clearance"]["walking_gates"] is True and r["summary"]["ik_ok"]
        if v == "V0":
            assert all(d is None or d == 0 for _, _, d in r["vs_V0"].values())
    for blk in ("drives", "passive", "summary", "contact", "friction", "sway", "accel", "clearance", "plain_gait_id",
                "over", "near", "over_with_passive", "rung", "cause", "look", "vs_V0"):
        assert blk in screen.SOURCES and blk in SCREEN["sources"], blk


@needs_screen
def test_screen_md_regenerates_from_screen_json():
    assert (LOOK / screen.SCREEN_MD).read_text() == screen.markdown(SCREEN)
    assert SCREEN["provenance"]["git"]["dirty"] is False  # the committed screen ran on a clean tree
    assert turning.MU == 0.8 and SCREEN["margins"]["peak"] == sizing.PEAK_MARGIN == 1.25
    assert SCREEN["margins"]["speed"] == sizing.SPEED_MARGIN == 1.15
