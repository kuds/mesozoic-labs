"""Tests for the Stage 1 body registry (``designlab/stage1/__init__.py``): Stage 1b's arm K (``STAGE1B_PLAN.md``
section 2) is registered beside the probe's three bodies, and nothing the probe's committed records depend on
changes with it.

Run from hardware/trex_form_study:  python -m pytest tests/test_stage1_bodies.py -q
Fast.  A key check recomputes a committed record's key; it needs the run directory's reference and transition arrays
and the round-two models (both git-ignored: ``python -m designlab.stage1 prepare`` and ``python -m designlab.refine``
regenerate them, or copy them from the main checkout) and is skipped without them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
from designlab import refine  # noqa: E402
from designlab.stage1 import (  # noqa: E402
    BODIES,
    CANDIDATE_IDS,
    PROBE_BODIES,
    SHA256_PREFIX,
    STUDY,
    candidate_id,
    case_dir,
    ledger,
    panels,
    plant,
    refs,
    train,
    tune,
)

RUN = ROOT / "stage1" / "runs" / "2026-11-08"
#: The probe's registry as Stage 1 ran it (``STAGE1_PLAN.md``, DECISION_LOG P16): case directory, plan's prefix.
PROBE = {"60-N-T-10": ("round2/refine/narrow_carpet_60", "993c5ac5"),
         "60-N-H-10": ("round2/refine/narrow_carpet_60", "96ef84c5"),
         "75-M-H-10": ("round2/refine/feet_75", "0517e4d3")}
#: Arm K: body -> its pick's round-two candidate id.
ARM_K = {"60-N-T-10-K": "60-N-T-10", "60-N-H-10-K": "60-N-H-10"}
K_CASE = "round2/refine/narrow_knee_60"


def _have_inputs(body: str) -> bool:
    """The run directory's reference and transition arrays and the round-two models a key check reads."""
    return (RUN / body / "refs" / "nominal.npz").exists() and (case_dir(body) / "models" / f"{body}.xml").exists()


# ------------------------------------------------------------------------------------------------
# The registry
# ------------------------------------------------------------------------------------------------
def test_the_probe_bodies_are_registered_as_stage_1_ran_them():
    assert PROBE_BODIES == tuple(PROBE)
    for body, (case, prefix) in PROBE.items():
        assert BODIES[body] == case and SHA256_PREFIX[body] == prefix
        assert candidate_id(body) == body and case_dir(body) == STUDY / case
    assert set(BODIES) == set(PROBE) | set(ARM_K) and set(SHA256_PREFIX) == set(BODIES)
    with pytest.raises(KeyError):
        candidate_id("60-N-T-10-X")


def test_arm_k_maps_to_the_narrow_knee_case_and_its_picks_ids():
    assert CANDIDATE_IDS == ARM_K
    for body, cid in ARM_K.items():
        assert BODIES[body] == K_CASE and candidate_id(body) == cid and case_dir(body) == STUDY / K_CASE
        rec = plant.record(body)
        assert rec["id"] == cid and rec["ok"] and rec["refine_case"] == "narrow_knee_60"
        assert len(SHA256_PREFIX[body]) == 8 and rec["xml_sha256"].startswith(SHA256_PREFIX[body])
        assert rec["selection"]["joints"]["knee"] == "robstride_06"
        for name in ("clearance", "turning", "envelope", "carpet", "sensors"):  # the screens the plant and refs read
            assert (case_dir(body) / "review" / "screening" / f"{cid}_{name}.json").exists()


def test_the_narrow_knee_case_is_narrow_carpet_60_with_knee_headroom():
    """``refine.CASES['narrow_knee_60']``: narrow_carpet_60's ids, overrides, walks and turn gap with knee_60's 15 %
    knee headroom (ROUND2_REVIEW.md section 7.3)."""
    k, n = refine.CASES["narrow_knee_60"], refine.CASES["narrow_carpet_60"]
    assert {a: v for a, v in k.items() if a not in ("headroom_by_group", "label")} == {
        a: v for a, v in n.items() if a != "label"}
    assert k["headroom_by_group"] == refine.CASES["knee_60"]["headroom_by_group"] == {"knee": 0.85}
    assert k["label"] == "60 cm narrow carpet-ready + 15 % knee headroom"
    assert tuple(sorted(ARM_K.values())) == tuple(sorted(k["ids"]))


def test_ppo_runs_are_judged_over_the_probe_bodies_only(tmp_path):
    """``train.JUDGED_RULE`` judges every run of a mode over the probe's bodies; arm K has no PPO runs, so it does not
    hold the probe's judged checkpoint back."""
    for mode, seeds in (("R3", train.SEEDS), ("R4", train.R4_SEEDS)):
        runs = train.expected_runs(tmp_path, mode)
        assert set(runs) == {f"{b}/{s}" for b in PROBE_BODIES for s in seeds}


@pytest.mark.parametrize("body", sorted(ARM_K))
def test_arm_k_carpet_swing_is_its_picks(body):
    """``refs.CARPET_LIFT``: an arm K body takes its pick's lift, the lowest of the carpet-lift pass that meets the
    target on every pile with the nominal walk within the drives (the same toe clearance as its pick's); no lift of
    the pass keeps the walking clearance gates (the joint ranges), so its carpet reference is outside the rule."""
    cid = ARM_K[body]
    lift, path = refs.CARPET_LIFT[body], refs.CARPET_LIFT_RECORD[body]
    assert lift == refs.CARPET_LIFT[cid] and "narrow_knee_60" in path
    row = next(r for r in json.loads((STUDY / path).read_text()) if r["id"] == cid)
    assert row["source"] == K_CASE and row["lifted"] == "nominal walk only"
    ok = {float(x): all(v["toe"]["nominal"]["meets_carpet_target"].values()) and v["drives_within_limits"]
          for x, v in row["lifts"].items()}
    assert lift == min(x for x, y in ok.items() if y)
    assert not any(v["walking_gates"] for v in row["lifts"].values())
    pick = next(r for r in json.loads((STUDY / refs.CARPET_LIFT_RECORD[cid]).read_text()) if r["id"] == cid)
    for pile, mm in row["lifts"][f"{lift:g}"]["toe"]["nominal"]["above_pile_top_mm"].items():
        assert mm == pytest.approx(pick["lifts"][f"{lift:g}"]["toe"]["nominal"]["above_pile_top_mm"][pile], abs=0.01)
    assert refs.reference_names(body) == refs.reference_names(cid)


# ------------------------------------------------------------------------------------------------
# The probe's records are untouched
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body", sorted(PROBE_BODIES))
def test_the_probe_bodies_plant_inputs_are_their_records_keys(body):
    """Every record key of a probe body holds the sha256 of the round-two files the plant reads
    (``panels.key_context``): the registry still reads the same files.  The model ledger's hash is the main
    checkout's only where the round-two models are present (without them the plant rebuilds the models, with another
    ledger JSON)."""
    stored = json.loads((RUN / body / "eval" / "R0" / "walk" / "nominal.json").read_text())["key"]["context"]
    now = panels.key_context(body)
    have_models = (case_dir(body) / "models" / f"{body}.xml").exists()
    assert set(now["plant_inputs"]) == set(stored["plant_inputs"])
    for k, v in stored["plant_inputs"].items():
        if k != "ledger" or have_models:
            assert now["plant_inputs"][k] == v, (body, k)
    assert now["env"] == stored["env"]
    pj = json.loads((RUN / body / "plant.json").read_text())
    assert plant.verify(body, plant.model_files(body)[0])["record_sha256"] == pj["model"]["record_sha256"]


@pytest.mark.parametrize("body", sorted(PROBE_BODIES))
def test_a_probe_body_panel_record_key_recomputes(body):
    """The key of a committed R0 record (R0 has no tuned gains, so its key names no file path) recomputes from the
    code as it is now with the record's own panel-sources hash.  That hash is the code's (``PANEL_SOURCES_RULE``), so
    any edit of the panels' modules changes it; everything else in the key is the body's."""
    if not _have_inputs(body):
        pytest.skip("the run directory's reference arrays or the round-two models are not here")
    rec = json.loads((RUN / body / "eval" / "R0" / "walk" / "nominal.json").read_text())
    run_dir = RUN / body
    inp, docs = tune.read_inputs(run_dir, panels.panel_files())
    planner = panels.Planner(run_dir, docs["refs.json"]["references"], docs["transitions.json"]["transitions"])
    row = panels.tracker_rows(body, run_dir, ("R0",))["R0"]
    cond = next(c for c in panels.conditions() if c.key == rec["key_name"])
    res = planner.resolve(cond)
    assert panels.record_key(body, row, cond, res, panels.SEEDS, inp, rec["key"]["sources"]) == rec["key"]
    now = panels.record_key(body, row, cond, res, panels.SEEDS, inp, panels.panel_sources()["sha256"])
    assert {k for k in set(now) | set(rec["key"]) if now.get(k) != rec["key"].get(k)} <= {"sources"}


def test_a_probe_body_ledger_key_recomputes():
    """A committed margin record's key (``ledger.Session.key``) recomputes from the code as it is now with the record's
    own panel-sources hash (each panel key it holds recomputed with it)."""
    body, margin = "60-N-T-10", "grip"
    if not _have_inputs(body):
        pytest.skip("the run directory's reference arrays or the round-two models are not here")
    ses = ledger.Session(body, RUN, rows=("R2",), names=(margin,))
    row = ses.row_names[0]
    stored = json.loads(ledger.record_path(RUN, body, row, margin).read_text())
    assert ses.ledger["sha256"] == stored["key"]["ledger"]
    src = stored["panel_sources"]
    ses.panel_keys = {(r, k): ledger._sha(panels.record_key(body, ses.rows[r], c, ses.resolved[k], ledger.SEEDS,
                                                            ses.inputs, src, ses.context))
                      for r in ses.rows for k, c in ses.conds.items()}
    m = next(x for x in ses.margins if x.name == margin)
    assert ses.key(row, m) == stored["key"]


# ------------------------------------------------------------------------------------------------
# Arm K's Stage 1 records (stage1/runs/2026-11-10-speed)
# ------------------------------------------------------------------------------------------------
K_RUN = ROOT / "stage1" / "runs" / "2026-11-10-speed"


@pytest.mark.parametrize("body", sorted(ARM_K))
def test_arm_k_prepare_records(body):
    """plant.json, refs.json and transitions.json of an arm K body: its own model (the narrow_knee_60 record's), the
    decided plant (decisions 11-12), every reference matching the records it names, every transition there."""
    from designlab.stage1 import transitions

    cid = ARM_K[body]
    pj = json.loads((K_RUN / body / "plant.json").read_text())
    rec = plant.record(body)
    assert pj["body"] == body and pj["model"]["matches_record"] and pj["model"]["matches_plan"]
    assert pj["model"]["record_sha256"] == rec["xml_sha256"] and pj["model"]["plan_prefix"] == SHA256_PREFIX[body]
    assert (pj["options"]["ankle_gains"], pj["options"]["ankle_armature"]) == ("joint", "linkage")
    assert pj["mass_kg"] == pytest.approx(rec["metrics"]["mass"])
    rj = json.loads((K_RUN / body / "refs.json").read_text())
    assert rj["body"] == body and rj["matches_records"] and rj["model"]["record_sha256"] == rec["xml_sha256"]
    assert set(rj["references"]) == set(refs.reference_names(body))
    for e in rj["references"].values():
        assert e["record_checks"] and all(c["match"] for c in e["record_checks"])
        for c in e["record_checks"]:
            assert cid in c["record"] and "narrow_knee_60" in c["record"] and body not in c["record"]
    tj = json.loads((K_RUN / body / "transitions.json").read_text())
    assert tj["body"] == body and set(tj["transitions"]) == set(transitions.TRANSITIONS)


@pytest.mark.parametrize("body", sorted(ARM_K))
def test_arm_k_gate_records(body):
    """gates.json of an arm K body: G1-G4 as Stage 1 ran them on its picks (the decided plant and G2 rule, decisions
    11-13; G4 at the decision-16 start with the panel rule, decisions 15-20), each verdict as its rule reads it."""
    from designlab.stage1 import gates

    doc = json.loads((K_RUN / body / "gates.json").read_text())
    assert doc["body"] == body and all(g in doc for g in ("G1", "G2", "G3", "G4")) and doc["same_plant_options"]
    g1, g2, g3, g4 = (doc[g] for g in ("G1", "G2", "G3", "G4"))
    assert all(h["matches_record"] and h["matches_plan"] for h in g1["hashes"].values())
    assert g1["pass"] == bool(all(s["stands"] for s in g1["stand"].values()) and g1["tendon_map"]["matches"]
                              and g1["repeatability"]["ok"] and g1["step_rate"]["ok"])
    assert g2["rule"] == "panel" and g2["decided"]["follows_all"] and g2["design_study_strides_screened_model"] is None
    assert g2["pass"] == (g2["all_match"] and g2["panel_pass"])
    assert g3["pass_"] == (not g3["failed"] and not g3["missing"])
    assert g4["judged_at"] == "start" and g4["rule_name"] == "panel" and g4["decided"]["follows_all"]
    assert set(g4["gated"]) == set(gates.G4_THRESHOLDS)
    assert g4["pass"] == all(g["panel_pass"] for g in g4["gated"].values())
    assert {k for k, g in g4["gated"].items() if not g["panel_pass"]} == set(g4["attribution"])
    assert doc["pass"] == (g1["pass"] and g2["pass"] and g3["pass_"] and g4["pass"])
    for g in (g1, g2, g3, g4):
        assert g["provenance"]["stage1_sources"]["unchanged_during_run"]
