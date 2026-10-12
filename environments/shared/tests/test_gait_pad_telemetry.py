"""Tests for the per-foot pad telemetry (``environments/shared/gait/pad_telemetry.py``).

Synthetic traces first (pure numpy): a two-foot statue whose feet each stand
on a box pad and three digits, edited one defect at a time.  Then every
manifest model: which feet are pad feet, and on a short zero-action rollout
through the floor-truth recorder, that the per-foot gate statistic's max IS
the gate's ``max_sole_cop_fore_aft``, bit for bit, and that a model with no
box sole gets no pad field at all.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from environments.shared.gait.pad_telemetry import (
    PAD_EPISODE_FIELDS,
    PAD_STEP_FIELDS,
    PadFoot,
    PadLayout,
    episode_pad_telemetry,
    pad_step_columns,
)
from environments.shared.gait.recorder import EpisodeTrace
from environments.shared.gait.stance_metrics import episode_stance_metrics
from environments.shared.model_id import ModelId

STEPS = 400
SETTLE = 100
#: Per foot: the pad carries 80 N and each of three digits 5 N (a 15 % digit share, the T. rex statue's order).
PAD_N = 80.0
DIGIT_N = 5.0


def _trace(steps: int = STEPS) -> EpisodeTrace:
    """A two-foot statue: per foot a box pad (support geom 0 / 4) and three digits, the pad CoP centred."""
    feet = np.array([[0.0, -0.15, 0.0], [0.0, 0.15, 0.0]])
    support = np.tile([PAD_N, DIGIT_N, DIGIT_N, DIGIT_N], 2)
    leg = np.full((steps, 2), PAD_N + 3 * DIGIT_N)
    zeros = np.zeros((steps, 2))
    return EpisodeTrace(
        model_id=ModelId("synthetic_biped", "anatomical"),
        dt=0.01,
        frame_skip=5,
        body_weight_n=float(leg[0].sum()),
        labels=("r", "l"),
        support_geom_foot=(0, 0, 0, 0, 1, 1, 1, 1),
        support_geom_names=tuple(f"{s}_{part}" for s in "rl" for part in ("pad", "d2", "d3", "d4")),
        leg_floor_mean=leg.copy(),
        leg_floor_min=leg.copy(),
        leg_floor_max=leg.copy(),
        leg_down_frac=np.ones((steps, 2)),
        support_floor_mean=np.tile(support, (steps, 1)),
        support_loaded_frac=np.ones((steps, 8)),
        offsupport_floor_mean=zeros.copy(),
        nonleg_floor_mean=np.zeros(steps),
        nonleg_floor_max=np.zeros(steps),
        total_floor_mean=leg.sum(axis=1),
        total_floor_max=leg.sum(axis=1),
        feet_airborne_substeps=np.zeros(steps, dtype=np.int64),
        interleg_force_mean=zeros.copy(),
        interleg_substep_frac=np.zeros(steps),
        leg_body_force_mean=zeros.copy(),
        leg_body_substep_frac=np.zeros(steps),
        sole_contacts_mean=np.full((steps, 2), 4.0),
        sole_cop=np.zeros((steps, 2, 2)),
        touch=leg.copy(),
        foot_pos=np.repeat(feet[None], steps, axis=0),
        root_pos=np.tile([0.0, 0.0, 0.5], (steps, 1)),
        root_quat=np.tile([1.0, 0.0, 0.0, 0.0], (steps, 1)),
        joint_qpos=np.zeros((steps, 3)),
        sole_tilt_deg=zeros.copy(),
        sole_roll_deg=zeros.copy(),
        sole_pitch_deg=zeros.copy(),
        sole_corner_lift=zeros.copy(),
        sole_min_height=zeros.copy(),
        action=np.zeros((steps, 4)),
        ctrl=np.zeros((steps, 4)),
        reward=np.ones(steps),
        spawn_foot_pos=feet.copy(),
        spawn_root_pos=np.array([0.0, 0.0, 0.5]),
        spawn_root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        spawn_sole_tilt_deg=np.zeros(2),
        home_sole_tilt_deg=np.zeros(2),
        truncated=True,
    )


LAYOUT = PadLayout(
    feet=(PadFoot(index=0, label="r", digit_support=(1, 2, 3)), PadFoot(index=1, label="l", digit_support=(5, 6, 7)))
)


def _with_cop(trace: EpisodeTrace, foot: int, values: Any) -> EpisodeTrace:
    cop = trace.sole_cop.copy()
    cop[:, foot, 0] = values
    return replace(trace, sole_cop=cop)


def _pads(trace: EpisodeTrace, layout: PadLayout = LAYOUT, settle: int = SETTLE) -> dict[str, float]:
    return episode_pad_telemetry(trace, layout, settle_steps=settle)


def _gate(trace: EpisodeTrace, settle: int = SETTLE) -> float:
    return episode_stance_metrics(trace, settle_steps=settle).max_sole_cop_fore_aft


# ── the signed CoP and the gate statistic, per foot ──────────────────────────


def test_the_statue_reads_a_centred_pad_and_its_digit_share_on_each_foot():
    result = _pads(_trace())
    assert set(result) == set(LAYOUT.episode_keys())
    assert LAYOUT.episode_keys() == tuple(f"{s}_{field}" for s in "rl" for field in PAD_EPISODE_FIELDS)
    for side in "rl":
        assert result[f"{side}_sole_cop_fore_aft"] == 0.0
        assert result[f"{side}_sole_cop_fore_aft_gate"] == 0.0
        assert result[f"{side}_digit_share"] == pytest.approx(3 * DIGIT_N / (PAD_N + 3 * DIGIT_N))


def test_the_sign_says_which_edge_and_which_foot():
    """Run A's posture: the right pad on its toe edge, the left one a little toward its heel."""
    trace = _with_cop(_with_cop(_trace(), 0, 1.0), 1, -0.2)
    result = _pads(trace)
    assert result["r_sole_cop_fore_aft"] == 1.0
    assert result["l_sole_cop_fore_aft"] == pytest.approx(-0.2)
    assert result["r_sole_cop_fore_aft_gate"] == 1.0
    assert result["l_sole_cop_fore_aft_gate"] == pytest.approx(0.2)
    # The gate keeps only the worse foot, unsigned.
    assert _gate(trace) == 1.0


def test_a_pad_rocking_heel_to_toe_reads_centred_signed_and_on_its_edges_in_the_gate_statistic():
    """Why both are kept: the signed mean of a rock is its centre, the gate's |CoP| its edges."""
    rock = np.where(np.arange(STEPS) % 2 == 0, 1.0, -1.0)
    result = _pads(_with_cop(_trace(), 0, rock))
    assert result["r_sole_cop_fore_aft"] == 0.0
    assert result["r_sole_cop_fore_aft_gate"] == 1.0


def test_only_the_window_counts():
    settle_edge = np.where(np.arange(STEPS) < SETTLE, 1.0, 0.25)
    result = _pads(_with_cop(_trace(), 1, settle_edge))
    assert result["l_sole_cop_fore_aft"] == pytest.approx(0.25)
    assert result["l_sole_cop_fore_aft_gate"] == pytest.approx(0.25)


def test_an_unloaded_step_is_left_out_of_the_signed_mean_and_counts_one_in_the_gate_statistic():
    """The gate's two unloaded cases: no sole floor force (NaN CoP), and a sole only grazing the floor."""
    trace = _with_cop(_trace(), 0, 0.4)
    cop = trace.sole_cop.copy()
    cop[SETTLE : SETTLE + 75, 0, 0] = np.nan  # a quarter of the window lifted
    contacts = trace.sole_contacts_mean.copy()
    contacts[SETTLE + 75 : SETTLE + 150, 0] = 0.0  # a quarter grazing (CoP finite, no loaded contact)
    trace = replace(trace, sole_cop=cop, sole_contacts_mean=contacts)
    result = _pads(trace)
    assert result["r_sole_cop_fore_aft"] == pytest.approx(0.4)
    assert result["r_sole_cop_fore_aft_gate"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.4)
    assert result["r_sole_cop_fore_aft_gate"] == _gate(trace)


def test_a_pad_never_loaded_has_no_signed_cop_and_reads_one_in_the_gate_statistic():
    result = _pads(_with_cop(_trace(), 1, np.nan))
    assert math.isnan(result["l_sole_cop_fore_aft"])
    assert result["l_sole_cop_fore_aft_gate"] == 1.0


@pytest.mark.parametrize("seed", range(5))
def test_the_max_of_the_per_foot_gate_statistic_is_the_gates_value_bit_for_bit(seed):
    rng = np.random.default_rng(seed)
    trace = _trace()
    cop = rng.uniform(-1.0, 1.0, size=(STEPS, 2, 2))
    cop[rng.random((STEPS, 2)) < 0.1] = np.nan
    contacts = rng.choice([0.0, 0.4, 2.0, 4.0], size=(STEPS, 2), p=[0.1, 0.1, 0.4, 0.4])
    trace = replace(trace, sole_cop=cop, sole_contacts_mean=contacts)
    result = _pads(trace)
    per_foot = [result["r_sole_cop_fore_aft_gate"], result["l_sole_cop_fore_aft_gate"]]
    assert max(per_foot) == _gate(trace)


def test_a_foot_whose_sole_is_not_a_box_has_no_cop_field_value():
    """The gate reads box soles only (a finite corner lift on every window step); neither does the telemetry."""
    trace = _with_cop(_trace(), 1, 0.9)
    lift = trace.sole_corner_lift.copy()
    lift[:, 1] = np.nan
    result = _pads(replace(trace, sole_corner_lift=lift))
    assert math.isnan(result["l_sole_cop_fore_aft"]) and math.isnan(result["l_sole_cop_fore_aft_gate"])
    assert result["r_sole_cop_fore_aft_gate"] == 0.0
    # The digit share does not depend on the sole's shape.
    assert result["l_digit_share"] == pytest.approx(0.15 / 0.95)


# ── the digit share ──────────────────────────────────────────────────────────


def test_the_digit_share_is_the_digits_window_force_over_the_whole_limbs():
    trace = _trace()
    support = trace.support_floor_mean.copy()
    leg = trace.leg_floor_mean.copy()
    support[SETTLE:, 1:4] = 0.0  # the right digits lift in the window...
    leg[SETTLE:, 0] = PAD_N + 10.0  # ...and a shin carries 10 N off the support set
    result = _pads(replace(trace, support_floor_mean=support, leg_floor_mean=leg))
    assert result["r_digit_share"] == 0.0
    assert result["l_digit_share"] == pytest.approx(15.0 / 95.0)
    # The settle window does not count.
    support = trace.support_floor_mean.copy()
    support[:SETTLE, 5:8] = 0.0
    assert _pads(replace(trace, support_floor_mean=support))["l_digit_share"] == pytest.approx(15.0 / 95.0)


def test_a_foot_with_no_digit_has_no_digit_share_field_and_a_window_with_no_load_reads_nan():
    """No digit declared (the compsognathus robot's sole, a dibothrosuchus pad): the field does not exist, rather
    than a 0 that would read as "the digits carry nothing".  NaN is kept for a field that exists but was not
    measured."""
    no_digits = PadLayout(feet=(PadFoot(0, "r", ()), PadFoot(1, "l", (5, 6, 7))))
    assert no_digits.episode_keys() == (
        "r_sole_cop_fore_aft",
        "r_sole_cop_fore_aft_gate",
        "l_sole_cop_fore_aft",
        "l_sole_cop_fore_aft_gate",
        "l_digit_share",
    )
    assert no_digits.step_keys() == ("r_sole_cop_fore_aft", "l_sole_cop_fore_aft", "l_digit_share")
    result = _pads(_trace(), no_digits)
    assert "r_digit_share" not in result and result["r_sole_cop_fore_aft_gate"] == 0.0
    assert tuple(pad_step_columns(_trace(), no_digits)) == no_digits.step_keys()
    leg = _trace().leg_floor_mean.copy()
    leg[:, 1] = 0.0
    assert math.isnan(_pads(replace(_trace(), leg_floor_mean=leg))["l_digit_share"])


# ── windows, layouts and arguments ───────────────────────────────────────────


def test_an_episode_that_ends_inside_the_settle_window_is_unmeasured():
    result = _pads(_trace(steps=80))
    assert set(result) == set(LAYOUT.episode_keys())
    assert all(math.isnan(value) for value in result.values())


def test_an_empty_layout_has_no_field():
    assert PadLayout().episode_keys() == () and PadLayout().step_keys() == ()
    assert _pads(_trace(), PadLayout()) == {}
    assert pad_step_columns(_trace(), PadLayout()) == {}


@pytest.mark.parametrize("settle_steps", [-1, 1.5, True])
def test_settle_steps_must_be_a_whole_non_negative_count(settle_steps):
    with pytest.raises(ValueError, match="settle_steps"):
        episode_pad_telemetry(_trace(), LAYOUT, settle_steps=settle_steps)


def test_the_step_columns_reproduce_the_episode_values():
    """The replay CSV's columns: the gate statistic over any window is the mean of |value| with a gap read as 1."""
    trace = _with_cop(_trace(), 0, np.linspace(-0.5, 1.0, STEPS))
    cop = trace.sole_cop.copy()
    cop[SETTLE + 10 : SETTLE + 40, 0, 0] = np.nan
    trace = replace(trace, sole_cop=cop)
    columns = pad_step_columns(trace, LAYOUT)
    assert tuple(columns) == LAYOUT.step_keys() == tuple(f"{s}_{field}" for s in "rl" for field in PAD_STEP_FIELDS)
    assert all(column.shape == (STEPS,) for column in columns.values())
    signed = columns["r_sole_cop_fore_aft"][SETTLE:]
    result = _pads(trace)
    assert np.nanmean(signed) == pytest.approx(result["r_sole_cop_fore_aft"])
    assert np.mean(np.where(np.isnan(signed), 1.0, np.abs(signed))) == pytest.approx(result["r_sole_cop_fore_aft_gate"])
    assert np.isnan(columns["r_sole_cop_fore_aft"][SETTLE + 10 : SETTLE + 40]).all()
    assert columns["l_digit_share"] == pytest.approx(np.full(STEPS, 15.0 / 95.0))
    leg = trace.leg_floor_mean.copy()
    leg[5, 1] = 0.0
    assert np.isnan(pad_step_columns(replace(trace, leg_floor_mean=leg), LAYOUT)["l_digit_share"][5])


# ── every manifest model ─────────────────────────────────────────────────────

#: Each model's pad feet and digit count per foot (the box soles of FOOT_SUPPORT; digits = its support less its sole).
PAD_FEET = {
    ModelId("tyrannosaurus_rex", "anatomical"): {"r": 3, "l": 3},
    ModelId("compsognathus_longipes", "anatomical"): {"r": 3, "l": 3},
    ModelId("compsognathus_longipes", "robot"): {"r": 0, "l": 0},
    ModelId("dibothrosuchus_elaphros", "anatomical"): {"fr": 0, "fl": 0, "rr": 0, "rl": 0},
    ModelId("velociraptor_mongoliensis", "anatomical"): {},
    ModelId("brachiosaurus_altithorax", "anatomical"): {},
}


def test_the_table_covers_every_manifest_model():
    from environments.shared.species_names import model_ids

    assert set(PAD_FEET) == set(model_ids())


def _rollout(model_id: ModelId, steps: int) -> tuple[EpisodeTrace, Any]:
    """One zero-action episode of *model_id*'s stance env through the floor-truth recorder, cut at *steps*."""
    pytest.importorskip("mujoco")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from environments.shared.config import load_stage_config
    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.recorder import SubstepContactRecorder
    from environments.shared.species_registry import env_class

    env = env_class(model_id)(**load_stage_config(model_id, "stance")["env_kwargs"])
    try:
        morphology = Morphology.from_env(env, model_id)
        with SubstepContactRecorder(env, morphology) as recorder:
            env.reset(seed=3042)
            action = np.zeros(env.action_space.shape)
            for _ in range(steps):
                _obs, reward, terminated, truncated, _info = env.step(action)
                recorder.end_step(action, float(reward))
                if terminated or truncated:
                    break
            trace = recorder.end_episode(terminated=bool(terminated), truncated=bool(truncated))
    finally:
        env.close()
    return trace, morphology


@pytest.mark.parametrize("model_id", sorted(PAD_FEET), ids=str)
def test_every_model_reads_its_pads_as_the_gate_does(model_id):
    """The layout follows FOOT_SUPPORT, and the per-foot gate statistic's max is the gate's own value."""
    trace, morphology = _rollout(model_id, steps=160)
    layout = PadLayout.from_morphology(morphology)
    assert {foot.label: len(foot.digit_support) for foot in layout.feet} == PAD_FEET[model_id]
    for foot in layout.feet:
        names = [trace.support_geom_names[index] for index in foot.digit_support]
        assert all(name.startswith(f"{foot.label}_") and "toe" in name for name in names), names
    settle = 100
    result = episode_pad_telemetry(trace, layout, settle_steps=settle)
    gate = episode_stance_metrics(trace, settle_steps=settle).max_sole_cop_fore_aft
    if not layout.feet:
        assert result == {} and pad_step_columns(trace, layout) == {}
        assert math.isnan(gate)  # no box sole: the gate's fore-aft CoP is unmeasured too
        return
    per_foot = [result[f"{foot.label}_sole_cop_fore_aft_gate"] for foot in layout.feet]
    assert max(per_foot) == gate
    for foot in layout.feet:
        signed = result[f"{foot.label}_sole_cop_fore_aft"]
        assert -1.0 <= signed <= 1.0
        assert abs(signed) <= result[f"{foot.label}_sole_cop_fore_aft_gate"] + 1e-12
        if foot.digit_support:
            assert 0.0 < result[f"{foot.label}_digit_share"] < 1.0
        else:
            assert f"{foot.label}_digit_share" not in result
