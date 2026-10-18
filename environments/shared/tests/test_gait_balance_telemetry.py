"""Tests for the recorder's opt-in balance capture and :mod:`environments.shared.gait.balance_telemetry`.

On real envs (no SB3): the capture is inert (the trajectory, the
observations, the rewards and every pre-existing trace field are bit-identical
with and without it), the floor forces it decodes are ``mj_contactForce``'s
rotated to the world, the statue's and a scripted nosedive's readings sit in
the bands the scripted probe of 2026-10-17 measured, each key exists only
where the model has what it measures, and a failure in the capture or its
reduction costs the balance fields alone.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from environments.shared.config import load_stage_config
from environments.shared.eval_diagnostics import sagittal_leg_dc_pairs
from environments.shared.gait import balance_telemetry
from environments.shared.gait.balance_telemetry import (
    FALL_ONLY_KEYS,
    GENERIC_BALANCE_KEYS,
    SPAN_BALANCE_KEYS,
    TIP_BALANCE_KEYS,
    BalanceLayout,
    episode_actuator_force_fraction,
    episode_balance_telemetry,
    episode_force_telemetry,
    heading_frames,
    hof_split,
    mirror_even_key,
)
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import EpisodeTrace, SubstepContactRecorder
from environments.shared.heading_observation import body_to_world, world_to_heading
from environments.shared.model_id import ModelId
from environments.shared.species_registry import get_species_config

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")
DIBOTHROSUCHUS_ELAPHROS = ModelId("dibothrosuchus_elaphros", "anatomical")
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")

#: The models whose feet have a box sole: the ones the evaluation attaches the recorder to.
BOX_SOLED = (TYRANNOSAURUS_REX, COMPSOGNATHUS_LONGIPES, COMPSOGNATHUS_LONGIPES_ROBOT, DIBOTHROSUCHUS_ELAPHROS)
#: The fields the balance capture adds to the trace.
BALANCE_FIELDS = (
    "com_pos",
    "spawn_com_pos",
    "actuator_force",
    "ankle_torque",
    "floor_force",
    "floor_cop",
    "floor_mu",
    "support_line_pos",
    "digit_tip_pos",
    "digit_tip_foot",
)
#: The per-foot rail suffixes ``progress_report._feet_of`` and ``PostureDriftWatchCallback._feet`` read as a foot.
RAIL_SUFFIXES = (
    "_sole_cop_fore_aft",
    "_sole_cop_fore_aft_gate",
    "_foot_cop_fore_aft",
    "_foot_cop_fore_aft_gate",
    "_sole_tilt_deg",
    "_sole_corner_lift_m",
    "_digit_share",
    "_toe_margin_nm",
    "_heel_margin_nm",
)
#: The scripted probe's settle window: its windows start at step 200, the T. rex stance's settle_steps.
SETTLE = 200


def make_env(model_id: ModelId, **overrides):
    kwargs = {**dict(load_stage_config(model_id, "stance")["env_kwargs"]), **overrides}
    return get_species_config(model_id).env_class(**kwargs)


@pytest.fixture
def envs():
    built = []

    def factory(model_id: ModelId, **overrides):
        env = make_env(model_id, **overrides)
        built.append(env)
        return env

    yield factory
    for env in built:
        env.close()


def _layout(env, model_id: ModelId) -> tuple[Morphology, BalanceLayout]:
    morphology = Morphology.from_env(env, model_id)
    return morphology, BalanceLayout.from_env(env, morphology, sagittal_leg_dc_pairs(env, morphology))


def _ankle_action(env, morphology, offset: float, step: int) -> np.ndarray:
    """The probe's posture: a mirror-symmetric ankle offset ramped in over steps 200-300, then held."""
    action = np.zeros(env.model.nu)
    for pair in sagittal_leg_dc_pairs(env, morphology):
        if pair.key.endswith("_ankle_deg"):
            action[pair.left] = pair.left_sign * offset
            action[pair.right] = pair.right_sign * offset
    return float(np.clip((step - 200) / 100.0, 0.0, 1.0)) * action


def _episode(env, model_id, *, steps, seed=3042, offset=0.0, balance=True, morphology=None, actions=None):
    """One recorded episode from ``reset(seed)``; returns the trace and the per-step (obs, reward, qpos, qvel)."""
    morphology = morphology or Morphology.from_env(env, model_id)
    recorder = SubstepContactRecorder(env, morphology, balance=balance).attach()
    rows = []
    try:
        env.reset(seed=seed)
        terminated = truncated = False
        for step in range(steps):
            action = _ankle_action(env, morphology, offset, step) if actions is None else actions[step]
            obs, reward, terminated, truncated, _info = env.step(action)
            recorder.end_step(action, reward)
            rows.append((obs.copy(), float(reward), env.data.qpos.copy(), env.data.qvel.copy()))
            if terminated or truncated:
                break
        trace = recorder.end_episode(terminated=terminated, truncated=truncated)
    finally:
        recorder.detach()
    return trace, rows


# ── inert ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_id", BOX_SOLED, ids=str)
def test_the_capture_leaves_the_episode_and_every_existing_field_bit_identical(envs, model_id):
    env = envs(model_id)
    actions = np.random.default_rng(5).uniform(-0.2, 0.2, size=(80, env.model.nu))
    unrecorded = []
    env.reset(seed=3042)
    for action in actions:
        obs, reward, terminated, truncated, _info = env.step(action)
        unrecorded.append((obs.copy(), float(reward), env.data.qpos.copy(), env.data.qvel.copy()))
        if terminated or truncated:
            break
    plain, plain_rows = _episode(env, model_id, steps=80, balance=False, actions=actions)
    traced, traced_rows = _episode(env, model_id, steps=80, balance=True, actions=actions)
    for rows in (plain_rows, traced_rows):
        for (obs, reward, qpos, qvel), (obs0, reward0, qpos0, qvel0) in zip(rows, unrecorded, strict=True):
            np.testing.assert_array_equal(obs, obs0)
            assert reward == reward0
            np.testing.assert_array_equal(qpos, qpos0)
            np.testing.assert_array_equal(qvel, qvel0)
    for name in EpisodeTrace.__dataclass_fields__:
        before, after = getattr(plain, name), getattr(traced, name)
        if name in BALANCE_FIELDS:
            assert before is None and after is not None, name
        elif isinstance(before, np.ndarray):
            np.testing.assert_array_equal(before, after, err_msg=name)
        else:
            assert before == after, name


# ── the floor forces are mj_contactForce's ───────────────────────────────────


class BoundaryForces:
    """A substep hook: each limb's (and the rest of the animal's) floor force, from ``mj_contactForce``."""

    def __init__(self, env, morphology: Morphology) -> None:
        self.env, self.morphology = env, morphology
        self.substeps: list[np.ndarray] = []
        self.ankle: list[np.ndarray] = []

    def __call__(self) -> None:
        model, data, morphology = self.env.model, self.env.data, self.morphology
        force = np.zeros((morphology.n_feet + 1, 3))
        local = np.zeros(6)
        for index in range(data.ncon):
            first, second = (int(geom) for geom in data.contact.geom[index])
            if morphology.is_floor[first] == morphology.is_floor[second]:
                continue
            other = second if morphology.is_floor[first] else first
            if not morphology.is_animal[other]:
                continue
            mujoco.mj_contactForce(model, data, index, local)
            world = data.contact.frame[index].reshape(3, 3).T @ local[:3]
            limb = int(morphology.geom_foot[other])
            force[limb if limb >= 0 else morphology.n_feet] += world if morphology.is_floor[first] else -world
        self.substeps.append(force)
        self.ankle.append(
            np.array(
                [
                    data.qfrc_actuator[model.jnt_dofadr[foot.ankle_joint]]
                    + data.qfrc_passive[model.jnt_dofadr[foot.ankle_joint]]
                    if foot.ankle_joint >= 0
                    else np.nan
                    for foot in morphology.feet
                ]
            )
        )


@pytest.mark.parametrize("model_id", BOX_SOLED, ids=str)
def test_the_floor_forces_are_mj_contact_force_on_the_animal_and_carry_its_weight(envs, model_id):
    env = envs(model_id)
    morphology = Morphology.from_env(env, model_id)
    reference = BoundaryForces(env, morphology)
    env._substep_probe_hook = reference  # the recorder chains it: both read every substep
    trace, _rows = _episode(env, model_id, steps=300, morphology=morphology, actions=np.zeros((300, env.model.nu)))
    frame_skip = morphology.frame_skip
    boundary = np.array(reference.substeps)[frame_skip - 1 :: frame_skip]
    assert trace.floor_force is not None and trace.floor_force.shape == boundary.shape
    scale = morphology.body_weight_n
    np.testing.assert_allclose(trace.floor_force, boundary, rtol=0, atol=1e-9 * scale)
    # Contact normals on the plane are vertical: each limb's F_z is its boundary substep's normal force.
    assert np.all(trace.floor_force[:, : morphology.n_feet, 2] >= -1e-9 * scale)
    carried = trace.floor_force[200:300].sum(axis=1)[:, 2].mean()
    assert carried == pytest.approx(scale, rel=0.01)
    ankles = np.array(reference.ankle)[frame_skip - 1 :: frame_skip]
    np.testing.assert_array_equal(trace.ankle_torque, ankles)


# ── the statue and a scripted nosedive (the T. rex, the probe's postures) ────


@pytest.fixture(scope="module")
def statue_episode():
    env = make_env(TYRANNOSAURUS_REX)
    try:
        morphology, layout = _layout(env, TYRANNOSAURUS_REX)
        trace, _rows = _episode(env, TYRANNOSAURUS_REX, steps=600, morphology=morphology)
    finally:
        env.close()
    return trace, layout


@pytest.fixture(scope="module")
def nosedive_episode():
    env = make_env(TYRANNOSAURUS_REX)
    try:
        morphology, layout = _layout(env, TYRANNOSAURUS_REX)
        trace, _rows = _episode(env, TYRANNOSAURUS_REX, steps=600, morphology=morphology, offset=-0.05)
    finally:
        env.close()
    return trace, layout


def test_the_statue_stands_inside_its_toe_line_with_a_quiet_hof_split_and_little_of_its_friction_cone(statue_episode):
    trace, layout = statue_episode
    assert trace.length == 600 and not trace.terminated
    balance = episode_balance_telemetry(trace, layout, settle_steps=SETTLE)
    forces = episode_force_telemetry(trace, layout, settle_steps=SETTLE)
    assert math.isfinite(balance["com_fore_aft_mean_m"])
    assert balance["xcom_toe_margin_min_m"] > 0.0 and balance["xcom_past_toe_fraction"] == 0.0
    assert math.isnan(balance["xcom_toe_lead_s"])  # it did not fall
    assert abs(balance["hof_cop_term_mean_mps2"]) < 0.05 and abs(balance["hof_counter_rotation_mean_mps2"]) < 0.05
    assert balance["xcom_tip_margin_min_m"] > balance["xcom_toe_margin_min_m"]  # the digit tips lie past the pad
    for label in layout.labels:
        assert forces[f"{label}_cone_use_mean"] < 0.1  # the probe: 0.042-0.056
        # The ankle torque over the foot's load is the whole-foot CoP's lever ahead of the ankle (probe: 0.144 m).
        window = slice(SETTLE, trace.length)
        index = layout.labels.index(label)
        load = trace.leg_floor_mean[window, index]
        lever = float(np.nansum(trace.foot_cop[window, index] * load) / load.sum())
        assert forces[f"{label}_ankle_moment_arm_m"] == pytest.approx(lever, abs=0.03)
        assert forces[f"{label}_ankle_moment_arm_m"] > 0.1  # + ahead of the ankle
    assert 0.0 < forces["actuator_force_frac_max"] < 1.0 and forces["actuator_force_limited_fraction"] == 0.0


def test_a_scripted_nosedive_shows_the_xcom_past_the_toe_line_well_before_the_fall(nosedive_episode):
    trace, layout = nosedive_episode
    assert trace.terminated and not trace.truncated and trace.length < 600  # the probe: terminated at step 500
    balance = episode_balance_telemetry(trace, layout, settle_steps=SETTLE)
    forces = episode_force_telemetry(trace, layout, settle_steps=SETTLE)
    assert 1.0 <= balance["xcom_toe_lead_s"] <= 2.5  # the probe: 1.64 s against the ankle-anchored toe line
    assert balance["xcom_past_toe_fraction"] > 0.0 and balance["xcom_toe_margin_min_m"] < 0.0
    assert balance["time_to_toe_min_s"] < 1.0
    assert max(forces[f"{label}_cone_use_max"] for label in layout.labels) > 0.15  # the probe: 0.236
    # The same episode cut by the time limit is not a fall: no lead time.
    assert math.isnan(
        episode_balance_telemetry(replace(trace, truncated=True), layout, settle_steps=SETTLE)["xcom_toe_lead_s"]
    )


def test_hofs_split_adds_up_to_the_forward_acceleration(nosedive_episode):
    trace, layout = nosedive_episode
    acceleration, cop_term, counter_rotation = hof_split(trace, layout)
    finite = np.isfinite(acceleration)
    assert finite[SETTLE:].all()
    np.testing.assert_allclose((cop_term + counter_rotation)[finite], acceleration[finite], rtol=0, atol=1e-12)
    forward, _pitch = heading_frames(trace.root_quat)
    total = trace.floor_force.sum(axis=1)
    np.testing.assert_allclose(acceleration, np.einsum("tj,tj->t", forward, total[:, :2]) / layout.mass_kg, rtol=1e-12)


def test_the_first_samples_velocity_spans_frame_skip_minus_one_substeps_from_the_spawn(statue_episode):
    """The spawn sample is the episode's first substep and the first boundary sample its last: (fs - 1) substeps."""
    trace, layout = statue_episode
    still = np.repeat(trace.com_pos[:1], trace.length, axis=0)
    forward, _pitch = heading_frames(trace.root_quat)
    spawn = still[0] - 0.002 * np.array([*forward[0], 0.0])
    moved = replace(trace, com_pos=still, spawn_com_pos=spawn)
    gap = (trace.frame_skip - 1) * trace.dt / trace.frame_skip
    omega0 = math.sqrt(layout.gravity / still[0, 2])
    ahead = float(np.dot(forward[0], (still[0] - trace.foot_pos[0].mean(axis=0))[:2]))
    expected = ahead + 0.002 / gap / omega0
    assert episode_balance_telemetry(moved, layout, settle_steps=0)["xcom_fore_aft_max_m"] == pytest.approx(expected)
    # Past the first sample the CoM stands still: the XcoM is the CoM.
    assert episode_balance_telemetry(moved, layout, settle_steps=1)["xcom_fore_aft_max_m"] == pytest.approx(
        float(np.einsum("tj,tj->t", forward, (still - trace.foot_pos.mean(axis=1))[:, :2])[1:].max())
    )
    # With one substep per step the spawn sample is the first boundary sample's own: its velocity is unmeasured,
    # and the maximum is the still CoM's.
    single = episode_balance_telemetry(replace(moved, frame_skip=1), layout, settle_steps=0)
    assert single["xcom_fore_aft_max_m"] == pytest.approx(
        float(np.einsum("tj,tj->t", forward, (still - trace.foot_pos.mean(axis=1))[:, :2])[1:].max())
    )


def test_the_window_means_and_the_actuator_fractions_read_the_trace(statue_episode):
    trace, layout = statue_episode
    window = slice(SETTLE, trace.length)
    balance = episode_balance_telemetry(trace, layout, settle_steps=SETTLE)
    forward, pitch = heading_frames(trace.root_quat)
    ahead = np.einsum("tj,tj->t", forward, (trace.com_pos - trace.foot_pos.mean(axis=1))[:, :2])
    assert balance["com_fore_aft_mean_m"] == pytest.approx(float(ahead[window].mean()), rel=1e-12)
    assert balance["root_pitch_mean_deg"] == pytest.approx(float(pitch[window].mean()), rel=1e-12)
    fraction = episode_actuator_force_fraction(trace, layout, settle_steps=SETTLE)
    expected = np.abs(trace.actuator_force[window]) / np.where(
        trace.actuator_force[window] >= 0, layout.force_limits[:, 1], -layout.force_limits[:, 0]
    )
    np.testing.assert_allclose(fraction, expected.mean(axis=0), rtol=1e-12)
    assert episode_force_telemetry(trace, layout, settle_steps=SETTLE)["actuator_force_frac_max"] == fraction.max()
    # Each sagittal pair's achieved offsets are about the leftward axis, from the home pose.
    for pair in layout.leg_pairs:
        left = pair.left_sign * (trace.joint_qpos[window, pair.left_column] - pair.left_home)
        right = pair.right_sign * (trace.joint_qpos[window, pair.right_column] - pair.right_home)
        assert balance[pair.difference_key] == pytest.approx(math.degrees(float((left - right).mean())), abs=1e-12)
        assert balance[pair.mean_key] == pytest.approx(math.degrees(float((left + right).mean() / 2)), abs=1e-12)


# ── which keys exist ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_id",
    (*BOX_SOLED, VELOCIRAPTOR_MONGOLIENSIS, BRACHIOSAURUS_ALTITHORAX),
    ids=str,
)
def test_each_key_exists_only_where_the_model_has_what_it_measures(envs, model_id):
    env = envs(model_id)
    morphology, layout = _layout(env, model_id)
    keys = layout.episode_keys()
    if model_id in (VELOCIRAPTOR_MONGOLIENSIS, BRACHIOSAURUS_ALTITHORAX):
        assert layout.is_empty() and keys == ()  # no box sole: no recorder, no field
        return
    assert set(GENERIC_BALANCE_KEYS) <= set(keys)
    tyrannosaurus = model_id == TYRANNOSAURUS_REX
    assert set(SPAN_BALANCE_KEYS) <= set(keys) if tyrannosaurus else not set(SPAN_BALANCE_KEYS) & set(keys)
    assert set(TIP_BALANCE_KEYS) <= set(keys) if tyrannosaurus else not set(TIP_BALANCE_KEYS) & set(keys)
    ankle_keys = {key for key in keys if key.endswith("_ankle_moment_arm_m")}
    assert ankle_keys == ({"r_ankle_moment_arm_m", "l_ankle_moment_arm_m"} if tyrannosaurus else set())
    for label in layout.labels:
        assert {f"{label}_shear_over_normal", f"{label}_cone_use_mean", f"{label}_cone_use_max"} <= set(keys)
    pairs = sagittal_leg_dc_pairs(env, morphology)
    assert [pair.difference_key for pair in layout.leg_pairs] == [
        "leg_q_" + pair.key.removeprefix("leg_dc_") for pair in pairs
    ]
    # No key reads as a foot to the readers that enumerate per-foot rail keys, or as a reward term or a cause.
    for key in keys:
        assert not key.endswith(RAIL_SUFFIXES), key
        assert not key.startswith(("term_", "termination_")), key
    assert set(FALL_ONLY_KEYS) <= set(keys) if tyrannosaurus else not set(FALL_ONLY_KEYS) & set(keys)


def test_an_empty_window_and_a_trace_without_balance_fields_read_nan(statue_episode):
    trace, layout = statue_episode
    for values in (
        episode_balance_telemetry(trace, layout, settle_steps=trace.length),
        episode_force_telemetry(trace, layout, settle_steps=trace.length),
        episode_balance_telemetry(replace(trace, com_pos=None), layout, settle_steps=SETTLE),
    ):
        assert values and all(math.isnan(value) for value in values.values())
    plain = episode_force_telemetry(replace(trace, actuator_force=None), layout, settle_steps=SETTLE)
    assert all(math.isnan(value) for value in plain.values())
    assert np.isnan(episode_actuator_force_fraction(trace, layout, settle_steps=trace.length)).all()
    with pytest.raises(ValueError, match="settle_steps"):
        episode_balance_telemetry(trace, layout, settle_steps=-1)


def test_a_pyramidal_cone_leaves_the_forces_unmeasured_and_the_kinematics_measured(envs):
    env = envs(DIBOTHROSUCHUS_ELAPHROS)
    env.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL  # in memory only, before any reset or recorder attach
    morphology, layout = _layout(env, DIBOTHROSUCHUS_ELAPHROS)
    trace, _rows = _episode(env, DIBOTHROSUCHUS_ELAPHROS, steps=60, morphology=morphology)
    balance = episode_balance_telemetry(trace, layout, settle_steps=10)
    forces = episode_force_telemetry(trace, layout, settle_steps=10)
    measured = ("com_fore_aft_mean_m", "root_pitch_mean_deg", "xcom_fore_aft_max_m")
    assert all(math.isfinite(balance[key]) for key in measured)
    assert all(math.isfinite(balance[pair.difference_key]) for pair in layout.leg_pairs) and layout.leg_pairs
    assert all(math.isnan(balance[key]) for key in balance if key.startswith("hof_"))
    assert all(math.isnan(value) for key, value in forces.items() if not key.startswith("actuator_force_"))
    assert math.isfinite(forces["actuator_force_frac_max"])


def test_the_mirror_even_key_of_a_leg_dc_key():
    assert mirror_even_key("leg_dc_l_minus_r_ankle_deg") == "leg_dc_l_r_mean_ankle_deg"
    assert mirror_even_key("leg_dc_fl_minus_fr_hip_pitch_deg") == "leg_dc_fl_fr_mean_hip_pitch_deg"


# ── the heading frames ───────────────────────────────────────────────────────


def test_the_heading_frames_are_the_per_sample_helpers_vectorised():
    rng = np.random.default_rng(11)
    quats = rng.normal(size=(1000, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    # The singular case: the root's x axis vertical (a quarter turn about y), where the helper falls back.
    quats = np.vstack([quats, [math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0]])
    forward, pitch = heading_frames(quats)
    for quat, row, angle in zip(quats, forward, pitch, strict=True):
        rotation = body_to_world(quat)
        np.testing.assert_allclose(row, world_to_heading(rotation)[0, :2], rtol=0, atol=1e-12)
        assert angle == pytest.approx(math.degrees(math.asin(np.clip(rotation[2, 0], -1, 1))), abs=1e-12)
    assert pitch[-1] == pytest.approx(90.0)  # nose straight up


# ── failure costs the balance fields, never the episode ──────────────────────


@pytest.mark.parametrize("where", ["hook", "reduction"])
def test_a_failing_capture_or_reduction_costs_only_the_balance_fields(envs, monkeypatch, caplog, where):
    env = envs(TYRANNOSAURUS_REX)
    actions = np.zeros((80, env.model.nu))
    plain, _rows = _episode(env, TYRANNOSAURUS_REX, steps=80, balance=False, actions=actions)
    if where == "hook":
        calls = {"n": 0}
        capture = SubstepContactRecorder._balance_capture

        def fail_on_the_50th(self, env_, data):
            calls["n"] += 1
            if calls["n"] == 50:
                raise FloatingPointError("a balance read went wrong")
            capture(self, env_, data)

        monkeypatch.setattr(SubstepContactRecorder, "_balance_capture", fail_on_the_50th)
    else:

        def fail(self, **_kwargs):
            raise FloatingPointError("a balance reduction went wrong")

        monkeypatch.setattr(SubstepContactRecorder, "_reduce_balance", fail)
    with caplog.at_level(logging.WARNING, logger="environments.shared.gait.recorder"):
        traced, _rows = _episode(env, TYRANNOSAURUS_REX, steps=80, balance=True, actions=actions)
    for name in EpisodeTrace.__dataclass_fields__:
        if name in BALANCE_FIELDS:
            assert getattr(traced, name) is None, name
        elif isinstance(getattr(plain, name), np.ndarray):
            np.testing.assert_array_equal(getattr(plain, name), getattr(traced, name), err_msg=name)
    assert sum("Balance telemetry" in record.message for record in caplog.records) == 1
    _morphology, layout = _layout(env, TYRANNOSAURUS_REX)
    assert all(math.isnan(value) for value in episode_balance_telemetry(traced, layout, settle_steps=10).values())


def test_a_foot_without_a_declared_ankle_reads_nan_torque_never_another_dofs(envs):
    env = envs(TYRANNOSAURUS_REX)
    morphology = Morphology.from_env(env, TYRANNOSAURUS_REX)
    right, left = morphology.feet
    undeclared = replace(
        morphology,
        feet=(
            replace(right, ankle_joint=-1, ankle_forward_m=math.nan, heel_line_m=math.nan, toe_line_m=math.nan),
            left,
        ),
    )
    reference = BoundaryForces(env, morphology)
    env._substep_probe_hook = reference
    trace, _rows = _episode(
        env, TYRANNOSAURUS_REX, steps=20, morphology=undeclared, actions=np.zeros((20, env.model.nu))
    )
    assert np.isnan(trace.ankle_torque[:, 0]).all() and np.isnan(trace.support_line_pos[:, 0]).all()
    ankles = np.array(reference.ankle)[morphology.frame_skip - 1 :: morphology.frame_skip]
    np.testing.assert_array_equal(trace.ankle_torque[:, 1], ankles[:, 1])
    assert np.isfinite(trace.support_line_pos[:, 1]).all()


def test_no_constraint_rows_give_zero_tangential_forces(envs):
    env = envs(TYRANNOSAURUS_REX)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, TYRANNOSAURUS_REX), balance=True)
    recorder._elliptic = True
    address, dim = np.array([4, -1, 9], dtype=np.int32), np.array([3, 3, 1], dtype=np.int32)
    np.testing.assert_array_equal(recorder._tangential_forces([(address, dim, np.zeros(0))]), np.zeros((3, 2)))
    efc = np.arange(12, dtype=np.float64)
    # A condim-1 contact has no tangential row, and a contact without rows reads 0, never another contact's.
    expected = [[5.0, 6.0], [0.0, 0.0], [0.0, 0.0]]
    np.testing.assert_array_equal(recorder._tangential_forces([(address, dim, efc)]), expected)
    # The episode is decoded in one pass: each substep's addresses read its own rows, a substep with no rows
    # reads 0 wherever it falls, and the result is the per-substep decodes one after another.
    second = (np.array([0, 2], dtype=np.int32), np.array([3, 3], dtype=np.int32), 100.0 + np.arange(5.0))
    empty = (np.array([1], dtype=np.int32), np.array([3], dtype=np.int32), np.zeros(0))
    rows = [(address, dim, efc), empty, second, empty]
    np.testing.assert_array_equal(
        recorder._tangential_forces(rows), expected + [[0.0, 0.0], [101.0, 102.0], [103.0, 104.0], [0.0, 0.0]]
    )
    assert recorder._tangential_forces([]).shape == (0, 2)
    recorder._elliptic = False
    assert np.isnan(recorder._tangential_forces(rows)).all() and recorder._tangential_forces(rows).shape == (7, 2)


def test_a_model_the_capture_cannot_describe_is_refused_so_the_caller_can_fall_back(envs, monkeypatch):
    from environments.shared.gait.recorder import RecorderError

    env = envs(TYRANNOSAURUS_REX)
    morphology = Morphology.from_env(env, TYRANNOSAURUS_REX)
    broken = replace(morphology, support_geom_foot=())
    with pytest.raises(RecorderError, match="balance capture"):
        SubstepContactRecorder(env, broken, balance=True)
    SubstepContactRecorder(env, broken, balance=False)  # the plain recorder needs none of it


def test_the_layout_reads_the_reset_keyframe_not_the_env_state(envs):
    env = envs(TYRANNOSAURUS_REX)
    env.reset(seed=3042)
    for _ in range(5):
        env.step(np.full(env.model.nu, 0.3))
    before = (env.data.qpos.copy(), env.data.qvel.copy())
    morphology, layout = _layout(env, TYRANNOSAURUS_REX)
    np.testing.assert_array_equal(env.data.qpos, before[0])
    np.testing.assert_array_equal(env.data.qvel, before[1])
    key = env.model.key_qpos[0]
    for pair in layout.leg_pairs:
        assert pair.left_home == key[7 + pair.left_column] and pair.right_home == key[7 + pair.right_column]
    assert layout.ankle_left_sign == (1.0, 1.0)  # both T. rex ankle axes point left
    assert layout.mass_kg == pytest.approx(morphology.body_weight_n / 9.81)
    assert balance_telemetry.actuator_force_limits(env.model).shape == (env.model.nu, 2)
