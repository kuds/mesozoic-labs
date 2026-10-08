"""Tests for the floor-truth recorder on real envs: inert, exact, and loud when it cannot be either.

The prototype's equivalence checks, as tests, on short rollouts of the cheap
species (trex for the pyramidal cone, compsognathus for the elliptic one):

* **inert** -- qpos is bit-identical with and without the recorder;
* **exact** -- per-step leg, support-geom and total floor forces equal a
  per-contact ``mj_contactForce`` loop (:class:`ContactForceLoop`, the
  reference implementation the batched decode replaced; measured max |diff|
  0.0 on all six species, so the tolerance is a formality), on both the
  one-gather decode and the general per-contact-dim one;
* **touch** is bitwise the env's ``_aggregated_foot_contact_forces()``, and
  the **spawn** sample bitwise the post-reset pose (the SB3 auto-reset path
  is ``test_gait_vec_path.py``, which needs SB3);
* the **hook** is chained (the trex behavior env's terrain probe still fires
  every substep), restored exactly, and refused when a foreign hook sits on
  top;
* every :class:`RecorderError` path fails closed.

The six-statue registry evidence (GAIT_QUALITY_PLAN_2026_09.md §3.1: the
foot / non-foot split counts only once each statue is shown putting its
floor load on registered foot geoms) runs one short statue episode per
species at the end.
"""

from __future__ import annotations

import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from environments.shared.config import load_stage_config
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import (
    RecorderError,
    SubstepContactRecorder,
    _corner_heights,
    _sole_angles,
    _sole_corners,
)
from environments.shared.gait.stance_metrics import episode_stance_metrics
from environments.shared.species_registry import get_species_config

SIX = ("trex", "velociraptor", "compsognathus", "compsognathus_robot", "dibothrosuchus", "brachiosaurus")


def make_env(species: str, **overrides):
    kwargs = {**dict(load_stage_config(species, "stance")["env_kwargs"]), **overrides}
    return get_species_config(species).env_class(**kwargs)


@pytest.fixture
def envs():
    built = []

    def factory(species: str, **overrides):
        env = make_env(species, **overrides)
        built.append(env)
        return env

    yield factory
    for env in built:
        env.close()


def random_actions(env, steps: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.6, 0.6, size=(steps, env.model.nu))


class ContactForceLoop:
    """The reference: a substep hook that calls ``mj_contactForce`` on every animal-floor contact, one by one."""

    def __init__(self, env, morphology: Morphology) -> None:
        self.env, self.morphology = env, morphology
        self.force = np.zeros(6)
        self.substeps: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.leg_mean: list[np.ndarray] = []
        self.support_mean: list[np.ndarray] = []
        self.total_max: list[float] = []

    def __call__(self) -> None:
        model, data, morphology = self.env.model, self.env.data, self.morphology
        leg = np.zeros(morphology.n_feet)
        support = np.zeros(len(morphology.support_geom_ids))
        total = 0.0
        for index in range(data.ncon):
            first, second = (int(geom) for geom in data.contact.geom[index])
            if morphology.is_floor[first] == morphology.is_floor[second]:
                continue
            other = second if morphology.is_floor[first] else first
            if not morphology.is_animal[other]:
                continue
            mujoco.mj_contactForce(model, data, index, self.force)
            normal = float(self.force[0])
            total += normal
            if morphology.geom_foot[other] >= 0:
                leg[morphology.geom_foot[other]] += normal
            if morphology.geom_support[other] >= 0:
                support[morphology.geom_support[other]] += normal
        self.substeps.append((leg, support, total))

    def end_step(self) -> None:
        self.leg_mean.append(np.mean([row[0] for row in self.substeps], axis=0))
        self.support_mean.append(np.mean([row[1] for row in self.substeps], axis=0))
        self.total_max.append(max(row[2] for row in self.substeps))
        self.substeps = []


# ── inert and exact ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("species", ["trex", "compsognathus"])
def test_recording_leaves_the_trajectory_bit_identical(envs, species):
    steps = 120
    trajectories = []
    for record in (False, True):
        env = envs(species)
        actions = random_actions(env, steps)
        recorder = SubstepContactRecorder(env, Morphology.from_env(env, species)).attach() if record else None
        env.reset(seed=3042)
        rows = []
        for action in actions:
            env.step(action)
            rows.append(np.concatenate([env.data.qpos, env.data.qvel]))
            if recorder is not None:
                recorder.end_step(action, 0.0)
        if recorder is not None:
            assert recorder.end_episode().length == steps
            recorder.detach()
        trajectories.append(np.array(rows))
    np.testing.assert_array_equal(trajectories[0], trajectories[1])


@pytest.mark.parametrize("species", ["trex", "compsognathus"])
@pytest.mark.parametrize("decode", ["one_gather", "per_contact_dim"])
def test_the_normal_forces_equal_an_mj_contact_force_loop(envs, species, decode):
    env = envs(species)
    morphology = Morphology.from_env(env, species)
    reference = ContactForceLoop(env, morphology)
    env._substep_probe_hook = reference  # the recorder chains it, so both read every substep
    recorder = SubstepContactRecorder(env, morphology).attach()
    if decode == "per_contact_dim":
        recorder._configure_decode(env.model)
        recorder._edge_offsets = None  # force the general branch every model takes with mixed condims
    env.reset(seed=3042)
    for action in random_actions(env, 60, seed=1):
        env.step(action)
        recorder.end_step(action, 0.0)
        reference.end_step()
    trace = recorder.end_episode()
    recorder.detach()
    assert env._substep_probe_hook is reference
    tolerance = 1e-9 * morphology.body_weight_n
    np.testing.assert_allclose(trace.leg_floor_mean, np.array(reference.leg_mean), rtol=0, atol=tolerance)
    np.testing.assert_allclose(trace.support_floor_mean, np.array(reference.support_mean), rtol=0, atol=tolerance)
    np.testing.assert_allclose(trace.total_floor_max, np.array(reference.total_max), rtol=0, atol=tolerance)
    assert trace.leg_floor_mean.sum() > 0.5 * morphology.body_weight_n * trace.length  # it measured real contact


@pytest.mark.parametrize("species", ["trex", "compsognathus"])
def test_touch_is_the_env_aggregate_and_the_samples_are_the_env_state(envs, species):
    env = envs(species)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, species)).attach()
    env.reset(seed=3043)
    spawn = (
        env.data.site_xpos[[foot.primary_site for foot in recorder._morphology.feet]].copy(),
        env.data.xpos[recorder._morphology.root_body].copy(),
        env.data.xquat[recorder._morphology.root_body].copy(),
    )
    aggregates, qpos, ctrl = [], [], []
    for action in random_actions(env, 40, seed=2):
        env.step(action)
        recorder.end_step(action, 1.0)
        aggregates.append(np.asarray(env._aggregated_foot_contact_forces()))
        qpos.append(env.data.qpos[7:].copy())
        ctrl.append(env.data.ctrl.copy())
    trace = recorder.end_episode(truncated=True, termination_reason="horizon")
    recorder.detach()
    np.testing.assert_array_equal(trace.touch, np.array(aggregates))
    np.testing.assert_array_equal(trace.spawn_foot_pos, spawn[0])
    np.testing.assert_array_equal(trace.spawn_root_pos, spawn[1])
    np.testing.assert_array_equal(trace.spawn_root_quat, spawn[2])
    np.testing.assert_array_equal(trace.joint_qpos, np.array(qpos))
    np.testing.assert_array_equal(trace.ctrl, np.array(ctrl))
    assert (trace.truncated, trace.terminated, trace.termination_reason) == (True, False, "horizon")
    np.testing.assert_array_equal(trace.reward, np.ones(40))


# ── the hook: chained, restored, never silently displaced ────────────────────


def test_the_behavior_env_terrain_probe_stays_chained_and_comes_back(monkeypatch):
    from environments.shared.behavior_env import SpeciesBehaviorMixin, get_behavior_env_class
    from environments.shared.direction_commands import DirectionCommandConfig

    calls: list[int] = []
    probe = SpeciesBehaviorMixin._probe_terrain_contacts

    def counting_probe(self):
        calls.append(self._step_count)
        return probe(self)

    monkeypatch.setattr(SpeciesBehaviorMixin, "_probe_terrain_contacts", counting_probe)
    env = get_behavior_env_class("trex")(commands=DirectionCommandConfig(cruise_speed=1.05, speed_scale=1.575))
    try:
        installed = env._substep_probe_hook
        assert installed == env._probe_terrain_contacts
        recorder = SubstepContactRecorder(env, Morphology.from_env(env, "trex")).attach()
        env.reset(seed=1)
        zero = np.zeros(env.model.nu)
        for _ in range(3):
            env.step(zero)
            recorder.end_step(zero)
        assert calls == [0] * 5 + [1] * 5 + [2] * 5  # frame_skip times per step, ahead of the recorder
        assert recorder.end_episode().length == 3
        recorder.detach()
        assert env._substep_probe_hook is installed
    finally:
        env.close()


def test_detach_restores_the_class_level_none_on_a_canonical_env(envs):
    env = envs("compsognathus")
    assert env._substep_probe_hook is None and "_substep_probe_hook" not in vars(env)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, "compsognathus"))
    recorder.detach()  # not attached: a no-op
    recorder.attach()
    assert "_substep_probe_hook" in vars(env) and recorder.attached
    recorder.detach()
    assert "_substep_probe_hook" not in vars(env) and env._substep_probe_hook is None and not recorder.attached


def test_hooks_unwind_in_order_and_a_foreign_hook_on_top_is_refused(envs):
    env = envs("compsognathus")
    morphology = Morphology.from_env(env, "compsognathus")
    first = SubstepContactRecorder(env, morphology).attach()
    with pytest.raises(RecorderError, match="already attached"):
        first.attach()
    second = SubstepContactRecorder(env, morphology).attach()
    with pytest.raises(RecorderError, match="on top"):
        first.detach()
    assert first.attached and env._substep_probe_hook is second._bound_hook  # nothing moved
    env.reset(seed=1)
    zero = np.zeros(env.model.nu)
    env.step(zero)
    first.end_step(zero)
    second.end_step(zero)  # both saw every substep
    second.detach()
    first.detach()
    assert env._substep_probe_hook is None


def test_the_context_manager_detaches_when_the_rollout_raises(envs):
    env = envs("compsognathus")
    with pytest.raises(ZeroDivisionError):
        with SubstepContactRecorder(env, Morphology.from_env(env, "compsognathus")):
            env.reset(seed=1)
            1 / 0
    assert "_substep_probe_hook" not in vars(env)


# ── failing closed ───────────────────────────────────────────────────────────


@pytest.fixture
def attached(envs):
    env = envs("trex")
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, "trex")).attach()
    yield env, recorder, np.zeros(env.model.nu)
    if recorder.attached and env._substep_probe_hook is recorder._bound_hook:
        recorder.detach()


def test_an_end_step_without_a_step_is_refused(attached):
    env, recorder, zero = attached
    env.reset(seed=1)
    with pytest.raises(RecorderError, match="fired 0 times"):
        recorder.end_step(zero)


def test_a_skipped_end_step_is_refused(attached):
    env, recorder, zero = attached
    env.reset(seed=1)
    env.step(zero)
    env.step(zero)
    with pytest.raises(RecorderError, match="fired 10 times"):
        recorder.end_step(zero)


def test_an_end_episode_with_no_steps_or_an_unclosed_step_is_refused(attached):
    env, recorder, zero = attached
    with pytest.raises(RecorderError, match="no recorded steps"):
        recorder.end_episode()
    env.reset(seed=1)
    env.step(zero)
    with pytest.raises(RecorderError, match="has not closed"):
        recorder.end_episode()


def test_a_new_episode_before_end_episode_is_refused(attached):
    env, recorder, zero = attached
    env.reset(seed=1)
    env.step(zero)
    recorder.end_step(zero)
    env.reset(seed=2)
    with pytest.raises(RecorderError, match="before end_episode"):
        env.step(zero)


def test_a_hook_overwritten_after_attach_is_reported_not_ignored(attached):
    env, recorder, zero = attached
    env._substep_probe_hook = lambda: None  # what stance_duty_validation.py does to the slot
    env.reset(seed=1)
    env.step(zero)
    with pytest.raises(RecorderError, match="fired 0 times"):
        recorder.end_step(zero)
    with pytest.raises(RecorderError, match="on top"):
        recorder.detach()
    del env._substep_probe_hook


def test_a_recorder_attached_mid_episode_is_refused(envs):
    env = envs("trex")
    zero = np.zeros(env.model.nu)
    env.reset(seed=1)
    env.step(zero)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, "trex")).attach()
    env.step(zero)
    with pytest.raises(RecorderError, match="mid-episode"):
        recorder.end_step(zero)
    env.reset(seed=1)  # the next episode records normally
    env.step(zero)
    recorder.end_step(zero)
    assert recorder.end_episode().length == 1
    recorder.detach()


def test_an_action_of_the_wrong_size_is_refused(attached):
    env, recorder, zero = attached
    env.reset(seed=1)
    env.step(zero)
    with pytest.raises(RecorderError, match="actuators"):
        recorder.end_step(np.zeros(env.model.nu + 1))


def test_a_morphology_for_another_env_is_refused(envs):
    env = envs("trex")
    morphology = Morphology.from_env(env, "trex")
    with pytest.raises(RecorderError, match="frame_skip"):
        SubstepContactRecorder(env, replace(morphology, frame_skip=10))
    with pytest.raises(RecorderError, match="another model"):
        SubstepContactRecorder(env, replace(morphology, geom_foot=morphology.geom_foot[:-1]))
    with pytest.raises(RecorderError, match="BaseDinoEnv"):
        SubstepContactRecorder(object(), morphology)


# ── what a step records ──────────────────────────────────────────────────────


def test_the_policy_command_is_recorded_clipped_and_an_absent_one_as_nan(attached):
    env, recorder, zero = attached
    env.reset(seed=1)
    command = np.zeros(env.model.nu)
    command[0], command[1] = 1.7, -0.995
    env.step(command)
    recorder.end_step(command, 2.5)
    env.step(zero)
    recorder.end_step()
    trace = recorder.end_episode()
    assert trace.action[0, 0] == 1.0 and trace.action[0, 1] == -0.995
    assert np.isnan(trace.action[1]).all()
    assert trace.reward[0] == 2.5 and math.isnan(trace.reward[1])
    metrics = episode_stance_metrics(trace, settle_steps=0)
    assert math.isnan(metrics.max_actuator_saturation_fraction) and math.isnan(metrics.reward)


def test_per_geom_mode_splits_the_total_floor_force_by_geom(envs):
    env = envs("compsognathus")
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, "compsognathus"), per_geom=True).attach()
    env.reset(seed=1)
    for action in random_actions(env, 20, seed=3):
        env.step(action)
        recorder.end_step(action)
    trace = recorder.end_episode()
    recorder.detach()
    assert trace.geom_floor_mean is not None and trace.geom_floor_mean.shape == (20, len(trace.geom_floor_names))
    np.testing.assert_allclose(trace.geom_floor_mean.sum(axis=1), trace.total_floor_mean, rtol=1e-12, atol=1e-12)
    support = [trace.geom_floor_names.index(name) for name in trace.support_geom_names]
    np.testing.assert_allclose(trace.geom_floor_mean[:, support], trace.support_floor_mean, rtol=1e-12, atol=1e-12)


def test_a_rolled_box_lifts_its_corners_and_reads_as_outer_edge_up(envs):
    """The geometry behind the trex audit's finding (pad rolled 3-6 degrees outer edge up, CoP on the inner edge)."""
    env = envs("trex")
    morphology = Morphology.from_env(env, "trex")
    right = morphology.feet[0]
    corners = _sole_corners(env.model, right)
    half_width = env.model.geom_size[right.sole_geom][1]
    angle = math.radians(4.0)
    # The right foot's outer edge is its -y edge: lifting it is a rotation by -angle about the forward axis.
    rotation = np.array([[1, 0, 0], [0, math.cos(-angle), -math.sin(-angle)], [0, math.sin(-angle), math.cos(-angle)]])
    frames = np.repeat(rotation.reshape(1, 1, 9), 2, axis=1)  # both feet, one sample
    heights = _corner_heights(np.zeros((1, 3)), frames[:, 0], corners)
    assert heights.max() - heights.min() == pytest.approx(2 * half_width * math.sin(angle))
    tilt, roll, pitch = _sole_angles(morphology, frames)
    assert tilt[0] == pytest.approx([4.0, 4.0])
    assert roll[0, 0] == pytest.approx(4.0)  # right foot: outer edge up
    assert roll[0, 1] == pytest.approx(-4.0)  # the same rotation lifts the left foot's INNER edge
    assert pitch[0] == pytest.approx([0.0, 0.0], abs=1e-12)
    toe_up = np.array([[math.cos(angle), 0, -math.sin(angle)], [0, 1, 0], [math.sin(angle), 0, math.cos(angle)]])
    _tilt, _roll, pitch = _sole_angles(morphology, np.repeat(toe_up.reshape(1, 1, 9), 2, axis=1))
    assert pitch[0] == pytest.approx([4.0, 4.0])


# ── the six statues put their floor load on the registry ─────────────────────


@pytest.mark.parametrize("species", SIX)
def test_each_statue_stands_on_its_registered_support_geoms(envs, species):
    env = envs(species)
    settle = int(round(1.0 / env.dt))  # 1 s
    zero = np.zeros(env.model.nu)
    with SubstepContactRecorder(env, Morphology.from_env(env, species)) as recorder:
        env.reset(seed=3042)
        for _ in range(settle + 150):
            env.step(zero)
            recorder.end_step(zero, 0.0)
        trace = recorder.end_episode(truncated=True)
    result = episode_stance_metrics(trace, settle_steps=settle)
    assert result.nonfoot_load_fraction == 0.0
    assert result.offsupport_load_fraction <= 0.01
    assert result.all_feet_support == 1.0
    assert result.settle_airborne_substeps == 0.0
    assert result.touchdown_rate == 0.0 and result.phantom_support_fraction == 0.0
    assert result.max_actuator_saturation_fraction == 0.0
    assert sum(result.foot_load_share) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("species", "seed", "airborne"),
    [("dibothrosuchus", 3045, 9.0), ("compsognathus_robot", 3052, 3.0), ("compsognathus", 3045, 0.0)],
)
def test_the_spawn_pop_is_counted_substep_by_substep(envs, species, seed, airborne):
    """The reset pop ``constants.SPAWN_GRACE_S`` exists for, pinned at the panel seeds that show it.

    Every leg's floor force at or below ``CONTACT_THRESHOLD_N`` on a substep
    is one airborne substep: the first two statues leave the floor entirely
    for 9 and 3 substeps after the reset (the largest counts their 40-episode
    panels record, seeds 3042-3081) and never inside the settle window.
    Exact, because the count IS the threshold's meaning: read at 1.0 N
    instead of 0.1 N, the anatomical compsognathus's one-substep pop on
    physics r1 (seed 3045) counted 2.  Its physics r2 toe armature removed
    the pop: 0 airborne substeps on every panel seed, pinned at the seed
    that showed it.
    """
    env = envs(species)
    settle = int(round(1.0 / env.dt))
    zero = np.zeros(env.model.nu)
    with SubstepContactRecorder(env, Morphology.from_env(env, species)) as recorder:
        env.reset(seed=seed)
        for _ in range(settle + 20):
            env.step(zero)
            recorder.end_step(zero, 0.0)
        trace = recorder.end_episode(truncated=True)
    result = episode_stance_metrics(trace, settle_steps=settle)
    assert result.spawn_airborne_substeps == airborne
    assert result.settle_airborne_substeps == 0.0
