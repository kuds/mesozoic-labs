"""Tests for the floor-truth recorder on real envs: inert, exact, and loud when it cannot be either.

The prototype's equivalence checks, as tests, on short rollouts of the cheap
species (dibothrosuchus switched to the pyramidal cone in memory, compsognathus for the elliptic one):

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
* every :class:`RecorderError` path fails closed;
* the **whole-foot centre of pressure** is what its definition says (an
  independent probe of every limb contact, with the ankle's own world
  anchor, reproduces it), is the vertical-load moment about the ankle axis
  over the limb's floor force within a tolerance stated from the sole's tilt,
  and its reading of the T. rex physics-r9 statue (seed 3042) is pinned; each
  hinged digit's load position is reproduced the same way.

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
from environments.shared.model_id import ModelId
from environments.shared.species_names import model_ids
from environments.shared.species_registry import get_species_config

VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")
DIBOTHROSUCHUS_ELAPHROS = ModelId("dibothrosuchus_elaphros", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")

MODEL_IDS = (
    TYRANNOSAURUS_REX,
    VELOCIRAPTOR_MONGOLIENSIS,
    COMPSOGNATHUS_LONGIPES,
    COMPSOGNATHUS_LONGIPES_ROBOT,
    DIBOTHROSUCHUS_ELAPHROS,
    BRACHIOSAURUS_ALTITHORAX,
)
#: One cheap species per friction cone, keyed by the cone (the pytest id), so both of the recorder's normal-force
#: decodes stay covered.  Every plant has declared the elliptic cone since D-D30 (plant_versions note 17;
#: test_friction_cone.py guards it), so none exercises the pyramidal decode -- the sum of a contact's 2 * (condim - 1)
#: pyramid edge forces -- any more, yet the recorder still serves any model that declares no cone, MuJoCo's
#: default.  dibothrosuchus (box feet, four contacts each) therefore runs with its model's cone switched to
#: pyramidal in memory, before any reset or recorder attach; compsognathus runs as built.
CONE_CASES = {
    "pyramidal": (DIBOTHROSUCHUS_ELAPHROS, mujoco.mjtCone.mjCONE_PYRAMIDAL),
    "elliptic": (COMPSOGNATHUS_LONGIPES, mujoco.mjtCone.mjCONE_ELLIPTIC),
}


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


def cone_env(envs, case: str):
    """The case's species and its env, the model's friction cone set to the case's (in memory only)."""
    model_id, cone = CONE_CASES[case]
    env = envs(model_id)
    env.model.opt.cone = cone
    return model_id, env


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


@pytest.mark.parametrize("case", list(CONE_CASES))
def test_each_case_runs_on_its_cone(envs, case):
    """The in-memory switch reaches the physics: MuJoCo builds the case's contact rows and the recorder decodes them."""
    model_id, env = cone_env(envs, case)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, model_id)).attach()
    env.reset(seed=3042)
    zero = np.zeros(env.model.nu)
    env.step(zero)
    recorder.end_step(zero, 0.0)
    recorder.detach()
    expected = (
        mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC
        if case == "elliptic"
        else mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
    )
    addresses = env.data.contact.efc_address[: env.data.ncon]
    assert addresses.size > 0 and set(env.data.efc_type[addresses[addresses >= 0]].tolist()) == {expected}
    assert recorder._elliptic == (case == "elliptic")


@pytest.mark.parametrize("case", list(CONE_CASES))
def test_recording_leaves_the_trajectory_bit_identical(envs, case):
    steps = 120
    trajectories = []
    for record in (False, True):
        model_id, env = cone_env(envs, case)
        actions = random_actions(env, steps)
        recorder = SubstepContactRecorder(env, Morphology.from_env(env, model_id)).attach() if record else None
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


@pytest.mark.parametrize("case", list(CONE_CASES))
@pytest.mark.parametrize("decode", ["one_gather", "per_contact_dim"])
def test_the_normal_forces_equal_an_mj_contact_force_loop(envs, case, decode):
    model_id, env = cone_env(envs, case)
    morphology = Morphology.from_env(env, model_id)
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


@pytest.mark.parametrize("case", list(CONE_CASES))
def test_touch_is_the_env_aggregate_and_the_samples_are_the_env_state(envs, case):
    model_id, env = cone_env(envs, case)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, model_id)).attach()
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
    env = get_behavior_env_class(TYRANNOSAURUS_REX)(
        commands=DirectionCommandConfig(cruise_speed=1.05, speed_scale=1.575)
    )
    try:
        installed = env._substep_probe_hook
        assert installed == env._probe_terrain_contacts
        recorder = SubstepContactRecorder(env, Morphology.from_env(env, TYRANNOSAURUS_REX)).attach()
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
    env = envs(COMPSOGNATHUS_LONGIPES)
    assert env._substep_probe_hook is None and "_substep_probe_hook" not in vars(env)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, COMPSOGNATHUS_LONGIPES))
    recorder.detach()  # not attached: a no-op
    recorder.attach()
    assert "_substep_probe_hook" in vars(env) and recorder.attached
    recorder.detach()
    assert "_substep_probe_hook" not in vars(env) and env._substep_probe_hook is None and not recorder.attached


def test_hooks_unwind_in_order_and_a_foreign_hook_on_top_is_refused(envs):
    env = envs(COMPSOGNATHUS_LONGIPES)
    morphology = Morphology.from_env(env, COMPSOGNATHUS_LONGIPES)
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
    env = envs(COMPSOGNATHUS_LONGIPES)
    with pytest.raises(ZeroDivisionError):
        with SubstepContactRecorder(env, Morphology.from_env(env, COMPSOGNATHUS_LONGIPES)):
            env.reset(seed=1)
            1 / 0
    assert "_substep_probe_hook" not in vars(env)


# ── failing closed ───────────────────────────────────────────────────────────


@pytest.fixture
def attached(envs):
    env = envs(TYRANNOSAURUS_REX)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, TYRANNOSAURUS_REX)).attach()
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
    env = envs(TYRANNOSAURUS_REX)
    zero = np.zeros(env.model.nu)
    env.reset(seed=1)
    env.step(zero)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, TYRANNOSAURUS_REX)).attach()
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
    env = envs(TYRANNOSAURUS_REX)
    morphology = Morphology.from_env(env, TYRANNOSAURUS_REX)
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
    env = envs(COMPSOGNATHUS_LONGIPES)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, COMPSOGNATHUS_LONGIPES), per_geom=True).attach()
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
    env = envs(TYRANNOSAURUS_REX)
    morphology = Morphology.from_env(env, TYRANNOSAURUS_REX)
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


# ── the whole-foot centre of pressure ────────────────────────────────────────


class WholeFootProbe:
    """An independent read of every limb-floor contact, chained under the recorder (it runs first, every substep).

    Per substep and contact: the ``mj_contactForce`` normal force and the point; per foot, the sole's frame and
    the ankle's WORLD anchor and axis (``xanchor`` / ``xaxis``, not the recorder's constant offset); per hinged
    digit, its capsule's frame and its MTP joint's world anchor.  :meth:`end_step` reduces the step's substeps
    to the definition (force-weighted, along the sole's forward axis from the ankle anchor) and to the physics
    (the vertical load's moment about the ankle axis over the limb's floor force), with the per-step bound on
    their difference that the sole's tilt sets.
    """

    def __init__(self, env, morphology: Morphology) -> None:
        self.env, self.morphology = env, morphology
        model = env.model
        self.force = np.zeros(6)
        self.digits = [
            (support, int(geom), int(joint))
            for support, (geom, joint) in enumerate(zip(morphology.support_geom_ids, morphology.support_geom_mtp))
            if joint >= 0
        ]
        self.reach = {geom: float(model.geom_size[geom][1] + model.geom_size[geom][0]) for _, geom, _ in self.digits}
        self.rows: list[tuple] = []
        self.definition: list[np.ndarray] = []
        self.moment_over_force: list[np.ndarray] = []
        self.bound: list[np.ndarray] = []
        self.load_pos: list[np.ndarray] = []

    def __call__(self) -> None:
        model, data, morphology = self.env.model, self.env.data, self.morphology
        feet = []
        for foot in morphology.feet:
            frame = data.geom_xmat[foot.sole_geom].reshape(3, 3).copy()
            feet.append((frame, data.xanchor[foot.ankle_joint].copy(), data.xaxis[foot.ankle_joint].copy()))
        digits = {
            support: (
                data.geom_xpos[geom].copy(),
                data.geom_xmat[geom].reshape(3, 3)[:, 2].copy(),
                data.xanchor[joint].copy(),
            )
            for support, geom, joint in self.digits
        }
        contacts = []
        for index in range(data.ncon):
            first, second = (int(geom) for geom in data.contact.geom[index])
            if morphology.is_floor[first] == morphology.is_floor[second]:
                continue
            other = second if morphology.is_floor[first] else first
            if not morphology.is_animal[other] or morphology.geom_foot[other] < 0:
                continue
            mujoco.mj_contactForce(model, data, index, self.force)
            contacts.append(
                (int(morphology.geom_foot[other]), other, float(self.force[0]), data.contact.pos[index].copy())
            )
        self.rows.append((feet, digits, contacts))

    def end_step(self) -> None:
        morphology = self.morphology
        n_feet, n_support = morphology.n_feet, len(morphology.support_geom_ids)
        weight, ahead, lever, bound = np.zeros(n_feet), np.zeros(n_feet), np.zeros(n_feet), np.zeros(n_feet)
        digit_weight, digit_pos = np.zeros(n_support), np.zeros(n_support)
        up = np.array([0.0, 0.0, 1.0])
        for feet, digits, contacts in self.rows:
            for foot, geom, normal, point in contacts:
                frame, anchor, axis = feet[foot]
                forward, lateral, up_sole = frame[:, 0], frame[:, 1], frame[:, 2]
                arm = point - anchor
                along = float(forward @ arm)
                sign = 1.0 if axis @ lateral > 0 else -1.0
                tilt = math.acos(min(1.0, abs(float(up_sole @ up))))
                weight[foot] += normal
                ahead[foot] += normal * along
                lever[foot] += normal * sign * float(np.cross(up, arm) @ axis)
                bound[foot] = max(
                    bound[foot], abs(float(up_sole @ arm)) * math.sin(tilt) + abs(along) * (1 - math.cos(tilt))
                )
                support = int(morphology.geom_support[geom])
                if support in digits:
                    centre, capsule, mtp = digits[support]
                    base = float(capsule @ (mtp - centre))
                    tip = self.reach[int(geom)] * (1.0 if base <= 0 else -1.0)
                    digit_weight[support] += normal
                    digit_pos[support] += normal * (float(capsule @ (point - centre)) - base) / (tip - base)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.definition.append(np.where(weight > 1e-9, ahead / weight, np.nan))
            self.moment_over_force.append(np.where(weight > 1e-9, lever / weight, np.nan))
            self.load_pos.append(np.where(digit_weight > 1e-9, digit_pos / digit_weight, np.nan))
        self.bound.append(bound)
        self.rows = []


def _whole_foot_rollout(envs, steps: int, *, jitter: float = 0.0, settle: int = 100):
    """A T. rex statue episode (seed 3042) through the recorder with :class:`WholeFootProbe` under it.

    After *settle* zero-action steps the command gets N(0, *jitter*) noise on every actuator, every step.
    """
    env = envs(TYRANNOSAURUS_REX)
    morphology = Morphology.from_env(env, TYRANNOSAURUS_REX)
    probe = WholeFootProbe(env, morphology)
    env._substep_probe_hook = probe
    rng = np.random.default_rng(7)
    with SubstepContactRecorder(env, morphology) as recorder:
        env.reset(seed=3042)
        for step in range(steps):
            action = np.zeros(env.model.nu)
            if step >= settle and jitter:
                action = np.clip(rng.normal(0.0, jitter, env.model.nu), -1.0, 1.0)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            recorder.end_step(action, 0.0)
            probe.end_step()
            if terminated or truncated:
                break
        trace = recorder.end_episode(terminated=bool(terminated), truncated=bool(truncated))
    return trace, probe, morphology


@pytest.mark.parametrize("jitter", [0.0, 0.1], ids=["statue", "jittered"])
def test_the_whole_foot_cop_is_its_definition_and_the_vertical_load_moment_about_the_ankle(envs, jitter):
    """``foot_cop`` is the force-weighted lever arm of every limb contact along the sole, from the ankle axis.

    Exact against the independent probe (the ankle's world anchor, ``mj_contactForce``), and within the stated
    tolerance of the vertical load's moment about the ankle axis divided by the limb's floor force: a lever arm
    measured along the sole, not along the floor, differs from it by at most ``|d . n| sin(tilt) + |d . f| (1 -
    cos(tilt))`` per contact (``d`` from the ankle anchor, ``n`` / ``f`` the sole's normal and forward axes, tilt
    the sole's angle from level), about the ankle's 0.23 m height times the sine of the tilt.  The bound is exact:
    in the sole frame the vertical's lever about the lateral ankle axis is ``v_n d_f - v_f d_n``, with the
    vertical's normal component ``v_n = cos(tilt)`` and its forward one ``|v_f| <= sin(tilt)``.  Not the ankle
    actuator's torque over the load, which also carries the floor's shear times that height.
    """
    trace, probe, morphology = _whole_foot_rollout(envs, 300, jitter=jitter)
    definition = np.array(probe.definition)
    moment = np.array(probe.moment_over_force)
    bound = np.array(probe.bound)
    assert np.array_equal(np.isnan(trace.foot_cop), np.isnan(definition))
    np.testing.assert_allclose(trace.foot_cop, definition, rtol=0, atol=1e-12)
    assert np.isfinite(trace.foot_cop[100:]).all()  # both feet stand on every step after the settle
    difference = np.abs(trace.foot_cop - moment)
    assert np.all(difference[np.isfinite(difference)] <= bound[np.isfinite(difference)] + 1e-12)
    if not jitter:
        # On the statue after the settle (the sole within about 1.3 degrees of level) the two agree within 2 mm
        # (measured 1.0, median 0.5, on the physics-r9 pad).  Under N(0, 0.1) command noise the sole tilts up to
        # about 5 degrees and they differ by up to 21 mm, inside the bound on every step.
        assert np.nanmax(difference[100:]) < 0.002
    # The span is the registry's, and the trace's whole-foot CoP sits well inside it on a statue.
    assert trace.foot_support_span.tolist() == [[0.06, 0.213], [0.06, 0.213]]
    assert 0.12 < np.nanmean(trace.foot_cop[100:]) < 0.17


def test_each_hinged_digits_load_position_is_reproduced_and_other_support_geoms_have_none(envs):
    trace, probe, morphology = _whole_foot_rollout(envs, 200)
    reference = np.array(probe.load_pos)
    hinged = [support for support, joint in enumerate(morphology.support_geom_mtp) if joint >= 0]
    assert [trace.support_geom_names[support] for support in hinged] == [
        f"{side}_toe_{digit}_geom" for side in "rl" for digit in ("d2", "d3", "d4")
    ]
    np.testing.assert_allclose(trace.support_load_pos[:, hinged], reference[:, hinged], rtol=0, atol=1e-12)
    others = [support for support in range(len(morphology.support_geom_ids)) if support not in hinged]
    assert np.isnan(trace.support_load_pos[:, others]).all()  # the pads
    # On the r10 statue the digits carry their load about halfway out (median 0.45; on the r9 statue near their
    # MTP joints, about 0.05-0.08 of their length).
    assert 0.3 < np.nanmedian(trace.support_load_pos[100:, hinged]) < 0.7


def test_a_foot_that_declares_no_ankle_has_no_whole_foot_cop(envs):
    """The anatomical compsognathus: a box pad and digits, but no declared span (its digits share the pad's body)."""
    env = envs(COMPSOGNATHUS_LONGIPES)
    morphology = Morphology.from_env(env, COMPSOGNATHUS_LONGIPES)
    zero = np.zeros(env.model.nu)
    with SubstepContactRecorder(env, morphology) as recorder:
        env.reset(seed=3042)
        for _ in range(30):
            env.step(zero)
            recorder.end_step(zero, 0.0)
        trace = recorder.end_episode(truncated=True)
    assert np.isnan(trace.foot_cop).all() and np.isnan(trace.foot_support_span).all()
    assert np.isnan(trace.support_load_pos).all()
    assert np.isfinite(trace.sole_cop[10:, :, 0]).all()  # the pad CoP is measured as before
    assert math.isnan(episode_stance_metrics(trace, settle_steps=10).max_foot_cop_fore_aft)


def test_the_statue_whole_foot_cop_reading(envs):
    """Seed 3042 of the T. rex statue panel, the full 1,000 steps: the whole-foot reading on the physics-r10 span
    0.060-0.213 m (the sole box's heel edge and the held-offset toe line).

    The foot proposal's own per-substep probe of this episode (written before this recorder, from its contact
    buffers and the ankle's world anchor, its series stored in float32; not in the repository) read 0.2046 (right
    foot) and 0.0608 (left), signed +0.2046 and +0.0126, with the whole-foot CoP 0.1522 m ahead of the right
    ankle on average, all within 3e-9 of the values pinned here.  On the physics-r9 pad and its span
    0.070-0.210 m the same seed read 0.1304 and 0.0458, 0.149 m ahead.
    """
    from environments.shared.gait.pad_telemetry import PadLayout, episode_pad_telemetry

    env = envs(TYRANNOSAURUS_REX)
    morphology = Morphology.from_env(env, TYRANNOSAURUS_REX)
    zero = np.zeros(env.model.nu)
    with SubstepContactRecorder(env, morphology) as recorder:
        env.reset(seed=3042)
        for _ in range(1000):
            env.step(zero)
            recorder.end_step(zero, 0.0)
        trace = recorder.end_episode(truncated=True)
    pads = episode_pad_telemetry(trace, PadLayout.from_morphology(morphology), settle_steps=200)
    assert pads["r_foot_cop_fore_aft_gate"] == pytest.approx(0.2045782572700054, abs=1e-6)
    assert pads["l_foot_cop_fore_aft_gate"] == pytest.approx(0.0608003291318380, abs=1e-6)
    assert pads["r_foot_cop_fore_aft"] == pytest.approx(0.2045782572700054, abs=1e-6)
    assert pads["l_foot_cop_fore_aft"] == pytest.approx(0.0126177583138373, abs=1e-6)
    assert np.mean(trace.foot_cop[200:, 0]) == pytest.approx(0.1521502366811554, abs=1e-6)
    assert episode_stance_metrics(trace, settle_steps=200).max_foot_cop_fore_aft == pads["r_foot_cop_fore_aft_gate"]


# ── the six statues put their floor load on the registry ─────────────────────


#: The most each statue's pads read of ``max_sole_cop_fore_aft`` on this episode (D-D28; seed 3042, steps 1 s
#: to 1 s + 150): trex 0.588 on its r10 sole box, where the key is report-only (D-D34; 0.441 on the r9 pad, 0.356
#: on r8, before the elliptic cone of D-D29), compsognathus
#: 0.173 on its r2 plant (heel side; 0.464 on r1, before D-D26), its robot 0.319, dibothrosuchus 0.758 on its r2
#: plant (0.699 on r1, its hind feet heel side on four contacts, before the elliptic cone of D-D30).  None where the
#: foot has no box sole, so the metric is unmeasured: velociraptor (no sole) and brachiosaurus (ellipsoids, whose
#: one contact sits where the authored pitch puts it).
STATUE_COP_FORE_AFT_CEILING = {
    TYRANNOSAURUS_REX: 0.60,
    VELOCIRAPTOR_MONGOLIENSIS: None,
    COMPSOGNATHUS_LONGIPES: 0.20,
    COMPSOGNATHUS_LONGIPES_ROBOT: 0.35,
    DIBOTHROSUCHUS_ELAPHROS: 0.78,
    BRACHIOSAURUS_ALTITHORAX: None,
}


def test_the_statue_tables_cover_every_model():
    """A new model fails here, naming the table, instead of going untested (review M4)."""
    assert set(MODEL_IDS) == set(model_ids()), "MODEL_IDS must list every manifest model"
    assert set(STATUE_COP_FORE_AFT_CEILING) == set(model_ids()), "STATUE_COP_FORE_AFT_CEILING must pin every model"


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=lambda m: f"{m.species}-{m.embodiment}")
def test_each_statue_stands_on_its_registered_support_geoms(envs, model_id):
    env = envs(model_id)
    settle = int(round(1.0 / env.dt))  # 1 s
    zero = np.zeros(env.model.nu)
    with SubstepContactRecorder(env, Morphology.from_env(env, model_id)) as recorder:
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
    ceiling = STATUE_COP_FORE_AFT_CEILING[model_id]
    if ceiling is None:
        assert math.isnan(result.max_sole_cop_fore_aft)
    else:
        assert 0.0 < result.max_sole_cop_fore_aft <= ceiling


@pytest.mark.parametrize(
    ("model_id", "seed", "airborne"),
    [
        (DIBOTHROSUCHUS_ELAPHROS, 3045, 8.0),
        (COMPSOGNATHUS_LONGIPES_ROBOT, 3052, 3.0),
        (COMPSOGNATHUS_LONGIPES, 3045, 0.0),
    ],
    ids=lambda value: f"{value.species}-{value.embodiment}" if isinstance(value, ModelId) else None,
)
def test_the_spawn_pop_is_counted_substep_by_substep(envs, model_id, seed, airborne):
    """The reset pop ``constants.SPAWN_GRACE_S`` exists for, pinned at the panel seeds that show it.

    Every leg's floor force at or below ``CONTACT_THRESHOLD_N`` on a substep
    is one airborne substep: the first two statues leave the floor entirely
    for 8 and 3 substeps after the reset (the largest counts their 40-episode
    panels record, seeds 3042-3081; the dibothrosuchus's was 9 before the
    elliptic cone of its physics r2, D-D30) and never inside the settle
    window.
    Exact, because the count IS the threshold's meaning: read at 1.0 N
    instead of 0.1 N, the anatomical compsognathus's one-substep pop on
    physics r1 (seed 3045) counted 2.  Its physics r2 toe armature removed
    the pop: 0 airborne substeps on every panel seed, pinned at the seed
    that showed it.
    """
    env = envs(model_id)
    settle = int(round(1.0 / env.dt))
    zero = np.zeros(env.model.nu)
    with SubstepContactRecorder(env, Morphology.from_env(env, model_id)) as recorder:
        env.reset(seed=seed)
        for _ in range(settle + 20):
            env.step(zero)
            recorder.end_step(zero, 0.0)
        trace = recorder.end_episode(truncated=True)
    result = episode_stance_metrics(trace, settle_steps=settle)
    assert result.spawn_airborne_substeps == airborne
    assert result.settle_airborne_substeps == 0.0
