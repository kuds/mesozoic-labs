"""The anatomical Compsognathus's stance-quality reward terms (D-D26).

``CompsognathusBiologicalEnv`` adds reward kwargs on top of
``CompsognathusEnv``'s reward, every one inert at its default.  These tests
pin, without the stage TOML's weights:

1. **Legacy identity.**  With every new kwarg at its default the reward,
   the observation, the termination and every legacy info key equal
   ``CompsognathusEnv._get_reward_info`` on the same dynamics, bit for bit,
   for the subclass and, on the linear interface, for ``CompsognathusEnv``
   itself; and the subclass's constructor repeats every inherited default.
2. **Each term's arithmetic**, from the state the env read: the bilateral
   term and the alive conditioning on the chosen aggregate while the legacy
   support gate stays on the MIN; per-foot load gating of flatness and
   width; the spawn and settled width references; the support-geom coverage
   that refuses an MTP-end-only digit and counts a loaded digit tip, read
   after each substep's physics; the floor-impact and airborne terms on the
   substep block; saturation and jerk on the raw command.
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest

from environments.compsognathus.envs import CompsognathusBiologicalEnv, CompsognathusEnv
from environments.shared.config import load_stage_config
from environments.shared.species_registry import get_species_config

#: Info keys the legacy reward writes; the subclass rewrites only reward_alive and reward_total.
LEGACY_INFO_KEYS = (
    "reward_forward",
    "reward_energy",
    "reward_posture",
    "reward_height",
    "reward_home_pose",
    "reward_smoothness",
    "reward_spin",
    "reward_drift",
    "reward_lateral",
    "reward_heading",
    "reward_gait",
    "reward_approach",
    "reward_target",
    "r_foot_contact",
    "l_foot_contact",
    "pelvis_height",
    "tilt_angle",
    "drift_distance",
)


def _stance_kwargs(**overrides):
    return {**load_stage_config("compsognathus", "stance")["env_kwargs"], **overrides}


def _legacy_kwargs():
    """The stance task with every new kwarg at its inert default."""
    names = set(inspect.signature(CompsognathusBiologicalEnv).parameters) - set(
        inspect.signature(CompsognathusEnv).parameters
    )
    return {key: value for key, value in _stance_kwargs().items() if key not in names}


def _rollout(env, actions, seed):
    env.reset(seed=seed)
    out = []
    for action in actions:
        out.append(env.step(action))
        if out[-1][2] or out[-1][3]:
            break
    return out


@pytest.mark.parametrize("sigma", [0.0, 0.3])
def test_default_kwargs_reproduce_the_legacy_reward_bit_for_bit(sigma):
    actions = np.clip(sigma * np.random.default_rng(7).standard_normal((150, 14)), -1.0, 1.0)
    new = CompsognathusBiologicalEnv(**_legacy_kwargs())
    legacy = CompsognathusBiologicalEnv(**_legacy_kwargs())
    legacy._get_reward_info = types.MethodType(CompsognathusEnv._get_reward_info, legacy)
    linear = CompsognathusEnv(**_legacy_kwargs())
    linear_new = CompsognathusBiologicalEnv(**_legacy_kwargs())
    linear_new._scale_action = types.MethodType(CompsognathusEnv._scale_action, linear_new)
    for first, second in ((new, legacy), (linear_new, linear)):
        a, b = _rollout(first, actions, 3042), _rollout(second, actions, 3042)
        assert len(a) == len(b)
        for (obs_a, rew_a, term_a, trunc_a, info_a), (obs_b, rew_b, term_b, trunc_b, info_b) in zip(a, b):
            np.testing.assert_array_equal(obs_a, obs_b)
            assert rew_a == rew_b and info_a["reward_total"] == info_b["reward_total"]
            assert (term_a, trunc_a) == (term_b, trunc_b)
            assert info_a["reward_alive"] == info_b["reward_alive"]
            for key in LEGACY_INFO_KEYS:
                assert info_a[key] == info_b[key], key
        first.close()
        second.close()


def test_the_subclass_repeats_every_inherited_constructor_default():
    """The explicit signature is what the task fingerprint reads, so it must not drift from the parent's."""
    parent = inspect.signature(CompsognathusEnv).parameters
    child = inspect.signature(CompsognathusBiologicalEnv).parameters
    assert set(parent) <= set(child)
    for name, param in parent.items():
        assert child[name].default == param.default, name


@pytest.fixture
def env():
    instance = CompsognathusBiologicalEnv(**_stance_kwargs())
    yield instance
    instance.close()


def _step_with_block(env, block, minimum, action=None):
    """Score one step with the substep block and MIN forces replaced by hand-built values."""
    env._substep_foot_force_block = lambda: np.asarray(block, dtype=np.float64)
    env._aggregated_foot_contact_forces = lambda: tuple(float(value) for value in minimum)
    action = np.zeros(env.action_space.shape) if action is None else action
    return env._get_reward_info(action)


def test_the_bilateral_and_alive_terms_read_the_mean_while_the_legacy_gate_keeps_the_min():
    """Both feet off the floor on one substep in ten: MIN reads zero, MEAN nine tenths of the load."""
    block = np.full((10, 2), 4.9)
    block[3] = 0.0
    weights = dict(
        bilateral_support_weight=0.5, foot_contact_saturation_force=4.4, support_conditioned_alive_fraction=0.3
    )
    for aggregation, quality in (("mean", 1.0), ("min", 0.0)):
        instance = CompsognathusBiologicalEnv(**_stance_kwargs(support_force_aggregation=aggregation, **weights))
        instance.reset(seed=1)
        reward, info = _step_with_block(instance, block, block.min(axis=0))
        assert info["bilateral_support_quality"] == pytest.approx(min(1.0, 0.9 * 4.9 / 4.4) if quality else 0.0)
        # The legacy support gate (both feet's MIN summed above 4% of body weight) is closed either way.
        assert info["raw_alive"] == 0.0 and info["reward_alive"] == 0.0 and info["reward_posture"] == 0.0
        assert info["reward_bilateral_support"] == pytest.approx(0.5 * info["bilateral_support_quality"])
        instance.close()


def test_alive_conditioning_scales_the_legacy_bonus_by_bilateral_support():
    instance = CompsognathusBiologicalEnv(
        **_stance_kwargs(support_conditioned_alive_fraction=0.3, support_force_aggregation="mean")
    )
    instance.reset(seed=1)
    block = np.tile([4.9, 2.2], (10, 1))  # the weaker foot at half the saturation force
    _, info = _step_with_block(instance, block, block.min(axis=0))
    assert info["raw_alive"] == instance.alive_bonus
    assert info["alive_gate"] == pytest.approx(0.7 + 0.3 * 2.2 / instance.foot_contact_saturation_force)
    assert info["reward_alive"] == pytest.approx(info["raw_alive"] * info["alive_gate"])
    instance.close()


def test_flatness_and_width_are_paid_only_on_a_loaded_foot():
    instance = CompsognathusBiologicalEnv(
        **_stance_kwargs(
            foot_flatness_weight=0.3,
            stance_width_weight=0.1,
            stance_width_reference="spawn",  # paid from the first step
            foot_terms_min_support_force=2.45,
        )
    )
    instance.reset(seed=1)
    _, loaded = _step_with_block(instance, np.full((10, 2), 4.9), (4.9, 4.9))
    tilts = instance._sole_tilts_deg()
    per_foot = np.exp(-np.square(tilts / instance.foot_flatness_tolerance_deg))
    assert loaded["foot_flatness_quality"] == pytest.approx(per_foot.mean())
    assert loaded["stance_width_quality"] > 0.9
    # The left foot carries 2 N: it forfeits its flatness half, and the pair the width.
    _, light = _step_with_block(instance, np.tile([4.9, 2.0], (10, 1)), (4.9, 2.0))
    assert light["foot_flatness_quality"] == pytest.approx(per_foot[0] / 2)
    assert light["stance_width_quality"] == 0.0 and light["reward_stance_width"] == 0.0
    instance.close()


def test_the_width_target_is_the_spawn_or_the_settled_width_never_the_keyframe():
    common = dict(stance_width_weight=0.1, foot_terms_min_support_force=0.0)
    spawn = CompsognathusBiologicalEnv(**_stance_kwargs(stance_width_reference="spawn", **common))
    settled = CompsognathusBiologicalEnv(
        **_stance_kwargs(stance_width_reference="settled", stance_width_settle_steps=5, **common)
    )
    zero = np.zeros(spawn.action_space.shape)
    for seed in (3, 4):
        spawn.reset(seed=seed)
        settled.reset(seed=seed)
        # The spawn reference is the width the reset produced, before any step.
        assert spawn._stance_width_target == pytest.approx(spawn._stance_width(spawn.data))
        for step in range(1, 8):
            _, _, _, _, info = spawn.step(zero)
            assert info["stance_width_target"] == spawn._spawn_stance_width
            _, _, _, _, late = settled.step(zero)
            if step <= 5:
                # Nothing is paid in the settle, so re-seating the feet there earns nothing.
                assert late["reward_stance_width"] == 0.0
            if step == 5:
                target = late["stance_width"]
            if step > 5:
                assert late["stance_width_target"] == target and late["reward_stance_width"] > 0.0
    spawn.close()
    settled.close()


def _floor_truth_split(env):
    """Each support geom's floor normal force, split by how far each contact lies from the MTP joint.

    Independent of the env's own classification (no capsule frame): a digit contact is distal when it lies
    farther from the joint (the origin of the foot body the digits hang from) than the capsule's centre does.
    Returns (all contacts, distal contacts only, pads always counted), one float per support slot.
    """
    data, model = env.data, env.model
    slots = {int(geom): slot for slot, geom in enumerate(env._support_geom_ids)}
    every, distal = np.zeros(len(slots)), np.zeros(len(slots))
    for index in range(data.ncon):
        contact = data.contact[index]
        if env.floor_geom_id not in (contact.geom1, contact.geom2):
            continue
        geom = contact.geom2 if contact.geom1 == env.floor_geom_id else contact.geom1
        if geom not in slots or contact.efc_address < 0:
            continue
        force = float(data.efc_force[contact.efc_address])
        every[slots[geom]] += force
        joint = data.xpos[model.geom_bodyid[geom]]
        far = np.linalg.norm(contact.pos - joint) > np.linalg.norm(data.geom_xpos[geom] - joint)
        if far or not env._support_geom_is_digit[slots[geom]]:
            distal[slots[geom]] += force
    return every, distal


def _toe_command(env, value):
    action = np.zeros(env.action_space.shape)
    for side in "rl":
        action[env.model.actuator(side + "_toe_act").id] = value
    return action


def test_coverage_refuses_a_digit_that_touches_only_with_its_mtp_end():
    """Toes raised (a positive toe command lifts the digit tips): the pad and the digits' joint ends carry the foot."""
    instance = CompsognathusBiologicalEnv(**_stance_kwargs(support_geom_coverage_weight=0.5))
    instance.reset(seed=3042)
    action = _toe_command(instance, 0.2)
    for _ in range(10):
        _, _, terminated, _, info = instance.step(action)
        assert not terminated
    every, distal = _floor_truth_split(instance)
    naive, loaded = every > 0.1, distal > 0.1
    # Some digit carries load through its joint end alone: any-contact coverage would count it.
    assert (naive & ~loaded & instance._support_geom_is_digit).any()
    np.testing.assert_array_equal(instance._support_geoms_loaded(), loaded)
    assert info["support_geom_coverage"] < naive.mean()
    assert info["reward_support_geom_coverage"] == pytest.approx(-0.5 * (1.0 - info["support_geom_coverage"]))
    instance.close()


def test_coverage_counts_a_digit_that_carries_the_foot_on_its_tip():
    """Toes pressed down (the digit-III tiptoe of the certified r1 marches): each foot stands on its digit-III tip."""
    instance = CompsognathusBiologicalEnv(**_stance_kwargs(support_geom_coverage_weight=0.5))
    instance.reset(seed=3042)
    action = _toe_command(instance, -0.6)
    for _ in range(30):
        instance.step(action)
    every, distal = _floor_truth_split(instance)
    digit_three = np.array([name == "toe_d3_geom" for _ in "rl" for name in instance.support_geoms])
    assert np.all(distal[digit_three] > 1.0)  # each tip carries over a newton
    np.testing.assert_array_equal(instance._support_geoms_loaded(), distal > 0.1)
    assert instance._support_geoms_loaded()[digit_three].all()
    instance.close()


def test_coverage_sums_each_geoms_contacts_before_the_threshold():
    """As the floor-truth recorder does: two 0.06 N contacts on one geom load it, one does not."""
    instance = CompsognathusBiologicalEnv(**_stance_kwargs(support_geom_coverage_weight=0.5))
    instance.reset(seed=1)
    centre = instance.data.geom_xpos[instance._support_geom_ids].copy()
    frame = instance.data.geom_xmat[instance._support_geom_ids].copy()
    pad, right_d3 = 0, 2
    # The right digit-III contact sits 20 mm toward its tip from the capsule centre.
    tip = centre[right_d3] + 0.02 * frame[right_d3][2::3] * instance._support_geom_tipward[right_d3]
    record = (
        np.array([pad, pad, right_d3, right_d3]),
        np.array([0.06, 0.06, 0.06, 0.03]),
        np.stack([centre[pad], centre[pad], tip, tip]),
        centre,
        frame,
    )
    loaded = instance._support_geoms_loaded_on([record])[0]
    assert loaded[pad] and not loaded[right_d3]
    assert loaded.sum() == 1
    instance.close()


def test_coverage_counts_the_substeps_after_each_physics_step():
    """The step's counts are its own ten substeps, each read after its mj_step (the hook's place in step())."""
    instance = CompsognathusBiologicalEnv(**_stance_kwargs(support_geom_coverage_weight=0.5))
    instance.reset(seed=3042)
    seen = []
    instance._substep_probe_hook = lambda: seen.append(_floor_truth_split(instance)[1] > 0.1)
    rng = np.random.default_rng(5)
    changed = 0
    for _ in range(40):
        seen.clear()
        instance.step(np.clip(0.3 * rng.standard_normal(instance.action_space.shape), -1.0, 1.0))
        assert len(seen) == instance.frame_skip
        np.testing.assert_array_equal(instance._support_geom_counts(), np.sum(seen, axis=0))
        changed += int(np.any(np.sum(seen, axis=0) % instance.frame_skip))
    assert changed  # some step's geoms load and unload within it, so a misplaced hook would read other substeps
    instance.close()


def test_coverage_is_the_gates_substep_fraction_and_full_on_the_settled_statue():
    instance = CompsognathusBiologicalEnv(**_stance_kwargs(support_geom_coverage_weight=0.5))
    instance.reset(seed=3042)
    # The settle seats the digit tips last (on this seed every geom is loaded from step 176).
    for _ in range(200):
        _, _, _, _, info = instance.step(np.zeros(instance.action_space.shape))
    assert instance._support_geom_counts_step == instance._step_count
    assert len(instance._support_geom_records) == instance.frame_skip
    assert info["support_geom_coverage"] == 1.0 and info["reward_support_geom_coverage"] == 0.0
    # The batched classification of the step's substeps equals classifying each substep on its own.
    per_substep = np.array(
        [instance._support_geoms_loaded_on([record])[0] for record in instance._support_geom_records]
    )
    np.testing.assert_array_equal(instance._support_geoms_loaded_on(instance._support_geom_records), per_substep)
    # A geom loaded on 4 of 10 substeps does not count; on 5 it does.
    counts = np.full(8, 10)
    counts[1], counts[6] = 4, 5
    instance._support_geom_counts = lambda: counts
    np.testing.assert_allclose(instance._support_geom_coverage(), [0.75, 1.0])
    instance.close()


def test_floor_impact_and_airborne_substeps_price_the_whole_block():
    instance = CompsognathusBiologicalEnv(
        **_stance_kwargs(floor_impact_weight=2.0, floor_impact_threshold_bw=1.5, airborne_substep_weight=1.0)
    )
    instance.reset(seed=1)
    weight = instance._body_weight_n
    block = np.full((10, 2), weight / 2)
    block[2:4] = 0.0  # two airborne substeps
    block[6] = (weight, 1.5 * weight)  # a 2.5 BW landing
    _, info = _step_with_block(instance, block, block.min(axis=0))
    assert info["peak_foot_force_bw"] == pytest.approx(2.5)
    assert info["airborne_substeps"] == 2
    assert info["reward_floor_impact"] == pytest.approx(-2.0 * (2.5 - 1.5))
    assert info["reward_airborne_substeps"] == pytest.approx(-1.0 * 2 / 10)
    _, quiet = _step_with_block(instance, np.full((10, 2), weight / 2), (weight / 2, weight / 2))
    assert quiet["reward_floor_impact"] == 0.0 and quiet["reward_airborne_substeps"] == 0.0
    # One foot unloaded is single support, not flight: airborne counts substeps with BOTH feet off the floor.
    single = np.full((10, 2), weight)
    single[1:6, 0] = 0.0
    single[7, 1] = 0.0
    _, one_foot = _step_with_block(instance, single, single.min(axis=0))
    assert one_foot["airborne_substeps"] == 0 and one_foot["reward_airborne_substeps"] == 0.0
    # The peak is the summed floor force's on one substep, not the sum of each foot's own maximum.
    staggered = np.full((10, 2), weight / 2)
    staggered[2, 0] = 2.0 * weight
    staggered[5, 1] = 2.0 * weight
    _, offset = _step_with_block(instance, staggered, staggered.min(axis=0))
    assert offset["peak_foot_force_bw"] == pytest.approx(2.5)
    assert offset["reward_floor_impact"] == pytest.approx(-2.0 * (2.5 - 1.5))
    # A crushing contact at a fall charges at most one body weight of excess: the fall penalty prices the fall.
    block[6] = (20.0 * weight, 20.0 * weight)
    _, slam = _step_with_block(instance, block, block.min(axis=0))
    assert slam["peak_foot_force_bw"] == pytest.approx(40.0)
    assert slam["reward_floor_impact"] == pytest.approx(-2.0 * instance._FLOOR_IMPACT_EXCESS_CAP_BW)
    instance.close()


def test_the_support_geoms_are_the_floor_truth_registry_entry():
    """The env names the gate's support set itself (no species env imports the gait package); they must agree."""
    from environments.shared.gait.morphology import SUPPORT_REGISTRY

    entry = SUPPORT_REGISTRY["compsognathus"]
    assert tuple("{s}_" + name for name in CompsognathusBiologicalEnv.support_geoms) == entry.support
    assert entry.sole == "{s}_plantar_pad" and entry.foot_site == "{s}_foot_touch_volume"


def test_saturation_and_jerk_price_the_raw_command():
    instance = CompsognathusBiologicalEnv(
        **_stance_kwargs(action_saturation_weight=0.5, action_saturation_threshold=0.9, action_jerk_weight=1.0)
    )
    assert instance.action_filter_cutoff_hz == 0.0  # no filter in front of the penalties
    instance.reset(seed=1)
    actions = [np.zeros(14), np.full(14, 0.5), np.full(14, -1.0)]
    infos = [instance.step(action)[4] for action in actions]
    assert infos[0]["reward_action_jerk"] == 0.0 and infos[1]["reward_action_jerk"] == 0.0
    second_difference = -1.0 - 2 * 0.5 + 0.0
    assert infos[2]["reward_action_jerk"] == pytest.approx(-1.0 * 14 * second_difference**2 / (14 * 16.0))
    assert infos[1]["reward_action_saturation"] == 0.0
    assert infos[2]["reward_action_saturation"] == pytest.approx(-0.5)
    instance.close()


def test_leg_home_pose_is_full_at_the_keyframe(env):
    env.reset(seed=1)
    env.data.qpos[:] = env._home_qpos
    _, info = env._get_reward_info(np.zeros(env.action_space.shape))
    assert info["leg_home_pose_quality"] == 1.0 and info["leg_home_pose_error"] == 0.0


@pytest.mark.parametrize(
    "bad",
    [
        {"support_force_aggregation": "median"},
        {"stance_width_reference": "keyframe"},
        {"stance_width_settle_steps": 0},
        {"stance_width_settle_steps": 2.5},
        {"stance_width_settle_steps": True},
        {"foot_contact_saturation_force": 0.0},
        {"support_conditioned_alive_fraction": 1.5},
        {"floor_impact_threshold_bw": -1.0},
        {"airborne_substep_weight": -0.1},
        {"support_geom_coverage_weight": float("nan")},
        {"action_saturation_threshold": 1.0},
    ],
)
def test_invalid_stance_terms_are_refused(bad):
    with pytest.raises(ValueError):
        CompsognathusBiologicalEnv(**bad)


#: The robot's stage task digests on main before the anatomical revision (configs/digest_snapshot.generated.txt
#: at 46e0b7c).  The stance-quality kwargs live on CompsognathusBiologicalEnv alone, so the robot's constructor, and
#: with it every robot task fingerprint, must not move.
ROBOT_TASK_SHA256 = {
    "stance": "sha256:26a8ec2b563e7bdc3808085b52edf0e8e0a785088d5f102f18274d4ab378c36c",
    "recovery": "sha256:84414f4c9c4d3544479c5fe21f7fe5fd56540cfa244a006bc264054d6eb6d98d",
    "locomotion": "sha256:ae9b535cdbab8bb95fc2e9482c6b119d0c68c4c34047fb2f7f245735bb31707b",
    "behavior": "sha256:5e42c9c772ff91e11040d65b15222a2463b24151c361360776df7746361ec045",
}


@pytest.mark.parametrize("stage", sorted(ROBOT_TASK_SHA256))
def test_the_robot_tasks_did_not_move_with_the_anatomical_reward(stage):
    from environments.shared.task_fingerprint import stage_task_fingerprint

    assert stage_task_fingerprint("compsognathus_robot", stage)["task_sha256"] == ROBOT_TASK_SHA256[stage]
    parameters = inspect.signature(get_species_config("compsognathus_robot").env_class).parameters
    assert "support_geom_coverage_weight" not in parameters and "bilateral_support_weight" not in parameters
