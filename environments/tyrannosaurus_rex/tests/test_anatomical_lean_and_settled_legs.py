"""The T. rex's two opt-in stance options: the standing lateral lean term (lateral_lean_weight,
lateral_lean_tolerance_m) and the settled leg target (leg_home_pose_reference = "settled").

Both are off by default, and off they change nothing: no reward, no info key, no task fingerprint and no recorded
stage config.  On, the lean term is a Gaussian on the centre of mass's lateral offset over the feet, gated on the
settle, a zero command and both feet loaded, and exactly mirror-invariant; the settled target is the statue's own
leg pose, mirror-symmetrised.
"""

from __future__ import annotations

import inspect
import json
import math

import mujoco
import numpy as np
import pytest

from environments.shared import symmetry as SY
from environments.shared.config import load_stage_config, save_stage_config
from environments.shared.model_id import ModelId
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import _effective_env_kwargs, stage_task_fingerprint
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
OPTION_KEYS = ("lateral_lean_weight", "lateral_lean_tolerance_m", "leg_home_pose_reference")
LEAN_INFO_KEYS = {"com_lateral_offset_m", "lateral_lean_quality", "reward_lateral_lean"}
#: The values the 2026-10 T. rex stance screen trained with (the mirror loss, the lean term and the settled leg target).
SCREENED = {"lateral_lean_weight": 0.25, "lateral_lean_tolerance_m": 0.015, "leg_home_pose_reference": "settled"}


@pytest.fixture(scope="module")
def stance_kwargs():
    return dict(load_stage_config(TYRANNOSAURUS_REX, "stance")["env_kwargs"])


def _roll(kwargs, steps, seed=3042, amplitude=0.05):
    env = TyrannosaurusRexAnatomicalEnv(**kwargs)
    try:
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(steps):
            _, reward, terminated, truncated, info = env.step(rng.uniform(-amplitude, amplitude, 15))
            out.append((reward, info))
            if terminated or truncated:
                break
        return out
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- off by default
def test_the_options_default_to_off():
    parameters = inspect.signature(TyrannosaurusRexAnatomicalEnv).parameters
    assert {key: parameters[key].default for key in OPTION_KEYS} == {
        "lateral_lean_weight": 0.0,
        "lateral_lean_tolerance_m": 0.015,
        "leg_home_pose_reference": "keyframe",
    }
    for entry in load_stage_manifest(TYRANNOSAURUS_REX).stages:
        env_kwargs = load_stage_config(TYRANNOSAURUS_REX, entry.reference)["env_kwargs"]
        assert not set(OPTION_KEYS) & set(env_kwargs), entry.id


def test_off_the_options_change_no_reward_and_no_info_key(stance_kwargs):
    default = _roll(stance_kwargs, 240)
    explicit = _roll({**stance_kwargs, "lateral_lean_weight": 0.0, "lateral_lean_tolerance_m": 0.03}, 240)
    assert len(default) == len(explicit) == 240
    for (reward_a, info_a), (reward_b, info_b) in zip(default, explicit):
        assert reward_a == reward_b
        assert list(info_a) == list(info_b) and not LEAN_INFO_KEYS & set(info_a)
        assert all(info_a[key] == info_b[key] for key in info_a if isinstance(info_a[key], float))


def test_off_the_options_move_no_task_fingerprint_and_no_recorded_config(tmp_path, stance_kwargs):
    for entry in load_stage_manifest(TYRANNOSAURUS_REX).stages:
        committed = stage_task_fingerprint(TYRANNOSAURUS_REX, entry.reference)
        explicit_off = {**load_stage_config(TYRANNOSAURUS_REX, entry.reference)["env_kwargs"], "lateral_lean_weight": 0}
        explicit_off.update(lateral_lean_tolerance_m=0.05, leg_home_pose_reference="keyframe")
        assert stage_task_fingerprint(TYRANNOSAURUS_REX, entry.reference, env_kwargs=explicit_off) == committed, (
            entry.id
        )
        assert not set(OPTION_KEYS) & set(committed["env"])
    # On, the option and every kwarg it makes live are hashed, defaults included.
    lean_on = _effective_env_kwargs(TYRANNOSAURUS_REX, {**stance_kwargs, "lateral_lean_weight": 0.25})
    assert (lean_on["lateral_lean_weight"], lean_on["lateral_lean_tolerance_m"]) == (0.25, 0.015)
    assert "leg_home_pose_reference" not in lean_on
    settled = _effective_env_kwargs(TYRANNOSAURUS_REX, {**stance_kwargs, "leg_home_pose_reference": "settled"})
    assert settled["leg_home_pose_reference"] == "settled" and "lateral_lean_tolerance_m" not in settled
    on = stage_task_fingerprint(TYRANNOSAURUS_REX, "stance", env_kwargs={**stance_kwargs, "lateral_lean_weight": 0.25})
    wider = stage_task_fingerprint(
        TYRANNOSAURUS_REX,
        "stance",
        env_kwargs={**stance_kwargs, "lateral_lean_weight": 0.25, "lateral_lean_tolerance_m": 0.02},
    )
    assert (
        len(
            {
                on["task_sha256"],
                wider["task_sha256"],
                stage_task_fingerprint(TYRANNOSAURUS_REX, "stance")["task_sha256"],
            }
        )
        == 3
    )
    # stage_config.json records no off option's defaults, and every key the TOML sets.
    config = load_stage_config(TYRANNOSAURUS_REX, "stance")
    path = save_stage_config(
        tmp_path / "off", "stance", config, "PPO", env_class=TyrannosaurusRexAnatomicalEnv, model_id=TYRANNOSAURUS_REX
    )
    recorded = json.loads(path.read_text())["reward_weights"]
    assert not set(OPTION_KEYS) & set(recorded) and "stance_width_reference" in recorded
    config_on = {**config, "env_kwargs": {**config["env_kwargs"], "lateral_lean_weight": 0.25}}
    path = save_stage_config(
        tmp_path / "on", "stance", config_on, "PPO", env_class=TyrannosaurusRexAnatomicalEnv, model_id=TYRANNOSAURUS_REX
    )
    recorded = json.loads(path.read_text())["reward_weights"]
    assert (recorded["lateral_lean_weight"], recorded["lateral_lean_tolerance_m"]) == (0.25, 0.015)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lateral_lean_weight": -0.1}, "lateral_lean_weight"),
        ({"lateral_lean_weight": math.inf}, "lateral_lean_weight"),
        ({"lateral_lean_tolerance_m": 0.0}, "lateral_lean_tolerance_m"),
        ({"leg_home_pose_reference": "settled_sym"}, "leg_home_pose_reference"),
    ],
)
def test_invalid_option_values_fail_fast(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TyrannosaurusRexAnatomicalEnv(**kwargs)


# ---------------------------------------------------------------------------------------------------- the lean term
def _offset_by_hand(env) -> float:
    """The lateral offset recomputed from the rotation matrix MuJoCo keeps (not the quaternion the env reads)."""
    data = env.data
    com = data.subtree_com[env.pelvis_id]
    midpoint = (data.site_xpos[env.r_foot_site_id] + data.site_xpos[env.l_foot_site_id]) / 2
    x_axis = data.xmat[env.pelvis_id].reshape(3, 3)[:, 0]
    forward = x_axis[:2] / np.linalg.norm(x_axis[:2])
    left = np.array([-forward[1], forward[0]])
    return float((com[:2] - midpoint[:2]) @ left)


def test_the_lean_term_is_a_gated_gaussian_of_the_lateral_offset(stance_kwargs):
    weight, tolerance = 0.25, 0.015
    kwargs = {**stance_kwargs, "lateral_lean_weight": weight, "lateral_lean_tolerance_m": tolerance}
    env = TyrannosaurusRexAnatomicalEnv(**kwargs)
    try:
        env.reset(seed=3059)
        rng = np.random.default_rng(1)
        settle = env.stance_width_settle_steps
        paid = 0
        for step in range(1, settle + 60):
            _, reward, terminated, _, info = env.step(rng.uniform(-0.05, 0.05, 15))
            assert not terminated
            offset = info["com_lateral_offset_m"]
            assert offset == pytest.approx(_offset_by_hand(env), abs=1e-12)
            quality = math.exp(-((offset / tolerance) ** 2))
            assert info["lateral_lean_quality"] == pytest.approx(quality, rel=1e-12)
            supported = info["r_foot_terms_supported"] > 0.5 and info["l_foot_terms_supported"] > 0.5
            if step <= settle or not supported:
                assert info["reward_lateral_lean"] == 0.0
            else:
                assert info["reward_lateral_lean"] == pytest.approx(weight * quality, rel=1e-12)
                paid += 1
            assert reward == info["reward_total"]
            assert reward == pytest.approx(_total_of_the_other_terms(info) + info["reward_lateral_lean"], abs=1e-12)
        assert paid > 0 and abs(info["com_lateral_offset_m"]) > 1e-5  # a real, paid lean was compared
        # A live command switches the term off (the body-relative command is zero in every stance).
        env._command = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        _, info = env._get_reward_info(np.zeros(15))
        assert info["reward_lateral_lean"] == 0.0 and info["lateral_lean_quality"] > 0.0
    finally:
        env.close()


def _total_of_the_other_terms(info) -> float:
    """The total of every other term, in _get_reward_info's order."""
    names = (
        "reward_forward", "reward_backward", "reward_drift", "reward_alive", "reward_energy", "reward_tail",
        "reward_bite", "reward_approach", "reward_head_proximity", "reward_head_clearance", "reward_neck_posture",
        "reward_tail_home_pose", "reward_posture", "reward_nosedive", "reward_height", "reward_bilateral_support",
        "reward_foot_load_balance", "reward_leg_home_pose", "reward_foot_flatness", "reward_stance_width",
        "reward_gait", "reward_smoothness", "reward_action_jerk", "reward_action_saturation", "reward_heading",
        "reward_lateral", "reward_spin", "reward_speed", "reward_idle", "reward_floor_impact",
        "reward_airborne_substeps",
    )  # fmt: skip
    return float(sum(info[name] for name in names))


def test_the_lean_term_needs_both_feet_loaded(stance_kwargs):
    kwargs = {**stance_kwargs, "lateral_lean_weight": 0.25, "foot_terms_min_support_force": 1e6}
    rows = _roll(kwargs, 260)
    assert len(rows) == 260
    assert all(info["reward_lateral_lean"] == 0.0 for _, info in rows)


def test_the_lateral_offset_flips_under_the_mirror(stance_kwargs):
    env = TyrannosaurusRexAnatomicalEnv(**stance_kwargs, lateral_lean_weight=0.25)
    try:
        mirror = SY.build_env_mirror(env, SY.HEADING_FREE_OBSERVATION_SCHEMA)
        env.reset(seed=3059)
        for action in np.random.default_rng(2).uniform(-0.2, 0.2, (40, 15)):
            env.step(action)
        mujoco.mj_forward(env.model, env.data)
        before = env._com_lateral_offset()
        _, info_before = env._get_reward_info(np.zeros(15))
        SY.mirror_env_state(env, env, mirror)
        after = env._com_lateral_offset()
        _, info_after = env._get_reward_info(np.zeros(15))
        assert abs(before) > 1e-4
        assert after == pytest.approx(-before, abs=1e-12)
        assert info_after["lateral_lean_quality"] == pytest.approx(info_before["lateral_lean_quality"], abs=1e-12)
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- settled legs
def test_the_settled_leg_target_is_mirror_symmetric():
    settled = np.asarray(TyrannosaurusRexAnatomicalEnv._LEG_SETTLED_QPOS)
    env = TyrannosaurusRexAnatomicalEnv(leg_home_pose_reference="settled")
    try:
        qpos_map = SY.derive_model_mirror(env.model).qpos
        q = env.model.key_qpos[env.home_keyframe_id].copy()
        q[env._leg_home_qpos_indices] = settled
        np.testing.assert_array_equal(qpos_map.apply(q)[env._leg_home_qpos_indices], settled)
        keyframe = env.model.key_qpos[env.home_keyframe_id][env._leg_home_qpos_indices]
        np.testing.assert_array_equal(env._leg_home_qpos, keyframe)  # the keyframe target itself is untouched
        assert np.degrees(np.max(np.abs(settled - keyframe))) == pytest.approx(2.60, abs=0.01)
    finally:
        env.close()


def test_the_settled_reference_centres_leg_home_pose_on_the_settled_pose(stance_kwargs):
    env = TyrannosaurusRexAnatomicalEnv(**{**stance_kwargs, "leg_home_pose_reference": "settled"})
    keyframe_env = TyrannosaurusRexAnatomicalEnv(**stance_kwargs)
    try:
        for e in (env, keyframe_env):
            e.reset(seed=3042)
            e.data.qpos[e._leg_home_qpos_indices] = e._LEG_SETTLED_QPOS
            mujoco.mj_forward(e.model, e.data)
        _, info = env._get_reward_info(np.zeros(15))
        _, keyframe_info = keyframe_env._get_reward_info(np.zeros(15))
        assert info["leg_home_pose_error"] == 0.0 and info["leg_home_pose_quality"] == 1.0
        assert keyframe_info["leg_home_pose_quality"] < 1.0
        assert info["reward_leg_home_pose"] == stance_kwargs["leg_home_pose_weight"]
        # Every other term is the keyframe env's.
        assert {k: v for k, v in info.items() if "leg_home_pose" not in k and k != "reward_total"} == {
            k: v for k, v in keyframe_info.items() if "leg_home_pose" not in k and k != "reward_total"
        }
        # The target survives a re-cache of the model's ids (a behavior env swapping its compiled model).
        env._cache_ids()
        _, info = env._get_reward_info(np.zeros(15))
        assert info["leg_home_pose_quality"] == 1.0
    finally:
        env.close()
        keyframe_env.close()


def test_the_settled_leg_pose_is_the_statues(stance_kwargs):
    """Re-measured as measured (the stance config, zero action, seeds 3042-3051, mirror-symmetrised), over steps
    200-399 instead of 200-999: within 2.5 mrad (0.14 degree).  A plant change that moves the statue's legs fails
    this, and the constant is re-measured with it."""
    env = TyrannosaurusRexAnatomicalEnv(**stance_kwargs)
    try:
        means = []
        for seed in range(3042, 3052):
            env.reset(seed=seed)
            window = []
            for step in range(400):
                env.step(np.zeros(15, dtype=np.float32))
                if step >= 200:
                    window.append(env.data.qpos[env._leg_home_qpos_indices].copy())
            means.append(np.mean(window, axis=0))
        mean = np.mean(means, axis=0)
        sign = np.array([1.0, -1.0, 1.0, 1.0])
        symmetric = np.concatenate([(mean[:4] + sign * mean[4:]) / 2, sign * (mean[:4] + sign * mean[4:]) / 2])
        np.testing.assert_allclose(symmetric, env._LEG_SETTLED_QPOS, rtol=0.0, atol=2.5e-3)
    finally:
        env.close()
