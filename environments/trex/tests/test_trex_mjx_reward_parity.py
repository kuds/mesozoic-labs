"""Focused parity coverage for the T. rex Stage-1 stance rewards."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco.mjx")


def _stance_only_env():
    from environments.trex.envs.trex_env import TRexEnv

    return TRexEnv(
        reset_noise_scale=0.0,
        forward_vel_weight=0.0,
        alive_bonus=1.0,
        energy_penalty_weight=0.0,
        tail_stability_weight=0.0,
        bite_bonus=0.0,
        bite_approach_weight=0.0,
        bite_head_proximity_weight=0.0,
        posture_weight=0.0,
        nosedive_weight=0.0,
        height_weight=0.6,
        height_target_tolerance=0.06,
        gait_symmetry_weight=0.0,
        smoothness_weight=0.0,
        heading_weight=0.0,
        lateral_penalty_weight=0.0,
        backward_vel_penalty_weight=0.0,
        drift_penalty_weight=0.0,
        spin_penalty_weight=0.0,
        speed_penalty_weight=0.0,
        idle_penalty_weight=0.0,
        foot_contact_weight=0.0,
        foot_contact_gate=0.0,
        bilateral_support_weight=0.6,
        foot_contact_saturation_force=350.0,
        foot_load_balance_weight=0.3,
        support_conditioned_alive_fraction=0.5,
        leg_home_pose_weight=0.5,
        leg_home_pose_tolerance=0.10,
        head_clearance_weight=0.35,
        head_clearance_target=0.90,
        head_clearance_tolerance=0.15,
        neck_posture_weight=0.20,
        neck_posture_tolerance=0.35,
    )


def test_jax_reward_composer_matches_gymnasium_stance_components():
    """Both backends must score an identical MuJoCo state identically."""
    from environments.shared.jax_reward_termination import compute_reward_components

    env = _stance_only_env()
    try:
        env.reset(seed=7)
        mujoco.mj_forward(env.model, env.data)
        _, sb3_info = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))

        reward_cfg = {
            "forward_vel_weight": 0.0,
            "alive_bonus": env.alive_bonus,
            "energy_penalty_weight": 0.0,
            "posture_weight": 0.0,
            "foot_contact_weight": 0.0,
            "bilateral_support_weight": env.bilateral_support_weight,
            "foot_contact_saturation_force": env.foot_contact_saturation_force,
            "foot_load_balance_weight": env.foot_load_balance_weight,
            "support_conditioned_alive_fraction": env.support_conditioned_alive_fraction,
            "leg_home_pose_weight": env.leg_home_pose_weight,
            "leg_home_pose_tolerance": env.leg_home_pose_tolerance,
            "head_clearance_weight": env.head_clearance_weight,
            "head_clearance_target": env.head_clearance_target,
            "head_clearance_tolerance": env.head_clearance_tolerance,
            "neck_posture_weight": env.neck_posture_weight,
            "neck_posture_tolerance": env.neck_posture_tolerance,
            "height_weight": env.height_weight,
            "height_target_tolerance": env.height_target_tolerance,
        }
        components = compute_reward_components(
            env.data,
            np.zeros(env.action_space.shape, dtype=np.float32),
            reward_cfg,
            root_body_id=env.pelvis_id,
            healthy_z_min=env.healthy_z_range[0],
            healthy_z_max=env.healthy_z_range[1],
            target_standing_z=0.9260,
            max_tilt_angle=env.max_tilt_angle,
            natural_forward_z=env._natural_forward_z,
            n_actuators=env.model.nu,
            foot_sensor_groups=(
                (env._sensor_r_foot, *env._sensor_r_foot_digits),
                (env._sensor_l_foot, *env._sensor_l_foot_digits),
            ),
            leg_home_pose_qpos_indices=tuple(int(index) for index in env._leg_home_qpos_indices),
            leg_home_pose_targets=jnp.asarray(env._leg_home_qpos),
            neck_posture_qpos_indices=tuple(int(index) for index in env._neck_home_qpos_indices),
            neck_posture_targets=jnp.asarray(env._neck_home_qpos),
            head_clearance_site_id=env.head_tip_site_id,
        )

        reward_pairs = {
            "reward_alive": "alive",
            "reward_bilateral_support": "bilateral_support",
            "reward_foot_load_balance": "foot_load_balance",
            "reward_leg_home_pose": "leg_home_pose",
            "reward_head_clearance": "head_clearance",
            "reward_neck_posture": "neck_posture",
            "reward_height": "height",
        }
        diagnostic_pairs = {
            "raw_alive": "_raw_alive",
            "alive_gate": "_alive_gate",
            "bilateral_support_quality": "_bilateral_support_quality",
            "foot_load_imbalance": "_foot_load_imbalance",
            "leg_home_pose_error": "_leg_home_pose_error",
            "leg_home_pose_quality": "_leg_home_pose_quality",
            "head_tip_z": "_head_tip_z",
            "head_pelvis_rel_z": "_head_pelvis_rel_z",
            "head_clearance_quality": "_head_clearance_quality",
            "neck_posture_error": "_neck_posture_error",
            "neck_posture_quality": "_neck_posture_quality",
            "height_error": "_height_error",
            "height_quality": "_height_quality",
        }
        for sb3_key, jax_key in {**reward_pairs, **diagnostic_pairs}.items():
            assert float(components[jax_key]) == pytest.approx(sb3_info[sb3_key], abs=1e-6), sb3_key

        component_total = sum(float(value) for key, value in components.items() if not key.startswith("_"))
        assert component_total == pytest.approx(sb3_info["reward_total"], abs=1e-6)
    finally:
        env.close()


def test_mjx_zero_defaults_preserve_p5_p7_policy_dimensions():
    """Reward-only plumbing must not move the p5 plant or p7 policy ABI."""
    import environments.trex.mjx_config  # noqa: F401
    from environments.shared.mjx_env import MJXDinoEnv

    env = MJXDinoEnv("trex", stage=1, num_envs=1)
    state = env.reset(jax.random.PRNGKey(9))
    weights = env.config.reward_weights

    assert env.action_dim == 15
    assert state.obs.shape == (1, 61)
    assert weights["bilateral_support_weight"] == 0.0
    assert weights["foot_load_balance_weight"] == 0.0
    assert weights["support_conditioned_alive_fraction"] == 0.0
    assert weights["leg_home_pose_weight"] == 0.0
    assert weights["head_clearance_weight"] == 0.0
    assert weights["neck_posture_weight"] == 0.0
    assert weights["height_target_tolerance"] == 0.0


def test_stage1_factory_enables_stance_terms_without_changing_dimensions():
    from environments.shared.jax_setup import create_env, setup_species

    ctx = setup_species("trex", stage=1)
    ctx.jax_kwargs = {
        **ctx.jax_kwargs,
        "reset_noise_scale": 0.0,
        "init_qpos_noise": 0.0,
        "init_yaw_noise": 0.0,
    }
    env = create_env(ctx, num_envs=1)
    state = env.reset(jax.random.PRNGKey(10))
    weights = env.config.reward_weights

    assert env.action_dim == 15
    assert state.obs.shape == (1, 61)
    assert weights["bilateral_support_weight"] == pytest.approx(0.6)
    assert weights["foot_contact_saturation_force"] == pytest.approx(350.0)
    assert weights["foot_load_balance_weight"] == pytest.approx(0.3)
    assert weights["support_conditioned_alive_fraction"] == pytest.approx(0.5)
    assert weights["leg_home_pose_weight"] == pytest.approx(0.5)
    assert weights["head_clearance_weight"] == pytest.approx(0.35)
    assert weights["neck_posture_weight"] == pytest.approx(0.20)
    assert weights["height_target_tolerance"] == pytest.approx(0.06)
    assert len(env._leg_home_pose_qpos_indices) == 8
    assert len(env._neck_posture_qpos_indices) == 3
    assert env._head_clearance_site_id is not None


@pytest.mark.parametrize("min_support_force", [0.0, 10.0])
def test_airborne_foot_load_balance_agrees_across_backends(min_support_force):
    """The ``[0, 0]`` case, which neither backend covered before.

    ``|R - L| / (R + L + 1e-8)`` is zero when both feet read zero, so an
    airborne plant scored as perfectly balanced -- cheaper than honest single
    support. Both paths must now report maximal imbalance, and must agree on
    where the unsupported threshold sits. See docs/STAGE1_SPLIT_PLAN.md
    section 7.1.
    """
    from environments.shared.reward_functions import reward_foot_load_balance
    from environments.trex.envs.trex_env import TRexEnv

    weight = 0.3
    env = TRexEnv(foot_load_balance_weight=weight, foot_load_balance_min_support_force=min_support_force)
    try:
        sb3_imbalance = env._foot_load_imbalance(0.0, 0.0)
    finally:
        env.close()

    jax_reward, jax_imbalance = reward_foot_load_balance(jnp.asarray([0.0, 0.0]), weight, min_support_force)
    numpy_reward, numpy_imbalance = reward_foot_load_balance(np.asarray([0.0, 0.0]), weight, min_support_force)

    assert sb3_imbalance == pytest.approx(1.0)
    assert float(jax_imbalance) == pytest.approx(1.0, abs=1e-6)
    assert float(numpy_imbalance) == pytest.approx(1.0, abs=1e-6)
    assert float(jax_reward) == pytest.approx(-weight, abs=1e-6)
    assert float(numpy_reward) == pytest.approx(-weight, abs=1e-6)


def test_grazing_contact_threshold_agrees_across_backends():
    """A token contact is denied credit identically on both paths."""
    from environments.shared.reward_functions import reward_foot_load_balance
    from environments.trex.envs.trex_env import TRexEnv

    forces = (0.25, 0.25)
    env = TRexEnv(foot_load_balance_weight=0.3, foot_load_balance_min_support_force=10.0)
    try:
        sb3_imbalance = env._foot_load_imbalance(*forces)
    finally:
        env.close()

    _, jax_imbalance = reward_foot_load_balance(jnp.asarray(forces), 0.3, 10.0)
    assert sb3_imbalance == pytest.approx(1.0)
    assert float(jax_imbalance) == pytest.approx(1.0, abs=1e-6)
