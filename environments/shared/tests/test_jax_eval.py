"""Tests for jax_eval module (CPU-based evaluation for JAX-trained policies)."""

from __future__ import annotations

import numpy as np
import pytest

from environments.shared.jax_eval import (
    EvalConfig,
    EvalResults,
    _apply_eval_reset_randomization,
    _posture_reward_for_eval,
    check_stage_gate,
    check_stage_gate_for_config,
)


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.n_episodes == 25
        assert cfg.max_episode_steps == 1000
        assert cfg.frame_skip == 5
        assert cfg.healthy_z_range == (0.3, 2.0)
        assert cfg.max_tilt_angle == pytest.approx(1.047)
        assert cfg.posture_target_forward_z is None
        assert cfg.root_body_id == 1
        assert cfg.sensor_quat_start == 6
        assert cfg.action_mapping == "midpoint/v1"
        assert cfg.reset_noise_scale == pytest.approx(0.01)
        assert cfg.init_qpos_noise == 0.0
        assert cfg.init_yaw_noise == 0.0
        assert cfg.target_distance_range is None
        assert cfg.target_lateral_range is None
        assert cfg.target_z == pytest.approx(0.5)
        assert cfg.forward_vel_max == pytest.approx(8.0)

    def test_custom_values(self):
        cfg = EvalConfig(
            n_episodes=10,
            max_episode_steps=500,
            frame_skip=3,
            healthy_z_range=(0.5, 1.5),
            max_tilt_angle=0.8,
            root_body_id=2,
        )
        assert cfg.n_episodes == 10
        assert cfg.max_episode_steps == 500
        assert cfg.frame_skip == 3
        assert cfg.healthy_z_range == (0.5, 1.5)
        assert cfg.max_tilt_angle == pytest.approx(0.8)
        assert cfg.root_body_id == 2

    def test_lean_aware_posture_decomposition_matches_target(self):
        natural_pitch = 0.35
        config = EvalConfig(posture_target_forward_z=-np.sin(natural_pitch))
        target_quat = np.array([np.cos(natural_pitch / 2.0), 0.0, np.sin(natural_pitch / 2.0), 0.0])
        upright_quat = np.array([1.0, 0.0, 0.0, 0.0])

        target_reward = _posture_reward_for_eval(target_quat, config, weight=1.5)
        upright_reward = _posture_reward_for_eval(upright_quat, config, weight=1.5)

        assert target_reward == pytest.approx(0.0, abs=1e-10)
        assert upright_reward < target_reward


class TestEvalResults:
    def test_empty_results(self):
        results = EvalResults()
        assert results.mean_reward == 0.0
        assert results.std_reward == 0.0
        assert results.mean_length == 0.0
        assert results.std_length == 0.0
        assert results.mean_forward_vel == 0.0
        assert results.mean_distance == 0.0
        assert results.mean_tilt == 0.0
        assert results.mean_height == 0.0

    def test_with_data(self):
        results = EvalResults()
        results.rewards = [10.0, 20.0, 30.0]
        results.lengths = [100, 200, 300]
        results.forward_vels = [1.0, 2.0, 3.0]
        results.distances = [5.0, 10.0, 15.0]
        results.tilt_angles = [0.1, 0.2, 0.3]
        results.pelvis_heights = [0.5, 0.6, 0.7]

        assert results.mean_reward == pytest.approx(20.0)
        assert results.std_reward == pytest.approx(np.std([10.0, 20.0, 30.0]))
        assert results.mean_length == pytest.approx(200.0)
        assert results.std_length == pytest.approx(np.std([100, 200, 300]))
        assert results.mean_forward_vel == pytest.approx(2.0)
        assert results.mean_distance == pytest.approx(10.0)
        assert results.mean_tilt == pytest.approx(0.2)
        assert results.mean_height == pytest.approx(0.6)

    def test_default_diag_reward_components(self):
        results = EvalResults()
        assert "forward" in results.diag_reward_components
        assert "alive" in results.diag_reward_components
        assert "energy" in results.diag_reward_components
        assert "posture" in results.diag_reward_components
        assert results.diag_reward_diagnostics == {}

    def test_independent_instances(self):
        """Ensure default_factory creates independent lists per instance."""
        r1 = EvalResults()
        r2 = EvalResults()
        r1.rewards.append(1.0)
        assert len(r2.rewards) == 0


def test_eval_reset_randomization_matches_effective_mjx_distribution():
    import mujoco

    from environments.shared.mjx_env import _get_model_path
    from environments.shared.mjx_utils import reset_mujoco_data_to_home

    model = mujoco.MjModel.from_xml_path(_get_model_path("trex"))
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "prey")
    config = EvalConfig(
        reset_noise_scale=0.10,
        init_qpos_noise=0.01,
        init_yaw_noise=0.10,
        target_distance_range=(10.0, 15.0),
        target_lateral_range=(-2.0, 2.0),
        target_z=0.5,
    )

    def randomized(seed):
        data = mujoco.MjData(model)
        reset_mujoco_data_to_home(model, data)
        home_qpos = data.qpos.copy()
        _apply_eval_reset_randomization(
            model,
            data,
            config,
            np.random.default_rng(seed),
            target_body_id,
        )
        mujoco.mj_forward(model, data)
        return data, home_qpos

    first, home_qpos = randomized(3042)
    repeated, _ = randomized(3042)

    np.testing.assert_allclose(first.qpos, repeated.qpos, rtol=0, atol=0)
    np.testing.assert_allclose(first.mocap_pos, repeated.mocap_pos, rtol=0, atol=0)
    joint_delta = first.qpos[7:] - home_qpos[7:]
    assert np.all(np.abs(joint_delta) <= 0.10)
    assert np.any(np.abs(joint_delta) > 0)
    assert np.all(np.abs(first.qpos[:2] - home_qpos[:2]) <= 0.01)
    assert not np.allclose(first.qpos[3:7], home_qpos[3:7])
    assert np.linalg.norm(first.qpos[3:7]) == pytest.approx(1.0)
    target_pos = first.xpos[target_body_id]
    assert 10.0 <= target_pos[0] <= 15.0
    assert -2.0 <= target_pos[1] <= 2.0
    assert target_pos[2] == pytest.approx(0.5)


class TestCheckStageGate:
    def test_passes_when_above_thresholds(self):
        results = EvalResults()
        results.rewards = [100.0, 200.0, 300.0]
        results.lengths = [500, 600, 700]
        passed, failures = check_stage_gate(results, gate_min_reward=50.0, gate_min_length=400)
        assert passed is True
        assert failures == []

    def test_fails_on_low_reward(self):
        results = EvalResults()
        results.rewards = [10.0, 20.0, 30.0]
        results.lengths = [500, 600, 700]
        passed, failures = check_stage_gate(results, gate_min_reward=50.0, gate_min_length=100)
        assert passed is False
        assert len(failures) == 1
        assert "reward" in failures[0]

    def test_fails_on_short_episodes(self):
        results = EvalResults()
        results.rewards = [100.0, 200.0, 300.0]
        results.lengths = [10, 20, 30]
        passed, failures = check_stage_gate(results, gate_min_reward=50.0, gate_min_length=100)
        assert passed is False
        assert len(failures) == 1
        assert "episode length" in failures[0]

    def test_fails_on_both(self):
        results = EvalResults()
        results.rewards = [1.0, 2.0, 3.0]
        results.lengths = [10, 20, 30]
        passed, failures = check_stage_gate(results, gate_min_reward=50.0, gate_min_length=100)
        assert passed is False
        assert len(failures) == 2

    def test_passes_with_default_thresholds(self):
        results = EvalResults()
        results.rewards = [-1000.0]
        results.lengths = [1]
        passed, failures = check_stage_gate(results)
        assert passed is True
        assert failures == []

    def test_exact_threshold(self):
        results = EvalResults()
        results.rewards = [50.0]
        results.lengths = [100]
        passed, failures = check_stage_gate(results, gate_min_reward=50.0, gate_min_length=100)
        assert passed is True
        assert failures == []


class TestUnmeasuredMetricsFailClosed:
    """An evaluation that measured nothing must not clear the thresholds.

    `nan < threshold` is False, so before the `finite_gate_metric` guard an
    unmeasured metric CLEARED every floor beneath it. The SB3 path closed this
    in `reporting.gates`; this path -- which gates JAX stages 2 and 3 and
    whose verdict `jax_setup` writes straight into `publication_gate_passed`
    -- still returned `(True, [])` for an evaluation with no usable episodes.
    """

    @staticmethod
    def _nan_results():
        results = EvalResults()
        results.rewards = [float("nan")]
        results.lengths = [1]
        results.forward_vels = [float("nan")]
        return results

    def test_nan_reward_fails_a_declared_rail(self):
        passed, failures = check_stage_gate(self._nan_results(), gate_min_reward=1950.0)
        assert passed is False
        assert any("no mean reward measurement available" in failure for failure in failures)

    def test_nan_forward_vel_fails_a_declared_floor(self):
        passed, failures = check_stage_gate(self._nan_results(), gate_min_forward_vel=2.0)
        assert passed is False
        assert any("no mean forward vel measurement available" in failure for failure in failures)

    def test_undeclared_thresholds_still_assert_nothing(self):
        """Only a declared threshold makes "unmeasurable" a refusal.

        An unset rail (-inf) and an unset floor (0) could not be failed
        before this guard either, so they must not start failing now — that
        would abort stages whose configs are entirely legitimate.
        """
        passed, failures = check_stage_gate(self._nan_results())
        assert passed is True
        assert failures == []

    def test_a_stance_gated_stage_2_config_with_no_usable_evaluation_fails(self):
        """End to end through the entry point `jax_setup` actually calls."""
        results = EvalResults()
        results.rewards = [float("nan")]
        results.lengths = [1]
        stage_config = {
            "stage": 2,
            "curriculum_kwargs": {
                "gate_kind": "reward_and_length/v1",
                "gate_schema_version": 1,
                "min_avg_reward": 100.0,
                "min_avg_episode_length": 750,
                "min_avg_forward_vel": 2.0,
                "required_consecutive": 3,
            },
        }
        passed, failures = check_stage_gate_for_config(results, stage_config)
        assert passed is False
        assert len(failures) == 3

    def test_a_genuinely_passing_stage_2_still_passes(self):
        """Guard against over-correction, on run 20260803_012355's real stage 2."""
        results = EvalResults()
        results.rewards = [3031.22] * 4
        results.lengths = [1000] * 4
        results.forward_vels = [3.94] * 4
        stage_config = {
            "stage": 2,
            "curriculum_kwargs": {
                "gate_kind": "reward_and_length/v1",
                "gate_schema_version": 1,
                "min_avg_reward": 100.0,
                "min_avg_episode_length": 750,
                "min_avg_forward_vel": 2.0,
                "required_consecutive": 3,
            },
        }
        assert check_stage_gate_for_config(results, stage_config) == (True, [])


class TestSuccessAndVelGates:
    """New gate criteria: forward velocity (stage 2) and success rate (stage 3)."""

    def _results(self):
        results = EvalResults()
        results.rewards = [100.0] * 4
        results.lengths = [500] * 4
        results.forward_vels = [1.0, 2.0, 3.0, 2.0]
        results.successes = [True, True, False, False]
        return results

    def test_mean_success_rate_property(self):
        assert self._results().mean_success_rate == 0.5
        assert EvalResults().mean_success_rate == 0.0

    def test_forward_vel_gate(self):
        passed, failures = check_stage_gate(self._results(), gate_min_forward_vel=1.5)
        assert passed
        passed, failures = check_stage_gate(self._results(), gate_min_forward_vel=5.0)
        assert not passed
        assert any("forward vel" in f for f in failures)

    def test_success_rate_gate(self):
        passed, failures = check_stage_gate(self._results(), gate_min_success_rate=0.5)
        assert passed
        passed, failures = check_stage_gate(self._results(), gate_min_success_rate=0.75)
        assert not passed
        assert any("success rate" in f for f in failures)

    def test_zero_thresholds_disable_new_gates(self):
        results = EvalResults()
        results.rewards = [1.0]
        results.lengths = [10]
        # No forward_vels/successes recorded at all -- gates must not fire.
        passed, failures = check_stage_gate(results)
        assert passed


@pytest.mark.parametrize("species", ("velociraptor", "trex"))
def test_cpu_evaluation_noise_free_reset_restores_complete_home_state(species):
    jax = pytest.importorskip("jax")
    pytest.importorskip("mujoco.mjx")
    import jax.numpy as jnp
    import mujoco

    from environments.shared.jax_eval import evaluate_policy_cpu
    from environments.shared.mjx_env import _get_model_path
    from environments.shared.mjx_utils import reset_mujoco_data_to_home

    model = mujoco.MjModel.from_xml_path(_get_model_path(species))
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    captured: dict[str, np.ndarray] = {}

    def get_obs(data):
        if not captured:
            for field in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
                captured[field] = np.asarray(jax.device_get(getattr(data, field)))
        return jnp.zeros(1)

    class ZeroNetwork:
        def apply(self, _params, _obs):
            return jnp.zeros(model.nu), jnp.zeros(model.nu), jnp.float32(0.0)

    evaluate_policy_cpu(
        model,
        {},
        ZeroNetwork(),
        None,
        get_obs_fn=get_obs,
        normalize_obs_fn=lambda obs, _stats: obs,
        scale_action_fn=lambda _action: jnp.asarray(model.key_ctrl[home_id]),
        reward_fn=lambda _data, _action, _cfg, **_kwargs: jnp.float32(0.0),
        reward_cfg={},
        config=EvalConfig(
            n_episodes=1,
            max_episode_steps=1,
            healthy_z_range=(0.0, 10.0),
            max_tilt_angle=np.pi,
            natural_forward_z=-1.0,
            nosedive_threshold=10.0,
            root_body_id=2,
            sensor_quat_start=6,
            action_mapping="home-keyframe-residual/v1",
            reset_noise_scale=0.0,
        ),
    )

    expected = mujoco.MjData(model)
    reset_mujoco_data_to_home(model, expected)
    mujoco.mj_forward(model, expected)
    for field, actual in captured.items():
        np.testing.assert_allclose(
            actual,
            np.asarray(getattr(expected, field)),
            rtol=0,
            atol=1e-7,
            err_msg=f"CPU evaluation reset field {field} diverged from the XML home keyframe",
        )


def test_cpu_evaluation_records_canonical_stance_components_and_diagnostics():
    pytest.importorskip("jax")
    pytest.importorskip("mujoco.mjx")
    import jax.numpy as jnp
    import mujoco

    from environments.shared.jax_eval import evaluate_policy_cpu
    from environments.shared.mjx_env import _get_model_path

    model = mujoco.MjModel.from_xml_path(_get_model_path("trex"))
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    class ZeroNetwork:
        def apply(self, _params, _obs):
            return jnp.zeros(model.nu), jnp.zeros(model.nu), jnp.float32(0.0)

    canonical_components = {
        "alive": 0.25,
        "height": 0.45,
        "bilateral_support": 0.60,
        "foot_load_balance": -0.20,
        "leg_home_pose": 0.50,
        "head_clearance": 0.35,
        "neck_posture": 0.20,
    }
    canonical_diagnostics = {
        "_bilateral_support_quality": 0.75,
        "_foot_load_imbalance": 0.10,
        "_raw_alive": 1.0,
        "_alive_gate": 0.25,
        "_height_error": 0.02,
        "_height_quality": 0.90,
    }
    scalar_step_kwargs: dict = {}

    def scalar_reward(_data, _action, _reward_cfg, **kwargs):
        scalar_step_kwargs.update(kwargs)
        return jnp.float32(0.0)

    def detailed_reward(_data, _action, _reward_cfg, **_kwargs):
        return {
            **{name: jnp.float32(value) for name, value in canonical_components.items()},
            **{name: jnp.float32(value) for name, value in canonical_diagnostics.items()},
        }

    results = evaluate_policy_cpu(
        model,
        {},
        ZeroNetwork(),
        None,
        get_obs_fn=lambda _data: jnp.zeros(1),
        normalize_obs_fn=lambda obs, _stats: obs,
        scale_action_fn=lambda _action: jnp.asarray(model.key_ctrl[home_id]),
        reward_fn=scalar_reward,
        reward_components_fn=detailed_reward,
        reward_cfg={"alive_bonus": 99.0, "height_weight": 99.0},
        config=EvalConfig(
            n_episodes=1,
            max_episode_steps=1,
            healthy_z_range=(0.0, 10.0),
            max_tilt_angle=np.pi,
            natural_forward_z=-1.0,
            nosedive_threshold=10.0,
            root_body_id=root_body_id,
            sensor_quat_start=6,
            termination_body_heights={"pelvis": 10.0},
            action_mapping="home-keyframe-residual/v1",
            reset_noise_scale=0.0,
        ),
    )

    for name, expected in canonical_components.items():
        assert results.diag_reward_components[name] == pytest.approx([expected])
    for name, expected in canonical_diagnostics.items():
        assert results.diag_reward_diagnostics[name[1:]] == pytest.approx([expected])
    # These would equal 99.0 if the old eval-only alive/height formulas had
    # remained authoritative.
    assert results.diag_reward_components["alive"] == pytest.approx([0.25])
    assert results.diag_reward_components["height"] == pytest.approx([0.45])
    assert bool(scalar_step_kwargs["additional_terminated"]) is True
    np.testing.assert_allclose(
        np.asarray(scalar_step_kwargs["initial_pos_2d"]),
        model.key_qpos[home_id, :2],
        rtol=0,
        atol=1e-7,
    )


def test_stage_evaluation_uses_effective_training_noise_and_detailed_rewards(monkeypatch):
    pytest.importorskip("jax")
    from types import SimpleNamespace

    import environments.shared.jax_eval as jax_eval
    import environments.shared.jax_setup as jax_setup
    from environments.shared.config import load_stage_config

    stage_config = load_stage_config("trex", 1)
    effective_env_kwargs = jax_setup.build_env_kwargs(
        stage_config["env_kwargs"],
        stage_config["jax_kwargs"],
    )
    # Deliberately NOT pinned to a literal: this test is about the COUPLING
    # (evaluation must run at the effective training noise), and the stage-1
    # operating point is a design decision that moves -- it went 0.10 -> 0.05
    # when stage 1 adopted the 1a point (PLANT_VALIDATION section 12).  Pinning
    # the literal here made a deliberate config change look like a JAX-eval
    # regression.  What must not silently change is that evaluation noise is
    # real, so guard that instead and compare everything downstream to this.
    training_noise = effective_env_kwargs["reset_noise_scale"]
    assert training_noise > 0.0, "evaluation would be noise-free, which is not the training distribution"

    def scalar_reward(*_args, **_kwargs):
        return 0.0

    def detailed_reward(*_args, **_kwargs):
        return {"bilateral_support": 0.0}

    monkeypatch.setattr(jax_setup, "make_obs_fn", lambda _ctx: scalar_reward)
    monkeypatch.setattr(jax_setup, "make_scale_action_fn", lambda _ctx: scalar_reward)
    monkeypatch.setattr(
        jax_setup,
        "make_reward_fns",
        lambda _ctx: (scalar_reward, detailed_reward, scalar_reward),
    )

    captured: list[dict] = []

    def fake_evaluate_policy_cpu(*_args, **kwargs):
        captured.append(kwargs)
        results = EvalResults()
        results.rewards = [4000.0]
        results.lengths = [1000]
        results.forward_vels = [0.0]
        results.distances = [0.0]
        results.successes = [False]
        return results

    monkeypatch.setattr(jax_eval, "evaluate_policy_cpu", fake_evaluate_policy_cpu)

    ctx = SimpleNamespace(
        species="trex",
        stage=1,
        stage_name="Balance",
        reward_cfg={},
        jax_kwargs={"rollout_len": 64},
        max_episode_steps=1000,
        frame_skip=5,
        healthy_z_range=(0.3, 2.0),
        max_tilt_angle=1.0,
        natural_forward_z=0.0,
        posture_target_forward_z=None,
        termination_body_heights={},
        termination_site_heights={},
        success_sites=(),
        success_threshold=0.3,
        target_body_name="prey",
        root_body_id=1,
        sensor_layout=SimpleNamespace(
            quat_start=6,
            gyro_start=0,
            foot_indices=(10, 11),
            foot_aux_indices=((), ()),
        ),
        action_mapping="midpoint/v1",
        forward_vel_max=8.0,
        target_standing_z=0.926,
        mj_model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002)),
    )
    env = SimpleNamespace(
        num_envs=2048,
        config=SimpleNamespace(
            reset_noise_scale=effective_env_kwargs["reset_noise_scale"],
            init_qpos_noise=effective_env_kwargs["init_qpos_noise"],
            init_yaw_noise=effective_env_kwargs["init_yaw_noise"],
            target_distance_range=(10.0, 15.0),
            target_lateral_range=(-2.0, 2.0),
            target_z=0.5,
        ),
    )

    _, _, stage_results, _, _ = jax_setup.run_stage_evaluation(
        ctx,
        env,
        np.zeros(1),
        object(),
        None,
        n_episodes=1,
    )

    assert len(captured) == 1
    assert captured[0]["config"].reset_noise_scale == pytest.approx(training_noise)
    assert captured[0]["config"].init_qpos_noise == pytest.approx(0.01)
    assert captured[0]["config"].init_yaw_noise == pytest.approx(0.10)
    assert captured[0]["config"].target_distance_range == (10.0, 15.0)
    assert captured[0]["config"].target_lateral_range == (-2.0, 2.0)
    assert captured[0]["config"].target_z == pytest.approx(0.5)
    assert captured[0]["reward_components_fn"] is detailed_reward
    assert stage_results["evaluation_reset_noise_scale"] == pytest.approx(training_noise)
    assert stage_results["evaluation_init_qpos_noise"] == pytest.approx(0.01)
    assert stage_results["evaluation_init_yaw_noise"] == pytest.approx(0.10)
    assert stage_results["evaluation_target_distance_range"] == [10.0, 15.0]
    assert stage_results["evaluation_target_lateral_range"] == [-2.0, 2.0]
    assert stage_results["evaluation_target_z"] == pytest.approx(0.5)
    assert stage_results["evaluation_seed"] == 42


def test_stage_evaluation_refuses_the_recovery_gate_instead_of_certifying_it(monkeypatch):
    """Review J #10, on the production path.

    ``run_stage_evaluation`` writes ``check_stage_gate_for_config``'s verdict
    into ``gate_passed`` / ``publication_gate_passed``.  Once ``jax_setup``
    accepted the semantic id ``"recovery"`` (it used to raise on
    ``stage >= 3`` before any gate check), a recovery stage fell through to
    the scalar check with every threshold unset, and a policy that fell in
    every pushed episode was persisted as a PASS.  The eval path must refuse
    the kind exactly as the in-training curriculum does.
    """
    pytest.importorskip("jax")
    from types import SimpleNamespace

    import environments.shared.jax_eval as jax_eval
    import environments.shared.jax_setup as jax_setup

    def scalar(*_args, **_kwargs):
        return 0.0

    def detailed(*_args, **_kwargs):
        return {"alive": 0.0}

    monkeypatch.setattr(jax_setup, "make_obs_fn", lambda _ctx: scalar)
    monkeypatch.setattr(jax_setup, "make_scale_action_fn", lambda _ctx: scalar)
    monkeypatch.setattr(jax_setup, "make_reward_fns", lambda _ctx: (scalar, detailed, scalar))

    def fallen_in_every_episode(*_args, **_kwargs):
        results = EvalResults()
        results.rewards = [-150.0] * 3
        results.lengths = [5] * 3
        results.forward_vels = [0.0] * 3
        results.distances = [0.0] * 3
        results.successes = [False] * 3
        return results

    monkeypatch.setattr(jax_eval, "evaluate_policy_cpu", fallen_in_every_episode)

    ctx = SimpleNamespace(
        species="trex",
        stage="recovery",
        stage_name="Recovery",
        reward_cfg={},
        jax_kwargs={"rollout_len": 64},
        max_episode_steps=1000,
        frame_skip=5,
        healthy_z_range=(0.3, 2.0),
        max_tilt_angle=1.0,
        natural_forward_z=0.0,
        posture_target_forward_z=None,
        termination_body_heights={},
        termination_site_heights={},
        success_sites=(),
        success_threshold=0.3,
        target_body_name="prey",
        root_body_id=1,
        sensor_layout=SimpleNamespace(
            quat_start=6,
            gyro_start=0,
            foot_indices=(10, 11),
            foot_aux_indices=((), ()),
        ),
        action_mapping="midpoint/v1",
        forward_vel_max=8.0,
        target_standing_z=0.926,
        mj_model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002)),
    )
    env = SimpleNamespace(
        num_envs=4,
        config=SimpleNamespace(
            reset_noise_scale=0.05,
            init_qpos_noise=0.01,
            init_yaw_noise=0.1,
            target_distance_range=(10.0, 15.0),
            target_lateral_range=(-2.0, 2.0),
            target_z=0.5,
        ),
    )

    _, _, stage_results, gate_passed, gate_failures = jax_setup.run_stage_evaluation(
        ctx,
        env,
        np.zeros(1),
        object(),
        None,
        n_episodes=3,
    )

    assert gate_passed is False
    assert stage_results["gate_passed"] is False
    assert stage_results["publication_gate_passed"] is False
    assert stage_results["gate_failures"] == gate_failures
    assert len(gate_failures) == 1
    assert "recovery_quality/v1" in gate_failures[0]
    assert "cannot evaluate" in gate_failures[0]


#: A schema-valid recovery gate, as configs/trex/recovery.toml declares it:
#: the three composite thresholds and NONE of the four scalar ones.
_RECOVERY_GATE = {
    "gate_schema_version": 1,
    "gate_kind": "recovery_quality/v1",
    "min_recovery_success_lcb": 0.70,
    "recovery_t_recover_steps": 66,
    "recovery_dwell_steps": 66,
}


class TestCheckStageGateForConfig:
    """The gate-kind-aware JAX gate (`check_stage_gate_for_config`).

    `jax_setup.run_stage_evaluation` used to call `check_stage_gate` directly,
    which reads four fixed thresholds and knows nothing about `gate_kind`. For
    a stance-gated stage that certified on `min_avg_reward` alone -- a rail
    the zero-action statue clears -- and wrote the verdict into
    `publication_gate_passed`.
    """

    HORIZON = 1000
    SETTLE = 200

    def _config(self, gate_kind="stance_quality/v1", **curriculum):
        base = {
            "gate_schema_version": 1,
            "gate_kind": gate_kind,
            "min_eval_episodes": 4,
            "required_consecutive": 3,
        }
        if gate_kind == "stance_quality/v1":
            base.update(
                {
                    "min_full_horizon_fraction": 0.95,
                    "max_unsupported_duty": 0.02,
                    "max_unsupported_duty_ucb": 0.02,
                    "settle_steps": self.SETTLE,
                    "min_avg_reward": 1950.0,
                }
            )
        else:
            base["min_avg_reward"] = 1950.0
        base.update(curriculum)
        return {"curriculum_kwargs": base, "env_kwargs": {"max_episode_steps": self.HORIZON}}

    def _results(self, *, duty, episodes=4, length=None, readings_per_side=1, reward=3271.8):
        from environments.shared.jax_eval import EvalResults

        results = EvalResults()
        length = length or self.HORIZON
        for _ in range(episodes):
            results.lengths.append(length)
            results.rewards.append(reward)
            results.forward_vels.append(0.0)
            results.distances.append(0.0)
            results.successes.append(False)
            post = max(0, length - self.SETTLE)
            airborne_steps = round(duty * post)
            for step in range(length):
                airborne = step >= self.SETTLE and (step - self.SETTLE) < airborne_steps
                for _ in range(readings_per_side):
                    results.diag_r_foot.append(0.0 if airborne else 50.0)
                    results.diag_l_foot.append(0.0 if airborne else 50.0)
        return results

    def test_the_zero_action_statue_passes(self):
        # By design: STAGE1_SPLIT_PLAN 2.3.1 says the statue's numbers are what
        # a 1a policy must MATCH, not beat.
        from environments.shared.jax_eval import check_stage_gate_for_config

        passed, failures = check_stage_gate_for_config(self._results(duty=0.0), self._config())
        assert passed, failures

    def test_the_chatterer_is_rejected(self):
        # The policy the gate exists to reject: full horizon, clears the reward
        # rail, but spends a third of its post-settling steps airborne.
        from environments.shared.jax_eval import check_stage_gate_for_config

        passed, failures = check_stage_gate_for_config(self._results(duty=1 / 3, reward=2133.4), self._config())
        assert not passed
        assert any("unsupported_duty" in failure for failure in failures)

    def test_fails_closed_when_foot_traces_are_absent(self):
        from environments.shared.jax_eval import EvalResults, check_stage_gate_for_config

        bare = EvalResults()
        bare.lengths = [self.HORIZON] * 4
        bare.rewards = [3271.8] * 4
        passed, failures = check_stage_gate_for_config(bare, self._config())
        assert not passed
        assert "cannot be evaluated" in failures[0]
        assert "reward rail alone" in failures[0]

    def test_fails_closed_on_the_quadruped_interleaving_defect(self):
        # Four foot sensors append two readings per side per step, so the
        # i % 2 routing in evaluate_policy_cpu no longer means right/left.
        from environments.shared.jax_eval import check_stage_gate_for_config

        passed, failures = check_stage_gate_for_config(self._results(duty=0.0, readings_per_side=2), self._config())
        assert not passed
        assert "do not align with episode boundaries" in failures[0]

    def test_a_declared_pilot_is_rejected_by_the_schema(self):
        # none/v1 never reaches a verdict here: the schema refuses a declared
        # non-advancing pilot in a run that advances, which is louder than
        # returning False.
        from environments.shared.curriculum.gate_schema import GateSchemaError
        from environments.shared.jax_eval import check_stage_gate_for_config

        with pytest.raises(GateSchemaError, match="non-advancing pilot"):
            check_stage_gate_for_config(
                self._results(duty=0.0),
                {"curriculum_kwargs": {"gate_schema_version": 1, "gate_kind": "none/v1"}},
            )

    def test_an_undeclared_gate_is_rejected(self):
        from environments.shared.curriculum.gate_schema import GateSchemaError
        from environments.shared.jax_eval import check_stage_gate_for_config

        with pytest.raises(GateSchemaError):
            check_stage_gate_for_config(self._results(duty=0.0), {"curriculum_kwargs": {"min_avg_reward": 1.0}})

    def test_reward_and_length_stages_are_unchanged(self):
        from environments.shared.jax_eval import check_stage_gate_for_config

        config = self._config(gate_kind="reward_and_length/v1", min_avg_episode_length=950)
        passed, _ = check_stage_gate_for_config(self._results(duty=0.0), config)
        assert passed

        short = self._results(duty=0.0, length=500)
        passed, failures = check_stage_gate_for_config(short, config)
        assert not passed
        assert any("episode length" in failure for failure in failures)

    def test_the_real_trex_stage1_config_rejects_the_chatterer(self):
        from environments.shared.config import load_stage_config
        from environments.shared.jax_eval import check_stage_gate_for_config

        config = load_stage_config("trex", 1)
        assert config["curriculum_kwargs"]["gate_kind"] == "stance_quality/v1"
        episodes = int(config["curriculum_kwargs"]["min_eval_episodes"])
        chatterer = self._results(duty=1 / 3, episodes=episodes, reward=2133.4)
        passed, _ = check_stage_gate_for_config(chatterer, config)
        assert not passed

    def _fallen(self, episodes=30):
        """Every episode ends in a fall within a few steps: the evidence a
        recovery gate exists to reject."""
        from environments.shared.jax_eval import EvalResults

        results = EvalResults()
        results.rewards = [-150.0] * episodes
        results.lengths = [5] * episodes
        results.forward_vels = [0.0] * episodes
        results.distances = [0.0] * episodes
        results.successes = [False] * episodes
        return results

    def test_a_recovery_gate_is_refused_rather_than_passed_vacuously(self):
        """Review J #10: recovery_quality/v1 validates and declares no scalar
        threshold, so the fall-through to check_stage_gate asserted nothing
        and certified a policy that fell in every pushed episode as
        (True, [])."""
        from environments.shared.jax_eval import check_stage_gate_for_config

        passed, failures = check_stage_gate_for_config(self._fallen(), {"curriculum_kwargs": dict(_RECOVERY_GATE)})
        assert passed is False
        assert len(failures) == 1
        assert "recovery_quality/v1" in failures[0]
        assert "cannot evaluate" in failures[0]

    def test_a_recovery_gate_is_refused_on_sky_high_evidence_even_with_its_optional_rail(self):
        from environments.shared.jax_eval import check_stage_gate_for_config

        railed = {"curriculum_kwargs": dict(_RECOVERY_GATE, min_avg_reward=100.0)}
        passed, failures = check_stage_gate_for_config(self._results(duty=0.0, reward=1e9), railed)
        assert passed is False
        assert "recovery_quality/v1" in failures[0]

    def test_the_shipped_recovery_config_is_refused_as_run_stage_evaluation_reads_it(self):
        """The exact review probe: configs/trex/recovery.toml, prepared the
        way run_stage_evaluation prepares it (jax_gate_thresholds applied)."""
        from environments.shared.config import load_stage_config
        from environments.shared.jax_curriculum import jax_gate_thresholds
        from environments.shared.jax_eval import check_stage_gate_for_config

        config = dict(load_stage_config("trex", "recovery"))
        assert config["curriculum_kwargs"]["gate_kind"] == "recovery_quality/v1"
        config["curriculum_kwargs"] = jax_gate_thresholds("recovery", config, species="trex")
        passed, failures = check_stage_gate_for_config(self._fallen(), config)
        assert passed is False
        assert "recovery_quality/v1" in failures[0]

    @pytest.mark.parametrize(
        "curriculum",
        [
            pytest.param({"gate_schema_version": 1, "gate_kind": "none/v1"}, id="none-v1"),
            pytest.param({"min_avg_reward": 1.0}, id="undeclared"),
            pytest.param(_RECOVERY_GATE, id="recovery"),
        ],
    )
    def test_a_gate_the_eval_cannot_judge_never_passes_by_default(self, curriculum):
        """Mirror of reporting.gates.evaluate_stage_gate: nothing certifies a
        stage by default.  none/v1 and an undeclared gate are refused by the
        schema (a raise, louder than False -- the two tests above pin it); a
        schema-valid kind with no evaluator here is refused with the reason
        as the verdict.  None may reach (True, ...) on any evidence."""
        from environments.shared.curriculum.gate_schema import GateSchemaError
        from environments.shared.jax_eval import check_stage_gate_for_config

        try:
            passed, failures = check_stage_gate_for_config(
                self._results(duty=0.0, reward=1e9), {"curriculum_kwargs": dict(curriculum)}
            )
        except GateSchemaError:
            return
        assert passed is False
        assert failures


def test_run_stage_evaluation_routes_through_the_gate_kind_aware_check(monkeypatch):
    """The integration, not just the function.

    ``run_stage_evaluation`` imported ``check_stage_gate`` -- the four-fixed-
    threshold one -- so a stance-gated stage was certified on ``min_avg_reward``
    alone and that verdict became ``publication_gate_passed``. Teaching
    ``check_stage_gate_for_config`` the stance criteria fixes nothing if this
    call site still reaches the old function.
    """
    import inspect

    from environments.shared import jax_setup

    source = inspect.getsource(jax_setup.run_stage_evaluation)
    assert "check_stage_gate_for_config(" in source
    # The legacy name must not survive as a call; it appears only in the
    # comment explaining why it was replaced.
    assert "check_stage_gate(" not in source.replace("check_stage_gate_for_config(", "")


def test_the_legacy_jax_gate_is_not_reachable_from_production_code():
    """Only the gate-kind-aware wrapper may call it."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    offenders = []
    for path in (repo / "environments").rglob("*.py"):
        if "tests" in path.parts or path.name == "jax_eval.py":
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "check_stage_gate(" in stripped.replace("check_stage_gate_for_config(", ""):
                # jax_curriculum has its own same-named adapter over a metrics
                # dict; it dispatches on gate_kind itself and is not this one.
                if path.name == "jax_curriculum.py":
                    continue
                offenders.append(f"{path.relative_to(repo)}:{number}")
    assert not offenders, f"legacy four-threshold gate called from: {offenders}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestEvaluationPushes:
    """JX3(e): CPU eval rolls the recovery stage's scheduled pushes.

    Before this pin ``evaluate_policy_cpu`` never wrote ``xfrc_applied``, so
    a recovery-stage evaluation certified push-free episodes while
    ``MJXDinoEnv`` trained against the derived 165.5 N shoves.  Schedule,
    force and body come from the shared perturbation module (the same
    ``push_schedule`` / ``external_push_force`` the MJX step traces), the
    seed draw is appended to the reset draws so a push-free eval of the same
    seed is unchanged, and a config that declares no pushes never touches
    ``xfrc_applied``.
    """

    # configs/trex/recovery.toml [env] perturbation block, verbatim (control
    # dt 0.01 s: interval 200 +/- 50 steps, duration 20 steps).
    _RECOVERY = {
        "perturbation_capture_velocity_multiple": 1.5,
        "perturbation_interval": 2.0,
        "perturbation_jitter": 0.5,
        "perturbation_duration": 0.2,
    }
    # Cheap schedule for the RNG-discipline checks: first push near step 30.
    _SHORT = {
        "perturbation_capture_velocity_multiple": 1.5,
        "perturbation_interval": 0.3,
        "perturbation_jitter": 0.05,
        "perturbation_duration": 0.1,
    }

    @staticmethod
    def _roll(max_steps, **perturbation):
        pytest.importorskip("jax")
        pytest.importorskip("mujoco.mjx")
        import jax.numpy as jnp
        import mujoco

        from environments.shared.jax_eval import evaluate_policy_cpu
        from environments.shared.mjx_env import _get_model_path

        model = mujoco.MjModel.from_xml_path(_get_model_path("trex"))
        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        first_qpos: list[np.ndarray] = []
        xfrc: list[np.ndarray] = []

        def get_obs(data):
            if not first_qpos:
                first_qpos.append(np.asarray(data.qpos).copy())
            return jnp.zeros(1)

        class ZeroNetwork:
            def apply(self, _params, _obs):
                return jnp.zeros(model.nu), jnp.zeros(model.nu), jnp.float32(0.0)

        def reward_fn(data, _action, _cfg, **_kwargs):
            # Post-step data carries the force written for this control step.
            xfrc.append(np.asarray(data.xfrc_applied[root_body_id, :3]).copy())
            return jnp.float32(0.0)

        evaluate_policy_cpu(
            model,
            {},
            ZeroNetwork(),
            None,
            get_obs_fn=get_obs,
            normalize_obs_fn=lambda obs, _stats: obs,
            scale_action_fn=lambda _action: jnp.asarray(model.key_ctrl[home_id]),
            reward_fn=reward_fn,
            reward_cfg={},
            config=EvalConfig(
                n_episodes=1,
                max_episode_steps=max_steps,
                healthy_z_range=(0.0, 10.0),
                max_tilt_angle=np.pi,
                natural_forward_z=-1.0,
                nosedive_threshold=10.0,
                root_body_id=root_body_id,
                action_mapping="home-keyframe-residual/v1",
                reset_noise_scale=0.05,
                seed=42,
                **perturbation,
            ),
        )
        return model, first_qpos[0], np.stack(xfrc)

    def test_a_no_push_config_leaves_xfrc_zero_throughout(self):
        _model, _qpos, xfrc = self._roll(300)
        assert xfrc.shape == (300, 3)
        assert not xfrc.any()

    def test_the_recovery_config_pushes_on_the_shared_schedule(self):
        import mujoco

        from environments.shared.perturbation import derive_push_parameters

        model, _qpos, xfrc = self._roll(300, **self._RECOVERY)
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        force_n = derive_push_parameters(model, capture_velocity_multiple=1.5, duration_s=0.2, keyframe_id=home_id)[
            "force_n"
        ]
        assert force_n == pytest.approx(165.5, abs=2.0)

        active = np.flatnonzero(np.abs(xfrc).sum(axis=1) > 0.0)
        # Exactly one 20-step window inside [150, 250]; the second push
        # (>= 350) lies past the roll.  Zero everywhere else.
        assert active.size == 20
        assert 150 <= active[0] <= 250
        np.testing.assert_array_equal(active, np.arange(active[0], active[0] + 20))
        np.testing.assert_allclose(np.linalg.norm(xfrc[active, :2], axis=1), force_n, rtol=1e-5)
        assert not xfrc[active, 2].any()
        # One heading per push, held for the whole window.
        np.testing.assert_allclose(xfrc[active], np.broadcast_to(xfrc[active[0]], (20, 3)), rtol=1e-6, atol=1e-6)

    def test_the_schedule_draw_is_appended_and_deterministic(self):
        _m, pushed_qpos, xfrc_a = self._roll(60, **self._SHORT)
        _m, free_qpos, xfrc_free = self._roll(60)
        _m, _q, xfrc_b = self._roll(60, **self._SHORT)
        assert xfrc_a.any() and not xfrc_free.any()
        # The reset draws are unchanged by the appended seed draw ...
        np.testing.assert_array_equal(pushed_qpos, free_qpos)
        # ... and the same eval seed yields the same schedule (pairing).
        np.testing.assert_array_equal(xfrc_a, xfrc_b)
