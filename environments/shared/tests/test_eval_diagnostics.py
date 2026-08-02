"""Tests for stage-aware SB3 evaluation diagnostics."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environments.shared import eval_diagnostics as eval_diagnostics_module
from environments.shared.eval_diagnostics import (
    StageAwareEvalCallback,
    StageGatePlateauCallback,
    success_metric_applicable,
)


class TestSuccessMetricApplicable:
    def test_requires_positive_configured_gate(self):
        assert success_metric_applicable({"curriculum_kwargs": {"min_success_rate": 0.5}}) is True
        assert success_metric_applicable({"curriculum_kwargs": {"min_success_rate": 0.0}}) is False
        assert success_metric_applicable({"curriculum_kwargs": {}}) is False
        assert success_metric_applicable({}) is False

    @pytest.mark.parametrize("species", ["velociraptor", "trex", "brachiosaurus"])
    def test_real_configs_enable_success_only_for_stage_three(self, species):
        from environments.shared.config import load_all_stages

        configs = load_all_stages(species)

        assert success_metric_applicable(configs[1]) is False
        assert success_metric_applicable(configs[2]) is False
        assert success_metric_applicable(configs[3]) is True


def _bare_stage_aware_callback(
    *,
    success_applicable: bool,
    settle_steps: int = 0,
) -> StageAwareEvalCallback:
    if success_applicable:
        pytest.importorskip("stable_baselines3")
    callback = object.__new__(StageAwareEvalCallback)
    callback.stage = 3 if success_applicable else 1
    callback.success_applicable = success_applicable
    callback.settle_steps = settle_steps
    callback._is_success_buffer = []
    callback.evaluations_forward_velocities = []
    callback.evaluations_unsupported_duties = []
    callback.evaluations_bilateral_duties = []
    callback._episode_forward_sums = {}
    callback._episode_forward_counts = {}
    callback._current_eval_forward_velocities = []
    callback._episode_steps = {}
    callback._episode_unsupported = {}
    callback._episode_bilateral = {}
    callback._episode_measured = {}
    callback._current_eval_unsupported_duties = []
    callback._current_eval_bilateral_duties = []
    return callback


class TestStageAwareEvalCallback:
    def test_inapplicable_success_does_not_populate_sb3_buffer(self):
        callback = _bare_stage_aware_callback(success_applicable=False)

        callback._log_success_callback(
            {"i": 0, "done": True, "info": {"is_success": True, "forward_vel": 1.0}},
            {},
        )

        assert callback._is_success_buffer == []

    def test_applicable_zero_success_remains_a_real_zero(self):
        callback = _bare_stage_aware_callback(success_applicable=True)

        callback._log_success_callback(
            {"i": 0, "done": True, "info": {"is_success": False, "forward_vel": 1.0}},
            {},
        )

        assert callback._is_success_buffer == [False]

    def test_captures_per_episode_mean_forward_velocity(self):
        callback = _bare_stage_aware_callback(success_applicable=False)

        callback._log_success_callback({"i": 0, "done": False, "info": {"forward_vel": 1.0}}, {})
        callback._log_success_callback({"i": 0, "done": True, "info": {"forward_vel": 3.0}}, {})

        assert callback._current_eval_forward_velocities == [2.0]

    def test_prints_na_for_inapplicable_evaluation(self, capsys):
        pytest.importorskip("stable_baselines3")
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback.eval_freq = 10
        callback.n_calls = 10
        callback.verbose = 1

        with patch.object(eval_diagnostics_module._EvalCallback, "_on_step", return_value=True):
            assert callback._on_step() is True

        assert "Success rate: N/A (not an active Stage 1 gate)" in capsys.readouterr().out
        assert callback.evaluations_forward_velocities == [[]]

    @pytest.mark.parametrize(
        ("success_applicable", "expected_successes", "expected_output"),
        [
            (False, False, "Success rate: N/A"),
            (True, True, "Success rate: 0.00%"),
        ],
    )
    def test_npz_schema_and_console_output(
        self,
        tmp_path,
        capsys,
        success_applicable,
        expected_successes,
        expected_output,
    ):
        gym = pytest.importorskip("gymnasium")
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        class TinyEvalEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.steps = 0
                return np.zeros(1, dtype=np.float32), {}

            def step(self, action):
                self.steps += 1
                done = self.steps >= 2
                info = {"forward_vel": float(self.steps)}
                if done:
                    info["is_success"] = False
                return np.zeros(1, dtype=np.float32), 1.0, done, False, info

        train_env = DummyVecEnv([lambda: Monitor(TinyEvalEnv())])
        eval_env = DummyVecEnv([lambda: Monitor(TinyEvalEnv())])
        model = MagicMock()
        model.get_env.return_value = train_env
        model.get_vec_normalize_env.return_value = None
        model.num_timesteps = 1
        model.logger = MagicMock()
        model.predict.side_effect = lambda observations, **kwargs: (
            np.zeros((observations.shape[0], 1), dtype=np.float32),
            None,
        )

        callback = StageAwareEvalCallback(
            eval_env,
            stage=3 if success_applicable else 1,
            success_applicable=success_applicable,
            log_path=str(tmp_path),
            eval_freq=1,
            n_eval_episodes=2,
            verbose=1,
        )
        callback.init_callback(model)

        assert callback.on_step() is True

        data = np.load(tmp_path / "evaluations.npz")
        assert ("successes" in data.files) is expected_successes
        if expected_successes:
            assert data["successes"].shape == (1, 2)
            assert data["successes"].tolist() == [[False, False]]
        assert callback.evaluations_forward_velocities == [[1.5, 1.5]]
        assert expected_output in capsys.readouterr().out


def _plateau_callback(
    curriculum_kwargs: dict,
    *,
    stage: int,
    plateau_window: int = 3,
    min_relative_variation: float = 0.05,
):
    eval_callback = SimpleNamespace(
        evaluations_results=[],
        evaluations_length=[],
        evaluations_forward_velocities=[],
        evaluations_successes=[],
        evaluations_unsupported_duties=[],
        evaluations_bilateral_duties=[],
    )
    callback = object.__new__(StageGatePlateauCallback)
    callback.eval_callback = cast(StageAwareEvalCallback, eval_callback)
    callback.horizon = 1000
    callback.stage = stage
    callback.curriculum_kwargs = dict(curriculum_kwargs)
    callback.plateau_window = plateau_window
    callback.min_relative_variation = min_relative_variation
    callback._last_seen_n_evals = 0
    callback._histories = {key: [] for key in callback._METRIC_PRIORITY}
    callback._plateau_active = False
    callback._plateau_metric = None
    if eval_diagnostics_module._SB3_AVAILABLE:
        model = MagicMock()
        model.logger = MagicMock()
        callback.init_callback(model)
    else:
        callback.__dict__["logger"] = MagicMock()
    callback.num_timesteps = 100_000
    return callback, eval_callback


def _add_evaluation(
    callback: StageGatePlateauCallback,
    eval_callback: SimpleNamespace,
    *,
    reward: float,
    length: float,
    forward_vel: float | None = None,
    success_rate: float | None = None,
    n_episodes: int = 10,
    forward_vel_episodes: int | None = None,
    success_episodes: int | None = None,
    unsupported_duty: float | None = None,
    bilateral_duty: float | None = None,
) -> None:
    eval_callback.evaluations_results.append([reward] * n_episodes)
    eval_callback.evaluations_length.append([length] * n_episodes)
    if unsupported_duty is not None:
        eval_callback.evaluations_unsupported_duties.append([unsupported_duty] * n_episodes)
    if bilateral_duty is not None:
        eval_callback.evaluations_bilateral_duties.append([bilateral_duty] * n_episodes)
    if forward_vel is not None:
        count = n_episodes if forward_vel_episodes is None else forward_vel_episodes
        eval_callback.evaluations_forward_velocities.append([forward_vel] * count)
    if success_rate is not None:
        count = n_episodes if success_episodes is None else success_episodes
        eval_callback.evaluations_successes.append([success_rate] * count)
    callback.num_timesteps += 50_000
    assert callback._on_step() is True


class TestStageGatePlateauCallback:
    def test_waits_for_full_evaluation_window(self, caplog):
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 100.0, "min_success_rate": 0.5},
            stage=3,
        )

        for _ in range(2):
            _add_evaluation(callback, eval_callback, reward=200.0, length=1000.0, success_rate=0.0)

        assert "EVALUATION PLATEAU" not in caplog.text

    def test_warns_once_for_stable_blocking_success_gate(self, caplog):
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 100.0, "min_success_rate": 0.5},
            stage=3,
        )

        with caplog.at_level("WARNING", logger="environments.shared.eval_diagnostics"):
            for _ in range(4):
                _add_evaluation(callback, eval_callback, reward=200.0, length=1000.0, success_rate=0.0)

        warnings = [record.message for record in caplog.records if "EVALUATION PLATEAU" in record.message]
        assert len(warnings) == 1
        assert "STAGE 3" in warnings[0]
        assert "success rate" in warnings[0]
        assert "latest 0.0%, required 50.0%" in warnings[0]
        assert "Training continues" in warnings[0]
        assert callback._plateau_active is True

    def test_rearms_only_after_meaningful_movement(self, caplog):
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 100.0, "min_success_rate": 0.5},
            stage=3,
        )

        with caplog.at_level("WARNING", logger="environments.shared.eval_diagnostics"):
            for _ in range(3):
                _add_evaluation(callback, eval_callback, reward=200.0, length=1000.0, success_rate=0.0)
            _add_evaluation(callback, eval_callback, reward=200.0, length=1000.0, success_rate=0.1)
            for _ in range(3):
                _add_evaluation(callback, eval_callback, reward=200.0, length=1000.0, success_rate=0.0)

        warnings = [record for record in caplog.records if "EVALUATION PLATEAU" in record.message]
        assert len(warnings) == 2
        assert callback._plateau_active is True

    def test_relative_range_uses_actual_success_gate_scale(self):
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 100.0, "min_success_rate": 0.5},
            stage=3,
        )

        for success_rate in (0.0, 0.04, 0.0):
            _add_evaluation(callback, eval_callback, reward=200.0, length=1000.0, success_rate=success_rate)

        callback.logger.record.assert_any_call("diagnostics/eval_plateau_relative_range", pytest.approx(0.08))

    def test_does_not_warn_when_all_configured_gates_pass(self, caplog):
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 100.0, "min_success_rate": 0.5},
            stage=3,
        )

        with caplog.at_level("WARNING", logger="environments.shared.eval_diagnostics"):
            for _ in range(3):
                _add_evaluation(callback, eval_callback, reward=200.0, length=1000.0, success_rate=0.6)

        assert "EVALUATION PLATEAU" not in caplog.text
        callback.logger.record.assert_any_call("diagnostics/eval_gate_met", 1.0)

    def test_stage_two_follows_forward_velocity_gate(self, caplog):
        callback, eval_callback = _plateau_callback(
            {
                "min_avg_reward": 100.0,
                "min_avg_episode_length": 750.0,
                "min_avg_forward_vel": 2.0,
            },
            stage=2,
        )

        with caplog.at_level("WARNING", logger="environments.shared.eval_diagnostics"):
            for _ in range(3):
                _add_evaluation(
                    callback,
                    eval_callback,
                    reward=200.0,
                    length=1000.0,
                    forward_vel=0.5,
                )

        assert "forward velocity" in caplog.text
        assert "required 2.000 m/s" in caplog.text

    def test_stage_one_prefers_balance_duration_over_reward(self, caplog):
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 100.0, "min_avg_episode_length": 750.0},
            stage=1,
        )

        with caplog.at_level("WARNING", logger="environments.shared.eval_diagnostics"):
            for _ in range(3):
                _add_evaluation(callback, eval_callback, reward=50.0, length=500.0)

        assert "episode length" in caplog.text
        assert "required 750.0 steps" in caplog.text

    def test_relative_range_prevents_warning_when_metric_moves(self, caplog):
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 100.0},
            stage=1,
        )

        with caplog.at_level("WARNING", logger="environments.shared.eval_diagnostics"):
            for reward in (50.0, 53.0, 56.0):
                _add_evaluation(callback, eval_callback, reward=reward, length=1000.0)

        assert "EVALUATION PLATEAU" not in caplog.text

    def test_zero_reward_gate_is_still_active(self, caplog):
        callback, eval_callback = _plateau_callback({"min_avg_reward": 0.0}, stage=1)

        with caplog.at_level("WARNING", logger="environments.shared.eval_diagnostics"):
            for _ in range(3):
                _add_evaluation(callback, eval_callback, reward=-1.0, length=1000.0)

        assert "mean reward" in caplog.text
        assert "required 0.000" in caplog.text

    def test_missing_active_gate_metric_skips_safely(self, caplog):
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 100.0, "min_avg_forward_vel": 2.0},
            stage=2,
        )

        with caplog.at_level("WARNING", logger="environments.shared.eval_diagnostics"):
            for _ in range(3):
                _add_evaluation(callback, eval_callback, reward=50.0, length=1000.0)

        assert "EVALUATION PLATEAU" not in caplog.text
        callback.logger.record.assert_any_call("diagnostics/eval_gate_data_complete", 0.0)

    def test_undersized_evaluation_cannot_claim_gate_met(self, caplog):
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 100.0, "min_eval_episodes": 10},
            stage=1,
        )

        for _ in range(3):
            _add_evaluation(callback, eval_callback, reward=200.0, length=1000.0, n_episodes=5)

        assert "EVALUATION PLATEAU" not in caplog.text
        callback.logger.record.assert_any_call("diagnostics/eval_gate_data_complete", 0.0)
        assert not any(
            call.args == ("diagnostics/eval_gate_met", 1.0) for call in callback.logger.record.call_args_list
        )

    def test_each_active_gate_requires_minimum_finite_samples(self):
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 100.0, "min_avg_forward_vel": 2.0, "min_eval_episodes": 10},
            stage=2,
        )

        for _ in range(3):
            _add_evaluation(
                callback,
                eval_callback,
                reward=200.0,
                length=1000.0,
                forward_vel=3.0,
                forward_vel_episodes=5,
            )

        callback.logger.record.assert_any_call("diagnostics/eval_gate_data_complete", 0.0)
        assert not any(
            call.args == ("diagnostics/eval_gate_met", 1.0) for call in callback.logger.record.call_args_list
        )

    def test_repeated_steps_without_new_eval_do_not_change_history(self):
        callback, eval_callback = _plateau_callback({"min_avg_reward": 100.0}, stage=1)
        _add_evaluation(callback, eval_callback, reward=50.0, length=1000.0)
        before = list(callback._histories["mean_reward"])

        for _ in range(10):
            assert callback._on_step() is True

        assert callback._histories["mean_reward"] == before

    def test_rejects_invalid_window(self):
        pytest.importorskip("stable_baselines3")
        with pytest.raises(ValueError, match="at least 2"):
            StageGatePlateauCallback(
                MagicMock(spec=StageAwareEvalCallback),
                stage=1,
                curriculum_kwargs={"min_avg_reward": 100.0},
                plateau_window=1,
            )


class TestStanceGateIsACeilingNotAFloor:
    """Duty must fail when it is too HIGH, unlike every other gate metric.

    The plateau machinery originally tested ``value < threshold`` for every
    metric. Read that way a policy chattering at duty 0.30 against a 0.02
    ceiling looks comfortably passing, and the callback would follow
    ``mean_reward`` instead -- handing out reward-tuning advice for a stage
    whose blocking criterion is foot contact.
    """

    def test_gate_metric_direction(self):
        floor = eval_diagnostics_module._GateMetric("mean_reward", "reward", 1950.0, 1200.0, 40)
        assert floor.is_failing()
        assert not eval_diagnostics_module._GateMetric("mean_reward", "reward", 1950.0, 3271.8, 40).is_failing()

        ceiling = eval_diagnostics_module._GateMetric(
            "mean_unsupported_duty", "unsupported duty", 0.02, 0.319, 40, ceiling=True
        )
        assert ceiling.is_failing()
        assert not eval_diagnostics_module._GateMetric(
            "mean_unsupported_duty", "unsupported duty", 0.02, 0.0, 40, ceiling=True
        ).is_failing()

    def test_missing_value_is_not_failing_in_either_direction(self):
        for ceiling in (False, True):
            metric = eval_diagnostics_module._GateMetric("k", "l", 0.02, None, 0, ceiling=ceiling)
            assert not metric.is_failing()

    def test_duty_blocks_ahead_of_reward_on_a_stance_stage(self, caplog):
        callback, eval_callback = _plateau_callback(
            {
                "min_avg_reward": 1950.0,
                "min_full_horizon_fraction": 0.95,
                "max_unsupported_duty": 0.02,
            },
            stage=1,
        )
        with caplog.at_level("WARNING", logger="environments.shared.eval_diagnostics"):
            for _ in range(4):
                # Clears the reward rail and the horizon floor; only duty fails.
                _add_evaluation(
                    callback,
                    eval_callback,
                    reward=2133.4,
                    length=1000.0,
                    unsupported_duty=0.319,
                )
        warnings = [r.message for r in caplog.records if "EVALUATION PLATEAU" in r.message]
        assert len(warnings) == 1
        assert "unsupported duty" in warnings[0]
        assert callback._plateau_metric == "mean_unsupported_duty"

    def test_a_stage_without_stance_thresholds_ignores_duty(self):
        callback, eval_callback = _plateau_callback({"min_avg_reward": 100.0}, stage=2)
        _add_evaluation(callback, eval_callback, reward=200.0, length=1000.0, unsupported_duty=0.9)
        keys = {metric.key for metric in callback._metrics_for_evaluation(0)}
        assert "mean_unsupported_duty" not in keys
        assert "full_horizon_fraction" not in keys

    def test_full_horizon_fraction_is_derived_from_episode_lengths(self):
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 1950.0, "min_full_horizon_fraction": 0.95}, stage=1
        )
        eval_callback.evaluations_results.append([3271.8] * 40)
        # 38 reach the 1000-step horizon, 2 fall at 400.
        eval_callback.evaluations_length.append([1000.0] * 38 + [400.0] * 2)
        metric = next(m for m in callback._metrics_for_evaluation(0) if m.key == "full_horizon_fraction")
        assert metric.value == pytest.approx(0.95)
        assert not metric.is_failing()


class TestStanceShareCapture:
    """Bilateral duty is captured to CALIBRATE a future threshold.

    It is not a gate criterion. The three stance shares sum to 1, so a falling
    unsupported duty says nothing about whether the feet are being planted --
    that time can land in single support. Bounding bilateral duty is the
    obvious fix, but the only reference available is the statue's 0.998, and
    the statue never shifts weight. Recording the share every evaluation is
    what turns the eventual threshold into evidence instead of a guess.
    """

    def test_both_shares_are_captured_per_episode(self):
        callback = _bare_stage_aware_callback(success_applicable=False, settle_steps=0)
        # Two steps planted on both feet, then one airborne, then done.
        for forces, done in (((400.0, 400.0), False), ((400.0, 400.0), False), ((0.0, 0.0), True)):
            callback._log_success_callback(
                {"i": 0, "done": done, "info": {"r_foot_contact": forces[0], "l_foot_contact": forces[1]}},
                {},
            )
        assert callback._current_eval_unsupported_duties == [pytest.approx(1 / 3)]
        assert callback._current_eval_bilateral_duties == [pytest.approx(2 / 3)]

    def test_shares_sum_with_single_support(self):
        """Single support counts toward neither share, which is the point."""
        callback = _bare_stage_aware_callback(success_applicable=False, settle_steps=0)
        for forces, done in (((400.0, 0.0), False), ((400.0, 400.0), False), ((0.0, 0.0), True)):
            callback._log_success_callback(
                {"i": 0, "done": done, "info": {"r_foot_contact": forces[0], "l_foot_contact": forces[1]}},
                {},
            )
        unsupported = callback._current_eval_unsupported_duties[0]
        bilateral = callback._current_eval_bilateral_duties[0]
        assert unsupported == pytest.approx(1 / 3)
        assert bilateral == pytest.approx(1 / 3)
        # The missing third is single support — the share a bilateral floor
        # would catch and an unsupported ceiling would not.
        assert 1.0 - unsupported - bilateral == pytest.approx(1 / 3)

    def test_shares_are_logged_even_when_the_gate_panel_is_incomplete(self):
        """The early, incomplete evaluations are the calibration data."""
        callback, eval_callback = _plateau_callback({"min_avg_reward": 100.0, "min_eval_episodes": 40}, stage=1)
        _add_evaluation(
            callback,
            eval_callback,
            reward=200.0,
            length=1000.0,
            n_episodes=5,  # below min_eval_episodes -> gate data incomplete
            unsupported_duty=0.19,
            bilateral_duty=0.66,
        )
        recorded = dict(call.args for call in callback.logger.record.call_args_list)
        assert recorded["diagnostics/eval_gate_data_complete"] == 0.0
        assert recorded["diagnostics/eval_unsupported_duty"] == pytest.approx(0.19)
        assert recorded["diagnostics/eval_bilateral_support_duty"] == pytest.approx(0.66)
