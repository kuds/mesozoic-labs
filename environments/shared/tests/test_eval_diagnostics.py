"""Tests for stage-aware SB3 evaluation diagnostics."""

import logging
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

    def test_missing_active_gate_metric_still_follows_what_is_measurable(self, caplog):
        # Forward velocity has no samples, but reward does and is plateaued
        # below its gate. Vetoing the whole diagnostic because one metric is
        # unmeasurable is what silenced stage 1a entirely in early training;
        # the warning instead names what could not be measured, so the reader
        # knows a more specific criterion may be the real blocker.
        callback, eval_callback = _plateau_callback(
            {"min_avg_reward": 100.0, "min_avg_forward_vel": 2.0},
            stage=2,
        )

        with caplog.at_level("WARNING", logger="environments.shared.eval_diagnostics"):
            for _ in range(3):
                _add_evaluation(callback, eval_callback, reward=50.0, length=1000.0)

        assert "mean reward" in caplog.text
        assert "forward velocity could not be measured" in caplog.text
        assert "may be the real blocker" in caplog.text
        # The panel is still honestly reported as incomplete.
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


class TestStanceScalarsMatchTheGate:
    """What the callback publishes must be what the curriculum decides."""

    HORIZON = 1000
    CURRICULUM = {
        "gate_kind": "stance_quality/v1",
        "min_full_horizon_fraction": 0.95,
        "max_unsupported_duty": 0.02,
        "max_unsupported_duty_ucb": 0.02,
        "settle_steps": 200,
        "min_eval_episodes": 40,
        "min_avg_reward": 1950.0,
    }

    def _callback(self, *, lengths, duties, bilateral=None, curriculum=None):
        pytest.importorskip("stable_baselines3")
        from unittest.mock import MagicMock

        from environments.shared.eval_diagnostics import StageGatePlateauCallback

        eval_callback = MagicMock()
        eval_callback.evaluations_results = [[3000.0] * len(lengths)]
        eval_callback.evaluations_length = [lengths]
        eval_callback.evaluations_forward_velocities = [[]]
        eval_callback.evaluations_successes = []
        eval_callback.evaluations_unsupported_duties = [duties]
        eval_callback.evaluations_bilateral_duties = [bilateral if bilateral is not None else [0.5] * len(lengths)]

        callback = StageGatePlateauCallback(
            eval_callback,
            stage=1,
            curriculum_kwargs=curriculum if curriculum is not None else self.CURRICULUM,
            horizon=self.HORIZON,
        )
        recorded: dict = {}
        callback.model = MagicMock()
        callback.model.logger.record.side_effect = lambda key, value: recorded.setdefault(key, []).append(value)
        callback.num_timesteps = 0
        return callback, recorded

    def test_the_duty_scalar_is_the_gated_value(self):
        # 35 clean full-horizon episodes at 0.01 plus 5 that die at 400 while
        # chattering at 0.60. The scalar used to average all 40 and read
        # 0.084 against a 0.02 ceiling the policy was actually clearing.
        from environments.shared.curriculum.stance_gate import stance_panel_from_episode_duties

        lengths = [1000.0] * 35 + [400.0] * 5
        duties = [0.01] * 35 + [0.60] * 5
        callback, recorded = self._callback(lengths=lengths, duties=duties)
        callback._process_evaluation(0)

        panel = stance_panel_from_episode_duties(
            episode_lengths=lengths,
            episode_duties=duties,
            episode_rewards=[3000.0] * 40,
            horizon=self.HORIZON,
        )
        assert recorded["diagnostics/eval_unsupported_duty"][0] == pytest.approx(panel.mean_unsupported_duty)
        assert recorded["diagnostics/eval_unsupported_duty"][0] == pytest.approx(0.01)

    def test_every_gating_quantity_is_published(self):
        lengths = [1000.0] * 40
        callback, recorded = self._callback(lengths=lengths, duties=[0.01] * 40)
        callback._process_evaluation(0)

        for scalar in (
            "diagnostics/eval_unsupported_duty",
            "diagnostics/eval_unsupported_duty_ucb",
            "diagnostics/eval_full_horizon_fraction",
            "diagnostics/eval_duty_episodes",
        ):
            assert scalar in recorded, scalar

    def test_bilateral_uses_the_same_episode_set_as_the_duty(self):
        # The shares are only readable together if they are measured over the
        # same episodes.
        lengths = [1000.0] * 35 + [400.0] * 5
        callback, recorded = self._callback(
            lengths=lengths,
            duties=[0.0] * 35 + [0.5] * 5,
            bilateral=[1.0] * 35 + [0.0] * 5,
        )
        callback._process_evaluation(0)
        assert recorded["diagnostics/eval_bilateral_support_duty"][0] == pytest.approx(1.0)

    def test_duty_episodes_is_published_even_when_zero(self):
        # The early-training signal that nothing survived the settling window.
        lengths = [120.0] * 40
        callback, recorded = self._callback(lengths=lengths, duties=[float("nan")] * 40)
        callback._process_evaluation(0)
        assert recorded["diagnostics/eval_duty_episodes"] == [0]
        assert recorded["diagnostics/eval_full_horizon_fraction"] == [0.0]


class TestPlateauSurvivesAnUnmeasurableMetric:
    """One unmeasurable metric must not switch the whole diagnostic off."""

    HORIZON = 1000

    def _run(self, *, lengths, duties, curriculum, evaluations=8):
        pytest.importorskip("stable_baselines3")
        from unittest.mock import MagicMock

        from environments.shared.eval_diagnostics import StageGatePlateauCallback

        eval_callback = MagicMock()
        eval_callback.evaluations_results = [[300.0] * len(lengths) for _ in range(evaluations)]
        eval_callback.evaluations_length = [list(lengths) for _ in range(evaluations)]
        eval_callback.evaluations_forward_velocities = [[] for _ in range(evaluations)]
        eval_callback.evaluations_successes = []
        eval_callback.evaluations_unsupported_duties = [list(duties) for _ in range(evaluations)]
        eval_callback.evaluations_bilateral_duties = [list(duties) for _ in range(evaluations)]

        callback = StageGatePlateauCallback(
            eval_callback, stage=1, curriculum_kwargs=curriculum, plateau_window=5, horizon=self.HORIZON
        )
        recorded: dict = {}
        callback.model = MagicMock()
        callback.model.logger.record.side_effect = lambda key, value: recorded.setdefault(key, []).append(value)
        callback.num_timesteps = 0

        warnings: list[str] = []
        handler = type("H", (logging.Handler,), {"emit": lambda _self, record: warnings.append(record.getMessage())})()
        diagnostics_logger = logging.getLogger("environments.shared.eval_diagnostics")
        diagnostics_logger.addHandler(handler)
        try:
            for index in range(evaluations):
                callback._process_evaluation(index)
        finally:
            diagnostics_logger.removeHandler(handler)
        return recorded, [message for message in warnings if "PLATEAU" in message]

    STANCE = {
        "gate_kind": "stance_quality/v1",
        "min_full_horizon_fraction": 0.95,
        "max_unsupported_duty": 0.02,
        "max_unsupported_duty_ucb": 0.02,
        "settle_steps": 200,
        "min_eval_episodes": 40,
        "min_avg_reward": 1950.0,
    }

    def test_a_dying_run_still_reports_the_length_plateau(self):
        # Episodes end at 120 steps, before settle_steps=200, so every duty is
        # NaN. Adding duty to the priority list used to make _process_evaluation
        # return early, silencing the length plateau warning that had worked --
        # in exactly the phase where a stuck policy needs flagging.
        recorded, warnings = self._run(lengths=[120.0] * 40, duties=[float("nan")] * 40, curriculum=self.STANCE)
        assert warnings, "the survival plateau must still be reported"
        assert "full-horizon episodes" in warnings[0]
        assert "episodes are ending early" in warnings[0]
        assert recorded["diagnostics/eval_gate_met"] == [0.0] * 8
        # Still honestly marked as an incomplete panel.
        assert recorded["diagnostics/eval_gate_data_complete"] == [0.0] * 8

    def test_a_partial_panel_is_never_reported_as_meeting_the_gate(self):
        # Length and reward clear their criteria; duty is unmeasurable. The
        # gate is not met, because it was not fully checked.
        recorded, _ = self._run(
            lengths=[1000.0] * 40,
            duties=[float("nan")] * 40,
            curriculum={**self.STANCE, "min_avg_reward": -1e9},
        )
        assert recorded["diagnostics/eval_gate_data_complete"] == [0.0] * 8
        assert recorded["diagnostics/eval_gate_met"] == [0.0] * 8

    def test_a_complete_panel_that_clears_everything_is_reported_met(self):
        recorded, warnings = self._run(
            lengths=[1000.0] * 40,
            duties=[0.001] * 40,
            curriculum={**self.STANCE, "min_avg_reward": -1e9},
        )
        assert recorded["diagnostics/eval_gate_data_complete"] == [1.0] * 8
        assert recorded["diagnostics/eval_gate_met"] == [1.0] * 8
        assert not warnings


class TestDiagnosticAgreesWithTheGate:
    """The curve must not contradict the gate it reports on.

    ``evaluate_stance_gate`` certifies on
    ``ceil(min_eval_episodes x min_full_horizon_fraction)`` duty episodes --
    38 of 40 for trex 1a. Requiring the full panel size here made a 39-of-40
    panel PASS the gate while ``eval_gate_met`` read 0, which is the same
    curve-vs-gate disagreement the stance scalars were fixed for.
    """

    HORIZON = 1000
    CURRICULUM = {
        "gate_kind": "stance_quality/v1",
        "min_full_horizon_fraction": 0.95,
        "max_unsupported_duty": 0.02,
        "max_unsupported_duty_ucb": 0.02,
        "settle_steps": 200,
        "min_eval_episodes": 40,
        "min_avg_reward": -1e9,
    }

    def _both_verdicts(self, n_short):
        pytest.importorskip("stable_baselines3")
        from unittest.mock import MagicMock

        from environments.shared.curriculum.stance_gate import (
            StanceGateThresholds,
            evaluate_stance_gate,
            stance_panel_from_episode_duties,
        )
        from environments.shared.eval_diagnostics import StageGatePlateauCallback

        lengths = [1000.0] * (40 - n_short) + [400.0] * n_short
        duties = [0.001] * (40 - n_short) + [0.5] * n_short
        rewards = [3000.0] * 40

        panel = stance_panel_from_episode_duties(
            episode_lengths=lengths, episode_duties=duties, episode_rewards=rewards, horizon=self.HORIZON
        )
        gate_passes, _ = evaluate_stance_gate(
            panel,
            StanceGateThresholds(
                min_full_horizon_fraction=0.95,
                max_unsupported_duty=0.02,
                max_unsupported_duty_ucb=0.02,
                settle_steps=200,
                min_eval_episodes=40,
                min_avg_reward=-1e9,
            ),
        )

        eval_callback = MagicMock()
        eval_callback.evaluations_results = [rewards]
        eval_callback.evaluations_length = [lengths]
        eval_callback.evaluations_forward_velocities = [[]]
        eval_callback.evaluations_successes = []
        eval_callback.evaluations_unsupported_duties = [duties]
        eval_callback.evaluations_bilateral_duties = [[0.9] * 40]

        callback = StageGatePlateauCallback(
            eval_callback, stage=1, curriculum_kwargs=self.CURRICULUM, horizon=self.HORIZON
        )
        recorded: dict = {}
        callback.model = MagicMock()
        callback.model.logger.record.side_effect = lambda key, value: recorded.setdefault(key, []).append(value)
        callback.num_timesteps = 0
        callback._process_evaluation(0)
        return gate_passes, recorded.get("diagnostics/eval_gate_met", [None])[0]

    @pytest.mark.parametrize("n_short", [0, 1, 2, 3, 5])
    def test_eval_gate_met_matches_the_gate_verdict(self, n_short):
        gate_passes, reported = self._both_verdicts(n_short)
        assert reported == (1.0 if gate_passes else 0.0), (
            f"{n_short} early failures: gate says {gate_passes}, curve says {reported}"
        )

    def test_the_boundary_is_the_gates_own_rule(self):
        from environments.shared.curriculum.stance_gate import required_duty_episodes

        assert required_duty_episodes(40, 0.95) == 38
        # 38 duty episodes is enough for both; 37 is enough for neither.
        assert self._both_verdicts(2) == (True, 1.0)
        assert self._both_verdicts(3) == (False, 0.0)
