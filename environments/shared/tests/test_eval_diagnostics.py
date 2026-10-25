"""Tests for stage-aware SB3 evaluation diagnostics."""

import logging
import math
from dataclasses import fields, replace
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
from environments.shared.model_id import ModelId

VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")


class TestSuccessMetricApplicable:
    def test_requires_positive_configured_gate(self):
        assert success_metric_applicable({"curriculum_kwargs": {"min_success_rate": 0.5}}) is True
        assert success_metric_applicable({"curriculum_kwargs": {"min_success_rate": 0.0}}) is False
        assert success_metric_applicable({"curriculum_kwargs": {}}) is False
        assert success_metric_applicable({}) is False

    def test_task_success_kind_applies_by_kind(self):
        """task_success/v1 gates on the sample through min_success_lcb, not a rate floor."""
        assert success_metric_applicable(
            {"curriculum_kwargs": {"gate_kind": "task_success/v1", "min_success_lcb": 0.5}}
        )

    def test_the_plateau_follows_the_lcb_bar_for_a_task_success_stage(self):
        callback, eval_callback = _plateau_callback(
            {"gate_kind": "task_success/v1", "min_success_lcb": 0.5, "min_eval_episodes": 30, "min_avg_reward": 361.0},
            stage=3,
        )
        eval_callback.evaluations_successes.append([1.0] * 20 + [0.0] * 10)
        metric = next(m for m in callback._metrics_for_evaluation(0) if m.key == "mean_success_rate")
        assert metric.label == "success rate (LCB95 bar)"
        assert metric.threshold == 0.5

    @pytest.mark.parametrize(
        "model_id",
        [VELOCIRAPTOR_MONGOLIENSIS, TYRANNOSAURUS_REX, BRACHIOSAURUS_ALTITHORAX],
        ids=lambda m: f"{m.species}-{m.embodiment}",
    )
    def test_real_configs_enable_success_only_for_stage_three(self, model_id):
        from environments.shared.config import load_all_stages

        configs = load_all_stages(model_id)

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
    callback.evaluations_action_dc = []
    callback.evaluations_action_ac_rms = []
    callback.evaluations_action_delta = []
    callback.evaluations_action_jerk = []
    callback.evaluations_reward_terms = []
    callback._action_sum = None
    callback._action_sq_sum = None
    callback._action_delta_sum = 0.0
    callback._action_jerk_sum = 0.0
    callback._action_count = 0
    callback._action_delta_count = 0
    callback._action_jerk_count = 0
    callback._prev_action = {}
    callback._prev_prev_action = {}
    callback._reward_term_sums = {}
    callback._reward_term_episodes = 0
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
    stage: "int | str",
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


class TestPlateauMessagesRenderForSemanticStages:
    """Regression: ``%d`` on a string stage raised inside ``getMessage()``.

    The recovery stage reaches the plateau callback as ``"recovery"``, and
    logging swallowed the TypeError into a '--- Logging error ---' traceback
    on stderr — replacing the one diagnostic the run needed.
    """

    def test_plateau_warning_and_cleared_message_render_for_recovery(self, caplog):
        callback, eval_callback = _plateau_callback({"min_avg_reward": 100.0}, stage="recovery")

        with caplog.at_level(logging.INFO, logger="environments.shared.eval_diagnostics"):
            for _ in range(3):
                _add_evaluation(callback, eval_callback, reward=50.0, length=1000.0)
            assert callback._plateau_active is True
            _add_evaluation(callback, eval_callback, reward=90.0, length=1000.0)
            assert callback._plateau_active is False

        # getMessage() is exactly what raised; render every record through it.
        messages = [record.getMessage() for record in caplog.records]
        assert any("STAGE recovery EVALUATION PLATEAU" in message for message in messages)
        assert any("Stage recovery evaluation plateau cleared" in message for message in messages)


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


class TestRecoveryStageStillGetsEvalDiagnostics:
    """A stage with no legacy scalar thresholds must not train dark.

    The recovery stage's ``recovery_quality/v1`` gate is judged post-stage
    from its frozen resolution, so its curriculum block declares none of the
    plateau metrics.  The metrics early-out in ``_process_evaluation`` used
    to fire before ``_record_stance_scalars``, so no ``diagnostics/eval_*``
    scalar and no ``gate_progress.npz`` append ever happened for the entire
    stage (gap review EE1) — the exact "gate criteria stayed invisible for
    the whole run" failure the surrounding machinery exists to prevent.
    """

    RECOVERY_CURRICULUM = {
        "gate_kind": "recovery_quality/v1",
        "min_recovery_success_lcb": 0.30,
        "recovery_t_recover_steps": 100,
        "recovery_dwell_steps": 50,
        "min_eval_episodes": 40,
    }

    def test_scalars_are_recorded_despite_no_plateau_metrics(self):
        harness = TestStanceScalarsMatchTheGate()
        callback, recorded = harness._callback(
            lengths=[1000.0] * 40,
            duties=[0.01] * 40,
            curriculum=self.RECOVERY_CURRICULUM,
        )
        callback._process_evaluation(0)

        # The metrics list is empty for this gate kind (nothing to follow
        # for plateau tracking) — but the per-eval capture must publish.
        assert "diagnostics/eval_unsupported_duty" in recorded
        assert "diagnostics/eval_full_horizon_fraction" in recorded
        assert "diagnostics/eval_duty_episodes" in recorded


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


class TestGateProgressIsPersisted:
    """The deterministic gate criteria must survive outside TensorBoard.

    They were computed every evaluation and recorded only to the SB3 logger,
    so post-hoc analysis of run 20260804_143747 had to substitute the
    TRAINING-rollout duty from `diagnostics.npz` — contaminated by exploration
    noise. It happened to agree to three decimals there; that was luck, not a
    property to rely on (issue #486).
    """

    @staticmethod
    def _callback(tmp_path):
        from environments.shared.eval_diagnostics import StageGatePlateauCallback

        cb = object.__new__(StageGatePlateauCallback)
        cb._gate_progress_dir = tmp_path
        cb.num_timesteps = 0
        return cb

    def test_each_evaluation_appends_a_row_and_rewrites_the_file(self, tmp_path):
        cb = self._callback(tmp_path)
        for step, duty in ((50_000, 0.42), (100_000, 0.31), (150_000, 0.28)):
            cb.num_timesteps = step
            cb._record_gate_progress(
                full_horizon_fraction=1.0,
                duty_episodes=40,
                unsupported_duty=duty,
                unsupported_duty_ucb=duty + 0.001,
                bilateral_support_duty=1.0 - duty,
                mean_reward=2500.0,
            )
        data = np.load(tmp_path / "gate_progress.npz")
        assert list(data["timesteps"]) == [50_000, 100_000, 150_000]
        assert list(data["unsupported_duty"]) == pytest.approx([0.42, 0.31, 0.28])
        assert list(data["unsupported_duty_ucb"]) == pytest.approx([0.421, 0.311, 0.281])
        assert list(data["bilateral_support_duty"]) == pytest.approx([0.58, 0.69, 0.72])

    def test_instances_do_not_share_the_class_level_defaults(self, tmp_path):
        """The class attributes exist for `object.__new__`; appending to them
        directly would leak one run's series into another's."""
        a, b = self._callback(tmp_path / "a"), self._callback(tmp_path / "b")
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        a.num_timesteps = 1000
        a._record_gate_progress(unsupported_duty=0.5)
        b.num_timesteps = 2000
        b._record_gate_progress(unsupported_duty=0.1)
        assert a._gate_progress_timesteps == [1000]
        assert b._gate_progress_timesteps == [2000]

    @staticmethod
    def _unwritable_dir(tmp_path):
        """A target `atomic_copy` genuinely cannot create.

        Not a merely absent directory: `atomic_copy` calls
        `mkdir(parents=True)`, so a missing path is created rather than
        failing, and asserting "does not raise" against one exercises the
        success path under a name that claims otherwise. Rooting the target
        under a regular *file* makes mkdir raise NotADirectoryError, and
        unlike `chmod 000` it still fails when the suite runs as root.
        """
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("", encoding="utf-8")
        return blocker / "gate"

    def test_a_missing_directory_is_created_rather_than_dropped(self, tmp_path):
        target = tmp_path / "does" / "not" / "exist"
        cb = self._callback(target)
        cb.num_timesteps = 50_000
        cb._record_gate_progress(unsupported_duty=0.42)
        assert list(np.load(target / "gate_progress.npz")["unsupported_duty"]) == pytest.approx([0.42])

    def test_an_unwritable_directory_does_not_sink_the_run(self, tmp_path):
        cb = self._callback(self._unwritable_dir(tmp_path))
        cb.num_timesteps = 50_000
        cb._record_gate_progress(unsupported_duty=0.42)  # must not raise
        assert cb._gate_progress_timesteps == [50_000]

    def test_an_unwritable_directory_warns_once_not_per_evaluation(self, tmp_path, caplog):
        """A 10M run evaluates ~200 times and would fail identically each time.

        200 stack traces would bury the below-baseline warning this run is
        actually meant to surface.
        """
        cb = self._callback(self._unwritable_dir(tmp_path))
        with caplog.at_level(logging.WARNING, logger="environments.shared.eval_diagnostics"):
            for step in range(1, 6):
                cb.num_timesteps = step * 50_000
                cb._record_gate_progress(unsupported_duty=0.42)
        assert sum("gate_progress.npz" in r.message for r in caplog.records) == 1
        assert len(cb._gate_progress_timesteps) == 5  # still accumulating in memory

    def test_a_short_series_is_dropped_rather_than_written_misaligned(self, tmp_path):
        """A key supplied on only some evaluations must not read as the first n.

        Today one call site supplies every key every time, so this cannot
        happen -- but a silently misaligned duty series is exactly the kind of
        wrong number a kill/continue decision would be made on.
        """
        cb = self._callback(tmp_path)
        cb.num_timesteps = 50_000
        cb._record_gate_progress(unsupported_duty=0.42, duty_episodes=40)
        cb.num_timesteps = 100_000
        cb._record_gate_progress(unsupported_duty=0.31)  # duty_episodes absent
        data = np.load(tmp_path / "gate_progress.npz")
        assert list(data["timesteps"]) == [50_000, 100_000]
        assert list(data["unsupported_duty"]) == pytest.approx([0.42, 0.31])
        assert "duty_episodes" not in data

    def test_no_directory_still_accumulates_in_memory(self, tmp_path):
        cb = self._callback(None)
        cb.num_timesteps = 50_000
        cb._record_gate_progress(unsupported_duty=0.42)
        assert cb._gate_progress["unsupported_duty"] == [0.42]


class TestDeterministicActionStatistics:
    """The gate scores the deterministic policy; only training rollouts were logged.

    Issue #489 had to recover the commanded pose and the tremor about it by
    inverting per-episode reward totals under a narrowband assumption, because
    every action number on disk came from exploration-noisy training rollouts.
    """

    @staticmethod
    def _feed(callback, actions, *, env_index=0, done_at=None, info=None):
        for step, action in enumerate(actions):
            callback._log_success_callback(
                {
                    "i": env_index,
                    "done": done_at is not None and step == done_at,
                    "info": dict(info or {}),
                    "actions": np.array([action]),
                },
                {},
            )

    def test_dc_and_ac_are_separated(self):
        """A steady offset with a tremor on top must not read as one number."""
        callback = _bare_stage_aware_callback(success_applicable=False)
        # Two actuators: the first holds 0.5 and shakes +/-0.1, the second
        # holds 0.0 and is still. A pooled std cannot express that.
        self._feed(callback, [[0.6, 0.0], [0.4, 0.0]] * 10, done_at=19)
        callback._publish_action_statistics()
        dc = callback.evaluations_action_dc[0]
        ac = callback.evaluations_action_ac_rms[0]
        assert dc == pytest.approx([0.5, 0.0])
        assert ac == pytest.approx([0.1, 0.0])

    def test_differences_invert_through_the_reward_formulas(self):
        """Sums are over actuators, so they feed straight back into the terms."""
        callback = _bare_stage_aware_callback(success_applicable=False)
        # Alternating +/-1 on one actuator: |delta| = 2 every step after the
        # first, and the second difference is -/+4.
        self._feed(callback, [[1.0], [-1.0]] * 8, done_at=15)
        callback._publish_action_statistics()
        # Averaged over differencing opportunities, not samples: 15 pairs and
        # 14 triples, so the alternation reads as its exact amplitude.
        assert callback.evaluations_action_delta[0] == pytest.approx(4.0)
        assert callback.evaluations_action_jerk[0] == pytest.approx(16.0)

    def test_the_settling_window_is_excluded(self):
        """Same window as the duty: the reset transient is not the tremor."""
        callback = _bare_stage_aware_callback(success_applicable=False, settle_steps=4)
        self._feed(callback, [[9.0]] * 4 + [[1.0]] * 4, done_at=7)
        callback._publish_action_statistics()
        assert callback.evaluations_action_dc[0] == pytest.approx([1.0])

    def test_an_evaluation_with_no_actions_records_nan_not_zero(self):
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback._publish_action_statistics()
        assert callback.evaluations_action_dc[0] == []
        assert math.isnan(callback.evaluations_action_delta[0])

    def test_a_missing_actions_local_does_not_raise(self):
        """`actions` is a local of SB3's evaluate_policy, not a contract."""
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback._log_success_callback({"i": 0, "done": True, "info": {}}, {})
        callback._publish_action_statistics()
        assert callback.evaluations_action_dc[0] == []

    def test_episodes_do_not_leak_their_last_action_into_the_next(self):
        """The difference across an episode boundary is not a real difference."""
        callback = _bare_stage_aware_callback(success_applicable=False)
        self._feed(callback, [[1.0]], done_at=0)
        self._feed(callback, [[-1.0]], done_at=0)
        callback._publish_action_statistics()
        # Two episodes of one step each: no within-episode pair exists, so the
        # boundary jump of 2.0 must not be charged. NaN rather than 0.0,
        # because nothing was measured -- 0.0 would read as "perfectly smooth".
        assert math.isnan(callback.evaluations_action_delta[0])
        assert callback._action_delta_count == 0

    def test_reward_terms_are_averaged_per_episode(self):
        callback = _bare_stage_aware_callback(success_applicable=False)
        self._feed(callback, [[0.0]] * 4, done_at=3, info={"reward_energy": -1.0})
        callback._publish_action_statistics()
        assert callback.evaluations_reward_terms[0] == {"reward_energy": pytest.approx(-4.0)}


class TestGateProgressCarriesTheNewSeries:
    @staticmethod
    def _plateau(eval_callback, tmp_path):
        cb = object.__new__(StageGatePlateauCallback)
        cb._gate_progress_dir = tmp_path
        cb.num_timesteps = 50_000
        cb.eval_callback = eval_callback
        cb.control_dt = 0.01
        return cb

    def test_action_statistics_and_terms_reach_the_file(self, tmp_path):
        ev = _bare_stage_aware_callback(success_applicable=False)
        ev.evaluations_action_dc = [[0.8, 0.8]]
        ev.evaluations_action_ac_rms = [[0.2, 0.2]]
        ev.evaluations_action_delta = [1.0]
        ev.evaluations_action_jerk = [1.699]
        ev.evaluations_reward_terms = [{"reward_energy": -53.68}]
        cb = self._plateau(ev, tmp_path)
        cb._record_gate_progress(**cb._action_statistics(0), **cb._reward_term_scalars(0))
        data = np.load(tmp_path / "gate_progress.npz")
        assert data["action_dc_rms"][0] == pytest.approx(0.8)
        assert data["action_ac_rms"][0] == pytest.approx(0.2)
        assert data["term_energy"][0] == pytest.approx(-53.68)
        # jerk/delta = 1.699 -> 2 sin(pi f dt) = 1.3034 -> ~22.6 Hz, the
        # tremor issue #489 had to derive by inverting reward totals.
        assert data["action_freq_hz"][0] == pytest.approx(22.6, abs=0.2)
        assert data["action_dc_per_actuator"].shape == (1, 2)

    def test_a_constant_action_reports_zero_frequency_not_nan(self, tmp_path):
        ev = _bare_stage_aware_callback(success_applicable=False)
        ev.evaluations_action_dc = [[0.5]]
        ev.evaluations_action_ac_rms = [[0.0]]
        ev.evaluations_action_delta = [0.0]
        ev.evaluations_action_jerk = [0.0]
        ev.evaluations_reward_terms = [{}]
        cb = self._plateau(ev, tmp_path)
        stats = cb._action_statistics(0)
        # delta == 0 leaves the ratio undefined; NaN is the honest answer and
        # 0 Hz would read as a measured slow drift.
        assert math.isnan(stats["action_freq_hz"])
        assert stats["action_dc_rms"] == pytest.approx(0.5)


# ── pad telemetry and the sagittal leg DC ────────────────────────────────────


class _StubRecorder:
    """Stands in for SubstepContactRecorder: counts closed steps and episodes, optionally fails."""

    def __init__(self, *, fail_on_step: "int | None" = None) -> None:
        self.steps = 0
        self.episodes = 0
        self.attached = True
        self.fail_on_step = fail_on_step

    def end_step(self, action, reward):
        self.steps += 1
        if self.fail_on_step is not None and self.steps == self.fail_on_step:
            raise RuntimeError("the substep hook fired 0 times")

    def end_episode(self, **_kwargs):
        self.episodes += 1
        return f"trace{self.episodes}"

    def detach(self):
        self.attached = False


PAD_KEYS = ("r_sole_cop_fore_aft", "r_sole_cop_fore_aft_gate", "r_digit_share")


def _pad_capturing_callback(monkeypatch, recorders, *, quota=2):
    """A bare capture callback mid-evaluation, its pad reduction replaced by one value per episode."""
    from environments.shared.gait import pad_telemetry

    callback = _bare_stage_aware_callback(success_applicable=False)
    callback.foot_pad_keys = PAD_KEYS
    callback._foot_setup = [(None, None, None, "layout")] * len(recorders)
    callback._pad_recorders = dict(enumerate(recorders))
    callback._pad_quota = {index: quota for index in range(len(recorders))}
    callback._pad_failed = False
    callback._current_eval_foot_pad = {key: [] for key in PAD_KEYS}

    def reduce(trace, layout, *, settle_steps):
        number = float(trace.removeprefix("trace"))
        return {"r_sole_cop_fore_aft": number / 10, "r_sole_cop_fore_aft_gate": number / 5, "r_digit_share": 0.15}

    monkeypatch.setattr(pad_telemetry, "episode_pad_telemetry", reduce)
    return callback


def _step(callback, *, env_index=0, done=False):
    callback._log_success_callback(
        {"i": env_index, "done": done, "info": {}, "actions": np.zeros((1, 2)), "reward": 1.0}, {}
    )


class TestPadTelemetryCapture:
    """The evaluation callback's side: one value per pad key per episode, aligned with the episode lengths."""

    def test_each_episode_appends_one_value_per_key_and_a_finished_sub_env_is_detached(self, monkeypatch):
        recorder = _StubRecorder()
        callback = _pad_capturing_callback(monkeypatch, [recorder])
        for done in (False, False, True, False, True):
            _step(callback, done=done)
        assert recorder.steps == 5 and recorder.episodes == 2
        assert not recorder.attached and callback._pad_recorders == {}  # its share of the panel is done
        callback._publish_pad_telemetry()
        assert callback.evaluations_foot_pad == [
            {"r_sole_cop_fore_aft": [0.1, 0.2], "r_sole_cop_fore_aft_gate": [0.2, 0.4], "r_digit_share": [0.15, 0.15]}
        ]

    def test_an_episode_without_a_recorder_is_nan_but_keeps_its_place(self, monkeypatch):
        callback = _pad_capturing_callback(monkeypatch, [_StubRecorder()], quota=3)
        _step(callback, done=True)
        callback._pad_recorders.clear()
        _step(callback, done=True)
        values = callback._current_eval_foot_pad["r_sole_cop_fore_aft"]
        assert values[0] == 0.1 and math.isnan(values[1]) and len(values) == 2

    def test_a_recorder_failure_costs_the_evaluation_not_the_run(self, monkeypatch, caplog):
        good, bad = _StubRecorder(), _StubRecorder(fail_on_step=2)
        callback = _pad_capturing_callback(monkeypatch, [good, bad])
        _step(callback, env_index=0, done=True)
        _step(callback, env_index=1)
        with caplog.at_level(logging.WARNING, logger="environments.shared.eval_diagnostics"):
            _step(callback, env_index=1, done=True)  # raises inside the recorder
        assert callback._pad_failed
        assert not good.attached and not bad.attached and callback._pad_recorders == {}
        assert sum("could not be recorded" in record.message for record in caplog.records) == 1
        callback._publish_pad_telemetry()
        # Every value of the evaluation is NaN, the measured one too: a part-recorded panel is not published.
        (published,) = callback.evaluations_foot_pad
        assert all(len(values) == 2 and all(math.isnan(v) for v in values) for values in published.values())

    def test_a_roll_rail_failure_costs_only_the_roll_rails(self, monkeypatch, caplog):
        from environments.shared.gait import pad_telemetry

        callback = _pad_capturing_callback(monkeypatch, [_StubRecorder()], quota=3)
        callback.foot_rail_keys = ("r_sole_tilt_deg", "r_sole_corner_lift_m")
        callback._rail_layouts = ["rails"]

        def rails(trace, layout, *, settle_steps):
            if trace != "trace1":
                raise FloatingPointError("a corner reading went wrong")
            return {"r_sole_tilt_deg": 1.5, "r_sole_corner_lift_m": 0.002}

        monkeypatch.setattr(pad_telemetry, "episode_rail_telemetry", rails)
        with caplog.at_level(logging.WARNING, logger="environments.shared.eval_diagnostics"):
            for _ in range(3):
                _step(callback, done=True)
        assert not callback._pad_failed
        callback._publish_pad_telemetry()
        (published,) = callback.evaluations_foot_pad
        # The pad fields (#620) keep every episode; only the episodes whose roll rails failed read NaN there.
        assert published["r_sole_cop_fore_aft"] == pytest.approx([0.1, 0.2, 0.3])
        assert published["r_sole_tilt_deg"][0] == 1.5 and published["r_sole_corner_lift_m"][0] == 0.002
        assert all(
            math.isnan(value) for value in published["r_sole_tilt_deg"][1:] + published["r_sole_corner_lift_m"][1:]
        )
        assert sum("Roll-rail telemetry" in record.message for record in caplog.records) == 1

    def test_a_callback_without_a_model_records_no_pad_field(self):
        pytest.importorskip("stable_baselines3")
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback.eval_freq = 10
        callback.n_calls = 10
        callback.verbose = 0
        with patch.object(eval_diagnostics_module._EvalCallback, "_on_step", return_value=True):
            assert callback._on_step() is True
        assert callback.evaluations_foot_pad == [{}] and callback.evaluations_leg_dc == [{}]
        assert callback.foot_pad_keys == () and callback.leg_dc_pairs == ()

    def test_seeded_prior_evaluations_carry_an_empty_pad_record(self):
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback.seed_prior_history(2)
        assert callback.evaluations_foot_pad == [{}, {}]

    def test_an_env_that_cannot_be_recorded_warns_once_and_leaves_the_keys_empty(self, caplog):
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback.model_id = TYRANNOSAURUS_REX
        callback.eval_env = SimpleNamespace()  # neither a DummyVecEnv nor a wrapper of one
        with caplog.at_level(logging.WARNING, logger="environments.shared.eval_diagnostics"):
            callback._begin_pad_capture()
            callback._begin_pad_capture()
        assert callback.foot_pad_keys == () and callback.leg_dc_pairs == ()
        assert sum("Pad telemetry" in record.message for record in caplog.records) == 1


class TestPadTelemetryReachesGateProgress:
    """The plateau callback's side: full-horizon means of the pad values and the leg DC, in the npz and TB."""

    @staticmethod
    def _plateau(
        tmp_path,
        *,
        lengths,
        pads,
        leg_dc=None,
        keys=PAD_KEYS[:1],
        pairs=(eval_diagnostics_module.SagittalLegPair("leg_dc_l_minus_r_ankle_deg", 1, 0, 1.0, 1.0),),
        curriculum_kwargs=None,
    ):
        callback, eval_callback = _plateau_callback({"min_avg_reward": 0.0, **(curriculum_kwargs or {})}, stage=1)
        callback._gate_progress_dir = tmp_path
        eval_callback.evaluations_length.append(lengths)
        eval_callback.evaluations_foot_pad = [pads]
        eval_callback.evaluations_leg_dc = [{} if leg_dc is None else leg_dc]
        eval_callback.foot_pad_keys = keys
        eval_callback.leg_dc_pairs = pairs
        return callback

    def test_a_pad_value_is_the_mean_over_full_horizon_episodes_and_the_leg_dc_is_the_captured_one(self, tmp_path):
        callback = self._plateau(
            tmp_path,
            lengths=[1000, 1000, 300, 1000],
            pads={"r_sole_cop_fore_aft": [0.9, float("nan"), -1.0, 0.7]},  # the fall and the unmeasured one drop
            leg_dc={"leg_dc_l_minus_r_ankle_deg": -3.165},
        )
        scalars = callback._foot_telemetry_scalars(0)
        assert scalars["r_sole_cop_fore_aft"] == pytest.approx(0.8)
        assert scalars["leg_dc_l_minus_r_ankle_deg"] == -3.165

    @pytest.mark.parametrize(
        ("lengths", "pads", "leg_dc"),
        [
            ([300, 400], {"r_sole_cop_fore_aft": [0.5, 0.5]}, None),  # nothing reached the horizon; no DC captured
            ([1000], {"r_sole_cop_fore_aft": [0.5, 0.5]}, {"leg_dc_l_minus_r_ankle_deg": float("nan")}),  # misaligned
            ([1000], {}, {"leg_dc_l_minus_r_ankle_deg": 1.5}),  # an evaluation that recorded no pad value
        ],
    )
    def test_an_unmeasured_value_is_nan_and_its_key_is_still_supplied(self, tmp_path, lengths, pads, leg_dc):
        callback = self._plateau(tmp_path, lengths=lengths, pads=pads, leg_dc=leg_dc)
        scalars = callback._foot_telemetry_scalars(0)
        # No gate key and no panel key; the mirror-even leg DC beside the left-minus-right one (the extensions).
        assert set(scalars) == {"r_sole_cop_fore_aft", "leg_dc_l_minus_r_ankle_deg", "leg_dc_l_r_mean_ankle_deg"}
        assert math.isnan(scalars["r_sole_cop_fore_aft"])
        assert math.isnan(scalars["leg_dc_l_minus_r_ankle_deg"]) or leg_dc == {"leg_dc_l_minus_r_ankle_deg": 1.5}

    def test_the_values_reach_the_npz_beside_the_gate_criteria_and_tb_gets_the_finite_ones(self, tmp_path):
        from environments.shared.curriculum.stance_gate import stance_panel_from_episode_duties

        callback = self._plateau(tmp_path, lengths=[1000, 1000], pads={"r_sole_cop_fore_aft": [0.4, float("nan")]})
        panel = stance_panel_from_episode_duties(
            episode_lengths=[1000.0, 1000.0], episode_duties=[0.0, 0.0], episode_rewards=[1.0, 1.0], horizon=1000
        )
        callback._record_stance_scalars(0, panel)
        data = np.load(tmp_path / "gate_progress.npz")
        assert data["r_sole_cop_fore_aft"].tolist() == pytest.approx([0.4])
        assert math.isnan(data["leg_dc_l_minus_r_ankle_deg"][0])
        recorded = {call.args[0] for call in callback.logger.record.call_args_list}
        assert "diagnostics/eval_r_sole_cop_fore_aft" in recorded
        assert "diagnostics/eval_leg_dc_l_minus_r_ankle_deg" not in recorded  # NaN is not logged


GATE_KEYS = ("r_sole_cop_fore_aft_gate", "l_sole_cop_fore_aft_gate")


class TestThePanelsOwnCopStatistic:
    """``max_sole_cop_fore_aft_mean`` and its bar fraction: the rail's per-episode number, not the per-foot means'."""

    @staticmethod
    def _scalars(tmp_path, *, lengths, right, left, bar=None):
        callback = TestPadTelemetryReachesGateProgress._plateau(
            tmp_path,
            lengths=lengths,
            pads={"r_sole_cop_fore_aft_gate": right, "l_sole_cop_fore_aft_gate": left},
            keys=GATE_KEYS,
            pairs=(),
            curriculum_kwargs=None if bar is None else {"max_sole_cop_fore_aft": bar},
        )
        return callback._foot_telemetry_scalars(0)

    def test_the_worse_foot_changing_between_episodes_reads_above_the_max_of_the_per_foot_means(self, tmp_path):
        # The edge is on the right pad in two episodes and on the left in one, as on a seed whose side wanders.
        scalars = self._scalars(tmp_path, lengths=[1000] * 3, right=[0.95, 0.2, 0.9], left=[0.2, 0.95, 0.2])
        assert max(scalars[key] for key in GATE_KEYS) == pytest.approx(0.6833333)  # the right foot's mean
        assert scalars["max_sole_cop_fore_aft_mean"] == pytest.approx((0.95 + 0.95 + 0.9) / 3)
        assert "max_sole_cop_fore_aft_over_bar_fraction" not in scalars  # the stage declares no bar

    def test_the_fraction_counts_full_horizon_episodes_over_the_declared_bar_as_the_gate_compares(self, tmp_path):
        nan = float("nan")
        scalars = self._scalars(
            tmp_path,
            lengths=[1000, 1000, 1000, 1000, 1000, 250],
            # over the bar on one foot; at the bar exactly (holds: the gate fails only value > bar); under it;
            # measured on one foot only (the other NaN, as on an episode where it was not a box foot); unmeasured; a fall.
            right=[0.81, 0.80, 0.30, nan, nan, 1.0],
            left=[0.10, 0.20, 0.79, 0.95, nan, 1.0],
            bar=0.8,
        )
        assert scalars["max_sole_cop_fore_aft_mean"] == pytest.approx((0.81 + 0.80 + 0.79 + 0.95) / 4)
        assert scalars["max_sole_cop_fore_aft_over_bar_fraction"] == pytest.approx(2 / 4)

    @pytest.mark.parametrize(
        ("lengths", "right", "left"),
        [
            ([400, 300], [0.9, 0.9], [0.9, 0.9]),  # no episode reached the horizon
            ([1000, 1000], [float("nan")] * 2, [float("nan")] * 2),  # a failed capture publishes NaN
            ([1000, 1000], [0.5, 0.5], [0.5]),  # misaligned feet
        ],
    )
    def test_unmeasured_panels_read_nan_and_keep_both_keys(self, tmp_path, lengths, right, left):
        scalars = self._scalars(tmp_path, lengths=lengths, right=right, left=left, bar=0.7)
        assert math.isnan(scalars["max_sole_cop_fore_aft_mean"])
        assert math.isnan(scalars["max_sole_cop_fore_aft_over_bar_fraction"])

    @pytest.mark.parametrize("bar", [None, True, "0.8", float("nan")])
    def test_no_fraction_without_a_finite_declared_bar(self, tmp_path, bar):
        callback = TestPadTelemetryReachesGateProgress._plateau(
            tmp_path,
            lengths=[1000],
            pads={"r_sole_cop_fore_aft_gate": [0.9]},
            keys=GATE_KEYS[:1],
            pairs=(),
            curriculum_kwargs={} if bar is None else {"max_sole_cop_fore_aft": bar},
        )
        scalars = callback._foot_telemetry_scalars(0)
        assert scalars["max_sole_cop_fore_aft_mean"] == pytest.approx(0.9)
        assert "max_sole_cop_fore_aft_over_bar_fraction" not in scalars

    def test_a_model_without_a_pad_has_no_panel_key(self, tmp_path):
        callback = TestPadTelemetryReachesGateProgress._plateau(
            tmp_path, lengths=[1000], pads={}, keys=(), curriculum_kwargs={"max_sole_cop_fore_aft": 0.8}
        )
        assert set(callback._foot_telemetry_scalars(0)) == {"leg_dc_l_minus_r_ankle_deg", "leg_dc_l_r_mean_ankle_deg"}


FOOT_GATE_KEYS = ("r_foot_cop_fore_aft_gate", "l_foot_cop_fore_aft_gate")


class TestThePanelsOwnWholeFootStatistic:
    """``max_foot_cop_fore_aft_mean``: the whole-foot rail's per-episode number, report-only (no stage declares a bar)."""

    @staticmethod
    def _scalars(tmp_path, *, pads, keys, curriculum_kwargs=None):
        callback = TestPadTelemetryReachesGateProgress._plateau(
            tmp_path, lengths=[1000] * 3, pads=pads, keys=keys, pairs=(), curriculum_kwargs=curriculum_kwargs
        )
        return callback._foot_telemetry_scalars(0)

    def test_the_mean_of_each_episodes_worse_foot_and_no_fraction_without_a_bar(self, tmp_path):
        pads = {"r_foot_cop_fore_aft_gate": [0.13, 0.9, 0.2], "l_foot_cop_fore_aft_gate": [0.05, 0.1, 0.85]}
        scalars = self._scalars(tmp_path, pads=pads, keys=FOOT_GATE_KEYS)
        assert scalars["max_foot_cop_fore_aft_mean"] == pytest.approx((0.13 + 0.9 + 0.85) / 3)
        assert "max_foot_cop_fore_aft_over_bar_fraction" not in scalars
        # The pad rail's bar does not count for the whole foot's statistic.
        assert "max_foot_cop_fore_aft_over_bar_fraction" not in self._scalars(
            tmp_path, pads=pads, keys=FOOT_GATE_KEYS, curriculum_kwargs={"max_sole_cop_fore_aft": 0.8}
        )
        assert "max_sole_cop_fore_aft_mean" not in scalars  # no pad gate key: no pad panel key

    def test_a_declared_bar_would_add_its_fraction_beside_the_pad_rails(self, tmp_path):
        """No stage declares ``max_foot_cop_fore_aft``; the arm is the pad rail's, so a later adoption needs no code."""
        pads = {
            "r_sole_cop_fore_aft_gate": [0.3, 0.3, 0.3],
            "r_foot_cop_fore_aft_gate": [0.13, 0.9, 0.2],
            "l_foot_cop_fore_aft_gate": [0.05, 0.1, 0.85],
        }
        keys = ("r_sole_cop_fore_aft_gate", *FOOT_GATE_KEYS)
        scalars = self._scalars(tmp_path, pads=pads, keys=keys, curriculum_kwargs={"max_foot_cop_fore_aft": 0.8})
        assert scalars["max_sole_cop_fore_aft_mean"] == pytest.approx(0.3)
        assert scalars["max_foot_cop_fore_aft_over_bar_fraction"] == pytest.approx(2 / 3)
        assert "max_sole_cop_fore_aft_over_bar_fraction" not in scalars


# ── the sagittal leg DC, in degrees about the leftward axis ──────────────────

#: Two mirrored legs: the left hip pitch axis points right (-y) and the right one left (+y), as on the
#: compsognathus robot; the knees both point right; the hip roll is not sagittal (axis x); the ankles are driven
#: by motors, not position servos.  The feet sit at y = +-0.1, so the leftward axis is +y.
_MIRRORED_LEGS = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 1"><freejoint/>
      <geom size="0.1"/>
      <body name="left_thigh" pos="0 0.1 0">
        <joint name="left_hip_pitch" axis="0 -1 0"/><joint name="left_hip_roll" axis="1 0 0"/>
        <geom size="0.05"/>
        <body name="left_shin" pos="0 0 -0.4"><joint name="left_knee" axis="0 -1 0"/><geom size="0.05"/>
          <body name="left_foot" pos="0 0 -0.4"><joint name="left_ankle" axis="0 -1 0"/><geom size="0.05"/>
            <site name="left_foot_site"/></body></body></body>
      <body name="right_thigh" pos="0 -0.1 0">
        <joint name="right_hip_pitch" axis="0 1 0"/><joint name="right_hip_roll" axis="1 0 0"/>
        <geom size="0.05"/>
        <body name="right_shin" pos="0 0 -0.4"><joint name="right_knee" axis="0 -1 0"/><geom size="0.05"/>
          <body name="right_foot" pos="0 0 -0.4"><joint name="right_ankle" axis="0 -1 0"/><geom size="0.05"/>
            <site name="right_foot_site"/></body></body></body>
    </body>
  </worldbody>
  <actuator>
    <position name="a0" joint="left_hip_pitch" kp="10" ctrlrange="-1.2 0.5"/>
    <position name="a1" joint="right_hip_pitch" kp="10" ctrlrange="-0.5 1.2"/>
    <position name="a2" joint="left_knee" kp="10" ctrlrange="-1 1"/>
    <position name="a3" joint="right_knee" kp="10" ctrlrange="-1 1"/>
    <position name="a4" joint="left_hip_roll" kp="10" ctrlrange="-1 1"/>
    <position name="a5" joint="right_hip_roll" kp="10" ctrlrange="-1 1"/>
    <motor name="a6" joint="left_ankle" ctrlrange="-1 1"/>
    <motor name="a7" joint="right_ankle" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def _mirrored_legs():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_MIRRORED_LEGS)
    home = np.array([0.4, -0.4, -0.3, -0.3, 0.0, 0.0, 0.0, 0.0])
    low, high = model.actuator_ctrlrange.T

    def scale(action):
        residual = np.clip(action, -1.0, 1.0)
        return home + np.where(residual < 0, residual * (home - low), residual * (high - home))

    env = SimpleNamespace(model=model, _scale_action=scale)
    feet = [
        SimpleNamespace(label="r", primary_site=model.site("right_foot_site").id),
        SimpleNamespace(label="l", primary_site=model.site("left_foot_site").id),
    ]
    return env, SimpleNamespace(feet=feet), home, low, high


class TestSagittalLegDc:
    def test_pairs_are_the_lateral_position_servos_signed_along_the_leftward_axis(self):
        from environments.shared.eval_diagnostics import SagittalLegPair, sagittal_leg_dc_pairs

        env, morphology, *_ = _mirrored_legs()
        assert sagittal_leg_dc_pairs(env, morphology) == (
            SagittalLegPair("leg_dc_l_minus_r_hip_pitch_deg", 0, 1, -1.0, 1.0),
            SagittalLegPair("leg_dc_l_minus_r_knee_deg", 2, 3, -1.0, -1.0),
            # no hip roll (axis x, not sagittal) and no ankle (motors: a control value is not an angle)
        )

    # Within every joint's range on both sides of home: the right hip has only 0.1 rad below its home control.
    @pytest.mark.parametrize("theta", [0.03, -0.03, 0.09])
    def test_a_mirror_symmetric_offset_reads_zero_and_an_antisymmetric_one_twice_the_angle(self, theta):
        from environments.shared.eval_diagnostics import leg_dc_degrees, sagittal_leg_dc_pairs

        env, morphology, home, low, high = _mirrored_legs()
        pairs = sagittal_leg_dc_pairs(env, morphology)

        def action(left_world, right_world):
            # Joint angle = sign along +y times the rotation about +y; the residual map inverted per side of home.
            target = home.copy()
            for pair in pairs:
                target[pair.left] += pair.left_sign * left_world
                target[pair.right] += pair.right_sign * right_world
            delta = target - home
            return np.where(delta < 0, delta / (home - low), delta / np.where(high > home, high - home, 1.0))

        symmetric = leg_dc_degrees(env, pairs, action(theta, theta))
        assert symmetric == pytest.approx(dict.fromkeys(symmetric, 0.0), abs=1e-9)
        antisymmetric = leg_dc_degrees(env, pairs, action(theta, -theta))
        assert antisymmetric == pytest.approx(dict.fromkeys(antisymmetric, math.degrees(2 * theta)))
        # The mirror-even mean reads the other half: the symmetric offset whole, the antisymmetric one as 0.
        mean = eval_diagnostics_module.leg_dc_mean_degrees(env, pairs, action(theta, theta))
        assert set(mean) == {"leg_dc_l_r_mean_hip_pitch_deg", "leg_dc_l_r_mean_knee_deg"}
        assert mean == pytest.approx(dict.fromkeys(mean, math.degrees(theta)))
        opposite = eval_diagnostics_module.leg_dc_mean_degrees(env, pairs, action(theta, -theta))
        assert opposite == pytest.approx(dict.fromkeys(opposite, 0.0), abs=1e-9)
        # The bug this replaced: on the mirrored hips the raw action difference reads a symmetric offset as one.
        raw = action(theta, theta)
        assert abs(raw[0] - raw[1]) > 0.01

    @pytest.mark.parametrize("dc", [[], [0.1] * 3, [float("nan")] * 8])
    def test_an_unmeasured_dc_reads_nan(self, dc):
        from environments.shared.eval_diagnostics import leg_dc_degrees, sagittal_leg_dc_pairs

        env, morphology, *_ = _mirrored_legs()
        values = leg_dc_degrees(env, sagittal_leg_dc_pairs(env, morphology), dc)
        assert len(values) == 2 and all(math.isnan(value) for value in values.values())

    def test_the_capture_callback_publishes_the_degrees_of_the_dc_it_just_published(self):
        from environments.shared.eval_diagnostics import sagittal_leg_dc_pairs

        env, morphology, *_ = _mirrored_legs()
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback.leg_dc_pairs = sagittal_leg_dc_pairs(env, morphology)
        callback._foot_setup = [(env, env.model, morphology, None)]
        callback.evaluations_action_dc = [[0.0] * 8, []]
        callback._publish_leg_dc()  # the DC of the last evaluation is empty: unmeasured
        callback.evaluations_action_dc.append([0.0] * 8)
        callback._publish_leg_dc()
        assert all(math.isnan(value) for value in callback.evaluations_leg_dc[0].values())
        assert callback.evaluations_leg_dc[1] == {
            "leg_dc_l_minus_r_hip_pitch_deg": 0.0,
            "leg_dc_l_minus_r_knee_deg": 0.0,
            "leg_dc_l_r_mean_hip_pitch_deg": 0.0,
            "leg_dc_l_r_mean_knee_deg": 0.0,
        }
        # Without the extensions the record holds exactly the left-minus-right keys.
        with patch.object(StageAwareEvalCallback, "report_extensions", False):
            callback._publish_leg_dc()
        assert set(callback.evaluations_leg_dc[2]) == {"leg_dc_l_minus_r_hip_pitch_deg", "leg_dc_l_minus_r_knee_deg"}

    def test_seeded_prior_evaluations_carry_an_empty_leg_dc_record(self):
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback.seed_prior_history(2)
        assert callback.evaluations_leg_dc == [{}, {}]


#: Every manifest model: its pad feet's whole-foot fields (a declared support span) and digit fields -- the digits'
#: key names and whether each is hinged (FOOT_SUPPORT and the MTP rule).
REAL_FOOT_FIELDS = {
    ModelId("tyrannosaurus_rex", "anatomical"): (True, (("toe_d2", True), ("toe_d3", True), ("toe_d4", True))),
    ModelId("compsognathus_longipes", "anatomical"): (
        False,
        (("toe_d2", False), ("toe_d3", False), ("toe_d4", False)),
    ),
    ModelId("compsognathus_longipes", "robot"): (False, ()),
    ModelId("dibothrosuchus_elaphros", "anatomical"): (False, ()),
    ModelId("velociraptor_mongoliensis", "anatomical"): (False, ()),
    ModelId("brachiosaurus_altithorax", "anatomical"): (False, ()),
}
#: Every manifest model: its box-soled feet, whether they have digits, and its sagittal leg DC pairs (FOOT_SUPPORT
#: and the actuated joints).
REAL_TELEMETRY = {
    ModelId("tyrannosaurus_rex", "anatomical"): (("r", "l"), True, (("l", "r"),)),
    ModelId("compsognathus_longipes", "anatomical"): (("r", "l"), True, (("l", "r"),)),
    ModelId("compsognathus_longipes", "robot"): (("r", "l"), False, (("l", "r"),)),
    ModelId("dibothrosuchus_elaphros", "anatomical"): (("fr", "fl", "rr", "rl"), False, (("fl", "fr"), ("rl", "rr"))),
    ModelId("velociraptor_mongoliensis", "anatomical"): ((), False, (("l", "r"),)),
    ModelId("brachiosaurus_altithorax", "anatomical"): ((), False, (("fl", "fr"), ("rl", "rr"))),
}
#: Every manifest model: its box-soled feet (FOOT_SUPPORT), the roll rails' feet -- the pad feet.
REAL_RAIL_FEET = {model_id: feet for model_id, (feet, _digits, _pairs) in REAL_TELEMETRY.items()}


def _balance_keys(model_id) -> tuple[str, ...]:
    """The model's balance and force keys (gait.balance_telemetry): only where the recorder runs (box soles)."""
    pad_feet, _digits, pairs = REAL_TELEMETRY[model_id]
    if not pad_feet:
        return ()
    tyrannosaurus = model_id == TYRANNOSAURUS_REX
    keys = [
        "com_fore_aft_mean_m",
        "root_pitch_mean_deg",
        "xcom_fore_aft_max_m",
        "hof_cop_term_mean_mps2",
        "hof_counter_rotation_mean_mps2",
        "hof_counter_rotation_rms_mps2",
    ]
    for left, right in pairs:
        for joint in ("hip_pitch", "knee", "ankle"):
            keys += [f"leg_q_{left}_minus_{right}_{joint}_deg", f"leg_q_{left}_{right}_mean_{joint}_deg"]
    if tyrannosaurus:
        keys += [
            "xcom_toe_margin_min_m",
            "xcom_toe_margin_mean_m",
            "xcom_heel_margin_min_m",
            "xcom_past_toe_fraction",
            "time_to_toe_min_s",
            "xcom_toe_lead_s",
            "xcom_tip_margin_min_m",
        ]
    keys += ["actuator_force_frac_max", "actuator_force_limited_fraction", "shear_over_normal"]
    for foot in pad_feet:
        keys += [f"{foot}_shear_over_normal", f"{foot}_cone_use_mean", f"{foot}_cone_use_max"]
        keys += [f"{foot}_ankle_moment_arm_m"] if tyrannosaurus else []
    return tuple(keys)


def _rail_keys(model_id) -> tuple[str, ...]:
    return tuple(
        f"{foot}_{field}" for foot in REAL_RAIL_FEET[model_id] for field in ("sole_tilt_deg", "sole_corner_lift_m")
    )


def _real_evaluation(
    model_id, tmp_path, monkeypatch, *, action=None, episodes=2, extra_steps=40, seed=0, evaluations=1
):
    """*evaluations* short evaluations of *model_id*'s stance on its real eval env (VecNormalize over a DummyVecEnv).

    Seeded: an unseeded reset draws a new spawn on every run, and some spawns fall before the settle window ends
    (the T. rex statue on reset seed 4 at step 178; the held left ankle on seeds 4 and 9), which leaves no
    post-settle step to measure.  Every posture these tests hold stands to the horizon from seed 0.
    """
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mujoco")
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.config import load_stage_config
    from environments.shared.eval_diagnostics import build_stage_evaluation_callbacks
    from environments.shared.gait import pad_telemetry
    from environments.shared.species_registry import env_class

    config = load_stage_config(model_id, "stance")
    settle = int(config["curriculum_kwargs"].get("settle_steps", 0))
    config["env_kwargs"]["max_episode_steps"] = settle + extra_steps
    cls = env_class(model_id)
    eval_env = VecNormalize(DummyVecEnv([lambda: Monitor(cls(**config["env_kwargs"]))]), norm_reward=False)
    eval_env.training = False
    eval_env.seed(seed)
    traces = []
    reduce = pad_telemetry.episode_pad_telemetry

    def keep(trace, layout, *, settle_steps):
        traces.append(trace)
        return reduce(trace, layout, settle_steps=settle_steps)

    monkeypatch.setattr(pad_telemetry, "episode_pad_telemetry", keep)
    callback, plateau = build_stage_evaluation_callbacks(
        eval_env,
        stage="stance",
        stage_config=config,
        model_id=model_id,
        gate_progress_dir=tmp_path,
        log_path=str(tmp_path),
        eval_freq=1,
        n_eval_episodes=episodes,
        deterministic=True,
    )
    nu = eval_env.action_space.shape[0]
    held = np.zeros(nu, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32)
    model = MagicMock()
    model.get_env.return_value = eval_env
    model.get_vec_normalize_env.return_value = None
    model.num_timesteps = 1
    model.logger = MagicMock()
    model.predict.side_effect = lambda observations, **kwargs: (np.tile(held, (observations.shape[0], 1)), None)
    callback.init_callback(model)
    plateau.init_callback(model)
    try:
        for _ in range(evaluations):
            assert callback.on_step() is True
            plateau.on_step()
        hook_left = "_substep_probe_hook" in vars(eval_env.venv.envs[0].unwrapped)
        callback.on_training_end()
    finally:
        eval_env.close()
    return callback, np.load(tmp_path / "gate_progress.npz"), traces, settle, hook_left


@pytest.mark.parametrize("model_id", sorted(REAL_TELEMETRY), ids=str)
def test_a_real_evaluation_records_each_models_pads_and_leg_dc(model_id, tmp_path, monkeypatch):
    """Every model: the pad keys of its box-soled feet (none on capsules and ellipsoids), the panel's own CoP
    statistic beside them (and its bar fraction where the stage declares the bar), its leg DC pairs, all in the npz;
    the recorder is off the env after the evaluation; and each episode's per-foot gate statistic tops out at the
    gate's own ``max_sole_cop_fore_aft``."""
    from environments.shared.config import load_stage_config
    from environments.shared.gait.pad_telemetry import FOOT_COP_EPISODE_FIELDS, PAD_EPISODE_FIELDS
    from environments.shared.gait.stance_metrics import episode_stance_metrics

    assert set(REAL_FOOT_FIELDS) == set(REAL_TELEMETRY) == set(REAL_RAIL_FEET)
    pad_feet, has_digits, pairs = REAL_TELEMETRY[model_id]
    rail_keys = _rail_keys(model_id)
    whole_foot, digits = REAL_FOOT_FIELDS[model_id]
    callback, data, traces, settle, hook_left = _real_evaluation(model_id, tmp_path, monkeypatch)
    fields = PAD_EPISODE_FIELDS if has_digits else tuple(f for f in PAD_EPISODE_FIELDS if f != "digit_share")
    fields += FOOT_COP_EPISODE_FIELDS if whole_foot else ()
    fields += tuple(
        f"{name}_{field}"
        for name, hinged in digits
        for field in ("load_share", "load_pos")
        if hinged or field != "load_pos"
    )
    pad_keys = tuple(f"{foot}_{field}" for foot in pad_feet for field in fields)
    leg_keys = tuple(
        f"leg_dc_{left}_minus_{right}_{joint}_deg" for left, right in pairs for joint in ("hip_pitch", "knee", "ankle")
    )
    mean_keys = tuple(
        f"leg_dc_{left}_{right}_mean_{joint}_deg" for left, right in pairs for joint in ("hip_pitch", "knee", "ankle")
    )
    balance_keys = _balance_keys(model_id)
    assert callback.foot_pad_keys == pad_keys
    assert callback.foot_rail_keys == rail_keys
    assert callback.balance_keys == balance_keys
    assert tuple(pair.key for pair in callback.leg_dc_pairs) == leg_keys
    assert not hook_left
    feet = tuple(f"{foot}_" for foot in pad_feet)
    # The per-foot force keys start with a foot label too; none ends in a rail suffix (gait.balance_telemetry).
    per_foot_balance = {key for key in balance_keys if key.startswith(feet)}
    assert {key for key in data.files if key.startswith(feet)} == set(pad_keys) | set(rail_keys) | per_foot_balance
    assert {key for key in data.files if key.startswith("leg_dc_")} == set(leg_keys) | set(mean_keys)
    assert all(data[key][0] == 0.0 for key in leg_keys + mean_keys)  # the zero action holds no offset
    (episodes,) = callback.evaluations_foot_pad
    assert set(episodes) == set(pad_keys) | set(rail_keys) | set(balance_keys)
    assert all(len(values) == 2 for values in episodes.values())
    assert len(traces) == (2 if pad_feet else 0)  # a pad-less model attaches no recorder at all
    gates = [episode_stance_metrics(trace, settle_steps=settle).max_sole_cop_fore_aft for trace in traces]
    for number, gate in enumerate(gates):
        assert max(episodes[f"{foot}_sole_cop_fore_aft_gate"][number] for foot in pad_feet) == gate
    # A standing statue measures every field its model has.
    assert all(math.isfinite(data[key][0]) for key in pad_keys + rail_keys)
    bar = load_stage_config(model_id, "stance")["curriculum_kwargs"].get("max_sole_cop_fore_aft")
    panel_keys = {key for key in data.files if key.startswith("max_sole_cop_fore_aft_")}
    # The whole foot's own statistic, on a model that declares a span: report-only, so never a bar fraction.
    assert "max_foot_cop_fore_aft" not in load_stage_config(model_id, "stance")["curriculum_kwargs"]
    whole_keys = {key for key in data.files if key.startswith("max_foot_cop_fore_aft_")}
    if whole_foot:
        metrics = [episode_stance_metrics(trace, settle_steps=settle).max_foot_cop_fore_aft for trace in traces]
        for number, metric in enumerate(metrics):
            assert max(episodes[f"{foot}_foot_cop_fore_aft_gate"][number] for foot in pad_feet) == metric
        assert whole_keys == {"max_foot_cop_fore_aft_mean"}
        assert data["max_foot_cop_fore_aft_mean"][0] == pytest.approx(np.mean(metrics))
    else:
        assert whole_keys == set()
    if not pad_feet:
        assert panel_keys == set()
        return
    assert data["max_sole_cop_fore_aft_mean"][0] == pytest.approx(np.mean(gates))
    if bar is None:
        assert panel_keys == {"max_sole_cop_fore_aft_mean"}
    else:
        assert data["max_sole_cop_fore_aft_over_bar_fraction"][0] == np.mean([gate > bar for gate in gates])


def test_a_held_left_ankle_offset_reads_as_a_negative_left_minus_right_ankle_dc(tmp_path, monkeypatch):
    """Run A's sign (left ankle -0.063 action units below the right): the T. rex, through the real env, in degrees."""
    callback, data, _traces, _settle, _hook = _real_evaluation(
        TYRANNOSAURUS_REX, tmp_path, monkeypatch, action=np.eye(15)[10] * -0.063, episodes=1
    )
    pair = {pair.key: pair for pair in callback.leg_dc_pairs}["leg_dc_l_minus_r_ankle_deg"]
    assert (pair.left, pair.right, pair.left_sign, pair.right_sign) == (10, 6, 1.0, 1.0)
    # The residual map below home: -0.063 of the span from the home control down to the range's minimum, in radians.
    from environments.shared.config import load_stage_config
    from environments.shared.species_registry import env_class

    env = env_class(TYRANNOSAURUS_REX)(**load_stage_config(TYRANNOSAURUS_REX, "stance")["env_kwargs"])
    try:
        span = float(env.unwrapped._home_ctrl[10] - env.unwrapped.model.actuator_ctrlrange[10, 0])
    finally:
        env.close()
    # The held action is float32, as a policy's is: 1e-5 degrees covers its rounding.
    assert data["leg_dc_l_minus_r_ankle_deg"][0] == pytest.approx(math.degrees(-0.063 * span), abs=1e-5)
    assert data["leg_dc_l_minus_r_ankle_deg"][0] < -3.0  # about run A's -3.2 degrees
    assert data["leg_dc_l_minus_r_knee_deg"][0] == 0.0


@pytest.mark.parametrize(("right_world", "expected"), [(0.03, 0.0), (-0.03, math.degrees(0.06))])
def test_the_robots_mirrored_hips_read_a_held_symmetric_offset_as_zero(tmp_path, monkeypatch, right_world, expected):
    """The compsognathus robot's hip pitch axes are mirrored (left -y, right +y) and its control ranges with them.

    Held through a real evaluation, a mirror-symmetric 0.03 rad hip offset reads 0 and an antisymmetric one twice
    the angle; the raw action difference read the symmetric one as -0.036 or +0.61 action units, by direction.
    """
    model_id = ModelId("compsognathus_longipes", "robot")
    from environments.shared.config import load_stage_config
    from environments.shared.species_registry import env_class

    env = env_class(model_id)(**load_stage_config(model_id, "stance")["env_kwargs"])
    try:
        unwrapped = env.unwrapped
        home = np.asarray(unwrapped._home_ctrl, dtype=float)
        low, high = unwrapped.model.actuator_ctrlrange.T
        names = [unwrapped.model.actuator(index).name for index in range(unwrapped.model.nu)]
    finally:
        env.close()
    left, right = names.index("left_hip_pitch_act"), names.index("right_hip_pitch_act")
    action = np.zeros(len(names))
    # Rotations of +0.03 rad (left) and right_world (right) about the leftward axis: left joint angle -0.03 (its axis
    # points right), right joint angle +right_world; each mapped back through the residual map's side of home.
    for index, angle in ((left, -0.03), (right, right_world)):
        action[index] = angle / (high[index] - home[index]) if angle > 0 else angle / (home[index] - low[index])
    assert abs(action[left] - action[right]) > 0.03  # what the action-unit difference used to report
    callback, data, _traces, _settle, _hook = _real_evaluation(
        model_id, tmp_path, monkeypatch, action=action, episodes=1
    )
    assert data["leg_dc_l_minus_r_hip_pitch_deg"][0] == pytest.approx(expected, abs=1e-5)  # a float32 action
    assert data["leg_dc_l_minus_r_knee_deg"][0] == 0.0 and data["leg_dc_l_minus_r_ankle_deg"][0] == 0.0


# ── the roll rails and the termination causes ────────────────────────────────

RAIL_KEYS = ("r_sole_tilt_deg", "r_sole_corner_lift_m", "l_sole_tilt_deg", "l_sole_corner_lift_m")


class TestTheRollRails:
    """``max_sole_tilt_deg_*`` and ``max_sole_corner_lift_m_*``: each episode's worse sole, as the gate reads it."""

    @staticmethod
    def _plateau(tmp_path, *, lengths, pads, rail_keys=RAIL_KEYS, curriculum_kwargs=None):
        callback = TestPadTelemetryReachesGateProgress._plateau(
            tmp_path, lengths=lengths, pads=pads, keys=(), pairs=(), curriculum_kwargs=curriculum_kwargs
        )
        callback.eval_callback.foot_rail_keys = rail_keys
        return callback

    def test_per_foot_means_and_the_mean_of_each_episodes_worse_sole_with_a_fraction_per_declared_bar(self, tmp_path):
        nan = float("nan")
        pads = {
            # the worse sole changes between episodes; the fourth episode falls and is left out
            "r_sole_tilt_deg": [2.5, 0.5, 1.0, 9.0],
            "l_sole_tilt_deg": [0.5, 2.2, 1.0, 9.0],
            "r_sole_corner_lift_m": [0.001, 0.005, nan, 0.02],
            "l_sole_corner_lift_m": [0.002, 0.001, 0.003, 0.02],
        }
        callback = self._plateau(
            tmp_path,
            lengths=[1000, 1000, 1000, 400],
            pads=pads,
            curriculum_kwargs={"max_sole_tilt_deg": 2.0, "max_sole_corner_lift_m": 0.004},
        )
        scalars = callback._foot_telemetry_scalars(0)
        assert scalars["r_sole_tilt_deg"] == pytest.approx(4.0 / 3)
        assert scalars["r_sole_corner_lift_m"] == pytest.approx(0.003)  # the unmeasured episode drops
        assert scalars["max_sole_tilt_deg_mean"] == pytest.approx((2.5 + 2.2 + 1.0) / 3)
        assert scalars["max_sole_tilt_deg_over_bar_fraction"] == pytest.approx(2 / 3)
        # An episode measured on one foot reads that foot, as the gate's maximum over the finite feet does.
        assert scalars["max_sole_corner_lift_m_mean"] == pytest.approx((0.002 + 0.005 + 0.003) / 3)
        assert scalars["max_sole_corner_lift_m_over_bar_fraction"] == pytest.approx(1 / 3)

    def test_no_fraction_without_a_declared_bar_and_no_key_without_a_box_sole(self, tmp_path):
        callback = self._plateau(
            tmp_path, lengths=[1000], pads={"fr_sole_tilt_deg": [1.7], "fl_sole_tilt_deg": [1.65]}, rail_keys=()
        )
        assert callback._foot_telemetry_scalars(0) == {}
        callback.eval_callback.foot_rail_keys = ("fr_sole_tilt_deg", "fl_sole_tilt_deg")
        scalars = callback._foot_telemetry_scalars(0)
        assert scalars == pytest.approx(
            {"fr_sole_tilt_deg": 1.7, "fl_sole_tilt_deg": 1.65, "max_sole_tilt_deg_mean": 1.7}
        )

    def test_the_rails_reach_the_npz_and_tensorboard(self, tmp_path):
        from environments.shared.curriculum.stance_gate import stance_panel_from_episode_duties

        callback = self._plateau(
            tmp_path,
            lengths=[1000],
            pads={key: [1.0] for key in RAIL_KEYS},
            curriculum_kwargs={"max_sole_tilt_deg": 2.0},
        )
        panel = stance_panel_from_episode_duties(
            episode_lengths=[1000.0], episode_duties=[0.0], episode_rewards=[1.0], horizon=1000
        )
        callback._record_stance_scalars(0, panel)
        data = np.load(tmp_path / "gate_progress.npz")
        assert {key for key in data.files if "sole_tilt" in key or "corner_lift" in key} == {
            *RAIL_KEYS,
            "max_sole_tilt_deg_mean",
            "max_sole_tilt_deg_over_bar_fraction",
            "max_sole_corner_lift_m_mean",
        }
        assert data["max_sole_tilt_deg_over_bar_fraction"].tolist() == [0.0]
        recorded = {call.args[0] for call in callback.logger.record.call_args_list}
        assert {"diagnostics/eval_max_sole_tilt_deg_mean", "diagnostics/eval_r_sole_corner_lift_m"} <= recorded


class TestTerminationCauses:
    """Each evaluation episode's cause, captured from its last info and counted per evaluation."""

    @pytest.mark.parametrize(
        ("info", "cause"),
        [
            ({"TimeLimit.truncated": True}, "truncated"),
            ({"TimeLimit.truncated": False, "termination_reason": "nosedive"}, "nosedive"),
            ({"termination_reason": "head_contact"}, "head_contact"),
            ({"TimeLimit.truncated": False}, "terminated"),
            ({"TimeLimit.truncated": False, "termination_reason": ""}, "terminated"),
        ],
    )
    def test_the_cause_of_an_episode(self, info, cause):
        assert StageAwareEvalCallback.termination_cause(info) == cause

    def test_each_finished_episode_appends_its_cause_and_the_evaluation_publishes_them(self):
        pytest.importorskip("stable_baselines3")
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback.eval_freq = 10
        callback.n_calls = 10
        callback.verbose = 0

        def evaluate():
            for index, (done, info) in enumerate(
                [
                    (False, {}),
                    (True, {"TimeLimit.truncated": False, "termination_reason": "nosedive"}),
                    (True, {"TimeLimit.truncated": True}),
                ]
            ):
                callback._log_success_callback({"i": 0, "done": done, "info": info, "actions": np.zeros((1, 2))}, {})
            return True

        with patch.object(eval_diagnostics_module._EvalCallback, "_on_step", side_effect=evaluate):
            assert callback._on_step() is True
        assert callback.evaluations_termination_causes == [["nosedive", "truncated"]]

    def test_seeded_prior_evaluations_carry_no_cause(self):
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback.seed_prior_history(2)
        assert callback.evaluations_termination_causes == [[], []]

    @staticmethod
    def _record(callback, eval_callback, causes, *, timesteps):
        from environments.shared.curriculum.stance_gate import stance_panel_from_episode_duties

        index = len(eval_callback.evaluations_length)
        eval_callback.evaluations_length.append([1000.0] * len(causes))
        eval_callback.evaluations_termination_causes.append(causes)
        callback.num_timesteps = timesteps
        panel = stance_panel_from_episode_duties(
            episode_lengths=[1000.0] * len(causes),
            episode_duties=[0.0] * len(causes),
            episode_rewards=[1.0] * len(causes),
            horizon=1000,
        )
        callback._record_stance_scalars(index, panel)

    def test_counts_per_evaluation_with_a_later_cause_reading_zero_on_the_earlier_rows(self, tmp_path):
        callback, eval_callback = _plateau_callback({"min_avg_reward": 0.0}, stage=1)
        callback._gate_progress_dir = tmp_path
        eval_callback.evaluations_termination_causes = []
        self._record(callback, eval_callback, ["truncated"] * 3, timesteps=100)
        self._record(callback, eval_callback, ["truncated", "nosedive", "nosedive"], timesteps=200)
        self._record(callback, eval_callback, ["head_contact", "truncated", "truncated"], timesteps=300)
        data = np.load(tmp_path / "gate_progress.npz")
        assert data["termination_truncated"].tolist() == [3.0, 1.0, 2.0]
        assert data["termination_nosedive"].tolist() == [0.0, 2.0, 0.0]
        assert data["termination_head_contact"].tolist() == [0.0, 0.0, 1.0]
        recorded = [call.args for call in callback.logger.record.call_args_list if call.args[0].startswith("termin")]
        assert ("terminations/eval_head_contact", 1.0) in recorded and ("terminations/eval_nosedive", 0.0) in recorded

    def test_rows_seeded_from_a_file_without_causes_read_nan(self, tmp_path):
        callback, eval_callback = _plateau_callback({"min_avg_reward": 0.0}, stage=1)
        callback._gate_progress_dir = tmp_path
        np.savez(tmp_path / "prior.npz", timesteps=np.array([50, 100]), mean_reward=np.array([1.0, 2.0]))
        assert callback.seed_prior_gate_progress(tmp_path / "prior.npz") == 2
        eval_callback.evaluations_length.extend([[], []])
        eval_callback.evaluations_termination_causes = [[], []]
        self._record(callback, eval_callback, ["nosedive", "truncated"], timesteps=150)
        data = np.load(tmp_path / "gate_progress.npz")
        assert data["timesteps"].tolist() == [50, 100, 150]
        for key, last in (("termination_nosedive", 1.0), ("termination_truncated", 1.0)):
            values = data[key].tolist()
            assert math.isnan(values[0]) and math.isnan(values[1]) and values[2] == last

    def test_an_evaluation_without_captured_causes_reads_nan_for_every_known_cause(self, tmp_path):
        callback, eval_callback = _plateau_callback({"min_avg_reward": 0.0}, stage=1)
        callback._gate_progress_dir = tmp_path
        eval_callback.evaluations_termination_causes = []
        self._record(callback, eval_callback, ["truncated"], timesteps=100)
        self._record(callback, eval_callback, [], timesteps=200)
        data = np.load(tmp_path / "gate_progress.npz")
        assert data["termination_truncated"][0] == 1.0 and math.isnan(data["termination_truncated"][1])


def test_the_plateau_warning_is_recorded_for_the_progress_report(caplog):
    callback, eval_callback = _plateau_callback({"min_avg_episode_length": 1000.0}, stage=1, plateau_window=3)
    with caplog.at_level(logging.WARNING, logger="environments.shared.eval_diagnostics"):
        for _ in range(3):
            _add_evaluation(callback, eval_callback, reward=10.0, length=400.0)
            callback._on_step()
    (warning,) = callback.progress_warnings
    assert warning["source"] == "evaluation_plateau" and warning["timesteps"] == callback.num_timesteps
    assert warning["message"] in caplog.messages


@pytest.mark.parametrize("model_id", sorted(REAL_RAIL_FEET), ids=str)
def test_a_real_evaluation_records_each_models_roll_rails_and_termination_causes(model_id, tmp_path, monkeypatch):
    """Every model: each episode's worse sole is the gate's own ``max_sole_tilt_deg`` / ``max_sole_corner_lift_m``,
    the npz holds their means (and the over-bar fractions where the stage declares the bars), and the panel's
    episodes, all cut by the time limit here, count as truncated in the npz and TensorBoard."""
    from environments.shared.config import load_stage_config
    from environments.shared.gait.stance_metrics import episode_stance_metrics

    callback, data, traces, settle, _hook = _real_evaluation(model_id, tmp_path, monkeypatch)
    assert callback.evaluations_termination_causes == [["truncated", "truncated"]]
    assert data["termination_truncated"].tolist() == [2.0]
    assert [key for key in data.files if key.startswith("termination_")] == ["termination_truncated"]
    recorded = [call.args for call in callback.logger.record.call_args_list]
    assert ("terminations/eval_truncated", 2.0) in recorded
    rail_keys = _rail_keys(model_id)
    curriculum = load_stage_config(model_id, "stance")["curriculum_kwargs"]
    rail_panel = {key for key in data.files if key.startswith(("max_sole_tilt_deg_", "max_sole_corner_lift_m_"))}
    if not rail_keys:
        assert rail_panel == set() and traces == []
        return
    (episodes,) = callback.evaluations_foot_pad
    for field, metric in (("sole_tilt_deg", "max_sole_tilt_deg"), ("sole_corner_lift_m", "max_sole_corner_lift_m")):
        feet = [key for key in rail_keys if key.endswith(f"_{field}")]
        if not feet:
            assert not any(key.startswith(f"{metric}_") for key in rail_panel)
            continue
        gates = [getattr(episode_stance_metrics(trace, settle_steps=settle), metric) for trace in traces]
        for number, gate in enumerate(gates):
            assert max(episodes[key][number] for key in feet) == gate
        assert data[f"{metric}_mean"][0] == pytest.approx(np.mean(gates))
        bar = curriculum.get(metric)
        assert (f"{metric}_over_bar_fraction" in data.files) == (bar is not None)
        if bar is not None:
            assert data[f"{metric}_over_bar_fraction"][0] == np.mean([gate > bar for gate in gates])


# ── the report-only extensions: the v2 screen, the eval-episode ledger, balance and forces ──────────

#: A synthetic v2 block for the synthetic statue of test_gait_pad_telemetry (400 steps, settle 100), with
#: generous bars but the pad's fore-aft one, and the hop-or-fall rail.
SCREEN_CURRICULUM = {
    "gate_kind": "stance_quality/v2",
    "min_eval_episodes": 4,
    "min_clean_stance_lcb": 0.05,
    "settle_steps": 100,
    "min_all_feet_support": 0.5,
    "max_touchdown_rate": 10.0,
    "max_window_displacement_m": 1.0,
    "min_foot_load_share": 0.1,
    "max_actuator_saturation_fraction": 0.5,
    "max_settle_airborne_substeps": 100.0,
    "max_settle_peak_floor_force_bw": 10.0,
    "max_sole_cop_fore_aft": 0.8,
    "max_hop_or_fall_episodes": 1,
    "min_avg_reward_statue_ratio": 0.5,
}


class _TraceRecorder(_StubRecorder):
    """A stub recorder whose episodes are the given traces, in turn."""

    def __init__(self, traces) -> None:
        super().__init__()
        self.traces = list(traces)
        self.flags: list[dict] = []

    def end_episode(self, **kwargs):
        self.flags.append(kwargs)
        self.episodes += 1
        return self.traces.pop(0)


def _screened_evaluation(traces, *, ledger_dir=None, timesteps=1000, causes=None, posture=None):
    """A bare capture callback run through one evaluation of *traces* (synthetic pad statues), as SB3 drives it.

    *posture*: a ``PostureLayout`` the callback resolved for the sub-env (its columns join the ledger, last).
    """
    pytest.importorskip("stable_baselines3")
    from environments.shared.curriculum.stance_gate_v2 import StanceV2Thresholds
    from environments.shared.episode_ledger import EpisodeLedger
    from environments.shared.gait.pad_telemetry import RailFoot, RailLayout
    from environments.shared.tests.test_gait_pad_telemetry import LAYOUT

    callback = _bare_stage_aware_callback(success_applicable=False, settle_steps=100)
    callback.eval_freq = 10
    callback.n_calls = 10
    callback.verbose = 0
    callback.n_eval_episodes = len(traces)
    callback.horizon = 400
    callback.control_dt = 0.01
    callback.screen_thresholds = StanceV2Thresholds.from_curriculum(SCREEN_CURRICULUM)
    callback.evaluations_results, callback.evaluations_length, callback.evaluations_timesteps = [], [], []
    callback.evaluations_screen_v2 = []
    callback._ledger = EpisodeLedger(
        None if ledger_dir is None else ledger_dir / "eval_episodes.csv",
        fixed_columns=eval_diagnostics_module.EVAL_LEDGER_FIXED_COLUMNS,
    )
    callback._foot_resolved = True
    callback.foot_pad_keys = LAYOUT.episode_keys()
    rails = RailLayout(feet=(RailFoot(0, "r"), RailFoot(1, "l")))
    callback.foot_rail_keys = rails.episode_keys()
    if posture is not None:
        callback.posture_columns = posture.columns()
        callback._posture_layouts = [posture]
    recorder = _TraceRecorder(traces)
    causes = causes or ["truncated" if trace.length >= 400 else "nosedive" for trace in traces]

    def evaluate():
        callback._foot_setup = [(None, None, None, LAYOUT)]
        callback._rail_layouts = [rails]
        callback._pad_recorders = {0: recorder}
        callback._pad_quota = {0: len(traces)}
        for trace, cause in zip(traces, causes):
            for step in range(trace.length):
                done = step == trace.length - 1
                info = {"reward_alive": 1.0, "reward_posture": 0.5 if trace.length >= 400 else 0.25}
                if done:
                    info.update(
                        {"TimeLimit.truncated": True}
                        if cause == "truncated"
                        else {"TimeLimit.truncated": False, "termination_reason": cause}
                    )
                callback._log_success_callback(
                    {"i": 0, "done": done, "info": info, "actions": np.zeros((1, 4)), "reward": 1.0}, {}
                )
        callback.evaluations_timesteps.append(timesteps)
        callback.evaluations_results.append([float(trace.length) for trace in traces])
        callback.evaluations_length.append([trace.length for trace in traces])
        return True

    with patch.object(eval_diagnostics_module._EvalCallback, "_on_step", side_effect=evaluate):
        assert callback._on_step() is True
    return callback, recorder


def _screen_traces():
    """Four synthetic statues: two clean, one on its right pad's toe edge, one that falls at step 300."""
    from environments.shared.tests.test_gait_pad_telemetry import _trace, _with_cop

    toe = _with_cop(_trace(), 0, 0.95)
    fell = replace(_trace(300), truncated=False, terminated=True)
    return [_trace(), toe, _trace(), fell]


class TestTheInTrainingScreen:
    """The stage's own v2 decision at yaw 0 on each evaluation's episodes, without the statue criteria."""

    def test_the_screen_is_the_gates_in_training_decision_on_the_same_episodes(self):
        from environments.shared.curriculum.stance_gate_v2 import evaluate_stance_v2_gate
        from environments.shared.gait.stance_metrics import episode_stance_metrics

        traces = _screen_traces()
        callback, recorder = _screened_evaluation(traces)
        expected = evaluate_stance_v2_gate(
            [episode_stance_metrics(trace, settle_steps=100) for trace in traces],
            callback.screen_thresholds,
            horizon=400,
            control_dt=0.01,
            in_training=True,
        )
        (entry,) = callback.evaluations_screen_v2
        assert entry == {**expected.as_dict(), "episode_reasons": [list(r) for r in expected.episode_reasons]}
        assert (entry["n_clean"], entry["n_episodes"], entry["hop_or_fall_episodes"]) == (2, 4, 1)
        assert entry["criterion_failures"]["max_sole_cop_fore_aft"] == 1 and entry["criterion_failures"]["horizon"] == 1
        assert entry["skipped_criteria"] == ["min_avg_reward_statue_ratio"]  # never a statue-relative judgement
        # The recorder was told how each episode ended.
        assert recorder.flags[1] == {"terminated": False, "truncated": True, "termination_reason": ""}
        assert recorder.flags[3] == {"terminated": True, "truncated": False, "termination_reason": "nosedive"}
        assert callback.__dict__["_current_eval_screen"] == []  # the per-episode metrics are dropped

    def test_the_plateau_records_a_fixed_key_set_in_its_own_tensorboard_section(self, tmp_path):
        from environments.shared.curriculum.stance_gate import stance_panel_from_episode_duties

        callback, _recorder = _screened_evaluation(_screen_traces())
        plateau, _ = _plateau_callback({"min_avg_reward": 0.0}, stage=1)
        plateau.horizon = 400
        plateau.eval_callback = callback
        plateau._gate_progress_dir = tmp_path
        scalars = plateau._screen_scalars(0)
        assert set(scalars) == set(eval_diagnostics_module.screen_scalar_keys(callback.screen_thresholds))
        assert scalars["screen_v2_clean_count"] == 2.0 and scalars["screen_v2_episodes"] == 4.0
        # 2 of 4 clean bounds at 0.098, over this synthetic bar of 0.05; one fall is within the hop-or-fall rail.
        assert scalars["screen_v2_yaw0_bars_met"] == 1.0 and scalars["screen_v2_hop_or_fall_episodes"] == 1.0
        assert scalars["screen_v2_clean_lcb"] == pytest.approx(callback.evaluations_screen_v2[0]["clean_lcb"])
        assert scalars["screen_v2_fail_max_sole_cop_fore_aft_count"] == 1.0
        assert scalars["screen_v2_fail_horizon_count"] == 1.0
        assert "screen_v2_fail_min_foot_load_share_statue_ratio_count" not in scalars  # not declared
        panel = stance_panel_from_episode_duties(
            episode_lengths=[400.0, 400.0, 400.0, 300.0],
            episode_duties=[0.0, 0.0, 0.0, 0.0],
            episode_rewards=[1.0, 1.0, 1.0, 1.0],
            horizon=400,
        )
        callback.evaluations_unsupported_duties = [[0.0] * 4]
        plateau._record_stance_scalars(0, panel)
        records = [call for call in plateau.logger.record.call_args_list]
        screen = [call for call in records if call.args[0].startswith("screen_v2/")]
        assert {call.args[0] for call in screen} == {f"screen_v2/{key[len('screen_v2_') :]}" for key in scalars}
        assert all(call.kwargs.get("exclude") == ("stdout", "log") for call in screen)
        assert not any("screen" in call.args[0] for call in records if call.args[0].startswith("diagnostics/"))
        data = np.load(tmp_path / "gate_progress.npz")
        assert {key for key in data.files if key.startswith("screen_v2_")} == set(scalars)

    def test_an_unjudged_evaluation_reads_nan_for_every_screen_key(self):
        callback, _recorder = _screened_evaluation(_screen_traces())
        callback.evaluations_screen_v2.append(None)
        plateau, _ = _plateau_callback({}, stage=1)
        plateau.eval_callback = callback
        scalars = plateau._screen_scalars(1)
        assert scalars and all(math.isnan(value) for value in scalars.values())

    def test_a_skipped_statue_criterion_reads_nan_not_the_gates_zero(self):
        """The anatomical Compsognathus declares ``min_foot_load_share_statue_ratio``, which the screen never judges.

        The gate's tally starts every declared criterion at 0, the skipped
        ones too; the screen's count for it must read NaN ("never judged"),
        not 0 ("no episode failed it"), in ``gate_progress.npz`` and so in
        TensorBoard, which records finite values only.
        """
        from environments.shared.config import load_stage_config
        from environments.shared.curriculum.stance_gate_v2 import evaluate_stance_v2_gate
        from environments.shared.gait.stance_metrics import episode_stance_metrics
        from environments.shared.tests.test_gait_pad_telemetry import LAYOUT, _trace

        curriculum = load_stage_config(ModelId("compsognathus_longipes", "anatomical"), "stance")["curriculum_kwargs"]
        thresholds = eval_diagnostics_module._screen_thresholds(curriculum)
        assert thresholds is not None and thresholds.min_foot_load_share_statue_ratio is not None
        skipped_key = "screen_v2_fail_min_foot_load_share_statue_ratio_count"
        # The key set stays the declaration's.
        assert skipped_key in eval_diagnostics_module.screen_scalar_keys(thresholds)
        result = evaluate_stance_v2_gate(
            [episode_stance_metrics(_trace(), settle_steps=thresholds.settle_steps) for _ in range(3)],
            thresholds,
            horizon=400,
            in_training=True,
        )
        assert "min_foot_load_share_statue_ratio" in result.skipped_criteria
        assert dict(result.criterion_failures)["min_foot_load_share_statue_ratio"] == 0  # the gate's tally: a 0
        plateau, eval_callback = _plateau_callback({}, stage=1)
        eval_callback.screen_thresholds = thresholds
        eval_callback.foot_pad_keys = LAYOUT.episode_keys()
        eval_callback.evaluations_screen_v2 = [result.as_dict()]
        scalars = plateau._screen_scalars(0)
        assert set(scalars) == set(eval_diagnostics_module.screen_scalar_keys(thresholds))
        assert math.isnan(scalars[skipped_key])
        # Every criterion the screen does judge keeps its count, the horizon's among them.
        judged = {f"screen_v2_fail_{key}_count": float(count) for key, count in result.criterion_failures}
        del judged[skipped_key]
        assert judged and all(scalars[key] == value for key, value in judged.items())
        assert scalars["screen_v2_fail_horizon_count"] == 0.0

    @pytest.mark.parametrize(
        ("model_id", "screened"),
        [
            (TYRANNOSAURUS_REX, True),  # stance_quality_heading/v1
            (ModelId("compsognathus_longipes", "anatomical"), True),  # stance_quality_heading/v1
            (ModelId("compsognathus_longipes", "robot"), False),  # stance_quality/v1
            (ModelId("dibothrosuchus_elaphros", "anatomical"), False),  # reward_and_length/v1
            (BRACHIOSAURUS_ALTITHORAX, False),  # reward_and_length/v1
        ],
        ids=str,
    )
    def test_the_screen_runs_only_on_a_stage_that_declares_v2s_keys(self, model_id, screened):
        from environments.shared.config import load_stage_config

        curriculum = load_stage_config(model_id, "stance")["curriculum_kwargs"]
        thresholds = eval_diagnostics_module._screen_thresholds(curriculum)
        assert (thresholds is not None) == screened
        if screened:
            keys = eval_diagnostics_module.screen_scalar_keys(thresholds)
            assert keys[:4] == (
                "screen_v2_episodes",
                "screen_v2_clean_count",
                "screen_v2_clean_lcb",
                "screen_v2_yaw0_bars_met",
            )

    def test_no_screen_key_reads_as_a_foot_a_reward_term_a_cause_or_a_recorded_gate_count(self):
        from environments.shared.curriculum.stance_gate_v2 import (
            CRITERION_BAR_RANGES,
            EPISODE_CRITERIA,
            StanceV2Thresholds,
        )

        fractions = {key for key, _low, _high, _why in CRITERION_BAR_RANGES}
        everything = {
            **SCREEN_CURRICULUM,
            **{key: 0.5 if key in fractions else 1.5 for key, _metric, _direction in EPISODE_CRITERIA},
        }
        keys = eval_diagnostics_module.screen_scalar_keys(StanceV2Thresholds.from_curriculum(everything))
        for key in keys:
            assert not key.endswith(RAIL_SUFFIXES) and not key.startswith(("term_", "termination_")), key
            assert key not in {"stance_clean_count", "n_stance_samples", "success_count", "n_success_samples"}


#: The per-foot rail suffixes progress_report._feet_of and PostureDriftWatchCallback._feet read as a foot.
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


class TestTheEvalEpisodeLedger:
    """One row per evaluation episode, aligned with SB3's own records, rewritten after every evaluation."""

    def test_one_row_per_episode_aligned_with_the_evaluation_records(self, tmp_path):
        import csv

        traces = _screen_traces()
        callback, _recorder = _screened_evaluation(traces, ledger_dir=tmp_path)
        with open(tmp_path / "eval_episodes.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        header = list(rows[0])
        assert tuple(header[:8]) == eval_diagnostics_module.EVAL_LEDGER_FIXED_COLUMNS
        assert [float(row["return"]) for row in rows] == callback.evaluations_results[0]
        assert [int(row["length"]) for row in rows] == callback.evaluations_length[0]
        assert [row["cause"] for row in rows] == callback.evaluations_termination_causes[0]
        assert [row["full_horizon"] for row in rows] == ["1", "1", "1", "0"]
        assert [row["screen_v2_clean"] for row in rows] == ["1", "0", "1", "0"]
        assert rows[1]["screen_v2_reasons"] == "max_sole_cop_fore_aft" and rows[0]["screen_v2_reasons"] == ""
        assert rows[3]["screen_v2_reasons"].split(";")[0] == "horizon"
        # The ledger-only columns, then the pad, roll-rail and maxima columns of each foot.
        assert header[8:18] == [
            "screen_v2_clean",
            "screen_v2_reasons",
            "saturated_actuator",
            "saturated_actuator_fraction",
            "saturated_actuator_positive_share",
            "saturated_actuator_ties",
            "actuators_saturated",
            "spawn_stance_width_change_m",
            "r_spawn_tilt_deg",
            "l_spawn_tilt_deg",
        ]
        # The statues command nothing: no actuator is named, and the fraction is the metric's 0.
        assert [row["saturated_actuator"] for row in rows] == [""] * 4
        assert [row["saturated_actuator_fraction"] for row in rows] == ["0.0"] * 4
        assert rows[0]["saturated_actuator_positive_share"] == "nan" and rows[0]["actuators_saturated"] == "0"
        assert float(rows[1]["max_sole_cop_fore_aft"]) == pytest.approx(0.95 * 300 / 300)
        assert rows[0]["spawn_stance_width_change_m"] == "nan"  # no balance layout: unmeasured, written as nan
        assert rows[0]["r_spawn_tilt_deg"] == "0.0"
        # The ledger holds columns, not one dict per row.
        ledger = callback._ledger
        assert len(ledger) == 4 and ledger.column("episode") == [0, 1, 2, 3]
        assert all(type(values) is list and len(values) == 4 for values in ledger._columns.values())

    def test_each_episode_names_the_actuator_behind_its_saturation_metric(self, tmp_path):
        import csv

        from environments.shared.gait.stance_metrics import episode_stance_metrics

        traces = _screen_traces()
        held = traces[0].action.copy()
        held[100:, 3] = 1.0  # the last actuator held at +1 over the whole window: over the screen's 0.5 bar
        held[100:200, 1] = -1.0  # a third of the window at -1: saturating, not the metric's actuator
        traces[0] = replace(traces[0], action=held)
        tied = traces[2].action.copy()
        tied[100:250, 2] = -1.0
        tied[250:, 0] = -1.0
        tied[250:, 2] = 1.0  # actuator 2 on the whole window, both signs; actuator 0 on half of it
        traces[2] = replace(traces[2], action=tied)
        unmeasured = traces[1].action.copy()
        unmeasured[300] = np.nan  # a window step with no action
        traces[1] = replace(traces[1], action=unmeasured)
        _screened_evaluation(traces, ledger_dir=tmp_path)
        with open(tmp_path / "eval_episodes.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        gates = [episode_stance_metrics(trace, settle_steps=100).max_actuator_saturation_fraction for trace in traces]
        # The synthetic sub-env has no model, so the actuator is named by its index.
        assert [row["saturated_actuator"] for row in rows] == ["actuator[3]", "", "actuator[2]", ""]
        for row, gate in zip(rows, gates):
            assert row["saturated_actuator_fraction"] == ("nan" if math.isnan(gate) else repr(gate))
        assert [row["saturated_actuator_positive_share"] for row in rows] == ["1.0", "nan", "0.5", "nan"]
        assert [row["saturated_actuator_ties"] for row in rows] == ["0", "nan", "0", "0"]
        assert [row["actuators_saturated"] for row in rows] == ["2", "nan", "2", "0"]
        assert "max_actuator_saturation_fraction" in rows[0]["screen_v2_reasons"].split(";")
        assert "max_actuator_saturation_fraction" in rows[1]["screen_v2_reasons"].split(";")  # NaN fails the bar

    def test_the_saturating_actuator_columns_read_as_no_rail_foot_gate_key_or_number(self):
        from environments.shared.curriculum.progress_report import _feet_of
        from environments.shared.curriculum.stance_gate_v2 import EPISODE_CRITERIA, STANCE_V2_PANEL_FIELDNAMES
        from environments.shared.gait.stance_metrics import SATURATED_ACTUATOR_COLUMNS
        from environments.shared.reporting import training_history
        from environments.shared.visualization import _gate_progress_groups

        columns = list(SATURATED_ACTUATOR_COLUMNS)
        gate_keys = {key for criterion in EPISODE_CRITERIA for key in criterion[:2]} | set(STANCE_V2_PANEL_FIELDNAMES)
        for column in columns:
            assert not column.endswith(RAIL_SUFFIXES) and column not in gate_keys, column
            assert not column.startswith(("max_", "min_", "term_", "termination_", "screen_v2_", "fall_")), column
            assert not column.startswith(("leg_dc_", "leg_q_", "reward_", "eval_", "plant_", "std_")), column
        row = {column: 0.0 for column in columns}
        assert not any(_feet_of(row, suffix.lstrip("_")) for suffix in RAIL_SUFFIXES)
        assert not any(_gate_progress_groups(columns).values())
        assert training_history.foot_labels(columns) == [] and training_history.lr_columns(columns) == {}

    def test_the_posture_columns_come_last_and_move_no_other_cell(self, tmp_path):
        import csv

        from environments.shared.gait.posture_telemetry import episode_posture_telemetry
        from environments.shared.tests.test_gait_posture_telemetry import _layout, _posture_trace

        posed = [replace(trace, joint_qpos=_posture_trace(trace.length)[0].joint_qpos) for trace in _screen_traces()]
        posed = [
            replace(trace, site_pos=_posture_trace(trace.length)[0].site_pos, site_names=("head_tip",))
            for trace in posed
        ]
        plain_dir, posture_dir = tmp_path / "plain", tmp_path / "posture"
        plain_dir.mkdir()
        posture_dir.mkdir()
        _screened_evaluation(list(posed), ledger_dir=plain_dir)
        _screened_evaluation(list(posed), ledger_dir=posture_dir, posture=_layout())
        tables = []
        for directory in (plain_dir, posture_dir):
            with open(directory / "eval_episodes.csv", newline="") as handle:
                reader = csv.DictReader(handle)
                tables.append((list(reader.fieldnames or []), list(reader)))
        (plain_header, plain_rows), (header, rows) = tables
        columns = list(_layout().columns())
        # Appended: every earlier column keeps its place and every cell its value.
        assert header == plain_header + columns
        assert [{key: row[key] for key in plain_header} for row in rows] == plain_rows
        for row, trace in zip(rows, posed):
            expected = episode_posture_telemetry(trace, _layout(), settle_steps=100)
            assert [row[key] for key in columns] == ["nan" if math.isnan(v) else repr(v) for v in expected.values()]
        assert float(rows[0]["neck_pitch_mean_deg"]) == pytest.approx(math.degrees(0.3))
        assert rows[3]["head_tip_height_mean_m"] != "nan"  # the fall at step 300 still has its window 100..299

    def test_a_posture_reduction_that_fails_costs_only_the_posture_columns(self, tmp_path, monkeypatch, caplog):
        import csv

        from environments.shared.gait import posture_telemetry
        from environments.shared.tests.test_gait_posture_telemetry import _layout

        def raising(trace, layout, *, settle_steps):
            raise FloatingPointError("a posture reduction went wrong")

        monkeypatch.setattr(posture_telemetry, "episode_posture_telemetry", raising)
        with caplog.at_level(logging.WARNING, logger=eval_diagnostics_module.__name__):
            _screened_evaluation(_screen_traces(), ledger_dir=tmp_path, posture=_layout())
        with open(tmp_path / "eval_episodes.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert all(row[key] == "nan" for row in rows for key in _layout().columns())
        assert [row["screen_v2_clean"] for row in rows] == ["1", "0", "1", "0"]  # the screen is untouched
        assert sum("Posture telemetry" in record.getMessage() for record in caplog.records) == 1

    def test_the_posture_columns_read_as_no_rail_foot_gate_key_or_left_right_offset(self):
        from environments.shared.curriculum.progress_report import _feet_of
        from environments.shared.curriculum.stance_gate_v2 import EPISODE_CRITERIA, STANCE_V2_PANEL_FIELDNAMES
        from environments.shared.gait.balance_telemetry import BALANCE_SCALAR_KEYS
        from environments.shared.gait.posture_telemetry import HEAD_POSTURE_COLUMNS, HIP_ROLL_FIELDS
        from environments.shared.gait.stance_metrics import StanceEpisodeMetrics
        from environments.shared.visualization import _gate_progress_groups

        columns = [
            *HEAD_POSTURE_COLUMNS,
            *(f"{foot}_{field}" for foot in ("r", "l", "fr", "rl") for field in HIP_ROLL_FIELDS),
        ]
        gate_keys = {key for criterion in EPISODE_CRITERIA for key in criterion[:2]} | set(STANCE_V2_PANEL_FIELDNAMES)
        metric_fields = {field.name for field in fields(StanceEpisodeMetrics)}
        for column in columns:
            assert not column.endswith(RAIL_SUFFIXES) and column not in gate_keys | metric_fields, column
            assert column not in BALANCE_SCALAR_KEYS, column
            assert not column.startswith(("max_", "min_", "term_", "termination_", "screen_v2_", "fall_")), column
            assert not column.startswith(("leg_dc_", "leg_q_", "reward_", "eval_", "plant_", "std_")), column
        row = {column: 0.0 for column in columns}
        assert not any(_feet_of(row, suffix.lstrip("_")) for suffix in RAIL_SUFFIXES)
        assert not any(_gate_progress_groups(columns).values())

    def test_every_evaluation_rewrites_the_whole_file(self, tmp_path):
        import csv

        callback, _recorder = _screened_evaluation(_screen_traces()[:2], ledger_dir=tmp_path)
        recorder = _TraceRecorder(_screen_traces()[:2])

        def evaluate():
            callback._pad_recorders = {0: recorder}
            callback._pad_quota = {0: 2}
            for trace in recorder.traces[:]:
                for step in range(trace.length):
                    done = step == trace.length - 1
                    callback._log_success_callback(
                        {"i": 0, "done": done, "info": {"TimeLimit.truncated": done}, "actions": np.zeros((1, 4))}, {}
                    )
            callback.evaluations_timesteps.append(2000)
            callback.evaluations_results.append([1.0, 2.0])
            callback.evaluations_length.append([400, 400])
            return True

        with patch.object(eval_diagnostics_module._EvalCallback, "_on_step", side_effect=evaluate):
            callback._on_step()
        with open(tmp_path / "eval_episodes.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [(row["evaluation"], row["timesteps"], row["episode"]) for row in rows] == [
            ("0", "1000", "0"),
            ("0", "1000", "1"),
            ("1", "2000", "0"),
            ("1", "2000", "1"),
        ]

    def test_a_failed_write_leaves_the_previous_file_whole_and_warns_once(self, tmp_path, monkeypatch, caplog):
        import os

        _screened_evaluation(_screen_traces()[:1], ledger_dir=tmp_path)
        before = (tmp_path / "eval_episodes.csv").read_bytes()
        real_replace = os.replace

        def refuse(source, destination):
            if str(destination).endswith("eval_episodes.csv"):
                raise OSError("the Drive mount went away")
            real_replace(source, destination)

        monkeypatch.setattr(os, "replace", refuse)
        callback, _recorder = _screened_evaluation(_screen_traces(), ledger_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="environments.shared.episode_ledger"):
            for _ in range(2):
                callback._ledger.extend([{"evaluation": 9}])
                assert callback._ledger.write() is False
        assert (tmp_path / "eval_episodes.csv").read_bytes() == before
        assert sum("eval_episodes.csv" in record.getMessage() for record in caplog.records) == 1

    def test_a_resume_keeps_the_prior_rows_at_or_before_the_resumed_step(self, tmp_path):
        _screened_evaluation(_screen_traces()[:2], ledger_dir=tmp_path, timesteps=1000)
        prior = (tmp_path / "eval_episodes.csv").read_text()
        from environments.shared.episode_ledger import EpisodeLedger

        callback = _bare_stage_aware_callback(success_applicable=False)
        assert callback.seed_prior_record(tmp_path, max_timesteps=None) == 0  # no ledger: nothing to seed
        callback._ledger = EpisodeLedger(None, fixed_columns=eval_diagnostics_module.EVAL_LEDGER_FIXED_COLUMNS)
        assert callback.seed_prior_record(tmp_path, max_timesteps=999) == 0
        callback._ledger = EpisodeLedger(None, fixed_columns=eval_diagnostics_module.EVAL_LEDGER_FIXED_COLUMNS)
        assert callback.seed_prior_record(tmp_path, max_timesteps=1000) == 2
        assert callback._ledger.column("timesteps") == ["1000", "1000"]
        assert (tmp_path / "eval_episodes.csv").read_text() == prior  # seeding writes nothing

    def test_without_the_extensions_no_ledger_is_written_and_the_records_are_mains(self, tmp_path, monkeypatch):
        pytest.importorskip("stable_baselines3")
        monkeypatch.setattr(StageAwareEvalCallback, "report_extensions", False)
        callback, _data, _traces, _settle, _hook = _real_evaluation(TYRANNOSAURUS_REX, tmp_path, monkeypatch)
        assert callback._ledger is None and not (tmp_path / "eval_episodes.csv").exists()
        assert callback.balance_keys == ()
        assert set(callback.evaluations_foot_pad[0]) == set(callback.foot_pad_keys + callback.foot_rail_keys)
        assert set(callback.evaluations_leg_dc[0]) == {pair.key for pair in callback.leg_dc_pairs}
        assert callback.__dict__.get("evaluations_screen_v2") == []
        assert callback.__dict__.get("evaluations_actuator_force_frac") == []
        assert callback.__dict__.get("evaluations_reward_terms_split") == []


class TestTheExtensionsOnARealEvaluation:
    """The T. rex: the ledger, the balance keys, the force matrix and the screen through its real eval env."""

    def test_the_ledger_and_the_balance_fields_and_no_ledger_only_column_in_the_npz(self, tmp_path, monkeypatch):
        import csv

        callback, data, traces, settle, _hook = _real_evaluation(TYRANNOSAURUS_REX, tmp_path, monkeypatch)
        with open(tmp_path / "eval_episodes.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2 and set(rows[0]) >= {"screen_v2_clean", "spawn_stance_width_change_m", "l_spawn_tilt_deg"}
        assert set(callback.balance_keys) <= set(rows[0])
        # The ledger-only columns never reach gate_progress.npz.
        assert not any("spawn" in key for key in data.files)
        assert "screen_v2_clean" not in data.files and "screen_v2_reasons" not in data.files
        assert not any("saturated_actuator" in key or key == "actuators_saturated" for key in data.files)
        assert all(math.isfinite(data[key][0]) for key in callback.balance_keys if key != "xcom_toe_lead_s")
        assert "xcom_toe_lead_s" not in data.files and math.isnan(data["fall_xcom_toe_lead_s"][0])  # no fall
        assert data["actuator_force_frac_per_actuator"].shape == (1, 15)
        assert data["screen_v2_episodes"].tolist() == [2.0]
        for key in data.files:  # no new key reads as a foot to the posture readers, as a term or as a cause
            if key in callback.balance_keys or key.startswith(("screen_v2_", "fall_", "full_horizon_term_")):
                assert not key.endswith(RAIL_SUFFIXES) and not key.startswith(("term_", "termination_")), key

    def test_the_ledger_names_the_held_tail_and_its_fraction_is_the_screens_metric(self, tmp_path, monkeypatch):
        import csv

        import mujoco

        from environments.shared.gait.stance_metrics import episode_stance_metrics
        from environments.shared.paths import ENVIRONMENTS_ROOT

        model = mujoco.MjModel.from_xml_path(str(TYRANNOSAURUS_REX.asset_path(ENVIRONMENTS_ROOT)))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        action = np.zeros(len(names))
        action[names.index("tail_3_pitch_act")] = 1.0  # the tail held up, as the four failed runs held it
        callback, data, traces, settle, _hook = _real_evaluation(
            TYRANNOSAURUS_REX, tmp_path, monkeypatch, action=action
        )
        with open(tmp_path / "eval_episodes.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(traces) == 2
        for row, trace in zip(rows, traces):
            gate = episode_stance_metrics(trace, settle_steps=settle).max_actuator_saturation_fraction
            assert row["saturated_actuator"] == "tail_3_pitch_act"
            assert row["saturated_actuator_fraction"] == repr(gate) == "1.0"
            assert row["saturated_actuator_positive_share"] == "1.0" and row["actuators_saturated"] == "1"
            assert row["saturated_actuator_ties"] == "0"
            assert "max_actuator_saturation_fraction" in row["screen_v2_reasons"].split(";")
        assert not any("saturated_actuator" in key for key in data.files)

    @pytest.mark.parametrize("failure", ["hook", "reducer", "constructor"])
    def test_a_failing_balance_capture_costs_only_the_balance_keys(self, tmp_path, monkeypatch, failure):
        from environments.shared.gait.recorder import RecorderError, SubstepContactRecorder

        plain_dir, failing_dir = tmp_path / "plain", tmp_path / "failing"
        with monkeypatch.context() as patched:
            patched.setattr(StageAwareEvalCallback, "_resolve_balance", lambda self, setup, pairs: None)
            plain, plain_data, _traces, _settle, _hook = _real_evaluation(
                TYRANNOSAURUS_REX, plain_dir, patched, evaluations=2
            )
        assert plain.balance_keys == ()
        if failure == "hook":
            capture = SubstepContactRecorder._balance_capture

            def raising(self, env, data):
                if int(env._step_count) == 10:
                    raise FloatingPointError("a balance read went wrong")
                capture(self, env, data)

            monkeypatch.setattr(SubstepContactRecorder, "_balance_capture", raising)
        elif failure == "reducer":

            def raising_reduction(self, **_kwargs):
                raise FloatingPointError("a balance reduction went wrong")

            monkeypatch.setattr(SubstepContactRecorder, "_reduce_balance", raising_reduction)
        else:
            build = SubstepContactRecorder.__init__

            def refusing(self, env, morphology, *, per_geom=False, balance=False):
                if balance:
                    raise RecorderError("no balance capture here")
                build(self, env, morphology, per_geom=per_geom, balance=balance)

            monkeypatch.setattr(SubstepContactRecorder, "__init__", refusing)
        failing, data, _traces, _settle, _hook = _real_evaluation(
            TYRANNOSAURUS_REX, failing_dir, monkeypatch, evaluations=2
        )
        # Two evaluations: training went on past the failure.
        assert len(failing.evaluations_foot_pad) == 2 and failing.balance_keys
        evaluations = (np.load(plain_dir / "evaluations.npz"), np.load(failing_dir / "evaluations.npz"))
        for key in ("timesteps", "results", "ep_lengths"):
            np.testing.assert_array_equal(evaluations[0][key], evaluations[1][key])
        for before, after in zip(plain.evaluations_foot_pad, failing.evaluations_foot_pad):
            for key, values in before.items():
                np.testing.assert_array_equal(values, after[key], err_msg=key)
                assert all(math.isfinite(value) for value in after[key] if key.endswith(RAIL_SUFFIXES[:2])), key
            assert all(math.isnan(value) for key in failing.balance_keys for value in after[key])
        for key in failing.balance_keys:
            if key != "xcom_toe_lead_s":
                assert np.isnan(data[key]).all(), key
        for key in plain_data.files:
            if key in data.files:
                np.testing.assert_array_equal(plain_data[key], data[key], err_msg=key)
        # The force matrix stays in the file, NaN where unmeasured (a failed reduction measures no fraction).
        assert data["actuator_force_frac_per_actuator"].shape == (2, 15)

    def test_a_failed_capture_appends_a_nan_force_row_and_keeps_the_matrix(self, tmp_path, monkeypatch):
        from environments.shared.gait.recorder import SubstepContactRecorder

        def refuse(self, action=None, reward=float("nan")):
            raise RuntimeError("the substep hook fired 0 times")

        monkeypatch.setattr(SubstepContactRecorder, "end_step", refuse)
        callback, data, _traces, _settle, _hook = _real_evaluation(TYRANNOSAURUS_REX, tmp_path, monkeypatch)
        (row,) = callback.evaluations_actuator_force_frac
        assert len(row) == 15 and all(math.isnan(value) for value in row)
        assert data["actuator_force_frac_per_actuator"].shape == (1, 15)
        assert np.isnan(data["actuator_force_frac_per_actuator"]).all()

    def test_the_ledger_ends_with_each_episodes_head_posture_and_hip_roll(self, tmp_path, monkeypatch):
        import csv

        from environments.shared.gait.posture_telemetry import episode_posture_telemetry

        callback, data, traces, settle, _hook = _real_evaluation(TYRANNOSAURUS_REX, tmp_path, monkeypatch)
        with open(tmp_path / "eval_episodes.csv", newline="") as handle:
            reader = csv.DictReader(handle)
            header, rows = list(reader.fieldnames or []), list(reader)
        columns = list(callback.posture_columns)
        assert columns == [
            "head_tip_height_mean_m",
            "head_tip_height_min_m",
            "neck_pitch_mean_deg",
            "head_pitch_mean_deg",
            "r_hip_roll_mean_deg",
            "r_hip_roll_max_abs_deg",
            "l_hip_roll_mean_deg",
            "l_hip_roll_max_abs_deg",
        ]
        assert header[-len(columns) :] == columns and header[-len(columns) - 1] == "l_ankle_moment_arm_m"
        (layout,) = callback._posture_layouts
        assert len(rows) == len(traces) == 2
        for row, trace in zip(rows, traces):
            assert trace.site_names == ("head_tip",)  # the recorder captured the head tip for these columns
            expected = episode_posture_telemetry(trace, layout, settle_steps=settle)
            assert [row[key] for key in columns] == [repr(value) for value in expected.values()]
            assert 0.3 < float(row["head_tip_height_min_m"]) <= float(row["head_tip_height_mean_m"]) < 2.0
        # Ledger-only: no posture column reaches gate_progress.npz.
        assert not set(columns) & set(data.files)
        assert not any("head_tip" in key or "hip_roll" in key or "neck_pitch" in key for key in data.files)

    def test_a_model_without_neck_hinges_writes_nan_there_and_measures_the_rest(self, tmp_path, monkeypatch):
        import csv

        robot = ModelId("compsognathus_longipes", "robot")
        callback, _data, _traces, _settle, _hook = _real_evaluation(robot, tmp_path, monkeypatch)
        with open(tmp_path / "eval_episodes.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        (layout,) = callback._posture_layouts
        assert set(layout.reasons) == {"neck_pitch_mean_deg", "head_pitch_mean_deg"}
        for row in rows:
            assert row["neck_pitch_mean_deg"] == row["head_pitch_mean_deg"] == "nan"
            for key in ("head_tip_height_mean_m", "r_hip_roll_mean_deg", "l_hip_roll_max_abs_deg"):
                assert math.isfinite(float(row[key])), key

    @pytest.mark.parametrize("failure", ["layout", "sites"])
    def test_a_posture_capture_that_fails_costs_only_its_columns(self, tmp_path, monkeypatch, failure):
        import csv

        from environments.shared.gait import posture_telemetry
        from environments.shared.gait.recorder import RecorderError, SubstepContactRecorder

        plain_dir, failing_dir = tmp_path / "plain", tmp_path / "failing"
        _real_evaluation(TYRANNOSAURUS_REX, plain_dir, monkeypatch)
        if failure == "layout":

            def refusing(cls, env, morphology):
                raise ValueError("no posture here")

            monkeypatch.setattr(posture_telemetry.PostureLayout, "from_env", classmethod(refusing))
        else:

            def refusing_sites(self, sites):
                raise RecorderError("no sites here")

            monkeypatch.setattr(SubstepContactRecorder, "record_sites", refusing_sites)
        callback, _data, _traces, _settle, _hook = _real_evaluation(TYRANNOSAURUS_REX, failing_dir, monkeypatch)
        tables = []
        for directory in (plain_dir, failing_dir):
            with open(directory / "eval_episodes.csv", newline="") as handle:
                reader = csv.DictReader(handle)
                tables.append((list(reader.fieldnames or []), list(reader)))
        (plain_header, plain_rows), (header, rows) = tables
        posture = [column for column in plain_header if column not in header or column in callback.posture_columns]
        kept = [column for column in plain_header if column not in posture]
        assert [{key: row[key] for key in kept} for row in rows] == [
            {key: row[key] for key in kept} for row in plain_rows
        ]
        if failure == "layout":  # the run cannot describe the model: no posture column, as with no balance layout
            assert callback.posture_columns == () and header == kept
        else:  # the head tip was not captured: its columns read NaN, the joints are still measured
            assert header == plain_header
            for row, plain in zip(rows, plain_rows):
                assert row["head_tip_height_mean_m"] == row["head_tip_height_min_m"] == "nan"
                assert row["neck_pitch_mean_deg"] == plain["neck_pitch_mean_deg"] != "nan"

    def test_a_padless_model_has_no_force_matrix_and_no_balance_key(self, tmp_path, monkeypatch):
        callback, data, _traces, _settle, _hook = _real_evaluation(VELOCIRAPTOR_MONGOLIENSIS, tmp_path, monkeypatch)
        assert callback.balance_keys == () and callback.evaluations_actuator_force_frac == [[]]
        assert "actuator_force_frac_per_actuator" not in data.files
        assert not any(key.startswith("screen_v2_") for key in data.files)  # no recorder, no screen
        # No recorder, so no trace to read a posture from: the ledger holds the outcome columns only.
        assert callback.posture_columns and not (tmp_path / "eval_episodes.csv").read_text().splitlines()[0].count(
            "hip"
        )


class TestTheBalanceKeysReachGateProgress:
    """The plateau's side: full-horizon means, the fall-only lead time and the mirror-even leg DC."""

    def test_the_full_horizon_means_and_the_fall_mean_of_the_lead_time(self, tmp_path):
        nan = float("nan")
        callback = TestPadTelemetryReachesGateProgress._plateau(
            tmp_path,
            lengths=[1000, 1000, 400, 600],
            pads={
                "com_fore_aft_mean_m": [0.01, 0.03, 0.5, 0.4],  # the falls are left out of the mean
                "xcom_toe_lead_s": [nan, nan, 1.5, 2.5],  # defined on the falls only
                "r_cone_use_max": [0.05, nan, 0.3, 0.3],
            },
            keys=(),
            pairs=(),
        )
        callback.eval_callback.balance_keys = ("com_fore_aft_mean_m", "xcom_toe_lead_s", "r_cone_use_max")
        scalars = callback._foot_telemetry_scalars(0)
        assert scalars == pytest.approx(
            {"com_fore_aft_mean_m": 0.02, "fall_xcom_toe_lead_s": 2.0, "r_cone_use_max": 0.05}
        )
        assert "xcom_toe_lead_s" not in scalars
        callback.eval_callback.evaluations_length = [[1000, 1000]]
        callback.eval_callback.evaluations_foot_pad = [{"xcom_toe_lead_s": [nan, nan]}]
        assert math.isnan(callback._foot_telemetry_scalars(0)["fall_xcom_toe_lead_s"])  # no fall: NaN, still supplied

    def test_the_matrix_row_is_the_mean_over_the_episodes_with_a_window(self):
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback.foot_pad_keys = PAD_KEYS
        callback.balance_keys = ("com_fore_aft_mean_m",)
        callback._foot_setup = [(None, SimpleNamespace(nu=3), None, None)]
        callback._current_eval_actuator_frac = [np.array([0.1, 0.2, np.nan]), np.array([0.3, 0.4, np.nan])]
        callback._publish_actuator_force_frac()
        callback._current_eval_actuator_frac = []  # no episode had a window: a NaN row, never []
        callback._publish_actuator_force_frac()
        first, second = callback.evaluations_actuator_force_frac
        assert first[:2] == pytest.approx([0.2, 0.3]) and math.isnan(first[2])
        assert len(second) == 3 and all(math.isnan(value) for value in second)
        callback.foot_pad_keys = ()  # a model whose recorder does not run
        callback._publish_actuator_force_frac()
        assert callback.evaluations_actuator_force_frac[-1] == []

    def test_the_extensions_tensorboard_tags_skip_the_console(self, tmp_path):
        from environments.shared.curriculum.stance_gate import stance_panel_from_episode_duties

        callback = TestPadTelemetryReachesGateProgress._plateau(
            tmp_path,
            lengths=[1000],
            pads={"r_sole_cop_fore_aft": [0.4], "com_fore_aft_mean_m": [0.02]},
            leg_dc={"leg_dc_l_minus_r_ankle_deg": 1.0, "leg_dc_l_r_mean_ankle_deg": 2.0},
        )
        callback.eval_callback.balance_keys = ("com_fore_aft_mean_m",)
        panel = stance_panel_from_episode_duties(
            episode_lengths=[1000.0], episode_duties=[0.0], episode_rewards=[1.0], horizon=1000
        )
        callback._record_stance_scalars(0, panel)
        calls = {call.args[0]: call.kwargs for call in callback.logger.record.call_args_list}
        assert calls["diagnostics/eval_com_fore_aft_mean_m"] == {"exclude": ("stdout", "log")}
        assert calls["diagnostics/eval_leg_dc_l_r_mean_ankle_deg"] == {"exclude": ("stdout", "log")}
        assert calls["diagnostics/eval_r_sole_cop_fore_aft"] == {}  # main's tags print as before
        assert calls["diagnostics/eval_leg_dc_l_minus_r_ankle_deg"] == {}


class TestTheRewardTermSplit:
    """Each reward term's per-episode mean over the full-horizon episodes and over the falls."""

    @staticmethod
    def _callback(episodes):
        callback = _bare_stage_aware_callback(success_applicable=False)
        callback.horizon = 4
        callback._reset_forward_velocity_capture()
        for length, alive in episodes:
            for step in range(length):
                callback._log_success_callback(
                    {"i": 0, "done": step == length - 1, "info": {"reward_alive": alive, "reward_total": 2 * alive}}, {}
                )
        callback._publish_action_statistics()
        callback._publish_reward_terms_split()
        return callback

    def test_the_two_kinds_add_up_to_the_evaluations_mean(self):
        callback = self._callback([(4, 1.0), (2, 0.5), (4, 3.0), (3, 0.1)])
        (terms,) = callback.evaluations_reward_terms
        (split,) = callback.evaluations_reward_terms_split
        assert split["full_horizon"]["reward_alive"] == pytest.approx((4 * 1.0 + 4 * 3.0) / 2)
        assert split["fall"]["reward_alive"] == pytest.approx((2 * 0.5 + 3 * 0.1) / 2)
        for key in terms:
            assert 2 * split["full_horizon"][key] + 2 * split["fall"][key] == pytest.approx(4 * terms[key], rel=1e-9)
        plateau, _ = _plateau_callback({}, stage=1)
        plateau.eval_callback = callback
        scalars = plateau._reward_term_split_scalars(0)
        assert set(scalars) == {f"{kind}_term_{key}" for kind in ("full_horizon", "fall") for key in ("alive", "total")}

    def test_a_kind_without_episodes_reads_nan(self):
        callback = self._callback([(4, 1.0), (4, 1.0)])
        (split,) = callback.evaluations_reward_terms_split
        assert split["full_horizon"]["reward_alive"] == 4.0 and math.isnan(split["fall"]["reward_alive"])


def test_seeded_prior_evaluations_carry_the_extensions_placeholders():
    callback = _bare_stage_aware_callback(success_applicable=False)
    callback.seed_prior_history(2)
    assert callback.evaluations_screen_v2 == [None, None]
    assert callback.evaluations_actuator_force_frac == [[], []]
    assert callback.evaluations_reward_terms_split == [{}, {}]


def test_the_recorded_history_reads_no_stance_v2_gate_from_the_npz_keys(tmp_path, monkeypatch):
    """The screen's keys are not the recorded-history gate's: a history built from every npz key reads None."""
    from environments.shared.config import load_stage_config
    from environments.shared.reporting.gates import _recorded_stance_v2_gate

    _callback, data, _traces, _settle, _hook = _real_evaluation(TYRANNOSAURUS_REX, tmp_path, monkeypatch)
    rows = [{key: float(data[key][0]) for key in data.files if data[key].ndim == 1}]
    curriculum = {
        **load_stage_config(TYRANNOSAURUS_REX, "stance")["curriculum_kwargs"],
        "gate_kind": "stance_quality/v2",
    }
    assert _recorded_stance_v2_gate(curriculum, rows) is None
    assert not set(data.files) & {"stance_clean_count", "n_stance_samples", "success_count", "n_success_samples"}
