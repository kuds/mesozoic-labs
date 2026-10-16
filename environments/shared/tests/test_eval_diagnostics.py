"""Tests for stage-aware SB3 evaluation diagnostics."""

import logging
import math
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
        assert set(scalars) == {"r_sole_cop_fore_aft", "leg_dc_l_minus_r_ankle_deg"}  # no gate key, no panel key
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
        assert set(callback._foot_telemetry_scalars(0)) == {"leg_dc_l_minus_r_ankle_deg"}


FOOT_GATE_KEYS = ("r_foot_cop_fore_aft_gate", "l_foot_cop_fore_aft_gate")


class TestThePanelsOwnWholeFootStatistic:
    """``max_foot_cop_fore_aft_mean``: the whole-foot rail's per-episode number, with its bar fraction where a stage
    declares the bar (the T. rex stance since physics r10, D-D34)."""

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
        """The arm is the pad rail's, so the T. rex's adoption of ``max_foot_cop_fore_aft`` (D-D34) needed no code."""
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
        }

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


def _real_evaluation(model_id, tmp_path, monkeypatch, *, action=None, episodes=2, extra_steps=40, seed=0):
    """One short evaluation of *model_id*'s stance on its real eval env (VecNormalize over a DummyVecEnv).

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
        assert callback.on_step() is True
        plateau.on_step()
        hook_left = "_substep_probe_hook" in vars(eval_env.venv.envs[0].unwrapped)
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

    assert set(REAL_FOOT_FIELDS) == set(REAL_TELEMETRY)
    pad_feet, has_digits, pairs = REAL_TELEMETRY[model_id]
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
    assert callback.foot_pad_keys == pad_keys
    assert tuple(pair.key for pair in callback.leg_dc_pairs) == leg_keys
    assert not hook_left
    assert {key for key in data.files if key.startswith(tuple(f"{foot}_" for foot in pad_feet))} == set(pad_keys)
    assert {key for key in data.files if key.startswith("leg_dc_")} == set(leg_keys)
    assert all(data[key][0] == 0.0 for key in leg_keys)  # the zero action holds no offset
    (episodes,) = callback.evaluations_foot_pad
    assert set(episodes) == set(pad_keys) and all(len(values) == 2 for values in episodes.values())
    assert len(traces) == (2 if pad_feet else 0)  # a pad-less model attaches no recorder at all
    gates = [episode_stance_metrics(trace, settle_steps=settle).max_sole_cop_fore_aft for trace in traces]
    for number, gate in enumerate(gates):
        assert max(episodes[f"{foot}_sole_cop_fore_aft_gate"][number] for foot in pad_feet) == gate
    # A standing statue measures every field its model has.
    assert all(math.isfinite(data[key][0]) for key in pad_keys)
    bar = load_stage_config(model_id, "stance")["curriculum_kwargs"].get("max_sole_cop_fore_aft")
    panel_keys = {key for key in data.files if key.startswith("max_sole_cop_fore_aft_")}
    # The whole foot's own statistic, on a model that declares a span, and its bar fraction where the stance
    # declares the bar (the T. rex since physics r10, D-D34), in the npz and in TensorBoard alike.
    whole_bar = load_stage_config(model_id, "stance")["curriculum_kwargs"].get("max_foot_cop_fore_aft")
    whole_keys = {key for key in data.files if key.startswith("max_foot_cop_fore_aft_")}
    recorded = {call.args[0]: call.args[1] for call in callback.model.logger.record.call_args_list}
    if whole_foot:
        metrics = [episode_stance_metrics(trace, settle_steps=settle).max_foot_cop_fore_aft for trace in traces]
        for number, metric in enumerate(metrics):
            assert max(episodes[f"{foot}_foot_cop_fore_aft_gate"][number] for foot in pad_feet) == metric
        assert data["max_foot_cop_fore_aft_mean"][0] == pytest.approx(np.mean(metrics))
        assert recorded["diagnostics/eval_max_foot_cop_fore_aft_mean"] == data["max_foot_cop_fore_aft_mean"][0]
        if whole_bar is None:
            assert whole_keys == {"max_foot_cop_fore_aft_mean"}
        else:
            fraction = np.mean([metric > whole_bar for metric in metrics])
            assert whole_keys == {"max_foot_cop_fore_aft_mean", "max_foot_cop_fore_aft_over_bar_fraction"}
            assert data["max_foot_cop_fore_aft_over_bar_fraction"][0] == fraction
            assert recorded["diagnostics/eval_max_foot_cop_fore_aft_over_bar_fraction"] == fraction
    else:
        assert whole_bar is None and whole_keys == set()
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
