"""Tests for W&B integration module."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environments.shared import wandb_integration


class TestIsAvailable:
    def test_returns_bool(self):
        result = wandb_integration.is_available()
        assert isinstance(result, bool)

    @patch.object(wandb_integration, "wandb", None)
    def test_returns_false_when_wandb_not_installed(self):
        assert not wandb_integration.is_available()

    @patch.object(wandb_integration, "wandb", MagicMock())
    def test_returns_true_when_wandb_installed(self):
        assert wandb_integration.is_available()


class TestGetGitHash:
    def test_returns_string(self):
        result = wandb_integration._get_git_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_returns_unknown_on_exception(self, mock_run):
        assert wandb_integration._get_git_hash() == "unknown"

    @patch("subprocess.run")
    def test_returns_unknown_on_nonzero_returncode(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert wandb_integration._get_git_hash() == "unknown"


class TestInitWandb:
    @patch.object(wandb_integration, "wandb", None)
    def test_returns_none_when_not_installed(self):
        result = wandb_integration.init_wandb("velociraptor", 1, {})
        assert result is None

    @patch.object(wandb_integration, "create_wandb_dashboard")
    @patch.object(wandb_integration, "setup_wandb_metrics")
    @patch.object(wandb_integration, "_get_git_hash", return_value="abc123")
    def test_initializes_run(self, mock_git, mock_setup, mock_dashboard):
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_run.url = "http://wandb.test"
        mock_wandb.init.return_value = mock_run

        with patch.object(wandb_integration, "wandb", mock_wandb):
            config = {
                "name": "balance",
                "env_kwargs": {"forward_vel_weight": 1.0},
                "ppo_kwargs": {"learning_rate": 3e-4},
                "sac_kwargs": {"tau": 0.005},
            }
            result = wandb_integration.init_wandb(
                "velociraptor", 1, config, tags=["test"], notes="test run"
            )

        assert result == mock_run
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["name"] == "velociraptor-stage1"
        assert "velociraptor" in call_kwargs["tags"]
        assert "test" in call_kwargs["tags"]
        assert call_kwargs["config"]["species"] == "velociraptor"
        assert call_kwargs["config"]["env/forward_vel_weight"] == 1.0
        assert call_kwargs["config"]["ppo/learning_rate"] == 3e-4
        assert call_kwargs["config"]["sac/tau"] == 0.005
        mock_setup.assert_called_once_with(1)
        mock_dashboard.assert_called_once_with(1)

    @patch.object(wandb_integration, "create_wandb_dashboard")
    @patch.object(wandb_integration, "setup_wandb_metrics")
    @patch.object(wandb_integration, "_get_git_hash", return_value="abc123")
    def test_no_optional_tags(self, mock_git, mock_setup, mock_dashboard):
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "run"
        mock_run.url = "http://test"
        mock_wandb.init.return_value = mock_run

        with patch.object(wandb_integration, "wandb", mock_wandb):
            wandb_integration.init_wandb("trex", 2, {"name": "loco"})

        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["tags"] == ["trex", "stage2"]


class TestWandbCallback:
    def test_init_sets_attributes(self):
        cb = wandb_integration.WandbCallback(
            log_freq=500, video_freq=10000, video_length=100, verbose=1
        )
        assert cb.log_freq == 500
        assert cb.video_freq == 10000
        assert cb.video_length == 100
        assert cb._last_video_step == 0

    @patch.object(wandb_integration, "wandb")
    def test_on_step_noop_when_no_run(self, mock_wandb):
        mock_wandb.run = None
        cb = wandb_integration.WandbCallback()
        cb.num_timesteps = 1000
        cb.locals = {}
        assert cb._on_step() is True
        mock_wandb.log.assert_not_called()

    @patch.object(wandb_integration, "wandb")
    def test_on_step_skips_when_not_log_freq(self, mock_wandb):
        mock_wandb.run = MagicMock()
        cb = wandb_integration.WandbCallback(log_freq=1000)
        cb.num_timesteps = 500
        cb.locals = {}
        assert cb._on_step() is True
        mock_wandb.log.assert_not_called()

    @patch.object(wandb_integration, "wandb")
    def test_on_step_logs_metrics_with_infos(self, mock_wandb):
        mock_wandb.run = MagicMock()
        cb = wandb_integration.WandbCallback(log_freq=1000)
        cb.num_timesteps = 1000
        cb.locals = {
            "infos": [
                {"reward_forward": 1.0, "reward_alive": 0.1, "forward_vel": 2.0},
                {"reward_forward": 2.0, "reward_alive": 0.2, "forward_vel": 3.0},
            ]
        }
        cb.model = MagicMock()
        cb.model.learning_rate = 3e-4
        cb.video_env = None

        assert cb._on_step() is True
        mock_wandb.log.assert_called_once()
        logged = mock_wandb.log.call_args[0][0]
        assert "train/timesteps" in logged
        assert logged["reward/reward_forward"] == pytest.approx(1.5)
        assert logged["reward/reward_alive"] == pytest.approx(0.15)
        assert logged["train/learning_rate"] == 3e-4

    @patch.object(wandb_integration, "wandb")
    def test_on_step_callable_learning_rate(self, mock_wandb):
        mock_wandb.run = MagicMock()
        cb = wandb_integration.WandbCallback(log_freq=1000)
        cb.num_timesteps = 1000
        cb.locals = {"infos": []}
        cb.model = MagicMock()
        cb.model.learning_rate = lambda progress: 3e-4 * progress
        cb.video_env = None

        cb._on_step()
        logged = mock_wandb.log.call_args[0][0]
        assert logged["train/learning_rate"] == pytest.approx(3e-4)

    @patch.object(wandb_integration, "wandb")
    def test_on_step_no_lr_attribute(self, mock_wandb):
        mock_wandb.run = MagicMock()
        cb = wandb_integration.WandbCallback(log_freq=1000)
        cb.num_timesteps = 1000
        cb.locals = {"infos": []}
        cb.model = MagicMock(spec=[])  # no learning_rate attr
        cb.video_env = None

        cb._on_step()
        logged = mock_wandb.log.call_args[0][0]
        assert "train/learning_rate" not in logged

    @patch.object(wandb_integration, "wandb")
    def test_on_step_triggers_video_recording(self, mock_wandb):
        mock_wandb.run = MagicMock()
        mock_wandb.Video = MagicMock()

        mock_env = MagicMock()
        mock_env.reset.return_value = np.zeros(10)
        mock_env.step.return_value = (
            np.zeros(10),
            np.array([0.0]),
            np.array([True]),
            [{}],
        )
        mock_env.render.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

        cb = wandb_integration.WandbCallback(
            log_freq=1000, video_env=mock_env, video_freq=1000
        )
        cb.num_timesteps = 1000
        cb._last_video_step = 0
        cb.locals = {"infos": []}
        mock_model = MagicMock(spec=[])
        mock_model.predict = MagicMock(return_value=(np.zeros(10), None))
        cb.model = mock_model

        cb._on_step()
        assert cb._last_video_step == 1000

    @patch.object(wandb_integration, "wandb")
    def test_record_video_noop_no_env(self, mock_wandb):
        mock_wandb.run = MagicMock()
        cb = wandb_integration.WandbCallback()
        cb.video_env = None
        cb._record_video()
        mock_wandb.log.assert_not_called()

    @patch.object(wandb_integration, "wandb")
    def test_record_video_noop_no_run(self, mock_wandb):
        mock_wandb.run = None
        cb = wandb_integration.WandbCallback(video_env=MagicMock())
        cb._record_video()
        mock_wandb.log.assert_not_called()

    @patch.object(wandb_integration, "wandb")
    def test_record_video_logs_frames(self, mock_wandb):
        mock_wandb.run = MagicMock()
        mock_wandb.Video = MagicMock()

        mock_env = MagicMock()
        mock_env.reset.return_value = np.zeros(10)
        mock_env.step.return_value = (
            np.zeros(10),
            np.array([0.0]),
            np.array([True]),
            [{}],
        )
        mock_env.render.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

        cb = wandb_integration.WandbCallback(video_env=mock_env, video_length=5)
        cb.num_timesteps = 1000
        cb.model = MagicMock()
        cb.model.predict.return_value = (np.zeros(10), None)

        cb._record_video()
        mock_wandb.log.assert_called_once()

    @patch.object(wandb_integration, "wandb")
    def test_record_video_render_exception(self, mock_wandb):
        mock_wandb.run = MagicMock()

        mock_env = MagicMock()
        mock_env.reset.return_value = np.zeros(10)
        mock_env.step.return_value = (
            np.zeros(10),
            np.array([0.0]),
            np.array([False]),
            [{}],
        )
        mock_env.render.side_effect = RuntimeError("render error")

        cb = wandb_integration.WandbCallback(video_env=mock_env, video_length=5)
        cb.num_timesteps = 1000
        cb.model = MagicMock()
        cb.model.predict.return_value = (np.zeros(10), None)

        cb._record_video()
        # No frames collected due to exception, no video logged
        mock_wandb.log.assert_not_called()

    @patch.object(wandb_integration, "wandb")
    def test_record_video_no_numpy(self, mock_wandb):
        mock_wandb.run = MagicMock()
        cb = wandb_integration.WandbCallback(video_env=MagicMock())
        cb.num_timesteps = 1000

        with patch.object(wandb_integration, "np", None):
            cb._record_video()
        mock_wandb.log.assert_not_called()

    @patch.object(wandb_integration, "wandb")
    def test_on_rollout_end_noop_no_run(self, mock_wandb):
        mock_wandb.run = None
        cb = wandb_integration.WandbCallback()
        cb._on_rollout_end()
        mock_wandb.log.assert_not_called()

    @patch.object(wandb_integration, "wandb")
    def test_on_rollout_end_logs_metrics(self, mock_wandb):
        mock_wandb.run = MagicMock()
        cb = wandb_integration.WandbCallback()
        cb.num_timesteps = 5000

        mock_logger = MagicMock()
        mock_logger.name_to_value = {
            "rollout/ep_rew_mean": 150.0,
            "rollout/ep_len_mean": 500,
            "rollout/not_numeric": "skip",
        }
        cb.model = MagicMock()
        cb.model.logger = mock_logger

        cb._on_rollout_end()
        mock_wandb.log.assert_called_once()
        logged = mock_wandb.log.call_args[0][0]
        assert "rollout/rollout_ep_rew_mean" in logged
        assert "rollout/rollout_ep_len_mean" in logged
        assert "rollout/rollout_not_numeric" not in logged

    @patch.object(wandb_integration, "wandb")
    def test_on_rollout_end_no_logger(self, mock_wandb):
        mock_wandb.run = MagicMock()
        cb = wandb_integration.WandbCallback()
        cb.num_timesteps = 5000
        cb.model = MagicMock(spec=[])  # no logger attribute
        cb._on_rollout_end()
        mock_wandb.log.assert_not_called()

    @patch.object(wandb_integration, "wandb")
    def test_on_rollout_end_empty_name_to_value(self, mock_wandb):
        mock_wandb.run = MagicMock()
        cb = wandb_integration.WandbCallback()
        cb.num_timesteps = 5000
        mock_logger = MagicMock()
        mock_logger.name_to_value = {}
        cb.model = MagicMock()
        cb.model.logger = mock_logger
        cb._on_rollout_end()
        mock_wandb.log.assert_not_called()


class TestLogEvalMetrics:
    @patch.object(wandb_integration, "wandb", None)
    def test_noop_when_not_installed(self):
        wandb_integration.log_eval_metrics({"mean_reward": 10.0}, stage=1)

    @patch.object(wandb_integration, "wandb")
    def test_noop_when_no_run(self, mock_wandb):
        mock_wandb.run = None
        wandb_integration.log_eval_metrics({"mean_reward": 10.0}, stage=1)
        mock_wandb.log.assert_not_called()

    @patch.object(wandb_integration, "wandb")
    def test_logs_numeric_metrics(self, mock_wandb):
        mock_wandb.run = MagicMock()
        results = {"mean_reward": 10.0, "mean_length": 500.0}
        wandb_integration.log_eval_metrics(results, stage=2, step=1000)

        mock_wandb.log.assert_called_once()
        logged = mock_wandb.log.call_args[0][0]
        assert logged["eval/stage"] == 2.0
        assert logged["eval/mean_reward"] == 10.0
        assert logged["eval/mean_length"] == 500.0
        assert mock_wandb.log.call_args[1]["step"] == 1000

    @patch.object(wandb_integration, "wandb")
    def test_logs_termination_counts(self, mock_wandb):
        mock_wandb.run = MagicMock()
        results = {
            "termination_counts": {"fallen": 3, "truncated": 7},
            "mean_reward": 10.0,
        }
        wandb_integration.log_eval_metrics(results, stage=1)

        logged = mock_wandb.log.call_args[0][0]
        assert logged["eval/termination/fallen"] == 3.0
        assert logged["eval/termination/truncated"] == 7.0

    @patch.object(wandb_integration, "wandb")
    def test_skips_non_numeric(self, mock_wandb):
        mock_wandb.run = MagicMock()
        results = {"some_string": "hello", "mean_reward": 5.0}
        wandb_integration.log_eval_metrics(results, stage=1)
        logged = mock_wandb.log.call_args[0][0]
        assert "eval/some_string" not in logged


class TestSetupWandbMetrics:
    @patch.object(wandb_integration, "wandb", None)
    def test_noop_when_not_installed(self):
        wandb_integration.setup_wandb_metrics(1)

    @patch.object(wandb_integration, "wandb")
    def test_noop_when_no_run(self, mock_wandb):
        mock_wandb.run = None
        wandb_integration.setup_wandb_metrics(1)
        mock_wandb.define_metric.assert_not_called()

    @patch.object(wandb_integration, "wandb")
    def test_defines_metrics_for_all_stages(self, mock_wandb):
        mock_wandb.run = MagicMock()
        wandb_integration.setup_wandb_metrics(2)

        assert mock_wandb.define_metric.call_count >= 5
        call_strs = [str(c) for c in mock_wandb.define_metric.call_args_list]
        assert any("eval/*" in c for c in call_strs)
        assert any("reward/*" in c for c in call_strs)


class TestCreateWandbDashboard:
    @patch.object(wandb_integration, "wandb", None)
    def test_noop_when_wandb_not_installed(self):
        wandb_integration.create_wandb_dashboard(1)

    @patch.object(wandb_integration, "wandb")
    def test_noop_when_no_run(self, mock_wandb):
        mock_wandb.run = None
        wandb_integration.create_wandb_dashboard(1)

    @patch.object(wandb_integration, "wandb")
    def test_fallback_when_workspaces_not_installed(self, mock_wandb):
        mock_wandb.run = MagicMock()
        mock_wandb.config = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "wandb_workspaces": None,
                "wandb_workspaces.workspaces": None,
                "wandb_workspaces.reports": None,
                "wandb_workspaces.reports.v2": None,
            },
        ):
            wandb_integration.create_wandb_dashboard(1)

        mock_wandb.config.update.assert_called_once()

    @patch("wandb_workspaces.workspaces.Workspace")
    @patch.object(wandb_integration, "wandb")
    def test_creates_workspace(self, mock_wandb, mock_workspace_cls):
        mock_wandb.run = MagicMock()
        mock_wandb.run.entity = "test-entity"

        wandb_integration.create_wandb_dashboard(2, entity="my-team")

        mock_workspace_cls.assert_called_once()
        mock_workspace_cls.return_value.save.assert_called_once()

    @patch("wandb_workspaces.workspaces.Workspace")
    @patch.object(wandb_integration, "wandb")
    def test_uses_run_entity_when_none(self, mock_wandb, mock_workspace_cls):
        mock_wandb.run = MagicMock()
        mock_wandb.run.entity = "auto-entity"

        wandb_integration.create_wandb_dashboard(1, entity=None)

        ws_call = mock_workspace_cls.call_args
        assert ws_call[1]["entity"] == "auto-entity"

    @patch("wandb_workspaces.workspaces.Workspace")
    @patch.object(wandb_integration, "wandb")
    def test_fallback_on_save_error(self, mock_wandb, mock_workspace_cls):
        mock_wandb.run = MagicMock()
        mock_wandb.run.entity = "test-entity"
        mock_wandb.config = MagicMock()
        mock_workspace_cls.return_value.save.side_effect = Exception("API error")

        wandb_integration.create_wandb_dashboard(1)

        mock_wandb.config.update.assert_called_once()

    @patch("wandb_workspaces.workspaces.Workspace")
    @patch.object(wandb_integration, "wandb")
    def test_stage3_section_open(self, mock_wandb, mock_workspace_cls):
        """Stage 3 should have the Strike section open."""
        mock_wandb.run = MagicMock()
        mock_wandb.run.entity = "e"

        wandb_integration.create_wandb_dashboard(3)

        mock_workspace_cls.assert_called_once()


class TestSaveDashboardConfigFallback:
    @patch.object(wandb_integration, "wandb")
    def test_saves_panel_config(self, mock_wandb):
        mock_wandb.config = MagicMock()
        wandb_integration._save_dashboard_config_fallback(2)

        mock_wandb.config.update.assert_called_once()
        call_args = mock_wandb.config.update.call_args
        panel_config = call_args[0][0]["dashboard_panels"]
        assert "master" in panel_config
        assert "stage1_balance" in panel_config
        assert "stage2_locomotion" in panel_config
        assert "stage3_strike" in panel_config
        assert call_args[1]["allow_val_change"] is True
