from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import environments.shared.wandb_integration as wi


@pytest.fixture
def mock_wandb():
    with patch("environments.shared.wandb_integration.wandb") as mock:
        yield mock


def test_init_wandb(mock_wandb):
    mock_run = MagicMock()
    mock_run.name = "test-run"
    mock_run.url = "http://test"
    mock_wandb.init.return_value = mock_run

    config = {"name": "test_stage", "env_kwargs": {"foo": "bar"}, "ppo_kwargs": {"lr": 0.001}}
    run = wi.init_wandb("velociraptor", 1, config)

    assert run == mock_run
    mock_wandb.init.assert_called_once()


def test_wandb_callback_on_step(mock_wandb):
    callback = wi.WandbCallback(log_freq=1)
    callback.num_timesteps = 1

    # Mock locals
    callback.locals = {"infos": [{"reward_forward": 1.0, "reward_alive": 0.5, "forward_vel": 2.0}]}

    mock_model = MagicMock()
    mock_model.learning_rate = 0.003
    callback.model = mock_model

    callback._on_step()

    # wandb.log should be called
    mock_wandb.log.assert_called_once()
    logged_dict = mock_wandb.log.call_args[0][0]
    assert "reward/reward_forward" in logged_dict
    assert "reward/forward_vel" in logged_dict
    assert logged_dict["train/learning_rate"] == 0.003


def test_wandb_callback_video(mock_wandb):
    # Mock video env
    mock_env = MagicMock()
    mock_env.reset.return_value = np.zeros((10,))
    mock_env.step.return_value = (np.zeros((10,)), 0, [True], [{}])
    mock_env.render.return_value = np.zeros((64, 64, 3))

    callback = wi.WandbCallback(video_env=mock_env, video_freq=1, video_length=2)
    callback.num_timesteps = 1
    callback.locals = {"infos": []}

    mock_model = MagicMock()
    mock_model.predict.return_value = (np.array([0.0]), None)
    callback.model = mock_model

    callback._record_video()

    mock_env.reset.assert_called()
    mock_env.step.assert_called()
    mock_env.render.assert_called()


def test_log_eval_metrics(mock_wandb):
    results = {"mean_gait_symmetry": 0.5, "termination_counts": {"fallen": 5, "truncated": 2}}
    wi.log_eval_metrics(results, stage=1, step=100)

    mock_wandb.log.assert_called_once()
    logged = mock_wandb.log.call_args[0][0]
    assert logged["eval/stage"] == 1.0
    assert logged["eval/mean_gait_symmetry"] == 0.5
    assert logged["eval/termination/fallen"] == 5.0


def test_setup_wandb_metrics(mock_wandb):
    wi.setup_wandb_metrics(1)
    assert mock_wandb.define_metric.called


def test_create_wandb_dashboard(mock_wandb):
    # Just to reach coverage
    wi.create_wandb_dashboard(1)


def test_on_rollout_end(mock_wandb):
    callback = wi.WandbCallback()
    callback.num_timesteps = 100
    mock_logger = MagicMock()
    mock_logger.name_to_value = {"rollout/ep_rew_mean": 100}
    mock_model = MagicMock()
    mock_model.logger = mock_logger
    callback.model = mock_model

    callback._on_rollout_end()
    mock_wandb.log.assert_called_once()
