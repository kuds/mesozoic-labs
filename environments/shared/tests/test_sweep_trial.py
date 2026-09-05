"""Tests for sweep trial.py — HPT arg parsing and override conversion."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from environments.shared.scripts.sweep import (
    _hpt_arg_to_override,
    _parse_hpt_extra_args,
)
from environments.shared.scripts.sweep.trial import _log_hardware_info


class TestLogHardwareInfo:
    """Hardware logging must remain best-effort so diagnostics cannot abort a trial."""

    def test_cuda_available_logs_24gb_using_total_memory(self):
        cuda = MagicMock()
        cuda.is_available.return_value = True
        cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        cuda.get_device_properties.return_value = SimpleNamespace(total_memory=24_000_000_000)
        fake_torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.1"))

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("environments.shared.scripts.sweep.trial.logger.info") as mock_info,
        ):
            _log_hardware_info()

        mock_info.assert_any_call("  GPU: %s (CUDA %s)", "NVIDIA RTX 4090", "12.1")
        mock_info.assert_any_call("  GPU memory: %.1f GB", 24.0)

    def test_cuda_unavailable_logs_cpu_only(self):
        cuda = MagicMock()
        cuda.is_available.return_value = False
        fake_torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda=None))

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("environments.shared.scripts.sweep.trial.logger.info") as mock_info,
        ):
            _log_hardware_info()

        mock_info.assert_any_call("  GPU: none (CPU-only training)")
        cuda.get_device_properties.assert_not_called()

    def test_torch_unavailable_logs_cpu_only(self):
        with (
            patch.dict("sys.modules", {"torch": None}),
            patch("environments.shared.scripts.sweep.trial.logger.info") as mock_info,
        ):
            _log_hardware_info()

        mock_info.assert_any_call("  GPU: torch not installed (CPU-only training)")

    def test_device_property_lookup_failure_does_not_raise(self):
        cuda = MagicMock()
        cuda.is_available.return_value = True
        cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        cuda.get_device_properties.side_effect = RuntimeError("CUDA device query failed")
        fake_torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.1"))

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("environments.shared.scripts.sweep.trial.logger.warning") as mock_warning,
        ):
            _log_hardware_info()

        mock_warning.assert_called_once_with(
            "  GPU diagnostics unavailable; continuing training: %s",
            cuda.get_device_properties.side_effect,
        )


# ── _hpt_arg_to_override ────────────────────────────────────────────────────


class TestHptArgToOverride:
    """Conversion of Vertex AI HPT arg names to --override dot notation."""

    def test_ppo_prefix(self):
        assert _hpt_arg_to_override("ppo_learning_rate", "0.0003") == "ppo.learning_rate=0.0003"

    def test_sac_prefix(self):
        assert _hpt_arg_to_override("sac_batch_size", "256") == "sac.batch_size=256"

    def test_env_prefix(self):
        assert _hpt_arg_to_override("env_alive_bonus", "2.0") == "env.alive_bonus=2.0"

    def test_curriculum_prefix(self):
        assert _hpt_arg_to_override("curriculum_warmup_timesteps", "50000") == "curriculum.warmup_timesteps=50000"

    def test_unknown_prefix_passthrough(self):
        assert _hpt_arg_to_override("unknown_param", "42") == "unknown_param=42"

    def test_net_arch_preset(self):
        assert _hpt_arg_to_override("ppo_net_arch", "medium") == "ppo.net_arch=medium"

    def test_multi_underscore_param(self):
        """Only the first underscore after the prefix becomes the dot separator."""
        assert _hpt_arg_to_override("ppo_clip_range", "0.2") == "ppo.clip_range=0.2"

    def test_env_multi_word_param(self):
        assert _hpt_arg_to_override("env_forward_vel_weight", "1.5") == "env.forward_vel_weight=1.5"


# ── _parse_hpt_extra_args ────────────────────────────────────────────────────


class TestParseHptExtraArgs:
    """Parsing of HPT-injected extra CLI args into override strings."""

    def test_equals_format(self):
        """Vertex AI HPT uses --key=value format."""
        extra = ["--ppo_learning_rate=0.0001", "--ppo_ent_coef=0.005"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.learning_rate=0.0001", "ppo.ent_coef=0.005"]

    def test_space_separated_format(self):
        """Fallback --key value format."""
        extra = ["--ppo_learning_rate", "0.0001", "--ppo_ent_coef", "0.005"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.learning_rate=0.0001", "ppo.ent_coef=0.005"]

    def test_mixed_formats(self):
        """Both formats can appear in the same arg list."""
        extra = ["--ppo_learning_rate=0.0001", "--ppo_ent_coef", "0.005"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.learning_rate=0.0001", "ppo.ent_coef=0.005"]

    def test_empty(self):
        assert _parse_hpt_extra_args([]) == []

    def test_boolean_flags_skipped(self):
        """Boolean flags (no value) are skipped."""
        extra = ["--some_flag", "--ppo_learning_rate=0.0001"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.learning_rate=0.0001"]

    def test_discrete_values(self):
        """Discrete HPT params like batch_size come as floats from Vertex AI."""
        extra = ["--ppo_batch_size=256.0", "--ppo_n_steps=2048.0"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.batch_size=256.0", "ppo.n_steps=2048.0"]

    def test_multiple_params_equals_format(self):
        """Realistic Vertex AI HPT injection with 5 parameters."""
        extra = [
            "--ppo_learning_rate=0.0001",
            "--ppo_ent_coef=0.005",
            "--ppo_batch_size=128.0",
            "--ppo_gamma=0.985",
            "--ppo_n_steps=2048.0",
        ]
        result = _parse_hpt_extra_args(extra)
        assert len(result) == 5
        assert "ppo.learning_rate=0.0001" in result
        assert "ppo.batch_size=128.0" in result

    def test_non_flag_tokens_skipped(self):
        """Bare tokens without -- prefix are skipped."""
        extra = ["stray_token", "--ppo_learning_rate=0.0001"]
        result = _parse_hpt_extra_args(extra)
        assert result == ["ppo.learning_rate=0.0001"]


# ── run_trial task-load-mode passthrough ─────────────────────────────────────


class TestRunTrialLoadMode:
    """A warm-started trial forwards its task load mode to train().

    Without it every trial launch-all chains from a previous stage's winner
    validated under train()'s default resume_same_stage and raised
    TaskFingerprintError at startup.
    """

    @staticmethod
    def _args(**overrides):
        base = dict(
            species="trex",
            stage=1,
            algorithm="ppo",
            timesteps=1000,
            n_envs=1,
            seed=42,
            eval_freq=100,
            save_freq=100,
            load=None,
            output_dir=None,
            wandb=False,
            no_tensorboard=True,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    @staticmethod
    def _train_kwargs(args, task_load_mode=None):
        from environments.shared.scripts.sweep.trial import run_trial

        train = MagicMock()
        stage_configs = {1: {"ppo_kwargs": {}, "curriculum_kwargs": {}}}
        with (
            patch("environments.shared.species_registry.get_species_config", return_value=object()),
            patch("environments.shared.config.load_all_stages", return_value=stage_configs),
            patch("environments.shared.train_base.train", train),
            patch("environments.shared.scripts.sweep.trial._log_hardware_info"),
        ):
            run_trial(args, [], task_load_mode=task_load_mode)
        train.assert_called_once()
        return train.call_args.kwargs

    def test_cli_load_mode_reaches_train(self):
        kwargs = self._train_kwargs(
            self._args(load="/gcs/b/sweeps/trex/stage1/7/models/best_model.zip", load_mode="initialize_next_stage")
        )
        assert kwargs["load_path"] == "/gcs/b/sweeps/trex/stage1/7/models/best_model.zip"
        assert kwargs["task_load_mode"] == "initialize_next_stage"

    def test_explicit_parameter_wins_over_args(self):
        kwargs = self._train_kwargs(self._args(load_mode="resume_same_stage"), task_load_mode="initialize_next_stage")
        assert kwargs["task_load_mode"] == "initialize_next_stage"

    def test_default_is_resume_same_stage(self):
        """No flag and no parameter keeps today's behaviour."""
        kwargs = self._train_kwargs(self._args())
        assert kwargs["task_load_mode"] == "resume_same_stage"

    def test_trial_cli_parses_load_mode(self):
        from environments.shared.scripts.sweep.__main__ import _build_parser
        from environments.shared.task_fingerprint import LOAD_MODES

        parser = _build_parser()
        args, _ = parser.parse_known_args(["trial", "--species", "trex", "--load-mode", "initialize_next_stage"])
        assert args.load_mode == "initialize_next_stage"
        args, _ = parser.parse_known_args(["trial", "--species", "trex"])
        assert args.load_mode == "resume_same_stage"
        assert set(
            parser._subparsers._group_actions[0].choices["trial"]._option_string_actions["--load-mode"].choices
        ) == set(LOAD_MODES)
        with pytest.raises(SystemExit):
            parser.parse_args(["trial", "--species", "trex", "--load-mode", "bogus"])
