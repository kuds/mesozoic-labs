"""Tests for environments.shared.curriculum.sb3_compat.

Every SB3-dependent entry point in the package must degrade predictably when
stable-baselines3 is not installed, so these cases span several submodules on
purpose — the shim's contract is what is under test."""

from unittest.mock import MagicMock, patch

import pytest

from environments.shared.curriculum import (
    CurriculumCallback,
    RewardRampCallback,
    StageWarmupCallback,
    load_vecnorm_stats,
)


class TestCallbacksWithoutSB3:
    """Test that SB3-dependent classes fail gracefully when SB3 is unavailable."""

    def test_curriculum_callback_raises_without_sb3(self):
        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                CurriculumCallback(
                    curriculum_manager=MagicMock(),
                    eval_env=MagicMock(),
                )

    def test_stage_warmup_callback_raises_without_sb3(self):
        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                StageWarmupCallback()

    def test_reward_ramp_callback_raises_without_sb3(self):
        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                RewardRampCallback()

    def test_load_vecnorm_stats_returns_false_without_sb3(self):
        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", False):
            result = load_vecnorm_stats("/any/path.pkl", MagicMock(), unsafe_skip_plant_validation=True)
            assert result is False
