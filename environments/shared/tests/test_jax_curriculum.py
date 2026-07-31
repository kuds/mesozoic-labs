"""Tests for the JAX curriculum gate logic (jax_curriculum.py).

These tests run without JAX installed — check_stage_gate is pure Python.
"""

from __future__ import annotations

import pytest

from environments.shared.curriculum.gate_schema import GATE_SCHEMA_VERSION, GateSchemaError
from environments.shared.jax_curriculum import check_stage_gate

#: The gate declaration every real stage config carries. Tests that exercise a
#: working gate must include it, because an undeclared gate is now fatal.
_GATE = {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "reward_and_length/v1"}


class TestCheckStageGate:
    def test_passes_when_reward_meets_threshold(self):
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0)}
        assert check_stage_gate({"mean_episode_return": 150.0}, stage_config) is True

    def test_fails_when_reward_below_threshold(self):
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0)}
        assert check_stage_gate({"mean_episode_return": 50.0}, stage_config) is False

    def test_reads_curriculum_kwargs_key_from_real_config(self):
        """Regression: the gate must read the loader's 'curriculum_kwargs'
        key.  It previously read 'curriculum' (the raw TOML table name),
        which load_stage_config never emits — min_reward defaulted to inf
        and every curriculum run stopped after stage 1.
        """
        from environments.shared.config import load_stage_config

        stage_config = load_stage_config("velociraptor", 1)
        assert "curriculum_kwargs" in stage_config
        min_reward = stage_config["curriculum_kwargs"]["min_avg_reward"]
        assert check_stage_gate({"mean_episode_return": min_reward + 1.0}, stage_config) is True
        assert check_stage_gate({"mean_episode_return": min_reward - 1.0}, stage_config) is False

    def test_prefers_episode_return_over_per_step_reward(self):
        """Regression: JaxTrainer eval_metrics carry BOTH a per-step
        'mean_reward' (~0.5-2 for a good policy) and an episode-level
        'mean_episode_return'.  The TOML min_avg_reward thresholds are
        episode-level, so the gate must use the episode return — gating on
        the per-step value failed every well-trained policy.
        """
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0)}
        realistic_trainer_metrics = {"mean_reward": 1.2, "mean_episode_return": 240.0}
        assert check_stage_gate(realistic_trainer_metrics, stage_config) is True

    def test_falls_back_to_mean_reward_when_no_episode_return(self):
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0)}
        assert check_stage_gate({"mean_reward": 150.0}, stage_config) is True
        assert check_stage_gate({"mean_reward": 50.0}, stage_config) is False

    def test_missing_gate_declaration_is_fatal(self):
        """Fail closed: an undeclared gate must not advance by default.

        This previously logged a warning and returned ``True``, so a stage
        whose reward threshold had been removed advanced unconditionally.
        Combined with the SB3 path's permissive StageThreshold defaults, that
        made "no gate" indistinguishable from "gate satisfied" on both
        backends. See docs/STAGE1_SPLIT_PLAN.md section 5.2.
        """
        with pytest.raises(GateSchemaError, match="no gate_kind declared"):
            check_stage_gate({"mean_reward": 0.0}, {})
        with pytest.raises(GateSchemaError, match="no gate_kind declared"):
            check_stage_gate({"mean_reward": 0.0}, {"curriculum_kwargs": {}})

    def test_declared_gate_without_a_reward_threshold_is_fatal(self):
        """A declared gate with nothing to check must raise, not pass."""
        with pytest.raises(GateSchemaError, match="no min_avg_reward"):
            check_stage_gate({"mean_episode_return": 0.0}, {"curriculum_kwargs": dict(_GATE)})

    def test_non_advancing_pilot_cannot_be_used_to_advance(self):
        """gate_kind "none/v1" is the recorded way to run without a gate."""
        pilot = {"curriculum_kwargs": {"gate_schema_version": 1, "gate_kind": "none/v1"}}
        with pytest.raises(GateSchemaError, match="non-advancing pilot"):
            check_stage_gate({"mean_episode_return": 1e9}, pilot)

    def test_unknown_gate_kind_is_fatal(self):
        bogus = {"curriculum_kwargs": {"gate_schema_version": 1, "gate_kind": "made_up/v9"}}
        with pytest.raises(GateSchemaError, match="unknown gate_kind"):
            check_stage_gate({"mean_episode_return": 1e9}, bogus)

    def test_misspelled_threshold_is_fatal_rather_than_ignored(self):
        """A typo used to silently disable the threshold it meant to set."""
        typo = {"curriculum_kwargs": dict(_GATE, min_avg_rewrad=100.0)}
        with pytest.raises(GateSchemaError, match="unrecognised"):
            check_stage_gate({"mean_episode_return": 0.0}, typo)
