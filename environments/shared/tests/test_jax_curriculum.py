"""Tests for the JAX curriculum gate logic (jax_curriculum.py).

These tests run without JAX installed — check_stage_gate is pure Python.
"""

from __future__ import annotations

import pytest

from environments.shared.curriculum.gate_schema import GATE_SCHEMA_VERSION, GateSchemaError
from environments.shared.curriculum.stance_gate import STANCE_GATE_KIND
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
        # The shipped config also declares a length floor, which the gate now
        # enforces, so hold length comfortably clear to isolate the reward
        # threshold this test is about.
        passing_length = stage_config["curriculum_kwargs"]["min_avg_episode_length"] + 1.0
        assert (
            check_stage_gate(
                {"mean_episode_return": min_reward + 1.0, "mean_episode_length": passing_length},
                stage_config,
            )
            is True
        )
        assert (
            check_stage_gate(
                {"mean_episode_return": min_reward - 1.0, "mean_episode_length": passing_length},
                stage_config,
            )
            is False
        )

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
        """A declared gate with nothing to check must raise, not pass.

        The raise now comes from the schema itself rather than a JAX-side
        check, so the SB3 path rejects the same config instead of falling back
        to StageThreshold's permissive ``min_avg_reward = -inf`` default.
        """
        with pytest.raises(GateSchemaError, match="missing required threshold"):
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


class TestLengthThresholdIsEnforced:
    """The length half of ``reward_and_length/v1`` must bind on JAX too.

    ``check_stage_gate`` used to read only ``min_avg_reward`` while the SB3
    ``CurriculumManager`` enforced reward AND length, so a stage carrying
    ``min_avg_episode_length`` advanced on reward alone under JAX.  That is the
    "one backend silently ignores a gate the other enforces" divergence
    ``gate_schema`` exists to prevent, and it became load-bearing when stage 1
    started encoding its full-horizon floor in that field.
    """

    def test_short_episodes_fail_even_with_sufficient_reward(self):
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0, min_avg_episode_length=950)}
        metrics = {"mean_episode_return": 5000.0, "mean_episode_length": 300.0}
        assert check_stage_gate(metrics, stage_config) is False

    def test_passes_when_both_thresholds_are_met(self):
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0, min_avg_episode_length=950)}
        metrics = {"mean_episode_return": 5000.0, "mean_episode_length": 1000.0}
        assert check_stage_gate(metrics, stage_config) is True

    def test_reward_still_binds_independently(self):
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0, min_avg_episode_length=950)}
        metrics = {"mean_episode_return": 50.0, "mean_episode_length": 1000.0}
        assert check_stage_gate(metrics, stage_config) is False

    def test_declared_length_gate_without_the_metric_is_fatal(self):
        """Half-enforcing must raise rather than pass on reward alone."""
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0, min_avg_episode_length=950)}
        with pytest.raises(GateSchemaError, match="no mean_episode_length"):
            check_stage_gate({"mean_episode_return": 5000.0}, stage_config)

    def test_reward_only_config_is_unaffected(self):
        """A stage that declares no length threshold keeps its old behaviour."""
        stage_config = {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0)}
        assert check_stage_gate({"mean_episode_return": 150.0}, stage_config) is True

    def test_every_stage_one_config_binds_a_full_horizon_floor(self):
        """Every shipped stage-1 config enforces a full-horizon requirement.

        The two gate kinds express it differently -- ``reward_and_length/v1``
        as a mean episode-step floor, ``stance_quality/v1`` as a fraction of
        episodes reaching the horizon -- but neither may be satisfied by
        unbounded reward alone. That is the property worth pinning: the
        specific field is an implementation detail of the kind.
        """
        from environments.shared.config import load_stage_config

        for species in ("trex", "velociraptor", "brachiosaurus", "dibothrosuchus"):
            cfg = load_stage_config(species, 1)
            curriculum = cfg["curriculum_kwargs"]
            if curriculum["gate_kind"] == STANCE_GATE_KIND:
                floor = curriculum["min_full_horizon_fraction"]
                generous = {
                    "mean_episode_return": 1e9,
                    "n_eval_episodes": curriculum["min_eval_episodes"],
                    "full_horizon_fraction": floor - 0.01,
                    "n_duty_episodes": curriculum["min_eval_episodes"],
                    "mean_unsupported_duty": 0.0,
                    "unsupported_duty_ucb": 0.0,
                }
            else:
                floor = curriculum["min_avg_episode_length"]
                generous = {"mean_episode_return": 1e9, "mean_episode_length": floor - 1}
            assert check_stage_gate(generous, cfg) is False, species
