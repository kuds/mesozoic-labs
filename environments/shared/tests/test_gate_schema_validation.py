"""The ``[curriculum]`` schema's fail-closed checks on the SB3 path.

``validate_gate_config`` runs on every SB3 curriculum run (through
``thresholds_from_configs``). These cases were tested only through the JAX
curriculum's ``check_stage_gate`` until D-D17 retired it (cleanup PR-B); the
schema itself stays, including the ``[curriculum.jax]`` override validation
(docs/CLEANUP_PLAN_2026_09.md §4.9).
"""

from __future__ import annotations

import pytest

from environments.shared.config import load_stage_config
from environments.shared.curriculum import thresholds_from_configs
from environments.shared.curriculum.gate_schema import (
    GATE_SCHEMA_VERSION,
    GateSchemaError,
    validate_gate_config,
)

_GATE = {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "reward_and_length/v1"}
_STANCE = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": "stance_quality/v1",
    "min_full_horizon_fraction": 0.95,
    "max_unsupported_duty": 0.02,
    "max_unsupported_duty_ucb": 0.02,
    "min_eval_episodes": 40,
}

#: A stage TOML declaring the shared bar and a ``[curriculum.jax]`` override for it.
_OVERRIDE_TOML = (
    '[stage]\nname = "t"\n[env]\nalive_bonus = 1.0\n'
    '[curriculum]\ngate_schema_version = 1\ngate_kind = "reward_and_length/v1"\n'
    "min_avg_reward = 100.0\n[curriculum.jax]\nmin_avg_reward = 40.0\n"
)


def _validate(curriculum: dict) -> str:
    return validate_gate_config(1, curriculum, advancement_enabled=True)


class TestGateDeclaration:
    def test_unknown_gate_kind_is_fatal(self):
        with pytest.raises(GateSchemaError, match="unknown gate_kind"):
            _validate({"gate_schema_version": 1, "gate_kind": "made_up/v9"})

    def test_misspelled_threshold_is_fatal_rather_than_ignored(self):
        with pytest.raises(GateSchemaError, match="unrecognised"):
            _validate(dict(_GATE, min_avg_rewrad=100.0))


class TestBackendOverrideTable:
    """``[curriculum.jax]`` is still validated, although no backend applies it."""

    def test_a_valid_override_table_passes(self):
        assert _validate(dict(_GATE, min_avg_reward=100.0, jax={"min_avg_reward": 40.0})) == "reward_and_length/v1"

    def test_unknown_key_inside_the_override_table_is_fatal(self):
        with pytest.raises(GateSchemaError, match="cannot be overridden"):
            _validate(dict(_GATE, min_avg_reward=100.0, jax={"min_avg_rewrad": 40.0}))

    def test_a_composite_criterion_cannot_be_overridden_per_backend(self):
        with pytest.raises(GateSchemaError, match="cannot be overridden"):
            _validate(dict(_STANCE, jax={"max_unsupported_duty": 0.5}))

    def test_overriding_a_field_the_kind_does_not_consume_is_fatal(self):
        with pytest.raises(GateSchemaError, match="does not consume"):
            _validate(dict(_STANCE, jax={"min_success_rate": 0.5}))

    def test_non_numeric_and_non_table_overrides_are_fatal(self):
        with pytest.raises(GateSchemaError, match="finite numbers"):
            _validate(dict(_GATE, min_avg_reward=100.0, jax={"min_avg_reward": "x"}))
        with pytest.raises(GateSchemaError, match="must be a table"):
            _validate(dict(_GATE, min_avg_reward=100.0, jax=40.0))

    def test_the_sb3_thresholds_ignore_the_override_table(self):
        configs = {1: {"curriculum_kwargs": dict(_GATE, min_avg_reward=100.0, jax={"min_avg_reward": 40.0})}}
        thresholds = thresholds_from_configs(configs)
        assert thresholds[1]["min_avg_reward"] == 100.0
        assert "jax" not in thresholds[1]

    def test_a_toml_sub_table_loads_and_validates(self, tmp_path):
        path = tmp_path / "good.toml"
        path.write_text(_OVERRIDE_TOML)
        curriculum = load_stage_config("trex", 1, config_path=str(path))["curriculum_kwargs"]
        assert curriculum["min_avg_reward"] == 100.0
        assert curriculum["jax"] == {"min_avg_reward": 40.0}
        assert _validate(curriculum) == "reward_and_length/v1"

    def test_an_unknown_key_in_the_toml_sub_table_is_rejected_by_the_schema(self, tmp_path):
        """The loader passes the misspelling through, so the schema must refuse it."""
        path = tmp_path / "bad.toml"
        path.write_text(_OVERRIDE_TOML.replace("[curriculum.jax]\nmin_avg_reward", "[curriculum.jax]\nmin_avg_rewrad"))
        curriculum = load_stage_config("trex", 1, config_path=str(path))["curriculum_kwargs"]
        assert curriculum["jax"] == {"min_avg_rewrad": 40.0}
        with pytest.raises(GateSchemaError, match="cannot be overridden"):
            _validate(curriculum)
