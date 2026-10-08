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


#: A complete stance_quality/v2 declaration: exactly its required keys (decision D-D23).
_STANCE_V2 = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": "stance_quality/v2",
    "min_eval_episodes": 40,
    "min_clean_stance_lcb": 0.80,
    "settle_steps": 200,
    "min_all_feet_support": 0.98,
    "max_touchdown_rate": 0.25,
    "max_window_displacement_m": 0.10,
    "min_foot_load_share": 0.30,
    "max_actuator_saturation_fraction": 0.10,
    "max_settle_airborne_substeps": 0,
    "max_settle_peak_floor_force_bw": 1.5,
}


class TestStanceQualityV2Declaration:
    """The v2 block is validated like every kind: required keys by name, other kinds' keys misplaced."""

    def test_a_complete_block_validates(self):
        assert _validate(_STANCE_V2) == "stance_quality/v2"
        assert _validate(dict(_STANCE_V2, max_sole_corner_lift_m=0.006, min_avg_reward_statue_ratio=0.6)) == (
            "stance_quality/v2"
        )

    @pytest.mark.parametrize("value", [80, 0.0, 1.0, -0.2, True, "0.8", float("nan")])
    def test_a_fraction_bar_outside_its_range_is_refused_at_config_load(self, value):
        """``max_sole_cop_fore_aft = 80`` (a percent typo) stops the run here, not after it trained (D-D28)."""
        assert _validate(dict(_STANCE_V2, max_sole_cop_fore_aft=0.8)) == "stance_quality/v2"
        with pytest.raises(GateSchemaError, match="max_sole_cop_fore_aft must be a number strictly between 0 and 1"):
            _validate(dict(_STANCE_V2, max_sole_cop_fore_aft=value))
        with pytest.raises(GateSchemaError, match="strictly between 0 and 1"):
            thresholds_from_configs({1: {"curriculum_kwargs": dict(_STANCE_V2, max_sole_cop_fore_aft=value)}})

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("max_window_airborne_substeps", -3),
            ("max_window_airborne_substeps", -0.5),
            ("max_window_airborne_substeps", "40"),
            ("max_window_airborne_substeps", True),
            ("max_window_peak_floor_force_bw", 0.5),
            ("max_window_peak_floor_force_bw", "2.0"),
            ("max_window_peak_floor_force_bw", float("nan")),
        ],
    )
    def test_a_window_hop_bar_below_its_floor_is_refused_at_config_load(self, key, value):
        """A window hop bar no episode can meet stops the run here too, not after it trained (D-D26)."""
        floors = dict(_STANCE_V2, max_window_airborne_substeps=0, max_window_peak_floor_force_bw=1.0)
        assert _validate(floors) == "stance_quality/v2"
        with pytest.raises(GateSchemaError, match=rf"{key} must be a number of at least"):
            _validate(dict(floors, **{key: value}))
        with pytest.raises(GateSchemaError, match=rf"{key} must be a number of at least"):
            thresholds_from_configs({1: {"curriculum_kwargs": dict(floors, **{key: value})}})

    def test_every_v2_key_reaches_the_threshold(self):
        """The manager copies each key of the kind onto a v2 stage's threshold: one left out drops its bar there."""
        from environments.shared.curriculum import StageThreshold
        from environments.shared.curriculum.stance_gate_v2 import STANCE_V2_THRESHOLD_KEYS

        whole = {"max_hop_or_fall_episodes": 1, "required_consecutive": 3, "max_settle_touchdowns": 2}
        # The window peak's floor is the body weight (WINDOW_HOP_KEY_FLOORS), refused under it at config load.
        floored = {"max_window_peak_floor_force_bw": 2.0}
        block = {**_STANCE_V2, **{key: 0.5 for key in STANCE_V2_THRESHOLD_KEYS - set(_STANCE_V2)}, **whole, **floored}
        threshold = thresholds_from_configs({1: {"curriculum_kwargs": block}})[1]
        assert STANCE_V2_THRESHOLD_KEYS <= set(threshold)
        copied = StageThreshold(**threshold).stance_v2_thresholds()
        assert all(getattr(copied, key) == block[key] for key in STANCE_V2_THRESHOLD_KEYS), copied

    @pytest.mark.parametrize(
        "missing",
        sorted(set(_STANCE_V2) - {"gate_schema_version", "gate_kind"}),
    )
    def test_each_required_key_is_reported_missing_by_name(self, missing):
        block = {key: value for key, value in _STANCE_V2.items() if key != missing}
        with pytest.raises(GateSchemaError, match=rf"missing required threshold field\(s\) \['{missing}'\]"):
            _validate(block)

    @pytest.mark.parametrize("key", ["max_unsupported_duty", "max_unsupported_duty_ucb"])
    def test_a_v1_only_key_left_in_a_v2_block_is_misplaced(self, key):
        """A switch from v1 that leaves the duty ceilings behind would imply a gate nobody enforces."""
        with pytest.raises(GateSchemaError, match=rf"does not consume threshold field\(s\) \['{key}'\]"):
            _validate(dict(_STANCE_V2, **{key: 0.02}))

    @pytest.mark.parametrize("key", ["min_clean_stance_lcb", "max_sole_corner_lift_m", "min_avg_reward_statue_ratio"])
    def test_a_v2_key_in_a_v1_block_is_misplaced(self, key):
        with pytest.raises(GateSchemaError, match=rf"does not consume threshold field\(s\) \['{key}'\]"):
            _validate(dict(_STANCE, **{key: 0.5}))

    def test_the_gate_view_holds_exactly_the_declared_keys(self):
        """Registration moves no digest: only DECLARED keys of the kind enter the view."""
        from environments.shared.curriculum.gate_schema import gate_config_view

        block = dict(_STANCE_V2, max_sole_corner_lift_m=0.006, timesteps=11_000_000, stance_report_episodes=40)
        view = gate_config_view(block)
        assert view["gate_kind"] == "stance_quality/v2" and view["gate_schema_version"] == 1
        assert set(view["thresholds"]) == set(_STANCE_V2) - {"gate_schema_version", "gate_kind"} | {
            "max_sole_corner_lift_m"
        }

    def test_the_v2_keys_reach_the_threshold_and_the_manager_refuses_the_kind(self):
        from environments.shared.curriculum import CurriculumManager, StageThreshold

        thresholds = thresholds_from_configs({1: {"curriculum_kwargs": dict(_STANCE_V2, max_sole_corner_lift_m=0.006)}})
        assert thresholds[1]["gate_kind"] == "stance_quality/v2"
        assert thresholds[1]["min_clean_stance_lcb"] == 0.80 and thresholds[1]["max_sole_corner_lift_m"] == 0.006
        threshold = StageThreshold(**thresholds[1])
        copied = threshold.stance_v2_thresholds()
        assert copied.min_clean_stance_lcb == 0.80 and copied.settle_steps == 200
        assert copied.max_sole_corner_lift_m == 0.006 and copied.min_sole_contacts is None
        assert copied.min_avg_reward is None and copied.min_full_horizon_fraction is None
        manager = CurriculumManager(species="velociraptor", stage_thresholds=thresholds)
        assert not manager.should_advance([1e9] * 40, [1000.0] * 40)

    def test_every_v2_key_is_copied_onto_the_threshold(self):
        """Each key the kind consumes reaches :class:`StageThreshold` and its :meth:`stance_v2_thresholds` copy.

        The shared keys are copied for every kind; the rest come from
        ``_STANCE_V2_COPIED_KEYS``, so a key added to the kind but not to the
        tuple would reach the offline judge and silently miss the copy the
        in-training screen reads.
        """
        from dataclasses import fields

        from environments.shared.curriculum import StageThreshold
        from environments.shared.curriculum.manager import _STANCE_V2_COPIED_KEYS
        from environments.shared.curriculum.stance_gate_v2 import STANCE_V2_THRESHOLD_KEYS

        shared = {
            "min_eval_episodes",
            "settle_steps",
            "min_avg_reward",
            "min_full_horizon_fraction",
            "required_consecutive",
        }
        assert len(set(_STANCE_V2_COPIED_KEYS)) == len(_STANCE_V2_COPIED_KEYS)
        assert set(_STANCE_V2_COPIED_KEYS) == STANCE_V2_THRESHOLD_KEYS - shared
        assert STANCE_V2_THRESHOLD_KEYS <= {spec.name for spec in fields(StageThreshold)}

    def test_an_unpopulated_threshold_cannot_be_field_copied_into_a_gate(self):
        from environments.shared.curriculum import StageThreshold

        with pytest.raises(ValueError, match="never populated"):
            StageThreshold(gate_kind="stance_quality/v2").stance_v2_thresholds()
        # The bar's own default is +inf: no bound clears it.
        assert StageThreshold().min_clean_stance_lcb == float("inf")
