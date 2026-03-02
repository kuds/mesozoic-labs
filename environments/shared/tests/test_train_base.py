"""Tests for shared training infrastructure (train_base.py)."""

import dataclasses

import pytest

from environments.shared.train_base import (
    SpeciesConfig,
    _apply_overrides,
    _cast_value,
    linear_schedule,
)

# ── linear_schedule ──────────────────────────────────────────────────────


class TestLinearSchedule:
    def test_returns_initial_at_start(self):
        sched = linear_schedule(1e-3, 1e-4)
        assert sched(1.0) == pytest.approx(1e-3)

    def test_returns_final_at_end(self):
        sched = linear_schedule(1e-3, 1e-4)
        assert sched(0.0) == pytest.approx(1e-4)

    def test_midpoint(self):
        sched = linear_schedule(1e-3, 1e-4)
        mid = sched(0.5)
        expected = 1e-4 + 0.5 * (1e-3 - 1e-4)
        assert mid == pytest.approx(expected)

    def test_constant_when_initial_equals_final(self):
        sched = linear_schedule(5e-4, 5e-4)
        assert sched(0.0) == pytest.approx(5e-4)
        assert sched(0.5) == pytest.approx(5e-4)
        assert sched(1.0) == pytest.approx(5e-4)


# ── _cast_value ──────────────────────────────────────────────────────────


class TestCastValue:
    def test_int(self):
        assert _cast_value("42") == 42

    def test_float(self):
        assert _cast_value("3.14") == pytest.approx(3.14)

    def test_float_encoded_int(self):
        """Vertex AI HPT sends integers as floats like '128.0'."""
        assert _cast_value("128.0") == 128
        assert isinstance(_cast_value("128.0"), int)

    def test_string(self):
        assert _cast_value("hello") == "hello"

    def test_bool_string(self):
        assert _cast_value("true") == "true"  # Not cast to bool

    def test_negative_float(self):
        assert _cast_value("-0.5") == pytest.approx(-0.5)

    def test_scientific_notation(self):
        assert _cast_value("1e-4") == pytest.approx(1e-4)


# ── _apply_overrides ─────────────────────────────────────────────────────


class TestApplyOverrides:
    @pytest.fixture()
    def configs(self):
        return {
            1: {
                "env_kwargs": {"alive_bonus": 2.0, "forward_vel_weight": 0.0},
                "ppo_kwargs": {"learning_rate": 1e-3, "batch_size": 256},
            },
            2: {
                "env_kwargs": {"alive_bonus": 1.5, "forward_vel_weight": 1.0},
                "ppo_kwargs": {"learning_rate": 5e-4, "batch_size": 128},
            },
        }

    def test_none_overrides_is_noop(self, configs):
        original = {k: dict(v) for k, v in configs.items()}
        _apply_overrides(configs, None)
        for stage in original:
            assert configs[stage]["env_kwargs"] == original[stage]["env_kwargs"]

    def test_empty_overrides_is_noop(self, configs):
        _apply_overrides(configs, [])

    def test_all_stage_override(self, configs):
        _apply_overrides(configs, ["ppo.learning_rate=1e-4"])
        assert configs[1]["ppo_kwargs"]["learning_rate"] == pytest.approx(1e-4)
        assert configs[2]["ppo_kwargs"]["learning_rate"] == pytest.approx(1e-4)

    def test_stage_scoped_override(self, configs):
        _apply_overrides(configs, ["2.ppo.learning_rate=5e-5"])
        assert configs[1]["ppo_kwargs"]["learning_rate"] == pytest.approx(1e-3)
        assert configs[2]["ppo_kwargs"]["learning_rate"] == pytest.approx(5e-5)

    def test_env_section(self, configs):
        _apply_overrides(configs, ["env.alive_bonus=5.0"])
        assert configs[1]["env_kwargs"]["alive_bonus"] == pytest.approx(5.0)
        assert configs[2]["env_kwargs"]["alive_bonus"] == pytest.approx(5.0)

    def test_float_encoded_int(self, configs):
        _apply_overrides(configs, ["ppo.batch_size=512.0"])
        assert configs[1]["ppo_kwargs"]["batch_size"] == 512
        assert isinstance(configs[1]["ppo_kwargs"]["batch_size"], int)


# ── SpeciesConfig ────────────────────────────────────────────────────────


class TestSpeciesConfig:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(SpeciesConfig)

    def test_required_fields(self):
        fields = {f.name for f in dataclasses.fields(SpeciesConfig)}
        expected = {
            "species",
            "env_class",
            "stage_descriptions",
            "height_label",
            "stage3_section_label",
            "success_keys",
        }
        assert expected == fields

    def test_from_velociraptor(self):
        from environments.velociraptor.envs.raptor_env import RaptorEnv

        cfg = SpeciesConfig(
            species="velociraptor",
            env_class=RaptorEnv,
            stage_descriptions="1=balance, 2=locomotion, 3=strike",
            height_label="Pelvis height",
            stage3_section_label="Hunting",
            success_keys=["strike_success"],
        )
        assert cfg.species == "velociraptor"
        assert cfg.env_class is RaptorEnv

    def test_from_trex(self):
        from environments.trex.envs.trex_env import TRexEnv

        cfg = SpeciesConfig(
            species="trex",
            env_class=TRexEnv,
            stage_descriptions="1=balance, 2=locomotion, 3=bite",
            height_label="Pelvis height",
            stage3_section_label="Hunting",
            success_keys=["bite_success"],
        )
        assert cfg.species == "trex"

    def test_from_brachiosaurus(self):
        from environments.brachiosaurus.envs.brachio_env import BrachioEnv

        cfg = SpeciesConfig(
            species="brachiosaurus",
            env_class=BrachioEnv,
            stage_descriptions="1=balance, 2=locomotion, 3=food_reach",
            height_label="Torso height",
            stage3_section_label="Food Reaching",
            success_keys=["food_reached"],
        )
        assert cfg.species == "brachiosaurus"
