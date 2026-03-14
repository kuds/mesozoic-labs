"""Tests for sweep search_space.py — search space loading and resolution."""

import pytest

from environments.shared.scripts.sweep import (
    _is_per_stage,
    _resolve_search_space,
    _search_space_for_stage,
    _settings_for_stage,
    _split_stage_block,
)

# ── _is_per_stage / _split_stage_block / _search_space_for_stage ─────────


class TestPerStageDetection:
    def test_flat_config(self):
        flat = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}}
        assert _is_per_stage(flat) is False

    def test_per_stage_config(self):
        per_stage = {
            "stage1": {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}},
            "stage2": {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 1e-4}},
        }
        assert _is_per_stage(per_stage) is True

    def test_partial_stage_keys(self):
        """Even having just stage1 makes it per-stage."""
        partial = {"stage1": {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}}}
        assert _is_per_stage(partial) is True


class TestSplitStageBlock:
    def test_separates_search_space_and_settings(self):
        block = {
            "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
            "trials": 30,
            "timesteps": 1000000,
            "parallel": 5,
        }
        search_space, settings = _split_stage_block(block)
        assert "ppo_learning_rate" in search_space
        assert search_space["ppo_learning_rate"]["type"] == "double"
        assert settings == {"trials": 30, "timesteps": 1000000, "parallel": 5}

    def test_all_search_params(self):
        block = {
            "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4},
            "ppo_ent_coef": {"type": "double", "min": 0.001, "max": 0.05},
        }
        search_space, settings = _split_stage_block(block)
        assert len(search_space) == 2
        assert len(settings) == 0

    def test_all_settings(self):
        block = {"trials": 10, "timesteps": 500000}
        search_space, settings = _split_stage_block(block)
        assert len(search_space) == 0
        assert len(settings) == 2

    def test_empty_block(self):
        search_space, settings = _split_stage_block({})
        assert search_space == {}
        assert settings == {}


class TestSearchSpaceForStage:
    def test_flat_returns_as_is(self):
        flat = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}}
        assert _search_space_for_stage(flat, 1) == flat
        assert _search_space_for_stage(flat, 2) == flat

    def test_per_stage_extracts_correct_stage(self):
        per_stage = {
            "stage1": {
                "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4},
                "trials": 20,
            },
            "stage2": {
                "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 1e-4},
            },
            "stage3": {
                "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 5e-5},
            },
        }
        space1 = _search_space_for_stage(per_stage, 1)
        # Should have only the search space param, not the "trials" setting
        assert "ppo_learning_rate" in space1
        assert "trials" not in space1

        space2 = _search_space_for_stage(per_stage, 2)
        assert space2["ppo_learning_rate"]["max"] == 1e-4

    def test_missing_stage_key_exits(self):
        per_stage = {"stage1": {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}}}
        with pytest.raises(SystemExit):
            _search_space_for_stage(per_stage, 3)


class TestSettingsForStage:
    def test_flat_returns_empty(self):
        flat = {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}}
        assert _settings_for_stage(flat, 1) == {}

    def test_per_stage_extracts_settings(self):
        per_stage = {
            "stage1": {
                "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4},
                "trials": 30,
                "timesteps": 1000000,
            },
        }
        settings = _settings_for_stage(per_stage, 1)
        assert settings == {"trials": 30, "timesteps": 1000000}

    def test_missing_stage_returns_empty(self):
        per_stage = {"stage1": {"trials": 10}}
        assert _settings_for_stage(per_stage, 3) == {}


# ── _resolve_search_space ────────────────────────────────────────────────


class TestResolveSearchSpace:
    def test_inline_json_takes_priority(self):
        inline = '{"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 1e-3}}'
        result = _resolve_search_space(inline, None, "ppo")
        assert "ppo_learning_rate" in result
        assert result["ppo_learning_rate"]["max"] == 1e-3

    def test_invalid_json_exits(self):
        with pytest.raises(SystemExit):
            _resolve_search_space("{bad json", None, "ppo")

    def test_default_ppo_space(self):
        result = _resolve_search_space(None, None, "ppo")
        assert "ppo_learning_rate" in result
        assert "ppo_ent_coef" in result
        assert "ppo_batch_size" in result

    def test_default_sac_space(self):
        result = _resolve_search_space(None, None, "sac")
        assert "sac_learning_rate" in result
        assert "sac_batch_size" in result

    def test_unknown_algorithm_exits_with_error(self):
        with pytest.raises(SystemExit):
            _resolve_search_space(None, None, "unknown_algo")


class TestResolveSearchSpaceLogging:
    def test_inline_json_logs_source(self, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            _resolve_search_space('{"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 1e-3}}', None, "ppo")
        assert "inline --search-space JSON" in caplog.text

    def test_default_space_logs_algorithm(self, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            _resolve_search_space(None, None, "sac")
        assert "default sac search space" in caplog.text
