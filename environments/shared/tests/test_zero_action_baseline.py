"""Tests for the zero-action baseline diagnostic and notebook record values."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from environments.shared.scripts.zero_action_baseline import gate_margin, score


class _StubEnv:
    """Minimal environment implementing the members used by ``score``."""

    def __init__(self, lengths, horizon=1000, reset_noise_scale=0.1):
        self.max_episode_steps = horizon
        self.reset_noise_scale = reset_noise_scale
        self.action_space = type("Box", (), {"shape": (3,)})()
        self._lengths = list(lengths)
        self._step = 0
        self._episode = -1

    def reset(self, seed=None):
        self._episode += 1
        self._step = 0

    def step(self, action):
        self._step += 1
        done = self._step >= self._lengths[self._episode]
        truncated = done and self._lengths[self._episode] >= self.max_episode_steps
        info = {} if truncated else {"termination_reason": "fallen"}
        return None, 1.0, done and not truncated, truncated, info

    def close(self):
        pass


def test_no_standing_episodes_produce_strict_json_record():
    """The notebook must write JSON nulls without subtracting from them."""
    result = score(_StubEnv([100, 250, 400]), episodes=3, seed=0)
    record = {
        **result,
        "min_avg_reward": 100.0,
        "margin_over_standing": gate_margin(100.0, result["reward_mean_standing"]),
    }

    assert result["n_standing"] == 0
    assert result["reward_mean_standing"] is None
    assert result["reward_std_standing"] is None
    assert record["margin_over_standing"] is None
    assert json.loads(json.dumps(record, allow_nan=False))["reward_mean_standing"] is None


def test_standing_episodes_produce_numeric_margin():
    result = score(_StubEnv([100, 1000, 1000]), episodes=3, seed=0)

    assert result["n_standing"] == 2
    assert result["reward_mean_standing"] == pytest.approx(1000.0)
    assert result["reward_std_standing"] == pytest.approx(0.0)
    assert result["full_horizon_share"] == pytest.approx(2 / 3)
    assert gate_margin(1200.0, result["reward_mean_standing"]) == pytest.approx(200.0)


def test_missing_gate_has_no_margin():
    assert gate_margin(None, 1000.0) is None


def test_score_contains_only_json_native_scalars():
    result = score(_StubEnv([100, 1000]), episodes=2, seed=0)

    for key, value in result.items():
        assert not isinstance(value, np.generic), f"{key} is a numpy scalar"


def test_notebook_preflight_writes_null_margin(tmp_path, monkeypatch):
    """Execute the production notebook cell for a statue that never stands."""
    import environments.shared.config as config_module
    import environments.shared.plant_contract as plant_contract_module
    import environments.shared.scripts.zero_action_baseline as baseline_module

    class _PlantIdentity:
        @staticmethod
        def to_dict():
            return {"plant_revision": 1}

    monkeypatch.setattr(
        baseline_module,
        "build_env",
        lambda species, stage: _StubEnv([100] * 40),
    )
    monkeypatch.setattr(
        config_module,
        "load_stage_config",
        lambda species, stage: {
            "curriculum_kwargs": {"min_avg_reward": 100.0},
            "env_kwargs": {},
        },
    )
    monkeypatch.setattr(
        plant_contract_module,
        "current_plant_identity",
        lambda species: _PlantIdentity(),
    )

    notebook_path = Path(__file__).resolve().parents[3] / "notebooks" / "sb3_training.ipynb"
    notebook = json.loads(notebook_path.read_text())
    preflight_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "Pre-flight: zero-action baseline" in "".join(cell.get("source", []))
    ]
    assert len(preflight_cells) == 1

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    namespace = {
        "SPECIES": "brachiosaurus",
        "LOG_BASE": tmp_path / "logs",
        "RUN_DIR": run_dir,
        "Path": Path,
        "datetime": datetime,
    }
    exec(compile(preflight_cells[0], str(notebook_path), "exec"), namespace)

    saved_path = next((tmp_path / "logs" / "brachiosaurus" / "zero_action_baselines").glob("*.json"))
    saved_text = saved_path.read_text()
    saved_result = json.loads(saved_text)["results"]["brachiosaurus"]

    assert "NaN" not in saved_text
    assert saved_result["reward_mean_standing"] is None
    assert saved_result["reward_std_standing"] is None
    assert saved_result["margin_over_standing"] is None
    assert json.loads((run_dir / "zero_action_baseline.json").read_text()) == json.loads(saved_text)
