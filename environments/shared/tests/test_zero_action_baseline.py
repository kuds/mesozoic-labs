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


def _stub_stage(monkeypatch, lengths, gates):
    """Stub ``preflight``'s environment (episode lengths per species), stage-1 gate and plant identity."""
    import environments.shared.scripts.zero_action_baseline as baseline_module

    class _PlantIdentity:
        @staticmethod
        def to_dict():
            return {"plant_revision": 1}

    monkeypatch.setattr(baseline_module, "build_env", lambda species, stage: _StubEnv(lengths[species]))
    monkeypatch.setattr(
        baseline_module,
        "load_stage_config",
        lambda species, stage: {"curriculum_kwargs": {"min_avg_reward": gates[species]}, "env_kwargs": {}},
    )
    monkeypatch.setattr(baseline_module, "current_plant_identity", lambda species: _PlantIdentity())
    return baseline_module


def _exec_preflight_cell(tmp_path, monkeypatch, run_dir):
    """Execute the production notebook cell (a call to ``preflight``, consolidation PR-14b) for a statue that
    never stands, with the stage-1 config stubbed."""
    _stub_stage(monkeypatch, {"brachiosaurus": [100] * 40}, {"brachiosaurus": 100.0})

    notebook_path = Path(__file__).resolve().parents[3] / "notebooks" / "sb3_training.ipynb"
    notebook = json.loads(notebook_path.read_text())
    preflight_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "Pre-flight: zero-action baseline" in "".join(cell.get("source", []))
    ]
    assert len(preflight_cells) == 1

    namespace = {"SPECIES": "brachiosaurus", "LOG_BASE": tmp_path / "logs", "RUN_DIR": run_dir}
    exec(compile(preflight_cells[0], str(notebook_path), "exec"), namespace)
    return next((tmp_path / "logs" / "brachiosaurus" / "zero_action_baselines").glob("*.json"))


def test_notebook_preflight_writes_null_margin(tmp_path, monkeypatch):
    """Execute the production notebook cell for a statue that never stands."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    saved_path = _exec_preflight_cell(tmp_path, monkeypatch, run_dir)
    saved_text = saved_path.read_text()
    saved_result = json.loads(saved_text)["results"]["brachiosaurus"]

    assert "NaN" not in saved_text
    assert saved_text == json.dumps(json.loads(saved_text), indent=2, sort_keys=True, allow_nan=False)
    assert json.loads(saved_text)["schema"] == "mesozoic.zero-action-baseline/v1"
    assert saved_result["verdict"] == "FAILS — a statue clears this gate"  # the gate (100) is the statue's mean
    assert saved_result["reward_mean_standing"] is None
    assert saved_result["reward_std_standing"] is None
    assert saved_result["margin_over_standing"] is None
    assert json.loads((run_dir / "zero_action_baseline.json").read_text()) == json.loads(saved_text)


@pytest.mark.parametrize("status", ["partial", "failed", "complete"])
def test_notebook_preflight_leaves_a_complete_bundle_untouched(tmp_path, monkeypatch, status):
    """A complete bundle is immutable (consolidation PR-14a): the run keeps the copy it was sealed with, and the
    per-species record outside the run is still written. A ``partial`` or ``failed`` bundle is rebuilt by the
    next node trained or judged in it, so its copy is refreshed as before."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "artifact_manifest.json").write_text(json.dumps({"status": status}))
    sealed = run_dir / "zero_action_baseline.json"
    sealed.write_text('{"captured_at": "sealed"}')
    before = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}

    saved_path = _exec_preflight_cell(tmp_path, monkeypatch, run_dir)

    assert json.loads(saved_path.read_text())["results"]["brachiosaurus"]["verdict"]
    if status == "complete":
        assert {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()} == before
    else:
        assert json.loads(sealed.read_text()) == json.loads(saved_path.read_text())


#: The table ``preflight`` prints and saves for the two-species case below (the pre-PR-14b cell's, byte for byte).
_TWO_SPECIES_TABLE = """\
zero-action baseline — stage 1, 4 episodes, seed 7

species                           reward  mean-std  standing  full-hz     gate  verdict
---------------------------------------------------------------------------------------
Tyrannosaurus Rex                  850.0     590.2    1000.0     75%     5000  OK
Velociraptor Mongoliensis          150.0     100.0         —      0%        —  NO GATE

A trained stage-1 policy must beat 'reward', 'mean-std' AND 'full-hz' to have
learned to balance at all, and 'standing' to have learned more than 'do not fall'."""


@pytest.mark.parametrize("with_log_base", [True, False])
def test_preflight_table_records_and_run_copy(tmp_path, monkeypatch, capsys, with_log_base):
    """Every species gets its own record, the table goes beside the trained species' record, and only that
    species' record is copied into the run; without ``LOG_BASE`` the table is printed and nothing is written."""
    baseline_module = _stub_stage(
        monkeypatch,
        {"trex": [1000, 1000, 1000, 400], "velociraptor": [100, 200, 100, 200]},
        {"trex": 5000.0, "velociraptor": None},
    )

    class _FrozenClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 24, 12, 0, 0)

    monkeypatch.setattr(baseline_module, "datetime", _FrozenClock)
    log_base, run_dir = tmp_path / "logs", tmp_path / "run"
    run_dir.mkdir()

    baseline_module.preflight(
        ["trex", "velociraptor"],
        stage=1,
        episodes=4,
        seed=7,
        species="velociraptor",
        log_base=log_base if with_log_base else None,
        run_dir=run_dir,
    )

    out = capsys.readouterr().out
    assert out.startswith(_TWO_SPECIES_TABLE + "\n")
    if not with_log_base:
        assert out.endswith("LOG_BASE not defined — run the storage-configuration cell to save results.\n")
        assert not log_base.exists() and not any(run_dir.iterdir())
        return
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*.*")) == [
        "logs/trex/zero_action_baselines/20260924_120000.json",
        "logs/velociraptor/zero_action_baselines/20260924_120000.json",
        "logs/velociraptor/zero_action_baselines/20260924_120000.txt",
        "run/zero_action_baseline.json",
    ]
    assert (log_base / "velociraptor/zero_action_baselines/20260924_120000.txt").read_text() == (
        _TWO_SPECIES_TABLE + "\n"
    )
    for name, verdict in (("trex", "OK"), ("velociraptor", "NO GATE")):
        record = json.loads((log_base / name / "zero_action_baselines/20260924_120000.json").read_text())
        assert record["captured_at"] == "2026-09-24T12:00:00"
        assert list(record["results"]) == [name] and record["results"][name]["verdict"] == verdict
    run_copy = (run_dir / "zero_action_baseline.json").read_text()
    assert run_copy == (log_base / "velociraptor/zero_action_baselines/20260924_120000.json").read_text()
