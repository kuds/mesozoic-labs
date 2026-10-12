"""Tests for the zero-action baseline diagnostic and notebook record values."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from environments.shared.model_id import ModelId
from environments.shared.scripts.zero_action_baseline import gate_margin, score

from .notebook_cells import code_cell

VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")


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

    monkeypatch.setattr(baseline_module, "build_env", lambda model_id, stage: _StubEnv(lengths[model_id]))
    monkeypatch.setattr(
        baseline_module,
        "load_stage_config",
        lambda model_id, stage: {"curriculum_kwargs": {"min_avg_reward": gates[model_id]}, "env_kwargs": {}},
    )
    monkeypatch.setattr(baseline_module, "current_plant_identity", lambda model_id: _PlantIdentity())
    return baseline_module


def _exec_preflight_cell(tmp_path, monkeypatch, run_dir):
    """Execute the production notebook cell (a call to ``preflight``, consolidation PR-14b) for a statue that
    never stands, with the stage-1 config stubbed."""
    _stub_stage(monkeypatch, {BRACHIOSAURUS_ALTITHORAX: [100] * 40}, {BRACHIOSAURUS_ALTITHORAX: 100.0})

    notebook_path = Path(__file__).resolve().parents[3] / "notebooks" / "sb3_training.ipynb"
    preflight_cell = code_cell(notebook_path, "Pre-flight: zero-action baseline")

    namespace = {
        "SPECIES": "brachiosaurus_altithorax",
        "EMBODIMENT": "anatomical",
        "MODEL_ID": BRACHIOSAURUS_ALTITHORAX,
        "LOG_BASE": tmp_path / "logs",
        "RUN_DIR": run_dir,
    }
    exec(compile(preflight_cell, str(notebook_path), "exec"), namespace)
    return next(
        (tmp_path / "logs" / "brachiosaurus_altithorax" / "anatomical" / "zero_action_baselines").glob("*.json")
    )


def test_notebook_preflight_writes_null_margin(tmp_path, monkeypatch):
    """Execute the production notebook cell for a statue that never stands."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    saved_path = _exec_preflight_cell(tmp_path, monkeypatch, run_dir)
    saved_text = saved_path.read_text()
    saved_result = json.loads(saved_text)["results"]["brachiosaurus_altithorax"]["anatomical"]

    assert "NaN" not in saved_text
    assert saved_text == json.dumps(json.loads(saved_text), indent=2, sort_keys=True, allow_nan=False)
    # Re-pinned: the record keys results by species, then embodiment (decision D-D33).
    assert json.loads(saved_text)["schema"] == "mesozoic.zero-action-baseline/v2"
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

    assert json.loads(saved_path.read_text())["results"]["brachiosaurus_altithorax"]["anatomical"]["verdict"]
    if status == "complete":
        assert {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()} == before
    else:
        assert json.loads(sealed.read_text()) == json.loads(saved_path.read_text())


#: The table ``preflight`` prints and saves for the two-model case below (the pre-PR-14b cell's, its first column the
#: model display name since decision D-D33).
_TWO_MODEL_TABLE = """\
zero-action baseline — stage 1, 4 episodes, seed 7

model                                         reward  mean-std  standing  full-hz     gate  verdict
---------------------------------------------------------------------------------------------------
Tyrannosaurus Rex (Anatomical)                 850.0     590.2    1000.0     75%     5000  OK
Velociraptor Mongoliensis (Anatomical)         150.0     100.0         —      0%        —  NO GATE

A trained stage-1 policy must beat 'reward', 'mean-std' AND 'full-hz' to have
learned to balance at all, and 'standing' to have learned more than 'do not fall'."""


@pytest.mark.parametrize("with_log_base", [True, False])
def test_preflight_table_records_and_run_copy(tmp_path, monkeypatch, capsys, with_log_base):
    """Every model gets its own record, the table goes beside the trained model's record, and only that
    model's record is copied into the run; without ``LOG_BASE`` the table is printed and nothing is written."""
    baseline_module = _stub_stage(
        monkeypatch,
        {TYRANNOSAURUS_REX: [1000, 1000, 1000, 400], VELOCIRAPTOR_MONGOLIENSIS: [100, 200, 100, 200]},
        {TYRANNOSAURUS_REX: 5000.0, VELOCIRAPTOR_MONGOLIENSIS: None},
    )

    class _FrozenClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 24, 12, 0, 0)

    monkeypatch.setattr(baseline_module, "datetime", _FrozenClock)
    log_base, run_dir = tmp_path / "logs", tmp_path / "run"
    run_dir.mkdir()

    baseline_module.preflight(
        [TYRANNOSAURUS_REX, VELOCIRAPTOR_MONGOLIENSIS],
        stage=1,
        episodes=4,
        seed=7,
        model_id=VELOCIRAPTOR_MONGOLIENSIS,
        log_base=log_base if with_log_base else None,
        run_dir=run_dir,
    )

    out = capsys.readouterr().out
    assert out.startswith(_TWO_MODEL_TABLE + "\n")
    if not with_log_base:
        assert out.endswith("LOG_BASE not defined — run the storage-configuration cell to save results.\n")
        assert not log_base.exists() and not any(run_dir.iterdir())
        return
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*.*")) == [
        "logs/tyrannosaurus_rex/anatomical/zero_action_baselines/20260924_120000.json",
        "logs/velociraptor_mongoliensis/anatomical/zero_action_baselines/20260924_120000.json",
        "logs/velociraptor_mongoliensis/anatomical/zero_action_baselines/20260924_120000.txt",
        "run/zero_action_baseline.json",
    ]
    assert (
        log_base / "velociraptor_mongoliensis/anatomical/zero_action_baselines/20260924_120000.txt"
    ).read_text() == (_TWO_MODEL_TABLE + "\n")
    for model_id, verdict in ((TYRANNOSAURUS_REX, "OK"), (VELOCIRAPTOR_MONGOLIENSIS, "NO GATE")):
        record = json.loads((model_id.path(log_base) / "zero_action_baselines/20260924_120000.json").read_text())
        assert record["captured_at"] == "2026-09-24T12:00:00"
        assert list(record["results"]) == [model_id.species]
        assert list(record["results"][model_id.species]) == [model_id.embodiment]
        assert record["results"][model_id.species][model_id.embodiment]["verdict"] == verdict
    run_copy = (run_dir / "zero_action_baseline.json").read_text()
    assert (
        run_copy
        == (log_base / "velociraptor_mongoliensis/anatomical/zero_action_baselines/20260924_120000.json").read_text()
    )


@pytest.mark.parametrize("passed", [True, False])
def test_a_stance_gated_stage_is_judged_by_the_stance_gate_on_the_statue(tmp_path, monkeypatch, capsys, passed):
    """KNOWN_ISSUES LOW (2026-09-28): a stance stage used to be judged by its reward RAIL, which the statue
    clears by design, so every stance stage printed "FAILS — a statue clears this gate". The statue is a
    stance gate's quality ceiling and must PASS it; the verdict is the stance report's own, on the same
    seeds, for both stance kinds."""
    import environments.shared.reporting.stance_report as stance_report

    baseline_module = _stub_stage(monkeypatch, {COMPSOGNATHUS_LONGIPES: [1000] * 4}, {COMPSOGNATHUS_LONGIPES: 100.0})
    monkeypatch.setattr(
        baseline_module,
        "load_stage_config",
        lambda model_id, stage: {
            "curriculum_kwargs": {"gate_kind": "stance_quality/v2", "min_avg_reward": 100.0},
            "env_kwargs": {},
        },
    )
    calls = []

    def fake_report(model_id, stage, *, stage_config, zero_action, episodes, seed):
        calls.append((model_id, stage, zero_action, episodes, seed))
        return {
            "gate_kind": "stance_quality/v2",
            "scored_gate_kind": "stance_quality/v2",
            "passed": passed,
            "failures": [] if passed else ["clean_stance_lcb 0.7856 < 0.8000 (36/40 episodes clean)"],
            "result": {"n_clean": 40 if passed else 36, "n_episodes": 40, "clean_lcb": 0.93 if passed else 0.79},
        }

    monkeypatch.setattr(stance_report, "build_stance_gate_report", fake_report)
    baseline_module.preflight(
        [COMPSOGNATHUS_LONGIPES],
        stage=1,
        episodes=4,
        seed=7,
        model_id=COMPSOGNATHUS_LONGIPES,
        log_base=tmp_path / "logs",
        run_dir=None,
    )
    assert calls == [(COMPSOGNATHUS_LONGIPES, 1, True, 4, 7)]
    out = capsys.readouterr().out
    record = json.loads(
        next(
            (tmp_path / "logs" / "compsognathus_longipes" / "anatomical" / "zero_action_baselines").glob("*.json")
        ).read_text()
    )
    result = record["results"]["compsognathus_longipes"]["anatomical"]
    assert result["stance_gate"]["passed"] is passed and result["stance_gate"]["n_episodes"] == 40
    if passed:
        assert result["verdict"] == "OK — the statue passes the stance gate"
    else:
        assert result["verdict"] == "CHECK GATE — the stance gate refuses the statue"
        assert "36/40 episodes clean" in out
    assert "FAILS — a statue clears this gate" not in out


def test_a_heading_certificate_stage_is_preflighted_at_its_zero_heading_cell_only(monkeypatch):
    """Decision 5 of 2026-10-11 (D-D32): the notebook preflight judges the statue at 0 deg -- the same block scored
    by the v2 code the certificate applies at each heading -- and the full six-heading grid runs at certification.
    The plant's heading invariance is the tests' (every plant declares the elliptic friction cone)."""
    import copy

    import environments.shared.reporting.stance_report as stance_report
    from environments.shared.config import load_stage_config
    from environments.shared.scripts.zero_action_baseline import statue_stance_verdict

    from .stance_heading_helpers import HEADING_CURRICULUM

    monkeypatch.setattr(
        stance_report, "build_stance_heading_report", lambda *a, **k: pytest.fail("the preflight rolled the grid")
    )
    config = copy.deepcopy(load_stage_config(COMPSOGNATHUS_LONGIPES, 1))
    config["env_kwargs"]["max_episode_steps"] = 150
    config["curriculum_kwargs"] = {
        **HEADING_CURRICULUM,
        "min_eval_episodes": 2,
        "min_clean_stance_lcb": 0.2,
        "settle_steps": 50,
        "max_sole_corner_lift_m": 0.003,
    }
    record = statue_stance_verdict(COMPSOGNATHUS_LONGIPES, 1, config, episodes=2, seed=3042)
    assert record["gate_kind"] == "stance_quality_heading/v1"
    assert record["scored_gate_kind"] == "stance_quality/v2"
    assert record["headings_deg"] == [0.0]
    assert record["passed"] is True and (record["n_clean"], record["n_episodes"]) == (2, 2)
    # The caller's block is never relabelled in place.
    assert config["curriculum_kwargs"]["gate_kind"] == "stance_quality_heading/v1"
