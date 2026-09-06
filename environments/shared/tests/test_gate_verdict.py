"""Tests for environments.shared.result_bundle.gate_verdict.

The verdict is what lets a later run reuse a node instead of training it
(BEHAVIOR_RECIPES_PLAN §4.2): it lives at the stage-directory root, it is
hash-bound to the handoff pair it judged, and its absence is never a pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from environments.shared.result_bundle import (
    GATE_VERDICT_FILENAME,
    GATE_VERDICT_SCHEMA,
    GateVerdictError,
    ResultBundleError,
    read_gate_verdict,
    sha256_file,
    verdict_is_reusable,
    write_gate_verdict,
)
from environments.shared.result_bundle import gate_verdict as gate_verdict_module


def _stage_dir(tmp_path: Path) -> Path:
    stage_dir = tmp_path / "run" / "01_stance"
    (stage_dir / "models").mkdir(parents=True)
    (stage_dir / "models" / "robust_best_model.zip").write_bytes(b"policy")
    (stage_dir / "models" / "robust_best_model_vecnorm.pkl").write_bytes(b"stats")
    return stage_dir


def _write(stage_dir: Path, **overrides: Any):
    kwargs: dict[str, Any] = dict(
        species="trex",
        stage=1,
        stage_id="stance",
        gate_kind="stance_quality/v1",
        gate_schema_version=1,
        passed=True,
        failures=[],
        task_sha256="sha256:" + "a" * 64,
        judged_by="reporting.stage_artifacts.generate_stage_artifacts",
        checkpoint=stage_dir / "models" / "robust_best_model.zip",
        normalization=stage_dir / "models" / "robust_best_model_vecnorm.pkl",
    )
    kwargs.update(overrides)
    return write_gate_verdict(stage_dir, **kwargs)


class TestWrite:
    def test_round_trips_every_field_and_hashes_the_pair(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        stage_result = {
            "stage": 1,
            "mean_reward": 12.5,
            "gate_passed": True,
            "model_path": str(stage_dir / "models" / "robust_best_model.zip"),
            "vecnorm_path": str(stage_dir / "models" / "robust_best_model_vecnorm.pkl"),
            "per_episode_rewards": [1.0, 2.0],  # not a persisted key: projected away
            "callbacks": object(),  # neither is this — and it is not serialisable
        }
        path = _write(stage_dir, failures=["duty_ucb"], passed=False, stage_result=stage_result)
        assert path == stage_dir / GATE_VERDICT_FILENAME
        verdict = json.loads(path.read_text())
        assert verdict == read_gate_verdict(stage_dir)
        assert verdict["schema"] == GATE_VERDICT_SCHEMA == "mesozoic.gate-verdict/v1"
        assert verdict["species"] == "trex"
        assert verdict["stage"] == 1 and verdict["stage_id"] == "stance"
        assert verdict["gate_kind"] == "stance_quality/v1" and verdict["gate_schema_version"] == 1
        assert verdict["passed"] is False and verdict["failures"] == ["duty_ucb"]
        assert verdict["checkpoint"] == "models/robust_best_model.zip"
        assert verdict["checkpoint_sha256"] == sha256_file(stage_dir / "models" / "robust_best_model.zip")
        assert verdict["normalization"] == "models/robust_best_model_vecnorm.pkl"
        assert verdict["normalization_sha256"] == sha256_file(stage_dir / "models" / "robust_best_model_vecnorm.pkl")
        assert verdict["task_sha256"] == "sha256:" + "a" * 64
        assert verdict["judged_by"] == "reporting.stage_artifacts.generate_stage_artifacts"
        assert verdict["judged_at"].endswith("+00:00")
        assert verdict["stage_result"] == {
            "stage": 1,
            "mean_reward": 12.5,
            "gate_passed": True,
            "model_path": stage_result["model_path"],
            "vecnorm_path": stage_result["vecnorm_path"],
        }
        assert set(verdict) == {
            "schema",
            "species",
            "stage",
            "stage_id",
            "gate_kind",
            "gate_schema_version",
            "passed",
            "failures",
            "checkpoint",
            "checkpoint_sha256",
            "normalization",
            "normalization_sha256",
            "task_sha256",
            "judged_at",
            "judged_by",
            "stage_result",
        }
        # A failed verdict is never reusable, however well hashed.
        assert not verdict_is_reusable(verdict)
        assert verdict_is_reusable(json.loads(_write(stage_dir).read_text()))

    def test_a_semantic_stage_records_its_id_as_the_stage(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        _write(stage_dir, stage="recovery", stage_id="recovery", gate_kind="recovery_quality/v1")
        verdict = read_gate_verdict(stage_dir)
        assert verdict["stage"] == "recovery" and verdict["stage_id"] == "recovery"
        assert verdict["stage_result"] is None

    def test_missing_handoff_records_null_hashes_and_is_not_reusable(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        _write(stage_dir, checkpoint=None, normalization=None, task_sha256=None)
        verdict = read_gate_verdict(stage_dir)
        assert verdict["passed"] is True
        assert verdict["checkpoint"] is None and verdict["checkpoint_sha256"] is None
        assert verdict["normalization"] is None and verdict["normalization_sha256"] is None
        assert verdict["task_sha256"] is None
        assert not verdict_is_reusable(verdict)
        # A hashed checkpoint without its sidecar is not a reusable pair either.
        _write(stage_dir, normalization=None)
        assert not verdict_is_reusable(read_gate_verdict(stage_dir))
        assert not verdict_is_reusable(None)

    def test_a_named_but_absent_handoff_fails_closed(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        with pytest.raises(GateVerdictError, match="missing handoff file"):
            _write(stage_dir, checkpoint=stage_dir / "models" / "nope.zip")
        assert read_gate_verdict(stage_dir) is None

    def test_malformed_inputs_are_refused_before_anything_is_written(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        with pytest.raises(GateVerdictError, match="'passed' must be a bool"):
            _write(stage_dir, passed=1)
        with pytest.raises(GateVerdictError, match="judged_by"):
            _write(stage_dir, judged_by="")
        assert read_gate_verdict(stage_dir) is None
        assert issubclass(GateVerdictError, ResultBundleError)

    def test_is_atomic(self, tmp_path, monkeypatch):
        stage_dir = _stage_dir(tmp_path)
        seen: list[Path] = []
        real = gate_verdict_module.atomic_write_text

        def spy(path, text, **kwargs):
            seen.append(Path(path))
            return real(path, text, **kwargs)

        monkeypatch.setattr(gate_verdict_module, "atomic_write_text", spy)
        _write(stage_dir)
        assert seen == [stage_dir / GATE_VERDICT_FILENAME]
        # A serialisation failure leaves neither the file nor a temp file.
        (stage_dir / GATE_VERDICT_FILENAME).unlink()
        with pytest.raises(TypeError):
            _write(stage_dir, stage_result={"gate_failures": [object()]})
        assert sorted(p.name for p in stage_dir.iterdir()) == ["models"]

    def test_lands_in_the_stage_dir_root_not_models_or_the_staged_tree(self, tmp_path):
        from environments.shared.reporting import stage_layout

        stage_dir = _stage_dir(tmp_path)
        (stage_dir / "replays").mkdir()
        (stage_dir / "replays" / "trex_ppo_stage1_best.mp4").write_bytes(b"v")
        path = _write(stage_dir)
        assert path.parent == stage_dir
        assert path.name == "gate_verdict.json"
        assert "models" not in path.relative_to(stage_dir).parts
        assert path not in set(stage_layout.iter_generated_artifacts(stage_dir))
        assert path not in set(stage_layout.iter_replay_files(stage_dir))

    def test_a_handoff_outside_the_stage_dir_is_recorded_absolute(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        elsewhere = tmp_path / "elsewhere.zip"
        elsewhere.write_bytes(b"x")
        _write(stage_dir, checkpoint=elsewhere)
        assert read_gate_verdict(stage_dir)["checkpoint"] == elsewhere.resolve().as_posix()


class TestRead:
    def test_absent_reads_as_none_never_as_a_pass(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        assert read_gate_verdict(stage_dir) is None
        assert read_gate_verdict(tmp_path / "does_not_exist") is None
        assert not verdict_is_reusable(read_gate_verdict(stage_dir))

    def test_wrong_schema_is_an_error(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        _write(stage_dir)
        path = stage_dir / GATE_VERDICT_FILENAME
        verdict = json.loads(path.read_text())
        verdict["schema"] = "mesozoic.gate-verdict/v2"
        path.write_text(json.dumps(verdict))
        with pytest.raises(GateVerdictError, match="declares schema 'mesozoic.gate-verdict/v2'"):
            read_gate_verdict(stage_dir)
        path.write_text("not json")
        with pytest.raises(GateVerdictError, match="not readable JSON"):
            read_gate_verdict(stage_dir)
        path.write_text("[]")
        with pytest.raises(GateVerdictError, match="JSON object"):
            read_gate_verdict(stage_dir)

    def test_non_bool_passed_is_an_error(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        _write(stage_dir)
        path = stage_dir / GATE_VERDICT_FILENAME
        for passed in (1, "true", None):
            verdict = json.loads(path.read_text())
            verdict["passed"] = passed
            path.write_text(json.dumps(verdict))
            with pytest.raises(GateVerdictError, match="'passed' must be a JSON boolean"):
                read_gate_verdict(stage_dir)
        verdict = json.loads(path.read_text())
        verdict["passed"] = True
        verdict["failures"] = "duty_ucb"
        path.write_text(json.dumps(verdict))
        with pytest.raises(GateVerdictError, match="'failures' must be a JSON list"):
            read_gate_verdict(stage_dir)

    def test_malformed_digest_is_an_error(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        path = _write(stage_dir)
        good = json.loads(path.read_text())
        for key in ("checkpoint_sha256", "normalization_sha256", "task_sha256"):
            for bad in ("abc", "sha256:" + "A" * 64, "sha256:" + "a" * 63, 5):
                path.write_text(json.dumps({**good, key: bad}))
                with pytest.raises(GateVerdictError, match=f"{key} must be sha256"):
                    read_gate_verdict(stage_dir)
        path.write_text(json.dumps(good))
        assert read_gate_verdict(stage_dir) == good
