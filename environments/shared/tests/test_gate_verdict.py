"""Tests for environments.shared.result_bundle.gate_verdict.

The verdict is what lets a later run reuse a node instead of training it
(BEHAVIOR_RECIPES_PLAN §4.2): it lives at the stage-directory root, it is
hash-bound to the handoff pair it judged, and its absence is never a pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from environments.shared.curriculum.gate_schema import gate_config_view
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
from environments.shared.result_bundle.hashing import gate_config_sha256

#: The block every verdict here is judged under (a stance stage's), and the
#: view the writer records as ``gate`` (decision D-A22).
STANCE_CURRICULUM = {
    "gate_kind": "stance_quality/v1",
    "gate_schema_version": 1,
    "timesteps": 11_000_000,
    "min_full_horizon_fraction": 0.95,
    "max_unsupported_duty": 0.02,
    "max_unsupported_duty_ucb": 0.02,
    "settle_steps": 200,
    "min_eval_episodes": 40,
    "min_avg_reward": 2100.0,
    "required_consecutive": 3,
    "collapse_patience": 10,
    "warmup_timesteps": 100_000,
}
GATE_CONFIG = gate_config_view(STANCE_CURRICULUM)


def _stage_dir(tmp_path: Path) -> Path:
    stage_dir = tmp_path / "run" / "01_stance"
    (stage_dir / "models").mkdir(parents=True)
    (stage_dir / "models" / "robust_best_model.zip").write_bytes(b"policy")
    (stage_dir / "models" / "robust_best_model_vecnorm.pkl").write_bytes(b"stats")
    return stage_dir


def _write_kwargs(stage_dir: Path) -> dict[str, Any]:
    return dict(
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
        gate_config=GATE_CONFIG,
    )


def _write(stage_dir: Path, **overrides: Any):
    kwargs = _write_kwargs(stage_dir)
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
        # D-A22: the gate the verdict was judged under travels with it — the
        # kind, the schema version and ONLY the thresholds the kind consumes.
        assert verdict["gate"] == GATE_CONFIG
        assert verdict["gate"] == {
            "gate_kind": "stance_quality/v1",
            "gate_schema_version": 1,
            "thresholds": {
                "max_unsupported_duty": 0.02,
                "max_unsupported_duty_ucb": 0.02,
                "min_avg_reward": 2100.0,
                "min_eval_episodes": 40,
                "min_full_horizon_fraction": 0.95,
                "required_consecutive": 3,
                "settle_steps": 200,
            },
        }
        assert verdict["gate_sha256"] == gate_config_sha256(GATE_CONFIG)
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
            "gate",
            "gate_sha256",
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

    def test_gate_sha256_is_the_digest_of_the_gate_config_view(self, tmp_path):
        """D-A22 / D-B7: thresholds only, numerics normalised, key order irrelevant."""
        stage_dir = _stage_dir(tmp_path)
        digest = json.loads(_write(stage_dir).read_text())["gate_sha256"]
        assert digest == gate_config_sha256(GATE_CONFIG)
        assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64

        def digest_of(block):
            return json.loads(_write(stage_dir, gate_config=gate_config_view(block)).read_text())["gate_sha256"]

        # A cosmetic TOML retype (100 vs 100.0) is the same gate.
        assert digest_of({**STANCE_CURRICULUM, "min_avg_reward": 2100}) == digest
        assert digest_of({**STANCE_CURRICULUM, "min_eval_episodes": 40.0}) == digest
        # Key order is irrelevant.
        assert digest_of(dict(reversed(list(STANCE_CURRICULUM.items())))) == digest
        # Schedule, collapse and shaping keys are not the gate.
        assert digest_of({**STANCE_CURRICULUM, "timesteps": 1}) == digest
        assert digest_of({**STANCE_CURRICULUM, "collapse_patience": 99, "collapse_min_evals": 3}) == digest
        assert digest_of({**STANCE_CURRICULUM, "warmup_timesteps": 5, "warmup_clip_range": 0.5}) == digest
        assert digest_of({k: v for k, v in STANCE_CURRICULUM.items() if k != "timesteps"}) == digest
        # A threshold edit, a dropped threshold, another kind or version: a different gate.
        assert digest_of({**STANCE_CURRICULUM, "max_unsupported_duty_ucb": 0.05}) != digest
        assert digest_of({k: v for k, v in STANCE_CURRICULUM.items() if k != "min_avg_reward"}) != digest
        assert digest_of({**STANCE_CURRICULUM, "gate_schema_version": 2}) != digest
        assert digest_of({**STANCE_CURRICULUM, "gate_kind": "recovery_quality/v1"}) != digest
        # A bool threshold value is not a numeric and is hashed verbatim.
        assert gate_config_sha256({"gate_kind": "x", "gate_schema_version": 1, "thresholds": {"a": True}}) != (
            gate_config_sha256({"gate_kind": "x", "gate_schema_version": 1, "thresholds": {"a": 1}})
        )
        # A numpy scalar threshold (a sweep override, a value read back from
        # evaluations.npz) is neither an int nor a float subclass: it digests
        # as its float, the file records its item, and the digest recorded is
        # the digest of the block on disk — so the reader accepts the file.
        numpy_typed = gate_config_view(
            {**STANCE_CURRICULUM, "min_avg_reward": np.float32(2100.0), "min_eval_episodes": np.int64(40)}
        )
        assert gate_config_sha256(numpy_typed) == digest
        assert (
            digest_of({**STANCE_CURRICULUM, "min_avg_reward": np.float32(2100.0), "min_eval_episodes": np.int64(40)})
            == digest
        )
        written = read_gate_verdict(stage_dir)
        assert written["gate"]["thresholds"]["min_avg_reward"] == 2100.0
        assert written["gate"]["thresholds"]["min_eval_episodes"] == 40
        assert written["gate_sha256"] == gate_config_sha256(written["gate"]) == digest

    def test_a_malformed_gate_config_is_refused_before_anything_is_written(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        for bad in (
            {"gate_kind": "stance_quality/v1", "gate_schema_version": 1},  # no thresholds
            {**GATE_CONFIG, "timesteps": 1},  # a key the view never carries
            {**GATE_CONFIG, "thresholds": [0.02]},  # thresholds not a mapping
            STANCE_CURRICULUM,  # the raw block, not its view
            None,
            "sha256:" + "a" * 64,
        ):
            with pytest.raises(
                GateVerdictError, match="gate_config.*must be the curriculum.gate_schema.gate_config_view"
            ):
                _write(stage_dir, gate_config=bad)
        # The keyword is REQUIRED: a writer that forgets it cannot write at all.
        kwargs = {k: v for k, v in _write_kwargs(stage_dir).items() if k != "gate_config"}
        with pytest.raises(TypeError, match="gate_config"):
            write_gate_verdict(stage_dir, **kwargs)
        assert read_gate_verdict(stage_dir) is None

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

    def test_malformed_gate_sha256_is_an_error(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        path = _write(stage_dir)
        good = json.loads(path.read_text())
        for bad in ("abc", "sha256:" + "A" * 64, "sha256:" + "a" * 63, 5):
            path.write_text(json.dumps({**good, "gate_sha256": bad}))
            with pytest.raises(GateVerdictError, match="gate_sha256 must be sha256"):
                read_gate_verdict(stage_dir)
        for bad_gate in ([], "stance_quality/v1", 1):
            path.write_text(json.dumps({**good, "gate": bad_gate}))
            with pytest.raises(GateVerdictError, match="'gate' must be a JSON object"):
                read_gate_verdict(stage_dir)
        # D-B8: the two fields must agree — a gate block edited after judging
        # no longer digests to the recorded gate_sha256, and the file is refused.
        edited = {**good, "gate": {**good["gate"], "thresholds": {**good["gate"]["thresholds"], "min_avg_reward": 1.0}}}
        path.write_text(json.dumps(edited))
        with pytest.raises(GateVerdictError, match="not the digest of the recorded gate block"):
            read_gate_verdict(stage_dir)
        path.write_text(json.dumps({**good, "gate_sha256": "sha256:" + "0" * 64}))
        with pytest.raises(GateVerdictError, match="not the digest of the recorded gate block"):
            read_gate_verdict(stage_dir)
        # A retyped threshold in the block still digests to the same gate.
        retyped = {
            **good,
            "gate": {**good["gate"], "thresholds": {**good["gate"]["thresholds"], "min_avg_reward": 2100}},
        }
        path.write_text(json.dumps(retyped))
        assert read_gate_verdict(stage_dir) == retyped

    def test_a_phase_a_verdict_without_gate_sha256_still_reads(self, tmp_path):
        """The schema is unchanged: a file judged before D-A22 reads (and reuse refuses it by rule 7)."""
        stage_dir = _stage_dir(tmp_path)
        path = _write(stage_dir)
        phase_a = {k: v for k, v in json.loads(path.read_text()).items() if k not in {"gate", "gate_sha256"}}
        path.write_text(json.dumps(phase_a))
        assert read_gate_verdict(stage_dir) == phase_a
        assert verdict_is_reusable(read_gate_verdict(stage_dir))
        # Either field alone is tolerated too (nullable, never cross-checked against nothing).
        path.write_text(json.dumps({**phase_a, "gate_sha256": None, "gate": None}))
        assert read_gate_verdict(stage_dir)["gate_sha256"] is None
        path.write_text(json.dumps({**phase_a, "gate_sha256": "sha256:" + "0" * 64}))
        assert read_gate_verdict(stage_dir)["gate_sha256"] == "sha256:" + "0" * 64
