"""Tests for environments.shared.task_fingerprint.

The fingerprint's job is to make "same task" a checkable claim: a resumed
checkpoint must provably continue the task it was trained on, and a
warm-start across a task boundary must leave a lineage record instead of
silence.
"""

from __future__ import annotations

import json

import pytest

from environments.shared.task_fingerprint import (
    MODEL_TASK_ATTRIBUTE,
    MODEL_TASK_LINEAGE_ATTRIBUTE,
    TaskFingerprintError,
    attach_task_fingerprint,
    attach_task_lineage,
    compute_task_fingerprint,
    derive_stage_task_fingerprint,
    validate_model_task,
    validate_recorded_task,
    write_task_fingerprint,
)

_PLANT = {
    "physics_sha256": "sha256:aaaa",
    "policy_interface_sha256": "sha256:bbbb",
    "model_path": "environments/trex/assets/trex.xml",
}
_ENV = {"alive_bonus": 1.0, "healthy_z_range": (0.70, 1.30), "max_episode_steps": 1000}


def _fingerprint(**overrides):
    kwargs = dict(
        species="trex",
        stage=1,
        backend="stable-baselines3",
        env_kwargs=_ENV,
        plant_identity=_PLANT,
        perturbation_manifest=None,
    )
    kwargs.update(overrides)
    return compute_task_fingerprint(**kwargs)


class TestCompute:
    def test_deterministic_and_key_order_independent(self):
        a = _fingerprint()
        b = _fingerprint(env_kwargs=dict(reversed(list(_ENV.items()))))
        assert a["task_sha256"] == b["task_sha256"]

    def test_every_task_defining_axis_moves_the_hash(self):
        base = _fingerprint()["task_sha256"]
        assert _fingerprint(stage=2)["task_sha256"] != base
        assert _fingerprint(backend="jax-mjx")["task_sha256"] != base
        assert _fingerprint(env_kwargs={**_ENV, "alive_bonus": 0.5})["task_sha256"] != base
        assert _fingerprint(plant_identity={**_PLANT, "physics_sha256": "sha256:cccc"})["task_sha256"] != base
        assert _fingerprint(perturbation_manifest={"force_n": 165.5, "interval_s": 2.0})["task_sha256"] != base

    def test_perturbation_block_records_the_schedule_implementation(self):
        payload = _fingerprint(perturbation_manifest={"force_n": 165.5})
        assert "push_schedule/v1" in payload["perturbation"]["schedule_implementation"]

    def test_sidecar_round_trips(self, tmp_path):
        payload = _fingerprint()
        path = write_task_fingerprint(tmp_path / "task_fingerprint.json", payload)
        assert json.loads(path.read_text()) == payload


class TestResumeSameStage:
    def test_matching_fingerprint_resumes_cleanly(self):
        current = _fingerprint()
        assert validate_recorded_task(current, current, mode="resume_same_stage") is None

    def test_mismatch_is_fatal_and_names_the_sections(self):
        recorded = _fingerprint()
        current = _fingerprint(perturbation_manifest={"force_n": 165.5})
        with pytest.raises(TaskFingerprintError, match="perturbation"):
            validate_recorded_task(recorded, current, mode="resume_same_stage")

    def test_missing_fingerprint_fails_closed_by_default(self):
        with pytest.raises(TaskFingerprintError, match="no task fingerprint"):
            validate_recorded_task(None, _fingerprint(), mode="resume_same_stage")

    def test_missing_fingerprint_warns_through_the_transition_valve(self, caplog):
        with caplog.at_level("WARNING"):
            result = validate_recorded_task(None, _fingerprint(), mode="resume_same_stage", allow_unfingerprinted=True)
        assert result is None
        assert "allow_unfingerprinted" in caplog.text

    def test_unknown_mode_is_fatal(self):
        with pytest.raises(TaskFingerprintError, match="unknown load mode"):
            validate_recorded_task(None, _fingerprint(), mode="warm_start")


class TestInitializeNextStage:
    def test_boundary_crossing_returns_lineage(self):
        parent = _fingerprint(stage=1)
        child = _fingerprint(stage=2, perturbation_manifest={"force_n": 165.5})
        lineage = validate_recorded_task(parent, child, mode="initialize_next_stage")
        assert lineage == {
            "mode": "initialize_next_stage",
            "parent_task_sha256": parent["task_sha256"],
            "child_task_sha256": child["task_sha256"],
            "parent_species": "trex",
            "parent_stage": 1,
        }

    def test_unfingerprinted_parent_records_parentless_lineage(self, caplog):
        child = _fingerprint(stage=2)
        with caplog.at_level("WARNING"):
            lineage = validate_recorded_task(None, child, mode="initialize_next_stage")
        assert lineage["parent_task_sha256"] is None
        assert lineage["child_task_sha256"] == child["task_sha256"]


class TestModelAttachment:
    def test_attach_validate_round_trip(self):
        class Model:
            pass

        model = Model()
        current = _fingerprint()
        attach_task_fingerprint(model, current)
        assert getattr(model, MODEL_TASK_ATTRIBUTE) == current
        assert validate_model_task(model, current, mode="resume_same_stage") is None

    def test_lineage_travels_on_the_child(self):
        class Model:
            pass

        model = Model()
        attach_task_lineage(model, {"mode": "initialize_next_stage", "parent_task_sha256": "sha256:x"})
        assert getattr(model, MODEL_TASK_LINEAGE_ATTRIBUTE)["parent_task_sha256"] == "sha256:x"

    def test_invalid_metadata_type_is_fatal(self):
        class Model:
            pass

        model = Model()
        setattr(model, MODEL_TASK_ATTRIBUTE, "not-a-mapping")
        with pytest.raises(TaskFingerprintError, match="invalid task fingerprint"):
            validate_model_task(model, _fingerprint(), mode="resume_same_stage")


class TestDeriveStageFingerprint:
    def test_push_free_stage_has_no_perturbation_block(self):
        payload = derive_stage_task_fingerprint(
            species="trex",
            stage=1,
            backend="stable-baselines3",
            env_kwargs=_ENV,
            plant_identity=_PLANT,
        )
        assert payload["perturbation"] is None

    def test_pushed_stage_derives_the_real_force_constants(self):
        pytest.importorskip("mujoco")
        payload = derive_stage_task_fingerprint(
            species="trex",
            stage=1,
            backend="stable-baselines3",
            env_kwargs={**_ENV, "perturbation_capture_velocity_multiple": 1.5},
            plant_identity=_PLANT,
        )
        # Same derivation as BaseDinoEnv/MJXDinoEnv: the r7 trex constants.
        assert payload["perturbation"]["force_n"] == pytest.approx(165.5, abs=2.0)
        assert payload["perturbation"]["interval_s"] == 2.0
        push_free = derive_stage_task_fingerprint(
            species="trex",
            stage=1,
            backend="stable-baselines3",
            env_kwargs=_ENV,
            plant_identity=_PLANT,
        )
        assert payload["task_sha256"] != push_free["task_sha256"]

    def test_pushed_stage_without_model_path_is_fatal(self):
        with pytest.raises(TaskFingerprintError, match="model_path"):
            derive_stage_task_fingerprint(
                species="trex",
                stage=1,
                backend="stable-baselines3",
                env_kwargs={"perturbation_capture_velocity_multiple": 1.5},
                plant_identity={"physics_sha256": "sha256:aaaa"},
            )

    def test_pushed_stage_resolves_model_path_from_any_cwd(self, tmp_path, monkeypatch):
        """The identity's repo-relative model_path must not depend on cwd.

        Regression: Colab adds the clone to sys.path without chdir-ing into
        it, so the first pushed-stage fingerprint (the notebook's recovery
        cell) crashed on ParseXML before the stage directory existed —
        run-all halted at the stance/recovery boundary with no artifact to
        show why, twice.
        """
        pytest.importorskip("mujoco")
        kwargs = dict(
            species="trex",
            stage="recovery",
            backend="stable-baselines3",
            env_kwargs={**_ENV, "perturbation_capture_velocity_multiple": 1.5},
            plant_identity=_PLANT,
        )
        at_root = derive_stage_task_fingerprint(**kwargs)
        monkeypatch.chdir(tmp_path)
        elsewhere = derive_stage_task_fingerprint(**kwargs)
        assert elsewhere["task_sha256"] == at_root["task_sha256"]

    def test_pushed_stage_names_a_missing_model_clearly(self):
        with pytest.raises(TaskFingerprintError, match="cannot find the plant model"):
            derive_stage_task_fingerprint(
                species="trex",
                stage="recovery",
                backend="stable-baselines3",
                env_kwargs={"perturbation_capture_velocity_multiple": 1.5},
                plant_identity={"model_path": "environments/trex/assets/no_such_model.xml"},
            )


class TestStageConfigWiring:
    def test_save_stage_config_writes_the_sidecar(self, tmp_path):
        from environments.shared.config import save_stage_config

        payload = _fingerprint()
        save_stage_config(
            tmp_path,
            1,
            {"name": "balance", "description": "", "env_kwargs": dict(_ENV)},
            "PPO",
            task_fingerprint=payload,
        )
        sidecar = json.loads((tmp_path / "task_fingerprint.json").read_text())
        assert sidecar == payload
        embedded = json.loads((tmp_path / "stage_config.json").read_text())["task_fingerprint"]
        assert embedded["task_sha256"] == payload["task_sha256"]
