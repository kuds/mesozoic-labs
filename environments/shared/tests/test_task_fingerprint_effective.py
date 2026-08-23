"""Effective-config fingerprints (schema v2) and the v1 schema valve.

Review F6: schema v1 hashed only the TOML-present ``[env]`` kwargs, so a
retuned constructor default — a real transition-kernel change, e.g. TRexEnv
``healthy_z_range`` — moved no hash and ``resume_same_stage`` would have
crossed it silently.  v2 hashes the effective config (constructor defaults
overlaid with the TOML kwargs).  These tests pin the new identity axis and
the dated valve that keeps v1-fingerprinted checkpoints (the
20260821_142144 run's among them) resumable — warned, never silently —
while an actual kwarg or section difference still fails closed.
"""

from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from environments.shared.task_fingerprint import (
    NON_TASK_ENV_PARAMS,
    TASK_FINGERPRINT_SCHEMA,
    TASK_FINGERPRINT_SCHEMA_V1,
    TaskFingerprintError,
    compute_task_fingerprint,
    derive_stage_task_fingerprint,
    validate_recorded_task,
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


def _v1_fingerprint(env_kwargs=_ENV, current=None):
    """A fingerprint exactly as schema v1 (04be963..HEAD) recorded it.

    ``env`` is the raw TOML-present kwargs; every other section is shared
    between v1 and v2, so it is copied from ``current`` (the perturbation
    block in particular must be identical for a valve pass).
    """
    if current is None:
        current = _fingerprint(env_kwargs=env_kwargs)
    payload = {
        "schema": TASK_FINGERPRINT_SCHEMA_V1,
        "species": current["species"],
        "stage": current["stage"],
        "backend": current["backend"],
        "plant": current["plant"],
        # json round-trip = the module's canonical form for TOML values
        # (tuples to lists, keys to str).
        "env": json.loads(json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in env_kwargs.items()})),
        "perturbation": current["perturbation"],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    payload["task_sha256"] = f"sha256:{digest}"
    return payload


def _retune_constructor_default(monkeypatch, env_class, name, value):
    """Rebind one constructor default in place, as a retuning commit would."""
    init = env_class.__init__
    kwdefaults = dict(init.__kwdefaults__ or {})
    if name in kwdefaults:
        kwdefaults[name] = value
        monkeypatch.setattr(init, "__kwdefaults__", kwdefaults)
        return
    names = [
        p.name
        for p in inspect.signature(init).parameters.values()
        if p.default is not inspect.Parameter.empty and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    defaults = list(init.__defaults__)
    assert len(names) == len(defaults) and name in names
    defaults[names.index(name)] = value
    monkeypatch.setattr(init, "__defaults__", tuple(defaults))


class TestEffectiveEnvSection:
    def test_schema_is_v2(self):
        payload = _fingerprint()
        assert payload["schema"] == TASK_FINGERPRINT_SCHEMA
        assert payload["schema"].endswith("/v2")

    def test_constructor_defaults_enter_the_env_section(self):
        env = _fingerprint()["env"]
        # TOML-present kwargs win over defaults; absent params come from
        # the TRexEnv constructor.
        assert env["alive_bonus"] == 1.0
        assert env["healthy_z_range"] == [0.70, 1.30]
        assert env["fall_penalty"] == -100.0
        assert env["nosedive_termination_threshold"] == 0.62

    def test_render_mode_is_not_task_identity(self):
        base = _fingerprint()
        with_render = _fingerprint(env_kwargs={**_ENV, "render_mode": "human"})
        assert with_render["task_sha256"] == base["task_sha256"]
        assert not NON_TASK_ENV_PARAMS & set(base["env"])

    def test_retuned_constructor_default_moves_the_hash(self, monkeypatch):
        """The F6 gap: a termination default is task identity, TOML or not."""
        from environments.trex.envs.trex_env import TRexEnv

        base = _fingerprint()
        _retune_constructor_default(monkeypatch, TRexEnv, "nosedive_termination_threshold", 0.55)
        retuned = _fingerprint()
        assert retuned["env"]["nosedive_termination_threshold"] == 0.55
        assert retuned["task_sha256"] != base["task_sha256"]

    def test_unknown_species_fails_closed(self):
        with pytest.raises(TaskFingerprintError, match="Unknown species"):
            _fingerprint(species="stegosaurus")


class TestSchemaV1Valve:
    def test_v1_record_resumes_through_the_valve_with_a_warning(self, caplog):
        current = _fingerprint()
        recorded = _v1_fingerprint(current=current)
        assert recorded["task_sha256"] != current["task_sha256"]
        with caplog.at_level("WARNING"):
            assert validate_recorded_task(recorded, current, mode="resume_same_stage") is None
        assert "schema-v1" in caplog.text
        assert "valve" in caplog.text

    def test_head_schema_recovery_fingerprint_remains_resumable(self, caplog):
        """The 20260821_142144 run's fingerprints must survive the bump."""
        pytest.importorskip("mujoco")
        from environments.shared.config import load_stage_config

        env_kwargs = load_stage_config("trex", "recovery")["env_kwargs"]
        current = derive_stage_task_fingerprint(
            species="trex",
            stage="recovery",
            backend="stable-baselines3",
            env_kwargs=env_kwargs,
            plant_identity=_PLANT,
        )
        recorded = _v1_fingerprint(env_kwargs=env_kwargs, current=current)
        with caplog.at_level("WARNING"):
            assert validate_recorded_task(recorded, current, mode="resume_same_stage") is None
        assert "valve" in caplog.text

    def test_valve_rejects_a_raw_kwarg_mismatch(self):
        recorded = _v1_fingerprint(env_kwargs={**_ENV, "alive_bonus": 0.5})
        with pytest.raises(TaskFingerprintError, match="different task"):
            validate_recorded_task(recorded, _fingerprint(), mode="resume_same_stage")

    def test_valve_rejects_a_recorded_kwarg_the_current_task_lacks(self):
        recorded = _v1_fingerprint(env_kwargs={**_ENV, "not_a_constructor_param": 1.0})
        with pytest.raises(TaskFingerprintError, match="different task"):
            validate_recorded_task(recorded, _fingerprint(), mode="resume_same_stage")

    def test_valve_accepts_a_kwarg_that_moved_into_the_defaults(self):
        # v1 pinned fall_penalty explicitly at the constructor's value; the
        # current TOML omits it — the effective config makes them equal.
        recorded = _v1_fingerprint(env_kwargs={**_ENV, "fall_penalty": -100.0})
        assert validate_recorded_task(recorded, _fingerprint(), mode="resume_same_stage") is None

    def test_valve_rejects_a_non_env_section_mismatch(self):
        current = _fingerprint()
        recorded = dict(_v1_fingerprint(current=current))
        recorded["plant"] = {"physics_sha256": "sha256:cccc", "policy_interface_sha256": "sha256:bbbb"}
        with pytest.raises(TaskFingerprintError, match="different task"):
            validate_recorded_task(recorded, current, mode="resume_same_stage")

    def test_valve_never_applies_between_v2_fingerprints(self):
        recorded = _fingerprint(env_kwargs={**_ENV, "alive_bonus": 0.5})
        with pytest.raises(TaskFingerprintError, match="different task"):
            validate_recorded_task(recorded, _fingerprint(), mode="resume_same_stage")

    def test_valve_never_applies_to_an_unknown_schema(self):
        current = _fingerprint()
        recorded = dict(_v1_fingerprint(current=current))
        recorded["schema"] = "mesozoic.task-fingerprint/v0"
        with pytest.raises(TaskFingerprintError, match="different task"):
            validate_recorded_task(recorded, current, mode="resume_same_stage")

    def test_initialize_next_stage_records_v1_parent_lineage_unvalved(self):
        parent = _v1_fingerprint()
        child = _fingerprint(stage=2)
        lineage = validate_recorded_task(parent, child, mode="initialize_next_stage")
        assert lineage["parent_task_sha256"] == parent["task_sha256"]
        assert lineage["child_task_sha256"] == child["task_sha256"]
