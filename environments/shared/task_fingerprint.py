"""Task/evaluation fingerprint and checkpoint load modes (stage 1b, W2).

Why this exists (STAGE1_SPLIT_PLAN §3.2): the scheduled-push hook lives in
``step()``, and the plant contract fingerprints observation/action
implementations — ``step`` appears nowhere in the policy-interface payload.
A pushed task therefore moves **no** ``policy_interface_revision``: every
existing checkpoint stays mechanically loadable while being unvalidated
for the new transition kernel and evaluation distribution.  That silence
is a provenance gap, not evidence of task equivalence.  This module closes
it with a second, task-level identity: what the policy was trained to
*do*, alongside the plant contract's what the policy *talks to*.

The fingerprint hashes the task-defining configuration — species, stage,
backend, the plant's physics/interface hashes, the full ``[env]`` kwargs
(reward, termination, reset, horizon), and the perturbation block with its
per-species **derived** force constants plus the schedule implementation
name (the PRF is part of task identity: change the hash function and every
recorded schedule means something else).

Two load modes, because "resume must not cross a task boundary silently"
must coexist with 1b *meaning* to warm-start from a 1a checkpoint across
exactly that boundary:

* ``resume_same_stage`` — the fingerprints must match exactly; a mismatch
  is an error naming the differing sections.
* ``initialize_next_stage`` — a mismatch is expected once and recorded as
  lineage (parent/child hashes travel with the new checkpoint), never
  silently.

Fail-closed core with one explicit, dated transition valve: no checkpoint
minted before 2026-08-15 carries a fingerprint, so callers may pass
``allow_unfingerprinted=True`` (train paths do, with a comment) to demote
the missing-fingerprint error to a warning until fingerprinted checkpoints
are the norm.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping

_logger = logging.getLogger(__name__)

TASK_FINGERPRINT_SCHEMA = "mesozoic.task-fingerprint/v1"
#: Names the schedule PRF + force-derivation implementation; bump when
#: environments/shared/perturbation.py changes meaning, not merely shape.
SCHEDULE_IMPLEMENTATION = "push_schedule/v1+lowbias32+capture-point"
#: Attribute Stable-Baselines3 persists inside the checkpoint ZIP.
MODEL_TASK_ATTRIBUTE = "mesozoic_task_fingerprint"
#: Warm-start lineage record, persisted the same way on the CHILD checkpoint.
MODEL_TASK_LINEAGE_ATTRIBUTE = "mesozoic_task_lineage"

LOAD_MODES = ("resume_same_stage", "initialize_next_stage")

_MISSING = object()


class TaskFingerprintError(RuntimeError):
    """A checkpoint's recorded task does not permit the requested load."""


def _canonical(value: Any) -> Any:
    """JSON-stable form: tuples to lists, dict keys sorted by dumps below."""
    if isinstance(value, Mapping):
        return {str(key): _canonical(sub) for key, sub in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # repr round-trips exactly; float() collapses numpy scalars first.
        return float(value)
    return str(value)


def compute_task_fingerprint(
    *,
    species: str,
    stage: int,
    backend: str,
    env_kwargs: Mapping[str, Any],
    plant_identity: Mapping[str, Any] | None,
    perturbation_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the canonical task payload and stamp its sha256.

    ``env_kwargs`` is the stage's TOML-derived ``[env]`` dict — the explicit
    task specification.  ``perturbation_manifest`` is the environment's
    derived per-species constants (``perturbation_manifest()``), or ``None``
    when pushes are off; it is recorded separately from the config keys
    because the same dimensionless multiple means different newtons on
    different plants, and the newtons are the task.
    """
    plant = None
    if plant_identity is not None:
        plant = {
            "physics_sha256": plant_identity.get("physics_sha256"),
            "policy_interface_sha256": plant_identity.get("policy_interface_sha256"),
        }
    perturbation = None
    if perturbation_manifest is not None:
        perturbation = _canonical(dict(perturbation_manifest))
        perturbation["schedule_implementation"] = SCHEDULE_IMPLEMENTATION
    payload: dict[str, Any] = {
        "schema": TASK_FINGERPRINT_SCHEMA,
        "species": species,
        "stage": int(stage),
        "backend": backend,
        "plant": plant,
        "env": _canonical(dict(env_kwargs)),
        "perturbation": perturbation,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    payload["task_sha256"] = f"sha256:{digest}"
    return payload


def write_task_fingerprint(path: str | Path, fingerprint: Mapping[str, Any]) -> Path:
    """Atomically write the task-fingerprint sidecar (like plant identity)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(dict(fingerprint), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)
    return path


def attach_task_fingerprint(model: Any, fingerprint: Mapping[str, Any]) -> None:
    """Attach the fingerprint so Stable-Baselines3 persists it in its ZIP."""
    setattr(model, MODEL_TASK_ATTRIBUTE, dict(fingerprint))


def attach_task_lineage(model: Any, lineage: Mapping[str, Any]) -> None:
    """Attach a warm-start lineage record to the child checkpoint."""
    setattr(model, MODEL_TASK_LINEAGE_ATTRIBUTE, dict(lineage))


def _differing_sections(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    sections = ("species", "stage", "backend", "plant", "env", "perturbation")
    return [name for name in sections if _canonical(recorded.get(name)) != _canonical(current.get(name))]


def validate_recorded_task(
    recorded: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    *,
    mode: str,
    artifact: str = "checkpoint",
    allow_unfingerprinted: bool = False,
) -> dict[str, Any] | None:
    """Validate a recorded fingerprint against the current task, per mode.

    Returns a lineage record for ``initialize_next_stage`` (always — the
    warm-start is recorded even when the tasks happen to match), ``None``
    for a clean same-stage resume.  Raises :class:`TaskFingerprintError`
    on a same-stage mismatch, an unknown mode, or a missing fingerprint
    without the explicit transition valve.
    """
    if mode not in LOAD_MODES:
        raise TaskFingerprintError(f"unknown load mode {mode!r}; known: {LOAD_MODES}")
    if recorded is None:
        message = (
            f"{artifact} carries no task fingerprint (minted before 2026-08-15); "
            f"its training task cannot be verified for {mode}"
        )
        if mode == "initialize_next_stage":
            _logger.warning("%s — recording a parentless lineage", message)
            return {
                "mode": mode,
                "parent_task_sha256": None,
                "child_task_sha256": current["task_sha256"],
            }
        if allow_unfingerprinted:
            _logger.warning("%s — loading only because allow_unfingerprinted was explicitly enabled", message)
            return None
        raise TaskFingerprintError(
            message
            + ". Pass allow_unfingerprinted=True only for deliberate transition of a pre-fingerprint checkpoint."
        )
    if mode == "resume_same_stage":
        if recorded.get("task_sha256") != current.get("task_sha256"):
            differing = _differing_sections(recorded, current)
            raise TaskFingerprintError(
                f"{artifact} was trained on a different task than it is being resumed into "
                f"(differing sections: {differing or ['task_sha256']}). A changed task must go through "
                "initialize_next_stage so the boundary is recorded as lineage, never resumed silently."
            )
        return None
    return {
        "mode": mode,
        "parent_task_sha256": recorded.get("task_sha256"),
        "child_task_sha256": current["task_sha256"],
        "parent_species": recorded.get("species"),
        "parent_stage": recorded.get("stage"),
    }


def validate_model_task(
    model: Any,
    current: Mapping[str, Any],
    *,
    mode: str,
    artifact: str = "checkpoint",
    allow_unfingerprinted: bool = False,
) -> dict[str, Any] | None:
    """Validate the fingerprint embedded in a loaded SB3 model, per mode."""
    raw = getattr(model, MODEL_TASK_ATTRIBUTE, _MISSING)
    if raw is _MISSING:
        return validate_recorded_task(
            None, current, mode=mode, artifact=artifact, allow_unfingerprinted=allow_unfingerprinted
        )
    if not isinstance(raw, Mapping):
        raise TaskFingerprintError(
            f"{artifact} contains invalid task fingerprint metadata of type {type(raw).__name__}"
        )
    return validate_recorded_task(
        raw, current, mode=mode, artifact=artifact, allow_unfingerprinted=allow_unfingerprinted
    )


def derive_stage_task_fingerprint(
    *,
    species: str,
    stage: int,
    backend: str,
    env_kwargs: Mapping[str, Any],
    plant_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compute a stage's task fingerprint without constructing an environment.

    Re-derives the perturbation manifest from the plant model named by the
    plant identity, with the same defaults ``BaseDinoEnv`` uses, so the
    fingerprint can be written with ``stage_config.json`` before any
    environment exists.  Derivation uses keyframe 0, which is the home
    keyframe for every current species; a species whose reset keyframe
    diverges must revisit this (BaseDinoEnv derives from its
    ``_reset_keyframe_id``).
    """
    manifest: dict[str, Any] | None = None
    multiple = float(env_kwargs.get("perturbation_capture_velocity_multiple", 0.0) or 0.0)
    if multiple > 0.0:
        if plant_identity is None or not plant_identity.get("model_path"):
            raise TaskFingerprintError(
                "a pushed task fingerprint needs the plant identity's model_path to derive the force constants"
            )
        import mujoco

        from .perturbation import derive_push_parameters

        model = mujoco.MjModel.from_xml_path(str(plant_identity["model_path"]))
        params = derive_push_parameters(
            model,
            capture_velocity_multiple=multiple,
            duration_s=float(env_kwargs.get("perturbation_duration", 0.20)),
        )
        manifest = {
            **params,
            "interval_s": float(env_kwargs.get("perturbation_interval", 2.0)),
            "jitter_s": float(env_kwargs.get("perturbation_jitter", 0.5)),
            "direction": env_kwargs.get("perturbation_direction", "uniform_horizontal"),
        }
    return compute_task_fingerprint(
        species=species,
        stage=stage,
        backend=backend,
        env_kwargs=env_kwargs,
        plant_identity=plant_identity,
        perturbation_manifest=manifest,
    )
