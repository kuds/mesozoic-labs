"""Fail-closed identities and paired checkpoints for the opt-in balance study.

These artifacts deliberately use an unregistered research species. They cannot
be loaded through the canonical Compsognathus policy path, even where their
observation and action dimensions agree. No generated plant manifest is written
or relaxed: the canonical source bytes are checked, and the local compiled
plant is fingerprinted with the public plant-contract API.

Checkpoint hashes detect damage and accidental pairing errors, not malicious
replacement of the entire bundle. Load only study bundles created by this
trusted training workflow; SB3 and VecNormalize contain executable pickle data.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import uuid
from dataclasses import asdict
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from environments.compsognathus import MODEL_PATHS
from environments.compsognathus.envs.compsognathus_env import CompsognathusEnv
from environments.shared.plant_contract import (
    FINGERPRINT_TOOL_VERSION,
    GENERATED_MANIFEST_PATH,
    MODEL_IDENTITY_ATTRIBUTE,
    REPOSITORY_ROOT,
    PlantCompatibilityError,
    PlantIdentity,
    PlantVersion,
    fingerprint_model_layers,
)

STUDY_IDENTITY_ATTRIBUTE = "_balance_study_identity"
STUDY_IDENTITY_SCHEMA = "mesozoic.compsognathus-balance-study/v1"
CHECKPOINT_SCHEMA = "mesozoic.compsognathus-balance-checkpoint/v1"
_PAIR_ATTRIBUTE = "_balance_study_checkpoint_pair"


def _json_copy(value: Any) -> Any:
    """Detach mutable input and reject nonportable or nonfinite config values."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "stable_baselines3": distribution_version("stable-baselines3"),
        "torch": distribution_version("torch"),
        "gymnasium": distribution_version("gymnasium"),
        "fingerprint_tool": str(FINGERPRINT_TOOL_VERSION),
    }


def _source_files() -> dict[str, str]:
    """Exact study/plant/config source closure, including shared implementation.

    The conservative shared-module closure also covers imported helpers without
    depending on which branches have executed in the current Python process.
    Tests, reports and generated run outputs are excluded.
    """
    manifest = json.loads(GENERATED_MANIFEST_PATH.read_text())
    source = manifest["plants"]["compsognathus"]["source"]
    canonical_model = MODEL_PATHS["biological"].relative_to(REPOSITORY_ROOT).as_posix()
    if source["root"] != canonical_model:
        raise PlantCompatibilityError("canonical Compsognathus source pointer changed")
    files: dict[str, str] = {}
    for dependency in source["dependencies"]:
        relative = dependency["logical_path"]
        path = (REPOSITORY_ROOT / relative).resolve()
        if not path.is_relative_to(REPOSITORY_ROOT):
            raise PlantCompatibilityError("canonical source dependency leaves repository")
        actual = _file_digest(path)
        if actual != dependency["content_sha256"]:
            raise PlantCompatibilityError(f"canonical model source changed: {relative}")
        files[relative] = actual
    patterns = (
        "environments/__init__.py",
        "environments/compsognathus/*.py",
        "environments/compsognathus/envs/*.py",
        "environments/compsognathus/experiments/*.py",
        "environments/compsognathus/scripts/train_balance_study.py",
        "environments/shared/*.py",
        "environments/shared/curriculum/*.py",
        "environments/shared/plant_contract/*.py",
        "configs/compsognathus/*.toml",
        "configs/plant_versions.toml",
        "configs/species_manifest.toml",
    )
    for pattern in patterns:
        for path in sorted(REPOSITORY_ROOT.glob(pattern)):
            files[path.relative_to(REPOSITORY_ROOT).as_posix()] = _file_digest(path)
    return dict(sorted(files.items()))


def _environment_snapshot(env: Any) -> dict[str, Any]:
    from .balance_env import CompsognathusBalanceEnv, FilteredCompsognathusBalanceEnv

    if type(env) not in (CompsognathusBalanceEnv, FilteredCompsognathusBalanceEnv):
        raise PlantCompatibilityError("balance identity requires an explicit balance-study environment")
    values = {
        name: getattr(env, name)
        for name in inspect.signature(CompsognathusEnv.__init__).parameters
        if name not in {"self", "render_mode"}
    }
    snapshot: dict[str, Any] = _json_copy(
        {
            "class": type(env).__name__,
            "arm": asdict(env.balance_arm),
            "env_kwargs": values,
            "action_filter_cutoff_hz": env.action_filter_cutoff_hz,
            "action_filter_alpha": env._action_filter_alpha,
            "control_timestep": env.dt,
        }
    )
    return snapshot


def study_source_fingerprint() -> str:
    """Fingerprint the exact source closure for freezing a multi-arm run plan."""
    return _digest(_source_files())


def build_study_identity(env: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot effective environment, full run config, sources and runtime.

    Call once per arm/seed before training, using an unwrapped study environment.
    The caller should include training/normalization settings and seed in config.
    This is a research identity, never a canonical-plant verification bypass.
    """
    snapshot = _environment_snapshot(env)
    source_files = _source_files()
    cutoff = float(env.action_filter_cutoff_hz)
    if cutoff not in (0.0, 10.0):
        raise PlantCompatibilityError("balance study only supports the declared off/10 Hz filter arms")
    species = "compsognathus_balance_study_" + ("off" if cutoff == 0.0 else "10hz")
    version = PlantVersion(species, 1, 1, 1, "bipedal-target/v1")
    layers = fingerprint_model_layers(env.model, env, version)
    declared_model = mujoco.MjModel.from_xml_path(str(MODEL_PATHS["biological"]))
    if layers != fingerprint_model_layers(declared_model, env, version):
        raise PlantCompatibilityError("balance-study compiled model differs from the unchanged canonical source")
    plant = PlantIdentity(
        species=species,
        model_path=MODEL_PATHS["biological"].relative_to(REPOSITORY_ROOT).as_posix(),
        physics_revision=1,
        policy_interface_revision=1,
        visual_revision=1,
        source_closure_sha256=_digest(source_files),
        **layers,
        nq=int(env.model.nq),
        nv=int(env.model.nv),
        nu=int(env.model.nu),
        observation_dim=int(np.prod(env.observation_space.shape)),
        action_dim=int(np.prod(env.action_space.shape)),
    )
    identity = {
        "schema": STUDY_IDENTITY_SCHEMA,
        "status": "research-only; not canonical production identity",
        "vector_backend": "stable_baselines3.common.vec_env.DummyVecEnv",
        "plant_identity": plant.to_dict(),
        "environment": snapshot,
        "config": _json_copy(dict(config)),
        "runtime": _runtime(),
        "source_files": source_files,
    }
    identity["identity_sha256"] = _digest(identity)
    return identity


def _checked_identity(identity: object) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or identity.get("schema") != STUDY_IDENTITY_SCHEMA:
        raise PlantCompatibilityError("missing or invalid balance-study identity")
    value: dict[str, Any] = _json_copy(dict(identity))
    recorded_digest = value.pop("identity_sha256", None)
    if recorded_digest != _digest(value):
        raise PlantCompatibilityError("balance-study identity digest mismatch")
    plant = PlantIdentity.from_mapping(value.get("plant_identity", {}))
    if plant.species not in {"compsognathus_balance_study_off", "compsognathus_balance_study_10hz"}:
        raise PlantCompatibilityError("balance-study artifact cannot claim a canonical species")
    value["identity_sha256"] = recorded_digest
    return value


def validate_study_identity(recorded: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> None:
    """Reject every arm/config/source/runtime mismatch, even at equal dimensions."""
    recorded_value = _checked_identity(recorded)
    expected_value = _checked_identity(expected)
    if recorded_value != expected_value:
        differences = sorted(key for key in expected_value if expected_value[key] != recorded_value.get(key))
        raise PlantCompatibilityError("balance-study identity mismatch: " + ", ".join(differences))


def validate_study_environment(env: Any, identity: Mapping[str, Any]) -> None:
    """Recompute identity to catch task/model/source mutation before save or load."""
    expected = _checked_identity(identity)
    validate_study_identity(build_study_identity(env, expected["config"]), expected)


def attach_study_identity(artifact: Any, identity: Mapping[str, Any]) -> None:
    """Tag a new model/normalizer; never relabel a canonical or different study."""
    value = _checked_identity(identity)
    plant = value["plant_identity"]
    prior_plant = getattr(artifact, MODEL_IDENTITY_ATTRIBUTE, None)
    prior_study = getattr(artifact, STUDY_IDENTITY_ATTRIBUTE, None)
    if prior_plant is not None and prior_plant != plant:
        raise PlantCompatibilityError("refusing to relabel an existing plant identity as a balance study")
    if prior_study is not None:
        validate_study_identity(prior_study, value)
    setattr(artifact, MODEL_IDENTITY_ATTRIBUTE, _json_copy(plant))
    setattr(artifact, STUDY_IDENTITY_ATTRIBUTE, _json_copy(value))


def _validate_artifact(artifact: Any, identity: Mapping[str, Any]) -> None:
    validate_study_identity(getattr(artifact, STUDY_IDENTITY_ATTRIBUTE, None), identity)
    if getattr(artifact, MODEL_IDENTITY_ATTRIBUTE, None) != identity["plant_identity"]:
        raise PlantCompatibilityError("artifact plant identity differs from its balance-study identity")


def _validate_vector_environment(env: Any, identity: Mapping[str, Any]) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    current = env.venv if isinstance(env, VecNormalize) else env
    if not isinstance(current, DummyVecEnv):
        raise PlantCompatibilityError("balance checkpoint validation requires a local DummyVecEnv")
    for wrapped in current.envs:
        validate_study_environment(wrapped.unwrapped, identity)


def _paths(path: str | Path) -> tuple[Path, Path, Path]:
    prefix = Path(path)
    return (
        Path(str(prefix) + ".model.zip"),
        Path(str(prefix) + ".vecnormalize.pkl"),
        Path(str(prefix) + ".manifest.json"),
    )


def save_study_checkpoint(model: Any, normalizer: Any, path: str | Path, identity: Mapping[str, Any]) -> Path:
    """Save an already-tagged PPO/VecNormalize pair under a checkpoint prefix.

    Files are ``prefix.model.zip``, ``prefix.vecnormalize.pkl`` and
    ``prefix.manifest.json``. The manifest is committed last; a partial save
    cannot pass the loader's hashes. Use a new prefix for each saved checkpoint.
    """
    expected = _checked_identity(identity)
    _validate_artifact(model, expected)
    _validate_artifact(normalizer, expected)
    _validate_vector_environment(normalizer, expected)
    if model.get_env() is not normalizer:
        raise PlantCompatibilityError("model and normalizer must belong to the same training environment")
    model_path, norm_path, manifest_path = _paths(path)
    if any(candidate.exists() for candidate in (model_path, norm_path, manifest_path)):
        raise FileExistsError("study checkpoint prefix already exists; use a new checkpoint prefix")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pair_id = str(uuid.uuid4())
    setattr(model, _PAIR_ATTRIBUTE, pair_id)
    setattr(normalizer, _PAIR_ATTRIBUTE, pair_id)
    model.save(str(model_path))
    normalizer.save(str(norm_path))
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "identity": expected,
        "pair_id": pair_id,
        "model": {"filename": model_path.name, "sha256": _file_digest(model_path)},
        "normalizer": {"filename": norm_path.name, "sha256": _file_digest(norm_path)},
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return manifest_path


def load_study_checkpoint(path: str | Path, expected_identity: Mapping[str, Any], env: Any) -> tuple[Any, Any]:
    """Load a trusted paired study checkpoint after hashes and identities pass.

    Supply the checkpoint prefix and a fresh, unnormalized DummyVecEnv with the
    exact study task. Canonical/historical checkpoints have no accepted path.
    The returned normalizer is frozen for evaluation with raw rewards.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    expected = _checked_identity(expected_identity)
    _validate_vector_environment(env, expected)
    if isinstance(env, VecNormalize):
        raise PlantCompatibilityError("load requires an unnormalized environment")
    model_path, norm_path, manifest_path = _paths(path)
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise PlantCompatibilityError("invalid balance-study checkpoint manifest")
        validate_study_identity(manifest.get("identity"), expected)
        for name, artifact_path in (("model", model_path), ("normalizer", norm_path)):
            entry = manifest[name]
            if entry["filename"] != artifact_path.name or entry["sha256"] != _file_digest(artifact_path):
                raise PlantCompatibilityError(f"balance-study {name} checkpoint hash or filename mismatch")
        pair_id = manifest["pair_id"]
        if not isinstance(pair_id, str) or not pair_id:
            raise PlantCompatibilityError("balance-study checkpoint has no pairing identity")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PlantCompatibilityError(f"cannot verify balance-study checkpoint: {exc}") from exc
    # Nothing containing pickle is loaded until BOTH hashes have been checked.
    normalizer = VecNormalize.load(str(norm_path), env)
    _validate_artifact(normalizer, expected)
    if getattr(normalizer, _PAIR_ATTRIBUTE, None) != pair_id:
        raise PlantCompatibilityError("normalizer belongs to a different checkpoint pair")
    model = PPO.load(str(model_path), env=normalizer, device="cpu")
    _validate_artifact(model, expected)
    if getattr(model, _PAIR_ATTRIBUTE, None) != pair_id:
        raise PlantCompatibilityError("model belongs to a different checkpoint pair")
    normalizer.training = False
    normalizer.norm_reward = False
    return model, normalizer
