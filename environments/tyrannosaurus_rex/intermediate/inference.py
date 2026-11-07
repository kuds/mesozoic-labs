"""Package and load the exact matched actor/statistics/plant/task for inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from environments.shared import robot_gait_adapter
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import current_plant_identity
from environments.shared.policy_loading import load_sb3_checkpoint, resolve_vecnorm_path
from environments.shared.task_fingerprint import stage_task_fingerprint, validate_model_task

from .env import TyrannosaurusRexIntermediateEnv

MODEL = ModelId("tyrannosaurus_rex", "intermediate")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_verified(directory: Path):
    identity = current_plant_identity(MODEL)
    actor, stats, _ = load_sb3_checkpoint(
        str(directory / "model.zip"),
        str(directory / "vecnormalize.pkl"),
        TyrannosaurusRexIntermediateEnv,
        plant_identity=identity,
    )
    task = stage_task_fingerprint(MODEL, "locomotion", plant_identity=identity)
    validate_model_task(actor, task, mode="resume_same_stage", artifact=str(directory / "model.zip"))
    return actor, stats, identity, task


def load_policy(directory: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    path = Path(directory)
    manifest = json.loads((path / "bundle.json").read_text())
    if manifest.get("schema") != "mesozoic.intermediate-gait/v1" or not MODEL.matches(manifest):
        raise ValueError("Wrong gait bundle format or robot identity")
    for name in ("model.zip", "vecnormalize.pkl"):
        if _sha256(path / name) != manifest["sha256"][name]:
            raise ValueError("Gait bundle file digest mismatch: " + name)
    adapter = manifest.get("adapter")
    if adapter is not None:
        fit = path / "fit.json"
        if not isinstance(adapter, dict) or adapter.get("schema") != "mesozoic.transition-correction/v1":
            raise ValueError("Unknown gait correction")
        if (
            _sha256(Path(robot_gait_adapter.__file__)) != adapter["implementation_sha256"]
            or _sha256(fit) != manifest["sha256"]["fit.json"]
        ):
            raise ValueError("Fitted correction or implementation differs from the gait bundle")
        fit_data = json.loads(fit.read_text())
        if (
            fit_data["best"] != adapter["coefficients"]
            or fit_data["frozen_PPO_bundle_sha256"] != adapter["base_sha256"]
        ):
            raise ValueError("Correction and fit evidence disagree")
    actor, stats, identity, task = _load_verified(path)
    if (
        manifest["plant"] != identity.to_dict()
        or manifest["task_sha256"] != task["task_sha256"]
        or manifest["xml_sha256"] != _sha256(TyrannosaurusRexIntermediateEnv.MODEL_PATH)
    ):
        raise ValueError("Gait bundle identity differs from the installed robot or task")

    def predict(observation: np.ndarray) -> np.ndarray:
        action, _ = actor.predict(stats.normalize_obs(observation), deterministic=True)
        if adapter is not None:
            return robot_gait_adapter.adapt(observation, action, adapter["coefficients"])
        return np.asarray(action)

    return predict


def package(checkpoint: Path, normalization: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    matched = resolve_vecnorm_path(str(checkpoint), None, allow_unnormalized=False)
    if matched is None or normalization.resolve() != Path(matched).resolve():
        raise ValueError("Checkpoint requires its matched VecNormalize sidecar")
    destination.mkdir(parents=True)
    try:
        shutil.copyfile(checkpoint, destination / "model.zip")
        shutil.copyfile(normalization, destination / "vecnormalize.pkl")
        actor, _, identity, task = _load_verified(destination)
        record = {
            "schema": "mesozoic.intermediate-gait/v1",
            **MODEL.record(),
            "stage": "locomotion",
            "algorithm": "PPO",
            "seed": actor.seed,
            "model_timesteps": actor.num_timesteps,
            "plant": identity.to_dict(),
            "task_sha256": task["task_sha256"],
            "xml_sha256": _sha256(TyrannosaurusRexIntermediateEnv.MODEL_PATH),
            "sha256": {name: _sha256(destination / name) for name in ("model.zip", "vecnormalize.pkl")},
            "training_checkpoint": checkpoint.name,
            "training_statistics": normalization.name,
            "normalization": "frozen running observation statistics; deterministic prediction",
        }
        (destination / "bundle.json").write_text(json.dumps(record, indent=2) + "\n")
    except Exception:
        shutil.rmtree(destination)
        raise


def attach_fit(source: Path, fit: Path, destination: Path) -> None:
    """Freeze a training-seed fit against the exact parent PPO and statistics."""
    record = json.loads((source / "bundle.json").read_text())
    fit_data = json.loads(fit.read_text())
    parent = _sha256(source / "bundle.json")
    if "adapter" in record or fit_data["frozen_PPO_bundle_sha256"] != parent:
        raise ValueError("Fit was not performed on this raw PPO bundle")
    if not any(row["coefficients"] == fit_data["best"] for row in fit_data["verification"]):
        raise ValueError("Selected correction was absent from training-seed verification")
    robot_gait_adapter.adapt(np.zeros(57), np.zeros(14), fit_data["best"])
    load_policy(source)
    destination.mkdir(parents=True)
    try:
        for name in ("model.zip", "vecnormalize.pkl"):
            shutil.copyfile(source / name, destination / name)
        shutil.copyfile(fit, destination / "fit.json")
        record["sha256"]["fit.json"] = _sha256(fit)
        record["adapter"] = {
            "schema": "mesozoic.transition-correction/v1",
            "base_sha256": parent,
            "implementation_sha256": _sha256(Path(robot_gait_adapter.__file__)),
            "coefficients": fit_data["best"],
            "training_seeds": fit_data["training_seeds"],
        }
        (destination / "bundle.json").write_text(json.dumps(record, indent=2) + "\n")
        load_policy(destination)
    except Exception:
        shutil.rmtree(destination)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export matched SB3 artifacts or attach a training-seed fit.")
    parser.add_argument("source", type=Path, help="Training checkpoint, or source bundle for --apply-fit")
    parser.add_argument("sidecar", type=Path, help="Matched statistics, or verified fit JSON for --apply-fit")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--apply-fit", action="store_true")
    args = parser.parse_args()
    if args.apply_fit:
        attach_fit(args.source, args.sidecar, args.destination)
    else:
        package(args.source, args.sidecar, args.destination)
