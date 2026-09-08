"""Load measured recovery judges without borrowing another species' calibration.

Profiles describe a fixed-command reference and a pre-registered qualification
target. Loading or freezing one does not establish that a learned policy meets
that target. The complete profile is hashed so changes to the height reference,
control clock, measurements or provenance invalidate an existing frozen judge.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import build_env, load_stage_config
from .curriculum.gate_resolver import GateResolutionError, thresholds_from_resolution
from .curriculum.recovery_gate import RecoveryGateThresholds
from .plant_contract import current_plant_identity
from .species_names import resolve_species_id
from .task_fingerprint import derive_stage_task_fingerprint

PROFILE_SCHEMA = "mesozoic.recovery-calibration/v1"
CONFIGS_ROOT = Path(__file__).resolve().parents[2] / "configs"
SAFE_SET_KEYS = {"height_error_max_m", "tilt_max_rad", "planar_speed_max_mps", "min_foot_force_n"}
CAPABILITY_KEYS = {
    "gate_kind",
    "min_recovery_success_lcb",
    "recovery_t_recover_steps",
    "recovery_dwell_steps",
    "min_eval_episodes",
    "min_paired_success_delta_lcb",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value: Any, name: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateResolutionError(f"calibration {name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum or (strict and number == minimum):
        raise GateResolutionError(f"calibration {name} is outside its physical range")
    return number


@dataclass(frozen=True)
class RecoveryCalibration:
    """Validated judge and qualification target; no policy performance claim."""

    profile: dict[str, Any]

    @property
    def safe_set(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.profile["safe_set"].items()}

    @property
    def height_reference_m(self) -> float:
        return float(self.profile["height_reference_m"])

    @property
    def thresholds(self) -> RecoveryGateThresholds:
        return thresholds_from_resolution({"capability_spec": self.profile["capability_spec"]})

    def evaluation_spec(self) -> dict[str, Any]:
        return {
            "schema": "mesozoic.recovery-evaluation/v1",
            "species": self.profile["species"],
            "calibration_id": self.profile["calibration_id"],
            "profile_sha256": "sha256:" + hashlib.sha256(_canonical(self.profile).encode()).hexdigest(),
            "control_dt_s": self.profile["control_dt_s"],
            "height_reference_m": self.height_reference_m,
            "safe_set": self.safe_set,
            "required_paired_nulls": list(self.profile["required_paired_nulls"]),
        }


def load_recovery_calibration(species: str, stage: int | str = "recovery", *, env: Any = None) -> RecoveryCalibration:
    """Require a current species profile, matching the actual measured task.

    There is deliberately no T-Rex fallback here. The legacy producer retains its
    historical T-Rex judge explicitly; all newly calibrated species require a file.
    """
    species = resolve_species_id(species)
    path = CONFIGS_ROOT / species / "recovery_calibration.json"
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict):
            raise ValueError("expected a JSON object")
        _canonical(profile)
    except (OSError, ValueError) as exc:
        raise GateResolutionError(f"cannot load recovery calibration {path}: {exc}") from exc
    if profile.get("schema") != PROFILE_SCHEMA or profile.get("species") != species:
        raise GateResolutionError(f"{path} has the wrong recovery calibration schema or species")
    if not isinstance(profile.get("calibration_id"), str) or not profile["calibration_id"].strip():
        raise GateResolutionError(f"{path} must declare a calibration_id")
    required_nulls = profile.get("required_paired_nulls")
    if (
        not isinstance(required_nulls, list)
        or len(required_nulls) != 2
        or any(not isinstance(name, str) for name in required_nulls)
        or set(required_nulls) != {"zero_action", "brace"}
    ):
        raise GateResolutionError(f"{path} must require unique zero_action and brace paired nulls")
    dt = _finite(profile.get("control_dt_s"), "control_dt_s", strict=True)
    height = _finite(profile.get("height_reference_m"), "height_reference_m", strict=True)
    safe = profile.get("safe_set")
    if not isinstance(safe, dict) or set(safe) != SAFE_SET_KEYS:
        raise GateResolutionError(f"{path} must declare all four safe_set measurements")
    for key in SAFE_SET_KEYS:
        _finite(safe[key], key, strict=key != "min_foot_force_n")
    if safe["tilt_max_rad"] >= math.pi / 2 or safe["height_error_max_m"] >= height:
        raise GateResolutionError(f"{path} safe_set admits a fallen posture")
    spec = profile.get("capability_spec")
    if not isinstance(spec, dict) or set(spec) != CAPABILITY_KEYS or spec["gate_kind"] != "recovery_quality/v1":
        raise GateResolutionError(f"{path} has an incomplete recovery capability_spec")
    for key in ("min_recovery_success_lcb", "min_paired_success_delta_lcb"):
        if _finite(spec[key], key, strict=True) > 1:
            raise GateResolutionError(f"calibration {key} must be at most one")
    for key in ("recovery_t_recover_steps", "recovery_dwell_steps", "min_eval_episodes"):
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (2 if key == "min_eval_episodes" else 1):
            raise GateResolutionError(f"calibration {key} must be a positive integer")
    config = load_stage_config(species, stage)
    measured_env = profile.get("recovery_env_kwargs")
    if not isinstance(measured_env, dict) or _canonical(measured_env) != _canonical(config["env_kwargs"]):
        raise GateResolutionError(f"{path} recovery environment changed; recalibrate this task")
    curriculum: Mapping[str, Any] = config["curriculum_kwargs"]
    if any(curriculum.get(key) != value for key, value in spec.items()):
        raise GateResolutionError(f"{path} capability_spec disagrees with the stage curriculum; recalibrate")
    plant_identity = current_plant_identity(species).to_dict()
    if profile.get("plant_identity") != plant_identity:
        raise GateResolutionError(f"{path} plant identity changed; recalibrate this plant")
    fingerprint = derive_stage_task_fingerprint(
        species=species,
        stage=stage,
        backend="stable-baselines3",
        env_kwargs=measured_env,
        plant_identity=plant_identity,
    )
    if profile.get("task_sha256") != fingerprint["task_sha256"]:
        raise GateResolutionError(f"{path} measured task fingerprint changed; recalibrate the recovery implementation")
    owns_env = env is None
    if owns_env:
        env = build_env(species, stage)
    try:
        if not math.isclose(float(env.dt), dt, rel_tol=0, abs_tol=1e-12):
            raise GateResolutionError(f"{path} control timestep changed; recalibrate the recovery clock")
        for key, value in measured_env.items():
            if not hasattr(env, key) or _canonical(getattr(env, key)) != _canonical(value):
                raise GateResolutionError(f"{path} actual environment {key} differs from the measured task")
        if not env.healthy_z_range[0] < height < env.healthy_z_range[1]:
            raise GateResolutionError(f"{path} height reference is outside the healthy range")
        if env.perturbation_capture_velocity_multiple <= 0:
            raise GateResolutionError(f"{path} recovery task has no scheduled pushes")
    finally:
        if owns_env:
            env.close()
    return RecoveryCalibration(profile)
