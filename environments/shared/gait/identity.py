"""Hash the detector, geometry registry, implementation, and fixed test panel."""

from __future__ import annotations

from typing import Any

from ..paths import REPOSITORY_ROOT, SHARED_ROOT
from ..result_bundle.hashing import canonical_json_sha256, sha256_file
from .morphology import FOOT_GEOMETRIES, FOOT_REGISTRY_VERSION
from .types import GaitProtocol

_SOURCES = (
    "gait/types.py",
    "gait/events.py",
    "gait/labels.py",
    "gait/metrics.py",
    "gait/morphology.py",
    "gait/recorder.py",
    "gait/seeds.py",
    "gait/identity.py",
    "gait/report.py",
    "curriculum/gait_gate.py",
    "curriculum/gate_schema.py",
    "curriculum/recovery_gate.py",
    "reporting/gates.py",
)


def current_implementation_hashes() -> dict[str, str]:
    """Exact producer/detector/classifier bytes used by this build."""
    shared = SHARED_ROOT
    return {(shared / name).relative_to(REPOSITORY_ROOT).as_posix(): sha256_file(shared / name) for name in _SOURCES}


def validate_measurement_protocol_identity(payload: dict[str, Any], species: str) -> bool:
    """Refuse stale code or registries even if an old declared digest matches."""
    registry = FOOT_GEOMETRIES.get(species)
    return bool(
        registry
        and payload.get("implementation_sha256") == current_implementation_hashes()
        and payload.get("foot_registry_version") == FOOT_REGISTRY_VERSION
        and payload.get("foot_names") == list(registry)
        and payload.get("foot_geometries") == {key: list(value) for key, value in registry.items()}
    )


def measurement_protocol(
    species: str,
    protocol: GaitProtocol,
    *,
    settle_s: float,
    direction_xy: tuple[float, float],
    horizon: int,
    physics_dt_s: float,
    control_dt_s: float,
    episodes: int,
    seed_start: int,
) -> dict[str, Any]:
    import mujoco

    return {
        "schema": "mesozoic.gait-protocol/v2",
        "detector": protocol.to_dict(),
        "foot_registry_version": FOOT_REGISTRY_VERSION,
        "foot_names": list(FOOT_GEOMETRIES[species]),
        "foot_geometries": {key: list(value) for key, value in FOOT_GEOMETRIES[species].items()},
        "implementation_sha256": current_implementation_hashes(),
        "sampling": {
            "mujoco_version": mujoco.__version__,
            "physics_dt_s": physics_dt_s,
            "control_dt_s": control_dt_s,
            "contact_timing": "initial reset sample plus latest solved contacts after each mj_step",
            "slip": "normal-force-weighted RMS relative tangential contact-point speed",
            "skid": "per-limb stance slip distance over trunk travel during the same stances",
            "phase": "continuous touchdown-to-touchdown limb phase; pair statistics are circular means",
            "clearance": "minimum signed registered-foot to declared-terrain geometry distance",
            "footprint": "load-weighted (force x time) centre of each stance",
            "step_length": "footprint ahead of the contralateral foot's previous footprint along the trunk axis "
            "(root-quaternion body x axis, averaged over one stride)",
            "step_through_stride": "consecutive steps of the two feet of a contralateral pair, both stepping through",
            "body_support_fraction": "non-foot ground impulse divided by total animal ground impulse",
        },
        "panel": {
            "horizon_control_steps": horizon,
            "settle_s": settle_s,
            "direction_xy": list(direction_xy),
            "episodes": episodes,
            "seed_start": seed_start,
            "seed_scheme": "explicit env.reset(seed=seed_start+i), one rollout per seed",
            "action_mode": "deterministic",
            "normalization": "frozen saved VecNormalize; no reward normalization",
            "scenario_distribution": "configured task reset distribution; no additional perturbations",
        },
    }


def protocol_sha256(payload: dict[str, Any]) -> str:
    return canonical_json_sha256(payload)
