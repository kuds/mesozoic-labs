"""Hash the detector, geometry registry, measurement implementation, and fixed test panel.

The implementation identity covers only the code that shapes a stored
metric or a verdict: the recorder, morphology, segmentation, labels and
metrics in ``gait/``, the pure gate in ``curriculum/gait_gate.py`` and the
exact binomial bound it judges the success count with
(``curriculum/binomial.py``). Every in-repo module a hashed file imports is
hashed too, except ``record_fields`` (a digest-format check) and
``species_names`` (the species alias table), which shape no metric;
``test_gait_preflight`` keeps it so. The panel plumbing (seed provenance,
report writing, this module) and the shared readers are not hashed, so
editing them neither revokes a certificate nor makes a planned hash stale.
``MEASUREMENT_VERSION`` names the measurement semantics explicitly; bump it
with any change to what a stored metric means. Runtime versions (MuJoCo,
NumPy, Python) are recorded beside a report for information, never hashed:
the reader replays the raw traces rather than trusting a build.
"""

from __future__ import annotations

import hashlib
import math
import platform
from pathlib import Path
from typing import Any

from ..paths import REPOSITORY_ROOT, SHARED_ROOT
from ..result_bundle.hashing import canonical_json_sha256
from .morphology import FOOT_GEOMETRIES, FOOT_REGISTRY_VERSION
from .types import GaitProtocol

PROTOCOL_SCHEMA = "mesozoic.gait-protocol/v3"
#: The measurement semantics, named explicitly; bump it with any change to
#: what a stored gait metric or verdict means. 1: the first observer (PR
#: #585); 2: the phase-coupling core (locomotion_gait/v2); 3: walk-first,
#: with divergence, heightfield clearance and the non-terrain contact diagnostic.
MEASUREMENT_VERSION = 3
#: Panel settings every in-repo producer uses (train_curriculum's selected
#: handoff panel, generate_stage_artifacts, the command line's defaults).
DEFAULT_SETTLE_S = 1.0
DEFAULT_DIRECTION_XY = (1.0, 0.0)

_SOURCES = (
    "gait/types.py",
    "gait/events.py",
    "gait/labels.py",
    "gait/metrics.py",
    "gait/morphology.py",
    "gait/recorder.py",
    "curriculum/gait_gate.py",
    "curriculum/binomial.py",
)


def _source_sha256(path: Path) -> str:
    # Line endings normalized: a CRLF checkout is the same implementation.
    return "sha256:" + hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def current_implementation_hashes() -> dict[str, str]:
    """Exact measurement/classification bytes used by this build."""
    shared = SHARED_ROOT
    return {(shared / name).relative_to(REPOSITORY_ROOT).as_posix(): _source_sha256(shared / name) for name in _SOURCES}


def runtime_versions() -> dict[str, str]:
    """Informational runtime record of a report; never part of a hash."""
    import mujoco
    import numpy

    return {"mujoco": str(mujoco.__version__), "numpy": str(numpy.__version__), "python": platform.python_version()}


def validate_measurement_protocol_identity(payload: dict[str, Any], species: str) -> bool:
    """Refuse stale code or registries even if an old declared digest matches."""
    registry = FOOT_GEOMETRIES.get(species)
    return bool(
        registry
        and payload.get("schema") == PROTOCOL_SCHEMA
        and payload.get("measurement_version") == MEASUREMENT_VERSION
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
    return {
        "schema": PROTOCOL_SCHEMA,
        "measurement_version": MEASUREMENT_VERSION,
        "detector": protocol.to_dict(),
        "foot_registry_version": FOOT_REGISTRY_VERSION,
        "foot_names": list(FOOT_GEOMETRIES[species]),
        "foot_geometries": {key: list(value) for key, value in FOOT_GEOMETRIES[species].items()},
        "implementation_sha256": current_implementation_hashes(),
        "sampling": {
            "physics_dt_s": physics_dt_s,
            "control_dt_s": control_dt_s,
            "contact_timing": "initial reset sample plus latest solved contacts after each mj_step",
            "slip": "normal-force-weighted RMS relative tangential contact-point speed",
            "skid": "per-limb stance slip distance over trunk travel during the same stances",
            "phase": "continuous touchdown-to-touchdown limb phase; pair statistics are circular means",
            "clearance": "minimum signed registered-foot distance to declared terrain: mj_geomDistance to a plane, "
            "the vertical gap of sampled foot surfaces above the env's height map on a heightfield",
            "footprint": "load-weighted (force x time) centre of each stance",
            "step_length": "footprint ahead of the contralateral foot's previous footprint along the line of "
            "progression (travel heading, or the trunk axis for a symmetric crab walk)",
            "step_through_stride": "consecutive steps of the two feet of a contralateral pair, both stepping through",
            "body_support_fraction": "non-foot ground impulse divided by total animal ground impulse",
            "divergence": "a MuJoCo instability reset ends the trace; the episode is invalid telemetry",
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


def normalized_direction(direction_xy: Any) -> tuple[float, float]:
    """The unit panel direction; ``ValueError`` for anything but a finite nonzero 2-vector."""
    try:
        values = [float(value) for value in direction_xy]
    except (TypeError, ValueError) as error:
        raise ValueError("gait direction must be a finite nonzero 2-vector") from error
    if len(values) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("gait direction must be a finite nonzero 2-vector")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("gait direction must be a finite nonzero 2-vector")
    return values[0] / norm, values[1] / norm


def stage_measurement_protocol(
    species_cfg: Any,
    stage_config: dict[str, Any],
    *,
    episodes: int,
    seed_start: int,
    protocol: GaitProtocol | None = None,
    settle_s: float = DEFAULT_SETTLE_S,
    direction_xy: tuple[float, float] = DEFAULT_DIRECTION_XY,
) -> dict[str, Any]:
    """The measurement protocol a stage's panel would hash, built without rolling an episode.

    Constructs the stage's environment only to validate the plant and read its
    timesteps; it never resets or steps it.
    """
    from ..plant_contract import current_plant_identity, validate_environment_plant

    env_kwargs = dict(stage_config.get("env_kwargs", {}))
    horizon = env_kwargs.get("max_episode_steps", 1000)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("gait horizon must be a positive integer")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("gait panel episodes must be a positive integer")
    if isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0:
        raise ValueError("gait panel seed must be a nonnegative integer")
    if not math.isfinite(settle_s) or settle_s < 0:
        raise ValueError("gait settle_s must be finite and nonnegative")
    direction = normalized_direction(direction_xy)
    env = species_cfg.env_class(**env_kwargs)
    try:
        validate_environment_plant(env, current_plant_identity(species_cfg.species), artifact="gait protocol")
        if horizon * float(env.dt) <= settle_s:
            raise ValueError("gait settling exclusion must leave a positive analysis window")
        return measurement_protocol(
            species_cfg.species,
            protocol or GaitProtocol(),
            settle_s=settle_s,
            direction_xy=direction,
            horizon=horizon,
            physics_dt_s=float(env.model.opt.timestep),
            control_dt_s=float(env.dt),
            episodes=episodes,
            seed_start=seed_start,
        )
    finally:
        env.close()
