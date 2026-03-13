"""Per-species, per-stage search spaces for Ray Tune sweeps.

Each search space is defined in the generic dict format used by
``constants.py`` (``{"type": "double", "min": ..., "max": ..., "scale": ...}``).
The ``to_ray_tune()`` converter translates these into ``ray.tune`` distribution
objects so the same definitions can be reused by both Vertex AI and Ray Tune.

Call ``build_search_space(species, stage, algorithm)`` to get a ready-to-use
``dict[str, ray.tune.sample.Domain]``.
"""

from __future__ import annotations

from typing import Any

from .constants import _DEFAULT_PPO_SEARCH_SPACE, _DEFAULT_SAC_SEARCH_SPACE

# ── Shared algorithm search spaces ──────────────────────────────────────────
# Extended from _DEFAULT_PPO_SEARCH_SPACE with n_epochs and net_arch.

_PPO_ALGO_SPACE: dict[str, dict[str, Any]] = {
    **{k: v for k, v in _DEFAULT_PPO_SEARCH_SPACE.items()},
    "ppo_n_epochs": {"type": "discrete", "values": [3, 6, 10]},
    "ppo_net_arch": {
        "type": "categorical",
        "values": ["small", "medium", "large", "deep", "tapered", "deep_tapered"],
    },
}

_SAC_ALGO_SPACE: dict[str, dict[str, Any]] = {
    **{k: v for k, v in _DEFAULT_SAC_SEARCH_SPACE.items()},
    "sac_net_arch": {
        "type": "categorical",
        "values": ["small", "medium", "large", "tapered", "deep_tapered"],
    },
}

# ── Per-species, per-stage environment reward search spaces ─────────────────

# --- Stage 1: Balance ---
_STAGE1_COMMON: dict[str, dict[str, Any]] = {
    "env_alive_bonus": {"type": "double", "min": 1.0, "max": 5.0, "scale": "linear"},
    "env_energy_penalty_weight": {"type": "double", "min": 0.01, "max": 0.1, "scale": "log"},
}

_STAGE1_BY_SPECIES: dict[str, dict[str, dict[str, Any]]] = {
    "velociraptor": {
        **_STAGE1_COMMON,
        "env_posture_weight": {"type": "double", "min": 0.5, "max": 3.0, "scale": "linear"},
        "env_nosedive_weight": {"type": "double", "min": 0.5, "max": 3.0, "scale": "linear"},
        "env_drift_penalty_weight": {"type": "double", "min": 0.0, "max": 0.5, "scale": "linear"},
        "env_spin_penalty_weight": {"type": "double", "min": 0.0, "max": 0.5, "scale": "linear"},
    },
    "trex": {
        **_STAGE1_COMMON,
        "env_posture_weight": {"type": "double", "min": 0.5, "max": 4.0, "scale": "linear"},
        "env_nosedive_weight": {"type": "double", "min": 0.5, "max": 5.0, "scale": "linear"},
        "env_height_weight": {"type": "double", "min": 0.5, "max": 2.0, "scale": "linear"},
        "env_spin_penalty_weight": {"type": "double", "min": 0.0, "max": 0.5, "scale": "linear"},
        "env_smoothness_weight": {"type": "double", "min": 0.02, "max": 0.2, "scale": "linear"},
    },
    "brachiosaurus": {
        **_STAGE1_COMMON,
        "env_gait_stability_weight": {"type": "double", "min": 0.02, "max": 0.2, "scale": "linear"},
    },
}

# --- Stage 2: Locomotion ---
_STAGE2_COMMON: dict[str, dict[str, Any]] = {
    "env_forward_vel_weight": {"type": "double", "min": 1.0, "max": 4.0, "scale": "linear"},
    "env_forward_vel_max": {"type": "double", "min": 2.0, "max": 5.0, "scale": "linear"},
    "env_alive_bonus": {"type": "double", "min": 0.1, "max": 2.0, "scale": "linear"},
    "env_energy_penalty_weight": {"type": "double", "min": 0.001, "max": 0.01, "scale": "log"},
    "env_heading_weight": {"type": "double", "min": 0.0, "max": 0.6, "scale": "linear"},
    "env_lateral_penalty_weight": {"type": "double", "min": 0.0, "max": 0.3, "scale": "linear"},
    "curriculum_warmup_timesteps": {"type": "discrete", "values": [50000, 100000, 200000, 300000]},
    "curriculum_warmup_clip_range": {"type": "double", "min": 0.01, "max": 0.05, "scale": "linear"},
    "curriculum_warmup_ent_coef": {"type": "double", "min": 0.005, "max": 0.05, "scale": "log"},
    "curriculum_ramp_timesteps": {"type": "discrete", "values": [200000, 500000, 1000000, 2000000]},
    "curriculum_ramp_start_value": {"type": "double", "min": 0.05, "max": 0.3, "scale": "linear"},
}

_STAGE2_BY_SPECIES: dict[str, dict[str, dict[str, Any]]] = {
    "velociraptor": {
        **_STAGE2_COMMON,
        "env_posture_weight": {"type": "double", "min": 0.0, "max": 0.5, "scale": "linear"},
        "env_nosedive_weight": {"type": "double", "min": 0.1, "max": 0.8, "scale": "linear"},
    },
    "trex": {
        **_STAGE2_COMMON,
        "env_posture_weight": {"type": "double", "min": 0.1, "max": 1.0, "scale": "linear"},
        "env_nosedive_weight": {"type": "double", "min": 0.3, "max": 2.0, "scale": "linear"},
        "env_height_weight": {"type": "double", "min": 0.5, "max": 2.0, "scale": "linear"},
    },
    "brachiosaurus": {
        **_STAGE2_COMMON,
        "env_gait_stability_weight": {"type": "double", "min": 0.01, "max": 0.1, "scale": "linear"},
    },
}

# --- Stage 3: Attack / Goal ---
_STAGE3_CURRICULUM: dict[str, dict[str, Any]] = {
    "curriculum_warmup_timesteps": {"type": "discrete", "values": [50000, 100000, 200000, 300000]},
    "curriculum_warmup_clip_range": {"type": "double", "min": 0.01, "max": 0.05, "scale": "linear"},
    "curriculum_warmup_ent_coef": {"type": "double", "min": 0.005, "max": 0.05, "scale": "log"},
    "curriculum_ramp_timesteps": {"type": "discrete", "values": [500000, 1000000, 2000000, 4000000]},
    "curriculum_ramp_start_value": {"type": "double", "min": 0.05, "max": 0.3, "scale": "linear"},
}

_STAGE3_COMMON: dict[str, dict[str, Any]] = {
    "env_forward_vel_weight": {"type": "double", "min": 0.1, "max": 1.0, "scale": "linear"},
    "env_alive_bonus": {"type": "double", "min": 0.01, "max": 0.1, "scale": "log"},
    "env_posture_weight": {"type": "double", "min": 0.0, "max": 0.3, "scale": "linear"},
    "env_nosedive_weight": {"type": "double", "min": 0.1, "max": 0.6, "scale": "linear"},
    "env_heading_weight": {"type": "double", "min": 0.1, "max": 0.6, "scale": "linear"},
    **_STAGE3_CURRICULUM,
}

_STAGE3_BY_SPECIES: dict[str, dict[str, dict[str, Any]]] = {
    "velociraptor": {
        **_STAGE3_COMMON,
        "env_strike_bonus": {"type": "double", "min": 10.0, "max": 100.0, "scale": "log"},
        "env_strike_approach_weight": {"type": "double", "min": 1.0, "max": 5.0, "scale": "linear"},
        "env_strike_proximity_weight": {"type": "double", "min": 0.1, "max": 1.0, "scale": "linear"},
        "env_strike_claw_proximity_weight": {"type": "double", "min": 0.5, "max": 4.0, "scale": "linear"},
    },
    "trex": {
        **_STAGE3_COMMON,
        "env_bite_bonus": {"type": "double", "min": 10.0, "max": 100.0, "scale": "log"},
        "env_bite_approach_weight": {"type": "double", "min": 1.0, "max": 5.0, "scale": "linear"},
        "env_lateral_penalty_weight": {"type": "double", "min": 0.1, "max": 0.5, "scale": "linear"},
    },
    "brachiosaurus": {
        "env_forward_vel_weight": {"type": "double", "min": 0.5, "max": 2.0, "scale": "linear"},
        "env_alive_bonus": {"type": "double", "min": 0.05, "max": 0.3, "scale": "linear"},
        "env_energy_penalty_weight": {"type": "double", "min": 0.0005, "max": 0.005, "scale": "log"},
        "env_gait_stability_weight": {"type": "double", "min": 0.005, "max": 0.05, "scale": "linear"},
        "env_food_reach_bonus": {"type": "double", "min": 5.0, "max": 50.0, "scale": "log"},
        "env_food_reach_threshold": {"type": "double", "min": 0.3, "max": 0.8, "scale": "linear"},
        "env_food_approach_weight": {"type": "double", "min": 0.5, "max": 3.0, "scale": "linear"},
        **_STAGE3_CURRICULUM,
    },
}

_STAGES_BY_SPECIES: dict[int, dict[str, dict[str, dict[str, Any]]]] = {
    1: _STAGE1_BY_SPECIES,
    2: _STAGE2_BY_SPECIES,
    3: _STAGE3_BY_SPECIES,
}


def _get_env_search_space(species: str, stage: int) -> dict[str, dict[str, Any]]:
    """Return the environment reward search space for a species and stage."""
    stage_map = _STAGES_BY_SPECIES.get(stage)
    if stage_map is None:
        raise ValueError(f"Unknown stage: {stage}. Must be 1, 2, or 3.")
    # Fall back to velociraptor for unknown species
    return stage_map.get(species, stage_map["velociraptor"])


def build_search_space(
    species: str,
    stage: int,
    algorithm: str,
) -> dict[str, dict[str, Any]]:
    """Build the complete search space (algo + env) in generic dict format.

    Returns a dict compatible with ``constants.py`` format::

        {"param_name": {"type": "double", "min": ..., "max": ..., "scale": ...}, ...}

    Use ``to_ray_tune()`` to convert to Ray Tune distribution objects.
    """
    algo_space = _PPO_ALGO_SPACE if algorithm == "ppo" else _SAC_ALGO_SPACE
    env_space = _get_env_search_space(species, stage)
    return {**algo_space, **env_space}


def to_ray_tune(search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Convert a generic search-space dict to Ray Tune distribution objects.

    Supports the same parameter types as ``trial._build_parameter_spec``:

    - ``double`` with ``scale="log"`` → ``tune.loguniform(min, max)``
    - ``double`` with ``scale="linear"`` → ``tune.uniform(min, max)``
    - ``discrete`` → ``tune.choice(values)``
    - ``categorical`` → ``tune.choice(values)``
    """
    from ray import tune

    result: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec.get("type", "double")
        if kind == "double":
            if spec.get("scale") == "log":
                result[name] = tune.loguniform(spec["min"], spec["max"])
            else:
                result[name] = tune.uniform(spec["min"], spec["max"])
        elif kind == "discrete":
            result[name] = tune.choice(spec["values"])
        elif kind == "categorical":
            result[name] = tune.choice(spec["values"])
        else:
            raise ValueError(f"Unknown parameter type {kind!r} for {name}")
    return result
