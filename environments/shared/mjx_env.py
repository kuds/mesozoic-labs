"""FROZEN (D-D17): the MJX observation interface the plant contract hashes and probes.

Nothing trains on MJX. D-D17 retired the JAX/MJX runtime (cleanup PR-B; it is
recoverable from the ``0.3.8`` tag or PR-B's first parent). trex, velociraptor,
brachiosaurus and dibothrosuchus still declare ``jax-mjx``, so
``plant_contract/policy_layer.py`` hashes the tokens of ``build_mjx_observation``
into their policy-interface digests, and its MJX probe executes it on the
``environments/<species>/mjx_config.py`` registrations whenever a plant identity
is computed with backend parity: on every SB3 training and evaluation run
(``validate_environment_plant``, from ``train_base`` and ``evaluation``) and in
``plant_contract --check``.

Do not edit, reformat or move ``build_mjx_observation``. A token change (a
refactor, an annotation, an f-string, a formatter release) moves four species'
plant identities and every task digest built on them; comments and docstrings
are not hashed. ``test_plant_contract_frozen_mjx.py`` names the function that
moved. The core goes once all four species have declared themselves SB3-only,
each inside a ``policy_interface_revision`` bump (docs/CLEANUP_PLAN_2026_09.md §4.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    # The retired MJXDinoEnv's config dataclass. build_mjx_observation's hashed
    # signature still names it; the plant contract passes a Mapping.
    MJXEnvConfig = Any

_SPECIES_CONFIGS: dict[str, dict[str, Any]] = {}


def register_species_mjx(species: str, **kwargs: Any) -> None:
    """Register a species configuration for MJX environments.

    Called from per-species ``mjx_config.py`` modules.
    """
    _SPECIES_CONFIGS[species] = kwargs


def build_mjx_observation(
    data: Any,
    target_pos: Any,
    config: MJXEnvConfig | Mapping[str, Any],
    command: Any = None,
) -> Any:
    """Build the production MJX policy observation from registered ABI data.

    The plant contract executes and fingerprints this small function directly,
    while both reset and step call the same implementation.  Keep reward and
    termination logic outside it so those changes do not invalidate a policy's
    observation/action interface.  ``command`` is the body-relative
    (v_x, v_y, yaw_rate) segment appended LAST (BEHAVIOR_RECIPES_PLAN §4.6);
    ``None`` means zeros.
    """
    from .obs_functions import SensorLayout, build_bipedal_obs, build_quadruped_obs

    def value(name: str, default: Any = None) -> Any:
        if isinstance(config, Mapping):
            return config[name] if name in config else default
        return getattr(config, name, default)

    body_ids = value("body_ids")
    # Dispatch on the registered root body, not on a species allow-list: every
    # quadruped registers a "torso" root and every biped a "pelvis" root, so a
    # new species is admitted by its own mjx_config rather than by editing this
    # function.  Editing it would re-fingerprint the policy interface of every
    # species that already shipped.
    quadrupedal = "torso" in body_ids
    root_name = "torso" if quadrupedal else "pelvis"
    root_body_id = int(body_ids[root_name])
    sensor_layout = SensorLayout(
        gyro_start=int(value("sensor_gyro_start")),
        accel_start=int(value("sensor_accel_start")),
        quat_start=int(value("sensor_quat_start")),
        foot_indices=tuple(int(index) for index in value("sensor_foot_indices")),
        foot_aux_indices=tuple(
            tuple(int(index) for index in group) for group in (value("sensor_foot_aux_indices", ()) or ())
        ),
    )
    observation_builder = build_quadruped_obs if quadrupedal else build_bipedal_obs
    return observation_builder(
        data.qpos,
        data.qvel,
        data.sensordata,
        data.xpos[root_body_id],
        target_pos,
        sensor_layout,
        command=command,
    )
