"""Mesozoic Labs -- dinosaur locomotion environments.

All environments are auto-registered with Gymnasium on import::

    import environments  # registers all species
    env = gymnasium.make("MesozoicLabs/Raptor-v0")

Individual species can also be imported directly::

    from environments.velociraptor.envs import RaptorEnv
    env = RaptorEnv()

Shared utilities (config, curriculum, metrics) are available even when
gymnasium/mujoco are not installed::

    from environments.shared.curriculum import RewardRampCallback
"""

import logging as _logging

try:
    from environments.brachiosaurus.envs import brachio_env as _brachio_env  # noqa: F401
    from environments.trex.envs import trex_env as _trex_env  # noqa: F401
    from environments.velociraptor.envs import raptor_env as _raptor_env  # noqa: F401
except ImportError:
    _logging.getLogger(__name__).debug(
        "Species environments not loaded (gymnasium/mujoco may not be installed)",
    )


def register_all():
    """No-op kept for backwards compatibility. Envs are registered on import."""
