"""Mesozoic Labs -- dinosaur locomotion environments.

All environments are auto-registered with Gymnasium on import::

    import environments  # registers all species
    env = gymnasium.make("MesozoicLabs/VelociraptorMongoliensis-Anatomical-v0")

Individual species can also be imported directly::

    from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv
    env = VelociraptorMongoliensisAnatomicalEnv()

Shared utilities (config, curriculum, metrics) are available even when
gymnasium/mujoco are not installed::

    from environments.shared.curriculum import RewardRampCallback
"""

import logging as _logging

try:
    from environments.brachiosaurus_altithorax.anatomical.env import brachio_env as _brachio_env  # noqa: F401
    from environments.compsognathus_longipes.anatomical.env import compsognathus_env as _compsognathus_env  # noqa: F401
    from environments.dibothrosuchus_elaphros.anatomical.env import (
        dibothrosuchus_env as _dibothrosuchus_env,  # noqa: F401
    )
    from environments.tyrannosaurus_rex.anatomical.env import trex_env as _trex_env  # noqa: F401
    from environments.velociraptor_mongoliensis.anatomical.env import raptor_env as _raptor_env  # noqa: F401
except ImportError:
    _logging.getLogger(__name__).debug(
        "Species environments not loaded (gymnasium/mujoco may not be installed)",
    )
