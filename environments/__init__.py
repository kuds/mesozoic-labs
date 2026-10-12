"""Mesozoic Labs -- dinosaur locomotion environments.

Every model is a (species, embodiment) pair (decision D-D33), and every one
in ``configs/species_manifest.toml`` is registered with Gymnasium on import,
under its built ID and its manifest entry point; no environment module is
imported until ``gymnasium.make`` needs it::

    import environments  # registers every model
    env = gymnasium.make("MesozoicLabs/VelociraptorMongoliensis-Anatomical-v0")

Each environment class can also be imported from its one module::

    from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv
    env = VelociraptorMongoliensisAnatomicalEnv()

No registration sets ``max_episode_steps``: every environment truncates at its
own (``BaseDinoEnv``).  Shared utilities (config, curriculum, metrics) are
available even when gymnasium/mujoco are not installed::

    from environments.shared.curriculum import RewardRampCallback
"""

import logging as _logging

try:
    import gymnasium as _gymnasium
except ImportError:
    _logging.getLogger(__name__).debug("Models not registered (gymnasium is not installed)")
else:
    from environments.shared.species_names import model_entry as _model_entry
    from environments.shared.species_names import model_ids as _model_ids

    for _model_id in _model_ids():
        _gymnasium.register(id=_model_id.gym_id, entry_point=str(_model_entry(_model_id)["env_entrypoint"]))
