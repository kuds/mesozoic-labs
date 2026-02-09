"""Mesozoic Labs -- dinosaur locomotion environments.

Environments are registered with Gymnasium when their modules are imported.
Use ``register_all()`` to register all species at once, or import
individual species to register only what you need::

    # Option 1: register all
    import environments
    environments.register_all()
    env = gymnasium.make("MesozoicLabs/Raptor-v0")

    # Option 2: import directly (auto-registers that species)
    from environments.velociraptor.envs import RaptorEnv
    env = RaptorEnv()
"""


def register_all():
    """Import all env modules, triggering their gym.register() calls."""
    from environments.velociraptor.envs import raptor_env  # noqa: F401
    from environments.brachiosaurus.envs import brachio_env  # noqa: F401
    from environments.trex.envs import trex_env  # noqa: F401
