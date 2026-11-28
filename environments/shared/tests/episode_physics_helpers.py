"""Test-only subclasses of every manifest model's env that opt into the per-episode physics randomisation.

No committed model lists the ``randomize_*`` keys yet, so the shared
randomisation tests run on subclasses that do: :func:`with_episode_physics`
lists the six keys after the species' own constructor parameters (so
``inspect.signature`` and the task fingerprint see the species' defaults and
the six keys) and declares one opt-in carve-out per key at off value ``None``,
``(name, None, (name,))``, as a species that opts in declares them (design
record ``docs/TRICERATOPS_ROBOT_PORT_2026_11.md`` §8.3).  The subclass installs
the randomisation through ``BaseDinoEnv._configure_episode_physics`` after
the species' own ``__init__``, which does not forward the keys.

Each subclass is named ``Randomized<species class name>`` and bound to that
very name in this module, so ``pickle`` finds it by reference;
:func:`with_episode_physics` returns the same class for the same species class.
:data:`RANDOMIZED_ENV_CLASSES` holds one per manifest model.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from environments.shared import species_registry
from environments.shared.episode_physics import EPISODE_PHYSICS_ENV_KEYS
from environments.shared.model_id import ModelId
from environments.shared.species_names import model_ids

#: The carve-outs an opted-in class declares in ``OPT_IN_TASK_KWARGS``: each key alone, at off value ``None``.
EPISODE_PHYSICS_CARVE_OUTS = tuple((name, None, (name,)) for name in EPISODE_PHYSICS_ENV_KEYS)


@functools.cache
def with_episode_physics(env_class: Any) -> Any:
    """A subclass of *env_class* that takes the six ``randomize_*`` keys, each defaulting to ``None``."""

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        ranges = {name: kwargs.pop(name, None) for name in EPISODE_PHYSICS_ENV_KEYS}
        env_class.__init__(self, *args, **kwargs)
        self._configure_episode_physics(**ranges)

    base = inspect.signature(env_class.__init__)
    keyword = inspect.Parameter.KEYWORD_ONLY
    __init__.__signature__ = base.replace(  # type: ignore[attr-defined]
        parameters=[
            *base.parameters.values(),
            *(inspect.Parameter(name, keyword, default=None) for name in EPISODE_PHYSICS_ENV_KEYS),
        ]
    )
    name = f"Randomized{env_class.__name__}"
    cls = type(
        name,
        (env_class,),
        {
            "__init__": __init__,
            "__module__": __name__,
            "__qualname__": name,
            "OPT_IN_TASK_KWARGS": (*getattr(env_class, "OPT_IN_TASK_KWARGS", ()), *EPISODE_PHYSICS_CARVE_OUTS),
        },
    )
    globals()[name] = cls
    return cls


#: Every manifest model's opted-in test class, each bound here under its own ``__name__``.
RANDOMIZED_ENV_CLASSES: dict[ModelId, Any] = {
    model_id: with_episode_physics(species_registry.env_class(model_id)) for model_id in model_ids()
}
