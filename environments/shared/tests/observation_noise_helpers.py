"""Test-only subclasses of every manifest model's env that opt into the observation noise.

No committed model lists the noise keys yet, so the shared noise tests run on
subclasses that do: :func:`with_observation_noise` lists both keys after the
species' own constructor parameters (so ``inspect.signature`` and the task
fingerprint see the species' defaults and the two keys) and declares the
opt-in carve-out at off value ``None``,
``("observation_noise_std", None, OBSERVATION_NOISE_ENV_KEYS)``, as a species
that opts in declares it (design record
``docs/TRICERATOPS_ROBOT_PORT_2026_11.md`` §8.3).  The subclass installs the
noise through ``BaseDinoEnv._configure_observation_noise`` after the species'
own ``__init__``, which does not forward the keys, as the delay's and the
randomisation's test subclasses do (``actuation_delay_helpers.py``,
``episode_physics_helpers.py``).

Each subclass is named ``Noisy<species class name>`` and bound to that very
name in this module, so ``pickle`` finds it by reference;
:func:`with_observation_noise` returns the same class for the same species
class.  :data:`NOISY_ENV_CLASSES` holds one per manifest model.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from environments.shared import species_registry
from environments.shared.model_id import ModelId
from environments.shared.observation_noise import OBSERVATION_NOISE_ENV_KEYS
from environments.shared.species_names import model_ids

#: The carve-out an opted-in class declares in ``OPT_IN_TASK_KWARGS``.
OBSERVATION_NOISE_CARVE_OUT = ("observation_noise_std", None, OBSERVATION_NOISE_ENV_KEYS)


@functools.cache
def with_observation_noise(env_class: Any) -> Any:
    """A subclass of *env_class* that takes ``observation_noise_std`` (default ``None``) and ``observation_noise_eval``."""

    def __init__(
        self: Any,
        *args: Any,
        observation_noise_std: Any = None,
        observation_noise_eval: Any = False,
        **kwargs: Any,
    ) -> None:
        env_class.__init__(self, *args, **kwargs)
        self._configure_observation_noise(observation_noise_std, observation_noise_eval)

    base = inspect.signature(env_class.__init__)
    keyword = inspect.Parameter.KEYWORD_ONLY
    __init__.__signature__ = base.replace(  # type: ignore[attr-defined]
        parameters=[
            *base.parameters.values(),
            inspect.Parameter("observation_noise_std", keyword, default=None),
            inspect.Parameter("observation_noise_eval", keyword, default=False),
        ]
    )
    name = f"Noisy{env_class.__name__}"
    cls = type(
        name,
        (env_class,),
        {
            "__init__": __init__,
            "__module__": __name__,
            "__qualname__": name,
            "OPT_IN_TASK_KWARGS": (*getattr(env_class, "OPT_IN_TASK_KWARGS", ()), OBSERVATION_NOISE_CARVE_OUT),
        },
    )
    globals()[name] = cls
    return cls


#: Every manifest model's opted-in test class, each bound here under its own ``__name__``.
NOISY_ENV_CLASSES: dict[ModelId, Any] = {
    model_id: with_observation_noise(species_registry.env_class(model_id)) for model_id in model_ids()
}
