"""Test-only subclasses of committed species envs that opt into the actuation delay.

No committed model lists the delay keys yet, so the shared delay tests run on
subclasses that do: :func:`with_actuation_delay` lists both keys after the
species' own constructor parameters (so ``inspect.signature`` and the task
fingerprint see the species' defaults and the two keys) and declares the
opt-in carve-out at off value ``None``, as a species that opts in declares it
(design record ``docs/TRICERATOPS_ROBOT_PORT_2026_11.md`` §8.3).  The subclass
installs the delay through ``BaseDinoEnv._configure_actuation_delay`` after
the species' own ``__init__``, which does not forward the keys.

Each subclass is named ``Delayed<species class name>`` and bound to that very
name in this module, so ``pickle`` finds it by reference (a ``SubprocVecEnv``
or a pickled env attribute needs that); :func:`with_actuation_delay` returns
the same class for the same species class.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from environments.compsognathus_longipes.robot.env import CompsognathusLongipesRobotEnv
from environments.shared.actuation_delay import ACTUATION_DELAY_ENV_KEYS
from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv


@functools.cache
def with_actuation_delay(env_class: Any) -> Any:
    """A subclass of *env_class* that takes ``actuation_delay_range_s`` and ``actuation_delay_eval_s``."""

    def __init__(
        self: Any,
        *args: Any,
        actuation_delay_range_s: Any = None,
        actuation_delay_eval_s: Any = None,
        **kwargs: Any,
    ) -> None:
        env_class.__init__(self, *args, **kwargs)
        self._configure_actuation_delay(actuation_delay_range_s, actuation_delay_eval_s)

    base = inspect.signature(env_class.__init__)
    keyword = inspect.Parameter.KEYWORD_ONLY
    __init__.__signature__ = base.replace(  # type: ignore[attr-defined]
        parameters=[
            *base.parameters.values(),
            *(inspect.Parameter(name, keyword, default=None) for name in ACTUATION_DELAY_ENV_KEYS),
        ]
    )
    name = f"Delayed{env_class.__name__}"
    return type(
        name,
        (env_class,),
        {
            "__init__": __init__,
            "__module__": __name__,
            "__qualname__": name,
            "OPT_IN_TASK_KWARGS": (
                *getattr(env_class, "OPT_IN_TASK_KWARGS", ()),
                ("actuation_delay_range_s", None, ACTUATION_DELAY_ENV_KEYS),
            ),
        },
    )


#: The Compsognathus robot (2 ms physics, ``frame_skip`` 10, home-keyframe residual), opted in.
DelayedCompsognathusLongipesRobotEnv = with_actuation_delay(CompsognathusLongipesRobotEnv)
#: The anatomical Velociraptor (2 ms physics, ``frame_skip`` 5, mirror-symmetric), opted in: the state mirror's tests.
DelayedVelociraptorMongoliensisAnatomicalEnv = with_actuation_delay(VelociraptorMongoliensisAnatomicalEnv)

#: Every opted-in test class above, each bound here under its own ``__name__``.
DELAYED_ENV_CLASSES = (DelayedCompsognathusLongipesRobotEnv, DelayedVelociraptorMongoliensisAnatomicalEnv)
