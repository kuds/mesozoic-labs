"""Interactive harnesses shared by the per-species ``scripts/``.

These are the implementations behind ``scripts/test_env.py``,
``scripts/test_actuators.py``, and ``scripts/view_model.py`` for every
species — hand-run smoke checks and MuJoCo viewers that print to stdout and
open windows, not automated tests.  The automated suite lives in
``environments/*/tests/`` and ``environments/shared/tests/``.

Nothing here is named ``test_*``: these modules used to sit in the package
root as ``test_env_base.py`` / ``test_actuators_base.py``, where pytest
collected their functions as tests and errored on the unfillable ``env_class``
and ``cfg`` arguments whenever anyone ran pytest against a path that reached
them.

* :mod:`~environments.shared.harnesses.env_smoke` — environment smoke checks
  (spaces, reset/step, rollouts, reward keys, determinism, bounds)
* :mod:`~environments.shared.harnesses.actuators` — drive each actuator with a
  sinusoid in the viewer to eyeball joint ranges and gains
* :mod:`~environments.shared.harnesses.viewer` — passive viewer for MJCF
  iteration
"""

from __future__ import annotations

from .actuators import ActuatorTestConfig, run_actuator_test
from .env_smoke import run_env_tests
from .viewer import ViewerConfig, view_model

__all__ = [
    "ActuatorTestConfig",
    "ViewerConfig",
    "run_actuator_test",
    "run_env_tests",
    "view_model",
]
