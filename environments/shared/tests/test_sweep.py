"""Tests for the hyperparameter sweep tool's pure functions.

This module has been split into focused test files for maintainability:

- :mod:`test_sweep_submit` — job submission, machine validation, accelerator normalization
- :mod:`test_sweep_search_space` — search space loading and resolution
- :mod:`test_sweep_state` — sweep state persistence (save/load)
- :mod:`test_sweep_results` — trial result collection, CSV export, model path resolution
- :mod:`test_sweep_constants` — presets and exceptions
- :mod:`test_sweep_trial` — HPT arg parsing and override conversion
- :mod:`test_sweep_orchestration` — credential refresh, dedup, dry-run helpers
- :mod:`test_sweep_ray_search_space` — build_search_space and save_search_space
- :mod:`test_sweep_reporting` — stage artifact generation (reporting integration)

All tests are importable from the original location via this compatibility shim.
Run ``pytest environments/shared/tests/test_sweep*.py`` to execute all sweep tests.
"""

# Re-export all test classes so ``pytest test_sweep.py`` still discovers them.
from .test_sweep_constants import *  # noqa: F401, F403
from .test_sweep_orchestration import *  # noqa: F401, F403
from .test_sweep_ray_search_space import *  # noqa: F401, F403
from .test_sweep_reporting import *  # noqa: F401, F403
from .test_sweep_results import *  # noqa: F401, F403
from .test_sweep_search_space import *  # noqa: F401, F403
from .test_sweep_state import *  # noqa: F401, F403
from .test_sweep_submit import *  # noqa: F401, F403
from .test_sweep_trial import *  # noqa: F401, F403
