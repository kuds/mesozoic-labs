"""The repository root every repository-relative path derives from.

The one place it is computed from ``__file__``; a module that needs the root
imports :data:`REPOSITORY_ROOT` from here.  ``test_repository_root.py``
refuses a second computation, with two listed exceptions: the scripts'
``sys.path`` bootstraps (they run before ``environments`` is importable) and
five older anchors of the ``configs`` directory kept under their own names
(two of them, ``stage_manifest._CONFIGS_DIR`` and
``recovery_calibration.CONFIGS_ROOT``, are patched by tests).

``plant_contract.constants`` rebinds the root as its own attribute and stays
the patch point for the plant contract: its consumers read
``constants.REPOSITORY_ROOT`` at call time, and the byte-hashed
``behavior_env.py`` and ``terrain_sampling.py`` import it as
``plant_contract.REPOSITORY_ROOT``.  The value enters repository-relative
paths that ``behavior_identity`` and the plant source closure hash, so it
must not change.

Expressed via a named anchor (``environments/shared``, this module's own
directory) rather than a bare ``parents[N]`` count, because a raw count
silently changes meaning when a module moves to a different depth and stays
byte-identical while doing so.  ``test_repository_root_resolves_to_the_repository``
pins the result.  Standard library only, so any module may import it.
"""

from __future__ import annotations

from pathlib import Path

#: ``environments/shared``, this module's directory.
SHARED_ROOT = Path(__file__).resolve().parent
#: The repository checkout: ``environments/shared`` two levels up.
REPOSITORY_ROOT = SHARED_ROOT.parents[1]
