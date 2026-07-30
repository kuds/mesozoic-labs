"""Optional Stable-Baselines3 import shim.

SB3 is an optional dependency (the ``train`` extra), so every callback module
subclasses whatever this resolves ``BaseCallback`` to.  Without SB3 installed
the names fall back to ``object`` and ``_SB3_AVAILABLE`` is ``False``, which
lets the curriculum manager itself be imported and tested on a bare install.
"""

from __future__ import annotations

try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv

    _SB3_AVAILABLE = True
except ImportError:
    BaseCallback = object  # type: ignore[misc,assignment]
    VecEnv = object  # type: ignore[misc,assignment]
    _SB3_AVAILABLE = False

__all__ = ["BaseCallback", "VecEnv", "_SB3_AVAILABLE"]
