"""The minimal gymnasium env the SB3 wiring tests build models and vec envs on.

It has no physics: the tests exercise callbacks, VecNormalize statistics and
checkpoint files around the env, never the env itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import gymnasium


def tiny_env_class(obs_dim: int = 1, info: dict[str, float] | None = None) -> type[gymnasium.Env]:
    """Return an env class with Box(-1, 1) float32 spaces: ``obs_dim`` observations, one action.

    Observations are zeros and rewards 0.0; no episode ends on its own. Every
    step returns a fresh copy of ``info`` (empty by default).
    """
    pytest.importorskip("gymnasium")
    import gymnasium as gym

    step_info = dict(info or {})

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, (obs_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(obs_dim, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(obs_dim, dtype=np.float32), 0.0, False, False, dict(step_info)

    return TinyEnv
