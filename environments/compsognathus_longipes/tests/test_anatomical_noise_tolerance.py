"""The anatomical statue survives PPO's initial exploration noise (the noise cliff, D-D26).

On the physics r1 plant with the linear home-keyframe residual, the zero-action
statue fell within about ten steps under zero-mean action noise at the stance
recipe's initial sigma (0.135, log_std_init -2.0): PPO never sampled the quiet
stance, and both certified r1 stances were noise-robust tiptoe marches.  Physics
r2 (its toe armature included) with the soft-cubic leg residual moved that
cliff past the recipe's sigma: on the stance task, seeds 3042+i with noise
stream default_rng(123456 + 17 i), the statue reaches the horizon on 40/40
episodes at sigma 0.10, on 39/40 at 0.135 and on 0/40 at 0.20 (median 180
steps; 18 of the first 20 end on the tail tip, 2 on tilt).  At 0.135 the
survival rate is about 0.94 on every seed block measured: 37/40 on seeds
3082-3121 and on 7042-7081, 38/40 on 9042-9081 with an independent noise
stream (default_rng(700001 + 31 i)), 151/160 pooled; all nine falls, at steps
174-938, end on the tail tip.  This pins both sides cheaply: sigma 0.10 must
reach the horizon on the first 5 episodes, and sigma 0.135 on at least 10 of
the first 12 (0.83, the review's "at least 0.8 of episodes").  It keeps 11
there, where the r1 plant kept 0/20; at the pooled rate a 12-episode block
keeps 10 or more with probability 0.97, so the pin leaves one fall of margin
under the measured rate rather than sitting on it.  The support-geom coverage term
is switched off here: it reads contacts and changes no trajectory, and skipping
its per-substep copies keeps the test short.
"""

from __future__ import annotations

import numpy as np
import pytest

from environments.compsognathus.envs import CompsognathusBiologicalEnv
from environments.shared.config import load_stage_config


def _survivors(sigma: float, episodes: int) -> int:
    kwargs = {**load_stage_config("compsognathus", "stance")["env_kwargs"], "support_geom_coverage_weight": 0.0}
    env = CompsognathusBiologicalEnv(**kwargs)
    survivors = 0
    try:
        for index in range(episodes):
            rng = np.random.default_rng(123456 + 17 * index)
            env.reset(seed=3042 + index)
            while True:
                action = np.clip(sigma * rng.standard_normal(env.action_space.shape), -1.0, 1.0)
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            survivors += int(truncated and not terminated)
    finally:
        env.close()
    return survivors


@pytest.mark.parametrize("sigma, episodes, minimum", [(0.10, 5, 5), (0.135, 12, 10)])
def test_the_statue_survives_the_recipes_initial_exploration_noise(sigma, episodes, minimum):
    assert _survivors(sigma, episodes) >= minimum
