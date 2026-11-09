"""The statue survives PPO's initial exploration noise: the stance recipe sits at the noise cliff (2026-11-06; r5 2026-11-09).

Every velociraptor stance trained at SB3's default action std 1.0, where the
stochastic policy's flight is the exploration noise: on the physics r4 statue,
under clip(N(0, 1)) commands no foot is down on 49% of the steps (r5: 36%).
The maintainer's decision of 2026-11-06 ports the T-Rex's quiet start (its
D-D27 ``log_std_init``) at this plant's own value, set by the rule the
anatomical Compsognathus's -2.0 follows (D-D26): start at the measured cliff's
edge.  On the stance task, seeds 3042+i with noise stream
default_rng(123456 + 17 i) (the Compsognathus test's protocol), the
zero-action statue of the life-size plant (physics r5, the maintainer's
decision of 2026-11-08) with that noise on its command reaches the horizon on
40/40 episodes at every sigma from 0.05 to 0.09, on 38/40 at 0.10 (seeds 3043
and 3078 nosedive at steps 637 and 410), 37/40 at 0.11, 33/40 at 0.12, 19/40
at 0.135, 8/40 at 0.15, 1/40 at 0.17 and 0/40 at 0.20; at 0.10, 37/40 on
seeds 3082-3121 with the same stream and 40/40 on 7042-7081 with an
independent one (default_rng(700001 + 31 i)): 115/120 pooled (r4's 0.07:
114/120).  Every fall from 0.10 to 0.12 is a nosedive (20 of the 21 at 0.135;
r4's fell on the tail); the recipe moved from exp(-2.66) = 0.070 to
exp(-2.30) = 0.100.

This pins the cliff on both sides of the recipe's sigma cheaply: sigma 0.07
(r4's start) must reach the horizon on the first 5 episodes, the recipe's own
sigma on at least 10 of the first 12 (it keeps 11; at the pooled rate a
12-episode block keeps 10 or more with probability 0.99), sigma 0.15 on at
most 5 of the first 12 (it keeps 1, seed 3045; at its 8/40 rate a 12-episode
block keeps 6 or more with probability 0.02) and sigma 0.20 on none of the
first 5.  A plant change that moves the cliff down past 0.10, or up far
enough that half the statue's episodes survive 0.15, fails here and moves
``log_std_init`` with it.  The reward terms read no action noise into the
trajectory, so the stance [env] as committed is rolled.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

MODEL_ID = ModelId("velociraptor_mongoliensis", "anatomical")


def _survivors(sigma: float, episodes: int) -> int:
    env = VelociraptorMongoliensisAnatomicalEnv(**load_stage_config(MODEL_ID, "stance")["env_kwargs"])
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


def test_the_recipe_starts_at_the_pinned_sigma():
    """``log_std_init`` is the sigma this module pins; move both together."""
    policy_kwargs = load_stage_config(MODEL_ID, "stance")["ppo_kwargs"]["policy_kwargs"]
    assert math.exp(policy_kwargs["log_std_init"]) == pytest.approx(0.10, abs=0.001)


@pytest.mark.parametrize("sigma, episodes, minimum", [(0.07, 5, 5), (0.10, 12, 10)])
def test_the_statue_survives_the_recipes_initial_exploration_noise(sigma, episodes, minimum):
    assert _survivors(sigma, episodes) >= minimum


def test_the_cliff_is_where_the_recipe_was_set():
    """Past the cliff the statue falls: a plant change that moves it re-opens the start's sigma."""
    assert _survivors(0.15, 12) <= 5
    assert _survivors(0.20, 5) == 0
