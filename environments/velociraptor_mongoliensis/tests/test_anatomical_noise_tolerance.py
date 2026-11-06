"""The statue survives PPO's initial exploration noise: the stance recipe sits at the r4 noise cliff (2026-11-06).

Every velociraptor stance trained at SB3's default action std 1.0, where the
stochastic policy's flight is the exploration noise: on the physics r4 statue,
under clip(N(0, 1)) commands no foot is down on 49% of the steps.  The
maintainer's decision of 2026-11-06 ports the T-Rex's quiet start (its D-D27
``log_std_init``) at this plant's own value, set by the rule the anatomical Compsognathus's -2.0 follows
(D-D26): start at the measured cliff's edge.  On the stance task, seeds
3042+i with noise stream default_rng(123456 + 17 i) (the Compsognathus
test's protocol), the zero-action statue with that noise on its command
reaches the horizon on 40/40 episodes at sigma 0.05 and 0.06, on 37/40 at
0.07 (seeds 3044, 3063 and 3073 end on the tail tip at steps 295, 220 and
970), on 32/40 at 0.08, 7/40 at 0.10 and 0/40 at 0.135; at 0.07, 37/40 on
seeds 3082-3121 with the same stream and 40/40 on 7042-7081 with an
independent one (default_rng(700001 + 31 i)): 114/120 pooled.  Every fall from
0.07 to 0.135 ends on the tail: the home-residual map's asymmetry turns
zero-mean noise into a backward lean.

This pins the cliff on both sides of the recipe's sigma, exp(-2.66) = 0.070,
cheaply: sigma 0.05 must reach the horizon on the first 5 episodes, the
recipe's own sigma on at least 10 of the first 12 (it keeps 11; at the pooled
rate a 12-episode block keeps 10 or more with probability 0.98), sigma 0.10 on
at most 5 of the first 12 (it keeps 2, seeds 3042 and 3046; at its 7/40 rate a
12-episode block keeps 6 or more with probability 0.01) and sigma 0.135
on none of the first 5.  A plant change that moves the cliff down past 0.07,
or up far enough that half the statue's episodes survive 0.10, fails here and
moves ``log_std_init`` with it.  The reward terms read no
action noise into the trajectory, so the stance [env] as committed is rolled.
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
    assert math.exp(policy_kwargs["log_std_init"]) == pytest.approx(0.07, abs=0.001)


@pytest.mark.parametrize("sigma, episodes, minimum", [(0.05, 5, 5), (0.07, 12, 10)])
def test_the_statue_survives_the_recipes_initial_exploration_noise(sigma, episodes, minimum):
    assert _survivors(sigma, episodes) >= minimum


def test_the_cliff_is_where_the_recipe_was_set():
    """Past the cliff the statue falls: a plant change that moves it re-opens the start's sigma."""
    assert _survivors(0.10, 12) <= 5
    assert _survivors(0.135, 5) == 0
