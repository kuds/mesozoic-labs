"""Shared six-heading paired rollouts for the heading-free observation tests (plant_versions note 18).

``test_heading_free_observation.py`` runs them over a short window in the
shared legs of the test matrix, ``test_heading_free_rollouts.py`` over the
full horizon with an SB3 policy in the SB3 job's full depth.  One seed is
reset in a fresh environment per heading, its whole scene turned
(``SpawnYaw(heading, whole_scene=True)``, the heading probe's own turn), and
stepped under a deterministic policy; with a heading-free observation and
the elliptic friction cone every heading is the same episode, so each trace
must equal 0 degrees' to the contact solver's tolerance.

The tolerances against what was measured on these plants (MuJoCo 3.10.0,
seed 3042, the zero action and the fixed linear policy below, 250 steps):
actions 0 (the float32 actions agree exactly), joint state 2.9e-13, reward
2.9e-12, observation 1.2e-10.  ``ACTION_ATOL`` is
``policy_loading.ACTION_DELTA_ATOL``'s precedent and ``FLOAT32_BUDGET`` the
design's float32 budget for corresponding states.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from environments.compsognathus.envs.compsognathus_env import CompsognathusBiologicalEnv
from environments.shared.config import load_stage_config
from environments.shared.reporting.stance_report import SpawnYaw, _apply_spawn_yaw
from environments.trex.envs.trex_env import TRexEnv

#: The heading-free species and their environments.
HEADING_FREE_ENVS = {"trex": TRexEnv, "compsognathus": CompsognathusBiologicalEnv}
#: The certificate's heading grid, in degrees.
HEADINGS_DEG = (-90.0, -45.0, 0.0, 45.0, 90.0, 180.0)
FLOAT32_BUDGET = 2e-6
ACTION_ATOL = 1e-6
STATE_ATOL = 1e-9
REWARD_ATOL = 1e-9

#: ``make_policy(env, observation_dim)`` returns a deterministic ``act(observation) -> action``.
PolicyFactory = Callable[[Any, int], Callable[[np.ndarray], np.ndarray]]


def stance_env(species: str) -> Any:
    return HEADING_FREE_ENVS[species](**load_stage_config(species, "stance")["env_kwargs"])


def zero_policy(env: Any, observation_dim: int) -> Callable[[np.ndarray], np.ndarray]:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    return lambda observation: action


def linear_policy(env: Any, observation_dim: int) -> Callable[[np.ndarray], np.ndarray]:
    """A fixed deterministic linear policy, small enough (gain 0.003) to stand the whole horizon."""
    weights = np.random.default_rng(7).normal(scale=0.003, size=(env.action_space.shape[0], observation_dim))
    return lambda observation: np.tanh(weights @ np.clip(observation, -10.0, 10.0)).astype(np.float32)


def paired_traces(species: str, seed: int, make_policy: PolicyFactory, steps: int) -> dict[float, dict[str, Any]]:
    """*seed*'s episode at every heading of the grid, each in a fresh env with its whole scene turned."""
    traces: dict[float, dict[str, Any]] = {}
    for heading in HEADINGS_DEG:
        env = stance_env(species)
        try:
            env.reset(seed=seed)
            observation = _apply_spawn_yaw(env, SpawnYaw(heading, whole_scene=True))
            act = make_policy(env, observation.shape[0])
            record: dict[str, list[Any]] = {"actions": [], "joints": [], "rewards": [], "observations": []}
            end = None
            for _ in range(steps):
                action = np.asarray(act(observation), dtype=np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                record["actions"].append(action)
                record["joints"].append(np.concatenate([env.data.qpos[7:], env.data.qvel[6:]]))
                record["rewards"].append(reward)
                record["observations"].append(observation)
                if terminated or truncated:
                    end = (bool(terminated), bool(truncated), info.get("termination_reason"))
                    break
            traces[heading] = {key: np.asarray(values) for key, values in record.items()}
            traces[heading]["end"] = end
        finally:
            env.close()
    return traces


def assert_paired(traces: dict[float, dict[str, Any]]) -> dict[str, float]:
    """Every heading's trace equals 0 degrees' to solver tolerance; returns the worst differences."""
    reference = traces[0.0]
    worst = {"actions": 0.0, "joints": 0.0, "rewards": 0.0, "observations": 0.0}
    for heading, trace in traces.items():
        assert len(trace["actions"]) == len(reference["actions"]), heading
        assert trace["end"] == reference["end"], heading
        for key in worst:
            worst[key] = max(worst[key], float(np.max(np.abs(trace[key] - reference[key]))))
    assert worst["actions"] <= ACTION_ATOL, worst
    assert worst["joints"] <= STATE_ATOL, worst
    assert worst["rewards"] <= REWARD_ATOL, worst
    assert worst["observations"] <= FLOAT32_BUDGET, worst
    return worst
