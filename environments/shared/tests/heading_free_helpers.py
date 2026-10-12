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
3.0e-12, observation 1.2e-10.  ``ACTION_ATOL`` is
``policy_loading.ACTION_DELTA_ATOL``'s precedent and ``FLOAT32_BUDGET`` the
design's float32 budget for corresponding states.

Behind a network policy the loop can turn the rounding of a rotated state
into a one-ulp difference of a float32 action, which a stable loop then
carries to the end without growing; the ``CLOSED_LOOP_*`` bounds admit that
and nothing near a leak, which moves the observation by O(1) from the first
step.  Over the full horizon of seeds 3042-3043 behind the frozen PPO policy
of ``test_heading_free_rollouts``: with torch's default threads the actions
agree exactly and joint state to 4.5e-13; single-threaded (as earlier tests
in the SB3 job leave torch) the compsognathus's actions differ by one ulp
(7.5e-9) at some step, its joint state by 7.3e-8 at the end and its
observation by 9.5e-7 (one float32 ulp of a joint velocity).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from environments.compsognathus_longipes.anatomical.env import CompsognathusLongipesAnatomicalEnv
from environments.shared.config import load_stage_config
from environments.shared.curriculum.stance_heading_gate import HEADING_GRID_DEG
from environments.shared.model_id import ModelId
from environments.shared.reporting.stance_report import SpawnYaw, _apply_spawn_yaw
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv

#: The heading-free models and their environments.
HEADING_FREE_ENVS = {
    ModelId("tyrannosaurus_rex", "anatomical"): TyrannosaurusRexAnatomicalEnv,
    ModelId("compsognathus_longipes", "anatomical"): CompsognathusLongipesAnatomicalEnv,
}
#: The certificate's heading grid, in degrees (``whole_scene_yaw/v1``, D-D32).
HEADINGS_DEG = HEADING_GRID_DEG
FLOAT32_BUDGET = 2e-6
ACTION_ATOL = 1e-6
STATE_ATOL = 1e-9
REWARD_ATOL = 1e-9
CLOSED_LOOP_STATE_ATOL = 1e-6
CLOSED_LOOP_OBSERVATION_ATOL = 1e-5

#: ``make_policy(env, observation_dim)`` returns a deterministic ``act(observation) -> action``.
PolicyFactory = Callable[[Any, int], Callable[[np.ndarray], np.ndarray]]


def stance_env(model_id: ModelId) -> Any:
    return HEADING_FREE_ENVS[model_id](**load_stage_config(model_id, "stance")["env_kwargs"])


def zero_policy(env: Any, observation_dim: int) -> Callable[[np.ndarray], np.ndarray]:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    return lambda observation: action


def linear_policy(env: Any, observation_dim: int) -> Callable[[np.ndarray], np.ndarray]:
    """A fixed deterministic linear policy, small enough (gain 0.003) to stand the whole horizon."""
    weights = np.random.default_rng(7).normal(scale=0.003, size=(env.action_space.shape[0], observation_dim))
    return lambda observation: np.tanh(weights @ np.clip(observation, -10.0, 10.0)).astype(np.float32)


def paired_traces(model_id: ModelId, seed: int, make_policy: PolicyFactory, steps: int) -> dict[float, dict[str, Any]]:
    """*seed*'s episode at every heading of the grid, each in a fresh env with its whole scene turned."""
    traces: dict[float, dict[str, Any]] = {}
    for heading in HEADINGS_DEG:
        env = stance_env(model_id)
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


def assert_paired(
    traces: dict[float, dict[str, Any]],
    *,
    state_atol: float = STATE_ATOL,
    observation_atol: float = FLOAT32_BUDGET,
) -> dict[str, float]:
    """Every heading's trace equals 0 degrees' to solver tolerance; returns the worst differences.

    *state_atol* and *observation_atol* bound the joint state and the
    observation; a closed loop behind a network policy passes the
    ``CLOSED_LOOP_*`` bounds.
    """
    reference = traces[0.0]
    worst = {"actions": 0.0, "joints": 0.0, "rewards": 0.0, "observations": 0.0}
    for heading, trace in traces.items():
        assert len(trace["actions"]) == len(reference["actions"]), heading
        assert trace["end"] == reference["end"], heading
        for key in worst:
            worst[key] = max(worst[key], float(np.max(np.abs(trace[key] - reference[key]))))
    assert worst["actions"] <= ACTION_ATOL, worst
    assert worst["joints"] <= state_atol, worst
    assert worst["rewards"] <= REWARD_ATOL, worst
    assert worst["observations"] <= observation_atol, worst
    return worst
