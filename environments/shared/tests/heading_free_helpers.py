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
carries to the end without growing.  The turn leaves float64 noise of about
1e-13; now and then that puts an observation element on the other side of a
float32 rounding boundary than at 0 degrees, the network reads an input one
ulp apart and, more rarely, returns an action element one ulp apart.  The
full-horizon comparison's ``CLOSED_LOOP_*`` bounds admit what the loop then
carries and nothing near a leak, which moves the observation by O(1) from
the first step: actions to ``ACTION_ATOL``, joint state and reward to 1e-6
(the reward's is ``test_heading_free_rollouts.METRIC_ATOL``'s precedent;
``REWARD_ATOL`` bounds the open loop only), and each observation element to
``float32_ulp_bound`` of 0 degrees' value at its step: four float32 ulps of
that value, or 1e-5 where that is larger.  Rounding to float32 moves a value
by at most half an ulp of its binade, so a float32 difference of more than
four ulps needs a float64 difference of at least 3.5 ulps (4 within a
binade).  The loop carried at most 0.95 ulp (below), so four ulps leave
about 3.7 times that, where two would leave 1.6.  The ulp term exceeds 1e-5
only on elements of magnitude 32 or more, here the T-Rex's foot forces (up
to about 650 N: four ulps are 1.2e-4 below 512 N and 2.4e-4 above), and a
leak that moves one by more than about 4.8e-7 of its value still fails.  In
a scratch copy, eps * sin(world yaw) added to the T-Rex's projected gravity
along z failed the full-horizon test for every eps from 1e-1 down to 1e-7
(below that the float32 observation does not change), and added to a foot
force, which behind this policy only the observation bound sees, from 1e-1
down to 1e-4.  The 250-step comparison keeps the tight default bounds.

Measured over the full horizon of seeds 3042-3061 behind the frozen PPO
policy of ``test_heading_free_rollouts`` with one torch thread (as earlier
tests in the SB3 job leave torch; MuJoCo 3.10.0, T-Rex physics r9), worst
over each species' 100 turned headings: the T-Rex's actions 1.63e-8, joint
state 6.93e-8, reward 2.09e-8 and observation 3.05e-5 (one float32 ulp of a
402 N foot force; the float64 foot forces behind the observation differed by
at most 0.95 ulp, 2.9e-5 N, and every other element by 9.5e-7), the
compsognathus's 2.70e-8, 3.20e-7, 5.4e-11 and 9.5e-7 (one float32 ulp of its
9.81 m/s^2 accelerometer reading).  A heading whose actions agree stays at
the open loop's level (joint state 4.8e-13, reward 3.1e-12, observation
1.9e-9).  Which seeds and headings get an action difference changes with
torch's thread count: with one thread 7 of the T-Rex's 20 seeds (3047, 3048,
3051, 3056, 3057, 3060 and 3061; 20 headings) and 8 of the compsognathus's
(22 headings, seeds 3042 and 3043 among them), with four the T-Rex's 3044,
3053, 3054, 3057 and 3060 (11 headings; reward 2.45e-8, joint state 9.4e-8,
observation one ulp).  Each of those T-Rex seeds failed the earlier bounds
(reward 1e-9, each observation element 1e-5), which the test passed only
because its seeds 3042-3043 are not among them; the compsognathus's rewards
and observations stayed within them.  With torch's default threads both
species' actions agreed exactly on seeds 3042-3043 (joint state to 4.5e-13).
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
#: ``test_heading_free_rollouts.METRIC_ATOL``'s precedent: a carried one-ulp action difference moves a reward by
#: about 2e-8 (module docstring).
CLOSED_LOOP_REWARD_ATOL = 1e-6
CLOSED_LOOP_OBSERVATION_ATOL = 1e-5
#: In a closed loop an observation element may also differ by this many float32 ulps of 0 degrees' value
#: (``float32_ulp_bound``): one ulp of rounding and three of headroom for the float64 difference the loop carries
#: (module docstring).
CLOSED_LOOP_OBSERVATION_ULPS = 4

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


def float32_ulp_bound(reference: np.ndarray, atol: float, ulps: int) -> np.ndarray:
    """Per element of a float32 *reference*: *ulps* float32 ulps of its value, or *atol* where that is larger."""
    spacing = np.spacing(np.abs(np.asarray(reference, dtype=np.float32))).astype(np.float64)
    return np.maximum(atol, ulps * spacing)


def assert_paired(
    traces: dict[float, dict[str, Any]],
    *,
    state_atol: float = STATE_ATOL,
    reward_atol: float = REWARD_ATOL,
    observation_atol: float = FLOAT32_BUDGET,
    observation_ulps: int = 0,
) -> dict[str, float]:
    """Every heading's trace equals 0 degrees' to solver tolerance; returns the worst differences.

    *state_atol*, *reward_atol* and *observation_atol* bound the joint state,
    the reward and the observation.  With *observation_ulps* each observation
    element is bounded by ``float32_ulp_bound`` of 0 degrees' value at its
    step instead: *observation_ulps* float32 ulps of that value, or
    *observation_atol* where that is larger.  A closed loop behind a network
    policy passes the ``CLOSED_LOOP_*`` bounds.
    """
    reference = traces[0.0]
    worst = {"actions": 0.0, "joints": 0.0, "rewards": 0.0, "observations": 0.0}
    beyond: list[dict[str, float]] = []
    for heading, trace in traces.items():
        assert len(trace["actions"]) == len(reference["actions"]), heading
        assert trace["end"] == reference["end"], heading
        for key in worst:
            worst[key] = max(worst[key], float(np.max(np.abs(trace[key] - reference[key]))))
        if observation_ulps:
            difference = np.abs(trace["observations"] - reference["observations"])
            bound = float32_ulp_bound(reference["observations"], observation_atol, observation_ulps)
            step, element = np.unravel_index(np.argmax(difference - bound), difference.shape)
            if difference[step, element] > bound[step, element]:
                beyond.append(
                    {
                        "heading": heading,
                        "step": int(step),
                        "element": int(element),
                        "difference": float(difference[step, element]),
                        "reference": float(reference["observations"][step, element]),
                        "bound": float(bound[step, element]),
                    }
                )
    assert worst["actions"] <= ACTION_ATOL, worst
    assert worst["joints"] <= state_atol, worst
    assert worst["rewards"] <= reward_atol, worst
    if observation_ulps:
        assert not beyond, (worst, beyond)
    else:
        assert worst["observations"] <= observation_atol, worst
    return worst
