"""Full-horizon six-heading paired rollouts on the heading-free observation (plant_versions note 18).

The design's acceptance list asks for paired full-horizon real-physics
rollouts at all six headings with a fixed normalizer and a deterministic
policy, comparing action and state traces and every floor-truth metric to the
contact solver's tolerance (docs/HEADING_INVARIANCE_DESIGN_2026_10.md,
"Migration and acceptance").  Since T-Rex physics r9 (D-D29) both species'
headings are physically identical, so with a heading-free observation the
six headings of one seed are one episode.

The policy is an untrained PPO's deterministic action behind a VecNormalize
whose statistics were gathered on 512 steps of small random actions
(N(0, 0.1), clipped) and then frozen, so the observation is normalized with a
realistic spread, as a trained policy's is.  Statistics gathered under the
zero action alone leave near-constant channels with a standard deviation
near 1e-3, and behind them the untrained policy's closed loop on the
compsognathus is unstable: a one-ulp float32 action difference (at step
145 of seed 3042; 226 at +90 degrees) grows step by step to 0.03 in action
and 0.33 rad in joint state by the end, at every heading but 0 degrees --
the rounding of a rotated state amplified by the policy, not a heading leak
(the observations agree to 2.3e-10 until then).  Behind the statistics below
the loop is stable: one-ulp action differences, when the arithmetic produces
any, are carried to the end without growing.  Two comparisons per species,
seeds 3042 and 3043 over the full 1000-step horizon: the float64 traces of
``heading_free_helpers.paired_traces``, to its closed-loop bounds (joint state
and reward to 1e-6, each observation element to four float32 ulps of
0 degrees' value or 1e-5, whichever is larger; its module docstring), and the
``stance_quality/v2`` floor-truth panel (``run_panel(..., floor_truth=True)``
with the heading probe's whole-scene ``SpawnYaw``), whose every
``StanceEpisodeMetrics`` field must agree across headings to ``METRIC_ATOL``.
Measured with torch's default threads, the actions agree exactly, joint state
to 4.5e-13 and every metric to 2e-10; with one thread, as earlier tests leave
torch in a full local run (CI's step runs this file in a process of its own,
at torch's default thread count), the compsognathus's actions differ by one
float32 ulp at some step, its joint state by 7.3e-8, its observation by
9.5e-7 and a metric by 5.4e-9 (the T-Rex's are unchanged).  Over seeds
3042-3061 with one thread the T-Rex's actions differ too, on 7 of the 20 seeds
(5 with four threads, three of them others), and the difference they carry
moves its reward by up to 2.1e-8 (2.45e-8 with four threads) and a foot force
by one float32 ulp (3.05e-5 N): beyond the 250-step comparison's reward bound
(1e-9) and a flat 1e-5 on every observation element, which this test applied
until 2026-10-25 and passed only because its seeds are not among them.

About two minutes per run, so it runs in the SB3 job's full depth only (the
nightly schedule, a manual dispatch or the ``full-ci`` label; python-ci.yml,
"Heading-free full-horizon paired rollouts"); the shared legs run the
250-step version of the trace comparison (``test_heading_free_observation.py``)
on every push.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.model_id import ModelId
from environments.shared.reporting.stance_report import SpawnYaw, run_panel  # noqa: E402

from .heading_free_helpers import (  # noqa: E402
    CLOSED_LOOP_OBSERVATION_ATOL,
    CLOSED_LOOP_OBSERVATION_ULPS,
    CLOSED_LOOP_REWARD_ATOL,
    CLOSED_LOOP_STATE_ATOL,
    HEADINGS_DEG,
    assert_paired,
    paired_traces,
    stance_env,
)

SEEDS = (3042, 3043)
#: Floor-truth metrics are float64 reductions of the recorded substeps; a one-ulp action difference moves them
#: by about 5e-9 (module docstring).
METRIC_ATOL = 1e-6


@pytest.fixture(
    scope="module",
    params=(ModelId("tyrannosaurus_rex", "anatomical"), ModelId("compsognathus_longipes", "anatomical")),
    ids=lambda m: f"{m.species}-{m.embodiment}",
)
def frozen_policy(request):
    """An untrained PPO behind a VecNormalize gathered on 512 steps of N(0, 0.1) actions, then frozen."""
    model_id = request.param
    normalizer = VecNormalize(DummyVecEnv([lambda: stance_env(model_id)]), norm_reward=False)
    normalizer.seed(3042)
    normalizer.reset()
    rng = np.random.default_rng(0)
    for _ in range(512):
        action = rng.normal(scale=0.1, size=(1, *normalizer.action_space.shape))
        normalizer.step(np.clip(action, -1.0, 1.0).astype(np.float32))
    normalizer.training = False
    model = PPO("MlpPolicy", normalizer, seed=0, device="cpu", policy_kwargs={"net_arch": [64, 64]})

    def predict(observation: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(normalizer.normalize_obs(observation), deterministic=True)[0])

    yield model_id, predict
    normalizer.close()


def test_full_horizon_traces_repeat_at_every_heading(frozen_policy):
    model_id, predict = frozen_policy
    horizon = int(load_stage_config(model_id, "stance")["env_kwargs"]["max_episode_steps"])
    for seed in SEEDS:
        traces = paired_traces(model_id, seed, lambda env, observation_dim: predict, horizon)
        assert len(traces[0.0]["actions"]) == horizon, f"{model_id} seed {seed} must stand the whole horizon"
        assert_paired(
            traces,
            state_atol=CLOSED_LOOP_STATE_ATOL,
            reward_atol=CLOSED_LOOP_REWARD_ATOL,
            observation_atol=CLOSED_LOOP_OBSERVATION_ATOL,
            observation_ulps=CLOSED_LOOP_OBSERVATION_ULPS,
        )


def _metric_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [dataclasses.asdict(metrics) for metrics in result["stance_metrics"]]


def test_full_horizon_floor_truth_metrics_repeat_at_every_heading(frozen_policy):
    model_id, predict = frozen_policy
    config = load_stage_config(model_id, "stance")
    env_kwargs = config["env_kwargs"]
    horizon = int(env_kwargs["max_episode_steps"])
    panels = {
        heading: run_panel(
            model_id,
            predict=predict,
            episodes=len(SEEDS),
            seed=SEEDS[0],
            settle_steps=int(config["curriculum_kwargs"]["settle_steps"]),
            horizon=horizon,
            env_kwargs=env_kwargs,
            floor_truth=True,
            spawn_yaw=SpawnYaw(heading, whole_scene=True),
        )
        for heading in HEADINGS_DEG
    }
    reference = panels[0.0]
    assert list(reference["lengths"]) == [horizon] * len(SEEDS)
    reference_rows = _metric_rows(reference)
    for heading, panel in panels.items():
        np.testing.assert_array_equal(panel["lengths"], reference["lengths"], err_msg=str(heading))
        assert panel["terminations"] == reference["terminations"], heading
        np.testing.assert_allclose(
            panel["rewards"], reference["rewards"], rtol=0.0, atol=METRIC_ATOL, err_msg=str(heading)
        )
        for row, expected in zip(_metric_rows(panel), reference_rows):
            assert row.keys() == expected.keys()
            for field, value in row.items():
                if isinstance(value, (bool, int, str)) or value is None:
                    assert value == expected[field], (heading, field)
                else:
                    # Floats and per-foot / per-geom tuples of floats; NaN where a metric is undefined.
                    np.testing.assert_allclose(
                        np.asarray(value, dtype=np.float64),
                        np.asarray(expected[field], dtype=np.float64),
                        rtol=0.0,
                        atol=METRIC_ATOL,
                        equal_nan=True,
                        err_msg=f"{heading} {field}",
                    )
