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
(the observations agree to 2.3e-10 until then).  The same seed behind the
statistics below agrees to 4.5e-13 in joint state over the whole horizon.
Two comparisons per species, seeds 3042 and 3043 over the full 1000-step
horizon: the float64 traces of ``heading_free_helpers.paired_traces``, and
the ``stance_quality/v2`` floor-truth panel (``run_panel(...,
floor_truth=True)`` with the heading probe's whole-scene ``SpawnYaw``), whose
every ``StanceEpisodeMetrics`` field must agree across headings.

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
from environments.shared.reporting.stance_report import SpawnYaw, run_panel  # noqa: E402

from .heading_free_helpers import HEADINGS_DEG, assert_paired, paired_traces, stance_env  # noqa: E402

SEEDS = (3042, 3043)
#: Floor-truth metrics are float64 reductions of the recorded substeps: equal to this across headings.
METRIC_ATOL = 1e-9


@pytest.fixture(scope="module", params=("trex", "compsognathus"))
def frozen_policy(request):
    """An untrained PPO behind a VecNormalize gathered on 512 steps of N(0, 0.1) actions, then frozen."""
    species = request.param
    normalizer = VecNormalize(DummyVecEnv([lambda: stance_env(species)]), norm_reward=False)
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

    yield species, predict
    normalizer.close()


def test_full_horizon_traces_repeat_at_every_heading(frozen_policy):
    species, predict = frozen_policy
    horizon = int(load_stage_config(species, "stance")["env_kwargs"]["max_episode_steps"])
    for seed in SEEDS:
        traces = paired_traces(species, seed, lambda env, observation_dim: predict, horizon)
        assert len(traces[0.0]["actions"]) == horizon, f"{species} seed {seed} must stand the whole horizon"
        assert_paired(traces)


def _metric_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [dataclasses.asdict(metrics) for metrics in result["stance_metrics"]]


def test_full_horizon_floor_truth_metrics_repeat_at_every_heading(frozen_policy):
    species, predict = frozen_policy
    config = load_stage_config(species, "stance")
    env_kwargs = config["env_kwargs"]
    horizon = int(env_kwargs["max_episode_steps"])
    panels = {
        heading: run_panel(
            species,
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
        np.testing.assert_allclose(panel["rewards"], reference["rewards"], rtol=0.0, atol=1e-8, err_msg=str(heading))
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
