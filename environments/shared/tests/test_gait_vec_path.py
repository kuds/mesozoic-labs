"""The floor-truth recorder under SB3's auto-resetting vec env gives the explicit loop's trace.

The in-training stance screen evaluates through ``VecNormalize(DummyVecEnv([Monitor(env)]))``,
where the recorder learns of a step only after ``DummyVecEnv`` has already
reset the env at ``done``: the env's own ``_aggregated_foot_contact_forces()``
tag is stale by then, and the post-reset pose is the NEXT episode's.  The
recorder samples everything inside the substep hook and detects the new
episode lazily, at its first substep, so the first vec episode must equal the
explicit seed-3042 episode bit for bit -- every array but ``reward``, which
``DummyVecEnv.buf_rews`` rounds to float32.  Needs SB3, so it runs in the SB3
job (``python-ci.yml``) and skips in the plain test matrix.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from environments.shared.config import load_stage_config
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import EpisodeTrace, SubstepContactRecorder
from environments.shared.species_registry import get_species_config

HORIZON = 40


def make_env(**overrides):
    kwargs = {**dict(load_stage_config("trex", "stance")["env_kwargs"]), "max_episode_steps": HORIZON, **overrides}
    return get_species_config("trex").env_class(**kwargs)


def test_the_vec_path_with_auto_reset_gives_the_explicit_trace():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    explicit_env = make_env()
    try:
        with SubstepContactRecorder(explicit_env, Morphology.from_env(explicit_env, "trex")) as recorder:
            explicit_env.reset(seed=3042)
            zero = np.zeros(explicit_env.model.nu)
            while True:
                _obs, reward, terminated, truncated, _info = explicit_env.step(zero)
                recorder.end_step(zero, reward)
                if terminated or truncated:
                    break
            explicit = recorder.end_episode(terminated=terminated, truncated=truncated)
    finally:
        explicit_env.close()

    vec_env = VecNormalize(DummyVecEnv([lambda: Monitor(make_env())]), training=False, norm_reward=False)
    try:
        vec_env.seed(3042)
        inner = vec_env.venv.envs[0].unwrapped
        traces = []
        with SubstepContactRecorder(inner, Morphology.from_env(inner, "trex")) as recorder:
            vec_env.reset()
            while len(traces) < 2:
                actions = np.zeros((1, inner.model.nu), dtype=np.float32)
                _obs, rewards, dones, infos = vec_env.step(actions)
                recorder.end_step(actions[0], float(rewards[0]))  # the env has already auto-reset
                if dones[0]:
                    truncated = bool(infos[0].get("TimeLimit.truncated", False))
                    traces.append(recorder.end_episode(terminated=not truncated, truncated=truncated))
            assert inner._step_count == 0 and not recorder.episode_open
    finally:
        vec_env.close()

    first, second = traces
    for spec in fields(EpisodeTrace):
        if spec.name in ("reward", "terminated", "truncated"):
            continue
        a, b = getattr(explicit, spec.name), getattr(first, spec.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=spec.name)
        else:
            assert a == b, spec.name
    np.testing.assert_allclose(first.reward, explicit.reward, rtol=1e-6)
    assert (first.truncated, explicit.truncated) == (True, True)
    # The unseeded second episode's spawn was caught at its first substep, after the auto-reset.
    assert second.length == HORIZON
    assert not np.array_equal(second.spawn_root_pos, first.spawn_root_pos)
