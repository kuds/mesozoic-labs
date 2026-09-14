"""The command frame helpers and the command-slice reseed plumbing (BEHAVIOR_RECIPES_PLAN §4.6, WS-C0).

Invariant 8: normalised command values are O(1) from the first step after a
command-mode load.  The plumbing under test is the Phase D hook — in Phase C
every stage runs ``command_mode = "none"`` and the environments refuse
anything else — pinned now so the reseed cannot be forgotten when the
channel goes live.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from environments.shared import command_frame
from environments.shared.command_frame import (
    COMMAND_COMPONENTS,
    COMMAND_ENV_KEYS,
    COMMAND_MODE_NONE,
    COMMAND_MODES,
    COMMAND_PROBE_VECTOR,
    COMMAND_RANGE,
    COMMAND_SEGMENT_NAME,
    COMMAND_WIDTH,
    command_slice,
    pad_running_stats,
    reseed_command_slice,
    validate_command_mode,
    zero_command,
)

OBS_DIM = 10


def _dummy_env_class():
    import gymnasium as gym

    class DummyCommandEnv(gym.Env):
        """A 10-dim Box env whose last three dims stand in for the command."""

        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(OBS_DIM, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}

    return DummyCommandEnv


def _saved_vecnorm_with_dead_command_slice(path):
    """Save a VecNormalize whose command slice accumulated var ≈ 1e-11 over 8M samples."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env_class = _dummy_env_class()
    saved = VecNormalize(DummyVecEnv([env_class]))
    saved.obs_rms.mean[:] = np.linspace(-0.5, 0.5, OBS_DIM)
    saved.obs_rms.var[:] = 2.0
    saved.obs_rms.mean[-COMMAND_WIDTH:] = 0.0
    saved.obs_rms.var[-COMMAND_WIDTH:] = 1e-11
    saved.obs_rms.count = 8e6
    saved.save(str(path))
    return env_class


def _live_command_normalised(vec_normalize) -> np.ndarray:
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[-COMMAND_WIDTH:] = 1.0
    return np.asarray(vec_normalize.normalize_obs(observation))[-COMMAND_WIDTH:]


def test_constants_match_plan():
    assert COMMAND_SEGMENT_NAME == "command"
    assert COMMAND_WIDTH == 3
    assert COMMAND_COMPONENTS == ("v_x_cmd", "v_y_cmd", "yaw_rate_cmd")
    assert len(COMMAND_COMPONENTS) == COMMAND_WIDTH
    assert COMMAND_RANGE == (-1.0, 1.0)
    assert COMMAND_MODE_NONE == "none"
    assert COMMAND_MODES == ("none", "heading", "heading_and_speed")
    assert COMMAND_ENV_KEYS == (
        "command_mode",
        "command_speed_range",
        "command_lateral_range",
        "command_yaw_rate_max",
        "command_switch_interval",
        "command_switch_jitter",
    )
    probe = np.asarray(COMMAND_PROBE_VECTOR, dtype=np.float64)
    assert probe.shape == (COMMAND_WIDTH,)
    assert np.all(np.isfinite(probe)) and np.all(probe != 0.0)
    assert np.all(np.abs(probe) <= 1.0)
    zeros = zero_command()
    assert zeros.dtype == np.float32 and zeros.shape == (COMMAND_WIDTH,) and not zeros.any()


def test_command_slice_is_the_trailing_width():
    assert command_slice(64) == slice(61, 64)
    assert np.arange(64)[command_slice(64)].tolist() == [61, 62, 63]
    with pytest.raises(ValueError, match="cannot carry"):
        command_slice(2)


def test_reseed_command_slice_sets_mean_zero_var_one_and_keeps_count():
    sb3_rms = pytest.importorskip("stable_baselines3.common.running_mean_std")
    rms = sb3_rms.RunningMeanStd(shape=(OBS_DIM,))
    rms.mean[:] = np.linspace(-1.0, 1.0, OBS_DIM)
    rms.var[:] = 2.5
    rms.var[-COMMAND_WIDTH:] = 1e-11
    rms.count = 8e6
    other_mean = rms.mean[:-COMMAND_WIDTH].copy()
    other_var = rms.var[:-COMMAND_WIDTH].copy()

    assert reseed_command_slice(rms) is None
    np.testing.assert_array_equal(rms.mean[-COMMAND_WIDTH:], 0.0)
    np.testing.assert_array_equal(rms.var[-COMMAND_WIDTH:], 1.0)
    assert rms.count == 8e6
    np.testing.assert_array_equal(rms.mean[:-COMMAND_WIDTH], other_mean)
    np.testing.assert_array_equal(rms.var[:-COMMAND_WIDTH], other_var)


def test_reseed_command_slice_refuses_a_dict_and_a_short_mean():
    with pytest.raises(ValueError, match="not a dict"):
        reseed_command_slice({"mean": np.zeros(5), "var": np.ones(5), "count": 1.0})
    with pytest.raises(ValueError, match="fewer than"):
        reseed_command_slice(SimpleNamespace(mean=np.zeros(2), var=np.ones(2), count=1.0))
    with pytest.raises(ValueError, match="ndarray mean and var"):
        reseed_command_slice(SimpleNamespace(mean=[0.0] * 5, var=[1.0] * 5, count=1.0))


def test_pad_running_stats_appends_a_reseeded_slice_and_carries_count():
    sb3_rms = pytest.importorskip("stable_baselines3.common.running_mean_std")
    parent = sb3_rms.RunningMeanStd(shape=(OBS_DIM,))
    parent.mean[:] = np.linspace(-1.0, 1.0, OBS_DIM)
    parent.var[:] = 3.0
    parent.count = 8e6
    parent_mean = parent.mean.copy()
    parent_var = parent.var.copy()

    padded = pad_running_stats(parent)
    assert isinstance(padded, sb3_rms.RunningMeanStd)
    assert padded is not parent
    assert padded.mean.shape == (OBS_DIM + COMMAND_WIDTH,)
    assert padded.var.shape == (OBS_DIM + COMMAND_WIDTH,)
    np.testing.assert_array_equal(padded.mean[:OBS_DIM], parent_mean)
    np.testing.assert_array_equal(padded.var[:OBS_DIM], parent_var)
    np.testing.assert_array_equal(padded.mean[OBS_DIM:], 0.0)
    np.testing.assert_array_equal(padded.var[OBS_DIM:], 1.0)
    assert padded.count == 8e6
    # The parent is untouched: the widen tool must be able to keep it for lineage.
    np.testing.assert_array_equal(parent.mean, parent_mean)
    assert parent.mean.shape == (OBS_DIM,)
    # Already reseeded: reseeding the padded slice again changes nothing.
    before = (padded.mean.copy(), padded.var.copy())
    reseed_command_slice(padded)
    np.testing.assert_array_equal(padded.mean, before[0])
    np.testing.assert_array_equal(padded.var, before[1])


@pytest.mark.parametrize("backend", ["stable-baselines3", "jax-mjx"])
@pytest.mark.parametrize("mode", list(COMMAND_MODES) + ["bogus"])
def test_validate_command_mode_accepts_none_on_both_backends_and_refuses_the_rest(backend, mode):
    if mode == "none":
        assert validate_command_mode(mode, backend=backend) == "none"
        return
    if mode not in COMMAND_MODES:
        with pytest.raises(ValueError, match=r"is not one of \('none', 'heading', 'heading_and_speed'\)"):
            validate_command_mode(mode, backend=backend)
        return
    expected = "not implemented on the jax-mjx backend" if backend == "jax-mjx" else "Phase D"
    with pytest.raises(ValueError, match=expected) as excinfo:
        validate_command_mode(mode, backend=backend)
    assert repr(mode) in str(excinfo.value)


def test_validate_command_mode_refuses_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown training backend"):
        validate_command_mode("none", backend="torchrl")


def test_load_vecnorm_stats_reseeds_command_slice_on_train_and_eval_envs(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.curriculum.checkpoints import load_vecnorm_stats

    path = tmp_path / "parent_vecnorm.pkl"
    env_class = _saved_vecnorm_with_dead_command_slice(path)

    for reseed in (False, True):
        train_env = VecNormalize(DummyVecEnv([env_class]))
        eval_env = VecNormalize(DummyVecEnv([env_class]))
        assert load_vecnorm_stats(
            str(path),
            train_env,
            eval_env,
            unsafe_skip_plant_validation=True,
            reseed_command_slice=reseed,
        )
        for destination in (train_env, eval_env):
            assert destination.obs_rms.count == 8e6
            np.testing.assert_array_equal(destination.obs_rms.var[:-COMMAND_WIDTH], 2.0)
            normalised = _live_command_normalised(destination)
            if reseed:
                # Invariant 8: a live command of 1.0 stays O(1) on both destinations.
                np.testing.assert_array_equal(destination.obs_rms.mean[-COMMAND_WIDTH:], 0.0)
                np.testing.assert_array_equal(destination.obs_rms.var[-COMMAND_WIDTH:], 1.0)
                assert np.all(np.abs(normalised) <= 1.5)
            else:
                # The default leaves the parent's dead slice in place and clips at clip_obs.
                np.testing.assert_array_equal(destination.obs_rms.var[-COMMAND_WIDTH:], 1e-11)
                assert np.all(np.abs(normalised) >= train_env.clip_obs)
        assert eval_env.training is False and eval_env.norm_reward is False
        # The two destinations hold independent statistics (copy semantics
        # kept): the eval copy is taken before the train reseed, so each
        # destination is reseeded on its own line — dropping either one
        # leaves that destination at var 1e-11 and fails above.
        assert train_env.obs_rms is not eval_env.obs_rms


@pytest.mark.parametrize(
    "task_load_mode,command_mode,expect_reseed",
    [
        ("initialize_next_stage", "none", False),
        ("initialize_next_stage", "heading", True),
        ("initialize_next_stage", "heading_and_speed", True),
        # A same-stage resume continues the SAME task: the parent's sidecar
        # already holds the command-slice statistics the policy trained under
        # (the carry_ret_rms reasoning), so a live mode must NOT reseed it.
        ("resume_same_stage", "heading", False),
        ("resume_same_stage", "none", False),
    ],
)
def test_load_vecnorm_into_envs_derives_reseed_from_command_mode(
    monkeypatch, task_load_mode, command_mode, expect_reseed
):
    # train_base imports load_vecnorm_stats from the curriculum package INSIDE
    # the function body, so the package attribute is the one to patch (A7).
    import environments.shared.curriculum as curriculum
    from environments.shared import train_base

    captured: dict = {}

    def fake_load_vecnorm_stats(vecnorm_path, train_env, eval_env, **kwargs):
        captured["path"] = vecnorm_path
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(curriculum, "load_vecnorm_stats", fake_load_vecnorm_stats)
    eval_env = SimpleNamespace(training=True, norm_reward=True)
    train_base._load_vecnorm_into_envs(
        "runs/parent/models/best_model",
        object(),
        eval_env,
        task_load_mode=task_load_mode,
        command_mode=command_mode,
    )
    assert captured["path"].endswith("best_model_vecnorm.pkl")
    assert captured["kwargs"].get("unsafe_skip_plant_validation") is True
    if task_load_mode == "resume_same_stage":
        assert captured["kwargs"]["carry_ret_rms"] is True
    else:
        assert "carry_ret_rms" not in captured["kwargs"]
    if expect_reseed:
        assert captured["kwargs"]["reseed_command_slice"] is True
    else:
        assert "reseed_command_slice" not in captured["kwargs"]


def test_load_vecnorm_into_envs_defaults_to_no_reseed(monkeypatch):
    import environments.shared.curriculum as curriculum
    from environments.shared import train_base

    captured: dict = {}
    monkeypatch.setattr(
        curriculum, "load_vecnorm_stats", lambda path, train_env, eval_env, **kwargs: captured.update(kwargs) or True
    )
    train_base._load_vecnorm_into_envs(
        "runs/parent/models/best_model", object(), SimpleNamespace(), task_load_mode="resume_same_stage"
    )
    assert captured.get("carry_ret_rms") is True
    assert "reseed_command_slice" not in captured


def test_load_sb3_checkpoint_reseed_flag(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    from environments.shared.policy_loading import load_sb3_checkpoint

    vecnorm_path = tmp_path / "tiny_vecnorm.pkl"
    env_class = _saved_vecnorm_with_dead_command_slice(vecnorm_path)
    model = sb3.PPO(
        "MlpPolicy",
        DummyVecEnv([env_class]),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={"net_arch": [8]},
        device="cpu",
        seed=0,
    )
    model_path = tmp_path / "tiny.zip"
    model.save(str(model_path))

    _, normalizer, resolved = load_sb3_checkpoint(str(model_path), str(vecnorm_path), env_class)
    assert resolved == str(vecnorm_path)
    np.testing.assert_array_equal(normalizer.obs_rms.var[-COMMAND_WIDTH:], 1e-11)
    assert np.all(np.abs(_live_command_normalised(normalizer)) >= normalizer.clip_obs)

    _, reseeded, _ = load_sb3_checkpoint(str(model_path), str(vecnorm_path), env_class, reseed_command_slice=True)
    np.testing.assert_array_equal(reseeded.obs_rms.mean[-COMMAND_WIDTH:], 0.0)
    np.testing.assert_array_equal(reseeded.obs_rms.var[-COMMAND_WIDTH:], 1.0)
    np.testing.assert_array_equal(reseeded.obs_rms.var[:-COMMAND_WIDTH], 2.0)
    assert reseeded.obs_rms.count == 8e6
    assert reseeded.training is False and reseeded.norm_reward is False
    assert np.all(np.abs(_live_command_normalised(reseeded)) <= 1.5)


def test_module_imports_without_stable_baselines3(monkeypatch):
    """A3: numpy-only at module level, and reseed works on any mean/var-bearing object."""
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "jax", None)
    monkeypatch.delitem(sys.modules, "environments.shared.command_frame", raising=False)
    try:
        fresh = importlib.import_module("environments.shared.command_frame")
        assert fresh is not command_frame
        stats = SimpleNamespace(mean=np.linspace(-1.0, 1.0, 8), var=np.full(8, 1e-11), count=8e6)
        fresh.reseed_command_slice(stats)
        np.testing.assert_array_equal(stats.mean[-3:], 0.0)
        np.testing.assert_array_equal(stats.var[-3:], 1.0)
        np.testing.assert_array_equal(stats.var[:-3], 1e-11)
        assert stats.count == 8e6
        assert fresh.zero_command().tolist() == [0.0, 0.0, 0.0]
    finally:
        # Restore the original module object so later imports share one identity.
        sys.modules["environments.shared.command_frame"] = command_frame
