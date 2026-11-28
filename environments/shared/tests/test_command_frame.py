"""The command frame helpers and the command-slice identity rule (BEHAVIOR_RECIPES_PLAN §4.6, WS-C0, D-D60).

Invariant 8: normalised command values are O(1) from the first step after a
command-mode load.  Since decision D-D60 the slice's statistics are held at
identity (mean 0, variance 1) in every stage, and a load applies one rule:
statistics at identity load as they are; a slice off identity is refused
where a live command would read it and re-imposed where the stage is
``"none"``.
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
    RETIRED_COMMAND_ENV_KEYS,
    CommandNormalizationError,
    command_identity_gap,
    command_slice,
    hold_command_identity,
    pad_running_stats,
    require_command_identity,
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
    # Decision D-D2: ONE command_config replaced the five numeric kwargs.
    assert COMMAND_ENV_KEYS == ("command_mode", "command_config")
    # The retired five, exported (the maintainer's answer of 2026-10-05), in their Phase C order.
    assert RETIRED_COMMAND_ENV_KEYS == (
        "command_speed_range",
        "command_lateral_range",
        "command_yaw_rate_max",
        "command_switch_interval",
        "command_switch_jitter",
    )
    assert not set(RETIRED_COMMAND_ENV_KEYS) & set(COMMAND_ENV_KEYS)
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


def test_hold_command_identity_sets_mean_zero_var_one_and_keeps_count():
    sb3_rms = pytest.importorskip("stable_baselines3.common.running_mean_std")
    rms = sb3_rms.RunningMeanStd(shape=(OBS_DIM,))
    rms.mean[:] = np.linspace(-1.0, 1.0, OBS_DIM)
    rms.var[:] = 2.5
    rms.var[-COMMAND_WIDTH:] = 1e-11
    rms.count = 8e6
    other_mean = rms.mean[:-COMMAND_WIDTH].copy()
    other_var = rms.var[:-COMMAND_WIDTH].copy()

    assert command_identity_gap(rms) == 1.0
    assert hold_command_identity(rms) is True
    np.testing.assert_array_equal(rms.mean[-COMMAND_WIDTH:], 0.0)
    np.testing.assert_array_equal(rms.var[-COMMAND_WIDTH:], 1.0)
    assert rms.count == 8e6
    np.testing.assert_array_equal(rms.mean[:-COMMAND_WIDTH], other_mean)
    np.testing.assert_array_equal(rms.var[:-COMMAND_WIDTH], other_var)
    # At identity: the gap is exactly 0, holding again changes nothing, and the check passes.
    assert command_identity_gap(rms) == 0.0
    assert hold_command_identity(rms) is False
    require_command_identity(rms, artifact="held", command_mode="heading")


def test_the_identity_gap_and_the_requirement_name_what_is_off():
    stats = SimpleNamespace(mean=np.zeros(OBS_DIM), var=np.ones(OBS_DIM), count=1.0)
    stats.mean[-2] = -0.25
    assert command_identity_gap(stats) == 0.25
    with pytest.raises(CommandNormalizationError, match=r"parent\.pkl: .*not at identity.*'heading_and_speed'"):
        require_command_identity(stats, artifact="parent.pkl", command_mode="heading_and_speed")
    stats.mean[:] = 0.0
    stats.var[-1] = 1.0 + 1e-12  # exact: no tolerance
    assert command_identity_gap(stats) > 0.0
    with pytest.raises(CommandNormalizationError):
        require_command_identity(stats, artifact="parent.pkl", command_mode="heading")
    stats.var[-1] = np.nan
    assert np.isnan(command_identity_gap(stats))
    with pytest.raises(CommandNormalizationError):
        require_command_identity(stats, artifact="parent.pkl", command_mode="heading")
    # A dim outside the slice is never read.
    stats.var[-1] = 1.0
    stats.mean[0] = 5.0
    assert command_identity_gap(stats) == 0.0


def test_an_identity_slice_normalises_a_command_to_itself_in_float32():
    """VecNormalize divides by sqrt(var + 1e-8): at var 1 that is a relative change of 5e-9, below float32's
    half spacing everywhere, so the normalised command is the raw command bit for bit."""
    sb3 = pytest.importorskip("stable_baselines3.common.vec_env")
    normalizer = sb3.VecNormalize(sb3.DummyVecEnv([_dummy_env_class()]))
    try:
        normalizer.obs_rms.mean[:] = 0.3
        normalizer.obs_rms.var[:] = 4.0
        hold_command_identity(normalizer.obs_rms)
        rng = np.random.default_rng(3042)
        raw = rng.uniform(-1.0, 1.0, size=(20_000, OBS_DIM)).astype(np.float32)
        raw[:4, -COMMAND_WIDTH:] = [[1.0, -1.0, 0.0], [1e-7, -1e-30, 0.5], [0.25, -0.5, 0.75], [-0.0, 1e-3, -1e-3]]
        normalised = normalizer.normalize_obs(raw)
        assert normalised.dtype == np.float32
        np.testing.assert_array_equal(normalised[:, -COMMAND_WIDTH:], raw[:, -COMMAND_WIDTH:])
    finally:
        normalizer.close()


def test_the_identity_helpers_refuse_a_dict_and_a_short_mean():
    for helper in (hold_command_identity, command_identity_gap):
        with pytest.raises(ValueError, match="not a dict"):
            helper({"mean": np.zeros(5), "var": np.ones(5), "count": 1.0})
        with pytest.raises(ValueError, match="fewer than"):
            helper(SimpleNamespace(mean=np.zeros(2), var=np.ones(2), count=1.0))
        with pytest.raises(ValueError, match="ndarray mean and var"):
            helper(SimpleNamespace(mean=[0.0] * 5, var=[1.0] * 5, count=1.0))


def test_pad_running_stats_appends_an_identity_slice_and_carries_count():
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
    # Born at identity: holding the padded slice changes nothing.
    before = (padded.mean.copy(), padded.var.copy())
    assert command_identity_gap(padded) == 0.0
    assert hold_command_identity(padded) is False
    np.testing.assert_array_equal(padded.mean, before[0])
    np.testing.assert_array_equal(padded.var, before[1])


@pytest.mark.parametrize("mode", list(COMMAND_MODES) + ["bogus"])
def test_validate_command_mode_accepts_every_mode_on_sb3(mode):
    if mode not in COMMAND_MODES:
        with pytest.raises(ValueError, match=r"is not one of \('none', 'heading', 'heading_and_speed'\)"):
            validate_command_mode(mode, backend="stable-baselines3")
        return
    # Phase D (consolidation PR-9) implemented both live modes on SB3.
    assert validate_command_mode(mode, backend="stable-baselines3") == mode
    assert not hasattr(command_frame, "SB3_COMMAND_REFUSAL")


@pytest.mark.parametrize("backend", ["torchrl", "jax-mjx"])
def test_validate_command_mode_refuses_an_unknown_backend(backend):
    # jax-mjx was removed with the frozen MJX interface core (plant_versions note 23).
    with pytest.raises(ValueError, match="unknown training backend"):
        validate_command_mode("none", backend=backend)
    assert not hasattr(command_frame, "MJX_COMMAND_REFUSAL")


def _saved_vecnorm_at_identity(path):
    """Save a VecNormalize as training saves it since D-D60: the command slice at identity, the rest learned."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env_class = _dummy_env_class()
    saved = VecNormalize(DummyVecEnv([env_class]))
    saved.obs_rms.mean[:] = np.linspace(-0.5, 0.5, OBS_DIM)
    saved.obs_rms.var[:] = 2.0
    hold_command_identity(saved.obs_rms)
    saved.obs_rms.count = 8e6
    saved.save(str(path))
    return env_class


@pytest.mark.parametrize("command_mode", COMMAND_MODES)
def test_load_vecnorm_stats_loads_identity_statistics_unchanged_for_every_mode(tmp_path, command_mode):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.curriculum.checkpoints import load_vecnorm_stats

    path = tmp_path / "parent_vecnorm.pkl"
    env_class = _saved_vecnorm_at_identity(path)
    train_env = VecNormalize(DummyVecEnv([env_class]))
    eval_env = VecNormalize(DummyVecEnv([env_class]))
    try:
        assert load_vecnorm_stats(
            str(path), train_env, eval_env, unsafe_skip_plant_validation=True, command_mode=command_mode
        )
        for destination in (train_env, eval_env):
            assert destination.obs_rms.count == 8e6
            np.testing.assert_array_equal(
                destination.obs_rms.mean[:-COMMAND_WIDTH], np.linspace(-0.5, 0.5, OBS_DIM)[:-3]
            )
            np.testing.assert_array_equal(destination.obs_rms.var[:-COMMAND_WIDTH], 2.0)
            assert command_identity_gap(destination.obs_rms) == 0.0
            # A live command of 1.0 enters the policy as 1.0.
            np.testing.assert_array_equal(_live_command_normalised(destination), 1.0)
        assert eval_env.training is False and eval_env.norm_reward is False
        assert train_env.obs_rms is not eval_env.obs_rms
    finally:
        train_env.close()
        eval_env.close()


def test_load_vecnorm_stats_reimposes_identity_for_a_none_stage_and_says_so(tmp_path, caplog):
    """A sidecar saved before D-D60 (the dead slice of a command-free stage): a "none" stage re-imposes identity
    on BOTH destinations, count and every other statistic kept, and logs the file and the deviation."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.curriculum.checkpoints import load_vecnorm_stats

    path = tmp_path / "parent_vecnorm.pkl"
    env_class = _saved_vecnorm_with_dead_command_slice(path)
    train_env = VecNormalize(DummyVecEnv([env_class]))
    eval_env = VecNormalize(DummyVecEnv([env_class]))
    try:
        with caplog.at_level("WARNING"):
            assert load_vecnorm_stats(
                str(path), train_env, eval_env, unsafe_skip_plant_validation=True, command_mode="none"
            )
        assert f"{path}: the command slice of its observation statistics was off identity" in caplog.text
        assert "identity re-imposed for this command-free stage" in caplog.text
        for destination in (train_env, eval_env):
            assert destination.obs_rms.count == 8e6
            np.testing.assert_array_equal(destination.obs_rms.var[:-COMMAND_WIDTH], 2.0)
            assert command_identity_gap(destination.obs_rms) == 0.0
            # The zero command a "none" stage emits normalises to zero under either statistics.
            assert not np.any(destination.normalize_obs(np.zeros(OBS_DIM, dtype=np.float32))[-COMMAND_WIDTH:])
        assert train_env.obs_rms is not eval_env.obs_rms
    finally:
        train_env.close()
        eval_env.close()


@pytest.mark.parametrize("command_mode", ["heading", "heading_and_speed"])
@pytest.mark.parametrize("carry_ret_rms", [False, True])
def test_load_vecnorm_stats_refuses_a_slice_off_identity_for_a_live_stage(tmp_path, command_mode, carry_ret_rms):
    """Fail closed before either destination changes, on a warm start and a same-stage resume alike."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.curriculum.checkpoints import load_vecnorm_stats

    path = tmp_path / "parent_vecnorm.pkl"
    env_class = _saved_vecnorm_with_dead_command_slice(path)
    train_env = VecNormalize(DummyVecEnv([env_class]))
    eval_env = VecNormalize(DummyVecEnv([env_class]))
    try:
        before = (train_env.obs_rms, eval_env.obs_rms, train_env.ret_rms)
        with pytest.raises(CommandNormalizationError, match=rf"parent_vecnorm\.pkl.*not at identity.*{command_mode!r}"):
            load_vecnorm_stats(
                str(path),
                train_env,
                eval_env,
                unsafe_skip_plant_validation=True,
                carry_ret_rms=carry_ret_rms,
                command_mode=command_mode,
            )
        assert (train_env.obs_rms, eval_env.obs_rms, train_env.ret_rms) == before
        assert eval_env.training is True, "nothing changed"
    finally:
        train_env.close()
        eval_env.close()


def test_load_vecnorm_stats_needs_a_known_command_mode(tmp_path):
    from environments.shared.curriculum.checkpoints import load_vecnorm_stats

    with pytest.raises(TypeError, match="command_mode"):
        load_vecnorm_stats(str(tmp_path / "x.pkl"), object(), unsafe_skip_plant_validation=True)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="command_mode='walk'"):
        load_vecnorm_stats(str(tmp_path / "x.pkl"), object(), unsafe_skip_plant_validation=True, command_mode="walk")


@pytest.mark.parametrize("task_load_mode", ["initialize_next_stage", "resume_same_stage"])
@pytest.mark.parametrize("command_mode", COMMAND_MODES)
def test_load_vecnorm_into_envs_passes_the_stage_command_mode_on_every_load(monkeypatch, task_load_mode, command_mode):
    """The identity rule (D-D60) lives in load_vecnorm_stats; train_base passes it the stage's mode on every load
    mode, a same-stage resume included (its own saves were held at identity)."""
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
    assert captured["kwargs"]["command_mode"] == command_mode
    assert "reseed_command_slice" not in captured["kwargs"]


def test_the_train_base_loaders_require_the_stage_command_mode(monkeypatch):
    """No default: a caller that left it out would load a live stage's statistics under the "none" branch, which
    re-imposes an off-identity slice instead of refusing it (the same rule as load_vecnorm_stats)."""
    import inspect

    import environments.shared.curriculum as curriculum
    from environments.shared import train_base

    for helper in (train_base._load_vecnorm_into_envs, train_base._post_training_eval_panels):
        parameter = inspect.signature(helper).parameters["command_mode"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, helper.__name__
        assert parameter.default is inspect.Parameter.empty, helper.__name__

    captured: dict = {}
    monkeypatch.setattr(
        curriculum, "load_vecnorm_stats", lambda path, train_env, eval_env, **kwargs: captured.update(kwargs) or True
    )
    with pytest.raises(TypeError, match="command_mode"):
        train_base._load_vecnorm_into_envs(  # type: ignore[call-arg]
            "runs/parent/models/best_model", object(), SimpleNamespace(), task_load_mode="resume_same_stage"
        )
    assert not captured, "nothing loads without the mode"


def test_load_sb3_checkpoint_hold_command_identity_flag(tmp_path):
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

    _, held, _ = load_sb3_checkpoint(str(model_path), str(vecnorm_path), env_class, hold_command_identity=True)
    np.testing.assert_array_equal(held.obs_rms.mean[-COMMAND_WIDTH:], 0.0)
    np.testing.assert_array_equal(held.obs_rms.var[-COMMAND_WIDTH:], 1.0)
    np.testing.assert_array_equal(held.obs_rms.var[:-COMMAND_WIDTH], 2.0)
    assert held.obs_rms.count == 8e6
    assert held.training is False and held.norm_reward is False
    assert np.all(np.abs(_live_command_normalised(held)) <= 1.5)


def test_module_imports_without_stable_baselines3(monkeypatch):
    """A3: numpy-only at module level, and the identity hold works on any mean/var-bearing object."""
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "jax", None)
    monkeypatch.delitem(sys.modules, "environments.shared.command_frame", raising=False)
    try:
        fresh = importlib.import_module("environments.shared.command_frame")
        assert fresh is not command_frame
        stats = SimpleNamespace(mean=np.linspace(-1.0, 1.0, 8), var=np.full(8, 1e-11), count=8e6)
        assert fresh.hold_command_identity(stats) is True
        np.testing.assert_array_equal(stats.mean[-3:], 0.0)
        np.testing.assert_array_equal(stats.var[-3:], 1.0)
        np.testing.assert_array_equal(stats.var[:-3], 1e-11)
        assert stats.count == 8e6
        assert fresh.zero_command().tolist() == [0.0, 0.0, 0.0]
    finally:
        # Restore the original module object so later imports share one identity.
        sys.modules["environments.shared.command_frame"] = command_frame
