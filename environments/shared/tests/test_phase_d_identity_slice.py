"""The command observation slice held at identity normalization (decision D-D60 (5)).

* **The hold** (``curriculum.command_identity``): ``CommandIdentityCallback``
  is first in every stage's callback list and re-imposes mean 0 / variance 1
  on the command slice after the reset update and after every step's update;
  the two save callbacks and the final save (after an interrupt too, and
  through ``CriticPhaseVecNormalize.save``) hold it themselves.  A stance
  computes exactly what it computed without the hold: its zero command
  normalises to zero under either statistics.  The terminal observation of a
  finished episode is re-expressed under the held statistics, so SAC's replay
  buffer stores the raw observations it stores without the hold.
* **A live stage** (a temporary ``[commands]`` TOML, never committed)
  fingerprints with its command section, trains with PPO (with and without
  the critic phase) and with SAC, writes every sidecar at identity, evaluates
  under identity statistics and reloads them.
* **The load rule** (``curriculum.load_vecnorm_stats``): statistics at
  identity load unchanged; a slice off identity (a sidecar saved before the
  rule) is refused for a live stage and re-imposed, with a warning, for a
  ``"none"`` one.
* **Warm starts and resumes**: a command-free parent's command columns are
  zeroed and proven blind; a live parent's are kept, and the child acts
  exactly as the parent on non-zero commands; a live stage resumes its own
  pair with the statistics, the weights and the eval env's copy as saved.
  Every one of them learns with the hold first in its callback list.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from environments.shared.command_frame import (
    COMMAND_WIDTH,
    CommandNormalizationError,
    command_identity_gap,
)
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import current_plant_identity
from environments.shared.species_registry import get_species_config
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import stage_task_fingerprint
from environments.shared.tests.test_phase_d_command_plumbing import WALK_COMMANDS, _live_config

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
OBS_DIM = 8

#: A follow-like widening of WALK_COMMANDS: turns, a wider speed range, an earlier switch.
FOLLOW_COMMANDS = """
[commands]
mode = "heading_and_speed"
cruise_speed = 0.5
speed_range = [0.1, 0.6]
stop_probability = 0.15
turn_increment_max = 0.6
switch_interval_s = 2.0
initial_heading_offset_max = 0.5
tracking_weight = 2.0
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_env_class(live: bool = True, settle: bool = False):
    """An 8-dim Box env whose last three dims are a command: live (varying, non-zero) or zero."""
    import gymnasium as gym

    class ToyCommandEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            if settle:
                self.stance_width_settle_steps = 4
            self._steps = 0

        def _obs(self):
            obs = self.np_random.normal(0.5, 2.0, OBS_DIM).astype(np.float32)
            obs[-COMMAND_WIDTH:] = self.np_random.uniform(-1.0, 1.0, COMMAND_WIDTH) if live else 0.0
            return obs

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._steps = 0
            return self._obs(), {}

        def step(self, action):
            self._steps += 1
            return self._obs(), float(-np.square(action).sum()), False, self._steps >= 16, {"step": self._steps}

    return ToyCommandEnv


def _off_identity(vec_normalize: Any) -> None:
    """Move the command slice off identity the way one step_wait update does under a live command."""
    vec_normalize.obs_rms.mean[-COMMAND_WIDTH:] = [0.2, -0.1, 0.05]
    vec_normalize.obs_rms.var[-COMMAND_WIDTH:] = [0.4, 0.9, 1.3]


def _saved_gap(path: "str | Path") -> float:
    with open(path, "rb") as handle:
        saved = pickle.load(handle)
    from stable_baselines3.common.vec_env import VecNormalize

    assert type(saved) is VecNormalize, "every sidecar is a stock VecNormalize"
    return command_identity_gap(saved.obs_rms)


def _stage_key(configs: dict, stage_id: str) -> Any:
    manifest = load_stage_manifest(COMPSOGNATHUS_LONGIPES)
    return next(reference for reference in configs if manifest.resolve(reference).id == stage_id)


def _small(config: dict, algorithm: str) -> dict:
    """*config* with a short horizon and a small, fast algorithm table (the task is unchanged but the horizon)."""
    config["env_kwargs"]["max_episode_steps"] = 64
    if algorithm == "ppo":
        config["ppo_kwargs"].update(n_steps=64, batch_size=64, n_epochs=1)
    else:
        config["sac_kwargs"].update(learning_starts=64, batch_size=32, buffer_size=1_000)
    return config


def _train(tmp_path: Path, configs: dict, key: Any, algorithm: str, *, steps: int = 128, **kwargs: Any) -> Any:
    from environments.shared import train_base

    options: dict[str, Any] = dict(
        n_envs=2,
        seed=3,
        eval_freq=64,
        save_freq=64,
        verbose=0,
        algorithm=algorithm,
        output_dir=str(tmp_path),
        use_tensorboard=False,
        post_eval_episodes=0,
        report_metrics=False,
        save_on_interrupt=False,
    )
    options.update(kwargs)
    return train_base.train(get_species_config(COMPSOGNATHUS_LONGIPES), configs, key, total_timesteps=steps, **options)


# ---------------------------------------------------------------------------
# The hold: the callback and every save path
# ---------------------------------------------------------------------------


def test_the_callback_holds_the_slice_at_training_start_and_after_every_step():
    """A callback after it sees the training statistics at identity on every step, and so does the eval env; the
    other dims and the count keep updating."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.curriculum import CommandIdentityCallback

    train_env = VecNormalize(DummyVecEnv([_toy_env_class()]))
    eval_env = VecNormalize(DummyVecEnv([_toy_env_class()]), training=False)
    _off_identity(eval_env)
    seen: list[float] = []

    class Observe(BaseCallback):
        def _on_training_start(self) -> None:
            seen.append(command_identity_gap(train_env.obs_rms))
            seen.append(command_identity_gap(eval_env.obs_rms))

        def _on_step(self) -> bool:
            seen.append(command_identity_gap(train_env.obs_rms))
            return True

    try:
        model = PPO("MlpPolicy", train_env, n_steps=16, batch_size=16, n_epochs=1, seed=0, device="cpu")
        model.learn(48, callback=[CommandIdentityCallback(eval_env=eval_env), Observe()])
        assert len(seen) == 2 + 48 and set(seen) == {0.0}
        assert command_identity_gap(train_env.obs_rms) == 0.0
        assert train_env.obs_rms.count > 48
        assert np.all(train_env.obs_rms.var[:-COMMAND_WIDTH] > 1.5), "the other dims learned the env's variance"
        # Without the hold a live command moves the slice on the very first update.
        stock = VecNormalize(DummyVecEnv([_toy_env_class()]))
        stock.reset()
        assert command_identity_gap(stock.obs_rms) > 0.1
        stock.close()
    finally:
        train_env.close()
        eval_env.close()


def test_the_callback_is_first_in_every_stage_callback_list(tmp_path):
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.curriculum import CommandIdentityCallback
    from environments.shared.train_base import _build_core_callbacks

    eval_env = VecNormalize(DummyVecEnv([_toy_env_class()]))
    try:
        callbacks, _, _ = _build_core_callbacks(
            {"CheckpointCallback": CheckpointCallback},
            eval_env,
            tmp_path / "models",
            tmp_path / "logs",
            1,
            1,
            100,
            100,
            0,
            {"curriculum_kwargs": {"min_avg_reward": 1.0}},
        )
    finally:
        eval_env.close()
    assert isinstance(callbacks[0], CommandIdentityCallback) and callbacks[0].eval_env is eval_env
    assert sum(isinstance(callback, CommandIdentityCallback) for callback in callbacks) == 1


def test_the_two_save_callbacks_hold_the_slice_before_they_write(tmp_path):
    from types import SimpleNamespace

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.curriculum import RobustBestModelCallback, SaveVecNormalizeCallback

    venv = VecNormalize(DummyVecEnv([_toy_env_class()]))
    model = SimpleNamespace(get_vec_normalize_env=lambda: venv, save=lambda path: Path(f"{path}.zip").write_bytes(b""))
    try:
        _off_identity(venv)
        best = SaveVecNormalizeCallback(save_path=str(tmp_path / "best_model_vecnorm.pkl"))
        best.model = model
        best._on_step()
        assert _saved_gap(tmp_path / "best_model_vecnorm.pkl") == 0.0

        _off_identity(venv)
        robust = RobustBestModelCallback(SimpleNamespace(evaluations_results=[[1.0, 2.0]]), model_dir=tmp_path)
        robust.model = model
        robust.num_timesteps = 64
        robust._on_step()
        assert _saved_gap(tmp_path / "robust_best_model_vecnorm.pkl") == 0.0
    finally:
        venv.close()


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("critic_phase", [False, True])
def test_the_final_save_holds_the_slice_after_an_interrupt(tmp_path, monkeypatch, remote, critic_phase):
    """A KeyboardInterrupt can end learn() between a step's update and the callback: the final save holds the
    slice itself, on the local and the mount path, and through CriticPhaseVecNormalize's stock pickle."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared import train_base
    from environments.shared.critic_phase_ppo import CriticPhaseVecNormalize

    if remote:
        monkeypatch.setattr(train_base, "_is_remote_mount_path", lambda path: True)
    env_class = _toy_env_class(settle=True)
    venv = (
        CriticPhaseVecNormalize(DummyVecEnv([env_class]), critic_phase_gamma=0.99)
        if critic_phase
        else VecNormalize(DummyVecEnv([env_class]))
    )
    model = PPO("MlpPolicy", VecNormalize(DummyVecEnv([env_class])), n_steps=16, batch_size=16, seed=0, device="cpu")
    try:
        _off_identity(venv)
        other = venv.obs_rms.mean[:-COMMAND_WIDTH].copy()
        final_path = train_base._save_final_and_sync_tb(model, venv, tmp_path, "locomotion", None, None)
        assert _saved_gap(f"{final_path}_vecnorm.pkl") == 0.0
        assert Path(f"{final_path}.zip").stat().st_size > 0
        np.testing.assert_array_equal(venv.obs_rms.mean[:-COMMAND_WIDTH], other)
    finally:
        venv.close()
        model.get_env().close()


def _logging_toy_env_class(live: bool, log: list[np.ndarray]):
    """The toy env, appending the last observation of every episode (its raw terminal observation) to *log*."""
    base = _toy_env_class(live=live)

    class LoggingToyCommandEnv(base):  # type: ignore[valid-type,misc]
        def step(self, action):
            result = super().step(action)
            if result[2] or result[3]:
                log.append(np.array(result[0], copy=True))
            return result

    return LoggingToyCommandEnv


def test_a_terminal_command_is_reexpressed_in_the_statistics_command_columns_alone():
    """The command columns are the last three of the statistics' width: a critic-phase env appends its phase after
    them, which stays as it is, as do the other columns and every env that did not finish."""
    from types import SimpleNamespace

    from environments.shared.curriculum.command_identity import reexpress_terminal_commands

    epsilon = 1e-8
    normalizer = SimpleNamespace(
        obs_rms=SimpleNamespace(mean=np.zeros(OBS_DIM), var=np.ones(OBS_DIM)), epsilon=epsilon, clip_obs=10.0
    )
    mean, var = np.array([0.2, -0.1, 0.05]), np.array([0.4, 0.9, 1.3])
    raw = np.array([0.5, -0.75, 1.0])
    phase = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    terminal = np.concatenate(
        [
            np.arange(OBS_DIM - COMMAND_WIDTH, dtype=np.float32),
            ((raw - mean) / np.sqrt(var + epsilon)).astype(np.float32),
            phase,
        ]
    )
    infos: list[dict[str, Any]] = [{"terminal_observation": terminal.copy()}, {"terminal_observation": terminal}, {}]
    assert reexpress_terminal_commands(normalizer, mean, var, [True, False, True], infos) == 1
    held = infos[0]["terminal_observation"]
    assert held.dtype == np.float32 and held.shape == terminal.shape
    np.testing.assert_array_equal(held[: OBS_DIM - COMMAND_WIDTH], terminal[: OBS_DIM - COMMAND_WIDTH])
    np.testing.assert_array_equal(held[OBS_DIM:], phase)
    np.testing.assert_allclose(held[OBS_DIM - COMMAND_WIDTH : OBS_DIM] * np.sqrt(1.0 + epsilon), raw, atol=1e-6, rtol=0)
    assert infos[1]["terminal_observation"] is terminal, "an env that did not finish is left alone"
    assert reexpress_terminal_commands(normalizer, mean, var, None, infos) == 0


@pytest.mark.parametrize("live", [False, True], ids=["none", "live"])
def test_sac_stores_the_raw_observations_under_the_hold(live):
    """SAC stores un-normalised observations (``get_original_obs``) and un-normalises a finished episode's terminal
    observation after the callbacks.  The hold re-expresses that terminal command under the held statistics, so the
    replay buffer holds what it holds without the hold: under a zero command bitwise, the learned weights too; under
    a live one bitwise but for the terminal commands, which equal the raw ones to the float32 rounding of the round
    trip, as without the hold (un-normalised under the held statistics alone they would be one update away).  The
    buffer's commands normalise to themselves when sampled."""
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.curriculum import CommandIdentityCallback

    fields = ("observations", "next_observations", "actions", "rewards", "dones", "timeouts")
    runs: dict[bool, dict[str, Any]] = {}
    for held in (False, True):
        log: list[np.ndarray] = []
        env_class = _logging_toy_env_class(live, log)
        venv = VecNormalize(DummyVecEnv([env_class, env_class]))
        # A live run never updates: the two policies would otherwise see different inputs and act differently.
        model = SAC(
            "MlpPolicy",
            venv,
            learning_starts=10_000 if live else 32,
            batch_size=16,
            buffer_size=1_000,
            seed=0,
            device="cpu",
            policy_kwargs={"net_arch": [16, 16]},
        )
        try:
            model.learn(160, callback=[CommandIdentityCallback()] if held else [])
            buffer = model.replay_buffer
            rows = buffer.pos
            stored = {name: np.array(getattr(buffer, name)[:rows], copy=True) for name in fields}
            terminals = np.argwhere(stored["timeouts"])
            assert len(terminals) == len(log) > 0
            terminal_next = stored["next_observations"][terminals[:, 0], terminals[:, 1]]
            np.testing.assert_allclose(
                terminal_next[:, -COMMAND_WIDTH:], np.asarray(log)[:, -COMMAND_WIDTH:], atol=1e-6, rtol=0
            )
            if held:
                assert command_identity_gap(venv.obs_rms) == 0.0
                np.testing.assert_array_equal(
                    venv.normalize_obs(stored["observations"])[..., -COMMAND_WIDTH:],
                    stored["observations"][..., -COMMAND_WIDTH:],
                )
            runs[held] = {
                "stored": stored,
                "terminals": terminals,
                "weights": {name: value.detach().clone() for name, value in model.policy.state_dict().items()},
            }
        finally:
            venv.close()
    stock, held_run = runs[False], runs[True]
    np.testing.assert_array_equal(held_run["terminals"], stock["terminals"])
    for name in fields:
        expected, actual = stock["stored"][name], held_run["stored"][name]
        if live and name == "next_observations":
            rows, envs = held_run["terminals"][:, 0], held_run["terminals"][:, 1]
            expected, actual = expected.copy(), actual.copy()
            np.testing.assert_allclose(
                actual[rows, envs, -COMMAND_WIDTH:], expected[rows, envs, -COMMAND_WIDTH:], atol=1e-6, rtol=0
            )
            expected[rows, envs, -COMMAND_WIDTH:] = actual[rows, envs, -COMMAND_WIDTH:] = 0.0
        np.testing.assert_array_equal(actual, expected, err_msg=name)
    if not live:
        assert not np.any(held_run["stored"]["next_observations"][..., -COMMAND_WIDTH:])
        for name, value in stock["weights"].items():
            assert torch.equal(held_run["weights"][name], value), name


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_the_hold_changes_nothing_a_stance_policy_computes(algorithm):
    """The stance's command is constant zero, whose mean stays exactly 0 while its variance decays: it normalises
    to zero under either statistics.  The same seeded run with and without the hold sees the same normalised
    observations and rewards on every step and ends with bitwise the same weights; the statistics differ in the
    command variance alone."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    from environments.shared.config import load_all_stages
    from environments.shared.curriculum import CommandIdentityCallback
    from environments.shared.policy_loading import _ensure_sb3
    from environments.shared.train_base import create_vec_env

    sb3 = _ensure_sb3()
    configs = load_all_stages(COMPSOGNATHUS_LONGIPES)
    key = _stage_key(configs, "stance")
    assert configs[key]["env_kwargs"].get("command_mode", "none") == "none"
    identity = current_plant_identity(COMPSOGNATHUS_LONGIPES)

    class Record(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.rows: list[tuple[np.ndarray, np.ndarray]] = []

        def _on_step(self) -> bool:
            self.rows.append((np.array(self.locals["new_obs"], copy=True), np.array(self.locals["rewards"], copy=True)))
            return True

    runs = {}
    for held in (False, True):
        env = create_vec_env(
            get_species_config(COMPSOGNATHUS_LONGIPES),
            configs,
            key,
            2,
            11,
            algorithm=algorithm,
            plant_identity=identity,
        )
        kwargs: dict[str, Any] = (
            {"n_steps": 32, "batch_size": 32, "n_epochs": 2}
            if algorithm == "ppo"
            else {"learning_starts": 32, "batch_size": 32, "buffer_size": 500}
        )
        model = sb3[algorithm.upper()](
            "MlpPolicy", env, seed=5, device="cpu", policy_kwargs={"net_arch": [32, 32]}, **kwargs
        )
        record = Record()
        callbacks = [CommandIdentityCallback(), record] if held else [record]
        model.learn(128, callback=callbacks)
        runs[held] = (
            {name: value.detach().clone() for name, value in model.policy.state_dict().items()},
            record.rows,
            env.obs_rms.mean.copy(),
            env.obs_rms.var.copy(),
            env.obs_rms.count,
        )
        env.close()
    (stock_weights, stock_rows, stock_mean, stock_var, stock_count) = runs[False]
    (held_weights, held_rows, held_mean, held_var, held_count) = runs[True]
    assert len(stock_rows) == len(held_rows) == 64
    for (stock_obs, stock_reward), (held_obs, held_reward) in zip(stock_rows, held_rows):
        assert not np.any(held_obs[:, -COMMAND_WIDTH:])
        np.testing.assert_array_equal(held_obs, stock_obs)
        np.testing.assert_array_equal(held_reward, stock_reward)
    assert stock_weights.keys() == held_weights.keys()
    for name, value in stock_weights.items():
        assert torch.equal(held_weights[name], value), name
    np.testing.assert_array_equal(held_mean, stock_mean)
    np.testing.assert_array_equal(held_mean[-COMMAND_WIDTH:], 0.0)
    np.testing.assert_array_equal(held_var[:-COMMAND_WIDTH], stock_var[:-COMMAND_WIDTH])
    np.testing.assert_array_equal(held_var[-COMMAND_WIDTH:], 1.0)
    # Without the hold the zero command's variance decays as 1e-4 / count.
    np.testing.assert_allclose(stock_var[-COMMAND_WIDTH:], 1e-4 / stock_count, rtol=1e-9)
    assert held_count == stock_count


# ---------------------------------------------------------------------------
# A live stage: train, save, evaluate and reload at identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("algorithm", "critic_phase"), [("ppo", False), ("ppo", True), ("sac", False)])
def test_a_live_stage_trains_saves_evaluates_and_reloads_at_identity(tmp_path, monkeypatch, algorithm, critic_phase):
    """A temporary live locomotion stage: its fingerprint has the command section, it trains a few hundred steps,
    every sidecar it writes (best, robust, periodic, final) holds the slice at identity, every evaluation copies
    identity statistics into the eval env, the post-training panel loads its handoff under the live mode, and the
    final pair reloads into a fresh live stage with the slice at identity and the rest as trained."""
    import stable_baselines3.common.callbacks as sb3_callbacks
    from stable_baselines3.common.vec_env import VecNormalize

    from environments.shared.config import load_all_stages
    from environments.shared.curriculum import load_vecnorm_stats
    from environments.shared.train_base import create_vec_env

    configs = load_all_stages(COMPSOGNATHUS_LONGIPES)
    key = _stage_key(configs, "locomotion")
    live = _small(_live_config(tmp_path / "configs", COMPSOGNATHUS_LONGIPES), algorithm)
    if critic_phase:
        live["ppo_kwargs"]["critic_phase"] = True
    configs[key] = live
    synced: list[float] = []
    sync = sb3_callbacks.sync_envs_normalization

    def recording_sync(env, eval_env):
        sync(env, eval_env)
        synced.append(command_identity_gap(eval_env.obs_rms))

    monkeypatch.setattr(sb3_callbacks, "sync_envs_normalization", recording_sync)
    output = tmp_path / "run"
    model = _train(output, configs, key, algorithm, steps=256, report_metrics=True, post_eval_episodes=1)
    assert "command" in model.mesozoic_task_fingerprint
    assert model.mesozoic_task_fingerprint == stage_task_fingerprint(
        COMPSOGNATHUS_LONGIPES, "locomotion", stage_config=live
    )
    assert synced and set(synced) == {0.0}

    models = output / "models"
    sidecars = sorted(models.glob("*.pkl"))
    names = {path.name for path in sidecars}
    assert {"best_model_vecnorm.pkl", "robust_best_model_vecnorm.pkl"} <= names
    assert any("_vecnormalize_" in name for name in names)
    (final,) = [path for path in sidecars if path.name.endswith("_final_vecnorm.pkl")]
    for path in sidecars:
        assert _saved_gap(path) == 0.0, path.name
        with open(path, "rb") as handle:
            assert pickle.load(handle).obs_rms.count > 64, "statistics the run trained"

    train_env = create_vec_env(
        get_species_config(COMPSOGNATHUS_LONGIPES),
        configs,
        key,
        1,
        0,
        algorithm=algorithm,
        plant_identity=current_plant_identity(COMPSOGNATHUS_LONGIPES),
    )
    eval_env = create_vec_env(
        get_species_config(COMPSOGNATHUS_LONGIPES),
        configs,
        key,
        1,
        1000,
        algorithm=algorithm,
        plant_identity=current_plant_identity(COMPSOGNATHUS_LONGIPES),
    )
    try:
        assert load_vecnorm_stats(
            str(final),
            train_env,
            eval_env,
            current_plant=current_plant_identity(COMPSOGNATHUS_LONGIPES),
            command_mode=live["env_kwargs"]["command_mode"],
        )
        saved = VecNormalize.load(str(final), train_env.venv)
        for destination in (train_env, eval_env):
            assert command_identity_gap(destination.obs_rms) == 0.0
            np.testing.assert_array_equal(destination.obs_rms.mean, saved.obs_rms.mean)
            np.testing.assert_array_equal(destination.obs_rms.var, saved.obs_rms.var)
        # The live env's commands reach the policy as they are.
        raw = train_env.venv.reset()
        assert np.any(raw[:, -COMMAND_WIDTH:]), "a live command"
        np.testing.assert_array_equal(train_env.normalize_obs(raw)[:, -COMMAND_WIDTH:], raw[:, -COMMAND_WIDTH:])
    finally:
        train_env.close()
        eval_env.close()


# ---------------------------------------------------------------------------
# The load rule and the warm starts, on real parents
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=["ppo", "sac"])
def parents(request, tmp_path_factory):
    """Per algorithm: a stance parent and a live walk parent, each trained by train_base (the slice held), and the
    stance parent's pair as a run before decision D-D60 left it (its slice at the zero command's decayed
    variance)."""
    import shutil

    from environments.shared.config import load_all_stages

    algorithm = request.param
    root = tmp_path_factory.mktemp(f"identity-parents-{algorithm}")
    configs = load_all_stages(COMPSOGNATHUS_LONGIPES)
    stance_key, walk_key = _stage_key(configs, "stance"), _stage_key(configs, "locomotion")
    configs[stance_key] = _small(configs[stance_key], algorithm)
    configs[walk_key] = _small(_live_config(root / "configs", COMPSOGNATHUS_LONGIPES), algorithm)
    stance = _train(root / "stance", configs, stance_key, algorithm, eval_freq=10_000, save_freq=10_000)
    walk = _train(root / "walk", configs, walk_key, algorithm, eval_freq=10_000, save_freq=10_000)
    del stance, walk
    (stance_final,) = (root / "stance" / "models").glob("*_final.zip")
    (walk_final,) = (root / "walk" / "models").glob("*_final.zip")
    for stem in (stance_final, walk_final):
        assert _saved_gap(str(stem.with_suffix("")) + "_vecnorm.pkl") == 0.0
    pre_rule = root / "pre_rule" / "stance_final"
    pre_rule.parent.mkdir()
    shutil.copyfile(stance_final, f"{pre_rule}.zip")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    saved = VecNormalize.load(
        str(stance_final.with_suffix("")) + "_vecnorm.pkl",
        DummyVecEnv([get_species_config(COMPSOGNATHUS_LONGIPES).env_class]),
    )
    saved.obs_rms.var[-COMMAND_WIDTH:] = 1e-4 / saved.obs_rms.count
    saved.save(f"{pre_rule}_vecnorm.pkl")
    saved.close()
    return {
        "algorithm": algorithm,
        "configs": configs,
        "stance": stance_final.with_suffix(""),
        "walk": walk_final.with_suffix(""),
        "pre_rule": pre_rule,
        "stance_key": stance_key,
        "walk_key": walk_key,
        "root": root,
    }


def _envs(configs: dict, key: Any, algorithm: str) -> tuple[Any, Any]:
    from environments.shared.train_base import create_vec_env

    return tuple(  # type: ignore[return-value]
        create_vec_env(
            get_species_config(COMPSOGNATHUS_LONGIPES),
            configs,
            key,
            1,
            seed,
            algorithm=algorithm,
            plant_identity=current_plant_identity(COMPSOGNATHUS_LONGIPES),
        )
        for seed in (0, 1000)
    )


def test_a_pre_rule_sidecar_is_refused_for_a_live_stage_and_reimposed_for_a_none_one(parents, caplog):
    from environments.shared.train_base import _load_vecnorm_into_envs

    configs, algorithm = parents["configs"], parents["algorithm"]
    identity = current_plant_identity(COMPSOGNATHUS_LONGIPES)
    with open(f"{parents['pre_rule']}_vecnorm.pkl", "rb") as handle:
        saved = pickle.load(handle).obs_rms
    assert command_identity_gap(saved) == 1.0 - 1e-4 / saved.count

    live_envs = _envs(configs, parents["walk_key"], algorithm)
    try:
        before = [env.obs_rms for env in live_envs]
        for mode in ("initialize_next_stage", "resume_same_stage"):
            with pytest.raises(CommandNormalizationError, match=r"stance_final_vecnorm\.pkl.*'heading_and_speed'"):
                _load_vecnorm_into_envs(
                    str(parents["pre_rule"]),
                    *live_envs,
                    plant_identity=identity,
                    task_load_mode=mode,
                    command_mode="heading_and_speed",
                )
        assert [env.obs_rms for env in live_envs] == before, "nothing loaded"
    finally:
        for env in live_envs:
            env.close()

    none_envs = _envs(configs, _stage_key(configs, "recovery"), algorithm)
    try:
        with caplog.at_level("WARNING"):
            _load_vecnorm_into_envs(
                str(parents["pre_rule"]),
                *none_envs,
                plant_identity=identity,
                task_load_mode="initialize_next_stage",
                command_mode="none",
            )
        assert "identity re-imposed for this command-free stage" in caplog.text
        for env in none_envs:
            assert command_identity_gap(env.obs_rms) == 0.0
            np.testing.assert_array_equal(env.obs_rms.var[:-COMMAND_WIDTH], saved.var[:-COMMAND_WIDTH])
            np.testing.assert_array_equal(env.obs_rms.mean, saved.mean)
    finally:
        for env in none_envs:
            env.close()


def _warm_start(parents: dict, parent: str, child_text: str, tmp_path: Path) -> dict[str, Any]:
    """Warm-start a live child from *parent* through ``_train_stage_body`` (the body the CLI and the notebook train
    through), capturing the model and its statistics where learn() would start, before anything updates them."""
    child = _small(_live_config(tmp_path / "configs", COMPSOGNATHUS_LONGIPES, text=child_text), parents["algorithm"])
    return _enter(parents, parent, child, "initialize_next_stage", tmp_path)


def _enter(parents: dict, parent: str, config: dict, task_load_mode: str, tmp_path: Path) -> dict[str, Any]:
    """Load *parent*'s pair into the live walk stage *config* under *task_load_mode* through ``_train_stage_body``,
    capturing the model, its statistics and the callback list where learn() would start."""
    from environments.shared import train_base
    from environments.shared.policy_loading import _ensure_sb3

    sb3 = _ensure_sb3()
    algorithm = parents["algorithm"]
    configs = dict(parents["configs"])
    key = parents["walk_key"]
    configs[key] = config
    captured: dict[str, Any] = {}
    cls = sb3[algorithm.upper()]

    def capture(self, **kwargs):
        captured["model"], captured["normalizer"] = self, self.get_vec_normalize_env()
        captured["callback"] = kwargs["callback"]
        return self

    original = cls.learn
    cls.learn = capture
    log_path = tmp_path / "child"
    (log_path / "models").mkdir(parents=True)
    try:
        trained = train_base._train_stage_body(
            sb3,
            get_species_config(COMPSOGNATHUS_LONGIPES),
            configs,
            key,
            log_path=log_path,
            model_dir=log_path / "models",
            total_timesteps=64,
            n_envs=1,
            seed=9,
            eval_freq=10_000,
            save_freq=10_000,
            use_subproc=False,
            verbose=0,
            algorithm=algorithm,
            use_tensorboard=False,
            plant_identity=current_plant_identity(COMPSOGNATHUS_LONGIPES),
            task_fingerprint=stage_task_fingerprint(COMPSOGNATHUS_LONGIPES, "locomotion", stage_config=configs[key]),
            parent_id=None,
            load_path=str(parents[parent]),
            task_load_mode=task_load_mode,
            vecnorm_load_path=str(parents[parent]),
            allow_fresh_vecnorm=False,
            save_on_interrupt=False,
            announce=False,
        )
    finally:
        cls.learn = original
    captured["trained"] = trained
    captured["config"] = configs[key]
    return captured


def _probe_observations(config: dict, count: int = 48) -> np.ndarray:
    """Real observations of the live child's env, every one with a non-zero command (some drawn, some its own)."""
    env = get_species_config(COMPSOGNATHUS_LONGIPES).env_class(**config["env_kwargs"])
    rng = np.random.default_rng(3042)
    rows: list[np.ndarray] = []
    try:
        obs, _ = env.reset(seed=21)
        while len(rows) < count:
            row = obs.copy()
            if len(rows) % 2:
                row[-COMMAND_WIDTH:] = rng.uniform(-1.0, 1.0, COMMAND_WIDTH)
            if np.any(row[-COMMAND_WIDTH:]):
                rows.append(row)
            obs, _, terminated, truncated, _ = env.step(rng.uniform(-0.3, 0.3, env.action_space.shape))
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
    return np.asarray(rows, dtype=np.float32)


def _parent_pair(stem: Path) -> tuple[Any, Any]:
    from environments.shared.policy_loading import load_sb3_checkpoint

    def factory():
        return get_species_config(COMPSOGNATHUS_LONGIPES).env_class()

    model, normalizer, _ = load_sb3_checkpoint(f"{stem}.zip", f"{stem}_vecnorm.pkl", factory)
    return model, normalizer


def _close(captured: dict[str, Any]) -> None:
    captured["trained"].train_env.close()
    captured["trained"].eval_env.close()


def test_a_live_to_live_warm_start_acts_exactly_as_the_parent_on_non_zero_commands(parents, tmp_path):
    """Walk into follow (a widened command distribution): the parent's statistics load as they are (the slice at
    identity), its command columns are kept, and the child's normalised inputs and deterministic actions on
    non-zero commands are bitwise the parent's."""
    import torch

    from environments.shared.policy_loading import _command_input_widths, _command_model_algorithm
    from environments.shared.task_fingerprint import MODEL_TASK_LINEAGE_ATTRIBUTE

    captured = _warm_start(parents, "walk", FOLLOW_COMMANDS, tmp_path)
    try:
        child, normalizer = captured["model"], captured["normalizer"]
        parent, parent_normalizer = _parent_pair(parents["walk"])
        lineage = getattr(child, MODEL_TASK_LINEAGE_ATTRIBUTE)
        assert lineage["mode"] == "initialize_next_stage" and "zeroed_command_parameters" not in lineage
        assert lineage["parent_task_sha256"] != lineage["child_task_sha256"], "another live task"
        assert command_identity_gap(normalizer.obs_rms) == 0.0
        np.testing.assert_array_equal(normalizer.obs_rms.mean, parent_normalizer.obs_rms.mean)
        np.testing.assert_array_equal(normalizer.obs_rms.var, parent_normalizer.obs_rms.var)
        observation_dim = int(child.observation_space.shape[0])
        widths = _command_input_widths(child, _command_model_algorithm(child), observation_dim)
        columns = slice(observation_dim - COMMAND_WIDTH, observation_dim)
        for name in widths:
            kept = child.policy.get_submodule(name).weight[:, columns]
            assert torch.count_nonzero(kept) > 0, name
            assert torch.equal(kept, parent.policy.get_submodule(name).weight[:, columns]), name
        raw = _probe_observations(captured["config"])
        inputs = normalizer.normalize_obs(raw)
        np.testing.assert_array_equal(inputs, parent_normalizer.normalize_obs(raw))
        np.testing.assert_array_equal(inputs[:, -COMMAND_WIDTH:], raw[:, -COMMAND_WIDTH:])
        actions = child.predict(inputs, deterministic=True)[0]
        np.testing.assert_array_equal(actions, parent.predict(inputs, deterministic=True)[0])
        # The commands matter to this policy: zeroing them changes its actions.
        zero = raw.copy()
        zero[:, -COMMAND_WIDTH:] = 0.0
        assert not np.array_equal(child.predict(normalizer.normalize_obs(zero), deterministic=True)[0], actions)
    finally:
        _close(captured)


def test_a_stance_parent_still_has_its_command_columns_zeroed_and_proven_blind(parents, tmp_path):
    from environments.shared.task_fingerprint import MODEL_TASK_LINEAGE_ATTRIBUTE

    captured = _warm_start(parents, "stance", WALK_COMMANDS, tmp_path)
    try:
        child, normalizer = captured["model"], captured["normalizer"]
        parent, parent_normalizer = _parent_pair(parents["stance"])
        lineage = getattr(child, MODEL_TASK_LINEAGE_ATTRIBUTE)
        assert lineage["zeroed_command_parameters"], "the assert_command_blind proof ran"
        assert command_identity_gap(normalizer.obs_rms) == 0.0
        raw = _probe_observations(captured["config"])
        zero = raw.copy()
        zero[:, -COMMAND_WIDTH:] = 0.0
        actions = child.predict(normalizer.normalize_obs(raw), deterministic=True)[0]
        np.testing.assert_array_equal(
            actions, parent.predict(parent_normalizer.normalize_obs(zero), deterministic=True)[0]
        )
    finally:
        _close(captured)


def test_a_live_child_refuses_a_pre_rule_parent_before_it_trains(parents, tmp_path):
    with pytest.raises(CommandNormalizationError, match="not at identity"):
        _warm_start(parents, "pre_rule", WALK_COMMANDS, tmp_path)
    assert not list((tmp_path / "child" / "models").iterdir()), "nothing saved"


def test_a_live_stage_resumes_its_own_pair_with_the_hold_first(parents, tmp_path):
    """``resume_same_stage`` of the live walk's own pair: the statistics load as saved (the slice at identity, the
    count carried) into the training env and the evaluation env, the weights and command columns are the parent's
    (nothing zeroed), and learn() gets the callback list the CLI and the notebook train with, the hold first and
    holding that evaluation env."""
    import torch

    from environments.shared.curriculum import CommandIdentityCallback
    from environments.shared.task_fingerprint import MODEL_TASK_LINEAGE_ATTRIBUTE

    captured = _enter(parents, "walk", parents["configs"][parents["walk_key"]], "resume_same_stage", tmp_path)
    try:
        child, normalizer, trained = captured["model"], captured["normalizer"], captured["trained"]
        parent, parent_normalizer = _parent_pair(parents["walk"])
        assert "command" in child.mesozoic_task_fingerprint
        assert child.mesozoic_task_fingerprint == parent.mesozoic_task_fingerprint, "the same live task"
        assert "zeroed_command_parameters" not in (getattr(child, MODEL_TASK_LINEAGE_ATTRIBUTE, None) or {})
        for destination in (normalizer, trained.eval_env):
            assert command_identity_gap(destination.obs_rms) == 0.0
            np.testing.assert_array_equal(destination.obs_rms.mean, parent_normalizer.obs_rms.mean)
            np.testing.assert_array_equal(destination.obs_rms.var, parent_normalizer.obs_rms.var)
            assert destination.obs_rms.count == parent_normalizer.obs_rms.count > 64
        assert trained.eval_env.training is False
        for name, value in parent.policy.state_dict().items():
            assert torch.equal(child.policy.state_dict()[name], value), name
        callbacks = captured["callback"].callbacks
        assert isinstance(callbacks[0], CommandIdentityCallback) and callbacks[0].eval_env is trained.eval_env
        assert sum(isinstance(callback, CommandIdentityCallback) for callback in callbacks) == 1
    finally:
        _close(captured)


def test_a_warm_start_learns_with_the_hold_first(parents, tmp_path):
    """The list a walk-to-follow warm start hands learn() starts with the hold, which holds the stage's eval env."""
    from environments.shared.curriculum import CommandIdentityCallback

    captured = _warm_start(parents, "walk", FOLLOW_COMMANDS, tmp_path)
    try:
        callbacks = captured["callback"].callbacks
        assert isinstance(callbacks[0], CommandIdentityCallback)
        assert callbacks[0].eval_env is captured["trained"].eval_env
        assert command_identity_gap(captured["trained"].eval_env.obs_rms) == 0.0
    finally:
        _close(captured)
