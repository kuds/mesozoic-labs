"""The critic phase's off path is main's: with no ``[ppo] critic_phase`` key, the trainer computes what main computed
at 4c9d5e9b, before the option existed.  Needs stable-baselines3 (the SB3 job).

- One subprocess runs ``critic_phase_off_scenario.py`` on this repository: the T. rex stance from scratch, a
  locomotion warm start across the node's edge, and a same-stage resume, through ``train_base.train``.
  - Anywhere: the phase module is never imported, the stock classes are built, nothing new is recorded, and the
    structure (classes, ``stage_config.json`` key sets and recipe digests, the archives' ``data`` keys) equals the
    one main recorded (``fixtures/critic_phase_off/golden_4c9d5e9b.json``).
  - In the environment the golden was made in (Python, platform, torch and its CPU capability, numpy, MuJoCo,
    gymnasium, SB3): every rollout buffer, parameter, optimizer tensor, normaliser statistic and sidecar is bitwise
    main's.  Elsewhere that comparison skips and names the fields that differ.
- In process: the trainer's load path on a stock archive trains bitwise as ``PPO.load`` does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[3]
SCENARIO = Path(__file__).with_name("critic_phase_off_scenario.py")
GOLDEN = Path(__file__).parent / "fixtures" / "critic_phase_off" / "golden_4c9d5e9b.json"
STOCK = {
    "policy": "stable_baselines3.common.policies.ActorCriticPolicy",
    "rollout_buffer": "stable_baselines3.common.buffers.RolloutBuffer",
    "vec_normalize": "stable_baselines3.common.vec_env.vec_normalize.VecNormalize",
}


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    record: dict[str, Any] = json.loads(GOLDEN.read_text())
    return record


@pytest.fixture(scope="module")
def off_run(tmp_path_factory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("critic_phase_off") / "head.json"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(SCENARIO), str(REPOSITORY), str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    result: dict[str, Any] = json.loads(out.read_text())
    return result


def test_the_golden_is_main_s_before_the_option_existed(golden):
    assert golden["source_commit"].startswith("4c9d5e9b")
    assert set(golden["scenarios"]) == {"S1", "S2", "S3"} and golden["phase_module_imported"] is False


def test_the_off_path_never_imports_the_phase_module_and_builds_the_stock_classes(off_run):
    assert off_run["phase_module_imported"] is False
    for name, scenario in off_run["scenarios"].items():
        structure = scenario["structure"]
        assert structure["model"] == "PPO", name  # main's T. rex stages declare no [ppo] symmetry
        assert {key: structure[key] for key in STOCK} == STOCK, name
        assert structure["buffer_obs_shape"] == [63], name
        assert structure["archive_policy_module"] == "stable_baselines3.common.policies", name
        assert structure["archive_buffer_module"] == "stable_baselines3.common.buffers", name
        assert structure["archive_buffer_kwargs"] == {}, name
        # Nothing new is recorded: no switch in the table, no drop or reset in the run block, no phase record.
        recorded = (
            structure["hyperparameter_keys"]
            + structure["run_keys"]
            + structure["archive_data_keys"]
            + structure["archive_policy_kwargs"]
        )
        assert not [key for key in recorded if "critic_phase" in key or "warm_start" in key], name


def test_the_off_path_records_main_s_structure(off_run, golden):
    import stable_baselines3

    if stable_baselines3.__version__ != golden["environment"]["stable_baselines3"]:
        pytest.skip(f"the golden's archive keys are SB3 {golden['environment']['stable_baselines3']}'s")
    for name, scenario in golden["scenarios"].items():
        assert off_run["scenarios"][name]["structure"] == scenario["structure"], name


def test_the_off_path_trains_bitwise_as_main_did(off_run, golden):
    differing = sorted(key for key, value in golden["environment"].items() if off_run["environment"].get(key) != value)
    if differing:
        pytest.skip(
            "the golden digests bind the environment they were made in; this one differs in "
            + ", ".join(
                f"{key} ({off_run['environment'].get(key)!r}, golden {golden['environment'][key]!r})"
                for key in differing
            )
        )
    mismatched = {
        name: sorted(
            key
            for key in set(scenario["digests"]) | set(off_run["scenarios"][name]["digests"])
            if scenario["digests"].get(key) != off_run["scenarios"][name]["digests"].get(key)
        )
        for name, scenario in golden["scenarios"].items()
    }
    assert not any(mismatched.values()), mismatched


# ---------------------------------------------------------------------------------------------------- the load path
class ToyEnv(gym.Env):
    """The T. rex interface's shapes (63 observations, 15 actions), random dynamics, 12-step episodes."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (63,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (15,), np.float32)

    def __init__(self):
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self.np_random.normal(0.3, 1.5, 63).astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = self.np_random.normal(0.3, 1.5, 63).astype(np.float32)
        reward = float(-0.1 * np.sum(np.square(action)) + obs[3] * action[4])
        return obs, reward, False, self._step_count >= 12, {"step": self._step_count}


def _venv(seed: int) -> VecNormalize:
    venv = VecNormalize(DummyVecEnv([lambda: Monitor(ToyEnv()) for _ in range(2)]), gamma=0.995)
    venv.seed(seed)
    return venv


_KWARGS: dict[str, Any] = {
    "n_steps": 32,
    "batch_size": 32,
    "n_epochs": 2,
    "gamma": 0.995,
    "learning_rate": 5e-5,
    "target_kl": None,
    "device": "cpu",
}


def _state(model) -> dict[str, np.ndarray]:
    arrays = {f"param/{k}": v.detach().numpy().copy() for k, v in model.policy.state_dict().items()}
    for key, state in model.policy.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            arrays[f"optimizer/{key}/{name}"] = np.asarray(value).copy()
    buffer = model.rollout_buffer
    for name in ("observations", "actions", "rewards", "values", "log_probs", "advantages", "returns"):
        arrays[f"buffer/{name}"] = np.array(getattr(buffer, name), copy=True)
    arrays["num_timesteps"] = np.asarray(model.num_timesteps)
    return arrays


@pytest.mark.parametrize("mode", ["resume_same_stage", "initialize_next_stage"])
def test_the_trainer_s_load_of_a_stock_archive_trains_bitwise_as_ppo_load(mode, tmp_path):
    """``_create_or_load_model`` with no phase on a stock archive (the path the warm-start drop and the log-std reset
    sit beside) gives the parameters, optimizer state and rollout buffer of ``PPO.load`` then ``learn``, bitwise."""
    from environments.shared.policy_loading import _ensure_sb3
    from environments.shared.train_base import _create_or_load_model

    parent = PPO("MlpPolicy", _venv(3), seed=3, policy_kwargs={"net_arch": [32, 32], "log_std_init": -1.5}, **_KWARGS)
    parent.learn(total_timesteps=64)
    parent.save(tmp_path / "parent.zip")

    trainer = _create_or_load_model(
        _ensure_sb3(),
        "ppo",
        {**_KWARGS, "seed": 21, "policy_kwargs": {"net_arch": [8]}},  # a loaded model keeps its own architecture
        _venv(21),
        str(tmp_path / "parent.zip"),
        task_load_mode=mode,
    )
    assert type(trainer) is PPO and "mesozoic_critic_phase" not in vars(trainer)
    trainer.learn(total_timesteps=64, reset_num_timesteps=mode == "initialize_next_stage")
    direct = PPO.load(str(tmp_path / "parent.zip"), env=_venv(21), seed=21, **_KWARGS)
    direct.learn(total_timesteps=64, reset_num_timesteps=mode == "initialize_next_stage")
    a, b = _state(trainer), _state(direct)
    assert a.keys() == b.keys()
    assert [key for key in a if not np.array_equal(a[key], b[key])] == []
