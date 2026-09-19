"""Regenerate the cross-interpreter SB3 archive fixtures beside this script.

Each archive is a tiny PPO or SAC model trained for a few steps on a 2-obs /
1-action box environment (never pickled into the archive) with the closure
``train_base.linear_schedule`` used to return as its learning rate (and clip
range for PPO), so its ``learning_rate`` / ``lr_schedule`` / ``clip_range``
members are cloudpickled closures carrying the SAVING interpreter's
bytecode -- the shape of every trex / compsognathus stage archive on Drive
trained before the loader PR. Run it once
per interpreter the fixtures should cover::

    MUJOCO_GL= python3.12 -m environments.shared.tests.fixtures.sb3_archives.make_fixtures
    MUJOCO_GL= python3.13 -m environments.shared.tests.fixtures.sb3_archives.make_fixtures

It writes ``<algo>_linear_py<major><minor>.zip`` and merges an entry into
``manifest.json``: the saving Python, SB3 and torch versions, the members
expected to carry bytecode, the probe observations and the deterministic
actions the freshly trained model produced for them. ``test_policy_loading``
loads every archive through :func:`~environments.shared.policy_loading.load_sb3_model`
under whatever interpreter runs the suite and checks those actions, and
shows the bare ``PPO.load`` dying in a subprocess on a foreign one.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO, SAC

from environments.shared.policy_loading import inspect_sb3_archive

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
PROBE_OBSERVATIONS = [[0.25, -0.5], [-0.75, 0.1], [0.0, 0.9], [0.6, 0.6]]


def legacy_linear_schedule(initial_lr: float, final_lr: float):
    """The closure ``train_base.linear_schedule`` returned before the loader PR, verbatim.

    Kept here (not imported) on purpose: ``train_base.linear_schedule`` now
    returns a ``LinearSchedule`` instance that pickles by reference, so a
    fresh archive carries no bytecode at all. Every archive on Drive that
    trained under a ``learning_rate_end`` before that change holds THIS
    closure, and the fixtures must reproduce them.
    """

    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)

    return schedule


class TinyBoxEnv(gym.Env):
    """Two observations, one action, eight-step episodes; only its spaces reach the archive."""

    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self) -> None:
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self.np_random.uniform(-1.0, 1.0, size=2).astype(np.float32), {}

    def step(self, action):
        self._t += 1
        observation = self.np_random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        return observation, float(-abs(float(action[0]))), self._t >= 8, False, {}


def _train(algorithm: str) -> Any:
    model: Any
    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            TinyBoxEnv(),
            learning_rate=legacy_linear_schedule(3e-4, 1e-5),
            clip_range=legacy_linear_schedule(0.2, 0.1),
            n_steps=32,
            batch_size=32,
            n_epochs=1,
            policy_kwargs={"net_arch": [8]},
            seed=0,
            device="cpu",
            verbose=0,
        )
        model.learn(64)
    else:
        model = SAC(
            "MlpPolicy",
            TinyBoxEnv(),
            learning_rate=legacy_linear_schedule(3e-4, 1e-5),
            buffer_size=256,
            learning_starts=16,
            train_freq=1,
            batch_size=16,
            policy_kwargs={"net_arch": [8]},
            seed=0,
            device="cpu",
            verbose=0,
        )
        model.learn(48)
    return model


def main(algorithms=("ppo", "sac")) -> None:
    tag = f"py{sys.version_info[0]}{sys.version_info[1]}"
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    for algorithm in algorithms:
        name = f"{algorithm}_linear_{tag}.zip"
        model = _train(algorithm)
        model.save(str(HERE / name))
        actions, _ = model.predict(np.asarray(PROBE_OBSERVATIONS, dtype=np.float32), deterministic=True)
        inspection = inspect_sb3_archive(HERE / name)
        if not inspection.bytecode_members:
            raise SystemExit(f"{name} carries no bytecode; the fixture must reproduce a closure-bearing archive")
        manifest[name] = {
            "algorithm": algorithm,
            "python": platform.python_version(),
            "python_minor": list(sys.version_info[:2]),
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "bytecode_members": sorted(inspection.bytecode_members),
            "observations": PROBE_OBSERVATIONS,
            "actions": np.asarray(actions, dtype=np.float64).ravel().tolist(),
        }
        print(f"wrote {name}: bytecode in {sorted(inspection.bytecode_members)}")
    MANIFEST.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")


if __name__ == "__main__":
    main()
