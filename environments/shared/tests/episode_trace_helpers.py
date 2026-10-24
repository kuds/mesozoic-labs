"""A tiny real T. rex checkpoint for the episode trace and the replay's command columns.

A PPO on the T. rex stance env, shortened to :data:`HORIZON` steps, whose
VecNormalize statistics have seen a few zero-action steps and whose
``tail_3_pitch_act`` output is biased to +1.1: the raw Gaussian mean then
sits beyond the action bound and the command on the rail, the shape of the
stance habit the trace exists to measure.  Both artifacts carry the
model's current plant identity, so every plant check runs as on a real one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest

from environments.shared.model_id import ModelId

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
#: The shortened horizon and settle window the tests roll.
HORIZON = 60
SETTLE_STEPS = 20
#: The actuator biased past the bound, and its bias.
BIASED_ACTUATOR = "tail_3_pitch_act"
BIAS = 1.1


class TinyCheckpoint(NamedTuple):
    model: Path
    vecnorm: Path
    stage_config: dict[str, Any]


def short_stance_config() -> dict[str, Any]:
    """The T. rex stance stage's config at :data:`HORIZON` steps with a :data:`SETTLE_STEPS` window."""
    from environments.shared.config import load_stage_config

    config = load_stage_config(TYRANNOSAURUS_REX, 1)
    config["env_kwargs"] = {**config["env_kwargs"], "max_episode_steps": HORIZON}
    config["curriculum_kwargs"] = {**config["curriculum_kwargs"], "settle_steps": SETTLE_STEPS}
    return config


def tiny_checkpoint(directory: Path) -> TinyCheckpoint:
    """Save the tiny checkpoint and its statistics under *directory*."""
    pytest.importorskip("mujoco")
    pytest.importorskip("stable_baselines3")
    import mujoco
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
    from environments.shared.species_registry import env_class

    config = short_stance_config()
    env_kwargs = config["env_kwargs"]
    cls = env_class(TYRANNOSAURUS_REX)
    normalizer = VecNormalize(DummyVecEnv([lambda: cls(**env_kwargs)]))
    try:
        model = PPO(
            "MlpPolicy",
            normalizer,
            seed=7,
            device="cpu",
            n_steps=16,
            batch_size=8,
            policy_kwargs={"net_arch": [16, 16]},
        )
        probe = cls(**env_kwargs)
        nu = int(probe.unwrapped.model.nu)
        index = mujoco.mj_name2id(probe.unwrapped.model, mujoco.mjtObj.mjOBJ_ACTUATOR, BIASED_ACTUATOR)
        probe.close()
        normalizer.reset()
        for _ in range(30):
            normalizer.step(np.zeros((1, nu), dtype=np.float32))
        action_net: Any = model.policy.action_net
        with torch.no_grad():
            action_net.bias[index] = BIAS
        identity = current_plant_identity(TYRANNOSAURUS_REX)
        for artifact in (model, normalizer):
            attach_plant_identity(artifact, identity)
        paths = TinyCheckpoint(directory / "model.zip", directory / "model_vecnorm.pkl", config)
        model.save(paths.model)
        normalizer.save(str(paths.vecnorm))
    finally:
        normalizer.close()
    return paths
