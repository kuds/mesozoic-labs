"""Regenerate the critic-phase archive fixture beside this script, with the code that wrote the phase-critic runs.

The stance runs that trained with ``[ppo] critic_phase`` were written by the never-merged screen branch
(claude/screen-trex-r10-phasecritic, 1dfe2327).  Their archives name ``environments.shared.critic_phase_ppo``'s
classes by reference, so what keeps them loading -- and dropping at a warm start -- is that module's names, kwargs
and record, which this archive pins.  Run it on a ``git archive`` export of that commit::

    git archive 1dfe2327 environments configs pyproject.toml | tar -x -C /some/empty/dir
    MUJOCO_GL= python -I -B make_fixture.py --tree /some/empty/dir

It trains a tiny PPO with the phase critic on a toy env of the T. rex interface's shapes (63 observations, 15
actions; no plant identity, so the fixture does not age with plant revisions; a constant learning rate, so the
archive carries no bytecode) for 64 steps, and writes ``ppo_critic_phase_1dfe2327.zip``, its
``ppo_critic_phase_1dfe2327_vecnorm.pkl`` and ``manifest.json``: the saving commit, Python, SB3 and torch
versions, the record, probe observations (the policy's normalised inputs), the deterministic actions the trained
model gave for them, and the critic's values on those observations with the settled phase appended.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STEM = "ppo_critic_phase_1dfe2327"
SOURCE_COMMIT = "1dfe2327f773007fa7fd68460dc0c53740936ff9"
SETTLE, HORIZON, GAMMA = 5, 12, 0.98


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tree", required=True, type=Path, help="a git archive export of 1dfe2327")
    tree = parser.parse_args().tree.resolve()
    sys.path.insert(0, str(tree))

    import gymnasium as gym
    import numpy as np
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from environments.shared import critic_phase as CP
    from environments.shared import critic_phase_ppo as CPP

    if not Path(CPP.__file__).resolve().is_relative_to(tree):
        raise SystemExit(f"critic_phase_ppo was imported from {CPP.__file__}, not from {tree}")
    torch.set_num_threads(1)

    class PhaseToyEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (63,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (15,), np.float32)

        def __init__(self) -> None:
            self.stance_width_settle_steps = SETTLE
            self._step_count = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._step_count = 0
            return self.np_random.normal(0.3, 1.5, 63).astype(np.float32), {}

        def step(self, action):
            self._step_count += 1
            obs = self.np_random.normal(0.3, 1.5, 63).astype(np.float32)
            gate = 1.0 if self._step_count > SETTLE else 0.0
            reward = float(gate - 0.1 * np.sum(np.square(action)) + obs[3] * action[4])
            return obs, reward, False, self._step_count >= HORIZON, {"step": self._step_count}

    venv = CPP.CriticPhaseVecNormalize(
        DummyVecEnv([lambda: Monitor(PhaseToyEnv()) for _ in range(2)]), critic_phase_gamma=GAMMA, gamma=GAMMA
    )
    venv.seed(7)
    spec = venv.critic_phase
    model: Any = PPO(
        env=venv,
        seed=7,
        n_steps=32,
        batch_size=32,
        n_epochs=2,
        learning_rate=3e-4,
        gamma=GAMMA,
        device="cpu",
        **CPP.new_model_kwargs(spec, {"net_arch": [8, 8], "log_std_init": -1.5}),
    )
    setattr(model, CP.CHECKPOINT_ATTRIBUTE, spec.record())  # as train_base records it
    model.learn(total_timesteps=64)
    model.save(str(HERE / f"{STEM}.zip"))
    venv.save(str(HERE / f"{STEM}_vecnorm.pkl"))

    probe = np.random.default_rng(0).normal(size=(4, 63)).astype(np.float32)
    actions, _ = model.predict(probe, deterministic=True)
    settled = np.concatenate([probe, spec.matrix([SETTLE] * 4).astype(np.float32)], axis=1)
    with torch.no_grad():
        values = model.policy.predict_values(torch.as_tensor(settled)).numpy().ravel()
    wide = model.policy.mlp_extractor.value_net[0].weight.detach().double()
    manifest = {
        "archive": f"{STEM}.zip",
        "sidecar": f"{STEM}_vecnorm.pkl",
        "source_commit": SOURCE_COMMIT,
        "python": platform.python_version(),
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "record": spec.record(),
        "policy_kwargs": {"net_arch": [8, 8], "log_std_init": -1.5, **spec.policy_kwargs()},
        "num_timesteps": int(model.num_timesteps),
        "observations": probe.tolist(),
        "actions": np.asarray(actions, dtype=np.float64).tolist(),
        "settled_values": values.astype(np.float64).tolist(),
        "phase_column_l2": torch.linalg.vector_norm(wide[:, 63:], dim=0).tolist(),
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {STEM}.zip, its sidecar and manifest.json from {tree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
