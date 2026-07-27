#!/usr/bin/env python3
"""Report the joint envelope a trained policy actually uses.

``compare_run_diagnostics.py`` reports RMS action change per joint per step,
which says how *fast* the commands move but not how *far* they range.  Sizing
an action-space change needs the second number: if a balance policy never
leaves +/-15 degrees of knee travel, the other 35 degrees of commanded range
are dead weight that only amplifies jitter.

Reports, per actuated joint, the commanded envelope (from ``data.ctrl``) and
the achieved envelope (from ``data.qpos``), both in degrees, alongside the
per-step command delta.  Deterministic rollouts, matching the eval videos.

Note the analytic floor: a policy whose RMS action change per joint is ``r``
must sweep a commanded envelope of at least ``r`` in normalised units, since it
cannot move further per step than the range it visits.  A measured envelope at
that floor means tight oscillation; well above it means genuine repositioning.

Usage::

    python environments/shared/scripts/joint_excursion_report.py \\
        trex 1 /path/to/run/stage1/models/robust_best_model.zip

    # VecNormalize sidecar is found automatically when it sits next to the
    # model as ``*_vecnorm.pkl``; pass --vecnorm to override.
"""

import argparse
import importlib
import sys
import tomllib
from pathlib import Path

import mujoco
import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

SPECIES_ENVS = {
    "trex": ("environments.trex.envs.trex_env", "TRexEnv"),
    "velociraptor": ("environments.velociraptor.envs.raptor_env", "RaptorEnv"),
    "brachiosaurus": ("environments.brachiosaurus.envs.brachio_env", "BrachioEnv"),
    "dibothrosuchus": ("environments.dibothrosuchus.envs.dibothrosuchus_env", "DibothrosuchusEnv"),
}
STAGE_STEMS = {
    1: "stage1_balance",
    2: "stage2_locomotion",
    3: {
        "trex": "stage3_bite",
        "velociraptor": "stage3_strike",
        "brachiosaurus": "stage3_food_reach",
        "dibothrosuchus": "stage3_snap",
    },
}
# Keys the TOML [env] block carries for the MJX path only, as in
# zero_action_baseline.py.
JAX_ONLY_ENV_KEYS = frozenset({"foot_contact_weight", "foot_contact_gate"})


def build_env(species: str, stage: int):
    module_name, class_name = SPECIES_ENVS[species]
    env_class = getattr(importlib.import_module(module_name), class_name)
    stem = STAGE_STEMS[stage]
    if isinstance(stem, dict):
        stem = stem[species]
    config_path = Path(_repo_root) / "configs" / species / f"{stem}.toml"
    env_section = tomllib.loads(config_path.read_text()).get("env", {})
    kwargs = {
        key: (tuple(value) if isinstance(value, list) else value)
        for key, value in env_section.items()
        if key not in JAX_ONLY_ENV_KEYS
    }
    return env_class(**kwargs)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("species", choices=sorted(SPECIES_ENVS))
    parser.add_argument("stage", type=int, choices=(1, 2, 3))
    parser.add_argument("model", help="path to the SB3 .zip policy")
    parser.add_argument("--vecnorm", default=None, help="VecNormalize .pkl (default: alongside the model)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3042)
    args = parser.parse_args(argv)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = build_env(args.species, args.stage)
    model = PPO.load(args.model, device="cpu")

    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        guess = Path(args.model).with_name(Path(args.model).stem + "_vecnorm.pkl")
        vecnorm_path = str(guess) if guess.exists() else None
    normalizer = None
    if vecnorm_path:
        normalizer = VecNormalize.load(
            vecnorm_path, DummyVecEnv([lambda: build_env(args.species, args.stage)])
        )
        normalizer.training = False
        normalizer.norm_reward = False

    mj = env.model
    ctrl_log: list[np.ndarray] = []
    qpos_log: list[np.ndarray] = []

    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + i)
        while True:
            obs_input = normalizer.normalize_obs(obs) if normalizer is not None else obs
            action, _ = model.predict(obs_input, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            ctrl_log.append(env.data.ctrl.copy())
            qpos_log.append(env.data.qpos.copy())
            if terminated or truncated:
                break

    ctrl = np.asarray(ctrl_log)
    qpos = np.asarray(qpos_log)
    print(f"\n{args.species} stage {args.stage}: {len(ctrl)} steps over {args.episodes} episodes")
    print(f"  {'joint':<18}{'cmd p5-p95':>14}{'cmd full':>11}{'achieved':>11}"
          f"{'ctrl span':>11}{'used':>7}{'delta/step':>12}")

    for a in range(mj.nu):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        jid = int(mj.actuator_trnid[a, 0])
        qadr = mj.jnt_qposadr[jid]
        lo, hi = mj.actuator_ctrlrange[a]
        span = np.degrees(hi - lo)

        c = np.degrees(ctrl[:, a])
        q = np.degrees(qpos[:, qadr])
        band = np.percentile(c, 95) - np.percentile(c, 5)
        full = c.max() - c.min()
        achieved = q.max() - q.min()
        delta = float(np.sqrt(np.mean(np.diff(c, axis=0) ** 2)))
        print(f"  {name:<18}{band:11.1f} deg{full:8.1f} deg{achieved:8.1f} deg"
              f"{span:8.0f} deg{100 * full / span:6.0f}%{delta:9.1f} deg")

    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
