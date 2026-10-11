"""Reproducible, non-certifying experiments for HEADING_INVARIANCE_DESIGN_2026_10.md."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from collections import Counter
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
from prototype import HeadingObservation, transform_observation

from environments.compsognathus.envs.compsognathus_env import CompsognathusBiologicalEnv
from environments.shared.config import load_stage_config
from environments.shared.curriculum.stance_gate_v2 import StanceV2Thresholds, evaluate_stance_v2_gate, statue_reference
from environments.shared.plant_contract import current_plant_identity
from environments.shared.reporting.stance_report import SpawnYaw, _apply_spawn_yaw, _load_policy, run_panel
from environments.trex.envs.trex_env import TRexEnv

FACTORIES = {"trex": TRexEnv, "compsognathus": CompsognathusBiologicalEnv}
ANGLES = [-180.0, -90.0, -45.0, 0.0, 45.0, 90.0, 179.9]


def environment(species):
    return FACTORIES[species](**load_stage_config(species, "stance")["env_kwargs"])


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def provenance():
    return {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            n: importlib.metadata.version(n) for n in ["numpy", "mujoco", "gymnasium", "stable-baselines3", "torch"]
        },
        "scripts_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
        },
        "status": "research only; no curriculum or publication certificate",
    }


def pointwise(species, seed, samples=100):
    """Independent states with roll, pitch, yaw and velocities, plus physical reset sensors."""
    e = environment(species)
    rng = np.random.default_rng(seed)
    j = e.model.nq + e.model.nv - 13
    errors = {name: [] for name in ["legacy", "gravity", "quat"]}
    blocks = {
        name: 0.0
        for name in [
            "joints",
            "quaternion",
            "gyro",
            "linear_velocity",
            "accelerometer",
            "contact",
            "target",
            "distance_command",
        ]
    }
    slices = [
        slice(0, j),
        slice(j, j + 4),
        slice(j + 4, j + 7),
        slice(j + 7, j + 10),
        slice(j + 10, j + 13),
        slice(j + 13, j + 15),
        slice(j + 15, j + 18),
        slice(j + 18, None),
    ]
    for index in range(samples):
        e.reset(seed=seed + index)
        # Nonzero roll/pitch and velocities avoid a yaw-only or zero-velocity proof.
        angles = rng.uniform([-0.3, -0.3, -math.pi], [0.3, 0.3, math.pi])
        q = np.empty(4)
        mujoco.mju_euler2Quat(q, angles, "xyz")
        e.data.qpos[3:7] = q
        e.data.qvel[:6] = rng.uniform(-0.3, 0.3, 6)
        e.data.qpos[2] += 0.4  # kinematic probe: avoid comparing different contact-solver branches
        state = (
            e.data.qpos.copy(),
            e.data.qvel.copy(),
            e.data.mocap_pos.copy(),
            e.data.mocap_quat.copy(),
            e._initial_prey_dir_2d.copy(),
        )
        reference = _apply_spawn_yaw(e, SpawnYaw(0.0, True))
        candidates = {name: transform_observation(reference, j, candidate=name) for name in ["gravity", "quat"]}
        for yaw in ANGLES:
            e.data.qpos[:], e.data.qvel[:], e.data.mocap_pos[:], e.data.mocap_quat[:], e._initial_prey_dir_2d = [
                v.copy() for v in state
            ]
            obs = _apply_spawn_yaw(e, SpawnYaw(yaw, True))
            errors["legacy"].append(float(np.max(np.abs(obs - reference))))
            for name, sl in zip(blocks, slices):
                blocks[name] = max(blocks[name], float(np.max(np.abs(obs[sl] - reference[sl]))))
            for name in candidates:
                candidate = transform_observation(obs, j, candidate=name)
                errors[name].append(float(np.max(np.abs(candidate - candidates[name]))))
    e.close()
    return {
        "species": species,
        "states": samples,
        "rotations_per_state": len(ANGLES),
        "max_abs_observation_difference": {k: max(v) for k, v in errors.items()},
        "legacy_block_max_difference": blocks,
    }


def rollout_sweep(species, seed, episodes, model=None, vecnorm=None):
    cfg = load_stage_config(species, "stance")
    kwargs = cfg["env_kwargs"]
    thresholds = StanceV2Thresholds.from_curriculum(cfg["curriculum_kwargs"])
    identity = current_plant_identity(species)
    e = environment(species)
    dt, nu = e.dt, e.model.nu
    e.close()
    predictors = {"zero_action": lambda obs: np.zeros(nu)}
    if model:
        predictors["seed50_checkpoint"], _ = _load_policy(
            str(model), str(vecnorm), partial(FACTORIES[species], **kwargs), plant_identity=identity
        )
    rows = []
    for yaw in [-90.0, -45.0, 0.0, 45.0, 90.0, 180.0]:
        panels = {}
        for label, predict in predictors.items():
            panels[label] = run_panel(
                species,
                predict=predict,
                episodes=episodes,
                seed=seed,
                settle_steps=thresholds.settle_steps,
                horizon=1000,
                env_kwargs=kwargs,
                plant_identity=identity,
                control_dt=dt,
                floor_truth=True,
                spawn_yaw=SpawnYaw(yaw, True),
            )
        baseline = statue_reference(panels["zero_action"]["stance_metrics"], horizon=1000)
        for label, panel in panels.items():
            verdict = evaluate_stance_v2_gate(
                panel["stance_metrics"], thresholds, horizon=1000, statue=baseline, control_dt=dt
            )
            rows.append(
                {
                    "controller": label,
                    "yaw_deg": yaw,
                    "full_horizon": int((panel["lengths"] == 1000).sum()),
                    "episodes": episodes,
                    "mean_length": float(panel["lengths"].mean()),
                    "mean_reward": float(panel["rewards"].mean()),
                    "clean_count_current_v2": verdict.n_clean,
                    "terminations": panel["terminations"],
                    "criteria_failures": dict(
                        Counter(r.split()[0] for reasons in verdict.episode_reasons for r in reasons)
                    ),
                    "episode_lengths": panel["lengths"].tolist(),
                    "episode_rewards": panel["rewards"].tolist(),
                }
            )
            print(species, rows[-1], flush=True)
    return {
        "species": species,
        "seed_start": seed,
        "rows": rows,
        "model_sha256": None if model is None else hashlib.sha256(model.read_bytes()).hexdigest(),
        "vecnorm_sha256": None if vecnorm is None else hashlib.sha256(vecnorm.read_bytes()).hexdigest(),
    }


def paired_policy_rollouts(species, model, normalizer, seed, episodes):
    """Same fresh model and frozen normalizer at every yaw; compare whole trajectories."""
    rows = []
    for yaw in [-90.0, 0.0, 90.0, 180.0]:
        env = environment(species)
        j = env.model.nq + env.model.nv - 13
        for index in range(episodes):
            env.reset(seed=seed + index)
            obs = transform_observation(_apply_spawn_yaw(env, SpawnYaw(yaw, True)), j)
            total, length = 0.0, 0
            while True:
                action, _ = model.predict(normalizer.normalize_obs(obs), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                obs = transform_observation(obs, j)
                total += reward
                length += 1
                if terminated or truncated:
                    break
            rows.append(
                {
                    "yaw_deg": yaw,
                    "seed": seed + index,
                    "length": length,
                    "reward": total,
                    "termination": info.get("termination_reason", "truncated"),
                }
            )
        env.close()
    return rows


def pilot(species, training_seed, timesteps, evaluation_seed):
    """Integration/update smoke experiment, deliberately too short for a skill claim."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = load_stage_config(species, "stance")
    vec = VecNormalize(
        DummyVecEnv([lambda: HeadingObservation(environment(species)) for _ in range(2)]), norm_reward=True
    )
    ppo = dict(cfg["ppo_kwargs"])
    # A small network and rollout shorten the experiment; production budgets/hyperparameters are not validated here.
    policy_kwargs = dict(ppo.get("policy_kwargs", {}))
    policy_kwargs["net_arch"] = [64, 64]
    model = PPO(
        "MlpPolicy",
        vec,
        seed=training_seed,
        n_steps=256,
        batch_size=128,
        n_epochs=3,
        learning_rate=3e-5,
        gamma=0.98,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        device="cpu",
    )
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    model.learn(total_timesteps=timesteps)
    changed = sum(
        not np.array_equal(before[k].numpy(), v.detach().numpy()) for k, v in model.policy.state_dict().items()
    )
    vec.training = False
    vec.norm_reward = False
    rows = paired_policy_rollouts(species, model, vec, evaluation_seed, 4)
    vec.close()
    return {
        "species": species,
        "training_seed": training_seed,
        "training_steps": model.num_timesteps,
        "changed_parameter_tensors": changed,
        "network": [64, 64],
        "production_recipe": False,
        "rows": rows,
    }


def main():
    import torch

    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pointwise", "rollouts", "pilot"])
    parser.add_argument("--species", choices=list(FACTORIES), required=True)
    parser.add_argument("--seed", type=int, default=18042)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--training-seed", type=int, default=11)
    parser.add_argument("--timesteps", type=int, default=8192)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--vecnorm", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    meta = provenance()
    if args.mode == "pointwise":
        result = pointwise(args.species, args.seed)
    elif args.mode == "rollouts":
        if bool(args.model) != bool(args.vecnorm):
            parser.error("model and vecnorm must be supplied together")
        result = rollout_sweep(args.species, args.seed, args.episodes, args.model, args.vecnorm)
    else:
        result = pilot(args.species, args.training_seed, args.timesteps, args.seed)
    meta.update(
        {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "duration_seconds": time.monotonic() - start,
            "result": result,
        }
    )
    save(args.output, meta)
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
