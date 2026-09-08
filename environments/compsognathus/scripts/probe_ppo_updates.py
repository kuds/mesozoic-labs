"""Compare early PPO updates; this is not convergence or stance-gate evidence.

python -m environments.compsognathus.scripts.probe_ppo_updates --output /tmp/compso-probe

Uses the current implementation for both arms. Only the original stance TOMLs
come from the pinned baseline commit. Each arm preserves the production rollout,
network, episode horizon, normalization, and complete learning-rate/entropy
schedules, then stops on a rollout boundary. No checkpoints are written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.evaluation import evaluate_policy  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import sync_envs_normalization  # noqa: E402

from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402
from environments.shared.train_base import (  # noqa: E402
    _maybe_ent_coef_decay_callback,
    _prepare_alg_kwargs,
    create_vec_env,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_REF = "21dfa4521b0822f0bee60be927893c2099c139c7"
N_ENVS = 4
N_EVAL_EPISODES = 10
SPECIES = ("compsognathus", "compsognathus_robot")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _seed(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 2**32 - 3001:
        raise argparse.ArgumentTypeError("must leave room for the held-out seed offset of 3000")
    return number


class BoundedProbePPO(PPO):
    """Stop before the next rollout without changing the full learning horizon."""

    def __init__(self, *args, probe_timesteps: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_timesteps = probe_timesteps
        self.update_records: list[dict] = []

    def collect_rollouts(self, *args, **kwargs):
        if self.num_timesteps >= self.probe_timesteps:
            return False
        return super().collect_rollouts(*args, **kwargs)

    def train(self):
        before = self._n_updates
        super().train()
        metrics = {
            key.removeprefix("train/"): value.item() if isinstance(value, np.generic) else value
            for key, value in self.logger.name_to_value.items()
            if key.startswith("train/")
        }
        metrics.update(
            timesteps=self.num_timesteps,
            effective_learning_rate=float(self.policy.optimizer.param_groups[0]["lr"]),
            effective_ent_coef=float(self.ent_coef),
            # SB3 counts an attempted epoch even if a minibatch triggers KL
            # stopping. This is not a count of fully completed epochs.
            epochs_counted=self._n_updates - before,
            policy_parameters_finite=all(bool(torch.isfinite(p).all()) for p in self.policy.parameters()),
        )
        self.update_records.append(metrics)


def _run_arm(species: str, arm: str, output: Path, baseline_ref: str, updates: int, seed: int) -> dict:
    stage_relative = f"configs/{species}/stance.toml"
    if arm == "baseline":
        config_bytes = subprocess.check_output(["git", "show", f"{baseline_ref}:{stage_relative}"], cwd=REPO_ROOT)
    else:
        config_bytes = (REPO_ROOT / stage_relative).read_bytes()
    config_path = output / f"{species}_{arm}.toml"
    config_path.write_bytes(config_bytes)
    config = load_stage_config(species, "stance", config_path=str(config_path))
    configs = {"stance": config}
    horizon = config["curriculum_kwargs"]["timesteps"]
    budget = updates * N_ENVS * config["ppo_kwargs"]["n_steps"]
    if budget > horizon:
        raise ValueError(f"Probe budget {budget} exceeds {species} {arm} stage budget {horizon}")
    species_cfg = get_species_config(species)
    env = create_vec_env(
        species_cfg, configs, "stance", N_ENVS, seed, algorithm="ppo", gamma=config["ppo_kwargs"]["gamma"]
    )
    eval_env = None
    try:
        model_kwargs, _, _ = _prepare_alg_kwargs(config, "ppo", 0, output, False)
        model = BoundedProbePPO("MlpPolicy", env, probe_timesteps=budget, seed=seed, device="cpu", **model_kwargs)
        model.set_logger(configure(format_strings=[]))
        callback = _maybe_ent_coef_decay_callback(config, "ppo", horizon)
        print(f"START {species} {arm} horizon={horizon} diagnostic_steps={budget}", flush=True)
        start = time.perf_counter()
        model.learn(total_timesteps=horizon, callback=callback)
        training_seconds = time.perf_counter() - start
        assert model.num_timesteps == budget
        assert len(model.update_records) == updates
        assert all(record["policy_parameters_finite"] for record in model.update_records)
        eval_env = create_vec_env(
            species_cfg, configs, "stance", 1, seed + 3000, algorithm="ppo", gamma=config["ppo_kwargs"]["gamma"]
        )
        sync_envs_normalization(env, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        eval_start = time.perf_counter()
        rewards, lengths = evaluate_policy(
            model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True, return_episode_rewards=True
        )
        evaluation_seconds = time.perf_counter() - eval_start
        assert np.all(np.isfinite(rewards))
    finally:
        if eval_env is not None:
            eval_env.close()
        env.close()
    return {
        "species": species,
        "arm": arm,
        "toml_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": config,
        "intended_stage_budget": horizon,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "updates": model.update_records,
        "evaluation": {
            "rewards": rewards,
            "lengths": lengths,
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),
            "full_horizon_fraction": float(np.mean(np.asarray(lengths) == config["env_kwargs"]["max_episode_steps"])),
            "certification_evaluation": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for configuration snapshots and results.json"
    )
    parser.add_argument(
        "--baseline-ref", default=BASELINE_REF, help="Git revision supplying only baseline stance TOMLs"
    )
    parser.add_argument("--updates", type=_positive_int, default=8, help="Completed rollout/update cycles per arm")
    parser.add_argument(
        "--seed", type=_seed, default=42, help="Training seed; deterministic evaluation uses seed + 3000"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    results = {
        "schema_version": 1,
        "purpose": "Early optimizer-update diagnostic, not a learning qualification or gate evaluation",
        "baseline_ref": args.baseline_ref,
        "working_tree_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        ),
        "python": sys.version,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("stable-baselines3", "torch", "numpy", "mujoco", "gymnasium")
        },
        "training_seed": args.seed,
        "evaluation_seed": args.seed + 3000,
        "n_envs": N_ENVS,
        "rollout_update_cycles_per_arm": args.updates,
        "deterministic_evaluation_episodes": N_EVAL_EPISODES,
        "evaluation_norm_reward": False,
        "evaluation_training": False,
        "notes": [
            "Every arm uses the current implementation, its own matched VecNormalize statistics, and its named production TOML configuration.",
            "Only the baseline TOML is fetched from baseline_ref; no historical library reconstruction is attempted.",
            "Production network, rollout size, minibatches, maximum epochs, initial action noise, and episode horizon are retained.",
            "Learning-rate progress uses the full configured stage budget; entropy uses the shared EntCoefDecayCallback.",
            "approx_kl and clip_fraction are the SB3 train logger values: approx_kl covers the last attempted epoch, not a strict trust-region bound.",
            "target_kl can stop an epoch after observing a large KL; it does not guarantee KL stays below the target.",
            "epochs_counted measures attempted epochs, not fully completed epochs; these logs cannot rule out a late minibatch KL stop.",
            "This short, single-seed probe diagnoses initial optimizer behavior; ten evaluation episodes are illustrative, not the stance certification panel.",
            "No checkpoints or production configuration files are written.",
        ],
        "arms": [],
    }
    for species in SPECIES:
        for arm in ("baseline", "candidate"):
            record = _run_arm(species, arm, output, args.baseline_ref, args.updates, args.seed)
            results["arms"].append(record)
            (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "species": species,
                        "arm": arm,
                        "training_seconds": record["training_seconds"],
                        "KL": [update["approx_kl"] for update in record["updates"]],
                        "clip_fraction": [update["clip_fraction"] for update in record["updates"]],
                        "evaluation": record["evaluation"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
