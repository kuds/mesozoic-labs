"""Prepare and run the opt-in Compsognathus filter × stance-reward study.

Each arm/seed is an independent fresh PPO run. Saved artifacts have explicit
study identities and cannot replace registered production checkpoints.
Short probes retain the complete learning-rate and entropy schedules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.compsognathus.experiments.balance_env import ARMS, make_balance_env
from environments.compsognathus.experiments.balance_identity import (
    attach_study_identity,
    build_study_identity,
    load_study_checkpoint,
    save_study_checkpoint,
    study_source_fingerprint,
    validate_study_identity,
)
from environments.compsognathus.experiments.balance_metrics import (
    BalanceTargets,
    evaluate_balance_episode,
    summarize_balance_panel,
)
from environments.shared.config import get_git_commit, load_stage_config
from environments.shared.constants import DEFAULT_CLIP_OBS, DEFAULT_CLIP_REWARD
from environments.shared.curriculum.stance_gate import StanceGateThresholds
from environments.shared.train_base import _maybe_ent_coef_decay_callback, _prepare_alg_kwargs

SCHEMA = "compsognathus.learned-balance-study/v1"


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_study_plan(*, training_seeds: tuple[int, ...] = (42, 43, 44)) -> dict:
    """Freeze matched budgets and disjoint screening/confirmation seed panels."""
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    if any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= (2**32 - 4) // 4 for s in training_seeds):
        raise ValueError("Training seeds must be integers with room for four environment seeds")
    evaluation_seeds = set(range(6042, 6082)) | set(range(7142, 7182))
    evaluation_seeds |= set(range(9042, 9082)) | set(range(10042, 10082))
    worker_seeds = {str(seed): [4 * seed + rank for rank in range(4)] for seed in training_seeds}
    if any(worker in evaluation_seeds for workers in worker_seeds.values() for worker in workers):
        raise ValueError("Training environment seeds overlap a screening or confirmation panel")
    config = load_stage_config("compsognathus", "stance")
    return _json_value(
        {
            "schema": SCHEMA,
            "source_commit": get_git_commit(),
            "implementation_sha256": study_source_fingerprint(),
            "runtime": {
                "python": platform.python_version(),
                **{
                    package: version(package)
                    for package in ("mujoco", "numpy", "torch", "stable-baselines3", "gymnasium")
                },
            },
            "purpose": "Learn quiet bilateral stance on canonical foot mechanics",
            "research_only": True,
            "arms": {name: asdict(arm) for name, arm in ARMS.items()},
            "training_seeds": training_seeds,
            "training_environment_seeds": worker_seeds,
            "n_envs": 4,
            "vector_backend": "DummyVecEnv",
            "algorithm": "PPO",
            "normalization": {
                "norm_obs": True,
                "norm_reward": True,
                "clip_obs": DEFAULT_CLIP_OBS,
                "clip_reward": DEFAULT_CLIP_REWARD,
                "gamma": config["ppo_kwargs"]["gamma"],
            },
            "stage_config": config,
            "training_budget": config["curriculum_kwargs"]["timesteps"],
            "screen_every_steps": 250_000,
            "screen_seeds": list(range(6042, 6082)),
            "confirmation_seeds": list(range(7142, 7182)),
            "probe_screen_seeds": list(range(9042, 9082)),
            "probe_confirmation_seeds": list(range(10042, 10082)),
            "behavior_targets": asdict(BalanceTargets()),
            "selection": "lexicographic physical balance selection_key; ties retain the earlier checkpoint",
            "projected_gate": "Unchanged stance criteria applied to canonical base reward, separately from training reward",
            "reward_references": {
                "schema": "compsognathus.balance-reward-reference/v1",
                "canonical_theoretical_episode_ceiling": 3000.0,
                "shaped_theoretical_episode_ceiling": 3750.0,
                "canonical_projected_reward_floor": 1800.0,
                "note": "Ceilings apply to the captured 1000-step zero-forward stance recipe. Measured home references are saved by calibrate.",
            },
            "notes": [
                "Arms differ only in the declared global filter and additive stance shaping; all retain 14 actuators.",
                "Targets are proposed study criteria, not validated hardware thresholds or production curriculum gates.",
                "Confirmation evaluates only the selected checkpoint, once, on the separate 40-seed panel.",
                "Small probes and home controllers never establish learned qualification.",
                "No checkpoint continuation or old-policy initialization is supported in v1.",
            ],
        }
    )


def prepare_study(output: Path, *, training_seeds: tuple[int, ...] = (42, 43, 44)) -> dict:
    output = Path(output)
    plan = make_study_plan(training_seeds=training_seeds)
    path = output / "study_plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Study plan differs from this source/config; choose a new output directory")
    else:
        _save(path, plan)
    return plan


def _load_plan(output: Path) -> dict:
    plan = json.loads((output / "study_plan.json").read_text())
    if plan.get("schema") != SCHEMA:
        raise ValueError("Unsupported study plan schema")
    current = make_study_plan(training_seeds=tuple(plan["training_seeds"]))
    if plan != current:
        raise ValueError("Prepared study plan no longer matches source/config; prepare a new study")
    return plan


class BalanceStudyPPO(PPO):
    """Stop probes after complete updates, keeping the full schedule horizon."""

    def __init__(self, *args, stop_after_steps: int, after_update=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_after_steps = stop_after_steps
        self.after_update = after_update
        self.update_records: list[dict] = []

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["after_update", "update_records"]

    def collect_rollouts(self, *args, **kwargs):
        if self.num_timesteps >= self.stop_after_steps:
            return False
        return super().collect_rollouts(*args, **kwargs)

    def train(self):
        super().train()
        record = {
            "timesteps": self.num_timesteps,
            "learning_rate": float(self.policy.optimizer.param_groups[0]["lr"]),
            "entropy_coefficient": float(self.ent_coef),
            "learned_action_std": float(torch.exp(self.policy.log_std).mean().detach().cpu()),
            "finite_parameters": all(bool(torch.isfinite(p).all()) for p in self.policy.parameters()),
            **{k: v for k, v in self.logger.name_to_value.items() if k.startswith("train/")},
        }
        self.update_records.append(_json_value(record))
        if not record["finite_parameters"]:
            raise FloatingPointError("Nonfinite policy parameters")
        if self.after_update is not None:
            self.after_update(self)


def _vector_env(arm: str, env_kwargs: dict, n_envs: int, seed: int) -> DummyVecEnv:
    env = DummyVecEnv([lambda: Monitor(make_balance_env(arm, **env_kwargs)) for _ in range(n_envs)])
    env.seed(seed)
    return env


def _evaluate(model, normalizer, arm: str, plan: dict, seeds: list[int], *, trace_path: Path | None = None) -> dict:
    config = plan["stage_config"]
    env = make_balance_env(arm, **config["env_kwargs"])
    was_training, was_norm_reward = normalizer.training, normalizer.norm_reward
    normalizer.training = False
    normalizer.norm_reward = False
    try:

        def predict(observation):
            return model.predict(normalizer.normalize_obs(observation), deterministic=True)[0]

        rows = [
            evaluate_balance_episode(
                env,
                predict,
                seed,
                trace_path=trace_path if index == 0 else None,
                settle_steps=config["curriculum_kwargs"]["settle_steps"],
            )
            for index, seed in enumerate(seeds)
        ]
        summary = summarize_balance_panel(
            rows,
            horizon=config["env_kwargs"]["max_episode_steps"],
            stance_thresholds=StanceGateThresholds.from_curriculum(config["curriculum_kwargs"]),
            targets=BalanceTargets(**plan["behavior_targets"]),
        )
        return {"episodes": rows, "summary": summary}
    finally:
        normalizer.training, normalizer.norm_reward = was_training, was_norm_reward
        env.close()


def train_balance_arm(
    output: Path,
    arm: str,
    seed: int,
    *,
    probe_updates: int | None = None,
    evaluation_episodes: int | None = None,
) -> dict:
    """Train one prepared arm/seed; full runs always confirm on 40 fresh seeds."""
    output = Path(output)
    plan = _load_plan(output)
    if arm not in ARMS or seed not in plan["training_seeds"]:
        raise ValueError("Arm/seed is not in the prepared study")
    if probe_updates is not None and (
        isinstance(probe_updates, bool) or not isinstance(probe_updates, int) or probe_updates <= 0
    ):
        raise ValueError("probe_updates must be positive")
    if evaluation_episodes is not None and (
        probe_updates is None
        or isinstance(evaluation_episodes, bool)
        or not isinstance(evaluation_episodes, int)
        or not 1 <= evaluation_episodes <= 40
    ):
        raise ValueError("Only probes may override evaluation episodes, within 1–40")
    torch.set_num_threads(1)
    config = plan["stage_config"]
    rollout = plan["n_envs"] * config["ppo_kwargs"]["n_steps"]
    budget = int(plan["training_budget"])
    stop = math.ceil(budget / rollout) * rollout if probe_updates is None else probe_updates * rollout
    if probe_updates is not None and stop > budget:
        raise ValueError("Probe exceeds the full study budget")
    tag = f"{arm}_seed{seed}" + (f"_probe{probe_updates}" if probe_updates is not None else "")
    run_dir = output / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = {"plan": plan, "arm": arm, "seed": seed, "probe_updates": probe_updates}
    bare = make_balance_env(arm, **config["env_kwargs"])
    try:
        identity = build_study_identity(bare, snapshot)
    finally:
        bare.close()
    _save(run_dir / "study_identity.json", identity)
    _save(run_dir / "run_config.json", snapshot)
    worker_seed = plan["training_environment_seeds"][str(seed)][0]
    vec = _vector_env(arm, config["env_kwargs"], plan["n_envs"], worker_seed)
    normalizer = VecNormalize(vec, **plan["normalization"])
    attach_study_identity(normalizer, identity)
    model_kwargs, _, _ = _prepare_alg_kwargs(config, "ppo", 0, run_dir, False)
    records: list[dict] = []
    best_key: tuple | None = None
    selected: Path | None = None
    next_screen = int(plan["screen_every_steps"])
    n_eval = evaluation_episodes or 40
    screen_seeds = plan["probe_screen_seeds" if probe_updates is not None else "screen_seeds"][:n_eval]
    confirmation_seeds = plan["probe_confirmation_seeds" if probe_updates is not None else "confirmation_seeds"][
        :n_eval
    ]
    started = time.perf_counter()

    def after_update(model):
        nonlocal next_screen, best_key, selected
        if model.num_timesteps < next_screen and model.num_timesteps < stop:
            return
        checkpoint = run_dir / "checkpoints" / f"step_{model.num_timesteps}"
        panel = _evaluate(model, normalizer, arm, plan, screen_seeds)
        save_study_checkpoint(model, normalizer, checkpoint, identity)
        _save(checkpoint.with_suffix(".screen.json"), panel)
        key = tuple(panel["summary"]["selection_key"])
        if best_key is None or key > best_key:
            best_key, selected = key, checkpoint
        records.append({"timesteps": model.num_timesteps, "checkpoint": checkpoint.name, **panel["summary"]})
        _save(run_dir / "screening_history.json", records)
        _save(run_dir / "updates.json", model.update_records)
        next_screen = (model.num_timesteps // plan["screen_every_steps"] + 1) * plan["screen_every_steps"]
        print(
            json.dumps(
                _json_value({"arm": arm, "seed": seed, "steps": model.num_timesteps, "screen": panel["summary"]})
            ),
            flush=True,
        )

    model = None
    try:
        model = BalanceStudyPPO(
            "MlpPolicy",
            normalizer,
            seed=seed,
            device="cpu",
            stop_after_steps=stop,
            after_update=after_update,
            **model_kwargs,
        )
        # PPO construction reseeds its vector env with the policy seed. Restore
        # disjoint worker streams before learn() performs the first reset.
        normalizer.seed(worker_seed)
        attach_study_identity(model, identity)
        model.set_logger(configure(str(run_dir / "training_logs"), ["csv"]))
        print(
            json.dumps(
                {"event": "start", "arm": arm, "seed": seed, "planned_steps": stop, "full_schedule_steps": budget}
            ),
            flush=True,
        )
        model.learn(total_timesteps=budget, callback=_maybe_ent_coef_decay_callback(config, "ppo", budget))
        if selected is None:
            raise RuntimeError("Training did not produce a screening checkpoint")
        confirmation_env = _vector_env(arm, config["env_kwargs"], 1, confirmation_seeds[0])
        loaded_normalizer = None
        try:
            loaded_model, loaded_normalizer = load_study_checkpoint(selected, identity, confirmation_env)
            confirmation = _evaluate(
                loaded_model,
                loaded_normalizer,
                arm,
                plan,
                confirmation_seeds,
                trace_path=run_dir / "selected_trace.csv",
            )
        finally:
            if loaded_normalizer is not None:
                loaded_normalizer.close()
            else:
                confirmation_env.close()
        _save(run_dir / "confirmation.json", confirmation)
        result = {
            "schema": SCHEMA,
            "arm": arm,
            "seed": seed,
            "training_steps": model.num_timesteps,
            "training_seconds": time.perf_counter() - started,
            "diagnostic_probe": probe_updates is not None,
            "selected_checkpoint": str(selected.relative_to(run_dir)),
            "confirmation": confirmation["summary"],
            "learned_balance_qualified": probe_updates is None and confirmation["summary"]["behavior_qualified"],
            "research_only": True,
            "production_advancement": False,
        }
        _save(run_dir / "run_summary.json", result)
        return result
    except Exception as exc:
        _save(run_dir / "run_failure.json", {"type": type(exc).__name__, "message": str(exc), "arm": arm, "seed": seed})
        raise
    finally:
        if model is not None:
            model.logger.close()
        normalizer.close()


def summarize_study(output: Path) -> dict:
    """Include every declared run, including missing, failed and unfinished runs."""
    output = Path(output)
    plan = _load_plan(output)
    rows = []
    for arm in plan["arms"]:
        for seed in plan["training_seeds"]:
            directory = output / "runs" / f"{arm}_seed{seed}"
            summary = directory / "run_summary.json"
            failure = directory / "run_failure.json"
            if summary.exists():
                result = json.loads(summary.read_text())
                snapshot = {"plan": plan, "arm": arm, "seed": seed, "probe_updates": None}
                if json.loads((directory / "run_config.json").read_text()) != snapshot:
                    raise ValueError(f"Completed run does not match this study: {directory.name}")
                env = make_balance_env(arm, **plan["stage_config"]["env_kwargs"])
                try:
                    validate_study_identity(
                        json.loads((directory / "study_identity.json").read_text()),
                        build_study_identity(env, snapshot),
                    )
                finally:
                    env.close()
                rollout = plan["n_envs"] * plan["stage_config"]["ppo_kwargs"]["n_steps"]
                expected_steps = math.ceil(plan["training_budget"] / rollout) * rollout
                if (
                    result["arm"] != arm
                    or result["seed"] != seed
                    or result["diagnostic_probe"]
                    or result["training_steps"] != expected_steps
                ):
                    raise ValueError(f"Completed run is not a full matched arm/seed: {directory.name}")
                confirmation = result["confirmation"]
                rows.append(
                    {
                        **result,
                        "status": "complete",
                        "full_horizon_fraction": confirmation["projected_stance_panel"]["full_horizon_fraction"],
                        "mean_canonical_reward": confirmation["projected_stance_panel"]["mean_reward"],
                        **confirmation["behavior_means"],
                    }
                )
            else:
                status = "failed" if failure.exists() else "unfinished" if directory.exists() else "not_started"
                rows.append({"arm": arm, "seed": seed, "status": status, "learned_balance_qualified": False})
    result = {
        "schema": SCHEMA,
        "expected_runs": len(rows),
        "runs": rows,
        "all_runs_complete": all(r["status"] == "complete" for r in rows),
    }
    _save(output / "comparison.json", result)
    fields = [
        "arm",
        "seed",
        "status",
        "training_steps",
        "learned_balance_qualified",
        "selected_checkpoint",
        "full_horizon_fraction",
        "mean_canonical_reward",
        "physics_bilateral_20pct_weight",
        "physics_unsupported",
        "weaker_foot_load_bw_mean",
        "physics_load_imbalance_mean",
        "pelvis_angular_velocity_rms",
        "com_vertical_std_over_home_height",
        "com_xy_sway_rms_over_home_height",
        "sole_pitch_motion_rms_rad",
        "max_foot_xy_drift_over_home_height",
        "positive_actuator_work_J",
        "joint_hard_stop_duty",
    ]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "calibrate", "train", "summarize"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--probe-updates", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare_study(args.output, training_seeds=tuple(args.training_seeds))
    elif args.mode == "calibrate":
        from environments.compsognathus.experiments.balance_calibration import (
            home_reference_report,
            reward_ordering_report,
        )

        plan = _load_plan(args.output)
        result = {
            "schema": "compsognathus.balance-calibration/v1",
            "ordering": reward_ordering_report(),
            "home_reference": home_reference_report(
                seeds=tuple(range(8042, 8082)),
                env_kwargs=plan["stage_config"]["env_kwargs"],
                settle_steps=plan["stage_config"]["curriculum_kwargs"]["settle_steps"],
            ),
            "learned_policy_qualification": False,
        }
        _save(args.output / "calibration.json", result)
    elif args.mode == "train":
        if args.arm is None:
            parser.error("train requires --arm")
        result = train_balance_arm(
            args.output,
            args.arm,
            args.seed,
            probe_updates=args.probe_updates,
            evaluation_episodes=args.evaluation_episodes,
        )
    else:
        result = summarize_study(args.output)
    print(json.dumps(_json_value(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
