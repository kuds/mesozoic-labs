"""Prepare and run the opt-in Compsognathus filter × stance-reward study.

Each arm/seed is an independent fresh PPO run. Saved artifacts have explicit
study identities and cannot replace registered production checkpoints.
Short probes retain the complete learning-rate and entropy schedules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
import uuid
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
    verify_study_checkpoint,
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
    plan: dict = _json_value(
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
                "Continuation uses verified study pairs with the original schedule; simulator episodes restart. Historical policies are references only.",
            ],
        }
    )
    return plan


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
    plan: dict = json.loads((output / "study_plan.json").read_text())
    if plan.get("schema") != SCHEMA:
        raise ValueError("Unsupported study plan schema")
    current = make_study_plan(training_seeds=tuple(plan["training_seeds"]))
    if plan != current:
        raise ValueError("Prepared study plan no longer matches source/config; prepare a new study")
    return plan


class BalanceStudyPPO(PPO):
    """Stop probes after complete updates, keeping the full schedule horizon."""

    def __init__(self, *args, stop_after_steps: int = 0, after_update=None, **kwargs):
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


def _read_json(path: Path) -> dict:
    value: dict = json.loads(path.read_text())
    return value


def _panel_binding(checkpoint: Path, seeds: list[int], plan: dict) -> dict:
    return {
        "checkpoint_manifest_sha256": hashlib.sha256(checkpoint.with_suffix(".manifest.json").read_bytes()).hexdigest(),
        "seeds": seeds,
        "horizon": plan["stage_config"]["env_kwargs"]["max_episode_steps"],
        "settle_steps": plan["stage_config"]["curriculum_kwargs"]["settle_steps"],
        "targets": plan["behavior_targets"],
    }


def _validate_panel(panel: dict, binding: dict, plan: dict) -> None:
    if panel.get("study_panel") != binding or [row["seed"] for row in panel["episodes"]] != binding["seeds"]:
        raise ValueError("Saved evaluation panel does not match this checkpoint and seed panel")
    summary = summarize_balance_panel(
        panel["episodes"],
        horizon=binding["horizon"],
        stance_thresholds=StanceGateThresholds.from_curriculum(plan["stage_config"]["curriculum_kwargs"]),
        targets=BalanceTargets(**binding["targets"]),
    )
    if _json_value(summary) != panel["summary"]:
        raise ValueError("Saved evaluation summary differs from its episode evidence")


def train_balance_arm(
    output: Path,
    arm: str,
    seed: int,
    *,
    probe_updates: int | None = None,
    evaluation_episodes: int | None = None,
    resume: bool = False,
) -> dict:
    """Train or continue one arm/seed; full runs confirm on 40 fresh seeds.

    Continuation retains weights, optimizer, normalizer and absolute schedules.
    The simulator and random streams reset to the recorded worker seeds, so it
    is not a bitwise continuation of the interrupted trajectory. Committed
    checkpoints are immutable; incomplete saves are preserved in quarantine.
    """
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
    n_eval = evaluation_episodes or 40
    snapshot = {"plan": plan, "arm": arm, "seed": seed, "probe_updates": probe_updates, "evaluation_episodes": n_eval}
    tag = f"{arm}_seed{seed}" + (f"_probe{probe_updates}" if probe_updates is not None else "")
    run_dir = output / "runs" / tag
    if run_dir.exists() and not resume:
        raise FileExistsError("Run already exists; use resume=True to validate and continue it")
    bare = make_balance_env(arm, **config["env_kwargs"])
    try:
        identity = build_study_identity(bare, snapshot)
    finally:
        bare.close()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    identity_path = run_dir / "study_identity.json"
    if config_path.exists():
        if _read_json(config_path) != snapshot:
            raise ValueError("Saved run configuration differs from this requested run")
    else:
        if any(run_dir.iterdir()):
            raise ValueError("Existing run has no configuration; preserve it and choose a new study")
        _save(config_path, snapshot)
    if identity_path.exists():
        validate_study_identity(_read_json(identity_path), identity)
    else:
        _save(identity_path, identity)
    screen_seeds = plan["probe_screen_seeds" if probe_updates is not None else "screen_seeds"][:n_eval]
    confirmation_seeds = plan["probe_confirmation_seeds" if probe_updates is not None else "confirmation_seeds"][
        :n_eval
    ]
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    # A manifest commits a pair. Preserve orphaned writes before reusing a step
    # number; a corrupt committed pair fails closed instead of silently falling back.
    for artifact in list(checkpoint_dir.glob("step_*.*")):
        if not artifact.exists():
            continue
        prefix = checkpoint_dir / artifact.name.split(".")[0]
        if not prefix.with_suffix(".manifest.json").exists():
            quarantine = run_dir / "incomplete_checkpoints" / str(uuid.uuid4())
            quarantine.mkdir(parents=True)
            for partial in checkpoint_dir.glob(prefix.name + ".*"):
                partial.rename(quarantine / partial.name)
    checkpoints = sorted(
        (
            path.with_name(path.name.removesuffix(".manifest.json"))
            for path in checkpoint_dir.glob("step_*.manifest.json")
        ),
        key=lambda path: int(path.name.removeprefix("step_")),
    )
    for checkpoint in checkpoints:
        verify_study_checkpoint(checkpoint, identity)
        step = int(checkpoint.name.removeprefix("step_"))
        if step <= 0 or step > stop or step % rollout:
            raise ValueError("Saved checkpoint is outside this run's completed PPO updates")
    latest = checkpoints[-1] if checkpoints else None
    from_steps = int(latest.name.removeprefix("step_")) if latest else 0
    records: list[dict] = []
    best_key: tuple | None = None
    selected: Path | None = None

    def remember(checkpoint: Path, panel: dict) -> None:
        nonlocal selected, best_key
        key = tuple(panel["summary"]["selection_key"])
        if best_key is None or key > best_key:
            best_key, selected = key, checkpoint
        records.append(
            {"timesteps": int(checkpoint.name.removeprefix("step_")), "checkpoint": checkpoint.name, **panel["summary"]}
        )

    def saved_panel(checkpoint: Path, seeds: list[int], path: Path, *, trace_path: Path | None = None) -> dict:
        binding = _panel_binding(checkpoint, seeds, plan)
        if path.exists():
            panel = _read_json(path)
            _validate_panel(panel, binding, plan)
            return panel
        vector = _vector_env(arm, config["env_kwargs"], 1, seeds[0])
        frozen = None
        try:
            policy, frozen = load_study_checkpoint(checkpoint, identity, vector)
            panel = _evaluate(policy, frozen, arm, plan, seeds, trace_path=trace_path)
        finally:
            (frozen if frozen is not None else vector).close()
        panel["study_panel"] = binding
        _save(path, panel)
        return panel

    # Completed jobs are checked without new training or evaluation. Missing
    # evidence in a declared-complete bundle is an error, never a silent rerun.
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        result = _read_json(summary_path)
        for checkpoint in checkpoints:
            panel = _read_json(checkpoint.with_suffix(".screen.json"))
            _validate_panel(panel, _panel_binding(checkpoint, screen_seeds, plan), plan)
            remember(checkpoint, panel)
        if selected is None or from_steps != stop:
            raise ValueError("Completed run has no complete training/checkpoint history")
        confirmation = _read_json(run_dir / "confirmation.json")
        _validate_panel(confirmation, _panel_binding(selected, confirmation_seeds, plan), plan)
        if (
            result.get("schema") != SCHEMA
            or result.get("arm") != arm
            or result.get("seed") != seed
            or result.get("training_steps") != stop
            or result.get("diagnostic_probe") != (probe_updates is not None)
            or result.get("selected_checkpoint") != str(selected.relative_to(run_dir))
            or result.get("confirmation") != confirmation["summary"]
            or result.get("learned_balance_qualified")
            != (probe_updates is None and confirmation["summary"]["behavior_qualified"])
            or result.get("research_only") is not True
            or result.get("production_advancement") is not False
        ):
            raise ValueError("Completed result does not match its verified training and confirmation evidence")
        return result

    history_path = run_dir / "resume_history.json"
    history: list[dict] = json.loads(history_path.read_text()) if history_path.exists() else []
    for previous in history:
        if previous["status"] == "started":
            previous["status"] = "disconnected"
    worker_seeds = plan["training_environment_seeds"][str(seed)]
    segment: dict = {
        "segment_id": len(history),
        "from_checkpoint": str(latest.relative_to(run_dir)) if latest else None,
        "from_steps": from_steps,
        "worker_seeds": worker_seeds,
        "simulator_reset": True,
        "status": "started",
        "to_steps": from_steps,
        "elapsed_seconds": 0.0,
    }
    history.append(segment)
    _save(history_path, history)
    worker_seed = worker_seeds[0]
    model_kwargs, _, _ = _prepare_alg_kwargs(config, "ppo", 0, run_dir, False)
    next_screen = (from_steps // plan["screen_every_steps"] + 1) * plan["screen_every_steps"]
    started = time.perf_counter()
    model = None
    normalizer = None
    vector = None

    def after_update(policy):
        nonlocal next_screen
        if policy.num_timesteps < next_screen and policy.num_timesteps < stop:
            return
        checkpoint = checkpoint_dir / f"step_{policy.num_timesteps}"
        save_study_checkpoint(policy, normalizer, checkpoint, identity)
        _save(checkpoint.with_suffix(".updates.json"), policy.update_records)
        panel = _evaluate(policy, normalizer, arm, plan, screen_seeds)
        panel["study_panel"] = _panel_binding(checkpoint, screen_seeds, plan)
        _save(checkpoint.with_suffix(".screen.json"), panel)
        remember(checkpoint, panel)
        _save(run_dir / "screening_history.json", records)
        _save(run_dir / "updates.json", policy.update_records)
        segment.update(to_steps=policy.num_timesteps, elapsed_seconds=time.perf_counter() - started)
        _save(history_path, history)
        next_screen = (policy.num_timesteps // plan["screen_every_steps"] + 1) * plan["screen_every_steps"]
        print(
            json.dumps(
                _json_value({"arm": arm, "seed": seed, "steps": policy.num_timesteps, "screen": panel["summary"]})
            ),
            flush=True,
        )

    try:
        for checkpoint in checkpoints:
            panel = saved_panel(checkpoint, screen_seeds, checkpoint.with_suffix(".screen.json"))
            remember(checkpoint, panel)
        _save(run_dir / "screening_history.json", records)
        vector = _vector_env(arm, config["env_kwargs"], plan["n_envs"], worker_seed)
        if latest is not None:
            model, normalizer = load_study_checkpoint(latest, identity, vector, model_class=BalanceStudyPPO)
            if model.num_timesteps != from_steps:
                raise ValueError("Loaded model step count differs from its checkpoint name")
            normalizer.training = plan["normalization"]["norm_reward"] or plan["normalization"]["norm_obs"]
            normalizer.norm_reward = plan["normalization"]["norm_reward"]
            # Callback initialization captures ent_coef. Restore the ORIGINAL
            # start coefficient so absolute-step decay is not applied twice.
            model.ent_coef = config["ppo_kwargs"]["ent_coef"]
            update_path = latest.with_suffix(".updates.json")
            model.update_records = json.loads(update_path.read_text()) if update_path.exists() else []
            model.after_update = after_update
            model.stop_after_steps = stop
        else:
            normalizer = VecNormalize(vector, **plan["normalization"])
            attach_study_identity(normalizer, identity)
            model = BalanceStudyPPO(
                "MlpPolicy",
                normalizer,
                seed=seed,
                device="cpu",
                stop_after_steps=stop,
                after_update=after_update,
                **model_kwargs,
            )
            attach_study_identity(model, identity)
        normalizer.seed(worker_seed)
        model.set_logger(configure(str(run_dir / "training_logs" / f"segment_{segment['segment_id']:03d}"), ["csv"]))
        print(
            json.dumps(
                {
                    "event": "resume" if latest else "start",
                    "arm": arm,
                    "seed": seed,
                    "from_steps": from_steps,
                    "planned_steps": stop,
                    "full_schedule_steps": budget,
                }
            ),
            flush=True,
        )
        if model.num_timesteps < stop:
            model.learn(
                total_timesteps=budget - model.num_timesteps,
                reset_num_timesteps=False,
                callback=_maybe_ent_coef_decay_callback(config, "ppo", budget),
            )
        if selected is None or model.num_timesteps != stop:
            raise RuntimeError("Training did not produce the complete planned screening history")
        confirmation = saved_panel(
            selected, confirmation_seeds, run_dir / "confirmation.json", trace_path=run_dir / "selected_trace.csv"
        )
        segment.update(status="completed", to_steps=model.num_timesteps, elapsed_seconds=time.perf_counter() - started)
        _save(history_path, history)
        _save(run_dir / "updates.json", model.update_records)
        result = {
            "schema": SCHEMA,
            "arm": arm,
            "seed": seed,
            "training_steps": model.num_timesteps,
            "training_seconds": sum(item["elapsed_seconds"] for item in history),
            "diagnostic_probe": probe_updates is not None,
            "selected_checkpoint": str(selected.relative_to(run_dir)),
            "confirmation": confirmation["summary"],
            "learned_balance_qualified": probe_updates is None and confirmation["summary"]["behavior_qualified"],
            "research_only": True,
            "production_advancement": False,
        }
        _save(summary_path, result)
        return result
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "arm": arm, "seed": seed}
        segment.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            to_steps=model.num_timesteps if model is not None else from_steps,
            elapsed_seconds=time.perf_counter() - started,
            error=error,
        )
        _save(history_path, history)
        _save(run_dir / "run_failure.json", error)
        raise
    finally:
        if model is not None and hasattr(model, "_logger"):
            model.logger.close()
        if normalizer is not None:
            normalizer.close()
        elif vector is not None:
            vector.close()


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
                snapshot = {"plan": plan, "arm": arm, "seed": seed, "probe_updates": None, "evaluation_episodes": 40}
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
                # A summary file alone cannot certify completion. Reuse the
                # trainer's read-only completed path to verify paired hashes,
                # checkpoint selection and both panels' episode evidence.
                result = train_balance_arm(output, arm, seed, resume=True)
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
    parser.add_argument("--resume", action="store_true")
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
            resume=args.resume,
        )
    else:
        result = summarize_study(args.output)
    print(json.dumps(_json_value(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
