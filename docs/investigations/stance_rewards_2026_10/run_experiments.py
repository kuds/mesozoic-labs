"""Calibrate and launch paired, non-certifying Stance reward pilots.

Run from the repository root; see README.md. Training is fresh-seed PPO,
not a continuation of the selected checkpoints. Those checkpoints are used
only for pricing calibration. All physical gate thresholds remain fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from prototype import VARIANTS, BudgetSchedule, ExperimentalReward, reward_deltas
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.shared.config import load_stage_config
from environments.shared.constants import DEFAULT_CLIP_OBS, DEFAULT_CLIP_REWARD
from environments.shared.curriculum.schedules import CosineSchedule, EntCoefDecayCallback, LinearSchedule
from environments.shared.curriculum.stance_gate_v2 import StanceV2Thresholds, evaluate_stance_v2_gate, statue_reference
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import SubstepContactRecorder
from environments.shared.gait.stance_metrics import episode_stance_metrics
from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
from environments.shared.reporting.stance_report import _load_policy
from environments.shared.species_names import species_display_name
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()
FACTORIES = {"trex": TRexEnv, "velociraptor": RaptorEnv}
PRODUCTION_BUDGET = {"trex": 11_000_000, "velociraptor": 6_000_000}
SEEDS = [42, 48, 50]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": {p.name: digest(p) for p in SCRIPT.parent.glob("*.py")},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "mujoco", "gymnasium", "stable-baselines3", "torch"]
        },
        "python": platform.python_version(),
        "device": "cpu",
        "status": "research only; no curriculum advancement or certification",
    }


def make_env(species, variant="control"):
    cfg = load_stage_config(species, "stance")
    settle = cfg["curriculum_kwargs"]["settle_steps"]
    return ExperimentalReward(FACTORIES[species](**cfg["env_kwargs"]), VARIANTS[species, variant], settle_steps=settle)


def measure_panel(species, predict, *, episodes, seed, variant="control", keep_traces=None):
    """Floor-truth metrics use BASE rewards; all candidate rewards price the same trajectory.

    The original task's gate is diagnostic here. A changed reward has a new
    task and would require its own frozen calibration before certification.
    """
    env = make_env(species, variant)
    rec = SubstepContactRecorder(env, Morphology.from_env(env, species)).attach()
    variants = [v for (s, _), v in VARIANTS.items() if s == species]
    metrics, rows = [], []
    traces = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            reset = getattr(predict, "reset", None)
            if callable(reset):
                reset(seed + ep)
            totals = {v.name: 0.0 for v in variants}
            term_totals = {v.name: Counter() for v in variants}
            nonterminal = {v.name: [] for v in variants}
            while True:
                action = np.asarray(predict(obs), dtype=np.float64).reshape(env.action_space.shape)
                obs, _, terminated, truncated, info = env.step(action)
                base_reward = info["experiment_base_reward"]
                rec.end_step(action, base_reward)
                floor = {
                    "cop": info["experiment_cop"],
                    "sole_loaded": info["experiment_sole_loaded"],
                    "airborne_fraction": info["experiment_airborne_fraction"],
                    "peak_floor_force_bw": info["experiment_peak_floor_force_bw"],
                    "support_coverage": info["experiment_support_coverage"],
                }
                for v in variants:
                    delta = reward_deltas(v, floor, info, env.unwrapped, env.steps - 1, env.settle_steps)
                    reward = base_reward + sum(delta.values())
                    totals[v.name] += reward
                    term_totals[v.name].update(delta)
                    if not terminated:
                        nonterminal[v.name].append(reward)
                if terminated or truncated:
                    trace = rec.end_episode(
                        terminated=terminated,
                        truncated=truncated,
                        termination_reason=info.get("termination_reason", ""),
                    )
                    metric = episode_stance_metrics(trace, settle_steps=env.settle_steps)
                    metrics.append(metric)
                    rows.append(
                        {
                            "seed": seed + ep,
                            "length": trace.length,
                            "rewards_by_variant": totals,
                            "reward_deltas": {k: dict(v) for k, v in term_totals.items()},
                            "mean_nonterminal_reward": {
                                k: float(np.mean(v)) if v else None for k, v in nonterminal.items()
                            },
                            "metrics_original_task": asdict(metric),
                        }
                    )
                    if keep_traces:
                        traces.append(
                            {
                                "cop": trace.sole_cop[:, :, 0],
                                "sole_loaded": trace.sole_contacts_mean > 0,
                                "peak_force_bw": trace.total_floor_max / trace.body_weight_n,
                                "airborne_fraction": trace.feet_airborne_substeps / trace.frame_skip,
                                "support_loaded_frac": trace.support_loaded_frac,
                                "base_reward": trace.reward,
                            }
                        )
                    break
            print(
                json.dumps(
                    clean(
                        {
                            "event": "evaluation_episode",
                            "species": species_display_name(species),
                            "variant": variant,
                            "seed": seed + ep,
                            "length": rows[-1]["length"],
                        }
                    )
                ),
                flush=True,
            )
    finally:
        rec.detach()
        env.close()
    if keep_traces:
        arrays = {f"episode_{i:02d}_{key}": value for i, trace in enumerate(traces) for key, value in trace.items()}
        Path(keep_traces).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(keep_traces, **arrays)
    return metrics, rows


def panel_summary(rows):
    return {
        "episodes": len(rows),
        "full_horizon": sum(r["length"] == 1000 for r in rows),
        "mean_rewards_by_variant": {
            k: float(np.mean([r["rewards_by_variant"][k] for r in rows])) for k in rows[0]["rewards_by_variant"]
        },
        "mean_nonterminal_reward": {
            k: float(
                np.nanmean(
                    [
                        r["mean_nonterminal_reward"][k] if r["mean_nonterminal_reward"][k] is not None else np.nan
                        for r in rows
                    ]
                )
            )
            for k in rows[0]["rewards_by_variant"]
        },
    }


class ExplorationProxy:
    """Raw N(0, exp(log_std_init)) actions; a pricing stress test, not an actual PPO policy."""

    def __init__(self, shape, std):
        self.shape, self.std = shape, std

    def reset(self, seed):
        self.rng = np.random.default_rng(seed + 910_000)

    def __call__(self, obs):
        return np.clip(self.rng.normal(0, self.std, self.shape), -1, 1)


def calibrate(args):
    species = args.species
    cfg = load_stage_config(species, "stance")
    factory = partial(FACTORIES[species], **cfg["env_kwargs"])
    test_env = factory()
    shape = test_env.action_space.shape
    control_dt = test_env.dt
    test_env.close()
    controllers = {"statue": lambda obs: np.zeros(shape)}
    input_hashes = {}
    labels = [42, 48] if species == "trex" else [42]
    for seed in labels:
        prefix = args.inputs / f"{species}_seed{seed}"
        model, stats = prefix.with_name(prefix.name + "_model.zip"), prefix.with_name(prefix.name + "_vecnorm.pkl")
        if not model.exists() or not stats.exists():
            raise FileNotFoundError(f"Missing paired calibration input {model} / {stats}")
        predict, _ = _load_policy(str(model), str(stats), factory, plant_identity=current_plant_identity(species))
        controllers[f"selected_seed{seed}"] = predict
        input_hashes[f"selected_seed{seed}"] = {"model_sha256": digest(model), "vecnorm_sha256": digest(stats)}
    std = np.exp(cfg["ppo_kwargs"]["policy_kwargs"].get("log_std_init", 0))
    controllers["initial_exploration_proxy"] = ExplorationProxy(shape, std)
    panels = {}
    statue = None
    for label, predict in controllers.items():
        metrics, rows = measure_panel(
            species,
            predict,
            episodes=args.episodes,
            seed=3042,
            keep_traces=args.output / species / f"calibration_{label}_traces.npz",
        )
        if label == "statue":
            statue = statue_reference(metrics, horizon=1000)
            save(
                args.output / species / "statue.json",
                {"rows": rows, "metrics": [asdict(m) for m in metrics], "provenance": provenance()},
            )
        thresholds = StanceV2Thresholds.from_curriculum(cfg["curriculum_kwargs"])
        gate = evaluate_stance_v2_gate(metrics, thresholds, horizon=1000, statue=statue, control_dt=control_dt)
        panels[label] = {
            "summary": panel_summary(rows),
            "original_task_gate_diagnostic": gate.as_dict(),
            "episodes": rows,
        }
        save(
            args.output / f"calibration_{species}.json",
            {
                "species": species_display_name(species),
                "provenance": provenance(),
                "input_hashes": input_hashes,
                "variants": [v.as_dict() for (s, _), v in VARIANTS.items() if s == species],
                "panels": panels,
            },
        )
        print(
            json.dumps(clean({"event": "calibration_panel", "controller": label, **panels[label]["summary"]})),
            flush=True,
        )


class Progress(BaseCallback):
    def __init__(self, output, metadata, vec):
        super().__init__()
        self.output, self.metadata, self.vec = output, metadata, vec
        self.start = time.monotonic()
        self.last_save = -1

    def snapshot(self, status="running"):
        self.model.save(self.output / "latest_model.zip")
        self.vec.save(self.output / "latest_vecnorm.pkl")
        save(
            self.output / "progress.json",
            {
                **self.metadata,
                "pid": os.getpid(),
                "status": status,
                "steps": self.model.num_timesteps,
                "elapsed_seconds": time.monotonic() - self.start,
            },
        )

    def _on_step(self):
        if self.model.num_timesteps // 32768 > self.last_save:
            self.last_save = self.model.num_timesteps // 32768
            self.snapshot()
            print(
                json.dumps(
                    {
                        "event": "training_progress",
                        "species": self.metadata["species"],
                        "variant": self.metadata["variant"],
                        "seed": self.metadata["training_seed"],
                        "steps": self.model.num_timesteps,
                    }
                ),
                flush=True,
            )
        return True


def train(args):
    species = args.species
    cfg = load_stage_config(species, "stance")
    metadata = {
        "species": species_display_name(species),
        "species_id": species,
        "behavior": "Stance",
        "variant": args.variant,
        "variant_parameters": VARIANTS[species, args.variant].as_dict(),
        "training_seed": args.seed,
        "requested_steps": args.timesteps,
        "n_envs": 4,
        "production_schedule_budget": PRODUCTION_BUDGET[species],
        "ppo_config": cfg["ppo_kwargs"],
        "base_env_config": cfg["env_kwargs"],
        "plant_identity": current_plant_identity(species).to_dict(),
        "provenance": provenance(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "experiment.json", metadata)
    if (args.output / "latest_model.zip").exists():
        raise RuntimeError("output already has a checkpoint; choose a fresh directory")
    vec = VecNormalize(
        DummyVecEnv([partial(lambda s, v: Monitor(make_env(s, v)), species, args.variant) for _ in range(4)]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=DEFAULT_CLIP_OBS,
        clip_reward=DEFAULT_CLIP_REWARD,
        gamma=cfg["ppo_kwargs"]["gamma"],
    )
    ppo = dict(cfg["ppo_kwargs"])
    end_entropy = ppo.pop("ent_coef_end", None)
    entropy_steps = ppo.pop("ent_coef_decay_timesteps", PRODUCTION_BUDGET[species])
    end_lr = ppo.pop("learning_rate_end")
    schedule_kind = ppo.pop("lr_schedule", "linear")
    base_schedule = (CosineSchedule if schedule_kind == "cosine" else LinearSchedule)(ppo["learning_rate"], end_lr)
    ppo["learning_rate"] = BudgetSchedule(base_schedule, args.timesteps, PRODUCTION_BUDGET[species])
    model = PPO("MlpPolicy", vec, seed=args.seed, device="cpu", **ppo)
    attach_plant_identity(model, current_plant_identity(species))
    attach_plant_identity(vec, current_plant_identity(species))
    metadata["initial_policy_sha256"] = hashlib.sha256(
        b"".join(v.detach().cpu().numpy().tobytes() for v in model.policy.state_dict().values())
    ).hexdigest()
    save(args.output / "experiment.json", metadata)
    initial = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
    progress = Progress(args.output, metadata, vec)
    callbacks = [progress]
    if end_entropy is not None:
        callbacks.insert(0, EntCoefDecayCallback(end_value=end_entropy, decay_timesteps=entropy_steps))
    try:
        model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks))
        progress.snapshot("evaluating")
        vec.training = False
        vec.norm_reward = False

        def predict(obs):
            return model.predict(vec.normalize_obs(obs), deterministic=True)[0]

        # Cold held-out seeds: not the published/calibration 3042 panel.
        metrics, rows = measure_panel(species, predict, episodes=args.episodes, seed=23042, variant=args.variant)
        statue_rows = json.loads((args.calibration / species / "statue.json").read_text())["rows"]
        # Reward baseline is the unchanged original task's paired statue;
        # the final held-out panel gets its own reset seeds.
        statue_metrics, held_statue = measure_panel(
            species, lambda obs: np.zeros(model.action_space.shape), episodes=args.episodes, seed=23042
        )
        statue = statue_reference(statue_metrics, horizon=1000)
        thresholds = StanceV2Thresholds.from_curriculum(cfg["curriculum_kwargs"])
        gate = evaluate_stance_v2_gate(
            metrics, thresholds, horizon=1000, statue=statue, control_dt=vec.envs[0].unwrapped.dt
        )
        results = {
            **metadata,
            "steps": model.num_timesteps,
            "changed_parameter_tensors": sum(
                not torch.equal(initial[k], v.detach().cpu()) for k, v in model.policy.state_dict().items()
            ),
            "evaluation_seed_start": 23042,
            "evaluation_episodes": args.episodes,
            "summary": panel_summary(rows),
            "original_task_gate_diagnostic": gate.as_dict(),
            "episodes": rows,
            "held_out_statue_summary": panel_summary(held_statue),
            "calibration_statue_episodes": len(statue_rows),
            "model_sha256": digest(args.output / "latest_model.zip"),
            "vecnorm_sha256": digest(args.output / "latest_vecnorm.pkl"),
        }
        save(args.output / "results.json", results)
        progress.snapshot("completed")
        print(
            json.dumps(
                clean(
                    {
                        "event": "pilot_completed",
                        "species": metadata["species"],
                        "variant": args.variant,
                        "training_seed": args.seed,
                        "steps": model.num_timesteps,
                        "summary": results["summary"],
                        "original_task_gate_diagnostic": gate.as_dict(),
                    }
                )
            ),
            flush=True,
        )
    except BaseException as exc:
        progress.snapshot("failed")
        save(args.output / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        vec.close()


def batch(args):
    for species in FACTORIES:
        calibration = json.loads((args.calibration / f"calibration_{species}.json").read_text())
        if len(calibration["panels"]) != (4 if species == "trex" else 3):
            raise RuntimeError(f"Incomplete calibration for {species}")
        if calibration["provenance"]["source_sha256"] != provenance()["source_sha256"]:
            raise RuntimeError("Source changed since calibration; re-run pricing before launching")
    jobs = []
    for seed in SEEDS:
        for species, variant in VARIANTS:
            directory = args.output / f"{species}_{variant}_seed{seed}"
            jobs.append(
                {"species": species, "variant": variant, "seed": seed, "output": str(directory), "status": "queued"}
            )
    manifest = {
        "provenance": provenance(),
        "steps_per_run": args.timesteps,
        "seeds": SEEDS,
        "workers": args.workers,
        "jobs": jobs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "manifest.json", manifest)
    active = []
    while any(job["status"] == "queued" for job in jobs) or active:
        for process, stream, job in active[:]:
            code = process.poll()
            if code is not None:
                stream.close()
                job["status"] = "completed" if code == 0 else "failed"
                job["exit_code"] = code
                active.remove((process, stream, job))
                save(args.output / "manifest.json", manifest)
        for job in [j for j in jobs if j["status"] == "queued"][: max(0, args.workers - len(active))]:
            directory = Path(job["output"])
            directory.mkdir(parents=True, exist_ok=True)
            stream = (directory / "training.log").open("w")
            command = [
                sys.executable,
                str(SCRIPT),
                "train",
                "--species",
                job["species"],
                "--variant",
                job["variant"],
                "--seed",
                str(job["seed"]),
                "--timesteps",
                str(args.timesteps),
                "--output",
                str(directory),
                "--calibration",
                str(args.calibration),
            ]
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            job.update(status="running", pid=process.pid, command=command)
            active.append((process, stream, job))
            save(args.output / "manifest.json", manifest)
            print(
                json.dumps(
                    {
                        "event": "pilot_started",
                        "species": species_display_name(job["species"]),
                        "variant": job["variant"],
                        "seed": job["seed"],
                        "pid": process.pid,
                    }
                ),
                flush=True,
            )
        time.sleep(2)
    print(
        json.dumps(
            {
                "event": "batch_completed",
                "completed": sum(j["status"] == "completed" for j in jobs),
                "failed": sum(j["status"] == "failed" for j in jobs),
            }
        ),
        flush=True,
    )
    if any(j["status"] == "failed" for j in jobs):
        raise RuntimeError("one or more pilots failed; inspect per-run error.json / training.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate", "train", "batch"])
    parser.add_argument("--species", choices=list(FACTORIES))
    parser.add_argument("--variant", default="control")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--timesteps", type=int, default=1_048_576)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode != "batch" and not args.species:
        parser.error("--species is required")
    if args.mode == "calibrate" and not args.inputs:
        parser.error("calibration requires --inputs")
    if args.mode in ("train", "batch") and not args.calibration:
        parser.error("training requires --calibration")
    if args.mode == "train" and (args.species, args.variant) not in VARIANTS:
        parser.error("variant is not defined for this species")
    if args.timesteps < 8192 or args.timesteps % 8192:
        parser.error("--timesteps must be a positive multiple of the 8192-step PPO rollout")
    if args.workers < 1:
        parser.error("--workers must be positive")
    torch.set_num_threads(1)
    {"calibrate": calibrate, "train": train, "batch": batch}[args.mode](args)


if __name__ == "__main__":
    main()
