#!/usr/bin/env python3
"""Hyperparameter sweep tool for Mesozoic Labs.

Three modes:

  **launch**      — Submit a Vertex AI Hyperparameter Tuning job for **one stage**.
                    Vertex AI uses Bayesian optimisation to run N parallel trials
                    and automatically find the best hyperparameter combination.
                    Returns immediately (non-blocking).
                    Requires: ``pip install google-cloud-aiplatform``

  **launch-all**  — Submit Stage 1, Stage 2, and Stage 3 HPT jobs sequentially in a
                    **single command**.  Each stage waits for the previous to finish,
                    automatically picks the best trial's checkpoint, and passes it as
                    the warm-start model for the next stage.

  **trial**       — Entry point used by each Vertex AI HPT trial worker running
                    inside Docker. Receives hyperparameter values injected by the
                    HPT service as ``--ppo_learning_rate X --ppo_ent_coef Y``
                    style args, converts them to ``--override`` dot notation, and
                    runs the training script.  Output is written to a per-trial
                    subdirectory (using the ``CLOUD_ML_TRIAL_ID`` env var) so that
                    each worker's checkpoint is preserved separately.

Usage examples::

    # ── single command: sweep all three stages end-to-end ───────────────────
    python environments/shared/scripts/sweep.py launch-all \\
        --species velociraptor --algorithm ppo \\
        --project YOUR_GCP_PROJECT --bucket YOUR_GCS_BUCKET \\
        --image us-central1-docker.pkg.dev/YOUR_PROJECT/mesozoic-labs/trainer:latest \\
        --trials 20 --parallel 5 \\
        --timesteps-stage1 500000 --timesteps-stage2 1000000 --timesteps-stage3 1500000

    # ── launch a single-stage PPO sweep (from your local machine) ───────────
    python environments/shared/scripts/sweep.py launch \\
        --species velociraptor --stage 1 --algorithm ppo \\
        --project YOUR_GCP_PROJECT --bucket YOUR_GCS_BUCKET \\
        --image us-central1-docker.pkg.dev/YOUR_PROJECT/mesozoic-labs/trainer:latest \\
        --trials 20 --parallel 5 --timesteps 500000

    # ── run a single trial locally (test before launching) ──────────────────
    python environments/shared/scripts/sweep.py trial \\
        --species velociraptor --stage 1 --algorithm ppo \\
        --timesteps 10000 --n-envs 1 \\
        --ppo_learning_rate 1e-4 --ppo_ent_coef 0.02 --ppo_batch_size 128

    # ── customise the search space with JSON ────────────────────────────────
    python environments/shared/scripts/sweep.py launch \\
        --species trex --stage 1 --algorithm ppo \\
        --project YOUR_PROJECT --bucket YOUR_BUCKET --image IMAGE_URI \\
        --trials 30 --parallel 5 \\
        --search-space '{
            "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
            "ppo_ent_coef":      {"type": "double", "min": 0.001, "max": 0.05, "scale": "log"},
            "ppo_batch_size":    {"type": "discrete", "values": [64, 128, 256]}
        }'

Metric optimised: ``best_mean_reward`` (maximised).
This is reported by each trial's ``train()`` call via ``cloudml-hypertune``.

Search space parameter naming convention:
  ``{algo}_{param}``  e.g. ``ppo_learning_rate``, ``sac_batch_size``
  ``env_{param}``     e.g. ``env_alive_bonus``, ``env_forward_vel_weight``

These are auto-converted to ``--override`` dot notation inside ``trial`` mode:
  ``ppo_learning_rate`` → ``ppo.learning_rate=<value>``
  ``env_alive_bonus``   → ``env.alive_bonus=<value>``
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add repo root to Python path so environments.* imports work
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Default search spaces ────────────────────────────────────────────────────
# Each entry: parameter_id -> {"type": ..., ...}
# parameter_id uses underscore notation to match Vertex AI HPT arg injection.
# The trial entry point converts them to dot notation for --override.

_DEFAULT_PPO_SEARCH_SPACE = {
    "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef": {"type": "double", "min": 1e-4, "max": 0.05, "scale": "log"},
    "ppo_batch_size": {"type": "discrete", "values": [64, 128, 256, 512]},
    "ppo_gamma": {"type": "double", "min": 0.97, "max": 0.999, "scale": "linear"},
    "ppo_n_steps": {"type": "discrete", "values": [1024, 2048, 4096]},
}

_DEFAULT_SAC_SEARCH_SPACE = {
    "sac_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "sac_batch_size": {"type": "discrete", "values": [128, 256, 512]},
    "sac_gamma": {"type": "double", "min": 0.97, "max": 0.999, "scale": "linear"},
}

_DEFAULT_SEARCH_SPACES = {
    "ppo": _DEFAULT_PPO_SEARCH_SPACE,
    "sac": _DEFAULT_SAC_SEARCH_SPACE,
}


def _hpt_arg_to_override(key: str, value: str) -> str:
    """Convert a Vertex AI HPT arg name to ``--override`` dot notation.

    Examples::

        "ppo_learning_rate", "0.0003" → "ppo.learning_rate=0.0003"
        "env_alive_bonus",   "2.0"    → "env.alive_bonus=2.0"
    """
    for prefix in ("ppo", "sac", "env"):
        if key.startswith(prefix + "_"):
            param = key[len(prefix) + 1 :]
            return f"{prefix}.{param}={value}"
    # Unrecognised prefix — pass through as-is (best-effort)
    return f"{key}={value}"


def run_trial(args: argparse.Namespace, extra_args: list[str]) -> None:
    """Run a single training trial.

    ``extra_args`` contains the hyperparameter values injected by Vertex AI
    HPT (e.g. ``['--ppo_learning_rate', '0.0003', '--ppo_ent_coef', '0.01']``).
    They are converted to ``--override`` format before calling ``train()``.

    Each Vertex AI worker gets a unique ``CLOUD_ML_TRIAL_ID`` environment
    variable.  When ``--output-dir`` is set, this ID is appended as a
    subdirectory so that every trial's checkpoint is written to its own
    path — which is how ``launch-all`` identifies the best trial later.
    """
    # Convert HPT-style args (--ppo_learning_rate 0.0003) to override format
    overrides: list[str] = []
    i = 0
    while i < len(extra_args):
        token = extra_args[i]
        if token.startswith("--"):
            key = token[2:]  # strip leading "--"
            # Peek at the next token — is it a value or another flag?
            if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                value = extra_args[i + 1]
                i += 2
            else:
                # Boolean flag — skip (shouldn't appear for HPT numeric params)
                i += 1
                continue
            overrides.append(_hpt_arg_to_override(key, value))
        else:
            i += 1

    if overrides:
        logger.info("Trial overrides from HPT: %s", overrides)
    else:
        logger.info("No HPT overrides received — using TOML defaults")

    # Make output unique per trial so each worker keeps its own checkpoint.
    # Vertex AI sets CLOUD_ML_TRIAL_ID on every worker container.
    trial_id = os.environ.get("CLOUD_ML_TRIAL_ID", "local")
    output_dir = f"{args.output_dir}/{trial_id}" if args.output_dir else None
    if output_dir:
        logger.info("Trial output directory: %s", output_dir)

    # Import the right species' training module
    if args.species == "velociraptor":
        from environments.velociraptor.scripts.train_sb3 import (
            STAGE_CONFIGS,
            _apply_overrides,
            train,
        )
    elif args.species == "brachiosaurus":
        from environments.brachiosaurus.scripts.train_sb3 import (
            STAGE_CONFIGS,
            _apply_overrides,
            train,
        )
    elif args.species == "trex":
        from environments.trex.scripts.train_sb3 import (
            STAGE_CONFIGS,
            _apply_overrides,
            train,
        )
    else:
        logger.error("Unknown species: %s", args.species)
        sys.exit(1)

    if overrides:
        _apply_overrides(STAGE_CONFIGS, overrides)

    train(
        stage=args.stage,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        load_path=args.load,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_subproc=False,
        verbose=1,
        algorithm=args.algorithm,
        use_wandb=args.wandb,
        output_dir=output_dir,
    )


def _build_parameter_spec(search_space: dict, hpt_module: Any) -> dict:
    """Convert a search-space dict to Vertex AI parameter spec objects."""
    hpt = hpt_module
    parameter_spec: dict = {}
    for param_id, spec in search_space.items():
        kind = spec.get("type", "double")
        if kind == "double":
            parameter_spec[param_id] = hpt.DoubleParameterSpec(
                min=float(spec["min"]),
                max=float(spec["max"]),
                scale=spec.get("scale", "linear"),
            )
        elif kind == "discrete":
            parameter_spec[param_id] = hpt.DiscreteParameterSpec(
                values=[float(v) for v in spec["values"]], scale="linear"
            )
        elif kind == "categorical":
            parameter_spec[param_id] = hpt.CategoricalParameterSpec(values=spec["values"])
        else:
            logger.warning("Unknown parameter type %r for %s — skipping", kind, param_id)
    return parameter_spec


def _collect_trial_results(hpt_job: Any, stage: int, stage_config: dict) -> list[dict]:
    """Extract per-trial hyperparameters and outcomes from a completed HPT job.

    Each returned dict contains:

    * ``trial_id`` — Vertex AI trial identifier
    * ``stage`` — curriculum stage number
    * one key per hyperparameter (e.g. ``ppo_learning_rate``)
    * ``best_mean_reward`` — final metric reported by the trial
    * ``reward_threshold`` — ``min_avg_reward`` from the stage TOML config
    * ``stage_passed`` — ``True`` when ``best_mean_reward >= reward_threshold``
    """
    threshold = stage_config.get("curriculum_kwargs", {}).get("min_avg_reward")
    rows: list[dict] = []
    for trial in hpt_job.trials:
        row: dict = {"trial_id": trial.id, "stage": stage}
        if hasattr(trial, "parameters") and trial.parameters:
            for param in trial.parameters:
                row[param.parameter_id] = param.value
        best_reward = None
        if trial.final_measurement and trial.final_measurement.metrics:
            for metric in trial.final_measurement.metrics:
                if metric.metric_id == "best_mean_reward":
                    best_reward = metric.value
        row["best_mean_reward"] = best_reward
        row["reward_threshold"] = threshold
        row["stage_passed"] = best_reward is not None and threshold is not None and best_reward >= threshold
        rows.append(row)
    return rows


def write_results_csv(rows: list[dict], path: str | Path) -> Path:
    """Write sweep trial results to a CSV file.

    Each row records the trial ID, stage, all hyperparameter values,
    ``best_mean_reward``, the curriculum ``reward_threshold``, and
    ``stage_passed`` (``True`` when ``best_mean_reward >= reward_threshold``).

    Args:
        rows: List of result dicts from :func:`_collect_trial_results`.
        path: Output CSV path (parent directories are created as needed).

    Returns:
        Path to the written CSV file.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        logger.warning("No trial rows to write — skipping CSV")
        return path

    fixed_cols = ["trial_id", "stage"]
    metric_cols = ["best_mean_reward", "reward_threshold", "stage_passed"]
    # Collect all hyperparameter column names across all rows (union, sorted)
    hparam_cols: list[str] = sorted({k for row in rows for k in row if k not in fixed_cols + metric_cols})
    fieldnames = fixed_cols + hparam_cols + metric_cols

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Sweep results written to: %s", path)
    return path


def _best_trial_model_path(hpt_job: Any, bucket: str, species: str, stage: int) -> str:
    """Return the GCS container-mount path of the best trial's final model.

    Each trial writes its checkpoint to::

        /gcs/<bucket>/sweeps/<species>/stage<N>/<trial_id>/models/stage<N>_final.zip

    This function inspects the completed HPT job, identifies the trial with
    the highest ``best_mean_reward``, and returns that checkpoint path so the
    next stage's sweep can warm-start from it.
    """
    best_trial = None
    best_value = float("-inf")
    for trial in hpt_job.trials:
        if trial.final_measurement and trial.final_measurement.metrics:
            for metric in trial.final_measurement.metrics:
                if metric.metric_id == "best_mean_reward" and metric.value > best_value:
                    best_value = metric.value
                    best_trial = trial
    if best_trial is None:
        raise RuntimeError(f"No completed trials with 'best_mean_reward' found in HPT job {hpt_job.resource_name}")
    trial_id = best_trial.id
    logger.info("Best trial for stage %d: id=%s  best_mean_reward=%.4f", stage, trial_id, best_value)
    return f"/gcs/{bucket}/sweeps/{species}/stage{stage}/{trial_id}/models/stage{stage}_final.zip"


def _submit_stage_sweep(
    *,
    aiplatform,
    hpt_module,
    species: str,
    stage: int,
    algorithm: str,
    timesteps: int,
    n_envs: int,
    trials: int,
    parallel: int,
    bucket: str,
    image: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    search_space: dict,
    load_path: str | None = None,
    wandb: bool = False,
    sync: bool = False,
):
    """Build and submit a single-stage HPT job. Returns the job object."""
    parameter_spec = _build_parameter_spec(search_space, hpt_module)
    if not parameter_spec:
        logger.error("No valid parameters in search space for stage %d — aborting", stage)
        sys.exit(1)

    output_base = f"/gcs/{bucket}/sweeps/{species}/stage{stage}"

    trial_args = [
        "environments/shared/scripts/sweep.py",
        "trial",
        "--species",
        species,
        "--stage",
        str(stage),
        "--algorithm",
        algorithm,
        "--timesteps",
        str(timesteps),
        "--n-envs",
        str(n_envs),
        "--output-dir",
        output_base,
    ]
    if load_path:
        trial_args += ["--load", load_path]
    if wandb:
        trial_args.append("--wandb")

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": image,
                "command": ["python"],
                "args": trial_args,
            },
        }
    ]

    display_name = f"{species}-stage{stage}-{algorithm}-sweep"

    logger.info("Submitting HPT job: %s", display_name)
    logger.info("  Trials: %d  |  Parallel: %d", trials, parallel)
    logger.info("  Search space:")
    for k, v in search_space.items():
        logger.info("    %-30s %s", k, v)
    if load_path:
        logger.info("  Warm-start model: %s", load_path)

    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=display_name,
        metric_spec={"best_mean_reward": "maximize"},
        parameter_spec=parameter_spec,
        max_trial_count=trials,
        parallel_trial_count=parallel,
        worker_pool_specs=worker_pool_specs,
    )

    hpt_job.run(sync=sync)

    logger.info("Job submitted: %s", hpt_job.resource_name)
    logger.info("Monitor at: https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs")
    logger.info("Results will be written to: gs://%s/sweeps/%s/stage%d/", bucket, species, stage)
    return hpt_job


def launch_sweep(args: argparse.Namespace) -> None:
    """Submit a Vertex AI Hyperparameter Tuning job for a single stage.

    Each trial runs ``sweep.py trial`` inside the Docker container. The
    HPT service injects the trial's parameter values as additional CLI args.
    """
    try:
        from google.cloud import aiplatform
        from google.cloud.aiplatform import hyperparameter_tuning as hpt
    except ImportError:
        logger.error("google-cloud-aiplatform is not installed.\nInstall it with:  pip install google-cloud-aiplatform")
        sys.exit(1)

    aiplatform.init(
        project=args.project,
        location=args.location,
        staging_bucket=f"gs://{args.bucket}",
    )

    # Load search space: JSON override or algorithm default
    if args.search_space:
        try:
            search_space = json.loads(args.search_space)
        except json.JSONDecodeError as exc:
            logger.error("Invalid --search-space JSON: %s", exc)
            sys.exit(1)
    else:
        search_space = _DEFAULT_SEARCH_SPACES.get(args.algorithm, _DEFAULT_PPO_SEARCH_SPACE)

    _submit_stage_sweep(
        aiplatform=aiplatform,
        hpt_module=hpt,
        species=args.species,
        stage=args.stage,
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        trials=args.trials,
        parallel=args.parallel,
        bucket=args.bucket,
        image=args.image,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        search_space=search_space,
        wandb=args.wandb,
        sync=False,  # fire-and-forget for single-stage launch
    )


def launch_all_stages(args: argparse.Namespace) -> None:
    """Run Stage 1 → Stage 2 → Stage 3 HPT sweeps sequentially in one command.

    Workflow:

    1. Submit Stage 1 sweep and wait for it to complete.
    2. Identify the best Stage 1 trial (by ``best_mean_reward``).
    3. Submit Stage 2 sweep, loading the best Stage 1 checkpoint.
    4. Identify the best Stage 2 trial.
    5. Submit Stage 3 sweep, loading the best Stage 2 checkpoint.

    Each stage uses the same ``--trials`` / ``--parallel`` budget, but can
    have its own timestep budget via ``--timesteps-stage1/2/3``.
    The search space is shared across all stages (use separate ``launch``
    calls if you want different spaces per stage).
    """
    try:
        from google.cloud import aiplatform
        from google.cloud.aiplatform import hyperparameter_tuning as hpt
    except ImportError:
        logger.error("google-cloud-aiplatform is not installed.\nInstall it with:  pip install google-cloud-aiplatform")
        sys.exit(1)

    aiplatform.init(
        project=args.project,
        location=args.location,
        staging_bucket=f"gs://{args.bucket}",
    )

    if args.search_space:
        try:
            search_space = json.loads(args.search_space)
        except json.JSONDecodeError as exc:
            logger.error("Invalid --search-space JSON: %s", exc)
            sys.exit(1)
    else:
        search_space = _DEFAULT_SEARCH_SPACES.get(args.algorithm, _DEFAULT_PPO_SEARCH_SPACE)

    # Per-stage timestep budgets
    timesteps_per_stage = [
        args.timesteps_stage1,
        args.timesteps_stage2,
        args.timesteps_stage3,
    ]

    load_path: str | None = None
    all_rows: list[dict] = []

    for stage in range(1, 4):
        timesteps = timesteps_per_stage[stage - 1]
        logger.info("=" * 60)
        logger.info("ALL-STAGES SWEEP  —  Stage %d / 3", stage)
        logger.info("=" * 60)

        hpt_job = _submit_stage_sweep(
            aiplatform=aiplatform,
            hpt_module=hpt,
            species=args.species,
            stage=stage,
            algorithm=args.algorithm,
            timesteps=timesteps,
            n_envs=args.n_envs,
            trials=args.trials,
            parallel=args.parallel,
            bucket=args.bucket,
            image=args.image,
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            accelerator_count=args.accelerator_count,
            search_space=search_space,
            load_path=load_path,
            wandb=args.wandb,
            sync=True,  # block until this stage's sweep is done
        )

        # Collect per-trial results and append to the running list
        from environments.shared.config import load_stage_config as _load_stage_config

        stage_config = _load_stage_config(args.species, stage)
        stage_rows = _collect_trial_results(hpt_job, stage, stage_config)
        all_rows.extend(stage_rows)

        if stage < 3:
            # Find the best trial and chain its checkpoint into the next stage
            load_path = _best_trial_model_path(hpt_job, args.bucket, args.species, stage)
            logger.info("Stage %d complete. Best model: %s", stage, load_path)

    # Write a combined CSV of every trial across all three stages
    csv_path = Path(f"sweep_results_{args.species}_{args.algorithm}.csv")
    write_results_csv(all_rows, csv_path)

    logger.info("=" * 60)
    logger.info("ALL-STAGES SWEEP COMPLETE for %s (%s)", args.species, args.algorithm)
    logger.info("All results at: gs://%s/sweeps/%s/", args.bucket, args.species)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mesozoic Labs hyperparameter sweep tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Mode")

    # ── trial mode ────────────────────────────────────────────────────────────
    trial = subparsers.add_parser(
        "trial",
        help="Run one sweep trial (used by Vertex AI HPT workers inside Docker)",
    )
    trial.add_argument("--species", required=True, choices=["velociraptor", "brachiosaurus", "trex"])
    trial.add_argument("--stage", type=int, choices=[1, 2, 3], default=1)
    trial.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    trial.add_argument("--timesteps", type=int, default=500000, help="Training timesteps per trial")
    trial.add_argument("--n-envs", type=int, default=4, help="Parallel environments per trial")
    trial.add_argument("--seed", type=int, default=42)
    trial.add_argument("--eval-freq", type=int, default=10000)
    trial.add_argument("--save-freq", type=int, default=50000)
    trial.add_argument("--load", type=str, default=None, help="Path to model checkpoint to warm-start from")
    trial.add_argument("--output-dir", type=str, default=None, help="Base output dir (GCS mount path on Vertex AI)")
    trial.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    # ── launch mode (single stage) ────────────────────────────────────────────
    launch = subparsers.add_parser(
        "launch",
        help="Submit a Vertex AI Hyperparameter Tuning job for one stage",
    )
    launch.add_argument("--species", required=True, choices=["velociraptor", "brachiosaurus", "trex"])
    launch.add_argument("--stage", type=int, choices=[1, 2, 3], default=1, help="Curriculum stage to sweep")
    launch.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    launch.add_argument("--timesteps", type=int, default=500000, help="Training timesteps per trial")
    launch.add_argument("--n-envs", type=int, default=4, help="Parallel environments per trial")
    launch.add_argument("--project", required=True, help="GCP project ID")
    launch.add_argument("--location", default="us-central1", help="GCP region")
    launch.add_argument("--bucket", required=True, help="GCS bucket name (without gs:// prefix)")
    launch.add_argument("--image", required=True, help="Docker image URI for trial workers")
    launch.add_argument("--trials", type=int, default=20, help="Maximum number of trials")
    launch.add_argument("--parallel", type=int, default=5, help="Parallel trials running at once")
    launch.add_argument("--machine-type", default="n1-standard-8", help="Vertex AI machine type")
    launch.add_argument(
        "--accelerator-type",
        default="NVIDIA_TESLA_T4",
        help="Vertex AI accelerator type (use 'None' for CPU-only)",
    )
    launch.add_argument("--accelerator-count", type=int, default=1)
    launch.add_argument(
        "--search-space",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "JSON search space override. If omitted, the default space for the chosen "
            "algorithm is used. See module docstring for format details."
        ),
    )
    launch.add_argument("--wandb", action="store_true", help="Enable W&B logging in each trial")

    # ── launch-all mode (all stages, sequential) ──────────────────────────────
    launch_all = subparsers.add_parser(
        "launch-all",
        help="Sweep all three curriculum stages end-to-end with a single command",
    )
    launch_all.add_argument("--species", required=True, choices=["velociraptor", "brachiosaurus", "trex"])
    launch_all.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    launch_all.add_argument("--n-envs", type=int, default=4, help="Parallel environments per trial")
    launch_all.add_argument("--project", required=True, help="GCP project ID")
    launch_all.add_argument("--location", default="us-central1", help="GCP region")
    launch_all.add_argument("--bucket", required=True, help="GCS bucket name (without gs:// prefix)")
    launch_all.add_argument("--image", required=True, help="Docker image URI for trial workers")
    launch_all.add_argument("--trials", type=int, default=20, help="Maximum number of trials per stage")
    launch_all.add_argument("--parallel", type=int, default=5, help="Parallel trials per stage")
    launch_all.add_argument("--timesteps-stage1", type=int, default=500000, help="Timesteps per Stage 1 trial")
    launch_all.add_argument("--timesteps-stage2", type=int, default=1000000, help="Timesteps per Stage 2 trial")
    launch_all.add_argument("--timesteps-stage3", type=int, default=1500000, help="Timesteps per Stage 3 trial")
    launch_all.add_argument("--machine-type", default="n1-standard-8", help="Vertex AI machine type")
    launch_all.add_argument(
        "--accelerator-type",
        default="NVIDIA_TESLA_T4",
        help="Vertex AI accelerator type (use 'None' for CPU-only)",
    )
    launch_all.add_argument("--accelerator-count", type=int, default=1)
    launch_all.add_argument(
        "--search-space",
        type=str,
        default=None,
        metavar="JSON",
        help="JSON search space override (applied to all stages). Defaults to the algorithm's default space.",
    )
    launch_all.add_argument("--wandb", action="store_true", help="Enable W&B logging in each trial")

    return parser


def main() -> None:
    parser = _build_parser()
    # parse_known_args so Vertex AI HPT can inject extra --param value pairs
    args, extra_args = parser.parse_known_args()

    if args.mode == "trial":
        run_trial(args, extra_args)
    elif args.mode == "launch":
        if extra_args:
            logger.warning("Ignoring unexpected args in launch mode: %s", extra_args)
        launch_sweep(args)
    elif args.mode == "launch-all":
        if extra_args:
            logger.warning("Ignoring unexpected args in launch-all mode: %s", extra_args)
        launch_all_stages(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
