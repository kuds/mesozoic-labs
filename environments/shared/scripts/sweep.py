#!/usr/bin/env python3
"""Hyperparameter sweep tool for Mesozoic Labs.

Two modes:

  **launch** — Submit a Vertex AI Hyperparameter Tuning job.
               Vertex AI uses Bayesian optimisation to run N parallel trials
               and automatically find the best hyperparameter combination.
               Requires: ``pip install google-cloud-aiplatform``

  **trial**  — Entry point used by each Vertex AI HPT trial worker running
               inside Docker. Receives hyperparameter values injected by the
               HPT service as ``--ppo_learning_rate X --ppo_ent_coef Y``
               style args, converts them to ``--override`` dot notation, and
               runs the training script.

Usage examples::

    # ── launch a Stage-1 PPO sweep (from your local machine) ───────────────
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
import sys
from pathlib import Path

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
            param = key[len(prefix) + 1:]
            return f"{prefix}.{param}={value}"
    # Unrecognised prefix — pass through as-is (best-effort)
    return f"{key}={value}"


def run_trial(args: argparse.Namespace, extra_args: list[str]) -> None:
    """Run a single training trial.

    ``extra_args`` contains the hyperparameter values injected by Vertex AI
    HPT (e.g. ``['--ppo_learning_rate', '0.0003', '--ppo_ent_coef', '0.01']``).
    They are converted to ``--override`` format before calling ``train()``.
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
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_subproc=False,
        verbose=1,
        algorithm=args.algorithm,
        use_wandb=args.wandb,
        output_dir=args.output_dir,
    )


def launch_sweep(args: argparse.Namespace) -> None:
    """Submit a Vertex AI Hyperparameter Tuning job.

    Each trial runs ``sweep.py trial`` inside the Docker container. The
    HPT service injects the trial's parameter values as additional CLI args.
    """
    try:
        from google.cloud import aiplatform
        from google.cloud.aiplatform import hyperparameter_tuning as hpt
    except ImportError:
        logger.error(
            "google-cloud-aiplatform is not installed.\n"
            "Install it with:  pip install google-cloud-aiplatform"
        )
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

    # Build Vertex AI parameter specs from the search space dict
    parameter_spec: dict = {}
    for param_id, spec in search_space.items():
        kind = spec.get("type", "double")
        if kind == "double":
            parameter_spec[param_id] = hpt.DoubleValueSpec(
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                scale=spec.get("scale", "linear"),
            )
        elif kind == "discrete":
            parameter_spec[param_id] = hpt.DiscreteValueSpec(
                values=[float(v) for v in spec["values"]]
            )
        elif kind == "categorical":
            parameter_spec[param_id] = hpt.CategoricalValueSpec(values=spec["values"])
        else:
            logger.warning("Unknown parameter type %r for %s — skipping", kind, param_id)

    if not parameter_spec:
        logger.error("No valid parameters in search space — aborting")
        sys.exit(1)

    # Output directory inside the GCS-mounted path (one sub-dir per trial)
    output_base = f"/gcs/{args.bucket}/sweeps/{args.species}/stage{args.stage}"

    # Fixed args for every trial; Vertex AI HPT appends the hyperparameter values
    trial_args = [
        "environments/shared/scripts/sweep.py",
        "trial",
        "--species", args.species,
        "--stage", str(args.stage),
        "--algorithm", args.algorithm,
        "--timesteps", str(args.timesteps),
        "--n-envs", str(args.n_envs),
        "--output-dir", output_base,
    ]
    if args.wandb:
        trial_args.append("--wandb")

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": args.machine_type,
                "accelerator_type": args.accelerator_type,
                "accelerator_count": args.accelerator_count,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": args.image,
                "command": ["python"],
                "args": trial_args,
            },
        }
    ]

    display_name = f"{args.species}-stage{args.stage}-{args.algorithm}-sweep"

    logger.info("Submitting HPT job: %s", display_name)
    logger.info("  Trials: %d  |  Parallel: %d", args.trials, args.parallel)
    logger.info("  Search space:")
    for k, v in search_space.items():
        logger.info("    %-30s %s", k, v)

    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=display_name,
        metric_spec={"best_mean_reward": "maximize"},
        parameter_spec=parameter_spec,
        max_trial_count=args.trials,
        parallel_trial_count=args.parallel,
        worker_pool_specs=worker_pool_specs,
    )

    hpt_job.run(sync=False)

    logger.info("Job submitted: %s", hpt_job.resource_name)
    logger.info(
        "Monitor at: https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs"
    )
    logger.info("Results will be written to: gs://%s/sweeps/%s/stage%d/", args.bucket, args.species, args.stage)


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
    trial.add_argument("--output-dir", type=str, default=None, help="Base output dir (GCS mount path on Vertex AI)")
    trial.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    # ── launch mode ───────────────────────────────────────────────────────────
    launch = subparsers.add_parser(
        "launch",
        help="Submit a Vertex AI Hyperparameter Tuning job",
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
