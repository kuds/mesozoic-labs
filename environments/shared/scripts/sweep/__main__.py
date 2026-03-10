"""CLI entry point for ``python -m environments.shared.scripts.sweep``."""

import argparse
import logging
import sys
from pathlib import Path

# Add repo root to Python path so environments.* imports work in Docker containers
_repo_root = str(Path(__file__).resolve().parents[4])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from .orchestration import launch_all_stages, launch_sweep
from .results import collect_results_from_disk, write_results_csv
from .trial import run_trial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mesozoic Labs hyperparameter sweep tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    trial.add_argument("--eval-freq", type=int, default=50000)
    trial.add_argument("--save-freq", type=int, default=50000)
    trial.add_argument("--load", type=str, default=None, help="Path to model checkpoint to warm-start from")
    trial.add_argument("--output-dir", type=str, default=None, help="Base output dir (GCS mount path on Vertex AI)")
    trial.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    trial.add_argument(
        "--no-tensorboard",
        action="store_true",
        default=False,
        help="Disable TensorBoard logging to reduce disk I/O and storage usage",
    )

    # ── launch mode (single stage) ────────────────────────────────────────────
    launch = subparsers.add_parser(
        "launch",
        help="Submit a Vertex AI Hyperparameter Tuning job for one stage",
    )
    launch.add_argument("--species", required=True, choices=["velociraptor", "brachiosaurus", "trex"])
    launch.add_argument("--stage", type=int, choices=[1, 2, 3], default=1, help="Curriculum stage to sweep")
    launch.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    launch.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Training timesteps per trial (default: 500k, or from search-space file)",
    )
    launch.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Parallel environments per trial (default: 4, or from search-space file)",
    )
    launch.add_argument("--project", required=True, help="GCP project ID")
    launch.add_argument("--location", default="us-central1", help="GCP region")
    launch.add_argument("--bucket", required=True, help="GCS bucket name (without gs:// prefix)")
    launch.add_argument("--image", required=True, help="Docker image URI for trial workers")
    launch.add_argument(
        "--trials", type=int, default=None, help="Maximum number of trials (default: 20, or from search-space file)"
    )
    launch.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Parallel trials running at once (default: 5, or from search-space file)",
    )
    launch.add_argument("--eval-freq", type=int, default=None, help="Eval frequency per trial (default: 50000)")
    launch.add_argument(
        "--save-freq", type=int, default=None, help="Checkpoint save frequency per trial (default: 50000)"
    )
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
            "JSON search space override (inline string). If omitted, the default space for the chosen "
            "algorithm is used. See module docstring for format details."
        ),
    )
    launch.add_argument(
        "--search-space-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file defining the search space. Supports per-stage "
            "sections (keys: stage1, stage2, stage3) or a flat dict for all stages."
        ),
    )
    launch.add_argument("--seed", type=int, default=42, help="Random seed for all trials (default: 42)")
    launch.add_argument("--wandb", action="store_true", help="Enable W&B logging in each trial")
    launch.add_argument(
        "--no-tensorboard",
        action="store_true",
        default=False,
        help="Disable TensorBoard logging in each trial to reduce disk I/O and storage usage",
    )
    launch.add_argument(
        "--load",
        type=str,
        default=None,
        help=(
            "Path to a model checkpoint to warm-start every trial from "
            "(e.g. best model from a prior stage). VecNormalize stats are "
            "loaded automatically from <load_path>_vecnorm.pkl if present."
        ),
    )
    launch.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from a previous run if a sweep state file exists "
            "(default: --resume). Use --no-resume to start fresh."
        ),
    )
    launch.add_argument(
        "--stage-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Maximum wall-clock seconds to wait for the HPT job to complete. "
            "Use 0 to submit and exit immediately without polling "
            "(fire-and-forget). If exceeded the job keeps running in the "
            "cloud — re-run with --resume to reconnect."
        ),
    )
    launch.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Seconds between job status checks while waiting for completion (default: 120)",
    )
    launch.add_argument(
        "--restart-job-on-worker-restart",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Restart the job if a worker is preempted (e.g. when using spot VMs). "
            "Maps to Vertex AI's restart_job_on_worker_restart parameter."
        ),
    )
    launch.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Print the resolved configuration (search space, machine type, "
            "trials, timesteps) without submitting the job. Useful for "
            "verifying settings before a large training run."
        ),
    )

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
    launch_all.add_argument(
        "--trials", type=int, default=20, help="Default max trials per stage (overridden by --trials-stageN)"
    )
    launch_all.add_argument(
        "--trials-stage1", type=int, default=None, help="Max trials for Stage 1 (defaults to --trials)"
    )
    launch_all.add_argument(
        "--trials-stage2", type=int, default=None, help="Max trials for Stage 2 (defaults to --trials)"
    )
    launch_all.add_argument(
        "--trials-stage3", type=int, default=None, help="Max trials for Stage 3 (defaults to --trials)"
    )
    launch_all.add_argument(
        "--parallel", type=int, default=5, help="Default parallel trials per stage (overridden by --parallel-stageN)"
    )
    launch_all.add_argument(
        "--parallel-stage1", type=int, default=None, help="Parallel trials for Stage 1 (defaults to --parallel)"
    )
    launch_all.add_argument(
        "--parallel-stage2", type=int, default=None, help="Parallel trials for Stage 2 (defaults to --parallel)"
    )
    launch_all.add_argument(
        "--parallel-stage3", type=int, default=None, help="Parallel trials for Stage 3 (defaults to --parallel)"
    )
    launch_all.add_argument(
        "--timesteps-stage1", type=int, default=None, help="Timesteps per Stage 1 trial (default: 500k)"
    )
    launch_all.add_argument(
        "--timesteps-stage2", type=int, default=None, help="Timesteps per Stage 2 trial (default: 1M)"
    )
    launch_all.add_argument(
        "--timesteps-stage3", type=int, default=None, help="Timesteps per Stage 3 trial (default: 1.5M)"
    )
    launch_all.add_argument("--eval-freq", type=int, default=None, help="Eval frequency per trial (default: 50000)")
    launch_all.add_argument(
        "--save-freq", type=int, default=None, help="Checkpoint save frequency per trial (default: 50000)"
    )
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
        help="JSON search space override (inline string, applied to all stages). Defaults to the algorithm's default space.",
    )
    launch_all.add_argument(
        "--search-space-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file defining the search space. Use per-stage keys "
            "(stage1, stage2, stage3) to give each stage its own search space, "
            "or a flat dict to apply the same space to all stages."
        ),
    )
    launch_all.add_argument("--seed", type=int, default=42, help="Random seed for all trials (default: 42)")
    launch_all.add_argument("--wandb", action="store_true", help="Enable W&B logging in each trial")
    launch_all.add_argument(
        "--no-tensorboard",
        action="store_true",
        default=False,
        help="Disable TensorBoard logging in each trial to reduce disk I/O and storage usage",
    )
    launch_all.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from the last completed stage if a sweep state file exists "
            "(default: --resume). Use --no-resume to start fresh."
        ),
    )
    launch_all.add_argument(
        "--force-continue",
        action="store_true",
        default=False,
        help=(
            "Continue to the next stage even when no trials pass the curriculum "
            "gate. The best trial (by reward) is selected regardless of gate "
            "status. Useful for running all three stages to completion."
        ),
    )
    launch_all.add_argument(
        "--stage-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Maximum wall-clock seconds to wait for each stage's HPT job to "
            "complete. Use 0 to submit and exit immediately without polling "
            "(fire-and-forget). If exceeded the job keeps running in the "
            "cloud — re-run with --resume to reconnect."
        ),
    )
    launch_all.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Seconds between job status checks while waiting for completion (default: 120)",
    )
    launch_all.add_argument(
        "--restart-job-on-worker-restart",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Restart the job if a worker is preempted (e.g. when using spot VMs). "
            "Maps to Vertex AI's restart_job_on_worker_restart parameter."
        ),
    )
    launch_all.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Print the resolved configuration for all stages without "
            "submitting any jobs. Useful for verifying settings before "
            "a large training run."
        ),
    )

    # ── collect-results mode ────────────────────────────────────────────────
    collect = subparsers.add_parser(
        "collect-results",
        help="Scan trial directories for metrics.json and produce a combined CSV",
    )
    collect.add_argument(
        "output_dir",
        type=str,
        help=(
            "Root directory containing stage sub-directories with trial results. "
            "For sweeps: .../sweeps/<species>  For curriculum: .../curriculum_<timestamp>"
        ),
    )
    collect.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Output CSV path (default: <output_dir>/collected_results.csv)",
    )
    collect.add_argument(
        "--species",
        type=str,
        default=None,
        help="Species name (used in log messages and plot titles)",
    )
    collect.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Algorithm name (used in plot titles when --plot is set)",
    )
    collect.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Only collect results for these stage numbers (default: all)",
    )
    collect.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate visualisation graphs alongside the CSV",
    )

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
        if args.parallel is not None and args.trials is not None and args.parallel > args.trials:
            logger.warning(
                "--parallel (%d) exceeds --trials (%d). Clamping parallel to %d.",
                args.parallel,
                args.trials,
                args.trials,
            )
            args.parallel = args.trials
        launch_sweep(args)
    elif args.mode == "launch-all":
        if extra_args:
            logger.warning("Ignoring unexpected args in launch-all mode: %s", extra_args)
        # Validate per-stage parallel <= trials where both are explicitly set
        for stage_num in (1, 2, 3):
            p = getattr(args, f"parallel_stage{stage_num}", None) or args.parallel
            t = getattr(args, f"trials_stage{stage_num}", None) or args.trials
            if p > t:
                logger.warning(
                    "Stage %d: --parallel (%d) exceeds --trials (%d). Clamping parallel to %d.",
                    stage_num,
                    p,
                    t,
                    t,
                )
                if getattr(args, f"parallel_stage{stage_num}", None) is not None:
                    setattr(args, f"parallel_stage{stage_num}", t)
                else:
                    args.parallel = t
        launch_all_stages(args)
    elif args.mode == "collect-results":
        if extra_args:
            logger.warning("Ignoring unexpected args in collect-results mode: %s", extra_args)
        rows = collect_results_from_disk(
            args.output_dir,
            species=args.species,
            stages=args.stages,
        )
        if not rows:
            logger.error("No results found — nothing to write.")
            sys.exit(1)
        if args.csv:
            csv_path = args.csv
        elif args.output_dir.startswith("gs://"):
            csv_path = args.output_dir.rstrip("/") + "/collected_results.csv"
        else:
            csv_path = str(Path(args.output_dir) / "collected_results.csv")
        write_results_csv(rows, csv_path)
        logger.info("Combined CSV written to: %s  (%d rows)", csv_path, len(rows))
        if args.plot:
            from .results import plot_sweep_results

            if args.species:
                species = args.species
            elif args.output_dir.startswith("gs://"):
                species = args.output_dir.rstrip("/").rsplit("/", 1)[-1]
            else:
                species = Path(args.output_dir).name
            algorithm = args.algorithm or "unknown"
            plot_sweep_results(csv_path, species, algorithm)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
