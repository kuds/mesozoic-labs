"""High-level sweep orchestration — single-stage and multi-stage launches."""

import argparse
import logging
import sys
import time
from pathlib import Path

from .constants import SweepStageError, _SweepJobFailed
from .results import (
    _best_trial_model_path,
    _best_trial_model_path_any,
    _collect_trial_results,
    plot_sweep_results,
    write_results_csv,
)
from .search_space import (
    _resolve_search_space,
    _search_space_for_stage,
    _settings_for_stage,
)
from .state import _load_sweep_state, _save_sweep_state
from .submit import _is_retryable_gcp_error, _submit_stage_sweep, _wait_for_job

logger = logging.getLogger(__name__)


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

    # Load search space: inline JSON > file > algorithm default
    resolved = _resolve_search_space(args.search_space, args.search_space_file, args.algorithm)
    search_space = _search_space_for_stage(resolved, args.stage)

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
        load_path=args.load,
        wandb=args.wandb,
    )


def launch_all_stages(args: argparse.Namespace) -> None:
    """Run Stage 1 → Stage 2 → Stage 3 HPT sweeps sequentially in one command.

    Workflow:

    1. Submit Stage 1 sweep and wait for it to complete.
    2. Identify the best Stage 1 trial (by ``best_mean_reward``).
    3. Submit Stage 2 sweep, loading the best Stage 1 checkpoint.
    4. Identify the best Stage 2 trial.
    5. Submit Stage 3 sweep, loading the best Stage 2 checkpoint.

    Each stage can have its own budget via ``--trials-stageN``,
    ``--parallel-stageN``, and ``--timesteps-stageN`` flags.  When a
    per-stage flag is omitted, the shared ``--trials`` / ``--parallel``
    default is used.

    The search space can be customised per stage using a JSON file with
    ``"stage1"`` / ``"stage2"`` / ``"stage3"`` top-level keys (see
    ``--search-space-file``).  A flat JSON file or inline ``--search-space``
    applies the same space to all three stages.
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

    # Load search space: inline JSON > file > algorithm default
    # If the file uses per-stage keys ("stage1", "stage2", "stage3"), each
    # stage gets its own search space.  Otherwise the same space is reused.
    resolved = _resolve_search_space(args.search_space, args.search_space_file, args.algorithm)

    # Per-stage budgets: CLI flags > search-space file settings > shared defaults
    # The search-space file can include "trials", "timesteps", "parallel",
    # "n_envs" alongside the search-space params.  CLI flags always win.
    cli_timesteps = [args.timesteps_stage1, args.timesteps_stage2, args.timesteps_stage3]
    cli_trials = [
        args.trials_stage1,
        args.trials_stage2,
        args.trials_stage3,
    ]
    cli_parallel = [
        args.parallel_stage1,
        args.parallel_stage2,
        args.parallel_stage3,
    ]

    load_path: str | None = None
    fixed_trial_args: list[str] | None = None
    all_rows: list[dict] = []
    sweep_start_time = time.monotonic()

    # Determine the net_arch HPT key for this algorithm (e.g. "ppo_net_arch")
    net_arch_key = f"{args.algorithm}_net_arch"

    # ── CLI args snapshot (persisted in state for reproducibility) ───────────
    cli_args_snapshot: dict = {
        "trials": args.trials,
        "parallel": args.parallel,
        "n_envs": args.n_envs,
        "machine_type": args.machine_type,
        "accelerator_type": args.accelerator_type,
        "accelerator_count": args.accelerator_count,
        "timesteps_stage1": args.timesteps_stage1,
        "timesteps_stage2": args.timesteps_stage2,
        "timesteps_stage3": args.timesteps_stage3,
        "trials_stage1": args.trials_stage1,
        "trials_stage2": args.trials_stage2,
        "trials_stage3": args.trials_stage3,
        "parallel_stage1": args.parallel_stage1,
        "parallel_stage2": args.parallel_stage2,
        "parallel_stage3": args.parallel_stage3,
        "image": args.image,
        "bucket": args.bucket,
        "project": args.project,
        "location": args.location,
        "force_continue": args.force_continue,
        "stage_timeout": getattr(args, "stage_timeout", None),
        "poll_interval": getattr(args, "poll_interval", 120),
    }

    # ── Resume: restore state from a previous (possibly interrupted) run ─────
    completed_stages: set[int] = set()
    partial_stages: dict[int, dict] = {}
    in_progress_stages: dict[int, dict] = {}
    sweep_state: dict = {
        "species": args.species,
        "algorithm": args.algorithm,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cli_args": cli_args_snapshot,
        "stages": {},
    }
    logger.info("Resume mode: %s", "enabled" if args.resume else "disabled")
    if args.resume:
        prev_state = _load_sweep_state(args.species, args.algorithm, bucket=args.bucket, project=args.project)
        if prev_state and prev_state.get("stages"):
            # Warn if CLI args differ from the previous run
            prev_cli = prev_state.get("cli_args", {})
            if prev_cli:
                changed = {
                    k for k in set(prev_cli) | set(cli_args_snapshot) if prev_cli.get(k) != cli_args_snapshot.get(k)
                }
                if changed:
                    logger.warning(
                        "CLI args differ from previous run on: %s. Current values will be used.",
                        ", ".join(sorted(changed)),
                    )

            sweep_state = prev_state
            sweep_state["cli_args"] = cli_args_snapshot  # always store current args

            for stg_key, stg_data in prev_state["stages"].items():
                stg_num = int(stg_key)
                if stg_data.get("status") == "completed":
                    completed_stages.add(stg_num)
                    # Restore load_path and fixed_trial_args from the last completed stage
                    if stg_data.get("best_model_path"):
                        load_path = stg_data["best_model_path"]
                    if stg_data.get("fixed_trial_args"):
                        fixed_trial_args = stg_data["fixed_trial_args"]
                    if stg_data.get("trial_rows"):
                        all_rows.extend(stg_data["trial_rows"])
                elif stg_data.get("status") == "in_progress":
                    in_progress_stages[stg_num] = stg_data
                    if stg_data.get("fixed_trial_args"):
                        fixed_trial_args = stg_data["fixed_trial_args"]
                elif stg_data.get("status") == "partial":
                    partial_stages[stg_num] = stg_data
                    if stg_data.get("fixed_trial_args"):
                        fixed_trial_args = stg_data["fixed_trial_args"]
            if completed_stages:
                logger.info(
                    "Resuming sweep: stages %s already completed. load_path=%s",
                    sorted(completed_stages),
                    load_path,
                )
            if in_progress_stages:
                for ips, ipd in sorted(in_progress_stages.items()):
                    logger.info(
                        "Resuming sweep: stage %d has an in-progress job: %s",
                        ips,
                        ipd.get("job_resource_name", "unknown"),
                    )
            if partial_stages:
                for ps, pd in sorted(partial_stages.items()):
                    logger.info(
                        "Resuming sweep: stage %d has %d partial trial results from a previous run.",
                        ps,
                        len(pd.get("trial_rows", [])),
                    )
    # ── End resume ───────────────────────────────────────────────────────────

    for stage in range(1, 4):
        # Skip stages that were already completed in a previous run
        if stage in completed_stages:
            logger.info("=" * 60)
            logger.info("ALL-STAGES SWEEP  —  Stage %d / 3  (SKIPPED — already completed)", stage)
            logger.info("=" * 60)
            continue

        stage_start_time = time.monotonic()
        search_space = _search_space_for_stage(resolved, stage)
        file_settings = _settings_for_stage(resolved, stage)

        # Resolve each setting: CLI flag > file setting > shared CLI default
        timesteps = cli_timesteps[stage - 1]
        if timesteps is None:
            timesteps = file_settings.get("timesteps", cli_timesteps[stage - 1])
        if timesteps is None:
            # Fall back to the argparse defaults (500k/1M/1.5M)
            timesteps = [500_000, 1_000_000, 1_500_000][stage - 1]

        trials = cli_trials[stage - 1]
        if trials is None:
            trials = file_settings.get("trials", args.trials)

        parallel = cli_parallel[stage - 1]
        if parallel is None:
            parallel = file_settings.get("parallel", args.parallel)

        n_envs = file_settings.get("n_envs", args.n_envs)

        # ── Partial recovery: check for trial results from a previous
        #    interrupted run of this stage ──────────────────────────────────
        partial_data = partial_stages.get(stage)
        partial_rows: list[dict] = []
        resume_run = 0
        if partial_data:
            partial_rows = [r for r in partial_data.get("trial_rows", []) if r.get("best_mean_reward") is not None]
            if partial_rows:
                resume_run = partial_data.get("resume_run", 0) + 1

        logger.info("=" * 60)
        logger.info("ALL-STAGES SWEEP  —  Stage %d / 3", stage)
        logger.info(
            "  Timesteps: %s  |  Trials: %d  |  Parallel: %d  |  n_envs: %d", timesteps, trials, parallel, n_envs
        )
        logger.info("=" * 60)

        poll_interval = getattr(args, "poll_interval", 120)
        stage_timeout = getattr(args, "stage_timeout", None)

        try:
            remaining_trials = trials - len(partial_rows)
            job_resource_name = None
            hpt_job = None
            reconnected = False

            # ── Reconnect to an in-progress job from a previous run ───────
            in_progress_data = in_progress_stages.get(stage)
            if in_progress_data and in_progress_data.get("job_resource_name"):
                prev_resource = in_progress_data["job_resource_name"]
                logger.info("Attempting to reconnect to previous job: %s", prev_resource)

                # Restore partial rows from runs that preceded the in-progress
                # job so they aren't lost across multiple resume cycles.
                prior_partial = in_progress_data.get("prior_partial_rows", [])
                if prior_partial:
                    partial_rows = partial_rows + prior_partial
                    remaining_trials = trials - len(partial_rows)
                    logger.info(
                        "Restored %d prior partial rows from earlier runs.",
                        len(prior_partial),
                    )

                try:
                    prev_job = aiplatform.HyperparameterTuningJob.get(prev_resource)
                    prev_state_name = prev_job.state.name if hasattr(prev_job.state, "name") else str(prev_job.state)

                    if "SUCCEEDED" in prev_state_name:
                        logger.info("Previous job already completed successfully.")
                        hpt_job = prev_job
                        job_resource_name = prev_resource
                        reconnected = True
                    elif any(s in prev_state_name for s in ("RUNNING", "QUEUED", "PENDING")):
                        logger.info("Previous job still running (state=%s) — resuming poll.", prev_state_name)
                        hpt_job = _wait_for_job(
                            prev_job,
                            aiplatform,
                            poll_interval=poll_interval,
                            timeout=stage_timeout,
                        )
                        job_resource_name = prev_resource
                        reconnected = True
                    else:
                        # Failed/cancelled — collect partial results
                        logger.warning(
                            "Previous job in state %s — collecting partial results.",
                            prev_state_name,
                        )
                        try:
                            from environments.shared.config import load_stage_config as _lsc_ip

                            _sc_ip = _lsc_ip(args.species, stage)
                            _ip_rows = _collect_trial_results(prev_job, stage, _sc_ip)
                            _ip_rows = [r for r in _ip_rows if r.get("best_mean_reward") is not None]
                            if _ip_rows:
                                _ip_out = f"/gcs/{args.bucket}/sweeps/{args.species}/stage{stage}"
                                ip_resume = in_progress_data.get("resume_run", 0)
                                if ip_resume:
                                    _ip_out = f"{_ip_out}_r{ip_resume}"
                                for row in _ip_rows:
                                    row["model_path"] = f"{_ip_out}/{row['trial_id']}/models/best_model.zip"
                                partial_rows = partial_rows + _ip_rows
                                remaining_trials = trials - len(partial_rows)
                                resume_run = ip_resume + 1
                                logger.info(
                                    "Recovered %d trials from previous in-progress job (%d total partial).",
                                    len(_ip_rows),
                                    len(partial_rows),
                                )
                        except Exception as ip_collect_exc:
                            logger.warning(
                                "Could not collect partial results from previous job: %s",
                                ip_collect_exc,
                            )
                except Exception as reconnect_exc:
                    logger.warning(
                        "Could not reconnect to previous job %s: %s. Will submit a new job.",
                        prev_resource,
                        reconnect_exc,
                    )

            # ── Submit new job if needed ───────────────────────────────────
            if not reconnected and remaining_trials > 0:
                if partial_rows:
                    logger.info(
                        "Resuming stage %d: %d trials recovered from previous run, %d remaining.",
                        stage,
                        len(partial_rows),
                        remaining_trials,
                    )

                hpt_job = _submit_stage_sweep(
                    aiplatform=aiplatform,
                    hpt_module=hpt,
                    species=args.species,
                    stage=stage,
                    algorithm=args.algorithm,
                    timesteps=timesteps,
                    n_envs=n_envs,
                    trials=remaining_trials,
                    parallel=parallel,
                    bucket=args.bucket,
                    image=args.image,
                    machine_type=args.machine_type,
                    accelerator_type=args.accelerator_type,
                    accelerator_count=args.accelerator_count,
                    search_space=search_space,
                    load_path=load_path,
                    fixed_trial_args=fixed_trial_args,
                    wandb=args.wandb,
                    resume_run=resume_run,
                )
                job_resource_name = getattr(hpt_job, "resource_name", None)

                # Save in-progress state immediately so the job can be
                # reconnected if the orchestrator is interrupted.
                # Include any partial_rows recovered from earlier runs so
                # they survive across multiple resume cycles.
                sweep_state["stages"][str(stage)] = {
                    "status": "in_progress",
                    "job_resource_name": job_resource_name,
                    "load_path": load_path,
                    "fixed_trial_args": fixed_trial_args,
                    "resume_run": resume_run,
                    "prior_partial_rows": partial_rows,
                    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                _save_sweep_state(
                    sweep_state,
                    args.species,
                    args.algorithm,
                    bucket=args.bucket,
                    project=args.project,
                )

                # Poll until the job completes
                hpt_job = _wait_for_job(
                    hpt_job,
                    aiplatform,
                    poll_interval=poll_interval,
                    timeout=stage_timeout,
                )

            # ── Collect results ────────────────────────────────────────────
            if hpt_job is not None:
                from environments.shared.config import load_stage_config as _load_stage_config

                stage_config = _load_stage_config(args.species, stage)
                new_rows = _collect_trial_results(hpt_job, stage, stage_config)

                # Tag new rows with model_path so _best_trial_model_path can
                # locate the correct checkpoint even when trials span multiple
                # GCS output directories (original run vs resumed run).
                output_base = f"/gcs/{args.bucket}/sweeps/{args.species}/stage{stage}"
                if resume_run:
                    output_base = f"{output_base}_r{resume_run}"
                for row in new_rows:
                    row["model_path"] = f"{output_base}/{row['trial_id']}/models/best_model.zip"

                stage_rows = partial_rows + new_rows
            elif remaining_trials <= 0:
                # All requested trials were already completed in a previous
                # partial run — no need to submit a new job.
                logger.info(
                    "Stage %d: all %d trials already completed in previous run. Skipping job submission.",
                    stage,
                    len(partial_rows),
                )
                stage_rows = list(partial_rows)
            else:
                stage_rows = list(partial_rows)

            all_rows.extend(stage_rows)

            stage_elapsed = time.monotonic() - stage_start_time
            stage_mins = stage_elapsed / 60
            logger.info(
                "Stage %d finished in %.1f min (%.0f s). Trials: %d total, %d passed.",
                stage,
                stage_mins,
                stage_elapsed,
                len(stage_rows),
                sum(1 for r in stage_rows if r.get("stage_passed")),
            )

            # Identify the best passing trial for this stage (used for
            # chaining stages 1→2→3 and for reporting in the saved state).
            try:
                best_model_path, best_row = _best_trial_model_path(stage_rows, args.bucket, args.species, stage)
            except SweepStageError:
                # No trials passed the curriculum gate.
                if args.force_continue and stage < 3:
                    # --force-continue: pick the best trial regardless of
                    # gate status and chain it into the next stage.
                    logger.warning(
                        "Stage %d: no trials passed curriculum gate. "
                        "--force-continue is set — selecting best trial anyway.",
                        stage,
                    )
                    best_model_path, best_row = _best_trial_model_path_any(
                        stage_rows,
                        args.bucket,
                        args.species,
                        stage,
                    )
                else:
                    best_model_path = None
                    best_row = None
                    if stage < 3:
                        raise  # stages 1-2 must pass to chain forward

            if best_row is not None:
                logger.info(
                    "Stage %d best passing trial: id=%s  reward=%.4f",
                    stage,
                    best_row["trial_id"],
                    best_row.get("best_mean_reward", 0),
                )

            if stage < 3 and best_model_path is not None:
                # Chain the best checkpoint into the next stage
                load_path = best_model_path
                logger.info("Stage %d complete. Best passing model: %s", stage, load_path)

                # Propagate the winning trial's net_arch to subsequent stages so
                # every trial loads the checkpoint with a matching architecture.
                # net_arch is only searched in stage 1; stages 2+ inherit it.
                best_net_arch = best_row.get(net_arch_key) if best_row else None
                if best_net_arch is not None:
                    fixed_trial_args = [f"--{net_arch_key}", str(best_net_arch)]
                    logger.info("Propagating %s=%s from best trial to stage %d", net_arch_key, best_net_arch, stage + 1)

            # Save state after each stage completes successfully
            sweep_state["stages"][str(stage)] = {
                "status": "completed",
                "job_resource_name": job_resource_name,
                "best_trial_id": best_row["trial_id"] if best_row else None,
                "best_mean_reward": best_row.get("best_mean_reward") if best_row else None,
                "best_model_path": best_model_path if stage < 3 else None,
                "fixed_trial_args": fixed_trial_args,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "trial_rows": stage_rows,
            }
            _save_sweep_state(
                sweep_state,
                args.species,
                args.algorithm,
                bucket=args.bucket,
                project=args.project,
            )

        except Exception as exc:
            if isinstance(exc, (SweepStageError, TimeoutError)) or _is_retryable_gcp_error(exc):
                # ── Partial trial recovery ────────────────────────────────
                # Try to salvage completed trials from the failed job so they
                # can be reused on the next ``--resume`` run.
                if isinstance(exc, _SweepJobFailed) and exc.hpt_job is not None:
                    try:
                        from environments.shared.config import load_stage_config as _lsc

                        _sc = _lsc(args.species, stage)
                        _new_partial = _collect_trial_results(exc.hpt_job, stage, _sc)
                        # Keep only trials that actually reported metrics
                        _new_partial = [r for r in _new_partial if r.get("best_mean_reward") is not None]
                        if _new_partial:
                            # Tag each row with its model_path for correct
                            # checkpoint resolution on resume.
                            _out = f"/gcs/{args.bucket}/sweeps/{args.species}/stage{stage}"
                            if resume_run:
                                _out = f"{_out}_r{resume_run}"
                            for row in _new_partial:
                                row["model_path"] = f"{_out}/{row['trial_id']}/models/best_model.zip"

                            all_partial = partial_rows + _new_partial
                            sweep_state["stages"][str(stage)] = {
                                "status": "partial",
                                "job_resource_name": getattr(exc.hpt_job, "resource_name", None),
                                "trial_rows": all_partial,
                                "fixed_trial_args": fixed_trial_args,
                                "resume_run": resume_run,
                            }
                            logger.info(
                                "Recovered %d completed trials from failed stage %d "
                                "(total partial: %d). These will be reused on resume.",
                                len(_new_partial),
                                stage,
                                len(all_partial),
                            )
                    except Exception as collect_exc:
                        logger.warning("Could not collect partial trial results: %s", collect_exc)

                logger.error(
                    "Stage %d failed: %s. Saving progress — completed stages: %s. "
                    "Re-run the same command to resume from where you left off.",
                    stage,
                    exc,
                    sorted(int(k) for k, v in sweep_state["stages"].items() if v.get("status") == "completed"),
                )
                _save_sweep_state(
                    sweep_state,
                    args.species,
                    args.algorithm,
                    bucket=args.bucket,
                    project=args.project,
                )
                sys.exit(1)
            raise

    # Write a combined CSV of every trial across all three stages
    csv_path = Path(f"sweep_results_{args.species}_{args.algorithm}.csv")
    write_results_csv(all_rows, csv_path)

    # Persist the CSV and graphs to GCS alongside the trial artifacts
    _gcs_bucket = None
    try:
        from google.cloud import storage as _gcs

        _gcs_client = _gcs.Client(project=args.project)
        _gcs_bucket = _gcs_client.bucket(args.bucket)
    except Exception as exc:
        logger.warning("Could not initialise GCS client for uploads: %s", exc)

    if _gcs_bucket is not None:
        gcs_csv_path = f"sweeps/{args.species}/{csv_path.name}"
        try:
            _gcs_bucket.blob(gcs_csv_path).upload_from_filename(str(csv_path))
            logger.info("Sweep CSV uploaded to: gs://%s/%s", args.bucket, gcs_csv_path)
        except Exception as exc:
            logger.warning("Failed to upload sweep CSV to GCS: %s. Local copy at: %s", exc, csv_path)

    # Generate sweep visualisation graphs
    plot_sweep_results(csv_path, args.species, args.algorithm)

    # Upload graphs to GCS alongside the CSV
    if _gcs_bucket is not None:
        for graph_name in ("sweep_trial_metrics.png", "sweep_hyperparameter_analysis.png"):
            graph_path = csv_path.parent / graph_name
            if graph_path.exists():
                gcs_graph_path = f"sweeps/{args.species}/{graph_name}"
                try:
                    _gcs_bucket.blob(gcs_graph_path).upload_from_filename(str(graph_path))
                    logger.info("Sweep graph uploaded to: gs://%s/%s", args.bucket, gcs_graph_path)
                except Exception as exc:
                    logger.warning("Failed to upload %s to GCS: %s", graph_name, exc)

    total_elapsed = time.monotonic() - sweep_start_time
    total_mins = total_elapsed / 60
    logger.info("=" * 60)
    logger.info(
        "ALL-STAGES SWEEP COMPLETE for %s (%s) in %.1f min (%.0f s)",
        args.species,
        args.algorithm,
        total_mins,
        total_elapsed,
    )
    logger.info("All results at: gs://%s/sweeps/%s/", args.bucket, args.species)
