"""High-level sweep orchestration — single-stage and multi-stage launches."""

import argparse
import logging
import os
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


# ── Shared helpers (used by both launch_sweep and launch_all_stages) ──────


def _reconnect_or_collect_partial(
    aiplatform,
    prev_resource: str,
    stage: int,
    species: str,
    bucket: str,
    partial_rows: list[dict],
    resume_run: int,
    poll_interval: int = 120,
    stage_timeout: int | None = None,
) -> tuple[object | None, str | None, bool, list[dict], int]:
    """Try to reconnect to a previous HPT job or collect its partial results.

    Returns ``(hpt_job, job_resource_name, reconnected, partial_rows, resume_run)``.
    """
    hpt_job = None
    job_resource_name = None
    reconnected = False

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
            partial_rows, resume_run = _try_collect_partial_from_job(
                prev_job,
                stage,
                species,
                bucket,
                partial_rows,
                resume_run,
            )
    except Exception as reconnect_exc:
        logger.warning(
            "Could not reconnect to previous job %s: %s. Will submit a new job.",
            prev_resource,
            reconnect_exc,
        )

    return hpt_job, job_resource_name, reconnected, partial_rows, resume_run


def _try_collect_partial_from_job(
    job,
    stage: int,
    species: str,
    bucket: str,
    existing_partial: list[dict],
    prev_resume_run: int,
) -> tuple[list[dict], int]:
    """Attempt to collect partial trial results from a completed/failed job.

    Returns ``(updated_partial_rows, new_resume_run)``.
    """
    try:
        from environments.shared.config import load_stage_config as _lsc

        stage_config = _lsc(species, stage)
        output_base = f"/gcs/{bucket}/sweeps/{species}/stage{stage}"
        if prev_resume_run:
            output_base = f"{output_base}_r{prev_resume_run}"
        new_rows = _collect_trial_results(job, stage, stage_config, output_base=output_base)
        new_rows = [r for r in new_rows if r.get("best_mean_reward") is not None]
        if new_rows:
            for row in new_rows:
                row["model_path"] = f"{output_base}/{row['trial_id']}/models/best_model.zip"
            updated = _dedup_trial_rows(existing_partial + new_rows)
            new_resume_run = prev_resume_run + 1
            logger.info(
                "Recovered %d trials from previous job (%d total partial).",
                len(new_rows),
                len(updated),
            )
            return updated, new_resume_run
    except Exception as exc:
        logger.warning("Could not collect partial results from previous job: %s", exc)
    return existing_partial, prev_resume_run


def _collect_and_tag_rows(
    hpt_job,
    stage: int,
    species: str,
    bucket: str,
    resume_run: int,
) -> list[dict]:
    """Collect trial results from a completed HPT job and tag with model_path."""
    from environments.shared.config import load_stage_config as _load_stage_config

    stage_config = _load_stage_config(species, stage)
    output_base = f"/gcs/{bucket}/sweeps/{species}/stage{stage}"
    if resume_run:
        output_base = f"{output_base}_r{resume_run}"
    new_rows = _collect_trial_results(hpt_job, stage, stage_config, output_base=output_base)
    for row in new_rows:
        row["model_path"] = f"{output_base}/{row['trial_id']}/models/best_model.zip"
    return new_rows


def _handle_stage_failure(
    exc: Exception,
    stage: int,
    species: str,
    algorithm: str,
    bucket: str,
    project: str | None,
    sweep_state: dict,
    state_stage_key: str,
    partial_rows: list[dict],
    resume_run: int,
    fixed_trial_args: list[str] | None = None,
) -> None:
    """Handle a stage failure: collect partial results, save state, exit.

    Handles retryable GCP errors, SweepStageError, and TimeoutError by
    attempting partial trial recovery, then saving state and exiting.
    For non-retryable errors, state is still saved before re-raising.
    """
    if isinstance(exc, (SweepStageError, TimeoutError)) or _is_retryable_gcp_error(exc):
        # Partial trial recovery
        if isinstance(exc, _SweepJobFailed) and exc.hpt_job is not None:
            partial_rows, _ = _try_collect_partial_from_job(
                exc.hpt_job,
                stage,
                species,
                bucket,
                partial_rows,
                resume_run,
            )
            if partial_rows:
                stage_data = {
                    "status": "partial",
                    "job_resource_name": getattr(exc.hpt_job, "resource_name", None),
                    "trial_rows": partial_rows,
                    "resume_run": resume_run,
                }
                if fixed_trial_args is not None:
                    stage_data["fixed_trial_args"] = fixed_trial_args
                sweep_state["stages"][state_stage_key] = stage_data

        logger.error(
            "Stage %d failed: %s. Saving progress — re-run the same command to resume.",
            stage,
            exc,
        )
        _save_sweep_state(sweep_state, species, algorithm, bucket=bucket, project=project)
        os._exit(1)

    # Non-retryable: save state so progress isn't lost, then re-raise
    _save_sweep_state(sweep_state, species, algorithm, bucket=bucket, project=project)
    raise


def _upload_results_to_gcs(
    csv_path: Path,
    species: str,
    bucket: str,
    project: str | None,
) -> None:
    """Upload CSV and sweep graphs to GCS (best-effort)."""
    try:
        from google.cloud import storage as _gcs

        _gcs_client = _gcs.Client(project=project)
        _gcs_bucket = _gcs_client.bucket(bucket)
        gcs_csv_path = f"sweeps/{species}/{csv_path.name}"
        _gcs_bucket.blob(gcs_csv_path).upload_from_filename(str(csv_path))
        logger.info("Sweep CSV uploaded to: gs://%s/%s", bucket, gcs_csv_path)

        for graph_name in ("sweep_trial_metrics.png", "sweep_hyperparameter_analysis.png"):
            graph_path = csv_path.parent / graph_name
            if graph_path.exists():
                gcs_graph_path = f"sweeps/{species}/{graph_name}"
                _gcs_bucket.blob(gcs_graph_path).upload_from_filename(str(graph_path))
                logger.info("Sweep graph uploaded to: gs://%s/%s", bucket, gcs_graph_path)
    except Exception as exc:
        logger.warning("Failed to upload sweep results to GCS: %s. Local copy at: %s", exc, csv_path)


# ── Dry-run summary ─────────────────────────────────────────────────────


def _print_dry_run_summary(
    *,
    species: str,
    algorithm: str,
    stages: list[int],
    search_spaces: dict[int, dict],
    timesteps: dict[int, int],
    trials: dict[int, int],
    parallel: dict[int, int],
    n_envs: dict[int, int],
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    image: str,
    bucket: str,
    seed: int,
) -> None:
    """Print resolved sweep configuration without submitting any jobs."""
    print("\n" + "=" * 60)
    print("DRY RUN — no jobs will be submitted")
    print("=" * 60)
    print(f"  Species:     {species}")
    print(f"  Algorithm:   {algorithm}")
    print(f"  Image:       {image}")
    print(f"  Bucket:      gs://{bucket}")
    print(f"  Machine:     {machine_type}")
    print(f"  Accelerator: {accelerator_type} x{accelerator_count}")
    print(f"  Seed:        {seed}")
    print()
    for s in stages:
        ss = search_spaces.get(s, {})
        print(f"  Stage {s}:")
        print(f"    Timesteps:  {timesteps[s]:,}")
        print(f"    Trials:     {trials[s]}  |  Parallel: {parallel[s]}  |  n_envs: {n_envs[s]}")
        print(f"    Search space ({len(ss)} params):")
        for param_id, spec in ss.items():
            kind = spec.get("type", "double")
            if kind == "double":
                print(f"      {param_id}: [{spec['min']}, {spec['max']}] scale={spec.get('scale', 'linear')}")
            elif kind == "discrete":
                print(f"      {param_id}: {spec['values']}")
            elif kind == "categorical":
                print(f"      {param_id}: {spec['values']} (categorical)")
        print()
    # Estimate rough cost
    total_trials = sum(trials[s] for s in stages)
    print(f"  Total trials across all stages: {total_trials}")
    print("=" * 60 + "\n")


def _resolve_credentials(project: str | None = None):
    """Resolve Google Cloud credentials using Application Default Credentials.

    Cloud Shell's metadata service can return malformed responses that break
    the default compute-engine credential flow.  Explicitly calling
    ``google.auth.default()`` uses ADC, which correctly picks up the
    user's ``gcloud auth`` session in Cloud Shell and service-account
    credentials on Vertex AI workers.

    The initial credential refresh is attempted eagerly so that transient
    metadata-server errors (e.g. the server returning a raw string instead
    of JSON) are caught and retried here rather than surfacing later as
    opaque gRPC ``AuthMetadataPlugin`` failures.

    Returns:
        A ``(credentials, project)`` tuple suitable for passing to
        ``aiplatform.init()``.
    """
    import google.auth
    import google.auth.transport.requests

    credentials, adc_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    # Eagerly refresh credentials so transient metadata-server errors
    # (TypeError from malformed JSON) are retried here instead of
    # propagating through the gRPC auth plugin at call time.
    _eager_refresh(credentials)

    resolved_project = project or adc_project
    if resolved_project is None:
        raise RuntimeError(
            "Could not determine GCP project.  Pass --project explicitly or "
            "set the GOOGLE_CLOUD_PROJECT / GCLOUD_PROJECT environment variable."
        )
    return credentials, resolved_project


def _eager_refresh(credentials, *, max_retries: int = 4, _request=None):
    """Refresh *credentials* with retries for transient metadata errors.

    The GCE metadata server can occasionally return a plain string instead
    of a JSON object, which causes ``google-auth`` to raise ``TypeError``
    (``string indices must be integers``).  Expired or revoked credentials
    raise ``RefreshError``, and network issues raise ``TransportError``.
    This helper retries all three with exponential back-off so the caller
    gets usable credentials.
    """
    # Build the tuple of retryable exception types.  google-auth may not be
    # installed in lightweight test environments, so we resolve dynamically.
    _retryable: tuple[type[BaseException], ...] = (TypeError,)
    try:
        import google.auth.exceptions as _gae

        _retryable = (TypeError, _gae.RefreshError, _gae.TransportError)
    except ImportError:
        pass

    if _request is None:
        import google.auth.transport.requests

        _request = google.auth.transport.requests.Request()
    delay = 1
    for attempt in range(1, max_retries + 1):
        try:
            credentials.refresh(_request)
            return
        except _retryable as exc:
            if attempt == max_retries:
                logger.error(
                    "Credential refresh failed after %d attempts: %s. "
                    "Try running 'gcloud auth application-default login' or "
                    "verify the service account has the required permissions.",
                    max_retries,
                    exc,
                )
                raise
            logger.warning(
                "Transient credential refresh error (attempt %d/%d): %s. Retrying in %ds …",
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
            delay *= 2


def _dedup_trial_rows(rows: list[dict]) -> list[dict]:
    """Deduplicate trial rows by ``trial_id``, keeping the last occurrence.

    When merging partial rows from multiple resume cycles, the same
    ``trial_id`` can appear more than once (e.g. if Vertex AI re-uses an
    ID across separate HPT jobs, or if rows are accidentally appended
    twice).  Later entries are assumed to have more complete data, so
    the last occurrence wins.
    """
    seen: dict[str, dict] = {}
    for row in rows:
        tid = row.get("trial_id")
        if tid is not None:
            seen[str(tid)] = row
        else:
            # Rows without a trial_id are kept unconditionally (shouldn't
            # happen in practice, but don't silently drop data).
            seen[f"_no_id_{id(row)}"] = row
    deduped = list(seen.values())
    if len(deduped) < len(rows):
        logger.info(
            "Deduplicated trial rows: %d → %d (removed %d duplicates)",
            len(rows),
            len(deduped),
            len(rows) - len(deduped),
        )
    return deduped


def launch_sweep(args: argparse.Namespace) -> None:
    """Submit a Vertex AI Hyperparameter Tuning job for a single stage.

    Each trial runs ``sweep.py trial`` inside the Docker container. The
    HPT service injects the trial's parameter values as additional CLI args.

    The job is submitted and then polled until completion (or ``--stage-timeout``
    is reached).  State is persisted to a local JSON file and GCS so the job
    can be reconnected to later by re-running the same command with ``--resume``.
    """
    try:
        from google.cloud import aiplatform
        from google.cloud.aiplatform import hyperparameter_tuning as hpt
    except ImportError:
        logger.error("google-cloud-aiplatform is not installed.\nInstall it with:  pip install google-cloud-aiplatform")
        sys.exit(1)

    credentials, project = _resolve_credentials(args.project)

    aiplatform.init(
        project=project,
        location=args.location,
        staging_bucket=f"gs://{args.bucket}",
        credentials=credentials,
    )

    stage = args.stage
    poll_interval = getattr(args, "poll_interval", 120)
    stage_timeout = getattr(args, "stage_timeout", None)
    resume = getattr(args, "resume", True)

    # Load search space: inline JSON > file > algorithm default
    resolved = _resolve_search_space(args.search_space, args.search_space_file, args.algorithm)
    search_space = _search_space_for_stage(resolved, stage)

    # Resolve job settings: CLI args (if set) > search-space file > hardcoded defaults.
    # The argparse defaults are None so we can detect when a flag was explicitly passed.
    file_settings = _settings_for_stage(resolved, stage)
    _HARDCODED_DEFAULTS = {"trials": 20, "timesteps": 500_000, "parallel": 5, "n_envs": 4}
    for key, hardcoded in _HARDCODED_DEFAULTS.items():
        cli_val = getattr(args, key)
        if cli_val is not None:
            continue  # CLI wins
        file_val = file_settings.get(key)
        if file_val is not None:
            logger.info("Using %s=%s from search-space file", key, file_val)
            setattr(args, key, file_val)
        else:
            setattr(args, key, hardcoded)

    # ── Dry run: print resolved config and exit ─────────────────────────
    if getattr(args, "dry_run", False):
        _print_dry_run_summary(
            species=args.species,
            algorithm=args.algorithm,
            stages=[stage],
            search_spaces={stage: search_space},
            timesteps={stage: args.timesteps},
            trials={stage: args.trials},
            parallel={stage: args.parallel},
            n_envs={stage: args.n_envs},
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            accelerator_count=args.accelerator_count,
            image=args.image,
            bucket=args.bucket,
            seed=getattr(args, "seed", 42),
        )
        return

    # ── State key used for single-stage launches ──────────────────────────
    # We store state under a stage-specific key so multiple single-stage
    # launches for different stages don't overwrite each other.
    state_stage_key = str(stage)

    # ── CLI args snapshot ─────────────────────────────────────────────────
    cli_args_snapshot: dict = {
        "mode": "launch",
        "stage": stage,
        "trials": args.trials,
        "parallel": args.parallel,
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "machine_type": args.machine_type,
        "accelerator_type": args.accelerator_type,
        "accelerator_count": args.accelerator_count,
        "image": args.image,
        "bucket": args.bucket,
        "project": args.project,
        "location": args.location,
    }

    sweep_state: dict = {
        "species": args.species,
        "algorithm": args.algorithm,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cli_args": cli_args_snapshot,
        "stages": {},
    }

    partial_rows: list[dict] = []
    resume_run = 0
    hpt_job = None
    job_resource_name = None
    reconnected = False

    # ── Resume: restore state from a previous run ─────────────────────────
    if resume:
        prev_state = _load_sweep_state(args.species, args.algorithm, bucket=args.bucket, project=args.project)
        if prev_state and prev_state.get("stages"):
            prev_stage_data = prev_state["stages"].get(state_stage_key)
            if prev_stage_data:
                status = prev_stage_data.get("status")

                if status == "completed":
                    logger.info(
                        "Stage %d already completed (best_reward=%.4f). Nothing to do. Use --no-resume to start fresh.",
                        stage,
                        prev_stage_data.get("best_mean_reward", 0),
                    )
                    return

                if status == "in_progress" and prev_stage_data.get("job_resource_name"):
                    prev_resource = prev_stage_data["job_resource_name"]
                    logger.info("Attempting to reconnect to previous job: %s", prev_resource)

                    prior_partial = prev_stage_data.get("prior_partial_rows", [])
                    if prior_partial:
                        partial_rows = list(prior_partial)
                        logger.info("Restored %d prior partial rows.", len(prior_partial))

                    ip_resume = prev_stage_data.get("resume_run", 0)
                    hpt_job, job_resource_name, reconnected, partial_rows, resume_run = _reconnect_or_collect_partial(
                        aiplatform,
                        prev_resource,
                        stage,
                        args.species,
                        args.bucket,
                        partial_rows,
                        ip_resume,
                        poll_interval,
                        stage_timeout,
                    )

                elif status == "partial":
                    partial_rows = [
                        r for r in prev_stage_data.get("trial_rows", []) if r.get("best_mean_reward") is not None
                    ]
                    if partial_rows:
                        resume_run = prev_stage_data.get("resume_run", 0) + 1
                        logger.info(
                            "Resuming stage %d: %d partial trials recovered from previous run.",
                            stage,
                            len(partial_rows),
                        )

            # Preserve state for other stages so we don't clobber them
            sweep_state = prev_state
            sweep_state["cli_args"] = cli_args_snapshot

    remaining_trials = args.trials - len(partial_rows)

    # ── Submit new job if needed ──────────────────────────────────────────
    try:
        if not reconnected and remaining_trials > 0:
            if partial_rows:
                logger.info(
                    "Stage %d: %d trials recovered, %d remaining.",
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
                timesteps=args.timesteps,
                n_envs=args.n_envs,
                trials=remaining_trials,
                parallel=args.parallel,
                bucket=args.bucket,
                image=args.image,
                machine_type=args.machine_type,
                accelerator_type=args.accelerator_type,
                accelerator_count=args.accelerator_count,
                search_space=search_space,
                load_path=args.load,
                wandb=args.wandb,
                eval_freq=getattr(args, "eval_freq", None),
                save_freq=getattr(args, "save_freq", None),
                resume_run=resume_run,
                restart_job_on_worker_restart=getattr(args, "restart_job_on_worker_restart", False),
                no_tensorboard=getattr(args, "no_tensorboard", False),
                seed=getattr(args, "seed", 42),
            )
            job_resource_name = getattr(hpt_job, "resource_name", None)

            # Save in-progress state immediately so the job can be
            # reconnected if the process is interrupted.
            sweep_state["stages"][state_stage_key] = {
                "status": "in_progress",
                "job_resource_name": job_resource_name,
                "load_path": args.load,
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

            # Poll until the job completes (--stage-timeout 0 means fire-and-forget)
            if stage_timeout == 0:
                logger.info(
                    "Job submitted (--stage-timeout 0). Re-run with --resume to check results.\n  Resource: %s",
                    job_resource_name,
                )
                os._exit(0)

            hpt_job = _wait_for_job(
                hpt_job,
                aiplatform,
                poll_interval=poll_interval,
                timeout=stage_timeout,
            )

        # ── Collect results ───────────────────────────────────────────────
        if hpt_job is not None:
            new_rows = _collect_and_tag_rows(hpt_job, stage, args.species, args.bucket, resume_run)
            stage_rows = _dedup_trial_rows(partial_rows + new_rows)
        elif remaining_trials <= 0:
            logger.info(
                "Stage %d: all %d trials already completed in previous run.",
                stage,
                len(partial_rows),
            )
            stage_rows = list(partial_rows)
        else:
            stage_rows = list(partial_rows)

        # Identify the best trial
        best_row: dict | None = None
        best_model_path: str | None = None
        try:
            best_model_path, best_row = _best_trial_model_path(stage_rows, args.bucket, args.species, stage)
        except SweepStageError:
            # Fall back to best trial regardless of gate status
            try:
                best_model_path, best_row = _best_trial_model_path_any(stage_rows, args.bucket, args.species, stage)
            except SweepStageError:
                logger.warning("No trials reported a valid reward for stage %d.", stage)

        if best_row is not None:
            logger.info(
                "Stage %d best trial: id=%s  reward=%.4f  passed=%s",
                stage,
                best_row["trial_id"],
                best_row.get("best_mean_reward", 0),
                best_row.get("stage_passed", False),
            )

        # Save completed state
        sweep_state["stages"][state_stage_key] = {
            "status": "completed",
            "job_resource_name": job_resource_name,
            "best_trial_id": best_row["trial_id"] if best_row else None,
            "best_mean_reward": best_row.get("best_mean_reward") if best_row else None,
            "best_model_path": best_model_path,
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

        # Apply quality scoring before writing CSV
        from .scoring import compute_quality_scores

        compute_quality_scores(stage_rows, stage)

        # Write results CSV and upload
        csv_path = Path(f"sweep_results_{args.species}_{args.algorithm}_stage{stage}.csv")
        write_results_csv(stage_rows, csv_path)
        plot_sweep_results(csv_path, args.species, args.algorithm)
        _upload_results_to_gcs(csv_path, args.species, args.bucket, args.project)

        logger.info("Stage %d sweep complete.", stage)

    except Exception as exc:
        _handle_stage_failure(
            exc,
            stage,
            args.species,
            args.algorithm,
            args.bucket,
            args.project,
            sweep_state,
            state_stage_key,
            partial_rows,
            resume_run,
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

    credentials, project = _resolve_credentials(args.project)

    aiplatform.init(
        project=project,
        location=args.location,
        staging_bucket=f"gs://{args.bucket}",
        credentials=credentials,
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

    # ── Dry run: resolve all settings, print summary, exit ──────────────
    if getattr(args, "dry_run", False):
        all_search_spaces: dict[int, dict] = {}
        all_timesteps: dict[int, int] = {}
        all_trials: dict[int, int] = {}
        all_parallel: dict[int, int] = {}
        all_n_envs: dict[int, int] = {}
        for s in range(1, 4):
            all_search_spaces[s] = _search_space_for_stage(resolved, s)
            fs = _settings_for_stage(resolved, s)
            all_timesteps[s] = cli_timesteps[s - 1] or fs.get("timesteps") or [500_000, 1_000_000, 1_500_000][s - 1]
            all_trials[s] = cli_trials[s - 1] or fs.get("trials", args.trials)
            all_parallel[s] = cli_parallel[s - 1] or fs.get("parallel", args.parallel)
            all_n_envs[s] = fs.get("n_envs", args.n_envs)
        _print_dry_run_summary(
            species=args.species,
            algorithm=args.algorithm,
            stages=[1, 2, 3],
            search_spaces=all_search_spaces,
            timesteps=all_timesteps,
            trials=all_trials,
            parallel=all_parallel,
            n_envs=all_n_envs,
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            accelerator_count=args.accelerator_count,
            image=args.image,
            bucket=args.bucket,
            seed=getattr(args, "seed", 42),
        )
        return

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
            timesteps = file_settings.get("timesteps")
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

                prior_partial = in_progress_data.get("prior_partial_rows", [])
                if prior_partial:
                    partial_rows = _dedup_trial_rows(partial_rows + prior_partial)
                    remaining_trials = trials - len(partial_rows)
                    logger.info(
                        "Restored %d prior partial rows from earlier runs.",
                        len(prior_partial),
                    )

                ip_resume = in_progress_data.get("resume_run", 0)
                hpt_job, job_resource_name, reconnected, partial_rows, resume_run = _reconnect_or_collect_partial(
                    aiplatform,
                    prev_resource,
                    stage,
                    args.species,
                    args.bucket,
                    partial_rows,
                    ip_resume,
                    poll_interval,
                    stage_timeout,
                )
                remaining_trials = trials - len(partial_rows)

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
                    eval_freq=getattr(args, "eval_freq", None),
                    save_freq=getattr(args, "save_freq", None),
                    resume_run=resume_run,
                    restart_job_on_worker_restart=getattr(args, "restart_job_on_worker_restart", False),
                    no_tensorboard=getattr(args, "no_tensorboard", False),
                    seed=getattr(args, "seed", 42),
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

                # Poll until the job completes (--stage-timeout 0 means fire-and-forget)
                if stage_timeout == 0:
                    logger.info(
                        "Stage %d job submitted (--stage-timeout 0). Re-run with --resume to check results.\n"
                        "  Resource: %s",
                        stage,
                        job_resource_name,
                    )
                    os._exit(0)

                hpt_job = _wait_for_job(
                    hpt_job,
                    aiplatform,
                    poll_interval=poll_interval,
                    timeout=stage_timeout,
                )

            # ── Collect results ────────────────────────────────────────────
            if hpt_job is not None:
                new_rows = _collect_and_tag_rows(hpt_job, stage, args.species, args.bucket, resume_run)
                stage_rows = _dedup_trial_rows(partial_rows + new_rows)
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
                "best_model_path": best_model_path,
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
            _handle_stage_failure(
                exc,
                stage,
                args.species,
                args.algorithm,
                args.bucket,
                args.project,
                sweep_state,
                str(stage),
                partial_rows,
                resume_run,
                fixed_trial_args=fixed_trial_args,
            )

    # Apply quality scoring per stage before writing combined CSV
    from .scoring import compute_quality_scores

    stages_in_rows = sorted({r["stage"] for r in all_rows})
    for s in stages_in_rows:
        stage_subset = [r for r in all_rows if r["stage"] == s]
        compute_quality_scores(stage_subset, s)

    # Write a combined CSV of every trial across all three stages
    csv_path = Path(f"sweep_results_{args.species}_{args.algorithm}.csv")
    write_results_csv(all_rows, csv_path)
    plot_sweep_results(csv_path, args.species, args.algorithm)
    _upload_results_to_gcs(csv_path, args.species, args.bucket, args.project)

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
