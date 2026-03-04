"""Vertex AI HPT job submission with retry logic."""

import logging
import sys
import time
from typing import Any

from .constants import _SweepJobFailed
from .trial import _build_parameter_spec

logger = logging.getLogger(__name__)


def _is_retryable_gcp_error(exc: Exception) -> bool:
    """Return ``True`` if *exc* is a transient GCP error worth retrying.

    Matches by class name so we don't need ``google-cloud-aiplatform``
    installed at import time (the exceptions live in ``google.api_core``).
    """
    retryable_names = {
        "ResourceExhausted",
        "ServiceUnavailable",
        "GoogleAPICallError",
        "TooManyRequests",
        "InternalServerError",
        "GatewayTimeout",
    }
    return type(exc).__name__ in retryable_names


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
    fixed_trial_args: list[str] | None = None,
    wandb: bool = False,
    sync: bool = False,
    resume_run: int = 0,
):
    """Build and submit a single-stage HPT job. Returns the job object.

    Args:
        fixed_trial_args: Extra CLI args appended verbatim to every trial's
            command line.  Used to inject hyperparameters that are *not* part
            of the search space but must match a prior stage's winning trial
            (e.g. ``["--ppo_net_arch", "medium"]``).
        resume_run: When resuming a partially completed stage, set to a
            positive integer so the new job writes to a separate GCS
            sub-directory (``stage{N}_r{resume_run}``) to avoid overwriting
            checkpoints from previous runs.
    """
    parameter_spec = _build_parameter_spec(search_space, hpt_module)
    if not parameter_spec:
        logger.error("No valid parameters in search space for stage %d — aborting", stage)
        sys.exit(1)

    output_base = f"/gcs/{bucket}/sweeps/{species}/stage{stage}"
    if resume_run:
        output_base = f"{output_base}_r{resume_run}"

    trial_args = [
        "-m",
        "environments.shared.scripts.sweep",
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
    if fixed_trial_args:
        trial_args += fixed_trial_args
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

    custom_job = aiplatform.CustomJob(
        display_name=f"{display_name}-trial",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=f"gs://{bucket}/sweeps/{species}/stage{stage}",
    )

    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=display_name,
        custom_job=custom_job,
        metric_spec={"best_mean_reward": "maximize"},
        parameter_spec=parameter_spec,
        max_trial_count=trials,
        parallel_trial_count=parallel,
    )

    # Retry loop for transient Vertex AI / quota errors.
    _RETRY_DELAYS = [60, 180, 300]  # seconds between retries
    last_exc: Exception | None = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            hpt_job.run(sync=sync)
            break  # success
        except Exception as exc:
            # Only retry on transient / quota errors from the Google API.
            _retryable = _is_retryable_gcp_error(exc)
            if not _retryable or attempt >= len(_RETRY_DELAYS):
                raise _SweepJobFailed(str(exc), hpt_job=hpt_job) from exc
            last_exc = exc
            delay = _RETRY_DELAYS[attempt]
            logger.warning(
                "Vertex AI error on attempt %d/%d for stage %d: %s. Retrying in %ds …",
                attempt + 1,
                len(_RETRY_DELAYS) + 1,
                stage,
                exc,
                delay,
            )
            time.sleep(delay)
    else:
        # All retries exhausted — should not reach here because we re-raise
        # above, but guard defensively.
        raise _SweepJobFailed(
            f"Job submission for stage {stage} failed after {len(_RETRY_DELAYS) + 1} attempts",
            hpt_job=hpt_job,
        ) from last_exc

    logger.info("Job submitted: %s", hpt_job.resource_name)
    logger.info("Monitor at: https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs")
    logger.info("Results will be written to: gs://%s/sweeps/%s/stage%d/", bucket, species, stage)
    return hpt_job
