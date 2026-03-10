"""Vertex AI HPT job submission with retry logic."""

import logging
import sys
import time

from .constants import _SweepJobFailed
from .trial import _build_parameter_spec

logger = logging.getLogger(__name__)

# Common short names → full Vertex AI accelerator enum labels.
_ACCELERATOR_ALIASES: dict[str, str] = {
    "T4": "NVIDIA_TESLA_T4",
    "V100": "NVIDIA_TESLA_V100",
    "P100": "NVIDIA_TESLA_P100",
    "A100": "NVIDIA_TESLA_A100",
    "L4": "NVIDIA_L4",
}

# Machine-type families supported by Vertex AI custom training jobs.
# See https://docs.cloud.google.com/vertex-ai/docs/training/configure-compute
_SUPPORTED_MACHINE_PREFIXES = (
    "n1-",
    "n2-",
    "e2-",
    "c2-",
    "m1-",
    "a2-",
    "a3-",
    "g2-",
)

# Machine-type families that support attaching GPU accelerators.
_GPU_COMPATIBLE_PREFIXES = (
    "n1-",
    "a2-",
    "a3-",
    "g2-",
)


def _validate_machine_type(machine_type: str, accelerator_type: str | None = None) -> None:
    """Raise ``ValueError`` early if the machine type is unsupported.

    When *accelerator_type* is not ``None``, also checks that the machine
    family supports GPU attachment (e.g. ``c2-`` and ``e2-`` do not).
    """
    if not any(machine_type.startswith(p) for p in _SUPPORTED_MACHINE_PREFIXES):
        raise ValueError(
            f'Machine type "{machine_type}" is not supported by Vertex AI custom training. '
            f"Supported families: {', '.join(p.rstrip('-') for p in _SUPPORTED_MACHINE_PREFIXES)}. "
            f'The default is "n1-standard-8".'
        )
    if accelerator_type is not None and not any(machine_type.startswith(p) for p in _GPU_COMPATIBLE_PREFIXES):
        raise ValueError(
            f'Accelerator "{accelerator_type}" is not supported for machine type '
            f'"{machine_type}". Use a GPU-compatible machine family '
            f"({', '.join(p.rstrip('-') for p in _GPU_COMPATIBLE_PREFIXES)}) "
            f'or set --accelerator-type=None for CPU-only. The default is "n1-standard-8".'
        )


def _normalize_accelerator_type(raw: str) -> str | None:
    """Return the canonical Vertex AI enum label, or *None* for CPU-only."""
    if raw.lower() == "none":
        return None
    upper = raw.upper()
    return _ACCELERATOR_ALIASES.get(upper, raw)


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


def _state_name(state) -> str:
    """Return the string name of a Vertex AI job state enum value."""
    return state.name if hasattr(state, "name") else str(state)


def _wait_for_job(
    hpt_job,
    aiplatform,
    *,
    poll_interval: int = 120,
    timeout: float | None = None,
):
    """Poll a running HPT job until completion with progress logging.

    Periodically refreshes the job state and logs trial progress, providing
    visibility into multi-hour sweeps.  When *timeout* is set, raises
    ``TimeoutError`` if the job hasn't finished within the given number of
    seconds — the job continues running in the cloud and can be resumed.

    Args:
        hpt_job: A submitted ``HyperparameterTuningJob`` with a valid
            ``resource_name``.
        aiplatform: The ``google.cloud.aiplatform`` module, used to refresh
            the job via ``HyperparameterTuningJob.get()``.
        poll_interval: Seconds between status checks (default 120).
        timeout: Maximum wall-clock seconds to wait.  ``None`` means wait
            indefinitely.

    Returns:
        The refreshed job object with final trial data.

    Raises:
        _SweepJobFailed: If the job ends in a failed or cancelled state.
        TimeoutError: If *timeout* is exceeded while the job is still running.
    """
    resource_name = hpt_job.resource_name
    start = time.monotonic()

    # timeout=0 means fire-and-forget — skip polling entirely.
    if timeout == 0:
        raise TimeoutError(
            f"Job {resource_name} submitted (--stage-timeout 0). "
            f"The job is running in the cloud — re-run with --resume to reconnect."
        )

    logger.info(
        "Waiting for job %s (poll every %ds, timeout %s) …",
        resource_name,
        poll_interval,
        f"{timeout / 3600:.1f}h" if timeout else "none",
    )

    # Check initial state — important for reconnection where the job may
    # already be finished.
    initial_state = _state_name(hpt_job.state)
    if "SUCCEEDED" in initial_state:
        logger.info("Job already completed successfully.")
        return hpt_job
    if "FAILED" in initial_state or "CANCELLED" in initial_state:
        raise _SweepJobFailed(
            f"Job {resource_name} already in terminal state {initial_state}",
            hpt_job=hpt_job,
        )

    while True:
        time.sleep(poll_interval)
        elapsed = time.monotonic() - start
        elapsed_min = elapsed / 60

        # Refresh job state from Vertex AI
        try:
            refreshed = aiplatform.HyperparameterTuningJob.get(resource_name)
        except Exception as exc:
            logger.warning("Failed to refresh job state (will retry): %s", exc)
            continue

        state = _state_name(refreshed.state)

        # Count trial progress
        trials = getattr(refreshed, "trials", None) or []
        n_total = len(trials)
        n_completed = sum(1 for t in trials if t.final_measurement)

        # Find best reward so far
        best_so_far: float | None = None
        for t in trials:
            if t.final_measurement and t.final_measurement.metrics:
                for m in t.final_measurement.metrics:
                    if m.metric_id == "best_mean_reward":
                        if best_so_far is None or m.value > best_so_far:
                            best_so_far = m.value

        reward_str = f"  best_reward={best_so_far:.4f}" if best_so_far is not None else ""
        logger.info(
            "  [%.0f min] state=%s | trials: %d/%d completed%s",
            elapsed_min,
            state,
            n_completed,
            n_total,
            reward_str,
        )

        # Check terminal state
        if "SUCCEEDED" in state:
            logger.info("Job completed successfully after %.1f min.", elapsed_min)
            return refreshed
        if "FAILED" in state or "CANCELLED" in state:
            raise _SweepJobFailed(
                f"Job {resource_name} ended with state {state} after {elapsed_min:.1f} min",
                hpt_job=refreshed,
            )

        # Check timeout
        if timeout is not None and elapsed > timeout:
            raise TimeoutError(
                f"Job {resource_name} timed out after {elapsed_min:.1f} min "
                f"(limit: {timeout / 60:.1f} min). The job is still running in "
                f"the cloud — re-run with --resume to reconnect."
            )


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
    eval_freq: int | None = None,
    save_freq: int | None = None,
    resume_run: int = 0,
    restart_job_on_worker_restart: bool = False,
    no_tensorboard: bool = False,
    seed: int = 42,
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
        restart_job_on_worker_restart: If ``True``, Vertex AI will
            automatically restart the job when a worker is preempted
            (e.g. spot/preemptible VMs).
        no_tensorboard: If ``True``, disable TensorBoard logging in each
            trial to reduce disk I/O and storage usage.
        seed: Random seed passed to each trial (default 42).
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
        "--seed",
        str(seed),
        "--output-dir",
        output_base,
    ]
    if eval_freq is not None:
        trial_args += ["--eval-freq", str(eval_freq)]
    if save_freq is not None:
        trial_args += ["--save-freq", str(save_freq)]
    if load_path:
        trial_args += ["--load", load_path]
    if fixed_trial_args:
        trial_args += fixed_trial_args
    if wandb:
        trial_args.append("--wandb")
    if no_tensorboard:
        trial_args.append("--no-tensorboard")

    resolved_accelerator = _normalize_accelerator_type(accelerator_type)
    _validate_machine_type(machine_type, resolved_accelerator)

    machine_spec: dict = {"machine_type": machine_type}
    if resolved_accelerator is not None:
        machine_spec["accelerator_type"] = resolved_accelerator
        machine_spec["accelerator_count"] = accelerator_count

    worker_pool_specs = [
        {
            "machine_spec": machine_spec,
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
    if restart_job_on_worker_restart:
        logger.info("  restart_job_on_worker_restart: enabled")

    custom_job = aiplatform.CustomJob(
        display_name=f"{display_name}-trial",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=f"gs://{bucket}/sweeps/{species}/stage{stage}{'_r' + str(resume_run) if resume_run else ''}",
    )

    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=display_name,
        custom_job=custom_job,
        metric_spec={"best_mean_reward": "maximize"},
        parameter_spec=parameter_spec,
        max_trial_count=trials,
        parallel_trial_count=parallel,
    )

    # Submit the job in our own background thread so we always have a
    # future we can inspect for errors.  The SDK's sync=False uses an
    # internal thread-pool whose future attribute name varies across
    # versions, making error detection unreliable.  By running
    # sync=True in our own ThreadPoolExecutor we get:
    #   a) Immediate visibility into creation errors (permission denied, etc.)
    #   b) The SDK handles the API call synchronously inside the thread.
    # We only need the job to be *created* — our _wait_for_job handles
    # the long-running poll with better logging.
    import concurrent.futures

    _RETRY_DELAYS = [60, 180, 300]  # seconds between retries
    _CREATION_TIMEOUT = 300
    for attempt in range(len(_RETRY_DELAYS) + 1):
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            _run_kwargs: dict = {"sync": True}
            if restart_job_on_worker_restart:
                _run_kwargs["restart_job_on_worker_restart"] = True
            _future = _executor.submit(lambda: hpt_job.run(**_run_kwargs))

            # Poll until the job resource is created (resource_name becomes
            # available) or the future raises an error.
            for _tick in range(_CREATION_TIMEOUT):
                # If the background thread failed, surface the real error.
                if _future.done():
                    _future.result()  # re-raises any exception

                try:
                    _ = hpt_job.resource_name
                    break  # resource created successfully
                except RuntimeError:
                    time.sleep(1)
            else:
                # Final check — if the thread finished with an error,
                # surface it instead of a generic timeout message.
                if _future.done():
                    _future.result()
                raise RuntimeError(
                    f"Job '{display_name}' resource was not created within "
                    f"{_CREATION_TIMEOUT}s. Check Vertex AI permissions, "
                    f"project configuration, and that the service account "
                    f"has roles/aiplatform.user."
                )

            # Job created — shut down executor without waiting for the
            # SDK's completion poll (our _wait_for_job replaces it).
            _executor.shutdown(wait=False)

            break  # success — exit retry loop

        except Exception as exc:
            _executor.shutdown(wait=False)

            # Only retry on transient / quota errors from the Google API.
            if not _is_retryable_gcp_error(exc) or attempt >= len(_RETRY_DELAYS):
                raise _SweepJobFailed(str(exc), hpt_job=hpt_job) from exc
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

            # Re-create job objects for the retry — the SDK marks them as
            # used after a run() call.
            custom_job = aiplatform.CustomJob(
                display_name=f"{display_name}-trial",
                worker_pool_specs=worker_pool_specs,
                base_output_dir=f"gs://{bucket}/sweeps/{species}/stage{stage}{'_r' + str(resume_run) if resume_run else ''}",
            )
            hpt_job = aiplatform.HyperparameterTuningJob(
                display_name=display_name,
                custom_job=custom_job,
                metric_spec={"best_mean_reward": "maximize"},
                parameter_spec=parameter_spec,
                max_trial_count=trials,
                parallel_trial_count=parallel,
            )

    job_name = hpt_job.resource_name
    logger.info("Job submitted: %s", job_name)
    logger.info("Monitor at: https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs")
    logger.info("Results will be written to: gs://%s/sweeps/%s/stage%d/", bucket, species, stage)
    return hpt_job
