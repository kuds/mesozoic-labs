"""Sweep state persistence — save/load/resume to local disk and GCS."""

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _sweep_state_local_path(species: str, algorithm: str) -> Path:
    """Return the local path for a sweep state file."""
    return Path(f"sweep_state_{species}_{algorithm}.json")


def _save_sweep_state(
    state: dict,
    species: str,
    algorithm: str,
    bucket: str | None = None,
    project: str | None = None,
) -> None:
    """Persist sweep progress to local disk and (best-effort) GCS.

    The state dict is written as JSON to a local file and, if *bucket* is
    provided, also uploaded to
    ``gs://<bucket>/sweeps/<species>/_sweep_state.json``.
    """
    local_path = _sweep_state_local_path(species, algorithm)
    # Write to a temp file then atomically rename to avoid corruption
    # if the process is killed mid-write (e.g. Ctrl+C, OOM, SIGKILL).
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=local_path.parent,
            prefix=f".{local_path.name}.",
            suffix=".tmp",
        )
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, local_path)
    except OSError:
        # Fallback: direct write (e.g. if the filesystem doesn't
        # support atomic rename from the temp directory).
        local_path.write_text(json.dumps(state, indent=2))
    logger.info("Sweep state saved locally: %s", local_path)

    if bucket:
        gcs_key = f"sweeps/{species}/_sweep_state_{algorithm}.json"
        try:
            from google.cloud import storage as _gcs

            _client = _gcs.Client(project=project)
            _bkt = _client.bucket(bucket)
            _bkt.blob(gcs_key).upload_from_string(json.dumps(state, indent=2))
            logger.info("Sweep state uploaded to: gs://%s/%s", bucket, gcs_key)
        except Exception as exc:
            logger.warning("Failed to upload sweep state to GCS: %s", exc)


def _load_sweep_state(
    species: str,
    algorithm: str,
    bucket: str | None = None,
    project: str | None = None,
) -> dict | None:
    """Load a previously saved sweep state file.

    Checks GCS first (if *bucket* is provided), then falls back to the
    local file.  Returns ``None`` if no state exists or if the state does
    not match *species* and *algorithm*.
    """
    state: dict | None = None

    # Try GCS first
    if bucket:
        gcs_key = f"sweeps/{species}/_sweep_state_{algorithm}.json"
        try:
            from google.cloud import storage as _gcs

            _client = _gcs.Client(project=project)
            _bkt = _client.bucket(bucket)
            blob = _bkt.blob(gcs_key)
            if blob.exists():
                state = json.loads(blob.download_as_text())
                logger.info("Loaded sweep state from: gs://%s/%s", bucket, gcs_key)
        except Exception as exc:
            logger.warning("Could not read sweep state from GCS: %s", exc)

    # Fall back to local file
    if state is None:
        local_path = _sweep_state_local_path(species, algorithm)
        if local_path.exists():
            try:
                state = json.loads(local_path.read_text())
                logger.info("Loaded sweep state from: %s", local_path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read local sweep state: %s", exc)

    # Validate species + algorithm match
    if state is not None:
        if state.get("species") != species or state.get("algorithm") != algorithm:
            logger.warning(
                "Sweep state species/algorithm (%s/%s) does not match current (%s/%s) — ignoring",
                state.get("species"),
                state.get("algorithm"),
                species,
                algorithm,
            )
            return None

    return state
