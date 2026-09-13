"""Run the complete prepared balance study with one resumable invocation.

The notebook owns historical references and calibration. This module runs four
short integration probes, then the matched twelve-run training comparison.
Completed jobs are trusted only after the trainer validates their saved bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from environments.compsognathus.scripts.train_balance_study import (
    _load_plan,
    summarize_study,
    train_balance_arm,
)

SUITE_SCHEMA = "compsognathus.balance-suite/v1"
ARMS = ("A", "B", "C", "D")
TRAINING_SEEDS = (42, 43, 44)
PROBE_UPDATES = 2
PROBE_EVALUATION_EPISODES = 4


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plan_digest(plan: dict) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _jobs(plan: dict, mode: str, *, strict_full: bool = True) -> list[dict]:
    if set(plan["arms"]) != set(ARMS):
        raise ValueError("The balance suite requires exactly arms A, B, C and D")
    seeds = set(plan["training_seeds"])
    if 42 not in seeds or (mode == "full" and strict_full and seeds != set(TRAINING_SEEDS)):
        raise ValueError("Full suites require training seeds 42, 43 and 44; smoke requires seed 42")
    rollout = plan["n_envs"] * plan["stage_config"]["ppo_kwargs"]["n_steps"]
    full_steps = math.ceil(plan["training_budget"] / rollout) * rollout
    jobs = [
        {
            "name": f"{arm}_seed42_probe{PROBE_UPDATES}",
            "phase": "smoke",
            "arm": arm,
            "seed": 42,
            "probe_updates": PROBE_UPDATES,
            "evaluation_episodes": PROBE_EVALUATION_EPISODES,
            "planned_training_steps": PROBE_UPDATES * rollout,
            "status": "pending",
        }
        for arm in ARMS
    ]
    if mode == "full":
        jobs.extend(
            {
                "name": f"{arm}_seed{seed}",
                "phase": "full",
                "arm": arm,
                "seed": seed,
                "probe_updates": None,
                "evaluation_episodes": None,
                "planned_training_steps": full_steps,
                "status": "pending",
            }
            for arm in ARMS
            for seed in (TRAINING_SEEDS if strict_full else plan["training_seeds"])
        )
    return jobs


def _finished(job: dict) -> bool:
    return job["status"] == "complete" or (
        job["status"] == "skipped" and job.get("skip_reason") == "validated_completed_run"
    )


def _next_job(jobs: list[dict]) -> dict | None:
    probes = [job for job in jobs if job["phase"] == "smoke" and job["status"] != "complete"]
    if probes:
        return dict(probes[0])
    remaining = [job for job in jobs if job["phase"] == "full" and job["status"] != "complete"]
    if not remaining:
        return None
    priority = {"failed": 0, "resumable": 1, "unfinished": 2, "not_started": 3}
    return dict(min(remaining, key=lambda job: priority[job["status"]]))


def _inspect(output: Path, plan: dict, validation_cache: dict[str, dict]) -> dict:
    """Inspect disk state; a caller-local cache avoids duplicate verification."""
    jobs = _jobs(plan, "full", strict_full=False)
    for job in jobs:
        directory = output / "runs" / job["name"]
        if (directory / "run_summary.json").is_file():
            if job["name"] not in validation_cache:
                kwargs: dict[str, Any] = {"resume": True}
                if job["phase"] == "smoke":
                    kwargs.update(probe_updates=PROBE_UPDATES, evaluation_episodes=PROBE_EVALUATION_EPISODES)
                try:
                    result = train_balance_arm(output, job["arm"], job["seed"], **kwargs)
                    if result.get("status") == "paused":
                        raise ValueError("Declared-complete job returned an unfinished result")
                    validation_cache[job["name"]] = {"status": "complete", "result": result}
                except Exception as exc:
                    validation_cache[job["name"]] = {
                        "status": "failed",
                        "invalid_completed_bundle": True,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
            job.update(validation_cache[job["name"]])
        elif directory.is_dir():
            committed = any((directory / "checkpoints").glob("step_*.manifest.json")) or any(
                (directory / "continuations").glob("step_*.manifest.json")
            )
            job["status"] = "resumable" if committed else "unfinished"
        else:
            job["status"] = "not_started"
    probes = [job for job in jobs if job["phase"] == "smoke"]
    full = [job for job in jobs if job["phase"] == "full"]
    completed_probes = sum(job["status"] == "complete" for job in probes)
    completed_full = sum(job["status"] == "complete" for job in full)
    return {
        "schema": SUITE_SCHEMA,
        "study_plan_sha256": _plan_digest(plan),
        "jobs": jobs,
        "completed_probe_runs": completed_probes,
        "expected_probe_runs": len(probes),
        "remaining_probe_runs": len(probes) - completed_probes,
        "completed_full_runs": completed_full,
        "expected_full_runs": len(full),
        "remaining_full_runs": len(full) - completed_full,
        "resumable_full_runs": sum(job["status"] == "resumable" for job in full),
        "unfinished_full_runs": sum(job["status"] == "unfinished" for job in full),
        "invalid_completed_runs": sum(job["status"] == "failed" for job in jobs),
        "next_job": _next_job(jobs),
        "all_jobs_complete": all(job["status"] == "complete" for job in jobs),
        "read_only": True,
    }


def inspect_balance_suite(output: str | Path) -> dict:
    """Read startup inventory and recommend the next probe or unfinished run.

    The standard three-seed plan has four probes and twelve full runs. An
    explicitly smaller smoke-only plan inventories its declared seeds only.
    Existing summaries are validated through the trainer's read-only completed
    path. Missing runs are never passed to the trainer, and no file is written.
    Both screened checkpoints and continuation-only checkpoints count toward
    resumability; the trainer verifies their integrity when resuming them.
    """
    output = Path(output)
    return _inspect(output, _load_plan(output), {})


def _update_totals(progress: dict) -> None:
    jobs = progress["jobs"]
    progress["completed_jobs"] = sum(_finished(job) for job in jobs)
    progress["failed_jobs"] = sum(job["status"] == "failed" for job in jobs)
    progress["prerequisite_blocked_jobs"] = sum(job.get("skip_reason") == "prerequisite_failed" for job in jobs)
    progress["all_jobs_complete"] = all(_finished(job) for job in jobs)
    progress["completed_training_steps"] = sum(job["planned_training_steps"] for job in jobs if _finished(job))
    seconds = [
        job["result"].get("training_seconds") for job in jobs if _finished(job) and isinstance(job.get("result"), dict)
    ]
    progress["recorded_training_seconds"] = sum(
        value for value in seconds if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
    )
    progress["updated_at"] = _timestamp()


def run_balance_suite(output: str | Path, *, mode: str = "session", session_hours: float | None = None) -> dict:
    """Run all prepared jobs, validating completed output on every invocation.

    ``session`` completes missing probes and advances one unfinished full run,
    preferring an existing run over a new one. Sessions have no time limit by
    default. An explicitly supplied ``session_hours`` cap includes startup
    validation and probes; the trainer checkpoints when that cap pauses it.
    ``full`` runs A–D probes (two PPO updates and four evaluation episodes at
    seed 42), then A–D full training at seeds 42/43/44. ``smoke`` runs only the
    probes. Every job delegates completion validation/resume to the trainer.

    A probe error blocks the full sweep. A full-run error is recorded while
    the remaining runs continue. KeyboardInterrupt records the current job
    and propagates, so Run all can resume from verified saved work later.
    """
    if mode not in ("session", "full", "smoke"):
        raise ValueError("mode must be 'session', 'full' or 'smoke'")
    if session_hours is not None and (
        isinstance(session_hours, bool)
        or not isinstance(session_hours, (int, float))
        or not math.isfinite(session_hours)
        or session_hours <= 0
        or not math.isfinite(float(session_hours) * 3600.0)
    ):
        raise ValueError("session_hours must be finite and positive")
    session_seconds = float(session_hours) * 3600.0 if session_hours is not None and mode == "session" else None
    started = time.monotonic() if session_seconds is not None else 0.0
    output = Path(output)
    plan = _load_plan(output)
    digest = _plan_digest(plan)
    progress_path = output / "suite_progress.json"
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else None
    if previous is not None and (previous.get("schema") != SUITE_SCHEMA or previous.get("study_plan_sha256") != digest):
        raise ValueError("Saved suite progress belongs to a different study; use a matching prepared output")
    validation_cache: dict[str, dict] = {}
    inventory_before = _inspect(output, plan, validation_cache)
    print(
        json.dumps(
            {
                "event": "suite_inventory",
                "completed_full_runs": inventory_before["completed_full_runs"],
                "remaining_full_runs": inventory_before["remaining_full_runs"],
                "resumable_full_runs": inventory_before["resumable_full_runs"],
                "invalid_completed_jobs": [
                    {"name": job["name"], "error": job["error"]}
                    for job in inventory_before["jobs"]
                    if job["status"] == "failed"
                ],
                "next_job": inventory_before["next_job"]["name"] if inventory_before["next_job"] else None,
            }
        ),
        flush=True,
    )
    jobs = _jobs(plan, "smoke" if mode == "smoke" else "full")
    for job in jobs:
        cached = validation_cache.get(job["name"])
        if cached and cached["status"] == "complete":
            job.update(cached, status="skipped", skip_reason="validated_completed_run", completion_validated=True)
        elif cached:
            job.update(cached)
    previous_jobs = {job["name"]: job for job in previous.get("jobs", [])} if previous else {}
    progress = {
        "schema": SUITE_SCHEMA,
        "study_plan_sha256": digest,
        "implementation_sha256": plan.get("implementation_sha256"),
        "mode": mode,
        "session_hours": session_hours if mode == "session" else None,
        "status": "running",
        "started_at": previous.get("started_at", _timestamp()) if previous else _timestamp(),
        "invocations": previous.get("invocations", 0) + 1 if previous else 1,
        "research_only": True,
        "production_advancement": False,
        "planned_training_steps": sum(job["planned_training_steps"] for job in jobs),
        "planned_probe_training_steps": sum(job["planned_training_steps"] for job in jobs if job["phase"] == "smoke"),
        "planned_full_training_steps": sum(job["planned_training_steps"] for job in jobs if job["phase"] == "full"),
        "jobs": jobs,
        "inventory_before": inventory_before,
    }

    def checkpoint() -> None:
        _update_totals(progress)
        _save(progress_path, progress)

    checkpoint()
    if mode == "session":
        full_candidates = [
            job for job in inventory_before["jobs"] if job["phase"] == "full" and job["status"] != "complete"
        ]
        priority = {"failed": 0, "resumable": 1, "unfinished": 2, "not_started": 3}
        full_candidate = min(full_candidates, key=lambda job: priority[job["status"]]) if full_candidates else None
        selected_full = full_candidate["name"] if full_candidate else None
        scheduled = [job for job in jobs if job["phase"] == "smoke" or job["name"] == selected_full]
    else:
        scheduled = jobs
    paused = False
    for job in scheduled:
        if _finished(job) or job["status"] == "failed":
            continue
        if job["phase"] == "full" and any(not _finished(probe) for probe in jobs if probe["phase"] == "smoke"):
            job.update(status="skipped", skip_reason="prerequisite_failed")
            checkpoint()
            continue
        remaining_seconds = session_seconds - (time.monotonic() - started) if session_seconds is not None else None
        if remaining_seconds is not None and remaining_seconds <= 0:
            progress["session_stop_reason"] = "time_budget_exhausted_before_next_job"
            paused = True
            break
        job["attempts"] = previous_jobs.get(job["name"], {}).get("attempts", 0) + 1
        job["status"] = "running"
        job["started_at"] = _timestamp()
        summary_existed = (output / "runs" / job["name"] / "run_summary.json").is_file()
        checkpoint()
        try:
            kwargs: dict[str, Any] = {"resume": True}
            if job["phase"] == "smoke":
                kwargs.update(probe_updates=PROBE_UPDATES, evaluation_episodes=PROBE_EVALUATION_EPISODES)
            if remaining_seconds is not None:
                kwargs["max_seconds"] = remaining_seconds
            result = train_balance_arm(output, job["arm"], job["seed"], **kwargs)
            job["result"] = result
            if result.get("status") == "paused":
                job.update(status="paused", finished_at=_timestamp())
                progress["session_stop_reason"] = "trainer_checkpointed_at_time_limit"
                paused = True
                checkpoint()
                break
            job["completion_validated"] = True
            validation_cache[job["name"]] = {"status": "complete", "result": result}
            if summary_existed:
                job.update(status="skipped", skip_reason="validated_completed_run")
            else:
                job["status"] = "complete"
        except KeyboardInterrupt:
            job.update(status="interrupted", finished_at=_timestamp())
            progress["status"] = "interrupted"
            checkpoint()
            raise
        except Exception as exc:
            job.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        job["finished_at"] = _timestamp()
        checkpoint()

    if any(job["status"] == "failed" for job in jobs if job["phase"] == "smoke"):
        for job in jobs:
            if job["phase"] == "full" and not _finished(job):
                job.update(status="skipped", skip_reason="prerequisite_failed")
    inventory_after = _inspect(output, plan, validation_cache)
    progress["inventory_after"] = inventory_after
    for key in ("completed_full_runs", "remaining_full_runs", "resumable_full_runs", "next_job"):
        progress[key] = inventory_after[key]
    if any(job["status"] == "failed" for job in jobs):
        progress["status"] = "complete_with_errors"
    elif paused:
        progress["status"] = "paused"
    elif all(_finished(job) for job in jobs):
        progress["status"] = "complete"
    else:
        progress["status"] = "session_complete" if mode == "session" else "complete_with_errors"
    try:
        progress["comparison"] = summarize_study(output)
    except Exception as exc:
        progress["status"] = "complete_with_errors"
        progress["comparison_error"] = {"type": type(exc).__name__, "message": str(exc)}
    checkpoint()
    _save(output / "suite_comparison.json", progress)
    print(
        json.dumps(
            {
                "event": "suite_session_result",
                "status": progress["status"],
                "completed_full_runs": progress["completed_full_runs"],
                "remaining_full_runs": progress["remaining_full_runs"],
                "next_job": progress["next_job"]["name"] if progress["next_job"] else None,
            }
        ),
        flush=True,
    )
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("session", "full", "smoke"), default="session")
    parser.add_argument(
        "--session-hours", type=float, default=None, help="Optional session time cap; omitted by default"
    )
    args = parser.parse_args()
    result = run_balance_suite(args.output, mode=args.mode, session_hours=args.session_hours)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] == "complete_with_errors":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
