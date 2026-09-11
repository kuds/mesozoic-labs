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


def _jobs(plan: dict, mode: str) -> list[dict]:
    if set(plan["arms"]) != set(ARMS):
        raise ValueError("The balance suite requires exactly arms A, B, C and D")
    seeds = set(plan["training_seeds"])
    if 42 not in seeds or (mode == "full" and seeds != set(TRAINING_SEEDS)):
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
            for seed in TRAINING_SEEDS
        )
    return jobs


def _finished(job: dict) -> bool:
    return job["status"] == "complete" or (
        job["status"] == "skipped" and job.get("skip_reason") == "validated_completed_run"
    )


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


def run_balance_suite(output: str | Path, *, mode: str = "full") -> dict:
    """Run all prepared jobs, validating completed output on every invocation.

    ``full`` runs A–D probes (two PPO updates and four evaluation episodes at
    seed 42), then A–D full training at seeds 42/43/44. ``smoke`` runs only the
    probes. Every job delegates completion validation/resume to the trainer.

    A probe error blocks the full sweep. A full-run error is recorded while
    the remaining runs continue. KeyboardInterrupt records the current job
    and propagates, so Run all can resume from verified saved work later.
    """
    if mode not in ("full", "smoke"):
        raise ValueError("mode must be 'full' or 'smoke'")
    output = Path(output)
    plan = _load_plan(output)
    digest = _plan_digest(plan)
    progress_path = output / "suite_progress.json"
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else None
    if previous is not None and (previous.get("schema") != SUITE_SCHEMA or previous.get("study_plan_sha256") != digest):
        raise ValueError("Saved suite progress belongs to a different study; use a matching prepared output")
    jobs = _jobs(plan, mode)
    previous_jobs = {job["name"]: job for job in previous.get("jobs", [])} if previous else {}
    progress = {
        "schema": SUITE_SCHEMA,
        "study_plan_sha256": digest,
        "implementation_sha256": plan.get("implementation_sha256"),
        "mode": mode,
        "status": "running",
        "started_at": previous.get("started_at", _timestamp()) if previous else _timestamp(),
        "invocations": previous.get("invocations", 0) + 1 if previous else 1,
        "research_only": True,
        "production_advancement": False,
        "planned_training_steps": sum(job["planned_training_steps"] for job in jobs),
        "planned_probe_training_steps": sum(job["planned_training_steps"] for job in jobs if job["phase"] == "smoke"),
        "planned_full_training_steps": sum(job["planned_training_steps"] for job in jobs if job["phase"] == "full"),
        "jobs": jobs,
    }

    def checkpoint() -> None:
        _update_totals(progress)
        _save(progress_path, progress)

    checkpoint()
    for job in jobs:
        if job["phase"] == "full" and any(not _finished(probe) for probe in jobs if probe["phase"] == "smoke"):
            job.update(status="skipped", skip_reason="prerequisite_failed")
            checkpoint()
            continue
        job["attempts"] = previous_jobs.get(job["name"], {}).get("attempts", 0) + 1
        job["status"] = "running"
        job["started_at"] = _timestamp()
        summary_existed = (output / "runs" / job["name"] / "run_summary.json").is_file()
        checkpoint()
        try:
            kwargs: dict[str, Any] = {"resume": True}
            if job["phase"] == "smoke":
                kwargs.update(probe_updates=PROBE_UPDATES, evaluation_episodes=PROBE_EVALUATION_EPISODES)
            result = train_balance_arm(output, job["arm"], job["seed"], **kwargs)
            job["result"] = result
            job["completion_validated"] = True
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

    progress["status"] = "complete" if all(_finished(job) for job in jobs) else "complete_with_errors"
    try:
        progress["comparison"] = summarize_study(output)
    except Exception as exc:
        progress["status"] = "complete_with_errors"
        progress["comparison_error"] = {"type": type(exc).__name__, "message": str(exc)}
    checkpoint()
    _save(output / "suite_comparison.json", progress)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    args = parser.parse_args()
    result = run_balance_suite(args.output, mode=args.mode)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
