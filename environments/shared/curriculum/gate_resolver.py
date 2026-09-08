"""The gate resolver: frozen, atomic gate resolution (stage 1b, W5).

STAGE1_SPLIT_PLAN §5: before a stage whose gate compares against measured
null baselines may advance anything, three things are frozen together into
one atomic artifact — the **capability spec** (the gate kind and its
thresholds, normative), the **null manifest** (each null controller's
measured panel on the registered seeds, evidential), and the **decision
procedure** (which statistics, at what level, on what panel sizes,
predeclared).  ``gate_resolution.json`` is that artifact.

The enforcement rules are the module's whole point, and every one fails
closed:

* **Missing resolution blocks advancement.**  "We have not measured the
  nulls" is a blocked gate, never a skipped criterion.
* **A stale resolution blocks advancement.**  The resolution records the
  task fingerprint it was measured under; a different current fingerprint
  (changed config, plant, backend, schedule implementation) means every
  frozen baseline describes a different task, and the error demands
  recalibration rather than proceeding.
* **The paired criterion consumes the frozen panel, not a fresh roll.**
  ``paired_differences_from_resolution`` aligns the policy's per-seed
  outcomes against the null manifest's recorded per-seed outcomes, so the
  gate's superiority claim always references baselines that were measured
  once, on declared seeds, and cannot drift between evaluations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .recovery_gate import (
    RecoveryGateResult,
    RecoveryGateThresholds,
    RecoveryPanel,
    binomial_ucb,
    evaluate_recovery_gate,
)

GATE_RESOLUTION_SCHEMA = "mesozoic.gate-resolution/v1"


class GateResolutionError(RuntimeError):
    """The gate cannot be evaluated: no resolution, or a stale one."""


def _null_manifest_entry(evidence: Any) -> dict[str, Any]:
    """Summarize one null controller's frozen panel.

    ``evidence`` is a ``RecoveryPanelEvidence``; recorded per seed so the
    paired criterion can align against it later, with the exact one-sided
    95% upper bound as the headline (a null that never succeeded is bounded,
    not zero — the split plan's 0/40 → 7.216%).
    """
    successes_by_seed = {str(seed): bool(success) for seed, success in sorted(evidence.successes_by_seed().items())}
    n = len(successes_by_seed)
    successes = sum(1 for outcome in successes_by_seed.values() if outcome)
    return {
        "controller_id": evidence.controller_id,
        "n_episodes": n,
        "n_successes": successes,
        "success_ucb95": binomial_ucb(successes, n) if n else None,
        "successes_by_seed": successes_by_seed,
        "safe_set": dict(evidence.safe_set),
    }


def build_gate_resolution(
    *,
    task_fingerprint: Mapping[str, Any],
    thresholds: RecoveryGateThresholds,
    null_evidence: Mapping[str, Any],
    panel_seed_start: int,
    evaluation_spec: Mapping[str, Any] | None = None,
    null_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze spec, nulls, and procedure into one hashed record."""
    payload: dict[str, Any] = {
        "schema": GATE_RESOLUTION_SCHEMA,
        "task_sha256": task_fingerprint["task_sha256"],
        "capability_spec": {
            "gate_kind": "recovery_quality/v1",
            "min_recovery_success_lcb": thresholds.min_recovery_success_lcb,
            "recovery_t_recover_steps": thresholds.t_recover_steps,
            "recovery_dwell_steps": thresholds.dwell_steps,
            "min_eval_episodes": thresholds.min_eval_episodes,
            "min_paired_success_delta_lcb": thresholds.min_paired_success_delta_lcb,
        },
        "null_manifest": {
            controller_id: _null_manifest_entry(evidence) for controller_id, evidence in sorted(null_evidence.items())
        },
        "decision_procedure": {
            "success_bound": "clopper-pearson one-sided LCB, alpha 0.05",
            "paired_bound": "student-t one-sided LCB on per-seed differences, alpha 0.05",
            "panel_seed_start": panel_seed_start,
            "pairing": "identical seeds imply identical push schedules (schedule PRF)",
        },
    }
    if isinstance(task_fingerprint.get("species"), str):
        payload["species"] = task_fingerprint["species"]
    # Optional for historical T-Rex records. New species freeze the complete
    # judge identity, including the formerly unrecorded height reference and
    # control clock, inside the same integrity hash as their null outcomes.
    if evaluation_spec is not None:
        payload["evaluation_spec"] = dict(evaluation_spec)
    if null_provenance is not None:
        # Config loading restores range tuples for constructors. Freeze their
        # JSON representation so strict write/read equality remains meaningful.
        payload["null_provenance"] = json.loads(json.dumps(dict(null_provenance), allow_nan=False))
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    payload["resolution_sha256"] = f"sha256:{digest}"
    return payload


def write_gate_resolution(stage_dir: "str | Path", resolution: Mapping[str, Any]) -> Path:
    """Atomically persist ``gate_resolution.json`` in the stage directory."""
    path = Path(stage_dir) / "gate_resolution.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(dict(resolution), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)
    return path


def require_gate_resolution(stage_dir: "str | Path", *, current_task_sha256: str) -> dict[str, Any]:
    """Load the stage's resolution, blocking on absence or staleness."""
    path = Path(stage_dir) / "gate_resolution.json"
    if not path.is_file():
        raise GateResolutionError(
            f"no gate_resolution.json in {stage_dir}: the null baselines have not been measured "
            "and frozen for this task, so the gate cannot advance anything (missing baselines "
            "block, they are never skipped)"
        )
    resolution: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if resolution.get("schema") != GATE_RESOLUTION_SCHEMA:
        raise GateResolutionError(
            f"{path} declares schema {resolution.get('schema')!r}; expected {GATE_RESOLUTION_SCHEMA!r}"
        )
    recorded_digest = resolution.get("resolution_sha256")
    payload = {key: value for key, value in resolution.items() if key != "resolution_sha256"}
    expected_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
    )
    if recorded_digest != expected_digest:
        raise GateResolutionError(
            f"{path} fails its own integrity hash (recorded {recorded_digest}, recomputed "
            f"{expected_digest}): the frozen record was edited after resolution. Re-resolve "
            "instead of trusting it."
        )
    recorded_sha = resolution.get("task_sha256")
    if recorded_sha != current_task_sha256:
        raise GateResolutionError(
            f"{path} was resolved for task {recorded_sha}, but the current task is "
            f"{current_task_sha256}: the frozen null baselines describe a different task "
            "(changed config, plant, backend, or schedule implementation). Recalibrate — "
            "re-measure the null panels under the current task — instead of proceeding."
        )
    return resolution


def thresholds_from_resolution(resolution: Mapping[str, Any]) -> RecoveryGateThresholds:
    spec = resolution["capability_spec"]
    return RecoveryGateThresholds(
        min_recovery_success_lcb=float(spec["min_recovery_success_lcb"]),
        t_recover_steps=int(spec["recovery_t_recover_steps"]),
        dwell_steps=int(spec["recovery_dwell_steps"]),
        min_eval_episodes=int(spec["min_eval_episodes"]),
        min_paired_success_delta_lcb=(
            None if spec.get("min_paired_success_delta_lcb") is None else float(spec["min_paired_success_delta_lcb"])
        ),
    )


def paired_differences_from_resolution(
    resolution: Mapping[str, Any],
    policy_successes_by_seed: Mapping[int, bool],
    *,
    null_controller_id: str,
) -> tuple[float, ...]:
    """Per-seed policy-minus-null differences against the FROZEN null panel."""
    manifest = resolution.get("null_manifest", {})
    if null_controller_id not in manifest:
        raise GateResolutionError(
            f"null controller {null_controller_id!r} is not in the frozen null manifest "
            f"({sorted(manifest)}); the paired criterion cannot be evaluated against it"
        )
    null_by_seed = {
        int(seed): bool(outcome) for seed, outcome in manifest[null_controller_id]["successes_by_seed"].items()
    }
    if set(null_by_seed) != set(policy_successes_by_seed):
        raise GateResolutionError(
            "policy panel seeds do not match the frozen null panel seeds: "
            f"policy-only {sorted(set(policy_successes_by_seed) - set(null_by_seed))}, "
            f"null-only {sorted(set(null_by_seed) - set(policy_successes_by_seed))}"
        )
    return tuple(float(policy_successes_by_seed[seed]) - float(null_by_seed[seed]) for seed in sorted(null_by_seed))


def required_paired_nulls(resolution: Mapping[str, Any], *, legacy_null: str = "zero_action") -> tuple[str, ...]:
    """Read the predeclared comparison family; historical gates keep one null."""
    judge = resolution.get("evaluation_spec")
    if judge is None:
        return (legacy_null,)
    required = judge.get("required_paired_nulls") if isinstance(judge, Mapping) else None
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(name, str) or name not in ("zero_action", "brace") for name in required)
        or len(set(required)) != len(required)
    ):
        raise GateResolutionError("evaluation_spec must declare unique known required_paired_nulls")
    if judge.get("species") in ("compsognathus", "compsognathus_robot") and set(required) != {"zero_action", "brace"}:
        raise GateResolutionError(
            "Compsognathus recovery requires paired comparisons against both zero_action and brace"
        )
    return tuple(required)


def _resolution_species(
    resolution: Mapping[str, Any], stage_dir: str | Path | None, expected_species: str | None
) -> str | None:
    """Resolve identity from explicit context and hashed or task-bound records."""
    identities = {expected_species} if expected_species is not None else set()
    judge = resolution.get("evaluation_spec")
    for record in (resolution, judge):
        if isinstance(record, Mapping) and isinstance(record.get("species"), str):
            identities.add(record["species"])
    if stage_dir is not None:
        for name in ("task_fingerprint.json", "stage_config.json"):
            path = Path(stage_dir) / name
            if not path.is_file():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise GateResolutionError(f"cannot verify recovery species from {path}: {exc}") from exc
            if name == "stage_config.json" and isinstance(record, Mapping):
                record = record.get("task_fingerprint")
            if (
                isinstance(record, Mapping)
                and record.get("task_sha256") == resolution.get("task_sha256")
                and isinstance(record.get("species"), str)
            ):
                identities.add(record["species"])
    if len(identities) > 1:
        raise GateResolutionError(f"recovery species identity disagrees across records: {sorted(identities)}")
    return next(iter(identities)) if identities else None


def validate_current_recovery_judge(
    resolution: Mapping[str, Any], *, stage_dir: str | Path | None = None, expected_species: str | None = None
) -> None:
    """A cached outcome cannot be re-certified after its measured judge moves."""
    judge = resolution.get("evaluation_spec")
    species = _resolution_species(resolution, stage_dir, expected_species)
    if judge is None:
        if species != "trex":
            raise GateResolutionError(
                "missing recovery evaluation_spec: only an identified historical T-Rex task may use the legacy judge; "
                "Compsognathus and unknown-species records require a current frozen calibration"
            )
        return
    if not isinstance(judge, Mapping) or not isinstance(judge.get("species"), str):
        raise GateResolutionError("frozen recovery evaluation_spec has no species identity")
    # Lazy: recovery_calibration imports the resolver's error/threshold types.
    from environments.shared.recovery_calibration import load_recovery_calibration

    calibration = load_recovery_calibration(judge["species"])
    if judge != calibration.evaluation_spec():
        raise GateResolutionError("frozen recovery evaluation_spec differs from the current calibration; re-freeze")
    if resolution.get("capability_spec") != calibration.profile["capability_spec"]:
        raise GateResolutionError("frozen recovery capability_spec differs from the current calibration; re-freeze")
    if resolution.get("task_sha256") != calibration.profile["task_sha256"]:
        raise GateResolutionError("frozen recovery task differs from the current calibration; re-freeze")


def evaluate_recovery_gate_from_resolution(
    stage_dir: "str | Path",
    *,
    current_task_sha256: str,
    policy_successes_by_seed: Mapping[int, bool],
    null_controller_id: str = "zero_action",
    expected_species: str | None = None,
):
    """The full W5 path: load the frozen resolution, pair, judge.

    This is the only supported way to produce an advancing recovery-gate
    verdict: thresholds come from the frozen capability spec, the paired
    differences come from the frozen null manifest, and both absence and
    staleness have already blocked upstream.
    """
    resolution = require_gate_resolution(stage_dir, current_task_sha256=current_task_sha256)
    validate_current_recovery_judge(resolution, stage_dir=stage_dir, expected_species=expected_species)
    thresholds = thresholds_from_resolution(resolution)
    required = required_paired_nulls(resolution, legacy_null=null_controller_id)
    if resolution.get("evaluation_spec") is not None and thresholds.min_paired_success_delta_lcb is None:
        raise GateResolutionError("the calibrated recovery comparison family requires a paired success threshold")
    outcomes = tuple(outcome for _seed, outcome in sorted(policy_successes_by_seed.items()))
    results: list[tuple[str, RecoveryGateResult]] = []
    for controller_id in required:
        paired = None
        if thresholds.min_paired_success_delta_lcb is not None:
            paired = paired_differences_from_resolution(
                resolution, policy_successes_by_seed, null_controller_id=controller_id
            )
        results.append(
            (controller_id, evaluate_recovery_gate(RecoveryPanel(outcomes, paired_null_differences=paired), thresholds))
        )
    if len(results) == 1:
        return results[0][1]
    first = results[0][1]
    paired_bounds = [result.paired_delta_lcb for _, result in results if result.paired_delta_lcb is not None]
    return RecoveryGateResult(
        passed=all(result.passed for _, result in results),
        failures=tuple(
            f"{controller_id}: {failure}" for controller_id, result in results for failure in result.failures
        ),
        success_fraction=first.success_fraction,
        success_lcb=first.success_lcb,
        paired_delta_lcb=min(paired_bounds) if paired_bounds else None,
    )
