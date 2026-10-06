"""A synthetic but internally consistent ``stance_quality/v2`` report, for tests that need no rollout.

:func:`v2_report` builds the report ``reporting.stance_report`` would write
for a panel of synthetic statue episodes (the perfect-statue trace of
``test_gait_stance_metrics.py``, measured with the block's settle window),
bound to a byte-file handoff pair it creates under ``<stage_dir>/models``,
to a measurement manifest that passes the judge's checks for the named
species (this checkout's constants and support-registry entry), and to the
synthetic task :data:`TASK_SHA256` -- which :func:`stage_record` records as
the task the stage ran, in the ``stage_config.json`` :func:`v2_report`
writes when the directory has none.  Every number in it is re-derivable
from its rows, so the judge, the backfill tool and publication accept it
unless a test edits it.  The real-env report is exercised in
``test_stance_gate_v2_report.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.curriculum.gate_schema import GATE_SCHEMA_VERSION, gate_config_view
from environments.shared.curriculum.stance_gate_v2 import (
    STANCE_GATE_V2_KIND,
    STANCE_V2_REPORT_SCHEMA,
    STATUE_POLICY,
    StanceV2Thresholds,
    evaluate_stance_v2_gate,
    statue_reference,
)
from environments.shared.gait.constants import MEASUREMENT_MANIFEST_SCHEMA, measurement_constants, measurement_sha256
from environments.shared.gait.morphology import SUPPORT_REGISTRY
from environments.shared.gait.stance_metrics import StanceEpisodeMetrics, episode_stance_metrics
from environments.shared.result_bundle import sha256_file

from .test_gait_stance_metrics import statue_trace

HORIZON = 1000

#: The task the synthetic panel rolled under and its stage records (a stand-in
#: for ``task_fingerprint.stage_task_fingerprint``'s digest).
TASK_SHA256 = "sha256:" + "7" * 64

#: A complete v2 block with one statue-relative criterion, so the statue block is exercised too.
V2_CURRICULUM: dict[str, Any] = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": STANCE_GATE_V2_KIND,
    "min_eval_episodes": 40,
    "min_clean_stance_lcb": 0.80,
    "settle_steps": 200,
    "min_all_feet_support": 0.98,
    "max_touchdown_rate": 0.25,
    "max_window_displacement_m": 0.10,
    "min_foot_load_share": 0.30,
    "max_actuator_saturation_fraction": 0.10,
    "max_settle_airborne_substeps": 0,
    "max_settle_peak_floor_force_bw": 1.5,
    "max_sole_corner_lift_m": 0.006,
    "min_foot_load_share_statue_ratio": 0.8,
}


def handoff_pair(stage_dir: Path, *, name: str = "robust_best_model") -> tuple[Path, Path]:
    """A complete handoff pair (byte files) under ``<stage_dir>/models``."""
    models = Path(stage_dir) / "models"
    models.mkdir(parents=True, exist_ok=True)
    model, vecnorm = models / f"{name}.zip", models / f"{name}_vecnorm.pkl"
    if not model.exists():
        model.write_bytes(f"{name} weights".encode())
    if not vecnorm.exists():
        vecnorm.write_bytes(f"{name} statistics".encode())
    return model, vecnorm


def stage_record(
    stage_dir: Path,
    curriculum: dict[str, Any] | None = None,
    *,
    species: str = "trex",
    task_sha256: str = TASK_SHA256,
) -> Path:
    """The ``stage_config.json`` the synthetic stage ran under: its horizon and step, the block and the task.

    The step is the 0.002 s x 5 the manifest's ``dt`` 0.01 records; the
    fingerprint snapshot is the shape ``config.save_stage_config`` writes.
    """
    path = Path(stage_dir) / "stage_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "species": species,
        "stage": 1,
        "name": "Balance",
        "description": "Stand",
        "reward_weights": {"max_episode_steps": HORIZON, "timestep": 0.002, "frame_skip": 5},
        "curriculum": dict(V2_CURRICULUM if curriculum is None else curriculum),
        "task_fingerprint": {"schema": "mesozoic.task-fingerprint/v2", "task_sha256": task_sha256},
        "run": {"seed": 42, "timesteps": 11_000_000},
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def measurement(species: str = "trex") -> dict[str, Any]:
    """A manifest the judge accepts for *species*: current constants and registry entry (feet abridged)."""
    return {
        "schema": MEASUREMENT_MANIFEST_SCHEMA,
        "constants": measurement_constants(),
        "species": species,
        "frame_skip": 5,
        "dt": 0.01,
        "body_weight_n": 100.0,
        "registry": SUPPORT_REGISTRY[species].as_manifest(),
        "feet": [],
    }


def v2_report(
    stage_dir: Path,
    curriculum: dict[str, Any] | None = None,
    *,
    n: int = 40,
    n_clean: int | None = None,
    defect: dict[str, Any] | None = None,
    species: str = "trex",
) -> dict[str, Any]:
    """The report a v2 panel of *n* synthetic statue episodes scores to (*n_clean* clean, the rest with *defect*).

    Writes :func:`stage_record` (under *curriculum*) into *stage_dir* when it
    holds no ``stage_config.json``, so the stage records the task the report
    rolled; a test that writes its own record keeps it.
    """
    from environments.shared.reporting.stance_report import _stance_v2_episode_rows

    block = dict(V2_CURRICULUM if curriculum is None else curriculum)
    if not (Path(stage_dir) / "stage_config.json").exists():
        stage_record(stage_dir, block, species=species)
    thresholds = StanceV2Thresholds.from_curriculum(block)
    clean = episode_stance_metrics(statue_trace(HORIZON), settle_steps=thresholds.settle_steps)
    k = n if n_clean is None else n_clean
    metrics: list[StanceEpisodeMetrics] = [clean] * k + [replace(clean, **(defect or {}))] * (n - k)
    statue = statue_reference([clean] * n, horizon=HORIZON) if thresholds.declares_statue_criteria() else None
    result = evaluate_stance_v2_gate(metrics, thresholds, horizon=HORIZON, statue=statue, control_dt=0.01)
    model, vecnorm = handoff_pair(stage_dir)
    manifest = measurement(species)
    touch = [{"unsupported_duty": 0.0, "bilateral_support_duty": 1.0, "single_support_duty": 0.0}] * n
    seed = PUBLICATION_SEED_START
    return {
        "schema": STANCE_V2_REPORT_SCHEMA,
        "species": species,
        "stage": 1,
        "gate_kind": block.get("gate_kind"),
        "scored_gate_kind": STANCE_GATE_V2_KIND,
        "policy": "robust_best_model.zip — synthetic",
        "episodes": n,
        "seed": seed,
        "settle_steps": thresholds.settle_steps,
        "horizon": HORIZON,
        "control_dt": 0.01,
        "passed": result.passed,
        "failures": list(result.failures),
        "thresholds": gate_config_view(block)["thresholds"],
        "measurement": manifest,
        "measurement_sha256": measurement_sha256(manifest),
        "handoff": {
            "checkpoint": model.name,
            "checkpoint_sha256": sha256_file(model),
            "normalization_sha256": sha256_file(vecnorm),
        },
        "task_sha256": TASK_SHA256,
        "statue": None
        if statue is None
        else {
            "policy": STATUE_POLICY,
            "reused_policy_panel": False,
            "seed": seed,
            "reference": statue.as_dict(),
            "episode_evidence": _stance_v2_episode_rows([clean] * n, None, touch, seed=seed, horizon=HORIZON),
        },
        "result": result.as_dict(),
        "metrics": {
            "reward_mean": result.mean_reward,
            "reward_std": 0.0,
            "episode_length_mean": float(HORIZON),
            "full_horizon_fraction": result.full_horizon_fraction,
            "n_clean": result.n_clean,
            "clean_fraction": result.clean_fraction,
            "clean_lcb": result.clean_lcb,
            "mean_unsupported_duty": 0.0,
            "bilateral_support_duty": 1.0,
            "single_support_duty": 0.0,
        },
        "checkpoint_plant_validated": True,
        "filter_actions_hz": None,
        "hold_constant": None,
        "impulse": None,
        "action": {},
        "terminations": {"truncated": n},
        "reward_components": {},
        "episode_evidence": _stance_v2_episode_rows(metrics, result.episode_reasons, touch, seed=seed, horizon=HORIZON),
    }
