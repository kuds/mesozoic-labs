"""A synthetic but internally consistent six-heading stance certificate, for tests that need no rollout.

:func:`heading_report` builds the ``mesozoic.stance-heading-report/v1``
certificate ``reporting.stance_report.build_stance_heading_report`` would
return for a grid of synthetic statue episodes (the perfect-statue trace of
``test_gait_stance_metrics.py``), with the (seed, heading) cells a test
names carrying a defect, bound exactly as :mod:`.stance_v2_helpers` binds a
v2 report: to a byte-file handoff pair under ``<stage_dir>/models``, to a
measurement manifest the judge accepts for the species, and to the
synthetic task :data:`~.stance_v2_helpers.TASK_SHA256` the stage's
``stage_config.json`` records.  Every number in it re-derives from its rows,
so the judge, publication, the backfill tool and reuse rule 8 accept it
unless a test edits it.  The rolled certificate is exercised in
``test_stance_heading_report.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
from environments.shared.curriculum.stance_gate_v2 import STATUE_POLICY, StanceV2Thresholds, statue_reference
from environments.shared.curriculum.stance_heading_gate import (
    FROZEN_VECNORMALIZE_MODE,
    HEADING_GRID_DEG,
    HEADING_PROTOCOL,
    STANCE_HEADING_GATE_KIND,
    STANCE_HEADING_REPORT_SCHEMA,
    HeadingEpisode,
    evaluate_stance_heading_gate,
)
from environments.shared.gait.constants import measurement_sha256
from environments.shared.gait.stance_metrics import episode_stance_metrics
from environments.shared.result_bundle import sha256_file

from .stance_v2_helpers import HORIZON, TASK_SHA256, V2_CURRICULUM, handoff_pair, measurement, stage_record
from .test_gait_stance_metrics import statue_trace

#: The v2 helper's block under the certificate's kind: the same keys and bars, one statue-relative criterion.
HEADING_CURRICULUM: dict[str, Any] = {**V2_CURRICULUM, "gate_kind": STANCE_HEADING_GATE_KIND}

#: A fall: the episode ends 75 steps in (seed 52's turned episodes ended on tail contact within ~150 steps).
FALL: dict[str, Any] = {"length": 75}

#: The design's seed-52 shape (``20261009_155723``): clean at 0, +45, +90 and 180; every seed falls at -90 and -45.
SEED_52_FALLS: dict[tuple[int, float], dict[str, Any]] = {
    (index, yaw): FALL for index in range(40) for yaw in (-90.0, -45.0)
}


def heading_report(
    stage_dir: Path,
    curriculum: dict[str, Any] | None = None,
    *,
    n: int = 40,
    defects: dict[tuple[int, float], dict[str, Any]] | None = None,
    species: str = "trex",
) -> dict[str, Any]:
    """The certificate a grid of *n* synthetic statue seeds at every heading scores to.

    *defects* maps ``(seed index, heading)`` to the metric fields that cell's
    policy episode carries instead of the clean statue's; the statue rows are
    the clean statue at every heading.  Writes :func:`~.stance_v2_helpers.stage_record`
    (under *curriculum*) into *stage_dir* when it holds no ``stage_config.json``.
    """
    from environments.shared.reporting.stance_report import _PROBE_MARKERS, _stance_v2_episode_rows

    block = dict(HEADING_CURRICULUM if curriculum is None else curriculum)
    if not (Path(stage_dir) / "stage_config.json").exists():
        stage_record(stage_dir, block, species=species)
    thresholds = StanceV2Thresholds.from_curriculum(block)
    clean = episode_stance_metrics(statue_trace(HORIZON), settle_steps=thresholds.settle_steps)
    seed = PUBLICATION_SEED_START
    seeds = [seed + index for index in range(n)]
    defects = defects or {}
    cells = {yaw: [replace(clean, **defects.get((index, yaw), {})) for index in range(n)] for yaw in HEADING_GRID_DEG}
    statue_metrics = [clean] * n
    statue = statue_reference(statue_metrics, horizon=HORIZON) if thresholds.declares_statue_criteria() else None
    verdict = evaluate_stance_heading_gate(
        [
            HeadingEpisode(seeds[index], yaw, metrics)
            for yaw, rows in cells.items()
            for index, metrics in enumerate(rows)
        ],
        thresholds,
        horizon=HORIZON,
        seeds=seeds,
        statue_by_heading=dict.fromkeys(HEADING_GRID_DEG, statue),
        control_dt=0.01,
    )
    touch = [{"unsupported_duty": 0.0, "bilateral_support_duty": 1.0, "single_support_duty": 0.0}] * n
    model, vecnorm = handoff_pair(stage_dir)
    manifest = measurement(species)
    headings = []
    for yaw, metrics in cells.items():
        cell = verdict.heading_result(yaw)
        assert cell is not None
        headings.append(
            {
                "yaw_deg": yaw,
                "turn": {"kind": "spawn_yaw/v1", "yaw_deg": yaw, "whole_scene": True},
                "result": cell.as_dict(),
                "metrics": {
                    "reward_mean": cell.mean_reward,
                    "reward_std": 0.0,
                    "episode_length_mean": sum(row.length for row in metrics) / n,
                    "full_horizon_fraction": cell.full_horizon_fraction,
                    "n_clean": cell.n_clean,
                    "clean_fraction": cell.clean_fraction,
                    "clean_lcb": cell.clean_lcb,
                    "mean_unsupported_duty": 0.0,
                    "bilateral_support_duty": 1.0,
                    "single_support_duty": 0.0,
                },
                "action": {},
                "terminations": {},
                "reward_components": {},
                "episode_evidence": [
                    {"yaw_deg": yaw, **row}
                    for row in _stance_v2_episode_rows(metrics, cell.episode_reasons, touch, seed=seed, horizon=HORIZON)
                ],
                "statue": None
                if statue is None
                else {
                    "policy": STATUE_POLICY,
                    "reused_policy_panel": False,
                    "seed": seed,
                    "reference": statue.as_dict(),
                    "episode_evidence": [
                        {"yaw_deg": yaw, **row}
                        for row in _stance_v2_episode_rows(statue_metrics, None, touch, seed=seed, horizon=HORIZON)
                    ],
                },
            }
        )
    view = gate_config_view(block)
    return {
        "schema": STANCE_HEADING_REPORT_SCHEMA,
        "species": species,
        "stage": 1,
        "gate_kind": block.get("gate_kind"),
        "scored_gate_kind": STANCE_HEADING_GATE_KIND,
        "protocol": HEADING_PROTOCOL,
        "heading_grid_deg": list(HEADING_GRID_DEG),
        "seeds": seeds,
        "seed": seed,
        "episodes": n,
        "policy": "robust_best_model.zip — synthetic",
        "policy_deterministic": True,
        "normalization": {"mode": FROZEN_VECNORMALIZE_MODE, "clip_obs": 10.0, "epsilon": 1e-8},
        "settle_steps": thresholds.settle_steps,
        "horizon": HORIZON,
        "control_dt": 0.01,
        "passed": verdict.passed,
        "failures": list(verdict.failures),
        "result": verdict.as_dict(),
        "thresholds": view["thresholds"],
        "gate_sha256": gate_config_sha256(view),
        "measurement": manifest,
        "measurement_sha256": measurement_sha256(manifest),
        "handoff": {
            "checkpoint": model.name,
            "checkpoint_sha256": sha256_file(model),
            "normalization_sha256": sha256_file(vecnorm),
        },
        "task_sha256": TASK_SHA256,
        "plant_identity": {},
        "checkpoint_plant_validated": True,
        **dict.fromkeys(_PROBE_MARKERS),
        "headings": headings,
    }


def seed52_like_report(stage_dir: Path, **kwargs: Any) -> dict[str, Any]:
    """The design's counterexample as a certificate: v2 would pass it at 0 deg (40/40); -90 and -45 fall on every seed."""
    return heading_report(stage_dir, defects=SEED_52_FALLS, **kwargs)


def heading_trunk_run(run_dir: Path, *, passed: bool = True, verdict: bool = True) -> Path:
    """A judged trex stance directory declaring the certificate (``ancestors_helpers.build_trunk_run``).

    Its handoff pair carries the trunk plant tag, its task is
    :data:`~.stance_v2_helpers.TASK_SHA256` and its ``[env]`` block the horizon
    and control step the synthetic certificate was rolled at; the verdict (when
    *verdict*) is judged under :data:`HEADING_CURRICULUM`.  Write the
    certificate itself with ``write_stance_heading_report(stage_dir,
    heading_report(stage_dir, ...))``.
    """
    from .ancestors_helpers import build_trunk_run

    stage_dir = build_trunk_run(
        run_dir, curriculum=HEADING_CURRICULUM, task_sha256=TASK_SHA256, passed=passed, verdict=verdict
    )
    path = stage_dir / "stage_config.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["reward_weights"] = {"max_episode_steps": HORIZON, "timestep": 0.002, "frame_skip": 5}
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return stage_dir
