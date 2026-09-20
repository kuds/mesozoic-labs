"""A widened root joins the run bundle without a training curve of its own.

``scripts/widen_checkpoint.py`` writes a ROOT stage directory that never
trained in its run (RESULT_BUNDLES.md: the run block carries the parent's seed,
budget and the ``widened_from_*`` lineage). The chain loop JUDGES it: the
30-episode final evaluation and the selected-checkpoint panel are measured
here, but there is no ``evaluations.npz``, so ``build_stage_results_from_eval_data``
leaves ``best_eval_*`` as ``""`` (unmeasured). The bundle writer's canonical
summary rule rejected that on 2026-09-20 (session 1 of NEXT_STEPS.md died at
its first bundle write, right after the stance passed its gate): a null
``best_eval_reward`` is accepted only for a stage whose deliverable record
names ``widened_from_run_id``; a root that trained here still needs its curve.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environments.shared.reporting import save_result_bundle
from environments.shared.result_bundle import ResultBundleError, audit_result_bundle, validate_result_bundle

from .result_bundle_helpers import _complete_bundle_inputs, _plant_identity

PARENT_RUN = "20260815_205206"
UNMEASURED_CURVE = (
    "best_eval_reward",
    "best_eval_std",
    "best_eval_length",
    "best_eval_std_length",
    "best_eval_timestep",
)


def _judged_root_without_a_curve(run_dir: Path, stage_results: list[dict], *, widened_from_run_id: str | None) -> None:
    root = stage_results[0]
    for key in UNMEASURED_CURVE:
        root[key] = ""
    config_path = run_dir / "stage1" / "stage_config.json"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    saved["run"] = {"seed": 42, "n_envs": 4, "timesteps": root["timesteps"]}
    if widened_from_run_id is not None:
        saved["run"]["widened_from_run_id"] = widened_from_run_id
    config_path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save(run_dir: Path, stage_results, stage_configs) -> dict[str, Path]:
    return save_result_bundle(
        stage_results,
        stage_configs,
        "velociraptor",
        "PPO",
        42,
        run_dir,
        backend="stable-baselines3",
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity("velociraptor"),
        run_id="widened-root-test",
    )


def test_a_judged_widened_root_publishes_with_a_null_training_curve(tmp_path: Path, stable_provenance: None) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _judged_root_without_a_curve(run_dir, stage_results, widened_from_run_id=PARENT_RUN)

    paths = _save(run_dir, stage_results, stage_configs)

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    root = summary["stages"]["1"]
    assert root["best_eval_reward"] is None and root["best_eval_std"] is None and root["best_eval_step"] is None
    # The numbers the gate was judged on are measured here and stay required.
    assert root["final_eval_reward"] == 51.0 and root["selected_model_reward"] == 2.0
    assert summary["provenance"]["deliverables"]["1"]["widened_from_run_id"] == PARENT_RUN
    assert "widened_from_run_id" not in summary["provenance"]["deliverables"]["2"]
    assert summary["stages"]["2"]["best_eval_reward"] == 77.0
    assert summary["bundle_status"] == "complete"
    assert validate_result_bundle(run_dir, require_complete=True)["status"] == "canonical-valid"
    assert audit_result_bundle(run_dir)["status"] == "canonical-valid"


def test_a_root_that_trained_here_still_needs_its_training_curve(tmp_path: Path, stable_provenance: None) -> None:
    run_dir = tmp_path / "run"
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm="PPO")
    _judged_root_without_a_curve(run_dir, stage_results, widened_from_run_id=None)

    with pytest.raises(ResultBundleError, match="best_eval_reward must be a finite number for canonical stage 1"):
        _save(run_dir, stage_results, stage_configs)
    assert not (run_dir / "summary.json").exists()
