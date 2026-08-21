"""Shared builders for the result-bundle test modules.

``test_result_bundle_*.py`` each cover one submodule of
``environments.shared.result_bundle``; these builders assemble the on-disk
run directories they all operate on.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TypedDict

from environments.shared.reporting import save_evaluation_episodes, save_result_bundle

_COMMIT = "a" * 40


_COMMIT = "a" * 40


class _InitializeResultBundleKwargs(TypedDict):
    species: str
    algorithm: str
    backend: str
    seed: int
    evaluation_seeds: list[int]
    evaluation_episodes: int
    parallel_envs: int
    plant_identity: dict[str, Any]
    run_id: str
    captured_at: str


def _plant_identity() -> dict[str, Any]:
    return {
        "schema": "mesozoic.plant-identity/v1",
        "species": "velociraptor",
        "model_path": "environments/velociraptor/assets/raptor.xml",
        "physics_revision": 1,
        "policy_interface_revision": 1,
        "visual_revision": 1,
        "source_closure_sha256": "sha256:" + "1" * 64,
        "policy_interface_sha256": "sha256:" + "2" * 64,
        "physics_sha256": "sha256:" + "3" * 64,
        "visual_sha256": "sha256:" + "4" * 64,
        "nq": 31,
        "nv": 30,
        "nu": 22,
        "observation_dim": 67,
        "action_dim": 22,
    }


def _stage_result(stage: int, *, model_path: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage": stage,
        "name": f"Stage {stage}",
        "description": f"Description for stage {stage}",
        "timesteps": stage * 100_000,
        "duration_seconds": stage * 10.0,
        "mean_reward": 50.0 + stage,
        "std_reward": 2.0,
        "mean_episode_length": 200.0 + stage,
        "std_episode_length": 4.9,
        "mean_forward_vel": 0.25 * stage,
        "std_forward_vel": 0.05,
        "mean_distance_traveled": 1.5 * stage,
        "mean_success_rate": stage / 3,
        "best_eval_reward": 75.0 + stage,
        "best_eval_std": 1.5,
        "best_eval_length": 210.0 + stage,
        "best_eval_timestep": stage * 90_000,
        "best_model_reward": float(stage + 1),
        "best_model_std_reward": 0.82,
        "best_model_length": 110.0,
        "best_model_std_length": 8.2,
        "best_model_fwd_vel": 0.2,
        "best_model_std_fwd_vel": 0.082,
        "best_model_distance": 2.0,
        "best_model_success_rate": 0.6667,
        "gate_passed": True,
        "publication_gate_passed": True,
        "plant_identity": _plant_identity(),
    }
    if model_path is not None:
        result["model_path"] = str(model_path)
    return result


def _stage_config(stage: int, algorithm: str) -> dict[str, Any]:
    algorithm_key = "jax_kwargs" if algorithm == "JAX_PPO" else f"{algorithm.lower()}_kwargs"
    return {
        "name": f"Stage {stage}",
        "description": f"Description for stage {stage}",
        "env_kwargs": {"forward_vel_weight": float(stage)},
        algorithm_key: {"learning_rate": 3e-4},
        "curriculum_kwargs": {"min_avg_reward": 1.0},
    }


def _legacy_stage_dirname(stage: int) -> str:
    return f"stage{stage}"


def _write_stage_configs(
    run_dir: Path,
    stage_configs: dict[int, dict[str, Any]],
    dirname=_legacy_stage_dirname,
) -> None:
    for stage, config in stage_configs.items():
        stage_dir = run_dir / dirname(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        if "jax_kwargs" in config:
            algorithm = "JAX_PPO"
        elif "sac_kwargs" in config:
            algorithm = "SAC"
        else:
            algorithm = "PPO"
        saved_config = {
            **config,
            "species": "velociraptor",
            "stage": stage,
            "algorithm": algorithm,
            "plant_identity": _plant_identity(),
        }
        (stage_dir / "stage_config.json").write_text(
            json.dumps(saved_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _snapshot_files(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _rewrite_csv_cell(path: Path, *, field: str, value: str) -> None:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    assert rows
    assert field in fieldnames
    rows[0][field] = value
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _complete_bundle_inputs(
    run_dir: Path,
    *,
    algorithm: str,
    dirname=_legacy_stage_dirname,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    stage_configs = {stage: _stage_config(stage, algorithm) for stage in (1, 2, 3)}
    _write_stage_configs(run_dir, stage_configs, dirname=dirname)
    stage_results = []
    for stage in (1, 2, 3):
        selected_model = run_dir / dirname(stage) / "models" / "best_model.pkl"
        selected_model.parent.mkdir(parents=True, exist_ok=True)
        selected_model.write_bytes(f"selected model stage {stage}".encode())
        selected_vecnorm = run_dir / dirname(stage) / "models" / "best_model_vecnorm.pkl"
        selected_vecnorm.write_bytes(f"selected normalization stage {stage}".encode())
        stage_result = _stage_result(stage, model_path=selected_model)
        stage_result["vecnorm_path"] = str(selected_vecnorm)
        stage_results.append(stage_result)
    for stage in (1, 2, 3):
        final_reward = 50.0 + stage
        final_forward_velocity = 0.25 * stage
        final_distance = 1.5 * stage
        save_evaluation_episodes(
            run_dir / dirname(stage),
            rewards=[
                final_reward - 2.449489743,
                final_reward,
                final_reward + 2.449489743,
            ],
            lengths=[194 + stage, 200 + stage, 206 + stage],
            forward_velocities=[
                final_forward_velocity - 0.061237244,
                final_forward_velocity,
                final_forward_velocity + 0.061237244,
            ],
            distances=[final_distance - 1.0, final_distance, final_distance + 1.0],
            successes=[episode < stage for episode in range(3)],
            evaluation_seed=101,
            checkpoint_label="final",
        )
        save_evaluation_episodes(
            run_dir / dirname(stage),
            rewards=[float(stage), float(stage + 1), float(stage + 2)],
            lengths=[100, 110, 120],
            forward_velocities=[0.1, 0.2, 0.3],
            distances=[1.0, 2.0, 3.0],
            successes=[False, True, True],
            evaluation_seed=101,
            checkpoint_label="selected",
        )
    return stage_results, stage_configs


def _complete_bundle(
    run_dir: Path,
    *,
    algorithm: str,
    backend: str,
    dirname=_legacy_stage_dirname,
) -> tuple[dict[str, Path], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    stage_results, stage_configs = _complete_bundle_inputs(run_dir, algorithm=algorithm, dirname=dirname)
    paths = save_result_bundle(
        stage_results,
        stage_configs,
        "velociraptor",
        algorithm,
        42,
        run_dir,
        backend=backend,
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity(),
        run_id=f"velociraptor-{backend}-{algorithm.lower()}-test",
    )
    return paths, stage_results, stage_configs
