"""Measure provisional Compsognathus push tasks against fixed-command PD nulls.

Run from the repository root::

    python -m environments.compsognathus.scripts.calibrate_recovery --output calibration.json

This calibrates the task, not a learned recovery policy. It does not consume the
publication seeds 3042--3081. Quiet, force-selection, and held-out diagnostic
panels use separate seeds. Neither null is passive: the model's position
actuators continue providing their built-in proportional/derivative feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from environments.shared.config import load_stage_config
from environments.shared.curriculum.recovery_gate import binomial_lcb, binomial_ucb
from environments.shared.perturbation import derive_push_parameters
from environments.shared.recovery_evaluation import constant_action_controller, roll_recovery_panel
from environments.shared.species_registry import get_species_config

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECIES = ("compsognathus", "compsognathus_robot")
QUIET_SEED = 1042
SELECTION_SEED = 2042
HELDOUT_SEED = 7042
STRESS_SEED = 8042
SETTLE_STEPS = 50
RECOVER_STEPS = 40
DWELL_STEPS = 20
MULTIPLES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FLOORS = {"height_error_max_m": 0.005, "tilt_max_rad": 0.08, "planar_speed_max_mps": 0.08}


def _metrics(env: Any) -> list[float]:
    return [
        float(env.data.qpos[2]),
        float(env._quat_to_tilt(env.data.qpos[3:7])),
        float(np.linalg.norm(env.data.qvel[:2])),
        float(min(env._foot_contact_forces())),
    ]


class TracedEnv:
    """Keep physical per-step measurements while the shared judge rolls episodes."""

    def __init__(self, env: Any):
        self.env = env
        self.traces: list[list[list[float]]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, **kwargs: Any) -> Any:
        self.traces.append([])
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> Any:
        result = self.env.step(action)
        self.traces[-1].append(_metrics(self.env))
        return result


def _new_env(species: str, multiple: float, interval: float = 2.0) -> Any:
    kwargs = dict(load_stage_config(species, "stance")["env_kwargs"])
    kwargs.update(
        perturbation_capture_velocity_multiple=multiple,
        perturbation_interval=interval,
        perturbation_jitter=0.25,
        perturbation_duration=0.20,
        perturbation_direction="uniform_horizontal",
    )
    return get_species_config(species).env_class(**kwargs)


def _brace(env: Any) -> np.ndarray:
    """Predeclared small symmetric hip/ankle bias; never fitted to pushed panels."""
    action = np.zeros(env.model.nu, dtype=np.float64)
    for index in range(env.model.nu):
        name = env.model.actuator(index).name
        if "hip_pitch" in name:
            action[index] = 0.005
        elif "ankle_act" in name:
            action[index] = -0.005
    return action


def _safe_mask(trace: np.ndarray, safe: dict[str, float], reference: float) -> np.ndarray:
    return (
        (np.abs(trace[:, 0] - reference) <= safe["height_error_max_m"])
        & (trace[:, 1] <= safe["tilt_max_rad"])
        & (trace[:, 2] <= safe["planar_speed_max_mps"])
        & (trace[:, 3] >= safe["min_foot_force_n"])
    )


def _geometry(env: Any) -> dict[str, Any]:
    data = mujoco.MjData(env.model)
    mujoco.mj_resetDataKeyframe(env.model, data, env._reset_keyframe_id)
    mujoco.mj_forward(env.model, data)
    floor = {index for index in range(env.model.ngeom) if env.model.geom_bodyid[index] == 0}
    contacts = np.asarray(
        [contact.pos[:2] for contact in data.contact if (int(contact.geom1) in floor) != (int(contact.geom2) in floor)]
    )
    params = derive_push_parameters(
        env.model, capture_velocity_multiple=1.0, duration_s=0.2, keyframe_id=env._reset_keyframe_id
    )
    push_body_id = int(params.get("push_body_id", params["root_body_id"]))
    return {
        "home_subtree_com_xyz_m": data.subtree_com[env.pelvis_id].tolist(),
        "home_root_body_com_xyz_m": data.xipos[env.pelvis_id].tolist(),
        "home_floor_contact_xy_m": contacts.tolist(),
        "home_contact_aabb_xy_min_m": contacts.min(axis=0).tolist(),
        "home_contact_aabb_xy_max_m": contacts.max(axis=0).tolist(),
        "root_body_name": env.model.body(env.pelvis_id).name,
        "root_body_mass_kg": float(env.model.body_mass[env.pelvis_id]),
        "push_body_name": env.model.body(push_body_id).name,
        "push_body_mass_kg": float(env.model.body_mass[push_body_id]),
        "home_push_body_com_xyz_m": data.xipos[push_body_id].tolist(),
    }


def _quiet(species: str, episodes: int) -> dict[str, Any]:
    env = _new_env(species, 0.0)
    traces = []
    lengths = []
    try:
        action = np.zeros(env.model.nu)
        for seed in range(QUIET_SEED, QUIET_SEED + episodes):
            env.reset(seed=seed)
            trace = []
            while True:
                _, _, terminated, truncated, _ = env.step(action)
                trace.append(_metrics(env))
                if terminated or truncated:
                    break
            traces.append(np.asarray(trace))
            lengths.append(len(trace))
        if min(lengths) != env.max_episode_steps:
            raise RuntimeError(f"{species}: quiet zero-command null did not survive every calibration episode")
        samples = np.concatenate([trace[SETTLE_STEPS:] for trace in traces])
        reference = float(np.median(samples[:, 0]))
        p999 = {
            "height_error_max_m": float(np.quantile(np.abs(samples[:, 0] - reference), 0.999)),
            "tilt_max_rad": float(np.quantile(samples[:, 1], 0.999)),
            "planar_speed_max_mps": float(np.quantile(samples[:, 2], 0.999)),
        }
        safe = {key: math.ceil(max(FLOORS[key], 1.5 * value) * 1e6) / 1e6 for key, value in p999.items()}
        # Per-step bilateral load is not a valid posture criterion during
        # stepping; contact force remains measured and recorded separately.
        safe["min_foot_force_n"] = 0.0
        episode_rows = []
        for index, episode_trace in enumerate(traces):
            mask = _safe_mask(episode_trace, safe, reference)
            unsafe = np.flatnonzero(~mask)
            first_dwell = next(
                (step for step in range(len(mask) - DWELL_STEPS + 1) if np.all(mask[step : step + DWELL_STEPS])),
                None,
            )
            episode_rows.append(
                {
                    "seed": QUIET_SEED + index,
                    "length": lengths[index],
                    "first_safe_dwell_start_s": None if first_dwell is None else (first_dwell + 1) * env.dt,
                    "last_unsafe_state_time_s": None if not len(unsafe) else (int(unsafe[-1]) + 1) * env.dt,
                    "postsettle_safe_fraction": float(np.mean(mask[SETTLE_STEPS:])),
                }
            )
        return {
            "source_controller": "zero_command_position_pd_not_certified_stance",
            "calibration_seeds": [QUIET_SEED, QUIET_SEED + episodes - 1],
            "postsettle_steps_per_episode": env.max_episode_steps - SETTLE_STEPS,
            "postsettle_samples": len(samples),
            "settle_steps": SETTLE_STEPS,
            "settle_seconds": SETTLE_STEPS * env.dt,
            "height_reference_m": reference,
            "quiet_p99_9": p999,
            "engineering_floors": FLOORS,
            "threshold_rule": "ceil(1e6 * max(engineering_floor, 1.5 * quiet_p99.9)) / 1e6",
            "safe_set": safe,
            "min_foot_force_p0_1_n": float(np.quantile(samples[:, 3], 0.001)),
            "episodes": episode_rows,
            "brace_action": _brace(env).tolist(),
            "actuator_names": [env.model.actuator(i).name for i in range(env.model.nu)],
            "plant_geometry": _geometry(env),
        }
    finally:
        env.close()


def _quiet_brace(species: str, calibration: dict[str, Any], episodes: int) -> dict[str, Any]:
    env = _new_env(species, 0.0)
    rows = []
    try:
        for seed in range(QUIET_SEED, QUIET_SEED + episodes):
            env.reset(seed=seed)
            trace = []
            while True:
                _, _, terminated, truncated, _ = env.step(_brace(env))
                trace.append(_metrics(env))
                if terminated or truncated:
                    break
            mask = _safe_mask(np.asarray(trace), calibration["safe_set"], calibration["height_reference_m"])
            rows.append(
                {
                    "seed": seed,
                    "length": len(trace),
                    "full_horizon": bool(truncated and len(trace) == env.max_episode_steps),
                    "postsettle_safe_fraction": float(np.mean(mask[SETTLE_STEPS:]))
                    if len(mask) > SETTLE_STEPS
                    else 0.0,
                }
            )
        return {"controller_id": "fixed_brace_pd", "episodes": rows}
    finally:
        env.close()


def _panel(
    species: str,
    calibration: dict[str, Any],
    *,
    multiple: float,
    episodes: int,
    seed: int,
    brace: bool = False,
    interval: float = 2.0,
) -> dict[str, Any]:
    env = _new_env(species, multiple, interval)
    traced = TracedEnv(env)
    try:
        action = _brace(env) if brace else np.zeros(env.model.nu)
        evidence = roll_recovery_panel(
            traced,
            constant_action_controller(action.tolist()),
            controller_id="fixed_brace_pd" if brace else "zero_command_pd",
            episodes=episodes,
            seed=seed,
            t_recover_steps=RECOVER_STEPS,
            dwell_steps=DWELL_STEPS,
            safe_set=calibration["safe_set"],
            height_reference=calibration["height_reference_m"],
        )
        successes = sum(row.success for row in evidence.episodes)
        full = sum(row.full_horizon for row in evidence.episodes)
        recovered = sum(row.recovered for row in evidence.shoves)
        reentry = [(row.recovery_step - row.end_step) * env.dt for row in evidence.shoves if row.recovered]
        rows = []
        for row in evidence.shoves:
            trace = np.asarray(traced.traces[row.episode - 1])
            response = trace[row.start_step : min(len(trace), row.end_step + RECOVER_STEPS + DWELL_STEPS)]
            rows.append(
                {
                    **asdict(row),
                    "reentry_delay_s": (row.recovery_step - row.end_step) * env.dt if row.recovered else None,
                    "response_peak_height_error_m": float(
                        np.max(np.abs(response[:, 0] - calibration["height_reference_m"]))
                    ),
                    "response_peak_tilt_rad": float(np.max(response[:, 1])),
                    "response_peak_planar_speed_mps": float(np.max(response[:, 2])),
                }
            )
        manifest = env.perturbation_manifest()
        return {
            "controller_id": evidence.controller_id,
            "action": action.tolist(),
            "seeds": [seed, seed + episodes - 1],
            "perturbation": manifest,
            "discrete_timing": {
                "control_dt_s": env.dt,
                "duration_steps": env._push_duration_steps,
                "actual_duration_s": env._push_duration_steps * env.dt,
                "actual_impulse_ns": env._push_force_n * env._push_duration_steps * env.dt,
                "interval_steps": env._push_interval_steps,
                "jitter_steps": env._push_jitter_steps,
                "minimum_start_spacing_s": (env._push_interval_steps - 2 * env._push_jitter_steps) * env.dt,
                "reentry_deadline_steps": RECOVER_STEPS,
                "dwell_steps": DWELL_STEPS,
            },
            "summary": {
                "episodes": episodes,
                "episode_successes": successes,
                "episode_success_fraction": successes / episodes,
                "episode_success_lcb95": binomial_lcb(successes, episodes),
                "episode_success_ucb95": binomial_ucb(successes, episodes),
                "full_horizon_episodes": full,
                "full_horizon_fraction": full / episodes,
                "mean_episode_length": float(np.mean([row.length for row in evidence.episodes])),
                "judged_shoves": len(evidence.shoves),
                "recovered_shoves": recovered,
                "recovered_shove_fraction": recovered / len(evidence.shoves) if evidence.shoves else None,
                "recovered_reentry_delay_p50_s": float(np.median(reentry)) if reentry else None,
                "recovered_reentry_delay_p95_s": float(np.quantile(reentry, 0.95)) if reentry else None,
                "recovered_reentry_delay_max_s": max(reentry) if reentry else None,
            },
            "episodes": [asdict(row) for row in evidence.episodes],
            "shoves": rows,
        }
    finally:
        env.close()


def _run_species(arguments: tuple[str, int, int, int, int, str | None]) -> dict[str, Any]:
    species, quiet_episodes, screen_episodes, heldout_episodes, screen_workers, checkpoint_dir = arguments
    calibration = _quiet(species, quiet_episodes)
    print(f"QUIET {species}: {json.dumps(calibration['safe_set'])}", flush=True)
    screens: list[dict[str, Any]] = []

    def checkpoint(extra: dict[str, Any] | None = None) -> None:
        if checkpoint_dir is not None:
            path = Path(checkpoint_dir) / f"{species}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            values = {"species": species, "quiet_calibration": calibration, "selection_panels": screens}
            values.update(extra or {})
            path.write_text(json.dumps(values, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")

    # Independent force levels can use a bounded worker pool. Sorting the
    # returned panels preserves the same selection with any worker count.
    with ProcessPoolExecutor(max_workers=screen_workers) as pool:
        screen = partial(_screen_panel, species, calibration, screen_episodes)
        for panel in pool.map(screen, MULTIPLES):
            screens.append(panel)
            checkpoint()
            print(
                f"SCREEN {species} {panel['perturbation']['capture_velocity_multiple']:g}: "
                f"{json.dumps(panel['summary'])}",
                flush=True,
            )
        # Refine sharp transitions on selection seeds only. First use four
        # interior points; if that still misses the admissible survival
        # bracket, refine the crossing once more at fivefold resolution.
        for refinement in range(2):
            if refinement and _candidates(screens):
                break
            brackets = [
                (lower, upper)
                for lower, upper in zip(screens, screens[1:])
                if lower["summary"]["episode_success_fraction"] > 0.5
                and upper["summary"]["episode_success_fraction"] <= 0.5
            ]
            if not brackets:
                break
            lower, upper = brackets[0]
            multiples = np.linspace(
                lower["perturbation"]["capture_velocity_multiple"],
                upper["perturbation"]["capture_velocity_multiple"],
                6,
            )[1:-1]
            for panel in pool.map(screen, multiples.tolist()):
                screens.append(panel)
                checkpoint()
                print(
                    f"REFINE {species} {panel['perturbation']['capture_velocity_multiple']:g}: "
                    f"{json.dumps(panel['summary'])}",
                    flush=True,
                )
            screens.sort(key=lambda panel: panel["perturbation"]["capture_velocity_multiple"])
    # An engineering task choice, not an optimizer or convergence result.
    # Prefer the first challenge that defeats >=50% of all-push episodes
    # while still allowing at least 25% full-horizon PD-null survival.
    candidates = _candidates(screens)
    if not candidates:
        raise RuntimeError(f"{species}: no screened multiple meets selection rule; expand/reconsider sweep")
    selected = candidates[0]["perturbation"]["capture_velocity_multiple"]
    print(f"SELECTED {species}: {selected:g}", flush=True)
    quiet_brace = _quiet_brace(species, calibration, quiet_episodes)
    heldout = [
        _panel(species, calibration, multiple=selected, episodes=heldout_episodes, seed=HELDOUT_SEED, brace=brace)
        for brace in (False, True)
    ]
    stress = [
        _panel(
            species,
            calibration,
            multiple=1.25 * selected,
            interval=1.9,
            episodes=heldout_episodes,
            seed=STRESS_SEED,
            brace=brace,
        )
        for brace in (False, True)
    ]
    result = {
        "species": species,
        "stance_env_kwargs": load_stage_config(species, "stance")["env_kwargs"],
        "quiet_calibration": calibration,
        "quiet_brace_check": quiet_brace,
        "selection_rule": "first ascending multiple with zero-command success <=0.5 and full-horizon survival >=0.25",
        "selected_capture_velocity_multiple": selected,
        "selection_panels": screens,
        "heldout_diagnostic_panels": heldout,
        "off_distribution_diagnostic_panels": stress,
    }
    checkpoint(result)
    for panel in heldout + stress:
        print(
            f"HELDOUT {species} {panel['controller_id']} {panel['perturbation']['capture_velocity_multiple']:g}: "
            f"{json.dumps(panel['summary'])}",
            flush=True,
        )
    return result


def _screen_panel(species: str, calibration: dict[str, Any], episodes: int, multiple: float) -> dict[str, Any]:
    return _panel(species, calibration, multiple=multiple, episodes=episodes, seed=SELECTION_SEED)


def _candidates(screens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        panel
        for panel in screens
        if panel["summary"]["episode_success_fraction"] <= 0.5 and panel["summary"]["full_horizon_fraction"] >= 0.25
    ]


def _episode_count(value: str) -> int:
    number = int(value)
    if not 2 <= number <= 200:
        raise argparse.ArgumentTypeError("episode count must be between 2 and 200 to preserve seed partitions")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--species", choices=SPECIES, nargs="+", default=list(SPECIES))
    parser.add_argument("--quiet-episodes", type=_episode_count, default=8)
    parser.add_argument("--screen-episodes", type=_episode_count, default=32)
    parser.add_argument("--heldout-episodes", type=_episode_count, default=40)
    parser.add_argument("--screen-workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument(
        "--checkpoint-dir", type=Path, help="Optional directory retaining intermediate species evidence"
    )
    args = parser.parse_args()
    arguments = [
        (
            species,
            args.quiet_episodes,
            args.screen_episodes,
            args.heldout_episodes,
            args.screen_workers,
            str(args.checkpoint_dir) if args.checkpoint_dir is not None else None,
        )
        for species in args.species
    ]
    source_paths = [
        "environments/compsognathus/scripts/calibrate_recovery.py",
        "environments/compsognathus/envs/compsognathus_env.py",
        "environments/shared/base_env.py",
        "environments/shared/perturbation.py",
        "environments/shared/recovery_evaluation.py",
        "environments/shared/curriculum/recovery_gate.py",
        "environments/compsognathus/assets/compsognathus.xml",
        "environments/compsognathus/assets/compsognathus_robot.xml",
        *[f"configs/{species}/stance.toml" for species in args.species],
    ]
    provenance = {
        "working_tree_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "python": sys.version,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "gymnasium")},
        "source_sha256": {path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() for path in source_paths},
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    with ProcessPoolExecutor(max_workers=len(arguments)) as pool:
        results = list(pool.map(_run_species, arguments))
    result = {
        "schema_version": 1,
        "purpose": "Provisional physical task calibration from fixed-command PD nulls; no learned capability certification",
        "provenance": provenance,
        "reserved_publication_seed_range_untouched": [3042, 3081],
        "excluded_discovery_diagnostic_seed_ranges": [[4042, 4081], [5042, 5081]],
        "time_index_convention": (
            "Recovery delay is (recovery_step - push_end_step) * control_dt, as in the shared judge. "
            "States are sampled after env.step: zero delay means safe at the first judged post-push sample, "
            "not literally instantaneous physical recovery."
        ),
        "limitations": [
            "No certified Compsognathus stance or recovery policy was available for calibration.",
            "Fixed commands retain the model's built-in position-actuator PD feedback; they are not passive dynamics.",
            "Capture-velocity scaling uses a linear-inverted-pendulum approximation and home contact AABB, not a stability proof.",
            "Forces act at the selected massive body's center of mass, not the whole robot subtree center of mass.",
            "Safe-set tolerances and timing are provisional engineering choices and must be audited against future certified stance.",
            "Per-shove outcomes are correlated within an episode; confidence bounds are computed only across episodes.",
            "The separate held-out and stress panels characterize task difficulty, not publication-policy performance.",
        ],
        "species": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
