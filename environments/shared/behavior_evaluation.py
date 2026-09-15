"""Diagnostic evaluation for opt-in direction/terrain behavior policies.

This module does not issue an acceptance or certification verdict.  Command
events share an episode and are reported descriptively, without pretending
that their observations are independent binomial trials.  Radial course
progress is kept separate from command tracking and survival.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from environments.shared.direction_commands import wrap_angle


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _true(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in ("true", "1")
    return isinstance(value, (bool, np.bool_, int, np.integer, float)) and value == 1


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _heading_error(row: Mapping[str, Any]) -> float | None:
    desired, actual = _number(row.get("desired_heading")), _number(row.get("actual_heading"))
    if desired is not None and actual is not None:
        return abs(wrap_angle(desired - actual))
    error = _number(row.get("tracking_error_heading", row.get("heading_error_rad")))
    return abs(error) if error is not None else None


def _active(row: Mapping[str, Any]) -> bool:
    speed = _number(row.get("desired_speed"))
    return speed is not None and speed > 0.0


def _tracking_ok(row: Mapping[str, Any]) -> bool:
    required = ("actual_speed", "actual_yaw_rate", "tracking_error_v", "tracking_error_yaw")
    return (
        _true(row.get("tracking_in_tolerance", False))
        and all(_number(row.get(key)) is not None for key in required)
        and (not _active(row) or _heading_error(row) is not None)
    )


def summarize_episode(
    infos: Sequence[Mapping[str, Any]],
    dt: float,
    *,
    planned_events: Sequence[Mapping[str, Any]] | None = None,
    horizon_steps: int | None = None,
    initial_heading: float | None = None,
    yaw_rate_max: float = 0.3,
    settling_base_s: float = 1.0,
    dwell_s: float = 1.0,
) -> dict[str, Any]:
    """Summarize post-step samples and independently planned command events.

    Each row's ``time_s`` is the *end* of its control step; when absent,
    rows are assumed consecutive at ``(index + 1) * dt``.  An event needs a
    planned window of at least ``1 + abs(initial heading error)/yaw_rate_max
    + dwell_s`` seconds.  The first continuous in-tolerance run must begin
    within that settling allowance and continue for ``dwell_s`` seconds.

    An unfinished short final event cannot pass just because its few samples
    look good.  Planned events after a fall remain present and unsettled.
    Their eligibility uses the nominal requested-heading change because an
    actual future heading does not exist; that distinction is recorded.
    Unknown full-horizon status is ``None`` when no horizon was supplied.
    """
    for name, value in (("dt", dt), ("yaw_rate_max", yaw_rate_max), ("dwell_s", dwell_s)):
        if _number(value) is None or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if _number(settling_base_s) is None or settling_base_s < 0.0:
        raise ValueError("settling_base_s must be finite and nonnegative")
    if horizon_steps is not None and (
        not isinstance(horizon_steps, Integral) or isinstance(horizon_steps, bool) or horizon_steps <= 0
    ):
        raise ValueError("horizon_steps must be a positive integer")
    rows: list[dict[str, Any]] = []
    for index, info in enumerate(infos):
        row = dict(info)
        time_s = _number(row.get("time_s", (index + 1) * dt))
        if time_s is None or time_s <= 0 or (rows and time_s <= rows[-1]["time_s"]):
            raise ValueError("step end times must be finite, positive and strictly increasing")
        row["time_s"] = time_s
        rows.append(row)
    duration = rows[-1]["time_s"] if rows else 0.0
    horizon = float(horizon_steps * dt) if horizon_steps is not None else duration
    if duration > horizon + 1e-9:
        raise ValueError("samples extend beyond the declared episode horizon")
    grouped: dict[int, list[dict[str, Any]]] = {}
    observed_events: dict[int, dict[str, Any]] = {}
    preceding_heading = _number(initial_heading)
    for row in rows:
        event_id = int(row["command_event_id"])
        if event_id not in observed_events:
            measured_error = _number(row.get("event_initial_heading_error_rad"))
            desired = _number(row.get("desired_heading"))
            actual_before = preceding_heading
            if measured_error is None and desired is not None and actual_before is not None:
                measured_error = wrap_angle(desired - actual_before)
            if measured_error is None:
                measured_error = _heading_error(row)
            observed_events[event_id] = {
                "event_id": event_id,
                "start_time_s": float(row["command_event_start_s"]),
                "desired_heading": desired,
                "desired_speed": float(row["desired_speed"]),
                "initial_heading_error_rad": abs(measured_error)
                if _active(row) and measured_error is not None
                else 0.0,
            }
        grouped.setdefault(event_id, []).append(row)
        preceding_heading = _number(row.get("actual_heading"))

    events = {int(event["event_id"]): dict(event) for event in (planned_events or [])}
    for event_id, observed in observed_events.items():
        events.setdefault(event_id, dict(observed))
    ordered = sorted(events.values(), key=lambda event: (float(event["start_time_s"]), int(event["event_id"])))
    ordered = [event for event in ordered if float(event["start_time_s"]) < horizon - 1e-9]
    results: list[dict[str, Any]] = []
    previous_desired = _number(initial_heading)
    for index, event in enumerate(ordered):
        event_id = int(event["event_id"])
        start = float(event["start_time_s"])
        end = float(ordered[index + 1]["start_time_s"]) if index + 1 < len(ordered) else horizon
        samples = grouped.get(event_id, [])
        desired_speed = float(event["desired_speed"])
        desired_heading = float(event["desired_heading"])
        if event_id in observed_events:
            initial_error = observed_events[event_id]["initial_heading_error_rad"]
            eligibility_basis = "observed_initial_heading"
        else:
            initial_error = abs(wrap_angle(desired_heading - previous_desired)) if previous_desired is not None else 0.0
            eligibility_basis = "nominal_request_change_for_unobserved_event"
        if desired_speed == 0.0:
            initial_error = 0.0
        allowance = settling_base_s + initial_error / yaw_rate_max
        eligible = end - start + 1e-9 >= allowance + dwell_s
        acquisition = None
        dwell_complete = None
        continuous = 0.0
        run_start = None
        previous_sample_time = None
        for sample in samples:
            time_s = float(sample["time_s"])
            contiguous = previous_sample_time is None or time_s - previous_sample_time <= dt + 1e-9
            if _tracking_ok(sample):
                if continuous == 0.0 or not contiguous:
                    run_start = max(start, time_s - dt)
                    continuous = 0.0
                continuous += min(dt, max(0.0, time_s - start))
                assert run_start is not None  # Every in-tolerance run records its first sample.
                if continuous + 1e-9 >= dwell_s and run_start <= start + allowance + 1e-9:
                    acquisition = float(run_start - start)
                    dwell_complete = time_s - start
                    break
            else:
                continuous = 0.0
                run_start = None
            previous_sample_time = time_s
        settled = eligible and dwell_complete is not None
        observed_end = float(samples[-1]["time_s"]) if samples else start
        if not eligible:
            status = "window_too_short"
        elif not samples:
            status = "not_observed"
        elif settled:
            status = "settled"
        elif observed_end + 1e-9 < start + allowance + dwell_s:
            status = "observation_incomplete"
        else:
            status = "not_settled"
        results.append(
            {
                "event_id": event_id,
                "start_time_s": start,
                "end_time_s": end,
                "desired_heading": desired_heading,
                "desired_speed": desired_speed,
                "heading_active": desired_speed > 0.0,
                "initial_heading_error_rad": initial_error,
                "settling_allowance_s": allowance,
                "required_window_s": allowance + dwell_s,
                "window_s": end - start,
                "eligibility_basis": eligibility_basis,
                "eligible": eligible,
                "observed_steps": len(samples),
                "observed_window_complete": observed_end + 1e-9 >= end,
                "settled": settled,
                "status": status,
                "acquisition_time_s": acquisition if settled else None,
                "dwell_complete_time_s": dwell_complete if settled else None,
                "mean_actual_speed": _mean([_number(sample.get("actual_speed")) for sample in samples]),
                "mean_abs_yaw_rate": _mean(
                    [
                        abs(event_yaw_value)
                        if (event_yaw_value := _number(sample.get("actual_yaw_rate"))) is not None
                        else None
                        for sample in samples
                    ]
                ),
            }
        )
        previous_desired = desired_heading

    event_results = {event["event_id"]: event for event in results}
    after_first = [row for row in rows if row["time_s"] > 1.0 + 1e-9]
    heading_after_first = [_heading_error(row) for row in after_first if _active(row)]
    heading_after_settle = []
    for row in after_first:
        row_event = event_results.get(int(row["command_event_id"]))
        if (
            row_event
            and _active(row)
            and row["time_s"] >= row_event["start_time_s"] + row_event["settling_allowance_s"]
        ):
            heading_after_settle.append(_heading_error(row))
    stop_rows = [row for row in rows if _number(row.get("desired_speed")) == 0.0]
    eligible = [event for event in results if event["eligible"]]
    stops = [event for event in results if not event["heading_active"]]
    eligible_stops = [event for event in stops if event["eligible"]]
    last = rows[-1] if rows else {}
    terminated = _true(last.get("terminated", False))
    truncated = _true(last.get("truncated", last.get("TimeLimit.truncated", False)))
    reason = last.get("termination_reason") or None
    full_horizon = len(rows) >= horizon_steps and not terminated if horizon_steps is not None else None
    heading_mean = _mean(heading_after_first)
    settled_heading_mean = _mean(heading_after_settle)
    return {
        "steps": len(rows),
        "duration_s": duration,
        "horizon_steps": horizon_steps,
        "horizon_s": horizon if horizon_steps is not None else None,
        "full_horizon": full_horizon,
        "terminated": terminated,
        "truncated": truncated,
        "early_termination": bool(terminated and duration < horizon - 1e-9),
        "fall": bool(terminated and reason not in (None, "terrain_boundary")),
        "termination_reason": reason,
        "reward": sum(_number(row.get("reward")) or 0.0 for row in rows),
        "mean_actual_speed": _mean([_number(row.get("actual_speed")) for row in rows]),
        "mean_actual_speed_after_1s": _mean([_number(row.get("actual_speed")) for row in after_first]),
        "active_heading_mae_deg_after_1s": math.degrees(heading_mean) if heading_mean is not None else None,
        "active_heading_samples_after_1s": len([value for value in heading_after_first if value is not None]),
        "active_heading_mae_deg_after_settle": math.degrees(settled_heading_mean)
        if settled_heading_mean is not None
        else None,
        "active_heading_samples_after_settle": len([value for value in heading_after_settle if value is not None]),
        "tracking_fraction": _fraction(sum(_tracking_ok(row) for row in rows), len(rows)),
        "event_count": len(results),
        "eligible_event_count": len(eligible),
        "settled_event_count": sum(event["settled"] for event in results),
        "eligible_event_settle_fraction": _fraction(sum(event["settled"] for event in eligible), len(eligible)),
        "unobserved_event_count": sum(event["observed_steps"] == 0 for event in results),
        "short_window_event_count": sum(not event["eligible"] for event in results),
        "stop_event_count": len(stops),
        "eligible_stop_event_count": len(eligible_stops),
        "settled_stop_event_count": sum(event["settled"] for event in stops),
        "stop_mean_speed": _mean([_number(row.get("actual_speed")) for row in stop_rows]),
        "stop_mean_abs_yaw_rate": _mean(
            [
                abs(stop_yaw_value) if (stop_yaw_value := _number(row.get("actual_yaw_rate"))) is not None else None
                for row in stop_rows
            ]
        ),
        "stop_tracking_fraction": _fraction(sum(_tracking_ok(row) for row in stop_rows), len(stop_rows)),
        "course_reached_radial_diagnostic": any(_true(row.get("course_reached", False)) for row in rows),
        "max_course_progress_m": max((_number(row.get("course_progress_m")) or 0.0 for row in rows), default=0.0),
        "events": results,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(value) for key, value in row.items()})


def _reset_info(vec_env: Any) -> dict[str, Any]:
    # VecNormalize's own reset_infos can be stale; the innermost VecEnv owns
    # the reset information produced by the concrete Gymnasium environment.
    current = vec_env
    while hasattr(current, "venv"):
        current = current.venv
    values = getattr(current, "reset_infos", None)
    return copy.deepcopy(values[0]) if values else {}


def evaluate_behavior(model: Any, vec_env: Any, *, episode_seeds: list[int], output_dir: Path) -> dict[str, Any]:
    """Evaluate a bound single-environment VecNormalize with frozen statistics.

    Seeds explicitly reset every episode.  The caller remains responsible
    for checkpoint/normalization selection and their hashes.  Normalization
    flags are restored even when inference fails.  This function deliberately
    does not save a gate verdict or overwrite a canonical result bundle.
    """
    if getattr(vec_env, "num_envs", None) != 1:
        raise ValueError("behavior evaluation requires exactly one vectorized environment")
    if not episode_seeds or any(
        not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0 for seed in episode_seeds
    ):
        raise ValueError("episode_seeds must contain nonnegative integers")
    if len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("episode_seeds must be distinct")
    if not hasattr(vec_env, "training") or not hasattr(vec_env, "norm_reward"):
        raise ValueError("vec_env must be a ready VecNormalize with training and norm_reward flags")
    output_dir = Path(output_dir)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    raw_env = vec_env.get_attr("unwrapped", indices=0)[0]
    dt = float(raw_env.dt)
    horizon_steps = int(raw_env.max_episode_steps)
    if not math.isfinite(dt) or dt <= 0 or horizon_steps <= 0:
        raise ValueError("environment control timestep and episode horizon must be positive")
    original_training, original_norm_reward = vec_env.training, vec_env.norm_reward
    episode_summaries: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    vec_env.training = False
    vec_env.norm_reward = False
    try:
        for episode_index, seed in enumerate(episode_seeds):
            vec_env.seed(int(seed))
            observation = vec_env.reset()
            reset_info = _reset_info(vec_env)
            controller = raw_env.direction_controller
            initial_heading = float(controller.events[0]["desired_heading"])
            planner = copy.deepcopy(controller)
            planner.update(horizon_steps * dt, initial_heading)
            planned = [event for event in planner.events if event["start_time_s"] < horizon_steps * dt - 1e-9]
            reset_record = {
                "episode": episode_index,
                "episode_seed": int(seed),
                "reset_info": reset_info,
                "terrain": copy.deepcopy(reset_info.get("terrain", {"family": "flat_plane"})),
                "command_manifest": controller.manifest(),
                "command_schedule_seed": controller.schedule_seed,
                "initial_heading": initial_heading,
                "dt": dt,
                "horizon_steps": horizon_steps,
                "planned_events": planned,
            }
            stem = f"episode_{episode_index:03d}_seed_{int(seed)}"
            _write_json(episodes_dir / f"{stem}_reset.json", reset_record)
            rows = []
            observed_ids = set()
            previous_heading = initial_heading
            for step in range(horizon_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, rewards, dones, batch_infos = vec_env.step(action)
                info = batch_infos[0]
                # Save scalars only; terminal_observation belongs to the
                # vectorization protocol, not to the behavior metric table.
                row = {
                    key: _json_value(value)
                    for key, value in info.items()
                    if value is None or isinstance(value, (str, bool, int, float, np.generic))
                }
                row.update(
                    episode=episode_index,
                    episode_seed=int(seed),
                    step=step + 1,
                    time_s=(step + 1) * dt,
                    reward=float(rewards[0]),
                    terminated=bool(dones[0] and not _true(info.get("TimeLimit.truncated", False))),
                    truncated=bool(dones[0] and _true(info.get("TimeLimit.truncated", False))),
                )
                event_id = int(row["command_event_id"])
                if event_id not in observed_ids:
                    row["event_initial_heading_error_rad"] = (
                        wrap_angle(float(row["desired_heading"]) - previous_heading) if _active(row) else 0.0
                    )
                    observed_ids.add(event_id)
                rows.append(row)
                previous_heading = float(row["actual_heading"])
                if dones[0]:
                    break
            summary = summarize_episode(
                rows,
                dt,
                planned_events=planned,
                horizon_steps=horizon_steps,
                initial_heading=initial_heading,
                yaw_rate_max=planner.config.yaw_rate_max,
            )
            summary.update(episode=episode_index, episode_seed=int(seed))
            _write_csv(episodes_dir / f"{stem}_steps.csv", rows)
            _write_json(episodes_dir / f"{stem}_events.json", summary["events"])
            _write_json(episodes_dir / f"{stem}_summary.json", summary)
            episode_summaries.append(summary)
            all_events.extend(
                {"episode": episode_index, "episode_seed": int(seed), **event} for event in summary["events"]
            )
    finally:
        vec_env.training = original_training
        vec_env.norm_reward = original_norm_reward

    count = len(episode_summaries)
    eligible_count = sum(episode["eligible_event_count"] for episode in episode_summaries)
    settled_count = sum(episode["settled_event_count"] for episode in episode_summaries)
    report = {
        "schema": "mesozoic.behavior-evaluation-diagnostics/v1",
        "purpose": "diagnostic evaluation; no acceptance or certification verdict",
        "protocol": {
            "episode_seeds": [int(seed) for seed in episode_seeds],
            "deterministic_actions": True,
            "normalization_statistics_frozen": True,
            "reward_normalization": False,
            "dt": dt,
            "horizon_steps": horizon_steps,
            "settling_base_s": 1.0,
            "settling_yaw_rate_rad_s": planner.config.yaw_rate_max,
            "dwell_s": 1.0,
            "heading_summary_initial_exclusion_s": 1.0,
            "event_statistics": "descriptive; events within an episode are correlated",
            "course_reached": "radial displacement diagnostic, separate from tracking and survival",
        },
        "episode_count": count,
        "full_horizon_count": sum(episode["full_horizon"] for episode in episode_summaries),
        "full_horizon_fraction": sum(episode["full_horizon"] for episode in episode_summaries) / count,
        "fall_count": sum(episode["fall"] for episode in episode_summaries),
        "early_termination_count": sum(episode["early_termination"] for episode in episode_summaries),
        "termination_reasons": dict(
            Counter(episode["termination_reason"] for episode in episode_summaries if episode["terminated"])
        ),
        "mean_actual_speed": _mean([episode["mean_actual_speed"] for episode in episode_summaries]),
        "mean_actual_speed_after_1s": _mean([episode["mean_actual_speed_after_1s"] for episode in episode_summaries]),
        "active_heading_mae_deg_after_1s": _mean(
            [episode["active_heading_mae_deg_after_1s"] for episode in episode_summaries]
        ),
        "active_heading_mae_deg_after_settle": _mean(
            [episode["active_heading_mae_deg_after_settle"] for episode in episode_summaries]
        ),
        "eligible_event_count": eligible_count,
        "settled_event_count": settled_count,
        "eligible_event_settle_fraction": _fraction(settled_count, eligible_count),
        "unobserved_event_count": sum(episode["unobserved_event_count"] for episode in episode_summaries),
        "short_window_event_count": sum(episode["short_window_event_count"] for episode in episode_summaries),
        "stop_event_count": sum(episode["stop_event_count"] for episode in episode_summaries),
        "eligible_stop_event_count": sum(episode["eligible_stop_event_count"] for episode in episode_summaries),
        "settled_stop_event_count": sum(episode["settled_stop_event_count"] for episode in episode_summaries),
        "stop_mean_speed": _mean([episode["stop_mean_speed"] for episode in episode_summaries]),
        "stop_mean_abs_yaw_rate": _mean([episode["stop_mean_abs_yaw_rate"] for episode in episode_summaries]),
        "course_reached_fraction_radial_diagnostic": sum(
            episode["course_reached_radial_diagnostic"] for episode in episode_summaries
        )
        / count,
        "episodes": episode_summaries,
        "outputs": {
            "summary": str(output_dir / "evaluation_summary.json"),
            "episodes": str(output_dir / "episodes.csv"),
            "command_events": str(output_dir / "command_events.csv"),
            "episode_details": str(episodes_dir),
        },
    }
    _write_csv(
        output_dir / "episodes.csv",
        [{key: value for key, value in episode.items() if key != "events"} for episode in episode_summaries],
    )
    _write_csv(output_dir / "command_events.csv", all_events)
    _write_json(output_dir / "evaluation_summary.json", report)
    return report
