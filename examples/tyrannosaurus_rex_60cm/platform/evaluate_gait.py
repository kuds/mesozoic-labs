"""Score the frozen, unsupported intermediate policy on independent command sequences.

Simulation state is used only after inference for scoring and replay. The policy
receives its usual observations, commands and selected appendage targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.robot_validation import GroundSupportProbe, require_unsupported

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = ROOT / "hardware/tyrannosaurus_rex_60cm/platform/gait"
PROTOCOL = HERE / "gait_protocol.json"
SIDES = ("left", "right")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_at(t: float, sign: int, profile: dict) -> tuple[float, float, float, float]:
    knots = profile["phase_knots_s"]
    forward = np.interp(t, knots, profile["forward_fraction"]) * profile["forward_mps"]
    yaw = sign * np.interp(t, knots, profile["yaw_fraction"]) * profile["yaw_radps"]
    phase = 2 * np.pi * profile["expression_hz"] * t
    return (
        float(forward),
        float(yaw),
        profile["neck_amplitude_rad"] * np.sin(phase),
        profile["tail_amplitude_rad"] * np.sin(phase + 0.5),
    )


def runs(mask: np.ndarray):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    yield from zip(edges[::2], edges[1::2], strict=True)


def walking_metrics(times, loads, clearance, dt):
    support_bouts = []
    swing_peaks = [[], []]
    for i in range(2):
        support = (loads[:, i] >= 0.55) & (loads[:, 1 - i] < 0.10)
        for start, end in runs(support):
            if (end - start) * dt >= 0.040 - 1e-9:
                support_bouts.append((times[start], i))
        for start, end in runs(loads[:, i] < 0.10):
            # Require lift-off and subsequent landing in the measurement window.
            if start and end < len(times) and (end - start) * dt >= 0.040 - 1e-9:
                swing_peaks[i].append(float(np.max(clearance[start:end, i])))
    events = [side for _, side in sorted(support_bouts)]
    alternations = sum(x != y for x, y in zip(events, events[1:]))
    medians = [float(np.median(p)) if p else None for p in swing_peaks]
    return alternations, medians, [len(p) for p in swing_peaks]


def sampled_slip(model, data, probe, bw):
    """Load-weighted speed of the material points actually contacting the plane."""
    forces = np.zeros(2)
    weighted = np.zeros(2)
    wrench = np.zeros(6)
    velocity = np.zeros(6)
    for k, contact in enumerate(data.contact[: data.ncon]):
        a, b = map(int, contact.geom)
        sole = b if a in probe.floor_geoms else a if b in probe.floor_geoms else -1
        if sole not in probe.foot_geoms:
            continue
        mujoco.mj_contactForce(model, data, k, wrench)
        side = probe.foot_geoms[sole]
        normal = max(0.0, wrench[0])
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, sole, velocity, 0)
        point_velocity = velocity[3:] + np.cross(velocity[:3], contact.pos - data.geom_xpos[sole])
        forces[side] += normal
        weighted[side] += normal * np.linalg.norm(point_velocity[:2])
    return np.divide(weighted, forces, out=np.full(2, np.nan), where=forces > 0.25 * bw)


def summarize(trace, dt, sign, duration, terminated, limits):
    t, xy, yaw, z, tilt, loads, clear, slip, expression, unexpected = (
        np.asarray(trace[k])
        for k in ("t", "xy", "yaw", "z", "tilt", "loads", "clear", "slip", "expression", "unexpected")
    )
    gait = (t >= 2) & (t < 12)
    straight = (t[1:] >= 2) & (t[1:] < 5)
    turn = (t[1:] >= 6) & (t[1:] < 9)
    # Forward displacement is projected into the robot's instantaneous heading.
    displ = np.diff(xy, axis=0)
    fwd = np.sum(displ * np.stack([np.cos(yaw[:-1]), np.sin(yaw[:-1])], axis=1), axis=1) / dt
    rates = np.diff(np.unwrap(yaw)) / dt
    stop = xy[t >= 14]
    stopped = float(np.sum(np.linalg.norm(np.diff(stop, axis=0), axis=1))) if len(stop) >= 2 else None
    alternations, medians, swings = walking_metrics(t[gait], loads[gait], clear[gait], dt)
    loaded_slips = slip[gait]
    loaded_slips = loaded_slips[np.isfinite(loaded_slips)]
    lag = max(1, round(0.1 / dt))
    # Speed and speed change average over separate 0.1 s windows to suppress instantaneous contact impulses.
    acceleration = np.linalg.norm(xy[2 * lag :] - 2 * xy[lag:-lag] + xy[: -2 * lag], axis=1) / (lag * dt) ** 2
    t_mid = t[lag:-lag]
    transitions = ((t_mid >= 1) & (t_mid < 2)) | ((t_mid >= 12) & (t_mid < 13))
    travel = np.ptp(expression[gait], axis=0) if np.any(gait) else (0.0, 0.0)
    metrics = {
        "duration_s": float(t[-1]),
        "forward_mps": float(np.mean(fwd[straight])) if np.any(straight) else None,
        "turn_radps": float(sign * np.mean(rates[turn])) if np.any(turn) else None,
        "alternations": alternations,
        "median_clearance_m": min(medians) if all(v is not None for v in medians) else None,
        "loaded_slip_p95_mps": float(np.percentile(loaded_slips, 95)) if loaded_slips.size else None,
        "sampled_force_bw": float(np.max(loads)),
        "tilt_p95_deg": float(np.percentile(tilt[gait], 95)) if np.any(gait) else None,
        "pelvis_height_p95_p05_m": float(np.percentile(z[gait], 95) - np.percentile(z[gait], 5))
        if np.any(gait)
        else None,
        "stop_drift_m": stopped,
        "transition_accel_p95_mps2": float(np.percentile(acceleration[transitions], 95))
        if np.any(transitions)
        else None,
        "neck_travel_rad": float(travel[0]),
        "tail_travel_rad": float(travel[1]),
        "swing_counts_left_right": swings,
        "median_clearance_left_right_m": medians,
        "unexpected_ground_contact_peak_bw": float(max(unexpected)),
    }
    failures = []
    if terminated or t[-1] < duration - dt * 0.5:
        failures.append("early_end")
    if metrics["unexpected_ground_contact_peak_bw"] > 0.05:
        failures.append("ground_contact_outside_soles")
    for rule, bound in limits.items():
        if rule == "max_allowed_failed_episodes_per_surface_and_direction":
            continue
        comparator, metric = rule.split("_", 1)
        value = metrics[metric]
        if value is None or not math.isfinite(value) or (value < bound if comparator == "min" else value > bound):
            failures.append(rule)
    return {"metrics": metrics, "passed": not failures, "failed_checks": failures}


def run_episode(env, policy, surface, sign, seed, panel, protocol, save_replay):
    model, data = env.model, env.data
    env.reset(seed=seed)
    require_unsupported(model, data)
    feet = {side: (side + "_sole",) for side in SIDES}
    probe = GroundSupportProbe(model, feet=feet)
    bw = float(mujoco.mj_getTotalmass(model) * abs(model.opt.gravity[2]))
    dt = float(env.dt)
    profile = protocol["commands"]
    duration = profile["phase_knots_s"][-1]
    trace = {k: [] for k in ("t", "xy", "yaw", "z", "tilt", "loads", "clear", "slip", "expression", "unexpected")}
    positions = [data.qpos.copy()] if save_replay else []
    end_reason = "time limit"
    terminated = False
    peaks = [0.0, 0.0]
    n_physics_steps = 0

    def record_substep():
        nonlocal n_physics_steps
        load, unexpected = probe.sample(data)
        peaks[0] = max(peaks[0], float(max(load)) / bw)
        peaks[1] = max(peaks[1], unexpected / bw)
        n_physics_steps += 1

    env.substep_callback = record_substep
    for step in range(round(duration / dt)):
        now = step * dt
        vx, wz, neck, tail = command_at(now, sign, profile)
        env.set_velocity_command(vx, wz)
        env.set_expression(neck, tail)
        _, _, terminated, truncated, info = env.step(policy(env.observe()))
        rot = data.body("pelvis").xmat.reshape(3, 3)
        xyz = data.body("pelvis").xpos
        load, unexpected = probe.sample(data)
        trace["t"].append((step + 1) * dt)
        trace["xy"].append(xyz[:2].copy())
        trace["yaw"].append(math.atan2(rot[1, 0], rot[0, 0]))
        trace["z"].append(float(xyz[2]))
        trace["tilt"].append(float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1)))))
        trace["loads"].append(load / bw)
        trace["unexpected"].append(unexpected / bw)
        trace["clear"].append(
            [mujoco.mj_geomDistance(model, data, g, model.geom("floor").id, 1.0, None) for g in probe.foot_geoms]
        )
        trace["slip"].append(sampled_slip(model, data, probe, bw))
        trace["expression"].append([float(data.joint(joint).qpos[0]) for joint in ("neck_pitch", "tail_yaw")])
        if save_replay:
            positions.append(data.qpos.copy())
        if terminated or truncated:
            end_reason = str(info.get("termination_reason") or ("terminated" if terminated else "truncated"))
            break
    env.substep_callback = None
    report = summarize(trace, dt, sign, duration, terminated, protocol[panel])
    report["metrics"].update(
        substep_peak_foot_force_bw=peaks[0],
        substep_unexpected_ground_contact_bw=peaks[1],
        physics_samples=n_physics_steps,
    )
    if peaks[0] > protocol[panel]["max_sampled_force_bw"]:
        report["failed_checks"].append("substep_peak_foot_force_bw")
    if peaks[1] > 0.05:
        report["failed_checks"].append("substep_unexpected_ground_contact_bw")
    report["passed"] = not report["failed_checks"]
    report.update(seed=seed, surface=surface, sign=sign, control_timestep_s=dt, end_reason=end_reason)
    return report, positions


def main():
    from environments.tyrannosaurus_rex.intermediate.env import TyrannosaurusRexIntermediateEnv

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bundle", type=Path, help="Frozen policy and normalization bundle")
    group.add_argument(
        "--reference", choices=("compact", "nominal", "wide"), help="Development-only, zero learned residual"
    )
    parser.add_argument("--panel", choices=("development", "confirmation"), default="development")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true", help="Record the first seed and positive turn, both surfaces")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    variant = args.reference or "nominal"
    xml = HERE / variant / "model.xml"
    if args.reference:
        if args.panel != "development":
            parser.error("Untrained reference trials are restricted to development")

        def policy(obs):
            return np.zeros(14, dtype=np.float32)
    else:
        from environments.tyrannosaurus_rex.intermediate.inference import load_policy

        policy = load_policy(args.bundle)
        if digest(xml) != digest(TyrannosaurusRexIntermediateEnv.MODEL_PATH):
            raise ValueError("The registered asset differs from the recorded candidate")
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / (args.panel + ".json")
    if report_path.exists():
        parser.error(f"{report_path} already exists; choose a new output directory to preserve the trial")
    package = ROOT / "environments/tyrannosaurus_rex/intermediate"
    assets = [
        PROTOCOL,
        Path(__file__),
        HERE / "gait_video.py",
        xml,
        xml.with_name("components.json"),
        HERE / "design.json",
        HERE / "build.py",
        ROOT / "environments/shared/intermediate_drive.py",
        ROOT / "environments/shared/robot_walking.py",
        ROOT / "environments/shared/robot_floor.py",
        ROOT / "environments/shared/robot_validation.py",
        package / "env.py",
        ROOT / "environments/shared/base_env.py",
        ROOT / "environments/shared/robot_rendering.py",
    ]
    if args.bundle:
        assets.extend(
            [
                package / "inference.py",
                ROOT / "environments/shared/robot_gait_adapter.py",
                ROOT / "configs/tyrannosaurus_rex/intermediate/locomotion.toml",
            ]
        )
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in assets}
    artifact_hashes = (
        {str(p.relative_to(args.bundle)): digest(p) for p in sorted(args.bundle.rglob("*")) if p.is_file()}
        if args.bundle
        else {}
    )
    seeds = protocol[args.panel + "_seeds"]
    episodes = []
    videos = {}
    for surface in protocol["surface"]:
        env = TyrannosaurusRexIntermediateEnv(surface=surface, max_episode_steps=1000, scene_xml=xml)
        try:
            for sign in protocol["turn_sign"]:
                for seed in seeds:
                    save = args.video and seed == seeds[0] and sign == 1
                    result, qpos = run_episode(env, policy, surface, sign, seed, args.panel, protocol, save)
                    episodes.append(result)
                    if save:
                        from .gait_video import write_video

                        filename = f"{args.panel}_{surface}_{seed}.mp4"
                        write_video(
                            args.out / filename,
                            env.model,
                            qpos,
                            env.dt,
                            surface,
                            result,
                            title=(variant + " / untrained reference") if args.reference else "Learned policy",
                        )
                        videos[filename] = digest(args.out / filename)
        finally:
            env.close()
    failures = Counter((e["surface"], e["sign"]) for e in episodes if not e["passed"])
    report = {
        "protocol": protocol["revision"],
        "panel": args.panel,
        "passed": max(failures.values(), default=0)
        <= protocol[args.panel]["max_allowed_failed_episodes_per_surface_and_direction"],
        "candidate": variant,
        "controller": "untrained zero residual on periodic reference"
        if args.reference
        else "trained deterministic residual",
        "sampling": "Force peaks sampled every physics step and compared to the limit. Kinematic, slip and gait metrics sampled once per control step.",
        "mujoco_version": mujoco.__version__,
        "source_sha256": source_hashes,
        "policy_sha256": artifact_hashes,
        "video_sha256": videos,
        "episodes": episodes,
    }
    changed = [str(p.relative_to(ROOT)) for p in assets if digest(p) != source_hashes[str(p.relative_to(ROOT))]]
    if args.bundle:
        changed += [name for name, recorded in artifact_hashes.items() if digest(args.bundle / name) != recorded]
    report["inputs_changed_during_run"] = changed
    if changed:
        report["passed"] = False
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{sum(e['passed'] for e in episodes)}/{len(episodes)} episodes pass: {report_path}")


if __name__ == "__main__":
    main()
