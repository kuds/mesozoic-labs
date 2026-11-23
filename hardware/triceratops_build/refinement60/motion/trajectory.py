"""Rebuild the four-beat joint references from the archived support plan and the supplied leg geometry."""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

DATA = Path(__file__).resolve().parent


def make_trajectory(crawl, kin, steps, params, shape):
    """Blend a continuous trunk path into the original support plan, then solve full leg IK."""
    corners = crawl.CORNERS
    press = np.array([0.0, 0.0, params["stance_press_m"]])
    planted = np.array([kin.neutral_ankle[c] - press for c in corners])
    dt = params["control_dt"]
    t = params["stand_s"]
    qg = {c: np.zeros(3) for c in corners}
    origins = [np.zeros(2)]
    times, body_path, feet_path = [0.0, t], [origins[0], origins[0]], [planted.copy(), planted.copy()]
    key_t, key_x = [0.0, t], [0.0, 0.0]
    phases = []
    for i, step in enumerate(steps):
        leg = corners.index(step["leg"])
        next_body = np.asarray(step["body"])
        b0 = origins[-1]
        p0, p1 = np.array(step["from"]) - press, np.array(step["to"]) - press
        shift, swing = params["shift_s"], params["swing_s"]
        for n in range(1, round((shift + swing) / dt) + 1):
            u = min(n * dt, shift + swing)
            bb = b0 + crawl.min_jerk(u / shift) * (next_body - b0)
            ff = planted.copy()
            if u > shift:
                v = np.clip((u - shift) / swing, 0.0, 1.0)
                if shape.get("foot") == "rounded_shelf":
                    fraction = params["lift_frac"]
                    progress = crawl.min_jerk((v - fraction / 2) / (1 - fraction))
                    envelope = min(crawl.min_jerk(v / fraction), crawl.min_jerk((1 - v) / fraction))
                    arch = envelope * (0.74 + 0.26 * np.sin(np.pi * v) ** 2)
                else:
                    progress = crawl.min_jerk(v)
                    arch = np.sin(np.pi * v) ** 1.5
                ff[leg] = p0 + progress * (p1 - p0)
                ff[leg, 2] += params["lift_m"] * arch
            times.append(t + u)
            body_path.append(bb)
            feet_path.append(ff)
        phases.append(dict(leg=step["leg"], leg_index=leg, swing_start=t + shift, swing_end=t + shift + swing))
        t += shift + swing
        origins.append(next_body)
        if i == 1 or (i >= 2 and (i - 1) % 4 == 0) or i == len(steps) - 1:
            key_t.append(t)
            key_x.append(next_body[0])
        planted[leg] = p1
    final = np.array([planted[:, 0].mean(), 0.0])
    for j in range(1, round(params["shift_s"] / dt) + 1):
        u = min(j * dt, params["shift_s"])
        times.append(t + u)
        body_path.append(origins[-1] + crawl.min_jerk(u / params["shift_s"]) * (final - origins[-1]))
        feet_path.append(planted.copy())
    t += params["shift_s"]
    key_t += [t, t + params["stand_s"]]
    key_x += [final[0], final[0]]
    times.append(key_t[-1])
    body_path.append(final)
    feet_path.append(planted.copy())
    times = np.array(times)
    body_path = np.array(body_path)
    # Use a uniform clock for filtering; the initial and final holds are full one-second spans.
    tu = np.arange(0.0, times[-1] + dt, dt)
    xref = PchipInterpolator(key_t, key_x)(times)
    for axis in (0, 1):
        filtered = gaussian_filter1d(np.interp(tu, times, body_path[:, axis]), shape["smooth_s"] / dt, mode="nearest")
        body_path[:, axis] = np.interp(times, tu, filtered)
    fade = np.array([crawl.min_jerk((v - 3.7) / 2.7) for v in times]) if shape.get("warmup") else 1
    body_path[:, 0] += fade * shape["blend"] * (xref - body_path[:, 0])
    body_path[:, 1] *= 1 - fade * (1 - shape.get("lateral_scale", 1.0))
    targets = []
    for frame, (bb, ff) in enumerate(zip(body_path, feet_path)):
        if frame == 0:
            targets.append(np.zeros(12))
            continue
        feet = dict(zip(corners, ff))
        qg, ok, _ = kin.pose(*bb, feet, qg)
        if not ok:
            raise RuntimeError(f"IK failed at {times[frame]:.3f} s")
        targets.append(np.concatenate([qg[c] for c in corners]))
    return times, np.array(targets), phases


def glance(t, crawl):
    # Once per entire exhibit pass. Independent of step phase.
    up = crawl.min_jerk((t - 5.2) / 2.0)
    down = crawl.min_jerk((t - 10.1) / 2.7)
    inspect = up - down
    nod = crawl.min_jerk((t - 6.3) / 1.9) - crawl.min_jerk((t - 9.0) / 2.6)
    return np.deg2rad([-9 * inspect, 3 * nod])


def exhibit_plan(crawl, model, gait):
    """Return the four-beat support plan, its controller clock and the walking parameters."""
    plan = json.loads((DATA / "reference_plan.json").read_text())
    params = plan["parameters"]
    steps = plan["steps"]
    phases = plan["phases"]
    if gait == "reference":
        # Preserve the archived reference vectors exactly for the independent baseline replay.
        with np.load(DATA / "reference_targets.npz") as targets:
            return (steps, targets["t"], targets["targets"], phases), params, 1.0
    factor = 0.8
    for step in steps[:2]:
        step["body"][0] = 0.0
    params["lift_m"] *= 0.8125
    shape = {**plan["shape"], "lateral_scale": plan["shape"]["lateral_scale"] * 0.88}
    kin = crawl.Kin(model)
    # Plantar targets are specified relative to each reference ankle. Apply those same offsets
    # to the active model, even when the two models happen to have identical pivot locations.
    for step in steps:
        corner = step["leg"]
        reference_ankle = np.array(plan["reference_ankles_xy_m"][corner])
        for key in ("from", "to"):
            offset = np.array(step[key][:2]) - reference_ankle
            step[key][:2] = (kin.neutral_ankle[corner][:2] + offset).tolist()
    times, targets, _ = make_trajectory(crawl, kin, steps, params, shape)
    for event in phases:
        event["swing_start"] *= factor
        event["swing_end"] *= factor
    for duration in ("swing_s", "shift_s", "stand_s"):
        params[duration] *= factor
    return (steps, times * factor, targets, phases), params, factor
