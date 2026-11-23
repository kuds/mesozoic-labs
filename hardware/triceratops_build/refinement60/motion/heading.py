"""Ground-truth evaluation of the live heading controller. None of these observations drive the gait."""

import math

import mujoco
import numpy as np

PROGRAMS = {
    "straight": [(0, 0, 0.25)],
    "left": [(0, 0, 0.25), (3, 40, 0.25)],
    "right": [(0, 0, 0.25), (3, -40, 0.25)],
    "course": [(0, 0, 0.25), (3, 35, 0.25), (9, -35, 0.25), (16, 0, 0.25)],
    "six-headings": [
        (0, 0, 0.25), (3, 15, 0.25), (9, 35, 0.25), (15, 10, 0.25),
        (21, -15, 0.25), (27, -35, 0.25), (33, 0, 0.25),
    ],
    "pivot": [(0, 0, 0.0), (3, 40, 0.0)],
    "pivot-right": [(0, 0, 0.0), (3, -40, 0.0)],
}
DURATION = {
    "straight": 12,
    "left": 12,
    "right": 12,
    "course": 23,
    "six-headings": 40,
    "pivot": 12,
    "pivot-right": 12,
}


def wrapped(x):
    return np.arctan2(np.sin(x), np.cos(x))


def observe(e, commanded_deg, commanded_speed):
    """Sample read-only ground truth and loaded floor-contact slip after each control step."""
    d, m = e.data, e.model
    R = d.xmat[e._chassis].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    slip = np.full(4, np.nan)
    force = np.empty(6)
    vel = np.empty(6)
    geom_ids = d.contact.geom[: d.ncon]
    for i, sole in enumerate(e._soles):
        hits = np.flatnonzero(
            ((geom_ids[:, 0] == sole) & (geom_ids[:, 1] == e._floor))
            | ((geom_ids[:, 1] == sole) & (geom_ids[:, 0] == e._floor))
        )
        if not len(hits):
            continue
        load = 0.0
        for j in hits:
            mujoco.mj_contactForce(m, d, j, force)
            load += force[0]
        if load > e.cfg.contact_threshold_n:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, int(sole), vel, 0)
            arm = d.contact.pos[: d.ncon][hits].mean(axis=0) - d.geom_xpos[sole]
            slip[i] = np.linalg.norm((vel[3:] + np.cross(vel[:3], arm))[:2])
    return (
        d.xpos[e._chassis].copy(),
        float(wrapped(yaw - e._heading)),
        commanded_deg,
        commanded_speed,
        [math.atan2(R[2, 1], R[2, 2]), math.asin(np.clip(-R[2, 0], -1, 1))],
        slip,
        d.qpos.copy(),
    )


def evaluate(trace, program, initial_heading):
    """Score each heading leg from its actual start position along its requested tangent."""
    t, xyz, yaw = trace["t"], trace["xyz"], trace["yaw_rad"]
    legs = []
    for i, (start, requested, speed) in enumerate(PROGRAMS[program]):
        end = PROGRAMS[program][i + 1][0] if i + 1 < len(PROGRAMS[program]) else DURATION[program]
        if t[-1] <= start:
            break
        which = (t >= start - 1e-7) & (t <= end + 1e-7)
        tt = t[which]
        pp = xyz[which, :2]
        yy = yaw[which]
        er = np.degrees(wrapped(yy - np.radians(requested)))
        at_end = tt >= end - 2 - 1e-7
        delta = pp[-1] - pp[0]
        seconds = float(tt[-1] - tt[0])
        if seconds <= 0:
            continue
        a = initial_heading + np.radians(requested)
        fwd = np.array([np.cos(a), np.sin(a)])
        cross = np.array([-fwd[1], fwd[0]])
        inside = abs(er) <= 5.0
        # First entry that persists for one full second, using sample times on both ends.
        n = 1 + round(1 / (t[1] - t[0]))
        ix = np.where(np.convolve(inside, np.ones(n, dtype=int), mode="valid") == n)[0] if len(inside) >= n else []
        tail = np.where(at_end)[0]
        last_speed = lateral_speed = bearing_error = None
        if len(tail) > 1:
            tail_velocity = (pp[tail[-1]] - pp[tail[0]]) / (tt[tail[-1]] - tt[tail[0]])
            last_speed = float(tail_velocity @ fwd)
            lateral_speed = float(tail_velocity @ cross)
            if speed > 0 and np.linalg.norm(tail_velocity) > 0.01:
                bearing_error = math.degrees(math.atan2(lateral_speed, last_speed))
        legs.append(
            {
                "start_s": start,
                "end_s": end,
                "observed_until_s": round(float(tt[-1]), 3),
                "requested_heading_deg": requested,
                "requested_speed_mps": speed,
                "completed": bool(tt[-1] >= end - 1e-6),
                "heading_last2s_mean_abs_error_deg": float(np.abs(er[at_end]).mean()) if at_end.any() else None,
                "final_actual_heading_deg": float(np.degrees(yy[-1])),
                "first_1s_in_5deg_band_s": round(float(tt[ix[0]] - start), 2) if len(ix) else None,
                "projected_mean_speed_mps": float(delta @ fwd / seconds),
                "projected_final2s_speed_mps": last_speed,
                "lateral_final2s_speed_mps": lateral_speed,
                "ground_track_final2s_signed_error_deg": bearing_error,
                "signed_end_crosstrack_m": float(delta @ cross),
                "max_abs_crosstrack_m": float(np.max(np.abs((pp - pp[0]) @ cross))),
                "chassis_displacement_m": float(np.linalg.norm(delta)),
                "max_planar_departure_m": float(np.max(np.linalg.norm(pp - pp[0], axis=1))),
            }
        )
    mask = t >= 3
    slips = trace["slip_mps"][mask]
    vals = slips[np.isfinite(slips)]
    return {
        "segments": legs,
        "abs_peak_roll_pitch_deg": np.max(np.abs(np.degrees(trace["roll_pitch_rad"])), axis=0).tolist(),
        "minimum_chassis_z_m": float(xyz[:, 2].min()),
        "loaded_slip_after3s_mean_mps": float(vals.mean()) if len(vals) else None,
        "loaded_slip_after3s_p95_mps": float(np.quantile(vals, 0.95)) if len(vals) else None,
        "slip_foot_samples": len(vals),
        "elapsed_s": round(float(t[-1]), 3),
    }
