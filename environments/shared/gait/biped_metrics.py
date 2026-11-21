"""Footfall, contact, placement, slip and posture measurements for flat-ground bipeds."""

import mujoco
import numpy as np

from environments.shared.gait import events
from environments.shared.gait.constants import CONTACT_THRESHOLD_N, DEBOUNCE_S
from environments.shared.gait.stance_track import convex_hull, signed_margin, yaw_pitch_roll


def bounds(values):
    return [float(np.min(values)), float(np.max(values))]


def median(values):
    return float(np.median(values)) if len(values) else None


def static_margin(env):
    force = np.zeros(6)
    points = []
    for i, c in enumerate(env.data.contact):
        if env.floor_geom_id in c.geom:
            mujoco.mj_contactForce(env.model, env.data, i, force)
            if force[0] > CONTACT_THRESHOLD_N:
                points.append(c.pos[:2])
    return signed_margin(convex_hull(points), env.data.subtree_com[env.pelvis_id])


def analyse(trace, clearance, start, model, *, pad_suffix="mtp_pad", forward_forces=None):
    """Touchdowns, support and slip are measured independently of controller phase."""
    out = {}
    window = slice(start, None)
    dt = trace.dt
    mask = events.down_mask(trace.leg_down_frac)
    down = np.column_stack([events.debounce(col, events.steps_for(DEBOUNCE_S, dt)) for col in mask.T])
    landings = []
    feet = {}
    for foot, side in enumerate(trace.labels):
        td = events.rising_edges(down[:, foot])
        td = td[td >= start]
        off = events.falling_edges(down[:, foot])
        swings, duties, advances = [], [], []
        pad_share = {"early": [], "late": []}
        late_toe_positive, late_pad_positive, late_toe_signed, late_toe_vertical, late_toe_share = [], [], [], [], []
        support = trace.support_floor_mean[:, np.asarray(trace.support_geom_foot) == foot]
        pad = trace.support_floor_mean[:, trace.support_geom_names.index(f"{side}_{pad_suffix}")]
        foot_clear = np.minimum(clearance[f"{side}_walking_toes"], clearance[f"{side}_pad"])
        for t1, t2 in zip(td, td[1:]):
            liftoff = off[(off > t1) & (off < t2)][0]
            duties.append((liftoff - t1) / (t2 - t1))
            fifth = max(1, (liftoff - t1) // 5)
            for phase, segment in (("early", slice(t1, t1 + fifth)), ("late", slice(liftoff - fifth, liftoff))):
                load = support[segment].sum()
                if load > 0:
                    pad_share[phase].append(pad[segment].sum() / load)
            if forward_forces is not None:
                late = slice(liftoff - fifth, liftoff)
                # Groups are digits II-IV together, then the pad. Positive and braking
                # forces on different digits cancel before rectifying each group.
                forward = forward_forces[late, :, foot, :]
                substep_dt = dt / trace.frame_skip
                toe_j, pad_j = np.maximum(forward, 0).sum(axis=(0, 1)) * substep_dt
                late_toe_positive.append(toe_j)
                late_pad_positive.append(pad_j)
                late_toe_signed.append(forward[:, :, 0].sum() * substep_dt)
                late_toe_vertical.append((support[late].sum() - pad[late].sum()) * dt)
                if toe_j + pad_j > 0:
                    late_toe_share.append(toe_j / (toe_j + pad_j))
            swings.append(np.max(foot_clear[liftoff:t2]))
            advances.append(trace.foot_pos[t2, foot, 0] - trace.foot_pos[t1, foot, 0])
        for step in td:
            landings.append((int(step), foot))
        slip = trace.leg_slip_speed[window, :, foot]
        loaded = slip[np.isfinite(slip)]
        positions = trace.foot_pos[td, foot, :2] - trace.root_pos[td, :2]
        feet[side] = {
            "touchdowns": len(td),
            "mtp_impulse_share_first_stance_fifth": median(pad_share["early"]),
            "mtp_impulse_share_last_stance_fifth": median(pad_share["late"]),
            "contact_fraction": float(np.mean(down[window, foot])),
            "complete_cycle_duty_median": median(duties),
            "stride_period_s_median": median(np.diff(td) * dt),
            "stride_advance_x_m_median": median(advances),
            "touchdown_ahead_of_pelvis_x_m_median": median(positions[:, 0]),
            "touchdown_offset_from_pelvis_y_m_median": median(positions[:, 1]),
            "swing_peak_ground_clearance_m_median": median(swings),
            "loaded_patch_slip_mps_mean": float(loaded.mean()) if loaded.size else None,
            "loaded_patch_slip_mps_p95": float(np.percentile(loaded, 95)) if loaded.size else None,
        }
        if forward_forces is not None:
            feet[side].update(
                late_toe_positive_forward_impulse_ns_median=median(late_toe_positive),
                late_toe_signed_forward_impulse_ns_median=median(late_toe_signed),
                late_toe_vertical_impulse_ns_median=median(late_toe_vertical),
                late_pad_positive_forward_impulse_ns_median=median(late_pad_positive),
                late_toe_positive_forward_impulse_share_median=median(late_toe_share),
            )
    landings.sort()
    alternating = [(b[0] != a[0] and b[1] != a[1]) for a, b in zip(landings, landings[1:])]
    pace_angles = []
    for (t1, f1), (t2, f2), (t3, f3) in zip(landings, landings[1:], landings[2:]):
        if t1 < t2 < t3 and f1 == f3 and f1 != f2:
            p1, p2, p3 = (trace.foot_pos[t, f, :2] for t, f in ((t1, f1), (t2, f2), (t3, f3)))
            backward, onward = p1 - p2, p3 - p2
            scale = np.linalg.norm(backward) * np.linalg.norm(onward)
            if scale > 0:
                pace_angles.append(np.degrees(np.arccos(np.clip(backward @ onward / scale, -1.0, 1.0))))
    signed_widths = []
    step_lengths = []
    for (t1, f1), (t2, f2) in zip(landings, landings[1:]):
        if f1 != f2 and t1 != t2:
            p1, p2 = trace.foot_pos[t1, f1], trace.foot_pos[t2, f2]
            signed_widths.append((p2[1] - p1[1]) * (1 if trace.labels[f2] == "l" else -1))
            step_lengths.append(p2[0] - p1[0])
    out.update(
        feet=feet,
        alternating_touchdown_fraction=float(np.mean(alternating)) if alternating else None,
        pace_angulation_deg_median=median(pace_angles),
        opposite_foot_step_x_m_median=median(step_lengths),
        opposite_foot_width_y_m_median=median(signed_widths),
        double_support_fraction=float(np.mean(np.all(down[window], axis=1))),
        airborne_substep_fraction=float(trace.feet_airborne_substeps[window].mean() / trace.frame_skip),
    )
    floor = trace.geom_floor_mean[window].mean(axis=0)
    total = floor.sum()
    out["floor_impulse_share"] = {
        name: float(load / total) for name, load in zip(trace.geom_floor_names, floor) if load > 0
    }
    out["unregistered_floor_load_fraction"] = (
        sum(v for k, v in out["floor_impulse_share"].items() if k not in trace.support_geom_names)
        if total > 0
        else None
    )
    out["floor_force_peak_bodyweights"] = float(trace.total_floor_max[window].max() / trace.body_weight_n)
    out["min_geom_floor_clearance_m"] = {k: float(np.min(v[window])) for k, v in clearance.items()}
    pos = trace.root_pos[window]
    yaw, pitch, roll = np.degrees(yaw_pitch_roll(trace.root_quat[window])).T
    out.update(
        pelvis_x_travel_m=float(pos[-1, 0] - pos[0, 0]),
        forward_speed_mps=float((pos[-1, 0] - pos[0, 0]) / ((len(pos) - 1) * dt)) if len(pos) > 1 else None,
        pelvis_y_drift_m=float(pos[-1, 1] - pos[0, 1]),
        pelvis_height_m=bounds(pos[:, 2]),
        pelvis_roll_deg=bounds(roll),
        pelvis_pitch_nose_up_deg=bounds(pitch),
        pelvis_yaw_deg=bounds(yaw),
        head_tip_height_m=bounds(trace.site_pos[window, 0, 2]),
        tail_tip_height_m=bounds(trace.site_pos[window, 1, 2]),
    )
    out["hinge_excursion_deg"] = {
        joint: bounds(np.degrees(trace.joint_qpos[window, model.joint(joint).qposadr[0] - 7]))
        for side in trace.labels
        for joint in (f"{side}_hip_pitch", f"{side}_knee", f"{side}_ankle")
    }
    return out
