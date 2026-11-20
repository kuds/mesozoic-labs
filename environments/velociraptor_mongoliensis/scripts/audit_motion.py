"""Measure the anatomical raptor on a flat floor using the actual Gym step path.

Run from the repository root, e.g. python -m environments.velociraptor_mongoliensis.scripts.audit_motion
--mode walk --seconds 10 --warmup 2. Forces, slip and clearances sample each 2 ms physics step.
Touchdown placements use the MTP reference site at the 10 ms control boundary. Coordinates are
world x forward and y left at the start of the trial. Output is JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.gait import events
from environments.shared.gait.constants import CONTACT_THRESHOLD_N, DEBOUNCE_S
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import SubstepContactRecorder
from environments.shared.gait.stance_track import convex_hull, signed_margin, yaw_pitch_roll
from environments.shared.model_id import ModelId
from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

MODEL = ModelId("velociraptor_mongoliensis", "anatomical")


def bounds(values):
    return [float(np.min(values)), float(np.max(values))]


def median(values):
    return float(np.median(values)) if len(values) else None


def geometry_groups(model, sides):
    """All rigid foot geoms, including collision-disabled digits. Render skins are separate."""
    names = [model.body(int(body)).name for body in model.geom_bodyid]
    groups = {}
    for side in sides:
        for key, bodies in (
            ("walking_toes", ["toe_d3", "toe_d4"]),
            ("sickle", ["toe_claw"]),
            ("digit_ii", ["toe_d2"]),
        ):
            selected = {f"{side}_{body}" for body in bodies}
            groups[f"{side}_{key}"] = [i for i, name in enumerate(names) if name in selected]
        groups[f"{side}_shaft"] = [model.geom(f"{side}_metatarsus_geom").id]
        groups[f"{side}_pad"] = [model.geom(f"{side}_mtp_pad").id]
    return groups


def static_margin(env):
    force = np.zeros(6)
    points = []
    for i, c in enumerate(env.data.contact):
        if env.floor_geom_id in c.geom:
            mujoco.mj_contactForce(env.model, env.data, i, force)
            if force[0] > CONTACT_THRESHOLD_N:
                points.append(c.pos[:2])
    return signed_margin(convex_hull(points), env.data.subtree_com[env.pelvis_id])


def analyse(trace, clearance, start, model):
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
        support = trace.support_floor_mean[:, np.asarray(trace.support_geom_foot) == foot]
        pad = trace.support_floor_mean[:, trace.support_geom_names.index(f"{side}_mtp_pad")]
        foot_clear = np.minimum(clearance[f"{side}_walking_toes"], clearance[f"{side}_pad"])
        for t1, t2 in zip(td, td[1:]):
            liftoff = off[(off > t1) & (off < t2)][0]
            duties.append((liftoff - t1) / (t2 - t1))
            fifth = max(1, (liftoff - t1) // 5)
            for phase, segment in (("early", slice(t1, t1 + fifth)), ("late", slice(liftoff - fifth, liftoff))):
                load = support[segment].sum()
                if load > 0:
                    pad_share[phase].append(pad[segment].sum() / load)
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


def run(args, seed):
    class ProbeEnv(VelociraptorMongoliensisAnatomicalEnv):
        MODEL_PATH = args.xml or VelociraptorMongoliensisAnatomicalEnv.MODEL_PATH

    env = ProbeEnv(
        reset_noise_scale=args.noise,
        strike_bonus=0.0,
        prey_distance_range=(1000.0, 1000.0),
        prey_lateral_range=(0.0, 0.0),
    )
    try:
        env.max_episode_steps = max(1, int(np.ceil(args.seconds / env.dt)))
        env.reset(seed=seed)
        controller = None
        if args.mode == "walk":
            from environments.velociraptor_mongoliensis.scripts.reference_walk import ReferenceWalk

            controller = ReferenceWalk(env)
        morphology = Morphology.from_env(env, MODEL)
        recorder = SubstepContactRecorder(env, morphology, per_geom=True)
        recorder.record_slip()
        recorder.record_sites([env.model.site(name).id for name in ("head_tip", "tail_tip")])
        groups = geometry_groups(env.model, morphology.labels)
        clearance = {key: [] for key in groups}
        start = round(args.warmup / env.dt)
        measure_skins = getattr(args, "skin_clearance", False)
        skin_names = [env.model.skin(i).name for i in range(env.model.nskin)] if measure_skins else []
        skin_scene = None
        skin_min = np.full(len(skin_names), np.inf)
        if skin_names:
            # Scene deformation runs on the CPU and does not need an OpenGL context.
            skin_scene = mujoco.MjvScene(env.model, maxgeom=env.model.ngeom + 100)
            skin_options = mujoco.MjvOption()
            skin_options.skingroup[:] = 1
            camera = mujoco.MjvCamera()

        def sample_clearance():
            if skin_scene is not None and env._step_count >= start:
                mujoco.mjv_updateScene(
                    env.model, env.data, skin_options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, skin_scene
                )
                vertex_z = skin_scene.skinvert.reshape(-1, 3)[:, 2]
                low = np.minimum.reduceat(vertex_z, env.model.skin_vertadr) - env.data.geom_xpos[env.floor_geom_id, 2]
                np.minimum(skin_min, low, out=skin_min)
            for key, group in groups.items():
                clearance[key].append(
                    min(mujoco.mj_geomDistance(env.model, env.data, i, env.floor_geom_id, 2.0, None) for i in group)
                )

        env._substep_probe_hook = sample_clearance  # The repository recorder chains this hook.
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        with recorder:
            while True:
                if controller is not None:
                    action = controller.action(env.data, env.dt)
                _, reward, terminated, truncated, info = env.step(action)
                recorder.end_step(action, reward)
                if terminated or truncated:
                    break
            trace = recorder.end_episode(terminated=terminated, truncated=truncated)
        result = dict(
            seed=seed,
            elapsed_s=trace.length * trace.dt,
            terminated=terminated,
            termination_reason=info.get("termination_reason"),
        )
        if measure_skins:
            result["min_rendered_skin_floor_clearance_m"] = {
                name: float(value) if np.isfinite(value) else None
                for name, value in zip(skin_names, skin_min, strict=True)
                if name.startswith(("metatarsal_", "ball_", "hallux_", "digit_", "ungual_"))
            }
        if start < trace.length:
            per_substep = {k: np.array(v).reshape(-1, env.frame_skip) for k, v in clearance.items()}
            result.update(analyse(trace, per_substep, start, env.model))
        if args.mode == "stand":
            margin = static_margin(env)
            result["final_com_margin_inside_loaded_contacts_m"] = float(margin) if np.isfinite(margin) else None
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skin-clearance",
        action="store_true",
        help="Measure the rendered foot-skin vertices against the flat floor at every physics substep",
    )
    parser.add_argument("--mode", choices=("stand", "walk"), default="stand")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=2.0, help="Discard this many initial seconds from metrics")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--xml", type=Path, help="Optional alternative model XML for a matched comparison")
    args = parser.parse_args()
    if not 0 <= args.warmup < args.seconds or args.noise < 0:
        parser.error("require 0 <= warmup < seconds and noise >= 0")
    xml = args.xml or VelociraptorMongoliensisAnatomicalEnv.MODEL_PATH
    xml_hash = hashlib.sha256(xml.read_bytes()).hexdigest()
    results = [run(args, s) for s in args.seeds]
    if hashlib.sha256(xml.read_bytes()).hexdigest() != xml_hash:
        parser.error("model XML changed during sampling; rerun against a single revision")
    print(
        json.dumps(
            dict(
                mode=args.mode,
                mujoco=mujoco.__version__,
                reset_noise=args.noise,
                xml_sha256=xml_hash,
                seconds=args.seconds,
                warmup_s=args.warmup,
                runs=results,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
