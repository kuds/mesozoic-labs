"""Measure the flat-ground anatomical T. rex through the normal Gym step and reset paths.

Run python -m environments.tyrannosaurus_rex.scripts.audit_motion from the repository root.
The reference controller supplies residual joint commands. Contact and slip are sampled
from the physics solver at each substep. Forward travel is along the starting world x axis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.gait.biped_metrics import analyse, static_margin
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import EpisodeTrace, SubstepContactRecorder
from environments.shared.model_id import ModelId
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv
from environments.tyrannosaurus_rex.scripts.reference_walk import ReferenceWalk


def _actuator_metrics(model: mujoco.MjModel, trace: EpisodeTrace, angles: list[np.ndarray], start: int) -> dict:
    """Compare solver forces and hinge positions to their physical limits and commands to ±1."""
    assert trace.actuator_force_substep is not None
    forces = trace.actuator_force_substep[start:].reshape(-1, model.nu)
    lower, upper = model.actuator_forcerange.T
    force_share = np.abs(forces) / np.where(forces < 0, -lower, upper)
    joints = model.actuator_trnid[:, 0]
    low_angle, high_angle = model.jnt_range[joints].T
    qpos = np.asarray(angles)[start * trace.frame_skip :]
    margin = np.minimum(qpos - low_angle, high_angle - qpos)
    names = [model.actuator(i).name for i in range(model.nu)]

    def per_actuator(values):
        return dict(zip(names, values.tolist(), strict=True))

    # A reading within 0.01% of the corresponding hard bound counts as at the limit.
    threshold = 0.9999
    return {
        "samples": len(forces),
        "abs_force_share_peak": per_actuator(force_share.max(axis=0)),
        "fraction_at_force_limit": per_actuator((force_share >= threshold).mean(axis=0)),
        "minimum_angular_margin_deg": per_actuator(np.rad2deg(margin.min(axis=0))),
        "fraction_at_action_limit": per_actuator((np.abs(trace.action[start:]) >= threshold).mean(axis=0)),
    }


def run(
    *,
    seed: int,
    noise: float,
    seconds: float,
    warmup: float,
    mode: str = "walk",
    env: TyrannosaurusRexAnatomicalEnv | None = None,
    predict: Callable[[np.ndarray], np.ndarray] | None = None,
    trace_path: Path | None = None,
    actuators: bool = False,
    foot_mechanics: bool = False,
) -> dict:
    """Use a supplied environment for policy replay. Its owner controls construction and closing."""
    if (predict is not None) != (mode == "policy") or mode not in {"walk", "stand", "policy"}:
        raise ValueError("policy mode requires predict, and walk/stand use their own command source")
    if not 0 <= warmup < seconds or noise < 0:
        raise ValueError("require 0 <= warmup < seconds and noise >= 0")
    owned = env is None
    if env is None:
        env = TyrannosaurusRexAnatomicalEnv(
            reset_noise_scale=noise, prey_distance_range=(100.0, 100.0), prey_lateral_range=(0.0, 0.0), bite_bonus=0.0
        )
    previous_hook = env._substep_probe_hook
    previous_horizon = env.max_episode_steps
    previous_noise = env.reset_noise_scale
    try:
        env.reset_noise_scale = noise
        env.max_episode_steps = max(1, int(np.ceil(seconds / env.dt)))
        observation, _ = env.reset(seed=seed)
        controller = ReferenceWalk(env) if mode == "walk" else None
        morphology = Morphology.from_env(env, ModelId("tyrannosaurus_rex", "anatomical"))
        recorder = SubstepContactRecorder(env, morphology, per_geom=True)
        recorder.record_slip()
        recorder.record_actuators(actuators)
        angle_samples: list[np.ndarray] = []
        actuated_qpos = env.model.jnt_qposadr[env.model.actuator_trnid[:, 0]]
        recorder.record_sites([env.model.site(name).id for name in ("head_tip", "tail_tip")])
        groups = {}
        for side in morphology.labels:
            groups[f"{side}_walking_toes"] = [env.model.geom(f"{side}_toe_d{j}_geom").id for j in (2, 3, 4)]
            groups[f"{side}_pad"] = [env.model.geom(f"{side}_plantar_geom").id]
            groups[f"{side}_shaft"] = [env.model.geom(f"{side}_metatarsus_geom").id]
        heights: dict[str, list[float]] = {label: [] for label in groups}

        forward_samples: list[np.ndarray] = []
        foot_gaps: list[float] = []
        feet = [groups[f"{side}_pad"] + groups[f"{side}_walking_toes"] for side in morphology.labels]
        geom_group = {
            geom: (side, int(index == 0)) for side, foot in enumerate(feet) for index, geom in enumerate(foot)
        }
        contact_force = np.zeros(6)

        foot_ids = np.array(feet)
        sizes = env.model.geom_size[foot_ids]
        left, right = morphology.labels.index("l"), morphology.labels.index("r")

        def sample_foot_mechanics():
            forward = np.zeros((len(feet), 2))
            for index, contact in enumerate(env.data.contact):
                first, second = contact.geom
                if first != env.floor_geom_id and second != env.floor_geom_id:
                    continue
                floor_first = first == env.floor_geom_id
                group = geom_group.get(second if floor_first else first)
                if group is not None:
                    mujoco.mj_contactForce(env.model, env.data, index, contact_force)
                    # The contact frame stores its world basis in rows. Force acts on geom 2.
                    fx = contact.frame.reshape(3, 3)[:, 0] @ contact_force[:3]
                    forward[group] += fx if floor_first else -fx
            forward_samples.append(forward)
            # Exact world-y projection of each support box and capsule. A positive
            # gap proves lateral separation even as the feet pass one another.
            lateral_basis = env.data.geom_xmat[foot_ids].reshape(len(feet), -1, 3, 3)[:, :, 1, :]
            half_width = sizes[:, :, 0] + np.abs(lateral_basis[:, :, 2]) * sizes[:, :, 1]
            half_width[:, 0] = (np.abs(lateral_basis[:, 0]) * sizes[:, 0]).sum(axis=1)
            centers = env.data.geom_xpos[foot_ids, 1]
            foot_gaps.append(
                float((centers[left] - half_width[left]).min() - (centers[right] + half_width[right]).max())
            )

        def sample_substep():
            if previous_hook is not None:
                previous_hook()
            if foot_mechanics:
                sample_foot_mechanics()
            if actuators:
                angle_samples.append(env.data.qpos[actuated_qpos])
            for name, geoms in groups.items():
                heights[name].append(
                    min(
                        mujoco.mj_geomDistance(env.model, env.data, geom, env.floor_geom_id, 3.0, None)
                        for geom in geoms
                    )
                )

        env._substep_probe_hook = sample_substep
        action = np.zeros(env.model.nu)
        qpos = []
        with recorder:
            while True:
                if controller is not None:
                    action = controller.action(env.data, env.dt)
                elif predict is not None:
                    action = predict(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                if trace_path is not None:
                    qpos.append(env.data.qpos.copy())
                recorder.end_step(action, reward)
                if terminated or truncated:
                    break
            trace = recorder.end_episode(terminated=terminated, truncated=truncated)
        start = round(warmup / env.dt)
        result = dict(
            seed=seed,
            elapsed_s=trace.length * trace.dt,
            terminated=terminated,
            termination_reason=info.get("termination_reason"),
        )
        clearance = {name: np.array(values).reshape(-1, env.frame_skip) for name, values in heights.items()}
        if trace_path is not None:
            np.savez_compressed(
                trace_path,
                qpos=qpos,
                times=np.arange(1, trace.length + 1) * trace.dt,
                labels=trace.labels,
                foot_pos=trace.foot_pos,
                loads=trace.leg_floor_mean,
                clearance=np.column_stack(
                    [
                        np.minimum(clearance[f"{s}_walking_toes"], clearance[f"{s}_pad"]).max(axis=1)
                        for s in trace.labels
                    ]
                ),
                pelvis=trace.root_pos,
            )
        if start < trace.length:
            forward_forces = None
            if foot_mechanics:
                forward_forces = np.asarray(forward_samples).reshape(trace.length, env.frame_skip, len(feet), 2)
                result["min_lateral_foot_gap_m"] = min(foot_gaps[start * env.frame_skip :])
                result["interleg_contact_substep_fraction"] = float(trace.interleg_substep_frac[start:].mean())
            result.update(
                analyse(trace, clearance, start, env.model, pad_suffix="plantar_geom", forward_forces=forward_forces)
            )
        if actuators:
            result["actuators"] = (
                _actuator_metrics(env.model, trace, angle_samples, start) if start < trace.length else None
            )
        if mode == "stand":
            margin = static_margin(env)
            result["final_com_margin_inside_loaded_contacts_m"] = float(margin) if np.isfinite(margin) else None
        return result
    finally:
        env._substep_probe_hook = previous_hook
        env.max_episode_steps = previous_horizon
        env.reset_noise_scale = previous_noise
        if owned:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["walk", "stand"], default="walk")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--trace", type=Path, help="Save the first seed’s qpos, foot forces and clearances as NPZ")
    parser.add_argument(
        "--actuators", action="store_true", help="Report post-warmup force limits, hinge margins and command clipping"
    )
    parser.add_argument(
        "--foot-mechanics", action="store_true", help="Measure forward toe and pad impulses and closest foot approach"
    )
    args = parser.parse_args()
    if not 0 <= args.warmup < args.seconds or args.noise < 0:
        parser.error("require 0 <= warmup < seconds and noise >= 0")
    digest = hashlib.sha256(TyrannosaurusRexAnatomicalEnv.MODEL_PATH.read_bytes()).hexdigest()
    results = [
        run(
            seed=s,
            mode=args.mode,
            noise=args.noise,
            seconds=args.seconds,
            warmup=args.warmup,
            trace_path=args.trace if i == 0 else None,
            actuators=args.actuators,
            foot_mechanics=args.foot_mechanics,
        )
        for i, s in enumerate(args.seeds)
    ]
    print(
        json.dumps(
            dict(
                mode=args.mode,
                noise=args.noise,
                warmup_s=args.warmup,
                model_xml_sha256=digest,
                mujoco=mujoco.__version__,
                runs=results,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
