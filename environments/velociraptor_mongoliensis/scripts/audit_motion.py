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

from environments.shared.gait.biped_metrics import analyse, static_margin
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import SubstepContactRecorder
from environments.shared.model_id import ModelId
from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

MODEL = ModelId("velociraptor_mongoliensis", "anatomical")


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


def run(args, seed):
    class ProbeEnv(VelociraptorMongoliensisAnatomicalEnv):
        MODEL_PATH = args.xml or VelociraptorMongoliensisAnatomicalEnv.MODEL_PATH

    env = ProbeEnv(reset_noise_scale=args.noise)
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
            result.update(analyse(trace, per_substep, start, env.model, pad_suffix="mtp_pad"))
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
