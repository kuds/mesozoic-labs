"""Compare fixed camera elevations on matched poses from an actual PPO rollout.

Alternative assemblies are only used by the renderer. Ground truth labels score
perception after the robot-side code returns. No evaluation label enters control.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from environments.shared import robot_camera
from environments.shared.autonomy import CameraGeometry, OnboardFollower, RGBDFrame, controller, perception
from environments.tyrannosaurus_rex.intermediate.env import TyrannosaurusRexIntermediateEnv
from environments.tyrannosaurus_rex.intermediate.inference import load_policy
from examples.tyrannosaurus_rex_60cm.platform import follow_demo, follow_scene

ROOT = Path(__file__).resolve().parents[3]
NOMINAL = ROOT / "examples/tyrannosaurus_rex_60cm/platform/nominal/model.xml"
DATA = ROOT / "hardware/tyrannosaurus_rex_60cm/platform"


def tilt_assembly(scene, destination, elevation):
    """Rotate lens and full-size camera case about fixed optical center."""
    pitch = -np.deg2rad(elevation)
    tree = ET.parse(scene)
    lens = tree.find(".//camera[@name='front_rgbd']")
    case = tree.find(".//geom[@name='oak_d_lite']")
    reserve = tree.find(".//geom[@name='rgbd_reserve']")
    assert lens is not None and case is not None and reserve is not None
    pos = np.fromstring(lens.attrib["pos"], sep=" ")
    c, s = np.cos(pitch), np.sin(pitch)
    center = pos - np.array([c, 0, -s]) * (float(case.attrib["size"].split()[0]) + 0.001)
    lens.set("xyaxes", f"0 -1 0 {s} 0 {c}")
    case.set("euler", f"0 {pitch} 0")
    for geom in (case, reserve):
        geom.set("pos", " ".join(map(str, center)))
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def run(policy, out):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        source = folder / "original.xml"
        follow_scene.build_scene(NOMINAL, source, "obstacle")
        actor = load_policy(policy)
        env = TyrannosaurusRexIntermediateEnv(scene_xml=source, surface="hard", max_episode_steps=1000)
        env.reset(seed=731)
        variants = {}
        for degrees in (12, 6, 0):
            path = folder / f"mount_{degrees}.xml"
            tilt_assembly(source, path, degrees)
            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            camera = robot_camera.RobotCamera(model, "front_rgbd", seed=731)
            camera.set_range_sites(("tof_left", "tof_right"))
            geometry = CameraGeometry(pitch_down_rad=-np.deg2rad(degrees))
            variants[degrees] = (
                model,
                data,
                camera,
                OnboardFollower(geometry),
                [],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        poses = hashlib.sha256()
        pitches = []
        try:
            for i in range(600):
                t = i * env.dt
                follow_scene.move_people(env.model, env.data, t, "obstacle")
                env.set_velocity_command(0.0 if t < 0.6 else 0.12, 0.18 if 4 < t < 7 else 0.0)
                env.set_expression(0.12 * math.sin(t * 1.8), 0.15 * math.sin(t * 1.5))
                if i % 5 == 0:
                    up = env.up
                    roll = math.atan2(up[1], up[2])
                    pitch = math.atan2(-up[0], math.hypot(up[1], up[2]))
                    pitches.append(np.rad2deg(pitch))
                    poses.update(env.data.qpos.tobytes())
                    for model, data, camera, follower, frames, _ in variants.values():
                        data.qpos[:] = env.data.qpos
                        follow_scene.move_people(model, data, t, "obstacle")
                        rgb, depth, tof = camera.capture(data)
                        packet = RGBDFrame(rgb, depth, tof, t, roll, pitch)
                        follower.heartbeat(t + 0.06)
                        follower.update(packet, t + 0.06)
                        snap = follower.snapshot()
                        labels = [follow_demo.selected_identity(model, data, camera, tr) for tr in snap["tracks"]]
                        geometry = follower.geometry
                        z = geometry.lens_height(roll, pitch)
                        mask = geometry.points(np.ones_like(depth), roll, pitch)[2] - z < -z / 5.0
                        valid = np.isfinite(geometry.points(depth, roll, pitch)[0]) & (
                            geometry.points(depth, roll, pitch)[0] > 0.39
                        )
                        qualified = True
                        for cols in np.array_split(np.arange(depth.shape[1]), 3):
                            region = mask[:, cols]
                            qualified &= bool(
                                region.sum() >= max(8, 0.001 * depth.size) and valid[:, cols][region].mean() >= 0.7
                            )
                        frames.append(
                            [
                                round(t, 3),
                                int(qualified),
                                int("visitor_a" in labels),
                                int("visitor_b" in labels),
                                int(qualified and "visitor_a" in labels),
                            ]
                        )
                _, _, term, trunc, _ = env.step(actor(env.observe()))
                if term or trunc:
                    break
        finally:
            env.close()
        comparison = {}
        for degrees, (_, _, _, _, rows, digest) in variants.items():
            comparison[str(degrees)] = {
                "observation_scene_sha256": digest,
                "counts": np.sum(np.asarray(rows)[:, 1:], axis=0).astype(int).tolist(),
                "samples": rows,
            }
        refs = [
            NOMINAL,
            NOMINAL.with_name("components.json"),
            NOMINAL.parent.parent / "design.json",
            Path(__file__),
            Path(robot_camera.__file__),
            Path(perception.__file__),
            Path(controller.__file__),
            Path(perception.__file__).parent / "data/silhouette_v1.json",
            Path(follow_scene.__file__),
            Path(follow_demo.__file__),
        ]
        report = {
            "policy_files": {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in policy.iterdir() if f.is_file()
            },
            "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in refs},
            "scenario": "identical PPO pose sequence on hard floor up to 12s; 0.12 m/s from 0.6s, +0.18 rad/s between 4-7s",
            "qpos_trace_sha256": poses.hexdigest(),
            "samples_10_hz": len(pitches),
            "estimated_nose_down_deg_5_50_95": np.percentile(pitches, [5, 50, 95]).round(2).tolist(),
            "columns": ["s", "floor_observable", "stable_track_a", "stable_track_b", "floor_and_target_a"],
            "elevation_degrees": comparison,
            "limitations": "One person arrangement and PPO trace. Alternative assembly rotates optics and case for rendering only. Bracket loads, COM, failure under different gaits and partial-view recognition in real crowds need validation.",
        }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("Paired frames", len(pitches), {k: v["counts"] for k, v in comparison.items()}, flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, default=DATA / "training/ppo_32768")
    p.add_argument("--output", type=Path, default=DATA / "autonomy/optical_mount.json")
    args = p.parse_args()
    run(args.policy, args.output)
