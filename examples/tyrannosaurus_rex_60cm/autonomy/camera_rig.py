"""Isolate optical navigation with ideal planar velocity tracking. Does not test gait."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import mujoco
import numpy as np

from environments.shared import robot_camera, robot_clearance
from environments.shared.autonomy import CameraGeometry, OnboardFollower, RGBDFrame, controller, perception
from examples.tyrannosaurus_rex_60cm.platform import follow_demo, follow_scene

ROOT = Path(__file__).resolve().parents[3]
DT = 0.1
MODEL = ROOT / "examples/tyrannosaurus_rex_60cm/platform/nominal/model.xml"
OUT = ROOT / "hardware/tyrannosaurus_rex_60cm/platform/autonomy"
CRITERIA = {
    "follow": "Track A for >=80% of frames after selection, no resolved ID switches, >=0.5 m forward progress, final optical distance 1.55–2.30 m.",
    "obstacle": "Follow criteria plus pelvis clears far face of fixture by >=0.35 m and sampled robot-to-fixture geom gap stays positive.",
    "authority": "Selection is issued once using the leftmost stable detection in a two-person image. Start once; never reselect automatically.",
}


def run(scenario, seed, seconds, out, video=False):
    rows = []
    elapsed = []
    identity = None
    wrong_ids = 0
    reviews = 0
    minimum_gap = None
    fixture_x = None
    with tempfile.TemporaryDirectory() as tmp:
        scene = Path(tmp) / "scene.xml"
        follow_scene.build_scene(MODEL, scene, scenario)
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        camera = robot_camera.RobotCamera(model, "front_rgbd", seed=seed)
        camera.set_range_sites(("tof_left", "tof_right"))
        target_controller = OnboardFollower(CameraGeometry(fovy_deg=camera.fovy_deg))
        yaw = 0.0
        start = float(data.qpos[0])
        picked = None
        fixture = model.geom("gallery_fixture").id if scenario == "obstacle" else None
        robot_geoms = [
            i
            for i in range(model.ngeom)
            if model.body_rootid[model.geom_bodyid[i]] == model.body("pelvis").id
            and model.geom_group[i] in (0, 1, 2, 3, 4)
        ]
        writer = None
        if video:
            import imageio_ffmpeg

            writer = imageio_ffmpeg.write_frames(
                str(out / (scenario + "_camera_rig.mp4")),
                (640, 488),
                fps=10,
                codec="libx264",
                quality=7,
                macro_block_size=1,
            )
            writer.send(None)
        try:
            for i in range(round(seconds / DT)):
                t = i * DT
                follow_scene.move_people(model, data, t, scenario)
                rgb, depth, tof = camera.capture(data)
                frame = RGBDFrame(rgb, depth, tof, t)
                target_controller.heartbeat(t + 0.06)
                before = time.perf_counter()
                target_controller.update(frame, t + 0.06)
                elapsed.append((time.perf_counter() - before) * 1e3)
                s = target_controller.snapshot()
                if picked is None and len(s["tracks"]) >= 2:
                    track = min(s["tracks"], key=lambda tr: tr["box"][0])
                    target_controller.select(track["id"], t + 0.06)
                    target_controller.start(t + 0.06)
                    picked = t
                s = target_controller.snapshot()
                track = next((tr for tr in s["tracks"] if tr["id"] == s["selected"]), None)
                # This branch only scores the output; it cannot change the command.
                label = follow_demo.selected_identity(model, data, camera, track) if track else None
                if label and label != "unresolved":
                    identity = identity or label
                    wrong_ids += label != identity
                    reviews += 1
                if fixture is not None:
                    gap = min(
                        robot_clearance.clearance_distance(
                            model, data, j, fixture, required_clearance_m=0.003, search_distance_m=2.0
                        )["min_distance_m"]
                        for j in robot_geoms
                    )
                    minimum_gap = gap if minimum_gap is None else min(minimum_gap, gap)
                    fixture_x = float(data.geom_xpos[fixture, 0] + model.geom_size[fixture, 0])
                command = target_controller.watchdog(t + 0.06)
                rows.append(
                    {
                        "s": round(t, 2),
                        "xy": data.qpos[:2].round(3).tolist(),
                        "state": command.state,
                        "vx": round(command.forward_mps, 3),
                        "wz": round(command.yaw_rps, 3),
                        "tracks": len(s["tracks"]),
                        "id": s["selected"],
                        "identity_score": label,
                        "range_m": track["forward_m"] if track else None,
                    }
                )
                if writer:
                    writer.send(
                        follow_demo.annotated(rgb, s, t, "Kinematic camera rig", scenario, float(data.qpos[0] - start))
                    )
                # Ideal unicycle actuator. Roll/pitch/heave and leg angles remain in home pose.
                mid = yaw + 0.5 * DT * command.yaw_rps
                data.qpos[:2] += DT * command.forward_mps * np.array([math.cos(mid), math.sin(mid)])
                yaw += DT * command.yaw_rps
                data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        finally:
            if writer:
                writer.close()
    progress = float(data.qpos[0] - start)
    post = [r for r in rows if picked is not None and r["s"] >= picked]
    seen = sum(r["range_m"] is not None for r in post) / max(1, len(post))
    end = rows[-1]["range_m"]
    passed = bool(
        identity == "visitor_a"
        and seen >= 0.80
        and reviews > 0
        and wrong_ids == 0
        and progress >= 0.5
        and end is not None
        and 1.55 <= end <= 2.30
    )
    if fixture_x is not None:
        passed = passed and data.qpos[0] >= fixture_x + 0.35 and minimum_gap is not None and minimum_gap > 0
    sources = [
        MODEL,
        MODEL.with_name("components.json"),
        MODEL.parent.parent / "design.json",
        Path(controller.__file__),
        Path(perception.__file__),
        Path(robot_camera.__file__),
        Path(robot_clearance.__file__),
        Path(follow_scene.__file__),
        Path(follow_demo.__file__),
        Path(__file__),
        Path(perception.__file__).parent / "data/silhouette_v1.json",
    ]
    report = {
        "scenario": scenario,
        "mode": "ideal planar velocity camera rig, fixed leg pose",
        "criteria": CRITERIA,
        "seed": seed,
        "duration_s": seconds,
        "capture_hz": 10,
        "packet_age_ms": 60,
        "selected_at_s": picked,
        "resolved_selected_identity": identity,
        "visible_fraction_after_selection": round(seen, 4),
        "forward_progress_m": round(progress, 3),
        "last_optical_range_m": end,
        "resolved_id_mismatches": wrong_ids,
        "resolved_frames": reviews,
        "minimum_fixture_gap_or_certified_bound_m_sampled_10hz": None if minimum_gap is None else round(minimum_gap, 4),
        "sensor_to_command_ms_excluding_render_p50_p95": np.percentile(elapsed, [50, 95]).round(2).tolist(),
        "state_frames": dict(Counter(r["state"] for r in rows)),
        "passed": bool(passed),
        "versions": {"opencv": cv2.__version__, "mujoco": mujoco.__version__, "numpy": np.__version__},
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "frames": rows,
    }
    if video:
        path = out / (scenario + "_camera_rig.mp4")
        report["video_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / (scenario + "_camera_rig.json")).write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k not in {"frames", "sha256", "criteria"}}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--duration", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=412)
    p.add_argument("--video", action="store_true")
    p.add_argument("--scenario", choices=["follow", "obstacle", "both"], default="both")
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    for name in ("follow", "obstacle") if args.scenario == "both" else (args.scenario,):
        run(name, args.seed, args.duration, args.output, args.video)
