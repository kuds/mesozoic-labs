"""Rendered observation-to-command experiments, with optional free-base PPO.

A fixed-pose perception bench and an unassisted free-base policy rollout share
the renderer, operator protocol and robot-side software. Their reports explicitly
declare which loop was closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import xml.etree.ElementTree as ET
import time
from collections import Counter, deque
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.autonomy import CameraGeometry, OnboardFollower, OperatorServer, RGBDFrame
from environments.shared.robot_camera import RobotCamera
from examples.tyrannosaurus_rex_60cm.platform.follow_scene import build_scene, move_people

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "nominal" / "model.xml"
SCENARIOS = ("follow", "obstacle", "crossing", "occlusion", "loss", "blocked", "disconnect")
CAPTURE_PERIOD = 0.1
FRAME_DELAY = 0.06


def digest(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def witness_pose():
    # Fixed side view from inside the gallery wall. This camera is for the
    # audience, and is never connected to the onboard process.
    eye = np.array([.85, -3.05, 1.16])
    target = np.array([1.68, .1, .70])
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0., 0., 1.]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    return eye, np.column_stack((right, up, backward))


def annotated(frame, status, t, mode, scenario, distance):
    """Camera-only recording for the detached kinematic rig."""
    from PIL import Image, ImageDraw, ImageFont
    picture = Image.fromarray(frame).resize((640,400),Image.Resampling.BILINEAR)
    board = Image.new("RGB",(640,488),"#172324")
    board.paste(picture,(0,0))
    pen = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=16)
    for track in status["tracks"]:
        box = tuple(2*p for p in track["box"])
        color = "#63F9AB" if track["id"] == status["selected"] else "#F4D494"
        pen.rectangle(box,outline=color,width=3)
        pen.text((box[0]+3,box[1]+2), f'Person {track["id"]}',fill=color,stroke_width=2,stroke_fill="#172324",font=font)
    pen.text((12,407),f'{mode}  |  {scenario}  |  {t:05.1f} s',fill="white",font=font)
    pen.text((12,432),f'{status["state"]}    requested {status["forward_mps"]:+.2f} m/s, {status["yaw_rps"]:+.2f} rad/s',fill="#9AE9C4",font=font)
    pen.text((12,457),f'Base displacement {distance:.2f} m    |    Camera playback 1x',fill="white",font=font)
    return np.asarray(board)


def annotated_witness(scene_image, frame, status, t, mode, scenario, distance):
    from PIL import Image, ImageDraw, ImageFont
    board = Image.new("RGB", (960, 448), "#172324")
    scene = Image.fromarray(scene_image).resize((608,380), Image.Resampling.BILINEAR)
    onboard = Image.fromarray(frame).resize((352,220), Image.Resampling.BILINEAR)
    board.paste(scene, (0, 0))
    board.paste(onboard, (608, 0))
    pen = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=17)
    for track in status["tracks"]:
        x0,y0,x1,y1 = track["box"]
        box = (608+1.1*x0,1.1*y0,608+1.1*x1,1.1*y1)
        color = "#63F9AB" if track["id"] == status["selected"] else "#F4D494"
        pen.rectangle(box, outline=color, width=2)
        pen.text((box[0]+2,box[1]+2), f'P{track["id"]}', fill=color, stroke_width=1, stroke_fill="#172324",font=font)
    pen.text((621,232),"TORSO CAMERA  /  10 Hz", fill="#BBBBB0",font=font)
    pen.text((621,260),status["state"].replace("_"," "), fill="#9AE9C4",font=ImageFont.load_default(size=24))
    pen.text((621,308),f'Requested {status["forward_mps"]:+.2f} m/s', fill="white",font=font)
    pen.text((621,334),f'Yaw {status["yaw_rps"]:+.2f} rad/s', fill="white",font=font)
    pen.text((12,390),f'{mode}  |  {scenario}  |  {t:05.1f} s at 1x', fill="white",font=font)
    pen.text((12,416),f'Base displacement {distance:.2f} m',fill="#9AE9C4",font=font)
    return np.asarray(board)


def selected_identity(model, data, camera, track):
    """Scoring only: compare the visible track center/range to projected people.

    A mismatch is a warning for manual inspection. Near-overlap is ambiguous.
    Ground truth never enters the detector, planner, selection or policy.
    """
    scores = []
    for name in ("visitor_a", "visitor_b"):
        center = data.xpos[model.body(name).id] + np.array([0,0,0.94])
        local = data.cam_xmat[camera.id].reshape(3,3).T @ (center-data.cam_xpos[camera.id])
        z = -float(local[2])
        if z <= 0:
            continue
        u = camera.width/2 + camera.focal*local[0]/z
        box = track["box"]
        score = abs(u - (box[0]+box[2])/2) / max(4, box[2]-box[0])
        score += abs(track["forward_m"]-z)/.55
        scores.append((score,name))
    scores.sort()
    if not scores or scores[0][0]>1.5 or (len(scores)>1 and scores[1][0]-scores[0][0]<.35):
        return "unresolved"
    return scores[0][1]


def run(args):
    import imageio_ffmpeg
    # Both mode and scene are explicit dependencies of every published report.
    from environments.shared.autonomy import controller, perception
    from environments.shared import robot_camera
    from examples.tyrannosaurus_rex_60cm.platform import follow_scene
    source = SOURCE
    mode = "PPO free base" if args.policy else "Fixed camera bench"
    args.output.mkdir(parents=True,exist_ok=True)
    weights = Path(perception.__file__).parent / "data/silhouette_v1.json"
    report = {"mode":mode,"scenario":args.scenario,"seed":args.seed,"surface":args.surface,
              "camera_hz":10, "capture_to_delivery_ms":60, "initial_selection":"manual" if args.web else "leftmost stable detection with two people visible",
              "criteria":"A successful integrated detour requires selection, no fall/contact or silent switch, passing the obstacle and forward base progress. A fixed bench cannot pass.",
              "sha256":{p.name:digest(p) for p in (source, HERE/"nominal/components.json", HERE/"design.json", Path(__file__),Path(robot_camera.__file__), Path(follow_scene.__file__), Path(perception.__file__), Path(controller.__file__),weights)}}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        scene = Path(tmp) / "gallery.xml"
        build_scene(source,scene,args.scenario)
        if args.room_depth_m != 10.0:
            layout = ET.parse(scene)
            wall = layout.find(".//geom[@name='far_wall']")
            wall.set("pos", f"{args.room_depth_m} 0 1.45")
            layout.write(scene, encoding="utf-8", xml_declaration=True)
        report["room_depth_m"] = args.room_depth_m
        report["sha256"]["gallery.xml"] = digest(scene)
        env = None
        if args.policy:
            from environments.tyrannosaurus_rex.intermediate.env import TyrannosaurusRexIntermediateEnv
            from environments.tyrannosaurus_rex.intermediate.inference import load_policy
            if digest(source) != digest(TyrannosaurusRexIntermediateEnv.MODEL_PATH):
                raise ValueError("Gallery and trained plant start from different XML bytes")
            actor = load_policy(args.policy)
            env = TyrannosaurusRexIntermediateEnv(scene_xml=scene, surface=args.surface, max_episode_steps=math.ceil(args.duration*50)+2)
            env.reset(seed=args.seed)
            env.set_velocity_command(0,0)
            model,data = env.model,env.data
            dt = env.dt
            report["policy_files"] = {p.name:digest(p) for p in sorted(args.policy.iterdir()) if p.is_file() and p.suffix in (".zip",".pkl",".json")}
        else:
            model = mujoco.MjModel.from_xml_path(str(scene))
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model,data,model.key("home").id)
            dt = .02
        move_people(model,data,0,args.scenario)
        camera = RobotCamera(model,"front_rgbd",seed=args.seed)
        camera.set_range_sites(("tof_left","tof_right"))
        follower = OnboardFollower(CameraGeometry(fovy_deg=camera.fovy_deg))
        video_path = args.output / (args.scenario + ".mp4")
        witness = RobotCamera(model, "front_rgbd", 384, 240) if args.video else None
        eye, basis = witness_pose()
        writer = imageio_ffmpeg.write_frames(str(video_path),(960,448),fps=10,codec="libx264",quality=7,macro_block_size=1) if args.video else None
        if writer:
            writer.send(None)
        queue = deque()
        t, picked, ended, next_capture = 0.,False,"time_limit",0.
        server = OperatorServer(follower,port=args.port,clock=lambda:t).start() if args.web else None
        if server:
            print(f"Operator view at http://127.0.0.1:{args.port}",flush=True)
        start = time.monotonic()
        root_id = model.body("pelvis").id
        initial_pos = data.xpos[root_id,:2].copy()
        first_identity = None
        obstacle_contact = False
        robot_geoms = model.body_rootid[model.geom_bodyid] == root_id
        scene_geoms = np.array([model.geom(i).name.startswith(("visitor_", "gallery_", "far_wall")) for i in range(model.ngeom)])
        def inspect_contacts():
            nonlocal obstacle_contact
            if not obstacle_contact and data.ncon:
                a, b = data.contact.geom.T
                obstacle_contact = bool(np.any((robot_geoms[a] & scene_geoms[b]) | (robot_geoms[b] & scene_geoms[a])))
        if env:
            env.substep_callback = inspect_contacts
        max_forward = 0.
        try:
            for step in range(math.ceil(args.duration/dt)):
                t = step*dt
                if args.web:
                    time.sleep(max(0., start+t-time.monotonic()))
                elif args.scenario!="disconnect" or t<3.0:
                    follower.heartbeat(t)
                move_people(model,data,t,args.scenario)
                if t+1e-8 >= next_capture:
                    rgb,depth,tof = camera.capture(data)
                    # Actor-side IMU estimator used for gravity alignment. A test
                    # tripod has zero roll and pitch. No simulator orientation truth.
                    up = env.up if env else np.array([0,0,1])
                    roll = math.atan2(up[1],up[2])
                    pitch = math.atan2(-up[0], math.hypot(up[1],up[2]))
                    packet = RGBDFrame(rgb,depth,tof,t,roll,pitch)
                    scene_image = witness._render_view(data,eye,basis)[0] if witness else None
                    queue.append((t+FRAME_DELAY,packet,scene_image))
                    next_capture += CAPTURE_PERIOD
                if queue and queue[0][0] <= t+1e-8:
                    _,frame,scene_image = queue.popleft()
                    follower.update(frame,t)
                    current = follower.snapshot()
                    if not args.web and not picked and len(current["tracks"]) >= 2:
                        first = min(current["tracks"],key=lambda tr:tr["box"][0])
                        follower.select(first["id"],t)
                        follower.start(t)
                        picked = True
                    current = follower.snapshot()
                    selected = next((tr for tr in current["tracks"] if tr["id"]==current["selected"]),None)
                    identity = selected_identity(model,data,camera,selected) if selected else None
                    if first_identity is None and identity in ("visitor_a","visitor_b"):
                        first_identity = identity
                    displacement = float(np.linalg.norm(data.xpos[root_id,:2]-initial_pos))
                    rows.append({"s":round(t,3),"state":current["state"],"selected":current["selected"],
                                 "visible_ids":[tr["id"] for tr in current["tracks"]],"identity_score":identity,
                                 "target_distance_m":selected["forward_m"] if selected else None,
                                 "vx_cmd":current["forward_mps"],"wz_cmd":current["yaw_rps"],
                                 "base_xy_m":(data.xpos[root_id,:2]-initial_pos).round(4).tolist()})
                    if server:
                        server.publish(frame.rgb)
                    if writer:
                        writer.send(annotated_witness(scene_image,frame.rgb,current,t,mode,args.scenario,displacement))
                cmd = follower.watchdog(t)
                if env:
                    env.set_velocity_command(cmd.forward_mps, cmd.yaw_rps)
                    env.set_expression(.12*math.sin(t*1.8),.15*math.sin(t*1.5))
                    _,_,term,trunc,_ = env.step(actor(env.observe()))
                    max_forward = max(max_forward,float(data.xpos[root_id,0]-initial_pos[0]))
                    if term or trunc:
                        ended = "environment_terminated" if term else "truncated"
                        break
        finally:
            if writer:
                writer.close()
            if server:
                server.close()
            if env:
                env.close()
        # Only a free-base physical trajectory can establish a detour.
        end_xy = (data.xpos[root_id,:2]-initial_pos).tolist()
        switch_frames = sum(r["identity_score"] in ("visitor_a","visitor_b") and r["identity_score"]!=first_identity for r in rows)
        report.update(seconds=round((step+1)*dt,3),end_reason=ended,selected_once=any(r["selected"] is not None for r in rows),
                      states=dict(Counter(r["state"] for r in rows)),matched_initial_identity=first_identity,
                      evaluation_identity_mismatch_frames=switch_frames,max_base_advance_m=max_forward, final_base_xy_m=end_xy,
                      obstacle_or_visitor_contact=obstacle_contact if env else None,
                      integrated_detour_pass=bool(env and args.scenario=="obstacle" and first_identity and switch_frames==0 and
                                                  ended=="time_limit" and not obstacle_contact and end_xy[0]>2.25 and
                                                  rows and rows[-1]["state"] in ("following","at_distance")),
                      frames=rows)
        if writer:
            report["sha256"][video_path.name]=digest(video_path)
    out = args.output / (args.scenario + ".json")
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(out,report["seconds"],report["states"],report["end_reason"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--policy",type=Path,help="Directory with a verified PPO and normalization pair")
    modes.add_argument("--fixed-bench",action="store_true",help="Static torso; perception and command-level tests only")
    p.add_argument("--scenario",choices=SCENARIOS,default="obstacle")
    p.add_argument("--room-depth-m", type=float, choices=(6.,10.), default=10., help="Visual background depth sensitivity in metres")
    p.add_argument("--duration",type=float,default=16.)
    p.add_argument("--seed",type=int,default=731)
    p.add_argument("--surface",choices=("hard","carpet"),default="hard")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--video",action="store_true")
    p.add_argument("--web",action="store_true")
    p.add_argument("--port",type=int,default=8765)
    args=p.parse_args()
    if not 0.2 <= args.duration <= 1800:
        p.error("duration must be between 0.2 and 1800 s")
    run(args)


if __name__ == "__main__":
    main()
