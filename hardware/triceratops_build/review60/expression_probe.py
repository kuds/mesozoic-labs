"""Compare mild head/tail cues and a declared excess-shell mass on the Option C plant.

Run with --build pointing to hardware/triceratops_build. Results are exploratory, for
the nominal capsule robot in MuJoCo. The underlying controller is the repository's.
"""

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

AXES = ["neck_pitch", "neck_yaw", "tail_yaw", "tail_pitch"]
SEEDS = [101, 102, 103]


def ramp(t, start, rise, hold, end):
    def smooth(x):
        u = np.clip(x, 0, 1)
        return u**3 * (10 - 15 * u + 6 * u * u)

    return smooth((t - start) / rise) - smooth((t - hold) / end)


def targets(t):
    return np.deg2rad(
        [
            5 * ramp(t, 4.1, 0.60, 5.05, 0.75),
            12 * ramp(t, 2.8, 0.9, 6.25, 1.05),
            -4 * ramp(t, 3.15, 1.0, 6.35, 1.35),
            2 * ramp(t, 4.2, 0.8, 5.7, 1.1),
        ]
    )


def extra_shell(source, dest):
    xml = ET.parse(source)
    extras = [
        ("chassis", (0, 0, 0.05), 0.18),
        ("head_yaw", (0.09, 0, 0.025), 0.06),
        ("tail", (-0.20, 0, 0), 0.04),
    ]
    for end in ["left_fore", "right_fore", "left_hind", "right_hind"]:
        extras.append((end + "_lower_link", (0, 0, -0.09), 0.03))
    for name, pos, kg in extras:
        body = xml.find(f".//body[@name='{name}']")
        if body is None:
            raise ValueError(name)
        ET.SubElement(
            body,
            "geom",
            name="study_extra_" + name,
            type="sphere",
            pos=" ".join(map(str, pos)),
            size=".01",
            mass=str(kg),
            rgba="0 0 0 0",
            contype="0",
            conaffinity="0",
        )
    xml.write(dest)


def trial(walk, model, cue, seed):
    env = walk.TrikeEnv(walk.plant_config(model_path=str(model), domain_randomization=False, observation_noise=True))
    obs, _ = env.reset(seed=seed, options={"command_speed": 0.25, "delay_steps": 10})
    ctrl = walk.TrotController(env, walk.WalkParams())
    m, d = env.model, env.data
    act_ids = [m.actuator(n).id for n in AXES]
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in AXES]
    root = m.body("chassis").id
    heading = env._heading
    fwd = np.array([np.cos(heading), np.sin(heading)])
    sideways = np.array([-fwd[1], fwd[0]])
    poses = []
    forces = []
    xy = []
    torque_squares = []
    contacts = []

    def record(_):
        if d.time >= 2:
            poses.append(np.rad2deg(d.qpos[qadr]).copy())
            forces.append(d.actuator_force[act_ids].copy())
            xy.append(d.xpos[root, :2].copy())
            torque_squares.append(d.actuator_force[env._leg_act].copy() ** 2)
            contacts.append(d.sensordata[env._touch_adr].copy())

    env.substep_hook = record
    cause = ""
    while d.time < 12 - 1e-8:
        if cue:
            d.ctrl[act_ids] = targets(d.time)
        obs, _, terminated, truncated, info = env.step(ctrl.act(obs))
        if terminated or truncated:
            cause = info.get("termination", "")
            break
    pos = np.array(poses)
    tq = np.array(forces)
    weights = np.sum(contacts, axis=0)
    dist = xy[-1] - xy[0]
    leg_cont = 1 / np.sqrt(env._inv_cont2)
    joint_rms = np.sqrt(np.mean(torque_squares, axis=0)) / leg_cont
    result = {
        "seed": seed,
        "cue": cue,
        "mass_kg": float(m.body_mass.sum()),
        "termination": cause,
        "sim_end_s": round(d.time, 5),
        "mean_speed_mps": float(dist @ fwd / (len(xy) * env.physics_dt)),
        "lateral_m": float(dist @ sideways),
        "front_load_share": float(weights[:2].sum() / weights.sum()),
        "max_leg_rms_ratio": float(max(joint_rms)),
        "leg_rms_driver": m.actuator(env._leg_act[np.argmax(joint_rms)]).name,
        "face_q_deg_range": np.array([pos.min(axis=0), pos.max(axis=0)]).T.tolist(),
        "face_rms_nm": np.sqrt((tq**2).mean(axis=0)).tolist(),
        "face_peak_abs_nm": np.max(abs(tq), axis=0).tolist(),
    }

    env.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    sys.path.insert(0, str(build / "walk"))
    import walk60

    src = build / "sim/generated/trike60_C_closed_capsule.xml"
    rows = []
    with tempfile.TemporaryDirectory() as td:
        weighted = Path(td) / "trike60_C_closed_capsule_extra.xml"
        extra_shell(src, weighted)
        for seed in SEEDS:
            for variant in (src, weighted):
                for cue in (False, True):
                    row = trial(walk60, variant, cue, seed)
                    rows.append(row)
                    print(
                        seed,
                        round(row["mass_kg"], 3),
                        cue,
                        round(row["mean_speed_mps"], 4),
                        flush=True,
                    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"mujoco": mujoco.__version__, "axes": AXES, "rows": rows}, indent=2))
