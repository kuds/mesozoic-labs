"""Record the standing trials and 1 kHz traces used in the capsule gait study."""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco as mj
import numpy as np

BUILD = Path(__file__).resolve().parent.parent
CASES = [
    ("clip", 0.25, 101, None, None),
    ("slow", 0.15, 101, None, None),
    ("J1", 0.25, 401, 1.0, None),
    ("J4", 0.25, 401, 0.4, 10.0),
]


def model_for(build):
    return build / "sim/generated/trike60_C_closed_capsule.xml"


def static(build, out):
    sys.path.insert(0, str(build / "sim"))
    import check_stand as cs
    import feet60_study as feet

    original = cs.GEN
    try:
        cs.GEN = feet._View(model_for(build))
        rows = [cs.run("60cm", "closed", seed) for seed in (0, 3, 4)]
    finally:
        cs.GEN = original
    (out / "static.json").write_text(json.dumps(rows, indent=2) + "\n")


def record(build, out, name, command, seed, mu, impratio):
    sys.path.insert(0, str(build / "walk"))
    import walk60 as walk

    model = model_for(build)
    settings = {
        "model_path": str(model),
        "domain_randomization": mu is not None,
        "observation_noise": True,
    }
    if mu is not None:
        settings["friction_range"] = (mu, mu)
    e = walk.TrikeEnv(walk.plant_config(**settings))
    if impratio is not None:
        e.model.opt.impratio = impratio
    c = walk.TrotController(e, walk.WalkParams())
    m, d = e.model, e.data
    obs, _ = e.reset(seed=seed, options={"command_speed": command, "delay_steps": 10})
    nt = round(e.cfg.episode_seconds / e.physics_dt) + e.cfg.frame_skip
    spec = {
        "t": (),
        "F": (4,),
        "touch": (4,),
        "com": (3,),
        "body": (3,),
        "head": (3,),
        "xmat": (3, 3),
        "sole": (4, 3),
        "solemat": (4, 9),
        "planned": (4,),
        "angles": (4,),
        "qpos": (m.nq,),
        "tau": (12,),
        "rpm": (12,),
        "cop": (2,),
    }
    a = {k: np.zeros((nt,) + sh) for k, sh in spec.items()}
    ids = {s: j for j, s in enumerate(e._soles)}
    force = np.empty(6)
    counter = 0

    def hook(env):
        nonlocal counter
        i = counter
        counter += 1
        a["t"][i] = d.time
        a["com"][i] = d.subtree_com[e._chassis]
        a["head"][i] = d.xipos[m.body("head_yaw").id]
        a["body"][i] = d.xpos[e._chassis]
        a["xmat"][i] = d.xmat[e._chassis].reshape(3, 3)
        a["tau"][i] = d.actuator_force[e._leg_act]
        a["rpm"][i] = d.actuator_velocity[e._leg_act] * 60 / (2 * np.pi)
        a["sole"][i] = d.geom_xpos[e._soles]
        a["solemat"][i] = d.geom_xmat[e._soles]
        a["qpos"][i] = d.qpos
        a["angles"][i] = d.qpos[e._ankle_qadr]
        a["touch"][i] = d.sensordata[e._touch_adr]
        a["planned"][i] = [not c.in_swing[k] for k in walk.CORNERS]
        for k in range(d.ncon):
            con = d.contact[k]
            id1, id2 = con.geom
            if id1 == e._floor and id2 in ids:
                j = ids[id2]
            elif id2 == e._floor and id1 in ids:
                j = ids[id1]
            else:
                continue
            mj.mj_contactForce(m, d, k, force)
            a["F"][i, j] += force[0]
            a["cop"][i] += force[0] * con.pos[:2]
        total = a["F"][i].sum()
        a["cop"][i] = a["cop"][i] / total if total else np.nan

    e.substep_hook = hook
    while True:
        obs, _reward, term, trunc, info = e.step(c.act(obs))
        if term or trunc:
            break
    assert not term, info
    a = {k: v[:counter] for k, v in a.items()}
    a.update(
        sole_size=m.geom_size[e._soles[0]],
        mass=m.body_mass.sum(),
        weight=-m.opt.gravity[2] * m.body_mass.sum(),
        command=command,
        seed=seed,
        heading=e._heading,
    )
    np.savez_compressed(out / f"{name}.npz", **a)
    e.close()
    print(name, counter, float(info["episode_forward_speed"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=BUILD)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory for static.json and the four transient NPZ traces",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build = args.build.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    static(build, out)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(record, build, out, *case) for case in CASES]:
            future.result()


if __name__ == "__main__":
    main()
