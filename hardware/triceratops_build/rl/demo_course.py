"""Heat, battery and loads over a museum demo, for the 60 cm build (decision D21). Simulation plus ESTIMATES.

1. **Segments, simulated.** One 20 s episode per activity (walking at 0.25 m/s, walking at 0.15 m/s, standing),
   nominal model, sensor noise on, servo delay 10 ms, flat floor, measured from 2 s. Each records, every 1 ms physics
   step: every leg drive's torque and speed, the electrical estimate (``analytics/gait_core.electrical_power``: copper
   loss 3 R (tau_drive / Kt)^2, the drives' net draw on a shared bus, the braking surplus), the joint bearing loads
   and the foot forces (``telemetry/loads.py``).
2. **The demo, modelled.** A loop of segments (default: 90 s walking at 0.25 m/s, 30 s standing) repeated for
   ``--minutes`` (default 20):
   - winding temperatures per drive (``telemetry/thermal.py``), calibrated from the maker's temperature-rise
     figures, for thermal time constants of 10, 20 and 40 min (not published: an assumption, reported as a range);
   - battery runtime and pack currents (``telemetry/power.py``), from the pack on record (146.5 Wh, 80 % usable,
     12S at 42.0-49.2 V in use; 45.6 V taken as nominal, 42.0 V as low charge) and a 30 W hotel load.
   The loads are per segment, the same rigid-model loads at any demo length.

The walker is the frozen model-based trot (``walk/walk60.py``) on the capsule-foot model by default, or a checkpoint
(``--ckpt-dir``; safe loaders as ``steer_eval.py``). Outputs ``demo_course.json`` and ``demo_temperatures.png``
(SIMULATION-labelled).

Usage: python hardware/triceratops_build/rl/demo_course.py --out DIR [--ckpt-dir DIR] [--minutes 20]
       [--walk-s 90 --stand-s 30] [--seed 700000]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / "analysis", HERE / "analytics", HERE.parent / "walk"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SEGMENTS = {"walk_0.25": 0.25, "walk_0.15": 0.15, "stand": 0.0}
PACK = {"pack_wh": 146.5, "usable_share": 0.80, "v_nominal": 45.6, "v_low": 42.0, "hotel_w": 30.0}
TIME_CONSTANTS_S = (600.0, 1200.0, 2400.0)
SEED = 700_000
WINDOW_S = 2.0


def trot_factory():
    import walk60
    import trike_env as te

    def make(command):
        env = te.TrikeEnv(walk60.plant_config(model_path=str(te.CAPSULE_MODEL), domain_randomization=False,
                                              observation_noise=True))
        ctrl = walk60.TrotController(env, walk60.WalkParams())
        return env, ctrl.act
    return make, "model-based trot (walk/walk60.py, frozen) on the capsule-foot model"


def policy_factory(ckpt_dir: Path):
    import steer_eval
    import trike_env as te

    ec = json.loads((ckpt_dir / "run_record.json").read_text())["settings"]["env_config"]
    cfg = te.flat_floor(te.TrikeEnvConfig.from_dict({**ec, "model_path": str(te.MODEL_DIR /
                                                                             Path(ec["model_path"]).name)}))
    cfg = cfg.replace(domain_randomization=False, observation_noise=True, delay_steps_fixed=10)
    probe = te.TrikeEnv(cfg)
    act, name = steer_eval.load_policy(ckpt_dir, probe)

    def make(command):
        return te.TrikeEnv(cfg), act
    return make, f"policy {name} ({ckpt_dir.name})"


def measure_segment(make, command: float, seed: int, seconds: float = 20.0) -> dict:
    import gait_core as gc
    import gait_report as gr
    import trike_env as te
    from telemetry.loads import LoadRecorder

    env, act = make(command)
    env.cfg = env.cfg.replace(episode_seconds=seconds)
    env.max_episode_steps = int(round(seconds / env.dt))
    m = env.model
    idx = np.asarray(env._leg_act, dtype=np.int64)
    names = [m.actuator(int(i)).name for i in idx]
    jn = [m.joint(int(m.actuator(int(i)).trnid[0])).name for i in idx]
    drives = gr.drive_table()
    groups = [gr.gp.group_of(n) for n in names]
    eta = np.array([drives[g]["transmission_efficiency"] for g in groups])
    kt = np.array([drives[g]["kt_nm_per_a_rms"] for g in groups])
    res = np.array([drives[g]["phase_resistance_ohm"] for g in groups])
    rec = LoadRecorder(m, jn, [f"{c}_sole" for c in te.CORNERS])
    log = {"t": [], "tau": [], "om": []}

    def hook(e):
        d = e.data
        log["t"].append(d.time)
        log["tau"].append(d.actuator_force[idx].copy())
        log["om"].append(d.actuator_velocity[idx].copy())
        if d.time >= WINDOW_S:
            rec(e.model, d)

    obs, _ = env.reset(seed=seed, options={"command_speed": command, "delay_steps": 10})
    env.substep_hook = hook
    fell, term = False, ""
    try:
        while True:
            obs, _, terminated, truncated, info = env.step(act(obs))
            if terminated or truncated:
                fell, term = bool(terminated), info["termination"]
                break
    finally:
        env.substep_hook = None
    t = np.asarray(log["t"])
    w = t >= WINDOW_S
    tau, om = np.asarray(log["tau"])[w], np.asarray(log["om"])[w]
    el = gc.electrical_power(tau, om, eta, kt, res)
    p = tau * om
    tau_drive = np.where(p >= 0, tau / eta, tau * eta)
    net = (el["drive_mech"] + el["copper"]).sum(1)
    return {"command_mps": command, "fell": fell, "termination": term, "seconds_measured": float(w.sum() * env.physics_dt),
            "joints": names, "groups": groups,
            "tau_drive_sq_mean": (tau_drive ** 2).mean(0).tolist(), "tau_drive_rms": np.sqrt((tau_drive ** 2).mean(0)).tolist(),
            "copper_w": el["copper"].mean(0).tolist(),
            "mean_draw_w": float(el["shared_bus"].mean()), "peak_draw_w": float(el["shared_bus"].max()),
            "peak_surplus_w": float(max(0.0, -net.min())), "surplus_share": float((net < 0).mean()),
            "loads": rec.summary(env.nominal_weight)}


def run(make, label: str, loop, minutes: float, seed: int, segment_s: float = 20.0) -> dict:
    import gait_report as gr
    from telemetry.power import demo_battery
    from telemetry.thermal import DriveThermal, demo_temperatures

    drives_json = json.loads((HERE.parent / "sim" / "data" / "drives.json").read_text())["leg_drives"]
    drives = gr.drive_table()
    segs = {name: measure_segment(make, SEGMENTS[name], seed + i, segment_s) for i, name in enumerate(SEGMENTS)}
    # the demo: the loop repeated for `minutes`
    seq, total = [], 0.0
    while total < minutes * 60.0 - 1e-9:
        for name, s in loop:
            take = min(s, minutes * 60.0 - total)
            if take <= 0:
                break
            seq.append((name, take))
            total += take
    joints, groups = segs["stand"]["joints"], segs["stand"]["groups"]
    therm, per_joint = {}, {}
    for j, (jn, g) in enumerate(zip(joints, groups)):
        dd = drives_json[drives[g]["drive"]]
        therm[jn] = DriveThermal(jn, drives[g]["kt_nm_per_a_rms"], drives[g]["phase_resistance_ohm"],
                                 float(dd["thermal_eq_nm"]))
        per_joint[jn] = [(sec, segs[name]["tau_drive_sq_mean"][j]) for name, sec in seq]
    temps = demo_temperatures(therm, per_joint, TIME_CONSTANTS_S)
    battery = demo_battery([{"seconds": sec, **{k: segs[name][k] for k in ("mean_draw_w", "peak_draw_w",
                                                                            "peak_surplus_w", "surplus_share")}}
                            for name, sec in seq], **PACK)
    # temperature traces of the hottest joint per drive group (20 min time constant), for the chart
    traces = {}
    for g in dict.fromkeys(groups):
        jg = [jn for jn, gg in zip(joints, groups) if gg == g]
        hot = max(jg, key=lambda jn: temps[jn]["20min"]["peak_c"])
        tr = therm[hot].calibrate(1200.0).run(per_joint[hot])
        traces[g] = {"joint": hot, "t_min": [t / 60 for t, _ in tr[::10]], "temp_c": [T for _, T in tr[::10]]}
    return {"simulation_only": True, "estimates": "temperatures, battery and currents are model estimates",
            "walker": label, "loop": [list(x) for x in loop], "minutes": minutes, "seed0": seed,
            "segments": segs, "temperatures": temps, "battery": battery, "traces_20min": traces,
            "pack": PACK, "time_constants_s": list(TIME_CONSTANTS_S)}


def chart(res: dict, out: Path, label: str) -> str:
    import chart_style as cs

    cs.setup()
    plt = cs.plt
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.09, right=0.8, top=0.8, bottom=0.11)
    for i, (g, tr) in enumerate(res["traces_20min"].items()):
        ax.plot(tr["t_min"], tr["temp_c"], color=cs.SERIES[i], lw=cs.LW, label=f"{g} ({tr['joint']})")
        ax.annotate(g, (tr["t_min"][-1], tr["temp_c"][-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                    color=cs.INK2, va="center")
    ax.axhline(100, color=cs.INK2, lw=0.9, ls="--")
    ax.annotate("100 C: the maker's temperature-rise reference", (0, 100), xytext=(2, 4),
                textcoords="offset points", fontsize=8, color=cs.INK2)
    ax.set_xlabel("demo time, minutes")
    ax.set_ylabel("winding temperature, C (estimate)")
    ax.set_title("Hottest drive of each group, thermal time constant 20 min")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=8)
    b = res["battery"]
    loop = ", ".join(f"{n} {s:.0f} s" for n, s in res["loop"])
    worst = max(((jn, v[k]["peak_c"]) for jn, v in res["temperatures"].items() for k in v), key=lambda x: x[1])
    cs.frame(fig, f"Demo course: heat and battery, {label}",
             f"Loop: {loop}, for {res['minutes']:.0f} min. Battery: {b['runtime_minutes']:.0f} min per pack "
             f"(mean draw {b['mean_total_w']:.0f} W incl. {b['hotel_w']:.0f} W hotel); peak pack current "
             f"{b['current_peak_a']['low_v']:.1f} A at {res['pack']['v_low']} V.\nHottest winding, time constants "
             f"10-40 min: {worst[0]} {worst[1]:.0f} C. ESTIMATES: a first-order thermal model calibrated to the "
             "maker's temperature-rise figures.",
             f"SIMULATION. {res['walker']}. Source: rl/demo_course.py.")
    name = "demo_temperatures.png"
    cs.save(fig, out / name)
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--walk-s", type=float, default=90.0)
    ap.add_argument("--stand-s", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--label", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    make, walker = policy_factory(Path(a.ckpt_dir)) if a.ckpt_dir else trot_factory()
    res = run(make, walker, [("walk_0.25", a.walk_s), ("stand", a.stand_s)], a.minutes, a.seed)
    label = a.label or ("model-based trot" if not a.ckpt_dir else Path(a.ckpt_dir).name)
    res["chart"] = chart(res, out, label)
    (out / "demo_course.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    b = res["battery"]
    print(f"walker: {walker}")
    for n, s in res["segments"].items():
        print(f"segment {n}: {'fell ' + s['termination'] if s['fell'] else 'no fall'}; mean draw {s['mean_draw_w']:.1f} W, "
              f"peak {s['peak_draw_w']:.0f} W, braking surplus peak {s['peak_surplus_w']:.0f} W")
    hot = sorted(((v["20min"]["peak_c"], jn) for jn, v in res["temperatures"].items()), reverse=True)[:3]
    print("hottest windings (20 min time constant): " + ", ".join(f"{jn} {T:.0f} C" for T, jn in hot))
    print(f"battery: {b['runtime_minutes']:.0f} min at {b['mean_total_w']:.0f} W; peak current "
          f"{b['current_peak_a']['low_v']:.1f} A at low charge")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
