"""Heat, battery and loads over a museum demo, for the 60 cm build (decision D21). Simulation plus ESTIMATES.

1. **Segments, simulated.** One 20 s episode per activity (walking at 0.25 m/s, walking at 0.15 m/s, standing),
   nominal model, sensor noise on, servo delay 10 ms, flat floor, measured from 2 s. Each records, every 1 ms physics
   step: every leg drive's torque and speed, the electrical estimate (``analytics/gait_core.electrical_power``: copper
   loss 3 R (tau_drive / Kt)^2, the drives' net draw on a shared bus, the braking surplus), the joint bearing loads
   and the foot forces (``telemetry/loads.py``).
2. **The demo, modelled.** A loop of segments (default: 90 s walking at 0.25 m/s, 30 s standing) repeated for
   ``--minutes`` (default: one pack's runtime on that loop), from the room's temperature in open air (``--room-c``,
   default 25 C, the thermal calibration's room; decision D48 checks heat rules in degrees at 35 C):
   - winding temperatures per drive (``telemetry/thermal.py``), calibrated from the maker's thermal-equilibrium
     torque, for thermal time constants of 4, 8 and 15 min (fitted to the maker's overload figures; a range), with
     the middle of the settled temperature cycle when the loop runs without a break;
   - battery runtime and pack currents (``telemetry/power.py``), from the pack on record (146.5 Wh, 80 % usable,
     12S at 42.0-49.2 V in use; 44.4 V nominal, 42.0 V as low charge) and a 30 W hotel load (25-40 W as a range).
     The peak current is given for the loop and for every measured segment.
   The loads are per segment, the same rigid-model loads at any demo length; ``loads_headline`` picks the largest.
   A segment that falls makes the demo INVALID (said in the JSON, the printout and the chart).

What it leaves out: turns, starts and stops (the segments are straight, and the first 2 s are not measured), the
head and tail servos, and the electrical losses ``telemetry/power.py`` lists.

The walker is the frozen model-based trot (``walk/walk60.py``) on the capsule-foot model by default, or a checkpoint
(``--ckpt-dir``; safe loaders as ``steer_eval.py``). Outputs ``demo_course.json`` and ``demo_temperatures.png``
(SIMULATION-labelled).

Usage: python hardware/triceratops_build/rl/demo_course.py --out DIR [--ckpt-dir DIR] [--minutes 20]
       [--walk-s 90 --stand-s 30] [--seed 700000] [--room-c 35]
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
# 2 x 6S 3300 mAh: 146.5 Wh = 12 x 3.7 V x 3.3 Ah, so 44.4 V nominal; 42.0 V is the drives' bus minimum (drives.json)
PACK = {"pack_wh": 146.5, "usable_share": 0.80, "v_nominal": 44.4, "v_low": 42.0, "hotel_w": 30.0}
HOTEL_RANGE_W = (25.0, 40.0)  # analytics/gait_report.HOTEL_RANGE_W
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


def _events_j(net: np.ndarray, dt: float) -> float:
    """The largest energy (J) of one contiguous braking event (net draw below zero)."""
    neg = np.r_[False, net < 0, False]
    edges = np.flatnonzero(np.diff(neg.astype(int)))
    return max((float(-net[a:b].sum() * dt) for a, b in zip(edges[::2], edges[1::2])), default=0.0)


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
    fell, term, speeds = False, "", []
    try:
        while True:
            obs, _, terminated, truncated, info = env.step(act(obs))
            if info["time_s"] > WINDOW_S:
                speeds.append(info["forward_speed"])
            if terminated or truncated:
                fell, term = bool(terminated), info["termination"]
                break
    finally:
        env.substep_hook = None
    t = np.asarray(log["t"])
    w = t >= WINDOW_S
    base = {"command_mps": command, "fell": fell, "termination": term, "joints": names, "groups": groups,
            "seconds_measured": float(w.sum() * env.physics_dt)}
    if not w.any():  # fell before the window: nothing to measure
        return {**base, "measured": False}
    tau, om = np.asarray(log["tau"])[w], np.asarray(log["om"])[w]
    el = gc.electrical_power(tau, om, eta, kt, res)
    p = tau * om
    tau_drive = np.where(p >= 0, tau / eta, tau * eta)
    net = (el["drive_mech"] + el["copper"]).sum(1)
    bus = el["shared_bus"]
    v = float(np.mean(speeds)) if speeds else 0.0
    return {**base, "measured": True, "speed_mps": v,
            "tau_drive_sq_mean": (tau_drive ** 2).mean(0).tolist(),
            "tau_drive_rms": np.sqrt((tau_drive ** 2).mean(0)).tolist(), "copper_w": el["copper"].mean(0).tolist(),
            "mean_draw_w": float(bus.mean()), "mean_sq_draw_w2": float((bus ** 2).mean()),
            "peak_draw_w": float(bus.max()), "peak_surplus_w": float(max(0.0, -net.min())),
            "surplus_share": float((net < 0).mean()), "brake_event_j_max": _events_j(net, env.physics_dt),
            "drive_energy_per_m_j": float(bus.mean() / v) if v > 0.05 else None,
            "loads": rec.summary(env.nominal_weight)}


def build_sequence(loop, minutes: float) -> list[tuple[str, float]]:
    """The loop's segments repeated until ``minutes``; zero-length entries are dropped, negative ones refused."""
    if any(s < 0 for _, s in loop):
        raise ValueError("loop durations must not be negative")
    loop = [(n, float(s)) for n, s in loop if s >= 1e-3]  # under 1 ms: nothing to model
    if not loop or minutes <= 0:
        raise ValueError("the demo needs a loop of positive length and positive minutes")
    seq, total, end = [], 0.0, minutes * 60.0
    while end - total > 1e-9:
        for name, s in loop:  # every entry is at least 1 ms long, so each pass makes progress
            take = min(s, end - total)
            seq.append((name, take))
            total += take
            if end - total <= 1e-9:
                break
    return seq


def loads_headline(segments: dict) -> dict:
    """The largest loads over all measured segments: joint force (99th percentile and maximum, body weights), bending
    (N m) and foot normal force (body weights), each with its joint or foot and segment."""
    best = {}

    def take(key, value, who, seg, extra):
        if value is not None and (key not in best or value > best[key]["value"]):
            best[key] = {"value": value, "where": who, "segment": seg, **extra}

    for sn, s in segments.items():
        if not s.get("measured"):
            continue
        for jn, j in s["loads"]["joints"].items():
            if j["force_bw"]:
                take("joint_force_p99_bw", j["force_bw"]["p99"], jn, sn, {"max_bw": j["force_bw"]["max"]})
                take("joint_force_max_bw", j["force_bw"]["max"], jn, sn, {})
            if j["bending_nm"]:
                take("bending_p99_nm", j["bending_nm"]["p99"], jn, sn, {"max_nm": j["bending_nm"]["max"]})
                take("bending_max_nm", j["bending_nm"]["max"], jn, sn, {})
        for fn, f in s["loads"]["feet"].items():
            if f["normal_bw"]:
                take("foot_max_bw", f["normal_bw"]["max"], fn, sn, {"p99_bw": f["normal_bw"]["p99"]})
    return best


def run(make, label: str, loop, minutes: float | None, seed: int, segment_s: float = 20.0,
        time_constants_s=None, room_c: float | None = None) -> dict:
    """Measure the segments, then model the demo: the loop for ``minutes``, or (None) for one pack's runtime.
    ``room_c``: the room the windings start in and shed heat to (None: the calibration's 25 C; the drives are
    calibrated at that room whatever ``room_c`` is, ``telemetry/thermal.py``)."""
    import gait_report as gr
    from telemetry.power import demo_battery, pack_currents
    from telemetry.thermal import RATING_AMBIENT_C, TIME_CONSTANTS_S, DriveThermal, demo_temperatures, tc_key

    tcs = tuple(time_constants_s or TIME_CONSTANTS_S)
    build_sequence(loop, 1.0)  # refuse a bad loop before the simulations
    drives_json = json.loads((HERE.parent / "sim" / "data" / "drives.json").read_text())["leg_drives"]
    drives = gr.drive_table()
    segs = {name: measure_segment(make, SEGMENTS[name], seed + i, segment_s) for i, name in enumerate(SEGMENTS)}
    fallen = [n for n, s in segs.items() if s["fell"]]
    base = {"simulation_only": True, "estimates": "temperatures, battery and currents are model estimates",
            "walker": label, "loop": [list(x) for x in loop], "seed0": seed, "segments": segs,
            "valid": not fallen, "fallen_segments": fallen, "loads_headline": loads_headline(segs),
            "pack": PACK, "hotel_range_w": list(HOTEL_RANGE_W), "time_constants_s": list(tcs),
            "room_c": RATING_AMBIENT_C if room_c is None else float(room_c)}
    if any(not s["measured"] for s in segs.values()):
        return {**base, "minutes": minutes, "temperatures": None, "battery": None, "traces": None}
    bkeys = ("mean_draw_w", "mean_sq_draw_w2", "peak_draw_w", "peak_surplus_w", "surplus_share", "brake_event_j_max")
    battery_of = lambda seq: demo_battery([{"seconds": sec, **{k: segs[n][k] for k in bkeys}}  # noqa: E731
                                           for n, sec in seq], hotel_range_w=HOTEL_RANGE_W, **PACK)
    one_pack = minutes is None
    if one_pack:  # one pack's runtime on this loop
        minutes = battery_of(build_sequence(loop, sum(s for _, s in loop) / 60.0))["runtime_minutes"]
    seq = build_sequence(loop, minutes)
    joints, groups = segs["stand"]["joints"], segs["stand"]["groups"]
    therm, per_joint = {}, {}
    for j, (jn, g) in enumerate(zip(joints, groups)):
        dd = drives_json[drives[g]["drive"]]
        therm[jn] = DriveThermal(jn, drives[g]["kt_nm_per_a_rms"], drives[g]["phase_resistance_ohm"],
                                 float(dd["thermal_eq_nm"]))
        per_joint[jn] = [(sec, segs[name]["tau_drive_sq_mean"][j]) for name, sec in seq]
    temps = demo_temperatures(therm, per_joint, tcs, room_c=room_c)
    battery = battery_of(seq)
    battery["current_peak_all_segments_a"] = pack_currents(
        max(s["peak_draw_w"] for s in segs.values()) + PACK["hotel_w"], v_nominal=PACK["v_nominal"],
        v_low=PACK["v_low"])
    battery["peak_segment"] = max(segs, key=lambda n: segs[n]["peak_draw_w"])
    # temperature traces of the hottest joint per drive group, at the middle time constant, for the chart
    shown = sorted(tcs)[len(tcs) // 2]
    traces = {"time_constant_s": shown, "groups": {}}
    for g in dict.fromkeys(groups):
        jg = [jn for jn, gg in zip(joints, groups) if gg == g]
        hot = max(jg, key=lambda jn: temps[jn][tc_key(shown)]["peak_c"])
        tr = therm[hot].calibrate(shown, room_c=room_c).run(per_joint[hot])
        step = max(1, len(tr) // 600)
        traces["groups"][g] = {"joint": hot, "t_min": [t / 60 for t, _ in tr[::step]],
                               "temp_c": [T for _, T in tr[::step]],
                               "continuous_c": temps[hot][tc_key(shown)]["continuous_c"]}
    return {**base, "minutes": minutes, "minutes_is_one_pack": one_pack, "temperatures": temps, "battery": battery,
            "traces": traces}


def _settled(c) -> str:
    return "no steady state (the winding heats without limit)" if c is None else \
        f"the settled cycle centres on {c:.1f} C"


def hottest(res: dict) -> dict:
    """The hottest winding at each time constant: its peak over the demo, and the highest middle of the settled
    cycle without a break (None if any drive runs away)."""
    from telemetry.thermal import tc_key

    out = {}
    for tc in res["time_constants_s"]:
        k = tc_key(tc)
        jn = max(res["temperatures"], key=lambda j: res["temperatures"][j][k]["peak_c"])
        cont = [v[k]["continuous_c"] for v in res["temperatures"].values()]
        out[k] = {"joint": jn, "peak_c": res["temperatures"][jn][k]["peak_c"],
                  "continuous_c": None if any(c is None for c in cont) else max(cont)}
    return out


def _bottom_clear_of_legend(fig, ax, leg, traces, margin_c: float = 3.0) -> float:
    """The y-axis bottom (a multiple of 5 C, never above the current one) that keeps every trace, and so its end
    label, at least ``margin_c`` above a lower-right legend over the minutes the legend spans. The legend's size on
    the page does not change with the y-limits, so its top stays at the same fraction of the axis height."""
    fig.canvas.draw()  # places the legend
    box = leg.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())  # axes fraction
    lo, hi = ax.get_ylim()
    t0 = ax.transData.inverted().transform(ax.transAxes.transform((box.x0, 0.0)))[0]
    low = min(T for tr in traces for t, T in zip(tr["t_min"], tr["temp_c"]) if t >= t0)
    need = (low - margin_c - box.y1 * hi) / (1.0 - box.y1)  # the bottom b at which b + box.y1 (hi - b) = low - margin
    return min(lo, 5.0 * math.floor(need / 5.0))


def chart(res: dict, out: Path, label: str) -> str:
    import chart_style as cs

    cs.setup()
    plt = cs.plt
    from telemetry.thermal import RATING_AMBIENT_C, tc_key

    room_c = res.get("room_c", RATING_AMBIENT_C)  # a result from before 27 Nov 2026 has no room: it was 25 C
    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.subplots_adjust(left=0.08, right=0.82, top=0.72, bottom=0.11)
    tr_all = res["traces"]
    shown = tr_all["time_constant_s"]
    for i, (g, tr) in enumerate(tr_all["groups"].items()):
        ax.plot(tr["t_min"], tr["temp_c"], color=cs.SERIES[i], lw=cs.LW, label=f"{g} ({tr['joint']})")
        ax.annotate(g, (tr["t_min"][-1], tr["temp_c"][-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                    color=cs.INK2, va="center")
    ax.axhline(100, color=cs.INK2, lw=0.9, ls="--")
    ax.annotate("100 C: the maker's temperature-rise reference", (0, 100), xytext=(2, 4),
                textcoords="offset points", fontsize=8, color=cs.INK2)
    ylim, legend = (20.0, 110.0), dict(loc="upper left", bbox_to_anchor=(0.0, 0.86))  # the 25 C chart, as before
    if room_c != RATING_AMBIENT_C:  # another room: every trace and its end label in frame, the legend lower right
        hi = max(max(tr["temp_c"]) for tr in tr_all["groups"].values())  # with the axis bottom taken clear of it
        ylim = (min(20.0, 5.0 * math.floor((room_c - 5.0) / 5.0)), max(110.0, 5.0 * math.ceil((hi + 5.0) / 5.0)))
        legend = dict(loc="lower right")
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"demo time, minutes (from {room_c:g} C, open air)")
    ax.set_ylabel("winding temperature, C (estimate)")
    ax.set_title(f"Hottest drive of each group, thermal time constant {shown / 60:g} min")
    leg = ax.legend(fontsize=8, **legend)
    if room_c != RATING_AMBIENT_C:
        ax.set_ylim(_bottom_clear_of_legend(fig, ax, leg, tr_all["groups"].values()), ylim[1])
    b = res["battery"]
    loop = ", ".join(f"{n} {s:.0f} s" for n, s in res["loop"])
    hot = hottest(res)
    k = tc_key(shown)
    rng = sorted(hot[tc_key(tc)]["peak_c"] for tc in res["time_constants_s"])
    lo_tc, hi_tc = min(res["time_constants_s"]) / 60, max(res["time_constants_s"]) / 60
    invalid = "" if res["valid"] else f"INVALID: segment(s) fell ({', '.join(res['fallen_segments'])}). "
    import textwrap

    one_pack = " (one pack)" if res["minutes_is_one_pack"] else ""
    room = "" if room_c == RATING_AMBIENT_C else f" in a {room_c:g} C room"
    sub = (f"{invalid}Loop: {loop}, for {res['minutes']:.0f} min{one_pack}. Battery: {b['runtime_minutes']:.0f} "
           f"min per pack at {b['mean_total_w']:.0f} W mean, incl. {b['hotel_w']:.0f} W hotel; peak pack current "
           f"{b['current_peak_a']['low_v']:.1f} A on the loop, {b['current_peak_all_segments_a']['low_v']:.1f} A in "
           f"any segment, at {res['pack']['v_low']} V.\nHottest winding{room}: {hot[k]['joint']} "
           f"{hot[k]['peak_c']:.1f} C "
           f"(time constant {shown / 60:g} min; {rng[0]:.1f}-{rng[-1]:.1f} C over {lo_tc:g}-{hi_tc:g} min); "
           f"without a break, {_settled(hot[k]['continuous_c'])}.\nESTIMATES: a first-order "
           "thermal model calibrated to the maker's thermal-equilibrium torque; leg drives only.")
    sub = "\n".join(textwrap.fill(par, 150) for par in sub.split("\n"))
    cs.frame(fig, f"Demo course: heat and battery, {label}", sub,
             f"SIMULATION. {res['walker']}. Source: rl/demo_course.py.")
    name = "demo_temperatures.png"
    cs.save(fig, out / name)
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--minutes", type=float, default=None, help="demo length (default: one pack's runtime)")
    ap.add_argument("--walk-s", type=float, default=90.0)
    ap.add_argument("--stand-s", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--label", default=None)
    ap.add_argument("--room-c", type=float, default=None,
                    help="room temperature, C (default: 25, the thermal calibration's room; D48: rules at 35)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    make, walker = policy_factory(Path(a.ckpt_dir)) if a.ckpt_dir else trot_factory()
    res = run(make, walker, [("walk_0.25", a.walk_s), ("stand", a.stand_s)], a.minutes, a.seed, room_c=a.room_c)
    label = a.label or ("model-based trot" if not a.ckpt_dir else Path(a.ckpt_dir).name)
    print(f"walker: {walker}")
    for n, s in res["segments"].items():
        if not s["measured"]:
            print(f"segment {n}: fell {s['termination']} before the measuring window")
            continue
        print(f"segment {n}: {'FELL ' + s['termination'] if s['fell'] else 'no fall'}; {s['speed_mps']:.3f} m/s; "
              f"mean draw {s['mean_draw_w']:.1f} W, peak {s['peak_draw_w']:.0f} W; braking surplus peak "
              f"{s['peak_surplus_w']:.0f} W, largest event {s['brake_event_j_max']:.2f} J")
    if res["battery"] is not None:
        res["chart"] = chart(res, out, label)
    (out / "demo_course.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    if not res["valid"]:
        print(f"INVALID demo: segment(s) fell: {', '.join(res['fallen_segments'])}")
    if res["battery"] is None:
        print("no demo modelled: a segment fell before the measuring window")
        print("wrote", out)
        return 1
    from telemetry.thermal import RATING_AMBIENT_C, tc_key

    b = res["battery"]
    hh = hottest(res)
    room = "" if res["room_c"] == RATING_AMBIENT_C else f" in a {res['room_c']:g} C room"
    for tc in res["time_constants_s"]:
        k = tc_key(tc)
        hot = sorted(((v[k]["peak_c"], jn) for jn, v in res["temperatures"].items()), reverse=True)[:3]
        print(f"hottest windings over {res['minutes']:.0f} min{room} (time constant {tc / 60:g} min): "
              + ", ".join(f"{jn} {T:.1f} C" for T, jn in hot) + f"; without a break, {_settled(hh[k]['continuous_c'])}")
    rt = ", ".join(f"{h} W hotel {m:.0f} min" for h, m in b["runtime_minutes_by_hotel_w"].items())
    print(f"battery: {b['runtime_minutes']:.0f} min at {b['mean_total_w']:.0f} W ({rt}); pack current at low "
          f"charge: mean {b['current_mean_a']['low_v']:.1f} A, RMS {b['current_rms_a']['low_v']:.1f} A, peak "
          f"{b['current_peak_a']['low_v']:.1f} A on the loop, {b['current_peak_all_segments_a']['low_v']:.1f} A in "
          f"any segment ({b['peak_segment']})")
    print(f"braking: drives' surplus peak {b['braking_peak_surplus_w']:.0f} W, beyond the hotel load "
          f"{b['braking_peak_beyond_hotel_w']:.0f} W; largest event {b['braking_event_j_max']:.2f} J")
    h = res["loads_headline"]
    print("loads (largest over the segments): "
          f"joint force p99 {h['joint_force_p99_bw']['value']:.2f} BW ({h['joint_force_p99_bw']['where']}, "
          f"{h['joint_force_p99_bw']['segment']}); joint force max {h['joint_force_max_bw']['value']:.2f} BW "
          f"({h['joint_force_max_bw']['where']}, {h['joint_force_max_bw']['segment']}); bending p99 "
          f"{h['bending_p99_nm']['value']:.1f} N m ({h['bending_p99_nm']['where']}, {h['bending_p99_nm']['segment']}); "
          f"bending max {h['bending_max_nm']['value']:.1f} N m ({h['bending_max_nm']['where']}, "
          f"{h['bending_max_nm']['segment']}); foot max {h['foot_max_bw']['value']:.2f} BW "
          f"({h['foot_max_bw']['where']}, {h['foot_max_bw']['segment']})")
    print("wrote", out)
    return 0 if res["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
