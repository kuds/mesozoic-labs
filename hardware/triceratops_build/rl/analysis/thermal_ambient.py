"""Winding temperatures over the stored D22 demo courses in a warmer room (decision D48, 27 Nov 2026: heat rules in
degrees assume a 35 C room). DESK ESTIMATES on SIMULATED torques: nothing here is measured.

Inputs, as stored by ``rl/demo_course.py`` on 26 Nov 2026:
- ``results/d22/demo/<run>/demo_course.json`` (commit 0133faa6): the default loop, 90 s walking at 0.25 m/s and 30 s
  standing, repeated for one pack, eight runs (128 and 160 mm, seeds 0-3);
- ``results/d22/demo_walkonly/<run>/demo_course.json`` (commit ad62d9a9): walking with no breaks for 96 min, four
  runs.
Each stores every leg drive's mean squared drive torque per simulated segment, the loop and the demo length. The
temperatures are recomputed from those, the way ``demo_course.run`` does it, for rooms of 25, 30, 35 and 40 C and the
three thermal time constants (4, 8, 15 min), two ways:

- **A, the consistent method:** each drive's thermal resistance and heat capacity (R_th, C) fitted at the rating
  test's 25 C room, then run in the warmer room (``telemetry/thermal.py``, ``room_c``). R_th and C describe the drive
  and how its surface sheds heat to the air; to first order a warmer room does not change them.
- **B, the optimistic bound:** refitted as if the maker's test (thermal-equilibrium torque held for 90 min, motor
  reaching 100 C) had started in the warmer room (``ambient_c`` set to the room). The same load must then reach the
  same 100 C from a warmer start, which needs a lower R_th: a drive that sheds heat better because the room is
  hotter (0.867 of A's R_th at 35 C, 65 K of rise against 75). B is the maker's rating taken to hold in any room,
  which is not physical: a lower bound on the temperature, not an estimate. For a drive at or above its rating B
  can even fall below the 25 C figure as the room warms, which no physical model does (a warmer room cannot cool a
  winding); so the optimistic figure is the larger of B and the stored 25 C figure (``optimistic_peak_c``).

**Check:** the 25 C column must reproduce every stored temperature figure exactly (peak, end, settled cycle, time to
100 C, cool-down, R_th and C, for every drive and time constant); the script stops if one differs.

Also, desk only, from the same model: the steady winding temperature under a constant load of 0.85 and 1.0 of the
continuous rating (the ratio rules' figures), the loads that settle at 90 and 100 C, and how much the 35 C peaks move
if the calibration's room is 26.5 C (the maker's X4-36-L table starts at about 26-27 C) instead of 25 C. And where
each D22 policy's worst drive would settle, walked without a break, at two loads that differ: its judged H_all
(``results/d22/judge_d22.json``: the gait report's randomised target case, the worst drive's RMS over its rating,
averaged over 5 episodes) and the demo course's own nominal walking segment (``walk_0.25`` in the stored segments,
18 s measured). The demo's no-break figures are the second.

The thermal fits depend only on a drive's figures, not on which joint it sits at, so each is computed once and
reused (``_cached_fits``); the exact 25 C reproduction checks this.

Output: ``results/heat35/thermal_ambient_2026-11-27.json`` and ``.md`` (small) and a printout.

Usage: python hardware/triceratops_build/rl/analysis/thermal_ambient.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL = HERE.parent
BUILD = RL.parent
for p in (RL, RL / "analytics"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from telemetry.thermal import (  # noqa: E402
    RATING_AMBIENT_C,
    TIME_CONSTANTS_S,
    DriveThermal,
    ThermalNode,
    demo_temperatures,
    tc_key,
)

D22 = HERE / "results" / "d22"
SETS = {"demo": "default loop (90 s walking at 0.25 m/s, 30 s standing; one pack)",
        "demo_walkonly": "walking with no breaks (96 min)"}
ROOMS_C = (25.0, 30.0, 35.0, 40.0)
METHODS = ("A", "B")
BENCH_RULE_C = 90.0  # bench/BENCH_PLAN_FORELEG.md:61, B3: winding at most 90 C at 35 C (written for another case)
LIMIT_C = 100.0  # the maker's temperature-rise reference (thermal_eq_nm: 90 min to 100 C, open air)
B3_LOOP_S = 30 * 60.0  # B3's loop is 30 minutes (bench/BENCH_PLAN_FORELEG.md:61)
CAL_ROOM_SENS_C = 26.5  # the maker's X4-36-L table starts at about 26-27 C (actuators.json, vendor_rated_definition)
SHOWN_TC = 480.0
TEMP_KEYS = ("time_constant_s", "r_th_k_per_w", "c_j_per_k", "peak_c", "end_c", "minutes_to_limit",
             "cooldown_minutes", "continuous_c", "steady_at_thermal_eq_c")


@contextlib.contextmanager
def _cached_fits():
    """Fit each drive's R_th and C once per (drive figures, time constant): the joint's name does not enter a fit."""
    orig = DriveThermal.calibrate
    fits: dict = {}

    def calibrate(self, time_constant_s, room_c=None):
        key = (replace(self, name=""), float(time_constant_s))
        if key not in fits:
            fits[key] = orig(self, time_constant_s)
        n = fits[key]
        return ThermalNode(self, n.r_th_k_per_w, n.c_j_per_k, n.step_s, room_c=room_c)

    DriveThermal.calibrate = calibrate
    try:
        yield
    finally:
        DriveThermal.calibrate = orig


def thermal_inputs(stored: dict):
    """The drives and per-drive loss histories of a stored demo, as ``demo_course.run`` builds them."""
    import demo_course
    import gait_report as gr

    drives_json = json.loads((BUILD / "sim" / "data" / "drives.json").read_text())["leg_drives"]
    drives = gr.drive_table()
    segs = stored["segments"]
    seq = demo_course.build_sequence([tuple(x) for x in stored["loop"]], stored["minutes"])
    joints, groups = segs["stand"]["joints"], segs["stand"]["groups"]
    therm, per_joint = {}, {}
    for j, (jn, g) in enumerate(zip(joints, groups)):
        dd = drives_json[drives[g]["drive"]]
        therm[jn] = DriveThermal(jn, drives[g]["kt_nm_per_a_rms"], drives[g]["phase_resistance_ohm"],
                                 float(dd["thermal_eq_nm"]))
        per_joint[jn] = [(sec, segs[name]["tau_drive_sq_mean"][j]) for name, sec in seq]
    return therm, per_joint, dict(zip(joints, groups))


def temperatures(therm, per_joint, room_c: float, method: str, tcs=TIME_CONSTANTS_S, cal_room_c=None):
    """``demo_temperatures`` in a room, by method A (fit at the rating room, room raised) or B (refit at the room).
    ``cal_room_c`` replaces the rating room of the fit (A only; a sensitivity)."""
    if method == "A":
        dr = therm if cal_room_c is None else {k: replace(d, ambient_c=cal_room_c) for k, d in therm.items()}
        return demo_temperatures(dr, per_joint, tcs, room_c=room_c)
    if method == "B":
        return demo_temperatures({k: replace(d, ambient_c=room_c) for k, d in therm.items()}, per_joint, tcs)
    raise ValueError(method)


def _node(d: DriveThermal, tc: float, room_c: float, method: str) -> ThermalNode:
    return d.calibrate(tc, room_c=room_c) if method == "A" else replace(d, ambient_c=room_c).calibrate(tc)


def hottest_summary(temps, therm, per_joint, room_c, method, tcs=TIME_CONSTANTS_S) -> dict:
    """Per time constant: the hottest winding's joint, peak, its peak in the first 30 min (B3's loop length), time
    to 90 and 100 C and minutes at or over 90 C; the highest middle of the settled cycle; how many drives peak over
    90 and over 100 C."""
    out = {}
    for tc in tcs:
        k = tc_key(tc)
        jn = max(temps, key=lambda j: temps[j][k]["peak_c"])
        tr = _node(therm[jn], tc, room_c, method).run(per_joint[jn])
        t90 = next((t for t, T in tr if T >= BENCH_RULE_C), None)
        over90 = sum(t1 - t0 for (t0, _), (t1, T1) in zip(tr, tr[1:]) if T1 >= BENCH_RULE_C)
        cont = [v[k]["continuous_c"] for v in temps.values()]
        out[k] = {"joint": jn, "peak_c": temps[jn][k]["peak_c"],
                  "peak_c_first_30_min": max(T for t, T in tr if t <= B3_LOOP_S + 1e-9),
                  "minutes_to_90_c": None if t90 is None else t90 / 60.0,
                  "minutes_to_100_c": temps[jn][k]["minutes_to_limit"],
                  "minutes_at_or_over_90_c": over90 / 60.0,
                  "settled_cycle_max_c": None if any(c is None for c in cont) else max(cont),
                  "drives_over_90_c": sum(v[k]["peak_c"] > BENCH_RULE_C for v in temps.values()),
                  "drives_over_100_c": sum(v[k]["peak_c"] > LIMIT_C for v in temps.values())}
    return out


def check_reproduces(stored_temps: dict, temps: dict) -> list[str]:
    """Every stored figure against the recomputed one, exactly; returns the differences (none expected)."""
    bad = []
    for jn, per in stored_temps.items():
        for k, v in per.items():
            for f in TEMP_KEYS:
                if temps[jn][k][f] != v[f]:
                    bad.append(f"{jn} {k} {f}: stored {v[f]!r}, recomputed {temps[jn][k][f]!r}")
    return bad


def steady_load_table(tcs=TIME_CONSTANTS_S, rooms=ROOMS_C) -> dict:
    """Desk: per drive, time constant and room (method A), the steady winding temperature at 0.85 and 1.0 of the
    continuous rating, the load fractions that settle at 90 and 100 C, and method B's R_th over method A's (the
    better cooling B assumes in that room). Closed form of the model's steady state:
    T - room = K x^2 (1 + alpha (room - T_ref)) / (1 - K x^2 alpha), K = 3 R_0 eq^2 / Kt^2 R_th."""
    import gait_report as gr

    drives_json = json.loads((BUILD / "sim" / "data" / "drives.json").read_text())["leg_drives"]
    seen, out = set(), {}
    for g, row in gr.drive_table().items():
        if row["drive"] in seen:
            continue
        seen.add(row["drive"])
        d = DriveThermal(row["drive"], row["kt_nm_per_a_rms"], row["phase_resistance_ohm"],
                         float(drives_json[row["drive"]]["thermal_eq_nm"]))
        per = {}
        for tc in tcs:
            fit = d.calibrate(tc)
            big_k = 3.0 * d.r_phase_ohm * d.thermal_eq_nm ** 2 / d.kt_nm_per_a ** 2 * fit.r_th_k_per_w
            for room in rooms:
                node = replace(fit, room_c=room)
                b = 1.0 + d.alpha * (room - d.r_ref_c)

                def load_for(temp_c):
                    rise = temp_c - room
                    return math.sqrt(rise / (big_k * (b + d.alpha * rise))) if rise > 0 else 0.0

                per[f"{tc_key(tc)}_room_{room:g}c"] = {
                    "time_constant_s": tc, "room_c": room,
                    "steady_c_at_0.85": node.steady_state_c((0.85 * d.thermal_eq_nm) ** 2),
                    "steady_c_at_1.00": node.steady_state_c(d.thermal_eq_nm ** 2),
                    "load_settling_at_90_c": load_for(90.0), "load_settling_at_100_c": load_for(100.0),
                    "r_th_b_over_a": replace(d, ambient_c=room).calibrate(tc).r_th_k_per_w / fit.r_th_k_per_w}
        out[row["drive"]] = {"kt_nm_per_a_rms": d.kt_nm_per_a, "phase_resistance_ohm": d.r_phase_ohm,
                             "thermal_eq_nm": d.thermal_eq_nm, "cases": per}
    return out


def optimistic_peaks(b_hottest: dict, stored_temps: dict) -> dict:
    """Per time constant: method B's hottest peak, the stored 25 C hottest peak, whether B is below it, and the
    optimistic figure, the larger of the two (a warmer room cannot cool a winding; B can say it does for a drive at or
    above its rating, since B takes the rating to hold in any room)."""
    out = {}
    for k, h in b_hottest.items():
        s25 = max(v[k]["peak_c"] for v in stored_temps.values())
        out[k] = {"b_peak_c": h["peak_c"], "stored_25c_peak_c": s25, "b_below_stored_25c": h["peak_c"] < s25,
                  "optimistic_peak_c": max(h["peak_c"], s25)}
    return out


def judged_load_table(tcs=TIME_CONSTANTS_S, rooms=ROOMS_C) -> dict:
    """Desk: the steady winding temperature (method A) at each D22 policy's judged worst-drive load H_all (the worst
    joint's RMS torque over its continuous rating, F2's figure: the gait report's randomised target case, the mean of
    each drive's ratio over 5 episodes; not the demo course's load, ``demo_walk_load_table``). The model's steady
    state in units of the rating is the same for every drive it is fitted to (each is fitted to the same 90 min /
    100 C point), so the X4-36-L stands for all; the X6-60 agrees to 0.1 C (``steady_load_table``)."""
    import gait_report as gr

    j = json.loads((D22 / "judge_d22.json").read_text())["runs"]
    drives_json = json.loads((BUILD / "sim" / "data" / "drives.json").read_text())["leg_drives"]
    row = gr.drive_table()["pitch"]
    d = DriveThermal(row["drive"], row["kt_nm_per_a_rms"], row["phase_resistance_ohm"],
                     float(drives_json[row["drive"]]["thermal_eq_nm"]))
    out = {"source": str((D22 / "judge_d22.json").relative_to(BUILD)), "drive_used": row["drive"], "runs": {}}
    for run, v in j.items():
        h = float(v["H_all"])
        out["runs"][run] = {"H_all": h, "steady_c": {
            f"{room:g}": {tc_key(tc): d.calibrate(tc, room_c=room).steady_state_c((h * d.thermal_eq_nm) ** 2)
                          for tc in tcs} for room in rooms}}
    return out


def demo_walk_load_table(tcs=TIME_CONSTANTS_S, rooms=ROOMS_C, set_name="demo") -> dict:
    """Desk: per stored demo course, the nominal walking segment's (``walk_0.25``) worst drive, its RMS torque over its
    continuous rating, and where it would settle walked without a break (method A): the demo's own no-break load
    (the walk-only demo courses' settled figures where one was run)."""
    out = {"source": f"rl/analysis/results/d22/{set_name}/<run>/demo_course.json, segments.walk_0.25", "runs": {}}
    for f in sorted((D22 / set_name).glob("*/demo_course.json")):
        stored = json.loads(f.read_text())
        therm, _, groups = thermal_inputs(stored)
        seg = stored["segments"]["walk_0.25"]
        loads = {jn: math.sqrt(q) / therm[jn].thermal_eq_nm for jn, q in zip(seg["joints"], seg["tau_drive_sq_mean"])}
        jn = max(loads, key=loads.get)
        d = therm[jn]
        out["runs"][f.parent.name] = {
            "joint": jn, "group": groups[jn], "seconds_measured": seg["seconds_measured"], "load": loads[jn],
            "steady_c": {f"{room:g}": {tc_key(tc): d.calibrate(tc, room_c=room).steady_state_c(
                (loads[jn] * d.thermal_eq_nm) ** 2) for tc in tcs} for room in rooms}}
    return out


def analyse(sets=tuple(SETS), rooms=ROOMS_C, tcs=TIME_CONSTANTS_S) -> dict:
    res = {"desk_estimates_on_simulated_torques": True, "open_air_no_shell": True,
           "date": "2026-11-27", "decision": "D48", "rating_room_c": RATING_AMBIENT_C, "rooms_c": list(rooms),
           "time_constants_s": list(tcs), "bench_rule_c": BENCH_RULE_C, "limit_c": LIMIT_C,
           "methods": {"A": "fit at the rating room (25 C), room raised: the consistent method",
                       "B": "refit as if the maker's test had started in the room: an optimistic bound"},
           "sets": {}}
    with _cached_fits():
        for s in sets:
            res["sets"][s] = {"description": SETS[s], "runs": {}}
            for f in sorted((D22 / s).glob("*/demo_course.json")):
                stored = json.loads(f.read_text())
                therm, per_joint, groups = thermal_inputs(stored)
                run = {"input": str(f.relative_to(BUILD)), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                       "valid": stored["valid"], "minutes": stored["minutes"],
                       "loop": stored["loop"], "walker": stored["walker"], "rooms": {}}
                for room in rooms:
                    run["rooms"][f"{room:g}"] = r = {}
                    for m in METHODS:
                        t = temperatures(therm, per_joint, room, m, tcs)
                        if room == RATING_AMBIENT_C:
                            bad = check_reproduces(stored["temperatures"], t)
                            if bad:
                                raise SystemExit(f"{f}: the 25 C rerun ({m}) differs from the stored result:\n"
                                                 + "\n".join(bad[:20]))
                        r[m] = {"hottest": hottest_summary(t, therm, per_joint, room, m, tcs),
                                "peaks_c_tau_480s": {jn: v[tc_key(SHOWN_TC)]["peak_c"] for jn, v in t.items()}
                                if SHOWN_TC in tcs else None}
                        if m == "B":
                            r[m]["optimistic"] = optimistic_peaks(r[m]["hottest"], stored["temperatures"])
                if 35.0 in rooms:
                    t = temperatures(therm, per_joint, 35.0, "A", tcs, cal_room_c=CAL_ROOM_SENS_C)
                    run["sensitivity_cal_room_26.5c_room_35c_A"] = {
                        tc_key(tc): max(v[tc_key(tc)]["peak_c"] for v in t.values()) for tc in tcs}
                run["reproduces_stored_25c"] = True
                run["groups"] = groups
                res["sets"][s]["runs"][f.parent.name] = run
        res["steady_loads"] = steady_load_table(tcs, rooms)
        res["judged_loads"] = judged_load_table(tcs, rooms)
        res["demo_walk_loads"] = demo_walk_load_table(tcs, rooms)
    return res


def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def markdown(res: dict) -> str:
    k8 = tc_key(SHOWN_TC)
    tcs = res["time_constants_s"]
    lines = ["# Winding temperatures in a warmer room (D48): tables", "",
             "27 Nov 2026. Generated by `rl/analysis/thermal_ambient.py` from the stored demo courses. DESK ESTIMATES "
             "(first-order thermal model, open air, no shell) on SIMULATED torques; nothing is measured. Method A: "
             "fitted at the 25 C rating room, room raised (consistent). Method B: refitted at the room (optimistic "
             "bound: the maker's rating taken to hold in any room, not physical). The 25 C rows reproduce the stored "
             "results exactly (checked for every drive and time constant). Written up in "
             "`rl/analysis/REPORT_HEAT_35C.md`.", ""]
    for s, block in res["sets"].items():
        runs = block["runs"]
        names = list(runs)
        lines += [f"## {s}: {block['description']}", "",
                  f"Hottest winding peak, C, at the {SHOWN_TC / 60:g} min time constant (in brackets: the "
                  f"{min(tcs) / 60:g}-{max(tcs) / 60:g} min range). Bench B3's figure: {BENCH_RULE_C:g} C at 35 C "
                  "(written for a measured winding on the 75 cm T20 foreleg, 30 min loop, shell on). A B cell marked * "
                  "is below the stored 25 C figure: B takes the maker's rating to hold in any room, so a drive at or "
                  "above its rating cools as the room warms, which no physical model does; there the optimistic "
                  "figure is the stored 25 C one.", "",
                  "| Room | Method | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for room in res["rooms_c"]:
            for m in METHODS:
                if room == RATING_AMBIENT_C and m == "B":
                    continue
                cells = []
                for n in names:
                    rm = runs[n]["rooms"][f"{room:g}"][m]
                    h = rm["hottest"]
                    pk = [h[tc_key(tc)]["peak_c"] for tc in tcs]
                    star = " *" if m == "B" and rm["optimistic"][k8]["b_below_stored_25c"] else ""
                    cells.append(f"{h[k8]['peak_c']:.1f} ({min(pk):.1f}-{max(pk):.1f}){star}")
                tag = "stored" if room == RATING_AMBIENT_C else m
                lines.append(f"| {room:g} C | {tag} | " + " | ".join(cells) + " |")
        lines += ["", f"Hottest joint at the {SHOWN_TC / 60:g} min time constant (the joints over every room and "
                  "both methods):", ""]
        cells = []
        for n in names:
            js = {runs[n]["rooms"][f"{room:g}"][m]["hottest"][k8]["joint"] for room in res["rooms_c"] for m in METHODS}
            cells.append(f"{n}: {', '.join(sorted(js))}")
        lines += ["- " + "; ".join(cells), "",
                  f"Method A, {SHOWN_TC / 60:g} min, the hottest winding: its peak in the first 30 min (B3's loop "
                  "length), C; minutes to 90 C / to 100 C; minutes at or over 90 C; drives over 90 C; the settled "
                  "cycle without a break (highest drive), C:", "",
                  "| Room | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for room in res["rooms_c"]:
            cells = []
            for n in names:
                h = runs[n]["rooms"][f"{room:g}"]["A"]["hottest"][k8]
                cells.append(f"{h['peak_c_first_30_min']:.1f}; {_f(h['minutes_to_90_c'], 0)} / "
                             f"{_f(h['minutes_to_100_c'], 0)}; "
                             f"{h['minutes_at_or_over_90_c']:.0f} min; {h['drives_over_90_c']}; "
                             f"{_f(h['settled_cycle_max_c'])}")
            lines.append(f"| {room:g} C | " + " | ".join(cells) + " |")
        if all("sensitivity_cal_room_26.5c_room_35c_A" in r for r in runs.values()):
            lines += ["", f"Sensitivity, method A at 35 C, fitted at a {CAL_ROOM_SENS_C:g} C rating room instead of "
                      f"25 C ({SHOWN_TC / 60:g} min): " + "; ".join(
                          f"{n} {runs[n]['sensitivity_cal_room_26.5c_room_35c_A'][k8]:.1f} C "
                          f"(25 C fit {runs[n]['rooms']['35']['A']['hottest'][k8]['peak_c']:.1f})" for n in names)]
        lines.append("")
    lines += ["## Steady loads (desk, method A)", "",
              "Steady winding temperature under a constant load, C, and the load (fraction of the continuous rating, "
              "the maker's thermal-equilibrium torque) that settles at 90 and at 100 C.", "",
              "| Drive | Time constant | Room | at 0.85 | at 1.00 | load to 90 C | load to 100 C |",
              "|---|---|---|---|---|---|---|"]
    rows: dict = {}
    for dn, d in res["steady_loads"].items():
        for key, c in d["cases"].items():
            cells = (f"{c['time_constant_s'] / 60:g} min | {c['room_c']:g} C | {_f(c['steady_c_at_0.85'])} | "
                     f"{_f(c['steady_c_at_1.00'])} | {c['load_settling_at_90_c']:.3f} | "
                     f"{c['load_settling_at_100_c']:.3f}")
            rows.setdefault(key, {}).setdefault(cells, []).append(dn)
    for key, by in rows.items():  # drives whose printed figures agree share a row
        for cells, dns in by.items():
            lines.append(f"| {', '.join(dns)} | {cells} |")
    jl = res["judged_loads"]
    names = list(jl["runs"])
    lines += ["", f"The gait report's randomised target case (H_all, F2's judged figure, `{jl['source']}`: the worst "
              "drive's RMS torque over its rating, averaged over 5 episodes on randomised models and floors). Where "
              "that drive would settle if walked without a break at that load, C, at the "
              f"{SHOWN_TC / 60:g} min time constant (in brackets: {min(tcs) / 60:g}-{max(tcs) / 60:g} min). Not the "
              "demo course's load: see the next table.", "",
              "| Room | " + " | ".join(f"{n} (H {jl['runs'][n]['H_all']:.3f})" for n in names) + " |",
              "|---|" + "---|" * len(names)]
    for room in res["rooms_c"]:
        cells = []
        for n in names:
            v = jl["runs"][n]["steady_c"][f"{room:g}"]
            vals = [v[tc_key(tc)] for tc in tcs]
            cells.append("runs away" if any(x is None for x in vals) else
                         f"{v[k8]:.1f} ({min(vals):.1f}-{max(vals):.1f})")
        lines.append(f"| {room:g} C | " + " | ".join(cells) + " |")
    wl = res.get("demo_walk_loads")
    if wl:
        names = list(wl["runs"])
        hj = {n: jl["runs"].get(n.replace("fore", "").replace("-", "_"), {}).get("H_all") for n in names}
        heavier = [n for n in names if hj[n] is not None and wl["runs"][n]["load"] > hj[n]]
        lighter = [n for n in names if hj[n] is not None and wl["runs"][n]["load"] < hj[n]]
        wo = res["sets"].get("demo_walkonly", {}).get("runs", {})
        gap = {tc: max((wl["runs"][n]["steady_c"][f"{room:g}"][tc_key(tc)]
                        - wo[n]["rooms"][f"{room:g}"]["A"]["hottest"][tc_key(tc)]["peak_c"]
                        for n in names if n in wo for room in res["rooms_c"]), default=None) for tc in tcs}
        lines += ["", "The demo course's own load: its nominal walking segment (`walk_0.25`, "
                  f"{wl['runs'][names[0]]['seconds_measured']:g} s measured, nominal model; `{wl['source']}`), the "
                  "worst drive's RMS torque over its rating and where that drive would settle walked without a "
                  f"break, C, method A, {SHOWN_TC / 60:g} min (in brackets: {min(tcs) / 60:g}-{max(tcs) / 60:g} "
                  f"min). It differs from H_all: heavier for {', '.join(heavier) or 'none'}; lighter for "
                  f"{', '.join(lighter) or 'none'}. So these are the demo's no-break figures. Where a walk-only demo "
                  f"was run they agree with its method A peaks to {_f(gap[SHOWN_TC])} C at {SHOWN_TC / 60:g} min; at "
                  f"{max(tcs) / 60:g} min its 96 min peaks are up to {_f(gap[max(tcs)])} C lower, not yet settled.",
                  "",
                  "| Room | " + " | ".join(f"{n} ({wl['runs'][n]['load']:.3f}, {wl['runs'][n]['joint']})"
                                         for n in names) + " |",
                  "|---|" + "---|" * len(names)]
        for room in res["rooms_c"]:
            cells = []
            for n in names:
                v = wl["runs"][n]["steady_c"][f"{room:g}"]
                vals = [v[tc_key(tc)] for tc in tcs]
                cells.append("runs away" if any(x is None for x in vals) else
                             f"{v[k8]:.1f} ({min(vals):.1f}-{max(vals):.1f})")
            lines.append(f"| {room:g} C | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(HERE / "results" / "heat35"))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = analyse()
    (out / "thermal_ambient_2026-11-27.json").write_text(json.dumps(res, indent=1) + "\n")
    md = markdown(res)
    (out / "thermal_ambient_2026-11-27.md").write_text(md)
    print(md)
    n = sum(len(b["runs"]) for b in res["sets"].values())
    print(f"25 C rerun reproduces the stored temperatures exactly on {n} of {n} demo courses")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
