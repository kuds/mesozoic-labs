"""Run 3's "reported, not judged" content (PREREG_RUN3.md, Reading): the full gait report's main figures and every
sweep point, for both seeds against run2b, read from the tools' stored JSON. Simulation only; no verdict here.

Usage: python hardware/triceratops_build/rl/analysis/run3_appendix.py [--out REPORT_RUN3_APPENDIX.md]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

RL = Path(__file__).resolve().parents[1]
RUNS = {"run 3 seed 0": "run3-s0", "run 3 seed 1": "run3-s1", "run2b": "run2b"}
FEET = ("left_fore", "right_fore", "left_hind", "right_hind")


def gait_all(run):
    return json.loads((RL / "analytics" / "results" / run / "gait_report.json").read_text())


def gait(run, case="target_0.25"):
    return gait_all(run)["cases"][case]["summary"]


def sweep(run):
    return json.load(gzip.open(RL / "robustness" / "results" / run / f"sweep_{run}.json.gz", "rt"))


def f(x, nd=2, unit=""):
    return "-" if x is None else f"{x:.{nd}f}{unit}"


def gait_tables(rows):
    g = {k: gait(v) for k, v in RUNS.items()}
    head = "| | " + " | ".join(RUNS) + " |\n|---|" + "---|" * len(RUNS) + "\n"
    out = ["## A. Gait report, 0.25 m/s target case (randomised, 10 ms delay, 5 episodes)\n"]

    def row(label, fn):
        out.append(f"| {label} | " + " | ".join(fn(g[k]) for k in RUNS) + " |")

    out.append("**Speed and straightness**\n\n" + head.rstrip("\n"))
    row("forward speed, m/s", lambda t: f(t["speed"]["forward_speed_mps"], 3))
    row("sideways drift, worst episode, m", lambda t: f(t["speed"]["sideways_drift_m_worst"], 2))
    row("heading change, worst episode, deg", lambda t: f(t["speed"]["heading_change_deg_worst"], 1))
    out.append("\n**Drive heat (RMS torque / continuous rating, mean of episodes) and torque limits (time at the "
               "episode's limit: the worst episode, averaged over the group's four drives)**\n\n" + head.rstrip("\n"))
    for j in g["run2b"]["drives"]["per_joint"]:
        row(j, lambda t, j=j: f(t["drives"]["per_joint"][j]["rms_over_continuous"]["mean"], 2))
    for grp in g["run2b"]["drives"]["groups"]:
        row(f"{grp}: time at the episode's torque limit, worst episode, mean of the group's drives, %",
            lambda t, grp=grp: f(100 * t["drives"]["groups"][grp]["saturated_fraction_episode_limit"], 1))
    out.append("\n**Gait**\n\n" + head.rstrip("\n"))
    row("pattern", lambda t: t["footfall"]["gait"]["class"])
    row("stride frequency, all feet, Hz", lambda t: f(t["footfall"]["stride_frequency_hz_all_feet"], 2))
    row("stride length, mean, m", lambda t: f(t["footfall"]["stride_length_m"]["mean"], 3))
    row("irregular strides, worst foot, %",
        lambda t: f(100 * max(t["footfall"]["irregular_stride_fraction"].values()), 0))
    row("contact gaps of 20-80 ms (all feet)", lambda t: str(sum(
        t["footfall"]["contact_gaps"][ft][b]["count"] for ft in FEET for b in ("20-50 ms", "50-80 ms"))))
    row("two diagonal feet down, % of time", lambda t: f(100 * t["footfall"]["feet_down_fraction"]["2 diagonal"], 0))
    out.append("\n**Power and run time (estimates)**\n\n" + head.rstrip("\n"))
    row("mechanical power, absolute, W", lambda t: f(t["power"]["mech_abs_w"], 1))
    row("electrical, shared bus, W", lambda t: f(t["power"]["electrical_shared_bus_w"], 1))
    row("walking time per pack, shared bus, 30 W hotel, min",
        lambda t: f(t["power"]["run_time_min_shared_bus"]["hotel_30w"], 0))
    out.append("\n**Feet and body**\n\n" + head.rstrip("\n"))
    row("impact peak, median, worst foot, body weights", lambda t: f(max(t["feet"]["impact_peak_bw_median"].values()), 2))
    row("stance slip ratio at the sole's origin, worst foot (on the capsule this counts rolling as slip; "
        "PREREG_RUN3.md)",
        lambda t: f(max(v["slip_ratio_mean"] for v in t["feet"]["stance_slip"].values()), 3))
    row("stance slip ratio at the contacts, worst foot",
        lambda t: f(max(v["slip_ratio_mean"] for v in t["feet"].get("stance_slip_at_contact", {}).values())
                    if t["feet"].get("stance_slip_at_contact") else None, 3))
    row("body roll: mean, deg / SD, deg", lambda t: f"{f(t['body']['roll_deg']['mean'], 2)} / "
                                                     f"{f(t['body']['roll_deg']['sd'], 2)}")
    row("body pitch: mean (negative: nose down), deg / SD, deg", lambda t: f"{f(t['body']['pitch_deg']['mean'], 2)} / "
                                                                          f"{f(t['body']['pitch_deg']['sd'], 2)}")
    out.append("\n**Balance (reported, not compared: the capsule stands on a line, run2b's box sole on an area)**\n\n"
               + head.rstrip("\n"))
    out.append(f"| centre of mass outside the support polygon, % of time | "
               f"{f(100 * g['run 3 seed 0']['balance']['com_margin_negative_fraction'], 1)} | "
               f"{f(100 * g['run 3 seed 1']['balance']['com_margin_negative_fraction'], 1)} | not compared |")
    labels = gait_all("run2b")["cases"]
    out.append("\n**The gait report's other cases** (5 episodes each)\n\n| case | measure | " + " | ".join(RUNS)
               + " |\n|---|---|" + "---|" * len(RUNS))
    for case in ("cmd_0.15", "nominal_0.25", "otherfoot_0.25"):
        c = {k: gait(v, case) for k, v in RUNS.items()}
        name = labels[case]["label"]
        for label, fn in (
                ("falls of 5", lambda t: str(t["falls"])),
                ("forward speed, m/s", lambda t: f((t.get("speed") or {}).get("forward_speed_mps"), 3)),
                ("pattern, stride frequency Hz", lambda t: f"{t['footfall']['gait']['class']}, "
                                                           f"{f(t['footfall']['stride_frequency_hz_all_feet'], 2)}"),
                ("irregular strides, worst foot, %",
                 lambda t: f(100 * max(t["footfall"]["irregular_stride_fraction"].values()), 0)),
                ("worst joint RMS / continuous rating",
                 lambda t: f(max(d["rms_over_continuous"]["mean"] for d in t["drives"]["per_joint"].values()), 2))):
            out.append(f"| {name} | {label} | " + " | ".join(fn(c[k]) for k in RUNS) + " |")
    sw = {k: gait_all(v)["sweep"] for k, v in RUNS.items()}
    out.append("\n**The gait report's command sweep** (falls; mean forward speed m/s)\n\n| command, m/s | "
               + " | ".join(RUNS) + " |\n|---|" + "---|" * len(RUNS))
    for cmd in sw["run2b"]:
        out.append(f"| {cmd} | " + " | ".join(f"{sw[k][cmd]['falls']}; {f(sw[k][cmd].get('forward_speed_mps_mean'), 3)}"
                                             for k in RUNS) + " |")
    rows.extend(out)


def sweep_tables(rows):
    s = {k: sweep(v) for k, v in RUNS.items()}
    out = ["\n## B. Every sweep point (nominal model, noise on, 8 episodes, 0.25 m/s unless stated)\n",
           "Each cell: falls of 8; speed m/s; sideways drift, mean, m; worst joint RMS / continuous rating; the "
           "largest share of time any joint spent at the nominal torque limit, % (that joint may differ from the "
           "hottest). Speed, drift and heat are over the episodes that did not "
           "fall; \"all fell\" has none.\n"]
    for fam in s["run2b"]["families"]:
        out.append(f"**{fam['family']}** ({fam['x_key']})\n")
        out.append("| x | " + " | ".join(RUNS) + " |\n|---|" + "---|" * len(RUNS))
        for r in fam["rows"]:
            cells = []
            for k in RUNS:
                p = s[k]["points"][r["key"]]["summary"]
                if p["falls"] == p["episodes"]:
                    cells.append(f"{p['falls']}, all fell")
                    continue
                cells.append(f"{p['falls']}; {f(p.get('speed_mps'), 3)}; {f(p.get('drift_abs_mean_m'), 2)}; "
                             f"{f(p.get('heat_worst'), 2)}; {f(100 * p['at_nominal_limit_worst'] if p.get('at_nominal_limit_worst') is not None else None, 1)}")
            x = "model" if r["x"] is None else (" ".join(map(str, r["x"])) if isinstance(r["x"], list) else r["x"])
            out.append(f"| {x} | " + " | ".join(cells) + " |")
        out.append("")
    rows.extend(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "REPORT_RUN3_APPENDIX.md"))
    a = ap.parse_args(argv)
    rows = ["# Run 3, appendix: everything reported, not judged (simulation)\n",
            "Generated by `analysis/run3_appendix.py` from the stored tool outputs (`analytics/results/run3-s0`, "
            "`run3-s1`, `run2b`; `robustness/results/...`). Run 3 and run2b differ in the foot, the slip cost, the "
            "drift cost, the training length and the terrain, push and friction training (see REPORT_RUN3.md), so "
            "no difference below is put down to one of them. Steps to 4 cm, ramps to 10 deg, pushes to 1.0 m/s and "
            "friction from 0.6 down to 0.3 were within run 3's training and not run2b's; both trained with delay "
            "0-10 ms, mass x0.9-1.1, gain x0.9-1.1, force limit x0.85-1.0 and friction 0.6-1.0 (the sweeps' "
            "recorded training ranges).\n"]
    gait_tables(rows)
    sweep_tables(rows)
    Path(a.out).write_text("\n".join(rows) + "\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
