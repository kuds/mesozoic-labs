"""Training dashboard: a run's periodic evaluations against training steps (PNG). Simulation only.

Usage:
  python hardware/triceratops_build/rl/analytics/training_dashboard.py \
      --run NAME EVAL_JSONL [RUN_RECORD_JSON] [--run NAME2 EVAL_JSONL2 [RECORD2]] [--out DIR]

EVAL_JSONL is a run's ``eval_during_training.jsonl``: one JSON object per periodic evaluation, written by train.py's
PeriodicEval callback: ``{"steps", "time_utc", "deterministic", "delay_steps", "episodes": [{"command_mps",
"forward_speed_mps", "fell", "termination", "length_s", "measures"?}]}``, one 20 s episode per evaluation command
(seeds 900000 + i, randomisation on, 10 ms delay, deterministic). Runs before the gait measures (trike60-pipe1)
carry only the speed fields; newer runs add ``measures`` (``rl/gait_metrics.measures``: drive heat and at-limit time
per group, slip ratio and debounced touchdowns per foot, drift, heading change, power). Both formats are read; the
gait panels say when a file has no measures. The file is read as data only (``json.loads`` per line); malformed
lines are skipped and counted.

RUN_RECORD_JSON (the run's ``run_record.json``) supplies the penalty schedules when it records them:
``penalty_ramp_steps`` / ``penalty_ramp_start`` (energy and slip penalties, 0.1 -> 1) and ``late_ramp_from`` /
``late_ramp_to`` (heat, saturation and short air-time penalties, 0 -> 1); the scales are train.py's own functions.

Writes ``training_<NAME>.png`` per run, ``training_compare.png`` for two or more runs, and ``training_dashboard.json``
(the parsed series) into --out (default: results/<first NAME>/ beside this file).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL = HERE.parent
for p in (str(HERE), str(RL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")
SHORT = {"left_fore": "LF", "right_fore": "RF", "left_hind": "LH", "right_hind": "RH"}
GROUPS = ("roll", "pitch", "front elbow", "hind knee")


# ---------------------------------------------------------------- parsing
def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_eval_lines(lines) -> dict:
    """Parse the lines of an eval_during_training.jsonl. Returns ``{"records": [...], "skipped": n, "format": ...}``,
    records sorted by steps (a later line for the same steps replaces an earlier one, as a resumed run re-evaluates),
    each ``{"steps", "episodes": [{"command", "speed", "fell", "termination", "length_s", "measures"}]}``."""
    by_steps: dict[int, dict] = {}
    skipped = 0
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            steps = int(obj["steps"])
            eps = []
            for e in obj.get("episodes", []):
                m = e.get("measures")
                eps.append({"command": _num(e.get("command_mps")), "speed": _num(e.get("forward_speed_mps")),
                            "fell": bool(e.get("fell", False)), "termination": str(e.get("termination", "")),
                            "length_s": _num(e.get("length_s")), "measures": m if isinstance(m, dict) else None})
        except (ValueError, KeyError, TypeError, AttributeError):
            skipped += 1
            continue
        by_steps[steps] = {"steps": steps, "episodes": eps}
    records = [by_steps[k] for k in sorted(by_steps)]
    has = any(e["measures"] for r in records for e in r["episodes"])
    return {"records": records, "skipped": skipped, "format": "with gait measures" if has else "speeds only"}


def parse_eval_file(path) -> dict:
    return parse_eval_lines(Path(path).read_text(encoding="utf-8", errors="replace").splitlines())


def schedules(record: dict | None) -> dict:
    """The run's penalty schedules from its run_record.json (settings), when recorded."""
    st = (record or {}).get("settings") or {}
    out = {}
    if st.get("penalty_ramp_steps") is not None:
        out["penalty"] = {"ramp_steps": int(st["penalty_ramp_steps"]), "start": float(st.get("penalty_ramp_start", 0.1))}
    if st.get("late_ramp_from") is not None and st.get("late_ramp_to") is not None:
        out["late"] = {"from": int(st["late_ramp_from"]), "to": int(st["late_ramp_to"])}
    return out


def schedule_curves(sch: dict, steps: np.ndarray) -> dict:
    from train import late_scale_at, penalty_scale_at  # train.py's own definitions

    out = {}
    if "penalty" in sch:
        out["penalty scale (energy, slip)"] = [penalty_scale_at(int(s), sch["penalty"]["ramp_steps"],
                                                                sch["penalty"]["start"]) for s in steps]
    if "late" in sch:
        out["late scale (heat, saturation, short air time)"] = [late_scale_at(int(s), sch["late"]["from"],
                                                                              sch["late"]["to"]) for s in steps]
    return out


def series(parsed: dict) -> dict:
    """Plot-ready series. Speeds and falls from every episode; gait measures from each evaluation's episode at the
    highest command (as train.py prints it)."""
    recs = parsed["records"]
    steps = np.array([r["steps"] for r in recs], dtype=float)
    cmds = sorted({e["command"] for r in recs for e in r["episodes"] if e["command"] is not None})
    out = {"steps": steps.tolist(), "commands": cmds, "speed": {}, "falls": [], "episodes_per_eval": [],
           "gait_from_command": None, "gait": {}}
    for c in cmds:
        out["speed"][f"{c:.2f}"] = [next((e["speed"] for e in r["episodes"] if e["command"] == c), None) for r in recs]
    out["falls"] = [sum(e["fell"] for e in r["episodes"]) for r in recs]
    out["episodes_per_eval"] = [len(r["episodes"]) for r in recs]
    if parsed["format"] != "with gait measures":
        return out
    top = max(cmds)
    out["gait_from_command"] = top

    def pick(r):
        e = next((e for e in r["episodes"] if e["command"] == top and e["measures"]), None)
        return e["measures"] if e else None

    ms = [pick(r) for r in recs]

    def get(m, *keys):
        for k in keys:
            if not isinstance(m, dict) or k not in m:
                return None
            m = m[k]
        return _num(m)

    g = out["gait"]
    g["rms_over_continuous_worst"] = {grp: [get(m, "drive_groups", grp, "rms_over_continuous_worst") for m in ms]
                                      for grp in GROUPS}
    g["at_limit_nominal"] = {grp: [get(m, "drive_groups", grp, "at_limit_nominal") for m in ms] for grp in GROUPS}
    g["slip_ratio"] = {c: [get(m, "slip_ratio", c) for m in ms] for c in CORNERS}
    g["touchdowns_per_s"] = {c: [get(m, "touchdowns_per_s", c) for m in ms] for c in CORNERS}
    g["sideways_drift_m"] = [get(m, "sideways_drift_m") for m in ms]
    g["heading_change_deg"] = [get(m, "heading_change_deg") for m in ms]
    g["power_w"] = [get(m, "power_w") for m in ms]
    return out


# ---------------------------------------------------------------- charts
def _xy(steps, vals):
    x = np.asarray(steps, dtype=float) / 1e6
    y = np.array([np.nan if v is None else v for v in vals], dtype=float)
    return x, y


def _mark_schedules(ax, sch: dict) -> None:
    import chart_style as cs

    if "late" in sch:
        ax.axvspan(sch["late"]["from"] / 1e6, sch["late"]["to"] / 1e6, color=cs.WASH, lw=0, zorder=0)
    if "penalty" in sch:
        ax.axvline(sch["penalty"]["ramp_steps"] / 1e6, color=cs.AXIS, lw=0.9, zorder=0)


def _no_data(ax, text):
    import chart_style as cs

    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=8.5, color=cs.MUTED, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _lines(ax, steps, data: dict, labels: dict, colors, end_fmt="{:.2f}"):
    import chart_style as cs

    for (k, vals), col in zip(data.items(), colors):
        x, y = _xy(steps, vals)
        if np.all(np.isnan(y)):
            continue
        ax.plot(x, y, color=col, lw=cs.LW, marker="o", ms=3.5, mec=cs.SURFACE, mew=1.0, label=labels.get(k, k))
        last = np.flatnonzero(~np.isnan(y))[-1]
        ax.annotate(end_fmt.format(y[last]), (x[last], y[last]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=cs.INK2, va="center")


def dashboard(name: str, parsed: dict, record: dict | None, out: Path, source: str) -> str:
    import chart_style as cs

    cs.setup()
    plt = cs.plt
    s = series(parsed)
    sch = schedules(record)
    steps = np.asarray(s["steps"])
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(3, 4, left=0.05, right=0.975, top=0.86, bottom=0.08, hspace=0.45, wspace=0.27)
    axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(12)]
    X = "training steps, millions"
    no_gait = ("not in this run's file: it records speeds only\n(runs before the gait measures, e.g. trike60-pipe1)"
               if parsed["format"] != "with gait measures" else None)

    # 1 speed per command
    ax = axes[0]
    cols = cs.SERIES[:3] if len(s["commands"]) <= 3 else cs.SERIES[:len(s["commands"])]
    _lines(ax, steps, s["speed"], {k: f"command {k} m/s" for k in s["speed"]}, cols, "{:.3f}")
    for c in s["commands"]:
        ax.axhline(c, color=cs.INK2, ls="--", lw=0.8)
    ax.set_title("Forward speed per command")
    ax.set_ylabel("m/s (dashed: command)")
    ax.legend(loc="upper left", fontsize=7.5)
    # 2 falls
    ax = axes[1]
    x, y = _xy(steps, s["falls"])
    width = 0.6 * (np.median(np.diff(x)) if len(x) > 1 else 0.25)
    ax.bar(x, y, width=width, color=cs.SERIES[0])
    ax.set_ylim(0, max(max(s["episodes_per_eval"] or [1]), 1) + 0.3)
    ax.set_title("Falls per evaluation")
    ax.set_ylabel(f"falls (of {max(s['episodes_per_eval'] or [0])} episodes)")
    if not any(s["falls"]):
        ax.text(0.5, 0.55, "no falls in any evaluation", transform=ax.transAxes, ha="center", fontsize=8.5,
                color=cs.INK2)
    # 3 schedules
    ax = axes[2]
    if sch:
        xs = np.linspace(0, max(steps.max() if steps.size else 1, sch.get("late", {}).get("to", 0)) * 1.0, 200)
        curves = schedule_curves(sch, xs)
        for (k, v), col in zip(curves.items(), cs.SERIES[:2]):
            ax.plot(xs / 1e6, v, color=col, lw=cs.LW, label=k)
        ax.set_ylim(-0.03, 1.08)
        ax.legend(loc="lower right", fontsize=7.5)
        ax.set_ylabel("scale of the penalty")
    else:
        _no_data(ax, "no schedules in the run record")
    ax.set_title("Penalty schedules (run record)")
    # 4 power
    ax = axes[3]
    if no_gait:
        _no_data(ax, no_gait)
    else:
        _lines(ax, steps, {"power": s["gait"]["power_w"]}, {}, cs.SERIES[:1], "{:.0f}")
        ax.set_ylabel("W, sum |torque x speed|")
    ax.set_title("Joint mechanical power")
    # 5 heat
    ax = axes[4]
    if no_gait:
        _no_data(ax, no_gait)
    else:
        _lines(ax, steps, s["gait"]["rms_over_continuous_worst"], {}, cs.SERIES[:4])
        ax.axhline(1.0, color=cs.INK, ls="--", lw=0.9)
        ax.set_ylabel("worst joint's RMS / rating")
        ax.legend(loc="best", fontsize=7.5)
    ax.set_title("Heat: RMS torque / continuous rating")
    # 6 at limit
    ax = axes[5]
    if no_gait:
        _no_data(ax, no_gait)
    else:
        _lines(ax, steps, {g: [None if v is None else 100 * v for v in vals]
                           for g, vals in s["gait"]["at_limit_nominal"].items()}, {}, cs.SERIES[:4], "{:.1f}")
        ax.set_ylabel("% of 1 ms steps at >= 99 % of the limit")
        ax.legend(loc="best", fontsize=7.5)
    ax.set_title("Time at the nominal torque limit")
    # 7 slip
    ax = axes[6]
    if no_gait:
        _no_data(ax, no_gait)
    else:
        _lines(ax, steps, s["gait"]["slip_ratio"], SHORT, cs.SERIES[:4])
        ax.set_ylabel("sole travel loaded / body travel")
        ax.legend(loc="best", fontsize=7.5, ncol=2)
    ax.set_title("Slip ratio per foot")
    # 8 touchdowns
    ax = axes[7]
    if no_gait:
        _no_data(ax, no_gait)
    else:
        _lines(ax, steps, s["gait"]["touchdowns_per_s"], SHORT, cs.SERIES[:4], "{:.1f}")
        ax.set_ylabel("per second (10 ms debounce)")
        ax.legend(loc="best", fontsize=7.5, ncol=2)
    ax.set_title("Debounced touchdowns per foot")
    # 9 drift
    ax = axes[8]
    if no_gait:
        _no_data(ax, no_gait)
    else:
        _lines(ax, steps, {"drift": s["gait"]["sideways_drift_m"]}, {}, cs.SERIES[:1], "{:+.2f}")
        ax.axhline(0, color=cs.AXIS, lw=0.8)
        ax.set_ylabel("m over 18 s (+ left)")
    ax.set_title("Sideways drift")
    # 10 heading
    ax = axes[9]
    if no_gait:
        _no_data(ax, no_gait)
    else:
        _lines(ax, steps, {"heading": s["gait"]["heading_change_deg"]}, {}, cs.SERIES[:1], "{:+.1f}")
        ax.axhline(0, color=cs.AXIS, lw=0.8)
        ax.set_ylabel("deg over 18 s")
    ax.set_title("Heading change")
    for i in range(10):
        if axes[i].get_xticks().size:
            _mark_schedules(axes[i], sch)
            axes[i].set_xlabel(X)
            axes[i].set_xlim(0, (steps.max() / 1e6 if steps.size else 1) * 1.08)
    # 11-12 notes
    for ax in axes[10:]:
        ax.axis("off")
    notes = [f"File: {source}", f"Format: {parsed['format']}; {len(parsed['records'])} evaluations"
             + (f"; {parsed['skipped']} malformed lines skipped" if parsed["skipped"] else ""),
             "Each evaluation: one 20 s episode per command, deterministic,",
             "randomisation on, 10 ms delay, fixed seeds (900000 + i).",
             (f"Gait panels: the episode at {s['gait_from_command']:.2f} m/s, measured from 2 s."
              if s["gait_from_command"] is not None else "Gait panels: no measures recorded."), ""]
    if sch:
        if "penalty" in sch:
            notes.append(f"Vertical line: energy and slip penalties reach full weight ({sch['penalty']['ramp_steps']:,} "
                         "steps).")
        if "late" in sch:
            notes.append(f"Shaded: heat, saturation and short air-time penalties ramp in ({sch['late']['from']:,} to "
                         f"{sch['late']['to']:,} steps).")
    else:
        notes.append("No run record: schedules not marked.")
    notes += ["", "One evaluation episode per point: these curves are noisy;", "a single point is not a verdict."]
    axes[10].text(0.0, 1.0, "\n".join(notes), transform=axes[10].transAxes, va="top", fontsize=8.2, color=cs.INK2,
                  linespacing=1.5)
    cs.frame(fig, f"Training progress: {name}",
             "Periodic deterministic evaluations during training (eval_during_training.jsonl), against training steps",
             f"SIMULATION. {name}. Source: rl/analytics/training_dashboard.py.")
    fname = f"training_{name}.png"
    cs.save(fig, out / fname)
    return fname


def compare(runs: list[tuple[str, dict, dict | None]], out: Path) -> str:
    """Two or more runs on shared axes: speed at each run's highest command, falls, worst heat ratio, mean slip."""
    import chart_style as cs

    cs.setup()
    plt = cs.plt
    runs = runs[:3]  # the categorical cap for overlapping lines; more runs: draw several comparisons
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(1, 4, left=0.05, right=0.985, top=0.78, bottom=0.14, wspace=0.28)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for (name, parsed, _rec), col in zip(runs, cs.SERIES):
        s = series(parsed)
        steps = np.asarray(s["steps"])
        top = f"{max(s['commands']):.2f}" if s["commands"] else None
        if top:
            x, y = _xy(steps, s["speed"][top])
            axes[0].plot(x, y, color=col, lw=cs.LW, marker="o", ms=3.5, label=f"{name} (command {top})")
        x, y = _xy(steps, s["falls"])
        axes[1].plot(x, y, color=col, lw=cs.LW, marker="o", ms=3.5, label=name)
        if s["gait"]:
            worst = [max([v[i] for v in s["gait"]["rms_over_continuous_worst"].values() if v[i] is not None], default=None)
                     for i in range(len(steps))]
            x, y = _xy(steps, worst)
            axes[2].plot(x, y, color=col, lw=cs.LW, marker="o", ms=3.5, label=name)
            slip = [np.mean([v[i] for v in s["gait"]["slip_ratio"].values() if v[i] is not None])
                    if any(v[i] is not None for v in s["gait"]["slip_ratio"].values()) else None for i in range(len(steps))]
            x, y = _xy(steps, slip)
            axes[3].plot(x, y, color=col, lw=cs.LW, marker="o", ms=3.5, label=name)
    titles = ("Forward speed at the highest command", "Falls per evaluation", "Worst joint RMS / rating",
              "Slip ratio, mean of the feet")
    ylabels = ("m/s", "falls", "RMS torque / continuous rating", "sole travel loaded / body travel")
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_title(t)
        ax.set_ylabel(yl)
        ax.set_xlabel("training steps, millions")
        if ax.get_lines():
            ax.legend(fontsize=7.5, loc="best")
        else:
            _no_data(ax, "no gait measures in these files")
    axes[2].axhline(1.0, color=cs.INK, ls="--", lw=0.9)
    cs.frame(fig, "Training runs compared", ", ".join(n for n, _, _ in runs),
             "SIMULATION. Source: rl/analytics/training_dashboard.py.")
    cs.save(fig, out / "training_compare.png")
    return "training_compare.png"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", action="append", required=True, metavar=("NAME", "EVAL_JSONL"),
                    help="NAME EVAL_JSONL [RUN_RECORD_JSON]; repeat for more runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    runs = []
    for spec in a.run:
        if len(spec) not in (2, 3):
            ap.error("--run takes NAME EVAL_JSONL [RUN_RECORD_JSON]")
        rec = json.loads(Path(spec[2]).read_text()) if len(spec) == 3 else None
        runs.append((spec[0], parse_eval_file(spec[1]), rec, spec[1]))
    out = Path(a.out) if a.out else HERE / "results" / runs[0][0]
    out.mkdir(parents=True, exist_ok=True)
    files = [dashboard(n, p, r, out, Path(src).name) for n, p, r, src in runs]
    if len(runs) > 1:
        files.append(compare([(n, p, r) for n, p, r, _ in runs], out))
    summary = {n: {"format": p["format"], "evaluations": len(p["records"]), "skipped_lines": p["skipped"],
                   "schedules": schedules(r), "series": series(p)} for n, p, r, _ in runs}
    (out / "training_dashboard.json").write_text(json.dumps({"charts": files, "runs": summary}, indent=1) + "\n")
    for f in files:
        assert (out / f).stat().st_size < 500 * 1024, f
    print("wrote", ", ".join(str(out / f) for f in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
