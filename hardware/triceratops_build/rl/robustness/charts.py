"""Charts of a robustness sweep (sweep.py's JSON): one PNG per sweep, labelled SIMULATION with the run name.

  python charts.py results/pipe1/sweep_pipe1.json   (rewrites the PNGs beside the JSON)

Small multiples sharing the swept factor on x, one measure per panel (never two y-scales on one plot): forward speed,
falls, sideways drift, the worst joint's RMS torque against its continuous rating, and the worst joint's time at its
drive limit; for the push sweep, falls, speed, the centre-of-mass velocity change achieved, the extra sideways
displacement, the recovery time and the peak roll. Lines join the point means (episodes that did not fall); dots are
single episodes. The training range is shaded. Colours: the dataviz reference palette (slots 1 and 2, validated).
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#ecebe5"
SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2, light mode
DPI = 110
MAX_BYTES = 500 * 1024

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 9, "savefig.facecolor": SURFACE,
})


# ---------------------------------------------------------------- data access
def _summ(res, key):
    return res["points"][key]["summary"]


def _eps(res, key):
    return res["points"][key]["episodes"]


def _fam(res, name):
    return next(f for f in res["families"] if f["family"] == name)


def series_from_family(res, fam, xmap=lambda x: x):
    """[(x, summary, episodes)] for one family, x mapped to a number."""
    return [(xmap(r["x"]), _summ(res, r["key"]), _eps(res, r["key"])) for r in fam["rows"]]


# ---------------------------------------------------------------- panels
def _line(ax, xs, ys, color, label=None):
    xs, ys = np.asarray(xs, float), np.asarray([np.nan if y is None else y for y in ys], float)
    ax.plot(xs, ys, color=color, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=3, label=label)
    ax.plot(xs, ys, ls="none", marker="o", ms=7, mfc=color, mec=SURFACE, mew=1.6, zorder=4)


def _dots(ax, pts, color):
    if pts:
        x, y = zip(*pts)
        ax.scatter(x, y, s=9, color=color, alpha=0.35, lw=0, zorder=2)


def _fall_labels(ax, rows, outward=False):
    """'k/N' beside each point with a fall; pushed outward (away from x = 0) when ``outward``."""
    for x, s, _ in rows:
        if s["falls"]:
            side = (1 if x > 0 else -1) if outward else 0
            ax.annotate(f"{s['falls']}/{s['episodes']}", (x, 100 * s["fall_rate"]), textcoords="offset points",
                        xytext=(7 * side, 7 if not side else -3), ha={1: "left", -1: "right", 0: "center"}[side],
                        va="bottom" if not side else "top", fontsize=7.5, color=INK2)


def _band(ax, rng, label=False):
    if rng is None:
        return
    ax.axvspan(rng[0], rng[1], color=BAND, lw=0, zorder=0)
    if label:
        ax.text((rng[0] + rng[1]) / 2, 1.0, "training range", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=8, color=MUTED)


def _hline(ax, y, text, side="right"):
    ax.axhline(y, color=MUTED, lw=0.8, zorder=1)
    x = 0.99 if side == "right" else 0.01
    ax.text(x, y, text, transform=ax.get_yaxis_transform(), ha=side, va="bottom", fontsize=7.5, color=INK2)


def standard_panels(axes, series, band, xlabel, breaks_x=(), drop_lines=(), xticks=None, xticklabels=None):
    """series: [(name, color, [(x, summary, episodes)], nominal_speed)]. Five panels, or six with the feet's slip."""
    ax_v, ax_f, ax_d, ax_h, ax_s = axes[:5]
    if len(axes) > 5:
        ax_p = axes[5]
        ax_p.set_title("Slip ratio, worst foot", pad=14)
        _band(ax_p, band)
        for name, color, rows, _ in series:
            _line(ax_p, [x for x, _, _ in rows], [s.get("slip_worst") for _, s, _ in rows], color)
        ax_p.set_ylim(bottom=0)
    titles = ("Forward speed (m/s)", "Falls (% of episodes)", "Sideways drift, |m| over 18 s",
              "Heat: worst joint RMS / rating", "Worst joint, time at limit (%)")
    for ax, t in zip(axes, titles):
        ax.set_title(t, pad=14)
        _band(ax, band, label=(ax is ax_v))
    for name, color, rows, _ in series:
        xs = [x for x, _, _ in rows]
        _line(ax_v, xs, [s.get("speed_mps") for _, s, _ in rows], color, name)
        _dots(ax_v, [(x, e["forward_speed_mps"]) for x, _, eps in rows for e in eps
                     if not e["fell"] and "forward_speed_mps" in e], color)
        _line(ax_f, xs, [100 * s["fall_rate"] for _, s, _ in rows], color)
        _fall_labels(ax_f, rows)
        _line(ax_d, xs, [s.get("drift_abs_mean_m") for _, s, _ in rows], color)
        _dots(ax_d, [(x, abs(e["sideways_drift_m"])) for x, _, eps in rows for e in eps
                     if not e["fell"] and "sideways_drift_m" in e], color)
        _line(ax_h, xs, [s.get("heat_worst") for _, s, _ in rows], color)
        _line(ax_s, xs, [None if s.get("at_point_limit_worst") is None else 100 * s["at_point_limit_worst"]
                         for _, s, _ in rows], color)
    for y, text in drop_lines:
        _hline(ax_v, y, text, side="left")
    for bx in breaks_x:
        for ax in (ax_v, ax_f):
            ax.axvline(bx, color=INK2, lw=0.8, zorder=1)
        ax_f.text(bx, 0.97, "first break ", transform=ax_f.get_xaxis_transform(), rotation=90, ha="right",
                  va="top", fontsize=7.5, color=INK2)
    _hline(ax_h, 1.0, "continuous rating")
    ax_f.set_ylim(-5, 112)
    ax_v.set_ylim(0, 1.15 * ax_v.get_ylim()[1])  # headroom for the training-range label
    ax_d.set_ylim(bottom=0)
    ax_s.set_ylim(bottom=0)
    ylo, yhi = ax_h.get_ylim()
    ax_h.set_ylim(min(ylo, 0.8), max(yhi, 1.1))
    for ax in axes:
        ax.set_xlabel(xlabel)
        if xticks is not None:
            ax.set_xticks(xticks)
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels)


def header(fig, res, title, subtitle, legend_handles=None):
    fig.text(0.012, 0.985, f"SIMULATION  ·  {res['run_label']}  ·  {title}", ha="left", va="top", fontsize=13,
             fontweight="bold", color=INK)
    width = int(fig.get_figwidth() * 15.5)  # characters per line at 8.5 pt
    fig.text(0.012, 1 - 0.5 / fig.get_figheight(), textwrap.fill(subtitle, width), ha="left", va="top", fontsize=8.5,
             color=INK2, linespacing=1.4)
    fig.text(0.988, 0.012, "Simulation of a model whose masses are estimates; one policy, so a report, not a verdict. "
             f"rl/robustness/sweep.py, {res['date']}.", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.988, 0.985), ncol=len(legend_handles),
                   fontsize=9, handlelength=1.8)


def _common_sub(res, extra=""):
    return (f"{res['seeds_per_point']} episodes per point (20 s, deterministic actions, observation noise on), every "
            f"other factor nominal: the model as built, {res['settings']['nominal_delay_ms']} ms servo delay, "
            f"{res['settings']['command_mps']} m/s command. Lines: mean of the episodes that did not fall; dots: single "
            f"episodes. {extra}")


def _handles(names_colors):
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=c, lw=2, marker="o", ms=6, mfc=c, mec=SURFACE, label=n) for n, c in names_colors]


def _breaks_of(fam):
    return [b["x"] for b in fam.get("breaks", {}).values() if b]


def _save(fig, path):
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    if path.stat().st_size > MAX_BYTES:
        raise RuntimeError(f"{path} is {path.stat().st_size} bytes, over {MAX_BYTES}")
    return path


# ---------------------------------------------------------------- one chart per sweep
def chart_delay(res, out):
    fams = [f for f in res["families"] if f["sweep"] == "delay"]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.6))
    series, drops, brk = [], [], []
    for fam, color in zip(fams, SERIES):
        rows = series_from_family(res, fam)
        series.append((fam["family"].split(", ")[1] + " command", color, rows, fam["nominal_speed"]))
        if fam["nominal_speed"]:
            drops.append((0.8 * fam["nominal_speed"], f"-20 % of nominal, {fam['family'].split(', ')[1]}"))
        brk += _breaks_of(fam)
    xs = [r["x"] for r in fams[0]["rows"]]
    standard_panels(axes, series, res["settings"]["training_ranges"]["delay_ms"], "servo delay (ms)",
                    breaks_x=sorted(set(brk)), drop_lines=drops, xticks=[x for x in xs if x != 2])
    for name, color, rows, _ in series:  # direct labels at the line ends (the two speeds stay apart)
        x, s, _ = rows[-1]
        if s.get("speed_mps") is not None:
            axes[0].annotate(name.split(" ")[0], (x, s["speed_mps"]), textcoords="offset points", xytext=(6, 0),
                             va="center", fontsize=8, color=INK2)
    header(fig, res, "servo delay sweep", _common_sub(res, "Delay swept at the 0.25 and the 0.15 m/s command."),
           _handles([(n, c) for n, c, _, _ in series]))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.74, bottom=0.14, wspace=0.28)
    return _save(fig, out / f"{res['run']}_delay.png")


def chart_friction(res, out):
    fam = _fam(res, "friction")
    rows = series_from_family(res, fam, lambda x: 0.85 if x is None else x)
    fig, axes = plt.subplots(1, 6, figsize=(19, 4.6))
    standard_panels(axes, [("friction", SERIES[0], rows, fam["nominal_speed"])],
                    res["settings"]["training_ranges"]["friction"], "sliding friction, floor and soles",
                    breaks_x=[0.85 if x is None else x for x in _breaks_of(fam)],
                    drop_lines=[(0.8 * fam["nominal_speed"], "-20 % of nominal")] if fam["nominal_speed"] else (),
                    xticks=[x for x, _, _ in rows if x != 0.85])
    header(fig, res, "floor friction sweep",
           _common_sub(res, "The model's own friction (floor 0.8, soles 0.85: 0.85 in contact) is the nominal point. "
                            "Slip ratio: sole travel while loaded / body travel (gait_pipe1's definition), the foot "
                            "with the highest mean."))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.74, bottom=0.14, wspace=0.28)
    return _save(fig, out / f"{res['run']}_friction.png")


def chart_mass(res, out):
    fm, fp = _fam(res, "mass, every body"), _fam(res, "payload on the chassis")
    nom = _summ(res, next(r["key"] for r in fm["rows"] if r["x"] == 1.0))["mass_kg"]
    rows_m = series_from_family(res, fm, lambda x: x * nom)
    rows_p = series_from_family(res, fp, lambda x: nom + x)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.6))
    series = [("every body's mass and inertia scaled", SERIES[0], rows_m, fm["nominal_speed"]),
              ("payload on the chassis", SERIES[1], rows_p, fp["nominal_speed"])]
    brk = [x * nom for x in _breaks_of(fm)] + [nom + x for x in _breaks_of(fp)]
    standard_panels(axes, series, (0.9 * nom, 1.1 * nom), "robot mass (kg)", breaks_x=brk,
                    drop_lines=[(0.8 * fm["nominal_speed"], "-20 % of nominal")] if fm["nominal_speed"] else (),
                    xticks=[round(x, 1) for x, _, _ in rows_m])
    header(fig, res, "mass sweep",
           _common_sub(res, f"Nominal {nom:.1f} kg. Payload +0.5 and +1.0 kg at the chassis's centre of mass. Shaded: "
                            "x0.9-1.1, the per-body training range."),
           _handles([(n, c) for n, c, _, _ in series]))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.74, bottom=0.14, wspace=0.28)
    return _save(fig, out / f"{res['run']}_mass.png")


def _two_row(res, out, fams, bands, xlabels, title, sub, fname, xmaps, ticklabels=None):
    fig, axes = plt.subplots(2, 5, figsize=(17, 8.2))
    for i, (fam, band, xl, xm) in enumerate(zip(fams, bands, xlabels, xmaps)):
        rows = series_from_family(res, fam, xm)
        xs = [x for x, _, _ in rows]
        standard_panels(axes[i], [(fam["family"], SERIES[0], rows, fam["nominal_speed"])], band, xl,
                        breaks_x=[xm(x) for x in _breaks_of(fam)],
                        drop_lines=[(0.8 * fam["nominal_speed"], "-20 % of nominal")] if fam["nominal_speed"] else (),
                        xticks=xs, xticklabels=ticklabels[i] if ticklabels else None)
        axes[i][0].text(-0.16, 0.5, fam["family"], transform=axes[i][0].transAxes, rotation=90, ha="center",
                        va="center", fontsize=10.5, fontweight="bold", color=INK)
    header(fig, res, title, sub)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.84, bottom=0.08, wspace=0.28, hspace=0.62)
    return _save(fig, out / fname)


def chart_drive(res, out):
    tr = res["settings"]["training_ranges"]
    return _two_row(res, out, [_fam(res, "force limit"), _fam(res, "position gain kp")],
                    [tr["force_scale"], tr["kp_scale"]], ["force limit, x nominal", "kp, x nominal"],
                    "drive strength sweep",
                    _common_sub(res, "The 12 leg drives only. Drive limit in the last panel: the scaled limit in the "
                                     "force-limit row."),
                    f"{res['run']}_drive.png", [lambda x: x, lambda x: x])


def chart_terrain(res, out):
    fs, fr = _fam(res, "step up"), _fam(res, "ramp up")
    xs_s = lambda x: 0.0 if x is None else 100 * x[1]  # noqa: E731
    xs_r = lambda x: 0.0 if x is None else x[1]  # noqa: E731
    return _two_row(res, out, [fs, fr], [None, None], ["step height (cm)", "ramp slope (deg)"], "step and ramp",
                    _common_sub(res, f"Added in memory {res['settings']['terrain_edge_m']} m ahead of the chassis; the "
                                     "policy is blind to it. Trained on a flat floor only."),
                    f"{res['run']}_terrain.png", [xs_s, xs_r])


def chart_push(res, out):
    fam = _fam(res, "sideways push")
    rows = series_from_family(res, fam)
    xs = [x for x, _, _ in rows]
    c = SERIES[0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.0))
    (a_f, a_v, a_c), (a_x, a_r, a_roll) = axes
    a_f.set_title("Falls (% of episodes)", pad=10)
    _line(a_f, xs, [100 * s["fall_rate"] for _, s, _ in rows], c)
    _fall_labels(a_f, rows, outward=True)
    a_f.set_ylim(-5, 112)
    a_v.set_title("Forward speed (m/s)", pad=10)
    _line(a_v, xs, [s.get("speed_mps") for _, s, _ in rows], c)
    _dots(a_v, [(x, e["forward_speed_mps"]) for x, _, eps in rows for e in eps
                if not e["fell"] and "forward_speed_mps" in e], c)
    if fam["nominal_speed"]:
        _hline(a_v, 0.8 * fam["nominal_speed"], "-20 % of nominal", side="left")
    a_v.set_ylim(bottom=0)
    a_c.set_title("Centre-of-mass velocity change achieved (m/s)", pad=10)  # paired with the unpushed episode
    a_c.plot(xs, np.abs(xs), color=MUTED, lw=0.8, zorder=1)
    a_c.text(xs[-1], abs(xs[-1]), "free body ", ha="right", va="bottom", fontsize=7.5, color=INK2)
    _line(a_c, xs, [(s.get("push") or {}).get("com_dv_achieved_mps_mean") for _, s, _ in rows], c)
    a_x.set_title("Extra sideways displacement within 3 s (m)", pad=10)
    _line(a_x, xs, [(s.get("push") or {}).get("lateral_extra_peak_m_mean") for _, s, _ in rows], c)
    _dots(a_x, [(x, e["push"]["lateral_extra_peak_m"]) for x, _, eps in rows for e in eps
                if e["push"].get("lateral_extra_peak_m") is not None], c)
    a_x.set_ylim(bottom=0)
    a_r.set_title("Recovery time (s): median, dots single episodes", pad=10)
    _line(a_r, xs, [(s.get("push") or {}).get("recovery_s_median") for _, s, _ in rows], c)
    _dots(a_r, [(x, e["push"]["recovery_s"]) for x, _, eps in rows for e in eps
                if e["push"].get("recovery_s") is not None], c)
    a_r.set_ylim(bottom=0)
    a_roll.set_title("Peak body roll after the push (deg), worst episode", pad=10)
    _line(a_roll, xs, [(s.get("push") or {}).get("roll_peak_deg_max") for _, s, _ in rows], c)
    a_roll.set_ylim(bottom=0)
    for bx in _breaks_of(fam):
        for ax in (a_f, a_v):
            ax.axvline(bx, color=INK2, lw=0.8, zorder=1)
    for ax in axes.ravel():
        ax.set_xlabel("push, free-body velocity change (m/s); + to the robot's left")
        ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        ax.axvline(0, color=AXIS, lw=0.8, zorder=0)
    st = res["settings"]
    header(fig, res, "sideways push",
           _common_sub(res, f"Push: a horizontal force on the chassis from t = {st['push_t_s']:g} s for "
                            f"{st['push_duration_s']:g} s, impulse = robot mass x dv\n(the velocity a free body would "
                            "gain). Extra displacement: against the same seed's unpushed episode. Recovery: the 0.5 s "
                            f"mean sideways speed back within {st['recover_vlat_mps']:g} m/s and held for "
                            f"{st['recover_hold_s']:g} s. No pushes in training."))
    fig.subplots_adjust(left=0.04, right=0.985, top=0.80, bottom=0.08, wspace=0.22, hspace=0.45)
    return _save(fig, out / f"{res['run']}_push.png")


CHARTS = {"delay": chart_delay, "friction": chart_friction, "mass": chart_mass, "drive": chart_drive,
          "push": chart_push, "terrain": chart_terrain}


def make_charts(res: dict, out: Path) -> dict:
    out = Path(out)
    present = {f["sweep"] for f in res["families"]}
    return {name: fn(res, out) for name, fn in CHARTS.items() if name in present}


if __name__ == "__main__":
    p = Path(sys.argv[1])
    for k, v in make_charts(json.loads(p.read_text()), p.parent).items():
        print(k, v, v.stat().st_size // 1024, "KB")
