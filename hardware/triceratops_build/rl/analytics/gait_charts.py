"""The gait report's charts (PNG, light surface), drawn from gait_report.py's results and payload. Simulation only.

Each function draws one file and returns its name. Time-series panels use the nominal-model episode (seed 100000,
t = 6-9 s); distributions and summary numbers use the 0.25 m/s randomised case (5 episodes) unless the chart says
otherwise. Every chart carries the SIMULATION tag and the run name."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

import chart_style as cs
import gait_core as gc

plt = cs.plt
CORNERS = gc.CORNERS
COP_COLOR, COM_COLOR = cs.SERIES[6], cs.SERIES[4]  # violet, magenta: validated as a pair (light)


def _run_label(R) -> str:
    return f"{R.get('label') or R['run']} ({R['policy']['steps']:,} steps)"


def _footer(R, what: str) -> str:
    return (f"SIMULATION. {_run_label(R)}. {what}. Deterministic actions, servo delay {R['settings']['delay_ms']} ms. "
            f"Source: rl/analytics/gait_report.py, {R['date']}.")


def _arr(x):
    return np.asarray(x, dtype=np.float64)


def _spans(t, flags):
    dt = float(np.median(np.diff(t)))
    return [(t[s], n * dt) for v, s, n in gc.runs(flags) if v]


def _end_label(ax, x, y, text, color=cs.INK2, dx=4, va="center", ha="left", size=8):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va=va, ha=ha, fontsize=size, color=color)


# ---------------------------------------------------------------- a. footfalls
def chart_footfall(R, P, out: Path) -> str:
    tr = P["cases"]["nominal_0.25"]["trace"]
    t = _arr(tr["t"])
    stp = np.asarray(tr["stp"], dtype=bool)
    deb = np.asarray(tr["deb"], dtype=bool)
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.35], left=0.105, right=0.99, top=0.83, bottom=0.21, wspace=0.06)
    ax = fig.add_subplot(gs[0])
    for j, c in enumerate(CORNERS):
        y = 3 - j
        ax.broken_barh(_spans(t, stp[:, j]), (y - 0.3, 0.6), facecolor=cs.FOOT_COLOR[c], edgecolor="none")
        brk = stp[:, j] & ~deb[:, j]  # breaks of 10-50 ms the env counts as touchdowns: shown as gaps in the bar
        ax.broken_barh(_spans(t, brk), (y - 0.3, 0.6), facecolor=cs.SURFACE, edgecolor="none")
    lf = gc.touchdowns(stp[:, 0])
    for k in lf:
        ax.axvline(t[k], color=cs.AXIS, lw=0.8, zorder=0)
    if len(lf) >= 2:
        a, b = t[lf[0]], t[lf[1]]
        ax.annotate("", (a, 3.62), (b, 3.62), arrowprops=dict(arrowstyle="<->", color=cs.INK2, lw=0.9))
        ax.text(0.5 * (a + b), 3.7, f"one stride, {b - a:.2f} s", ha="center", va="bottom", fontsize=8, color=cs.INK2)
    ax.set_yticks([3, 2, 1, 0], [cs.FOOT_LABEL[c] for c in CORNERS])
    ax.set_ylim(-0.6, 4.0)
    ax.set_xlim(t[0], t[-1])
    ax.grid(False)
    ax.set_xlabel("time, s (nominal model, observation noise off, seed 100000)")
    ax.set_title("Stance per foot (bars); vertical lines: left-fore touchdowns")
    ax.tick_params(axis="y", length=0)
    ax.text(0.0, -0.12, "Bars: stance at step level (contact breaks under 50 ms merged). White slivers inside a bar: "
            "10-50 ms breaks that the env's\n10 ms debounce counts as touchdowns (the sole lifts under 1.5 mm).",
            transform=ax.transAxes, fontsize=7.8, color=cs.INK2, va="top")

    tx = fig.add_subplot(gs[1])
    tx.axis("off")
    S = R["cases"]["target_0.25"]["summary"]["footfall"]
    n_ep = R["cases"]["target_0.25"]["summary"]["episodes"]
    g = S["gait"]
    dtm = S.get("diagonal_timing") or {}
    mono = dict(family="DejaVu Sans Mono", fontsize=8.3, color=cs.INK2, va="top")
    big = dict(fontsize=9.3, color=cs.INK, va="top")
    lines = [(f"0.25 m/s, randomised, {n_ep} episodes", dict(fontsize=9.5, fontweight="bold", color=cs.INK, va="top")),
             (f"Gait class: {g['class']} ({g['detail']})", big)]
    if dtm:
        lines.append((f"Diagonal pairs land {dtm['pair_interval_LFRH_to_RFLH']:.2f} of a stride apart (0.5 = even);",
                      big))
        lines.append((f"limb phase left {dtm['limb_phase_left']:.2f}, right {dtm['limb_phase_right']:.2f} (trot 0.5)", big))
    lines += [(f"Stride {S['stride_frequency_hz_all_feet']:.2f} Hz, {1 / S['stride_frequency_hz_all_feet']:.2f} s, "
               f"{100 * S['stride_length_m']['mean']:.1f} cm", big), ("", mono)]
    hdr = f"{'':17s}{'LF':>6s}{'RF':>6s}{'LH':>6s}{'RH':>6s}"
    lines.append((hdr, mono))

    def row(name, vals, fmt="{:6.2f}"):
        return (f"{name:17s}" + "".join(fmt.format(v) if v is not None else f"{'-':>6s}" for v in vals), mono)

    lines.append(row("duty factor", [S["duty_factor"][c] for c in CORNERS]))
    lines.append(row("phase after LF", [0.0] + [S["phase_offset_after_LF"][c]["mean_cycles"] for c in CORNERS[1:]]))
    lines.append(row("phase lock R", [None] + [S["phase_offset_after_LF"][c]["resultant_length"] for c in CORNERS[1:]]))
    lines.append(row("steps per s", [S["steps_per_s"][c] for c in CORNERS]))
    lines.append(row("env touchdowns/s", [S["touchdowns_per_s"][c] for c in CORNERS]))
    lines.append(row("stance, ms", [S["stance_ms"][c]["median"] for c in CORNERS], "{:6.0f}"))
    lines.append(("", mono))
    lines.append((f"{'symmetry index, %':17s}{'fore':>6s}{'hind':>6s}{'sides':>7s}", mono))
    for name, v in S["symmetry_index_pct"].items():
        vals = [v["fore (LF vs RF)"], v["hind (LH vs RH)"], v["sides (LF+LH vs RF+RH)"]]
        lines.append((f"{name.replace('_', ' '):17s}" + "".join(f"{x:6.0f}" if x is not None else "     -" for x in vals[:2])
                      + (f"{vals[2]:7.0f}" if vals[2] is not None else "      -"), mono))
    lines.append(("", mono))
    fd = S["feet_down_fraction"]
    lines.append((f"feet down: 2 diagonal {100 * fd['2 diagonal']:.0f} %, 3 feet {100 * fd['3']:.0f} %,", mono))
    lines.append((f"  1 foot {100 * fd['1']:.0f} %, 4 feet {100 * fd['4']:.0f} %, other pairs "
                  f"{100 * (fd['2 same side'] + fd['2 front'] + fd['2 hind']):.0f} %, none {100 * fd['0']:.1f} %", mono))
    y = 1.02
    for text, kw in lines:
        tx.text(0.0, y, text, transform=tx.transAxes, **kw)
        y -= 0.048
    tx.text(0.0, y - 0.01, "Phase: when that foot lands, as a fraction of the LF stride after an\nLF touchdown (trot: RF 0.5, "
            "LH 0.5, RH 0). Gait class: nearest ideal\npattern within 0.125 cycles on every foot, else 'other'.\n"
            "Symmetry index: 100 |L - R| / mean (0 is symmetric).", transform=tx.transAxes, fontsize=7.6,
            color=cs.MUTED, va="top", linespacing=1.35)
    cs.frame(fig, "Footfalls: when each foot is on the ground",
             f"{_run_label(R)}: the gait diagram (left) is the clean nominal-model episode (observation noise off); the "
             "numbers (right) are the "
             "randomised 0.25 m/s case", _footer(R, "Contact = touch load > 2 N, debounced"))
    cs.save(fig, out / "footfall.png")
    return "footfall.png"


# ---------------------------------------------------------------- b. weight distribution
def chart_weight(R, P, out: Path) -> str:
    prof = np.concatenate([_arr(p).reshape(-1, 101, 4) for p in P["cases"]["target_0.25"]["stride_load_profiles"]])
    S = R["cases"]["target_0.25"]["summary"]["weight"]
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.3, 1], left=0.06, right=0.985, top=0.82, bottom=0.1, hspace=0.55,
                          wspace=0.12)
    ph = np.linspace(0, 100, 101)
    ymax = float(np.percentile(prof[:, :, :], 90, axis=0).max()) * 1.15
    for j, c in enumerate(CORNERS):
        ax = fig.add_subplot(gs[0, j])
        lo, hi = np.percentile(prof[:, :, j], 10, axis=0), np.percentile(prof[:, :, j], 90, axis=0)
        ax.fill_between(ph, lo, hi, color=cs.FOOT_COLOR[c], alpha=0.14, lw=0)
        ax.plot(ph, prof[:, :, j].mean(0), color=cs.FOOT_COLOR[c])
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(cs.FOOT_LABEL[c].replace("  ", " "))
        ax.set_xlabel("% of the left-fore stride")
        if j == 0:
            ax.set_ylabel("vertical load / body weight")
        else:
            ax.tick_params(labelleft=False)
        ax.text(0.97, 0.96, f"peak {S['peak_load_bw_worst'][c]:.2f}\n99th pct {S['p99_load_bw_mean'][c]:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=cs.INK2)
    ax = fig.add_subplot(gs[1, :2])
    shares = [100 * S["share"][c] for c in CORNERS]
    ys = np.arange(4)[::-1]
    ax.barh(ys, shares, height=0.55, color=[cs.FOOT_COLOR[c] for c in CORNERS])
    for yy, v in zip(ys, shares):
        ax.text(v + 0.6, yy, f"{v:.0f} %", va="center", fontsize=8.5, color=cs.INK)
    ax.set_yticks(ys, [cs.FOOT_LABEL[c].split()[0] for c in CORNERS])
    ax.axvline(25, color=cs.INK2, lw=0.9, ls="--")
    ax.text(25.4, 3.55, "equal share 25 %", fontsize=7.8, color=cs.INK2, va="bottom")
    ax.set_xlim(0, max(shares) * 1.25)
    ax.set_xlabel("share of the time-averaged load, %")
    ax.set_title("Load share per foot")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    tx = fig.add_subplot(gs[1, 2:])
    tx.axis("off")
    txt = (f"Front {100 * S['front_share']:.0f} %  /  rear {100 * (1 - S['front_share']):.0f} %\n"
           f"Left {100 * S['left_share']:.0f} %  /  right {100 * (1 - S['left_share']):.0f} %\n"
           f"Mean total load {S['mean_total_load_bw']:.3f} x body weight (a check: 1.000)\n"
           f"{S['strides']} strides pooled from 5 episodes; line = mean, band = 10th-90th percentile.\n"
           "Peak = largest 1 ms value in any episode; 99th pct = mean over episodes.\n"
           f"Floor truth (sole-floor normal forces) and the touch sensors agree in "
           f"{100 * S['touch_vs_floor_state_agreement_min']:.1f} % of samples.")
    tx.text(0.02, 0.95, txt, transform=tx.transAxes, va="top", fontsize=9, color=cs.INK2, linespacing=1.5)
    cs.frame(fig, "Weight distribution: each foot's vertical load over a stride",
             f"{_run_label(R)}, 0.25 m/s command, randomisation on, 5 episodes. Strides run from one left-fore "
             "touchdown to the next.", _footer(R, "Loads from the sole-floor contact forces"))
    cs.save(fig, out / "weight_distribution.png")
    return "weight_distribution.png"


# ---------------------------------------------------------------- c. balance
def chart_balance(R, P, out: Path) -> str:
    tr = P["cases"]["nominal_0.25"]["trace"]
    t = _arr(tr["t"])
    h = _arr(tr["heading"])
    lat = np.array([-h[1], h[0]])
    cop, com = _arr(tr["cop"]), _arr(tr["com"])
    cxy = _arr(tr["corners_xy"])
    stp = np.asarray(tr["stp"], dtype=bool)
    down = np.asarray(tr["down_ft"], dtype=bool)
    t0, t1 = 6.0, 7.3
    sel = (t >= t0) & (t < t1)
    i_sel = np.flatnonzero(sel)
    p0 = com[i_sel[0]]

    def frm(p):
        d = np.asarray(p)[..., :2] - p0
        return np.stack([d @ h, d @ lat], -1)

    fig = plt.figure(figsize=(12.5, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.3], height_ratios=[3.2, 1], left=0.06, right=0.985, top=0.78,
                          bottom=0.1, wspace=0.17, hspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    for j, c in enumerate(CORNERS):
        for v, s0, n in gc.runs(stp[:, j]):
            mid = s0 + n // 2
            if not v or not sel[mid]:
                continue
            poly = frm(cxy[mid, j])
            ax.add_patch(Polygon(poly, closed=True, fc=cs.WASH, ec=cs.AXIS, lw=0.8, zorder=1))
            cx, cy = poly.mean(0)
            ax.text(cx, cy, gc.SHORT[c], ha="center", va="center", fontsize=7.5, color=cs.MUTED, zorder=2)
    # one support polygon: the middle sample of the window with exactly two (diagonal) feet loaded
    two = np.flatnonzero(sel & (down.sum(1) == 2))
    if two.size:
        k = two[two.size // 2]
        hull = gc.convex_hull(frm(cxy[k][down[k]].reshape(-1, 2)))
        ax.add_patch(Polygon(hull, closed=True, fc="none", ec=cs.INK2, lw=1.0, ls="--", zorder=3))
        feet_k = "+".join(gc.SHORT[c] for c, d_ in zip(CORNERS, down[k]) if d_)
        ax.annotate(f"dashed: support polygon at t = {t[k]:.2f} s ({feet_k} in stance)", hull[np.argmin(hull[:, 1])],
                    xytext=(0, -14), textcoords="offset points", fontsize=7.8, color=cs.INK2, ha="left")
    pc, pm = frm(cop[sel]), frm(com[sel])
    ax.scatter(pc[::4, 0], pc[::4, 1], s=3, color=COP_COLOR, alpha=0.45, lw=0, zorder=4)
    ax.plot(pm[:, 0], pm[:, 1], color=COM_COLOR, lw=cs.LW, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel(f"forward, m (from the CoM at t = {t0:.1f} s)")
    ax.set_ylabel("left, m")
    ax.set_title(f"Top view, t = {t0:.1f}-{t1:.1f} s; footprints at mid-stance")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, color=COP_COLOR, label="centre of pressure (every 4 ms)"),
                       Line2D([], [], color=COM_COLOR, lw=cs.LW, label="centre of mass, ground projection"),
                       Patch(fc=cs.WASH, ec=cs.AXIS, label="sole at mid-stance")], loc="upper left",
              bbox_to_anchor=(0, -0.14), ncol=2, fontsize=8)
    ax2 = fig.add_subplot(gs[0, 1])
    m2 = (t >= t0) & (t < t1)
    mc, mm = 1000 * _arr(tr["cop_margin"])[m2], 1000 * _arr(tr["com_margin"])[m2]
    tt = t[m2]
    lo = min(np.nanmin(mm), np.nanmin(mc), -10) * 1.1
    ax2.axhspan(lo, 0, color=cs.WASH, zorder=0, lw=0)
    ax2.text(tt[0] + 0.01, lo * 0.93, "outside the support polygon", fontsize=8, color=cs.INK2, va="bottom")
    ax2.axhline(0, color=cs.INK2, lw=0.9)
    ax2.plot(tt, mc, color=COP_COLOR, lw=1.0)
    ax2.plot(tt, mm, color=COM_COLOR, lw=cs.LW)
    ax2.set_ylim(lo, max(np.nanmax(mm), np.nanmax(mc)) * 1.12)
    ax2.set_xlim(tt[0], tt[-1])
    ax2.tick_params(labelbottom=False)
    ax2.set_ylabel("margin, mm")
    ax2.set_title("Margin inside the support polygon over time")
    ax2.legend(handles=[Line2D([], [], color=COP_COLOR, lw=1.2, label="CoP"),
                        Line2D([], [], color=COM_COLOR, lw=cs.LW, label="CoM")], loc="upper right", ncol=2)
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    raw = np.asarray(tr["down_raw"], dtype=bool)
    ax3.step(tt, raw[m2].sum(1), where="post", color=cs.AXIS, lw=0.8, label="raw 2 N contact")
    ax3.step(tt, down[m2].sum(1), where="post", color=cs.INK2, lw=1.2, label="in stance (step contact)")
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.set_ylim(-0.3, 4.3)
    ax3.set_ylabel("feet")
    ax3.legend(loc="lower right", ncol=2, fontsize=7.5)
    ax3.set_xlabel("time, s (nominal model, observation noise off, seed 100000)")
    cs.frame(fig, "Balance: centre of pressure and centre of mass against the feet on the ground",
             f"{_run_label(R)}. Support polygon: hull of the "
             f"{gc.SOLE_POINTS[R['cases']['nominal_0.25']['summary'].get('sole_kind', 'box')]} of the feet in stance (step contact, as the "
             "footfalls).\nMargin: distance inside the nearest edge (negative outside). " + _balance_numbers(R),
             _footer(R, "Panels: nominal model, noise off, 0.25 m/s; figures in the subtitle: randomised, 5 episodes"))
    cs.save(fig, out / "balance.png")
    return "balance.png"


def _balance_numbers(R) -> str:
    S = R["cases"]["target_0.25"]["summary"]["balance"]
    N = R["cases"]["nominal_0.25"]["summary"]["balance"]
    Rw = R["cases"]["target_0.25"]["summary"]["balance_raw_contact"]
    return (f"0.25 m/s randomised: CoM outside {100 * S['com_margin_negative_fraction']:.1f} % of the time (nominal "
            f"{100 * N['com_margin_negative_fraction']:.1f} %), median CoM margin {S['com_margin_median_mm']:.0f} mm;\nCoP "
            f"within 5 mm of an edge {100 * S['cop_near_edge_fraction']:.0f} %. Footnote: on raw 2 N contact, which counts "
            f"1-2 ms solver rebounds as lifts, the CoM reads outside {100 * Rw['com_margin_negative_fraction']:.0f} %.")


# ---------------------------------------------------------------- d. drives
def chart_drive_envelopes(R, P, out: Path) -> str:
    H = P["cases"]["target_0.25"]["drive_hist"]
    D = R["drives"]
    fig = plt.figure(figsize=(12.5, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.03], left=0.06, right=0.94, top=0.86, bottom=0.08, hspace=0.3,
                          wspace=0.16)
    vmax = max(np.asarray(H[g]["counts"]).max() for g in H)
    norm = LogNorm(vmin=1, vmax=vmax)
    mesh = None
    n_joint = {"roll": 4, "pitch": 4, "front elbow": 2, "hind knee": 2}
    for k, g in enumerate(H):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        xe, ye = _arr(H[g]["rpm_edges"]), _arr(H[g]["nm_edges"])
        C = np.ma.masked_equal(np.asarray(H[g]["counts"], dtype=float).T, 0)
        mesh = ax.pcolormesh(xe, ye, C, cmap=cs.SEQ, norm=norm, rasterized=True)
        d = D[g]
        refs = [("model limit", d["model_limit_nm"], "-", cs.INK, 1.0),
                ("maker peak", d["maker_peak_nm"], "--", cs.INK2, 0.9),
                ("maker rated", d["maker_rated_nm"], ":", cs.INK2, 1.1),
                ("continuous", d["continuous_nm"], "-.", cs.MUTED, 0.9)]
        for name, v, ls, col, lw in refs:
            for sgn in (1, -1):
                ax.axhline(sgn * v, color=col, ls=ls, lw=lw)
            ax.text(xe[0] + 0.01 * (xe[-1] - xe[0]), v, f"{name} {v:.3g}", fontsize=7.3, color=cs.INK2, va="bottom",
                    ha="left")
        sa = d["speed_allowance_rpm"]
        for sgn in (1, -1):
            ax.axvline(sgn * sa, color=cs.INK2, ls="--", lw=0.9)
        ax.text(sa, ye[0] + 0.03 * (ye[-1] - ye[0]), f"speed allowance {sa:.0f} rpm ", fontsize=7.3, color=cs.INK2,
                ha="right", va="bottom", rotation=90)
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
        ax.grid(False)
        ax.set_title(f"{g.capitalize()} ({d['drive']}, {n_joint[g]} joints)")
        ax.set_xlabel("joint speed, rpm")
        ax.set_ylabel("joint torque, N m")
        out_n = H[g]["outside"]
        if out_n:
            ax.text(0.01, 0.01, f"{out_n} samples beyond the plotted range", transform=ax.transAxes, fontsize=7,
                    color=cs.MUTED)
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("1 ms samples per cell (log scale)", color=cs.INK2)
    cb.outline.set_visible(False)
    cs.frame(fig, "Drive envelopes: torque against speed at every 1 ms step",
             f"{_run_label(R)}, 0.25 m/s, randomised, 5 episodes x 18 s, all joints of each group pooled. Joint side: the "
             "flexion drives work through a 1:1 linkage at 0.90 efficiency,\nso their maker figures are x 0.90. Model limit "
             "= 0.70 x maker peak; continuous = the maker's thermal-equilibrium torque.",
             _footer(R, "Maker figures from sim/data/drives.json"))
    cs.save(fig, out / "drive_envelopes.png")
    return "drive_envelopes.png"


def chart_drive_loads(R, P, out: Path) -> str:
    S = R["cases"]["target_0.25"]["summary"]["drives"]["per_joint"]
    names = list(S)
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(1, 2, left=0.13, right=0.985, top=0.8, bottom=0.1, wspace=0.08)
    ys = np.arange(len(names))[::-1]
    for k, (key, title, xlabel, scale, ref) in enumerate((
            ("rms_over_continuous", "RMS torque / continuous rating", "ratio (1.0 = at the rating)", 1.0, 1.0),
            ("at_nominal_limit", "Time at the model's torque limit (>= 99 %)", "% of 1 ms steps", 100.0, None))):
        ax = fig.add_subplot(gs[k])
        mean = np.array([S[n][key]["mean"] for n in names]) * scale
        worst = np.array([S[n][key]["worst"] for n in names]) * scale
        ax.barh(ys, mean, height=0.55, color=cs.SERIES[0])
        ax.scatter(worst, ys, s=36, color=cs.INK2, edgecolor=cs.SURFACE, linewidth=1.5, zorder=3)
        top = max(worst.max(), 1.25 if ref else 1.0)
        for yy, m_, w_ in zip(ys, mean, worst):
            if scale == 1 or w_ > 0.05:
                ax.text(max(m_, w_) + top * 0.02, yy, f"{m_:.2f}" if scale == 1 else f"{m_:.1f} %", va="center",
                        fontsize=8, color=cs.INK)
        if ref:
            ax.axvline(ref, color=cs.INK, lw=0.9, ls="--")
            ax.text(ref, len(names) - 0.35, " continuous rating", fontsize=8, color=cs.INK2, va="bottom")
        ax.set_xlim(0, top * 1.12)
        ax.set_ylim(-0.7, len(names) - 0.1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        if k == 0:
            ax.set_yticks(ys, names)
        else:
            ax.set_yticks(ys, [""] * len(names))
    fig.legend(handles=[Patch(fc=cs.SERIES[0], label="mean of 5 episodes (value printed)"),
                        Line2D([], [], ls="", marker="o", ms=6, mfc=cs.INK2, mec=cs.SURFACE, label="worst episode")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.985, 0.905), fontsize=8)
    cs.frame(fig, "Drive loads per joint: heat and time at the torque limit",
             f"{_run_label(R)}, 0.25 m/s, randomised, 5 episodes. Continuous ratings at the joint: roll and pitch 7.5,\n"
             "front elbow 10.8, hind knee 6.75 N m. Model limits: 21 / 21 / 37.8 / 18.9 N m.",
             _footer(R, "RMS over t >= 2 s stands in for steady walking; not a thermal simulation"))
    cs.save(fig, out / "drive_loads.png")
    return "drive_loads.png"


# ---------------------------------------------------------------- e. power
def chart_power(R, P, out: Path) -> str:
    tr = P["cases"]["nominal_0.25"]["trace"]
    t = _arr(tr["t"])
    n = 20
    series = [("electrical, per drive (no reuse)", gc.moving_average(_arr(tr["p_elec"]), n), cs.SERIES[0]),
              ("electrical, shared bus", gc.moving_average(_arr(tr["p_bus"]), n), cs.SERIES[1]),
              ("copper loss", gc.moving_average(_arr(tr["p_cu"]), n), cs.SERIES[2]),
              ("positive mechanical work", gc.moving_average(_arr(tr["p_pos"]), n), cs.SERIES[3])]
    fig = plt.figure(figsize=(12.5, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], left=0.17, right=0.975, top=0.78, bottom=0.13, hspace=0.45)
    ax = fig.add_subplot(gs[0])
    for name, y, col in series:
        ax.plot(t, y, color=col, lw=1.2)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("time, s (nominal model, observation noise off, seed 100000; 20 ms moving average)")
    ax.set_ylabel("power, W (12 leg drives)")
    ax.set_title("Leg power over three strides")
    ax.legend(handles=[Line2D([], [], color=c, lw=cs.LW, label=f"{nm} (mean {np.mean(y):.0f} W)")
                       for nm, y, c in series], loc="upper left", ncol=2)
    ax2 = fig.add_subplot(gs[1])
    keys = [k for k in ("target_0.25", "cmd_0.15", "nominal_0.25", "ankleroll_0.25", "otherfoot_0.25")
            if k in R["cases"]]
    hotel = R["settings"]["hotel_w"]
    ys = np.arange(len(keys))[::-1]
    cols = [cs.SERIES[3], cs.SERIES[2], cs.MUTED]
    for yy, k in zip(ys, keys):
        p = R["cases"][k]["summary"]["power"]
        parts = [p["electrical_per_drive_w"] - p["copper_w"], p["copper_w"], hotel]
        x0 = 0.0
        for v, col in zip(parts, cols):
            ax2.barh(yy, v - 1.0, left=x0 + 0.5, height=0.55, color=col)  # 1 W surface gap between segments
            if v > 8 and col != cs.MUTED:  # the hotel segment's value is in the legend
                ax2.text(x0 + v / 2, yy, f"{v:.0f}", ha="center", va="center", fontsize=8,
                         color=cs.INK if col == cs.SERIES[3] else "#ffffff")
            x0 += v
        bus = p["electrical_shared_bus_w"] + hotel
        ax2.plot([bus, bus], [yy - 0.38, yy + 0.38], color=cs.INK, lw=1.6)
        rt_d = p["run_time_min_per_drive"][f"hotel_{int(hotel)}w"]
        rt_b = p["run_time_min_shared_bus"][f"hotel_{int(hotel)}w"]
        ax2.text(x0 + 3, yy, f"per drive {x0:.0f} W -> {rt_d:.0f} min;  shared bus {bus:.0f} W -> {rt_b:.0f} min",
                 va="center", fontsize=8.3, color=cs.INK)
    ax2.set_yticks(ys, [R["cases"][k]["label"].replace(", randomised", "").replace(", observation noise off", ", noise off")
                        for k in keys])
    ax2.set_xlim(0, max(R["cases"][k]["summary"]["power"]["electrical_per_drive_w"] for k in keys) * 1.8 + hotel)
    ax2.set_xlabel("mean power, W")
    ax2.set_title("Energy budget while walking (5 episodes each) and run time on 146.5 Wh x 80 %")
    ax2.grid(axis="y", visible=False)
    ax2.tick_params(axis="y", length=0)
    ax2.legend(handles=[Patch(fc=cs.SERIES[3], label="positive work at the drives"),
                        Patch(fc=cs.SERIES[2], label="copper loss 3 R (torque / Kt)^2"),
                        Patch(fc=cs.MUTED, label=f"hotel load, assumed {hotel:.0f} W"),
                        Line2D([], [], color=cs.INK, lw=1.6, label="shared-bus total (tick)")],
               loc="upper left", ncol=4, fontsize=8, bbox_to_anchor=(-0.01, -0.3))
    T = R["cases"]["target_0.25"]["summary"]["power"]
    cs.frame(fig, "Power and energy: what walking costs the battery",
             f"{_run_label(R)}. At 0.25 m/s: joint mechanical {T['mech_abs_w']:.0f} W (sum |torque x speed|), cost of "
             f"transport {T['cost_of_transport_mech']:.2f} (mechanical).\nTorque (copper loss) dominates the electrical "
             "figures. They are estimates: unmodelled gearbox, iron and driver losses would raise them;\nthe per-drive figure "
             "ignores bus-level reuse of braking energy (which would lower it); the shared-bus figure assumes all of it is "
             "reused at once.",
             _footer(R, "Kt and phase resistance from the T. rex actuator catalog"))
    cs.save(fig, out / "power.png")
    return "power.png"


# ---------------------------------------------------------------- f. body and head
def chart_steadiness(R, P, out: Path) -> str:
    tr = P["cases"]["nominal_0.25"]["trace"]
    t = _arr(tr["t"])
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.0, 1.15], left=0.07, right=0.99, top=0.85, bottom=0.08, hspace=0.38,
                          wspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    for name, y, col in (("roll", _arr(tr["roll"]), cs.SERIES[0]), ("pitch", _arr(tr["pitch"]), cs.SERIES[1])):
        ax1.plot(t, y, color=col, lw=1.2, label=name)
    ax1.set_ylabel("chassis angle, deg")
    ax1.set_title("Chassis roll and pitch")
    ax1.legend(loc="upper right", ncol=2)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t, _arr(tr["height_mm"]), color=cs.SERIES[0], lw=1.2)
    ax2.set_ylabel("chassis height, mm")
    ax2.set_title("Chassis height")
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    acc, acc_lo = _arr(tr["head_acc"]), _arr(tr["head_acc_le5hz"])
    ax3.plot(t, acc[:, 2], color=cs.SERIES[2], lw=0.9, label="vertical, to 20 Hz")
    ax3.plot(t, acc[:, 1], color=cs.SERIES[1], lw=0.9, label="lateral, to 20 Hz")
    ax3.plot(t, acc_lo[:, 2], color=cs.SERIES[0], lw=cs.LW + 0.4, label="vertical, to 5 Hz (the visible bob)")
    ax3.axhline(0, color=cs.AXIS, lw=0.8)
    ax3.set_ylabel("head acceleration, m/s^2")
    ax3.set_title("Head acceleration (gravity excluded): to 20 Hz, and the part at or below 5 Hz")
    ax3.legend(loc="upper right", ncol=3, fontsize=7.8)
    ax3.set_xlabel("time, s (nominal model, observation noise off, seed 100000)")
    ax3.set_xlim(t[0], t[-1])
    for a in (ax1, ax2):
        a.tick_params(labelbottom=False)
    tx = fig.add_subplot(gs[:, 1])
    tx.axis("off")
    mono = dict(family="DejaVu Sans Mono", fontsize=8.2, color=cs.INK2, va="top")
    keys = [("target_0.25", "0.25 r"), ("cmd_0.15", "0.15 r"), ("nominal_0.25", "0.25 n")]
    lines = [(f"{'':19s}" + "".join(f"{lab:>8s}" for _, lab in keys), mono)]

    def row(name, f, fmt="{:8.1f}"):
        return (f"{name:19s}" + "".join(fmt.format(f(R["cases"][k]["summary"])) for k, _ in keys), mono)

    lines += [row("roll SD, deg", lambda s: s["body"]["roll_deg"]["sd"], "{:8.2f}"),
              row("pitch SD, deg", lambda s: s["body"]["pitch_deg"]["sd"], "{:8.2f}"),
              row("roll peak, deg", lambda s: s["body"]["roll_deg"]["peak_abs_worst"]),
              row("pitch peak, deg", lambda s: s["body"]["pitch_deg"]["peak_abs_worst"]),
              row("height range, mm", lambda s: s["body"]["height_mm"]["range_p1_p99"]),
              row("head bob, mm", lambda s: s["head"]["bob_mm_p1_p99"]),
              row("head sway, mm", lambda s: s["head"]["sway_mm_p1_p99"]),
              row("head acc RMS, m/s^2", lambda s: s["head"]["acc_rms_mps2"]),
              row("  <= 5 Hz (bob)", lambda s: s["head"]["acc_rms_le5hz_mps2"]),
              row("  5-20 Hz (shake)", lambda s: s["head"]["acc_rms_5_20hz_mps2"]),
              row("  vertical, all", lambda s: s["head"]["acc_rms_by_axis_mps2"]["vertical"]),
              row("head acc peak", lambda s: s["head"]["acc_peak_mps2_worst"]),
              row("  <= 5 Hz", lambda s: s["head"]["acc_peak_le5hz_mps2_worst"])]
    y = 0.99
    tx.text(0.0, y, "Mean of 5 episodes (peaks: worst episode)", transform=tx.transAxes, fontsize=9.5,
            fontweight="bold", color=cs.INK, va="top")
    tx.text(0.0, y - 0.035, "r: randomised; n: nominal model, noise off; m/s command", transform=tx.transAxes, fontsize=7.8,
            color=cs.MUTED, va="top")
    y -= 0.03
    y -= 0.06
    for text, kw in lines:
        tx.text(0.0, y, text, transform=tx.transAxes, **kw)
        y -= 0.041
    T = R["cases"]["target_0.25"]["summary"]
    tx.text(0.0, y - 0.02, f"How it reads: the head bobs about {T['head']['bob_mm_p1_p99']:.0f} mm and sways\nabout "
            f"{T['head']['sway_mm_p1_p99']:.0f} mm with every step (bob at {T['head']['bob_dominant_hz']:.1f} Hz): a brisk "
            f"nod,\nnot a glide. The visible part (<= 5 Hz) is {T['head']['acc_rms_le5hz_mps2'] / 9.81:.2f} g RMS; most\n"
            f"of the {T['head']['acc_rms_mps2'] / 9.81:.2f} g to 20 Hz is a 5-20 Hz shake from the steps.\n"
            "The body itself is steady to about a degree.\n\nRanges: 1st-99th percentile. Bob and sway: after removing\n"
            "a 1 s moving average (the walk's own progress).", transform=tx.transAxes, fontsize=8.1, color=cs.INK2,
            va="top", linespacing=1.4)
    cs.frame(fig, "Body and head steadiness: how the walk looks to a visitor",
             f"{_run_label(R)}. Time series: nominal model, observation noise off, 0.25 m/s. Head: centre of mass of the "
             "head (yaw) body.",
             _footer(R, "Head acceleration = derivative of the head's velocity"))
    cs.save(fig, out / "steadiness.png")
    return "steadiness.png"


# ---------------------------------------------------------------- g. feet
def _strip(ax, data: dict, ylabel: str, title: str, fmt: str):
    rng = np.random.default_rng(0)
    for j, c in enumerate(CORNERS):
        v = _arr(data[c])
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        ax.scatter(j + rng.uniform(-0.18, 0.18, v.size), v, s=9, color=cs.FOOT_COLOR[c], alpha=0.55, lw=0)
        med = float(np.median(v))
        ax.plot([j - 0.3, j + 0.3], [med, med], color=cs.INK, lw=cs.LW)
        ax.text(j + 0.32, med, fmt.format(med), va="center", fontsize=8, color=cs.INK)
    ax.set_xticks(range(4), [gc.SHORT[c] for c in CORNERS])
    ax.set_xlim(-0.6, 3.75)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


def chart_feet(R, P, out: Path) -> str:
    C = P["cases"]["target_0.25"]
    pool = lambda key: {c: np.concatenate([_arr(e[c]) for e in C[key]]) for c in CORNERS}  # noqa: E731
    fig = plt.figure(figsize=(12.5, 8.0))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.985, top=0.85, bottom=0.08, hspace=0.38, wspace=0.17)
    ax = fig.add_subplot(gs[0, 0])
    _strip(ax, pool("clearance_mm"), "highest point of the sole's lowest corner, mm", "Swing clearance (per step)",
           "{:.0f} mm")
    ax.set_ylim(0, None)
    ax = fig.add_subplot(gs[0, 1])
    _strip(ax, pool("touchdown_speed_mps"), "sole speed downward at touchdown, m/s", "Touchdown impact speed", "{:.2f}")
    ax.set_ylim(min(0, ax.get_ylim()[0]), None)
    ax = fig.add_subplot(gs[1, 0])
    edges = _arr(C["slip_hist"][0]["edges"])
    mid = 0.5 * (edges[1:] + edges[:-1])
    for c in CORNERS:
        cnt = sum(_arr(e["counts"][c]) for e in C["slip_hist"])
        cdf = np.cumsum(cnt) / cnt.sum()
        ax.plot(np.r_[0, edges[1:]], np.r_[0, cdf], color=cs.FOOT_COLOR[c], lw=cs.LW)
        p95 = float(np.interp(0.95, cdf, edges[1:]))
        ax.plot([p95], [0.95], "o", ms=5, color=cs.FOOT_COLOR[c], mec=cs.SURFACE, mew=1.5)
    S = R["cases"]["target_0.25"]["summary"]["feet"]["stance_slip"]
    ax.axhline(0.95, color=cs.AXIS, lw=0.8)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("sole horizontal speed while loaded, m/s")
    ax.set_ylabel("fraction of loaded time below")
    ax.set_title("Slip in stance (dots: 95th percentile)")
    ax.legend(handles=[Line2D([], [], color=cs.FOOT_COLOR[c], lw=cs.LW,
                              label=f"{gc.SHORT[c]}: slip ratio {S[c]['slip_ratio_mean']:.2f}") for c in CORNERS],
              loc="lower right", fontsize=8)
    ax = fig.add_subplot(gs[1, 1])
    roll_key = next((k for k in ("target_0.25", "ankleroll_0.25") if k in R["cases"]
                     and "ankle_roll_deg" in R["cases"][k]["summary"]["feet"]), None)
    if roll_key:
        A = R["cases"][roll_key]["summary"]["feet"]
        lim = A["ankle_roll_limit_deg"]
        for j, c in enumerate(CORNERS):
            v = A["ankle_roll_deg"][c]
            ax.plot([v["min_worst"], v["max_worst"]], [3 - j, 3 - j], color=cs.FOOT_COLOR[c], lw=1.0)
            ax.plot([v["p1_mean"], v["p99_mean"]], [3 - j, 3 - j], color=cs.FOOT_COLOR[c], lw=7, solid_capstyle="round")
            ax.text(v["max_worst"] + 0.6, 3 - j, f"{v['p1_mean']:+.1f} to {v['p99_mean']:+.1f} deg", va="center",
                    fontsize=8, color=cs.INK)
        for sgn in (1, -1):
            ax.axvline(sgn * lim, color=cs.INK, ls="--", lw=0.9)
        ax.text(lim, 4.25, f" joint limit +-{lim:.0f} deg", ha="left", fontsize=8, color=cs.INK2)
        ax.set_yticks([3, 2, 1, 0], [gc.SHORT[c] for c in CORNERS])
        ax.set_xlim(-lim * 1.15, lim * 1.6)
        ax.set_ylim(-0.6, 4.6)
        ax.set_xlabel("passive ankle-roll joint angle, deg")
        ax.set_title("Roll-ankle angle range (roll-ankle model, 5 episodes)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.legend(handles=[Line2D([], [], color=cs.INK2, lw=7, label="1st-99th percentile (mean)"),
                           Line2D([], [], color=cs.INK2, lw=1.0, label="min-max (worst episode)")],
                  loc="upper left", fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no roll-ankle case", ha="center", color=cs.MUTED)
    cs.frame(fig, "Feet: clearance, impact, slip and the passive roll ankle",
             f"{_run_label(R)}, 0.25 m/s, randomised, 5 episodes (steps: contact breaks under 50 ms merged). Lines: "
             "medians.", _footer(R, "Sole kinematics from the box sole geoms; slip as in rl/analysis/gait_pipe1.py"))
    cs.save(fig, out / "feet.png")
    return "feet.png"


# ---------------------------------------------------------------- h. speed tracking
def chart_speed(R, P, out: Path) -> str:
    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], left=0.065, right=0.985, top=0.83, bottom=0.12, wspace=0.18)
    ax = fig.add_subplot(gs[0])
    for k, col in (("target_0.25", cs.SERIES[0]), ("cmd_0.15", cs.SERIES[1])):
        c = P["cases"][k]
        t, v = _arr(c["step_t"]), _arr(c["step_v"])
        cmd = R["cases"][k]["command_mps"]
        ax.plot(t, v, color=col, lw=0.6, alpha=0.35)
        ma = gc.moving_average(v, 21)
        ax.plot(t, ma, color=col, lw=cs.LW)
        ax.axhline(cmd, color=cs.INK2, ls="--", lw=0.9)
        ax.text(0.2, cmd + 0.004, f"command {cmd:.2f} m/s", fontsize=8, color=cs.INK2, va="bottom")
        _end_label(ax, t[-1], ma[-25:].mean(), f"{R['cases'][k]['summary']['speed']['forward_speed_mps']:.3f} m/s",
                   cs.INK)
    ax.axvline(2.0, color=cs.AXIS, lw=0.9)
    ax.text(2.05, 0.005, "measured from 2 s", fontsize=7.8, color=cs.MUTED)
    ax.set_xlim(0, 21.5)
    ax.set_ylim(0, 0.33)
    ax.set_xlabel("time, s (seed 100000, randomised)")
    ax.set_ylabel("forward speed, m/s")
    ax.set_title("Commanded and actual speed over an episode")
    ax.legend(handles=[Line2D([], [], color=cs.SERIES[0], lw=cs.LW, label="0.25 m/s command"),
                       Line2D([], [], color=cs.SERIES[1], lw=cs.LW, label="0.15 m/s command"),
                       Line2D([], [], color=cs.INK2, lw=0.6, alpha=0.6, label="per 20 ms control step (thin), "
                                                                               "0.42 s average (thick)")],
              loc="upper left", fontsize=8, ncol=1)
    ax2 = fig.add_subplot(gs[1])
    sw = R["sweep"]
    cmds = np.array([v["command_mps"] for v in sw.values()])
    ax2.plot([0, 0.3], [0, 0.3], color=cs.INK2, ls="--", lw=0.9)
    ax2.text(0.285, 0.29, "speed = command", fontsize=8, color=cs.INK2, ha="right", va="bottom", rotation=0)
    means = []
    falls = 0
    for v in sw.values():
        xs = np.full(len(v["episodes"]), v["command_mps"])
        ys = [e["forward_speed_mps"] for e in v["episodes"]]
        fl = [e["fell"] for e in v["episodes"]]
        ok = [not f for f in fl]
        ax2.scatter(xs[ok], np.asarray(ys)[ok], s=40, color=cs.SERIES[0], edgecolor=cs.SURFACE, lw=1.5, zorder=3)
        if any(fl):
            falls += sum(fl)
            ax2.scatter(xs[fl], np.asarray(ys)[fl], s=60, marker="X", color=cs.CRITICAL, zorder=4)
        means.append(v["forward_speed_mps_mean"])
    ax2.plot(cmds, means, color=cs.SERIES[0], lw=cs.LW, zorder=2)
    for x_, m_ in zip(cmds, means):
        if x_ in (0.0, 0.25, 0.27):
            ax2.annotate(f"{m_:.3f}", (x_, m_), xytext=(-4, 8), textcoords="offset points", fontsize=8, color=cs.INK,
                         ha="right")
    n_ep = sum(len(v["episodes"]) for v in sw.values())
    ax2.text(0.02, 0.97, (f"falls: {falls} of {n_ep} episodes" + ("" if falls else " (none)")), transform=ax2.transAxes,
             fontsize=8.5, color=cs.INK, va="top")
    hnd = [Line2D([], [], ls="", marker="o", ms=6, mfc=cs.SERIES[0], mec=cs.SURFACE, label="one episode (window mean)"),
           Line2D([], [], color=cs.SERIES[0], lw=cs.LW, label="mean of 3 seeds")]
    if falls:
        hnd.append(Line2D([], [], ls="", marker="X", ms=7, color=cs.CRITICAL, label="fall"))
    ax2.legend(handles=hnd, loc="lower right", fontsize=8)
    ax2.set_xlim(-0.01, 0.3)
    ax2.set_ylim(-0.01, 0.3)
    ax2.set_aspect("equal")
    ax2.set_xlabel("commanded speed, m/s")
    ax2.set_ylabel("actual forward speed, m/s (t >= 2 s)")
    ax2.set_title("Speed sweep, seeds 100000-100002")
    cs.frame(fig, "Speed tracking: does it walk at the speed it is asked for?",
             f"{_run_label(R)}, randomisation on. Forward = along the heading at reset.",
             _footer(R, "Sweep at 0.15 and 0.25 m/s reuses the 5-episode cases' first 3 seeds"))
    cs.save(fig, out / "speed_tracking.png")
    return "speed_tracking.png"


def make_all(R: dict, P: dict, out: Path) -> list[str]:
    cs.setup()
    out = Path(out)
    made = []
    for f in (chart_footfall, chart_weight, chart_balance, chart_drive_envelopes, chart_drive_loads, chart_power,
              chart_steadiness, chart_feet, chart_speed):
        try:
            made.append(f(R, P, out))
        except Exception as exc:  # a chart that cannot be drawn (e.g. a policy that never steps) is named, not fatal:
            plt.close("all")      # the numbers in gait_report.json do not depend on the charts
            made.append(f"{f.__name__}: not drawn ({type(exc).__name__}: {exc})")
            print(f"WARNING: {made[-1]}")
    return made
