"""Charts of a run's training health and periodic checks (robot-agnostic; decision D21). Simulation only.

Usage:
  python hardware/triceratops_build/rl/telemetry/health_dashboard.py --label NAME [--health train_health.jsonl]
      [--checks checks_during_training.jsonl] --out DIR

Writes ``training_health_<NAME>.png`` (from ``health.py``'s file) and ``training_checks_<NAME>.png`` (from
``checks.py``'s file), whichever inputs are given, plus ``training_health_<NAME>.json`` (the plotted series).
Small multiples, one measure per panel on one axis, at most four series per panel in the validated categorical
order of ``rl/analytics/chart_style.py``, each labelled directly. Health lines are smoothed with a centred rolling
mean over ``--window`` rollouts (default 20); the JSON keeps the raw values. Every chart carries the SIMULATION tag.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "analytics"), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import chart_style as cs  # noqa: E402
from telemetry.checks import read_checks  # noqa: E402
from telemetry.health import latest_branch, read_health  # noqa: E402

plt = cs.plt

ALGO_PANELS = (  # (key in "algo" or the top level, title, y label, log y)
    ("approx_kl", "Update size: approximate KL", "KL per update", False),
    ("clip_fraction", "Share of samples clipped", "fraction", False),
    ("action_std", "Exploration: action standard deviation", "std (policy output units; axis from 0)", False),
    ("explained_variance", "Value fit: explained variance", "1 = perfect, 0 = none", False),
    ("value_loss", "Value loss", "loss (log scale)", True),
    ("grad_norm", "Gradient norm after the update (PPO clips it)", "L2 norm", False),
)


def rolling(y, w: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0 or w <= 1:
        return y
    out = np.full_like(y, np.nan)
    h = w // 2
    for i in range(y.size):
        seg = y[max(0, i - h): i + h + 1]
        seg = seg[np.isfinite(seg)]
        out[i] = seg.mean() if seg.size else np.nan
    return out


def family(reason: str) -> str:
    """An ending reason's family: per-geom floor contacts ("floor_contact:<geom>") count as one."""
    return str(reason).split(":", 1)[0]


def health_series(rows: list[dict]) -> dict:
    rows = latest_branch(rows, "timesteps")
    steps = np.array([r["timesteps"] for r in rows], dtype=np.float64)
    s = {"steps": steps.tolist()}
    for key, *_ in ALGO_PANELS:
        vals = [(r.get(key) if key in ("action_std", "grad_norm") else (r.get("algo") or {}).get(key)) for r in rows]
        s[key] = [None if v is None else float(v) for v in vals]
    terms = Counter()
    for r in rows:
        for k, v in (r.get("rewards") or {}).items():
            if not k.startswith("reward_total") and v is not None and math.isfinite(v):
                terms[k] += abs(v)
    s["reward_terms_by_size"] = [k for k, _ in terms.most_common()]
    s["rewards"] = {k: [(r.get("rewards") or {}).get(k) for r in rows] for k in s["reward_terms_by_size"]}
    fam = [Counter() for _ in rows]
    for c, r in zip(fam, rows):
        for k, v in (r.get("endings") or {}).items():
            c[family(k)] += int(v)
    endings = sum(fam, Counter())
    s["endings_total"] = dict(endings)
    s["episodes"] = [int(r.get("episodes") or 0) for r in rows]
    s["endings"] = {k: [c.get(k, 0) for c in fam] for k in endings}
    return s


def key_is_ceiling(title: str) -> bool:
    return "clips it" in title


def _panel(ax, x, y, title, ylabel, w, log=False, color=None):
    y = np.array([np.nan if v is None else v for v in y], dtype=np.float64)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if not np.isfinite(y).any():
        ax.text(0.5, 0.5, "not recorded", ha="center", va="center", transform=ax.transAxes, color=cs.MUTED)
        return
    ys = rolling(y, w)
    ax.plot(x / 1e6, ys, color=color or cs.SERIES[0], lw=cs.LW)
    ax.ticklabel_format(axis="y", useOffset=False, style="plain") if not log else None
    if log and np.nanmin(ys) > 0:
        ax.set_yscale("log")
    if key_is_ceiling(title) or "axis from 0" in ylabel:  # a ceiling, or a slow drift, read against zero
        ax.set_ylim(0, max(1e-9, float(np.nanmax(ys))) * 1.15)
    last = ys[np.isfinite(ys)][-1]
    ax.annotate(f"{last:.3g}", (x[np.isfinite(ys)][-1] / 1e6, last), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=cs.INK2, va="center")


def _labelled_lines(ax, x, series: dict, w: int):
    """At most four named series in the fixed categorical order, in a legend, and labelled at their ends where the
    end labels do not collide (the legend always carries identity)."""
    ends = []
    for i, (name, y) in enumerate(series.items()):
        y = rolling(np.array([np.nan if v is None else v for v in y], dtype=np.float64), w)
        ok = np.isfinite(y)
        if not ok.any():
            continue
        ax.plot(x[ok] / 1e6, y[ok], color=cs.SERIES[i], lw=cs.LW, label=name)
        ends.append((name, x[ok][-1] / 1e6, y[ok][-1]))
    if not ends:
        return
    ax.legend(loc="best", fontsize=7.5)
    lo, hi = ax.get_ylim()
    placed = []
    for name, xe, ye in ends:
        frac = (ye - lo) / (hi - lo) if hi > lo else 0.5
        if all(abs(frac - f) > 0.05 for f in placed):
            ax.annotate(name, (xe, ye), xytext=(4, 0), textcoords="offset points", fontsize=8, color=cs.INK2,
                        va="center")
            placed.append(frac)


def health_figure(rows: list[dict], label: str, out: Path, w: int = 20) -> str:
    cs.setup()
    s = health_series(rows)
    x = np.array(s["steps"])
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    fig.subplots_adjust(left=0.055, right=0.965, top=0.85, bottom=0.08, wspace=0.36, hspace=0.42)
    for ax, (key, title, ylabel, log) in zip(axes.flat[:6], ALGO_PANELS):
        _panel(ax, x, s[key], title, ylabel, w, log)
    ax = axes.flat[6]
    ax.set_title("Reward terms, the four largest (per step)")
    ax.set_ylabel("mean per step")
    _labelled_lines(ax, x, {k: s["rewards"][k] for k in s["reward_terms_by_size"][:4]}, w)
    ax.axhline(0, color=cs.AXIS, lw=0.8)
    ax = axes.flat[7]
    ax.set_title("Why episodes ended (share per rollout)")
    ax.set_ylabel("share of the episodes that ended")
    eps = np.array(s["episodes"], dtype=np.float64)
    top = [k for k, _ in Counter(s["endings_total"]).most_common(3)]
    shares = {}
    for k in top:
        with np.errstate(invalid="ignore", divide="ignore"):
            shares[k] = list(np.where(eps > 0, np.array(s["endings"][k]) / eps, np.nan))
    if len(s["endings_total"]) > 3:
        other = np.sum([np.array(v) for k, v in s["endings"].items() if k not in top], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            shares["other"] = list(np.where(eps > 0, other / eps, np.nan))
    _labelled_lines(ax, x, shares, w)
    ax.set_ylim(-0.02, 1.02)
    if not s["endings_total"]:
        ax.text(0.5, 0.5, "no episode ended", ha="center", va="center", transform=ax.transAxes, color=cs.MUTED)
    for ax in axes.flat:
        ax.set_xlabel("training steps, millions")
    cs.frame(fig, f"Training health: {label}",
             f"One point per rollout and its update, centred rolling mean over {w} rollouts; a resumed session "
             "replaces the steps its predecessor ran past its checkpoint. Learning diagnostics as Stable-Baselines3 "
             "records them;\nreward terms and endings from the training envs' step info (floor contacts grouped). "
             "The gradient norm is after PPO's clipping, so it sits at the clip value while clipping binds.",
             "SIMULATION. Source: rl/telemetry/health_dashboard.py from train_health.jsonl (rl/telemetry/health.py).")
    name = f"training_health_{label}.png"
    cs.save(fig, out / name)
    return name


def checks_series(rows: list[dict]) -> dict:
    rows = latest_branch(rows, "steps")
    steps = [r["steps"] for r in rows]
    names = list(dict.fromkeys(n for r in rows for n in r["points"]))
    s = {"steps": steps, "points": {}}
    for n in names:
        falls, speed = [], []
        for r in rows:
            eps = [e for e in (r["points"].get(n) or []) if "error" not in e]
            falls.append(sum(bool(e.get("fell")) for e in eps) / len(eps) if eps else None)
            ok = [e["forward_speed_mps"] for e in eps if not e.get("fell") and e.get("forward_speed_mps") is not None]
            speed.append(float(np.mean(ok)) if ok else None)
        s["points"][n] = {"fall_share": falls, "speed_mps": speed}
    return s


def checks_figure(rows: list[dict], label: str, out: Path) -> str:
    cs.setup()
    s = checks_series(rows)
    x = np.array(s["steps"], dtype=np.float64)
    names = list(s["points"])[:4]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.subplots_adjust(left=0.06, right=0.9, top=0.8, bottom=0.12, wspace=0.3)
    _labelled_lines(axes[0], x, {n: s["points"][n]["fall_share"] for n in names}, 1)
    axes[0].set_title("Falls at each check point")
    axes[0].set_ylabel("share of the check's episodes that fell")
    axes[0].set_ylim(-0.02, 1.02)
    _labelled_lines(axes[1], x, {n: s["points"][n]["speed_mps"] for n in names}, 1)
    axes[1].set_title("Speed at each check point (episodes that did not fall)")
    axes[1].set_ylabel("forward speed, m/s")
    for ax in axes:
        ax.set_xlabel("training steps, millions")
    extra = f" (first four of {len(s['points'])} shown; the JSON has all)" if len(s["points"]) > 4 else ""
    cs.frame(fig, f"Checks during training: {label}",
             f"A few deterministic episodes per point at each check{extra}. Progress readings, not a verdict.",
             "SIMULATION. Source: rl/telemetry/health_dashboard.py from checks_during_training.jsonl "
             "(rl/telemetry/checks.py).")
    name = f"training_checks_{label}.png"
    cs.save(fig, out / name)
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--health", default=None)
    ap.add_argument("--checks", default=None)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files, data = [], {}
    if a.health:
        rows = read_health(a.health)
        if rows:
            files.append(health_figure(rows, a.label, out, a.window))
            data["health"] = health_series(rows)
    if a.checks:
        rows = read_checks(a.checks)
        if rows:
            files.append(checks_figure(rows, a.label, out))
            data["checks"] = checks_series(rows)
    (out / f"training_health_{a.label}.json").write_text(json.dumps({"charts": files, **data}, indent=1) + "\n")
    print("wrote", ", ".join(str(out / f) for f in files) or "no charts (no rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
