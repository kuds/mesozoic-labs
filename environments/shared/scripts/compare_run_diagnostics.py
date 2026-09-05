"""Compare policy/optimizer diagnostics and eval curves across training runs.

``diagnostics.npz`` already carries SB3's per-update ``train/*`` metrics under
``algo_*`` keys (policy std, entropy loss, approx KL, clip fraction, explained
variance), but nothing plots or prints them — every investigation so far has
hand-parsed the npz. This script turns that into one command, and reports two
runs side by side so a single-variable config change can be attributed.

The first/last-20% window methodology matches
``docs/investigations/TREX_STAGE1_LEG_JITTER.md`` so numbers stay comparable
with that investigation's table.

``action_delta`` is reported as RMS action change per joint per step
(``sqrt(action_delta / n_actuators)``) because the raw sum scales with the
actuator count and is not comparable across species. For reference on that
scale: <0.1 is smooth, 0.3-0.5 is severe jitter, ~1.0 is full-scale thrash.

Usage::

    python environments/shared/scripts/compare_run_diagnostics.py \\
        pre=/path/to/run_a/stage1 post=/path/to/run_b/stage1

    # labels are optional; the directory name is used when omitted
    python environments/shared/scripts/compare_run_diagnostics.py run_a/stage1
"""

import json
import sys
from pathlib import Path

import numpy as np

# Signals worth comparing between two runs, in report order.  Each entry is
# (npz key, display label, n_actuators-normalized?).
_ALGO_SIGNALS = (
    ("algo_std", "policy std (algo_std)", False),
    ("algo_entropy_loss", "entropy loss", False),
    ("algo_approx_kl", "approx KL", False),
    ("algo_clip_fraction", "clip fraction", False),
    ("algo_explained_variance", "explained variance", False),
    ("algo_value_loss", "value loss", False),
)

_ENV_SIGNALS = (
    ("action_delta", "RMS action change / joint / step", True),
    ("abs_speed", "body speed (m/s)", False),
    ("drift_distance", "drift from spawn (m)", False),
    ("tilt_angle", "tilt (rad)", False),
    ("pelvis_height", "pelvis height (m)", False),
)

_WINDOW = 0.20


def _finite(arr):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _window_means(values):
    """Return (first 20% mean, last 20% mean) of a 1-D series."""
    vals = _finite(values)
    if vals.size == 0:
        return None, None
    n = max(1, int(round(vals.size * _WINDOW)))
    return float(vals[:n].mean()), float(vals[-n:].mean())


def _n_actuators(stage_dir: Path) -> int | None:
    """Read the action dimension from the stage's saved plant identity."""
    for name in ("stage_config.json", "plant_identity.json"):
        path = stage_dir / name
        if not path.exists():
            continue
        try:
            blob = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        plant = blob.get("plant_identity", blob)
        dim = plant.get("action_dim") or plant.get("nu")
        if dim:
            return int(dim)
    return None


class Run:
    """One stage directory's diagnostics + evaluation history."""

    def __init__(self, label: str, stage_dir: Path):
        self.label = label
        self.stage_dir = stage_dir
        diag_path = stage_dir / "diagnostics.npz"
        if not diag_path.exists():
            raise SystemExit(f"{diag_path} not found")
        self.diag = np.load(str(diag_path))
        self.n_act = _n_actuators(stage_dir)
        eval_path = stage_dir / "evaluations.npz"
        self.evals = np.load(str(eval_path)) if eval_path.exists() else None

    def signal(self, key: str, normalize: bool):
        """Window means for *key*, RMS-per-joint normalized when asked."""
        if key not in self.diag.files:
            return None, None
        first, last = _window_means(self.diag[key])
        if normalize and first is not None and self.n_act:
            first = float(np.sqrt(first / self.n_act))
            last = float(np.sqrt(last / self.n_act))
        return first, last

    def termination_shares(self) -> dict:
        """Mean fraction per termination reason over the last 20% of training."""
        out = {}
        for key in self.diag.files:
            if not key.startswith("term_") or key == "term_timesteps":
                continue
            _, last = _window_means(self.diag[key])
            if last:
                out[key[len("term_") :]] = last
        return out

    def eval_curve(self):
        """Return (timesteps, per-eval mean reward, per-eval mean length)."""
        if self.evals is None:
            return None
        return (
            np.asarray(self.evals["timesteps"], dtype=float),
            np.asarray(self.evals["results"], dtype=float).mean(axis=1),
            np.asarray(self.evals["ep_lengths"], dtype=float).mean(axis=1),
        )

    def full_horizon_share(self, horizon: int = 1000) -> float | None:
        """Fraction of episodes in the final evaluation that hit the horizon.

        Mean episode length hides a bimodal fall distribution; this is the
        signal that actually separates a reliable balance policy from a
        lucky-average one.
        """
        if self.evals is None:
            return None
        lengths = np.asarray(self.evals["ep_lengths"], dtype=float)[-1]
        return float((lengths >= horizon).mean())


def _delta(a, b):
    if a is None or b is None:
        return ""
    diff = b - a
    return f"{diff:+.3f}"


def _report_signals(runs, signals, heading):
    print(f"\n{heading}")
    print("-" * 96)
    header = f"{'signal':34s}"
    for run in runs:
        header += f"{run.label + ' first→last':>28s}"
    if len(runs) == 2:
        header += f"{'Δ(last)':>12s}"
    print(header)
    for key, label, normalize in signals:
        cells, lasts = [], []
        for run in runs:
            first, last = run.signal(key, normalize)
            lasts.append(last)
            cells.append("n/a" if first is None else f"{first:.3f} → {last:.3f}")
        if all(c == "n/a" for c in cells):
            continue
        row = f"{label:34s}" + "".join(f"{c:>28s}" for c in cells)
        if len(runs) == 2:
            row += f"{_delta(lasts[0], lasts[1]):>12s}"
        print(row)


def _report_eval(runs):
    print("\nEvaluation history")
    print("-" * 96)
    for run in runs:
        curve = run.eval_curve()
        if curve is None:
            print(f"{run.label}: no evaluations.npz")
            continue
        ts, rewards, lengths = curve
        best = int(rewards.argmax())
        share = run.full_horizon_share()
        print(f"{run.label}:")
        print(f"    evaluations           {len(ts)} over {ts[-1]:,.0f} steps")
        print(f"    best mean reward      {rewards[best]:.1f} at {ts[best]:,.0f} steps")
        print(f"    final mean reward     {rewards[-1]:.1f}  ({rewards[-1] / rewards[best]:.0%} of peak)")
        print(f"    final mean ep length  {lengths[-1]:.1f}")
        if share is not None:
            print(f"    final eval: {share:.0%} of episodes reached the 1000-step horizon")
        # Decile trace makes a peak-then-decline shape obvious without a plot.
        step = max(1, len(rewards) // 10)
        trace = " ".join(f"{r:.0f}" for r in rewards[::step])
        print(f"    reward decile trace   {trace}")


def _report_terminations(runs):
    print("\nTermination mix over the last 20% of training")
    print("-" * 96)
    reasons = sorted({r for run in runs for r in run.termination_shares()})
    if not reasons:
        print("(no termination data recorded)")
        return
    shares = [run.termination_shares() for run in runs]
    header = f"{'reason':34s}" + "".join(f"{run.label:>28s}" for run in runs)
    print(header)
    for reason in reasons:
        row = f"{reason:34s}" + "".join(f"{s.get(reason, 0.0):>27.1%} " for s in shares)
        print(row)


def compare(runs):
    print("=" * 96)
    print("Run diagnostics comparison")
    print("=" * 96)
    for run in runs:
        print(f"{run.label:10s} {run.stage_dir}  (action_dim={run.n_act or 'unknown'})")
    _report_signals(runs, _ALGO_SIGNALS, "Policy / optimizer diagnostics (first 20% → last 20%)")
    _report_signals(runs, _ENV_SIGNALS, "Rollout behaviour (first 20% → last 20%)")
    _report_terminations(runs)
    _report_eval(runs)
    print()


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    runs = []
    for arg in argv:
        label, sep, path = arg.partition("=")
        if not sep:
            label, path = Path(arg).name, arg
        stage_dir = Path(path).expanduser()
        if not stage_dir.is_dir():
            raise SystemExit(f"not a directory: {stage_dir}")
        runs.append(Run(label, stage_dir))
    compare(runs)


if __name__ == "__main__":
    main(sys.argv[1:])
