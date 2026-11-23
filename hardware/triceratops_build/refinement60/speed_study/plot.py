"""Plot matched six-trial speed sweeps from the checked-in measurements (Matplotlib)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {128: "#9e4a23", 160: "#126e63"}


def load(name):
    return json.loads((HERE / name).read_text())


def main():
    walk = load("walk.json")["runs"]
    steering = load("steering-main.json")["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.65))
    fig.patch.set_facecolor("#fbfaf6")
    for ax in axes:
        ax.set_facecolor("#fbfaf6")
        ax.grid(axis="y", color="#dfdfd7")
        ax.spines[["top", "right"]].set_visible(False)
    for length, color in COLORS.items():
        rows = [r for r in walk if r["upper_mm"] == length]
        cadences = sorted(set(r["cadence"] for r in rows))
        passes = [sum(r["pass_all"] for r in rows if r["cadence"] == c) for c in cadences]
        axes[0].plot(
            cadences,
            passes,
            "o-" if length == 128 else "s--",
            color=color,
            markersize=4,
            linewidth=1.4,
            label=f"{length} mm",
        )
        rows = sorted((r for r in steering if r["length_mm"] == length), key=lambda r: r["command_mps"])
        speeds = [r["command_mps"] for r in rows]
        style = "o-" if length == 128 else "s--"
        axes[1].plot(speeds, [r["all_six_held"] for r in rows], style, color=color, linewidth=1.4, markersize=4)
        axes[2].plot(speeds, [r["any_foot_peak_N"] for r in rows], style, color=color, linewidth=1.4, markersize=4)
    for ax in axes[1:]:
        ax.axvline(0.30, color="#555953", lw=1, ls=":")
        ax.set_xlabel("Commanded speed (m/s)")
    axes[0].set(title="Straight four-beat walk", xlabel="Cadence multiplier", ylabel="Trials passing the walk criteria")
    axes[1].set(title="Moving six-heading course", ylabel="Trials acquiring all six headings")
    axes[2].set(title="Moving six-heading course", ylabel="Largest single-foot force (N)")
    for ax in axes[:2]:
        ax.set_yticks([0, 2, 4, 6])
        ax.set_ylim(-0.3, 6.55)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(loc="upper right", frameon=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.24, wspace=0.36)
    fig.text(
        0.065,
        0.92,
        "60 cm Triceratops   |   Walking and steering speed limits",
        size=19,
        weight="bold",
        color="#24382e",
    )
    fig.text(
        0.065,
        0.04,
        "Three seeds × two conditions per setting. Dotted line marks the stock 0.30 m/s steering command ceiling.\n"
        "Directional forces include the recorded portion of failed runs and exclude the landing after a tilt stop.",
        size=9,
        color="#4a534e",
    )
    fig.savefig(HERE / "media/speed-limits.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    main()
