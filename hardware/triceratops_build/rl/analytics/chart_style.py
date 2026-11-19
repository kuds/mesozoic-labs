"""Shared chart style for the analytics charts (light PNGs), from the dataviz reference palette.

Categorical slots in fixed order (blue, orange, aqua, yellow; validated as an adjacent set, light mode: worst CVD
Delta E 9.1, normal-vision 22.9; aqua and yellow sit below 3:1 on the surface, so every chart labels its series
directly and the JSON / Markdown carry the numbers). Feet keep one colour in every chart. Reference lines and text
use ink tones, never series colours. Every chart carries the SIMULATION tag and the run name."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WASH = "#f0efec"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CRITICAL = "#d03b3b"
FOOT_COLOR = {"left_fore": SERIES[0], "right_fore": SERIES[1], "left_hind": SERIES[2], "right_hind": SERIES[3]}
FOOT_LABEL = {"left_fore": "LF  left fore", "right_fore": "RF  right fore", "left_hind": "LH  left hind",
              "right_hind": "RH  right hind"}
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
LW = 1.5  # points: 2 px at 100 dpi is 1.44 pt
DPI = 100


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2, "font.size": 9,
        "lines.linewidth": LW, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "font.family": "DejaVu Sans"})


def frame(fig, title: str, subtitle: str, footer: str) -> None:
    """Title and subtitle top left, the SIMULATION tag top right, the source line bottom left."""
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.012, 0.945, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.988, 0.985, "SIMULATION", ha="right", va="top", fontsize=11, fontweight="bold", color=INK,
             bbox=dict(boxstyle="round,pad=0.35", fc=WASH, ec=AXIS, lw=0.8))
    fig.text(0.012, 0.012, footer, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def save(fig, path) -> None:
    """Save as PNG; a file over 400 KB is re-saved with a 256-colour palette (the 500 KB limit per chart)."""
    from pathlib import Path

    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    if Path(path).stat().st_size > 400 * 1024:
        from PIL import Image

        im = Image.open(path).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                                      dither=Image.Dither.NONE)
        im.save(path, optimize=True)
