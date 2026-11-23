"""Draw the neutral left foreleg of both measured plants on one consistent millimetre scale."""

import tempfile
from pathlib import Path

import matplotlib
import mujoco as mj
import numpy as np

matplotlib.use("Agg")
from compare import HERE, SHORT, restore_forelegs
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


def project(ax, model, short):
    m = mj.MjModel.from_xml_path(str(model))
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, m.key("neutral").id)
    mj.mj_forward(m, d)
    prefix = "left_fore_"
    pivot = [1000 * d.xanchor[m.joint(prefix + joint).id][[0, 2]].copy() for joint in ("pitch", "flexion", "ankle")]
    offset = pivot[0][0]
    for v in pivot:
        v[0] -= offset
    shoulder, elbow, ankle = pivot
    ax.axhline(0, lw=1, color="#677a79", zorder=1)
    ax.axvline(0, ymin=0.03, ymax=0.95, lw=0.9, ls=":", color="#879391", zorder=1)
    for start, stop, color in ((shoulder, elbow, "#1d6760"), (elbow, ankle, "#71998d")):
        ax.plot(*np.vstack([start, stop]).T, lw=10, solid_capstyle="round", color=color, zorder=2)
    if short:
        bracket = m.geom(prefix + "offset_motor_mount")
        half = 1000 * bracket.size[0] * d.geom(bracket.name).xmat.reshape(3, 3)[[0, 2], 0]
        center = 1000 * d.geom(bracket.name).xpos[[0, 2]] - [offset, 0]
        ax.plot(*np.vstack([center - half, center + half]).T, lw=6, color="#828887", zorder=1.5)
    sole = m.geom(prefix + "sole")
    radius = sole.size[0] * 1000
    center = 1000 * d.geom(sole.name).xpos[[0, 2]] - [offset, 0]
    ax.plot([center[0] - 20, center[0] + 20], [center[1]] * 2, solid_capstyle="round", lw=1, color="#36484a")
    for x in np.linspace(center[0] - sole.size[1] * 1000, center[0] + sole.size[1] * 1000, 35):
        ax.add_patch(Circle((x, center[1]), radius, color="#36484a", zorder=2))
    for name, fc, ec in (
        ("x6_flexion_stator", "#f0c88d", "#a05a19"),
        ("x4_pitch_stator", "#cfdddd", "#496565"),
    ):
        g = m.geom(prefix + name)
        center = 1000 * d.geom(g.name).xpos[[0, 2]] - [offset, 0]
        ax.add_patch(Circle(center, 1000 * g.size[0], ec=ec, fc=fc, lw=1.8, alpha=0.90, zorder=3))
        label = "X6" if name.startswith("x6") else "X4"
        ax.text(*center, label, ha="center", va="center", fontweight="bold", color=ec, fontsize=11, zorder=4)
    ax.scatter(elbow[0], elbow[1], s=35, c="#ffffff", edgecolors="#1d6760", lw=1.3, zorder=5)
    ax.text(elbow[0] - 13, elbow[1] - 9, "Elbow", fontsize=10, ha="right", color="#325c55")
    ax.set_title(
        "128 mm upper link" if short else "188 mm upper link", loc="left", fontsize=14, fontweight="bold", pad=12
    )
    ax.set_xlim(-185, 95)
    ax.set_ylim(-13, 365)
    ax.set_xticks([-150, -100, -50, 0, 50])
    ax.set_yticks([0, 100, 200, 300])
    ax.set_aspect("equal")
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", color="#e4e8e8")
    ax.tick_params(length=0, labelcolor="#627273")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.9), dpi=160, sharey=True)
    with tempfile.TemporaryDirectory() as temp:
        longer = Path(temp) / "long.xml"
        restore_forelegs(longer)
        project(axes[0], SHORT, True)
        project(axes[1], longer, False)
    fig.subplots_adjust(top=0.80, bottom=0.17, wspace=0.08, left=0.08, right=0.97)
    fig.text(
        0.10, 0.948, "Two forelegs in the 60 cm chassis", va="top", fontsize=21, fontweight="bold", color="#203536"
    )
    fig.text(
        0.10,
        0.879,
        "Neutral left side view  ·  Same ankle and floor  ·  Robot faces right",
        fontsize=11,
        color="#516566",
    )
    axes[0].set_ylabel("Height above floor (mm)", color="#3b5051")
    fig.text(0.53, 0.091, "Fore–aft offset from shoulder (mm)", ha="center", color="#3b5051")
    fig.text(
        0.10,
        0.034,
        "X4 shoulder pitch and X6 flexion housings shown. Roll motor, shell and linkage guards omitted.",
        fontsize=8.5,
        color="#516566",
    )
    fig.savefig(HERE / "forelegs.png", facecolor="white")


if __name__ == "__main__":
    main()
