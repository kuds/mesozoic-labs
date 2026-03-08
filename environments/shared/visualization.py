"""Training visualization utilities.

Produces diagnostic graphs from training artifacts (``evaluations.npz``
and ``diagnostics.npz``).  Originally defined inline in the Colab
training notebook, now shared so both the notebook and sweep trial
worker generate consistent graphs.

All functions use ``matplotlib.pyplot`` via the ``Agg`` backend when
called headless (no display).  The notebook can continue to call
``plt.show()`` after each function returns.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

logger = logging.getLogger(__name__)

# Type alias for the (stage_num, stage_dir) tuples used throughout.
StageDirs = Sequence[Tuple[int, "str | Path"]]


def plot_training_curves(
    stage_dirs: StageDirs,
    stage_configs: Dict[int, Dict[str, Any]],
    species: str,
    algorithm: str,
    save_path: "str | Path | None" = None,
) -> "Any":
    """Plot evaluation reward, episode length, tilt angle, and forward velocity.

    Produces a 2x2 grid and optionally saves to *save_path*.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    species_title = species.title()

    for stage_num, stage_dir in stage_dirs:
        stage_dir = Path(stage_dir)
        eval_log = stage_dir / "evaluations.npz"
        if not eval_log.exists():
            logger.info("No evaluation log found for stage %d.", stage_num)
            continue

        data = np.load(eval_log)
        timesteps = data["timesteps"]
        results = data["results"]
        label = f"Stage {stage_num}: {stage_configs[stage_num]['name']}"

        # Reward curve
        mean_rewards = np.mean(results, axis=1)
        std_rewards = np.std(results, axis=1)
        axes[0, 0].plot(timesteps, mean_rewards, label=label)
        axes[0, 0].fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
        )

        # Episode length curve
        if "ep_lengths" in data:
            ep_lengths = data["ep_lengths"]
            mean_lengths = np.mean(ep_lengths, axis=1)
            std_lengths = np.std(ep_lengths, axis=1)
            axes[0, 1].plot(timesteps, mean_lengths, label=label)
            axes[0, 1].fill_between(
                timesteps,
                mean_lengths - std_lengths,
                mean_lengths + std_lengths,
                alpha=0.2,
            )

        # Curriculum threshold lines
        cur = stage_configs[stage_num].get("curriculum_kwargs", {})
        min_reward = cur.get("min_avg_reward")
        min_length = cur.get("min_avg_episode_length")
        color = axes[0, 0].get_lines()[-1].get_color()
        if min_reward is not None:
            axes[0, 0].axhline(y=min_reward, color=color, linestyle="--", alpha=0.5)
        if min_length is not None:
            axes[0, 1].axhline(y=min_length, color=color, linestyle="--", alpha=0.5)

        # Tilt angle and forward velocity from diagnostics
        diag_log = stage_dir / "diagnostics.npz"
        if diag_log.exists():
            diag = np.load(diag_log)
            if "tilt_angle" in diag and "timesteps" in diag:
                diag_ts = diag["timesteps"]
                tilt = np.degrees(diag["tilt_angle"])
                axes[1, 0].plot(diag_ts, tilt, label=label, color=color)
            if "forward_vel" in diag and "timesteps" in diag:
                diag_ts = diag["timesteps"]
                fwd_vel = diag["forward_vel"]
                axes[1, 1].plot(diag_ts, fwd_vel, label=label, color=color)

    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].set_ylabel("Mean Reward")
    axes[0, 0].set_title(f"{species_title} {algorithm} - Eval Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Timesteps")
    axes[0, 1].set_ylabel("Mean Episode Length (steps)")
    axes[0, 1].set_title(f"{species_title} {algorithm} - Episode Length")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Timesteps")
    axes[1, 0].set_ylabel("Mean Tilt Angle (degrees)")
    axes[1, 0].set_title(f"{species_title} {algorithm} - Tilt Angle")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Timesteps")
    axes[1, 1].set_ylabel("Mean Forward Velocity (m/s)")
    axes[1, 1].set_title(f"{species_title} {algorithm} - Speed")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training curves saved to: %s", save_path)

    return fig


def plot_diagnostics_graphs(
    stage_dirs: StageDirs,
    stage_configs: Dict[int, Dict[str, Any]],
    species: str,
    algorithm: str,
    save_dir: "str | Path | None" = None,
    show: bool = True,
) -> "tuple":
    """Create two 2x2 diagnostic figures tracking advanced training metrics.

    Figure 1 -- Locomotion Health (``locomotion_health.png``):
      - Termination Breakdown, Cost of Transport,
        Pelvis Height, Reward Decomposition

    Figure 2 -- Behavioral Metrics (``behavioral_metrics.png``):
      - Gait Symmetry + Stride Frequency, Heading Alignment,
        Prey Distance, Strike Success Rate

    Returns ``(fig1, fig2)``.  When *show* is ``False``, figures are
    closed after saving (headless / sweep usage).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from environments.shared.diagnostics import DiagnosticsCallback

    species_title = species.title()
    _REWARD_COMPONENTS = [k for k in DiagnosticsCallback.REWARD_KEYS if k != "reward_total"]

    # -----------------------------------------------------------
    # Figure 1: Locomotion Health Metrics
    # -----------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(
        f"{species_title} {algorithm} \u2013 Locomotion Health",
        fontsize=14,
        fontweight="bold",
    )
    _reward_colors = plt.cm.tab10.colors
    _all_term_data: dict = {}

    for stage_num, stage_dir in stage_dirs:
        stage_dir = Path(stage_dir)
        diag_log = stage_dir / "diagnostics.npz"
        if not diag_log.exists():
            continue
        diag = np.load(diag_log)
        label = f"Stage {stage_num}: {stage_configs[stage_num]['name']}"
        ts = diag["timesteps"] if "timesteps" in diag else None
        if ts is None or len(ts) == 0:
            continue
        color = _reward_colors[stage_num % 10]

        # [0,0] Collect termination breakdown per stage
        _term_keys = [k for k in diag.files if k.startswith("term_") and k != "term_timesteps"]
        if _term_keys:
            _all_term_data[label] = {k[5:]: float(np.mean(diag[k])) for k in _term_keys}

        # [0,1] Cost of Transport
        if "reward_energy" in diag and "forward_vel" in diag:
            _energy = np.abs(diag["reward_energy"])
            _fwd = np.maximum(diag["forward_vel"], 0.01)
            axes1[0, 1].plot(ts, _energy / _fwd, label=label, color=color)

        # [1,0] Pelvis Height
        if "pelvis_height" in diag:
            axes1[1, 0].plot(ts, diag["pelvis_height"], label=label, color=color)

        # [1,1] Reward Decomposition
        for _ci, _rkey in enumerate(_REWARD_COMPONENTS):
            if _rkey in diag:
                axes1[1, 1].plot(
                    ts,
                    diag[_rkey],
                    label=f"S{stage_num} {_rkey.replace('reward_', '')}",
                    color=_reward_colors[_ci % len(_reward_colors)],
                    linestyle=["-", "--", "-."][stage_num % 3],
                )

    # Render termination breakdown as grouped bar chart
    _ax_term = axes1[0, 0]
    if _all_term_data:
        _all_reasons = sorted({r for d in _all_term_data.values() for r in d})
        _x = np.arange(len(_all_term_data))
        _w = 0.8 / max(len(_all_reasons), 1)
        for _ri, _reason in enumerate(_all_reasons):
            _vals = [_all_term_data[_lbl].get(_reason, 0.0) for _lbl in _all_term_data]
            _offset = (_ri - len(_all_reasons) / 2 + 0.5) * _w
            _ax_term.bar(_x + _offset, _vals, _w, label=_reason)
        _ax_term.set_xticks(_x)
        _ax_term.set_xticklabels(
            list(_all_term_data.keys()),
            rotation=15,
            ha="right",
            fontsize=8,
        )
        _ax_term.legend(fontsize=7, loc="upper right")
    else:
        _ax_term.text(
            0.5,
            0.5,
            "No termination data yet",
            transform=_ax_term.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
    _ax_term.set_title(f"{species_title} {algorithm} \u2013 Termination Breakdown")
    _ax_term.set_ylabel("Mean Fraction")
    _ax_term.grid(True, alpha=0.3)

    axes1[0, 1].set_xlabel("Timesteps")
    axes1[0, 1].set_ylabel("Cost of Transport (energy / speed)")
    axes1[0, 1].set_title(f"{species_title} {algorithm} \u2013 Cost of Transport")
    axes1[0, 1].legend(fontsize=8)
    axes1[0, 1].grid(True, alpha=0.3)

    axes1[1, 0].set_xlabel("Timesteps")
    axes1[1, 0].set_ylabel("Pelvis Height (m)")
    axes1[1, 0].set_title(f"{species_title} {algorithm} \u2013 Pelvis Height")
    axes1[1, 0].legend()
    axes1[1, 0].grid(True, alpha=0.3)

    axes1[1, 1].set_xlabel("Timesteps")
    axes1[1, 1].set_ylabel("Mean Reward Component")
    axes1[1, 1].set_title(f"{species_title} {algorithm} \u2013 Reward Decomposition")
    axes1[1, 1].legend(fontsize=7, loc="upper left")
    axes1[1, 1].grid(True, alpha=0.3)

    fig1.tight_layout()
    if save_dir is not None:
        _p1 = Path(save_dir) / "locomotion_health.png"
        fig1.savefig(_p1, dpi=150, bbox_inches="tight")
        logger.info("Locomotion health graph saved to: %s", _p1)
    if not show:
        plt.close(fig1)

    # -----------------------------------------------------------
    # Figure 2: Behavioral Metrics
    # -----------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(
        f"{species_title} {algorithm} \u2013 Behavioral Metrics",
        fontsize=14,
        fontweight="bold",
    )

    for stage_num, stage_dir in stage_dirs:
        stage_dir = Path(stage_dir)
        diag_log = stage_dir / "diagnostics.npz"
        if not diag_log.exists():
            continue
        diag = np.load(diag_log)
        label = f"Stage {stage_num}: {stage_configs[stage_num]['name']}"
        ts = diag["timesteps"] if "timesteps" in diag else None
        if ts is None or len(ts) == 0:
            continue
        color = plt.cm.tab10.colors[stage_num % 10]

        # [0,0] Gait Symmetry + Stride Frequency proxy
        if "l_foot_contact" in diag and "r_foot_contact" in diag:
            _l = diag["l_foot_contact"]
            _r = diag["r_foot_contact"]
            _gait_sym = 1.0 - np.abs(_l - _r) / (_l + _r + 1e-8)
            _stride_proxy = (_l + _r) / 2.0
            axes2[0, 0].plot(
                ts,
                _gait_sym,
                label=f"{label} \u2013 gait sym",
                color=color,
                linestyle="-",
            )
            axes2[0, 0].plot(
                ts,
                _stride_proxy,
                label=f"{label} \u2013 stride freq",
                color=color,
                linestyle="--",
                alpha=0.7,
            )

        # [0,1] Heading Alignment
        if "heading_alignment" in diag:
            axes2[0, 1].plot(ts, diag["heading_alignment"], label=label, color=color)

        # [1,0] Prey / Food Distance
        if "prey_distance" in diag:
            axes2[1, 0].plot(ts, diag["prey_distance"], label=label, color=color)

        # [1,1] Strike / Bite / Food Success Rate
        if "strike_success" in diag:
            axes2[1, 1].plot(ts, diag["strike_success"], label=label, color=color)

    axes2[0, 0].set_xlabel("Timesteps")
    axes2[0, 0].set_ylabel("Gait Symmetry (\u2013) / Stride Freq proxy (--)")
    axes2[0, 0].set_title(f"{species_title} {algorithm} \u2013 Gait Symmetry + Stride Frequency")
    axes2[0, 0].legend(fontsize=8)
    axes2[0, 0].grid(True, alpha=0.3)

    axes2[0, 1].set_xlabel("Timesteps")
    axes2[0, 1].set_ylabel("Heading Alignment (cos \u03b8)")
    axes2[0, 1].set_title(f"{species_title} {algorithm} \u2013 Heading Alignment")
    axes2[0, 1].legend()
    axes2[0, 1].grid(True, alpha=0.3)

    axes2[1, 0].set_xlabel("Timesteps")
    axes2[1, 0].set_ylabel("Prey Distance (m)")
    axes2[1, 0].set_title(f"{species_title} {algorithm} \u2013 Prey Distance")
    axes2[1, 0].legend()
    axes2[1, 0].grid(True, alpha=0.3)

    axes2[1, 1].set_xlabel("Timesteps")
    axes2[1, 1].set_ylabel("Strike Success Rate")
    axes2[1, 1].set_title(f"{species_title} {algorithm} \u2013 Strike Success Rate")
    axes2[1, 1].legend()
    axes2[1, 1].grid(True, alpha=0.3)

    fig2.tight_layout()
    if save_dir is not None:
        _p2 = Path(save_dir) / "behavioral_metrics.png"
        fig2.savefig(_p2, dpi=150, bbox_inches="tight")
        logger.info("Behavioral metrics graph saved to: %s", _p2)
    if not show:
        plt.close(fig2)

    return fig1, fig2
