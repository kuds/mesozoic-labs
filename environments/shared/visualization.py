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
    show: bool = True,
) -> "Any":
    """Plot evaluation reward, episode length, tilt angle, and forward velocity.

    Produces a 2x2 grid and optionally saves to *save_path*.
    When *show* is ``False``, the figure is closed after saving
    (headless / sweep usage).
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
    if not show:
        plt.close(fig)

    return fig


def plot_diagnostics_graphs(
    stage_dirs: StageDirs,
    stage_configs: Dict[int, Dict[str, Any]],
    species: str,
    algorithm: str,
    save_dir: "str | Path | None" = None,
    show: bool = True,
) -> "tuple":
    """Create two diagnostic figures tracking advanced training metrics.

    Figure 1 -- Locomotion Health (``locomotion_health.png``, 2x2):
      - Termination Breakdown, Cost of Transport,
        Pelvis Height, Reward Decomposition

    Figure 2 -- Behavioral Metrics (``behavioral_metrics.png``, 3x2):
      - Gait Symmetry + Stride Frequency, Heading Alignment,
        Prey Distance, Strike Success Rate,
        Distance Traveled (cumulative path length),
        Drift Distance (displacement from spawn)

    Returns ``(fig1, fig2)``.  When *show* is ``False``, figures are
    closed after saving (headless / sweep usage).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from environments.shared.diagnostics import DiagnosticsCallback

    species_title = species.title()
    _REWARD_COMPONENTS = [k for k in DiagnosticsCallback.REWARD_KEYS if k != "reward_total"]

    # Mapping from diagnostic reward key → config weight parameter name(s).
    # A signal "matters" to a stage if any of its config weights is non-zero.
    _REWARD_KEY_TO_CONFIG_WEIGHTS: Dict[str, Tuple[str, ...]] = {
        "reward_forward": ("forward_vel_weight",),
        "reward_backward": ("backward_vel_penalty_weight",),
        "reward_drift": ("drift_penalty_weight",),
        "reward_alive": ("alive_bonus",),
        "reward_energy": ("energy_penalty_weight",),
        "reward_tail": ("tail_stability_weight",),
        "reward_gait": ("gait_symmetry_weight", "gait_stability_weight"),
        "reward_posture": ("posture_weight",),
        "reward_nosedive": ("nosedive_weight",),
        "reward_height": ("height_weight",),
        "reward_smoothness": ("smoothness_weight",),
        "reward_heading": ("heading_weight",),
        "reward_lateral": ("lateral_penalty_weight",),
        "reward_spin": ("spin_penalty_weight",),
        "reward_strike": ("strike_bonus",),
        "reward_approach": ("strike_approach_weight", "bite_approach_weight", "food_approach_weight"),
        "reward_proximity": ("strike_proximity_weight",),
        "reward_claw_proximity": ("strike_claw_proximity_weight",),
        "reward_bite": ("bite_bonus",),
        "reward_food": ("food_reach_bonus",),
    }

    def _signal_active(reward_key: str, stage_num: int) -> bool:
        """Return True if *reward_key* has a non-zero weight in this stage's config."""
        env_kw = stage_configs.get(stage_num, {}).get("env_kwargs", {})
        weight_keys = _REWARD_KEY_TO_CONFIG_WEIGHTS.get(reward_key, ())
        if not weight_keys:
            # Unknown reward key — show it to be safe.
            return True
        return any(abs(env_kw.get(wk, 0.0)) > 0 for wk in weight_keys)

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

        # [1,1] Reward Decomposition — only signals with non-zero weight
        for _ci, _rkey in enumerate(_REWARD_COMPONENTS):
            if _rkey in diag and _signal_active(_rkey, stage_num):
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
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 15))
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

        # [0,1] Heading Alignment (with std band to reveal spinning)
        if "heading_alignment" in diag:
            ha = diag["heading_alignment"]
            axes2[0, 1].plot(ts, ha, label=label, color=color)
            if "heading_alignment_std" in diag:
                ha_std = diag["heading_alignment_std"]
                axes2[0, 1].fill_between(
                    ts,
                    ha - ha_std,
                    ha + ha_std,
                    alpha=0.2,
                    color=color,
                )

        # [1,0] Prey / Food Distance
        if "prey_distance" in diag:
            axes2[1, 0].plot(ts, diag["prey_distance"], label=label, color=color)

        # [1,1] Strike / Bite / Food Success Rate
        if "strike_success" in diag:
            axes2[1, 1].plot(ts, diag["strike_success"], label=label, color=color)

        # [2,0] Distance Traveled (cumulative XY path length)
        if "distance_traveled" in diag:
            axes2[2, 0].plot(ts, diag["distance_traveled"], label=label, color=color)

        # [2,1] Drift Distance (displacement from spawn)
        if "drift_distance" in diag:
            axes2[2, 1].plot(ts, diag["drift_distance"], label=label, color=color)

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

    axes2[2, 0].set_xlabel("Timesteps")
    axes2[2, 0].set_ylabel("Distance Traveled (m)")
    axes2[2, 0].set_title(f"{species_title} {algorithm} \u2013 Distance Traveled (path length)")
    axes2[2, 0].legend()
    axes2[2, 0].grid(True, alpha=0.3)

    axes2[2, 1].set_xlabel("Timesteps")
    axes2[2, 1].set_ylabel("Drift Distance (m)")
    axes2[2, 1].set_title(f"{species_title} {algorithm} \u2013 Drift Distance (displacement from spawn)")
    axes2[2, 1].legend()
    axes2[2, 1].grid(True, alpha=0.3)

    fig2.tight_layout()
    if save_dir is not None:
        _p2 = Path(save_dir) / "behavioral_metrics.png"
        fig2.savefig(_p2, dpi=150, bbox_inches="tight")
        logger.info("Behavioral metrics graph saved to: %s", _p2)
    if not show:
        plt.close(fig2)

    return fig1, fig2


def plot_trial_comparison(
    analysis_rows: "list[dict[str, Any]]",
    species: str,
    stage: int,
    save_path: "str | Path | None" = None,
    show: bool = True,
) -> "Any":
    """Plot a 2x3 comparison of top sweep trials.

    Produces a six-panel figure comparing trials on:
      - [0,0] Eval reward (with std error bars)
      - [0,1] Forward velocity (with std error bars)
      - [0,2] Total distance
      - [1,0] Gait symmetry
      - [1,1] Cost of transport
      - [1,2] Reward vs velocity scatter

    Each *analysis_rows* entry is expected to contain keys produced by
    :class:`~environments.shared.metrics.LocomotionMetrics` aggregation:
    ``eval_reward``, ``eval_reward_std``, ``fwd_vel_m/s``, ``fwd_vel_std``,
    ``distance_m``, ``gait_symmetry``, ``cost_of_transport``, and ``trial``
    (a short label for the x-axis).

    Args:
        analysis_rows: List of dicts, one per trial, with the keys above.
        species: Species name for the figure title.
        stage: Curriculum stage number for the figure title.
        save_path: If provided, save the figure to this path.
        show: If ``False``, close the figure after saving (headless mode).

    Returns:
        The matplotlib Figure object.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping trial comparison plot")
        return None

    if not analysis_rows:
        logger.warning("No analysis rows — skipping trial comparison plot")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{species.title()} Stage {stage} — Top {len(analysis_rows)} Trials Comparison", fontsize=14)
    trial_labels = [str(r.get("trial", f"T{i}"))[:12] for i, r in enumerate(analysis_rows)]
    x = range(len(analysis_rows))

    def _vals(key, default=0):
        return [r.get(key, default) for r in analysis_rows]

    # Reward
    axes[0, 0].bar(x, _vals("eval_reward"), yerr=_vals("eval_reward_std"), capsize=4, alpha=0.8, edgecolor="black")
    axes[0, 0].set_ylabel("Mean Reward")
    axes[0, 0].set_title("Eval Reward")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(trial_labels, rotation=45, ha="right", fontsize=8)

    # Forward velocity
    axes[0, 1].bar(
        x, _vals("fwd_vel_m/s"), yerr=_vals("fwd_vel_std"), capsize=4, alpha=0.8, color="tab:orange", edgecolor="black"
    )
    axes[0, 1].set_ylabel("m/s")
    axes[0, 1].set_title("Forward Velocity")
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(trial_labels, rotation=45, ha="right", fontsize=8)

    # Distance
    axes[0, 2].bar(x, _vals("distance_m"), alpha=0.8, color="tab:green", edgecolor="black")
    axes[0, 2].set_ylabel("meters")
    axes[0, 2].set_title("Total Distance")
    axes[0, 2].set_xticks(list(x))
    axes[0, 2].set_xticklabels(trial_labels, rotation=45, ha="right", fontsize=8)

    # Gait symmetry
    axes[1, 0].bar(x, _vals("gait_symmetry"), alpha=0.8, color="tab:purple", edgecolor="black")
    axes[1, 0].set_ylabel("Symmetry")
    axes[1, 0].set_title("Gait Symmetry")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xticks(list(x))
    axes[1, 0].set_xticklabels(trial_labels, rotation=45, ha="right", fontsize=8)

    # Cost of transport
    axes[1, 1].bar(x, _vals("cost_of_transport"), alpha=0.8, color="tab:red", edgecolor="black")
    axes[1, 1].set_ylabel("CoT")
    axes[1, 1].set_title("Cost of Transport (lower = better)")
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(trial_labels, rotation=45, ha="right", fontsize=8)

    # Reward vs Velocity scatter
    axes[1, 2].scatter(_vals("fwd_vel_m/s"), _vals("eval_reward"), s=80, alpha=0.8, edgecolors="black")
    for i, label in enumerate(trial_labels):
        axes[1, 2].annotate(
            label, (_vals("fwd_vel_m/s")[i], _vals("eval_reward")[i]), fontsize=7, ha="left", va="bottom"
        )
    axes[1, 2].set_xlabel("Forward Velocity (m/s)")
    axes[1, 2].set_ylabel("Eval Reward")
    axes[1, 2].set_title("Reward vs Velocity")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Trial comparison plot saved to: %s", save_path)
    if not show:
        plt.close(fig)

    return fig
