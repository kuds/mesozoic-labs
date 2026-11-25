"""Training visualization utilities.

Produces diagnostic graphs from training artifacts (``evaluations.npz``
and ``diagnostics.npz``).  Originally defined inline in the Colab
training notebook, now shared so the notebook and
``reporting.generate_stage_artifacts`` generate consistent graphs.

All functions use ``matplotlib.pyplot`` via the ``Agg`` backend when
called headless (no display).  The notebook can continue to call
``plt.show()`` after each function returns.
"""

from __future__ import annotations

import logging
import math
import zlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .direction_commands import stand_weights_of
from .model_id import ModelId
from .species_names import model_display_name
from .stage_manifest import StageManifestError, stage_display_name, stage_label

logger = logging.getLogger(__name__)

# Type alias for the (stage_ref, stage_dir) tuples used throughout: a stage
# reference is a legacy number (1-3) or a semantic stage id ("recovery").
StageDirs = Sequence[tuple[int | str, str | Path]]

#: The one model whose Stage-1 stance figure is drawn (:func:`plot_stance_diagnostics`);
#: keyed by model since decision D-D33 (it was keyed to the retired T-Rex species ID).
STANCE_DIAGNOSTICS_MODEL_ID = ModelId("tyrannosaurus_rex", "anatomical")


def _stage_style_index(stage_ref: "int | str", model_id: ModelId | None = None) -> int:
    """Stable small int for a stage reference, driving color/linestyle picks.

    Integer stages pass through unchanged so stages 1-3 keep their exact
    historical colors and linestyles.  A semantic id uses its manifest
    position when the model's stage manifest resolves it; otherwise a crc32
    fallback — NOT ``hash()``, which varies per process — keeps the style
    deterministic across runs.  Regression (F9): ``"recovery" % 10`` raised
    TypeError inside stage_artifacts' broad try/except, silently costing
    recovery runs every diagnostics figure.
    """
    if isinstance(stage_ref, int):
        return stage_ref
    if model_id is not None:
        from environments.shared.stage_manifest import StageManifestError, load_stage_manifest

        try:
            return load_stage_manifest(model_id).by_id(stage_ref).position
        except StageManifestError:
            pass
    return zlib.crc32(str(stage_ref).encode("utf-8")) % 1000


def _safe_legend(ax, **kwargs) -> None:
    """Call ``ax.legend()`` only when there are labeled artists, avoiding matplotlib warnings."""
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(**kwargs)


def _has_finite(diag, key: str) -> bool:
    """Return True if *diag* holds *key* with at least one finite value.

    Presence is not evidence: DiagnosticsCallback saves EVERY info key for
    every species, so a key the environment never emits is an all-NaN series
    (a biped's rr/rl_foot_contact, allosaurus's heading_alignment). Keying a
    layout off ``key in diag`` rendered an empty "diagonal pair" panel for
    bipeds (narrow-tolerance postmortem, item 6).
    """
    import numpy as np

    if key not in diag:
        return False
    arr = np.asarray(diag[key], dtype=float)
    return arr.size > 0 and bool(np.isfinite(arr).any())


def _smooth(values, window: int = 50):
    """Apply a rolling-mean smoothing filter to *values*.

    Uses ``numpy.convolve`` with a uniform kernel.  Returns the original
    array unmodified when it has fewer points than *window*.
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # 'same' keeps the output length equal to the input length
    smoothed = np.convolve(values, kernel, mode="same")
    # Fix edge effects: at the borders the convolution averages over
    # fewer elements, so we leave the raw values for the first/last
    # half-window points.
    hw = window // 2
    smoothed[:hw] = values[:hw]
    smoothed[-hw:] = values[-hw:]
    return smoothed


def plot_training_curves(
    stage_dirs: StageDirs,
    stage_configs: dict[int | str, dict[str, Any]],
    model_id: ModelId,
    algorithm: str,
    save_path: "str | Path | None" = None,
    show: bool = True,
) -> "Any":
    """Plot evaluation reward, episode length, tilt angle, and forward velocity.

    Produces a 2x2 grid and optionally saves to *save_path*.
    When *show* is ``False``, the figure is closed after saving
    (headless usage, e.g. ``generate_stage_artifacts``).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    model_title = model_display_name(model_id)

    for stage_num, stage_dir in stage_dirs:
        stage_dir = Path(stage_dir)
        eval_log = stage_dir / "evaluations.npz"
        if not eval_log.exists():
            logger.info("No evaluation log found for stage %s.", stage_num)
            continue

        data = np.load(eval_log)
        timesteps = data["timesteps"]
        results = data["results"]
        label = f"Stage {stage_num}: {stage_display_name(stage_num, stage_configs[stage_num]['name'])}"

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
        elif cur.get("min_full_horizon_fraction") is not None:
            # stance_quality/v1 states the survival requirement as a fraction
            # of episodes rather than a mean step count, so the length panel
            # would otherwise lose its threshold line entirely. Draw the
            # equivalent step count so the plot still shows what must be met.
            horizon = stage_configs[stage_num].get("env_kwargs", {}).get("max_episode_steps", 1000)
            axes[0, 1].axhline(
                y=float(cur["min_full_horizon_fraction"]) * float(horizon),
                color=color,
                linestyle=":",
                alpha=0.5,
            )

        # Tilt angle and forward velocity from diagnostics
        diag_log = stage_dir / "diagnostics.npz"
        if diag_log.exists():
            diag = np.load(diag_log)
            if "tilt_angle" in diag and "timesteps" in diag:
                diag_ts = diag["timesteps"]
                tilt = np.degrees(diag["tilt_angle"])
                axes[1, 0].plot(diag_ts, _smooth(tilt), label=label, color=color)
            if "forward_vel" in diag and "timesteps" in diag:
                diag_ts = diag["timesteps"]
                fwd_vel = diag["forward_vel"]
                axes[1, 1].plot(diag_ts, _smooth(fwd_vel), label=label, color=color)

    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].set_ylabel("Mean Reward")
    axes[0, 0].set_title(f"{model_title} {algorithm} - Eval Reward")
    _safe_legend(axes[0, 0])
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Timesteps")
    axes[0, 1].set_ylabel("Mean Episode Length (steps)")
    axes[0, 1].set_title(f"{model_title} {algorithm} - Episode Length")
    _safe_legend(axes[0, 1])
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Timesteps")
    axes[1, 0].set_ylabel("Mean Tilt Angle (degrees)")
    axes[1, 0].set_title(f"{model_title} {algorithm} - Tilt Angle")
    _safe_legend(axes[1, 0])
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Timesteps")
    axes[1, 1].set_ylabel("Mean Forward Velocity (m/s)")
    axes[1, 1].set_title(f"{model_title} {algorithm} - Speed")
    _safe_legend(axes[1, 1])
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training curves saved to: %s", save_path)
    if not show:
        plt.close(fig)

    return fig


def termination_fraction_keys(files: Iterable[str]) -> list[str]:
    """The ``diagnostics.npz`` keys the termination panel reads as fractions: every ``term_`` key but ``term_timesteps``.

    The per-rollout counts beside them are ``rollout_term_count_<reason>``,
    outside this prefix by design: under ``term_`` they would be plotted on
    the fraction axis as a reason named ``count_<reason>``.
    """
    return [key for key in files if key.startswith("term_") and key != "term_timesteps"]


#: Mapping from diagnostic reward key -> config weight parameter name(s).  A signal "matters" to a stage if any
#: of its config weights is non-zero (``plot_diagnostics_graphs``' reward decomposition).  A reward key added to
#: ``DiagnosticsCallback.REWARD_KEYS`` gets its entry here in the same change.
_REWARD_KEY_TO_CONFIG_WEIGHTS: dict[str, tuple[str, ...]] = {
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
    "reward_bilateral_support": ("bilateral_support_weight",),
    "reward_foot_load_balance": ("foot_load_balance_weight",),
    "reward_leg_home_pose": ("leg_home_pose_weight",),
    "reward_head_clearance": ("head_clearance_weight",),
    "reward_neck_posture": ("neck_posture_weight",),
    "reward_tail_home_pose": ("tail_home_pose_weight",),
    "reward_neck_home_pose": ("neck_home_pose_weight",),
    "reward_foot_flatness": ("foot_flatness_weight",),
    "reward_stance_width": ("stance_width_weight",),
    "reward_floor_impact": ("floor_impact_weight",),
    "reward_airborne_substeps": ("airborne_substep_weight",),
    "reward_action_saturation": ("action_saturation_weight",),
    "reward_action_jerk": ("action_jerk_weight",),
    "reward_lateral_lean": ("lateral_lean_weight",),
    "reward_stance_stagger": ("stance_stagger_weight",),
    "reward_stance_cop_band": ("stance_cop_band_weight",),
    "reward_support_geom_coverage": ("support_geom_coverage_weight",),
    "reward_smoothness": ("smoothness_weight",),
    "reward_heading": ("heading_weight",),
    "reward_lateral": ("lateral_penalty_weight",),
    "reward_spin": ("spin_penalty_weight",),
    "reward_idle": ("idle_penalty_weight",),
}
#: The live command's tracking term (decision D-D60): ``DiagnosticsCallback`` records it as one of its
#: ``COMMAND_TRACKING_KEYS``, only from a run whose env emitted it, and its weight is the command config's
#: ``tracking_weight``, not an ``[env]`` kwarg.  The reward decomposition draws it after every other term.
_TRACKING_REWARD_KEY = "reward_tracking"
#: The tracking term's own colour and line width in the reward decomposition.  Every other term takes a tab10
#: colour by its index, and those wrap every ten terms, so the tracking term's index would share one with an active
#: walk term (``reward_tail``, ``reward_height``, ``reward_stance_width``) at the same per-stage line style; black is
#: no tab10 colour, and the heavier line marks the walk's dominant term apart from the terms it outweighs.
_TRACKING_REWARD_COLOR = "black"
_TRACKING_REWARD_LINEWIDTH = 2.5


def _tracking_weight(command_config: Any) -> float:
    """The command config's ``tracking_weight`` (a loaded config or its record); 0 without a live command."""
    if command_config is None:
        return 0.0
    value = (
        command_config.get("tracking_weight")
        if isinstance(command_config, Mapping)
        else getattr(command_config, "tracking_weight", None)
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 0.0
    return float(value)


def plot_diagnostics_graphs(
    stage_dirs: StageDirs,
    stage_configs: dict[int | str, dict[str, Any]],
    model_id: ModelId,
    algorithm: str,
    save_dir: "str | Path | None" = None,
    show: bool = True,
) -> "tuple":
    """Create two diagnostic figures tracking advanced training metrics.

    Figure 1 -- Locomotion Health (``locomotion_health.png``):
      - Termination Breakdown, Pelvis Height,
        Reward Decomposition across the bottom row

    Figure 2 -- Behavioral Metrics (``behavioral_metrics.png``):
      - Distance Traveled (cumulative path length) across the top row,
        Heading Alignment and Drift Distance (displacement from spawn)
        below it.  A panel no stage recorded (allosaurus emits no
        heading alignment) reads "No data recorded" instead of drawing
        blank.  The hunt's prey-distance and strike-success row left
        with it (decision D-D59).

    Neither figure draws a cost of transport or a gait symmetry:
    ``diagnostics.npz`` holds rollout means of the action penalty
    (``reward_energy``) and of each foot's contact force, which
    measure neither (an in-phase hop and an alternating walk give
    equal left and right means).

    Returns ``(fig1, fig2)``.  When *show* is ``False``, figures are
    closed after saving (headless usage, e.g. ``generate_stage_artifacts``).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from environments.shared.diagnostics import DiagnosticsCallback

    model_title = model_display_name(model_id)
    # The live command's tracking term last, so every other term keeps its colour on a command-free stage, whose
    # diagnostics.npz holds no reward_tracking series.
    _REWARD_COMPONENTS = [k for k in DiagnosticsCallback.REWARD_KEYS if k != "reward_total"] + [_TRACKING_REWARD_KEY]

    def _signal_active(reward_key: str, stage_num: "int | str") -> bool:
        """Return True if *reward_key* has a non-zero weight in this stage's config.

        A live stage's stand block counts (decision D-D62): a term its ``[env]``
        leaves at 0 is paid on the stand episodes when the block names its weight.
        The tracking term is active on a live stage whose command config's
        ``tracking_weight`` is above 0.
        """
        env_kw = stage_configs.get(stage_num, {}).get("env_kwargs", {})
        if reward_key == _TRACKING_REWARD_KEY:
            live = str(env_kw.get("command_mode", "none")) != "none"
            return live and _tracking_weight(env_kw.get("command_config")) > 0.0
        weight_keys = _REWARD_KEY_TO_CONFIG_WEIGHTS.get(reward_key, ())
        if not weight_keys:
            # Unknown reward key — show it to be safe.
            return True
        stand = stand_weights_of(env_kw.get("command_config"))
        return any(abs(env_kw.get(wk, 0.0)) > 0 or abs(stand.get(wk, 0.0)) > 0 for wk in weight_keys)

    # -----------------------------------------------------------
    # Figure 1: Locomotion Health Metrics (the reward decomposition,
    # the densest panel, takes the whole bottom row)
    # -----------------------------------------------------------
    fig1, axes1 = plt.subplot_mosaic(
        [["termination", "pelvis_height"], ["rewards", "rewards"]],
        figsize=(14, 10),
    )
    fig1.suptitle(
        f"{model_title} {algorithm} \u2013 Locomotion Health",
        fontsize=14,
        fontweight="bold",
    )
    # Call the colormap instead of touching .colors — the stubs type
    # plt.cm.tab10 as the base Colormap, which has no .colors attribute.
    _reward_colors = [plt.cm.tab10(i) for i in range(plt.cm.tab10.N)]
    _all_term_data: dict = {}

    for stage_num, stage_dir in stage_dirs:
        stage_dir = Path(stage_dir)
        diag_log = stage_dir / "diagnostics.npz"
        if not diag_log.exists():
            continue
        diag = np.load(diag_log)
        label = f"Stage {stage_num}: {stage_display_name(stage_num, stage_configs[stage_num]['name'])}"
        ts = diag["timesteps"] if "timesteps" in diag else None
        if ts is None or len(ts) == 0:
            continue
        style_idx = _stage_style_index(stage_num, model_id)
        color = _reward_colors[style_idx % 10]

        # Termination breakdown, collected per stage
        _term_keys = termination_fraction_keys(diag.files)
        if _term_keys:
            _all_term_data[label] = {k[5:]: float(np.mean(diag[k])) for k in _term_keys}

        # Pelvis Height
        if "pelvis_height" in diag:
            axes1["pelvis_height"].plot(ts, _smooth(diag["pelvis_height"]), label=label, color=color)

        # Reward Decomposition — only signals with non-zero weight
        for _ci, _rkey in enumerate(_REWARD_COMPONENTS):
            if _rkey in diag and _signal_active(_rkey, stage_num):
                # Only a live stage draws the tracking term, so a command-free stage's lines are unchanged.
                _tracking = _rkey == _TRACKING_REWARD_KEY
                _line_style: dict[str, Any] = {"linewidth": _TRACKING_REWARD_LINEWIDTH} if _tracking else {}
                axes1["rewards"].plot(
                    ts,
                    _smooth(diag[_rkey]),
                    label=f"S{stage_num} {_rkey.replace('reward_', '')}",
                    color=_TRACKING_REWARD_COLOR if _tracking else _reward_colors[_ci % len(_reward_colors)],
                    linestyle=["-", "--", "-."][style_idx % 3],
                    **_line_style,
                )

    # Render termination breakdown as grouped bar chart
    _ax_term = axes1["termination"]
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
    _ax_term.set_title(f"{model_title} {algorithm} \u2013 Termination Breakdown")
    _ax_term.set_ylabel("Mean Fraction")
    _ax_term.grid(True, alpha=0.3)

    axes1["pelvis_height"].set_xlabel("Timesteps")
    axes1["pelvis_height"].set_ylabel("Pelvis Height (m)")
    axes1["pelvis_height"].set_title(f"{model_title} {algorithm} \u2013 Pelvis Height")
    _safe_legend(axes1["pelvis_height"])
    axes1["pelvis_height"].grid(True, alpha=0.3)

    axes1["rewards"].set_xlabel("Timesteps")
    axes1["rewards"].set_ylabel("Mean Reward Component")
    axes1["rewards"].set_title(f"{model_title} {algorithm} \u2013 Reward Decomposition")
    _safe_legend(axes1["rewards"], fontsize=7, loc="upper left")
    axes1["rewards"].grid(True, alpha=0.3)

    fig1.tight_layout()
    if save_dir is not None:
        _p1 = Path(save_dir) / "locomotion_health.png"
        fig1.savefig(_p1, dpi=150, bbox_inches="tight")
        logger.info("Locomotion health graph saved to: %s", _p1)
    if not show:
        plt.close(fig1)

    # -----------------------------------------------------------
    # Figure 2: Behavioral Metrics (distance traveled, which every
    # species records, across the top row, then one row of two panels)
    # -----------------------------------------------------------
    fig2, axes2 = plt.subplot_mosaic(
        [["distance", "distance"], ["heading", "drift"]],
        figsize=(14, 10),
    )
    fig2.suptitle(
        f"{model_title} {algorithm} \u2013 Behavioral Metrics",
        fontsize=14,
        fontweight="bold",
    )

    for stage_num, stage_dir in stage_dirs:
        stage_dir = Path(stage_dir)
        diag_log = stage_dir / "diagnostics.npz"
        if not diag_log.exists():
            continue
        diag = np.load(diag_log)
        label = f"Stage {stage_num}: {stage_display_name(stage_num, stage_configs[stage_num]['name'])}"
        ts = diag["timesteps"] if "timesteps" in diag else None
        if ts is None or len(ts) == 0:
            continue
        color = plt.cm.tab10(_stage_style_index(stage_num, model_id) % 10)

        # Heading Alignment (with std band to reveal spinning)
        if _has_finite(diag, "heading_alignment"):
            ha = _smooth(diag["heading_alignment"])
            axes2["heading"].plot(ts, ha, label=label, color=color)
            if "heading_alignment_std" in diag:
                ha_std = _smooth(diag["heading_alignment_std"])
                axes2["heading"].fill_between(
                    ts,
                    ha - ha_std,
                    ha + ha_std,
                    alpha=0.2,
                    color=color,
                )

        # Distance Traveled (cumulative XY path length)
        if _has_finite(diag, "distance_traveled"):
            axes2["distance"].plot(ts, _smooth(diag["distance_traveled"]), label=label, color=color)

        # Drift Distance (displacement from spawn)
        if _has_finite(diag, "drift_distance"):
            axes2["drift"].plot(ts, _smooth(diag["drift_distance"]), label=label, color=color)

    axes2["heading"].set_xlabel("Timesteps")
    axes2["heading"].set_ylabel("Heading Alignment (cos \u03b8)")
    axes2["heading"].set_title(f"{model_title} {algorithm} \u2013 Heading Alignment")
    _safe_legend(axes2["heading"])
    axes2["heading"].grid(True, alpha=0.3)

    axes2["distance"].set_xlabel("Timesteps")
    axes2["distance"].set_ylabel("Distance Traveled (m)")
    axes2["distance"].set_title(f"{model_title} {algorithm} \u2013 Distance Traveled (path length)")
    _safe_legend(axes2["distance"])
    axes2["distance"].grid(True, alpha=0.3)

    axes2["drift"].set_xlabel("Timesteps")
    axes2["drift"].set_ylabel("Drift Distance (m)")
    axes2["drift"].set_title(f"{model_title} {algorithm} \u2013 Drift Distance (displacement from spawn)")
    _safe_legend(axes2["drift"])
    axes2["drift"].grid(True, alpha=0.3)

    for _ax in axes2.values():
        if not _ax.get_lines():
            _ax.text(
                0.5,
                0.5,
                "No data recorded",
                transform=_ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )

    fig2.tight_layout()
    if save_dir is not None:
        _p2 = Path(save_dir) / "behavioral_metrics.png"
        fig2.savefig(_p2, dpi=150, bbox_inches="tight")
        logger.info("Behavioral metrics graph saved to: %s", _p2)
    if not show:
        plt.close(fig2)

    return fig1, fig2


def plot_foot_contacts(
    stage_dirs: StageDirs,
    stage_configs: dict[int | str, dict[str, Any]],
    model_id: ModelId,
    algorithm: str,
    save_path: "str | Path | None" = None,
    show: bool = True,
) -> "Any":
    """Plot per-foot contact force over training time.

    Auto-detects bipedal (2 feet: R/L) vs quadrupedal (4 feet: FR/FL/RR/RL)
    from the diagnostics data.  For bipeds, plots 2 lines per stage.  For
    quadrupeds, plots 4 individual lines plus 2 diagonal-pair composite
    signals (Diag-A = FR+RL, Diag-B = FL+RR).

    Args:
        stage_dirs: Sequence of ``(stage_ref, stage_dir)`` tuples.
        stage_configs: Mapping of stage reference to config dict.
        model_id: The model, whose display name titles the figure.
        algorithm: Algorithm name for the figure title.
        save_path: If provided, save the figure to this path.
        show: If ``False``, close the figure after saving (headless mode).

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    model_title = model_display_name(model_id)

    # Detect whether any stage has quadrupedal data (an all-NaN rr/rl series
    # would also route a biped's R/L lines through the four-foot branch
    # under FR/FL labels)
    is_quadruped = False
    for _, stage_dir in stage_dirs:
        diag_log = Path(stage_dir) / "diagnostics.npz"
        if diag_log.exists():
            diag = np.load(diag_log)
            if _has_finite(diag, "rr_foot_contact") or _has_finite(diag, "rl_foot_contact"):
                is_quadruped = True
                break

    if is_quadruped:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        ax_feet = axes[0]
        ax_diag = axes[1]
    else:
        fig, ax_feet = plt.subplots(1, 1, figsize=(14, 5))
        ax_diag = None

    for stage_num, stage_dir in stage_dirs:
        stage_dir = Path(stage_dir)
        diag_log = stage_dir / "diagnostics.npz"
        if not diag_log.exists():
            continue
        diag = np.load(diag_log)
        ts = diag.get("timesteps")
        if ts is None or len(ts) == 0:
            continue

        label_base = f"Stage {stage_num}: {stage_display_name(stage_num, stage_configs[stage_num]['name'])}"

        has_r = _has_finite(diag, "r_foot_contact")
        has_l = _has_finite(diag, "l_foot_contact")
        has_rr = _has_finite(diag, "rr_foot_contact")
        has_rl = _has_finite(diag, "rl_foot_contact")

        if has_r and has_l and has_rr and has_rl:
            # Quadrupedal: 4 individual feet
            fr = diag["r_foot_contact"]
            fl = diag["l_foot_contact"]
            rr = diag["rr_foot_contact"]
            rl = diag["rl_foot_contact"]

            ax_feet.plot(ts, _smooth(fr), label=f"{label_base} \u2013 FR", color="tab:blue", alpha=0.8)
            ax_feet.plot(ts, _smooth(fl), label=f"{label_base} \u2013 FL", color="tab:orange", alpha=0.8)
            ax_feet.plot(ts, _smooth(rr), label=f"{label_base} \u2013 RR", color="tab:green", alpha=0.8)
            ax_feet.plot(ts, _smooth(rl), label=f"{label_base} \u2013 RL", color="tab:red", alpha=0.8)

            # Diagonal pair composites
            if ax_diag is not None:
                diag_a = np.maximum(fr, rl)  # FR + RL
                diag_b = np.maximum(fl, rr)  # FL + RR
                ax_diag.plot(
                    ts,
                    _smooth(diag_a),
                    label=f"{label_base} \u2013 Diag A (FR+RL)",
                    color="tab:blue",
                    linewidth=1.5,
                )
                ax_diag.plot(
                    ts,
                    _smooth(diag_b),
                    label=f"{label_base} \u2013 Diag B (FL+RR)",
                    color="tab:orange",
                    linewidth=1.5,
                )
        elif has_r and has_l:
            # Bipedal: 2 feet
            ax_feet.plot(
                ts,
                _smooth(diag["r_foot_contact"]),
                label=f"{label_base} \u2013 Right",
                color="tab:blue",
                alpha=0.8,
            )
            ax_feet.plot(
                ts,
                _smooth(diag["l_foot_contact"]),
                label=f"{label_base} \u2013 Left",
                color="tab:orange",
                alpha=0.8,
            )

    ax_feet.set_xlabel("Timesteps")
    ax_feet.set_ylabel("Mean Contact Force")
    ax_feet.set_title(f"{model_title} {algorithm} \u2013 Per-Foot Contact Force")
    _safe_legend(ax_feet, fontsize=8)
    ax_feet.grid(True, alpha=0.3)

    if ax_diag is not None:
        ax_diag.set_xlabel("Timesteps")
        ax_diag.set_ylabel("Mean Diagonal Pair Contact")
        ax_diag.set_title(f"{model_title} {algorithm} \u2013 Diagonal Pair Contact (walk/trot pattern)")
        _safe_legend(ax_diag, fontsize=8)
        ax_diag.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Foot contact plot saved to: %s", save_path)
    if not show:
        plt.close(fig)

    return fig


def plot_stance_diagnostics(
    stage_dirs: StageDirs,
    stage_configs: dict[int | str, dict[str, Any]],
    model_id: ModelId,
    algorithm: str,
    save_path: "str | Path | None" = None,
    show: bool = True,
) -> "Any":
    """Plot reporting-only Stage-1 stance measurements.

    Returns ``None`` for every model but :data:`STANCE_DIAGNOSTICS_MODEL_ID`
    and when no stance fields have been recorded, so older runs and other
    models retain their existing artifact set.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from environments.shared.stance_diagnostics import STANCE_INFO_KEYS

    if model_id != STANCE_DIAGNOSTICS_MODEL_ID:
        return None

    panels: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "Foot support duty",
            (
                "r_foot_contact_duty",
                "l_foot_contact_duty",
                "bilateral_support_duty",
                "single_support_duty",
                "unsupported_duty",
            ),
        ),
        (
            "Foot load distribution",
            (
                "r_foot_load_share",
                "l_foot_load_share",
                "foot_load_balance",
                "foot_load_imbalance",
            ),
        ),
        ("Leg home pose", ("leg_home_pose_error",)),
        (
            "Head / neck posture",
            ("head_tip_z", "head_pelvis_rel_z", "neck_posture_error"),
        ),
        (
            "Pelvis pose",
            ("pelvis_height", "tilt_angle", "height_error"),
        ),
        (
            "Drift / yaw",
            ("drift_distance", "pelvis_yaw_vel", "pelvis_angular_vel"),
        ),
    )

    available = False
    for _, stage_dir in stage_dirs:
        diag_path = Path(stage_dir) / "diagnostics.npz"
        if diag_path.exists():
            with np.load(diag_path) as diag:
                if any(
                    key in diag and np.asarray(diag[key]).size > 0 and np.isfinite(diag[key]).any()
                    for key in STANCE_INFO_KEYS
                ):
                    available = True
                    break
    if not available:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{model_display_name(model_id)} {algorithm} – Stance Diagnostics",
        fontsize=14,
        fontweight="bold",
    )

    for stage_num, stage_dir in stage_dirs:
        diag_path = Path(stage_dir) / "diagnostics.npz"
        if not diag_path.exists():
            continue
        with np.load(diag_path) as diag:
            if "timesteps" not in diag or len(diag["timesteps"]) == 0:
                continue
            timesteps = diag["timesteps"]
            stage_name = stage_display_name(stage_num, stage_configs.get(stage_num, {}).get("name"))
            for axis, (title, keys) in zip(axes.flat, panels, strict=True):
                for key in keys:
                    if key not in diag:
                        continue
                    axis.plot(
                        timesteps,
                        _smooth(diag[key]),
                        label=f"S{stage_num} {key.replace('_', ' ')}",
                    )
                axis.set_title(title)
                axis.set_xlabel("Timesteps")
                axis.grid(True, alpha=0.3)
                _safe_legend(axis, fontsize=7)
            fig.text(0.99, 0.01, f"S{stage_num}: {stage_name}", ha="right", fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Stance diagnostics plot saved to: %s", save_path)
    if not show:
        plt.close(fig)
    return fig


#: ``plot_gate_progress``'s panels, in order: a group is drawn when ``gate_progress.npz`` holds any of its keys.
GATE_PROGRESS_PANELS: tuple[str, ...] = ("reward", "cop", "over_bar", "screen", "balance", "leg_dc")
#: The advisories' over-bar thresholds drawn on the over-bar panel: PostureDrift's and PostureTrend's defaults.
_OVER_BAR_LINES: tuple[tuple[float, str], ...] = ((0.5, "PostureDrift 0.5"), (0.1, "PostureTrend 0.1"))


def _gate_progress_groups(keys: Sequence[str]) -> dict[str, list[str]]:
    """The ``gate_progress.npz`` keys of each of :data:`GATE_PROGRESS_PANELS`' groups (only those present)."""
    groups: dict[str, list[str]] = {
        "reward": [key for key in ("mean_reward", "full_horizon_fraction") if key in keys],
        "cop": [
            key
            for key in keys
            if key.endswith(("_sole_cop_fore_aft", "_foot_cop_fore_aft")) and not key.startswith("max_")
        ],
        "over_bar": [key for key in keys if key.endswith("_over_bar_fraction")],
        "screen": [key for key in ("screen_v2_clean_lcb",) if key in keys],
        "balance": [key for key in ("com_fore_aft_mean_m", "xcom_toe_margin_min_m") if key in keys],
        "leg_dc": [key for key in keys if key.startswith(("leg_dc_", "leg_q_"))],
    }
    if "mean_reward" not in groups["reward"]:
        groups["reward"] = []
    return {name: groups[name] for name in GATE_PROGRESS_PANELS if groups[name]}


def _records_a_live_command(stage_dir: "str | Path") -> bool:
    """Whether *stage_dir*'s recorded ``stage_config.json`` task has a command section (a live-command stage)."""
    from .config import read_recorded_stage_config

    record = read_recorded_stage_config(stage_dir) or {}
    fingerprint = record.get("task_fingerprint")
    return isinstance(fingerprint, dict) and isinstance(fingerprint.get("command"), dict)


def plot_gate_progress(
    stage_dir: "str | Path",
    *,
    curriculum_kwargs: "dict[str, Any] | None" = None,
    model_id: ModelId | None = None,
    save_path: "str | Path | None" = None,
    show: bool = False,
    live_command: bool | None = None,
) -> Any:
    """One figure of a stage's ``gate_progress.npz``: a panel per group it holds, against millions of steps.

    1. ``mean_reward``, with the run's zero-action (statue) reward when its
       ``zero_action_baseline.json`` covers the model and the stage runs no
       live command, and ``full_horizon_fraction`` on a twin axis (that
       baseline is the stance's statue, so on a live-command stage, whose
       *live_command* is true or, when ``None``, whose recorded
       ``stage_config.json`` task has a command section, it would compare
       the walk with another task; decision D-D62);
    2. the signed per-foot pad and whole-foot centres of pressure
       (``<label>_sole_cop_fore_aft``, ``<label>_foot_cop_fore_aft``), with
       each declared bar at plus and minus;
    3. each rail's over-bar fraction, with PostureDrift's 0.5 and
       PostureTrend's 0.1 dashed;
    4. the in-training v2 screen's ``screen_v2_clean_lcb`` against the
       stage's ``min_clean_stance_lcb``;
    5. ``com_fore_aft_mean_m`` and ``xcom_toe_margin_min_m``, with 0;
    6. the leg DC, left-minus-right and mirror-even, commanded
       (``leg_dc_``, solid) and achieved (``leg_q_``, dashed).

    Returns the figure, or ``None`` when there is no file or no row.
    Report-only; reads the file and writes *save_path* only.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(stage_dir) / "gate_progress.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        series = {key: np.asarray(data[key], dtype=float) for key in data.files}
    timesteps = series.get("timesteps")
    if timesteps is None or not timesteps.size:
        return None
    steps = timesteps / 1e6
    rows = {key: values for key, values in series.items() if values.shape == timesteps.shape and key != "timesteps"}
    groups = _gate_progress_groups(tuple(rows))
    if not groups:
        return None
    curriculum = dict(curriculum_kwargs or {})

    def bar(key: str) -> float | None:
        value = curriculum.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            return None
        return float(value)

    figure, axes = plt.subplots(len(groups), 1, figsize=(12, 3.2 * len(groups)), sharex=True, squeeze=False)
    title = model_display_name(model_id) if model_id is not None else Path(stage_dir).name
    figure.suptitle(f"{title} – gate progress ({Path(stage_dir).name})", fontsize=13, fontweight="bold")
    for axis, (name, keys) in zip(axes[:, 0], groups.items()):
        if name == "reward":
            axis.plot(steps, rows["mean_reward"], label="mean reward", color="C0")
            statue = None
            if live_command is None:
                live_command = _records_a_live_command(stage_dir)
            if model_id is not None and not live_command:
                from .curriculum.baseline_watch import read_zero_action_baseline

                statue = read_zero_action_baseline(Path(stage_dir).parent, model_id)
            if statue is not None:
                axis.axhline(statue, color="0.4", linestyle=":", label=f"zero action {statue:.0f}")
            axis.set_ylabel("reward")
            if "full_horizon_fraction" in rows:
                twin = axis.twinx()
                twin.plot(steps, rows["full_horizon_fraction"], color="C1", alpha=0.7, label="full horizon")
                twin.set_ylim(-0.05, 1.05)
                twin.set_ylabel("full-horizon fraction")
                _safe_legend(twin, loc="lower right", fontsize=8)
        elif name == "cop":
            for key in keys:
                style = "-" if key.endswith("_sole_cop_fore_aft") else "--"
                axis.plot(steps, rows[key], style, label=key)
            for key, color in (("max_sole_cop_fore_aft", "C3"), ("max_foot_cop_fore_aft", "C4")):
                limit = bar(key)
                if limit is not None:
                    for sign in (1.0, -1.0):
                        axis.axhline(sign * limit, color=color, linestyle=":", label=f"{key} bar" if sign > 0 else None)
            axis.axhline(0.0, color="0.7", linewidth=0.8)
            axis.set_ylabel("signed CoP (half-lengths)")
        elif name == "over_bar":
            for key in keys:
                axis.plot(steps, rows[key], label=key)
            for level, label in _OVER_BAR_LINES:
                axis.axhline(level, color="0.5", linestyle="--", linewidth=0.8, label=label)
            axis.set_ylim(-0.05, 1.05)
            axis.set_ylabel("over-bar fraction")
        elif name == "screen":
            axis.plot(steps, rows["screen_v2_clean_lcb"], label="screen clean LCB (yaw 0, no statue criteria)")
            limit = bar("min_clean_stance_lcb")
            if limit is not None:
                axis.axhline(limit, color="C3", linestyle=":", label=f"min_clean_stance_lcb {limit:.2f}")
            axis.set_ylim(-0.05, 1.05)
            axis.set_ylabel("clean LCB")
        elif name == "balance":
            for key in keys:
                axis.plot(steps, rows[key], label=key)
            axis.axhline(0.0, color="0.5", linewidth=0.8)
            axis.set_ylabel("m")
        else:
            for key in keys:
                axis.plot(steps, rows[key], "-" if key.startswith("leg_dc_") else "--", label=key)
            axis.axhline(0.0, color="0.7", linewidth=0.8)
            axis.set_ylabel("degrees")
        axis.grid(alpha=0.3)
        _safe_legend(axis, loc="upper left", fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel("steps (millions)")
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Gate progress figure saved to: %s", save_path)
    if not show:
        plt.close(figure)
    return figure


#: The ``command_tracking/v1`` figures ``generate_stage_artifacts`` draws into ``figures/`` (decision D-D61).
COMMAND_TRACKING_FIGURE = "command_tracking.png"
COMMAND_GATE_PROGRESS_FIGURE = "command_gate_progress.png"
#: The tracking fraction over training: ``diagnostics.npz``'s rollout mean of the env's in-tolerance flag.
_TRACKING_HISTORY_KEY = "tracking_in_tolerance"
#: Each profile's bar on the per-episode in-tolerance fraction (after the settle), drawn against that history.
_TRACKING_BAR_KEYS: dict[str, str] = {"walk": "min_cruise_tracking_fraction", "follow": "min_follow_tracking_fraction"}


def _mapping(value: Any) -> dict[str, Any]:
    """*value* when it is a JSON object, else an empty one (the figures read recorded files leniently)."""
    return value if isinstance(value, dict) else {}


def _true_runs(times: Any, flags: Any) -> list[tuple[float, float]]:
    """The ``(start, end)`` time spans of the consecutive steps whose flag is set (each step ends at its time)."""
    spans: list[tuple[float, float]] = []
    start = None
    previous = 0.0
    for time_s, flag in zip(times, flags):
        if flag and start is None:
            start = previous
        if not flag and start is not None:
            spans.append((start, previous))
            start = None
        previous = float(time_s)
    if start is not None:
        spans.append((start, previous))
    return spans


def _command_tracking_history(stage_dir: Path) -> "tuple[Any, Any] | None":
    """``(timesteps, in-tolerance fraction)`` from the stage's ``diagnostics.npz``, or ``None`` when not recorded."""
    import numpy as np

    path = stage_dir / "diagnostics.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        if _TRACKING_HISTORY_KEY not in data.files or "timesteps" not in data.files:
            return None
        timesteps = np.asarray(data["timesteps"], dtype=float)
        fraction = np.asarray(data[_TRACKING_HISTORY_KEY], dtype=float)
    if fraction.shape != timesteps.shape or not np.isfinite(fraction).any():
        return None
    return timesteps, fraction


def _recorded_cruise_ramp(stage_dir: "str | Path") -> "tuple[float, int] | None":
    """The cruise ramp ``(start_fraction, timesteps)`` *stage_dir*'s recorded task declares; ``None`` without one."""
    from .config import read_recorded_stage_config

    record = read_recorded_stage_config(stage_dir) or {}
    fingerprint = record.get("task_fingerprint")
    command = fingerprint.get("command") if isinstance(fingerprint, dict) else None
    config = command.get("config") if isinstance(command, dict) else None
    if not isinstance(config, dict):
        return None
    start, timesteps = config.get("cruise_ramp_start_fraction"), config.get("cruise_ramp_timesteps")
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return None
    if isinstance(timesteps, bool) or not isinstance(timesteps, int) or timesteps <= 0:
        return None
    return float(start), int(timesteps)


def _recorded_same_stage_resumes(stage_dir: "str | Path") -> list[tuple[str, int | None]]:
    """The checkpoints *stage_dir*'s recorded run resumed its own stage from, oldest first.

    Each is its file name and, for a load that continued the run, the step
    count it resumed at; ``None`` for a load that restarted the step counter.
    ``save_stage_config`` records a root's resume as ``run.load_mode``
    ``"resume_same_stage"`` with its ``load_path``, and the resume of a stage
    that entered from its parent under ``run.resume_load_path``
    (:data:`~environments.shared.config.RESUME_LINEAGE_KEYS`);
    ``record_resume_session`` lists every session that continued the record
    under ``run.resume_sessions``.  As ``train_base._is_resume_continuation``
    reads a load, only a PPO load of one of the stage's own periodic
    checkpoints (``<label>_<steps>_steps.zip``,
    ``policy_loading._PERIODIC_CHECKPOINT_RE``) continues the run: a SAC load
    (the recorded ``algorithm``; nothing persists its replay buffer) or a
    checkpoint whose name carries no step count (``best_model.zip``,
    ``<label>_final.zip``) restarts the counter and seeds nothing.  A record
    that names no algorithm or stage is not held to them.
    """
    from .config import RESUME_SESSIONS_KEY, read_recorded_stage_config
    from .policy_loading import _PERIODIC_CHECKPOINT_RE

    record = read_recorded_stage_config(stage_dir) or {}
    run = record.get("run")
    if not isinstance(run, dict):
        return []
    algorithm = record.get("algorithm")
    restarts = isinstance(algorithm, str) and algorithm.upper() != "PPO"
    try:
        label = stage_label(record["stage"]) if "stage" in record else None
    except StageManifestError:
        label = None
    paths = [run.get("load_path")] if run.get("load_mode") == "resume_same_stage" else []
    sessions = run.get(RESUME_SESSIONS_KEY)
    if isinstance(sessions, list):
        paths.extend(session.get("resume_load_path") for session in sessions if isinstance(session, dict))
    # The latest session's checkpoint; already listed when record_resume_session recorded that session.
    paths.append(run.get("resume_load_path"))
    resumes: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for path in paths:
        if not isinstance(path, str) or not path or path in seen:
            continue
        seen.add(path)
        name = Path(path).name
        stem = name[: -len(".zip")] if name.endswith(".zip") else name
        match = _PERIODIC_CHECKPOINT_RE.match(stem)
        continues = match is not None and not restarts and label in (None, match.group(1))
        resumes.append((name, int(match.group(2)) if continues and match is not None else None))
    return resumes


def plot_command_tracking(
    stage_dir: "str | Path",
    *,
    curriculum_kwargs: "dict[str, Any] | None" = None,
    model_id: ModelId | None = None,
    save_path: "str | Path | None" = None,
    show: bool = False,
) -> Any:
    """A ``command_tracking/v1`` stage's tracking: commanded against achieved over panel episodes, and over training.

    One column per chart episode of ``command_tracking_trace.csv``
    (``reporting.command_tracking_trace``: the panel's first episode whose
    reset drew a cruise command and its first that drew a stand), each with

    1. the commanded speed against the achieved speed along the requested
       heading (and the planar speed), with the velocity tolerance (the
       stop-speed tolerance on a stand) and the start of the window the
       gate averages over (2 s for the speed ratio, 1 s for the hold);
    2. the commanded against the achieved yaw rate, with the yaw-rate
       tolerance (the stop yaw-rate tolerance on a stand);
    3. the heading error, with the heading tolerance and the first event's
       settle deadline (1 s + |initial error| / yaw_rate_max); on a stand,
       the distance from the spawn instead;

    and the steps the gate's per-step test holds shaded.  The tolerances
    and each episode's verdict are the command-tracking report's; a title
    says whether the re-rolled episode's record equals the certified one.
    Only the trace rows rolled on the report's handoff checkpoint are drawn
    (``command_tracking_trace.trace_rows_of``): a trace of another pair, or
    a trace without a report, is left out.
    The bottom panel is the tracking fraction over training, the training
    rollouts' mean of the env's in-tolerance flag (``diagnostics.npz``'s
    ``tracking_in_tolerance``, all steps of a stochastic policy, both
    families), against the profile's per-episode tracking bar; "not
    recorded" when the file holds none.  The training rollouts run the
    cruise ramp's commanded speeds (decision D-D62), not the end state the
    panel and the bar are judged on, so when the recorded
    ``stage_config.json`` task declares a ramp its window is shaded: there a
    standing policy reads partly in tolerance at the ramp's slow commands,
    and a walker's fraction can fall as the commands rise.  The shading
    stops at the history's last rollout, so the axis follows the data on a
    run shorter than its ramp, and the label then says the ramp continues
    past the run.

    Returns the figure, or ``None`` when the stage has neither a trace nor a
    tracking history.  Report-only; reads the stage directory and writes
    *save_path* only.
    """
    import math
    import textwrap

    import matplotlib.pyplot as plt
    import numpy as np

    from .curriculum.command_tracking_evidence import COMMAND_TRACKING_REPORT_JSON
    from .curriculum.command_tracking_gate import (
        SETTLE_BASE_S,
        SPEED_SETTLE_S,
        STAND_FAMILY,
        STAND_SETTLE_S,
    )
    from .file_io import read_json_object
    from .reporting.command_tracking_trace import (
        COMMAND_TRACKING_TRACE_CSV,
        read_command_tracking_trace,
        trace_episodes,
        trace_rows_of,
    )

    stage_dir = Path(stage_dir)
    report = read_json_object(stage_dir / COMMAND_TRACKING_REPORT_JSON) or {}
    trace = read_command_tracking_trace(stage_dir / COMMAND_TRACKING_TRACE_CSV) or []
    current = trace_rows_of(trace, report)
    if len(current) != len(trace):
        logger.warning(
            "%s: %d of %d trace rows were not rolled on the report's handoff checkpoint; they are not drawn",
            stage_dir / COMMAND_TRACKING_TRACE_CSV,
            len(trace) - len(current),
            len(trace),
        )
    episodes = trace_episodes(current)
    history = _command_tracking_history(stage_dir)
    if not episodes and history is None:
        return None
    measurement = _mapping(report.get("measurement"))
    tolerances = _mapping(measurement.get("tolerances"))
    yaw_rate_max = measurement.get("yaw_rate_max")
    curriculum = dict(report.get("thresholds") or {})
    curriculum.update(curriculum_kwargs or {})
    evidence = {
        int(entry["record"]["seed"]): entry
        for entry in report.get("episode_evidence") or ()
        if isinstance(entry, dict)
        and isinstance(entry.get("record"), dict)
        and isinstance(entry["record"].get("seed"), int)
    }

    def tolerance(name: str) -> float | None:
        value = tolerances.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        return float(value)

    def column(rows: Sequence[Any], key: str) -> Any:
        return np.asarray([float(row[key]) for row in rows], dtype=float)

    columns = max(1, len(episodes))
    panel_rows = 3 if episodes else 0
    figure = plt.figure(figsize=(7.0 * columns, 2.6 * panel_rows + 3.0))
    grid = figure.add_gridspec(panel_rows + 1, columns)
    title = model_display_name(model_id) if model_id is not None else stage_dir.name
    figure.suptitle(f"{title} – command tracking ({stage_dir.name})", fontsize=13, fontweight="bold")
    for index, (seed, family, rows) in enumerate(episodes):
        times = column(rows, "time_s")
        commanded = column(rows, "desired_speed")
        flags = [bool(row["in_tolerance"]) for row in rows]
        stand = family == STAND_FAMILY
        axes = [figure.add_subplot(grid[row, index]) for row in range(3)]
        speed_axis, yaw_axis, third_axis = axes
        speed_tolerance = tolerance("stop_speed_tolerance" if stand else "velocity_tolerance")
        if speed_tolerance is not None:
            low = np.zeros_like(commanded) if stand else commanded - speed_tolerance
            speed_axis.fill_between(
                times, low, commanded + speed_tolerance, color="0.88", step="post", linewidth=0, label="tolerance"
            )
        speed_axis.plot(times, commanded, color="0.2", linestyle="--", drawstyle="steps-post", label="commanded speed")
        speed_axis.plot(times, column(rows, "speed_along_heading_mps"), color="C0", label="achieved, along the heading")
        speed_axis.plot(
            times, column(rows, "actual_speed"), color="C0", linewidth=0.8, alpha=0.45, label="achieved, planar"
        )
        window = STAND_SETTLE_S if stand else SPEED_SETTLE_S
        speed_axis.axvline(window, color="0.5", linestyle=":", label=f"judged from {window:g} s")
        speed_axis.set_ylabel("speed (m/s)")

        commanded_yaw = column(rows, "command_yaw_rate")
        yaw_tolerance = tolerance("stop_yaw_rate_tolerance" if stand else "yaw_rate_tolerance")
        if yaw_tolerance is not None:
            centre = np.zeros_like(commanded_yaw) if stand else commanded_yaw
            yaw_axis.fill_between(
                times,
                centre - yaw_tolerance,
                centre + yaw_tolerance,
                color="0.88",
                step="post",
                linewidth=0,
                label="tolerance",
            )
        yaw_axis.plot(times, commanded_yaw, color="0.2", linestyle="--", drawstyle="steps-post", label="commanded")
        yaw_axis.plot(times, column(rows, "actual_yaw_rate"), color="C1", label="achieved")
        yaw_axis.set_ylabel("yaw rate (rad/s)")

        entry = evidence.get(seed) or {}
        record = _mapping(entry.get("record"))
        if stand:
            third_axis.plot(times, column(rows, "course_progress_m"), color="C4", label="distance from the spawn")
            third_axis.set_ylabel("m")
        else:
            heading_tolerance = tolerance("heading_tolerance")
            if heading_tolerance is not None:
                bound = math.degrees(heading_tolerance)
                third_axis.axhspan(-bound, bound, color="0.88", linewidth=0, label="tolerance")
            third_axis.plot(times, np.degrees(column(rows, "heading_error_rad")), color="C4", label="heading error")
            initial = record.get("initial_heading_error_rad")
            if isinstance(initial, (int, float)) and isinstance(yaw_rate_max, (int, float)) and yaw_rate_max > 0:
                deadline = SETTLE_BASE_S + abs(float(initial)) / float(yaw_rate_max)
                third_axis.axvline(deadline, color="0.5", linestyle=":", label=f"settle deadline {deadline:.2f} s")
            third_axis.axhline(0.0, color="0.7", linewidth=0.8)
            third_axis.set_ylabel("heading error (deg)")
        third_axis.set_xlabel("time (s)")

        if "success" in entry:
            failed = sorted({str(reason).split(":", 1)[0] for reason in entry.get("reasons") or ()})
            shown = ", ".join(failed[:4]) + (f" and {len(failed) - 4} more" if len(failed) > 4 else "")
            verdict = "passed" if entry["success"] else f"failed: {shown}"
        else:
            verdict = "not in the report"
        matches = rows[0]["record_matches_report"] if rows else False
        provenance = "the certified episode" if matches else "re-rolled; its record is not the report's"
        speed_axis.set_title(
            f"{family} episode, seed {seed}, commanded {commanded[0]:.3g} m/s ({provenance})\n"
            + textwrap.fill(verdict, width=90),
            fontsize=9,
        )
        for axis in axes:
            # Over the tolerance bands, so the steps the gate's per-step test holds stay visible.
            for number, (start, end) in enumerate(_true_runs(times, flags)):
                axis.axvspan(
                    start, end, color="C2", alpha=0.15, linewidth=0, label="in tolerance" if number == 0 else None
                )
            axis.grid(alpha=0.3)
            _safe_legend(axis, loc="upper right", fontsize=7)

    training = figure.add_subplot(grid[panel_rows, :])
    if history is not None:
        timesteps, fraction = history
        training.plot(timesteps / 1e6, fraction, color="C0", label="in-tolerance fraction of training steps")
        recorded = timesteps[np.isfinite(timesteps)]
        if recorded.size:
            # The marker says where the recorded series starts (the first rollout's step).  A PPO continuation
            # continues diagnostics.npz from the prior session's file (DiagnosticsCallback.seed_prior_diagnostics),
            # so a continued stage's series normally starts where the run did; one that starts after a checkpoint
            # the stage continued from holds no row from before it (the prior file was missing or unreadable, or
            # the resume predates the continuation), and says so.  Any other same-stage resume (a SAC load, a
            # checkpoint without a step count) restarted the step counter and seeded nothing, wherever its
            # series starts: the steps before its checkpoint are not in this record, which says so first.
            first = float(recorded.min())
            label = f"training history recorded from step {first:,.0f}"
            resumes = _recorded_same_stage_resumes(stage_dir)
            restarted = [name for name, steps in resumes if steps is None]
            resumed = [steps for _, steps in resumes if steps is not None and steps < first]
            if restarted:
                label += (
                    f" (a resume from {restarted[-1]} restarted the step counter: the steps before that "
                    "checkpoint are not in this record)"
                )
            elif resumed:
                label += (
                    f" (after the step-{min(resumed):,} checkpoint this stage resumed from: the rows before it "
                    "were not continued)"
                )
            training.axvline(first / 1e6, color="C0", linestyle=":", linewidth=0.9, label=label)
        bar_key = _TRACKING_BAR_KEYS.get(str(report.get("profile") or curriculum.get("profile")))
        bar_value = curriculum.get(bar_key) if bar_key else None
        if isinstance(bar_value, (int, float)) and not isinstance(bar_value, bool):
            training.axhline(
                float(bar_value),
                color="C3",
                linestyle="--",
                label=f"{bar_key} {float(bar_value):.2f} (panel, after the settle)",
            )
        ramp = _recorded_cruise_ramp(stage_dir)
        title = "Tracking fraction over training (training rollouts, the env's tolerances)"
        if ramp is not None:
            start_fraction, ramp_timesteps = ramp
            label = f"cruise ramp: commanded speeds {start_fraction:g} to 1 x the end state"
            # Clipped to the history's last rollout, so a run shorter than its ramp (a quick test) keeps the axis
            # its data spans; the label says the ramp goes on past it.
            last = float(recorded.max()) if recorded.size else float(ramp_timesteps)
            if ramp_timesteps > last:
                length = f"{ramp_timesteps / 1e6:.1f}M" if ramp_timesteps >= 1_000_000 else f"{ramp_timesteps / 1e3:g}k"
                label += f" (ramp to {length} continues past the run)"
            training.axvspan(
                0.0,
                min(float(ramp_timesteps), last) / 1e6,
                color="0.6",
                alpha=0.15,
                linewidth=0,
                label=label,
            )
            title = (
                "Tracking fraction over training (training rollouts at the cruise ramp's commanded speeds, "
                "the env's tolerances)"
            )
        training.set_ylim(-0.05, 1.05)
        training.set_xlabel("steps (millions)")
        training.set_ylabel("in-tolerance fraction")
        training.set_title(title, fontsize=10)
        training.grid(alpha=0.3)
        _safe_legend(training, loc="lower right", fontsize=8)
    else:
        training.axis("off")
        training.text(
            0.5,
            0.5,
            "Tracking fraction over training: not recorded (diagnostics.npz holds no tracking_in_tolerance)",
            ha="center",
            va="center",
            fontsize=10,
            color="0.3",
        )
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Command tracking figure saved to: %s", save_path)
    if not show:
        plt.close(figure)
    return figure


#: The gated statistics of each profile the gate-progress view draws: ``(family, statistic, bar key, label)``.
_COMMAND_GATE_STATISTICS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "walk": (
        ("cruise", "survival_lcb", "min_cruise_survival_lcb", "cruise survival LCB95"),
        ("cruise", "success_lcb", "min_cruise_success_lcb", "cruise success LCB95"),
        ("stand", "success_fraction", "min_stand_success_fraction", "stand success fraction"),
    ),
    "follow": (
        ("follow", "survival_lcb", "min_follow_survival_lcb", "follow survival LCB95"),
        ("follow", "success_lcb", "min_follow_success_lcb", "follow success LCB95"),
    ),
}


def _family_statistic(families: Any, family: str, statistic: str) -> float | None:
    """One family's statistic from a ``CommandTrackingResult.as_dict()`` families list, or ``None``."""
    import math

    for block in families or ():
        if isinstance(block, dict) and block.get("family") == family:
            value = block.get(statistic)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return None
            return float(value)
    return None


def plot_command_gate_progress(
    stage_dir: "str | Path",
    *,
    model_id: ModelId | None = None,
    verdict: bool | None = None,
    save_path: "str | Path | None" = None,
    show: bool = False,
) -> Any:
    """A ``command_tracking/v1`` stage's gate-progress view: its gated statistics at each checkpoint against the bars.

    One panel per gated statistic of the profile (walk: cruise survival and
    success LCB95, stand success fraction; follow: survival and success
    LCB95, and the paired difference against the reference when the gate
    declares it).  Each panel has

    * every handoff candidate the stage's choice ranked
      (``models/handoff_choice.json``: ``robust_best_model``,
      ``best_model``, the final pair) on the SELECTION panel, in the order
      training saved them (each zip's ``num_timesteps``), a filled marker
      for a candidate that passed and a ring for one that failed;
    * the certified handoff on the CERTIFICATION panel
      (``command_tracking_report.json``), a star;
    * the gate's bar, dashed.

    The last panel counts the certification panel's episodes failing each
    criterion, per family: which clause holds the stage back.

    The title states *verdict*, the stage's verdict (the judge's, which can
    refuse a report that records a pass); without one it says what the
    report records ("report records PASS"), never as the verdict.

    Returns the figure, or ``None`` when the stage has no report.
    Report-only; reads the stage directory and writes *save_path* only.
    """
    import math

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from .curriculum.checkpoints import HANDOFF_CHOICE_FILENAME
    from .curriculum.command_tracking_evidence import COMMAND_TRACKING_REPORT_JSON
    from .file_io import read_json_object
    from .task_fingerprint import read_checkpoint_attribute

    stage_dir = Path(stage_dir)
    report = read_json_object(stage_dir / COMMAND_TRACKING_REPORT_JSON)
    if report is None:
        return None
    profile = str(report.get("profile"))
    thresholds = _mapping(report.get("thresholds"))
    result = _mapping(report.get("result"))
    choice = read_json_object(stage_dir / "models" / HANDOFF_CHOICE_FILENAME) or {}

    candidates: list[tuple[str, int | None, dict[str, Any]]] = []
    for entry in choice.get("candidates") or ():
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        steps = read_checkpoint_attribute(stage_dir / "models" / f"{entry['name']}.zip", "num_timesteps")
        candidates.append((entry["name"], int(steps) if isinstance(steps, (int, float)) else None, entry))
    candidates.sort(key=lambda item: (item[1] is None, item[1] if item[1] is not None else 0))
    checkpoint = _mapping(report.get("handoff")).get("checkpoint")
    certified = (checkpoint if isinstance(checkpoint, str) else "").removesuffix(".zip") or "the handoff"

    statistics = list(_COMMAND_GATE_STATISTICS.get(profile, ()))
    paired_bar = thresholds.get("min_follow_reference_delta_lcb")
    if profile == "follow" and paired_bar is not None:
        statistics.append(("", "paired_delta_lcb", "min_follow_reference_delta_lcb", "paired success difference LCB95"))
    panels = len(statistics) + 1
    columns = 2
    rows = math.ceil(panels / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.8 * rows), squeeze=False)
    title = model_display_name(model_id) if model_id is not None else stage_dir.name
    if verdict is None:
        stated = f"report records {'PASS' if report.get('passed') is True else 'FAIL'}"
    else:
        stated = "PASS" if verdict else "FAIL"
    figure.suptitle(
        f"{title} – {report.get('scored_gate_kind')} {profile} gate progress: {stated} ({stage_dir.name})",
        fontsize=13,
        fontweight="bold",
    )
    labels = [name if steps is None else f"{name}\n{steps:,} steps" for name, steps, _ in candidates]
    names = [name for name, _, _ in candidates]
    certified_x = names.index(certified) if certified in names else len(names)
    if certified not in names:
        labels.append(certified)
    selection_start = choice.get("selection_seed_start")
    for axis, (family, statistic, bar_key, label) in zip(axes.flat, statistics):
        xs: list[int] = []
        ys: list[float] = []
        for x, (_name, _steps, entry) in enumerate(candidates):
            value = _family_statistic(entry.get("families"), family, statistic)
            if value is None:
                axis.annotate("n/a", (x, 0.0), ha="center", fontsize=8, color="0.4")
                continue
            xs.append(x)
            ys.append(value)
            axis.plot(
                [x],
                [value],
                marker="o",
                markersize=8,
                color="C0",
                markerfacecolor="C0" if entry.get("passed") else "white",
                linestyle="none",
                label=f"selection panel (seeds from {selection_start})" if len(xs) == 1 else None,
            )
        if len(xs) > 1 and all(steps is not None for _, steps, _ in candidates):
            axis.plot(xs, ys, color="C0", linewidth=1.0, alpha=0.5)
        if statistic == "paired_delta_lcb":
            value = result.get("paired_delta_lcb")
            value = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        else:
            value = _family_statistic(result.get("families"), family, statistic)
        if value is not None:
            axis.plot(
                [certified_x],
                [value],
                marker="*",
                markersize=14,
                color="C3",
                linestyle="none",
                label=f"certified, {certified} (seeds from {report.get('seed_start')})",
            )
        bar = thresholds.get(bar_key)
        if isinstance(bar, (int, float)) and not isinstance(bar, bool):
            axis.axhline(float(bar), color="0.3", linestyle="--", linewidth=1.0, label=f"{bar_key} {float(bar):.2f}")
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_xlim(-0.5, max(len(labels), 1) - 0.5)
        axis.set_ylim(-1.05 if statistic == "paired_delta_lcb" else -0.05, 1.05)
        axis.set_title(label, fontsize=10)
        axis.grid(alpha=0.3)
        _safe_legend(axis, loc="best", fontsize=7)

    failures_axis = axes.flat[len(statistics)]
    tallies = [
        (f"{block.get('family')}: {criterion}", int(count), int(block.get("n_episodes") or 0))
        for block in result.get("families") or ()
        if isinstance(block, dict)
        for criterion, count in sorted((block.get("criterion_failures") or {}).items())
    ]
    n_episodes = result.get("n_episodes")
    if tallies:
        positions = list(range(len(tallies)))
        failures_axis.barh(positions, [count for _, count, _ in tallies], color="C3", alpha=0.7, height=0.6)
        failures_axis.set_yticks(positions)
        failures_axis.set_yticklabels([name for name, _, _ in tallies], fontsize=8)
        failures_axis.invert_yaxis()
        for position, (_, count, total) in zip(positions, tallies):
            failures_axis.annotate(
                f"{count} / {total}",
                (count, position),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        failures_axis.set_xlim(0, max(count for _, count, _ in tallies) * 1.25 + 0.5)
        failures_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        failures_axis.set_xlabel("episodes failing the criterion")
        failures_axis.grid(axis="x", alpha=0.3)
    else:
        failures_axis.axis("off")
        failures_axis.text(0.5, 0.5, "every episode passed every criterion", ha="center", va="center", color="0.3")
    failures_axis.set_title(
        f"Episodes failing each criterion (certification panel, {n_episodes} episodes)", fontsize=10
    )
    for axis in list(axes.flat)[panels:]:
        axis.axis("off")
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Command gate progress figure saved to: %s", save_path)
    if not show:
        plt.close(figure)
    return figure
