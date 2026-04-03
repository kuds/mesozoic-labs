"""Visualization utilities for JAX/MJX training results.

Provides reusable plotting functions for training curves, locomotion
diagnostics, foot contact analysis, and video recording. Used by the
JAX training notebook and can be imported by any analysis script.

Usage::

    from environments.shared.jax_viz import (
        plot_training_curves,
        plot_locomotion_diagnostics,
        plot_reward_components,
        record_training_video,
        extract_video_frames,
        create_frame_collage,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _smooth(values: list | np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling-mean smoothing filter."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    hw = window // 2
    smoothed[:hw] = values[:hw]
    smoothed[-hw:] = values[-hw:]
    return smoothed


def plot_training_curves(
    reward_history: list[float],
    loss_history: list[float],
    episode_return_history: list[float],
    diagnostics_history: list[dict[str, Any]],
    *,
    species: str = "",
    stage: int = 1,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Any:
    """Plot 3x3 grid of training curves.

    Args:
        reward_history: Per-update mean reward values.
        loss_history: Per-update total loss values.
        episode_return_history: Per-update episode returns (may contain NaN).
        diagnostics_history: Per-update dicts with keys: ``grad_norm``,
            ``policy_loss``, ``value_loss``, ``entropy``, ``approx_kl``,
            ``fall_rate``, ``episode_length``, ``clip_fraction``, ``mean_std``.
        species: Species name for the plot title.
        stage: Curriculum stage number for the plot title.
        output_path: If provided, save the figure to this path.
        show: Whether to call ``plt.show()``.

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    window = min(20, len(reward_history) // 4 + 1)

    # Reward per step
    ax = axes[0, 0]
    ax.plot(reward_history, "b-", alpha=0.3, label="per-update")
    if window > 1 and len(reward_history) >= window:
        smoothed = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(reward_history)), smoothed, "b-", linewidth=2, label=f"{window}-update avg")
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Mean Reward/Step")
    ax.set_title("Reward per Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode return
    ax = axes[0, 1]
    valid_returns = [(i, r) for i, r in enumerate(episode_return_history) if not np.isnan(r)]
    if valid_returns:
        idxs, vals = zip(*valid_returns)
        ax.plot(idxs, vals, "g-", alpha=0.3, label="per-update")
        if window > 1 and len(vals) >= window:
            smoothed_ret = np.convolve(vals, np.ones(window) / window, mode="valid")
            ax.plot(
                range(idxs[0] + window - 1, idxs[0] + window - 1 + len(smoothed_ret)),
                smoothed_ret,
                "g-",
                linewidth=2,
                label=f"{window}-update avg",
            )
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Episode Return")
    ax.set_title("Episode Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Total loss
    ax = axes[0, 2]
    ax.plot(loss_history, "r-", alpha=0.5)
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 0]
    grad_norms = [d["grad_norm"] for d in diagnostics_history]
    ax.plot(grad_norms, "g-", alpha=0.5)
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm (pre-clip)")
    ax.grid(True, alpha=0.3)

    # Policy loss vs Value loss
    ax = axes[1, 1]
    pi_losses = [d["policy_loss"] for d in diagnostics_history]
    v_losses = [d["value_loss"] for d in diagnostics_history]
    ax.plot(pi_losses, "b-", alpha=0.5, label="Policy loss")
    ax.plot(v_losses, "r-", alpha=0.5, label="Value loss")
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Loss")
    ax.set_title("Policy vs Value Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy and KL divergence
    ax = axes[1, 2]
    entropies = [d["entropy"] for d in diagnostics_history]
    ax.plot(entropies, "purple", alpha=0.7, label="Entropy")
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax2 = ax.twinx()
    kls = [d["approx_kl"] for d in diagnostics_history]
    ax2.plot(kls, "orange", alpha=0.7, label="Approx KL")
    ax2.set_ylabel("Approx KL")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Fall rate
    ax = axes[2, 0]
    fall_rates = [d["fall_rate"] for d in diagnostics_history]
    ax.plot(fall_rates, "brown", alpha=0.5)
    if window > 1 and len(fall_rates) >= window:
        smoothed_falls = np.convolve(fall_rates, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(fall_rates)), smoothed_falls, "brown", linewidth=2)
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Fall Rate")
    ax.set_title("Episode Termination Rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Episode length
    ax = axes[2, 1]
    valid_lengths = [
        (i, d["episode_length"])
        for i, d in enumerate(diagnostics_history)
        if not np.isnan(d.get("episode_length", float("nan")))
    ]
    if valid_lengths:
        idxs_l, vals_l = zip(*valid_lengths)
        ax.plot(idxs_l, vals_l, "teal", alpha=0.5)
        if window > 1 and len(vals_l) >= window:
            smoothed_len = np.convolve(vals_l, np.ones(window) / window, mode="valid")
            ax.plot(
                range(idxs_l[0] + window - 1, idxs_l[0] + window - 1 + len(smoothed_len)),
                smoothed_len,
                "teal",
                linewidth=2,
            )
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Steps")
    ax.set_title("Mean Episode Length")
    ax.grid(True, alpha=0.3)

    # Clip fraction and mean std
    ax = axes[2, 2]
    clip_fracs = [d["clip_fraction"] for d in diagnostics_history]
    mean_stds = [d["mean_std"] for d in diagnostics_history]
    ax.plot(clip_fracs, "red", alpha=0.5, label="Clip fraction")
    ax.set_ylabel("Clip Fraction")
    ax.set_xlabel("PPO Update")
    ax3 = ax.twinx()
    ax3.plot(mean_stds, "blue", alpha=0.5, label="Mean std")
    ax3.set_ylabel("Mean Std")
    ax.legend(loc="upper left")
    ax3.legend(loc="upper right")
    ax.set_title("Clip Fraction & Policy Std")
    ax.grid(True, alpha=0.3)

    title = "JAX/MJX Training Diagnostics"
    if species:
        title = f"{species.title()} {title} (Stage {stage})"
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), dpi=150)
    if show:
        plt.show()

    return fig


def plot_locomotion_diagnostics(
    eval_results: Any,
    *,
    species: str = "",
    stage: int = 1,
    max_tilt_angle: float = 1.047,
    healthy_z_range: tuple[float, float] = (0.3, 2.0),
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Any:
    """Plot locomotion health diagnostics from evaluation results.

    Args:
        eval_results: ``EvalResults`` instance from ``jax_eval.evaluate_policy_cpu``.
        species: Species name for the plot title.
        stage: Curriculum stage number.
        max_tilt_angle: Maximum tilt angle (radians) for reference line.
        healthy_z_range: (min, max) healthy height for reference lines.
        output_dir: If provided, save plots to this directory.
        show: Whether to call ``plt.show()``.

    Returns:
        Tuple of (main_fig, foot_fig) where foot_fig may be None.
    """
    import matplotlib.pyplot as plt

    diag_tilt = eval_results.diag_tilt
    diag_fwd_vel = eval_results.diag_fwd_vel
    diag_pelvis_h = eval_results.diag_pelvis_h
    diag_l_foot = eval_results.diag_l_foot
    diag_r_foot = eval_results.diag_r_foot
    diag_energy = eval_results.diag_energy
    diag_reward_components = eval_results.diag_reward_components

    fig_diag, axes_d = plt.subplots(2, 3, figsize=(18, 10))
    _steps = np.arange(len(diag_tilt))

    # Tilt angle (degrees)
    ax = axes_d[0, 0]
    ax.plot(_steps, _smooth(np.degrees(diag_tilt)), "b-", alpha=0.8)
    ax.axhline(np.degrees(max_tilt_angle), color="red", linestyle="--", alpha=0.5, label="Max tilt")
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Tilt (degrees)")
    ax.set_title("Tilt Angle")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Forward velocity
    ax = axes_d[0, 1]
    ax.plot(_steps, _smooth(diag_fwd_vel), "g-", alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Forward Vel (m/s)")
    ax.set_title("Forward Velocity")
    ax.grid(True, alpha=0.3)

    # Pelvis height
    ax = axes_d[0, 2]
    ax.plot(_steps, _smooth(diag_pelvis_h), "teal", alpha=0.8)
    ax.axhline(healthy_z_range[0], color="red", linestyle="--", alpha=0.5, label="Z min")
    ax.axhline(healthy_z_range[1], color="red", linestyle="--", alpha=0.5, label="Z max")
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Height (m)")
    ax.set_title("Pelvis Height")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Foot contacts
    ax = axes_d[1, 0]
    _sym = None
    if diag_l_foot and diag_r_foot:
        _foot_steps = np.arange(len(diag_l_foot))
        ax.plot(_foot_steps, _smooth(diag_l_foot), "b-", alpha=0.6, label="Left foot")
        ax.plot(_foot_steps, _smooth(diag_r_foot), "r-", alpha=0.6, label="Right foot")
        _l = np.array(diag_l_foot)
        _r = np.array(diag_r_foot)
        _sym = 1.0 - np.abs(_l - _r) / (_l + _r + 1e-8)
        ax.plot(_foot_steps, _smooth(_sym), "purple", alpha=0.5, linestyle="--", label="Gait symmetry")
        ax.legend()
        ax.set_xlabel("Eval Step")
        ax.set_ylabel("Contact Force / Symmetry")
        ax.set_title("Foot Contacts & Gait Symmetry")
    else:
        ax.text(0.5, 0.5, "No foot contact data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title("Foot Contacts (N/A)")
    ax.grid(True, alpha=0.3)

    # Reward decomposition — only show components with non-zero signal
    ax = axes_d[1, 1]
    _comp_steps = np.arange(len(diag_reward_components.get("forward", diag_reward_components.get("alive", []))))
    for comp_name, comp_vals in diag_reward_components.items():
        arr = np.asarray(comp_vals)
        if len(arr) == 0 or np.allclose(arr, 0.0):
            continue
        ax.plot(_comp_steps[: len(arr)], _smooth(comp_vals), alpha=0.7, label=comp_name)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Reward Component")
    ax.set_title("Reward Decomposition")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cost of transport
    ax = axes_d[1, 2]
    _energy_arr = np.abs(np.array(diag_energy))
    _fwd_arr = np.maximum(np.array(diag_fwd_vel), 0.01)
    _cot = _energy_arr / _fwd_arr
    ax.plot(_steps, _smooth(_cot), "brown", alpha=0.8)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Cost of Transport")
    ax.set_title("Cost of Transport")
    ax.grid(True, alpha=0.3)

    title = "Locomotion Diagnostics"
    if species:
        title = f"{species.title()} Stage {stage} -- {title}"
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        diag_path = Path(output_dir) / "locomotion_health.png"
        plt.savefig(str(diag_path), dpi=150)
    if show:
        plt.show()

    # Foot contact detail plot
    fig_foot = None
    if diag_l_foot and diag_r_foot and _sym is not None:
        fig_foot, axes_f = plt.subplots(1, 3, figsize=(15, 4))
        _foot_steps = np.arange(len(diag_l_foot))

        ax = axes_f[0]
        ax.plot(_foot_steps, _smooth(diag_l_foot, window=20), "b-", alpha=0.7, label="Left")
        ax.plot(_foot_steps, _smooth(diag_r_foot, window=20), "r-", alpha=0.7, label="Right")
        ax.set_title("Raw Contact Forces")
        ax.set_xlabel("Eval Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes_f[1]
        _stride_proxy = (np.array(diag_l_foot) + np.array(diag_r_foot)) / 2.0
        ax.plot(_foot_steps, _smooth(_stride_proxy, window=20), "green", alpha=0.7)
        ax.set_title("Stride Frequency Proxy")
        ax.set_xlabel("Eval Step")
        ax.grid(True, alpha=0.3)

        ax = axes_f[2]
        ax.plot(_foot_steps, _smooth(_sym, window=20), "purple", alpha=0.7)
        ax.set_ylim(0, 1.1)
        ax.set_title("Gait Symmetry")
        ax.set_xlabel("Eval Step")
        ax.grid(True, alpha=0.3)

        foot_title = "Foot Contact Analysis"
        if species:
            foot_title = f"{species.title()} -- {foot_title}"
        plt.suptitle(foot_title, fontsize=13, fontweight="bold")
        plt.tight_layout()

        if output_dir:
            foot_path = Path(output_dir) / "foot_contacts.png"
            plt.savefig(str(foot_path), dpi=150)
        if show:
            plt.show()

    return fig_diag, fig_foot


def plot_reward_components(
    reward_component_history: list[dict[str, float]],
    *,
    species: str = "",
    stage: int = 1,
    healthy_z_min: float | None = None,
    natural_forward_z: float | None = None,
    nosedive_threshold: float | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Any:
    """Plot reward component diagnostics over training.

    Creates a 2x2 grid: per-component curves, stacked area chart,
    body state vs termination thresholds, and foot contact rate.

    Args:
        reward_component_history: List of per-update dicts with reward
            component values. Keys starting with ``_`` are treated as
            internal state diagnostics (``_pelvis_z``, ``_forward_z``,
            ``_has_foot_contact``). The ``update`` key holds the update
            number.
        species: Species name for the plot title.
        stage: Curriculum stage number.
        healthy_z_min: Minimum healthy pelvis height (for reference line).
        natural_forward_z: Natural forward-Z value (for nosedive line).
        nosedive_threshold: How far below *natural_forward_z* triggers
            termination (for reference line).
        output_path: If provided, save the figure to this path.
        show: Whether to call ``plt.show()``.

    Returns:
        The matplotlib Figure object, or ``None`` if no data.
    """
    if not reward_component_history:
        return None

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Reward Component Diagnostics — {species} Stage {stage}",
        fontsize=14,
    )

    updates = [d["update"] for d in reward_component_history]
    reward_keys = [
        k
        for k in reward_component_history[0]
        if not k.startswith("_") and k != "update"
    ]

    # --- Plot 1: Per-component reward curves ---
    ax = axes[0, 0]
    for key in reward_keys:
        vals = [d.get(key, 0.0) for d in reward_component_history]
        ax.plot(updates, vals, label=key, linewidth=1.5)
    ax.set_xlabel("Update")
    ax.set_ylabel("Mean reward component")
    ax.set_title("Per-component rewards")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Stacked area chart ---
    ax = axes[0, 1]
    pos_keys = [
        k
        for k in reward_keys
        if any(d.get(k, 0) > 0 for d in reward_component_history)
    ]
    neg_keys = [
        k
        for k in reward_keys
        if any(d.get(k, 0) < 0 for d in reward_component_history)
    ]
    for key in pos_keys:
        vals = [d.get(key, 0.0) for d in reward_component_history]
        ax.fill_between(updates, 0, vals, alpha=0.4, label=key)
    for key in neg_keys:
        vals = [d.get(key, 0.0) for d in reward_component_history]
        ax.fill_between(updates, 0, vals, alpha=0.4, label=key)
    ax.set_xlabel("Update")
    ax.set_ylabel("Reward")
    ax.set_title("Reward composition")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: State diagnostics (pelvis_z, forward_z) ---
    ax = axes[1, 0]
    if "_pelvis_z" in reward_component_history[0]:
        ax.plot(
            updates,
            [d["_pelvis_z"] for d in reward_component_history],
            label="pelvis_z",
            color="blue",
        )
        if healthy_z_min is not None:
            ax.axhline(
                y=healthy_z_min,
                color="blue",
                linestyle="--",
                alpha=0.5,
                label=f"z_min={healthy_z_min}",
            )
    if "_forward_z" in reward_component_history[0]:
        ax2 = ax.twinx()
        ax2.plot(
            updates,
            [d["_forward_z"] for d in reward_component_history],
            label="forward_z",
            color="red",
        )
        if natural_forward_z is not None and nosedive_threshold is not None:
            nd_thresh = natural_forward_z - nosedive_threshold
            ax2.axhline(
                y=nd_thresh,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"nosedive={nd_thresh:.2f}",
            )
        ax2.set_ylabel("forward_z", color="red")
        ax2.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("Update")
    ax.set_ylabel("pelvis_z", color="blue")
    ax.set_title("Body state vs termination thresholds")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Foot contact rate ---
    ax = axes[1, 1]
    if "_has_foot_contact" in reward_component_history[0]:
        fc_vals = [d["_has_foot_contact"] for d in reward_component_history]
        ax.plot(updates, fc_vals, label="foot contact rate", color="green", linewidth=2)
        ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Update")
    ax.set_ylabel("Foot contact rate")
    ax.set_title("Foot contact (alive bonus gate)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def record_training_video(
    mj_model: Any,
    params: Any,
    network: Any,
    obs_stats: Any,
    *,
    get_obs_fn: Any,
    normalize_obs_fn: Any,
    scale_action_fn: Any,
    reward_fn: Any | None = None,
    reward_cfg: dict[str, float] | None = None,
    max_episode_steps: int = 1000,
    frame_skip: int = 5,
    root_body_id: int = 1,
    healthy_z_range: tuple[float, float] = (0.3, 2.0),
    max_tilt_angle: float = 1.047,
    natural_forward_z: float = 0.0,
    nosedive_threshold: float = 0.5,
    termination_body_heights: dict[str, float] | None = None,
    termination_site_heights: dict[str, float] | None = None,
    success_sites: tuple[str, ...] = (),
    success_threshold: float = 0.3,
    target_body: str | None = None,
    sensor_quat_start: int = 6,
    output_path: str | Path | None = None,
    fps: int = 50,
    height: int = 480,
    width: int = 640,
    camera_track_body: str | None = None,
    camera_distance: float = 3.0,
    camera_azimuth: float = 135.0,
    camera_elevation: float = -20.0,
    show: bool = True,
) -> tuple[list, float]:
    """Record a video of the trained policy using CPU MuJoCo.

    Args:
        mj_model: MuJoCo model.
        params: JAX network parameters (deterministic: uses mean action).
        network: Flax ActorCritic network module.
        obs_stats: RunningMeanStd observation statistics.
        get_obs_fn: Function(mjx_data) -> obs array.
        normalize_obs_fn: Function(obs, obs_stats) -> normalized obs.
        scale_action_fn: Function(action) -> scaled ctrl.
        reward_fn: Optional function(mjx_data, action, reward_cfg) -> scalar.
        reward_cfg: Optional reward config dict (required if reward_fn provided).
        max_episode_steps: Maximum steps per episode.
        frame_skip: Number of physics steps per action.
        root_body_id: MuJoCo body ID of root body (for termination check).
        healthy_z_range: (min, max) height for termination.
        max_tilt_angle: Maximum tilt angle (radians) for termination.
        natural_forward_z: Natural forward-Z for the species (nosedive baseline).
        nosedive_threshold: How far below natural_forward_z triggers termination.
        termination_body_heights: Dict mapping body name -> z threshold for
            floor contact termination. Episode ends if any body drops below
            its threshold.
        termination_site_heights: Dict mapping site name -> z threshold for
            extremity termination. Episode ends if any site drops below
            its threshold (more precise than body checks for tips).
        sensor_quat_start: Index into sensordata where root quaternion starts.
        output_path: If provided, save video to this path (requires mediapy).
        fps: Frames per second for the video.
        height: Render height in pixels.
        width: Render width in pixels.
        camera_track_body: Name of MuJoCo body to track (e.g. "pelvis", "torso").
        camera_distance: Camera distance from tracked body.
        camera_azimuth: Camera azimuth angle in degrees.
        camera_elevation: Camera elevation angle in degrees.
        show: Whether to display the video inline (requires mediapy).

    Returns:
        (frames, episode_reward) tuple.
    """
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx

    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # Configure camera with tracking and zoom (matching SB3 render behaviour)
    camera = mujoco.MjvCamera()
    if camera_track_body is not None:
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, camera_track_body)
    camera.distance = camera_distance
    camera.azimuth = camera_azimuth
    camera.elevation = camera_elevation

    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    # Resolve body-height termination checks
    _body_checks: list[tuple[int, float]] = []
    if termination_body_heights:
        for bname, z_thresh in termination_body_heights.items():
            bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                _body_checks.append((bid, z_thresh))

    # Resolve site-height termination checks
    _site_checks: list[tuple[int, float]] = []
    if termination_site_heights:
        for sname, z_thresh in termination_site_heights.items():
            sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid >= 0:
                _site_checks.append((sid, z_thresh))

    # Resolve success site IDs and target (prey/food) body
    _success_site_ids: list[int] = []
    for sname in success_sites:
        sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, sname)
        if sid >= 0:
            _success_site_ids.append(sid)
    _target_body_id = -1
    if target_body is not None:
        _target_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, target_body)

    frames = []
    episode_reward = 0.0

    for step in range(max_episode_steps):
        cpu_data = mjx.put_data(mj_model, mj_data)
        obs_raw = get_obs_fn(cpu_data)
        obs = normalize_obs_fn(obs_raw, obs_stats)

        mean, _log_std, _value = network.apply(params, obs)
        action = jnp.clip(mean, -1.0, 1.0)

        ctrl = np.array(scale_action_fn(action))
        mj_data.ctrl[:] = ctrl
        for _ in range(frame_skip):
            mujoco.mj_step(mj_model, mj_data)

        renderer.update_scene(mj_data, camera)
        frames.append(renderer.render())

        if reward_fn is not None and reward_cfg is not None:
            cpu_data = mjx.put_data(mj_model, mj_data)
            r = float(reward_fn(cpu_data, action, reward_cfg))
            episode_reward += r

        # Height termination
        body_z = mj_data.xpos[root_body_id, 2]
        if body_z < healthy_z_range[0] or body_z > healthy_z_range[1]:
            break

        # Tilt termination (matches jax_eval and mjx_env logic)
        root_quat = mj_data.sensordata[sensor_quat_start : sensor_quat_start + 4]
        tilt = float(np.arccos(np.clip(1.0 - 2.0 * (root_quat[1] ** 2 + root_quat[2] ** 2), -1, 1)))
        if tilt > max_tilt_angle:
            break

        # Nosedive termination (excessive forward pitch)
        w, x, y, z = root_quat[0], root_quat[1], root_quat[2], root_quat[3]
        forward_z = float(2.0 * (x * z - w * y))
        if forward_z < natural_forward_z - nosedive_threshold:
            break

        # Body-height floor contact termination
        if any(mj_data.xpos[bid, 2] < zt for bid, zt in _body_checks):
            break

        # Site-height termination (extremities like snout tip)
        if any(mj_data.site_xpos[sid, 2] < zt for sid, zt in _site_checks):
            break

        # Stage 3 success: proximity-based contact detection
        if _success_site_ids and _target_body_id >= 0:
            target_pos = mj_data.xpos[_target_body_id]
            if any(
                float(np.linalg.norm(mj_data.site_xpos[sid] - target_pos)) < success_threshold
                for sid in _success_site_ids
            ):
                break

    renderer.close()

    if output_path:
        try:
            import mediapy

            mediapy.write_video(str(output_path), frames, fps=fps)
        except ImportError:
            pass

    if show:
        try:
            import mediapy

            mediapy.show_video(frames, fps=fps)
        except ImportError:
            pass

    return frames, episode_reward


def extract_video_frames(
    source: str | Path | list[np.ndarray],
    output_dir: str | Path,
    *,
    num_frames: int = 10,
    fmt: str = "png",
    prefix: str = "frame",
) -> list[Path]:
    """Extract evenly-spaced frames from a video file or frame list.

    Works in Google Colab and standard Python environments.  Accepts
    either a video file path (MP4, etc.) or a list of numpy RGB arrays
    (as returned by :func:`record_training_video`).

    Args:
        source: Path to a video file, or list of RGB numpy arrays.
        output_dir: Directory to save extracted frames.
        num_frames: Number of frames to extract (evenly spaced).
        fmt: Image format — ``"png"`` or ``"jpg"``.
        prefix: Filename prefix (files are ``{prefix}_001.png``, etc.).

    Returns:
        List of paths to the saved frame images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(source, (str, Path)):
        frames = _read_video_frames(Path(source))
    else:
        frames = source

    if not frames:
        return []

    # Select evenly-spaced frame indices
    n = len(frames)
    num_frames = min(num_frames, n)
    indices = [int(round(i * (n - 1) / (num_frames - 1))) for i in range(num_frames)] if num_frames > 1 else [0]

    saved: list[Path] = []
    for seq, idx in enumerate(indices, start=1):
        filename = f"{prefix}_{seq:03d}.{fmt}"
        path = output_dir / filename
        _save_frame(frames[idx], path, fmt)
        saved.append(path)

    return saved


def _read_video_frames(video_path: Path) -> list[np.ndarray]:
    """Read all frames from a video file using available backend."""
    # Try mediapy first (common in Colab), then imageio, then cv2
    try:
        import mediapy

        return list(mediapy.read_video(str(video_path)))
    except (ImportError, Exception):
        pass

    try:
        import imageio.v3 as iio

        return [np.array(f) for f in iio.imread(str(video_path), plugin="pyav")]
    except (ImportError, Exception):
        pass

    try:
        import cv2

        frames = []
        cap = cv2.VideoCapture(str(video_path))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    except ImportError:
        raise ImportError("No video backend available. Install one of: mediapy, imageio[pyav], or opencv-python")


def _save_frame(frame: np.ndarray, path: Path, fmt: str) -> None:
    """Save a single RGB numpy frame as an image file."""
    try:
        from PIL import Image

        Image.fromarray(frame).save(str(path))
        return
    except ImportError:
        pass

    try:
        import imageio.v3 as iio

        iio.imwrite(str(path), frame)
        return
    except ImportError:
        pass

    try:
        import cv2

        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return
    except ImportError:
        raise ImportError("No image backend available. Install one of: Pillow, imageio, or opencv-python")


def create_frame_collage(
    source: str | Path | list[np.ndarray],
    output_path: str | Path | None = None,
    *,
    num_frames: int = 10,
    cols: int = 5,
    frame_height: int = 360,
    title: str | None = None,
    fps: int = 50,
    show: bool = True,
) -> Any:
    """Create a collage of evenly-spaced frames with frame numbers.

    Works in Google Colab and standard Python environments.  Accepts
    either a video file path (MP4, etc.) or a list of numpy RGB arrays
    (as returned by :func:`record_training_video`).

    Args:
        source: Path to a video file, or list of RGB numpy arrays.
        output_path: If provided, save collage image to this path.
        num_frames: Number of frames to include in the collage.
        cols: Number of columns in the grid.
        frame_height: Height of each frame thumbnail in pixels.
        title: Optional super-title for the collage.
        fps: Frames per second (used to compute timestamps).
        show: Whether to call ``plt.show()``.

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    if isinstance(source, (str, Path)):
        frames = _read_video_frames(Path(source))
    else:
        frames = source

    if not frames:
        fig, ax = plt.subplots(1, 1)
        ax.text(0.5, 0.5, "No frames", ha="center", va="center")
        return fig

    n = len(frames)
    num_frames = min(num_frames, n)
    indices = [int(round(i * (n - 1) / (num_frames - 1))) for i in range(num_frames)] if num_frames > 1 else [0]

    rows = (num_frames + cols - 1) // cols

    # Scale figure size from frame aspect ratio
    sample = frames[0]
    aspect = sample.shape[1] / sample.shape[0]
    cell_w = 3.5 * aspect
    cell_h = 3.5
    fig, axes = plt.subplots(rows, cols, figsize=(cell_w * cols, cell_h * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, (ax_row, ax_col) in enumerate([(r, c) for r in range(rows) for c in range(cols)]):
        ax = axes[ax_row, ax_col]
        if i < len(indices):
            idx = indices[i]
            t = idx / fps
            ax.imshow(frames[idx])
            ax.set_title(f"Frame {idx}  ({t:.1f}s)", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= len(indices):
            ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
