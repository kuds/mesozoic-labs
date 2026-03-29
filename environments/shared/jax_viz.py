"""Visualization utilities for JAX/MJX training results.

Provides reusable plotting functions for training curves, locomotion
diagnostics, foot contact analysis, and video recording. Used by the
JAX training notebook and can be imported by any analysis script.

Usage::

    from environments.shared.jax_viz import (
        plot_training_curves,
        plot_locomotion_diagnostics,
        record_training_video,
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
        ax.plot(range(window - 1, len(reward_history)), smoothed, "b-", linewidth=2,
                label=f"{window}-update avg")
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
            ax.plot(range(idxs[0] + window - 1, idxs[0] + window - 1 + len(smoothed_ret)),
                    smoothed_ret, "g-", linewidth=2, label=f"{window}-update avg")
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
        (i, d["episode_length"]) for i, d in enumerate(diagnostics_history)
        if not np.isnan(d.get("episode_length", float("nan")))
    ]
    if valid_lengths:
        idxs_l, vals_l = zip(*valid_lengths)
        ax.plot(idxs_l, vals_l, "teal", alpha=0.5)
        if window > 1 and len(vals_l) >= window:
            smoothed_len = np.convolve(vals_l, np.ones(window) / window, mode="valid")
            ax.plot(range(idxs_l[0] + window - 1, idxs_l[0] + window - 1 + len(smoothed_len)),
                    smoothed_len, "teal", linewidth=2)
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

    # Reward decomposition
    ax = axes_d[1, 1]
    _comp_steps = np.arange(len(diag_reward_components["forward"]))
    for comp_name, comp_vals in diag_reward_components.items():
        ax.plot(_comp_steps, _smooth(comp_vals), alpha=0.7, label=comp_name)
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
    output_path: str | Path | None = None,
    fps: int = 50,
    height: int = 480,
    width: int = 640,
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
        output_path: If provided, save video to this path (requires mediapy).
        fps: Frames per second for the video.
        height: Render height in pixels.
        width: Render width in pixels.
        show: Whether to display the video inline (requires mediapy).

    Returns:
        (frames, episode_reward) tuple.
    """
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx

    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

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

        renderer.update_scene(mj_data)
        frames.append(renderer.render())

        if reward_fn is not None and reward_cfg is not None:
            cpu_data = mjx.put_data(mj_model, mj_data)
            r = float(reward_fn(cpu_data, action, reward_cfg))
            episode_reward += r

        body_z = mj_data.xpos[root_body_id, 2]
        if body_z < healthy_z_range[0] or body_z > healthy_z_range[1]:
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
