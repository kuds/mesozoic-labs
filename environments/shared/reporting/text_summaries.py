"""Human-readable stage and run summary text files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .formatting import format_duration


def write_stage_summary(
    stage_dir,
    results_dict: dict[str, Any],
    species: str,
    algorithm: str,
) -> Path:
    """Write a text summary for a single completed stage to its directory.

    Returns the path to the written summary file.
    """
    from ..config import get_library_version

    summary_path = Path(stage_dir) / "stage_summary.txt"
    mean_len = results_dict.get("mean_episode_length", 0)
    std_len = results_dict.get("std_episode_length", 0)
    sim_dt = results_dict.get("sim_dt", 0.01)
    avg_duration_s = mean_len * sim_dt
    mean_vel = results_dict.get("mean_forward_vel", 0.0)
    std_vel = results_dict.get("std_forward_vel", 0.0)
    lines = [
        f"Mesozoic Labs: Stage {results_dict['stage']} Summary",
        "=" * 50,
        "",
        f"Species:        {species.title()}",
        f"Stage:          {results_dict['stage']} ({results_dict['name']})",
        f"Description:    {results_dict['description']}",
        f"Algorithm:      {algorithm}",
        f"Version:        {get_library_version()}",
        f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timesteps:      {results_dict['timesteps']:,}",
        f"Duration:       {format_duration(results_dict['duration_seconds'])}",
        f"Final eval:     {results_dict['mean_reward']:.2f} +/- {results_dict['std_reward']:.2f}",
        f"Avg ep length:  {mean_len:.1f} +/- {std_len:.1f} steps ({avg_duration_s:.2f}s sim time)",
        f"Avg fwd vel:    {mean_vel:.2f} +/- {std_vel:.2f} m/s",
    ]
    best_r = results_dict.get("best_eval_reward", "")
    if best_r != "":
        best_s = results_dict.get("best_eval_std", "")
        best_ts = results_dict.get("best_eval_timestep", "")
        best_len = results_dict.get("best_eval_length", "")
        best_len_s = results_dict.get("best_eval_std_length", "")
        ts_label = f"  (at {best_ts:,} steps)" if isinstance(best_ts, int) else ""
        lines.append(f"Best eval:      {best_r} +/- {best_s}{ts_label}")
        if best_len != "":
            best_dur_s = best_len * sim_dt
            lines.append(f"Best ep length: {best_len} +/- {best_len_s} steps ({best_dur_s:.2f}s sim time)")
    bm_r = results_dict.get("best_model_reward", "")
    if bm_r != "":
        bm_s = results_dict.get("best_model_std_reward", "")
        bm_len = results_dict.get("best_model_length", "")
        bm_len_s = results_dict.get("best_model_std_length", "")
        bm_vel = results_dict.get("best_model_fwd_vel", "")
        bm_vel_s = results_dict.get("best_model_std_fwd_vel", "")
        bm_sr = results_dict.get("best_model_success_rate", "")
        bm_n_episodes = results_dict.get("best_model_n_episodes", 30)
        lines.append("")
        lines.append(f"Best Model Evaluation ({bm_n_episodes} episodes)")
        lines.append("-" * 40)
        lines.append(f"  Reward:       {bm_r} +/- {bm_s}")
        if bm_len != "":
            bm_dur_s = bm_len * sim_dt
            lines.append(f"  Ep length:    {bm_len} +/- {bm_len_s} steps ({bm_dur_s:.2f}s sim time)")
        if bm_vel != "":
            lines.append(f"  Fwd vel:      {bm_vel} +/- {bm_vel_s} m/s")
        if bm_sr != "":
            lines.append(f"  Success rate: {bm_sr:.0%}")
    _model_path = results_dict.get("model_path", "")
    if _model_path:
        _model_path = str(_model_path)
        if not Path(_model_path).suffix:
            _model_path += ".zip"
        lines.append(f"Best model:     {_model_path}")
        if "vecnorm_path" in results_dict:
            lines.append(f"VecNormalize:   {results_dict['vecnorm_path']}")
        lines.append("")
    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text)
    return summary_path


def write_training_summary(
    run_dir,
    stage_results_list: list[dict[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    n_envs: int,
    quick_test: bool = False,
) -> Path:
    """Write a training summary text file to the run directory.

    Returns the path to the written summary file.
    """
    summary_path = Path(run_dir) / "training_summary.txt"
    total_duration = sum(r["duration_seconds"] for r in stage_results_list)

    lines = [
        "Mesozoic Labs Training Summary",
        "=" * 50,
        "",
        f"Species:        {species.title()}",
        f"Algorithm:      {algorithm}",
        f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seed:           {seed}",
        f"Quick test:     {quick_test}",
        f"Parallel envs:  {n_envs}",
        f"Run directory:  {run_dir}",
        "",
    ]

    for r in stage_results_list:
        mean_len = r.get("mean_episode_length", 0)
        std_len = r.get("std_episode_length", 0)
        sim_dt = r.get("sim_dt", 0.01)
        mean_vel = r.get("mean_forward_vel", 0.0)
        std_vel = r.get("std_forward_vel", 0.0)
        lines.extend(
            [
                f"Stage {r['stage']}: {r['name']}",
                f"  Description:    {r['description']}",
                f"  Timesteps:      {r['timesteps']:,}",
                f"  Duration:       {format_duration(r['duration_seconds'])}",
                f"  Final eval:     {r['mean_reward']:.2f} +/- {r['std_reward']:.2f}",
                f"  Avg ep length:  {mean_len:.1f} +/- {std_len:.1f} steps ({mean_len * sim_dt:.2f}s sim time)",
                f"  Avg fwd vel:    {mean_vel:.2f} +/- {std_vel:.2f} m/s",
            ]
        )
        best_r = r.get("best_eval_reward", "")
        if best_r != "":
            best_s = r.get("best_eval_std", "")
            best_ts = r.get("best_eval_timestep", "")
            ts_label = f"  (at {best_ts:,} steps)" if isinstance(best_ts, int) else ""
            lines.append(f"  Best eval:      {best_r} +/- {best_s}{ts_label}")
        model_path = r.get("model_path")
        if model_path is not None:
            model_path_str = str(model_path)
            if not Path(model_path_str).suffix:
                model_path_str += ".zip"
            lines.append(f"  Best model:     {model_path_str}")
        lines.append("")

    lines.extend(
        [
            "-" * 50,
            f"Total training time: {format_duration(total_duration)}",
        ]
    )

    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text)
    return summary_path
