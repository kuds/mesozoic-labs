"""Training result reporting utilities.

Provides functions to write human-readable stage and training summaries as
well as machine-readable JSON result files.  These were originally defined
inline in the Colab training notebook and are now shared so that both the
notebook and CLI training scripts can produce consistent output.
"""

import json as _json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string (e.g. ``2h 15m 30s``)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def format_duration_hms(seconds: float) -> str:
    """Format seconds as ``H:MM:SS``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def write_stage_summary(
    stage_dir,
    results_dict: Dict[str, Any],
    species: str,
    algorithm: str,
) -> Path:
    """Write a text summary for a single completed stage to its directory.

    Returns the path to the written summary file.
    """
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
        lines.append("")
        lines.append("Best Model Evaluation (30 episodes)")
        lines.append("-" * 40)
        lines.append(f"  Reward:       {bm_r} +/- {bm_s}")
        if bm_len != "":
            bm_dur_s = bm_len * sim_dt
            lines.append(f"  Ep length:    {bm_len} +/- {bm_len_s} steps ({bm_dur_s:.2f}s sim time)")
        if bm_vel != "":
            lines.append(f"  Fwd vel:      {bm_vel} +/- {bm_vel_s} m/s")
        if bm_sr != "":
            lines.append(f"  Success rate: {bm_sr:.0%}")
    lines.extend(
        [
            f"Best model:     {results_dict['model_path']}.zip",
            f"VecNormalize:   {results_dict['vecnorm_path']}",
            "",
        ]
    )
    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text)
    return summary_path


def write_training_summary(
    run_dir,
    stage_results_list: List[Dict[str, Any]],
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
        lines.extend(
            [
                f"  Best model:     {r['model_path']}.zip",
                "",
            ]
        )

    lines.extend(
        [
            "-" * 50,
            f"Total training time: {format_duration(total_duration)}",
        ]
    )

    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text)
    return summary_path


def save_results_json(
    stage_results_list: List[Dict[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    results_dir: "str | Path",
) -> Path:
    """Save a ``summary.json`` to *results_dir*.

    Creates a machine-readable record of the training run that can be
    used to auto-generate the README results table and website content.

    Returns the path to the written JSON file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    total_duration = sum(r["duration_seconds"] for r in stage_results_list)
    total_timesteps = sum(r["timesteps"] for r in stage_results_list)
    final_result = stage_results_list[-1]

    stages = {}
    for r in stage_results_list:
        stage_data = {
            "name": r["name"],
            "timesteps": r["timesteps"],
            "avg_reward": round(r["mean_reward"], 2),
            "std_reward": round(r["std_reward"], 2),
            "training_time_seconds": round(r["duration_seconds"], 1),
            "training_time": format_duration_hms(r["duration_seconds"]),
        }
        if "mean_forward_vel" in r:
            stage_data["avg_forward_vel"] = round(r["mean_forward_vel"], 2)
        stages[str(r["stage"])] = stage_data

    summary = {
        "species": species,
        "algorithm": algorithm,
        "hardware": "Google Colab T4 GPU",
        "seed": seed,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "stages": stages,
        "total_timesteps": total_timesteps,
        "total_training_time_seconds": round(total_duration, 1),
        "total_training_time": format_duration_hms(total_duration),
        "final_avg_reward": round(final_result["mean_reward"], 2),
    }

    summary_path = results_dir / "summary.json"
    summary_path.write_text(_json.dumps(summary, indent=2) + "\n")
    return summary_path
