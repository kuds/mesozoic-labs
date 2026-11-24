"""Human-readable stage and run summary text files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ..curriculum.command_tracking_gate import COMMAND_TRACKING_GATE_KIND, WALK_PROFILE
from ..model_id import ModelId
from ..species_names import model_display_name
from ..stage_manifest import stage_display_name
from .formatting import format_duration


def _forward_velocity(results: dict[str, Any]) -> str:
    """``mean +/- std m/s``, or ``not measured`` when the panel did not run.

    The velocity keys are absent (never zero) when no post-training panel
    measured them, and a summary that printed ``0.00 m/s`` for that case was
    indistinguishable from a policy that stood still (review ER4).
    """
    mean_vel = results.get("mean_forward_vel")
    if mean_vel is None or mean_vel == "":
        return "not measured"
    std_vel = results.get("std_forward_vel")
    std_text = f" +/- {std_vel:.2f}" if isinstance(std_vel, (int, float)) else ""
    return f"{mean_vel:.2f}{std_text} m/s"


def _command_tracking_lines(results: dict[str, Any]) -> list[str]:
    """A ``command_tracking/v1`` stage's family statistics (decision D-D61), or nothing for another kind.

    The numbers are the ``command_*`` stage-result keys the judge re-derived
    from an admitted report (``stage_artifacts._apply_stage_gate`` copies
    nothing from a refused one); an unrecorded number prints as ``n/a``.
    """
    if results.get("gate_kind") != COMMAND_TRACKING_GATE_KIND or not results.get("command_profile"):
        return []

    def shown(key: str, spec: str = ".4f") -> str:
        value = results.get(key)
        if value is None or value == "" or isinstance(value, bool):
            return "n/a"
        if not isinstance(value, (int, float)):
            return str(value)
        if spec == "d":
            return str(int(value)) if float(value).is_integer() else str(value)
        return format(value, spec)

    profile = results["command_profile"]
    lines = [f"  Profile:      {profile} ({shown('command_n_episodes', 'd')} panel episodes)"]
    if profile == WALK_PROFILE:
        lines.append(
            f"  Cruise:       {shown('command_cruise_successes', 'd')}/{shown('command_cruise_episodes', 'd')} "
            f"successful; success LCB95 {shown('command_cruise_success_lcb')}, survival LCB95 "
            f"{shown('command_cruise_survival_lcb')}"
        )
        lines.append(
            f"  Stand:        {shown('command_stand_successes', 'd')}/{shown('command_stand_episodes', 'd')} "
            f"holding the stop ({shown('command_stand_success_fraction', '.3f')}); falls "
            f"{shown('command_stand_falls', 'd')}"
        )
    else:
        lines.append(
            f"  Follow:       {shown('command_follow_successes', 'd')}/{shown('command_follow_episodes', 'd')} "
            f"successful; success LCB95 {shown('command_follow_success_lcb')}, survival LCB95 "
            f"{shown('command_follow_survival_lcb')}"
        )
        if results.get("command_paired_delta_lcb") is not None:
            lines.append(f"  Reference:    paired success difference LCB95 {shown('command_paired_delta_lcb')}")
    if results.get("command_handoff_selected"):
        lines.append(f"  Handoff:      {results['command_handoff_selected']} (chosen by the gated statistic)")
    return lines


def write_stage_summary(
    stage_dir,
    results_dict: dict[str, Any],
    model_id: ModelId,
    algorithm: str,
    *,
    evaluation_episodes: int | None = None,
) -> Path:
    """Write a text summary for a single completed stage to its directory.

    *evaluation_episodes* is the size of the panel the ``best_model_*``
    results were rolled on (``stage_artifacts.EVALUATION_PANEL_EPISODES``
    from :func:`~environments.shared.reporting.generate_stage_artifacts`);
    without it the best-model section states no episode count.  The success
    rate line appears only when the panel measured one (a model that declares
    success keys).

    Returns the path to the written summary file.
    """
    from ..config import get_library_version

    summary_path = Path(stage_dir) / "stage_summary.txt"
    mean_len = results_dict.get("mean_episode_length", 0)
    std_len = results_dict.get("std_episode_length", 0)
    sim_dt = results_dict.get("sim_dt", 0.01)
    avg_duration_s = mean_len * sim_dt
    lines = [
        f"Mesozoic Labs: Stage {results_dict['stage']} Summary",
        "=" * 50,
        "",
        f"Model:          {model_display_name(model_id)}",
        f"Stage:          {results_dict['stage']} ({stage_display_name(results_dict['stage'], results_dict['name'])})",
        f"Description:    {results_dict['description']}",
        f"Algorithm:      {algorithm}",
        f"Version:        {get_library_version()}",
        f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timesteps:      {results_dict['timesteps']:,}",
        f"Duration:       {format_duration(results_dict['duration_seconds'])}",
        f"Final eval:     {results_dict['mean_reward']:.2f} +/- {results_dict['std_reward']:.2f}",
        f"Avg ep length:  {mean_len:.1f} +/- {std_len:.1f} steps ({avg_duration_s:.2f}s sim time)",
        f"Avg fwd vel:    {_forward_velocity(results_dict)}",
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
        panel = f" ({evaluation_episodes} episodes)" if evaluation_episodes is not None else ""
        lines.append(f"Best Model Evaluation{panel}")
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

    # The gate verdict, and every criterion that did not hold. Written here
    # because this was the only stage artifact a human reads that did not
    # record it: `collected_results.csv` carries the boolean but not the
    # reasons, and the reasons otherwise existed only in the exception text
    # of a Colab cell -- which is exactly what is lost when the runtime
    # disconnects on gate failure. Run 20260802_203215 advanced on a verdict
    # no artifact contradicted; a stage summary that states the failing
    # criteria makes that contradiction visible without re-running anything.
    gate_passed = results_dict.get("publication_gate_passed", results_dict.get("gate_passed"))
    if gate_passed is not None:
        lines.append("Curriculum Gate")
        lines.append("-" * 40)
        lines.append(f"  Verdict:      {'PASS' if gate_passed else 'FAIL'}")
        lines.extend(_command_tracking_lines(results_dict))
        for failure in results_dict.get("gate_failures") or []:
            lines.append(f"  - {failure}")
        lines.append("")
    summary_text = "\n".join(lines) + "\n"
    summary_path.write_text(summary_text)
    return summary_path


def write_training_summary(
    run_dir,
    stage_results_list: list[dict[str, Any]],
    model_id: ModelId,
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
        f"Model:          {model_display_name(model_id)}",
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
        lines.extend(
            [
                f"Stage {r['stage']}: {stage_display_name(r['stage'], r['name'])}",
                f"  Description:    {r['description']}",
                f"  Timesteps:      {r['timesteps']:,}",
                f"  Duration:       {format_duration(r['duration_seconds'])}",
                f"  Final eval:     {r['mean_reward']:.2f} +/- {r['std_reward']:.2f}",
                f"  Avg ep length:  {mean_len:.1f} +/- {std_len:.1f} steps ({mean_len * sim_dt:.2f}s sim time)",
                f"  Avg fwd vel:    {_forward_velocity(r)}",
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
