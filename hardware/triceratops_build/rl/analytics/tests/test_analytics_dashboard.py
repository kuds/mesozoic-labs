"""The training dashboard reads both eval_during_training.jsonl formats (speeds only, as trike60-pipe1; and with the
gait measures of rl/gait_metrics.py), tolerates bad lines, and marks the schedules a run record carries."""

import json
from pathlib import Path

import numpy as np
import pytest

import training_dashboard as td

PIPE1_LIKE = [
    {"steps": 250000, "time_utc": "x", "deterministic": True, "delay_steps": 10, "episodes": [
        {"command_mps": 0.0, "forward_speed_mps": 0.002, "fell": False, "termination": "", "length_s": 20.0},
        {"command_mps": 0.15, "forward_speed_mps": 0.021, "fell": False, "termination": "", "length_s": 20.0},
        {"command_mps": 0.25, "forward_speed_mps": 0.046, "fell": False, "termination": "", "length_s": 20.0}]},
    {"steps": 500000, "time_utc": "x", "deterministic": True, "delay_steps": 10, "episodes": [
        {"command_mps": 0.0, "forward_speed_mps": 0.001, "fell": False, "termination": "", "length_s": 20.0},
        {"command_mps": 0.15, "forward_speed_mps": 0.108, "fell": False, "termination": "", "length_s": 20.0},
        {"command_mps": 0.25, "forward_speed_mps": 0.167, "fell": True, "termination": "tilt", "length_s": 7.2}]},
]


def measures(slip, rms_knee, drift):
    """The keys of rl/gait_metrics.measures that the dashboard reads (a real output carries more)."""
    groups = ("roll", "pitch", "front elbow", "hind knee")
    return {"forward_speed_mps": 0.24, "sideways_drift_m": drift, "heading_change_deg": 2.0, "power_w": 42.0,
            "slip_ratio": {c: slip for c in td.CORNERS}, "touchdowns_per_s": {c: 3.0 for c in td.CORNERS},
            "drive_groups": {g: {"rms_over_continuous_worst": rms_knee if g == "hind knee" else 0.8,
                                 "worst_joint": "x", "at_limit_nominal": 0.01, "at_limit_episode": 0.03} for g in groups}}


def new_format(steps, slip, rms, drift):
    eps = [{"command_mps": c, "forward_speed_mps": 0.9 * c, "fell": False, "termination": "", "length_s": 20.0,
            "measures": measures(slip + c, rms, drift)} for c in (0.0, 0.15, 0.25)]
    return {"steps": steps, "time_utc": "x", "deterministic": True, "delay_steps": 10, "episodes": eps}


def lines(objs):
    return [json.dumps(o) for o in objs]


def test_speeds_only_format():
    p = td.parse_eval_lines(lines(PIPE1_LIKE))
    assert p["format"] == "speeds only" and p["skipped"] == 0 and len(p["records"]) == 2
    s = td.series(p)
    assert s["commands"] == [0.0, 0.15, 0.25]
    assert s["speed"]["0.25"] == [0.046, 0.167]
    assert s["falls"] == [0, 1] and s["gait"] == {} and s["gait_from_command"] is None


def test_gait_measures_format_reads_the_highest_command():
    p = td.parse_eval_lines(lines([new_format(250000, 0.3, 1.2, 0.1), new_format(500000, 0.2, 1.05, -0.05)]))
    assert p["format"] == "with gait measures"
    s = td.series(p)
    assert s["gait_from_command"] == 0.25
    assert s["gait"]["slip_ratio"]["left_fore"] == pytest.approx([0.55, 0.45])  # slip + 0.25 at the top command
    assert s["gait"]["rms_over_continuous_worst"]["hind knee"] == [1.2, 1.05]
    assert s["gait"]["sideways_drift_m"] == [0.1, -0.05]
    assert s["gait"]["power_w"] == [42.0, 42.0]


def test_bad_lines_are_skipped_and_a_repeated_evaluation_replaces_the_earlier():
    text = lines(PIPE1_LIKE) + ["", "{not json", json.dumps({"no_steps": 1}), json.dumps({"steps": "abc"}),
                                json.dumps(dict(PIPE1_LIKE[0], episodes=PIPE1_LIKE[1]["episodes"]))]
    p = td.parse_eval_lines(text)
    assert p["skipped"] == 3
    assert [r["steps"] for r in p["records"]] == [250000, 500000]
    assert p["records"][0]["episodes"][2]["speed"] == 0.167  # the later line for 250000 steps


def test_mixed_and_partial_measures():
    rec = new_format(750000, 0.2, 1.0, 0.0)
    del rec["episodes"][2]["measures"]["drive_groups"]["pitch"]
    rec["episodes"][2]["measures"]["slip_ratio"]["right_hind"] = None
    p = td.parse_eval_lines(lines(PIPE1_LIKE + [rec]))
    s = td.series(p)
    assert p["format"] == "with gait measures"
    assert s["gait"]["rms_over_continuous_worst"]["pitch"] == [None, None, None]
    assert s["gait"]["slip_ratio"]["right_hind"] == [None, None, None]
    assert s["gait"]["slip_ratio"]["left_fore"][:2] == [None, None]  # the speeds-only evaluations


def test_schedules_from_run_records():
    assert td.schedules(None) == {}
    pipe1 = {"settings": {"penalty_ramp_steps": 2000000, "penalty_ramp_start": 0.1}}
    newer = {"settings": {"penalty_ramp_steps": 2000000, "penalty_ramp_start": 0.1, "late_ramp_from": 1000000,
                          "late_ramp_to": 3000000}}
    assert set(td.schedules(pipe1)) == {"penalty"}
    sch = td.schedules(newer)
    curves = td.schedule_curves(sch, np.array([0, 1_000_000, 2_000_000, 3_000_000]))
    pen, late = curves.values()
    assert pen == pytest.approx([0.1, 0.55, 1.0, 1.0])
    assert late == pytest.approx([0.0, 0.0, 0.5, 1.0])


def test_dashboards_render_both_formats(tmp_path):
    rec = {"settings": {"penalty_ramp_steps": 2000000, "penalty_ramp_start": 0.1, "late_ramp_from": 1000000,
                        "late_ramp_to": 3000000}}
    old = td.parse_eval_lines(lines(PIPE1_LIKE))
    new = td.parse_eval_lines(lines([new_format(s, 0.3 - s / 1e7, 1.2 - s / 1e7, 0.0)
                                     for s in range(250000, 3000001, 250000)]))
    f1 = td.dashboard("old", old, None, tmp_path, "a.jsonl")
    f2 = td.dashboard("new", new, rec, tmp_path, "b.jsonl")
    f3 = td.compare([("old", old, None), ("new", new, rec)], tmp_path)
    for f in (f1, f2, f3):
        size = (tmp_path / f).stat().st_size
        assert 10_000 < size < 500 * 1024


PIPE1_FILE = Path("/root/.claude/projects/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/tool-results/"
                  "mcp-Google_Drive-blob-1795062375513-wklqar.bin")


@pytest.mark.skipif(not PIPE1_FILE.exists(), reason="pipe1's downloaded eval_during_training.jsonl is not on this machine")
def test_the_real_pipe1_file_matches_runs_md():
    p = td.parse_eval_file(PIPE1_FILE)
    s = td.series(p)
    assert p["format"] == "speeds only" and p["skipped"] == 0
    assert s["steps"][0] == 250000 and len(s["steps"]) == 8
    # RUNS.md's table: 0.046 at 250k, 0.237 at 2.0M for the 0.25 m/s command; no falls
    assert round(s["speed"]["0.25"][0], 3) == 0.046 and round(s["speed"]["0.25"][-1], 3) == 0.237
    assert sum(s["falls"]) == 0
