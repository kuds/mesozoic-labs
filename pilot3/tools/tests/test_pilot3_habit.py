"""Tests of pilot3_habit.py on synthetic ledgers and synthetic traces (pytest; walking pilot 3 scratch, never merged).

The criteria's synthetic cases: 4 and 5 of 10 over the bar; consecutive and split flags; an alarm at the window's
first and last evaluations; a duplicated row set after a resume; an evaluation with fewer than 10 rows; an
UNRESOLVED-only alarm (a POSSIBLE ALARM, settled or not by the trace); a missing column, which must read NOT RUN.
Plus the trace's CMD, PARK and DRIVEN rules and the seed reading. FIX2: the seed mode's refusals. FIX3: the ledger
must be the judged copy's own file (by resolved path, file name included), the window must cover the deciding
checkpoint (first evaluation 131,072, last less than 131,072 short), and no mode overwrites an existing --json.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot3_habit as ph  # noqa: E402

GRID = [131_072 * k for k in range(1, 62)]  # the 61 evaluations of an 8M run, through 7,995,392
HEADER = ["evaluation", "timesteps", "episode", "env", "return", "length", "saturated_actuator",
          "saturated_actuator_fraction", "saturated_actuator_positive_share", "saturated_actuator_ties",
          "actuators_saturated", "head_tip_height_mean_m", "neck_pitch_mean_deg", "head_pitch_mean_deg",
          "tail_3_pitch_act_command_mean"]
QUIET = ("", "0.0", "nan", "0", "0")


def row(step, ep, sat=QUIET, tail3=0.01, ev=0):
    name, frac, pos, ties, n = sat
    return {"evaluation": ev, "timesteps": step, "episode": ep, "env": 0, "return": 1000.0, "length": 1000,
            "saturated_actuator": name, "saturated_actuator_fraction": frac, "saturated_actuator_positive_share": pos,
            "saturated_actuator_ties": ties, "actuators_saturated": n, "head_tip_height_mean_m": 0.9,
            "neck_pitch_mean_deg": 16.0, "head_pitch_mean_deg": 14.0, "tail_3_pitch_act_command_mean": tail3}


def over(name, frac=0.5, ties=0, n=1):
    return (name, str(frac), "1.0", str(ties), str(n))


def evaluation(step, n_over=0, name="tail_3_pitch_act", n_rows=10, sat_over=None, tail3=0.01):
    sat_over = sat_over or over(name)
    return [row(step, ep, sat_over if ep < n_over else QUIET, tail3) for ep in range(n_rows)]


def ledger(evals: dict | None = None, grid=GRID, header=HEADER, extra_rows=()):
    rows = []
    evals = evals or {}
    for step in grid:
        rows += evals.get(step, evaluation(step))
    rows += list(extra_rows)
    return header, [{k: str(r.get(k, "")) for k in header} for r in rows]


def read(evals=None, through=8_000_000, **kw):
    header, rows = ledger(evals, **kw)
    return ph.ledger_reading(header, rows, through)


def test_a_quiet_ledger_reads_no_flag_and_61_evaluations():
    lr = read()
    assert lr["window"] == {"first": 131_072, "last": 7_995_392, "evaluations": 61, "through": 8_000_000}
    assert all(lr["groups"][g]["alarms"] == [] and lr["groups"][g]["flagged"] == [] for g in ("TAIL", "NECK", "ANY"))
    assert lr["td"]["overall"] == pytest.approx(0.01)


def test_4_of_10_is_not_flagged_and_5_of_10_is():
    s1, s2 = GRID[9], GRID[10]
    lr = read({s1: evaluation(s1, 4), s2: evaluation(s2, 4)})
    assert lr["groups"]["TAIL"]["flagged"] == [] and lr["groups"]["TAIL"]["alarms"] == []
    lr = read({s1: evaluation(s1, 5), s2: evaluation(s2, 5)})
    assert lr["groups"]["TAIL"]["flagged"] == [s1, s2] and lr["groups"]["TAIL"]["alarms"] == [s2]


def test_the_bar_is_strictly_above_0_10():
    s1, s2 = GRID[9], GRID[10]
    lr = read({s: evaluation(s, 10, sat_over=over("tail_3_pitch_act", 0.10)) for s in (s1, s2)})
    assert lr["groups"]["TAIL"]["flagged"] == []
    lr = read({s: evaluation(s, 10, sat_over=over("tail_3_pitch_act", 0.1001)) for s in (s1, s2)})
    assert lr["groups"]["TAIL"]["alarms"] == [s2]


def test_consecutive_flags_alarm_and_split_flags_do_not():
    a, b, c, d = GRID[20], GRID[21], GRID[22], GRID[23]
    lr = read({a: evaluation(a, 6), c: evaluation(c, 6)})  # flag, no flag, flag
    assert lr["groups"]["TAIL"]["flagged"] == [a, c] and lr["groups"]["TAIL"]["alarms"] == []
    lr = read({a: evaluation(a, 6), b: evaluation(b, 6), c: evaluation(c, 6), d: evaluation(d, 2)})
    assert lr["groups"]["TAIL"]["alarms"] == [b, c]


def test_an_alarm_at_the_window_s_first_and_last_evaluations():
    first, second = GRID[0], GRID[1]
    lr = read({first: evaluation(first, 5, "neck_pitch_act"), second: evaluation(second, 5, "neck_pitch_act")})
    assert lr["groups"]["NECK"]["alarms"] == [second] and lr["groups"]["TAIL"]["alarms"] == []
    last2, last = GRID[-2], GRID[-1]
    lr = read({last2: evaluation(last2, 7), last: evaluation(last, 7)})
    assert lr["groups"]["TAIL"]["alarms"] == [last] == [7_995_392]


def test_evaluations_after_the_window_are_not_read():
    after1, after2 = 8_126_464, 8_257_536
    header, rows = ledger(extra_rows=evaluation(after1, 10) + evaluation(after2, 10))
    lr = ph.ledger_reading(header, rows, 8_000_000)
    assert lr["groups"]["TAIL"]["alarms"] == [] and lr["after_window"] == [after1, after2]
    assert ph.ledger_reading(header, rows, None)["groups"]["TAIL"]["alarms"] == [after2]


def test_a_collapse_stopped_window_ends_at_its_newest_pair():
    s1, s2 = 3_670_016, 3_801_088
    lr = read({s1: evaluation(s1, 6), s2: evaluation(s2, 6)}, through=3_500_000)
    assert lr["window"]["last"] == 3_407_872 and lr["groups"]["TAIL"]["alarms"] == []


def test_a_duplicated_row_set_after_a_resume_reads_the_last_written_set():
    s = GRID[30]
    earlier_flagged = evaluation(s, 8)
    lr = read({GRID[29]: evaluation(GRID[29], 8), s: earlier_flagged},
              extra_rows=evaluation(s, 0))  # the resume re-evaluated s: quiet
    assert lr["superseded_row_sets"] == {str(s): [10]}
    assert lr["groups"]["TAIL"]["flagged"] == [GRID[29]] and lr["groups"]["TAIL"]["alarms"] == []
    lr = read({GRID[29]: evaluation(GRID[29], 8), s: evaluation(s, 0)}, extra_rows=evaluation(s, 8))
    assert lr["groups"]["TAIL"]["alarms"] == [s] and lr["superseded_row_sets"] == {str(s): [10]}


def test_a_repeated_episode_index_inside_one_block_starts_a_new_set():
    s = GRID[30]
    lr = read({s: evaluation(s, 0) + evaluation(s, 9), GRID[31]: evaluation(GRID[31], 9)})
    assert lr["superseded_row_sets"] == {str(s): [10]} and lr["groups"]["TAIL"]["alarms"] == [GRID[31]]


def test_an_evaluation_with_fewer_than_10_rows_is_printed_and_not_flagged():
    s1, s2 = GRID[9], GRID[10]
    lr = read({s1: evaluation(s1, 9, n_rows=9), s2: evaluation(s2, 9)})
    assert lr["incomplete_evaluations"] == [(s1, 9)] and lr["groups"]["TAIL"]["flagged"] == [s2]
    assert lr["groups"]["TAIL"]["alarms"] == []
    assert any("other than 10 rows" in line for line in ph.describe_ledger(lr, "x"))


def test_an_unresolved_only_alarm_is_a_possible_alarm():
    s1, s2 = GRID[12], GRID[13]
    tied_leg = over("r_knee_act", 0.4, ties=1, n=2)
    evals = {s: [row(s, ep, over("tail_3_pitch_act") if ep < 3 else (tied_leg if ep < 6 else QUIET))
                 for ep in range(10)] for s in (s1, s2)}
    lr = read(evals)
    assert lr["groups"]["TAIL"]["alarms"] == [] and lr["groups"]["TAIL"]["possible_alarms"] == [s2]
    assert lr["groups"]["NECK"]["possible_alarms"] == []  # 3 tied leg episodes alone flag nothing
    untied = {s: [row(s, ep, over("tail_3_pitch_act") if ep < 3 else (over("r_knee_act") if ep < 6 else QUIET))
                  for ep in range(10)] for s in (s1, s2)}
    assert read(untied)["groups"]["TAIL"]["possible_alarms"] == []  # no tie: not UNRESOLVED
    assert read(untied)["groups"]["ANY"]["alarms"] == [s2]


def test_actuator_names_and_groups():
    assert ph.actuator_index("tail_1_yaw_act") == 12 and ph.group_of(12) == "TAIL"
    assert ph.actuator_index("actuator[2]") == 2 and ph.group_of(2) == "NECK"
    assert ph.actuator_index("actuator_7") == 7 and ph.group_of(7) == "LEGS"
    assert ph.actuator_index("") is None and ph.group_of(None) is None
    assert ph.TAIL_PITCH == (11, 13, 14) and ph.GROUPS["NECK"] == (0, 1, 2)


def test_a_missing_column_reads_not_run():
    header = [c for c in HEADER if c != "saturated_actuator_ties"]
    h, rows = ledger(header=header)
    lr = ph.ledger_reading(h, rows, 8_000_000)
    assert lr["read"] is False and "saturated_actuator_ties" in lr["not_run"][0]
    assert ph.seed_reading(lr, trace(quiet=True))["reading"] == "NOT RUN"


def test_td_per_2m_blocks():
    evals = {s: evaluation(s, 0, tail3=(0.1 if s <= 2_000_000 else -0.3)) for s in GRID}
    td = read(evals)["td"]
    assert td["per_2M"]["1-2000000"] == pytest.approx(0.1) and td["per_2M"]["2000001-4000000"] == pytest.approx(0.3)
    assert td["evaluations"] == 61


# ---------------------------------------------------------------- the trace


def episode(seed, length=1000, cmd=None, window_cmd=None, saturate=()):
    """A synthetic traced episode: commands 0 (or cmd per actuator), window steps 100-999 set to window_cmd, and the
    actuators in `saturate` at +1.0 on 20% of the window."""
    base = [0.0] * 15
    if cmd:
        for a, v in cmd.items():
            base[a] = v
    rows = [list(base) for _ in range(length)]
    for t in range(100, length):
        if window_cmd:
            for a, v in window_cmd.items():
                rows[t][a] = v
        for a in saturate:
            if (t - 100) % 5 == 0:
                rows[t][a] = 1.0
    return {"seed": seed, "length": length, "commands": rows}


def trace(quiet=False, eps=None):
    return ph.trace_reading(eps if eps is not None else [episode(s) for s in ph.TRACE_SEEDS])


def test_cmd_is_the_mean_of_full_horizon_window_means():
    eps = [episode(s, window_cmd={14: 0.7}, cmd={14: -1.0}) for s in ph.TRACE_SEEDS[:15]]
    eps += [episode(s, length=400, window_cmd={14: -1.0}) for s in ph.TRACE_SEEDS[15:]]
    tr = trace(eps=eps)
    assert tr["full_horizon"] == 15 and tr["cmd"]["tail_3_pitch_act"] == pytest.approx(0.7)
    assert [p["actuator"] for p in tr["park"]] == ["tail_3_pitch_act"] and tr["driven"] == []


@pytest.mark.parametrize("a, value, park, driven", [
    (14, 0.60, ["tail_3_pitch_act"], []), (14, 0.5999, [], []), (11, 0.65, ["tail_1_pitch_act"], []),
    (13, -0.60, [], ["tail_2_pitch_act"]), (12, 0.70, [], ["tail_1_yaw_act"]), (12, -0.61, [], ["tail_1_yaw_act"]),
    (0, 0.95, [], []),  # a neck command is no PARK or DRIVEN (the neck reads through the ledger alarms)
])
def test_park_is_tail_up_on_a_pitch_actuator_and_driven_either_sign(a, value, park, driven):
    tr = trace(eps=[episode(s, window_cmd={a: value}) for s in ph.TRACE_SEEDS])
    assert [p["actuator"] for p in tr["park"]] == park and [p["actuator"] for p in tr["driven"]] == driven


def test_fewer_than_10_full_horizon_episodes_forms_no_cmd():
    eps = [episode(s) for s in ph.TRACE_SEEDS[:9]] + [episode(s, length=500) for s in ph.TRACE_SEEDS[9:]]
    tr = trace(eps=eps)
    assert tr["cmd_formed"] is False and "CMD not formed" in tr["not_run"][0]
    assert ph.seed_reading(read(), tr)["reading"] == "NOT RUN"


def test_the_bar_in_the_trace_counts_full_horizon_episodes_per_actuator():
    eps = [episode(s, saturate=(14,) if i < 2 else ()) for i, s in enumerate(ph.TRACE_SEEDS)]
    tr = trace(eps=eps)
    assert tr["episodes_over_bar"]["tail_3_pitch_act"] == 2 and tr["settles"]["TAIL"] == ["tail_3_pitch_act"]
    eps = [episode(s, saturate=(14,) if i < 1 else ()) for i, s in enumerate(ph.TRACE_SEEDS)]
    assert trace(eps=eps)["settles"]["TAIL"] == []


# ---------------------------------------------------------------- the seed reading


def possible_tail_ledger():
    s1, s2 = GRID[12], GRID[13]
    tied_leg = over("r_knee_act", 0.4, ties=1, n=2)
    return read({s: [row(s, ep, over("tail_3_pitch_act") if ep < 3 else (tied_leg if ep < 6 else QUIET))
                     for ep in range(10)] for s in (s1, s2)})


def test_seed_reading_no_habit_habit_and_not_run():
    assert ph.seed_reading(read(), trace())["reading"] == "NO HABIT"
    s1, s2 = GRID[0], GRID[1]
    alarm = read({s1: evaluation(s1, 5), s2: evaluation(s2, 5)})
    r = ph.seed_reading(alarm, trace())
    assert r["reading"] == "HABIT" and r["tail_habit"] and r["habits"][0]["text"] == "TAIL ALARM at 262,144"
    r = ph.seed_reading(alarm, None)  # a habit found is not undone by an unread trace
    assert r["reading"] == "HABIT" and r["not_run"] == ["no trace"]
    assert ph.seed_reading(read(), None)["reading"] == "NOT RUN"


def test_a_neck_alarm_is_a_neck_habit():
    s1, s2 = GRID[40], GRID[41]
    r = ph.seed_reading(read({s: evaluation(s, 6, "head_pitch_act") for s in (s1, s2)}), trace())
    assert r["reading"] == "HABIT" and r["neck_habit"] and not r["tail_habit"]


def test_a_leg_alarm_is_never_a_habit():
    s1, s2 = GRID[40], GRID[41]
    lr = read({s: evaluation(s, 10, "l_knee_act") for s in (s1, s2)})
    assert lr["groups"]["LEGS"]["alarms"] == [s2]
    assert ph.seed_reading(lr, trace())["reading"] == "NO HABIT"


def test_a_possible_alarm_is_settled_by_the_trace():
    lr = possible_tail_ledger()
    settles = trace(eps=[episode(s, saturate=(13,) if i < 2 else ()) for i, s in enumerate(ph.TRACE_SEEDS)])
    r = ph.seed_reading(lr, settles)
    assert r["reading"] == "HABIT" and r["habits"][0]["kind"] == "TAIL POSSIBLE ALARM settled by trace"
    assert ph.seed_reading(lr, trace())["reading"] == "NO HABIT"
    r = ph.seed_reading(lr, None)
    assert r["reading"] == "NOT RUN" and any("POSSIBLE ALARM" in x for x in r["not_run"])


def test_park_and_driven_are_habits():
    r = ph.seed_reading(read(), trace(eps=[episode(s, window_cmd={14: 0.8, 12: -0.7}) for s in ph.TRACE_SEEDS]))
    assert r["reading"] == "HABIT" and [h["kind"] for h in r["habits"]] == ["PARK", "DRIVEN"]


def test_the_ledger_cli_writes_json(tmp_path):
    header, rows = ledger({GRID[0]: evaluation(GRID[0], 5), GRID[1]: evaluation(GRID[1], 5)})
    path = tmp_path / "eval_episodes.csv"
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "l.json"
    r = subprocess.run([sys.executable, "-I", "-B", str(TOOLS / "pilot3_habit.py"), "ledger", str(path), "--through",
                        "8000000", "--json", str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc["reading"]["groups"]["TAIL"]["alarms"] == [262_144] and doc["tool_sha256"].startswith("sha256:")
    assert "TAIL: 2 flagged evaluations; 1 ALARMS (262,144-262,144)" in r.stdout


# ---------------------------------------------------------------- FIX2: the seed mode's refusals (before any read)

HEAD = "3ae1c95967547ada569ad71aebaa88999afa02cd"


def _binding(tmp_path=None, run="pilot3_walk_s42", **over):
    """A pilot cell's binding; its "run" is the judged copy tmp_path/<run> (FIX3: compared by resolved path)."""
    where = None if run is None else str(tmp_path / run) if tmp_path is not None else f"/copies/{run}"
    b = {"seed": 42, "head": HEAD, "run": where, "validation_only": False, "off_schedule": False,
         "expected_budget": 8_000_000, "steps": 8_000_000}
    b.update(over)
    return b


def _ledger(tmp_path, run="pilot3_walk_s42", name="eval_episodes.csv"):
    path = tmp_path / run / "03_locomotion" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("timesteps\n")
    return path


def test_seed_accepts_a_pilot_cell_with_its_own_run_s_ledger(tmp_path):
    assert ph.seed_problems(_binding(tmp_path), seed=42, head=HEAD, dirty="", ledger=_ledger(tmp_path)) == []


def test_seed_refuses_a_retired_run_s_ledger_and_a_ledger_elsewhere(tmp_path):
    b = _binding(tmp_path, run="pilot3_walk_s42_b")
    got = ph.seed_problems(b, seed=42, head=HEAD, dirty="", ledger=_ledger(tmp_path))
    assert len(got) == 1 and "is not the judged run's own ledger" in got[0]
    assert str(tmp_path / "pilot3_walk_s42_b" / "03_locomotion" / "eval_episodes.csv") in got[0]
    other = tmp_path / "eval_episodes.csv"
    other.write_text("timesteps\n")
    assert len(ph.seed_problems(_binding(tmp_path), seed=42, head=HEAD, dirty="", ledger=other)) == 1
    got = ph.seed_problems(_binding(tmp_path, run=None), seed=42, head=HEAD, dirty="", ledger=_ledger(tmp_path))
    assert len(got) == 1 and "<no run recorded>" in got[0]


def test_seed_refuses_another_file_in_the_run_s_stage_folder(tmp_path):
    """FIX3 (the third review's M3): the file name is part of the check; RUN/03_locomotion/other.csv is refused."""
    got = ph.seed_problems(_binding(tmp_path), seed=42, head=HEAD, dirty="", ledger=_ledger(tmp_path, name="other.csv"))
    assert len(got) == 1 and "is not the judged run's own ledger" in got[0]
    stage_dir_csv = tmp_path / "pilot3_walk_s42" / "eval_episodes.csv"
    stage_dir_csv.write_text("timesteps\n")
    assert len(ph.seed_problems(_binding(tmp_path), seed=42, head=HEAD, dirty="", ledger=stage_dir_csv)) == 1


def test_seed_refuses_a_same_named_copy_outside_the_judged_run(tmp_path):
    """FIX3 (the third review's M1): a stale download with the judged run's folder name, elsewhere, is refused; the
    judged copy's own ledger, named through a relative or symlinked path, is accepted."""
    judged = _binding(tmp_path / "judged")
    stale = _ledger(tmp_path / "downloads")
    assert stale.parent.parent.name == Path(judged["run"]).name
    got = ph.seed_problems(judged, seed=42, head=HEAD, dirty="", ledger=stale)
    assert len(got) == 1 and "is not the judged run's own ledger" in got[0]
    own = _ledger(tmp_path / "judged")
    link = tmp_path / "link_to_judged"
    link.symlink_to(tmp_path / "judged", target_is_directory=True)
    via_link = link / own.relative_to(tmp_path / "judged")
    assert ph.seed_problems(judged, seed=42, head=HEAD, dirty="", ledger=via_link) == []


def test_seed_refuses_a_dirty_tree_an_off_schedule_and_a_validation_only_cell(tmp_path):
    led = _ledger(tmp_path)
    got = ph.seed_problems(_binding(tmp_path), seed=42, head=HEAD, dirty=" M environments/x.py", ledger=led)
    assert len(got) == 1 and "local edits" in got[0]
    got = ph.seed_problems(_binding(tmp_path, off_schedule=True,
                                    off_schedule_reason="OFF-SCHEDULE: resumed at step 32,768"),
                           seed=42, head=HEAD, dirty="", ledger=led)
    assert len(got) == 1 and "off-schedule" in got[0]
    got = ph.seed_problems(_binding(tmp_path, off_schedule=True), seed=42, head=HEAD, dirty="", ledger=led,
                           validation=True)
    assert len(got) == 1 and "off-schedule" in got[0]
    val = _binding(tmp_path, validation_only=True, expected_budget=50_000)
    got = ph.seed_problems(val, seed=42, head=HEAD, dirty="", ledger=led)
    assert len(got) == 1 and "validation-only (run budget 50000)" in got[0]
    assert ph.seed_problems(val, seed=42, head=HEAD, dirty="", ledger=led, validation=True) == []


def test_seed_lists_every_problem_at_once(tmp_path):
    got = ph.seed_problems(_binding(tmp_path, seed=52, head="4d5f4e7", validation_only=True), seed=42, head=HEAD,
                           dirty="x", ledger=tmp_path / "x.csv")
    assert len(got) == 5


# ---------------------------------------------------------------- FIX3: the window must cover the deciding checkpoint


def test_a_full_window_covers_the_deciding_checkpoint():
    assert ph.window_problems(read(), 8_000_000) == []
    assert ph.window_problems(read(through=5_500_000), 5_500_000) == []  # collapse-stopped: 5,373,952
    assert ph.window_problems(read(through=2_000_000), 2_000_000) == []  # 1,966,080, 33,920 short


def test_a_stale_copy_cut_short_of_the_deciding_checkpoint_is_a_window_problem():
    """The third review's probe: a ledger cut at 3,800,000 read through 8,000,000 (28 evaluations, NO HABIT where the
    whole ledger has a TAIL ALARM) must not be read."""
    cut = [s for s in GRID if s <= 3_800_000]
    lr = read(grid=cut)
    assert lr["window"]["last"] == 3_670_016 and lr["window"]["evaluations"] == 28
    got = ph.window_problems(lr, 8_000_000)
    assert len(got) == 1 and "3,670,016, 4,329,984 steps short" in got[0]


def test_the_window_s_tail_boundary_is_one_grid_step():
    """Last evaluation 131,072 short (one evaluation missing at the end) is refused; 131,071 short is accepted."""
    lr = read(grid=GRID[:-1])  # last 7,864,320
    assert ph.window_problems(lr, 7_864_320 + 131_071) == []
    assert len(ph.window_problems(lr, 7_864_320 + 131_072)) == 1
    assert len(ph.window_problems(lr, 8_000_000)) == 1  # 135,680 short


def test_a_copy_without_its_first_rows_is_a_window_problem():
    got = ph.window_problems(read(grid=GRID[1:]), 8_000_000)
    assert len(got) == 1 and "first evaluation is 262,144, not 131,072" in got[0]
    got = ph.window_problems(read(grid=[]), 8_000_000)
    assert len(got) == 1 and "no evaluation at or before" in got[0]


def test_a_resumed_grid_covers_the_checkpoint_and_a_missing_column_is_left_to_not_run():
    """After a RESUME at S the evaluations fall at S + 131,072 k; the window still starts at 131,072 and ends within
    one grid step of the deciding checkpoint. A ledger missing a column is not refused here (it reads NOT RUN)."""
    s = 2_500_000
    grid = [g for g in GRID if g <= s] + [s + 131_072 * k for k in range(1, 42)]
    assert ph.window_problems(read(grid=grid), 8_000_000) == []
    lr = read(header=[c for c in HEADER if c != "saturated_actuator_ties"])
    assert lr["read"] is False and ph.window_problems(lr, 8_000_000) == []


def test_the_window_grid_is_131072_and_the_run_s_own_only_for_a_validation_read():
    cad = {"curriculum_cadence": {"eval_freq": 32_768}}
    assert ph.EVAL_EVERY == 131_072
    assert ph.window_grid(dict(_binding(), **cad)) == 131_072
    assert ph.window_grid(dict(_binding(), **cad), validation=True) == 131_072  # not validation-only
    val = dict(_binding(validation_only=True), **cad)
    assert ph.window_grid(val) == 131_072 and ph.window_grid(val, validation=True) == 32_768
    for bad in (None, 0, -5, True, "32768"):
        assert ph.window_grid(dict(val, curriculum_cadence={"eval_freq": bad}), validation=True) == 131_072
    smoke = read(grid=[32_768, 65_536])
    assert ph.window_problems(smoke, 65_536, 32_768) == [] and len(ph.window_problems(smoke, 65_536)) == 1


def _seed_main_setup(tmp_path, monkeypatch, grid):
    """A judged cell, its run copy holding a ledger on `grid`, the checkpoint pair, and stand-ins for the tree
    (environments' location, git's HEAD and a clean status) and for the trace (20 quiet full-horizon episodes; it
    records that it was rolled). No simulation and no git command runs."""
    import types

    run = tmp_path / "copies" / "pilot3_walk_s42"
    header, rows = ledger(grid=grid)
    lp = run / "03_locomotion" / "eval_episodes.csv"
    lp.parent.mkdir(parents=True)
    with lp.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    models = run / "03_locomotion" / "models"
    models.mkdir()
    (models / "stage2_8000000_steps.zip").write_bytes(b"zip")
    (models / "stage2_vecnormalize_8000000_steps.pkl").write_bytes(b"pkl")
    cell = tmp_path / "OUT" / "s42" / "8000000"
    cell.mkdir(parents=True)
    b = _binding(tmp_path / "copies", checkpoint=str(models / "stage2_8000000_steps.zip"),
                 vecnorm=str(models / "stage2_vecnormalize_8000000_steps.pkl"))
    b.update(checkpoint_sha256=ph.sha256_file(b["checkpoint"]), vecnorm_sha256=ph.sha256_file(b["vecnorm"]))
    (cell / "binding.json").write_text(json.dumps(b))
    fake_tree = tmp_path / "tree" / "environments"
    fake_tree.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "environments",
                        types.SimpleNamespace(__file__=str(fake_tree / "__init__.py")))
    monkeypatch.setattr(ph.subprocess, "check_output",
                        lambda cmd, **kw: HEAD + "\n" if "rev-parse" in cmd else "")
    rolled = []

    def fake_roll(checkpoint, vecnorm, seeds=ph.TRACE_SEEDS):
        rolled.append(checkpoint)
        return [{"seed": s, "length": 1000, "commands": [[0.0] * 15] * 1000} for s in seeds]

    monkeypatch.setattr(ph, "roll_trace", fake_roll)
    return lp, cell, rolled


def test_seed_main_refuses_a_stale_judged_copy_before_the_trace(tmp_path, monkeypatch, capsys):
    """FIX3, through main: the judged copy's own ledger, stale (cut at 3,800,000), is refused after reading and
    before the trace; nothing is written."""
    lp, cell, rolled = _seed_main_setup(tmp_path, monkeypatch, [s for s in GRID if s <= 3_800_000])
    out = tmp_path / "HABIT" / "s42.json"
    out.parent.mkdir()
    rc = ph.main(["seed", "--seed", "42", "--ledger", str(lp), "--cell", str(cell), "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 2 and not out.exists() and rolled == []
    assert "refusing (nothing written): the ledger's last evaluation" in text and "4,329,984 steps short" in text


def test_seed_main_reads_a_covering_window_and_records_the_check(tmp_path, monkeypatch, capsys):
    lp, cell, rolled = _seed_main_setup(tmp_path, monkeypatch, GRID)
    out = tmp_path / "s42.json"
    rc = ph.main(["seed", "--seed", "42", "--ledger", str(lp), "--cell", str(cell), "--json", str(out)])
    assert rc == 0 and len(rolled) == 1, capsys.readouterr().out
    doc = json.loads(out.read_text())
    assert doc["reading"] == "NO HABIT" and doc["window_check"] == {
        "grid": 131_072, "deciding_steps": 8_000_000, "first": 131_072, "last": 7_995_392, "problems": []}
    assert doc["ledger_reading"]["window"]["evaluations"] == 61 and doc["validation_only"] is False


def test_every_mode_refuses_an_existing_json_target(tmp_path):
    header, rows = ledger()
    path = tmp_path / "eval_episodes.csv"
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    target = tmp_path / "binding.json"
    target.write_text("{}")
    for argv in (["ledger", str(path), "--json", str(target)],
                 ["trace", "--checkpoint", "a.zip", "--vecnorm", "b.pkl", "--json", str(target)],
                 ["seed", "--seed", "42", "--ledger", str(path), "--cell", str(tmp_path), "--json", str(target)]):
        assert ph.main(argv) == 2 and target.read_text() == "{}"
    assert ph.refuse_existing_json(None) is None and ph.refuse_existing_json(str(tmp_path / "new.json")) is None
