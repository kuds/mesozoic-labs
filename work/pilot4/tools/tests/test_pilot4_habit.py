"""Tests of pilot4_habit.py on synthetic ledgers and synthetic traces (pytest; walking pilot 4 scratch, never merged).

Walking pilot 3's tests/test_pilot3_habit.py (sha256 a19fcd12...) adapted to pilot 4's cadence (an evaluation every
65,536 steps: 30 evaluations through 1,966,080 for the 2,000,000 budget, the window's first evaluation at 65,536, the
coverage rule one 65,536 step), TD per 500,000 steps, the pilot-4 RUN_ID (pilot4_walk_s<seed>) and the deciding
checkpoint 2,000,000. Every synthetic case of pilot 3's is kept: 4 and 5 of 10 over the bar; consecutive and split
flags; an alarm at the window's first and last evaluations; a duplicated row set after a resume; an evaluation with
fewer than 10 rows; an UNRESOLVED-only alarm (a POSSIBLE ALARM, settled or not by the trace); a missing column (NOT
RUN); the trace's CMD, PARK and DRIVEN rules; the seed reading; the seed mode's refusals; the ledger as the judged
copy's own file; the window's coverage; no overwrite of --json.
New for pilot 4: the constants the TOOL PLAN changes; every definition (constants and functions) equal to pilot 3's
pinned file, read as text and compared by syntax tree, never imported from pilot 3's folder; pilot 3's and pilot 4's
ledger readings equal on the same ledgers but for the two changed fields (the grid gaps and the TD blocks); a resume
from any intact pair (250,000 and later) still covers the deciding checkpoint; the four judged checkpoints' windows;
the report-only wording of the record and the printed reading.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot4_habit as ph  # noqa: E402

PILOT3_TOOLS = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/"
                    "pilot3/tools")
PILOT3_HABIT = PILOT3_TOOLS / "pilot3_habit.py"
PILOT3_HABIT_SHA256 = "56f1c78deaea9ff64ef028eb10718304c944e90435a9b676b4c28f33855f1388"

GRID = [65_536 * k for k in range(1, 31)]  # the 30 evaluations of a 2M run, through 1,966,080
BUDGET = 2_000_000
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


def read(evals=None, through=BUDGET, **kw):
    header, rows = ledger(evals, **kw)
    return ph.ledger_reading(header, rows, through)


# ---------------------------------------------------------------- pilot 4's constants and pilot 3's definitions


def _pilot3_source() -> str:
    data = PILOT3_HABIT.read_bytes()
    assert hashlib.sha256(data).hexdigest() == PILOT3_HABIT_SHA256, "pilot 3's pinned pilot3_habit.py changed"
    return data.decode("utf-8")


def _pilot3_module() -> types.ModuleType:
    """Pilot 3's pinned file executed from its text in a fresh module (nothing imported from pilot 3's folder, no
    bytecode written there)."""
    mod = types.ModuleType("pilot3_habit_pinned")
    mod.__file__ = str(PILOT3_HABIT)
    sys.modules["pilot3_habit_pinned"] = mod  # dataclass/annotation lookups need the module registered
    try:
        exec(compile(_pilot3_source(), str(PILOT3_HABIT), "exec"), mod.__dict__)
    finally:
        sys.modules.pop("pilot3_habit_pinned", None)
    return mod


def _defs(source: str) -> dict:
    """{name: ast dump without docstrings} of the module's top-level functions and constant assignments."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                     and isinstance(node.body[0].value, ast.Constant)
                                     and isinstance(node.body[0].value.value, str)) else node.body
            out[node.name] = ast.dump(ast.Module(body=body, type_ignores=[])) + ast.dump(node.args)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.dump(node.value)
    return out


def test_the_tool_plan_s_constants():
    assert ph.EVAL_EVERY == 65_536 and ph.TD_BLOCK == 500_000 and ph.SCHEMA == "walking-pilot4-habit/v1"
    assert len(GRID) == 30 and GRID[-1] == 1_966_080 < BUDGET < GRID[-1] + ph.EVAL_EVERY


UNCHANGED_CONSTANTS = ("ACTUATORS", "GROUPS", "HABIT_GROUPS", "TAIL_PITCH", "BAR", "SATURATION_ABS", "FLAG_MIN",
                       "EPISODES_PER_EVAL", "PARK_MIN", "TRACE_SEEDS", "TRACE_WINDOW", "HORIZON", "FULL_MIN",
                       "POSSIBLE_SETTLE_MIN", "LEDGER_REQUIRED", "TD_COLUMN", "POSTURE_COLUMNS")
UNCHANGED_FUNCTIONS = ("_f", "actuator_index", "group_of", "read_ledger", "row_sets", "episode_state", "_alarms",
                       "trace_reading", "seed_reading", "sha256_file", "window_grid", "window_problems",
                       "refuse_existing_json", "_tool_sha")


def test_every_definition_is_pilot_3_s():
    """The TOOL PLAN: "definitions unchanged". Constants and functions that carry a definition are pilot 3's pinned
    file's, syntax tree for syntax tree (docstrings aside); the functions that changed are only the ones listed."""
    p3, p4 = _defs(_pilot3_source()), _defs(Path(ph.__file__).read_text(encoding="utf-8"))
    for name in UNCHANGED_CONSTANTS + UNCHANGED_FUNCTIONS:
        assert p4[name] == p3[name], name
    changed = sorted(n for n in set(p3) | set(p4) if p3.get(n) != p4.get(n))
    assert changed == ["EVAL_EVERY", "HEAD_TIP_REFS", "SCHEMA", "TD_BLOCK", "describe_ledger", "describe_trace",
                       "ledger_reading", "main", "roll_trace", "seed_problems"]


CODE_EDITS = {  # pilot 3's function text -> pilot 4's, the only code changes in the three changed functions
    "ledger_reading": [("!= 131_072]", "!= EVAL_EVERY]"),
                       ('"per_2M": {k: sum(v) / len(v) for k, v in blocks.items()}}',
                        '"block_steps": TD_BLOCK, "per_block": {k: sum(v) / len(v) for k, v in blocks.items()}}')],
    "roll_trace": [('artifact="pilot3_habit trace environment"', 'artifact="pilot4_habit trace environment"')],
    "seed_problems": [('"reads it for a check, never for Q4")',
                       '"reads it for a check of the tool, never as the seed\'s readout")')],
}


def test_the_changed_functions_change_only_what_the_plan_names():
    """ledger_reading: the grid gaps read EVAL_EVERY and the TD blocks are per TD_BLOCK (block_steps, per_block);
    roll_trace: the plant check's artifact label; seed_problems: the validation message's wording. Pilot 3's
    function text with exactly these edits has pilot 4's syntax tree (docstrings aside)."""
    p3 = _pilot3_source()
    d4 = _defs(Path(ph.__file__).read_text(encoding="utf-8"))
    for fn, pairs in CODE_EDITS.items():
        node = next(n for n in ast.parse(p3).body if isinstance(n, ast.FunctionDef) and n.name == fn)
        src = ast.get_source_segment(p3, node)
        assert _defs(src)[fn] != d4[fn], fn
        for old, new in pairs:
            assert src.count(old) == 1, (fn, old)
            src = src.replace(old, new)
        assert _defs(src)[fn] == d4[fn], fn


@pytest.mark.parametrize("evals", [
    {},  # quiet
    {GRID[3]: evaluation(GRID[3], 6), GRID[4]: evaluation(GRID[4], 6)},  # a TAIL ALARM
    {GRID[7]: evaluation(GRID[7], 5, "neck_pitch_act"), GRID[8]: evaluation(GRID[8], 9, "neck_yaw_act")},
    {GRID[12]: [row(GRID[12], ep, over("tail_3_pitch_act") if ep < 3 else (over("r_knee_act", 0.4, 1, 2) if ep < 6
                                                                             else QUIET)) for ep in range(10)],
     GRID[13]: [row(GRID[13], ep, over("tail_3_pitch_act") if ep < 3 else (over("r_knee_act", 0.4, 1, 2) if ep < 6
                                                                             else QUIET)) for ep in range(10)]},
    {GRID[20]: evaluation(GRID[20], 9, n_rows=9), GRID[21]: evaluation(GRID[21], 10, "l_knee_act")},
])
def test_pilot_3_s_reading_and_pilot_4_s_agree_but_for_the_grid_gaps_and_the_td_blocks(evals):
    p3 = _pilot3_module()
    header, rows = ledger(evals, extra_rows=evaluation(GRID[10], 7))  # a superseded row set too
    r3, r4 = p3.ledger_reading(header, rows, BUDGET), ph.ledger_reading(header, rows, BUDGET)
    assert p3.seed_reading(r3, None) == ph.seed_reading(r4, None)
    assert p3.seed_reading(r3, trace()) == ph.seed_reading(r4, trace())
    assert r3["grid_gaps"] != [] and r4["grid_gaps"] == []  # 65,536 apart: a gap on pilot 3's grid, none on ours
    t3, t4 = r3.pop("td"), r4.pop("td")
    r3.pop("grid_gaps"), r4.pop("grid_gaps")
    assert r3 == r4 and r4["superseded_row_sets"] == {str(GRID[10]): [10]}
    assert t3["overall"] == t4["overall"] and t3["evaluations"] == t4["evaluations"] == 30
    assert list(t3["per_2M"]) == ["1-2000000"] and t4["block_steps"] == 500_000
    assert list(t4["per_block"]) == ["1-500000", "500001-1000000", "1000001-1500000", "1500001-2000000"]


# ---------------------------------------------------------------- pilot 3's ledger cases on pilot 4's grid


def test_a_quiet_ledger_reads_no_flag_and_30_evaluations():
    lr = read()
    assert lr["window"] == {"first": 65_536, "last": 1_966_080, "evaluations": 30, "through": 2_000_000}
    assert all(lr["groups"][g]["alarms"] == [] and lr["groups"][g]["flagged"] == [] for g in ("TAIL", "NECK", "ANY"))
    assert lr["td"]["overall"] == pytest.approx(0.01) and lr["grid_gaps"] == []


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
    assert lr["groups"]["NECK"]["alarms"] == [second] == [131_072] and lr["groups"]["TAIL"]["alarms"] == []
    last2, last = GRID[-2], GRID[-1]
    lr = read({last2: evaluation(last2, 7), last: evaluation(last, 7)})
    assert lr["groups"]["TAIL"]["alarms"] == [last] == [1_966_080]


def test_evaluations_after_the_window_are_not_read():
    after1, after2 = 2_031_616, 2_097_152
    header, rows = ledger(extra_rows=evaluation(after1, 10) + evaluation(after2, 10))
    lr = ph.ledger_reading(header, rows, BUDGET)
    assert lr["groups"]["TAIL"]["alarms"] == [] and lr["after_window"] == [after1, after2]
    assert ph.ledger_reading(header, rows, None)["groups"]["TAIL"]["alarms"] == [after2]


def test_a_collapse_stopped_window_ends_at_its_newest_pair():
    """A run stopped after 1,750,000 (its newest intact pair) is read through it: the evaluations after it are not."""
    s1, s2 = 1_769_472, 1_835_008
    lr = read({s1: evaluation(s1, 6), s2: evaluation(s2, 6)}, through=1_750_000)
    assert lr["window"]["last"] == 1_703_936 and lr["groups"]["TAIL"]["alarms"] == []
    assert ph.window_problems(lr, 1_750_000) == []  # 46,064 short


def test_a_duplicated_row_set_after_a_resume_reads_the_last_written_set():
    s = GRID[20]
    earlier_flagged = evaluation(s, 8)
    lr = read({GRID[19]: evaluation(GRID[19], 8), s: earlier_flagged},
              extra_rows=evaluation(s, 0))  # the resume re-evaluated s: quiet
    assert lr["superseded_row_sets"] == {str(s): [10]}
    assert lr["groups"]["TAIL"]["flagged"] == [GRID[19]] and lr["groups"]["TAIL"]["alarms"] == []
    lr = read({GRID[19]: evaluation(GRID[19], 8), s: evaluation(s, 0)}, extra_rows=evaluation(s, 8))
    assert lr["groups"]["TAIL"]["alarms"] == [s] and lr["superseded_row_sets"] == {str(s): [10]}


def test_a_repeated_episode_index_inside_one_block_starts_a_new_set():
    s = GRID[20]
    lr = read({s: evaluation(s, 0) + evaluation(s, 9), GRID[21]: evaluation(GRID[21], 9)})
    assert lr["superseded_row_sets"] == {str(s): [10]} and lr["groups"]["TAIL"]["alarms"] == [GRID[21]]


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
    lr = ph.ledger_reading(h, rows, BUDGET)
    assert lr["read"] is False and "saturated_actuator_ties" in lr["not_run"][0]
    assert ph.seed_reading(lr, trace(quiet=True))["reading"] == "NOT RUN"


def test_td_per_500000_step_blocks():
    """TD per TD_BLOCK (500,000): the 30 evaluations fall 7, 8, 7 and 8 to the four blocks of the 2M run."""
    evals = {s: evaluation(s, 0, tail3=(0.1 if s <= 500_000 else (-0.3 if s <= 1_500_000 else 0.05))) for s in GRID}
    td = read(evals)["td"]
    assert td["block_steps"] == 500_000 and "per_2M" not in td
    assert td["per_block"]["1-500000"] == pytest.approx(0.1)
    assert td["per_block"]["500001-1000000"] == pytest.approx(0.3)
    assert td["per_block"]["1000001-1500000"] == pytest.approx(0.3)
    assert td["per_block"]["1500001-2000000"] == pytest.approx(0.05)
    assert td["evaluations"] == 30
    assert td["overall"] == pytest.approx((7 * 0.1 + 8 * 0.3 + 7 * 0.3 + 8 * 0.05) / 30)
    assert [sum(1 for s in GRID if (s - 1) // 500_000 == k) for k in range(4)] == [7, 8, 7, 8]
    text = "\n".join(ph.describe_ledger(read(evals), "x"))
    assert "per 500,000 steps: 1-500000 0.1000, 500001-1000000 0.3000" in text


def test_grid_gaps_are_read_against_65536():
    lr = read(grid=GRID[:5] + GRID[6:])  # one evaluation missing
    assert lr["grid_gaps"] == [(GRID[4], GRID[6])]
    assert any("evaluation steps not 65,536 apart" in line for line in ph.describe_ledger(lr, "x"))
    assert read(grid=[131_072 * k for k in range(1, 16)])["grid_gaps"] != []  # pilot 3's grid is not ours


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


def test_the_trace_s_head_tip_line_names_its_references_as_pilot_3_s():
    eps = [dict(episode(s), head_tip_z=[0.9] * 1000, neck_pitch=[10.0] * 1000, head_pitch=[5.0] * 1000)
           for s in ph.TRACE_SEEDS]
    lines = ph.describe_trace(trace(eps=eps))
    assert lines[-1].endswith("(window means; pilot 3's stance parent 0.904 m, the statue 0.959 m)")


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
    assert r["reading"] == "HABIT" and r["tail_habit"] and r["habits"][0]["text"] == "TAIL ALARM at 131,072"
    r = ph.seed_reading(alarm, None)  # a habit found is not undone by an unread trace
    assert r["reading"] == "HABIT" and r["not_run"] == ["no trace"]
    assert ph.seed_reading(read(), None)["reading"] == "NOT RUN"


def test_a_neck_alarm_is_a_neck_habit():
    s1, s2 = GRID[25], GRID[26]
    r = ph.seed_reading(read({s: evaluation(s, 6, "head_pitch_act") for s in (s1, s2)}), trace())
    assert r["reading"] == "HABIT" and r["neck_habit"] and not r["tail_habit"]


def test_a_leg_alarm_is_never_a_habit():
    s1, s2 = GRID[25], GRID[26]
    lr = read({s: evaluation(s, 10, "l_knee_act") for s in (s1, s2)})
    assert lr["groups"]["LEGS"]["alarms"] == [s2]
    assert ph.seed_reading(lr, trace())["reading"] == "NO HABIT"
    lines = ph.describe_ledger(lr, "x")
    assert any(line.startswith("  LEGS (not part of the seed reading): 2 flagged") for line in lines)


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
    r = subprocess.run([sys.executable, "-I", "-B", str(TOOLS / "pilot4_habit.py"), "ledger", str(path), "--through",
                        "2000000", "--json", str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc["reading"]["groups"]["TAIL"]["alarms"] == [131_072] and doc["tool_sha256"].startswith("sha256:")
    assert doc["schema"] == "walking-pilot4-habit/v1" and doc["report_only"] is True
    assert "TAIL: 2 flagged evaluations; 1 ALARMS (131,072-131,072)" in r.stdout
    assert "65,536-1,966,080" in r.stdout and "30 evaluations" in r.stdout


# ---------------------------------------------------------------- the seed mode's refusals (before any read)

HEAD = "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8"


def _binding(tmp_path=None, run="pilot4_walk_s42", **over):
    """A pilot cell's binding; its "run" is the judged copy tmp_path/<run> (compared by resolved path)."""
    where = None if run is None else str(tmp_path / run) if tmp_path is not None else f"/copies/{run}"
    b = {"seed": 42, "head": HEAD, "run": where, "validation_only": False, "off_schedule": False,
         "expected_budget": BUDGET, "steps": BUDGET}
    b.update(over)
    return b


def _ledger(tmp_path, run="pilot4_walk_s42", name="eval_episodes.csv"):
    path = tmp_path / run / "03_locomotion" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("timesteps\n")
    return path


def test_seed_accepts_a_pilot_cell_with_its_own_run_s_ledger(tmp_path):
    assert ph.seed_problems(_binding(tmp_path), seed=42, head=HEAD, dirty="", ledger=_ledger(tmp_path)) == []


def test_seed_refuses_a_retired_run_s_ledger_and_a_ledger_elsewhere(tmp_path):
    b = _binding(tmp_path, run="pilot4_walk_s42_b")
    got = ph.seed_problems(b, seed=42, head=HEAD, dirty="", ledger=_ledger(tmp_path))
    assert len(got) == 1 and "is not the judged run's own ledger" in got[0]
    assert str(tmp_path / "pilot4_walk_s42_b" / "03_locomotion" / "eval_episodes.csv") in got[0]
    other = tmp_path / "eval_episodes.csv"
    other.write_text("timesteps\n")
    assert len(ph.seed_problems(_binding(tmp_path), seed=42, head=HEAD, dirty="", ledger=other)) == 1
    got = ph.seed_problems(_binding(tmp_path, run=None), seed=42, head=HEAD, dirty="", ledger=_ledger(tmp_path))
    assert len(got) == 1 and "<no run recorded>" in got[0]


def test_seed_refuses_another_file_in_the_run_s_stage_folder(tmp_path):
    """The file name is part of the check; RUN/03_locomotion/other.csv is refused (pilot 3's FIX3)."""
    got = ph.seed_problems(_binding(tmp_path), seed=42, head=HEAD, dirty="", ledger=_ledger(tmp_path, name="other.csv"))
    assert len(got) == 1 and "is not the judged run's own ledger" in got[0]
    stage_dir_csv = tmp_path / "pilot4_walk_s42" / "eval_episodes.csv"
    stage_dir_csv.write_text("timesteps\n")
    assert len(ph.seed_problems(_binding(tmp_path), seed=42, head=HEAD, dirty="", ledger=stage_dir_csv)) == 1


def test_seed_refuses_a_same_named_copy_outside_the_judged_run(tmp_path):
    """A stale download with the judged run's folder name, elsewhere, is refused; the judged copy's own ledger, named
    through a relative or symlinked path, is accepted (pilot 3's FIX3)."""
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
                                    off_schedule_reason="OFF-SCHEDULE: resumed from a file that is not periodic"),
                           seed=42, head=HEAD, dirty="", ledger=led)
    assert len(got) == 1 and "off-schedule" in got[0]
    got = ph.seed_problems(_binding(tmp_path, off_schedule=True), seed=42, head=HEAD, dirty="", ledger=led,
                           validation=True)
    assert len(got) == 1 and "off-schedule" in got[0]
    val = _binding(tmp_path, validation_only=True, expected_budget=50_000)
    got = ph.seed_problems(val, seed=42, head=HEAD, dirty="", ledger=led)
    assert len(got) == 1 and "validation-only (run budget 50000)" in got[0]
    assert "never as the seed's readout" in got[0] and "Q4" not in got[0]
    assert ph.seed_problems(val, seed=42, head=HEAD, dirty="", ledger=led, validation=True) == []


def test_seed_lists_every_problem_at_once(tmp_path):
    got = ph.seed_problems(_binding(tmp_path, seed=52, head="31d02e4", validation_only=True), seed=42, head=HEAD,
                           dirty="x", ledger=tmp_path / "x.csv")
    assert len(got) == 5


# ---------------------------------------------------------------- the window must cover the deciding checkpoint


def test_a_full_window_covers_the_deciding_checkpoint():
    assert ph.window_problems(read(), BUDGET) == []  # 1,966,080, 33,920 short
    assert ph.window_problems(read(through=1_750_000), 1_750_000) == []  # collapse-stopped: 1,703,936


@pytest.mark.parametrize("steps, last, evals", [(500_000, 458_752, 7), (1_000_000, 983_040, 15),
                                               (1_500_000, 1_441_792, 22), (2_000_000, 1_966_080, 30)])
def test_the_four_judged_checkpoints_windows(steps, last, evals):
    """The judged checkpoints 0.5 / 1.0 / 1.5M (trajectory) and 2M (deciding): each window covers its checkpoint."""
    lr = read(through=steps)
    assert lr["window"]["first"] == 65_536 and lr["window"]["last"] == last and lr["window"]["evaluations"] == evals
    assert ph.window_problems(lr, steps) == [] and steps - last < ph.EVAL_EVERY


def test_a_stale_copy_cut_short_of_the_deciding_checkpoint_is_a_window_problem():
    """Pilot 3's third review's probe on pilot 4's grid: a ledger cut at 950,000 read through 2,000,000 must not be
    read."""
    cut = [s for s in GRID if s <= 950_000]
    lr = read(grid=cut)
    assert lr["window"]["last"] == 917_504 and lr["window"]["evaluations"] == 14
    got = ph.window_problems(lr, BUDGET)
    assert len(got) == 1 and "917,504, 1,082,496 steps short" in got[0]


def test_the_window_s_tail_boundary_is_one_grid_step():
    """Last evaluation 65,536 short (one evaluation missing at the end) is refused; 65,535 short is accepted."""
    lr = read(grid=GRID[:-1])  # last 1,900,544
    assert ph.window_problems(lr, 1_900_544 + 65_535) == []
    assert len(ph.window_problems(lr, 1_900_544 + 65_536)) == 1
    assert len(ph.window_problems(lr, BUDGET)) == 1  # 99,456 short


def test_a_copy_without_its_first_rows_is_a_window_problem():
    got = ph.window_problems(read(grid=GRID[1:]), BUDGET)
    assert len(got) == 1 and "first evaluation is 131,072, not 65,536" in got[0]
    got = ph.window_problems(read(grid=[]), BUDGET)
    assert len(got) == 1 and "no evaluation at or before" in got[0]
    pilot3_grid = [131_072 * k for k in range(1, 16)]  # a ledger on pilot 3's cadence is not pilot 4's
    assert len(ph.window_problems(read(grid=pilot3_grid), BUDGET)) == 1


@pytest.mark.parametrize("s", [250_000, 500_000, 750_000, 1_000_000, 1_250_000, 1_500_000, 1_750_000])
def test_a_resume_from_any_intact_pair_covers_the_checkpoint(s):
    """D11: a RESUME from any intact periodic pair (250,000 or later); its evaluations then fall at S + 65,536 k (SB3's
    EvalCallback counts from 0 in each learn() call), and the window still starts at 65,536 and ends within one grid
    step of 2,000,000. A ledger missing a column is not refused here (it reads NOT RUN)."""
    grid = [g for g in GRID if g <= s] + [s + 65_536 * k for k in range(1, 40) if s + 65_536 * k <= 2_015_232]
    lr = read(grid=grid)
    assert ph.window_problems(lr, BUDGET) == [] and lr["window"]["first"] == 65_536
    assert BUDGET - lr["window"]["last"] < 65_536
    lr = read(header=[c for c in HEADER if c != "saturated_actuator_ties"])
    assert lr["read"] is False and ph.window_problems(lr, BUDGET) == []


def test_the_window_grid_is_65536_and_the_run_s_own_only_for_a_validation_read():
    cad = {"curriculum_cadence": {"eval_freq": 32_768}}
    assert ph.EVAL_EVERY == 65_536
    assert ph.window_grid(dict(_binding(), **cad)) == 65_536
    assert ph.window_grid(dict(_binding(), **cad), validation=True) == 65_536  # not validation-only
    val = dict(_binding(validation_only=True), **cad)
    assert ph.window_grid(val) == 65_536 and ph.window_grid(val, validation=True) == 32_768
    for bad in (None, 0, -5, True, "32768"):
        assert ph.window_grid(dict(val, curriculum_cadence={"eval_freq": bad}), validation=True) == 65_536
    smoke = read(grid=[32_768, 65_536])
    assert ph.window_problems(smoke, 65_536, 32_768) == [] and len(ph.window_problems(smoke, 65_536)) == 1


def _seed_main_setup(tmp_path, monkeypatch, grid):
    """A judged cell, its run copy holding a ledger on `grid`, the checkpoint pair, and stand-ins for the tree
    (environments' location, git's HEAD and a clean status) and for the trace (20 quiet full-horizon episodes; it
    records that it was rolled). No simulation and no git command runs."""
    run = tmp_path / "copies" / "pilot4_walk_s42"
    header, rows = ledger(grid=grid)
    lp = run / "03_locomotion" / "eval_episodes.csv"
    lp.parent.mkdir(parents=True)
    with lp.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    models = run / "03_locomotion" / "models"
    models.mkdir()
    (models / "stage2_2000000_steps.zip").write_bytes(b"zip")
    (models / "stage2_vecnormalize_2000000_steps.pkl").write_bytes(b"pkl")
    cell = tmp_path / "OUT" / "s42" / "2000000"
    cell.mkdir(parents=True)
    b = _binding(tmp_path / "copies", checkpoint=str(models / "stage2_2000000_steps.zip"),
                 vecnorm=str(models / "stage2_vecnormalize_2000000_steps.pkl"))
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
    """Through main: the judged copy's own ledger, stale (cut at 950,000), is refused after reading and before the
    trace; nothing is written."""
    lp, cell, rolled = _seed_main_setup(tmp_path, monkeypatch, [s for s in GRID if s <= 950_000])
    out = tmp_path / "HABIT" / "s42.json"
    out.parent.mkdir()
    rc = ph.main(["seed", "--seed", "42", "--ledger", str(lp), "--cell", str(cell), "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 2 and not out.exists() and rolled == []
    assert "refusing (nothing written): the ledger's last evaluation" in text and "1,082,496 steps short" in text


def test_seed_main_reads_a_covering_window_and_records_the_check(tmp_path, monkeypatch, capsys):
    lp, cell, rolled = _seed_main_setup(tmp_path, monkeypatch, GRID)
    out = tmp_path / "s42.json"
    rc = ph.main(["seed", "--seed", "42", "--ledger", str(lp), "--cell", str(cell), "--json", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and len(rolled) == 1, text
    doc = json.loads(out.read_text())
    assert doc["reading"] == "NO HABIT" and doc["window_check"] == {
        "grid": 65_536, "deciding_steps": 2_000_000, "first": 65_536, "last": 1_966_080, "problems": []}
    assert doc["ledger_reading"]["window"]["evaluations"] == 30 and doc["validation_only"] is False
    assert doc["schema"] == "walking-pilot4-habit/v1" and doc["report_only"] is True
    assert doc["run"] == "pilot4_walk_s42" and doc["deciding_checkpoint"]["steps"] == 2_000_000
    assert "s42 HABIT READING (report only; it picks nothing): NO HABIT" in text


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
