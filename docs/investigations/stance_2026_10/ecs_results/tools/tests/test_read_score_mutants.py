"""Lane R tests: mutants of read_ecs.py and score_ecs.py are killed by the batteries (build list v2 item 9).

Each mutant is a copy of the tool with one text replacement, written under <ECS>/tmp/lane_r_tests/mutants/ and loaded
as its own module; the real tools stay as they are.  A mutant is killed when a battery that passes on the real tool
raises on the mutant."""
import importlib.util
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load("test_read_score_fixtures")
TR = _load("test_read_score_reader")
TS = _load("test_read_score_scorer")
READ = F.TOOLS / "read_ecs.py"
SCORE = F.TOOLS / "score_ecs.py"

READER_MUTANTS = {
    "window_lower_edge": ("    def inwin(s):\n        return S < s <= T",
                          "    def inwin(s):\n        return S <= s <= T", TR.battery_window),
    "window_upper_edge": ("    def inwin(s):\n        return S < s <= T", "    def inwin(s):\n        return S < s < T",
                          TR.battery_window),
    "alarm_straddle_counted": ('any_alarms=[s for s in any_alarms if S < s - GRID and s <= T]',
                               'any_alarms=[s for s in any_alarms if S < s and s <= T]', TR.battery_window),
    "tail_missing_counts": ('(("tail", C6_TAIL_RE)', '(("other", C6_TAIL_RE)', TR.battery_window),
    "flag_threshold": ("if int(e.get(\"any\", 0)) >= FLAG_ROWS) != sorted(any_all)",
                       "if int(e.get(\"any\", 0)) >= FLAG_ROWS + 1) != sorted(any_all)", TR.battery_window),
    "r4_tolerance": ("bad_ent = [i for i in range(n_exp) if not (ent[i] == want_ent[i])]",
                     "bad_ent = [i for i in range(n_exp) if not (abs(ent[i] - want_ent[i]) <= 1e-6)]", TR.battery_r4),
    "r4_arm_b_as_a": ('want_ent = p_ent if arm == "A" else np.zeros_like(p_ent)', "want_ent = p_ent", TR.battery_r4),
    "r4_row_count": ("    n_exp = E.expected_update_rows(S, T)\n    steps",
                     "    n_exp = E.expected_update_rows(S, T) + 1\n    steps", TR.battery_r4),
    "r4_end_rollout_ignored": ("    if late:\n        fails.append", "    if False:\n        fails.append",
                               TR.battery_r4),
    "r4_lr_unchecked": ("bad_lr = [i for i in range(n_exp) if not (lr[i] == p_lr[i])]", "bad_lr = []", TR.battery_r4),
    "c6_rows_as_grid": ('("rows", C6_ROWS_RE)', '("grid", C6_ROWS_RE)', TR.battery_c6),
}

SCORER_MUTANTS = {
    "side_0.05": ("SIDE = Fraction(str(E.ALPHA_PER_SIDE))", "SIDE = Fraction(1, 20)", TS.battery_te),
    "p_up_strict": ("up = sum((p for v, p in d.items() if v >= t), Fraction(0))",
                    "up = sum((p for v, p in d.items() if v > t), Fraction(0))", TS.battery_te),
    "fill_reversed": ('up = (a, nA + miss["A"], b + miss["B"], nB + miss["B"])',
                      'up = (a + miss["A"], nA + miss["A"], b, nB + miss["B"])', TS.battery_chk8),
    "censored_as_counted": ('missing = c["cls"] == LOST or c["outcome"] == CENSORED', 'missing = c["cls"] == LOST',
                            TS.battery_fill),
    "p_down_on_up_table": ("p_down = pvals(down_cells)[1]", "p_down = pvals(up_cells)[1]", TS.battery_fill),
    "low_note_at_half": ("elif 2 * a < nA:", "elif 2 * a <= nA:", TS.battery_lines),
    "addition_from_filled": ("        if c_up <= SIDE:\n", "        if p_up <= SIDE:\n", TS.battery_lines),
    "not_informative_one_start": ("all(low_note(notes[s]) for s in START_ORDER)",
                                  "any(low_note(notes[s]) for s in START_ORDER)", TS.battery_lines),
    "stratum_off_by_one": ("for a in range(max(0, m - nB), min(nA, m) + 1))",
                           "for a in range(max(0, m - nB), min(nA, m)))", TS.battery_brute),
}


def test_batteries_pass_on_the_real_tools():
    for battery in (TR.battery_window, TR.battery_r4, TR.battery_c6):
        assert battery(F.R)
    for battery in (TS.battery_te, TS.battery_chk8, TS.battery_fill, TS.battery_lines, TS.battery_brute):
        assert battery(F.S)


@pytest.mark.parametrize("key", sorted(READER_MUTANTS))
def test_reader_mutant_killed(key):
    old, new, battery = READER_MUTANTS[key]
    m = F.load_mutant(f"read_ecs_mut_{key.replace('.', '_')}", READ, old, new)
    with pytest.raises(Exception):  # noqa: B017 - any failure of the battery kills the mutant
        battery(m)


@pytest.mark.parametrize("key", sorted(SCORER_MUTANTS))
def test_scorer_mutant_killed(key):
    old, new, battery = SCORER_MUTANTS[key]
    m = F.load_mutant(f"score_ecs_mut_{key.replace('.', '_')}", SCORE, old, new)
    m.pvals.cache_clear()
    m.stratum_null.cache_clear()
    with pytest.raises(Exception):  # noqa: B017 - any failure of the battery kills the mutant
        battery(m)
