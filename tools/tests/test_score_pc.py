"""PC scorer and adapted tools: tests on synthetic roots (synth_pc.py) and unit tests of the PC readings.
PC AMENDMENT (2026-11-01): THE ONE TEST is phase vs the concurrent switch-off runs (the off values here are STANCE6's
measured TDs, used as numbers only), MISSING ranks, the switch-off arm's launch check / C4 (p) / C8, six-run roots,
outcome blindness, and STANCE6 report only.

    cd /tmp && PY -I -B -m pytest -q -p no:cacheprovider ST/tools/tests/test_score_pc.py
"""
from __future__ import annotations

import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_pc as S  # noqa: E402

C, CP, SP = S.C, S.CP, S.SP
P = SP.P
SUFFIX = "(descriptive: 3 phase-critic runs beside 3 concurrent switch-off runs; KNOWN LIMITS)"
COMP_TD = [0.38822738704353094, 0.6353434591453841, 0.15433899719014801]      # s60, s42, s80 [measured, comparators]
LOWER_TEXT = "LOWER THAN EVERY CONCURRENT SWITCH-OFF RUN (exact permutation p = 1/20)"
NO_RAIL_BESIDE = "; report only beside it: (2/3)^3 = 0.296 at STANCE6's 1 of 3 and (3/5)^3 = 0.216 at LTSB r9's 2 of 5"


# ================================================================================================ THE ONE TEST
LO, MID, HI = sorted(COMP_TD)


@pytest.mark.parametrize("pc,k", [                   # DESIGN/comparators.out's placements, the same values
    ([LO * 0.1, LO * 0.2, LO * 0.3], 1),
    ([LO * 0.1, LO * 0.2, (LO + MID) / 2], 2),
    ([LO * 0.1, LO * 0.2, HI + 0.1], 6),
    ([LO * 0.1, HI + 0.1, HI + 0.2], 14),
    ([HI + 0.1, HI + 0.2, HI + 0.3], 20),
])
def test_one_test_placements(pc, k):
    r = P.one_test(pc, COMP_TD)
    assert r["k"] == k and r["n_splits"] == 20 and r["p"] == f"{k}/20"
    assert r["lower"] is (k == 1)
    assert r["text"] == (LOWER_TEXT if k == 1 else f"NOT SEPARATED (p = {k}/20)")
    assert r["caveat"] == P.CAVEAT and "conservative" in r["caveat"] and "A5" in r["caveat"]
    assert "concurrent" in r["caveat"] and "GPU nondeterminism" in r["caveat"]
    assert r["note"] == ("p = 1/20 only on complete separation: every phase TD below every switch-off TD; TD's form (the "
                         "whole-run mean; 2 windows looked at) was chosen with STANCE6's values in view, the six runs' "
                         "values are unseen at the pin")
    assert r["phase_missing"] == [False] * 3 and r["off_missing"] == [False] * 3 and r["off"] == COMP_TD


def test_one_test_against_an_exact_oracle():
    """An independent oracle: exact rational sums (Fraction of each float), every one of the 20 splits."""
    import itertools
    import random
    from fractions import Fraction

    rng = random.Random(3042)
    for _ in range(300):
        pc = [round(rng.uniform(0, 1), rng.choice((2, 6))) for _ in range(3)]
        vals = [Fraction(v) for v in pc + COMP_TD]
        obs = sum(vals[:3])
        k = sum(1 for idx in itertools.combinations(range(6), 3) if sum(vals[i] for i in idx) <= obs)
        assert P.one_test(pc, COMP_TD)["k"] == k, pc


def test_one_test_near_ties_are_exact():
    """Fixer I1: every PC TD strictly below s80's reads k = 1, even one ulp below (math.fsum misordered these)."""
    import itertools
    import random
    from fractions import Fraction

    lo = min(COMP_TD)
    near = math.nextafter(lo, 0.0)
    for a, b in ((0.0, 0.0), (0.01, 0.02), (near, near), (lo / 3, lo / 7), (0.1, 0.12)):
        assert P.one_test([near, a, b], COMP_TD)["k"] == 1, (a, b)
    rng = random.Random(3043)
    for _ in range(2000):
        base = rng.choice(COMP_TD)
        pc = [math.nextafter(base, rng.choice((0.0, 1.0))), rng.choice(COMP_TD) * rng.choice((1.0, 0.5)),
              math.nextafter(rng.choice(COMP_TD), 0.0)]
        rng.shuffle(pc)
        vals = [Fraction(v) for v in pc + COMP_TD]
        obs = sum(vals[:3])
        k = sum(1 for idx in itertools.combinations(range(6), 3) if sum(vals[i] for i in idx) <= obs)
        assert P.one_test(pc, COMP_TD)["k"] == k, pc


def test_one_test_separation_bar_is_the_lowest_off_td():
    lo = min(COMP_TD)
    assert P.one_test([lo - 1e-9, 0.0, 0.0], COMP_TD)["k"] == 1
    # a phase value exactly at the lowest switch-off TD ties: ties count against H1 (2 splits reach the phase sum)
    assert P.one_test([lo, 0.0, 0.0], COMP_TD)["k"] == 2


def _oracle_k(phase, off, M=10 ** 9):
    """An independent oracle of the MISSING rule: +infinity -> +M, -infinity -> -M (M far beyond every finite TD, the
    limit the rule names), exact Fractions over every one of the 20 splits."""
    import itertools
    from fractions import Fraction

    def f(v):
        return Fraction(M) if v == math.inf else (Fraction(-M) if v == -math.inf else Fraction(v))
    vals = [f(v) for v in list(phase) + list(off)]
    obs = sum(vals[:3])
    return sum(1 for idx in itertools.combinations(range(6), 3) if sum(vals[i] for i in idx) <= obs)


def test_one_test_missing_phase_enters_highest_never_helps():
    r = P.one_test([0.0, 0.0, math.inf], COMP_TD)
    assert r["k"] == _oracle_k([0.0, 0.0, math.inf], COMP_TD) == 11
    assert r["phase_missing"] == [False, False, True] and r["phase"][2] is None and not r["lower"]
    assert P.one_test([math.inf] * 3, COMP_TD)["k"] == 20


def test_one_test_missing_off_enters_lowest_never_helps():
    """A missing switch-off run enters at -infinity (the lowest rank): complete separation is then impossible."""
    r = P.one_test([0.0, 0.0, 0.0], [COMP_TD[0], COMP_TD[1], -math.inf])
    assert r["k"] == _oracle_k([0.0, 0.0, 0.0], [COMP_TD[0], COMP_TD[1], -math.inf]) == 11
    assert r["off_missing"] == [False, False, True] and r["off"][2] is None and not r["lower"]
    assert P.one_test([0.0] * 3, [-math.inf] * 3)["k"] == 20


def test_one_test_missing_ranks_against_an_oracle():
    """Every mix of missing runs (each side 0-3 missing, random finite values) against the +M/-M oracle; any missing
    run gives k >= 11 (never LOWER)."""
    import itertools
    import random

    rng = random.Random(3044)
    for _ in range(400):
        ph = [rng.choice((math.inf, round(rng.uniform(0, 1), 3))) for _ in range(3)]
        of = [rng.choice((-math.inf, round(rng.uniform(0, 1), 3))) for _ in range(3)]
        r = P.one_test(ph, of)
        assert r["k"] == _oracle_k(ph, of), (ph, of)
        if any(not math.isfinite(v) for v in ph + of):
            assert r["k"] >= 11 and not r["lower"], (ph, of, r["k"])
    for ph, of in itertools.product(([0.1, 0.2, 0.3],), ([0.5, 0.6, 0.7],)):
        assert P.one_test(ph, of)["k"] == 1


def test_one_test_refuses_wrong_side_infinities_and_nan():
    with pytest.raises(ValueError):
        P.one_test([0.1, 0.2, -math.inf], COMP_TD)
    with pytest.raises(ValueError):
        P.one_test([0.1, 0.2, 0.3], [0.4, 0.5, math.inf])
    with pytest.raises(ValueError):
        P.one_test([0.1, 0.2, float("nan")], COMP_TD)


def test_one_test_needs_three_and_three():
    with pytest.raises(ValueError):
        P.one_test([0.1, 0.2], COMP_TD)
    with pytest.raises(ValueError):
        P.one_test([0.1, 0.2, 0.3], COMP_TD[:2])


# ================================================================================================ TD, early signs
def _rows(spec):
    """spec: {step: [(cmd or None, frac, actuator), ...]}"""
    out = []
    for s, rs in spec.items():
        for i, (cmd, frac, act) in enumerate(rs):
            out.append({"timesteps": str(s), "episode": str(i), "tail_3_pitch_act_command_mean": "" if cmd is None else str(cmd),
                        "saturated_actuator_fraction": str(frac), "saturated_actuator": act, "return": "4400",
                        "length": "1000", "screen_v2_clean": "1"})
    return out


def test_tail_drive_skips_missing_and_blank_cells():
    rows = _rows({131072: [(0.2, 0, ""), (-0.4, 0, "")], 262144: [(None, 0, ""), ("nan", 0, "")],
                  4194304: [(0.9, 0, ""), (None, 0, "")]})
    per = P.per_evaluation(rows)
    td = P.tail_drive(per)
    assert per[131072]["cmd_l"] == pytest.approx(-0.1) and per[262144]["cmd_l"] is None
    assert td["missing"] == [262144] and td["n"] == 2
    assert td["td"] == pytest.approx((0.1 + 0.9) / 2) and td["td_j"] == pytest.approx(0.9) and td["n_j"] == 1


def test_early_signs_firsts():
    rows = _rows({131072: [(0.31, 0, ""), (0.31, 0.5, "neck_pitch_act")],
                  262144: [(0.65, 0.5, "tail_2_pitch_act")],
                  393216: [(0.95, 0.5, "tail_3_pitch_act")],
                  524288: [(0.61, 0.05, "tail_3_pitch_act")]})
    e = P.early_signs(P.per_evaluation(rows), {"tail_flagged": [393216], "tail_alarms": []})
    assert e["firsts"] == {"first_t3_over": 393216, "FOB_tail": 262144, "first_over_any": 131072,
                           "cmd_l_ge_0.30": 131072, "cmd_l_ge_0.60": 262144, "cmd_l_ge_0.90": 393216}
    assert e["n_cmd_l_ge_060"] == 3 and e["cmd_l_max"] == (393216, 0.95) and e["cmd_l_last"] == (524288, 0.61)
    assert e["m5_tail_flagged"] == [393216] and e["m5_tail_alarms"] == []


# ================================================================================================ HABIT ELSEWHERE
def _m1(cmd=None, over=None, length=1000):
    """An m1_summary.json-like record: cmd {a: CMD}, over {episode: (actuator, sign str)}."""
    eps = []
    for i in range(20):
        acts = {n: {"over_bar": False, "sign": None} for n in C.ACTUATOR_NAMES}
        if over and i in over:
            a, sg = over[i]
            acts[C.ACTUATOR_NAMES[a]] = {"over_bar": True, "sign": sg}
        eps.append({"length": length, "full_horizon": length >= 1000, "actuators": acts})
    return {"cmd_all_actuators": {str(a): (cmd or {}).get(a, 0.0) for a in range(15)}, "episodes": eps}


def _habit(rows=(), classes=None, m1=None, header=None):
    js = list(C.J_STEPS)
    return P.habit_elsewhere(C, header or S.LEDGER_COLS, list(rows), classes or {}, m1 or {}, js, C.R1_STEP)


def test_habit_free_when_nothing_holds():
    m1 = {str(j): _m1() for j in C.J_STEPS} | {"final": _m1()}
    h = _habit(m1=m1)
    assert h["status"] == "FREE" and h["saturating"] == [] and h["drive_only"] == []
    assert "tail_3_pitch_act" not in h["per_actuator"] and len(h["per_actuator"]) == 14


def test_habit_t_site_and_tie_counts_in_each_sign():
    m1 = {"final": _m1(over={0: (0, "+"), 1: (0, "=")}), "4194304": _m1(over={0: (2, "+"), 1: (2, "-")})}
    h = _habit(m1=m1)
    assert h["per_actuator"]["neck_pitch_act"]["sites"] == ["T"] and h["per_actuator"]["neck_pitch_act"]["status"] == "SATURATING"
    assert h["per_actuator"]["head_pitch_act"]["sites"] == []          # one + and one -: no sign reaches 2
    assert h["status"] == "SATURATING" and h["saturating"] == ["neck_pitch_act"]


def test_habit_t_site_at_a_j_only_label():
    """D6 (R11): the T site is read at every traced checkpoint, a J label included."""
    m1 = {"5242880": _m1(over={0: (0, "+"), 1: (0, "+")}), "final": _m1()}
    v = _habit(m1=m1)["per_actuator"]["neck_pitch_act"]
    assert v["sites"] == ["T"] and v["trace_sites"] == ["5242880"]


def test_habit_t_needs_full_horizon_episodes():
    rec = _m1(over={0: (0, "+"), 1: (0, "+")}, length=600)
    assert _habit(m1={"final": rec})["per_actuator"]["neck_pitch_act"]["sites"] == []


def test_habit_d_site_j_hits_and_mean():
    m1 = {"4194304": _m1(cmd={2: -0.61}), "5242880": _m1(cmd={2: -0.60}), "final": _m1(cmd={2: 0.0})}
    h = _habit(m1=m1)
    assert h["per_actuator"]["head_pitch_act"]["sites"] == ["D"] and h["per_actuator"]["head_pitch_act"]["status"] == "DRIVE ONLY"
    assert h["status"] == "DRIVE ONLY" and h["drive_only"] == ["head_pitch_act"]
    # the mean of CMD(R1) and CMD(R2): one formed (R1 absent) at 0.60
    h2 = _habit(m1={"final": _m1(cmd={13: 0.60})})
    assert h2["per_actuator"]["tail_2_pitch_act"]["sites"] == ["D"]
    h3 = _habit(m1={"final": _m1(cmd={13: 0.70}), str(C.R1_STEP): _m1(cmd={13: 0.49})})   # mean 0.595
    assert h3["per_actuator"]["tail_2_pitch_act"]["sites"] == []


def test_habit_l_and_a_sites():
    over = S.row("leg", sign="+")                          # r_hip_pitch_act (3) over the bar, resolved
    led = S.clean_ledger()
    s1, s2 = C.GRID_STEPS[10], C.GRID_STEPS[11]
    led[s1] = S.evaluation([(over, 5), (S.row("clean"), 5)])
    led[s2] = S.evaluation([(over, 5), (S.row("clean"), 5)])
    header, rows = P.ledger_rows(S.ledger_bytes(led))
    classes = {"any_alarms": [{"step": s2, "actuators": {"r_hip_pitch_act": 10}}]}
    h = _habit(rows, classes, header=header)
    v = h["per_actuator"]["r_hip_pitch_act"]
    assert v["sites"] == ["L", "A"] and v["own_alarms"] == [s2] and v["any_alarms_naming"] == [s2]
    # four over-bar episodes: no own alarm, and A alone still saturates
    led[s1] = S.evaluation([(over, 4), (S.row("clean"), 6)])
    header, rows = P.ledger_rows(S.ledger_bytes(led))
    v = _habit(rows, classes, header=header)["per_actuator"]["r_hip_pitch_act"]
    assert v["sites"] == ["A"] and v["status"] == "SATURATING"


def test_habit_a_site_needs_five_rows_in_one_evaluation():
    """Fixer D3: an ANY ALARM mostly on tail_3 with one neck row per evaluation does not make the neck SATURATING;
    five neck rows in one of the two evaluations do."""
    t3, neck = S.row("tail", 14, "+"), S.row("tail", 0, "+")        # act 0 = neck_pitch_act
    led = S.clean_ledger()
    s1, s2 = C.GRID_STEPS[10], C.GRID_STEPS[11]
    led[s1] = S.evaluation([(t3, 4), (neck, 1), (S.row("clean"), 5)])
    led[s2] = S.evaluation([(t3, 4), (neck, 1), (S.row("clean"), 5)])
    header, rows = P.ledger_rows(S.ledger_bytes(led))
    classes = {"any_alarms": [{"step": s2, "actuators": {"tail_3_pitch_act": 8, "neck_pitch_act": 2}}]}
    v = _habit(rows, classes, header=header)["per_actuator"]["neck_pitch_act"]
    assert v["sites"] == [] and v["status"] == "FREE" and v["any_alarms_naming"] == []
    led[s1] = S.evaluation([(neck, 5), (S.row("clean"), 5)])
    header, rows = P.ledger_rows(S.ledger_bytes(led))
    classes = {"any_alarms": [{"step": s2, "actuators": {"tail_3_pitch_act": 4, "neck_pitch_act": 6}}]}
    v = _habit(rows, classes, header=header)["per_actuator"]["neck_pitch_act"]
    assert v["sites"] == ["A"] and v["any_alarms_naming"] == [s2] and v["status"] == "SATURATING"


# ================================================================================================ THE READING (unit)
def _run(seed, cls="CLEAR", gate="PASS", habit="FREE", tail_alarms=(), kind=None, verdict="COMPLETE", trace_rail=()):
    d = dict(seed=seed, run=f"{C.RUN_ID_PREFIX}{seed}", verdict=verdict)
    if verdict == "COMPLETE":
        d["classes"] = dict(cls=cls, kind=kind, tail_alarms=list(tail_alarms), gate=gate,
                            gate_why=None if gate == "PASS" else ["heading +90: x"], trace_rail=list(trace_rail))
        d["habit"] = dict(status=habit, saturating=["neck_pitch_act"] if habit == "SATURATING" else [])
    return d


def test_reading_line4_text_exact():
    rd = P.reading([_run(60), _run(42), _run(80)], [])
    assert rd["line"] == 4
    assert rd["text"] == ("NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, THE STANCE GATE CERTIFICATE PASSED IN 3 OF 3 "
                          "(GATE CERTIFICATE: final checkpoint, measured on a screen task; not certification), NO "
                          "SATURATING HABIT ON ANOTHER ACTUATOR: CONSISTENT WITH THE PHASE CRITIC REMOVING THE HABIT, NOT "
                          "CONFIRMATORY " + SUFFIX)
    assert rd["no_rail_chance"] == ("[model, plug-in] if the phase critic changed nothing, no RAIL in 3 runs has chance "
                                    "not formed (no switch-off run read to a class)" + NO_RAIL_BESIDE)
    assert rd["driven_tails"] == [] and rd["off"] is None and rd["off_text"] == "CONCURRENT SWITCH-OFF RUNS: not given"


def test_reading_driven_tail_is_printed_in_lines_4_and_5():
    """Fixer D4: a tail actuator held hard negative (DRIVEN, report only) is named in the line, which stays as ROUND's
    classes put it."""
    r42 = _run(42)
    r42["classes"]["driven"] = [dict(actuator="tail_3_pitch_act", mean=-0.7)]
    rd = P.reading([_run(60), r42, _run(80)], [])
    assert rd["line"] == 4 and ", WITH A DRIVEN TAIL (tail_3_pitch_act MEAN_a -0.7000 IN s42): CONSISTENT" in rd["text"]
    rd = P.reading([_run(60, gate="FAIL"), r42, _run(80)], [])
    assert rd["line"] == 5 and rd["text"] == ("NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, WITH: 1 GATE FAIL, WITH A "
                                              "DRIVEN TAIL (tail_3_pitch_act MEAN_a -0.7000 IN s42) " + SUFFIX)
    assert rd["no_rail_chance"] is not None


def test_reading_lines_1_and_2_print_no_no_rail_chance():
    rd = P.reading([_run(60), _run(42, cls="RAIL", kind="RAIL-LEDGER-PERSISTENT", tail_alarms=[3932160]), _run(80)], [])
    assert rd["line"] == 1 and rd["no_rail_chance"] is None
    rd = P.reading([_run(60, verdict="LOST"), _run(42), _run(80)], [])
    assert rd["line"] == 2 and rd["no_rail_chance"] is None
    assert P.reading([_run(60, cls="PARKED"), _run(42), _run(80)], [])["no_rail_chance"] is not None


def test_reading_line2_before_line3():
    """D6 (R4): an incomplete seed with a PARKED run reads line 2, not line 3."""
    rd = P.reading([_run(60, verdict="LOST"), _run(42, cls="PARKED"), _run(80)], [])
    assert rd["line"] == 2 and rd["u"] == 1 and rd["p"] == 1


def test_reading_ledger_basis_before_presence():
    """D6 (R7): a COMPLETE ledger rail and a PRESENCE rail read "(ledger)"."""
    rd = P.reading([_run(60, verdict="LOST"), _run(42, cls="RAIL", kind="RAIL-LEDGER-PERSISTENT", tail_alarms=[3932160]),
                    _run(80)], [dict(run="x", detail={"step": 5})])
    assert rd["line"] == 1 and rd["r"] == 2 and "(ledger):" in rd["text"] and "LOST run" not in rd["text"]


def test_reading_line1_ledger_and_bases():
    rd = P.reading([_run(60), _run(42, cls="RAIL", kind="RAIL-LEDGER-PERSISTENT", tail_alarms=[3932160]), _run(80)], [])
    assert rd["line"] == 1 and rd["r"] == 1
    assert rd["text"] == ("THE TAIL RAIL FORMED IN 1 OF 3 PHASE-CRITIC RUNS (ledger): THE CRITIC'S PHASE INPUT FROM STEP 0 "
                          "DID NOT KEEP EVERY RUN OFF THE TAIL RAIL " + SUFFIX)
    assert rd["first_tail_alarms"][0]["first_tail_alarm"] == 3932160
    assert rd["first_tail_alarms"][0]["beside_off_same_seed"] == "not read"            # no switch-off runs given
    assert "beside_stance6_s42" not in rd["first_tail_alarms"][0]                    # STANCE6 is report only
    rd = P.reading([_run(60, verdict="LOST"), _run(42), _run(80)], [dict(run="x", detail={"step": 5})])
    assert rd["line"] == 1 and "(ledger, a LOST run)" in rd["text"] and rd["u"] == 1
    rd = P.reading([_run(60, cls="RAIL", kind="RAIL-END-ONLY", trace_rail=[{"drive_backed": True}]), _run(42), _run(80)], [])
    assert "(end-only, drive-backed)" in rd["text"]
    rd = P.reading([_run(60, cls="RAIL", kind="RAIL-END-ONLY", trace_rail=[{"drive_backed": False}]), _run(42), _run(80)], [])
    assert "(end-only, not drive-backed: weak evidence;" in rd["text"]


def test_reading_line2_3_5_texts_exact():
    rd = P.reading([_run(60, verdict="LOST"), _run(42, cls="UNREAD", gate="FAIL"), _run(80)], [])
    assert rd["line"] == 2 and rd["u"] == 2
    assert rd["text"] == ("INCOMPLETE: NO TAIL RAIL IN THE 1 PHASE-CRITIC RUN(S) READ TO 12M; 2 NOT READ (LOST OR STOPPED "
                          "WITHOUT A RAIL): NOT A NO-RAIL READING " + SUFFIX)
    rd = P.reading([_run(60, cls="PARKED"), _run(42, cls="PARKED", gate="FAIL"), _run(80)], [])
    assert rd["line"] == 3 and rd["text"] == ("NO TAIL RAIL; THE TAIL DRIVE PERSISTS: PARKED IN 2 OF 3 PHASE-CRITIC RUNS " + SUFFIX)
    rd = P.reading([_run(60, gate="FAIL"), _run(42), _run(80)], [])
    assert rd["line"] == 5 and rd["text"] == ("NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, WITH: 1 GATE FAIL " + SUFFIX)
    rd = P.reading([_run(60, habit="SATURATING"), _run(42, habit="DRIVE ONLY"), _run(80)], [])
    assert rd["line"] == 5 and rd["hs"] == 1 and rd["hd"] == 1
    assert rd["text"] == ("NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, WITH: 1 WITH A SATURATING HABIT ON ANOTHER "
                          "ACTUATOR " + SUFFIX)
    rd = P.reading([_run(60, gate="FAIL", habit="SATURATING"), _run(42, gate="FAIL"), _run(80)], [])
    assert rd["text"].startswith("NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, WITH: 2 GATE FAIL, 1 WITH A SATURATING")
    assert any("GATE FAIL" in x for x in rd["named"]) and any("SATURATING habit" in x for x in rd["named"])
    # a DRIVE ONLY habit alone keeps line 4
    assert P.reading([_run(60, habit="DRIVE ONLY"), _run(42), _run(80)], [])["line"] == 4


def _off(seed, cls="CLEAR", gate="PASS", habit="FREE", tail_alarms=(), verdict="COMPLETE"):
    d = _run(seed, cls=cls, gate=gate, habit=habit, tail_alarms=tail_alarms, verdict=verdict)
    d["run"] = f"{C.ARM_PREFIX['off']}{seed}"
    return d


def test_reading_off_runs_are_printed_beside_and_move_no_line():
    """PC AMENDMENT: the switch-off runs' classes never move the phase runs' line; they are counted beside, each line-1
    run's first TAIL ALARM is printed beside its same-seed switch-off run's, and the no-RAIL chance uses their share."""
    ph = [_run(60), _run(42), _run(80)]
    off = [_off(60), _off(42, cls="RAIL", tail_alarms=[3407872]), _off(80, cls="PARKED", habit="SATURATING")]
    rd = P.reading(ph, [], off, [])
    assert rd["line"] == 4 and rd["r"] == 0 and rd["off"]["r"] == 1 and rd["off"]["p"] == 1 and rd["off"]["hs"] == 1
    assert rd["off_text"] == ("CONCURRENT SWITCH-OFF RUNS (printed beside; they decide no line): TAIL RAIL in 1 "
                              "(PRESENCE 0), PARKED in 1, GATE PASS in 3, SATURATING habit elsewhere in 1, DRIVE ONLY in "
                              "0, of 3; 0 not read")
    assert rd["no_rail_chance"] == ("[model, plug-in] if the phase critic changed nothing, no RAIL in 3 runs has chance "
                                    "(1 - 1/3)^3 = 0.296 at the concurrent switch-off runs' 1 of 3" + NO_RAIL_BESIDE)
    # the same phase runs with clean switch-off runs: the same line and text
    rd2 = P.reading(ph, [], [_off(60), _off(42), _off(80)], [])
    assert rd2["line"] == rd["line"] and rd2["text"] == rd["text"]
    assert "(1 - 0/3)^3 = 1.000" in rd2["no_rail_chance"]
    # a line-1 phase run: its first TAIL ALARM beside the same-seed switch-off run's
    ph1 = [_run(60), _run(42, cls="RAIL", kind="RAIL-LEDGER-PERSISTENT", tail_alarms=[3932160]), _run(80)]
    rd3 = P.reading(ph1, [], off, [])
    assert rd3["line"] == 1 and rd3["first_tail_alarms"][0]["beside_off_same_seed"] == 3407872
    # a LOST switch-off run is "not read" beside and counts in u; a PRESENCE rail of a switch-off run counts in r
    rd4 = P.reading(ph, [], [_off(60, verdict="LOST"), _off(42), _off(80)],
                    [dict(run=f"{C.ARM_PREFIX['off']}60", seed=60, detail={"step": 7})])
    assert rd4["line"] == 4 and rd4["off"]["u"] == 1 and rd4["off"]["r"] == 1 and rd4["off"]["n_read"] == 3


# ================================================================================================ synthetic roots
OFF_CMD = 0.5


def _round(tmp_path, seeds=(60, 42, 80), cmd=0.05, per_seed=None, off_seeds=(60, 42, 80), off_cmd=OFF_CMD,
           off_per_seed=None):
    """PC AMENDMENT: six runs by default -- the phase runs (cmd) and the concurrent switch-off runs (off_cmd)."""
    rnd = S.Round(tmp_path / "r")
    runs = {}
    for s in seeds:
        kw = dict(ledger=S.ledger_with_cmd(cmd))
        kw.update((per_seed or {}).get(s, {}))
        runs[s] = S.standard_run(rnd, s, **kw)
    for s in off_seeds:
        kw = dict(ledger=S.ledger_with_cmd(off_cmd))
        kw.update((off_per_seed or {}).get(s, {}))
        runs[("off", s)] = S.standard_run(rnd, s, arm="off", **kw)
    return rnd, runs


def _score(rnd):
    out, status = rnd.score()
    return out, status


def _set_view_pins(rnd):
    """Both arms' view pins: a default synthetic run's own view (PC AMENDMENT: the switch-off arm's too)."""
    rnd.pins["stage_config_view_sha256"] = S.view_pin_of(S.Run(f"{C.RUN_ID_PREFIX}60", S.T0, rnd.pins))
    rnd.pins["off_arm"]["stage_config_view_sha256"] = S.view_pin_of(S.Run(f"{C.ARM_PREFIX['off']}60", S.T0, rnd.pins))


def test_root_all_clear_reads_line4_and_one_test_lower(tmp_path):
    rnd, _runs = _round(tmp_path, cmd=0.05)
    out, status = _score(rnd)
    assert out["incomplete"] == [], out["incomplete"]
    assert [x["verdict"] for x in out["per_seed"]] == ["COMPLETE"] * 3
    rd = out["reading"]
    assert rd["line"] == 4, rd["text"]
    assert rd["one_test"]["k"] == 1 and rd["one_test"]["text"] == LOWER_TEXT
    assert [x["td"]["td"] for x in out["per_seed"]] == [pytest.approx(0.05)] * 3
    assert [x["verdict"] for x in out["per_seed_off"]] == ["COMPLETE"] * 3
    assert [x["td"]["td"] for x in out["per_seed_off"]] == [pytest.approx(OFF_CMD)] * 3
    assert rd["one_test"]["off"] == [pytest.approx(OFF_CMD)] * 3 and rd["one_test"]["off_missing"] == [False] * 3
    assert status == SP.EXIT_FINAL, out["provisional"]
    text = SP.render(out)
    assert "READING: line 4: NO TAIL RAIL OR PARK IN 3 OF 3" in text
    assert "THE ONE TEST (phase vs the concurrent switch-off runs): LOWER THAN EVERY CONCURRENT SWITCH-OFF RUN" in text
    assert "OFF-vs-STANCE6 REPLICATION (REPORT ONLY" in text and "STANCE6 (historical, REPORT ONLY" in text


def test_root_tail_rail_reads_line1(tmp_path):
    led = S.ledger_with_cmd(0.95)
    for s in C.GRID_STEPS[30:33]:
        led[s] = [dict(r, tail_3_pitch_act_command_mean="0.95") for r in S.evaluation([(S.row("tail", 14, "+"), 6), (S.row("clean"), 4)])]
    rnd, _ = _round(tmp_path, per_seed={42: dict(ledger=led)})
    out, _st = _score(rnd)
    rd = out["reading"]
    assert rd["line"] == 1 and "(ledger)" in rd["text"] and rd["r"] == 1
    c42 = next(x for x in out["per_seed"] if x["seed"] == 42)
    assert c42["classes"]["cls"] == "RAIL" and c42["early"]["firsts"]["first_t3_over"] == C.GRID_STEPS[30]
    assert c42["td"]["td"] == pytest.approx(0.95)
    assert rd["one_test"]["k"] > 1


def test_root_c8_not_in_force_is_lost_and_td_infinite(tmp_path):
    rnd, runs = _round(tmp_path)
    S.write_c8(rnd.root / "panels" / runs[80][0].run_id / "final" / "c8", verdict="NOT IN FORCE")
    out, _st = _score(rnd)
    x80 = next(x for x in out["per_seed"] if x["seed"] == 80)
    assert x80["verdict"] == "LOST"
    comp = out["runs"][runs[80][0].run_id]["completeness"]
    assert comp["C8"]["ok"] is False and "not of the study's configuration" in comp["C8"]["note"]
    rd = out["reading"]
    assert rd["line"] == 2 and rd["u"] == 1
    assert rd["one_test"]["k"] == 11 and rd["one_test"]["phase_missing"] == [False, False, True]


def test_root_c8_absent_is_pending_and_incomplete(tmp_path):
    rnd, runs = _round(tmp_path)
    (rnd.root / "panels" / runs[60][0].run_id / "4194304" / "c8" / "manip_check.json").unlink()
    out, status = _score(rnd)
    assert status == SP.EXIT_INCOMPLETE and out["reading"] is None
    assert any("PENDING (C8" in x for x in out["incomplete"]), out["incomplete"]


def test_root_c8_record_of_another_file_is_not_counted(tmp_path):
    rnd, runs = _round(tmp_path)
    _set_view_pins(rnd)
    rnd.write_meta()
    S.make_trigger_stores(rnd.root, rnd.runlog)
    S.bind_panel_records(rnd.root)
    f = rnd.root / "panels" / runs[60][0].run_id / "final" / "c8" / "manip_check.json"
    rec = json.loads(f.read_text())
    rec["checkpoint"]["sha256"] = "0" * 64
    f.write_text(json.dumps(rec))
    out, _st = SP.score(rnd.root, rnd.runlog, rnd.pins_path, None, comparators=S.comparators_cached())
    comp = out["runs"][runs[60][0].run_id]["completeness"]
    assert comp["C8"]["ok"] is None and any("another file" in x for x in comp["c8_other_file"])


def test_root_c8_wrong_seeds_not_a_record(tmp_path):
    rnd, runs = _round(tmp_path)
    S.write_c8(rnd.root / "panels" / runs[60][0].run_id / "final" / "c8", seeds=(3042, 3044))
    out, _st = _score(rnd)
    comp = out["runs"][runs[60][0].run_id]["completeness"]
    assert comp["C8"]["ok"] is None and "PENDING" in comp["C8"]["note"]


def test_root_switch_off_fails_c4p_lost(tmp_path):
    rnd, runs = _round(tmp_path, per_seed={60: dict(sc_over={"hyperparameters.critic_phase": False})})
    out, _st = _score(rnd)
    comp = out["runs"][runs[60][0].run_id]["completeness"]
    c4p = [f for f in comp["C4"]["value"] if f["name"].startswith("C4 (p)")]
    assert len(c4p) == 1 and c4p[0]["ok"] is False and comp["verdict"] == "LOST"
    assert out["reading"]["line"] == 2


def test_root_switch_as_string_true_is_not_on(tmp_path):
    rnd, runs = _round(tmp_path, per_seed={60: dict(sc_over={"hyperparameters.critic_phase": "true"})})
    out, _st = _score(rnd)
    assert out["runs"][runs[60][0].run_id]["completeness"]["verdict"] == "LOST"


def test_root_parked_reads_line3(tmp_path):
    traces = {lab: {"cmd": {14: 0.7}} for lab in [*map(str, C.J_STEPS), "final"]}
    rnd, _ = _round(tmp_path, per_seed={60: dict(traces=traces)})
    out, _st = _score(rnd)
    x60 = next(x for x in out["per_seed"] if x["seed"] == 60)
    assert x60["classes"]["cls"] == "PARKED" and out["reading"]["line"] == 3


def test_root_gate_fail_reads_line5(tmp_path):
    rnd, _ = _round(tmp_path, per_seed={80: dict(cert=False)})
    out, _st = _score(rnd)
    assert out["reading"]["line"] == 5 and "WITH: 1 GATE FAIL" in out["reading"]["text"]


def test_root_habit_elsewhere_reads_line5(tmp_path):
    traces = {"final": {"other_flags": {0: (0, 1), 1: (0, 1)}}}
    rnd, _ = _round(tmp_path, per_seed={42: dict(traces=traces)})
    out, _st = _score(rnd)
    x42 = next(x for x in out["per_seed"] if x["seed"] == 42)
    assert x42["habit"]["saturating"] == ["neck_pitch_act"] and x42["habit"]["per_actuator"]["neck_pitch_act"]["sites"] == ["T"]
    assert out["reading"]["line"] == 5 and "1 WITH A SATURATING HABIT ON ANOTHER ACTUATOR" in out["reading"]["text"]


def test_root_stopped_complete_run_enters_the_test_at_infinity(tmp_path):
    """D6 (R12): a STOPPED COMPLETE run (UNREAD) enters THE ONE TEST at +infinity, not at its own TD."""
    E = C.GRID_STEPS[50]
    rnd, _ = _round(tmp_path, per_seed={42: dict(ledger=S.ledger_with_cmd(0.05, E=E), E=E, hop_watch=True)})
    out, _st = _score(rnd)
    x42 = next(x for x in out["per_seed"] if x["seed"] == 42)
    assert x42["verdict"] == "COMPLETE" and x42["classes"]["cls"] == "UNREAD" and x42["finished"] is False
    assert x42["td"]["td"] == pytest.approx(0.05)
    ot = out["reading"]["one_test"]
    assert out["reading"]["line"] == 2 and ot["phase_missing"] == [False, True, False] and ot["p"] == "11/20"


def test_root_never_launched_seed_is_incomplete_line2(tmp_path):
    rnd, _ = _round(tmp_path, seeds=(60, 42))
    out, _st = _score(rnd)
    assert out["reading"]["line"] == 2 and out["reading"]["u"] == 1
    assert out["reading"]["one_test"]["phase_missing"] == [False, False, True]


def test_launch_check_fact8_and_replacement_ii(tmp_path):
    rnd = S.Round(tmp_path / "r")
    seed = 60
    first = f"{C.RUN_ID_PREFIX}{seed}"
    run0 = S.Run(first, S.T0, rnd.pins, sc_over={"hyperparameters.critic_phase": False}, ledger=S.ledger_with_cmd(0.05))
    rnd.add_note(S.T0, "LAUNCH", first, **S.launch_fields(seed, rnd.pins["branch_head"]))
    S.write_store(rnd.root, "F0", run0, run0.files(), S.T0 + timedelta(minutes=30))
    rnd.add_note(S.T0 + timedelta(minutes=40), "STOP", first)
    rnd.add_note(S.T0 + timedelta(minutes=40), "TRIGGER-d", first)
    rnd.add_note(S.T0 + timedelta(minutes=45), "DELETED", first)
    b = first + "_b"
    run_b, _d = S.Run(b, S.T0 + timedelta(minutes=50), rnd.pins, ledger=S.ledger_with_cmd(0.05)), None
    rnd.add_note(S.T0 + timedelta(minutes=50), "RELAUNCH", b, **S.launch_fields(seed, rnd.pins["branch_head"]))
    end_t = S.T0 + timedelta(hours=16)
    rnd.add_note(end_t, "TRIGGER-a", b)
    rnd.add_note(end_t, "FOUND-GONE", b)
    S.write_store(rnd.root, "F2", run_b, run_b.files(), end_t + timedelta(hours=7))
    rnd.add_note(end_t + timedelta(hours=7), "F2-START", b)
    S.write_panels(rnd.root, run_b)
    for s in (42, 80):
        S.standard_run(rnd, s, ledger=S.ledger_with_cmd(0.05))
    out, _st = rnd.score()
    lc = out["launch_checks"][first]
    assert 8 in lc["faults"] and lc["faults"] == [8], lc
    d60 = next(d for d in out["seeds"] if d["seed"] == seed and d["arm"] == "pc")
    assert d60["run"] == b and d60["replaced"] == first, d60["why"]
    assert next(x for x in out["per_seed"] if x["seed"] == seed)["verdict"] == "COMPLETE"


def test_launch_check_fact8_states(tmp_path):
    pins = S.pins_for_tests()
    store = type("St", (), {})()
    for sc, want in (({"hyperparameters": {"critic_phase": True}}, True), ({"hyperparameters": {}}, False),
                     ({"hyperparameters": {"critic_phase": 1}}, False)):
        f = CP.critic_phase_fact("(8)", sc, pins)
        assert f["ok"] is want, (sc, f)
    assert CP.critic_phase_fact("(8)", None, pins)["ok"] is None
    p2 = dict(pins, critic_phase_switch="UNFILLED")
    assert CP.critic_phase_fact("(8)", {"hyperparameters": {"critic_phase": True}}, p2)["ok"] is None
    del store


def test_provisional_lists_validations_and_head(tmp_path):
    rnd, _ = _round(tmp_path)
    _set_view_pins(rnd)
    rnd.write_meta()
    ap = json.loads((rnd.root / "APPROVED").read_text())
    ap["validations"]["VP2"] = "FAIL"
    ap["branch_head"] = "b" * 40
    (rnd.root / "APPROVED").write_text(json.dumps(ap))
    S.make_trigger_stores(rnd.root, rnd.runlog)
    S.bind_panel_records(rnd.root)
    out, status = SP.score(rnd.root, rnd.runlog, rnd.pins_path, None, comparators=S.comparators_cached())
    assert status == SP.EXIT_PROVISIONAL
    assert any("validation VP2" in x for x in out["provisional"]) and any("branch HEAD" in x for x in out["provisional"])
    assert any("APPROVED sha256" in x and "no RUNLOG line" in x for x in out["provisional"])   # APPROVED rewritten


def test_provisional_f2_pins(tmp_path):
    rnd, _ = _round(tmp_path)
    _set_view_pins(rnd)
    S.make_trigger_stores(rnd.root, rnd.runlog) if rnd.runlog.exists() else None
    for kw, want in ((dict(f2_pins=False), "F2_PINS.json absent"),
                     (dict(f2_over={"validations": {"VP1": "PASS"}}), "validation VP4 (F2_PINS.json)"),
                     (dict(f2_over={"tool_sha256": {"score_pc.py": "0" * 64}}), "tool score_pc.py sha256"),
                     (dict(runlog_pins=False), "APPROVED sha256"),
                     (dict(runlog_pins="approved"), "F2_PINS.json sha256")):
        rnd.write_meta(**kw)
        S.make_trigger_stores(rnd.root, rnd.runlog)
        S.bind_panel_records(rnd.root)
        out, status = SP.score(rnd.root, rnd.runlog, rnd.pins_path, None, comparators=S.comparators_cached())
        assert status == SP.EXIT_PROVISIONAL and any(want in x for x in out["provisional"]), (kw, out["provisional"])
    rnd.write_meta()
    out, status = SP.score(rnd.root, rnd.runlog, rnd.pins_path, None, comparators=S.comparators_cached())
    assert status == SP.EXIT_FINAL, out["provisional"]


def test_comparators_reproduce_the_design_table():
    comp = S.comparators_cached()["runs"]
    assert [round(comp[s]["td"]["td"], 6) for s in (60, 42, 80)] == [0.388227, 0.635343, 0.154339]
    assert [round(comp[s]["td"]["td_j"], 6) for s in (60, 42, 80)] == [0.577658, 0.808751, 0.217914]
    assert [comp[s]["classes"]["cls"] for s in (60, 42, 80)] == ["PARKED", "RAIL", "CLEAR"]
    assert comp[42]["classes"]["first_tail_alarm"] == 3932160 and comp[42]["classes"]["tail_alarms"] == 22
    assert comp[80]["habit"]["sites"] == {"neck_pitch_act": ["L", "A", "T", "D"], "head_pitch_act": ["D"]}
    assert comp[60]["habit"]["status"] == comp[42]["habit"]["status"] == "FREE"
    assert [comp[s]["early"]["n_cmd_l_ge_060"] for s in (60, 42, 80)] == [37, 57, 1]


# ================================================================================================ PC AMENDMENT: six runs
def _off_x(out, seed):
    return next(x for x in out["per_seed_off"] if x["seed"] == seed)


def test_root_six_runs_not_lower_when_an_off_run_is_low(tmp_path):
    """The one test reads the concurrent switch-off runs' TDs, not STANCE6's: one switch-off run below the phase runs
    (0.01 < 0.05) breaks complete separation (k = 4 here: splits with that run and two of the 0.05 runs, and the
    phase split itself)."""
    rnd, _ = _round(tmp_path, off_per_seed={42: dict(ledger=S.ledger_with_cmd(0.01))})
    out, _st = _score(rnd)
    ot = out["reading"]["one_test"]
    assert out["reading"]["line"] == 4 and ot["k"] == 4 and ot["text"] == "NOT SEPARATED (p = 4/20)"
    assert ot["off"][1] == pytest.approx(0.01) and ot["stance6_report_only"]["td"][1] == pytest.approx(COMP_TD[1])


def test_root_missing_off_run_enters_lowest(tmp_path):
    """A switch-off seed never launched enters at -infinity: never LOWER; the phase runs' line is unchanged."""
    rnd, _ = _round(tmp_path, off_seeds=(60, 80))
    out, _st = _score(rnd)
    rd = out["reading"]
    assert rd["line"] == 4 and rd["off"]["u"] == 1
    assert _off_x(out, 42)["verdict"] == "NEVER LAUNCHED"
    assert rd["one_test"]["off_missing"] == [False, True, False] and rd["one_test"]["k"] == 11
    assert not rd["one_test"]["lower"]


def test_root_missing_phase_run_enters_highest(tmp_path):
    rnd, _ = _round(tmp_path, seeds=(60, 80))
    out, _st = _score(rnd)
    ot = out["reading"]["one_test"]
    assert out["reading"]["line"] == 2 and ot["phase_missing"] == [False, True, False] and ot["k"] == 11


def test_root_off_run_reading_a_phase_fails_c8_lost(tmp_path):
    """A switch-off run whose manipulation-check record reads a phase critic (IN FORCE) fails C8: LOST, missing."""
    rnd, runs = _round(tmp_path, off_per_seed={80: dict(c8={"final": "IN FORCE"})})
    out, _st = _score(rnd)
    rid = runs[("off", 80)][0].run_id
    comp = out["runs"][rid]["completeness"]
    assert comp["C8"]["ok"] is False and "stock critic" in comp["C8"]["note"] and comp["verdict"] == "LOST"
    assert comp["C8"]["name"].endswith("a stock critic, no phase input (switch-off arm)")
    assert out["reading"]["one_test"]["off_missing"] == [False, False, True]


def test_root_off_run_with_the_switch_on_fails_c4p_lost(tmp_path):
    rnd, runs = _round(tmp_path, off_per_seed={60: dict(sc_over={"hyperparameters.critic_phase": True})})
    out, _st = _score(rnd)
    comp = out["runs"][runs[("off", 60)][0].run_id]["completeness"]
    c4p = [f for f in comp["C4"]["value"] if f["name"].startswith("C4 (p)")]
    assert len(c4p) == 1 and c4p[0]["ok"] is False and c4p[0]["name"].endswith("switch OFF (no key)")
    assert comp["verdict"] == "LOST" and out["reading"]["one_test"]["off_missing"] == [True, False, False]
    # the phase arm's pins are not the switch-off arm's: a phase run read against them would fail C4
    rid_pc = runs[60][0].run_id
    assert out["runs"][rid_pc]["completeness"]["verdict"] == "COMPLETE"


def test_root_off_rail_is_beside_and_replication_block(tmp_path):
    led = S.ledger_with_cmd(0.95)
    for st in C.GRID_STEPS[30:33]:
        led[st] = [dict(r, tail_3_pitch_act_command_mean="0.95")
                   for r in S.evaluation([(S.row("tail", 14, "+"), 6), (S.row("clean"), 4)])]
    rnd, _ = _round(tmp_path, off_per_seed={42: dict(ledger=led)})
    out, _st = _score(rnd)
    rd = out["reading"]
    assert rd["line"] == 4 and rd["off"]["r"] == 1 and _off_x(out, 42)["classes"]["cls"] == "RAIL"
    assert "(1 - 1/3)^3 = 0.296 at the concurrent switch-off runs' 1 of 3" in rd["no_rail_chance"]
    assert rd["one_test"]["k"] == 1                                    # 0.05 x 3 against 0.5, 0.95, 0.5
    rep = out["replication"]
    assert [x["seed"] for x in rep["rows"]] == [60, 42, 80] and rep["rows"][1]["off"]["cls"] == "RAIL"
    assert rep["rows"][1]["stance6"]["cls"] == "RAIL" and rep["rows"][0]["stance6"]["cls"] == "PARKED"
    assert rep["totals"]["off"]["rail"] == 1 and rep["totals"]["stance6"]["rail"] == 1
    assert "REPORT ONLY" in rep["note"] and "A6" in rep["note"]


def test_root_stance6_unreadable_holds_nothing_open(tmp_path):
    """STANCE6 is report only: comparators that cannot be read leave the reading FINAL and the test unchanged."""
    rnd, _ = _round(tmp_path)
    _set_view_pins(rnd)
    rnd.write_meta()
    S.make_trigger_stores(rnd.root, rnd.runlog)
    S.bind_panel_records(rnd.root)
    orig = P.comparators

    def broken(*a, **k):
        raise ValueError("comparator inputs not as pinned (test)")
    P.comparators = broken
    try:
        out, status = SP.score(rnd.root, rnd.runlog, rnd.pins_path, None)
    finally:
        P.comparators = orig
    assert status == SP.EXIT_FINAL and out["incomplete"] == [] and out["comparators"] is None
    assert out["reading"]["one_test"]["k"] == 1 and "stance6_report_only" not in out["reading"]["one_test"]
    assert "STANCE6 (REPORT ONLY): not read (ValueError" in SP.render(out)


def test_root_outcome_blindness_off_after_a_phase_snapshot_is_missing(tmp_path):
    """An off run launched after a phase run's results were opened (a SNAPSHOT line) enters the test as missing; the
    phase-off start gaps are printed, report only."""
    late = S.T0 + timedelta(hours=3)
    rnd, runs = _round(tmp_path, off_seeds=(60, 42))
    S.standard_run(rnd, 80, arm="off", launch=late, ledger=S.ledger_with_cmd(OFF_CMD))
    rnd.add_note(S.T0 + timedelta(hours=2), "SNAPSHOT", runs[42][0].run_id, note="report-only look")
    out, _st = _score(rnd)
    bl = out["blindness"]
    off80 = f"{C.ARM_PREFIX['off']}80"
    assert bl["not_blind"] == [off80] and bl["first_opening"]["kind"] == "SNAPSHOT"
    assert bl["seeds"][80]["gap_minutes_report_only"] == 180.0 and bl["seeds"][60]["gap_minutes_report_only"] == 0.0
    assert bl["seeds"][80]["blind"] is False and bl["seeds"][60]["blind"] is True
    ot = out["reading"]["one_test"]
    assert ot["off_missing"] == [False, False, True] and ot["not_blind_entered_as_missing"] == [off80] and ot["k"] == 11
    assert _off_x(out, 80)["verdict"] == "COMPLETE"                     # its classes are still printed beside
    # the same late start with no opening before it is blind (staggered starts are allowed)
    rnd2, runs2 = _round(tmp_path / "b", off_seeds=(60, 42))
    S.standard_run(rnd2, 80, arm="off", launch=late, ledger=S.ledger_with_cmd(OFF_CMD))
    out2, _st2 = _score(rnd2)
    assert out2["blindness"]["not_blind"] == [] and out2["reading"]["one_test"]["k"] == 1


def test_launch_check_off_arm_facts(tmp_path):
    """PC AMENDMENT: a switch-off run's launch check reads its own arm's pins: no fault on its own files; the switch
    ON in an off run is fault (8); an off run's files under the phase RUN_ID fault (3) and (8)."""
    rnd = S.Round(tmp_path / "r")
    _set_view_pins(rnd)
    off = f"{C.ARM_PREFIX['off']}42"
    run_off = S.Run(off, S.T0, rnd.pins, ledger=S.ledger_with_cmd(0.5))
    rnd.add_note(S.T0, "LAUNCH", off, **S.launch_fields(42, rnd.pins["branch_head"], "off"))
    st = S.write_store(rnd.root, "F0", run_off, run_off.files(), S.T0 + timedelta(minutes=30))
    rnd.write_meta()
    notes, _o = C.parse_runlog(rnd.runlog, C.RUN_IDS)
    fm = rnd.root / "fetch" / "manifest.csv"
    lc = CP.launch_check(off, C.FetchStore(st, fm, "F0", off), notes, rnd.pins)
    assert lc["faults"] == [] and lc["facts"][8]["name"] == "(8) critic-phase switch OFF (no key) in stage_config.json"
    assert lc["facts"][3]["ok"] is True and lc["facts"][8]["value"]["present"] is False
    bad = S.Run(off, S.T0, rnd.pins, ledger=S.ledger_with_cmd(0.5), sc_over={"hyperparameters.critic_phase": True})
    st2 = S.write_store(rnd.root, "F0", bad, bad.files(), S.T0 + timedelta(minutes=30))
    assert CP.launch_check(off, C.FetchStore(st2, fm, "F0", off), notes, rnd.pins)["faults"] == [8]
    # the switch-off file read as a phase run's: the phase pins differ -> (3) and (8); its run_id differs -> (7)
    pc = f"{C.RUN_ID_PREFIX}42"
    rnd.add_note(S.T0, "LAUNCH", pc, **S.launch_fields(42, rnd.pins["branch_head"]))
    rnd.write_meta()
    notes, _o = C.parse_runlog(rnd.runlog, C.RUN_IDS)
    run_off2 = S.Run(off, S.T0, rnd.pins, ledger=S.ledger_with_cmd(0.5))
    run_off2.run_id = pc                                      # the store's folder name; the files stay the off run's
    st3 = S.write_store(rnd.root, "F0", run_off2, S.Run(off, S.T0, rnd.pins, ledger=S.ledger_with_cmd(0.5)).files(),
                        S.T0 + timedelta(minutes=30))
    assert CP.launch_check(pc, C.FetchStore(st3, fm, "F0", pc), notes, rnd.pins)["faults"] == [3, 7, 8]


def test_switch_fact_off_states():
    pins = S.pins_for_tests()
    off = f"{C.ARM_PREFIX['off']}60"
    for sc, want in (({"hyperparameters": {}}, True), ({"hyperparameters": {"critic_phase": False}}, False),
                     ({"hyperparameters": {"critic_phase": None}}, False), ({"hyperparameters": {"critic_phase": True}}, False),
                     ({}, True)):
        f = CP.critic_phase_fact("(8)", sc, pins, off)
        assert f["ok"] is want, (sc, f)
    assert CP.critic_phase_fact("(8)", None, pins, off)["ok"] is None
    p2 = dict(pins, off_arm="UNFILLED")
    assert CP.critic_phase_fact("(8)", {"hyperparameters": {}}, p2, off)["ok"] is None
    assert CP.arm_pin(p2, "hyperparameters_sha256", off) is None
    # a phase run still reads the phase pin
    assert CP.critic_phase_fact("(8)", {"hyperparameters": {}}, pins, f"{C.RUN_ID_PREFIX}60")["ok"] is False
    assert CP.arm_pin(pins, "hyperparameters_sha256", off) == pins["off_arm"]["hyperparameters_sha256"]
    assert CP.arm_pin(pins, "hyperparameters_sha256", off) != pins["hyperparameters_sha256"]
    assert CP.arm_pin(pins, "plant_identity", off) == pins["plant_identity"]


def test_c8_holds_per_arm_and_manip_stock_record_shape(tmp_path):
    """The switch-off arm's C8 form is exactly what manip_check.manipulation_checks writes for a critic with no phase
    input; a phase critic's records (IN FORCE, or NOT IN FORCE with zero phase weights) do not meet it."""
    MC = S._load("manip_check")

    class Stock:                                     # a policy with no critic_phase attribute: a stock critic
        pass
    rec = MC.manipulation_checks(Stock(), [[0.0] * 63] * 4, [0, 1, 2, 3])
    off, pc = f"{C.ARM_PREFIX['off']}42", f"{C.RUN_ID_PREFIX}42"
    assert CP.c8_holds(rec, off) is True and CP.c8_holds(rec, pc) is False
    widened = dict(rec, spec={"schema": "mesozoic.critic-phase/v1"},
                   checks={"(i) phase in force": dict(holds=False, phase_weights_nonzero=0)})
    assert CP.c8_holds(widened, off) is False
    assert CP.c8_holds(dict(verdict="IN FORCE", spec={"x": 1}, failing=[]), off) is False
    assert CP.c8_holds(dict(verdict="IN FORCE"), pc) is True
    no_spec_key = {k: v for k, v in rec.items() if k != "spec"}
    assert CP.c8_holds(no_spec_key, off) is False


def test_six_run_ids_tokens_and_arms():
    assert len(C.RUN_IDS) == 6 and set(C.RUN_IDS.values()) == {60, 42, 80}
    assert C.arm_of("stance7_r10LTSBoff_s42_b") == "off" and C.arm_of("stance7_r10LTSBpc_s42") == "pc"
    assert C.arm_of("stance6_r10LTSB_s42") is None and CP.base_seed("stance7_r10LTSBoff_s80_b") == 80
    assert C.RUN_TOKEN_RE.fullmatch("stance7_r10LTSBoff_s60_b") and C.RUN_TOKEN_RE.fullmatch("stance7_r10LTSBpc_s60")
    assert C.ARM_LABELS["off"][42] == "r10 LTSB switch-off s42" and C.ARM_CRITIC_PHASE == {"pc": "True", "off": "False"}
    assert "SNAPSHOT" in C.KINDS and "SNAPSHOT" in C.AFTER_F2_KINDS and "off_arm" in C.PIN_KEYS
    MR = S._load("measure_r10round")
    assert MR.ROUND_RUN_RE.match("stance7_r10LTSBoff_s42_b") and not MR.ROUND_RUN_RE.match("stance7_r10LTSBoff_s43")


# ================================================================================ FIXER (re-review M1): five amended rules
def test_root_blindness_an_off_snapshot_is_no_opening(tmp_path):
    """Re-review M1 (X1): an opening is a PHASE run's F2-START, REFETCH-START or SNAPSHOT line only; a switch-off run's
    SNAPSHOT before a later switch-off launch leaves that launch blind."""
    late = S.T0 + timedelta(hours=3)
    rnd, runs = _round(tmp_path, off_seeds=(60, 42))
    S.standard_run(rnd, 80, arm="off", launch=late, ledger=S.ledger_with_cmd(OFF_CMD))
    rnd.add_note(S.T0 + timedelta(hours=2), "SNAPSHOT", runs[("off", 60)][0].run_id, note="report-only look")
    out, _st = _score(rnd)
    bl = out["blindness"]
    assert bl["not_blind"] == [] and bl["seeds"][80]["blind"] is True
    assert bl["first_opening"]["kind"] == "F2-START" and C.arm_of(bl["first_opening"]["run"]) == "pc"
    ot = out["reading"]["one_test"]
    assert ot["off_missing"] == [False] * 3 and ot["not_blind_entered_as_missing"] == [] and ot["k"] == 1


def test_root_blindness_an_off_launch_in_the_opening_minute_is_not_blind(tmp_path):
    """Re-review M1 (X2): "not earlier" -- a switch-off LAUNCH line in the same minute as a PHASE run's SNAPSHOT line is
    NOT BLIND, even written above it (RUNLOG times are minutes; the order within a minute is not read)."""
    t = S.T0 + timedelta(hours=2)
    rnd, runs = _round(tmp_path, off_seeds=(60, 42))
    S.standard_run(rnd, 80, arm="off", launch=t, ledger=S.ledger_with_cmd(OFF_CMD))
    rnd.add_note(t, "SNAPSHOT", runs[42][0].run_id, note="report-only look")
    out, _st = _score(rnd)
    off80 = f"{C.ARM_PREFIX['off']}80"
    lines = rnd.runlog.read_text().splitlines()
    assert [i for i, x in enumerate(lines) if f"LAUNCH {off80} " in x] < [i for i, x in enumerate(lines) if "SNAPSHOT" in x]
    bl = out["blindness"]
    assert bl["not_blind"] == [off80] and bl["seeds"][80]["blind"] is False
    assert bl["first_opening"]["kind"] == "SNAPSHOT" and bl["first_opening"]["time"] == bl["seeds"][80]["off_launch"]
    ot = out["reading"]["one_test"]
    assert ot["off_missing"] == [False, False, True] and ot["not_blind_entered_as_missing"] == [off80] and ot["k"] == 11


def test_c8_off_arm_reads_the_stock_critic_why():
    """Re-review M1 (X3): the switch-off arm's C8 form includes (i)'s why exactly as manip_check.manipulation_checks
    writes it for a stock critic; a stock-shaped record with another why, or none, does not meet it."""
    MC = S._load("manip_check")

    class Stock:                                     # a policy with no critic_phase attribute: a stock critic
        pass
    rec = MC.manipulation_checks(Stock(), [[0.0] * 63] * 4, [0, 1, 2, 3])
    off = f"{C.ARM_PREFIX['off']}42"
    assert rec["checks"]["(i) phase in force"]["why"] == CP.STOCK_CRITIC_WHY and CP.c8_holds(rec, off) is True
    for why in ("the phase weights are zero", "", None):
        other = json.loads(json.dumps(rec))
        other["checks"]["(i) phase in force"]["why"] = why
        assert CP.c8_holds(other, off) is False, why
    no_why = json.loads(json.dumps(rec))
    del no_why["checks"]["(i) phase in force"]["why"]
    assert CP.c8_holds(no_why, off) is False


def test_launch_check_cli_refuses_an_unfilled_switch_pin_per_arm(tmp_path, capsys):
    """Re-review M1 (X5): the launch-check CLI refuses (INPUT ERROR, exit 2) when the run's arm's switch pin is
    unfilled -- off_arm.critic_phase_switch for a switch-off run, critic_phase_switch for a phase run -- and runs on the
    same files (exit 0, no fault) with the pins filled."""
    rnd = S.Round(tmp_path / "r")
    _set_view_pins(rnd)
    off, pc = f"{C.ARM_PREFIX['off']}42", f"{C.RUN_ID_PREFIX}42"
    for rid, arm in ((off, "off"), (pc, "pc")):
        run = S.Run(rid, S.T0, rnd.pins, ledger=S.ledger_with_cmd(0.5))
        rnd.add_note(S.T0, "LAUNCH", rid, **S.launch_fields(42, rnd.pins["branch_head"], arm))
        S.write_store(rnd.root, "F0", run, run.files(), S.T0 + timedelta(minutes=30))
    rnd.write_meta()

    def cli(rid, pins, name):
        p = tmp_path / f"pins_{name}.json"
        p.write_text(json.dumps(pins))
        capsys.readouterr()
        rc = CP.main(["launch-check", "--run", rid, "--round", str(rnd.root), "--runlog", str(rnd.runlog),
                      "--pins", str(p)])
        return rc, capsys.readouterr().out
    for rid in (off, pc):
        rc, text = cli(rid, rnd.pins, "filled")
        assert rc == 0 and "INPUT ERROR" not in text and f"LAUNCH CHECK {rid}" in text, text
        assert "FAULTS (computed): none" in text, text
    unf_off = json.loads(json.dumps(rnd.pins))
    unf_off["off_arm"]["critic_phase_switch"] = "UNFILLED"
    rc, text = cli(off, unf_off, "unfilled_off")
    assert rc == 2 and "INPUT ERROR: pins unfilled: ['critic_phase_switch']" in text, text
    rc, text = cli(pc, unf_off, "unfilled_off")                  # a phase run reads the top-level pin: filled
    assert rc == 0 and "INPUT ERROR" not in text, text
    unf_pc = json.loads(json.dumps(rnd.pins))
    unf_pc["critic_phase_switch"] = "UNFILLED"
    rc, text = cli(pc, unf_pc, "unfilled_pc")
    assert rc == 2 and "INPUT ERROR: pins unfilled: ['critic_phase_switch']" in text, text


def _off_b_round(root, snapshot_minutes):
    """Five runs as _round makes them, and switch-off s80 replaced: its first run launched at T0 with the switch ON
    (launch-check fault (8)), stopped at +40 min, deleted at +45 min, its _b relaunched at +50 min (REPLACEMENT (ii));
    a PHASE run's SNAPSHOT line at T0 + snapshot_minutes.  F2 fetches both s80 RUN_IDs (every launched RUN_ID)."""
    rnd, runs = _round(root, off_seeds=(60, 42))
    first = f"{C.ARM_PREFIX['off']}80"
    cut = C.GRID_STEPS[3]
    run0 = S.Run(first, S.T0, rnd.pins, sc_over={"hyperparameters.critic_phase": True},
                 ledger=S.ledger_with_cmd(OFF_CMD, E=cut), cut_at=cut)
    rnd.add_note(S.T0, "LAUNCH", first, **S.launch_fields(80, rnd.pins["branch_head"], "off"))
    S.write_store(rnd.root, "F0", run0, run0.files(), S.T0 + timedelta(minutes=30))
    rnd.add_note(S.T0 + timedelta(minutes=40), "STOP", first)
    rnd.add_note(S.T0 + timedelta(minutes=40), "TRIGGER-d", first)
    rnd.add_note(S.T0 + timedelta(minutes=45), "DELETED", first)
    b = first + "_b"
    tb = S.T0 + timedelta(minutes=50)
    run_b = S.Run(b, tb, rnd.pins, ledger=S.ledger_with_cmd(OFF_CMD))
    rnd.add_note(tb, "RELAUNCH", b, **S.launch_fields(80, rnd.pins["branch_head"], "off"))
    end_t = tb + timedelta(hours=15)
    rnd.add_note(end_t, "TRIGGER-a", b)
    rnd.add_note(end_t, "FOUND-GONE", b)
    for r in (run0, run_b):
        S.write_store(rnd.root, "F2", r, r.files(), end_t + timedelta(hours=7))
        rnd.add_note(end_t + timedelta(hours=7), "F2-START", r.run_id)
    S.write_panels(rnd.root, run_b)
    rnd.add_note(S.T0 + timedelta(minutes=snapshot_minutes), "SNAPSHOT", runs[42][0].run_id, note="report-only look")
    out, _st = _score(rnd)
    return out, first, b, tb


def test_root_blindness_reads_the_off_seeds_relaunch(tmp_path):
    """Re-review M1 (X10): when REPLACEMENT made a _b, the switch-off seed's run is the _b and the blindness rule reads
    its RELAUNCH line, not the first run's LAUNCH: a _b relaunched after a PHASE run's SNAPSHOT is NOT BLIND although
    its first run launched before it; the same _b relaunched before the SNAPSHOT is blind."""
    out, first, b, tb = _off_b_round(tmp_path / "a", snapshot_minutes=48)
    assert out["launch_checks"][first]["faults"] == [8]
    d80 = next(d for d in out["seeds"] if d["seed"] == 80 and d["arm"] == "off")
    assert d80["run"] == b and d80["replaced"] == first, d80["why"]
    bl = out["blindness"]
    assert bl["seeds"][80]["off_run"] == b and bl["seeds"][80]["off_launch"] == tb.isoformat()
    assert bl["not_blind"] == [b] and bl["seeds"][80]["blind"] is False
    ot = out["reading"]["one_test"]
    assert ot["off_missing"] == [False, False, True] and ot["not_blind_entered_as_missing"] == [b] and ot["k"] == 11
    assert _off_x(out, 80)["verdict"] == "COMPLETE"                      # its classes are still printed beside
    out2, _f, b2, _tb = _off_b_round(tmp_path / "b", snapshot_minutes=55)
    assert out2["blindness"]["not_blind"] == [] and out2["blindness"]["seeds"][80]["blind"] is True
    assert out2["reading"]["one_test"]["k"] == 1 and out2["reading"]["one_test"]["off_missing"] == [False] * 3
