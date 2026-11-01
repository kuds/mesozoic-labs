"""PC scorer and adapted tools: tests on synthetic roots (synth_pc.py) and unit tests of the PC readings.

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
SUFFIX = ("(descriptive: 3 runs against the stance6 round's 3, a historical comparison; KNOWN LIMITS)")
COMP_TD = [0.38822738704353094, 0.6353434591453841, 0.15433899719014801]      # s60, s42, s80 [measured, comparators]


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
    assert r["text"] == ("LOWER THAN EVERY STANCE6 RUN (exact permutation p = 1/20)" if k == 1 else f"NOT SEPARATED (p = {k}/20)")
    assert r["caveat"] == P.CAVEAT and "conservative" in r["caveat"] and "A5" in r["caveat"]
    assert r["note"] == ("equivalently, every PC TD < 0.154339 (STANCE6 s80's, the lowest comparator); the statistic was "
                         "chosen with STANCE6's values in view (2 windows looked at: a strict bound of 2/20)")


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


def test_one_test_separation_bar_is_s80():
    lo = min(COMP_TD)
    assert P.one_test([lo - 1e-9, 0.0, 0.0], COMP_TD)["k"] == 1
    # a PC value exactly at s80's TD ties: ties count against H1 (2 splits reach the PC sum)
    assert P.one_test([lo, 0.0, 0.0], COMP_TD)["k"] == 2


def test_one_test_infinity_never_helps():
    r = P.one_test([0.0, 0.0, math.inf], COMP_TD)
    assert r["k"] == 20 and r["pc_infinite"] == [False, False, True] and r["pc"][2] is None


def test_one_test_needs_three_and_three():
    with pytest.raises(ValueError):
        P.one_test([0.1, 0.2], COMP_TD)


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
                                    "(2/3)^3 = 0.296 at STANCE6's 1 of 3 and (3/5)^3 = 0.216 at LTSB r9's 2 of 5")
    assert rd["driven_tails"] == []


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
    assert rd["first_tail_alarms"][0]["first_tail_alarm"] == 3932160 and rd["first_tail_alarms"][0]["beside_stance6_s42"] == 3932160
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


# ================================================================================================ synthetic roots
def _round(tmp_path, seeds=(60, 42, 80), cmd=0.05, per_seed=None):
    rnd = S.Round(tmp_path / "r")
    runs = {}
    for s in seeds:
        kw = dict(ledger=S.ledger_with_cmd(cmd))
        kw.update((per_seed or {}).get(s, {}))
        runs[s] = S.standard_run(rnd, s, **kw)
    return rnd, runs


def _score(rnd):
    out, status = rnd.score()
    return out, status


def test_root_all_clear_reads_line4_and_one_test_lower(tmp_path):
    rnd, _runs = _round(tmp_path, cmd=0.05)
    out, status = _score(rnd)
    assert out["incomplete"] == [], out["incomplete"]
    assert [x["verdict"] for x in out["per_seed"]] == ["COMPLETE"] * 3
    rd = out["reading"]
    assert rd["line"] == 4, rd["text"]
    assert rd["one_test"]["k"] == 1 and rd["one_test"]["text"].startswith("LOWER THAN EVERY STANCE6 RUN")
    assert [x["td"]["td"] for x in out["per_seed"]] == [pytest.approx(0.05)] * 3
    assert status == SP.EXIT_FINAL, out["provisional"]
    text = SP.render(out)
    assert "READING: line 4: NO TAIL RAIL OR PARK IN 3 OF 3" in text and "THE ONE TEST: LOWER THAN EVERY" in text


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
    assert rd["one_test"]["k"] == 20 and rd["one_test"]["pc_infinite"] == [False, False, True]


def test_root_c8_absent_is_pending_and_incomplete(tmp_path):
    rnd, runs = _round(tmp_path)
    (rnd.root / "panels" / runs[60][0].run_id / "4194304" / "c8" / "manip_check.json").unlink()
    out, status = _score(rnd)
    assert status == SP.EXIT_INCOMPLETE and out["reading"] is None
    assert any("PENDING (C8" in x for x in out["incomplete"]), out["incomplete"]


def test_root_c8_record_of_another_file_is_not_counted(tmp_path):
    rnd, runs = _round(tmp_path)
    rnd.pins["stage_config_view_sha256"] = S.view_pin_of(S.Run(f"{C.RUN_ID_PREFIX}60", S.T0, rnd.pins))
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
    assert out["reading"]["line"] == 2 and ot["pc_infinite"] == [False, True, False] and ot["p"] == "20/20"


def test_root_never_launched_seed_is_incomplete_line2(tmp_path):
    rnd, _ = _round(tmp_path, seeds=(60, 42))
    out, _st = _score(rnd)
    assert out["reading"]["line"] == 2 and out["reading"]["u"] == 1
    assert out["reading"]["one_test"]["pc_infinite"] == [False, False, True]


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
    d60 = next(d for d in out["seeds"] if d["seed"] == seed)
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
    rnd.pins["stage_config_view_sha256"] = S.view_pin_of(S.Run(f"{C.RUN_ID_PREFIX}60", S.T0, rnd.pins))
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
    rnd.pins["stage_config_view_sha256"] = S.view_pin_of(S.Run(f"{C.RUN_ID_PREFIX}60", S.T0, rnd.pins))
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
