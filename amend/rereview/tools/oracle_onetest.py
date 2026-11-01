"""Re-reviewer's own check of THE ONE TEST (amended): an exact permutation oracle written here, against
pc_common.one_test on random values and against score_pc.score on synthetic six-run roots built in this folder.

    PY -I -B oracle_onetest.py ROOTS_DIR
"""
import itertools
import math
import random
import sys
from datetime import timedelta
from fractions import Fraction
from pathlib import Path

ST = Path("/home/user/work/pc_study")
sys.path.insert(0, str(ST / "tools/tests"))
import synth_pc as S  # noqa: E402

C, CP, SP = S.C, S.CP, S.SP
P = SP.P
FAILS = []


def check(cond, what):
    print(("OK   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


# ------------------------------------------------------------------------------------------------ my oracle
def oracle_k(phase, off):
    """The criteria's rule read literally: p = #(3-subsets of the 6 values whose 3-sum <= the phase 3-sum) / 20,
    a missing phase run = +M, a missing off run = -M, M a concrete number larger than any difference of finite
    3-sums (so the order is the limit for large M); exact Fractions."""
    fin = [Fraction(v) for v in list(phase) + list(off) if math.isfinite(v)]
    M = 1 + 2 * sum(abs(v) for v in fin) * 3
    vals = [Fraction(M) if v == math.inf else (-Fraction(M) if v == -math.inf else Fraction(v))
            for v in list(phase) + list(off)]
    obs = sum(vals[:3])
    return sum(1 for idx in itertools.combinations(range(6), 3) if sum(vals[i] for i in idx) <= obs)


def oracle_k_perm(phase, off):
    """The same by relabelling: all 720 orderings of the six values, the first three taken as 'phase'; each split
    occurs 36 times."""
    fin = [Fraction(v) for v in list(phase) + list(off) if math.isfinite(v)]
    M = 1 + 6 * sum(abs(v) for v in fin)
    vals = [Fraction(M) if v == math.inf else (-Fraction(M) if v == -math.inf else Fraction(v))
            for v in list(phase) + list(off)]
    obs = sum(vals[:3])
    n = sum(1 for perm in itertools.permutations(range(6)) if sum(vals[i] for i in perm[:3]) <= obs)
    assert n % 36 == 0
    return n // 36


# ------------------------------------------------------------------------------------------------ A: unit
rng = random.Random(20261101)
n_cases = 0
for _ in range(3000):
    pool = [round(rng.uniform(0, 1), rng.choice((1, 2, 3))) for _ in range(6)]
    ph, of = pool[:3], pool[3:]
    for i in range(3):
        if rng.random() < 0.2:
            ph[i] = math.inf
        if rng.random() < 0.2:
            of[i] = -math.inf
    r = P.one_test(ph, of)
    k = oracle_k(ph, of)
    n_cases += 1
    if r["k"] != k or r["k"] != oracle_k_perm(ph, of) if n_cases % 10 == 0 else r["k"] != k:
        FAILS.append(f"unit mismatch {ph} {of}: tool {r['k']} oracle {k}")
    allfin = all(math.isfinite(v) for v in ph + of)
    sep = allfin and max(ph) < min(of)
    if (r["k"] == 1) != sep or r["lower"] != sep:
        FAILS.append(f"separation rule {ph} {of}: k {r['k']}")
    if not allfin and r["k"] < 11:
        FAILS.append(f"missing gives k < 11: {ph} {of} k {r['k']}")
    if r["lower"] != (r["k"] == 1) or r["p"] != f"{r['k']}/20":
        FAILS.append(f"text/p {r}")
check(not [f for f in FAILS if f.startswith(("unit", "separation", "missing", "text"))],
      f"pc_common.one_test equals my oracle on {n_cases} random mixes (ties by rounding, +inf phase, -inf off); "
      "k = 1 exactly on complete separation; any missing run k >= 11")
# every missing pattern exhaustively (64), finite values random: minimum k over patterns with any missing
mins = {}
for pat in itertools.product((0, 1), repeat=6):
    if not any(pat):
        continue
    worst = 21
    for _ in range(200):
        v = [rng.uniform(0, 1) for _ in range(6)]
        ph = [math.inf if pat[i] else v[i] for i in range(3)]
        of = [-math.inf if pat[3 + i] else v[3 + i] for i in range(3)]
        k = P.one_test(ph, of)["k"]
        check_k = oracle_k(ph, of)
        if k != check_k:
            FAILS.append(f"pattern {pat} mismatch")
        worst = min(worst, k)
    mins[pat] = worst
check(min(mins.values()) == 11, f"over all 63 missing patterns x 200 draws the smallest k is {min(mins.values())} "
      "(one missing run, either arm); never LOWER")
# near ties that float sums misorder
a = [0.1, 0.2, 0.3]
b = [0.6, 0.0, 0.0]
check(P.one_test(a, b)["k"] == oracle_k(a, b), f"near tie 0.1+0.2+0.3 vs 0.6: tool {P.one_test(a, b)['k']} "
      f"oracle {oracle_k(a, b)} (float sum 0.1+0.2+0.3 = {0.1 + 0.2 + 0.3!r})")
check(P.one_test([0.3, 0.3, 0.3], [0.3, 0.3, 0.3])["k"] == 20, "all six tied: k = 20 (ties against H1)")
check(P.one_test([0.05, 0.06, 0.30], [0.30, 0.40, 0.50])["k"] == 2, "a phase TD equal to the lowest off TD: k = 2, "
      "not LOWER (strictly below is needed)")
for bad in (([0.1, 0.2, -math.inf], [0.3, 0.4, 0.5]), ([0.1, 0.2, 0.3], [0.3, 0.4, math.inf]),
            ([0.1, math.nan, 0.3], [0.3, 0.4, 0.5])):
    try:
        P.one_test(*bad)
        check(False, f"refusal {bad}")
    except ValueError:
        check(True, f"refused {bad}")

# ------------------------------------------------------------------------------------------------ B: roots
ROOTS = Path(sys.argv[1])
T0 = S.T0


def build(name, phase_cmd, off_cmd, phase_kw=None, off_kw=None, extra=None):
    rnd = S.Round(ROOTS / name)
    runs = {}
    for s, c in zip(C.SEEDS, phase_cmd):
        if c is None:
            continue
        kw = dict(ledger=S.ledger_with_cmd(c))
        kw.update((phase_kw or {}).get(s, {}))
        runs[("pc", s)] = S.standard_run(rnd, s, **kw)
    for s, c in zip(C.SEEDS, off_cmd):
        if c is None:
            continue
        kw = dict(ledger=S.ledger_with_cmd(c))
        kw.update((off_kw or {}).get(s, {}))
        runs[("off", s)] = S.standard_run(rnd, s, arm="off", **kw)
    if extra:
        extra(rnd, runs)
    out, status = rnd.score()
    return out, status


def root_case(name, phase_cmd, off_cmd, want_missing_phase, want_missing_off, want_line=None, build_off=None, **kw):
    out, status = build(name, phase_cmd, build_off or off_cmd, **kw)
    if out["incomplete"]:
        check(False, f"{name}: incomplete {out['incomplete'][:3]}")
        return out
    ot = out["reading"]["one_test"]
    ph = [math.inf if m else c for m, c in zip(want_missing_phase, phase_cmd)]
    of = [-math.inf if m else c for m, c in zip(want_missing_off, off_cmd)]
    k_nominal = oracle_k(ph, of)
    ph_t = [math.inf if v is None else v for v in ot["phase"]]
    of_t = [-math.inf if v is None else v for v in ot["off"]]
    k_tool_values = oracle_k(ph_t, of_t)
    ok = (ot["k"] == k_nominal == k_tool_values and ot["phase_missing"] == list(want_missing_phase)
          and ot["off_missing"] == list(want_missing_off)
          and ot["text"] == ("LOWER THAN EVERY CONCURRENT SWITCH-OFF RUN (exact permutation p = 1/20)" if k_nominal == 1
                             else f"NOT SEPARATED (p = {k_nominal}/20)")
          and (want_line is None or out["reading"]["line"] == want_line))
    check(ok, f"{name}: scorer k {ot['k']} ({ot['text']}), my oracle {k_nominal} on the nominal values and "
          f"{k_tool_values} on the scorer's TDs; missing phase {ot['phase_missing']} off {ot['off_missing']}; "
          f"not blind {ot.get('not_blind_entered_as_missing')}; line {out['reading']['line']}; status {status}")
    txt = SP.render(out)
    check("THE ONE TEST (phase vs the concurrent switch-off runs)" in txt and "OUTCOME BLINDNESS" in txt
          and "OFF-vs-STANCE6 REPLICATION" in txt, f"{name}: render prints the test, the blindness block and the "
          "replication block")
    return out


F = [False] * 3
root_case("sep", [0.05, 0.06, 0.07], [0.30, 0.40, 0.50], F, F, 4)
root_case("overlap", [0.05, 0.45, 0.07], [0.30, 0.40, 0.50], F, F, 4)
root_case("boundary_tie", [0.05, 0.06, 0.30], [0.30, 0.40, 0.50], F, F, 4)
root_case("phase_higher", [0.60, 0.70, 0.80], [0.30, 0.40, 0.50], F, F)
root_case("off_c8_reads_phase", [0.05, 0.06, 0.07], [0.30, 0.40, 0.50], F, [False, False, True], 4,
          off_kw={80: dict(c8={"final": "IN FORCE"})})
root_case("phase_c8_stock", [0.05, 0.06, 0.07], [0.30, 0.40, 0.50], [False, True, False], F,
          phase_kw={42: dict(c8={"final": S.STOCK})})
root_case("both_never_launched", [0.05, None, 0.07], [0.30, 0.40, None], [False, True, False],
          [False, False, True], 2)


def snap_then_late_off(rnd, runs):
    # off s80 launched 3 h late, after a SNAPSHOT of phase s60 at +2 h; off s60 SNAPSHOT at +1 h is no opening
    S.standard_run(rnd, 80, arm="off", launch=T0 + timedelta(hours=3), ledger=S.ledger_with_cmd(0.50))
    rnd.add_note(T0 + timedelta(hours=1), "SNAPSHOT", runs[("off", 60)][0].run_id, note="off look (no opening)")
    rnd.add_note(T0 + timedelta(hours=2), "SNAPSHOT", runs[("pc", 60)][0].run_id, note="phase look")


o = root_case("off_after_phase_snapshot", [0.05, 0.06, 0.07], [0.30, 0.40, 0.50], F, [False, False, True], 4,
              extra=snap_then_late_off, build_off=[0.30, 0.40, None])
o = build("off_after_phase_snapshot_b", [0.05, 0.06, 0.07], [0.30, 0.40, None], extra=snap_then_late_off)[0]
check(o["blindness"]["first_opening"]["kind"] == "SNAPSHOT" and C.arm_of(o["blindness"]["first_opening"]["run"]) == "pc",
      "an off run's SNAPSHOT is not an opening; the first opening is the phase SNAPSHOT")


def off_snapshot_only(rnd, runs):
    S.standard_run(rnd, 80, arm="off", launch=T0 + timedelta(hours=3), ledger=S.ledger_with_cmd(0.50))
    rnd.add_note(T0 + timedelta(hours=1), "SNAPSHOT", runs[("off", 60)][0].run_id, note="off look")


root_case("late_off_blind", [0.05, 0.06, 0.07], [0.30, 0.40, 0.50], F, F, 4, extra=off_snapshot_only,
          build_off=[0.30, 0.40, None])


def late_written_snapshot(rnd, runs):
    # a phase SNAPSHOT at +2 h written into RUNLOG only after the F2-START lines (a late line): still an opening
    S.standard_run(rnd, 80, arm="off", launch=T0 + timedelta(hours=3), ledger=S.ledger_with_cmd(0.50))
    rnd.late.append(S.note(T0 + timedelta(hours=2), "SNAPSHOT", runs[("pc", 42)][0].run_id, note="written late"))


root_case("late_written_snapshot", [0.05, 0.06, 0.07], [0.30, 0.40, 0.50], F, [False, False, True], 4,
          extra=late_written_snapshot, build_off=[0.30, 0.40, None])


def off_b_after_opening(rnd, runs):
    # off s42: first launch faults (8) (the key present), stopped, _b relaunched at +50 min; a phase SNAPSHOT at
    # +45 min came first -> the _b (the seed's run) is not blind
    seed = 42
    first = f"{C.ARM_PREFIX['off']}{seed}"
    run0 = S.Run(first, T0, rnd.pins, sc_over={"hyperparameters.critic_phase": False}, ledger=S.ledger_with_cmd(0.4),
                 cut_at=C.GRID_STEPS[1])
    rnd.add_note(T0, "LAUNCH", first, **S.launch_fields(seed, rnd.pins["branch_head"], "off"))
    S.write_store(rnd.root, "F0", run0, run0.files(), T0 + timedelta(minutes=30))
    rnd.add_note(T0 + timedelta(minutes=40), "STOP", first)
    rnd.add_note(T0 + timedelta(minutes=40), "TRIGGER-d", first)
    rnd.add_note(T0 + timedelta(minutes=45), "DELETED", first)
    rnd.add_note(T0 + timedelta(minutes=45), "SNAPSHOT", runs[("pc", 80)][0].run_id, note="phase look")
    b = first + "_b"
    tb = T0 + timedelta(minutes=50)
    run_b = S.Run(b, tb, rnd.pins, ledger=S.ledger_with_cmd(0.4))
    rnd.add_note(tb, "RELAUNCH", b, **S.launch_fields(seed, rnd.pins["branch_head"], "off", RUN_ID=b))
    end_t = tb + timedelta(hours=15)
    rnd.add_note(end_t, "TRIGGER-a", b)
    rnd.add_note(end_t, "FOUND-GONE", b)
    S.write_store(rnd.root, "F2", run_b, run_b.files(), end_t + timedelta(hours=7))
    rnd.add_note(end_t + timedelta(hours=7), "F2-START", b)
    S.write_panels(rnd.root, run_b)
    S.write_store(rnd.root, "F2", run0, run0.files(), end_t + timedelta(hours=7))
    rnd.add_note(end_t + timedelta(hours=7), "F2-START", first)


o = root_case("off_b_after_opening", [0.05, 0.06, 0.07], [0.30, 0.40, 0.50], F, [False, True, False], 4,
              extra=off_b_after_opening, build_off=[0.30, None, 0.50])
d42 = next(d for d in o["seeds"] if d["arm"] == "off" and d["seed"] == 42)
check(d42["run"].endswith("_b") and o["launch_checks"][f"{C.ARM_PREFIX['off']}42"]["faults"] == [8],
      f"off s42: fault (8) on the first run (key present), the _b is the seed's run ({d42['run']}); its RELAUNCH after "
      f"the phase SNAPSHOT is not blind: {o['blindness']['not_blind']}")

print(f"FAIL count {len(FAILS)}")
for f in FAILS[:20]:
    print("  ", f)
sys.exit(1 if FAILS else 0)
