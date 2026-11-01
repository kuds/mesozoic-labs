"""PC study (phase-critic Colab round): the readings the PC criteria add to the round's machinery -- the ledger's early
signs, TAIL DRIVE, HABIT ELSEWHERE (L/A/T/D), THE ONE TEST, THE READING (lines 1-5) and the comparators (STANCE6's
three runs recomputed from their FMAN-checked F2 copies, score.json and the round's own M1 records).  Report only;
never trains.  Standard library (numpy lazily for a probe file).  Imported by path by score_pc.py and the tests.
PC AMENDMENT (2026-11-01, ST/amend/AMEND_NOTES.txt; the maintainer's CONTROL (b), "make them the test's control"):
THE ONE TEST compares the 3 phase runs with the 3 CONCURRENT SWITCH-OFF runs (phase LOWER; a missing phase run enters
as +infinity, the highest rank; a missing switch-off run as -infinity, the lowest); THE READING's lines are the phase
runs' (as reviewed) with the switch-off runs' classes printed beside; STANCE6 (comparators()) is REPORT ONLY: printed
beside and in the OFF-vs-STANCE6 replication block (replication()), deciding nothing.

Definitions (ST/criteria.txt; [ROUND l.N] = S2/r10round/final_lean/criteria_r10round_final.txt line N):
  CMD_L(e)  the mean over evaluation e's ledger rows of tail_3_pitch_act_command_mean (finite cells only), BS2
            read_bs.ledger_direct's definition (S2/r10round/bias_screen2/tools/read_bs.py lines 621-660, sha256 06f4173d...).
  over-bar  a ledger row with saturated_actuator_fraction > 0.10 (BAR, ROUND l.166), named by saturated_actuator.
  TD        the mean over the run's evaluations of |CMD_L(e)|; an evaluation with no finite cell is MISSING and skipped.
  HABIT ELSEWHERE  LB's O6 rule (S2/r10round/long_blind/criteria_work/criteria_v1.txt lines 208-222) per actuator a other
            than tail_3_pitch_act: L an own alarm, A an ANY ALARM attributing a (a named in at least 5 of the over-bar
            rows of one of the alarm's two evaluations; fixer D3), T >= 2 full-horizon M1 episodes over the
            BAR on a in one sign at one traced checkpoint (an exact sign tie counts once in each sign), D |CMD_a| >= 0.60
            in one sign at >= 2 of the J checkpoints (with the final) or the mean of CMD_a(R1), CMD_a(R2) (the formed one
            if one) >= 0.60 in size; SATURATING = L, A or T; DRIVE ONLY = D alone; FREE = none.  The L/T/D arithmetic is
            measure_lb.habit_summary's (S2/r10round/long_blind/measure/tools/measure_lb.py lines 547-661, sha256
            5fb29b52...), A added as the PC text adds it.
"""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

VERSION = ("pc_common pc-3 (2026-11-01 amendment: THE ONE TEST phase vs the concurrent switch-off runs, MISSING ranks, "
           "the switch-off classes beside, STANCE6 report only; pc-2 fixer: A site >= 5 rows, DRIVEN tail in lines 4-5, "
           "wording, exact sums; PC criteria: early signs, TD, HABIT ELSEWHERE, THE ONE TEST, THE READING)")

S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
S2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
COMP_FETCH = S1 / "foot/stance6/fetch"
COMP_FMAN = COMP_FETCH / "manifest.csv"
COMP_PANELS = S1 / "foot/stance6/panels"
COMP_SCORE = S2 / "r10round/f2/score.json"
COMP_RUNS = {60: "stance6_r10LTSB_s60", 42: "stance6_r10LTSB_s42", 80: "stance6_r10LTSB_s80"}
STANCE6_S42_FIRST_TAIL_ALARM = 3932160        # report only (PC AMENDMENT: STANCE6 is printed beside, deciding nothing)

BAR = 0.10
CMD_LINES = (0.30, 0.60, 0.90)
J_FIRST = 4194304
HABIT_LINE = 0.60
T3 = "tail_3_pitch_act"
TAIL_ACTS = ("tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")
N_SPLITS = math.comb(6, 3)
SUFFIX = "(descriptive: 3 phase-critic runs beside 3 concurrent switch-off runs; KNOWN LIMITS)"     # PC AMENDMENT
A_SITE_MIN_ROWS = 5                            # fixer D3: A needs a named in >= 5 over-bar rows of one of the two
CAVEAT = ("concurrent control: the 6 runs share HEAD, the notebook, the seeds and the schedule, started within the gaps "
          "printed; their exchangeability under no effect is assumed up to seed effects: each phase run and its "
          "same-seed switch-off run start from the same network (A5), a positive within-seed dependence that makes "
          "the unpaired test conservative [inferred]; GPU nondeterminism within today's image remains; same-seed pairs "
          "printed, report only (KNOWN LIMITS)")
ONE_TEST_NOTE = ("p = 1/20 only on complete separation: every phase TD below every switch-off TD; TD's form (the "
                 "whole-run mean; 2 windows looked at) was chosen with STANCE6's values in view, the six runs' values are "
                 "unseen at the pin")
GATE_LABEL = "(GATE CERTIFICATE: final checkpoint, measured on a screen task; not certification)"
NO_RAIL_CHANCE = ("[model, plug-in] if the phase critic changed nothing, no RAIL in 3 runs has chance {off}; report only "
                  "beside it: (2/3)^3 = 0.296 at STANCE6's 1 of 3 and (3/5)^3 = 0.216 at LTSB r9's 2 of 5")
OFF_BESIDE = ("CONCURRENT SWITCH-OFF RUNS (printed beside; they decide no line): TAIL RAIL in {r} (PRESENCE {pr}), PARKED "
              "in {p}, GATE PASS in {g}, SATURATING habit elsewhere in {hs}, DRIVE ONLY in {hd}, of 3; {u} not read")
LINE_HEAD = {
    1: "THE TAIL RAIL FORMED IN {r} OF 3 PHASE-CRITIC RUNS {basis}: THE CRITIC'S PHASE INPUT FROM STEP 0 DID NOT KEEP "
       "EVERY RUN OFF THE TAIL RAIL",
    2: "INCOMPLETE: NO TAIL RAIL IN THE {read} PHASE-CRITIC RUN(S) READ TO 12M; {u} NOT READ (LOST OR STOPPED WITHOUT A "
       "RAIL): NOT A NO-RAIL READING",
    3: "NO TAIL RAIL; THE TAIL DRIVE PERSISTS: PARKED IN {p} OF 3 PHASE-CRITIC RUNS",
    4: "NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, THE STANCE GATE CERTIFICATE PASSED IN 3 OF 3 "
       "(GATE CERTIFICATE: final checkpoint, measured on a screen task; not certification), NO SATURATING HABIT ON "
       "ANOTHER ACTUATOR{driven}: CONSISTENT WITH THE PHASE CRITIC REMOVING THE HABIT, NOT CONFIRMATORY",
    5: "NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, WITH: {items}{driven}",
}
BASIS_LEDGER = "(ledger)"
BASIS_LOST = "(ledger, a LOST run)"
BASIS_END_DRIVE = "(end-only, drive-backed)"
BASIS_END_WEAK = ("(end-only, not drive-backed: weak evidence; false-rail chance if the phase critic removed the habit "
                  "0.003-0.11 per round, up to 0.31 (ROUND's EVID figures) [inferred, EVID])")


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- the ledger
def ledger_rows(raw: bytes):
    """(header, rows) of eval_episodes.csv bytes."""
    rd = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    return list(rd.fieldnames or []), list(rd)


def per_evaluation(rows, actuators=()) -> dict:
    """Per evaluation (integer timesteps), BS2 ledger_direct's quantities plus returns, falls, screen_v2_clean and the
    evaluation command mean of each named actuator (report only)."""
    by = {}
    for r in rows:
        t = num(r.get("timesteps"))
        if t is not None and t == int(t):
            by.setdefault(int(t), []).append(r)
    per = {}
    for s in sorted(by):
        rs = by[s]
        cmd = [c for c in (num(r.get("tail_3_pitch_act_command_mean")) for r in rs) if c is not None]
        over = [r for r in rs if (num(r.get("saturated_actuator_fraction")) or 0.0) > BAR]
        rets = [x for x in (num(r.get("return")) for r in rs) if x is not None]
        lens = [x for x in (num(r.get("length")) for r in rs) if x is not None]
        acts = {}
        for a in actuators:
            v = [x for x in (num(r.get(a + "_command_mean")) for r in rs) if x is not None]
            acts[a] = sum(v) / len(v) if v else None
        over_by = {}
        for r in over:
            k = str(r.get("saturated_actuator"))
            over_by[k] = over_by.get(k, 0) + 1
        per[s] = dict(n=len(rs), cmd_l=sum(cmd) / len(cmd) if cmd else None, n_cmd=len(cmd), n_over=len(over),
                      n_tail_over=sum(1 for r in over if r.get("saturated_actuator") in TAIL_ACTS),
                      n_t3_over=sum(1 for r in over if r.get("saturated_actuator") == T3), over_by=over_by,
                      ret=sum(rets) / len(rets) if rets else None, falls=sum(1 for x in lens if x < 1000),
                      clean=sum(1 for r in rs if str(r.get("screen_v2_clean")).strip().lower() in ("true", "1", "1.0")),
                      acts=acts)
    return per


def early_signs(per: dict, m5res: dict | None = None) -> dict:
    """THE LEDGER'S EARLY SIGNS (Q1; printed, never a class)."""
    f = {"first_t3_over": next((s for s, e in per.items() if e["n_t3_over"] > 0), None),
         "FOB_tail": next((s for s, e in per.items() if e["n_tail_over"] > 0), None),
         "first_over_any": next((s for s, e in per.items() if e["n_over"] > 0), None)}
    for line in CMD_LINES:
        f[f"cmd_l_ge_{line:.2f}"] = next((s for s, e in per.items() if e["cmd_l"] is not None and e["cmd_l"] >= line),
                                         None)
    cl = [(s, e["cmd_l"]) for s, e in per.items() if e["cmd_l"] is not None]
    out = dict(firsts=f, n_cmd_l_ge_060=sum(1 for _s, v in cl if v >= 0.60), n_evaluations_with_cmd=len(cl),
               cmd_l_max=(max(cl, key=lambda x: x[1]) if cl else None), cmd_l_last=(cl[-1] if cl else None))
    if m5res is not None:
        out["m5_tail_flagged"] = list(m5res.get("tail_flagged") or [])
        out["m5_tail_alarms"] = list(m5res.get("tail_alarms") or [])
    return out


def tail_drive(per: dict, j_first: int = J_FIRST) -> dict:
    """TD: the mean over the evaluations of |CMD_L|; MISSING evaluations (no finite cell) listed and skipped.  The
    J-window form (evaluations from j_first) beside it, report only."""
    used = [(s, abs(e["cmd_l"])) for s, e in per.items() if e["cmd_l"] is not None]
    missing = [s for s, e in per.items() if e["cmd_l"] is None]
    jj = [v for s, v in used if s >= j_first]
    return dict(td=math.fsum(v for _s, v in used) / len(used) if used else None, n=len(used), missing=missing,
                td_j=math.fsum(jj) / len(jj) if jj else None, n_j=len(jj))


# ---------------------------------------------------------------------------------------------------- THE ONE TEST
def one_test(phase: list, off: list) -> dict:
    """THE ONE TEST (PC AMENDMENT: the 3 phase runs against the 3 CONCURRENT SWITCH-OFF runs, decided before any data).
    Exact one-sided unpaired permutation test over the C(6,3) = 20 splits of the six TDs, H1 phase LOWER: p = (splits
    whose 3-sum is <= the phase 3-sum) / 20, ties counting against H1; LOWER only at p = 1/20 (complete separation).
    MISSING rules against H1: a phase run without a TD enters as +infinity (above every value: the highest rank), a
    switch-off run without one as -infinity (below every value: the lowest rank); +infinity is refused on the off side
    and -infinity on the phase side.  A 3-sum is the pair (the number of +infinity in it minus the number of -infinity,
    the exact sum of its finite values), ordered lexicographically: the limit of +M / -M for a large M, so a missing
    run moves a split by a whole rank and never ties by itself; any missing run gives k >= 11.  Finite sums are exact:
    rational arithmetic on the float values (fractions.Fraction of each float), compared exactly (fixer I1: math.fsum
    is correctly rounded, not exact, and could misorder near ties)."""
    if len(phase) != 3 or len(off) != 3:
        raise ValueError("THE ONE TEST needs 3 phase and 3 switch-off values")
    vals = [float(x) for x in phase] + [float(x) for x in off]
    if any(math.isnan(v) for v in vals):
        raise ValueError("THE ONE TEST: a NaN value")
    if any(v == -math.inf for v in vals[:3]) or any(v == math.inf for v in vals[3:]):
        raise ValueError("THE ONE TEST: a missing phase run enters at +infinity and a missing switch-off run at "
                         "-infinity, never the other way")

    def exact_sum(xs):
        n_inf = sum(1 for v in xs if v == math.inf) - sum(1 for v in xs if v == -math.inf)
        return (n_inf, sum((Fraction(v) for v in xs if math.isfinite(v)), Fraction(0)))

    obs = exact_sum(vals[:3])
    k = 0
    for idx in itertools.combinations(range(6), 3):
        tot = exact_sum([vals[i] for i in idx])
        k += tot <= obs
    p = Fraction(k, N_SPLITS)
    text = ("LOWER THAN EVERY CONCURRENT SWITCH-OFF RUN (exact permutation p = 1/20)" if k == 1
            else f"NOT SEPARATED (p = {k}/20)")
    return dict(k=k, n_splits=N_SPLITS, p=f"{k}/{N_SPLITS}", p_float=float(p), lower=k == 1, text=text,
                note=ONE_TEST_NOTE, phase=[None if not math.isfinite(v) else v for v in vals[:3]],
                phase_missing=[not math.isfinite(v) for v in vals[:3]],
                off=[None if not math.isfinite(v) else v for v in vals[3:]],
                off_missing=[not math.isfinite(v) for v in vals[3:]], caveat=CAVEAT)


def fisher_lower(a: int, n: int, ref_a: int, ref_n: int) -> float:
    """One-sided Fisher exact p (PC fewer): P(X <= a), hypergeometric with n of N = n + ref_n drawn, K = a + ref_a."""
    N, K = n + ref_n, a + ref_a
    tot = math.comb(N, n)
    return sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(0, a + 1) if 0 <= n - x <= N - K) / tot


# ---------------------------------------------------------------------------------------------------- HABIT ELSEWHERE
def habit_elsewhere(C, header, rows, classes: dict, m1_records: dict, js, r1, extra_labels=()) -> dict:
    """Per actuator a other than tail_3_pitch_act: L, A, T, D (module docstring) with their evidence, and the status
    SATURATING / DRIVE ONLY / FREE; the run's status is the strongest of its actuators'.  m1_records: label ->
    lane A's m1_summary.json (episodes[].actuators[name].over_bar / sign / full_horizon; cmd_all_actuators)."""
    names = C.ACTUATOR_NAMES
    over_by_eval = {n: {} for n in names}
    for r in rows:
        s = C.grid_step(r.get("timesteps"))
        if s is None:
            continue
        x = C.decode_row(r, header)
        if not x["any_over"] or x["act"] is None:
            continue
        n = names[x["act"]]
        over_by_eval[n][s] = over_by_eval[n].get(s, 0) + 1
    labels = sorted({*(str(j) for j in js), *([str(r1)] if r1 else []), "final", *(str(x) for x in extra_labels)},
                    key=lambda x: (x == "final", int(x) if x != "final" else 0))
    jl = [str(j) for j in js] + ["final"]
    any_alarms = classes.get("any_alarms") or []
    out = {}
    for a, n in enumerate(names):
        if n == T3:
            continue
        ev = over_by_eval[n]
        flagged = sorted(s for s, k in ev.items() if k >= 5)
        own = [s for s in flagged if s - C.GRID in flagged]
        any_named = [x.get("step") for x in any_alarms if n in (x.get("actuators") or {})
                     and x.get("step") is not None
                     and max(ev.get(x["step"] - C.GRID, 0), ev.get(x["step"], 0)) >= A_SITE_MIN_ROWS]
        m1 = {}
        for lab in labels:
            rec = m1_records.get(lab)
            if not rec:
                continue
            cm = (rec.get("cmd_all_actuators") or {}).get(str(a))
            pos = neg = tie = 0
            for ep in rec.get("episodes", []):
                ae = (ep.get("actuators") or {}).get(n) or {}
                if ep.get("full_horizon") and ae.get("over_bar"):
                    sg = ae.get("sign")
                    pos += sg == "+"
                    neg += sg == "-"
                    tie += sg == "="
            m1[lab] = dict(cmd=cm, over_bar_full_plus=pos, over_bar_full_minus=neg, over_bar_full_tie=tie)
        t_at = [lab for lab, v in m1.items() if (v["over_bar_full_plus"] + v["over_bar_full_tie"]) >= 2
                or (v["over_bar_full_minus"] + v["over_bar_full_tie"]) >= 2]
        d_why = []
        for sg in (1, -1):
            hits = [lab for lab in jl if lab in m1 and m1[lab]["cmd"] is not None and sg * m1[lab]["cmd"] >= HABIT_LINE]
            x = m1.get(str(r1), {}).get("cmd") if r1 else None
            y = m1.get("final", {}).get("cmd")
            mean = (x + y) / 2 if (x is not None and y is not None) else (x if x is not None else y)
            if len(hits) >= 2:
                d_why.append(f"{'+' if sg > 0 else '-'}: CMD >= 0.60 at {hits}")
            if mean is not None and sg * mean >= HABIT_LINE:
                d_why.append(f"{'+' if sg > 0 else '-'}: mean of CMD(R1), CMD(R2) {mean:+.4f}")
        sites = [s for s, hold in (("L", bool(own)), ("A", bool(any_named)), ("T", bool(t_at)), ("D", bool(d_why)))
                 if hold]
        status = ("SATURATING" if any(s in sites for s in ("L", "A", "T")) else "DRIVE ONLY" if "D" in sites
                  else "FREE")
        out[n] = dict(index=a, sites=sites, status=status, own_alarms=own, any_alarms_naming=any_named,
                      trace_sites=t_at, drive=d_why, ledger_over_total=sum(ev.values()),
                      first_over=min(ev) if ev else None, last_over=max(ev) if ev else None, m1=m1)
    sat = sorted(n for n, v in out.items() if v["status"] == "SATURATING")
    drv = sorted(n for n, v in out.items() if v["status"] == "DRIVE ONLY")
    run_status = "SATURATING" if sat else ("DRIVE ONLY" if drv else "FREE")
    return dict(per_actuator=out, saturating=sat, drive_only=drv, status=run_status, labels=labels,
                labels_without_m1=[lab for lab in labels if lab not in m1_records])


# ---------------------------------------------------------------------------------------------------- THE READING
def arm_counts(per_seed: list, presence_rails: list) -> dict:
    """PC AMENDMENT: the counts THE READING forms, for any arm's three runs (the switch-off runs' are printed beside the
    phase runs' line and decide nothing): r (COMPLETE RAIL plus PRESENCE RAILS), u (not COMPLETE, or UNREAD), p, g, hs,
    hd, and n_read (runs read to a TAIL CLASS other than UNREAD, plus PRESENCE RAILS)."""
    r = u = p = g = hs = hd = n = 0
    for d in per_seed:
        if d.get("verdict") != "COMPLETE":
            u += 1
            continue
        c = d["classes"]
        r += c["cls"] == "RAIL"
        u += c["cls"] == "UNREAD"
        p += c["cls"] == "PARKED"
        n += c["cls"] != "UNREAD"
        if c["cls"] != "UNREAD":
            g += c["gate"] == "PASS"
        h = (d.get("habit") or {}).get("status")
        hs += h == "SATURATING"
        hd += h == "DRIVE ONLY"
    return dict(r=r + len(presence_rails), r_complete=r, presence_rails=len(presence_rails), u=u, p=p, g=g, hs=hs,
                hd=hd, n_read=n + len(presence_rails))


def _no_rail_chance(off: dict | None) -> str:
    """PC AMENDMENT: the plug-in no-RAIL chance at the concurrent switch-off runs' rail share (r of n_read), STANCE6's
    and LTSB r9's printed beside, report only ([model, plug-in], as ROUND's TABLE E printed such figures)."""
    if not off or not off.get("n_read"):
        return NO_RAIL_CHANCE.format(off="not formed (no switch-off run read to a class)")
    r, n = off["r"], off["n_read"]
    ch = (1 - Fraction(r, n)) ** 3
    return NO_RAIL_CHANCE.format(off=f"(1 - {r}/{n})^3 = {float(ch):.3f} at the concurrent switch-off runs' {r} of {n}")


def reading(per_seed: list, presence_rails: list, off_per_seed: list | None = None, off_presence_rails=()) -> dict:
    """THE READING (PC text): the first line that applies, with the counts.  per_seed: one dict per seed of the PHASE
    runs with verdict ('COMPLETE' or other) and, when COMPLETE, classes (score_r10round.classes_of) and habit
    (habit_elsewhere); the line is theirs alone.  PC AMENDMENT: off_per_seed / off_presence_rails, the concurrent
    switch-off runs (same form): their counts and classes are printed beside (OFF_BESIDE), each line-1 run's first TAIL
    ALARM beside the same-seed switch-off run's, and the plug-in no-RAIL chance uses their rail share; none of it moves
    the line."""
    r = u = p = g = hs = hd = 0
    basis, firsts, driven = [], [], []
    off = arm_counts(off_per_seed, list(off_presence_rails)) if off_per_seed is not None else None
    off_first, off_classes = {}, []
    for d in off_per_seed or []:
        c = d.get("classes")
        if d.get("verdict") == "COMPLETE" and c:
            off_first[d.get("seed")] = (c.get("tail_alarms") or [None])[0]
            off_classes.append(dict(seed=d.get("seed"), run=d.get("run"), cls=c.get("cls"), kind=c.get("kind"),
                                    gate=c.get("gate"), outcome=c.get("outcome"),
                                    habit=(d.get("habit") or {}).get("status")))
        else:
            off_classes.append(dict(seed=d.get("seed"), run=d.get("run"), verdict=d.get("verdict")))
    for pr in off_presence_rails or ():
        off_first.setdefault(pr.get("seed"), (pr.get("detail") or {}).get("step"))
    for d in per_seed:
        if d.get("verdict") != "COMPLETE":
            u += 1
            continue
        c = d["classes"]
        if c["cls"] == "RAIL":
            r += 1
            basis.append("ledger" if c["tail_alarms"] else "end")
            firsts.append(dict(seed=d.get("seed"), run=d.get("run"), first_tail_alarm=(c["tail_alarms"] or [None])[0],
                               kind=c["kind"], beside_off_same_seed=off_first.get(d.get("seed"), "not read")))
        elif c["cls"] == "UNREAD":
            u += 1
        elif c["cls"] == "PARKED":
            p += 1
        if c["cls"] != "UNREAD":
            g += c["gate"] == "PASS"
        for x in c.get("driven") or []:                   # fixer D4: a DRIVEN tail is printed in lines 4 and 5
            driven.append(f"{x.get('actuator')} MEAN_a {float(x.get('mean')):+.4f} IN s{d.get('seed')}")
        h = (d.get("habit") or {}).get("status")
        hs += h == "SATURATING"
        hd += h == "DRIVE ONLY"
    r_total = r + len(presence_rails)
    for pr in presence_rails:
        firsts.append(dict(run=pr.get("run"), presence_rail=True,
                           first_tail_alarm=(pr.get("detail") or {}).get("step"),
                           beside_off_same_seed=off_first.get(pr.get("seed"), "not read")))
    if r_total >= 1:
        if "ledger" in basis:
            b = BASIS_LEDGER
        elif presence_rails:
            b = BASIS_LOST
        else:
            ends = [x for d in per_seed if d.get("classes") and d["classes"].get("kind") == "RAIL-END-ONLY"
                    for x in d["classes"]["trace_rail"]]
            b = BASIS_END_DRIVE if any(x["drive_backed"] for x in ends) else BASIS_END_WEAK
        line, head = 1, LINE_HEAD[1].format(r=r_total, basis=b)
    elif u >= 1:
        line, head = 2, LINE_HEAD[2].format(read=3 - u, u=u)
    elif p >= 1:
        line, head = 3, LINE_HEAD[3].format(p=p)
    elif g == 3 and hs == 0:
        line, head = 4, LINE_HEAD[4].format(driven=_driven_clause(driven))
    else:
        items = [x for x in (f"{3 - g} GATE FAIL" if 3 - g else None,
                             f"{hs} WITH A SATURATING HABIT ON ANOTHER ACTUATOR" if hs else None) if x]
        line, head = 5, LINE_HEAD[5].format(items=", ".join(items), driven=_driven_clause(driven))
    named = []
    if line == 5:
        for d in per_seed:
            c = d.get("classes")
            if not c:
                continue
            if c["gate"] == "FAIL":
                named.append(f"{d.get('run')}: GATE FAIL {c.get('gate_why')}")
            hb = d.get("habit") or {}
            if hb.get("saturating"):
                named.append(f"{d.get('run')}: SATURATING habit on {hb['saturating']}")
    if off is not None:
        off_text = OFF_BESIDE.format(r=off["r"], pr=off["presence_rails"], p=off["p"], g=off["g"], hs=off["hs"],
                                     hd=off["hd"], u=off["u"])
    else:
        off_text = "CONCURRENT SWITCH-OFF RUNS: not given"
    return dict(line=line, text=f"{head} {SUFFIX}", r=r_total, r_complete=r, presence_rails=len(presence_rails), u=u,
                p=p, g=g, hs=hs, hd=hd, first_tail_alarms=firsts, named=named, driven_tails=driven,
                off=off, off_text=off_text, off_classes=off_classes,
                no_rail_chance=_no_rail_chance(off) if line in (3, 4, 5) else None)


def _driven_clause(driven: list) -> str:
    """Fixer D4: ", WITH A DRIVEN TAIL (<actuator> MEAN_a <value> IN s<seed>; ...)" or "" (ROUND's DRIVEN, l.278-279,
    report only: it changes no class and no line)."""
    return f", WITH A DRIVEN TAIL ({'; '.join(driven)})" if driven else ""


# ---------------------------------------------------------------------------------------------------- the comparators
def _fman_rows():
    with open(COMP_FMAN, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _checked_copy(run_id: str, rel: str, rows) -> tuple[Path, str]:
    """The F2 .copy1 of rel, its sha256 equal to the F2 fetch-manifest COPY row and the run's artifact_manifest entry
    (DESIGN/comparators.py check_copy's rule); raises ValueError otherwise."""
    copies = COMP_FETCH / "F2" / run_id / "copies"
    p = copies / (rel + ".copy1")
    got = sha256_file(p)
    f2 = [r for r in rows if r["fetch"] == "F2" and r["run_id"] == run_id and r["path"] == rel
          and r["status"] == "COPY" and r["copy"] == "1"]
    man = json.loads((copies / "artifact_manifest.json.copy1").read_text(encoding="utf-8"))
    ent = [e for e in man["files"] if e["path"] == rel]
    want = ent[0]["sha256"].split(":", 1)[-1] if ent else None
    if not (f2 and f2[-1]["sha256"] == got and want == got):
        raise ValueError(f"{run_id} {rel}: copy {got} does not match its manifests")
    return p, got


def comparators(C, pins: dict | None = None) -> dict:
    """STANCE6's three runs, recomputed (never transcribed): TD (all; J window), early signs, the round's classes
    (score.json at its pinned sha256), HABIT ELSEWHERE from the round's own M1 records (S1/foot/stance6/panels), the
    on-policy tail_3 command (probe, last 8 rows; report only), returns and falls.  Raises ValueError on a sha256
    mismatch with the pins (pins["comparators"]) or a manifest."""
    want = (pins or {}).get("comparators") or {}
    got_score, got_fman = sha256_file(COMP_SCORE), sha256_file(COMP_FMAN)
    if want and (want.get("score_json_sha256") != got_score or want.get("fetch_manifest_sha256") != got_fman):
        raise ValueError(f"comparator inputs: score.json {got_score} / manifest {got_fman} are not the pins {want}")
    rows = _fman_rows()
    score = json.loads(COMP_SCORE.read_text(encoding="utf-8"))
    per_seed = {d["seed"]: d for d in score["per_seed"]}
    out = dict(score_json_sha256=got_score, fetch_manifest_sha256=got_fman, runs={})
    for seed, run_id in COMP_RUNS.items():
        p, h = _checked_copy(run_id, "01_stance/eval_episodes.csv", rows)
        header, lrows = ledger_rows(p.read_bytes())
        per = per_evaluation(lrows, C.ACTUATOR_NAMES)
        res = C.m5(header, lrows)
        cls = per_seed[seed]["classes"]
        js = list(C.J_STEPS)
        m1 = {}
        m1_sha = {}
        for lab in [str(j) for j in js] + ["final"]:
            f = COMP_PANELS / run_id / lab / "m1" / "m1_summary.json"
            if f.is_file():
                m1[lab] = json.loads(f.read_text(encoding="utf-8"))
                m1_sha[lab] = sha256_file(f)
        habit = habit_elsewhere(C, header, lrows, cls, m1, js, C.R1_STEP,
                                [str(v) for v in (cls.get("resolution") or []) if isinstance(v, (int, str))])
        probe = None
        try:
            pp, _ph = _checked_copy(run_id, "01_stance/policy_probe.npz", rows)
            import numpy as np

            with np.load(pp, allow_pickle=False) as z:
                mu = np.asarray(z["rollout_mu_mean"], dtype=np.float64)
                ut = np.asarray(z["update_timesteps"])
            probe = dict(last8_mean=float(np.mean(mu[-8:, 14])), rows=[int(ut[-8]), int(ut[-1])])
        except Exception as exc:  # noqa: BLE001 - report only
            probe = dict(error=f"{type(exc).__name__}: {exc}")
        rets = [e["ret"] for e in per.values() if e["ret"] is not None]
        out["runs"][seed] = dict(
            run=run_id, ledger_sha256=h, evaluations=len(per), td=tail_drive(per), early=early_signs(per, res),
            classes=dict(cls=cls.get("cls"), kind=cls.get("kind"), tail_alarms=len(cls.get("tail_alarms") or []),
                         first_tail_alarm=(cls.get("tail_alarms") or [None])[0], gate=cls.get("gate"),
                         gate_why=cls.get("gate_why"), outcome=cls.get("outcome"),
                         any_alarms=len(cls.get("any_alarms") or []), mean=cls.get("mean")),
            habit=dict(status=habit["status"], saturating=habit["saturating"], drive_only=habit["drive_only"],
                       sites={n: v["sites"] for n, v in habit["per_actuator"].items() if v["sites"]},
                       m1_sha256=m1_sha, labels_without_m1=habit["labels_without_m1"]),
            probe_tail3=probe, mean_return=sum(rets) / len(rets) if rets else None,
            falls=sum(e["falls"] for e in per.values()), rows=sum(e["n"] for e in per.values()))
    return out


# ---------------------------------------------------------------------------------------------------- PC AMENDMENT
REPLICATION_NOTE = ("REPORT ONLY: with the switch OFF the code equals 8737cd3d bitwise (A6, measured on CPU), so the "
                    "concurrent switch-off runs replicate STANCE6's configuration on today's Colab image; this block "
                    "prints each switch-off run beside STANCE6's same seed and decides nothing")


def replication(off_per_seed: list, comps: dict | None) -> dict:
    """PC AMENDMENT, REPORT ONLY: the OFF-vs-STANCE6 replication block.  Per seed: the switch-off run's TAIL CLASS (kind),
    first TAIL ALARM, GATE, RUN OUTCOME, HABIT ELSEWHERE, TD (and the J window) and CMD_L >= 0.60 count beside
    STANCE6's same seed (comparators(), recomputed from its pinned files), and the arm totals (RAIL, PARKED, GATE PASS,
    TD mean).  Seeds do not pair runs (GPU training is not bit-reproducible): nothing here is tested."""
    rows = []
    cr = (comps or {}).get("runs") or {}
    for d in off_per_seed:
        s = d.get("seed")
        c = d.get("classes") or {}
        x = dict(seed=s, run=d.get("run"), verdict=d.get("verdict"))
        if d.get("verdict") == "COMPLETE" and c:
            x["off"] = dict(cls=c.get("cls"), kind=c.get("kind"), first_tail_alarm=(c.get("tail_alarms") or [None])[0],
                            gate=c.get("gate"), outcome=c.get("outcome"), habit=(d.get("habit") or {}).get("status"),
                            saturating=(d.get("habit") or {}).get("saturating"), td=(d.get("td") or {}).get("td"),
                            td_j=(d.get("td") or {}).get("td_j"),
                            n_cmd_l_ge_060=(d.get("early") or {}).get("n_cmd_l_ge_060"))
        cs = cr.get(s)
        if cs:
            x["stance6"] = dict(run=cs["run"], cls=cs["classes"]["cls"], kind=cs["classes"]["kind"],
                                first_tail_alarm=cs["classes"]["first_tail_alarm"], gate=cs["classes"]["gate"],
                                outcome=cs["classes"]["outcome"], habit=cs["habit"]["status"],
                                saturating=cs["habit"]["saturating"], td=cs["td"]["td"], td_j=cs["td"]["td_j"],
                                n_cmd_l_ge_060=cs["early"]["n_cmd_l_ge_060"])
        rows.append(x)

    def totals(key):
        xs = [x[key] for x in rows if key in x]
        tds = [v["td"] for v in xs if v.get("td") is not None]
        return dict(read=len(xs), rail=sum(v["cls"] == "RAIL" for v in xs), parked=sum(v["cls"] == "PARKED" for v in xs),
                    gate_pass=sum(v["gate"] == "PASS" for v in xs), td_mean=(math.fsum(tds) / len(tds)) if tds else None)

    return dict(note=REPLICATION_NOTE, rows=rows, totals=dict(off=totals("off"), stance6=totals("stance6")),
                stance6_available=bool(cr))
