"""Design-lens reviewer: an independent implementation of the PC criteria's new readings, written from the text of
ST/criteria.txt (lines 138-212) and ROUND's definitions it cites, without reusing pc_common's code.  Report only.

    PY -I -B indep_score.py           # runs every comparison and prints the result

Compares, against ST/tools/pc/pc_common.py (imported by path only to be compared):
  1. THE READING (lines 181-199) on every combination of per-seed states (exhaustive) plus PRESENCE rails;
  2. THE ONE TEST (lines 202-212) with exact rationals, on random and adversarial values, ties and +infinity;
  3. TD (lines 154-157) recomputed from the raw STANCE6 ledgers (FMAN-checked copies), vs DESIGN/comparators.out;
  4. HABIT ELSEWHERE L/A/T/D (lines 143-148) on STANCE6's real ledgers + M1 records + score.json any_alarms, and on
     random synthetic ledgers / M1 records; CMD_a recomputed from M1 episodes (window means of episodes > 200 steps).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import itertools
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

ST = Path("/home/user/work/pc_study")
PC = ST / "tools" / "pc"
S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
S2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
FETCH = S1 / "foot/stance6/fetch"
PANELS = S1 / "foot/stance6/panels"
SCORE = S2 / "r10round/f2/score.json"
RUNS = {60: "stance6_r10LTSB_s60", 42: "stance6_r10LTSB_s42", 80: "stance6_r10LTSB_s80"}

NAMES = ("neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
         "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act", "tail_1_pitch_act",
         "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")
T3 = 14
TAIL = (11, 12, 13, 14)
GRID = 131072
NEVAL = 91
J = (4194304, 5242880, 6291456, 8388608, 10485760)
R1 = 10485760


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


sys.dont_write_bytecode = True
if "--pc" in sys.argv:
    PC = Path(sys.argv[sys.argv.index("--pc") + 1])
QUICK = "--quick" in sys.argv
C = load("r10round_common", PC / "r10round_common.py")
P = load("pc_common", PC / "pc_common.py")


def fnum(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# ============================================================================== 1. THE READING (my own, from the text)
SUFFIX = "(descriptive: 3 runs against the stance6 round's 3, a historical comparison; KNOWN LIMITS)"


def my_reading(seeds, presence):
    """seeds: list of 3 dicts {complete: bool, cls, gate, habit, alarm: bool, endonly_drive: bool}; presence: count."""
    r = sum(1 for s in seeds if s["complete"] and s["cls"] == "RAIL") + presence
    u = sum(1 for s in seeds if (not s["complete"]) or s["cls"] == "UNREAD")
    p = sum(1 for s in seeds if s["complete"] and s["cls"] == "PARKED")
    g = sum(1 for s in seeds if s["complete"] and s["gate"] == "PASS")
    hs = sum(1 for s in seeds if s["complete"] and s["habit"] == "SATURATING")
    if r >= 1:
        if any(s["complete"] and s["cls"] == "RAIL" and s["alarm"] for s in seeds):
            b = "(ledger)"
        elif presence:
            b = "(ledger, a LOST run)"
        elif any(s["complete"] and s["cls"] == "RAIL" and not s["alarm"] and s["endonly_drive"] for s in seeds):
            b = "(end-only, drive-backed)"
        else:
            b = "(end-only, not drive-backed"
        head = (f"THE TAIL RAIL FORMED IN {r} OF 3 PHASE-CRITIC RUNS {b}")
        return 1, head
    if u >= 1:
        return 2, (f"INCOMPLETE: NO TAIL RAIL IN THE {3 - u} PHASE-CRITIC RUN(S) READ TO 12M; {u} NOT READ (LOST OR "
                   "STOPPED WITHOUT A RAIL): NOT A NO-RAIL READING " + SUFFIX)
    if p >= 1:
        return 3, f"NO TAIL RAIL; THE TAIL DRIVE PERSISTS: PARKED IN {p} OF 3 PHASE-CRITIC RUNS " + SUFFIX
    if g == 3 and hs == 0:
        return 4, ("NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, THE STANCE GATE CERTIFICATE PASSED IN 3 OF 3, NO "
                   "SATURATING HABIT ON ANOTHER ACTUATOR " + SUFFIX)
    items = []
    if 3 - g:
        items.append(f"{3 - g} GATE FAIL")
    if hs:
        items.append(f"{hs} WITH A SATURATING HABIT ON ANOTHER ACTUATOR")
    return 5, "NO TAIL RAIL OR PARK IN 3 OF 3 PHASE-CRITIC RUNS, WITH: " + ", ".join(items) + " " + SUFFIX


def to_pc(seed, s):
    d = dict(seed=seed, run=f"x{seed}", verdict="COMPLETE" if s["complete"] else s.get("why", "LOST"))
    if s["complete"]:
        kind = None
        if s["cls"] == "RAIL":
            kind = "RAIL-LEDGER-PERSISTENT" if s["alarm"] else "RAIL-END-ONLY"
        d["classes"] = dict(cls=s["cls"], kind=kind, tail_alarms=[3932160] if (s["cls"] == "RAIL" and s["alarm"]) else [],
                            gate=s["gate"], gate_why=None if s["gate"] == "PASS" else ["x"],
                            trace_rail=[dict(drive_backed=s["endonly_drive"])] if s["cls"] == "RAIL" else [])
        d["habit"] = dict(status=s["habit"], saturating=["neck_pitch_act"] if s["habit"] == "SATURATING" else [])
    return d


def check_reading():
    states = []
    for complete in (True, False):
        if not complete:
            for why in ("LOST", "NEVER LAUNCHED"):
                states.append(dict(complete=False, cls=None, gate=None, habit=None, alarm=False, endonly_drive=False,
                                   why=why))
            continue
        for cls in ("RAIL", "UNREAD", "PARKED", "CLEAR"):
            for gate in ("PASS", "FAIL"):
                if cls == "UNREAD" and gate == "PASS":
                    continue                       # a STOPPED run is GATE FAIL (ROUND l.280-283)
                for habit in ("FREE", "DRIVE ONLY", "SATURATING"):
                    for alarm in ((True, False) if cls == "RAIL" else (False,)):
                        for ed in ((True, False) if (cls == "RAIL" and not alarm) else (False,)):
                            states.append(dict(complete=True, cls=cls, gate=gate, habit=habit, alarm=alarm,
                                               endonly_drive=ed))
    n = bad = 0
    lines = {}
    examples = []
    for combo in itertools.product(states, repeat=3):
        for presence in (0, 1, 2):
            n += 1
            mine = my_reading(combo, presence)
            pcs = [to_pc(s, st) for s, st in zip((60, 42, 80), combo)]
            pr = [dict(run="lost", detail={"step": 1}, fetch="F2")] * presence
            got = P.reading(pcs, pr)
            ok = got["line"] == mine[0] and (got["text"].startswith(mine[1]) if mine[0] == 1 else got["text"] == mine[1])
            lines[mine[0]] = lines.get(mine[0], 0) + 1
            if not ok:
                bad += 1
                if len(examples) < 5:
                    examples.append((combo, presence, mine, got["line"], got["text"]))
    print(f"READING: {n} combinations (states {len(states)}^3 x presence 0-2); disagreements {bad}; lines {lines}")
    for e in examples:
        print("   DISAGREE", e)
    return bad


# ============================================================================== 2. THE ONE TEST
def my_one_test(pc, comp):
    vals = list(pc) + list(comp)
    inf = [not math.isfinite(v) for v in vals]
    fr = [None if i else Fraction(v) for v, i in zip(vals, inf)]

    def ssum(idx):
        if any(inf[i] for i in idx):
            return None                                    # +infinity
        return sum(fr[i] for i in idx)

    obs = ssum((0, 1, 2))
    k = 0
    for idx in itertools.combinations(range(6), 3):
        s = ssum(idx)
        if obs is None:
            k += 1                                         # every sum <= +infinity (ties count against H1)
        elif s is not None and s <= obs:
            k += 1
    return k


def check_one_test():
    comp = [0.38822738704353094, 0.6353434591453841, 0.15433899719014801]
    rng = random.Random(20261031)
    bad = 0
    cases = []
    for _ in range(2000 if QUICK else 20000):
        pc = []
        for _j in range(3):
            c = rng.random()
            if c < 0.05:
                pc.append(math.inf)
            elif c < 0.25:
                pc.append(rng.choice(comp))                # exact ties with a comparator
            elif c < 0.35:
                pc.append(comp[2] * (1 - 1e-15))           # just below s80
            else:
                pc.append(rng.uniform(0, 1))
        cases.append(pc)
    cases += [[0.0, 0.0, 0.0], [comp[2]] * 3, [comp[2], 0, 0], [0.1, 0.1, 0.2], [0.01, 0.02, 0.5], [0.01, 0.02, 0.7],
              [comp[0], comp[1], comp[2]], [math.inf] * 3]
    seen = {}
    for pc in cases:
        k_mine = my_one_test(pc, comp)
        r = P.one_test(pc, comp)
        seen[k_mine] = seen.get(k_mine, 0) + 1
        near = any(v != c and math.isfinite(v) and abs(v - c) < 1e-12 for v in pc for c in comp)
        if (r["k"] != k_mine or r["lower"] != (k_mine == 1)) and not (QUICK and near):
            bad += 1
            if bad < 5:
                print("   ONE TEST DISAGREE", pc, k_mine, r["k"])
    # placements are NOT a function of rank alone for the sum statistic (DESIGN_NOTES 2.4 / comparators.out table):
    same_place = {tuple(pc): my_one_test(pc, comp) for pc in ([0.01, 0.02, 0.5], [0.1, 0.12, 0.2],
                                                                [0.01, 0.02, 0.7], [0.1, 0.12, 0.9])}
    # lower => every PC value < min(comp); and conversely (finite)
    eq_bad = 0
    for pc in cases:
        lower = my_one_test(pc, comp) == 1
        if lower != all(math.isfinite(v) and v < min(comp) for v in pc):
            eq_bad += 1
    print(f"ONE TEST: {len(cases)} cases; disagreements {bad}; k values seen {dict(sorted(seen.items()))}; "
          f"LOWER <=> all PC < 0.154339 violated {eq_bad} times")
    print(f"ONE TEST: same rank placement, different p (sum statistic): {same_place}")
    return bad + eq_bad


# ============================================================================== 3. TD from raw ledgers
def sha(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def fman_ok(run, rel, path):
    with open(FETCH / "manifest.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["fetch"] == "F2" and r["run_id"] == run and r["path"] == rel
                and r["status"] == "COPY"]
    return any(r["sha256"] == sha(path) for r in rows)


def my_td(rows):
    by = {}
    for r in rows:
        t = fnum(r.get("timesteps"))
        if t is None:
            continue
        by.setdefault(int(t), []).append(fnum(r.get("tail_3_pitch_act_command_mean")))
    cmd = {}
    for s, v in by.items():
        v = [x for x in v if x is not None]
        if v:
            cmd[s] = Fraction(sum(Fraction(x) for x in v), len(v))
    td = sum(abs(x) for x in cmd.values()) / len(cmd)
    jj = [abs(x) for s, x in cmd.items() if s >= 4194304]
    return float(td), float(sum(jj) / len(jj)), len(cmd), len(jj)


def check_td():
    want = {60: (0.388227, 0.577658), 42: (0.635343, 0.808751), 80: (0.154339, 0.217914)}
    bad = 0
    out = {}
    for seed, run in RUNS.items():
        rel = "01_stance/eval_episodes.csv"
        p = FETCH / "F2" / run / "copies" / (rel + ".copy1")
        ok = fman_ok(run, rel, p)
        rows = list(csv.DictReader(io.StringIO(p.read_text(encoding="utf-8"))))
        td, tdj, n, nj = my_td(rows)
        theirs = P.tail_drive(P.per_evaluation(rows))
        out[seed] = (round(td, 6), round(tdj, 6), n, nj)
        if not ok or round(td, 6) != want[seed][0] or round(tdj, 6) != want[seed][1] or abs(theirs["td"] - td) > 1e-12:
            bad += 1
        print(f"TD s{seed}: manifest-checked {ok}; mine {td:.9f} (J {tdj:.9f}; n {n}, n_J {nj}); pc_common "
              f"{theirs['td']:.9f} (J {theirs['td_j']:.9f}); comparators.out {want[seed]}")
    return bad, out


# ============================================================================== 4. HABIT ELSEWHERE (my own)
def my_over_row(r, header):
    """(any-over?, actuator index or None) by M5 (a) as ROUND l.143-150 / THIS12 351-358 read it (independent code)."""
    cols = set(header)
    clean = fnum(r.get("screen_v2_clean")) if "screen_v2_clean" in cols else None
    reasons = r.get("screen_v2_reasons") or ""
    listed = "max_actuator_saturation_fraction" in {x.split(":", 1)[0].strip() for x in reasons.split(";") if x.strip()}
    frac = fnum(r.get("saturated_actuator_fraction"))
    a = (r.get("saturated_actuator") or "").strip()
    act = NAMES.index(a) if a in NAMES else None
    if a.startswith("actuator[") and a.endswith("]"):
        try:
            act = int(a[9:-1])
        except ValueError:
            act = None
    if clean is not None and not listed:
        return False, act                                  # under the bar (or a contradiction: missing, not over)
    if clean is not None:
        over = True
    elif frac is not None:
        over = frac > 0.10
    else:
        over = False
    if frac is not None and frac > 0.10 and act in TAIL:
        over = True
    return over, act


def my_cmd(rec, a):
    eps = [e for e in rec.get("episodes", []) if (e.get("length") or 0) > 200]
    v = [e["actuators"][NAMES[a]]["cmd_window_mean"] for e in eps if e["actuators"][NAMES[a]].get("cmd_window_mean")
         is not None]
    return (sum(v) / len(v)) if v else None


def my_habit(header, rows, any_alarms, m1, use_episode_cmd=False):
    over = {}
    for r in rows:
        t = fnum(r.get("timesteps"))
        if t is None or int(t) % GRID or not (1 <= int(t) // GRID <= NEVAL):
            continue
        o, act = my_over_row(r, header)
        if o and act is not None:
            over.setdefault(act, {}).setdefault(int(t), 0)
            over[act][int(t)] += 1
    res = {}
    for a in range(15):
        if a == T3:
            continue
        n = NAMES[a]
        flagged = {s for s, k in over.get(a, {}).items() if k >= 5}
        L = any((s - GRID) in flagged for s in flagged)
        A = any(n in (x.get("actuators") or {}) for x in any_alarms)
        T = False
        cm = {}
        for lab, rec in m1.items():
            plus = minus = tie = 0
            for e in rec.get("episodes", []):
                ae = e["actuators"].get(n) or {}
                if e.get("full_horizon") and ae.get("over_bar"):
                    sg = ae.get("sign")
                    plus += sg == "+"
                    minus += sg == "-"
                    tie += sg == "="
            if plus + tie >= 2 or minus + tie >= 2:
                T = True
            cm[lab] = my_cmd(rec, a) if use_episode_cmd else (rec.get("cmd_all_actuators") or {}).get(str(a))
        D = False
        for sg in (1, -1):
            hits = sum(1 for lab in [str(j) for j in J] + ["final"] if cm.get(lab) is not None and sg * cm[lab] >= 0.60)
            x, y = cm.get(str(R1)), cm.get("final")
            mean = (x + y) / 2 if (x is not None and y is not None) else (x if x is not None else y)
            if hits >= 2 or (mean is not None and sg * mean >= 0.60):
                D = True
        sites = [k for k, v in (("L", L), ("A", A), ("T", T), ("D", D)) if v]
        st = "SATURATING" if (L or A or T) else ("DRIVE ONLY" if D else "FREE")
        res[n] = (sites, st)
    return res


def check_habit_real():
    score = json.loads(SCORE.read_text(encoding="utf-8"))
    cls = {d["seed"]: d["classes"] for d in score["per_seed"]}
    bad = 0
    for seed, run in RUNS.items():
        rel = "01_stance/eval_episodes.csv"
        p = FETCH / "F2" / run / "copies" / (rel + ".copy1")
        raw = p.read_bytes()
        header, rows = P.ledger_rows(raw)
        m1 = {}
        for lab in [str(j) for j in J] + ["final"]:
            f = PANELS / run / lab / "m1" / "m1_summary.json"
            if f.is_file():
                m1[lab] = json.loads(f.read_text(encoding="utf-8"))
        mine = my_habit(header, rows, cls[seed].get("any_alarms") or [], m1)
        mine_ep = my_habit(header, rows, cls[seed].get("any_alarms") or [], m1, use_episode_cmd=True)
        theirs = P.habit_elsewhere(C, header, rows, cls[seed], m1, list(J), R1)
        tsites = {n: (v["sites"], v["status"]) for n, v in theirs["per_actuator"].items()}
        diff = {n: (mine[n], tsites[n]) for n in mine if mine[n] != tsites[n]}
        diff_ep = {n: (mine_ep[n], mine[n]) for n in mine if mine_ep[n] != mine[n]}
        nz = {n: v for n, v in mine.items() if v[0]}
        print(f"HABIT s{seed} (real): mine {nz}; disagreements with pc_common {diff}; "
              f"CMD from episodes vs cmd_all_actuators differ {diff_ep}; labels with M1 {sorted(m1)}")
        bad += len(diff)
    return bad


def synth_ledger(rng):
    header = ["timesteps", "episode", "screen_v2_clean", "screen_v2_reasons", "saturated_actuator",
              "saturated_actuator_fraction", "saturated_actuator_positive_share", "saturated_actuator_ties",
              "actuators_saturated", "length", "tail_3_pitch_act_command_mean"]
    rows = []
    p_over = rng.choice([0.0, 0.02, 0.1, 0.45])
    hot = rng.sample(range(14), rng.randint(0, 3))
    hot = [a if a < 14 else 13 for a in hot]
    for k in range(1, NEVAL + 1):
        s = k * GRID
        for e in range(10):
            mode = rng.random()
            screen = rng.random() < 0.6
            a = rng.choice(hot + [rng.randrange(15)]) if hot else rng.randrange(15)
            if mode >= p_over:
                frac = rng.choice([0.0, 0.05, 0.10])
                rows.append(dict(timesteps=str(s), episode=str(e), screen_v2_clean="1" if screen else "",
                                 screen_v2_reasons="", saturated_actuator="", saturated_actuator_fraction=str(frac),
                                 saturated_actuator_positive_share="nan", saturated_actuator_ties="0",
                                 actuators_saturated="0", length="1000", tail_3_pitch_act_command_mean="0.1"))
            else:
                frac = rng.choice([0.11, 0.5, 0.9])
                reasons = "max_actuator_saturation_fraction: 0.9" if screen else ""
                rows.append(dict(timesteps=str(s), episode=str(e), screen_v2_clean="0" if screen else "",
                                 screen_v2_reasons=reasons, saturated_actuator=NAMES[a],
                                 saturated_actuator_fraction=str(frac), saturated_actuator_positive_share="0.9",
                                 saturated_actuator_ties=str(rng.choice([0, 0, 1])),
                                 actuators_saturated=str(rng.choice([1, 1, 2])), length="1000",
                                 tail_3_pitch_act_command_mean="0.2"))
    return header, rows


def synth_m1(rng):
    m1 = {}
    p_ob = rng.choice([0.0, 0.002, 0.01, 0.05])
    for lab in [str(j) for j in J] + ["final"]:
        if rng.random() < 0.1:
            continue
        eps = []
        for i in range(20):
            length = 1000 if rng.random() < 0.9 else rng.choice([150, 600])
            acts = {}
            for a, n in enumerate(NAMES):
                ob = rng.random() < p_ob
                acts[n] = dict(over_bar=ob, sign=rng.choice(["+", "-", "="]) if ob else None,
                               cmd_window_mean=rng.uniform(-1, 1))
            eps.append(dict(length=length, full_horizon=length >= 1000, actuators=acts))
        cmd = {str(a): rng.choice([rng.uniform(-0.5, 0.5)] * 6 + [rng.uniform(-1, 1), 0.6, -0.6, 0.5999999])
               for a in range(15)}
        m1[lab] = dict(cmd_all_actuators=cmd, episodes=eps)
    return m1


def check_habit_synth(n=None):
    n = n or (150 if QUICK else 600)
    rng = random.Random(3042)
    bad = 0
    seen = {}
    for _ in range(n):
        header, rows = synth_ledger(rng)
        m1 = synth_m1(rng)
        res = C.m5(header, rows)
        # any_alarms as classes_of builds them (actuators named in the two evaluations' any-over rows)
        any_alarms = []
        for s in res["any_alarms"]:
            names = {}
            for e in (res["evals"][s - GRID], res["evals"][s]):
                for k, v in e["any_by"].items():
                    names[k] = names.get(k, 0) + v
            any_alarms.append(dict(step=s, actuators=names))
        mine = my_habit(header, rows, any_alarms, m1)
        theirs = P.habit_elsewhere(C, header, rows, {"any_alarms": any_alarms}, m1, list(J), R1)
        for nme, v in mine.items():
            t = (theirs["per_actuator"][nme]["sites"], theirs["per_actuator"][nme]["status"])
            seen[v[1]] = seen.get(v[1], 0) + 1
            if v != t:
                bad += 1
                if bad < 5:
                    print("   HABIT DISAGREE", nme, v, t)
    print(f"HABIT (synthetic): {n} ledgers x 14 actuators; disagreements {bad}; statuses {seen}")
    return bad


def main():
    tot = 0
    tot += check_reading()
    tot += check_one_test()
    b, _ = check_td()
    tot += b
    tot += check_habit_real()
    tot += check_habit_synth()
    print(f"TOTAL DISAGREEMENTS {tot}")
    return 0 if tot == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
