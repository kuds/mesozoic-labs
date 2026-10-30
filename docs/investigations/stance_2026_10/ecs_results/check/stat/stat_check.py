"""Independent re-derivation of the ECS statistic (check lens: THE STATISTIC).

Written by the stat skeptic from the criteria text (<ECS>/criteria.txt THE ONE PRE-REGISTERED STATISTIC, R1) and the
round's ANY/TAIL definitions (CRIT6 CLASSES, M5; THIS12 lines 297-364).  It imports none of read_ecs, score_ecs,
ecs_common or M5 (ledger_r10round / r10round_common); only the standard library and numpy (for evaluations.npz).
Reads only; writes only into its own folder (stat_out.json, stat_out.txt).

    python -I -B stat_check.py
"""
import csv
import hashlib
import io
import json
import math
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
ECS = Path(S2) / "r10round/ecs"
F2 = Path(S1) / "foot/stance6/fetch/F2"

GRID = 131072
N_EVALS = 91                       # grid 131,072 ... 11,927,552
BAR = 0.10
WINDOW_START = 200
SAT_KEY = "max_actuator_saturation_fraction"
TAIL_IDX = (11, 12, 13, 14)
# the actuator order (index -> name); "actuator[<i>]" also names actuator i (THIS12 line 364)
ACT = ("neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
       "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act", "tail_1_pitch_act",
       "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")

STARTS = {
    "s42": dict(S=3_145_728, E=4_718_592, run_id="stance6_r10LTSB_s42",
                parent_sha="d1bcb78964691cf194e6276f06740d1f24aa69ad36831a046ad6307c1c4db11e",
                seeds={"A": range(1101, 1152, 10), "B": range(1201, 1252, 10)}),
    "s80": dict(S=4_194_304, E=5_767_168, run_id="stance6_r10LTSB_s80",
                parent_sha="b677ce762e739513c18eaa35ffb7b149c0532753746259d5037356580e934d13",
                seeds={"A": range(1301, 1352, 10), "B": range(1401, 1452, 10)}),
}
S60 = dict(run_id="stance6_r10LTSB_s60", trunc=8_388_608,
           sha="1821fd2e9f65d58cb8a0dc83d096230fb04af3448f850657055de001ef8e3466")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fnum(cell):
    """finite float or None (NaN, blank, unparseable -> None)."""
    if cell is None:
        return None
    s = str(cell).strip()
    if s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def act_of(cell):
    if cell is None:
        return None
    s = str(cell).strip()
    if s in ACT:
        return ACT.index(s)
    if s.startswith("actuator[") and s.endswith("]") and s[9:-1].isdigit():
        i = int(s[9:-1])
        return i if i < len(ACT) else None
    return None


def keys_of(reasons):
    return {x.split(":", 1)[0].strip() for x in (reasons or "").split(";") if x.strip()}


def read_csv(path):
    text = Path(path).read_bytes().decode("utf-8")
    rd = csv.DictReader(io.StringIO(text, newline=""))
    return list(rd.fieldnames), list(rd)


def row_inputs(r):
    """(any_status, any_over, tail_status, tail_over, tail_act, sign, unres, why) per THIS12 351-364."""
    clean = fnum(r.get("screen_v2_clean"))
    screen = clean is not None
    listed = SAT_KEY in keys_of(r.get("screen_v2_reasons"))
    frac = fnum(r.get("saturated_actuator_fraction"))
    length = fnum(r.get("length"))
    a = act_of(r.get("saturated_actuator"))
    ties = fnum(r.get("saturated_actuator_ties"))
    nsat = fnum(r.get("actuators_saturated"))
    pos = fnum(r.get("saturated_actuator_positive_share"))
    # contradiction: measured screen, key not listed, fraction over the bar -> ANY and TAIL missing
    if screen and not listed:
        if frac is not None and frac > BAR:
            return dict(any="missing", any_over=False, tail="missing", tail_over=False, act=None, sign=None,
                        unres=False, why="contradiction")
        return dict(any="measured", any_over=False, tail="measured", tail_over=False, act=None, sign=None,
                    unres=False, why=None)
    # ANY input
    if screen:                         # listed
        any_over = True
    elif frac is not None:             # unmeasured screen: the fraction
        any_over = frac > BAR
    else:
        any_over = None
    # TAIL input
    tail_status, tail_over, tact, sign, unres, why = "measured", False, None, None, False, None
    if frac is None:
        if length is None or length > WINDOW_START:
            tail_status, why = "missing", "fraction missing, episode outlived the window start"
    elif frac > BAR:
        if a is None:
            tail_status, why = "missing", "over the bar, no actuator named"
        elif a in TAIL_IDX:
            tail_over, tact = True, a
            sign = "?" if pos is None else ("+" if pos >= 0.5 else "-")
        elif ties is None or nsat is None:
            tail_status, why = "missing", "over the bar on a non-tail actuator, ties/actuators_saturated missing"
        else:
            unres = (ties >= 1) or (nsat >= 2)
    if tail_over:
        any_over = True                # a row over the bar on a tail actuator is any-over
    if any_over is None:
        return dict(any="missing", any_over=False, tail=tail_status, tail_over=False, act=None, sign=None,
                    unres=False, why=why or "screen unmeasured and fraction missing")
    if tail_status == "missing":
        tail_over, tact, sign, unres = False, None, None, False
    return dict(any="measured", any_over=bool(any_over), tail=tail_status, tail_over=tail_over, act=a if any_over
                else None, sign=sign, unres=unres, why=why, frac=frac, listed=listed)


def on_grid(cell):
    v = fnum(cell)
    if v is None or v != int(v):
        return None
    v = int(v)
    if v % GRID or not (1 <= v // GRID <= N_EVALS):
        return None
    return v


def group(rows):
    by, off = {}, []
    for i, r in enumerate(rows):
        s = on_grid(r.get("timesteps"))
        if s is None:
            off.append((i + 2, r.get("timesteps")))
        else:
            by.setdefault(s, []).append(r)
    return by, off


def evaluate(rows_at_step):
    n_any = n_tail = n_unres = 0
    any_missing, tail_missing, any_by, tail_by = [], [], {}, {}
    for r in rows_at_step:
        x = row_inputs(r)
        ep = r.get("episode")
        if x["any"] == "missing":
            any_missing.append((ep, x["why"]))
        if x["tail"] == "missing":
            tail_missing.append((ep, x["why"]))
        if x["any_over"]:
            n_any += 1
            nm = ACT[x["act"]] if x.get("act") is not None else "(unnamed)"
            any_by[nm] = any_by.get(nm, 0) + 1
        if x["tail_over"]:
            n_tail += 1
            k = ACT[x["act"]] + x["sign"]
            tail_by[k] = tail_by.get(k, 0) + 1
        if x["unres"]:
            n_unres += 1
    eps = [fnum(r.get("episode")) for r in rows_at_step]
    eps_ok = len(rows_at_step) == 10 and sorted(e for e in eps if e is not None) == [float(i) for i in range(10)] \
        and None not in eps
    return dict(n=len(rows_at_step), eps_ok=eps_ok, any=n_any, tail=n_tail, unres=n_unres,
                any_flag=n_any >= 5, tail_flag=n_tail >= 5, any_missing=any_missing, tail_missing=tail_missing,
                any_by=any_by, tail_by=tail_by)


def alarms(steps_flag, lo, hi):
    """alarms at s in (lo, hi] whose first evaluation s - GRID is also in (lo, hi]; straddling ones (first = lo)."""
    inside = [s for s in sorted(steps_flag) if lo < s <= hi and (s - GRID) in steps_flag and (s - GRID) > lo]
    straddle = [s for s in sorted(steps_flag) if s == lo + GRID and lo in steps_flag]
    return inside, straddle


def read_ledger_window(path, S, T):
    header, rows = read_csv(path)
    by, off = group(rows)
    ev = {s: evaluate(rs) for s, rs in by.items()}
    want = [S + GRID * k for k in range(1, (T - S) // GRID + 1)]
    problems = []
    for i, ts in off:
        problems.append(f"row line {i}: timesteps {ts!r} off the grid")
    for s in sorted(ev):
        if s > T:
            problems.append(f"evaluation {s} beyond T")
    for s in want:
        e = ev.get(s)
        if e is None:
            problems.append(f"window evaluation {s} absent")
            continue
        if not e["eps_ok"]:
            problems.append(f"window evaluation {s}: n {e['n']}, episodes not 0-9 once")
        for ep, why in e["any_missing"]:
            problems.append(f"window evaluation {s} ep {ep}: ANY input missing ({why})")
    extra_window = [s for s in ev if S < s <= T and s not in want]
    if extra_window:
        problems.append(f"unexpected window evaluations {extra_window}")
    tail_missing_window = [(s, ep, why) for s in want if s in ev for ep, why in ev[s]["tail_missing"]]
    # parent / seeded part (t <= S): grid S/GRID evaluations, 10 rows, inputs measured, flags
    pgrid = [GRID * k for k in range(1, S // GRID + 1)]
    pproblems = []
    for s in pgrid:
        e = ev.get(s)
        if e is None:
            pproblems.append(f"parent evaluation {s} absent")
            continue
        if not e["eps_ok"]:
            pproblems.append(f"parent evaluation {s}: n {e['n']}, episodes bad")
        for ep, why in e["any_missing"]:
            pproblems.append(f"parent evaluation {s} ep {ep}: ANY missing ({why})")
        for ep, why in e["tail_missing"]:
            pproblems.append(f"parent evaluation {s} ep {ep}: TAIL missing ({why})")
    any_f = {s for s, e in ev.items() if e["any_flag"]}
    tail_f = {s for s, e in ev.items() if e["tail_flag"]}
    win_any = sorted(s for s in any_f if S < s <= T)
    win_tail = sorted(s for s in tail_f if S < s <= T)
    any_al, any_str = alarms(any_f, S, T)
    tail_al, tail_str = alarms(tail_f, S, T)
    parent_flagged = sorted(s for s in any_f if s <= S)
    parent_tail_flagged = sorted(s for s in tail_f if s <= S)
    parent_over_rows = sorted((s, ev[s]["any"]) for s in ev if s <= S and ev[s]["any"])
    ret_T = None
    if T in by:
        vals = [fnum(r.get("return")) for r in by[T]]
        ret_T = sum(vals) / len(vals) if None not in vals else None
    return dict(header=header, rows=rows, by=by, ev=ev, n_rows=len(rows), n_cols=len(header),
                window_steps=want, window_problems=problems, tail_missing_window=tail_missing_window,
                parent_problems=pproblems, parent_flagged=parent_flagged, parent_tail_flagged=parent_tail_flagged,
                parent_over_rows=parent_over_rows, parent_at_S=ev.get(S),
                any_flagged=win_any, first_any=win_any[0] if win_any else None, n_any_flagged=len(win_any),
                any_alarms=any_al, any_straddle=any_str, tail_flagged=win_tail, tail_alarms=tail_al,
                tail_straddle=tail_str, mean_return_T=ret_T)


# ------------------------------------------------------------------------------------------------ the statistic
def hyper(n_a, n_b, m):
    """P(a = k) given arm sizes and the start's total positives m (Fractions)."""
    tot = comb(n_a + n_b, m)
    return {k: Fraction(comb(n_a, k) * comb(n_b, m - k), tot) for k in range(max(0, m - n_b), min(n_a, m) + 1)}


def cmh(tables):
    """tables: list of (a, nA, b, nB).  Exact stratified test conditional on each start's total.  Returns (P_UP,
    P_DOWN) as Fractions: P(X >= x), P(X <= x), X = sum a."""
    dist = {0: Fraction(1)}
    x = 0
    for a, na, b, nb in tables:
        h = hyper(na, nb, a + b)
        nd = {}
        for s, p in dist.items():
            for k, q in h.items():
                nd[s + k] = nd.get(s + k, 0) + p * q
        dist = nd
        x += a
    up = sum((p for s, p in dist.items() if s >= x), Fraction(0))
    down = sum((p for s, p in dist.items() if s <= x), Fraction(0))
    return up, down


def fisher(a, na, b, nb):
    h = hyper(na, nb, a + b)
    return (sum((p for k, p in h.items() if k >= a), Fraction(0)),      # A greater
            sum((p for k, p in h.items() if k <= a), Fraction(0)))      # B greater


def frac_str(f):
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


ALPHA = Fraction(1, 40)
LINE1 = ("REMOVING THE REMAINING ENTROPY BONUS LOWERED THE SHARE OF CONTINUATIONS WITH AN ANY-FLAGGED EVALUATION BY E "
         "(pooled over both states; per-start tables beside)")
LINE2 = ("REMOVING THE REMAINING ENTROPY BONUS RAISED THE SHARE OF CONTINUATIONS WITH AN ANY-FLAGGED EVALUATION BY E "
         "(pooled over both states; per-start tables beside)")
LINE3 = ("NO DIFFERENCE DETECTED IN WHETHER AN ANY-FLAGGED EVALUATION CAME BY E FROM THESE STATES (not tested: a shift "
         "of the first flag within the window, a modest effect, the early phase)")
LINE3_NI = ("NO DIFFERENCE DETECTED; NOT INFORMATIVE: arm A showed the outcome in fewer than half its counted "
            "continuations, or none was counted, at both starts")


def statistic(outcomes):
    """outcomes: {start: {arm: [cls/outcome tuples]}} where each item is 'POSITIVE', 'NEGATIVE', 'MISSING'."""
    per = {}
    for st in ("s42", "s80"):
        A, B = outcomes[st]["A"], outcomes[st]["B"]
        aP, aN, aM = A.count("POSITIVE"), A.count("NEGATIVE"), A.count("MISSING")
        bP, bN, bM = B.count("POSITIVE"), B.count("NEGATIVE"), B.count("MISSING")
        counted = (aP, aP + aN, bP, bP + bN)
        up_t = (aP, aP + aN + aM, bP + bM, bP + bN + bM)        # against B LOWER: A missing NEG, B missing POS
        down_t = (aP + aM, aP + aN + aM, bP, bP + bN + bM)      # against B HIGHER: A missing POS, B missing NEG
        nA = aP + aN
        notes = []
        if nA == 0:
            notes.append("NO ARM-A CONTINUATION COUNTED")
        elif 2 * aP < nA:
            notes.append(f"LOW ARM-A RATE ({aP} of {nA}): little power for B LOWER at this start")
        if nA > 0 and aP == nA:
            notes.append(f"ARM A AT {aP} OF {aP}: no room for B HIGHER at this start")
        per[st] = dict(counted=counted, up=up_t, down=down_t, missing=(aM, bM), notes=notes,
                       fisher_counted=fisher(*counted), fisher_up=fisher(*up_t), fisher_down=fisher(*down_t))
    p_up = cmh([per[s]["up"] for s in per])[0]
    p_down = cmh([per[s]["down"] for s in per])[1]
    c_up, c_down = cmh([per[s]["counted"] for s in per])
    n_missing = sum(sum(per[s]["missing"]) for s in per)
    if p_up <= ALPHA:
        line, text = 1, LINE1
    elif p_down <= ALPHA:
        line, text = 2, LINE2
    else:
        line = 3
        both_low = all(any(n.startswith("LOW ARM-A RATE") or n.startswith("NO ARM-A") for n in per[s]["notes"])
                       for s in per)
        text = LINE3_NI if both_low else LINE3
    additions = []
    if line != 1 and c_up <= ALPHA:
        additions.append(f"(B LOWER holds only without the {n_missing} LOST or censored continuations)")
    if line != 2 and c_down <= ALPHA:
        additions.append(f"(B HIGHER holds only without the {n_missing} LOST or censored continuations)")
    return dict(per=per, p_up=p_up, p_down=p_down, counted_p_up=c_up, counted_p_down=c_down, line=line, text=text,
                additions=additions, n_missing=n_missing)


# ------------------------------------------------------------------------------------------------ TABLE E (report)
def binom(n, k, p):
    return comb(n, k) * p ** k * (1 - p) ** (n - k)


def power_tables():
    k = 6
    rej_up, rej_down = [], []
    for a1 in range(k + 1):
        for b1 in range(k + 1):
            for a2 in range(k + 1):
                for b2 in range(k + 1):
                    u, d = cmh([(a1, k, b1, k), (a2, k, b2, k)])
                    if u <= ALPHA:
                        rej_up.append((a1, b1, a2, b2))
                    elif d <= ALPHA:
                        rej_down.append((a1, b1, a2, b2))

    def prob(rej, pa1, pb1, pa2, pb2):
        return sum(binom(k, a1, pa1) * binom(k, b1, pb1) * binom(k, a2, pa2) * binom(k, b2, pb2)
                   for a1, b1, a2, b2 in rej)
    pw = {}
    for pa in (0.6, 0.75, 0.9):
        for pb in (0.0, 0.1, 0.25, 0.4, 0.5):
            pw[(pa, pb)] = prob(rej_up, pa, pb, pa, pb)
    ph = {pa: prob(rej_down, pa, 1.0, pa, 1.0) for pa in (0.9, 0.75, 0.6)}
    # symmetric table (equal at both starts)
    tab = {}
    for a in range(k + 1):
        for b in range(k + 1):
            u, d = cmh([(a, k, b, k), (a, k, b, k)])
            tab[(a, b)] = "L" if u <= ALPHA else ("H" if d <= ALPHA else ".")
    size = 0.0
    for i in range(5, 96):
        for j in range(5, 96):
            p1, p2 = i / 100, j / 100
            size = max(size, prob(rej_up, p1, p1, p2, p2), prob(rej_down, p1, p1, p2, p2))
    return pw, ph, tab, size


# ------------------------------------------------------------------------------------------------ main
def main():
    out = {"runs": {}, "disagreements": []}
    L = []
    dis = out["disagreements"]
    score = json.loads((ECS / "score/score.json").read_text())
    sc_by = {c["name"]: c for c in score["classes"]}
    L.append(f"criteria sha256 {sha(ECS / 'criteria.txt')}")
    L.append(f"score.json sha256 {sha(ECS / 'score/score.json')}  score.txt sha256 {sha(ECS / 'score/score.txt')}")
    outcomes = {st: {"A": [], "B": []} for st in STARTS}
    rows_tbl = []
    for st, cfg in STARTS.items():
        S, E = cfg["S"], cfg["E"]
        ph, prow = read_csv(F2 / cfg["run_id"] / "copies/01_stance/eval_episodes.csv.copy1")
        p_by, _ = group(prow)
        for arm in ("A", "B"):
            for seed in cfg["seeds"][arm]:
                name = f"{st}_{arm}_{seed}"
                rd = ECS / "runs" / name
                meta = json.loads((rd / "run_meta.json").read_text())
                chk = json.loads((rd / "check.json").read_text())
                T = meta.get("T")
                cls = None
                if chk.get("passed") is True and meta.get("stop_reason") == "end" and T == E and chk.get("T") == E \
                        and chk.get("stop_reason") == "end" and meta.get("num_timesteps_end") == E:
                    cls = "COUNTED"
                elif chk.get("passed") is True and meta.get("stop_reason") == "collapse" and T is not None and T < E:
                    cls = "STOPPED"
                else:
                    cls = "UNCLEAR"
                led = rd / "01_stance/eval_episodes.csv"
                w = read_ledger_window(led, S, T if T else E)
                # seeded parent rows equal the F2 parent's rows at or before S, cell for cell
                seeded_eq = None
                if w["header"] == ph:
                    mine = [r for r in w["rows"] if (fnum(r.get("timesteps")) or 0) <= S]
                    par = [r for r in prow if (fnum(r.get("timesteps")) or 0) <= S]
                    seeded_eq = (len(mine) == len(par)) and all(m == p for m, p in zip(mine, par))
                else:
                    seeded_eq = f"header differs ({len(w['header'])} vs {len(ph)})"
                # the return check (info): ledger return vs evaluations.npz results at the same step/episode
                ret_check = None
                try:
                    import numpy as np
                    with np.load(rd / "01_stance/evaluations.npz", allow_pickle=False) as z:
                        tsz = z["timesteps"].tolist()
                        res = z["results"]
                    bad = 0
                    checked = 0
                    for s in w["window_steps"]:
                        if s not in tsz:
                            bad += 10
                            continue
                        i = tsz.index(s)
                        for r in w["by"].get(s, []):
                            ep = int(fnum(r["episode"]))
                            checked += 1
                            if float(res[i][ep]) != fnum(r["return"]):
                                bad += 1
                    ret_check = dict(checked=checked, mismatched=bad)
                except Exception as exc:  # noqa: BLE001
                    ret_check = dict(error=repr(exc))
                if w["window_problems"]:
                    outcome = "LOST(C6)"
                elif cls == "COUNTED":
                    outcome = "POSITIVE" if w["any_flagged"] else "NEGATIVE"
                elif cls == "STOPPED":
                    outcome = "POSITIVE" if w["any_flagged"] else "CENSORED"
                else:
                    outcome = "?"
                outcomes[st][arm].append(outcome if outcome in ("POSITIVE", "NEGATIVE") else "MISSING")
                # compare with read.json and score.json
                r1 = json.loads((ECS / "score/read" / name / "read.json").read_text())["r1"]
                m5 = json.loads((ECS / "score/read" / name / "m5.json").read_text())
                sc = sc_by[name]
                cmp = {
                    "score.cls": (sc["cls"], cls), "score.outcome": (sc["outcome"], outcome), "score.T": (sc["T"], T),
                    "r1.positive": (r1["positive"], bool(w["any_flagged"])),
                    "r1.first_any_flag": (r1["first_any_flag"], w["first_any"]),
                    "r1.n_any_flagged": (r1["n_any_flagged"], w["n_any_flagged"]),
                    "r1.any_flagged": (r1["any_flagged"], w["any_flagged"]),
                    "r1.any_alarms": (r1["any_alarms"], w["any_alarms"]),
                    "r1.any_alarms_straddling_S": (r1["any_alarms_straddling_S"], w["any_straddle"]),
                    "r1.tail_flagged": (r1["tail_flagged"], w["tail_flagged"]),
                    "r1.tail_alarms": (r1["tail_alarms"], w["tail_alarms"]),
                    "r1.tail_alarms_straddling_S": (r1["tail_alarms_straddling_S"], w["tail_straddle"]),
                    "r1.flagged_at_or_before_S": (r1["flagged_at_or_before_S"], w["parent_flagged"]),
                    "r1.window_evaluations": (r1["window_evaluations"], w["window_steps"]),
                    "r1.window_evaluations_absent": (r1["window_evaluations_absent"],
                                                     [s for s in w["window_steps"] if s not in w["ev"]]),
                    "r1.c6_ok": (r1["c6_ok"], not w["window_problems"]),
                    "r1.c6_tail_missing": (len(r1["c6_tail_missing"]), len(w["tail_missing_window"])),
                    "m5.c6.passed": (m5["c6"]["passed"], not w["window_problems"] and not w["parent_problems"]),
                }
                for s in w["window_steps"]:
                    pe = r1["per_evaluation"].get(str(s))
                    me = w["ev"].get(s)
                    if pe is None or me is None:
                        cmp[f"r1.per_eval.{s}"] = (pe, me and me["n"])
                        continue
                    cmp[f"r1.per_eval.{s}"] = ((pe["n"], pe["any"], pe["tail"], pe["unres"], pe["any_by"],
                                                pe["tail_by"]),
                                               (me["n"], me["any"], me["tail"], me["unres"], me["any_by"],
                                                me["tail_by"]))
                # M5's whole-ledger per-evaluation counts against mine (every evaluation)
                for s, me in w["ev"].items():
                    pe = m5["per_evaluation"].get(str(s))
                    if pe is None or (pe["n"], pe["any"], pe["tail"], pe["unres"]) != \
                            (me["n"], me["any"], me["tail"], me["unres"]):
                        cmp[f"m5.per_eval.{s}"] = (pe and (pe["n"], pe["any"], pe["tail"], pe["unres"]),
                                                   (me["n"], me["any"], me["tail"], me["unres"]))
                diffs = {k: v for k, v in cmp.items() if json.dumps(v[0], sort_keys=True, default=str)
                         != json.dumps(v[1], sort_keys=True, default=str)}
                for k, (theirs, mine) in diffs.items():
                    dis.append(f"{name}: {k}: theirs {theirs!r} mine {mine!r}")
                pS = w["parent_at_S"]
                rec = dict(name=name, cls=cls, outcome=outcome, T=T, n_rows=w["n_rows"], n_cols=w["n_cols"],
                           window_problems=w["window_problems"], parent_problems=w["parent_problems"],
                           tail_missing_window=w["tail_missing_window"], seeded_rows_equal_parent=seeded_eq,
                           parent_flagged=w["parent_flagged"], parent_tail_flagged=w["parent_tail_flagged"],
                           parent_over_rows=w["parent_over_rows"],
                           parent_at_S=dict(n=pS["n"], any=pS["any"], tail=pS["tail"]) if pS else None,
                           first_any=w["first_any"], n_any_flagged=w["n_any_flagged"], any_flagged=w["any_flagged"],
                           any_alarms=w["any_alarms"], any_straddle=w["any_straddle"], tail_flagged=w["tail_flagged"],
                           tail_alarms=w["tail_alarms"], tail_straddle=w["tail_straddle"],
                           per_eval={s: (w["ev"][s]["any"], w["ev"][s]["tail"], w["ev"][s]["any_by"])
                                     for s in w["window_steps"] if s in w["ev"]},
                           mean_return_T=w["mean_return_T"], return_check=ret_check,
                           compared_fields=len(cmp), differing_fields=sorted(diffs),
                           ledger_sha256=sha(led), read_json_ledger_sha256=r1.get("ledger_sha256"),
                           m5_json_ledger_sha256=m5.get("sha256"))
                if rec["ledger_sha256"] != rec["m5_json_ledger_sha256"]:
                    dis.append(f"{name}: ledger sha256 now {rec['ledger_sha256']} vs m5.json {m5.get('sha256')}")
                if rec["ledger_sha256"] != rec["read_json_ledger_sha256"]:
                    dis.append(f"{name}: ledger sha256 now {rec['ledger_sha256']} vs read.json "
                               f"{rec['read_json_ledger_sha256']}")
                out["runs"][name] = rec
                rows_tbl.append(rec)
    # ---------------------------------------------------------------- per continuation table
    L.append("")
    L.append("PER CONTINUATION (window (S, T]; mine)")
    L.append("  name        class    outcome   T          rows cols win-probs par-probs seeded=F2 parS(any,tail) "
             "first-ANY  nflag ANY-ALARM(count,first) TAIL-flag(count) TAIL-ALARM(count) straddle ret@T")
    for r in rows_tbl:
        L.append(f"  {r['name']:11} {r['cls']:8} {r['outcome']:9} {r['T']:<10} {r['n_rows']:4} {r['n_cols']:4} "
                 f"{len(r['window_problems']):9} {len(r['parent_problems']):9} {str(r['seeded_rows_equal_parent']):9} "
                 f"({r['parent_at_S']['any']},{r['parent_at_S']['tail']})         "
                 f"{str(r['first_any']):10} {r['n_any_flagged']:5} ({len(r['any_alarms'])},"
                 f"{r['any_alarms'][0] if r['any_alarms'] else None})       {len(r['tail_flagged']):3}"
                 f"              {len(r['tail_alarms']):3}            {r['any_straddle'] + r['tail_straddle']} "
                 f"{r['mean_return_T']:.1f}  retcheck {r['return_check']}")
        L.append(f"      ANY-flagged {r['any_flagged']}  ANY ALARM {r['any_alarms']}")
        L.append(f"      TAIL-flagged {r['tail_flagged']}  TAIL ALARM {r['tail_alarms']}")
        L.append(f"      per window evaluation (any, tail, any_by): "
                 + "; ".join(f"{s}:{v[0]},{v[1]}{'' if not v[2] else ' ' + json.dumps(v[2])}"
                             for s, v in r["per_eval"].items()))
        if r["parent_over_rows"]:
            L.append(f"      parent rows over the bar (step, count) at or before S: {r['parent_over_rows']}")
        L.append(f"      fields compared with score.json/read.json/m5.json {r['compared_fields']}, differing "
                 f"{r['differing_fields']}")
    # ---------------------------------------------------------------- the statistic
    stt = statistic(outcomes)
    L.append("")
    L.append("THE STATISTIC (mine; Fractions)")
    for st in ("s42", "s80"):
        p = stt["per"][st]
        a, na, b, nb = p["counted"]
        L.append(f"  {st}: as counted a {a}/{na}, b {b}/{nb}; missing A {p['missing'][0]}, B {p['missing'][1]}; "
                 f"P_UP table {p['up']}; P_DOWN table {p['down']}")
        fc, fu, fd = p["fisher_counted"], p["fisher_up"], p["fisher_down"]
        L.append(f"       Fisher one-sided: as counted A greater {frac_str(fc[0])} ({float(fc[0]):.4f}), B greater "
                 f"{frac_str(fc[1])} ({float(fc[1]):.4f}); A greater on the B-LOWER fill {frac_str(fu[0])}; "
                 f"B greater on the B-HIGHER fill {frac_str(fd[1])}")
        L.append(f"       notes: {p['notes']}")
    L.append(f"  X = {stt['per']['s42']['up'][0] + stt['per']['s80']['up'][0]} (P_UP table), "
             f"{stt['per']['s42']['down'][0] + stt['per']['s80']['down'][0]} (P_DOWN table)")
    L.append(f"  P_UP (filled) = {frac_str(stt['p_up'])} ({float(stt['p_up']):.6f}); P_DOWN (filled) = "
             f"{frac_str(stt['p_down'])} ({float(stt['p_down']):.6f}); as counted {frac_str(stt['counted_p_up'])} / "
             f"{frac_str(stt['counted_p_down'])}; alpha per side 1/40")
    L.append(f"  READING: line {stt['line']}: {stt['text']}")
    for st in ("s42", "s80"):
        a, na, b, nb = stt["per"][st]["counted"]
        L.append(f"    {st}: arm A {a} of {na}, arm B {b} of {nb}")
    L.append(f"  additions: {stt['additions']}")
    # compare with score.json
    sp = score
    cmpS = {
        "status": (sp["status"], "final"), "provisional": (sp["provisional"], []),
        "line": (sp["line"], stt["line"]), "line_text": (sp["line_text"], stt["text"]),
        "additions": (sp["additions"], stt["additions"]), "p_up": (sp["p_up"], frac_str(stt["p_up"])),
        "p_down": (sp["p_down"], frac_str(stt["p_down"])),
        "counted_p_up": (sp["counted_p_up"], frac_str(stt["counted_p_up"])),
        "counted_p_down": (sp["counted_p_down"], frac_str(stt["counted_p_down"])),
        "n_missing": (sp["n_missing"], stt["n_missing"]),
    }
    for st in ("s42", "s80"):
        p = stt["per"][st]
        cmpS[f"tables.{st}.counted"] = (sp["tables"][st]["counted"], list(p["counted"]))
        cmpS[f"tables.{st}.up"] = (sp["tables"][st]["up"], list(p["up"]))
        cmpS[f"tables.{st}.down"] = (sp["tables"][st]["down"], list(p["down"]))
        cmpS[f"tables.{st}.missing"] = (sp["tables"][st]["missing"], {"A": p["missing"][0], "B": p["missing"][1]})
        cmpS[f"notes.{st}"] = (sp["notes"][st], p["notes"])
        cmpS[f"fisher.{st}.counted"] = (sp["fisher"][st]["counted"], [frac_str(x) for x in p["fisher_counted"]])
        cmpS[f"fisher.{st}.up"] = (sp["fisher"][st]["up"], [frac_str(x) for x in p["fisher_up"]])
        cmpS[f"fisher.{st}.down"] = (sp["fisher"][st]["down"], [frac_str(x) for x in p["fisher_down"]])
    for k, (theirs, mine) in cmpS.items():
        if json.dumps(theirs, sort_keys=True) != json.dumps(mine, sort_keys=True):
            dis.append(f"score.json {k}: theirs {theirs!r} mine {mine!r}")
    L.append(f"  score.json fields compared {len(cmpS)}; differing "
             f"{[k for k, (t, m) in cmpS.items() if json.dumps(t, sort_keys=True) != json.dumps(m, sort_keys=True)]}")
    # wording against the criteria text (the quoted strings, whitespace-normalised)
    crit = " ".join((ECS / "criteria.txt").read_text().split())
    for nm, t in (("LINE1", LINE1), ("LINE2", LINE2), ("LINE3", LINE3), ("LINE3_NI", LINE3_NI)):
        L.append(f"  my {nm} string found verbatim in criteria.txt (whitespace-normalised): {t in crit}")
    stxt = (ECS / "score/score.txt").read_text()
    L.append(f"  score.txt carries my line text verbatim: {stt['text'] in stxt}")
    # ---------------------------------------------------------------- (c) parents
    L.append("")
    L.append("(c) PARENT LEDGERS TRUNCATED TO timesteps <= E (F2 copy1), my ANY flag")
    out["parents"] = {}
    for st, cfg in STARTS.items():
        pth = F2 / cfg["run_id"] / "copies/01_stance/eval_episodes.csv.copy1"
        h, rows = read_csv(pth)
        rows = [r for r in rows if (fnum(r.get("timesteps")) or 0) <= cfg["E"]]
        by, off = group(rows)
        ev = {s: evaluate(rs) for s, rs in by.items()}
        fl = sorted(s for s, e in ev.items() if e["any_flag"])
        grid = [GRID * k for k in range(1, cfg["E"] // GRID + 1)]
        c6 = (not off and all(s in ev and ev[s]["eps_ok"] and not ev[s]["any_missing"] and not ev[s]["tail_missing"]
                              for s in grid) and all(s in grid for s in ev))
        win = [s for s in fl if cfg["S"] < s <= cfg["E"]]
        over = sorted((s, ev[s]["any"], ev[s]["any_by"]) for s in ev if ev[s]["any"] and s <= cfg["S"])
        rec = dict(sha256=sha(pth), sha_ok=sha(pth) == cfg["parent_sha"], c6=c6, positive=bool(fl),
                   first_flag=fl[0] if fl else None, n_flagged=len(fl), flagged=fl, flagged_in_window=win,
                   any_alarms=alarms(set(fl), 0, cfg["E"])[0], over_rows_at_or_before_S=over)
        out["parents"][st] = rec
        L.append(f"  {st}: sha ok {rec['sha_ok']}; C6-like {c6}; {'POSITIVE' if fl else 'NEGATIVE'}; first flag "
                 f"{rec['first_flag']}; flagged {len(fl)} {fl}; in (S,E] {len(win)}; over-bar rows <= S {over}")
    pth = F2 / S60["run_id"] / "copies/01_stance/eval_episodes.csv.copy1"
    if pth.exists():
        h, rows = read_csv(pth)
        rows = [r for r in rows if (fnum(r.get("timesteps")) or 0) <= S60["trunc"]]
        by, off = group(rows)
        ev = {s: evaluate(rs) for s, rs in by.items()}
        fl = sorted(s for s, e in ev.items() if e["any_flag"])
        out["parents"]["s60"] = dict(sha_ok=sha(pth) == S60["sha"], positive=bool(fl), flagged=fl)
        L.append(f"  s60 (V4 negative control, <= 8,388,608): sha ok {sha(pth) == S60['sha']}; "
                 f"{'POSITIVE' if fl else 'NEGATIVE'}; flagged {fl}")
    # ---------------------------------------------------------------- TABLE E (report only)
    pw, phh, tab, size = power_tables()
    L.append("")
    L.append("TABLE E (report only; my exact enumeration)")
    L.append("  sym table a\\b: " + " | ".join("".join(tab[(a, b)] for b in range(7)) for a in range(7)))
    for pa in (0.6, 0.75, 0.9):
        L.append(f"  P(B LOWER) A {pa}: " + "  ".join(f"{pw[(pa, pb)]:.3f}" for pb in (0.0, 0.1, 0.25, 0.4, 0.5)))
    L.append("  P(B HIGHER) vs B 1.0: " + "  ".join(f"A {pa} {phh[pa]:.3f}" for pa in (0.9, 0.75, 0.6)))
    L.append(f"  size under the null, max per side on the 0.01 grid 0.05-0.95: {size:.5f}")
    out["table_e"] = dict(power={f"{k[0]},{k[1]}": v for k, v in pw.items()}, b_higher={str(k): v for k, v in
                                                                                         phh.items()},
                          sym="".join(tab[(a, b)] for a in range(7) for b in range(7)), size=size)
    # ---------------------------------------------------------------- end
    L.append("")
    L.append(f"DISAGREEMENTS ({len(dis)}):")
    for d in dis:
        L.append(f"  {d}")
    out["statistic"] = dict(line=stt["line"], text=stt["text"], additions=stt["additions"],
                            p_up=frac_str(stt["p_up"]), p_down=frac_str(stt["p_down"]),
                            counted_p_up=frac_str(stt["counted_p_up"]), counted_p_down=frac_str(stt["counted_p_down"]),
                            per={s: dict(counted=v["counted"], up=v["up"], down=v["down"], notes=v["notes"],
                                         fisher_counted=[frac_str(x) for x in v["fisher_counted"]],
                                         fisher_up=[frac_str(x) for x in v["fisher_up"]],
                                         fisher_down=[frac_str(x) for x in v["fisher_down"]])
                                 for s, v in stt["per"].items()})
    (HERE / "stat_out.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    (HERE / "stat_out.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
