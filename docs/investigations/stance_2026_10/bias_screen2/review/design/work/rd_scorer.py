"""Design reviewer's INDEPENDENT scorer for BS2 synthetic roots, written from <BS2>/criteria.txt (THE PRIMARY OUTCOME, THE
TEST, THE FILL, THE INFORMATIVENESS CONDITION, THE READING, THE SECONDARIES, RUN HANDLING), not from score_bs.py's code.

Differences in method from score_bs.py (on purpose):
  - the fill is implemented with large finite stand-ins (-M / +M, M = 10**6) instead of (infinite part, finite part)
    pairs; with |values| < 100 and 8 entries this gives the same comparisons as the limit;
  - the median is computed by hand (sorted list, midpoint of the two middle values);
  - p counts relabelings with sum >= observed via an explicit loop over index sets built by bit masks.
Usage: imported by rd_compare.py.
"""
import json
import math
from fractions import Fraction
from pathlib import Path

S, E, ROLL = 2_097_152, 3_538_944, 8_192
B_S = Fraction(135930855, 1073741824)            # criteria: exactly 135930855/1073741824
GATE = Fraction(1, 5)
ALPHA = Fraction(1, 20)
SEEDS = (2501, 2511, 2521, 2531)
CTRL, BLIND = "control", "critic_blind_tail3"
QUEUE = [(a, s) for s in SEEDS for a in (CTRL, BLIND)]
PRIMARY_STEPS = [3_473_408 + ROLL * i for i in range(8)]
M = Fraction(10 ** 6)
LINE1 = ("NOT INFORMATIVE: THE CONTROL ARM DID NOT REPRODUCE THE CLIMB OF THE ON-POLICY TAIL_3 COMMAND (its median "
         "primary rose by less than 0.20 from the value at S, or no control primary was read); no reading about the "
         "critic is made")
ADD1 = "(the filled p is <= 0.05, but the control arm did not climb: no reading about the critic)"
ADD3 = "(the as-counted p, report only, is <= 0.05; the filled p decides)"
SUP, NOTSUP = "SUPPORTED (p <= 0.05, unadjusted)", "NOT SUPPORTED"
BONF = "(survives Bonferroni over the 3 secondaries)"
UNDER1 = "(the control arm did not climb on the primary)"
SECS = (("SEC-G", BLIND), ("SEC-CMD", CTRL), ("SEC-MARGIN", BLIND))     # (key, the arm hypothesised HIGHER)


def name(arm, seed):
    return f"{arm}_{seed}"


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def perm_p(higher, other):
    """p = share of the C(n, k) relabelings whose 'higher'-arm sum >= the observed; values Fractions (stand-ins
    included).  Returns (count, total) or None if an arm is empty."""
    if not higher or not other:
        return None
    vals = list(higher) + list(other)
    n, k = len(vals), len(higher)
    obs = sum(higher, Fraction(0))
    count = total = 0
    for mask in range(1 << n):
        if bin(mask).count("1") != k:
            continue
        total += 1
        s = sum((vals[i] for i in range(n) if mask >> i & 1), Fraction(0))
        if s >= obs:
            count += 1
    return count, total


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return None
    if n % 2:
        return v[n // 2]
    return (v[n // 2 - 1] + v[n // 2]) / 2


def classify(root, arm, seed, lost):
    nm = name(arm, seed)
    if nm in lost:
        return "LOST", None
    ck = jload(Path(root) / "runs" / nm / "check.json")
    if ck is None or ck.get("passed") is not True or ck.get("failures"):
        return "UNDECIDED", None
    if ck.get("stop_reason") == "end" and ck.get("T") == E:
        return "COUNTED", E
    if ck.get("stop_reason") == "collapse" and S < ck.get("T") < E:
        return "STOPPED", ck["T"]
    raise ValueError("odd check")


def read(root):
    root = Path(root)
    lj = jload(root / "lost.json")
    lost = (lj or {}).get("lost", {})
    out = []
    for arm, seed in QUEUE:
        cls, T = classify(root, arm, seed, lost)
        out.append({"name": name(arm, seed), "arm": arm, "cls": cls, "T": T})
    return out


def primary(root, c):
    if c["cls"] != "COUNTED":
        return None
    m = jload(Path(root) / "reads" / c["name"] / "markers.json")
    st, mu = m["probe"]["steps"], m["probe"]["rollout_mu14_mean"]
    vals = []
    for s in PRIMARY_STEPS:
        idx = [i for i, x in enumerate(st) if x == s]
        if len(idx) != 1 or not finite(mu[idx[0]]):
            return None
        vals.append(Fraction(mu[idx[0]]))
    return sum(vals, Fraction(0)) / 8


def secondary(root, c, key, prim):
    if c["cls"] != "COUNTED" or prim is None:
        return None
    m = jload(Path(root) / "reads" / c["name"] / "markers.json")
    if key in ("SEC-G", "SEC-MARGIN"):
        rows = [r for r in m.get("loop", []) if r.get("steps") == E]
        if len(rows) != 1:
            return None
        v = rows[0].get("OWN", {}).get("P", {}).get("G" if key == "SEC-G" else "b_star")
    else:
        e = m.get("ledger", {}).get("direct", {}).get("per_evaluation", {}).get(str(E))
        v = e.get("cmd_l") if isinstance(e, dict) else None
    return Fraction(v) if finite(v) else None


def score(root):
    cl = read(root)
    if any(c["cls"] == "UNDECIDED" for c in cl):
        return {"status": "incomplete"}
    prim = {c["name"]: primary(root, c) for c in cl}
    filled = {CTRL: [], BLIND: []}
    read_ = {CTRL: [], BLIND: []}
    for c in cl:
        v = prim[c["name"]]
        if v is None:
            filled[c["arm"]].append(-M if c["arm"] == CTRL else M)
        else:
            filled[c["arm"]].append(v)
            read_[c["arm"]].append(v)
    pf = perm_p(filled[CTRL], filled[BLIND])
    pc = perm_p(read_[CTRL], read_[BLIND])
    med = median(read_[CTRL])
    rise = None if med is None else med - B_S
    informative = rise is not None and rise >= GATE
    pfv = Fraction(*pf)
    if not informative:
        line = 1
    elif pfv <= ALPHA:
        line = 2
    else:
        line = 3
    adds = []
    if line == 1 and pfv <= ALPHA:
        adds.append(ADD1)
    if line == 3 and pc is not None and Fraction(*pc) <= ALPHA:
        adds.append(ADD3)
    secs = {}
    for key, hi in SECS:
        lo = BLIND if hi == CTRL else CTRL
        f = {CTRL: [], BLIND: []}
        r = {CTRL: [], BLIND: []}
        for c in cl:
            v = secondary(root, c, key, prim[c["name"]])
            if v is None:
                f[c["arm"]].append(-M if c["arm"] == hi else M)
            else:
                f[c["arm"]].append(v)
                r[c["arm"]].append(v)
        spf = perm_p(f[hi], f[lo])
        spc = perm_p(r[hi], r[lo])
        sup = Fraction(*spf) <= ALPHA
        notes = []
        if sup and Fraction(*spf) <= ALPHA / 3:
            notes.append(BONF)
        if line == 1:
            notes.append(UNDER1)
        secs[key] = {"p_filled": spf, "p_as_counted": spc, "supported": sup, "verdict": SUP if sup else NOTSUP,
                     "notes": notes, "n_read": {a: len(r[a]) for a in (CTRL, BLIND)}}
    return {"status": "complete", "line": line, "additions": adds, "p_filled": pf, "p_as_counted": pc,
            "control_median": med, "control_rise": rise, "informative": informative,
            "missing": {a: 4 - len(read_[a]) for a in (CTRL, BLIND)}, "secondaries": secs,
            "primaries": prim}
