"""An independent reading of PS's criteria (design review), written from <PS>/criteria.txt THE PRIMARY / THE TEST /
THE INFORMATIVENESS CONDITION / THE READING / THE SECONDARIES, not from score_bs.py.  It reads a PS-shaped root's raw
files (SCREEN.json, lost.json, runs/<name>/check.json, reads/<name>/markers.json, the probe file the markers name) and
returns the line, the p values, the gate and each secondary's verdict.  The fill is implemented with finite sentinels
(+/- BIG, BIG >> every finite value), not with (infinite part, finite part) pairs."""
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

S, E, R8 = 2_097_152, 3_538_944, 8192
B_S = Fraction(135930855, 2 ** 30)          # criteria THE PRIMARY
PRIMARY_STEPS = [S + R8 * k for k in range(168, 176)]
GATE = Fraction(1, 5)
ALPHA = Fraction(1, 20)
BIG = Fraction(10 ** 9)
LINE_KEYS = {1: "NOT INFORMATIVE", 2: "REDUCED", 3: "NO REDUCTION DETECTED"}
SECS = {  # key: (family, arm predicted HIGHER)
    "SEC-CMD": ("efficacy", "control"), "SEC-MARGIN": ("efficacy", "critic_phase"),
    "SEC-MECH": ("efficacy", "control"), "SEC-TAIL2": ("guard", "critic_phase")}


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def perm_p(higher, other):
    """p = #{4-subsets of the 8 whose sum >= the `higher` arm's observed sum} / C(8,4); None values filled: a missing
    `higher` value -> -BIG, a missing `other` value -> +BIG.  Returns (count, total)."""
    hv = [(-BIG if v is None else v) for v in higher]
    ov = [(BIG if v is None else v) for v in other]
    allv = hv + ov
    obs = sum(hv, Fraction(0))
    n = len(hv)
    cnt = tot = 0
    for sub in itertools.combinations(range(len(allv)), n):
        tot += 1
        if sum((allv[i] for i in sub), Fraction(0)) >= obs:
            cnt += 1
    return cnt, tot


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return None
    if n % 2:
        return v[n // 2]
    return (v[n // 2 - 1] + v[n // 2]) / 2


def mean8(seq_steps, seq_vals, steps):
    out = []
    for s in steps:
        hits = [i for i, x in enumerate(seq_steps) if x == s]
        if len(hits) != 1 or not fin(seq_vals[hits[0]]):
            return None
        out.append(Fraction(seq_vals[hits[0]]))
    return sum(out, Fraction(0)) / len(out)


def read_root(root):
    root = Path(root)
    scr = json.loads((root / "SCREEN.json").read_text())
    lost = {}
    if (root / "lost.json").exists():
        lost = json.loads((root / "lost.json").read_text()).get("lost", {})
    entries = []
    for arm, seed in scr["queue"]:
        name = f"{arm}_{seed}"
        e = {"name": name, "arm": arm, "seed": seed, "cls": "UNDECIDED", "prim": None, "sec": {k: None for k in SECS}}
        if name in lost:
            e["cls"] = "LOST"
            entries.append(e)
            continue
        cp = root / "runs" / name / "check.json"
        if not cp.exists():
            entries.append(e)
            continue
        ch = json.loads(cp.read_text())
        if ch.get("passed") is not True or ch.get("failures"):
            entries.append(e)
            continue
        if ch.get("stop_reason") == "end" and ch.get("T") == E:
            e["cls"] = "COUNTED"
        elif ch.get("stop_reason") == "collapse":
            e["cls"] = "STOPPED"
        entries.append(e)
        if e["cls"] != "COUNTED":
            continue
        m = json.loads((root / "reads" / name / "markers.json").read_text())
        pr = m["probe"]
        e["prim"] = mean8(pr["steps"], pr["rollout_mu14_mean"], PRIMARY_STEPS)
        if e["prim"] is None:
            continue                               # UNREADABLE: every secondary MISSING too
        per = m.get("ledger", {}).get("direct", {}).get("per_evaluation", {})
        ev = per.get(str(E))
        if isinstance(ev, dict) and fin(ev.get("cmd_l")):
            e["sec"]["SEC-CMD"] = Fraction(ev["cmd_l"])
        rows = [r for r in m.get("loop", []) if r.get("steps") == E]
        if len(rows) == 1:
            b = rows[0].get("OWN", {}).get("P", {}).get("b_star")
            if fin(b):
                e["sec"]["SEC-MARGIN"] = Fraction(b)
            s = rows[0].get("critic_slope_obs6", {}).get("ONPOLICY", {}).get("effective", {}).get("dV_dq3_per_deg")
            if fin(s):
                e["sec"]["SEC-MECH"] = Fraction(s)
        with np.load(pr["file"], allow_pickle=False) as z:
            ut = [int(x) for x in z["update_timesteps"]]
            col13 = [float(x) for x in np.asarray(z["rollout_mu_mean"])[:, 13].astype(np.float64)]
        e["sec"]["SEC-TAIL2"] = mean8(ut, col13, PRIMARY_STEPS)
    return entries


def reading(entries):
    ctrl = [e for e in entries if e["arm"] == "control"]
    ph = [e for e in entries if e["arm"] != "control"]
    if any(e["cls"] == "UNDECIDED" for e in entries):
        return {"status": "incomplete"}
    filled = perm_p([e["prim"] for e in ctrl], [e["prim"] for e in ph])
    cr = [e["prim"] for e in ctrl if e["prim"] is not None]
    pr = [e["prim"] for e in ph if e["prim"] is not None]
    asc = perm_p(cr, pr) if cr and pr else None
    med = median(cr)
    rise = None if med is None else med - B_S
    informative = rise is not None and rise >= GATE
    pf = Fraction(*filled)
    line = 1 if not informative else (2 if pf <= ALPHA else 3)
    adds = []
    if line == 1 and pf <= ALPHA:
        adds.append("not_informative_but_p")
    if line == 3 and asc is not None and Fraction(*asc) <= ALPHA:
        adds.append("as_counted")
    secs = {}
    for key, (fam, hi) in SECS.items():
        hv = [e["sec"][key] for e in entries if e["arm"] == hi]
        ov = [e["sec"][key] for e in entries if e["arm"] != hi]
        c, t = perm_p(hv, ov)
        p = Fraction(c, t)
        sup = p <= ALPHA
        notes = []
        if sup and fam == "efficacy" and p <= ALPHA / 3:
            notes.append("bonferroni")
        if sup and fam == "guard":
            notes.append("guard")
        if line == 1:
            notes.append("line1")
        secs[key] = {"count": c, "supported": sup, "notes": notes}
    return {"status": "decided", "line": line, "filled": filled, "as_counted": asc, "rise": rise,
            "informative": informative, "additions": adds, "secs": secs}


if __name__ == "__main__":
    import sys
    r = reading(read_root(sys.argv[1]))
    print(json.dumps({k: (str(v) if isinstance(v, Fraction) else v) for k, v in r.items()}, default=str, indent=1))
