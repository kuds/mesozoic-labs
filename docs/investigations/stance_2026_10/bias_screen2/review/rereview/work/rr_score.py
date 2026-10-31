"""Re-reviewer's independent scorer, written from criteria.txt alone (THE PRIMARY OUTCOME, THE TEST, THE FILL, THE
INFORMATIVENESS CONDITION, THE READING, THE SECONDARIES, RUN HANDLING, D7's PROVISIONAL item).  It does not import
score_bs.py.  Compares its results with <root>/score/score.json.   PY -I -B rr_score.py <root> [<root> ...]"""
import itertools, json, math, sys
from fractions import Fraction
from pathlib import Path

S, E, ROLL = 2097152, 3538944, 8192
B_S = Fraction(135930855, 1073741824)
assert B_S == Fraction(0.1265954738482833)
GATE, ALPHA, BONF = Fraction(1, 5), Fraction(1, 20), Fraction(1, 60)
STEPS8 = [E - ROLL * k for k in range(8, 0, -1)]
assert STEPS8[0] == 3473408 and STEPS8[-1] == 3530752
SEEDS = (2501, 2511, 2521, 2531)
CTRL, BLIND = "control", "critic_blind_tail3"
QUEUE = [(a, s) for s in SEEDS for a in (CTRL, BLIND)]


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except FileNotFoundError:
        return None


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def ptest(hi, lo):
    """hi, lo: lists of (inf, Fraction); p that the `hi` arm is higher: #{splits with hi-sized group sum >= obs} / n."""
    if not hi or not lo:
        return None
    vals = hi + lo
    k = len(hi)
    add = lambda xs: (sum(x[0] for x in xs), sum((x[1] for x in xs), Fraction(0)))
    obs = add(hi)
    combos = list(itertools.combinations(range(len(vals)), k))
    c = sum(1 for cb in combos if add([vals[i] for i in cb]) >= obs)
    return Fraction(c, len(combos))


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def score(root):
    root = Path(root)
    lost = (jload(root / "lost.json") or {}).get("lost") or {}
    ents = []
    for arm, seed in QUEUE:
        name = f"{arm}_{seed}"
        e = dict(name=name, arm=arm, seed=seed)
        if name in lost:
            e["cls"] = "LOST"
        else:
            ck = jload(root / "runs" / name / "check.json")
            if ck is None or ck.get("passed") is not True:
                e["cls"] = "UNDECIDED"
            elif ck["stop_reason"] == "end" and ck["T"] == E:
                e["cls"] = "COUNTED"
            elif ck["stop_reason"] == "collapse" and S < ck["T"] < E:
                e["cls"] = "STOPPED"
            else:
                raise SystemExit(f"{name}: bad check")
        e["m"] = jload(root / "reads" / name / "markers.json") if e["cls"] in ("COUNTED", "STOPPED") else None
        ents.append(e)
    if any(e["cls"] == "UNDECIDED" for e in ents):
        return {"status": "incomplete"}
    # primaries
    for e in ents:
        e["primary"] = None
        if e["cls"] == "COUNTED":
            st, col = e["m"]["probe"]["steps"], e["m"]["probe"]["rollout_mu14_mean"]
            vals = []
            for s in STEPS8:
                idx = [i for i, x in enumerate(st) if x == s]
                if len(idx) != 1 or not fin(col[idx[0]]):
                    vals = None
                    break
                vals.append(Fraction(col[idx[0]]))
            e["primary"] = sum(vals, Fraction(0)) / 8 if vals else None
    ctrl = [e for e in ents if e["arm"] == CTRL]
    blind = [e for e in ents if e["arm"] == BLIND]
    INF = lambda sgn: (sgn, Fraction(0))
    fill_c = [(0, e["primary"]) if e["primary"] is not None else INF(-1) for e in ctrl]
    fill_b = [(0, e["primary"]) if e["primary"] is not None else INF(+1) for e in blind]
    p_f = ptest(fill_c, fill_b)
    p_c = ptest([(0, e["primary"]) for e in ctrl if e["primary"] is not None],
                [(0, e["primary"]) for e in blind if e["primary"] is not None])
    med = median([e["primary"] for e in ctrl if e["primary"] is not None])
    rise = med - B_S if med is not None else None
    informative = rise is not None and rise >= GATE
    line = 1 if not informative else (2 if p_f <= ALPHA else 3)
    adds = []
    if line == 1 and p_f <= ALPHA:
        adds.append("filled p <= 0.05 but not informative")
    if line == 3 and p_c is not None and p_c <= ALPHA:
        adds.append("as-counted p <= 0.05")
    # paired sign flip (report only)
    diffs = []
    for s in SEEDS:
        c = next(e for e in ctrl if e["seed"] == s)["primary"]
        b = next(e for e in blind if e["seed"] == s)["primary"]
        if c is not None and b is not None:
            diffs.append(c - b)
    if diffs:
        obs = sum(diffs, Fraction(0))
        sg = list(itertools.product((1, -1), repeat=len(diffs)))
        p_sf = Fraction(sum(1 for t in sg if sum((a * d for a, d in zip(t, diffs)), Fraction(0)) >= obs), len(sg))
    else:
        p_sf = None
    # secondaries
    def loop_at(m, steps):
        return [r for r in (m.get("loop") or []) if type(r.get("steps")) is int and r["steps"] == steps]

    def sec_val(key, e):
        if e["cls"] != "COUNTED" or e["primary"] is None:
            return None
        m = e["m"]
        if key in ("SEC-G", "SEC-MARGIN"):
            rows = loop_at(m, E)
            if len(rows) != 1:
                return None
            v = rows[0]["OWN"]["P"]["G" if key == "SEC-G" else "b_star"]
        else:
            pe = m["ledger"]["direct"]["per_evaluation"].get(str(E))
            if not isinstance(pe, dict):
                return None
            v = pe.get("cmd_l")
        return Fraction(v) if fin(v) else None

    secs = {}
    for key, hi_arm in (("SEC-G", BLIND), ("SEC-CMD", CTRL), ("SEC-MARGIN", BLIND)):
        lo_arm = CTRL if hi_arm == BLIND else BLIND
        hi = [sec_val(key, e) for e in ents if e["arm"] == hi_arm]
        lo = [sec_val(key, e) for e in ents if e["arm"] == lo_arm]
        pf = ptest([(0, v) if v is not None else INF(-1) for v in hi], [(0, v) if v is not None else INF(+1) for v in lo])
        pc = ptest([(0, v) for v in hi if v is not None], [(0, v) for v in lo if v is not None])
        secs[key] = dict(p_filled=pf, p_as_counted=pc, supported=pf <= ALPHA, bonferroni=pf <= BONF)
    # G(E) > G(S) count per arm
    gcount = {}
    for arm, es in ((CTRL, ctrl), (BLIND, blind)):
        above = of = 0
        for e in es:
            if e["cls"] != "COUNTED":
                continue
            re_, rs = loop_at(e["m"], E), loop_at(e["m"], S)
            if len(re_) == 1 and len(rs) == 1 and fin(re_[0]["OWN"]["P"]["G"]) and fin(rs[0]["OWN"]["P"]["G"]):
                of += 1
                above += re_[0]["OWN"]["P"]["G"] > rs[0]["OWN"]["P"]["G"]
        gcount[arm] = (above, of)
    d7 = sorted(e["name"] for e in ents if e["cls"] == "COUNTED" and len(loop_at(e["m"], E)) != 1)
    return dict(status="complete", line=line, p_filled=p_f, p_as_counted=p_c, median=med, rise=rise,
                informative=informative, additions=len(adds), p_signflip=p_sf, secs=secs, gcount=gcount, d7=d7,
                primaries={e["name"]: e["primary"] for e in ents})


def compare(root):
    mine = score(root)
    theirs = json.loads((Path(root) / "score" / "score.json").read_text())
    rows = []
    F = lambda s: None if s is None else Fraction(s)
    def chk(what, a, b):
        rows.append((what, a == b, str(a), str(b)))
    chk("line", mine["line"], theirs["line"])
    chk("p_filled", mine["p_filled"], F(theirs["p_filled"]["p"]))
    chk("p_as_counted", mine["p_as_counted"], F((theirs.get("p_as_counted") or {}).get("p")))
    chk("informative", mine["informative"], theirs["informative"])
    chk("control_median(float)", float(mine["median"]) if mine["median"] is not None else None, theirs["control_median"])
    chk("control_rise(float)", float(mine["rise"]) if mine["rise"] is not None else None, theirs["control_rise"])
    chk("n_additions", mine["additions"], len(theirs["additions"]))
    chk("p_signflip", mine["p_signflip"], F((theirs.get("paired_sign_flip") or {}).get("p")))
    for c in theirs["classes"]:
        ex = c.get("primary_exact")
        chk(f"primary {c['name']}", mine["primaries"][c["name"]], F(ex) if ex else None)
    for s in theirs["secondaries"]:
        m = mine["secs"][s["key"]]
        chk(f"{s['key']} p_filled", m["p_filled"], F(s["p_filled"]["p"]))
        chk(f"{s['key']} p_as_counted", m["p_as_counted"], F((s.get("p_as_counted") or {}).get("p")))
        chk(f"{s['key']} supported", m["supported"], s["supported"])
        chk(f"{s['key']} bonferroni note", m["bonferroni"], any("Bonferroni" in n for n in s.get("notes") or []))
    g = theirs["g_end_above_start"]
    for arm in (CTRL, BLIND):
        chk(f"G(E)>G(S) {arm}", mine["gcount"][arm], (g[arm]["above"], g[arm]["of"]))
    d7_theirs = sorted(x.split(":")[0] for x in theirs["provisional"] if "loop rows at E" in x)
    chk("D7 provisional names", mine["d7"], d7_theirs)
    return rows


if __name__ == "__main__":
    bad = 0
    for root in sys.argv[1:]:
        print(f"== {root}")
        for what, ok, a, b in compare(root):
            bad += not ok
            print(f"  {'OK  ' if ok else 'DIFF'} {what}: mine {a} | score_bs {b}")
    print(f"disagreements: {bad}")
    sys.exit(1 if bad else 0)
