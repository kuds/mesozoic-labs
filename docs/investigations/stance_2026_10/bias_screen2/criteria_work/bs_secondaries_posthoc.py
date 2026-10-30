"""Context only (post hoc on BS's finished data; NOT a test of anything): BS2's three secondaries' statistics computed on
BS's own report-only values (<BS>/score/score.json report_only), with the same exact permutation rule.  Reads only.
Usage: PY -I -B bs_secondaries_posthoc.py"""
import itertools
import json
from fractions import Fraction

BS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/bias_screen"
sj = json.load(open(BS + "/score/score.json"))
ro = sj["report_only"]
E = 2490368


def perm_p(hi, lo):
    """share of 70 relabelings whose 'hi'-arm sum >= observed (ties against)."""
    vals = [Fraction(v) for v in hi + lo]
    obs = sum(Fraction(v) for v in hi)
    cnt = sum(1 for c in itertools.combinations(range(8), 4) if sum(vals[i] for i in c) >= obs)
    return Fraction(cnt, 70)


def get(name, key):
    r = ro[name]
    if key == "cmd_l":
        return r["cmd_l_last"]
    row = [x for x in r["loop"] if x["steps"] == E][0]
    return row[key]


arms = {"control": [f"control_{s}" for s in (2101, 2111, 2121, 2131)],
        "blind": [f"critic_blind_tail3_{s}" for s in (2101, 2111, 2121, 2131)]}
for key, direction in (("OWN_G", "blind higher"), ("cmd_l", "blind lower"), ("OWN_b_star", "blind higher")):
    c = [get(n, key) for n in arms["control"]]
    b = [get(n, key) for n in arms["blind"]]
    p = perm_p(b, c) if direction == "blind higher" else perm_p(c, b)
    print(f"{key:<11} control {[round(x, 4) for x in c]} blind {[round(x, 4) for x in b]} ({direction}): p = {p} = "
          f"{float(p):.4f}")
