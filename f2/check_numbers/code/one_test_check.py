"""Independent THE ONE TEST (criteria.txt lines 289-302), written from the text only.

Exact one-sided unpaired permutation test over the C(6,3) = 20 ways to split the six TD values 3 + 3.
p = (# splits whose 3-sum <= the PHASE 3-sum) / 20, ties against H1 (<= counts).
Missing: PHASE at +infinity, OFF at -infinity; a 3-sum with infinities is ordered by
(n(+inf) - n(-inf), exact sum of its finite values). Finite sums exact (Fraction of each float).
LOWER text only at p = 1/20 (complete separation).
Inputs: the TD floats from td_check.json (own computation, fsum form; the exact-rational form is also run).
"""
import itertools
import json
import sys
from fractions import Fraction

PINF, MINF = "+inf", "-inf"


def key(vals):
    n = sum(1 for v in vals if v == PINF) - sum(1 for v in vals if v == MINF)
    s = sum((Fraction(v) for v in vals if not isinstance(v, str)), Fraction(0))
    return (n, s)


def one_test(phase, off):
    six = list(phase) + list(off)
    obs = key(phase)
    k = 0
    for idx in itertools.combinations(range(6), 3):
        if key([six[i] for i in idx]) <= obs:
            k += 1
    sep = all((not isinstance(p, str)) for p in phase) and all((not isinstance(o, str)) for o in off) and \
        max(phase) < min(off)
    text = ("LOWER THAN EVERY CONCURRENT SWITCH-OFF RUN (exact permutation p = 1/20)" if k == 1
            else f"NOT SEPARATED (p = {k}/20)")
    return {"k": k, "p": f"{k}/20", "complete_separation": sep, "text": text}


def main():
    td = json.load(open(sys.argv[1]))
    out = {}
    for form in ("fsum", "exact", "seq"):
        ph = [td[f"stance7_r10LTSBpc_s{s}"]["TD"][form] for s in (60, 42, 80)]
        of = [td[f"stance7_r10LTSBoff_s{s}"]["TD"][form] for s in (60, 42, 80)]
        out[form] = {
            "phase": ph, "off": of,
            "pinned_blindness_OFF_not_blind_missing": one_test(ph, [MINF] * 3),
            "as_if_blind_OFF_finite": one_test(ph, of),
        }
    # rule probes (synthetic, to show the oracle follows the text)
    out["probes"] = {
        "one_off_missing_rest_separated": one_test([0.01, 0.02, 0.03], [MINF, 0.5, 0.6])["k"],
        "one_phase_missing_rest_separated": one_test([PINF, 0.02, 0.03], [0.4, 0.5, 0.6])["k"],
        "both_missing": one_test([PINF, 0.02, 0.03], [MINF, 0.5, 0.6])["k"],
        "fully_reversed": one_test([0.4, 0.5, 0.6], [0.01, 0.02, 0.03])["k"],
        "one_off_below_one_phase": one_test([0.01, 0.02, 0.3], [0.2, 0.5, 0.6])["k"],
    }
    json.dump(out, sys.stdout, indent=1, default=str)
    print()


if __name__ == "__main__":
    main()
