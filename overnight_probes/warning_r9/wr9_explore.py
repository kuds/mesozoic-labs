"""warning_r9 EXPLORATORY (PREREG_STAGE2 section 4, last item; not a test): r9 bank_ss trail8 at the 1,048,576-step
marks, maxima, and whether ANY threshold on it separates the O2u-alarm runs from the others (r9), side by side with
r10 (whose outcomes there are the named FOB).  Read only."""
import sys
import numpy as np
sys.path.insert(0, ".")
from wr9_apply import sources, probe, per_eval, trailing, first_eval
marks = [1048576 * k for k in range(1, 12)] + [11927552]
for which in ("r10", "r9"):
    print(f"=== {which}: trail8(bank_ss) at the marks (M steps)")
    print("run  " + " ".join(f"{m / 1e6:6.2f}" for m in marks) + " | first outcome used")
    S = {}
    for run, (lp, pp) in sources(which).items():
        e = per_eval(lp, unnamed=(which == "r9"))
        pr = probe(pp)
        t, tr = pr["t"], trailing(pr["bank_ss"], 8)
        o = first_eval(e, "fob") if which == "r10" else first_eval(e, "any_alarm")
        o1 = first_eval(e, "over_any_ep")
        S[run] = (t, tr, o, o1, int(e["n_over"].sum()))
        vals = [tr[np.flatnonzero(t <= m)[-1]] for m in marks]
        print(f"{run}  " + " ".join(f"{v:+6.3f}" for v in vals) + f" | {'FOB' if which == 'r10' else 'O2u alarm'} {o}; first over-bar {o1}; over-bar episodes {S[run][4]}")
    pos = {r: v for r, v in S.items() if v[2] is not None}
    neg = {r: v for r, v in S.items() if v[2] is None}
    print("  positives: max trail8 before the outcome: " + ", ".join(f"{r} {np.max(v[1][v[0] < v[2]]):+.3f}" for r, v in pos.items()))
    print("  negatives: max trail8 over the run (to 11,927,552): " + ", ".join(f"{r} {np.max(v[1][v[0] <= 11927552]):+.3f}" for r, v in neg.items()))
    for H in (2_097_152, 3_145_728, 4_194_304):
        print(f"  negatives: max trail8 up to {H:,}: " + ", ".join(f"{r} {np.max(v[1][v[0] <= H]):+.3f}" for r, v in neg.items())
              + " | positives up to the same step: " + ", ".join(f"{r} {np.max(v[1][v[0] <= min(H, v[2] - 1)]):+.3f}" for r, v in pos.items()))
    lo = min(np.max(v[1][v[0] < v[2]]) for v in pos.values())
    hi = max(np.max(v[1][v[0] <= 11927552]) for v in neg.values()) if neg else float("-inf")
    print(f"  separating threshold exists (min positive pre-outcome max {lo:+.3f} > max negative {hi:+.3f}): {lo > hi}")
