"""Static split of the tail_3-angle statistics swap into its mean part and its variance (scale) part.  Network
evaluation only on stored seed-3042 states (ckpts/states); no simulation, no training.
PYTHONPATH=<policies/code_copy>:<this folder> python -B mean_vs_var.py > mean_vs_var.txt
For each end (12 s42, 12 s80), on the end's own states: loop gain with the end's own statistics; with only obs 6's
variance from the start (end mean kept); with only obs 6's mean from the start (end variance kept); with both.
And the start network on the start's states with only obs 6's variance / mean / both taken from each end."""
import sys
from pathlib import Path
import numpy as np
from probe_common import NAMES, SE, START_RUN, T3Q, Agent, label, make_factory, stats, sweep_states
HERE = Path(__file__).resolve().parent
factory = make_factory()
for lin in ("s42", "s80"):
    s0, e0 = SE[lin]
    mS, vS = stats(START_RUN[lin], s0)
    start = Agent(START_RUN[lin], s0, factory)
    XS = np.load(HERE / "ckpts" / "states" / f"start_{lin}_3042.npy")
    rows = []
    for run in [r for r in NAMES if r.startswith(lin)]:
        ag = Agent(run, e0, factory)
        X = np.load(HERE / "ckpts" / "states" / f"{label(run, e0)}_3042.npy")
        mE, vE = ag.own_mean.copy(), ag.own_var.copy()
        out = []
        for (m, v) in ((mE, vE), (mE, vS), (mS, vE), (mS, vS)):
            mm, vv = mE.copy(), vE.copy()
            mm[T3Q], vv[T3Q] = m[T3Q], v[T3Q]
            ag.set_stats(mm, vv)
            out.append(sweep_states(ag, X)["loop_gain"])
        for (m, v) in ((mS, vS), (mS, vE), (mE, vS), (mE, vE)):
            mm, vv = mS.copy(), vS.copy()
            mm[T3Q], vv[T3Q] = m[T3Q], v[T3Q]
            start.set_stats(mm, vv)
            out.append(sweep_states(start, XS)["loop_gain"])
        rows.append(out)
        print(f"{lin} {run}: end W on X_end: own {out[0]:+.3f} | var_S {out[1]:+.3f} | mean_S {out[2]:+.3f} | both_S {out[3]:+.3f}"
              f"   start W on X_S: own {out[4]:+.3f} | var_E {out[5]:+.3f} | mean_E {out[6]:+.3f} | both_E {out[7]:+.3f}")
    med = np.median(np.array(rows), axis=0)
    print(f"{lin} MEDIAN: end W: own {med[0]:+.3f}, start var only {med[1]:+.3f}, start mean only {med[2]:+.3f}, both {med[3]:+.3f}; "
          f"start W: own {med[4]:+.3f}, end var only {med[5]:+.3f}, end mean only {med[6]:+.3f}, both {med[7]:+.3f}")
