"""Verifier: Jz (d command / d z_tail3) growth on MATCHED z ranges.  The probe's x5.4 compares the start's slope over
its own sweep's z range (-0.6..+7.1, where tanh compresses) with the end's over -0.8..+2.5.  Here both networks see the
same z range: (i) the end's range: G(W_end, N_end) / G(W_start, N_start[6<-end]); (ii) the start's range:
G(W_end, N_end[6<-start]) / G(W_start, N_start); both on each end's own seed-3042 states (and on X_start for (i)).
With identical tail_3 statistics the scale cancels, so the gain ratio is the Jz ratio.  Network evaluation only.
PYTHONPATH=<verify_probe/code>:<verify_probe> python -B v_jz.py > v_jz.txt"""
import sys
import numpy as np
import torch
from v_analyze import S42, load_states
from v_common import SE, START, Net, Norm, gain
torch.set_num_threads(1)
st = load_states("v_ckpts.states_*.npz")
s0, e0 = SE["s42"]
netS, nS = Net(START["s42"], s0), Norm(START["s42"], s0)
XS = st["ck__start_s42__3042"]
r_end, r_start, r_end_XS = [], [], []
for r in S42:
    netE, nE = Net(r, e0), Norm(r, e0)
    X = st[f"ck__{r}@{e0}__3042"]
    r_end.append(gain(netE, nE, X) / gain(netS, nS.mixed(nE, dims=(6,)), X))
    r_start.append(gain(netE, nE.mixed(nS, dims=(6,)), X) / gain(netS, nS, X))
    r_end_XS.append(gain(netE, nE, XS) / gain(netS, nS.mixed(nE, dims=(6,)), XS))
for name, v in (("end's z range, X_end", r_end), ("end's z range, X_start", r_end_XS), ("start's z range, X_end", r_start)):
    v = np.array(v)
    print(f"Jz ratio end/start on the {name}: median {np.median(v):.2f} [{v.min():.2f}..{v.max():.2f}]")
print(f"compensation for the scale change alone would need x{np.median([Norm(START['s42'], s0).scale_deg() / Norm(r, e0).scale_deg() for r in S42]):.2f}")
sys.exit(0)
