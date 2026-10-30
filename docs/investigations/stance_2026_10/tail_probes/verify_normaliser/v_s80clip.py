"""Verifier: does the 0-6 deg sweep hit the normaliser's clip (|z| = 10) in s80?  s80's tail_3 sd is ~0.31 deg, so
z = 10 is reached near 0.11 + 10 x 0.31 = 3.2 deg; beyond it the network sees a constant input.  Per s80 end (own
seed-3042 states, v_s80ends.jsonl / npz): the clip onset under own and start statistics, the share of sweep points
clipped, and the loop gain over 0-6 deg (verifier's), and over 0-2.5 deg (no clip under any statistics), own vs the
start's tail_3 statistics, plus the start's weights with each end's tail_3 statistics.  Also the s42 clip onset.
Network evaluation only.  PYTHONPATH=<verify_probe/code>:<verify_probe> python -B v_s80clip.py > v_s80clip.txt"""
import json
import math
import sys
import numpy as np
import torch
from v_analyze import load_states
from v_common import PLANT, SE, START, Net, Norm
torch.set_num_threads(1)
TH = np.arange(0.0, 6.01, 0.25)


def g(net, norm, X, hi):
    m = TH <= hi + 1e-9
    sl = []
    for s in X:
        b = np.repeat(np.asarray(s, dtype=np.float32)[None, :], len(TH), axis=0)
        b[:, 6] = np.radians(TH).astype(np.float32)
        mu = net.mean(np.stack([norm(o) for o in b]))[:, 14]
        sl.append(np.polyfit(TH[m], mu[m], 1)[0])
    return PLANT * float(np.median(sl))


def onset(n):
    return math.degrees(n.mean[6] + n.clip * math.sqrt(n.var[6] + n.eps))


st = load_states("v_s80ends.states_*.npz")
s0, e0 = SE["s80"]
netS, nS = Net(START["s80"], s0), Norm(START["s80"], s0)
XS = st["ck__start_s80__3042"]
print(f"s42 clip onset (start stats) {onset(Norm(START['s42'], SE['s42'][0])):.2f} deg; s80 start stats {onset(nS):.2f} deg")
probe = {}
for l in open(__import__('pathlib').Path(__file__).resolve().parent.parent / "probe" / "static.json"):
    pass
pj = json.load(open(__import__('pathlib').Path(__file__).resolve().parent.parent / "probe" / "static.json"))
pown = {r["run"]: r["own"] for r in pj["s80"] if r["step"] == e0}
rows = []
print("run          onset own  own 0-6 (probe)   own 0-2.5  t3q_S 0-6  t3q_S 0-2.5 | startW: own 0-2.5  t3q_E 0-2.5")
for r in [f"s80_A_{n}" for n in range(1301, 1352, 10)] + [f"s80_B_{n}" for n in range(1401, 1452, 10)]:
    netE, nE = Net(r, e0), Norm(r, e0)
    X = st[f"ck__{r}@{e0}__3042"]
    d = dict(run=r, on=onset(nE), o6=g(netE, nE, X, 6), o25=g(netE, nE, X, 2.5), s6=g(netE, nE.mixed(nS, dims=(6,)), X, 6),
             s25=g(netE, nE.mixed(nS, dims=(6,)), X, 2.5), so25=g(netS, nS, X, 2.5), se25=g(netS, nS.mixed(nE, dims=(6,)), X, 2.5))
    rows.append(d)
    print(f"{r}  {d['on']:.2f}      {d['o6']:+.3f} ({pown[r]:+.3f})   {d['o25']:+.3f}     {d['s6']:+.3f}     {d['s25']:+.3f}     | "
          f"{d['so25']:+.3f}          {d['se25']:+.3f}")
A = lambda k: np.array([d[k] for d in rows])  # noqa: E731
print(f"median: own 0-6 {np.median(A('o6')):+.3f}, own 0-2.5 {np.median(A('o25')):+.3f}; end W tail_3 stats from S: change 0-6 "
      f"{np.median(A('s6') - A('o6')):+.3f}, 0-2.5 {np.median(A('s25') - A('o25')):+.3f} ({int((A('s25') < A('o25')).sum())}/12 lower); "
      f"start W with end tail_3 stats, change 0-2.5 {np.median(A('se25') - A('so25')):+.3f}")
print(f"share of the 0-6 deg sweep points with |z6| = 10 (clipped), own stats: median "
      f"{np.median([np.mean(np.radians(TH) > math.radians(d['on'])) for d in rows]):.2f}")
sys.exit(0)
