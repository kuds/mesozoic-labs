"""Reviewer's scratch check (bit-identity lens): does a walk's Stage 1-format file depend on the process's
gait accuracy state at the moment gait.inverse_dynamics is called, after look.generate has restored it?
Nothing is written in the checkout."""
import hashlib
import io
import json
import sys
import time

import numpy as np

from designlab import gait, sizing
from designlab.stage1 import refs
from designlab.stage2 import look

body = sys.argv[1] if len(sys.argv) > 1 else "75-M-H-10"
t0 = time.perf_counter()
src = look.source(body)
print(f"source {time.perf_counter() - t0:.1f} s; accuracy at start: {gait.ACCURACY}")
g = look.generate(src, sizing.NOMINAL)
print(f"after look.generate: process accuracy {gait.ACCURACY}; info accuracy {g['info']['accuracy']}; "
      f"{time.perf_counter() - t0:.1f} s")
M = src["M"]
idr_state = gait.inverse_dynamics(M, g)  # whatever state the process was left in
state = gait.ACCURACY
gait.set_accuracy("r2")
idr_r2 = gait.inverse_dynamics(M, g)
for k in ("qd", "qdd", "tau"):
    d = np.abs(np.asarray(idr_state[k]) - np.asarray(idr_r2[k]))
    rows = np.unique(np.nonzero(d.reshape(len(d), -1).max(1) > 0)[0])
    print(f"{k}: max |diff| {d.max():.3e} on frames {rows.tolist()[:8]}{'...' if len(rows) > 8 else ''} ({len(rows)} frames)")


def sha(idr):
    buf = io.BytesIO()
    np.savez(buf, **refs.arrays(src, dict(traj=g, idr=idr)))
    return hashlib.sha256(buf.getvalue()).hexdigest()


rec = json.load(open(f"stage1/runs/2026-11-08/{body}/refs.json"))["references"]["nominal"]["sha256"]
a, b = sha(idr_state), sha(idr_r2)
print(f"sha256 with inverse dynamics at process state {state}: {a[:16]}  equals Stage 1's record: {a == rec}")
print(f"sha256 with inverse dynamics at r2:                 {b[:16]}  equals Stage 1's record: {b == rec}")
print(f"recorded                                             {rec[:16]}")
