"""Verifier: s80_A_1321 seed 3072 (fell in my gate-panel run): termination reason, and the same seed in the lens's
copy tree is not available, so it is re-run twice on main to confirm determinism.  PYTHONPATH=main_tree python -B fall_check.py"""
import numpy as np
import run_verify as rv
fac = rv.factory()
for rep in (1, 2):
    ep, acts = rv.roll("s80_A_1321", 3072, fac)
    tr = ep.track
    z = tr.com[:, 2]
    print(f"rep {rep}: steps {ep.steps} terminated {ep.terminated} reason {ep.termination_reason}; CoM height at spawn {z[0]:.3f} m, "
          f"min {z.min():.3f} m at t {tr.t[np.argmin(z)]:.2f} s, last {z[-1]:.3f} m")
for who in ("start_s80", "s80_A_1301", "statue"):
    ep, _ = rv.roll(who, 3072, fac)
    print(f"{who} seed 3072: steps {ep.steps} stood {ep.stood} reason {ep.termination_reason}")
