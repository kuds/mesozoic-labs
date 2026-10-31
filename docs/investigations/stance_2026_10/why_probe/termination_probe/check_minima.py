"""Minima of the lowest tail point and the tail tip over NON-terminated episodes, per condition (offline; for the
summary's statement about how close the tail comes to the floor in standing states).
    PY -I -B check_minima.py > check_minima.txt"""
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_rollouts as A  # noqa: E402

for tag in ["p1048576_free", "p2097152_free", "c2511_3145728_free", "c2531_3145728_free", "c2511_E_free", "c2531_E_free",
            "b2511_E_free", "p2097152_hold0.0", "p2097152_hold-1.0", "p2097152_hold1.0"]:
    c = A.Cond(tag)
    nt = ~c.terms
    lo, tp = c.low[nt], c.tip[nt]
    i = int(np.argmin(lo))
    late = nt & (c.t >= 50)
    print(f"{tag:20s} non-terminated: min lowest {lo.min():.3f} m (at t={c.t[nt][i]}), min tip {tp.min():.3f} m;  "
          f"t >= 50: min lowest {c.low[late].min():.3f} m, min tip {c.tip[late].min():.3f} m")
