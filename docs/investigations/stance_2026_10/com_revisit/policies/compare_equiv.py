"""Compare equiv.py's output from the sealed copy and from main 378ae2f8, bit for bit.

python3 compare_equiv.py COPY.npz MAIN.npz
"""

import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
keys = sorted(set(a.files) | set(b.files))
same = differ = 0
for k in keys:
    if k not in a.files or k not in b.files:
        print(f"{k}: only in {'copy' if k in a.files else 'main'}")
        differ += 1
        continue
    x, y = a[k], b[k]
    if x.dtype.kind in "fc":
        eq = x.shape == y.shape and np.array_equal(x, y)
        diff = float(np.max(np.abs(x - y))) if x.shape == y.shape else float("nan")
        print(f"{k:28s} {'IDENTICAL' if eq else 'DIFFERS'} shape {x.shape} max|diff| {diff:.3g}")
    else:
        eq = bool(np.all(x == y))
        print(f"{k:28s} {'IDENTICAL' if eq else 'DIFFERS'} {str(x)[:16]} / {str(y)[:16]}")
    same += eq
    differ += not eq
print(f"identical {same}, differing {differ}")
