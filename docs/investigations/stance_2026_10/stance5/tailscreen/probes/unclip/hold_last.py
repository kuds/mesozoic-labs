"""Per-episode last-100-step mu at each held a (s42/s80): does any episode's policy want more than the held a?"""
import sys
import numpy as np
for pre in sys.argv[1:]:
    z = np.load(pre + ".npz")
    for a in ("1.05", "1.073", "1.10"):
        v = np.array([z[f"hold_{a}_{i}_mu"][900:].mean() for i in range(20)])
        print(pre.rsplit("/", 1)[1], a, "last100 mu max", round(v.max(), 4), "argmax ep", int(v.argmax()), "eps with mu>a", int((v > float(a)).sum()))
