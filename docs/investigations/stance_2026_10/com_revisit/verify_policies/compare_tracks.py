"""Verifier: my main-tree rollouts (float64) cast to float32 against the lens's copy-tree float32 tracks, exactly.
python -I -B compare_tracks.py LENS_RUN_TRACKS_DIR MY_NPZ [prefix_len_steps]
"""
import sys
import numpy as np
lens_dir, mine_path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else None
mine = np.load(mine_path)
keys = sorted({k.rsplit("_", 1)[0] for k in mine.files if k.endswith("_com")})
same = diff = 0
for key in keys:
    who, seed = key.rsplit("_", 1)
    z = np.load(f"{lens_dir}/{who}.npz")
    i = [int(s) for s in z["seeds"]].index(int(seed))
    for mk, lk in (("com", "com"), ("quat", "quat"), ("feet", "feet"), ("load", "load"), ("heading", "heading"), ("act", "action")):
        if f"{key}_{mk}" not in mine.files:
            continue
        a = mine[f"{key}_{mk}"].astype(np.float32)
        b = z[f"{lk}_{i}"]
        if n is not None:
            a, b = a[: n + (0 if mk == "act" else 1)], b[: n + (0 if mk == "act" else 1)]
        eq = a.shape == b.shape and np.array_equal(a, b)
        same += eq; diff += not eq
        print(f"{key:18s} {mk:8s} {'IDENTICAL' if eq else 'DIFFERS'} {a.shape} max|d| {np.max(np.abs(a.astype(float) - b.astype(float))) if a.shape == b.shape else 'shape'}")
print(f"identical {same}, differing {diff}")
