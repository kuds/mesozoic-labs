import sys, numpy as np
for p in sys.argv[1:]:
    print("==", p)
    with np.load(p, allow_pickle=False) as z:
        for k in z.files:
            a = z[k]
            print(f"  {k}: dtype={a.dtype} shape={a.shape}" + (f" first={a.ravel()[:3]} last={a.ravel()[-3:]}" if a.size and a.dtype.kind in 'fiub' else (f" val={a.ravel()[:3]}" if a.size else "")))
