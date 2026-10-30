import sys, zipfile, numpy as np
for p in sys.argv[1:]:
    print("==", p)
    with np.load(p, allow_pickle=False) as z:
        for k in z.files:
            a = z[k]
            print(f"  {k}: dtype {a.dtype} shape {a.shape}")
