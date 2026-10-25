"""Hold runs: does the policy's tail_3 mu keep climbing over the episode with the tail held? Quarter means (pooled over 20 eps) and q3 quarter means."""
import sys
import numpy as np
for pre in sys.argv[1:]:
    z = np.load(pre + ".npz")
    modes = sorted({k.rsplit("_", 2)[0] for k in z.files})
    print("####", pre.rsplit("/", 1)[1])
    for m in modes:
        mu = np.stack([z[f"{m}_{i}_mu"] for i in range(20)])
        q3 = np.stack([z[f"{m}_{i}_q3"] for i in range(20)])
        qs = [mu[:, a:a + 200].mean() for a in (200, 400, 600, 800)]
        qq = [q3[:, a:a + 200].mean() for a in (200, 400, 600, 800)]
        print(f"{m:11s} mu quarters {np.round(qs, 4).tolist()}  last100 {mu[:, 900:].mean():.4f}   q3 quarters {np.round(qq, 5).tolist()}")
