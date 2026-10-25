"""Convergence of tail_3 mu in the unclip run, and the apparent loop gain g = (mu_u - mu_c)/(mu_u - 1) at end of episode."""
import sys
import numpy as np
z = np.load(sys.argv[1] + ".npz")
print("ep  base_mu[999] unclip_mu@600 @800 @900 @999  d(900->999) unclip_q3@999  g_end")
for i in range(20):
    mb, mu, q3 = z[f"base_{i}_mu"], z[f"unclip_{i}_mu"], z[f"unclip_{i}_q3"]
    if mb[200:].max() < 1.0:
        continue
    g = (mu[-1] - mb[-1]) / (mu[-1] - 1.0)
    print(f"{i:2d}  {mb[-1]:.4f}      {mu[600]:.4f} {mu[800]:.4f} {mu[900]:.4f} {mu[-1]:.4f}  {mu[-1]-mu[900]:+.5f}   {q3[-1]:+.5f}  {g:.3f}")
