"""Verifier: what makes s42_B_1251 seed 3050's XcoM leave the support polygon in 2-3 s (POL-8), from my main-tree
rollout (sim_equiv/equiv_main.npz).  python -I -B support_transient.py NPZ"""
import sys
import numpy as np
z = np.load(sys.argv[1])
for k in ("s42_B_1251_3050", "statue_3059", "s80_A_1311_3059"):
    t = z[f"{k}_t"]; cm = z[f"{k}_com_margin"]; xm = z[f"{k}_xcom_margin"]; n = z[f"{k}_npts"]; a = z[f"{k}_area"]
    nr, nl = z[f"{k}_near_r"], z[f"{k}_near_l"]; load = z[f"{k}_load"]; sp = z[f"{k}_com_speed"]
    w = (t >= 2.0 - 1e-9)
    i = np.flatnonzero(w)[np.nanargmin(xm[w])]
    j = np.flatnonzero(w)[np.nanargmin(cm[w])]
    w3 = t >= 3.0 - 1e-9
    print(f"{k}: least XcoM margin from 2 s {1000*xm[i]:+.1f} mm at t {t[i]:.2f} s; least CoM margin {1000*cm[j]:+.1f} mm at t {t[j]:.2f} s")
    for idx in sorted(set([i, j])):
        lo, hi = max(idx - 3, 0), min(idx + 4, len(t))
        for q in range(lo, hi):
            print(f"   t {t[q]:.2f} pts {n[q]:2d} (near r {nr[q]:2d}, l {nl[q]:2d}) hull area {1e4*a[q]:7.1f} cm2 load r/l {load[q,0]:6.1f}/{load[q,1]:6.1f} N "
                  f"com margin {1000*cm[q]:+6.1f} xcom margin {1000*xm[q]:+6.1f} mm  CoM speed {1000*sp[q]:6.1f} mm/s")
    print(f"   from 3 s: hull area median {1e4*np.median(a[w3]):.1f} cm2, min {1e4*a[w3].min():.1f}; points median {np.median(n[w3]):.0f}; "
          f"CoM speed in 2-3 s max {1000*sp[w & ~w3].max():.1f} mm/s; samples in 2-3 s with < 3 points on a foot: r {int((nr[w & ~w3] < 3).sum())}, l {int((nl[w & ~w3] < 3).sum())}")
k = "s42_B_1251_3050"
t = z[f"{k}_t"]; xm = z[f"{k}_xcom_margin"]; sp = z[f"{k}_com_speed"]
neg = np.flatnonzero(xm < 0)
print(f"{k}: samples with XcoM outside the polygon: t = {', '.join(f'{x:.2f}' for x in t[neg])}")
print(f"{k}: CoM speed RMS per second 0-5 s (mm/s): " + " ".join(f"{1000*np.sqrt(np.mean(sp[(t>=a-1e-9)&(t<a+1-1e-9)]**2)):.1f}" for a in range(5)))
