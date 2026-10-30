"""Implied statistics of the frames added between consecutive checkpoints (from the running mean/var and counts).
python -B implied_window.py > implied_window.txt   (reads stats.json; arithmetic only)
RunningMeanStd is a pooled count-weighted mean and variance, so for counts N0 < N1:
  m_new = (N1 m1 - N0 m0) / (N1 - N0);  E[x^2]_new = (N1 (v1 + m1^2) - N0 (v0 + m0^2)) / (N1 - N0)."""
import json, math
import numpy as np
st = json.load(open("stats.json"))["runs"]
SE = {"s42": (3145728, 4718592), "s80": (4194304, 5767168)}
for lin in ("s42", "s80"):
    runs = [r for r in st if r.startswith(lin)]
    print(f"=== {lin}: tail_3 angle (obs 6) of the frames added in each 262,144-step interval, median over 12 runs")
    print("  interval end   frames   mean deg   sd deg   (running: mean, sd)")
    steps = list(range(SE[lin][0], SE[lin][1] + 1, 262144))
    for a, b in zip(steps[:-1], steps[1:]):
        ms, ss, rm, rs, nn = [], [], [], [], []
        for r in runs:
            e0, e1 = st[r][str(a)], st[r][str(b)]
            N0, N1 = e0["count"], e1["count"]
            m0, m1, v0, v1 = e0["mean"][6], e1["mean"][6], e0["var"][6], e1["var"][6]
            m = (N1 * m1 - N0 * m0) / (N1 - N0)
            ex2 = (N1 * (v1 + m1 ** 2) - N0 * (v0 + m0 ** 2)) / (N1 - N0)
            ms.append(math.degrees(m)); ss.append(math.degrees(math.sqrt(max(ex2 - m * m, 0))))
            rm.append(math.degrees(m1)); rs.append(math.degrees(math.sqrt(v1))); nn.append(N1 - N0)
        print(f"  {b:10d} {np.median(nn):9.0f} {np.median(ms):+9.3f} {np.median(ss):8.3f}   ({np.median(rm):+.3f}, {np.median(rs):.3f})")
    e = st[runs[0]][str(SE[lin][0])]
    print(f"  start pair running stats over its {e['count']:.0f} frames: mean {math.degrees(e['mean'][6]):+.3f} deg, sd {math.degrees(math.sqrt(e['var'][6])):.3f} deg")
