"""Summarise out/loop_*.jsonl (verify_loop.py).  Read only.  python -B loop_summary.py > loop_summary.txt"""
import json
from pathlib import Path
import numpy as np
V = Path(__file__).resolve().parent / "out"
for f in ("loop_s42.jsonl", "loop_s80.jsonl"):
    rows = [json.loads(l) for l in open(V / f)]
    head = rows[0]
    gain = head["plant_tail3_deg_at_cmd"]
    g = gain["1.0"] - gain["0.0"]
    print(f"{f}: obs index of tail_3 {head['obs_index_tail3']}; statue tail_3 at cmd 0 {gain['0.0']:+.3f} deg, at +1 {gain['1.0']:+.3f} deg -> plant gain {g:.3f} deg per unit command")
    for cond in ("unforced", "clamped"):
        rs = [r for r in rows[1:] if r["cond"] == cond]
        print(f"  condition {cond}: obs-only sweep of tail_3 angle 0..6 deg, everything else in the observation fixed")
        print(f"    {'policy':12s} {'d cmd14/deg':>11s} {'loop gain':>9s} {'cmd14@0':>8s} {'cmd14@5':>8s} {'d head/deg':>10s} {'d neck/deg':>10s}  obs tail deg at states")
        s14 = []
        sh = []
        for r in rs:
            s = r["tail3"]["slope_per_deg"]
            s14.append(s)
            sh.append(r["head_pitch"]["slope_per_deg"])
            print(f"    {r['who']:12s} {s:+11.4f} {s * g:+9.3f} {r['tail3']['at0']:+8.3f} {r['tail3']['at5']:+8.3f} "
                  f"{r['head_pitch']['slope_per_deg']:+10.4f} {r['neck_pitch']['slope_per_deg']:+10.4f}  "
                  f"{np.median(r['obs_tail3_deg_at_states']):+.2f}")
        ends = [r for r in rs if not r["who"].startswith("start")]
        s14e = np.array([r["tail3"]["slope_per_deg"] for r in ends])
        she = np.array([r["head_pitch"]["slope_per_deg"] for r in ends])
        print(f"    ENDS: d cmd14/deg median {np.median(s14e):+.4f} (>0 {int((s14e > 0).sum())}/{len(s14e)}), loop gain median {np.median(s14e) * g:+.3f}; "
              f"d head_pitch cmd/deg median {np.median(she):+.4f} (<0 {int((she < 0).sum())}/{len(she)}); x5.2 deg = {np.median(she) * 5.2:+.3f}")
