"""The 20 s check of the s42 pitch illusion (run_long.py: run/long_base, run/long_toward).  Read only; plain numpy.

python -B analyze_long.py > analysis_long.txt

Pairs illusion and baseline by (policy, seed) (12 s42 ends + start_s42, seeds 3042, 3046, 3050, 3055, 3061; 2,000
steps); compares the 5-10 s window (as in the 10 s arms) with the last 5 s (15-20 s), and prints the 1 s time course.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(d):
    out = {}
    for line in open(HERE / d / "episodes.jsonl"):
        r = json.loads(line)
        out[(r["who"], r["seed"])] = r
    return out


def main():
    b = load("run/long_base")
    t = load("run/long_toward")
    keys = sorted(set(b) & set(t))
    print(f"paired episodes {len(keys)}; stood: base {sum(b[k]['stood'] for k in keys)}, illusion {sum(t[k]['stood'] for k in keys)}")
    for grp, sel in (("start_s42", lambda w: w == "start_s42"), ("s42 ends", lambda w: w.startswith("s42"))):
        ks = [k for k in keys if sel(k[0])]
        whos = sorted({k[0] for k in ks})
        print("=" * 100)
        print(f"{grp}: {len(whos)} policies, {len(ks)} paired episodes")
        rows = [("own14 5-10 s", lambda r: r["own_mean"][14]), ("own14 15-20 s", lambda r: r["own14_late"]),
                ("sat share 5-10 s", lambda r: r["own14_sat_share"]), ("sat share 15-20 s", lambda r: r["own14_sat_late"]),
                ("tail3 deg 5-10 s", lambda r: r["tail3_deg"]), ("tail3 deg 15-20 s", lambda r: r["tail3_deg_late"]),
                ("pitch deg 5-10 s", lambda r: r["pitch_deg"]), ("pitch deg 15-20 s", lambda r: r["pitch_deg_late"]),
                ("seen pitch 5-10 s", lambda r: r["seen_pitch_deg"]), ("seen pitch 15-20 s", lambda r: r["seen_pitch_deg_late"]),
                ("railed 15-20 s (own14 >= 0.99)", lambda r: float(r["own14_late"] >= 0.99))]
        for name, f in rows:
            per = defaultdict(list)
            for k in ks:
                per[k[0]].append((f(b[k]), f(t[k])))
            bm = np.array([np.mean([x[0] for x in per[w]]) for w in whos])
            tm = np.array([np.mean([x[1] for x in per[w]]) for w in whos])
            d = tm - bm
            print(f"   {name:32s} base {np.median(bm):+.3f}  illusion {np.median(tm):+.3f}  diff median {np.median(d):+.3f}"
                  f" [{d.min():+.3f},{d.max():+.3f}] policies +{int((d > 1e-9).sum())}/-{int((d < -1e-9).sum())}")
        for name, fld in (("own14", "own14_bins_1s"), ("tail3 deg", "tail3_deg_bins_1s"), ("pitch deg", "pitch_deg_bins_1s"),
                          ("seen pitch", "seen_pitch_deg_bins_1s")):
            bb = np.mean([b[k][fld] for k in ks], axis=0)
            tt = np.mean([t[k][fld] for k in ks], axis=0)
            print(f"   1 s bins {name:10s} base " + " ".join(f"{v:+.2f}" for v in bb))
            print(f"   1 s bins {name:10s} illu " + " ".join(f"{v:+.2f}" for v in tt))
        print("   per policy (mean over 5 seeds) own14 15-20 s, base -> illusion:")
        for w in whos:
            kk = [k for k in ks if k[0] == w]
            print(f"     {w:11s} {np.mean([b[k]['own14_late'] for k in kk]):+.3f} -> {np.mean([t[k]['own14_late'] for k in kk]):+.3f}"
                  f"   (5-10 s {np.mean([b[k]['own_mean'][14] for k in kk]):+.3f} -> {np.mean([t[k]['own_mean'][14] for k in kk]):+.3f})")


if __name__ == "__main__":
    main()
