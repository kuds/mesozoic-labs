"""Verifier: re-derive the reward-term deltas behind the swaps' return changes from the probe's swaps/episodes.jsonl
(read only; its returns are checked against my own rollouts in v_analyze part C).  Per policy: median over seeds of the
paired (same seed) change; then median over policies and policies up/down.  Also how much of the median return change
the named term covers, per policy (term change / return change, median).  No simulation.
python -B v_terms.py > v_terms.txt"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
P = Path(__file__).resolve().parent.parent / "probe" / "swaps" / "episodes.jsonl"
by = defaultdict(dict)
for l in open(P):
    r = json.loads(l)
    by[(r["cond"], r["who"])][r["seed"]] = r
for lin in ("s42", "s80"):
    for cond in ("endW_startN", "endW_startN_tail", "endW_startN_t3q", "startW_endN", "startW_endN_tail"):
        whos = sorted(w for (c, w) in by if c == cond and w.startswith(lin))
        if not whos:
            continue
        keys = [k for k in by[(cond, whos[0])][3042]["terms"] if k != "reward_total"]
        per = {k: [] for k in keys + ["return"]}
        for w in whos:
            ref = by[("own", f"start_{lin}")] if cond.startswith("startW") else by[("own", w)]
            d = by[(cond, w)]
            for k in keys:
                per[k].append(np.median([d[s]["terms"][k] - ref[s]["terms"][k] for s in d]))
            per["return"].append(np.median([d[s]["return"] - ref[s]["return"] for s in d]))
        R = np.array(per["return"])
        top = sorted(keys, key=lambda k: -abs(np.median(per[k])))[:4]
        txt = ", ".join(f"{k[7:]} {np.median(per[k]):+.1f} ({int((np.array(per[k]) > 0).sum())}/{int((np.array(per[k]) < 0).sum())})" for k in top)
        cover = np.median(np.array(per[top[0]]) / np.where(R == 0, np.nan, R))
        print(f"{lin} {cond:17s} RETURN {np.median(R):+.1f} ({int((R > 0).sum())}/{int((R < 0).sum())}) [{R.min():+.1f}..{R.max():+.1f}]; {txt}; "
              f"{top[0][7:]} / return per policy, median {cover:+.2f}")
