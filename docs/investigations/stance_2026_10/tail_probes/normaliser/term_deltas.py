"""Which reward terms carry the return change of each swap (paired by policy and seed against the own-statistics
baseline; median over seeds per policy, then median over the 12 policies, and policies up/down).  Read only.
python -B term_deltas.py > term_deltas.txt   (reads swaps/episodes.jsonl)"""
import json
from collections import defaultdict
import numpy as np
rows = [json.loads(l) for l in open("swaps/episodes.jsonl")]
by = defaultdict(dict)
for r in rows:
    by[(r["cond"], r["who"])][r["seed"]] = r
for lin in ("s42", "s80"):
    conds = ["endW_startN", "endW_startN_tail", "endW_startN_t3q", "startW_endN", "startW_endN_tail"]
    for c in conds:
        whos = sorted(w for (cc, w) in by if cc == c and w.startswith(lin))
        if not whos:
            continue
        per = defaultdict(list)
        for w in whos:
            base = by[("own", f"start_{lin}")] if c.startswith("startW") else by[("own", w)]
            d = by[(c, w)]
            keys = d[3042]["terms"].keys()
            for k in keys:
                per[k].append(np.median([d[s]["terms"][k] - base[s]["terms"][k] for s in d if s in base]))
            per["RETURN"].append(np.median([d[s]["return"] - base[s]["return"] for s in d if s in base]))
        items = sorted(per.items(), key=lambda kv: -abs(np.median(kv[1])))
        print(f"{lin} {c}: " + "; ".join(f"{k.replace('reward_', '')} {np.median(v):+.1f} ({sum(x > 0 for x in v)}/{sum(x < 0 for x in v)})"
                                         for k, v in items[:7] if k != "reward_total"))
