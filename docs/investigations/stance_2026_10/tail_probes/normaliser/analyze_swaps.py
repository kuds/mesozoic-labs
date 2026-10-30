"""Summarise the closed-loop statistics swaps (run_swaps.py -> swaps/episodes.jsonl).  Read only; no simulation.

PYTHONPATH=<this folder> python -B analyze_swaps.py > swaps.txt   (also writes swaps_summary.json)

Per condition and policy: median over the 20 seeds of the loop gain (each seed's gain is the median over its 6
states), of the policy's own tail_3 command over steps 500-1000 and its share at >= 0.99 ("parks at the rail"), the
tail_3 angle, the return, falls.  Paired against the policy's own-statistics baseline (same weights, same seed):
return and window-return change, per-policy median and policies up/down, exact two-sided sign test.
The own baseline is also checked against the clamp runner's unforced episodes (clamp/run/<lin>_unforced and
clamp/start/<lin>_unforced): the return must be identical for every (policy, seed).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLAMP = HERE.parent / "clamp"


def sign_p(k, n):
    if n == 0:
        return float("nan")
    lo = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, lo + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> int:
    rows = [json.loads(l) for l in open(HERE / "swaps" / "episodes.jsonl")]
    by = defaultdict(dict)  # (cond, who) -> seed -> row
    for r in rows:
        by[(r["cond"], r["who"])][r["seed"]] = r
    # identity check against the clamp runner's unforced episodes
    ref = {}
    for lin in ("s42", "s80"):
        for f in (CLAMP / "run" / f"{lin}_unforced" / "episodes.jsonl", CLAMP / "start" / f"{lin}_unforced" / "episodes.jsonl"):
            if f.exists():
                for line in open(f):
                    q = json.loads(line)
                    ref[(q["who"], q["seed"])] = q
    n_eq = n_cmp = 0
    maxd = 0.0
    for (cond, who), d in by.items():
        if cond != "own":
            continue
        for seed, r in d.items():
            q = ref.get((who, seed))
            if q is None:
                continue
            n_cmp += 1
            dd = abs(q["return"] - r["return"])
            maxd = max(maxd, dd)
            n_eq += int(dd == 0.0 and abs(q["win"]["500_1000"]["own_mean"][14] - r["own14_mean_5_10"]) < 5e-7)
    print("CLOSED-LOOP STATISTICS SWAPS (run_swaps.py -> swaps/episodes.jsonl; analyze_swaps.py)")
    print(f"own-statistics baseline vs the clamp runner's unforced episodes: {n_eq}/{n_cmp} identical return and own tail_3 "
          f"command (max |return diff| {maxd:.3g})")
    summary = {}
    for lin in ("s42", "s80"):
        start = f"start_{lin}"
        base_start = by.get(("own", start), {})
        conds = sorted({c for (c, w) in by if by[(c, w)] and next(iter(by[(c, w)].values()))["lin"] == lin})
        order = ["own", "endW_startN", "endW_startN_tail", "endW_startN_t3q", "startW_endN", "startW_endN_tail"]
        conds = [c for c in order if c in conds]
        print(f"\n================ {lin} ================")
        if base_start:
            g = [r["sweep"]["loop_gain"] for r in base_start.values() if r["sweep"]]
            print(f"start, own statistics: gain {np.median(g):+.3f} (seed 3042 {base_start[3042]['sweep']['loop_gain']:+.3f}); own14 "
                  f"{np.median([r['own14_mean_5_10'] for r in base_start.values()]):+.3f}; sat share "
                  f"{np.median([r['own14_sat_5_10'] for r in base_start.values()]):.2f}; tail3 "
                  f"{np.median([r['tail3_deg_5_10'] for r in base_start.values()]):+.2f} deg; return "
                  f"{np.median([r['return'] for r in base_start.values()]):.1f}; falls {sum(r['terminated'] for r in base_start.values())}")
        for cond in conds:
            whos = sorted(w for (c, w) in by if c == cond and w.startswith(lin))  # this lineage's ends only
            if not whos:
                continue
            startw = cond.startswith("startW")
            print(f"\n-- {cond}  ({'start weights' if startw else 'end weights'}; paired against "
                  f"{'the start with its own statistics' if startw else 'the same end with its own statistics'})")
            print(f"   {'who':12s} {'gain':>7s} {'g3042':>7s} {'own14':>7s} {'sat':>5s} {'tail3':>6s} {'return':>8s} {'d ret':>8s} "
                  f"{'d win':>8s} {'up/dn':>6s} {'falls':>5s}")
            agg = defaultdict(list)
            for w in whos:
                d = by[(cond, w)]
                base = base_start if startw else by.get(("own", w), {})
                g = [r["sweep"]["loop_gain"] for r in d.values() if r["sweep"]]
                ret = [r["return"] for r in d.values()]
                pairs = [(d[s]["return"] - base[s]["return"], d[s]["window_return_200_1000"] - base[s]["window_return_200_1000"])
                         for s in d if s in base]
                dr = [p[0] for p in pairs]
                dw = [p[1] for p in pairs]
                up = sum(x > 0 for x in dr)
                row = {"gain": float(np.median(g)) if g else float("nan"),
                       "g3042": d[3042]["sweep"]["loop_gain"] if 3042 in d and d[3042]["sweep"] else float("nan"),
                       "own14": float(np.nanmedian([r["own14_mean_5_10"] for r in d.values()])),
                       "sat": float(np.nanmedian([r["own14_sat_5_10"] for r in d.values()])),
                       "tail3": float(np.nanmedian([r["tail3_deg_5_10"] for r in d.values()])),
                       "ret": float(np.median(ret)), "dret": float(np.median(dr)) if dr else float("nan"),
                       "dwin": float(np.median(dw)) if dw else float("nan"), "up": up, "n": len(dr),
                       "falls": int(sum(r["terminated"] for r in d.values()))}
                for k, v in row.items():
                    agg[k].append(v)
                summary[f"{cond}|{w}"] = row
                print(f"   {w:12s} {row['gain']:+7.3f} {row['g3042']:+7.3f} {row['own14']:+7.3f} {row['sat']:5.2f} {row['tail3']:+6.2f} "
                      f"{row['ret']:8.1f} {row['dret']:+8.1f} {row['dwin']:+8.1f} {up:>3d}/{len(dr) - up:<2d} {row['falls']:5d}")
            n = len(whos)
            dret = np.array(agg["dret"])
            k = int((dret > 0).sum())
            if cond == "own":
                print(f"   MEDIAN over {n}: gain {np.median(agg['gain']):+.3f} [{min(agg['gain']):+.3f}, {max(agg['gain']):+.3f}]; "
                      f"own14 {np.median(agg['own14']):+.3f}; sat share {np.median(agg['sat']):.2f}; tail3 {np.median(agg['tail3']):+.2f} deg; "
                      f"return {np.median(agg['ret']):.1f}; falls {sum(agg['falls'])}/{20 * n}  (baseline: no paired change)")
                continue
            print(f"   MEDIAN over {n}: gain {np.median(agg['gain']):+.3f} [{min(agg['gain']):+.3f}, {max(agg['gain']):+.3f}]; "
                  f"own14 {np.median(agg['own14']):+.3f}; sat share {np.median(agg['sat']):.2f}; tail3 {np.median(agg['tail3']):+.2f} deg; "
                  f"d return {np.median(dret):+.1f} [{dret.min():+.1f}, {dret.max():+.1f}] ({k} up / {n - k} down, sign p {sign_p(k, n):.4f}); "
                  f"d window {np.median(agg['dwin']):+.1f}; falls {sum(agg['falls'])}/{20 * n}")
            if not startw and cond != "own":
                own_g = [summary.get(f"own|{w}", {}).get("gain", float("nan")) for w in whos]
                diff = np.array(agg["gain"]) - np.array(own_g)
                print(f"   gain vs the same end's own-statistics gain: median change {np.nanmedian(diff):+.3f} "
                      f"({int((diff < 0).sum())} lower / {int((diff > 0).sum())} higher)")
    json.dump(summary, open(HERE / "swaps_summary.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
