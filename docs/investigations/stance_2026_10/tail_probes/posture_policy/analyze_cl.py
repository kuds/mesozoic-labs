"""Test 2 closed loop: the pitch illusion against each policy's own baseline (run_base.py outputs).  Read only.

python -B analyze_cl.py > analysis_cl.txt     (plain numpy; no simulation)

1. Bit identity of the baseline against the clamp runner's unforced arms (return per (policy, seed)).
2. Baseline levels per lineage: the policy's own tail_3 command (window 5-10 s), its share of window steps at >= 0.99,
   tail_3 angle, pelvis pitch (asin of the window-mean forward z), return.  Median over 20 seeds per policy, then the
   median and range over policies.
3. Each illusion arm minus the baseline, paired by (policy, seed): per policy the median over seeds and the count of
   seeds up / down; across policies the median, range, count and an exact two-sided sign test (description only).
4. The rail's formation: time course of the own tail_3 command (bins from run_base.EDGES), first bin >= 0.9.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLAMP = HERE.parent / "clamp"
NAMES = ([f"s42_A_{n}" for n in range(1101, 1152, 10)] + [f"s42_B_{n}" for n in range(1201, 1252, 10)]
         + [f"s80_A_{n}" for n in range(1301, 1352, 10)] + [f"s80_B_{n}" for n in range(1401, 1452, 10)])


def load(d: Path) -> dict:
    out = {}
    p = d / "episodes.jsonl"
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        out[(r["who"], r["seed"])] = r
    return out


def sign_p(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    m = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, m + 1)) / 2 ** n
    return min(1.0, 2 * p)


def key(r: dict, k: str) -> float:
    if k == "own14":
        return r["own_mean"][14]
    if k == "head_cmd":
        return r["own_mean"][2]
    if k == "neck_cmd":
        return r["own_mean"][0]
    if k == "fore_aft_mm":
        return 1000 * r["metrics5"]["com_fore_aft_mean_m"] if r.get("metrics5") else float("nan")
    if k == "height_mm":
        return 1000 * r["metrics5"]["com_height_mean_m"] if r.get("metrics5") else float("nan")
    if k == "nosedive_sum":
        return r["terms_500_1000"].get("reward_nosedive", float("nan"))
    if k == "rail_ep":
        return float(r["own_mean"][14] >= 0.99)
    return r[k]


KEYS = ["own14", "own14_sat_share", "raw14_mean", "tail3_deg", "pitch_deg", "seen_pitch_deg", "head_cmd", "neck_cmd",
        "fore_aft_mm", "height_mm", "nosedive_sum", "return"]


def groups(who: str) -> str:
    return who if who.startswith("start") or who == "statue" else who[:3]


def main() -> None:
    base = load(HERE / "run/base")
    print("=" * 110)
    print("1. BASELINE BIT IDENTITY against the clamp runner's unforced arms (return per policy and seed)")
    refs = {}
    for sub in ("run/s42_unforced", "run/s80_unforced", "start/s42_unforced", "start/s80_unforced", "replay/statue"):
        p = CLAMP / sub / "episodes.jsonl"
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                who = r["who"]
                if sub.startswith("start"):
                    who = "start_" + sub.split("/")[1][:3]
                if r.get("force") is None:
                    refs[(who, r["seed"])] = r["return"]
    same = sum(1 for k, r in base.items() if k in refs and refs[k] == r["return"])
    have = sum(1 for k in base if k in refs)
    print(f"   {same} of {have} matched episodes have an identical return (baseline has {len(base)} episodes;"
          f" {len(base) - have} without a reference)")
    stood = sum(1 for r in base.values() if r["stood"])
    print(f"   stood the full 1,000 steps: {stood} of {len(base)}")

    print("=" * 110)
    print("2. BASELINE LEVELS (window steps 501-1000; per-policy median over 20 seeds; median [range] over policies)")
    per = defaultdict(lambda: defaultdict(list))
    for (who, seed), r in base.items():
        for k in KEYS + ["rail_ep"]:
            per[who][k].append(key(r, k))
    for g in ("statue", "start_s42", "start_s80", "s42", "s80"):
        whos = [w for w in per if groups(w) == g]
        line = []
        for k in KEYS + ["rail_ep"]:
            meds = [float(np.nanmedian(per[w][k])) if k != "rail_ep" else float(np.mean(per[w][k])) for w in whos]
            if len(whos) == 1:
                line.append(f"{k} {meds[0]:+.4g}")
            else:
                line.append(f"{k} {np.median(meds):+.4g} [{min(meds):+.4g},{max(meds):+.4g}]")
        print(f"   {g:10s} (n_policies {len(whos)}): " + "; ".join(line))
    print("   per policy (own14 median, rail share of seeds, tail3 deg, pitch deg):")
    for w in ["start_s42", "start_s80"] + NAMES:
        if w in per:
            print(f"     {w:11s} own14 {np.median(per[w]['own14']):+.3f} rail_seeds {np.mean(per[w]['rail_ep']):.2f}"
                  f" tail3 {np.median(per[w]['tail3_deg']):+.2f} pitch {np.median(per[w]['pitch_deg']):+.3f}"
                  f" seen {np.median(per[w]['seen_pitch_deg']):+.3f}")

    for arm, label in (("toward", "TOWARD the other lineage (s42 +1.10 deg nose-up seen; s80 -1.10 deg)"),
                       ("away", "AWAY from the other lineage (s42 -1.10 deg; s80 +1.10 deg)"),
                       ("towardgrav", "TOWARD, GRAVITY + ACCELEROMETER CHANNELS ONLY, prey direction untouched"
                                      " (s42 +1.10 deg; added on resume)")):
        ill = {}
        for lin in ("s42", "s80"):
            ill.update(load(HERE / f"run/{arm}_{lin}"))
        if not ill:
            continue
        print("=" * 110)
        print(f"3. PITCH ILLUSION {label}: illusion minus baseline, paired by (policy, seed)")
        pairs = defaultdict(lambda: defaultdict(list))
        for k_, r in ill.items():
            b = base.get(k_)
            if b is None:
                continue
            for k in KEYS:
                pairs[k_[0]][k].append(key(r, k) - key(b, k))
            pairs[k_[0]]["rail_ep_ill"].append(key(r, "rail_ep"))
            pairs[k_[0]]["rail_ep_base"].append(key(b, "rail_ep"))
            pairs[k_[0]]["stood"].append(r["stood"])
        for g in ("start_s42", "start_s80", "s42", "s80"):
            whos = [w for w in pairs if groups(w) == g]
            if not whos:
                continue
            nep = sum(len(pairs[w]["own14"]) for w in whos)
            nstood = sum(sum(pairs[w]["stood"]) for w in whos)
            print(f"   {g} ({len(whos)} policies, {nep} paired episodes, stood {nstood}/{nep}):")
            for k in KEYS:
                meds = np.array([float(np.nanmedian(pairs[w][k])) for w in whos])
                ups = int((meds > 0).sum())
                dns = int((meds < 0).sum())
                ep = np.concatenate([pairs[w][k] for w in whos])
                eu, ed = int((ep > 0).sum()), int((ep < 0).sum())
                if len(whos) == 1:
                    print(f"     {k:16s} median {meds[0]:+.4g}; seeds +{eu}/-{ed}")
                else:
                    print(f"     {k:16s} median {np.median(meds):+.4g} [{meds.min():+.4g},{meds.max():+.4g}];"
                          f" policies +{ups}/-{dns} p {sign_p(ups, ups + dns):.4f}; episodes +{eu}/-{ed}")
            rb = np.mean(np.concatenate([pairs[w]["rail_ep_base"] for w in whos]))
            ri = np.mean(np.concatenate([pairs[w]["rail_ep_ill"] for w in whos]))
            print(f"     railed episodes (window own14 mean >= 0.99): baseline {rb:.3f} -> illusion {ri:.3f}")
            if len(whos) > 1:
                for w in whos:
                    print(f"       {w:11s} d own14 {np.median(pairs[w]['own14']):+.3f} (+{int((np.array(pairs[w]['own14'])>0).sum())}/-{int((np.array(pairs[w]['own14'])<0).sum())})"
                          f" d tail3 {np.median(pairs[w]['tail3_deg']):+.2f} deg, d pitch {np.median(pairs[w]['pitch_deg']):+.3f},"
                          f" d seen {np.median(pairs[w]['seen_pitch_deg']):+.3f}, rail {np.mean(pairs[w]['rail_ep_base']):.2f}->{np.mean(pairs[w]['rail_ep_ill']):.2f}")
        print("   time course of the own tail_3 command (mean over seeds; bins 0-4 s by 0.25 s, then 4-10 s by 1 s):")
        for g in ("start_s42", "start_s80", "s42", "s80"):
            whos = sorted({w for (w, _s) in ill if groups(w) == g})
            if not whos:
                continue
            bb = np.mean([base[(w, s)]["own14_bins"] for (w, s) in ill if groups(w) == g], axis=0)
            ib = np.mean([ill[(w, s)]["own14_bins"] for (w, s) in ill if groups(w) == g], axis=0)
            tb = np.mean([base[(w, s)]["tail3_deg_bins"] for (w, s) in ill if groups(w) == g], axis=0)
            ti = np.mean([ill[(w, s)]["tail3_deg_bins"] for (w, s) in ill if groups(w) == g], axis=0)

            def first(x):
                idx = np.where(np.asarray(x) >= 0.9)[0]
                return f"bin {idx[0]}" if len(idx) else "never"
            print(f"     {g:10s} base  own14 " + " ".join(f"{v:+.2f}" for v in bb) + f"  (first >= 0.9: {first(bb)})")
            print(f"     {g:10s} illus own14 " + " ".join(f"{v:+.2f}" for v in ib) + f"  (first >= 0.9: {first(ib)})")
            print(f"     {g:10s} base  tail3 " + " ".join(f"{v:+.1f}" for v in tb))
            print(f"     {g:10s} illus tail3 " + " ".join(f"{v:+.1f}" for v in ti))


if __name__ == "__main__":
    main()
