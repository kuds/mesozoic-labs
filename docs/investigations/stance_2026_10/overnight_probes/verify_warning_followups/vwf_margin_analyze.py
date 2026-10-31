"""Verifier's analysis of the margin sweep: (1) re-derive M from warning_margin/episodes.jsonl with own code; (2) compare
own re-run (vwf_margin_episodes.jsonl) per seed; (3) held-out seeds 4001-4020; (4) fragility of the pre-registered
verdicts (rho = 1.0, Q1 AS CLOSE) via paired bootstrap over seeds.  No simulation."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

NOTES = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes")
HERE = NOTES / "verify_warning_followups"
rng = np.random.default_rng(7)


def thresholds(rows, key_kind, key_val, key_park, kind="bias"):
    """Per-seed T_s = smallest visited nudge value that parks (0 if the baseline parks; 2.0 if none, by construction)."""
    by = defaultdict(list)
    for r in rows:
        by[r["seed"]].append(r)
    T = {}
    for s, rr in by.items():
        if any(r[key_kind] == "none" and r[key_park] for r in rr):
            T[s] = 0.0
            continue
        parked = [r[key_val] for r in rr if r[key_kind] == kind and r[key_park]]
        T[s] = min(parked) if parked else (2.0 if kind == "bias" else float("inf"))
    return T


def M_of(T):
    v = np.sort(np.array(list(T.values()), dtype=float))
    return float(v[10]) if len(v) >= 11 else float("nan")


def boot_M(T, n=4000):
    v = np.array(list(T.values()), dtype=float)
    return np.array([np.sort(rng.choice(v, len(v)))[10] for _ in range(n)])


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


orig = [json.loads(x) for x in (NOTES / "warning_margin/episodes.jsonl").read_text().splitlines() if x.strip()]
mine = [json.loads(x) for x in (HERE / "vwf_margin_episodes.jsonl").read_text().splitlines() if x.strip()] \
    if (HERE / "vwf_margin_episodes.jsonl").exists() else []
pred = json.loads((NOTES / "warning_margin/wm_predict.json").read_text())
ew = pd.read_csv(NOTES / "early_warning/ew_series.csv")

CK = [("s80", 10_485_760), ("s80", 11_534_336), ("s80", 12_001_280), ("s60", 6_291_456), ("s60", 7_340_032),
      ("s42", 2_097_152), ("s42", 3_145_728)]
print("1. Own re-derivation of M_bias from warning_margin/episodes.jsonl (raw rows; own threshold code)")
To = {}
for run, st in CK:
    rr = [r for r in orig if r["run"] == run and r["ckpt_steps"] == st]
    # own parked rule from the raw count, not the stored flag
    for r in rr:
        r["park2"] = r["n_rail_win"] > 250
    assert all(r["park2"] == r["parked"] for r in rr)
    T = thresholds(rr, "kind", "value", "park2")
    To[(run, st)] = T
    Tobs = thresholds(rr, "kind", "value", "park2", kind="obs")
    n12 = sum(1 for r in rr if r["kind"] == "obs" and r["value"] == 12.0 and r["park2"])
    print(f"  {run} {st:>9d}: n seeds {len(T)}; M_bias {M_of(T):.4f}; M_obs {M_of(Tobs):.4f}; obs12 parked {n12}/20; "
          f"episodes {len(rr)}; terminated {sum(r['terminated'] for r in rr)}")
print()
print("2. b*_P exact (ew_series.csv: 1 - G_own - c0_own) and the rank fragility")
bst = {}
for run, st in CK:
    q = ew[(ew["run"] == run) & (ew["steps"] == st)].iloc[0]
    bst[(run, st)] = 1 - q["G_own"] - q["c0_own"]
    print(f"  {run} {st:>9d}: G {q['G_own']:+.5f} c0 {q['c0_own']:+.5f} b*_P {bst[(run, st)]:.5f}; M {M_of(To[(run, st)]):.4f}")
Ms = [M_of(To[c]) for c in CK]
print(f"  Spearman(M, b*_P) = {spearman(Ms, [bst[c] for c in CK]):+.3f}")
# bootstrap the rank correlation over seeds (seeds resampled independently per checkpoint)
B = {c: boot_M(To[c]) for c in CK}
rhos = np.array([spearman([B[c][i] for c in CK], [bst[c] for c in CK]) for i in range(4000)])
print(f"  bootstrap over seeds: P(rho = 1.000) {np.mean(rhos > 0.999):.3f}; rho 2.5/50/97.5% "
      f"{np.quantile(rhos, [0.025, 0.5, 0.975]).round(3).tolist()}")
for k in ("bstar_S", "bstar_L"):
    r2 = np.array([spearman([B[c][i] for c in CK], [pred[f'{c[0]}_{c[1]}'][k] for c in CK]) for i in range(4000)])
    print(f"  {k}: point rho {spearman(Ms, [pred[f'{c[0]}_{c[1]}'][k] for c in CK]):+.3f}; bootstrap P(rho=1) {np.mean(r2 > 0.999):.3f}; "
          f"median {np.median(r2):+.3f}")
print()
print("3. Paired (same seeds) bootstrap of differences in M (s80 vs pre-FOB)")


def paired(c1, c2, n=4000):
    s = sorted(set(To[c1]) & set(To[c2]))
    a = np.array([To[c1][x] for x in s])
    b = np.array([To[c2][x] for x in s])
    d = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s))
        d.append(np.sort(a[i])[10] - np.sort(b[i])[10])
    d = np.array(d)
    return M_of(dict(zip(s, a))) - M_of(dict(zip(s, b))), np.quantile(d, [0.025, 0.975])


for c1, c2 in [((("s80", 12_001_280)), ("s42", 3_145_728)), (("s80", 11_534_336), ("s42", 3_145_728)),
               (("s80", 12_001_280), ("s60", 7_340_032)), (("s80", 11_534_336), ("s80", 12_001_280)),
               (("s80", 10_485_760), ("s80", 11_534_336))]:
    dd, ci = paired(c1, c2)
    print(f"  M{c1} - M{c2} = {dd:+.4f}, paired 95% [{ci[0]:+.4f}, {ci[1]:+.4f}]")
cut = max(M_of(To[("s42", 3_145_728)]), M_of(To[("s60", 7_340_032)])) + 0.05
for c in [("s80", 11_534_336), ("s80", 12_001_280)]:
    bb = B[c] - np.maximum(B[("s42", 3_145_728)], B[("s60", 7_340_032)]) - 0.05
    print(f"  Q1 {c}: M {M_of(To[c]):.4f} vs cut {cut:.4f} (margin {cut - M_of(To[c]):+.4f}); bootstrap P(AS CLOSE) {np.mean(bb <= 0):.3f}")
print()
if mine:
    print("4. Own re-run vs original, same seeds (3042-3061): per-seed T_s and every shared episode")
    key = lambda r, kv, kk: (r["run"], r.get("ckpt", r.get("ckpt_steps")), r[kk], round(r[kv], 9), r["seed"])  # noqa: E731
    od = {key(r, "value", "kind"): r for r in orig}
    n_cmp = n_same = 0
    maxdiff = 0.0
    for r in mine:
        k = key(r, "value", "kind")
        if k in od:
            n_cmp += 1
            o = od[k]
            same = (o["n_rail_win"] == r["n_railed"]) and (o["parked"] == r["parked"])
            if o["cmd_win"] is not None and r["cmd_mean_win"] is not None:
                maxdiff = max(maxdiff, abs(o["cmd_win"] - r["cmd_mean_win"]))
            n_same += same
    print(f"  shared episodes {n_cmp}; identical railed count and parked flag {n_same}; max |window command diff| {maxdiff:.2e}")
    Tm = {}
    for run, st in CK:
        for lo, hi, tag in ((3042, 3061, "same seeds 3042-3061"), (4001, 4020, "held-out seeds 4001-4020")):
            rr = [r for r in mine if r["run"] == run and r["ckpt"] == st and lo <= r["seed"] <= hi]
            if not rr:
                continue
            T = thresholds(rr, "kind", "value", "parked")
            Tm[(run, st, tag)] = T
            n12 = sum(1 for r in rr if r["kind"] == "obs" and r["value"] == 12.0 and r["parked"])
            n12t = sum(1 for r in rr if r["kind"] == "obs" and r["value"] == 12.0)
            bm = boot_M(T)
            extra = ""
            if tag.startswith("same"):
                agree = sum(abs(T[s] - To[(run, st)][s]) < 1e-12 for s in T)
                extra = f"; per-seed T_s identical to original {agree}/{len(T)}"
            print(f"  {run} {st:>9d} [{tag}]: n {len(T)}; M {M_of(T):.4f} boot95 [{np.quantile(bm, .025):.4f}, "
                  f"{np.quantile(bm, .975):.4f}]; median T {np.median(list(T.values())):.4f}; obs12 parked {n12}/{n12t}; "
                  f"terminated {sum(r['terminated'] for r in rr)}{extra}")
    print()
    print("5. Held-out seeds: pre-registered verdicts recomputed (Q1 rule; C1 tolerance; ranking among the available rows)")
    H = {c[:2]: v for c, v in Tm.items() if c[2].startswith("held")}
    if ("s42", 3_145_728) in H:
        ref = max(M_of(H[c]) for c in [("s42", 3_145_728), ("s60", 7_340_032)] if c in H)
        for c in [("s80", 11_534_336), ("s80", 12_001_280)]:
            if c in H:
                v = "AS CLOSE" if M_of(H[c]) <= ref + 0.05 else "FARTHER"
                print(f"  Q1 held-out {c}: M {M_of(H[c]):.4f} vs max pre-FOB {ref:.4f} + 0.05 -> {v}")
        for c, T in H.items():
            b = bst[c]
            ok = abs(M_of(T) - b) <= max(0.05, 0.2 * b)
            print(f"  C1 held-out {c}: M {M_of(T):.4f} b*_P {b:.4f} err {M_of(T) - b:+.4f} match {ok}")
        cs = list(H)
        print(f"  held-out ranking: M order {[c for c in sorted(cs, key=lambda c: M_of(H[c]))]}; "
              f"b*_P order {[c for c in sorted(cs, key=lambda c: bst[c])]}")
        for c1, c2 in [(("s80", 12_001_280), ("s42", 3_145_728)), (("s80", 12_001_280), ("s60", 7_340_032)),
                       (("s80", 11_534_336), ("s80", 12_001_280))]:
            if c1 in H and c2 in H:
                a = np.array([H[c1][s] for s in sorted(H[c1])])
                bb = np.array([H[c2][s] for s in sorted(H[c2])])
                d = [np.sort(a[i])[10] - np.sort(bb[i])[10] for i in (rng.integers(0, 20, 20) for _ in range(4000))]
                print(f"  held-out paired M{c1} - M{c2} = {M_of(H[c1]) - M_of(H[c2]):+.4f}, 95% "
                      f"[{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]")
