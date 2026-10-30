"""Verifier: analysis of my own rollouts (v_loop.py) and static network evaluation on MY stored states.

PYTHONPATH=<verify_probe/code>:<verify_probe> python -B v_analyze.py > v_analyze.txt

A. Per-checkpoint loop gain (seed 3042, own statistics) for all 73 s42 checkpoints: my gain vs the probe's
   ckpts/episodes.jsonl (read only), medians per step, OLS gain on scale, Spearman, log-log slope, Jz = slope/scale.
B. Static, on my seed-3042 states, all 12 s42 ends and all 12 s80 ends: the 2x2 G(W, N | X) on X_end and X_start,
   the normaliser / weight parts, symmetric share (and a log-scale variant), tail_3-only / tail-8 swaps, tail_3 mean-only
   and variance-only swaps (both orders), layer grafts, the first-layer tail_3 column, and the sweep's linearity.
C. Closed loop: my v_swaps.jsonl against the probe's swaps/episodes.jsonl, row by row (same condition, policy, seed),
   and my own per-condition summary; in-loop normalised-observation checks.
Network evaluation only (no simulation, no training).
"""
from __future__ import annotations

import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from v_common import PLANT, SE, START, TAIL8, THETAS, Net, Norm, gain, sweep_one

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
PROBE = HERE.parent / "probe"
S42 = [f"s42_A_{n}" for n in range(1101, 1152, 10)] + [f"s42_B_{n}" for n in range(1201, 1252, 10)]
S80 = [f"s80_A_{n}" for n in range(1301, 1352, 10)] + [f"s80_B_{n}" for n in range(1401, 1452, 10)]


def ols(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1 - np.sum((y - A @ c) ** 2) / np.sum((y - y.mean()) ** 2)
    return c[0], c[1], r2


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y)[0])


def load_states(pattern):
    out = {}
    for f in sorted(glob.glob(str(HERE / pattern))):
        z = np.load(f)
        out.update({k: z[k] for k in z.files})
    return out


def part_a(states):
    print("=== A. per-checkpoint loop gain, s42, seed 3042, own statistics (v_ckpts.jsonl)")
    mine = {json.loads(l)["who"]: json.loads(l) for l in open(HERE / "v_ckpts.jsonl")}
    probe = {}
    for l in open(PROBE / "ckpts" / "episodes.jsonl"):
        r = json.loads(l)
        if r["lin"] == "s42" and r["seed"] == 3042:
            probe["start_s42" if r["label"] == "start_s42" else r["label"]] = r
    dg = [abs(mine[k]["gain"] - probe[k]["sweep"]["loop_gain"]) for k in mine]
    dr = [abs(mine[k]["return"] - probe[k]["return"]) for k in mine]
    nst = 0
    for k in mine:
        ps = np.load(PROBE / "ckpts" / "states" / f"{probe[k]['label']}_3042.npy")
        nst += int(np.array_equal(ps, states[f"ck__{k}__3042"]))
    print(f"  {len(mine)} checkpoints; vs probe: max|gain diff| {max(dg):.2e}, max|return diff| {max(dr):.2e}; "
          f"states byte-identical {nst}/{len(mine)}")
    s0, e0 = SE["s42"]
    steps = list(range(s0, e0 + 1, 262_144))
    print("  step      gain(med)  scale/deg  Jz(med)")
    jz_by = {}
    for t in steps:
        keys = ["start_s42"] if t == s0 else [f"{r}@{t}" for r in S42]
        g = [mine[k]["gain"] for k in keys]
        sc = [mine[k]["scale6_per_deg"] for k in keys]
        jz = [mine[k]["gain"] / PLANT / mine[k]["scale6_per_deg"] for k in keys]
        jz_by[t] = jz
        print(f"  {t:8d}  {np.median(g):+.3f}     {np.median(sc):.3f}      {np.median(jz):+.4f}")
    x = np.array([mine[k]["scale6_per_deg"] for k in mine])
    y = np.array([mine[k]["gain"] for k in mine])
    b, a, r2 = ols(x, y)
    bl, al, r2l = ols(np.log(x), np.log(y))
    jz = y / PLANT / x
    print(f"  OLS over {len(x)}: gain = {b:+.3f} x scale {a:+.3f}, R^2 {r2:.2f}; Spearman {spearman(x, y):+.2f}; "
          f"log-log slope {bl:+.2f} (R^2 {r2l:.2f}); Spearman(Jz, gain) {spearman(jz, y):+.2f}; Spearman(scale, Jz) {spearman(x, jz):+.2f}")
    print(f"  Jz end/start: median of per-run ratios {np.median(np.array(jz_by[e0]) / jz_by[s0][0]):.2f} "
          f"[{min(jz_by[e0]) / jz_by[s0][0]:.2f}..{max(jz_by[e0]) / jz_by[s0][0]:.2f}]; ratio of medians "
          f"{np.median(jz_by[e0]) / jz_by[s0][0]:.2f}")


def curve(net, norm, X):
    """median over states of the raw action-14 mean at each theta."""
    return np.median(np.stack([sweep_one(net, norm, s)[1] for s in X]), axis=0)


def part_b(states, lin, ends):
    s0, e0 = SE[lin]
    print(f"\n=== B. static decomposition on my seed-3042 states, {lin} ({len(ends)} ends)")
    if lin == "s42":
        XS = states["ck__start_s42__3042"]
    else:
        XS = None
        k = [k for k in states if k.startswith("own__start_s80__3042")]
        XS = states[k[0]] if k else None
    netS, nS = Net(START[lin], s0), Norm(START[lin], s0)
    rows = []
    for r in ends:
        if lin == "s42":
            XE = states[f"ck__{r}@{e0}__3042"]
        else:
            k = f"own__{r}__3042"
            if k not in states:
                continue
            XE = states[k]
        netE, nE = Net(r, e0), Norm(r, e0)
        d = {"run": r}
        for tag, X in (("E", XE), ("S", XS)):
            if X is None:
                continue
            d[f"ee{tag}"] = gain(netE, nE, X)
            d[f"es{tag}"] = gain(netE, nE.mixed(nS), X)
            d[f"se{tag}"] = gain(netS, nS.mixed(nE), X)
            d[f"ss{tag}"] = gain(netS, nS, X)
        d["e_t3q_S"] = gain(netE, nE.mixed(nS, dims=(6,)), XE)
        d["e_tail_S"] = gain(netE, nE.mixed(nS, dims=TAIL8), XE)
        d["e_var_S"] = gain(netE, nE.mixed(nS, dims=(6,), part="var"), XE)
        d["e_mean_S"] = gain(netE, nE.mixed(nS, dims=(6,), part="mean"), XE)
        if XS is not None:
            d["s_var_E"] = gain(netS, nS.mixed(nE, dims=(6,), part="var"), XS)
            d["s_mean_E"] = gain(netS, nS.mixed(nE, dims=(6,), part="mean"), XS)
            d["s_both_E"] = gain(netS, nS.mixed(nE, dims=(6,)), XS)
        for L in (0, 1, 2):
            d[f"graftS{L}_Nend"] = gain(netE.graft(netS, L), nE, XE)
            d[f"graftE{L}_Nend"] = gain(netS.graft(netE, L), nE, XE)
        w6e, w6s = netE.L[0][0][:, 6].double().numpy(), netS.L[0][0][:, 6].double().numpy()
        d["w6_ratio"] = float(np.linalg.norm(w6e) / np.linalg.norm(w6s))
        d["w6_cos"] = float(w6e @ w6s / np.linalg.norm(w6e) / np.linalg.norm(w6s))
        d["w6_norm"] = (float(np.linalg.norm(w6s)), float(np.linalg.norm(w6e)))
        d["scale"] = (nS.scale_deg(), nE.scale_deg())
        c_own, c_sw = curve(netE, nE, XE), curve(netE, nE.mixed(nS, dims=(6,)), XE)
        d["lin_own"] = (float(np.polyfit(THETAS[:4], c_own[:4], 1)[0]), float(np.polyfit(THETAS[3:], c_own[3:], 1)[0]))
        d["lin_sw"] = (float(np.polyfit(THETAS[:4], c_sw[:4], 1)[0]), float(np.polyfit(THETAS[3:], c_sw[3:], 1)[0]))
        d["zrange_own"] = ((0 - nE.mean[6]) / math.sqrt(nE.var[6] + nE.eps), (math.radians(6) - nE.mean[6]) / math.sqrt(nE.var[6] + nE.eps))
        d["zrange_S"] = ((0 - nS.mean[6]) / math.sqrt(nS.var[6] + nS.eps), (math.radians(6) - nS.mean[6]) / math.sqrt(nS.var[6] + nS.eps))
        rows.append(d)
    med = lambda k: float(np.median([d[k] for d in rows]))  # noqa: E731
    print("  run          own(ee)  endW,N_S  startW,N_E  startW,N_S | t3q_S  tail8_S  var_S  mean_S | w6 ratio  cos")
    for d in rows:
        print(f"  {d['run']}  {d['eeE']:+.3f}   {d['esE']:+.3f}    {d['seE']:+.3f}     {d['ssE']:+.3f}   | {d['e_t3q_S']:+.3f} "
              f"{d['e_tail_S']:+.3f} {d['e_var_S']:+.3f} {d['e_mean_S']:+.3f} | {d['w6_ratio']:.3f}  {d['w6_cos']:.4f}")
    for tag, name in (("E", "X_end"), ("S", "X_start")):
        if f"ee{tag}" not in rows[0]:
            continue
        ee, es, se, ss = (np.array([d[f"{k}{tag}"] for d in rows]) for k in ("ee", "es", "se", "ss"))
        tot = ee - ss
        nE_, nS_ = ee - es, se - ss
        wS_, wE_ = es - ss, ee - se
        share = 0.5 * (nE_ + nS_) / tot
        lshare = 0.5 * (np.log(ee / es) + np.log(se / ss)) / np.log(ee / ss) if np.all(np.sign(np.r_[ee, es, se, ss]) > 0) else None
        print(f"  2x2 on {name}: G(We,Ne) {np.median(ee):+.3f}  G(We,Ns) {np.median(es):+.3f}  G(Ws,Ne) {np.median(se):+.3f}  "
              f"G(Ws,Ns) {np.median(ss):+.3f}")
        print(f"     normaliser part: at end W {np.median(nE_):+.3f} ({int((nE_ > 0).sum())}/{len(rows)} >0), at start W "
              f"{np.median(nS_):+.3f} ({int((nS_ > 0).sum())}/{len(rows)} >0); weight part: at N_S {np.median(wS_):+.3f} "
              f"({int((wS_ > 0).sum())}/{len(rows)} >0), at N_E {np.median(wE_):+.3f} ({int((wE_ > 0).sum())}/{len(rows)} >0)")
        print(f"     symmetric normaliser share, median over runs {np.median(share):+.2f} [{share.min():+.2f}..{share.max():+.2f}]"
              + (f"; log-scale share {np.median(lshare):+.2f}" if lshare is not None else "")
              + f"; interaction G(We,Ne)-G(We,Ns)-G(Ws,Ne)+G(Ws,Ns) median {np.median(ee - es - se + ss):+.3f}")
    print(f"  end W on X_end: own {med('eeE'):+.3f}; tail_3 both from S {med('e_t3q_S'):+.3f}; tail 8 from S {med('e_tail_S'):+.3f}; "
          f"var only {med('e_var_S'):+.3f}; mean only {med('e_mean_S'):+.3f}")
    if "s_var_E" in rows[0]:
        print(f"  start W on X_S: own {med('ssS'):+.3f}; end tail_3 var only {med('s_var_E'):+.3f}; mean only {med('s_mean_E'):+.3f}; "
              f"both {med('s_both_E'):+.3f}")
    vm = np.array([d["e_t3q_S"] - d["e_var_S"] for d in rows])
    mo = np.array([d["e_mean_S"] - d["eeE"] for d in rows])
    vo = np.array([d["e_var_S"] - d["eeE"] for d in rows])
    vb = np.array([d["e_t3q_S"] - d["e_mean_S"] for d in rows])
    print(f"  mean effect: at end variance {np.median(mo):+.3f}, at start variance {np.median(vm):+.3f}; variance effect: "
          f"at end mean {np.median(vo):+.3f}, at start mean {np.median(vb):+.3f}")
    print(f"  variance swap ratio var_S/own {np.median([d['e_var_S'] / d['eeE'] for d in rows]):.2f} vs scale ratio "
          f"{np.median([d['scale'][0] / d['scale'][1] for d in rows]):.2f} (linear expectation)")
    print(f"  grafts with end stats on X_end: start L0 into end {med('graftS0_Nend') - med('eeE'):+.3f} "
          f"({sum(d['graftS0_Nend'] < d['eeE'] for d in rows)}/{len(rows)} lower), start L2 {med('graftS1_Nend') - med('eeE'):+.3f} "
          f"({sum(d['graftS1_Nend'] < d['eeE'] for d in rows)}/{len(rows)}), start OUT {med('graftS2_Nend') - med('eeE'):+.3f} "
          f"({sum(d['graftS2_Nend'] < d['eeE'] for d in rows)}/{len(rows)}); medians {med('graftS0_Nend'):+.3f} / "
          f"{med('graftS1_Nend'):+.3f} / {med('graftS2_Nend'):+.3f}")
    print(f"  end layer alone into start (end stats): L0 {med('graftE0_Nend'):+.3f}, L2 {med('graftE1_Nend'):+.3f}, "
          f"OUT {med('graftE2_Nend'):+.3f} (start W with end stats {med('seE'):+.3f})")
    print(f"  first-layer tail_3 column: ||w6|| start {rows[0]['w6_norm'][0]:.3f}, end median "
          f"{np.median([d['w6_norm'][1] for d in rows]):.3f}; ratio [{min(d['w6_ratio'] for d in rows):.3f}..{max(d['w6_ratio'] for d in rows):.3f}]; "
          f"cos [{min(d['w6_cos'] for d in rows):.4f}..{max(d['w6_cos'] for d in rows):.4f}]")
    print(f"  sweep linearity (raw action-14 slope per deg, 0-3 deg vs 3-6 deg, medians): own stats "
          f"{np.median([d['lin_own'][0] for d in rows]):+.4f} / {np.median([d['lin_own'][1] for d in rows]):+.4f}; "
          f"tail_3 stats from S {np.median([d['lin_sw'][0] for d in rows]):+.4f} / {np.median([d['lin_sw'][1] for d in rows]):+.4f}")
    print(f"  z range of the 0-6 deg sweep: own end stats {np.median([d['zrange_own'][0] for d in rows]):+.2f}..."
          f"{np.median([d['zrange_own'][1] for d in rows]):+.2f}; start stats {rows[0]['zrange_S'][0]:+.2f}..{rows[0]['zrange_S'][1]:+.2f}")
    return rows


def part_c():
    print("\n=== C. closed loop: my v_swaps.jsonl vs the probe's swaps/episodes.jsonl")
    mine = [json.loads(l) for l in open(HERE / "v_swaps.jsonl")]
    probe = {}
    for l in open(PROBE / "swaps" / "episodes.jsonl"):
        r = json.loads(l)
        probe[(r["cond"], r["who"], r["seed"])] = r
    n = neq = 0
    dmax = {"return": 0.0, "own14": 0.0, "tail3": 0.0, "gain": 0.0}
    for r in mine:
        q = probe.get((r["cond"], r["who"], r["seed"]))
        if q is None:
            continue
        n += 1
        d = {"return": abs(r["return"] - q["return"]), "own14": abs(r["own14_5_10"] - q["own14_mean_5_10"]),
             "tail3": abs(r["tail3_deg_5_10"] - q["tail3_deg_5_10"]), "gain": abs(r["gain"] - q["sweep"]["loop_gain"])}
        for k in d:
            dmax[k] = max(dmax[k], d[k])
        neq += int(d["return"] < 1e-6)
        assert (q["donor"] is None) == (r["donor"] is None) and (q["dims"] or None) == (r["dims"] or None), (r, q["donor"], q["dims"])
    print(f"  {n} episodes matched by (condition, policy, seed); return equal within 1e-6 in {neq}/{n}; max |diff|: "
          + ", ".join(f"{k} {v:.2e}" for k, v in dmax.items()))
    print(f"  in-loop z6 check: max over all episodes |z6 - formula(condition stats)| {max(r['z6_formula_maxdiff'] for r in mine):.2e}")
    by = defaultdict(dict)
    for r in mine:
        by[(r["cond"], r["who"])][r["seed"]] = r
    base_start = {"s42": by[("own", "start_s42")], "s80": by[("own", "start_s80")]}
    print("  cond              who          gain   own14  rail  tail3   z6(cond) z6(own)  d return [paired]  up/dn  falls")
    for (cond, who), d in by.items():
        lin = who[-3:] if who.startswith("start") else who[:3]
        seeds = sorted(d)
        if cond.startswith("startW"):
            ref = base_start[lin]
        else:
            ref = by[("own", who)]
        dr = np.array([d[s]["return"] - ref[s]["return"] for s in seeds])
        print(f"  {cond:17s} {who:11s} {np.median([d[s]['gain'] for s in seeds]):+.3f}  {np.median([d[s]['own14_5_10'] for s in seeds]):+.3f}  "
              f"{np.median([d[s]['rail_5_10'] for s in seeds]):.2f}  {np.median([d[s]['tail3_deg_5_10'] for s in seeds]):+.2f}   "
              f"{np.median([d[s]['z6_mean'] for s in seeds]):+.2f}    {np.median([d[s]['z6_mean_ownstats'] for s in seeds]):+.2f}   "
              f"{np.median(dr):+7.1f}            {int((dr > 0).sum())}/{int((dr < 0).sum())}  {sum(d[s]['terminated'] for s in seeds)}")


def main():
    states = load_states("v_ckpts.states_*.npz")
    part_a(states)
    part_b(states, "s42", S42)
    if (HERE / "v_swaps.jsonl").exists():
        sw = load_states("v_swaps.states_*.npz")
        part_b(sw, "s80", ["s80_B_1421", "s80_A_1331"])
        part_c()
    return 0


if __name__ == "__main__":
    sys.exit(main())
