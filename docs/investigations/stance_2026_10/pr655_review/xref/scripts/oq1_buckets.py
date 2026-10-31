"""Critic error by episode step in PR #655's Open Question 1 buckets, from the why-probe's existing rollouts.

Read only: loads the phase probe's rollouts/<label>.npz (and verify_why's fresh runs) and recomputes the
return-to-go G exactly as phase_analyze.py does (normalised reward, gamma 0.98, SB3's truncation bootstrap
V(terminal obs) at the 1,000-step limit). No simulation, no network, no training.

Outputs (stdout and the JSON path given as argv[1]):
  per label and per bucket (0-50, 50-150, 150-200, 200-250, 250-1000; the note's OQ1 buckets) and per 25-step bin
  in 0-300: mean V, mean G, mean V - G (normalised, the critic's units) with an episode-clustered bootstrap CI,
  and the same converted to raw reward units with the checkpoint's recorded sigma_r = sqrt(ret_rms.var + eps).
  Also the gated terms' own return-to-go (all four, and stance_width alone, i.e. the only gated term on main)
  by bucket, and the raw per-step means of each gated term over settled states.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

S2 = Path(os.environ["XREF_S2"])  # the earlier session scratchpad (the run log's S2), read only
PP = S2 / "com_revisit/notes/phase_probe/rollouts"
VW = S2 / "com_revisit/notes/verify_why/runs"
SOURCES = [(lab, PP / f"{lab}.npz") for lab in ("P1", "P2", "C11a", "C11E", "C31a", "C31E", "B11E", "N80")]
SOURCES += [("FR1_C11E_stoch", VW / "FR1_C11E_stoch.npz"), ("FR3_P2_stoch", VW / "FR3_P2_stoch.npz"),
            ("FR2_C11E_det", VW / "FR2_C11E_det.npz")]
GATED = ("reward_stance_width", "reward_lateral_lean", "reward_stance_stagger", "reward_stance_cop_band")
BUCKETS = [(0, 50), (50, 150), (150, 200), (200, 250), (250, 1000)]
FINE = [(a, a + 25) for a in range(0, 300, 25)]
NBOOT = 400
SEED = 20261031


def returns(rn, ep, k, ep_trunc, ep_vterm, gamma):
    G = np.zeros(len(rn))
    for j in range(len(ep_trunc)):
        r = np.flatnonzero(ep == j)
        assert np.array_equal(k[r], np.arange(len(r)))
        acc = ep_vterm[j] if ep_trunc[j] else 0.0
        for t in range(len(r) - 1, -1, -1):
            acc = rn[r[t]] + gamma * acc
            G[r[t]] = acc
    return G


def plain_returns(x, ep, k, n_ep, gamma):
    out = np.zeros(len(x))
    for j in range(n_ep):
        r = np.flatnonzero(ep == j)
        acc = 0.0
        for t in range(len(r) - 1, -1, -1):
            acc = x[r[t]] + gamma * acc
            out[r[t]] = acc
    return out


def bucket_stats(vals, ep, k, n_ep, lo, hi, rng_w):
    sel = (k >= lo) & (k < hi)
    per_ep = np.full(n_ep, np.nan)
    for j in range(n_ep):
        m = sel & (ep == j)
        if m.any():
            per_ep[j] = vals[m].mean()
    ok = np.isfinite(per_ep)
    pe = per_ep[ok]
    boots = []
    for w in rng_w:
        ww = w[ok]
        boots.append(float((pe * ww).sum() / ww.sum()))
    return float(pe.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    out_path = Path(sys.argv[1])
    res = {}
    for label, path in SOURCES:
        if not path.exists():
            print(f"{label}: missing {path}")
            continue
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        gamma = float(meta["gamma"])
        sigma = float(meta["sigma_r"])
        ep = z["ep"].astype(np.int64)
        k = z["k"].astype(np.int64)
        V = z["V"].astype(np.float64)
        fresh = "r_norm" not in z.files
        rn = (z["rn"] if fresh else z["r_norm"]).astype(np.float64)
        terms = (z["info"] if fresh else z["terms"]).astype(np.float64)
        keys = [str(x) for x in (z["info_keys"] if fresh else z["reward_keys"])]
        ep_trunc = z["ep_trunc"].astype(bool)
        ep_vterm = z["ep_vterm"].astype(np.float64)
        n_ep = len(ep_trunc)
        G = returns(rn, ep, k, ep_trunc, ep_vterm, gamma)
        gate_all = terms[:, [keys.index(g) for g in GATED]].sum(1) / sigma
        gate_w = terms[:, keys.index("reward_stance_width")] / sigma
        Gg = plain_returns(gate_all, ep, k, n_ep, gamma)
        Gw = plain_returns(gate_w, ep, k, n_ep, gamma)
        rng = np.random.default_rng(SEED)
        W = [np.bincount(rng.integers(0, n_ep, n_ep), minlength=n_ep).astype(float) for _ in range(NBOOT)]
        settled = k >= 200
        raw_gated = {g: float(terms[settled, keys.index(g)].mean()) for g in GATED}
        r = {"sigma_r": sigma, "n_episodes": n_ep, "num_timesteps": meta.get("num_timesteps"),
             "blind": meta.get("blind"), "deterministic": "det" in label,
             "raw_gated_per_settled_step": raw_gated,
             "raw_gated_sum": float(sum(raw_gated.values())),
             "buckets": {}, "fine": {}}
        for lo, hi in BUCKETS:
            m_vg, ci_vg = bucket_stats(V - G, ep, k, n_ep, lo, hi, W)
            r["buckets"][f"{lo}-{hi}"] = {
                "V": bucket_stats(V, ep, k, n_ep, lo, hi, W)[0],
                "G": bucket_stats(G, ep, k, n_ep, lo, hi, W)[0],
                "V_minus_G": m_vg, "V_minus_G_ci": ci_vg,
                "V_minus_G_raw": m_vg * sigma, "V_minus_G_raw_ci": [c * sigma for c in ci_vg],
                "G_gate4": bucket_stats(Gg, ep, k, n_ep, lo, hi, W)[0],
                "G_width_only": bucket_stats(Gw, ep, k, n_ep, lo, hi, W)[0],
            }
        for lo, hi in FINE:
            m_vg, ci_vg = bucket_stats(V - G, ep, k, n_ep, lo, hi, W)
            r["fine"][f"{lo}-{hi}"] = {"V_minus_G": m_vg, "V_minus_G_ci": ci_vg, "V_minus_G_raw": m_vg * sigma,
                                       "G_gate4": bucket_stats(Gg, ep, k, n_ep, lo, hi, W)[0]}
        if fresh and "buf_advantages" in z.files:
            # SB3's own GAE(0.98, 0.95) advantages of the rollout buffer (cut at the 2,048-step rollout ends),
            # by the step index k of the state they belong to; pooled over states (no episode clustering).
            A = z["buf_advantages"].astype(np.float64).ravel()
            kk = z["env_k"].astype(np.int64).ravel()
            r["sb3_adv_by_bucket"] = {f"{lo}-{hi}": float(A[(kk >= lo) & (kk < hi)].mean()) for lo, hi in BUCKETS}
            r["sb3_adv_by_bucket_raw"] = {b: v * sigma for b, v in r["sb3_adv_by_bucket"].items()}
        res[label] = r
        print(f"=== {label} t={r['num_timesteps']} sigma_r {sigma:.3f} eps {n_ep} blind {r['blind']}  raw gated per "
              f"settled step: " + ", ".join(f"{g[7:]} {v:+.4f}" for g, v in raw_gated.items())
              + f"  sum {r['raw_gated_sum']:+.4f}")
        for b, d in r["buckets"].items():
            print(f"  {b:>9}: V-G {d['V_minus_G']:+.4f} [{d['V_minus_G_ci'][0]:+.4f},{d['V_minus_G_ci'][1]:+.4f}] "
                  f"norm = {d['V_minus_G_raw']:+.2f} raw [{d['V_minus_G_raw_ci'][0]:+.2f},"
                  f"{d['V_minus_G_raw_ci'][1]:+.2f}]   G_gate4 {d['G_gate4']:+.4f} G_width {d['G_width_only']:+.4f}")
        print("  fine V-G raw: " + "  ".join(f"{b}:{d['V_minus_G_raw']:+.2f}" for b, d in r["fine"].items()))
        if "sb3_adv_by_bucket" in r:
            print("  SB3 GAE advantage by bucket (norm / raw): " + "  ".join(
                f"{b}:{v:+.4f}/{v * sigma:+.2f}" for b, v in r["sb3_adv_by_bucket"].items()))
    out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
