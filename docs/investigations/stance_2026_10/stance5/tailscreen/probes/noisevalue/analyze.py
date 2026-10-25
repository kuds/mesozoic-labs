"""Reduce value_s42.json / value_s80.json: paired differences, paired bootstrap 95% CIs, judged-file cross-checks."""
import json
import sys

import numpy as np

HERE = "scratchpad2/tailround/probes/noisevalue"
JUDGED = {"s42": "scratchpad/foot/stance5/judged/stance5_LTSB_s42/judge_4194304.json",
          "s80": "scratchpad/foot/stance5/replication/judged/stance5_LTSB_s80/judge_4194304.json"}
B = 20000


def boot(d, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


for tag in ("s42", "s80"):
    D = json.load(open(f"{HERE}/value_{tag}.json"))
    jd = json.load(open(JUDGED[tag]))
    print(f"===== {tag} {D['label']} num_timesteps {D['num_timesteps']} sigma_t3 {D['sigma_t3']:.4f} "
          f"sigma mean {np.mean(D['sigma']):.4f}")
    print(f"  zip sha256 {D['zip_sha256'][:16]} judged {jd['model_sha256'][:23]}; vecnorm {D['vecnorm_sha256'][:16]} judged {jd['vecnorm_sha256'][:23]}")
    # cross-checks against the judged file
    pan = {e["seed"]: e["reward"] for e in jd["panel"]["episodes_detail"]}
    det = D["runs"]["det_free"]
    dd = [abs(r["return"] - pan[r["seed"]]) for r in det]
    print(f"  xcheck det_free return vs judged panel reward (20 seeds): max |diff| {max(dd):.3e}")
    own = {e["seed"]: e["reward"] for e in jd["noise_matched_own"]["episodes_detail"]}
    st = D["runs"]["stoch_free"]
    do = [abs(r["return"] - own[r["seed"]]) for r in st if r["seed"] in own]
    print(f"  xcheck stoch_free return vs judged noise_matched_own reward ({len(do)} seeds): max |diff| {max(do):.3e}; "
          f"judged own falls {jd['noise_matched_own']['falls']}")
    for mode in ("stoch", "det"):
        runs = {c: D["runs"][f"{mode}_{c}"] for c in ("free", "hold0", "hold09")}
        n = len(runs["free"])
        assert all([r["seed"] for r in runs[c]] == [r["seed"] for r in runs["free"]] for c in runs)
        print(f"  --- {mode} N={n}")
        for c, rows in runs.items():
            ret = np.array([r["return"] for r in rows])
            win = np.array([r["window_return"] for r in rows])
            L = np.array([r["length"] for r in rows])
            f99 = np.array([r["t3_frac99"] for r in rows if r["t3_frac99"] is not None])
            f90 = np.array([r["t3_frac90"] for r in rows if r["t3_frac90"] is not None])
            mu = np.array([r["t3_mu_mean"] for r in rows if r["t3_mu_mean"] is not None])
            mugt1 = np.array([r["t3_mu_gt1"] for r in rows if r["t3_mu_gt1"] is not None])
            q3 = np.array([r["q3_mean"] for r in rows if r["q3_mean"] is not None])
            cmd = np.array([r["t3_cmd_mean"] for r in rows if r["t3_cmd_mean"] is not None])
            falls = [r["seed"] for r in rows if r["fell"]]
            print(f"    {c:7s} return mean {ret.mean():8.2f} (sd {ret.std(ddof=1):6.2f})  window mean {win.mean():8.2f} "
                  f"(sd {win.std(ddof=1):6.2f})  falls {len(falls)} {falls}  length mean {L.mean():.1f} min {L.min()}  "
                  f"t3 frac99 {f99.mean():.4f} (eps>0.10: {int((f99 > 0.10).sum())}) frac90 {f90.mean():.4f} "
                  f"cmd {cmd.mean():+.4f} mu {mu.mean():+.4f} mu>1 {mugt1.mean():.4f} q3 {q3.mean():+.4f}")
        for c in ("hold0", "hold09"):
            for key in ("return", "window_return"):
                d = np.array([b[key] - a[key] for a, b in zip(runs["free"], runs[c], strict=True)])
                lo, hi = boot(d)
                print(f"    {c} - free {key:13s}: mean {d.mean():+8.2f}  95% CI [{lo:+8.2f}, {hi:+8.2f}]  "
                      f"median {np.median(d):+8.2f}  wins {int((d > 0).sum())}/{n}")
        # window per-step, full-horizon only both conditions, and per-term decomposition of hold0 - free
        for c in ("hold0", "hold09"):
            pairs = [(a, b) for a, b in zip(runs["free"], runs[c], strict=True) if not a["fell"] and not b["fell"]]
            keys = sorted(set().union(*[set(a["window_terms"]) | set(b["window_terms"]) for a, b in pairs]))
            diffs = {k: np.mean([b["window_terms"].get(k, 0.0) - a["window_terms"].get(k, 0.0) for a, b in pairs]) for k in keys}
            big = {k[7:]: round(v, 2) for k, v in sorted(diffs.items(), key=lambda kv: -abs(kv[1])) if abs(v) >= 0.5}
            print(f"    {c} - free window terms (both full-horizon, {len(pairs)} pairs, |diff|>=0.5): {big}")
        # free-condition window return vs t3 frac99 (within condition, across episodes)
        fr = runs["free"]
        x = np.array([r["t3_frac99"] for r in fr]); y = np.array([r["window_return"] for r in fr])
        d0 = np.array([b["window_return"] - a["window_return"] for a, b in zip(fr, runs["hold0"], strict=True)])
        if x.std() > 0:
            print(f"    free: corr(frac99, window) {np.corrcoef(x, y)[0, 1]:+.3f}; corr(frac99, hold0-free window) "
                  f"{np.corrcoef(x, d0)[0, 1]:+.3f}")
            hi = x > 0.10
            if hi.any() and (~hi).any():
                print(f"    hold0-free window: episodes frac99>0.10 (n={hi.sum()}) {d0[hi].mean():+.2f}; others (n={(~hi).sum()}) {d0[~hi].mean():+.2f}")
