"""Reduce roll_unclip.py outputs.  python -I analyze.py PREFIX [PREFIX ...]"""
import json
import sys

import numpy as np

WIN0 = 200
A_STRAIGHT = 1.073


def q(x, p):
    return float(np.quantile(x, p))


def ep_stats(z, mode, i, L):
    w = slice(WIN0, L)
    mu, q3, f, ctrl, jl, act = (z[f"{mode}_{i}_{k}"] for k in ("mu", "q3", "force", "ctrl", "jlim", "act"))
    muw, q3w = mu[w], q3[w]
    quarters = [float(mu[a:min(a + 200, L)].mean()) if L > a else None for a in (200, 400, 600, 800)]
    q3q = [float(q3[a:min(a + 200, L)].mean()) if L > a else None for a in (200, 400, 600, 800)]
    t = np.arange(WIN0, L)
    slope = float(np.polyfit(t, muw, 1)[0] * 100) if L > WIN0 + 10 else None
    first_ge1 = int(np.argmax(mu >= 1.0)) if (mu >= 1.0).any() else None
    first_ge107 = int(np.argmax(mu >= A_STRAIGHT)) if (mu >= A_STRAIGHT).any() else None
    return {"seed_idx": i, "L": int(L),
            "mu_mean": float(muw.mean()), "mu_q05": q(muw, .05), "mu_q50": q(muw, .5), "mu_q95": q(muw, .95),
            "mu_max": float(muw.max()), "mu_last100": float(mu[max(WIN0, L - 100):L].mean()),
            "mu_quarters": quarters, "mu_slope_per100": slope,
            "mu_ge1": float(np.mean(muw >= 1.0)), "mu_ge099": float(np.mean(muw >= 0.99)),
            "mu_ge1073": float(np.mean(muw >= A_STRAIGHT)),
            "first_step_mu_ge1": first_ge1, "first_step_mu_ge1073": first_ge107,
            "act_frac99": float(np.mean(np.abs(np.clip(act[w], -1, 1)) >= 0.99)),
            "q3_mean": float(q3w.mean()), "q3_q05": q(q3w, .05), "q3_q50": q(q3w, .5), "q3_q95": q(q3w, .95),
            "q3_max": float(q3w.max()), "q3_last100": float(q3[max(WIN0, L - 100):L].mean()), "q3_quarters": q3q,
            "q3_ge_m0005": float(np.mean(q3w >= -0.005)), "q3_abs_le_001": float(np.mean(np.abs(q3w) <= 0.01)),
            "q3_ge0": float(np.mean(q3w >= 0.0)),
            "force_mean": float(f[w].mean()), "force_max": float(f[w].max()),
            "ctrl_mean": float(ctrl[w].mean()), "ctrl_max": float(ctrl[w].max()),
            "jlim_nonzero_steps": int(np.sum(np.abs(jl[w]) > 1e-9))}


def main():
    for pre in sys.argv[1:]:
        d = json.load(open(pre + ".json"))
        z = np.load(pre + ".npz")
        t3 = d["t3_index"]
        print(f"\n######## {d['label']}  num_timesteps {d['num_timesteps']}  half_range {d['half_range']}  "
              f"zip {d['zip_sha256'][:12]} pkl {d['vecnorm_sha256'][:12]}")
        # reproduction checks
        same = True
        for i in range(20):
            for k in ("mu", "act", "ctrl", "q3", "force", "jlim", "rew"):
                a, b = z[f"base_{i}_{k}"], z[f"wrapclip_{i}_{k}"]
                if a.shape != b.shape or not np.array_equal(a, b):
                    same = False
                    print("  MISMATCH base vs wrapclip", i, k)
        rb, rw = d["runs"]["base"], d["runs"]["wrapclip"]
        same &= all(x["return"] == y["return"] and x["length"] == y["length"] for x, y in zip(rb, rw))
        print(f"  wrapclip == base bitwise over 20 episodes (mu, act, ctrl, q3, force, constraint, reward, L, return): {same}")
        print(f"  base predict vs clip(mu) mismatched steps: {sum(r['base_predict_mismatch_steps'] for r in rb)}")
        stats = {}
        for mode in ("base", "unclip"):
            rows = d["runs"][mode]
            S = [ep_stats(z, mode, i, r["length"]) for i, r in enumerate(rows)]
            stats[mode] = S
            f99 = np.mean([r["frac99_all"][t3] for r in rows])
            others = max(max(v for j, v in enumerate(r["frac99_all"]) if j != t3) for r in rows)
            flagged = sum(max(r["frac99_all"]) > 0.10 for r in rows)
            print(f"\n  == {mode}: falls {sum(r['fell'] for r in rows)}  lengths {sorted(set(r['length'] for r in rows))}"
                  f"  t3 frac99 of clipped cmd (episode mean) {f99:.4f}  flagged(>0.10 any act) {flagged}/20"
                  f"  other-actuator max frac99 {others:.4f}  return mean {np.mean([r['return'] for r in rows]):.1f}"
                  f"  window return mean {np.mean([r['window_return'] for r in rows]):.1f}")
            # pooled over window steps
            mu = np.concatenate([z[f"{mode}_{i}_mu"][WIN0:r["length"]] for i, r in enumerate(rows)])
            q3 = np.concatenate([z[f"{mode}_{i}_q3"][WIN0:r["length"]] for i, r in enumerate(rows)])
            fo = np.concatenate([z[f"{mode}_{i}_force"][WIN0:r["length"]] for i, r in enumerate(rows)])
            ct = np.concatenate([z[f"{mode}_{i}_ctrl"][WIN0:r["length"]] for i, r in enumerate(rows)])
            print(f"  pooled window ({len(mu)} steps): mu mean {mu.mean():.4f} q05 {q(mu,.05):.4f} q25 {q(mu,.25):.4f} "
                  f"q50 {q(mu,.5):.4f} q75 {q(mu,.75):.4f} q95 {q(mu,.95):.4f} q99 {q(mu,.99):.4f} max {mu.max():.4f}; "
                  f"share >=0.99 {np.mean(mu>=0.99):.4f} >=1 {np.mean(mu>=1):.4f} >=1.073 {np.mean(mu>=1.073):.4f}")
            print(f"     q3 mean {q3.mean():+.4f} q05 {q(q3,.05):+.4f} q50 {q(q3,.5):+.4f} q95 {q(q3,.95):+.4f} "
                  f"max {q3.max():+.4f}; share q3>=-0.005 {np.mean(q3>=-0.005):.4f} q3>=0 {np.mean(q3>=0):.4f}"
                  f"  |q3|<=0.01 {np.mean(np.abs(q3)<=0.01):.4f}")
            print(f"     actuator force mean {fo.mean():+.3f} q05 {q(fo,.05):+.3f} q95 {q(fo,.95):+.3f} max {fo.max():+.3f} N.m;"
                  f" ctrl mean {ct.mean():+.4f} max {ct.max():+.4f} rad; ctrl>=0.3999 share {np.mean(ct>=0.3999):.4f}")
            qm = np.array([s["mu_quarters"] for s in S], dtype=float)
            q3m = np.array([s["q3_quarters"] for s in S], dtype=float)
            print("     episode-mean of mu by window quarter (200-399,400-599,600-799,800-999):",
                  np.round(np.nanmean(qm, axis=0), 4).tolist(), " q3:", np.round(np.nanmean(q3m, axis=0), 4).tolist())
            sl = [s["mu_slope_per100"] for s in S]
            print(f"     per-episode mu slope per 100 steps over window: median {np.median(sl):+.4f} "
                  f"min {min(sl):+.4f} max {max(sl):+.4f}; positive in {sum(x > 0 for x in sl)}/20")
            print("     ep  L    mu_mean  q05    q50    q95    max    last100 slope  >=1    >=1.073 1st>=1 1st>=1.073"
                  "  q3_mean  q3_q50  q3_max  q3last100 q3>=-.005 F_mean F_max  ctrlmax jlim")
            for s, r in zip(S, rows):
                print(f"     {s['seed_idx']:2d} {s['L']:4d} {s['mu_mean']:.4f} {s['mu_q05']:.4f} {s['mu_q50']:.4f} "
                      f"{s['mu_q95']:.4f} {s['mu_max']:.4f} {s['mu_last100']:.4f} {s['mu_slope_per100']:+.4f} "
                      f"{s['mu_ge1']:.3f} {s['mu_ge1073']:.3f}  {str(s['first_step_mu_ge1']):>5} {str(s['first_step_mu_ge1073']):>5}"
                      f"   {s['q3_mean']:+.4f} {s['q3_q50']:+.4f} {s['q3_max']:+.4f} {s['q3_last100']:+.4f}  "
                      f"{s['q3_ge_m0005']:.3f}   {s['force_mean']:+.2f} {s['force_max']:+.2f} {s['ctrl_max']:+.4f} "
                      f"{s['jlim_nonzero_steps']}  {'FELL ' + r['termination_reason'] if r['fell'] else ''}")
        json.dump(stats, open(pre + "_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
