"""Track 6: VecNormalize obs statistics vs the reset pop; phase observability (pop, stance_width step-200 switch)."""
import json
import sys
import numpy as np
from multiprocessing import Pool

D = "scratchpad"
sys.path.insert(0, D)
FAM = {"qpos": slice(0, 21), "qvel": slice(21, 42), "gravity": slice(42, 45), "gyro": slice(45, 48),
       "linvel": slice(48, 51), "accel": slice(51, 54), "contact": slice(54, 56), "prey_dir": slice(56, 59),
       "prey_dist": slice(59, 60), "command": slice(60, 63)}


def job(args):
    sigma, seed = args
    from common import make_env
    global _ENV
    if "_ENV" not in globals():
        _ENV = make_env()
    env, u = _ENV
    rng = np.random.default_rng(seed + 777)
    obs, _ = env.reset(seed=seed)
    O, Rw, SW = [obs.copy()], [], []
    end = "truncated"
    for t in range(1000):
        a = np.clip(rng.normal(0.0, sigma, u.model.nu), -1, 1) if sigma > 0 else np.zeros(u.model.nu)
        obs, r, term, trunc, info = env.step(a.astype(np.float32))
        Rw.append(r)
        SW.append(info.get("reward_stance_width", 0.0))
        if term or trunc:
            end = info.get("termination_reason", "truncated") if term else "truncated"
            break
        O.append(obs.copy())
    return dict(sigma=sigma, seed=seed, obs=np.array(O), R=np.array(Rw), SW=np.array(SW), end=end)


def auc(pos, neg):
    x = np.concatenate([pos, neg]); r = x.argsort().argsort() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


if __name__ == "__main__":
    jobs = [(s, seed) for s in (0.0, 0.14, 0.22) for seed in range(100, 112)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=3)
    out = {}
    for sigma in (0.0, 0.14, 0.22):
        rs = [r for r in res if r["sigma"] == sigma]
        lens = [len(r["R"]) for r in rs]
        ends = [r["end"] for r in rs]
        X = np.concatenate([r["obs"] for r in rs]); T = np.concatenate([np.arange(len(r["obs"])) for r in rs])
        mu, sd = X.mean(0), X.std(0) + 1e-8
        quiet = T >= 200; pop = T < 50; mid = (T >= 100) & (T < 200)
        sdq = X[quiet].std(0) + 1e-8 if quiet.any() else np.full(X.shape[1], np.nan)
        Z = (X - mu) / sd
        fam = {}
        for f, sl in FAM.items():
            dev2 = ((X[:, sl] - mu[sl]) ** 2)
            share_pop = float(dev2[pop].sum() / max(dev2.sum(), 1e-12))
            fam[f] = dict(std_all_over_quiet=float(np.nanmedian(sd[sl] / sdq[sl])), pop_share_of_var=share_pop,
                          pop_absz_p99=float(np.percentile(np.abs(Z[pop][:, sl]), 99)), pop_frac_clip10=float((np.abs(Z[pop][:, sl]) > 10).mean()),
                          quiet_absz_p99=float(np.percentile(np.abs(Z[quiet][:, sl]), 99)) if quiet.any() else None,
                          all_frac_clip10=float((np.abs(Z[:, sl]) > 10).mean()))
        # phase observability: |normalized qvel| and full normalized-obs distance from the quiet centroid
        vel = np.linalg.norm(Z[:, FAM["qvel"]], axis=1)
        if quiet.any():
            cq = Z[quiet].mean(0)
            dist = np.linalg.norm(Z - cq, axis=1)
            obs_auc = dict(pop_vs_quiet_vel=auc(vel[pop], vel[quiet]), pop_vs_quiet_dist=auc(dist[pop], dist[quiet]),
                           s100_200_vs_quiet_vel=auc(vel[mid], vel[quiet]), s100_200_vs_quiet_dist=auc(dist[mid], dist[quiet]),
                           s150_200_vs_200_250_dist=auc(dist[(T >= 150) & (T < 200)], dist[(T >= 200) & (T < 250)]))
        else:
            obs_auc = None
        R = [r["R"] for r in rs if len(r["R"]) == 1000]
        sw = [r["SW"] for r in rs if len(r["SW"]) == 1000]
        out[str(sigma)] = dict(lens=lens, ends=ends, fam=fam, auc=obs_auc, n_pop_frac=float(pop.mean()),
                               rew_150_200=float(np.mean([x[150:200].mean() for x in R])) if R else None,
                               rew_200_250=float(np.mean([x[200:250].mean() for x in R])) if R else None,
                               sw_200_250=float(np.mean([x[200:250].mean() for x in sw])) if sw else None)
    json.dump(out, open(f"{D}/t6_vecnorm.json", "w"), indent=1)
    for s, o in out.items():
        print(f"\n=== sigma {s}: lens {o['lens']} ends {sorted(set(o['ends']))} pop-sample share {o['n_pop_frac']:.3f}")
        print("  reward/step 150-200 %.3f 200-250 %.3f (stance_width 200-250 %.3f)" % (o["rew_150_200"] or np.nan, o["rew_200_250"] or np.nan, o["sw_200_250"] or np.nan))
        for f, v in o["fam"].items():
            print(f"  {f:9s} std_all/std_quiet {v['std_all_over_quiet']:6.2f} pop share of var {v['pop_share_of_var']:.2f} pop |z|p99 {v['pop_absz_p99']:6.2f} pop clip>10 {v['pop_frac_clip10']:.4f} quiet |z|p99 {v['quiet_absz_p99'] if v['quiet_absz_p99'] is None else round(v['quiet_absz_p99'],2)} all clip {v['all_frac_clip10']:.5f}")
        print("  AUC", o["auc"])
