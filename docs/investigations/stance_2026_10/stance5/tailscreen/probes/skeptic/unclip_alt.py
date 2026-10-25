"""Independent unclip check: no method overrides. ctrlrange[14] widened in memory to +-0.40, and the env is sent
a14 = mu14 * 0.2094/0.40 so ctrl14 = 0.2094*mu14 (filter is linear). Base = stock env + model.predict(normalize_obs)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from common import setup
zp, pkl, out = sys.argv[1:4]
J, ctx, model, norm = setup(zp, pkl)
import torch
pol = model.policy
def mu_of(obs):
    o = norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None]).astype(np.float32)
    with torch.no_grad():
        return pol.get_distribution(torch.as_tensor(o)).distribution.mean.numpy()[0].astype(np.float64), o
T3 = 14; HALF = 0.40; R0 = 0.2094
res = {}
for mode in ("base", "alt_unclip"):
    env = ctx.env_class(**ctx.env_kwargs); u = env.unwrapped
    assert abs(u.model.actuator_ctrlrange[T3, 1] - R0) < 1e-9 and u._home_ctrl[T3] == 0
    if mode == "alt_unclip":
        u.model.actuator_ctrlrange[T3] = [-HALF, HALF]
    q3adr = int(u.model.jnt_qposadr[u.model.joint("tail_3_pitch").id])
    rows = []
    for seed in range(3042, 3062):
        obs, _ = env.reset(seed=seed); mus, q3s, cmds, ctrls = [], [], [], []; ret = 0.0; L = 0
        while True:
            mu, o = mu_of(obs)
            if mode == "base":
                act = model.predict(o[0], deterministic=True)[0]
            else:
                act = np.clip(mu, -1, 1); act[T3] = mu[T3] * R0 / HALF
            obs, r, term, trunc, info = env.step(act)
            ret += float(r); L += 1
            mus.append(mu[T3]); q3s.append(float(u.data.qpos[q3adr])); cmds.append(float(np.clip(mu[T3], -1, 1))); ctrls.append(float(u.data.ctrl[T3]))
            if term or trunc: break
        mus, q3s, cmds = map(np.asarray, (mus, q3s, cmds)); w = slice(200, L)
        rows.append({"seed": seed, "L": L, "ret": ret, "frac99": float(np.mean(np.abs(cmds[w]) >= 0.99)),
                     "win_mu_mean": float(mus[w].mean()), "win_mu_max": float(mus[w].max()), "q3_max": float(q3s[w].max()),
                     "last100_mu": float(mus[-100:].mean()), "d_last100": float(mus[-50:].mean() - mus[-100:-50].mean()),
                     "ctrl_max": float(np.max(ctrls))})
    env.close()
    res[mode] = rows
    print(mode, "falls", sum(r["L"] < 1000 for r in rows), "ret", round(np.mean([r["ret"] for r in rows]), 2),
          "frac99", round(np.mean([r["frac99"] for r in rows]), 4), "pooled mu mean", round(np.mean([r["win_mu_mean"] for r in rows]), 4),
          "mu max", round(max(r["win_mu_max"] for r in rows), 4), "q3 max", round(max(r["q3_max"] for r in rows), 5), flush=True)
    for r in rows: print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(res, open(out, "x"), indent=1)
