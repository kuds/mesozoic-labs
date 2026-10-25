"""Local mean-shift test under the policy's own noise (independent implementation, normalize_obs + distribution.mean).
tail_3 = clip(mu14 + delta + sigma14*z14), others clip(mu + sigma z); z ~ default_rng(seed) for all 15 per step.
delta=0 on the first NFREE seeds reproduces noisevalue's stoch_free bitwise check."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from common import setup
zp, pkl, out, nfree, nshift = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
J, ctx, model, norm = setup(zp, pkl)
import torch
pol = model.policy
sigma = np.exp(pol.log_std.detach().numpy().astype(np.float64))
env = ctx.env_class(**ctx.env_kwargs); low, high = env.action_space.low.astype(np.float64), env.action_space.high.astype(np.float64)
T3 = 14
def ep(seed, delta):
    rng = np.random.default_rng(seed); obs, _ = env.reset(seed=seed); ret = win = 0.0; L = 0; cmds = []
    while True:
        o = norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None]).astype(np.float32)
        with torch.no_grad():
            mu = pol.get_distribution(torch.as_tensor(o)).distribution.mean.numpy()[0].astype(np.float64)
        mu[T3] += delta
        act = np.clip(mu + sigma * rng.standard_normal(mu.shape), low, high)
        obs, r, term, trunc, info = env.step(act); ret += float(r)
        if L >= 200: win += float(r); cmds.append(act[T3])
        L += 1
        if term or trunc: break
    c = np.asarray(cmds)
    return {"seed": seed, "L": L, "ret": ret, "win": win, "cmd_mean": float(c.mean()), "frac99": float(np.mean(np.abs(c) >= 0.99))}
res = {"sigma_t3": float(sigma[T3])}
res["free"] = [ep(3042 + i, 0.0) for i in range(nfree)]; print("free done", flush=True)
for d in (+0.15, -0.15):
    res[f"shift{d:+.2f}"] = [ep(3042 + i, d) for i in range(nshift)]; print("shift", d, "done", flush=True)
json.dump(res, open(out, "x"), indent=1)
