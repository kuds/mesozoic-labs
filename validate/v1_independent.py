"""V1, independently of run_bs.py's helpers: each arm's saved pair against the F2 start pair, tensor by tensor
(policy.pth, policy.optimizer.pth with torch.load weights_only) and array by array (the pkl's obs_rms / ret_rms, the
VecNormalize settings); the saved zip's data (num_timesteps, _total_timesteps, gae_lambda, ent_coef).  Reads only."""
import io
import json
import pickle
import sys
import zipfile

import numpy as np
import torch

F = ("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2/"
     "stance6_r10LTSB_s42/copies/01_stance/models/")
V = sys.argv[1]
ok = True


def members(z):
    with zipfile.ZipFile(z) as f:
        pol = torch.load(io.BytesIO(f.read("policy.pth")), map_location="cpu", weights_only=True)
        opt = torch.load(io.BytesIO(f.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
        data = json.loads(f.read("data"))
    return pol, opt, data


def flat_opt(o):
    out = {}
    for i, st in o["state"].items():
        for k, v in st.items():
            out[f"{i}.{k}"] = v
    return out, o["param_groups"]


p0, o0, d0 = members(F + "stage1_1048576_steps.zip.copy1")
with open(F + "stage1_vecnormalize_1048576_steps.pkl.copy1", "rb") as fh:
    v0 = pickle.load(fh)
for arm in ("control", "critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm"):
    run = f"{V}/{arm}_9101/saved/"
    p1, o1, d1 = members(run + "stage1_1048576_steps.zip")
    with open(run + "stage1_vecnormalize_1048576_steps.pkl", "rb") as fh:
        v1 = pickle.load(fh)
    eq_p = list(p0) == list(p1) and all(p0[k].dtype == p1[k].dtype and torch.equal(p0[k], p1[k]) for k in p0)
    s0, g0 = flat_opt(o0)
    s1, g1 = flat_opt(o1)
    eq_o = list(s0) == list(s1) and all(torch.equal(torch.as_tensor(s0[k]), torch.as_tensor(s1[k])) for k in s0)
    eq_g = json.dumps(g0, sort_keys=True, default=str) == json.dumps(g1, sort_keys=True, default=str)
    eq_v = all(np.array_equal(getattr(getattr(v0, r), f), getattr(getattr(v1, r), f))
               and np.asarray(getattr(getattr(v0, r), f)).dtype == np.asarray(getattr(getattr(v1, r), f)).dtype
               for r in ("obs_rms", "ret_rms") for f in ("mean", "var", "count"))
    eq_s = all(getattr(v0, a) == getattr(v1, a) for a in ("clip_obs", "clip_reward", "gamma", "epsilon", "norm_obs",
                                                          "norm_reward"))
    cls = (type(v1.obs_rms).__name__, type(v1.ret_rms).__name__)
    data = {k: d1.get(k) for k in ("num_timesteps", "_total_timesteps", "gae_lambda", "ent_coef", "_n_updates")}
    good = eq_p and eq_o and eq_g and eq_v and eq_s and cls == ("RunningMeanStd", "RunningMeanStd") and \
        data["num_timesteps"] == 1048576 and data["_n_updates"] == d0.get("_n_updates")
    ok &= good
    print(f"{arm:20s} policy {len(p1)} tensors equal {eq_p}; Adam {len(s1)} entries equal {eq_o}, param_groups {eq_g}; "
          f"pkl arrays {eq_v}, settings {eq_s}, classes {cls}; data {data} -> {'PASS' if good else 'FAIL'}")
print("V1 INDEPENDENT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
