"""Independent start-pair check (check_build).  Reads only.  Does not import the BS2 tools.

argv: F2_ZIP F2_PKL PROBE_NPZ V1_ROOT
Loads the F2 zip (zipfile + torch.load weights_only) and the F2 pkl (pickle; SB3/numpy/gymnasium classes only),
recomputes the parameter / Adam / VecNormalize digests with the scheme the harness records (sha256 over
name + dtype + shape + bytes, sorted keys), compares them with every V1 arm's run_meta, compares tensor by tensor with
each V1 arm's saved zip/pkl, compares the pkl statistics with the parent's probe row whose update_timesteps == S, and
recomputes the deterministic bank actions / values from the raw bank observations with a plain-torch forward.
"""
import hashlib
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

F2_ZIP, F2_PKL, PROBE, V1 = sys.argv[1:5]
S = 2_097_152
ARMS = ("control", "critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm")


def h(arrays):
    d = hashlib.sha256()
    for name, a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        d.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return "sha256:" + d.hexdigest()


def load_zip(p):
    with zipfile.ZipFile(p) as z:
        data = json.loads(z.read("data"))
        pol = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
        opt = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
    return data, pol, opt


def params_arr(sd):
    return [(k, sd[k].detach().cpu().numpy()) for k in sorted(sd)]


def adam_arr(o):
    out = []
    for i, v in sorted(o.get("state", {}).items()):
        for n in sorted(v):
            x = v[n]
            out.append((f"{i}.{n}", np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)))
    return out


def vn_arr(vn):
    return [("obs.mean", vn.obs_rms.mean), ("obs.var", vn.obs_rms.var), ("obs.count", np.float64(vn.obs_rms.count)),
            ("ret.mean", vn.ret_rms.mean), ("ret.var", vn.ret_rms.var), ("ret.count", np.float64(vn.ret_rms.count))]


out = {}
data, pol, opt = load_zip(F2_ZIP)
with open(F2_PKL, "rb") as fh:
    vn = pickle.load(fh)  # noqa: S301 - the pinned F2 copy (sha256 checked separately)
mine = {"params": h(params_arr(pol)), "adam": h(adam_arr(opt)), "vecnormalize": h(vn_arr(vn)),
        "obs_rms": h(vn_arr(vn)[:3])}
out["mine_digests"] = mine
out["zip_data"] = {k: data[k] for k in ("num_timesteps", "_n_updates", "_total_timesteps", "ent_coef", "gae_lambda",
                                        "n_steps", "batch_size", "n_epochs", "gamma", "target_kl")}
out["opt_step"] = sorted({float(v["step"]) for v in opt["state"].values()})
out["param_groups_lr"] = [g.get("lr") for g in opt["param_groups"]]
out["tensor_names"] = {k: list(v.shape) for k, v in pol.items()}
out["few_tensors"] = {
    "log_std": [round(float(x), 7) for x in pol["log_std"]],
    "action_net.bias[14]": float(pol["action_net.bias"][14]),
    "value_net.bias": float(pol["value_net.bias"][0]),
    "value_net.weight[0,:4]": [float(x) for x in pol["value_net.weight"][0, :4]],
    "mlp_extractor.value_net.0.weight[:,6] sum|.|": float(pol["mlp_extractor.value_net.0.weight"][:, 6].abs().sum()),
    "mlp_extractor.policy_net.0.weight[0,:3]": [float(x) for x in pol["mlp_extractor.policy_net.0.weight"][0, :3]],
}
out["vecnorm"] = {"class": type(vn).__name__, "obs_rms_class": type(vn.obs_rms).__name__,
                  "ret_rms_class": type(vn.ret_rms).__name__, "obs_count": float(vn.obs_rms.count),
                  "ret_count": float(vn.ret_rms.count), "ret_var": float(np.asarray(vn.ret_rms.var).ravel()[0]),
                  "obs_mean[6]": float(vn.obs_rms.mean[6]), "obs_var[6]": float(vn.obs_rms.var[6]),
                  "clip_obs": float(vn.clip_obs), "clip_reward": float(vn.clip_reward), "epsilon": float(vn.epsilon),
                  "gamma": float(vn.gamma), "norm_obs": bool(vn.norm_obs), "norm_reward": bool(vn.norm_reward),
                  "training": bool(vn.training)}

# V1 run_meta digests and the V1 saved pair, every arm
cmp = {}
for arm in ARMS:
    rd = Path(V1) / f"{arm}_9201"
    m = json.loads((rd / "run_meta.json").read_text())
    ds = m["digest_start"]
    v1 = m["v1"]
    sd, spol, sopt = load_zip(rd / "saved" / f"stage1_{S}_steps.zip")
    with open(rd / "saved" / f"stage1_vecnormalize_{S}_steps.pkl", "rb") as fh:
        svn = pickle.load(fh)  # noqa: S301
    teq = all(torch.equal(pol[k], spol[k]) for k in pol) and set(pol) == set(spol)
    a1, a2 = adam_arr(opt), adam_arr(sopt)
    aeq = len(a1) == len(a2) and all(n1 == n2 and np.array_equal(x1, x2) for (n1, x1), (n2, x2) in zip(a1, a2))
    veq = all(np.array_equal(np.asarray(x1), np.asarray(x2)) for (_, x1), (_, x2) in zip(vn_arr(vn), vn_arr(svn)))
    cmp[arm] = {
        "run_meta_digest_start_equal_mine": {k: ds[k] == mine[k] for k in ("params", "adam", "vecnormalize",
                                                                            "obs_rms")},
        "v1_loaded_equal_mine": {"params": v1["loaded"]["params"] == mine["params"],
                                 "adam": v1["loaded"]["adam"] == mine["adam"],
                                 "vecnormalize": v1["loaded_vecnormalize"] == mine["vecnormalize"]},
        "saved_zip_tensors_equal_f2": teq, "saved_adam_equal_f2": aeq, "saved_pkl_stats_equal_f2": veq,
        "saved_pkl_classes": [type(svn.obs_rms).__name__, type(svn.ret_rms).__name__],
        "saved_num_timesteps": sd["num_timesteps"], "saved_n_updates": sd["_n_updates"],
        "n_updates_start": ds["n_updates"], "num_timesteps_start": ds["num_timesteps"],
        "ent_coef_after_load": m.get("ent_coef_after_load"), "gae_lambda_after_load": m.get("gae_lambda_after_load"),
        "bank_tail3": m["v1"]["bank"]["tail3_statue_settled"],
    }
out["v1_compare"] = cmp

# the parent's probe row at S
z = np.load(PROBE, allow_pickle=False)
ut = z["update_timesteps"]
rows = np.flatnonzero(ut == S)
out["probe_rows_at_S"] = rows.tolist()
r = int(rows[0])
out["probe_vs_pkl"] = {
    "obs_mean_equal": bool(np.array_equal(z["obs_rms_mean"][r].astype(np.float64), np.asarray(vn.obs_rms.mean))),
    "obs_mean_maxabs": float(np.max(np.abs(z["obs_rms_mean"][r].astype(np.float64) - vn.obs_rms.mean))),
    "obs_var_maxabs": float(np.max(np.abs(z["obs_rms_var"][r].astype(np.float64) - vn.obs_rms.var))),
    "obs_count": [float(z["obs_rms_count"][r]), float(vn.obs_rms.count)],
    "ret_var": [float(z["ret_rms_var"][r]), float(np.asarray(vn.ret_rms.var).ravel()[0])],
    "probe_dtypes": [str(z["obs_rms_mean"].dtype), str(z["obs_rms_var"].dtype)],
}

# plain-torch forward on the raw bank observations (SB3 default Tanh; net_arch [512, 256] for pi and vf)
raw = z["bank_raw_obs"].astype(np.float64)
nobs = np.clip((raw - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon), -vn.clip_obs, vn.clip_obs)
x = torch.as_tensor(nobs.astype(np.float32))


def mlp(prefix, t):
    t = torch.tanh(t @ pol[f"{prefix}.0.weight"].T + pol[f"{prefix}.0.bias"])
    return torch.tanh(t @ pol[f"{prefix}.2.weight"].T + pol[f"{prefix}.2.bias"])


with torch.no_grad():
    mu = mlp("mlp_extractor.policy_net", x) @ pol["action_net.weight"].T + pol["action_net.bias"]
    v = (mlp("mlp_extractor.value_net", x) @ pol["value_net.weight"].T + pol["value_net.bias"]).squeeze(-1)
    xb = x.clone()
    xb[:, 6] = 0.0
    vb = (mlp("mlp_extractor.value_net", xb) @ pol["value_net.weight"].T + pol["value_net.bias"]).squeeze(-1)
labels = [str(s) for s in z["bank_labels"]]
ss = [i for i, s in enumerate(labels) if s == "statue_settled"]
an = z["bank_action_now"][r]
out["bank_forward"] = {
    "statue_settled_rows": ss,
    "mine_tail3_mu": [float(mu[i, 14]) for i in ss],
    "probe_tail3": [float(an[i, 14]) for i in ss],
    "max_abs_mu_vs_probe_all": float(np.max(np.abs(mu.numpy() - an))),
    "max_abs_clipmu_vs_probe_all": float(np.max(np.abs(np.clip(mu.numpy(), -1, 1) - an))),
    "max_abs_value_vs_probe": float(np.max(np.abs(v.numpy() - z["bank_value_now"][r]))),
    "value_mine": [round(float(a), 6) for a in v],
    "value_blind_obs6_minus_plain": [round(float(a - b), 6) for a, b in zip(vb, v)],
}
print(json.dumps(out, indent=1))
