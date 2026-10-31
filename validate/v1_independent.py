"""PS V1, independently of run_bs.py's helpers: each arm's saved pair against the F2 start pair (2,097,152), tensor by
tensor (policy.pth, policy.optimizer.pth with torch.load weights_only) and array by array (the pkl's obs_rms / ret_rms,
the VecNormalize settings); the saved zip's data against the start zip's.  control: everything equal (BS2's V1).
critic_phase: everything equal EXCEPT mlp_extractor.value_net.0.weight[:, 60:63], exactly zero in the saved zip (and
nonzero in the start zip), and the observation statistics' slots 60-62 (mean 0 / var 1 in the saved pkl; mean 0 /
var ~2.4e-11 in the start pkl); the actor's tensors (policy_net.*, action_net.*, log_std) bitwise equal; Adam moments
and step equal (the slots' moments zero in both).  Reads only.  Derived from BS2's validate/v1_independent.py.

    PY -I -B v1_independent.py V1_DIR SEED
"""
import hashlib
import io
import json
import pickle
import sys
import zipfile

import numpy as np
import torch

F = ("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2/"
     "stance6_r10LTSB_s42/copies/01_stance/models/")
ZIP0, PKL0 = F + "stage1_2097152_steps.zip.copy1", F + "stage1_vecnormalize_2097152_steps.pkl.copy1"
FMAN_SHA = {ZIP0: "83b6c88edb933005adcb2089bdac67654f4e77809ca5506f4e47364995ef1b61",      # FMAN line 241
            PKL0: "991d434f385187e87301755ddf9a12fe2b3adb65cf186e7e32b175b3574bed90"}      # FMAN line 243
S = 2_097_152
LO, HI = 60, 63
VNAME = "mlp_extractor.value_net.0.weight"
V, SEED = sys.argv[1], int(sys.argv[2])
ok = True
for p, want in FMAN_SHA.items():
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(f"start file {p.rsplit('/', 1)[1]} sha256 {got} == FMAN {got == want}")
    ok &= got == want


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


KEYS = ("num_timesteps", "_total_timesteps", "gae_lambda", "ent_coef", "_n_updates", "n_epochs", "batch_size",
        "n_steps", "gamma")
p0, o0, d0 = members(ZIP0)
with open(PKL0, "rb") as fh:
    v0 = pickle.load(fh)  # noqa: S301 - the pinned F2 file (sha256 checked above)
print(f"start zip data: { {k: d0.get(k) for k in KEYS} }")
names = list(p0)
vidx = names.index(VNAME)
pidx = names.index("mlp_extractor.policy_net.0.weight")
s0, g0 = flat_opt(o0)
print(f"start zip: {VNAME}[:, 60:63] max |w| {float(p0[VNAME][:, LO:HI].abs().max()):.6f}; Adam moments on the slots "
      f"(critic, actor) max |.| {float(s0[f'{vidx}.exp_avg'][:, LO:HI].abs().max())}, "
      f"{float(s0[f'{pidx}.exp_avg'][:, LO:HI].abs().max())}; pkl slots mean {v0.obs_rms.mean[LO:HI].tolist()} var "
      f"{v0.obs_rms.var[LO:HI].tolist()}")
for arm in ("control", "critic_phase"):
    phase = arm == "critic_phase"
    run = f"{V}/{arm}_{SEED}/saved/"
    p1, o1, d1 = members(run + f"stage1_{S}_steps.zip")
    with open(run + f"stage1_vecnormalize_{S}_steps.pkl", "rb") as fh:
        v1 = pickle.load(fh)  # noqa: S301 - written by this validation run
    same_keys = list(p0) == list(p1) and all(p0[k].dtype == p1[k].dtype and p0[k].shape == p1[k].shape for k in p0)
    differ = [k for k in p0 if not torch.equal(p0[k], p1[k])]
    actor = [k for k in p0 if k.startswith(("mlp_extractor.policy_net.", "action_net.")) or k == "log_std"]
    actor_eq = all(torch.equal(p0[k], p1[k]) for k in actor)
    if phase:
        keep = torch.ones(p0[VNAME].shape[1], dtype=torch.bool)
        keep[LO:HI] = False
        eq_p = (same_keys and differ == [VNAME] and torch.equal(p0[VNAME][:, keep], p1[VNAME][:, keep])
                and bool(torch.all(p1[VNAME][:, LO:HI] == 0.0)) and float(p0[VNAME][:, LO:HI].abs().max()) > 0.0)
    else:
        eq_p = same_keys and differ == []
    s1, g1 = flat_opt(o1)
    eq_o = list(s0) == list(s1) and all(torch.equal(torch.as_tensor(s0[k]), torch.as_tensor(s1[k])) for k in s0)
    eq_g = json.dumps(g0, sort_keys=True, default=str) == json.dumps(g1, sort_keys=True, default=str)
    mask = np.ones(len(v0.obs_rms.mean), dtype=bool)
    if phase:
        mask[LO:HI] = False
    eq_v = (all(np.array_equal(getattr(v0.ret_rms, f), getattr(v1.ret_rms, f)) for f in ("mean", "var", "count"))
            and np.array_equal(v0.obs_rms.mean[mask], v1.obs_rms.mean[mask])
            and np.array_equal(v0.obs_rms.var[mask], v1.obs_rms.var[mask]) and v0.obs_rms.count == v1.obs_rms.count
            and v1.obs_rms.mean.dtype == v0.obs_rms.mean.dtype and v1.obs_rms.var.dtype == v0.obs_rms.var.dtype)
    slots = {"mean": v1.obs_rms.mean[LO:HI].tolist(), "var": v1.obs_rms.var[LO:HI].tolist()}
    eq_slots = slots == ({"mean": [0.0] * 3, "var": [1.0] * 3} if phase else
                         {"mean": v0.obs_rms.mean[LO:HI].tolist(), "var": v0.obs_rms.var[LO:HI].tolist()})
    eq_s = all(getattr(v0, a) == getattr(v1, a) for a in ("clip_obs", "clip_reward", "gamma", "epsilon", "norm_obs",
                                                          "norm_reward"))
    cls = (type(v1.obs_rms).__name__, type(v1.ret_rms).__name__)
    data = {k: d1.get(k) for k in KEYS}
    good = (eq_p and actor_eq and eq_o and eq_g and eq_v and eq_slots and eq_s
            and cls == ("RunningMeanStd", "RunningMeanStd") and data["num_timesteps"] == S == d0["num_timesteps"]
            and data["_n_updates"] == d0["_n_updates"] and data["_total_timesteps"] == d0["_total_timesteps"]
            == 12_000_000 and data["gae_lambda"] == 0.95 and data["ent_coef"] == 0.005)
    ok &= good
    print(f"{arm:13s} policy {len(p1)} tensors: differing {differ}; as expected {eq_p}; actor {len(actor)} tensors "
          f"bitwise {actor_eq}; Adam {len(s1)} entries equal {eq_o}, param_groups {eq_g}; pkl arrays (slots aside "
          f"for critic_phase) {eq_v}, slots {slots} as expected {eq_slots}, settings {eq_s}, classes {cls}; data "
          f"{data} -> {'PASS' if good else 'FAIL'}")
print("V1 INDEPENDENT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
