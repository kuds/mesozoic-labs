"""The critic-blind hook on a plainly built policy (check_build).  Reads only; no environment, no optimizer step.

argv: F2_ZIP BS_HOOKS_PY PARENT_PROBE PKL
Builds a stock SB3 ActorCriticPolicy (net_arch [512, 256], log_std_init -1.5) from the F2 start zip's state_dict,
then swaps its mlp_extractor class with bs_hooks.critic_blind_class exactly as install_arm does, and checks on the
probe bank (normalised with the start pkl) and random perturbations of observation 6: the four value paths do not
move, the actor's distribution is bitwise the unswapped one, and the blinded value equals the plain value of the
observation with index 6 set to 0.
"""
import importlib.util
import io
import json
import pickle
import sys
import zipfile

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

F2_ZIP, HOOKS, PROBE, PKL = sys.argv[1:5]
torch.manual_seed(0)
with zipfile.ZipFile(F2_ZIP) as z:
    sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
pol = ActorCriticPolicy(spaces.Box(-np.inf, np.inf, (63,), np.float32), spaces.Box(-1, 1, (15,), np.float32),
                        lambda _: 0.0, net_arch=[512, 256], log_std_init=-1.5)
pol.load_state_dict(sd)
pol.eval()
spec = importlib.util.spec_from_file_location("bs_hooks_check", HOOKS)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
with open(PKL, "rb") as fh:
    vn = pickle.load(fh)  # noqa: S301
raw = np.load(PROBE)["bank_raw_obs"].astype(np.float64)
obs = np.clip((raw - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon), -vn.clip_obs, vn.clip_obs)
x = torch.as_tensor(obs, dtype=torch.float32)
x = torch.cat([x, torch.randn(246, 63)])          # 256 rows: bank + random normalised observations
acts = torch.clamp(torch.randn(256, 15) * 0.3, -1, 1)


def paths(p, t):
    with torch.no_grad():
        v1 = p.predict_values(t).flatten()
        _, v2, _ = p.forward(t, deterministic=True)
        v3, lp, ent = p.evaluate_actions(t, acts)
        mu = p.get_distribution(t).distribution.mean
    return v1, v2.flatten(), v3.flatten(), lp, mu


def perturbed(t):
    out = []
    for kind in ("add1", "neg", "set3", "rand"):
        u = t.clone()
        if kind == "add1":
            u[:, 6] += 1.0
        elif kind == "neg":
            u[:, 6] = -u[:, 6]
        elif kind == "set3":
            u[:, 6] = 3.0
        else:
            u[:, 6] = torch.randn(len(u)) * 2
        out.append(u)
    return out


res = {}
base_plain = paths(pol, x)
res["plain_dv_max"] = max(float((paths(pol, u)[0] - base_plain[0]).abs().max()) for u in perturbed(x))
cls_before = type(pol.mlp_extractor)
pol.mlp_extractor.__class__ = H.critic_blind_class(cls_before)
base_blind = paths(pol, x)
res["blind_class"] = type(pol.mlp_extractor).__name__
res["blind_dv_max_per_path"] = [max(float((paths(pol, u)[i] - base_blind[i]).abs().max()) for u in perturbed(x))
                                for i in range(3)]
res["blind_dmu_max"] = max(float((paths(pol, u)[4] - base_blind[4]).abs().max()) for u in perturbed(x))
res["actor_mean_bitwise_equal_unswapped"] = bool(torch.equal(base_blind[4], base_plain[4]))
res["log_prob_bitwise_equal_unswapped"] = bool(torch.equal(base_blind[3], base_plain[3]))
x0 = x.clone()
x0[:, 6] = 0.0
pol.mlp_extractor.__class__ = cls_before
plain_at0 = paths(pol, x0)[0]
res["blind_equals_plain_at_obs6_0"] = bool(torch.equal(base_blind[0], plain_at0))
xc = x.clone()
pol.mlp_extractor.__class__ = H.critic_blind_class(cls_before)
paths(pol, x)
res["input_not_mutated"] = bool(torch.equal(x, xc))
pol.mlp_extractor.__class__ = cls_before
res["state_dict_keys_equal"] = sorted(pol.state_dict()) == sorted(sd)
res["bank_value_plain"] = [round(float(v), 6) for v in base_plain[0][:10]]
res["bank_value_blind"] = [round(float(v), 6) for v in base_blind[0][:10]]
print(json.dumps(res, indent=1))
