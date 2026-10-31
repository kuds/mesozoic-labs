"""PS build: read-only inspection of the start pair's command / phase slots (60, 61, 62).

Prints: the start pkl's obs_rms mean / var at 58..62 and its count; the zip's policy_net.0 / value_net.0 weight
columns 60..62 (max |w|) and their Adam moments (exp_avg, exp_avg_sq max |.|) and the Adam step; the mirror map's
perm / sign / names at 58..62; whether the code copy's MlpPolicy shares the features extractor.  Reads only.
"""
import io
import json
import pickle
import sys
import zipfile

import numpy as np
import torch

S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
F2 = S1 + "/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance/models/"
ZIP = F2 + "stage1_2097152_steps.zip.copy1"
PKL = F2 + "stage1_vecnormalize_2097152_steps.pkl.copy1"
CODE = sys.argv[1]
sys.path.insert(0, CODE)
out = {}
with open(PKL, "rb") as fh:
    vn = pickle.load(fh)
rms = vn.obs_rms
out["obs_rms_class"] = type(rms).__name__
out["obs_rms_mean_58_62"] = [float(x) for x in rms.mean[58:63]]
out["obs_rms_var_58_62"] = [float(x) for x in rms.var[58:63]]
out["obs_rms_mean_60_62_exact_zero"] = bool(np.all(rms.mean[60:63] == 0.0))
out["obs_rms_count"] = float(rms.count)
out["epsilon"], out["clip_obs"] = float(vn.epsilon), float(vn.clip_obs)
out["norm_slot_value_of_raw_zero"] = [float(x) for x in (0.0 - rms.mean[60:63]) / np.sqrt(rms.var[60:63] + vn.epsilon)]
out["norm_slot_value_of_raw_one"] = [float(x) for x in np.clip((1.0 - rms.mean[60:63])
                                                                / np.sqrt(rms.var[60:63] + vn.epsilon), -10, 10)]
with zipfile.ZipFile(ZIP) as z:
    pol = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    opt = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
    data = json.loads(z.read("data").decode())
out["policy_keys"] = [f"{k} {tuple(v.shape)}" for k, v in pol.items()]
for name in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
    w = pol[name]
    out[name + " max|w| cols 60-62"] = float(w[:, 60:63].abs().max())
    out[name + " max|w| cols 0-59"] = float(w[:, :60].abs().max())
names = list(pol.keys())
out["optimizer_param_groups"] = [{k: v for k, v in g.items() if k != "params"} | {"n_params": len(g["params"])}
                                 for g in opt["param_groups"]]
for idx, st in sorted(opt["state"].items()):
    name = names[idx]
    if name in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
        out[f"adam[{idx}] {name}"] = {
            "step": float(st["step"]), "exp_avg max| cols 60-62": float(st["exp_avg"][:, 60:63].abs().max()),
            "exp_avg_sq max cols 60-62": float(st["exp_avg_sq"][:, 60:63].abs().max()),
            "exp_avg max| cols 0-59": float(st["exp_avg"][:, :60].abs().max())}
out["policy_kwargs"] = data.get("policy_kwargs")
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.plant_contract import current_plant_identity  # noqa: E402
from environments.shared.symmetry import MirrorMap  # noqa: E402

mid = ModelId("tyrannosaurus_rex", "anatomical")
mm = MirrorMap.load(mid, current_plant_identity(mid), verify=False)
out["mirror_obs_dim"] = mm.obs.dim
out["mirror_obs_58_62"] = [{"i": i, "perm": int(mm.obs.perm[i]), "sign": float(mm.obs.sign[i]),
                            "name": mm.obs.names[i] if mm.obs.names else ""} for i in range(58, 63)]
out["mirror_obs_perm_6"] = [int(mm.obs.perm[6]), float(mm.obs.sign[6])]
print(json.dumps(out, indent=1, default=str))
