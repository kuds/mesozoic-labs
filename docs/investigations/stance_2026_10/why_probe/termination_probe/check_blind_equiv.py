"""Check that run_probe.py's effective blinded critic (normalised obs[6] zeroed before predict_values) equals BS2's hook
(bs_hooks.critic_blind_class installed on the policy's mlp_extractor), on synthetic normal inputs (no rollout data).
Also checks autograd dV/dz6 against a finite difference on the same inputs (plain critic of p2097152)."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_common as T  # noqa: E402

T.setup_simulation("equiv")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.torch_layers import MlpExtractor  # noqa: E402

spec = importlib.util.spec_from_file_location("bs_hooks", T.BS2 + "/tools/bs_hooks.py")
H = importlib.util.module_from_spec(spec)
sys.modules["bs_hooks"] = H
spec.loader.exec_module(H)

factory = T.env_factory()
rng = np.random.default_rng(12345)
Z = rng.normal(size=(256, 63)).astype(np.float32)
model, norm = T.load_pair("b2511_E", factory)
pol = model.policy
with torch.no_grad():
    Zm = torch.as_tensor(Z.copy())
    Zm[:, 6] = 0.0
    mine = pol.predict_values(Zm)[:, 0].numpy()
    raw = pol.predict_values(torch.as_tensor(Z))[:, 0].numpy()
pol.mlp_extractor.__class__ = H.critic_blind_class(MlpExtractor)
with torch.no_grad():
    hook = pol.predict_values(torch.as_tensor(Z))[:, 0].numpy()
pol.mlp_extractor.__class__ = MlpExtractor
print("b2511_E: max |mine - hook| =", float(np.abs(mine - hook).max()), " max |raw - hook| =", float(np.abs(raw - hook).max()))

model, norm = T.load_pair("p2097152", factory)
pol = model.policy
Zt = torch.as_tensor(Z.copy()).requires_grad_(True)
g = torch.autograd.grad(pol.predict_values(Zt).sum(), Zt)[0][:, 6].numpy()
h = 1e-2
with torch.no_grad():
    Zp, Zn = torch.as_tensor(Z.copy()), torch.as_tensor(Z.copy())
    Zp[:, 6] += h
    Zn[:, 6] -= h
    fd = ((pol.predict_values(Zp) - pol.predict_values(Zn))[:, 0] / (2 * h)).numpy()
print("p2097152: autograd vs FD dV/dz6: max abs diff", float(np.abs(g - fd).max()), " mean |g|", float(np.abs(g).mean()))
