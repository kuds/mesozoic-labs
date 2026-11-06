"""e2e lane: the stand-in 2,000,000 zip against its parent's 8M zip, tensor by tensor (policy.pth and
policy.optimizer.pth read with torch.load(weights_only=True); nothing else unpickled, no repository import, nothing
trains): which tensors and columns differ, their values there, log_std, the Adam step counts, num_timesteps.

usage: python -I -B weights_facts.py LANE OUT.json
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import torch

LANE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RUNS = Path(__file__).resolve().parents[5] / ("pilot3/runs/final")


def load(z: Path, member: str):
    with zipfile.ZipFile(z) as zf:
        return torch.load(io.BytesIO(zf.read(member)), weights_only=True, map_location="cpu")


def data(z: Path) -> dict:
    with zipfile.ZipFile(z) as zf:
        return json.loads(zf.read("data"))


res = {}
for s in (42, 52, 60):
    new_z = LANE / "runs" / f"pilot4_walk_s{s}" / "03_locomotion" / "models" / "stage2_2000000_steps.zip"
    old_z = RUNS / f"pilot3_walk_s{s}" / "03_locomotion" / "models" / "stage2_8000000_steps.zip"
    pn, po = load(new_z, "policy.pth"), load(old_z, "policy.pth")
    changed = {}
    for k, v in po.items():
        d = (pn[k].double() - v.double()).abs()
        n = int((d > 0).sum())
        if n:
            idx = torch.nonzero(d > 0)
            changed[k] = {"n_changed": n, "columns": sorted(set(idx[:, 1].tolist())) if idx.shape[1] > 1 else None,
                          "new_values_max_abs": float(pn[k].double()[d > 0].abs().max())}
    on, oo = load(new_z, "policy.optimizer.pth"), load(old_z, "policy.optimizer.pth")
    mom = {}
    steps = set()
    for i, st in oo["state"].items():
        sn = on["state"][i]
        for key, val in st.items():
            if key == "step":
                steps.add((float(val), float(sn[key])))
                continue
            d = (sn[key].double() - val.double()).abs()
            n = int((d > 0).sum())
            if n:
                idx = torch.nonzero(d > 0)
                mom[f"{i}.{key}"] = {"n_changed": n,
                                     "columns": sorted(set(idx[:, 1].tolist())) if idx.shape[1] > 1 else None}
    dn, do = data(new_z), data(old_z)
    res[s] = {"policy_tensors": len(po), "policy_tensors_changed": changed,
              "log_std_bit_equal": bool(torch.equal(pn["log_std"], po["log_std"])),
              "log_std_mean": float(pn["log_std"].mean()),
              "adam_moments_changed": mom, "adam_step_pairs_parent_new": sorted(steps),
              "num_timesteps_parent_new": [do.get("num_timesteps"), dn.get("num_timesteps")],
              "task_sha256_parent_new": [(do.get("mesozoic_task_fingerprint") or {}).get("task_sha256"),
                                         (dn.get("mesozoic_task_fingerprint") or {}).get("task_sha256")],
              "ent_coef_parent_new": [do.get("ent_coef"), dn.get("ent_coef")]}
OUT.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
