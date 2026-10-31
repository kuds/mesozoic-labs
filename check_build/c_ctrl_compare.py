"""PS build checker: compare PS's control no-update run at BS2's seed 9201 (PS/check_build/ctrl_root/validate/
control_9201) with BS2's own V2 control_9201 (read only): every evidence key both files carry, per rollout; the
evaluation ledger and the probe rows up to the last step both reached.  Prints the comparison.
    PY -I -B c_ctrl_compare.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
A = Path(S2) / "r10round/bias_screen2/validate/v2final/control_9201"
B = Path("/home/user/work/phase_screen/check_build/ctrl_root/validate/control_9201")


def rows(p):
    return [json.loads(x) for x in open(p)]


ra, rb = rows(A / "arm_evidence.jsonl"), rows(B / "arm_evidence.jsonl")
n = min(len(ra), len(rb))
print(f"BS2 rollouts {len(ra)}, PS rollouts {len(rb)}, compared {n}")
skip = {"clock"}
common = sorted((set(ra[0]) & set(rb[0])) - skip)
only_ps = sorted(set(rb[0]) - set(ra[0]))
only_bs2 = sorted(set(ra[0]) - set(rb[0]))
print("keys compared:", len(common), "| only in PS:", only_ps, "| only in BS2:", only_bs2)
bad = {}
for i in range(n):
    for k in common:
        if ra[i][k] != rb[i][k]:
            bad.setdefault(k, []).append(i)
print("differing keys (rollout indices):", bad if bad else "none")
for k in ("digest_env_side", "digest_rewards", "digest_values", "digest_advantages", "obs_rms_digest"):
    if k in common:
        print(f"  {k}: {sum(ra[i][k] == rb[i][k] for i in range(n))}/{n} equal")
print("PS-only phase keys on control:", {k: rb[0].get(k) for k in only_ps if not k.startswith("dmu") and
                                         not k.startswith("dv")})
print("control dmu_phase_mode per rollout (actor responds to the slots):",
      [round(r["dmu_phase_mode"], 4) for r in rb])
# the evaluation ledger and the probe up to the shorter run
for rel in ("01_stance/logs/eval_episodes.csv", "01_stance/eval_episodes.csv"):
    if (A / rel).is_file() and (B / rel).is_file():
        with open(A / rel) as fa, open(B / rel) as fb:
            ca, cb = list(csv.DictReader(fa)), list(csv.DictReader(fb))
        cols = [c for c in ca[0] if c in cb[0] and "time" not in c.lower() and "wall" not in c.lower()
                and "second" not in c.lower() and "utc" not in c.lower()]
        lim = max(int(r["timesteps"]) for r in cb)
        ca = [r for r in ca if int(r["timesteps"]) <= lim]
        cb = [r for r in cb if int(r["timesteps"]) <= lim]
        m = min(len(ca), len(cb))
        diff = [c for c in cols if any(ca[i][c] != cb[i][c] for i in range(m))]
        print(f"ledger {rel}: rows <= {lim}: BS2 {len(ca)} PS {len(cb)}, compared {m} rows x {len(cols)} columns; "
              f"differing: {diff}; timesteps covered {sorted({int(r['timesteps']) for r in cb})[-3:]}")
found = sorted(p.relative_to(B) for p in B.rglob("*") if p.is_file())
print("PS run files:", [str(p) for p in found][:40])
for rel in [p for p in found if p.name == "policy_probe.npz"]:
    za, zb = np.load(A / rel), np.load(B / rel)
    ta, tb = np.asarray(za["update_timesteps"]), np.asarray(zb["update_timesteps"])
    m = int(np.sum(tb <= ta.max()))
    keys = [k for k in zb.files if k in za.files and k != "bank_build_seconds"]   # wall time only
    diff = []
    for k in keys:
        x, y = np.asarray(za[k]), np.asarray(zb[k])
        if x.ndim and y.ndim and x.shape[0] == ta.shape[0] and y.shape[0] == tb.shape[0]:
            x, y = x[:m], y[:m]
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            diff.append(k)
    print(f"probe {rel}: rows BS2 {len(ta)} PS {len(tb)}, compared first {m}; arrays differing: {diff}")
sys.exit(0 if not bad else 1)
