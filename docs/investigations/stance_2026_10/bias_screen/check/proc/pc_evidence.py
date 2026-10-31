"""Process lens: per-rollout arm evidence (arm_evidence.jsonl) for every continuation, read independently of
bs_hooks.evidence_problems; plus the probe's row steps and the 8 primary rows.  Reads only; prints."""
import json
import os
import sys
from fractions import Fraction

import numpy as np

BS = sys.argv[1]
S, E, R = 1048576, 2490368, 8192
screen = json.load(open(os.path.join(BS, "SCREEN.json")))
want_steps = [S + R * k for k in range(1, 176)]
first_env = {}
second_env = {}
prim = {}
for arm, seed in screen["queue"]:
    name = f"{arm}_{seed}"
    rd = os.path.join(BS, "runs", name)
    rows = [json.loads(x) for x in open(os.path.join(rd, "arm_evidence.jsonl"))]
    m = json.load(open(os.path.join(rd, "run_meta.json")))
    p = []
    steps = [r["num_timesteps"] for r in rows]
    if steps != want_steps:
        p.append("steps")
    for r in rows:
        k = r["num_timesteps"]
        if r["buffer_gae_lambda"] != 0.95 or r["model_gae_lambda"] != 0.95:
            p.append(f"{k} lambda")
        if not r["dmu_actor"] > 0:
            p.append(f"{k} actor blind")
        if r["state_problems"]:
            p.append(f"{k} state {r['state_problems']}")
        if r["obs_rms_equal_start"]:
            p.append(f"{k} obs_rms frozen")
        if not r["adv_err_vs_gae095"] <= 1e-4 * max(1.0, r["adv_scale"]) * 10:
            p.append(f"{k} adv err {r['adv_err_vs_gae095']}")
        if not r["gae095_vs_mc"] > 1e-3:
            p.append(f"{k} gae095 == mc")
        if r["returns_minus_adv_minus_values"] > 1e-5:
            p.append(f"{k} returns")
        if r["buffer_values_vs_critic"] > 1e-5 * r["value_scale"]:
            p.append(f"{k} buffer values not the run's critic {r['buffer_values_vs_critic']}")
        if arm == "critic_blind_tail3":
            for f in ("dv_predict_values", "dv_forward", "dv_evaluate_actions", "dv_mirrored"):
                if r[f] != 0.0:
                    p.append(f"{k} {f} {r[f]}")
            if not r["buffer_values_vs_raw_value_mlp"] > 1e-5 * r["value_scale"]:
                p.append(f"{k} blind values == raw")
        else:
            for f in ("dv_predict_values", "dv_forward", "dv_evaluate_actions", "dv_mirrored"):
                if not r[f] > 0:
                    p.append(f"{k} control {f} {r[f]}")
            if r["buffer_values_vs_raw_value_mlp"] > 1e-5 * r["value_scale"]:
                p.append(f"{k} control values != raw MLP {r['buffer_values_vs_raw_value_mlp']}")
    digs = [r["obs_rms_digest"] for r in rows]
    if len(set(digs)) != 175:
        p.append("obs_rms digests not distinct")
    # summary in run_meta equals the rows
    ev = m["arm_evidence"]
    chk = {"max_adv_err_vs_gae095": max(r["adv_err_vs_gae095"] for r in rows),
           "min_dmu_actor": min(r["dmu_actor"] for r in rows),
           "max_dv_predict_values": max(r["dv_predict_values"] for r in rows),
           "min_dv_predict_values": min(r["dv_predict_values"] for r in rows),
           "max_dv_mirrored": max(r["dv_mirrored"] for r in rows)}
    for k2, v in chk.items():
        if k2 in ev and ev[k2] != v:
            p.append(f"summary {k2} {ev[k2]} != rows {v}")
    first_env.setdefault(seed, {})[arm] = rows[0]["digest_env_side"]
    second_env.setdefault(seed, {})[arm] = rows[1]["digest_env_side"]
    clocks = [r["clock"]["utc"] for r in rows]
    # probe
    with np.load(os.path.join(rd, "01_stance", "policy_probe.npz"), allow_pickle=False) as z:
        keys = list(z.keys())
        ut = np.asarray(z["update_timesteps"]).astype(np.int64)
        mu = np.asarray(z["rollout_mu_mean"])
    win = ut[ut > S]
    if list(win) != want_steps:
        p.append(f"probe window rows {len(win)}")
    if (ut <= S).sum() != 128:
        p.append(f"probe seeded rows {(ut <= S).sum()}")
    last8 = want_steps[-8:]
    vals = []
    for st in last8:
        idx = np.nonzero(ut == st)[0]
        if len(idx) != 1 or not np.isfinite(mu[idx[0], 14]):
            p.append(f"primary row {st}")
        else:
            vals.append(Fraction(float(mu[idx[0], 14])))
    prim[name] = float(sum(vals) / len(vals)) if len(vals) == 8 else None
    print(f"{name:26s} rows {len(rows)} clocks {clocks[0]}..{clocks[-1]} dv_pred min {min(r['dv_predict_values'] for r in rows):.4g}"
          f" max {max(r['dv_predict_values'] for r in rows):.4g} primary {prim[name]!r} problems {p[:6]}{'...' if len(p) > 6 else ''} ({len(p)})")
print("probe keys", keys)
print("pair env side, first rollout equal across arms:", {s: len(set(d.values())) == 1 for s, d in first_env.items()})
print("pair env side, second rollout equal across arms:", {s: len(set(d.values())) == 1 for s, d in second_env.items()})
BS_ = -0.03157372563146055
ctrl = sorted(v for n, v in prim.items() if n.startswith("control"))
med = (ctrl[1] + ctrl[2]) / 2
print("control primaries", ctrl, "median", med, "rise", med - BS_, "informative", med - BS_ >= 0.10)
