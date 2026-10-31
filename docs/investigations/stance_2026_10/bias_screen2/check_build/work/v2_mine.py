"""Independent reading of BS2's V2 (check_build).  Reads only; no BS2 tool imported.

argv: V2_DIR F2_ZIP PARENT_PROBE
Per arm: the raw arm_evidence.jsonl rows, run_meta digests, the saved zips at 2,359,296 and final (tensors and Adam
state against the F2 start zip, _n_updates), and the run's probe rows against the parent's schedule columns.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

V2, F2_ZIP, PPROBE = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
S, T, R8 = 2_097_152, 2_359_296, 8192
ARMS = ("control", "critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm")
SEED = 9201


def load_zip(p):
    with zipfile.ZipFile(p) as z:
        data = json.loads(z.read("data"))
        pol = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
        opt = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
    return data, pol, opt


def same_state(a, b):
    (_, p1, o1), (_, p2, o2) = a, b
    pe = set(p1) == set(p2) and all(torch.equal(p1[k], p2[k]) for k in p1)
    s1, s2 = o1["state"], o2["state"]
    oe = set(s1) == set(s2) and all(set(s1[i]) == set(s2[i]) and all(
        torch.equal(torch.as_tensor(s1[i][n]), torch.as_tensor(s2[i][n])) for n in s1[i]) for i in s1)
    ge = [g["lr"] for g in o1["param_groups"]] == [g["lr"] for g in o2["param_groups"]]
    return pe, oe, ge


start = load_zip(F2_ZIP)
pp = np.load(PPROBE, allow_pickle=False)
put = pp["update_timesteps"].astype(np.int64)
out = {}
ev_all = {}
for arm in ARMS:
    rd = V2 / f"{arm}_{SEED}"
    m = json.loads((rd / "run_meta.json").read_text())
    ev = [json.loads(x) for x in (rd / "arm_evidence.jsonl").read_text().splitlines() if x.strip()]
    ev_all[arm] = ev
    steps = [e["num_timesteps"] for e in ev]
    r = {"mode": m["mode"], "status": m["status"], "exit": m["exit"], "T": m["T"], "train_calls": m.get("train_calls"),
         "n_rows": len(ev), "steps_ok": steps == [S + R8 * k for k in range(1, 32)],
         "digest_params_start_eq_end": m["digest_start"]["params"] == m["digest_end"]["params"],
         "digest_adam_start_eq_end": m["digest_start"]["adam"] == m["digest_end"]["adam"],
         "n_updates_start_end": [m["digest_start"]["n_updates"], m["digest_end"]["n_updates"]],
         "arm_applied_classes": [m["arm_applied"].get("mlp_extractor_class"), m["arm_applied"].get("obs_rms_class")],
         "code_seal_after": m.get("code_seal_after"), "modules_outside_code": m.get("modules_outside_code")}
    for key in ("dv_predict_values", "dv_forward", "dv_evaluate_actions", "dv_mirrored"):
        vals = [e.get(key) for e in ev]
        r[key] = {"exact_zero": sum(1 for v in vals if v == 0.0), "positive": sum(1 for v in vals if v and v > 0),
                  "min": min(vals), "max": max(vals)}
    r["dmu_actor_positive"] = sum(1 for e in ev if e["dmu_actor"] > 0)
    r["dmu_actor_min"] = min(e["dmu_actor"] for e in ev)
    r["buffer_vs_critic_max"] = max(e["buffer_values_vs_critic"] for e in ev)
    r["buffer_vs_raw_mlp_min"] = min(e["buffer_values_vs_raw_value_mlp"] for e in ev)
    r["adv_err_vs_gae095_max"] = max(e["adv_err_vs_gae095"] for e in ev)
    r["adv_err_vs_mc_min"] = min(e["adv_err_vs_mc"] for e in ev)
    r["adv_err_vs_mc_max"] = max(e["adv_err_vs_mc"] for e in ev)
    r["gae_lambda_buffer_model"] = sorted({(e["buffer_gae_lambda"], e["model_gae_lambda"]) for e in ev})
    r["state_problems"] = sorted({p for e in ev for p in e.get("state_problems") or []})
    r["obs_rms_equal_start"] = sum(1 for e in ev if e["obs_rms_equal_start"])
    # saved zips: no optimizer step (tensors and Adam state equal the F2 start; _n_updates unchanged)
    for name in ("stage1_2359296_steps.zip", "stage1_final.zip"):
        z = load_zip(rd / "01_stance" / "models" / name)
        pe, oe, ge = same_state(start, z)
        r[name] = {"params_equal_F2": pe, "adam_state_equal_F2": oe, "lr_groups_equal_F2": ge,
                   "n_updates": z[0]["_n_updates"], "num_timesteps": z[0]["num_timesteps"],
                   "adam_step": sorted({float(v["step"]) for v in z[2]["state"].values()})}
    # the run's probe: rows in (S, T), schedule columns equal the parent's at the same update_timesteps
    with np.load(rd / "01_stance" / "policy_probe.npz", allow_pickle=False) as z:
        ut = z["update_timesteps"].astype(np.int64)
        w = (ut > S) & (ut < T)
        idx = [int(np.flatnonzero(put == t)[0]) for t in ut[w]]
        r["probe"] = {"rows_le_S": int(np.sum(ut <= S)), "rows_in_(S,T)": int(np.sum(w)), "rows_ge_T": int(np.sum(ut >= T)),
                      "ent_equal_parent": int(np.sum(z["ent_coef"][w] == pp["ent_coef"][idx])),
                      "lr_equal_parent": int(np.sum(z["learning_rate"][w] == pp["learning_rate"][idx])),
                      "seeded_rows_equal_parent_mu14": bool(np.array_equal(z["rollout_mu_mean"][ut <= S],
                                                                           pp["rollout_mu_mean"][put <= S])),
                      "mu14_window_first_last": [float(z["rollout_mu_mean"][w][0, 14]),
                                                 float(z["rollout_mu_mean"][w][-1, 14])]}
    out[arm] = r
pair = {}
for arm in ARMS[1:]:
    pair[arm] = {k: sum(1 for a, b in zip(ev_all["control"], ev_all[arm]) if a[k] == b[k])
                 for k in ("digest_env_side", "digest_rewards", "digest_values", "digest_advantages")}
out["pairing_vs_control_equal_counts"] = pair
print(json.dumps(out, indent=1))
