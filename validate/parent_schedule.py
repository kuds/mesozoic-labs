"""BS2 step 3: the parent's records over the new window (files only; nothing simulated, nothing written but stdout).

    PY -I -B parent_schedule.py <BS2 tools dir>

Reads the parent's F2 copies (stance6_r10LTSB_s42, read only, each sha256-checked against bs_common.START, which carries
FMAN's values) and reports, for the window (S, E] of bs_common:
  - the probe's coverage: update_timesteps rows, the row at S, the rows S + 8,192 k (k = 1 .. (E - S)/8,192 - 1), and
    rows at or beyond E (the scorer's parent reference column needs (S, E));
  - the schedule CHECK compares (PROBE_CHECK_COLUMNS: ent_coef, learning_rate) per row, against the code's own formulas
    (EntCoefDecayCallback: ent_coef = 0.005 + min(1, t / 7,000,000) (0 - 0.005); SB3 + LinearSchedule:
    lr = 1e-5 + (1 - t / 12,000,000) (3e-5 - 1e-5)), as float32;
  - the start zip's own records (its 'data' member read as JSON, nothing unpickled): num_timesteps, _n_updates,
    _total_timesteps, ent_coef, gae_lambda, n_steps, n_epochs, batch_size, gamma;
  - the start pkl's statistics (the code's VecNormalize pickle, loaded with the venv's SB3) against the probe's
    obs_rms row at S (float32), and the counts;
  - the ledger's (eval_episodes.csv) evaluations in (S, E] and their episode counts, and the evaluations.npz grid;
  - B_S candidate: the mean of the probe's rollout_mu_mean[:, 14] over the 8 rows ending at S (exact Fraction mean of
    the float32 values as float64, the BS scorer's mean_over form), and the same at the BS start 1,048,576 as a check of
    the form against BS's pinned -0.03157372563146055;
  - the last 8 update rows before E (the primary's rows in the BS form) and the first 8 after S.
"""
import csv
import hashlib
import importlib.util
import io
import json
import pickle
import sys
import zipfile
from fractions import Fraction

import numpy as np

spec = importlib.util.spec_from_file_location("bs_common", sys.argv[1] + "/bs_common.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

S, E, R = C.S, C.E, C.ROLLOUT
ST = C.START
out = {"S": S, "E": E, "LENGTH": C.LENGTH}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---- sha256 of every input against bs_common.START (= FMAN, re-checked by check_inputs.py)
files = {"zip": (C.f2_file(ST["zip_rel"]), ST["zip_sha256"]), "pkl": (C.f2_file(ST["pkl_rel"]), ST["pkl_sha256"])}
for n in (*C.PARENT_RECORDS, "stage_config.json"):
    files[n] = (C.f2_record(n), ST["records"][n])
out["sha256_ok"] = {k: sha(p) == w for k, (p, w) in files.items()}
assert all(out["sha256_ok"].values()), out["sha256_ok"]

# ---- grid
out["grid"] = {"E_mod_eval_freq": E % C.EVAL_FREQ, "E_mod_save_freq": E % C.SAVE_FREQ,
               "S_mod_save_freq": S % C.SAVE_FREQ,
               "n_evals": C.LENGTH // C.EVAL_FREQ, "evals": [S + C.EVAL_FREQ * k for k in range(1, C.N_EVALS + 1)],
               "rollouts": C.LENGTH // R, "update_rows": C.expected_update_rows(S, E),
               "saved_steps": list(C.SAVED_STEPS), "validate_until": C.VALIDATE_UNTIL,
               "f2_pairs_in_[S,E]": [s for s in range(1_048_576, E + 1, 1_048_576) if s >= S]}

# ---- probe
with np.load(files["policy_probe.npz"][0], allow_pickle=False) as z:
    ut = np.asarray(z["update_timesteps"]).astype(np.int64)
    ent = np.asarray(z["ent_coef"], dtype=np.float32)
    lr = np.asarray(z["learning_rate"], dtype=np.float32)
    mu = np.asarray(z["rollout_mu_mean"], dtype=np.float32)
    om, ov, oc = (np.asarray(z[k]) for k in ("obs_rms_mean", "obs_rms_var", "obs_rms_count"))
diffs = np.diff(ut)
row_S = np.flatnonzero(ut == S)
want = [S + R * k for k in range(1, C.expected_update_rows(S, E) + 1)]
where = {int(u): i for i, u in enumerate(ut.tolist())}
missing = [s for s in want if s not in where]
out["probe"] = {"rows": int(len(ut)), "first": int(ut[0]), "last": int(ut[-1]),
                "all_spaced_8192": bool((diffs == R).all()),
                "unique": bool(len(set(ut.tolist())) == len(ut)), "row_at_S": row_S.tolist(),
                "rows_at_or_before_S": int((ut <= S).sum()), "window_rows_expected": len(want),
                "window_rows_missing": missing, "rows_at_or_after_E": int((ut >= E).sum()),
                "covers_E": bool(ut[-1] >= E)}
idx = [where[s] for s in want]
t = ut[idx].astype(np.float64)
ent_f = np.float32(0.005 + np.minimum(1.0, t / 7_000_000.0) * (0.0 - 0.005))
lr_f = np.float32(1e-5 + (1.0 - t / 12_000_000.0) * (3e-5 - 1e-5))
out["schedule_over_window"] = {
    "ent_coef_first_last": [float(ent[idx[0]]), float(ent[idx[-1]])],
    "learning_rate_first_last": [float(lr[idx[0]]), float(lr[idx[-1]])],
    "ent_coef_formula_equal_rows": int((ent[idx] == ent_f).sum()),
    "ent_coef_formula_max_abs": float(np.max(np.abs(ent[idx] - ent_f))),
    "lr_formula_equal_rows": int((lr[idx] == lr_f).sum()), "lr_formula_max_abs": float(np.max(np.abs(lr[idx] - lr_f))),
    "lr_formula_max_rel": float(np.max(np.abs(lr[idx] - lr_f) / lr_f)), "rows": len(idx)}
# the whole run, to show the anchors (7M for ent_coef, 12M for lr)
tt = ut.astype(np.float64)
lr_all = np.float32(1e-5 + (1 - tt / 12e6) * 2e-5)
out["schedule_whole_run"] = {
    "ent_coef_formula_max_abs": float(np.max(np.abs(ent - np.float32(0.005 + np.minimum(1.0, tt / 7e6) * -0.005)))),
    "lr_formula_max_rel": float(np.max(np.abs(lr - lr_all) / lr_all)),
    "ent_coef_at_7M_and_after_zero": bool((ent[ut >= 7_000_000] == 0).all())}


def fmean(steps):
    vals = []
    for s in steps:
        i = np.flatnonzero(ut == s)
        assert len(i) == 1, s
        vals.append(Fraction(float(mu[i[0], C.ACT_TAIL3])))
    m = sum(vals, Fraction(0)) / len(vals)
    return float(m), [int(np.flatnonzero(ut == s)[0]) for s in steps], [float(v) for v in vals]


bs_steps = [S - R * k for k in range(7, -1, -1)]
b_s, b_rows, b_vals = fmean(bs_steps)
old_steps = [1_048_576 - R * k for k in range(7, -1, -1)]
b_old, _, _ = fmean(old_steps)
last8 = [E - R * k for k in range(8, 0, -1)]
l8, l8_rows, l8_vals = fmean(last8)
first8 = [S + R * k for k in range(1, 9)]
f8, _, _ = fmean(first8)
out["B_S"] = {"steps": bs_steps, "rows": b_rows, "values": b_vals, "mean": b_s, "mean_repr": repr(b_s),
              "check_at_1048576": {"mean": b_old, "BS_pinned": -0.03157372563146055,
                                   "equal": b_old == -0.03157372563146055}}
out["last8_before_E"] = {"steps": last8, "rows": l8_rows, "parent_mean_report_only": l8}
out["first8_after_S"] = {"steps": first8, "parent_mean_report_only": f8}

# ---- the start zip's own records ('data' is JSON; nothing unpickled)
with zipfile.ZipFile(files["zip"][0]) as zf:
    data = json.loads(zf.read("data"))
    names = zf.namelist()
    import torch

    opt = torch.load(io.BytesIO(zf.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
steps_adam = sorted({int(v["step"]) for v in opt["state"].values()})
keep = ("num_timesteps", "_n_updates", "_total_timesteps", "ent_coef", "gae_lambda", "gamma", "n_steps", "n_epochs",
        "batch_size", "_current_progress_remaining", "seed", "n_envs")
out["start_zip"] = {k: data.get(k) for k in keep}
out["start_zip"]["members"] = names
out["start_zip"]["adam_step_values"] = steps_adam
out["start_zip"]["adam_lr_param_groups"] = [g.get("lr") for g in opt["param_groups"]]
n_up = data.get("_n_updates")
out["start_zip"]["n_updates_vs_trains"] = {"n_updates": n_up, "n_epochs": data.get("n_epochs"),
                                           "trains_implied": (n_up / data.get("n_epochs")) if n_up else None,
                                           "rollouts_to_S": S // R}

# ---- the start pkl's statistics against the probe's row at S
with open(files["pkl"][0], "rb") as fh:
    vn = pickle.load(fh)  # noqa: S301 - the pinned parent file (sha256 checked above)
i = int(row_S[0])
out["start_pkl"] = {"class": type(vn).__name__, "obs_rms_class": type(vn.obs_rms).__name__,
                    "ret_rms_class": type(vn.ret_rms).__name__, "obs_count": float(vn.obs_rms.count),
                    "ret_count": float(vn.ret_rms.count), "clip_obs": float(vn.clip_obs), "epsilon": float(vn.epsilon),
                    "training": bool(vn.training), "norm_obs": bool(vn.norm_obs), "norm_reward": bool(vn.norm_reward),
                    "probe_row_S_obs_count": float(oc[i]),
                    "obs_mean_f32_equal_probe_row_S": bool(np.array_equal(np.float32(vn.obs_rms.mean),
                                                                          np.float32(om[i]))),
                    "obs_var_f32_equal_probe_row_S": bool(np.array_equal(np.float32(vn.obs_rms.var),
                                                                         np.float32(ov[i]))),
                    "obs_count_equal_probe_row_S": float(vn.obs_rms.count) == float(oc[i]),
                    "t3_norm_mean_deg": float(np.degrees(vn.obs_rms.mean[C.OBS_TAIL3])),
                    "t3_norm_sd_deg": float(np.degrees(np.sqrt(vn.obs_rms.var[C.OBS_TAIL3])))}

# ---- the ledger and evaluations.npz
with open(files["eval_episodes.csv"][0], newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
by = {}
for r in rows:
    try:
        tv = int(float(r["timesteps"]))
    except (TypeError, ValueError):
        continue
    by[tv] = by.get(tv, 0) + 1
evs = [S + C.EVAL_FREQ * k for k in range(1, C.N_EVALS + 1)]
out["ledger"] = {"rows": len(rows), "first": min(by), "last": max(by), "at_S": by.get(S),
                 "window_evals": {s: by.get(s) for s in evs}, "window_complete": all(by.get(s) for s in evs),
                 "evals_after_E": sum(1 for s in by if s > E),
                 "has_cmd_column": "tail_3_pitch_act_command_mean" in rows[0]}
with np.load(files["evaluations.npz"][0], allow_pickle=False) as z:
    ts = np.asarray(z["timesteps"]).astype(np.int64)
out["evaluations_npz"] = {"n": int(len(ts)), "first": int(ts[0]), "last": int(ts[-1]),
                          "window_present": [int(s) for s in evs if s in set(ts.tolist())]}
with open(files["train_episodes.csv"][0], newline="", encoding="utf-8") as fh:
    head = fh.readline().strip().split(",")
    last_line = None
    for line in fh:
        last_line = line
out["train_episodes"] = {"columns": len(head), "has_timesteps": "timesteps" in head}
if "timesteps" in head and last_line:
    out["train_episodes"]["last_timesteps"] = last_line.split(",")[head.index("timesteps")]
with np.load(files["gate_progress.npz"][0], allow_pickle=False) as z:
    keys = list(z.keys())
    gk = next((k for k in keys if "timestep" in k), None)
    out["gate_progress"] = {"keys": keys[:12], "timestep_key": gk,
                            "last": int(np.asarray(z[gk]).reshape(-1)[-1]) if gk else None}
print(json.dumps(out, indent=1, default=str))
