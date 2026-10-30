"""LB step 4: the parent's records over the whole window (S, E] (files only; nothing simulated; prints JSON to stdout).

    PY -I -B parent_schedule.py <LB tools dir>

Derived from BS2's validate/parent_schedule.py (S2/r10round/bias_screen2/validate/parent_schedule.py).  Reads the
parent's F2 copies (stance6_r10LTSB_s42, read only; the five records and the start pair sha256-checked against
bs_common.START = FMAN; the other F2 pairs read as found and their sha256 printed) and the code copy's stance.toml
(read only), and reports:
  - how the parent's run ended: the probe's last update row, the final zip's num_timesteps / _n_updates /
    _total_timesteps ('data' member, JSON; nothing unpickled), the evaluations' and gate_progress' last rows, the train
    ledger's last row: E;
  - the probe's coverage of (S, E]: one row at every S + 8,192 k, k = 1 .. (E - S)/8,192 (the row at E is the parent's
    update on the rollout ending at E, which a continuation does not make);
  - the schedule CHECK compares (ent_coef, learning_rate) on every window row against the code's formulas
    (EntCoefDecayCallback, absolute steps, 7,000,000; SB3 progress over _total_timesteps 12,000,000), float32;
  - the evaluation grid (evaluations.npz, gate_progress.npz, eval_episodes.csv): every S + 131,072 k in (S, E];
  - n_updates at each F2 pair (PPO epochs actually run; target_kl stops some early): epochs per update over the window
    (the update's cost, for TIMING);
  - the stage's own early stops in the window: the hop watch (stance.toml hop_watch_*: unsupported duty > bar at an
    evaluation >= after_timesteps stops training) on the parent's own gate_progress series; the collapse backstop
    (curriculum/early_stopping.py, re-implemented here line by line: min_evals, smoothing window, warm-up, floor,
    patience) replayed on the parent's evaluations.npz as a continuation from S sees them (16 seeded evaluations, the
    patience counter starting at 0 at S): when it could first fire, the observations it gets, and how many consecutive
    drops the parent's own window had;
  - the last 8 update rows before E (steps) and the first 8 after S.
"""
import csv
import hashlib
import importlib.util
import json
import sys
import tomllib
import zipfile
from pathlib import Path

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


files = {"zip": (C.f2_file(ST["zip_rel"]), ST["zip_sha256"]), "pkl": (C.f2_file(ST["pkl_rel"]), ST["pkl_sha256"])}
for n in (*C.PARENT_RECORDS, "stage_config.json"):
    files[n] = (C.f2_record(n), ST["records"][n])
out["sha256_ok"] = {k: sha(p) == w for k, (p, w) in files.items()}
assert all(out["sha256_ok"].values()), out["sha256_ok"]

# ---- the grid this harness uses
out["grid"] = {"E_mod_rollout": E % R, "E_over_rollout": E // R, "E_mod_eval_freq": E % C.EVAL_FREQ,
               "E_mod_save_freq": E % C.SAVE_FREQ, "n_evals": C.N_EVALS,
               "first_last_eval": [S + C.EVAL_FREQ, S + C.EVAL_FREQ * C.N_EVALS],
               "rollouts": C.LENGTH // R, "update_rows": C.expected_update_rows(S, E),
               "saved_steps_n": len(C.SAVED_STEPS), "saved_first_last": [C.SAVED_STEPS[0], C.SAVED_STEPS[-1]],
               "saved_pairs_with_start_and_final": len(C.SAVED_STEPS) + 2,
               "hop_watch_first_eval": C.HOP_WATCH_FIRST_EVAL, "validate_until": C.VALIDATE_UNTIL}

# ---- the probe
with np.load(files["policy_probe.npz"][0], allow_pickle=False) as z:
    ut = np.asarray(z["update_timesteps"]).astype(np.int64)
    ent = np.asarray(z["ent_coef"], dtype=np.float32)
    lr = np.asarray(z["learning_rate"], dtype=np.float32)
    mu = np.asarray(z["rollout_mu_mean"], dtype=np.float64)
where = {int(u): i for i, u in enumerate(ut.tolist())}
want = [S + R * k for k in range(1, (E - S) // R + 1)]               # (S, E], the row at E included
missing = [s for s in want if s not in where]
out["probe"] = {"rows": int(len(ut)), "first": int(ut[0]), "last": int(ut[-1]),
                "all_spaced_8192": bool((np.diff(ut) == R).all()), "unique": len(set(ut.tolist())) == len(ut),
                "rows_at_or_before_S": int((ut <= S).sum()), "row_at_S": where.get(S),
                "window_rows_(S,E]_expected": len(want), "window_rows_missing": missing,
                "rows_after_E": int((ut > E).sum()), "row_at_E": where.get(E),
                "continuation_rows_(S,E)": C.expected_update_rows(S, E)}
idx = [where[s] for s in want]
t = ut[idx].astype(np.float64)
ent_f = np.float32(0.005 + np.minimum(1.0, t / 7_000_000.0) * (0.0 - 0.005))
lr_f = np.float32(1e-5 + (1.0 - t / 12_000_000.0) * (3e-5 - 1e-5))
in7 = t >= 7_000_000
out["schedule_over_window"] = {
    "rows": len(idx), "ent_coef_formula_equal_rows": int((ent[idx] == ent_f).sum()),
    "ent_coef_formula_max_abs": float(np.max(np.abs(ent[idx] - ent_f))),
    "lr_formula_equal_rows": int((lr[idx] == lr_f).sum()), "lr_formula_max_abs": float(np.max(np.abs(lr[idx] - lr_f))),
    "ent_coef_first_last": [float(ent[idx[0]]), float(ent[idx[-1]])],
    "learning_rate_first_last": [float(lr[idx[0]]), float(lr[idx[-1]])],
    "ent_coef_zero_rows_from_7M": int((ent[idx][in7] == 0).sum()), "rows_from_7M": int(in7.sum()),
    "first_row_ent_coef_zero": int(t[in7][0]) if in7.any() else None,
    "lr_at_E_below_1e-5": bool(lr[where[E]] < np.float32(1e-5)) if E in where else None}
tt = ut.astype(np.float64)
out["schedule_whole_run"] = {
    "ent_coef_equal_rows": int((ent == np.float32(0.005 + np.minimum(1.0, tt / 7e6) * -0.005)).sum()),
    "lr_equal_rows": int((lr == np.float32(1e-5 + (1 - tt / 12e6) * 2e-5)).sum()), "rows": int(len(ut))}
out["last8_before_E_steps"] = [E - R * k for k in range(8, 0, -1)]
out["first8_after_S_steps"] = [S + R * k for k in range(1, 9)]
out["parent_mean_mu14_last8_report_only"] = float(np.mean([mu[where[s], C.ACT_TAIL3]
                                                          for s in out["last8_before_E_steps"]]))

# ---- the F2 pairs: the end, and n_updates (epochs run) along the window
models = Path(C.f2_copies()) / C.STAGE_DIR / "models"
pairs = []
for zp in sorted(models.glob("stage1_*.zip.copy1")):
    with zipfile.ZipFile(zp) as zf:
        data = json.loads(zf.read("data"))
    pairs.append({"file": zp.name, "sha256": sha(zp), "num_timesteps": data.get("num_timesteps"),
                  "_n_updates": data.get("_n_updates"), "_total_timesteps": data.get("_total_timesteps"),
                  "ent_coef": data.get("ent_coef"), "n_epochs": data.get("n_epochs")})
pairs.sort(key=lambda d: d["num_timesteps"])
for a, b in zip(pairs, pairs[1:], strict=False):
    trains = (b["num_timesteps"] - a["num_timesteps"]) // R
    b["epochs_per_train_since_previous"] = (b["_n_updates"] - a["_n_updates"]) / trains if trains else None
out["f2_pairs"] = pairs
final = next(p for p in pairs if p["file"] == "stage1_final.zip.copy1")
out["end"] = {"final_zip_num_timesteps": final["num_timesteps"], "final_zip_total_timesteps": final["_total_timesteps"],
              "probe_last_row": int(ut[-1]), "E_equals_both": final["num_timesteps"] == int(ut[-1]) == E}
win = [p for p in pairs if S <= p["num_timesteps"] <= E]
first, last = win[0], win[-1]
out["epochs_per_train_window"] = {
    "from": first["num_timesteps"], "to": last["num_timesteps"],
    "mean": (last["_n_updates"] - first["_n_updates"]) / ((last["num_timesteps"] - first["num_timesteps"]) // R),
    "note": "the final pair is saved after the update on the rollout ending at E (the parent's SB3 loop)"}

# ---- evaluations, gate_progress, ledgers
with np.load(files["evaluations.npz"][0], allow_pickle=False) as z:
    ets = np.asarray(z["timesteps"]).astype(np.int64)
    eres = np.asarray(z["results"], dtype=np.float64)
evs = [S + C.EVAL_FREQ * k for k in range(1, C.N_EVALS + 1)]
out["evaluations_npz"] = {"n": int(len(ets)), "first": int(ets[0]), "last": int(ets[-1]),
                          "at_or_before_S": int((ets <= S).sum()), "window_present": int(sum(s in set(ets.tolist())
                                                                                           for s in evs)),
                          "window_expected": len(evs), "after_E": int((ets > E).sum()),
                          "episodes_per_eval": sorted({int(x) for x in np.asarray([len(r) for r in eres])})}
with np.load(files["gate_progress.npz"][0], allow_pickle=False) as z:
    gts = np.asarray(z["timesteps"]).astype(np.int64)
    duty = np.asarray(z["unsupported_duty"], dtype=np.float64)
out["gate_progress"] = {"n": int(len(gts)), "last": int(gts[-1]), "equal_eval_grid": gts.tolist() == ets.tolist()}
with open(files["eval_episodes.csv"][0], newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
by = {}
for r in rows:
    by[int(float(r["timesteps"]))] = by.get(int(float(r["timesteps"])), 0) + 1
out["eval_ledger"] = {"rows": len(rows), "evaluations": len(by), "last": max(by),
                      "window_complete_10_each": all(by.get(s) == 10 for s in evs),
                      "columns_evaluation_episode_timesteps": [c in rows[0] for c in ("evaluation", "episode",
                                                                                      "timesteps")]}
with open(files["train_episodes.csv"][0], newline="", encoding="utf-8") as fh:
    rd = csv.DictReader(fh)
    trows = list(rd)
tcols = list(trows[0])
step_col = next((c for c in ("timesteps", "num_timesteps", "end_timesteps") if c in tcols), None)
out["train_ledger"] = {"rows": len(trows), "columns": len(tcols), "step_column": step_col,
                       "last_step": trows[-1].get(step_col) if step_col else None}

# ---- the stage's own early stops (stance.toml, the code copy)
toml = tomllib.loads((Path(C.CODE) / "configs/tyrannosaurus_rex/anatomical/stance.toml").read_text())
cur = toml["curriculum"]
hop = {k: cur.get(k) for k in ("hop_watch_max_unsupported_duty", "hop_watch_after_timesteps", "hop_watch_stop")}
bar, after = float(hop["hop_watch_max_unsupported_duty"]), int(hop["hop_watch_after_timesteps"])
eligible = gts >= after
trips = [int(s) for s, d in zip(gts[eligible], duty[eligible], strict=True) if np.isfinite(d) and d > bar]
out["hop_watch"] = {"settings": hop, "first_eligible_eval": int(gts[eligible][0]) if eligible.any() else None,
                    "eligible_evals_in_parent": int(eligible.sum()),
                    "parent_max_duty_from_after": float(np.nanmax(duty[eligible])) if eligible.any() else None,
                    "parent_evals_over_bar_from_after": trips, "parent_would_have_stopped": bool(trips),
                    "harness_constant_first_eval": C.HOP_WATCH_FIRST_EVAL}
col = {"min_evals": int(cur.get("collapse_min_evals", 12)), "patience": int(cur.get("collapse_patience", 8)),
       "drop_fraction": float(cur.get("collapse_drop_fraction", 0.4)),
       "smoothing_window": int(cur.get("collapse_smoothing_window", 5)),
       "peak_warmup_timesteps": float(cur.get("collapse_peak_warmup_timesteps", 0.0))}
if "collapse_peak_floor" in cur:
    col["peak_floor"] = float(cur["collapse_peak_floor"])
else:
    col["peak_floor"] = float(cur["collapse_peak_floor_fraction"]) * float(cur["collapse_peak_floor_reference"])
means = eres.mean(axis=1)


def collapse_replay(start_index):
    """early_stopping.EvalCollapseEarlyStopCallback._on_step on evaluations[: n] for n = start_index+1 .., the counter
    starting at 0 (a fresh callback: the run's start or a resume), as each new evaluation arrives."""
    drops, best, first_obs, n_obs, fire = 0, 0, None, 0, None
    w = col["smoothing_window"]
    for n in range(start_index + 1, len(means) + 1):
        if n < max(col["min_evals"], w):
            continue
        m = means[:n]
        rm = np.array([np.median(m[i - w + 1: i + 1]) for i in range(w - 1, n)])
        if col["peak_warmup_timesteps"] > 0:
            ok = ets[: rm.size] >= col["peak_warmup_timesteps"]
            if not ok.any():
                drops = 0
                continue
            rm = rm[ok]
        peak = float(rm.max())
        thr = (1 - col["drop_fraction"]) * peak
        armed = peak > 0 and peak >= col["peak_floor"]
        n_obs += 1
        first_obs = first_obs or int(ets[n - 1])
        if armed and m[-1] < thr:
            drops += 1
            best = max(best, drops)
            if drops >= col["patience"] and fire is None:
                fire = int(ets[n - 1])
        else:
            drops = 0
    return {"first_observation_at": first_obs, "observations": n_obs, "max_consecutive_drops": best,
            "would_fire_at": fire}


i_S = int(np.flatnonzero(ets == S)[0])
k_first = max(col["min_evals"], col["smoothing_window"]) - (i_S + 1)           # window evaluations before counting
out["collapse"] = {"settings": col, "seeded_evaluations_at_S": i_S + 1,
                   "first_counting_window_eval": S + C.EVAL_FREQ * max(1, k_first),
                   "earliest_possible_stop": S + C.EVAL_FREQ * (max(1, k_first) + col["patience"] - 1),
                   "parent_replay_from_S": collapse_replay(i_S),
                   "peak_at_S": None}
w = col["smoothing_window"]
mS = means[: i_S + 1]
rmS = np.array([np.median(mS[i - w + 1: i + 1]) for i in range(w - 1, len(mS))])
okS = ets[: rmS.size] >= col["peak_warmup_timesteps"]
out["collapse"]["peak_at_S"] = float(rmS[okS].max()) if okS.any() else None
out["collapse"]["armed_at_S"] = bool(okS.any() and rmS[okS].max() >= col["peak_floor"])
print(json.dumps(out, indent=1, default=str))
