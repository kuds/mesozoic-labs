"""Independent window arithmetic and parent-record coverage (check_build).  Reads only; no BS2 tool imported.

argv: F2_STAGE_DIR (the copies/01_stance folder)
"""
import csv
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
S, LENGTH = 2_097_152, 1_441_792
E = S + LENGTH
R8, EV, SV = 4 * 2048, 131_072, 262_144
out = {"S": S, "E": E, "S_is_2^21": S == 2 ** 21, "E_div_EV": [E // EV, E % EV], "E_div_SV": [E // SV, E % SV],
       "S_mod_SV": S % SV, "LENGTH_le_1.5M": LENGTH <= 1_500_000}
out["evals"] = [S + EV * k for k in range(1, LENGTH // EV + 1)]
out["n_evals"] = len(out["evals"])
out["rollouts"] = LENGTH // R8
out["update_rows_in_(S,E)"] = [S + R8, E - R8, len(range(S + R8, E, R8))]
out["saved_steps_periodic_in_(S,E)"] = [t for t in range(S + 1, E) if t % SV == 0]
out["F2_pairs_in_[S,E]"] = [t for t in range(S, E + 1) if t % 1_048_576 == 0]
out["validate_until"] = S + 2 * EV

z = np.load(D / "policy_probe.npz.copy1", allow_pickle=False)
ut = z["update_timesteps"].astype(np.int64)
out["probe"] = {"rows": int(len(ut)), "first": int(ut[0]), "last": int(ut[-1]),
                "spaced_8192": bool(np.all(np.diff(ut) == R8)), "row_at_S": np.flatnonzero(ut == S).tolist(),
                "rows_le_S": int(np.sum(ut <= S)), "rows_in_(S,E)": int(np.sum((ut > S) & (ut < E))),
                "row_at_E": np.flatnonzero(ut == E).tolist(), "rows_ge_E": int(np.sum(ut >= E)),
                "window_rows_idx": [int(np.flatnonzero(ut == S + R8)[0]), int(np.flatnonzero(ut == E - R8)[0])],
                "ent_dtype": str(z["ent_coef"].dtype), "lr_dtype": str(z["learning_rate"].dtype)}
w = (ut > S) & (ut < E)
ent, lr = z["ent_coef"][w].astype(np.float64), z["learning_rate"][w].astype(np.float64)
t = ut[w].astype(np.float64)
cand = {}
for name, tt in (("t", t), ("t-8192", t - R8)):
    e = 0.005 + (0.0 - 0.005) * np.minimum(1.0, tt / 7e6)
    pr = 1.0 - tt / 12e6
    lrf = 1e-5 + (3e-5 - 1e-5) * pr
    cand[name] = {"ent_max_abs": float(np.max(np.abs(e.astype(z["ent_coef"].dtype) - z["ent_coef"][w]))),
                  "lr_max_abs": float(np.max(np.abs(lrf.astype(z["learning_rate"].dtype) - z["learning_rate"][w]))),
                  "ent_max_abs_f64": float(np.max(np.abs(e - ent))), "lr_max_rel_f64": float(np.max(np.abs(lrf - lr) / lr))}
out["schedule_candidates"] = cand
out["ent_first_last"] = [float(ent[0]), float(ent[-1])]
out["lr_first_last"] = [float(lr[0]), float(lr[-1])]
mu14 = z["rollout_mu_mean"][:, 14].astype(np.float64)
i8 = np.flatnonzero((ut > S - 8 * R8) & (ut <= S))
out["B_S"] = {"rows": i8.tolist(), "steps": ut[i8].tolist(),
              "mean_float": float(np.mean(mu14[i8])),
              "mean_exact": float(sum(Fraction(float(x)) for x in mu14[i8]) / 8)}
i8b = np.flatnonzero((ut > 1_048_576 - 8 * R8) & (ut <= 1_048_576))
out["B_S_at_1048576_exact"] = float(sum(Fraction(float(x)) for x in mu14[i8b]) / 8)
iE = np.flatnonzero((ut > E - 9 * R8) & (ut < E))
out["parent_last8_before_E"] = {"rows": iE.tolist(), "steps": ut[iE].tolist(),
                                "mean_exact": float(sum(Fraction(float(x)) for x in mu14[iE]) / 8)}
out["parent_finite_window_mu14"] = bool(np.all(np.isfinite(mu14[w])))

ev = np.load(D / "evaluations.npz.copy1", allow_pickle=False)
ts = ev["timesteps"].astype(np.int64)
res = ev["results"]
inwin = [int(x) for x in ts if S < x <= E]
out["evaluations_npz"] = {"keys": list(ev.keys()), "n": int(len(ts)), "first": int(ts[0]), "last": int(ts[-1]),
                          "in_(S,E]": inwin, "equal_expected": inwin == out["evals"],
                          "episodes_per_eval_in_window": sorted({int(res[i].shape[0]) for i in range(len(ts))
                                                                 if S < ts[i] <= E})}
with open(D / "eval_episodes.csv.copy1", newline="") as fh:
    rd = csv.DictReader(fh)
    cols = rd.fieldnames
    tcol = next(c for c in ("timesteps", "timestep", "eval_timesteps", "num_timesteps") if c in cols)
    cnt = {}
    has_cmd = "tail_3_pitch_act_command_mean" in cols
    cmd_bad = 0
    for row in rd:
        s = int(float(row[tcol]))
        if S < s <= E:
            cnt[s] = cnt.get(s, 0) + 1
            if has_cmd:
                try:
                    if not np.isfinite(float(row["tail_3_pitch_act_command_mean"])):
                        cmd_bad += 1
                except ValueError:
                    cmd_bad += 1
out["eval_episodes_csv"] = {"time_col": tcol, "per_eval_in_window": cnt,
                            "evals_equal_expected": sorted(cnt) == out["evals"],
                            "has_tail3_cmd_col": has_cmd, "tail3_cmd_nonfinite_in_window": cmd_bad}
with open(D / "train_episodes.csv.copy1", newline="") as fh:
    rd = csv.DictReader(fh)
    cols = rd.fieldnames
    tcol = next((c for c in ("timesteps", "timestep", "num_timesteps", "global_step") if c in cols), None)
    vals = [int(float(r[tcol])) for r in rd] if tcol else []
out["train_episodes_csv"] = {"time_col": tcol, "rows": len(vals), "max": max(vals) if vals else None,
                             "rows_in_(S,E]": sum(1 for v in vals if S < v <= E)}
gp = np.load(D / "gate_progress.npz.copy1", allow_pickle=False)
gk = [k for k in gp.keys() if "timestep" in k]
out["gate_progress"] = {"keys_timestep": gk, **{k: [int(np.min(gp[k])), int(np.max(gp[k])),
                                                     int(np.sum((gp[k] > S) & (gp[k] <= E)))] for k in gk}}
print(json.dumps(out, indent=1, default=str))
