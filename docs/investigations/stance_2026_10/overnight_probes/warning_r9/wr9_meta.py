"""warning_r9 step (c): r9 METADATA only (keys, shapes, labels, bank sha1, row spacing, ledger header/cadence).
Prints no signal value (no bank_action_now, rollout_mu, CMD_L or over-bar value)."""
import hashlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from wr9_apply import sources, F2
for run, (lp, pp) in sources("r9").items():
    print(f"=== {run}")
    with np.load(pp, allow_pickle=False) as d:
        for k in d.files:
            a = d[k]
            print(f"  npz {k:34s} {str(a.dtype):10s} {a.shape}")
        t = np.asarray(d["update_timesteps"]).astype(np.int64)
        print(f"  update_timesteps {t[0]}..{t[-1]} rows {len(t)} spacing {sorted(set(np.diff(t).tolist()))[:5]}")
        print(f"  bank_labels {np.asarray(d['bank_labels']).astype(str).tolist()}")
        bo = np.asarray(d["bank_raw_obs"])
        print(f"  bank_raw_obs sha1 {hashlib.sha1(bo.tobytes()).hexdigest()[:10]} shape {bo.shape} dtype {bo.dtype}")
    cols = pd.read_csv(lp, nrows=0).columns.tolist()
    need = ["timesteps", "saturated_actuator_fraction", "saturated_actuator", "tail_3_pitch_act_command_mean"]
    print(f"  ledger columns {len(cols)}; needed present: {[c in cols for c in need]}")
    ts = pd.read_csv(lp, usecols=["timesteps"])["timesteps"]
    g = ts.value_counts().sort_index()
    print(f"  ledger rows {len(ts)}, evals {len(g)}, timesteps {g.index.min()}..{g.index.max()}, "
          f"cadence {sorted(set(np.diff(g.index.values).tolist()))[:5]}, episodes/eval {sorted(set(g.values.tolist()))}")
with np.load(F2 / "stance6_r10LTSB_s42/copies/01_stance/policy_probe.npz.copy1", allow_pickle=False) as d:
    bo = np.asarray(d["bank_raw_obs"])
    print(f"r10 bank_raw_obs sha1 {hashlib.sha1(bo.tobytes()).hexdigest()[:10]} shape {bo.shape} dtype {bo.dtype}")
