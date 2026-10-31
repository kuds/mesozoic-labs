"""Design reviewer: B_S, the parent's primary at E and its trajectory, recomputed from the F2 probe (read only).
Independent of criteria_work/parent_window.py and score_bs.py (written without reading their code bodies).
Usage: PY -I -B rd_parent.py > rd_parent.out
"""
import csv
import hashlib
import sys
from fractions import Fraction

import numpy as np

S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
F2 = S1 + "/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance"
PROBE = F2 + "/policy_probe.npz.copy1"
LEDGER = F2 + "/eval_episodes.csv.copy1"
S, E, ROLL, EV = 2_097_152, 3_538_944, 8_192, 131_072


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


print("probe sha256", sha(PROBE))
print("ledger sha256", sha(LEDGER))
z = np.load(PROBE, allow_pickle=False)
ut = z["update_timesteps"]
mu14 = z["rollout_mu_mean"][:, 14]
print("rows", len(ut), "first", ut[0], "last", ut[-1], "unique", len(set(ut.tolist())) == len(ut),
      "contiguous on 8192 grid", bool(np.all(np.diff(ut) == ROLL)) and ut[0] == ROLL)


def mean_rows(steps):
    vals = []
    for s in steps:
        idx = np.nonzero(ut == s)[0]
        assert len(idx) == 1, s
        v = mu14[idx[0]]
        assert np.isfinite(v)
        vals.append(Fraction(float(v)))
    return sum(vals, Fraction(0)) / len(vals), [int(np.nonzero(ut == s)[0][0]) for s in steps]


base_steps = [S - ROLL * k for k in range(7, -1, -1)]
bs, rows = mean_rows(base_steps)
print("B_S steps", base_steps[0], "...", base_steps[-1], "rows", rows[0], "-", rows[-1])
print("B_S exact", bs, "float", float(bs), "repr", repr(float(bs)))
print("B_S == 0.1265954738482833 ?", Fraction(0.1265954738482833) == bs)
prim_steps = [E - ROLL * k for k in range(8, 0, -1)]
pe, rows = mean_rows(prim_steps)
print("primary steps", prim_steps, "rows", rows)
print("parent primary at E exact", pe, "float", repr(float(pe)))
print("rise", repr(float(pe - bs)), "exact", pe - bs)
# window rows (S, E): S + 8192k, k = 1..175
win = [S + ROLL * k for k in range(1, (E - S) // ROLL)]
print("window rows", len(win), win[0], win[-1], "all finite",
      all(np.isfinite(mu14[np.nonzero(ut == s)[0][0]]) for s in win))
# does a row exist at E?  (the rollout ending at E has no update in the continuation)
print("parent row at E exists:", bool(np.any(ut == E)), "(the parent's update at E exists; a continuation's does not)")
print("trajectory: 8-row mean ending at t (t < E), at E the primary's rows")
traj = {}
for t in range(S + EV, E + 1, EV):
    end = t if t < E else E - ROLL
    v, _ = mean_rows([end - ROLL * k for k in range(7, -1, -1)])
    traj[t] = v
    print(f"  {t:>9} {float(v):+.17f} rise {float(v - bs):+.5f}  exact {v}")
# the alternative definition at E: rows ending at E itself (the parent has a row at E)
v_alt, _ = mean_rows([E - ROLL * k for k in range(7, -1, -1)])
print("alt at E with the parent's own row at E (rows ending at E):", repr(float(v_alt)))
# row SDs
b_rows = np.array([float(mu14[np.nonzero(ut == s)[0][0]]) for s in base_steps])
p_rows = np.array([float(mu14[np.nonzero(ut == s)[0][0]]) for s in prim_steps])
print("row SD at S", b_rows.std(ddof=1), "at E", p_rows.std(ddof=1))
# first 8 window rows
f8, _ = mean_rows([S + ROLL * k for k in range(1, 9)])
print("first 8 window rows", float(f8))
# statue_settled: not recomputed here (bank rows' labels)
print("bank labels", list(z["bank_labels"]))
# ledger: CMD_L per evaluation (mean of tail_3_pitch_act_command_mean over the evaluation's episodes)
with open(LEDGER, newline="", encoding="utf-8") as fh:
    rd = csv.DictReader(fh)
    cols = rd.fieldnames
    rows_l = list(rd)
print("ledger columns with tail_3:", [c for c in cols if "tail_3" in c][:12])
per = {}
for r in rows_l:
    try:
        t = int(float(r["timesteps"]))
    except (KeyError, ValueError):
        continue
    per.setdefault(t, []).append(r)
for t in range(S, 3_932_160 + 1, EV):
    rs = per.get(t, [])
    cm = [float(r["tail_3_pitch_act_command_mean"]) for r in rs if r.get("tail_3_pitch_act_command_mean") not in (None, "")]
    print(f"  ledger {t:>9}: n {len(rs)} CMD_L {np.mean(cm) if cm else float('nan'):+.4f}")
sys.stdout.flush()
