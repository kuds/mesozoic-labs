"""BS2 design input: the PARENT's own data over BS2's window (S = 2,097,152, E = 3,538,944), read only.

The parent is r10 stance6 s42 (the F2 copies, the parent's own records; no continuation exists).  Computes, from the
F2 policy_probe.npz.copy1 (numpy, allow_pickle False) and eval_episodes.csv.copy1, each checked against FMAN sha256:
  - B_S: the mean of rollout_mu_mean[:, 14] over the parent's 8 rows ending at S (update_timesteps S - 7*8192 ... S),
    float32 widened to float64, averaged in exact rational arithmetic (as score_bs.py does);
  - the parent's primary at E: the same over the window's last 8 update rows before E (S + 8192 k, k = 168..175);
  - the trajectory: the 8-row mean ending at each 131,072 evaluation step in (S, E], and the first 8 window rows;
  - BS's B_S at 1,048,576 recomputed (a reproduction check against BS criteria / check/stat/recompute.out);
  - the statue_settled bank command (bank_action_now[:, i, 14], i = 0 and 5, their mean), report-only form;
  - the parent's ledger tail command CMD_L (read_bs.ledger_direct's definition: per evaluation, the mean over its
    episodes of tail_3_pitch_act_command_mean), with over-bar counts (saturated_actuator_fraction > 0.10) per
    evaluation in (S, E] and a little beyond, for the record.
Usage: PY -I -B parent_window.py  (prints; no file written).
"""
import csv
import hashlib
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
F2 = S1 / "foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance"
FMAN = S1 / "foot/stance6/fetch/manifest.csv"
PROBE = F2 / "policy_probe.npz.copy1"
LEDGER = F2 / "eval_episodes.csv.copy1"
STEP = 8192
GRID = 131072
S, E = 2097152, 3538944
S_BS, E_BS = 1048576, 2490368
TAIL_ACTS = ("tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fman_sha(stored_as):
    with open(FMAN, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("stored_as") == str(stored_as) and r.get("status") == "COPY"]
    assert len(rows) == 1, (stored_as, len(rows))
    return rows[0]["sha256"]


for p in (PROBE, LEDGER):
    got, want = sha(p), fman_sha(p)
    print(f"sha256 {p.name}: {got} FMAN {want} equal {got == want}")
    assert got == want

with np.load(PROBE, allow_pickle=False) as z:
    u = z["update_timesteps"].astype(np.int64)
    mu = z["rollout_mu_mean"][:, 14]
    bank = z["bank_action_now"][:, :, 14]
    labels = [str(x) for x in z["bank_labels"]]
    nrows = len(u)
print(f"probe rows {nrows}, update_timesteps {int(u[0])} .. {int(u[-1])}; unique {len(set(u.tolist())) == nrows}; "
      f"all on the 8,192 grid {bool(np.all(u % STEP == 0))}; contiguous {bool(np.all(np.diff(u) == STEP))}")
print(f"rollout_mu_mean dtype {mu.dtype}; bank_labels {labels}")


def rows_for(steps):
    idx = []
    for s in steps:
        w = np.flatnonzero(u == s)
        assert len(w) == 1, (s, len(w))
        idx.append(int(w[0]))
    return idx


def exact_mean(idx, col=None):
    col = mu if col is None else col
    vals = [Fraction(float(np.float64(col[i]))) for i in idx]
    assert all(math.isfinite(float(v)) for v in vals)
    return sum(vals) / len(vals)


def eight_ending_at(t):
    return [t - STEP * k for k in range(7, -1, -1)]


# BS reproduction
bs_bs = exact_mean(rows_for(eight_ending_at(S_BS)))
bs_primary = exact_mean(rows_for([E_BS - STEP * k for k in range(8, 0, -1)]))
print(f"\nBS reproduction: B_S(1,048,576) {float(bs_bs)!r} (BS criteria -0.03157372563146055); parent primary at "
      f"2,490,368 {float(bs_primary)!r} (BS 0.19955920986831188); rise {float(bs_primary - bs_bs)!r}")

# BS2
bs_rows = rows_for(eight_ending_at(S))
B_S = exact_mean(bs_rows)
prim_steps = [E - STEP * k for k in range(8, 0, -1)]
prim_rows = rows_for(prim_steps)
P_E = exact_mean(prim_rows)
first_steps = [S + STEP * k for k in range(1, 9)]
first8 = exact_mean(rows_for(first_steps))
print(f"\nBS2 B_S: rows {bs_rows[0]}..{bs_rows[-1]} (update_timesteps {eight_ending_at(S)[0]:,} .. {S:,})")
print(f"  B_S exact {B_S.numerator}/{B_S.denominator} = {float(B_S)!r}")
print(f"  per-row values: {[round(float(mu[i]), 5) for i in bs_rows]}")
print(f"BS2 parent primary at E: rows {prim_rows[0]}..{prim_rows[-1]} (update_timesteps {prim_steps[0]:,} .. "
      f"{prim_steps[-1]:,})")
print(f"  primary exact {P_E.numerator}/{P_E.denominator} = {float(P_E)!r}")
print(f"  per-row values: {[round(float(mu[i]), 5) for i in prim_rows]}")
print(f"  RISE (parent primary at E - B_S) = {float(P_E - B_S)!r}")
print(f"  first 8 window rows ({first_steps[0]:,} .. {first_steps[-1]:,}) mean {float(first8)!r}")
n_upd = (E - S) // STEP - 1
print(f"  window: (E - S) = {E - S:,} steps = {(E - S) // STEP} rollouts, {n_upd} updates; window rows "
      f"{S + STEP:,} .. {S + STEP * n_upd:,}; E on the 131,072 grid: {E % GRID == 0} (E / 131,072 = {E // GRID})")
win = rows_for([S + STEP * k for k in range(1, n_upd + 1)])
print(f"  all {len(win)} parent window rows finite: {bool(np.all(np.isfinite(mu[win])))}")

print("\nTrajectory: the parent's 8-row mean of rollout_mu_mean[:, 14] ending at each evaluation step (rise from B_S)")
traj = {}
for t in range(S, E + 1, GRID):
    end = t if t < E else E - STEP          # at E itself there is no row (the last row is E - 8,192)
    m = exact_mean(rows_for(eight_ending_at(end)))
    traj[t] = m
    print(f"  {t:>9,} (rows ending {end:,}): {float(m):+.5f}  rise {float(m - B_S):+.5f}")

print("\nBeyond E (for context only; not the window): 8-row means ending at")
for t in (3670016 - STEP, 4194304 - STEP):
    if (u == t).sum() == 1:
        m = exact_mean(rows_for(eight_ending_at(t)))
        print(f"  {t:>9,}: {float(m):+.5f}  rise {float(m - B_S):+.5f}")

# within-run row-to-row noise of the 8-row mean: SD of the 8 single rows at S and at E
for name, idx in (("at S", bs_rows), ("at E", prim_rows)):
    v = mu[idx].astype(np.float64)
    print(f"row SD {name}: {v.std(ddof=1):.5f} (SE of the 8-row mean if rows were independent "
          f"{v.std(ddof=1) / math.sqrt(8):.5f})")

# statue_settled bank command, report-only form (bank rows 0 and 5)
ss_idx = [i for i, lab in enumerate(labels) if "statue_settled" in lab]
print(f"\nstatue_settled bank rows {ss_idx}")
ssm = bank[:, ss_idx].astype(np.float64).mean(axis=1)
for name, steps in (("8 rows ending at S", eight_ending_at(S)), ("last 8 rows before E", prim_steps)):
    idx = rows_for(steps)
    print(f"  statue_settled tail_3 command, {name}: {float(np.mean(ssm[idx])):+.4f}")

# ledger CMD_L
with open(LEDGER, newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
by = {}
for r in rows:
    try:
        t = float(r["timesteps"])
    except (TypeError, ValueError, KeyError):
        continue
    if math.isfinite(t) and t == int(t):
        by.setdefault(int(t), []).append(r)


def num(r, k):
    try:
        v = float(r.get(k))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


print("\nParent ledger CMD_L (mean over the evaluation's episodes of tail_3_pitch_act_command_mean; read_bs "
      "ledger_direct), over-bar episodes (saturated_actuator_fraction > 0.10): any / tail / tail_3")
for t in sorted(by):
    if not (S - GRID <= t <= E + 4 * GRID):
        continue
    rs = by[t]
    cmd = [c for c in (num(r, "tail_3_pitch_act_command_mean") for r in rs) if c is not None]
    over = [r for r in rs if (num(r, "saturated_actuator_fraction") or 0.0) > 0.10]
    ntail = sum(1 for r in over if r.get("saturated_actuator") in TAIL_ACTS)
    nt3 = sum(1 for r in over if r.get("saturated_actuator") == "tail_3_pitch_act")
    rets = [c for c in (num(r, "episode_return") for r in rs) if c is not None]
    tag = " <- S" if t == S else (" <- E" if t == E else ("" if t <= E else "  (beyond E)"))
    print(f"  {t:>9,}  n {len(rs):>2}  CMD_L {sum(cmd) / len(cmd):+.4f}  max {max(cmd):+.4f}  over {len(over)} "
          f"tail {ntail} t3 {nt3}  mean return {sum(rets) / len(rets) if rets else float('nan'):.1f}{tag}")
print("ledger columns include episode_return:", "episode_return" in rows[0])
