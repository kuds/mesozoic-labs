"""LB design input: the PARENT's own records over LB's window (S = 2,097,152, E = 12,001,280), read only.

The parent is r10 stance6 s42 (the F2 copies; no continuation exists).  From the F2 policy_probe.npz.copy1 (numpy,
allow_pickle False), eval_episodes.csv.copy1 and evaluations.npz.copy1, each checked against its FMAN COPY row:
  PROBE   B_S (BS2's, reproduced); the on-policy tail_3 command (rollout_mu_mean[:, 14]) as the 8-row mean ending at
          every evaluation step in (S, E] (rise from B_S); at E the 8 rows before E (the continuation-comparable form:
          a continuation makes no update on the rollout ending at E, so its last row is E - 8,192); the same at
          3,538,944 (BS2's E); the window maximum; per actuator (all 15) the last-8-before-E mean, the D-1 window
          means and the largest |trailing 8-row mean| with its step; the training-row saturated share per actuator;
          std of action 14, the tail_3 normaliser, the schedules; the statue_settled bank tail_3 command.
  LEDGER  per evaluation in (S, E]: n, CMD_L (mean of tail_3_pitch_act_command_mean; read_bs.ledger_direct's form),
          episodes with saturated_actuator_fraction > 0.10 by named actuator and sign (positive share >= 0.5), flagged
          (>= 5 of 10) and tail-flagged evaluations, alarms (the second of two consecutive) and spells (a simple
          re-derivation for the design, NOT the pinned M5: compared below with LB/measure's pinned-tool reading);
          every actuator's per-evaluation command mean (<name>_command_mean), its largest |mean| and the evaluations
          with |mean| >= 0.60; CMD_L firsts (0.30 / 0.60 / 0.90) and at the last evaluation (11,927,552); returns
          and falls (length < 1000) per evaluation.
  PINNED  the measurement builder's reading of the same parent with the round's pinned tools
          (LB/measure/validate/stance6_r10LTSB_s42/measure_lb.json): alarms, flagged evaluations, classes; and the
          early-warning loop values at the parent's F2 pairs (S2/com_revisit/notes/early_warning/ew_r10.jsonl).
Usage: PY -I -B parent_long.py  (prints; writes nothing).  Paths are derived from this file's place (LB/criteria_work).
"""
import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

CW = Path(__file__).resolve().parent
S2 = CW.parents[2]
S1 = S2.parent.parent / "0fdf74a4-2989-56c9-a773-92101004d7de" / "scratchpad"
F2 = S1 / "foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance"
FMAN = S1 / "foot/stance6/fetch/manifest.csv"
MEAS = S2 / "r10round/long_blind/measure/validate/stance6_r10LTSB_s42/measure_lb.json"
EWJ = S2 / "com_revisit/notes/early_warning/ew_r10.jsonl"
STEP, GRID, SAVE = 8192, 131072, 262144
S, E = 2097152, 12001280
E_BS2 = 3538944
D1 = ((0, 4194304), (4194304, 8388608), (8388608, E + 1))
TAIL = ("tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")


def short(p):
    s = str(p)
    for k, v in (("S1", S1), ("S2", S2)):
        s = s.replace(str(v), k)
    return s


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fman_sha(stored_as):
    with open(FMAN, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("stored_as") == str(stored_as) and r.get("status") == "COPY"]
    assert len(rows) == 1, (short(stored_as), len(rows))
    return rows[0]["sha256"]


PROBE = F2 / "policy_probe.npz.copy1"
LEDGER = F2 / "eval_episodes.csv.copy1"
EVALS = F2 / "evaluations.npz.copy1"
print("== INPUTS (sha256 against FMAN COPY rows)")
for p in (PROBE, LEDGER, EVALS):
    got, want = sha(p), fman_sha(p)
    print(f"  {short(p)}: {got} FMAN {want} equal {got == want}")
    assert got == want

# ------------------------------------------------------------------------------------------------------------ PROBE
with np.load(PROBE, allow_pickle=False) as z:
    u = z["update_timesteps"].astype(np.int64)
    MU = z["rollout_mu_mean"].astype(np.float64)
    SAT = z["rollout_action_saturated_share"].astype(np.float64)
    STD = z["std_per_actuator"].astype(np.float64)
    OM, OV = z["obs_rms_mean"].astype(np.float64), z["obs_rms_var"].astype(np.float64)
    ENT, LR = z["ent_coef"].astype(np.float64), z["learning_rate"].astype(np.float64)
    BANK = z["bank_action_now"][:, :, 14].astype(np.float64)
    BLAB = [str(x) for x in z["bank_labels"]]
mu = z_mu14 = MU[:, 14]
idx_of = {int(t): i for i, t in enumerate(u)}
assert len(idx_of) == len(u)
win = [S + STEP * k for k in range(1, (E - S) // STEP + 1)]
print(f"\n== PROBE: {len(u)} rows, {int(u[0]):,} .. {int(u[-1]):,}; window rows (S, E] present "
      f"{sum(t in idx_of for t in win)} of {len(win)} (incl. the parent's row at E, which a continuation lacks)")


def rows(ts):
    return [idx_of[t] for t in ts]


def exact_mean(ix, col):
    vals = [Fraction(float(np.float32(col[i]))) for i in ix]
    return sum(vals) / len(vals)


def eight_ending(t):
    return [t - STEP * k for k in range(7, -1, -1)]


B_S = exact_mean(rows(eight_ending(S)), mu)
print(f"  B_S (8 rows ending at S) exact {B_S.numerator}/{B_S.denominator} = {float(B_S)!r}")
pE = exact_mean(rows(eight_ending(E - STEP)), mu)
pE_par = exact_mean(rows(eight_ending(E)), mu)
p35 = exact_mean(rows(eight_ending(E_BS2 - STEP)), mu)
print(f"  primary form at E (8 rows {E - 8 * STEP:,} .. {E - STEP:,}, the last 8 updates before E): {float(pE):+.5f} "
      f"rise {float(pE - B_S):+.5f}; the parent's own 8 rows ending at E (incl. its update row at E): "
      f"{float(pE_par):+.5f}")
print(f"  at 3,538,944 (BS2's E; rows ending 3,530,752): {float(p35):+.5f} rise {float(p35 - B_S):+.5f}")

trail = np.full(len(u), np.nan)
for i in range(7, len(u)):
    trail[i] = MU[i - 7:i + 1, 14].mean()
wi = [idx_of[t] for t in win]
imax = max(wi, key=lambda i: trail[i])
imin_late = min((i for i in wi if u[i] >= 4194304), key=lambda i: trail[i])
print(f"  window max of the trailing 8-row mean: {trail[imax]:+.4f} at {int(u[imax]):,}; its minimum from 4,194,304: "
      f"{trail[imin_late]:+.4f} at {int(u[imin_late]):,}")
for thr in (0.30, 0.60, 0.90):
    first = next((int(u[i]) for i in wi if trail[i] >= thr), None)
    print(f"  first trailing 8-row mean >= {thr:.2f}: {first if first is None else f'{first:,}'}")
nrow_ge = {thr: sum(1 for i in wi if trail[i] >= thr) for thr in (0.6, 0.9)}
print(f"  window rows with the trailing 8-row mean >= 0.60: {nrow_ge[0.6]} of {len(wi)}; >= 0.90: {nrow_ge[0.9]}")

print("\n  TRAJECTORY: the 8-row mean of rollout_mu_mean[:, 14] ending at each evaluation step (rise from B_S); at E the"
      " 8 rows before E")
traj = {}
for t in list(range(S + GRID, E, GRID)) + [E]:
    end = t if t < E else E - STEP
    m = exact_mean(rows(eight_ending(end)), mu)
    traj[t] = float(m)
line = []
for k, (t, m) in enumerate(traj.items()):
    line.append(f"{t / 1e6:6.3f}M {m:+.3f}")
    if len(line) == 6:
        print("   " + " | ".join(line))
        line = []
if line:
    print("   " + " | ".join(line))

NAMES = None
with open(LEDGER, newline="", encoding="utf-8") as fh:
    rd = csv.DictReader(fh)
    LROWS = list(rd)
    HEAD = rd.fieldnames
NAMES = [c[: -len("_command_mean")] for c in HEAD if c.endswith("_command_mean")]
assert len(NAMES) == 15 and NAMES[14] == "tail_3_pitch_act", NAMES
print("\n  PER ACTUATOR (probe, on-policy rollout means): last 8 before E | D-1 window means [S,4.19M) [4.19,8.39M)"
      " [8.39M,E) | largest |trailing 8-row mean| (step) | training-row saturated share: window mean, max trailing-8")
for a, n in enumerate(NAMES):
    last8 = float(exact_mean(rows(eight_ending(E - STEP)), MU[:, a]))
    wm = []
    for lo, hi in D1:
        sel = [i for i in wi if max(lo, S) < u[i] < hi and u[i] < E]
        wm.append(float(MU[sel, a].mean()) if sel else float("nan"))
    tr = np.array([MU[i - 7:i + 1, a].mean() for i in wi])
    j = int(np.argmax(np.abs(tr)))
    st = np.array([SAT[i - 7:i + 1, a].mean() for i in wi])
    print(f"   {n:20s} {last8:+.3f} | {wm[0]:+.3f} {wm[1]:+.3f} {wm[2]:+.3f} | {tr[j]:+.3f} ({int(u[wi[j]]):,}) | "
          f"{float(SAT[wi, a].mean()):.3f} {float(st.max()):.3f}")

for t in (S, E_BS2 - STEP, 7000000 // STEP * STEP, E - STEP):
    i = idx_of[t]
    print(f"  row {t:,}: tail_3 normaliser mean {math.degrees(OM[i, 6]):+.3f} deg SD "
          f"{math.degrees(math.sqrt(OV[i, 6])):.3f} deg; std14 {STD[i, 14]:.4f}; ent_coef {ENT[i]:.6f}; "
          f"learning_rate {LR[i]:.4e}")
z0 = next(int(u[i]) for i in wi if ENT[i] == 0.0)
print(f"  ent_coef first 0.0 at {z0:,}; rows with ent_coef 0 in the window: {sum(1 for i in wi if ENT[i] == 0.0)}")
ss = [i for i, lab in enumerate(BLAB) if lab == "statue_settled"]
ssm = BANK[:, ss].mean(axis=1)
for nm, t in (("S", S), ("3,530,752", E_BS2 - STEP), ("E - 8,192", E - STEP)):
    print(f"  statue_settled bank tail_3 command, 8 rows ending at {nm}: "
          f"{float(np.mean(ssm[rows(eight_ending(t))])):+.4f}")
tr_ss = np.array([ssm[i - 7:i + 1].mean() for i in wi])
print(f"  statue_settled 8-row max over the window {tr_ss.max():+.4f} at {int(u[wi[int(np.argmax(tr_ss))]]):,}")

# ------------------------------------------------------------------------------------------------------------ LEDGER


def num(r, k):
    try:
        v = float(r.get(k))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


by = {}
for r in LROWS:
    t = num(r, "timesteps")
    if t is not None and t == int(t):
        by.setdefault(int(t), []).append(r)
evs = sorted(t for t in by if S < t <= E)
print(f"\n== LEDGER: {len(LROWS)} rows, {len(by)} evaluations ({min(by):,} .. {max(by):,}); in (S, E]: {len(evs)} "
      f"({evs[0]:,} .. {evs[-1]:,}); rows per evaluation {sorted({len(by[t]) for t in evs})}; before or at S: "
      f"{sum(1 for t in by if t <= S)} evaluations")
per = {}
for t in sorted(by):
    rs = by[t]
    cmd = [c for c in (num(r, "tail_3_pitch_act_command_mean") for r in rs) if c is not None]
    over = []
    for r in rs:
        if (num(r, "saturated_actuator_fraction") or 0.0) > 0.10:
            pos = num(r, "saturated_actuator_positive_share")
            over.append((r.get("saturated_actuator") or "?", "?" if pos is None else ("+" if pos >= 0.5 else "-")))
    acts = {}
    for n in NAMES:
        v = [c for c in (num(r, f"{n}_command_mean") for r in rs) if c is not None]
        acts[n] = sum(v) / len(v) if v else None
    rets = [c for c in (num(r, "return") for r in rs) if c is not None]
    lens = [c for c in (num(r, "length") for r in rs) if c is not None]
    per[t] = dict(n=len(rs), cmd_l=sum(cmd) / len(cmd) if cmd else None, over=over, acts=acts,
                  ret=sum(rets) / len(rets) if rets else None,
                  falls=[(int(num(r, "episode") or -1), int(num(r, "length") or -1), r.get("cause"))
                         for r in rs if (num(r, "length") or 1000) < 1000])
flag = [t for t in evs if len(per[t]["over"]) >= 5]
tflag = [t for t in evs if sum(1 for a, _ in per[t]["over"] if a in TAIL) >= 5]
alarms = [t for t in flag if t - GRID in flag]
talarms = [t for t in tflag if t - GRID in tflag]


def spells(ts):
    out, cur = [], []
    for t in ts:
        if cur and t - cur[-1] == GRID:
            cur.append(t)
        else:
            if cur:
                out.append(cur)
            cur = [t]
    if cur:
        out.append(cur)
    return [(c[0], c[-1], len(c)) for c in out]


first_over = next((t for t in evs if per[t]["over"]), None)
print(f"  my re-derivation (fraction > 0.10 only): over-bar episodes in (S, E] {sum(len(per[t]['over']) for t in evs)}; "
      f"first {first_over:,}; flagged {len(flag)}, tail-flagged {len(tflag)}; alarms {len(alarms)} (first "
      f"{alarms[0] if alarms else None:,}), tail alarms {len(talarms)}")
print(f"  flagged spells (first, last, evaluations): {[(f'{a:,}', f'{b:,}', k) for a, b, k in spells(flag)]}")
byact = {}
for t in evs:
    for a, sg in per[t]["over"]:
        byact.setdefault((a, sg), 0)
        byact[(a, sg)] += 1
print(f"  over-bar episodes by actuator and sign: {byact}")
for lo, hi in D1:
    print(f"    D-1 window [{max(lo, S):,}, {min(hi, E):,}): over-bar {sum(len(per[t]['over']) for t in evs if lo <= t < hi)}")
cl = [(t, per[t]["cmd_l"]) for t in evs]
for thr in (0.30, 0.60, 0.90):
    f = next((t for t, c in cl if c is not None and c >= thr), None)
    k = sum(1 for _, c in cl if c is not None and c >= thr)
    print(f"  CMD_L first >= {thr:.2f}: {f if f is None else f'{f:,}'}; evaluations >= {thr:.2f}: {k} of {len(cl)}")
print(f"  CMD_L at S {per[S]['cmd_l']:+.4f}; at 3,538,944 {per[E_BS2]['cmd_l']:+.4f}; at the last evaluation "
      f"{evs[-1]:,} {per[evs[-1]]['cmd_l']:+.4f}; max {max(c for _, c in cl):+.4f} at "
      f"{max(cl, key=lambda x: x[1])[0]:,}; min from 4,194,304 {min(c for t, c in cl if t >= 4194304):+.4f} at "
      f"{min(((t, c) for t, c in cl if t >= 4194304), key=lambda x: x[1])[0]:,}")
print("  CMD_L per evaluation (step in M: value / over-bar count):")
line = []
for t in evs:
    line.append(f"{t / 1e6:6.3f} {per[t]['cmd_l']:+.3f}/{len(per[t]['over'])}")
    if len(line) == 8:
        print("   " + "  ".join(line))
        line = []
if line:
    print("   " + "  ".join(line))
print("  PER ACTUATOR (ledger command means per evaluation, (S, E]): largest |mean| (step); evaluations with mean >= "
      "+0.60 / <= -0.60; value at the last evaluation")
for n in NAMES:
    vals = [(t, per[t]["acts"][n]) for t in evs if per[t]["acts"][n] is not None]
    t_, v_ = max(vals, key=lambda x: abs(x[1]))
    hi = sum(1 for _, v in vals if v >= 0.60)
    lo = sum(1 for _, v in vals if v <= -0.60)
    print(f"   {n:20s} {v_:+.3f} ({t_:,}); >= +0.60 {hi:2d}; <= -0.60 {lo:2d}; last {per[evs[-1]]['acts'][n]:+.3f}")
rets = [(t, per[t]["ret"]) for t in evs]
falls = [(t, f) for t in evs for f in per[t]["falls"]]
print(f"  returns: mean over the window's evaluations {sum(r for _, r in rets) / len(rets):.1f}; first {rets[0][1]:.1f}, "
      f"last {rets[-1][1]:.1f}; min {min(r for _, r in rets):.1f}; falls (length < 1000) {len(falls)}: {falls[:10]}")

# ------------------------------------------------------------------------------------------------------------ EVALS
with np.load(EVALS, allow_pickle=False) as z:
    et, er, el = z["timesteps"].astype(np.int64), z["results"], z["ep_lengths"]
sel = [i for i, t in enumerate(et) if S < t <= E]
print(f"\n== evaluations.npz: {len(et)} evaluations; in (S, E] {len(sel)}; mean return {float(er[sel].mean()):.1f}; "
      f"episodes shorter than 1000: {int((el[sel] < 1000).sum())}")

# ------------------------------------------------------------------------------------------------------------ PINNED
m = json.loads(MEAS.read_text())
ms = m.get("m5_summary") or {}
c = m.get("classes") or {}
print(f"\n== PINNED (the measurement builder's measure_lb.json of the parent; the round's pinned M5 and classes)")
print(f"  status {m.get('status')}; E {m.get('E'):,}; tail class {c.get('tail_class')}; gate {c.get('gate_class')}; "
      f"outcome {c.get('run_outcome') or c.get('outcome')}")
for k in ("any_alarms", "tail_alarms", "any_flagged", "tail_flagged"):
    v = ms.get(k) or []
    print(f"  {k}: {len(v)}; first {v[0] if v else None}; last {v[-1] if v else None}")
print(f"  any_alarms_by_actuator {ms.get('any_alarms_by_actuator')}")
print(f"  my alarms == pinned any_alarms: {alarms == list(ms.get('any_alarms') or [])}; my tail alarms == pinned: "
      f"{talarms == list(ms.get('tail_alarms') or [])}; my flagged == pinned any_flagged: "
      f"{flag == list(ms.get('any_flagged') or [])}")
print(f"  pinned tail-flagged spells: {[(f'{a:,}', f'{b:,}', k) for a, b, k in spells(list(ms.get('tail_flagged') or []))]}")

# ------------------------------------------------------------------------------------------------------------ EW
print("\n== EARLY-WARNING LOOP VALUES at the three parents' F2 pairs (ew_r10.jsonl; OWN primary sweep; b* = 1 - G - c0,"
      " computed here)")
for k, ln in enumerate(EWJ.read_text().splitlines(), 1):
    d = json.loads(ln)
    run = d.get("run") or d.get("run_id") or d.get("label")
    st = d.get("steps") or d.get("step")
    o = (d.get("own") or {}).get("primary") or {}
    G, c0 = o.get("G"), o.get("c0")
    bs = None if G is None or c0 is None else 1.0 - G - c0
    print(f"  line {k:2d} {run} {st:>10,}: G {G:+.3f} c0 {c0:+.3f} b* {bs:+.3f} own14 {d.get('own14_win_median'):+.3f} "
          f"over-bar own episodes {d.get('n_over_bar')} of {d.get('n_episodes')}")
