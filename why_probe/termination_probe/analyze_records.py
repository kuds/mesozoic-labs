"""Records (read only): PREREG M1.2, M1.3, M1.4, P1.2, P1.3.

    PY -I -B analyze_records.py > analysis_records.txt   (also writes analysis_records.json)

M1.2  parent s42 (F2 train_episodes.csv.copy1): episodes with 1,048,576 < timesteps <= 3,538,944 by cause, per
      262,144-step bin; and timesteps <= 1,048,576 (descriptive).
M1.3  BS2's eight runs' 01_stance/train_episodes.csv: episodes with 2,097,152 < timesteps <= 3,538,944 by cause, per run.
M1.4  evaluation ledgers (parent F2 eval_episodes.csv.copy1, BS2 runs' eval_episodes.csv), the same windows, by cause.
P1.2  parent's tail_contact share in (1,048,576, 3,538,944] <= 2%.
P1.3  per-run tail_contact rate, blind > control, exact one-sided permutation test (70 relabelings); UNTESTABLE with
      fewer than 3 tail_contact episodes over the eight runs.
"""
import csv
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_common as T  # noqa: E402

S, E = 2_097_152, 3_538_944
LO = 1_048_576
BIN = 262_144
RUNS = [f"{a}_{s}" for s in (2501, 2511, 2521, 2531) for a in ("control", "critic_blind_tail3")]


def rows(path):
    with open(path, newline="") as fh:
        yield from csv.DictReader(fh)


def causes(path, lo, hi, ts_key="timesteps"):
    c = Counter()
    lengths = []
    for r in rows(path):
        ts = int(float(r[ts_key]))
        if lo < ts <= hi:
            c[r["cause"]] += 1
            lengths.append(int(float(r["length"])))
    return c, lengths


def share(c, key):
    n = sum(c.values())
    return (c.get(key, 0) / n) if n else float("nan")


out = {}
print("RECORDS (read only).  cause = the recorded end of each episode.\n")
# ------------------------------------------------------------------------------------------------ M1.2
tr = f"{T.F2}/train_episodes.csv.copy1"
ev = f"{T.F2}/eval_episodes.csv.copy1"
print(f"M1.2 parent s42 training records {tr}")
c_win, len_win = causes(tr, LO, E)
c_early, len_early = causes(tr, 0, LO)
n_win = sum(c_win.values())
print(f"  (1,048,576, 3,538,944]: {n_win} episodes; by cause {dict(c_win.most_common())}")
print(f"     tail_contact share {share(c_win, 'tail_contact'):.4%}; any early end share "
      f"{1 - share(c_win, max(c_win, key=c_win.get)):.4%} (largest class = {max(c_win, key=c_win.get)})")
print(f"     median length {sorted(len_win)[len(len_win) // 2] if len_win else None}")
print(f"  (0, 1,048,576] (descriptive): {sum(c_early.values())} episodes; by cause {dict(c_early.most_common())}")
bins = []
lo = LO
while lo < E:
    hi = min(lo + BIN, E)
    c, _ = causes(tr, lo, hi)
    bins.append({"lo": lo, "hi": hi, "n": sum(c.values()), "causes": dict(c)})
    print(f"   bin ({lo:>9,d}, {hi:>9,d}]: n {sum(c.values()):4d}  {dict(c.most_common())}")
    lo = hi
p12 = share(c_win, "tail_contact")
verdict_p12 = "SUPPORTED" if p12 <= 0.02 else "NOT SUPPORTED"
print(f"  P1.2 (prediction: tail_contact <= 2% of training episodes in the window): {p12:.4%} -> {verdict_p12}")
out["M1.2"] = {"window": [LO, E], "n": n_win, "causes": dict(c_win), "early": dict(c_early), "bins": bins,
               "tail_contact_share": p12, "P1.2": verdict_p12}

# ------------------------------------------------------------------------------------------------ M1.3 / P1.3
print(f"\nM1.3 BS2 runs' training records, ({S:,d}, {E:,d}]")
per_run = {}
for name in RUNS:
    p = f"{T.BS2}/runs/{name}/01_stance/train_episodes.csv"
    c, ln = causes(p, S, E)
    n = sum(c.values())
    per_run[name] = {"n": n, "causes": dict(c), "tail_contact_rate": c.get("tail_contact", 0) / n if n else None,
                     "median_length": sorted(ln)[len(ln) // 2] if ln else None}
    print(f"  {name:24s} n {n:4d}  {dict(c.most_common())}")
ctrl = [per_run[f"control_{s}"]["tail_contact_rate"] for s in (2501, 2511, 2521, 2531)]
blnd = [per_run[f"critic_blind_tail3_{s}"]["tail_contact_rate"] for s in (2501, 2511, 2521, 2531)]
tot_tc = sum(per_run[n]["causes"].get("tail_contact", 0) for n in RUNS)
allr = ctrl + blnd
obs = sum(blnd) / 4 - sum(ctrl) / 4
ge = 0
for idx in itertools.combinations(range(8), 4):
    b = [allr[i] for i in idx]
    c_ = [allr[i] for i in range(8) if i not in idx]
    if sum(b) / 4 - sum(c_) / 4 >= obs - 1e-15:
        ge += 1
p13 = ge / 70
if tot_tc < 3:
    v13 = "UNTESTABLE"
else:
    v13 = "SUPPORTED" if p13 <= 0.05 else "NOT SUPPORTED"
print(f"  per-run tail_contact rate: control {['%.4f' % x for x in ctrl]}  blind {['%.4f' % x for x in blnd]}")
print(f"  total tail_contact episodes over the eight runs: {tot_tc}")
print(f"  P1.3 (B-genuine: blind > control): mean difference {obs:+.5f}; exact one-sided permutation p = {ge}/70 = "
      f"{p13:.4f} -> {v13}")
# secondary: any early end (cause other than the truncation label, read from the data as the commonest cause)
any_ctrl, any_blnd = [], []
for s in (2501, 2511, 2521, 2531):
    for arm, lst in (("control", any_ctrl), ("critic_blind_tail3", any_blnd)):
        r = per_run[f"{arm}_{s}"]
        trunc = max(r["causes"], key=r["causes"].get) if r["causes"] else None
        lst.append(1 - r["causes"].get(trunc, 0) / r["n"] if r["n"] else None)
print(f"  (secondary, report only) per-run share of episodes NOT ending in the commonest cause: control "
      f"{['%.4f' % x for x in any_ctrl]}  blind {['%.4f' % x for x in any_blnd]}")
out["M1.3"] = {"window": [S, E], "per_run": per_run, "control_rates": ctrl, "blind_rates": blnd,
               "total_tail_contact": tot_tc, "mean_diff": obs, "perm_p": p13, "P1.3": v13,
               "any_end_share_control": any_ctrl, "any_end_share_blind": any_blnd}

# ------------------------------------------------------------------------------------------------ M1.4
print("\nM1.4 evaluation ledgers (deterministic evaluation episodes), by cause")
c_ev, _ = causes(ev, LO, E)
print(f"  parent F2 eval (1,048,576, 3,538,944]: n {sum(c_ev.values())}  {dict(c_ev.most_common())}")
ev_runs = {}
for name in RUNS:
    c, _ = causes(f"{T.BS2}/runs/{name}/01_stance/eval_episodes.csv", S, E)
    ev_runs[name] = dict(c)
    print(f"  {name:24s} n {sum(c.values()):3d}  {dict(c.most_common())}")
# list the non-truncated evaluation episodes in BS2 (which evaluation, which run)
print("  BS2 evaluation episodes not ending at the horizon (run, timesteps, episode, cause, length):")
for name in RUNS:
    for r in rows(f"{T.BS2}/runs/{name}/01_stance/eval_episodes.csv"):
        ts = int(float(r["timesteps"]))
        if S < ts <= E and str(r.get("full_horizon", "")).strip().lower() not in ("true", "1", "1.0"):
            print(f"     {name} {ts} ep {r['episode']} {r['cause']} len {r['length']}")
out["M1.4"] = {"parent_eval": dict(c_ev), "bs2_eval": ev_runs}
Path(T.ME / "analysis_records.json").write_text(json.dumps(out, indent=1) + "\n")
print("\nwritten analysis_records.json")
