"""Verifier's own ledger + probe event check (no simulation, read only).  Does not import ew_* code.

    python -B vew_events.py > vew_events.txt

Ledger rows: 10 deterministic eval episodes per 131,072 steps.  over-bar = saturated_actuator_fraction > 0.10 on the
named (most saturated) actuator; tail over-bar = that name is a tail actuator.  CMD_L = mean over the evaluation's
episodes of tail_3_pitch_act_command_mean.  Probe: bank_action_now[:, statue_settled rows, 14] mean, 8-row trailing
mean (causal).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
ECS = SCR / "r10round/ecs/runs"
TAIL = {"tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"}
R10 = ("s42", "s60", "s80")
ECSR = sorted(p.name for p in ECS.iterdir() if p.is_dir())
S = {"s42": 3145728, "s80": 4194304}


def led(run):
    f = (F2 / f"stance6_r10LTSB_{run}/copies/01_stance/eval_episodes.csv.copy1") if run in R10 else (ECS / run / "01_stance/eval_episodes.csv")
    d = pd.read_csv(f)
    d["ob"] = d["saturated_actuator_fraction"] > 0.10
    d["tob"] = d["ob"] & d["saturated_actuator"].isin(TAIL)
    g = d.groupby("timesteps")
    e = pd.DataFrame({"n": g.size(), "n_ob": g["ob"].sum(), "n_tob": g["tob"].sum(),
                      "cmd": g["tail_3_pitch_act_command_mean"].mean(),
                      "t3deg": g["tail_3_pitch_mean_deg"].mean()}).reset_index()
    return e


def probe(run):
    f = (F2 / f"stance6_r10LTSB_{run}/copies/01_stance/policy_probe.npz.copy1") if run in R10 else (ECS / run / "01_stance/policy_probe.npz")
    with np.load(f) as d:
        t = np.asarray(d["update_timesteps"])
        ss = np.asarray(d["bank_labels"]) == "statue_settled"
        v = np.asarray(d["bank_action_now"][:, ss, 14], dtype=np.float64).mean(axis=1)
    tr = np.array([v[max(0, i - 7):i + 1].mean() for i in range(len(v))])
    return t, v, tr


def first(t, mask, after=0):
    i = np.flatnonzero(mask & (t > after))
    return int(t[i[0]]) if len(i) else None


def main():
    res = {}
    print("LEDGER EVENTS (my own reduction).  FOB = first eval with a tail actuator named over the bar in >= 1 episode.")
    print(f"{'run':12s} {'n_eval':>6s} {'FOB':>9s} {'FOBt3':>9s} {'cmd>=.3':>9s} {'cmd>=.6':>9s} {'cmd>=.9':>9s} {'maxCMD':>7s} {'at':>9s} {'nTOB':>5s} {'ECS: FOB after S':>16s}")
    for run in R10 + tuple(ECSR):
        e = led(run)
        t = e["timesteps"].to_numpy()
        after = S.get(run[:3], 0) if run not in R10 else 0
        fob = first(t, (e["n_tob"] > 0).to_numpy())
        fob_after = first(t, (e["n_tob"] > 0).to_numpy(), after)
        c3 = first(t, (e["cmd"] >= 0.3).to_numpy(), after)
        c6 = first(t, (e["cmd"] >= 0.6).to_numpy(), after)
        c9 = first(t, (e["cmd"] >= 0.9).to_numpy(), after)
        i = int(np.argmax(e["cmd"].to_numpy()))
        res[run] = {"fob": fob, "fob_after_S": fob_after, "cmd30": c3, "cmd60": c6, "cmd90": c9,
                    "max_cmd": float(e["cmd"].max()), "max_cmd_at": int(t[i]), "n_tob_eps": int(e["n_tob"].sum()),
                    "last_eval": int(t.max()), "n_eval": int(len(e))}
        print(f"{run:12s} {len(e):6d} {str(fob):>9s} {'':>9s} {str(c3):>9s} {str(c6):>9s} {str(c9):>9s} {e['cmd'].max():+7.3f} {t[i]:9d} {int(e['n_tob'].sum()):5d} {str(fob_after):>16s}")
    print()
    print("CMD_L at selected r10 steps (ledger):")
    for run in R10:
        e = led(run).set_index("timesteps")
        print(" ", run, " ".join(f"{s/1e6:.2f}M:{e.loc[s, 'cmd']:+.3f}" for s in (2883584, 3145728, 6291456, 6422528, 7340032, 9830400, 10485760, 11534336, 11796480) if s in e.index))
    print()
    print("PROBE statue_settled tail_3 raw mean, 8-row trailing mean (every 8,192 steps)")
    for run in R10:
        t, v, tr = probe(run)
        F = res[run]["fob"]
        pre = tr[t < F] if F else tr
        print(f"  {run}: rows {len(t)}; max trailing pre-FOB {pre.max():+.3f} at {int(t[t < F][np.argmax(pre)]) if F else int(t[np.argmax(pre)])};"
              f" whole-run max {tr.max():+.3f}; raw max {v.max():+.3f}")
        for thr in (0.10, 0.12, 0.14, 0.15, 0.16, 0.176, 0.18, 0.186):
            c = first(t, tr > thr)
            lead = (F - c) / 1e6 if (c is not None and F is not None and c < F) else None
            print(f"     thr {thr:.3f}: first {c}  lead {lead}")
        res[run]["probe_tr_max_preFOB"] = float(pre.max())
        res[run]["probe_tr_max"] = float(tr.max())
        # when does the trailing mean fall back below 0.05 after its first crossing of 0.15 ?
        c = first(t, tr > 0.15)
        if c is not None:
            back = first(t, tr < 0.05, c)
            print(f"     after crossing 0.15 at {c}: first back below 0.05 at {back}")
    m80 = []
    for run in ECSR:
        if run.startswith("s80"):
            t, v, tr = probe(run)
            m80.append((float(tr[t > S["s80"]].max()), run))
    print(f"  ECS s80 after S: max trailing {max(m80)[0]:+.3f} ({max(m80)[1]}); per run " + " ".join(f"{r}:{m:+.3f}" for m, r in sorted(m80, reverse=True)[:4]))
    m42 = []
    for run in ECSR:
        if run.startswith("s42"):
            t, v, tr = probe(run)
            m42.append(float(tr[t > S["s42"]].max()))
    print(f"  ECS s42 after S: trailing max range {min(m42):+.3f}..{max(m42):+.3f}")
    # Is the ECS probe history before S the parent's?
    t42, v42, _ = probe("s42")
    tA, vA, _ = probe("s42_A_1101")
    common = np.intersect1d(t42, tA[tA <= S["s42"]])
    if len(common):
        dv = max(abs(v42[np.searchsorted(t42, c)] - vA[np.searchsorted(tA, c)]) for c in common)
        print(f"  ECS s42_A_1101 rows at or before S: {len(common)}, max |diff| vs r10 s42 probe {dv:.2e}")
    json.dump(res, open(Path(__file__).with_name("vew_events.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
