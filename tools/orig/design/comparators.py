"""PC study design: the stance6 round's three runs (the historical comparators) read from their F2 copies.

Report only; reads the round's files read only; writes nothing but its stdout (redirected by the caller into
ST/design/comparators.out).  Run: cd /tmp && nice -n 10 PY -I -B ST/design/comparators.py > ST/design/comparators.out

What it computes, per run (stance6_r10LTSB_s60, _s42, _s80), from 01_stance/eval_episodes.csv (.copy1, checked against
the F2 fetch manifest row and the run's artifact_manifest.json entry) and the round's scorer output score.json:
  - the ledger tail_3 command CMD_L per evaluation (mean over the evaluation's rows of tail_3_pitch_act_command_mean;
    BS2 read_bs.ledger_direct's definition), its firsts at 0.30 / 0.60 / 0.90, its max, last, count >= 0.60;
  - over-bar rows (saturated_actuator_fraction > 0.10): the first evaluation with any, with a tail actuator named
    (FOB_tail), with tail_3_pitch_act named (first_t3_over); counts by actuator;
  - the TAIL DRIVE candidates: the mean over evaluations of |CMD_L| (all 91; from 4,194,304);
  - per other actuator: the evaluation-level command-mean extremes (|mean over the 10 rows| >= 0.60 count);
  - returns (mean per evaluation; the run's mean over all evaluations and over the last 10), falls (rows with length
    < 1000), screen_v2_clean share;
  - the round's classes from score.json (TAIL CLASS and kind, tail alarms, ANY alarms by actuator, GATE, OUTCOME,
    MEAN_a);
  - the on-policy tail_3 command (policy_probe.npz rollout_mu_mean[:, 14], mean of the last 8 rows; BS2's primary
    form), report only.
Then the exact 3-vs-3 permutation distribution and Fisher tables used by the design (DESIGN_NOTES section 3).
"""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import zipfile
from fractions import Fraction
from pathlib import Path

S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
S2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
FETCH = S1 / "foot/stance6/fetch"
FMAN = FETCH / "manifest.csv"
SCORE = S2 / "r10round/f2/score.json"
RUNS = {60: "stance6_r10LTSB_s60", 42: "stance6_r10LTSB_s42", 80: "stance6_r10LTSB_s80"}
BAR = 0.10
GRID = 131072
TAIL_ACTS = ("tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")
ACTS = ("neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
        "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act") + TAIL_ACTS
J_FIRST = 4194304


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fman_rows():
    with open(FMAN, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(r, k):
    try:
        v = float(r.get(k))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def check_copy(run_id: str, rel: str, rows) -> Path:
    copies = FETCH / "F2" / run_id / "copies"
    p = copies / (rel + ".copy1")
    got = sha256(p)
    f2 = [r for r in rows if r["fetch"] == "F2" and r["run_id"] == run_id and r["path"] == rel
          and r["status"] == "COPY" and r["copy"] == "1"]
    man = json.loads((copies / "artifact_manifest.json.copy1").read_text(encoding="utf-8"))
    ent = [e for e in man["files"] if e["path"] == rel]
    want_man = ent[0]["sha256"].split(":", 1)[-1] if ent else None
    ok = bool(f2) and f2[-1]["sha256"] == got and want_man == got
    print(f"  file {rel}.copy1 sha256 {got[:16]}... F2-manifest {'=' if f2 and f2[-1]['sha256'] == got else '!='} "
          f"artifact-manifest {'=' if want_man == got else '!='} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(f"{run_id} {rel}: copy does not match its manifests")
    return p


def ledger(run_id: str, rows):
    p = check_copy(run_id, "01_stance/eval_episodes.csv", rows)
    with open(p, newline="", encoding="utf-8") as fh:
        data = list(csv.DictReader(fh))
    by = {}
    for r in data:
        t = num(r, "timesteps")
        if t is not None and t == int(t):
            by.setdefault(int(t), []).append(r)
    per = {}
    for s, rs in sorted(by.items()):
        cmd = [c for c in (num(r, "tail_3_pitch_act_command_mean") for r in rs) if c is not None]
        over = [r for r in rs if (num(r, "saturated_actuator_fraction") or 0.0) > BAR]
        rets = [x for x in (num(r, "return") for r in rs) if x is not None]
        lens = [x for x in (num(r, "length") for r in rs) if x is not None]
        clean = [r.get("screen_v2_clean") for r in rs]
        acts = {}
        for a in ACTS:
            v = [x for x in (num(r, a + "_command_mean") for r in rs) if x is not None]
            acts[a] = sum(v) / len(v) if v else None
        per[s] = dict(n=len(rs), cmd_l=sum(cmd) / len(cmd) if cmd else None, n_over=len(over),
                      n_tail_over=sum(1 for r in over if r.get("saturated_actuator") in TAIL_ACTS),
                      n_t3_over=sum(1 for r in over if r.get("saturated_actuator") == "tail_3_pitch_act"),
                      over_by={a: sum(1 for r in over if r.get("saturated_actuator") == a)
                               for a in {str(r.get("saturated_actuator")) for r in over}},
                      ret=sum(rets) / len(rets) if rets else None, falls=sum(1 for x in lens if x < 1000),
                      clean=sum(1 for c in clean if str(c).strip().lower() in ("true", "1", "1.0")),
                      acts=acts)
    return per


def firsts(per):
    f = {"first_over_any": next((s for s, e in per.items() if e["n_over"] > 0), None),
         "FOB_tail": next((s for s, e in per.items() if e["n_tail_over"] > 0), None),
         "first_t3_over": next((s for s, e in per.items() if e["n_t3_over"] > 0), None)}
    for line in (0.30, 0.60, 0.90):
        f[f"cmd_l_ge_{line:.2f}"] = next((s for s, e in per.items() if e["cmd_l"] is not None and e["cmd_l"] >= line),
                                         None)
    return f


def probe_tail3(run_id: str, rows):
    rel = "01_stance/policy_probe.npz"
    p = check_copy(run_id, rel, rows)
    import numpy as np  # noqa: PLC0415

    with np.load(p, allow_pickle=False) as z:
        mu = np.asarray(z["rollout_mu_mean"], dtype=np.float64)
        steps = np.asarray(z["update_timesteps"]) if "update_timesteps" in z.files else None
    last8 = [Fraction(float(x)) for x in mu[-8:, 14]]
    rng = (int(steps[-8]), int(steps[-1])) if steps is not None else None
    return float(sum(last8) / 8), rng, mu.shape


def perm_p(pc, ctrl):
    """Exact one-sided unpaired permutation p (PC lower), statistic = sum of the PC group; ties count against."""
    allv = list(pc) + list(ctrl)
    obs = sum(pc)
    n = len(pc)
    splits = list(itertools.combinations(range(len(allv)), n))
    k = sum(1 for idx in splits if sum(allv[i] for i in idx) <= obs + 1e-15)
    return Fraction(k, len(splits))


def hyper_p(x_pc, n_pc, x_c, n_c):
    """One-sided Fisher exact p for 'PC has fewer events': P(X <= x_pc) under the hypergeometric."""
    K = x_pc + x_c
    N = n_pc + n_c
    tot = math.comb(N, n_pc)
    return Fraction(sum(math.comb(K, i) * math.comb(N - K, n_pc - i) for i in range(0, x_pc + 1)
                        if 0 <= n_pc - i <= N - K), tot)


def main():
    rows = fman_rows()
    score = json.loads(SCORE.read_text(encoding="utf-8"))
    print(f"score.json {SCORE} sha256 {sha256(SCORE)}")
    per_seed = {d["seed"]: d for d in score["per_seed"]}
    td_all, td_j = {}, {}
    for seed, run_id in RUNS.items():
        print(f"\n=== {run_id} (seed {seed})")
        per = ledger(run_id, rows)
        evals = sorted(per)
        grid_ok = evals == [GRID * k for k in range(1, 92)] and all(per[s]["n"] == 10 for s in evals)
        print(f"  evaluations {len(evals)} ({evals[0]} .. {evals[-1]}), 10 rows each and on the grid: {grid_ok}")
        f = firsts(per)
        print(f"  firsts {json.dumps(f)}")
        cl = [(s, per[s]["cmd_l"]) for s in evals if per[s]["cmd_l"] is not None]
        mx = max(cl, key=lambda x: x[1])
        mn = min(cl, key=lambda x: x[1])
        print(f"  CMD_L max {mx[1]:+.4f} at {mx[0]}; min {mn[1]:+.4f} at {mn[0]}; last {cl[-1][1]:+.4f} at {cl[-1][0]};"
              f" evaluations with CMD_L >= 0.60: {sum(1 for _, v in cl if v >= 0.60)} of {len(cl)};"
              f" with |CMD_L| >= 0.60: {sum(1 for _, v in cl if abs(v) >= 0.60)}")
        td_all[seed] = sum(abs(v) for _, v in cl) / len(cl)
        jj = [abs(v) for s, v in cl if s >= J_FIRST]
        td_j[seed] = sum(jj) / len(jj)
        sg = [v for _, v in cl]
        print(f"  TAIL DRIVE mean |CMD_L| all {len(cl)} evaluations {td_all[seed]:.4f}; from {J_FIRST} ({len(jj)} "
              f"evaluations) {td_j[seed]:.4f}; signed mean all {sum(sg) / len(sg):+.4f}")
        over = {}
        for s in evals:
            for a, c in per[s]["over_by"].items():
                over[a] = over.get(a, 0) + c
        print(f"  over-bar rows by actuator (whole run): {json.dumps(over, sort_keys=True)}")
        for a in ACTS:
            v = [(s, per[s]["acts"][a]) for s in evals if per[s]["acts"][a] is not None]
            big = [s for s, x in v if abs(x) >= 0.60]
            ex = max(v, key=lambda t: abs(t[1]))
            if a in ("tail_1_pitch_act", "tail_2_pitch_act", "tail_3_pitch_act", "neck_pitch_act", "head_pitch_act") \
                    or big:
                print(f"    {a:18s} evaluation command mean: largest |.| {ex[1]:+.3f} at {ex[0]}; evaluations with "
                      f"|.| >= 0.60: {len(big)}; last {v[-1][1]:+.3f}")
        rets = [per[s]["ret"] for s in evals]
        print(f"  returns: mean over all evaluations {sum(rets) / len(rets):.1f}; last 10 {sum(rets[-10:]) / 10:.1f}; "
              f"last {rets[-1]:.1f}; falls (rows < 1000 steps) {sum(per[s]['falls'] for s in evals)} of "
              f"{10 * len(evals)} (last 10 evaluations {sum(per[s]['falls'] for s in evals[-10:])}); screen_v2_clean "
              f"{sum(per[s]['clean'] for s in evals)} of {10 * len(evals)}")
        d = per_seed[seed]
        c = d["classes"]
        print(f"  ROUND CLASSES: TAIL CLASS {c.get('cls')} {c.get('kind') or ''}; tail alarms {len(c.get('tail_alarms') or [])}"
              f" (first {(c.get('tail_alarms') or [None])[0]}); flagged evaluations {c.get('flagged_evaluations')};"
              f" trace rail {c.get('trace_rail')}; parked {c.get('parked')}; driven {c.get('driven')}")
        print(f"    MEAN_a {json.dumps({k: round(v, 4) for k, v in (c.get('mean') or {}).items()})}")
        anyal = c.get("any_alarms") or []
        print(f"    ANY ALARMS {len(anyal)}: {[(a.get('step'), a.get('actuators')) for a in anyal]}")
        for key in ("gate", "gate_class", "outcome", "run_outcome"):
            if key in c:
                print(f"    {key}: {json.dumps(c[key])[:600]}")
        gv = check_copy(run_id, "01_stance/gate_verdict.json", rows)
        g = json.loads(gv.read_text(encoding="utf-8"))
        print(f"  the notebook's own gate (gate_verdict.json, report only): passed {g.get('passed')} on "
              f"{g.get('checkpoint')} (gate_kind {g.get('gate_kind')}, judged_at {g.get('judged_at')})")
        try:
            val, last_step, shape = probe_tail3(run_id, rows)
            print(f"  on-policy tail_3 command, last 8 probe rows (report only): {val:+.5f} (rows at update_timesteps {last_step[0]} .. {last_step[1]}; "
                  f"probe shape {shape})")
        except Exception as exc:  # noqa: BLE001
            print(f"  on-policy tail_3 command: not read ({type(exc).__name__}: {exc})")

    print("\n=== THE DESIGN'S TABLES")
    for name, td in (("all 91 evaluations", td_all), (f"from {J_FIRST}", td_j)):
        vals = sorted(td.values())
        print(f"  TAIL DRIVE ({name}): comparators {json.dumps({k: round(v, 6) for k, v in td.items()})}; the lowest "
              f"{vals[0]:.6f} (repr {vals[0]!r})")
    ctrl = list(td_all.values())
    lo = min(ctrl)
    print("  exact 3-vs-3 permutation p (one-sided, PC lower) for PC values placed among the comparators "
          "(TAIL DRIVE over all evaluations):")
    for label, pc in (("all three below the lowest comparator", [lo * 0.1, lo * 0.2, lo * 0.3]),
                      ("two below the lowest, one between the lowest and the middle",
                       [lo * 0.1, lo * 0.2, (lo + sorted(ctrl)[1]) / 2]),
                      ("two below the lowest, one above the highest", [lo * 0.1, lo * 0.2, max(ctrl) + 0.1]),
                      ("one below the lowest, two above the highest", [lo * 0.1, max(ctrl) + 0.1, max(ctrl) + 0.2]),
                      ("all three above the highest", [max(ctrl) + 0.1, max(ctrl) + 0.2, max(ctrl) + 0.3])):
        print(f"    {label}: p = {perm_p(pc, ctrl)} = {float(perm_p(pc, ctrl)):.3f}")
    print(f"  the number of 3+3 splits: {math.comb(6, 3)}; minimum attainable p {Fraction(1, math.comb(6, 3))}")
    print("  Fisher exact one-sided p (PC fewer) for a binary outcome against the round's counts:")
    for nm, xc in (("tail habit (RAIL or PARKED): round 2 of 3", 2), ("ANY ALARM: round 2 of 3", 2),
                   ("POSITIVE missing (NEGATIVE): round 2 of 3", 2)):
        print(f"    {nm}: " + "; ".join(f"PC {x} of 3 -> p {hyper_p(x, 3, xc, 3)} = {float(hyper_p(x, 3, xc, 3)):.3f}"
                                         for x in range(4)))
    print(f"    tail RAIL only: round 1 of 3: " + "; ".join(
        f"PC {x} of 3 -> p {float(hyper_p(x, 3, 1, 3)):.3f}" for x in range(4)))
    print("  pooled with LTSB r9 (2 of 5 ANY ALARM runs) as the round's Statistic A comparator, 0 of 3 PC: "
          f"p {float(hyper_p(0, 3, 4, 8)):.3f} (against 4 of 8: r9 2 of 5 + stance6 2 of 3; report only)")


if __name__ == "__main__":
    main()
