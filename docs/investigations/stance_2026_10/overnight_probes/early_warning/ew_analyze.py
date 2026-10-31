"""EARLY-WARNING CURVE: line the per-checkpoint loop readouts up against each run's ledger events and the probe bank,
and score candidate rules (report only; no simulation).

    python -B ew_analyze.py > ew_analyze.txt

Inputs: ew_r10.jsonl, ew_ecs.jsonl (ew_measure.py), ew_events.json (ew_events.py).
Outcome event per run: the first TAIL over-bar ledger episode (FOB).  Lead time = FOB - the first checkpoint at which
the rule fires (checkpoints strictly before FOB count; a firing at or after FOB has lead <= 0).
Threshold choice (TRAIN): r10 s42 (positive) and r10 s80 (negative) only.  For each signal the threshold is the
midpoint between s80's maximum over ALL its checkpoints (1.05M-12.0M) and s42's maximum before its FOB; a signal whose
s42 pre-FOB maximum is not above s80's maximum gets no threshold.  The same rule is applied at native resolution in
section 4.  Disclosure: the rule was fixed after the r10 series (s60 included) had been printed, before any ECS
continuation was scored, so s60 is a seen test and the ECS lineages are the unseen test.
Variant TRAIN-early: s80's maximum over its checkpoints up to 8,388,608 only (before its own late tail rise, ledger
CMD_L >= 0.30 from 9,830,400).  TEST (never used to choose a threshold): r10 s60 (positive, FOB 8,126,464), the 12 ECS
s42 continuations (positives; start S = r10 s42 3,145,728) and the 12 ECS s80 continuations (negatives; start r10 s80
4,194,304, whose readouts belong to TRAIN).
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

import numpy as np

import ew_common as C

R10 = [json.loads(l) for l in open(C.HERE / "ew_r10.jsonl")]
ECSR = [json.loads(l) for l in open(C.HERE / "ew_ecs.jsonl")] if (C.HERE / "ew_ecs.jsonl").exists() else []
EV = json.load(open(C.HERE / "ew_events.json"))
ROWS = defaultdict(list)
for r in R10 + ECSR:
    ROWS[r["run"]].append(r)
for k in ROWS:
    ROWS[k].sort(key=lambda r: r["steps"])


def fin(x, cap=9.99):
    return cap if (x is None or not math.isfinite(x)) else x


SIG = {
    "own14": lambda r: r["own14_win_median"],
    "c0_own": lambda r: r["own"]["primary"]["c0"],
    "G_own": lambda r: r["own"]["primary"]["G"],
    "FP_own": lambda r: fin(r["own"]["primary"]["FP"]),
    "pitch_own": lambda r: -r["own"]["pitch"]["pitch_slope"],  # sign flipped: + = nose-down raises the command
    "G_own_6": lambda r: r["own"]["legacy"]["G"],
    "FP_own_6": lambda r: fin(r["own"]["legacy"]["FP"]),
    "c0_ref": lambda r: r["ref_statue_settled"]["primary"]["c0"],
    "G_ref": lambda r: r["ref_statue_settled"]["primary"]["G"],
    "FP_ref": lambda r: fin(r["ref_statue_settled"]["primary"]["FP"]),
    "pitch_ref": lambda r: -r["ref_statue_settled"]["pitch"]["pitch_slope"],
    "c0_bank": lambda r: r["ref"]["primary"]["c0"],
    "G_bank": lambda r: r["ref"]["primary"]["G"],
    "FP_bank": lambda r: fin(r["ref"]["primary"]["FP"]),
}


def ledger_at(run, steps):
    s = EV[run]["events"]["series"]
    t = s["timesteps"]
    return s["cmd_l"][t.index(steps)] if steps in t else float("nan")


def probe_at(run, steps, key):
    p = EV[run]["probe"]
    t = p["t"]
    return p[key][t.index(steps)] if steps in t else float("nan")


def add_ext(r):
    r["_cmdL"] = ledger_at(r["run"], r["steps"])
    r["_bank_ss"] = probe_at(r["run"], r["steps"], "bank_ss")
    r["_bank_mean"] = probe_at(r["run"], r["steps"], "bank_mean")
    r["_rollout_mu"] = probe_at(r["run"], r["steps"], "rollout_mu")


for rs in ROWS.values():
    for r in rs:
        add_ext(r)
SIG["CMD_L"] = lambda r: r["_cmdL"]
SIG["probe_bank_ss"] = lambda r: r["_bank_ss"]
SIG["probe_rollout_mu"] = lambda r: r["_rollout_mu"]


def fob(run):
    return EV[run]["events"]["first_over_tail_ep_overall"]


def mark_events(run, lo, hi):
    ev = EV[run]["events"]
    names = [("first_cmd30_overall", "CMD_L>=.3"), ("first_park_overall", "PARK"), ("first_over_tail_ep_overall", "FOB-tail"),
             ("first_over_any_ep_overall", "FOB-any"), ("first_tail_flag_overall", "TAILFLAG"),
             ("first_any_flag_overall", "ANYFLAG"), ("first_tail_alarm_overall", "TAILALARM"),
             ("first_any_alarm_overall", "ANYALARM"), ("first_rail_overall", "RAIL(CMD_L>=.9)")]
    return " ".join(n for k, n in names if ev[k] is not None and lo < ev[k] <= hi)


def p(*a):
    print(*a)


def section_validation():
    p("=" * 118)
    p("0. VALIDATION")
    p("=" * 118)
    diffs = []
    for run, rs in ROWS.items():
        for r in rs:
            if "dup_of" in r:
                continue
            pr = EV[run]["probe"]
            if r["steps"] in pr["t"]:
                i = pr["t"].index(r["steps"])
                d = float(np.max(np.abs(np.array(r["mu14_bank"]) - np.array(pr["bank_rows"][i]))))
                diffs.append((run, r["steps"], d))
    if diffs:
        worst = max(diffs, key=lambda x: x[2])
        p(f"- my network's raw tail_3 mean on the probe bank against the probe's recorded bank_action_now (raw mean) at the"
          f" same update row: {len(diffs)} checkpoints, max |diff| {worst[2]:.1e} ({worst[0]} {worst[1]}); so the probe"
          f" row at step N is the checkpoint-N network under the checkpoint-N statistics")
    ok = [r for rs in ROWS.values() for r in rs if "dup_of" not in r]
    p(f"- checkpoints measured: {len(ok)} unique, {sum('dup_of' in r for rs in ROWS.values() for r in rs)} duplicates (ECS starts)")
    nf = [(r['run'], r['steps'], r['n_full']) for r in ok if r['n_full'] < r['n_episodes']]
    p(f"- episodes not reaching 1,000 steps: {sum(r['n_episodes'] - r['n_full'] for r in ok)} of {sum(r['n_episodes'] for r in ok)} {nf[:6]}")
    zt = max(r["own"]["primary"]["z_tail_at_max"] for r in ok)
    zr = max(r["ref_statue_settled"]["primary"]["z_tail_at_max"] for r in ok)
    zp = max(max(r["own"]["pitch"]["z_pitch_absmax"], r["ref_statue_settled"]["pitch"]["z_pitch_absmax"]) for r in ok)
    cl = sum(r["own"]["primary"]["n_points_clipped"] + r["ref_statue_settled"]["primary"]["n_points_clipped"] + r["own"]["pitch"]["n_points_clipped"] for r in ok)
    cl6 = sum(r["own"]["legacy"]["n_points_clipped"] > 0 for r in ok)
    p(f"- primary 0-1 deg sweep: max tail_3 z {max(zt, zr):+.2f} (clip 10); pitch sweep max |z| on the rotated channels {zp:.2f};"
      f" clipped sweep points {cl}.  Legacy 0-6 deg sweep reaches the clip at {cl6} of {len(ok)} checkpoints.")
    ratios = [r["own"]["primary"]["slope_hi_half"] / r["own"]["primary"]["slope_lo_half"] for r in ok
              if abs(r["own"]["primary"]["slope_lo_half"]) > 0.01]
    ratios_r = [r["ref_statue_settled"]["primary"]["slope_hi_half"] / r["ref_statue_settled"]["primary"]["slope_lo_half"] for r in ok
                if abs(r["ref_statue_settled"]["primary"]["slope_lo_half"]) > 0.01]
    p(f"- linearity of the primary sweep (slope 0.5-1.0 deg / slope 0-0.5 deg, |lo| > 0.01): OWN median {np.median(ratios):.3f}"
      f" [{np.min(ratios):.3f}, {np.max(ratios):.3f}] n {len(ratios)}; REF median {np.median(ratios_r):.3f} [{np.min(ratios_r):.3f}, {np.max(ratios_r):.3f}] n {len(ratios_r)}")
    oc = np.array([[r["own14_win_median"], r["_cmdL"]] for r in ok if math.isfinite(r["_cmdL"])])
    p(f"- own14 (10 seeds 5001-5010) against the ledger's CMD_L at the same step (10 eval episodes): n {len(oc)},"
      f" median |diff| {np.median(np.abs(oc[:, 0] - oc[:, 1])):.3f}, max {np.max(np.abs(oc[:, 0] - oc[:, 1])):.3f}")
    fpo = np.array([[r["own"]["primary"]["FP"], r["own14_win_median"]] for r in ok if math.isfinite(r["own"]["primary"]["FP"])])
    p(f"- linear fixed point FP_own (clipped to [-1, 1]) against the measured own14: n {len(fpo)}, median |diff|"
      f" {np.median(np.abs(np.clip(fpo[:, 0], -1, 1) - fpo[:, 1])):.3f}")
    cpu = {k: [r["cpu_s"][k] for r in ok] for k in ("load", "ref_sweeps", "rollouts", "own_sweeps")}
    p("- CPU seconds per checkpoint (process_time, one thread, nice 15, beside other jobs): " +
      ", ".join(f"{k} median {np.median(v):.3f} [{np.min(v):.3f}, {np.max(v):.3f}]" for k, v in cpu.items()))


def fmt(x, w=6, d=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return f"{'inf' if x is not None and x > 0 else 'nan':>{w}s}"
    return f"{x:+{w}.{d}f}"


def section_series():
    p("")
    p("=" * 118)
    p("1. SERIES, r10 runs (every saved checkpoint).  OWN = own settled states (10 seeds x 6 snapshots); REF = the probe")
    p("   bank's two statue_settled rows.  G, c0, FP on the primary 0-1 deg sweep; G6 = legacy 0-6 deg G.  pitch = d cmd14")
    p("   per deg nose-DOWN seen (sign flipped so + = posture pathway as in s42).  CMD_L = ledger tail_3 command at the")
    p("   same step; bank_ss / roll_mu = the policy probe's statue_settled action and training-row mean at that update.")
    p("   Events = ledger events in (previous checkpoint, this checkpoint].")
    p("=" * 118)
    hdr = ("   steps     own14  CMD_L |  c0_own  G_own  FP_own G6_own pitchO |  c0_ref  G_ref  FP_ref pitchR | G_bank bank_ss roll_mu"
           " | pitch_held tail_deg | events")
    for run in C.R10_RUNS:
        p(f"--- {run}  (first tail over-bar episode: {fob(run)})")
        p(hdr)
        prev = 0
        for r in ROWS[run]:
            o, f = r["own"], r["ref_statue_settled"]
            p(f"  {r['steps']:>9d} {fmt(r['own14_win_median'])} {fmt(r['_cmdL'])} | {fmt(o['primary']['c0'])} {fmt(o['primary']['G'])}"
              f" {fmt(o['primary']['FP'])} {fmt(o['legacy']['G'])} {fmt(-o['pitch']['pitch_slope'])} | {fmt(f['primary']['c0'])}"
              f" {fmt(f['primary']['G'])} {fmt(f['primary']['FP'])} {fmt(-f['pitch']['pitch_slope'])} | {fmt(r['ref']['primary']['G'])}"
              f" {fmt(r['_bank_ss'])} {fmt(r['_rollout_mu'])} | {fmt(r['pitch_deg_win_median'], 7, 2)} {fmt(r['tail3_deg_win_median'], 6, 2)}"
              f" | {mark_events(run, prev, r['steps'])}")
            prev = r["steps"]
    if not ECSR:
        return
    for lin in ("s42", "s80"):
        runs = [k for k in C.ECS_RUNS if k.startswith(lin)]
        steps = sorted({r["steps"] for k in runs for r in ROWS[k]})
        p(f"--- ECS {lin} lineage: median [min, max] over its 12 continuations (the first row is the shared start)")
        p("   steps     own14                 c0_own                G_own                 FP_own                pitchO"
          "                c0_ref                G_ref                 FP_ref")
        for st in steps:
            rs = [r for k in runs for r in ROWS[k] if r["steps"] == st]
            cells = []
            for name in ("own14", "c0_own", "G_own", "FP_own", "pitch_own", "c0_ref", "G_ref", "FP_ref"):
                v = np.array([SIG[name](r) for r in rs])
                cells.append(f"{np.median(v):+.3f} [{v.min():+.2f},{v.max():+.2f}]")
            p(f"  {st:>9d} " + "  ".join(cells))
        # per-continuation event steps
        p(f"   per continuation: first tail over-bar episode / first TAIL flag / first TAIL alarm (ledger, after S)")
        for k in runs:
            ev = EV[k]["events"]
            p(f"     {k}: FOB-tail {ev['first_over_tail_ep']}  TAILFLAG {ev['first_tail_flag']}  TAILALARM {ev['first_tail_alarm']}"
              f"  FOB-any {ev['first_over_any_ep_overall']}  ANYFLAG {ev['first_any_flag']}  over-bar acts {','.join(ev['over_acts'])}")


def first_fire(rs, sig, thr, before=None):
    for r in rs:
        if before is not None and r["steps"] >= before:
            return None
        v = SIG[sig](r)
        if math.isfinite(v) and v > thr:
            return r["steps"]
    return None


def first_fire_any(rs, sig, thr):
    for r in rs:
        v = SIG[sig](r)
        if math.isfinite(v) and v > thr:
            return r["steps"]
    return None


def section_rules():
    p("")
    p("=" * 118)
    p("2. CANDIDATE RULES: 'signal > threshold' at a checkpoint.  Threshold = midpoint of (r10 s80 maximum, r10 s42 pre-FOB")
    p("   maximum), chosen on TRAIN = r10 s42 + r10 s80; TEST = r10 s60 (seen before the rule was fixed), ECS s42 x12 and")
    p("   ECS s80 x12 (unseen).  Lead = FOB - first firing checkpoint (steps; None = never fired before FOB).")
    p("=" * 118)
    s42, s80, s60 = ROWS["s42"], ROWS["s80"], ROWS["s60"]
    F42, F60 = fob("s42"), fob("s60")
    ecs42 = [k for k in C.ECS_RUNS if k.startswith("s42")]
    ecs80 = [k for k in C.ECS_RUNS if k.startswith("s80")]
    for variant, s80set in (("TRAIN (s80 all checkpoints)", s80), ("TRAIN-early (s80 to 8,388,608)", [r for r in s80 if r["steps"] <= 8_388_608])):
        p(f"--- {variant}")
        p("  signal          s80max   s42 pre-FOB values              thr     | s42 lead  s60 fire  s60 lead | ECS42 fire<=FOB (lead med)"
          " | ECS80 runs/ckpts firing | r10 s80 first fire (all ckpts)")
        for sig in SIG:
            v80 = [SIG[sig](r) for r in s80set if math.isfinite(SIG[sig](r))]
            if not v80:
                continue
            m80 = max(v80)
            pre = [(r["steps"], SIG[sig](r)) for r in s42 if r["steps"] < F42]
            m42 = max(v for _, v in pre if math.isfinite(v))
            pre_s = " ".join(fmt(v, 5, 2) for _, v in pre)
            if not m42 > m80:
                p(f"  {sig:15s} {fmt(m80)}   {pre_s:30s}  none    | s42's pre-FOB maximum is below s80's maximum")
                continue
            thr = 0.5 * (m80 + m42)
            f42 = first_fire(s42, sig, thr, F42)
            f60 = first_fire_any(s60, sig, thr)
            lead60 = (F60 - f60) if f60 is not None else None
            leads = []
            n_fire = 0
            for k in ecs42:
                ff = first_fire_any(ROWS[k], sig, thr)
                fb = EV[k]["events"]["first_over_tail_ep_overall"]
                if ff is not None and ff < fb:
                    n_fire += 1
                    leads.append(fb - ff)
            fire80 = [k for k in ecs80 if any(SIG[sig](r) > thr for r in ROWS[k] if "dup_of" not in r)]
            nck = sum(1 for k in ecs80 for r in ROWS[k] if "dup_of" not in r)
            nck_f = sum(1 for k in ecs80 for r in ROWS[k] if "dup_of" not in r and SIG[sig](r) > thr)
            r80 = first_fire_any(s80, sig, thr)
            p(f"  {sig:15s} {fmt(m80)}   {pre_s:30s} {thr:+.3f}  | {F42 - f42:>8d}  {str(f60):>9s} {str(lead60):>8s} | "
              f"{n_fire:2d}/12 ({int(np.median(leads)) if leads else '-'}) | {len(fire80):2d}/12 {nck_f:3d}/{nck:3d} | {r80}")


def section_crossings():
    p("")
    p("=" * 118)
    p("3. NATURAL CROSSINGS (fixed, physically meaningful lines, not fitted): first checkpoint with G > 0 (the loop turns")
    p("   positive), FP > 0.6 (the linear fixed point at the PARK line), FP >= 1 (fixed point at or past the rail), own14 > 0.3")
    p("=" * 118)
    lines = [("G_own", 0.0), ("G_ref", 0.0), ("G_own_6", 0.0), ("FP_own", 0.6), ("FP_own", 0.999), ("FP_ref", 0.6),
             ("own14", 0.3), ("c0_own", 0.2), ("CMD_L", 0.3), ("pitch_own", 0.05)]
    runs = list(C.R10_RUNS) + ([k for k in C.ECS_RUNS] if ECSR else [])
    p("  run         FOB-tail   " + "  ".join(f"{s}>{t:g}".rjust(12) for s, t in lines))
    for run in runs:
        rs = ROWS[run]
        cells = []
        for s, t in lines:
            ff = first_fire_any(rs, s, t)
            cells.append(f"{(str(ff) if ff is not None else '-'):>12s}")
        p(f"  {run:11s} {str(fob(run)):>9s}  " + "  ".join(cells))


def _fmt_lead(f, F):
    return "-" if f is None else (f"{(F - f) / 1e6:+.2f}M" if F is not None else f"@{f / 1e6:.2f}M")


def section_grid():
    p("")
    p("=" * 118)
    p("2b. THRESHOLD GRID at checkpoint resolution (every threshold shown, so the fragility of any one choice is visible).")
    p("    s42 / s60: lead = FOB - first firing checkpoint (M steps; '-' = no firing before FOB).  r10 s80: first firing")
    p("    checkpoint over its whole run (none = '-').  ECS s80: continuations with a firing at a checkpoint after S / 12.")
    p("    ECS s42: continuations firing at a checkpoint before their own FOB / 12 (S itself counts; it is shared).")
    p("=" * 118)
    grids = {"G_own": [-0.5, -0.25, 0.0, 0.1, 0.2, 0.25, 0.3, 0.35], "G_ref": [-0.25, 0.0, 0.1, 0.2, 0.25, 0.3],
             "G_own_6": [-0.25, 0.0, 0.1, 0.2, 0.25], "c0_own": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
             "own14": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65], "FP_own": [0.2, 0.3, 0.4, 0.5, 0.6, 0.65],
             "CMD_L": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "c0_ref": [0.05, 0.1, 0.15, 0.2], "probe_bank_ss": [0.05, 0.1, 0.15, 0.2],
             "pitch_own": [0.02, 0.05, 0.08]}
    F42, F60 = fob("s42"), fob("s60")
    ecs42 = [k for k in C.ECS_RUNS if k.startswith("s42")]
    ecs80 = [k for k in C.ECS_RUNS if k.startswith("s80")]
    p("  signal           thr   | s42 lead | s60 lead | r10 s80 first | ECS80 fire | ECS42 fire<FOB")
    for sig, ths in grids.items():
        for thr in ths:
            f42 = first_fire(ROWS["s42"], sig, thr, F42)
            f60 = first_fire(ROWS["s60"], sig, thr, F60)
            f80 = first_fire_any(ROWS["s80"], sig, thr)
            n80 = sum(any(SIG[sig](r) > thr for r in ROWS[k] if "dup_of" not in r) for k in ecs80) if ECSR else -1
            n42 = sum(first_fire(ROWS[k], sig, thr, EV[k]["events"]["first_over_tail_ep_overall"]) is not None for k in ecs42) if ECSR else -1
            p(f"  {sig:15s} {thr:+5.2f}  | {_fmt_lead(f42, F42):>8s} | {_fmt_lead(f60, F60):>8s} | {('-' if f80 is None else f'{f80 / 1e6:.2f}M'):>13s} |"
              f" {n80:5d}/12   | {n42:5d}/12")
        p("")


def trailing(x, k):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        j = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[j]) / (i + 1 - j)
    return out


def section_native():
    p("")
    p("=" * 118)
    p("4. NATIVE-RESOLUTION SIGNALS ALREADY RECORDED IN TRAINING (no new measurement): the probe bank's statue_settled")
    p("   tail_3 mean (every 8,192 steps; 8-row TRAILING mean = 65,536 steps, causal) and the ledger's CMD_L (every 131,072")
    p("   steps).  Leads in M steps before FOB.  r10 s80: first crossing over its whole run.  ECS: after S only (the rows")
    p("   before S are the parent run's).  ECS s42 'pre-S' = the crossing already happened in the shared parent history.")
    p("=" * 118)
    ecs42 = [k for k in C.ECS_RUNS if k.startswith("s42")]
    ecs80 = [k for k in C.ECS_RUNS if k.startswith("s80")]
    S = {"s42": 3_145_728, "s80": 4_194_304}

    def series(run, which):
        if which == "bank_ss":
            pr = EV[run]["probe"]
            return np.array(pr["t"]), trailing(pr["bank_ss"], 8)
        if which == "rollout_mu":
            pr = EV[run]["probe"]
            return np.array(pr["t"]), trailing(pr["rollout_mu"], 8)
        se = EV[run]["events"]["series"]
        return np.array(se["timesteps"]), np.array(se["cmd_l"], dtype=float)

    def cross(t, v, thr, after=0):
        i = np.flatnonzero((v > thr) & (t > after))
        return int(t[i[0]]) if len(i) else None

    for which, ths in (("bank_ss", [0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18]), ("CMD_L", [0.2, 0.3, 0.4, 0.5, 0.6]),
                       ("rollout_mu", [0.2, 0.3, 0.4, 0.5])):
        p(f"--- {which}: maxima  s42 pre-FOB {np.max(series('s42', which)[1][series('s42', which)[0] < fob('s42')]):+.3f}"
          f"  s60 pre-FOB {np.max(series('s60', which)[1][series('s60', which)[0] < fob('s60')]):+.3f}"
          f"  r10 s80 whole run {np.max(series('s80', which)[1]):+.3f}"
          f"  r10 s80 to 8.39M {np.max(series('s80', which)[1][series('s80', which)[0] <= 8_388_608]):+.3f}"
          f"  ECS s80 after S {max(np.max(series(k, which)[1][series(k, which)[0] > S['s80']]) for k in ecs80):+.3f}")
        t42_, v42_ = series("s42", which)
        rule = 0.5 * (np.max(series("s80", which)[1]) + np.max(v42_[t42_ < fob("s42")]))
        p(f"  rule threshold (midpoint of r10 s80 max and s42 pre-FOB max) = {rule:+.3f}; it is the row marked '*'")
        p("  thr    | s42 first (lead)      | s60 first (lead)      | r10 s80 first | ECS80 runs firing after S | ECS42 firing before FOB")
        for thr in sorted(ths + [round(rule, 3)]):
            t42, v42 = series("s42", which)
            t60, v60 = series("s60", which)
            t80, v80 = series("s80", which)
            c42, c60, c80 = cross(t42, v42, thr), cross(t60, v60, thr), cross(t80, v80, thr)
            n80 = sum(cross(*series(k, which), thr, S["s80"]) is not None for k in ecs80)
            n42 = 0
            pre = 0
            for k in ecs42:
                tk, vk = series(k, which)
                c = cross(tk, vk, thr)
                if c is not None and c < EV[k]["events"]["first_over_tail_ep_overall"]:
                    n42 += 1
                    pre += int(c <= S["s42"])
            l42 = f"{c42} ({(fob('s42') - c42) / 1e6:+.2f}M)" if c42 is not None and c42 < fob("s42") else str(c42)
            l60 = f"{c60} ({(fob('s60') - c60) / 1e6:+.2f}M)" if c60 is not None and c60 < fob("s60") else str(c60)
            p(f"  {thr:+.3f}{'*' if abs(thr - round(rule, 3)) < 1e-9 else ' '}| {l42:21s} | {l60:21s} | {str(c80):>13s} | {n80:5d}/12                  | {n42:2d}/12 ({pre} pre-S)")
        p("")


def section_ecs_leads():
    if not ECSR:
        return
    p("")
    p("=" * 118)
    p("5. ECS s42 continuations, per continuation (262,144-step checkpoints from the shared start S = 3,145,728)")
    p("=" * 118)
    p("  run         FOB-tail   | own14 / c0_own / G_own / FP_own at each checkpoint up to FOB")
    for k in [k for k in C.ECS_RUNS if k.startswith("s42")]:
        F = EV[k]["events"]["first_over_tail_ep_overall"]
        cells = [f"{r['steps'] / 1e6:.2f}M:{SIG['own14'](r):+.2f}/{SIG['c0_own'](r):+.2f}/{SIG['G_own'](r):+.2f}/{SIG['FP_own'](r):+.2f}"
                 for r in ROWS[k] if r["steps"] <= F]
        p(f"  {k:11s} {F:>9d}  | " + "  ".join(cells))


def write_csv():
    """One row per (run, checkpoint): every signal, the ledger CMD_L and the probe values at that step, and the run's
    FOB.  Written to ew_series.csv."""
    cols = list(SIG)
    with open(C.HERE / "ew_series.csv", "w") as fh:
        fh.write("run,steps,dup_of,fob_tail," + ",".join(cols) + ",pitch_held_deg,tail3_deg,n_over_bar_det\n")
        for run in list(C.R10_RUNS) + [k for k in C.ECS_RUNS if k in ROWS]:
            for r in ROWS[run]:
                vals = [SIG[c](r) for c in cols]
                fh.write(f"{run},{r['steps']},{'/'.join(map(str, r.get('dup_of', [])))},{fob(run) or ''},"
                         + ",".join(f"{v:.5f}" for v in vals)
                         + f",{r['pitch_deg_win_median']:.4f},{r['tail3_deg_win_median']:.4f},{r['n_over_bar']}\n")


def main() -> int:
    section_validation()
    section_series()
    section_rules()
    section_grid()
    section_crossings()
    section_native()
    section_ecs_leads()
    write_csv()
    return 0


if __name__ == "__main__":
    sys.exit(main())
