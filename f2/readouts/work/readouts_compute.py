# Readouts analyst (report only, no simulation): computes D-1, D-2, D-11..D-15, the entropy check and the interim
# comparison from F2's manifest-verified copies and the pinned lane B tools' outputs (ledger_<run>.json,
# k3panel_<run>.json, r9ledger_*.json, written by the tools into this folder).  Reads npz with allow_pickle=False;
# never unpickles or imports any downloaded file.  Writes readouts_data.json and per-update std CSVs.
import csv
import hashlib
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

S1 = pathlib.Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
S0 = pathlib.Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
ROUND = S0 / "foot/stance6"
F2 = ROUND / "fetch/F2"
OUT = S1 / "r10round/f2/readouts"
W = OUT / "work"
RUNS = {"stance6_r10LTSB_s60": 60, "stance6_r10LTSB_s42": 42, "stance6_r10LTSB_s80": 80}
SHORT = {"stance6_r10LTSB_s60": "s60", "stance6_r10LTSB_s42": "s42", "stance6_r10LTSB_s80": "s80"}
ACT = ("neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act",
       "r_knee_act", "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act",
       "tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")
TAIL = {11, 12, 13, 14}
FINAL = 12001280
WIN = [("[0,4194304)", 0, 4194304), ("[4194304,8388608)", 4194304, 8388608), ("[8388608,12001280]", 8388608, FINAL + 1)]
WIN7 = [("[0,7000000)", 0, 7000000), ("[7000000,12001280]", 7000000, FINAL + 1)]
K3_STEPS = [k * 1048576 for k in range(1, 12)] + [FINAL]
STATUE_LEDGER = S1 / "r10round/runbranch/build/v8/head_statue/eval_episodes.csv"
SNAP = {1: S1 / "r10round/interim/snap1", 2: S1 / "r10round/interim/snap2"}

verified = json.loads((W / "verified_copies.json").read_text())


def vpath(rid, rel):
    p = verified[rid]["paths"][rel]
    assert p["verified"], (rid, rel)
    return pathlib.Path(p["verified"])


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_csv(path):
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def r(x, k=4):
    return None if x is None else round(float(x), k)


def in_win(s, lo, hi):
    return lo <= s < hi


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ statue reference (TEL columns, same ledger metric)
sh, srows = read_csv(STATUE_LEDGER)
STATUE = {}
for c in sh:
    vals = [num(x.get(c)) for x in srows]
    vals = [v for v in vals if v is not None]
    if vals:
        STATUE[c] = sum(vals) / len(vals)
STATUE_META = dict(file=str(STATUE_LEDGER), sha256=sha(STATUE_LEDGER), rows=len(srows),
                   note="zero action (statue) on the run-branch head tree of the V8 check, 10 evaluation episodes; "
                        "same 194-column ledger as the runs")

# statue M2 panel (harness metric), 40 episodes
sp = json.loads((S1 / "r10round/f2/statue_m2/panel.json").read_text())
STATUE_M2 = {}
for k in ("head_tip_z_m", "neck_pitch_deg", "head_pitch_deg", "pitch_deg", "roll_deg", "pelvis_z_m", "com_fa_mm",
          "com_lat_mm", "q1_deg", "q2_deg", "q3_deg", "chord_base_deg", "chord_dist_deg", "tip_z_m"):
    vals = [num(e.get(k)) for e in sp["episodes"]]
    vals = [v for v in vals if v is not None]
    STATUE_M2[k] = (sum(vals) / len(vals)) if vals else None

pins = json.loads((S1 / "r10round/lean/build/laneB/tools/pins_r10round.json").read_text())
TEL_COLS = pins["tel_columns"] if "tel_columns" in pins and isinstance(pins["tel_columns"], list) else pins["pins"]["tel_columns"]

data = {"statue_ledger": STATUE_META, "statue_ledger_values": {k: r(v, 5) for k, v in STATUE.items() if k in TEL_COLS
        or k in ("com_fore_aft_mean_m", "root_pitch_mean_deg")},
        "statue_m2": {k: r(v, 5) for k, v in STATUE_M2.items()}, "runs": {}}

for rid, seed in RUNS.items():
    sr = SHORT[rid]
    R = {"run_id": rid, "seed": seed, "sources": {}}
    led_csv = vpath(rid, "01_stance/eval_episodes.csv")
    probe_p = vpath(rid, "01_stance/policy_probe.npz")
    th_p = vpath(rid, "01_stance/figures/training_history/training_history.json")
    sc_p = vpath(rid, "01_stance/stage_config.json")
    pr_p = vpath(rid, "01_stance/progress.json")
    te_p = vpath(rid, "01_stance/train_episodes.csv")
    for k, p in (("eval_episodes.csv", led_csv), ("policy_probe.npz", probe_p), ("training_history.json", th_p),
                 ("stage_config.json", sc_p), ("progress.json", pr_p), ("train_episodes.csv", te_p)):
        R["sources"][k] = dict(path=str(p), sha256=sha(p))
    L = json.loads((W / f"ledger_{sr}.json").read_text())
    R["sources"]["ledger_tool_json"] = str(W / f"ledger_{sr}.json")
    assert L["sha256"] == R["sources"]["eval_episodes.csv"]["sha256"]
    hdr, rows = read_csv(led_csv)

    # ---------------------------------------------------------- D-1
    pe = {int(k): v for k, v in L["per_evaluation"].items()}
    steps = sorted(pe)
    by_step = defaultdict(list)
    for x in rows:
        by_step[int(float(x["timesteps"]))].append(x)
    over_rows = []  # rows over the bar: (step, episode, actuator, sign)
    for x in rows:
        f = num(x.get("saturated_actuator_fraction"))
        if f is not None and f > 0.10:
            ps = num(x.get("saturated_actuator_positive_share"))
            sign = "+" if (ps is not None and ps >= 0.5) else "-"
            over_rows.append((int(float(x["timesteps"])), int(float(x["episode"])), x.get("saturated_actuator"), sign,
                              f, int(float(x.get("saturated_actuator_ties") or 0)),
                              int(float(x.get("actuators_saturated") or 0))))
    table = []
    for s in steps:
        e = pe[s]
        signed = Counter(f"{a}{sg}" for (st, ep, a, sg, f, t, n) in over_rows if st == s)
        rail = Counter()
        at_rail_rows = 0
        for x in by_step[s]:
            for tok in (x.get("rail_actuators") or "").split(";"):
                tok = tok.strip()
                if tok:
                    rail[tok] += 1
            if (num(x.get("actuators_at_rail")) or 0) > 0:
                at_rail_rows += 1
        if e["any"] or e["tail"] or e["unres"] or rail:
            table.append(dict(step=s, n=e["n"], any=e["any"], tail=e["tail"], unres=e["unres"],
                              any_flagged=e["any"] >= 5, tail_flagged=e["tail"] >= 5, tail_by=e["tail_by"],
                              any_by_signed=dict(signed), tel_rail=dict(rail), tel_at_rail_rows=at_rail_rows,
                              missing=len(e["any_missing"]) + len(e["tail_missing"])))
    # agreement of TEL rail names with the M5 over-bar rows, row by row
    agree = Counter()
    for x in rows:
        f = num(x.get("saturated_actuator_fraction"))
        over = f is not None and f > 0.10
        has_rail = bool((x.get("rail_actuators") or "").strip())
        agree[(over, has_rail)] += 1
    d1 = L["d1"]
    # window shares by actuator (incl. OTHER) and flagged evaluations
    win = {}
    for name, lo, hi in WIN + WIN7:
        ev = [s for s in steps if in_win(s, lo, hi)]
        nrows = sum(pe[s]["n"] for s in ev)
        cnt = Counter(f"{a}{sg}" for (st, ep, a, sg, f, t, n) in over_rows if in_win(st, lo, hi))
        tail_over = sum(v for k, v in cnt.items() if k[:-1] in {ACT[i] for i in TAIL})
        other_over = sum(v for k, v in cnt.items() if k[:-1] not in {ACT[i] for i in TAIL})
        win[name] = dict(evaluations=len(ev), rows=nrows, over_by_actuator_sign=dict(cnt), tail_over=tail_over,
                         other_over=other_over, tail_share=r(tail_over / nrows if nrows else None, 4),
                         other_share=r(other_over / nrows if nrows else None, 4),
                         tail_flagged_evals=sum(1 for s in ev if pe[s]["tail"] >= 5),
                         any_flagged_evals=sum(1 for s in ev if pe[s]["any"] >= 5),
                         other_flagged_evals=sum(1 for s in ev if pe[s]["any"] >= 5 and pe[s]["tail"] < 5),
                         tail_flagged_share=r(sum(1 for s in ev if pe[s]["tail"] >= 5) / len(ev) if ev else None, 4),
                         any_alarms=[a for a in L["any_alarms"] if in_win(a, lo, hi)],
                         tail_alarms=[a for a in L["tail_alarms"] if in_win(a, lo, hi)],
                         rows_tel_rail=sum(1 for x in rows if in_win(int(float(x["timesteps"])), lo, hi)
                                           and (x.get("rail_actuators") or "").strip()))
    first_any_sat = min((st for (st, *_rest) in over_rows), default=None)
    # flagged spells for ANY (other actuators)
    def spells(flags):
        out, cur = [], []
        for s in steps:
            if flags(s) and (not cur or s - cur[-1] == 131072):
                cur.append(s)
            elif flags(s):
                out.append((cur[0], cur[-1], len(cur)))
                cur = [s]
            else:
                if cur:
                    out.append((cur[0], cur[-1], len(cur)))
                cur = []
        if cur:
            out.append((cur[0], cur[-1], len(cur)))
        return out
    R["D1"] = dict(
        tool_stdout=str(W / f"ledger_{sr}.txt"), c6=L.get("c6"),
        evaluations_read=len(steps), rows=len(rows),
        first_tail_saturated_episode=d1["first_tail_saturated_episode"],
        first_tail_flagged_evaluation=d1["first_tail_flagged_evaluation"],
        tail_alarm_steps=d1["tail_alarm_steps"], any_alarm_steps=d1["any_alarm_steps"],
        any_alarm_count=d1["any_alarm_count"], tail_flagged_spells=d1["tail_flagged_spells"],
        any_flagged_spells=spells(lambda s: pe[s]["any"] >= 5),
        tail_flagged_evaluations=L["tail_flagged"], any_flagged_evaluations=L["any_flagged"],
        needs_resolution=L["needs_resolution"],
        first_any_saturated_episode=first_any_sat,
        rows_over_bar_total=len(over_rows),
        rows_over_bar_by_actuator_sign=dict(Counter(f"{a}{sg}" for (st, ep, a, sg, f, t, n) in over_rows)),
        tail_unresolved_total=sum(pe[s]["unres"] for s in steps),
        any_missing_total=sum(len(pe[s]["any_missing"]) for s in steps),
        tail_missing_total=sum(len(pe[s]["tail_missing"]) for s in steps),
        ties_or_multi_over_bar=sum(1 for (st, ep, a, sg, f, t, n) in over_rows if t >= 1 or n >= 2),
        tool_window_shares=d1["window_shares"], windows=win, table=table,
        tel_rail_vs_m5_rows={"over_and_tel_rail": agree[(True, True)], "over_no_tel_rail": agree[(True, False)],
                              "tel_rail_not_over": agree[(False, True)], "neither": agree[(False, False)]})

    # ---------------------------------------------------------- probe (D-2, D-15, entropy)
    with np.load(probe_p, allow_pickle=False) as z:
        T = z["update_timesteps"].astype(np.int64)
        STD = z["std_per_actuator"].astype(np.float64)
        ENT = z["ent_coef"].astype(np.float64)
        LR = z["learning_rate"].astype(np.float64)
        M6 = {k: z[k].astype(np.float64) for k in ("rollout_mu_beyond_high_share", "rollout_mu_beyond_low_share",
                                                     "rollout_action_clip_high_share", "rollout_action_clip_low_share",
                                                     "rollout_action_saturated_share", "rollout_mu_mean",
                                                     "rollout_mu_q50", "rollout_mu_q95")}
        OBSM = z["obs_rms_mean"].astype(np.float64)
        OBSV = z["obs_rms_var"].astype(np.float64)
        rows_per = z["rollout_rows"]
    n_up = len(T)
    # ent_coef against the stage_config schedule
    sc = json.loads(sc_p.read_text())
    hp = sc["hyperparameters"]
    e0, e1, dT = float(hp["ent_coef"]), float(hp["ent_coef_end"]), float(hp["ent_coef_decay_timesteps"])
    sched = np.where(T < dT, e0 + (e1 - e0) * T / dT, e1)
    first_zero = int(T[np.argmax(ENT <= 0.0)]) if (ENT <= 0.0).any() else None

    def at(step):
        k = int(np.searchsorted(T, step, side="left"))
        k = min(k, n_up - 1)
        return k

    def ent_at(step):
        k = at(step)
        return dict(update_timesteps=int(T[k]), ent_coef=float(ENT[k]), schedule=float(sched[k]))

    R["settings"] = dict(stage_config=str(sc_p), ent_coef=e0, ent_coef_end=e1, ent_coef_decay_timesteps=dT,
                         log_std_init=hp.get("policy_kwargs", {}).get("log_std_init"),
                         policy_kwargs=hp.get("policy_kwargs"), symmetry=hp.get("symmetry"),
                         symmetry_kappa=hp.get("symmetry_kappa"), symmetry_sigma_floor=hp.get("symmetry_sigma_floor"),
                         symmetry_mirror_critic=hp.get("symmetry_mirror_critic"),
                         action_saturation_weight=sc["reward_weights"].get("action_saturation_weight"),
                         action_saturation_threshold=sc["reward_weights"].get("action_saturation_threshold"),
                         learning_rate=hp.get("learning_rate"), learning_rate_end=hp.get("learning_rate_end"),
                         n_steps=hp.get("n_steps"), batch_size=hp.get("batch_size"), n_epochs=hp.get("n_epochs"),
                         target_kl=hp.get("target_kl"), use_sde=hp.get("use_sde"))
    R["entropy_record"] = dict(updates=n_up, first_update=int(T[0]), last_update=int(T[-1]),
                               rows_per_rollout=sorted(set(int(v) for v in rows_per)),
                               ent_first=float(ENT[0]), ent_last=float(ENT[-1]),
                               max_abs_diff_from_schedule=float(np.max(np.abs(ENT - sched))),
                               first_update_at_zero=first_zero,
                               at={str(s): ent_at(s) for s in (1048576, 3670016, 3801088, 3932160, 4194304, 4849664,
                                                               5505024, 6291456, 7000000, 8126464, FINAL)})

    # per-update std CSV
    with open(OUT / f"std_per_update_{sr}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["update_timesteps", "ent_coef", "learning_rate"] + [f"std_{a}" for a in ACT])
        for k in range(n_up):
            wr.writerow([int(T[k]), f"{ENT[k]:.7g}", f"{LR[k]:.7g}"] + [f"{v:.6g}" for v in STD[k]])
    R["sources"]["std_per_update_csv"] = str(OUT / f"std_per_update_{sr}.csv")

    # D-15 std at K3 points and others
    std_at = {}
    for s in [0] + K3_STEPS:
        if s == 0:
            std_at["0 (log_std_init)"] = [round(math.exp(-1.5), 6)] * 15
        else:
            k = at(s)
            std_at[str(int(T[k]))] = [round(float(v), 6) for v in STD[k]]
    legs = [3, 4, 5, 6, 7, 8, 9, 10]
    below_008 = {}
    for i in range(15):
        idx = np.where(STD[:, i] < 0.08)[0]
        below_008[ACT[i]] = int(T[idx[0]]) if len(idx) else None
    pair_gap = float(np.max(np.abs(STD[:, 3:7] - STD[:, 7:11])))
    win_std = {}
    for name, lo, hi in WIN + WIN7:
        m = (T >= lo) & (T < hi)
        win_std[name] = {ACT[i]: dict(mean=r(STD[m, i].mean(), 5), start=r(STD[m, i][0], 5), end=r(STD[m, i][-1], 5))
                         for i in range(15)}
    R["D15"] = dict(source=str(probe_p), updates=n_up, std_at_k3_points=std_at,
                    first_update_below_0p08=below_008, leg_pair_max_abs_gap=pair_gap, window_std=win_std)

    # ---------------------------------------------------------- D-2 (M6)
    def m6_summary(i, lo, hi):
        m = (T >= lo) & (T < hi)
        o = {}
        for k in ("rollout_mu_beyond_high_share", "rollout_mu_beyond_low_share", "rollout_action_clip_high_share",
                  "rollout_action_clip_low_share", "rollout_action_saturated_share"):
            a = M6[k][m, i]
            o[k.replace("rollout_", "")] = dict(mean=r(a.mean(), 5), max=r(a.max(), 5),
                                                 max_at=int(T[m][int(np.argmax(a))]),
                                                 rollouts_ge_0p05=int((a >= 0.05).sum()), rollouts=int(m.sum()))
        for k in ("rollout_mu_mean", "rollout_mu_q50", "rollout_mu_q95"):
            a = M6[k][m, i]
            o[k.replace("rollout_", "")] = dict(mean=r(a.mean(), 4), min=r(a.min(), 4), max=r(a.max(), 4))
        return o

    d2 = {"source": str(probe_p), "rollout_step_meaning": "update_timesteps of the PPO update the rollout fed",
          "actuators": {}}
    for i in (11, 12, 13, 14, 4, 8, 0):
        d2["actuators"][ACT[i]] = {name: m6_summary(i, lo, hi) for name, lo, hi in WIN + WIN7 + [("whole", 0, FINAL + 1)]}
    t3 = M6["rollout_mu_beyond_high_share"][:, 14]
    idx = np.where(t3 >= 0.05)[0]
    first05 = int(T[idx[0]]) if len(idx) else None
    idx01 = np.where(t3 >= 0.01)[0]
    first01 = int(T[idx01[0]]) if len(idx01) else None
    idx_after7 = np.where((t3 >= 0.05) & (T >= 7000000))[0]
    onset = R["D1"]["first_tail_saturated_episode"]
    flag = R["D1"]["first_tail_flagged_evaluation"]
    alarm = R["D1"]["tail_alarm_steps"][0] if R["D1"]["tail_alarm_steps"] else None
    d2["tail_3_first_rollout_mu_beyond_high_ge_0p05"] = first05
    d2["tail_3_first_rollout_mu_beyond_high_ge_0p01"] = first01
    d2["tail_3_rollouts_ge_0p05_after_7M"] = int(len(idx_after7))
    d2["tail_3_first_ge_0p05_after_7M"] = int(T[idx_after7[0]]) if len(idx_after7) else None
    d2["lead_lag_steps_vs_D1"] = {
        "vs_first_tail_saturated_episode": (first05 - onset) if (first05 is not None and onset is not None) else None,
        "vs_first_tail_flagged_evaluation": (first05 - flag) if (first05 is not None and flag is not None) else None,
        "vs_first_tail_alarm": (first05 - alarm) if (first05 is not None and alarm is not None) else None,
        "sign": "negative = the rollout share reached 0.05 before the ledger event (drive leads)"}
    # neck_pitch low-bound series (s80's actuator) - report-only extension of D-2's set
    nk = M6["rollout_mu_beyond_low_share"][:, 0]
    idxn = np.where(nk >= 0.05)[0]
    d2["neck_pitch_first_rollout_mu_beyond_low_ge_0p05"] = int(T[idxn[0]]) if len(idxn) else None
    idxn1 = np.where(nk >= 0.01)[0]
    d2["neck_pitch_first_rollout_mu_beyond_low_ge_0p01"] = int(T[idxn1[0]]) if len(idxn1) else None
    # train_episodes.csv: training falls by window (no actuator columns in this file)
    th_, trows = read_csv(te_p)
    tw = {}
    for name, lo, hi in WIN + WIN7:
        sel = [x for x in trows if in_win(int(float(x["timesteps"])), lo, hi)]
        tw[name] = dict(episodes=len(sel), not_truncated=sum(1 for x in sel if x.get("cause") != "truncated"),
                        causes=dict(Counter(x.get("cause") for x in sel if x.get("cause") != "truncated")))
    d2["train_episodes"] = dict(source=str(te_p), columns=th_, by_window=tw)
    R["D2"] = d2

    # ---------------------------------------------------------- D-13 (M8) and D-11 ledger com/pitch
    m8 = L["m8"]["per_evaluation"]
    m8 = {int(k): v for k, v in m8.items()}
    numeric = [c for c in TEL_COLS if c not in ("rail_actuators", "parked_actuators")]
    d13 = {"source": str(W / f"ledger_{sr}.json") + " m8 (mean over full-horizon episodes per evaluation)",
           "absent": L["m8"]["absent"], "windows": {}, "statue": {}, "final_eval": {}, "first_eval": {}, "extremes": {}}
    for c in numeric:
        series = [(s, m8[s].get(c)) for s in sorted(m8) if m8[s].get(c) is not None]
        for name, lo, hi in WIN:
            d13["windows"].setdefault(name, {})[c] = r(mean([v for s, v in series if in_win(s, lo, hi)]), 5)
        if c in STATUE:
            d13["statue"][c] = r(STATUE[c], 5)
        if series:
            d13["final_eval"][c] = r(series[-1][1], 5)
            d13["first_eval"][c] = r(series[0][1], 5)
            vmin = min(series, key=lambda t: t[1])
            vmax = max(series, key=lambda t: t[1])
            d13["extremes"][c] = dict(min=r(vmin[1], 5), min_at=vmin[0], max=r(vmax[1], 5), max_at=vmax[0])
    # text columns: rail / parked names per window (full-horizon episodes)
    for c in ("rail_actuators", "parked_actuators"):
        for name, lo, hi in WIN:
            cnt = Counter()
            for s in sorted(m8):
                if in_win(s, lo, hi):
                    for k, v in (m8[s].get(c) or {}).items():
                        cnt[k] += v
            d13["windows"].setdefault(name, {})[c] = dict(cnt)
    # HEAD LOW evaluations (window-mean head tip < 0.75 m)
    d13["head_low_evaluations"] = [s for s in sorted(m8) if (m8[s].get("head_tip_height_mean_m") or 9) < 0.75]
    d13["head_low_rows"] = sum(1 for x in rows if (num(x.get("head_tip_height_mean_m")) or 9) < 0.75)
    d13["full_horizon_episodes_min"] = min(m8[s]["full_horizon_episodes"] for s in m8)
    d13["evaluation_falls"] = [(int(float(x["timesteps"])), int(float(x["episode"])), int(float(x["length"])))
                               for x in rows if (num(x.get("length")) or 0) < 1000]
    R["D13"] = d13

    # ledger com_fore_aft_mean_m and root_pitch_mean_deg per evaluation (D-11)
    cp = {}
    for c in ("com_fore_aft_mean_m", "root_pitch_mean_deg"):
        per = {}
        for s in steps:
            vals = [num(x.get(c)) for x in by_step[s] if (num(x.get("length")) or 0) >= 1000]
            per[s] = mean(vals)
        series = [(s, v) for s, v in per.items() if v is not None]
        cp[c] = dict(statue=r(STATUE.get(c), 5),
                     windows={name: r(mean([v for s, v in series if in_win(s, lo, hi)]), 5) for name, lo, hi in WIN},
                     first=(series[0][0], r(series[0][1], 5)), final=(series[-1][0], r(series[-1][1], 5)),
                     min=min(series, key=lambda t: t[1]), max=max(series, key=lambda t: t[1]),
                     per_evaluation={str(s): r(v, 5) for s, v in series})
        cp[c]["min"] = (cp[c]["min"][0], r(cp[c]["min"][1], 5))
        cp[c]["max"] = (cp[c]["max"][0], r(cp[c]["max"][1], 5))
    # ---------------------------------------------------------- D-11 training-history report
    th = json.loads(th_p.read_text())
    run0 = th["runs"][0]
    pages_dir = led_csv.parent / "figures/training_history"
    pages = []
    for rel in sorted(verified[rid]["paths"]):
        if rel.startswith("01_stance/figures/training_history/"):
            pages.append(dict(page=rel.split("/")[-1], verified=verified[rid]["paths"][rel]["verified"] is not None,
                              sha256=verified[rid]["paths"][rel]["sha256"]))
    sat = run0["saturation"]
    ckt = []
    for row in run0["checkpoint_table"]:
        ckt.append({k: row.get(k) for k in ("at", "timesteps", "episodes", "full_horizon", "clean", "falls",
                                             "saturated", "flagged_actuators", "com_fore_aft_mean_m",
                                             "head_tip_height_mean_m", "neck_pitch_mean_deg", "tail_tip_chord_mean_deg",
                                             "std_checkpoint", "std_tail_3_pitch", "std_r_hip_roll")})
    obsn = run0["obs_normalization"]
    pc = obsn.get("probe", {}).get("pickle_checks", [])
    ck = obsn.get("checkpoints", [])
    drift = []
    for c in ck:
        drift.append(dict(name=c.get("name"), timesteps=c.get("timesteps"), shift_prev_max=c.get("shift_prev_max"),
                          shift_prev_median=c.get("shift_prev_median"), scale_prev_max=c.get("scale_prev_max"),
                          shift_final_max=c.get("shift_final_max"), scale_final_max=c.get("scale_final_max"),
                          top=[(t["dim"], t["shift"]) for t in (c.get("shift_prev_top") or [])],
                          bank_change_inf=c.get("bank_change_inf")))
    R["D11"] = dict(source=str(th_p), pages_dir=str(pages_dir), pages=pages, schema=th.get("schema"),
                    generated_at=th.get("generated_at"), title=th.get("title"), errors=th.get("errors"),
                    run_errors=run0.get("errors"), absent=run0.get("absent"), body_unmeasured=run0.get("body_unmeasured"),
                    not_recorded_count=len(th.get("not_recorded") or []), plant=run0.get("plant", {}).get("status"),
                    status=run0.get("status"), final_step=run0.get("final_step"),
                    saturation=dict(first_saturated_episode_at=sat.get("first_saturated_episode_at"),
                                    first_flagged_at=sat.get("first_flagged_at"), n_flagged=len(sat.get("flagged") or []),
                                    alarm=sat.get("alarm"), first_alarm_at=sat.get("first_alarm_at"),
                                    runs=sat.get("runs"),
                                    equals_ledger_any_flagged=(sat.get("flagged") == L["any_flagged"])),
                    checkpoint_steps=run0.get("checkpoint_steps"), checkpoint_table=ckt,
                    rollout_named=run0["rollout_shares"].get("named"),
                    rollout_level=run0["rollout_shares"].get("level"),
                    rollout_first_at_level={a: v.get("first_at_level") for a, v in run0["rollout_shares"]["per_actuator"].items()
                                            if any(x is not None for x in (v.get("first_at_level") or {}).values())},
                    rollout_peaks={a: v.get("max") for a, v in run0["rollout_shares"]["per_actuator"].items()
                                   if a in ("tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act",
                                            "neck_pitch_act", "r_hip_roll_act", "l_hip_roll_act")},
                    policy_std_checkpoints_read=len(run0["policy_std"].get("checkpoints") or []),
                    exploration_reference=run0["exploration"].get("reference"),
                    exploration_pairs_tied=run0["exploration"].get("pairs_tied"),
                    obs_norm_most_drifting_dims=obsn.get("most_drifting_dims"),
                    obs_norm_checkpoints=len(ck), obs_norm_errors=obsn.get("errors"),
                    probe_pickle_checks=dict(n=len(pc), equal_float32=sum(1 for p in pc if p.get("equal_float32"))),
                    obs_norm_drift=drift, judge_k3=run0.get("judge_k3"),
                    ledger_com_and_pitch=cp)

    # ---------------------------------------------------------- D-14 (M7)
    K = json.loads((W / f"k3panel_{sr}.json").read_text())
    R["D14"] = dict(tool_json=str(W / f"k3panel_{sr}.json"), tool_stdout=str(W / f"k3panel_{sr}.txt"), raw=K)
    # own_sigma vs probe std at the panel's step
    kj = json.loads(vpath(rid, "01_stance/k3_panel.json").read_text())
    own_vs_probe = []
    for p in kj.get("panels", []):
        s = int(p["timesteps"])
        k = at(s)
        osig = p.get("own_sigma")
        if osig is None:
            osig = (p.get("tests", {}).get("own", {}) or {}).get("sigma")
        if osig is not None:
            own_vs_probe.append(dict(step=s, probe_update=int(T[k]),
                                     max_abs_diff=float(np.max(np.abs(np.array(osig, dtype=float) - STD[k])))))
    R["D14"]["own_sigma_vs_probe_std"] = own_vs_probe
    R["D14"]["panel_keys"] = sorted(kj.get("panels", [{}])[0].keys()) if kj.get("panels") else []
    # judge K3 (M4) presence
    jd = {}
    for lab in ("6291456", "final"):
        d = ROUND / "panels" / rid / lab / "m4"
        jd[lab] = sorted(str(p) for p in d.glob("*")) if d.exists() else "pending (no m4 folder)"
    R["D14"]["judge_m4"] = jd

    # ---------------------------------------------------------- entropy check: std at onsets, cross-run
    ev_onsets = {"first_tail_saturated_episode": R["D1"]["first_tail_saturated_episode"],
                 "first_tail_flagged_evaluation": R["D1"]["first_tail_flagged_evaluation"],
                 "first_tail_alarm": alarm,
                 "first_any_saturated_episode": first_any_sat,
                 "first_any_flagged_evaluation": L["any_flagged"][0] if L["any_flagged"] else None,
                 "first_any_alarm": L["any_alarms"][0] if L["any_alarms"] else None}
    R["onsets"] = ev_onsets
    R["_T"] = T.tolist()
    R["_STD"] = STD.tolist()
    R["_ENT"] = ENT.tolist()
    R["_t3hi"] = t3.tolist()
    R["_nklo"] = nk.tolist()
    R["progress_policy"] = json.loads(pr_p.read_text()).get("policy")
    R["wall_clock_s"] = json.loads(pr_p.read_text()).get("wall_clock_s")

    # ---------------------------------------------------------- interim: prefix check of the snapshot ledgers
    inter = {}
    for k, d in SNAP.items():
        sp_ = d / rid / "eval_episodes.csv"
        if not sp_.exists():
            inter[k] = "absent"
            continue
        h2, r2 = read_csv(sp_)
        same_header = h2 == hdr
        n2 = len(r2)
        eq = all(r2[i] == rows[i] for i in range(n2)) if same_header and n2 <= len(rows) else False
        last = int(float(r2[-1]["timesteps"])) if r2 else None
        # K3 panels of the snapshot vs F2's
        kp = d / rid / "k3_panel.json"
        k3eq = None
        if kp.exists():
            ks = json.loads(kp.read_text())
            fp = {int(p["timesteps"]): (p.get("own_falls"), p.get("joint_falls")) for p in kj.get("panels", [])}
            sp2 = {int(p["timesteps"]): (p.get("own_falls"), p.get("joint_falls")) for p in ks.get("panels", [])}
            k3eq = all(fp.get(s) == v for s, v in sp2.items())
            k3n = len(sp2)
        inter[k] = dict(file=str(sp_), sha256=sha(sp_), rows=n2, last_eval=last, header_equal=same_header,
                        rows_equal_to_F2_prefix=eq, k3_panels=k3n if kp.exists() else None,
                        k3_falls_equal_to_F2=k3eq)
    R["interim_prefix"] = inter
    data["runs"][rid] = R

# ------------------------------------------------------------------ cross-run std at onsets
cross = {}
probe_steps = [3670016, 3801088, 3932160, 4849664, 5505024, 7000000, 8126464, FINAL]
for s in probe_steps:
    row = {}
    for rid in RUNS:
        T = np.array(data["runs"][rid]["_T"])
        STD = np.array(data["runs"][rid]["_STD"])
        k = min(int(np.searchsorted(T, s)), len(T) - 1)
        row[SHORT[rid]] = {ACT[i]: round(float(STD[k, i]), 5) for i in (0, 1, 2, 4, 11, 12, 13, 14)}
        row[SHORT[rid]]["update_timesteps"] = int(T[k])
    cross[str(s)] = row
data["cross_run_std"] = cross

# std change across the onset (per run): +-524,288 around the first alarm (tail or any)
around = {}
for rid in RUNS:
    R = data["runs"][rid]
    T = np.array(R["_T"]); STD = np.array(R["_STD"])
    ons = R["onsets"]["first_tail_alarm"] or R["onsets"]["first_any_alarm"] or R["onsets"]["first_tail_saturated_episode"]
    if ons is None:
        continue
    out = {"onset": ons}
    for d in (-1048576, -524288, 0, 524288, 1048576):
        k = min(int(np.searchsorted(T, ons + d)), len(T) - 1)
        out[str(d)] = {ACT[i]: round(float(STD[k, i]), 5) for i in (0, 2, 11, 12, 13, 14)}
    # log-std slope per 1M steps before and after
    def slope(lo, hi, i):
        m = (T >= lo) & (T < hi)
        if m.sum() < 3:
            return None
        return float(np.polyfit(T[m] / 1e6, np.log(STD[m, i]), 1)[0])
    out["log_std_slope_per_1M_before"] = {ACT[i]: r(slope(ons - 1048576, ons, i), 4) for i in (0, 11, 13, 14)}
    out["log_std_slope_per_1M_after"] = {ACT[i]: r(slope(ons, ons + 1048576, i), 4) for i in (0, 11, 13, 14)}
    around[SHORT[rid]] = out
data["std_around_onset"] = around

# r9 ledger summaries (pinned tool on the r9 LTSB ledgers) and r9 final policy std
r9 = {}
for n, p in (("stance5_LTSB_s52", "runs/stance5_LTSB_s52"), ("stance5_LTSB_s60", "runs/stance5_LTSB_s60"),
             ("stance5_LTSB_s42", "runs/stance5_LTSB_s42"), ("stance5_LTSB_s70", "replication/runs/stance5_LTSB_s70"),
             ("stance5_LTSB_s80", "replication/runs/stance5_LTSB_s80")):
    J = json.loads((W / f"r9ledger_{n}.json").read_text())
    pe = {int(k): v for k, v in J["per_evaluation"].items()}
    fl = [s for s in sorted(pe) if pe[s]["any"] >= 5]
    over = {name: sum(pe[s]["any"] for s in pe if in_win(s, lo, hi)) for name, lo, hi in WIN7}
    rows_ = {name: sum(pe[s]["n"] for s in pe if in_win(s, lo, hi)) for name, lo, hi in WIN7}
    base = S0 / "foot/stance5" / p / "01_stance"
    sc = json.loads((base / "stage_config.json").read_text())["hyperparameters"]
    pol = json.loads((base / "progress.json").read_text()).get("policy")
    r9[n] = dict(ledger=str(base / "eval_episodes.csv"), ledger_sha256=J["sha256"], summary=J["r9_summary"],
                 flagged_before_7M=sum(1 for s in fl if s < 7000000), flagged_after_7M=sum(1 for s in fl if s >= 7000000),
                 alarms=J["any_alarms"], alarms_after_7M=[a for a in J["any_alarms"] if a >= 7000000],
                 over_rows_by_window=over, rows_by_window=rows_,
                 schedule=dict(ent_coef=sc.get("ent_coef"), ent_coef_end=sc.get("ent_coef_end"),
                               ent_coef_decay_timesteps=sc.get("ent_coef_decay_timesteps"),
                               log_std_init=sc.get("policy_kwargs", {}).get("log_std_init"),
                               symmetry_sigma_floor=sc.get("symmetry_sigma_floor")),
                 final_policy=pol, progress=str(base / "progress.json"), stage_config=str(base / "stage_config.json"))
data["r9"] = r9

for rid in RUNS:
    for k in ("_T", "_STD", "_ENT", "_t3hi", "_nklo"):
        data["runs"][rid].pop(k)
(W / "readouts_data.json").write_text(json.dumps(data, indent=1, default=str) + "\n")
print("ok", W / "readouts_data.json")
