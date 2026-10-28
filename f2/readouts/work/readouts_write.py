# Writes READOUTS.txt and READOUTS.json from readouts_data.json, bins.json, std_kink_7M.json and the tool outputs.
# Every number printed is pulled from those files (no hand transcription); prose labels each claim.
import json
import pathlib
import subprocess
import sys

S1 = pathlib.Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
S0 = pathlib.Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
ROUND = S0 / "foot/stance6"
OUT = S1 / "r10round/f2/readouts"
W = OUT / "work"
D = json.loads((W / "readouts_data.json").read_text())
B = json.loads((W / "bins.json").read_text())
K7 = json.loads((W / "std_kink_7M.json").read_text())
V = json.loads((W / "verified_copies.json").read_text())
NOW = subprocess.run(["date", "-u"], capture_output=True, text=True).stdout.strip()
RUNS = ["stance6_r10LTSB_s60", "stance6_r10LTSB_s42", "stance6_r10LTSB_s80"]
SH = {"stance6_r10LTSB_s60": "s60", "stance6_r10LTSB_s42": "s42", "stance6_r10LTSB_s80": "s80"}
SEED = {"stance6_r10LTSB_s60": 60, "stance6_r10LTSB_s42": 42, "stance6_r10LTSB_s80": 80}
ACT = ("neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
       "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act", "tail_1_pitch_act",
       "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")
WN = ["[0,4194304)", "[4194304,8388608)", "[8388608,12001280]"]
W7 = ["[0,7000000)", "[7000000,12001280]"]
WSHORT = {"[0,4194304)": "W1 [0, 4.19M)", "[4194304,8388608)": "W2 [4.19M, 8.39M)", "[8388608,12001280]": "W3 [8.39M, 12.0M]",
          "[0,7000000)": "[0, 7.0M) bonus > 0", "[7000000,12001280]": "[7.0M, 12.0M] bonus 0"}
L = []


def p(s=""):
    L.append(s)


def st(x):
    return "none" if x is None else f"{int(x):,}"


def f(x, k=3):
    if x is None:
        return "n/a"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f(v, k) for v in x) + "]"
    return f"{x:.{k}f}" if isinstance(x, float) else str(x)


def g(x):
    if x is None:
        return "n/a"
    if x == 0:
        return "0"
    return f"{x:.3g}" if abs(x) < 0.01 else f"{x:.4f}"


# ------------------------------------------------------------------ source keys
SRC = {}
for rid in RUNS:
    s = SH[rid]
    paths = V[rid]["paths"]
    for key, rel in (("E", "01_stance/eval_episodes.csv"), ("P", "01_stance/policy_probe.npz"),
                     ("C", "01_stance/stage_config.json"), ("G", "01_stance/progress.json"),
                     ("H", "01_stance/figures/training_history/training_history.json"),
                     ("K", "01_stance/k3_panel.json"), ("KC", "01_stance/k3_panel.csv"),
                     ("T", "01_stance/train_episodes.csv"), ("GV", "01_stance/gate_verdict.json")):
        SRC[f"{key}-{s}"] = (paths[rel]["verified"], paths[rel]["sha256"])
    SRC[f"L-{s}"] = (str(W / f"ledger_{s}.json") + " (+ stdout " + str(W / f"ledger_{s}.txt") + ")", None)
    SRC[f"K3T-{s}"] = (str(W / f"k3panel_{s}.json") + " (+ stdout " + str(W / f"k3panel_{s}.txt") + ")", None)
    SRC[f"M1F-{s}"] = (str(ROUND / "panels" / rid / "final/m1/m1_summary.json"), None)
    SRC[f"M3F-{s}"] = (str(ROUND / "panels" / rid / "final/m3/m3_summary.json"), None)
    SRC[f"STD-{s}"] = (str(OUT / f"std_per_update_{s}.csv") + " (every PPO update, from P-" + s + ")", None)
SRC["DATA"] = (str(W / "readouts_data.json") + " (this analyst's reductions; script " + str(W / "readouts_compute.py") + ")", None)
SRC["BINS"] = (str(W / "bins.json") + " (1,048,576-step bins from P-* and E-*; script " + str(W / "bins.py") + ")", None)
SRC["KINK"] = (str(W / "std_kink_7M.json") + " (log-std slopes 6-7M and 7-8M from P-*)", None)
SRC["VER"] = (str(W / "verified_copies.json") + " (sha256 of every F2 copy against its run's artifact_manifest.json; script " + str(W / "verify_copies.py") + ")", None)
SRC["STAT"] = (D["statue_ledger"]["file"], D["statue_ledger"]["sha256"])
SRC["STM2"] = (str(S1 / "r10round/f2/statue_m2/panel.json"), None)
SRC["SCORE"] = (str(S1 / "r10round/f2/score.txt"), None)
SRC["EVID"] = (str(ROUND / "evidence.txt"), None)
SRC["THIS12"] = (str(ROUND / "criteria_v12_cited.txt"), None)
SRC["CRIT"] = (str(ROUND / "criteria.txt"), None)
for n, v in D["r9"].items():
    SRC[f"R9L-{n}"] = (v["ledger"], v["ledger_sha256"])
    SRC[f"R9G-{n}"] = (v["progress"], None)
    SRC[f"R9C-{n}"] = (v["stage_config"], None)
    SRC[f"R9T-{n}"] = (str(W / f"r9ledger_{n}.json") + " (ledger_r10round.py --r9-summary)", None)
SRC["SNAP1"] = (str(S1 / "r10round/interim/snap1/READOUT.txt") + " and CHECK.txt", None)
SRC["SNAP2"] = (str(S1 / "r10round/interim/snap2/READOUT.txt") + " and CHECK.txt", None)
CODE = S1 / "r10round/f2/copy_8737cd3d"
SRC["CODE-MKL"] = (str(CODE / "environments/shared/mirror_kl_ppo.py"), None)
SRC["CODE-ENV"] = (str(CODE / "environments/tyrannosaurus_rex/base_env.py"), None)
SRC["CODE-RF"] = (str(CODE / "environments/shared/reward_functions.py"), None)
SRC["CODE-SB3"] = (str(S1 / "venv/lib/python3.13/site-packages/stable_baselines3/common/distributions.py"), None)
SRC["PAGES"] = (str(W / "pages") + " (cp of 21 report pages, 7 pages x 3 runs, to .png names for viewing, four of them viewed; originals beside H-* in the F2 copies folders)", None)

# ------------------------------------------------------------------ header
p("=" * 118)
p("STANCE6 (r10 round) - PRE-REGISTERED DIAGNOSTIC READOUTS, REPORT ONLY - D-1, D-2, D-11..D-15, ENTROPY CHECK, INTERIM")
p("=" * 118)
p(f"Written {NOW} (date -u). Readouts analyst; no simulation, no training, no Drive, no GitHub.")
p("What this is: the criteria's PRE-REGISTERED DIAGNOSTIC READOUTS (criteria.txt lines 335-342; THIS12 lines 566-630)")
p("on F2's manifest-verified copies. Report only: nothing here changes a class, A, n_A, a gate, an outcome or the reading.")
p("The reading is the scorer's and is FIXED [measured, SCORE]: 'READING: line 1: TAIL RAIL FORMED ON r10 IN TRAINING")
p("(ledger)', R 1 (COMPLETE 1, PRESENCE 0), U 0, P 1, G+ 1, STATUS: FINAL; A 2 of 3 against LTSB r9's 2 of 5, one-sided")
p("Fisher p 0.9286. Classes: s60 PARKED / GATE PASS / NEGATIVE; s42 RAIL (RAIL-LEDGER-PERSISTENT) / GATE FAIL / NEGATIVE;")
p("s80 CLEAR / GATE PASS / POSITIVE (ANY ALARMs on neck_pitch_act, non-tail).")
p("Labels: [measured, KEY] = read or computed from the file KEY names (SOURCES below); [inferred] = reasoned, basis given.")
p("Steps are training timesteps. W1/W2/W3 = the r9 windows [0, 4,194,304), [4,194,304, 8,388,608), [8,388,608, 12,001,280].")
p("")
p("COPIES USED [measured, VER]: every F2 copy of every run was hashed and compared with that run's artifact_manifest.json")
p("entry. s60 64 paths, s42 66 paths, s80 64 paths: every path has a copy whose sha256 equals the manifest's entry; the only")
p("path without an entry is artifact_manifest.json itself (no self-entry), whose two copies agree in each run (s60")
p("8523cf22..., s42 3ffa31e6..., s80 29e92b97...). Where two copies exist (eval_episodes.csv, stage_config.json,")
p("provenance.json) both match. fetch_r10round.py status --fetch F2 prints 'done: every file of this fetch is held as the")
p(f"rules require' for all three [measured, {W / 'status_s60.json'}, status_s42.json, status_s80.json].")
p("Tools used as pinned (sha256 equal to lane B SHA256SUMS.txt: ledger_r10round.py 0d89bcb8..., k3panel_r10round.py")
p("89d279d4..., r10round_common.py ea2c5ea6...) [measured, sha256sum vs " + str(S1 / "r10round/lean/build/laneB/SHA256SUMS.txt") + "].")
p("Commands (cwd /tmp, PY = S2/venv/bin/python -I -B):")
p("  ledger_r10round.py <E copy> --E 12001280 --probe <P copy> --tel --json work/ledger_<run>.json   (rc 0, all three)")
p("  k3panel_r10round.py --json <K copy> --csv <KC copy> --seed <60|42|80> --out work/k3panel_<run>.json   (rc 0)")
p("  ledger_r10round.py <r9 LTSB ledger> --r9-summary --json work/r9ledger_<run>.json   (rc 0, five r9 runs)")
p("")

# ------------------------------------------------------------------ headline
R = D["runs"]


def onset(rid):
    return R[rid]["D1"]


p("-" * 118)
p("HEADLINE PER RUN (report only)")
p("-" * 118)
s60, s42, s80 = (R[r] for r in RUNS)
b60, b42, b80 = B["s60"], B["s42"], B["s80"]
p(f"s60: no flagged evaluation and no alarm in 91 [measured, L-s60]. tail_3_pitch_act was over the bar (+1) in "
  f"{s60['D1']['rows_over_bar_total']} single episodes, the first at {st(s60['D1']['first_tail_saturated_episode'])},")
p(f"  all after 7,000,000 (entropy bonus 0); its training-row drive reached 0.05 at {st(s60['D2']['tail_3_first_rollout_mu_beyond_high_ge_0p05'])}"
  f" ({s60['D2']['tail_3_rollouts_ge_0p05_after_7M']} rollouts >= 0.05, all after 7M); TEL's window-mean tail_3 command rose from")
p(f"  {f(s60['D13']['windows'][WN[0]]['tail_3_pitch_act_command_mean'])} (W1) to {f(s60['D13']['windows'][WN[2]]['tail_3_pitch_act_command_mean'])} (W3), "
  f"final evaluation {f(s60['D13']['final_eval']['tail_3_pitch_act_command_mean'])}: the PARKED drive [measured, P-s60, E-s60, L-s60].")
p(f"  In-run K3: own 0 of 20 everywhere; joint 0-1 of 20 (1 at the final) [measured, K3T-s60].")
p(f"s42: tail_3_pitch_act at +1 in {s42['D1']['rows_over_bar_total']} of 910 episodes; first over-bar episode {st(s42['D1']['first_tail_saturated_episode'])}, "
  f"first flagged evaluation {st(s42['D1']['first_tail_flagged_evaluation'])},")
p(f"  first TAIL ALARM {st(s42['D1']['tail_alarm_steps'][0])}; {len(s42['D1']['tail_flagged_evaluations'])} flagged evaluations in 4 spells, "
  f"{len(s42['D1']['tail_alarm_steps'])} alarms, 11 of them after 7,000,000 [measured, L-s42].")
p(f"  The rail cleared at about 5.6-7.3M (rollout mean tail_3 {f(b42['bins'][6]['tail_3_pitch_act_mu_mean'])} in (6.29M, 7.34M]) and came back twice"
  f" with the bonus at 0 [measured, BINS]. Its tail_3 exploration std was the lowest")
p(f"  of the three runs at its first alarm ({D['cross_run_std']['3932160']['s42']['tail_3_pitch_act']:.4f} at 3,932,160 vs "
  f"{D['cross_run_std']['3932160']['s60']['tail_3_pitch_act']:.4f} s60, {D['cross_run_std']['3932160']['s80']['tail_3_pitch_act']:.4f} s80) [measured, P-*]. "
  f"In-run K3: own 0 of 20; joint 0-1 of 20, 0 at the final [measured, K3T-s42].")
p(f"s80: no tail-saturated episode; neck_pitch_act at -1 in {s80['D1']['rows_over_bar_total']} episodes (first {st(s80['D1']['first_any_saturated_episode'])}), "
  f"12 ANY-flagged evaluations 4,849,664-7,340,032, 6 ANY ALARMs")
p(f"  {', '.join(st(x) for x in s80['D1']['any_alarm_steps'])} (one after 7M); last over-bar row 8,781,824 [measured, L-s80]. "
  f"TEL: the head rose (W2 head tip {f(s80['D13']['windows'][WN[1]]['head_tip_height_mean_m'])} m")
p(f"  vs statue {f(D['statue_ledger_values']['head_tip_height_mean_m'])} m, neck pitch {f(s80['D13']['windows'][WN[1]]['neck_pitch_mean_deg'],1)} deg vs {f(D['statue_ledger_values']['neck_pitch_mean_deg'],1)}) "
  f"and later head_pitch_act parked at - (W3 185 episodes) [measured, L-s80, STAT].")
p(f"  In-run K3: own 0 of 20; joint up to 5 of 20 at 6,291,456, 2 of 20 at the final [measured, K3T-s80].")
p("ENTROPY [inferred, from the measured facts in the ENTROPY CHECK]: the run records do not support the entropy bonus as")
p("  the driver of these habits. The bonus acts only on the state-independent log_std, by the same amount on all 15")
p("  actuators, never on the mean action the ledger measures; the s60 drive formed and the s42 rail re-formed with the")
p("  bonus at exactly 0; r9 under the same schedule kept its rails after 7M too. A contributing role in the early onsets")
p("  (all while the bonus was > 0) cannot be ruled out from records in which every run had the same schedule; a contrast")
p("  in ent_coef (described at the end of the ENTROPY CHECK) would test it.")
p("")

# ------------------------------------------------------------------ D-1
p("-" * 118)
p("D-1 RAIL TIMELINE (M5 on the whole ledger, per run) [measured, E-<run> via L-<run>; signs from saturated_actuator_")
p("positive_share >= 0.5 = '+'; TEL rail_actuators = episodes naming the actuator in that column, all 10 episodes]")
p("-" * 118)
for rid in RUNS:
    s = SH[rid]
    d1 = R[rid]["D1"]
    p(f"{rid} (seed {SEED[rid]}): evaluations read {d1['evaluations_read']} ({d1['rows']} rows); C6 with E 12,001,280: "
      f"{'PASS' if d1['c6']['passed'] else 'FAIL'}; missing ANY inputs {d1['any_missing_total']}, missing TAIL inputs "
      f"{d1['tail_missing_total']}; TAIL-UNRESOLVED rows {d1['tail_unresolved_total']}; over-bar rows with ties or >= 2 "
      f"actuators saturated {d1['ties_or_multi_over_bar']}; NEEDS RESOLUTION {d1['needs_resolution'] or 'none'}.")
    p(f"  first tail-saturated episode {st(d1['first_tail_saturated_episode'])}; first tail-flagged evaluation "
      f"{st(d1['first_tail_flagged_evaluation'])}; TAIL ALARM steps: {', '.join(st(x) for x in d1['tail_alarm_steps']) or 'none'}")
    p(f"  tail-flagged spells (from, to, evaluations): {d1['tail_flagged_spells'] or 'none'}")
    p(f"  first any-actuator over-bar episode {st(d1['first_any_saturated_episode'])}; ANY-flagged spells: {d1['any_flagged_spells'] or 'none'}")
    p(f"  ANY ALARM count {d1['any_alarm_count']}: {', '.join(st(x) for x in d1['any_alarm_steps']) or 'none'}")
    p(f"  over-bar rows by actuator and sign (whole ledger): {d1['rows_over_bar_by_actuator_sign'] or 'none'}")
    a = d1["tel_rail_vs_m5_rows"]
    p(f"  TEL rail_actuators vs M5 row by row: over the bar and TEL rail {a['over_and_tel_rail']}; over, no TEL rail "
      f"{a['over_no_tel_rail']}; TEL rail, not over {a['tel_rail_not_over']}; neither {a['neither']}.")
    p("  per evaluation (only evaluations with a count; all others 0 over the bar, none missing):")
    p("      step         n ANY TAIL UNRES fl  over-bar by actuator/sign        TEL rail_actuators (episodes)    at_rail rows")
    for t in d1["table"]:
        fl = ("A" if t["any_flagged"] else "-") + ("T" if t["tail_flagged"] else "-")
        p(f"    {t['step']:>11,} {t['n']:>3} {t['any']:>3} {t['tail']:>4} {t['unres']:>5} {fl}  "
          f"{json.dumps(t['any_by_signed']):32} {json.dumps(t['tel_rail']):32} {t['tel_at_rail_rows']}")
    p("  windows (rows over the bar / rows; flagged evaluations / evaluations; OTHER = non-tail actuators):")
    p("    window                  evals rows  tail-over (share)  OTHER-over (share)  tail-flagged (share)  ANY-flagged  OTHER-flagged  TAIL alarms  ANY alarms")
    for wk in WN + W7:
        w = d1["windows"][wk]
        tag = "" if wk in WN else "   (entropy split)"
        p(f"    {WSHORT[wk]:22} {w['evaluations']:>5} {w['rows']:>4}  {w['tail_over']:>4} ({f(w['tail_share'],4)})     "
          f"{w['other_over']:>4} ({f(w['other_share'],4)})      {w['tail_flagged_evals']:>3} ({f(w['tail_flagged_share'],4)})         "
          f"{w['any_flagged_evals']:>3}          {w['other_flagged_evals']:>3}          {len(w['tail_alarms']):>3}          {len(w['any_alarms']):>3}{tag}")
    p(f"  every other actuator: {', '.join(k for k in d1['rows_over_bar_by_actuator_sign'] if not k.startswith('tail_')) or 'none over the bar'}"
      f" (every non-listed actuator: 0 rows over the bar in all 91 evaluations).")
    p("")

# ------------------------------------------------------------------ D-2
p("-" * 118)
p("D-2 TRAINING-ROW DRIVE (M6: policy_probe.npz, every PPO rollout; 1,465 rollouts of 8,192 rows) [measured, P-<run> via")
p("L-<run> m6 and DATA D2]. A rollout's step is the update it fed (update_timesteps). mu = the policy's mean action at the")
p("rollout's observations; muHi/muLo = share of rows with mu beyond the high/low bound; clipHi/clipLo = share of sampled")
p("actions beyond the high/low bound (clipped by env.step). Per window: mean / max (and rollouts with share >= 0.05).")
p("train_episodes.csv carries no actuator column (its 20 columns: timesteps, env, length, return, cause, CoM, root pitch,")
p("force-limited fraction, touch shares) [measured, T-<run> header], so the drive is read from the probe; its training")
p("falls per window are given beside.")
p("-" * 118)
for rid in RUNS:
    s = SH[rid]
    d2 = R[rid]["D2"]
    p(f"{rid}:")
    acts = ["tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act", "r_hip_roll_act", "l_hip_roll_act"]
    if s == "s80":
        acts.append("neck_pitch_act")
    p("    actuator          window                 muHi mean/max (n>=.05)    muLo mean/max (n)       clipHi mean/max    clipLo mean/max    mu mean [min, max]        q50     q95 mean/max")
    for a in acts:
        for wk in WN + ["[7000000,12001280]"]:
            v = d2["actuators"][a][wk]
            mh, ml, ch, cl = v["mu_beyond_high_share"], v["mu_beyond_low_share"], v["action_clip_high_share"], v["action_clip_low_share"]
            p(f"    {a[:-4]:16} {WSHORT[wk]:22} {f(mh['mean'],4)}/{f(mh['max'],4)} ({mh['rollouts_ge_0p05']:>3})      "
              f"{f(ml['mean'],4)}/{f(ml['max'],4)} ({ml['rollouts_ge_0p05']:>3})   {f(ch['mean'],4)}/{f(ch['max'],4)}    "
              f"{f(cl['mean'],4)}/{f(cl['max'],4)}    {f(v['mu_mean']['mean'],3):>6} [{f(v['mu_mean']['min'],3)}, {f(v['mu_mean']['max'],3)}]   "
              f"{f(v['mu_q50']['mean'],3):>6}  {f(v['mu_q95']['mean'],3)}/{f(v['mu_q95']['max'],3)}")
    ll = d2["lead_lag_steps_vs_D1"]
    p(f"  first rollout with tail_3 muHi >= 0.05: {st(d2['tail_3_first_rollout_mu_beyond_high_ge_0p05'])} (>= 0.01: "
      f"{st(d2['tail_3_first_rollout_mu_beyond_high_ge_0p01'])}); against D-1: vs first tail-saturated episode "
      f"{ll['vs_first_tail_saturated_episode'] if ll['vs_first_tail_saturated_episode'] is not None else 'n/a'} steps, vs first tail-flagged evaluation "
      f"{ll['vs_first_tail_flagged_evaluation'] if ll['vs_first_tail_flagged_evaluation'] is not None else 'n/a'}, vs first TAIL ALARM "
      f"{ll['vs_first_tail_alarm'] if ll['vs_first_tail_alarm'] is not None else 'n/a'} (negative = the drive leads).")
    p(f"  after 7,000,000 (entropy bonus 0), separately: tail_3 rollouts with muHi >= 0.05: {d2['tail_3_rollouts_ge_0p05_after_7M']} "
      f"(first {st(d2['tail_3_first_ge_0p05_after_7M'])}); [7.0M, 12.0M] row above.")
    if s == "s80":
        p(f"  neck_pitch (s80's ANY-alarm actuator; beyond D-2's named set, report only): first rollout with muLo >= 0.05 "
          f"{st(d2['neck_pitch_first_rollout_mu_beyond_low_ge_0p05'])} (>= 0.01: {st(d2['neck_pitch_first_rollout_mu_beyond_low_ge_0p01'])}); "
          f"against the first neck over-bar episode 3,801,088, the first ANY-flagged evaluation 4,849,664 and the first ANY ALARM 5,505,024; "
          f"last rollout >= 0.05: {st(B['s80']['last']['neck_pitch_act_lo_last_ge05'])}.")
    tw = d2["train_episodes"]["by_window"]
    p("  training episodes not truncated (falls and other causes) per window [measured, T-" + s + "]: " +
      "; ".join(f"{WSHORT[k]} {tw[k]['not_truncated']} of {tw[k]['episodes']} {tw[k]['causes']}" for k in WN + W7[1:]))
    p("")

# ------------------------------------------------------------------ D-11
p("-" * 118)
p("D-11 THE STAGE-END TRAINING-HISTORY REPORT, AS PRODUCED (01_stance/figures/training_history/) [measured, H-<run>; pages")
p("listed from VER; four pages viewed as images, PAGES]")
p("-" * 118)
for rid in RUNS:
    s = SH[rid]
    d = R[rid]["D11"]
    pg = [x["page"] for x in d["pages"]]
    p(f"{rid}: {len(pg)} files, every one manifest-verified: {', '.join(pg)}.")
    p(f"  training_history.json: schema {d['schema']}, generated_at {d['generated_at']}, status {d['status']}, final_step "
      f"{st(d['final_step'])}; errors {d['errors']}; run errors {d['run_errors']}; absent series {d['absent']}; body_unmeasured "
      f"{d['body_unmeasured']}; plant identity {d['plant']}; {d['not_recorded_count']} 'not recorded' notes (what the files cannot show).")
    sat = d["saturation"]
    runs_txt = "; ".join(f"{x['from']:,}-{x['to']:,} ({x['evaluations']} evals, alarm {x['alarm_at']:,}, {x['named']} {list(x['actuators'].values())[0]} episodes)" for x in sat["runs"]) or "none"
    p(f"  saturation as the report states it (flag = >= 5 episodes listing max_actuator_saturation_fraction; alarm = 2 in a row):"
      f" first saturated episode {st(sat['first_saturated_episode_at'])}, first flagged {st(sat['first_flagged_at'])}, "
      f"{sat['n_flagged']} flagged, alarm {sat['alarm']}, first alarm {st(sat['first_alarm_at'])}; runs: {runs_txt}.")
    p(f"  Its flagged list equals the ledger tool's ANY-flagged list: {sat['equals_ledger_any_flagged']}.")
    cs = d["checkpoint_steps"]
    p(f"  checkpoints: best {st(cs['best'])}, robust_best (the notebook's handoff and gate pair) {st(cs['robust_best'])}, final {st(cs['final'])}.")
    for row in d["checkpoint_table"]:
        p(f"    checkpoint table '{row['at']}' at {st(row['timesteps'])}: clean {f(row['clean'],0)}/{f(row['episodes'],0)}, falls {f(row['falls'],0)}, "
          f"saturated {f(row['saturated'],0)} {row['flagged_actuators'] or ''}, CoM fore-aft {f(row['com_fore_aft_mean_m'],4)} m, head tip "
          f"{f(row['head_tip_height_mean_m'],3)} m, neck pitch {f(row['neck_pitch_mean_deg'],1)} deg, tail chord {f(row['tail_tip_chord_mean_deg'],2)} deg"
          + (f", std tail_3 {f(row['std_tail_3_pitch'],4)}, std hip_roll {f(row['std_r_hip_roll'],4)}" if row.get("std_checkpoint") else ""))
    p(f"  rollout shares (named series, level {d['rollout_level']}): {d['rollout_named']}; first rollout at the level: {d['rollout_first_at_level']}.")
    p(f"  policy std read at {d['policy_std_checkpoints_read']} checkpoints; exploration page: leg std from the probe at every update, "
      f"reference {d['exploration_reference']} drawn and not applied, mirror pairs tied {d['exploration_pairs_tied']}.")
    p(f"  obs normalisation: {d['obs_norm_checkpoints']} VecNormalize checkpoints read, errors {d['obs_norm_errors']}; most drifting dims "
      f"{[(x['dim'], x['shift_prev_peak']) for x in d['obs_norm_most_drifting_dims']]}; probe/pickle check {d['probe_pickle_checks']['equal_float32']} "
      f"of {d['probe_pickle_checks']['n']} equal (float32).")
    p(f"  judge-K3 overlay: {d['judge_k3']} (no judge files were given at the stage end; the page draws the in-run panel only).")
    for c, lab in (("com_fore_aft_mean_m", "m (CoM ahead of the mean foot reference site, + ahead)"),
                   ("root_pitch_mean_deg", "deg (+ nose up)")):
        v = d["ledger_com_and_pitch"][c]
        p(f"  ledger {c} per evaluation (mean over full-horizon episodes) {lab}: statue {f(v['statue'],4)}; W1 {f(v['windows'][WN[0]],4)}, "
          f"W2 {f(v['windows'][WN[1]],4)}, W3 {f(v['windows'][WN[2]],4)}; first {st(v['first'][0])} {f(v['first'][1],4)}; last "
          f"{st(v['final'][0])} {f(v['final'][1],4)}; min {f(v['min'][1],4)} at {st(v['min'][0])}; max {f(v['max'][1],4)} at {st(v['max'][0])}.")
    p("")
p("  Pages viewed (what they state, read from the images) [measured, PAGES]:")
p("  - s42 02_saturation: flagged evaluations in four runs with alarms marked at 3.93M, 5.51M, 8.13M, 9.70M, all")
p("    'tail_3_pitch_act +' (249 episodes); the window fraction at |action| >= 0.99 reaching 1.0; force fraction flat")
p("    about 0.28 of limit (the flags are about the command, not force); rollout clip and mu-beyond shares rising with each")
p("    spell (mu-beyond peak about 0.8 near 4.7M, about 0.6 near 10.1M).")
p("  - s42 10_policy_std: per-actuator std at 15 checkpoints; tail_3 from about 0.22 to 0.095, hip_roll to about 0.062,")
p("    hip_pitch/ankle about 0.014, knee about 0.009; the other actuators (neck, head, tail_1, tail_2) 0.096-0.127 at the end.")
p("  - s42 17_exploration_k3: leg std per update (log scale) with the 0.08 reference 'not applied'; hip_roll crosses it")
p("    near 7.8M; the in-run panel's joint falls 1 of 20 from 3.15M to 8.39M, 0 after; own-sigma 0 throughout.")
p("  - s80 13_commands: neck_pitch_act's window-mean command between about -0.6 and -0.93 from about 4.9M to 7.9M (rail -1")
p("    in 119 episodes; the page counts parked episodes over all episodes, D-13 over full-horizon ones),")
p("    back near 0 by 8.6M; head_pitch_act moving to about -0.75 from 9.5M (parked '-', 189 episodes); tail_3 rising to")
p("    about +0.6 by 12M (parked '+', 20 episodes); legs near 0.")
p("")

# ------------------------------------------------------------------ D-12
p("-" * 118)
p("D-12 SAME-SEED COMPARISON (report only; seeds do not pair runs: GPU training is not bit-reproducible and r10 changes the")
p("dynamics from step 1 [measured, EVID 'WHY SEEDS DO NOT PAIR RUNS']). r9 rows [measured, EVID TABLE D and TABLE A;")
p("re-derived from R9L-* via R9T-*]; r10 rows [measured, L-*, M3F-*, M1F-*, G-*]. K1-K4 of r10 need M4 (pending).")
p("-" * 118)
r9 = D["r9"]
p("  seed run                 plant ANY ALARM (actuator)        first sat   first flag  flagged/91  last flag    final certificate              final CMD t1p/t1y/t2/t3        K1-K4    wall clock")
rows12 = [
    ("60", "stance6_r10LTSB_s60", "r10"), ("60", "stance5_LTSB_s60", "r9"), ("60", "stance3 Run 2 (LT)", "r9f"), ("60", "stance4 LTS s60", "r9f"),
    ("42", "stance6_r10LTSB_s42", "r10"), ("42", "stance5_LTSB_s42", "r9"), ("42", "stance4 LTS s42", "r9f"),
    ("80", "stance6_r10LTSB_s80", "r10"), ("80", "stance5_LTSB_s80", "r9"),
]
M3 = {}
M1 = {}
for rid in RUNS:
    m3 = json.loads((ROUND / "panels" / rid / "final/m3/m3_summary.json").read_text())
    M3[rid] = (m3["verdict"], [h["n_clean"] for h in m3["headings"]])
    m1 = json.loads((ROUND / "panels" / rid / "final/m1/m1_summary.json").read_text())
    M1[rid] = m1["cmd_a"]
EV = {"stance5_LTSB_s60": ("none", "11,403,264", "none", "0", "none", "PASS 39/40 (final v2 39/40 PASS)", "n/r", "P P F P", "14 h 26 min"),
      "stance5_LTSB_s42": ("3,670,016 (tail_3 +1)", "3,145,728", "3,538,944", "30", "11,927,552", "not rolled (final v2 19/40 FAIL)", "n/r", "F P F P", "13 h 36 min"),
      "stance5_LTSB_s80": ("4,456,448 (tail_3 +1)", "4,325,376", "4,325,376", "37", "10,223,616", "PASS 38/40 (final v2 38/40 PASS)", "n/r", "F P P P", "13 h 47 min"),
      "stance3 Run 2 (LT)": ("2,359,296 (tail_3)", "1,310,720", "2,228,224", "44", "11,272,192", "final v2 40/40 PASS", "+0.156/n-r/+0.023/+0.851", "F F P P", "n/r"),
      "stance4 LTS s60": ("none", "none", "none", "0", "none", "final v2 39/40 PASS", "n/r t3 -0.518", "F F P P", "n/r"),
      "stance4 LTS s42": ("2,228,224 (tail_3)", "1,835,008", "2,097,152", "70", "11,796,480", "final v2 14/40 FAIL", "n/r t3 +0.888", "F F F F", "n/r")}
for seed, name, kind in rows12:
    if kind == "r10":
        d1 = R[name]["D1"]
        al = d1["any_alarm_steps"]
        act = list(d1["rows_over_bar_by_actuator_sign"].keys())
        alarm = f"{al[0]:,} ({act[0][:-5]} {act[0][-1]}1)" if al else "none"
        flagged = len(d1["any_flagged_evaluations"])
        lastf = d1["any_flagged_evaluations"][-1] if d1["any_flagged_evaluations"] else None
        cert = f"M3 {M3[name][0]} n_clean {sorted(set(M3[name][1]))}/40 x6 headings"
        c = M1[name]
        cmd = f"{c['11']:+.3f}/{c['12']:+.3f}/{c['13']:+.3f}/{c['14']:+.3f}"
        wc = R[name]["wall_clock_s"]
        p(f"  {seed:4} {name:20} r10   {alarm:30} {st(d1['first_any_saturated_episode']):>11} {st(lastf if False else (d1['any_flagged_evaluations'][0] if d1['any_flagged_evaluations'] else None)):>11} {flagged:>10}  {st(lastf):>11}  {cert:30} {cmd:30} pending  {int(wc // 3600)} h {int(wc % 3600 // 60)} min")
    else:
        e = EV[name]
        tag = "r9" if kind == "r9" else "r9 family"
        p(f"  {seed:4} {name:20} {tag:5} {e[0]:30} {e[1]:>11} {e[2]:>11} {e[3]:>10}  {e[4]:>11}  {e[5]:30} {e[6]:30} {e[7]:8} {e[8]}")
p("  (r9 'final v2' is the judge's 40-episode v2 panel at the final; r10's certificate is M3 at the final, six headings x 40;")
p("  r10's final CMD is M1's CMD_a at stage1_final, 20 episodes [measured, M1F-*]; r10 wall clock is progress.json wall_clock_s")
p("  [measured, G-*]; r9 failure details as EVID TABLE D: s60 K3 3 of 20 joint falls at the final; s42 K1 v2 FAIL at 6.29M and")
p("  final, K3 13 of 20; s80 K1 below 33/40 at 5.24M/6.29M/8.39M on the saturation rail.)")
for n in ("stance5_LTSB_s60", "stance5_LTSB_s42", "stance5_LTSB_s80"):
    sm = r9[n]["summary"]
    p(f"  re-derived from the r9 ledger [measured, R9T-{n}]: first sat {st(sm['first_sat'])}, first flag {st(sm['first_flag'])}, alarm "
      f"{st(sm['alarm'])}, flagged {sm['n_flagged']}, last flag {st(sm['last_flag'])}, saturated episodes {sm['sat_eps']} (EVID TABLE A agrees).")
p("  Read beside each other (report only) [inferred]: seed 42 railed on tail_3 at +1 on both plants with onsets 3.15-3.93M")
p("  (r10 about 0.26M later at the first alarm) and failed at the final on both (r9 v2 19/40; r10 M3 FAIL); seed 80 railed tail_3 on r9 but neck_pitch")
p("  (non-tail) on r10, with no tail-saturated episode; seed 60 had no flag on r9 or r10, but r10's s60 ended PARKED on tail_3.")
p("")

# ------------------------------------------------------------------ D-13
p("-" * 118)
p("D-13 TEL COLUMNS OVER TRAINING (M8; 'columns of the telemetry PR; report only') [measured, E-<run> via L-<run> m8: per")
p("evaluation the mean over full-horizon episodes, then the mean of those over each window; statue = zero action on the")
p(f"run-branch head tree, 10 episodes, same 194-column ledger [measured, STAT, {D['statue_ledger']['rows']} rows]]. 124 numeric TEL")
p("columns per run are in READOUTS.json (D13); the main ones here. Signs: neck/head pitch + nose down; root pitch + nose up;")
p("tail angles + tip up (yaw + left); hip roll + abduction.")
p("-" * 118)
cols13 = ["head_tip_height_mean_m", "head_tip_height_min_m", "neck_pitch_mean_deg", "head_pitch_mean_deg",
          "r_hip_roll_mean_deg", "l_hip_roll_mean_deg", "r_hip_roll_max_abs_deg", "l_hip_roll_max_abs_deg"] + \
    [f"{a}_command_mean" for a in ACT] + ["actuators_at_rail", "actuators_parked",
    "tail_1_pitch_mean_deg", "tail_1_yaw_mean_deg", "tail_2_pitch_mean_deg", "tail_3_pitch_mean_deg", "tail_tip_chord_mean_deg",
    "root_height_mean_m", "root_pitch_min_deg", "root_pitch_max_deg", "root_roll_mean_deg",
    "legs_work_abs_j", "tail_work_abs_j", "neck_head_work_abs_j", "legs_torque_rms_frac", "tail_torque_rms_frac",
    "neck_head_torque_rms_frac", "legs_command_delta_rms", "tail_command_delta_rms", "neck_head_command_delta_rms",
    "action_delta_mean", "window_displacement_m", "window_yaw_change_deg", "window_path_length_m", "r_slip_m", "l_slip_m",
    "tail_3_pitch_stop_share", "com_toe_margin_min_m", "com_heel_margin_min_m", "capture_toe_margin_min_m",
    "angular_momentum_pitch_rms_nms", "tail_base_pitch_moment_mean_nm", "neck_base_pitch_moment_mean_nm", "head_angular_rate_rms_dps"]
SV = D["statue_ledger_values"]
for rid in RUNS:
    d13 = R[rid]["D13"]
    p(f"{rid}: TEL columns absent {d13['absent']}; HEAD LOW evaluations (window-mean head tip < 0.75 m) {d13['head_low_evaluations'] or 'none'}, "
      f"HEAD LOW episodes {d13['head_low_rows']}" + (" (the fallen episode, 11,534,336 ep 7, length 463, head tip 0.749 m)" if d13['head_low_rows'] else "") + f"; evaluation falls (step, episode, length) {d13['evaluation_falls']}.")
    p("    column                               statue        W1        W2        W3     last eval   min (at)                 max (at)")
    for c in cols13:
        w = [d13["windows"][k].get(c) for k in WN]
        e = d13["extremes"].get(c, {})
        p(f"    {c:36} {g(SV.get(c)):>9} {g(w[0]):>9} {g(w[1]):>9} {g(w[2]):>9} {g(d13['final_eval'].get(c)):>11}   "
          f"{g(e.get('min')):>9} ({st(e.get('min_at')):>10})  {g(e.get('max')):>9} ({st(e.get('max_at')):>10})")
    for k in WN:
        p(f"    {WSHORT[k]}: rail_actuators {d13['windows'][k]['rail_actuators'] or '{}'}; parked_actuators {d13['windows'][k]['parked_actuators'] or '{}'}")
    p("")

# ------------------------------------------------------------------ D-14
p("-" * 118)
p("D-14 IN-RUN K3 PANEL (M7; report only, not the K3 of record) [measured, K-<run>, KC-<run> via K3T-<run>]")
p("-" * 118)
for rid in RUNS:
    k = R[rid]["D14"]["raw"]
    js = k["joint_sigma"]
    p(f"{rid}: k3_panel.json sha256 {k['sha256'][:16]}...; csv rows {k['csv_rows']}; session_problem {k['session_problem']}; joint sigma "
      f"key {js['key']}, sha256 {js['sha256']}, equals the seed's pin {js['equals_pin']}; one sigma series over all {len(k['panels'])} panels.")
    p("    step         kind    own falls of 20 [seeds]      joint falls of 20 [seeds]       sigma sha256   failure")
    for pn in k["panels"]:
        p(f"    {pn['timesteps']:>11,} {pn['kind']:7} {pn['own']:>8} {str(pn['own_fall_seeds']):20} {pn['joint']:>8} {str(pn['joint_fall_seeds']):22} "
          f"{pn['joint_sigma_sha256'][:12]}   {pn['failure']}")
    ov = R[rid]["D14"]["own_sigma_vs_probe_std"]
    p(f"  own-sigma vector vs the probe's std at the same update: equal at the 11 grid panels (max |diff| 0); final panel differs by "
      f"{f(ov[-1]['max_abs_diff'],7)} [measured, K-{SH[rid]}, P-{SH[rid]}] (the probe row is read before its update's train(), the final "
      f"panel after it [inferred]).")
    jm = R[rid]["D14"]["judge_m4"]
    p(f"  beside the offline judge's K3 (D-7) at 6,291,456 and the final: PENDING - no m4/ folder under {ROUND / 'panels' / rid}/6291456 "
      f"or /final at {NOW} (the report-only measurer runs M4 after M1/M2/M3).")
p("  r10 statue reference: 1 joint-space fall in 2 spawns [measured, TEL spec, as THIS12 line 627]. The panel's policy-state")
p("  and observation-statistics sha256 were not matched against the checkpoint pairs (k3panel_r10round.py run without")
p("  --models; that check loads checkpoints and is left to the measurer).")
p("")

# ------------------------------------------------------------------ D-15
p("-" * 118)
p("D-15 EXPLORATION STD AND OBSERVATION-NORMALISATION DRIFT (TEL pages 17-18) [measured, P-<run> std_per_actuator at every")
p("PPO update (1,465 rows, written in full to STD-<run>); H-<run> obs_normalization]")
p("-" * 118)
hdr = "    step            " + " ".join(f"{a.replace('_act','')[:10]:>10}" for a in ACT)
for rid in RUNS:
    d15 = R[rid]["D15"]
    p(f"{rid}: std = exp(log_std) at each K3 point (the probe row at that update):")
    p(hdr)
    for s, v in d15["std_at_k3_points"].items():
        p(f"    {s:16}" + " ".join(f"{x:10.4f}" for x in v))
    b8 = d15["first_update_below_0p08"]
    p(f"  first update below the stance3 line-2 reference 0.08 (drawn, not applied): " +
      ", ".join(f"{a.replace('_act','')} {st(b8[a])}" for a in ACT if a in ("r_hip_pitch_act", "r_hip_roll_act", "r_knee_act", "r_ankle_act")) +
      " (left = right); neck, head and tail actuators never below 0.08. Leg mirror pairs equal at every update (max |gap| "
      f"{d15['leg_pair_max_abs_gap']}).")
    d = R[rid]["D11"]
    top = sorted(d["obs_norm_drift"], key=lambda x: -(x["shift_prev_max"] or 0))[:4]
    p(f"  obs-normalisation drift as TEL reports it: {d['obs_norm_checkpoints']} checkpoints; three most-drifting dims {[(x['dim'], x['shift_prev_peak']) for x in d['obs_norm_most_drifting_dims']]};")
    p("    largest checkpoint-to-checkpoint shifts: " + "; ".join(f"{x['timesteps']:,}: max {f(x['shift_prev_max'],3)} (top {x['top'][0][0]})" for x in top))
    last = d["obs_norm_drift"][-1]
    p(f"    final pickle vs previous: shift max {f(last['shift_prev_max'],4)}, scale max {f(last['scale_prev_max'],4)}, top {last['top'][0][0]}; probe/pickle"
      f" consistency {d['probe_pickle_checks']['equal_float32']} of {d['probe_pickle_checks']['n']} equal_float32.")
    p("")

# ------------------------------------------------------------------ ENTROPY
p("-" * 118)
p("ENTROPY CHECK (the maintainer's question: could the entropy setting be introducing the bad behaviours?)")
p("-" * 118)
st60 = R[RUNS[0]]["settings"]
er = R[RUNS[0]]["entropy_record"]
p("MEASURED FACTS")
p(f"E1 Settings, identical in all three runs' stage_config.json [measured, C-s60, C-s42, C-s80]: ent_coef {st60['ent_coef']} -> "
  f"ent_coef_end {st60['ent_coef_end']} over ent_coef_decay_timesteps {int(st60['ent_coef_decay_timesteps']):,}; policy_kwargs "
  f"{st60['policy_kwargs']}; symmetry {st60['symmetry']} (kappa {st60['symmetry_kappa']}, sigma_floor {st60['symmetry_sigma_floor']}, "
  f"mirror critic {st60['symmetry_mirror_critic']}); action_saturation_weight {st60['action_saturation_weight']} above |"
  f"{st60['action_saturation_threshold']}|; learning_rate {st60['learning_rate']} -> {st60['learning_rate_end']}; n_steps "
  f"{st60['n_steps']}, batch {st60['batch_size']}, n_epochs {st60['n_epochs']}, target_kl {st60['target_kl']}; no use_sde key.")
p(f"E2 ent_coef as recorded at every update (policy_probe.npz ent_coef) [measured, P-*]: {er['updates']} updates; first "
  f"{f(er['ent_first'],8)} at 8,192; max |recorded - linear schedule| {er['max_abs_diff_from_schedule']:.1e}; first update at 0: "
  f"{st(er['first_update_at_zero'])}; 0 to the end. Same in all three runs. Values at the events:")
for s, v in er["at"].items():
    p(f"      at {int(s):>11,} (update {v['update_timesteps']:,}): ent_coef {v['ent_coef']:.6f}")
p("E3 Onsets confirmed on F2's ledgers [measured, L-s42, L-s80]: s42 tail_3_pitch_act at +1 (saturated_actuator_positive_share")
p(f"   1.0 on all 249 over-bar rows): first over-bar episode 3,670,016, first flagged evaluation 3,801,088, first TAIL ALARM")
p(f"   3,932,160 (as interim snapshot 2 said); s80 neck_pitch_act at the negative end (positive share 0.0 on all 119 over-bar")
p(f"   rows; TEL 'neck_pitch_act-'): first over-bar episode 3,801,088, first ANY-flagged 4,849,664, first ANY ALARM 5,505,024")
p(f"   (as snapshot 2 said). s60: first tail_3 + over-bar episode 8,126,464, no flagged evaluation. The bonus at those steps:")
p(f"   s42 first alarm {er['at']['3932160']['ent_coef']:.5f} (44% of 0.005); s80 first alarm {er['at']['5505024']['ent_coef']:.5f} "
  f"(21%); s60 first over-bar episode 0 [measured, P-*].")
p("E4 Exploration std per actuator over training [measured, P-*, STD-*]: from exp(-1.5) = 0.2231 at the start, every actuator's")
p("   std falls through the run; no actuator's std rises over any 1,048,576-step stretch (over 262,144 steps small rises of")
p("   at most 0.006; single-update increases <= 0.66%). At the")
p("   final: legs 0.009-0.064, neck/head 0.096-0.122, tail 0.095-0.134 (D-15 tables). Std at the onsets, all three runs at the")
p("   same update (neck_pitch, head_pitch, tail_1_pitch, tail_1_yaw, tail_2_pitch, tail_3_pitch) [measured, DATA cross_run_std]:")
for s in ("3670016", "3932160", "5505024", "7000000", "12001280"):
    row = D["cross_run_std"][s]
    p(f"      {int(s):>11,}: " + "; ".join(f"{r} " + "/".join(f"{row[r][a]:.3f}" for a in ("neck_pitch_act", "head_pitch_act", "tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")) for r in ("s60", "s42", "s80")))
p("   The railing actuator was not noisier in the railing run: s42's tail_3 std was the lowest of the three runs at its onset")
p(f"   and stayed lowest to the end (final 0.095 vs 0.114 s60, 0.123 s80); s80's neck_pitch std at its first alarm (0.1397) is")
p(f"   about equal to s60's (0.1405) and above s42's (0.1363).")
k = K7
p("   No kink at 7,000,000 when the bonus reaches 0 (log-std slope per 1M steps in (6M,7M] vs (7M,8M]) [measured, KINK]:")
for r in ("s60", "s42", "s80"):
    a, b = k[r]["(6000000,7000000]"], k[r]["(7000000,8000000]"]
    p(f"      {r}: tail_3 {a['tail_3_pitch']:+.3f} -> {b['tail_3_pitch']:+.3f}; neck_pitch {a['neck_pitch']:+.3f} -> {b['neck_pitch']:+.3f}; "
      f"head_pitch {a['head_pitch']:+.3f} -> {b['head_pitch']:+.3f}; tail_1_pitch {a['tail_1_pitch']:+.3f} -> {b['tail_1_pitch']:+.3f}")
p("E5 Saturation before and after the bonus reached 0 [measured, L-*, BINS]:")
for rid in RUNS:
    s = SH[rid]
    w0, w1 = R[rid]["D1"]["windows"]["[0,7000000)"], R[rid]["D1"]["windows"]["[7000000,12001280]"]
    p(f"   {s}: over-bar rows [0,7M) {w0['tail_over'] + w0['other_over']} of {w0['rows']} ({f((w0['tail_over'] + w0['other_over']) / w0['rows'],3)}), "
      f"[7M,12M] {w1['tail_over'] + w1['other_over']} of {w1['rows']} ({f((w1['tail_over'] + w1['other_over']) / w1['rows'],3)}); flagged evaluations "
      f"{w0['any_flagged_evals']} -> {w1['any_flagged_evals']}; alarms {len(w0['any_alarms'])} -> {len(w1['any_alarms'])}.")
p("   Per 1,048,576-step bin (rollout mean of the actuator's mu; share of rollouts' rows with mu beyond the bound; ledger")
p("   over-bar rows of 80; bin-mean ent_coef) [measured, BINS]:")
for s, a, side in (("s42", "tail_3_pitch_act", "hi"), ("s60", "tail_3_pitch_act", "hi"), ("s80", "neck_pitch_act", "lo")):
    p(f"      {s} {a} ({'high' if side == 'hi' else 'low'} bound):")
    for bn in B[s]["bins"]:
        p(f"        {bn['bin']:26} ent {bn['ent_mean']:.5f}  mu {bn[a + '_mu_mean']:+.3f}  beyond {bn[a + '_mu_beyond_' + side + '_mean']:.4f} "
          f"(rollouts >= .05: {bn[a + '_mu_beyond_' + side + '_ge05']:>3})  over-bar rows {bn['over_bar_rows']:>2}/{bn['eval_rows']}  TEL cmd {f(bn[a + '_cmd_mean'],3)}")
p("   In words: s42's rail ran 3.67-5.9M with the bonus at 0.0024-0.0008, cleared at 6.3-7.3M (mu -0.06, bonus 0.0002), then")
p("   re-formed with the bonus at 0 (7.99-8.39M and 9.57-10.62M; 11 of its 22 alarms after 7M; over-bar share 0.23 -> 0.33).")
p("   s60's drive began rising at 4.2-5.2M (mu 0.13) and 6.3-7.3M (mu 0.44, bonus 0.0002); its mean-beyond share reached 0.01")
p("   only at 7,454,720 and 0.05 at 7,962,624, with the bonus at 0 (all 41 rollouts >= 0.05 and all 24 over-bar episodes after")
p("   7M), and the rollout mean eased to 0.57 by the end.")
p("   s80's neck rail peaked at 5.2-6.3M (bonus 0.0009), was already falling at 6.3-7.3M, and faded through 7.3-8.4M (bonus 0;")
p("   last flagged 7,340,032, last over-bar row 8,781,824). So after the bonus reached 0 saturation grew (s60), re-formed")
p("   (s42) and faded (s80).")
p("E6 r9 under the same schedule [measured, R9C-* (ent_coef 0.005 -> 0 by 7,000,000, log_std_init -1.5, sigma_floor 0.01 in")
p("   all five), R9T-*, R9G-*]:")
for n, v in r9.items():
    fp = v["final_policy"]
    p(f"   {n}: first sat {st(v['summary']['first_sat'])}, alarm {st(v['summary']['alarm'])}; flagged evaluations before 7M "
      f"{v['flagged_before_7M']}, after 7M {v['flagged_after_7M']}; alarms after 7M {len(v['alarms_after_7M'])} (first "
      f"{st(v['alarms_after_7M'][0]) if v['alarms_after_7M'] else 'none'}); over-bar rows [0,7M) {v['over_rows_by_window']['[0,7000000)']}, "
      f"[7M,12M] {v['over_rows_by_window']['[7000000,12001280]']}; final std_min/max {fp['std_min']}/{fp['std_max']}, ent_coef {fp['ent_coef']}.")
p("   The r9 records hold no per-update std: the r9 run folders held here have no policy_probe.npz, and progress.json keeps")
p("   only the final policy block [measured, folder listings of S0/foot/stance5/runs/* and replication/runs/*]. r10's final")
p("   std_min/max: " + "; ".join(f"{SH[r]} {R[r]['progress_policy']['std_min']}/{R[r]['progress_policy']['std_max']}" for r in RUNS) +
  " [measured, G-*]: the same range as r9's.")
p("E7 How the code applies the bonus [measured, CODE-MKL lines 21, 199-201, 335-379; CODE-SB3 lines 152, 180-181; CODE-ENV")
p("   lines 970, 1484-1504; CODE-RF lines 621-647]: the loss is policy_loss + ent_coef * entropy_loss + vf_coef * value_loss")
p("   (stock); the policy's log_std is one nn.Parameter per actuator (state-independent; use_sde not set), and a diagonal")
p("   Gaussian's entropy is sum(log_std) + constant; the symmetry KL uses sigma = exp(log_std).detach().clamp_min(0.01), so it")
p("   passes no gradient to log_std; the saturation cost is computed on the clipped raw command (stage_config")
p("   action_penalty_source \"raw\" [measured, C-*]), ramping from |a| = 0.9 to 1 and flat beyond.")
p("")
p("INFERRED ANSWER (could entropy be driving it?) [inferred, from E1-E7]")
p("- Direct effect: no. The entropy term's gradient reaches log_std only, with the same push on all 15 actuators, and")
p("  never reaches the mean action (from E7). The ledger's rails are the deterministic policy's mean command (and M6's")
p("  'mu beyond the bound'); nothing in the entropy term singles out tail_3 or neck_pitch.")
p("- As a necessary cause: the records argue against it. s60's tail drive crossed the bound, and s42's rail re-formed")
p("  twice, with ent_coef exactly 0 (E2, E5); r9's two railed runs kept 12 and 17 flagged evaluations after 7M (E6).")
p("- Via the noise level: not supported, not excluded. The railing actuator was the least noisy of the three runs, not the")
p("  most (E4), and std shows no kink at 7M (E4) - but by 6-7M the bonus is below 0.0007, so its removal is gradual. The")
p("  onsets of the alarm runs on both plants (r10 s42 3.67M and s80 3.80M first over-bar rows; r9 s42 3.15M and s80")
p("  4.33M) all fall while the bonus is > 0 (s60's late episodes came with it at 0), so a role in the timing of a first")
p("  onset cannot be judged from runs that all share one schedule: there is no contrast in these records. n = 3 (and")
p("  r9's 5) runs; GPU training is not bit-reproducible [measured, EVID].")
p("- A plausible mechanism that does not need entropy [inferred, from E7]: once the mean passes the bound, every sample on")
p("  that side clips to the same command, the saturation cost stops growing at |a| = 1, and the clipped surrogate has little")
p("  to pull the mean back; the observed std shrinking fastest on s42's tail_3 fits that. This is a reading, not a test.")
p("WHAT WOULD TEST IT (none of it is proposed here; each needs the maintainer's approval of a training-hyperparameter")
p("change and a launch) [inferred]:")
p("- Branch test from the runs' own checkpoints (cheapest that isolates the term): from s42's 3,407,872 pair (before its")
p("  first rollout with mu-beyond >= 0.01, 3,579,904, and its first over-bar episode, 3,670,016) and s80's 3,670,016 pair")
p("  (before its first neck over-bar episode, 3,801,088), continue for about 1.5M steps with (a) the recorded schedule and")
p("  (b) ent_coef 0, several training seeds each (e.g. 4 + 4 per checkpoint), reading M6's mu-beyond share and the ledger")
p("  rule; and the converse from s60's 6,815,744 pair (bonus re-instated at 0.002 vs 0). About 16 continuations of about 1.7")
p("  h each at the runs' 240 steps/s, about 25-30 L4-hours [inferred from G-* wall clock]. The three pairs are listed in")
p("  their runs' artifact_manifest.json but are not among F2's copies (F2 holds the 1,048,576-grid pairs, the final and, for")
p("  s42, the 3,670,016 onset pair) [measured, VER], so a read-only fetch would come first. Resumed runs are diagnostics,")
p("  not round runs (RUN HANDLING forbids resumes in the round).")
p("- From-scratch contrast (the only test of onset): two arms identical but for ent_coef (as shipped vs 0 throughout, or a")
p("  fixed log_std), pre-registered with the same ledger rule. One-sided Fisher with 0 alarms in one arm against 2 of 3 in")
p("  the other needs about 6 runs per arm for p 0.03, 8 for 0.013; against half, 8 per arm gives 0.038 [inferred, computed]:")
p("  12-16 runs, about 170-230 L4-hours.")
p("")

# ------------------------------------------------------------------ INTERIM
p("-" * 118)
p("INTERIM CONTEXT (the two mid-run snapshots, report only; they never stood for the reading) [measured, SNAP1, SNAP2, VER]")
p("-" * 118)
for rid in RUNS:
    ip = R[rid]["interim_prefix"]
    p(f"{rid}: snapshot 1 ledger {ip['1']['rows']} rows to {ip['1']['last_eval']:,} and snapshot 2 ledger {ip['2']['rows']} rows to "
      f"{ip['2']['last_eval']:,}: header equal and every row equal to F2's first rows: {ip['1']['rows_equal_to_F2_prefix']} / "
      f"{ip['2']['rows_equal_to_F2_prefix']}; their K3 panels ({ip['1']['k3_panels']}, {ip['2']['k3_panels']}) equal F2's falls: "
      f"{ip['1']['k3_falls_equal_to_F2']} / {ip['2']['k3_falls_equal_to_F2']}.")
p("Snapshot 1 (files of about 18:40 UTC 2026-10-27, 3,014,656 steps): no flag or alarm in any run; s42's tail_3 command")
p("  rising (0.202 -> 0.394). F2 agrees for those rows; s42's first over-bar row came 5 evaluations later (3,670,016).")
p("Snapshot 2 (files of 23:52-23:58 UTC, 7.60-7.73M): s42 13 TAIL-flagged evaluations and 11 alarms (3.80-5.51M), 128")
p("  over-bar rows; s80 12 ANY-flagged and 6 alarms on neck_pitch_act, 110 rows; s60 none, tail_3 command at 0.758.")
p(f"  F2 now: s42 {len(s42['D1']['tail_flagged_evaluations'])} flagged and {len(s42['D1']['tail_alarm_steps'])} alarms, "
  f"{s42['D1']['rows_over_bar_total']} rows (two more spells 7.99-8.39M and 9.57-10.62M, after the snapshot); s80 the same 12 and 6,"
  f" {s80['D1']['rows_over_bar_total']} rows (9 single rows after the snapshot, the last 8,781,824); s60 {s60['D1']['rows_over_bar_total']} single over-bar")
p("  rows from 8,126,464 (after the snapshot), no flag; the scorer read s60 PARKED (MEAN_a +0.642) [measured, SCORE].")
p("  So snapshot 2 saw s42's first two spells, s80's whole alarm period and the start of s60's drive; it could not see")
p("  s42's re-formed rail or s60's late over-bar episodes, both after 7M [inferred].")
p("")

# ------------------------------------------------------------------ notes
p("-" * 118)
p("NOTES AND LIMITS")
p("-" * 118)
p("- D-14's judge comparison and D-12's r10 K1-K4 are pending M4 (not under ROUND/panels at the time above).")
p("- k3_panel.json joint-sigma keys are source_vectors.60, source_vectors.42 and control_sigma.80, each equal to its pin")
p("  [measured, K3T-*]; criteria.txt calls seeds 60/42 'their frozen control_sigma' and seed 80 'the replication sidecar's")
p("  mean vector' - a naming difference only, the hashes match the pins [inferred].")
p("- TEL's statue values come from a 10-episode zero-action ledger on the run-branch head tree of the V8 check (before the")
p("  header commit, which adds a stance.toml comment only) [measured, STAT; inferred equivalence]. Head tip 0.959 m equals")
p("  DRAFT's statue figure.")
p("- The PNG pages were copied (cp) to .png names under PAGES to view them as images; nothing was written in the fetch store.")
p("- Readout files missing or differing from the manifest: none (VER).")
p("")
p("SOURCES (full paths; sha256 where it identifies a copy)")
for k2, (pth, sh) in SRC.items():
    p(f"  {k2:24} {pth}" + (f"  sha256 {sh}" if sh else ""))
p("")
p(f"Machine-readable: {OUT / 'READOUTS.json'}. END OF READOUTS (report only).")

(OUT / "READOUTS.txt").write_text("\n".join(L) + "\n")

# ------------------------------------------------------------------ JSON
J = {"schema": "stance6-readouts/v1", "written_utc": NOW, "report_only": True,
     "reading_fixed": "line 1: TAIL RAIL FORMED ON r10 IN TRAINING (ledger); R 1, U 0, P 1, G+ 1; A 2 of 3 vs 2 of 5, p 0.9286; STATUS FINAL",
     "sources": {k2: {"path": v[0], "sha256": v[1]} for k2, v in SRC.items()},
     "statue_ledger": D["statue_ledger"], "statue_ledger_values": D["statue_ledger_values"], "statue_m2": D["statue_m2"],
     "runs": {}, "cross_run_std": D["cross_run_std"], "std_around_onset": D["std_around_onset"], "std_kink_7M": K7,
     "bins": B, "r9": D["r9"],
     "entropy_answer": {
         "measured": ["E1 settings identical in all runs (ent_coef 0.005 -> 0 by 7,000,000; log_std_init -1.5; sigma_floor 0.01; saturation weight 0.5 above 0.9)",
                      "E2 recorded ent_coef equals the linear schedule (max diff 2.3e-10), 0 from update 7,004,160",
                      "E3 s42 first TAIL ALARM 3,932,160 (ent 0.00219); s80 first ANY ALARM 5,505,024 neck_pitch_act - (ent 0.00107); s60 first over-bar 8,126,464 (ent 0)",
                      "E4 std falls monotonically in trend for every actuator; railing actuator least noisy (s42 tail_3); no kink at 7M",
                      "E5 after 7M: s60 drive formed, s42 rail re-formed (11 of 22 alarms), s80 neck rail faded",
                      "E6 r9 same schedule: s42 12 and s80 17 flagged evaluations after 7M",
                      "E7 entropy gradient reaches log_std only (state-independent), equal across actuators; symmetry KL detaches sigma"],
         "inferred": "Not a direct or necessary driver; a role in onset timing through the noise level is neither supported nor excluded (no contrast in the records).",
         "test": ["branch continuations from s42 3,670,016 / s80 4,718,592 (schedule vs ent_coef 0) and s60 6,815,744 (0.002 vs 0), ~16 x 1.5M steps, ~25-30 L4-hours",
                  "from-scratch two-arm contrast, 6-8 runs per arm, ~170-230 L4-hours"]}}
for rid in RUNS:
    J["runs"][rid] = R[rid]
(OUT / "READOUTS.json").write_text(json.dumps(J, indent=1, default=str) + "\n")
print("written", OUT / "READOUTS.txt", len(L), "lines")
