"""PS statistic check: the MANIPULATION section of the score against run_meta / check.json / arm_evidence.jsonl.

Own code, reads only. Per continuation: check.json (passed, no failures, continuation, T = E, probe detail, sources'
sha256 now); run_meta (done, exit 0, stop "end" at E, the arm's classes, transform verified, "in force", phase_state_end,
control's digest_load = digest_start and probe not phase-fed); arm_evidence.jsonl re-aggregated with own code from the
175 per-rollout lines (the arm's signature on every rollout and no other's), compared with run_meta's arm_evidence
summary and with score.json's manipulation record; score.txt's MANIPULATION line checked to carry those values."""
import hashlib
import json
import sys
from pathlib import Path

PS = Path("/home/user/work/phase_screen")
S, E, ROLL = 2_097_152, 3_538_944, 8192
ACT = ("dmu_phase_mode", "dmu_phase_forward_actions", "dmu_phase_log_prob", "dmu_phase_entropy", "dmu_phase_mirrored")
CRIT = ("dv_phase_predict_values", "dv_phase_forward", "dv_phase_evaluate_actions", "dv_phase_mirrored")
CLASSES = {"control": ("MlpExtractor", "SymmetricRunningMeanStd"),
           "critic_phase": ("CriticPhaseMlpExtractor", "PhaseSlotsSymmetricRunningMeanStd")}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


score = json.loads((PS / "score/score.json").read_text())
score_txt = (PS / "score/score.txt").read_text().splitlines()
manip_lines = {}
in_m = False
for ln in score_txt:
    if ln.startswith("MANIPULATION"):
        in_m = True
        continue
    if in_m and not ln.startswith("  "):
        break
    if in_m:
        manip_lines[ln.split()[0]] = ln

problems = []
n_cmp = [0, 0]


def cmp(label, a, b):
    n_cmp[0] += 1
    if a != b:
        n_cmp[1] += 1
        problems.append(f"{label}: mine {a!r} vs theirs {b!r}")
        print(f"    [DIFF] {label}: mine {a!r} vs theirs {b!r}")


summary = {}
for arm, seed in [(a, s) for s in (2901, 2911, 2921, 2931) for a in ("control", "critic_phase")]:
    name = f"{arm}_{seed}"
    rd = PS / "runs" / name
    meta = json.loads((rd / "run_meta.json").read_text())
    chk = json.loads((rd / "check.json").read_text())
    ev = [json.loads(x) for x in (rd / "arm_evidence.jsonl").read_text().splitlines() if x.strip()]
    sm = meta["arm_evidence"]
    sc = score["report_only"][name]["manipulation"]
    bad = []
    # check.json
    if not (chk["passed"] is True and chk["failures"] == [] and chk["mode"] == "continuation" and chk["T"] == E
            and chk["E"] == E and chk["end"] == E and chk["stop_reason"] == "end" and chk["arm"] == arm
            and chk["seed"] == seed):
        bad.append("check.json identity / pass")
    pdt = chk["items"]["probe_detail"]
    if not (pdt["rows_expected"] == 175 and pdt["rows_in_window"] == 175 and pdt["n_rows_at_or_after_T"] == 0
            and pdt["seeded_rows_at_or_before_S"] == 256 and pdt["ent_coef_mismatches"] == 0
            and pdt["learning_rate_mismatches"] == 0 and pdt["obs_rms_rows_equal_start"] == 0):
        bad.append(f"probe detail {pdt}")
    if any(chk["items"][k] for k in ("run_meta", "evidence", "probe")):
        bad.append("check items not empty")
    for k, (p, h) in chk["sources"].items():
        if sha(p) != h:
            bad.append(f"source {k} changed since CHECK")
    # run_meta
    if not (meta["status"] == "done" and meta["exit"] == 0 and meta["stop_reason"] == "end" and meta["T"] == E
            and meta["num_timesteps_end"] == E and meta["S"] == S and meta["E"] == E and meta["arm"] == arm):
        bad.append("run_meta status / end")
    aa = meta["arm_applied"]
    if (aa.get("mlp_extractor_class"), aa.get("obs_rms_class")) != CLASSES[arm]:
        bad.append(f"classes {aa.get('mlp_extractor_class')} / {aa.get('obs_rms_class')}")
    if meta.get("arm_state_end") != "in force":
        bad.append(f"arm_state_end {meta.get('arm_state_end')}")
    if aa.get("gae_lambda") != 0.95 or aa.get("rollout_buffer_gae_lambda") != 0.95:
        bad.append("gae_lambda at load")
    if arm == "control":
        if meta.get("phase_state_end") is not None or meta.get("phase_definition") is not None:
            bad.append("control: phase record in run_meta")
        if meta["digest_load"] != meta["digest_start"]:
            bad.append("control: digest_load != digest_start")
        if meta["probe_phase"].get("installed") is not False:
            bad.append("control: probe phase-fed")
    else:
        if aa.get("transform_verified") is not True:
            bad.append("phase: transform not verified")
        pse = meta.get("phase_state_end") or {}
        if not (pse.get("policy_phase_cols_unchanged") is True and pse.get("policy_phase_adam_max_abs") == 0.0
                and pse.get("phase_stats_frozen") is True and pse.get("mirror_phase_self_map") is True):
            bad.append(f"phase_state_end {pse}")
        if meta["probe_phase"].get("installed") is not True or not all(k >= 200 for k in meta["probe_phase"]["k"]):
            bad.append("phase: probe not phase-fed with settled k")
        if meta["digest_load"]["params"] == meta["digest_start"]["params"] or \
                meta["digest_load"]["adam"] != meta["digest_start"]["adam"]:
            bad.append("phase: load transform (params changed, Adam not) not as stated")
    # per-rollout evidence, own aggregation
    steps = [e["num_timesteps"] for e in ev]
    if len(ev) != 175 or steps != [S + ROLL * k for k in range(1, 176)]:
        bad.append(f"evidence lines {len(ev)} / steps")
    per_bad = []
    for i, e in enumerate(ev):
        tag = f"rollout {i + 1}"
        if not (e["buffer_gae_lambda"] == 0.95 and e["model_gae_lambda"] == 0.95):
            per_bad.append(f"{tag} lambda")
        if not (e["adv_err_vs_gae095"] <= 1e-4 * max(1.0, e["adv_scale"]) and e["gae095_vs_mc"] > 1e-3):
            per_bad.append(f"{tag} GAE")
        if not e["returns_minus_adv_minus_values"] <= 1e-5:
            per_bad.append(f"{tag} returns")
        if not e["dv_predict_values"] > 0.0:
            per_bad.append(f"{tag} C4 obs6")
        if e["obs_rms_equal_start"] is not False or e["state_problems"]:
            per_bad.append(f"{tag} stats / state")
        if i and e["obs_rms_digest_no_phase"] == ev[i - 1]["obs_rms_digest_no_phase"]:
            per_bad.append(f"{tag} obs stats did not move")
        if arm == "control":
            if e.get("phase_rows") is not None or e.get("phase_buffer_max_err") is not None:
                per_bad.append(f"{tag} control phase evidence")
            if not max(e[k] for k in ACT) > 0.0:
                per_bad.append(f"{tag} control actor does not respond to the command slots")
        else:
            if any(e[k] != 0.0 for k in ACT):
                per_bad.append(f"{tag} C1 actor moved with the phase")
            if not (e["policy_phase_cols_unchanged"] is True and e["policy_phase_adam_max_abs"] == 0.0):
                per_bad.append(f"{tag} C1 actor phase columns / Adam")
            if i == 0:
                if not (e["n_updates_since_load"] == 0 and e["value_phase_cols_max_abs"] == 0.0
                        and all(e[k] == 0.0 for k in CRIT)):
                    per_bad.append(f"{tag} C2 zero-init")
            else:
                if not (e["n_updates_since_load"] >= 1 and e["value_phase_cols_max_abs"] > 0.0
                        and all(e[k] > 0.0 for k in CRIT)):
                    per_bad.append(f"{tag} C3 learned")
                if e["mirror_phase_tensors_ok"] is not True:
                    per_bad.append(f"{tag} mirror tensors")
            if not (e["phase_buffer_max_err"] == 0.0 and e["phase_k_start_consistent"] is True
                    and e["phase_k_end_consistent"] is True and e["phase_rows"] == 8192):
                per_bad.append(f"{tag} C5 recount")
            if not (e["phase_stats_frozen"] is True and e["mirror_phase_self_map"] is True
                    and e["envs_swapped"] is True):
                per_bad.append(f"{tag} C6 / C7")
    bad += per_bad[:10]
    # own aggregates compared with run_meta's summary and with score.json's manipulation record
    acts = [max(e[k] for k in ACT if isinstance(e.get(k), (int, float))) for e in ev]
    mine = {"n": len(ev), "buffer_gae_lambda": sorted({e["buffer_gae_lambda"] for e in ev}),
            "max_adv_err_vs_gae095": max(e["adv_err_vs_gae095"] for e in ev),
            "min_dmu_actor": min(e["dmu_actor"] for e in ev),
            "min_dv_predict_values": min(e["dv_predict_values"] for e in ev),
            "max_dv_predict_values": max(e["dv_predict_values"] for e in ev),
            "max_buffer_values_vs_critic": max(e["buffer_values_vs_critic"] for e in ev),
            "state_problems": [], "max_dmu_phase": max(acts), "min_dmu_phase": min(acts)}
    pe = [e for e in ev if e.get("phase_buffer_max_err") is not None]
    mine["phase_rows_recorded"] = len(pe)
    if pe:
        zero = [e for e in pe if e["n_updates_since_load"] == 0]
        upd = [e for e in pe if e["n_updates_since_load"] >= 1]
        mine.update({
            "phase_n_zero_init": len(zero), "phase_n_updated": len(upd),
            "zero_init_max_value_phase_cols": max(e["value_phase_cols_max_abs"] for e in zero),
            "zero_init_max_dv_phase": max(e[k] for e in zero for k in CRIT),
            "updated_min_value_phase_cols": min(e["value_phase_cols_max_abs"] for e in upd),
            "updated_min_dv_phase": {k: min(e[k] for e in upd) for k in CRIT},
            "max_phase_buffer_max_err": max(e["phase_buffer_max_err"] for e in pe),
            "phase_all_k_consistent": all(e["phase_k_start_consistent"] and e["phase_k_end_consistent"] for e in pe),
            "phase_all_phase_stats_frozen": all(e["phase_stats_frozen"] is True for e in pe),
            "phase_all_mirror_phase_self_map": all(e["mirror_phase_self_map"] is True for e in pe),
            "phase_all_policy_phase_cols_unchanged": all(e["policy_phase_cols_unchanged"] is True for e in pe),
            "phase_all_envs_swapped": all(e["envs_swapped"] is True for e in pe),
            "max_policy_phase_adam_max_abs": max(e["policy_phase_adam_max_abs"] for e in pe),
            "phase_rows_presettle_total": sum(e["phase_rows_presettle"] for e in pe),
            "phase_k_max": max(e["phase_k_max"] for e in pe)})
    for k, v in mine.items():
        cmp(f"{name} {k} vs run_meta", v, sm.get(k))
        cmp(f"{name} {k} vs score.json", v, sc.get(k))
    for k, v in (("mlp_extractor_class", aa.get("mlp_extractor_class")), ("obs_rms_class", aa.get("obs_rms_class")),
                 ("transform_verified", aa.get("transform_verified")), ("arm_state_end", meta.get("arm_state_end"))):
        cmp(f"{name} {k} vs score.json", v, sc.get(k))
    # the score.txt line carries these values
    ln = manip_lines.get(name, "")
    must = [f"{mine['n']} rollouts", f"min {mine['min_dv_predict_values']}", f"actor dmu min {mine['min_dmu_actor']}",
            f"|adv - GAE(0.95)| max {mine['max_adv_err_vs_gae095']}", f"{CLASSES[arm][0]} / {CLASSES[arm][1]}",
            "hook in force"]
    if arm == "control":
        must += ["no phase evidence (0 rollouts with phase rows)", f"(min {mine['min_dmu_phase']})"]
    else:
        must += [f"max {mine['max_dmu_phase']}", "before / after the first update 1 / 174",
                 f"critic phase response before {mine['zero_init_max_dv_phase']}",
                 str(mine["updated_min_dv_phase"]), f"columns after (min) {mine['updated_min_value_phase_cols']}",
                 f"max error {mine['max_phase_buffer_max_err']}", f"rows {mine['phase_rows_presettle_total']}",
                 f"k max {mine['phase_k_max']}", "transform verified True", "counters consistent True",
                 "slots frozen True", "mirror self-map True", "actor phase columns unchanged True (Adam 0.0)",
                 "envs swapped True"]
    miss = [m for m in must if m not in ln]
    n_cmp[0] += 1
    if miss:
        n_cmp[1] += 1
        problems.append(f"{name}: score.txt line lacks {miss}")
    summary[name] = {"bad": bad, "n_rollouts": len(ev), "check_passed": chk["passed"], "mine": mine}
    print(f"  {name:<18} check passed {chk['passed']} failures {chk['failures']}; rollouts {len(ev)} at S + 8192k, "
          f"k = 1..175; signature problems {bad or 'none'}; score.txt line carries the values: {not miss}")
    if arm == "control":
        print(f"      C4 min dv(obs6) {mine['min_dv_predict_values']}; actor responds to command slots min "
              f"{mine['min_dmu_phase']}; phase rows 0; digest_load == digest_start; probe not phase-fed")
    else:
        print(f"      C1 actor change max {mine['max_dmu_phase']}; C2 rollout 1 critic phase response "
              f"{mine['zero_init_max_dv_phase']} cols {mine['zero_init_max_value_phase_cols']}; C3 rollouts 2-175 min "
              f"per path {mine['updated_min_dv_phase']} cols min {mine['updated_min_value_phase_cols']}; C4 min "
              f"dv(obs6) {mine['min_dv_predict_values']}; C5 max err {mine['max_phase_buffer_max_err']}; "
              f"pre-settle rows {mine['phase_rows_presettle_total']}, k max {mine['phase_k_max']}")
print(f"\nMANIPULATION comparisons: {n_cmp[0]} made, {n_cmp[1]} differing; problems {problems or 'none'}")
json.dump({"summary": summary, "n_cmp": n_cmp, "problems": problems}, open(sys.argv[1], "w"), indent=1, default=str)
