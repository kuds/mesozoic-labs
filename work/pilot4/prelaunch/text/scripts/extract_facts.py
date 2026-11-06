"""Read-only extraction of every number the final pilot-4 text cites from the pre-launch lanes' records.

Paths are worked out from this script's own location (<SP>/work/pilot4/prelaunch/text/scripts/), so no absolute path is
spelled here. Writes data/facts_text.json and prints a readable summary. Nothing is rolled, trained or written outside
the text lane's folder.
"""
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
PRE = os.path.dirname(LANE)
P4 = os.path.dirname(PRE)
WORK = os.path.dirname(P4)
TOL = os.path.join(PRE, "tol")
SET = os.path.join(PRE, "setup")
SMK = os.path.join(PRE, "smoke")
TOOLS = os.path.join(P4, "tools")
SEEDS = ("42", "52", "60")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with open(path) as f:
        return json.load(f)


def text(path):
    with open(path, errors="replace") as f:
        return f.read()


def pct(vals, p):
    a = np.asarray(sorted(vals), dtype=float)
    return float(np.percentile(a, p))


def r1(x):
    return None if x is None else round(float(x), 1)


facts = {}
out = []

# ---------------------------------------------------------------- tolerance lane
tol = {}
for s in SEEDS:
    d = load(os.path.join(TOL, "val_final", f"baseline_s{s}.json"))
    parts = d["parts"]
    chk = d["checks"]
    rec = d["recovery"]
    eps = d["panel"]["episodes"]
    beyond_1e5 = [
        (e["reset_seed"], e["angle_deg"], abs(e["heading_minus_twin_at_1000_deg"] - e["angle_deg"]))
        for e in rec["episodes"]
        if e.get("heading_minus_twin_at_1000_deg") is not None
        and abs(e["heading_minus_twin_at_1000_deg"] - e["angle_deg"]) > 1e-5
    ]
    # v3's entry rows: resets 3042-3061; yaw over episodes longer than 110 steps; D, R, W0 over full-length ones
    first20 = [e for e in eps if 3042 <= e["reset_seed"] <= 3061]
    yaw20 = [e["yaw_change_deg"] for e in first20 if e["T"] > 110 and e.get("yaw_change_deg") is not None]
    full20 = [e for e in first20 if e["full"]]
    row = {
        "n": len(first20),
        "n_full": len(full20),
        "yaw_med": r1(np.median(yaw20)),
        "yaw_p95_abs": r1(pct([abs(v) for v in yaw20], 95)),
        "D_p95": r1(pct([e["D_deg"] for e in full20], 95)),
        "D_med": r1(np.median([e["D_deg"] for e in full20])),
        "D_max": r1(max(e["D_deg"] for e in full20)),
        "R_med": r1(np.median([e["R_deg"] for e in full20])),
        "R_p95": r1(pct([e["R_deg"] for e in full20], 95)),
        "W0_p95": r1(pct([e["W0_deg"] for e in full20], 95)),
    }
    tol[s] = {
        "tool_sha256": d["tool_sha256"],
        "twin_tol": d["constants"]["TWIN_HEADING_TOL_DEG"],
        "head": d["head"],
        "labels": d["labels"]["counts"],
        "seed_outcome": d["labels"]["seed_outcome"],
        "scorer_sha256": d["labels"]["scorer_sha256"],
        "falls": [(e["reset_seed"], e["T"]) for e in eps if e.get("label") == "FALL"],
        "a": [parts["a"]["verdict"], r1(parts["a"]["median_signed_deg"]), r1(parts["a"]["p95_abs_deg"]), parts["a"]["n"]],
        "b": [parts["b"]["verdict"], r1(parts["b"]["D_p95_deg"]), r1(parts["b"]["W0_p95_deg"]), parts["b"]["n"]],
        "b_crab_cross": [round(parts["b"]["report_crab_median_deg"], 1), round(parts["b"]["report_end_cross_track_median_m"], 2)],
        "c": [parts["c"]["verdict"], parts["c"]["n_counting"], r1(parts["c"]["median_r_deg"]), r1(parts["c"]["p95_r_deg"]),
              parts["c"]["within_11_5"], parts["c"]["within_5"], parts["c"]["share_removed_median"]],
        "straight": parts["straight"],
        "twin": chk["turned_twin"],
        "prefix": [chk["prefix"]["equal"], chk["prefix"]["episodes"]],
        "obs651": [chk["obs651"]["only_command_channels"], chk["obs651"]["episodes"], chk["obs651"]["channels_seen"]],
        "zero_nudge": chk["zero_nudge"],
        "prey_in_place_med_r": r1(rec["prey_in_place"]["summary"]["median_r_deg"]),
        "turned_at_reset": [r1(rec["turned_at_reset"]["summary"]["median_r_deg"]), rec["turned_at_reset"]["summary"]["within_11_5"]],
        "beyond_1e-5": beyond_1e5,
        "entry_row_3042_3061": row,
        "crab_cross_full_panel": [round(float(np.median([e["crab_median_deg"] for e in eps if e["full"]])), 1),
                                  round(float(np.median([e["end_cross_track_m"] for e in eps if e["full"]])), 2)],
        "crab_cross_3042_3061": [round(float(np.median([e["crab_median_deg"] for e in full20])), 1),
                                 round(float(np.median([e["end_cross_track_m"] for e in full20])), 2)],
        "yaw_repro_max_abs_diff_deg": d["panel"]["yaw_repro_max_abs_diff_deg"],
        "all_rolled_match": d["panel"]["all_rolled_match"],
    }
facts["tol"] = tol
p3val = {}
for s in SEEDS:
    d = load(os.path.join(TOL, "val_final", f"p3val_s{s}.json"))
    p = d.get("parts", {})
    p3val[s] = {
        "mode": d.get("mode"),
        "all_rolled_match": d.get("panel", {}).get("all_rolled_match"),
        "n_episodes": len(d.get("panel", {}).get("episodes", [])),
        "yaw_repro_max_abs_diff_deg": d.get("panel", {}).get("yaw_repro_max_abs_diff_deg"),
        "verdicts": [p.get("a", {}).get("verdict"), p.get("b", {}).get("verdict"), p.get("c", {}).get("verdict")],
        "steps": d.get("steps"),
    }
facts["p3val"] = p3val
ts = load(os.path.join(TOL, "tests_final", "summary.json"))
facts["tests"] = {k: ([v["tests"], v["failures"], v["errors"], v["skipped"]] if "tests" in v else v["summary_line"])
                  for k, v in ts["files"].items()}
facts["tests_total"] = [ts["total_pytest_tests"], ts["total_pytest_failed_or_error"], ts["total_pytest_skipped"]]
e2e = load(os.path.join(TOL, "e2e", "e2e_summary.json"))
fx = load(os.path.join(TOL, "e2e", "e2e_fix_checks.json"))
facts["e2e"] = [sum(1 for c in e2e["checks"] if c["ok"]), len(e2e["checks"])]
fxc = fx["checks"] if isinstance(fx, dict) and "checks" in fx else fx
facts["e2e_fix"] = [sum(1 for c in fxc if c.get("ok")), len(fxc)]
sums = os.path.join(TOOLS, "SHA256SUMS.txt")
facts["sha256sums"] = [sha(sums), sum(1 for ln in text(sums).splitlines() if ln and not ln.startswith("#"))]
facts["tool_sha"] = {n: sha(os.path.join(TOOLS, n)) for n in (
    "pilot4_straight.py", "pilot4_score.py", "pilot4_judge.py", "nb_pilot_runner4.py", "iso.py", "pilot4_video.py",
    "pilot4_habit.py", "pilot4_noise.py", "pilot3_vg.py", "pilot4_mirror.py", "pilot4_mech.py")}
for n in ("sha256sums_check.txt", "before_list_check.txt"):
    t = text(os.path.join(TOL, "pin", n))
    facts["tol_pin_" + n] = [t.count(": OK\n"), t.count(": FAILED")]

# ---------------------------------------------------------------- setup lane
fh = load(os.path.join(SET, "data", "facts_head.json"))
setup = {}
for s in SEEDS:
    f = fh[s]
    setup[s] = {
        "runner_exit": f["runner_exit"],
        "learn_called": f["learn_called"],
        "guard_hits": len(f["guard_hits"]) + len(f["guard_hits_final"]),
        "entry_lines_ok": f["entry_lines_ok"],
        "tensors_changed": {k: [v["n_changed"], v["changed_columns"], v["loaded_values_at_changed_max_abs"]]
                            for k, v in f["tensors_changed"].items()},
        "n_tensors": f["n_tensors"],
        "log_std_bit_equal": f["log_std_bit_equal_parent"],
        "log_std_mean": round(f["log_std_mean_min_max"][0], 4),
        "adam_step": f["adam_step_pairs_parent_loaded"],
        "lr": f["optimizer_lr_parent_loaded"][1],
        "vn_train_60_63": [f["vecnorm_train"]["mean_60_63"], f["vecnorm_train"]["var_60_63"],
                           f["vecnorm_train"]["mean_0_60_bit_equal_parent"], f["vecnorm_train"]["var_0_60_bit_equal_parent"]],
        "ret_rms": [f["vecnorm_train"]["ret_rms_mean"], f["vecnorm_train"]["ret_rms_var"], f["vecnorm_train"]["ret_rms_count"]],
        "git_commit": f["stage_config_git_commit"],
        "task": [f["task_recorded"], f["task_tree"]],
        "hp": f["hp_ppo_tree"],
        "judge_accepted": f["judge_accepted"],
        "judge_binding_problems": f["judge_binding_problems"],
        "start_actions": f["start_actions"],
    }
facts["setup"] = setup
cmpd = load(os.path.join(SET, "data", "compare_head_vs_7700f28.json"))
facts["setup_compare_keys"] = list(cmpd.keys())[:20]
for n in ("pt_p0.txt", "pt_p1.txt", "pt_p2.txt"):
    t = text(os.path.join(SET, "logs", n)).strip().splitlines()
    facts["setup_" + n] = [ln for ln in t if re.search(r"(passed|failed|deselected)", ln)][-1:]
for n in ("notebook_cells.txt", "plant_contract_check.txt", "species_catalog_check.txt", "digest_check_0e6a57b.txt"):
    t = text(os.path.join(SET, "logs", n)).strip().splitlines()
    facts["setup_" + n] = t[-2:]
gd = text(os.path.join(SET, "logs", "golden_diff_31d02e42_vs_generated_0e6a57b.txt")).splitlines()
facts["setup_golden_diff"] = [sum(1 for ln in gd if ln.startswith("-") and not ln.startswith("---")),
                              sum(1 for ln in gd if ln.startswith("+") and not ln.startswith("+++"))]
fails = []
for n in ("pt_p0.txt", "pt_p1.txt", "pt_p2.txt"):
    for ln in text(os.path.join(SET, "logs", n)).splitlines():
        if ln.startswith("FAILED ") or ln.startswith("XFAIL "):
            fails.append(ln.split(" - ")[0][:150])
facts["setup_failed_xfail"] = fails

# ---------------------------------------------------------------- smoke lane
ev = load(os.path.join(SMK, "out", "smoke_evidence_launch_s42.json"))
rec = ev["record"]
facts["smoke_launch"] = {
    "exit": rec["exit"], "tool_sha256": rec["tool_sha256"], "head": rec["clone_head"], "roles": rec["roles"],
    "run_all": rec["run_all"], "override": [rec["override"]["before"], rec["override"]["after"],
                                            rec["override"]["hyperparameters_sha256_before"],
                                            rec["override"]["hyperparameters_sha256_after"]],
    "entry_ok": rec["entry_lines"]["ok"], "first_eval": rec["entry_lines"]["first_eval_line"],
    "missing": rec["entry_lines"]["missing"], "forbidden": rec["entry_lines"]["forbidden_present"],
    "other_warm_start": rec["entry_lines"]["other_warm_start_lines"],
}
facts["smoke_timing"] = load(os.path.join(SMK, "out", "timing.json"))
lin = []
for m in re.finditer(r'"mesozoic_task_lineage"', text(os.path.join(SMK, "out", "smoke_evidence_launch_s42.json"))):
    lin.append(m.start())
facts["smoke_lineage_mentions_launch"] = len(lin)
sc_l = load(os.path.join(SMK, "runs", "launch", "pilot4_walk_s42", "03_locomotion", "stage_config.json"))
sc_r = load(os.path.join(SMK, "runs", "resume", "pilot4_walk_s42", "03_locomotion", "stage_config.json"))


def flat(o, p=""):
    r = {}
    if isinstance(o, dict):
        for k, v in o.items():
            r.update(flat(v, p + "." + k if p else k))
    else:
        r[p] = o
    return r


fl, fr = flat(sc_l), flat(sc_r)
facts["smoke_resume_changed_leaves"] = sorted(k for k in set(fl) | set(fr) if fl.get(k) != fr.get(k))
facts["smoke_stage_config"] = {k: fl.get(k) for k in (
    "git_commit", "run.load_mode", "run.parent_checkpoint_sha256", "run.parent_task_sha256", "run.n_envs",
    "run.timesteps", "run.seed", "run.hyperparameters_sha256", "curriculum.eval_freq", "curriculum.checkpoint_freq",
    "curriculum.eval_episodes", "curriculum.max_checkpoints", "curriculum.warmup_timesteps",
    "curriculum.ramp_start_value", "curriculum.initialize_from_own_checkpoint", "task_fingerprint.task_sha256",
    "reward_weights.command_mode", "reward_weights.prey_lateral_range", "reward_weights.walk_heading_price_weight",
    "reward_weights.walk_heading_price_scale_rad")}
facts["smoke_resume_values"] = {k: fr.get(k) for k in ("run.timesteps", "run.resume_load_path",
                                                       "run.resume_checkpoint_sha256", "run.load_mode")}
st = text(os.path.join(SMK, "judge", "status.txt")).splitlines()
rc = {}
for ln in st:
    m = re.match(r"(\S+) rc=(\d+)", ln)
    if m:
        rc[m.group(1)] = int(m.group(2))
facts["smoke_judge_rc"] = rc
facts["smoke_judge_refused"] = sorted(k for k, v in rc.items() if v == 2)
facts["smoke_judge_accepted_judged"] = sorted(k for k, v in rc.items() if v == 0 and re.match(r"[AB]\d", k))
probes = {}
for cell in ("OUT/s42/32768", "OUT/s42/65536", "OUT/s42/50000", "OUT_resume/s42/65536"):
    pj = load(os.path.join(SMK, "judge", cell, "probe.json"))
    labels = {}
    bj = load(os.path.join(SMK, "judge", cell, "binding.json"))
    probes[cell] = {
        "episodes": pj["meta"]["episodes"],
        "drift": pj["meta"]["env_kwargs_drift_vs_recorded"],
        "interface": pj["meta"]["plant_identity_policy_interface_revision"],
        "validation_only": bj.get("validation_only"),
        "binding_keys": sorted(bj.keys())[:40],
        "lengths": sorted(set(pj["summary"]["length"].values())) if isinstance(pj["summary"]["length"], dict) else None,
    }
facts["smoke_probes"] = probes
sj = load(os.path.join(SMK, "readouts", "straight", "s42_65536.json"))
pa = sj["parts"]
facts["smoke_straight"] = {
    "tool": sj["tool_sha256"], "all_rolled_match": sj["panel"]["all_rolled_match"],
    "n": len(sj["panel"]["episodes"]),
    "a": [pa["a"]["verdict"], r1(pa["a"]["median_signed_deg"]), r1(pa["a"]["p95_abs_deg"])],
    "b": [pa["b"]["verdict"], r1(pa["b"]["D_p95_deg"]), r1(pa["b"]["W0_p95_deg"])],
    "c": [pa["c"]["verdict"], pa["c"]["n_counting"], r1(pa["c"]["median_r_deg"]), r1(pa["c"]["p95_r_deg"]),
          pa["c"]["within_11_5"], pa["c"]["within_5"]],
    "straight": pa["straight"],
    "prefix": [sj["checks"]["prefix"]["equal"], sj["checks"]["prefix"]["episodes"]],
    "obs651": [sj["checks"]["obs651"]["only_command_channels"], sj["checks"]["obs651"]["episodes"]],
    "psi_cmd": sj["psi_cmd"]["values_deg"], "psi_info": sj["psi_cmd"]["info_check_max_abs_diff_rad"],
}
sv = load(os.path.join(SMK, "readouts", "score", "score_validation_65536.json"))
sa = load(os.path.join(SMK, "readouts", "score", "score_as_is.json"))
facts["smoke_score"] = {
    "validation_pilot": [sv["pilot"]["outcome"], sv["pilot"]["possible_outcomes"]],
    "validation_seed_primary": sv["per_seed_primary"],
    "as_is_pilot": sa.get("pilot", {}).get("outcome"),
    "existence": sv["existence"]["reading"],
}
hs = load(os.path.join(SMK, "readouts", "habit", "s42_65536_seed.json"))
facts["smoke_habit_reading"] = hs.get("reading", hs.get("verdict"))
for n in ("sha256sums_check_start.txt", "sha256sums_check_end.txt"):
    t = text(os.path.join(SMK, "pin", n))
    facts["smoke_pin_" + n] = [t.count(": OK\n"), t.count("FAILED")]
for n in ("V1_video_egl.log", "V2_video_osmesa.log"):
    t = text(os.path.join(SMK, "readouts", "logs", n)).strip().splitlines()
    facts["smoke_" + n] = t[-1:]

with open(os.path.join(LANE, "data", "facts_text.json"), "w") as f:
    json.dump(facts, f, indent=1, default=str)


def show(k, v):
    print(f"{k}: {json.dumps(v, default=str)[:1200]}")


for k, v in facts.items():
    if isinstance(v, dict) and k in ("tol", "setup", "p3val"):
        for s, vv in v.items():
            for kk, x in vv.items():
                show(f"{k}.{s}.{kk}", x)
    else:
        show(k, v)
sys.exit(0)
