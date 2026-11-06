"""prelaunch/setup: the facts v3 lists for train()'s pre-learning path, read from this lane's records per seed.

Usage: python -I -B extract_facts.py OUT.json   (reads runs/s<seed>/setup_check.json; imports no repository code)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

D = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/prelaunch/setup")
FIRST = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")
out = {}
for seed in (42, 52, 60):
    d = json.loads((D / "runs" / f"s{seed}" / "setup_check.json").read_text())
    w, opt, vn, sc, ma = (d.get(k, {}) for k in ("weights", "optimizer", "vecnormalize", "stage_config",
                                                 "model_attributes"))
    cap = (d.get("capture") or {}).get("at_learn") or {}
    rr = (d.get("runner") or {}).get("record") or {}
    sa = d.get("start_actions_on_pilot3_observations") or {}
    ro = d.get("rollout") or {}
    jb = d.get("judge_binding") or {}
    sh = d.get("shaping_simulation") or {}
    out[str(seed)] = {
        "runner_exit": (d.get("runner") or {}).get("exit"), "runner_sha256": d.get("runner_sha256"),
        "clone_head": rr.get("clone_head"), "notebook_sha256": rr.get("notebook_sha256"),
        "roles": (d.get("runner") or {}).get("roles"), "run_all": (d.get("runner") or {}).get("run_all"),
        "manual_cell_stopped_at_learn": rr.get("manual_cell_stopped_at_learn"), "learn_called": rr.get("learn_called"),
        "guard_hits": rr.get("guard_hits"), "guard_hits_final": d.get("guard_hits_final"),
        "entry_lines_ok": (rr.get("entry_lines") or {}).get("ok"),
        "entry_lines_missing": (rr.get("entry_lines") or {}).get("missing"),
        "entry_lines_forbidden_present": (rr.get("entry_lines") or {}).get("forbidden_present"),
        "entry_lines_other_warm_start": (rr.get("entry_lines") or {}).get("other_warm_start_lines"),
        "entry_lines_required": (rr.get("entry_lines") or {}).get("required"),
        "parent_pair": rr.get("parent_pair"), "locomotion_curriculum_at_end": rr.get("locomotion_curriculum_at_end"),
        "environments_imported_from": d.get("environments_imported_from"),
        "runner_environments_imported_from": rr.get("environments_imported_from"),
        "pairs_unchanged": d.get("pair_sha256_before") == d.get("pair_sha256_after"),
        "errors": d.get("errors"),
        "tensors_changed": w.get("tensors_changed"), "n_tensors": w.get("n_tensors"),
        "parent_command_column_l2": w.get("parent_command_column_l2"),
        "loaded_command_columns_max_abs": w.get("loaded_command_columns_max_abs"),
        "columns_0_60_bit_equal": w.get("columns_0_60_bit_equal"),
        "log_std_bit_equal_parent": w.get("log_std_bit_equal_parent"),
        "log_std_mean_min_max": [(w.get("log_std") or {}).get(k) for k in ("mean", "min", "max")],
        "adam_step_pairs_parent_loaded": opt.get("step_pairs(parent,loaded)"),
        "adam_params_with_changed_moments_or_listed": {
            n: {k: v for k, v in e.items() if "n_changed" in k or "cmd_cols" in k or k.startswith("step")}
            for n, e in (opt.get("params_with_changed_moments_or_listed") or {}).items()},
        "optimizer_lr_parent_loaded": [opt.get("param_groups_lr_parent"), opt.get("param_groups_lr_loaded")],
        "vecnorm_parent": vn.get("parent"),
        "vecnorm_train": {k: (vn.get("train_env") or {}).get(k) for k in (
            "mean_0_60_bit_equal_parent", "var_0_60_bit_equal_parent", "mean_60_63", "var_60_63", "count",
            "ret_rms_mean", "ret_rms_var", "ret_rms_count", "training", "num_envs")},
        "vecnorm_eval": {k: (vn.get("eval_env") or {}).get(k) for k in (
            "mean_0_60_bit_equal_parent", "var_0_60_bit_equal_parent", "mean_60_63", "var_60_63", "count",
            "ret_rms_count", "training", "num_envs")},
        "at_learn": cap,
        "stage_config_git_commit": sc.get("git_commit"), "stage_config_run": sc.get("run"),
        "task_recorded": sc.get("task_fingerprint_task_sha256"), "task_tree": sc.get("tree_task_sha256"),
        "hp_ppo_tree": sc.get("hyperparameters_sha256_PPO_tree"), "hp_sac_tree": sc.get("hyperparameters_sha256_SAC_tree"),
        "curriculum_changed_vs_pilot3": sc.get("curriculum_changed_vs_pilot3"),
        "reward_weights_changed_vs_pilot3": sc.get("reward_weights_changed_vs_pilot3"),
        "hyperparameters_changed_vs_pilot3": sc.get("hyperparameters_changed_vs_pilot3"),
        "model_attributes": ma,
        "start_actions": {k: sa.get(k) for k in ("env", "steps", "terminated", "steps_with_nonzero_command_inputs",
                                                  "max_abs_action_delta_new_vs_pilot3",
                                                  "max_abs_mean_delta_new_vs_pilot3",
                                                  "max_abs_value_delta_new_vs_pilot3")},
        "rollout_live": {k: ro.get(k) for k in ("steps", "terminated", "max_action_delta_vs_pilot3_on_zeroed",
                                                 "max_mean_delta_vs_pilot3_on_zeroed",
                                                 "max_value_delta_vs_pilot3_on_zeroed",
                                                 "max_action_delta_pilot3_unprepared_on_live", "spawn_heading_rad",
                                                 "desired_heading_at_reset", "reset_obs_60_63")},
        "shaping_simulation": sh,
        "judge_accepted": jb.get("accepted"), "judge_binding_problems": jb.get("binding_problems"),
        "judge_anchors": jb.get("anchors"),
    }
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str) + "\n")
print("written", sys.argv[1])
