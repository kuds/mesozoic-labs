"""prelaunch/setup: summarise the head's set-up check per seed and compare every number with the postfix/code records.

Usage: python -I -B compare_head.py OUT.json   (imports no repository code; reads JSON, npy and npz files only)

1. Builds, per seed, the same summary row as postfix/code's summarize.py (its key list and reads, unchanged), from
   this lane's runs/s<seed>/setup_check.json and p3_side.json; "same_as_0bcb0cf_check" becomes
   "same_as_7700f28_check" (the same five comparisons, against postfix/code's runs/s<seed>/setup_check.json).
2. Compares that row leaf by leaf with postfix/code/logs/setup_summary.json (7700f28) and lists every leaf that
   differs (bitwise: floats by ==, -0.0 apart from 0.0, NaN equal to NaN; types must match).
3. Deep-diffs the whole setup_check.json and p3_side.json against postfix/code's, per seed, listing every differing
   leaf path (the runner and capture sections hold paths and timings; listed, then classified).
4. Compares the saved arrays (actions, rollout series) bit for bit and the replay/kwargs JSON files by sha256.
5. Compares this lane's copied stage_config.json, plant_identity.json, task_fingerprint.json with the fix lane's
   copies at the same head (tools/out/fix/setup/s<seed>/stage), leaf by leaf.
6. Compares the judge-bound keys across the three seeds.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

D = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/prelaunch/setup")
PF = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/postfix/code")
FIXSETUP = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/tools/out/fix/setup")
SEEDS = (42, 52, 60)
ROLL_KEYS = ("steps", "terminated", "price_sum", "price_min_step", "obs62", "obs60", "theta_deg", "metrics",
             "obs62_saturated_steps", "price_saturated_steps", "return_built_env", "return_twin",
             "obs_0_60_mismatch_steps", "state_mismatch_steps", "flag_mismatch_steps", "reward_identity_fail_steps",
             "reward_total_identity_fail_steps", "common_info_keys_with_any_mismatch", "info_keys_only_in_built_env",
             "info_keys_only_in_twin", "max_action_delta_vs_pilot3_on_zeroed", "max_value_delta_vs_pilot3_on_zeroed")


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def leaves_diff(a, b, path="") -> list:
    """Every leaf path where a and b differ (a key present on one side only is a difference)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in a or k not in b:
                out.append([p, a.get(k, "<absent>"), b.get(k, "<absent>")])
            else:
                out += leaves_diff(a[k], b[k], p)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += leaves_diff(x, y, f"{path}[{i}]")
        return out
    return [] if same(a, b) else [[path, a, b]]


def n_leaves(a) -> int:
    if isinstance(a, dict):
        return sum(n_leaves(v) for v in a.values())
    if isinstance(a, list):
        return sum(n_leaves(v) for v in a) if a else 1
    return 1


def short(v, n=160):
    s = json.dumps(v, default=str)
    return s if len(s) <= n else s[:n] + "..."


def summary_row(d: dict, p3s: dict | None, old: dict) -> dict:
    """postfix/code summarize.py's row, unchanged but for the comparison base (7700f28's own record)."""
    w, vn, sc, ma, sa, ro = (d.get(k, {}) for k in ("weights", "vecnormalize", "stage_config", "model_attributes",
                                                     "start_actions_on_pilot3_observations", "rollout"))
    row = {
        "environments_imported_from": d.get("environments_imported_from"),
        "errors": d.get("errors"), "guard_hits_final": d.get("guard_hits_final"),
        "learn_reached": d.get("learn_reached"), "manual_cell_stopped_by_learn_stop": d.get("manual_cell_stopped_by_learn_stop"),
        "pairs_unchanged": d.get("pair_sha256_before") == d.get("pair_sha256_after"),
        "git_commit": sc.get("git_commit"), "load_mode": (sc.get("run") or {}).get("load_mode"),
        "lineage_mode": (ma.get("lineage") or {}).get("mode"),
        "zeroed_command_parameters": (ma.get("lineage") or {}).get("zeroed_command_parameters"),
        "task_recorded": sc.get("task_fingerprint_task_sha256"), "task_tree": sc.get("tree_task_sha256"),
        "parent_task": (sc.get("run") or {}).get("parent_task_sha256"),
        "hp_ppo_recorded": (sc.get("run") or {}).get("hyperparameters_sha256"), "hp_ppo_tree": sc.get("hyperparameters_sha256_PPO_tree"),
        "tensors_changed": {k: [v.get("n_changed"), v.get("changed_columns"), v.get("loaded_values_at_changed_max_abs")]
                            for k, v in (w.get("tensors_changed") or {}).items()},
        "columns_0_60_bit_equal": w.get("columns_0_60_bit_equal"),
        "log_std_bit_equal_parent": w.get("log_std_bit_equal_parent"),
        "obs_rms_train": {k: (vn.get("train_env") or {}).get(k) for k in ("mean_0_60_bit_equal_parent", "var_0_60_bit_equal_parent", "mean_60_63", "var_60_63", "ret_rms_count")},
        "obs_rms_eval": {k: (vn.get("eval_env") or {}).get(k) for k in ("mean_0_60_bit_equal_parent", "var_0_60_bit_equal_parent", "mean_60_63", "var_60_63")},
        "start_actions": {k: sa.get(k) for k in ("steps", "steps_with_nonzero_command_inputs", "max_abs_action_delta_new_vs_pilot3", "max_abs_mean_delta_new_vs_pilot3", "max_abs_value_delta_new_vs_pilot3")},
        "rollout": {k: ro.get(k) for k in ROLL_KEYS},
        "price_weight_zero_twin": ro.get("price_weight_zero_twin"),
        "same_as_7700f28_check": {
            "rollout": {k: same(ro.get(k), (old.get("rollout") or {}).get(k)) for k in ROLL_KEYS},
            "start_actions": same(sa, old.get("start_actions_on_pilot3_observations")),
            "vecnormalize": same(vn, old.get("vecnormalize")),
            "weights_log_std": same(w.get("log_std"), (old.get("weights") or {}).get("log_std")),
            "extra_episodes": same(d.get("extra_episodes"), old.get("extra_episodes")),
        },
    }
    if p3s is not None:
        row["p3_side"] = {k: p3s.get(k) for k in ("environments_imported_from", "pilot3_env_kwargs_equal_built_p3x", "pilot3_env_kwargs_diff")}
        for k in ("twin_replay_vs_built_tree", "p3x_replay_vs_built_tree", "p3x_builtactions_replay_vs_built_tree"):
            row["p3_side"][k] = p3s.get(k)
        row["p3_side"]["pilot3_closed_loop_actions_bit_equal"] = (p3s.get("pilot3_closed_loop") or {}).get("actions_bit_equal_built_tree_closed_loop")
    return row


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


out: dict = {"summary": {}, "vs_setup_summary_7700f28": {}, "deep_diff_setup_check": {}, "deep_diff_p3_side": {},
             "arrays": {}, "files_sha256": {}, "stage_records_vs_fix_lane_0e6a57b": {}, "judge": {}}
ref_summary = json.loads((PF / "logs" / "setup_summary.json").read_text())
for seed in SEEDS:
    run, old_run = D / "runs" / f"s{seed}", PF / "runs" / f"s{seed}"
    d = json.loads((run / "setup_check.json").read_text())
    old = json.loads((old_run / "setup_check.json").read_text())
    p3s = json.loads((run / "p3_side.json").read_text()) if (run / "p3_side.json").is_file() else None
    old_p3s = json.loads((old_run / "p3_side.json").read_text())
    row = summary_row(d, p3s, old)
    out["summary"][str(seed)] = row
    ref = dict(ref_summary[str(seed)])
    ref_same = ref.pop("same_as_0bcb0cf_check")
    mine = {k: v for k, v in row.items() if k != "same_as_7700f28_check"}
    diffs = leaves_diff(mine, ref)
    out["vs_setup_summary_7700f28"][str(seed)] = {
        "leaves_compared": n_leaves(ref), "differing_leaves": [[p, a, b] for p, a, b in diffs],
        "postfix_same_as_0bcb0cf_all_true": all(v for v in ref_same["rollout"].values())
        and all(v for k, v in ref_same.items() if k != "rollout"),
        "head_same_as_7700f28_all_true": all(row["same_as_7700f28_check"]["rollout"].values())
        and all(v for k, v in row["same_as_7700f28_check"].items() if k != "rollout"),
    }
    full = leaves_diff(d, old)
    by_top: dict = {}
    for p, a, b in full:
        top = p.split(".")[0].split("[")[0]
        by_top.setdefault(top, []).append([p, short(a), short(b)])
    out["deep_diff_setup_check"][str(seed)] = {
        "leaves_in_head_record": n_leaves(d), "leaves_in_7700f28_record": n_leaves(old),
        "sections_with_differences": {k: len(v) for k, v in by_top.items()},
        "sections_identical": sorted(k for k in set(d) & set(old) if k not in by_top),
        "differences": by_top,
    }
    if p3s is not None:
        out["deep_diff_p3_side"][str(seed)] = [[p, short(a), short(b)] for p, a, b in leaves_diff(p3s, old_p3s)]
    arr = {}
    for name in ("actions_built_rollout.npy", "actions_p3x_closed_loop.npy"):
        a, b = np.load(run / name), np.load(old_run / name)
        arr[name] = {"shape": list(a.shape), "dtype": str(a.dtype), "bit_equal": bool(a.shape == b.shape and a.dtype == b.dtype
                                                                                        and a.tobytes() == b.tobytes())}
    za, zb = np.load(run / "rollout_series.npz"), np.load(old_run / "rollout_series.npz")
    arr["rollout_series.npz"] = {k: bool(k in zb.files and za[k].shape == zb[k].shape and za[k].tobytes() == zb[k].tobytes())
                                 for k in za.files}
    arr["rollout_series.npz_keys_equal"] = sorted(za.files) == sorted(zb.files)
    out["arrays"][str(seed)] = arr
    out["files_sha256"][str(seed)] = {
        name: {"head": sha(run / name), "7700f28": sha(old_run / name), "equal": sha(run / name) == sha(old_run / name)}
        for name in ("kwargs.json", "replay_twin_built_tree.json", "replay_p3x_built_tree.json",
                     "replay_p3x_builtactions_built_tree.json", "actions_built_rollout.npy",
                     "actions_p3x_closed_loop.npy", "rollout_series.npz")}
    st = {}
    for name in ("stage_config.json", "plant_identity.json", "task_fingerprint.json"):
        a = json.loads((run / "stage" / name).read_text())
        fx = FIXSETUP / f"s{seed}" / "stage" / name
        if fx.is_file():
            b = json.loads(fx.read_text())
            st[name] = {"leaves": n_leaves(a), "differing_leaves": [[p, short(x), short(y)] for p, x, y in leaves_diff(a, b)],
                        "bytes_equal": (run / "stage" / name).read_bytes() == fx.read_bytes()}
        else:
            st[name] = "no fix-lane copy"
    out["stage_records_vs_fix_lane_0e6a57b"][str(seed)] = st
    jb = d.get("judge_binding") or {}
    out["judge"][str(seed)] = {k: jb.get(k) for k in ("accepted", "binding_problems", "binding_info", "run_id_problem",
                                                     "commit_problem", "plant_problems", "env_kwargs_drift",
                                                     "checks_separately", "lineage_zip_equals_in_memory",
                                                     "edge_zip_data_num_timesteps", "judge_sha256", "anchors")}
# 6. the judge-bound keys across the seeds
bound = {s: json.loads((D / "runs" / f"s{s}" / "judge_bound_keys.json").read_text()) for s in SEEDS}
cross = {}
for s in SEEDS[1:]:
    cross[f"s42_vs_s{s}"] = [[p, short(a), short(b)] for p, a, b in leaves_diff(bound[42], bound[s])]
out["judge_bound_keys_across_seeds"] = cross
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str) + "\n")
for seed in SEEDS:
    v = out["vs_setup_summary_7700f28"][str(seed)]
    dd = out["deep_diff_setup_check"][str(seed)]
    print(seed, "summary leaves", v["leaves_compared"], "differing", len(v["differing_leaves"]),
          [p for p, _, _ in v["differing_leaves"]], "| same_as_7700f28 all", v["head_same_as_7700f28_all_true"],
          "| deep sections differing", dd["sections_with_differences"], "| judge accepted",
          out["judge"][str(seed)]["accepted"])
