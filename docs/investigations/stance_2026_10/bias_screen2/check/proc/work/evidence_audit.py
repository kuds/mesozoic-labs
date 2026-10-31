"""My own per-rollout reader of runs/<name>/arm_evidence.jsonl (no project code): the hook in force on every rollout."""
import json, math, os, sys
BS2 = sys.argv[1]
S, R = 2_097_152, 8192
QUEUE = [("control", 2501), ("critic_blind_tail3", 2501), ("control", 2511), ("critic_blind_tail3", 2511),
         ("control", 2521), ("critic_blind_tail3", 2521), ("control", 2531), ("critic_blind_tail3", 2531)]
DV = ("dv_predict_values", "dv_forward", "dv_evaluate_actions", "dv_mirrored")
allprob = {}
first = {}
for arm, seed in QUEUE:
    name = f"{arm}_{seed}"
    rows = [json.loads(l) for l in open(os.path.join(BS2, "runs", name, "arm_evidence.jsonl"))]
    meta = json.load(open(os.path.join(BS2, "runs", name, "run_meta.json")))["arm_evidence"]
    prob = []
    steps = [r["num_timesteps"] for r in rows]
    if steps != [S + R * k for k in range(1, 176)]:
        prob.append("steps are not S + 8192 k, k = 1..175, once each in order")
    blind = arm == "critic_blind_tail3"
    stats = {k: [] for k in DV + ("dmu_actor", "bvr_over_scale", "bvc_over_scale", "adv_err_rel", "adv_mc", "rav")}
    digests = set()
    prev_count = None
    prev_clock = None
    for i, r in enumerate(rows):
        k = i + 1
        for f in DV:
            v = r[f]
            stats[f].append(v)
            if blind and not (v == 0.0):
                prob.append(f"row {k}: {f} = {v!r} (blind must be exactly 0.0)")
            if not blind and not (isinstance(v, float) and math.isfinite(v) and v > 0.0):
                prob.append(f"row {k}: {f} = {v!r} (control must change, > 0)")
        if not (r["dmu_actor"] > 0):
            prob.append(f"row {k}: actor mean does not change with obs 6")
        stats["dmu_actor"].append(r["dmu_actor"])
        vs = r["value_scale"]
        stats["bvr_over_scale"].append(r["buffer_values_vs_raw_value_mlp"] / vs)
        stats["bvc_over_scale"].append(r["buffer_values_vs_critic"] / vs)
        if blind and not (r["buffer_values_vs_raw_value_mlp"] > 1e-5 * vs):
            prob.append(f"row {k}: buffer values not distinct from the unblinded value MLP")
        if not (r["buffer_values_vs_critic"] <= 1e-5 * vs):
            prob.append(f"row {k}: buffer values differ from the run's critic by {r['buffer_values_vs_critic']}")
        if not (r["buffer_gae_lambda"] == 0.95 and r["model_gae_lambda"] == 0.95):
            prob.append(f"row {k}: gae_lambda")
        rel = r["adv_err_vs_gae095"] / r["adv_scale"]
        stats["adv_err_rel"].append(rel)
        if not rel <= 1e-4:
            prob.append(f"row {k}: advantages differ from GAE(0.95) by {rel}")
        stats["adv_mc"].append(r["adv_err_vs_mc"])
        if not r["adv_err_vs_mc"] > 1e-4 * r["adv_scale"]:
            prob.append(f"row {k}: advantages equal the lambda-1 recursion")
        stats["rav"].append(r["returns_minus_adv_minus_values"])
        if not r["returns_minus_adv_minus_values"] <= 1e-5 * max(1.0, vs):
            prob.append(f"row {k}: returns != adv + values ({r['returns_minus_adv_minus_values']})")
        if r["obs_rms_equal_start"] is not False:
            prob.append(f"row {k}: obs stats equal the start")
        if r["obs_rms_digest"] in digests:
            prob.append(f"row {k}: obs_rms digest repeats (stats did not move)")
        digests.add(r["obs_rms_digest"])
        if prev_count is not None and not r["obs_rms_count"] > prev_count:
            prob.append(f"row {k}: obs_rms count did not grow")
        prev_count = r["obs_rms_count"]
        if r["frozen_update"] is not None:
            prob.append(f"row {k}: frozen_update {r['frozen_update']}")
        if r["state_problems"]:
            prob.append(f"row {k}: state problems {r['state_problems']}")
        if prev_clock is not None and not r["clock"]["cpu_seconds"] >= prev_clock:
            prob.append(f"row {k}: clock went back")
        prev_clock = r["clock"]["cpu_seconds"]
    # the run_meta summary equals the rows
    chk = {
        "n": len(rows), "max_dv_predict_values": max(stats["dv_predict_values"]),
        "max_dv_forward": max(stats["dv_forward"]), "max_dv_evaluate_actions": max(stats["dv_evaluate_actions"]),
        "max_dv_mirrored": max(stats["dv_mirrored"]), "min_dv_predict_values": min(stats["dv_predict_values"]),
        "min_dmu_actor": min(stats["dmu_actor"]),
        "max_adv_err_vs_gae095": max(r["adv_err_vs_gae095"] for r in rows),
        "min_adv_err_vs_mc": min(r["adv_err_vs_mc"] for r in rows),
        "max_buffer_values_vs_critic": max(r["buffer_values_vs_critic"] for r in rows),
        "min_buffer_values_vs_raw_value_mlp": min(r["buffer_values_vs_raw_value_mlp"] for r in rows),
        "obs_rms_distinct_digests": len(digests),
        "steps_first_last": [steps[0], steps[-1]],
    }
    for k2, v in chk.items():
        if meta.get(k2) != v:
            prob.append(f"run_meta arm_evidence[{k2}] {meta.get(k2)!r} != rows {v!r}")
    first[name] = (rows[0]["digest_env_side"], rows[0]["digest_rewards"], rows[0]["digest_values"],
                   rows[1]["digest_env_side"])
    allprob[name] = prob
    print(f"{name}: rows {len(rows)}; dv max over 4 paths {max(max(stats[f]) for f in DV):.6g}, "
          f"dv min over 4 paths {min(min(stats[f]) for f in DV):.6g}; dmu_actor min {min(stats['dmu_actor']):.4g}; "
          f"|buffer - raw value MLP| / scale min {min(stats['bvr_over_scale']):.3g}; "
          f"|buffer - critic| / scale max {max(stats['bvc_over_scale']):.3g}; adv rel err max {max(stats['adv_err_rel']):.3g}; "
          f"distinct obs digests {len(digests)}; problems {len(prob)}")
    for p in prob[:10]:
        print("    ", p)
print()
for seed in (2501, 2511, 2521, 2531):
    c, b = first[f"control_{seed}"], first[f"critic_blind_tail3_{seed}"]
    print(f"seed {seed}: first rollout env side equal {c[0] == b[0]}, rewards equal {c[1] == b[1]}, "
          f"values equal {c[2] == b[2]}; second rollout env side equal {c[3] == b[3]}")
print("TOTAL PROBLEMS:", sum(len(v) for v in allprob.values()))
