"""Process lens: compare each run_meta.json / check.json with the pins (APPROVED), SCREEN.json and file hashes now.
Reads only; writes nothing (prints)."""
import hashlib
import json
import os
import sys

BS = sys.argv[1]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


appr = json.load(open(os.path.join(BS, "APPROVED")))
appr_sha = sha(os.path.join(BS, "APPROVED"))
crit_sha = sha(os.path.join(BS, "criteria.txt"))
scr_sha = sha(os.path.join(BS, "SCREEN.json"))
tools_now = {k: sha(os.path.join(BS, "tools", k)) for k in appr["tool_sha256"]}
print("APPROVED sha256 now", appr_sha)
print("criteria now", crit_sha, "== APPROVED", crit_sha == appr["criteria_sha256"])
print("SCREEN now", scr_sha, "== APPROVED", scr_sha == appr["screen_sha256"])
for k, v in tools_now.items():
    print("tool", k, v == appr["tool_sha256"][k])
screen = json.load(open(os.path.join(BS, "SCREEN.json")))
queue = [(a, s) for a, s in screen["queue"]]
print("queue", queue, "n", len(queue))
S, E = 1048576, 2490368
problems = []
for arm, seed in queue:
    name = f"{arm}_{seed}"
    rd = os.path.join(BS, "runs", name)
    m = json.load(open(os.path.join(rd, "run_meta.json")))
    c = json.load(open(os.path.join(rd, "check.json")))
    p = []
    def want(cond, msg):
        if not cond:
            p.append(msg)
    want(m["name"] == name, "name")
    want(m["status"] == "done", f"status {m['status']}")
    want(m["exit"] == 0, f"exit {m['exit']}")
    want(m["stop_reason"] == "end", f"stop {m['stop_reason']}")
    want(m["arm"] == arm and m["seed"] == seed, "arm/seed")
    want(m["argv"][1:] == ["--arm", arm, "--seed", str(seed)], f"argv {m['argv']}")
    want(m["S"] == S and m["E"] == E and m["T"] == E and m["target"] == E, "S/E/T")
    want(m["num_timesteps_end"] == E, "num_timesteps_end")
    want(m["stop_at_end"]["flag"] is True and m["stop_at_end"]["step"] == E, "stop_at_end")
    want(m["criteria_sha256"] == crit_sha == appr["criteria_sha256"], "criteria sha")
    want(m["approved_sha256"] == appr_sha, "approved sha")
    want(m["screen_sha256"] == scr_sha, "screen sha")
    want(m["code"]["tree_digest"] == appr["code_tree_digest"], "code digest")
    want(m["code"]["commit"] == "8737cd3d86431de1bdd0fcdb49a0d956a45e084d" and m["code"]["files"] == 1406, "code commit/files")
    for k, v in m["tool_sha256"].items():
        want(v == appr["tool_sha256"][k] == tools_now[k], f"tool {k}")
    want(set(m["tool_sha256"]) == {"run_bs.py", "read_bs.py", "run_waves_bs.sh", "bs_common.py", "bs_hooks.py"},
         f"tool keys {sorted(m['tool_sha256'])}")
    want(m["digest_start"]["num_timesteps"] == S and m["digest_end"]["num_timesteps"] == E, "digest steps")
    want(m["digest_end"]["n_updates"] > m["digest_start"]["n_updates"], "n_updates grew")
    want(m["digest_end"]["params"] != m["digest_start"]["params"], "params changed")
    want(m["code_seal_after"] == "unchanged", "seal")
    want(m["modules_outside_code"] == [], "modules outside")
    want(m["arm_state_end"] == "in force", "arm state end")
    want(m["start_pair"]["zip_sha256"] == "0f765bd692371f212555f6b203c4f2ce14aaa05df62f58202017c227424d2b80", "zip")
    want(m["start_pair"]["pkl_sha256"] == "52e52a7b9820af808265acb2d4f6d1573e6edebb550393db8ee47d9676229bcb", "pkl")
    want(m["parent_records"]["policy_probe.npz"] == "4e0e7220668faf124d394bffb760ba246d54bed8f48cb79925cb36234257fc11", "probe parent")
    want(m["heartbeat"]["rollout_end"] == E - 8192, "heartbeat")
    want(m["arm_evidence"]["n"] == 175, "evidence n")
    want(m["arm_evidence"]["steps_first_last"] == [S + 8192, E - 8192], "evidence steps")
    want(m["arm_evidence"]["state_problems"] == [], "state problems")
    want(m["collapse"]["tripped"] is False, "collapse")
    ae = m["arm_applied"]
    ev = m["arm_evidence"]
    if arm == "control":
        want(ae["mlp_extractor_class"] == "MlpExtractor" and ae["mlp_extractor_class_before"] == "MlpExtractor", "ctrl class")
        want(ev["min_dv_predict_values"] > 0, "ctrl critic sees obs6")
    else:
        want(ae["mlp_extractor_class"] == "CriticBlindTail3MlpExtractor", f"blind class {ae['mlp_extractor_class']}")
        for k in ("max_dv_predict_values", "max_dv_forward", "max_dv_evaluate_actions", "max_dv_mirrored"):
            want(ev.get(k) == 0.0, f"blind {k} {ev.get(k)}")
        want(ev.get("min_buffer_values_vs_raw_value_mlp", 0) > 0, "blind buffer vs raw")
    want(ev["min_dmu_actor"] > 0, "actor sees obs6")
    want(ev["buffer_gae_lambda"] == [0.95] and ev["model_gae_lambda"] == [0.95], "lambda")
    # check.json
    want(c["passed"] is True and c["failures"] == [], "check failed")
    want(c["tool_sha256"] == appr["tool_sha256"]["read_bs.py"], "check tool")
    want(c["name"] == name and c["arm"] == arm and c["seed"] == seed and c["T"] == E and c["stop_reason"] == "end", "check id")
    src = c["sources"]
    want(src["run_meta.json"][1] == sha(os.path.join(rd, "run_meta.json")), "run_meta changed since CHECK")
    want(src["policy_probe.npz"][1] == sha(os.path.join(rd, "01_stance", "policy_probe.npz")), "probe changed since CHECK")
    want(src["arm_evidence.jsonl"][1] == sha(os.path.join(rd, "arm_evidence.jsonl")), "evidence changed since CHECK")
    want(src["parent policy_probe.npz"][1] == sha(src["parent policy_probe.npz"][0]), "parent probe changed")
    pd = c["items"]["probe_detail"]
    want(pd["rows_in_window"] == 175 and pd["ent_coef_mismatches"] == 0 and pd["learning_rate_mismatches"] == 0, "probe detail")
    print(f"{name:26s} start {m['utc_start']} end {m['utc_end']} check {c['utc']} upd {m['digest_start']['n_updates']}->{m['digest_end']['n_updates']}"
          f" class {ae['mlp_extractor_class']} problems {p}")
    problems += [(name, x) for x in p]
print("TOTAL problems", problems)
