"""e2e lane: read every output of phases A-C and write e2e/e2e_summary.json, a list of named checks (ok, detail) in
pilot 3's e2e style, plus the numbers they rest on. Reads JSON, text and npz only (np.load, allow_pickle False);
imports no repository module; trains nothing.

usage: python -I -B e2e_summary.py LANE
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

LANE = Path(sys.argv[1])
E = LANE / "e2e"
OUT = E / "OUT"
LG = E / "logs"
TOOLS = LANE.parents[1] / "tools"  # tolerance lane: LANE is pilot4/prelaunch/tol
STRAIGHT_VAL = LANE / "val_final"
READOUTS = TOOLS / "out" / "readouts" / "data"
SEEDS = (42, 52, 60)
STEPS = (500_000, 1_000_000, 1_500_000, 2_000_000)
HEAD = "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8"
PARENT = {42: "sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4",
          52: "sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00",
          60: "sha256:2c066770802541e47fb453b8dbd04b030bc8053d4224af27a52eb0281e4645a2"}
TASK = "sha256:914a1caee2c086a09ef66e9e6d4ad234dd085b69c4da687cc29af82b110fcd7a"
HP = "sha256:fbc94bef69a91d5a65f1df99adb9c35e7897a259affaa3e7b7ba4477418eaa62"
res: dict = {"checks": [], "numbers": {}}


def check(name: str, ok, detail="") -> None:
    res["checks"].append({"name": name, "ok": bool(ok), "detail": detail if isinstance(detail, str) else
                          json.dumps(detail, default=str)})


def status() -> dict:
    out = {}
    for ln in (E / "status.txt").read_text().splitlines():
        m = re.match(r"(\S+) rc=(\d+) elapsed_s=(\d+)", ln)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def j(p: Path):
    return json.loads(p.read_text())


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def eq(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return a == b


st = status()
res["status"] = {k: {"rc": v[0], "elapsed_s": v[1]} for k, v in st.items()}

# ---------------------------------------------------------------- the set-up path and the stand-in runs
setup = {s: j(LANE / "setup" / f"s{s}" / "edge_pairs.json") for s in SEEDS}
for s in SEEDS:
    d = setup[s]
    rr = d["runner_record"]
    check(f"s{s} set-up path: the notebook's manual cell reached learn() (stopped there, nothing trained), no guard hit",
          d["runner_exit"] == 0 and rr["manual_cell_stopped_at_learn"] and rr["guard_hits"] == []
          and d["guard_hits_final"] == [] and rr["run_all"]["stopped_by_the_guard"]
          and rr["entry_lines"]["ok"] and rr["clone_head"] == HEAD,
          {"learn_called": rr["learn_called"], "entry_lines_missing": rr["entry_lines"]["missing"]})
    lin = d["lineage_in_memory"]
    check(f"s{s} edge: lineage initialize_next_stage, parent stage 2, command columns zeroed (in memory and in each "
          "saved zip)", lin.get("mode") == "initialize_next_stage" and lin.get("parent_stage") == 2
          and lin.get("zeroed_command_parameters") and all(eq(v["data_lineage"], lin) for v in d["saved"].values()),
          lin.get("zeroed_command_parameters"))
    rl = d["reloaded_2M_vs_original_8M_on_zeroed_obs"]
    check(f"s{s} edge: the saved 2,000,000 pair acts bit-identically to pilot 3's 8M pair on command-zeroed "
          "observations (1,000 steps) and to the in-memory model",
          rl["max_abs_action_delta"] == 0.0 and rl["max_abs_action_delta_vs_in_memory"] == 0.0 and rl["steps"] == 1000,
          rl)
wf = j(LANE / "setup" / "weights_facts.json")
check("edge weights (saved 2,000,000 zip against the parent's 8M zip): only columns 60-62 of the policy's and the "
      "value's first layers changed (1,536 entries each, now 0); log_std bitwise; Adam moments unchanged and step "
      "counts kept; task 782d95b6... -> 914a1cae...",
      all(set(wf[str(s)]["policy_tensors_changed"]) == {"mlp_extractor.policy_net.0.weight",
                                                       "mlp_extractor.value_net.0.weight"}
          and all(v["n_changed"] == 1536 and v["columns"] == [60, 61, 62] and v["new_values_max_abs"] == 0.0
                  for v in wf[str(s)]["policy_tensors_changed"].values())
          and wf[str(s)]["log_std_bit_equal"] and wf[str(s)]["adam_moments_changed"] == {}
          and all(a == b for a, b in wf[str(s)]["adam_step_pairs_parent_new"])
          and wf[str(s)]["task_sha256_parent_new"] == ["sha256:782d95b607d0086d0cefe475c9caf8ac420c1f2b74d23b786808aaee98232a6a",
                                                       TASK] for s in SEEDS),
      {s: {"adam_steps": wf[str(s)]["adam_step_pairs_parent_new"], "log_std_mean": wf[str(s)]["log_std_mean"],
           "ent_coef": wf[str(s)]["ent_coef_parent_new"]} for s in SEEDS})
sf = j(LANE / "setup" / "sidecar_facts.json")
check("edge sidecars: obs_rms columns 0-59 bit-equal to the parent's, the command slice 60-62 reseeded to mean 0 / "
      "var 1 (parent var ~3e-12), ret_rms fresh (mean 0, var 1, count 1e-4)",
      all(sf[str(s)]["obs_0_60_mean_bit_equal"] and sf[str(s)]["obs_0_60_var_bit_equal"]
          and sf[str(s)]["obs_60_63_mean"][0] == [0.0, 0.0, 0.0] and sf[str(s)]["obs_60_63_var"][0] == [1.0, 1.0, 1.0]
          and sf[str(s)]["ret_rms"][0] == [0.0, 1.0, 0.0001] for s in SEEDS),
      {s: {"var_60_63": sf[str(s)]["obs_60_63_var"], "ret_rms": sf[str(s)]["ret_rms"]} for s in SEEDS})
cmp = j(LANE / "setup" / "records_vs_setup_check.json")
diffs = {k: sorted(v["differing"]) for k, v in cmp.items() if isinstance(v, dict)}
check("the set-up path's records at 0e6a57b equal the setup check's at 0bcb0cf but for git_commit, run.label and "
      "run.load_path (plant_identity.json and task_fingerprint.json equal)",
      all(v == [] for k, v in diffs.items() if not k.endswith("stage_config.json"))
      and all(v == ["git_commit", "run.label", "run.load_path"] for k, v in diffs.items()
              if k.endswith("stage_config.json")), diffs)
res["numbers"]["records_vs_setup_check"] = diffs

# ---------------------------------------------------------------- the judge
check("judge --check-only: exit 0 on all 12 cells", all(st.get(f"judge_check_s{s}_{k}", (9,))[0] == 0
                                                     for s in SEEDS for k in STEPS))
check("judge --check-only wrote nothing (OUT absent afterwards)", st.get("out_absent_after_check_only", (9,))[0] == 0)
check("judge: exit 0 on all 12 cells", all(st.get(f"judge_s{s}_{k}", (9,))[0] == 0 for s in SEEDS for k in STEPS),
      {f"s{s}_{k}": st.get(f"judge_s{s}_{k}") for s in SEEDS for k in STEPS})
probes, bindings = {}, {}
for s in SEEDS:
    for k in STEPS:
        cell = OUT / f"s{s}" / str(k)
        probes[(s, k)] = j(cell / "probe.json")
        bindings[(s, k)] = j(cell / "binding.json")
b_ok = []
for (s, k), b in bindings.items():
    eb = b.get("env_binding", {})
    b_ok.append(b.get("schema") == "walking-pilot4-judge/v1" and b.get("seed") == s and b.get("steps") == k
                and b.get("head") == HEAD and b.get("final_fallback") is False and b.get("off_schedule") is False
                and b.get("validation_only") is False and b.get("cadence_checked") is True
                and b.get("task_sha256") == TASK and b.get("hyperparameters_sha256") == HP
                and b.get("parent_checkpoint_sha256") == PARENT[s] and b.get("n_envs") == 4
                and b.get("lineage", {}).get("parent_stage") == 2
                and bool(b.get("lineage", {}).get("zeroed_command_parameters"))
                and eb.get("command_mode") == "heading" and eb.get("prey_lateral_range") == [0.0, 0.0]
                and eb.get("walk_heading_price_weight") == 0.2 and eb.get("walk_heading_price_scale_rad") == 0.5
                and eb.get("command_config") == b.get("committed_command_config")
                and b.get("checkpoint_sha256") == sha(Path(b["checkpoint"]))
                and b.get("vecnorm_sha256") == sha(Path(b["vecnorm"])))
check("binding.json on all 12 cells: schema v1, seed, steps, head, no fallback, on schedule, not validation-only, "
      "cadence checked, task 914a1cae, recipe fbc94bef, the seed's parent, n_envs 4, the lineage, the heading "
      "command with the committed table, prey [0, 0], price 0.2 / 0.5, pair sha256s", all(b_ok), b_ok)
p_ok, labels = [], {}
for (s, k), p in probes.items():
    m = p["meta"]
    p_ok.append(len(p["episodes"]) == 40 and m.get("seed_scheme") == "panel" and m.get("seed_base") == 3042
                and m.get("settle_steps_gait_window") == 100 and m.get("env_kwargs_drift_vs_recorded") == {}
                and m.get("plant_identity_policy_interface_revision") == 15
                and m.get("checkpoint_sha256") == bindings[(s, k)]["checkpoint_sha256"])
check("probe.json on all 12 cells: 40 panel episodes from 3042, settle 100, no env-kwargs drift from the record, "
      "interface r15, the judged checkpoint", all(p_ok), p_ok)
res["numbers"]["leg_length_m"] = sorted({p["meta"].get("robot", {}).get("leg_length_m") for p in probes.values()})
same_eps = {}
for s in SEEDS:
    ref = probes[(s, 2_000_000)]["episodes"]
    same_eps[s] = all(eq(probes[(s, k)]["episodes"], ref) for k in STEPS)
check("the probe's 40 episode rows are identical across a seed's four stand-in cells (one set of weights)",
      all(same_eps.values()), same_eps)
check("a second judgement of a cell is refused (exit 2)", st.get("judge_again_s42_2000000_expect2", (9,))[0] == 2,
      (LG / "judge_again_s42_2000000.log").read_text().strip()[-300:])

# ---------------------------------------------------------------- tampered and dirty cases
tamper = {
    "tamper_wrong_parent_cli_expect2": ("tamper_wrong_parent_cli.log", "parent"),
    "tamper_parent_record_expect2": ("tamper_parent_record.log", "parent"),
    "tamper_command_table_expect2": ("tamper_command_table.log", "command"),
    "tamper_raw_parent_expect2": ("tamper_raw_parent.log", "lineage"),
    "setup_check_record_verbatim_expect2": ("setup_check_record_verbatim.log", "git_commit"),
    "dirty_judge_expect2": ("dirty_judge.log", "edit"),
    "dirty_straight_expect2": ("dirty_straight.log", "edit"),
    "dirty_habit_expect2": ("dirty_habit.log", "edit"),
}
for name, (log, word) in tamper.items():
    text = (LG / log).read_text()
    refusal = [ln for ln in text.splitlines() if "refus" in ln.lower()]
    check(f"{name.replace('_expect2', '')}: refused, exit 2, the reason names '{word}'",
          st.get(name, (9,))[0] == 2 and any(word in ln.lower() for ln in refusal),
          refusal[-1][:600] if refusal else text[-400:])
    res["numbers"].setdefault("refusals", {})[name] = refusal
check("every tampered and dirty case wrote nothing",
      all(st.get(f"tamper_out_absent_{c}", (9,))[0] == 0 for c in
          ("wrong_parent_cli", "parent_record", "command_table", "raw_parent", "setup_check_record"))
      and st.get("dirty_outputs_absent", (9,))[0] == 0)
check("the setup check's own 0bcb0cf record passes every other check under --allow-commit-mismatch --check-only",
      st.get("setup_check_record_allow_mismatch_check_only_expect0", (9,))[0] == 0,
      (LG / "setup_check_record_allow_mismatch.log").read_text().strip()[-400:])
check("the dirty clone was restored clean", st.get("dirty_restore", (9,))[0] == 0)
mf = j(LANE / "tampered" / "make_final.json")
check("final-model fallback (s42 copy with no 2,000,000 pair, a final pair re-saved by SB3 at num_timesteps 2,015,232, "
      "lineage kept): --final-fallback --check-only accepts it, exit 0",
      st.get("final_fallback_check_only_expect0", (9,))[0] == 0 and mf["final_fallback"]["final_num_timesteps"] == 2_015_232
      and bool(mf["final_fallback"]["lineage"]), (LG / "final_fallback_check_only.log").read_text().strip()[-300:])
check("the same copy without --final-fallback is refused (the 2,000,000 pair is missing), exit 2",
      st.get("final_fallback_without_flag_expect2", (9,))[0] == 2,
      (LG / "final_fallback_without_flag.log").read_text().strip()[-300:])
check("a final zip at num_timesteps 2,000,000 (not the uninterrupted end) is refused under --final-fallback, exit 2",
      st.get("final_fallback_short_expect2", (9,))[0] == 2
      and "2,015,232" in (LG / "final_fallback_short.log").read_text(),
      (LG / "final_fallback_short.log").read_text().strip()[-300:])
check("the final-fallback checks wrote nothing", st.get("final_fallback_out_absent", (9,))[0] == 0)

# ---------------------------------------------------------------- the straightness tool
recs = {}
for s in SEEDS:
    for k in STEPS:
        p = E / "straight" / f"s{s}_{k}.json"
        recs[(s, k)] = j(p) if p.is_file() else None
check("straight seed mode: a record for all 12 cells, exit 0",
      all(recs[x] is not None for x in recs) and all(st.get(f"straight_s{s}_{k}", (9,))[0] == 0
                                                     for s in SEEDS for k in STEPS),
      {f"s{s}_{k}": st.get(f"straight_s{s}_{k}") for s in SEEDS for k in STEPS})
v_ok, verdicts = [], {}
for (s, k), r in recs.items():
    if r is None:
        v_ok.append(False)
        continue
    cell = OUT / f"s{s}" / str(k)
    pa = r["parts"]
    verdicts[f"s{s}_{k}"] = [pa["a"]["verdict"], pa["b"]["verdict"], pa["c"]["verdict"], pa["straight"]]
    v_ok.append(r["mode"] == "seed" and r["validation"] is False and r["partial"] is False
                and r["panel"]["all_rolled_match"] is True and r["binding_sha256"] == sha(cell / "binding.json")
                and r["probe_json_sha256"] == sha(cell / "probe.json")
                and r["checkpoint_sha256"] == bindings[(s, k)]["checkpoint_sha256"]
                and r["panel"]["yaw_repro_max_abs_diff_deg"] <= 1e-12)
check("each straightness record is the record of its cell (binding and probe sha256, checkpoint), seed mode, not "
      "validation, not partial, every re-rolled episode equal to probe.json's, yaw reproduced to 1e-12", all(v_ok),
      v_ok)
check("(a) FAIL, (b) FAIL, (c) FAIL, STRAIGHT FAIL on every cell (v3's entry baseline: all fail)",
      all(v == ["FAIL", "FAIL", "FAIL", "FAIL"] for v in verdicts.values()), verdicts)
res["numbers"]["straight_verdicts"] = verdicts
# against the straightness lane's baseline-mode records (the same weights, prepared in memory)
cmp_rows = {}
for s in SEEDS:
    r, base = recs[(s, 2_000_000)], j(STRAIGHT_VAL / f"baseline_s{s}.json")
    pa, pb = r["parts"], base["parts"]
    same = {part: {kk: eq(pa[part].get(kk), pb[part].get(kk)) for kk in pb[part] if kk not in ("verdict",)}
            for part in ("a", "b", "c")}
    eps_same = all(eq({kk: e.get(kk) for kk in ("reset_seed", "T", "yaw_change_deg", "D_deg", "W0_deg", "R_deg")},
                      {kk: f.get(kk) for kk in ("reset_seed", "T", "yaw_change_deg", "D_deg", "W0_deg", "R_deg")})
                   for e, f in zip(r["panel"]["episodes"], base["panel"]["episodes"]))
    rec_a = {(e["reset_seed"], e["angle_deg"]): e.get("r_deg") for e in r["recovery"]["episodes"]}
    rec_b = {(e["reset_seed"], e["angle_deg"]): e.get("r_deg") for e in base["recovery"]["episodes"]}
    cmp_rows[s] = {"parts_equal": {p: all(v.values()) for p, v in same.items()},
                   "parts_unequal_keys": {p: [kk for kk, v in vv.items() if not v] for p, vv in same.items()},
                   "panel_rows_equal": eps_same and len(r["panel"]["episodes"]) == len(base["panel"]["episodes"]),
                   "recovery_r_equal": eq(rec_a, rec_b) if rec_a.keys() == rec_b.keys() else False,
                   "a": [pa["a"]["median_signed_deg"], pa["a"]["p95_abs_deg"]],
                   "b": [pa["b"]["D_p95_deg"], pa["b"]["W0_p95_deg"]],
                   "c": [pa["c"]["n_counting"], pa["c"]["median_r_deg"], pa["c"]["p95_r_deg"],
                         pa["c"]["within_11_5"], pa["c"]["within_5"]]}
check("seed mode on the stand-in 2,000,000 cells equals the straightness lane's baseline mode (v3's entry baseline) "
      "on (a), (b), (c), every panel row's yaw, D, W0, R, and every recovery episode's r",
      all(all(v["parts_equal"].values()) and v["panel_rows_equal"] and v["recovery_r_equal"]
          for v in cmp_rows.values()), cmp_rows)
res["numbers"]["straight_vs_baseline"] = cmp_rows
same_traj = {s: all(eq(recs[(s, k)]["parts"], recs[(s, 2_000_000)]["parts"]) for k in STEPS) for s in SEEDS}
check("the four checkpoints of a seed give the same parts (one set of weights)", all(same_traj.values()), same_traj)

# ---------------------------------------------------------------- the scorer
sc = {n: j(E / "score" / f"{n}.json") for n in ("score_no_review_no_look", "score_review", "score_review_look_straight",
                                              "score_review_look_not_straight", "score_review_look_straight_habit",
                                              "score_review_no_straight")}
for n in sc:
    job = "score_without_straight_records" if n == "score_review_no_straight" else n
    check(f"{n}: exit 0", st.get(job, (9,))[0] == 0)
lab = {}
for s in SEEDS:
    cell = sc["score_review"]["seeds"][str(s)]["2000000"]
    lab[s] = {"counts": cell.get("counts"), "outcome": cell.get("outcome")}
check("pilot 3's rule at 2,000,000: WALKS on every seed (s42 WALK 40; s52 WALK 39 + FALL 1; s60 WALK 40)",
      all(lab[s]["outcome"] == "WALKS" for s in SEEDS)
      and lab[42]["counts"].get("WALK") == 40 and lab[52]["counts"].get("WALK") == 39
      and lab[52]["counts"].get("FALL") == 1 and lab[60]["counts"].get("WALK") == 40, lab)
res["numbers"]["labels_2M"] = lab


def outcomes(d):
    return {s: (d["seed_outcomes"][str(s)]["outcome"], d["seed_outcomes"][str(s)].get("label")) for s in SEEDS}


o = {n: outcomes(d) for n, d in sc.items()}
pil = {n: d["pilot"]["outcome"] for n, d in sc.items()}
res["numbers"]["seed_outcomes"] = o
res["numbers"]["pilot_outcomes"] = pil
res["numbers"]["gait_reading"] = {n: d.get("gait_reading") for n, d in sc.items()}
res["numbers"]["existence"] = {n: d.get("existence", {}).get("reading") for n, d in sc.items()}
res["numbers"]["report_only_count"] = {n: d.get("report_only_count", {}).get("k") for n, d in sc.items()}
check("no video review: every seed pending, the pilot outcome PENDING_REVIEW",
      all(v[0] == "pending" for v in o["score_no_review_no_look"].values())
      and pil["score_no_review_no_look"] == "PENDING_REVIEW", [o["score_no_review_no_look"], pil["score_no_review_no_look"]])
for n in ("score_review", "score_review_look_straight", "score_review_look_not_straight",
          "score_review_look_straight_habit"):
    check(f"{n}: CURVED WALKS on every seed, pilot outcome CURVED WALKS, gait reading WALKS, existence NONE SEEN",
          all(v[0] == "CURVED WALKS" for v in o[n].values()) and pil[n] == "CURVED WALKS"
          and sc[n]["gait_reading"]["by_rule"]["outcome"] == "WALKS"
          and sc[n]["existence"]["reading"] == "NONE SEEN", [o[n], pil[n], sc[n].get("gait_reading", {}).get("by_rule")])
check("no look is required ((a) and (b) fail), and the look 'straight' cannot rescue a failed part",
      all(sc["score_review_look_straight"]["straightness_look_required"].get(str(s)) is None for s in SEEDS)
      and o["score_review_look_straight"] == o["score_review"],
      sc["score_review_look_straight"].get("straightness_look_required"))
check("without the straightness records the video-agreed WALKS seeds are pending (awaiting the record)",
      all(v[0] == "pending" for v in o["score_review_no_straight"].values()), o["score_review_no_straight"])

# ---------------------------------------------------------------- habit, noise, mirror, mech, video
hb = {}
for s in SEEDS:
    p = E / "habit" / f"habit_seed_s{s}.json"
    hb[s] = j(p) if p.is_file() else None
check("habit seed mode on every 2,000,000 cell: exit 0, a report-only record naming the cell's checkpoint, 30 "
      "evaluations in the window",
      all(st.get(f"habit_seed_s{s}", (9,))[0] == 0 and hb[s] is not None and hb[s].get("report_only") is True
          and hb[s]["deciding_checkpoint"]["checkpoint_sha256"] == bindings[(s, 2_000_000)]["checkpoint_sha256"]
          and hb[s]["ledger_reading"]["window"]["evaluations"] == 30 for s in SEEDS),
      {s: (hb[s] or {}).get("reading") for s in SEEDS})
res["numbers"]["habit"] = {s: {"reading": (hb[s] or {}).get("reading"),
                               "trace_full_horizon": ((hb[s] or {}).get("trace") or {}).get("full_horizon")}
                           for s in SEEDS}
check("habit ledger mode on every stand-in ledger: exit 0", all(st.get(f"habit_ledger_s{s}", (9,))[0] == 0
                                                                 for s in SEEDS))
nz = {s: j(E / "noise" / f"noise_s{s}.json") for s in SEEDS}
check("noise readout on every run: exit 0", all(st.get(f"noise_s{s}", (9,))[0] == 0 for s in SEEDS))
noise_rows = {}
for s in SEEDS:
    ck, ps = nz[s]["checkpoints"], (nz[s]["references"].get("parent_std") or {}).get(str(s))
    per = ck.get("periodic", []) if isinstance(ck, dict) else []
    noise_rows[s] = {"steps": [r.get("steps") for r in per],
                     "equal_parent": [r.get("mean") == ps["mean"] and r.get("min") == ps["min"] and r.get("max") == ps["max"]
                                      for r in per] if isinstance(ps, dict) else None,
                     "mean": [r.get("mean") for r in per], "parent": ps,
                     "tensorboard": nz[s]["tensorboard"].get("not_run", "read"),
                     "forward": nz[s]["forward"].get("not_run", "read"), "seed_from_run_id": nz[s]["references"].get("seed")}
check("noise: exp(log_std) at the four stand-in checkpoints equals the seed's parent (the edge keeps log_std); the "
      "TensorBoard and forward parts read 'not run' (no event file, no gate_progress.npz in a stand-in run)",
      all(v["steps"] == list(STEPS) and v["equal_parent"] and all(v["equal_parent"]) and v["seed_from_run_id"] == s
          and v["tensorboard"] != "read" and v["forward"] != "read" for s, v in noise_rows.items()), noise_rows)
res["numbers"]["noise"] = noise_rows
for nm, key in (("mirror", "mirror"), ("mech", "mech")):
    check(f"{nm} on every 2,000,000 cell: exit 0", all(st.get(f"{key}_s{s}_2000000", (9,))[0] == 0 for s in SEEDS),
          {s: st.get(f"{key}_s{s}_2000000") for s in SEEDS})
mir_cmp, mech_cmp = {}, {}
for s in SEEDS:
    pm = E / "readouts" / f"mirror_s{s}_2000000.json"
    pp = READOUTS / "final" / "entry_p4" / f"mirror_parent_s{s}.json"
    if pm.is_file() and pp.is_file():
        a, b = j(pm), j(pp)
        mir_cmp[s] = {"summary_equal": eq(a["summary"], b["summary"]),
                      "mirror_kl_equal": eq(a["mirror_kl"].get("own_statistics"), b["mirror_kl"].get("own_statistics"))
                      and eq(a["mirror_kl"].get("symmetrised_statistics"), b["mirror_kl"].get("symmetrised_statistics")),
                      "actuators_equal": eq(a["actuator_mean_pi_minus_pim_steps_from_100"],
                                            b["actuator_mean_pi_minus_pim_steps_from_100"]),
                      "handedness_deg": a["summary"].get("handedness_deg"),
                      "all_pi_episodes_match_probe": a.get("all_pi_episodes_match_probe")}
    pm = E / "readouts" / f"mech_s{s}_2000000.json"
    pp = READOUTS / "entry_p4" / f"mech_parent_s{s}.json"
    if pm.is_file() and pp.is_file():
        a, b = j(pm), j(pp)
        mech_cmp[s] = {"aggregate_equal": eq(a["aggregate_full_length"], b["aggregate_full_length"]),
                       "leg_power_equal": eq(a["leg_power"], b["leg_power"]),
                       "all_episodes_match_probe": a.get("all_episodes_match_probe"),
                       "ledger_asymmetry": {kk: a["ledger_asymmetry"].get(kk) for kk in
                                            ("read", "from_timesteps", "to_timesteps", "max_over_min")}}
check("mirror on the cells equals the readouts lane's --parent records (handedness, flips, mirror-KL, actuators), "
      "every pi episode equal to probe.json's",
      len(mir_cmp) == 3 and all(v["summary_equal"] and v["mirror_kl_equal"] and v["actuators_equal"]
                                and v["all_pi_episodes_match_probe"] for v in mir_cmp.values()), mir_cmp)
check("mech on the cells equals the readouts lane's --parent records (aggregates, leg power), every episode equal "
      "to probe.json's", len(mech_cmp) == 3 and all(v["aggregate_equal"] and v["leg_power_equal"]
                                                    and v["all_episodes_match_probe"] for v in mech_cmp.values()),
      mech_cmp)
res["numbers"]["mirror_vs_parent"] = mir_cmp
res["numbers"]["mech_vs_parent"] = mech_cmp
vlog = (LG / "video_s42_2000000.log").read_text()
res["not_run"] = {"video_render": {
    "why": "MUJOCO_GL=disable is binding in this lane: mujoco.Renderer has no OpenGL context, so no frame is rendered "
           "and no episode is re-rolled for video (pilot 3's e2e rendered under xvfb-run with MUJOCO_GL=glfw)",
    "rc": st.get("video_s42_2000000_from_score"), "refused_before_renderer": "refusing" in vlog,
    "last_line": vlog.strip().splitlines()[-1][:400] if vlog.strip() else ""}}
check("video tool: its cell checks pass (no refusal) and it reads the scorer's listing; it stops at the renderer "
      "(MUJOCO_GL=disable), so the render path is NOT RUN here", "refusing" not in vlog
      and st.get("video_s42_2000000_from_score", (0,))[0] != 0 and "Renderer" in vlog,
      res["not_run"]["video_render"]["last_line"])
for ph in ("a", "b", "c"):
    check(f"the shared tree is clean after phase {ph.upper()}", st.get(f"tree_clean_after_phase_{ph}", (9,))[0] == 0)
bad = [c for c in res["checks"] if not c["ok"]]
res["failed"] = len(bad)
res["failed_names"] = [c["name"] for c in bad]
(E / "e2e_summary.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
for c in res["checks"]:
    print(("ok   " if c["ok"] else "FAIL ") + c["name"])
print(f"{len(res['checks']) - len(bad)}/{len(res['checks'])} checks true")
