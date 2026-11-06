"""Fix lane: the fix stage's own end-to-end checks, beside the e2e lane's copied e2e_summary.py. Reads JSON and text
only (and pilot4_score.py's constants, a standard-library module); imports no repository module; trains nothing.

usage: python -I -B e2e_fix_checks.py LANE TOOLS
Writes LANE/e2e/e2e_fix_checks.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

LANE, TOOLS = Path(sys.argv[1]), Path(sys.argv[2])
E = LANE / "e2e"
OUT = E / "OUT"
LG = E / "logs"
SEEDS = (42, 52, 60)
STEPS = (500_000, 1_000_000, 1_500_000, 2_000_000)
HEAD = "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8"
sys.path.insert(0, str(TOOLS))
import pilot4_score as ps  # noqa: E402  (plain Python, standard library only)

res: dict = {"checks": [], "numbers": {}}


def check(name: str, ok, detail="") -> None:
    res["checks"].append({"name": name, "ok": bool(ok), "detail": detail if isinstance(detail, str) else
                          json.dumps(detail, default=str)})


def j(p: Path):
    return json.loads(p.read_text())


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


st = {}
for ln in (E / "status.txt").read_text().splitlines():
    m = re.match(r"(\S+) rc=(\d+) elapsed_s=(\d+)", ln)
    if m:
        st[m.group(1)] = int(m.group(2))


def log(name: str) -> str:
    p = LG / name
    return p.read_text() if p.is_file() else ""


def refusal(text: str) -> str:
    lines = [ln for ln in text.splitlines() if "refus" in ln.lower()]
    return lines[-1][:700] if lines else text.strip()[-400:]


# ---------------------------------------------------------------- the runner (review_code should-fix 1)
rec = j(E / "runner_not_quick_test" / "record.json")
check("runner, real notebook: a launch with QUICK_TEST false and neither --stop-at-learn nor --plan-only is refused "
      "(exit 2) before any cell (no cell edited or run, no runner.stdout.log)",
      st.get("runner_not_quick_test_expect2") == 2 and st.get("runner_not_quick_test_ran_no_cell") == 0
      and rec.get("exit") == 2 and rec.get("cells") == [] and "QUICK_TEST is False, not true" in rec.get("refused", ""),
      rec.get("refused"))
runner_sha = sha(TOOLS / "nb_pilot_runner4.py")
edge = {s: j(LANE / "setup" / f"s{s}" / "edge_pairs.json") for s in SEEDS}
check("the set-up path ran through the fixed runner (its sha256) with QUICK_TEST false under --stop-at-learn, "
      "which the new rule leaves open",
      all(edge[s]["runner_sha256"] == runner_sha and edge[s]["runner_exit"] == 0
          and edge[s]["runner_record"]["manual_cell_stopped_at_learn"] for s in SEEDS),
      {s: edge[s]["runner_sha256"] for s in SEEDS})

# ---------------------------------------------------------------- the judge (validity should-fix 1 and 2)
bk = {}
for s in SEEDS:
    for k in STEPS:
        b = j(OUT / f"s{s}" / str(k) / "binding.json")
        bk[f"s{s}_{k}"] = [b.get("env_kwargs_drift"), b.get("probe_env_kwargs_drift"), b.get("text_anchors_checked")]
check("binding.json on all 12 cells records an empty env-kwargs drift (the judge's own and the probe's) and that the "
      "text's digests and table were checked", all(v == [{}, {}, True] for v in bk.values()), bk)
check("the judge without --expect-head passes on the shared tree at 0e6a57b (the default is the text's head)",
      st.get("judge_default_head_check_only_expect0") == 0, log("judge_default_head_check_only.log").strip()[-300:])
parent = log("clone_parent_head.txt").strip()
check("the judge without --expect-head refuses a clean clone on the pilot branch at HEAD~1 for its head (exit 2)",
      st.get("judge_default_head_at_parent_expect2") == 2 and parent and parent != HEAD
      and f"tree HEAD {parent} is not {HEAD}" in log("judge_default_head_at_parent.log"),
      refusal(log("judge_default_head_at_parent.log")))
check("my clone was restored to 0e6a57b, clean", st.get("clone_restore_head") == 0)
t = log("tamper_env_drift.log")
check("the validity review's drift case (reward_weights.forward_vel_weight 3.0, task digest as recorded) is refused "
      "before the probe (exit 2), the reason naming the drift",
      st.get("tamper_env_drift_expect2") == 2 and "differ from the env this tree builds" in t
      and '"forward_vel_weight": {"recorded_run": 3.0, "toml_main": 2.0}' in t and "running:" not in t, refusal(t))
check("the fix stage's judge cases wrote nothing", st.get("fix_outputs_absent") == 0)
pod = sorted((LANE / "data" / "probe_on_drift").glob("*_locomotion.json"))
probe_drift = j(pod[0])["meta"]["env_kwargs_drift_vs_recorded"] if pod else None
plog = (LANE / "logs" / "probe_on_drift.log").read_text() if (LANE / "logs" / "probe_on_drift.log").is_file() else ""
check("the frozen probe itself, run on the drifted copy (1 panel episode, logs/probe_on_drift.log), only warns and "
      "records the same drift the judge now refuses on",
      probe_drift == {"forward_vel_weight": {"toml_main": 2.0, "recorded_run": 3.0}}
      and "WARNING: env kwargs drifted" in plog
      and '{"forward_vel_weight": {"recorded_run": 3.0, "toml_main": 2.0}}' in t, probe_drift)

# ---------------------------------------------------------------- the straightness records (pinned tool, constants)
tool_sha = sha(TOOLS / "pilot4_straight.py")
recs = {f"s{s}_{k}": j(E / "straight" / f"s{s}_{k}.json") for s in SEEDS for k in STEPS}
check("every straightness record was written by the pinned tool (STRAIGHT_TOOL_SHA256 = the file's sha256) with "
      "v3's constants, and its verdicts follow from its values",
      tool_sha == ps.STRAIGHT_TOOL_SHA256
      and all(r["tool_sha256"] == tool_sha and r["constants"] == ps.STRAIGHT_CONSTANTS
              and ps.recomputed_verdicts(r["parts"]) == {k: r["parts"][k]["verdict"] for k in ("a", "b", "c")}
              for r in recs.values()), {"tool": tool_sha, "pinned": ps.STRAIGHT_TOOL_SHA256})
crab = {k: [r["parts"]["b"].get("report_crab_median_deg"), r["parts"]["b"].get("report_end_cross_track_median_m")]
        for k, r in recs.items() if k.endswith("_2000000")}
check("each record carries the report-only crab and end cross-track medians (fix stage)",
      all(v[0] is not None and v[1] is not None for v in crab.values()), crab)
res["numbers"]["crab_cross_track_2M"] = crab
t = log("straight_drift_cell.log")
check("the straightness tool refuses a cell whose probe.json records an env-kwargs drift (exit 2, nothing written)",
      st.get("straight_drift_cell_expect2") == 2 and st.get("straight_drift_output_absent") == 0
      and "env-kwargs drift record is not empty" in t, refusal(t))

# ---------------------------------------------------------------- the scorer (validity should-fix 3, drift)
sd = j(E / "score" / "score_drift_cells.json")
probs = sd["seeds"]["42"]["2000000"]["problems"]
check("scorer on the drifted cells: s42's cell is not scored (its problem names the drift), s42 has no gait outcome; "
      "s52 and s60 read CURVED WALKS",
      st.get("score_drift_cells") == 0 and any("env kwargs drift" in p for p in probs)
      and sd["per_seed_primary"]["42"] is None and sd["seed_outcomes"]["52"]["outcome"] == "CURVED WALKS"
      and sd["seed_outcomes"]["60"]["outcome"] == "CURVED WALKS",
      {"s42_problems": probs, "pilot": sd["pilot"]["outcome"]})
sf = j(E / "score" / "score_forged_record.json")
fr = sf["straight"]["42"]["2000000"]
kinds = sorted({p.split(" ")[0] for p in fr["problems"]})
check("scorer with the validity review's forged s42 record (PASS over FAIL values, another tool, looser bars): the "
      "record is refused for its tool, its constants and its verdicts; s42 waits for a record (pending); s52, s60 "
      "CURVED WALKS; no STRAIGHT WALKS anywhere",
      st.get("score_forged_record") == 0 and not fr["read"] and kinds == ["constants", "part", "tool_sha256"]
      and sf["seed_outcomes"]["42"]["outcome"] == "pending"
      and "straightness record" in sf["seed_outcomes"]["42"]["awaited"]
      and sf["existence"]["reading"] != "STRAIGHT WALK EXISTS",
      {"problems": fr["problems"], "s42": sf["seed_outcomes"]["42"]["outcome"], "pilot": sf["pilot"]["outcome"],
       "existence": sf["existence"]["reading"]})
sr = j(E / "score" / "score_review.json")
check("scorer with the genuine records: all twelve read (none refused by the new checks)",
      all(sr["straight"][str(s)][str(k)]["read"] for s in SEEDS for k in STEPS),
      {f"s{s}_{k}": sr["straight"][str(s)][str(k)]["problems"] for s in SEEDS for k in STEPS})
check("the shared tree is clean after phase D", st.get("tree_clean_after_phase_d") == 0)
check("the shared tree is clean after the fix stage's judge cases", st.get("tree_clean_after_phase_a_fix") == 0)
check("the shared tree is clean after phase 0", st.get("tree_clean_after_phase_0") == 0)
bad = [c for c in res["checks"] if not c["ok"]]
res["failed"] = len(bad)
res["failed_names"] = [c["name"] for c in bad]
(E / "e2e_fix_checks.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
for c in res["checks"]:
    print(("ok   " if c["ok"] else "FAIL ") + c["name"])
print(f"{len(res['checks']) - len(bad)}/{len(res['checks'])} checks true")
