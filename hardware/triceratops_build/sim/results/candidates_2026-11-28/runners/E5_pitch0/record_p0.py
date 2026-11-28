"""E5 pitch-0 branch: run record and harness checks (E5_PITCH0_PLAN.md items 1, 4, 10, 11), from the scratch logs and
episode files. Writes e5p0_run_record.json and e5p0_harness_checks.json to the staging folder first."""
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e5_pitch0 as R  # noqa: E402

C = R.C
L = R.SCR / "logs"
E5EPI = C.CAND_SCRATCH / "E5" / "episodes"
STAGE = R.SCR / "staging"
STAGE.mkdir(exist_ok=True)
sizes = [s for s in ("0.60", "0.50") if (L / f"heldout_{s}.log").exists()]

rec = {"experiment": "E5 pitch-0 branch (changed after a result)", "date": C.DATE, "stages": [], "workers": 2}
for size in sizes:
    for kind in ("tune", "heldout"):
        pools = []
        for ln in (L / f"{kind}_{size}.log").read_text().splitlines():
            m = re.match(r"\[(\d+)/(\d+) (\d+)s\] \('([\d.]+)', '(\w+)'", ln)
            if m and int(m[1]) == int(m[2]):
                pools.append({"stage": m[5], "episodes": int(m[2]), "wall_s": int(m[3])})
        rec["stages"].append({"size": size, "command": f"run_e5_pitch0.py {kind} --size {size} --workers 2",
                              "pools": pools,
                              "sha256_before": (L / f"sha256_before_{kind}_{size}.txt").read_text().splitlines()})
rec["seeds"] = {"s1": list(C.E5_SCREEN_SEEDS), "s2": list(C.E5_TUNING_SEEDS), "s3": list(C.E5_TUNING_SEEDS),
                "heldout": {k: list(v) for k, v in R.HELD.items()},
                "heldout_frozen": {k: list(R.HELD[k]) for k in R.FROZEN_SETS}}
rec["choice_sha256_at_write"] = {s: (R.SCR / f"e5p0_choice_{s}.sha256").read_text().splitlines() for s in sizes}
rec["choice_sha256_now"] = {s: R.sha(R.OUT / f"e5p0_choice_{s}.json") for s in sizes}
rec["plan_sha256_recorded"] = (R.SCR / "plan_sha256.txt").read_text().splitlines()
rec["plan_sha256_now"] = R.sha(R.PLAN)
rec["provenance"] = C.provenance(Path(__file__), extra=R._extra("0.50"))

hc = {"experiment": "E5 pitch-0 branch (changed after a result)", "evidence": "harness checks (code, not hypotheses)",
      "date": C.DATE}
# 1. setting 00 (the frozen controller, pitch already 0) reproduces E5's stored stage-1 episodes
c1 = {}
for size in sizes:
    rows, nd = [], 0
    for cond in C.E5_SCREEN_CONDITIONS:
        for s in C.E5_SCREEN_SEEDS:
            a = json.loads((E5EPI / size / "s1" / f"H{size}-s1-00__{cond}__{s}.json").read_text())
            b = R.load_ep(size, "s1", f"H{size}-p0-s1-00", cond, s)
            diff = [k for k in ("record", "rpm_peak", "rpm_p99", "slip", "fell", "slow_gait", "worst_hind_knee",
                                "walk_params", "body_pitch_deg")
                    if json.dumps(a[k], sort_keys=True) != json.dumps(b[k], sort_keys=True)]
            nd += len(diff)
            rows.append({"condition": cond, "seed": s, "differing_fields": diff})
    c1[size] = {"episodes": len(rows), "differing_fields_total": nd, "pass": nd == 0, "rows": rows}
hc["check1_setting00_reproduces_E5_stage1"] = c1
# 2. stage 2 reran its shared stage-1 episodes
hc["check2_stage2_vs_stage1"] = {}
for size in sizes:
    d = json.loads((R.OUT / f"e5p0_choice_{size}.json").read_text())["determinism_check_stage2_vs_stage1"]
    hc["check2_stage2_vs_stage1"][size] = {**d, "pass": len(d["mismatching_fields"]) == 0}
# 3. every episode file has body pitch 0 and an allowed seed for its stage
c3 = {}
for size in sizes:
    n, bad = 0, []
    for p in glob.glob(str(R.EPI / size / "*" / "*.json")):
        e = json.loads(Path(p).read_text())
        n += 1
        if e["body_pitch_deg"] != 0.0 or e["seed"] not in R.ALLOWED_SEEDS[e["stage"]]:
            bad.append(Path(p).name)
    c3[size] = {"episode_files": n, "bad": bad, "pass": not bad}
hc["check3_all_pitch_0_and_seeds_allowed"] = c3
# 4. choice files and plan unchanged since written
hc["check4_files_unchanged"] = {
    "choice": {s: {"at_write": rec["choice_sha256_at_write"][s][0].split()[0], "now": rec["choice_sha256_now"][s],
                   "pass": rec["choice_sha256_at_write"][s][0].split()[0] == rec["choice_sha256_now"][s]} for s in sizes},
    "plan": {"recorded": rec["plan_sha256_recorded"][0].split()[0], "now": rec["plan_sha256_now"],
             "pass": rec["plan_sha256_recorded"][0].split()[0] == rec["plan_sha256_now"]}}
# 5. the choice was written before any held-out episode file existed
c5 = {}
for size in sizes:
    ch = json.loads((R.OUT / f"e5p0_choice_{size}.json").read_text())
    c5[size] = {"heldout_episode_files_present_when_written": ch["heldout_episode_files_present_when_written"],
                "pass": ch["heldout_episode_files_present_when_written"] == []}
hc["check5_choice_before_heldout"] = c5
hc["provenance"] = rec["provenance"]
(STAGE / "e5p0_run_record.json").write_text(json.dumps(rec, indent=1) + "\n")
(STAGE / "e5p0_harness_checks.json").write_text(json.dumps(hc, indent=1) + "\n")
print(json.dumps({k: ({s: v[s]["pass"] for s in v} if k != "check4_files_unchanged" else
                      {"choice": {s: v["choice"][s]["pass"] for s in v["choice"]}, "plan": v["plan"]["pass"]})
                  for k, v in hc.items() if k.startswith("check")}))
print(json.dumps(rec["stages"]))
