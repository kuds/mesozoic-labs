"""E5 harness checks, written to the results folder (they test the code, not any hypothesis):
  1. E5's frozen setting at 60 cm (H0.60-s1-00, stage 1) reproduces E1's frozen-trot episodes (same function, model,
     seeds 101-104, conditions J1, J3, push back 0.5, push right 0.5) field for field (E1's record vs E5's record).
  2. Stage 2 rerun of the 20 episodes it shares with stage 1, per setting (from the choice files).
  3. Post-hoc (decided after the 60 cm held-out result was seen; changes no result): the push is applied in the
     held-out push episodes (check_push_applied.py: applied steps, impulse; the rerun equals the stored episode).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e5 as R  # noqa: E402

C = R.C
E1 = C.CAND_SCRATCH / "E1" / "episodes"
out = {"experiment": "E5", "evidence": "harness checks (code, not hypotheses)", "date": C.DATE}
rows, n_diff = [], 0
for cond in ("J1", "J3", "push_back_0.5", "push_right_0.5"):
    for s in (101, 102, 103, 104):
        a = json.loads((E1 / f"trot__{cond}__{s}.json").read_text())["record"]
        b = R.load_ep("0.60", "s1", "H0.60-s1-00", cond, s)["record"]
        diff = sorted(k for k in set(a) | set(b) if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True))
        n_diff += len(diff)
        rows.append({"condition": cond, "seed": s, "differing_fields": diff, "fields": len(b)})
out["check1_E1_reproduced"] = {"episodes": len(rows), "differing_fields_total": n_diff, "rows": rows,
                               "pass": n_diff == 0}
out["check2_stage2_vs_stage1"] = {}
for sk in ("0.60", "0.50"):
    ch = json.loads((R.OUT / f"e5_choice_{sk}.json").read_text())
    d = ch["determinism_check_stage2_vs_stage1"]
    out["check2_stage2_vs_stage1"][sk] = {**d, "pass": len(d["mismatching_fields"]) == 0}
pa = json.loads((R.SCR / "check_push_applied.json").read_text())
out["check3_push_applied_post_hoc"] = {
    "label": "post hoc: decided after the 60 cm held-out result was seen; reruns 4 stored held-out push episodes "
             "into a separate scratch folder and changes no result",
    "rows": [{k: r[k] for k in ("size", "setting", "condition", "seed", "rerun_equals_stored")} |
             {k: r["push"][k] for k in ("applied_steps", "impulse_ns", "force_n", "roll_peak_deg")} for r in pa],
    "pass": all(r["rerun_equals_stored"] and r["push"]["applied_steps"] == 100 for r in pa)}
out["provenance"] = C.provenance(Path(__file__))
(R.OUT / "e5_harness_checks.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: (v.get("pass") if isinstance(v, dict) and "pass" in v else
                      {kk: vv.get("pass") for kk, vv in v.items()} if isinstance(v, dict) else v)
                  for k, v in out.items() if k.startswith("check")}))
print(json.dumps(out["check3_push_applied_post_hoc"]["rows"]))
