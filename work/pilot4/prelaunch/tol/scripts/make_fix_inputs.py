"""Fix lane: inputs for the fix stage's end-to-end refusals (json and shutil only; no repository import; nothing
trained).

usage: python -I -B make_fix_inputs.py LANE

1. LANE/e2e/OUT_drift/: byte copies of the three judged 2,000,000 cells, then s42's probe.json meta
   env_kwargs_drift_vs_recorded set to {"forward_vel_weight": {"toml_main": 2.0, "recorded_run": 3.0}} (a cell as the
   pre-fix judge wrote one for the validity review's drifted run).
2. LANE/e2e/inputs/forged_s42_2000000.json: the s42 2,000,000 straightness record with the validity review's forgery:
   PASS verdicts over its own FAIL values, tool_sha256 all zeros, looser bars in its constants.
Writes LANE/e2e/inputs/make_fix_inputs.json.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

LANE = Path(sys.argv[1])
E = LANE / "e2e"
out = {}


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


for s in (42, 52, 60):
    src, dst = E / "OUT" / f"s{s}" / "2000000", E / "OUT_drift" / f"s{s}" / "2000000"
    shutil.copytree(src, dst)
p = E / "OUT_drift" / "s42" / "2000000" / "probe.json"
doc = json.loads(p.read_text())
doc["meta"]["env_kwargs_drift_vs_recorded"] = {"forward_vel_weight": {"toml_main": 2.0, "recorded_run": 3.0}}
p.write_text(json.dumps(doc))
out["OUT_drift"] = {"s42_probe_json": sha(p), "edited": "meta.env_kwargs_drift_vs_recorded"}
rec = json.loads((E / "straight" / "s42_2000000.json").read_text())
rec["tool_sha256"] = "sha256:" + "0" * 64
rec["constants"].update(A_P95_ABS_MAX_DEG=40.0, B_D_P95_MAX_DEG=30.0, B_W0_P95_MAX_DEG=35.0, C_MEDIAN_MAX_DEG=25.0,
                        C_P95_MAX_DEG=45.0)
for k in ("a", "b", "c"):
    rec["parts"][k]["verdict"] = "PASS"
rec["parts"]["straight"] = "PASS"
(E / "inputs").mkdir(parents=True, exist_ok=True)
fp = E / "inputs" / "forged_s42_2000000.json"
fp.write_text(json.dumps(rec, indent=1))
out["forged"] = {"file": str(fp), "sha256": sha(fp),
                 "values": {"a": [rec["parts"]["a"]["median_signed_deg"], rec["parts"]["a"]["p95_abs_deg"]],
                            "b": [rec["parts"]["b"]["D_p95_deg"], rec["parts"]["b"]["W0_p95_deg"]],
                            "c": [rec["parts"]["c"]["median_r_deg"], rec["parts"]["c"]["p95_r_deg"]]}}
(E / "inputs" / "make_fix_inputs.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
