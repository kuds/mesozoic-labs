"""Re-derive each certificate's verdict from its grid CSV alone (the publication path's statistic), report only."""
import json, sys
from pathlib import Path
from environments.shared.model_id import ModelId
from environments.shared.config import load_stage_config
from environments.shared.curriculum.stance_gate_v2 import StanceV2Thresholds
from environments.shared.curriculum.stance_heading_gate import (read_stance_heading_panel, statue_references,
                                                                 evaluate_stance_heading_gate, HEADING_GRID_DEG)
O = Path(sys.argv[1])
cur = load_stage_config(ModelId("tyrannosaurus_rex", "anatomical"), 1)["curriculum_kwargs"]
th = StanceV2Thresholds.from_curriculum(cur)
out = {}
for d in sorted(O.iterdir()):
    csv = d / "stance_heading_panel_selected.csv"
    if not csv.exists() or d.name.startswith("smoke"):
        continue
    ev = read_stance_heading_panel(csv)
    seeds = [ev.seed_start + i for i in range(ev.n_seeds)]
    st = statue_references(ev.statue, horizon=ev.horizon, grid=HEADING_GRID_DEG)
    res = evaluate_stance_heading_gate(ev.policy, th, horizon=ev.horizon, seeds=seeds, statue_by_heading=st,
                                       control_dt=ev.control_dt, grid=HEADING_GRID_DEG)
    rep = json.loads((d / "stance_heading_report.json").read_text())
    r = res.as_dict()
    agree = (r["passed"] == rep["passed"] and r["n_clean_seeds"] == rep["result"]["n_clean_seeds"]
             and abs(r["clean_seed_lcb"] - rep["result"]["clean_seed_lcb"]) < 1e-12)
    out[d.name] = {"rederived_passed": r["passed"], "n_clean_seeds": r["n_clean_seeds"], "lcb": r["clean_seed_lcb"],
                   "agrees_with_report": agree, "csv_task_sha256": ev.task_sha256, "csv_checkpoint_sha256": ev.checkpoint_sha256}
    print(d.name, out[d.name])
(O / "rederive_from_csv.json").write_text(json.dumps(out, indent=1))
