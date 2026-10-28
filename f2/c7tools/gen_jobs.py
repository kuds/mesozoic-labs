"""Write the two sequential job lists (two simulation slots) from plan.json: tab-separated run, label, kind, zip, pkl, stage_config."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
plan = json.loads((D / "plan.json").read_text())
m1, m3 = [], []
for run, d in plan.items():
    sc = d["stage_config"]["chosen"]
    for lab, it in d["labels"].items():
        row = [run, lab, "m1", it["zip"]["chosen"], it["pkl"]["chosen"], sc]
        assert all(row), row
        m1.append(row)
        if lab == "final":
            m3.append([run, lab, "m3", it["zip"]["chosen"], it["pkl"]["chosen"], sc])
slot1 = m3[:2]
slot2 = m1 + m3[2:]
for name, rows in (("jobs_slot1.tsv", slot1), ("jobs_slot2.tsv", slot2)):
    (D / name).write_text("".join("\t".join(r) + "\n" for r in rows))
    print(name, len(rows))
