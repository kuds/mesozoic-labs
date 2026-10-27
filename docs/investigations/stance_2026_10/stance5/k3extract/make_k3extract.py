"""Write the compact per-run K3 extract of the five band runs' judge files.

For each run, every ``judge_<steps>.json`` and ``judge_final.json`` (the files the training-history report's judge-K3
readout reads, ``plot_training_history.py --judged``) is trimmed to the keys that readout reads, written with indent 1
under ``<out>/<run id>/`` with its own name, so ``--judged <out>/<run id>`` reads the extract as it reads the judge's
own directory.  ``<out>/k3_extract.csv`` indexes the files: one row per file, with its source's sha256.

Usage: python3 make_k3extract.py <scratch root> <out>
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

#: The five band runs (arm LTSB) and the judge directories the stance5 replication's score of record read.
RUNS = {
    "stance5_LTSB_s52": "foot/stance5/judged/stance5_LTSB_s52",
    "stance5_LTSB_s60": "foot/stance5/judged/stance5_LTSB_s60",
    "stance5_LTSB_s42": "foot/stance5/judged/stance5_LTSB_s42",
    "stance5_LTSB_s70": "foot/stance5/replication/judged/stance5_LTSB_s70",
    "stance5_LTSB_s80": "foot/stance5/replication/judged/stance5_LTSB_s80",
}
JUDGE_FILE = re.compile(r"^judge_(?P<slot>\d+|final)\.json$")
TOP_KEYS = ("schema", "mode", "smoke", "noise_run", "seed", "steps")
PANELS = ("noise_matched_control_joint", "noise_matched_own")
PANEL_KEYS = ("n", "falls", "fall_seeds", "seeds", "sigma", "sigma_vector")
EPISODE_KEYS = ("seed", "length", "reached_horizon")
COLUMNS = (
    "run_id",
    "seed",
    "schema",
    "file",
    "slot",
    "timesteps",
    "joint_falls",
    "joint_n",
    "own_falls",
    "own_n",
    "spawns",
    "source",
    "source_sha256",
)


def trim(record: dict) -> dict:
    """The keys the judge-K3 readout reads, in the source's order, with each panel's episodes cut to three keys."""
    out = {key: record[key] for key in TOP_KEYS if key in record}
    out["checks"] = {"checkpoint_num_timesteps": record["checks"]["checkpoint_num_timesteps"]}
    for panel in PANELS:
        block = record[panel]
        kept = {key: block[key] for key in PANEL_KEYS if key in block}
        kept["episodes_detail"] = [{key: row[key] for key in EPISODE_KEYS} for row in block["episodes_detail"]]
        out[panel] = kept
    return out


def main(root: Path, out: Path) -> int:
    rows = []
    for run_id, relative in RUNS.items():
        source_dir = root / relative
        target = out / run_id
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source_dir.glob("judge_*.json")):
            match = JUDGE_FILE.match(path.name)
            if match is None:
                continue
            data = path.read_bytes()
            record = json.loads(data)
            compact = trim(record)
            (target / path.name).write_text(json.dumps(compact, indent=1) + "\n", encoding="utf-8")
            joint, own = compact["noise_matched_control_joint"], compact["noise_matched_own"]
            slot = match.group("slot")
            steps = compact["checks"]["checkpoint_num_timesteps"] if slot == "final" else int(slot)
            rows.append(
                {
                    "run_id": run_id,
                    "seed": compact["seed"],
                    "schema": compact["schema"],
                    "file": path.name,
                    "slot": slot,
                    "timesteps": steps,
                    "joint_falls": joint["falls"],
                    "joint_n": joint["n"],
                    "own_falls": own["falls"],
                    "own_n": own["n"],
                    "spawns": "-".join(str(seed) for seed in joint["seeds"]),
                    "source": f"{relative}/{path.name}",
                    "source_sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    rows.sort(key=lambda row: (list(RUNS).index(row["run_id"]), row["timesteps"]))
    with open(out / "k3_extract.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} judge files, {len(RUNS)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
