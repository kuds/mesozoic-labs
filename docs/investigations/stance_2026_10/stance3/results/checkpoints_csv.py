"""One row per stance3 judge file: the counts the criteria read, recomputed by the frozen scorer's panel_counts.

    python3 -I checkpoints_csv.py JUDGED_ROOT OUT.csv [--check SCORE.json]

JUDGED_ROOT/<run id>/judge_<slot>.json are judge_stance3.py's files.  Every number below comes from
score_stance3.panel_counts (imported unchanged from ../judge/) on the file's per-episode rows, or, for a tail panel,
from the file's own "tail" block.  Report only: no criterion is decided here (score_stance3.py decides K1-K4).
With --check, every J and selection cell is compared with the scorer's own output and any difference stops the script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "judge"))
import score_stance3 as S  # noqa: E402

RUNS = [(run, arm, seed, name) for run, arm, seed, name in S.RUNS]
# (slot in the file name, the score table's label, role)
SLOTS = [
    ("4194304", "4.2M", "J"),
    ("5242880", "5.2M", "J"),
    ("6291456", "6.3M", "J"),
    ("8388608", "8.4M", "J"),
    ("10485760", "10.5M", "J"),
    ("final", "final", "J"),
    ("3145728", "3.1M", "report (judged like J)"),
    ("robust_best", "robust_best", "selection S0"),
    ("best", "best", "selection S1"),
    ("11534336", "11.5M", "selection S3"),
    ("9961472", "9.96M", "report (night control at E)"),
    ("fresh_final", "final, fresh panel", "report (fresh panel 7042-7081)"),
    ("tail_final", "final, tail panel", "report (forward-start tail panel)"),
]
COLUMNS = [
    "run",
    "run_id",
    "arm",
    "seed",
    "slot",
    "role",
    "checkpoint",
    "checkpoint_step",
    "panel",
    "verdict",
    "clean",
    "n",
    "lcb",
    "full_horizon",
    "hop_or_fall",
    "k2a",
    "k2b",
    "xc_median_mm",
    "statue_xc_median_mm",
    "xc_offset_mm",
    "roll_rail",
    "noise_joint_falls",
    "noise_own_falls",
    "tail_falls",
    "statue_tail_falls",
    "model_sha256",
]


def r1(v):
    return "" if v is None else f"{v:.1f}"


def row_for(path: str, run: str, arm: str, seed: int, name: str, slot: str, label: str, role: str) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    checks = d.get("checks") or {}
    assert d.get("schema") == S.JUDGE_SCHEMA and d.get("arm") == arm and d.get("seed") == seed, path
    assert not checks.get("hard_failures"), (path, checks.get("hard_failures"))
    assert (d.get("git") or {}).get("head") == S.HEADS[arm] and not (d.get("git") or {}).get("status"), path
    assert (checks.get("relaxed") or {}).get("task_binding") == "require", path
    out = {
        "run": name,
        "run_id": run,
        "arm": "L+T" if arm == "LT" else arm,
        "seed": seed,
        "slot": label,
        "role": role,
        "checkpoint": d.get("checkpoint"),
        "checkpoint_step": checks.get("checkpoint_num_timesteps"),
        "model_sha256": d.get("model_sha256"),
    }
    if d.get("mode") == "tail":
        t = d["tail"]
        out.update(panel=f"{t['n']} tail seeds", n=t["n"], tail_falls=t["falls"], statue_tail_falls=t["statue_falls"])
        return out
    c = S.panel_counts(d)
    seeds = d["panel"]["seeds"]
    out.update(
        panel=f"{seeds[0]}-{seeds[1]}",
        verdict=c["verdict"],
        clean=c["n_clean"],
        n=c["n"],
        lcb=f"{c['lcb']:.4f}",
        full_horizon=c["full"],
        hop_or_fall=c["hop_or_fall"],
        k2a=c["k2a"],
        k2b=c["k2b"],
        xc_median_mm=r1(c["xc_median_mm"]),
        statue_xc_median_mm=r1(c["statue_xc_median_mm"]),
        xc_offset_mm=r1(c["xc_offset_mm"]),
        roll_rail=c["roll_rail"],
        noise_joint_falls=c["noise_joint_falls"],
        noise_own_falls=c["noise_own_falls"],
    )
    out["_counts"] = c
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("judged_root")
    ap.add_argument("out_csv")
    ap.add_argument("--check", help="the scorer's --json output; J and selection cells must agree")
    a = ap.parse_args()
    rows = []
    for run, arm, seed, name in RUNS:
        for slot, label, role in SLOTS:
            path = os.path.join(a.judged_root, run, f"judge_{slot}.json")
            if not os.path.exists(path):
                continue
            rows.append(row_for(path, run, arm, seed, name, slot, label, role))
    if a.check:
        with open(a.check) as fh:
            score = json.load(fh)
        keys = (
            "n_clean",
            "lcb",
            "verdict",
            "full",
            "hop_or_fall",
            "k2a",
            "k2b",
            "xc_median_mm",
            "statue_xc_median_mm",
            "xc_offset_mm",
            "roll_rail",
            "noise_joint_falls",
            "noise_own_falls",
        )
        checked = 0
        for r in rows:
            cells = score["runs"][r["run_id"]]["cells"]
            label = {"11.5M": "S3"}.get(r["slot"], r["slot"])
            if "_counts" not in r or label not in cells or cells[label] is None:
                continue
            cell = cells[label]
            if os.path.basename(cell["path"]) != f"judge_{dict((lab, s) for s, lab, _ in SLOTS)[r['slot']]}.json":
                continue
            diff = {k: (r["_counts"][k], cell[k]) for k in keys if r["_counts"][k] != cell[k]}
            assert not diff, (r["run_id"], r["slot"], diff)
            checked += 1
        print(f"{checked} cells agree with {os.path.basename(a.check)}")
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
