"""One row per stance4 judge file: the counts the criteria read, recomputed by the frozen scorer's panel_counts.

    python3 -I checkpoints_csv.py JUDGED_ROOT OUT.csv [--check SCORE.json] [--scorer SCORE_STANCE3.py]

JUDGED_ROOT/<run id>/judge_<slot>.json are judge_stance4.py's files (schema stance4-judge/v1, arm LTS) for the three
stance4 runs.  Every number below comes from score_stance3.panel_counts on the file's per-episode rows (the function
score_stance4.py loads unchanged from score_stance3.py), or, for a tail or common-noise panel, from the file's own
"tail" or "common_noise" block.  --scorer names the score_stance3.py to load; the default is the committed copy at
../../stance3/judge/score_stance3.py, whose code is the scratch original's apart from its SCRATCH constant.
Report only: no criterion is decided here (score_stance4.py decides K1-K4).  With --check, every cell of the scorer's
--json output that names the same judge file is compared, and any difference stops the script.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCORER = os.path.join(HERE, "..", "..", "stance3", "judge", "score_stance3.py")
JUDGE_SCHEMA = "stance4-judge/v1"
ARM = "LTS"
HEAD = "c1e146a82658112a9a1c0838250020b214ef303a"  # claude/screen-trex-r9-armlt-stagger, the config commit
RUNS = (("stance4_LTS_s52", 52, "Run 1"), ("stance4_LTS_s60", 60, "Run 2"), ("stance4_LTS_s42", 42, "Run 3"))
# (slot in the file name, the score table's label, role)
SLOTS = [
    ("1048576", "1.0M", "report (1,048,576-step)"),
    ("2097152", "2.1M", "report (1,048,576-step)"),
    ("3145728", "3.1M", "report (1,048,576-step)"),
    ("4194304", "4.2M", "J"),
    ("5242880", "5.2M", "J"),
    ("6291456", "6.3M", "J"),
    ("7340032", "7.3M", "report (1,048,576-step)"),
    ("8388608", "8.4M", "J"),
    ("9437184", "9.4M", "report (1,048,576-step)"),
    ("10485760", "10.5M", "J"),
    ("11534336", "11.5M", "report (1,048,576-step)"),
    ("final", "final", "J"),
    ("robust_best", "robust_best", "selection S0"),
    ("best", "best", "selection S1"),
    ("fresh_final", "final, fresh panel", "report (fresh panel 7042-7081)"),
    ("tail_final", "final, tail panel", "report (forward-start tail panel)"),
    ("common_noise_final", "final, common noise", "report (common-noise panel)"),
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
    "common_noise_falls",
    "model_sha256",
]
CHECK_KEYS = (
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


def load_scorer(path: str):
    spec = importlib.util.spec_from_file_location("score_stance3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def r1(v):
    return "" if v is None else f"{v:.1f}"


def row_for(sc, path: str, run: str, seed: int, name: str, label: str, role: str) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    checks = d.get("checks") or {}
    assert d.get("schema") == JUDGE_SCHEMA and d.get("arm") == ARM and d.get("seed") == seed, path
    assert not checks.get("hard_failures"), (path, checks.get("hard_failures"))
    assert (d.get("git") or {}).get("head") == HEAD and not (d.get("git") or {}).get("status"), path
    assert (checks.get("relaxed") or {}).get("task_binding") == "require", path
    out = {
        "run": name,
        "run_id": run,
        "arm": ARM,
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
    if d.get("mode") == "common_noise":
        c = d["common_noise"]
        out.update(panel=f"{c['n']} common-noise seeds", n=c["n"], common_noise_falls=c["falls"])
        return out
    c = sc.panel_counts(d)
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
    out["_file"] = os.path.basename(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("judged_root")
    ap.add_argument("out_csv")
    ap.add_argument("--check", help="score_stance4.py's --json output; every cell naming the same file must agree")
    ap.add_argument("--scorer", default=DEFAULT_SCORER, help="the score_stance3.py whose panel_counts is used")
    a = ap.parse_args()
    sc = load_scorer(a.scorer)
    rows = []
    for run, seed, name in RUNS:
        for slot, label, role in SLOTS:
            path = os.path.join(a.judged_root, run, f"judge_{slot}.json")
            if not os.path.exists(path):
                continue
            rows.append(row_for(sc, path, run, seed, name, label, role))
    if a.check:
        with open(a.check) as fh:
            score = json.load(fh)
        checked = 0
        for r in rows:
            if "_counts" not in r:
                continue
            cell = (score["stagger"][r["run_id"]]["cells"]).get(r["slot"])
            if cell is None or os.path.basename(cell["path"]) != r["_file"]:
                continue
            diff = {k: (r["_counts"][k], cell[k]) for k in CHECK_KEYS if r["_counts"][k] != cell[k]}
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
