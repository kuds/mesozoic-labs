"""Build the e2e stand-in pilot-4 run folders and the tampered copies (json, csv, shutil only; nothing unpickled, no
repository import, nothing trained). Scratch helper of the e2e lane.

usage: python -I -B make_runs.py LANE SETUP_CHECK_FAKE_RUNS PILOT3_RUNS

LANE/runs/pilot4_walk_s<seed>/03_locomotion/
  stage_config.json, plant_identity.json, task_fingerprint.json  the set-up path's own records at 0e6a57b
                       (LANE/setup/s<seed>/stage, written by train() before learn(); byte copies)
  models/stage2_<steps>_steps.zip, stage2_vecnormalize_<steps>_steps.pkl for 500,000 / 1,000,000 / 1,500,000 /
                       2,000,000: the edge-prepared model saved by SB3's CheckpointCallback (LANE/setup/s<seed>/models;
                       byte copies; the same weights at every step)
  eval_episodes.csv    a STAND-IN ledger: the seed's own pilot-3 8M run's last 30 evaluations (rows of evaluations
                       31-60, file order kept), renumbered 0-29 and restamped at 65,536 x (k + 1); every other column
                       verbatim (pilot 3's env, pilot 3's training: a pipeline stand-in, not a pilot-4 ledger)
LANE/tampered/<case>/pilot4_walk_s42/03_locomotion/  copies of s42's run with the 2,000,000 pair only:
  parent_record        stage_config.json run.parent_checkpoint_sha256 set to s52's parent zip sha256
  command_table        stage_config.json reward_weights.command_config.yaw_gain 1.5 -> 2.0
  raw_parent           the pair replaced by pilot 3's s42 8M pair byte for byte (lineage: parent stage 1, nothing zeroed)
  setup_check_record   the stage records replaced by the setup check's own at 0bcb0cf (build/check/judge/fake_runs),
                       verbatim (git_commit 0bcb0cf)
Prints and writes LANE/runs/make_runs.json (sha256 of every file written, and what each tampered copy changed).
"""

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

LANE, FAKE, P3RUNS = (Path(a) for a in sys.argv[1:4])
STEPS = (500_000, 1_000_000, 1_500_000, 2_000_000)
P52_ZIP = "sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00"
RECORDS = ("stage_config.json", "plant_identity.json", "task_fingerprint.json")
out: dict = {"runs": {}, "tampered": {}}


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def standin_ledger(src: Path, dst: Path) -> dict:
    with open(src, newline="") as fh:
        rd = csv.DictReader(fh)
        fields = rd.fieldnames
        rows = list(rd)
    evals = sorted({int(r["evaluation"]) for r in rows})
    keep = evals[-30:]
    remap = {e: i for i, e in enumerate(keep)}
    n = 0
    with open(dst, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            e = int(r["evaluation"])
            if e not in remap:
                continue
            r = dict(r, evaluation=str(remap[e]), timesteps=str(65_536 * (remap[e] + 1)))
            wr.writerow(r)
            n += 1
    src_steps = sorted({int(r["timesteps"]) for r in rows if int(r["evaluation"]) in remap})
    return {"source": str(src), "source_sha256": sha(src), "source_evaluations_kept": [keep[0], keep[-1]],
            "source_timesteps_kept": [src_steps[0], src_steps[-1]], "rows_written": n,
            "restamped_timesteps": [65_536, 65_536 * 30]}


for seed in (42, 52, 60):
    st = LANE / "runs" / f"pilot4_walk_s{seed}" / "03_locomotion"
    (st / "models").mkdir(parents=True, exist_ok=False)
    for name in RECORDS:
        shutil.copy2(LANE / "setup" / f"s{seed}" / "stage" / name, st / name)
    for steps in STEPS:
        for name in (f"stage2_{steps}_steps.zip", f"stage2_vecnormalize_{steps}_steps.pkl"):
            shutil.copy2(LANE / "setup" / f"s{seed}" / "models" / name, st / "models" / name)
    led = standin_ledger(P3RUNS / f"pilot3_walk_s{seed}" / "03_locomotion" / "eval_episodes.csv",
                         st / "eval_episodes.csv")
    out["runs"][seed] = {"ledger": led,
                         "files": {str(p.relative_to(st)): sha(p) for p in sorted(st.rglob("*")) if p.is_file()}}

src = LANE / "runs" / "pilot4_walk_s42" / "03_locomotion"


def copy_case(case: str, *, pair_from: Path | None = None, records_from: Path | None = None) -> Path:
    st = LANE / "tampered" / case / "pilot4_walk_s42" / "03_locomotion"
    (st / "models").mkdir(parents=True, exist_ok=False)
    for name in RECORDS:
        shutil.copy2((records_from or src) / name, st / name)
    if pair_from is None:
        shutil.copy2(src / "models" / "stage2_2000000_steps.zip", st / "models" / "stage2_2000000_steps.zip")
        shutil.copy2(src / "models" / "stage2_vecnormalize_2000000_steps.pkl",
                     st / "models" / "stage2_vecnormalize_2000000_steps.pkl")
    else:
        shutil.copy2(pair_from / "stage2_8000000_steps.zip", st / "models" / "stage2_2000000_steps.zip")
        shutil.copy2(pair_from / "stage2_vecnormalize_8000000_steps.pkl",
                     st / "models" / "stage2_vecnormalize_2000000_steps.pkl")
    return st


st = copy_case("parent_record")
sc = json.loads((st / "stage_config.json").read_text())
before = sc["run"]["parent_checkpoint_sha256"]
sc["run"]["parent_checkpoint_sha256"] = P52_ZIP
(st / "stage_config.json").write_text(json.dumps(sc, indent=2))
out["tampered"]["parent_record"] = {"run.parent_checkpoint_sha256": [before, P52_ZIP]}

st = copy_case("command_table")
sc = json.loads((st / "stage_config.json").read_text())
before = sc["reward_weights"]["command_config"]["yaw_gain"]
sc["reward_weights"]["command_config"]["yaw_gain"] = 2.0
(st / "stage_config.json").write_text(json.dumps(sc, indent=2))
out["tampered"]["command_table"] = {"reward_weights.command_config.yaw_gain": [before, 2.0]}

st = copy_case("raw_parent", pair_from=P3RUNS / "pilot3_walk_s42" / "03_locomotion" / "models")
out["tampered"]["raw_parent"] = {"pair": "pilot 3's s42 8M pair, byte for byte, as stage2_2000000_steps.*",
                                 "zip": sha(st / "models" / "stage2_2000000_steps.zip")}

st = copy_case("setup_check_record", records_from=FAKE / "pilot4_walk_s42" / "03_locomotion")
out["tampered"]["setup_check_record"] = {
    "records_from": str(FAKE / "pilot4_walk_s42" / "03_locomotion"),
    "git_commit": json.loads((st / "stage_config.json").read_text()).get("git_commit"),
    "files": {n: sha(st / n) for n in RECORDS}}
(LANE / "runs" / "make_runs.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
