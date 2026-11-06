"""Make the tampered run copies the pilot-4 judge's refusals read (smoke lane; pilot 3's make_copies.py adapted).

    python -I make_copies4.py LAUNCH_RUN RESUME_RUN COPIES_DIR

LAUNCH_RUN and RESUME_RUN are this lane's copies of the smoke's run folders (pilot4_walk_s42). Each copy is
COPIES_DIR/<case>/pilot4_walk_s42/ (the folder name is the RUN_ID the judge checks; case wrong_run_id is named
pilot3_walk_s42) and holds 03_locomotion/{stage_config.json, plant_identity.json, task_fingerprint.json,
stage_summary.txt if present} and models/ with the files named below, byte for byte, then ONE change:
  parent_sha_record   run.parent_checkpoint_sha256 set to s52's 8M zip sha256          (judged at 65,536)
  command_table       reward_weights.command_config.yaw_gain 1.5 -> 1.0 (one value)  (judged at 65,536)
  env_kwarg_drift     reward_weights.forward_vel_weight 2.0 -> 3.0 (one value)         (judged at 65,536)
  other_head          git_commit set to pilot 3's head 31d02e42...                     (judged at 65,536)
  n_envs_2            run.n_envs set to 2                                              (judged at 65,536)
  warm_start_log_std  run.warm_start_log_std set to -2.0 (pilot 3's record)            (judged at 65,536)
  critic_dropped      run.critic_phase_dropped set to pilot 3's record shape           (judged at 65,536)
  parent_task         run.parent_task_sha256 set to the child's own task 914a1cae...   (judged at 65,536)
  no_lineage          stage2_65536_steps.zip's JSON data member without mesozoic_task_lineage (every other member
                      and the sidecar byte for byte)                                   (judged at 65,536)
  wrong_run_id        the folder named pilot3_walk_s42                                 (judged at 65,536)
  fallback_wrong      stage2_final.zip's JSON data member num_timesteps set to 49,152 (--final-fallback --steps 50000)
  resume_nonperiodic  (from RESUME_RUN) run.resume_load_path renamed to .../stage2_final (not a periodic file)
copies.json records each file's sha256, the change, and the value before and after.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

PAIR = ("stage2_65536_steps.zip", "stage2_vecnormalize_65536_steps.pkl")
FINAL = ("stage2_final.zip", "stage2_final_vecnorm.pkl")
S52_PARENT = "sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00"
PILOT3_HEAD = "31d02e42d148511e50f9cba9a598d9a9d36c23af"
CHILD_TASK = "sha256:914a1caee2c086a09ef66e9e6d4ad234dd085b69c4da687cc29af82b110fcd7a"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def base_copy(run: Path, dst: Path, models: tuple[str, ...]) -> None:
    (dst / "03_locomotion" / "models").mkdir(parents=True)
    for name in ("stage_config.json", "plant_identity.json", "task_fingerprint.json", "stage_summary.txt"):
        src = run / "03_locomotion" / name
        if src.is_file():
            shutil.copyfile(src, dst / "03_locomotion" / name)
    for name in models:
        shutil.copyfile(run / "03_locomotion" / "models" / name, dst / "03_locomotion" / "models" / name)


def edit_stage_config(dst: Path, fn) -> dict:
    p = dst / "03_locomotion" / "stage_config.json"
    sc = json.loads(p.read_text())
    change = fn(sc)
    p.write_text(json.dumps(sc, indent=2))
    return change


def rewrite_zip_data(src: Path, dst: Path, fn) -> dict:
    """Copy the zip member by member; the JSON data member alone is edited by fn (returns the change)."""
    change = {}
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "data":
                doc = json.loads(data)
                change = fn(doc)
                data = json.dumps(doc, indent=4).encode()
            zout.writestr(info, data)
    return change


def setter(path: str, value):
    def fn(sc):
        keys = path.split(".")
        cur = sc
        for k in keys[:-1]:
            cur = cur[k]
        before = cur.get(keys[-1], "<absent>")
        cur[keys[-1]] = value
        return {"key": path, "before": before, "after": value}
    return fn


def main() -> int:
    launch, resume, out = (Path(a).resolve() for a in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=False)
    record = {"launch_run": str(launch), "resume_run": str(resume), "copies": {}}

    def finish(case, dst, change):
        record["copies"][case] = {"run_copy": str(dst), "change": change,
                                  "files": {str(p.relative_to(dst)): sha(p) for p in sorted(dst.rglob("*"))
                                            if p.is_file()}}

    edits = {
        "parent_sha_record": setter("run.parent_checkpoint_sha256", S52_PARENT),
        "command_table": setter("reward_weights.command_config.yaw_gain", 1.0),
        "env_kwarg_drift": setter("reward_weights.forward_vel_weight", 3.0),
        "other_head": setter("git_commit", PILOT3_HEAD),
        "n_envs_2": setter("run.n_envs", 2),
        "warm_start_log_std": setter("run.warm_start_log_std", -2.0),
        "critic_dropped": setter("run.critic_phase_dropped", {"inputs": 3, "kind": "episode_phase"}),
        "parent_task": setter("run.parent_task_sha256", CHILD_TASK),
    }
    for case, fn in edits.items():
        dst = out / case / "pilot4_walk_s42"
        base_copy(launch, dst, PAIR)
        finish(case, dst, edit_stage_config(dst, fn))

    dst = out / "no_lineage" / "pilot4_walk_s42"
    base_copy(launch, dst, (PAIR[1],))

    def drop_lineage(doc):
        before = doc.pop("mesozoic_task_lineage", "<absent>")
        return {"key": "data member mesozoic_task_lineage", "before_keys": sorted(before) if isinstance(before, dict)
                else before, "after": "<absent>"}
    finish("no_lineage", dst, rewrite_zip_data(launch / "03_locomotion" / "models" / PAIR[0],
                                               dst / "03_locomotion" / "models" / PAIR[0], drop_lineage))

    dst = out / "wrong_run_id" / "pilot3_walk_s42"
    base_copy(launch, dst, PAIR)
    finish("wrong_run_id", dst, {"key": "run folder name", "before": "pilot4_walk_s42", "after": "pilot3_walk_s42"})

    dst = out / "fallback_wrong" / "pilot4_walk_s42"
    base_copy(launch, dst, (FINAL[1],))

    def num_ts(doc):
        before = doc.get("num_timesteps")
        doc["num_timesteps"] = 49_152
        return {"key": "stage2_final.zip data num_timesteps", "before": before, "after": 49_152}
    finish("fallback_wrong", dst, rewrite_zip_data(launch / "03_locomotion" / "models" / FINAL[0],
                                                   dst / "03_locomotion" / "models" / FINAL[0], num_ts))

    dst = out / "resume_nonperiodic" / "pilot4_walk_s42"
    base_copy(resume, dst, PAIR)

    def nonperiodic(sc):
        before = sc["run"].get("resume_load_path")
        after = str(Path(str(before)).with_name("stage2_final"))
        sc["run"]["resume_load_path"] = after
        return {"key": "run.resume_load_path", "before": before, "after": after}
    finish("resume_nonperiodic", dst, edit_stage_config(dst, nonperiodic))

    (out / "copies.json").write_text(json.dumps(record, indent=1))
    for case, c in record["copies"].items():
        print(case, json.dumps(c["change"], default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
