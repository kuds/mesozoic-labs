"""Compare the resumed run with the unbroken launch (smoke lane; reads only; report only, as pilot 3's).

    python -I compare_resume.py LAUNCH_RUN RESUME_RUN BEFORE.json AFTER.json OUT.json

* stage_config.json before and after the RESUME cell: every leaf that differs (dotted paths), and the judge-bound keys
  the task names (load_mode, parent sha256, parent_task_sha256, the env and curriculum keys) before and after;
* every zip both runs hold (the 32,768 and 65,536 pairs, the final model, best and robust best): each member's sha256
  equal or not, the JSON data member's keys that differ, num_timesteps, and the lineage record
  (mesozoic_task_lineage) in each;
* the sidecars' bytes equal or not.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

BOUND = ["run.load_mode", "run.parent_checkpoint_sha256", "run.parent_task_sha256", "run.seed", "run.n_envs",
         "run.timesteps", "run.resume_load_path", "run.resume_checkpoint_sha256", "run.hyperparameters_sha256",
         "run.critic_phase_dropped", "run.warm_start_log_std", "git_commit", "task_fingerprint.task_sha256",
         "curriculum.warmup_timesteps", "curriculum.ramp_start_value", "curriculum.initialize_from_own_checkpoint",
         "curriculum.warm_start_log_std", "curriculum.eval_freq", "curriculum.eval_episodes",
         "curriculum.checkpoint_freq", "curriculum.max_checkpoints", "hyperparameters.critic_phase",
         "reward_weights.command_mode", "reward_weights.command_config", "reward_weights.prey_lateral_range",
         "reward_weights.walk_heading_price_weight", "reward_weights.walk_heading_price_scale_rad",
         "reward_weights.walk_support_gate_bw", "reward_weights.airborne_substep_weight",
         "reward_weights.forward_vel_max", "reward_weights.walk_touchdown_weight"]


def get(d, dotted):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return "<absent>"
    return cur


def leaves(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(leaves(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def zinfo(p: Path) -> dict:
    with zipfile.ZipFile(p) as z:
        members = {n: sha(z.read(n)) for n in z.namelist()}
        data = json.loads(z.read("data"))
    return {"members": members, "data": data}


def main() -> int:
    launch, resume, before_p, after_p, out = (Path(a).resolve() for a in sys.argv[1:6])
    before, after = json.loads(before_p.read_text()), json.loads(after_p.read_text())
    lb, la = leaves(before), leaves(after)
    rep: dict = {"stage_config_leaves_changed_by_resume": {k: {"before": lb.get(k, "<absent>"),
                                                               "after": la.get(k, "<absent>")}
                                                           for k in sorted(set(lb) | set(la))
                                                           if lb.get(k, "<absent>") != la.get(k, "<absent>")},
                 "bound_keys": {k: {"before": get(before, k), "after": get(after, k),
                                    "kept": get(before, k) == get(after, k)} for k in BOUND}}
    lm, rm = launch / "03_locomotion" / "models", resume / "03_locomotion" / "models"
    zips = {}
    for p in sorted(lm.glob("*.zip")):
        q = rm / p.name
        if not q.is_file():
            zips[p.name] = "only in the launch run"
            continue
        a, b = zinfo(p), zinfo(q)
        dk = sorted(k for k in set(a["data"]) | set(b["data"]) if a["data"].get(k) != b["data"].get(k))
        zips[p.name] = {"members_equal": {n: a["members"].get(n) == b["members"].get(n)
                                          for n in sorted(set(a["members"]) | set(b["members"]))},
                        "data_keys_differ": dk,
                        "num_timesteps": [a["data"].get("num_timesteps"), b["data"].get("num_timesteps")],
                        "lineage_launch": a["data"].get("mesozoic_task_lineage"),
                        "lineage_resume": b["data"].get("mesozoic_task_lineage"),
                        "lineage_equal": a["data"].get("mesozoic_task_lineage") == b["data"].get("mesozoic_task_lineage")}
    for q in sorted(rm.glob("*.zip")):
        if not (lm / q.name).is_file():
            zips[q.name] = "only in the resumed run"
    rep["zips"] = zips
    rep["sidecars_equal"] = {p.name: ((rm / p.name).is_file() and p.read_bytes() == (rm / p.name).read_bytes())
                             for p in sorted(lm.glob("*.pkl"))}
    Path(out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str)[:12000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
