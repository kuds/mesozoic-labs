"""Collect the pilot-4 notebook smoke's evidence (smoke lane; reads only). Pilot 3's evidence2.py adapted to pilot 4.

    python -I evidence4.py NBOUT RUN_DIR CLONE TOOLS REF_KEYS.json OUT.json [--resume]

NBOUT the runner's --out (record.json, runner.stdout.log, runner.stderr.log); RUN_DIR the run copy (pilot4_walk_s42);
CLONE the clone the smoke ran in (its HEAD); TOOLS the pinned pilot-4 tools (pilot4_habit.py's ledger columns);
REF_KEYS the setup lane's judge_bound_keys.json (the 2,000,000-step entry at the head, learn() stopped), against which
every judge-bound key of the smoke's stage_config.json, plant_identity.json and zip lineage is compared. Nothing is
unpickled but the policy weights (torch.load weights_only=True) for exp(log_std).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True

JUDGE_KEYS = [
    "species", "embodiment", "stage", "git_commit", "run.load_mode", "run.seed", "run.parent_checkpoint_sha256",
    "run.timesteps", "run.resume_load_path", "run.hyperparameters_sha256", "run.n_envs", "run.parent_task_sha256",
    "run.critic_phase_dropped", "run.warm_start_log_std", "task_fingerprint.task_sha256",
    "curriculum.warmup_timesteps", "curriculum.ramp_start_value", "curriculum.initialize_from_own_checkpoint",
    "curriculum.warm_start_log_std", "curriculum.eval_freq", "curriculum.eval_episodes", "curriculum.checkpoint_freq",
    "curriculum.max_checkpoints", "hyperparameters.critic_phase", "reward_weights.command_mode",
    "reward_weights.command_config", "reward_weights.prey_lateral_range", "reward_weights.walk_heading_price_weight",
    "reward_weights.walk_heading_price_scale_rad", "reward_weights.walk_support_gate_bw",
    "reward_weights.airborne_substep_weight", "reward_weights.forward_vel_max", "reward_weights.walk_touchdown_weight",
]
HEADER_PREFIXES = ("Running locally", "Repository ref", "Model:", "Algorithm:", "Quick test mode:", "Behavior ",
                   "In-run K3 panel:", "Critic phase:", "Run directory:", "Node ", "Timesteps:",
                   "VecNormalize loaded from:", "Warm start", "nb_pilot_runner4: cadence", "Resuming ",
                   "Newest intact periodic checkpoint:", "Matched VecNormalize:", "Checkpoint steps:",
                   "runner_snapshot:", "locomotion: gate verdict", "Result bundle not rewritten", "Eval num_timesteps=")


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def get(d, dotted):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return "<absent>"
    return cur


def zip_data(p: Path) -> dict:
    with zipfile.ZipFile(p) as z:
        return json.loads(z.read("data"))


def zip_members(p: Path) -> list:
    with zipfile.ZipFile(p) as z:
        return z.namelist()


def policy_std(p: Path):
    import torch

    with zipfile.ZipFile(p) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), weights_only=True, map_location="cpu")
    std = sd["log_std"].double().exp()
    return {"mean": round(float(std.mean()), 6), "min": round(float(std.min()), 6), "max": round(float(std.max()), 6),
            "n": int(std.numel())}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    resume = "--resume" in sys.argv
    nbout, run, clone, tools, ref_path, out = (Path(a).resolve() for a in args[:6])
    ev: dict = {"nbout": str(nbout), "run": str(run), "phase": "resume" if resume else "launch"}
    rec = json.loads((nbout / "record.json").read_text())
    ev["record"] = {k: rec.get(k) for k in ("tool_sha256", "notebook_sha256", "clone_head", "clone_status_before",
                                            "roles", "run_all", "override", "override_refused", "exit", "started",
                                            "ended", "ipython_stand_in", "locomotion_curriculum_at_end", "run_dir",
                                            "environments_resolves_to", "environments_imported_from",
                                            "cadence_override_requested", "phase", "parent_pair", "entry_lines")}
    ev["cells"] = [(c["cell"], c["ok"], c["seconds"], c.get("exception"), (c.get("message") or "")[:200])
                   for c in rec.get("cells", [])]
    stdout = (nbout / "runner.stdout.log").read_text().splitlines()
    stderr = (nbout / "runner.stderr.log").read_text()
    ev["header_lines"] = [(i + 1, ln) for i, ln in enumerate(stdout) if ln.startswith(HEADER_PREFIXES)]
    ev["warm_start_lines_stdout"] = [(i + 1, ln) for i, ln in enumerate(stdout) if "Warm start" in ln]
    ev["warm_start_on_stderr"] = [ln for ln in stderr.splitlines() if "Warm start" in ln]
    ev["eval_lines"] = [(i + 1, ln) for i, ln in enumerate(stdout) if ln.startswith("Eval num_timesteps=")]
    ev["gate_lines"] = [(i + 1, ln) for i, ln in enumerate(stdout)
                        if "gate verdict" in ln or ln.startswith("Result bundle not rewritten")]
    ev["warning_lines_stdout"] = [(i + 1, ln[:300]) for i, ln in enumerate(stdout) if "WARNING" in ln]
    ev["stderr_lines_not_indented"] = [ln[:300] for ln in stderr.splitlines() if ln and not ln.startswith(" ")][:80]
    # stage_config.json against the setup lane's judge-bound keys (the 2,000,000-step entry at the head)
    st = run / "03_locomotion"
    sc = json.loads((st / "stage_config.json").read_text())
    ref = json.loads(ref_path.read_text())
    ref_sc = ref["stage_config.json"]
    keys = {k: get(sc, k) for k in JUDGE_KEYS}
    ev["judge_bound_keys"] = keys
    ev["judge_bound_keys_differing_from_setup_entry"] = {k: {"smoke": keys[k], "setup_entry_2M": ref_sc.get(k)}
                                                         for k in JUDGE_KEYS if keys[k] != ref_sc.get(k)}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip()
    rb, cur = sc.get("run", {}), sc.get("curriculum", {})
    ev["stage_config_other"] = {
        "run": rb, "curriculum": cur, "git_commit_equals_clone_head": sc.get("git_commit") == head,
        "clone_head": head, "name": sc.get("name"), "label": get(sc, "run.label"),
        "load_path": get(sc, "run.load_path"), "vecnorm_path": get(sc, "run.vecnorm_path"),
        "resume_checkpoint_sha256": get(sc, "run.resume_checkpoint_sha256"),
        "reward_weights_keys": sorted((sc.get("reward_weights") or {}).keys()),
        "hyperparameter_keys": sorted((sc.get("hyperparameters") or {}).keys()),
    }
    pi = json.loads((st / "plant_identity.json").read_text())
    ev["plant_identity"] = {k: pi.get(k) for k in ("physics_revision", "physics_sha256", "policy_interface_sha256",
                                                    "policy_interface_revision", "observation_dim", "action_dim")}
    ev["plant_identity_equal_to_setup_entry"] = all(pi.get(k) == v for k, v in ref["plant_identity.json"].items())
    tf = st / "task_fingerprint.json"
    ev["task_fingerprint_json"] = json.loads(tf.read_text()).get("task_sha256") if tf.is_file() else None
    # Checkpoints: lineage (zip data member), num_timesteps, tensorboard_log, exp(log_std)
    models = st / "models"
    ref_lin = ref["zip data member mesozoic_task_lineage"]["record"]
    ck = {}
    for p in sorted(models.glob("*.zip")):
        d = zip_data(p)
        lin = d.get("mesozoic_task_lineage")
        lin_val = None
        if isinstance(lin, dict):
            lin_val = {k: v for k, v in lin.items() if not k.startswith(":")}
        ck[p.name] = {"bytes": p.stat().st_size, "sha256": sha(p), "num_timesteps": d.get("num_timesteps"),
                      "members": zip_members(p), "tensorboard_log": d.get("tensorboard_log"),
                      "lineage_key_present": "mesozoic_task_lineage" in d,
                      "lineage_raw_keys": sorted(lin.keys()) if isinstance(lin, dict) else repr(lin)[:200],
                      "lineage": lin_val, "policy_std": policy_std(p),
                      "data_keys_mesozoic": sorted(k for k in d if k.startswith("mesozoic"))}
    for p in sorted(models.glob("*.pkl")):
        ck[p.name] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    ev["checkpoints"] = ck
    ev["setup_entry_lineage_record"] = ref_lin
    # Ledger columns against pilot4_habit.py's
    spec = importlib.util.spec_from_file_location("pilot4_habit_cols", tools / "pilot4_habit.py")
    ph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ph)
    with open(st / "eval_episodes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    header = list(rows[0].keys()) if rows else []
    need = list(ph.LEDGER_REQUIRED) + [ph.TD_COLUMN, *ph.POSTURE_COLUMNS]
    ev["ledger"] = {"rows": len(rows), "columns": len(header),
                    "rows_per_timesteps": dict(Counter(r["timesteps"] for r in rows)),
                    "habit_columns_missing": [c for c in need if c not in header],
                    "all_habit_columns_present": all(c in header for c in need),
                    "saturated_actuators_seen": dict(Counter(r.get("saturated_actuator") for r in rows))}
    # TensorBoard
    tb = sorted((st / "tensorboard").rglob("events.out.tfevents.*")) if (st / "tensorboard").is_dir() else []
    ev["tensorboard"] = {"files": [str(p.relative_to(run)) for p in tb], "sizes": [p.stat().st_size for p in tb]}
    if tb:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        per_dir = {}
        for d in sorted({p.parent for p in tb}):
            acc = EventAccumulator(str(d), size_guidance={"scalars": 0})
            acc.Reload()
            tags = acc.Tags().get("scalars", [])
            row = {"n_scalar_tags": len(tags)}
            for tag in ("train/std", "train/value_loss", "train/explained_variance", "time/fps", "train/approx_kl",
                        "train/clip_fraction", "train/learning_rate", "train/entropy_loss", "eval/mean_reward",
                        "rollout/ep_rew_mean"):
                if tag in tags:
                    row[tag] = [(e.step, round(e.value, 6), round(e.wall_time, 3)) for e in acc.Scalars(tag)]
            per_dir[str(d.relative_to(run))] = row
        ev["tensorboard"]["per_dir"] = per_dir
    # Files and provenance
    ev["run_files"] = sorted(f"{p.relative_to(run)} {p.stat().st_size}" for p in run.rglob("*") if p.is_file())
    for name in ("stage_summary.txt",):
        p = st / name
        ev[name] = re.findall(r"^(?:Timesteps|Training time|Duration|Wall).*$", p.read_text(), re.M) if p.is_file() \
            else None
    prov_p = run / "provenance.json"
    if prov_p.is_file():
        prov = json.loads(prov_p.read_text())
        ev["provenance"] = {k: prov.get(k) for k in ("dependency_versions", "repository_commit", "repository_dirty",
                                                     "hardware", "python", "platform")}
    gv = st / "gate_verdict.json"
    ev["gate_verdict"] = json.loads(gv.read_text()) if gv.is_file() else None
    Path(out).write_text(json.dumps(ev, indent=1, default=str))
    print(json.dumps({k: ev[k] for k in ("record", "header_lines", "eval_lines", "gate_lines",
                                         "warm_start_lines_stdout", "warm_start_on_stderr",
                                         "judge_bound_keys_differing_from_setup_entry", "plant_identity",
                                         "plant_identity_equal_to_setup_entry")}, indent=1, default=str)[:30000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
