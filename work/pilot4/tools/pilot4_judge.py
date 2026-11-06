"""Judge ONE checkpoint of a walking-pilot-4 run: the gait probe's 40-episode floor-truth panel.

Scratch, never merged: walking pilot 3's pilot3_judge.py (sha256 df5ea214...) adapted for walking pilot 4 as the
pre-registration's TOOL PLAN says (criteria_pilot4 draft v3, "TOOL PLAN", "Run handling", "How it is judged"). Run with
the pilot branch's clean tree as the working directory (a clean clone of pilot-walk-r10-p4 at the head the runs
recorded), isolated, e.g.
    cd TREE && ISO_WT=TREE python3 -I TOOLS/iso.py TOOLS/pilot4_judge.py RUN --seed 42 --steps 2000000
        --parent-sha256 sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4 --out OUT
        [--expect-head HEAD]
RUN is a local copy of the pilot run directory, named by its RUN_ID, pilot4_walk_s<seed> or pilot4_walk_s<seed>_b
(paths relative to the run, as on Drive):
    03_locomotion/models/stage2_<steps>_steps.zip   03_locomotion/models/stage2_vecnormalize_<steps>_steps.pkl
    03_locomotion/stage_config.json                 03_locomotion/plant_identity.json
    03_locomotion/models/stage2_final.zip and stage2_final_vecnorm.pkl, or 03_locomotion/stage_summary.txt
        (optional: the run-end record below; required under --final-fallback)
It writes OUT/s<seed>/<steps>/probe.json (the probe's JSON), the probe's trace and plots beside it, and binding.json;
pilot4_score.py and pilot4_straight.py read probe.json and binding.json. Nothing is written when a check refuses.
An off-schedule run judged with --report-off-schedule is written to OUT/off_schedule/<RUN's directory name>/s<seed>/
<steps>/ instead, never to OUT/s<seed>/<steps> (pilot 3's rule; under pilot 4's inert shaping only a resume from a
checkpoint that is not periodic, or a tree whose shaping is not inert, can be off-schedule).

Checks before rolling anything (each refusal names its reason; pilot 3's kept, then pilot 4's):
  * the tree is clean and on a walking-pilot head (--expect-head pins one): its checked-out branch starts with
    BRANCH_PREFIX (pilot 3's, kept), or, detached, its HEAD is the tip of a local branch that does; its git reads take
    no optional locks;
  * the run copy's directory name is a pilot-4 RUN_ID, pilot4_walk_s<seed> or pilot4_walk_s<seed>_b, of --seed;
  * stage_config.json records the locomotion stage (2) of tyrannosaurus_rex/anatomical, an initialize_next_stage load,
    the seed --seed, the pilot's budget 2,000,000 (--expect-budget), and a git_commit equal to the tree's HEAD: 7 to 40
    hex digits that begin HEAD; no git_commit is refused (--allow-commit-mismatch relaxes only this one);
  * the parent: --parent-sha256 must be the --seed's own pilot-3 8M zip in PARENT_SHA256 (the text's per-seed table),
    and run.parent_checkpoint_sha256 must equal it;
  * a resumed run (run.resume_load_path set: the notebook's RESUME cell): no resume is off-schedule when this tree's
    stage-entry shaping is inert (warmup_timesteps 0 and ramp_start_value equal to the [env] forward_vel_weight, 2.0);
    otherwise pilot 3's rule (a resume before ramp_timesteps is OFF-SCHEDULE). A resume from a file that is not a
    periodic stage2_<steps>_steps checkpoint is refused. The budget of a resumed run is its recorded run.timesteps
    plus the step it resumed from;
  * its hyperparameters_sha256 and task_fingerprint.task_sha256 equal the ones this tree derives for the stage;
    (fix stage) at the pilot budget the tree's derivations must also be the text's own, HYPERPARAMETERS_SHA256
    fbc94bef... and TASK_SHA256 914a1cae..., and its committed [env.command_config] the text's COMMAND_TABLE, so a
    record and a tree that moved together are refused; --expect-head defaults to PILOT_HEAD (0e6a57b), checked after
    the branch rule;
  * (fix stage) the recorded env kwargs equal the env this tree builds: env_kwargs_drift, the probe's own drift
    record (every recorded or committed [env] key against {constructor defaults, **committed env_kwargs}), must be
    empty; after the probe its own meta.env_kwargs_drift_vs_recorded must be empty too (the probe only warns);
  * plant_identity.json's physics and policy-interface digests equal the tree's;
  * entry_problems (in place of pilot 3's warm-start checks): no run.critic_phase_dropped record, no warm_start_log_std
    record (run or curriculum), curriculum warmup_timesteps 0, ramp_start_value 2.0, initialize_from_own_checkpoint
    true, run.n_envs 4, no critic_phase key in the recorded hyperparameters, run.parent_task_sha256 782d95b6... (pilot
    3's command-free task), and the judged zip's lineage record (read from its JSON data member, never unpickled):
    mode initialize_next_stage, parent_stage 2, zeroed_command_parameters;
  * env binding: the recorded env kwargs (stage_config.json's reward_weights) carry command_mode "heading", the
    tree's committed [env.command_config] table key for key (same keys, same values of the same JSON types),
    prey_lateral_range [0, 0], walk_heading_price_weight 0.2 and walk_heading_price_scale_rad 0.5, and R6's four keys
    (walk_support_gate_bw 0.04, airborne_substep_weight 1.0, forward_vel_max 1.5, walk_touchdown_weight 0.1);
  * cadence (a "Run handling" record): curriculum eval_freq 65536, eval_episodes 10, checkpoint_freq 250000,
    max_checkpoints 0, checked at the pilot budget only (a validation-only run, --expect-budget other than 2,000,000,
    records it in binding.json unchecked: the notebook smoke overrides the cadence in memory);
  * the judged checkpoint is an intact pair: both files exist and pass the tree's checkpoint_pair_problem; the probe's
    own loader then enforces the plant contract (exit 3 = refused).
--final-fallback (criteria "Final-model fallback"): judge the run's final pair (stage2_final.zip with
stage2_final_vecnorm.pkl) in place of the deciding checkpoint, only when --steps is the budget, that periodic pair is
missing or broken, the final pair is intact, and the final zip's num_timesteps (read from its JSON data member) equals
the uninterrupted end: whole rollouts of N_ENVS x n_steps (16,384) at or past the budget (2,000,000 -> 2,015,232,
PILOT_UNINTERRUPTED_END). The cell is still OUT/s<seed>/<steps>/, and binding.json records final_fallback with the
reason. A resumed run's final ends elsewhere, so it is refused.
--check-only: run every check above and stop before the probe; nothing is written.
Run-end record (binding.json run_end, pilot 3's): the newest periodic checkpoint, the newest INTACT periodic pair, and
the steps the run ended at (stage2_final.zip's num_timesteps, else stage_summary.txt's "Timesteps:").
One run per seed in OUT (pilot 3's): the judge refuses to write OUT/s<seed>/<steps> while OUT/s<seed>/ holds a cell
judged from a run copy with another directory name (the RUN_ID); a restarted seed's retired cells are moved, never
deleted, to OUT/retired/<retired RUN_ID>/s<seed>/<steps>/ before its _b run is judged.
Probe arguments are fixed: --stage locomotion --episodes 40 --seed-scheme panel --seed-base 3042 --settle-steps 100.
MUJOCO_GL is passed to the probe as set, else "disable".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT_BUDGET = 2_000_000
PILOT_UNINTERRUPTED_END = 2_015_232  # 123 rollouts of 4 x 4096
#: Each seed's own parent: its pilot-3 periodic zip at 8,000,000 (the text's table, "Parent and entry").
PARENT_SHA256 = {
    42: "41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4",
    52: "2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00",
    60: "2c066770802541e47fb453b8dbd04b030bc8053d4224af27a52eb0281e4645a2",
}
#: Pilot 3's command-free task, which every parent pair records ("Parent and entry").
PARENT_TASK_SHA256 = "782d95b607d0086d0cefe475c9caf8ac420c1f2b74d23b786808aaee98232a6a"
LINEAGE_ATTRIBUTE = "mesozoic_task_lineage"
LINEAGE_MODE = "initialize_next_stage"
LINEAGE_PARENT_STAGE = 2
WARMUP_TIMESTEPS = 0
RAMP_START_VALUE = 2.0
N_ENVS = 4
COMMAND_MODE = "heading"
PREY_LATERAL_RANGE = (0.0, 0.0)
PRICE_KEYS = {"walk_heading_price_weight": 0.2, "walk_heading_price_scale_rad": 0.5}
R6_KEYS = {"walk_support_gate_bw": 0.04, "airborne_substep_weight": 1.0, "forward_vel_max": 1.5,
           "walk_touchdown_weight": 0.1}
CADENCE = {"eval_freq": 65536, "eval_episodes": 10, "checkpoint_freq": 250000, "max_checkpoints": 0}
#: The text's own values (v3 "Run handling", first-hour list; "What is trained"), fix stage: at the pilot budget the
#: judging tree must derive these, so a record and a tree that moved together are refused; --expect-head defaults to
#: the head the runs record ("Code").
PILOT_HEAD = "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8"
HYPERPARAMETERS_SHA256 = "sha256:fbc94bef69a91d5a65f1df99adb9c35e7897a259affaa3e7b7ba4477418eaa62"
TASK_SHA256 = "sha256:914a1caee2c086a09ef66e9e6d4ad234dd085b69c4da687cc29af82b110fcd7a"
COMMAND_TABLE = {"cruise_speed": 1.5, "stop_probability": 0.0, "turn_increment_max": 0.0, "straight_probability": 1.0,
                 "speed_scale": 1.5, "lateral_speed_scale": 1.0, "yaw_rate_scale": 0.6, "yaw_gain": 1.5,
                 "yaw_rate_max": 0.3, "turn_slowdown": True, "minimum_turn_speed_fraction": 0.25}
BRANCH_PREFIX = "claude/pilot-walk-"
RUN_ID_RE = re.compile(r"pilot4_walk_s(\d+)(_b)?")
PROBE_ARGS = ["--species", "tyrannosaurus_rex", "--stage", "locomotion", "--episodes", "40", "--seed-scheme", "panel",
              "--seed-base", "3042", "--settle-steps", "100"]
PERIODIC_RE = re.compile(r"stage2_(\d+)_steps(\.zip)?$")
FINAL_ZIP, FINAL_VEC = "stage2_final.zip", "stage2_final_vecnorm.pkl"
CADENCE_KEYS = tuple(CADENCE)


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def norm_sha(value: object) -> str | None:
    if not value:
        return None
    s = str(value).strip().lower()
    s = s[len("sha256:"):] if s.startswith("sha256:") else s
    return s if re.fullmatch(r"[0-9a-f]{64}", s) else None


def refuse(msg: str) -> int:
    print(f"refusing (nothing written): {msg}", flush=True)
    return 2


def git(tree: Path, *args: str) -> str:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.check_output(["git", *args], cwd=tree, text=True, env=env).strip()


def read_zip_data(path: Path) -> dict:
    """An SB3 archive's JSON 'data' member (json and zipfile only: nothing is unpickled)."""
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("data"))


def is_number(value: object) -> bool:
    """A JSON number that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def same_json(a: object, b: object) -> bool:
    """Equal values of the same JSON types, recursively: 4 is not 4.0, True is not 1, a list is a tuple's twin."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def table_differences(recorded: object, committed: object) -> list[str]:
    """How a recorded [env.command_config] table differs from the committed one, key for key ([] when it does not)."""
    if not isinstance(committed, dict):
        return [f"the tree's committed command_config is not a table ({type(committed).__name__})"]
    if not isinstance(recorded, dict):
        return [f"the recorded command_config is not a table: {recorded!r}"]
    out = [f"{k} missing" for k in committed if k not in recorded]
    out += [f"{k} = {recorded[k]!r} added" for k in recorded if k not in committed]
    out += [f"{k} = {recorded[k]!r} ({type(recorded[k]).__name__}), committed {committed[k]!r} "
            f"({type(committed[k]).__name__})" for k in committed if k in recorded
            and not same_json(recorded[k], committed[k])]
    return out


def pilot_branch_from(current: str, head_refs: list[str]) -> tuple[str | None, str]:
    """(the walking-pilot branch the tree is on, how; or None, why not). *current* is `git branch --show-current`
    ("" when detached); *head_refs* the full ref names whose tip is HEAD."""
    if current:
        if current.startswith(BRANCH_PREFIX):
            return current, "checked out"
        return None, f"tree branch {current!r} is not a walking-pilot branch"
    names = sorted(r[len("refs/heads/"):] for r in head_refs if r.startswith("refs/heads/" + BRANCH_PREFIX))
    if names:
        return names[0], "detached HEAD at the tip of the local branch"
    return None, "the tree's HEAD is detached and is the tip of no local walking-pilot branch"


def run_id_problem(run_name: str, seed: int) -> str | None:
    """None when *run_name* is pilot4_walk_s<seed> or pilot4_walk_s<seed>_b; else why not."""
    m = RUN_ID_RE.fullmatch(run_name)
    if not m:
        return (f"the run copy's folder {run_name!r} is not a pilot-4 RUN_ID (pilot4_walk_s<seed> or "
                f"pilot4_walk_s<seed>_b)")
    if int(m.group(1)) != seed or m.group(1) != str(int(m.group(1))):
        return f"the run copy's folder {run_name} is not seed {seed}'s RUN_ID (pilot4_walk_s{seed}[_b])"
    return None


def shaping_inert(warmup_timesteps: object, ramp_start_value: object, forward_vel_weight: object) -> bool:
    """Whether the stage-entry shaping does nothing: no warm-up and a forward-weight ramp that starts at the weight."""
    return (is_number(warmup_timesteps) and float(warmup_timesteps) == 0.0 and is_number(ramp_start_value)
            and is_number(forward_vel_weight) and float(ramp_start_value) == float(forward_vel_weight))


def resume_check(run_block: dict, ramp_timesteps: int, warmup_timesteps: int, ramp_start_value: object = None,
                 forward_vel_weight: object = None) -> tuple[int | None, str | None]:
    """(resume step or None, off-schedule reason or None) for a stage_config run block.

    Pilot 4 (D11): no resume is off-schedule when the shaping is inert (warmup_timesteps 0 and ramp_start_value equal
    to the [env] forward_vel_weight); otherwise pilot 3's rule. A resume from a file that is not a periodic
    checkpoint has no step and is refused, as in pilot 3."""
    resumed = run_block.get("resume_load_path")
    if not resumed:
        return None, None
    m = PERIODIC_RE.match(Path(str(resumed)).name)
    if not m:
        return None, f"resumed from {resumed}, which is not a periodic stage2_<steps>_steps checkpoint"
    step = int(m.group(1))
    if shaping_inert(warmup_timesteps, ramp_start_value, forward_vel_weight):
        return step, None
    shaping_end = max(int(ramp_timesteps), int(warmup_timesteps))
    if step < shaping_end:
        return step, (f"OFF-SCHEDULE: resumed at step {step:,} < ramp_timesteps {int(ramp_timesteps):,} (warm-up "
                      f"{int(warmup_timesteps):,}): the RESUME cell attaches no warm-up or forward-weight ramp, so steps "
                      f"{step:,}-{shaping_end:,} trained at the full forward weight; restart the seed from the parent "
                      f"in a fresh RUN_ID")
    return step, None


def cell_path(out: Path, seed: int, steps: int, off_schedule: str | None, run: Path) -> Path:
    """Where one judgement goes: OUT/s<seed>/<steps>, or OUT/off_schedule/<run dir name>/s<seed>/<steps> for an
    off-schedule run judged with --report-off-schedule (pilot 1's, unchanged)."""
    if off_schedule:
        return out / "off_schedule" / run.name / f"s{seed}" / str(steps)
    return out / f"s{seed}" / str(steps)


def run_end(models: Path, budget: int | None, stage_dir: Path, pair_problem=None) -> dict:
    """Pilot 3's run-end record: the periodic and intact periodic pairs and where the run ended. pair_problem is the
    tree's checkpoint_pair_problem (None: not checked, the intact fields are None)."""
    steps = sorted(int(m.group(1)) for p in models.glob("stage2_*_steps.zip") if (m := PERIODIC_RE.match(p.name)))
    intact, broken = None, None
    if pair_problem is not None:
        intact, broken = [], {}
        for s in steps:
            vec = models / f"stage2_vecnormalize_{s}_steps.pkl"
            why = f"missing sidecar {vec.name}" if not vec.is_file() else pair_problem(
                models / f"stage2_{s}_steps.zip", vec)
            if why is None:
                intact.append(s)
            else:
                broken[str(s)] = str(why)
    final = models / FINAL_ZIP
    end, source = None, None
    if final.is_file():
        try:
            with zipfile.ZipFile(final) as z:
                end, source = int(json.loads(z.read("data"))["num_timesteps"]), "stage2_final.zip num_timesteps"
        except (KeyError, ValueError, zipfile.BadZipFile) as exc:
            source = f"stage2_final.zip unreadable: {type(exc).__name__}"
    if end is None and (stage_dir / "stage_summary.txt").is_file():
        m = re.search(r"^Timesteps:\s*([\d,]+)", (stage_dir / "stage_summary.txt").read_text(), re.M)
        if m:
            end, source = int(m.group(1).replace(",", "")), "stage_summary.txt Timesteps"
    return {"periodic_steps": steps, "newest_periodic_steps": steps[-1] if steps else None,
            "intact_periodic_steps": intact,
            "newest_intact_periodic_steps": (intact[-1] if intact else None) if intact is not None else None,
            "broken_periodic": broken,
            "final_num_timesteps": end, "final_source": source, "budget": budget,
            "stopped_early": None if end is None or budget is None else end < budget}


def uninterrupted_end(budget: int, rollout: int) -> int:
    """The num_timesteps an unbroken run ends at: whole rollouts at or past the budget (SB3 trains while
    num_timesteps < total_timesteps)."""
    if budget <= 0 or rollout <= 0:
        raise ValueError(f"budget {budget} and rollout {rollout} must be positive")
    return math.ceil(budget / rollout) * rollout


def _number_problem(where: str, value: object, want: float) -> str | None:
    if value is None:
        return f"{where} is absent (not {want!r})"
    if not is_number(value) or float(value) != float(want):
        return f"{where} is {value!r}, not {want!r}"
    return None


def lineage_problems(lineage: object) -> list[str]:
    """The judged zip's lineage record: mode initialize_next_stage, parent_stage 2, zeroed_command_parameters (a
    non-empty list of parameter names). Every problem is listed."""
    if lineage is None:
        return [f"the judged zip records no {LINEAGE_ATTRIBUTE} (no edge lineage: not a pilot-4 fine-tune)"]
    if not isinstance(lineage, dict):
        return [f"the judged zip's {LINEAGE_ATTRIBUTE} is not a record: {lineage!r}"]
    problems = []
    if lineage.get("mode") != LINEAGE_MODE:
        problems.append(f"the judged zip's lineage mode is {lineage.get('mode')!r}, not {LINEAGE_MODE!r}")
    stage = lineage.get("parent_stage")
    if isinstance(stage, bool) or not isinstance(stage, int) or stage != LINEAGE_PARENT_STAGE:
        problems.append(f"the judged zip's lineage parent_stage is {stage!r}, not {LINEAGE_PARENT_STAGE}")
    zeroed = lineage.get("zeroed_command_parameters")
    if zeroed is None:
        problems.append("the judged zip's lineage records no zeroed_command_parameters (the edge zeroed no command "
                        "columns)")
    elif (not isinstance(zeroed, list) or not zeroed
          or not all(isinstance(name, str) and name for name in zeroed)):
        problems.append(f"the judged zip's lineage zeroed_command_parameters is {zeroed!r}, not a non-empty list of "
                        f"parameter names")
    return problems


def entry_problems(sc: dict, lineage: object, lineage_note: str | None = None) -> list[str]:
    """Pilot 4's entry checks on a stage_config.json and the judged zip's lineage record (in place of pilot 3's
    warm_start_problems). Every problem is listed."""
    run_block = sc.get("run") or {}
    cur = sc.get("curriculum") or {}
    hyper = sc.get("hyperparameters") or {}
    problems = []
    if "critic_phase_dropped" in run_block:
        problems.append(f"run.critic_phase_dropped is recorded ({run_block['critic_phase_dropped']!r}): pilot 4's "
                        f"parents read no critic phase, so no drop is expected")
    for where, block in (("run", run_block), ("curriculum", cur)):
        if isinstance(block, dict) and "warm_start_log_std" in block:
            problems.append(f"{where}.warm_start_log_std is recorded ({block['warm_start_log_std']!r}): pilot 4 keeps "
                            f"each walker's own log_std")
    for where, value, want in (("curriculum.warmup_timesteps", cur.get("warmup_timesteps"), WARMUP_TIMESTEPS),
                               ("curriculum.ramp_start_value", cur.get("ramp_start_value"), RAMP_START_VALUE)):
        why = _number_problem(where, value, want)
        if why:
            problems.append(why)
    own = cur.get("initialize_from_own_checkpoint")
    if own is not True:
        problems.append(f"curriculum.initialize_from_own_checkpoint is {own!r}, not true")
    n_envs = run_block.get("n_envs")
    if isinstance(n_envs, bool) or n_envs != N_ENVS:
        problems.append(f"run.n_envs is {n_envs!r}, not {N_ENVS}")
    if not isinstance(hyper, dict):
        problems.append(f"the recorded hyperparameters are not a table: {hyper!r}")
    elif "critic_phase" in hyper:
        problems.append(f"the recorded hyperparameters hold a critic_phase key ({hyper['critic_phase']!r})")
    if norm_sha(run_block.get("parent_task_sha256")) != PARENT_TASK_SHA256:
        problems.append(f"run.parent_task_sha256 is {run_block.get('parent_task_sha256')!r}, not pilot 3's "
                        f"command-free task sha256:{PARENT_TASK_SHA256}")
    if lineage_note:
        problems.append(lineage_note)
    else:
        problems += lineage_problems(lineage)
    return problems


def env_problems(sc: dict, committed_table: object) -> list[str]:
    """The env binding: the recorded env kwargs (reward_weights) carry the live straight command (command_mode
    "heading" and the committed table key for key), the prey on the spawn line, the price and R6's four keys."""
    rw = sc.get("reward_weights")
    if not isinstance(rw, dict):
        return [f"stage_config.json records no env kwargs table (reward_weights: {rw!r})"]
    problems = []
    if rw.get("command_mode") != COMMAND_MODE:
        problems.append(f"reward_weights.command_mode is {rw.get('command_mode')!r}, not {COMMAND_MODE!r}")
    diff = table_differences(rw.get("command_config"), committed_table)
    if diff:
        problems.append("reward_weights.command_config is not the committed [env.command_config] table key for key: "
                        + "; ".join(diff))
    prey = rw.get("prey_lateral_range")
    if (not isinstance(prey, (list, tuple)) or len(prey) != len(PREY_LATERAL_RANGE)
            or not all(is_number(v) and float(v) == w for v, w in zip(prey, PREY_LATERAL_RANGE))):
        problems.append(f"reward_weights.prey_lateral_range is {prey!r}, not {list(PREY_LATERAL_RANGE)!r}")
    for key, want in (*PRICE_KEYS.items(), *R6_KEYS.items()):
        why = _number_problem(f"reward_weights.{key}", rw.get(key), want)
        if why:
            problems.append(why)
    return problems


def env_kwargs_drift(recorded: object, env_kwargs: dict, defaults: dict) -> dict:
    """The gait probe's own drift record (gait_probe.py lines 1651-1668), computed before the probe: for every key of
    the recorded env kwargs (stage_config.json reward_weights) or of this tree's committed [env], the value the env
    constructor receives here ({constructor defaults, **env_kwargs}, a tuple read as a list) against the recorded one,
    compared as the probe compares them (Python ==); {} when the run trained in this tree's env."""
    rec = recorded if isinstance(recorded, dict) else {}
    effective = {**defaults, **env_kwargs}
    drift = {}
    for k in sorted(set(rec) | set(env_kwargs)):
        a = effective.get(k, "<absent>")
        b = rec.get(k, "<absent>")
        a_cmp = list(a) if isinstance(a, tuple) else a
        if a_cmp != b:
            drift[k] = {"toml_main": a_cmp, "recorded_run": b}
    return drift


def drift_problem(drift: object) -> str | None:
    """None for an empty drift record; else the refusal (fix stage: the probe only warns)."""
    if not drift:
        return None
    return (f"the run's recorded env kwargs (stage_config.json reward_weights) differ from the env this tree builds "
            f"(constructor defaults and the committed [env]), so the probe would judge in another env than training's: "
            f"{json.dumps(drift, sort_keys=True, default=str)}")


def anchor_problems(hp_here: str, task_here: str, committed_table: object) -> list[str]:
    """Fix stage: the judging tree's own derivations against the text's values (checked at the pilot budget)."""
    problems = []
    if hp_here != HYPERPARAMETERS_SHA256:
        problems.append(f"this tree's recipe digest {hp_here} is not v3's {HYPERPARAMETERS_SHA256}")
    if task_here != TASK_SHA256:
        problems.append(f"this tree's task digest {task_here} is not v3's {TASK_SHA256}")
    if not same_json(committed_table, COMMAND_TABLE):
        problems.append("this tree's committed [env.command_config] is not v3's table: "
                        + "; ".join(table_differences(committed_table, COMMAND_TABLE)))
    return problems


def cadence_problems(sc: dict) -> list[str]:
    """The recorded cadence ("Run handling"): eval_freq 65536, eval_episodes 10, checkpoint_freq 250000,
    max_checkpoints 0."""
    cur = sc.get("curriculum") or {}
    problems = []
    for key, want in CADENCE.items():
        value = cur.get(key) if isinstance(cur, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value != want:
            problems.append(f"curriculum.{key} is {value!r}, not {want}")
    return problems


def parent_problems(seed: int, want_parent: str, recorded: object) -> list[str]:
    """--parent-sha256 must be the --seed's own 8M parent (PARENT_SHA256) and the run must record it."""
    problems = []
    own = PARENT_SHA256.get(seed)
    if own is None:
        problems.append(f"--seed {seed} is not a pilot-4 seed ({', '.join(str(s) for s in sorted(PARENT_SHA256))}): "
                        f"it has no parent in the table")
    elif want_parent != own:
        whose = [f"s{s}'s" for s, v in sorted(PARENT_SHA256.items()) if v == want_parent]
        problems.append(f"--parent-sha256 sha256:{want_parent} is not s{seed}'s own 8M parent sha256:{own}"
                        + (f" (it is {whose[0]})" if whose else ""))
    if norm_sha(recorded) != want_parent:
        problems.append(f"parent_checkpoint_sha256 {recorded} is not the chosen parent sha256:{want_parent}")
    return problems


def binding_problems(sc: dict, *, seed: int, want_parent: str, expect_budget: int, ramp: int, warmup: int,
                     ramp_start: object, forward_weight: object, hp_here: str, task_here: str,
                     committed_table: object, lineage: object, lineage_note: str | None = None,
                     report_off_schedule: bool, kwargs_drift: dict | None = None) -> tuple[list[str], dict]:
    """Every stage_config.json check that needs no tree (pilot 3's kept and pilot 4's), and what they read.

    ramp, warmup and ramp_start are this tree's curriculum ramp_timesteps, warmup_timesteps and ramp_start_value,
    forward_weight its [env] forward_vel_weight; hp_here and task_here the digests it derives; committed_table its
    [env.command_config]; lineage the judged zip's record (lineage_note: why it could not be read); kwargs_drift
    env_kwargs_drift's record (None: not computed). At the pilot budget the tree's digests and table must also be the
    text's (anchor_problems), as the cadence is checked there.
    Returns (problems, info) with info = {"budget", "resume_step", "off_schedule", "resumed", "cadence_checked"}."""
    run_block = sc.get("run") or {}
    problems = []
    if sc.get("species") != "tyrannosaurus_rex" or sc.get("embodiment") != "anatomical":
        problems.append(f"model {sc.get('species')}/{sc.get('embodiment')}")
    if sc.get("stage") not in (2, "2", "locomotion"):
        problems.append(f"stage {sc.get('stage')!r}")
    if run_block.get("load_mode") != "initialize_next_stage":
        problems.append(f"load_mode {run_block.get('load_mode')!r}")
    if run_block.get("seed") != seed:
        problems.append(f"seed {run_block.get('seed')!r} (not --seed {seed})")
    problems += parent_problems(seed, want_parent, run_block.get("parent_checkpoint_sha256"))
    budget = run_block.get("timesteps")
    resumed = run_block.get("resume_load_path")
    resume_step, off_schedule = resume_check(run_block, ramp, warmup, ramp_start, forward_weight)
    if resumed:  # the RESUME cell records the segment it trained; the budget is that plus the checkpoint it resumed
        budget = (budget or 0) + resume_step if resume_step is not None else None
    if budget != expect_budget:
        problems.append(f"budget {budget!r} (not {expect_budget})" + (f"; resumed from {resumed}" if resumed else ""))
    if off_schedule and not report_off_schedule:
        problems.append(off_schedule)
    if run_block.get("hyperparameters_sha256") != hp_here:
        problems.append(f"hyperparameters_sha256 {run_block.get('hyperparameters_sha256')} != this tree's {hp_here}")
    fp = sc.get("task_fingerprint")
    recorded_task = fp.get("task_sha256") if isinstance(fp, dict) else None
    if recorded_task is None:
        problems.append("stage_config.json records no task_fingerprint.task_sha256")
    elif recorded_task != task_here:
        problems.append(f"task_sha256 {recorded_task} != this tree's {task_here}")
    problems += entry_problems(sc, lineage, lineage_note)
    problems += env_problems(sc, committed_table)
    why_drift = drift_problem(kwargs_drift)
    if why_drift:
        problems.append(why_drift)
    cadence_checked = expect_budget == PILOT_BUDGET
    if cadence_checked:
        problems += cadence_problems(sc)
        problems += anchor_problems(hp_here, task_here, committed_table)
    return problems, {"budget": budget, "resume_step": resume_step, "off_schedule": off_schedule,
                      "resumed": bool(resumed), "cadence_checked": cadence_checked}


def final_fallback_problems(models: Path, *, steps: int, expect_budget: int, rollout: int, resumed: bool,
                            pair_problem) -> tuple[list[str], dict]:
    """Whether the final pair may stand in for the deciding checkpoint (--final-fallback). pair_problem is the tree's
    checkpoint_pair_problem. Returns (problems, record)."""
    problems = []
    end = uninterrupted_end(expect_budget, rollout)
    if expect_budget == PILOT_BUDGET and end != PILOT_UNINTERRUPTED_END:
        problems.append(f"the tree's rollout {rollout:,} ends the {PILOT_BUDGET:,}-step budget at {end:,}, not the "
                        f"text's uninterrupted end {PILOT_UNINTERRUPTED_END:,}")
    deciding = (models / f"stage2_{steps}_steps.zip", models / f"stage2_vecnormalize_{steps}_steps.pkl")
    if not deciding[0].exists():
        deciding_state = f"missing ({deciding[0].name})"
    else:
        deciding_state = pair_problem(*deciding)
    record = {"deciding_pair": [str(p) for p in deciding], "deciding_pair_problem": deciding_state,
              "uninterrupted_end": end, "rollout": rollout, "final_num_timesteps": None}
    if steps != expect_budget:
        problems.append(f"--final-fallback stands in only for the deciding checkpoint at the budget {expect_budget:,}, "
                        f"not --steps {steps:,}")
    if deciding_state is None:
        problems.append(f"the deciding pair {deciding[0].name} is intact: judge it, not the final pair")
    if resumed:
        problems.append("a resumed run's final pair does not end at the uninterrupted end; no fallback for it")
    final = models / FINAL_ZIP
    if final.is_file():
        try:
            record["final_num_timesteps"] = int(read_zip_data(final)["num_timesteps"])
        except (KeyError, ValueError, TypeError, zipfile.BadZipFile) as exc:
            problems.append(f"{FINAL_ZIP} has no readable num_timesteps ({type(exc).__name__}: {exc})")
    got = record["final_num_timesteps"]
    if got is not None and got != end:
        problems.append(f"{FINAL_ZIP} records num_timesteps {got:,}, not the uninterrupted end {end:,} of the "
                        f"{expect_budget:,}-step budget (whole rollouts of {rollout:,})")
    return problems, record


def other_run_cells(out: Path, seed: int, run_name: str) -> list[str]:
    """The cells under OUT/s<seed>/ judged from a run copy whose directory name is not run_name ("<steps> (<run>)");
    a cell whose binding.json cannot be read is listed too. Unfinished .partial folders are skipped (pilot 3's)."""
    found = []
    for b in sorted((out / f"s{seed}").glob("*/binding.json")):
        if b.parent.name.endswith(".partial"):
            continue
        try:
            name = Path(str(json.loads(b.read_text()).get("run") or "")).name
        except (OSError, ValueError, AttributeError) as exc:
            found.append(f"{b.parent.name} (unreadable binding.json: {type(exc).__name__})")
            continue
        if name != run_name:
            found.append(f"{b.parent.name} ({name or 'no run recorded'})")
    return found


def commit_problem(recorded: object, head: str) -> str | None:
    """None when stage_config.json's git_commit is 7 to 40 hex digits that begin the tree's HEAD; else why."""
    if recorded is None or str(recorded).strip() == "":
        return "stage_config.json records no git_commit"
    rec = str(recorded).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", rec) or not head.lower().startswith(rec):
        return f"git_commit {recorded} != tree HEAD {head}"
    return None


def zip_lineage(path: Path) -> tuple[object, str | None]:
    """(the judged zip's lineage record, or None; why it could not be read, or None)."""
    try:
        return read_zip_data(path).get(LINEAGE_ATTRIBUTE), None
    except (KeyError, ValueError, AttributeError, zipfile.BadZipFile) as exc:
        return None, f"the judged zip has no readable data member ({type(exc).__name__}: {exc})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent-sha256", required=True,
                    help="sha256 of the seed's own parent, its pilot-3 8M zip: "
                         + ", ".join(f"s{s} sha256:{v}" for s, v in sorted(PARENT_SHA256.items())))
    ap.add_argument("--expect-head", default=PILOT_HEAD,
                    help=f"the tree's HEAD must start with this (default: the head the runs record, {PILOT_HEAD})")
    ap.add_argument("--allow-commit-mismatch", action="store_true")
    ap.add_argument("--report-off-schedule", action="store_true",
                    help="judge an off-schedule resumed run anyway; binding.json marks it and the scorer skips it")
    ap.add_argument("--expect-budget", type=int, default=PILOT_BUDGET,
                    help="validation only: accept a run trained to another budget (the scorer then refuses the cell; "
                         "the cadence is recorded unchecked)")
    ap.add_argument("--final-fallback", action="store_true",
                    help="judge the final pair in place of a missing or broken deciding pair (criteria: final-model "
                         "fallback)")
    ap.add_argument("--check-only", action="store_true", help="run every check, then stop before the probe; write nothing")
    a = ap.parse_args()
    want_parent = norm_sha(a.parent_sha256)
    if want_parent is None:
        return refuse(f"--parent-sha256 {a.parent_sha256!r} is not a sha256")

    tree = Path.cwd().resolve()
    head = git(tree, "rev-parse", "HEAD")
    dirty = git(tree, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        return refuse(f"the tree {tree} has local edits")
    current = git(tree, "branch", "--show-current")
    head_refs = git(tree, "for-each-ref", "--points-at", "HEAD", "--format=%(refname)", "refs/heads/").splitlines()
    branch, branch_how = pilot_branch_from(current, head_refs)
    if branch is None:
        return refuse(branch_how)
    if a.expect_head and not head.startswith(a.expect_head):
        return refuse(f"tree HEAD {head} is not {a.expect_head}")

    import inspect

    import environments
    from environments.shared.config import hyperparameters_sha256, load_stage_config
    from environments.shared.curriculum import checkpoint_pair_problem
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.species_names import require_model_id
    from environments.shared.species_registry import get_species_config
    from environments.shared.task_fingerprint import stage_task_fingerprint

    if not str(Path(environments.__file__).resolve()).startswith(str(tree) + "/"):
        return refuse(f"environments imported from {environments.__file__}, not from {tree}")
    mid = require_model_id("tyrannosaurus_rex", "anatomical")
    run = Path(a.run).resolve()
    stage_dir = run / "03_locomotion"
    models = stage_dir / "models"
    if a.final_fallback:
        ckpt, vec = models / FINAL_ZIP, models / FINAL_VEC
    else:
        ckpt = models / f"stage2_{a.steps}_steps.zip"
        vec = models / f"stage2_vecnormalize_{a.steps}_steps.pkl"
    for p in (ckpt, vec, stage_dir / "stage_config.json", stage_dir / "plant_identity.json"):
        if not p.is_file():
            return refuse(f"checkpoint missing or incomplete run copy: {p}")
    pair_problem = checkpoint_pair_problem(ckpt, vec)
    if pair_problem is not None:
        return refuse(f"not an intact pair: {pair_problem}")
    sc = json.loads((stage_dir / "stage_config.json").read_text())
    run_block = sc.get("run", {})
    config = load_stage_config(mid, 2)
    cur = config.get("curriculum_kwargs", {})
    env_kwargs = config.get("env_kwargs", {})
    ramp, warmup = int(cur.get("ramp_timesteps", 0)), int(cur.get("warmup_timesteps", 0))
    ramp_start, forward_weight = cur.get("ramp_start_value"), env_kwargs.get("forward_vel_weight")
    committed_table = env_kwargs.get("command_config")
    rollout = N_ENVS * int(config.get("ppo_kwargs", {}).get("n_steps", 0))
    hp_here = hyperparameters_sha256(config, "ppo")
    task_here = stage_task_fingerprint(mid, 2, stage_config=config)["task_sha256"]
    lineage, lineage_note = zip_lineage(ckpt)
    env_class = get_species_config(mid).env_class
    defaults = {name: prm.default for name, prm in inspect.signature(env_class.__init__).parameters.items()
                if prm.default is not inspect.Parameter.empty}
    kwargs_drift = env_kwargs_drift(sc.get("reward_weights"), dict(env_kwargs), defaults)
    problems = []
    why_run = run_id_problem(run.name, a.seed)
    if why_run:
        problems.append(why_run)
    found, info = binding_problems(sc, seed=a.seed, want_parent=want_parent, expect_budget=a.expect_budget,
                                   ramp=ramp, warmup=warmup, ramp_start=ramp_start, forward_weight=forward_weight,
                                   hp_here=hp_here, task_here=task_here, committed_table=committed_table,
                                   lineage=lineage, lineage_note=lineage_note,
                                   report_off_schedule=a.report_off_schedule, kwargs_drift=kwargs_drift)
    problems += found
    off_schedule = info["off_schedule"]
    fallback = None
    if a.final_fallback:
        fb_problems, fallback = final_fallback_problems(models, steps=a.steps, expect_budget=a.expect_budget,
                                                        rollout=rollout, resumed=info["resumed"],
                                                        pair_problem=checkpoint_pair_problem)
        problems += fb_problems
    plant = current_plant_identity(mid).to_dict()
    rec_plant = json.loads((stage_dir / "plant_identity.json").read_text())
    for k in ("physics_sha256", "policy_interface_sha256"):
        if rec_plant.get(k) != plant.get(k):
            problems.append(f"plant {k} {rec_plant.get(k)} != this tree's {plant.get(k)}")
    commit_note = None
    why_commit = commit_problem(sc.get("git_commit"), head)
    if why_commit:
        if a.allow_commit_mismatch:
            commit_note = f"{why_commit} (allowed)"
        else:
            problems.append(why_commit)
    if problems:
        return refuse("; ".join(problems))

    cell = cell_path(Path(a.out).resolve(), a.seed, a.steps, off_schedule, run)
    others = [] if off_schedule else other_run_cells(Path(a.out).resolve(), a.seed, run.name)
    if others:
        return refuse(f"OUT/s{a.seed} holds cells judged from another run than {run.name}: {', '.join(others)}. A "
                      f"restarted seed's retired-run cells are moved, never deleted, to OUT/retired/<retired RUN_ID>/"
                      f"s{a.seed}/<steps>/ before its _b run is judged (criteria, Run handling); this refusal concerns "
                      f"OUT, not the run")
    if cell.exists():
        return refuse(f"{cell} exists (one judgement per checkpoint, never overwritten)")
    if a.check_only:
        print(f"all checks passed (--check-only: nothing rolled or written); would judge {ckpt} into {cell}"
              + (f"  [{off_schedule}]" if off_schedule else "")
              + ("  [final-model fallback]" if a.final_fallback else ""), flush=True)
        return 0
    tmp = cell.with_name(cell.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    run_id = f"s{a.seed}_{a.steps}" + ("_final" if a.final_fallback else "")
    cmd = [sys.executable, "-I", str(HERE / "iso.py"), str(HERE / "gait_probe_pilot.py"), "--embodiment", "anatomical",
           "--", *PROBE_ARGS, "--checkpoint", str(ckpt), "--vecnorm", str(vec), "--run-id", run_id,
           "--out", str(tmp), "--stage-config-json", str(stage_dir / "stage_config.json")]
    env = dict(os.environ, ISO_WT=str(tree), MUJOCO_GL=os.environ.get("MUJOCO_GL", "disable"))
    print("running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=tree, env=env)
    if rc != 0:
        shutil.rmtree(tmp)
        return refuse(f"the gait probe exited {rc} (3 = checkpoint refused by the plant contract or loader)")
    probe_json = tmp / f"tyrannosaurus_rex_{run_id}_locomotion.json"
    shutil.copyfile(probe_json, tmp / "probe.json")
    try:
        probe_drift = json.loads(probe_json.read_text())["meta"]["env_kwargs_drift_vs_recorded"]
    except (KeyError, TypeError, ValueError) as exc:
        shutil.rmtree(tmp)
        return refuse(f"the probe's JSON has no meta.env_kwargs_drift_vs_recorded ({type(exc).__name__}: {exc})")
    why_drift = drift_problem(probe_drift)
    if why_drift:  # fix stage: the probe only warns; the judge refuses (its own pre-probe check should have)
        shutil.rmtree(tmp)
        return refuse(f"{why_drift} (the probe's own record)")
    cur_rec = sc.get("curriculum") or {}
    rw = sc.get("reward_weights") or {}
    binding = {"schema": "walking-pilot4-judge/v1", "tree": str(tree), "branch": branch, "branch_how": branch_how,
               "head": head, "run": str(run), "run_id": run.name,
               "seed": a.seed, "steps": a.steps, "checkpoint": str(ckpt), "checkpoint_sha256": sha(ckpt),
               "vecnorm": str(vec), "vecnorm_sha256": sha(vec), "stage_config_run": run_block,
               "stage_config_git_commit": sc.get("git_commit"), "commit_note": commit_note,
               "parent_checkpoint_sha256": f"sha256:{want_parent}",
               "parent_task_sha256": run_block.get("parent_task_sha256"),
               "entry": {"warmup_timesteps": cur_rec.get("warmup_timesteps"),
                         "ramp_start_value": cur_rec.get("ramp_start_value"),
                         "initialize_from_own_checkpoint": cur_rec.get("initialize_from_own_checkpoint"),
                         "critic_phase_dropped_recorded": "critic_phase_dropped" in run_block,
                         "warm_start_log_std_recorded": "warm_start_log_std" in run_block
                         or "warm_start_log_std" in cur_rec},
               "lineage": lineage,
               "env_binding": {"command_mode": rw.get("command_mode"), "command_config": rw.get("command_config"),
                               "prey_lateral_range": rw.get("prey_lateral_range"),
                               **{k: rw.get(k) for k in (*PRICE_KEYS, *R6_KEYS)}},
               "committed_command_config": committed_table,
               "env_kwargs_drift": kwargs_drift, "probe_env_kwargs_drift": probe_drift,
               "text_anchors_checked": info["cadence_checked"],
               "task_sha256": task_here,
               "n_envs": run_block.get("n_envs"),
               "curriculum_cadence": {k: cur_rec.get(k) for k in CADENCE_KEYS},
               "cadence_checked": info["cadence_checked"],
               "hyperparameters_sha256": hp_here, "plant": {k: plant.get(k) for k in ("physics_revision", "physics_sha256",
                                                                                     "policy_interface_sha256")},
               "ramp_timesteps": ramp, "warmup_timesteps": warmup, "ramp_start_value": ramp_start,
               "forward_vel_weight": forward_weight,
               "shaping_inert": shaping_inert(warmup, ramp_start, forward_weight),
               "resume_step": info["resume_step"],
               "off_schedule": bool(off_schedule), "off_schedule_reason": off_schedule,
               "final_fallback": bool(a.final_fallback), "final_fallback_record": fallback,
               "run_end": run_end(models, a.expect_budget, stage_dir, checkpoint_pair_problem),
               "probe_args": PROBE_ARGS, "expected_budget": a.expect_budget,
               "validation_only": a.expect_budget != PILOT_BUDGET}
    (tmp / "binding.json").write_text(json.dumps(binding, indent=1, default=str))
    tmp.rename(cell)
    print(f"wrote {cell}" + (f"  [{off_schedule}]" if off_schedule else "")
          + ("  [final-model fallback]" if a.final_fallback else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
