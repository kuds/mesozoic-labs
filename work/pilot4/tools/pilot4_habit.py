"""Walking pilot 4's habit readout, REPORT ONLY: ledger alarms, the PARK/DRIVEN trace, the head tip.

Scratch, never merged: walking pilot 3's pilot3_habit.py (sha256 56f1c78d...) adapted as the pilot-4 pre-registration's
TOOL PLAN says (criteria_pilot4 draft v3: "TOOL PLAN", D10, "Report only": "The habit readout (pilot 3's definitions
[P3 L797-847], on this run's ledger, 30 evaluations from 65,536; the trace at the deciding checkpoint); it picks
nothing"). Built and pinned by sha256 before launch. REPORT ONLY: pilot 4 has no Q4 question, so this readout picks
nothing and never the outcome (pilot4_score.py --habit prints it report only). Changed against pilot 3's file: the
evaluation cadence EVAL_EVERY 65,536 (the window's first evaluation, its spacing and the coverage rule), TD_BLOCK
500,000, the ledger and RUN_ID paths (pilot4_walk_s<seed>[_b]), the schema and the report-only wording. Every
definition (groups, the bar, UNRESOLVED, FLAGGED, ALARM, POSSIBLE ALARM, the trace, CMD, PARK, DRIVEN, the seed reading
and the refusals) is pilot 3's, unchanged.

Groups (the r10 stance rounds' actuator numbering, stance6 "CLASSES PER RUN"): TAIL 11-14 (tail pitch 11, 13, 14),
NECK 0-2 (neck pitch, neck yaw, head pitch), LEGS 3-10 (printed, never a habit).

LEDGER (plain Python; the scored run's 03_locomotion/eval_episodes.csv; for a _b restart the _b run's alone, a retired
run's ledger is printed with --retired-ledger, never read):
  * Row sets: rows are grouped by their timesteps in file order; a second block of rows for a step already seen, or a
    repeated episode index within one block, starts a new row set. Where a step has more than one row set (a resume),
    the last-written set is read and the earlier ones printed as superseded.
  * Window: every evaluation from the first to the last at or before --through (the deciding checkpoint's steps; at
    2M all 30 evaluations 65,536-1,966,080; pilot 4 has no warm-up). An evaluation with other than 10 rows is printed
    and reads as not flagged (the criteria name fewer than 10; more than 10 cannot arise at eval_episodes 10).
  * Over the bar on a group: saturated_actuator names an actuator of the group and saturated_actuator_fraction > 0.10
    (the share of steps at |command| >= 0.99, stance6's M5 reading). UNRESOLVED for a group: over 0.10 with an actuator
    of another group named and saturated_actuator_ties >= 1 or actuators_saturated >= 2.
  * FLAGGED evaluation (TAIL or NECK): at least 5 of its 10 episodes over the bar on that group. ALARM: the second of
    two consecutive flagged evaluations of the window (consecutive in the window's list of read evaluations). An alarm
    that forms only when the UNRESOLVED episodes are counted as the group's is a POSSIBLE ALARM, settled by the trace:
    a habit when at least 2 of the trace's full-horizon episodes put one actuator of that group over the bar.
  * Report only: the same reading for LEGS and for ANY actuator (the stance rounds' Statistic A input: an ANY-flagged
    evaluation has at least 5 of its 10 episodes over the bar on any actuator); TD, the mean over the window's
    evaluations of |the evaluation's mean tail_3 pitch command| (tail_3_pitch_act_command_mean), per TD_BLOCK
    (500,000) steps and overall; the head tip, neck pitch and head pitch per evaluation (episode means).
TRACE (needs the tree: run through iso.py on the clean clone at the judged head): 20 deterministic episodes at reset
seeds 3042-3061 of the deciding checkpoint with its sidecar, rolled by main's episode-trace library
(environments/shared/reporting/episode_trace.py: load_trace_policy and trace_episode, the stance judge's rollout:
frozen VecNormalize, predict(deterministic=True) and its clip) in the tree's locomotion env, every actuator's command
per step. Full-horizon = 1,000 steps. CMD_a = the mean, over the full-horizon episodes, of each episode's mean command
of actuator a over steps 100-999 (the gait window). PARK: CMD_a >= +0.60 for a tail pitch actuator. DRIVEN: |CMD_a| >=
0.60 for any tail actuator PARK does not count. With fewer than 10 full-horizon episodes CMD is not formed: NOT RUN.
Per episode and actuator, the bar: the share of window steps at |command| >= 0.99 above 0.10. Report only: the head
tip's height (the head_tip site's world z after each control step; the ledger samples it one physics step earlier)
and the neck and head pitch joint angles, window means.
SEED READING (report only): HABIT when it has a TAIL ALARM, a NECK ALARM, a POSSIBLE ALARM settled as a habit, a PARK
or a DRIVEN tail; else NOT RUN when any part could not be read (a ledger column, the trace, CMD, a POSSIBLE ALARM with
no trace to settle it); else NO HABIT. (A habit found is not undone by an unread part [inferred: "HABIT when it has
..."].)

    python3 pilot4_habit.py ledger LEDGER.csv [--through STEPS] [--json OUT.json]
    cd TREE && ISO_WT=TREE python3 -I iso.py pilot4_habit.py seed --seed 42
        --ledger RUN/03_locomotion/eval_episodes.csv --cell OUT/s42/2000000 [--retired-ledger OLD.csv]
        --json HABIT/s42.json
    cd TREE && ISO_WT=TREE python3 -I iso.py pilot4_habit.py trace --checkpoint ZIP --vecnorm PKL [--json OUT.json]
RUN is the judged run copy, named by its RUN_ID (pilot4_walk_s<seed>, or pilot4_walk_s<seed>_b for a restart).
--cell is the judged cell (pilot4_judge.py's OUT/s<seed>/<steps>): its binding.json names the checkpoint pair (the
final pair under the fallback), whose sha256s must still match, and the head, which must be this tree's. Pilot 3's
refusals, kept: seed refuses, writing nothing, when the tree has local edits (git status --porcelain
--untracked-files=no, as the judge), when the cell is off-schedule, when it is validation-only (unless --validation,
for checks only: its record says validation_only), and when --ledger is not that very file of the judged copy: its
resolved path must equal binding "run" (the run copy the judge read, resolved) / 03_locomotion / eval_episodes.csv
(a retired run's ledger, a same-named copy elsewhere, or another file in the run's 03_locomotion/, is refused). After
reading the ledger and before the trace, seed also refuses, writing nothing, a ledger whose window does not cover the
deciding checkpoint (window_problems): its first evaluation not at EVAL_EVERY (65,536), or its last at or before the
deciding steps 65,536 or more short of them (a stale download or a lagging sync that stops early, or a copy that lost
its first rows). A validation-only cell read with --validation is checked on its run's recorded curriculum eval_freq
(the smoke's 32,768) instead. Every mode refuses an existing --json target (exit 2, nothing written): a record is
never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ACTUATORS = ("neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
             "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act", "tail_1_pitch_act",
             "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")
GROUPS = {"NECK": (0, 1, 2), "LEGS": tuple(range(3, 11)), "TAIL": (11, 12, 13, 14)}
HABIT_GROUPS = ("TAIL", "NECK")
TAIL_PITCH = (11, 13, 14)
BAR = 0.10
SATURATION_ABS = 0.99
FLAG_MIN = 5
EPISODES_PER_EVAL = 10
EVAL_EVERY = 65_536  # pilot 4's evaluation cadence: the window's first evaluation, its spacing and the coverage rule
PARK_MIN = 0.60
TRACE_SEEDS = tuple(range(3042, 3062))
TRACE_WINDOW = (100, 1000)  # steps 100-999
HORIZON = 1000
FULL_MIN = 10
POSSIBLE_SETTLE_MIN = 2
TD_BLOCK = 500_000  # pilot 4: TD per 500,000 steps (pilot 3's 2,000,000 would be the whole 2M run)
LEDGER_REQUIRED = ("timesteps", "episode", "saturated_actuator", "saturated_actuator_fraction",
                   "saturated_actuator_ties", "actuators_saturated")
TD_COLUMN = "tail_3_pitch_act_command_mean"
POSTURE_COLUMNS = ("head_tip_height_mean_m", "neck_pitch_mean_deg", "head_pitch_mean_deg")
SCHEMA = "walking-pilot4-habit/v1"
HEAD_TIP_REFS = "pilot 3's stance parent 0.904 m, the statue 0.959 m"  # pilot 3's printed references, relabelled


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def actuator_index(name: str) -> int | None:
    """The actuator a ledger name means: its index in ACTUATORS, or N for 'actuator[N]' / 'actuator_N'; else None."""
    name = (name or "").strip()
    if name in ACTUATORS:
        return ACTUATORS.index(name)
    for prefix, suffix in (("actuator[", "]"), ("actuator_", "")):
        if name.startswith(prefix) and name.endswith(suffix):
            core = name[len(prefix): len(name) - len(suffix) if suffix else None]
            if core.isdigit():
                return int(core)
    return None


def group_of(index: int | None) -> str | None:
    for g, members in GROUPS.items():
        if index in members:
            return g
    return None


# ---------------------------------------------------------------- the ledger


def read_ledger(path: "str | Path") -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def row_sets(rows: list[dict]) -> dict[int, list[list[dict]]]:
    """{step: [row set, ...]} in file order. A new set starts when a step's rows resume after another step's, or when
    an episode index repeats inside the current block."""
    sets: dict[int, list[list[dict]]] = {}
    prev_step = None
    for row in rows:
        try:
            step = int(float(row["timesteps"]))
        except (TypeError, ValueError):
            continue
        current = sets.setdefault(step, [])
        episode = row.get("episode")
        if prev_step != step or not current or any(r.get("episode") == episode for r in current[-1]):
            current.append([])
        current[-1].append(row)
        prev_step = step
    return sets


def episode_state(row: dict) -> dict:
    """One ledger row: its named actuator, group, fraction and whether it is over the bar (per group) or UNRESOLVED."""
    frac = _f(row.get("saturated_actuator_fraction"))
    index = actuator_index(row.get("saturated_actuator", ""))
    grp = group_of(index)
    over = math.isfinite(frac) and frac > BAR
    ties, sat = _f(row.get("saturated_actuator_ties")), _f(row.get("actuators_saturated"))
    ambiguous = (math.isfinite(ties) and ties >= 1) or (math.isfinite(sat) and sat >= 2)
    return {"actuator": index, "group": grp, "fraction": frac, "over": over,
            "over_group": {g: over and grp == g for g in GROUPS},
            "unresolved": {g: over and grp is not None and grp != g and ambiguous for g in GROUPS}}


def _alarms(flags: list[bool]) -> list[int]:
    """Indices that are the second of two consecutive flagged evaluations."""
    return [i for i in range(1, len(flags)) if flags[i] and flags[i - 1]]


def ledger_reading(header: list[str], rows: list[dict], through: int | None = None) -> dict:
    """The ledger part of the habit readout (every number it prints; module docstring)."""
    missing = [c for c in LEDGER_REQUIRED if c not in header]
    out: dict = {"columns_missing": missing, "read": not missing}
    if missing:
        out["not_run"] = [f"ledger column(s) missing: {missing}"]
        return out
    sets = row_sets(rows)
    steps = sorted(s for s in sets if through is None or s <= through)
    out["after_window"] = sorted(s for s in sets if through is not None and s > through)
    superseded = {s: [len(x) for x in sets[s][:-1]] for s in steps if len(sets[s]) > 1}
    evals = []
    for s in steps:
        chosen = sets[s][-1]
        states = [episode_state(r) for r in chosen]
        n = len(chosen)
        ev = {"step": s, "rows": n, "complete": n == EPISODES_PER_EVAL,
              "over": {g: sum(st["over_group"][g] for st in states) for g in GROUPS},
              "unresolved": {g: sum(st["unresolved"][g] for st in states) for g in GROUPS},
              "over_any": sum(st["over"] for st in states),
              "named_over": sorted({ACTUATORS[st["actuator"]] if st["actuator"] is not None and st["actuator"] <
                                    len(ACTUATORS) else str(st["actuator"]) for st in states if st["over"]})}
        for col in (TD_COLUMN, *POSTURE_COLUMNS):
            vals = [_f(r.get(col)) for r in chosen] if col in header else []
            finite = [v for v in vals if math.isfinite(v)]
            ev[col] = sum(finite) / len(finite) if finite else None
        evals.append(ev)
    out["evaluations"] = evals
    out["superseded_row_sets"] = {str(s): v for s, v in superseded.items()}
    out["incomplete_evaluations"] = [(e["step"], e["rows"]) for e in evals if not e["complete"]]
    gaps = [(evals[i - 1]["step"], evals[i]["step"]) for i in range(1, len(evals))
            if evals[i]["step"] - evals[i - 1]["step"] != EVAL_EVERY]
    out["grid_gaps"] = gaps
    groups = {}
    for g in (*GROUPS, "ANY"):
        if g == "ANY":
            flags = [e["complete"] and e["over_any"] >= FLAG_MIN for e in evals]
            flags_u = flags
        else:
            flags = [e["complete"] and e["over"][g] >= FLAG_MIN for e in evals]
            flags_u = [e["complete"] and e["over"][g] + e["unresolved"][g] >= FLAG_MIN for e in evals]
        alarms = [evals[i]["step"] for i in _alarms(flags)]
        possible = [evals[i]["step"] for i in _alarms(flags_u) if evals[i]["step"] not in alarms]
        groups[g] = {"flagged": [e["step"] for e, f in zip(evals, flags, strict=True) if f], "alarms": alarms,
                     "possible_alarms": possible,
                     "unresolved_episodes": sum(e["unresolved"].get(g, 0) for e in evals) if g != "ANY" else 0}
    out["groups"] = groups
    out["window"] = {"first": steps[0] if steps else None, "last": steps[-1] if steps else None,
                     "evaluations": len(steps), "through": through}
    if TD_COLUMN in header:
        tds = [(e["step"], abs(e[TD_COLUMN])) for e in evals if e[TD_COLUMN] is not None]
        blocks: dict = {}
        for s, v in tds:
            k = (s - 1) // TD_BLOCK
            blocks.setdefault(f"{k * TD_BLOCK + 1}-{(k + 1) * TD_BLOCK}", []).append(v)
        out["td"] = {"overall": sum(v for _, v in tds) / len(tds) if tds else None, "evaluations": len(tds),
                     "block_steps": TD_BLOCK, "per_block": {k: sum(v) / len(v) for k, v in blocks.items()}}
    else:
        out["td"] = {"overall": None, "note": f"no {TD_COLUMN} column (TD not read; report only)"}
    return out


# ---------------------------------------------------------------- the trace


def trace_reading(episodes: list[dict]) -> dict:
    """CMD, PARK, DRIVEN and the per-episode bar from traced episodes: [{"seed", "length", "commands": (T, 15)
    nested list or array, optional "head_tip_z", "neck_pitch", "head_pitch": (T,)}]."""
    lo, hi = TRACE_WINDOW
    full = [e for e in episodes if int(e["length"]) == HORIZON]
    per_episode = []
    for e in episodes:
        cmds = [list(map(float, row)) for row in e["commands"]]
        rec = {"seed": e["seed"], "length": int(e["length"]), "full_horizon": int(e["length"]) == HORIZON}
        window = cmds[lo:hi] if rec["full_horizon"] else cmds[lo:]
        if window:
            n = len(window)
            rec["window_mean_command"] = [sum(r[a] for r in window) / n for a in range(len(ACTUATORS))]
            rec["saturated_share"] = [sum(abs(r[a]) >= SATURATION_ABS for r in window) / n for a in range(len(ACTUATORS))]
        else:
            rec["window_mean_command"] = rec["saturated_share"] = None
        for key in ("head_tip_z", "neck_pitch", "head_pitch"):
            series = e.get(key)
            if series is not None and len(series) > lo:
                vals = [float(v) for v in list(series)[lo:hi]]
                rec[f"{key}_window_mean"] = sum(vals) / len(vals)
        per_episode.append(rec)
    out = {"episodes": per_episode, "full_horizon": len(full), "cmd_formed": len(full) >= FULL_MIN}
    if not out["cmd_formed"]:
        out["not_run"] = [f"only {len(full)} of {len(episodes)} traced episodes are full-horizon (< {FULL_MIN}): "
                          "CMD not formed"]
        return out
    fulls = [r for r in per_episode if r["full_horizon"]]
    cmd = [sum(r["window_mean_command"][a] for r in fulls) / len(fulls) for a in range(len(ACTUATORS))]
    out["cmd"] = {ACTUATORS[a]: cmd[a] for a in range(len(ACTUATORS))}
    out["park"] = [{"actuator": ACTUATORS[a], "cmd": cmd[a]} for a in TAIL_PITCH if cmd[a] >= PARK_MIN]
    parked = {p["actuator"] for p in out["park"]}
    out["driven"] = [{"actuator": ACTUATORS[a], "cmd": cmd[a]} for a in GROUPS["TAIL"]
                     if abs(cmd[a]) >= PARK_MIN and ACTUATORS[a] not in parked]
    bar_counts = {ACTUATORS[a]: sum(1 for r in fulls if r["saturated_share"][a] > BAR) for a in range(len(ACTUATORS))}
    out["episodes_over_bar"] = bar_counts
    out["settles"] = {g: [ACTUATORS[a] for a in GROUPS[g] if bar_counts[ACTUATORS[a]] >= POSSIBLE_SETTLE_MIN]
                      for g in GROUPS}
    for key in ("head_tip_z", "neck_pitch", "head_pitch"):
        vals = [r[f"{key}_window_mean"] for r in fulls if f"{key}_window_mean" in r]
        out[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
    return out


def roll_trace(checkpoint: str, vecnorm: str, seeds=TRACE_SEEDS) -> list[dict]:
    """Roll the trace with the tree's own episode-trace library (needs `environments`, from the tree)."""
    import gymnasium as gym
    import numpy as np
    from environments.shared.config import load_stage_config
    from environments.shared.plant_contract import (
        current_plant_identity,
        validate_environment_plant,
    )
    from environments.shared.reporting.episode_trace import (
        ActuatorLayout,
        load_trace_policy,
        trace_episode,
    )
    from environments.shared.species_names import require_model_id
    from environments.shared.species_registry import env_class as model_env_class

    mid = require_model_id("tyrannosaurus_rex", "anatomical")
    stage_config = load_stage_config(mid, 2)
    env_kwargs = dict(stage_config["env_kwargs"])
    env_class = model_env_class(mid)
    plant = current_plant_identity(mid)
    policy = load_trace_policy(checkpoint, vecnorm, lambda: env_class(**env_kwargs), plant_identity=plant)
    base = env_class(**env_kwargs)

    class _Posture(gym.Wrapper):
        """Pass-through; records the head tip's world z after each control step (report only)."""

        def __init__(self, env):
            super().__init__(env)
            import mujoco

            m = env.unwrapped.model
            self.site = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "head_tip"))
            self.z: list[float] = []

        def reset(self, **kw):
            self.z = []
            return self.env.reset(**kw)

        def step(self, action):
            out = self.env.step(action)
            d = self.env.unwrapped.data
            self.z.append(float(d.site_xpos[self.site][2]) if self.site >= 0 else math.nan)
            return out

    env = _Posture(base)
    episodes = []
    try:
        validate_environment_plant(base, plant, artifact="pilot4_habit trace environment")
        layout = ActuatorLayout.from_model(base.unwrapped.model)
        if tuple(layout.names) != ACTUATORS:
            raise SystemExit(f"the env's actuators {layout.names} are not this tool's {ACTUATORS}")
        for seed in seeds:
            tr = trace_episode(env, policy, seed=int(seed), layout=layout)
            cmd = np.asarray(tr.values["command"], dtype=np.float64)
            jp = np.asarray(tr.values["joint_pos"], dtype=np.float64)
            episodes.append({"seed": int(seed), "length": tr.length, "commands": cmd.tolist(),
                             "head_tip_z": list(env.z), "neck_pitch": np.degrees(jp[:, 0]).tolist(),
                             "head_pitch": np.degrees(jp[:, 2]).tolist(), "return": tr.episode_return,
                             "termination_reason": tr.termination_reason})
    finally:
        env.close()
    return episodes


# ---------------------------------------------------------------- the seed reading


def seed_reading(ledger: dict, trace: "dict | None") -> dict:
    """HABIT / NO HABIT / NOT RUN and the habits found (each with its group and source)."""
    habits, not_run = [], list(ledger.get("not_run") or [])
    if trace is None:
        not_run.append("no trace")
    elif not trace.get("cmd_formed"):
        not_run += trace.get("not_run") or ["CMD not formed"]
    if ledger.get("read"):
        for g in HABIT_GROUPS:
            gr = ledger["groups"][g]
            if gr["alarms"]:
                habits.append({"kind": f"{g} ALARM", "group": g, "source": "ledger", "steps": gr["alarms"],
                               "text": f"{g} ALARM at {', '.join(f'{s:,}' for s in gr['alarms'])}"})
            if gr["possible_alarms"]:
                if trace is None or not trace.get("cmd_formed"):
                    not_run.append(f"{g} POSSIBLE ALARM at {gr['possible_alarms']} with no trace to settle it")
                elif trace["settles"][g]:
                    habits.append({"kind": f"{g} POSSIBLE ALARM settled by trace", "group": g,
                                   "source": "ledger+trace", "steps": gr["possible_alarms"],
                                   "actuators": trace["settles"][g],
                                   "text": f"{g} POSSIBLE ALARM at {', '.join(f'{s:,}' for s in gr['possible_alarms'])}"
                                           f", settled by the trace ({', '.join(trace['settles'][g])} over the bar in "
                                           f">= {POSSIBLE_SETTLE_MIN} full-horizon episodes)"})
    if trace is not None and trace.get("cmd_formed"):
        for p in trace["park"]:
            habits.append({"kind": "PARK", "group": "TAIL", "source": "trace", "actuator": p["actuator"],
                           "cmd": p["cmd"], "text": f"PARK {p['actuator']} CMD {p['cmd']:+.3f}"})
        for p in trace["driven"]:
            habits.append({"kind": "DRIVEN", "group": "TAIL", "source": "trace", "actuator": p["actuator"],
                           "cmd": p["cmd"], "text": f"DRIVEN {p['actuator']} CMD {p['cmd']:+.3f}"})
    reading = "HABIT" if habits else ("NOT RUN" if not_run else "NO HABIT")
    return {"reading": reading, "habits": habits, "not_run": not_run,
            "tail_habit": any(h["group"] == "TAIL" for h in habits),
            "neck_habit": any(h["group"] == "NECK" for h in habits)}


# ---------------------------------------------------------------- printing and the CLI


def sha256_file(path: "str | Path") -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe_ledger(lr: dict, label: str) -> list[str]:
    if not lr.get("read"):
        return [f"{label}: NOT READ ({'; '.join(lr.get('not_run') or [])})"]
    w = lr["window"]
    lines = [f"{label}: {w['evaluations']} evaluations {w['first']:,}-{w['last']:,}"
             + (f" (through {w['through']:,}; {len(lr['after_window'])} after it not read)" if w["through"] else "")
             if w["evaluations"] else f"{label}: no evaluation in the window"]
    for g in (*HABIT_GROUPS, "LEGS", "ANY"):
        gr = lr["groups"][g]
        tag = "" if g in HABIT_GROUPS else " (not part of the seed reading)"
        lines.append(f"  {g}{tag}: {len(gr['flagged'])} flagged evaluations; {len(gr['alarms'])} ALARMS"
                     + (f" ({gr['alarms'][0]:,}-{gr['alarms'][-1]:,})" if gr["alarms"] else "")
                     + (f"; POSSIBLE ALARMS at {gr['possible_alarms']}" if gr["possible_alarms"] else "")
                     + (f"; {gr['unresolved_episodes']} UNRESOLVED episodes" if gr["unresolved_episodes"] else ""))
    if lr["superseded_row_sets"]:
        lines.append(f"  superseded row sets (the last-written set read): {lr['superseded_row_sets']}")
    if lr["incomplete_evaluations"]:
        lines.append(f"  evaluations with other than 10 rows (not flagged): {lr['incomplete_evaluations']}")
    if lr["grid_gaps"]:
        lines.append(f"  evaluation steps not {EVAL_EVERY:,} apart (printed): {lr['grid_gaps'][:6]}")
    td = lr["td"]
    if td.get("overall") is not None:
        lines.append(f"  TD {td['overall']:.4f} over {td['evaluations']} evaluations; per {TD_BLOCK:,} steps: "
                     + ", ".join(f"{k} {v:.4f}" for k, v in td["per_block"].items())
                     + "  (the PHASE runs in stance 0.008-0.017)")
    else:
        lines.append(f"  TD not read ({td.get('note')})")
    tips = [e["head_tip_height_mean_m"] for e in lr["evaluations"] if e.get("head_tip_height_mean_m") is not None]
    if tips:
        lines.append(f"  head tip per evaluation: first {tips[0]:.3f} m, last {tips[-1]:.3f} m, min {min(tips):.3f} m, "
                     f"max {max(tips):.3f} m ({HEAD_TIP_REFS})")
    return lines


def describe_trace(tr: dict) -> list[str]:
    lines = [f"trace: {len(tr['episodes'])} episodes (seeds {tr['episodes'][0]['seed']}-{tr['episodes'][-1]['seed']}), "
             f"{tr['full_horizon']} full-horizon"]
    if not tr["cmd_formed"]:
        return lines + [f"  {'; '.join(tr['not_run'])}"]
    lines.append("  CMD (tail): " + ", ".join(f"{ACTUATORS[a]} {tr['cmd'][ACTUATORS[a]]:+.3f}" for a in GROUPS["TAIL"]))
    lines.append("  CMD (neck): " + ", ".join(f"{ACTUATORS[a]} {tr['cmd'][ACTUATORS[a]]:+.3f}" for a in GROUPS["NECK"]))
    lines.append(f"  PARK {[p['actuator'] for p in tr['park']] or 'none'}; DRIVEN {[p['actuator'] for p in tr['driven']] or 'none'}")
    over = {k: v for k, v in tr["episodes_over_bar"].items() if v}
    lines.append(f"  full-horizon episodes over the bar, per actuator: {over or 'none'}")
    if tr.get("head_tip_z_mean") is not None:
        lines.append(f"  head tip {tr['head_tip_z_mean']:.3f} m, neck pitch {tr['neck_pitch_mean']:.1f} deg, head pitch "
                     f"{tr['head_pitch_mean']:.1f} deg (window means; {HEAD_TIP_REFS})")
    return lines


def seed_problems(binding: dict, *, seed: int, head: str, dirty: str, ledger: "str | Path",
                  validation: bool = False) -> list[str]:
    """Pilot 3's FIX2/FIX3 rule: every reason the seed mode refuses a cell before reading anything (each one listed)."""
    problems = []
    if binding.get("seed") != seed:
        problems.append(f"the cell judged seed {binding.get('seed')}, not --seed {seed}")
    if binding.get("head") != head:
        problems.append(f"the cell was judged at {binding.get('head')}, this tree is {head}")
    if dirty:
        problems.append("this tree has local edits (git status --porcelain --untracked-files=no is not empty)")
    if binding.get("off_schedule"):
        problems.append(f"the cell is off-schedule ({binding.get('off_schedule_reason')}): never scored, never read")
    if binding.get("validation_only") and not validation:
        problems.append(f"the cell is validation-only (run budget {binding.get('expected_budget')}); --validation "
                        "reads it for a check of the tool, never as the seed's readout")
    run = binding.get("run")
    own = Path(str(run)).resolve() / "03_locomotion" / "eval_episodes.csv" if run else None
    if own is None or Path(ledger).resolve() != own:  # FIX3: the judged copy's own file, not a same-named one
        problems.append(f"--ledger {ledger} is not the judged run's own ledger ({own or '<no run recorded>'}, in "
                        "the run copy the cell's binding.json names)")
    return problems


def window_grid(binding: dict, validation: bool = False) -> int:
    """The evaluation spacing window_problems checks against: pilot 4's EVAL_EVERY (65,536); for a validation-only
    cell read with --validation, its run's recorded curriculum eval_freq when that is a positive integer."""
    if validation and binding.get("validation_only"):
        freq = (binding.get("curriculum_cadence") or {}).get("eval_freq")
        if isinstance(freq, int) and not isinstance(freq, bool) and freq > 0:
            return freq
    return EVAL_EVERY


def window_problems(lr: dict, deciding_steps: int, grid: int = EVAL_EVERY) -> list[str]:
    """Why the ledger's window does not cover the deciding checkpoint (pilot 3's "Window" with pilot 4's cadence: every
    evaluation from the first, 65,536, to the last at or before the deciding checkpoint): no evaluation in it, a first
    evaluation other than `grid`, or a last one `grid` or more short of the deciding steps. [] when it covers it, and
    for a ledger whose columns could not be read (that already reads NOT RUN)."""
    if not lr.get("read"):
        return []
    w = lr["window"]
    if not w["evaluations"]:
        return [f"the ledger has no evaluation at or before the deciding checkpoint {deciding_steps:,}"]
    problems = []
    if w["first"] != grid:
        problems.append(f"the ledger's first evaluation is {w['first']:,}, not {grid:,} (its first rows are missing)")
    short = int(deciding_steps) - w["last"]
    if short >= grid:
        problems.append(f"the ledger's last evaluation at or before the deciding checkpoint {int(deciding_steps):,} is "
                        f"{w['last']:,}, {short:,} steps short ({grid:,} or more: a stale or partial copy)")
    return problems


def refuse_existing_json(path: "str | None") -> "str | None":
    """FIX3: a --json target that already exists is refused (a record is never overwritten)."""
    if path and Path(path).exists():
        return f"--json {path} exists (never overwritten; move it or name another file)"
    return None


def _tool_sha() -> str:
    return sha256_file(__file__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    pl = sub.add_parser("ledger", help="the ledger part alone (plain Python)")
    pl.add_argument("ledger")
    pl.add_argument("--through", type=int, default=None)
    pl.add_argument("--json", default=None)
    ps = sub.add_parser("seed", help="one seed's reading: ledger + trace at the judged cell (needs the tree)")
    ps.add_argument("--seed", type=int, required=True)
    ps.add_argument("--ledger", required=True)
    ps.add_argument("--cell", required=True, help="the judged cell OUT/s<seed>/<steps> (its binding.json)")
    ps.add_argument("--retired-ledger", action="append", default=[], help="a retired run's ledger: printed, never read")
    ps.add_argument("--json", required=True)
    ps.add_argument("--validation", action="store_true",
                    help="read a validation-only cell (a check of the tool, never the seed's readout; the record says "
                         "so)")
    pt = sub.add_parser("trace", help="the trace alone on a checkpoint pair (e.g. the parent: the habit baseline)")
    pt.add_argument("--checkpoint", required=True)
    pt.add_argument("--vecnorm", required=True)
    pt.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    why = refuse_existing_json(a.json)
    if why:
        print(f"refusing (nothing written): {why}", flush=True)
        return 2

    if a.mode == "ledger":
        header, rows = read_ledger(a.ledger)
        lr = ledger_reading(header, rows, a.through)
        print("\n".join(describe_ledger(lr, f"ledger {a.ledger}")))
        if a.json:
            Path(a.json).write_text(json.dumps({"schema": SCHEMA, "mode": "ledger", "report_only": True,
                                                "tool_sha256": _tool_sha(),
                                                "ledger": str(a.ledger), "ledger_sha256": sha256_file(a.ledger),
                                                "reading": lr}, indent=1, default=str))
        return 0

    if a.mode == "trace":
        episodes = roll_trace(a.checkpoint, a.vecnorm)
        tr = trace_reading(episodes)
        print("\n".join(describe_trace(tr)))
        if a.json:
            Path(a.json).write_text(json.dumps({"schema": SCHEMA, "mode": "trace", "report_only": True,
                                                "tool_sha256": _tool_sha(),
                                                "checkpoint": a.checkpoint, "checkpoint_sha256": sha256_file(a.checkpoint),
                                                "vecnorm": a.vecnorm, "vecnorm_sha256": sha256_file(a.vecnorm),
                                                "trace": tr}, indent=1, default=str))
        return 0

    cell = Path(a.cell)
    binding = json.loads((cell / "binding.json").read_text())
    import environments

    tree = Path(environments.__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=tree, text=True,
                                    env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()
    problems = seed_problems(binding, seed=a.seed, head=head, dirty=dirty, ledger=a.ledger, validation=a.validation)
    ckpt, vec = Path(binding["checkpoint"]), Path(binding["vecnorm"])
    for p, want in ((ckpt, binding.get("checkpoint_sha256")), (vec, binding.get("vecnorm_sha256"))):
        if not p.is_file() or sha256_file(p) != want:
            problems.append(f"{p} is missing or no longer the judged file ({want})")
    if problems:
        print("refusing (nothing written): " + "; ".join(problems), flush=True)
        return 2
    header, rows = read_ledger(a.ledger)
    lr = ledger_reading(header, rows, int(binding["steps"]))
    grid = window_grid(binding, a.validation)
    wp = window_problems(lr, int(binding["steps"]), grid)
    if wp:  # FIX3: a window that does not cover the deciding checkpoint is refused before the trace
        print("refusing (nothing written): " + "; ".join(wp), flush=True)
        return 2
    episodes = roll_trace(str(ckpt), str(vec))
    tr = trace_reading(episodes)
    reading = seed_reading(lr, tr)
    retired = []
    for path in a.retired_ledger:
        h, r = read_ledger(path)
        retired.append({"ledger": path, "reading": ledger_reading(h, r, None)})
    doc = {"schema": SCHEMA, "mode": "seed", "report_only": True, "tool_sha256": _tool_sha(), "seed": a.seed,
           "tree": str(tree),
           "head": head, "ledger": str(a.ledger), "ledger_sha256": sha256_file(a.ledger),
           "run": Path(str(binding.get("run"))).name, "validation_only": bool(binding.get("validation_only")),
           "deciding_checkpoint": {"steps": binding["steps"], "checkpoint": str(ckpt),
                                   "checkpoint_sha256": binding["checkpoint_sha256"],
                                   "vecnorm_sha256": binding["vecnorm_sha256"],
                                   "final_fallback": bool(binding.get("final_fallback"))},
           "window_check": {"grid": grid, "deciding_steps": int(binding["steps"]),
                            "first": (lr.get("window") or {}).get("first"),
                            "last": (lr.get("window") or {}).get("last"), "problems": wp},
           **reading, "ledger_reading": lr, "trace": tr, "retired_ledgers_printed_never_read": retired}
    lines = describe_ledger(lr, f"s{a.seed} ledger") + describe_trace(tr)
    for r in retired:
        lines += describe_ledger(r["reading"], f"retired ledger {r['ledger']} (printed, never read)")
    lines.append(f"s{a.seed} HABIT READING (report only; it picks nothing): {reading['reading']}"
                 + (": " + "; ".join(h["text"] for h in reading["habits"]) if reading["habits"] else "")
                 + (f"  (not read: {'; '.join(reading['not_run'])})" if reading["not_run"] else ""))
    print("\n".join(lines))
    Path(a.json).write_text(json.dumps(doc, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
