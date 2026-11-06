"""Score walking pilot 4's judged panels against its fixed criteria (criteria_pilot4 draft v3, "Seed outcome").

Scratch, never merged: pilot 3's pilot3_score.py (sha256 286ce04a...) adapted for walking pilot 4 as v3's TOOL PLAN
says. Every label, guard and threshold is pilot 3's (pilot 1's): label_episode, seed_outcome, pilot_outcome,
reviewed_outcome and pilot_outcome_reviewed are byte-identical (test_pilot4_score.py compares their source with the
pinned file). Changed: CHECKPOINTS (500,000, 1,000,000, 1,500,000, 2,000,000), PRIMARY 2,000,000, the replay RUN_ID
pilot4_walk_s<seed>, the renderer pilot4_video.py, the judge's name in one problem text. Removed: the Q4 habit branch
(q4_reading) and pilot 3's existence reading (WALK EXISTS); --habit is printed report only. Added, in new functions:
--straight (pilot4_straight.py's seed-mode JSON, one record per seed and checkpoint), --straight-review (the
straightness look's JSON), the four seed outcomes, the pilot outcome over them (PENDING_REVIEW), its final reading,
the report-only count of seeds passing (a), (b) and the look (Q23), the existence reading (STRAIGHT WALK EXISTS), (c)'s
NOT READ printed with its counts, "STRAIGHTNESS LOOK REQUIRED", and, report only, the speed and gait medians beside
pilot 3's 8M medians (constants with their source).

Plain Python, no repository import. Reads what pilot4_judge.py and pilot4_straight.py wrote:
    OUT/s<seed>/<steps>/probe.json and binding.json      (one 40-episode panel per seed and checkpoint)
    STRAIGHT/s<seed>_<steps>.json                        (pilot4_straight.py seed --cell OUT/s<seed>/<steps>)
and prints, per seed and checkpoint, the label counts and pilot 3's seed outcome (the gait reading), the STRAIGHT
parts at the deciding checkpoint, the seed outcome, the episodes the video review must watch, the seeds the
straightness look must see, then the pilot outcome, its final reading, the gait reading, the report-only count and the
existence reading, and the report-only lines (habit, trajectory STRAIGHT parts, speed and gait against pilot 3's 8M).

    python3 pilot4_score.py OUT --seeds 42,52,60 [--json OUT/score.json] [--primary 2000000] [--validation]
                                [--video-review REVIEW.json] [--straight S.json ...] [--straight-review LOOK.json]
                                [--closed-seeds 52,60] [--habit HABIT.json ...]

--seeds names the seeds the pilot launched (required): a launched seed with nothing judged counts as missing, and a
directory for a seed not named is reported and ignored.

Episode labels, seed outcome by the rule, video review, one run per seed, collapse stop, final-model fallback and the
cells never scored: pilot 3's, unchanged (its docstring, kept here in short):
  FALL   the episode ended before 1,000 steps
  HOP    at least 0.5 touchdowns per second (all feet) and a simultaneous-touchdown fraction >= 0.5, at any speed
  STILL  window displacement speed below 0.10 m/s
  WALK   alternation index >= 0.6, lead swaps per stride >= 1.0, flight fraction <= 0.2, jointlag_frac_hip_pitch in
         [0.35, 0.65], floor_stride_frequency_hz <= 5.0, floor_duty_L and floor_duty_R <= 0.80
  OTHER  anything else
  Pilot 3's seed outcome at a checkpoint (40 episodes): WALKS if WALK >= 30 and FALL <= 4; else HOPS if HOP >= 20; else
  FALLS if FALL >= 5; else STILL if STILL >= 20; else MIXED. Read at the primary checkpoint (2,000,000); a
  collapse-stopped seed at its newest intact periodic pair; a seed decided at its final pair (pilot4_judge.py
  --final-fallback) says so. Video review (--video-review {"42": "agree" | "disagree", ...}): REQUIRED for every seed
  WALKS by the rule; "disagree" makes it MIXED ("WALKS by the rule, not on video"). A cell is never scored when its
  binding.json is missing, validation-only (unless --validation) or off-schedule, or its panel is not 40 episodes of
  panel 3042 with a 100-step window; a seed whose cells come from more than one run is not scored; OUT/off_schedule/
  and OUT/retired/ are listed, never scored.

STRAIGHT (v3 "STRAIGHT"; computed by pilot4_straight.py, read here): (a) end heading, (b) weave, (c) recovery, each
PASS / FAIL / NOT READ, and STRAIGHT = (a) and (b) and (c). A record is read for a seed only when it is the seed-mode
record of that very cell: schema walking-pilot4-straight/v1, mode seed, not partial, not a validation record (unless
--validation), its seed and steps, binding_sha256 and probe_json_sha256 equal to the sha256 of the cell's binding.json
and probe.json, its checkpoint and sidecar sha256s, head and final_fallback equal to the binding's, every panel
episode re-rolled equal to probe.json (panel.all_rolled_match), and verdicts that are PASS / FAIL / NOT READ with
STRAIGHT PASS exactly when all three are; and (fix stage) written by the pinned pilot4_straight.py (tool_sha256 equal
to STRAIGHT_TOOL_SHA256) with v3's bars and windows (constants equal to STRAIGHT_CONSTANTS), each part's verdict
following from the record's own values under those bars (recomputed_verdicts). Otherwise its problems are printed and
the seed has no STRAIGHT reading. (Fix stage) a cell whose probe.json records an env-kwargs drift (the run's recorded
env kwargs against the judging tree's) is not scored (score_cell).
Records at the trajectory checkpoints (500,000-1,500,000) are printed, report only.

Seed outcome (v3, first that applies, at the deciding checkpoint):
  NOT WALKS (<label>)    pilot 3's seed outcome is not WALKS (HOPS, FALLS, STILL or MIXED), or it is WALKS and the
                         video review disagrees (MIXED: "WALKS by the rule, not on video")
  CURVED WALKS           WALKS and video-agreed, and (a) or (b) is not PASS; or both PASS and the look reads
                         "not straight" ("straight by the rule, not on video")
  STRAIGHT NO RECOVERY   WALKS and video-agreed, (a) and (b) PASS, the look "straight", and (c) FAIL or NOT READ
                         (printed which, NOT READ with its counts)
  STRAIGHT WALKS         WALKS and video-agreed, (a), (b) and (c) PASS, and the look "straight"
  pending                a WALKS seed whose video review, straightness record or (when (a) and (b) pass) look is
                         still missing and could change its outcome; its possible outcomes are listed.
  missing                no scored gait outcome (pilot 3's rule); CLOSED by --closed-seeds when nothing can score it.
Straightness look (--straight-review {"42": "straight" | "not straight", ...} or {"42": {"verdict": ..., "note":
...}}): required for every seed that is WALKS, video-agreed and passes (a) and (b), whatever (c) reads; "not straight"
makes it CURVED WALKS; a look for any other seed is recorded and changes nothing.
Pilot outcome: pilot 3's pilot_outcome over the seed outcomes (NOT WALKS shares across its sub-labels): the outcome at
least 2 launched seeds share; MIXED when every launched seed has an outcome and no two share one; INCOMPLETE while a
missing seed could still decide it; PENDING_REVIEW while a pending seed could change it (every combination of the
pending seeds' possible outcomes is tried: when all give one outcome it stands). The final reading (--closed-seeds):
when the pilot outcome is INCOMPLETE and every missing seed is CLOSED, the scored seeds alone, by the same rule.
Printed beside it, never instead of it: the gait reading (pilot 3's pilot outcome in this env, pilot_outcome_reviewed,
with pilot 3's final reading); the count of seeds WALKS, video-agreed, (a) and (b) PASS and the look "straight"
(report only; Q23); the existence reading: STRAIGHT WALK EXISTS when a seed is STRAIGHT WALKS, whatever the pilot
outcome; else PENDING while a pending seed could still become one; else NONE SEEN; final only when no seed is pending
and every missing seed is CLOSED.
Report only (they change no outcome): --habit (pilot4_habit.py's JSON: HABIT / NO HABIT / NOT RUN per seed, read at
the deciding checkpoint; it picks nothing); the STRAIGHT parts at the trajectory checkpoints; the speed and gait
medians at the deciding checkpoint beside pilot 3's 8M medians (P3_8M, measured from pilot 3's judged 8M cells).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

HORIZON = 1000
STILL_SPEED = 0.10
WALK_MIN_ALTERNATION = 0.6
WALK_MIN_LEAD_SWAPS = 1.0
WALK_MAX_FLIGHT = 0.2
WALK_HIP_LAG_KEY = "jointlag_frac_hip_pitch"
WALK_HIP_LAG_BAND = (0.35, 0.65)
WALK_STRIDE_KEY = "floor_stride_frequency_hz"
WALK_MAX_STRIDE_HZ = 5.0
WALK_DUTY_KEYS = ("floor_duty_L", "floor_duty_R")
WALK_MAX_DUTY = 0.80
VIDEO_DUTY = 0.8
SWING_KEYS = ("floor_swing_mean_s_L", "floor_swing_mean_s_R")
SHORT_SWING_S = 0.03  # report only: every false WALK found had a mean floor swing of 0.020-0.021 s (two control steps)
HOP_MIN_SIMULTANEOUS = 0.5
HOP_MIN_TOUCHDOWNS_PER_S = 0.5
EPISODES = 40
PANEL_SEED_BASE = 3042
SEED_WALKS_MIN = 30
SEED_WALKS_MAX_FALLS = 4
SEED_HOPS_MIN = 20
SEED_FALLS_MIN = 5
SEED_STILL_MIN = 20
CHECKPOINTS = (500_000, 1_000_000, 1_500_000, 2_000_000)
PRIMARY = 2_000_000
LAUNCHED = 3  # the seeds the pilot launches (42, 52, 60): the k of 3 the readings print
OUTCOMES = ("WALKS", "HOPS", "FALLS", "STILL", "MIXED")
GUARDS = ("hip_lag", "stride", "duty")
NOTEBOOK_REPLAY = ("MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/<RUN_ID>/03_locomotion/replays/"
                   "tyrannosaurus_rex_anatomical_ppo_stage2_final.mp4")


def _num(row: dict, key: str) -> float:
    v = row.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v


def walk_contact_tests(row: dict) -> bool:
    """The section 5.5 contact bar alone (alternation, lead swaps, flight)."""
    return (_num(row, "floor_alternation_index") >= WALK_MIN_ALTERNATION
            and _num(row, "floor_lead_swaps_per_stride") >= WALK_MIN_LEAD_SWAPS
            and _num(row, "floor_flight_fraction") <= WALK_MAX_FLIGHT)


def hip_lag_in_band(row: dict) -> bool:
    lag = _num(row, WALK_HIP_LAG_KEY)
    return WALK_HIP_LAG_BAND[0] <= lag <= WALK_HIP_LAG_BAND[1]


def stride_frequency_ok(row: dict) -> bool:
    """The stride cap: floor_stride_frequency_hz <= 5.0 (undefined fails)."""
    return _num(row, WALK_STRIDE_KEY) <= WALK_MAX_STRIDE_HZ


def foot_duty_max(row: dict) -> float:
    """The higher of the two feet's floor duty; NaN when either is undefined."""
    duties = [_num(row, k) for k in WALK_DUTY_KEYS]
    return max(duties) if all(math.isfinite(d) for d in duties) else math.nan


def duty_ok(row: dict) -> bool:
    """The duty guard: each foot's floor duty <= 0.80 (undefined fails)."""
    return foot_duty_max(row) <= WALK_MAX_DUTY


def guards_failed(row: dict) -> list:
    """Which of the three WALK guards beyond the contact bar the row fails."""
    return [name for name, ok in (("hip_lag", hip_lag_in_band(row)), ("stride", stride_frequency_ok(row)),
                                  ("duty", duty_ok(row))) if not ok]


def label_episode(row: dict) -> str:
    if _num(row, "length") < HORIZON:
        return "FALL"
    if (_num(row, "floor_touchdowns_per_s_all_feet") >= HOP_MIN_TOUCHDOWNS_PER_S
            and _num(row, "floor_simultaneous_fraction") >= HOP_MIN_SIMULTANEOUS):
        return "HOP"
    if not (_num(row, "speed_displacement_mps") >= STILL_SPEED):
        return "STILL"
    if walk_contact_tests(row) and not guards_failed(row):
        return "WALK"
    return "OTHER"


def foot_swing_min(row: dict) -> float:
    """The shorter of the two feet's mean floor swing (s); NaN when either is undefined. Report only."""
    swings = [_num(row, k) for k in SWING_KEYS]
    return min(swings) if all(math.isfinite(x) for x in swings) else math.nan


def video_watch(rows: list, labels: list) -> tuple:
    """(episodes to watch, number of STILL episodes over the duty line left out): every WALK, and every episode that
    is not STILL with a foot's floor duty above 0.8. Each entry flags short_swing (report only): a mean floor swing
    of at most 0.03 s on either foot, the signature of every false WALK found (0.020-0.021 s)."""
    watch, still_over = [], 0
    for i, (r, lab) in enumerate(zip(rows, labels)):
        d = foot_duty_max(r)
        over = math.isfinite(d) and d > VIDEO_DUTY
        if lab == "STILL":
            still_over += int(over)
            continue
        if lab == "WALK" or over:
            sw = foot_swing_min(r)
            watch.append({"episode": i, "reset_seed": PANEL_SEED_BASE + i, "label": lab,
                          "foot_duty_max": None if not math.isfinite(d) else round(d, 4),
                          "foot_swing_min_s": None if not math.isfinite(sw) else round(sw, 4),
                          "short_swing": bool(math.isfinite(sw) and sw <= SHORT_SWING_S),
                          "speed_forward_vel_mps": _rounded(_num(r, "speed_forward_vel_mps")),
                          "why": "WALK" if lab == "WALK" else "foot duty > 0.8"})
    return watch, still_over


def seed_outcome(counts: Counter) -> str:
    if counts["WALK"] >= SEED_WALKS_MIN and counts["FALL"] <= SEED_WALKS_MAX_FALLS:
        return "WALKS"
    if counts["HOP"] >= SEED_HOPS_MIN:
        return "HOPS"
    if counts["FALL"] >= SEED_FALLS_MIN:
        return "FALLS"
    if counts["STILL"] >= SEED_STILL_MIN:
        return "STILL"
    return "MIXED"


def pilot_outcome(per_seed: dict) -> dict:
    """per_seed: every LAUNCHED seed -> its outcome (None when missing or invalid)."""
    present = {s: o for s, o in per_seed.items() if o is not None}
    missing = sorted(s for s, o in per_seed.items() if o is None)
    n = len(per_seed)
    if not present:
        return {"outcome": None, "basis": f"no launched seed of {n} has a scored outcome", "missing": missing}
    if n == 1:
        (seed, o), = present.items()
        return {"outcome": o, "basis": f"single seed (s{seed})", "missing": missing}
    tally = Counter(present.values())
    top, k = tally.most_common(1)[0]
    if k >= 2:
        return {"outcome": top, "basis": f"{k} of {n} launched seeds", "missing": missing}
    if missing:
        return {"outcome": "INCOMPLETE", "basis": f"no outcome shared by 2 of {n} launched seeds yet "
                                                  f"({dict(tally)}); the missing seeds could decide it",
                "missing": missing}
    return {"outcome": "MIXED", "basis": f"no outcome shared by 2 of {n} seeds: {dict(tally)}", "missing": missing}


def read_video_review(path: "str | Path | None") -> dict:
    """{seed (int): {"verdict": "agree" | "disagree", "note": str}} from the maintainer's review file."""
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    out = {}
    for k, v in raw.items():
        verdict, note = (v.get("verdict"), v.get("note", "")) if isinstance(v, dict) else (v, "")
        verdict = str(verdict).strip().lower()
        if verdict not in ("agree", "disagree"):
            raise SystemExit(f"{path}: seed {k}: verdict must be 'agree' or 'disagree', not {verdict!r}")
        out[int(str(k).lstrip("s"))] = {"verdict": verdict, "note": note}
    return out


def reviewed_outcome(rule: "str | None", verdict: "str | None") -> tuple:
    """(outcome after the video review, review state) for one seed's rule outcome."""
    if rule != "WALKS":
        return rule, ("recorded, changes nothing" if verdict else "not needed")
    if verdict == "agree":
        return "WALKS", "agree"
    if verdict == "disagree":
        return "MIXED", "disagree"
    return "WALKS", "pending"


def pilot_outcome_reviewed(rule_per_seed: dict, review: dict) -> dict:
    """The pilot outcome after the video review; PENDING_VIDEO_REVIEW while a pending WALKS seed could change it."""
    reviewed, pending = {}, []
    for s, o in rule_per_seed.items():
        out, state = reviewed_outcome(o, (review.get(s) or {}).get("verdict"))
        reviewed[s] = out
        if state == "pending":
            pending.append(s)
    if not pending:
        res = pilot_outcome(reviewed)
        res.update(final=True, pending_video_review=[])
        return res
    possible = {}
    for combo in itertools.product(("WALKS", "MIXED"), repeat=len(pending)):
        trial = {**reviewed, **dict(zip(pending, combo))}
        res = pilot_outcome(trial)
        possible.setdefault(res["outcome"], res)
    if len(possible) == 1:
        (res,) = possible.values()
        res = dict(res, final=True, pending_video_review=sorted(pending))
        res["basis"] += f"; the same whatever the pending video review of {['s%d' % s for s in sorted(pending)]} finds"
        return res
    return {"outcome": "PENDING_VIDEO_REVIEW", "final": False, "pending_video_review": sorted(pending),
            "possible_outcomes": sorted(str(o) for o in possible),
            "basis": f"WALKS by the rule for {['s%d' % s for s in sorted(pending)]}, not yet confirmed on video; "
                     f"the outcome is one of {sorted(str(o) for o in possible)}",
            "missing": pilot_outcome(reviewed)["missing"]}


def _median(xs: list) -> float:
    xs = sorted(x for x in xs if isinstance(x, (int, float)) and math.isfinite(x))
    if not xs:
        return math.nan
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


REPORT_KEYS = ("speed_displacement_mps", "floor_touchdowns_per_s_all_feet", "floor_alternation_index",
               "floor_simultaneous_fraction", "floor_lead_swaps_per_stride", "floor_flight_fraction",
               "floor_single_support_fraction", "floor_stride_frequency_hz", "floor_stride_length_over_L",
               "left_leads_fraction", "froude", WALK_HIP_LAG_KEY, *WALK_DUTY_KEYS, *SWING_KEYS,
               "speed_forward_vel_mps", "yaw_change_deg")


def score_panel(doc: dict, binding: dict | None = None, validation: bool = False) -> dict:
    rows = doc.get("episodes", [])
    meta = doc.get("meta", {})
    problems = []
    if binding is None:
        problems.append("no binding.json (not written by pilot4_judge.py)")
    else:
        if binding.get("validation_only") and not validation:
            problems.append(f"validation-only judge (run budget {binding.get('expected_budget')})")
        if binding.get("off_schedule"):
            problems.append(f"off-schedule run: {binding.get('off_schedule_reason')}")
    if len(rows) != EPISODES:
        problems.append(f"{len(rows)} episodes, not {EPISODES}")
    if meta.get("seed_scheme") != "panel" or meta.get("seed_base") != PANEL_SEED_BASE:
        problems.append(f"seed scheme {meta.get('seed_scheme')} base {meta.get('seed_base')}, not panel 3042")
    if meta.get("settle_steps_gait_window") != 100:
        problems.append(f"gait window starts at {meta.get('settle_steps_gait_window')}, not 100")
    labels = [label_episode(r) for r in rows]
    counts = Counter(labels)
    # report only: OTHER episodes that pass the contact bar, by the guards that reject them (a staggered or scissor
    # hop fails the hip-lag band or the stride cap; a short-lift shuffle-hop the duty guard)
    rejected = Counter()
    for r, lab in zip(rows, labels):
        if lab == "OTHER" and walk_contact_tests(r):
            rejected["+".join(guards_failed(r))] += 1
    combos = ["+".join(c) for n in (1, 2, 3) for c in itertools.combinations(GUARDS, n)]
    watch, still_over = video_watch(rows, labels)
    out = {"counts": {k: counts.get(k, 0) for k in ("WALK", "HOP", "STILL", "FALL", "OTHER")},
           "outcome": seed_outcome(counts) if not problems else None, "problems": problems,
           "labels": labels,
           "other_passing_contact_bar": {k: rejected.get(k, 0) for k in combos},
           "video_watch": watch, "still_with_foot_duty_over_0_8": still_over,
           "walk_with_short_swing": sum(1 for e in watch if e["label"] == "WALK" and e["short_swing"]),
           "probe_classes": dict(Counter(str(r.get("floor_gait_class")) for r in rows)),
           "medians": {k: _median([_num(r, k) for r in rows]) for k in REPORT_KEYS},
           "mean_reward": _median([_num(r, "reward") for r in rows]),
           "run_end": (binding or {}).get("run_end"),
           "run": Path(str(binding["run"])).name if (binding or {}).get("run") else None,
           "final_fallback": bool((binding or {}).get("final_fallback")),
           "still_report": still_report(rows, labels),
           "checkpoint_sha256": meta.get("checkpoint_sha256"), "vecnorm_sha256": meta.get("vecnorm_sha256")}
    return out


def pilot4_cell_problems(doc: dict) -> list:
    """Fix stage: pilot 4's checks on a judged cell beyond pilot 3's (score_panel is kept as pilot 3's): the probe's
    own env-kwargs drift record (the run's recorded env kwargs against the tree's) must be empty."""
    drift = (doc.get("meta") or {}).get("env_kwargs_drift_vs_recorded")
    if drift:
        return [f"env kwargs drift: the run's recorded env kwargs differ from the env it was judged in "
                f"({json.dumps(drift, sort_keys=True, default=str)})"]
    return []


def score_cell(doc: dict, binding: dict | None = None, validation: bool = False) -> dict:
    """score_panel (pilot 3's) plus pilot4_cell_problems: a problem leaves the cell unscored (outcome None)."""
    out = score_panel(doc, binding, validation)
    extra = pilot4_cell_problems(doc)
    if extra:
        out["problems"] += extra
        out["outcome"] = None
    return out


def seed_result(cells: dict, primary: int) -> dict:
    """cells: steps (int) -> scored cell or None. The seed's outcome, how it was reached, and the checkpoint read."""
    prim = cells.get(primary)
    if prim is not None and prim["outcome"] is not None:
        return {"outcome": prim["outcome"], "basis": f"checkpoint {primary:,}", "steps": primary}
    ends = [c["run_end"] for c in cells.values() if c is not None and c.get("run_end")]
    stopped = [e for e in ends if e.get("stopped_early")]
    if prim is None and stopped:
        e = max(stopped, key=lambda e: e.get("final_num_timesteps") or 0)
        if e.get("intact_periodic_steps") is not None:  # FIX2: the criteria read the newest INTACT pair
            newest, which = e.get("newest_intact_periodic_steps"), "newest intact periodic pair"
        else:
            newest, which = e.get("newest_periodic_steps"), ("newest periodic checkpoint by file name (this binding "
                                                             "lists no intact pairs)")
        if newest is None:
            return {"outcome": None, "collapse_stopped": True, "steps": None,
                    "basis": f"collapse-stopped at {e.get('final_num_timesteps')} steps, with no {which}"}
        cell = cells.get(newest)
        if cell is not None and cell["outcome"] is not None:
            return {"outcome": cell["outcome"], "collapse_stopped": True, "steps": newest,
                    "basis": f"collapse-stopped at {e.get('final_num_timesteps'):,} steps ({e.get('final_source')}); "
                             f"outcome read from its {which} {newest:,}"}
        return {"outcome": None, "collapse_stopped": True, "steps": None,
                "basis": f"collapse-stopped at {e.get('final_num_timesteps')} steps, but its {which} "
                         f"{newest} has no valid judgement: judge it"}
    if prim is not None:
        return {"outcome": None, "steps": None,
                "basis": f"checkpoint {primary:,} judged but not scored: {prim['problems']}"}
    judged = sorted(s for s, c in cells.items() if c is not None)
    return {"outcome": None, "steps": None,
            "basis": "missing" + (f" (judged so far: {', '.join(f'{s:,}' for s in judged)})" if judged
                                  else " (nothing judged)")}


def one_run_problem(cells: dict) -> "str | None":
    """FIX2: None when every judged cell of a seed comes from one run copy (binding "run", by directory name); else
    the problem, which unscores every cell of the seed (a restart's retired cells belong under OUT/retired/)."""
    runs = sorted({c["run"] for c in cells.values() if c is not None and c.get("run")})
    if len(runs) <= 1:
        return None
    return (f"cells from more than one run {runs}: move the retired run's cells to OUT/retired/<retired RUN_ID>/"
            "s<seed>/<steps>/ (criteria, Run handling)")


# ---------------------------------------------------------------- walking pilot 3's added readings (kept; Q4 removed)

STILL_STEPPING_TD = 0.5  # report only: STILL episodes at 0.5 touchdowns per second (all feet) or more
HABIT_READINGS = ("HABIT", "NO HABIT", "NOT RUN")


def _rounded(x: float, nd: int = 4) -> "float | None":
    return round(x, nd) if isinstance(x, (int, float)) and math.isfinite(x) else None


def still_report(rows: list, labels: list) -> dict:
    """Report only (criteria "Episode labels", new): for each STILL episode the touchdowns per second (all feet), the
    alternation index, the yaw change and speed_forward_vel_mps; their medians; and how many STILL episodes step at
    0.5 touchdowns per second or more (stepping in place, creeping or circling all read STILL)."""
    eps = []
    for i, (r, lab) in enumerate(zip(rows, labels, strict=True)):
        if lab == "STILL":
            eps.append({"episode": i, "touchdowns_per_s": _rounded(_num(r, "floor_touchdowns_per_s_all_feet")),
                        "alternation": _rounded(_num(r, "floor_alternation_index")),
                        "yaw_change_deg": _rounded(_num(r, "yaw_change_deg"), 2),
                        "speed_forward_vel_mps": _rounded(_num(r, "speed_forward_vel_mps"))})
    still = [r for r, lab in zip(rows, labels, strict=True) if lab == "STILL"]
    tds = [_num(r, "floor_touchdowns_per_s_all_feet") for r in still]
    return {"n": len(still), "episodes": eps,
            "stepping_at_0_5_touchdowns_per_s": sum(1 for td in tds if td >= STILL_STEPPING_TD),
            "median_touchdowns_per_s": _median(tds),
            "median_alternation": _median([_num(r, "floor_alternation_index") for r in still]),
            "median_abs_yaw_change_deg": _median([abs(_num(r, "yaw_change_deg")) for r in still])}


def final_fallback_basis(sr: dict, cells: dict) -> dict:
    """Mark a seed decided at its final pair (pilot3_judge.py --final-fallback): "decided at the final model
    (fallback)". The outcome is unchanged; only the basis says where it was read."""
    cell = cells.get(sr.get("steps")) if sr.get("steps") is not None else None
    if cell is not None and cell.get("final_fallback") and sr.get("outcome") is not None:
        sr = dict(sr, final_fallback=True,
                  basis=sr["basis"] + ": decided at the final model (fallback; stage2_final, the deciding checkpoint's "
                                      "policy plus the last update)")
    return sr


def read_closed_seeds(text: "str | None", seeds: list) -> list:
    """--closed-seeds: the launched seeds the run log closes (restart spent, run lost or refused), comma separated."""
    if not text:
        return []
    closed = sorted({int(x.strip().lstrip("s")) for x in text.split(",") if x.strip()})
    unknown = [s for s in closed if s not in seeds]
    if unknown:
        raise SystemExit(f"--closed-seeds {unknown} are not launched seeds {seeds}")
    return closed


def read_habit(paths: "list | None") -> dict:
    """{seed (int): record} from pilot3_habit.py's JSON files (each one seed's record, or {"seeds": {seed: record}})."""
    out: dict = {}
    for path in paths or []:
        doc = json.loads(Path(path).read_text())
        recs = doc["seeds"].items() if isinstance(doc.get("seeds"), dict) else [(doc.get("seed"), doc)]
        for k, rec in recs:
            seed = int(str(k).lstrip("s"))
            if seed in out:
                raise SystemExit(f"{path}: a second habit record for s{seed}")
            out[seed] = dict(rec, file=str(path))
    return out


def habit_text(st: dict) -> str:
    if st["reading"] != "HABIT":
        return st["reading"] + (f" ({st['why']})" if st.get("why") else "")
    return "HABIT: " + "; ".join(h.get("text") or h.get("kind", "?") for h in st["habits"])


def final_reading(rule_per_seed: dict, review: dict, pilot: dict, closed: list) -> dict:
    """The final reading (criteria "Pilot outcome"): when the pilot outcome is INCOMPLETE (or none) and every missing
    seed is CLOSED, the pilot is read from its scored seeds alone by pilot 1's rule (pilot_outcome_reviewed) applied as
    if only they had been launched: two that differ read MIXED; one reads its own outcome, "one seed only"; none reads
    NO OUTCOME. Never the pilot outcome itself."""
    missing = sorted(s for s, o in rule_per_seed.items() if o is None)
    if pilot.get("outcome") not in ("INCOMPLETE", None) or not missing:
        return {"applies": False, "why": "the pilot outcome is not INCOMPLETE" if pilot.get("outcome") is not None
                else "no seed is missing"}
    still_open = [s for s in missing if s not in closed]
    if still_open:
        return {"applies": False, "why": f"missing seeds not CLOSED: {['s%d' % s for s in still_open]}"}
    scored = {s: o for s, o in rule_per_seed.items() if o is not None}
    if scored:
        res = pilot_outcome_reviewed(scored, review)
        reading, final, basis = res["outcome"], bool(res.get("final", True)), res["basis"]
    else:
        reading, final, basis = "NO OUTCOME", True, "no scored seed"
    one = len(scored) == 1
    text = (f"FINAL READING: {reading}{' (one seed only)' if one else ''} from {len(scored)} scored seeds of "
            f"{len(rule_per_seed)} (closed: {', '.join('s%d' % s for s in missing)})")
    return {"applies": True, "reading": reading, "final": final and reading != "PENDING_VIDEO_REVIEW",
            "k_scored": len(scored), "one_seed_only": one, "closed": missing, "basis": basis, "text": text}


def habit_report(seed: int, rec: "dict | None", deciding_sha: "str | None") -> dict:
    """Report only (v3 D10: the habit readout picks nothing): one seed's habit reading, HABIT, NO HABIT or NOT RUN, its
    groups, and why. NOT RUN when the record is missing or malformed, when it is validation-only (a check of the tool),
    or when its trace read another checkpoint than the one the seed's outcome is read at (pilot 3's habit_status
    checks, without its Q4 wording)."""
    def not_run(why: str) -> dict:
        return {"reading": "NOT RUN", "why": why, "habits": []}

    if rec is None:
        return not_run(f"no habit record for s{seed}")
    if rec.get("validation_only"):
        return not_run(f"s{seed}'s habit record is validation-only (a check of the tool)")
    reading = rec.get("reading")
    if reading not in HABIT_READINGS:
        return not_run(f"habit record for s{seed} reads {reading!r}")
    traced = (rec.get("deciding_checkpoint") or {}).get("checkpoint_sha256")
    if deciding_sha and traced and traced != deciding_sha:
        return not_run(f"s{seed}'s habit record read checkpoint {traced}, not the deciding checkpoint {deciding_sha}")
    if deciding_sha and not traced and reading != "NOT RUN":
        return not_run(f"s{seed}'s habit record names no deciding checkpoint")
    habits = list(rec.get("habits") or [])
    if reading == "HABIT" and not habits:
        return not_run(f"s{seed}'s habit record reads HABIT but lists none")
    return {"reading": reading, "why": "; ".join(rec.get("not_run") or []) or None, "habits": habits}


# ---------------------------------------------------------------- walking pilot 4: STRAIGHT, seed and pilot outcomes

STRAIGHT_SCHEMA = "walking-pilot4-straight/v1"  # pilot4_straight.py's record
#: Fix stage: a record is read only when the pinned pilot4_straight.py wrote it (its sha256 in SHA256SUMS.txt) with
#: v3's bars and windows (its constants(), written into every record; tuples as JSON lists), and only when each part's
#: verdict follows from the record's own values under those bars.
STRAIGHT_TOOL_SHA256 = "sha256:93d5c558ebffb80036f0f054e918629c17af07d0a18e03ac844532ec2d487848"
STRAIGHT_CONSTANTS = {
    "HORIZON": 1000, "PANEL_EPISODES": 40, "PANEL_SEED_BASE": 3042, "SETTLE": 100, "WINDOW_FIRST_STATE": 101,
    "MEAN_HALF": 50, "K_FIRST": 151, "A_MEDIAN_ABS_MAX_DEG": 5.0, "A_P95_ABS_MAX_DEG": 15.0, "B_D_P95_MAX_DEG": 11.5,
    "B_W0_P95_MAX_DEG": 18.5, "C_MEDIAN_MAX_DEG": 5.0, "C_P95_MAX_DEG": 11.5, "C_MIN_COUNTING": 30,
    "C_MIN_NET_MOVE_M": 0.35, "PCT": 95, "RECOVERY_SEEDS": list(range(3042, 3062)), "NUDGE_ANGLES_DEG": [15.0, -15.0],
    "NUDGE_STATE": 650, "READ_K": 950, "RESET_TURN_STATE": 0, "RESET_READ_K": 500, "OBS651_TOL": 0.001,
    "TWIN_HEADING_TOL_DEG": 1e-05, "YAW_REPRO_TOL_DEG": 1e-12, "INFO_HEADING_TOL_RAD": 1e-09}
PART_VERDICTS = ("PASS", "FAIL", "NOT READ")
LOOK_VERDICTS = ("straight", "not straight")
SEED_OUTCOMES = ("STRAIGHT WALKS", "STRAIGHT NO RECOVERY", "CURVED WALKS", "NOT WALKS")
PENDING = "pending"
PILOT_PENDING = "PENDING_REVIEW"
VIDEO_DISAGREE_LABEL = "MIXED: WALKS by the rule, not on video"
LOOK_DISAGREE_LABEL = "straight by the rule, not on video"
# one parts reading per STRAIGHT outcome, tried for a WALKS seed whose straightness record is not read yet
_ANY_PARTS = ({"a": "PASS", "b": "PASS", "c": "PASS"}, {"a": "PASS", "b": "PASS", "c": "FAIL"},
              {"a": "FAIL", "b": "FAIL", "c": "FAIL"})


def file_sha256(path: "str | Path") -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_straight_review(path: "str | Path | None") -> dict:
    """{seed (int): {"verdict": "straight" | "not straight", "note": str}} from the maintainer's straightness look
    ({"42": "straight", ...} or {"42": {"verdict": ..., "note": ...}}; case and repeated spaces ignored)."""
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: the look must be a JSON object of per-seed verdicts")
    out = {}
    for k, v in raw.items():
        verdict, note = (v.get("verdict"), v.get("note", "")) if isinstance(v, dict) else (v, "")
        verdict = " ".join(str(verdict).strip().lower().split())
        if verdict not in LOOK_VERDICTS:
            raise SystemExit(f"{path}: seed {k}: the look must be 'straight' or 'not straight', not {verdict!r}")
        out[int(str(k).lstrip("s"))] = {"verdict": verdict, "note": note}
    return out


def read_straight(paths: "list | None") -> dict:
    """{(seed, steps): record} from pilot4_straight.py's JSON files (one record each, its path added as "file"); a file
    without an integer seed and steps, and a second record for one seed and checkpoint, are refused."""
    out: dict = {}
    for path in paths or []:
        doc = json.loads(Path(path).read_text())
        ok = isinstance(doc, dict) and all(isinstance(doc.get(k), int) and not isinstance(doc.get(k), bool)
                                           for k in ("seed", "steps"))
        if not ok:
            raise SystemExit(f"{path}: not a straightness record (no integer seed and steps)")
        key = (doc["seed"], doc["steps"])
        if key in out:
            raise SystemExit(f"{path}: a second straightness record for s{key[0]} at {key[1]:,} ({out[key]['file']})")
        out[key] = dict(doc, file=str(path))
    return out


def straight_record_problems(rec: dict, seed: int, steps: int, cell_dir: "str | Path",
                             validation: bool = False) -> list:
    """Why a straightness record cannot be read for the cell OUT/s<seed>/<steps> (empty when it can): it must be
    pilot4_straight.py's seed-mode record of that very cell (module docstring, "STRAIGHT")."""
    cell_dir = Path(cell_dir)
    p = []
    if rec.get("schema") != STRAIGHT_SCHEMA:
        p.append(f"schema {rec.get('schema')!r}, not {STRAIGHT_SCHEMA}")
    if rec.get("mode") != "seed":
        p.append(f"mode {rec.get('mode')!r}, not seed (a baseline record is never scored)")
    if rec.get("partial") is not False:
        p.append(f"partial {rec.get('partial')!r} (a partial record has no verdict)")
    if rec.get("validation") is not False and not validation:
        p.append(f"validation {rec.get('validation')!r}: a validation record (pilot4_straight.py seed --validation) is "
                 "never scored")
    if rec.get("seed") != seed or rec.get("steps") != steps:
        p.append(f"a record of s{rec.get('seed')} at {rec.get('steps')}, not of s{seed} at {steps}")
    bpath, ppath = cell_dir / "binding.json", cell_dir / "probe.json"
    if not (bpath.is_file() and ppath.is_file()):
        p.append(f"no judged cell at {cell_dir} (binding.json and probe.json)")
        return p
    binding = json.loads(bpath.read_text())
    for key, want in (("binding_sha256", file_sha256(bpath)), ("probe_json_sha256", file_sha256(ppath)),
                      ("checkpoint_sha256", binding.get("checkpoint_sha256")),
                      ("vecnorm_sha256", binding.get("vecnorm_sha256")), ("head", binding.get("head"))):
        if rec.get(key) is None or rec.get(key) != want:
            p.append(f"{key} {rec.get(key)} is not this cell's {want}")
    ff = rec.get("final_fallback")
    if not isinstance(ff, bool) or ff != bool(binding.get("final_fallback")):
        p.append(f"final_fallback {ff!r} is not this cell's {bool(binding.get('final_fallback'))}")
    panel = rec.get("panel") if isinstance(rec.get("panel"), dict) else {}
    if panel.get("all_rolled_match") is not True:
        p.append(f"the re-rolled panel does not equal probe.json (all_rolled_match {panel.get('all_rolled_match')!r}, "
                 f"episodes {panel.get('mismatched')})")
    if rec.get("tool_sha256") != STRAIGHT_TOOL_SHA256:
        p.append(f"tool_sha256 {rec.get('tool_sha256')} is not the pinned pilot4_straight.py {STRAIGHT_TOOL_SHA256}")
    const = rec.get("constants")
    if not isinstance(const, dict) or const != STRAIGHT_CONSTANTS:
        off = ({k: const.get(k, "<absent>") for k in sorted(set(STRAIGHT_CONSTANTS) | set(const))
                if const.get(k, "<absent>") != STRAIGHT_CONSTANTS.get(k, "<absent>")}
               if isinstance(const, dict) else const)
        p.append(f"constants {off} are not v3's bars and windows (the pinned tool's STRAIGHT_CONSTANTS)")
    parts = rec.get("parts") if isinstance(rec.get("parts"), dict) else {}
    verdicts = {k: parts[k].get("verdict") if isinstance(parts.get(k), dict) else None for k in ("a", "b", "c")}
    bad = {k: v for k, v in verdicts.items() if v not in PART_VERDICTS}
    want = "PASS" if all(v == "PASS" for v in verdicts.values()) else "FAIL"
    redo = recomputed_verdicts(parts)
    wrong = {k: f"{verdicts[k]}, its values read {redo[k]}" for k in ("a", "b", "c")
             if k not in bad and redo.get(k) != verdicts[k]}
    if wrong:
        p.append(f"part verdicts that do not follow from the record's own values under v3's bars: {wrong}")
    if bad:
        p.append(f"part verdicts {bad} are not PASS / FAIL / NOT READ")
    elif parts.get("straight") != want:
        p.append(f"STRAIGHT {parts.get('straight')!r} disagrees with its parts {verdicts} (want {want})")
    return p


def _record_number(x: object) -> "float | None":
    """A record's value as a number: the strict-JSON "inf" string as inf; None, a bool or anything else as None."""
    if x == "inf":
        return math.inf
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


def _at_most(x: "float | None", bar: float) -> bool:
    """pilot4_straight._le: a finite value at or below the bar (inf, None and nan fail)."""
    return x is not None and math.isfinite(x) and x <= bar


def recomputed_verdicts(parts: dict) -> dict:
    """Fix stage: each part's verdict as pilot4_straight.py reads it from the values it records (part_a, part_b,
    part_c), under STRAIGHT_CONSTANTS' bars: (a) and (b) NOT READ with no episode (n 0), (c) NOT READ below
    C_MIN_COUNTING counting episodes; else PASS when every bar holds ("<=", inf fails), else FAIL."""
    k = STRAIGHT_CONSTANTS
    out = {}
    a = parts.get("a") if isinstance(parts.get("a"), dict) else {}
    if not a.get("n"):
        out["a"] = "NOT READ"
    else:
        med = _record_number(a.get("median_signed_deg"))
        ok = (_at_most(None if med is None else abs(med), k["A_MEDIAN_ABS_MAX_DEG"])
              and _at_most(_record_number(a.get("p95_abs_deg")), k["A_P95_ABS_MAX_DEG"]))
        out["a"] = "PASS" if ok else "FAIL"
    b = parts.get("b") if isinstance(parts.get("b"), dict) else {}
    if not b.get("n"):
        out["b"] = "NOT READ"
    else:
        ok = (_at_most(_record_number(b.get("D_p95_deg")), k["B_D_P95_MAX_DEG"])
              and _at_most(_record_number(b.get("W0_p95_deg")), k["B_W0_P95_MAX_DEG"]))
        out["b"] = "PASS" if ok else "FAIL"
    c = parts.get("c") if isinstance(parts.get("c"), dict) else {}
    n_counting = c.get("n_counting")
    if isinstance(n_counting, bool) or not isinstance(n_counting, int) or n_counting < k["C_MIN_COUNTING"]:
        out["c"] = "NOT READ"
    else:
        ok = (_at_most(_record_number(c.get("median_r_deg")), k["C_MEDIAN_MAX_DEG"])
              and _at_most(_record_number(c.get("p95_r_deg")), k["C_P95_MAX_DEG"]))
        out["c"] = "PASS" if ok else "FAIL"
    return out


def straight_summary(rec: dict) -> dict:
    """The values the scorer prints from a readable record (each part's verdict, values and counts)."""
    parts = rec["parts"]

    def keep(d: dict, keys: tuple) -> dict:
        return {k: d.get(k) for k in keys if k in d}

    return {"a": keep(parts["a"], ("verdict", "why", "n", "median_signed_deg", "p95_abs_deg", "min_deg", "max_deg",
                                   "n_full_of_panel", "note_36")),
            "b": keep(parts["b"], ("verdict", "why", "n", "D_p95_deg", "W0_p95_deg", "D_median_deg", "D_max_deg",
                                   "W0_minus_D_min_deg", "W0_minus_D_max_deg", "report_R_median_deg",
                                   "report_R_p95_deg", "report_last_mean_error_median_deg", "report_crab_median_deg",
                                   "report_end_cross_track_median_m")),
            "c": keep(parts["c"], ("verdict", "why", "n_episodes", "n_counting", "left_out_ended_before_nudge",
                                   "left_out_not_yet_passed", "median_r_deg", "p95_r_deg", "within_11_5", "within_5",
                                   "fell_after_nudge", "moved_too_little", "share_removed_median")),
            "straight": parts["straight"], "file": rec.get("file"), "tool_sha256": rec.get("tool_sha256"),
            "validation": rec.get("validation"), "plot": (rec.get("files") or {}).get("plot")}


def _fmt(x: object, nd: int = 1) -> str:
    """A number to nd decimals; the record's "inf" string as it is; None, a bool or anything else as n/a."""
    if isinstance(x, str):
        return x
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return "n/a"
    if not math.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    return f"{x:.{nd}f}"


def c_counts_text(c: dict) -> str:
    return (f"counting {c.get('n_counting')} of {c.get('n_episodes')} (left out: "
            f"{c.get('left_out_ended_before_nudge')} ended before the nudge, "
            f"{c.get('left_out_not_yet_passed')} not yet past the prey at 6.5 s)")


def straight_text(sm: dict) -> str:
    """One line: the three parts with their values, (c)'s counts (and its why when NOT READ), and STRAIGHT."""
    a, b, c = sm["a"], sm["b"], sm["c"]
    ta = f"(a) {a.get('verdict')}" + (
        f": {a.get('why')}" if a.get("verdict") == "NOT READ" else
        f": median {_fmt(a.get('median_signed_deg'))}, p95 |yaw| {_fmt(a.get('p95_abs_deg'))} deg (n {a.get('n')})")
    tb = f"(b) {b.get('verdict')}" + (
        f": {b.get('why')}" if b.get("verdict") == "NOT READ" else
        f": D p95 {_fmt(b.get('D_p95_deg'))}, W0 p95 {_fmt(b.get('W0_p95_deg'))} deg (n {b.get('n')}; report only: R "
        f"median {_fmt(b.get('report_R_median_deg'))}, p95 {_fmt(b.get('report_R_p95_deg'))}, last mean error median "
        f"{_fmt(b.get('report_last_mean_error_median_deg'))}, crab median {_fmt(b.get('report_crab_median_deg'))} deg, "
        f"end cross-track median {_fmt(b.get('report_end_cross_track_median_m'), 2)} m)")
    tc = f"(c) {c.get('verdict')}: " + (
        f"{c.get('why')}; " if c.get("verdict") == "NOT READ" else
        f"median r {_fmt(c.get('median_r_deg'))}, p95 r {_fmt(c.get('p95_r_deg'))} deg, within 11.5 deg "
        f"{c.get('within_11_5')}; ") + c_counts_text(c)
    return f"{ta} | {tb} | {tc} | STRAIGHT {sm.get('straight')}"


def decide_seed(rule: "str | None", video: "str | None", parts: dict, look: "str | None") -> tuple:
    """The seed outcome, first that applies (v3 "Seed outcome"), for known inputs: rule = pilot 3's seed outcome at the
    deciding checkpoint (None: missing); video "agree" | "disagree"; parts {"a", "b", "c"} -> PASS | FAIL | NOT READ;
    look "straight" | "not straight". Returns (outcome, label): the NOT WALKS sub-label, which part failed, or (c)'s
    reading."""
    if rule is None:
        return None, "missing"
    if rule != "WALKS":
        return "NOT WALKS", rule
    if video == "disagree":
        return "NOT WALKS", VIDEO_DISAGREE_LABEL
    if parts["a"] != "PASS" or parts["b"] != "PASS":
        return "CURVED WALKS", ", ".join(f"({k}) {parts[k]}" for k in ("a", "b") if parts[k] != "PASS")
    if look == "not straight":
        return "CURVED WALKS", LOOK_DISAGREE_LABEL
    if parts["c"] == "PASS":
        return "STRAIGHT WALKS", "(a), (b) and (c) PASS, the look agrees"
    return "STRAIGHT NO RECOVERY", f"(c) {parts['c']}"


def awaited_inputs(rule: "str | None", video: "str | None", parts: "dict | None", look: "str | None") -> list:
    """What a WALKS seed still waits for: its video review; its straightness record (unless the review disagrees); its
    look (unless the review disagrees, or the record shows (a) or (b) not passing)."""
    if rule != "WALKS":
        return []
    out = ["video review"] if video is None else []
    if video == "disagree":
        return out
    if parts is None:
        out.append("straightness record")
    if look is None and (parts is None or parts["a"] == parts["b"] == "PASS"):
        out.append("straightness look")
    return out


def straight_seed_outcome(rule: "str | None", video: "str | None", parts: "dict | None", look: "str | None") -> dict:
    """One seed's outcome. Every input not yet given (a WALKS seed's video review, its straightness record as parts
    None, its look) is tried each way; the outcome is decided when every try gives one outcome, else it is "pending"
    and its possible outcomes are listed. A missing seed (rule None) reads None."""
    walks = rule == "WALKS"
    videos = [video] if (video is not None or not walks) else ["agree", "disagree"]
    tries = [parts] if parts is not None else list(_ANY_PARTS)
    looks = [look] if look is not None else list(LOOK_VERDICTS)
    possible: dict = {}
    for v, pa, lk in itertools.product(videos, tries, looks):
        o, label = decide_seed(rule, v, pa, lk)
        possible.setdefault(o, label)
    awaited = awaited_inputs(rule, video, parts, look)
    if len(possible) == 1:
        ((o, label),) = possible.items()
        if awaited and o is not None:
            label += f" (whatever the awaited {', '.join(awaited)} reads)"
        return {"outcome": o, "label": label, "possible": [o], "awaited": awaited}
    return {"outcome": PENDING, "label": None, "possible": sorted(possible, key=str), "awaited": awaited}


def straight_pilot_outcome(possible: dict) -> dict:
    """The pilot outcome (v3 "Pilot outcome"): pilot 3's pilot_outcome over the seed outcomes, so NOT WALKS shares
    across its sub-labels. possible: every launched seed -> its possible outcomes ([outcome] when decided, [None] when
    missing, more than one while pending). Every combination of the pending seeds' outcomes is tried: when all give one
    pilot outcome it stands (final); otherwise PENDING_REVIEW, with the possible pilot outcomes listed."""
    pending = sorted(s for s, p in possible.items() if len(p) > 1)
    fixed = {s: p[0] for s, p in possible.items() if len(p) == 1}
    if not pending:
        res = pilot_outcome(fixed)
        return dict(res, final=True, pending=[])
    outcomes: dict = {}
    for combo in itertools.product(*(possible[s] for s in pending)):
        chosen = dict(zip(pending, combo, strict=True))
        res = pilot_outcome({s: chosen[s] if s in chosen else fixed[s] for s in possible})
        outcomes.setdefault(res["outcome"], res)
    names = ["s%d" % s for s in pending]
    if len(outcomes) == 1:
        (res,) = outcomes.values()
        res = dict(res, final=True, pending=pending)
        res["basis"] += f"; the same whatever the pending seeds {names} read"
        return res
    possible_outcomes = sorted(str(o) for o in outcomes)
    return {"outcome": PILOT_PENDING, "final": False, "pending": pending, "possible_outcomes": possible_outcomes,
            "basis": f"{names} pending (video review, straightness record or look); the outcome is one of "
                     f"{possible_outcomes}",
            "missing": sorted(s for s, o in fixed.items() if o is None)}


def straight_final_reading(possible: dict, pilot: dict, closed: list) -> dict:
    """The final reading (v3 "Pilot outcome", pilot 3's [P3 L648-653]): when the pilot outcome is INCOMPLETE (or none)
    and every missing seed is CLOSED, the scored seeds alone by the same rule (straight_pilot_outcome), as if only they
    had been launched: two that differ read MIXED; one reads its own outcome, "one seed only"; none reads NO OUTCOME.
    Never the pilot outcome itself."""
    missing = sorted(s for s, p in possible.items() if list(p) == [None])
    if pilot.get("outcome") not in ("INCOMPLETE", None) or not missing:
        return {"applies": False, "why": "the pilot outcome is not INCOMPLETE" if pilot.get("outcome") is not None
                else "no seed is missing"}
    still_open = [s for s in missing if s not in closed]
    if still_open:
        return {"applies": False, "why": f"missing seeds not CLOSED: {['s%d' % s for s in still_open]}"}
    scored = {s: p for s, p in possible.items() if list(p) != [None]}
    if scored:
        res = straight_pilot_outcome(scored)
        reading, final, basis = res["outcome"], bool(res.get("final", True)), res["basis"]
    else:
        reading, final, basis = "NO OUTCOME", True, "no scored seed"
    one = len(scored) == 1
    text = (f"FINAL READING: {reading}{' (one seed only)' if one else ''} from {len(scored)} scored seeds of "
            f"{len(possible)} (closed: {', '.join('s%d' % s for s in missing)})")
    return {"applies": True, "reading": reading, "final": final and reading != PILOT_PENDING, "k_scored": len(scored),
            "one_seed_only": one, "closed": missing, "basis": basis, "text": text}


def straight_existence(per_seed: dict, closed: "list | None" = None) -> dict:
    """The existence reading (v3 "Pilot outcome"): STRAIGHT WALK EXISTS when at least one seed is STRAIGHT WALKS,
    whatever the pilot outcome; else PENDING while a pending seed could still become one; else NONE SEEN. Final only
    when no seed is pending and every missing seed is CLOSED. per_seed: seed -> straight_seed_outcome's dict."""
    a = sorted(s for s, r in per_seed.items() if r["outcome"] == "STRAIGHT WALKS")
    pending = sorted(s for s, r in per_seed.items() if r["outcome"] == PENDING)
    could = [s for s in pending if "STRAIGHT WALKS" in per_seed[s]["possible"]]
    open_missing = sorted(s for s, r in per_seed.items() if r["outcome"] is None and s not in (closed or []))
    reading = "STRAIGHT WALK EXISTS" if a else ("PENDING" if could else "NONE SEEN")
    return {"reading": reading, "A": a, "k": len(a), "launched": len(per_seed), "pending": pending,
            "pending_could_be_straight_walks": could, "open_missing": open_missing,
            "final": not pending and not open_missing}


def straight_count(inputs: dict) -> dict:
    """Report only (v3 Q23): the seeds that are WALKS by the rule, video-agreed, (a) and (b) PASS and whose look reads
    "straight", and the seeds that could still join them. inputs: seed -> (rule, video, parts or None, look)."""
    counted, could = [], []
    for s, (rule, video, parts, look) in sorted(inputs.items()):
        if rule != "WALKS" or video == "disagree" or look == "not straight":
            continue
        if parts is not None and not parts["a"] == parts["b"] == "PASS":
            continue
        (counted if (video == "agree" and parts is not None and look == "straight") else could).append(s)
    n = len(inputs)
    text = (f"{len(counted)} of {n} seeds are WALKS, video-agreed, pass (a) and (b) and the look agrees "
            f"({['s%d' % s for s in counted]})"
            + (f"; could still join (review, record or look pending): {['s%d' % s for s in could]}" if could else ""))
    return {"k": len(counted), "launched": n, "seeds": counted, "could_still_count": could, "text": text}


def look_required(rule: "str | None", video: "str | None", parts: "dict | None") -> "str | None":
    """Whether the straightness look is required (v3: every seed WALKS, video-agreed and passing (a) and (b), whatever
    (c) reads); None when it is not."""
    if rule != "WALKS" or video == "disagree":
        return None
    if parts is None:
        return "not yet known: no straightness record read at the deciding checkpoint"
    if not parts["a"] == parts["b"] == "PASS":
        return None
    return "REQUIRED" if video == "agree" else "REQUIRED if the video review agrees (pending)"


# ---------------------------------------------------------------- report only: pilot 3's 8M references (constants)

P3_8M_SOURCE = (
    "pilot 3's judged 8,000,000 cells, scored by the pinned pilot3_score.py (sha256 286ce04a...): "
    "SP/work/pilot3/judging/final_score/score_final.json (sha256 "
    "dd023b229d66a18421ebe4edf1143848b7b2add822476d84296ff3ef6b960744), each seed's 'medians' (REPORT_KEYS over the 40 "
    "panel episodes), 'counts' and 'probe_classes'; equal to the medians recomputed from SP/work/pilot3/judging/OUT/"
    "s<seed>/8000000/probe.json (s42 b82b0f8c..., s52 09a919f1..., s60 0e4617ea...) [measured; the run log's 16:29 UTC "
    "note of 2026-11-05 and v3 'Where pilot 3 ended' print them rounded]")
P3_8M = {
    42: {"counts": {"WALK": 40, "HOP": 0, "STILL": 0, "FALL": 0, "OTHER": 0}, "probe_classes": {"walk": 40},
         "medians": {"speed_displacement_mps": 2.7192661167192784, "floor_touchdowns_per_s_all_feet": 4.833333333333334,
                     "floor_alternation_index": 1.0, "floor_simultaneous_fraction": 0.0,
                     "floor_lead_swaps_per_stride": 1.9425763399285736, "floor_flight_fraction": 0.0,
                     "floor_single_support_fraction": 0.9416666666666667, "floor_stride_frequency_hz": 2.47332453714471,
                     "floor_stride_length_over_L": 1.2324115131923228, "left_leads_fraction": 0.3288888888888889,
                     "froude": 0.8485635248858403, "jointlag_frac_hip_pitch": 0.4634146341463415,
                     "floor_duty_L": 0.4838888888888889, "floor_duty_R": 0.5738888888888889,
                     "floor_swing_mean_s_L": 0.21232683982683984, "floor_swing_mean_s_R": 0.17047619047619045,
                     "speed_forward_vel_mps": 2.701325344455515, "yaw_change_deg": -24.702163535330687}},
    52: {"counts": {"WALK": 39, "HOP": 0, "STILL": 0, "FALL": 1, "OTHER": 0},
         "probe_classes": {"walk": 39, "slide (moves without stepping)": 1},
         "medians": {"speed_displacement_mps": 2.499960918912708, "floor_touchdowns_per_s_all_feet": 4.0,
                     "floor_alternation_index": 1.0, "floor_simultaneous_fraction": 0.0,
                     "floor_lead_swaps_per_stride": 1.9658760164748126, "floor_flight_fraction": 0.0,
                     "floor_single_support_fraction": 0.96, "floor_stride_frequency_hz": 2.0395928155434833,
                     "floor_stride_length_over_L": 1.354989209530121, "left_leads_fraction": 0.5505555555555556,
                     "froude": 0.6825939022799071, "jointlag_frac_hip_pitch": 0.5416666666666666,
                     "floor_duty_L": 0.5138888888888888, "floor_duty_R": 0.525,
                     "floor_swing_mean_s_L": 0.23411764705882354, "floor_swing_mean_s_R": 0.23833333333333334,
                     "speed_forward_vel_mps": 2.4227806639612024, "yaw_change_deg": -47.8811637201546}},
    60: {"counts": {"WALK": 40, "HOP": 0, "STILL": 0, "FALL": 0, "OTHER": 0}, "probe_classes": {"walk": 40},
         "medians": {"speed_displacement_mps": 2.224758695331208, "floor_touchdowns_per_s_all_feet": 5.444444444444445,
                     "floor_alternation_index": 1.0, "floor_simultaneous_fraction": 0.0,
                     "floor_lead_swaps_per_stride": 2.013926509923271, "floor_flight_fraction": 0.0,
                     "floor_single_support_fraction": 0.9022222222222223,
                     "floor_stride_frequency_hz": 2.7420766411806063,
                     "floor_stride_length_over_L": 0.9203266680463682, "left_leads_fraction": 0.6355555555555555,
                     "froude": 0.5653043076894544, "jointlag_frac_hip_pitch": 0.6, "floor_duty_L": 0.45,
                     "floor_duty_R": 0.6455555555555555, "floor_swing_mean_s_L": 0.19898333333333335,
                     "floor_swing_mean_s_R": 0.128775, "speed_forward_vel_mps": 2.2048364691376525,
                     "yaw_change_deg": -23.758085142416938}},
}
P3_8M_SHORT = {"speed_displacement_mps": "v", "speed_forward_vel_mps": "fwd", "floor_alternation_index": "alt",
               "floor_simultaneous_fraction": "sim", "floor_lead_swaps_per_stride": "swaps",
               "floor_flight_fraction": "flight", WALK_HIP_LAG_KEY: "hip lag",
               "floor_single_support_fraction": "single", "floor_stride_frequency_hz": "stride Hz",
               "floor_stride_length_over_L": "stride/L", "floor_duty_L": "duty L", "floor_duty_R": "duty R",
               "floor_swing_mean_s_L": "swing L", "floor_swing_mean_s_R": "swing R",
               "floor_touchdowns_per_s_all_feet": "td/s", "left_leads_fraction": "left leads", "froude": "Froude",
               "yaw_change_deg": "yaw"}


def p3_8m_comparison(cell: "dict | None", seed: int) -> "dict | None":
    """Report only (v3 "Report only"): a scored cell's medians, counts, probe classes and guard counts beside pilot
    3's 8M constants for the same seed; None for a seed without a reference or without a cell."""
    ref = P3_8M.get(seed)
    if ref is None or cell is None:
        return None
    return {"source": P3_8M_SOURCE,
            "medians": {k: {"pilot4": cell["medians"].get(k), "pilot3_8m": ref["medians"][k]} for k in REPORT_KEYS},
            "counts": {"pilot4": cell["counts"], "pilot3_8m": ref["counts"]},
            "probe_classes": {"pilot4": cell["probe_classes"], "pilot3_8m": ref["probe_classes"]},
            "other_passing_contact_bar": cell["other_passing_contact_bar"],
            "walk_with_short_swing": cell["walk_with_short_swing"]}


def p3_8m_text(cmp: dict) -> str:
    nd = {"yaw_change_deg": 1, "speed_displacement_mps": 3, "speed_forward_vel_mps": 3}
    return "; ".join(f"{P3_8M_SHORT[k]} {_fmt(v['pilot4'], nd.get(k, 2))} ({_fmt(v['pilot3_8m'], nd.get(k, 2))})"
                     for k, v in cmp["medians"].items())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--seeds", required=True, help="the launched seeds, comma separated (e.g. 42,52,60)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--primary", type=int, default=PRIMARY)
    ap.add_argument("--validation", action="store_true",
                    help="accept validation-only judges and validation straightness records (never for the verdict)")
    ap.add_argument("--video-review", default=None, help="JSON of the maintainer's per-seed video verdicts")
    ap.add_argument("--straight", action="append", default=[],
                    help="pilot4_straight.py seed-mode JSON (repeatable): one record per seed and checkpoint")
    ap.add_argument("--straight-review", default=None,
                    help="JSON of the maintainer's per-seed straightness look ('straight' | 'not straight')")
    ap.add_argument("--closed-seeds", default=None,
                    help="launched seeds the run log CLOSES (restart spent, run lost or refused): enables the final "
                         "reading")
    ap.add_argument("--habit", action="append", default=[], help="pilot4_habit.py JSON (repeatable): report only")
    a = ap.parse_args(argv)
    out = Path(a.out)
    seeds = sorted({int(s) for s in a.seeds.split(",") if s.strip()})
    review = read_video_review(a.video_review)
    look = read_straight_review(a.straight_review)
    closed = read_closed_seeds(a.closed_seeds, seeds)
    habit = read_habit(a.habit)
    straight = read_straight(a.straight)
    on_disk = sorted(int(p.name[1:]) for p in out.glob("s*") if p.is_dir() and p.name[1:].isdigit())
    undeclared = [s for s in on_disk if s not in seeds]
    result = {"criteria": {"still_speed_mps": STILL_SPEED,
                           "walk": {"min_alternation": WALK_MIN_ALTERNATION,
                                    "min_lead_swaps_per_stride": WALK_MIN_LEAD_SWAPS, "max_flight": WALK_MAX_FLIGHT,
                                    "hip_lag_key": WALK_HIP_LAG_KEY, "hip_lag_band": list(WALK_HIP_LAG_BAND),
                                    "stride_key": WALK_STRIDE_KEY,
                                    "max_stride_frequency_hz": WALK_MAX_STRIDE_HZ, "duty_keys": list(WALK_DUTY_KEYS),
                                    "max_foot_duty": WALK_MAX_DUTY},
                           "hop_min_simultaneous": HOP_MIN_SIMULTANEOUS,
                           "hop_min_touchdowns_per_s": HOP_MIN_TOUCHDOWNS_PER_S, "episodes": EPISODES,
                           "seed": {"walks": [SEED_WALKS_MIN, SEED_WALKS_MAX_FALLS], "hops": SEED_HOPS_MIN,
                                    "falls": SEED_FALLS_MIN, "still": SEED_STILL_MIN},
                           "video_review": {"required_for": "WALKS", "watch": "every WALK episode, and every episode "
                                            f"that is not STILL with a foot's floor duty > {VIDEO_DUTY}",
                                            "notebook_replay": NOTEBOOK_REPLAY},
                           "straight": {"record_schema": STRAIGHT_SCHEMA, "part_verdicts": list(PART_VERDICTS),
                                        "look_verdicts": list(LOOK_VERDICTS), "seed_outcomes": list(SEED_OUTCOMES),
                                        "look_required_for": "WALKS, video-agreed, (a) and (b) PASS"}},
              "primary": a.primary, "validation": a.validation, "launched_seeds": seeds,
              "undeclared_seed_dirs_ignored": undeclared, "seeds": {}, "per_seed": {}, "per_seed_primary": {},
              "per_seed_reviewed": {}, "video_review_file": a.video_review,
              "video_review_verdicts": {str(k): v for k, v in sorted(review.items())}, "video_review_required": {},
              "straight_files": list(a.straight), "straight_review_file": a.straight_review,
              "straight_review_verdicts": {str(k): v for k, v in sorted(look.items())}, "straight": {},
              "straightness_look_required": {}, "seed_outcomes": {}}
    inputs = {}
    for s in seeds:
        sdir = out / f"s{s}"
        steps_all = {*CHECKPOINTS, a.primary}
        if sdir.is_dir():
            steps_all |= {int(p.name) for p in sdir.iterdir() if p.is_dir() and p.name.isdigit()}
        cells = {}
        for steps in sorted(steps_all):
            p = sdir / str(steps) / "probe.json"
            if not p.is_file():
                cells[steps] = None
                continue
            b = p.with_name("binding.json")
            binding = json.loads(b.read_text()) if b.is_file() else None
            cells[steps] = score_cell(json.loads(p.read_text()), binding, a.validation)
        mixed = one_run_problem(cells)
        if mixed:
            for c in cells.values():
                if c is not None:
                    c["problems"].append(mixed)
                    c["outcome"] = None
        result["seeds"][str(s)] = {str(k): v for k, v in cells.items()}
        sr = final_fallback_basis(seed_result(cells, a.primary), cells)
        verdict = (review.get(s) or {}).get("verdict")
        rev_outcome, rev_state = reviewed_outcome(sr["outcome"], verdict)
        sr["video_review"] = rev_state
        sr["outcome_after_video_review"] = rev_outcome
        result["per_seed"][str(s)] = sr
        result["per_seed_primary"][str(s)] = sr["outcome"]
        result["per_seed_reviewed"][str(s)] = rev_outcome
        cell = cells.get(sr["steps"]) if sr.get("steps") is not None else None
        result["video_review_required"][str(s)] = None if cell is None else {
            "steps": sr["steps"], "episodes": cell["video_watch"],
            "still_with_foot_duty_over_0_8": cell["still_with_foot_duty_over_0_8"],
            "render": f"tools/pilot4_video.py OUT/s{s}/{sr['steps']} --from-score <this JSON>",
            "notebook_replay": NOTEBOOK_REPLAY.replace("<RUN_ID>", f"pilot4_walk_s{s}")
                               + f" (or pilot4_walk_s{s}_b if the seed was restarted)",
            "required": sr["outcome"] == "WALKS", "review_state": rev_state}
        # STRAIGHT: the record of the deciding cell decides; records of other cells are printed, report only
        recs = {}
        for (rs, steps), rec in sorted(straight.items()):
            if rs != s:
                continue
            probs = straight_record_problems(rec, s, steps, sdir / str(steps), a.validation)
            recs[str(steps)] = {"file": rec["file"], "deciding": steps == sr.get("steps"), "read": not probs,
                                "problems": probs, "summary": None if probs else straight_summary(rec)}
        deciding = recs.get(str(sr["steps"])) if sr.get("steps") is not None else None
        parts = ({k: deciding["summary"][k]["verdict"] for k in ("a", "b", "c")}
                 if deciding is not None and deciding["read"] else None)
        lk = (look.get(s) or {}).get("verdict")
        so = straight_seed_outcome(sr["outcome"], verdict, parts, lk)
        if so["outcome"] == "STRAIGHT NO RECOVERY" and parts["c"] == "NOT READ":
            c = deciding["summary"]["c"]
            so["label"] += f": {c.get('why')}; {c_counts_text(c)}"
        so.update(steps=sr.get("steps"), parts=parts, video_review=verdict, look=lk,
                  straight_record=None if deciding is None else deciding["file"],
                  straight_record_problems=None if deciding is None else deciding["problems"])
        result["straight"][str(s)] = recs
        result["seed_outcomes"][str(s)] = so
        need = look_required(sr["outcome"], verdict, parts)
        result["straightness_look_required"][str(s)] = None if need is None else {
            "steps": sr["steps"], "state": need, "look": lk,
            "notebook_replay": result["video_review_required"][str(s)]["notebook_replay"],
            "plot": (deciding["summary"]["plot"] if deciding is not None and deciding["read"] else None)}
        inputs[s] = (sr["outcome"], verdict, parts, lk)
    result["straight_records_ignored"] = sorted(rec["file"] for (rs, _), rec in straight.items() if rs not in seeds)
    reports = {}
    for p in sorted((out / "off_schedule").glob("*/s*/*/probe.json")):
        b = p.with_name("binding.json")
        cell = score_cell(json.loads(p.read_text()), json.loads(b.read_text()) if b.is_file() else None, a.validation)
        reports[str(p.parent.relative_to(out))] = {"counts": cell["counts"], "outcome": cell["outcome"],
                                                   "problems": cell["problems"]}
    result["off_schedule_reports"] = reports
    result["retired_cells_never_scored"] = sorted(str(p.parent.relative_to(out))
                                                  for p in (out / "retired").glob("*/s*/*/probe.json"))
    rule_per_seed = {s: result["per_seed_primary"][str(s)] for s in seeds}
    gait = pilot_outcome_reviewed(rule_per_seed, review) if seeds else {
        "outcome": None, "basis": "no launched seeds given", "missing": [], "final": True, "pending_video_review": []}
    result["closed_seeds"] = closed
    result["gait_reading"] = {"by_rule": pilot_outcome(rule_per_seed) if seeds else None, "reviewed": gait,
                              "final_reading": final_reading(rule_per_seed, review, gait, closed)}
    possible = {s: result["seed_outcomes"][str(s)]["possible"] for s in seeds}
    result["pilot"] = straight_pilot_outcome(possible) if seeds else {
        "outcome": None, "basis": "no launched seeds given", "missing": [], "final": True, "pending": []}
    result["final_reading"] = straight_final_reading(possible, result["pilot"], closed)
    result["report_only_count"] = straight_count(inputs)
    result["existence"] = straight_existence({s: result["seed_outcomes"][str(s)] for s in seeds}, closed)
    deciding_sha = {}
    for s in seeds:
        steps = result["per_seed"][str(s)].get("steps")
        cell = result["seeds"][str(s)].get(str(steps)) if steps is not None else None
        deciding_sha[s] = (cell or {}).get("checkpoint_sha256")
    result["habit_files"] = list(a.habit)
    result["habit_report_only"] = {str(s): habit_report(s, habit.get(s), deciding_sha.get(s)) for s in seeds}
    result["p3_8m_reference"] = {}
    for s in seeds:
        steps = result["per_seed"][str(s)].get("steps")
        cell = result["seeds"][str(s)].get(str(steps)) if steps is not None else None
        result["p3_8m_reference"][str(s)] = p3_8m_comparison(cell, s)

    print(f"Walking pilot 4 score ({out}); launched seeds {seeds}; primary checkpoint {a.primary:,}"
          + ("  [--validation: validation judges and records accepted; never for the verdict]" if a.validation else ""))
    for s, cells in result["seeds"].items():
        for steps, cell in cells.items():
            if cell is None:
                print(f"  s{s} {int(steps):>9,}: missing")
                continue
            m = cell["medians"]
            print(f"  s{s} {int(steps):>9,}: {cell['outcome'] or 'NOT SCORED'}  {cell['counts']}  "
                  f"v {m['speed_displacement_mps']:.3f} m/s (fwd {m['speed_forward_vel_mps']:.3f}), "
                  f"alt {m['floor_alternation_index']:.2f}, "
                  f"sim {m['floor_simultaneous_fraction']:.2f}, swaps {m['floor_lead_swaps_per_stride']:.2f}, "
                  f"flight {m['floor_flight_fraction']:.2f}, hip lag {m[WALK_HIP_LAG_KEY]:.2f}, "
                  f"stride {m['floor_stride_frequency_hz']:.2f} Hz, duty L/R {m['floor_duty_L']:.2f}/"
                  f"{m['floor_duty_R']:.2f}; OTHER passing the contact bar, by rejecting guards: "
                  f"{ {k: v for k, v in cell['other_passing_contact_bar'].items() if v} }"
                  + (f"  PROBLEMS: {cell['problems']}" if cell["problems"] else "")
                  + ("  [final-model fallback]" if cell.get("final_fallback") else ""))
            st = cell["still_report"]
            if st["n"]:
                print(f"      STILL {st['n']}: {st['stepping_at_0_5_touchdowns_per_s']} at >= {STILL_STEPPING_TD} "
                      f"touchdowns/s; medians {st['median_touchdowns_per_s']:.2f} touchdowns/s, alternation "
                      f"{st['median_alternation']:.2f}, |yaw change| {st['median_abs_yaw_change_deg']:.1f} deg "
                      "(report only)")
        sr = result["per_seed"][s]
        print(f"  s{s} gait (pilot 3's seed outcome): {sr['outcome'] or 'none'} ({sr['basis']}); video review: "
              f"{sr['video_review']}" + (f" -> {sr['outcome_after_video_review']}" if sr["video_review"] == "disagree"
                                         else ""))
        so = result["seed_outcomes"][s]
        walks = sr["outcome"] == "WALKS"
        for steps, rec in result["straight"][s].items():
            where = "deciding" if rec["deciding"] else "not the deciding checkpoint: report only"
            if rec["deciding"] and not walks:
                where += "; the seed is not WALKS: report only"
            body = straight_text(rec["summary"]) if rec["read"] else f"record not read: {'; '.join(rec['problems'])}"
            print(f"  s{s} STRAIGHT at {int(steps):,} ({where}): {body}")
        if walks and so["straight_record"] is None:
            print(f"  s{s} STRAIGHT at {sr['steps']:,}: no straightness record given (pilot4_straight.py seed --cell "
                  f"OUT/s{s}/{sr['steps']})")
        if so["outcome"] == PENDING:
            print(f"  s{s} SEED OUTCOME: pending ({', '.join(so['awaited'])}); possible: {so['possible']}")
        else:
            print(f"  s{s} SEED OUTCOME: {so['outcome'] or 'none (missing)'}"
                  + (f" ({so['label']})" if so["outcome"] is not None else ""))
    if undeclared:
        print(f"  ignored (not a launched seed): {['s%d' % s for s in undeclared]}")
    for f in result["straight_records_ignored"]:
        print(f"  ignored straightness record (not a launched seed): {f}")
    for where, rep in result["off_schedule_reports"].items():
        print(f"  reported, never scored: {where} {rep['counts']}")
    for where in result["retired_cells_never_scored"]:
        print(f"  retired run's cell, never scored: {where}")
    print("VIDEO REVIEW REQUIRED (criteria_pilot4 \"Video review\", pilot 3's rule): for every seed that is WALKS by "
          "the rule, even when its verdict cannot change the pilot outcome; a seed's WALKS counts only when the "
          "maintainer has watched these and agrees they show alternating single-leg steps; record the verdicts in a "
          "JSON file ({\"42\": \"agree\" | \"disagree\", ...}) and pass --video-review.")
    for s in seeds:
        req = result["video_review_required"][str(s)]
        if req is None:
            print(f"  s{s}: no scored checkpoint yet, nothing to list")
            continue
        walk = [str(e["episode"]) + ("*" if e["short_swing"] else "") for e in req["episodes"] if e["why"] == "WALK"]
        duty = [f"{e['episode']} ({e['label']}, {e['foot_duty_max']})" for e in req["episodes"] if e["why"] != "WALK"]
        state = (f"WALKS by the rule: REQUIRED, review {req['review_state']}" if req["required"]
                 else f"{result['per_seed'][str(s)]['outcome']}: not required, listed for reference")
        print(f"  s{s} ({state}) at {req['steps']:,}: panel episodes (reset seed 3042 + episode; render with "
              f"{req['render']}): "
              f"WALK [{', '.join(walk) or 'none'}]; foot duty > {VIDEO_DUTY} [{'; '.join(duty) or 'none'}]"
              + (f"; {req['still_with_foot_duty_over_0_8']} STILL episodes over the duty line not listed"
                 if req["still_with_foot_duty_over_0_8"] else "")
              + f". Notebook replay: {req['notebook_replay']}"
              + ("  (* = a foot's mean floor swing <= 0.03 s, the short-lift signature: watch it at quarter speed)"
                 if any(e["short_swing"] for e in req["episodes"] if e["why"] == "WALK") else ""))
    print("STRAIGHTNESS LOOK REQUIRED (criteria_pilot4 \"Video review\"): for every seed that is WALKS, video-agreed "
          "and passes (a) and (b), whatever (c) reads: the notebook replay and the straightness tool's top-down paths "
          "and heading curves; record {\"42\": \"straight\" | \"not straight\", ...} and pass --straight-review; "
          "\"not straight\" makes the seed CURVED WALKS.")
    for s in seeds:
        req = result["straightness_look_required"][str(s)]
        if req is None:
            print(f"  s{s}: not required")
            continue
        plot = req["plot"] or {}
        print(f"  s{s} at {req['steps']:,}: {req['state']}; look {req['look'] or 'pending'}; plot "
              f"{plot.get('path') or 'n/a'} ({plot.get('sha256') or 'n/a'}); replay {req['notebook_replay']}")
    pil = result["pilot"]
    print(f"PILOT OUTCOME: {pil['outcome']} ({pil['basis']})"
          + (f"; missing or invalid: {['s%d' % s for s in pil['missing']]}" if pil.get("missing") else "")
          + ("" if pil.get("final", True) else "  NOT FINAL: review, record or look pending")
          + ("  [validation]" if a.validation else ""))
    subs = {s: result["seed_outcomes"][str(s)]["label"] for s in seeds
            if result["seed_outcomes"][str(s)]["outcome"] == "NOT WALKS"}
    if subs:
        print(f"  NOT WALKS sub-labels: {', '.join(f's{s} {lab}' for s, lab in subs.items())}")
    fr = result["final_reading"]
    if fr["applies"]:
        print(fr["text"] + f" [{fr['basis']}]" + ("" if fr["final"] else "  NOT FINAL: review, record or look pending"))
    g = result["gait_reading"]
    print(f"GAIT READING (pilot 3's pilot outcome in this env; printed beside the pilot outcome, never instead of it): "
          f"{g['reviewed']['outcome']} ({g['reviewed']['basis']})"
          + ("" if g["reviewed"].get("final", True) else "  NOT FINAL: video review pending")
          + (f"; {g['final_reading']['text']}" if g["final_reading"]["applies"] else ""))
    print(f"REPORT ONLY (Q23): {result['report_only_count']['text']}")
    ex = result["existence"]
    print(f"EXISTENCE: {ex['reading']} (A = {['s%d' % s for s in ex['A']]}, k = {ex['k']} of {ex['launched']}; pending "
          f"{['s%d' % s for s in ex['pending']]}, of which could still be STRAIGHT WALKS "
          f"{['s%d' % s for s in ex['pending_could_be_straight_walks']]})"
          + (f"  [not settled: {['s%d' % s for s in ex['open_missing']]} still missing and not closed]"
             if ex["open_missing"] else "") + ("" if ex["final"] else "  NOT FINAL"))
    for s in seeds:
        st = result["habit_report_only"][str(s)]
        print(f"REPORT ONLY habit (pilot4_habit.py; picks nothing): s{s} {habit_text(st)}")
    for s in seeds:
        cmp = result["p3_8m_reference"][str(s)]
        if cmp is None:
            continue
        steps = result["per_seed"][str(s)]["steps"]
        print(f"REPORT ONLY s{s} at {steps:,} against pilot 3's 8M (in parentheses; constants P3_8M): "
              f"{p3_8m_text(cmp)}; counts {cmp['counts']['pilot4']} ({cmp['counts']['pilot3_8m']}); classes "
              f"{cmp['probe_classes']['pilot4']} ({cmp['probe_classes']['pilot3_8m']}); OTHER passing the contact bar "
              f"{ {k: v for k, v in cmp['other_passing_contact_bar'].items() if v} }; short-swing WALKs "
              f"{cmp['walk_with_short_swing']}")
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
