"""Score walking pilot 3's gait-probe panels against its fixed criteria (criteria_pilot3.txt).

Scratch, never merged: pilot 1's pilot_score.py (sha256 b4297e13...) adapted for walking pilot 3. Every label, guard,
threshold and outcome rule is pilot 1's: seed_outcome, pilot_outcome, reviewed_outcome and pilot_outcome_reviewed are
byte-identical in their bodies. Changed: CHECKPOINTS (2,000,000, 4,000,000, 6,000,000, 8,000,000), PRIMARY 8,000,000,
the replay RUN_ID pilot3_walk_s<seed>, the renderer pilot3_video.py. Added, in new functions, under the criteria's
"What each outcome means": the existence reading, the final reading (--closed-seeds) and the Q4 reading (--habit,
pilot3_habit.py's JSON); "VIDEO REVIEW REQUIRED" for every seed that is WALKS by the rule; a seed decided at its final
pair (pilot3_judge.py --final-fallback) is reported "decided at the final model (fallback)"; report only,
speed_forward_vel_mps beside the displacement speed and, for STILL episodes, touchdowns per second, alternation and
yaw change, with the count of STILL episodes at 0.5 touchdowns per second or more.

Plain Python, no repository import. Reads what pilot3_judge.py wrote:
    OUT/s<seed>/<steps>/probe.json and binding.json      (one 40-episode panel per seed and checkpoint)
and prints, per seed and checkpoint, the per-episode label counts and the seed outcome, then the episodes the video
review must watch ("VIDEO REVIEW REQUIRED"), then the pilot outcome, the final reading, the existence reading and Q4.

    python3 pilot3_score.py OUT --seeds 42,52,60 [--json OUT/score.json] [--primary 8000000]
                                [--video-review REVIEW.json] [--closed-seeds 52,60] [--habit HABIT.json ...]

--seeds names the seeds the pilot launched (required): a launched seed with nothing judged counts as missing, and a
directory for a seed not named is reported and ignored.

Episode labels (first match wins; every metric is the probe's floor-truth channel over its gait window, which
starts 1 s after reset; an undefined metric fails the test that reads it):
  FALL   the episode ended before 1,000 steps
  HOP    at least 0.5 touchdowns per second (all feet) and a simultaneous-touchdown fraction >= 0.5
         (classify_biped's hop rule), at any speed
  STILL  window displacement speed below 0.10 m/s
  WALK   all six: alternation index >= 0.6, lead swaps per stride >= 1.0, flight fraction <= 0.2 (the gait plan's
         section 5.5 trex locomotion pilot bar); the hip-pitch joint lag jointlag_frac_hip_pitch in [0.35, 0.65]; the
         stride frequency floor_stride_frequency_hz <= 5.0; and the duty guard: each foot's floor duty
         (floor_duty_L and floor_duty_R, the fraction of the window that foot is on the floor, from the probe's raw
         per-step contact, before any debounce) <= 0.80.
         Why the duty guard: the probe's 2-step debounce fills a one-step lift, so a hop whose feet lift for only
         1-2 control steps reads a stride frequency far below its real cadence (a 6-12 Hz hop read 1.4-4.8 Hz) and
         can pass the band and the cap. Its feet are down most of the time: the round-3 verifier's 332 such false
         WALKs (a true 6-12 Hz, two generators) had both feet's duty 0.842-0.929; the audit's real walker reads
         0.489-0.624 (higher foot 0.600-0.624). The guard rejects all 332 and keeps the real walker 30/30.
         Known limit (criteria.txt, "Known limits"): the faster the chatter, the lower the duty it can reach while
         still fooling the cap (about 1 - 0.01 x cadence in Hz); over 6-25 Hz, 470 distinct false WALKs of the
         round-2 rule were found, and at 0.80 the 61 that still read WALK are all at a true 16-22 Hz (duty
         0.740-0.800, every one with a mean floor swing of 0.020-0.021 s). A genuine creeping walk with a foot down
         more than 80% of the time reads OTHER.
  OTHER  anything else (scoot, shuffle, staggered or scissor hop, short-lift shuffle-hop, irregular)
Seed outcome at a checkpoint (40 episodes): WALKS if WALK >= 30 and FALL <= 4; else HOPS if HOP >= 20; else FALLS if
FALL >= 5; else STILL if STILL >= 20; else MIXED.
A seed's outcome is the one at the primary checkpoint (8,000,000). A seed the collapse backstop stopped early (its
binding run_end: a final model short of the budget) has no primary checkpoint: its outcome is the seed outcome at its
newest INTACT periodic pair (FIX2: the run_end's newest_intact_periodic_steps, which pilot3_judge.py records; a binding
without that list falls back to the newest periodic zip by file name, and says so), marked collapse-stopped, and it
counts. Any other seed without a valid primary judgement is missing. (FIX3 note: only a binding from the judge before
FIX2 lacks that list; the pinned judge always writes it. Both judges refuse a pair that is not intact, so the
fallback never reads a broken pair: when the newest zip by file name is broken it has no judgement and the seed
reads missing.)
Q4's habit reading (FIX3): a habit record marked validation_only (pilot3_habit.py seed --validation, a check of the
tool) reads NOT RUN.
One run per seed (FIX2): every cell of a seed must come from one run copy (binding.json "run", by directory name, the
RUN_ID); when they do not, none of that seed's cells is scored and the problem names the runs. A restarted seed's
retired-run cells belong under OUT/retired/<retired RUN_ID>/s<seed>/<steps>/, which this scorer lists and never
scores.
Video review (REQUIRED for WALKS; criteria.txt "Video review"): a seed's WALKS counts only when the maintainer has
watched its videos and agrees they show alternating single-leg steps. Per seed, at the checkpoint its outcome is read
at, the episodes to watch are every WALK-labelled episode and every episode with a foot's floor duty above 0.8 that is
not STILL (a standing policy's feet are down all the time; those are counted, not listed), rendered by
tools/pilot3_video.py, plus the notebook's final replay of the seed's run (a WALK episode whose mean floor swing is
0.03 s or less on either foot is starred: report only, the signature of every false WALK found). --video-review names
a JSON file of the
maintainer's verdicts, {"42": "agree" | "disagree", ...} (or {"42": {"verdict": ..., "note": ...}}): "agree" keeps a
seed's WALKS; "disagree" makes it MIXED ("WALKS by the rule, not on video"); a WALKS seed with no verdict is pending.
Verdicts for seeds whose outcome is not WALKS are recorded and change nothing.
Pilot outcome: the seed outcome shared by at least 2 of the launched seeds; MIXED when every launched seed has an
outcome and no two share one; INCOMPLETE (no outcome) when no two share one but a missing seed could still decide.
With one launched seed, that seed's outcome, marked single-seed. While a WALKS seed's review is pending, the outcome
is computed for either verdict of every pending seed: when they all give the same outcome it stands; otherwise the
pilot outcome is PENDING_VIDEO_REVIEW (never WALKS) and the possible outcomes are listed.
A cell is never scored (outcome None, reasons listed) when: binding.json is missing; it is validation-only (unless
--validation); it is off-schedule (a run resumed before ramp_timesteps); the panel is not 40 episodes of panel seeds
3042... with a 100-step gait window. pilot3_judge.py writes an off-schedule judgement (--report-off-schedule) under
OUT/off_schedule/<run>/s<seed>/<steps>/, which this scorer lists as reported and never reads as a seed's cell.
"""

from __future__ import annotations

import argparse
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
CHECKPOINTS = (2_000_000, 4_000_000, 6_000_000, 8_000_000)
PRIMARY = 8_000_000
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
        problems.append("no binding.json (not written by pilot3_judge.py)")
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


# ---------------------------------------------------------------- walking pilot 3: added readings (new functions)

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


def habit_status(seed: int, rec: "dict | None", deciding_sha: "str | None") -> dict:
    """One seed's habit reading as Q4 reads it: HABIT, NO HABIT or NOT RUN, its groups, and why.

    NOT RUN when the record is missing or malformed, or when its trace read another checkpoint than the one the
    seed's outcome is read at (the criteria's habit is read at the deciding checkpoint), or (FIX3) when the record is
    validation-only (a check of the tool, never read for Q4)."""
    if rec is None:
        return {"reading": "NOT RUN", "why": f"no habit record for s{seed}", "tail_habit": False, "neck_habit": False,
                "habits": []}
    if rec.get("validation_only"):
        return {"reading": "NOT RUN", "why": f"s{seed}'s habit record is validation-only (a check of the tool, never "
                                             "read for Q4)", "tail_habit": False, "neck_habit": False, "habits": []}
    reading = rec.get("reading")
    if reading not in HABIT_READINGS:
        return {"reading": "NOT RUN", "why": f"habit record for s{seed} reads {reading!r}", "tail_habit": False,
                "neck_habit": False, "habits": []}
    traced = (rec.get("deciding_checkpoint") or {}).get("checkpoint_sha256")
    if deciding_sha and traced and traced != deciding_sha:
        return {"reading": "NOT RUN", "why": f"s{seed}'s habit record read checkpoint {traced}, not the deciding "
                                             f"checkpoint {deciding_sha}", "tail_habit": False, "neck_habit": False,
                "habits": []}
    if deciding_sha and not traced and reading != "NOT RUN":
        return {"reading": "NOT RUN", "why": f"s{seed}'s habit record names no deciding checkpoint", "tail_habit": False,
                "neck_habit": False, "habits": []}
    habits = list(rec.get("habits") or [])
    if reading == "HABIT" and not habits:
        return {"reading": "NOT RUN", "why": f"s{seed}'s habit record reads HABIT but lists none", "tail_habit": False,
                "neck_habit": False, "habits": []}
    return {"reading": reading, "why": "; ".join(rec.get("not_run") or []) or None,
            "tail_habit": any(h.get("group") == "TAIL" for h in habits),
            "neck_habit": any(h.get("group") == "NECK" for h in habits), "habits": habits}


def habit_text(st: dict) -> str:
    if st["reading"] != "HABIT":
        return st["reading"] + (f" ({st['why']})" if st.get("why") else "")
    return "HABIT: " + "; ".join(h.get("text") or h.get("kind", "?") for h in st["habits"])


def existence_reading(rule_per_seed: dict, review: dict, closed: "list | None" = None) -> dict:
    """The existence reading (criteria "What each outcome means" (2)). A = the seeds WALKS by the rule and agreed on
    video. WALK EXISTS when A has a seed, whatever the pilot outcome; else PENDING while a WALKS-by-rule seed's review
    is pending; else NO WALK SEEN. It never changes the pilot outcome. a_final: no review pending and no missing seed
    that is not CLOSED (A can no longer change)."""
    walks = sorted(s for s, o in rule_per_seed.items() if o == "WALKS")
    verdict = {s: (review.get(s) or {}).get("verdict") for s in walks}
    agreed = [s for s in walks if verdict[s] == "agree"]
    pending = [s for s in walks if verdict[s] is None]
    open_missing = sorted(s for s, o in rule_per_seed.items() if o is None and s not in (closed or []))
    reading = "WALK EXISTS" if agreed else ("PENDING" if pending else "NO WALK SEEN")
    return {"reading": reading, "A": agreed, "walks_by_rule": walks, "pending_video_review": pending,
            "video_disagreed": [s for s in walks if verdict[s] == "disagree"], "k": len(agreed),
            "launched": len(rule_per_seed), "open_missing": open_missing, "a_final": not pending and not open_missing}


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


def q4_reading(existence: dict, pilot: dict, final: dict, habit: dict, rule_per_seed: dict,
               deciding_sha: "dict | None" = None) -> dict:
    """Q4 (criteria "What each outcome means" (3)), read once the pilot outcome is final (or the final reading
    applies) and the existence reading is not PENDING (and no WALKS-by-rule review is pending, so A is settled):
      Q4 YES             WALK EXISTS and every seed in A reads NO HABIT;
      Q4 NO, habit back  WALK EXISTS and a seed in A reads HABIT ("(neck only)" when no seed in A has a tail habit);
      Q4 UNREAD          WALK EXISTS, no seed in A reads HABIT and one reads NOT RUN; also when the final reading
                         rests on one scored seed or none and NO WALK SEEN;
      otherwise (NO WALK SEEN) Q4 is NO by the pilot outcome or the final reading.
    Every line prints k of n and the pilot outcome; a habit in a seed outside A is printed with that seed."""
    deciding_sha = deciding_sha or {}
    statuses = {s: habit_status(s, habit.get(s), deciding_sha.get(s)) for s in sorted(rule_per_seed)}
    outside = [f"s{s} {rule_per_seed[s] or 'missing'}, {habit_text(st)}" for s, st in statuses.items()
               if s not in existence["A"] and st["reading"] == "HABIT"]
    base = {"reading": None, "statuses": {str(s): st for s, st in statuses.items()}, "habit_outside_A": outside}
    pilot_final = pilot.get("outcome") in OUTCOMES and bool(pilot.get("final", True))
    final_final = bool(final.get("applies")) and bool(final.get("final"))
    if not (pilot_final or final_final):
        return dict(base, text=f"Q4 NOT READ YET: the pilot outcome ({pilot.get('outcome')}) is not final and the final "
                               "reading does not apply")
    if existence["reading"] == "PENDING" or existence["pending_video_review"]:
        return dict(base, text="Q4 NOT READ YET: video review pending for "
                               f"{['s%d' % s for s in existence['pending_video_review']]} (A is not settled)")
    k, n = existence["k"], existence["launched"]
    if pilot_final:
        x = pilot["outcome"]
        ctx = (f"{k} of {n}; pilot outcome {x}" if x == "WALKS" or k == 0
               else f"minority: {k} of {n}; pilot outcome {x}")
    else:
        x = final["reading"]
        ctx = (f"{k} of {n}; final reading {x}" if x == "WALKS" or k == 0
               else f"minority: {k} of {n}; final reading {x}") + (
            " (one seed only)" if final.get("one_seed_only") else "")
    if existence["reading"] == "WALK EXISTS":
        a = existence["A"]
        with_habit = [s for s in a if statuses[s]["reading"] == "HABIT"]
        not_run = [s for s in a if statuses[s]["reading"] == "NOT RUN"]
        if with_habit:
            neck_only = not any(statuses[s]["tail_habit"] for s in a)
            reading = "Q4 NO, habit back" + (" (neck only)" if neck_only else "")
            detail = "; ".join(f"s{s} {habit_text(statuses[s])}" for s in with_habit)
        elif not_run:
            reading = "Q4 UNREAD"
            detail = "; ".join(f"s{s} {habit_text(statuses[s])}" for s in not_run)
        else:
            reading = "Q4 YES"
            detail = "every seed in A reads NO HABIT: " + ", ".join(f"s{s}" for s in a)
    else:
        if not pilot_final and final.get("k_scored", 0) <= 1:
            reading, detail = "Q4 UNREAD", "the final reading rests on one scored seed or none, and NO WALK SEEN"
        else:
            reading = f"Q4 NO by the {'pilot outcome' if pilot_final else 'final reading'}: {x}"
            detail = "NO WALK SEEN"
    text = f"{reading} ({ctx}): {detail}"
    if existence["open_missing"]:
        text += (f"  [provisional: {['s%d' % s for s in existence['open_missing']]} not yet scored and not closed; "
                 "the existence reading and Q4 may change when they are]")
    return dict(base, reading=reading, context=ctx, detail=detail, text=text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--seeds", required=True, help="the launched seeds, comma separated (e.g. 42,52,60)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--primary", type=int, default=PRIMARY)
    ap.add_argument("--validation", action="store_true", help="accept validation-only judges (never for the verdict)")
    ap.add_argument("--video-review", default=None, help="JSON of the maintainer's per-seed video verdicts")
    ap.add_argument("--closed-seeds", default=None,
                    help="launched seeds the run log CLOSES (restart spent, run lost or refused): enables the final reading")
    ap.add_argument("--habit", action="append", default=[], help="pilot3_habit.py JSON (repeatable): Q4's habit reading")
    a = ap.parse_args(argv)
    out = Path(a.out)
    seeds = sorted({int(s) for s in a.seeds.split(",") if s.strip()})
    review = read_video_review(a.video_review)
    closed = read_closed_seeds(a.closed_seeds, seeds)
    habit = read_habit(a.habit)
    on_disk = sorted(int(p.name[1:]) for p in out.glob("s*") if p.is_dir() and p.name[1:].isdigit())
    undeclared = [s for s in on_disk if s not in seeds]
    result = {"criteria": {"still_speed_mps": STILL_SPEED,
                           "walk": {"min_alternation": WALK_MIN_ALTERNATION, "min_lead_swaps_per_stride": WALK_MIN_LEAD_SWAPS,
                                    "max_flight": WALK_MAX_FLIGHT, "hip_lag_key": WALK_HIP_LAG_KEY,
                                    "hip_lag_band": list(WALK_HIP_LAG_BAND), "stride_key": WALK_STRIDE_KEY,
                                    "max_stride_frequency_hz": WALK_MAX_STRIDE_HZ, "duty_keys": list(WALK_DUTY_KEYS),
                                    "max_foot_duty": WALK_MAX_DUTY},
                           "hop_min_simultaneous": HOP_MIN_SIMULTANEOUS,
                           "hop_min_touchdowns_per_s": HOP_MIN_TOUCHDOWNS_PER_S, "episodes": EPISODES,
                           "seed": {"walks": [SEED_WALKS_MIN, SEED_WALKS_MAX_FALLS], "hops": SEED_HOPS_MIN,
                                    "falls": SEED_FALLS_MIN, "still": SEED_STILL_MIN},
                           "video_review": {"required_for": "WALKS", "watch": "every WALK episode, and every episode "
                                            f"that is not STILL with a foot's floor duty > {VIDEO_DUTY}",
                                            "notebook_replay": NOTEBOOK_REPLAY}},
              "primary": a.primary, "validation": a.validation, "launched_seeds": seeds,
              "undeclared_seed_dirs_ignored": undeclared, "seeds": {}, "per_seed": {}, "per_seed_primary": {},
              "per_seed_reviewed": {}, "video_review_file": a.video_review,
              "video_review_verdicts": {str(k): v for k, v in sorted(review.items())}, "video_review_required": {}}
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
            cells[steps] = score_panel(json.loads(p.read_text()), binding, a.validation)
        mixed = one_run_problem(cells)
        if mixed:
            for c in cells.values():
                if c is not None:
                    c["problems"].append(mixed)
                    c["outcome"] = None
        result["seeds"][str(s)] = {str(k): v for k, v in cells.items()}
        sr = final_fallback_basis(seed_result(cells, a.primary), cells)
        rev_outcome, rev_state = reviewed_outcome(sr["outcome"], (review.get(s) or {}).get("verdict"))
        sr["video_review"] = rev_state
        sr["outcome_after_video_review"] = rev_outcome
        result["per_seed"][str(s)] = sr
        result["per_seed_primary"][str(s)] = sr["outcome"]
        result["per_seed_reviewed"][str(s)] = rev_outcome
        cell = cells.get(sr["steps"]) if sr.get("steps") is not None else None
        result["video_review_required"][str(s)] = None if cell is None else {
            "steps": sr["steps"], "episodes": cell["video_watch"],
            "still_with_foot_duty_over_0_8": cell["still_with_foot_duty_over_0_8"],
            "render": f"tools/pilot3_video.py OUT/s{s}/{sr['steps']} --from-score <this JSON>",
            "notebook_replay": NOTEBOOK_REPLAY.replace("<RUN_ID>", f"pilot3_walk_s{s}")
                               + f" (or pilot3_walk_s{s}_b if the seed was restarted)",
            "required": sr["outcome"] == "WALKS", "review_state": rev_state}
    reports = {}
    for p in sorted((out / "off_schedule").glob("*/s*/*/probe.json")):
        b = p.with_name("binding.json")
        cell = score_panel(json.loads(p.read_text()), json.loads(b.read_text()) if b.is_file() else None, a.validation)
        reports[str(p.parent.relative_to(out))] = {"counts": cell["counts"], "outcome": cell["outcome"],
                                                   "problems": cell["problems"]}
    result["off_schedule_reports"] = reports
    result["retired_cells_never_scored"] = sorted(str(p.parent.relative_to(out))
                                                  for p in (out / "retired").glob("*/s*/*/probe.json"))
    rule_per_seed = {s: result["per_seed_primary"][str(s)] for s in seeds}
    result["pilot_by_rule"] = pilot_outcome(rule_per_seed) if seeds else None
    result["pilot"] = pilot_outcome_reviewed(rule_per_seed, review) if seeds else {
        "outcome": None, "basis": "no launched seeds given", "missing": [], "final": True, "pending_video_review": []}
    deciding_sha = {}
    for s in seeds:
        steps = result["per_seed"][str(s)].get("steps")
        cell = result["seeds"][str(s)].get(str(steps)) if steps is not None else None
        deciding_sha[s] = (cell or {}).get("checkpoint_sha256")
    result["closed_seeds"] = closed
    result["final_reading"] = final_reading(rule_per_seed, review, result["pilot"], closed)
    result["existence"] = existence_reading(rule_per_seed, review, closed)
    result["habit_files"] = list(a.habit)
    result["q4"] = q4_reading(result["existence"], result["pilot"], result["final_reading"], habit, rule_per_seed,
                              deciding_sha)

    print(f"Walking pilot score ({out}); launched seeds {seeds}; primary checkpoint {a.primary:,}")
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
        print(f"  s{s} seed outcome: {sr['outcome'] or 'none'} ({sr['basis']}); video review: {sr['video_review']}"
              + (f" -> {sr['outcome_after_video_review']}" if sr["video_review"] == "disagree" else ""))
    if undeclared:
        print(f"  ignored (not a launched seed): {['s%d' % s for s in undeclared]}")
    for where, rep in result["off_schedule_reports"].items():
        print(f"  reported, never scored: {where} {rep['counts']}")
    for where in result["retired_cells_never_scored"]:
        print(f"  retired run's cell, never scored: {where}")
    print("VIDEO REVIEW REQUIRED (criteria_pilot3.txt \"Video review\"): for every seed that is WALKS by the rule, "
          "even when its verdict cannot change the pilot outcome; a seed's WALKS counts only when the maintainer has "
          "watched these and agrees they show alternating single-leg steps; record the verdicts in a JSON file "
          "({\"42\": \"agree\" | \"disagree\", ...}) and pass --video-review.")
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
    pil = result["pilot"]
    print(f"PILOT OUTCOME: {pil['outcome']} ({pil['basis']})" + (f"; missing or invalid: {pil['missing']}" if pil["missing"] else "")
          + ("" if pil.get("final", True) else "  NOT FINAL: video review pending"))
    fr = result["final_reading"]
    if fr["applies"]:
        print(fr["text"] + f" [{fr['basis']}]" + ("" if fr["final"] else "  NOT FINAL: video review pending"))
    ex = result["existence"]
    print(f"EXISTENCE: {ex['reading']} (A = {['s%d' % s for s in ex['A']]}, k = {ex['k']} of {ex['launched']}; WALKS by "
          f"the rule {['s%d' % s for s in ex['walks_by_rule']]}; review pending {['s%d' % s for s in ex['pending_video_review']]}; "
          f"video-disagreed {['s%d' % s for s in ex['video_disagreed']]})"
          + (f"  [not settled: {['s%d' % s for s in ex['open_missing']]} still missing and not closed]"
             if ex["open_missing"] else ""))
    print(result["q4"]["text"])
    for line in result["q4"]["habit_outside_A"]:
        print(f"  habit outside A (evidence for a later audit, changes no branch): {line}")
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
