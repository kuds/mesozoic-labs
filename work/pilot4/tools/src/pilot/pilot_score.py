"""Score the walking pilot's gait-probe panels against the fixed criteria (foot/meanwhile/pilot/criteria.txt).

Plain Python, no repository import. Reads what pilot_judge.py wrote:
    OUT/s<seed>/<steps>/probe.json and binding.json      (one 40-episode panel per seed and checkpoint)
and prints, per seed and checkpoint, the per-episode label counts and the seed outcome, then the episodes the video
review must watch ("VIDEO REVIEW REQUIRED"), then the pilot outcome.

    python3 pilot_score.py OUT --seeds 42,52,60 [--json OUT/score.json] [--primary 4194304]
                               [--video-review REVIEW.json]

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
A seed's outcome is the one at the primary checkpoint (4,194,304). A seed the collapse backstop stopped early (its
binding run_end: a final model short of the budget) has no primary checkpoint: its outcome is the seed outcome at its
newest periodic checkpoint, marked collapse-stopped, and it counts. Any other seed without a valid primary judgement
is missing.
Video review (REQUIRED for WALKS; criteria.txt "Video review"): a seed's WALKS counts only when the maintainer has
watched its videos and agrees they show alternating single-leg steps. Per seed, at the checkpoint its outcome is read
at, the episodes to watch are every WALK-labelled episode and every episode with a foot's floor duty above 0.8 that is
not STILL (a standing policy's feet are down all the time; those are counted, not listed), rendered by
tools/pilot_video.py, plus the notebook's final replay of the seed's run (a WALK episode whose mean floor swing is
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
3042... with a 100-step gait window. pilot_judge.py writes an off-schedule judgement (--report-off-schedule) under
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
CHECKPOINTS = (2_097_152, 3_145_728, 4_194_304)
PRIMARY = 4_194_304
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
               "left_leads_fraction", "froude", WALK_HIP_LAG_KEY, *WALK_DUTY_KEYS, *SWING_KEYS)


def score_panel(doc: dict, binding: dict | None = None, validation: bool = False) -> dict:
    rows = doc.get("episodes", [])
    meta = doc.get("meta", {})
    problems = []
    if binding is None:
        problems.append("no binding.json (not written by pilot_judge.py)")
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
        newest = e.get("newest_periodic_steps")
        cell = cells.get(newest) if newest is not None else None
        if cell is not None and cell["outcome"] is not None:
            return {"outcome": cell["outcome"], "collapse_stopped": True, "steps": newest,
                    "basis": f"collapse-stopped at {e.get('final_num_timesteps'):,} steps ({e.get('final_source')}); "
                             f"outcome read from its newest periodic checkpoint {newest:,}"}
        return {"outcome": None, "collapse_stopped": True, "steps": None,
                "basis": f"collapse-stopped at {e.get('final_num_timesteps')} steps, but its newest periodic checkpoint "
                         f"{newest} has no valid judgement: judge it"}
    if prim is not None:
        return {"outcome": None, "steps": None,
                "basis": f"checkpoint {primary:,} judged but not scored: {prim['problems']}"}
    judged = sorted(s for s, c in cells.items() if c is not None)
    return {"outcome": None, "steps": None,
            "basis": "missing" + (f" (judged so far: {', '.join(f'{s:,}' for s in judged)})" if judged
                                  else " (nothing judged)")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--seeds", required=True, help="the launched seeds, comma separated (e.g. 42,52,60)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--primary", type=int, default=PRIMARY)
    ap.add_argument("--validation", action="store_true", help="accept validation-only judges (never for the verdict)")
    ap.add_argument("--video-review", default=None, help="JSON of the maintainer's per-seed video verdicts")
    a = ap.parse_args(argv)
    out = Path(a.out)
    seeds = sorted({int(s) for s in a.seeds.split(",") if s.strip()})
    review = read_video_review(a.video_review)
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
        result["seeds"][str(s)] = {str(k): v for k, v in cells.items()}
        sr = seed_result(cells, a.primary)
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
            "render": f"tools/pilot_video.py OUT/s{s}/{sr['steps']} --from-score <this JSON>",
            "notebook_replay": NOTEBOOK_REPLAY.replace("<RUN_ID>", f"pilot_walk_s{s}")
                               + f" (or pilot_walk_s{s}_b if the seed was restarted)"}
    reports = {}
    for p in sorted((out / "off_schedule").glob("*/s*/*/probe.json")):
        b = p.with_name("binding.json")
        cell = score_panel(json.loads(p.read_text()), json.loads(b.read_text()) if b.is_file() else None, a.validation)
        reports[str(p.parent.relative_to(out))] = {"counts": cell["counts"], "outcome": cell["outcome"],
                                                   "problems": cell["problems"]}
    result["off_schedule_reports"] = reports
    rule_per_seed = {s: result["per_seed_primary"][str(s)] for s in seeds}
    result["pilot_by_rule"] = pilot_outcome(rule_per_seed) if seeds else None
    result["pilot"] = pilot_outcome_reviewed(rule_per_seed, review) if seeds else {
        "outcome": None, "basis": "no launched seeds given", "missing": [], "final": True, "pending_video_review": []}

    print(f"Walking pilot score ({out}); launched seeds {seeds}; primary checkpoint {a.primary:,}")
    for s, cells in result["seeds"].items():
        for steps, cell in cells.items():
            if cell is None:
                print(f"  s{s} {int(steps):>9,}: missing")
                continue
            m = cell["medians"]
            print(f"  s{s} {int(steps):>9,}: {cell['outcome'] or 'NOT SCORED'}  {cell['counts']}  "
                  f"v {m['speed_displacement_mps']:.3f} m/s, alt {m['floor_alternation_index']:.2f}, "
                  f"sim {m['floor_simultaneous_fraction']:.2f}, swaps {m['floor_lead_swaps_per_stride']:.2f}, "
                  f"flight {m['floor_flight_fraction']:.2f}, hip lag {m[WALK_HIP_LAG_KEY]:.2f}, "
                  f"stride {m['floor_stride_frequency_hz']:.2f} Hz, duty L/R {m['floor_duty_L']:.2f}/"
                  f"{m['floor_duty_R']:.2f}; OTHER passing the contact bar, by rejecting guards: "
                  f"{ {k: v for k, v in cell['other_passing_contact_bar'].items() if v} }"
                  + (f"  PROBLEMS: {cell['problems']}" if cell["problems"] else ""))
        sr = result["per_seed"][s]
        print(f"  s{s} seed outcome: {sr['outcome'] or 'none'} ({sr['basis']}); video review: {sr['video_review']}"
              + (f" -> {sr['outcome_after_video_review']}" if sr["video_review"] == "disagree" else ""))
    if undeclared:
        print(f"  ignored (not a launched seed): {['s%d' % s for s in undeclared]}")
    for where, rep in result["off_schedule_reports"].items():
        print(f"  reported, never scored: {where} {rep['counts']}")
    print("VIDEO REVIEW REQUIRED (criteria.txt \"Video review\"): a seed's WALKS counts only when the maintainer has "
          "watched these and agrees they show alternating single-leg steps; record the verdicts in a JSON file "
          "({\"42\": \"agree\" | \"disagree\", ...}) and pass --video-review.")
    for s in seeds:
        req = result["video_review_required"][str(s)]
        if req is None:
            print(f"  s{s}: no scored checkpoint yet, nothing to list")
            continue
        walk = [str(e["episode"]) + ("*" if e["short_swing"] else "") for e in req["episodes"] if e["why"] == "WALK"]
        duty = [f"{e['episode']} ({e['label']}, {e['foot_duty_max']})" for e in req["episodes"] if e["why"] != "WALK"]
        print(f"  s{s} at {req['steps']:,}: panel episodes (reset seed 3042 + episode; render with {req['render']}): "
              f"WALK [{', '.join(walk) or 'none'}]; foot duty > {VIDEO_DUTY} [{'; '.join(duty) or 'none'}]"
              + (f"; {req['still_with_foot_duty_over_0_8']} STILL episodes over the duty line not listed"
                 if req["still_with_foot_duty_over_0_8"] else "")
              + f". Notebook replay: {req['notebook_replay']}"
              + ("  (* = a foot's mean floor swing <= 0.03 s, the short-lift signature: watch it at quarter speed)"
                 if any(e["short_swing"] for e in req["episodes"] if e["why"] == "WALK") else ""))
    pil = result["pilot"]
    print(f"PILOT OUTCOME: {pil['outcome']} ({pil['basis']})" + (f"; missing or invalid: {pil['missing']}" if pil["missing"] else "")
          + ("" if pil.get("final", True) else "  NOT FINAL: video review pending"))
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
