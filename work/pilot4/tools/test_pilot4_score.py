"""Cases for pilot4_score.py (plain Python; run: python3 -I -B test_pilot4_score.py [PILOT_TREE], best through iso.py:
    cd LANE && ISO_WT=TREE MUJOCO_GL=disable python3 -I TOOLS/iso.py TOOLS/test_pilot4_score.py TREE).

Walking pilot 4's copy of pilot 3's test_pilot3_score.py (sha256 e7af7200...), adapted only where it names checkpoints,
the budget (2,000,000; checkpoints every 250,000), the RUN_ID pilot4_walk_s<seed> or the readings pilot 4 changed (the
pilot outcome now reads the four seed outcomes; pilot 3's reading is the "gait reading"; Q4 and pilot 3's existence
reading are gone), plus the cases named "p4: ..." for v3's TOOL PLAN: the five functions byte-identical to the pinned
pilot3_score.py, --straight and --straight-review, every seed-outcome path, the pilot outcome over them, the final
reading, the report-only count (Q23), the existence reading, PENDING_REVIEW, (c)'s NOT READ with its counts, --habit
report only, and pilot 3's 8M references. Pilot 3's label, guard and fixture cases are the rest, unchanged.

Optional local evidence (a case that needs it is skipped and listed when it is absent):
  PILOT3_SCORE          pilot 3's pinned pilot3_score.py (default: ../../pilot3/tools/pilot3_score.py from this file)
  PILOT3_JUDGING        pilot 3's judging folder (OUT/, final_score/) (default: ../../pilot3/judging)
  PILOT4_STRAIGHT_VAL   the straightness tool's validation records (p3val_s<seed>.json), re-run by the tolerance stage
                        with the pinned tool (default: ../prelaunch/tol/val_final from this file's folder)
  PILOT4_JUDGE_E2E      the judge lane's end-to-end OUT (a real pilot-4 binding) (default: out/judge/e2e/OUT)
With numpy importable and a pilot tree given (argument or PILOT_TREE), the synthetic fixture rows are recomputed from
the tree's frozen probe and must equal the fixture (pilot 3's check). Nothing here trains or imports the repository
but the probe file itself.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot4_score as ps  # noqa: E402

CASES = json.loads((HERE / "fixtures" / "gait_cases.json").read_text())
SYN = {k: v["row"] for k, v in CASES["synthetic"].items()}
REAL = CASES["real_walker"]["rows"]
P3_SCORE = Path(os.environ.get("PILOT3_SCORE") or HERE.parent.parent / "pilot3" / "tools" / "pilot3_score.py")
P3_JUDGING = Path(os.environ.get("PILOT3_JUDGING") or HERE.parent.parent / "pilot3" / "judging")
STRAIGHT_VAL = Path(os.environ.get("PILOT4_STRAIGHT_VAL") or HERE.parent / "prelaunch" / "tol" / "val_final")
JUDGE_E2E = Path(os.environ.get("PILOT4_JUDGE_E2E") or HERE / "out" / "judge" / "e2e" / "OUT")
P3_SCORE_SHA256 = "286ce04a2c5f4e34b9f3e146431a14cbd40767050f83749f5968b9e270a97018"

WALK = {"length": 1000, "speed_displacement_mps": 0.3, "floor_touchdowns_per_s_all_feet": 2.0,
        "floor_alternation_index": 0.95, "floor_simultaneous_fraction": 0.02, "floor_lead_swaps_per_stride": 1.9,
        "floor_flight_fraction": 0.0, "jointlag_frac_hip_pitch": 0.5, "floor_stride_frequency_hz": 1.0,
        "floor_duty_L": 0.6, "floor_duty_R": 0.6, "floor_swing_mean_s_L": 0.4, "floor_swing_mean_s_R": 0.4}
# The September seed-42 walker's floor-truth numbers (gait plan section 4.3 A; stride 7.67 Hz, audit CSV): a hop.
HOP = {"length": 1000, "speed_displacement_mps": 1.1, "floor_touchdowns_per_s_all_feet": 15.0,
       "floor_alternation_index": 0.01, "floor_simultaneous_fraction": 0.97, "floor_lead_swaps_per_stride": 0.0,
       "floor_flight_fraction": 0.35, "jointlag_frac_hip_pitch": 0.0, "floor_stride_frequency_hz": 7.67,
       "floor_duty_L": 0.59, "floor_duty_R": 0.6}
STATUE = {"length": 1000, "speed_displacement_mps": 0.002, "floor_touchdowns_per_s_all_feet": 0.0,
          "floor_alternation_index": None, "floor_simultaneous_fraction": None, "floor_lead_swaps_per_stride": None,
          "floor_flight_fraction": 0.0, "jointlag_frac_hip_pitch": None, "floor_stride_frequency_hz": None,
          "floor_duty_L": 1.0, "floor_duty_R": 1.0}
FALL = dict(WALK, length=412)
SCOOT = {"length": 1000, "speed_displacement_mps": 0.4, "floor_touchdowns_per_s_all_feet": 3.0,
         "floor_alternation_index": 0.4, "floor_simultaneous_fraction": 0.2, "floor_lead_swaps_per_stride": 0.05,
         "floor_flight_fraction": 0.1, "jointlag_frac_hip_pitch": 0.1, "floor_stride_frequency_hz": 3.0,
         "floor_duty_L": 0.7, "floor_duty_R": 0.7}

cases = []
skipped = []


def check(name, got, want):
    cases.append((name, got == want, got, want))


def skip(name, why):
    skipped.append(f"{name}: {why}")


def raises(fn, *needles) -> str:
    """'refused' when fn() raises SystemExit naming every needle; else what happened."""
    try:
        fn()
    except SystemExit as exc:
        return "refused" if all(n in str(exc) for n in needles) else f"other: {exc}"
    return "accepted"


# Episode labels: hand-made
check("walk", ps.label_episode(WALK), "WALK")
check("hop", ps.label_episode(HOP), "HOP")
check("statue", ps.label_episode(STATUE), "STILL")
check("fall beats walk", ps.label_episode(FALL), "FALL")
check("scoot", ps.label_episode(SCOOT), "OTHER")
check("hop in place counts as hop", ps.label_episode(dict(HOP, speed_displacement_mps=0.01)), "HOP")
check("a rare synchronous re-seat is not a hop", ps.label_episode(dict(STATUE, floor_simultaneous_fraction=1.0,
                                                                        floor_touchdowns_per_s_all_feet=0.2)), "STILL")
check("slow walk is still", ps.label_episode(dict(WALK, speed_displacement_mps=0.09)), "STILL")
check("walk edge 0.10 m/s", ps.label_episode(dict(WALK, speed_displacement_mps=0.10)), "WALK")
check("alternation 0.59 fails", ps.label_episode(dict(WALK, floor_alternation_index=0.59)), "OTHER")
check("alternation 0.60 passes", ps.label_episode(dict(WALK, floor_alternation_index=0.60)), "WALK")
check("lead swaps 0.99 fails", ps.label_episode(dict(WALK, floor_lead_swaps_per_stride=0.99)), "OTHER")
check("flight 0.21 fails", ps.label_episode(dict(WALK, floor_flight_fraction=0.21)), "OTHER")
check("flight 0.20 passes", ps.label_episode(dict(WALK, floor_flight_fraction=0.20)), "WALK")
check("undefined lead swaps fails", ps.label_episode(dict(WALK, floor_lead_swaps_per_stride=None)), "OTHER")
check("NaN speed is still", ps.label_episode(dict(WALK, speed_displacement_mps=math.nan)), "STILL")
check("sim 0.49 is not a hop", ps.label_episode(dict(HOP, floor_simultaneous_fraction=0.49)), "OTHER")
# The hip-pitch lag band [0.35, 0.65]
check("hip lag 0.35 passes", ps.label_episode(dict(WALK, jointlag_frac_hip_pitch=0.35)), "WALK")
check("hip lag 0.3499 fails", ps.label_episode(dict(WALK, jointlag_frac_hip_pitch=0.3499)), "OTHER")
check("hip lag 0.65 passes", ps.label_episode(dict(WALK, jointlag_frac_hip_pitch=0.65)), "WALK")
check("hip lag 0.6501 fails", ps.label_episode(dict(WALK, jointlag_frac_hip_pitch=0.6501)), "OTHER")
check("hip lag ~1 (in phase, wrapped) fails", ps.label_episode(dict(WALK, jointlag_frac_hip_pitch=0.998)), "OTHER")
check("undefined hip lag fails", ps.label_episode(dict(WALK, jointlag_frac_hip_pitch=None)), "OTHER")
check("missing hip lag key fails", ps.label_episode({k: v for k, v in WALK.items() if k != "jointlag_frac_hip_pitch"}),
      "OTHER")
# The stride cap floor_stride_frequency_hz <= 5.0
check("stride 5.0 Hz passes", ps.label_episode(dict(WALK, floor_stride_frequency_hz=5.0)), "WALK")
check("stride 5.01 Hz fails", ps.label_episode(dict(WALK, floor_stride_frequency_hz=5.01)), "OTHER")
check("stride 0.5 Hz passes (no lower bound)", ps.label_episode(dict(WALK, floor_stride_frequency_hz=0.5)), "WALK")
check("undefined stride frequency fails", ps.label_episode(dict(WALK, floor_stride_frequency_hz=None)), "OTHER")
check("NaN stride frequency fails", ps.label_episode(dict(WALK, floor_stride_frequency_hz=math.nan)), "OTHER")
check("missing stride frequency key fails",
      ps.label_episode({k: v for k, v in WALK.items() if k != "floor_stride_frequency_hz"}), "OTHER")
# The duty guard: each foot's floor duty (floor_duty_L, floor_duty_R) <= 0.80
check("duty 0.80 on both feet passes", ps.label_episode(dict(WALK, floor_duty_L=0.80, floor_duty_R=0.80)), "WALK")
check("left duty 0.8001 fails", ps.label_episode(dict(WALK, floor_duty_L=0.8001)), "OTHER")
check("right duty 0.8001 fails", ps.label_episode(dict(WALK, floor_duty_R=0.8001)), "OTHER")
check("duty 0.0 passes (no lower bound)", ps.label_episode(dict(WALK, floor_duty_L=0.0, floor_duty_R=0.0)), "WALK")
check("undefined duty fails", ps.label_episode(dict(WALK, floor_duty_L=None)), "OTHER")
check("NaN duty fails", ps.label_episode(dict(WALK, floor_duty_R=math.nan)), "OTHER")
check("missing duty key fails", ps.label_episode({k: v for k, v in WALK.items() if k != "floor_duty_R"}), "OTHER")
check("a high duty fails the duty guard alone", ps.guards_failed(dict(WALK, floor_duty_R=0.9)), ["duty"])
check("guards failed, all three", ps.guards_failed(dict(WALK, jointlag_frac_hip_pitch=0.1, floor_stride_frequency_hz=8,
                                                         floor_duty_L=0.9)), ["hip_lag", "stride", "duty"])
check("the duty guard reads the higher foot", ps.foot_duty_max(dict(WALK, floor_duty_L=0.55, floor_duty_R=0.81)), 0.81)

# Episode labels: rows the probe's own analysis computed (fixtures/gait_cases.json)
st = SYN["staggered_hop_30ms"]
check("staggered hop 30 ms @ 8.3 Hz: probe reads alternation 1.0", st["floor_alternation_index"], 1.0)
check("staggered hop 30 ms: probe reads simultaneous 0.0", st["floor_simultaneous_fraction"], 0.0)
check("staggered hop 30 ms: the old contact bar passes it", ps.walk_contact_tests(st), True)
check("staggered hop 30 ms: probe's floor_gait_class is 'walk' (so it is no guard)", st["floor_gait_class"], "walk")
check("staggered hop 30 ms: hip lag 0.25", round(st["jointlag_frac_hip_pitch"], 6), 0.25)
check("staggered hop 30 ms is not WALK", ps.label_episode(st), "OTHER")
check("staggered hop 40 ms (hip lag 1/3) is not WALK", ps.label_episode(SYN["staggered_hop_40ms"]), "OTHER")
check("staggered hop 50 ms @ 8.3 Hz (hip lag 5/12) passes the hip-lag band", ps.hip_lag_in_band(SYN["staggered_hop_50ms"]),
      True)
check("staggered hop 50 ms @ 8.3 Hz is not WALK: the stride cap rejects it", ps.label_episode(SYN["staggered_hop_50ms"]),
      "OTHER")
check("synchronous hop is HOP", ps.label_episode(SYN["synchronous_hop"]), "HOP")
check("probe-computed alternating walk is WALK", ps.label_episode(SYN["alternating_walk"]), "WALK")

# The round-2 verifier's grid and the final rule's limits ("ct_" rows, continuous-time generator, probe-computed).
# Expected labels under the final rule; the comment says which test decides.
CT_EXPECT = {
    # staggered two-footed hops, hips following the stagger: 20 ms merges into one event (HOP); 30-50 ms pass the
    # contact bar, and 50 ms (7.7-8.3 Hz) and 40-50 ms (9.2-10 Hz) also pass the hip-lag band: the stride cap decides
    **{f"ct_stagger_hop_{f}Hz_20ms": "HOP" for f in ("7.7", "8.3", "9.2", "10.0")},
    **{f"ct_stagger_hop_{f}Hz_{o}ms": "OTHER" for f in ("7.7", "8.3", "9.2", "10.0") for o in (30, 40, 50)},
    "ct_stagger_hop_9.2Hz_40ms_hips_in_phase": "OTHER",   # lead swaps 0, lag 0
    "ct_stagger_hop_8.3Hz_30ms_left_lead": "OTHER",       # lead swaps 0
    "ct_scissor_hop_8.3Hz_30ms": "OTHER",                 # lag 0.5: the stride cap decides
    "ct_walk_1.0Hz_sym": "WALK",
    "ct_walk_1.6Hz_sym_noisy": "WALK",
    "ct_walk_0.7Hz_slow": "WALK",
    "ct_walk_1.2Hz_pelvis2f": "WALK",
    "ct_limp_walk_1.2Hz_40pct": "WALK",                   # lag 0.40
    "ct_limp_walk_1.2Hz_30pct": "OTHER",                  # lag 0.30: a false negative of the band (known limit)
    "ct_alt_run_3Hz_flight": "OTHER",                     # flight 0.4
    "ct_walk_4.0Hz_sym": "WALK",                          # near the real walker's 4.05 Hz
    "ct_alt_shuffle_5.5Hz": "OTHER",                      # a genuine alternation at 5.5 Hz: false negative of the cap
    "ct_scissor_hop_4.5Hz_30ms": "WALK",                  # known limit: a scissor hop at <= 5 Hz reads WALK
    "ct_stagger_hop_4.5Hz_40ms": "OTHER",                 # lag 0.18
    "ct_stagger_hop_4.5Hz_80ms": "WALK",                  # known limit: offset >= 35% of the cycle at <= 5 Hz
    # final round: short-lift hops (feet lift for 1-2 control steps; the debounce fills a lift, so the stride reads
    # <= 5 Hz) pass the contact bar, the band and the cap; the duty guard rejects them, down to a duty of 0.80
    "ct_shortlift_hop_6Hz_d0.90_60ms": "OTHER",           # the round-3 verifier's quoted case: duty 0.90
    "ct_shortlift_hop_11Hz_d0.85_-40ms": "OTHER",         # the lowest duty of the verifier's 332: 0.842
    "ct_chatter_15Hz_d0.80_40ms_scissor": "OTHER",        # duty 0.802
    "ct_chatter_22Hz_d0.74_-40ms": "WALK",                # known limit: a 22 Hz chatter at duty 0.74
    "ct_walk_1.0Hz_d0.75": "WALK",                        # a creeping walk at duty 0.75
    "ct_walk_1.0Hz_d0.85": "OTHER",                       # a creeping walk at duty 0.85: false negative of the guard
}
EV_EXPECT = {
    "ev_shortlift_hop_6Hz_d0.90_-50ms_hip0.4": "OTHER",
    "ev_shortlift_hop_8Hz_d0.90_-30ms_hip0.6": "OTHER",
    "ev_shortlift_hop_10Hz_d0.85_-30ms_follow": "OTHER",  # the verifier's quoted 10 Hz case
    "ev_shortlift_hop_12Hz_d0.85_-30ms_hip0.8": "OTHER",
    "ev_chatter_16Hz_d0.80_-30ms_hip0.6": "WALK",         # known limit: duty exactly 0.800 at 16 Hz
    "ev_walk_4Hz_d0.62": "WALK",                          # the real walker's regime
    "ev_walk_1Hz_d0.70": "WALK",
    "ev_walk_1Hz_d0.85": "OTHER",                         # a creeping walk at duty 0.85: false negative of the guard
}
check("every ct_ fixture row has an expected label", sorted(k for k in SYN if k.startswith("ct_")), sorted(CT_EXPECT))
check("every ev_ fixture row has an expected label", sorted(k for k in SYN if k.startswith("ev_")), sorted(EV_EXPECT))
for name, want in {**CT_EXPECT, **EV_EXPECT}.items():
    check(f"{name} is {want}", ps.label_episode(SYN[name]), want)
# Which hop rows pass both the contact bar and the hip-lag band: above 5 Hz the stride cap alone rejects them
BAND_PASSING_HOPS = ["ct_scissor_hop_4.5Hz_30ms", "ct_scissor_hop_8.3Hz_30ms", "ct_stagger_hop_10.0Hz_40ms",
                     "ct_stagger_hop_10.0Hz_50ms", "ct_stagger_hop_4.5Hz_80ms", "ct_stagger_hop_7.7Hz_50ms",
                     "ct_stagger_hop_8.3Hz_50ms", "ct_stagger_hop_9.2Hz_40ms", "ct_stagger_hop_9.2Hz_50ms",
                     "staggered_hop_50ms"]
STAGGER_FAMILY = {k: r for k, r in SYN.items() if "hop" in k and "shortlift" not in k}
check("staggered/scissor hop rows passing the contact bar and the hip-lag band",
      sorted(k for k, r in STAGGER_FAMILY.items() if ps.walk_contact_tests(r) and ps.hip_lag_in_band(r)),
      BAND_PASSING_HOPS)
for name in BAND_PASSING_HOPS:
    r = SYN[name]
    above = r["floor_stride_frequency_hz"] > 5.0
    check(f"{name} ({r['floor_stride_frequency_hz']:.2f} Hz, lag {r['jointlag_frac_hip_pitch']:.2f}): "
          + ("only the stride cap rejects it" if above else "nothing rejects it (known limit)"),
          (ps.stride_frequency_ok(r), ps.label_episode(r)), (False, "OTHER") if above else (True, "WALK"))
check("every staggered/scissor hop row reads a stride above 5 Hz except the 4.5 Hz limit cases",
      sorted(k for k, r in STAGGER_FAMILY.items() if r["floor_stride_frequency_hz"] <= 5.0),
      ["ct_scissor_hop_4.5Hz_30ms", "ct_stagger_hop_4.5Hz_40ms", "ct_stagger_hop_4.5Hz_80ms"])
check("the staggered/scissor hops' feet are down 67-71% of the time: the duty guard never decides them",
      (all(ps.duty_ok(r) for r in STAGGER_FAMILY.values()),
       round(min(ps.foot_duty_max(r) for r in STAGGER_FAMILY.values()), 2),
       round(max(ps.foot_duty_max(r) for r in STAGGER_FAMILY.values()), 2)), (True, 0.67, 0.71))
# The short-lift family (the round-3 verifier's finding and this round's extension): every one passed the round-2
# rule (contact bar, hip-lag band, stride cap) because the debounce fills its one-step lifts, so the measured stride
# reads <= 5 Hz at a true 6-22 Hz; each has a mean floor swing of 0.020-0.021 s on both feet. The duty guard rejects
# every one whose higher foot is down more than 80% of the time; the 16 and 22 Hz chatters below it read WALK.
SHORT_LIFT = sorted(k for k in SYN if "shortlift" in k or "chatter" in k)
check("short-lift rows", SHORT_LIFT,
      ["ct_chatter_15Hz_d0.80_40ms_scissor", "ct_chatter_22Hz_d0.74_-40ms", "ct_shortlift_hop_11Hz_d0.85_-40ms",
       "ct_shortlift_hop_6Hz_d0.90_60ms", "ev_chatter_16Hz_d0.80_-30ms_hip0.6",
       "ev_shortlift_hop_10Hz_d0.85_-30ms_follow", "ev_shortlift_hop_12Hz_d0.85_-30ms_hip0.8",
       "ev_shortlift_hop_6Hz_d0.90_-50ms_hip0.4", "ev_shortlift_hop_8Hz_d0.90_-30ms_hip0.6"])
for name in SHORT_LIFT:
    r = SYN[name]
    d = ps.foot_duty_max(r)
    check(f"{name}: the round-2 rule passed it (contact bar, band, cap; measured stride "
          f"{r['floor_stride_frequency_hz']:.2f} Hz)",
          (ps.walk_contact_tests(r), ps.hip_lag_in_band(r), ps.stride_frequency_ok(r)), (True, True, True))
    check(f"{name}: mean floor swing 0.020-0.021 s on both feet",
          all(0.0199 <= r[k] <= 0.0212 for k in ("floor_swing_mean_s_L", "floor_swing_mean_s_R")), True)
    check(f"{name} (higher foot duty {d:.3f}): " + ("rejected by the duty guard alone" if d > 0.80
                                                     else "below the guard, reads WALK (known limit)"),
          (ps.guards_failed(r), ps.label_episode(r)), (["duty"], "OTHER") if d > 0.80 else ([], "WALK"))
# The real walker: compsognathus 20260921_203149 final, 30 episodes
real_labels = Counter(ps.label_episode(r) for r in REAL)
check("real walker: 30 episodes", len(REAL), 30)
check("real walker: 30/30 WALK", dict(real_labels), {"WALK": 30})
check("real walker: every hip lag inside the band",
      all(0.35 <= r["jointlag_frac_hip_pitch"] <= 0.65 for r in REAL), True)
check("real walker: probe class walk 30/30", Counter(r["floor_gait_class"] for r in REAL)["walk"], 30)
check("real walker: every foot's duty under the 0.80 guard (higher foot 0.600-0.624)",
      (all(ps.duty_ok(r) for r in REAL), round(min(ps.foot_duty_max(r) for r in REAL), 3),
       round(max(ps.foot_duty_max(r) for r in REAL), 3)), (True, 0.6, 0.624))
check("real walker: mean floor swing 0.092-0.126 s (no short-swing flag)",
      (round(min(ps.foot_swing_min(r) for r in REAL), 3), any(ps.foot_swing_min(r) <= ps.SHORT_SWING_S for r in REAL)),
      (0.092, False))
check("real walker: every stride frequency under the 5 Hz cap (3.97-4.10 Hz)",
      (all(r["floor_stride_frequency_hz"] <= 5.0 for r in REAL),
       round(min(r["floor_stride_frequency_hz"] for r in REAL), 2), round(max(r["floor_stride_frequency_hz"] for r in REAL), 2)),
      (True, 3.97, 4.1))
check("real walker panel (30 walk + 10 still) is WALKS", ps.seed_outcome(real_labels + Counter(STILL=10)), "WALKS")

# Seed outcomes (pilot 3's rule, the gait reading)
check("30 walks 4 falls", ps.seed_outcome(Counter(WALK=30, FALL=4, OTHER=6)), "WALKS")
check("30 walks 5 falls", ps.seed_outcome(Counter(WALK=30, FALL=5, OTHER=5)), "FALLS")
check("29 walks", ps.seed_outcome(Counter(WALK=29, OTHER=11)), "MIXED")
check("20 hops", ps.seed_outcome(Counter(HOP=20, STILL=20)), "HOPS")
check("19 hops 21 still", ps.seed_outcome(Counter(HOP=19, STILL=21)), "STILL")
check("hops with falls", ps.seed_outcome(Counter(HOP=25, FALL=10, OTHER=5)), "HOPS")
check("19 still", ps.seed_outcome(Counter(STILL=19, OTHER=21)), "MIXED")
check("40 staggered hops is MIXED (OTHER)", ps.seed_outcome(Counter(ps.label_episode(st) for _ in range(40))), "MIXED")

# Pilot outcomes (every launched seed is a key)
check("2 of 3 walk", ps.pilot_outcome({42: "WALKS", 52: "WALKS", 60: "HOPS"})["outcome"], "WALKS")
check("3 different", ps.pilot_outcome({42: "WALKS", 52: "STILL", 60: "HOPS"})["outcome"], "MIXED")
check("missing seed, 2 agree", ps.pilot_outcome({42: "HOPS", 52: None, 60: "HOPS"})["outcome"], "HOPS")
check("missing seed, 2 differ: incomplete", ps.pilot_outcome({42: "HOPS", 52: None, 60: "WALKS"})["outcome"],
      "INCOMPLETE")
check("two missing: incomplete", ps.pilot_outcome({42: "HOPS", 52: None, 60: None})["outcome"], "INCOMPLETE")
check("single seed", ps.pilot_outcome({42: "WALKS"})["basis"], "single seed (s42)")
check("all missing", ps.pilot_outcome({42: None, 52: None, 60: None})["outcome"], None)


def write_cell(root: Path, seed: int, steps: int, rows: list, binding: dict | None, meta: dict) -> None:
    d = root / f"s{seed}" / str(steps)
    d.mkdir(parents=True)
    (d / "probe.json").write_text(json.dumps({"meta": meta, "episodes": rows}))
    if binding is not None:
        (d / "binding.json").write_text(json.dumps(binding))


def run_main(argv: list) -> tuple:
    """(exit status, printed text, the --json document or None): main() with its output captured."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ps.main(argv)
    doc = None
    if "--json" in argv:
        doc = json.loads(Path(argv[argv.index("--json") + 1]).read_text())
    return rc, buf.getvalue(), doc


# End to end on synthetic directories
META = {"seed_scheme": "panel", "seed_base": 3042, "settle_steps_gait_window": 100}
OK = {"validation_only": False, "expected_budget": 2000000, "off_schedule": False}
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    write_cell(root, 42, ps.PRIMARY, [HOP] * 40, OK, META)
    write_cell(root, 52, ps.PRIMARY, [HOP] * 30 + [STATUE] * 10, OK, META)
    write_cell(root, 60, ps.PRIMARY, [WALK] * 40, OK, META)
    write_cell(root, 42, ps.CHECKPOINTS[0], [WALK] * 40, OK, dict(META, seed_scheme="vecenv"))
    write_cell(root, 60, ps.CHECKPOINTS[1], [WALK] * 40, dict(OK, validation_only=True, expected_budget=50000), META)
    write_cell(root, 60, ps.CHECKPOINTS[0], [WALK] * 40, None, META)
    write_cell(root, 99, ps.PRIMARY, [WALK] * 40, OK, META)  # a seed nobody launched
    _, _, res = run_main([tmp, "--seeds", "42,52,60", "--json", str(root / "score.json")])
    check("e2e gait reading (pilot 3's pilot outcome)", res["gait_reading"]["reviewed"]["outcome"], "HOPS")
    check("p4: e2e pilot outcome: two HOPS seeds share NOT WALKS, whatever s60's pending reading",
          (res["pilot"]["outcome"], res["pilot"]["final"]), ("NOT WALKS", True))
    check("e2e s60 walks", res["per_seed_primary"]["60"], "WALKS")
    check("e2e wrong seed scheme is not scored", res["seeds"]["42"][str(ps.CHECKPOINTS[0])]["outcome"], None)
    check("e2e missing checkpoint", res["seeds"]["52"][str(ps.CHECKPOINTS[1])], None)
    check("e2e validation-only judge is not scored", res["seeds"]["60"][str(ps.CHECKPOINTS[1])]["outcome"], None)
    check("e2e judge without binding is not scored", res["seeds"]["60"][str(ps.CHECKPOINTS[0])]["outcome"], None)
    check("e2e undeclared seed dir ignored", res["undeclared_seed_dirs_ignored"], [99])
    check("e2e undeclared seed not scored", "99" in res["per_seed"], False)
    _, _, resv = run_main([tmp, "--seeds", "42,52,60", "--validation", "--json", str(root / "score_val.json")])
    check("e2e --validation accepts it", resv["seeds"]["60"][str(ps.CHECKPOINTS[1])]["outcome"], "WALKS")

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # report only: OTHER episodes that pass the contact bar, by the test that rejects them
    rows = ([SYN["ct_scissor_hop_8.3Hz_30ms"]] * 20 + [SYN["ct_stagger_hop_8.3Hz_30ms"]] * 10
            + [SYN["ct_limp_walk_1.2Hz_30pct"]] * 5 + [SYN["ev_shortlift_hop_10Hz_d0.85_-30ms_follow"]] * 3
            + [SYN["ct_walk_1.0Hz_sym"]] * 2)
    write_cell(root, 42, ps.PRIMARY, rows, OK, META)
    _, _, doc = run_main([tmp, "--seeds", "42", "--json", str(root / "r.json")])
    cell = doc["seeds"]["42"][str(ps.PRIMARY)]
    check("report-only: OTHER passing the contact bar, by the guards that reject it", cell["other_passing_contact_bar"],
          {"hip_lag": 5, "stride": 20, "duty": 3, "hip_lag+stride": 10, "hip_lag+duty": 0, "stride+duty": 0,
           "hip_lag+stride+duty": 0})
    check("report-only counts do not change the outcome (2 WALK of 40: MIXED)", (cell["counts"]["WALK"], cell["outcome"]),
          (2, "MIXED"))
    check("watch list: the 2 WALK and the 3 short-lift hops over the duty line, in panel order",
          [(e["episode"], e["label"], e["why"]) for e in cell["video_watch"]],
          [(35, "OTHER", "foot duty > 0.8"), (36, "OTHER", "foot duty > 0.8"), (37, "OTHER", "foot duty > 0.8"),
           (38, "WALK", "WALK"), (39, "WALK", "WALK")])

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # only s42 judged, three launched: missing seeds count as missing, not as a single-seed pilot
    write_cell(root, 42, ps.PRIMARY, [HOP] * 40, OK, META)
    _, _, r = run_main([tmp, "--seeds", "42,52,60", "--json", str(root / "a.json")])
    check("launched 3, judged 1: INCOMPLETE (pilot outcome and gait reading)",
          (r["pilot"]["outcome"], r["gait_reading"]["reviewed"]["outcome"]), ("INCOMPLETE", "INCOMPLETE"))
    check("launched 3, judged 1: missing listed", (r["pilot"]["missing"], r["gait_reading"]["reviewed"]["missing"]),
          ([52, 60], [52, 60]))
    _, _, b = run_main([tmp, "--seeds", "42", "--json", str(root / "b.json")])
    check("launched 1: single seed", (b["pilot"]["basis"], b["gait_reading"]["reviewed"]["basis"]),
          ("single seed (s42)", "single seed (s42)"))

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # s42: an off-schedule judgement at the primary checkpoint (judged with --report-off-schedule) is never scored
    write_cell(root, 42, ps.PRIMARY, [WALK] * 40,
               dict(OK, off_schedule=True, off_schedule_reason="OFF-SCHEDULE: resumed at step 500,000"), META)
    # s52: collapse-stopped at 1,900,544; judged at 500,000, 1,000,000 and its newest periodic 1,750,000
    end = {"periodic_steps": [250000 * i for i in range(1, 8)], "newest_periodic_steps": 1750000,
           "final_num_timesteps": 1900544, "final_source": "stage2_final.zip num_timesteps", "budget": 2000000,
           "stopped_early": True}
    for steps, rows in ((500000, [WALK] * 40), (1000000, [FALL] * 10 + [WALK] * 30),
                        (1750000, [FALL] * 25 + [STATUE] * 15)):
        write_cell(root, 52, steps, rows, dict(OK, run_end=end), META)
    # s60: stopped early but its newest periodic checkpoint was not judged
    end60 = dict(end, newest_periodic_steps=1750000)
    write_cell(root, 60, 1000000, [HOP] * 40, dict(OK, run_end=end60), META)
    _, _, r = run_main([tmp, "--seeds", "42,52,60", "--json", str(root / "c.json")])
    check("off-schedule primary is not scored", r["seeds"]["42"][str(ps.PRIMARY)]["outcome"], None)
    check("off-schedule seed is missing", (r["per_seed_primary"]["42"], r["seed_outcomes"]["42"]["outcome"]),
          (None, None))
    check("collapse-stopped seed scored from its newest periodic checkpoint", r["per_seed_primary"]["52"], "FALLS")
    check("collapse-stopped seed marked", r["per_seed"]["52"].get("collapse_stopped"), True)
    check("p4: a collapse-stopped seed's outcome is read at that checkpoint (NOT WALKS (FALLS) at 1,750,000)",
          (r["seed_outcomes"]["52"]["outcome"], r["seed_outcomes"]["52"]["label"], r["seed_outcomes"]["52"]["steps"]),
          ("NOT WALKS", "FALLS", 1750000))
    check("collapse-stopped seed, newest checkpoint unjudged: missing", r["per_seed_primary"]["60"], None)
    check("collapse-stopped seed, newest checkpoint unjudged: says so",
          "has no valid judgement" in r["per_seed"]["60"]["basis"], True)
    check("pilot with one scored seed of three: INCOMPLETE", r["pilot"]["outcome"], "INCOMPLETE")
    check("FIX2: a binding without the intact list falls back to the newest zip by file name, and says so",
          "newest periodic checkpoint by file name" in r["per_seed"]["52"]["basis"], True)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # FIX2: collapse-stopped, newest periodic pair broken (1,750,000: not judged, the judge refuses it); the newest
    # INTACT pair 1,500,000 is judged and decides (the review's bin/collapse_newest_broken.py case, pilot 4's numbers)
    end = {"periodic_steps": [250000 * i for i in range(1, 8)], "newest_periodic_steps": 1750000,
           "intact_periodic_steps": [250000 * i for i in range(1, 7)], "newest_intact_periodic_steps": 1500000,
           "broken_periodic": {"1750000": "missing sidecar stage2_vecnormalize_1750000_steps.pkl"},
           "final_num_timesteps": 1900544, "final_source": "stage2_final.zip num_timesteps", "budget": 2000000,
           "stopped_early": True}
    for steps, rows in ((500000, [STATUE] * 40), (1000000, [STATUE] * 40), (1500000, [FALL] * 6 + [STATUE] * 34)):
        write_cell(root, 42, steps, rows, dict(OK, run_end=end), META)
    # s52: collapse-stopped with no intact periodic pair at all
    end52 = dict(end, intact_periodic_steps=[], newest_intact_periodic_steps=None)
    write_cell(root, 52, 500000, [STATUE] * 40, dict(OK, run_end=end52), META)
    _, _, r = run_main([tmp, "--seeds", "42,52", "--json", str(root / "i.json")])
    check("FIX2: collapse-stopped, newest pair broken: read at the newest intact pair",
          (r["per_seed_primary"]["42"], r["per_seed"]["42"]["steps"]), ("FALLS", 1500000))
    check("FIX2: the basis names the newest intact periodic pair",
          "newest intact periodic pair 1,500,000" in r["per_seed"]["42"]["basis"], True)
    check("FIX2: collapse-stopped with no intact pair: missing, says so",
          (r["per_seed_primary"]["52"], "with no newest intact periodic pair" in r["per_seed"]["52"]["basis"]),
          (None, True))

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # FIX2: one run per seed. s42 was restarted under _b, but the retired run's 500,000 and 1,000,000 cells were left
    run_a = dict(OK, run="/copies/pilot4_walk_s42")
    run_b = dict(OK, run="/copies/pilot4_walk_s42_b")
    write_cell(root, 42, 500000, [STATUE] * 40, run_a, META)
    write_cell(root, 42, 1000000, [STATUE] * 40, run_a, META)
    write_cell(root, 42, ps.PRIMARY, [HOP] * 40, run_b, META)
    write_cell(root, 52, ps.PRIMARY, [HOP] * 40, dict(OK, run="/copies/pilot4_walk_s52"), META)
    _, _, r = run_main([tmp, "--seeds", "42,52", "--json", str(root / "m.json")])
    check("FIX2: a seed whose cells come from two runs is not scored",
          (r["per_seed_primary"]["42"], r["seeds"]["42"][str(ps.PRIMARY)]["outcome"]), (None, None))
    check("FIX2: the problem names both runs",
          "['pilot4_walk_s42', 'pilot4_walk_s42_b']" in r["seeds"]["42"]["500000"]["problems"][-1], True)
    check("FIX2: a seed from one run is scored as before", r["per_seed_primary"]["52"], "HOPS")
    # the retired cells moved, never deleted, to OUT/retired/<run>/s42/<steps>/
    for steps in ("500000", "1000000"):
        dst = root / "retired" / "pilot4_walk_s42" / "s42" / steps
        dst.parent.mkdir(parents=True, exist_ok=True)
        (root / "s42" / steps).rename(dst)
    _, _, r = run_main([tmp, "--seeds", "42,52", "--json", str(root / "m2.json")])
    check("FIX2: after the move the _b run is scored", r["per_seed_primary"]["42"], "HOPS")
    check("FIX2: the retired cells are listed, never scored", r["retired_cells_never_scored"],
          ["retired/pilot4_walk_s42/s42/1000000", "retired/pilot4_walk_s42/s42/500000"])
    check("FIX2: one_run_problem ignores cells without a recorded run",
          ps.one_run_problem({1: {"run": None}, 2: {"run": "a"}, 3: None}), None)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # s42 judged off schedule: the judge's --report-off-schedule wrote it under OUT/off_schedule/<run>/; the seed was
    # restarted under "_b" and that run judged into OUT/s42/ as usual (v3 expects none: D11)
    off = root / "off_schedule" / "RUN_s42" / "s42" / str(ps.PRIMARY)
    off.mkdir(parents=True)
    (off / "probe.json").write_text(json.dumps({"meta": META, "episodes": [WALK] * 40}))
    (off / "binding.json").write_text(json.dumps(dict(OK, off_schedule=True, off_schedule_reason="OFF-SCHEDULE: x")))
    write_cell(root, 42, ps.PRIMARY, [HOP] * 40, OK, META)
    _, _, r = run_main([tmp, "--seeds", "42", "--json", str(root / "o.json")])
    check("off-schedule report beside the restart: the seed's outcome is the restart's", r["per_seed_primary"]["42"], "HOPS")
    check("off-schedule report is listed, never scored",
          (list(r["off_schedule_reports"]), r["off_schedule_reports"][f"off_schedule/RUN_s42/s42/{ps.PRIMARY}"]["outcome"]),
          ([f"off_schedule/RUN_s42/s42/{ps.PRIMARY}"], None))
    check("off_schedule/ is not read as a seed directory", r["undeclared_seed_dirs_ignored"], [])

# Video review: the watch list, the per-seed verdicts and the gait reading while a review is pending
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    chatter = SYN["ct_chatter_22Hz_d0.74_-40ms"]           # WALK by the rule, short swing (known limit)
    shortlift = SYN["ev_shortlift_hop_10Hz_d0.85_-30ms_follow"]  # OTHER, a foot over the duty line
    rows = [WALK] * 31 + [chatter] + [shortlift] * 2 + [STATUE] * 5 + [SCOOT]
    write_cell(root, 42, ps.PRIMARY, rows, OK, META)
    _, _, r = run_main([tmp, "--seeds", "42", "--json", str(root / "v0.json")])
    cell = r["seeds"]["42"][str(ps.PRIMARY)]
    req = r["video_review_required"]["42"]
    check("watch list: 32 WALK + 2 over the duty line; STILL over the line counted, not listed; SCOOT not listed",
          (len(req["episodes"]), sorted({e["why"] for e in req["episodes"]}), req["still_with_foot_duty_over_0_8"],
           any(e["episode"] == 39 for e in req["episodes"])),
          (34, ["WALK", "foot duty > 0.8"], 5, False))
    check("watch list carries the panel reset seed", req["episodes"][0]["reset_seed"], 3042)
    check("short-swing flag on the 22 Hz chatter (WALK, episode 31) and the two short-lift hops (OTHER, 32-33) only",
          [e["episode"] for e in req["episodes"] if e["short_swing"]], [31, 32, 33])
    check("walk_with_short_swing count", cell["walk_with_short_swing"], 1)
    check("notebook replay named for the seed's run", "pilot4_walk_s42/03_locomotion/replays/" in req["notebook_replay"]
          and req["notebook_replay"].split(" ")[0].endswith("tyrannosaurus_rex_anatomical_ppo_stage2_final.mp4"), True)
    check("p4: the renderer named is pilot4_video.py",
          req["render"].startswith("tools/pilot4_video.py OUT/s42/2000000"), True)
    check("single seed WALKS by the rule, no review: the gait reading is PENDING_VIDEO_REVIEW, never WALKS",
          (r["per_seed_primary"]["42"], r["per_seed"]["42"]["video_review"], r["gait_reading"]["reviewed"]["outcome"],
           r["gait_reading"]["reviewed"]["final"], r["gait_reading"]["by_rule"]["outcome"]),
          ("WALKS", "pending", "PENDING_VIDEO_REVIEW", False, "WALKS"))
    check("p4: ... and the pilot outcome is PENDING_REVIEW (video review, record and look awaited)",
          (r["pilot"]["outcome"], r["pilot"]["final"], r["seed_outcomes"]["42"]["awaited"]),
          ("PENDING_REVIEW", False, ["video review", "straightness record", "straightness look"]))
    (root / "agree.json").write_text(json.dumps({"42": "agree"}))
    (root / "disagree.json").write_text(json.dumps({"s42": {"verdict": "Disagree", "note": "both feet land together"}}))
    _, _, r1 = run_main([tmp, "--seeds", "42", "--video-review", str(root / "agree.json"),
                         "--json", str(root / "v1.json")])
    check("review agrees: the gait reading is WALKS, final",
          (r1["per_seed_reviewed"]["42"], r1["gait_reading"]["reviewed"]["outcome"],
           r1["gait_reading"]["reviewed"]["final"]), ("WALKS", "WALKS", True))
    check("p4: review agrees, no straightness record: the seed is pending, the pilot outcome PENDING_REVIEW",
          (r1["seed_outcomes"]["42"]["outcome"], r1["seed_outcomes"]["42"]["awaited"], r1["pilot"]["outcome"]),
          ("pending", ["straightness record", "straightness look"], "PENDING_REVIEW"))
    _, _, r2 = run_main([tmp, "--seeds", "42", "--video-review", str(root / "disagree.json"),
                         "--json", str(root / "v2.json")])
    check("review disagrees: the seed is MIXED by pilot 3's rule, the rule's WALKS kept on record",
          (r2["per_seed_primary"]["42"], r2["per_seed_reviewed"]["42"], r2["gait_reading"]["reviewed"]["outcome"],
           r2["video_review_verdicts"]["42"]["note"]), ("WALKS", "MIXED", "MIXED", "both feet land together"))
    check("p4: review disagrees: NOT WALKS (MIXED: WALKS by the rule, not on video), final, no look required",
          (r2["seed_outcomes"]["42"]["outcome"], r2["seed_outcomes"]["42"]["label"], r2["pilot"]["outcome"],
           r2["pilot"]["final"], r2["straightness_look_required"]["42"]),
          ("NOT WALKS", "MIXED: WALKS by the rule, not on video", "NOT WALKS", True, None))
    (root / "bad.json").write_text(json.dumps({"42": "yes"}))
    check("a verdict other than agree/disagree is refused",
          raises(lambda: ps.main([tmp, "--seeds", "42", "--video-review", str(root / "bad.json")]), "agree"), "refused")
rev = ps.pilot_outcome_reviewed
check("WALKS, WALKS, HOPS with both reviews pending: PENDING (WALKS or MIXED)",
      (rev({42: "WALKS", 52: "WALKS", 60: "HOPS"}, {})["outcome"],
       rev({42: "WALKS", 52: "WALKS", 60: "HOPS"}, {})["possible_outcomes"]), ("PENDING_VIDEO_REVIEW", ["MIXED", "WALKS"]))
check("WALKS, WALKS, HOPS, both agree: WALKS",
      rev({42: "WALKS", 52: "WALKS", 60: "HOPS"}, {42: {"verdict": "agree"}, 52: {"verdict": "agree"}})["outcome"], "WALKS")
check("WALKS, WALKS, HOPS, one disagrees: MIXED",
      rev({42: "WALKS", 52: "WALKS", 60: "HOPS"}, {42: {"verdict": "agree"}, 52: {"verdict": "disagree"}})["outcome"],
      "MIXED")
check("WALKS x3, two agree, one pending: WALKS stands whatever the third review finds",
      (lambda o: (o["outcome"], o["final"], o["pending_video_review"]))(
          rev({42: "WALKS", 52: "WALKS", 60: "WALKS"}, {42: {"verdict": "agree"}, 52: {"verdict": "agree"}})),
      ("WALKS", True, [60]))
check("HOPS, HOPS, WALKS pending: HOPS (final)",
      (lambda o: (o["outcome"], o["final"]))(rev({42: "HOPS", 52: "HOPS", 60: "WALKS"}, {})), ("HOPS", True))
check("a verdict on a non-WALKS seed changes nothing",
      rev({42: "HOPS", 52: "HOPS", 60: "STILL"}, {42: {"verdict": "disagree"}})["outcome"], "HOPS")
check("WALKS, WALKS, missing, reviews pending: PENDING (WALKS, INCOMPLETE or MIXED)",
      rev({42: "WALKS", 52: "WALKS", 60: None}, {})["possible_outcomes"], ["INCOMPLETE", "MIXED", "WALKS"])


# ======================================================================== p3: walking pilot 3's readings, kept
# report only: STILL episodes' touchdowns, alternation, yaw; speed_forward_vel_mps
STEP_IN_PLACE = dict(STATUE, floor_touchdowns_per_s_all_feet=1.2, floor_alternation_index=0.9, yaw_change_deg=-35.0,
                     speed_forward_vel_mps=0.01, floor_duty_L=0.7, floor_duty_R=0.7)
_rows = [STATUE] * 30 + [STEP_IN_PLACE] * 8 + [dict(WALK, speed_forward_vel_mps=-0.25)] * 2
_labels = [ps.label_episode(r) for r in _rows]
_st = ps.still_report(_rows, _labels)
check("p3: still report counts the STILL episodes stepping at >= 0.5 touchdowns/s",
      (_st["n"], _st["stepping_at_0_5_touchdowns_per_s"], _st["episodes"][30]["yaw_change_deg"]), (38, 8, -35.0))
_w, _ = ps.video_watch(_rows, _labels)
check("p3: the watch list carries speed_forward_vel_mps (a backward WALK is listed with it)",
      [(e["episode"], e["speed_forward_vel_mps"]) for e in _w], [(38, -0.25), (39, -0.25)])

# pilot 3's final reading, kept for the gait reading
F = ps.final_reading
_fr = F({42: "HOPS", 52: "STILL", 60: None}, {}, ps.pilot_outcome({42: "HOPS", 52: "STILL", 60: None}), [60])
check("p3: gait final reading, two scored seeds that differ read MIXED", (_fr["applies"], _fr["reading"], _fr["text"]),
      (True, "MIXED", "FINAL READING: MIXED from 2 scored seeds of 3 (closed: s60)"))
_fr = F({42: "STILL", 52: None, 60: None}, {}, ps.pilot_outcome({42: "STILL", 52: None, 60: None}), [52, 60])
check("p3: gait final reading, one scored seed reads its outcome, one seed only",
      (_fr["reading"], _fr["one_seed_only"], _fr["text"]),
      ("STILL", True, "FINAL READING: STILL (one seed only) from 1 scored seeds of 3 (closed: s52, s60)"))
_fr = F({42: None, 52: None, 60: None}, {}, ps.pilot_outcome({42: None, 52: None, 60: None}), [42, 52, 60])
check("p3: gait final reading, no scored seed reads NO OUTCOME", (_fr["reading"], _fr["k_scored"]), ("NO OUTCOME", 0))
check("p3: no gait final reading while a missing seed is not closed",
      F({42: "HOPS", 52: "STILL", 60: None}, {}, ps.pilot_outcome({42: "HOPS", 52: "STILL", 60: None}), [])["applies"],
      False)
check("p3: no gait final reading when the pilot outcome stands",
      F({42: "HOPS", 52: "HOPS", 60: None}, {}, ps.pilot_outcome({42: "HOPS", 52: "HOPS", 60: None}), [60])["applies"],
      False)
_fr = F({42: "WALKS", 52: "STILL", 60: None}, {}, ps.pilot_outcome_reviewed({42: "WALKS", 52: "STILL", 60: None}, {}),
        [60])
check("p3: gait final reading with a pending review of a WALKS seed: MIXED either way, final",
      (_fr["reading"], _fr["final"]), ("MIXED", True))
check("p3: --closed-seeds naming a seed not launched is refused",
      raises(lambda: ps.read_closed_seeds("52,99", [42, 52, 60]), "99"), "refused")


# ======================================================================== p4: walking pilot 4's changes and readings
check("p4: CHECKPOINTS and PRIMARY", (ps.CHECKPOINTS, ps.PRIMARY), ((500000, 1000000, 1500000, 2000000), 2000000))

# The five functions v3's TOOL PLAN keeps byte-identical: their source against the pinned pilot3_score.py, and
# against the sha256s of that source recorded here (the same as pilot 1's for the four pilot 3 already pinned)
_SRC = (HERE / "pilot4_score.py").read_text()
_P3_BODIES = {
    "label_episode": "693b2261794b160cce7fd185c2d132cdb501e9d0b09480d207d283e3e7b98967",
    "seed_outcome": "d4734f60cd1a0456d40286f1e571a4cb5414cf3140dcf4c367a21209300b0f43",
    "pilot_outcome": "b9f83dbcd22e162a9e252567dbac92e9ac8ae22433bcfdd997b94830e798d2e8",
    "reviewed_outcome": "65178231056c7c2a9dda4a98298f1d9611c2cc0d16d514d6ea719f0fc3eb35e4",
    "pilot_outcome_reviewed": "ba9ceb0d6741ffbbeb3fcc5207ebe5834c8426f3aa12b026a069252eef5a893d"}


def _segments(src: str) -> dict:
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef):
            out[f"def {node.name}"] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            for t in node.targets:
                out[f"= {t.id}"] = ast.get_source_segment(src, node)
    return out


_P4 = _segments(_SRC)
for _name, _sha in _P3_BODIES.items():
    check(f"p4: {_name} source has the recorded sha256 (pilot 3's)",
          hashlib.sha256(_P4[f"def {_name}"].encode()).hexdigest(), _sha)
# Kept from pilot 3 unchanged besides the five: the helpers the five and the kept readings use
KEPT = ["= EPISODES", "= GUARDS", "= HABIT_READINGS", "= HOP_MIN_SIMULTANEOUS", "= HOP_MIN_TOUCHDOWNS_PER_S",
        "= HORIZON", "= LAUNCHED", "= NOTEBOOK_REPLAY", "= OUTCOMES", "= PANEL_SEED_BASE", "= REPORT_KEYS",
        "= SEED_FALLS_MIN", "= SEED_HOPS_MIN", "= SEED_STILL_MIN", "= SEED_WALKS_MAX_FALLS", "= SEED_WALKS_MIN",
        "= SHORT_SWING_S", "= STILL_SPEED", "= STILL_STEPPING_TD", "= SWING_KEYS", "= VIDEO_DUTY", "= WALK_DUTY_KEYS",
        "= WALK_HIP_LAG_BAND", "= WALK_HIP_LAG_KEY", "= WALK_MAX_DUTY", "= WALK_MAX_FLIGHT", "= WALK_MAX_STRIDE_HZ",
        "= WALK_MIN_ALTERNATION", "= WALK_MIN_LEAD_SWAPS", "= WALK_STRIDE_KEY", "def _median", "def _num",
        "def _rounded", "def duty_ok", "def final_fallback_basis", "def final_reading", "def foot_duty_max",
        "def foot_swing_min", "def guards_failed", "def habit_text", "def hip_lag_in_band", "def label_episode",
        "def one_run_problem", "def pilot_outcome", "def pilot_outcome_reviewed", "def read_closed_seeds",
        "def read_habit", "def read_video_review", "def reviewed_outcome", "def seed_outcome", "def seed_result",
        "def still_report", "def stride_frequency_ok", "def video_watch", "def walk_contact_tests"]
if P3_SCORE.is_file():
    _p3src = P3_SCORE.read_text()
    check("p4: the pinned pilot3_score.py read is the pinned file (sha256 286ce04a...)",
          hashlib.sha256(P3_SCORE.read_bytes()).hexdigest(), P3_SCORE_SHA256)
    _P3 = _segments(_p3src)
    for _name in _P3_BODIES:
        check(f"p4: {_name} is byte-identical to the pinned pilot3_score.py's", _P4[f"def {_name}"],
              _P3[f"def {_name}"])
    check("p4: every other kept definition is byte-identical to pilot 3's",
          sorted(k for k in _P3 if k in _P4 and _P3[k] == _P4[k]), sorted(KEPT))
    check("p4: changed against pilot 3: CHECKPOINTS, PRIMARY, main, score_panel (the judge's name only)",
          sorted(k for k in _P3 if k in _P4 and _P3[k] != _P4[k]), ["= CHECKPOINTS", "= PRIMARY", "def main",
                                                                    "def score_panel"])
    check("p4: score_panel differs from pilot 3's only in the judge's name",
          _P4["def score_panel"].replace("pilot4_judge.py", "pilot3_judge.py"), _P3["def score_panel"])
    check("p4: removed: pilot 3's Q4 branch, its existence reading and habit_status",
          sorted(k for k in _P3 if k not in _P4), ["def existence_reading", "def habit_status", "def q4_reading"])
else:
    skip("p4: byte-identity against the pinned pilot3_score.py", f"no file {P3_SCORE}")
check("p4: the scorer imports only the standard library (no repository import)",
      sorted({(n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0] for n in ast.walk(ast.parse(_SRC))
              if isinstance(n, (ast.Import, ast.ImportFrom)) for a in (n.names if isinstance(n, ast.Import) else [n])}),
      ["__future__", "argparse", "collections", "hashlib", "itertools", "json", "math", "pathlib", "sys"])

# ---- the seed outcome, every path (v3 "Seed outcome", first that applies)
P_ALL = {"a": "PASS", "b": "PASS", "c": "PASS"}
SO = ps.straight_seed_outcome


def _o(*args):
    r = SO(*args)
    return (r["outcome"], r["label"])


for rule in ("HOPS", "FALLS", "STILL", "MIXED"):
    check(f"p4: pilot 3's {rule} is NOT WALKS ({rule}), whatever the review, record and look",
          {_o(rule, v, p, lk) for v in (None, "agree", "disagree") for p in (None, P_ALL)
           for lk in (None, "straight", "not straight")}, {("NOT WALKS", rule)})
check("p4: a missing seed has no outcome", (_o(None, None, None, None), SO(None, None, None, None)["possible"]),
      ((None, "missing"), [None]))
check("p4: WALKS, video disagrees: NOT WALKS (MIXED: WALKS by the rule, not on video), whatever the parts and look",
      {_o("WALKS", "disagree", p, lk) for p in (None, P_ALL, dict(P_ALL, a="FAIL")) for lk in (None, "straight")},
      {("NOT WALKS", "MIXED: WALKS by the rule, not on video")})
check("p4: WALKS, agreed, (a) FAIL: CURVED WALKS (a) FAIL", _o("WALKS", "agree", dict(P_ALL, a="FAIL"), None),
      ("CURVED WALKS", "(a) FAIL"))
check("p4: WALKS, agreed, (b) FAIL: CURVED WALKS (b) FAIL", _o("WALKS", "agree", dict(P_ALL, b="FAIL"), None),
      ("CURVED WALKS", "(b) FAIL"))
check("p4: WALKS, agreed, (a) and (b) FAIL: CURVED WALKS naming both",
      _o("WALKS", "agree", dict(P_ALL, a="FAIL", b="FAIL"), None), ("CURVED WALKS", "(a) FAIL, (b) FAIL"))
check("p4: (a) NOT READ is not a pass: CURVED WALKS (literal reading; impossible on a WALKS seed's panel)",
      _o("WALKS", "agree", dict(P_ALL, a="NOT READ"), None), ("CURVED WALKS", "(a) NOT READ"))
check("p4: the look never makes a seed pass (a) or (b): (b) FAIL with the look 'straight' is CURVED WALKS",
      _o("WALKS", "agree", dict(P_ALL, b="FAIL"), "straight"), ("CURVED WALKS", "(b) FAIL"))
check("p4: (a) FAIL with (c) PASS is CURVED WALKS (the order of the rule)",
      _o("WALKS", "agree", dict(P_ALL, a="FAIL", c="PASS"), "straight"), ("CURVED WALKS", "(a) FAIL"))
check("p4: the veto: (a), (b), (c) PASS and the look 'not straight' is CURVED WALKS (not on video)",
      _o("WALKS", "agree", P_ALL, "not straight"), ("CURVED WALKS", "straight by the rule, not on video"))
check("p4: the veto also on a (c) failure", _o("WALKS", "agree", dict(P_ALL, c="FAIL"), "not straight"),
      ("CURVED WALKS", "straight by the rule, not on video"))
check("p4: (a), (b) PASS, the look agrees, (c) FAIL: STRAIGHT NO RECOVERY (c) FAIL",
      _o("WALKS", "agree", dict(P_ALL, c="FAIL"), "straight"), ("STRAIGHT NO RECOVERY", "(c) FAIL"))
check("p4: (a), (b) PASS, the look agrees, (c) NOT READ: STRAIGHT NO RECOVERY (c) NOT READ",
      _o("WALKS", "agree", dict(P_ALL, c="NOT READ"), "straight"), ("STRAIGHT NO RECOVERY", "(c) NOT READ"))
check("p4: (a), (b), (c) PASS and the look agrees: STRAIGHT WALKS", _o("WALKS", "agree", P_ALL, "straight"),
      ("STRAIGHT WALKS", "(a), (b) and (c) PASS, the look agrees"))
check("p4: a decided seed lists one possible outcome and awaits nothing",
      (SO("WALKS", "agree", P_ALL, "straight")["possible"], SO("WALKS", "agree", P_ALL, "straight")["awaited"]),
      (["STRAIGHT WALKS"], []))
check("p4: a curved seed needs no look (decided without it)",
      SO("WALKS", "agree", dict(P_ALL, a="FAIL"), None)["awaited"], [])
_cases_pending = [
    ("video review pending, all PASS, look straight", ("WALKS", None, P_ALL, "straight"),
     ["NOT WALKS", "STRAIGHT WALKS"], ["video review"]),
    ("look pending, all PASS", ("WALKS", "agree", P_ALL, None), ["CURVED WALKS", "STRAIGHT WALKS"],
     ["straightness look"]),
    ("look pending, (c) NOT READ", ("WALKS", "agree", dict(P_ALL, c="NOT READ"), None),
     ["CURVED WALKS", "STRAIGHT NO RECOVERY"], ["straightness look"]),
    ("look pending, (c) FAIL", ("WALKS", "agree", dict(P_ALL, c="FAIL"), None),
     ["CURVED WALKS", "STRAIGHT NO RECOVERY"], ["straightness look"]),
    ("no straightness record, no look", ("WALKS", "agree", None, None),
     ["CURVED WALKS", "STRAIGHT NO RECOVERY", "STRAIGHT WALKS"], ["straightness record", "straightness look"]),
    ("no straightness record, look straight", ("WALKS", "agree", None, "straight"),
     ["CURVED WALKS", "STRAIGHT NO RECOVERY", "STRAIGHT WALKS"], ["straightness record"]),
    ("nothing yet", ("WALKS", None, None, None),
     ["CURVED WALKS", "NOT WALKS", "STRAIGHT NO RECOVERY", "STRAIGHT WALKS"],
     ["video review", "straightness record", "straightness look"]),
    ("video review pending, (a) FAIL", ("WALKS", None, dict(P_ALL, a="FAIL"), None), ["CURVED WALKS", "NOT WALKS"],
     ["video review"]),
]
for _name, _args, _poss, _aw in _cases_pending:
    _r = SO(*_args)
    check(f"p4: pending ({_name}): possible outcomes and what it awaits",
          (_r["outcome"], _r["possible"], _r["awaited"]), ("pending", _poss, _aw))
_r = SO("WALKS", "agree", None, "not straight")
check("p4: no record but the look 'not straight': CURVED WALKS whatever the record reads (the veto or a failed part)",
      (_r["outcome"], _r["possible"], _r["awaited"]), ("CURVED WALKS", ["CURVED WALKS"], ["straightness record"]))
# exhaustive: every known combination reads the first rule that applies, and a pending seed lists exactly the outcomes
# of its tries
_verd = ("PASS", "FAIL", "NOT READ")
_bad = []
for rule, v, a_, b_, c_, lk in itertools.product(("WALKS", "HOPS", None), ("agree", "disagree"), _verd, _verd, _verd,
                                                 ("straight", "not straight")):
    o, _lab = ps.decide_seed(rule, v, {"a": a_, "b": b_, "c": c_}, lk)
    if rule is None:
        want = None
    elif rule != "WALKS" or v == "disagree":
        want = "NOT WALKS"
    elif a_ != "PASS" or b_ != "PASS" or lk == "not straight":
        want = "CURVED WALKS"
    else:
        want = "STRAIGHT WALKS" if c_ == "PASS" else "STRAIGHT NO RECOVERY"
    if o != want:
        _bad.append((rule, v, a_, b_, c_, lk, o, want))
check("p4: decide_seed over all 324 known combinations (3 rules x 2 reviews x 27 parts x 2 looks) reads v3's first "
      "rule that applies", (_bad, 3 * 2 * 27 * 2), ([], 324))

# ---- the pilot outcome over the seed outcomes
PO = ps.straight_pilot_outcome
check("p4: two STRAIGHT WALKS and a CURVED WALKS: STRAIGHT WALKS (2 of 3)",
      (lambda o: (o["outcome"], o["basis"], o["final"]))(PO({42: ["STRAIGHT WALKS"], 52: ["STRAIGHT WALKS"],
                                                            60: ["CURVED WALKS"]})),
      ("STRAIGHT WALKS", "2 of 3 launched seeds", True))
check("p4: STRAIGHT WALKS, STRAIGHT NO RECOVERY, CURVED WALKS: MIXED (Q23's case)",
      PO({42: ["STRAIGHT WALKS"], 52: ["STRAIGHT NO RECOVERY"], 60: ["CURVED WALKS"]})["outcome"], "MIXED")
check("p4: three CURVED WALKS: CURVED WALKS (3 of 3)",
      PO({42: ["CURVED WALKS"], 52: ["CURVED WALKS"], 60: ["CURVED WALKS"]})["basis"], "3 of 3 launched seeds")
check("p4: two STRAIGHT NO RECOVERY beside a NOT WALKS: STRAIGHT NO RECOVERY",
      PO({42: ["STRAIGHT NO RECOVERY"], 52: ["NOT WALKS"], 60: ["STRAIGHT NO RECOVERY"]})["outcome"],
      "STRAIGHT NO RECOVERY")
check("p4: NOT WALKS shares across its sub-labels (FALLS and STILL): NOT WALKS",
      PO({42: ["NOT WALKS"], 52: ["NOT WALKS"], 60: ["STRAIGHT WALKS"]})["outcome"], "NOT WALKS")
check("p4: a missing seed beside two that differ: INCOMPLETE",
      (lambda o: (o["outcome"], o["missing"]))(PO({42: ["STRAIGHT WALKS"], 52: [None], 60: ["CURVED WALKS"]})),
      ("INCOMPLETE", [52]))
check("p4: a missing seed beside two that agree: their outcome",
      PO({42: ["CURVED WALKS"], 52: [None], 60: ["CURVED WALKS"]})["outcome"], "CURVED WALKS")
check("p4: every seed missing: no outcome", PO({42: [None], 52: [None], 60: [None]})["outcome"], None)
check("p4: a single launched seed reads its own outcome", PO({42: ["STRAIGHT WALKS"]})["basis"], "single seed (s42)")
_po = PO({42: ["STRAIGHT WALKS"], 52: ["STRAIGHT WALKS"], 60: ["CURVED WALKS", "STRAIGHT WALKS"]})
check("p4: a pending look that cannot change the outcome: it stands, final, the pending seed named",
      (_po["outcome"], _po["final"], _po["pending"], "whatever the pending seeds ['s60'] read" in _po["basis"]),
      ("STRAIGHT WALKS", True, [60], True))
_po = PO({42: ["STRAIGHT WALKS"], 52: ["CURVED WALKS", "STRAIGHT WALKS"], 60: ["CURVED WALKS"]})
check("p4: a pending look that decides: PENDING_REVIEW, its possible outcomes listed",
      (_po["outcome"], _po["final"], _po["possible_outcomes"], _po["pending"]),
      ("PENDING_REVIEW", False, ["CURVED WALKS", "STRAIGHT WALKS"], [52]))
_po = PO({42: ["NOT WALKS", "STRAIGHT WALKS"], 52: [None], 60: ["CURVED WALKS"]})
check("p4: a pending seed beside a missing seed and a CURVED WALKS: INCOMPLETE whatever it reads",
      (_po["outcome"], _po["final"], _po["pending"], _po["missing"]), ("INCOMPLETE", True, [42], [52]))
_po = PO({42: ["NOT WALKS", "CURVED WALKS"], 52: [None], 60: ["CURVED WALKS"]})
check("p4: a pending seed beside a missing seed that it could decide: PENDING_REVIEW (CURVED WALKS or INCOMPLETE)",
      (_po["outcome"], _po["possible_outcomes"], _po["missing"]), ("PENDING_REVIEW", ["CURVED WALKS", "INCOMPLETE"],
                                                                   [52]))
_po = PO({42: ["NOT WALKS", "STRAIGHT WALKS"], 52: ["NOT WALKS", "CURVED WALKS"], 60: ["CURVED WALKS"]})
check("p4: two pending seeds: every combination tried",
      (_po["outcome"], _po["possible_outcomes"]), ("PENDING_REVIEW", ["CURVED WALKS", "MIXED", "NOT WALKS"]))

# ---- the final reading over the seed outcomes
FR = ps.straight_final_reading
_poss = {42: ["STRAIGHT WALKS"], 52: ["CURVED WALKS"], 60: [None]}
_fr = FR(_poss, PO(_poss), [60])
check("p4: final reading, two scored seeds that differ read MIXED",
      (_fr["applies"], _fr["reading"], _fr["final"], _fr["text"]),
      (True, "MIXED", True, "FINAL READING: MIXED from 2 scored seeds of 3 (closed: s60)"))
_poss = {42: ["STRAIGHT NO RECOVERY"], 52: [None], 60: [None]}
_fr = FR(_poss, PO(_poss), [52, 60])
check("p4: final reading, one scored seed: its own outcome, one seed only",
      (_fr["reading"], _fr["one_seed_only"], _fr["text"]),
      ("STRAIGHT NO RECOVERY", True,
       "FINAL READING: STRAIGHT NO RECOVERY (one seed only) from 1 scored seeds of 3 (closed: s52, s60)"))
_poss = {42: [None], 52: [None], 60: [None]}
check("p4: final reading, no scored seed: NO OUTCOME", FR(_poss, PO(_poss), [42, 52, 60])["reading"], "NO OUTCOME")
_poss = {42: ["STRAIGHT WALKS"], 52: ["CURVED WALKS"], 60: [None]}
check("p4: no final reading while a missing seed is not CLOSED", FR(_poss, PO(_poss), [])["applies"], False)
_poss = {42: ["CURVED WALKS"], 52: ["CURVED WALKS"], 60: [None]}
check("p4: no final reading when the pilot outcome stands", FR(_poss, PO(_poss), [60])["applies"], False)
_poss = {42: ["CURVED WALKS", "STRAIGHT WALKS"], 52: ["STRAIGHT WALKS"], 60: [None]}
_fr = FR(_poss, PO(_poss), [60])
check("p4: no final reading while the pilot outcome is PENDING_REVIEW (pilot 3's order: finish the review first)",
      (PO(_poss)["outcome"], _fr["applies"], _fr["why"]),
      ("PENDING_REVIEW", False, "the pilot outcome is not INCOMPLETE"))
_poss = {42: ["CURVED WALKS", "STRAIGHT WALKS"], 52: [None], 60: [None]}
_fr = FR(_poss, PO(_poss), [52, 60])
check("p4: final reading on one scored seed that is pending: PENDING_REVIEW, one seed only, not final",
      (PO(_poss)["outcome"], _fr["applies"], _fr["reading"], _fr["one_seed_only"], _fr["final"]),
      ("INCOMPLETE", True, "PENDING_REVIEW", True, False))

# ---- the existence reading
EX = ps.straight_existence
_x = EX({42: SO("WALKS", "agree", P_ALL, "straight"), 52: SO("FALLS", None, None, None),
         60: SO("STILL", None, None, None)})
check("p4: existence: one STRAIGHT WALKS seed beside two NOT WALKS is STRAIGHT WALK EXISTS (whatever the pilot "
      "outcome)",
      (_x["reading"], _x["A"], _x["k"], _x["final"]), ("STRAIGHT WALK EXISTS", [42], 1, True))
_x = EX({42: SO("WALKS", "agree", P_ALL, None), 52: SO("FALLS", None, None, None), 60: SO("STILL", None, None, None)})
check("p4: existence: a pending look that could make one is PENDING", (_x["reading"], _x["final"]), ("PENDING", False))
_x = EX({42: SO("WALKS", "agree", dict(P_ALL, c="FAIL"), None), 52: SO("FALLS", None, None, None),
         60: SO("STILL", None, None, None)})
check("p4: existence: a pending look that cannot make one (c failed) is NONE SEEN, not final",
      (_x["reading"], _x["pending_could_be_straight_walks"], _x["final"]), ("NONE SEEN", [], False))
_x = EX({42: SO("WALKS", "agree", dict(P_ALL, a="FAIL"), None), 52: SO(None, None, None, None),
         60: SO("STILL", None, None, None)}, closed=[])
check("p4: existence: NONE SEEN with a missing seed not closed is not settled",
      (_x["reading"], _x["open_missing"], _x["final"]), ("NONE SEEN", [52], False))
_x = EX({42: SO("WALKS", "agree", dict(P_ALL, a="FAIL"), None), 52: SO(None, None, None, None),
         60: SO("STILL", None, None, None)}, closed=[52])
check("p4: existence: once the missing seed is CLOSED, NONE SEEN is final", (_x["reading"], _x["final"]),
      ("NONE SEEN", True))
_x = EX({42: SO("WALKS", "agree", P_ALL, "straight"), 52: SO("WALKS", None, None, None),
         60: SO("STILL", None, None, None)})
check("p4: existence: STRAIGHT WALK EXISTS stands beside a pending seed, not final",
      (_x["reading"], _x["pending"], _x["final"]), ("STRAIGHT WALK EXISTS", [52], False))

# ---- the report-only count (Q23) and the look's scope
CT = ps.straight_count
_c = CT({42: ("WALKS", "agree", P_ALL, "straight"), 52: ("WALKS", "agree", dict(P_ALL, c="FAIL"), "straight"),
         60: ("WALKS", "agree", dict(P_ALL, b="FAIL"), "straight")})
check("p4: count (Q23): (a), (b) PASS and the look agrees on 2 of 3, whatever (c) reads",
      (_c["k"], _c["seeds"], _c["could_still_count"]), (2, [42, 52], []))
_c = CT({42: ("WALKS", None, P_ALL, None), 52: ("WALKS", "agree", P_ALL, "not straight"),
         60: ("HOPS", None, None, None)})
check("p4: count: a pending seed could still join, a vetoed one and a NOT WALKS one cannot",
      (_c["k"], _c["could_still_count"]), (0, [42]))
_c = CT({42: ("WALKS", "agree", P_ALL, None), 52: ("WALKS", "agree", dict(P_ALL, c="NOT READ"), None)})
check("p4: count: (a) and (b) PASS with the look still pending is not counted yet (it could join)",
      (_c["k"], _c["seeds"], _c["could_still_count"]), (0, [], [42, 52]))
_c = CT({42: ("WALKS", "disagree", P_ALL, "straight"), 52: ("WALKS", "agree", None, None)})
check("p4: count: a video-disagreed seed is not counted; one with no record could still join",
      (_c["k"], _c["could_still_count"], _c["launched"]), (0, [52], 2))
_c = CT({42: ("WALKS", None, P_ALL, "straight")})
check("p4: count: a pending video review is not counted even with the look 'straight' (it could join)",
      (_c["k"], _c["seeds"], _c["could_still_count"]), (0, [], [42]))
LR = ps.look_required
check("p4: the look is required for WALKS, video-agreed, (a) and (b) PASS, whatever (c) reads",
      [LR("WALKS", "agree", dict(P_ALL, c=c)) for c in _verd], ["REQUIRED"] * 3)
check("p4: the look is not required when (a) or (b) fails, the video disagrees, or the seed is not WALKS",
      [LR("WALKS", "agree", dict(P_ALL, a="FAIL")), LR("WALKS", "agree", dict(P_ALL, b="NOT READ")),
       LR("WALKS", "disagree", P_ALL), LR("STILL", None, P_ALL), LR(None, None, None)], [None] * 5)
check("p4: the look while the review is pending, and with no record yet",
      (LR("WALKS", None, P_ALL), LR("WALKS", "agree", None)),
      ("REQUIRED if the video review agrees (pending)",
       "not yet known: no straightness record read at the deciding checkpoint"))

# ---- the look's file
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "look.json").write_text(json.dumps({"42": "straight", "s52": {"verdict": "Not  Straight", "note": "crabs"},
                                                "60": " STRAIGHT "}))
    check("p4: the look file: plain, s-prefixed and note forms, case and spaces ignored",
          ps.read_straight_review(root / "look.json"),
          {42: {"verdict": "straight", "note": ""}, 52: {"verdict": "not straight", "note": "crabs"},
           60: {"verdict": "straight", "note": ""}})
    for _bad_look in ("curved", "agree", "not_straight", None, True):
        (root / "bad.json").write_text(json.dumps({"42": _bad_look}))
        check(f"p4: a look of {_bad_look!r} is refused",
              raises(lambda: ps.read_straight_review(root / "bad.json"), "straight"), "refused")
    (root / "list.json").write_text(json.dumps(["straight"]))
    check("p4: a look file that is not an object is refused",
          raises(lambda: ps.read_straight_review(root / "list.json"), "object"), "refused")
    check("p4: no look file reads nothing", ps.read_straight_review(None), {})


# ---- straightness records: built to match a synthetic cell, the way pilot4_straight.py's seed mode writes them
def sha_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def binding4(seed: int, **over) -> dict:
    b = dict(OK, schema="walking-pilot4-judge/v1", seed=seed, head="0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8",
             run=f"/copies/pilot4_walk_s{seed}", checkpoint_sha256=f"sha256:{seed:064d}",
             vecnorm_sha256=f"sha256:{seed + 1:064d}", final_fallback=False)
    b.update(over)
    return b


def straight_rec(cell: Path, a="PASS", b="PASS", c="PASS", n_counting=40, not_passed=0, ended=0, **over) -> dict:
    """A seed-mode record of the cell whose values give its verdicts (fix stage: the scorer recomputes them), written
    by the pinned tool (tool_sha256) with v3's constants."""
    binding = json.loads((cell / "binding.json").read_text())
    pa = {"verdict": a, "n": 40, "median_signed_deg": -1.2 if a == "PASS" else -24.7,
          "p95_abs_deg": 9.5 if a == "PASS" else 35.2, "bars": {"median_abs_max_deg": 5.0, "p95_abs_max_deg": 15.0}}
    if a == "NOT READ":
        pa = {"verdict": a, "why": "no full-length panel episode", "n": 0}
    pb = {"verdict": b, "n": 40, "D_p95_deg": 8.1 if b == "PASS" else 30.3, "W0_p95_deg": 14.0 if b == "PASS" else 35.3,
          "report_R_median_deg": 6.0, "report_R_p95_deg": 9.0, "report_last_mean_error_median_deg": -0.5,
          "report_crab_median_deg": 3.0, "report_end_cross_track_median_m": -1.25}
    if b == "NOT READ":
        pb = {"verdict": b, "why": "no full-length panel episode", "n": 0}
    pc = {"n_episodes": 40, "n_counting": n_counting, "left_out_ended_before_nudge": ended,
          "left_out_not_yet_passed": not_passed, "within_11_5": 38, "within_5": 30, "fell_after_nudge": 0,
          "moved_too_little": 0, "share_removed_median": {"plus": 0.9, "minus": 0.88}, "verdict": c}
    if c == "NOT READ":
        pc.update(median_r_deg=None, p95_r_deg=None,
                  why=f"prey not yet passed at 6.5 s in {not_passed} of 40; {n_counting} counting, fewer than 30")
    else:
        pc.update(median_r_deg=2.0 if c == "PASS" else 22.8, p95_r_deg=9.0 if c == "PASS" else "inf")
    rec = {"schema": ps.STRAIGHT_SCHEMA, "mode": "seed", "validation": False, "validation_only_cell": False,
           "partial": False, "tool_sha256": ps.STRAIGHT_TOOL_SHA256,
           "constants": json.loads(json.dumps(ps.STRAIGHT_CONSTANTS)), "head": binding.get("head"),
           "seed": binding.get("seed"), "steps": int(cell.name), "run": binding.get("run"), "cell": str(cell),
           "binding_sha256": sha_file(cell / "binding.json"), "probe_json_sha256": sha_file(cell / "probe.json"),
           "final_fallback": bool(binding.get("final_fallback")), "checkpoint_sha256": binding.get("checkpoint_sha256"),
           "vecnorm_sha256": binding.get("vecnorm_sha256"), "panel": {"all_rolled_match": True, "mismatched": []},
           "parts": {"a": pa, "b": pb, "c": pc, "straight": "PASS" if a == b == c == "PASS" else "FAIL"},
           "files": {"plot": {"path": f"/straight/s{binding.get('seed')}.png", "sha256": "sha256:" + "cd" * 32}}}
    rec.update(over)
    return rec


def put_rec(root: Path, rec: dict, name: str) -> str:
    p = root / "straight" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec))
    return str(p)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    write_cell(root, 42, ps.PRIMARY, [WALK] * 40, binding4(42), META)
    cdir = root / "s42" / str(ps.PRIMARY)
    good = straight_rec(cdir)
    RP = ps.straight_record_problems
    check("p4: a record made from this very cell is read", RP(good, 42, ps.PRIMARY, cdir), [])
    _bad = {
        "schema": dict(good, schema="walking-pilot3-straight/v1"),
        "mode baseline": dict(good, mode="baseline"),
        "partial": dict(good, partial=True),
        "partial missing": {k: v for k, v in good.items() if k != "partial"},
        "a validation record": dict(good, validation=True),
        "another seed": dict(good, seed=52),
        "another checkpoint": dict(good, steps=1500000),
        "binding_sha256 of another binding": dict(good, binding_sha256="sha256:" + "0" * 64),
        "probe_json_sha256 of another probe.json": dict(good, probe_json_sha256="sha256:" + "0" * 64),
        "another checkpoint sha256": dict(good, checkpoint_sha256="sha256:" + "1" * 64),
        "another sidecar sha256": dict(good, vecnorm_sha256="sha256:" + "1" * 64),
        "no sidecar sha256": {k: v for k, v in good.items() if k != "vecnorm_sha256"},
        "another head": dict(good, head="31d02e42d148511e50f9cba9a598d9a9d36c23af"),
        "final_fallback disagrees": dict(good, final_fallback=True),
        "final_fallback missing": {k: v for k, v in good.items() if k != "final_fallback"},
        "a re-rolled episode that did not match": dict(good, panel={"all_rolled_match": False, "mismatched": [7]}),
        "no panel block": {k: v for k, v in good.items() if k != "panel"},
        "a verdict that is not PASS/FAIL/NOT READ": dict(good, parts=dict(good["parts"], b=dict(good["parts"]["b"],
                                                                                              verdict="pass"))),
        "a missing part": dict(good, parts={k: v for k, v in good["parts"].items() if k != "c"}),
        "STRAIGHT PASS with (c) NOT READ": dict(good, parts=dict(good["parts"], c=straight_rec(
            cdir, c="NOT READ", n_counting=26, not_passed=14)["parts"]["c"])),
        "STRAIGHT FAIL with every part PASS": dict(good, parts=dict(good["parts"], straight="FAIL")),
        "STRAIGHT NOT READ": dict(good, parts=dict(good["parts"], straight="NOT READ")),
        # fix stage (reviews: the scorer trusted a record's verdicts, its tool and its bars)
        "another tool": dict(good, tool_sha256="sha256:" + "0" * 64),
        "no tool sha256": {k: v for k, v in good.items() if k != "tool_sha256"},
        "a looser (b) bar": dict(good, constants=dict(good["constants"], B_D_P95_MAX_DEG=15.0)),
        "a later read point": dict(good, constants=dict(good["constants"], READ_K=900)),
        "a constant added": dict(good, constants=dict(good["constants"], EXTRA=1)),
        "no constants": {k: v for k, v in good.items() if k != "constants"},
        "(a) PASS whose median fails": dict(good, parts=dict(good["parts"], a=dict(good["parts"]["a"],
                                                                                   median_signed_deg=7.0))),
        "(a) PASS whose p95 is inf": dict(good, parts=dict(good["parts"], a=dict(good["parts"]["a"], p95_abs_deg="inf"))),
        "(b) PASS whose W0 p95 fails": dict(good, parts=dict(good["parts"], b=dict(good["parts"]["b"],
                                                                                   W0_p95_deg=18.6))),
        "(c) PASS with 29 counting": dict(good, parts=dict(good["parts"], c=dict(good["parts"]["c"], n_counting=29))),
        "(c) PASS whose p95 r is inf": dict(good, parts=dict(good["parts"], c=dict(good["parts"]["c"],
                                                                                   p95_r_deg="inf"))),
        "(a) FAIL whose values pass": dict(good, parts=dict(good["parts"], a=dict(good["parts"]["a"], verdict="FAIL"),
                                                             straight="FAIL")),
        "(b) NOT READ with 40 episodes": dict(good, parts=dict(good["parts"], b=dict(good["parts"]["b"],
                                                                                     verdict="NOT READ"),
                                                               straight="FAIL")),
    }
    for _name, _rec in _bad.items():
        _p = RP(_rec, 42, ps.PRIMARY, cdir)
        check(f"p4: a record is refused for one reason: {_name}", len(_p), 1)
    check("p4: a validation record is read under --validation", RP(dict(good, validation=True), 42, ps.PRIMARY, cdir,
                                                                    validation=True), [])
    check("p4: a record for a cell that does not exist is refused",
          any("no judged cell" in x for x in RP(good, 42, ps.PRIMARY, root / "s42" / "1500000")), True)
    # fix stage: the validity review's forged record (the entry walkers' FAIL values with PASS verdicts, another tool,
    # looser bars) is refused for each of the three
    _forged = dict(good, tool_sha256="sha256:" + "0" * 64,
                   constants=dict(good["constants"], A_P95_ABS_MAX_DEG=40.0, B_D_P95_MAX_DEG=30.0,
                                  B_W0_P95_MAX_DEG=35.0, C_MEDIAN_MAX_DEG=25.0, C_P95_MAX_DEG=45.0),
                   parts={"a": dict(good["parts"]["a"], median_signed_deg=-25.3, p95_abs_deg=31.7),
                          "b": dict(good["parts"]["b"], D_p95_deg=25.2, W0_p95_deg=30.0),
                          "c": dict(good["parts"]["c"], median_r_deg=22.8, p95_r_deg=40.2), "straight": "PASS"})
    _fp = RP(_forged, 42, ps.PRIMARY, cdir)
    check("p4: the review's forged record is refused for its tool, its bars and its verdicts",
          (len(_fp), [x.split(" ")[0] for x in _fp]), (3, ["tool_sha256", "constants", "part"]))
    check("p4: the forged record's verdicts read FAIL from its own values",
          ps.recomputed_verdicts(_forged["parts"]), {"a": "FAIL", "b": "FAIL", "c": "FAIL"})
    _RV = ps.recomputed_verdicts
    _pa, _pb, _pc = good["parts"]["a"], good["parts"]["b"], good["parts"]["c"]
    check("p4: recomputed verdicts at the bars (inclusive; inf and None fail; NOT READ by n and by 30 counting)",
          [_RV({"a": dict(_pa, median_signed_deg=m, p95_abs_deg=p), "b": _pb, "c": _pc})["a"]
           for m, p in ((5.0, 15.0), (-5.0, 15.0), (5.0000001, 15.0), (0.0, 15.0000001), (0.0, "inf"), (None, 1.0))]
          + [_RV({"a": _pa, "b": dict(_pb, D_p95_deg=d, W0_p95_deg=w), "c": _pc})["b"]
             for d, w in ((11.5, 18.5), (11.5000001, 1.0), (1.0, 18.5000001), ("inf", 1.0))]
          + [_RV({"a": _pa, "b": _pb, "c": dict(_pc, n_counting=n, median_r_deg=m, p95_r_deg=p)})["c"]
             for n, m, p in ((30, 5.0, 11.5), (29, 0.0, 0.0), (30, 5.0000001, 1.0), (40, 1.0, "inf"), (True, 0, 0))]
          + [_RV({"a": {"n": 0}, "b": {"n": 0}, "c": {}})[k] for k in ("a", "b", "c")],
          ["PASS", "PASS", "FAIL", "FAIL", "FAIL", "FAIL", "PASS", "FAIL", "FAIL", "FAIL",
           "PASS", "NOT READ", "FAIL", "FAIL", "NOT READ", "NOT READ", "NOT READ", "NOT READ"])
    _tool = HERE / "pilot4_straight.py"
    if _tool.is_file():
        check("p4: STRAIGHT_TOOL_SHA256 is the sha256 of the pilot4_straight.py beside this scorer",
              ps.STRAIGHT_TOOL_SHA256, sha_file(_tool))
    else:
        skip("p4: STRAIGHT_TOOL_SHA256 is the sha256 of pilot4_straight.py", f"no {_tool}")
    check("p4: the summary keeps (c)'s counts and the plot",
          (lambda sm: (sm["c"]["n_counting"], sm["c"]["left_out_not_yet_passed"], sm["plot"]["path"], sm["straight"]))(
              ps.straight_summary(good)), (40, 0, "/straight/s42.png", "PASS"))
    _nr = ps.straight_summary(straight_rec(cdir, c="NOT READ", n_counting=26, not_passed=14))
    check("p4: (c) NOT READ is printed with its why and its counts",
          ps.straight_text(_nr).split(" | ")[2],
          "(c) NOT READ: prey not yet passed at 6.5 s in 14 of 40; 26 counting, fewer than 30; counting 26 of 40 "
          "(left out: 0 ended before the nudge, 14 not yet past the prey at 6.5 s)")
    check("p4: an infinite p95 r (the record's \"inf\") is printed as inf",
          "p95 r inf deg" in ps.straight_text(ps.straight_summary(straight_rec(cdir, c="FAIL"))), True)
    check("p4: _fmt reads the record's forms", [ps._fmt(x) for x in (2.04, "inf", None, True, math.inf, -math.inf,
                                                                       math.nan, 3)],
          ["2.0", "inf", "n/a", "n/a", "inf", "-inf", "nan", "3.0"])
    # read_straight: placement and refusals
    p1 = put_rec(root, good, "a.json")
    check("p4: read_straight places a record by seed and steps, with its file",
          (list(ps.read_straight([p1])), ps.read_straight([p1])[(42, ps.PRIMARY)]["file"]), ([(42, ps.PRIMARY)], p1))
    p2 = put_rec(root, good, "b.json")
    check("p4: two records for one seed and checkpoint are refused",
          raises(lambda: ps.read_straight([p1, p2]), "second straightness record"), "refused")
    for _name, _rec in (("no seed", {k: v for k, v in good.items() if k != "seed"}),
                        ("a string steps", dict(good, steps="2000000")), ("a bool seed", dict(good, seed=True)),
                        ("a list", [good])):
        _p = put_rec(root, _rec, "x.json")
        check(f"p4: a file that is not a record is refused ({_name})", raises(lambda p=_p: ps.read_straight([p]),
                                                                              "not a straightness record"), "refused")

# ---- end to end: the four seed outcomes, NOT READ, a missing review, the veto, NOT WALKS, mixed labels
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for s in (42, 52, 60):
        write_cell(root, s, ps.PRIMARY, [WALK] * 40 if s != 60 else [HOP] * 40, binding4(s), META)
        write_cell(root, s, 1500000, [WALK] * 40, binding4(s, steps=1500000), META)
    cell = {s: root / f"s{s}" / str(ps.PRIMARY) for s in (42, 52, 60)}
    files = [put_rec(root, straight_rec(cell[42]), "s42.json"),
             put_rec(root, straight_rec(cell[52], c="NOT READ", n_counting=22, not_passed=18), "s52.json"),
             put_rec(root, straight_rec(cell[60], a="FAIL", b="FAIL", c="FAIL"), "s60.json"),
             put_rec(root, straight_rec(root / "s42" / "1500000", a="FAIL"), "s42_1500000.json")]
    (root / "agree.json").write_text(json.dumps({"42": "agree", "52": "agree"}))
    (root / "look.json").write_text(json.dumps({"42": "straight", "52": "straight", "60": "not straight"}))
    common = [tmp, "--seeds", "42,52,60", "--video-review", str(root / "agree.json")]
    for f in files:
        common += ["--straight", f]
    rc, text, r = run_main(common + ["--straight-review", str(root / "look.json"), "--json", str(root / "e.json")])
    so = r["seed_outcomes"]
    check("p4: e2e: s42 STRAIGHT WALKS, s52 STRAIGHT NO RECOVERY ((c) NOT READ with its counts), s60 NOT WALKS (HOPS)",
          [(so[s]["outcome"], so[s]["label"]) for s in ("42", "52", "60")],
          [("STRAIGHT WALKS", "(a), (b) and (c) PASS, the look agrees"),
           ("STRAIGHT NO RECOVERY", ("(c) NOT READ: prey not yet passed at 6.5 s in 18 of 40; 22 counting, fewer "
                                     "than 30; counting 22 of 40 (left out: 0 ended before the nudge, 18 not yet past "
                                     "the prey at 6.5 s)")),
           ("NOT WALKS", "HOPS")])
    check("p4: e2e: three different outcomes read MIXED, final", (r["pilot"]["outcome"], r["pilot"]["final"]),
          ("MIXED", True))
    check("p4: e2e: the gait reading beside it (pilot 3's rule: WALKS 2 of 3)",
          (r["gait_reading"]["reviewed"]["outcome"], r["gait_reading"]["reviewed"]["basis"]),
          ("WALKS", "2 of 3 launched seeds"))
    check("p4: e2e: the report-only count is 2 of 3 (s42, s52)", (r["report_only_count"]["k"],
                                                                 r["report_only_count"]["seeds"]), (2, [42, 52]))
    check("p4: e2e: the existence reading is STRAIGHT WALK EXISTS, final",
          (r["existence"]["reading"], r["existence"]["A"], r["existence"]["final"]),
          ("STRAIGHT WALK EXISTS", [42], True))
    check("p4: e2e: the look a NOT WALKS seed got is recorded and changes nothing; the look is listed for s42 and s52",
          (so["60"]["look"], r["straightness_look_required"]["60"],
           [r["straightness_look_required"][s]["state"] for s in ("42", "52")]),
          ("not straight", None, ["REQUIRED", "REQUIRED"]))
    check("p4: e2e: the 1,500,000 record is read report only and does not decide",
          (r["straight"]["42"]["1500000"]["read"], r["straight"]["42"]["1500000"]["deciding"],
           r["straight"]["42"]["2000000"]["deciding"]), (True, False, True))
    check("p4: e2e: the printout carries (c)'s NOT READ with its counts, the NOT WALKS sub-label and the readings",
          ["s52 STRAIGHT at 2,000,000 (deciding): " in text,
           "(c) NOT READ: prey not yet passed at 6.5 s in 18 of 40" in text,
           "NOT WALKS sub-labels: s60 HOPS" in text, "PILOT OUTCOME: MIXED" in text,
           "GAIT READING (pilot 3's pilot outcome in this env" in text, "REPORT ONLY (Q23): 2 of 3" in text,
           "EXISTENCE: STRAIGHT WALK EXISTS" in text,
           ("s60 STRAIGHT at 2,000,000 (deciding; the seed is not WALKS: report only)") in text,
           "s42 STRAIGHT at 1,500,000 (not the deciding checkpoint: report only)" in text],
          [True] * 9)
    check("p4: e2e: exit status 0, and no Q4 key in the JSON", (rc, "q4" in r, "existence" in r), (0, False, True))
    # a missing review: s52 pending its video review
    (root / "agree42.json").write_text(json.dumps({"42": "agree"}))
    args = [x if x != str(root / "agree.json") else str(root / "agree42.json") for x in common]
    _, text, r = run_main(args + ["--straight-review", str(root / "look.json"), "--json", str(root / "m.json")])
    check("p4: e2e, missing review: s52 pending (NOT WALKS or STRAIGHT NO RECOVERY), the pilot outcome PENDING_REVIEW",
          (r["seed_outcomes"]["52"]["outcome"], r["seed_outcomes"]["52"]["possible"], r["pilot"]["outcome"],
           r["pilot"]["possible_outcomes"], "PILOT OUTCOME: PENDING_REVIEW" in text),
          ("pending", ["NOT WALKS", "STRAIGHT NO RECOVERY"], "PENDING_REVIEW", ["MIXED", "NOT WALKS"], True))
    check("p4: e2e, missing review: the existence reading stands (s42) but is not final",
          (r["existence"]["reading"], r["existence"]["final"]), ("STRAIGHT WALK EXISTS", False))
    # the veto: s42's look "not straight"
    (root / "veto.json").write_text(json.dumps({"42": "not straight", "52": "straight"}))
    _, _, r = run_main(common + ["--straight-review", str(root / "veto.json"), "--json", str(root / "v.json")])
    check("p4: e2e, veto: s42 CURVED WALKS (straight by the rule, not on video); no STRAIGHT WALKS left",
          (r["seed_outcomes"]["42"]["outcome"], r["seed_outcomes"]["42"]["label"], r["pilot"]["outcome"],
           r["existence"]["reading"], r["report_only_count"]["k"]),
          ("CURVED WALKS", "straight by the rule, not on video", "MIXED", "NONE SEEN", 1))
    # no look at all: s42 and s52 pending their look
    _, _, r = run_main(common + ["--json", str(root / "n.json")])
    check("p4: e2e, no look: s42 pending (CURVED or STRAIGHT WALKS), s52 pending (CURVED or NO RECOVERY)",
          (r["seed_outcomes"]["42"]["possible"], r["seed_outcomes"]["52"]["possible"], r["pilot"]["outcome"],
           r["existence"]["reading"]),
          (["CURVED WALKS", "STRAIGHT WALKS"], ["CURVED WALKS", "STRAIGHT NO RECOVERY"], "PENDING_REVIEW", "PENDING"))
    # the habit readout is report only: a HABIT record changes nothing
    (root / "h42.json").write_text(json.dumps({"schema": "walking-pilot4-habit/v1", "mode": "seed", "report_only": True,
                                               "seed": 42, "reading": "HABIT", "not_run": [],
                                               "habits": [{"kind": "PARK", "group": "TAIL", "text": "TAIL PARK"}],
                                               "deciding_checkpoint": {"checkpoint_sha256": f"sha256:{42:064d}"}}))
    _, text, r = run_main(common + ["--straight-review", str(root / "look.json"), "--habit", str(root / "h42.json"),
                                    "--json", str(root / "h.json")])
    check("p4: --habit is printed report only and changes no outcome",
          (r["habit_report_only"]["42"]["reading"], r["habit_report_only"]["52"]["reading"],
           r["seed_outcomes"]["42"]["outcome"], r["pilot"]["outcome"],
           "REPORT ONLY habit (pilot4_habit.py; picks nothing): s42 HABIT: TAIL PARK" in text),
          ("HABIT", "NOT RUN", "STRAIGHT WALKS", "MIXED", True))
    # a record of another cell (the 1,500,000 record renamed as the deciding one) is refused: the seed is pending
    other = straight_rec(root / "s42" / "1500000", steps=ps.PRIMARY)
    bad_files = [f for f in files if not f.endswith("s42.json")] + [put_rec(root, other, "s42_other.json")]
    args = [tmp, "--seeds", "42,52,60", "--video-review", str(root / "agree.json"), "--straight-review",
            str(root / "look.json")]
    for f in bad_files:
        args += ["--straight", f]
    _, text, r = run_main(args + ["--json", str(root / "o.json")])
    check("p4: e2e: a record of another cell is refused, named, and its seed waits for a record",
          (r["straight"]["42"]["2000000"]["read"], any("binding_sha256" in x for x in r["straight"]["42"]["2000000"]
                                                        ["problems"]), r["seed_outcomes"]["42"]["outcome"],
           r["seed_outcomes"]["42"]["awaited"], "record not read: " in text),
          (False, True, "pending", ["straightness record"], True))
    # a validation record: refused without --validation, read with it
    val = put_rec(root, straight_rec(cell[42], validation=True), "s42_val.json")
    args = [tmp, "--seeds", "42", "--video-review", str(root / "agree.json"), "--straight-review",
            str(root / "look.json"), "--straight", val]
    _, _, r = run_main(args + ["--json", str(root / "val0.json")])
    _, _, rv = run_main(args + ["--validation", "--json", str(root / "val1.json")])
    check("p4: a validation straightness record is never scored, but is read under --validation",
          (r["seed_outcomes"]["42"]["outcome"], rv["seed_outcomes"]["42"]["outcome"], rv["validation"]),
          ("pending", "STRAIGHT WALKS", True))
    # a record for a seed not launched is listed and ignored
    _, text, r = run_main([tmp, "--seeds", "52,60", "--straight", files[0], "--json", str(root / "ign.json")])
    check("p4: a record for a seed not launched is listed and ignored",
          (r["straight_records_ignored"], "42" in r["straight"]), ([files[0]], False))

# mixed labels across seeds: FALLS, STILL and a STRAIGHT WALKS: NOT WALKS shares across its sub-labels
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    write_cell(root, 42, ps.PRIMARY, [FALL] * 10 + [WALK] * 30, binding4(42), META)
    write_cell(root, 52, ps.PRIMARY, [STATUE] * 40, binding4(52), META)
    write_cell(root, 60, ps.PRIMARY, [WALK] * 40, binding4(60), META)
    f60 = put_rec(root, straight_rec(root / "s60" / str(ps.PRIMARY)), "s60.json")
    (root / "agree.json").write_text(json.dumps({"60": "agree"}))
    (root / "look.json").write_text(json.dumps({"60": "straight"}))
    _, text, r = run_main([tmp, "--seeds", "42,52,60", "--video-review", str(root / "agree.json"), "--straight-review",
                           str(root / "look.json"), "--straight", f60, "--json", str(root / "x.json")])
    check("p4: mixed labels: NOT WALKS (FALLS), NOT WALKS (STILL), STRAIGHT WALKS read NOT WALKS (2 of 3)",
          ([(r["seed_outcomes"][s]["outcome"], r["seed_outcomes"][s]["label"]) for s in ("42", "52", "60")],
           r["pilot"]["outcome"], r["pilot"]["basis"]),
          ([("NOT WALKS", "FALLS"), ("NOT WALKS", "STILL"),
            ("STRAIGHT WALKS", "(a), (b) and (c) PASS, the look agrees")],
           "NOT WALKS", "2 of 3 launched seeds"))
    check("p4: mixed labels: the sub-labels are printed; the gait reading reads MIXED; a walk still exists",
          ("NOT WALKS sub-labels: s42 FALLS, s52 STILL" in text, r["gait_reading"]["reviewed"]["outcome"],
           r["existence"]["reading"]), (True, "MIXED", "STRAIGHT WALK EXISTS"))
    check("p4: a NOT WALKS seed with no straightness record awaits nothing",
          (r["seed_outcomes"]["42"]["awaited"], r["straightness_look_required"]["42"]), ([], None))

# the final-model fallback, the closed seeds and the final readings end to end
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    write_cell(root, 42, ps.PRIMARY, [WALK] * 36 + [STATUE] * 4, binding4(42, final_fallback=True), META)
    write_cell(root, 52, ps.PRIMARY, [STATUE] * 40, binding4(52), META)
    for steps in ps.CHECKPOINTS[:3]:
        write_cell(root, 52, steps, [STATUE] * 40, binding4(52), META)
    (root / "review.json").write_text(json.dumps({"42": "agree"}))
    _, _, r = run_main([tmp, "--seeds", "42,52,60", "--video-review", str(root / "review.json"), "--closed-seeds", "60",
                        "--json", str(root / "p.json")])
    check("p4: e2e, a fallback-decided seed says so",
          ("decided at the final model (fallback" in r["per_seed"]["42"]["basis"],
           r["per_seed"]["42"].get("final_fallback")), (True, True))
    check("p4: e2e, s42 pending its record and look, s52 NOT WALKS (STILL), s60 closed: INCOMPLETE whatever s42 reads",
          (r["pilot"]["outcome"], r["pilot"]["final"], r["pilot"]["pending"]), ("INCOMPLETE", True, [42]))
    check("p4: e2e, the final reading: MIXED whatever s42 reads (two scored seeds, s60 closed)",
          (r["final_reading"]["applies"], r["final_reading"]["reading"], r["final_reading"]["final"],
           r["final_reading"]["closed"]), (True, "MIXED", True, [60]))
    check("p4: e2e, the gait reading keeps pilot 3's final reading (INCOMPLETE, then MIXED)",
          (r["gait_reading"]["reviewed"]["outcome"], r["gait_reading"]["final_reading"]["reading"]),
          ("INCOMPLETE", "MIXED"))
    check("p4: e2e, the trajectory checkpoints are scored but do not decide",
          [r["seeds"]["52"][str(s)]["outcome"] for s in ps.CHECKPOINTS], ["STILL"] * 4)
    cdir = root / "s42" / str(ps.PRIMARY)
    ok_ff = put_rec(root, straight_rec(cdir), "ff.json")
    bad_ff = put_rec(root, straight_rec(cdir, final_fallback=False), "ff_bad.json")
    _, _, r1 = run_main([tmp, "--seeds", "42,52,60", "--video-review", str(root / "review.json"), "--straight", ok_ff,
                         "--json", str(root / "p1.json")])
    _, _, r2 = run_main([tmp, "--seeds", "42,52,60", "--video-review", str(root / "review.json"), "--straight", bad_ff,
                         "--json", str(root / "p2.json")])
    check("p4: e2e, a fallback cell's record must say final_fallback true",
          (r1["straight"]["42"]["2000000"]["read"], r2["straight"]["42"]["2000000"]["read"]), (True, False))
    check("p4: e2e, with s60 open the final reading does not apply",
          run_main([tmp, "--seeds", "42,52,60", "--json", str(root / "p3.json")])[2]["final_reading"]["applies"], False)
    (root / "dup.json").write_text(json.dumps({"seeds": {"42": {"reading": "NO HABIT"}}}))
    (root / "dup2.json").write_text(json.dumps({"seed": 42, "reading": "NO HABIT"}))
    check("p3: two habit records for one seed are refused",
          raises(lambda: ps.main([tmp, "--seeds", "42,52,60", "--habit", str(root / "dup.json"), "--habit",
                                  str(root / "dup2.json")]), "second habit record"), "refused")

# a collapse-stopped seed: its STRAIGHT record is read at the newest intact pair, which decides
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    end = {"periodic_steps": [250000 * i for i in range(1, 8)], "newest_periodic_steps": 1750000,
           "intact_periodic_steps": [250000 * i for i in range(1, 8)], "newest_intact_periodic_steps": 1750000,
           "broken_periodic": {}, "final_num_timesteps": 1900544, "final_source": "stage2_final.zip num_timesteps",
           "budget": 2000000, "stopped_early": True}
    write_cell(root, 42, 1750000, [WALK] * 40, binding4(42, run_end=end), META)
    write_cell(root, 42, 1500000, [WALK] * 40, binding4(42, run_end=end), META)
    f1 = put_rec(root, straight_rec(root / "s42" / "1750000", c="FAIL"), "a.json")
    f2 = put_rec(root, straight_rec(root / "s42" / "1500000"), "b.json")
    (root / "agree.json").write_text(json.dumps({"42": "agree"}))
    (root / "look.json").write_text(json.dumps({"42": "straight"}))
    _, _, r = run_main([tmp, "--seeds", "42", "--video-review", str(root / "agree.json"), "--straight-review",
                        str(root / "look.json"), "--straight", f1, "--straight", f2, "--json", str(root / "c.json")])
    check("p4: a collapse-stopped seed is decided by the record at its newest intact pair (1,750,000: (c) FAIL)",
          (r["seed_outcomes"]["42"]["outcome"], r["seed_outcomes"]["42"]["steps"],
           r["straight"]["42"]["1750000"]["deciding"], r["straight"]["42"]["1500000"]["deciding"]),
          ("STRAIGHT NO RECOVERY", 1750000, True, False))

# fix stage: a cell whose probe.json records an env-kwargs drift is not scored (the probe only warns; review finding)
_drift = {"forward_vel_weight": {"toml_main": 2.0, "recorded_run": 3.0}}
check("p4: pilot4_cell_problems: an empty or absent drift record is no problem, a drift is",
      (ps.pilot4_cell_problems({"meta": dict(META, env_kwargs_drift_vs_recorded={})}),
       ps.pilot4_cell_problems({"meta": META}), ps.pilot4_cell_problems({})),
      ([], [], []))
_sc = ps.score_cell({"meta": dict(META, env_kwargs_drift_vs_recorded=_drift), "episodes": [WALK] * 40}, OK)
check("p4: score_cell: a drifted cell has no outcome and names the drift",
      (_sc["outcome"], _sc["problems"]),
      (None, ['env kwargs drift: the run\'s recorded env kwargs differ from the env it was judged in '
              '({"forward_vel_weight": {"recorded_run": 3.0, "toml_main": 2.0}})']))
check("p4: score_cell: an undrifted cell scores as score_panel does",
      ps.score_cell({"meta": dict(META, env_kwargs_drift_vs_recorded={}), "episodes": [WALK] * 40}, OK),
      ps.score_panel({"meta": dict(META, env_kwargs_drift_vs_recorded={}), "episodes": [WALK] * 40}, OK))
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    write_cell(root, 42, ps.PRIMARY, [WALK] * 40, binding4(42), dict(META, env_kwargs_drift_vs_recorded=_drift))
    write_cell(root, 52, ps.PRIMARY, [WALK] * 40, binding4(52), dict(META, env_kwargs_drift_vs_recorded={}))
    _, text, r = run_main([tmp, "--seeds", "42,52", "--json", str(root / "d.json")])
    check("p4: e2e, a drifted cell's seed has no gait outcome (missing); the other is scored",
          (r["per_seed_primary"]["42"], r["per_seed_primary"]["52"], r["seed_outcomes"]["42"]["outcome"],
           "env kwargs drift" in " ".join(r["seeds"]["42"][str(ps.PRIMARY)]["problems"])),
          (None, "WALKS", None, True))

# ---- habit_report (report only)
HR = ps.habit_report
_h = {"reading": "NO HABIT", "habits": [], "not_run": [], "deciding_checkpoint": {"checkpoint_sha256": "sha256:aa"}}
check("p4: habit report: a record of the deciding checkpoint is read", HR(42, _h, "sha256:aa")["reading"], "NO HABIT")
check("p4: habit report: no record reads NOT RUN", (HR(42, None, "sha256:aa")["reading"], HR(42, None, None)["why"]),
      ("NOT RUN", "no habit record for s42"))
check("p4: habit report: a validation-only record reads NOT RUN",
      HR(42, dict(_h, validation_only=True), "sha256:aa")["reading"], "NOT RUN")
check("p4: habit report: another checkpoint's record reads NOT RUN", HR(42, _h, "sha256:bb")["reading"], "NOT RUN")
check("p4: habit report: a malformed reading reads NOT RUN", HR(42, dict(_h, reading="YES"), "sha256:aa")["reading"],
      "NOT RUN")
check("p4: habit report: HABIT with no habit listed reads NOT RUN",
      HR(42, dict(_h, reading="HABIT"), "sha256:aa")["reading"], "NOT RUN")
check("p4: habit report: a record naming no checkpoint reads NOT RUN",
      HR(42, {k: v for k, v in _h.items() if k != "deciding_checkpoint"}, "sha256:aa")["reading"], "NOT RUN")
check("p4: habit report: the text form",
      ps.habit_text(HR(42, dict(_h, reading="HABIT", habits=[{"text": "NECK ALARM"}]), "sha256:aa")),
      "HABIT: NECK ALARM")

# ---- pilot 3's 8M references (report only constants)
check("p4: P3_8M speeds round to v3's 'Where pilot 3 ended' (2.719 / 2.500 / 2.225, forward 2.701 / 2.423 / 2.205)",
      [(round(ps.P3_8M[s]["medians"]["speed_displacement_mps"], 3),
        round(ps.P3_8M[s]["medians"]["speed_forward_vel_mps"], 3)) for s in (42, 52, 60)],
      [(2.719, 2.701), (2.5, 2.423), (2.225, 2.205)])
check("p4: P3_8M alternation 1.00, flight 0.00, duty L/R 0.48/0.57, 0.51/0.53, 0.45/0.65 (v3)",
      [tuple(round(ps.P3_8M[s]["medians"][k], 2) for k in ("floor_alternation_index", "floor_flight_fraction",
                                                             "floor_duty_L", "floor_duty_R")) for s in (42, 52, 60)],
      [(1.0, 0.0, 0.48, 0.57), (1.0, 0.0, 0.51, 0.53), (1.0, 0.0, 0.45, 0.65)])
check("p4: P3_8M counts WALK 40/40, 39/40 (1 FALL), 40/40 (v3)",
      [ps.P3_8M[s]["counts"] for s in (42, 52, 60)],
      [{"WALK": 40, "HOP": 0, "STILL": 0, "FALL": 0, "OTHER": 0},
       {"WALK": 39, "HOP": 0, "STILL": 0, "FALL": 1, "OTHER": 0},
       {"WALK": 40, "HOP": 0, "STILL": 0, "FALL": 0, "OTHER": 0}])
check("p4: P3_8M holds every REPORT_KEYS median for each seed",
      [sorted(ps.P3_8M[s]["medians"]) == sorted(ps.REPORT_KEYS) for s in (42, 52, 60)], [True] * 3)
check("p4: every REPORT_KEYS key has a short printed name", sorted(ps.P3_8M_SHORT), sorted(ps.REPORT_KEYS))
_fs = P3_JUDGING / "final_score" / "score_final.json"
if _fs.is_file() and all((P3_JUDGING / "OUT" / f"s{s}" / "8000000" / "probe.json").is_file() for s in (42, 52, 60)):
    check("p4: P3_8M's source file is the one it names (sha256 dd023b22...)", sha_file(_fs),
          "sha256:dd023b229d66a18421ebe4edf1143848b7b2add822476d84296ff3ef6b960744")
    _fsd = json.loads(_fs.read_text())
    for s in (42, 52, 60):
        _rows = json.loads((P3_JUDGING / "OUT" / f"s{s}" / "8000000" / "probe.json").read_text())["episodes"]
        _again = {k: ps._median([ps._num(r, k) for r in _rows]) for k in ps.REPORT_KEYS}
        _cell = _fsd["seeds"][str(s)]["8000000"]
        check(f"p4: P3_8M s{s} equals pilot 3's final score and the medians recomputed from its probe.json",
              (ps.P3_8M[s]["medians"] == _cell["medians"] == _again, ps.P3_8M[s]["counts"] == _cell["counts"],
               ps.P3_8M[s]["probe_classes"] == _cell["probe_classes"]), (True, True, True))
else:
    skip("p4: P3_8M against pilot 3's judged cells", f"no {_fs} or OUT/s<seed>/8000000/probe.json")

# ---- real data, read only: pilot 3's judged 8M cells with the straightness lane's validation records of those cells
_recs = [STRAIGHT_VAL / f"p3val_s{s}.json" for s in (42, 52, 60)]
_rev = P3_JUDGING / "final_score" / "video_review.json"
if (P3_JUDGING / "OUT").is_dir() and _rev.is_file() and all(p.is_file() for p in _recs):
    with tempfile.TemporaryDirectory() as tmp:
        args = [str(P3_JUDGING / "OUT"), "--seeds", "42,52,60", "--primary", "8000000", "--video-review", str(_rev)]
        for p in _recs:
            args += ["--straight", str(p)]
        rc, text, r = run_main(args + ["--validation", "--json", str(Path(tmp) / "real.json")])
        so = r["seed_outcomes"]
        check("p4: real cells (pilot 3's 8M, validation records): CURVED WALKS on all three, (a) and (b) FAIL",
              [(so[s]["outcome"], so[s]["label"]) for s in ("42", "52", "60")],
              [("CURVED WALKS", "(a) FAIL, (b) FAIL")] * 3)
        check("p4: real cells: the pilot outcome CURVED WALKS (3 of 3), the gait reading WALKS (3 of 3)",
              (r["pilot"]["outcome"], r["pilot"]["basis"], r["gait_reading"]["reviewed"]["outcome"],
               r["gait_reading"]["reviewed"]["basis"]),
              ("CURVED WALKS", "3 of 3 launched seeds", "WALKS", "3 of 3 launched seeds"))
        check("p4: real cells: existence NONE SEEN (final), count 0 of 3, no look required",
              (r["existence"]["reading"], r["existence"]["final"], r["report_only_count"]["k"],
               [r["straightness_look_required"][s] for s in ("42", "52", "60")]),
              ("NONE SEEN", True, 0, [None] * 3))
        check("p4: real cells: (c)'s values as the straightness lane recorded them (median r, counting)",
              [(round(r["straight"][s]["8000000"]["summary"]["c"]["median_r_deg"], 1),
                r["straight"][s]["8000000"]["summary"]["c"]["n_counting"]) for s in ("42", "52", "60")],
              [(round(json.loads(p.read_text())["parts"]["c"]["median_r_deg"], 1),
                json.loads(p.read_text())["parts"]["c"]["n_counting"]) for p in _recs])
        check("p4: real cells: the 8M medians equal the P3_8M constants (the report-only comparison wired)",
              [all(v["pilot4"] == v["pilot3_8m"] for v in r["p3_8m_reference"][s]["medians"].values())
               for s in ("42", "52", "60")], [True] * 3)
        check("p4: real cells: the records carry the pinned tool's sha256 and v3's constants (fix stage)",
              [(json.loads(p.read_text())["tool_sha256"] == ps.STRAIGHT_TOOL_SHA256,
                json.loads(p.read_text())["constants"] == ps.STRAIGHT_CONSTANTS,
                ps.recomputed_verdicts(json.loads(p.read_text())["parts"])
                == {k: json.loads(p.read_text())["parts"][k]["verdict"] for k in ("a", "b", "c")}) for p in _recs],
              [(True, True, True)] * 3)
        check("p4: real cells: exit 0 and the validation mark printed",
              (rc, "[--validation: validation judges and records accepted; never for the verdict]" in text,
               "PILOT OUTCOME: CURVED WALKS (3 of 3 launched seeds)  [validation]" in text), (0, True, True))
        _, _, r0 = run_main(args + ["--json", str(Path(tmp) / "real0.json")])
        check("p4: real cells without --validation: the validation records are refused, so each seed waits for a "
              "record and the pilot outcome is PENDING_REVIEW",
              ([r0["seed_outcomes"][s]["outcome"] for s in ("42", "52", "60")], r0["pilot"]["outcome"],
               all(any("validation record" in x for x in r0["straight"][s]["8000000"]["problems"])
                   for s in ("42", "52", "60"))),
              (["pending"] * 3, "PENDING_REVIEW", True))
else:
    skip("p4: real pilot-3 cells with the straightness lane's records", f"no {P3_JUDGING / 'OUT'}, {_rev} or {_recs}")
if (JUDGE_E2E / "s42" / "2000000" / "binding.json").is_file():
    with tempfile.TemporaryDirectory() as tmp:
        _, text, r = run_main([str(JUDGE_E2E), "--seeds", "42", "--json", str(Path(tmp) / "j.json")])
        check("p4: the judge lane's real pilot-4 binding (stand-in weights, every episode a FALL) reads NOT WALKS "
              "(FALLS)",
              (r["seeds"]["42"]["2000000"]["problems"], r["seeds"]["42"]["2000000"]["counts"]["FALL"],
               r["seed_outcomes"]["42"]["outcome"], r["seed_outcomes"]["42"]["label"], r["pilot"]["outcome"]),
              ([], 40, "NOT WALKS", "FALLS", "NOT WALKS"))
else:
    skip("p4: the judge lane's pilot-4 cell", f"no {JUDGE_E2E / 's42' / '2000000' / 'binding.json'}")

# Recompute the probe-built cases from the frozen probe when possible (numpy + a pilot tree)
tree = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PILOT_TREE")) or ""
try:
    import numpy  # noqa: F401
    have_np = True
except ImportError:
    have_np = False
if have_np and tree and (Path(tree) / "docs/investigations/gait_2026_09/gait_probe.py").is_file():
    import make_gait_cases as mgc
    spec = importlib.util.find_spec("environments")
    env_origin = getattr(spec, "origin", None) if spec is not None else None
    print(f"(environments resolves to {env_origin}; the tree is {Path(tree).resolve()})")
    check("environments resolves inside the tree given (run through iso.py)",
          bool(env_origin) and str(Path(env_origin).resolve()).startswith(str(Path(tree).resolve()) + os.sep), True)
    probe, path = mgc.load_probe(Path(tree))
    check("fixture probe sha256 equals the tree's probe", mgc.sha(path), CASES["probe_sha256"])
    again = mgc.synthetic_rows(probe)
    for name, c in CASES["synthetic"].items():
        check(f"recomputed {name} equals the fixture", again[name]["row"], c["row"])
else:
    print(f"(probe recomputation skipped: numpy {'yes' if have_np else 'no'}, tree {tree or 'not given'})")
    skip("probe recomputation", f"numpy {'yes' if have_np else 'no'}, tree {tree or 'not given'}")

failed = [c for c in cases if not c[1]]
for name, ok, got, want in cases:
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
for line in skipped:
    print(f"skip {line}")
print(f"{len(cases) - len(failed)}/{len(cases)} passed, {len(skipped)} skipped")
sys.exit(1 if failed else 0)
