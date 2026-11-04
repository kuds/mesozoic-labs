"""Cases for pilot3_score.py (plain Python; run: python3 -I -B test_pilot3_score.py [PILOT_TREE]).

Walking pilot 3's copy of pilot 1's test_pilot_score.py (sha256 f96b50f9...), adapted only where it names checkpoints
or the budget (8,000,000; checkpoints every 500,000; the RUN_ID pilot3_walk_s<seed>), plus the cases named "p3: ..."
for the added readings. Pilot 1's 254 cases (with the fixture recomputation) are the rest.

The staggered-hop, scissor-hop, synchronous-hop, short-lift-hop, chatter and walk rows in fixtures/gait_cases.json were
computed by the frozen gait probe's own analyze_episode (make_gait_cases.py; the "ct_" rows with the round-2
verifier's continuous-time generator, the "ev_" rows with the round-3 verifier's event generator); the real walker
rows are the probe's re-run of the audit's compsognathus walker. With numpy importable and
a pilot tree given (argument or PILOT_TREE), the synthetic rows are recomputed from the probe and must equal the
fixture.
"""

from __future__ import annotations

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
import pilot3_score as ps  # noqa: E402

CASES = json.loads((HERE / "fixtures" / "gait_cases.json").read_text())
SYN = {k: v["row"] for k, v in CASES["synthetic"].items()}
REAL = CASES["real_walker"]["rows"]

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


def check(name, got, want):
    cases.append((name, got == want, got, want))


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

# Seed outcomes
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


# End to end on synthetic directories
META = {"seed_scheme": "panel", "seed_base": 3042, "settle_steps_gait_window": 100}
OK = {"validation_only": False, "expected_budget": 8000000, "off_schedule": False}
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    write_cell(root, 42, ps.PRIMARY, [HOP] * 40, OK, META)
    write_cell(root, 52, ps.PRIMARY, [HOP] * 30 + [STATUE] * 10, OK, META)
    write_cell(root, 60, ps.PRIMARY, [WALK] * 40, OK, META)
    write_cell(root, 42, ps.CHECKPOINTS[0], [WALK] * 40, OK, dict(META, seed_scheme="vecenv"))
    write_cell(root, 60, ps.CHECKPOINTS[1], [WALK] * 40, dict(OK, validation_only=True, expected_budget=50000), META)
    write_cell(root, 60, ps.CHECKPOINTS[0], [WALK] * 40, None, META)
    write_cell(root, 99, ps.PRIMARY, [WALK] * 40, OK, META)  # a seed nobody launched
    ps.main([tmp, "--seeds", "42,52,60", "--json", str(root / "score.json")])
    res = json.loads((root / "score.json").read_text())
    check("e2e pilot outcome", res["pilot"]["outcome"], "HOPS")
    check("e2e s60 walks", res["per_seed_primary"]["60"], "WALKS")
    check("e2e wrong seed scheme is not scored", res["seeds"]["42"][str(ps.CHECKPOINTS[0])]["outcome"], None)
    check("e2e missing checkpoint", res["seeds"]["52"][str(ps.CHECKPOINTS[1])], None)
    check("e2e validation-only judge is not scored", res["seeds"]["60"][str(ps.CHECKPOINTS[1])]["outcome"], None)
    check("e2e judge without binding is not scored", res["seeds"]["60"][str(ps.CHECKPOINTS[0])]["outcome"], None)
    check("e2e undeclared seed dir ignored", res["undeclared_seed_dirs_ignored"], [99])
    check("e2e undeclared seed not scored", "99" in res["per_seed"], False)
    ps.main([tmp, "--seeds", "42,52,60", "--validation", "--json", str(root / "score_val.json")])
    resv = json.loads((root / "score_val.json").read_text())
    check("e2e --validation accepts it", resv["seeds"]["60"][str(ps.CHECKPOINTS[1])]["outcome"], "WALKS")

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # report only: OTHER episodes that pass the contact bar, by the test that rejects them
    rows = ([SYN["ct_scissor_hop_8.3Hz_30ms"]] * 20 + [SYN["ct_stagger_hop_8.3Hz_30ms"]] * 10
            + [SYN["ct_limp_walk_1.2Hz_30pct"]] * 5 + [SYN["ev_shortlift_hop_10Hz_d0.85_-30ms_follow"]] * 3
            + [SYN["ct_walk_1.0Hz_sym"]] * 2)
    write_cell(root, 42, ps.PRIMARY, rows, OK, META)
    ps.main([tmp, "--seeds", "42", "--json", str(root / "r.json")])
    cell = json.loads((root / "r.json").read_text())["seeds"]["42"][str(ps.PRIMARY)]
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
    ps.main([tmp, "--seeds", "42,52,60", "--json", str(root / "a.json")])
    r = json.loads((root / "a.json").read_text())
    check("launched 3, judged 1: INCOMPLETE", r["pilot"]["outcome"], "INCOMPLETE")
    check("launched 3, judged 1: missing listed", r["pilot"]["missing"], [52, 60])
    ps.main([tmp, "--seeds", "42", "--json", str(root / "b.json")])
    check("launched 1: single seed", json.loads((root / "b.json").read_text())["pilot"]["basis"], "single seed (s42)")

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # s42: an off-schedule judgement at the primary checkpoint (judged with --report-off-schedule) is never scored
    write_cell(root, 42, ps.PRIMARY, [WALK] * 40,
               dict(OK, off_schedule=True, off_schedule_reason="OFF-SCHEDULE: resumed at step 500,000"), META)
    # s52: collapse-stopped at 3,801,088; judged at 2,000,000, 3,000,000 and its newest periodic 3,500,000
    end = {"periodic_steps": [500000 * i for i in range(1, 8)], "newest_periodic_steps": 3500000,
           "final_num_timesteps": 3801088, "final_source": "stage2_final.zip num_timesteps", "budget": 8000000,
           "stopped_early": True}
    for steps, rows in ((2000000, [WALK] * 40), (3000000, [FALL] * 10 + [WALK] * 30), (3500000, [FALL] * 25 + [STATUE] * 15)):
        write_cell(root, 52, steps, rows, dict(OK, run_end=end), META)
    # s60: stopped early but its newest periodic checkpoint was not judged
    end60 = dict(end, newest_periodic_steps=3500000)
    write_cell(root, 60, 3000000, [HOP] * 40, dict(OK, run_end=end60), META)
    ps.main([tmp, "--seeds", "42,52,60", "--json", str(root / "c.json")])
    r = json.loads((root / "c.json").read_text())
    check("off-schedule primary is not scored", r["seeds"]["42"][str(ps.PRIMARY)]["outcome"], None)
    check("off-schedule seed is missing", r["per_seed_primary"]["42"], None)
    check("collapse-stopped seed scored from its newest periodic checkpoint", r["per_seed_primary"]["52"], "FALLS")
    check("collapse-stopped seed marked", r["per_seed"]["52"].get("collapse_stopped"), True)
    check("collapse-stopped seed, newest checkpoint unjudged: missing", r["per_seed_primary"]["60"], None)
    check("collapse-stopped seed, newest checkpoint unjudged: says so",
          "has no valid judgement" in r["per_seed"]["60"]["basis"], True)
    check("pilot with one scored seed of three: INCOMPLETE", r["pilot"]["outcome"], "INCOMPLETE")
    check("FIX2: a binding without the intact list falls back to the newest zip by file name, and says so",
          "newest periodic checkpoint by file name" in r["per_seed"]["52"]["basis"], True)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # FIX2: collapse-stopped, newest periodic pair broken (6,000,000: not judged, the judge refuses it); the newest
    # INTACT pair 5,500,000 is judged and decides (the review's bin/collapse_newest_broken.py case)
    end = {"periodic_steps": [500000 * i for i in range(1, 13)], "newest_periodic_steps": 6000000,
           "intact_periodic_steps": [500000 * i for i in range(1, 12)], "newest_intact_periodic_steps": 5500000,
           "broken_periodic": {"6000000": "missing sidecar stage2_vecnormalize_6000000_steps.pkl"},
           "final_num_timesteps": 6062080, "final_source": "stage2_final.zip num_timesteps", "budget": 8000000,
           "stopped_early": True}
    for steps, rows in ((2000000, [STATUE] * 40), (4000000, [STATUE] * 40), (5500000, [FALL] * 6 + [STATUE] * 34)):
        write_cell(root, 42, steps, rows, dict(OK, run_end=end), META)
    # s52: collapse-stopped with no intact periodic pair at all
    end52 = dict(end, intact_periodic_steps=[], newest_intact_periodic_steps=None)
    write_cell(root, 52, 2000000, [STATUE] * 40, dict(OK, run_end=end52), META)
    ps.main([tmp, "--seeds", "42,52", "--json", str(root / "i.json")])
    r = json.loads((root / "i.json").read_text())
    check("FIX2: collapse-stopped, newest pair broken: read at the newest intact pair",
          (r["per_seed_primary"]["42"], r["per_seed"]["42"]["steps"]), ("FALLS", 5500000))
    check("FIX2: the basis names the newest intact periodic pair",
          "newest intact periodic pair 5,500,000" in r["per_seed"]["42"]["basis"], True)
    check("FIX2: collapse-stopped with no intact pair: missing, says so",
          (r["per_seed_primary"]["52"], "with no newest intact periodic pair" in r["per_seed"]["52"]["basis"]),
          (None, True))

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # FIX2: one run per seed. s42 was restarted under _b, but the retired run's 2M and 4M cells were left in OUT/s42
    run_a = dict(OK, run="/copies/pilot3_walk_s42")
    run_b = dict(OK, run="/copies/pilot3_walk_s42_b")
    write_cell(root, 42, 2000000, [STATUE] * 40, run_a, META)
    write_cell(root, 42, 4000000, [STATUE] * 40, run_a, META)
    write_cell(root, 42, ps.PRIMARY, [HOP] * 40, run_b, META)
    write_cell(root, 52, ps.PRIMARY, [HOP] * 40, dict(OK, run="/copies/pilot3_walk_s52"), META)
    ps.main([tmp, "--seeds", "42,52", "--json", str(root / "m.json")])
    r = json.loads((root / "m.json").read_text())
    check("FIX2: a seed whose cells come from two runs is not scored",
          (r["per_seed_primary"]["42"], r["seeds"]["42"][str(ps.PRIMARY)]["outcome"]), (None, None))
    check("FIX2: the problem names both runs",
          "['pilot3_walk_s42', 'pilot3_walk_s42_b']" in r["seeds"]["42"]["2000000"]["problems"][-1], True)
    check("FIX2: a seed from one run is scored as before", r["per_seed_primary"]["52"], "HOPS")
    # the retired cells moved, never deleted, to OUT/retired/<run>/s42/<steps>/
    for steps in ("2000000", "4000000"):
        dst = root / "retired" / "pilot3_walk_s42" / "s42" / steps
        dst.parent.mkdir(parents=True, exist_ok=True)
        (root / "s42" / steps).rename(dst)
    ps.main([tmp, "--seeds", "42,52", "--json", str(root / "m2.json")])
    r = json.loads((root / "m2.json").read_text())
    check("FIX2: after the move the _b run is scored", r["per_seed_primary"]["42"], "HOPS")
    check("FIX2: the retired cells are listed, never scored", r["retired_cells_never_scored"],
          ["retired/pilot3_walk_s42/s42/2000000", "retired/pilot3_walk_s42/s42/4000000"])
    check("FIX2: one_run_problem ignores cells without a recorded run",
          ps.one_run_problem({1: {"run": None}, 2: {"run": "a"}, 3: None}), None)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # s42 resumed off schedule: pilot_judge --report-off-schedule wrote it under OUT/off_schedule/<run>/; the seed was
    # restarted under "_b" and that run judged into OUT/s42/ as usual
    off = root / "off_schedule" / "RUN_s42" / "s42" / str(ps.PRIMARY)
    off.mkdir(parents=True)
    (off / "probe.json").write_text(json.dumps({"meta": META, "episodes": [WALK] * 40}))
    (off / "binding.json").write_text(json.dumps(dict(OK, off_schedule=True, off_schedule_reason="OFF-SCHEDULE: x")))
    write_cell(root, 42, ps.PRIMARY, [HOP] * 40, OK, META)
    ps.main([tmp, "--seeds", "42", "--json", str(root / "o.json")])
    r = json.loads((root / "o.json").read_text())
    check("off-schedule report beside the restart: the seed's outcome is the restart's", r["per_seed_primary"]["42"], "HOPS")
    check("off-schedule report is listed, never scored",
          (list(r["off_schedule_reports"]), r["off_schedule_reports"][f"off_schedule/RUN_s42/s42/{ps.PRIMARY}"]["outcome"]),
          ([f"off_schedule/RUN_s42/s42/{ps.PRIMARY}"], None))
    check("off_schedule/ is not read as a seed directory", r["undeclared_seed_dirs_ignored"], [])

# Video review: the watch list, the per-seed verdicts and the pilot outcome while a review is pending
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    chatter = SYN["ct_chatter_22Hz_d0.74_-40ms"]           # WALK by the rule, short swing (known limit)
    shortlift = SYN["ev_shortlift_hop_10Hz_d0.85_-30ms_follow"]  # OTHER, a foot over the duty line
    rows = [WALK] * 31 + [chatter] + [shortlift] * 2 + [STATUE] * 5 + [SCOOT]
    write_cell(root, 42, ps.PRIMARY, rows, OK, META)
    ps.main([tmp, "--seeds", "42", "--json", str(root / "v0.json")])
    r = json.loads((root / "v0.json").read_text())
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
    check("notebook replay named for the seed's run", "pilot3_walk_s42/03_locomotion/replays/" in req["notebook_replay"]
          and req["notebook_replay"].split(" ")[0].endswith("tyrannosaurus_rex_anatomical_ppo_stage2_final.mp4"), True)
    check("single seed WALKS by the rule, no review: PENDING_VIDEO_REVIEW, never WALKS",
          (r["per_seed_primary"]["42"], r["per_seed"]["42"]["video_review"], r["pilot"]["outcome"], r["pilot"]["final"],
           r["pilot_by_rule"]["outcome"]), ("WALKS", "pending", "PENDING_VIDEO_REVIEW", False, "WALKS"))
    (root / "agree.json").write_text(json.dumps({"42": "agree"}))
    (root / "disagree.json").write_text(json.dumps({"s42": {"verdict": "Disagree", "note": "both feet land together"}}))
    ps.main([tmp, "--seeds", "42", "--video-review", str(root / "agree.json"), "--json", str(root / "v1.json")])
    r1 = json.loads((root / "v1.json").read_text())
    check("review agrees: WALKS, final", (r1["per_seed_reviewed"]["42"], r1["pilot"]["outcome"], r1["pilot"]["final"]),
          ("WALKS", "WALKS", True))
    ps.main([tmp, "--seeds", "42", "--video-review", str(root / "disagree.json"), "--json", str(root / "v2.json")])
    r2 = json.loads((root / "v2.json").read_text())
    check("review disagrees: the seed is MIXED, the rule's WALKS kept on record",
          (r2["per_seed_primary"]["42"], r2["per_seed_reviewed"]["42"], r2["pilot"]["outcome"],
           r2["video_review_verdicts"]["42"]["note"]), ("WALKS", "MIXED", "MIXED", "both feet land together"))
    (root / "bad.json").write_text(json.dumps({"42": "yes"}))
    try:
        ps.main([tmp, "--seeds", "42", "--video-review", str(root / "bad.json")])
        bad = "accepted"
    except SystemExit as exc:
        bad = "refused" if "agree" in str(exc) else f"other: {exc}"
    check("a verdict other than agree/disagree is refused", bad, "refused")
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


# ======================================================================== p3: walking pilot 3's changes and readings
check("p3: CHECKPOINTS and PRIMARY", (ps.CHECKPOINTS, ps.PRIMARY), ((2000000, 4000000, 6000000, 8000000), 8000000))
import ast  # noqa: E402
import hashlib  # noqa: E402

_SRC = (HERE / "pilot3_score.py").read_text()
_PILOT1_BODIES = {  # sha256 of each function's source in pilot 1's pilot_score.py (b4297e13...)
    "seed_outcome": "d4734f60cd1a0456d40286f1e571a4cb5414cf3140dcf4c367a21209300b0f43",
    "pilot_outcome": "b9f83dbcd22e162a9e252567dbac92e9ac8ae22433bcfdd997b94830e798d2e8",
    "reviewed_outcome": "65178231056c7c2a9dda4a98298f1d9611c2cc0d16d514d6ea719f0fc3eb35e4",
    "pilot_outcome_reviewed": "ba9ceb0d6741ffbbeb3fcc5207ebe5834c8426f3aa12b026a069252eef5a893d"}
for _node in ast.parse(_SRC).body:
    if isinstance(_node, ast.FunctionDef) and _node.name in _PILOT1_BODIES:
        check(f"p3: {_node.name} is byte-identical to pilot 1's",
              hashlib.sha256(ast.get_source_segment(_SRC, _node).encode()).hexdigest(), _PILOT1_BODIES[_node.name])

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

E = ps.existence_reading
_ag, _dis = {"verdict": "agree"}, {"verdict": "disagree"}
check("p3: existence, one agreed WALKS seed of three is WALK EXISTS (minority)",
      (lambda r: (r["reading"], r["A"], r["k"]))(E({42: "WALKS", 52: "STILL", 60: "STILL"}, {42: _ag})),
      ("WALK EXISTS", [42], 1))
check("p3: existence, a pending review is PENDING",
      E({42: "WALKS", 52: "STILL", 60: "STILL"}, {})["reading"], "PENDING")
check("p3: existence, a disagreed WALKS seed is NO WALK SEEN",
      E({42: "WALKS", 52: "STILL", 60: "STILL"}, {42: _dis})["reading"], "NO WALK SEEN")
check("p3: existence, no WALKS seed is NO WALK SEEN", E({42: "HOPS", 52: "STILL", 60: None}, {})["reading"],
      "NO WALK SEEN")
check("p3: existence, one agreed and one pending is WALK EXISTS, A not settled",
      (lambda r: (r["reading"], r["a_final"]))(E({42: "WALKS", 52: "WALKS", 60: "STILL"}, {42: _ag})),
      ("WALK EXISTS", False))
check("p3: a minority WALKS seed now needs its review (pilot 1's outcome stays STILL, final)",
      (lambda o: (o["outcome"], o["final"], o["pending_video_review"]))(
          ps.pilot_outcome_reviewed({42: "WALKS", 52: "STILL", 60: "STILL"}, {})), ("STILL", True, [42]))


def _hab(reading, *habits, sha="sha256:aa", not_run=()):
    return {"reading": reading, "habits": [{"kind": k, "group": g, "text": f"{g} {k}"} for k, g in habits],
            "deciding_checkpoint": {"checkpoint_sha256": sha}, "not_run": list(not_run)}


def _q4(rule, review, habit, closed=(), sha=None):
    pil = ps.pilot_outcome_reviewed(rule, review)
    fr = ps.final_reading(rule, review, pil, list(closed))
    ex = ps.existence_reading(rule, review, list(closed))
    return ps.q4_reading(ex, pil, fr, habit, rule, sha or {s: "sha256:aa" for s in rule})


_r = _q4({42: "WALKS", 52: "STILL", 60: "STILL"}, {42: _ag}, {42: _hab("NO HABIT")})
check("p3: Q4 YES on a minority, printed with k of 3 and the pilot outcome",
      (_r["reading"], _r["context"]), ("Q4 YES", "minority: 1 of 3; pilot outcome STILL"))
_r = _q4({42: "WALKS", 52: "WALKS", 60: "STILL"}, {42: _ag, 52: _ag},
         {42: _hab("HABIT", ("TAIL ALARM", "TAIL")), 52: _hab("NO HABIT")})
check("p3: two WALKS seeds disagreeing on habit read NO, habit back",
      (_r["reading"], _r["context"]), ("Q4 NO, habit back", "2 of 3; pilot outcome WALKS"))
_r = _q4({42: "WALKS", 52: "WALKS", 60: "WALKS"}, {42: _ag, 52: _ag, 60: _ag},
         {42: _hab("HABIT", ("NECK ALARM", "NECK")), 52: _hab("NO HABIT"), 60: _hab("NO HABIT")})
check("p3: a neck-only habit reads NO, habit back (neck only)", _r["reading"], "Q4 NO, habit back (neck only)")
_r = _q4({42: "WALKS", 52: "WALKS", 60: "WALKS"}, {42: _ag, 52: _ag, 60: _ag},
         {42: _hab("HABIT", ("NECK ALARM", "NECK")), 52: _hab("HABIT", ("PARK", "TAIL")), 60: _hab("NO HABIT")})
check("p3: a neck habit beside a tail habit in A is not neck only", _r["reading"], "Q4 NO, habit back")
_r = _q4({42: "WALKS", 52: "WALKS", 60: "HOPS"}, {42: _ag, 52: _ag}, {42: _hab("NO HABIT"), 52: _hab("NOT RUN")})
check("p3: NOT RUN in A with no HABIT reads UNREAD", _r["reading"], "Q4 UNREAD")
_r = _q4({42: "WALKS", 52: "WALKS", 60: "HOPS"}, {42: _ag, 52: _ag}, {42: _hab("NO HABIT")})
check("p3: a seed in A without a habit record reads NOT RUN, so UNREAD",
      (_r["reading"], _r["statuses"]["52"]["why"]), ("Q4 UNREAD", "no habit record for s52"))
_r = _q4({42: "WALKS", 52: "STILL", 60: "STILL"}, {42: _ag}, {42: _hab("NO HABIT", sha="sha256:bb")})
check("p3: a habit record of another checkpoint reads NOT RUN", (_r["reading"], _r["statuses"]["42"]["reading"]),
      ("Q4 UNREAD", "NOT RUN"))
_r = _q4({42: "WALKS", 52: "WALKS", 60: "STILL"}, {42: _ag, 52: _ag},
         {42: _hab("HABIT", ("DRIVEN tail_1_yaw_act -0.65", "TAIL")), 52: _hab("NO HABIT")})
check("p3: a DRIVEN tail is a habit", _r["reading"], "Q4 NO, habit back")
_r = _q4({42: "WALKS", 52: "STILL", 60: "STILL"}, {42: _ag}, {42: dict(_hab("NO HABIT"), validation_only=True)})
check("FIX3: a validation-only habit record reads NOT RUN (Q4 UNREAD, not YES)",
      (_r["reading"], _r["statuses"]["42"]["reading"], "validation-only" in (_r["statuses"]["42"]["why"] or "")),
      ("Q4 UNREAD", "NOT RUN", True))
_r = _q4({42: "WALKS", 52: "STILL", 60: "STILL"}, {42: _ag},
         {42: dict(_hab("HABIT", ("PARK", "TAIL")), validation_only=True)})
check("FIX3: a validation-only HABIT record is not read either", _r["statuses"]["42"]["reading"], "NOT RUN")
_r = _q4({42: "WALKS", 52: "STILL", 60: "STILL"}, {42: _ag}, {42: dict(_hab("NO HABIT"), validation_only=False)})
check("FIX3: validation_only false is read as before", _r["reading"], "Q4 YES")
_r = _q4({42: "STILL", 52: "STILL", 60: "HOPS"}, {}, {52: _hab("HABIT", ("TAIL ALARM at 1,310,720", "TAIL"))})
check("p3: NO WALK SEEN reads Q4 NO by the pilot outcome, a habit outside A printed with its seed",
      (_r["reading"], _r["context"], _r["habit_outside_A"]),
      ("Q4 NO by the pilot outcome: STILL", "0 of 3; pilot outcome STILL",
       ["s52 STILL, HABIT: TAIL TAIL ALARM at 1,310,720"]))
_r = _q4({42: "WALKS", 52: "STILL", 60: "STILL"}, {}, {42: _hab("NO HABIT")})
check("p3: a pending review leaves Q4 not read", (_r["reading"], _r["text"].startswith("Q4 NOT READ YET")), (None, True))
_r = _q4({42: "WALKS", 52: "WALKS", 60: "WALKS"}, {42: _ag, 52: _ag}, {})
check("p3: an agreed pair with a third review pending leaves Q4 not read (A not settled)",
      (_r["reading"], "s60" in _r["text"]), (None, True))
_r = _q4({42: "HOPS", 52: "STILL", 60: None}, {}, {})
check("p3: INCOMPLETE without closed seeds leaves Q4 not read", _r["reading"], None)

F = ps.final_reading
_fr = F({42: "HOPS", 52: "STILL", 60: None}, {}, ps.pilot_outcome({42: "HOPS", 52: "STILL", 60: None}), [60])
check("p3: final reading, two scored seeds that differ read MIXED", (_fr["applies"], _fr["reading"], _fr["text"]),
      (True, "MIXED", "FINAL READING: MIXED from 2 scored seeds of 3 (closed: s60)"))
_r = _q4({42: "HOPS", 52: "STILL", 60: None}, {}, {}, closed=[60])
check("p3: Q4 by the final reading MIXED", (_r["reading"], _r["context"]),
      ("Q4 NO by the final reading: MIXED", "0 of 3; final reading MIXED"))
_fr = F({42: "STILL", 52: None, 60: None}, {}, ps.pilot_outcome({42: "STILL", 52: None, 60: None}), [52, 60])
check("p3: final reading, one scored seed reads its outcome, one seed only",
      (_fr["reading"], _fr["one_seed_only"], _fr["text"]),
      ("STILL", True, "FINAL READING: STILL (one seed only) from 1 scored seeds of 3 (closed: s52, s60)"))
_r = _q4({42: "STILL", 52: None, 60: None}, {}, {}, closed=[52, 60])
check("p3: one scored seed and NO WALK SEEN reads Q4 UNREAD", _r["reading"], "Q4 UNREAD")
_fr = F({42: None, 52: None, 60: None}, {}, ps.pilot_outcome({42: None, 52: None, 60: None}), [42, 52, 60])
check("p3: final reading, no scored seed reads NO OUTCOME", (_fr["reading"], _fr["k_scored"]), ("NO OUTCOME", 0))
check("p3: no scored seed: Q4 UNREAD", _q4({42: None, 52: None, 60: None}, {}, {}, closed=[42, 52, 60])["reading"],
      "Q4 UNREAD")
_r = _q4({42: "WALKS", 52: None, 60: None}, {42: _ag}, {42: _hab("NO HABIT")}, closed=[52, 60])
check("p3: one scored WALKS seed, agreed, the others closed: Q4 YES on the final reading",
      (_r["reading"], _r["context"]), ("Q4 YES", "1 of 3; final reading WALKS (one seed only)"))
check("p3: no final reading while a missing seed is not closed",
      F({42: "HOPS", 52: "STILL", 60: None}, {}, ps.pilot_outcome({42: "HOPS", 52: "STILL", 60: None}), [])["applies"],
      False)
check("p3: no final reading when the pilot outcome stands",
      F({42: "HOPS", 52: "HOPS", 60: None}, {}, ps.pilot_outcome({42: "HOPS", 52: "HOPS", 60: None}), [60])["applies"],
      False)
_fr = F({42: "WALKS", 52: "STILL", 60: None}, {}, ps.pilot_outcome_reviewed({42: "WALKS", 52: "STILL", 60: None}, {}),
        [60])
check("p3: final reading with a pending review of a WALKS seed: MIXED either way, final",
      (_fr["reading"], _fr["final"]), ("MIXED", True))
try:
    ps.read_closed_seeds("52,99", [42, 52, 60])
    _bad = "accepted"
except SystemExit as exc:
    _bad = "refused" if "99" in str(exc) else f"other: {exc}"
check("p3: --closed-seeds naming a seed not launched is refused", _bad, "refused")

# end to end: the final-model fallback, the readings in score.json, the habit files, the closed seeds
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    META_SHA = dict(META, checkpoint_sha256="sha256:aa")
    write_cell(root, 42, ps.PRIMARY, [WALK] * 36 + [STATUE] * 4, dict(OK, final_fallback=True), META_SHA)
    write_cell(root, 52, ps.PRIMARY, [STATUE] * 40, OK, META_SHA)
    for steps in ps.CHECKPOINTS[:3]:
        write_cell(root, 52, steps, [STATUE] * 40, OK, META_SHA)
    (root / "review.json").write_text(json.dumps({"42": "agree"}))
    (root / "h42.json").write_text(json.dumps(dict(_hab("HABIT", ("NECK ALARM at 655,360", "NECK")), seed=42)))
    (root / "h52.json").write_text(json.dumps({"seeds": {"52": _hab("NO HABIT")}}))
    ps.main([tmp, "--seeds", "42,52,60", "--video-review", str(root / "review.json"), "--closed-seeds", "60",
             "--habit", str(root / "h42.json"), "--habit", str(root / "h52.json"), "--json", str(root / "p3.json")])
    r = json.loads((root / "p3.json").read_text())
    check("p3: e2e, a fallback-decided seed says so", ("decided at the final model (fallback" in r["per_seed"]["42"]["basis"],
                                                      r["per_seed"]["42"].get("final_fallback")), (True, True))
    check("p3: e2e, the pilot outcome is pilot 1's (INCOMPLETE), the final reading MIXED",
          (r["pilot"]["outcome"], r["final_reading"]["reading"], r["final_reading"]["closed"]), ("INCOMPLETE", "MIXED", [60]))
    check("p3: e2e, existence and Q4", (r["existence"]["reading"], r["q4"]["reading"]),
          ("WALK EXISTS", "Q4 NO, habit back (neck only)"))
    check("p3: e2e, video review required only for the WALKS-by-rule seed",
          (r["video_review_required"]["42"]["required"], r["video_review_required"]["52"]["required"]), (True, False))
    check("p3: e2e, the replay is named pilot3_walk_s<seed>",
          r["video_review_required"]["52"]["notebook_replay"].split("/")[6:8], ["pilot3_walk_s52", "03_locomotion"])
    check("p3: e2e, the 2M-6M checkpoints are scored but do not decide",
          [r["seeds"]["52"][str(s)]["outcome"] for s in ps.CHECKPOINTS], ["STILL"] * 4)
    (root / "dup.json").write_text(json.dumps({"seeds": {"42": _hab("NO HABIT")}}))
    try:
        ps.main([tmp, "--seeds", "42,52,60", "--habit", str(root / "h42.json"), "--habit", str(root / "dup.json")])
        _dup = "accepted"
    except SystemExit as exc:
        _dup = "refused" if "second habit record" in str(exc) else f"other: {exc}"
    check("p3: two habit records for one seed are refused", _dup, "refused")

# Recompute the probe-built cases from the frozen probe when possible (numpy + a pilot tree)
tree = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PILOT_TREE")) or ""
try:
    import numpy  # noqa: F401
    have_np = True
except ImportError:
    have_np = False
if have_np and tree and (Path(tree) / "docs/investigations/gait_2026_09/gait_probe.py").is_file():
    import make_gait_cases as mgc
    probe, path = mgc.load_probe(Path(tree))
    check("fixture probe sha256 equals the tree's probe", mgc.sha(path), CASES["probe_sha256"])
    again = mgc.synthetic_rows(probe)
    for name, c in CASES["synthetic"].items():
        check(f"recomputed {name} equals the fixture", again[name]["row"], c["row"])
else:
    print(f"(probe recomputation skipped: numpy {'yes' if have_np else 'no'}, tree {tree or 'not given'})")

failed = [c for c in cases if not c[1]]
for name, ok, got, want in cases:
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
print(f"{len(cases) - len(failed)}/{len(cases)} passed")
sys.exit(1 if failed else 0)
