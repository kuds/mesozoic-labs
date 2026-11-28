# E1 clash reach: summary (28 Nov 2026)

Run as pre-registered in `design/PREREG_CANDIDATES_2026_11_28.md` (section 4 E1; frozen code `sim/candidate_checks.py`
sha256 `008aa308debdf1b6`, plan.json `1403a4b326aaf2aa`, the adopted model `372ac07b14338c26`, commit `fb1e80c8`; all
17 build-file hashes equal plan.json's). **SIMULATION** logs (MuJoCo 3.10.0, the build's env at its 1 ms step) and
**DESK** geometry (`sim/check_packaging60.py`'s parts through `ClashGeometry`). Nothing is measured, and nothing here is
adopted: any stop is the owner's decision.

Files: `e1_summary.json` (the verdict, every controller's judged figures, the reported tables, every episode, provenance),
`e1_stops.csv`, `e1_by_condition.csv`, `e1_joint_percentiles.csv`, `e1_beyond_cap.json` (supporting, added after a
result). Raw logs, per-pose clearances and the runners are in the session scratch folder,
`/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/E1/` (`run_e1.py`
sha256 `e8b576e79d7c8197`, `summarise_e1.py`, `beyond_cap_e1.py`).

## What ran

- The four fore128 policies (checkpoints checked against their stored sweeps, loaded through the sweep's loaders): walk
  0.25 m/s, walk 0.15 m/s, stand, sideways pushes of 0.5 and 1.0 m/s left and right at the 0.25 m/s command; seeds
  700000-700003 and 710000-710003 in every condition. 224 episodes of 20 s, 10 ms delay (10 steps at 1 ms, every
  episode).
- The frozen trot (`walk60.TrotController(WalkParams())`, `PitchedTrot` at 0 deg): J1-J4, push back 0.5 and push right
  0.5 m/s, seeds 101-104. 24 episodes.
- Joint angles every 10 ms: 480,024 poses in all (s0 and s2 112,000 each, s1 106,430, s3 101,594, trot 48,000).
  Falls: 14 of 224 policy episodes (all "tilt", all at 1.0 m/s pushes: s1 push left 1.0 5 of 8; s3 push right 1.0 8
  of 8 and push left 1.0 1 of 8), 0 of 24 trot episodes; no "unstable_simulation" ending; no pose dropped as non-finite.
  Fallen episodes are included up to their termination (as pre-registered); the reading with them left out is the same.

## The rule's outcomes

**#9 NOT REACHED; #10 NOT REACHED.** No controller has any logged pose with clearance < 0 (0 of 5 for each; the rule
needs 2). No logged pose is even under 5 mm (0.000 % of samples on every controller, both clashes, soles in or aside).

Every logged pose read the frozen geometry's 50 mm cap (`check_packaging60.CAP_M`): the figure is ">= 50 mm"
everywhere. Supporting, **added after a result** (not part of the rule; `e1_beyond_cap.json`): the actual least,
bracketed by the parts' containing-capsule lower bound over every pose, and an exact `mj_geomDistance` (0.5 m cap) at the
30 poses with the smallest bound:

| Controller | #9 (either side), lower bound / measured, mm | #10, lower bound / measured, mm |
|---|---|---|
| fore128-s0 | 160.7 / 160.7 (the two left soles) | 103.0 / 129.6 |
| fore128-s1 | 174.5 / 178.7 | 89.0 / 129.6 |
| fore128-s2 | 179.2 / 184.7 | 89.4 / 129.6 |
| fore128-s3 | 177.1 / 181.9 | 80.7 / 129.6 |
| trot | 205.7 / 210.1 | 107.6 / 129.6 |

(#10's measured 129.6 mm is the two hind shoulder yokes, which do not move with the legs; the hind soles' own least is
between the lower bound and that figure.)

**Stops** (each applied to both sides; ACCEPTABLE if on each of the five controllers the least clearance after the stop
is >= 5 mm and at most 1 % of its samples are clipped). Since every pose is >= 50 mm clear, only the clip share decides.

| Clash | Stop | Record's least over the narrowed full range, mm (DESK, context) | Worst clipped share over the five | ACCEPTABLE |
|---|---|---|---|---|
| #10 | hind inward roll >= -17 deg | 4.79 | 0.00 % | yes |
| #10 | hind inward roll >= -16 deg | 18.11 | 0.00 % | yes |
| #10 | hind inward roll >= -14 deg | 44.82 | 0.00 % | yes |
| #10 | coupled software limit at 5 mm | - | 0.00 % | yes |
| #9 | fore pitch <= +20 / +10 / +5 / 0 deg | -48.00 / -25.50 / 0.53 / 26.38 | 7.49 / 55.83 / 90.30 / 100.00 % | no / no / no / no |
| #9 | hind pitch >= -30 deg | -48.00 | 0.00 % | yes |
| #9 | hind pitch >= -20 / -15 / -10 deg | -42.08 / -11.39 / 19.32 | 77.21 / 96.08 / 98.91 % | no / no / no |
| #9 | both, fore <= p and hind >= -p, p = 20 / 15 / 10 deg | -34.02 / 1.46 / 44.16 | 78.63 / 97.56 / 99.22 % | no / no / no |
| #9 | elbow and knee flexion >= -30 deg | -48.00 | 0.00 % | yes |
| #9 | elbow and knee flexion >= -20 / -10 deg | -24.02 / 36.87 | 8.52 / 50.65 % | no / no |
| #9 | coupled software limit at 5 mm | - | 0.00 % | yes |

Per-controller shares are in `e1_stops.csv` (for example fore pitch <= +5 deg clips 47.55 / 46.59 / 30.98 / 7.34 % of
s0-s3 and 90.30 % of the trot; flexion >= -10 deg clips 50.65 / 36.44 / 1.29 / 23.90 % and 0.00 % of the trot).

## Reading it (reported, not judged)

- The logged ranges stay well inside the model's joint ranges. Extremes over all five controllers: roll -10.29 to +9.65
  deg (the trot's left fore, s2's right hind; model -18 / +24), fore pitch up to +30.24 deg (s2; model +38), hind pitch
  down to -24.32 deg (s2; model -38), elbow / knee flexion down to -23.53 deg (s0; model -47). The policies' own targets
  are bounded to the neutral pose (0) +- 0.15 rad roll and +- 0.40 rad pitch and flexion (the run records'
  `action_scale_*`; the env clips to them). The record's overlapping poses for #9 need, together, the fore leg pitched
  +9.5 deg or more, the hind leg -19 deg or less, both legs flexed and the two rolls within 21 deg, and its deepest
  are at elbow and knee flexion near -46 / -47 deg; #10 sits only at hind roll -18 deg. Both deepest poses are outside
  these controllers' target ranges, and no logged pose came within 160 mm (#9) or 80 mm (#10) (lower bounds above).
- For **#10**, the -16 and -14 deg hind inward roll stops are ACCEPTABLE on these logs and also clear the two hind legs
  over the whole narrowed range by 18.1 and 44.8 mm (the record, DESK). The -17 deg stop is ACCEPTABLE here but clears
  the full range by only 4.79 mm.
- For **#9**, the stops that are ACCEPTABLE here (hind pitch >= -30 deg, flexion >= -30 deg) do not on their own remove
  the overlap over the full narrowed range (the record: -48.0 mm for each); every single stop that does (fore pitch <=
  +5 or 0, hind pitch >= -10, both at 15 or 10, flexion >= -10) clips more than 1 % of at least one controller's
  samples (worst of the five 50.65-100 %). The
  coupled software limit is ACCEPTABLE (no sample under 5 mm). A combination of the two ACCEPTABLE stops, or a stop at
  the logged extremes, is not among the record's candidates and was not tested.
- What this does not cover: the stops are read on today's controllers' logged poses, not on gaits trained with them;
  turning, starts, stops, falls onto the ground and handling are not in the logs; no 50 cm model was logged.
