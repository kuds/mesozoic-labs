# E2 transmission and speed cap: results (28 Nov 2026)

Pre-registration: `design/PREREG_CANDIDATES_2026_11_28.md` (section 4 E2; section 3 items 3-6, 13, 14). Code
`sim/candidate_checks.py` (sha256 `008aa308debdf1b6`, unchanged), runner `e2_run.py` (sha256 `f63f0b297250d30c`,
kept in the session scratch folder `candidates/E2/code/`), commit `fb1e80c8`, MuJoCo 3.10.0. **DESK** = the build's
thermal model and arithmetic on simulated torques; **SIMULATION** = MuJoCo episodes of `rl/trike_env.py`;
**MEASURED** = none. Nothing here is adopted: a reduction, a transmission or a drive move is the owner's decision.

Files here: `e2a_grid.csv` (every E2(a) row, per policy and seed block), `e2a_summary.json` (verdicts, speed checks,
per-block inputs, provenance), `e2a_speed_x6_allowance_supporting.csv` (item 6's X6 figure, supporting),
`e2b_summary.json` (every comparison, run statistics, verdicts, provenance), `e2b_comparisons.csv`, `e2_checks.json`
(input checks, harness check, script hashes).

## What was run

- **SIMULATION, 1 ms, adopted model (B1):** the four fore128 policies x 5 conditions (walk 0.25, push left 0.5, push
  right 0.5, stand, walk 0.15) x seeds 700000-700003 and 710000-710003 = 160 episodes. These are E1's episodes (same
  code path, same seeds). After the run they were compared with the E1 session's own files: **160 of 160
  bit-identical** (sq_motor, sq_brake, speed histogram, poses; `e2_checks.json`).
- **SIMULATION, 4 new demo courses** (`rl/demo_course.py` `run` via `demo_make`, seed0 710000, default loop): all valid,
  one-pack minutes 93.31 / 83.76 / 85.50 / 90.46 (s0-s3).
- **SIMULATION, 0.25 ms (frame_skip 80, delay 40 steps = 10 ms):** B025, R, rod, belt, cable and belt_1span x 4 policies
  x 3 conditions x 8 seeds = 576 episodes. With B1's 96 that makes the pre-registered 672.
- **Item 4(b):** no episode in any start run ended "unstable_simulation" (and none fell), so `e2b_judged_runs` asked for
  no 0.125 ms rerun and B0125 was not run (output: "item 4(b): rerun at 0.125 ms: nothing").
- **Harness check, not a result:** one 3 s episode per start run, fore128-s0, seed 12345 (not an evaluation seed). The
  delay was 10 / 40 steps = 0.010 s and the control step 0.02 s. The episode's torques, logged over 2-3 s, are not used
  anywhere.

## E2(a) DESK: C1, C2 and the speed check (rule: worst of the four policies on both seed blocks)

The split mapping was used "exact": the B1 split reproduces every stored and new demo's `tau_drive_sq_mean` to at
most 1.2e-14 relative, against the 1e-6 threshold. With no reduction, today's figures are C1 / C2 = 76.3 C / 0.761
(s0, block A), 74.9 / 0.750 (s0 B), 74.4 / 0.746 (s1 A), 76.4 / 0.762 (s1 B), 93.3 / 0.881 (s2 A), 91.0 / 0.867
(s2 B), 98.3 / 0.914 (s3 A) and 97.3 / 0.909 (s3 B).

**Verdict: no judged (transmission, r, placement) PASSES; 0 of 45.**

| Placement | r | C1 worst, C (rod / belt / cable) | C2 worst (rod / belt / cable) | Reduced drives' worst p99 / peak, rpm (cap 83 / 91.3) | C1 | C2 | Speed |
|---|---|---|---|---|---|---|---|
| pitch | 1.0 | 103.2 / 100.1 / 102.1 | 0.942 / 0.924 / 0.936 | 65.7 / 85.6 | fail | fail | pass |
| pitch | 1.1 | 89.0 / 86.6 / 88.2 | 0.856 / 0.840 / 0.851 | 72.3 / 94.2 | pass | fail | fail |
| pitch | 1.2 | 79.0 / 77.1 / 78.3 | 0.785 / 0.770 / 0.780 | 78.8 / 102.7 | pass | pass | **fail (peak)** |
| pitch | 1.25 | 75.0 / 73.3 / 74.4 | 0.753 / 0.739 / 0.748 | 82.1 / 107.0 | pass | pass | **fail (peak)** |
| pitch | 1.3 | 73.0 / 73.0 / 73.0 | 0.739 / 0.739 / 0.739 | 85.4 / 111.3 | pass | pass | **fail (p99, peak)** |
| knees | 1.0-1.3 | 98.3 (all) | 0.914 (all) | 71.7-93.2 / 96.6-125.6 | fail | fail | fail |
| both | 1.0 | 103.2 / 100.1 / 102.1 | 0.942 / 0.924 / 0.936 | 71.7 / 96.6 | fail | fail | fail |
| both | 1.1 | 89.0 / 86.6 / 88.2 | 0.856 / 0.840 / 0.851 | 78.9 / 106.3 | pass | fail | fail |
| both | 1.2 | 79.0 / 77.1 / 78.3 | 0.785 / 0.770 / 0.780 | 86.0 / 115.9 | pass | pass | fail |
| both | 1.25 | 75.0 / 73.3 / 74.4 | 0.753 / 0.739 / 0.748 | 89.6 / 120.8 | pass | pass | fail |
| both | 1.3 | 71.6 / 70.1 / 71.1 | 0.724 / 0.711 / 0.720 | 93.2 / 125.6 | pass | pass | fail |

- **Where it fails.**
  - Every C1 / C2 worst is policy s3's left hind hip pitch, except pitch at r = 1.3, where it is s2's left hind knee
    (a drive that reduction does not touch).
  - Every speed failure comes from policy s3 alone:
    - at hip pitch, its right fore pitch drive: p99 65.5 / 65.7 rpm and peak 82.6 / 85.6 rpm at r = 1 (blocks
      A / B), so x r over 91.3 rpm at peak from r = 1.1 (block B) and r = 1.2 (both blocks), and over 83 rpm at
      p99 at r = 1.3;
    - at the knees, its left fore elbow, an X6-60: peak 96.6 / 94.9 rpm already at r = 1.
- **Knee-only reductions** leave C1 and C2 unchanged at 98.3 C / 0.914, since the deciding drive is a hip pitch.
- **Per policy (supporting, not the verdict).** At hip pitch, r = 1.2-1.3, s0, s1 and s2 pass C1, C2 and the speed
  check on both blocks with every transmission; s3 passes C1 and C2 but not the speed check.
- **Item 6's X6 allowance, shown beside (supporting).** With the X6 at 108 rpm (peak 118.8;
  `gait_pipe1.RPM_ALLOW`) and the X4s at 83:
  - the speed check alone would pass at knees r = 1.0-1.2 and both r = 1.0, but those rows fail C1 and C2;
  - at pitch and both from r = 1.1, s3 still fails on its X4 right fore pitch;
  - no verdict changes.
- **Rod at the build's 0.90 (supporting).** At hip pitch: r = 1.2 gives 82.8 C / 0.813 (s3 fails C2), r = 1.25 gives
  78.4 / 0.780, r = 1.3 gives 74.7 / 0.750; the speed check fails as above. The control row (rod 0.90, knees, r = 1)
  is today's figures exactly.
- **Walk 0.15 m/s, shown beside (not judged).** s3's hip pitch: p99 55.4-55.6 rpm, peak 72.9-77.1 rpm at r = 1.

## E2(b) SIMULATION: r = 1.25 at hip pitch and knees, series stiffness, 83 rpm cap

Rule: COMPATIBLE if at least 3 of 4 policies pass every condition against both B1 and B025 on the same seeds. Pass
means no new falls, |mean speed - baseline| <= 0.02 m/s, and worst-drive RMS / continuous (drive side) - baseline
<= 0.03.

| Variant | Role | Verdict | Policies passing | New falls | Speed change vs B1 and B025, m/s | Worst-drive ratio change | Samples at or over the cap (worst episode) | Largest spring deflection, mrad |
|---|---|---|---|---|---|---|---|---|
| rod (eta 0.95) | judged | **COMPATIBLE** | 4 of 4 | 0 | -0.0028 to +0.0010 | -0.212 to -0.128 | 1.24 % | 2.0 |
| belt (0.98) | judged | **COMPATIBLE** | 4 of 4 | 0 | +0.0060 to +0.0097 | -0.252 to -0.099 | 1.79 % | 32.0 |
| cable (0.96) | judged | **COMPATIBLE** | 4 of 4 | 0 | -0.0022 to +0.0008 | -0.211 to -0.119 | 1.12 % | 6.2 |
| R (rigid, 0.95) | supporting | passes the same rule | 4 of 4 | 0 | -0.0013 to +0.0015 | -0.181 to -0.108 | 0.77 % | - |
| belt_1span | supporting | passes the same rule | 4 of 4 | 0 | +0.0057 to +0.0098 | -0.302 to -0.128 | 1.49 % | 61.7 |
| rod at eta 0.90 (same episodes, re-mapped) | supporting | passes the same rule | 4 of 4 | - | as rod | -0.179 to -0.104 | - | - |

- **Falls and stability.** No episode in any run fell or ended unstable.
- **Pairing.** Every comparison paired all 8 seeds.
- **Each policy's worst-drive ratio over the three conditions** (walking and pushes, 2-20 s, not the demo loop's C2):
  0.836-0.999 in the baselines, 0.678-0.828 in the rod variant.
- **The cap binds almost only on s3**: 0.77-1.79 % of its samples, against at most 0.11 % on the other policies.
- **Drive speeds still exceed the cap.** The cap removes driving torque but leaves braking torque, so impacts and the
  spring push the drive past 83 rpm:
  - worst drive p99 is 82.3-83.9 rpm;
  - peaks are 94.4 (R), 114.8 (rod), 104.0 (belt), 110.2 (cable) and 149.7 rpm (belt_1span);
  - B1 peaks at 108.7 rpm, on s3's X6 elbow.

## Changes from the pre-registration

1. **E1's episodes were run by this runner** rather than read from E1's files, because E1's results did not exist when
   E2 started. Decided before any E2 run. The identity check against E1's 160 episodes was done afterwards and found
   them identical.
2. **The new demo courses use `room_c` None** (25 C), as the stored ones do. Only their segment torques and one-pack
   minutes are used; C1 is recomputed at 35 C by `heat_rules`. Decided before the run.
3. **The cap share counts the whole episode** (every physics step from reset), not the 2-20 s window.
4. **The X6 supporting reading's script was written after the E2(a) output was seen.** Its content is item 6's
   pre-registered "shown beside" figure, it is not judged and it changes no outcome.
5. **No verdict, rule, seed or threshold was changed after a result.**

## What it means (desk and simulation; no hardware)

- **For A (60 cm, today's drives).**
  - Pre-registered outcome of (a): "C1 and C2 pass only where the speed check fails". With today's four policies, the
    83 rpm cap blocks the reduction route at 60 cm.
  - The block is one policy, s3, the hot one. Its right fore hip pitch already runs at 0.99-1.03 of 83 rpm at peak.
  - A 1.2-1.3:1 hip-pitch reduction would bring every policy under both heat lines on the desk.
  - Pre-registered outcome of (b): "Compliance of this size does not break today's policies; the choice rests on
    packaging, upkeep and cost (desk)". In simulation the 83 rpm cap at r = 1.25 is tolerated in straight walking and
    0.5 m/s side pushes: no new falls, and speed within 0.010 m/s.
  - This does not override (a). (a) is the lead's criterion on today's uncapped gaits. The cap model is an idealised
    firmware envelope, and the maker's real behaviour above 83 rpm is unknown.
  - A reduction for A would want a policy trained with the cap and the reduction, or a policy other than s3 (E3's
    per-policy reading).
- **On cables.** At these stiffnesses (upper bounds; no rod-end, tooth or construction-stretch compliance), rod, belt
  and cable are equally compatible with today's policies in simulation.
  - This simulation gives no evidence against a cable or belt, nor for the claim that stretch fails the policies.
  - Wear, pretension loss and upkeep were not modelled.
  - Desk caveats still open: the belt needs 385 / 770 N per span to keep both spans taut, which no source gives, and
    the pre-registration's sizing gives the 15 mm belt's tight side 1,571 N on the X6 knee, against about 2,890 N
    ultimate per span, so a wider belt would be wanted there.
  - The lead's view that a 1.1-1.3 reduction helps heat holds on the desk. "Within the 83 rpm drive limit" does not
    hold for policy s3 under the pre-registered speed check.
  - The claim that drives moved off the legs help packaging and cooling was not tested: leg inertia and mounting are
    unchanged.
- **For B (50 cm, L1).** Not tested here. The study gives joint speed rising as S^-0.5 (s3's hip pitch at 1.10 of the
  cap at 50 cm), so on today's policies a reduction there would fail the speed check by more than at 60 cm. This is an
  inference, not a result.
