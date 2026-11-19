# The trike60-run2b-ankle policy against the first one (simulation)

19 Nov 2026. Analysis of the owner's second Colab run (8,003,584 steps, roll-ankle foot, the new reward; run commit
e92fd9ef), following `PLAN_RUN2B.md`, which was committed (97cbf038) before any of the tools below were run on this
policy. It is compared with the first policy, trike60-pipe1 (2,007,040 steps, PR #669's foot, the first reward). One
policy each, one training seed each: every number is reported, not judged. Simulation of a model whose masses and
contact are estimates.

Sources (tool output, not retyped by hand except where a table says so):

- Gait report: `rl/analytics/results/run2b/gait_report.md` and `.json`; pipe1's in `rl/analytics/results/pipe1/`.
- Robustness sweep: `rl/robustness/results/run2b/summary_run2b.md` and `sweep_run2b.json.gz`; pipe1's in
  `rl/robustness/results/pipe1/`.
- Training: `rl/analytics/results/training_pipe1_vs_run2b/` (dashboard JSON and `training_compare.png`) and
  `rl/RUNS.md`.

## Answer first

- **Cooler, more efficient and steadier than the first policy, at the same speed.** At the 0.25 m/s command run2b walks
  at 0.245 m/s (pipe1 0.238) with no falls, in a clean trot: 1.22 strides per second of 20 cm each, against pipe1's
  2.39 Hz and 10 cm. Every drive stays under its continuous rating in every case of the gait report: the worst is the
  left hind knee at 0.72 of its rating (mean of 5 episodes; 0.77 in the worst; 0.86 for the right fore pitch in the
  worst episode on the other foot), against pipe1's 1.10 (1.15). The feet slip about a quarter as much (slip ratio
  0.05-0.09 against 0.23-0.29). The leg drives' electrical estimate falls from 106-123 W to 40-50 W, so the estimated
  walking time on one battery set roughly doubles, from 46-52 to 88-101 min (30 W hotel load assumed). The head's shake
  at 5-20 Hz halves (4.4 against 9.4 m/s² RMS). Run2b differs from pipe1 in the reward, the foot, the training length
  (8.0 M against 2.0 M steps), the penalty schedules and the phase clock (off), so these differences cannot be credited
  to any one change.
- **Hind knee (F12):** this policy loads the X4 hind knee to 0.72 of its continuous rating, below the budget's
  average-load estimate of 1.00. One cooler policy reports a lower load; it does not settle F12.
- **Time at the torque limit is now about 1 % on several drives, which one depending on the condition** (pipe1:
  3.7-22.5 %). On the model as built (no randomisation) the right fore pitch drive is at its nominal limit 0.9 % of the
  time, just under the plan's 1 % flag. In the sweep: the hind pitch drives under extra mass, payload, slopes, higher kp
  and delays of 20 ms and more (1.1-11.7 %); the right hind knee on slippery floors (1.2-2.2 % at friction 0.15-0.3) and
  at mass x0.8 (1.0 %); the right fore pitch at most other points (0.9-1.6 %, up to 13 % on a 3 cm step). This is a
  different measure from heat: brief peaks at the limit, not sustained load.
- **At the edges it is weaker than pipe1.** No sweep produced a fall except sideways pushes beyond the requested
  range, as for pipe1. At 0.25 m/s both break at the same points (60 ms delay, 3 cm step, 8° ramp, 1.25 m/s push), but
  run2b slows more there: on a 3 cm step to 0.114 m/s (-54 %, pipe1 -39 %), on an 8° ramp to 0.162 m/s (-34 %, pipe1
  -21 %). At 0.15 m/s run2b also breaks at 60 ms delay (0.104 m/s, -29 %), where pipe1 does not (-7 %). Pushed to the
  left at 1.25 m/s (free-body; impulse 15.9 N·s), 7 of 8 episodes fell (pipe1 1 of 8); to the right 4 of 8 for both.
- **On steps, ramps and slippery floors it holds a straight line much worse than pipe1.** Mean sideways drift over
  18 s, against 0.08 m at the nominal point (pipe1 0.06 m): 1.13-1.15 m on 2-3 cm steps (pipe1 0.08-0.12), 0.66-0.87 m
  on 5-8° ramps (pipe1 0.13-0.16), 0.32-0.78 m at friction 0.3-0.15 (pipe1 0.12-0.05), 0.35 m at mass x0.8, 0.21-0.31 m
  at kp x0.6-0.8, 0.27 m at 15 ms delay; on the unseen foot 0.36 m mean, 0.77 m worst, with heading changes up to 12°.
  It drifts less than pipe1 at 21 of the 68 sweep points, by 0.01-0.09 m except at 40 and 60 ms delay at 0.25 m/s
  (0.26 and 0.28 m less); section 8 lists them. The sweep's break rule (a fall or a speed drop over 20 %) does not count
  drift, so the sweep summary's headline lists friction, mass and kp as clean; for straightness they are not.
- **Still asymmetric, and the slow gait is irregular.** The right-fore and left-hind diagonal carries 59 % of the
  weight and stands longer (duty factors 0.60 and 0.63 against 0.46 and 0.44), as in pipe1. At 0.15 m/s the gait is
  not classed as a trot and up to 51 % of one foot's strides are irregular.
- **What it means:** on flat ground at the trained conditions, in simulation, this policy walks cooler and more
  efficiently than the first; away from them it slows more and wanders sideways. The proposals at the end (terrain,
  lower friction with a cost on drift, symmetry, slow walking) follow from that. On hardware nothing here is a verdict:
  one policy, and the model-based walk (task 5, D10) is still to be measured.

## Flags (from the plan; reported, not pass or fail)

| Flag | Gait report cases (0.25 target, 0.15, nominal, other foot) | Robustness sweep (8 episodes per point, noise on, no randomisation) |
|---|---|---|
| Any joint's RMS / rating above 1.0 | **No.** Worst 0.72 / 0.65 / 0.71 / 0.72 | Yes, outside the trained ranges only: delay 40-60 ms (1.12-1.30), mass x1.6 (1.06), kp x1.5 (1.08), step 2-3 cm (1.17-1.40) |
| Any joint at its nominal limit more than 1 % of steps | **No.** Worst 0.5 % in one episode (left hind knee, randomised); 0.9 % (right fore pitch) on the nominal model, 0.8-1.0 % per episode | **Yes**, at 36 of the 68 distinct points with drive figures (8 episodes per point pooled; the two 1.5 m/s push points, where every episode fell, have none). Right fore pitch: friction 1.0-1.2 (1.1-1.3 %), payload 0.5-2 kg (1.0 %), kp x1.2 (1.6 %), delay 15 ms (1.4 %), 1 cm step (1.2 %), 2-3 cm steps (6.0-13.0 %), pushes of -1.25 to -0.5 and +0.75 to +1.0 m/s (1.0-1.3 %). Hind pitch: delay 20-60 ms (1.8-11.7 %), mass x1.2-1.6 (1.1-4.5 %), payload 3 kg (1.5 %), kp x1.5 (6.1 %), ramps 3-8° (2.3-2.8 %). Right hind knee: friction 0.15-0.3 (1.2-2.2 %), mass x0.8 (1.0 %). The nominal point: 0.9 %. pipe1: 3.7-22.5 % at every point where the nominal limit can be reached (not at the reduced force limits) |
| Any foot's slip ratio above 0.25 | **No.** Worst 0.09 | Yes, at the edges: friction 0.15 (0.30), delay 60 ms (0.35; 0.69 at 0.15 m/s), delay 40 ms at 0.15 m/s (0.35), 3 cm step (0.89), 8° ramp (0.27) |
| CoM outside the step-contact support polygon more than 5 % of the time | **No.** 0.2 / 0.0 / 0.2 / 0.3 % | Not measured by the sweep |
| Any fall in the requested robustness ranges | **No** | **No.** Falls only at pushes of 1.25 and 1.5 m/s free-body, beyond the requested ±0.5 m/s (added to find the edge) |

The time at the nominal torque limit is lower by definition whenever domain randomisation is on: each episode's force
limit is drawn at 0.85-1.0 of nominal, so a drive saturating at its episode's limit rarely reaches the nominal one. That is
why the randomised gait-report cases show 0.2 % while the unrandomised nominal case and the sweep show about 1 %, and
why the training evaluations (randomised) show 0 % throughout. The nominal case and the sweep are the better guide.

## 1. Drive heat (RMS torque over 18 s / continuous rating, 0.25 m/s, randomised, mean of 5 episodes (worst))

| Joint | pipe1 | run2b |
|---|---|---|
| Fore roll, left / right | 0.47 (0.52) / 0.59 (0.65) | 0.26 (0.27) / 0.22 (0.23) |
| Fore pitch, left / right | 1.02 (1.09) / 1.02 (1.11) | 0.52 (0.59) / 0.66 (0.72) |
| Front elbow (X6), left / right | 0.94 (0.97) / 0.96 (0.99) | 0.61 (0.64) / 0.58 (0.59) |
| Hind roll, left / right | 0.71 (0.78) / 0.58 (0.63) | 0.32 (0.34) / 0.31 (0.32) |
| Hind pitch, left / right | 0.83 (0.93) / 0.66 (0.77) | 0.56 (0.62) / 0.61 (0.63) |
| Hind knee (X4), left / right | **1.10 (1.15)** / 0.92 (0.94) | **0.72 (0.77)** / 0.62 (0.65) |

Copied from the two gait reports' section d. Continuous ratings at the joint: X4 roll and pitch 7.5 N·m, hind knee
6.75 N·m, X6 elbow 10.8 N·m. The hind knee's margin in this gait is 28 % (mean) or 23 % (worst episode). In the other
cases the worst joint is 0.65 (0.15 m/s), 0.71 (nominal model) and 0.72 (other foot, right fore pitch) as means of
episodes; in the worst single episode 0.69, 0.72 and **0.86** (other foot, right fore pitch), the closest any joint in
the gait report comes to 1.0. Time above the maker's rated torque: worst joint 5.6 % of steps (right hind pitch),
against 22.8 % (pipe1's left hind knee). The front elbows still reach their torque limit briefly in both runs: their
per-episode peaks equal each seed's randomised force limit (32.4-37.1 N·m; 37.8 N·m, the model's limit, on the nominal
model), so the peak torque is set by the limit, not by the gait, and does not distinguish the two policies. Peak speeds
are lower (worst 65 rpm against 79). RMS over 18 s stands in for steady walking; it is not a thermal simulation.

In training, the worst joint passed 1.0 between about 0.5 M and 1.5 M steps (highest 1.15, hind knee, at 1.5 M), while
the heat penalty was still switched off or ramping (it starts at 1 M and is at full strength at 3 M). It was under 1.0
from 1.75 M, and in the last million steps it varied between 0.68 and 0.73 with no clear trend.

## 2. Torque limits

At 0.25 m/s, randomised: only the left hind knee reaches its nominal limit (0.2 % of steps, 0.5 % in the worst
episode). Nominal model, noise off: right fore pitch 0.9 % (pipe1: left fore pitch 7.9 %). Sweep: see the flags table.
Each run's limit is the same model limit (21 N·m pitch and roll, 37.8 N·m elbow, 18.9 N·m knee at the joint).

## 3. Gait

| | pipe1 | run2b |
|---|---|---|
| Class (0.25 m/s) | other (nearest trot, 0.13 cycles off; 4 of 5 episodes "other") | **trot** in all 5 episodes (largest phase error 0.07 cycles) |
| Stride frequency, length | 2.39 Hz, 10.0 cm | **1.22 Hz, 20.0 cm** |
| Duty factor LF / RF / LH / RH | 0.44 / 0.60 / 0.66 / 0.48 | 0.46 / 0.60 / 0.63 / 0.44 |
| Feet down: diagonal pair / three / four | 82 / 13 / 3 % | 89 / 9 / 2 % |
| Symmetry index, duty factor: fore / hind / sides (0 = symmetric) | 32 / 32 / 1 | 27 / 35 / 4 |
| Diagonal timing: second pair follows the first after | 0.42 of a stride | 0.42 of a stride |
| 0.15 m/s | other; irregular strides up to 27 % | other; irregular strides up to 51 % (worst foot) |

The asymmetry is the same kind as pipe1's: one diagonal pair (right fore + left hind) stands about 0.6 of the stride and
carries more load; the other about 0.45. Both policies show it, on different feet and rewards, so it may come from the
model (for example the mass layout) as much as the training; that is not tested here.

## 4. Weight and balance (0.25 m/s, randomised)

| | pipe1 | run2b |
|---|---|---|
| Load share LF / RF / LH / RH | 21 / 34 / 27 / 18 % | 23 / 32 / 27 / 18 % |
| Front / rear | 56 / 44 % | 55 / 45 % |
| Peak foot load, worst episode, × body weight (LF / RF / LH / RH) | 3.07 / 3.33 / 2.13 / 2.11 | 2.54 / 2.20 / 1.90 / 2.19 |
| CoM outside the step-contact support polygon | 1.6 % | **0.2 %** |
| CoM margin, median | 34 mm | 40 mm |
| Centre of pressure within 5 mm of the polygon's edge | 7 % | 0 % |

## 5. Feet (0.25 m/s, randomised)

| | pipe1 | run2b |
|---|---|---|
| Slip ratio, mean of episodes (LF / RF / LH / RH) | 0.24 / 0.23 / 0.26 / 0.23 | 0.06 / 0.05 / 0.09 / 0.07 |
| Stance slip speed, 95th percentile, worst foot | 0.59 m/s | 0.24 m/s |
| Swing clearance, median, fore / hind | 22-26 / 11-13 mm | 30-31 / 13-16 mm |
| Swings under 5 mm (hind) | 2-4 % | 0 % |
| Touchdown speed, median, fore / hind | 0.68 / 0.18-0.25 m/s | 0.49-0.51 / 0.25-0.39 m/s |
| Peak load rate, median, fore / hind (body weights per s) | 286-349 / 157-163 | 217-236 / 116-140 |

The roll ankles stay within -3.7° to +5.1° of their ±20° range.

## 6. Power and run time (0.25 m/s, randomised; estimates)

| | pipe1 | run2b |
|---|---|---|
| Joint mechanical power, sum of abs(torque × speed) | 42 W | 22 W |
| Leg electrical estimate, shared bus / per drive | 106 / 123 W | 40 / 50 W |
| Walking time on one battery set, 30 W hotel load (bus / per drive) | 52 / 46 min | 101 / 88 min |
| Same, per drive, 25 / 40 W hotel load | (not compared) | 94 / 78 min |

Copper loss dominates the electrical estimate (38 of 40-50 W for run2b), so it falls with the RMS torques. The loss
model is simple (gearbox, iron and driver losses not modelled); the hotel load is an assumption.

## 7. Head steadiness (0.25 m/s, randomised)

| | pipe1 | run2b |
|---|---|---|
| Head bob / sway (1-99 % range) | 42 / 28 mm | 42 / 29 mm |
| Head acceleration RMS to 20 Hz | 11.1 m/s² | 5.1 m/s² |
| of which at or below 5 Hz (visible bob) / 5-20 Hz (step shake) | 5.9 / 9.4 m/s² | 2.7 / 4.4 m/s² |
| Body roll / pitch SD | 0.8° / 1.0° | 1.4° / 0.9° |
| Body height range | 17 mm | 28 mm |

The head moves through the same distance but half as abruptly; the body rolls more with the longer stride. The
largest spectral peak of the head's height is at 1.7 Hz, neither the stride frequency (1.22 Hz) nor twice it; not
investigated.

## 8. Robustness (one factor at a time, 0.25 m/s unless stated; speed, mean of 8 episodes)

"Break": the first point outward from nominal with a fall or a speed more than 20 % below the nominal point's.

| Sweep (trained range) | pipe1: break | run2b: break | Notes |
|---|---|---|---|
| Delay at 0.25 m/s (0-10 ms) | 60 ms: 0.177 m/s, -26 % | 60 ms: 0.171 m/s, -31 % | run2b equal or faster up to 40 ms |
| Delay at 0.15 m/s | none to 60 ms (0.134 m/s, -7 %) | 60 ms: 0.104 m/s, -29 % | run2b heat 1.25-1.30 at 40-60 ms |
| Friction (0.6-1.0) | none, 0.15-1.2 | none, 0.15-1.2 | run2b slows to 0.223 m/s at 0.15 (-9 %; pipe1 0.242) and **drifts sideways 0.78 m mean, 0.99 m worst** (pipe1 0.05, 0.15 m); see the drift table below |
| Mass, every body (x0.9-1.1) | none, x0.8-1.6 | none, x0.8-1.6 | heat at x1.6: 1.06 against 1.64 |
| Payload on the chassis | none, to +3 kg | none, to +3 kg | heat at +3 kg: 0.91 against 1.36 |
| Force limit (x0.85-1.0) | none, x0.4-1 | none, x0.4-1 | run2b keeps 0.242 m/s at x0.4 (pipe1 0.225) |
| Position gain kp (x0.9-1.1) | none, x0.4-1.5 | none, x0.4-1.5 | run2b slows to 0.205 m/s at x0.4 (-17 %; pipe1 -7 %) |
| Sideways push (none) | 1.25 m/s: right 4/8, left 1/8 fell | 1.25 m/s: right 4/8, **left 7/8** fell | both 8/8 at 1.5 m/s and 0/8 at 1.0 m/s (12.7 N·s) |
| Step up (flat) | 3 cm: 0.145 m/s, -39 % | 3 cm: 0.114 m/s, -54 % | at 2 cm run2b -18 % (pipe1 -5 %), heat 1.17 |
| Ramp up (flat) | 8°: 0.188 m/s, -21 % | 8°: 0.162 m/s, -34 % | at 5° run2b -12 %, pipe1 -13 % |

Speeds and falls are from the two `sweep_*.json.gz` files, point by point (same seeds, same points). Push sides are as
the sweep tool names them; the run2b sweep's commit message (601b002b) has the two sides swapped, and this table and
the tool's summary are the correct ones. No sweep included terrain or pushes in training, so the step, ramp and push
edges are of policies that have never seen them. One possible reason run2b gives way sooner on steps and ramps is that
its torque and heat penalties make it reluctant to use the extra torque these need; that is a guess, not tested.

**Sideways drift** (mean of the absolute sideways displacement over 18 s, 8 episodes; worst episode in brackets; from
the same files). The sweep's break rule does not use it.

| Point | pipe1 | run2b |
|---|---|---|
| Nominal (10 ms, friction 0.85, 0.25 m/s) | 0.06 (0.18) m | 0.08 (0.16) m |
| Friction 0.15 / 0.2 / 0.3 | 0.05 / 0.07 / 0.12 m | **0.78 (0.99) / 0.56 / 0.32 m** |
| Step 2 cm / 3 cm | 0.08 / 0.12 m | **1.13 (1.26) / 1.15 (1.33) m** |
| Ramp 3° / 5° / 8° | 0.19 / 0.13 / 0.16 m | 0.22 / **0.66 / 0.87 (0.94) m** |
| Mass x0.8 | 0.06 m | 0.35 m |
| kp x0.6 / x0.8 | 0.08 / 0.06 m | 0.31 / 0.21 m |
| Delay 15 ms | 0.08 m | 0.27 m |
| Delay 40 / 60 ms | 0.47 / 0.36 m | 0.22 / 0.08 m |
| Delay 0-15 ms at 0.15 m/s | 0.04 m | 0.06-0.12 m |

Over the 68 points with figures, run2b drifts more than pipe1 (by more than 5 mm) at 37, less at 21 and within 5 mm at
10. The largest gaps against run2b are on 2-3 cm steps (+1.04 m), friction 0.15-0.2 (+0.49 to +0.73 m) and ramps of 5-8°
(+0.53 to +0.70 m); the largest in its favour are at 40 and 60 ms delay at 0.25 m/s (-0.26 and -0.28 m). Every other
gap is 0.3 m or less. The 21 points where it drifts less: delays of 20, 40 and 60 ms at 0.25 m/s, force limits x0.4-0.7, payload 0.5-2 kg, kp x1.2-1.5, friction 0.6, 0.8 and 1.2, mass x1.1-1.2, and pushes of 0.4-1.25 m/s to the right. Drift of 1.1-1.3 m in
18 s would take the robot out of a 1 m-wide lane started at its centre.

## 9. The other foot (PR #669's pitch-only ankle, never seen in training)

0.244 m/s, no falls, a trot at 1.25 Hz and 19 cm; 43-54 W electrical; CoM outside the support polygon 0.3 % (one
episode 1.2 %). Speed and gait hold, but it turns and drifts more than on its own foot: sideways drift per episode
-0.04, -0.77, -0.40, -0.25 and -0.34 m (worst 0.77 m, against 0.23 m on its own foot and pipe1's 0.43 m on its unseen
foot), heading change up to 12° (own foot 4°); worst joint 0.72 of its rating as a mean, 0.86 in the worst episode
(right fore pitch). Five episodes: it walks on the old foot without falling, but not as well. F11's crawl results,
where PR #669's foot fails, concern a slow scripted crawl, not a trot.

## What this does not show

- One policy per run and one training seed: a verdict needs two seeds or two controllers.
- Simulation of a model whose masses are estimates; contact is MuJoCo's soft contact. RMS torque over 18 s stands in for
  heat; there is no thermal model. Electrical figures come from a simple loss model; the hotel load is assumed.
- The policy observes the body's true velocity (`obs_privileged_lin_vel`), which the robot would have to estimate.
- Training was on a flat floor with no pushes; the terrain and push results are generalisation, not trained skill.
- Resume on Colab is still untested.

## Changes after a result

None to the measures or the flags. The gait report's cases were switched to the run's own model before the analysis
(committed with the plan, 97cbf038). Added while writing, from the tools' existing output and not in the plan's list:
the sideways drift (section 8), which the sweep records but its break rule does not use.

Revised after the independent review (first version 6d7526b9; corrections only, no tool re-run):

- The first version called the front pitch drives' peak torque "the tightest margin". The sweep shows the drive at its
  limit depends on the condition (hind pitch under load, delay and slope; right hind knee on low friction), and time at
  the limit is a different measure from heat. Rewritten, and the flags table now lists every point over 1 %.
- The first version reported drift only for low friction and misquoted pipe1's figure (0.05-0.08 m; it is
  0.05-0.12 m). Drift is now reported across the sweep, where it is larger still on steps and ramps.
- Wording that went beyond the records was removed: that the reward changes "did what they were for" (reward, foot,
  training length, schedules and phase clock all differ); that F12 is "not reproduced"; that this is "a usable walking
  policy"; and in section 9 that it walks on the other foot "almost exactly as on its own".
- The first re-check found the drift comparison still too broad ("less only at long delays"; run2b drifts less at 21
  of 68 points). The second found the list of those 21 points loose (it missed 20 ms at 0.25 m/s, and "friction
  0.6-1.2", "payload" and "pushes to the right" each included points that go the other way) and "the large differences
  are all on its side" contradicted by the 40-60 ms delay points. The list is now generated from the sweep files and
  the counts are stated here and in F14; RUNS.md was narrowed to steps, ramps and slippery floors.
- Section 1's elbow-peak sentence, the other foot's worst episode, "same points" (true at 0.25 m/s only), "under-counted"
  (lower by definition) and pipe1's 3.7-22.5 % were corrected.

## Proposals for the next training run (for the owner; nothing started)

1. **Terrain and pushes in training:** random small steps (to about 3 cm), ramps (to about 8°) and sideways pushes.
2. **Slippery floors:** widen the friction range down to about 0.3 and give sideways drift and heading error a cost.
3. **Symmetry and slow walking:** a mirrored-pair symmetry measure in the evaluations, and more training time at low
   commands.
4. **A second seed** of the same setup, so the next comparison can be a verdict.
5. **Estimated velocity instead of the simulator's**, when the controller moves toward hardware.
6. **The stop and resume test** on Colab, still not exercised.
