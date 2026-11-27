# Plan: the first steering run, on the 128 mm foreleg, with the heat margin (D20, D25, D26)

26 Nov 2026. A plan, not a result. Simulation only. The pass and fail rules come in a separate pre-registration,
committed and independently reviewed before any judged run.

## Why

- **D20:** the robot follows speed and heading commands that can change at any time, including turning on the spot.
  The command interface, its reward and its measures are built and reviewed (`PLAN_STEERING_AND_TELEMETRY.md`,
  `steer_eval.py`). No steering policy has been trained yet.
- **D25:** the build model is now `sim/generated/trike60_C_closed_capsule_fore128.xml`.
- **D26:** the foreleg comparison found 128 mm's worst drive at 0.83-0.95 of its continuous rating at 0.25 m/s on flat
  ground, and over the rating on long climbs, with the whole robot 10 % heavier, at servo delays of 20 ms or more,
  and in a demo with no standing breaks (`analysis/REPORT_FORELEG_D22.md`, F20). The steering run is to carry:
  - a stronger heat cost;
  - training on longer climbs, whole-body mass changes and wider delays;
  - a heat rule that must pass.

## The owner's choices (26 Nov 2026)

- **D28, terrain:** flat for now. Steering trains on flat ground with pushes and the slippery-floor end. The owner
  added the same day that the 60 cm robot will be demoed on flat ground only; uneven terrain may come later but is not
  important now. So this run's rules judge flat-ground walking, turning and heat. Steps, ramps and the longer climbs
  D26 named are measured and reported, not trained or judged.
- **D29, heat cost:** price the hottest drive. Keep today's average, and add a cost on the hottest drive's running
  heat above a margin.
- **D30, runs:** two dev runs of about 3 M steps, then two judged seeds of 10 M steps.

## What has to be built first

1. **Terrain:** none in this run (D28). The training config turns it off; pushes and the slippery-floor end stay.
2. **A heat cost that sees the hottest drive (D29).** Today's cost (`w_heat` 0.3) is the mean of (torque / continuous
   rating)² over all 12 drives. One drive at its rating therefore costs a twelfth of what it would alone, and that
   one drive is what fails F2.
3. **The wider training ranges:**
   - a whole-body mass factor drawn per episode, on top of today's per-body draws;
   - servo delays to 20 ms;
   - a payload on the chassis.

   Longer climbs wait for terrain (D28).
4. **A steering preset** (`configs/`), the notebook's menu entry, tests, and the regression digest kept for the
   straight task.

Each piece has its own tests and its own independent review before any run relies on it.

**Status, 27 Nov 2026:** built (925550e0), independently reviewed, fixed (09da2491), re-checked independently and
fixed again before any run (`rl/README.md`, "The steering run's training additions"):
- the whole-robot factors had reset every episode to the model's spring pose (no leg noise, soles at 0 mm): fixed;
- the evaluations and analysis tools had inherited the whole-robot draws, so the target case would have measured a
  robot about 4 % heavier on average: they now draw the per-body factors only, as for F20, and the heavy robots
  are the sweep's stress points;
- the heat average: the first version started at 0 at each reset (it priced 0.85-1.0 of rating at 38-61 %); the
  second, bias-corrected from 2 s, front-loaded the cost onto the first second after 2 s (the re-check). It now
  starts at 2 s from the margin, which weights every step alike and, on gaits well over the margin, prices
  0.80-0.82 of F2's own price (more than F2 near or under the margin: 1.26 on the coolest group); chosen
  from seven candidates measured on the D22 policies (`rl/analysis/steer_heat_average.py`);
- the dev-run weights are 1 and 2. The README's rule (a gait with exploration noise earns 0.3 per step more than
  standing) is applied at the noise when the cost reaches full weight, 3 M steps: 0.123-0.128 in the D22 runs on
  the same schedules (their train_health logs; the straight task with terrain, 0-10 ms delays and no heat-peak
  cost, so a guide only). There weight 1 keeps +0.52-0.57, weight 2 is on the line (+0.27-0.31) and weight 3 does
  not (+0.01-0.09). Earlier picks, both before any run: 1 and 3, then 0.5 and 1
  on an assumed noise. Each dev run's own noise at 1-3 M is checked before the judged weight is fixed;
- the notebook refuses a steer1 preset on any model but the 128 mm one.

## The runs

- **Dev runs** (proposed): two trainings on Colab, heat weights 1 and 2 (`steer1_dev_heat1`, `steer1_dev_heat2`),
  the same seed. They check that the steering task learns on the 128 mm model and set the heat cost's weight before
  the rules are written, by a rule fixed before the runs (`runs/trike60-steer1-dev_STEPS.md`, section 5). They are
  not judged. **Length (D31, 27 Nov 2026): the 10 M schedule, read at 5 M.** The
  review found that at about 3 M steps the heat cost reaches full weight only at the last step: train.py's late ramp
  runs from 1 M to 3 M and the push and friction ramp from 1 M to 4 M, and the learning rate follows TOTAL_STEPS, so
  a 3 M run also anneals faster than a judged 10 M run. With TOTAL_STEPS 10 M, the 5 M save has had 2 M steps at full
  heat weight and 1 M at full pushes and friction, on a judged run's schedule. Steps: `runs/trike60-steer1-dev_STEPS.md`.
- **Judged runs:** the pre-registered seeds at 10 M steps each, run as run 3 was. Each takes about 4.5-5 h on Colab
  High-RAM CPU (the 128 mm foreleg runs took 4.5-5.1 h; all eight foreleg runs 3.6-5.1 h).
- **Measurement:** the gait report, the sweep and the foot loads as for D22, plus `steer_eval.py`'s courses and the
  demo course with turns. Randomised measurements draw the per-body factors only, so the target case is comparable
  with F20; the whole robot x1.1, the payloads and the 20 ms delay are the sweep's points.

## Rules to pre-register (a sketch, to be fixed after the dev runs)

- **Walking:** F1 and F2 as before, on a straight request.
- **Steering:** speed and turn-rate tracking error, heading error a set time after each change, turning on the spot.
- **Heat (D26):**
  - the worst drive under a margin below its rating (for example 0.85) in the target case;
  - under its rating at a set of stress points: the whole robot 10 % heavier, 20 ms delay, and the demo course with
    turns. A long climb is measured and reported, not judged, since this run does not train on terrain (D28).
- **Seeds:** two judged seeds (D30).

## Not in this run

- Real-robot sensing (no simulator-only velocity): the run after (D20).
- Measured masses and drives: the bench.

## Cost and time

- **Session work:** building and reviewing the four pieces above is about 1-2 days.
- **Colab:** two dev runs, read at 5 M steps (2.3-2.6 h in at the 128 mm foreleg runs' 542-617 steps per second;
  4.5-5.1 h if left to finish), then the judged runs at 4.5-5.1 h each, run in parallel as before.
