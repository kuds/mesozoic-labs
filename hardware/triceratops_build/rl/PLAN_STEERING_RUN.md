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

## The runs

- **Dev runs** (proposed): two short trainings, about 3 M steps each, on Colab. They check that the steering task
  learns on the 128 mm model and set the heat cost's weight before the rules are written. They are not judged.
- **Judged runs:** the pre-registered seeds at 10 M steps each, run as run 3 was. Each takes about 3-5 h on Colab
  High-RAM CPU (the foreleg runs took 3.6-5.1 h).
- **Measurement:** the gait report, the sweep and the foot loads as for D22, plus `steer_eval.py`'s courses and the
  demo course with turns.

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
- **Colab:** two dev runs of about 1.5 h each, then the judged runs at 3-5 h each, run in parallel as before.
