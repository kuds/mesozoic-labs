# Plan: the robustness sweep of the model-based trot (written before it is run)

19 Nov 2026. Owner's choice after the walk's judged run (`REPORT_WALK60.md`). Simulation only. This sweep is
**reported, not judged**, like the learned policies' sweeps (`rl/robustness/`). Any change to it after a result
will be labelled "changed after a result".

## What is run

- **Controller:** `walk/walk60.py`, `TrotController` with `WalkParams()` defaults. These are frozen and unchanged
  since the judged run.
- **Plant:** `walk60.plant_config()`: the training env on the roll-ankle model, 100 Hz control, action filter off.
  The sweep turns randomisation off and keeps observation noise on, as for the learned policies.
- **Sweep:** `rl/robustness/sweep.py`'s own points, physics changes, push, terrain, logging, summaries, break rule
  and charts. The only change is the source of the actions: `run_episode` gains an optional `controller`
  (unused for policies, so their sweeps are unchanged).
- **Driver:** `walk/sweep_walk60.py` builds the controller in each worker and passes it in.
- **Points:** the same families and values as run2b's and pipe1's sweeps:
  - delay at 0.25 and 0.15 m/s;
  - friction;
  - mass and payload;
  - force limit and kp;
  - sideways push;
  - step up and ramp up.
- **Episodes:** 8 per point, seeds 200000-200007, 20 s, measured from 2 s.
- **Not run:** the learned-policy reproduction check (it compares with a policy's analysis).
- **Output:** `walk/results/sweep_walk60/`.

## How it is read

- **Break:** the first point outward from nominal with a fall, or a speed more than 20 % below the nominal point's.
  This is the sweep's own rule.
- **Comparison:** set against run2b's breaks (`rl/robustness/results/run2b/`) and pipe1's.
- **Also reported:** sideways drift, heat and time at the torque limit, set against run2b. The walk's drift follows
  its simulated gyro drift, as in the judged run.
- **Limits of the comparison:**
  - The two controllers differ in control rate, action range, velocity input and gait.
  - The comparison is between controllers as built, not a verdict on either.
  - The learned policies were trained with randomisation in ranges the sweep shades as "training ranges". The walk
    was developed and judged with the same randomisation, on friction 0.4 and 1.0.
