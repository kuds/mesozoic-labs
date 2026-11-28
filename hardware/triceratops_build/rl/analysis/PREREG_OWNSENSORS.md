# Pre-registration: one run on the robot's own sensors (D46)

27 Nov 2026. **Simulation only, except where marked desk:** L3's winding temperatures (a thermal model run on
simulated torques), the foot cells' price and lead time (a parts-list estimate) and the odds table are desk
estimates. Nothing is measured: nothing is built. Written before the run, and before any own-sensors policy exists:
nothing from it informed what follows. The measures, the lines, the readings and the follow-ups are fixed here;
anything changed after a result will be labelled "changed after a result", with its first wording kept.

**This is not a judged verdict.** No owner-chosen rule decides anything here. The run is one seed, and two seeds (or
two controllers) are needed before anything is called a result. What follows fixes in advance how the one seed will
be read: as no sign of a cost, a sign of a cost, or inconclusive. Each reading names the next step to put to the
owner.

**Disclosure.** The lines below were set after the four fore128 runs had been measured (`REPORT_FORELEG_D22.md`). Each
of L1-L7 is that measure's worst fore128 seed plus a tolerance. L8's line is one sweep grid step inside the earliest
fore128 break (a break at 40 ms, so the line is a break at 30 ms or less), and F1 and F2 are run 3's rules
(`PREREG_RUN3.md:63-64`). They are this pre-registration's own lines; the owner set no rule for this run. For heat
and falls the tolerance is borrowed from D24: the owner set it for another comparison, the D22 leg lengths' four-seed
averages against each other, as run 3's seed scatter (`DECISION_LOG.md`, D24). Here it is applied to one new seed
against the worst of the four fore128 seeds. For L1 and L5-L7 the four seeds' own standard deviation is used.

## The question (D46)

The trained policies read 11 of their 68 inputs from signals the robot will not have (DECISION_LOG.md, D46):
- `foot_load_over_weight` (4): the core build has no foot load cells (`sim/build_model60.py:12`). Four cells with
  their per-leg links cost +$254, with a 21-23 week lead (parts-list estimate of 18 Nov 2026,
  `budget/size_budget_out.txt:108`, dated in D46, `DECISION_LOG.md` (D46); desk);
- `ankle_pitch_pos` (4): the passive ankles have no encoder (the parts list has an ankle hub and no sensor there,
  `design/PARTS_60CM.md`);
- `base_lin_vel_body_PRIVILEGED` (3): the simulator's true body velocity, which no sensor gives
  (`rl/trike_env.py:127`).

The owner chose to train without them first and to decide on foot sensors afterwards (D46, 27 Nov 2026). This run
asks whether the D22 128 mm policy walks as well without the three, on the measures the robot's flat-floor demo needs.

## What is trained

- **Code:** the reviewed commit that `runs/trike60-ownsensors_STEPS.md` names, run from the training notebook at that
  commit on Colab. The tools at that commit measure it.
- **Model:** `sim/generated/trike60_C_closed_capsule_fore128.xml` (12.751 kg), the adopted design (D25, D41).
- **Setup:** the four fore128 runs' settings, with the three inputs off. Those runs are `trike60-fore128-s0` to `-s3`,
  at commit f84a6919 (`runs/trike60-d22_STEPS.md`).
  - Preset: `configs/run3_terrain_ownsensors.json`, which is `configs/run3_terrain.json` plus
    `obs_foot_loads`, `obs_ankle_angles` and `obs_privileged_lin_vel` all false. Notebook ENV_PRESET
    `run3_terrain_ownsensors`; the notebook refuses it with any model but fore128.
  - 10,000,000 steps, 8 envs, and train.py's defaults: PPO with n_steps 1024, batch 2048, 5 epochs, learning rate
    3e-4 to 3e-5, gamma 0.99, GAE 0.95, clip 0.2, entropy 0, log std -1.6, ELU networks of 256 and 256.
  - Schedules: penalty scale 0.1 to 1 over 2 M steps; late costs from 1 M to 3 M; terrain from 1 M to 4 M.
  - Evaluations every 250,000 steps.
  - A test checks that train.py with the notebook's arguments and this preset reproduces the fore128-s0 run record's
    settings exactly, except the three inputs (`tests/test_obs_own_sensors.py`,
    `test_ownsensors_preset_is_the_d22_fore128_setup_with_three_inputs_off`). The record is
    `analysis/results/d22/run_record_fore128-s0.json`.
  - The only other config keys are those added since f84a6919, at their off defaults: the steering run's heat-peak
    cost and its whole-robot mass and payload draws.
- **The environment is otherwise the D22 runs' own.** On the test's D22 rollout (fore128 model, run 3's preset,
  randomisation, noise, terrain and a push), f84a6919's `trike_env.py` and aa150f89's match bit for bit in
  observations, rewards and endings. Their infos match too, except one key added since: `r_heat_peak`, which is
  always 0 here. This was compared step by step on 27 Nov 2026.
- **Seed:** SEED 0, run as `trike60-ownsensors-s0`. With fewer inputs, the network's first layer and so all its
  initial weights differ from fore128-s0's, and the episodes part after the first actions. So this is not a paired
  comparison with seed 0. It is read against all four fore128 seeds.
- **Stop rules (the D22 runs'):**
  - A stopped session resumes under the same RUN_ID.
  - A run that cannot resume is restarted once under a new RUN_ID with the same seed. The restart is reported and
    its first files are kept.
  - If the restart also fails, nothing is read.
  - The run is never extended or shortened.
- **What is read:** the final pair (10,002,432 steps, or after a resume the last pair train.py writes, between
  10,000,000 and 10,008,184 steps), never a checkpoint chosen among others.

### The observation: 57 floats

**Kept** (the robot's own sensors):
- 12 leg joint angles and 12 joint speeds (the drives' encoders);
- gravity in the body frame and the gyro (the IMU);
- the heading error from the gyro-integrated heading estimate, which drifts (no magnetometer, `rl/trike_env.py:120`);
- the filtered and raw previous actions;
- the commanded speed.

**Removed:** exactly the 11 channels above. Their noise is still drawn, so the random stream does not shift. A test
checks two things side by side, with the same seeds and actions. First, every other channel equals the full
observation's, bit for bit. Second, every reward term and info is identical.

**What else reads those channels.**
- Rewards and endings read the simulator's state, not the observation, so all are unchanged:
  - contact, air time and slip read `data.sensordata`;
  - speeds read the chassis displacement;
  - endings read tilt, height and body contact.
- The analysis tools read the simulator too: the gait report, the sweep, `foot_loads.py`, `demo_course.py` and
  `gait_metrics.py`.
- Telemetry reads the observation only through VecNormalize, whatever its width: the periodic checks normalise the
  whole vector (`rl/telemetry/checks.py:56-58`), and the training-health log summarises its variances as min, median
  and max (`rl/telemetry/health.py:151-153`). Nothing there breaks, and no file in `rl/`, `walk/` or `sim/` indexes
  observation channels by number. But this run's `obs_var` summary covers 57 channels, not 68, so it is not compared
  with the fore128 runs' logs.
- The model-based trot (`walk/walk60.py:224`) reads `foot_load_over_weight` from the observation, to gate its stance
  legs. It needs the default, and it is not part of this run.

**The critic.** train.py uses SB3's `PPO("MlpPolicy")`, with separate policy and value networks on the same
observation (`rl/train.py:736`, `:745`).
- The build has no asymmetric actor-critic (a critic with privileged inputs), and none is built for this run. So the
  critic loses the three inputs too.
- A real robot needs only the actor to go without them. This run is therefore harder to train than a deployed
  policy requires.
- A shortfall here may come from training, not from what the actor can sense. Follow-up (c) below is for that case.

## How it is measured

Measured with the tools at the run commit: in the tree if `rl/` has not changed since that commit, otherwise
through a `TRIKE_ENV_DIR` export of it, as for the D22 runs. The tools and settings are the D22 runs'
(`PREREG_FORELEG_D22.md`, "How it is measured"):
- **The gait report** (`analytics/gait_report.py`).
  - Target case: 0.25 m/s, randomised, 10 ms delay, 5 episodes on seeds 100000-100004, 20 s, deterministic, read
    from 2 s.
  - Also its 0.15 m/s and nominal cases.
- **The robustness sweep** (`robustness/sweep.py`).
  - Nominal model, observation noise on, 8 episodes per point on seeds 200000-200007.
  - The families: delay at 0.25 and 0.15 m/s; friction; mass and payload; force limit and kp; sideways pushes both
    ways; steps and ramps.
- **The demo course** (`demo_course.py --ckpt-dir`), default loop: 90 s walking at 0.25 m/s and 30 s standing, for one
  pack.
  - Run in a 35 C room (D48), with method A: drives fitted at the 25 C rating room, then run in the warmer room
    (`analysis/thermal_ambient.py`).
  - The run commit must contain `demo_course.py --room-c`. That option is being added in the D48 batch and is
    uncommitted at aa150f89. If the run commit lacks it, the 35 C figure is taken on the same checkpoint with the first
    commit that has it, and this is disclosed.
  - The 25 C figure is reported too.
- **`analysis/foot_loads.py`** (reported only).
- **The fore128 references:** the stored outputs of the four runs.
  - `analytics/results/fore128-s*/gait_report.json`;
  - `robustness/results/fore128-s*/sweep_fore128-s*.json.gz`;
  - `analysis/results/d22/judge_d22.json`;
  - for 35 C, method A in `analysis/results/heat35/thermal_ambient_2026-11-27.json` (desk; the D48 batch's output,
    uncommitted at aa150f89).
- If the sweep has the 16-19 ms delay points by then (step sheet, part 2), they are run on this policy too, and
  reported only.

## Measures and lines (fixed now)

"Clearly worse" means beyond the line. Each of L1-L7 is the worst of the four fore128 seeds plus a tolerance: for heat
and falls D24's, borrowed (see the disclosure above), otherwise the four seeds' sample standard deviation (SD). L8's
line is one sweep grid step inside the earliest fore128 break; F1 and F2 are run 3's rules. Fields are read from the
tools' JSON, not the rounded Markdown.

| # | Measure | Field | fore128 s0 / s1 / s2 / s3 | Line: clearly worse if | Basis |
|---|---|---|---|---|---|
| F1 | Flat walking (must pass) | gait report `cases.target_0.25.summary.speed.forward_speed_mps`, `falls` | 0.235 / 0.244 / 0.242 / 0.240 m/s; 0 falls of 5 each | speed under 0.225 m/s, or any fall | `PREREG_RUN3.md:63` (written for run 3: the same setup on the 188 mm leg) |
| F2 | Drive heat (must pass) | the largest `drives.per_joint[*].rms_over_continuous.mean`, target case (H_all) | 0.828 / 0.888 / 0.931 / 0.945 | any drive at or over 1.0 of its continuous rating | `PREREG_RUN3.md:64` |
| L1 | Speed at the 0.25 m/s command | as F1 | worst 0.2348, SD 0.0039 m/s | under 0.231 m/s | worst - SD |
| L2 | Worst-drive heat margin | H_all, as F2 | worst 0.945 | over 0.980 | D24's +0.035 (DECISION_LOG.md, D24), borrowed: written for the D22 four-seed averages (run 3's seed scatter), used here for one seed against the worst fore128 seed |
| L3 | Heat in a 35 C room (D48). Desk estimate on simulated torques | demo course, hottest winding peak, 8 min time constant, method A: a first-order thermal model of each drive, in open air with no shell, run on the policy's simulated torques | 76.3 / 74.4 / 93.3 / 98.3 C (desk) | over 104.3 C | D24's 0.035, borrowed as for L2, turned into degrees: in the same thermal model a steady load of 0.85 of rating settles at 87.1 C in a 35 C room and 1.00 at 113.0 C. That is 6.0 C per 0.035 near the rating (`results/heat35/thermal_ambient_2026-11-27.md:75`, desk). The fore128 values and this conversion are the D48 batch's heat35 tables, uncommitted and not yet checked at aa150f89 (see the note below the table) |
| L4 | Falls, N | D22's N: falls at run 3's judged sweep points (2 and 3 cm steps, 5 and 8 deg ramps, friction 0.3, pushes of -1.25, -1.0, +1.0 and +1.25 m/s; 8 episodes each) plus the gait report's target, 0.15 m/s and nominal cases (5 each) | 14 / 24 / 9 / 27 | over 34 | D24's +7, borrowed as for L2 (written for the D22 four-seed averages) |
| L5 | Pushes inside the training range, both ways | sweep falls at -0.75, -0.5, -0.4, -0.3, +0.3, +0.4, +0.5 and +0.75 m/s (free-body dv; 64 episodes) | 0 / 0 / 0 / 2 | over 3 of 64 | worst + SD (1.0) |
| L6 | Heading drift | gait report target case, \|`heading_change_deg_worst`\| | 9.2 / 10.0 / 11.2 / 18.1 deg | over 22.2 deg | worst + SD (4.06) |
| L7 | Sideways drift | gait report target case, \|`sideways_drift_m`\| (mean of the 5 episodes) | 0.419 / 0.305 / 0.483 / 0.483 m | over 0.567 m | worst + SD (0.084) |
| L8 | Delay robustness | the sweep's delay families at 0.25 and 0.15 m/s; its own break rule: a fall, or a speed more than 20 % below the nominal point's | 0.25 m/s: first break at 40 ms on s0 and s3, none to 60 ms on s1 and s2; 0.15 m/s: none to 60 ms on any | a break at 30 ms or less at either command | every fore128 seed walked to 30 ms without a break at both commands |

Sources for the fore128 values:
- F1, F2 and N: `REPORT_FORELEG_D22.md:135-139` and `analysis/results/d22/judge_d22.json`.
- L3: `results/heat35/thermal_ambient_2026-11-27.md:14` (desk). These tables, and `analysis/thermal_ambient.py` that
  writes them, are the D48 batch's output: uncommitted at aa150f89, with their own independent check pending. L3's
  four values and its conversion are rechecked against them once that batch is checked and committed. If they change
  before this run is measured, L3 takes the checked values and the change is disclosed here; a change after a result
  is labelled "changed after a result".
- L5 and L8: the stored sweeps, as tabulated in `REPORT_FORELEG_D22_APPENDIX.md:182-190` (0.25 m/s) and `:196-204` (0.15 m/s) for L8, and `:269-285` for L5; the break
  rows are the two delay rows of each run's "Where it breaks" table:
  `robustness/results/fore128-s0/summary_fore128-s0.md:23-24`, `fore128-s1/summary_fore128-s1.md:22-23`,
  `fore128-s2/summary_fore128-s2.md:22-23` and `fore128-s3/summary_fore128-s3.md:23-24`.
- L6 and L7: `REPORT_FORELEG_D22_APPENDIX.md:67-69` (unrounded in each `gait_report.json`).

Pushes in the sweep are sideways only: + is to the left, - to the right. Each side is also reported on its own, at
+1.0 and +1.25 m/s against -1.0 and -1.25 m/s. Fore128's falls of 16 on the left were 6 / 15 / 8 / 11, and on the
right 8 / 9 / 1 / 16.

## Readings (fixed now; not verdicts)

- **No sign of a cost:** F1 and F2 pass, and no line is crossed. The policy walks without the three inputs as well
  as the four seeds that had them, as far as one seed can show.
- **A sign of a cost:** F1 or F2 fails, or two or more lines are crossed.
- **Inconclusive:** F1 and F2 pass and exactly one line is crossed.

A run that does better than the four seeds on a measure is reported as such. Being better crosses no line.

### What each reading means for D46's later decision (foot load cells)

One run removes three inputs, and foot load cells would restore one of them. So no single reading of this run favours
buying the cells on its own. The follow-ups are fixed here so that their reading is fixed too. Each is a proposal to
the owner, at about 5 Colab hours per run.

- **No sign of a cost:** this run gives no case for the foot load cells.
  - Next: (a) a second seed of the same run (SEED 1), because two seeds are needed before it counts.
  - The cells may still be wanted for another reason this run does not test. The model-based trot, the build's
    second controller, gates its stance legs on the foot loads (`walk/walk60.py:224`).
- **A sign of a cost:** this favours the foot-cell question but does not decide it.
  - Next: (a), and (b) a run with the foot loads restored: `obs_foot_loads` on, the other two off, SEED 0,
    otherwise this run's settings.
  - **Foot load cells are favoured** when (b) passes F1 and F2 and crosses none of the lines that this run crossed.
  - If (b) still crosses them, the cost comes from the body velocity or the ankle angles, which the cells would not
    fix. Then the candidates are (c) an asymmetric critic (a train.py change, not built), a velocity estimator, or
    ankle encoders (unpriced).
- **Inconclusive:** (a) first. Then the same reading again, taking each measure's worse value of the two seeds,
  and the follow-ups above as that reading says.

**The cells' lead time** is 21-23 weeks (parts-list estimate of 18 Nov 2026, `budget/size_budget_out.txt:108`;
desk): about five months from an order to the cells' arrival. The one-foreleg bench's B9 tests the same cell earlier
(`bench/BENCH_PLAN_FORELEG.md:43`, `:67`). This is lead-time arithmetic, not a plan, and nothing is ordered.

## What one seed can and cannot conclude (desk estimate, before the run)

This is desk arithmetic on the four fore128 seeds' values in the table above. The method, in full, so that it can be
redone from this page:
- **Each measure's chance for one new seed** to land beyond its line (L1-L4, L6 and L7, F1's speed under 0.225 m/s
  and F2's 1.0) comes from a Student's t prediction with 3 degrees of freedom. It is centred on the four seeds' mean,
  with scale SD x sqrt(1 + 1/4). The unrounded seed values are those of the tools' JSON (speed 0.234773, 0.243824,
  0.241865, 0.239869 m/s; H_all 0.827586, 0.887707, 0.930910, 0.945385; 35 C 76.3082, 74.3768, 93.2673, 98.2779 C;
  heading 9.2428, 10.0301, 11.1547, 18.1073 deg; drift 0.4187, 0.3050, 0.4828, 0.4834 m), with L1's line at 0.2309
  m/s and L6's at 22.17 deg.
- L5 and L8, counts and a category, are given a rough 0.10 each.
- **F1/L1 and F2/L2 are the same measure at two lines.** F1's speed failure lies inside L1's line, and F2's inside
  L2's. So within "F1 and F2 pass", L1 and L2 are taken at their chances given that F1 and F2 pass:
  (P(L1) - P(F1)) / (1 - P(F1)), and the same for L2 with F2.
- The other measures are treated as independent of each other and of these. The reading's three chances then follow
  from the count of lines crossed, times the chance that F1 and F2 both pass; "a sign of a cost" also takes the
  chance that F1 or F2 fails.
- F1's falls are left out: no fore128 target episode fell, 0 of 20.
- For the "worse" rows, heat (H_all, F2 and L3) and falls (L4) are shifted up by that many of their own SDs; the
  other measures are not shifted.

The script was a scratch file and is not committed. Treating F1/L1 and F2/L2 as independent too, instead of nested,
would give 0.39 / 0.34 / 0.27 in the first row. On the assumptions above:

| Truth | No sign of a cost | Inconclusive | A sign of a cost |
|---|---|---|---|
| Removing the inputs changes nothing | 0.44 | 0.33 | 0.24 |
| Heat and falls truly one SD worse (about +0.053 of rating, +8 falls) | 0.22 | 0.31 | 0.48 |
| Two SD worse | 0.04 | 0.14 | 0.82 |

- So even with no real change, one seed gives "a sign of a cost" about a quarter of the time, mostly because the
  fore128 seeds' heat already sits close to 1.0. A run with no real change fails F2 about 9 % of the time.
- A sign of a cost leads to two cheap runs, not to a purchase. Missing a real cost would only show on hardware, with
  a 21-23 week lead behind it. So the design accepts false signs rather than missing a real cost.
- These figures are rough. Heat and the 35 C figure move together, as do falls and pushes. Four seeds give a poor
  estimate of a spread.
- "No sign of a cost" does not mean the inputs are worthless. It means one seed without them did not land outside
  four seeds with them.

## Reported, not part of the reading

- Everything else the three tools give, against the four fore128 seeds:
  - the 0.15 m/s and nominal cases;
  - foot slip, gait pattern and stride, time at the torque limit, power and runtime per pack;
  - run 3's T1-T5 values;
  - every sweep point (heat at 15 and 20 ms delay: fore128 0.87-1.02 and 1.07-1.20 of rating at 0.25 m/s,
    `REPORT_FORELEG_D22_APPENDIX.md:186-187`).
- The demo course at 25 C, its standing segment, and the time to 90 C and 100 C at 35 C.
- The 35 C figure is shown beside bench B3's line, "Winding ≤ 90 °C at 35 °C" (`bench/BENCH_PLAN_FORELEG.md:61`).
  B3 was written for a measured winding on the 75 cm T20 foreleg, over a 30-minute loop with the shell on: another
  case. Two of the four fore128 seeds are already over it (93.3 and 98.3 C), so it is not a line here.
- The training curves, `eval_during_training.jsonl` and `train_health.jsonl` (action noise at 3 M steps, as the
  steering rule reads it).

## Not covered

- **Fore-aft pushes.** They are neither trained (the preset pushes sideways) nor measured (`sweep.py` pushes
  sideways only).
- **Turning.** This is the straight task, as for the D22 runs.
- **Real sensor behaviour.** The IMU's latency and its bias under foot impacts, encoder quantisation, and the drive
  model's idealisations (no joint friction, ideal position servos) are not modelled. The noise is white Gaussian
  (`rl/README.md`, observation noise).
- **The model-based trot without foot loads.**
- **An asymmetric critic.**

## Expected cost

- **Training:** 4.5-5.1 hours on Colab (8 vCPUs, High-RAM). The four fore128 runs, s0 to s3, took 5.13, 4.51, 5.07
  and 4.85 h, at 542, 617, 548 and 573 env steps per second (`analysis/results/d22/run_record_fore128-s*.json`,
  `sessions[0]`, `wall_s` and `env_steps_per_s`; simulation runs on Colab). Installs and the final test add a few
  minutes. Fewer inputs change the simulation not at all, and the policy's own cost little.
- **Measuring here:** about 20-30 minutes. Each fore128 sweep took 710-903 s on 2 workers (`episodes_run` 565,
  `wall_s`, in each `sweep_fore128-s*.json.gz`), and the gait report and the demo course take a few minutes more.
