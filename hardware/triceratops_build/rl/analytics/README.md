# Gait-quality analytics for the 60 cm Triceratops (simulation)

Tools that turn a trained walking policy, or a training run's periodic evaluations, into gait-quality numbers and
charts. **Everything here is simulation** of a model whose masses and contact are estimates, and one policy at one
checkpoint is reported, not judged (a verdict needs two seeds or two controllers). The definitions extend
`../analysis/gait_pipe1.py` (the first gait analysis, `../analysis/REPORT_GAIT_PIPE1.md`) and reuse its numbers
exactly where they overlap.

| File | What it is |
|---|---|
| `gait_report.py` | Runs a checkpoint through the episodes below, logs every 1 ms physics step, writes `results/<run>/gait_report.json` (every summary), `gait_report_episodes.json.gz` (per-episode detail, gzip), `gait_report.md` and nine charts |
| `gait_core.py` | The measures that need no simulator (debounced footfalls, steps, phase offsets, gait class, symmetry, support-polygon margins, stride resampling); unit-tested |
| `gait_charts.py`, `chart_style.py` | The report's charts; the shared style (light surface, fixed categorical order, SIMULATION tag, run name, PNG under 500 KB) |
| `training_dashboard.py` | Plots a run's `eval_during_training.jsonl` against training steps, with the penalty schedules from its `run_record.json` |
| `tests/` | `pytest hardware/triceratops_build/rl/analytics/tests` (36 tests; about 5 s) |
| `results/pipe1/` | The report and training dashboard of trike60-pipe1 (2,007,040 steps) |

## How to run

```bash
PY=python  # with mujoco 3.10, torch (CPU), stable-baselines3 2.9, gymnasium, numpy, matplotlib, pillow
# a checkpoint folder: one *_<n>_steps.zip, its *_vecnormalize_<n>_steps.pkl, and the run's run_record.json
$PY hardware/triceratops_build/rl/analytics/gait_report.py --ckpt-dir CKPT_DIR --run pipe1 --label trike60-pipe1
#   --workers 2 (default; 35 episodes in about 65 s on 2 CPU workers), --out DIR, --cache FILE (keeps the chart
#   payload) and --charts-from FILE (redraws charts and Markdown without simulating)
$PY hardware/triceratops_build/rl/analytics/training_dashboard.py \
    --run trike60-pipe1 PATH/eval_during_training.jsonl PATH/run_record.json [--run NAME2 JSONL2 RECORD2] --out DIR
```

**Loading is safe by construction:** nothing stored in a checkpoint runs. The policy weights load with
`torch.load(weights_only=True)` into a freshly built ELU 256-256 policy and the normalisation statistics through a
restricted unpickler; both loaders are imported from `../analysis/gait_pipe1.py`. Never use `PPO.load` or plain pickle
on downloaded checkpoints.

**Environment:** `../trike_env.py`, rebuilt from the run record's `settings.env_config` (model path re-pointed to
`sim/generated/`), or an exported copy of `rl/` named by `TRIKE_ENV_DIR` (as `gait_pipe1.py` reads it). For
trike60-pipe1 (trained at commit 6aa3a679, 70-float observation, 1.5 Hz phase clock) the current `trike_env.py`
(e92fd9ef and later; a different file) reproduces the committed `gait_pipe1_2026-11-19.json` **bit for bit** in all four
cases compared (speed, drift, power, every joint's RMS torque and time at limit, duty factors, slip ratios, body
angles: largest difference 2e-16), so the current file is used and the cross-check is written into every report.

## Episodes and windows

As `gait_pipe1.py`: deterministic actions (the policy mean), servo delay fixed at 10 ms, seeds 100000-100004,
domain randomisation and observation noise on (friction, masses, gains, force limits, sensor noise, heading drift),
20 s episodes, **measured over t >= 2 s** (18 s). Cases: 0.25 m/s (the museum target) and 0.15 m/s; 0.25 m/s on the
nominal model (no randomisation, no noise), whose seed 100000 at t = 6-9 s gives the time-series charts; 0.25 m/s on
the roll-ankle model `trike60_C_closed_ankleroll.xml`; and a speed sweep 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.27 m/s on
seeds 100000-100002 (0.15 and 0.25 reuse the cases' episodes). Every physics step (1 ms) is logged through the env's
`substep_hook`. Means are over episodes unless "worst" (the worst episode) or "pooled" (all strides or steps of the
case's episodes together) is stated.

## Definitions

### Contact, steps and footfalls (`footfall.png`)

- **Loaded:** the foot's touch sensor above the env's `contact_threshold_n` (2 N). In this model the touch sensor
  equals the floor-truth sole normal force (the sum of `mj_contactForce` normal components over the sole-floor
  contacts, decoded with `environments/shared/floor_contact/reduction.py`): the report checks it (agreement 100 %).
- **Env debounce (10 ms):** the env's rule (`TrikeEnv.contact_debounce_s`, `gait_metrics.debounced_touchdowns`): a foot
  counts as lifted only after 10 consecutive unloaded steps, and only after its first landing since reset; a touchdown
  is the first loaded step after a lift. Touchdowns per second on this rule equal the training evaluation's count
  (checked per episode, and in a test).
- **Steps:** env-debounced contact, then the shared library's `environments/shared/gait/events.debounce` with a 50 ms
  window (unloaded gaps under 50 ms filled, then loaded blips under 50 ms dropped). Why: in pipe1's 0.25 m/s
  episodes the env-debounced contact still breaks for 10-50 ms inside a stance with the sole lifting a median under
  1.5 mm (at most 2.2 mm; a contact-solver rebound), while 95 % of the gaps of 50 ms or more (the swings) lift it 7 mm
  or more, and only 15 gaps in 5 episodes fall between 20 and 80 ms. At 0.15 m/s and on the roll-ankle model the hind
  feet also make 20-80 ms shuffles (lifts of about 1-3 mm; 39 per episode at 0.15 m/s and 27 on the roll-ankle
  model, against 3 at 0.25 m/s), so the split is less clean there; the
  JSON's `contact_gaps` table gives the counts and lifts per foot and case. The
  T. rex library's own window is 20 ms per control step; this tool works per 1 ms physics step and chooses 50 ms from
  that gap distribution. Strides, phases, duty factor, stance and swing, clearance and touchdown impacts use steps.
- **Duty factor:** fraction of the window a foot is in stance (steps; the env-debounced and raw figures are reported
  too, raw being `gait_pipe1.py`'s).
- **Stride period / frequency:** median interval between a foot's step touchdowns / its inverse (Hz); all feet pooled
  for the case figure. **Irregular strides:** periods more than 25 % from the median.
- **Stride length:** chassis travel along the reset heading between consecutive left-fore touchdowns (m).
- **Phase offset after LF:** for each LF stride, the first touchdown of the other foot in it, as a fraction of that
  stride; circular mean (cycles, 0-1) and resultant length R (1 = perfectly locked) over the strides.
- **Gait class:** the nearest of the ideal offsets (RF, LH, RH after LF): trot (0.5, 0.5, 0), pace (0.5, 0, 0.5),
  bound (0, 0.5, 0.5), walk, lateral sequence (0.5, 0.75, 0.25) or diagonal sequence (0.5, 0.25, 0.75), pronk (0, 0, 0),
  by the largest circular error over the three feet; the class only if that error is at most 0.125 cycles (half the
  spacing between neighbouring patterns, so classes cannot overlap) and every foot is locked (R >= 0.5), else
  `other` (with the nearest pattern named). Pronk reports as `other (pronk)`.
- **Diagonal timing:** Hildebrand's limb phase per side (fore touchdown after the same side's hind touchdown, as a
  fraction of the stride: 0.5 trot, 0.25 lateral-sequence walk, 0 pace); each fore foot's lag after its diagonal hind
  partner (0 in a perfect trot); the interval from the LF+RH pair to the RF+LH pair (0.5 when even).
- **Feet down:** fractions of the window with 4, 3, 2 (diagonal, same side, front, hind), 1 and 0 feet in stance.
- **Symmetry index** (Robinson): 100 |L - R| / ((L + R) / 2), %, 0 symmetric; for duty factor, stance duration,
  stance length (chassis travel during a stance) and mean load, for the fore pair, the hind pair and the two sides.

### Weight distribution (`weight_distribution.png`)

- **Vertical load / body weight:** floor-truth sole normal force (the floor is level) divided by the episode's
  weight (its randomised mass x g). **Over a stride:** each LF-to-LF stride resampled onto 0-100 %, pooled over the
  case's episodes; line = mean, band = 10th-90th percentile.
- **Share:** a foot's mean load over the window / the four feet's; front = LF + RF, left = LF + LH.
- **Peak foot load:** the largest 1 ms value in any episode (x body weight), and the mean of the episodes' 99th
  percentiles (less sensitive to single-step contact spikes).

### Balance (`balance.png`)

- **Support polygon:** convex hull (shared `stance_track.convex_hull`) of the four bottom corners of each sole (box
  88 x 64 mm) whose floor-truth load is above 2 N, at every 1 ms step.
- **Centre of pressure (CoP):** the load-weighted mean of those soles' contact points.
- **Centre of mass (CoM):** the whole robot's (`subtree_com` of the free body), projected on the floor.
- **Margin:** shared `stance_track.signed_margin`: + inside, the distance to the nearest edge; outside, minus the
  distance past the most violated edge line (exact except beyond a corner, where it is a lower bound). A margin below
  -1 um counts as outside. **Reported:** the fraction of the window with the CoM outside (a trot carries its CoM
  outside the diagonal strip part of the time: by design a dynamic gait), split by the number of feet loaded; the
  median CoM margin; the CoP within 5 mm of an edge (the foot pivots about that edge); the CoP outside (cannot happen
  with this construction: a check); the fraction with no foot loaded.

### Drives (`drive_envelopes.png`, `drive_loads.png`)

All at the joint, every 1 ms step of the window. The flexion joints work through a 1:1 linkage at 0.90 efficiency
(`sim/data/drives.json`), so their maker figures are taken x 0.90:

| Group (drive) | Model limit = 0.70 x peak | Maker peak | Maker rated | Continuous (thermal equilibrium) | Speed allowance |
|---|---|---|---|---|---|
| roll, pitch (X4-36-L) | 21 N m | 30 | 10.5 | 7.5 | 83 rpm |
| front elbow (X6-60) | 37.8 | 54 | 18 | 10.8 | 108 rpm |
| hind knee (X4-36-L) | 18.9 | 27 | 9.45 | 6.75 | 83 rpm |

- **Envelope:** 2-D histogram (log count per cell) of joint torque against joint speed, all joints of a group and all
  episodes pooled, with the five reference lines.
- **RMS / continuous:** RMS torque over the window / the continuous rating (heat stand-in; not a thermal model).
- **At the nominal limit:** fraction of steps with |torque| >= 99 % of the model's limit (`gait_pipe1.py`'s
  definition; the randomised episode limit, 0.85-1.0 x nominal, is in the JSON). Also: peak torque and speed, the
  fraction above the maker's rated torque and above the speed allowance.

### Power and energy (`power.png`)

- **Joint mechanical power:** mean of sum |torque x joint speed| over the 12 leg drives (`gait_pipe1.py`'s figure);
  positive work rate (sum of max(torque x speed, 0)) and negative (braking) separately.
- **Electrical estimate (a lower bound):** sum over the 12 leg drives of max(torque x speed, 0) / linkage efficiency
  + copper loss 3 R (tau_drive / Kt)^2, with tau_drive = joint torque / linkage efficiency, Kt the output-side torque
  constant (N m per A rms) and R the per-phase (Y-equivalent) resistance from the T. rex actuator catalog
  (`hardware/trex_form_study/designlab/data/actuators.json`: X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm;
  `drives.json` cites that catalog but has no Kt). Braking work is not recovered (the power board's ideal diodes block
  regeneration, `PARTS_PREMIUM.md`). Not modelled: gearbox friction, iron and driver losses, head and tail servos.
- **Hotel load:** an assumption, 30 W (25-40 W reported): Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and
  four CAN buses, IMU, safety MCU, drive standby, DC-DC losses. The parts list gives only a total walking estimate
  (150-280 W, mass-scaled from the PR), no per-item electrical figures.
- **Cost of transport:** power / (m g v): mechanical (sum |torque x speed|), and electrical with the hotel load.
- **Run time:** 146.5 Wh (two 6S 3300 packs) x 80 % usable / (leg electrical estimate + hotel load).

### Body and head (`steadiness.png`)

- **Chassis roll and pitch:** degrees from the chassis rotation matrix (as `gait_pipe1.py`); SD about the mean
  (oscillation), RMS about zero (`gait_pipe1.py`'s figure), 1st-99th percentile range, worst absolute peak.
- **Height:** chassis origin height; SD and 1st-99th percentile range (mm); dominant frequency.
- **Head:** the centre of mass of the head (`head_yaw` body: skull, frill and horns). Acceleration = the time
  derivative of its world velocity (`mj_objectVelocity`), low-passed at 20 Hz (zero phase, FFT), so gravity is not in
  it; RMS of the magnitude, per axis (fore-aft, lateral along the reset heading, vertical), peak (and the unfiltered
  peak in the JSON). Bob and sway: vertical and lateral position after removing a 1 s moving average (the walk's own
  progress), 1st-99th percentile range (mm).

### Feet (`feet.png`)

- **Swing clearance:** in each complete step-level swing, the highest point reached by the sole's lowest bottom corner
  above the floor (mm); swings under 5 mm counted as scuffs.
- **Touchdown impact:** the sole's downward speed at the sample before each step touchdown (m/s); peak load rate, the
  largest 5 ms rise of that foot's load in the first 30 ms after touchdown (body weights per second); peak load in the
  first 50 ms (body weights).
- **Slip in stance:** the sole's horizontal speed while loaded (raw 2 N contact, as `gait_pipe1.py`): mean, 95th
  percentile, and the **slip ratio**, the sole's travel while loaded / the chassis's forward travel.
- **Roll-ankle angle** (roll-ankle model only): the passive ankle-roll joint angle, 1st-99th percentile and
  min-max, the stance-only range and the time within 2 % of its +-20 deg limit. The ankle pitch range is reported for
  every case.

### Speed tracking (`speed_tracking.png`)

- **Forward speed:** chassis travel along the heading at reset over the window / window length (`gait_pipe1.py`;
  the sweep's light episodes use the control-step positions over the same window).
- **Over an episode:** the env's per-control-step (20 ms) forward speed, raw and as a 0.42 s (one stride) moving
  average; mean |step speed - command| and |mean speed - command| over the window.
- **Sweep:** window speed per episode against the command, the mean of 3 seeds, falls marked (any termination before
  20 s).

## Training dashboard (`training_dashboard.py`)

Reads `eval_during_training.jsonl` (train.py's PeriodicEval: every `--eval-every` steps, one deterministic 20 s
episode per evaluation command, randomisation on, 10 ms delay, seeds 900000 + i). Two formats: **speeds only**
(trike60-pipe1: `command_mps`, `forward_speed_mps`, `fell`, `termination`, `length_s`) and **with gait measures** (newer
runs add `measures`, the output of `rl/gait_metrics.measures`). Plotted against training steps: forward speed per
command (dashed: the command), falls per evaluation, the penalty schedules, and, from the episode at the highest
command (as train.py prints it), joint mechanical power, the worst joint's RMS / continuous rating per drive group,
time at the nominal torque limit per group, slip ratio and debounced touchdowns per foot, sideways drift and heading
change. Every panel marks the schedules the run record holds: a vertical line where the energy and slip penalties
reach full weight (`penalty_ramp_steps`, pipe1: 2 M) and a shaded band while the heat, saturation and short air-time
penalties ramp in (`late_ramp_from` to `late_ramp_to`, newer runs: 1 M to 3 M); the scale curves are train.py's own
`penalty_scale_at` and `late_scale_at`. Lines that are not JSON, lack `steps` or have a bad `steps` are skipped and
counted; a later line for the same steps replaces the earlier one (a resumed run re-evaluates). With two or more runs
it also draws `training_compare.png`. Each point is one episode: the curves are noisy, and a point is not a verdict.

## What these tools do not show

- Anything about the real robot: masses, contact, friction and drive models are estimates; nothing is measured on
  parts. The electrical figure is a lower bound from a simple loss model and the hotel load is assumed.
- A thermal result: RMS torque over 18 s stands in for steady walking.
- A verdict on a policy or a design: one checkpoint of one training seed.
- What visitors would see from a camera: the head figures are the head body's motion; the stills in
  `../analysis/results/` show the robot itself.
- Robustness beyond the listed cases (pushes, slopes, other floors, other delays): not run here.
