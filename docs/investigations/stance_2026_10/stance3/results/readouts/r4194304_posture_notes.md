# Posture DC and the backward drift: stance3 checkpoints at 4,194,304 steps

This is report-only. Nothing on Drive or in either clone was changed; both clones still show `git status --porcelain` empty.

## How the numbers were made

- `roll.py` builds each clone's stance stage env from that stage's `[env]` kwargs exactly as the stage loads them, the same way `judge_stance3.py`'s Ctx does. It ran through `tools/iso.py`, with `ISO_WT` set to the L clone for Run 1 and the statue, and to the LT clone for Runs 2 and 3.
- Checkpoints load with `environments.shared.policy_loading.load_sb3_checkpoint` and the VecNormalize stats, frozen. The normalisation is the one `stance_report._load_policy` applies.
- **Actions.** env.step receives `model.predict(normalized, deterministic=True)[0]`. SB3 clips that to [-1, 1], which is also what the judge's predict sends. The raw Gaussian mean is recorded beside it.
- **Rollout setup.** One env with seeds 3042-3081 rolled in order, heading 0. A balance-mode SubstepContactRecorder is attached; it only copies mjData.
- **Raw versus clipped.** Every action statistic in the tables states whether it uses the raw or the clipped value. The joint, COM and CoP statistics come from the physics state after env.step, so they used the clipped action.
- **Reproduction check, all 160 episodes.** Policy panels and statue match the judge JSONs:
  - per-episode length and termination: 0 mismatches;
  - per-foot `<label>_foot_cop_fore_aft` and `com_mtp3_head_m`: |difference| = 0;
  - returns: within 1e-11.
- **Statue.** It was rolled once, in the L clone, and matches all three judges' statue rows exactly.
- **Angle conventions:**
  - Angles are absolute qpos or offsets from the home keyframe, in degrees.
  - Every pitch joint's axis is +y, the leftward axis. A positive change tilts the child's up-vector forward, the same sign as eval_episodes' `leg_q_` keys.
  - Root pitch is asin(R20), positive nose up.
- **COM frames:**
  - "COM-MTP3" is the judge's `com_mtp3_head`: the COM ahead of the mean MTP III anchor along the heading.
  - "foot-site" is eval_episodes' `com_fore_aft_mean_m` frame. The two differ by a near-constant offset: COM-MTP3 minus foot-site is 79.97-80.02 mm in all four sets.

## 1. Window means, steps 200-999, full-horizon episodes

Values are the mean over episodes of the per-episode window mean. Full tables, with standard deviations and both sides, are in `tables/item1_window_means.csv` and `tables/items0_1_2.txt`.

| quantity (offset from home unless absolute) | Run 1 L s60 (38) | Run 2 LT s60 (40) | Run 3 LT s52 (38) | statue (39) |
|---|---|---|---|---|
| hip pitch, r / l (deg) | -6.05 / -6.00 | -2.53 / -2.53 | -2.70 / -2.63 | -2.75 / -2.74 |
| knee, r / l | +0.64 / +0.63 | -0.60 / -0.62 | +0.23 / +0.25 | -0.40 / -0.40 |
| ankle, r / l | +3.49 / +3.45 | +1.90 / +1.95 | +2.21 / +2.13 | +1.67 / +1.66 |
| MTP III, r / l | -0.13 / -0.12 | -0.13 / -0.14 | -0.10 / -0.12 | -0.12 / -0.12 |
| neck pitch (abs, + = head down) | +16.93 | +10.04 | -10.30 | +12.28 |
| head pitch (abs) | +9.90 | +0.51 | +6.08 | +12.59 |
| tail 1 / 2 / 3 pitch (abs; the stops are +-12.0) | -12.07 / -12.02 / -7.14 | -12.07 / -11.75 / -0.38 | -12.06 / -12.00 / -5.07 | -12.07 / -11.61 / -5.30 |
| root pitch (deg, + nose up) | -1.86 | -1.28 | -0.26 | -1.54 |
| COM-MTP3 (mm) | +40.9 | +62.5 | +48.0 | +63.9 |
| COM, foot-site frame (mm) | -39.1 | -17.5 | -32.0 | -16.1 |
| COM height (mm) | 830.4 | 847.1 | 861.2 | 838.2 |
| COM ahead of the mean heel line (mm) | +50.9 | +72.5 | +58.0 | +73.9 |

Absolute angles: hip -8.65 / -5.13 / -5.30 / -5.35; knee -24.2 / -25.4 / -24.6 / -25.2; ankle 80.4 / 78.8 / 79.1 / 78.6; MTP III 12.4 in all four. Values are listed in the same order as the table columns: Run 1, Run 2, Run 3, statue.

Left-right differences are within 0.1 deg in every run.

### Commanded leg DC (window mean of the action)

- **Clipped equals raw** everywhere except Run 2's tail_3_pitch:
  - Run 1 legs: hip pitch -0.05, knee +0.02, ankle +0.03.
  - Runs 2 and 3: legs within 0.01 of zero.
- **Run 2's tail_3_pitch actuator:**
  - clipped mean +0.997, raw +1.045;
  - |a| >= 0.9 on 99% of window steps;
  - over 10% saturation in all 40 episodes, the only actuator over that threshold;
  - |raw| > 1 on 95.5% of steps.

  This is the actuator behind Run 2's `max_actuator_saturation_fraction` failure. It holds tail 3 at -0.4 deg against the statue's passive -5.3 deg, while tails 1 and 2 rest on their -12 deg stops in every run and in the statue.

### Joint attribution of the COM difference against the statue

`kin.py` measures the foot-rooted kinematic sensitivity: COM-MTP3 change per degree, with the soles (metatarsus bodies) held at fixed orientation. Values at Run 1's mean pose:

| joint | mm per deg |
|---|---|
| ankle | -10.6 |
| knee | -5.3 |
| hip pitch | -0.8 |
| tail 1 | -0.55 |
| tail 2 | -0.34 |
| tail 3 | -0.14 |
| head | -0.17 |
| neck | about 0 |

Run 1 sits 23.0 mm behind the statue (measured). The kinematic prediction is -21.1 mm, made up of:

| joint | contribution (mm) |
|---|---|
| ankle | -19.3 |
| knee | -5.6 |
| hip | +2.8 |
| neck and head | +0.6 |
| tail | +0.4 |

Run 3 sits 15.9 mm behind the statue; the prediction is -14.8 mm:

| joint | contribution (mm) |
|---|---|
| ankle | -5.5 |
| knee | -3.6 |
| neck raised 22.6 deg | -6.0 |

Run 2 sits 1.4 mm behind; the prediction is -1.4 mm.

## 3. Training trend from eval_episodes.csv (`tables/eval_trend_*.csv`, `tables/eval_trend_all.txt`, `figs/eval_trend.png`)

Run 1, means over full-horizon episodes:

| phase | COM, foot-site (mm) | root pitch (deg) | hip offset (deg) | knee offset (deg) | ankle offset (deg) | r / l whole-foot CoP | XcoM heel margin min (mm) |
|---|---|---|---|---|---|---|---|
| 1.9-2.4M | -8.1 | -0.78 | -1.49 | +0.27 | +0.56 | +0.19 / +0.18 | 80.4 |
| 3.8-4.6M | -42.5 | -1.97 | -5.98 | -0.30 | +4.22 | -0.29 / -0.34 | 42.9 |
| last evaluation, 4.59M | -56.5 | | | | | | 29.4 |

Least-squares slopes after 2.3M, per million steps:

| run | COM (mm) | hip (deg) | ankle (deg) | knee (deg) | root pitch (deg) | XcoM heel margin (mm) |
|---|---|---|---|---|---|---|
| Run 1 | -14.0 | -1.97 | +1.43 | -0.12 | -0.58 | -15.7 |
| Run 2 | -1.6 | -0.14 | +0.10 | | | |
| Run 3 | -4.8 | +0.00 | -0.07 | | | |

Run 2 is flat at about -17 mm. Run 3 drifts from -24 to -33 mm.

**Which joint carries Run 1's drift: the ankle.** The ankle offset rose +3.7 deg between the 1.9-2.4M base and the 3.8-4.6M mean. At -10.6 mm per deg that gives -38.9 mm.

| period | hip (mm) | knee (mm) | ankle (mm) | predicted (mm) | measured (mm) | residual (mm) |
|---|---|---|---|---|---|---|
| mean of evaluations >= 3.8M | +3.6 | +3.0 | -38.9 | -32.2 | -34.5 | -2.3 |
| 4.19M evaluation | +3.5 | -2.3 | -28.2 | -27.0 | -29.5 | |

`tables/attrib_eval_trend_*.csv` has the per-evaluation breakdown.

- **The hip.** It moves more in degrees (-4.3 to -4.5 deg) but hardly moves the COM. The trunk COM sits nearly level with the hip, 803 mm above MTP III against 846 mm for the hip joint. The hip flexion re-levels the pelvis after the shank and thigh tilt back, and the pelvis ends 1.0-1.2 deg more nose-down.
- **The tail.** It is not in the CSV. At 4.2M tails 1 and 2 sit on their -12 deg stops, as they do in every run and the statue. Run 1's tail 3 is 1.8 deg lower than the statue's, worth +0.3 mm. The tail therefore cannot carry tens of millimetres. That is inferred from the sensitivities and the joint ranges: tail 1 over its whole range is worth at most about 13 mm.
- **Same joint as the earlier forward lean, opposite sign.** This drift is ankle-carried like the earlier forward lean, but now the ankle offset grows positive, so the shank tilts back over the planted foot.

## 2. Falls

### Data and method

**Files:**
- `tables/item2_fall_<run>_<seed>_kin.csv`: 10-step bins, each with the same-step full-horizon mean as a reference column.
- `tables/item2_events.txt`: the event order and the printed bins.
- `tables/item2_fall_<run>_<seed>.csv`: the first pass.

**Per-step quantities:**
- **Per-foot CoP.** The signed whole-foot CoP from `whole_foot_cop`, averaged over loaded steps in each bin. It is the same reduction the judge's `<label>_foot_cop_fore_aft` averages: -1 is the heel line, +1 the toe line.
- **Sole pitch and corner lift.** `kin_steps.py` rebuilds the pose from the recorded qpos and COM and reports, per foot:
  - the metatarsus pitch, positive toe-up;
  - the lift of the highest sole corner.

**Event rule.** Each quantity is compared with the envelope (min to max) of the same run's full-horizon episodes at the same step.

**Excluded event.** A sole corner lift above 4 mm in steps 0-49 happens in all 38 of 38 full-horizon episodes of both runs. It is a spawn transient and is not counted as an event.

### Run 1, seed 3059: tail_contact at 339

**Direction: backward and to the left.**

| at the last step | value |
|---|---|
| COM behind MTP III | 296 mm |
| COM behind the heel line | 299 mm |
| root pitch | +9.9 deg, nose up |
| COM | 86 mm left |
| roll | -4.9 deg |
| COM height | 749 mm, against a reference of 831 |

**Event order (first step):**

| step | event |
|---|---|
| 47 | COM lateral leaves its envelope |
| 70 | left sole pitch leaves its envelope |
| 91 | yaw leaves its envelope; it reaches +24 deg by step 210, against a full-horizon mean of 0.6 deg |
| 136 | the only leg joint more than 2 deg outside its envelope before the crossing: right hip pitch -0.6, envelope -8.4 to -2.6, so less flexed |
| from 150 | the right foot is toe-loaded (+1.0) and the left foot heel-loaded (-0.99); the left sole rolls toe-up, from +2.1 deg at bin 160 to +11.7 at bin 240 |
| 222 | **XcoM behind the heel line** |
| 224 | root pitch leaves its envelope |
| 241 | both CoPs at the heel |
| 249 | **COM behind the heel line** |
| 292 | first leg joint more than 5 deg outside its envelope: left ankle -4.6, envelope +0.4 to +7.2 |
| 296 | COM height falls more than 10 mm below its envelope |

COM-heel margin by bin:

| bin | 200 | 220 | 240 | 250 | 300 |
|---|---|---|---|---|---|
| COM-heel margin (mm) | 50.0 | 30.1 | 4.9 | -6.9 | -106 |

Leg angles at bin 330, offsets from home in degrees: hip -14.5 / -12.9, knee -2.3 / -7.8, ankle -2.9 / -8.3 (r / l).

### Run 1, seed 3063: tail_contact at 300

**Direction: backward and to the right.**

| at the last step | value |
|---|---|
| COM behind MTP III | 283 mm |
| root pitch | +10.7 deg |
| COM | 219 mm right |
| roll | +12.3 deg |
| yaw | -34 deg |

**Event order (first step):**

| step | event |
|---|---|
| 83 | yaw leaves its envelope |
| 103 | both CoPs at the heel; the COM is still 35 mm ahead of the heel line and partly recovers, to 40 mm at bin 160 |
| 138 | root pitch leaves its envelope |
| 143 | right sole rolls toe-up: +3.3 deg at bin 160, +9.8 at bin 200 |
| 191 | right knee 2 deg outside its envelope: -2.9, envelope -0.9 to +1.5 |
| 197 | **XcoM behind the heel line** |
| 218 | **COM behind the heel line** |
| 241 | first leg joint more than 5 deg outside its envelope: right ankle -3.9, envelope +1.2 to +7.0 |
| 253 | COM height falls more than 10 mm below its envelope |

### Run 1: which comes first

**Measured.** In both falls the COM goes behind the heel before a leg joint gives way. The feet pivot onto the heel line and the XcoM, then the COM, cross it before any leg joint leaves the full-horizon envelope by 5 deg:
- seed 3059: XcoM at 222, COM at 249, first 5-deg joint excursion at 292;
- seed 3063: XcoM at 197, COM at 218, first 5-deg joint excursion at 241.

At the 2-deg level, the right-knee excursion in seed 3063 comes 6 steps before the XcoM crossing, so there the two are concurrent. The large joint excursions come during the collapse: hips flexing to -13 to -14 deg, ankles to -8 to -10 deg (seed 3059 left ankle -8.3 at bin 330; seed 3063 right ankle -10.2 at bin 290).

**Inferred.** Both falls start from a twisting, heel-rolling stance: the body yaws 24-34 deg, one foot is toe-loaded and the other heel-loaded, and a sole rolls toe-up. They are not an ankle buckle in a symmetric stance.

### Run 3, seed 3050 (fallen at 218) and seed 3063 (fallen at 281)

**Direction: sideways to the left, slightly forward.**

| at the last step | seed 3050 | seed 3063 |
|---|---|---|
| COM left of the foot sites | 609 mm | 610 mm |
| roll | -41.5 deg | -41.4 deg |
| root pitch | -5.7 deg | -6.7 deg |
| COM ahead of MTP III | 133 mm | 153 mm |

The COM stays 90-170 mm ahead of the heel line throughout, so there is no heel event.

**Event order:**
- **Step 30: lateral offset already outside the full-horizon envelope.** The COM is already left of the feet: +38.2 mm (seed 3050) and +26.8 mm (seed 3063), against a full-horizon range of -27.0 to +27.6.
- **From the spawn: the right foot never flattens.**

  | right foot | seed 3050 | seed 3063 |
  |---|---|---|
  | sole pitch at spawn (deg) | -5.5 | -6.0 |
  | sole pitch at the end (deg) | -13 | -14 |
  | heel corner lift (mm) | 17 rising to 86 | 18 rising to 96 |
  | CoP | +1.0 at the toe throughout | +1.0 at the toe throughout |

- **Right ankle.** It leaves its envelope by 2 deg at step 72 (seed 3050) and 66 (seed 3063), and by 5 deg at 114 and 136, then opens to about +12 deg.
- **Then the roll.** Both hip rolls go negative, to -7 to -13 deg, and the body rolls to the left.

Here a joint, the right ankle, moves out of its envelope long before the root pitch (150 and 174) or the COM height (152 and 189) do. The fall is lateral, not backward.

## Files

- `roll.py`: the rollout. Its outputs are in `out/*.npz` and `out/*.json`, with the per-episode reproduction fields.
- `analyze.py`: items 0-2, first pass, written to `tables/items0_1_2.txt` and `tables/item1_window_means.csv`.
- `kin.py`: the sensitivities, in `tables/kin_sensitivity.*`.
- `attrib.py`: the attributions, in `tables/attrib*.{txt,csv}`.
- `kin_steps.py` and `falls.py`: item 2 events and bins.
- `eval_trend.py`: item 3, written to `tables/eval_trend_*` and `figs/eval_trend.png`.
- `sat_check.py` and `spawn_check.py`: the saturation and spawn checks.
