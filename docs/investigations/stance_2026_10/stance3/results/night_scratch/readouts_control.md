# Night continuations: the no-change Colab controls at E = 9,961,472 (report only)

Written 2026-10-19, 07:27-07:55 UTC. This is a report only. Nothing on Drive or GitHub changed, nothing was pushed, and nothing ran in /home/user/mesozoic-labs. Every simulation ran as one foreground process at a time, beside the 2-core extras queue (cert jobs) and the night readouts' own rollouts.

- **Clones.** Checked after the last rollout, all clean: screen_s3_LT at 1af2d4d7, screen_s3_L at 7e6b557d, exp_night at 80328abf (not used here).
- **Numbers.** "Measured" means read from a file or computed from a rollout I ran. Inferences are marked "inferred".
- **Paths.** S3 = scratchpad/foot/stance3, C = S3/night/cont, K = C/control.
- **Companion.** The arms' readouts are in C/readouts_night.md. Its section 6 had to use the 10,485,760 control, 0.52M after E. This file supplies the control at E itself, as spec section 6 item 4 asks.

## 0. Setup

### 0.1 The fetched files (task step 1)

| run | pkl A = pkl B (sha256) | zip testzip | zip num_timesteps / seed | zip sha256 |
|---|---|---|---|---|
| stance3_armLT_s52 (Run 3) | yes, 231b8338...7e6c | OK (None) | 9,961,472 / 52 | d63d1294...07f4 |
| stance3_armL_s60 (Run 1) | yes, 5308fb63...bff1 | OK (None) | 9,961,472 / 60 | 2e8d77bd...058a |
| stance3_armLT_s60 (Run 2) | yes, 64590392...3e3f | OK (None) | 9,961,472 / 60 | 4dff0393...27ce |

- Each zip holds data, pytorch_variables.pth, policy.pth (1,334,460 bytes), policy.optimizer.pth, _stable_baselines3_version and system_info.txt.
- **The pkls were hand-transcribed from base64**, so I also checked them statically with pickletools, which never unpickles (K/scripts/pkl_smooth.py -> K/pkl_smooth.txt):
  - The obs mean and var are finite.
  - In every run, the E obs mean lies within 0.039 obs-sd of the linear interpolation between the 8,388,608 and 10,485,760 pkls.
  - Inferred: the transcriptions are plausible. Section 2 adds an independent check for Run 2.
- **The fetched csvs** (eval_episodes.csv, train_episodes.csv) are byte-identical to the copies already in S3/runs/<RUN>/01_stance/: the refreshed 12M ledgers placed at 06:00. Copying them there, as instructed, changed nothing.

### 0.2 Placement (task step 2)

I ran `python3 -I S3/tools/place.py 9961472 <RUN> <A dir> <zip>` from S3/tools. The sha256 of each placed zip and pkl equals the fetched file's. The config files (plant_identity, stage_config, task_fingerprint) were already placed and match.

### 0.3 R1, the full judge (task step 3)

I ran `S3/tools/judge_mode.sh <RUN> <ARM> <SEED> 9961472 judge_9961472` once per run, one at a time, in this order:

| run | time | screen clone |
|---|---|---|
| Run 3 | 4 min 32 s | screen_s3_LT |
| Run 1 | 4 min 3 s | screen_s3_L |
| Run 2 | 4 min 28 s | screen_s3_LT |

- **Binding.** All three are fully bound: task_binding "require" and no relaxations. Every check matches: physics sha256, seed, task (d4a3b8f8 for L+T, cd963518 for arm L), checkpoint num_timesteps, mirror map, and every stage_config field.
- **Output.** S3/judged/<RUN>/judge_9961472.json, with logs at S3/judged/logs/<RUN>_judge_9961472.log.
- **Status.** This is a periodic slot, so the files are report only.

### 0.4 R2, the split readout (task step 4)

- **The rollouts.** S3/night/diag_split/roll_split.py, unchanged (sha256 dbb6aeea...), in the arm's screen clone, on S3/runs/<RUN>/01_stance. There are three rollouts:
  - Run 3 at E, K/out/r3ctl_9961472.npz;
  - Run 1 at E, K/out/r1ctl_9961472.npz;
  - Run 1's start at 8,388,608, K/out/r1_8388608.npz.
- **The readout code** is the night readouts' own C/r2/scripts/analyze_night.py, imported unchanged (sha256 76b9b7fb...), so the control numbers are computed exactly as the arms' are. The script is K/scripts/r2_control.py, with output in K/r2_control.txt and K/r2_control.json.
- **Reproduction against the judge file: passed for all four sets**, including the reused Run 3 start rollout diag_split/out/r3_8388608.npz. On all 40 seeds:
  - the seeds, lengths and terminations are equal;
  - |foot_cop_fore_aft - judge| is 0 on both feet;
  - |pad metric - judge| is 0;
  - the K2a and K2b flags are equal;
  - each rollout's model sha256 equals its judge file's.
  - The short episodes are reproduced too: Run 1 at E, 3046 at 338 and 3063 at 290, both "fallen".
- **Determinism.** My Run 1 8,388,608 rollout equals the one the night readouts made (C/r2/out/r1_8388608.npz) in all 156 arrays.
- **Start rows.** They equal the spec's numbers: R3 20.9 / 44.0 / 20.1 -> 24.2 / 0.685 / -0.253 / -0.974 / +2.00 / 0.99 / +0.62; R1 29.3 mm on exact sites.

### 0.5 R3, the ledgers (task step 5)

The Colab eval ledgers from 8,388,608 to E: 13 evaluations of 10 episodes each, with no duplicates. The script is K/scripts/r3_control.py, which uses C/r3/ledger_night.py's row and summary code unchanged; its output is K/r3_control.txt.

## 1. Spec 7.2 rows for the controls

Columns: the start (the judged Colab 8,388,608 file and its R2 rollout), the Colab control at E, and for reference the arm at E (from C/readouts_night.md). The "control side" column is where the control falls on each arm's pre-registered bars. The bars were written for the arms. Applying them to the control is a reading, not a test of a hypothesis.

### 1.1 Run 3 control against the Run A lines (H_A, R3 + stagger)

| readout | start | **control @E** | A @E | supports H_A | refutes H_A | control side |
|---|---|---|---|---|---|---|
| window stagger sd (mm) | 20.9 | **9.7** | 7.5 | <= 10 | >= 15 | supports side |
| six-seed mean \|s\| (mm) | 44.0 | **10.0** (+1.1, -16.2, +22.2, -4.3, -15.8, -0.4) | 13.3 | <= 20 | >= 35 | supports side |
| stagger sd i200 -> i999 (mm) | 20.1 -> 24.2 (grows) | **19.3 -> 6.1** (closes) | 10.7 -> 5.6 | i999 < i200 | i999 > i200 | supports side |
| r-l hip servo-error slope | +0.62 | **+0.21** (r-l hip force -0.26 N·m/mm; start -0.90; statue +1.57) | +0.52 | <= 0 | >= +0.5 | between the bars |
| CoP split sd | 0.685 | **0.490** | 0.471 | <= 0.60 | >= 0.68 | supports side |
| K2a / K2b / roll | 6 / 3 / 3 | **0 / 0 / 0** | 0 / 0 / 0 | K2a <= 3, K2b <= 3, roll <= 1 | K2a >= 6 with stagger sd <= 10 | supports side |
| ankle offset; common CoP (expected unchanged under A) | +2.00 deg; -0.253 | **+1.17 deg; -0.179** | +1.02 deg; -0.133 | | | both moved without any term |

### 1.2 Run 3 control against the Run B lines (H_B, R3 + anchor)

| readout | start | **control @E** | B @E | supports H_B | refutes H_B | control side |
|---|---|---|---|---|---|---|
| ankle offset, mean (deg) | +2.00 | **+1.17** (r +1.15, l +1.19) | +0.13 | <= +1.1 | >= +1.8 | between the bars |
| common CoP | -0.253 | **-0.179** | -0.039 | >= -0.15 | <= -0.22 | between the bars |
| x_c vs statue (B expected 7-10 mm forward) | -21.2 mm | **-17.0 mm** (+4.2 forward) | -5.4 mm (+15.8) | | | |
| K2a | 6 | **0** | 10 | <= 3 | ankle <= +1.1 and common >= -0.15 but K2a >= 6 | supports side (the refute condition is not met) |
| window stagger sd (mm) | 20.9 | **9.7** | 18.1 | >= 15 (consistent) | <= 10: "the stagger closed without its term" | **at the <= 10 line: the unchanged recipe closed the stagger with no term** |

### 1.3 Run 1 control against the Run C lines (H_C, R1 + stagger)

| readout | start | **control @E** | C @E | supports H_C | refutes H_C | control side |
|---|---|---|---|---|---|---|
| window stagger sd (mm) | 29.3 (exact sites) | **24.2** | 22.6 | <= 15 | >= 20 | refute side, like C |
| mean \|ankle L-R\| (deg) | 2.44 | **1.87** | 1.52 | (falls with the stagger, inferred) | | fell, less than in C |
| K2a, K2b | 0, 0 | **2, 2** (seeds 3046 and 3063 = the two panel falls) | 2, 2 (3046, 3063 = its two falls) | <= 3 each | | the same count on the same seeds |
| K3 noise falls (exploratory) | joint 7/20, own 0/20 | **joint 14/20, own 2/20** (own: 3046, 3050) | joint 10/20, own 1/20 (3046) | no prediction | | worse than C on both |

### 1.4 Other measured rows

| quantity | R3 start | **R3 control @E** | R1 start | **R1 control @E** |
|---|---|---|---|---|
| verdict (clean / 40, LCB) | FAIL (34, 0.725) | **PASS (40, 0.928)** | PASS (38, 0.851) | **FAIL (35, 0.755; hop_or_fall 2 > 1)** |
| x_c - statue (mm) | -21.2 | -17.0 | -28.0 | -8.7 |
| reward ratio to statue | 0.979 | 0.984 | 0.989 | **0.940** |
| heel-foot min CoP | -0.974 | -0.795 | -0.655 | -0.400 |
| CoP split sd; corr(split, stagger) | 0.685; -0.929 | 0.490; -0.982 | 0.288; +0.201 | 0.278; -0.826 |
| common CoP | -0.253 | -0.179 | -0.349 | -0.062 |
| ankle offset, mean (deg) | +2.00 | +1.17 | +3.12 | +0.72 |
| window mean \|stagger\| (mm) | 15.2 | 7.2 | 24.5 | 20.1 |
| six-seed \|s\| (mm) | 44.0 | 10.0 | 44.0 | 34.5 |
| stagger sd i200 -> i999 (mm) | 20.1 -> 24.2 | 19.3 -> 6.1 | 24.0 -> 33.1 | 24.4 -> 25.9 |
| r-l hip servo slope; force per mm | +0.62; -0.90 | +0.21; -0.26 | +0.06; -0.08 | +0.11; -0.19 |
| corr(yaw change i200->i999, bearing); mean \|creep\| (deg) | +0.880; 1.56 | +0.705; 2.00 | +0.949; 2.35 | +0.872; 1.95 |
| foot slide r / l, max (mm) | 20.1 / 19.0, 79.1 | 18.9 / 18.4, 80.4 | 18.6 / 20.2, 76.8 | 16.3 / 16.9, 60.3 |
| episode yaw change, mean \|.\| / max (deg) | 3.18 / 11.2 | 4.58 / 11.8 | 3.37 / 10.0 | 2.83 / 9.3 |
| knee / hip-pitch offset from settled (deg) | -0.44 / -0.64 | +0.47 / -0.91 | -1.08 / -1.89 | -1.03 / +1.62 |
| stagger-term cost per step (w 0.30, tol 45 mm) | 0.0449 | 0.0151 | 0.0876 | 0.0672 |
| anchor-term cost per step (w 0.4, tol 0.10) | 0.0502 | 0.0185 | 0.1128 | 0.0232 |
| R5 from R2: worst-actuator share at \|a\| >= 0.99; max window \|a\| | 0.000 | 0.000; 0.980 (tail_2_pitch) | 0.000; 0.430 | 0.000; 0.729 (head_pitch) |

## 2. R1 (full judge), R5 and R6: start, E and (context) 10,485,760

These come from K/scripts/r1_control.py, which imports C/r1_summary.py's summ() unchanged. The output is K/r1_control.txt and K/r1_control.json.

| | R3 8.39M | **R3 @E** | R3 10.49M | R1 8.39M | **R1 @E** | R1 10.49M | R2 8.39M | **R2 @E** | R2 10.49M |
|---|---|---|---|---|---|---|---|---|---|
| verdict | FAIL | **PASS** | PASS | PASS | **FAIL** | PASS | PASS | **FAIL** | PASS |
| clean / 40 (LCB) | 34 (0.725) | **40 (0.928)** | 40 (0.928) | 38 (0.851) | **35 (0.755)** | 39 (0.887) | 39 (0.887) | **12 (0.183)** | 39 (0.887) |
| full / hop-or-fall / panel falls | 40/0/0 | **40/0/0** | 40/0/0 | 40/0/0 | **38/2/2** (3046 at 338, 3063 at 290) | 40/0/0 | 40/0/0 | **40/28/0** | 40/0/0 |
| K2a / K2b / roll | 6/3/3 | **0/0/0** | 0/0/0 | 0/0/2 | **2/2/3** | 0/0/1 | 1/0/0 | **0/0/0** | 0/0/1 |
| x_c - statue (mm) | -21.2 | **-17.0** | -6.0 | -28.0 | **-8.7** | -6.7 | -11.6 | **-4.7** | -6.9 |
| reward ratio | 0.979 | **0.984** | 0.996 | 0.989 | **0.940** | 0.980 | 0.975 | **0.974** | 0.975 |
| noise joint / own falls (of 20) | 0/0 | **2/0** (3059, 3061) | 0/0 | 7/0 | **14/2** (own: 3046, 3050) | 7/0 | 2/0 | **1/0** | 1/0 |
| own sigma mean | 0.0806 | 0.0741 | 0.0728 | 0.0808 | 0.0745 | 0.0729 | 0.0774 | 0.0713 | 0.0696 |
| **R5**: episodes failing saturation (0.10) | 0 | **0** | 0 | 0 | **0** | 0 | 0 | **28** | 0 |
| side door (touchdown, support, settle touchdowns, yaw) | 0 | **0** | 0 | 0 | **support 1** (3063: 0.978 < 0.98) | 0 | 0 | **0** | 0 |
| **R6**: criterion failures, episodes per reason | pad 6, sole tilt 2, corner lift 3 | **none** | none | sole tilt 2, corner lift 1 | **horizon 2, pad 2, sole tilt 5, corner lift 2, window displacement 2, load share 1, load share windowed 1, sole contacts 2, all-feet support 1** | sole tilt 1 | pad 1 | **saturation 28** | sole tilt 1 |

**R6 detail, Run 1 at E.**
- Three unclean episodes are sole tilt only: 3045 at 2.0015 deg, 3047 at 2.44 and 3059 at 2.22.
- Every other reason belongs to the two falls:
  - 3046: sole tilt 22.8 deg, displacement 0.82 m.
  - 3063: sole tilt 31.7 deg, displacement 0.78 m, all-feet support 0.978, load share 0.23.
- Neither fall fails the yaw-change criterion.

**Run 2 at E (no night arm; report only).**
- The judge fails it on saturation alone: 28/40 episodes are over the 0.10 bar, and in 12 of them the fraction is 1.0 (one more at 0.994).
- **This is not a transcription artefact.** The Colab eval ledger, written on Colab independently of the fetched pkl, lists max_actuator_saturation_fraction as an unclean reason in this many episodes:

  | evaluation | 9.31M | 9.44M | 9.57M | 9.70M | 9.83M | 9.96M | 10.09M | 10.22M | 10.35M | 10.49M |
  |---|---|---|---|---|---|---|---|---|---|---|
  | saturated episodes (of 10) | 8 | 10 | 10 | 10 | 9 | 8 | 9 | 5 | 0 | 0 |

- So the 9.96M checkpoint sits inside a roughly 1M-step saturation episode that cleared by 10.35M.
- Inferred: it recurs Run 2's earlier pinned-actuator mode. The run log names tail_3_pitch_act pinned until about 6.2M. The actuator at E was not identified, because I made no R2 rollout for Run 2.

## 3. Safety flags (spec 7.2) applied to the controls

| flag | Run 3 control @E | Run 1 control @E | C @E (for reference) |
|---|---|---|---|
| clean below start for a NEW reason | no (40 >= 34) | **YES**: 35 < 38; falls, support, displacement, load share and sole contacts are new | YES (36 < 38) |
| hop-or-fall >= 2 | no (0) | **YES** (2) | YES (2) |
| any panel fall | no | **YES** (3046, 3063) | YES (3046, 3063) |
| joint-noise falls >= 3/20 or own >= 2/20 (bar written for A and B) | no (2/20 joint, 0/20 own) | measured 14/20 joint, 2/20 own (own would meet the bar) | measured 10/20, 1/20 |
| any episode failing touchdown / support / settle touchdowns / yaw change | no | **YES**: support 1 (3063, in a fall) | YES: support 1, yaw 2 (in its falls) |
| saturation >= 2 episodes over 0.10 | no (0) | no (0) | no (0) |
| roll rail up by >= 2 | no (3 -> 0) | no (2 -> 3) | no (2 -> 2) |
| x_c outside [-60, +30] mm of statue | no (-17.0) | no (-8.7) | no (+5.5) |
| reward ratio < 0.95 | no (0.984) | **YES** (0.940) | YES (0.946) |
| hop-watch or collapse stop | none: the Colab run continued to 12M (judge_final exists) | none: the same | no |

**C's safety flags also appear in the no-change control at E.**
- Of C's five flags, four are present in the control, measured on the same panel seeds:
  - falls on 3046 and 3063;
  - hop-or-fall 2;
  - a new unclean reason;
  - reward ratio below 0.95 (0.940, against C's 0.946).
- The fifth also shows: an all-feet support failure, in fall 3063.
- C's yaw-change failures (2) do not appear in the control.
- Inferred: apart from the yaw-change failures, these flags are not attributable to the stagger term. They belong to this lineage's state around 9.96M, which the unchanged recipe had cleared again by 10.49M (PASS, 0 falls).

## 4. R3: Colab eval ledgers, 8,388,608 -> E (no simulation)

Stagger here is ledger_stagger.py's linear FK map of the ledger's window-mean leg columns, over full-horizon episodes. These ledger episodes are the eval callback's seeds, not the judge panel's.

**Run 3 (stance3_armLT_s52), per evaluation:**

| step | clean | pad over | K2b out | roll | stagger sd / mean\|s\| (mm) | ankle - settled (deg) | \|ankle L-R\| | \|CoP r-l\| | common CoP | COM (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| 8,388,608 | 7 | 3 | 0 | 0 | 16.6 / 19.1 | +2.05 | 1.42 | 0.707 | -0.253 | -37.3 |
| 8,519,680 | 7 | 3 | 2 | 1 | 30.3 / 23.0 | +1.88 | 1.30 | 0.688 | -0.256 | -37.8 |
| 8,650,752 | 7 | 3 | 2 | 0 | 24.6 / 19.7 | +2.12 | 1.32 | 0.718 | -0.258 | -37.7 |
| 8,781,824 | 6 | 4 | 0 | 0 | 20.1 / 16.9 | +2.01 | 1.20 | 0.766 | -0.235 | -36.2 |
| 8,912,896 | 7 | 3 | 0 | 0 | 14.7 / 14.7 | +1.94 | 1.04 | 0.710 | -0.226 | -35.7 |
| 9,043,968 | 8 | 2 | 0 | 0 | 19.3 / 15.2 | +1.77 | 1.17 | 0.618 | -0.229 | -35.8 |
| 9,175,040 | 7 | 3 | 3 | 3 | 51.9 / 38.2 | +1.55 | 1.67 | 0.714 | -0.333 | -43.5 |
| 9,306,112 | 6 | 4 | 2 | 2 | 30.1 / 26.1 | +1.69 | 1.51 | 0.855 | -0.275 | -39.0 |
| 9,437,184 | 8 | 2 | 2 | 2 | 32.0 / 24.1 | +1.61 | 1.47 | 0.637 | -0.272 | -38.9 |
| 9,568,256 | 8 | 2 | 1 | 1 | 18.2 / 14.4 | +1.74 | 0.89 | 0.576 | -0.268 | -38.5 |
| 9,699,328 | 10 | 0 | 0 | 0 | 10.2 / 7.9 | +1.49 | 0.52 | 0.379 | -0.232 | -36.1 |
| 9,830,400 | 9 | 1 | 1 | 1 | 15.7 / 10.9 | +1.41 | 0.63 | 0.424 | -0.236 | -36.4 |
| 9,961,472 | 8 | 2 | 1 | 0 | 14.8 / 11.8 | +1.24 | 0.75 | 0.605 | -0.197 | -33.7 |
| block to 8,912,896 | 27/40 | 13 | 4 | 1 | 24.0 / 18.6 | +1.99 | 1.21 | 0.720 | -0.244 | -36.9 |
| block to 9,437,184 | 29/40 | 11 | 7 | 7 | 35.8 / 25.9 | +1.65 | 1.45 | 0.706 | -0.277 | -39.3 |
| block to 9,961,472 | 35/40 | 5 | 3 | 2 | 15.9 / 11.2 | +1.47 | 0.70 | 0.496 | -0.233 | -36.2 |

All 130 episodes reached full horizon.

**Run 1 (stance3_armL_s60), per evaluation:**

| step | clean | pad over | K2b out | roll | stagger sd / mean\|s\| (mm) | ankle - settled (deg) | \|ankle L-R\| | \|CoP r-l\| | common CoP | COM (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| 8,388,608 | 10 | 0 | 0 | 0 | 31.1 / 28.1 | +3.11 | 2.78 | 0.253 | -0.349 | -44.3 |
| 8,519,680 | 9 | 1 | 1 | 1 | 25.0 / 23.8 | +2.98 | 2.40 | 0.286 | -0.356 | -44.7 |
| 8,650,752 | 10 | 0 | 0 | 0 | 20.3 / 16.6 | +2.71 | 1.68 | 0.121 | -0.308 | -41.3 |
| 8,781,824 | 10 | 0 | 0 | 0 | 21.6 / 17.1 | +3.21 | 1.68 | 0.179 | -0.396 | -47.4 |
| 8,912,896 | 10 | 0 | 0 | 0 | 22.4 / 18.6 | +2.31 | 1.82 | 0.216 | -0.290 | -40.2 |
| 9,043,968 | 10 | 0 | 0 | 0 | 25.8 / 21.6 | +2.22 | 2.03 | 0.185 | -0.269 | -38.6 |
| 9,175,040 | 9 | 1 | 1 | 1 | 30.4 / 24.9 | +2.17 | 2.24 | 0.308 | -0.282 | -39.5 |
| 9,306,112 | 9 | 0 | 0 | 1 | 17.9 / 15.0 | +1.97 | 1.40 | 0.268 | -0.241 | -36.8 |
| 9,437,184 | 10 | 0 | 0 | 0 | 22.8 / 16.1 | +1.55 | 1.53 | 0.259 | -0.180 | -32.4 |
| 9,568,256 | 9 | 0 | 0 | 1 | 22.1 / 17.6 | +1.62 | 1.56 | 0.208 | -0.190 | -33.1 |
| 9,699,328 | 10 | 0 | 0 | 0 | 17.9 / 15.8 | +1.12 | 1.46 | 0.269 | -0.111 | -27.7 |
| 9,830,400 | 9 | 0 | 0 | 0 | 22.2 / 18.0 | +0.75 | 1.68 | 0.265 | -0.055 | -23.8 |
| 9,961,472 | 10 | 0 | 0 | 0 | 23.5 / 20.2 | +0.86 | 1.87 | 0.258 | -0.081 | -25.5 |
| block to 8,912,896 | 39/40 | 1 | 1 | 1 | 23.5 / 18.9 | +2.80 | 1.88 | 0.198 | -0.337 | -43.4 |
| block to 9,437,184 | 38/40 | 1 | 1 | 2 | 25.2 / 19.4 | +1.98 | 1.80 | 0.255 | -0.243 | -36.8 |
| block to 9,961,472 | 38/40 | 0 | 0 | 1 | 21.9 / 17.9 | +1.09 | 1.64 | 0.250 | -0.109 | -27.5 |

One episode did not reach full horizon: at 8,519,680, length 253, "fallen".

**Run 2 (stance3_armLT_s60), blocks:**
- Stagger sd 3.7 / 6.1 / 5.0 mm.
- Ankle offset +0.42 / +0.46 / +0.13 deg.
- Clean 40 / 22 / 3 of 40. The drop is the saturation in section 2.

**How the ledger squares with the judge panel.**
- **Run 3.** The ledger's block to E still has stagger sd 15.9 mm, and its single evaluations swing from 10.2 to 51.9 mm inside the window. The judge panel's exact-site stagger sd at E is 9.7 mm.
  - The two measure different episodes (eval seeds against panel seeds) with a different stagger estimate (leg-angle FK against foot sites).
  - Inferred: the ledger's spread across evaluations is the natural swing this window shows under the unchanged recipe.
- **Run 1.** The two agree: the panel's 24.2 mm against the ledger's last block at 21.9 mm.

## 5. What the control adds to the spec 7.3 reading (a reading of the fixed rules, not a verdict)

**1. A's stagger, split and K2 lines are matched by the unchanged recipe at the same step.**
- At E, the Colab control falls on the supports side of every H_A line except the hip-slope line, where it is between the bars at +0.21, and A sits at +0.52, on the refute side:

  | readout | control | A |
  |---|---|---|
  | window stagger sd (mm) | 9.7 | 7.5 |
  | six-seed \|s\| (mm) | 10.0 | 13.3 |
  | stagger sd i200 -> i999 (mm) | 19.3 -> 6.1 | 10.7 -> 5.6 |
  | CoP split sd | 0.490 | 0.471 |
  | K2a / K2b / roll | 0 / 0 / 0 | 0 / 0 / 0 |

- The control also moved the ankle level (+2.00 -> +1.17 deg) and the common CoP (-0.253 -> -0.179) with no term, about as far as A did (+1.02, -0.133).
- So against the no-change control, A's readouts carry no effect larger than the window's natural swing. Under spec section 6 item 3 that swing is large: the control's own ledger blocks went 24.0 -> 35.8 -> 15.9 mm.

**2. The harness-matched A-vs-B pair still separates the stagger, but the control changes what that means.**
- B kept its stagger (18.1 mm) while both A (7.5) and the unchanged Colab recipe (9.7) closed theirs.
- So the A-vs-B stagger difference could be B's anchor holding the stagger open (B's split sd grew to 1.029), rather than A's term closing it. This is inferred: the two cannot be told apart tonight.
- On B's own line, the control reads "the stagger closed without its term".

**3. Rule 4** ("A, B and the Colab control all improve alike: natural or harness recovery, no attribution; run a local no-change continuation first"):
- A and the control both meet "K2a <= 3 and the stagger closing". B does not (K2a 10, stagger 18.1).
- So rule 4 is not met as worded. But its intent applies to A, which improved exactly as the control did.
- The spec's own follow-up for any "win" is a local no-change continuation of the same checkpoint (same launcher, no override). That is what would separate the term's effect from the recipe's.

**4. B's anchor lines.**
- B moved its targeted readouts well beyond the control: ankle +0.13 against +1.17, common CoP -0.039 against -0.179.
- Its K2a of 10 is against the control's 0 on the same panel. Inferred: B's refutation is, if anything, stronger with the control in view, because the unchanged recipe reached K2a 0 from the same start.

**5. C's lines.**
- The control's stagger sd, 24.2 mm, is on the refute side like C's (22.6). C is not separable from the control on its mechanism line (1.6 mm apart).
- C's safety flags appear in the control at the same step on the same seeds: falls on 3046 and 3063, hop-or-fall 2, a new reason, support 3063, ratio 0.940. They are not the term's, except possibly C's two yaw-change failures.
- K3 joint-noise falls are 14/20 in the control against 10/20 in C, exploratory.

**6. The 10.49M offset.** C/readouts_night.md section 6 compared the arms with the 10,485,760 control. At E itself the pattern is the same for Run 3 (control already K2a 0 and stagger 9.7 at E). Run 1 differs: at E the control is in a transient FAIL (2 falls) that is gone by 10.49M (PASS, 39 clean).

## Files

| item | path |
|---|---|
| Judge files (fully bound, periodic slot, report only) | S3/judged/stance3_armLT_s52/judge_9961472.json, S3/judged/stance3_armL_s60/judge_9961472.json, S3/judged/stance3_armLT_s60/judge_9961472.json; logs S3/judged/logs/<RUN>_judge_9961472.log; queue S3/judged/logs/queue.log |
| Placed checkpoints | S3/runs/<RUN>/01_stance/models/stage1_9961472_steps.zip and stage1_vecnormalize_9961472_steps.pkl |
| R1/R5/R6 | K/scripts/r1_control.py -> K/r1_control.txt, K/r1_control.json |
| R2 rollouts | K/out/r3ctl_9961472.npz, K/out/r1ctl_9961472.npz, K/out/r1_8388608.npz (logs K/out/log_*.txt; roll_split sha256 K/out/roll_split.sha256) |
| R2 readouts | K/scripts/r2_control.py -> K/r2_control.txt, K/r2_control.json |
| R3 | K/scripts/r3_control.py -> K/r3_control.txt |
| pkl plausibility | K/scripts/pkl_smooth.py -> K/pkl_smooth.txt |

Not done here: R4 for the controls. The Colab TensorBoard and train.log are not local, and the task did not ask for it.
