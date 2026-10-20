# Night continuations: readouts R1-R6 at E = 9,961,472 (report only)

Written 2026-10-19, 07:05-07:40 UTC. This is a report only. Nothing on Drive or GitHub changed, nothing was pushed, and nothing ran in /home/user/mesozoic-labs. Every simulation ran as one process at a time, beside the 2-core extras queue.

- **Clones.** All three were checked after the last rollout and are clean: screen_s3_LT at 1af2d4d7, screen_s3_L at 7e6b557d, exp_night at 80328abf.
- **Numbers.** "Measured" means read from a file or computed from a rollout I ran. Inferences are marked "inferred".
- **Paths.** S3 = scratchpad/foot/stance3 and C = S3/night/cont.

## 0. Setup

**Endpoint.** E = 9,961,472, the fallback in spec section 4.3. All three runs have their E checkpoint. Its lineage:

- **Session 1** started at 04:30 from the Colab 8,388,608 pair. The container restart at about 05:08 killed it at 9.22-9.23M.
- **Session 2** started at 05:31. All three runs resumed together from their own 8,912,896 checkpoint, with the same seed, so the A/B pairing holds. The resume log says "dropping 2 prior evaluation(s) recorded after the resumed checkpoint".
- **Session 2 ended** with the restart at about 06:41, at 10.38-10.40M. It was not relaunched.
- **Files used.** The E checkpoint comes from session 2. Session 1's lines past 8,912,896 are left out of R4. Each eval ledger holds 10 rows per evaluation with no duplicates. train_episodes.csv is monotonic.

**R1 path.** I used `build/judge_exp.sh` with `JUDGED_ROOT=C/judged`, the path the build validated:

- It runs the judge in exp_night at 80328abf. For arm L it adds the arm overlay.
- The flags are `--task-binding report --allow-commit-mismatch --stage-subdir .`.
- The build validated it two ways:
  - `build/checks/equiv_compare.txt`: the panel, the statue and both noise tests are identical to the screen judge on seeds 3042 and 3043, for both arms;
  - `build/checks/g2_compare.txt`: config, task and plant are identical to the screen clones'.
- The output files are relaxed-binding and report only, so score_stance3.py refuses them and they never enter K1-K4.
- Each file rolled the arm's own task: d4a3b8f8 for L+T and cd963518 for arm L.

**R2 path.** I ran `diag_split/roll_split.py` unchanged, in the arm's SCREEN clone, as the spec says.

- A wrapper, `C/r2/scripts/roll_night.py`, only points roll_split's RUNS at a symlink layout.
- Each rollout reproduces its R1 file exactly on all 40 seeds:
  - lengths and terminations are equal;
  - |foot_cop_fore_aft - judge| is 0 on both feet;
  - |pad metric - judge| is 0;
  - the K2a and K2b flags are equal.
- That holds for A, B and C at E, and for the two start checkpoints (R3 and R1 at 8,388,608). See `C/r2/r2_night.txt`, top.
- Since R2 rolls the screen clone and R1 the exp_night clone, this also cross-checks the judge path on all 40 seeds of each run.
- **Start values.** The start values are recomputed from `diag_split/out/r3_8388608.npz` with the same code, and equal the spec's numbers: 20.9, 44.0, 20.1 -> 24.2, 0.685, -0.253, -0.974, +2.00, 0.99 and +0.62.
- **Run 1's start.** I rolled Run 1's start (S3/runs/stance3_armL_s60, 8,388,608) as the exact-site baseline spec 7.1 asks for. Its window stagger sd is 29.3 mm, the same as the FK figure.

## 1. Spec 7.2 tables: start (Colab 8,388,608 judge file and R2 rollout) against E

### Run A, `night_R3_stagger` (R3 + stagger penalty)

| readout | start | E | supports H_A | refutes H_A | side |
|---|---|---|---|---|---|
| window stagger sd (mm) | 20.9 | **7.5** | <= 10 | >= 15 | supports |
| six-seed mean \|s\| (mm) | 44.0 | **13.3** (-6.1, -8.9, +19.4, -15.5, -11.8, +18.2) | <= 20 | >= 35 | supports |
| stagger sd i200 -> i999 (mm) | 20.1 -> 24.2 (grows) | **10.7 -> 5.6** (closes) | i999 < i200 | i999 > i200 | supports |
| r-l hip servo-error slope | +0.62 | **+0.52** (r-l hip force -0.90 -> -0.93 N·m/mm; statue +1.57) | <= 0 | >= +0.5 | **refute side** (at the bar) |
| CoP split sd | 0.685 | **0.471** | <= 0.60 | >= 0.68 | supports |
| K2a / K2b / roll | 6 / 3 / 3 | **0 / 0 / 0** | K2a <= 3, K2b <= 3, roll <= 1 | K2a >= 6 with stagger sd <= 10 | supports |
| ankle offset; common CoP (expected unchanged) | +2.00 deg; -0.253 | **+1.02 deg; -0.133** | | | NOT unchanged: both moved about halfway to the 6.29M levels (+1.08, -0.113) |

Also measured:

| quantity | start | E |
|---|---|---|
| clean | 34/40 | 40/40 (LCB 0.9278, verdict PASS) |
| x_c - statue | -21.2 mm | -13.7 mm |
| heel-foot min CoP | -0.974 | -0.726 |
| mean \|ankle L-R\| | 0.99 | 0.38 deg |
| corr(yaw change i200->i999, bearing) | +0.880 | +0.154 |
| mean \|yaw creep\| | 1.56 | 0.92 deg |
| corr(split, stagger) | -0.929 | -0.961 |
| foot slide r / l | 20.1 / 19.0 mm | 19.3 / 18.7 mm |
| episode yaw change, mean \|.\| / max | 3.18 / 11.2 deg | 4.16 / 11.4 deg |
| stagger-term cost per step | 0.045 | 0.0084 |
| falls | 0 | 0 |
| noise (joint / own) | 0/20, 0/20 | 0/20, 0/20 |

### Run B, `night_R3_anchor` (R3 + ankle anchor)

| readout | start | E | supports H_B | refutes H_B | side |
|---|---|---|---|---|---|
| ankle offset, mean (deg) | +2.00 | **+0.13** (r +0.16, l +0.10) | <= +1.1 | >= +1.8 | supports |
| common CoP | -0.253 | **-0.039** | >= -0.15 | <= -0.22 | supports |
| x_c (expected 7-10 mm forward) | -21.2 mm vs statue | **-5.4 mm** (+15.8 mm forward) | | | moved more than the inferred 7-10 |
| K2a | 6 | **10** (3050, 3052, 3062, 3067, 3069, 3071, 3072, 3073, 3074, 3081) | <= 3 | ankle <= +1.1 and common >= -0.15 but K2a >= 6 | **REFUTES** (all three conditions met) |
| window stagger sd (mm) | 20.9 | **18.1** | >= 15 (consistent) | <= 10 (A's attribution fails) | consistent (>= 15) |

Also measured:

| quantity | start | E |
|---|---|---|
| K2b | 3 | 13 |
| clean | 34/40 | 30/40 (LCB 0.6129, FAIL) |
| CoP split sd | 0.685 | 1.029 (grew) |
| heel-foot min CoP | -0.974 | -0.963 |
| six-seed \|s\| | 44.0 | 34.8 mm |
| stagger sd i200 -> i999 | 20.1 -> 24.2 | 17.7 -> 19.0 mm |
| r-l hip slope | +0.62 | +1.09 (force -1.62 N·m/mm) |
| corr(yaw, bearing) | +0.880 | +0.778 |
| \|ankle L-R\| | 0.99 | 0.64 deg |
| roll | 3 | 1 |
| falls | 0 | 0 |
| noise | 0/0 | 0/0 |

The side door named in spec section 8 (knee or hip carrying the lean) does not show. Knee / hip-pitch offset from settled went from -0.44 / -0.64 to +0.28 / -0.21 deg, both closer to settled.

### Run C, `night_R1_stagger` (R1 + stagger penalty)

| readout | start | E | supports H_C | refutes H_C | side |
|---|---|---|---|---|---|
| window stagger sd (mm) | 29.3 (exact sites, re-measured; FK gave 29.3) | **22.6** | <= 15 | >= 20 | **REFUTES** |
| mean \|ankle L-R\| (deg) | 2.44 | **1.52** | (falls with the stagger, inferred) | | fell |
| K2a, K2b | 0, 0 | **2, 2** (seeds 3046 and 3063 = the two panel falls) | <= 3 each | | within <= 3, but both are falls (see the safety flags) |
| K3 noise falls (exploratory) | joint 7/20, own 0/20 | **joint 10/20, own 1/20** (own: 3046) | no prediction | | |

Also measured:

| quantity | start | E |
|---|---|---|
| verdict | PASS | FAIL (clean_stance_lcb 0.7856, hop_or_fall 2 > 1) |
| clean | 38/40 | 36/40 |
| ankle offset | +3.12 | +0.76 deg |
| common CoP | -0.349 | +0.105 |
| x_c - statue | -28.0 mm | +5.5 mm |
| reward ratio to statue | 0.989 | 0.946 |
| six-seed \|s\| | 44.0 | 35.4 mm |
| stagger sd i200 -> i999 | 24.0 -> 33.1 | 19.7 -> 23.8 mm |
| corr(split, stagger) | +0.201 | -0.932 (now strongly negative, as in the L+T runs) |
| CoP split sd | 0.288 | 0.313 |
| r-l hip slope | +0.06 | +0.14 |
| corr(yaw, bearing) | +0.949 | +0.822 |
| knee / hip-pitch offset from settled | -1.08 / -1.89 | -2.51 / +2.33 deg |

## 2. R1 (full judge), R5 and R6: start against E

| | A start | A E | B start | B E | C start | C E |
|---|---|---|---|---|---|---|
| verdict | FAIL | PASS | FAIL | FAIL | PASS | FAIL |
| clean / 40 (LCB) | 34 (0.725) | 40 (0.928) | 34 (0.725) | 30 (0.613) | 38 (0.851) | 36 (0.786) |
| full / hop-or-fall / panel falls | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 | 40 / 0 / 0 | 38 / 2 / 2 (3046 at 342, 3063 at 337, "fallen") |
| K2a / K2b / roll | 6 / 3 / 3 | 0 / 0 / 0 | 6 / 3 / 3 | 10 / 13 / 1 | 0 / 0 / 2 | 2 / 2 / 2 |
| x_c median; minus statue (64.0-64.05) | 42.8; -21.2 | 50.4; -13.7 | 42.8; -21.2 | 58.6; -5.4 | 36.0; -28.0 | 69.5; +5.5 |
| reward ratio to statue | 0.979 | 0.983 | 0.979 | 0.985 | 0.989 | 0.946 |
| noise joint / own falls (20 each) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 7 / 0 | 10 / 1 |
| own sigma mean | 0.0806 | 0.0743 | 0.0806 | 0.0749 | 0.0808 | 0.0743 |
| **R5**: episodes failing saturation (0.10) | 0 | 0 | 0 | 0 | 0 | 0 |
| **R6**: criterion failures, episodes per reason | pad 6, sole tilt 2, corner lift 3 | none | pad 6, sole tilt 2, corner lift 3 | pad 10, corner lift 1 | sole tilt 2, corner lift 1 | horizon 2, pad 2, sole tilt 4, corner lift 2, window displacement 2, foot load share 2, load share windowed 1, sole contacts 2, all-feet support 1, yaw change 2 |

**R5 from the R2 rollouts:**

- The worst actuator's share of window steps at |a| >= 0.99 is 0.000 in every episode of all five sets.
- The maximum window |a| is 0.59 for A, 0.95 for B (tail_2_pitch) and 0.75 for C.

**R6.** In C, every new reason except the two sole-tilt-only episodes (3047 and 3059, 2.38 and 2.17 deg) belongs to the two fall episodes. In those falls the yaw change is 47.4 and 52.0 deg, and all-feet support is 0.876 on 3063.

## 3. Safety flags (spec 7.2, every run; any one overrides "supports")

| flag | A | B | C |
|---|---|---|---|
| clean below start for a NEW reason | no (40 >= 34) | no: 30 < 34, but only the start's reasons (pad, corner lift) | **YES**: 36 < 38; falls, support, yaw, displacement and load share are new |
| hop-or-fall >= 2 | no (0) | no (0) | **YES** (2) |
| any panel fall | no | no | **YES** (3046, 3063) |
| joint-noise falls >= 3/20 or own >= 2/20 (A, B) | no (0/0) | no (0/0) | n/a for the bar; measured 10/20 joint (start 7), 1/20 own |
| any episode failing touchdown / all-feet support / settle touchdowns / yaw change | no | no | **YES**: support 1, yaw change 2, all in the fall episodes |
| saturation >= 2 episodes over 0.10 | no (0) | no (0) | no (0) |
| roll rail up by >= 2 | no (3 -> 0) | no (3 -> 1) | no (2 -> 2) |
| x_c outside [-60, +30] mm of statue | no (-13.7) | no (-5.4) | no (+5.5) |
| reward ratio < 0.95 | no (0.983) | no (0.985) | **YES** (0.946) |
| hop-watch or collapse stop | no | no | no |

For the last row: no hop_watch.json and no ENDED_ON_ITS_OWN.json exist. The runs were killed by the container restart at 06:41, not by a watch.

- **R4 value-mismatch flag** (explained variance more than 0.2 below the first update after 524,288 steps): no flag in any run. The minimum EV after 8,912,896 is:
  - A 0.939, against 0.967 at the first update;
  - B 0.926, against 0.966;
  - C 0.948, against 0.935.

## 4. R3: eval ledgers 8.39M -> E (no simulation; `C/r3/ledger_night.py` -> `C/r3/ledger_night.txt`)

Pooled per 524,288-step block of 40 episodes. Stagger is from design/scripts/ledger_stagger.py's FK map, over full-horizon episodes.

| block ending | run | clean | pad over | K2b out | roll | stagger sd / mean\|s\| (mm) | ankle - settled (deg) | \|CoP r-l\| | common CoP | COM (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| 8,912,896 | A | 36 | 4 | 2 | 2 | 18.7 / 13.0 | +1.65 | 0.529 | -0.212 | -34.5 |
| 9,437,184 | A | 39 | 1 | 1 | 1 | 9.4 / 7.3 | +0.97 | 0.398 | -0.126 | -28.5 |
| 9,961,472 | A | 39 | 1 | 0 | 0 | 8.8 / 6.8 | +1.29 | 0.358 | -0.159 | -30.9 |
| 8,912,896 | B | 32 | 8 | 7 | 4 | 24.9 / 17.0 | +1.16 | 0.683 | -0.153 | -30.5 |
| 9,437,184 | B | 32 | 8 | 9 | 3 | 17.1 / 12.2 | +0.45 | 0.718 | -0.047 | -22.9 |
| 9,961,472 | B | 36 | 4 | 7 | 1 | 13.0 / 9.5 | +0.19 | 0.673 | -0.011 | -20.4 |
| 8,912,896 | C | 39 | 1 | 1 | 1 | 26.1 / 24.0 | +3.78 | 0.181 | -0.447 | -51.1 |
| 9,437,184 | C | 35 | 5 | 3 | 3 | 21.8 / 19.8 | +2.98 | 0.281 | -0.305 | -41.2 |
| 9,961,472 | C | 34 | 1 | 1 | 6 | 19.8 / 15.7 | +1.52 | 0.237 | -0.004 | -20.3 |
| 8,912,896 | Colab R3 (no change) | 27 | 13 | 4 | 1 | 24.0 / 18.6 | +1.99 | 0.720 | -0.244 | -36.9 |
| 9,437,184 | Colab R3 | 29 | 11 | 7 | 7 | 35.8 / 25.9 | +1.65 | 0.706 | -0.277 | -39.3 |
| 9,961,472 | Colab R3 | 35 | 5 | 3 | 2 | 15.9 / 11.2 | +1.47 | 0.496 | -0.233 | -36.2 |
| 8,912,896 | Colab R1 (no change) | 39 | 1 | 1 | 1 | 23.5 / 18.9 | +2.80 | 0.198 | -0.337 | -43.4 |
| 9,437,184 | Colab R1 | 38 | 1 | 1 | 2 | 25.2 / 19.4 | +1.98 | 0.255 | -0.243 | -36.8 |
| 9,961,472 | Colab R1 | 38 | 0 | 0 | 1 | 21.9 / 17.9 | +1.09 | 0.250 | -0.109 | -27.5 |

Single evaluations at 8,388,608 in the Colab ledger:
- R3: stagger sd 16.6 mm, ankle +2.05 deg, |CoP r-l| 0.707, common CoP -0.253.
- R1: stagger sd 31.1 mm, ankle +3.11 deg.

Non-full-horizon evaluation episodes, 8.39M to E:

| run | step | length | cause |
|---|---|---|---|
| A | 9,306,112 | 465 | fallen |
| B | 9,306,112 | 573 | fallen |
| C | 8,912,896 | 707 | tail_contact |
| C | 9,437,184 | 350 | tail_contact |
| C | 9,699,328 | 263 | tail_contact |
| Colab R1 | 8,519,680 | 253 | fallen |
| Colab R3 | none | | |

## 5. R4: training side (`C/r4/train_side.py` -> `C/r4/train_side.txt`)

No tensorboard package is installed. train.log's per-update [progress] lines carry the same SB3 values. Bin means over 262,144 steps, on the lineage of the E checkpoint:

| run | first update: EV / value loss / approx_kl / clip / std / mirror KL / lr | EV per bin (8.65M ... 9.96M) | approx_kl per bin | std at the last bin |
|---|---|---|---|---|
| A | 0.967 / 6.2e-4 / 0.024 / 0.23 / 0.0806 / 0.075 / 1.60e-5 | 0.967, 0.965, 0.964, 0.968, 0.971, 0.975 | 0.021-0.024 | 0.0749 |
| B | 0.966 / 5.3e-4 / 0.024 / 0.26 / 0.0805 / 0.074 / 1.60e-5 | 0.960, 0.961, 0.965, 0.963, 0.958, 0.963 | 0.023 | 0.0753 |
| C | 0.935 / 1.4e-3 / 0.018 / 0.19 / 0.0808 / 0.068 / 1.60e-5 | 0.945, 0.970, 0.974, 0.977, 0.977, 0.976 | 0.020-0.023 | 0.0747 |

- **The rest of the bins.**
  - Learning rate: 1.58e-5 -> 1.36e-5 in all three.
  - Clip fraction: 0.20-0.23 (A), 0.23-0.24 (B), 0.20-0.22 (C).
  - Mirror KL: 0.055-0.058 (A), 0.059-0.063 (B), 0.054-0.068 (C).
  - ent_coef is 0 throughout.
- **A against B (same start, same seed).** B's EV is 0.003-0.013 below A's in 5 of the 6 bins. In the bin ending at 9.18M it is 0.001 above.
- **Each term's own rollout mean, first bin -> E bin.**
  - A: |stagger| 14.7 -> 8.2 mm; term -0.030 -> -0.008 per step.
  - B: ankle error 1.46 -> 0.16 deg; term -0.027 -> -0.002.
  - C: |stagger| 23.7 -> 18.4 mm; term -0.070 -> -0.044.
- **Training falls per 262,144-step bin** (train_episodes.csv, about 260 episodes per bin):
  - A: 0, 0, 0, 1, 0, 0.
  - B: 0, 0, 1, 2, 0, 1 (the last is a nosedive at length 194).
  - C: 1, 0, 4, 9, 7, 8, a fraction of 0.015-0.034 from 9.04M on: tail_contact 12, fallen 7, head_contact 6, nosedive 4 over the window.
  - "Late" here means length > 200. That is my definition, because none is written down: A 0-1, B 0-2, C 0-9 per bin.
- hop_watch.json is absent and the collapse watch did not fire in any run.

## 6. The no-change control

- **The control was not judged at E here.** The Colab 9,961,472 checkpoints are being fetched by another job (S3/fetch/r9961472, in progress at 07:25).
- **The nearest judged control is 10,485,760**, 0.52M after E. Both judge files (S3/judged/<RUN>/judge_10485760.json) give:
  - Run 3 (stance3_armLT_s52): PASS, clean 40, K2a 0, K2b 0, roll 0, x_c -6.0 mm vs statue, ratio 0.996, noise 0/0.
  - Run 1 (stance3_armL_s60): PASS, clean 39, K2a 0, K2b 0, roll 1, x_c -6.7 mm, ratio 0.980, noise joint 7/20, own 0/20.
- **What that means for A's K2a count.** The unchanged recipe also cleared Run 3's K2a by 10.49M. So A's count is not attributable to the term against the control (spec section 6, item 2). The harness-matched readouts are the A-vs-B pair:
  - stagger sd: A 7.5 mm, B 18.1 mm;
  - ankle level: A +1.02 deg, B +0.13 deg.
- **Ledger, not judge:** the Colab R3 ledger's last block to E still shows stagger sd 15.9 mm, against A's 8.8 and B's 13.0.

## 7. The spec 7.3 rules applied to these numbers (a reading of the fixed rules, not a verdict)

- **Case 1** needs "A supports, B leaves the stagger at >= 15 mm".
  - B's stagger is 18.1 mm, so B's half holds.
  - A supports on its stagger, six-seed, closure, split and K2 lines.
  - A's restoring-torque line is at the refute bar: +0.52 against >= +0.5.
  - A's "ankle and common unchanged" expectation failed: both moved about halfway to the 6.29M levels.
  - C refutes (22.6 mm, against >= 20). Case 1 then reads: "the closure is not general: propose a from-scratch screen with the term rather than a commit."
- **Case 2 / 3.** B met its ankle and common lines, but K2a went to 10, so H_B is refuted by its own pre-registered line. Neither case applies.
- **Case 4** needs all three to improve alike. B did not improve (K2a 6 -> 10), so case 4 does not apply. But the 10.49M control did recover (section 6).
- **Case 6.** C did not meet its mechanism (stagger 22.6 mm), so case 6 is not met as worded. C does carry five safety flags:
  - falls;
  - hop-or-fall 2;
  - a new unclean reason;
  - support and yaw failures, inside the falls;
  - reward ratio 0.946.

## Files

| item | path |
|---|---|
| R1 judge files (with .exp.json and .log) | C/judged/night_R3_stagger/judge_9961472.json, C/judged/night_R3_anchor/judge_9961472.json, C/judged/night_R1_stagger/judge_9961472.json; queue log C/judged/queue.log |
| R1/R5/R6 summary | C/r1_summary.py -> C/r1_summary.txt, C/r1_summary.json |
| R2 rollouts | C/r2/out/{night_R3_stagger,night_R3_anchor,night_R1_stagger}_9961472.npz, C/r2/out/r1_8388608.npz (logs log_*.txt) |
| R2 wrapper and analysis | C/r2/scripts/roll_night.py, C/r2/scripts/analyze_night.py -> C/r2/r2_night.txt, C/r2/r2_readouts.json |
| R3 | C/r3/ledger_night.py -> C/r3/ledger_night.txt |
| R4 | C/r4/train_side.py -> C/r4/train_side.txt |
