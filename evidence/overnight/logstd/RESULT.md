# LOGSTD scan: how much policy noise the stance parent tolerates in main's T. rex locomotion env

Stage folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/logstd/` (called S below). Worked 03:53-05:15 UTC (write-up due 08:30).

This was CPU evaluation only. Nothing was trained or fitted. The repository, the export, the physics and the checkpoint files were not changed. The checkpoint files are read-only on disk and have the same sha256 before and after the scan (`out/ckpt_sha256_start.txt`, `out/ckpt_sha256_end.txt`). log_std was set only on the in-memory policy object of each process, and nothing was saved. Simulations ran one process at a time through `scripts/r.sh` (nice -n 10, OMP_NUM_THREADS=1, TMPDIR=S/tmp). Every command is in `CMDLOG.txt`.

"Stands" means the episode reaches 1000 control steps (10 s) without early termination. "Return" is the undiscounted reward summed over the episode. w is forward_vel_weight.

## Short answers

1. **Largest uniform log_std at which the parent stands in at least 36 of 40 stochastic episodes.**
   - On the brief's grid (-2.5 to -1.5 in steps of 0.25), the answer is **-2.0 (std 0.135)**. It stood 40/40 in each of three independent noise streams (120/120). At -1.75 (std 0.174) it stood 32, 31 and 28 of 40. [measured: out/analyze_stdout.txt]
   - I added three points between those two values on the same 40 seeds and three streams:

     | log_std | std | stood (torch / np / np2 stream) |
     |---|---|---|
     | -1.9375 | 0.144 | 40 / 39 / 39 |
     | -1.875 | 0.153 | 40 / 38 / 38 |
     | -1.8125 | 0.163 | 38 / 37 / **35** |

     So **-1.875 is the largest value that passes the 36/40 bar in every stream**. -1.8125 fails in one of the three streams. [measured]
   - Pilot 3's -1.5 stood 4, 2 and 3 of 40 (9/120). The torch stream reproduces the REWARD stage's 36/40 terminations seed for seed. [measured]
2. **Which floor variant passes. The two floors in the brief are the uniform settings themselves.**
   - The parent's largest learned std is 0.1107 (tail_1_yaw, log_std -2.201), which is below exp(-2.0) = 0.135. So max(learned, exp(-2.0)) gives exp(-2.0) on all 15 actuators, and max(learned, exp(-1.75)) gives exp(-1.75) on all of them.
   - The floor runs reproduce the uniform runs bit for bit in both streams tested (same log_std vector; all 40 episodes have the same length, return and final state). **max(learned, exp(-2.0)) passes (= uniform -2.0); max(learned, exp(-1.75)) fails (= uniform -1.75).** [measured]
   - Extra: two floors that do differ from uniform, because they keep the learned neck and tail std, were also run:
     - max(learned, exp(-2.25)) differs only on tail_1_yaw;
     - max(learned, exp(-2.5)) differs on all neck and tail actuators.

     Both stood 80/80, and their readouts match the uniform -2.25 and -2.5 runs to within noise. [measured]
3. **Exploration exists in every setting that stands, but the foot lifts are not steps.**
   - Floor-truth lift-offs (the frozen probe's rule) per setting:
     - Own std and -2.5: none (1 episode in 120 at -2.5).
     - -2.25: lift-offs in about half the episodes (65/120), 0.7 per episode.
     - -2.0: lift-offs in every episode (120/120), 7.3 swings per episode, both feet in 114/120.
     - -1.875: 16.4 swings per episode.
     - -1.75: 27.9 swings per episode.
   - These swings are short and low:
     - The median swing lasts 0.02 s, which is two control steps, the probe's debounce minimum.
     - The median clearance is 1.5-2.7 mm (-2.25 to -1.75). In standing settings a swing reaches 1 cm in at most 8/120 episodes and 2 cm in none (2 cm first appears at -1.8125, in 4/120 episodes).
     - The left/right order is random (probe alternation index 0.50-0.52).
   - For comparison, the hand-built FINAL3 walk has 48 mm clearance, alternation 1.0 and a 1.05 Hz stride (EXISTENCE stage output, not re-run here).
   - The probe labels -2.0 "stand (stepping / tapping in place)" in 118/120 episodes. At -1.875 and below it labels most episodes "shuffle (forward travel mostly sliding in contact)". [measured: out/extra2_stdout.txt, out/analyze_stdout.txt]
4. **Displacement: noise makes the parent slide backward, and the spread is small.**
   - Mean 10-s displacement along the prey direction, and its spread:

     | setting | mean displacement | sd across episodes |
     |---|---|---|
     | deterministic parent | -0.031 m | 0.020 m |
     | -2.5 | -0.049 m | 0.022 m |
     | -2.25 | -0.096 m | 0.025 m |
     | -2.0 | -0.226 m | 0.037 m |
     | -1.875 | -0.336 m (full-length episodes) | 0.050 m |
     | -1.75 | -0.495 m (full-length episodes) | 0.089 m |

   - From -2.25 to -1.75, every full-length episode ends behind its start. The most forward one ends at -0.024 m at -2.25, -0.146 m at -2.0 and -0.208 m at -1.875. With the own std and at -2.5, some end up to +0.022 m ahead, as the statue does (+0.024 m). [measured: out/extra3_stdout.txt]
5. **Applied noise.**
   - The [-1, 1] clip removes almost nothing: the applied pre-filter std is 0.986-1.014 times the set std on every actuator of every stochastic run (525 run-actuator pairs).
   - The env's 10 Hz command filter (α = 0.3859) passes **0.49 of it on every actuator** (measured ratio 0.483-0.499; white-noise theory sqrt(α/(2-α)) = 0.489).
   - The noise that reaches the plant is therefore:

     | setting | std after the filter |
     |---|---|
     | own learned std | 0.0035 knee, 0.006 hip pitch and ankle, 0.025 hip roll, 0.041-0.049 neck/head, 0.047-0.055 tail |
     | -2.5 | 0.040 |
     | -2.25 | 0.052 |
     | -2.0 | 0.066 |
     | -1.875 | 0.075 |
     | -1.75 | 0.085 |
     | -1.5 | 0.109 |

     [measured: env._action_filter_state read every step, minus the low-passed policy mean; ranges in out/extra3_stdout.txt]
6. **Returns per 10-s episode.**

   | behaviour | w 0.2 | w 2.0 |
   |---|---|---|
   | statue | 1094.7 | 1092.6 |
   | deterministic parent | 1095.9 | 1093.7 |
   | own std | 1095.6-1095.7 | 1093.4 |
   | -2.0 | 1086.9-1087.5 | 1070.4-1071.2 |
   | -1.875 | 1038-1084 | 1013-1059 |
   | -1.75 | 870-956 | 837-922 |
   | -1.5 | 348-422 | 321-395 |

   [measured]
   - Paired against the statue on the same seed, -2.0 loses 7.1 to 7.8 at w 0.2 and 21.4 to 22.2 at w 2.0. About 16 of the 22 lost at w 2.0 is the forward term (the backward slide). [measured: out/extra_stdout.txt]

**Judgement** [inferred, from the measurements above]:
- **Stand rate against log_std.** The stand rate falls off a cliff between std 0.135 and 0.22. Pooled over three streams:
  - 1.00 at -2.0;
  - 0.98 at -1.9375;
  - 0.97 at -1.875;
  - 0.92 at -1.8125;
  - 0.76 at -1.75;
  - 0.08 at -1.5.
- **Safest start.** -2.0 is the largest grid value with no falls (0/120; 95% Wilson interval for the stand rate 0.97-1.00). Its exploration is micro-unloading in place, about 7 brief lifts per 10 s, plus a slow backward slide.
- **The edge.** -1.875 roughly doubles the lifts (16 per episode) and still meets the bar, with 4 falls in 120 episodes. If its true stand rate were the Wilson lower bound (0.917), a fresh 40-episode panel would reach 36/40 only 77% of the time [arithmetic: out/arith_stdout.txt]. It is the edge, not a margin.
- **The parent's own std (0.007-0.11) and -2.5** give no lift-offs or almost none (0/120 and 1/120 episodes). That matches pilot 1's parent (std 0.095), which stood and never stepped.
- **No setting that stands produces step-like lifts.** Lifts of 2 cm or more appear only once falls begin (4/120 episodes at -1.8125, 7/120 at -1.75, 44/120 at -1.5). The noise level that keeps the parent standing does not by itself show the policy a step. What PPO does with micro-lifts and a backward slide is not measured here. Training could also move log_std: the pilot's warm-up entropy term is 0.02, which pilot 3's own notes say acts on log_std unclipped [doc claim: pilot3/final/criteria_pilot3.txt lines 471-474]. A start at -1.875 would sit at the edge of the falling regime.

## Method

- **Env.** `build_env(ModelId("tyrannosaurus_rex", "anatomical"), "locomotion")` from the export main_4d5f4e7.
  - The forward weight is set through the env's own `set_reward_weight`. Every episode was run at w 0.2, and the return at w 2.0 is NF + 2.0·F from the same episode's term sums.
  - That shortcut was verified: for the first 3 seeds of every run except the statue (108 episodes in 36 runs, plus 2 in the smoke runs), the episode was re-run at w 2.0. In all of them the length, end reason and final state were identical, and the return matched NF + w·F to within 4.5e-13. On every step of every episode, the info reward terms plus the fall penalty equal the returned reward to within 4.5e-13 per episode. [measured: "checks" in out/analyze_stdout.txt]
- **Parent.** `stage1_final.zip` and `stage1_final_vecnorm.pkl` from `.../fetch/stance7_r10LTSBpc_s42/01_stance/models/`, loaded with the export's `load_sb3_checkpoint` (frozen observation statistics, `plant_identity` not passed), as in the REWARD stage.
  - The policy mean is the SB3 actor's distribution mean.
  - log_std is a 15-vector (no gSDE). It was set in memory with `policy.log_std.copy_` and is printed in each JSON's meta.
- **Noise streams.** There are three independent streams. Within a stream, every setting sees the same normal draws (common random numbers).
  - `torch`: `torch.manual_seed(reset seed)` after `env.reset`, then SB3 `predict(deterministic=False)`. This is the REWARD stage's draw: my torch runs reproduce its own-std, -2.0, -1.5 and deterministic records (100 episodes; same length and end, |Δreturn| ≤ 2.1e-12, from summation order).
  - `np`: numpy PCG64(SeedSequence([seed, 20261104])), with a = clip(mean + std·N(0, 1), -1, 1).
  - `np2`: the same rule with key 20261105.
  - The deterministic parent uses `predict(deterministic=True)`. The statue (`zero`) uses action 0.
- **Seeds.** Reset seeds 3042-3081 (40) for every setting and stream.
- **Episodes.** Every episode was driven by the frozen gait probe's own `rollout_episode`, from `docs/investigations/gait_2026_09/gait_probe.py` in the export (sha256 61651b16…; loaded read-only with importlib).
  - That function installs the probe's floor-truth hook in `env._substep_probe_hook` for the episode and restores the previous hook afterwards.
  - Episodes were driven only by `env.reset(seed)` and `env.step(a)`. After each episode, every recorded action was asserted to be finite and inside [-1, 1]; SB3's predict and the numpy path both clip before the call.
- **Floor contact.** A foot is down on a step when its floor normal force exceeds 0.1 N on at least half the substeps (the probe's `floor_` rule). Contacts are then debounced with the probe's own 2-step `debounce`, and the probe's `_events` gives touchdowns and lift-offs.
  - A *swing* is a lift-off followed by a touchdown of the same foot.
  - An *upright* swing ends at least 0.5 s before the end of a terminated episode. Every swing of a full-length episode counts.
  - Clearance follows the probe's definition: the maximum foot height during the swing minus the foot's median height while down.
  - The swing pairing over the whole episode is mine. The probe's own windowed analysis (`analyze_episode`, 1 s settle) gives the gait class and is stored in each episode record.
  - Reset artefact: in the statue, deterministic, own-std and -2.5 runs, 36-38 of 40 episodes show one touchdown with no lift-off (a foot that is first loaded just after reset). That is why "any touchdown" is not used as an exploration measure.
- **Applied std.** Raw applied noise = a − clip(mean), with a the action sent to `env.step`. After the filter: the env's own filter state (`_action_filter_state`, read-only, every step) minus the low-pass of clip(mean) with the env's α. Statistics are pooled over steps 1 to T−1. A check recomputing the filter from a matched the env's state exactly (max difference 0.0).
- **Displacement.** Free-joint xy at the end minus after reset, projected on the env's reset-time prey direction (the REWARD stage's definition). The probe's root-body displacement is stored as `disp_fwd_rootbody`.

## Results table (seeds 3042-3081; per stream torch / np / np2)

Stood = full 1000 steps. Ends = causes of early termination, summed over streams. Swings = upright floor-truth swings per episode (pooled). Disp = mean 10-s displacement along the prey direction over all episodes (pooled), with the sd over full-length episodes. Filt std = applied std after the filter (legs). [measured: out/analyze_stdout.txt, out/extra2_stdout.txt]

| setting (std) | stood | pooled (Wilson 95%) | ends | mean length | return w 0.2 | return w 2.0 | swings/ep | eps with a swing ≥1 cm / ≥2 cm | disp m (sd, full eps) | filt std |
|---|---|---|---|---|---|---|---|---|---|---|
| statue (0) | 40 | 40/40 | — | 1000 | 1094.7 | 1092.6 | 0 | 0 / 0 | -0.029 (0.019) | 0 |
| deterministic | 40 | 40/40 | — | 1000 | 1095.9 | 1093.7 | 0 | 0 / 0 | -0.031 (0.020) | 0 |
| own (0.007-0.111) | 40/40/40 | 120/120 (0.97-1.00) | — | 1000 | 1095.6-1095.7 | 1093.4 | 0 | 0 / 0 | -0.031 (0.020) | 0.003-0.055 |
| u-2.5 (0.082) | 40/40/40 | 120/120 (0.97-1.00) | — | 1000 | 1092.8-1093.3 | 1089.3-1089.8 | 0.01 | 1 / 0 | -0.049 (0.022) | 0.040 |
| u-2.25 (0.105) | 40/40/40 | 120/120 (0.97-1.00) | — | 1000 | 1090.9-1091.4 | 1084.1-1084.5 | 0.70 | 0 / 0 | -0.096 (0.025) | 0.052 |
| **u-2.0 (0.135)** | **40/40/40** | **120/120 (0.97-1.00)** | — | 1000 | 1086.9-1087.5 | 1070.4-1071.2 | 7.33 | 1 / 0 | -0.226 (0.037) | 0.066 |
| u-1.9375 (0.144) | 40/39/39 | 118/120 (0.94-1.00) | fallen 2 | 1000/986/982 | 1085.4/1066.1/1060.9 | 1065.2/1045.4/1040.7 | 11.32 | 3 / 0 | -0.283 (0.043) | 0.070 |
| **u-1.875 (0.153)** | **40/38/38** | **116/120 (0.92-0.99)** | fallen 3, head_contact 1 | 1000/968/966 | 1083.6/1041.4/1038.1 | 1058.9/1018.1/1013.2 | 16.38 | 8 / 0 | -0.338 (0.050) | 0.075 |
| u-1.8125 (0.163) | 38/37/35 | 110/120 (0.85-0.95) | fallen 8, head_contact 2 | 973/953/917 | 1043.0/1018.5/970.2 | 1011.1/989.8/939.8 | 22.04 | 17 / 4 | -0.421 (0.090) | 0.080 |
| u-1.75 (0.174) | 32/31/28 | 91/120 (0.67-0.83) | fallen 20, head_contact 9 | 917/907/853 | 956.0/942.9/870.1 | 921.5/905.7/836.5 | 27.88 | 42 / 7 | -0.488 (0.089) | 0.085 |
| u-1.5 (0.223) | 4/2/3 | 9/120 (0.04-0.14) | head_contact 58, fallen 50, nosedive 2, tail_contact 1 | 534/468/476 | 421.7/347.5/356.2 | 394.6/321.2/339.3 | 26.38 | 97 / 44 | -0.327 (0.343) | 0.109 |
| f-2.0 = u-2.0 | 40/40 (torch/np) | identical to u-2.0 | | | | | | | | |
| f-1.75 = u-1.75 | 32/31 (torch/np) | identical to u-1.75 | | | | | | | | |
| extra f-2.25 | 40/40 (torch/np) | 80/80 | — | 1000 | 1091.0/1091.4 | 1084.1/1084.5 | 0.66 | 0 / 0 | -0.096 (0.024) | 0.052 legs, 0.107 tail |
| extra f-2.5 | 40/40 (torch/np) | 80/80 | — | 1000 | 1093.1/1093.3 | 1089.5/1089.8 | 0.01 | 1 / 0 | -0.049 (0.022) | 0.040 legs, 0.045 neck, 0.050 tail |

Notes on the table:
- **Return spread.** Within a stream, the sd of the return over standing settings is 4.3-7.0. Settings with falls are bimodal (for example sd 276-349 at -1.75). The time of early termination: 2.6-4.3 s at -1.9375; 2.4-4.9 s at -1.875; 1.8-6.5 s at -1.8125; 1.7-9.8 s at -1.75 (means 5.1-5.9 s); 1.5-9.7 s at -1.5 (means 4.3-4.8 s). [measured]
- **Displacement spread over all episodes, per stream** (falls widen it): -2.0 sd 0.034-0.040 m; -1.875 0.050-0.139; -1.75 0.24-0.30; -1.5 0.56-0.61. [measured]
- **Episodes with any upright lift-off** (out of 120): 0 (own), 1 (-2.5), 65 (-2.25), 120 (-2.0 and every larger std). Episodes with upright swings on both feet: 0, 0, 12, 114 (-2.0), 119 (-1.9375), 118 (-1.875), 117 (-1.8125), 118 (-1.75), 119 (-1.5). [measured: out/extra2_stdout.txt]
- **Swing size and order.** Median clearance of upright swings: 1.5 mm (-2.25), 2.0 (-2.0), 2.3 (-1.875), 2.7 (-1.75), 4.0 (-1.5). 99th percentile: 7-9 mm from -2.25 to -1.875, 11 mm at -1.8125, 14 mm at -1.75 and 25 mm at -1.5. The largest at -2.0, -1.9375 and -1.875 is 18 mm. Median swing duration: 0.02 s up to -1.75 and 0.03 s at -1.5. Probe alternation index over full episodes: 0.49-0.52 for -2.0 to -1.5 (0.67-0.87 at -2.25, from only 9-14 episodes with enough events). [measured: out/extra2_stdout.txt]
- **Probe floor gait class.** Own std and -2.5: "stand" 120/120. -2.25: "stand" 106, "stand (stepping / tapping in place)" 14. -2.0: tapping 118, shuffle 1, stand 1. -1.875: shuffle 83, tapping 31, walk 5, staggered scoot 1. -1.75: shuffle 112, walk 5, staggered scoot 2, tapping 1.
- **The probe's "walk" label** appears on 15 of the 480 episodes from -1.9375 to -1.75 (0-3 per stream). In those episodes:
  - the L/R order does alternate more (alternation 0.70-0.85);
  - the swings are as small as elsewhere: median clearance 1.3-3.1 mm, at most 11 mm in the 14 full-length ones; median duration 0.02-0.03 s, at most 0.08 s;
  - every one ends 0.31-0.69 m behind its start.

  [measured: out/extra3_stdout.txt] The label therefore reflects the probe's order rule on micro-lifts during a backward slide, not a stepping gait. [inferred]
- **Falls are not tied to particular reset seeds** in the edge settings. At -1.875 the 4 falls are on 4 different seeds. At -1.8125, 1 seed fell in two streams, against 0.8 expected by chance. At -1.75, 4 seeds fell in two or more streams, against 5.9 expected. [measured: out/extra_stdout.txt; inferred: "not tied", from the overlap count]
- **The filtered command's own std over time** (legs, mean over episodes) is 0.015 for the deterministic parent, 0.019 for own std, 0.050 at -2.5, 0.074 at -2.0, 0.082 at -1.875, 0.091 at -1.75 and 0.113 at -1.5. This includes the policy's closed-loop reaction. [measured]

## Limits

- **Frozen statistics.** The observation statistics are the checkpoint's, frozen. The pilot's training env carries them and then updates them, which changes the policy's inputs over training. Not measured.
- **Fixed policy.** Each setting is a fixed policy with fixed noise. PPO updates the mean and log_std from the first rollout on (and the warm-up's entropy coefficient 0.02 acts on log_std). The stand rates describe the warm start only.
- **Sample size.** Each setting has 40 reset seeds and two or three noise streams (80-120 episodes). Wilson intervals are given. The 36/40 bar was applied per 40-episode stream panel. With pooled rates the pass/fail line falls between -1.875 (0.967) and -1.8125 (0.917).
- **Exploration measures.** Lift-off counts and clearances use the frozen probe's floor rule with my swing pairing over the whole episode. The probe's own windowed counts are in each record (`probe`). Exploration is measured as foot events and end displacement only. Per-step velocity or state-visitation spread, and anything about the advantage signal PPO would see, were not measured.
- **Weights.** The returns at w 0.2 and 2.0 assume the dynamics do not depend on w. That was verified on 108 re-run episodes (plus 2 in the smoke runs). The ramp's intermediate weights are linear in w for every episode (REWARD stage, confirmed by VERIFY_REWARD).
- **Plant identity.** The parent was loaded without a plant-identity check, as in the REWARD stage. VERIFY_REWARD found that the current plant identity accepts this parent.

## Slips (all recorded in CMDLOG.txt)

- **Not logged when they ran.** The first read-only commands (03:53-03:57Z: ls, cat, sed, grep of the REWARD/VERIFY folders, the export's probe, env and filter code) ran before CMDLOG existed or before being logged. They were logged literally after the fact in the NOTE block at the top of CMDLOG.
- **System python3.** One JSON reader (wall times of REWARD's runs) used the system `python3 -I`, without -B, nice, OMP_NUM_THREADS or TMPDIR. No simulation. Logged as SLIP.
- **Unlogged no-op.** One pointless shell no-op (an echo piped to `head -0`, then an echo) ran unlogged at about 04:03Z. It was logged after the fact. It wrote no file.
- **chain2.sh cut off.** chain2.sh was stopped at about 04:52Z by the harness's 30-minute background limit, which I had left at its default. That happened during the `f-2.5 np` run, which wrote no JSON. Its partial log is kept as `out/f-2.5_np_stdout_KILLED.txt` and not used. No process was left running (pgrep at 04:56:35Z). The cut-off runs (f-2.5 np; own, -2.5, -2.25, -1.5 np2) were re-run unchanged in chain3.sh.
- **Analysis while simulating.** analyze.py (a JSON reader, not a simulation) was run five times (04:05-04:46Z) while a simulation chain was running, through r.sh under nice.
- **Not under nice.** The read-only shell commands run through log.sh (cat, sed, grep, ls, sha256sum, find, du, diff, mv) did not run under nice. The Monitor-tool watch, the foreground wait loops and the chain shells did run under nice. The harness wrote its own background-task output files under `/tmp/claude-0/.../tasks/`, outside S. I wrote nothing there myself.
- **Two script versions.** scan.py (chain1) and scan2.py (chain2/3) differ only by the added `zero` setting and `np2` stream. The diff is in `out/scan_vs_scan2_diff.txt`. scan.py was edited once, before chain1, to take displacement from qpos; the smoke version is kept as `tmp/scan_v1_smoke.py`.
- **Torch cache.** Torch created `tmp/torchinductor_root` inside S's TMPDIR.
- **Not done.** No training, no repository, export, physics or checkpoint change, no network, no git writes. No venv was created and nothing was installed. The folder is about 21 MB.

## Files in S

- **Scripts.**
  - `scripts/r.sh`: logs the literal command, then runs it under nice -n 10, OMP_NUM_THREADS=1 and TMPDIR.
  - `scripts/log.sh`: logs a read-only shell command, then runs it.
  - `scripts/scan.py` and `scripts/scan2.py`: the episodes.
  - `scripts/chain1.sh`, `scripts/chain2.sh`, `scripts/chain3.sh`: sequential queues.
  - `scripts/analyze.py`, `scripts/extra.py`, `scripts/extra2.py`, `scripts/extra3.py`: JSON readers.
  - `scripts/arith.py`: arithmetic.
- **Per-run outputs.**
  - `out/<setting>_<stream>.json`: per-episode records. These hold the term sums, NF and F, termination, displacement, per-foot lift-offs, touchdowns and swings with clearance, the probe's class, and per-actuator noise sums. The meta holds the log_std used, the checkpoint log_std, actuator names, α and the w-2.0 re-run checks.
  - `out/<setting>_<stream>_contacts.npz`: per-step floor-truth contact booleans and pelvis xy for every episode.
  - `out/*_stdout.txt`: per-episode lines.
  - `out/det.*`, `out/zero.*`: the deterministic parent and the statue.
- **Tables.**
  - `out/analyze_stdout.txt`: main tables, per-actuator noise, checks, floor-vs-uniform identity, reproduction of the REWARD stage.
  - `out/extra_stdout.txt`: per-seed fall overlap; paired return against the statue.
  - `out/extra2_stdout.txt`: pooled exploration readouts.
  - `out/arith_stdout.txt`: std values and binomial chances.
  - `out/extra3_stdout.txt`: checks of the applied-std ranges, the probe's "walk" episodes and the end displacements.
  - `out/scan_vs_scan2_diff.txt`: the diff between the two scan scripts.
  - `out/summary.json`.
- **Smoke runs:** `out/smoke_*`.
