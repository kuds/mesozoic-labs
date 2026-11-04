# VERIFY observability: independent adversarial check of the OBSERVABILITY stage

Folder: /tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/verify_observability/
Stage checked: /tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/observability/RESULT.md
Worked 01:37-01:52 UTC, container restart, resumed 01:54-02:58 UTC on 2026-11-04 (CMDLOG line 'RESUME after container restart').
CPU only. Nothing was trained or fitted: no RL, no behaviour cloning, no network, no regression. Every number is a
non-parametric nearest-neighbour statistic (1-NN Delta test = conditional-variance estimate, z-scored Euclidean
distances in exact float64, held-out episodes vs a disjoint pool, cluster bootstrap over held-out episodes).
Repository, export and physics untouched. Simulations one process at a time, nice -n 10, OMP_NUM_THREADS=1,
TMPDIR in this folder. Labels: [measured] = computed here; [inferred] = reasoning from code or numbers.

## Verdict: CONFIRMED WITH CORRECTIONS

All four headlines reproduce. On the stage's own recordings my independent code gives the same numbers to the
printed digit (on-cycle and all four noise levels). On fresh seeds the conclusions hold, but several magnitudes and
qualifiers need correcting (below).

| headline | stage seeds, my code | fresh seeds, my code | verdict |
|---|---|---|---|
| 1. O leaves 1.27 % raw / 0.52 % desired; floor 0.16-0.17 % | 1.27 % [0.97, 1.57] / 0.52 % [0.41, 0.63]; floor 0.16 / 0.17 % | 1.06 % [0.76, 1.39] / 0.52 % [0.39, 0.66]; floor 0.19 / 0.21 % | CONFIRMED (fresh raw lower) |
| 2. prev action / filter / phase recover ~nothing on-cycle; O fixes phase and filter | unit gains +0.02 pp, CIs incl. 0 (phase lower bound 0.000); 4x +0.22-0.43 raw, +0.13 des; phase p90 0.0044 cycles, stance 99.82 %, F|O 0.21 % | unit +0.05-0.11 raw, +0.02-0.05 des, CIs EXCLUDE 0 for prev action and filter; 4x +0.23-0.34 raw, +0.07-0.20 des; phase p90 0.0044, stance 99.82 %, F|O 0.29 % | CONFIRMED WITH CORRECTIONS |
| 3. step timer brings residual to floor; 89 % of raw residual at the switch | 1.27 -> 0.19 %, +1.07 pp [0.80, 1.36]; 89 % in 8.8 % + 1.3 % of steps; hip roll 2.8-2.9 %, ankle 1.5-1.6 % | 1.06 -> 0.25 %, +0.81 pp [0.54, 1.11]; 82 % (desired 53 %) in 8.8 % + 1.4 % of steps; hip roll 2.4 %, ankle 1.2 % | CONFIRMED (fresh magnitudes smaller) |
| 4. off-cycle the filter state matters most | identical to stage at sigma 0.1/0.2/0.3 | F|O 12.0/19.0/26.6 %; F gain des +0.90/+1.73/+4.06 pp (4x +2.69/+3.15/+8.33); survival 40/38/12 of 40 | CONFIRMED WITH CORRECTIONS |

## What was checked

1. **Recordings are what the env gives.** [measured] My own recorder (scripts/vrec.py, importing the existence
   stage's wlib.py, sha256 d5b8e6cf... = the stage's copy) re-rolled seeds 0 and 100: observation, raw action,
   desired command, filter state before/after, root height and all walker-memory fields are bitwise equal to the
   stage's pool_s0 / held_s100 (out/vcheck_rep_stdout.txt). Over 200 fresh episodes obs[0:21] = qpos[7:],
   obs[21:42] = qvel[6:], obs[54:56] = the walker's sensed touch forces, max difference 0.
2. **The observation equals the policy's input.** [measured] The training path (train_base.make_env + Monitor) and
   build_env return bitwise-equal 63-value observations over a full 1000-step walker episode; SB3 VecNormalize with
   the repo defaults (norm_obs True, clip 10, eps 1e-8) equals the hand z-score to 4e-7 (out/vcheck_env_stdout.txt).
   The locomotion stage does not enable the critic-phase wrapper. Code (_get_obs, export): qpos 21, qvel 21,
   projected gravity 3, gyro 3, heading-frame velocity 3, **pelvis accelerometer 3 (dims 51-53; not in the task
   text)**, foot forces 2, prey direction 3 + distance 1, command 3 (zeros). No previous action, filter state, phase or
   root height: confirmed. Dropping the accelerometer changes F|O only 0.21 -> 0.22 %, so it is not what fixes
   the filter state on-cycle.
3. **Independent re-analysis of the stage's data** (scripts/vana.py; out/A_stage_data.json, out/vsum_A_stdout.txt)
   and of the noise data (out/C_stage_noise_s00..s30.json, out/vsum_C*_stdout.txt). [measured]
4. **Fresh rollouts, new seeds** [measured]: on-cycle pool 500-559 + held 600-619 (80/80 full episodes;
   out/B_fresh.json, out/vsum_AB_stdout.txt); noise sigma 0.1/0.2/0.3 with pool 500-529 + held 600-609 and an
   independent noise stream (out/D_fresh_noise_s*.json, out/vsum_D*_stdout.txt).
5. **Standardisation** [measured]: z-scoring as VecNormalize does is what the stage used. Alternatives move O little:
   pool incl. settle steps 1.38 / 0.56 %, median/IQR 1.24 / 0.50 % (stage seeds). Raw units: 1.46 / 1.13 % (fresh
   1.75 / 1.38 %). Clip at 10 touches 2.0e-5 (fresh 2.5e-6) of values; dims 60-62 constant.
6. **Sham controls** [measured]: permuted blocks (rows permuted separately in pool and held) and N(0,1) blocks behave
   as the stage says. I added a redundant sham: a copy of 8 observation dims (leg joint angles), i.e. no information
   beyond O. It gains +0.04 / +0.07 pp raw at 1x (stage / fresh seeds) and +0.42 raw / +0.24 desired at 4x (fresh),
   and beats its own permuted control by +0.9 / +1.7 pp raw. So the 1x and 4x gains of previous action, filter state
   and phase on-cycle are the size of pure metric reweighting, and beating a permuted control shows target relevance,
   not information beyond O.

## Corrections

1. Headline 2: "+0.02 pp each at unit weight (CI includes 0)" holds on the stage's seeds only. On fresh seeds the
   unit-weight gains are +0.11 (prev action), +0.06 (filter), +0.05 (phase) pp raw and +0.05 / +0.04 / +0.02 pp
   desired, with CIs excluding 0 for prev action and filter. Still negligible (the redundant sham gives the same).
   "+0.2-0.4 pp at 4x" is the raw action (phase x4 is +0.43 on stage seeds); for the desired command it is +0.13
   (stage) and +0.07-0.20 (fresh). [measured]
2. Headline 4: in the noise runs "previous raw action" is the previous APPLIED action (walker raw + noise, clipped;
   rec_noise.py sets aprev = a_app). That is the right analogue for a policy, but it is not the raw action. [measured: code]
3. Headline 4: "the phase adds at most 0.4 pp" holds at unit weight. At 4x it adds +0.47 / +0.59 / +1.14 pp desired
   (stage seeds; CI excludes 0 at sigma 0.1-0.2) and +0.38 / +0.45 / +0.47 (fresh). [measured]
4. Headline 4: "the filter state becomes the input that matters most" holds for the desired command among inputs a
   policy could get. At sigma 0.1 the walker's step timer (+2.0 / +2.3 pp) beats the unit-weight filter state (+0.9).
   For the RAW action, which is what a policy outputs, the previous applied action recovers more than the filter
   state at sigma 0.1-0.2 on both seed sets (+2.4 / +2.3 vs +0.8 / +1.2 pp stage; +2.4 / +2.3 vs +0.5 / +1.0 fresh),
   and filter x4 gives +0.6 / -0.6 pp (stage) and +1.5 / -0.8 pp (fresh), still below the previous applied action
   (raw oracle floors 17-34 %, so coarse). The key result "the data favour the
   filter state over the previous raw action" needs "for the desired command". [measured]
5. Headline 4 mechanism [inferred from wlib.py lines 386-394 + measured]: the walker computes its desired command
   without its filter estimate (f_state enters only the inversion). So the filter state's gain on the desired command
   is a proxy for the walker's recent commands, not a dependence of the command on the plant's filter. The walker's
   previous desired command (walker memory, not available to a policy) recovers more: +1.1 / +2.8 / +6.5 pp (stage),
   +1.3 / +2.6 / +5.8 pp (fresh), 4x +3.7 / +7.3 / +14.8 and +4.2 / +6.3 / +12.0.
6. Headline 4 magnitudes on fresh seeds: F undetermined 12 / 19 / 27 % (stage 12 / 21 / 30); F gain at 4x
   +2.7 / +3.2 / +8.3 pp (stage +2.2 / +3.9 / +11.5); prev applied action +0.05 / -0.15 / +0.64 pp (stage +0.1 / -0.4
   / +2.5); survival 40/40, 38/40, 12/40 (stage 10/40). The "+2.5 pp at sigma 0.3" for the previous action did not
   replicate in size (+0.6). [measured]
7. Key result "raw-unit features leave 1.56 % / 1.26 %": a float32 artefact. The stage's knn() orders candidates by
   float32 |q|^2 + |r|^2 - 2 q.r; with touch forces up to 1149 N this cancels badly. Same data, exact float64:
   1.46 % / 1.13 %; the stage-like float32 code reproduces 1.56 / 1.26 (out/vraw32_stdout.txt). The z-scored results
   are unaffected (identical in float64). Direction (z-scoring helps, mostly for the desired command) holds. [measured]
8. Key result "in the late-touchdown bin hip roll is 81-86 % unexplained, ankle 47-48 %": these are the bin residuals
   divided by each actuator's whole-pool variance, not a share of the bin's variance. Relative to the bin's own
   variance (stage seeds) hip roll is 0.38-0.70 and ankle 2.9-3.4 (residual larger than the bin variance). Fresh: 0.64-0.70 and
   0.36-0.40 of pool variance. [measured]
9. Key result pool scaling: on fresh seeds the excess over the floor falls 1.29 -> 1.23 -> 0.87 pp raw (15 / 30 / 60
   episodes; stage seeds 1.20 / 1.29 / 1.11). The scaling argument is weaker than stated. The stronger evidence for
   genuine aliasing is that the step timer, or same-control-step stratification (S:O 0.16 % stage / 0.21 % fresh),
   removes it. [measured]
10. Key result Markov "98.3 % determined on-cycle": reproduced (residual 0.0170) only with the noise script's
    30-episode pool (seeds 0-29 / 100-109); with the 60-episode pool it is 0.011 (stage seeds) / 0.012 (fresh).
    Off-cycle Markov numbers were not re-run. [measured]
11. Stage RESULT.md "raw action clipped on 0.30 % of walking steps": 0.30 % of leg-actuator values; 2.4 % of walking
    steps have at least one clipped leg actuator (both seed sets). [measured]
12. Permuted-control margins depend on the random permutation (my draws: +0.3 to +1.1 pp raw on stage seeds,
    +0.7 to +1.7 fresh, against the stage's 0.5-1.2). Phase-only (no O) differs (0.074 vs 0.051 raw), most likely from
    tie-breaking among identical 2-dim phase values; not a headline. [measured]

## Labels

Stage labels are mostly right: the measured numbers are measured and reproduce, the code fact about the filter
(f_t = 0.614 f_(t-1) + 0.386 a_t, alpha = dt/(RC + dt) = 0.3859, environments/shared/action_filter.py and
base_env._filter_action, which clips first) is correct, and the background section is marked as unchecked. Issues:
the filter-state recommendation is labelled [inferred] but lacks "desired command only" (correction 4); the
late-touchdown percentages are mis-described (8); the Markov 98.3 % omits its 30-episode pool (10); the raw-unit
numbers are measured but carry the float32 artefact (7). The stage's own rule slips were not audited beyond reading.

## Rule slips (mine)

* First attempt (01:37-01:52Z): the first two shell calls were logged after the fact with approximate times. At
  01:39:56Z a grep over the stage's CMDLOG, RESULT.md and scripts used the device-directory path prefix as its
  pattern, to check the stage's compliance. It names no device, but I report it in case the rule covers that prefix.
  Its output was not saved and the check was not repeated.
* After the restart, two read-only commands (date and ls; cat CMDLOG and ls) ran before the RESUME line; the line
  names them.
* File edits to scripts/vana.py were made with the Edit tool and with venv-python heredoc string-replaces run
  directly (logged as NOTEs, not literally). scripts/vsum.py was edited with the Edit tool. A py_compile check
  created scripts/__pycache__, removed afterwards.
* Wait loops and quick read-only tails were run directly; most were logged literally before running. The 02:19:19Z
  wait-loop entry did not match the command actually run; a later NOTE gives the real text.
* The fresh on-cycle recording job ended with EXIT 1 from its final two-file 'tail -2' (invalid syntax); all 80
  recordings completed.
* Between 02:22Z and 02:43Z two analysis processes, and for six minutes one recorder, ran at the same time. Only one
  simulation ran at any time.
* No repository, export or physics change; no git; no network; nothing trained or fitted; folder 132 MB.

## Files

* scripts/: L.sh (logging wrapper), vrec.py (recorder), vcheck_rep.py, vcheck_env.py, vana.py (analysis), vsum.py
  (side-by-side), vraw32.py (precision check).
* out/: rep/ (re-rolled seeds 0, 100), fresh/ (seeds 500-559, 600-619), fnoise/ (sigma 0.1-0.3, seeds 500-529,
  600-609); A_stage_data.json, B_fresh.json, C_stage_noise_s00-30.json, D_fresh_noise_s10-30.json and their
  *_stdout.txt; vsum_*_stdout.txt; vcheck_*_stdout.txt; vraw32_stdout.txt; sha256_wlib.txt. smoke.json and
  smoke2.json are development runs, superseded.
* Kept from the first attempt (complete and valid): out/rep/, vcheck_rep_stdout.txt, vcheck_env_stdout.txt and the
  scripts. Its partial A_stage_data run was redone in full.
