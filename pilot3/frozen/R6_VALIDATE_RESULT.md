# Walking pilot 3, R6 VALIDATE stage (2026-11-04, 14:50-16:05 UTC)

Stage folder: `SP/work/pilot3/r6_validate/` (SP = `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad`).
Branch head validated: `a3f1b378aede66b0ee8630197f36a7c374ec77c0` (local `claude/pilot-walk-r10-p3`, three new pilot-only
commits b7a146d, 4dfc7cb, a3f1b37 on 3ae1c959; `SP/wt_p3` clean before and after, never written).
Labels: [measured: file] = read from a file this stage wrote or a command it ran; [measured as recorded] = an earlier
stage's record; [inferred] = reasoning. Every command is in `CMDLOG.txt` (UTC BEGIN/END lines; NOTE lines for harness
file writes). CPU only, one simulation or training process at a time, `nice -n 10`, `OMP_NUM_THREADS=1`, TMPDIR and
every cache inside this folder. Nothing was pushed, committed, or written outside this folder (side effects: last section).
The only training was the pre-registered local CPU smoke (two QUICK_TEST runs, one resumed). No physics, config or
repository change.

## Short answers

1. **The env's own R6 equals SCREEN's offline R6p, and SCREEN's ranking holds on the env's own returns.** [measured:
   out/v2/analysis_stdout.txt, out/v2/A_*.json]
   - 16 behaviours (statue, parent deterministic, W02, the six stable walks FINAL3/V01/V05/V24/V18/V35, the matched hops
     HF020/HF030/HF050/H23/H24/HF120 and H20) x seeds 3042-3061 = 320 episodes, 319,673 steps, re-rolled in the
     head's R6 locomotion env through the frozen probe's rollout.
   - Every trajectory is bitwise SCREEN's (forward_vel, foot sites, quaternion, root, floor fractions, touch forces,
     actions: 320/320). The reward does not enter the dynamics.
   - Against SCREEN's published R6p (out/screen_main.json), 634 of 640 episode-weight pairs agree within 1e-9 (max
     4.5e-13). The other 6 are W02 seeds 3044, 3052 and 3054 at both weights: SCREEN pays a touchdown at k = T-4 on the
     last step (0.2469, 0.2477, 0.2469 at w_td 0.1), which the env, paying at k+4, never pays. With those removed:
     640/640 within 4.5e-13.
   - Per step, the env's `walk_touchdown_unit` equals SCREEN's `events_probe` ported verbatim with pay delay 4 on all
     319,673 steps, on the env's own touch flags and on the probe's floor flags alike. Touch and floor flags disagree on
     0 of 639,346 foot-steps in these behaviours.
   - The 30 live weight-0.2 episodes equal the offline forward-term swap of the weight-2.0 episodes to 2.3e-13.
   - Ranking, from the env's own returns: every stable walk is above the best standing behaviour and above its matched
     hop on every seed at both w 0.2 and w 2.0 (120/120 walk-seed pairs in each of the four tests). The worst margins
     equal SCREEN's R6p row to the printed digit: +104.3 (V24; seed minimum +93.2) and +980.0 (V24-H23; +969.1) at
     w 0.2; +332.4 (FINAL3; +309.6) and +900.7 (V35-HF120; +823.3) at w 2.0.
   - Against the 2.0 m/s hop H20 (SCREEN's item 5): every walk is ahead at w 0.2; at w 2.0 the 0.5-1.2 m/s walks are
     ahead on 20/20 seeds, while FINAL3 (-188.8) and V01 (-84.3) trail it (1/20 seeds each), as SCREEN reported.
2. **Stance is untouched.** [measured: out/stance/cfg_compare.txt, out/digest/, out/stance/compare_stdout.txt]
   - The stance, recovery and behavior stage configs (load_stage_config, canonical JSON) and their task, gate, recipe
     (PPO/SAC) and recorded-view (PPO/SAC) digests are identical at 3ae1c959 and the head; so are the recorded
     reward_weights. Only locomotion's moved: task 5f835d65 -> 782d95b6, recipe 8263f82f/a958a8fb -> 85881fc4/05cce74d,
     view 08c92901/0442ad41 -> c0f35553/cb47b068; its gate 02602cb0 is unchanged.
   - `digest_snapshot --block-optional-backends --check` at the head: current, 649 lines, 0 errors. The golden's diff
     3ae1c959 -> head moves no stance, recovery, behavior-stage, plant or policy line.
   - Bit-exact reward captures (`digest_snapshot --exact`) of both trees: every T. rex stance, recovery and behavior-stage
     line and every other species' line identical (36,848 lines). What moved is locomotion (128 moved, 108 new lines) and
     the 11 T. rex behavior recipes (they inherit locomotion's [env]; commit 4dfc7cb's message says so).
   - The s42 parent's deterministic 40-episode stance panel (seeds 3042-3081): per-step digests (observation, action,
     reward, flags, every info item, qpos, qvel, filter state) identical on 40/40 episodes at the head and at 3ae1c959,
     the same panel digest, 40/40 full-horizon. The VARIANTS stage's own panel script, run unchanged at the head,
     reproduces its main-4d5f4e7 record (overnight/variants/out/stance_policy40_V0.json) in every field of every episode
     (40/40). No R6 info key appears in the stance env; its R6 switches read 0.0 and 0.0 and it builds no tracker.
3. **The parent at log_std -2.0 stands 40/40 under R6.** [measured: out/p3/summary.txt, out/p3/u-2.0_torch_R6.json]
   - Stochastic (the LOGSTD stage's torch stream), seeds 3042-3081, head R6 env: 40/40 episodes run 1,000 steps
     untermined. All 40 trajectories are bitwise the LOGSTD stage's -2.0 torch run on main (length, end, final qpos and
     root, per-step floor contacts and pelvis xy).
   - The probe labels all 40 "stand (stepping / tapping in place)".
   - R6 return: 1083.3 at w 0.2 (range 1064.8-1092.2) and 1055.8 at w 2.0. Main's reward on the same episodes gave
     1086.9 and 1070.4, so R6 costs -3.6 and -14.6. The noisy parent slides back about 0.23 m, and the forward term is
     5/3 steeper under the 1.5 cap.
   - The touchdown term pays it 1.50 per episode (unit 15.0; 1.4 paid of 7.8 registered touchdowns; 33/40 episodes paid
     something).
   - The support gate fires on 4.1 steps per episode (38/40 episodes), forfeiting 2.05 alive. The airborne term costs
     0.57.
   - For scale: the statue under R6 earns 1094.4 at w 0.2 and the deterministic parent 1095.8 (part 1).
4. **The pre-launch smoke re-ran as the smoke stage ran it and passes everything but the pinned judge.** That judge
   refuses every step because it hard-codes -1.5. [measured: out/smoke_evidence_launch_s42.*, nb/, judge/logs/,
   judge_diag/logs/]
   - Notebook smoke: exit 0, the same cells, guard and override, hyperparameters_sha256 85881fc4 before and after.
   - The two Warm start lines on stdout 91-92 (after the VecNormalize line 89 and its rule 90, before the first Eval
     line 93, not on stderr). The first is string-equal to the frozen text. The second reads
     `Warm start: policy log_std set to -2 (std 0.135) on all 15 actuators (the parent's std: mean 0.057, min 0.007, max 0.111; Adam moments kept)`.
   - stage_config.json passes 22/22 checks: the smoke's 17 with -2.0, plus the four R6 reward_weights and the
     unchanged gate digest.
   - First Eval: `Eval num_timesteps=32768, episode_reward=1095.25 +/- 3.93`.
   - The ledger has every column pilot3_habit.py reads: 0 alarms, TD 0.0093.
   - RESUME: no Warm start line, both -2.0 records kept, resume path and sha recorded, run.timesteps 17,232.
   - The pinned `pilot3_judge.py` refuses all ten validation steps, writing nothing. The cause in every case is
     `pilot3_judge.py:85 WARM_START_LOG_STD = -1.5`, enforced at lines 206-211: "run.warm_start_log_std is -2.0, not
     -1.5; curriculum.warm_start_log_std is -2.0, not -1.5". The head is not hard-coded (--expect-head), and the recipe
     digest is derived from the tree.
   - A diagnostic copy with only that constant at -2.0 (not pinned) reproduces the smoke's ten outcomes exactly: J1,
     J7 and J10 accept with 40/40 STILL; J2-J6, J8 and J9 refuse, each for its own reason only.
   - `pilot3_noise.py:39` (`WARM_START_STD = math.exp(-1.5)`) prints a stale reference (0.2231). It does not refuse.

## 1. SCREEN's behaviours in the head's R6 env

**How.** `scripts/v2run.py` (one process per behaviour, `bin/sim.sh`):
- Env and probe:
  - The env is the head tree's (`judge_tree`, a clean --shared clone at a3f1b378)
    `build_env(ModelId("tyrannosaurus_rex","anatomical"), "locomotion")`.
  - Its R6 settings are asserted: forward_vel_max 1.5, airborne_substep_weight 1.0, walk_support_gate_bw 0.04,
    walk_touchdown_weight 0.1, floor-impact and lateral-lean weights 0; BW 840.9132 N; L 0.8766 m; stride scale
    0.08766 m.
  - Episodes run through the frozen probe's `rollout_episode`. Its sha256 61651b16 is the same as main 4d5f4e7's.
- Controllers:
  - Unchanged copies of the HOP stage's `hoplib/wlib/wlib2/defaults`. Their sha256 equal SCREEN's copies, the
    EXISTENCE stage's wlib.py and the SPEED stage's wlib2.py [measured: out/v2/ctrl_sha256.txt].
  - Parameters are read in place from `overnight/speed/out` and `overnight/hop/out`.
- Parent: the branch's `pilot_checkpoints/stance7_r10LTSBpc_s42_final` (sha256 1c9d87c3, the same bytes as SCREEN's
  fetch copy), deterministic.
- Safety: every action was finite and inside [-1, 1], and no controller call changed the env state (0 writes in 320
  episodes).
- Offline analysis: `scripts/v2analyze.py`, no simulation.

**In-run checks, summed over 320 episodes (319,673 steps)** [measured: out/v2/analysis_stdout.txt]:

| check | result |
|---|---|
| trajectory bitwise SCREEN's record (forward_vel, foot_pos, quat, root_pos, floor_frac, touch_min, float32 actions, length) | 320/320 episodes |
| step reward = env-order sum of its recorded terms (29 terms, airborne, touchdown, fall), bitwise | 319,673/319,673 steps |
| reward_forward = w clip(v/1.5); reward_airborne_substeps = -airborne/5 | all steps, bitwise |
| walk_support = 1[r_foot_contact + l_foot_contact > 0.04 BW] | all steps |
| reward_alive, posture, height = SCREEN's ungated recorded value x walk_support | all steps, bitwise, each term |
| every other (non-forward, non-gated) term = SCREEN's recorded value; r/l_foot_contact = SCREEN's | all steps |
| env's r/l_foot_down = touch block majority rule recomputed here | 639,346/639,346 foot-steps |
| env's touch flags vs the probe's floor flags | 0 disagreements |
| walk_touchdown_unit = SCREEN's events_probe ported, pay delay 4 (touch flags; floor flags) | 319,673 and 319,673 steps |
| reward_walk_touchdown = 0.1 x unit | all steps |

**Returns against SCREEN's R6p.**
- 640 episode-weight pairs: raw max |env - SCREEN| = 0.2477 (W02 only). 634 of 640 are within 1e-9 (round-off, max
  4.5e-13).
- SCREEN pays touchdowns at k = T-4, delivered on the last step (W02 seeds 3044, 3052, 3054: -0.2469, -0.2477, -0.2469
  at both weights). After removing those pays, 640/640 agree within 4.5e-13.
- No stable walk, hop, statue or parent episode has a k = T-4 pay on these seeds.
- Live w 0.2 (statue, parent, V24, H23, V35, H20 x 3042-3046, through `set_reward_weight`): 30/30 equal the offline swap
  (max 2.3e-13), with the same lengths and the same touchdown sums.

**Ranking (SCREEN's verdict test on the env's own returns; standing = best of statue, parent, W02 per seed; the parent
is the best on 19 seeds, the statue on 1):**

| source | w | worst walk vs best standing: mean (walk) | per-seed min | won | worst walk vs matched hop: mean (pair) | per-seed min | won | every walk ahead on every seed? |
|---|---|---|---|---|---|---|---|---|
| env R6 (head) | 0.2 | +104.3 (V24) | +93.2 | 120/120 | +980.0 (V24-H23) | +969.1 | 120/120 | YES |
| env R6 (head) | 2.0 | +332.4 (FINAL3) | +309.6 | 120/120 | +900.7 (V35-HF120) | +823.3 | 120/120 | YES |
| SCREEN R6p (published) | 0.2 | +104.3 (V24) | +93.2 | 120/120 | +980.0 (V24-H23) | +969.1 | 120/120 | YES |
| SCREEN R6p (published) | 2.0 | +332.4 (FINAL3) | +309.6 | 120/120 | +900.7 (V35-HF120) | +823.3 | 120/120 | YES |

Per walk at w 0.2 / 2.0 (mean; seed minimum) [measured: out/v2/analysis_stdout.txt]:

| walk | vs standing | vs matched hop | vs H20 |
|---|---|---|---|
| FINAL3 | +104.4 (+97.8) / +332.4 (+309.6) | HF020 +994.1 (+986.7) / +977.2 (+963.2) | +1015.6 / -188.8 (1 of 20 won) |
| V01 | +106.2 (+93.2) / +436.9 (+390.9) | HF030 +989.3 (+971.8) / +982.9 (+955.2) | +1017.4 / -84.3 (1 of 20 won) |
| V05 | +132.3 (+113.3) / +653.0 (+576.2) | HF050 +1013.9 (+1001.8) / +1006.4 (+971.3) | +1043.5 / +131.8 (+41.7) |
| V24 | +104.3 (+93.5) / +778.0 (+708.7) | H23 +980.0 (+969.1) / +938.5 (+909.5) | +1015.5 / +256.8 (+174.3) |
| V18 | +134.6 (+118.4) / +1041.8 (+952.0) | H24 +1001.7 (+980.1) / +906.5 (+862.7) | +1045.8 / +520.5 (+417.5) |
| V35 | +141.4 (+122.6) / +1199.8 (+1072.4) | HF120 +1003.9 (+987.4) / +900.7 (+823.3) | +1052.6 / +678.5 (+537.9) |

Mean return per episode, env R6 at w 0.2 / 2.0 (the same as SCREEN's R6p to two decimals except W02, -0.04):
- standing: statue 1094.43 / 1091.40; parent 1095.76 / 1092.44; W02 1038.37 / 1062.83;
- walks: FINAL3 1200.23 / 1424.85; V01 1201.97 / 1529.39; V05 1228.07 / 1745.46; V24 1200.08 / 1870.49;
  V18 1230.40 / 2134.25; V35 1237.20 / 2292.22;
- hops: HF020 206.17 / 447.69; HF030 212.69 / 546.51; HF050 214.17 / 739.08; H23 220.05 / 932.03; H24 228.73 / 1227.79;
  HF120 233.25 / 1391.48; H20 184.59 / 1613.70 (one H20 fall, seed 3056 at step 673, as SCREEN's).

What R6 pays and charges, per episode:
- Touchdown unit sums: the walks 900-928, W02 81.8, everything else 0.
- Unsupported steps: the walks 0-19.1, the hops 505-568.
- Gate forfeit: the walks 0-14.7, the hops 388-450.
- Airborne term: the walks 0 to -2.5, the hops -486 to -523.

[measured: out/v2/A_*.json]

## 2. Stance untouched

- **Configs and digests** [measured: out/stance/cfg_compare.txt, cfg_head.json, cfg_3ae1c959.json]:
  - `scripts/cfgdump.py` ran on each tree in its own process. The 3ae1c959 tree is the implement stage's export
    `r6_impl/tree_3ae1c959`, used read only after checking it against git: its 596 files equal the git blobs of
    3ae1c959, with no extra file and no missing environments/, configs/, pyproject or conftest file
    [measured: out/stance/impl_tree_3ae1c959_verify.txt]. Its file list was unchanged after every use.
  - For stance, recovery and behavior, these are identical at both trees: the canonical JSON of load_stage_config,
    env_kwargs, task_sha256 (stance 533fff20...), gate, hyperparameters PPO/SAC, the recorded stage_config view PPO/SAC
    (written by the real save_stage_config) and the recorded reward_weights.
  - The head adds the two keys to OPT_IN_TASK_KWARGS with default 0.0, so no stage that leaves them off records them.
- **Generated files at the head** (through iso.py on judge_tree) [measured: out/digest/]: plant_contract --check
  "Plant manifest is current"; species_catalog --check current; digest_snapshot --block-optional-backends --check
  "current: 649 lines", 0 errors.
- **Bit-exact reward captures** [measured: out/digest/exact_compare.txt]:
  - `digest_snapshot --block-optional-backends --exact` ran at the head and at 3ae1c959 (37,244 and 37,016 lines,
    0 errors each).
  - Identical: every T. rex stance, recovery and behavior-stage line and every line of the other species.
  - Moved: T. rex locomotion (128 moved, 108 new: the R6 info keys) and the 11 T. rex behavior recipes (2-4 moved,
    6-12 new each).
- **The s42 parent's deterministic stance panel** [measured: out/stance/compare_stdout.txt, hash_head.json,
  hash_3ae1c959.json, stance_policy40_V0_head.json]:
  - (a) `scripts/stance_hash.py` ran once per tree. Its per-step digests are identical on 40/40 episodes; panel digest
    b14a4ad70158171c... at both trees; 40/40 full-horizon; reward sums 3720.20-3790.84 (mean 3774.83).
  - (b) The VARIANTS stage's `stance.py` (sha256 8e93b349, copied unchanged) ran at the head with the same checkpoint
    paths. It equals that stage's record on main 4d5f4e7 in every field of 40/40 episodes and in its top-level fields
    (wall time aside).
  - The head's stance env has walk_support_gate_bw 0.0, walk_touchdown_weight 0.0, `_walk_touchdowns` None and no R6
    info key on any step.

## 3. The parent at log_std -2.0 under R6

**How.** `scripts/p3run.py` reproduces the LOGSTD stage's `scan.py` "torch" stream:
- The parent and frozen normaliser load through load_sb3_checkpoint. Every log_std is set to -2.0 in memory, and
  nothing is saved.
- `torch.manual_seed(reset seed)` runs right after reset inside the probe's rollout. The policy mean is computed (no
  draw), then `predict(deterministic=False)` draws the action.
- Seeds 3042-3081 at w 0.2. Seeds 3042-3044 were also run live at w 2.0, and NF + 2F matched to 2.3e-13 with the same
  length and final qpos.

**Results** [measured: out/p3/summary.txt]:
- Stood (1,000 steps, no termination): 40/40. Every trajectory is bitwise the LOGSTD stage's u-2.0 torch record on main,
  across all seven compared fields.
- Probe: "stand (stepping / tapping in place)" 40/40. Displacement -0.23 m (-0.30 to -0.15).
- Return, R6: 1083.33 at w 0.2 (1064.76-1092.21) and 1055.76 at w 2.0. Main's, same episodes [measured as recorded:
  overnight/logstd/out/u-2.0_torch.json]: 1086.90 and 1070.36.
- Touchdown term: 1.50 per episode (unit 14.99, 0-50.7). Registered 7.8, paid 1.4. 33/40 episodes have a pay.
- Support gate: 4.1 unsupported steps per episode (0-8; 38/40 episodes), alive forfeited 2.05 (0-4).
- Airborne term: -0.57 (0 to -4.0; 2.85 airborne substeps).
- [inferred] At the ramp start the noisy -2.0 parent scores about 11 below the R6 statue (1094.4) and 12 below the
  deterministic parent. The touchdown pay its micro-steps earn (about 1.5) is smaller than what its backward slide costs
  on the steeper forward scale.

## 4. The pre-launch smoke, re-run

The same steps as `SP/work/pilot3/smoke/`, at the new head:
- Hand-made venv (pyvenv.cfg, a python symlink, a .pth to the optB and venv_sb3 site-packages; no venv tool, no
  activation scripts): Python 3.13.18, torch 2.13.0+cpu, numpy 2.5.4, SB3 2.9.0, MuJoCo 3.10.0, gymnasium 1.4.0,
  tensorboard 2.21.0, IPython 9.18.1.
- Two --shared clones (`clone` and `judge_tree`) at a3f1b378, both clean at the start.
- The smoke's params JSONs, copied byte-identical (sha256 6d787662, 7c4f81b4); the pinned runner nb_pilot_runner3.py
  (7f533128); and the private Xvfb wrapper.
- Pinned tools: `sha256sum -c` of all 73 SHA256SUMS entries OK at the start and at the end, with identical output.

| item | at a3f1b378 (this stage) [measured] | the smoke at 3ae1c959 (-1.5) [measured as recorded] |
|---|---|---|
| notebook smoke (launch_s42), cadence override 32,768 | exit 0, 14:58:38-15:07:55; cells setup 3, config 6, resume 18, chain 20, manual 22, disconnect 32; guard stopped Run all at 20; manual 22 (437.8 s) then 32 | exit 0; same cells |
| override | eval/checkpoint 131,072/500,000 -> 32,768/32,768; hyperparameters_sha256 85881fc4 before and after | 8263f82f before and after |
| Warm start line 1 | string-equal to the frozen text, stdout 91 | same, 91 |
| Warm start line 2 | `... log_std set to -2 (std 0.135) on all 15 actuators (the parent's std: mean 0.057, min 0.007, max 0.111; Adam moments kept)`, stdout 92, not on stderr; order VecNormalize 89, rule 90, first Eval 93 | `-1.5 (std 0.223)`, 92 |
| stage_config.json | 22/22: load_mode initialize_next_stage; parent 1c9d87c3; critic_phase_dropped = the parent's record; run and curriculum warm_start_log_std -2.0; n_envs 4; no critic_phase key; eval_freq/checkpoint_freq 32,768, eval_episodes 10, max_checkpoints 0; git_commit = head; hyperparameters_sha256 = snapshot 85881fc4; task_sha256 = snapshot 782d95b6; reward_weights forward_vel_max 1.5, airborne_substep_weight 1.0, walk_support_gate_bw 0.04, walk_touchdown_weight 0.1; gate 02602cb0; seed 42; timesteps 50,000; stage 2 locomotion | 17/17 (-1.5, 8263f82f, 5f835d65) |
| first Eval | `Eval num_timesteps=32768, episode_reward=1095.25 +/- 3.93`; 65,536 at 1092.31 +/- 3.83; gate FAIL (forward vel -0.00) and "Result bundle not rewritten" printed | 1096.43 +/- 3.12; 1094.29 +/- 2.78 |
| ledger eval_episodes.csv | 192 columns (header byte-identical to the smoke's), 10 rows per evaluation, every pilot3_habit.py column; habit ledger: 0 alarms, TD 0.0093 | 192, 0 alarms, TD 0.0094 |
| TensorBoard | child's own 03_locomotion/tensorboard/PPO_1; 240 scalar tags = the smoke's 244 + `reward_terms/eval_walk_touchdown` (0.0 at both evaluations) - 5 fall/termination tags never emitted (no training fall at -2.0); train/std 0.1362, 0.1366, 0.1367; value_loss 0.084 -> 0.009; explained variance 0.04 -> 0.26; every archive's tensorboard_log is the child's | 244 tags; train/std 0.2245-0.2252 |
| checkpoints | pairs at 32,768 and 65,536, final at 65,536; exp(log_std) mean 0.1362 / 0.1367 / 0.1367 | 0.2245 / 0.2252 |
| replay | stage2_final.mp4 h264 640x480 100 fps 1,000 frames 10.0 s, sha256 c00681d8...; provenance commit a3f1b378, not dirty | 4fbab5c8... |
| RESUME (rtest) | stopped after the 32,768 pair (exit 75); its policy.pth and optimizer bitwise the unbroken run's; RESUME cell exit 0 (15:11-15:15): "Checkpoint steps: 32,768 of 50,000 - remaining budget: 17,232", 0 Warm start lines on stdout and stderr; kept critic_phase_dropped and both -2.0 records, recorded resume_load_path and resume_checkpoint_sha256 (b00e494e), timesteps 17,232, load_mode initialize_next_stage, head; ended at 65,536 with a pair and a final model; Eval 1095.54 against the unbroken 1092.31 (not bitwise, as before) | the same |
| pilot3_noise.py (pinned) on both runs | exit 0; prints "warm start 0.2231" as its reference (stale; pilot3_noise.py:39 and docstring :8); forward term -2.955 at 32,768 and -5.029 (unbroken) / -2.700 (resumed) at 65,536, on the new 1.5 m/s scale | -1.78, -3.06 |
| parent probe (gait_probe_pilot.py through iso.py, 40 episodes, as the smoke's command) | 40/40 STILL, floor and touch class stand 40/40, lengths 1,000, displacement speed at most 0.0023 m/s, 0 touchdowns, floor duty 1.00/1.00, L 0.8766 m, policy interface r15 | 40/40 STILL |

**pilot3_judge.py, validation only, on judge_tree (--expect-head a3f1b378)** [measured: judge/logs/rcs.txt, J*.log]:
- Every step exits 2 and writes nothing. Tree status stays 0 entries and the tool hashes are unchanged.
- The reason, in every step:
  - `run.warm_start_log_std is -2.0, not -1.5; curriculum.warm_start_log_std is -2.0, not -1.5`
  - J4 reads `run.warm_start_log_std is absent (not -1.5); curriculum.warm_start_log_std is -2.0, not -1.5`.
- Where it is hard-coded:
  - `SP/work/pilot3/tools/pilot3_judge.py` line 85 `WARM_START_LOG_STD = -1.5`, used at lines 206-211, and the
    docstring line 39.
  - J1, J7 and J10, which the smoke accepted, are refused for this reason alone. J2-J6, J8 and J9 are refused for their
    own reason plus this one.
- Nothing refuses because of the head or the recipe digest. The head is a command-line argument, and the digest is
  derived from the tree (85881fc4 matched).

**The same ten steps with a diagnostic copy** (`judge_diag/tools/pilot3_judge.py`, sha256 bc483493; the only change is
line 85 -> -2.0; NOT a pinned tool; beside unchanged copies of iso.py and gait_probe_pilot.py) [measured:
judge_diag/logs/, out/judge_diag_bindings_summary.txt, out/judge_diag_cells_labels.txt]:
- J1 accepts the 65,536 pair: exit 0; validation_only true; warm_start records -2.0/-2.0; run_end periodic 32,768 and
  65,536, final 65,536; 40/40 STILL.
- J2 refuses pilot 1's parent; J3 no critic_phase_dropped; J4 "run.warm_start_log_std is absent (not -2.0)"; J5 n_envs
  2; J6 git_commit 3ae1c959 != HEAD a3f1b378. Each exits 2 and writes nothing.
- J7 --final-fallback is accepted (OUT/s42/50000, final 65,536, 40/40 STILL).
- J8 refuses 49,152 ("not the uninterrupted end 65,536").
- J9 gives the OFF-SCHEDULE refusal.
- J10 writes only OUT/off_schedule/pilot3_walk_s42_rtest/s42/65536: 40/40 STILL, budget 17,232 + 32,768 = 50,000.
- These are the smoke's ten outcomes. The judge needs nothing else for -2.0.

**Pinned files that still hard-code the old values** (grep; out/tools_stale_constants_grep.txt, 23 lines; the same list
as DESIGN 10.2):
- Tools: pilot3_judge.py :39 and :85; pilot3_noise.py :8 and :39.
- Tests:
  - tests/test_pilot3_judge.py :39, :44, :65, :84, :347, :377 (-1.5) and :279-:284 (commit-prefix cases with 3ae1c959,
    value-agnostic);
  - tests/e2e_parent_cell.py :4, :85, :105;
  - tests/test_pilot3_noise.py :102, :121, :130 (0.2231);
  - tests/test_nb_pilot_runner3.py :141 and tests/test_pilot3_habit.py :305 (synthetic values, value-agnostic).
- tools/tree (the tests' tree facts) is still at 3ae1c959.
- The tools' test suite was not run here.

## Findings for the next stages

1. **The pinned judge refuses the pilot as configured** (section 4). It needs the -2.0 constant, re-pinned with its tests
   and noise reference, before any judging. [measured]
2. **The evaluation ledger carries no reward-term columns.** DESIGN section 3.2 says eval_episodes.csv "gains the
   reward_walk_touchdown sum"; that does not hold. The header is byte-identical to the -1.5 smoke's and has no reward_*
   column at all. The evaluation's per-term means go to TensorBoard as `reward_terms/eval_walk_touchdown` (and
   `eval_airborne_substeps`), not to the CSV. [measured: out/smoke_evidence_launch_s42.json "r6_columns",
   out/tb_tags_vs_smoke.txt]
3. **The 11 T. rex behavior recipes resolve to R6** through locomotion's [env]. The R6 keys appear only in
   locomotion.toml, as the rule asks, but these recipes' task digests and bit-exact reward captures move. Commit
   4dfc7cb's message discloses it. It is not a stance change. [measured: golden diff, out/digest/exact_compare.txt]
4. **The frozen text's second Warm start line (-1.5, std 0.223) and every -1.5, 8263f82f and 5f835d65 reference are
   stale.** The re-measured line is in section 4. This is for the text-amendment stage. [measured]
5. **The criteria's TensorBoard tag count (244) is not a fixed number.** Fall and termination tags appear only when a
   training episode falls. [measured]

## Not run (and why)

- The held-out panel 3062-3081 and SCREEN's other behaviours (V29, V33, HF150, H25, parent_ls15). The task named seeds
  3042-3061 and the listed behaviours.
- The smoke's scripted rollout, render check, pilot3_video.py and replay_check. They are not in this stage's list, and
  the video tool would have needed a judge-accepted cell, which only the diagnostic judge produced.
- pilot3_vg.py.
- The tools' and the repository's test suites. The implement stage ran the R6 tests and suites.
- `zero_action_baseline.py` for the R6 statue (design V4). Part 1 measured the R6 statue directly: 1091.40 at w 2.0
  and 1094.43 at w 0.2 on 3042-3061.

## Side effects outside this folder, and slips

- **Side effects:**
  - Xvfb (the notebook smoke's private display) compiled and removed a keymap in /var/lib/xkb. The folder's mtime moved
    to 14:58:38 UTC, and it still holds only README.compiled, as at the smoke.
  - The --shared clones read git objects from the main repository's object store (read only).
  - No entry under the system temp folder (depth 1) or /root/.cache or /root/.config is newer than this stage's marker.
  - SP/wt_p3: head a3f1b378, 0 status entries. judge_tree: 0 entries. clone: only its ignored logs/.
- **Slips:**
  1. **CMDLOG literalness (reads).** Read-only ls/cat/sed/grep/git commands of 14:50-14:57 ran before or outside the
     logger. They are listed in words in a NOTE, whose typed end time (15:02) was wrong and is corrected by the next
     NOTE. Harness Read-tool reads are NOTE lines.
  2. **One read used the system python3.** A read-only JSON key listing of the VARIANTS record ran with the system
     python3 -c, not the prescribed interpreter, before the logger existed.
  3. **CMDLOG literalness (batch scripts).** The SIM BEGIN lines written by `scripts/batch_digest.sh` and
     `scripts/batch_probe_judge.sh` paraphrase their commands ("PY", "tools/iso.py", "\$S"). Each script's full text is
     logged literally where it was written. `bin/sim.sh`'s lines (parts 1-3) are literal.
  4. **Analyses beside a simulation.** Several read-only analyses ran while one simulation or training process was
     running:
     - cfgdump.py (it builds the plant identity but steps no env), evidence2.py, pilot3_habit.py ledger and
       pilot3_noise.py ran beside the notebook smoke, its RESUME or the part-1 batch;
     - v2analyze.py ran beside the stance batch;
     - the waiting loops ran at normal priority beside the simulations, and one of them read the harness's task-output
       file.
     No two simulations or trainings ever overlapped.
  5. **Adapted smoke scripts.** `nb/evidence2.py` is the smoke's evidence.py with five exact replacements (diff in
     CMDLOG: -2.0, the task-digest label, the R6 and gate checks, the ledger's R6 columns). `judge/run_judge.sh`
     differs from the smoke's only in HEAD and its comment. The diagnostic judge copy differs from the pinned one only
     in line 85. The expected second Warm start line is taken from the commit message and DESIGN (nb/expected_warm_start.txt).
  6. **No forbidden-device check.** No check was made for the null or zero device in this folder's text. Such a check
     would itself have to spell the forbidden string. None was written knowingly; every discard went to a file in this
     folder.
- Folder size at the end: 368 MB, under the 400 MB limit (clone 175 MB, judge_tree 132 MB, judge 35 MB, out 19 MB).

## Files

- Logs: `CMDLOG.txt`; `bin/run.sh` (literal logger), `bin/sim.sh` (simulation wrapper), `bin/env.sh`, `bin/withx`.
- Part 1:
  - scripts: `scripts/v2run.py`, `scripts/v2analyze.py`, `scripts/batch_v2.sh`, `scripts/ctrl/`;
  - outputs: `out/v2/A_<B>.json`, `_stdout.txt`, `_steps.npz`, `out/v2/w02_*`, `out/v2/analysis.json`,
    `analysis_stdout.txt`.
- Part 2:
  - scripts: `scripts/cfgdump.py`, `scripts/stance_hash.py`, `scripts/stance_variants/` (VARIANTS' stance.py and vlib.py,
    unchanged), `scripts/stance_compare.py`, `scripts/runmod.py`, `scripts/batch_digest.sh`;
  - outputs: `out/stance/`, `out/digest/`.
- Part 3: `scripts/p3run.py`, `scripts/batch_stance_p3.sh`, `out/p3/`.
- Part 4:
  - runner inputs and outputs: `nb/` (params, plan_launch, launch_s42, rtest_launch, rtest_resume, consoles, rc files,
    evidence2.py, expected_warm_start.txt);
  - runs: `clone/logs/.../ppo_quick_test/pilot3_walk_s42{,_rtest}`;
  - evidence: `out/smoke_evidence_launch_s42.{json,txt}`, `out/habit_ledger_smoke_s42.*`, `out/noise/`,
    `out/launch_vs_rtest_32768_members.txt`, `out/rtest_resume_stage_config_check.txt`,
    `out/unbroken_vs_resumed_65536.txt`, `out/tb_tags_vs_smoke.txt`;
  - parent probe: `probe_parent/`, `out/probe_parent_labels.txt`;
  - judge: `judge/` (make_copies.py, copies, run_judge.sh, logs), `judge_diag/` (diagnostic only);
  - tool checks: `out/tools_sha256_check_{start,end}.txt`, `out/tools_stale_constants_grep.txt`.
