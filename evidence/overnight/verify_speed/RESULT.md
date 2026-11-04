# VERIFY_SPEED: independent adversarial check of the SPEED stage

My folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/verify_speed/`. Paths below are relative to it unless they start with `speed/`, which means the SPEED stage folder `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/speed/`.

I worked 02:13-02:50 UTC on 2026-11-04 (write-up due 07:30 UTC).

**Resume:** this folder did not exist when I started. No CMDLOG, scripts or partial outputs from a first attempt were left to keep, so everything here was redone. The `RESUME after container restart` line is in CMDLOG.txt.

Units: m, s, m/s, deg, N.m. L = 0.8766 m, which I computed myself as the mean height of the hip-pitch anchors at the home keyframe. Fr = v²/(g L).

## Verdict: CONFIRMED WITH CORRECTIONS

- **Everything I re-ran reproduces bit for bit.**
  - I ran 326 episodes on the real plant in 36 runs of my own. 249 of them are distinct full episodes; some seed sets ran twice, once with traces saved. I also ran 10 HYPOTHETICAL episodes in 2 runs.
  - Every episode I re-ran ends at exactly the SPEED stage's final root xy (or the EXISTENCE stage's), to the last bit, on every seed.
  - My own floor-truth contact and gait code gives the same alternation, flight, double support, speed, stride and frequency as the stage's probe numbers. [measured: out/cmp_*.txt, out/sum_all.txt]
- **The main claim holds and gets stronger on new seeds.** The fastest walk that passes the stage's test is 1.2 m/s (V35).
  - Seeds 0-4: 5/5. Seeds 0-14: 14/15.
  - Fresh seeds 15-39, which the stage never ran: 25/25 pass.
  - Total: 39/40 seeds pass. [measured: out/sum_all.txt]
- **The plant was driven only through env.step with actions in [-1, 1], and no world-position feedback was used.**
  - Guards on every step: no action outside [-1, 1]. The controller call never changed the env's MjData. Physics time moved only inside env.step. The compiled model was unchanged.
  - Shadow test: the actions do not change when the root is shifted 4.7 m in the floor plane (differences of 1.9e-12 and below on 15 + 5 + 5 seeds; 2.9e-6 on V29).
  - Rewards add up: each step's reward equals reward_total and the sum of its reward_* terms (plus the fall penalty), to 1.3e-15 or better.
- **Eight corrections** (listed below). Two change numbers in the stage's limits and key results. One changes how the airborne steps at 1.5 m/s are described. The rest are attribution and labelling problems, including two conclusions about the real plant that were drawn from HYPOTHETICAL models.

## How I verified (independent of the stage's runner and probe)

`scripts/vrun.py`, plus `scripts/vrun2.py`, which is the same file with `--save` and `--hypo` added:
- It builds the plant with the export's own factory `build_env(ModelId("tyrannosaurus_rex","anatomical"), "locomotion")`.
  - Config sha256 100002a2…ef70b; max_episode_steps 1000; reset_noise_scale 0.05; dt 0.01 s with 5 substeps; 10 Hz action filter; fall_penalty -150. [measured: out/*_stdout.txt "desc" line]
- My own loop: `env.reset(seed)`, then `env.step(a)`.
- Actions come from the stage's controller (`speed/scripts/wlib2.py` Walker2, or the unchanged EXISTENCE `wlib.Walker`). I imported it read-only, with no bytecode written to the stage folder, using the exact `params` stored in the stage's `out/<tag>.json`.
- Measurements are my own:
  - Floor truth comes from my own hook in the env's `_substep_probe_hook` slot. It records each leg's floor normal force, where a leg is the subtree of r_thigh or l_thigh (7 geoms each), plus any floor contact by a geom outside the legs.
  - Gait metrics are my own code: 2-step debounce, L/R merge within 2 steps, alternation, duty, double support, stride, and lead swaps.
  - The actuator audit uses the stage's stated definitions, computed from my own records.
- `scripts/vcheck_cfg.py` compares every stage JSON with its literal command line in the stage's CMDLOG.

## Headline by headline

1. **"1.2 m/s (Froude 0.17) is the fastest speed that passes; env.step only, actions in [-1, 1], no world-position feedback."**
   - **CONFIRMED.** V35 seeds 0-4: 5/5 pass. At 1.35 m/s, no variant passes 5/5 on seeds 0-4 (V27 3/5, V34 4/5). V27 also passes only 20/40 over seeds 0-39. [measured: out/sum_all.txt, out/gate_quantity.txt]
   - Note on Froude: the stage computes Fr from the displacement speed. The probe's own Froude uses forward_vel, and for V35 it is 0.157-0.171 rather than 0.166-0.171, because heading drifts by up to 22 deg. [measured: out/froude_check.txt]
2. **"V35 5/5 (0-4), 14/15 (0-14), 15/15 full, 1.194-1.211 m/s; V19 5/5 and 9/10 held-out; 0.5 (V05), 0.7 (V24), 1.0 (V18, 15/15) pass."**
   - **CONFIRMED**, every number, bit for bit.
     - V35 seed 8 is the only failure: one airborne step.
     - V19 held-out seed 11 falls at 8.15 s.
     - V05 runs 0.526-0.532 m/s over 3-10 s and 0.512-0.524 m/s over 5-10 s.
     - V24 runs 0.699-0.703 m/s. V18 + H18: 15/15. [measured: out/cmp_V35_V19.txt, out/cmp_H19_V18_H18.txt, out/cmp_V24_V05_V23.txt]
   - Fresh seeds 15-39 all pass: V05 25/25, V24 25/25, V18 25/25, V35 25/25. [measured: out/sum_fresh.txt]
3. **"The EXISTENCE failures at 0.88 m/s and above were controller-limited; eight changes took the walk from 0.31 to 1.2 m/s."**
   - **CONFIRMED as an inference.** [inferred: the same plant walks at 1.2 m/s for 10 s under a different controller, so the plant did not cap it at 0.88]
   - Both end points reproduce: V01 runs 0.305-0.311 m/s and V35 runs 1.194-1.211 m/s.
   - Six intermediate ledger outcomes also reproduce bit for bit: V06 0/5, V10 5/5, V11 0/5, V15 5/5, V16 5/5, V17 4/5. [measured: out/sum_batch3.txt]
   - I did not re-verify each change's trace diagnosis (the mechanisms).
4. **"Above 1.2 m/s the controller fails first, not hip roll, pad roll, torque or joint range"**, with V21 0/5, 1.35 m/s 15/15 survive, 1.5 m/s 15/15 survive, bounces fail strict on 7/15 and 15/15, 1.8 m/s 0/5.
   - **CONFIRMED WITH CORRECTIONS.**
   - Every count reproduces. V21 falls at 7.53, 7.10, 5.88, 7.46 and 8.45 s. V27 + H27: 15/15 full, airborne steps on 7/15. V29 + H29: 15/15 full, airborne steps on 15/15. V33 falls at 7.59, 7.81, 7.79, 9.70 and 6.74 s. [measured: out/sum_all.txt]
   - Corrections 2, 3, 4 and 5 below apply to this headline.
5. **"No running gait appeared: every surviving episode is probe class 'walk', double support 0.06-0.21. Plant ceiling at least 1.2 m/s strict, 1.5 m/s with rare bounces."**
   - **CONFIRMED WITH CORRECTIONS.**
   - All 249 distinct full real-plant episodes I ran are class walk by the probe's decision rule; none is run, hop or other. Double support is 0.064-0.186 over t ≥ 1 s and 0.050-0.212 over 5-10 s. [measured: out/sum_all.txt]
   - "1.5 m/s with rare bounces" needs qualifying:
     - All 15 seeds have airborne steps (0.22-1.44% of control steps).
     - About half of those steps come just before touchdown, not after it (correction 2).
     - One seed fails alternation (correction 4).

## Corrections

1. **The substep-level strict ceiling is 0.5 m/s, not 0.7 m/s.** This is in the stage's limits.
   - V24 at 0.7 m/s has one physics substep with both legs unloaded after t = 1 s on seeds 0, 1 and 4.
   - V10 at 0.7 has 3 such substeps on 3 seeds.
   - The only 0.7-command variant with none, V15, reaches just 0.661-0.674 m/s.
   - V05 and V23 at 0.5 have none. [measured: out/cmp_V24_V05_V23.txt; speed/out/V10_v070_crouch.json and V15_v070_ground.json audit; out/sum_batch3.txt]
2. **At 1.5 m/s the airborne steps are not all "landing bounces about 0.03 s after a stance switch".**
   - At 1.35 m/s that description holds: all 10 airborne steps on seeds 0-14 fall 0.01-0.03 s after a controller stance switch.
   - At 1.5 m/s (V29 + H29, seeds 0-14), 37 of 75 airborne steps fall 0.01-0.02 s *before* the next registered touchdown, 0.18-0.20 s after the previous switch, with the trailing foot on 0-2 of 5 substeps. These are brief flight intervals before touchdown. The other 38 are post-switch bounces. [measured: out/trace3_vS29.txt, out/trace3_vS29h.txt, out/trace3_vS27.txt, out/trace3_vS27h.txt]
   - The stage's evidence (`speed/out/an_air_events_V29.txt`) covers seed 0 only, which has 2 events, both after a switch.
   - Double support stays at 0.13-0.17, so the probe still classes these episodes walk.
3. **The "not limiting up to 1.5 m/s on surviving episodes" bounds come from V29 seeds 0-4 only.** On the held-out H29 seeds 5-14, all of which survive, my audit gives:
   - hip-roll demand beyond its ctrl range: up to 5.6% (stage: ≤ 3.2%)
   - hip-roll deliverable p95: up to 0.82 (stage: ≤ 0.74)
   - stance-pad roll p95: up to 8.4 deg (stage: ≤ 7.2)
   - minimum knee margin: 8.3 deg (stage: ≥ 8.5)
   - minimum ankle margin: 10.8 deg (stage: ≥ 13)
   - Within the stage's bounds: force/forcerange peak 0.75 (≤ 0.80) and p95 0.52. [measured: out/cmp_H29_V33.txt audit block]
   - [inferred: these are still well inside the servo and joint limits] So "not limiting" can stand, but not with the quoted numbers.
4. **Alternation at 1.5 m/s is 14/15, not 15/15.** "1.5 m/s survives 15/15" is true. But H29 seed 11 has alternation 0.884, below 0.9, and the headline leaves this out. [measured: out/cmp_H29_V33.txt]
5. **"The controller fails first" is shown only for the T_sw 0.32 runaway at 1.35 m/s.**
   - For that case the evidence holds. My traces of all 5 V21 seeds show the same pattern as the stage's seed-0 table: regular 0.25 s steps, then one long step (0.32-0.41 s), long steps (0.58-0.68 L), the speed running past 1.7 m/s, and then the fall. Stance hip-roll |b| stays ≤ 0.92 and pad roll ≤ 4.3 deg until the collapse step. [measured: out/trace2_V21.txt]
   - The stage cites only seed 0 (|b| ≤ 0.89, ±4 deg); I see slightly higher values on other seeds.
   - For the strict no-flight failure at higher cadence, the cause is unidentified, as the stage itself says. Attributing it to the controller is an inference, not a finding.
6. **Two conclusions about the real plant were drawn from HYPOTHETICAL models.**
   - From P2 (half leg damping): "so damping is not the single binding limit for this controller", in the headline's key results under a [measured] label.
   - From P1 (stiff pad): "the soft pad is not the cause" of the bounces.
   - In both cases a controller tuned on the real plant was run on a changed plant, so getting worse does not show that the real property isn't limiting. [inferred]
   - The P1 and P2 numbers themselves reproduce bit for bit: P1 gives 2/5 full and 0/5 strict; P2 gives 0/5 with falls at 8.97, 7.04, 6.15, 6.35 and 6.28 s. Both are labelled HYPOTHETICAL in my tags, JSONs and printed lines. [measured: out/cmp_HYPO_shadow.txt]
   - No HYPOTHETICAL run fed any real-plant number. [measured: out/check_cfg.txt]
7. **Interpretive sentences carry [measured] labels or none.**
   - The five headline sentences carry no labels, including "were controller-limited" and "the controller fails first".
   - RESULT.md's bottom line 2 ("controller-limited, as suspected") is labelled [measured].
   - The key results "so this failure is controller-limited" and "so damping is not the single binding limit" sit inside [measured] brackets.
   - These are inferences.
8. **Minor definitional points.**
   - Froude in the ladder uses the displacement speed (headline 1).
   - "Double support 0.06-0.21" mixes the 3-10 s and 5-10 s windows. My ranges are 0.064-0.186 over t ≥ 1 s and 0.050-0.212 over 5-10 s.

## Key results: what I re-measured

- **V35 gait at 1.2 m/s, 5-10 s window, seeds 0-14.** Matches every stated range. [measured: out/cmp_V35_V19.txt, out/steptime_V35.txt, out/trace_V35.txt]
  - speed 1.194-1.211 m/s
  - stride 0.58-0.67 L, mean 0.64 L
  - stride frequency 2.06-2.35 Hz; step median 0.240 s
  - duty 0.56-0.60; double support 0.146-0.180
  - alternation 0.923-1.000
  - 0 forced (time-out) stance switches
- **Actuator use at 1.2 m/s (V35, 15 seeds).** Identical to `speed/out/an_show_budget.txt`. [measured: out/cmp_V35_V19.txt audit block]
  - force/forcerange: peak 0.52 or less, p95 0.335 or less
  - deliverable fraction at p95: hip pitch 0.44-0.45, hip roll 0.50-0.55, knee 0.43-0.46, ankle 0.58-0.64
  - demand beyond the ctrl range: ankle 1.6-2.3%, hip pitch 0.8-1.8%, knee 0.1-1.0%, hip roll 0-0.2%
  - minimum joint margins: hip pitch 29.6, hip roll 11.8, knee 9.1, ankle 12.1 deg
  - stance-pad roll p5 down to -5.51 deg, p95 up to +4.96 deg
- **Ankle budget at 1.35 and 1.5 m/s.**
  - Deliverable p95: 0.67-0.80 (V27) and 0.83-0.90 (V29).
  - Raw action clipped: 12-16% and 21-25% of steps.
  - Both as stated, for seeds 0-4. On H29 the ankle reaches 0.76-0.92 and 17-27%.
  - I did not re-measure the swing joint speeds or the share taken by passive damping. [measured: out/cmp_V27.txt, out/cmp_V29.txt, out/cmp_H29_V33.txt]
- **Airborne physics substeps.** 8-18 per episode at 1.2 m/s (V35) and 1-7 at 1.0 m/s (V18 + H18), as stated. [measured: out/sum_all.txt, out/cmp_*.txt]
- **Whole-episode mean speed.**
  - The stage's CoM-based figures reproduce: 0.896-0.913 m/s at the 1.2 command.
  - In the gate's own quantity (the episode mean of info["forward_vel"], computed in the env's eval callback and checked in curriculum/manager.py), over seeds 0-39:
    - 1.0 command: 0.758 m/s mean
    - 1.2 command: 0.897 m/s mean, 0/40 episodes reach 1.0
    - 1.35 command (T_sw 0.28): 1.004 m/s mean, 30/40 episodes reach 1.0
  - So the stage's inference ("about 1.35 m/s steady to average 1.0") holds in the gate's quantity. [measured: out/gate_quantity.txt]
  - [inferred: same numbers] The strict 1.2 m/s walk alone would not meet the stage's 1.0 m/s average-speed gate.
- **FINAL3 baseline.** It reproduces bit for bit both with the unchanged `wlib.Walker` and with Walker2 with every option off, against the EXISTENCE probe JSON on seeds 0-2. [measured: out/cmp_base.txt, out/cmp_w2.txt]
- **Window check.** Starting the alternation and flight window at the walk start (t = 0.5 s) instead of 1 s changes no pass or fail: V35 14/15, V27 3/5. [measured: out/win_V35.txt, out/win_V27.txt]

## Configs, seeds and HYPOTHETICAL bookkeeping

- All 45 stage run JSONs match their literal CMDLOG command line in params, seeds, controller class, export path and HYPOTHETICAL flag; there are 0 mismatches. [measured: out/check_cfg.txt]
- Each held-out run has the same params as its seeds 0-4 run: H18 = V18, H19 = V19, H27 = V27, H29 = V29.
- P1 has the same params as V29, and P2 the same as V33; only P1 and P2 carry HYPOTHETICAL.
- `speed/scripts/wlib2.py` was patched up to 01:19 UTC, after V18 and V19 had run at 01:06-01:07. The final file still reproduces those runs bit for bit, so the later patches did not change earlier behaviour. [measured: out/cmp_*.txt; speed/CMDLOG.txt]

## Rewards (main's locomotion reward)

- Bookkeeping holds on every step of every episode: reward = reward_total = Σ reward_* (+ fall_penalty -150 on a non-success termination), maximum absolute error 1.3e-15. [measured: out/sum_all.txt "rerr"]
- Mean returns over full episodes rise monotonically with speed. [measured: out/returns_table.txt, out/gate_quantity.txt]

  | walk | mean return |
  |---|---|
  | FINAL3 (0.21 m/s) | 1234 |
  | 0.31 m/s | 1300 |
  | 0.5 | 1434 |
  | 0.7 | 1489 |
  | 1.0 | 1655 |
  | 1.2 | 1766 |
  | 1.35 | 1855 |
  | 1.5 | 1881 |

- The forward term carries the rise: 148 → 847 per episode. The alive term is 500 throughout. The penalties are small, with nosedive at most -12.
- For comparison, the config's zero-action statue reference is 1092.6. That figure is the config's own and was not measured by me.

## Not re-verified

- The trace diagnosis behind each of the eight controller changes. I re-ran outcomes only.
- V20, V22, V25, V26, V28, V30, V32 and V36 (only their configs were checked).
- Swing joint speeds and the damping share of the ankle span (`speed/out/an_swing_speed.txt`).
- The swing/stance saturation split (`an_sat_split_*`).
- Yaw, slip and clearance figures.
- Whether the controller's inputs are equivalent to the policy observation. Only translation invariance was tested.
- Whether the stage's files mention the forbidden device. I deliberately did not check this, because any search for it would itself reference it.

## Rule slips

**The stage's own** (it self-reported nine; I add these):
- (a) HYPOTHETICAL-driven claims about the real plant (correction 6).
- (b) Interpretive sentences labelled [measured] or unlabelled (correction 7).
- (c) Single-seed files cited for all-seed statements: `an_air_events_V29.txt` (wrong at 1.5 m/s, correction 2) and `an_limits_V21.txt` (holds, correction 5).
- (d) Budget bounds quoted from seeds 0-4 as if they covered all surviving episodes (correction 3).
- (e) The wrong substep-ceiling statement (correction 1).

**Mine:**
1. The first three shell commands (date/ls, ls/df, and the mkdir with heredocs) ran before CMDLOG existed. They are recorded after the fact in a NOTE, literally.
2. The scripts vrun.py, vrun2.py (six edits), vcheck_cfg.py, vcompare.py, vsummary.py, vtrace.py, vtrace2.py, vtrace3.py and vwin.py were written with the Write/Edit tools, not shell commands. Each is recorded by a NOTE line, and the sha256 of vrun.py and vrun2.py is logged.
3. The NOTE lines are themselves appended by `echo` commands, which are not logged separately.
4. vcheck_cfg.py imports the stage's `defaults.py` read-only to merge parameters the way its runner did. That is a use of the stage's code beyond regenerating actions.
5. My runner sets the env's `_substep_probe_hook` slot to my own read-only hook during each episode and restores it afterwards. This does not change physics or code, and the bit-for-bit agreement shows it did not change any trajectory.
6. One process listing (`ps`, 02:26 UTC) printed the harness's own shell-wrapper command line to my terminal. That line contains the forbidden device name. My command did not name it, and nothing from it was written to any file.
7. My analysis-only commands (no simulation) ran without `nice -n 10`.
8. RESULT.md was written with the Write tool and corrected with the Edit tool. One logged attempt to patch it with a Python string replace failed on its first assertion and changed nothing.
9. The Froude comparison was first printed only to the terminal (02:22:54 UTC). I regenerated it into `out/froude_check.txt` so that its label cites a file.

The rest of my run followed the rules:
- Every simulation ran one process at a time, through the logged wrapper `scripts/L`, which sets OMP_NUM_THREADS=1 and a TMPDIR inside my folder, under `nice -n 10`. The batches waited on each other.
- There was no training or fitting.
- The repository, the export and the physics were not changed. The two HYPOTHETICAL runs edited only their own process's compiled model, in memory.
- I wrote only inside my folder. The folder is about 9 MB.

## Files

- `CMDLOG.txt`
- `scripts/L` (the logging wrapper)
- `scripts/v*.py`
- `out/v*.json` and `out/v*_stdout.txt` (one per run)
- `out/*_s<seed>.npz` (saved traces)
- `out/cmp_*.txt` (per-seed comparison with the stage)
- `out/sum_*.txt`, `out/check_cfg.txt`, `out/trace*.txt`, `out/win_*.txt`, `out/gate_quantity.txt`, `out/returns_table.txt`, `out/steptime_V35.txt`
- `out/batch*_progress.txt`
