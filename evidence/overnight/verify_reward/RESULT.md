# VERIFY_REWARD: independent check of the REWARD stage

Verifier folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/verify_reward/` (called V below).
Stage checked: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/reward/` (called REWARD below).
Worked 01:37-01:52 UTC, then (after a container restart at about 01:52) 01:54-02:12 UTC. Time box 07:30. Every command is in `V/CMDLOG.txt`.

This was CPU analysis only. Nothing was trained or fitted. The repository, the export and the physics were not changed, and no HYPOTHETICAL model was built. Every simulation ran as one process at a time through `V/scripts/run.sh` (nice -n 10, OMP_NUM_THREADS=1, TMPDIR in V). Episodes were driven only by `env.step(a)`; every action was asserted finite and inside [-1, 1] before the call.

"Return" is the undiscounted sum of rewards over one episode (1000 control steps: 10 s for the T. rex at dt 0.01 s, 20 s for compsognathus at dt 0.02 s). Rewards are dimensionless reward units; speeds are in m/s; w is forward_vel_weight.

## Verdict: CONFIRMED WITH CORRECTIONS

Every number in REWARD's six headlines reproduced. Every T. rex and compsognathus episode REWARD recorded, on the same seed and weight, gives the same length, end reason, return and per-term sums in my re-run (bit-identical: |Δreturn| = 0 on 540 of 540 matched episodes, 460 T. rex and 80 compsognathus; my own numpy copy of the parent's actor, which REWARD did not use, differs from the SB3 path by at most 1.3e-7 per episode, from float32 vs float64). The rewards add up: on every step, my explicit term list plus the fall penalty equals the reward env.step returned to within 2.8e-14, and to within 7.7e-12 per episode; the forward term recomputed from qvel and the prey direction matches info to 0.0 on every step. [measured: V/out/compare_trex_stdout.txt, V/out/compare_compso_stdout.txt]

Three corrections, all in wording or scope, none in sign:
1. **Headline 3 (pilot 3's start).** "Falls in 36 of 40 episodes" counts every early termination: 17 fallen, 18 head_contact, 1 nosedive (4 reach 1000 steps). The 456 / 422 returns are the 20-seed means (seeds 3042-3061); on all 40 seeds they are 421.7 / 394.6. "About 640 below both standing and walking" holds only at weight 0.2 (638.5 below the statue, 644.0 below FINAL3, 643.2 below f2v0.25). At weight 2.0 the gaps are 670.8 (statue), 812.9 (FINAL3) and 878.6 (f2v0.25). The 456 / 422 also come from one noise draw. My own numpy noise stream on the same 20 seeds gives 338.9 / 322.6 (18 of 20 terminated: 6 fallen, 11 head_contact, 1 tail_contact), and on 40 seeds 397.3 / 375.3 (35 of 40 terminated). The gap to the statue at weight 0.2 is therefore 638-756 depending on the draw, and the conclusion (far below both standing and walking) holds for both draws. [measured: V/out/extra_tables_stdout.txt]
2. **Headline 5 (hop).** The sentence "Most of the walk's 9.6-per-episode deficit (lateral sway, heading wobble, tail swing) comes from alternating single support" is an interpretation and carries no [inferred] label in the headline. It also conflicts with REWARD's own measured per-step table (its RESULT.md line 160: "nearly the same in single and double support ... not from which foot is down"). My independent per-step re-run agrees with that table: FINAL3's lateral cost is -0.00370 per step on one-foot steps and -0.00371 on two-foot steps, so the sway is a property of the gait, not of the support state; the tail cost is about twice as large on one-foot steps (-0.00164 vs -0.00084). Lateral + heading + tail make up 6.14 of the 9.64 deficit (64%), so "most" is right. Suggested wording: "Most of the walk's 9.6-per-episode deficit is its sway (lateral -3.6, tail -1.4, heading -1.2) [measured]; a left-right-symmetric two-footed hop might not pay it [inferred]." [measured: V/out/vsteps_stdout.txt]
3. **Smaller items.**
   - The key result "drifts backward at 0.075 m/s" is the 20-seed episode mean; REWARD's RESULT.md line 162 gives 0.083 m/s, the step-weighted mean of the same episodes. Both are measured. The 5.7% "neither foot loaded" is the step-weighted 20-seed share (my episode mean over 20 seeds: 5.5%; over 40 seeds: 5.7%).
   - The limits list says only f2v0.25 and f2v0.35 read world position (k_yp 0.05). v06_v0.5 (k_yp 0.1) and v10 (k_yp 0.05) do too. FINAL3 has k_yp 0. As REWARD says, this does not change what the reward pays. [measured: V/out/params_check2_stdout.txt]
   - The limit "loaded with no plant-identity check" can be closed: `load_sb3_checkpoint` with the export's current plant identity ACCEPTS the parent (physics_revision 10, identical physics, policy-interface and source-closure sha256). [measured: V/out/plant_check_stdout.txt]
   - "The first 24k steps": with the callback's 10k-step buckets, the weight is 0.218 at 20k and 0.227 at 30k, so f2v0.25 overtakes FINAL3 from step 30k. [inferred: arithmetic from advancement.py:670-686]

## Headline by headline

| # | REWARD's headline (short) | my re-run | verdict |
|---|---|---|---|
| 1 | Weight 0.2: FINAL3 1100.1 vs statue 1094.6, +5.5 (0.5%), 18/20 seeds | 1100.10 vs 1094.56, +5.54 [-1.54, +13.56], 18/20; terms forward +15.18, height -4.33, lateral -3.58, tail -1.40, heading -1.16, nosedive +0.88 | CONFIRMED |
| 2 | Weight 2.0: f2v0.25 1300.6, FINAL3 1234.9, parent 1093.9, statue 1092.7; +142 / +208 on 20/20; forward decides; tie at 0.127-0.158 | 1300.58, 1234.88, 1093.91, 1092.74; +142.13 / +207.83, 20/20 each; forward +151.77 / +225.67; tie 0.1270 / 0.1581. Extra episodes at w 0.47 and 1.1 (5 seeds) match NF + w·F to 2.5e-12 | CONFIRMED |
| 3 | Pilot-3 start (log_std -1.5, stochastic): falls 36/40, 456 / 422, about 640 below standing and walking; parent stands 20/20 with own std and with -2.0 | 456.09 / 421.96 (20 seeds), 421.68 / 394.56 (40 seeds); 36/40 terminated (17 fallen, 18 head_contact, 1 nosedive); own std 1095.72 / 1093.69, 20/20 full; -2.0 1086.49 / 1069.84, 20/20 full | CONFIRMED WITH CORRECTIONS (item 1) |
| 4 | f2v0.35, v06_v0.5, v10 fell at about 2.1-3.1 s, 20/20; lose 776-1020; discounted lose 38-130 | 20/20 fallen each; mean 3.07 / 2.14 / 2.06 s (range 1.77-3.74 s); loss 905.46 / 1008.76 / 1019.63 at w 0.2 and 775.93 / 914.48 / 915.70 at w 2.0; discounted loss 37.96-130.23 | CONFIRMED |
| 5 | Reward pays speed regardless of gait; all support/contact/impact/airborne weights 0; alive ungated; a 1.05-1.57 m/s hop paid +840 to +1256 at w 2.0 vs +152 for the walk | Weights from my own build: all those terms 0, support_conditioned_alive_fraction 0, height_target_tolerance 0 (one-sided height); 1000 × 2.0 × v/2.5 = 840-1256; FINAL3's forward advantage over the statue +151.77 (its raw forward sum 149.75) | CONFIRMED WITH CORRECTIONS (item 2) |
| 6 | Compsognathus: FINAL3 loses -86 at w 0.1 (0/20 wins), wins +258 at w 8.0 (20/20), ties at 2.08 (~125k steps); reward gates alive, posture, height on support | statue 1499.94 / 1499.85; FINAL3 1413.83 / 1757.65 (19/20 full, one fall at 17.38 s); -86.11 (0/20), +257.81 (20/20); tie 2.078 → step 125.3k; alive -25.98, posture -25.91, height -24.44; 4.57% of the walk's steps have no foot at 5% body weight, which accounts for 22.7 of the 26.0 alive loss; support gate confirmed in code | CONFIRMED |

## Reproduction, T. rex (seeds 3042-3061 unless stated)

Return per 10 s episode, mean over 20 seeds. [measured: V/out/compare_trex_stdout.txt, V/out/extra_tables_stdout.txt]

| behaviour | full | end | v m/s | w 0.2 | Δ statue (wins) | w 2.0 | Δ statue (wins) | discounted 0.2 / 2.0 |
|---|---|---|---|---|---|---|---|---|
| (a) statue | 20/20 | truncated | -0.003 | 1094.56 | — | 1092.74 | — | 216.97 / 215.26 |
| (b) FINAL3 | 20/20 | truncated | 0.187 | 1100.10 | +5.54 (18) | 1234.88 | +142.13 (20) | 217.69 / 234.71 |
| (c) f2v0.25 | 20/20 | truncated | 0.280 | 1099.29 | +4.73 (18) | 1300.58 | +207.83 (20) | 217.75 / 244.30 |
| (d) f2v0.35 | 0/20 | fallen 20 | 0.565 | 189.10 | -905.46 (0) | 316.82 | -775.93 (0) | 138.16 / 177.30 |
| (e) parent, deterministic (SB3 and my numpy actor) | 20/20 | truncated | -0.003 | 1095.91 | +1.35 (19) | 1093.91 | +1.17 (19) | 217.51 / 215.65 |
| (f) parent, log_std -1.5, REWARD's torch draw | 2/20 | 10 fallen, 7 head, 1 nosedive | -0.075 | 456.09 | -638.47 (0) | 421.96 | -670.79 (0) | 166.63 / 156.69 |
| (f) same, my numpy draw | 2/20 | 6 fallen, 11 head, 1 tail | -0.024 | 338.92 | -755.64 (0) | 322.61 | -770.14 (0) | 151.91 / 145.83 |
| extra: parent, own std, stochastic | 20/20 | truncated | -0.003 | 1095.72 | +1.15 (19) | 1093.69 | +0.95 (18) | 217.49 / 215.63 |
| extra: parent, log_std -2.0 | 20/20 | truncated | -0.023 | 1086.49 | -8.08 (1) | 1069.84 | -22.91 (0) | 216.16 / 211.50 |
| extra: v06_v0.5 | 0/20 | fallen 20 | 0.600 | 85.80 | -1008.76 (0) | 178.27 | -914.48 (0) | 92.90 / 131.82 |
| extra: v10 | 0/20 | fallen 20 | 0.711 | 74.93 | -1019.63 (0) | 177.04 | -915.70 (0) | 86.75 / 134.01 |

- Statue panel at weight 2.0, seeds 3042-3081: 1092.602, sd 5.414 (n-1) / 5.346 (n), 40/40 full. locomotion.toml:122 gives 1092.60 ± 5.35 with 40/40 (in the comment of collapse_peak_floor_reference, line 99). [measured]
- Return along the ramp (NF + w·F): statue 1094.56 / 1094.29 / 1093.65 / 1092.74; FINAL3 1100.10 / 1120.32 / 1167.49 / 1234.88; f2v0.25 1099.29 / 1129.49 / 1199.94 / 1300.58 at w 0.2 / 0.47 / 1.1 / 2.0. Measured episodes at 0.47 and 1.1 (seeds 3042-3046, statue and FINAL3) match the prediction to 1.6e-12 and 2.5e-12. [measured: V/out/extra_tables_stdout.txt]
- Break-even: FINAL3's non-forward deficit is 9.64 per episode and f2v0.25's 17.84, giving 0.118 / 0.220 m/s at w 0.2 and 0.0095 / 0.0198 m/s at w 2.0. [measured deficits; inferred: assumes a walk at that speed pays the same deficit]
- Support state per step: statue 99.8% two-foot; FINAL3 91.4% one-foot; f2v0.35 4.8% neither; parent with log_std -1.5 5.5% neither (my draw 6.1%); parent with log_std -2.0 0.5% neither. Pelvis height: statue 0.9263 m, FINAL3 0.9228 m, f2v0.25 0.9173 m. [measured]
- The parent's per-actuator std, mapped to actuator names: knee 0.0070, hip pitch 0.0127, ankle 0.0125, hip roll 0.0520, neck/head 0.083-0.100, tail 0.096-0.111; log_std -1.5 (std 0.2231) is 2.0-31.8 times these. [measured: V/out/actnames_stdout.txt]

## Reproduction, compsognathus (seeds 3042-3061, ramp 0.1 to 8.0)

| behaviour | full | w 0.1 | Δ statue (wins) | w 8.0 | Δ statue (wins) |
|---|---|---|---|---|---|
| statue | 20/20 | 1499.94 ± 0.06 | — | 1499.85 ± 0.39 | — |
| FINAL3 (0.044 m/s) | 19/20 (fall at 17.38 s, seed 3043) | 1413.83 ± 53.71 | -86.11 [-304.04, -42.86] (0) | 1757.65 ± 59.55 | +257.81 [+20.31, +307.31] (20) |

Terms vs the statue (per episode): alive -25.98, posture -25.91, height -24.44, spin -5.06, energy -3.09, heading -2.38, lateral -2.17, fall -1.25, forward +4.35 (w 0.1) / +348.27 (w 8.0). Tie weight 2.078. The built env's locomotion.toml has no ramp keys, so the code defaults (0.1, 500k) apply; warm-up 100k steps, ent_coef 0.005. [measured: V/out/compare_compso_stdout.txt, V/out/inspect_env_stdout.txt, V/out/compso_support_stdout.txt]

## Rules checks on REWARD

- **Env and seeds.** REWARD's per-episode records use seeds 3042-3061 at both weights for every behaviour, plus 3062-3081 for the statue at w 2.0 and for (f), as it says; each matched my re-run on (seed, weight). Every `[env]` key of both locomotion.toml files reached the built env unchanged. [measured]
- **Only env.step, actions in [-1, 1].** REWARD's rollout.py drives episodes only through `env.step(a)` and sets the weight only through `set_reward_weight`. Its only other in-memory change is `policy.log_std.fill_` on the parent's policy, which is not part of the plant. Since every one of its episodes reproduced bit-for-bit under my driver, which asserts |a| ≤ 1 on every step, its action sequences were inside [-1, 1]. [measured: V/out/reward_scripts_grep.txt, compare outputs]
- **No HYPOTHETICAL model.** REWARD's scripts contain no write to any model, geom, actuator, option, ctrl, qpos/qvel or xfrc field, and no solref or gainprm edit. My two fresh builds give the same physics hash, and the checkpoint's plant identity matches the current plant. [measured]
- **Labels.** REWARD's RESULT.md labels nearly every claim. Unlabelled interpretive sentences: the JSON headline 5 sentence in correction 2; RESULT.md line 160, "They come from the gait's sway, not from which foot is down", placed under a [measured] header; and the JSON headline 6 phrase "faced a worse start", whose basis is the measured margins it then gives (-86 against +5.5). "Compsognathus ... did learn to walk on r1" comes from the brief and was not checked by either stage. [inferred: my reading]
- **REWARD's own slips** (non-literal and after-the-fact CMDLOG lines, system python3 for JSON readers, loop shells not under nice) are self-reported; I found none it left out. From the timestamps in its CMDLOG, its simulations ran in sequence (each simulation starts at or after the previous one's exit line; only JSON-reading analysis runs were interleaved). The REWARD folder is 32 MB. [measured: grep of REWARD CMDLOG.txt and du at 01:58-01:59Z; timestamps only]
- **Doc quotes.** The export's GAIT_QUALITY_PLAN_2026_09.md line 394 and lines 86 and 89 say what REWARD quotes (alive 164 / 205 and forward 261 / 475 airborne; hops at 7.7 Hz, 1.05 m/s and 9.2 Hz, 1.57 m/s). [measured: code/doc read]

## What I did not check

- PPO advantages, critic/value targets under VecNormalize, and the ±50 reward clip; what training would actually find (no training allowed; REWARD did not claim these either).
- The September hops themselves (doc claims only), and any hop simulation (none was required).
- Whether compsognathus actually learned to walk on r1 (brief claim).

## What I re-ran, and how it differs from REWARD's own code

My scripts (`V/scripts/vroll.py`, `compare.py`, `extra_tables.py`, `inspect_env.py`, `inspect_parent.py`, `plant_check.py`, `actnames.py`, `params_check.py`, `params_check2.py`, `compso_support.py`, `vsteps.py`, `arith.py`) were written for this check. They import nothing from REWARD. They import the EXISTENCE walker (`existence/scripts/wlib.py`) and its `defaults.py` read-only, only to regenerate the walkers' actions; wlib computes on private MjData copies and never writes the env's data or model (its own docstring; my grep of it at 01:56Z found no write to the env).

- **Env.** `build_env(ModelId(species, "anatomical"), "locomotion")` from the export, which runs `validate_compiled_plant`. The weight is set with `env.unwrapped.set_reward_weight("forward_vel_weight", w)`, the call RewardRampCallback makes. Every `[env]` key of both locomotion.toml files reached the built env unchanged (18 T. rex keys, 23 compsognathus keys, 0 mismatches), and two fresh builds give the same hash of the physics arrays (T. rex 567c9d9ca6dc3065, compsognathus 6d821b9e5bd1669f). [measured: V/out/inspect_env_stdout.txt]
- **Reward bookkeeping, done differently.** REWARD summed every `info` key starting with `reward_`. I summed an explicit list of term names copied from each species' reward total (T. rex: 29 terms plus the three added only when their weight is above 0; compsognathus: the 14 `components`), plus `fall_penalty` on a non-success termination, and compared that with the reward `env.step` returned, on every step.
- **Forward term recomputed from the state.** On every step: w·clip(qvel[0:2]·d / 2.5, -1, 1) (T. rex) or w·clip(qvel[0:2]·d, ±0.25) (compsognathus), d = the env's reset-time prey direction, compared with `info["reward_forward"]`.
- **Walker parameters rebuilt** from `existence/scripts/defaults.py` plus the variant ledger's `params_vs_default`, and compared with the `<tag>_<species>_s0.json` params REWARD read: identical for all six walker runs (FINAL3 both species, f2v0.25, f2v0.35, v06_v0.5, v10). [measured: V/out/*_stdout.txt, "params rebuilt" lines]
- **Parent actor re-implemented.** Besides SB3's `predict` path (REWARD's), I ran the parent through my own numpy copy of the actor (Linear 63→512, tanh, 512→256, tanh, action_net 256→15, from the checkpoint's state_dict) with the observation statistics read straight from the sidecar pickle (clip 10, eps 1e-8).
- **Independent noise for (f).** Besides REWARD's draw (torch seeded with the reset seed, through SB3's predict), I ran pilot 3's start with my own numpy noise stream (std exp(-1.5) = 0.223 on the numpy actor's mean, clipped to [-1, 1]), on the same 40 seeds.
- **Ramp points.** Extra episodes at w 0.47 and 1.1 (seeds 3042-3046, statue and FINAL3) test the claim that the return is exactly NF + w·F.

## Code reads (ramp, warm-up, warm start, reward terms)

All [measured: code read of the export main_4d5f4e7; commands in CMDLOG at 01:37-01:40Z and 01:56-02:00Z]:
- `_stage_entry_shaping_callbacks` (train_base.py:1314-1367) adds StageWarmupCallback and RewardRampCallback only for an `initialize_next_stage` load that has a parent and a checkpoint. The ramp runs from `ramp_start_value` (T. rex 0.2, locomotion.toml:162; compsognathus: code default 0.1, its locomotion.toml has no ramp key) to the `[env]` forward_vel_weight (2.0; compsognathus 8.0) over `ramp_timesteps` (2,000,000, locomotion.toml:161; compsognathus default 500,000).
- RewardRampCallback (advancement.py:601-686) sets the start value at training start, then updates once per 10k-step bucket, linear in `num_timesteps / ramp_timesteps`, rounded to 3 decimals, through `env_method("set_reward_weight", ...)` on the training VecEnv only.
- A cross-stage load is not a resume continuation (`_is_resume_continuation` returns False unless the mode is `resume_same_stage`, train_base.py:556-585), so `model.learn(reset_num_timesteps=not resuming)` (train_base.py:1830) starts the counter at 0. The weight is 0.47 at 300k steps and 1.1 at 1M steps [inferred: arithmetic, V/out/arith_stdout.txt].
- The warm-up (locomotion.toml:158-160: 300k steps, clip_range 0.02, ent_coef 0.02) changes PPO settings only; it does not touch the reward.
- Training's `make_env` (train_base.py:369-394) builds `env_class(**env_kwargs)` and adds only SB3's Monitor; `build_env` (config.py:401-429) builds the same class from the same `[env]` keys. So the env evaluated here is the env the pilot trains on, apart from the ramped weight.
- T. rex alive: `raw_alive * ((1 - f) + f * bilateral_support_quality)` with f = support_conditioned_alive_fraction = 0, so alive is ungated (tyrannosaurus_rex/base_env.py:950-958). Height is the legacy one-sided term because height_target_tolerance = 0 (base_env.py:1060-1078).
- Compsognathus alive, posture and height are multiplied by `support = float(right + left > _contact_threshold)` (compsognathus_longipes/base_env.py:250-262); `_contact_threshold` = 0.3924 N in the built env [measured: V/out/inspect_env_stdout.txt].
- Term bounds used in the hop argument: smoothness = -w·Σ(Δa)²/(4·n) (at most w = 0.05 per step), energy = -w·mean(a²) (at most 0.002), lateral = -w·clip(|v_lat|/5, 0, 1) (at most 0.1), tail = -w·min(|ω_tip|/max, 1) (at most 0.05) (reward_functions.py:107-133, 615-629; shared/base_env.py:585-604). REWARD's bounds are right.

## My slips (all recorded in V/CMDLOG.txt)

- **Forbidden device, twice.** (1) First attempt, about 01:50Z: to check REWARD for references to the forbidden discard device, I ran a grep whose search pattern held fragments of that device's path, and logged the pattern in CMDLOG (01:50:00Z line; SLIP NOTE at 01:50:10Z). At 02:03:52Z I replaced that log line's text with a redaction note, so the log no longer carries the fragments; this is the only edit of an earlier CMDLOG line, and it is itself logged. (2) This attempt, 01:59:44Z: one read-only progress check (date, ls -t, cat of a not-yet-written output) redirected its stderr to that device. Logged as a SLIP NOTE at 01:59:52Z without the path. Neither touched a simulation or any output used here.
- **Progress checks not logged when they ran.** Eight read-only progress checks between 01:56:58Z and 01:59:44Z (date, ls -t, tail, grep -c) were logged after the fact in the 01:59:52Z note. From 02:00Z on, every check is logged before it runs.
- **Not under nice:** the Monitor-tool watch loop (01:58:43Z), the two background wait loops (Bash run_in_background) and the chain shells' own bash processes (chain4.sh, chain5.sh were launched under nice -n 10; the run_in_background wait loops were not). Every python process ran through run.sh with nice -n 10, OMP_NUM_THREADS=1 and TMPDIR=V/tmp. Read-only Bash commands (cat, sed, grep, ls) were not run under nice.
- The first attempt's own CMDLOG has a few non-literal lines (summarised python3 -c JSON readers at 01:38:51Z and 01:43:04-01:43:10Z, which used the system python3 without nice/OMP/TMPDIR; abbreviated sed ranges such as `sed -n 819,852p;910,1394p` at 01:38:03Z). They read files only.
- My Read-tool views of the harness's background-task output files (under `/tmp/claude-0/.../tasks/`) were not logged in CMDLOG; they only read.
- RESULT.md was assembled from the drafted sections in V/tmp by one `python3 -I -c` string replace, with the system python3, not under nice (logged; no simulation).
- The harness wrote its own background-task output files under `/tmp/claude-0/.../tasks/`, outside V; I wrote nothing there myself.
- Torch created `V/tmp/torchinductor_root` inside V's TMPDIR.
- The restart left one partial log, kept as `V/out/trex_pnpls15_stdout_KILLED_by_restart.txt` and not used.
- No training, no repository, export or physics change, no HYPOTHETICAL model, simulations one at a time (chain4 then chain5, each run waiting for the previous one), no network, no git writes, no venv created and nothing installed (the shared venv's python had what was needed).

## Resume after the container restart

A container restart at about 01:52 UTC stopped the first attempt during the `pnp_ls15` run (its partial log is kept as `V/out/trex_pnpls15_stdout_KILLED_by_restart.txt` and not used). CMDLOG has the line "RESUME after container restart" at 01:54:38Z.

- **Kept from the first attempt (complete and valid):** `trex_zero` (seeds 3042-3081), `trex_final3`, `trex_f2v025`, `trex_f2v035`, `trex_sb3det`, `trex_pnpdet`, `trex_sb3ls15` (seeds 3042-3081) (each .json and _stdout.txt), `smoke_zero`, `smoke_walker`, `inspect_parent_stdout.txt`, `arith_stdout.txt`. Each finished with exit 0 in CMDLOG and wrote its JSON. vroll.py was edited once during that attempt (01:46Z), only to add the `sb3_own` branch, so those runs and the later ones use the same code for every behaviour they share. All of these reproduced REWARD's records bit-for-bit.
- **Superseded:** `compare_trex_partial.json` / `_stdout.txt` (replaced by `compare_trex.*` over all behaviours).
- **Redone or new after the restart (chain4.sh, chain5.sh, then single runs):** `trex_pnpls15`, `compso_zero`, `compso_final3`, `inspect_env`, `trex_v065`, `trex_v10`, `trex_sb3ls20`, `trex_sb3own`, `trex_ramp_zero`, `trex_ramp_final3`, `plant_check`, `actnames`, `compare_trex`, `compare_compso`, `extra_tables`, `params_check`, `params_check2`, `compso_support`, `vsteps`, `reward_scripts_grep.txt`.

## Files in V

- Scripts: `scripts/run.sh` (wrapper: logs the literal command, then runs it under nice -n 10, OMP_NUM_THREADS=1, TMPDIR=V/tmp); `chain1-5.sh` (sequential queues); `vroll.py` (episodes, explicit-term sum check, forward recompute); `vsteps.py` (per-step rates by support state); `compare.py`, `extra_tables.py`, `compso_support.py`, `params_check.py`, `params_check2.py` (JSON readers); `inspect_env.py`, `inspect_parent.py`, `plant_check.py`, `actnames.py` (load-only); `arith.py`.
- Results: `out/*.json` (per-episode records); `out/*_stdout.txt` (logs); tables in `out/compare_trex_stdout.txt`, `out/compare_compso_stdout.txt`, `out/extra_tables_stdout.txt`, `out/vsteps_stdout.txt`.
- Note: `params_check.py` compared full parameter dicts against `final3_params.json`, which holds only the 7 non-default keys, so its "None" differences are not mismatches; `params_check2.py` compares shared keys (identical).


