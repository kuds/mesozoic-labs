# REWARD stage: does main's T. rex locomotion reward prefer an alternating walk?

Stage folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/reward/`. Worked 00:42-01:12 UTC (time box 04:30).
This was analysis only. Nothing was trained or fitted. The repository, the export and the physics were not changed, and no HYPOTHETICAL (modified) model was built. Simulations ran one process at a time, under `nice -n 10`, with `OMP_NUM_THREADS=1` and TMPDIR inside this folder. Every command is in `CMDLOG.txt`.
"Return" means the undiscounted reward summed over one episode. Locomotion episodes last 1000 control steps: 10 s for the T. rex (dt 0.01 s) and 20 s for compsognathus (dt 0.02 s).

## Short answers

1. **At the start of the ramp (forward_vel_weight 0.2), a slow walk earns the most, but only just.** On the same 20 reset seeds:
   - FINAL3 walk: 1100.1 per episode.
   - Statue (zero action): 1094.6.
   - Stance parent, deterministic: 1095.9.

   FINAL3 beats the statue by +5.5 per episode (0.5%), on 18 of 20 paired seeds. These terms decide it:
   - forward velocity: +15.2
   - height (the walking pelvis is 3.5 mm lower): -4.3
   - lateral sway velocity: -3.6
   - tail: -1.4
   - heading: -1.2
   - nosedive: +0.9

   [measured: out/analyze_trex_stdout.txt]

   Pilot 3's actual starting point is the parent with log_std -1.5 and stochastic actions. It earns 456, which is **638 below the statue**, because it **falls in 18 of 20 episodes** (36 of 40 when seeds 3062-3081 are added). [measured: out/analyze_trex_stdout.txt, out/trex_parent_ls15_p2.json]

2. **At the end of the ramp (weight 2.0), the walks win clearly and forward velocity alone decides it.**
   - f2v0.25: 1300.6
   - FINAL3: 1234.9
   - parent, deterministic: 1093.9
   - statue: 1092.7
   - pilot-3 start: 422
   - failed fast attempt (f2v0.35): 317

   Margins over the statue: FINAL3 +142 (forward +152), f2v0.25 +208 (forward +226), both on 20 of 20 seeds. [measured: out/analyze_trex_stdout.txt]

3. **A slow walk beats standing everywhere on the ramp.**
   - The two lines cross at forward weight 0.127 for FINAL3 and 0.158 for f2v0.25. Both are below the ramp's starting value of 0.2. [measured: out/analyze_trex_stdout.txt, the ramp section]
   - The margin is small until mid-ramp. FINAL3 leads by +5.5 at 0.2, +26 at 0.47 (the end of the 300k-step warm-up) and +74 at 1.1 (1M steps).
   - Below a weight of about 0.22, more speed does not pay. The faster walk's extra crouch and sway cost 88 per episode per m/s, against 400·w per episode per m/s of forward reward. At 0.2 the two cancel: f2v0.25 earns 0.8 less than FINAL3. The faster walk pulls ahead only after about 24k training steps. [measured: return lines; inferred: linear in the weight]

4. **Speed a walk needs to beat the statue.** I used the measured non-forward deficit of the walk relative to the statue: 9.6 per episode for FINAL3 and 17.8 for f2v0.25. A full-length walk then beats the statue when its mean speed exceeds:
   - 0.118-0.220 m/s at weight 0.2;
   - 0.0095-0.020 m/s at weight 2.0.

   Per step, the walk wins when w·v/2.5 exceeds about 0.010-0.018. [measured deficit: out/analyze_trex_stdout.txt, break-even section; inferred: assumes a walk at that speed pays FINAL3's or f2v0.25's deficit]

   Falling cancels any speed gain. Every fast attempt fell, on all 20 seeds each (f2v0.35 at a mean 3.1 s, v06_v0.5 at 2.1 s, v10 at 2.1 s). Against the statue they lost 905-1020 per episode at weight 0.2 and 776-916 at weight 2.0. The loss is still 38-130 in the discounted return from reset (γ = 0.995). [measured: out/analyze_trex_stdout.txt, out/analyze_trex_extra_stdout.txt]

5. **Does the reward pay anything that a two-footed hop would get more cheaply? Yes, mainly speed.**
   - The locomotion reward does not depend on which gait produces the speed. It never reads which foot is down, whether the animal is in the air, or how hard it lands. Every support, contact, impact and airborne term has weight 0, and the alive bonus is not gated on support. [measured: out/inspect_weights_stdout.txt]
   - At weight 2.0, a 1.05-1.57 m/s gait (the September hops' speeds) would earn +840 to +1256 per episode from forward velocity. The slow walk earns +152, so the hop's figure is 5.5-8.3 times larger. [inferred: from the forward-velocity formula; assumes no fall and other terms near the statue's]
   - Most of the walk's small deficit comes from lateral sway, heading wobble and tail swing. All three come from alternating single support, and a left-right-symmetric two-footed hop would not pay them. [inferred]
   - The costs a hop could incur are capped. Smoothness is at most 0.05 per step, energy at most 0.002, tail at most 0.05 and lateral at most 0.1. A ±1 command flipped every step reaches the plant, and the smoothness and energy terms, through the 10 Hz filter at amplitude 0.239, which keeps 5.7% of the raw step-to-step change. For comparison, forward velocity pays 0.84-1.26 per step at those speeds. [inferred: from the term formulas and the measured weights]
   - The locomotion reward also no longer pays two-footed support. The parent's stance reward did (bilateral support 0.6, half the alive bonus gated on it, airborne-substep penalty 1.0, floor-impact penalty 2.0). [measured: fetched checkpoint 01_stance/stage_config.json]

6. **Compsognathus did learn to walk, yet its walk loses at its own ramp start.** Its ramp runs from 0.1 to 8.0 over 500k steps (code defaults; its config sets neither).
   - At weight 0.1, FINAL3 earns 1413.8 against the statue's 1499.9: **-86 per episode, losing on 20 of 20 seeds**.
   - The loss comes from the alive, posture and height terms, which pay nothing on unsupported steps (-26, -26 and -24). It also loses on spin (-5) and energy (-3).
   - At weight 8.0 the walk wins by +258 (20 of 20). The crossing is at weight 2.08, about 125k steps into its ramp.

   [measured: out/analyze_compso_stdout.txt]

   So the T. rex reward is at least as favourable to a slow walk at the ramp start as the compsognathus reward was. Unlike the compsognathus reward, the T. rex reward does not charge for being in the air. [inferred: comparison of the two measured ramps and of the code]

**What this means for pilot 3** [inferred: from the measurements above]. At pilot 3's start, about 640 per episode separates its falling start from both standing and walking. The walk's edge over standing is +5.5 to +26 until the end of the warm-up. That is 1-4% of the falling gap, so the first thing the reward teaches is to stop falling. From weight 0.47 to 2.0 the reward does favour walking over standing by +26 to +208. But any gait that sustains 1 m/s or more without falling earns more than these slow walks, and the reward does not prefer an alternating gait to a hop at the same speed.

## Setup: the reward the pilot trains on

- **Env.** Built with `build_env(ModelId(species, "anatomical"), "locomotion")` from the export's `environments/shared/config.py:401-429`. That call passes every `[env]` key of `configs/<species>/anatomical/locomotion.toml` to the env class, as training's `make_env` does (`train_base.py:369-394`). Training only adds a Monitor wrapper, which does not change the reward. Weights read from the built env are in `out/inspect_weights_stdout.txt` and `out/weights.json`.
- **T. rex terms with a non-zero weight**, read from the built env [measured: out/inspect_weights_stdout.txt]:
  - forward velocity: forward_vel_weight × clip(v·d/2.5, -1, 1), where v is the pelvis root velocity and d is the unit vector toward the prey at reset (prey angle -10.3 to +11.4 deg on these seeds);
  - alive: 0.5, ungated (support_conditioned_alive_fraction 0, so the gate at base_env.py:954-955 equals 1);
  - height: 0.3 × clip((z_pelvis - 0.7)/(0.926 - 0.7), 0, 1), one-sided (base_env.py:1063-1078);
  - heading: 0.3 × cos;
  - nosedive: 0.5;
  - posture: 0.2;
  - lateral velocity: 0.1;
  - smoothness: 0.05;
  - tail: 0.05;
  - energy: 0.002;
  - fall: -150.

  These are zero: bilateral support, foot load balance (and its airborne penalty), floor impact, airborne substeps, foot flatness, stance width, leg/neck/tail home pose, action saturation, jerk, spin, speed, idle, gait symmetry, drift and backward. `action_penalty_source` is "filtered".
- **Ramp** [measured: code read]. `train_base._stage_entry_shaping_callbacks` (train_base.py:1314-1367) adds `RewardRampCallback("forward_vel_weight", start = ramp_start_value 0.2, end = [env] forward_vel_weight 2.0, ramp_timesteps 2,000,000)` (locomotion.toml:161-162), but only on a warm start across a stage edge (`initialize_next_stage`). advancement.py:601-686 then:
  - ramps the weight linearly;
  - updates it once every 10k training steps, rounded to 3 decimals;
  - writes it through the env's `set_reward_weight` on the training VecEnv only (`model.get_env()`, advancement.py:640-645).

  A cross-stage load trains with `reset_num_timesteps=True` (train_base.py:1830), so the ramp starts from 0. The weight is 0.47 at 300k steps and 1.1 at 1M steps. I set the weight the same way, with `env.unwrapped.set_reward_weight`.
  [inferred: the evaluation env keeps 2.0 throughout, because the callback writes only the training env]
- **Warm-up** [measured: advancement.py:462-598, locomotion.toml:158-160]. `StageWarmupCallback` changes PPO's clip_range (0.02) and ent_coef (0.02) for 300k steps. **It does not touch the reward.**
- **Reward normalisation in training** (VecNormalize norm_reward True, clip_reward 50; constants.py:20-22). It divides every reward by one running scalar, which keeps the order of behaviours unchanged. The clip at ±50 would bind only if the running standard deviation of the discounted return fell below 3. [inferred: SB3 semantics; not measured]
- **Validation of the env build.** The statue on seeds 3042-3081 at weight 2.0 gives **1092.60 ± 5.41 (sample sd), 40 of 40 full horizon**. That reproduces the config's own reference, 1092.60 ± 5.35 with 40/40 (locomotion.toml:99-122; 5.41 × √(39/40) = 5.34). [measured: out/trex_zero.json + out/trex_zero_p2.json]
- **Reward bookkeeping check.** On every step of every episode, the env's returned reward equals the sum of the `info` `reward_*` terms (except `reward_total`), plus fall_penalty on a non-success termination. The largest per-step difference is 2.8e-14 and the largest per-episode difference is 1.2e-11. [measured: "sum check" column in out/analyze_*_stdout.txt]
- **Same dynamics at both weights.** Each episode ran at both weights (0.2 and 2.0; compsognathus 0.1 and 8.0) from the same reset seed. The predicted return NF + w·F matches the measured return to 1e-12 at both weights (the "pred-measured" column).
- **Behaviours.**
  - (a) Zero action, the statue.
  - (b) FINAL3, (c) f2v0.25 and (d) f2v0.35. These are the EXISTENCE stage's walker (`existence/scripts/wlib.py`, imported read-only), with parameters read from `existence/out/<tag>_<species>_s0.json`.
    - f2v0.25 and f2v0.35 use k_yp 0.05, a world-position term.
    - FINAL3 averages 0.187 m/s over the whole episode along the prey direction, and 0.209 m/s after the first 1 s, which matches the EXISTENCE figure of 0.21. f2v0.25: 0.280 and 0.311 m/s.
  - (e) The stance parent `stage1_final.zip` with `stage1_final_vecnorm.pkl`. It was loaded with `load_sb3_checkpoint` (no plant identity was passed). Observations were normalised with frozen statistics and actions came from `model.predict(deterministic=True)`; that path uses the actor only.
  - (f) The same parent with `policy.log_std` set to -1.5 in memory (nothing was saved) and `deterministic=False`. Torch was seeded with the reset seed, so both weights see the same noise. SB3's `predict` clips actions to [-1, 1], as PPO's rollout does. The checkpoint's own std is 0.007-0.013 on hip pitch, knee and ankle, 0.052 on hip roll, and 0.08-0.11 on neck and tail. -1.5 means a std of 0.223, 2-32 times higher.
- **Seeds.** Reset seeds 3042-3061 for everything.
  - Added: 3062-3081 for the statue at weight 2.0 (to complete the 40-episode reference) and for (f) at both weights.
  - Extra behaviours, labelled as extras: `parent_sto` (stochastic, own log_std), `parent_ls20` (log_std -2.0), and two more failed fast attempts, `v06_v0.5` (v_des 0.5) and `v10` (v_des 0.4, kv_x 0.5, max_x 0.8).

## T. rex results (seeds 3042-3061, 20 episodes each)

Return per 10 s episode (mean ± sd [min, max]). "Full" means episodes that reached 1000 steps. Speed is the episode-mean pelvis velocity along the prey direction. [measured: out/analyze_trex_stdout.txt, out/analyze_trex_extra_stdout.txt]

| behaviour | full | how it ends | speed m/s | return, w 0.2 | Δ vs statue (seeds won) | return, w 2.0 | Δ vs statue (seeds won) | discounted, w 0.2 / 2.0 |
|---|---|---|---|---|---|---|---|---|
| (a) statue | 20/20 | truncated | -0.003 | 1094.56 ± 6.16 [1071.6, 1099.0] | — | 1092.74 ± 6.75 | — | 216.97 / 215.26 |
| (b) FINAL3 walk | 20/20 | truncated | 0.187 | **1100.10** ± 7.18 [1075.3, 1106.1] | +5.54 [-1.54, +13.56] (18/20) | 1234.88 ± 11.60 | +142.13 [+129.9, +151.0] (20/20) | 217.69 / 234.71 |
| (c) f2v0.25 walk | 20/20 | truncated | 0.280 | 1099.29 ± 6.91 | +4.73 [-5.03, +10.49] (18/20) | **1300.58** ± 9.61 | +207.83 [+198.8, +215.3] (20/20) | 217.75 / 244.30 |
| (d) f2v0.35 (fails) | 0/20 | fallen 20, at 2.2-3.7 s | 0.565 | 189.10 ± 56.02 | -905.46 (0/20) | 316.82 ± 94.81 | -775.93 (0/20) | 138.16 / 177.30 |
| (e) parent, deterministic | 20/20 | truncated | -0.003 | 1095.91 ± 5.42 | +1.35 [-0.46, +4.98] (19/20) | 1093.91 ± 6.00 | +1.17 (19/20) | 217.51 / 215.65 |
| (f) parent, log_std -1.5, stochastic | 2/20 | fallen 10, head_contact 7, nosedive 1; mean length 569 steps | -0.075 | 456.09 ± 323.72 [5.9, 1057.3] | -638.47 (0/20) | 421.96 ± 315.05 | -670.79 (0/20) | 166.63 / 156.69 |
| (f) on seeds 3062-3081 | 2/20 | head_contact 11, fallen 7; mean length 500 | — | 387.27 ± 312.62 | — | 367.16 ± 289.21 | -725.30 vs the statue on 3062-3081 | — |
| extra: parent, own log_std, stochastic | 20/20 | truncated | -0.003 | 1095.72 ± 5.23 | +1.15 (19/20) | 1093.69 ± 5.83 | +0.95 (18/20) | 217.49 / 215.63 |
| extra: parent, log_std -2.0, stochastic | 20/20 | truncated | -0.023 | 1086.49 ± 5.94 | -8.08 (1/20) | 1069.84 ± 6.31 | -22.91 (0/20) | 216.16 / 211.50 |
| extra: v06_v0.5 (fails) | 0/20 | fallen 20, mean 2.14 s | 0.600 | 85.80 ± 13.06 | -1008.76 | 178.27 ± 29.52 | -914.48 | 92.90 / 131.82 |
| extra: v10 (fails) | 0/20 | fallen 20, mean 2.06 s | 0.711 | 74.93 ± 19.59 | -1019.63 | 177.04 ± 19.54 | -915.70 | 86.75 / 134.01 |

The discounted return is Σ 0.995^t r_t from the reset (PPO's γ for this stage).

Mean per-term sums per episode at weight 0.2. Columns that are 0 for every behaviour are omitted. At weight 2.0 only `forward` changes, by a factor of 10. [measured: out/analyze_trex_stdout.txt]

| term | statue | FINAL3 | f2v0.25 | f2v0.35 | parent det | parent ls -1.5 |
|---|---|---|---|---|---|---|
| alive | 500.00 | 500.00 | 500.00 | 153.72 | 500.00 | 284.48 |
| height | 299.97 | 295.64 | 288.34 | 85.28 | 299.95 | 165.77 |
| heading | 296.27 | 295.11 | 295.11 | 90.03 | 296.67 | 161.46 |
| forward (w 0.2) | -0.20 | 14.97 | 22.36 | 14.19 | -0.22 | -3.79 |
| forward (w 2.0) | -2.02 | 149.75 | 223.65 | 141.91 | -2.22 | -37.92 |
| lateral | -0.12 | -3.70 | -4.05 | -1.40 | -0.06 | -0.72 |
| tail | -0.17 | -1.57 | -2.07 | -0.83 | -0.09 | -1.60 |
| nosedive | -1.13 | -0.24 | -0.26 | -1.70 | -0.29 | -12.38 |
| posture | -0.06 | -0.04 | -0.05 | -0.12 | -0.04 | -2.02 |
| energy | 0.00 | -0.06 | -0.08 | -0.07 | -0.01 | -0.03 |
| smoothness | 0.00 | -0.01 | -0.01 | -0.01 | -0.00 | -0.08 |
| fall penalty | 0 | 0 | 0 | -150.00 | 0 | -135.00 |
| **return, w 0.2** | 1094.56 | 1100.10 | 1099.29 | 189.10 | 1095.91 | 456.09 |

**Return along the ramp.** For each behaviour the return is exactly NF + w·F. Here F is the forward term at unit weight and NF is the sum of all other terms. The weight w at training step t is 0.2 + 1.8·t/2M. [measured: out/analyze_trex_stdout.txt, out/analyze_trex_extra_stdout.txt]

| behaviour | NF | F | R(0.2), step 0 | R(0.47), 300k | R(1.1), 1M | R(2.0), 2M+ | weight at which it ties the statue |
|---|---|---|---|---|---|---|---|
| statue | 1094.77 | -1.01 | 1094.56 | 1094.29 | 1093.65 | 1092.74 | — |
| FINAL3 | 1085.13 | 74.87 | 1100.10 | 1120.32 | 1167.49 | 1234.88 | 0.127 (below the ramp start) |
| f2v0.25 | 1076.93 | 111.82 | 1099.29 | 1129.49 | 1199.94 | 1300.58 | 0.158 (below the ramp start) |
| f2v0.35 | 174.91 | 70.95 | 189.10 | 208.26 | 252.96 | 316.82 | 12.8 (never on this ramp) |
| parent det | 1096.13 | -1.11 | 1095.91 | 1095.61 | 1094.91 | 1093.91 | ahead by +1.4 to +1.2 throughout |
| parent ls -1.5 | 459.88 | -18.96 | 456.09 | 450.97 | 439.02 | 421.96 | never (it walks backward, so higher weight hurts it) |

**Per-step rates by support state.** "Double" means both feet carry at least 42 N (5% of body weight) on the step's lowest substep touch force; "single" means exactly one does. [measured: out/steps_trex_stdout.txt]
- **Statue:** 99.8% double support; pelvis at 0.9263 m.
- **FINAL3:** 91% single support; pelvis at 0.9228 m (3.5 mm lower). Height costs 1.33 per episode per mm below 0.926 m [inferred: from the formula].
  - The lateral term costs 0.0037 per step, which corresponds to a mean |lateral pelvis velocity| of 0.185 m/s.
  - These costs are nearly the same in single and double support. They come from the gait's sway, not from which foot is down.
- **f2v0.25:** pelvis at 0.9173 m (9 mm lower).
- **Noisy parent (log_std -1.5):** 52% double support, 42% single, 5.7% with neither foot above 42 N. It drifts backward at 0.083 m/s and pays a nosedive cost of 0.022 per step before it falls.

## Hop reasoning, from the terms (no hop controller was run)

- **Only six quantities reach the reward:** pelvis velocity (forward and lateral), pelvis height, pelvis orientation (heading, posture, nosedive), tail-tip angular speed, the filtered command (energy, smoothness), and the fall. No foot or contact quantity appears in any term with a non-zero weight. [measured: out/inspect_weights_stdout.txt; code tyrannosaurus_rex/base_env.py:910-1329]
- **Flight is free.** The alive, height and heading terms pay during flight steps exactly as during support. [measured: weights; code]
- **Height is one-sided** (base_env.py:1063-1078). Pelvis height above 0.926 m earns no more than standing, so a hop's flight phase gains nothing, and its landing crouch would cost 1.33 per episode per mm. The net sign for a real hop is unknown. [inferred]
- **Speed pays 0.8 per step per m/s at weight 2.0, and nothing else in the reward comes close.** The upper bounds of the motion-priced terms, per step, are:
  - smoothness: 0.05;
  - energy: 0.002;
  - tail: 0.05;
  - lateral: 0.1;
  - posture: 0.2·(tilt/60°)²;
  - nosedive: 0.5 × the forward-pitch excess.

  Each smoothness and energy bound is reached only if every actuator is pinned at ±1 or swings its full range every step. Even a ±1 command flipped every 10 ms reaches these terms at amplitude 0.239 (α = 0.386; 5.7% of its raw step-to-step change). [inferred: arithmetic in CMDLOG at 01:07Z]
- **Comparison with the hops.** The September hops (7.7 and 9.2 Hz, 35-43% airborne, 1.05-1.57 m/s; export docs/GAIT_QUALITY_PLAN_2026_09.md lines 86 and 89, not re-measured here) would be paid 840-1256 per episode for speed at weight 2.0. The slow walk gets 150. [inferred]
- **Compsognathus prices flight; the T. rex reward does not.** The compsognathus reward multiplies alive, posture and height by a support flag: the sum of both feet's lowest substep force must exceed 0.39 N (compsognathus_longipes/base_env.py:250-261). Each unsupported step therefore forfeits 1.3 of its 1.5 per step. [measured: code]
  - Its FINAL3 walk lost 76.3 per episode on these three terms. Most of that came from the 4.6% of steps where neither foot carried 5% of body weight on the step's lowest substep force: on those steps alive, posture and height averaged 0.0006, 0.0004 and 0.0004 per step, against 0.5, 0.4 and 0.4 standing. The rest came from one fall at 17.4 s. [measured: out/analyze_compso_stdout.txt, out/steps_compso_stdout.txt]
- **The export's own audit agrees.** docs/GAIT_QUALITY_PLAN_2026_09.md line 394 says that in the T. rex locomotion reward "alive 0.5 with no contact condition ... paid airborne ... nothing rewards alternation". It reports the September hops being paid alive 164 / 205 and forward 261 / 475 per episode while airborne (seeds 42 / 44). [doc claim, not re-measured here]

## Compsognathus (seeds 3042-3061; ramp 0.1 to 8.0)

The ramp values are the train_base.py:1362-1364 defaults, because compsognathus_longipes/anatomical/locomotion.toml sets no ramp keys. [measured: code]

| behaviour | full | return, w 0.1 | Δ vs statue | return, w 8.0 | Δ vs statue |
|---|---|---|---|---|---|
| statue | 20/20 | 1499.94 ± 0.06 | — | 1499.85 ± 0.39 | — |
| FINAL3 walk (0.044 m/s) | 19/20 (1 fallen at 17.4 s) | 1413.83 ± 53.71 | -86.11 [-304.0, -42.9] (0/20) | 1757.65 ± 59.55 | +257.81 [+20.3, +307.3] (20/20) |

Terms for FINAL3 relative to the statue, per episode:

| term | w 0.1 | w 8.0 |
|---|---|---|
| forward | +4.35 | +348.27 |
| alive | -25.98 | -25.98 |
| posture | -25.91 | -25.91 |
| height | -24.44 | -24.44 |
| spin | -5.06 | -5.06 |
| energy | -3.09 | -3.09 |
| heading | -2.38 | -2.38 |
| lateral | -2.17 | -2.17 |
| fall | -1.25 | -1.25 |

The walk ties the statue at weight 2.08, about 125k steps into the 500k-step ramp. [measured: out/analyze_compso_stdout.txt]

Scale comparison [inferred: arithmetic from the formulas]. The forward term's slope relative to the statue's per-step reward is 8.3 statue-rates per unit of v/√(gL) for compsognathus, against 2.1 for the T. rex. Speed earns compsognathus about 3.8 times more of its standing reward than it earns the T. rex. On the other hand, compsognathus' cap is 0.25 m/s (0.16 √(gL)), while the T. rex gate needs 1.0 m/s (0.34 √(gL)).

## Limits

- The walks are hand-built controllers (the EXISTENCE stage's), not learned policies. The break-even speeds assume a faster walk pays the same non-forward deficit as FINAL3 or f2v0.25. Among those two walks, the deficit grows by about 88 per episode per m/s. A learned walk could sway or crouch more or less than these.
- f2v0.25 and f2v0.35 read world position through k_yp 0.05, which the policy observation does not carry. That does not change what the reward pays for them.
- The hop is reasoned from the reward formulas and the measured weights, not simulated. The +840 to +1256 figures assume a hop that never falls and pays statue-like non-forward terms.
- Episode returns and the discounted return from the reset are the reported readouts. PPO's per-state advantages, its value targets under VecNormalize, and what training would actually find were not measured.
- The pilot's training env updates its observation statistics as it runs. I used the checkpoint's frozen statistics.
- (f) uses one noise draw per seed (40 seeds in total). The fall rate 36/40 has a 95% Wilson interval of about 0.77-0.96 [inferred: arithmetic]. The extra log_std points (own std and -2.0: 20/20 full episodes each) bracket where falling sets in, but they are not a scan.
- The parent was loaded without a plant-identity check (`plant_identity=None`). The pilot's warm start would check it.

## Slips (all recorded in CMDLOG.txt)

- Many read-only commands (cat, sed, grep, ls of the export, the existence folder, the fetched checkpoint folder and the persisted FEAS_RESULTS copy) were not logged literally when they ran. Early CMDLOG lines abbreviate paths or summarise commands. All are listed after the fact in the CMDLOG NOTE (about 01:08Z).
- One CMDLOG line gave a wrong launch time ("01:2xZ") for the first background loop; an appended line corrects it to 00:49:27Z.
- Inline python3 JSON readers (no simulation) used the system python3 without nice, OMP_NUM_THREADS or TMPDIR. They were logged after the fact.
- The Monitor-tool watch loops and the shells of the two background for-loops ran without nice. Every python process inside them went through `scripts/r.sh` with nice -n 10, OMP_NUM_THREADS=1 and TMPDIR.
- The harness wrote its own background-task output files (containing "done") under `/tmp/claude-0/.../tasks/`, outside this folder.
- Torch created `tmp/torchinductor_root` inside this folder's TMPDIR.
- No venv was created: the shared venv's python had everything needed and nothing was installed.

## Files

- Scripts in `scripts/`:
  - `r.sh`: wrapper that logs the literal command and applies nice, OMP_NUM_THREADS and TMPDIR.
  - `inspect_weights.py`: weight dump.
  - `rollout.py`: episodes, per-term sums and the bookkeeping check.
  - `analyze.py`: tables, paired differences, break-even and ramp lines.
  - `steps.py`: per-step rates by support state.
- Results in `out/`:
  - Per-episode records: `trex_<behaviour>.json`, `compso_<behaviour>.json`.
  - Per-step arrays at the first weight: `*_steps.npz`.
  - Tables: `analyze_trex_stdout.txt`, `analyze_trex_extra_stdout.txt`, `analyze_compso_stdout.txt`, `steps_trex_stdout.txt`, `steps_compso_stdout.txt`.
  - Summaries: `summary_trex.json`, `summary_trex_extra.json`, `summary_compso.json`, `steps_*.json`.
  - Weights: `weights.json`, `inspect_weights_stdout.txt`.
  - Smoke runs: `smoke_*`.
