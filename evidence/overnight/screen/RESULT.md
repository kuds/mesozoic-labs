# SCREEN: which minimal reward change ranks a walk above standing and an equally fast hop?

Stage folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/screen/`. Paths below are relative to it.

- **When:** 05:01-05:55 UTC on 2026-11-04 (time box: write-up by 09:15 UTC).
- **What was run:** 21 behaviours on the real r10 T. rex locomotion env of main 4d5f4e7 (the export's own `build_env(ModelId("tyrannosaurus_rex","anatomical"), "locomotion")`), reset seeds 3042-3061, 20 episodes each. A held-out panel (seeds 3062-3081) repeats 12 of them.
  - Episodes were driven only through `env.step(action)`, with every action asserted finite and inside [-1, 1].
  - The hand-built controllers read the env state read-only (they compute on private MjData copies).
  - The stance parent ran through `load_sb3_checkpoint` with its frozen observation statistics and `predict(deterministic=True)`.
- **What was not done:**
  - No training or fitting of any kind.
  - No change to the repository, the export, any config or the physics. No HYPOTHETICAL plant.
  - Every HYPOTHETICAL reward setting and gate is recomputed offline from per-step quantities the env recorded.
  - The only weight ever set in a live env is the forward weight, through the env's own `set_reward_weight` (the call `RewardRampCallback` makes), for 30 weight-0.2 check episodes.
- **How:** one simulation process at a time through `scripts/sim.sh` (nice -n 10, OMP_NUM_THREADS=1, TMPDIR in this folder, `optB/venv/bin/python -I -B`). Every command is in `CMDLOG.txt`.
- **Probe:** the frozen gait probe `docs/investigations/gait_2026_09/gait_probe.py` of the export, loaded read-only (sha256 in `out/SHA256SUMS_inputs.txt`). I used its own `rollout_episode`, floor-truth substep hook and `analyze_episode`; the class is taken over t ≥ 1 s.
- **Units:** m, s, m/s. BW = 840.9 N; L = 0.8766 m (the probe's leg length). "Return" is the undiscounted sum over one 10 s, 1000-step episode. w is forward_vel_weight: 0.2 at the start of the ramp, 2.0 at its end.
- **Standing** below means the best of three on the same seed: the statue (zero action), the stance parent (deterministic) and W02 (stepping in place). **Stable walks** means the six walks that finish all 20 episodes: FINAL3, V01, V05, V24, V18, V35 (0.2 to 1.2 m/s). Each is paired with an F hop at the same speed command.

## Short answer

1. **Main's reward, reproduced exactly, prefers the hop.**
   - Reproduction: I rebuilt main's per-step reward from the recorded terms, in the env's own summation order. 643,346 of 643,346 steps and 660 of 660 returns are bit-identical to what `env.step` returned (both panels). The weight-0.2 returns recomputed offline equal the env's own weight-0.2 runs exactly (30 of 30 episodes, difference 0.0). [measured: out/screen_main_stdout.txt, out/screen_HO_stdout.txt]
   - At w 2.0, each hop from 0.7 to 1.2 m/s beats the walk at the same speed on every seed: V24−H23 −60.1, V18−H24 −85.7, V35−HF120 −111.0 per episode (0 of 20 seeds each).
   - At w 0.2 the same hops beat the same walks by 28-35 per episode (0 of 20 each), and the crouched 0.7 m/s walk V24 loses to standing (−16.8, 0 of 20).
   - The slow pairs (0.2-0.5 m/s) already go the walk's way, by +11 to +31 per episode (20/20 seeds, except 0.3 m/s at w 2.0 on 19/20 and 0.5 m/s at w 2.0 on 17/20). [measured: out/md_main_stdout.txt, Margins]
2. **No single change passes both tests at both ends of the ramp.** [measured: out/verdict_main_stdout.txt, out/verdict_HO_stdout.txt]
   - R1 (the compsognathus support gate) and R2 (R1 + the airborne penalty) fix the hop. The walk leads its matched hop by +325 to +421 (R1) and +836 to +922 (R2) per episode, on every seed, at both weights.
   - They do nothing for standing at w 0.2: V24 stays at −17.1 (0/20). The 1.0-1.2 m/s walks drop to +2 to +4 (13-17 of 20 seeds), because R1 charges them for landing bounces and quick stance handovers (item 4).
   - R3 (`forward_vel_max` 2.5 → 1.5) fixes standing at w 0.2 on the first panel: every walk is ahead on every seed, the worst being V24 at +13.1 (seed minimum +1.9). But it widens the hop's lead (V35−HF120 −138.9 at w 2.0).
   - R5 (event reward, w_td 0.1) fixes standing (≥ +71 at w 0.2) and the hop at w 0.2 (≥ +53). At w 2.0 it does not fix the hop above 0.7 m/s: V18−H24 −2.1 (8/20), V35−HF120 −24.2 (2/20). Even at w_td 0.2 (40% of the alive bonus) it wins 119 of 120 walk-seed pairs on the first panel and 59 of 60 on the held-out panel. [measured: out/verdict_wtd0.2_stdout.txt, out/verdict_HO_wtd0.2_stdout.txt]
3. **Minimal setting.**
   - **Of the six listed settings, R6 (= R2 + R3 + R5) is the only one that passes on all 40 seeds.** Every stable walk beats standing and its matched hop at both w 0.2 and w 2.0 on every seed of both panels.
     - Standing margin: ≥ +101 at w 0.2 and ≥ +332 at w 2.0 (seed minima +87 and +307).
     - Matched-hop margin: ≥ +894 (seed minimum +778).
   - **R4 (= R2 + R3) passes on the first panel but not on the held-out one.** Its standing margin at w 0.2 is thin: V24 +12.8 (seed minimum +1.9) on seeds 3042-3061, and +14.9 with one held-out seed at −2.5. [measured: out/verdict_HO_stdout.txt]
   - **The airborne penalty is not what decides this test.** The two-change extras do as well as R4 and R6:
     - R1 + R3 (R13) behaves like R4: the same thin standing margin and the same held-out failure, with hop margins ≥ +296.
     - **R1 + R5 (R15) is the smallest combination that passes on all 40 seeds:** standing ≥ +71 at w 0.2 (seed minimum +59), matched hop ≥ +411. It still passes on all 40 seeds at w_td 0.05 (standing +27.1 and +28.9, seed minima +18.9 and +14.3). At w_td 0.025 it fails (V24 +5.0, 117/120). [measured: out/verdict_wtd0.05_stdout.txt, out/verdict_HO_wtd0.05_stdout.txt, out/verdict_wtd0.025_stdout.txt]
   - **Two later checks do not change this.** [measured: out/checks_stdout.txt, out/verdict_main_stdout.txt, out/verdict_HO_stdout.txt]
     - R5's causal touchdown detector misses touchdowns that the probe counts: on average 1.3, 2.9 and 3.9 per episode at 0.7, 1.0 and 1.2 m/s (at most 8 on one V29 episode). A landing after a one-step bounce is not registered. It matches the probe on every episode of the 0.2-0.5 m/s walks, and within 0-2 per episode for the hops and stepping in place. Taking the touchdowns from the probe's own debounce instead (R5p, extra) pays the 1.0-1.2 m/s walks 92-93 instead of 84-87. R5p alone still fails at w 2.0 (V35−HF120 −18.2, 96 of 120 walk-seed pairs). R15p and R6p pass on all 40 seeds.
     - A support gate that fires only on steps with a fully airborne substep (R1x, extra) gives the same verdicts. R15x passes on all 40 seeds. R13x, like R13, leaves the crouched walk below standing on 1 of the 40 seeds.
   - [inferred: every setting is affine in w] A margin that is positive at both ends of the ramp is positive along all of it.
4. **What the changes cost the walk** (per episode, mean over seeds 3042-3061) [measured: out/md_main_stdout.txt, "What each change does"]:
   - **R1:** 0 to 0.2 for the walks up to 0.7 m/s; 8.4 at 1.0 m/s; 14.7 at 1.2 m/s; 46.6 for the 1.5 m/s walk with landing bounces (V29). It removes alive, height and posture on the 11-19 control steps per episode (60 for V29) on which the two feet's substep-minimum touch forces sum to 33.6 N or less. It does not depend on w.
     - About half of those steps (5.8 of 10.8 at 1.0 m/s, 6.9 of 19.1 at 1.2 m/s) have no substep with both feet unloaded. They are quick stance handovers, in which each foot's minimum falls on a different substep. [measured: out/checks_stdout.txt]
     - Gating only steps that contain a fully airborne substep (R1x) costs these walks 4.0 and 9.4 instead of 8.4 and 14.7, and charges the hops about the same (−416 to −448). [measured: out/an_R1x_R5p_costs.txt]
   - **The airborne penalty** adds 0-2.5 for the walks up to 1.2 m/s (14.9 for V29).
   - **R3** pays the walks more, not less: +10 to +47 at w 0.2 and +100 to +465 at w 2.0. All of them stay below 1.5 m/s, and the knob raises the slope below the cap by 2.5/1.5.
   - **R5** pays the walks +84 to +92 and stepping in place +8.
   - **The hops** lose 388-464 under R1 and 873-990 under R2.
5. **The fastest hop is a separate question.** At w 2.0 the 2.0 m/s hop (H20) still out-earns every walk under R1 (1780 against at most 1738).
   - Under R15 it beats V18 and every slower walk (1780 against 1725 for V18) but not V35 (1824).
   - Under R13: V35 leads it by +65, V18 trails by −93.
   - Only settings with the airborne penalty put the 0.5-1.2 m/s walks above it: R2 by +167 to +478, R4 by +40 to +586, R6 by +132 to +673.
   - The 0.2 m/s walk stays below it under every setting except R25 (R2 + R5). The 0.3 m/s walk beats it only under R2 and R25. [measured: out/md_main_stdout.txt, "best hop"]
6. **Gates** (offline, on the same episodes; counts are per episode, of 20) [measured: out/md_main_stdout.txt and out/md_HO_stdout.txt, Gates]:
   - **G0 (current):**
     - It passes no strict walk: V18 0/20 and V35 0/20 on both panels.
     - It passes the 1.4-2.0 m/s hops (19-20 of 20) and the 1.2 m/s hop on 8 of 20.
     - It passes the failed fast walk V33, which falls on 17 of 20 episodes, on 12/20 and 13/20 episodes. It also passes V33 as a panel (mean length 837 and 827, at least 750; mean forward_vel 1.07 and 1.10).
   - **G1** (forward_vel over the last 5 s ≥ 1.0 m/s, plus the probe's walk class):
     - V35 passes 20/20; every hop 0/20.
     - V18 passes only 2/20 and 3/20: its last-5 s mean is 0.979-0.985 m/s along the prey direction.
     - V33 still passes 12/20 and 14/20, because falls after 7.5 s meet the length ≥ 750 rule. On the held-out panel it even passes as a panel.
     - Requiring full-length episodes cuts V33 to 3/20 and leaves V35 at 20/20.
   - **G2** (the same at 0.7 m/s): V18 and V35 20/20; V24 2/20 and 8/20 (its mean is 0.686-0.691); every hop 0/20.
7. **Recommendation** [inferred: for the maintainer, from the measurements above; nothing here changes the repository]:
   - The smallest change I found that robustly makes this reward prefer the walk at both ends of the ramp has two parts:
     - the compsognathus support gate on alive, posture and height (4% BW on the per-foot substep-minimum touch), which removes the hop's advantage;
     - a small alternating-touchdown event reward (w_td 0.05-0.1 per step equivalent), which lifts the walk above standing at the ramp start.

     That is R15. R6 adds the airborne penalty and `forward_vel_max` 1.5, and is needed only if a faster hop must also lose to a slower walk.
   - If the support gate is adopted, gating on "a substep with both feet unloaded" (R1x) rather than on "summed per-foot minima" (the compsognathus rule) halves what it charges the 1.0 m/s walk and leaves the hop's charge unchanged.
   - The event detector should follow the probe's debounce, which tolerates a one-step landing bounce (R5p).
   - `forward_vel_max` 1.5 without the event reward (R4, R13) gets the ramp start right only by a few points: −2.5 on the worst held-out seed.
   - Do not port the stance reward's floor-impact term at its 1.4 BW threshold. It would charge the 1.0-1.2 m/s walks 178-212 per episode [measured: out/impact_stdout.txt], several times their margin over standing at the ramp start.
   - For the gate, use G1 or G2 on full-length episodes. G0 accepts a falling fast walk and fast hops, and rejects every strict hand-built walk.

## Behaviours

Sources:
- the SPEED stage's walkers (`wlib2.Walker2`, params from `speed/out/<tag>.json`) and FINAL3 (`wlib.Walker`, EXISTENCE);
- the HOP stage's F hop (`hoplib.Hopper`, params from `hop/out/H23_v070_F.json`). Only `v_mps` changes for HF020, HF030, HF050, HF120 and HF150, which are speed commands the HOP stage did not validate;
- W02 (stepping in place) from the HOP stage;
- the stance parent `stage1_final.zip`.

The controller files are unchanged copies (sha256 in `out/copied_sha256.txt`).

Seeds 3042-3061 [measured: out/md_main_stdout.txt]:

| behaviour | what | full (of 20) | last-5 s forward_vel (m/s) | probe class (t ≥ 1 s) | unsupported steps / ep (R1) | airborne substep share |
|---|---|---|---|---|---|---|
| statue | zero action | 20 | 0.000 | stand 20 | 0 | 0 |
| parent | stance parent, deterministic | 20 | 0.000 | stand 20 | 0 | 0 |
| parent_ls15 | extra: pilot 3's start (log_std −1.5 in memory, stochastic) | 2 | −0.073 | mixed (shuffle, irregular, stepping in place) | 30.3 | 0.016 |
| W02 | stepping in place | 20 | 0.024 | stand (stepping in place) 20 | 0 | 0 |
| FINAL3 | walk, 0.2 m/s | 20 | 0.211 | walk 20 | 0 | 0 |
| V01 | walk, 0.3 m/s | 20 | 0.303 | walk 20 | 0 | 0 |
| V05 | walk, 0.5 m/s | 20 | 0.510 | walk 20 | 0.2 | 0 |
| V24 | walk, 0.7 m/s | 20 | 0.686 | walk 20 | 0.3 | 0 |
| V18 | walk, 1.0 m/s | 20 | 0.979 | walk 20 | 10.8 | 0.001 |
| V35 | walk, 1.2 m/s | 20 | 1.171 | walk 20 | 19.1 | 0.003 |
| V29 | walk, 1.5 m/s, with landing bounces | 17 (falls 7.1-8.7 s) | 1.341 | walk 20 | 60.0 | 0.016 |
| V33 | failed fast walk (1.8 command) | 3 (falls 6.6-9.3 s) | 1.441 | walk 19 | 67.0 | 0.036 |
| HF020 | hop, 0.2 m/s | 20 | 0.197 | hop 20 | 505 | 0.486 |
| HF030 | hop, 0.3 m/s | 20 | 0.292 | hop 20 | 508 | 0.489 |
| HF050 | hop, 0.5 m/s | 20 | 0.497 | hop 20 | 521 | 0.501 |
| H23 | hop, 0.7 m/s | 20 | 0.695 | hop 20 | 529 | 0.510 |
| H24 | hop, 1.0 m/s | 20 | 0.991 | hop 20 | 548 | 0.517 |
| HF120 | hop, 1.2 m/s | 20 | 1.202 | hop 20 | 567 | 0.514 |
| H25 | hop, 1.4 m/s | 20 | 1.412 | hop 20 | 578 | 0.521 |
| HF150 | hop, 1.5 m/s | 20 | 1.499 | hop 20 | 584 | 0.526 |
| H20 | hop, 2.0 m/s | 19 (fall 6.73 s) | 1.994 | hop 20 | 568 | 0.531 |

- **Speed-matched pairs:** FINAL3/HF020, V01/HF030, V05/HF050, V24/H23, V18/H24, V35/HF120; also V29/HF150, but V29 falls on 3 seeds. Within each stable pair the mean last-5 s speeds differ by 0.009-0.031 m/s. The walk is the faster one in the three pairs below 0.7 m/s, and the hop in the three from 0.7 m/s up.
- **Held-out panel** (seeds 3062-3081; statue, parent, W02, FINAL3, V24, V18, V35, V33, H23, H24, HF120, H20): the same picture. Every stable walk and hop runs 20/20, H20 falls twice (7.60, 8.31 s) and V33 17 times. [measured: out/md_HO_stdout.txt]
- **Cross-check against the earlier stages' own records** (same seed, same behaviour), 340 returns:
  - from REWARD, at both weights: the statue, FINAL3, the parent and the noisy parent (whose torch draw reproduces);
  - from HOP: H23, H24, H25, H20, W02 and V24.

  They agree to 3.0e-12 or better; the remainder is summation order. [measured: out/crossref_stdout.txt]

## Reproduction of main (the current setting)

- Per step, I rebuilt the reward as the env does (tyrannosaurus_rex/base_env.py `_get_reward_info`). That is the left-to-right sum of the 29 terms in the env's order, with the forward term recomputed as w·clip(v/2.5, −1, 1) from the recorded `info["forward_vel"]`. The −150 fall penalty is added on a non-success termination.
- Checks:
  - The three optional terms (floor impact, airborne substeps, lateral lean) are 0 on every step, as their weights are 0.
  - Result: bit-identical to `env.step`'s reward on 643,346 of 643,346 steps. The forward term recomputed from `forward_vel` equals `info["reward_forward"]` exactly. All 660 returns are bit-identical. [measured: out/screen_main_stdout.txt, out/screen_HO_stdout.txt]
- Weight 0.2: 30 episodes (statue, parent, V24, H23, V35, H20 × seeds 3042-3046) were run again with the env's forward weight set to 0.2 through `set_reward_weight`.
  - Their trajectories are identical to the weight-2.0 runs.
  - The offline weight-0.2 return of the weight-2.0 episode equals the env's own weight-0.2 return exactly (difference 0.0, 1000/1000 steps bit-identical). [measured: out/screen_main_stdout.txt "w02"]
- The weight never enters the dynamics, so every return at w 0.2 below is exact. [inferred: from the check above and the code]

## The hypothetical settings

All of them are computed OFFLINE. Anything not named is main's locomotion reward, unchanged. Every quantity is one the env recorded on that step: an `info` entry, or the probe's floor-truth arrays. The definitions of R1-R6 (with the event rule as first written, now R5a), R3s and the gates were written into CMDLOG.txt (05:09:06Z) before any episode was run. The R5 correction (05:16:38Z) came before any number was computed. The extras came later, as stated where each one is introduced.

- **main**: the env's own total, rebuilt as above.
- **R1** (the compsognathus support gate): alive, posture and height are each multiplied by
  support = 1[r_foot_contact + l_foot_contact > 0.04·BW].
  - The rule is compsognathus_longipes/base_env.py:200 and :250: `support = float(right + left > self._contact_threshold)`, with threshold body_mass·|g|·0.04, applied to that env's `_aggregated_foot_contact_forces()`.
  - `r_foot_contact` and `l_foot_contact` are the T. rex env's values of the same method: per-foot MIN touch force over the step's five substeps. With BW = 840.9 N (the env's `_body_weight_n`, pelvis subtree mass × g), the threshold is 33.6 N.
  - The T. rex posture term is a penalty (−0.2·min(tilt/max_tilt, 1)²), whereas compsognathus' posture term is a bonus, so gating it by support forgives tilt in the air. The extra R1np gates only alive and height. It differs from R1 by at most 3 per episode. [measured: out/md_main_stdout.txt]
- **R2** = R1 − 1.0·airborne_substeps/5. This is the stance reward's term (tyrannosaurus_rex/base_env.py:1220-1229; weight 1.0 in the stance parent's stage_config.json): the share of the step's substeps on which every foot's touch is ≤ 0.1 N.
- **R3**: `forward_vel_max` 2.5 → 1.5 in w·clip(v/vmax, −1, 1).
  - This is the env's own knob. It caps the pay at 1.5 m/s and also raises the slope below the cap by 2.5/1.5.
  - The extra **R3s** keeps the slope and only caps: w·clip(v, −2.5, 1.5)/2.5. It changes the stable walks by at most 4 per episode and costs the 1.2-2.0 m/s hops 22-278 at w 2.0. It passes no test that main fails. [measured: out/verdict_main_stdout.txt]
- **R4** = R2 + R3.
- **R5** = main + the event reward below. **R6** = R4 + R5.
- **Extras:**
  - R13 = R1 + R3, R15 = R1 + R5, R25 = R2 + R5, R35 = R3 + R5 (to find the smallest combination).
  - R5a and R6a use the pre-declared event rule; see slip 1.
  - Added after the consistency checks (out/checks_stdout.txt):
    - R5p, R15p, R6p take the touchdowns from the probe's own debounce and `_events` (non-causal by a step or two), paid at the same step as R5.
    - R1x, R15x, R13x gate alive, posture and height only on steps on which some substep has every foot at 0.1 N or less (`airborne_substeps` > 0).
- **Not included:** a bilateral-support term, as instructed.

### R5: the event reward as built here (the simplest version of GQ-15 (a) I could state)

GQ-15 (a) (docs/GAIT_QUALITY_PLAN_2026_09.md:197; §5.3 row `gait_phase_weight`) pays w·(T_i ÷ (feet·dt))·P·A·S at each debounced floor-truth touchdown. I kept its normalisation and its S. I replaced the phase match P by a yes/no test of alternation and non-simultaneity, and dropped A (the swing-duration factor).

1. **Foot down** on a control step: the floor-truth normal force on the foot's bodies exceeds 0.1 N on at least half of the five substeps. This is the frozen probe's `contact_arrays` "floor" rule.
2. **Touchdown of foot i registered at step k**: the foot is down at k−1 and k, and not down at k−3 and k−2. This is the probe's 2-step debounce (MIN_RUN_STEPS = 2), made causal by a one-step delay.
3. **Paid only if both hold:**
   - Alternation: the previous registered touchdown, of either foot, was by the other foot j.
   - Not simultaneous: foot j has no registered touchdown within ±2 control steps of k. This is the probe's rule for a simultaneous `B` event (MERGE_STEPS = 2). The pay is delivered at step k+2, so the reward stays causal.
4. **Amount**: w_td · min(T_i, 1.2 s)/(2·dt) · S.
   - T_i is the time since foot i's previous registered touchdown, i.e. that foot's own cycle time. The first touchdown is timed from the reset.
   - S = clip(Δx/(0.1·L), 0, 1), where Δx = (foot i − foot j)·(pelvis heading), measured at the registration step. Δx is how far the landing foot lands ahead of the other one. S reaches 1 at Δx = 8.8 cm.
   - If every touchdown is paid at S = 1, this averages w_td per control step whatever the cadence: two touchdowns per cycle T, each paid T/(2·dt)·w_td.
5. **Weight**: w_td = 0.1, pre-declared as one fifth of the alive bonus. The term is linear in w_td, so I also report a scan (0.025, 0.05, 0.1, 0.15, 0.2) and the break-even values.

**What it pays in practice** (unit-weight sums per episode, seeds 3042-3061) [measured: out/sens_stdout.txt]:
- The stable walks: 836-920 (V29: 810), i.e. about 0.84-0.92 of the per-step rate over a whole episode, because the 0.5 s settle and the first touchdowns are not paid.
- Stepping in place (W02): 80.5. Its feet land 0.8 cm apart along the heading (median), so S is small.
- The statue, the parent and every hop: 0. With the pre-declared one-sided rule (R5a), the hops get 0.5-4.1.
- The noisy parent: 68 (+6.8 per episode at w_td 0.1).

**What it can pay by construction** [inferred]: any alternating stepping that lands one foot ahead of the other, a run with flight included. R2 is what prices the flight.

## Verdict per setting

For each setting and ramp end: the worst of the six stable walks against the best standing behaviour on the same seed, and the worst walk against its speed-matched hop. "Seeds won" counts walk-seed pairs. Generated by scripts/verdict.py. [measured]

Seeds 3042-3061 (out/verdict_main_stdout.txt):

| setting | w | worst walk vs best standing: mean (walk) | per-seed min | seeds won (of walks x seeds) | worst walk vs matched hop: mean (pair) | per-seed min | seeds won | every walk ahead on every seed? |
|---|---|---|---|---|---|---|---|---|
| main | 0.2 | -16.8 (V24) | -25.4 | 89/120 | -35.1 (V24-H23) | -55.0 | 60/120 | no |
| main | 2.0 | +140.9 (FINAL3) | +125.8 | 120/120 | -111.0 (V35-HF120) | -173.3 | 56/120 | no |
| R1 | 0.2 | -17.1 (V24) | -25.4 | 83/120 | +380.8 (V24-H23) | +373.6 | 120/120 | no |
| R1 | 2.0 | +140.9 (FINAL3) | +125.8 | 120/120 | +324.7 (V35-HF120) | +273.3 | 120/120 | YES |
| R2 | 0.2 | -17.1 (V24) | -25.4 | 76/120 | +890.4 (V24-H23) | +879.2 | 120/120 | no |
| R2 | 2.0 | +140.9 (FINAL3) | +125.8 | 120/120 | +836.0 (V35-HF120) | +780.9 | 120/120 | YES |
| R3 | 0.2 | +13.1 (V24) | +1.9 | 120/120 | -37.0 (V24-H23) | -60.1 | 60/120 | no |
| R3 | 2.0 | +242.2 (FINAL3) | +219.9 | 120/120 | -138.9 (V35-HF120) | -224.3 | 51/120 | no |
| R4 | 0.2 | +12.8 (V24) | +1.9 | 120/120 | +888.6 (V24-H23) | +877.2 | 120/120 | YES |
| R4 | 2.0 | +242.2 (FINAL3) | +219.9 | 120/120 | +808.0 (V35-HF120) | +729.8 | 120/120 | YES |
| R5 | 0.2 | +71.4 (V24) | +63.2 | 120/120 | +53.2 (V24-H23) | +29.4 | 120/120 | YES |
| R5 | 2.0 | +231.1 (FINAL3) | +215.4 | 120/120 | -24.2 (V35-HF120) | -88.8 | 90/120 | no |
| R6 | 0.2 | +101.1 (V24) | +90.5 | 120/120 | +976.9 (V24-H23) | +966.2 | 120/120 | YES |
| R6 | 2.0 | +332.4 (FINAL3) | +309.6 | 120/120 | +894.7 (V35-HF120) | +814.2 | 120/120 | YES |
| R13 | 0.2 | +12.9 (V24) | +1.9 | 120/120 | +378.9 (V24-H23) | +371.4 | 120/120 | YES |
| R13 | 2.0 | +242.2 (FINAL3) | +219.9 | 120/120 | +296.7 (V35-HF120) | +222.2 | 120/120 | YES |
| R15 | 0.2 | +71.2 (V24) | +63.2 | 120/120 | +469.1 (V24-H23) | +461.1 | 120/120 | YES |
| R15 | 2.0 | +231.1 (FINAL3) | +215.4 | 120/120 | +411.4 (V35-HF120) | +357.7 | 120/120 | YES |
| R25 | 0.2 | +71.2 (V24) | +63.2 | 120/120 | +978.7 (V24-H23) | +968.2 | 120/120 | YES |
| R25 | 2.0 | +231.1 (FINAL3) | +215.4 | 120/120 | +922.7 (V35-HF120) | +865.3 | 120/120 | YES |
| R35 | 0.2 | +101.4 (V24) | +90.5 | 120/120 | +51.0 (V35-HF120) | +24.3 | 120/120 | YES |
| R35 | 2.0 | +332.4 (FINAL3) | +309.6 | 120/120 | -52.2 (V35-HF120) | -139.9 | 75/120 | no |
| R3s | 0.2 | -16.8 (V24) | -25.4 | 89/120 | -35.1 (V24-H23) | -52.8 | 60/120 | no |
| R3s | 2.0 | +140.9 (FINAL3) | +125.8 | 120/120 | -93.1 (V35-HF120) | -151.3 | 56/120 | no |
| R1np | 0.2 | -17.1 (V24) | -25.4 | 83/120 | +380.9 (V24-H23) | +373.7 | 120/120 | no |
| R1np | 2.0 | +140.9 (FINAL3) | +125.8 | 120/120 | +325.1 (V35-HF120) | +273.6 | 120/120 | YES |
| R5a | 0.2 | +71.4 (V24) | +63.2 | 120/120 | +53.1 (V24-H23) | +29.3 | 120/120 | YES |
| R5a | 2.0 | +231.1 (FINAL3) | +215.4 | 120/120 | -24.3 (V35-HF120) | -88.9 | 90/120 | no |
| R6a | 0.2 | +101.1 (V24) | +90.5 | 120/120 | +976.8 (V24-H23) | +966.0 | 120/120 | YES |
| R6a | 2.0 | +332.4 (FINAL3) | +309.6 | 120/120 | +894.7 (V35-HF120) | +814.2 | 120/120 | YES |
| R5p | 0.2 | +74.6 (V24) | +66.1 | 120/120 | +56.3 (V24-H23) | +38.4 | 120/120 | YES |
| R5p | 2.0 | +231.1 (FINAL3) | +215.4 | 120/120 | -18.2 (V35-HF120) | -79.8 | 96/120 | no |
| R15p | 0.2 | +74.4 (V24) | +66.1 | 120/120 | +472.2 (V24-H23) | +465.2 | 120/120 | YES |
| R15p | 2.0 | +231.1 (FINAL3) | +215.4 | 120/120 | +417.4 (V35-HF120) | +366.7 | 120/120 | YES |
| R6p | 0.2 | +104.3 (V24) | +93.2 | 120/120 | +980.0 (V24-H23) | +969.1 | 120/120 | YES |
| R6p | 2.0 | +332.4 (FINAL3) | +309.6 | 120/120 | +900.7 (V35-HF120) | +823.3 | 120/120 | YES |
| R1x | 0.2 | -17.0 (V24) | -25.4 | 83/120 | +380.7 (V24-H23) | +372.8 | 120/120 | no |
| R1x | 2.0 | +140.9 (FINAL3) | +125.8 | 120/120 | +328.0 (V35-HF120) | +277.1 | 120/120 | YES |
| R15x | 0.2 | +71.3 (V24) | +63.2 | 120/120 | +469.0 (V24-H23) | +461.1 | 120/120 | YES |
| R15x | 2.0 | +231.1 (FINAL3) | +215.4 | 120/120 | +414.7 (V35-HF120) | +361.5 | 120/120 | YES |
| R13x | 0.2 | +13.0 (V24) | +1.9 | 120/120 | +378.9 (V24-H23) | +370.7 | 120/120 | YES |
| R13x | 2.0 | +242.2 (FINAL3) | +219.9 | 120/120 | +300.0 (V35-HF120) | +226.1 | 120/120 | YES |

Held-out seeds 3062-3081, walks FINAL3, V24, V18, V35 (V01, V05 and the 0.2-0.5 m/s hops were not re-run) (out/verdict_HO_stdout.txt):

| setting | w | worst walk vs best standing: mean (walk) | per-seed min | seeds won (of walks x seeds) | worst walk vs matched hop: mean (pair) | per-seed min | seeds won | every walk ahead on every seed? |
|---|---|---|---|---|---|---|---|---|
| main | 0.2 | -15.2 (V24) | -30.5 | 55/80 | -35.7 (V35-HF120) | -59.2 | 0/60 | no |
| main | 2.0 | +141.0 (FINAL3) | +123.6 | 80/80 | -112.8 (V35-HF120) | -180.6 | 0/60 | no |
| R1 | 0.2 | -15.2 (V24) | -30.5 | 53/80 | +381.3 (V24-H23) | +362.3 | 60/60 | no |
| R1 | 2.0 | +141.1 (FINAL3) | +123.6 | 80/80 | +329.7 (V35-HF120) | +248.7 | 60/60 | YES |
| R2 | 0.2 | -15.2 (V24) | -30.5 | 50/80 | +890.4 (V24-H23) | +868.5 | 60/60 | no |
| R2 | 2.0 | +141.1 (FINAL3) | +123.6 | 80/80 | +837.6 (V35-HF120) | +757.9 | 60/60 | YES |
| R3 | 0.2 | +15.0 (V24) | -2.5 | 79/80 | -37.8 (V35-HF120) | -64.5 | 0/60 | no |
| R3 | 2.0 | +241.8 (FINAL3) | +217.4 | 80/80 | -133.6 (V35-HF120) | -254.0 | 0/60 | no |
| R4 | 0.2 | +14.9 (V24) | -2.5 | 79/80 | +888.7 (V24-H23) | +865.1 | 60/60 | no |
| R4 | 2.0 | +241.9 (FINAL3) | +217.4 | 80/80 | +816.8 (V35-HF120) | +689.3 | 60/60 | YES |
| R5 | 0.2 | +73.1 (V24) | +59.0 | 80/80 | +50.4 (V35-HF120) | +29.9 | 60/60 | YES |
| R5 | 2.0 | +231.2 (FINAL3) | +213.5 | 80/80 | -26.7 (V35-HF120) | -91.5 | 35/60 | no |
| R6 | 0.2 | +103.2 (V24) | +87.0 | 80/80 | +977.0 (V24-H23) | +954.6 | 60/60 | YES |
| R6 | 2.0 | +332.1 (FINAL3) | +307.3 | 80/80 | +902.9 (V35-HF120) | +778.0 | 60/60 | YES |
| R13 | 0.2 | +14.9 (V24) | -2.5 | 79/80 | +379.6 (V24-H23) | +358.9 | 60/60 | no |
| R13 | 2.0 | +241.9 (FINAL3) | +217.4 | 80/80 | +305.3 (V18-H24) | +169.9 | 60/60 | YES |
| R15 | 0.2 | +73.0 (V24) | +59.0 | 80/80 | +469.5 (V24-H23) | +451.8 | 60/60 | YES |
| R15 | 2.0 | +231.3 (FINAL3) | +213.5 | 80/80 | +415.8 (V35-HF120) | +337.5 | 60/60 | YES |
| R25 | 0.2 | +73.0 (V24) | +59.0 | 80/80 | +978.7 (V24-H23) | +958.0 | 60/60 | YES |
| R25 | 2.0 | +231.3 (FINAL3) | +213.5 | 80/80 | +923.7 (V35-HF120) | +847.0 | 60/60 | YES |
| R35 | 0.2 | +103.2 (V24) | +87.0 | 80/80 | +48.3 (V35-HF120) | +24.7 | 60/60 | YES |
| R35 | 2.0 | +332.0 (FINAL3) | +307.3 | 80/80 | -47.5 (V35-HF120) | -165.2 | 21/60 | no |
| R3s | 0.2 | -15.2 (V24) | -30.5 | 55/80 | -33.9 (V24-H23) | -57.0 | 0/60 | no |
| R3s | 2.0 | +141.0 (FINAL3) | +123.6 | 80/80 | -91.0 (V35-HF120) | -167.8 | 0/60 | no |
| R1np | 0.2 | -15.2 (V24) | -30.5 | 53/80 | +381.4 (V24-H23) | +362.3 | 60/60 | no |
| R1np | 2.0 | +141.1 (FINAL3) | +123.6 | 80/80 | +330.1 (V35-HF120) | +249.3 | 60/60 | YES |
| R5a | 0.2 | +73.1 (V24) | +59.0 | 80/80 | +50.2 (V35-HF120) | +29.8 | 60/60 | YES |
| R5a | 2.0 | +231.2 (FINAL3) | +213.5 | 80/80 | -26.9 (V35-HF120) | -91.6 | 35/60 | no |
| R6a | 0.2 | +103.2 (V24) | +87.0 | 80/80 | +976.9 (V24-H23) | +954.4 | 60/60 | YES |
| R6a | 2.0 | +332.1 (FINAL3) | +307.3 | 80/80 | +902.7 (V35-HF120) | +777.8 | 60/60 | YES |
| R5p | 0.2 | +76.5 (V24) | +61.7 | 80/80 | +56.9 (V35-HF120) | +32.2 | 60/60 | YES |
| R5p | 2.0 | +231.2 (FINAL3) | +213.5 | 80/80 | -20.2 (V35-HF120) | -89.2 | 37/60 | no |
| R15p | 0.2 | +76.4 (V24) | +61.7 | 80/80 | +472.9 (V24-H23) | +454.5 | 60/60 | YES |
| R15p | 2.0 | +231.3 (FINAL3) | +213.5 | 80/80 | +422.3 (V35-HF120) | +342.3 | 60/60 | YES |
| R6p | 0.2 | +105.3 (FINAL3) | +89.7 | 80/80 | +980.4 (V24-H23) | +957.3 | 60/60 | YES |
| R6p | 2.0 | +332.1 (FINAL3) | +307.3 | 80/80 | +909.4 (V35-HF120) | +782.8 | 60/60 | YES |
| R1x | 0.2 | -15.2 (V24) | -30.5 | 53/80 | +381.2 (V24-H23) | +362.3 | 60/60 | no |
| R1x | 2.0 | +141.0 (FINAL3) | +123.6 | 80/80 | +332.6 (V35-HF120) | +253.3 | 60/60 | YES |
| R15x | 0.2 | +73.0 (V24) | +59.0 | 80/80 | +469.4 (V24-H23) | +451.8 | 60/60 | YES |
| R15x | 2.0 | +231.2 (FINAL3) | +213.5 | 80/80 | +418.7 (V35-HF120) | +342.0 | 60/60 | YES |
| R13x | 0.2 | +14.9 (V24) | -2.5 | 79/80 | +379.5 (V24-H23) | +358.9 | 60/60 | no |
| R13x | 2.0 | +241.8 (FINAL3) | +217.4 | 80/80 | +307.6 (V18-H24) | +174.4 | 60/60 | YES |

Event-weight scan, R15 = R1 + R5 and R5 alone, worst case over walks (out/verdict_wtd*_stdout.txt, out/verdict_HO_wtd*_stdout.txt): see the files; summary in the short answer, item 3.

## Returns per episode (seeds 3042-3061)

Generated by scripts/mdtables.py (out/md_main_stdout.txt, which also holds the extras R13, R15, R25, R35, R3s, R1np, R5a, R6a). [measured]

### Return per episode, mean over seeds 3042-3061, forward weight 0.2 (rank in brackets, 1 = highest of the 21)

| behaviour | main | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---|---|---|---|---|---|---|
| statue | 1094.6 (14) | 1094.6 (7) | 1094.6 (7) | 1094.4 (16) | 1094.4 (8) | 1094.6 (16) | 1094.4 (9) |
| parent | 1095.9 (13) | 1095.9 (6) | 1095.9 (6) | 1095.8 (15) | 1095.8 (7) | 1095.9 (15) | 1095.8 (8) |
| parent_ls15 | 456.1 (21) | 432.5 (21) | 422.6 (12) | 453.6 (21) | 420.1 (12) | 462.9 (21) | 426.9 (12) |
| W02 | 1029.1 (19) | 1029.1 (9) | 1029.1 (9) | 1030.2 (19) | 1030.2 (10) | 1037.2 (19) | 1038.3 (10) |
| FINAL3 | 1100.1 (10) | 1100.1 (3) | 1100.1 (2) | 1110.1 (12) | 1110.1 (5) | 1190.2 (4) | 1200.2 (5) |
| V01 | 1097.4 (12) | 1097.4 (5) | 1097.4 (5) | 1112.0 (11) | 1112.0 (4) | 1187.4 (5) | 1202.0 (4) |
| V05 | 1113.3 (8) | 1113.1 (1) | 1113.1 (1) | 1136.3 (9) | 1136.1 (3) | 1205.3 (1) | 1228.1 (2) |
| V24 | 1079.1 (15) | 1078.9 (8) | 1078.8 (8) | 1108.9 (14) | 1108.6 (6) | 1167.4 (6) | 1196.9 (6) |
| V18 | 1107.8 (9) | 1099.4 (4) | 1098.4 (3) | 1148.0 (7) | 1138.5 (2) | 1191.4 (3) | 1222.1 (3) |
| V35 | 1115.1 (6) | 1100.4 (2) | 1097.9 (4) | 1161.6 (5) | 1144.4 (1) | 1201.9 (2) | 1231.2 (1) |
| V29 | 1062.3 (18) | 1015.7 (10) | 1000.9 (10) | 1110.0 (13) | 1048.6 (9) | 1143.3 (10) | 1129.6 (7) |
| V33 | 796.6 (20) | 746.4 (11) | 717.3 (11) | 833.9 (20) | 754.6 (11) | 862.4 (20) | 820.3 (11) |
| HF020 | 1068.8 (17) | 681.2 (19) | 195.4 (14) | 1079.5 (18) | 206.2 (20) | 1068.8 (18) | 206.2 (20) |
| HF030 | 1077.7 (16) | 686.7 (18) | 197.9 (13) | 1092.5 (17) | 212.7 (19) | 1077.7 (17) | 212.7 (19) |
| HF050 | 1097.6 (11) | 691.7 (17) | 190.8 (15) | 1120.9 (10) | 214.2 (18) | 1097.6 (14) | 214.2 (18) |
| H23 | 1114.2 (7) | 698.1 (13) | 188.4 (16) | 1145.8 (8) | 220.0 (16) | 1114.2 (13) | 220.0 (16) |
| H24 | 1135.7 (4) | 701.2 (12) | 184.4 (17) | 1180.0 (4) | 228.7 (14) | 1135.7 (11) | 228.7 (14) |
| HF120 | 1148.1 (3) | 697.8 (14) | 184.0 (18) | 1197.4 (3) | 233.3 (13) | 1148.1 (9) | 233.3 (13) |
| H25 | 1153.7 (2) | 694.5 (16) | 173.1 (19) | 1202.7 (2) | 222.0 (15) | 1153.7 (8) | 222.0 (15) |
| HF150 | 1161.6 (1) | 697.7 (15) | 172.0 (20) | 1207.5 (1) | 217.8 (17) | 1161.6 (7) | 217.8 (17) |
| H20 | 1120.2 (5) | 672.0 (20) | 148.9 (21) | 1155.9 (6) | 184.6 (21) | 1120.2 (12) | 184.6 (21) |

### Return per episode, mean over seeds 3042-3061, forward weight 2.0 (rank in brackets, 1 = highest of the 21)

| behaviour | main | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---|---|---|---|---|---|---|
| statue | 1092.7 (19) | 1092.7 (16) | 1092.7 (11) | 1091.4 (19) | 1091.4 (15) | 1092.7 (19) | 1091.4 (15) |
| parent | 1093.9 (18) | 1093.9 (15) | 1093.9 (10) | 1092.4 (18) | 1092.4 (14) | 1093.9 (18) | 1092.4 (14) |
| parent_ls15 | 422.0 (21) | 398.4 (21) | 388.5 (20) | 396.7 (21) | 363.2 (21) | 428.8 (21) | 370.1 (21) |
| W02 | 1043.8 (20) | 1043.8 (17) | 1043.8 (13) | 1054.7 (20) | 1054.7 (16) | 1051.8 (20) | 1062.7 (16) |
| FINAL3 | 1234.9 (16) | 1234.9 (13) | 1234.9 (9) | 1334.7 (16) | 1334.7 (12) | 1325.0 (15) | 1424.9 (11) |
| V01 | 1293.9 (14) | 1293.9 (12) | 1293.9 (7) | 1439.4 (14) | 1439.4 (10) | 1383.9 (14) | 1529.4 (8) |
| V05 | 1423.7 (12) | 1423.5 (8) | 1423.5 (5) | 1653.7 (12) | 1653.5 (6) | 1515.7 (11) | 1745.5 (6) |
| V24 | 1481.3 (10) | 1481.1 (7) | 1481.1 (4) | 1779.3 (11) | 1779.0 (4) | 1569.6 (9) | 1867.3 (4) |
| V18 | 1650.1 (8) | 1641.7 (4) | 1640.7 (3) | 2051.8 (8) | 2042.4 (3) | 1733.7 (8) | 2126.0 (3) |
| V35 | 1751.6 (6) | 1736.9 (3) | 1734.4 (1) | 2216.7 (6) | 2199.5 (2) | 1838.4 (6) | 2286.2 (1) |
| V29 | 1785.1 (5) | 1738.4 (2) | 1723.6 (2) | 2262.1 (5) | 2200.7 (1) | 1866.0 (4) | 2281.6 (2) |
| V33 | 1438.5 (11) | 1388.3 (10) | 1359.1 (6) | 1811.5 (10) | 1732.1 (5) | 1504.3 (12) | 1797.9 (5) |
| HF020 | 1213.7 (17) | 826.2 (20) | 340.3 (21) | 1321.0 (17) | 447.7 (20) | 1213.7 (17) | 447.7 (20) |
| HF030 | 1278.0 (15) | 887.0 (19) | 398.1 (19) | 1426.4 (15) | 546.5 (19) | 1278.0 (16) | 546.5 (19) |
| HF050 | 1412.5 (13) | 1006.6 (18) | 505.8 (18) | 1645.8 (13) | 739.1 (18) | 1412.5 (13) | 739.1 (18) |
| H23 | 1541.4 (9) | 1125.3 (14) | 615.6 (17) | 1857.8 (9) | 932.0 (17) | 1541.4 (10) | 932.0 (17) |
| H24 | 1735.9 (7) | 1301.4 (11) | 784.6 (16) | 2179.1 (7) | 1227.8 (13) | 1735.9 (7) | 1227.8 (13) |
| HF120 | 1862.6 (4) | 1412.3 (9) | 898.5 (15) | 2355.6 (4) | 1391.5 (11) | 1862.6 (5) | 1391.5 (12) |
| H25 | 1982.8 (3) | 1523.6 (6) | 1002.1 (14) | 2472.5 (3) | 1491.8 (9) | 1982.8 (3) | 1491.8 (10) |
| HF150 | 2051.2 (2) | 1587.3 (5) | 1061.6 (12) | 2509.5 (2) | 1519.9 (8) | 2051.2 (2) | 1519.9 (9) |
| H20 | 2227.9 (1) | 1779.8 (1) | 1256.6 (8) | 2585.0 (1) | 1613.7 (7) | 2227.9 (1) | 1613.7 (7) |

## Margins: each walk against the best standing behaviour and against its speed-matched hop (seeds 3042-3061)

Mean of the per-seed paired difference [min, max]; (n) = seeds of 20 on which the walk is ahead. Standing: on each seed the best of the three.


Forward weight 0.2:

| setting | FINAL3 vs stand | FINAL3 vs HF020 | V01 vs stand | V01 vs HF030 | V05 vs stand | V05 vs HF050 | V24 vs stand | V24 vs H23 | V18 vs stand | V18 vs H24 | V35 vs stand | V35 vs HF120 | V29 vs stand | V29 vs HF150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | +4.2 [-2, +12] (16) | +31.3 [+25, +40] (20) | +1.5 [-11, +10] (13) | +19.7 [+9, +30] (20) | +17.3 [+2, +25] (20) | +15.7 [+8, +21] (20) | -16.8 [-25, -8] (0) | -35.1 [-43, -24] (0) | +11.9 [+2, +18] (20) | -27.9 [-39, -18] (0) | +19.2 [+2, +30] (20) | -32.9 [-55, -19] (0) | -33.6 [-467, +53] (15) | -99.3 [-509, +7] (1) |
| R1 | +4.2 [-2, +12] (16) | +418.9 [+412, +427] (20) | +1.5 [-11, +10] (13) | +410.7 [+398, +419] (20) | +17.2 [+2, +24] (20) | +421.4 [+412, +428] (20) | -17.1 [-25, -10] (0) | +380.8 [+374, +393] (20) | +3.5 [-10, +11] (17) | +398.2 [+385, +415] (20) | +4.5 [-10, +14] (17) | +402.7 [+391, +414] (20) | -80.2 [-498, +6] (5) | +318.0 [-85, +409] (19) |
| R2 | +4.2 [-2, +12] (16) | +904.7 [+898, +917] (20) | +1.5 [-11, +10] (13) | +899.6 [+882, +909] (20) | +17.2 [+2, +24] (20) | +922.3 [+912, +930] (20) | -17.1 [-25, -10] (0) | +890.4 [+879, +903] (20) | +2.4 [-11, +11] (14) | +913.9 [+894, +924] (20) | +2.0 [-12, +12] (13) | +914.0 [+898, +928] (20) | -95.1 [-509, -5] (0) | +828.9 [+439, +930] (20) |
| R3 | +14.3 [+8, +22] (20) | +30.6 [+24, +40] (20) | +16.2 [+3, +26] (20) | +19.4 [+9, +31] (20) | +40.5 [+23, +49] (20) | +15.3 [+6, +21] (20) | +13.1 [+2, +22] (20) | -37.0 [-46, -25] (0) | +52.2 [+41, +59] (20) | -32.0 [-44, -22] (0) | +65.9 [+44, +78] (20) | -35.7 [-60, -23] (0) | +14.3 [-436, +103] (16) | -97.4 [-526, +9] (1) |
| R4 | +14.3 [+8, +22] (20) | +903.9 [+897, +917] (20) | +16.2 [+3, +26] (20) | +899.3 [+882, +909] (20) | +40.3 [+22, +48] (20) | +921.9 [+911, +930] (20) | +12.8 [+2, +21] (20) | +888.6 [+877, +902] (20) | +42.7 [+28, +52] (20) | +909.8 [+889, +921] (20) | +48.7 [+30, +60] (20) | +911.2 [+894, +926] (20) | -47.2 [-478, +45] (12) | +830.7 [+421, +931] (20) |
| R5 | +94.3 [+88, +103] (20) | +121.5 [+115, +131] (20) | +91.5 [+79, +101] (20) | +109.7 [+100, +120] (20) | +109.3 [+94, +118] (20) | +107.7 [+99, +114] (20) | +71.4 [+63, +79] (20) | +53.2 [+46, +63] (20) | +95.5 [+85, +106] (20) | +55.7 [+45, +66] (20) | +106.0 [+84, +120] (20) | +53.8 [+29, +72] (20) | +47.4 [-404, +140] (17) | -18.3 [-446, +92] (16) |
| R6 | +104.4 [+98, +113] (20) | +994.1 [+987, +1007] (20) | +106.2 [+93, +116] (20) | +989.3 [+972, +999] (20) | +132.3 [+113, +141] (20) | +1013.9 [+1002, +1022] (20) | +101.1 [+91, +108] (20) | +976.9 [+966, +990] (20) | +126.3 [+111, +140] (20) | +993.4 [+978, +1010] (20) | +135.4 [+111, +148] (20) | +997.9 [+978, +1012] (20) | +33.8 [-415, +132] (16) | +911.7 [+484, +1016] (20) |
| R13 | +14.3 [+8, +22] (20) | +418.1 [+411, +427] (20) | +16.2 [+3, +26] (20) | +410.4 [+397, +419] (20) | +40.3 [+22, +48] (20) | +421.1 [+411, +428] (20) | +12.9 [+2, +21] (20) | +378.9 [+371, +392] (20) | +43.8 [+29, +52] (20) | +394.1 [+380, +413] (20) | +51.2 [+32, +61] (20) | +399.9 [+385, +413] (20) | -32.4 [-467, +56] (16) | +319.8 [-102, +410] (19) |
| R15 | +94.3 [+88, +103] (20) | +509.0 [+502, +518] (20) | +91.5 [+79, +101] (20) | +500.7 [+488, +509] (20) | +109.2 [+93, +117] (20) | +513.4 [+503, +520] (20) | +71.2 [+63, +78] (20) | +469.1 [+461, +480] (20) | +87.1 [+73, +100] (20) | +481.8 [+468, +504] (20) | +91.3 [+71, +104] (20) | +489.4 [+472, +504] (20) | +0.7 [-435, +92] (16) | +398.9 [-22, +493] (19) |

Forward weight 2.0:

| setting | FINAL3 vs stand | FINAL3 vs HF020 | V01 vs stand | V01 vs HF030 | V05 vs stand | V05 vs HF050 | V24 vs stand | V24 vs H23 | V18 vs stand | V18 vs H24 | V35 vs stand | V35 vs HF120 | V29 vs stand | V29 vs HF150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | +140.9 [+126, +150] (20) | +21.2 [+11, +41] (20) | +199.9 [+170, +218] (20) | +15.9 [-0, +43] (19) | +329.8 [+280, +345] (20) | +11.2 [-11, +25] (17) | +387.4 [+344, +405] (20) | -60.1 [-81, -33] (0) | +556.2 [+501, +574] (20) | -85.7 [-112, -64] (0) | +657.7 [+573, +686] (20) | -111.0 [-173, -76] (0) | +691.1 [+27, +873] (20) | -266.1 [-897, -36] (0) |
| R1 | +140.9 [+126, +150] (20) | +408.7 [+398, +425] (20) | +199.9 [+170, +218] (20) | +406.9 [+388, +431] (20) | +329.6 [+279, +344] (20) | +416.9 [+394, +431] (20) | +387.2 [+344, +405] (20) | +355.8 [+336, +384] (20) | +547.8 [+492, +569] (20) | +340.3 [+307, +381] (20) | +643.0 [+561, +670] (20) | +324.7 [+273, +356] (20) | +644.5 [-3, +825] (19) | +151.2 [-473, +366] (16) |
| R2 | +140.9 [+126, +150] (20) | +894.5 [+884, +913] (20) | +199.9 [+170, +218] (20) | +895.7 [+872, +918] (20) | +329.6 [+279, +344] (20) | +917.8 [+894, +932] (20) | +387.1 [+344, +405] (20) | +865.5 [+846, +891] (20) | +546.7 [+490, +568] (20) | +856.1 [+830, +890] (20) | +640.5 [+558, +668] (20) | +836.0 [+781, +859] (20) | +629.7 [-15, +813] (19) | +662.1 [+50, +887] (20) |
| R3 | +242.2 [+220, +255] (20) | +13.7 [-1, +41] (19) | +346.9 [+301, +372] (20) | +13.0 [-9, +54] (15) | +561.2 [+486, +581] (20) | +7.8 [-24, +29] (17) | +686.8 [+617, +713] (20) | -78.5 [-109, -40] (0) | +959.4 [+870, +988] (20) | -127.2 [-170, -87] (0) | +1124.2 [+994, +1163] (20) | -138.9 [-224, -95] (0) | +1169.7 [+334, +1375] (20) | -247.4 [-1071, -25] (0) |
| R4 | +242.2 [+220, +255] (20) | +887.0 [+873, +909] (20) | +346.9 [+301, +372] (20) | +892.9 [+865, +928] (20) | +561.0 [+485, +581] (20) | +914.4 [+880, +936] (20) | +686.6 [+617, +713] (20) | +847.0 [+818, +885] (20) | +949.9 [+859, +982] (20) | +814.6 [+772, +866] (20) | +1107.0 [+979, +1146] (20) | +808.0 [+730, +849] (20) | +1108.2 [+292, +1315] (20) | +680.8 [-123, +898] (19) |
| R5 | +231.1 [+215, +240] (20) | +111.3 [+101, +131] (20) | +289.9 [+260, +308] (20) | +105.9 [+90, +134] (20) | +421.8 [+371, +437] (20) | +103.2 [+80, +118] (20) | +475.7 [+432, +494] (20) | +28.2 [+7, +54] (20) | +639.8 [+585, +658] (20) | -2.1 [-29, +25] (8) | +744.4 [+654, +776] (20) | -24.2 [-89, +14] (2) | +772.1 [+90, +958] (20) | -185.1 [-834, +49] (1) |
| R6 | +332.4 [+310, +344] (20) | +977.2 [+963, +1000] (20) | +436.9 [+391, +462] (20) | +982.9 [+955, +1019] (20) | +653.0 [+576, +673] (20) | +1006.4 [+971, +1029] (20) | +774.9 [+706, +803] (20) | +935.3 [+906, +971] (20) | +1033.5 [+942, +1066] (20) | +898.2 [+855, +954] (20) | +1193.7 [+1061, +1234] (20) | +894.7 [+814, +940] (20) | +1189.2 [+355, +1399] (20) | +761.8 [-61, +983] (19) |
| R13 | +242.2 [+220, +255] (20) | +401.2 [+387, +426] (20) | +346.9 [+301, +372] (20) | +404.0 [+381, +442] (20) | +561.0 [+485, +581] (20) | +413.6 [+380, +435] (20) | +686.6 [+617, +713] (20) | +337.4 [+308, +377] (20) | +951.0 [+860, +983] (20) | +298.8 [+249, +357] (20) | +1109.5 [+982, +1148] (20) | +296.7 [+222, +347] (20) | +1123.0 [+303, +1327] (20) | +169.9 [-647, +383] (16) |
| R15 | +231.1 [+215, +240] (20) | +498.9 [+488, +516] (20) | +289.9 [+260, +308] (20) | +496.9 [+478, +522] (20) | +421.6 [+370, +437] (20) | +508.9 [+485, +523] (20) | +475.4 [+432, +494] (20) | +444.1 [+425, +470] (20) | +631.4 [+575, +653] (20) | +423.9 [+390, +470] (20) | +729.7 [+642, +760] (20) | +411.4 [+358, +447] (20) | +725.5 [+59, +909] (20) | +232.1 [-410, +450] (17) |

### What each change does to each behaviour: mean (setting - main) per episode


Forward weight 0.2:

| behaviour | R1 | R2 | R3 | R4 | R5 | R6 | R13 | R15 | R3s |
|---|---|---|---|---|---|---|---|---|---|
| statue | +0.0 | +0.0 | -0.1 | -0.1 | +0.0 | -0.1 | -0.1 | +0.0 | +0.0 |
| parent | +0.0 | +0.0 | -0.1 | -0.1 | +0.0 | -0.1 | -0.1 | +0.0 | +0.0 |
| parent_ls15 | -23.5 | -33.5 | -2.5 | -36.0 | +6.8 | -29.2 | -26.1 | -16.7 | +0.0 |
| W02 | +0.0 | +0.0 | +1.1 | +1.1 | +8.1 | +9.1 | +1.1 | +8.1 | +0.0 |
| FINAL3 | +0.0 | +0.0 | +10.0 | +10.0 | +90.1 | +100.1 | +10.0 | +90.1 | +0.0 |
| V01 | +0.0 | +0.0 | +14.6 | +14.6 | +90.0 | +104.6 | +14.6 | +90.0 | -0.0 |
| V05 | -0.2 | -0.2 | +23.0 | +22.8 | +92.0 | +114.8 | +22.8 | +91.8 | +0.0 |
| V24 | -0.2 | -0.3 | +29.8 | +29.5 | +88.3 | +117.8 | +29.6 | +88.1 | +0.0 |
| V18 | -8.4 | -9.5 | +40.2 | +30.7 | +83.6 | +114.3 | +31.8 | +75.2 | +0.0 |
| V35 | -14.7 | -17.2 | +46.5 | +29.3 | +86.7 | +116.0 | +31.8 | +72.0 | -0.4 |
| V29 | -46.6 | -61.5 | +47.7 | -13.8 | +81.0 | +67.2 | +1.1 | +34.3 | -3.5 |
| V33 | -50.2 | -79.4 | +37.3 | -42.1 | +65.8 | +23.7 | -12.9 | +15.6 | -6.1 |
| HF020 | -387.5 | -873.3 | +10.7 | -862.6 | +0.0 | -862.6 | -376.8 | -387.5 | -0.0 |
| HF030 | -391.0 | -879.8 | +14.8 | -865.0 | +0.0 | -865.0 | -376.2 | -391.0 | +0.0 |
| HF050 | -405.9 | -906.8 | +23.3 | -883.4 | +0.0 | -883.4 | -382.6 | -405.9 | +0.0 |
| H23 | -416.1 | -925.8 | +31.6 | -894.1 | +0.0 | -894.1 | -384.5 | -416.1 | +0.0 |
| H24 | -434.5 | -951.3 | +44.3 | -907.0 | +0.0 | -907.0 | -390.2 | -434.5 | -0.1 |
| HF120 | -450.3 | -964.1 | +49.3 | -914.8 | +0.0 | -914.8 | -401.0 | -450.3 | -2.2 |
| H25 | -459.2 | -980.7 | +49.0 | -931.7 | +0.0 | -931.7 | -410.3 | -459.2 | -7.5 |
| HF150 | -463.9 | -989.6 | +45.8 | -943.8 | +0.0 | -943.8 | -418.1 | -463.9 | -12.0 |
| H20 | -448.1 | -971.3 | +35.7 | -935.6 | +0.0 | -935.6 | -412.4 | -448.1 | -27.8 |

Forward weight 2.0:

| behaviour | R1 | R2 | R3 | R4 | R5 | R6 | R13 | R15 | R3s |
|---|---|---|---|---|---|---|---|---|---|
| statue | +0.0 | +0.0 | -1.3 | -1.3 | +0.0 | -1.3 | -1.3 | +0.0 | +0.0 |
| parent | +0.0 | +0.0 | -1.5 | -1.5 | +0.0 | -1.5 | -1.5 | +0.0 | +0.0 |
| parent_ls15 | -23.5 | -33.5 | -25.3 | -58.7 | +6.8 | -51.9 | -48.8 | -16.7 | -0.0 |
| W02 | +0.0 | +0.0 | +10.9 | +10.9 | +8.1 | +18.9 | +10.9 | +8.1 | +0.0 |
| FINAL3 | +0.0 | +0.0 | +99.8 | +99.8 | +90.1 | +190.0 | +99.8 | +90.1 | +0.0 |
| V01 | +0.0 | +0.0 | +145.5 | +145.5 | +90.0 | +235.5 | +145.5 | +90.0 | +0.0 |
| V05 | -0.2 | -0.2 | +229.9 | +229.8 | +92.0 | +321.8 | +229.8 | +91.8 | -0.0 |
| V24 | -0.2 | -0.3 | +298.0 | +297.7 | +88.3 | +386.0 | +297.7 | +88.1 | +0.0 |
| V18 | -8.4 | -9.5 | +401.7 | +392.2 | +83.6 | +475.8 | +393.3 | +75.2 | -0.0 |
| V35 | -14.7 | -17.2 | +465.0 | +447.8 | +86.7 | +534.6 | +450.3 | +72.0 | -3.9 |
| V29 | -46.6 | -61.5 | +477.1 | +415.6 | +81.0 | +496.6 | +430.4 | +34.3 | -35.0 |
| V33 | -50.2 | -79.4 | +373.0 | +293.6 | +65.8 | +359.4 | +322.8 | +15.6 | -61.5 |
| HF020 | -387.5 | -873.3 | +107.3 | -766.0 | +0.0 | -766.0 | -280.2 | -387.5 | +0.0 |
| HF030 | -391.0 | -879.8 | +148.4 | -731.5 | +0.0 | -731.5 | -242.6 | -391.0 | -0.0 |
| HF050 | -405.9 | -906.8 | +233.3 | -673.5 | +0.0 | -673.5 | -172.6 | -405.9 | -0.0 |
| H23 | -416.1 | -925.8 | +316.4 | -609.4 | +0.0 | -609.4 | -99.7 | -416.1 | -0.0 |
| H24 | -434.5 | -951.3 | +443.2 | -508.1 | +0.0 | -508.1 | +8.7 | -434.5 | -0.8 |
| HF120 | -450.3 | -964.1 | +493.0 | -471.1 | +0.0 | -471.1 | +42.7 | -450.3 | -21.8 |
| H25 | -459.2 | -980.7 | +489.7 | -491.0 | +0.0 | -491.0 | +30.5 | -459.2 | -74.6 |
| HF150 | -463.9 | -989.6 | +458.3 | -531.3 | +0.0 | -531.3 | -5.6 | -463.9 | -120.4 |
| H20 | -448.1 | -971.3 | +357.1 | -614.2 | +0.0 | -614.2 | -91.1 | -448.1 | -278.1 |

## Gates (offline)

Seeds 3042-3061:

| behaviour | G0 per episode | G1 per episode | G2 per episode | G1 full-length only | G2 full-length only | panel mean forward_vel | panel mean last-5 s | mean length | walk class (of 20) | panel G0 | panel G1 | panel G2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| statue | 0 | 0 | 0 | 0 | 0 | -0.003 | -0.000 | 1000 | 0 | fail | fail | fail |
| parent | 0 | 0 | 0 | 0 | 0 | -0.003 | 0.000 | 1000 | 0 | fail | fail | fail |
| parent_ls15 | 0 | 0 | 0 | 0 | 0 | -0.075 | -0.073 | 569 | 0 | fail | fail | fail |
| W02 | 0 | 0 | 0 | 0 | 0 | 0.020 | 0.024 | 1000 | 0 | fail | fail | fail |
| FINAL3 | 0 | 0 | 0 | 0 | 0 | 0.187 | 0.211 | 1000 | 20 | fail | fail | fail |
| V01 | 0 | 0 | 0 | 0 | 0 | 0.273 | 0.303 | 1000 | 20 | fail | fail | fail |
| V05 | 0 | 0 | 0 | 0 | 0 | 0.431 | 0.510 | 1000 | 20 | fail | fail | fail |
| V24 | 0 | 0 | 2 | 0 | 2 | 0.559 | 0.686 | 1000 | 20 | fail | fail | fail |
| V18 | 0 | 2 | 20 | 2 | 20 | 0.753 | 0.979 | 1000 | 20 | fail | fail | pass |
| V35 | 0 | 20 | 20 | 20 | 20 | 0.884 | 1.171 | 1000 | 20 | fail | pass | pass |
| V29 | 13 | 18 | 19 | 16 | 17 | 1.031 | 1.341 | 972 | 20 | pass | pass | pass |
| V33 | 12 | 12 | 12 | 3 | 3 | 1.070 | 1.441 | 837 | 19 | pass | fail | fail |
| HF020 | 0 | 0 | 0 | 0 | 0 | 0.201 | 0.197 | 1000 | 0 | fail | fail | fail |
| HF030 | 0 | 0 | 0 | 0 | 0 | 0.278 | 0.292 | 1000 | 0 | fail | fail | fail |
| HF050 | 0 | 0 | 0 | 0 | 0 | 0.437 | 0.497 | 1000 | 0 | fail | fail | fail |
| H23 | 0 | 0 | 0 | 0 | 0 | 0.593 | 0.695 | 1000 | 0 | fail | fail | fail |
| H24 | 0 | 0 | 0 | 0 | 0 | 0.834 | 0.991 | 1000 | 0 | fail | fail | fail |
| HF120 | 8 | 0 | 0 | 0 | 0 | 0.992 | 1.202 | 1000 | 0 | fail | fail | fail |
| H25 | 20 | 0 | 0 | 0 | 0 | 1.151 | 1.412 | 1000 | 0 | pass | fail | fail |
| HF150 | 20 | 0 | 0 | 0 | 0 | 1.236 | 1.499 | 1000 | 0 | pass | fail | fail |
| H20 | 19 | 0 | 0 | 0 | 0 | 1.586 | 1.994 | 984 | 0 | pass | fail | fail |

Held-out seeds 3062-3081:

| behaviour | G0 per episode | G1 per episode | G2 per episode | G1 full-length only | G2 full-length only | panel mean forward_vel | panel mean last-5 s | mean length | walk class (of 20) | panel G0 | panel G1 | panel G2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| statue | 0 | 0 | 0 | 0 | 0 | -0.003 | -0.000 | 1000 | 0 | fail | fail | fail |
| parent | 0 | 0 | 0 | 0 | 0 | -0.003 | 0.000 | 1000 | 0 | fail | fail | fail |
| W02 | 0 | 0 | 0 | 0 | 0 | 0.019 | 0.023 | 1000 | 0 | fail | fail | fail |
| FINAL3 | 0 | 0 | 0 | 0 | 0 | 0.186 | 0.210 | 1000 | 20 | fail | fail | fail |
| V24 | 0 | 0 | 8 | 0 | 8 | 0.562 | 0.691 | 1000 | 20 | fail | fail | fail |
| V18 | 0 | 3 | 20 | 3 | 20 | 0.758 | 0.985 | 1000 | 20 | fail | fail | pass |
| V35 | 0 | 20 | 20 | 20 | 20 | 0.889 | 1.180 | 1000 | 20 | fail | pass | pass |
| V33 | 13 | 14 | 14 | 3 | 3 | 1.103 | 1.505 | 827 | 20 | pass | pass | pass |
| H23 | 0 | 0 | 0 | 0 | 0 | 0.594 | 0.696 | 1000 | 0 | fail | fail | fail |
| H24 | 0 | 0 | 0 | 0 | 0 | 0.834 | 0.999 | 1000 | 0 | fail | fail | fail |
| HF120 | 8 | 0 | 0 | 0 | 0 | 0.996 | 1.199 | 1000 | 0 | fail | fail | fail |
| H20 | 20 | 0 | 0 | 0 | 0 | 1.598 | 2.032 | 980 | 0 | pass | fail | fail |


## Other readouts

- **Steady state (t = 5-10 s), per-step rate × 1000** [measured: out/md_main_stdout.txt and out/md_HO_stdout.txt, "Steady-window reward rate"]. This removes the start-up, and the picture is the same:
  - Under main at w 0.2: V24 1089.0 against the statue's 1095.6, and H23 1123.7.
  - Under R13 at w 0.2: V24 1125.4, which is +30 over the statue in steady walking. Its thin episode margin comes from the 0.5 s settle and the start-up.
  - Under R1, the 1.2 m/s walk's landing-bounce cost concentrates here: V35 1143.9 → 1120.3 at w 0.2.
- **Discounted return from reset** (γ = 0.995, PPO's for this stage) [measured: out/tables_main_stdout.txt, section J]. At w 0.2 it is dominated by the first two seconds, when the walkers are still settling and starting:
  - main: statue 217.0, V24 212.3, V35 214.4, H23 216.9, HF150 221.3.
  - R4: V24 215.4, still below the statue's 216.8.
  - R6: stable walks 227.3-234.4, statue 216.8, hops 92.0-97.0.
- **Pilot 3's actual start** (the parent with log_std −1.5, stochastic; extra). It reproduces the REWARD stage's draw exactly and falls or touches its head on 18 of 20 seeds. [measured: out/crossref_stdout.txt, out/md_main_stdout.txt]
  - Each setting moves it by −23.5 (R1), −33.5 (R2), +6.8 (R5) and −29.2 (R6) at w 0.2, against its main return of 456.
  - Its gap to standing (about −640) is unchanged in kind under every setting. Under R2/R4/R6 the hops at w 0.2 (149-233) even fall below it (420-427). Even so, a hop still earns about 0.06-0.11 per step under R2 at w 0.2 [measured: steady-rate table], so surviving while hopping still beats falling. [inferred]
- **The stance reward's floor-impact term** (2.0 × max(0, step peak / BW − 1.4); extra, not one of R1-R6) [measured: out/impact_stdout.txt]:
  - It would charge the walks 0.1 (FINAL3), 1.5 (V01), 73 (V05), 59 (V24), 178 (V18) and 212 (V35) per episode.
  - It would charge the hops 967-1211 and the noisy parent 171.
  - The 1.0-1.2 m/s walks land at 2.1 BW at p95.
- **Consistency checks** [measured: out/checks_stdout.txt]:
  - R5's registered touchdowns (from t ≥ 1 s) against the probe's own floor touchdown counts over the same window:
    - identical on every episode of FINAL3, V01, V05, H23, HF120 and H20;
    - 0-2 fewer for W02 and H24;
    - fewer for the faster walks: V24 0-5 (mean 1.3), V18 1-5 (2.9), V35 3-6 (3.9), V29 0-8 (2.3).

    The missing ones are landings preceded by a one-step contact blip, which the probe's debounce merges and the causal rule does not. R5p (the probe's own extraction) is reported alongside.
  - R1's unsupported steps that contain a fully airborne substep, of all unsupported steps:
    - V18 5.1 of 10.8; V35 12.2 of 19.1; V29 44.0 of 60.0;
    - the hops 528.5 of 528.7 (H23) to 562.5 of 567.8 (H20).
- **Sensitivity of the free parameters** [measured: out/sens_stdout.txt]:
  - R1's threshold barely matters, because a foot is either loaded or not. From 0.1 N (any touch) to 25% BW, the unsupported steps per episode go 10.8-13.3 for V18, 19.1-21.2 for V35 and 505-585 for the hops.
  - R5's S length matters only for stepping in place: W02 earns 159 at 0.05 L, 80 at 0.1 L and 40 at 0.2 L, against 525-931 for the stable walks across the three (the 0.2 m/s FINAL3, whose feet land 9.6 cm apart, drops to 525 at 0.2 L).
  - The 1.2 s cap, cut to 0.8 s, takes 15-17% from the slowest walks (FINAL3, V01). A merge window of 1 or 3 steps changes any walk by 1% or less.
  - The one-sided simultaneity rule (R5a) pays the symmetric F hops 0.5-4.1 per episode at unit weight. A staggered hop would be paid more. [inferred: from the rule]

## Limits

- **These are episode returns of fixed behaviours, not training.** The screen ranks hand-built and checkpointed behaviours under each setting. It says which of these behaviours a setting prefers. It does not say what PPO would find, how fast, or whether a learned policy would find a gait none of these shows (a run, a shuffle, a staggered hop, a stomp). [inferred]
- **One controller family per gait.** A learned walk could crouch, sway or bounce more or less than these walkers, which changes what R1/R2 cost it. These hops land at 4.5-5.2 Hz with about half of all substeps airborne. A shorter-flight hop like September's (7.7-9.2 Hz, 35-43% airborne) would lose less under R1/R2 [inferred: in proportion to its unsupported share], and R5 would still pay it nothing.
- **Speed matching is by command.** At 0.7-1.2 m/s the hop is 0.009-0.031 m/s faster than its walk. At w 2.0 that is worth about 7-25 per episode of the hop's 60-111 lead under main. [inferred: 800 per episode per m/s]
- **The standing margin at the ramp start is small for the crouched walk** under R4 and R13 (+1.9 and −2.5 on the worst seeds), and it depends on how a walker starts up. A learned walk with a quicker start could clear it, and one with a deeper crouch could not. [inferred]
- **R5 is my simplest reading of GQ-15 (a)**, not the plan's full kit: there is no phase match P and no swing factor A. Its parameters (0.1 L, 1.2 s cap, 2-step merge, w_td 0.1) were fixed before any number was computed and not tuned. The break-even scan shows how much weight it needs.
- **Episodes and seeds.** Reset seeds of the locomotion config (reset noise 0.05); no pushes, no command changes, 10 s episodes. The held-out panel covers 12 of the 21 behaviours.
- **The offline recomputation** assumes the hypothetical settings would not change what the env does, which holds because the reward does not enter the dynamics. It reproduces main and the env's weight-0.2 runs exactly.

## Slips

1. **The pre-declared R5 had a flaw.** I found it before any number was computed and kept the original as R5a.
   - The flaw: its simultaneity test looked only at the other foot's previous touchdown. In a two-footed hop, the first-processed foot of each simultaneous landing passes it, and is paid whenever it lands ahead of the other foot (a staggered hop).
   - The fix: the probe's two-sided merge rule (R5), with the pay delayed by 2 steps so it stays causal. Both versions are reported; on these symmetric hops they differ by at most 0.4 per episode at w_td 0.1.
   - The edit is logged in a NOTE at 05:16:38Z.
2. **CMDLOG literalness.**
   - The first mkdir line (05:01:34Z) shows expanded paths, not the literal text; this is noted at 05:09:06Z.
   - Read-only commands from before the log existed (05:01) and the reads of 05:02-05:09 are logged after the fact as NOTE lines. One NOTE first gave their end time wrongly; the next line corrects it.
   - About six read-only progress checks or greps ran before their log line was written. Each is listed in a later NOTE. The clock times I typed inside those NOTEs ran ahead of the real clock; a NOTE (05:26:02Z) says the line timestamps are authoritative.
   - One logged heredoc line did not quote the read-only commands that followed it in the same call; a later NOTE adds them.
   - File-write and edit-tool changes to my own scripts and drafts are NOTE lines, not literal commands. sed patches are logged literally.
3. **Bugs of my own, fixed:**
   - The first analysis draft compared a dict with an array. It was fixed before its first run.
   - `tables.py` first read its input under the wrong tag, because `sim.sh` passes its TAG as the script's first argument. The crashed run's output was overwritten by the fixed run.
4. **Analysis alongside simulation.** Several offline analysis runs (screen_test, screen_test2, screen_pre, tables_pre, sens, screen_main and the w_td scans) ran under sim.sh while a simulation batch was running. They only read files. No two simulations ever ran at once.
5. **Background waits.** The batch shells ran under nice -n 10. The shell tool's background wait loops (`until grep ...; sleep 5`) did not; they only read the console files.
6. **No hand-made venv.** `optB/venv` already holds torch 2.13, stable_baselines3 2.9, gymnasium, mujoco 3.10 and numpy, which is everything the stance parent needs. So I made no .pth venv, as in the REWARD and HOP stages.
7. **R5's touchdown detector undercounts the faster walks.** I found this with checks.py after the main tables were written. R5 is kept as designed, and R5p (the probe's own extraction) is reported next to it; no verdict changes.
8. **One abbreviated CMDLOG line.** The re-assembly of tmp/part_tables.md at about 05:47Z is logged as "same block as at the previous assembly" plus the two changes, not quoted in full. The full previous block is logged literally just above it.
9. **Outputs overwritten by deterministic re-runs:**
   - `out/sens_stdout.txt` was first written at about 05:28Z, before parent_ls15 and HF020 existed, and was overwritten by a re-run that includes them.
   - `out/screen_main.json`, `out/screen_HO.json` and both verdict files were overwritten when the extras were added. Regenerating the markdown tables from the new JSONs gives byte-identical files (`cmp` of md_main/md_main_check and md_HO/md_HO_check), so every number quoted for the original settings is unchanged.
10. **Folder size:** 212 MB (mostly the per-episode npz files), under the 300 MB limit. Torch created `tmp/torchinductor_root` inside this folder's TMPDIR when the parent was loaded.
11. **One non-sim.sh python call.** `out/an_R1x_R5p_costs.txt` came from an inline read-only `optB/venv/bin/python -I -B -c` JSON summary, run under nice and OMP_NUM_THREADS=1 but without TMPDIR. It is logged literally.

## Files

- `CMDLOG.txt`: literal commands with UTC times, plus NOTE lines.
- Running episodes:
  - `scripts/runs.py`: the episode runner (frozen probe rollout, a pass-through recorder that asserts every action is in [-1, 1], a per-step npz).
  - `scripts/sim.sh`: the wrapper. It logs the literal command and applies nice, OMP_NUM_THREADS=1 and TMPDIR.
  - `scripts/batch1.sh`, `scripts/batch2.sh`: the sequential queues.
- Controllers, copied unchanged from the HOP stage (sha256 in `out/copied_sha256.txt`): `scripts/wlib.py`, `scripts/wlib2.py`, `scripts/hoplib.py`, `scripts/defaults.py`.
- Offline analysis (no simulation):
  - `scripts/screen.py`: reproduction and all the settings.
  - `scripts/tables.py`, `scripts/mdtables.py`, `scripts/verdict.py`: the tables.
  - `scripts/crossref.py`: the check against the REWARD and HOP stages' JSONs.
  - `scripts/sens.py`: parameter sensitivity.
  - `scripts/impact.py`: the floor-impact extra.
  - `scripts/checks.py`: the touchdown-count and unsupported-step checks (`out/checks_stdout.txt`), plus `out/an_R1x_R5p_costs.txt` (an inline read-only JSON summary).
- Per episode: `out/<B>_s<seed>.npz` and `out/<B>.json`.
  - The npz holds reward, fall penalty, all `reward_*` components, 14 info scalars, forward_vel, foot and root positions, the pelvis quaternion, floor-truth contact fractions and forces, touch forces and actions.
  - The json holds lengths, ends, returns, probe classes and gait metrics.
  - Held-out panel: `out/HO_<B>*`. Weight-0.2 check runs: `out/w02_<B>*`.
- Results:
  - Main panel: `out/screen_main.json` and `_stdout.txt`, `out/tables_main_stdout.txt`, `out/md_main_stdout.txt` (every table in full), `out/verdict_main_stdout.txt`.
  - Held-out panel: `out/screen_HO*`, `out/md_HO_stdout.txt`, `out/verdict_HO*_stdout.txt`.
  - Event-weight scan: `out/screen_wtd*`, `out/verdict_wtd*_stdout.txt`.
  - `out/crossref_stdout.txt`, `out/sens_stdout.txt`, `out/impact_stdout.txt`, `out/SHA256SUMS_inputs.txt`.
- Drafts of sections in `tmp/`.
