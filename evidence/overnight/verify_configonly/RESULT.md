# VERIFY CONFIGONLY: independent check of the config-only reward screen

Folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/verify_configonly/`. Paths below are relative to it.

- **When:** 07:09-07:45 UTC on 2026-11-04 (deadline 10:45 UTC).
- **What was checked:** the CONFIGONLY stage (`overnight/configonly/RESULT.md`, its `out/` and `CMDLOG.txt`) and its four headlines.
- **Inputs (read-only):**
  - SCREEN's per-step records `overnight/screen/out/[HO_]<B>_s<seed>.npz` and `<B>.json`, and its `screen_main.json` and `screen_HO.json` for R4, R6, R13 and R15.
  - SCREEN's controllers (`screen/scripts/hoplib.py`, `wlib.py`, `wlib2.py`, `defaults.py`), imported read-only with `python -I -B`.
  - The export `feas/main_4d5f4e7` (code, `locomotion.toml`, docs, the frozen gait probe).
  - The stance parent checkpoint under `fetch/stance7_r10LTSBpc_s42`.
- **What was run:**
  - My own code, written from the env source, not from CONFIGONLY's scripts:
    - `scripts/ver.py`: rebuilds main's per-step reward and recomputes every checked setting offline.
    - `scripts/vv.py`, `scripts/envv.py`, `scripts/rcmp.py`: the verdicts.
    - `scripts/gd.py`: the gait-symmetry mechanism.
    - `scripts/val.py`, `scripts/val2.py`: offline recomputation against the re-rolls.
  - 54 simulation processes, run one at a time and queued by `scripts/batch.sh` and `scripts/batch2.sh`: 605 deterministic re-rolled episodes (`scripts/rr.py`, `scripts/rr2.py`). Each builds the env with `build_env(..., **overrides)`, so the config dict is changed in memory only.
- **Rules kept:** CPU only and no training. Nothing in the repository, the export, the configs on disk or the physics was changed. Nothing was written outside this folder except by the shell tool's own task-output files (slip 6). Every setting other than main is HYPOTHETICAL. Every computing or writing command went through `run.sh`, which logs the literal command with its UTC time in `CMDLOG.txt` and runs it under nice -n 10, OMP_NUM_THREADS=1 and TMPDIR in this folder.

## Verdict: CONFIRMED WITH CORRECTIONS

All four headlines reproduce with my own code, to the decimal CONFIGONLY printed. The recommended setting I6 was also re-rolled under the env itself on all 580 episodes its test needs. The corrections are small: one wrong [inferred] statement, and three imprecise ranges or wordings. None of them changes an answer.

| CONFIGONLY headline | my result | verdict |
|---|---|---|
| 1. The C1-C4 ladder fails at every rung, at both weights and on both panels. C1's knob pays alive by `bilateral_support_quality` (both feet loaded, 100 N each). It takes 433-460 from walks, 254-297 from hops and 1.2 from the statue. Every walk sits 430-466 below standing at w 0.2, and the hop's lead grows. The later rungs cannot undo it | Every number reproduced (§3). The C4 rung was also re-rolled under the env on 3 episodes and matches bitwise | CONFIRMED (correction 4: the per-walk means are 426-454 and 466 is the worst seed) |
| 2. `gait_symmetry_weight` does not reward alternation. At unit weight it pays the statue 997.6, the hops 917-982 and the walks 704-943, for three reasons | Reproduced, and the replay equals the env's own `alternation_ratio` bitwise on 4 re-rolled episodes (§3) | CONFIRMED (correction 2: the reset touchdowns are one foot on step 1 and the other a few steps later, in either order, not "R then L") |
| 3. I6 (airborne 1.0 + LB penalty 0.8 at 33.6 N + idle 0.1, cap 2.5) passes the whole test on all 40 seeds at both weights. Worst seed +41.6 over standing at w 0.2, +875.9 / +760.0 over the matched hop, +108.6 over H20. Against R15 +59.0 and R6 +87.0, +954.6 / +778.0, +41.7 | Reproduced offline, and at w 2.0 straight from the env's own returns on 580 re-rolled episodes: +182.2 over standing, +760.0 over the matched hop, +108.6 over H20, every pair ahead. R4, R6, R13 and R15 reproduced from SCREEN's json (§4) | CONFIRMED (correction 1: more idle weight does buy margin) |
| 4. X4v12 (pre-declared) passes everything: +12.0 over standing, +25.8 over H20. A 1.5 cap fails one held-out seed by 2.5, as R4 and R13 did | Reproduced, and re-rolled under the env on 2 episodes at w 0.2, bitwise (§4) | CONFIRMED |

**Corrections:**

1. **"More idle weight buys nothing" is wrong.** CONFIGONLY marks it [inferred: the term is linear in its weight]. Once stepping in place (W02) is the best non-walk, each +0.1 of `idle_penalty_weight` still widens the walk's lead. The lead grows by 0.1 × (W02's 51 idle units − a walk's 3-4), about +4.8.
   - I7 (I6 with idle 0.2, my probe) gives a worst seed of +46.4 against I6's +41.6 at w 0.2, and +187.0 against +182.2 at w 2.0. The mean margin of every walk over standing also rises, by +4.9 to +5.1 at w 0.2 and +4.7 to +4.8 at w 2.0 [measured: out/vv_stdout.txt].
   - So the term gives diminishing returns, but not none. CONFIGONLY's other wording, "the idle term cannot push it much further", is fair.
2. **The gait-reset wording.** Under the statue and the parent, exactly two touchdowns are ever logged: one foot on control step 1 and the other a few steps later (step 2-4 on the seeds printed in out/gd_stdout.txt), in either order (seed 3043 logs L then R). The ratio is 1.0 from step 3 (median) onward [measured: out/gd_stdout.txt].
   - The mechanism CONFIGONLY gives is right: the initial contacts count as touchdowns and the history never decays.
   - "The reset is logged as two touchdowns (R then L)" is not literally right.
3. **The LB0 agreement range.** "At the default 0 N it fires on the same steps to within 1.1 per episode" holds for the first panel's behaviour means of the 20 deterministic behaviours (H20 differs by 1.1) [measured: out/unsup_check.txt].
   - On the held-out panel H20 differs by 1.45 on average. Single episodes differ by up to 3 steps.
   - Pilot 3's actual start, `parent_ls15` (stochastic, not in the test), differs by 5.75 on average and by up to 13 steps.
   - This does not affect any verdict: LB0's own margins are reproduced.
4. **The C1 standing range.** "Every walk sits 430-466 below standing at w 0.2" mixes means and the worst seed. The per-walk mean margins over both panels are −425.7 (V35) to −453.7 (V01), and the worst walk-seed pair is −466.4 [measured: out/vv_stdout.txt]. CONFIGONLY's body says "about 430-455", which is close.

One unreported slip in CONFIGONLY's log, of the kind it already reports, is in §6.

## 1. What each knob computes (re-derived from the code of main 4d5f4e7)

Paths are relative to the export's `environments/`. r and l are each foot's touch force, taken as the MIN over the control step's 5 physics substeps (`shared/base_env.py:979-992`; `frame_skip` 5, `tyrannosaurus_rex/base_env.py:215`). These are the `r_foot_contact` and `l_foot_contact` that SCREEN recorded.

Main's `locomotion.toml` leaves every knob below at its class default, except `forward_vel_max` (2.5), `gait_symmetry_weight` (0.0, set explicitly) and `height_weight` (0.3). `backward_vel_penalty_weight` stays at 0, so `forward_vel_max` reaches the reward only through the forward term. The curriculum ramp changes only `forward_vel_weight` (`shared/train_base.py:1354-1366`), so the other weights of I6 and X4v12 hold through the ramp, and "w 0.2 / w 2.0" are its two ends.

| knob | what the code computes | where | CONFIGONLY |
|---|---|---|---|
| `support_conditioned_alive_fraction` f | alive = 0.5 · ((1 − f) + f · q), with q = min over both feet of clip(force / `foot_contact_saturation_force`, 0, 1) and the saturation force 100 N by default. So q is 1 only when both feet carry 100 N or more, and 0 when either reads 0 | trex base_env.py:950-959, 765-772, 937-940; shared/reward_functions.py:314-331 | correct |
| `airborne_substep_weight` a | −a · (number of the step's substeps on which every foot reads ≤ 0.1 N) / 5. It reads the per-substep forces, not the MIN, and is added after the 29-term sum only when a > 0, before the fall penalty | trex :1219-1233, :1323-1324; `_AIRBORNE_TOUCH_N` 0.1 at :203; shared/base_env.py:1355-1358 | correct |
| `foot_load_balance_airborne_penalty` p, with `foot_load_balance_min_support_force` m | −`foot_load_balance_weight` · imbalance − p · [r + l ≤ m]. `foot_load_balance_weight` is 0 in locomotion, so only the penalty is left. At m = 0.04 BW (33.636 N) the flag is exactly SCREEN's R1 "unsupported" flag | trex :774-788, :946-948; shared/reward_functions.py:334-397 | correct |
| `forward_vel_max` vmax | w · clip(v / vmax, −1, 1) | shared/base_env.py:638-660; reward_functions.py:49-63 | correct |
| `gait_symmetry_weight` g | g · alternation_ratio. A foot "touches down" on the step its substep-MIN force goes from ≤ 0.1 N to > 0.1 N. R is appended before L. The last 20 touchdowns are kept, and the state resets only at episode reset. The ratio is the share of consecutive pairs that switch foot, or 0 while fewer than 2 are logged | shared/base_env.py:709-787; trex :499, :1201-1206, :1394 | correct (correction 2) |
| `idle_penalty_weight` wi | −wi · (1 − \|v\| / 0.05) when the planar speed \|v\| = ‖qvel[0:2]‖ is below `idle_velocity_threshold` (0.05 m/s), else 0 | shared/base_env.py:694-707; reward_functions.py:651-665 | correct |
| `floor_impact_weight` | −weight · max(0, step peak of the summed per-substep touch / body weight − threshold), added after the sum | trex :1219-1231, :1321-1322 | correct |
| `bilateral_support_weight` | weight · q, so it pays both feet loaded (standing) | trex :937-940 | correct |
| `foot_contact_weight`, `foot_contact_gate` | stored (:355-356) and never read by the CPU reward. `shared/task_fingerprint.py:100-101` says only the retired MJX reward read them | — | correct |

[measured] Each formula above reproduces the env's own reward. Applied to SCREEN's records, they equal the env's per-step reward from the re-rolls with the config changed in memory (§2). Also:

- The q recomputed from r and l equals the recorded `bilateral_support_quality` on every step (max |diff| 0.0).
- The height term rebuilt from `pelvis_height` equals the recorded term bitwise on every episode.
- `airborne_substeps` takes the values 0-5 on H23 seed 3042.

Per-episode diagnostics, first panel, mean over seeds [measured: out/vv_stdout.txt, diagnostics table]:

- **q = 0** on 85.7-90.7% of a stable walk's steps, and **q = 1** on 99.74% of the statue's. CONFIGONLY: "86-91%" and "99.7%".
- **Airborne substeps:** hops 2429-2629, stable walks 0-12.6, V29 74, V33 146.
- **Steps with r + l ≤ 33.6 N:** hops 505-584, V18 10.9, V35 19.2, V29 60, V33 67.
- **Peak floor force, p95 of the step peak:** V18 and V35 2.1 BW, hops 3.3-4.1 BW.
- **No recorded step** has r + l between 33.6 and 33.636 N, so the recommended 33.6 N and the 33.636 N used offline give identical flags.

## 2. Reproduction and re-roll validation

- **Main, offline** [measured: out/ver_stdout.txt]. My rebuild of main's per-step reward, in the env's summation order with the fall penalty added last, equals the recorded `env.step` reward on 643,346 of 643,346 steps and 660 of 660 returns (both panels, all 21 + 12 behaviours). Every return also equals SCREEN's json.
- **Re-rolls under main** [measured: out/val_stdout.txt, out/val2_stdout.txt]. The statue, parent, V24, V35, H23 and HF120 on seed 3042 were re-rolled through `build_env` with no overrides. Their rewards equal SCREEN's records bitwise, and their actions and pelvis positions are identical.
- **Re-rolls under I6** (HYPOTHETICAL, in memory: `airborne_substep_weight` 1.0, `foot_load_balance_airborne_penalty` 0.8, `foot_load_balance_min_support_force` 33.6, `idle_penalty_weight` 0.1) [measured: out/batch_console.txt, out/envv_stdout.txt].
  - 580 episodes were run at the config weight 2.0: the 16 test behaviours on 3042-3061, the 11 held-out ones on 3062-3081, and V29 and V33.
  - Every episode's actions, forward_vel and pelvis positions are bitwise equal to SCREEN's record (551/551 in batch 1, 52/52 in batch 2, 2/2 test runs).
  - My offline I6 return equals the env's return bitwise on 580 of 580 episodes.
- **Per-step check.** In `rr.py` I took |v| with a row-wise `norm(axis=1)`, which differs from the env's one-vector norm in the last bit on 1,231 of 580,431 steps. So the offline per-step reward from that speed matches on 588,361 of 588,431 re-rolled steps, and differs by at most 2.2e-16 on the rest. The returns are still bitwise equal.
  - `rr2.py` stores the velocity and takes the norm one vector at a time, as the env does. With that speed, the I6 idle term equals the env's `reward_idle` on 5,000 of 5,000 steps. The offline I6 reward equals the env's on 5,000 of 5,000 steps.
  - The speed arrays are bitwise identical across configs and weights (13 of 13 compared), so the speed does not depend on the config.
- **The five chosen episodes** (statue, walks V24 and V35, hops H23 and HF120; seed 3042). Env return from the HYPOTHETICAL re-roll (I6 set in memory) against my offline recomputation from SCREEN's records [measured: out/val_stdout.txt, out/val2_stdout.txt]:

| episode | main (w 2.0) | I6, w 2.0: env / offline | I6, w 0.2: env / offline | per-step bitwise (w 2.0 / w 0.2) |
|---|---|---|---|---|
| statue | 1095.022386 | 1015.712134 / 1015.712134 | 1016.570899 / 1016.570899 | 1000 / 1000 |
| V24 (0.7 m/s walk) | 1467.711794 | 1467.711794 / 1467.711794 | 1075.273084 / 1075.273084 | 1000 / 1000 |
| V35 (1.2 m/s walk) | 1741.423095 | 1725.223095 / 1725.223095 | 1096.524348 / 1096.524348 | 1000 / 1000 |
| H23 (0.7 m/s hop) | 1540.684457 | 606.950836 / 606.950836 | 180.803332 / 180.803332 | 1000 / 1000 |
| HF120 (1.2 m/s hop) | 1841.368531 | 878.609549 / 878.609549 | 174.682903 / 174.682903 | 1000 / 1000 |

- **Other settings re-rolled in memory, at w 0.2** (offline = env, bitwise on every step):
  - X4v12: statue 1095.777782, V35 1164.080778.
  - C4 with g 0.1 (`support_conditioned_alive_fraction` 1.0 + airborne 1.0 + cap 1.5 + gait 0.1): statue 1194.617539, V24 768.411976, H23 467.528073.
- **G1 re-roll** (`gait_symmetry_weight` 1.0, w 2.0) on the statue, V24, V18 and H23. My replay of the touchdown rule equals the env's `alternation_ratio` (= `reward_gait`) on 4,000 of 4,000 steps. The offline G1 reward equals the env's bitwise. The ratios sum to 998.0, 959.6, 740.2 and 984.2 per episode.

## 3. The task's ladder C1-C4 and the gait-symmetry helper (HYPOTHETICAL, offline)

Margins are per episode. "Seed min" is the worst walk-seed pair over both panels: 200 walk-standing pairs, 180 walk-hop pairs and 140 walk-H20 pairs. Standing is the best of the statue, the parent and W02 on each seed [measured: out/vv_stdout.txt].

| setting | walk − standing, seed min (w 0.2 / 2.0) | walk − matched hop, seed min (w 0.2 / 2.0) | CONFIGONLY | PASS |
|---|---|---|---|---|
| main | −30.5 / +123.6 | −59.2 / −180.6 | same | no |
| C1 (f 1.0) | −466.4 / −331.1 | −219.2 / −342.0 | same | no |
| C2 (+ a 1.0) | −466.4 / −331.1 | +287.4 / +166.0 | same | no |
| C2h (+ a 0.5) | −466.4 / −331.1 | +35.1 / −86.3 | same | no |
| C3 (C2 + vmax 1.5) | −452.2 / −237.3 | +282.2 / +98.5 | same | no |
| C4, g 0.05 / 0.1 / 0.25 | −454.7, −460.9, −488.5 / −238.0, −244.2, −262.8 | +272.6, +258.7, +211.2 / +92.7, +87.0, +65.8 | same | no |

- **C1's cost against main** (first panel, mean per episode) [measured]:
  - stable walks −433.1 (V24) to −459.5 (V05);
  - hops −254.2 (HF020) to −296.4 (HF150);
  - statue −1.2, parent −1.1, W02 −444.4.
  
  These match CONFIGONLY's "433-460", "254-297" and "1.2".
- **C1's per-walk mean margin over standing at w 0.2** is −425.7 (V35) to −453.7 (V01); the worst seed is −466.4 (correction 4).
- **C1 with the saturation force at 0.1 N** (`C1sat01`): the stable walks still lose 428.7-453.4 per episode, so they keep 46.6-71.3 of their 500 alive. CONFIGONLY: "47-71".
- **The single flight knobs:**
  - Airborne 1.0 (A1): +445.4 / +324.0 over the matched hop. Airborne 0.5 (A05): +193.1 / +71.7. The LB penalty 0.8 at 33.6 N: +369.6 / +252.1. At the default 0 N: +369.6 / +251.3.
  - All leave standing at −30.5 at w 0.2.
  - What they cost the stable walks: A1 0 to 2.5, LB 0 to 15.3.
  - V24's deficit to the statue under main is the height term, −50.3 per episode, against +44.9 of forward pay at w 0.2 (+45.2 on the held-out panel).
- **`gait_symmetry_weight` at unit weight (G1), per episode** [measured]:
  - statue 997.6, parent 997.9;
  - W02 965.2;
  - stable walks: FINAL3 942.0, V01 943.0, V05 909.1, V24 911.5, V18 704.5, V35 771.7;
  - F hops 951.1-982.4, H20 916.6.
  
  So standing earns most, then the hops, then the walks.
- **Mechanism** [measured: out/gd_stdout.txt, first panel, mean per episode]:
  - Statue and parent: exactly 2 touchdowns are ever logged (correction 2).
  - F hops: both feet touch down on the same control step 44-66 times (of 97-143 logged touchdowns). The code appends R then L, so each two-footed landing yields alternations.
  - V18 and V35: 23.5 and 23.7 same-foot repeats, 21.1 and 17.0 of them after an unloading of 2 control steps or less (landing bounces).
  - The docs say the same: GAIT_QUALITY_PLAN_2026_09.md:394 ("nothing rewards alternation" for the T. rex), :396 (reset touchdowns, history never decays) and KNOWN_ISSUES.md:912-944 (2026-09-28 note: "a synchronous two-foot landing appends R then L").

## 4. The passing config-only settings (HYPOTHETICAL)

Same test and columns. The 0.5-1.2 m/s walks (V05, V24, V18, V35) are tested against H20 on every seed [measured: out/vv_stdout.txt, out/envv_stdout.txt, out/rcmp_stdout.txt]. "CONFIGONLY" means its §4 table gives the same figures.

| setting | walk − standing, seed min (w 0.2 / 2.0) | walk − matched hop (w 0.2 / 2.0) | 0.5-1.2 walk − H20 (w 0.2 / 2.0) | main test | H20 test | CONFIGONLY |
|---|---|---|---|---|---|---|
| **I6** = A1 + LB + idle 0.1 (post hoc) | **+41.6 / +182.2** | **+875.9 / +760.0** | **+880.3 / +108.6** | YES | YES | same |
| I6 from the env's own returns (580 re-rolls, w 2.0) | — / +182.2 | — / +760.0 | — / +108.6 | YES | YES | — |
| I5 = A1 + LB + idle 0.05 | +8.3 / +162.4 | +875.8 / +760.0 | +880.3 / +108.6 | YES | YES | same |
| I2 = A1 + idle 0.1 | +41.6 / +182.2 | +445.4 / +324.0 | +444.3 / −333.1 | YES | no | same |
| I1w02 = A1 + idle 0.02 | −15.0 / +139.2 | +445.4 / +324.0 | — | no | no | same |
| I7 = I6 with idle 0.2 (my probe) | +46.4 / +187.0 | +875.9 / +760.1 | +880.3 / +108.6 | YES | YES | — |
| **X4v12** = A1 + LB + cap 1.2 (pre-declared) | **+12.0 / +276.0** | **+870.4 / +734.9** | **+893.9 / +25.8** | YES | YES | same |
| X4v14, X4v13, X4v10 (caps 1.4, 1.3, 1.0) | +2.5, +8.2, +18.1 / +234.1, +253.4, +334.6 | ≥ +868.5 / ≥ +692.8 | H20 at w 2.0: −30.1, −7.8, +130.1 | YES | no, no, YES | same |
| X4 (cap 1.5), X24 (A1 + cap 1.5) | −2.5 (V24, held-out seed 3073) / +217.4 | +872.5, +440.1 / +692.6, +265.4 | H20 at w 2.0: −43.1, −507.9 | no | no | same |
| X2 = A1 + LB | −30.5 / +123.6 | +875.8 / +760.0 | +880.2 / +108.5 | no | YES | same |
| X24h0 = A1 + height_weight 0 | +0.7 / +119.1 | +469.1 / +349.3 | H20 at w 2.0: −338.8 | YES | no | same |
| R6 (SCREEN, code) | +87.0 / +307.3 | +954.6 / +778.0 | +960.0 / +41.7 | YES | YES | same |
| R15 (SCREEN, code) | +59.0 / +213.5 | +451.8 / +337.5 | +457.0 / −316.1 | YES | no | same |
| R4, R13 (SCREEN, code) | −2.5 / +217.4 | +865.1, +358.9 / +689.3, +169.9 | H20 at w 2.0: −49.4, −588.6 | no | no | same |

- **What I6 costs against main** (first panel, mean per episode, the same at both weights) [measured]:
  - statue −77.2, parent −91.9, W02 −5.1;
  - stable walks −0.3 (FINAL3, V01) to −18.2 (V35);
  - V29 −63.1, V33 −83.1;
  - F hops and H20 −892.6 to −977.8.
  
  These match CONFIGONLY's "statue 77, parent 92, W02 5, walks 0.3-18.2, V29 63, V33 83". Its "hops 893-993" goes up to HF150, which I did not re-roll under I6. My offline A1 + LB cost for HF150 is −992.6 before its small idle term, which agrees [inferred].
- **Idle units per episode** (the sum of 1 − |v| / 0.05 over steps with |v| < 0.05; first panel mean) [measured]:
  - statue 772.4 (held-out 794.1);
  - parent 919.5;
  - W02 51.1;
  - stable walks 3.3-3.4 (at most 10.2 on any episode);
  - hops H23 6.6, HF120 6.1 and H20 4.3 (CONFIGONLY: HF020 25.5, the other hops 5.6-8.2).
- **Which standing behaviour binds.** Under I6 the best standing behaviour is W02 on 38 of 40 seeds at w 0.2 and 40 of 40 at w 2.0. Under I5 (idle 0.05) it is the statue on all 40. This matches CONFIGONLY's "from about 0.09 upward, stepping in place is the best non-walk".
- **Discounted return** (γ 0.995 from reset, w 0.2, first panel mean) [measured]:
  - I6: V24 212.0, statue 208.2, W02 206.5, parent 203.5.
  - X4v12: V24 217.4, parent 217.3, statue 216.8.
  - Main: V24 212.3, parent 217.5.
  - R6 (SCREEN): walks 227.3-234.4, statue 216.8.
  
  These match CONFIGONLY's figures and its "about 0-4 units" margin.
- **The gate argument** [inferred, as CONFIGONLY labels it]. V35's episode-mean forward_vel is 0.884 at a 1.17 m/s cruise, and V29's 1.031 at 1.34 (SCREEN's json). A cap at or below 1.4 therefore stops paying close to the speed the 1.0 m/s gate needs. I agree with that reading, and with keeping I6's cap at 2.5.

## 5. What this does not show

These points restate CONFIGONLY's limits; I agree with them.

- These are 21 fixed hand-built behaviours, not training. That I6 ranks them correctly does not say what PPO would learn under I6.
- I6's standing margin at w 0.2 rests on charging stillness. W02 pays only 51 idle units, and a learned sway that keeps |v| ≥ 0.05 m/s would pay none [inferred].
- I6 was chosen after the first-panel idle verdicts had been read. The held-out panel then confirmed it, but that panel has no V01, V05 or 0.2-0.5 m/s hops.
- Every passing setting moves the statue's return: I6 by −77.2 per episode. `collapse_peak_floor_reference` (1092.6) would need re-measuring [inferred: from locomotion.toml's comment].

## 6. CONFIGONLY's labels, process and slips

- **Pre-declaration** [measured: configonly/CMDLOG.txt].
  - X4v12 is in the 06:27:10 PRE-DECLARED note, before any number was computed.
  - The cap scan, the height extras and the first idle settings were declared at 06:32:20 and later, after the first verdict tables.
  - X2 and I6 were declared at 06:55:44, after the first-panel idle verdicts had been read and before they were computed.
  
  RESULT.md says all of this (slip 1, "extra" labels). Calling I6 "post hoc" is correct.
- **[measured] / [inferred] labels.**
  - The measured claims I re-derived hold, apart from the imprecise ranges in corrections 2-4.
  - "More idle weight buys nothing" is labelled [inferred], and the inference is wrong (correction 1).
  - The recommendation and the gate argument are labelled [inferred], correctly.
- **Its validations.** Its out/val_HYP_stdout.txt, out/idle_*_stdout.txt and out/cfg_main_stdout.txt show what its RESULT says: 16,000 of 16,000 steps, 660 of 660 re-rolled episodes, and alternation_ratio equal on 643,346 steps. I did not re-run its combined 10-knob dict. My own in-memory re-rolls (I6, X4v12, C4 with g 0.1, G1) cover the knobs that matter here, and every one is bitwise equal to the offline recomputation.
- **CMDLOG:**
  - Commands are logged literally with UTC times.
  - A search of its CMDLOG, scripts and RESULT.md for any device path ("/dev/") finds nothing.
  - Simulations ran one at a time.
- **One unreported slip, of the kind it already reports.** Its 07:08:41 NOTE says RESULT.md was edited "between 06:41 and 07:10". The end time 07:10 is later than the note's own timestamp and than RESULT.md's modification time (07:08:41). This is a third typed clock time that ran ahead of the real clock. Its slip 2 covers two, and no later note corrects this one. It is minor.
- **Not checked:** whether CONFIGONLY's files contain an AI model name. Searching for one would need me to type the names.

## 7. My own slips

1. **Read-only checks outside `run.sh`.** Before the wrapper existed, I ran date and ls of overnight/ and configonly/ (07:09). Later I ran two progress checks, at 07:14:53 and 07:15:42: date, cat or tail of out/batch_console.txt, and ls | wc. All are listed in NOTE lines. The `echo ... >> CMDLOG.txt` lines that write NOTEs also ran outside it.
2. **File-tool writes appear as NOTE lines, not literal commands.** These are rr.py, ver.py, vv.py (also edited with the edit tool), val.py, val2.py, gd.py, envv.py and RESULT.md. rr2.py and two extensions of ver.py were made by python text edits, which run.sh logged literally. One sed edit of vv.py was logged literally too.
3. **Offline analysis alongside simulation.** The first pass of ver.py, vv.py, val.py, gd.py and rcmp.py ran while batch 1 was simulating. They only read files. No two simulations ever ran at once.
4. **Waiting.** One background wait loop (`until grep ...; do sleep 10`) and three foreground waits (`timeout 280/290 tail -f ... | grep -m1`) ran through run.sh, so they are logged and niced. The shell tool's own background wrapper did not run under nice; the batches it launched did. run.sh nested inside batch.sh applies nice twice (capped at 19).
5. **rr.py's speed.** A row-wise norm differs in the last bit from the env's (§2). I found it and closed it with rr2.py. All returns were bitwise equal regardless.
6. **Files outside this folder.** The shell tool writes its own background-task output files in the session's tasks folder, and it saved one long tool output under the session's tool-results folder. These are not mine to avoid. They hold only console text (grep counts, a time, the CONFIGONLY RESULT text I read). I read two of them with the file-read tool.
7. **A draft wording error.** A draft of this file said "every hop step has 0-5 airborne substeps" after checking one episode. It was corrected to "H23 seed 3042" before this version.
8. **A typed clock time ran ahead of the clock.** One of my NOTE lines said RESULT.md was rewritten "at about 07:50", but the rewrite happened at about 07:43-07:44. A later NOTE corrects it. This is the same kind of slip I flagged in CONFIGONLY.
9. **Nothing else.** No null-device references, no network, no git writes, nothing under /home/user/mesozoic-labs, no training. The folder is 24M at the end.

## Files

- `CMDLOG.txt`: literal commands with UTC times, plus NOTE lines.
- `run.sh`: the logging wrapper.
- Scripts (`scripts/`):
  - `rr.py` and `rr2.py`: the in-memory re-rolls. `batch.sh` and `batch2.sh`: the sequential queues.
  - `ver.py`: offline reproduction of main and every checked setting. `vv.py`: the verdict tables. `envv.py`: the I6 verdict from the env's own returns. `rcmp.py`: R4, R6, R13 and R15 from SCREEN's json.
  - `val.py` and `val2.py`: offline against re-roll, per step. `gd.py`: the gait-symmetry mechanism.
- Outputs (`out/`):
  - `ver.json`: every return, discounted returns and diagnostics. `ver_stdout.txt`, `vv_stdout.txt`, `envv_stdout.txt`, `val_stdout.txt`, `val2_stdout.txt`, `gd_stdout.txt`, `rcmp_stdout.txt`, `unsup_check.txt`.
  - `*_pass1_*` files: the first pass, run before batch 1 finished.
  - `batch_console.txt` and `batch2_console.txt`: one line per re-rolled episode, with the trajectory-identity flags.
  - `rr/` and `rr2/`: per-step env reward and speed of every re-rolled episode, as small npz files.
