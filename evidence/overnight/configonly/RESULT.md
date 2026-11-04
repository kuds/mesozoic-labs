# CONFIG SCREEN: can existing T. rex config knobs alone make every walk beat standing and an equally fast hop?

Stage folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/configonly/`. Paths below are relative to it.

- **When:** 06:22-07:08 UTC on 2026-11-04 (time box: write-up by 09:30 UTC).
- **Inputs:** the SCREEN stage's recorded episodes (`overnight/screen/out/<B>_s<seed>.npz` and `HO_<B>_s<seed>.npz`, read-only): 21 behaviours on seeds 3042-3061 and 12 of them on the held-out seeds 3062-3081, on the r10 T. rex locomotion env of main 4d5f4e7 (the export `feas/main_4d5f4e7`). SCREEN's own R6, R15, R13 and R4 returns are read from its `out/screen_main.json` and `out/screen_HO.json`.
- **What was run:**
  - Offline: main's per-step reward rebuilt from the recorded terms, then every HYPOTHETICAL setting recomputed with the env's own arithmetic and summation order (`scripts/cfg.py`).
  - Deterministic re-rolls of saved behaviours with a config dict changed in memory only (`build_env(..., **overrides)`, `scripts/reroll.py`, `scripts/reroll_lite.py`), one simulation process at a time, to validate the offline recomputation and to record the speed the idle term reads.
- **What was not done:** no training; no change to the repository, the export, any config on disk or the physics. Every setting below is HYPOTHETICAL. Nothing was written outside this folder.
- **How:** every command that computed or wrote went through `scripts/run.sh` (or `scripts/batch_lite.sh`). These write the literal command with its UTC time to `CMDLOG.txt` and run it under nice -n 10, OMP_NUM_THREADS=1 and TMPDIR inside this folder, with `optB/venv/bin/python -I -B`. A few read-only progress checks ran outside them and are listed in NOTE lines (slip 3).
- **Units and terms:** as in SCREEN. Return = undiscounted sum over one 10 s, 1000-step episode. w = forward_vel_weight, 0.2 at the ramp start and 2.0 at its end. Standing = per seed, the best of the statue, the stance parent and W02 (stepping in place). Stable walks = FINAL3, V01, V05, V24, V18, V35 (0.2-1.2 m/s), each matched with the F hop at the same speed command (HF020, HF030, HF050, H23, H24, HF120). H20 is the 2.0 m/s hop. "40 seeds" = 3042-3061 plus the held-out 3062-3081, where the held-out panel has FINAL3, V24, V18 and V35 (not V01, V05 or the 0.2-0.5 m/s hops).
- **Test** (as SCREEN): PASS = every stable walk above standing and above its matched hop on every recorded seed of both panels, at w 0.2 and at w 2.0. Second test: every 0.5-1.2 m/s walk above H20 on every seed at both weights.

## Short answer

1. **Main is reproduced exactly.** [measured: out/cfg_*_stdout.txt, out/val_*_stdout.txt, out/idle_*_stdout.txt]
   - Rebuilding main's per-step reward from SCREEN's records gives 643,346 of 643,346 steps and 660 of 660 returns bit-identical to `env.step`.
   - Re-rolling all 660 episodes under main gives the same rewards bitwise.
   - A re-roll under one combined HYPOTHETICAL override dict matches my offline recomputation bitwise on 16,000 of 16,000 steps.
2. **The task's ladder C1 → C4 fails at every rung, at both weights, on both panels.** [measured: §2]
   - **C1 is not the compsognathus gate.** `support_conditioned_alive_fraction` multiplies alive by `bilateral_support_quality`, which needs both feet loaded (up to 100 N each). It is not "some support".
     - A walk's single support therefore counts as unsupported, and that is 86-91% of its steps.
     - C1 takes 433-460 per episode from each stable walk, 254-297 from each hop and 1.2 from the statue.
     - Every walk drops below standing (worst seed −466.4 at w 0.2, −331.1 at w 2.0), and the hop's lead over its walk grows.
   - **The later rungs cannot undo C1.** C2 (`airborne_substep_weight` 1.0) fixes the matched hop (worst seed +287 at w 0.2, +166 at w 2.0). C3 (cap 1.5) and C4 (`gait_symmetry_weight` 0.05-0.25) still leave the worst walk 440-463 below standing on average at w 0.2 (worst seed −452 to −489).
   - Lowering `foot_contact_saturation_force` does not rescue C1: at 0.1 N the stable walks keep 47-71 of their 500 alive per episode.
3. **`gait_symmetry_weight` pays standing most.** [measured + code: §1]
   - At unit weight it pays the statue 997.6 per episode (of 1000), the two-footed hops 917-982 and the walks 704-943 (the 1.0 m/s walk least).
   - The causes:
     - The reset is logged as two touchdowns (R, L), and the 20-touchdown history never decays.
     - A simultaneous landing is logged as R then L, which counts as an alternation.
     - The per-step MIN contact turns a walk's landing bounces into extra same-foot touchdowns.
   - This is what the gait plan's audit and KNOWN_ISSUES.md:912-944 record. The T. rex weight is 0, so "nothing rewards alternation".
4. **A different config-only amendment does pass.** The passing settings below are HYPOTHETICAL and use existing knobs only. Except X4v12, they were added post hoc and declared before computing (slip 1). [measured: §4]
   - **The hop half needs one knob that prices flight.** Either `airborne_substep_weight` 1.0, or `foot_load_balance_airborne_penalty` 0.8 with `foot_load_balance_min_support_force` 33.6 N (SCREEN's R1 flag as a penalty).
     - With either, every walk leads its matched hop on all 40 seeds at both weights.
     - They cost the stable walks at most 2.5 and 15.3 per episode respectively.
   - **The standing half needs a second knob.**
     - `idle_penalty_weight` 0.1 gives a worst seed of +41.6 at w 0.2 (0.05 gives +8.3).
     - `forward_vel_max` at or below 1.4 gives +2.5 at 1.4, +12.0 at 1.2 and +18.1 at 1.0. At 1.5 it fails by 2.5 on one held-out seed, like R4 and R13.
   - **I6 = `airborne_substep_weight` 1.0 + `foot_load_balance_airborne_penalty` 0.8 (min support 33.6 N) + `idle_penalty_weight` 0.1 passes the whole test.**
     - Every stable walk is above standing and above its matched hop on all 40 seeds at both weights.
     - Every 0.5-1.2 m/s walk is above the 2.0 m/s hop on every seed.
   - **I6's margins beside R6 and R15 (worst seed):**

     | margin | I6 | R6 | R15 |
     |---|---|---|---|
     | over standing, w 0.2 | +41.6 | +87.0 | +59.0 |
     | over standing, w 2.0 | +182.2 | +307.3 | +213.5 |
     | over the matched hop, w 0.2 / w 2.0 | +875.9 / +760.0 | +954.6 / +778.0 | +451.8 / +337.5 |
     | 0.5-1.2 m/s walks over H20, w 2.0 | +108.6 | +41.7 | fails |

   - **The pre-declared X4v12** (the same flight prices plus `forward_vel_max` 1.2) also passes everything, with +12.0 over standing and +25.8 over H20.
5. **What it costs.** [measured: §4; inferred where marked]
   - I6 costs the stable walks 0.3-18.2 per episode (V35's landing bounces under the LB penalty), the 1.5 m/s walk V29 63 and the failed fast walk V33 83.
   - It costs the hops 893-993, the statue 77, the parent 92 and stepping in place (W02) 5.
   - The idle term charges stillness, not the absence of alternation. W02 pays only 51 × w_idle per episode against the statue's 772 × w_idle, so from w_idle ≈ 0.09 the margin is V24 − W02, and more idle weight buys nothing [inferred: linear in w_idle]. A learned fidget might evade it too [inferred].
   - The cap variants instead stop paying above the cap. At 1.2 that is below the speed the 1.0 m/s gate needs [inferred: Limits].
   - In discounted return at w 0.2 every config-only margin is small: under I6, V24 has 212.0 against the statue's 208.2 and W02's 206.5.
6. **Recommendation** [inferred, for the maintainer; nothing here changes the repository]:
   - If the change must be config-only, use I6: `airborne_substep_weight` 1.0, `foot_load_balance_airborne_penalty` 0.8, `foot_load_balance_min_support_force` 33.6 and `idle_penalty_weight` 0.1. Keep `forward_vel_max` 2.5.
   - Re-measure `collapse_peak_floor_reference`, since the statue's return moves by −77.
   - Do not use `support_conditioned_alive_fraction` or `gait_symmetry_weight` for this purpose: both rank standing above walking.
   - Do not lower `forward_vel_max` below the gate's needs.
   - R15's event reward remains the better fix. It pays alternation itself: R15 changes the walks by +72 to +92 per episode and stepping in place by +8. So its standing margin does not depend on charging stillness, which stepping in place nearly escapes (−5 against the statue's −77 under I6).

## 1. What each knob computes (code of main 4d5f4e7)

File paths are relative to the export's `environments/`. r and l are each foot's touch force, taken as the MIN over the control step's five physics substeps (`_aggregated_foot_contact_forces`, the `r_foot_contact` and `l_foot_contact` info keys). The per-episode numbers are means over seeds 3042-3061 [measured: out/vd_main_stdout.txt "Diagnostics", out/satdiag_stdout.txt].

- **`support_conditioned_alive_fraction` f** (tyrannosaurus_rex/base_env.py:950-959).
  - It computes alive = 0.5 · ((1 − f) + f · q).
  - q is `bilateral_support_quality` (:937-940, :765-772; shared/reward_functions.py:314-331): the MIN over both feet of clip(r or l / `foot_contact_saturation_force`, 0, 1). The saturation force is 100 N by default.
  - So q needs both feet loaded at once. It is not the compsognathus support flag (r + l > 4% of body weight, where one foot is enough) that SCREEN's R1 used.
  - q is 1 on 99.7% of the statue's steps and on 40-49% of a hop's steps (stance on both feet). It is 1 on only 7.5-12% of a stable walk's steps and 0 on 86-91% of them: on 86-91% of its steps one foot reads 0.1 N or less.
  - At f = 1 the statue keeps 498.8 of its 500 alive per episode, the hops 199-246, the walks 40.5-66.9.
  - Lowering the saturation force does not help: at 0.1 N the stable walks keep 46.6-71.3 (W02 75.1) and the hops 200-246.
- **`airborne_substep_weight` a** (tyrannosaurus_rex/base_env.py:1208-1233, added after the main sum at :1323-1324).
  - It computes −a × (number of the step's substeps on which every foot reads ≤ 0.1 N) / 5. This prices flight directly. It is SCREEN's R2 term.
  - Airborne substeps per episode (of 5000): hops 2429-2629; stable walks 0-12.6; V29 74; V33 146.
- **`foot_load_balance_airborne_penalty` p**, with `foot_load_balance_min_support_force` m (tyrannosaurus_rex/base_env.py:774-788, 946-948; shared/reward_functions.py:334-397).
  - It computes −p on every step with r + l ≤ m. This does not depend on `foot_load_balance_weight`, which is 0 in locomotion, so the imbalance part stays 0.
  - With m = 33.6 N (4% of body weight) it is exactly SCREEN's R1 "unsupported" flag, used as a penalty instead of a gate.
  - Unsupported steps per episode: hops 505-584; V18 10.8; V35 19.1; V29 60; V33 67; the other walks and standing 0-0.3.
  - At m = 0 (the default) the flag fires on the same steps to within 1.1 per episode, because the touch sensors read exactly 0 in flight. Its docstring calls m = 0 a no-op in practice, but that was measured on stance runs.
- **`forward_vel_max` vmax** (shared/base_env.py:638-660; shared/reward_functions.py:49-63).
  - It computes w · clip(v / vmax, −1, 1) (SCREEN's R3 at 1.5).
  - Speed above vmax earns nothing. Below vmax the pay per m/s rises by 2.5/vmax over main's 2.5.
- **`gait_symmetry_weight` g** (tyrannosaurus_rex/base_env.py:1200-1206; shared/base_env.py:713-787; the T. rex env calls `_init_gait_state()` at :499 with the default 0.1 N threshold).
  - Each control step, a foot "touches down" when its substep-MIN touch goes from ≤ 0.1 N to > 0.1 N.
  - The env keeps the last 20 touchdowns and pays g × (share of consecutive pairs that switch foot) on every step.
  - Three properties of the helper keep it from rewarding alternation:
    - **The reset counts as two touchdowns, and the history never decays.** Both feet are found in contact on the first steps and are logged as R and then L, so the ratio is 1.0 from the third or fourth control step. A statue that never lifts a foot keeps that 1.0 for the whole episode. The statue earns 997.6 of a possible 1000 per episode at g = 1, and the parent 998.0.
    - **A two-footed landing counts as an alternation.** Both feet landing on the same step append R then L, so the F hops earn 917-982.
    - **Walking is counted against itself.** A one-step landing bounce or a quick handover drops a foot's substep-MIN to 0 and back, which logs an extra touchdown of the same foot. The slow walks earn 909-943, V18 704.5 and V35 771.7. W02 (stepping in place) earns 965.
  - So the term pays standing most, then the hops, then the walks, which is the wrong order for this test.
  - The gait plan's audit says the same:
    - GAIT_QUALITY_PLAN_2026_09.md:394 lists nothing that rewards alternation for the T. rex, because the weight is 0 in locomotion.toml.
    - Its line 396 and KNOWN_ISSUES.md:912-944 record the never-decaying history, the reset touchdowns and (2026-09-28 update) that "a synchronous two-foot landing appends R then L, so a bounce scores 1.000, as do a true alternating walk and the statue".
    - The plan keeps the code byte-identical and sets the weight to 0 (§5.3).
- **Other knobs** (all read in the same file):
  - `floor_impact_weight` with `floor_impact_threshold_bw` (:1219-1226, :1321-1322) prices each step's peak floor force above the threshold. It charges landings, not flight.
  - `bilateral_support_weight` (:937-940) pays q, so it rewards both feet loaded.
  - `height_weight` (:1050-1082) pays clip((pelvis height − 0.70) / (0.926 − 0.70), 0, 1).
  - `idle_penalty_weight` with `idle_velocity_threshold` (shared/reward_functions.py:651-665) charges w_idle · (1 − |v| / threshold) when the planar speed |v| is below the threshold. It is the one knob that prices standing still. It does not price flight or reward alternation, so it is an extra here.
  - `foot_contact_weight` and `foot_contact_gate` are stored (:355-356) but never used in the T. rex reward.
  - `foot_terms_min_support_force` only gates the flatness and width terms, whose weights are 0 in locomotion.

## 2. The task's ladder C1-C4

None of the rungs passes, at either weight, on either panel [measured: out/comb_stdout.txt, out/vd_main_stdout.txt, out/vd_HO_stdout.txt]. Margins are per episode. "Seed min" is the worst walk-seed pair over both panels.

| setting | config (HYPOTHETICAL; rest = main) | w 0.2: worst walk − standing, mean (walk) / seed min | w 2.0: walk − standing, seed min | walk − matched hop, seed min (w 0.2 / w 2.0) | PASS |
|---|---|---|---|---|---|
| main | locomotion.toml | −16.8 (V24) / −30.5 | +123.6 | −59.2 / −180.6 | no |
| C1 | support_conditioned_alive_fraction 1.0 | −453.7 (V01) / −466.4 | −331.1 | −219.2 / −342.0 | no |
| C2 | C1 + airborne_substep_weight 1.0 | −453.7 (V01) / −466.4 | −331.1 | +287.4 / +166.0 | no |
| C2h | C1 + airborne_substep_weight 0.5 | −453.7 (V01) / −466.4 | −331.1 | +35.1 / −86.3 | no |
| C3 | C2 + forward_vel_max 1.5 | −440.2 (FINAL3) / −452.2 | −237.3 | +282.2 / +98.5 | no |
| C3h | C2h + forward_vel_max 1.5 | −440.2 (FINAL3) / −452.2 | −237.3 | +29.9 / −161.2 | no |
| C4 g 0.05 | C3 + gait_symmetry_weight 0.05 | −443.0 (FINAL3) / −454.7 | −238.0 | +272.6 / +92.7 | no |
| C4 g 0.1 | C3 + gait_symmetry_weight 0.1 | −445.8 (FINAL3) / −460.9 | −244.2 | +258.7 / +87.0 | no |
| C4 g 0.25 | C3 + gait_symmetry_weight 0.25 | −462.7 (V18) / −488.5 | −262.8 | +211.2 / +65.8 | no |
| C4h g 0.05 / 0.1 / 0.25 | C3h + gait_symmetry_weight | −443.0 / −445.8 / −462.2 (mean) | −238.0 / −244.2 / −262.8 | +20.3, +6.2, −41.3 / −167.0, −172.7, −190.0 | no |

- **C1 is what breaks the ladder.** It takes 433-460 per episode from the stable walks and 254-297 from the hops, at both weights, because q is 0 through a walk's single support. [measured: out/vd_main_stdout.txt, "What each setting does"]
  - All six walks fall about 430-455 below standing at w 0.2. The statue loses only 1.2.
  - The hop's lead over its walk grows (V24 − H23: −35.1 under main, −200.7 under C1 at w 0.2).
- **C2 (the airborne term) fixes the hop but not standing.**
  - At a = 1.0 every walk leads its matched hop on all 40 seeds at both weights (seed minimum +287 and +166).
  - At a = 0.5 the 1.2 m/s pair goes the hop's way at w 2.0: V35 − HF120 averages −13.2, the worst walk-seed pair is −86.3, and 97 of 120 pairs are ahead on the first panel.
- **C3 (forward_vel_max 1.5) lifts the walks at w 2.0 but not out of C1's hole.** At w 2.0, 80 of 120 walk-seed pairs are ahead of standing on the first panel and 60 of 80 on the held-out one: V05, V24, V18 and V35 lead, but FINAL3 and V01 do not.
- **C4 (gait symmetry) moves standing further ahead.**
  - At g = 0.1 it pays the statue +99.8 per episode, the hops +91.7 to +98.2 and the stable walks +70.4 to +94.3 (V18 least) [measured: column G of the same table].
  - Every g makes the standing gap and the hop margins worse, not better.

## 3. Other single knobs that price flight or reward alternation

Each knob below is applied alone on main. Same columns as above, plus what the knob does to the stable walks and to the matched hops [measured: out/comb_stdout.txt; out/vd_main_stdout.txt "What each setting does", first panel, per episode].

| setting | config | w 0.2: worst walk − standing, mean / seed min | w 2.0: seed min | walk − matched hop, seed min (w 0.2 / 2.0) | PASS | change to stable walks | change to matched hops (w 0.2) |
|---|---|---|---|---|---|---|---|
| A1 | airborne_substep_weight 1.0 | −16.9 (V24) / −30.5 | +123.6 | +445.4 / +324.0 | no (standing at w 0.2) | 0 to −2.5 | −486 to −517 |
| A05 | airborne_substep_weight 0.5 | −16.9 / −30.5 | +123.6 | +193.1 / +71.7 | no (standing at w 0.2) | 0 to −1.3 | −243 to −258 |
| LB | foot_load_balance_airborne_penalty 0.8, min_support_force 33.6 N | −17.1 / −30.5 | +123.6 | +369.6 / +252.1 | no (standing at w 0.2) | 0 to −15.3 | −404 to −454 |
| LB05 | same, penalty 0.5 | −17.0 / −30.5 | +123.6 | +212.1 / +91.9 | no (standing at w 0.2) | 0 to −9.6 | (5/8 of LB) |
| LB0 | penalty 0.8, min_support_force 0 (default) | −17.0 / −30.5 | +123.6 | +369.6 / +251.3 | no (standing at w 0.2) | 0 to −15.3 | −404 to −454 |
| FI1.4 | floor_impact_weight 2.0, threshold 1.4 BW | −192.9 (V35) / −223.1 | +123.6 | +785.6 / +674.3 | no (standing at w 0.2) | −0.1 to −212 | −967 to −1091 |
| FI2.0 | same, threshold 2.0 BW | −98.6 (V35) / −126.6 | +123.6 | +468.7 / +348.2 | no | 0 to −118 | about −514 to −672 |
| FI2.5 | same, threshold 2.5 BW | −52.2 (V35) / −79.0 | +123.6 | +242.1 / +143.9 | no | 0 to −71 | −221 to −399 |
| FI3.0 | same, threshold 3.0 BW | −20.1 (V35) / −48.0 | +123.6 | +80.1 / −23.3 | no | 0 to −38 | about −58 to −197 |
| BS | bilateral_support_weight 0.5 | −453.7 / −466.4 | −331.1 | −219.2 / −342.0 | no | +40.5 to +66.9 (statue +498.8) | +211 to +246 |
| G | gait_symmetry_weight 0.1 | −25.5 (V24) / −40.5 | +114.3 | −82.2 / −199.8 | no | +70.4 to +94.3 (statue +99.8) | +95 to +98 |
| V15 | forward_vel_max 1.5 (= SCREEN R3) | +13.1 (V24) / −2.5 | +217.4 | −64.5 / −254.0 | no | +10.0 to +46.5 (w 0.2); +99.8 to +465.0 (w 2.0) | +11 to +49 (w 0.2) |

- **The hop half has a config-only fix.**
  - `airborne_substep_weight` 0.5-1.0 alone, or `foot_load_balance_airborne_penalty` 0.8 alone (the R1 flag as a penalty, at 33.6 N or at the default 0 N), puts every walk ahead of its matched hop on all 40 seeds at both weights.
  - The margins match SCREEN's R1 and R2. On the first panel LB's seed minima are +380.8 (w 0.2) and +275.5 (w 2.0), against R1's +373.6 and +273.3. They cost the stable walks at most 2.5 (A1) or 15.3 (LB, V35's landing bounces) per episode.
- **None of these single knobs fixes standing at w 0.2.** V24 stays about 17 below standing, just as under main, because none of them touches the statue or the crouched walk.
  - Under main, V24's deficit against the statue is the height term (−50.3 per episode). Its forward pay at w 0.2 makes up only +44.9 [measured: out/vd_main_stdout.txt, "V24 minus statue"].
- **`floor_impact_weight` prices landings, and the fast walks land hard too.**
  - Its p95 step peak is 2.10-2.12 BW for V18 and V35, against 3.3-4.1 BW for the hops.
  - So any threshold that charges the hops enough also costs V35 38-212 per episode, and V35 drops below standing at w 0.2. At 3.0 BW the hop already wins 3 of 180 walk-seed pairs at w 2.0 (worst −23.3).
- **`bilateral_support_weight` and `gait_symmetry_weight` pay standing more than walking.** They work against both halves of the test.

## 4. Config-only settings that pass, beside R6 and R15

Below, over both panels: "seed min" is the worst walk-seed pair over 40 seeds (first-panel walks on 3042-3061, held-out walks on 3062-3081). The last column is the change in the six stable walks' mean return against main, on the first panel [measured: out/comb_stdout.txt and out/comb_all_stdout.txt; SCREEN's R-settings from its screen_*.json].

| setting | config (HYPOTHETICAL; rest = main) | PASS (40 seeds, w 0.2 and 2.0) | w 0.2: worst walk − standing, mean (walk) / seed min | w 2.0: − standing, seed min | − matched hop, seed min (w 0.2 / 2.0) | 0.5-1.2 m/s walks > H20 on all 40 seeds, both w | − H20, seed min (w 0.2 / 2.0) | change to stable walks vs main (w 0.2; w 2.0) |
|---|---|---|---|---|---|---|---|---|
| X24 | A1 + forward_vel_max 1.5 | no (one held-out seed) | +13.1 (V24) / −2.5 | +217.4 | +440.1 / +265.4 | no | +442.9 / −507.9 | +10.0 to +44.0; +99.8 to +462.5 |
| X4 | LB + A1 + forward_vel_max 1.5 | no (one held-out seed) | +12.8 (V24) / −2.5 | +217.4 | +872.5 / +692.6 | no | +878.9 / −43.1 | +10.0 to +30.4; +99.8 to +447.2 |
| X24v1.4 | A1 + forward_vel_max 1.4 (extra) | YES | +16.1 (FINAL3) / +2.5 | +234.1 | +440.7 / +265.6 | no | +446.8 / −494.9 | +11.8 to +51.1; +117.7 to +533.4 |
| X24v1.3 | A1 + forward_vel_max 1.3 (extra) | YES | +18.2 (FINAL3) / +8.2 | +253.4 | +442.3 / +278.5 | no | +451.7 / −472.6 | +13.8 to +58.4; +138.2 to +606.4 |
| X24v12 | A1 + forward_vel_max 1.2 (extra) | YES | +20.6 (FINAL3) / +12.0 | +276.0 | +445.2 / +307.7 | no | +457.9 / −439.0 | +16.2 to +65.7; +162.2 to +680.1 |
| A05v12 | A05 + forward_vel_max 1.2 (extra) | YES | +20.6 (FINAL3) / +12.0 | +276.0 | +192.9 / +48.0 | no | +190.5 / −708.6 | +16.2 to +67.0; +162.2 to +681.3 |
| X13v12 | LB + forward_vel_max 1.2 (extra) | YES | +20.6 (FINAL3) / +12.0 | +276.0 | +364.2 / +215.5 | no | +389.5 / −513.4 | +16.2 to +55.4; +162.2 to +667.3 |
| X24v1.0 | A1 + forward_vel_max 1.0 (extra) | YES | +27.0 (FINAL3) / +18.1 | +334.6 | +448.5 / +337.4 | no | +474.7 / −334.7 | +22.5 to +81.1; +224.6 to +821.7 |
| X4v1.4 | LB + A1 + forward_vel_max 1.4 (extra) | YES | +16.1 (FINAL3) / +2.5 | +234.1 | +871.9 / +692.8 | no | +882.8 / −30.1 | +11.8 to +37.6; +117.7 to +518.1 |
| X4v1.3 | LB + A1 + forward_vel_max 1.3 (extra) | YES | +18.2 (FINAL3) / +8.2 | +253.4 | +871.2 / +705.7 | no | +887.7 / −7.8 | +13.8 to +45.6; +138.2 to +591.1 |
| **X4v12** | **LB + A1 + forward_vel_max 1.2 (pre-declared)** | **YES** | **+20.6 (FINAL3) / +12.0** | **+276.0** | **+870.4 / +734.9** | **YES** | **+893.9 / +25.8** | **+16.2 to +54.4; +162.2 to +664.7** |
| X4v1.1 | LB + A1 + forward_vel_max 1.1 (extra) | YES | +23.5 (FINAL3) / +14.9 | +302.6 | +869.5 / +758.1 | YES | +901.4 / +71.8 | +19.1 to +63.5; +190.6 to +737.5 |
| X4v1.0 | LB + A1 + forward_vel_max 1.0 (extra) | YES | +27.0 (FINAL3) / +18.1 | +334.6 | +868.5 / +757.4 | YES | +910.7 / +130.1 | +22.5 to +72.4; +224.6 to +811.8 |
| X24h0 | A1 + height_weight 0 (extra) | YES | +8.4 (FINAL3) / +0.7 | +119.1 | +469.1 / +349.3 | no | +485.5 / −338.8 | −295.6 to −249.7 (both w) |
| X24v15h15 | A1 + forward_vel_max 1.5 + height_weight 0.15 (extra) | YES | +16.5 (FINAL3) / +8.4 | +219.6 | +452.8 / +275.5 | no | +463.5 / −511.1 | −137.8 to −89.1; −48.0 to +329.4 |
| X4h15 | X4 + height_weight 0.15 (extra) | YES | +16.5 (FINAL3) / +8.4 | +219.6 | +884.2 / +702.7 | no | +899.5 / −46.3 | −137.8 to −95.3; −48.0 to +314.1 |
| X2 | LB + A1 (extra; no cap change) | no (standing at w 0.2) | −17.1 (V24) / −30.5 | +123.6 | +875.8 / +760.0 | YES | +880.2 / +108.5 | −17.8 to 0 (both w) |
| IX | idle_penalty_weight 0.05 alone (extra) | no (hops) | +23.0 (V24) / +8.3 | +162.4 | −59.2 / −180.6 | no | −80.3 / −864.1 | −0.2 (both w) |
| I1w02 | A1 + idle_penalty_weight 0.02 (extra) | no | −0.2 (V24) / −15.0 | +139.2 | +445.4 / +324.0 | no | +444.3 / −333.4 | −2.6 to −0.1 |
| I1 | A1 + idle_penalty_weight 0.05 (extra) | YES | +22.9 (V24) / +8.3 | +162.4 | +445.4 / +324.0 | no | +444.3 / −333.3 | −2.7 to −0.2 (both w) |
| IT | A1 + idle_penalty_weight 0.05, idle_velocity_threshold 0.1 (extra) | YES | +26.9 (V24) / +12.2 | +166.4 | +446.0 / +324.6 | no | +445.0 / −332.4 | −3.1 to −0.6 (both w) |
| I2 | A1 + idle_penalty_weight 0.1 (extra) | YES | +54.5 (V24) / +41.6 | +182.2 | +445.4 / +324.0 | no | +444.3 / −333.1 | −2.9 to −0.3 (both w) |
| I3w02 | X4 + idle_penalty_weight 0.02 (extra) | YES | +29.5 (V24) / +13.0 | +233.1 | +872.5 / +692.7 | no | +878.9 / −43.1 | +9.9 to +30.4; +99.8 to +447.1 |
| I3 | X4 + idle_penalty_weight 0.05 (extra) | YES | +52.6 (V24) / +36.2 | +256.3 | +872.5 / +692.7 | no | +878.9 / −43.1 | +9.8 to +30.3; +99.7 to +447.0 |
| I5 | LB + A1 + idle_penalty_weight 0.05 (extra) | YES | +22.7 (V24) / +8.3 | +162.4 | +875.8 / +760.0 | YES | +880.3 / +108.6 | −18.0 to −0.2 (both w) |
| **I6** | **LB + A1 + idle_penalty_weight 0.1 (extra; cap stays 2.5)** | **YES** | **+54.2 (V24) / +41.6** | **+182.2** | **+875.9 / +760.0** | **YES** | **+880.3 / +108.6** | **−18.2 to −0.3 (both w)** |
| R13 (SCREEN, code) | R1 gate + forward_vel_max 1.5 | no (one held-out seed) | +12.9 (V24) / −2.5 | +217.4 | +358.9 / +169.9 | no | +368.4 / −588.6 | +10.0 to +31.8; +99.8 to +450.3 |
| R4 (SCREEN, code) | R1 gate + airborne 1.0 + forward_vel_max 1.5 | no (one held-out seed) | +12.8 (V24) / −2.5 | +217.4 | +865.1 / +689.3 | no | +872.8 / −49.4 | +10.0 to +30.7; +99.8 to +447.8 |
| **R15 (SCREEN, code)** | **R1 gate + R5 event reward** | **YES** | **+71.2 (V24) / +59.0** | **+213.5** | **+451.8 / +337.5** | **no** | **+457.0 / −316.1** | **+72.0 to +91.8 (both w)** |
| **R6 (SCREEN, code)** | **R4 + R5** | **YES** | **+101.1 (V24) / +87.0** | **+307.3** | **+954.6 / +778.0** | **YES** | **+960.0 / +41.7** | **+100.1 to +117.8; +190.0 to +534.6** |

- **The matched-hop half needs only one existing knob.**
  - Either `airborne_substep_weight` 1.0 (A1) or `foot_load_balance_airborne_penalty` 0.8 with `foot_load_balance_min_support_force` 33.6 N (LB) is enough.
  - A1's seed minima (+445 at w 0.2, +308 at w 2.0 in X24v12) are R15's (+452, +338). LB + A1 (X4v12: +870, +735) comes close to R6 (+955, +778).
  - `airborne_substep_weight` 0.5 also passes, by a smaller margin (+48 at w 2.0).
- **The standing half has two config-only levers.** A1 and LB do not move it: they do not touch the statue or the slow walks.
  - **`idle_penalty_weight`** (an extra here, since it prices standing still rather than flight) is the stronger lever.
    - It costs the statue 772 × w_idle per episode, the parent 920 × w_idle, stepping in place (W02) 51 × w_idle, the walks 3-4 × w_idle (the first 10 steps) and the hops 4-26 × w_idle [measured: out/idle_units_stdout.txt].
    - At 0.1 (I2, I6) the worst seed at w 0.2 is +41.6; at 0.05 it is +8.3; at 0.02 it fails (−15.0).
    - It does not change the cap, so nothing changes at the gate's speed.
  - **`forward_vel_max` at or below 1.4.** The worst seed is +2.5 at 1.4, +8.2 at 1.3, +12.0 at 1.2 and +18.1 at 1.0.
    - At 1.5 one held-out seed of V24 stays at −2.5, which is why X24, X4, R4 and R13 fail.
    - Below the cap the forward pay per m/s is steeper, but nothing is paid above it (see Limits).
  - `height_weight` 0 also passes, but only just (worst seed +0.7).
  - The best config-only standing margin is +41.6 (I2, I6), against +59.0 for R15 and +87.0 for R6. The idle term cannot push it much further.
    - Once w_idle is above about 0.09 [inferred: from the means], W02 (stepping in place, which pays only 51 units) is the best non-walk on most seeds.
    - So at 0.1 the margin is already V24 − W02 [measured: the per-behaviour changes in out/vd_main_all_stdout.txt: statue −77.2, parent −91.9, W02 −5.1 at 0.1].
    - R15 and R6 get theirs from the event reward, which pays the walks 72-92 per episode and W02 8.
- **Against the 2.0 m/s hop**, two kinds of setting pass on all 40 seeds:
  - LB + A1 with no cap change (X2, I5, I6): worst seed +108.6 at w 2.0.
  - LB + A1 + a cap of 1.2 or below (X4v12: +25.8).
  
  R6 has +41.7. Lowering the cap to 1.3-1.5 hurts this test [inferred: from the forward formula]. Any cap at or below 2.0 raises H20's forward pay from 0.8 w to w per step, but it raises a 0.5 m/s walk's pay by more only when the cap is below 1.25. A1 alone leaves V05 or V24 below H20 on some seeds. As under R6, FINAL3 and V01 (0.2-0.3 m/s) stay below H20 at w 2.0 on most seeds: under I6, FINAL3 has −15.6 (2 of 20 seeds ahead) and V01 +43.4 (17 of 20).
- **In discounted terms** (γ = 0.995 from reset, first panel), the config-only margin over standing at w 0.2 is small [measured: out/disc_stdout.txt].
  - Under X4v12, V24 has 217.4 against the parent's 217.3 and the statue's 216.8.
  - Under I6, V24 has 212.0 against W02's 206.5 and the statue's 208.2. FINAL3-V35 have 212.0-218.9.
  - Under main, V24 has 212.3 against 217.5.
  - SCREEN measured R6's stable walks at 227.3-234.4 against the statue's 216.8.
- **What the passing settings cost the walks** [measured: out/vd_main_stdout.txt, "What each setting does"].
  - The flight prices cost the stable walks little: A1 0-2.5 per episode, LB 0-15.3 (V35's landing bounces). The 1.5 m/s walk with landing bounces (V29) loses 14.8 under A1 and 48 under LB; the failed fast walk (V33) loses 29 and 54.
  - The cap raises every walk's return, by +16 to +68 at w 0.2 and +162 to +683 at w 2.0. But it pays nothing for speed above the cap (see Limits).
  - Setting `height_weight` to 0 or 0.15 takes 125-296 from every walk and 150-300 from the statue. It pays a crouch the same as an upright stance.
- **The failed fast walk** V33 (falls on 17 of 20 seeds) ranks 5th of 21 at w 2.0 under X4v12 (1853.9, above V05 and V24), as it does under R6 (5th, 1797.9). None of these settings deals with it; that is the gate's job (SCREEN: G1/G2 on full-length episodes).

## 5. Validation and reproduction

- **Main reproduced exactly from the recorded terms** [measured: out/cfg_main_stdout.txt, out/cfg_HO_stdout.txt].
  - Rebuilding the reward in the env's summation order gives 643,346 of 643,346 steps and 660 of 660 returns bit-identical to what `env.step` returned: 407,219 steps and 420 episodes on the first panel, 236,127 and 240 on the held-out one.
  - The same holds for the general formulas evaluated at main's values: the alive term with f = 0, the height term at 0.3, and q recomputed from r and l (difference 0.0).
  - Main's returns at both weights equal SCREEN's on every episode (out/vd_*_stdout.txt, first line).
- **The pre-declared settings' numbers did not move when the extras were added.** 38,280 returns were compared and 0 differ [measured: out/v1_vs_v2_check.txt].
- **Re-roll under main.** The statue, V35 and H20 on seed 3042 were re-rolled through `build_env` with no overrides. Rewards are bitwise equal to SCREEN's record, and the trajectories and actions are identical [measured: out/val_main_stdout.txt].
- **Re-roll under one combined HYPOTHETICAL override.** The dict, set in memory only, was: `support_conditioned_alive_fraction` 1.0, `airborne_substep_weight` 1.0, `forward_vel_max` 1.5, `gait_symmetry_weight` 0.1, `foot_load_balance_airborne_penalty` 0.8, `foot_load_balance_min_support_force` 33.636, `floor_impact_weight` 2.0 at `floor_impact_threshold_bw` 2.5, `bilateral_support_weight` 0.5, `height_weight` 0.15 and `idle_penalty_weight` 0.05. [measured: out/val_HYP_stdout.txt]
  - It ran the statue, W02, FINAL3, V18, V35, H24, HF120 and H20 on seeds 3042-3043.
  - The env's per-step reward equals the offline recomputation bitwise on 16,000 of 16,000 steps.
  - Trajectories and actions are identical to SCREEN's.
  - The `alternation_ratio` replayed from the recorded r and l equals the env's own info key on all 19,000 re-rolled steps.
  - Offline I use 0.04 × 840.9132 N = 33.636528 N. No recorded step has r + l between that and 33.636 [measured: out/bw_band_check.txt].
- **Lite re-roll of all 660 episodes under main** (for the idle term's |v|). All 660 episodes (420 + 240) were re-rolled one process at a time, 06:35-07:04 UTC [measured: out/idle_main_stdout.txt, out/idle_HO_stdout.txt].
  - Rewards and forward_vel are bitwise equal to SCREEN's records on 660 of 660 episodes.
  - The replayed `alternation_ratio` equals the env's info key on 643,346 of 643,346 steps.

## Limits

- **Fixed behaviours, not training** (as in SCREEN). The screen says which of these 21 behaviours a setting ranks higher. It does not say what PPO would find under it. A shuffle, a stomp, a staggered hop or a deeper crouch was not tested.
- **The config-only standing margins at w 0.2 are thinner than R15's and R6's** (worst seed +41.6 under I6 and +2.5 to +18 under the caps, against +59 for R15 and +87 for R6). [inferred]
  - The cap variants owe theirs to the steeper forward slope below the cap, so a walk that starts up more slowly, or crouches more than V24, could lose it.
  - The idle variants owe theirs to charging stillness. Stepping in place pays almost none of that charge (W02: 51 units against the statue's 772), and a learned sway or fidget that keeps |v| ≥ 0.05 m/s would pay none of it.
  - Under I6 the best non-walk is already stepping in place, at 54 below V24 on average.
- **A cap at or below 1.4 m/s caps the incentive below the curriculum gate's needs.** [inferred: from the gate keys in locomotion.toml and SCREEN's gate table]
  - The locomotion gate needs an episode-mean forward_vel of at least 1.0 m/s. SCREEN measured that a 1.17 m/s cruise (V35) averages 0.884 over an episode and a 1.34 m/s cruise (V29) 1.03.
  - Under `forward_vel_max` 1.2 the reward pays nothing for speed above 1.2, so it stops paying before the gate's speed.
  - At 1.4 it pays up to about V29's cruise.
  - At 1.5, the value SCREEN tested, the held-out seed fails.
- **Panels.** The held-out panel has FINAL3, V24, V18 and V35, not V01, V05 or the 0.2-0.5 m/s hops. So the test against H20 on that panel covers V24-V35 only, and 0.5 m/s walk V05 is the one that fails it on the first panel for caps of 1.3 and above.
- **One controller family per gait.** The matched hops are SCREEN's F hops (4.5-5.2 Hz, about half of all substeps airborne). A shorter-flight hop would lose proportionally less to `airborne_substep_weight` and to the LB penalty. [inferred]
- **The idle extra** reads |v| from a re-roll (the speed is not in SCREEN's records). Its threshold is the default 0.05 m/s except in IT (0.1 m/s, which helps by +4 on the worst seed).
- **Every setting here moves the statue's return.** I6 moves it by −77, and the caps by a few units at most (−2.2 at cap 1.2, w 2.0). locomotion.toml's own rule then asks for `collapse_peak_floor_reference` (1092.6) to be re-measured. [inferred: from its comment]

## Slips

1. **Post-hoc extras.** The vmax 1.0-1.4 scan with A1 or LB + A1, the `height_weight` extras and the idle settings were added after the first verdict tables (out/vd_*_stdout_v1.txt). Each was declared in CMDLOG before its numbers were computed, and each is marked "extra" in the tables.
   - The cap 1.2 setting X4v12 is pre-declared.
   - The idle grid I1w02, I3w02 and I5 was added after I1-IT were declared, but before any |v| was read.
   - X2 (LB + A1 alone) and I6 (LB + A1 + idle 0.1) were declared after the first-panel idle verdicts had been read, and before they were computed. I6, the setting I recommend, is therefore post hoc. It was then checked on the held-out panel, where it passes.
2. **Clock times typed inside two NOTE texts ran ahead of the real clock.** A later NOTE corrects them. The line timestamps are authoritative.
3. **Some read-only checks ran outside `run.sh`** (`cat` or `tail` of the batch console, `date`, `grep` and `cut` of CMDLOG and of out/vd_HO_stdout.txt). Each is listed in a later NOTE. One of them used a foreground `sleep 1`.
4. **File-tool writes are NOTE lines, not literal commands.** These are the file-write and edit-tool changes to my own scripts (cfg.py, vd.py, val.py, idle.py, comb.py, satdiag.py) and to RESULT.md. The python and sed text edits are logged literally.
5. **Analysis alongside simulation.** vd.py, comb.py, satdiag.py and the band check ran while the lite re-roll batch was running. They only read files. No two simulations ever ran at once.
6. **The background waits.** The batch ran under nice -n 10 through run.sh. The shell tool's own background wrapper, a monitor and two foreground wait loops (`until grep ...; sleep 5`) did not; they only read the console file and are listed in NOTE lines.
7. **One text edit lost a word.** A python `-c` edit of RESULT.md was inside a double-quoted shell string, so the backquoted `alternation_ratio` was taken as a command (bash printed "command not found") and dropped. I restored the word with the edit tool. It is logged literally.
8. **Draft numbers corrected.** Several figures in the RESULT.md drafts were wrong when first written (single-support share, flag agreement at m = 0, the C2h, C3 and FI wording, the matched-hop impact range, the I6 hop range). They were corrected by logged `sed` edits before this final version.

## Files

- `CMDLOG.txt`: literal commands with UTC times, plus NOTE lines (pre-declarations, file-tool writes, after-the-fact notes).
- Scripts (`scripts/`):
  - `run.sh`: the logging wrapper. `batch_lite.sh`: the sequential re-roll queue.
  - `cfg.py`: the offline reproduction and every setting. `cfg_v1.py` is its first version, before the extras.
  - `vd.py`: per-panel verdicts and tables. `comb.py`: the table over both panels. `idle.py`: the idle extra and the lite re-roll check. `val.py`: the re-roll validation. `satdiag.py`: C1 under lower saturation forces. `disc.py`: the discounted-return readout.
  - `reroll.py` and `reroll_lite.py`: SCREEN's `runs.py` with an in-memory override dict (diffs in CMDLOG).
  - `runs.py`, `hoplib.py`, `wlib.py`, `wlib2.py`, `defaults.py`: unchanged copies from SCREEN (sha256 in `out/copied_sha256.txt`).
- Outputs (`out/`):
  - `cfg_main.json` and `cfg_HO.json` (every return), with `_v1` copies from before the extras. `cfg_*_all.json` adds the idle settings.
  - `vd_main_stdout.txt` and `vd_HO_stdout.txt`: verdict tables, mean returns with ranks, per-walk margins, change against main, diagnostics. `vd_*_v1` copies are from before the extras, and `vd_*_all*` include the idle settings.
  - `comb_stdout.txt` and `comb_all_stdout.txt`: the headline table over both panels.
  - `val_main_stdout.txt`, `val_HYP_stdout.txt`, `idle_*_stdout.txt`: the validations.
  - `satdiag_stdout.txt`, `bw_band_check.txt`, `v1_vs_v2_check.txt`, `disc_stdout.txt`, `idle_units_stdout.txt`.
  - `reroll/`: the re-rolled episodes, small npz files.
