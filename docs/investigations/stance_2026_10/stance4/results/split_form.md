# Where did the left-right split go? stance4 against the stance3 controls

Generated 2026-10-20 06:01 UTC by split_form_report.py from split_form.json (split_form_data.py). Report only. Stored files were read; nothing was rolled out or trained. Nothing here changes the stance4 criteria (SP/foot/stance4/criteria.txt, sha256 978a93da..., re-checked) or any stance4 verdict. Fails stay fails.

"Measured" means read or computed from stored files. "Inferred" is marked. SP = scratchpad.

stance4 J checkpoints with judge files: 4.19M; with diagnostic panels: 4.19M. The controls have judge files and panels at all six J.

## Bottom line

1. **Measured: at 4.19M the stance4 feet are almost in line.** Window stagger sd 4.8 mm (s52), 2.3 mm (s60), 1.2 mm (s42). The zero-action statue has 2.5 mm. Control Run 2 had 11.9 mm at the same checkpoint.
2. **Measured: the left-right CoP split did not shrink.** CoP split sd 0.99 (s52), 1.09 (s60), 0.48 (s42). The controls at 4.19M: Run 2 1.05, Run 3 0.34. K2a 12, 11, 0 against 10 and 2. Seed for seed, s60 matches its control and s52 is about three times its control.
3. **Measured: per mm of stagger, the stance4 split is 3.2x, 7.3x and 6.4x what the statue's flat-sole geometry gives** (0.064 CoP units per mm, corr -0.996). Where the controls' stagger was measured the ratio is 1.39 (Run 2 4.19M), 1.26 (Run 3 6.29M) and 0.51 (Run 3 8.39M). The seven continuation runs of the replicate study have 0.39-1.29.
4. **The inference holds, with one correction.** In stance4 (s52, s60) the stagger is at or near the statue's level and the split stays, so the split is now a per-foot loading split (measured). In the controls the split is carried mostly by the positional stagger itself, at about the statue's geometric coupling; a loading part beyond the geometry is modest early (ratio 1.3-1.4) and reversed at Run 3 8.39M (0.51), where the split is smaller than the stagger alone would give (measured). So "both" is right for the controls only in a weak sense: the stagger is the main carrier there. s42 is in the saturation habit; its split is half the size and it passes K2a.
5. **Measured: the drivers did not change.** In both arms the split is mostly a function of the spawn's leg asymmetry and the prey bearing. A two-factor fit (spawn stagger, prey bearing) explains R2 0.91, 0.89, 0.90 of the stance4 split at 4.19M and 0.86 of Run 2's. Mirrored legs with the prey ahead removes it in every run at 4.19M (K2a 0, 0, 0, 1, 0; split sd 0.15, 0.11, 0.16, 0.07, 0.03; order Run 2, Run 3, s52, s60, s42).
6. **Measured: in s60 and s42 the policy over-corrects the spawn stagger.** Their window stagger runs against the spawn stagger (corr -0.56 and -0.31; statue +0.85, s52 +0.57, Run 2 +0.74). The split still follows the small residual stagger's sign: the heel-loaded foot is the forward foot in 37, 36 and 26 of 40 (Run 2 4.19M: 36). So in s60 and s42 the spawn-to-split sign is the opposite of the statue's. Inferred: the left-right twist mode is still there; the term shrank its foot displacement, and the ankle-torque asymmetry that sets the CoP stayed.
7. **Measured, training evals:** in stance4 the eval split rises first (s52 from about 1.2M, s60 from about 2.5M) and the pad rail follows (first flags at 1.57M and 2.49M). Neither run has a saturation flag. s42 has saturation flags from 1.83M (the pre-registered alarm from 2.10M) and few pad flags. On the same eval spawns, seed 52's control (Run 3) had one pad flag in 36 evaluations (at 4.59M); seed 60's control (Run 2) had pad flags from 1.70M and saturation (>= 5 of 10) from 2.23M. In s52 the eval split is held at nearly equal ankle angles (l - r ankle sd 0.10-0.23 deg from 2M, against 0.33-0.66 deg in the controls).
8. **Implication (inferred):** the stagger term removed the carrier it measures, not the split. The next term should act on what remains, the per-foot CoP. The CoP-band prototype's band [-0.6, +0.4] has either foot's window mean outside it in 20, 22 and 11 of 40 stance4 episodes at 4.19M (Run 3 at 4.19M: 0). Training under mirrored spawns with the prey ahead is a diagnosis, not a fix, and it changes the judged test if it goes into the task's [env]. Per-foot load is already observable. The v6b spring mainly resists a common ankle opening, which is not what fails at 4.19M. Section 7 has the reasoning.

## 1. Sources and definitions

- Judge files: SP/foot/stance3/judged/<run>/judge_<J>.json (controls) and SP/foot/stance4/judged/<run>/judge_<J>.json (stance4). K2a and K2b are the judge's counts. The per-foot whole-foot CoP (foot_cop_fore_aft r, l; window steps 200-999, mean over loaded steps) gives the split (r - l), the common CoP ((r + l) / 2) and the heel min (lowest foot CoP over the full-horizon episodes). These are the K2b statistic's own numbers, in span half-lengths (-1 heel line, +1 toe line).
- Panel files: SP/foot/stance4/panels/out/<run>/panels_<J>.json (panels4.py). Variants: standard (stance4 at every J; controls only at Run 3 8.39M), mirror_legs, prey_ahead, mirror_prey_ahead. The window stagger is the right foot site minus the left, along the pelvis heading, window mean, mm (+ = right foot ahead).
- The controls' standard-spawn stagger is not in their judge files, and panels4.py rolled the standard variant for the controls only at Run 3 8.39M. It is taken from last night's stance3 diagnosis rollouts (SP/foot/stance3/night/diag_split/tables/per_episode.csv: Run 2 4.19M, Run 3 6.29M and 8.39M). At every other control J it is not measured (n/m).
- Checks (measured): the judge-derived split sd and common CoP equal the panel or diag_split values exactly wherever both exist; the panel's standard K2a and K2b equal the judge's; the panel's model sha256 equals the judge's; my recompute of each variant's split statistics from its episodes equals panels4.py's own numbers.
- Statue coupling: the zero-action statue on the standard spawn (seeds 3042-3081, n 39) gives split = -0.0639 x stagger (per mm), R2 0.993. "Ratio to statue coupling" = split sd / (0.0639 x stagger sd). Near 1: the split is what the foot positions alone produce. Above 1: extra loading split. Below 1: the policy loads against the geometry.
- Eval proxy for the stagger: eval_episodes.csv has no stagger, but it has the window-mean l - r hip pitch, knee and ankle angles. analyze_split.py's linear FK map (stagger = -(0.863 dHip + 0.5453 dKnee + 0.2138 dAnk), rad, r - l) fits the diag_split rollouts with slope 1.0008, R2 0.99999 (n 319, measured). Applied to eval rows it is a proxy (inferred), on 10 eval episodes per evaluation, not the judge panel. The eval spawn columns are identical between each control and the stance4 run of the same seed in every shared evaluation (s52: 340 of 340, s60: 350 of 350; measured), so those comparisons are on the same spawns.

## 2. Standard spawn, per run and J checkpoint

| run | J | judge (clean/40) | K2a | K2b | window stagger sd (mm) | CoP split sd | common CoP | heel min | ratio to statue coupling | heel foot = forward foot | either foot outside [-0.6, +0.4] | stagger source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Run 2 (control, s60) | 4.19M | FAIL (0) | 10 | 14 | 11.9 | 1.05 | +0.03 | -0.90 | 1.39 | 36/40 | 21/40 | diag_split |
| Run 2 (control, s60) | 5.24M | FAIL (0) | 6 | 10 | n/m | 0.92 | -0.02 | -0.81 | n/m | n/m | 16/40 | not measured |
| Run 2 (control, s60) | 6.29M | FAIL (36) | 2 | 6 | n/m | 0.82 | -0.07 | -0.87 | n/m | n/m | 13/40 | not measured |
| Run 2 (control, s60) | 8.39M | PASS (39) | 1 | 0 | n/m | 0.52 | -0.13 | -0.87 | n/m | n/m | 2/40 | not measured |
| Run 2 (control, s60) | 10.49M | PASS (39) | 0 | 0 | n/m | 0.44 | -0.05 | -0.53 | n/m | n/m | 1/40 | not measured |
| Run 2 (control, s60) | final | PASS (40) | 0 | 0 | n/m | 0.40 | -0.10 | -0.53 | n/m | n/m | 1/40 | not measured |
| Run 3 (control, s52) | 4.19M | FAIL (38) | 2 | 2 | n/m | 0.34 | -0.17 | -0.55 | n/m | n/m | 0/40 | not measured |
| Run 3 (control, s52) | 5.24M | PASS (40) | 0 | 0 | n/m | 0.53 | -0.08 | -0.59 | n/m | n/m | 2/40 | not measured |
| Run 3 (control, s52) | 6.29M | PASS (40) | 0 | 0 | 7.2 | 0.58 | -0.11 | -0.78 | 1.26 | 38/40 | 4/40 | diag_split |
| Run 3 (control, s52) | 8.39M | FAIL (34) | 6 | 3 | 20.9 | 0.68 | -0.25 | -0.97 | 0.51 | 39/40 | 15/40 | panel |
| Run 3 (control, s52) | 10.49M | PASS (40) | 0 | 0 | n/m | 0.48 | -0.04 | -0.74 | n/m | n/m | 3/40 | not measured |
| Run 3 (control, s52) | final | PASS (39) | 1 | 4 | n/m | 0.46 | +0.18 | -0.52 | n/m | n/m | 16/40 | not measured |
| stance4 s52 | 4.19M | FAIL (28) | 12 | 16 | 4.8 | 0.99 | +0.07 | -0.84 | 3.22 | 37/40 | 20/40 | panel |
| stance4 s52 | 5.24M, 6.29M, 8.39M, 10.49M, final | not yet judged at writing | | | | | | | | | | |
| stance4 s60 | 4.19M | FAIL (29) | 11 | 16 | 2.3 | 1.09 | -0.01 | -0.92 | 7.30 | 36/40 | 22/40 | panel |
| stance4 s60 | 5.24M, 6.29M, 8.39M, 10.49M, final | not yet judged at writing | | | | | | | | | | |
| stance4 s42 | 4.19M | FAIL (0) | 0 | 2 | 1.2 | 0.48 | +0.11 | -0.33 | 6.37 | 26/40 | 11/40 | panel |
| stance4 s42 | 5.24M, 6.29M, 8.39M, 10.49M, final | not yet judged at writing | | | | | | | | | | |
| statue (zero action) | - | - | 1 | 1 | 2.5 | 0.16 | | | 1.00 (reference) | | | diag_split |

Read across (measured): the controls' split sd ranges 0.34-1.05 over J and is lower at the last two J in both; the stance4 split at 4.19M is at the top of that range in s52 and s60. The common CoP of stance4 is near zero (+0.07, -0.01, +0.11), so the s52 and s60 K2a fails come from the split alone, as Run 2's did at 4.19M (common +0.03). Run 3's 8.39M failure needed a common backward shift (-0.25) on top of its split, as the stance3 diagnosis found.

## 3. Mirrored legs and prey ahead, per run and J

| run | J | mirror + prey: K2a | split sd | stagger sd (mm) | ratio to statue coupling | mirror only: K2a | split sd | prey ahead only: K2a | split sd |
|---|---|---|---|---|---|---|---|---|---|
| Run 2 (control, s60) | 4.19M | 0 | 0.15 | 2.5 | 0.97 | 1 | 0.65 | 4 | 0.84 |
| Run 2 (control, s60) | 5.24M | 0 | 0.10 | 1.7 | 0.92 | 0 | 0.54 | 2 | 0.77 |
| Run 2 (control, s60) | 6.29M | 0 | 0.09 | 2.6 | 0.52 | 0 | 0.52 | 0 | 0.59 |
| Run 2 (control, s60) | 8.39M | 0 | 0.12 | 1.4 | 1.38 | 0 | 0.38 | 0 | 0.40 |
| Run 2 (control, s60) | 10.49M | 0 | 0.07 | 5.8 | 0.18 | 0 | 0.29 | 0 | 0.30 |
| Run 2 (control, s60) | final | 0 | 0.08 | 1.8 | 0.66 | 0 | 0.28 | 0 | 0.30 |
| Run 3 (control, s52) | 4.19M | 0 | 0.11 | 4.4 | 0.39 | 0 | 0.24 | 2 | 0.25 |
| Run 3 (control, s52) | 5.24M | 0 | 0.06 | 3.1 | 0.31 | 0 | 0.26 | 0 | 0.47 |
| Run 3 (control, s52) | 6.29M | 0 | 0.25 | 5.1 | 0.76 | 0 | 0.35 | 0 | 0.52 |
| Run 3 (control, s52) | 8.39M | 1 | 0.49 | 15.0 | 0.51 | 3 | 0.55 | 4 | 0.64 |
| Run 3 (control, s52) | 10.49M | 0 | 0.49 | 11.6 | 0.66 | 0 | 0.47 | 0 | 0.42 |
| Run 3 (control, s52) | final | 0 | 0.42 | 11.8 | 0.55 | 0 | 0.43 | 1 | 0.46 |
| stance4 s52 | 4.19M | 0 | 0.16 | 2.7 | 0.95 | 4 | 0.71 | 2 | 0.70 |
| stance4 s60 | 4.19M | 1 | 0.07 | 0.7 | 1.75 | 2 | 0.72 | 2 | 0.70 |
| stance4 s42 | 4.19M | 0 | 0.03 | 0.5 | 0.89 | 0 | 0.37 | 0 | 0.30 |

Read across (measured): with both asymmetries removed, K2a is 0-1 and the split sd is 0.03-0.25 in every run and J, except Run 3 from 8.39M on (split sd 0.42-0.49 with a 12-15 mm self-made stagger; K2b 5 at final). Each single change leaves about two thirds of the stance4 split (split sd 0.70-0.72 for s52 and s60 with either change alone, against 0.99-1.09 standard). Run 3's late split does not need the spawn asymmetry; stance4 at 4.19M shows no such split.

## 4. Test of the inference

The inference from the panels was: "in the controls the split shows as both a positional stagger and a loading split; in stance4 the stagger is near zero but the loading split remains".

### 4.1 How the split is carried

| set | stagger sd (mm) | split sd | split expected from the stagger at the statue coupling | ratio | residual split sd | corr(split, stagger) |
|---|---|---|---|---|---|---|
| Run 2 4.19M (control) | 11.9 | 1.05 | 0.76 | 1.39 | 0.36 | -0.97 |
| Run 3 6.29M (control) | 7.2 | 0.58 | 0.46 | 1.26 | 0.19 | -0.96 |
| Run 3 8.39M (control) | 20.9 | 0.68 | 1.34 | 0.51 | 0.75 | -0.93 |
| stance4 s52 4.19M | 4.8 | 0.99 | 0.31 | 3.22 | 0.72 | -0.92 |
| stance4 s60 4.19M | 2.3 | 1.09 | 0.15 | 7.30 | 0.96 | -0.89 |
| stance4 s42 4.19M | 1.2 | 0.48 | 0.08 | 6.37 | 0.45 | -0.52 |
| continuation A at 9.96M (stagger term) | 7.5 | 0.47 | 0.48 | 0.99 | 0.13 | -0.96 |
| continuation ST2 at 9.96M (stagger term) | 7.7 | 0.56 | 0.49 | 1.12 | 0.10 | -0.99 |
| continuation ST3 at 9.96M (stagger term) | 4.2 | 0.35 | 0.27 | 1.29 | 0.09 | -0.99 |
| continuation NCL at 9.96M (no change) | 25.9 | 0.94 | 1.66 | 0.57 | 0.92 | -0.89 |
| continuation NCC at 9.96M (no change) | 9.7 | 0.49 | 0.62 | 0.79 | 0.17 | -0.98 |
| continuation NC2 at 9.96M (no change) | 14.8 | 0.66 | 0.95 | 0.70 | 0.33 | -0.98 |
| continuation NC3 at 9.96M (no change) | 29.3 | 0.73 | 1.87 | 0.39 | 1.44 | -0.72 |
| statue | 2.5 | 0.16 | 0.16 | 1.00 | - | -1.00 |

The residual is the sd of (split - statue slope x stagger), from the per-episode values (the continuation rows from the readout's summary statistics). Measured readings:

- Controls: the split is about what their stagger produces through the flat-sole geometry (ratio 0.5-1.4). Early (Run 2 4.19M, Run 3 6.29M) there is a modest loading excess on top (ratio 1.3-1.4, residual 0.19-0.36). At Run 3 8.39M the policy loads against the geometry (ratio 0.51).
- Continuations from Run 3 8.39M: with the stagger term the ratio is 0.99-1.29 and the split shrank with the stagger (split sd 0.35-0.56 against 0.49-0.94). Started from an existing posture, the term reduced both.
- stance4 from scratch: the stagger is at or below the statue's own, and the split is 3-7 times what that stagger gives. Most of the split sits in the residual (0.45-0.96). This is the loading split.
- Inferred mechanism: the stance3 diagnosis found each foot's CoP set by its ankle torque over its load (corr +0.99). With the feet in line, a CoP split needs an ankle-torque difference that the geometry no longer supplies. panels4.py does not record ankle angles or torques; the eval rows give only the l - r ankle angle (section 5).

### 4.2 What drives it (standard spawn and single-change variants)

| run | J | variant | R2 spawn stagger + prey bearing | R2 spawn stagger only | R2 prey bearing only | split per mm spawn stagger | split per deg bearing | corr(window stagger, spawn stagger) |
|---|---|---|---|---|---|---|---|---|
| Run 2 (control, s60) | 4.19M | standard | 0.86 | 0.46 | 0.46 | -0.0148 | +0.126 | +0.74 |
| Run 3 (control, s52) | 6.29M | standard | 0.69 | 0.49 | 0.24 | -0.0087 | +0.049 | +0.63 |
| Run 3 (control, s52) | 8.39M | standard | 0.51 | 0.31 | 0.24 | -0.0080 | +0.058 | +0.50 |
| stance4 s52 | 4.19M | standard | 0.91 | 0.44 | 0.53 | -0.0136 | +0.128 | +0.57 |
| stance4 s60 | 4.19M | standard | 0.89 | 0.31 | 0.64 | +0.0123 | -0.157 | -0.56 |
| stance4 s42 | 4.19M | standard | 0.90 | 0.32 | 0.64 | +0.0055 | -0.069 | -0.31 |
| Run 2 (control, s60) | 4.19M | mirror only (bearing) | - | - | 0.87 | - | +0.115 | - |
| Run 2 (control, s60) | 4.19M | prey ahead only (spawn) | - | 0.78 | - | -0.0164 | - | +0.86 |
| Run 3 (control, s52) | 4.19M | mirror only (bearing) | - | - | 0.83 | - | +0.041 | - |
| Run 3 (control, s52) | 4.19M | prey ahead only (spawn) | - | 0.68 | - | -0.0052 | - | +0.71 |
| stance4 s52 | 4.19M | mirror only (bearing) | - | - | 0.90 | - | +0.127 | - |
| stance4 s52 | 4.19M | prey ahead only (spawn) | - | 0.84 | - | -0.0142 | - | +0.67 |
| stance4 s60 | 4.19M | mirror only (bearing) | - | - | 0.93 | - | -0.130 | - |
| stance4 s60 | 4.19M | prey ahead only (spawn) | - | 0.79 | - | +0.0138 | - | -0.75 |
| stance4 s42 | 4.19M | mirror only (bearing) | - | - | 0.91 | - | -0.066 | - |
| stance4 s42 | 4.19M | prey ahead only (spawn) | - | 0.75 | - | +0.0058 | - | -0.33 |
| statue | - | standard | - | 0.66 | - | (corr -0.81) | - | +0.85 |

Measured readings: at 4.19M both drivers act in both arms, with similar strength (mirror only: R2 0.83-0.93 on the bearing; prey ahead only: R2 0.68-0.85 on the spawn stagger). The sign of the map differs between runs: s60 and s42 put the heel on the other foot from s52, Run 2, Run 3 and the statue. In s60 and s42 the window stagger runs against the spawn stagger. A mirror-symmetric policy allows either sign. Inferred: the split is a learned response to the two asymmetries, not a passive one; the stagger term changed its form, not its cause.

### 4.3 Verdict on the inference

- stance4: holds for s52 and s60 (measured: stagger sd 4.8 and 2.3 mm, split sd 0.99 and 1.09, ratio 3.2 and 7.3). s42 has the same form at a smaller size (stagger 1.2 mm, split 0.48, ratio 6.4) inside its saturation habit.
- Controls: the split is carried mainly by the stagger, at about the geometric coupling, with a modest loading excess early and a counter-loading late (measured where the stagger was measured: 3 of the 12 control J points). "Both" overstates the loading part.
- Seed-matched at 4.19M (measured, n = 2 pairs; the runs diverge in training, so this is descriptive): s60 split 1.09 against its control's 1.05; s52 0.99 against 0.34. The term did not reduce the split in either pair.

## 5. Training-time trajectory (eval_episodes.csv)

Ten eval episodes per evaluation, every 131,072 steps. Bins of 1,048,576 steps (8 evaluations, 80 episodes). Pad rail = max_sole_cop_fore_aft > 0.80 (the K2a statistic on eval rows). Saturation = max_actuator_saturation_fraction in screen_v2_reasons. Split sd and the FK stagger proxy pool the bin's full-horizon episodes. All measured, the proxy inferred as above.

| run | bin end | pad rail /80 | saturation /80 | evals with >= 5 saturated | clean /80 | eval split sd | proxy stagger sd (mm) | ratio to statue coupling | l - r ankle sd (deg) | corr(split, l - r ankle) |
|---|---|---|---|---|---|---|---|---|---|---|
| Run 2 (control, s60) | 1.05M | 0 | 0 | 0 | 80 | 0.23 | 4.8 | 0.75 | 0.41 | +0.68 |
| Run 2 (control, s60) | 2.10M | 3 | 1 | 0 | 76 | 0.57 | 6.4 | 1.41 | 0.37 | +0.93 |
| Run 2 (control, s60) | 3.15M | 11 | 71 | 8 | 3 | 0.81 | 10.5 | 1.21 | 0.66 | +0.91 |
| Run 2 (control, s60) | 4.19M | 21 | 72 | 8 | 4 | 1.11 | 14.1 | 1.24 | 0.66 | +0.88 |
| Run 2 (control, s60) | 5.24M | 14 | 77 | 8 | 0 | 0.93 | 10.1 | 1.45 | 0.50 | +0.86 |
| Run 2 (control, s60) | 6.29M | 7 | 51 | 6 | 24 | 0.83 | 8.4 | 1.55 | 0.35 | +0.62 |
| Run 3 (control, s52) | 1.05M | 0 | 2 | 0 | 77 | 0.20 | 3.3 | 0.95 | 0.31 | +0.35 |
| Run 3 (control, s52) | 2.10M | 0 | 0 | 0 | 79 | 0.24 | 5.6 | 0.67 | 0.40 | +0.87 |
| Run 3 (control, s52) | 3.15M | 0 | 0 | 0 | 80 | 0.33 | 5.0 | 1.03 | 0.33 | +0.92 |
| Run 3 (control, s52) | 4.19M | 0 | 1 | 0 | 79 | 0.33 | 5.7 | 0.92 | 0.43 | +0.72 |
| Run 3 (control, s52) | 5.24M | 1 | 7 | 1 | 72 | 0.45 | 6.4 | 1.11 | 0.40 | +0.94 |
| Run 3 (control, s52) | 6.29M | 3 | 0 | 0 | 77 | 0.54 | 6.7 | 1.26 | 0.36 | +0.95 |
| stance4 s52 | 1.05M | 0 | 0 | 0 | 79 | 0.28 | 2.0 | 2.20 | 0.15 | +0.24 |
| stance4 s52 | 2.10M | 14 | 0 | 0 | 65 | 0.87 | 4.2 | 3.24 | 0.23 | -0.12 |
| stance4 s52 | 3.15M | 10 | 0 | 0 | 70 | 0.90 | 4.0 | 3.53 | 0.10 | -0.49 |
| stance4 s52 | 4.19M | 13 | 0 | 0 | 67 | 0.96 | 5.0 | 2.99 | 0.18 | -0.20 |
| stance4 s60 | 1.05M | 0 | 0 | 0 | 76 | 0.21 | 2.5 | 1.31 | 0.26 | -0.54 |
| stance4 s60 | 2.10M | 0 | 0 | 0 | 78 | 0.19 | 2.5 | 1.19 | 0.23 | -0.13 |
| stance4 s60 | 3.15M | 6 | 0 | 0 | 72 | 0.69 | 6.2 | 1.74 | 0.40 | +0.38 |
| stance4 s60 | 4.19M | 14 | 0 | 0 | 66 | 0.90 | 3.4 | 4.16 | 0.36 | -0.78 |
| stance4 s42 | 1.05M | 0 | 0 | 0 | 70 | 0.20 | 2.0 | 1.54 | 0.18 | -0.06 |
| stance4 s42 | 2.10M | 1 | 14 | 1 | 62 | 0.33 | 2.0 | 2.57 | 0.22 | -0.73 |
| stance4 s42 | 3.15M | 3 | 79 | 8 | 0 | 0.45 | 3.3 | 2.14 | 0.35 | -0.73 |
| stance4 s42 | 4.19M | 0 | 80 | 8 | 0 | 0.48 | 2.5 | 2.97 | 0.23 | -0.49 |

Per evaluation (stance4, from 131,072 steps; one number per evaluation):

- stance4 s52 to 4,456,448 steps (34 evaluations):
  - pad rail: 0 0 0 0 0 0 0 0 0 0 0 1 0 4 4 5 3 2 2 2 0 0 0 1 1 3 3 0 1 1 0 4 2 2
  - saturation: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  - saturation alarm (pre-registered: two evaluations in a row with >= 5 of 10): 0 pairs
- stance4 s60 to 4,587,520 steps (35 evaluations):
  - pad rail: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 3 0 4 2 0 1 2 1 2 2 1 1 0
  - saturation: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  - saturation alarm (pre-registered: two evaluations in a row with >= 5 of 10): 0 pairs
- stance4 s42 to 4,718,592 steps (36 evaluations):
  - pad rail: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  - saturation: 0 0 0 0 0 0 0 0 0 0 0 0 0 3 3 8 10 10 10 9 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
  - saturation alarm (pre-registered: two evaluations in a row with >= 5 of 10): 20 pairs, first flagged evaluation at 2,097,152
- Run 2 (control, s60), same spawns, to 4,718,592 steps: pad rail 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 1 0 2 2 5 1 5 2 2 2 3 2 2 3 1 5 1 0; saturation 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 9 10 9 10 9 10 6 8 9 10 9 9 7 10 8 10 10 10 10 10
- Run 3 (control, s52), same spawns, to 4,718,592 steps: pad rail 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0; saturation 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 5

Measured readings:

- s52 and s60 never show a saturation flag; their clean losses are pad-rail episodes.
- The stance4 eval split grows from about 0.2 to 0.9 while the proxy stagger stays at 2-6 mm. The ratio is 3.0-3.5 in s52 from the 2.10M bin on. In s60 it rises from 1.7 (3.15M bin, proxy 6.2 mm) to 4.2 (4.19M bin, proxy 3.4 mm): the stagger fell while the split grew. In the controls the eval ratio stays 0.7-1.6. Run 2's split grew with its stagger (proxy 10-14 mm) and with saturation.
- Ankle angles: in the controls the split tracks the l - r ankle angle (corr +0.72 to +0.93 in the 2.10M-4.19M bins; l - r sd 0.33-0.66 deg). In s52 the l - r ankle sd is 0.10-0.23 deg and the correlation is weak (-0.12 to -0.49). In s60 and s42 it has the opposite sign (-0.78 and -0.49 in the 4.19M bin). Inferred: the stance4 loading split is not the passive result of an ankle-angle difference; it comes from torque (commanded) differences at similar angles.
- The contact pattern is the same in both arms: the split goes with the r - l digit load share at corr -0.81 to -0.99 in every run in the 3.15M and 4.19M bins (the same sign everywhere).
- s42's saturation habit started at 1.83M and has not let go at 4.72M; its split stays at 0.3-0.5.

## 6. What it does not show

- One stance4 J checkpoint with panels (4.19M). The 5.24M fetch was launched at 05:46 UTC; this file is regenerated by re-running split_form_data.py then split_form_report.py when later panels land.
- n = 3 stagger runs, 2 controls, and no test. The controls are stance3's runs, not retrained (criteria, the control arm).
- The controls' standard-spawn stagger is measured at 3 of 12 J points. Elsewhere only the eval proxy covers them.
- Ankle torques are not in any file read here, and ankle angles only as eval window means (l - r). The loading mechanism is inferred from the stance3 diagnosis (Run 3 8.39M, Run 2 4.19M) and the eval ankle columns. A stance4 rollout with roll_split.py would measure it.

## 7. What it implies for the next round (inferred)

Decision line 5 names the per-foot CoP band or the v6b ankle spring. The criteria are fixed; this ranks options for the maintainer.

1. **Per-foot CoP band (scratch commit f94452ad, off by default).** It acts on the quantity that remains: each foot's whole-foot CoP, wherever the foot stands. Its band [-0.6, +0.4] sees 20, 22 and 11 of 40 stance4 episodes at 4.19M by window mean (measured; the term charges per step). A stagger-free loading split cannot pass it, unlike the stagger term. The stagger term's lesson applies: a price on one carrier moved the split to another. The band's guards (min load share 0.35, pad band [-0.65, +0.65]) close the two next carriers named in its design (load moved between feet, load moved onto the digits). It does not price which foot is forward, so a positional stagger could return; the stagger term can stay on beside it. Cost: +45% env step time (design.md). First choice.
2. **v6b ankle spring** (250 N.m/rad about 78.6 deg, plant revision). A passive spring acts on ankle angles. In s52 the split is held at nearly equal ankle angles (eval l - r ankle sd 0.10-0.23 deg, weak correlation with the split; measured), so a spring would add about the same torque to both ankles and leave the difference that sets the split. In s60 the ankle angles do differ (sd 0.36 deg in the 4.19M bin), with the opposite sign to the controls; what a spring does there cannot be read from these files. It resists offsets from its reference, mainly the common ankle opening that Run 3 8.39M needed for its K2a fails. At 4.19M the stance4 common CoP is near zero, so the spring addresses the part that is not failing now. It strands every r9 checkpoint and cannot be a continuation. Second line, held for a common-shift failure.
3. **Training under mirrored spawn noise with the prey ahead.** Measured: in the panels this removes most of the split at every J of every run (K2a 0-1), except Run 3's self-made split from 8.39M on. But it removes the drivers, not the response. The judge rolls under the branch's own stance [env] (criteria, HOW IT IS JUDGED). If the change goes into the task's [env], the judged panel changes with it: that is a change to the test, a maintainer decision, not a fix. If it is training-only, the policy meets asymmetric spawns and off-axis prey first at judging, and the noise tests (K3) perturb the legs asymmetrically. Useful as a diagnosis or as an early curriculum stage, not as the next term.
4. **Observing per-foot load.** The stance3 diagnosis found the observation already carries it: the stagger is a linear function of the observed leg angles (R2 1.00000), and each foot's CoP is linearly predictable from its MTP angles (R2 0.84-0.87; 0.965 with leg angles and touch). The policy can see the split. It keeps it because the reward allows it. Not re-measured on stance4. Last choice.

Ranking (inferred): CoP band first, with the stagger term kept on; mirrored spawn and prey ahead as a report-only diagnostic arm or a maintainer decision about the test; v6b spring held for a common-shift failure; observation last.

## 8. Files

- split_form.json: every number above, per run, J, variant and evaluation, with source sha256s.
- split_form_data.py (reads judge, panel, diag_split, eval and readout files; no project code is called) and split_form_report.py (this file).
