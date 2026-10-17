# T. rex stance run 20261011_165957: the right pad on its toe edge

2026-10-12 · main 8565d299 (T. rex physics r9, policy interface r15). Data root `scratchpad/` = `scratchpad/`. CoP is given in pad half-lengths (hl, 0.070 m), with + toe-ward. "(inference)" marks a claim not measured directly.

## 1. Answer

Run A's right pad stands on its toe edge because the policy holds a small left/right-antisymmetric leg offset (left ankle about -3.7°, right knee about +2.0°). With both feet planted, an ankle or knee offset moves the other foot's CoP by 0.19-0.23 hl per degree, so this offset uses up the pad's 28 N·m of toe-ward moment margin; the same coupling exists in both Compsognathus models, and physics r9 does not cause it. Nothing in training prices this: the stance reward is flat in CoP (statue 3755.7 per episode, edge postures 3762-3767), and at the policy's own exploration noise the statue falls in 5-20 of 20 episodes with its per-step CoP past 0.8 on 96% of steps, so neither the base reward, a per-step CoP penalty (probe 1) nor a settled leg reference (probe 2) kept 1M-step training off the edge.

## 2. What happened in run A, mechanically

Run A: seed 56, commit cdffea89, 12,001,280 steps, 17 h 23 min. The gate judged robust_best_model (eval 3770.1 ± 15.7 at 3.2M steps): 1/40 clean, LCB 0.0013.

- `max_sole_cop_fore_aft` (bar 0.80, `configs/tyrannosaurus_rex/anatomical/stance.toml:732`) fails 39/40, median 0.9999.
- Other failures: sole tilt 4/40 (worst 2.85°), corner lift 3/40 (6.0 mm), sole contacts 2/40, settle peak 1/40.
- 40/40 reach the horizon; worst drift 1.6 cm (`drive_runs/runA/*stance_gate_report.txt`).

| Per foot | Statue (zero action) | Run A checkpoint, closed loop (40 ep) | Run A mean action held open loop (20 ep) |
|---|---|---|---|
| Gate statistic (max over feet) | 0.287 median | 0.9999 median (right foot) | 0.972 |
| Right pad signed fore-aft CoP | +0.188 | not recorded per foot (1) | +1.000 |
| Left pad signed fore-aft CoP | +0.198 | not recorded (1) | +0.137 |
| Right / left sole pitch (toe-down -) | -0.06° / -0.07° | -0.67° / -0.02° | -1.01° / - |
| Right foot: pad / digits share | 82.7 / 17.3 % | 98 / 2.1 % (0.6/0.7/0.8) | 97.4 / 2.6 % |
| Left foot: pad / digits share | 83.0 / 17.0 % | 76 / 24 % | 81.1 / 18.9 % |
| Right ankle torque | 59.6 N·m | - | 83-84 N·m |
| Right pad heel / toe corner force | 147.8 / 205.5 N | - | 5.4 / 398.4 N (2) |

(1) Run A's artifacts keep only the per-episode maximum over feet, the sole pitch and the load split. The left pad is near the statue by its pitch and split (inference). (2) Seeds 3042-3046, steps 300-999.

Sources: `foot/probe-2/judge/statue.txt`, `foot/survey/md_statue.txt`, `foot/mechanism/ablate_table.txt` (row `full`), `foot/verify-measure/padx.py`, `foot/mechanism/an_moment.py`.

**The pad is a two-station lever.** In MuJoCo, box-plane contact puts the pad's floor force only at its heel corners (7.0 cm ahead of the ankle) and its toe corners (21.0 cm ahead). Every contact above 0.1 N (20,662 in the statue, 14,468 in the hold) sits at x = 0.070 or 0.210 m in the metatarsus frame, so the pad CoP reads +1 once the heel corners carry nothing. About 93% of the statue's digit load acts at the digit bases inside the pad's heel band (bases 62.0 N, tips 4.7 N), so unloading the heel also unloads the digits.

**The moment budget.** The whole-foot CoP equals the ankle moment divided by Fz, once the foot's own weight (1.76 kg, 1.34 N·m) is included. At Fz 420.5 N the pad can carry 29.3-88.2 N·m about the ankle. The statue uses 59.6 N·m (CoP 14.2 cm ahead of the ankle), which leaves 28 N·m toe-ward and 30 N·m heel-ward: 1.7-1.9° of ankle servo error at 900 + 40 N·m/rad. Under run A's held action the ankle carries 83-84 N·m and the CoP sits at 20.3-20.5 cm, at the toe-corner line of the 1° toe-down pad (20.55-20.62 cm; per seed, from 0.09 cm beyond the line to 1.9 cm short of it). This is a torque window, not a target-offset window: an ankle target offset moves its own pad by only -0.036 to -0.039 hl/deg.

**The driver is an antisymmetric leg command.** In run A's mean action, left minus right ankle is -0.063 action units and right minus left knee +0.042 (about -3.7° left ankle, +2.0° right knee). The other offsets are hip pitch +3.4/+3.5°, neck pitch -4.0° and tail_1 pitch +2.9°. Held-pose ablation, 20 seeds, ramped in from step 200 (`foot/mechanism/ablate_table.txt`):

| Held action | Right pad signed CoP | CoP > 0.8 | Clean |
|---|---|---|---|
| statue | +0.183 | 0/20 | 20/20 |
| full mean action | +1.000 | 19/20 | 0/20 |
| hip pitch only | +0.073 | 0/20 | 20/20 |
| all but the left ankle | +0.292 | 0/20 | 17/20 |
| left ankle only | +0.883 | 7/20 (12 fell) | 1/20 |
| all but neck, head and tail | +1.000 | 19/20 | 0/20 |
| mirrored mean action | left pad +0.980 | 20/20 (left) | 0/20 |

Scaling the action from x0.1 to x0.9 moves the right gate CoP linearly from 0.275 to 0.940, and CoP failures from 0 to 18/20 (`foot/mechanism/scale_mirror.jsonl`).

**Why a few degrees suffice.** With both feet planted, the legs form a closed chain. One ankle or knee target offset moves the other foot's CoP by -0.19 to -0.23 hl/deg, and its own foot's by -0.04 (ankle) or +0.11 (knee) hl/deg (10 seeds, ±1-2°, 15/160 falls; `foot/survey/sens_table.txt`). Ramped in after settling, one ankle offset brings the other pad to 0.80 at -3.1 to -3.25° for the right ankle (0-1/5 falls) and -3.5° for the left (0/5 falls) (`foot/survey/thresholds.txt`). The posture is not a sum of single-joint slopes: run A's knee and ankle offsets without its hip pitch fall in 10/20 episodes, and adding the hip offsets keeps the left pad centred (+0.051).

**When it formed.** Run A's left-right ankle DC difference (`gate_progress.npz`, bin means):

- +0.007 up to 0.35M steps;
- -0.018 at 0.35-0.65M;
- -0.032 at 0.65-0.85M;
- -0.049 at 0.85-1.0M;
- -0.048 to -0.066 from there to 12M.

The right-left knee difference went from +0.026 to +0.035-0.042. The edge posture was in place about 11M steps before the gate verdict.

**The closed loop adds feedback.** For a trained 1M policy (probe 2, settled-reference arm, seed 56), removing its leg DC moves the right pad only from 0.906 to 0.805 (10/20 still above 0.8). The open-loop holds show the DC offset is sufficient, not that it is the whole mechanism.

Ruled out:

- **Physics r9.** The same front-edge pad appeared on r8: the seed-44 left pad (`stance_gate_v2.py:117-121`), production seed 42 (0/40 clean, CoP 0.992) and pilot control seed 48 (CoP 0.990). The r9 statue is 39/40 clean.
- **The policy interface.** Run A used the 64-dim observation; the 63-dim heading-free main reproduces the edge at 1M (probe 0).
- **A T. rex-specific foot.** The Compsognathus anatomical and robot models couple the feet the same way (other foot -0.22/-0.39 and +0.22 hl/deg) and reach 0.80 at 2.0-3.9° single-ankle offsets.
- **Toe compliance.** With toes 50x stiffer (2000 N·m/rad), the statue's digits carry 28%, and the held action still puts the right pad at 0.924 (17/20 above 0.8; `foot/mechanism/scratch_toes2000.jsonl`).
- **Neck, head and tail.** Without them, 19/20 episodes are still on the edge, and on its own each moves a pad by at most 0.02 hl. The held action does shift the CoM 2.9 cm forward, as part of the leg pattern.

## 3. Why the reward allowed it

1. **No term reads CoP.** The stance reward is summed at `environments/tyrannosaurus_rex/base_env.py:1137-1172` (fall -100 at the shared `base_env.py:1358`). The foot terms read summed pad and digit touch, pad tilt (`geom_xmat`) and the foot sites. Nothing reads contact positions or the pad/digit split.
2. **The return is flat in CoP.**
   - The statue earns 3755.7 per episode, 3,081.7 of it (82%) from seven saturated terms.
   - Scaled run A action, x0.1/x0.5/x0.9: 3758.9/3766.8/3762.4, while the right CoP goes 0.275 → 0.623 → 0.940.
   - Run A's leg posture minus a hips-only posture: -3.41 ± 4.26 per episode (n = 28), for gate CoP 0.941 vs 0.218 and 1 vs 28 clean episodes (`foot/reward/pairs.txt`).
   - The terms that see the edge: foot_flatness -5.9 (its 3° tolerance, `stance.toml:131`, charges about 10% of a foot's share at 1° toe-down), foot_load_balance -4.2, nosedive -1.3.
   - They are cancelled by leg_home_pose +3.4, posture +1.9, height +1.5 and tail_home_pose +0.8.
   - On the gate panel the policy earns 3763.8 and the statue 3755.7.
3. **It has happened before.** On r8 the production edge policy (seed 42, 0/40 clean) earned 3783.39 and the passing seed 52 earned 3782.2 (`docs/investigations/STANCE_REWARD_PILOT_FINDINGS_2026_10.md` §1.2). Both 1M controls on main reproduce it now (§5).
4. **Training noise hides the posture.** PPO optimizes the stochastic policy (`log_std_init = -1.5`: σ 0.22 at the start, `stance.toml:448`; about 0.16 at 1M).
   - At the trained policies' own mean σ of 0.16 the statue falls in 5/20 episodes, and at 0.19 in 20/20.
   - With the same noise draws, the statue earns 438-2,251 less per episode than the trained policies. Without noise, the same policies earn 10.6-29.6 less than the statue (`foot/probe-2/judge/noise_s48.txt`, `noise_s56.txt`, on the settled-reference reward).
   - With the policy's noise added, the statue's per-step pad CoP is past 0.8 on 96% of steps. The per-step CoP cost is 0.951 for the statue and 0.955-0.956 for the trained edge policies (`foot/probe-1/check/noise_saturation.txt`).
   - (inference) The training return is dominated by surviving the noise, and the deterministic posture the gate judges is nearly invisible in it.
5. **The keyframe pull is real, but it is not the cause.** leg_home_pose is centred on the authored keyframe, and the statue settles with hips -2.74° and ankles +1.67° away from it (`foot/probe-2/check/check_switch.txt`). The term pays +16.0 per episode for the benign hip correction (the hips-only posture is 38/40 clean) and +3.4 for the legs' antisymmetric part. A symmetric ankle move toward the keyframe is blocked by nosedive (sym_ankle -0.75°: leg_home_pose +6.0, nosedive -28.5). Re-centring the term on the settled pose (probe 2) left the edge in place.
6. **The observation is not the binding limit (inference).** The foot_contact channels carry no CoP information (R² ≤ 0.13). The passive toe angles predict pad CoP on held-out postures (quadratic fit, rms 0.07-0.29 hl). The full body observation fits CoP within a posture (R² 0.99) but not across held-out postures (rms up to 2.5) (`foot/reward/observability.txt`).

## 4. Anatomy review per species

Statue (zero action) panels: 40 seeds, steps 200-999, physics as on main. Sources: `foot/anatomy/summary.txt`, `zones_summary.txt`, `snapshot.txt`, `geometry.txt`; `foot/survey/md_statue.txt`, `thresholds.txt`, `sens_table.txt`.

| Model | Floor contact geometry | Statue load split | Statue fore-aft CoP / bar | Edge sensitivity | Plausibility | Robot relevance |
|---|---|---|---|---|---|---|
| T. rex anatomical | box pad 14 x 9 cm, 7-21 cm ahead of the ankle; three passive digit capsules (40 N·m/rad, unactuated) | pad 82.7 %; digits II/III/IV 6.5/4.7/7.3 %; about 93 % of digit load at the bases, tips 1-3 N | 0.287 [0.10-0.48] / 0.80; stands on a lateral edge (abs outer CoP 0.954, roll ≤ 1.5°, not gated) | other foot -0.19 to -0.23 hl/deg; 0.80 at -3.1 to -3.5° on one ankle | digits cannot bear tip load (25 % on the d3 tip needs 28.6° of rotation, computed); splay 16.3° vs 51-80°, print W/L 0.61 vs 0.78-1.00 in tyrannosaurid tracks (McCrea et al. 2014); pad ends at the d2 tip, 6-9 cm short of d3's end | support = the pad only; 28 N·m toe-ward margin |
| Velociraptor anatomical | metatarsus capsule + digit III and IV capsules (servoed, kp 50 + 20 N·m/rad); digit II claw off the floor | metatarsus 28.7 %, digit III 54.1 %, digit IV 17.4 %; 35 % of load in the distal 30 % of the foot | no box sole, no CoP rail (its v2 gate uses duty and coverage) | statue falls at -3.0 / +2.9° on both ankles | functionally didactyl with a large load at the MTP region, as in deinonychosaur tracks (Xing et al. 2013: didactyl, digit II base on a large rounded MTP pad) | capsule contacts: no sole CoP to monitor |
| Compsognathus anatomical | box pad 5.6 x 3.2 cm + three digit capsules on one servoed toe joint (kp 12) | pad 41.2 %, digits 19.7/19.1/20.1 %; 23.6 % of load behind the MTP joint | 0.150 [0.143-0.155], heel side / 0.70 | other foot -0.22 (ankle), -0.39 (knee) hl/deg; 0.80 at 2.0-2.9° | three load-bearing digits fit a functionally tridactyl pes (Ostrom 1978: digit I reduced and reverted, V vestigial); pad heel corners sit 1.1 cm behind the MTP joint (inference: acceptable only as an MTP pad) | - |
| Compsognathus robot | flat box sole 10 x 9 cm, four corners | sole 100 % | 0.302, toe side / none (stance_quality/v1, no sole bars) | own foot -0.17, other +0.22 hl/deg; 0.80 at 3.0-3.9° | - | same closed-chain coupling; an edge stance would pass its current gate |
| Dibothrosuchus anatomical (quadruped) | one box per foot (fore 4.6 x 2.6 cm, hind 5.6 x 3.0 cm), no digits | 14.4 % of body weight per fore foot, 35.6 % per hind foot; 77-87 % of each foot's load behind the toe joint | 0.745 [0.714-0.773], heel side / none declared | 0.80 at -0.31° on all ankles | Early Jurassic crocodylomorph tracks (Batrachopus) are digitigrade and functionally tetradactyl (Olsen & Padian 1986, p. 261); the model loads its heel corners (inference: mismatch) | the statue sits 0.055 under a 0.80 bar, so a CoP rail would need recalibration first |
| Brachiosaurus anatomical (quadruped) | one ellipsoid per foot (semi-axes 12 x 10 x 4 cm): a single contact point | 18.7 % of body weight per fore foot, 31.1 % per hind foot | no sole metric; the fore ellipsoids pitch 26°, contact at -0.82 of the half-length, 5.2-5.4 cm ahead of the wrist (hind 7.9-8.5 cm) | not applicable | not reviewed against the literature here | point feet: no support area per foot |

What it means:

- **Plausibility.** The T. rex foot is mechanically a 14 x 9 cm plate: the pad takes 83% of the load, and the digits take theirs mostly at their bases. Tyrannosaurid tracks imply a wider, splayed tridactyl support.
  - Digit III carries the least digit load (4.7%) only because digit load sits at the bases and follows lateral pad roll. The earlier ostrich and track comparisons overstate the contrast (Corrections 5).
  - (computed, inference) Load-bearing digits out to the digit III tip (29.8 cm ahead of the ankle) would raise the toe-ward capacity from 88 to about 125 N·m at 420.5 N. That would not price the antisymmetric direction: with stiff toes the pad still reads 0.924.
  - Foot changes are anatomy proposals, separate from this failure; none is recommended as its fix.
- **Robots.** (inference, from measured pieces)
  - The failure transfers to any pair of position-servoed legs in double support. The closed chain lets a few degrees of antisymmetric offset load one foot's front edge with no visible lean. The Compsognathus robot shows the same coupling (+0.22 hl/deg).
  - CoP equals ankle moment divided by Fz (within 0.1 N·m after the foot's weight), so ankle torque from motor current, or a foot force/torque sensor, measures it directly. The current foot_contact channels do not (R² ≤ 0.13). A robot without passive toes also lacks the toe-angle channel that predicts CoP in the T. rex.
  - Box-plane contact loads only the corners, so contact-count metrics are discretised. The moment balance, and with it the CoP metric, is physical.

## 5. Probe results

Main 8565d299, T. rex anatomical stance, 1,048,576 steps, deterministic panel on seeds 3042-3081 (`foot/probe-0/runs/analysis.txt`, `foot/probe-1/results.txt`, `foot/probe-2/results.txt`). Each was re-judged independently, with identical results.

| Arm | Seed | Final: clean /40 (LCB) | Final: CoP > 0.8 (r/l) | Final: signed CoP r / l | Paired base reward - statue | 524k: clean (CoP fails) | Reward-selected robust_best: step, clean |
|---|---|---|---|---|---|---|---|
| control (main as is) | 48 | 21 (0.385) | 19 (2/17) | +0.451 / +0.780 | +6.0 ± 2.5 | 33 (6) | - |
| control | 56 | 9 (0.123) | 31 (23/7) (1) | +0.880 / +0.449 | -1.4 ± 2.4 | 33 (3) | - |
| per-step CoP margin, w 0.10, safe 0.65 | 48 | 12 (0.183) | 26 (0/26) | +0.195 / +0.918 | +4.3 ± 3.1 | 28 (7) | 350k, 34 |
| per-step CoP margin | 56 | 1 (0.001) | 38 (37/1) | +0.997 / +0.268 | +9.0 ± 2.1 | 14 (25) | 150k, 35 |
| leg_home_pose centred on the settled pose | 48 | 28 (0.560) | 8 (8/0) | +0.494 / +0.255 | +15.5 ± 2.0 | 32 (4) | 150k, 34 |
| leg_home_pose centred on the settled pose | 56 | 21 (0.385) | 18 (17/1) | +0.660 / -0.262 | +5.7 ± 3.8 | 34 (3) | 50k, 39 |
| statue | - | 39 (0.887) | 1 (0/1) | +0.188 / +0.198 | 0 | - | - |
| run A (12M, robust_best at 3.2M) | 56 | 1 (0.0013) | 39 | - | +8.1 (panel means) | - | - |

(1) Includes one episode that fell at step 192; 30 measured above 0.8. Paired reward is the mean ± SE over seeds where both reach the horizon (n = 37-39).

What they show:

- **Main reproduces run A within 1M steps on both seeds.** The side depends on the seed: seed 48 mostly left, seed 56 mostly right. On seed 56 the per-episode right-minus-left CoP difference correlates 0.90 with the knee asymmetry. The edge forms between 0.5M and 1M, so 1M screens are informative.
- **The per-step CoP margin did not help at this weight and budget** (26 vs 19 and 38 vs 31 CoP failures). Seed 56 reproduces run A's verdict exactly (1/40, LCB 0.0013).
  - Its training value saturated: the cost was flat at -0.075 per step from 0.26M on both seeds.
  - With the policy's noise added, the statue scores 0.951 on it, against 0.955-0.956 for the trained edge policies.
  - The same term cut CoP failures from 62 to 21 of 120 in the r8 pilot (3 seeds, interface r13). Why the results differ is open.
- **Centring leg_home_pose on the settled pose did not remove the edge** (8 and 18 CoP failures; gate LCB 0.560 and 0.385).
  - It had fewer failures than the control, but within the control's own seed spread (19 vs 31) and under a different training driver, so the reduction is not attributable to the arm.
  - Seed 48's edge moved from the left pad to the right.
  - On the arm's own reward, the deterministic finals earn 29.5 and 21.5 less than the statue, yet PPO kept them: with their own noise they earn 438-571 more than the statue, which falls in 5/20 episodes.
- **Reward-based checkpoint selection is not a safeguard.** At 1M it picks near-statue checkpoints (50k-350k steps, 34-39/40 clean). In run A it picked the edge policy at 3.2M.

Limits:

- two training seeds per arm;
- 1M steps, against run A's 12M;
- final checkpoints only;
- the control ran on `probe_train.py` and the arms on `train_base`, with different random streams;
- the 40 episodes of one policy are not replicates (the episode-level Fisher test between the two control seeds gives p = 0.010);
- one CoP weight and one safe fraction;
- the noise tests use 6-20 episodes and were not run on control checkpoints.

## 6. Recommendation

**Do first (no change to the plant or the task):**

1. *Telemetry (a normal PR).* At every periodic eval, record each foot's deterministic signed pad CoP, its digit share and the left-right sagittal leg DC, in `gate_progress.npz` and in the selected replay CSV, which lacks the per-foot CoP today. Run A's asymmetry was visible at 0.65-0.85M steps, about 11M steps and 16 h before its verdict. The contact decode costs 31% more time per step, so restrict it to eval episodes.
2. *Offline noise-ranking check (scratch only, about an hour on 4 CPUs).* Extend `foot/probe-1/check/noise_saturation.py`.
   - What to measure: each candidate term's per-step mean after settling, with training noise (σ 0.16 and 0.22, the same draws per seed, 20 episodes).
   - Postures: the statue, hips-only, and run A's action scaled x0.5, scaled x0.9 and held in full.
   - Candidates:
     - the per-step CoP margin (the known failure, as the method's control);
     - a CoP margin on a moving average of about 1 s of the signed pad CoP;
     - a smooth (unclipped) ankle-moment margin: ankle torque divided by Fz, against the pad span;
     - a quadratic on the left-right difference of ankle, knee and hip pitch, relative to the settled statue.
   - Why these (inference): a quadratic in a quantity that responds about linearly to the action keeps its slope under zero-mean noise (E[(μ+ε)²] = μ² + σ²); the clipped per-step CoP deficit does not.
   - Pass: the term ranks statue ≈ hips-only > x0.5 > x0.9 ≈ held run A action, with a statue-to-hold gap of at least 20 per episode (the edge postures' measured deterministic advantage is +14 to +18) and at least 5 SE. It must also cost the deterministic statue less than 5 per episode.

**Needs the maintainer's approval:**

- The reward term that passes check 2 (it changes `task_sha256`).
- Optionally, lower exploration noise (`log_std_init` -1.5 → about -2.5, σ 0.08). With σ at half the trained std, the statue stood in 6/6 episodes; at a quarter, its per-step CoP passed 0.8 on 20% of steps instead of 96%. This interacts with the D-D27 action-term calibration.
- Centring leg_home_pose on the settled pose, as an accuracy fix (the statue's term goes from 470.3 to 496.6 per episode). It has not been shown to reduce edges.
- Gate changes:
  - a noise- or disturbance-robustness criterion, so the stage certifies something the statue cannot;
  - sole rails for the Compsognathus robot;
  - a lateral CoP bar;
  - a whole-foot CoP bar if the digits become load-bearing.
- Any change to foot geometry (T. rex splay and digit length, load-bearing digits; Dibothrosuchus digits). These are anatomy proposals with their own measurements, not fixes for this failure.

**Next training run (a screen):**

- Model and stage: `ModelId("tyrannosaurus_rex", "anatomical")`, stage 1 stance on main (physics r9, interface r15).
- Budget: 1,048,576 steps per run, with a checkpoint every 131,072 steps.
- Arms: the control and the candidate that passed check 2 (optionally the lower-σ arm), both on the same driver (`train_base`).
- Training seeds: 42, 48, 52, 56 and 60. Seeds 48 and 56 are known to reach the left and right edge on main at 1M.
- Judging: each seed's 524k and final checkpoints with stance_quality/v2 on seeds 3042-3081, plus the noise-survival test.
- Cost on this 4-CPU machine: about 39 min per run alone, about 4 h for 10 runs two at a time, plus 76 s per panel.

**What counts as success:**

- The control reproduces the edge in the same batch (at least 10 CoP failures on at least 2 of 5 seeds); otherwise the batch is uninformative.
- The candidate, at the final checkpoint:
  - at most 3/40 CoP failures on at least 4 of 5 seeds;
  - at least 37/40 clean (LCB ≥ 0.80) on at least 3 of 5;
  - no more falls, tilt or corner-lift failures than the control;
  - falls under its own exploration noise no worse than the control's.
- The comparison is made per training seed: a one-sided exact rank-sum test on per-seed CoP-failure counts, p ≤ 0.05 (5 vs 5 seeds allows that).
- Then a full-budget confirmation: 12M steps on seeds 56, 48 and one more (about 7.2 h each on this machine; run A took 17.4 h on Colab). Require stance_quality/v2 and the six-heading certificate on both the selected checkpoint and the final one.

**Not recommended:** another unchanged 12M production stance run on main; the per-step CoP margin alone at w 0.10; a physics change for this failure.

## 7. Open questions

1. Why did the same per-step CoP term help on r8 (62 → 21 of 120 CoP failures, 3 seeds) but not on r9 with interface r15 (2 seeds)? Physics, interface, driver and seed noise have not been separated.
2. Is the antisymmetric drift a free walk along a flat direction of the reward, or a product of learning to survive the exploration noise? Its side changes with the seed (r8 seed 44 left, probe seed 48 left, seed 56 right in both run A and the probe). The noise test ran only on the settled-reference checkpoints.
3. What should the stance stage certify beyond the statue? The statue passes, and so do reward-selected 50k-step checkpoints. What the trained policies add is survival under noise (+438 to +2,251 per episode).
4. With load-bearing digits (stiff-toe variant: pad 0.924, digits 27%), should the bar read the pad CoP or the whole-foot CoP? The whole-foot CoP was not measured there.
5. The T. rex statue stands on a lateral pad edge (abs outer CoP 0.954), documented as acceptable (`STANCE_HACK_AUDIT_2026_10.md:1085-1087`). Is it acceptable for a robot?
6. Dibothrosuchus's statue sits at CoP 0.745 on the heel side, 0.31° of ankle offset from 0.80. This model property needs settling before any quadruped CoP rail.
7. Run A's left-foot fore-aft CoP is not in its artifacts. Telemetry item 1 fixes this for future runs.

## Corrections to the earlier findings

1. In the statue, 92.4-93.6% of the digit load is at the bases, not 96%.
2. Under run A's held action, the whole-foot CoP is at the pitched toe-corner line (20.3-20.5 cm vs 20.55-20.62 cm; per seed, from 0.09 cm beyond to 1.9 cm short). It is not 0.4 cm short of a 20.88 cm edge. The 1.5-2 N·m gap between the actuator and floor moments is the foot's weight (1.34 N·m), not the ankle spring (+0.22 N·m). The remaining margin is 0-8 N·m, not "about 4".
3. The seven saturated reward terms sum to 3,081.7 (82%), not 3,062.
4. Energy is not zero for a nonzero constant action (-0.5 to -0.65 per episode for the run A holds). Impact and airborne are zero at the statue because it does not stomp.
5. "Digit III carries the least load, against anatomy" is overstated. The ostrich figures (77-84%) are for running, from two didactyl animals; track digit width is not load; one McCrea et al. print has digit IV widest. The model's order reflects lateral pad roll.
6. "Ramped ankle offsets reach 0.80 near the toppling limit" is withdrawn. The left ankle reaches 0.808 at -3.5° with 0/5 falls (still 0/5 at -4.0°); right-ankle falls stay at 1/5 from -3.25° to -4.0° (n = 5).
7. "Run A's foot pattern follows from linear superposition of statue slopes" is withdrawn. Its knee and ankle offsets alone fall in 10/20 episodes and leave the left pad at 0.34, not -0.02; the hip offsets are needed.
8. Probe 2: the episode-level Fisher p-values (0.017, 0.005) are not evidence for the arm; the same test between the two control seeds gives p = 0.010. "Halves the edge" and "the edge stays on the right pad" (seed 48 moved from left to right) are withdrawn.
9. The keyframe pull of leg_home_pose (root-cause item 3) is not supported as the cause: removing it left the edge in place (probe 2).
10. The 3.6° pad window and the 1.7° toe margin are ankle torque (servo-error) windows, not commanded-offset windows.
11. Probe 0's "31 CoP failures" on seed 56 includes one episode that fell at step 192; 30 were measured above 0.8.
