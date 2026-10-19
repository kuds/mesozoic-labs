# The per-foot split at Run 3 8.39M: stagger, twist and observation (report only)

Written 2026-10-19, 02:35-03:05 UTC. This is a report only. Nothing was changed in either clone, on Drive or on GitHub. Both clones are clean, at 7e6b557d (L) and 1af2d4d7 (L+T), before and after. Every rollout ran as one process at a time.

"Measured" means I ran or read it. "Inferred" is marked where it appears.

## Bottom line

1. **The 8.39M heel split is a fore-aft stagger of the feet, as in Run 2 at 4.2M, but larger, and it does not come only from the spawn.**
   - In 39 of 40 episodes the heel-loaded foot is the forward one, and in all 6 K2a episodes. corr(CoP split, stagger) = -0.93.
   - The 6 K2a episodes keep a window-mean stagger of 32-58 mm. The other 34 average 10 mm, and the statue 2 mm.
   - The stagger sd grows during the window at 8.39M: 20.1 mm at step 200, 24.2 mm at step 999. At 6.29M it falls from 11.7 to 5.5 mm. For the statue it falls from 8.7 to 0.5 mm.
2. **The ankles are not near their settled targets at 8.39M.**
   - Every episode's ankles sit +2.0 deg (mean) more open than the settled 1.3714 rad.
   - In the split episodes, the forward (heel) foot is +2.2 to +3.4 deg off (0.039-0.059 rad), and the rear foot +0.1 to +0.9 deg.
   - No episode has both ankles within 0.005 rad.
   - Run 2 at 4.2M is different. There the forward foot was +0.35 to +0.98 deg off and the rear foot -0.52 to +0.05 deg (panel mean +0.25 deg), so it was near the target.
3. **Each foot's CoP is set by its ankle torque over its load.** The correlation is +0.99 in every set.
   - With a stagger, the flat soles force the two ankle angles apart.
   - At 8.39M the policy commands nearly symmetric ankle targets. The angle difference alone becomes a servo-error difference: 2.0-2.2 deg on the heel foot against 4.2-5.1 deg on the rear foot (statue 3.8 deg). That puts the forward foot on its heel.
4. **The K2a failure needs two things at once:** the split, and a common backward shift of both feet's CoP.
   - The common shift is -0.14 in CoP units from 6.29M to 8.39M, the same +2 deg ankle opening.
   - By arithmetic, 8.39M's common shift with 6.29M's split puts 2 episodes past -0.86. 6.29M's common shift with 8.39M's split puts 0 there.
5. **The policy has no restoring torque on the twist mode. The statue does.**
   - The stagger is a whole-body twist: the pelvis yaws over the planted feet, and the feet pivot with it (the legs have no hip yaw joint).
   - The statue's fixed symmetric targets push back on a hip-pitch asymmetry with a servo error of slope -1.00 per unit of asymmetry. Its r-l hip force is +1.57 N·m per mm of stagger.
   - The policy's hip commands lead its hip angles: the slope is +0.62 at 8.39M, +0.74 at 6.29M and +1.38 for Run 2 at 4.2M, with r-l hip forces of -0.90, -1.24 and -2.07 N·m/mm. The force has the opposite sign to the statue's.
6. **At 8.39M only, the policy slowly turns in place toward the prey bearing** during the window.
   - corr(yaw change, bearing - yaw at step 200) = +0.88, against +0.04 at 6.29M.
   - Each degree of that turn adds -6.0 mm of stagger.
7. **Counterfactual rollouts of the 8.39M policy:**

   | Spawn | K2a |
   |---|---|
   | standard | 6 |
   | prey straight ahead | 4 |
   | mirrored legs | 3 |
   | mirrored legs and prey ahead | 1 |

   - Even the last one still builds a 15 mm sd stagger from the remaining small asymmetries (root velocity, neck and tail draws).
   - The statue with mirrored legs stays at 0.2 mm.
8. **The observation contains all the information. There is no explicit channel.**
   - The stagger is an exact linear function of the 8 observed leg angles (R2 = 1.00000).
   - Each foot's CoP is linearly predictable from observed channels: R2 0.84-0.87 from its 3 MTP angles alone, 0.965 with leg angles and touch.
   - The policy also observes the prey direction.
9. **Which candidate terms see the split** (Run 3 8.39M, K2a episodes against the clean ones):

   | Term | AUC | Notes |
   |---|---|---|
   | Ankle anchor | 0.495 | Does not see it. It sees the common ankle opening (0.050 per step at 8.39M against 0.0006 for the statue). |
   | L+T leg_home_pose (already on) | 0.84 | Sees it partly (+0.0375 per step). The policy keeps the split anyway, at a total reward cost of -0.060 per step. |
   | Stagger penalty | 1.00 | At sigma 10 mm it saturates at its maximum in the split episodes. At sigma 30 mm it does not. |
   | Tight per-foot CoP band, [-0.6, +0.4] | 1.00 | |
   | Per-foot CoP band at the K2b edges | 0.97 | Pays almost nothing (+0.0008 per step). |
   | COM band [0, 84] mm | 0.50 | Zero everywhere. |

## How the numbers were made

- **The rollout script: `roll_split.py`, run through `tools/iso.py` with ISO_WT set to the LT clone.**
  - It builds the judge's `Ctx("LT")` and rolls with `stance_report.run_panel(floor_truth=True)` on seeds 3042-3081 at heading 0.
  - The policy is loaded with `load_sb3_checkpoint` and predicts with `model.predict(deterministic=True)` on the VecNormalize-normalised observation.
  - Readings are taken after each `env.step`, through a wrapper on `SubstepContactRecorder.end_step`, as the judge does.
  - The info dict and the state just after reset are read through wrappers placed on the env *instance* when the recorder attaches.
  - A first try wrapped the env *class*. The plant contract then refused it (`policy_interface_sha256` changed), so nothing ran with it.
- **Reproduction check** (`tables/analysis.txt`, top):
  - For Run 3 8.39M, Run 3 6.29M, Run 2 4.2M and the statue, every seed's length and termination equals the judge JSON.
  - Each foot's `foot_cop_fore_aft`, the pad metric and the K2a/K2b flags are identical, with |difference| 0.
  - The statue falls on 3081 at 292 (nosedive), as in the judge.
  - The reset state is identical for the policy and the statue (max |qpos difference| 0).
- **Definitions:**
  - The window is recorder indices 200-999, the judge's window. Index i is the state after i+1 env.steps. "Reset" is the state after `env.reset`, before any step.
  - Stagger = the r foot site minus the l foot site, along the pelvis heading (mm, + = right foot ahead). The foot sites are `r_foot`/`l_foot`, on the metatarsus body, the judge's `primary_site`.
  - CoP is the judge's signed whole-foot CoP: the mean of `whole_foot_cop` over the loaded window steps, in span half-lengths, with - toward the heel.
  - Pelvis yaw is the heading of the root's x axis.
  - Servo error = `data.ctrl - q`.
- **Counterfactual rollouts.** These are report only and are not the judge's task:
  - `--spawn mirror_legs`: after the env's own reset, the left leg's 7 hinge qpos and qvel are set to the right leg's (hip roll sign-flipped), the root is re-settled with the env's own `_settle_root_on_ground`, and the observation is rebuilt with `_get_obs`.
  - `--env-override prey_lateral_range=[0,0]`: the RNG draw count is unchanged, so the joint noise is the same per seed.
  - A 4-episode check with `reset_noise_scale=0` and the prey ahead.
- **Scripts, all in this directory:**
  - `roll_split.py`; `run_all.sh` and `run_more.sh`, which are sequential.
  - `jac.py` (FK sensitivities, no dynamics).
  - `analyze_split.py`, which writes `tables/analysis.txt` and `tables/per_episode.csv`.
  - `analyze_split2.py`, which writes `tables/analysis2.txt`.
  - `noisefree_check.py`, which writes `tables/noisefree.txt`.
  - The npz files are in `out/` (463 MB).

## 1. Run 3 at 8.39M against 6.29M, per episode

The full 40-row table is in `tables/analysis.txt` ("Item 1"). It has the stagger at reset, index 0, index 200 and the window mean; each foot's CoP; each ankle's offset from settled; each ankle's servo error; MTP d3; hip pitch; hip roll; pelvis yaw at reset and in the window; foot yaws; and the 6.29M columns. `tables/per_episode.csv` has every field for every set, including MTP d2/d4 and the knees.

**The six K2a episodes at 8.39M:**

| seed | stagger reset / i200 / window (mm) | CoP r / l | ankle off settled r / l (deg) | servo err r / l (deg) | MTP d3 r / l (deg) | hip pitch r / l (deg) | hip roll r / l | pelvis yaw window (deg) | prey bearing | 6.29M: stagger window, CoP r / l |
|---|---|---|---|---|---|---|---|---|---|---|
| 3050 | -83.9 / -26.4 / -32.4 | +0.531 / -0.862 | +0.10 / +2.23 | 5.07 / 1.99 | 12.21 / 12.69 | -4.42 / -6.97 | +0.03 / +0.82 | -6.13 | +9.2 | -9.5, +0.461 / -0.595 |
| 3051 | -31.0 / -27.7 / -32.7 | +0.348 / -0.902 | +0.93 / +3.30 | 4.66 / 2.05 | 12.26 / 12.65 | -4.70 / -7.31 | +0.22 / +0.70 | -0.11 | +4.6 | -12.5, +0.378 / -0.595 |
| 3052 | +30.4 / +45.7 / +57.1 | -0.973 / +0.148 | +2.94 / +0.40 | 2.05 / 4.32 | 12.84 / 12.29 | -8.09 / -4.42 | -0.52 / -0.95 | -3.61 | -8.3 | +16.2, -0.778 / +0.489 |
| 3071 | -53.4 / -42.0 / -51.8 | +0.206 / -0.964 | +0.55 / +3.39 | 4.39 / 2.06 | 12.28 / 12.83 | -4.42 / -8.01 | +0.85 / +0.42 | +0.43 | +0.1 | -12.0, +0.299 / -0.569 |
| 3073 | -27.2 / -23.6 / -31.9 | +0.358 / -0.878 | +0.85 / +3.10 | 4.75 / 2.03 | 12.25 / 12.68 | -4.81 / -7.34 | +0.15 / +0.79 | +1.69 | +9.6 | -10.2, +0.347 / -0.540 |
| 3081 | +72.5 / +57.3 / +58.2 | -0.974 / +0.136 | +2.86 / +0.34 | 2.18 / 4.18 | 12.80 / 12.29 | -8.12 / -4.45 | -0.47 / -0.97 | +2.01 | +1.8 | +15.0, -0.707 / +0.413 |

The foot yaws equal the pelvis yaw to within about 0.3 deg in every episode. There is no hip yaw joint, so a foot can only turn with the body by pivoting on the floor.

**Per-set summary (measured; full-horizon episodes):**

| | Stagger sd: reset / i200 / window / i999 (mm) | Retained (window on reset, mm/mm) | CoP split sd | Common CoP mean | corr(split, stagger) | Ankle off settled, mean (deg) | K2a / K2b |
|---|---|---|---|---|---|---|---|
| Run 3 8.39M | 45.1 / 20.1 / 20.9 / 24.2 | 0.231 | 0.685 | -0.253 | -0.929 | +2.00 | 6 / 3 |
| Run 3 6.29M | 45.1 / 11.7 / 7.2 / 5.5 | 0.101 | 0.585 | -0.113 | -0.958 | +1.08 | 0 / 0 |
| Run 2 4.2M | 45.1 / 15.9 / 11.9 / 10.6 | 0.196 | 1.052 | +0.030 | -0.972 | +0.25 | 10 / 14 |
| statue | 44.2 / 8.7 / 2.5 / 0.5 | 0.047 | 0.159 | +0.058 | -0.996 | 0.00 | 1 / 1 (the fall) |

**Is the 8.39M heel split driven by a residual stagger, as in Run 2 at 4.2M?** Yes for the mechanism; partly for the cause.

- **What is the same as Run 2:**
  - The forward foot is heel-loaded and the rear foot toe-loaded: 39/40 here, 36/40 in Run 2, and every split episode in both.
  - The stagger is carried by a hip-pitch difference, with corr(stagger, r-l hip pitch) = -0.995. FK of the leg-angle differences reproduces it with corr 1.000.
  - The heel foot's ankle servo error sits near 2.0-2.7 deg in both runs (statue 3.8).
- **What is different:**
  - The stagger is larger: |window stagger| in the split episodes averages 44.0 mm against Run 2's 17.5 mm.
  - It is only partly a spawn residual. Its correlation with the reset stagger is +0.50 (Run 2 +0.74). It grows in the window as the pelvis turns toward the prey (section 2), and a mirrored spawn still produces 15.5 mm sd.
  - CoP split per mm of stagger is smaller: -0.030/mm against Run 2's -0.086 and the statue's -0.064.
  - The ankle commands are symmetric. The r-l ankle servo-error variance comes 87% from the angle difference at 8.39M. In Run 2 at 4.2M it came 75% from an asymmetric ankle *command*, and at 6.29M 69%.
  - A backward common shift is added (-0.253, against Run 2's +0.030).
- **6.29M to 8.39M, per seed:**
  - The split pattern is the same seeds: corr(split 6.29M, split 8.39M) = +0.922, and corr(stagger 6.29M, 8.39M) = +0.881.
  - The window stagger grew 2.5 times (slope).
  - The common CoP fell by -0.139 (sd 0.043).
  - The heel foot's CoP in the 6 episodes went -0.595 to -0.862, -0.595 to -0.902, -0.778 to -0.973, -0.569 to -0.964, -0.540 to -0.878, and -0.707 to -0.974.
  - The share of that move carried by the common shift is 0.37, 0.55, 1.37, 0.62, 0.48 and 1.02.

**Are the ankles near their settled targets in the split episodes?** No, as listed in the bottom line (item 2).

- The heel foot's ankle is off by more than in any clean episode: max |.| over 40 episodes is 3.39 deg (0.059 rad).
- The ankle *angle* is not what decides the split. The torque is: corr(foot CoP, ankle torque/load) = +0.990, corr(foot CoP, servo error) = +0.988.

## 2. Where the spawn stagger comes from, and how it closes

**The reset code** (`environments/shared/base_env.py` `reset`, LT clone):

- `mj_resetDataKeyframe(home)`.
- `qpos[7:] += U(-s, s)` on all 21 hinges: neck 3, tail 4, and per leg hip pitch, hip roll, knee, ankle and 3 passive toes.
- `qvel[:] += U(-s, s)` on all 27 DOFs, including the root's linear and angular velocity.
- A root-height normal draw. This is state-inert since the settle step below; it is kept only for the RNG sequence.
- `_spawn_target`: prey at distance U(10, 15) m and lateral U(-2, 2) m. `prey_lateral_range` is not set in `stance.toml`, so the species default applies.
- `_settle_root_on_ground`: a vertical root shift only.
- `s = reset_noise_scale = 0.05` (`stance.toml`). The judge spawns at heading 0.

**Measured:**

- The leg draws lie in ±0.0498 rad with sd 0.0288 (0.05/√3 = 0.0289).
- FK at the keyframe (`tables/jac.json`): a foot site's fore-aft position moves -0.867 m/rad at hip pitch, -0.5425 at the knee, -0.218 at the ankle, and 0 at hip roll and the toes. The site is on the metatarsus.
- So the expected reset stagger sd is 42.7 mm. Hip pitch carries 68.7% of the variance, the knee 26.9% and the ankle 4.3%.
- Measured reset stagger: sd 45.1 mm, mean |.| 37.2, range -121.7 to +72.5 mm. The linear FK prediction from each seed's draws matches with corr 1.0000 (max error 1.09 mm).
- Per seed, the hip-pitch draws contribute sd 36.8 mm, the knee 21.0 and the ankle 10.1.

**Zero action against the policy at the start:**

- The reset state is identical for both.
- At index 0, |policy - statue| stagger averages 0.6 mm (max 1.6) at 8.39M.
- By index 10 both have closed about 30% (mean |.| 24.2 against 24.6 mm). This is the common landing transient.
- They part after about 50 steps. At index 200 the mean |.| is 14.1 mm for the policy and 7.3 for the statue.

**How the statue closes it:**

- Zero action means the same home targets on both legs. In a staggered stance the forward leg's hip is more flexed than its target and the rear leg's less, so the servos pull both legs toward the same pose.
- The r-l hip servo error equals -(r-l hip angle), with slope -1.00. The r-l hip force is +1.57 N·m per mm of stagger.
- Because the feet are planted, the legs can only match if the pelvis turns square to the foot line, or the feet slide.
- **Rotation:** 58% of the closure comes from rotation (slope 0.579). The statue's pelvis yaw at step 999 tracks the yaw a slide-free closure would need, atan(stagger/width), with corr 0.956 and slope 0.59.
- **Pivot:** the feet pivot with the pelvis (foot yaw change 4.40 / 4.39 deg against the pelvis's 4.37 deg).
- **Slide:** the rest is foot slide, mean |.| 12.7 / 13.4 mm.
- The stagger sd goes 44.2 at reset, 8.7 at index 200, 0.5 mm at index 999.

**Why a policy keeps it.** The statements below are measured; the causal reading is inferred.

1. **No restoring torque.** The policy's hip-pitch commands lead its hip angles. The r-l hip servo error per unit of r-l hip angle is:

   | | Slope | r-l hip force (N·m/mm) |
   |---|---|---|
   | Run 3 8.39M | +0.62 | -0.90 |
   | Run 3 6.29M | +0.74 | -1.24 |
   | Run 2 4.2M | +1.38 | -2.07 |
   | Statue | -1.00 | +1.57 |

   - The command asymmetry is larger than the joint asymmetry: sd r-l hip ctrl 2.55 deg against 1.51 deg of r-l hip angle at 8.39M. Its corr with the stagger is -0.94.
   - So the policy holds the twisted pose, where the statue's servos push it out.
2. **It turns toward the prey (8.39M only).**
   - In the window the pelvis turns toward the prey bearing: corr(yaw change from step 200 to 999, bearing - yaw at step 200) = +0.880, slope 0.25.
   - At 6.29M the corr is +0.035, in Run 2 at 4.2M -0.008, and for the statue -0.59.
   - With the feet planted and pivoting, the stagger changes -6.0 mm per degree of that turn (corr -0.997).
   - The stance task pays `heading_weight` 0.1 × cos(yaw - bearing). The bearing reaches ±11 deg (prey lateral ±2 m at 10-15 m), and the policy observes the prey direction.
   - Example, seed 3052 (bearing -8.3 deg): yaw goes -1.7 to -5.3 deg from step 200 to 999, and the stagger +45.7 to +67.4 mm.
3. **It amplifies small asymmetries.**
   - With mirrored leg noise the 8.39M policy builds a 15.5 mm sd window stagger from 0. The sd is 12.6 mm at index 200, and it correlates -0.99 with yaw.
   - The statue with mirrored legs stays below 1 mm (0.2 mm window).
   - With no noise and the prey ahead, the policy drifts only -1.3 to -1.5 mm by step 999 (4 episodes; `tables/noisefree.txt`).
   - So the seed is the spawn asymmetry the mirror leaves in place. Eight reset asymmetries (root vy, wx and wz; neck yaw; tail_1 yaw; their rates; the prey bearing) give R2 0.52 against the step-200 stagger (n 40, 9 parameters; `analysis2.txt` H). The largest single correlations are root lateral velocity +0.48, tail_1 yaw +0.37 and neck yaw +0.34.
   - The policy's twist mode keeps that asymmetry and grows it.
4. **The reward already prices it.**
   - Split episodes earn 0.060 per step less than clean ones (-48 per window, about 1.3%): leg_home_pose -0.0375, foot_flatness -0.0139, foot_load_balance -0.0082, head_clearance -0.0079.
   - Turning back square needs a body twist on planted feet, with spin, speed and drift terms on. The policy does not find it.
   - It does close the largest spawn staggers by turning: seed 3063 went -121.7 to +3.7 mm with a -12.7 deg yaw, and seeds 3046 and 3072 went from about +70 mm to under 3 mm.
   - Inferred: this is a missing restoring behaviour, not a missing price.

**Counterfactuals on the 8.39M policy** (40 seeds; `tables/analysis.txt` per-set summary):

| Spawn / prey | Window stagger sd | CoP split sd | Common CoP | Heel min | K2a | K2b |
|---|---|---|---|---|---|---|
| standard (judge task) | 20.9 | 0.685 | -0.253 | -0.974 | 6 | 3 |
| prey straight ahead | 19.5 | 0.640 | -0.260 | -0.978 | 4 | 3 |
| mirrored leg noise | 15.5 | 0.552 | -0.254 | -0.872 | 3 | 0 |
| mirrored + prey ahead | 15.0 | 0.488 | -0.265 | -0.810 | 1 | 0 |
| statue, mirrored | 0.2 | 0.014 | +0.056 | +0.027 | 0 | 0 |

The common backward shift (-0.25) is the same in every row, and is present with no noise at all (CoP r/l about -0.25/-0.27). It belongs to the policy.

## 3. What the stance policy observes

`TyrannosaurusRexEnv._get_obs`, `environments/tyrannosaurus_rex/base_env.py`. It is identical in the L and L+T clones (diffed). The interface is "bipedal-target-heading-free/v1", 63 values:

| Block | Size | Notes |
|---|---|---|
| `qpos[7:]` | 21 | neck_pitch, neck_yaw, head_pitch, tail_1_pitch, tail_1_yaw, tail_2_pitch, tail_3_pitch; per leg hip_pitch, hip_roll, knee, ankle, toe_d2/d3/d4 (the passive MTP hinges) |
| `qvel[6:]` | 21 | the same joints |
| projected gravity | 3 | pelvis frame, from the root quaternion |
| pelvis gyro | 3 | sensor frame |
| pelvis linear velocity | 3 | gravity-aligned heading frame |
| pelvis accelerometer | 3 | |
| foot contact | 2 | per foot, the INSTANTANEOUS sum of the pad touch sensor and the 3 digit touch sensors (`_foot_contact_forces`) |
| prey direction | 3 | pelvis frame |
| prey distance | 1 | |
| body-relative command | 3 | zeros under command_mode "none" |

- **No explicit channel** for:
  - foot positions, or the stagger;
  - per-foot CoP or pad/digit load split (the 4 touch sensors are summed into 1 number per foot);
  - sole tilt, COM position or contact locations;
  - world yaw (heading-free).
- **Implicit**, measured as linear least squares over the window steps:
  - The stagger is linear in the 8 observed leg angles, R2 = 1.00000 in every set (FK).
  - Each foot's CoP: R2 0.836 / 0.869 from that foot's 3 MTP angles alone at 8.39M, 0.94-0.96 from its 4 leg angles, and 0.965-0.967 from MTP + leg + touch. The statue's MTP-only R2 is only 0.24-0.29, because its CoP barely moves.
  - The touch channels equal the floor-truth limb load (corr 1.000). The recorder's substep MIN is what was regressed; the observation carries the last substep's value.
- So the policy can see the split and the stagger, but only through FK and toe deflection.

## 4. Would each candidate term see the split?

Window-mean cost per step, steps 200-999. The "split" episodes are Run 3 8.39M's 6 K2a episodes; Run 2's are its 14 K2b episodes.

Term definitions:

- **anchor**: 0.4 × the mean over ankles of (1 - exp(-(e/0.10)^2)), with e about 1.3713665.
- **legpose**: 1.0 × (1 - `leg_home_pose_quality`), from the env's own info, so it is exactly L+T's term.
- **stagger**: 0.25 × (1 - exp(-(s/σ)^2)), at σ 10 mm and at σ 30 mm.
- **CoP band**: 0.25 × the mean over feet of (1 - exp(-(u/25 mm)^2)) on each loaded step's whole-foot CoP, with u the distance outside the band. The band is either the K2b edges [-0.929, +0.571] or a tight [-0.6, +0.4].
- **COM band**: as in `design_incentive/costs.py`: 0.25, [0, 84] mm of COM ahead of MTP III, σ 25 mm.

| Group | anchor | legpose (on) | stagger σ10 | stagger σ30 | CoP band K2b | CoP band tight | COM band |
|---|---|---|---|---|---|---|---|
| R3 8.39M split (6) | 0.0499 | 0.1374 | 0.2500 | 0.2044 | 0.0009 | 0.0729 | 0 |
| R3 8.39M clean (34) | 0.0502 | 0.0999 | 0.1345 | 0.0364 | 0.0001 | 0.0028 | 0 |
| R3 6.29M (40) | 0.0149 | 0.0342 | 0.0800 | 0.0146 | 0.0000 | 0.0020 | 0 |
| R2 4.2M split (14) | 0.0030 | 0.0326 | 0.2280 | 0.0730 | 0.0510 | 0.1138 | 0 |
| R2 4.2M rest (26) | 0.0026 | 0.0236 | 0.0713 | 0.0135 | 0.0008 | 0.0073 | 0 |
| statue | 0.0006 | 0.0177 | 0.0196 | 0.0028 | 0.0004 | 0.0020 | 0 |
| R3 8.39M mirrored spawn | 0.0536 | 0.1122 | 0.1526 | 0.0513 | 0.0000 | 0.0057 | 0 |

Inside Run 3 8.39M, split against clean:

| Term | AUC | Δ per step | Δ per window |
|---|---|---|---|
| anchor | 0.495 | -0.0003 | -0.26 |
| legpose | 0.838 | +0.0375 | +30 |
| stagger σ10 | 1.000 | +0.115 | +92 |
| stagger σ30 | 1.000 | +0.168 | +134 |
| CoP band K2b | 0.971 | +0.0008 | +0.6 |
| CoP band tight | 1.000 | +0.070 | +56 |
| COM band | 0.5 | 0 | 0 |

Spearman over the 80 Run 3 episodes:

| Term | with \|split\| | with heel depth |
|---|---|---|
| anchor | +0.01 | +0.38 |
| legpose | +0.17 | +0.52 |
| stagger σ10 | +0.86 | +0.91 |
| CoP band tight | +0.69 | +0.63 |

Readings:

- **Ankle anchor.** It is blind to the split. In a split one ankle opens and the other closes about the common level, so the mean over ankles barely moves.
  - It does see the common backward ankle opening: 0.050 at 8.39M, 0.015 at 6.29M, 0.0006 for the statue. The cost rises within the episode at 8.39M, from 0.045 at step 200 to 0.053 at step 999.
  - On Run 2's split it pays 0.0030 against 0.0026.
- **L+T leg_home_pose (on).** It sees the split through the hip-pitch asymmetry. The extra cost is r/l hip pitch 0.024/0.029 against 0.007/0.007 per step. It did not prevent the split.
- **Stagger.** It sees the split directly and is computable from the observation through FK.
  - At σ 10 mm it saturates (0.250, the maximum) in every split episode, so it gives no gradient at 30-60 mm. σ 30 mm keeps its slope there.
  - It also charges the 8.39M clean episodes (0.036 at σ 30), because the twist is everywhere, and charges the statue 0.003.
- **Per-foot CoP band.** It sees the outcome whatever its cause: stagger, ankle-command asymmetry (as in Run 2), or common shift.
  - At the K2b edges it pays almost nothing, because the per-step CoP rarely passes -0.929.
  - The tight band separates both runs.
  - It needs contact-space code in the env reward. The env has no per-foot CoP; only the recorder computes one.
- **COM band [0, 84] mm.** It is silent. The COM sits 31-50 mm ahead of MTP III in the split episodes (statue 64), inside the band.

## Implications (inferred unless marked)

1. **The 8.39M K2a failure has two separable parts, the split and the common backward shift, and each candidate covers at most one.**
   - The anchor and the COM band act on the common part. The arithmetic says removing the 6.29M-to-8.39M common shift would have left 0 episodes past -0.86 (measured arithmetic).
   - But Run 2 at 4.2M failed K2 with a near-zero common shift (+0.03) and ankles within about 1 deg. The anchor would have paid 0.003 there. A fore-aft term alone does not cover the split mode.
2. **More joint-space price on the legs is a weak lever for the split.**
   - L+T already pays +0.0375 per step for it, and -0.060 per step in total reward, and the policy keeps it.
   - The policy's hip commands are anti-restoring on the twist mode (+0.62 against the statue's -1.00).
3. **A stagger (twist) term is the most direct joint-space handle,** and its input is exactly observable. It needs code and moves the task digest.
   - It should not saturate at 20 mm: use σ of 30 mm or more, or a broad component.
   - It is stance-only, since walking needs a stagger.
   - Whether it would make the policy learn the restoring twist, or just trade against other terms, is untested.
4. **A per-foot CoP band on a tight band is the most outcome-aligned option.** It catches Run 2's ankle-command form and Run 3's geometric form alike. It needs contact-space reward code.
5. **Removing the sources helps on a fixed policy but does not fix it.**
   - With the 8.39M policy fixed, mirrored leg noise cut K2a from 6 to 3, prey-ahead from 6 to 4, and both from 6 to 1 (measured).
   - The policy still grows a 15 mm sd stagger from small asymmetries.
   - Both are task changes: the reset distribution and `prey_lateral_range`. Training under them is untested.
   - The heading term (0.1 × cos to the prey) is the only reason in the stance task for a policy to turn in place. The turn appears only at 8.39M.
6. **The v6b passive ankle spring was not rolled here.**
   - From the torque law: in a stagger the forward foot's ankle is the more open one, so a parallel spring centred near settled adds heel-ward torque on that foot, the same sign as the servo's.
   - It would address the common ankle opening, not the split.

## Files

- Notes: this file.
- Tables:
  - `tables/analysis.txt`: reproduction check, the item-1 table, per-set summaries, decomposition, mechanism, item 2, item 4.
  - `tables/analysis2.txt`: command asymmetry, yaw creep, observation regressions, term time courses, per-joint leg-pose cost, per-term reward split against clean, restoring torque.
  - `tables/per_episode.csv`: every field, every set.
  - `tables/jac.json`: FK, joint addresses, actuator kp and home ctrl.
  - `tables/noisefree.txt`.
  - `tables/log_1-7.txt`.
- Rollouts in `out/`:
  - `r3_8388608.npz`, `r3_6291456.npz`, `r2_4194304.npz`, `statue_LT.npz`;
  - `r3_8388608_mirror.npz`, `statue_LT_mirror.npz`;
  - `r3_8388608_prey0.npz`, `r3_8388608_mirror_prey0.npz`;
  - `r3_8388608_noisefree_prey0_4ep.npz`, `statue_noisefree_prey0_4ep.npz`.
