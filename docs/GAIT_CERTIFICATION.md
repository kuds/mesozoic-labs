# Physical gait reports and certification

The shared gait observer measures contact, stepping, slip and progress for the exact saved policy and normalization pair. Existing training configs and historical verdicts remain unchanged. They produce **report-only** development evidence; enforcement requires an explicit `locomotion_gait/v2` curriculum declaration.

A certificate qualifies a declared engineering event on a particular simulated plant. It does not establish biological plausibility, identify walking versus grounded running from energetics, or establish hardware performance. See the broader [gait quality plan](GAIT_QUALITY_PLAN_2026_09.md) and [historical audit](investigations/GAIT_AUDIT_2026_09.md).

## Scope: walking first

The gait owners decided on 2026-10-04 to certify **walking** first:

- **Step-to is pathological for every profile.** A gait in which one foot lands level with or behind the other foot instead of passing it is not certified as a walk or as an alternating gait. The trained compsognathus 1001 policy (`compsognathus_20261001_225856`), whose left foot lands a median 0.06–0.10 `L` behind the right along its line of travel (and further behind along its trunk, which crabs 11°), is a step-to gait and fails. Step-through is enforced by default; it is no longer an optional rail.
- **Neither trunk yaw nor a crab decides it.** Step-through is judged along the **line of progression** (the travel heading; the trunk axis only for a crab walk whose strides are clearly symmetric along it), and the two steps of a stride must not be grossly lopsided, so turning the trunk a few degrees can neither turn a step-to into a walk nor a walk into a step-to (walk-first round-2 findings, below).
- **Lenient, not strict.** The gait need not be textbook. Timing has a wide tolerance (0.15 cycle around a duty-shift-aware anti-phase target, 0.20 before a locked pair counts as off-gait), one off-gait budget of 15 % of the window absorbs pauses, stumbles and short bouts (a stop costs only its standing time), and every bar keeps a stated margin from the worst genuine development value (table below).
- **Walking profiles first.** `biped_walk` and `quadruped_walk` certify a walk. `biped_alternating` stays as the lenient run-allowed alternating profile for stages whose speed bar asks for running (the velociraptor stage, 2.0 m/s, Froude 0.82), with the same participation, step-through and anti-cheat rails but flight allowed.
- **Deferred.** Running, jumping, trot, pace and gallop profiles are deferred; no walk/trot/pace boundary is drawn (any symmetrical quadruped walking gait is a walk, see below). Hildebrand's labels remain report-only diagnostics.

Recommended profiles for the current stages (a certification config always declares its profile; report-only panels without one default by foot count and the Froude number `v^2 / (g L)` of the stage's speed bar, `default_gait_profile`):

| Species | Feet | Stage speed bar | Leg `L` | Froude of the bar | Profile |
| --- | --- | --- | --- | --- | --- |
| compsognathus | 2 | 0.08 m/s | 0.244 m | 0.003 | `biped_walk` |
| compsognathus robot | 2 | 0.04 m/s | 0.207 m | 0.001 | `biped_walk` |
| T. rex | 2 | 1.0 m/s | 0.877 m | 0.12 | `biped_walk` |
| velociraptor | 2 | 2.0 m/s | 0.50 m | 0.82 (≥ 0.5, the walk-run transition; Alexander 1989) | `biped_alternating` |
| brachiosaurus | 4 | 0.75 m/s | — | — | `quadruped_walk` |
| *Dibothrosuchus* | 4 | 0.9 m/s | 0.292 m | 0.28 | `quadruped_walk` |

## Measurement

Recording keeps v1's sources and adds the trunk orientation:

- Record the initial reset state and every physics substep through the existing probe hook. The observer does not advance dynamics, alter observations/rewards, or consume reset RNG. A previous hook runs first and is restored on exit.
- Sum active solved normal forces between explicitly registered distal-foot geometries and declared terrain. Whole legs are not feet. Foot-on-foot force and nonfoot ground force are separate diagnostics; inactive proximity contacts carry no support. This follows [MuJoCo's contact representation](https://mujoco.readthedocs.io/en/latest/computation/index.html#contact).
- Normalize load by the animal subtree's weight, excluding prey/food. `L` is the declared home-keyframe mean hind-hip anchor height above the authored reference plane; it is a scale reference, not a reconstructed anatomical measurement.
- Measure tangential velocity at each loaded contact point relative to terrain, including rigid-foot rotation.
- Record the free joint's quaternion (`root_quat_wxyz`, w x y z) at every sample. It gives the trunk's own axis, which decides the crab-walk frame and the body-frame step-to rule below. Each sample's quaternion is normalised; a trace without it, or with a nonfinite or degenerate (zero) quaternion, fails closed, and there is no fallback in the certified path. (The development benchmark adapter replays legacy traces recorded before this key with the travel heading standing in for the trunk axis and says so in its metrics; it never certifies, and such a trace cannot see a crab.)

The measurement core (`gait/events.py`, `gait/metrics.py`, `gait/labels.py`) treats a gait as a phase-locked pattern of limb oscillators (Collins and Stewart 1993; Golubitsky et al. 1999) and confirms it with Hildebrand's footfall estimator:

1. **Contact segmentation per limb.** A foot is loaded above 0.01 BW / n limbs. Unloads are merged back into stance when they are sub-dwell chatter (at most 10 ms), scuffs (never clear 0.01 `L` and move the foot less than 0.03 `L`, or are short) or impact bounces (shorter than 0.25 and lower than 0.3 of the limb's reference swing, the upper quartile of its clear swings). Loads shorter than 6 ms are dropped only after merging, so a touchdown is the first contact of a stance and an impact bounce never delays it. A swing is valid when it clears 0.01 `L` and repositions the foot 0.03 `L`.
2. **Standing and the moving clock.** The trunk **stands** where its horizontal speed over a centred 0.1-stride window (at least one sample either side) is below 0.05 of the upper quartile of that speed over the window. Stride periods and limb phase run on the **moving clock**, wall time with standing removed, so a stop does not stretch the strides around it: it is charged to the budget once, as its standing time (walk-first round 2; before, a 0.25–0.5 s stop at brachiosaurus or *Dibothrosuchus* cadence cost a whole stride or two of undefined phase and asymmetric strides). Participation and support shares (duty factor, load share, light stance, flight, unloaded time) are taken over moving time, so a pause on three legs does not unbalance the pair load ratio and standing double support does not pad a run's duty factor.
3. **Continuous phase.** Each limb's phase rises linearly between touchdowns on the moving clock. A stride longer than twice its *local cadence* (the larger of the median strides of up to three neighbouring strides of the same limb on either side) is a pause and leaves the phase undefined, so a limb that stops cycling while the trunk moves is a pause, but a genuine change of cadence (a walk-to-run transition) is not. Before the first and after the last touchdown the phase is extrapolated for as long as a normal stride may last (two references), so a standing start or a final stop inside the window keeps a defined phase.
4. **Pair statistics.** For every limb pair, the relative phase gives a time-weighted circular mean and mean resultant length (phase locking), a centred one-stride sliding estimate, and Hildebrand's per-stride footfall phases pooled in both directions. A duty-normalised stance-overlap index is 0 for anti-phase occupancy and 1 for in-phase occupancy.
5. **One template: alternation.** Every profile reads the same stored template, `templates.alternation`: the contralateral pairs (r/l; fore and hind on four legs) alternate. A pair is on template within 0.15 cycle of anti-phase anchored at touchdown *or* at mid-stance: with unequal duty factors `d_i`, `d_j`, evenly spaced mid-stances put the touchdown lag at `0.5 + (d_i - d_j) / 2`, so a genuine left-heavy walker (compsognathus 0921: duty 0.62 / 0.49) is symmetric at mid-stance. The duty asymmetry itself is judged once, by the pair duty ratio. On four legs there is no limb-phase partition: lateral- and diagonal-sequence walks, lateral- and diagonal-couplet walks, and trots and paces at walking duty are all walks; but the girdles must step together, so the fore-hind (ipsilateral) pairs must stay locally locked at whatever lag they keep (`ipsilateral_locking_min` is stored; a fore pair pattering at twice the hind cadence is uncoordinated stepping).
6. **One off-gait budget.** The **off-gait fraction** is the share of the analysis window spent demonstrably in something other than the declared gait, the union of: undefined phase (a limb not cycling or pausing); a contralateral pair locally locked more than 0.20 cycle (the template tolerance plus 0.05) from both anchors (a hop, bound, skip or gallop, whatever the lag); uncoordinated (locally unlocked) bouts, contralateral or, on four legs, ipsilateral, lasting at least two pooled strides or 10 % of the window, whichever is shorter; **standing**; **strides in place** (a footprint advancing less than 0.10 `L` along the travel heading: marking time, tapping, freezing; one isolated short stride that is at least a quarter standing, an adjusting step during a stop or the short first step from a standing start, costs only its standing time); and, on four legs, **asymmetric strides** (both contralateral pairs' per-stride footfall phases, on the moving clock, more than the tolerance from both anchors at once: a canter or gallop stride; strides holding standing time are not judged). Its components are stored beside it (`undefined_phase_fraction`, `off_template_locked_fraction`, `uncoordinated_bout_fraction`, `standing_fraction`, `in_place_stride_fraction`, `asymmetric_stride_fraction`, `longest_off_gait_s`, `off_gait_strides`) and are diagnostics only; the gate judges the one budget.
7. **Footprints, the line of progression and step-through.** A footprint is the load-weighted centre of a stance (force × held time), so a light toe touch ahead of the foot or a slide back under light load does not move it. A **step** is a footprint's advance past the other foot's latest footprint (centred no later than it) at the step's mid-time, measured in two frames: along the **travel heading** (the root displacement over a centred three-stride window) and along the **trunk axis** (the horizontal projection of the root body's x axis, `fx = 1 - 2(y² + z²)`, `fy = 2(xy + wz)`, averaged over a centred one-stride window to remove the stride-periodic yaw wobble). Per stride (consecutive steps of the two feet of a pair) the **step symmetry** is the shorter step over the longer (0 when the shorter one does not step forward), median over strides. The **line of progression** (`step_frame`) is the travel heading, unless the strides are at least 0.6 symmetric along the trunk and more symmetric there than along the travel (a crab walk: the body steps evenly along its own axis while it travels off it); step length, stride step-through and step symmetry are reported along it. When the line of progression is the travel heading but a foot's median step along the trunk is under 0.01 `L` (the feet come together relative to the pelvis), the travel-frame symmetry is also reported as `body_frame_step_to_symmetry`. Each frame's per-foot steps and symmetry are stored as diagnostics, with the trunk's crab angle. Stride length, skid and the in-place rule use the travel heading; only the progress rail (`mean_speed_mps`) uses the declared direction.
8. **Physical quantities.** Duty factor and load share per limb; contralateral load and duty ratios; on four legs the lighter girdle's share of the foot impulse, the ratio of the girdles' mean duty factors and the girdle-unloaded fraction (the lighter girdle under 0.12 of the foot impulse over a centred **half-stride** window, which sees a girdle unloaded on alternate strides); the light-stance fraction per foot (stance time under 0.1 BW / n); stride length (footprint to footprint, median over complete strides); swing floor contact and swing slip; the skid fraction (stance slip over trunk travel, a ratio of sums), the glide-stance fraction (stances whose median slip speed exceeds 0.6 of trunk speed) and the **walking glide-stance fraction** (above 0.4 of trunk speed: a walking stance is long enough for its median to ignore touchdown skid); the trunk height over `L` that the trunk keeps for all but a tenth of the window (10th percentile; the median is a diagnostic); `body_support_fraction`; foot-on-foot time; contact **flight** fraction; and the **unloaded fraction**, flight measured on load (the feet together under a quarter of their mean load, about body weight), which a light toe contact bridging a ballistic phase does not hide.
9. **Diagnostics (never verdict inputs).** Hildebrand per-stride labels and their distribution, Froude number, stride over Alexander's prediction, limb-phase mean and concentration, phase coordination index, alternation index, hop-flight fraction, per-foot step-length median, lower quartile and step-through fraction in the line of progression and the per-frame steps, the step-to bout fraction, the trunk's crab angle, the ipsilateral locking, and the contralateral offsets at both anchors.

Every reduction that reaches a stored metric is order-deterministic (`math.fsum`, sequential `cumsum`, scalar `math` trigonometry and exact elementwise arithmetic; no BLAS reductions and no SIMD transcendental ufuncs), and stored values are rounded to six decimals. The gate judges only stored values, so the reader's re-judge from `metrics_json` and its hash-checked replay from the raw traces reproduce the in-process verdict across machines. A mirrored or rotated episode (the quaternion transformed with it) measures the same. Missing, malformed, transposed, negative or nonfinite required telemetry, a missing or degenerate trunk quaternion, a non-positive body weight or leg length, or an unknown foot registry fails closed; unavailable JSON metrics are `null`.

`GaitProtocol` holds every metric-shaping setting above (template tolerance 0.15 and off-gait widening 0.05 cycle; uncoordinated-bout length two strides or 10 % of the window; pause factor 2 with three neighbouring strides; trunk-axis window one stride; step-through floor 0 `L`; crab-walk symmetry 0.6; body-frame step-through 0.01 `L`; in-place stride 0.10 `L`; standing window 0.1 stride and speed fraction 0.05; travel-heading window three strides and 0.05 `L` minimum travel; swing floor band 0.01 `L`; glide ratios 0.6 and 0.4 (walking); light load 0.1 BW / n; unloaded load fraction 0.25; girdle window half a stride and local share 0.12). They are hashed into `measurement_protocol_sha256`; changing one requires a new planned hash and a fresh panel. The gate bars below are separate, so retuning a bar is a re-judge of stored evidence.

## Profiles and criteria

| Profile | Template | Profile-specific bars (provisional) |
| --- | --- | --- |
| `biped_walk` | r/l alternate | walking support: contact flight ≤ 0.10 and unloaded time ≤ 0.15 of moving time, every limb's duty ≥ 0.35 (no duty > 0.5 requirement: compsognathus 0921's right foot is at 0.494), walking glide ≤ 0.15 of stance time |
| `biped_alternating` | r/l alternate | run allowed: flight ≤ 0.65 (sprint duty about 0.2; Weyand et al. 2000) |
| `quadruped_walk` | fore and hind pairs alternate, girdles locked together | walking support as `biped_walk` with every limb's duty ≥ 0.42 (below it a trot or pace has two suspensions per stride); girdle participation (lighter girdle ≥ 0.18 of the foot impulse, girdle duty ratio ≥ 0.60, girdle unloaded over half a stride ≤ 0.05 of the window) |

Every profile also requires step-through along the line of progression (each foot's median step ≥ 0.05 `L`; in ≥ 60 % of strides both steps step through), step symmetry along it ≥ 0.10 (no stride grossly lopsided: a leader-switching step-to or a step-to run dressed up by noise or trunk yaw), and, when a foot lands beside or behind the other along the trunk, travel steps at least 0.35 symmetric (`min_body_frame_step_to_symmetry`: otherwise the gait is a step-to on a crabbing path).

Each qualifying episode must jointly meet every declared criterion: it completes its horizon (and the analysis window, tolerance 1.5 × the largest sample interval), keeps the stage's speed bar (`min_episode_forward_vel`), and passes every rail below. All criteria are explicit curriculum keys with no hidden defaults; a key that the profile does not consume (a girdle bar on two feet, a walking-support bar such as `min_walking_duty`, `max_unloaded_fraction` or `max_walk_glide_stance_fraction` on `biped_alternating`) is refused rather than ignored, and the retired round-2 keys (`max_stall_fraction`, `max_asymmetric_bout_fraction`, `min_lead_exchange_fraction`, `min_template_coverage`, the trot/pace synchrony and walk-band bars) are unknown keys. Failure reasons read `group/rail: value (worst limb or pair) op bar`; the `group/rail` prefix is stable, so reports aggregate failures by rail.

**Margin table (walk-first round 2).** Provisional values come from `provisional_gait_criteria(profile)`. They were calibrated on development data only: the bake-off DEV split (synthetic footfall patterns at j0/j1 and the j3/j5 development seeds, one seed of physics puppets, and the re-recorded replays of the trained compsognathus 0921, velociraptor, T. rex, robot and *Dibothrosuchus* policies with the trunk quaternion), with every previous attack set (verification attacks, hardening rounds 1 and 2, and the walk-first round-1 attacks: biped-walk hacks, quadruped-walk hacks and genuine-walk false-reject probes) as development cases. Each cell is the worst genuine development value over native (2 ms), 10 ms and 20 ms sampling, its source, and the margin `(value - bar) / bar` for a floor or `(bar - value) / bar` for a cap; the last column is the nearest pathological value that the rail refuses. The body-frame step-to rail never fires on development data; its cell is the worst genuine probe (compsognathus 0921's footfalls with its trunk turned 12° further).

| Group | Criterion | Bar | `biped_walk`: worst genuine (margin) | `biped_alternating` | `quadruped_walk` | Nearest pathological |
| --- | --- | --- | --- | --- | --- | --- |
| participation | `min_limb_phase_coverage` | 0.8 | 1 (puppet compsognathus_walk; +25%) | 0.978 (real velo0922; +22%) | 1 (puppet dibothrosuchus_pace; +25%) | 0.8 (QH/pup_walk_rear_bouts) |
| participation | `min_limb_duty` | 0.1 | 0.48 (syn replica_compsognathus_walk, 10 ms; +380%) | 0.287 (real velo0922, 20 ms; +187%) | 0.482 (syn quad_pace, 10 ms; +382%) | 0.0933 (syn replica_dibothrosuchus_skid, 20 ms) |
| support | `min_walking_duty` | 0.35 / 0.42 | 0.48 (syn replica_compsognathus_walk, 10 ms; +37%) | — | 0.482 (syn quad_pace, 10 ms; +15%) | 0.35 (FR/scale_biped_run_L0p1) |
| participation | `min_relative_limb_load_share` | 0.36 | 0.911 (real comp0921; +153%) | 0.859 (real velo0922, 20 ms; +139%) | 0.439 (puppet dibothrosuchus_pace, 10 ms; +22%) | 0.355 (QHW/pup_hind2x_walk) |
| participation | `max_light_stance_fraction` | 0.18 | 0.11 (puppet compsognathus_walk, 20 ms; +39%) | 0.11 (puppet compsognathus_walk, 20 ms; +39%) | 0.0569 (puppet dibothrosuchus_pace, 20 ms; +68%) | 0.18 (real dibo0928, 20 ms) |
| participation | `min_pair_load_ratio` | 0.7 | 0.837 (real comp0921; +20%) | 0.753 (real velo0922, 20 ms; +8%) | 0.877 (puppet dibothrosuchus_pace, 20 ms; +25%) | 0.7 (QH/pup_walk_rf_tap_light, 20 ms) |
| participation | `min_pair_duty_ratio` | 0.7 | 0.801 (real comp0921; +14%) | 0.756 (real velo0922, 20 ms; +8%) | 0.909 (syn replica_brachiosaurus_walk, 20 ms; +30%) | 0.698 (BHJ/x15_limp_trex, 10 ms) |
| participation | `min_complete_cycles_per_foot` | 3 | 5 (puppet trex_walk; +67%) | 5 (puppet trex_run; +67%) | 4 (syn replica_brachiosaurus_walk; +33%) | 2 (QH/brachio_P20_stop_go_B) |
| stepping | `min_valid_swing_fraction` | 0.75 | 0.974 (real comp0921, 20 ms; +30%) | 0.974 (real comp0921, 20 ms; +30%) | 1 (puppet dibothrosuchus_pace; +33%) | 0.746 (BH2/composite_stepto_skip_mark_skim, 20 ms) |
| stepping | `min_median_swing_clearance_over_leg` | 0.02 | 0.0385 (syn replica_compsognathus_walk, 20 ms; +92%) | 0.0385 (syn replica_compsognathus_walk, 20 ms; +92%) | 0.0324 (puppet dibothrosuchus_pace, 20 ms; +62%) | 0.0199 (real trex0930, 20 ms) |
| stepping | `min_stride_length_over_leg` | 0.2 | 0.293 (syn biped_fast_walk; +46%) | 0.293 (syn biped_fast_walk; +46%) | 0.266 (puppet dibothrosuchus_walk; +33%) | 0.2 (QHW/pup_hind2x_walk, 10 ms) |
| stepping | `max_swing_ground_fraction` | 0.5 | 0.351 (syn replica_compsognathus_walk, 20 ms; +30%) | 0.351 (syn replica_compsognathus_walk, 20 ms; +30%) | 0.32 (puppet dibothrosuchus_walk, 20 ms; +36%) | 0.503 (real trex0925, 10 ms) |
| stepping | `max_swing_slip_fraction` | 0.1 | 0.00197 (real comp0921, 20 ms; +98%) | 0.00197 (real comp0921, 20 ms; +98%) | 0.00448 (puppet dibothrosuchus_walk, 20 ms; +96%) | 0.43 (VA/a17_combined_sliding_shuffle, 20 ms) |
| stepping | `min_step_length_over_leg` | 0.05 | 0.146 (syn biped_fast_walk; +193%) | 0.146 (syn biped_fast_walk; +193%) | 0.133 (puppet dibothrosuchus_walk; +165%) | 0.05 (BHW/A11_real_velo0922_stepto_r_d0.00_ownyaw) |
| stepping | `min_step_through_stride_fraction` | 0.6 | 0.895 (real comp0921; +49%) | 0.895 (real comp0921; +49%) | 1 (puppet dibothrosuchus_pace; +67%) | 0.598 (real trex0925, 10 ms) |
| stepping | `min_step_symmetry` | 0.1 | 0.431 (real comp0921; +331%) | 0.312 (real velo0922, 20 ms; +212%) | 0.771 (syn quad_walk_slow; +671%) | 0.0981 (syn quad_transverse_gallop) |
| stepping | `min_body_frame_step_to_symmetry` | 0.35 | 0.441 (probe BHW/F03_real_comp0921_yaw+12; +26%) | 0.441 (probe BHW/F03_real_comp0921_yaw+12; +26%) | — | 0.279 (BH2Q/real_velo0922_st_d0.06_crab12) |
| support | `max_flight_fraction` | 0.1 / 0.65 | 0.0463 (syn replica_compsognathus_walk, 20 ms; +54%) | 0.398 (syn replica_velociraptor_run, 20 ms; +39%) | 0.0222 (syn quad_pace, 20 ms; +78%) | 0.101 (syn quad_canter) |
| support | `max_unloaded_fraction` | 0.15 | 0.0886 (puppet compsognathus_walk, 20 ms; +41%) | — | 0.0667 (syn quad_trot, 20 ms; +56%) | 0.151 (BH/x08_hop_bouts_phase0_comp, 10 ms) |
| support | `max_body_support_fraction` | 0.01 | 0 (syn replica_compsognathus_walk, 20 ms; +100%) | 0 (syn replica_velociraptor_run, 20 ms; +100%) | 0 (syn replica_dibothrosuchus_trot, 20 ms; +100%) | 0.0309 (QHW/kneel_knee_contact, 20 ms) |
| support | `max_foot_foot_contact_fraction` | 0.02 | 0 (syn replica_compsognathus_walk, 20 ms; +100%) | 0 (syn replica_velociraptor_run, 20 ms; +100%) | 0 (syn replica_dibothrosuchus_trot, 20 ms; +100%) | 0.187 (syn biped_stacked_walk, 20 ms) |
| support | `max_skid_fraction` | 0.35 | 0.169 (puppet compsognathus_walk; +52%) | 0.293 (real velo0922, 20 ms; +16%) | 0.129 (puppet dibothrosuchus_pace, 20 ms; +63%) | 0.351 (FR2/V_velo_surge0.3, 10 ms) |
| support | `max_glide_stance_fraction` | 0.1 | 0 (syn replica_compsognathus_walk, 20 ms; +100%) | 0.0092 (real velo0929, 10 ms; +91%) | 0 (syn replica_dibothrosuchus_trot, 20 ms; +100%) | 0.167 (puppet compsognathus_robot_shuffle, 20 ms) |
| support | `max_walk_glide_stance_fraction` | 0.15 | 0 (syn replica_compsognathus_walk, 20 ms; +100%) | — | 0 (syn replica_dibothrosuchus_trot, 20 ms; +100%) | 0.186 (FR2/V_velo_skiddy, 10 ms) |
| support | `min_trunk_height_over_leg` | 0.5 | 0.9 (syn biped_fast_walk; +80%) | 0.9 (syn biped_aerial_run; +80%) | 0.9 (syn quad_pace; +80%) | 0.35 (QH2/brach_kneel_walk_z035) |
| participation | `min_girdle_load_share` | 0.18 | — | — | 0.223 (puppet dibothrosuchus_pace, 10 ms; +24%) | 0.179 (QH/rear_phantom_trot65_D, 20 ms) |
| participation | `min_girdle_duty_ratio` | 0.6 | — | — | 0.847 (puppet dibothrosuchus_trot, 10 ms; +41%) | 0.589 (QH/pup_walk_rear_bouts, 10 ms) |
| participation | `max_girdle_unloaded_fraction` | 0.05 | — | — | 0 (syn replica_dibothrosuchus_trot, 20 ms; +100%) | 0.0504 (QH/pronk_load_trot_contacts_j4_D) |
| coupling | `min_phase_locking` | 0.5 | 0.886 (syn replica_compsognathus_walk, 20 ms; +77%) | 0.599 (real velo0922, 20 ms; +20%) | 0.864 (syn quad_walk_diagonal, 20 ms; +73%) | 0.5 (BH/x08_hop_bouts_phase0_comp) |
| coupling | `max_alternation_phase_offset` | 0.15 | 0.0427 (syn biped_walk_stance_chatter, 20 ms; +72%) | 0.0427 (syn biped_walk_stance_chatter, 20 ms; +72%) | 0.0669 (syn replica_brachiosaurus_walk, 10 ms; +55%) | 0.151 (syn biped_double_step, 10 ms) |
| coupling | `max_alternating_overlap_index` | 0.5 | 0.114 (syn replica_compsognathus_walk, 10 ms; +77%) | 0.164 (syn biped_grounded_run, 20 ms; +67%) | 0.196 (syn quad_trot, 10 ms; +61%) | 0.504 (QHW/fore3to2, 20 ms) |
| persistence | `max_off_gait_fraction` | 0.15 | 0.00874 (real comp0921; +94%) | 0.14 (real velo0929, 20 ms; +7%) | 0.00444 (syn replica_brachiosaurus_walk, 20 ms; +97%) | 0.15 (QH2/dibo_trot_stutter_0.15s_every1) |

The thin margins are deliberate and are open owner decisions:

- *Pair duty and load ratios ≥ 0.70* (+14 % and +20 % for compsognathus 0921, the left-heavy genuine walker; +8 % for the velociraptor at 20 ms). Lowering both to 0.65 would admit the antalgic-limp probes at 0.66–0.70 (`r1_limp_066`, `x15_limp_trex`) and a phantom-contact walk, so the bars stay at 0.70.
- *Off-gait budget 0.15 for `biped_alternating`*: the velociraptor spends 0.095 of a window off-gait at 2 ms (+37 %) but 0.14 at 20 ms (+7 %), where a 0.28 s stride has 14 samples. The recorder stores every 2 ms physics substep. A 0.20 budget for the run profile would admit stagger-, skip- and hop-bout probes filling 15–20 % of the window.
- *Skid ≤ 0.35* is +21 % for the velociraptor at 2 ms and +16 % at 20 ms (touchdown skid); skating probes at 0.40–0.49 fail.
- *Quadruped walking duty ≥ 0.42* is +15 % from the genuine duty-0.5 trots and paces (0.482 at 10 ms); it is the walk-first labels' own boundary (a trot or pace below 0.42 is a flying gait). The flying trots and paces built at duty 0.36–0.40 measure 0.34–0.40 and fail; 0.40 would leave them at the bar.
- *Step symmetry ≥ 0.10* sits far below every development walker (+212 % at worst, the velociraptor at 20 ms) but only 21 % above the worst genuine probe, the velociraptor's own run sheared onto an 8° crab (0.121 along its travel); a step-to run with its trailing foot level with the leader measures 0.076–0.086.
- *Body-frame step-to symmetry ≥ 0.35* is +26 % from compsognathus 0921's footfalls with the trunk turned 12° further (0.441) and 20 % above a velociraptor step-to whose trailing foot lands 0.06 `L` behind in the body frame on a 12° crab (0.279).

Physical readings of the other bars are unchanged from the hardening rounds: a stride moves a foot at least a fifth of the leg (micro-step shuffles sit at 0.09–0.15 `L`); a swing clears the scuff height for at least half its duration and slides at most a tenth of its travel; a stance bears weight (at most 18 % of stance time under 0.1 BW / n); at most a third of the trunk's stance travel may come from sliding feet, at most a tenth of stance time may glide at 0.6 of trunk speed, and a walking stance may glide at 0.4 of trunk speed for at most 15 % of stance time; the lighter girdle carries at least 18 % of the foot impulse, and never less than 12 % over half a stride for more than 5 % of the window; the trunk stays at least half a leg high for 90 % of the window (kneeling bouts and crouching probes at 0.30–0.35 `L` fail); the feet carry at least a quarter of their mean load for 85 % of moving time.

`required_consecutive`, when present, must be 1 because the gate judges one fixed panel. The optional panel reward rail is explicit. No genuine trained quadruped policy exists yet (both *Dibothrosuchus* policies fail on light stance, contralateral load and duty asymmetry and flight), so quadruped bars rest on synthetic and puppet gaits.

### Walk-first round-2 findings and dispositions

The walk-first round-1 red team (biped-walk hacks, quadruped-walk hacks, genuine-walk false rejects) found these holes; each is now a development case:

- **Closed by measurement.** Step-to gaits whose trunk is yawed 7–17° off their travel (synthetic, compsognathus 0921 and 1001 footfalls, a leader-switching step-to whose yaw follows the leader, velociraptor step-to runs) and the mirror false rejects (genuine walks with the trunk turned 6–13° further, a velociraptor run turned 15°): step-through is judged along the line of progression with step symmetry, so trunk yaw alone no longer decides the verdict in either direction (all 14 step-to cases refused and all 10 turned-trunk genuine cases accepted, at 2, 10 and 20 ms). A run whose 20 % ballistic flight is bridged by a 2.5 % BW toe contact: refused on unloaded time (0.24). Quadruped girdles stepping at two or three times each other's cadence: ipsilateral unlocking is off-gait time. Flying trots masked by 0.03 BW phantom contacts and pronk loads behind phantom timing: unloaded time 0.23–0.61. Alternate-stride wheelbarrowing and rearing behind phantoms: the half-stride girdle window. Skating in two stances of three or on alternate hind stances at 47–58 % of trunk speed: walking glide. Kneeling for 40 % of the window: the 10th-percentile trunk height. Flying trots and paces at duty 0.36–0.40: the quadruped walking duty. Slow-cadence stops, standing starts and final stops (brachiosaurus, *Dibothrosuchus*, T. rex): the moving clock, moving-time participation shares, the edge extrapolation and the stop-aware in-place rule; a single mid-walk pause now costs its standing time only, so the effective allowance for one pause is the full 15 %.
- **Partly closed.** One uncoordinated stride at brachiosaurus cadence (2 s of a 9 s window): uncoordinated bouts now count from 10 % of the window, but a single random-phase stride often stays locally locked near the template and passes (6/6), as does a single canter stride in 2/6 seeds; one stride is arguably a stumble.
- **Declined.** A gentle turn that drops the progress along the declared direction under the speed bar (a 45° turn at 1.0 m/s against a 0.9 m/s bar) fails `mean_speed_mps` by design: progress is the stage's task, and only the owner can grant a heading tolerance. A brachiosaurus walk with 2.4 s strides and a 3 m walker with 2.3 s strides fail `min_complete_cycles_per_foot` in a 9 s window: the evidence floor stays at three whole strides per foot, and such plants declare a longer horizon (below). A token-stance rail (stances under a quarter of the limb's median impulse) was not added: genuine brachiosaurus controls reach 0.20–0.25 because a 9 s window holds only four or five stances per foot.

### Open decisions and known limitations

- **Residual attacks inside the lenient bars.** Of the previous attack sets' must-refuse cases still labelled pathological, these are accepted at 2 ms (per episode): a limp whose 28 % gallop bouts sit 0.175 cycle from its duty-shifted target, inside the 0.20 timing tolerance (6/6); stagger-, skip-, scramble- and canter-bout probes near the 15 % budget (1–5/6 each); one random-phase or canter stride at brachiosaurus cadence (6/6, 2/6) and a double step in a quadruped walk (1/6); a trot that stutters 0.22 s every other stride and a trot with a phantom fore contact every fourth stride (pair load 0.75) (6/6 each); a walk whose left hind foot phantom-taps every third stride (1/6); and a micro-step walk at the clearance and swing-floor bars (6/6). No 40-episode panel can reach 37/40 from the 1–5/6 cases.
- **The crab-walk frame.** A step-to along the line of travel whose trunk is yawed exactly enough to make its strides symmetric along the trunk (about `atan(stride / 2 × stance width)`, 13–22° for compsognathus' 0.6 `L` stance) is geometrically the same footprint pattern as a symmetric crab walk; the checker accepts both (`crab_walk_min_symmetry` 0.6). The attacks built on real footfalls stay below 0.47 symmetry along the trunk and fail; a synthetic step-to tuned to that yaw would pass.
- **Edge decelerations.** A stop at the end of the window (or a start at its beginning) modelled as slow motion, in which cadence slows with speed over the last stride, leaves the phase interpolation across that stride off template: at brachiosaurus, *Dibothrosuchus* and T. rex cadence 0–3 of 6 such episodes pass. Stops with a near-constant cadence, mid-window pauses and standing starts pass.
- **Legacy traces.** Traces recorded before `root_quat_wxyz` are measured in the development adapter with the travel heading as the trunk axis, so the trunk frame equals the travel frame: a step-to whose body-frame placement is sheared onto a 7–12° crab passes there as a lopsided step-through, and genuine crab walks crabbing 20–30° fail on step length; with the recorded quaternion all of them are judged correctly.
- **Hesitations, missed steps and standing starts (budget).** A synthetic walk with whole-body hesitations of 2–2.3 times the local cadence spends up to 0.18–0.31 of its window off-gait and fails some or most episodes (the real compsognathus and velociraptor bases with the same hesitations stay under 0.035). Pauses and standing starts inside the 15 % budget pass. A plant that genuinely needs longer to start declares a longer `settle_s`.
- **Long strides.** `min_complete_cycles_per_foot` (3) counts strides that lie wholly in the window, so a 9 s window cannot certify a four-beat walk slower than about 2.25 s per stride (a 3 m sauropod at Froude 0.03–0.05, or a 1 m brachiosaurus walking 2.4 s strides at twice Alexander's stride length); a panel for such a plant declares `horizon ≥ settle + 4.5 × slowest stride` (2000 steps for the stage-2 brachiosaurus config).
- **Progress is along the declared direction.** A policy turning off the task direction fails the speed bar by design.
- **Labels are descriptive.** Grounded running is labelled from duty and trunk bouncing, not energetics. An absence of flight does not distinguish a walk from grounded running ([Rubenson et al. 2004](https://pmc.ncbi.nlm.nih.gov/articles/PMC1691699/)); `biped_walk`'s support rule is a walking-support rule, not an energetic one.
- **Other body plans.** The code paths are generic over limb pairs, but the foot registry and contralateral pairs are defined for two and four legs only.

## Freeze the protocol without viewing the panel

Reserve the registered publication block **3042–3081** for the 40-episode certification panel. Do not use it for reward design, detector/threshold calibration, checkpoint selection or retries after inspecting outcomes. A code check can exclude recorded seed roles; it cannot prove that a human never tuned against held-out results.

Plan the hash before collecting held-out telemetry:

```bash
python -m environments.shared.scripts.gait_report trex --stage locomotion \
  --protocol-only --env-json frozen_env.json \
  --episodes 40 --seed 3042 --settle-s 1.0 > planned_protocol.json
```

`--protocol-only` constructs and validates the plant, prints the measurement payload/hash, and rolls no episodes. It needs no output directory and writes no report, even if `--out-dir` is supplied. `--protocol-json` can provide a plain object of detector options. `--direction X Y` is normalized before hashing.

Freeze the resulting `measurement_protocol_sha256`, the chosen profile, all gate criteria, exact `min_eval_episodes`, and `gait_panel_seed_start` in the proposed stage TOML. The hash pins detector options, implementation bytes, geometry registry, MuJoCo version, sampling rates, direction, settling exclusion, horizon and panel. Source changes require a new planned hash before evaluation.

## Development reports and saved-policy replay

Extract the exact recorded environment constructor kwargs into a plain JSON object. `--env-json` replaces the current config's environment kwargs; it does not merge them or silently recreate an old task from current defaults.

```bash
python -m environments.shared.scripts.gait_report trex --stage locomotion \
  --model replay/models/policy.zip --vecnorm replay/models/policy_vecnorm.pkl \
  --env-json frozen_env.json --episodes 3 --seed 9000 --settle-s 1.0 \
  --out-dir fresh_development_report
```

Three pilot episodes are development evidence, never a certification result. `--zero-action` replaces `--model` for a negative control and cannot certify. With existing reward gates, even a saved-policy report that clears provisional diagnostic criteria remains report-only.

Copy the checkpoint's recorded `stage_config.json` beside the checkpoint or one directory above it, and copy nearby run `provenance.json` when available. Keep the explicit matched VecNormalize file with the pair. The saved config must record `run.seed` and `run.n_envs`, corroborated by the checkpoint's JSON metadata. Known training seeds `seed+rank`, selection seed `seed+1000`, and recorded development/calibration/selection roles must not overlap certification seeds. Missing or conflicting seed provenance makes diagnostics ineligible and strict certification refuse.

Saved policies require compatible plant metadata and the exact task fingerprint. Normalization is frozen and reward normalization is disabled during inference; there is no sidecar guessing or unnormalized fallback. `--allow-legacy-plant` is diagnostic only and cannot mint a certificate.

After locking a new `locomotion_gait/v2` gate, use its explicit `--config gait_gate.toml`, the same frozen environment/protocol options, the selected pair, and the declared 40 episodes starting at 3042. Do not extend a failing panel or choose a favorable subset.

## Evidence and interpretation

- `gait_traces/episode_0000.npz` etc. retain numeric physics-substep telemetry for every episode.
- `gait_panel.csv` retains episode metrics and checkpoint, normalization, task and protocol bindings; `metrics_json` includes complete per-foot and timing diagnostics.
- `gait_report.json` records plant/task identities, geometry/scale references, detector/panel settings, seed provenance, file hashes, qualification failures and confidence statistics. Readers verify bindings and replay metrics from the raw traces rather than trusting a stored pass flag.
- Changed checkpoint/normalization bytes during measurement refuse the result. Validation/load failures invalidate an earlier report with an incomplete, noncertifying record. Completed result bundles, including nested or symlinked output paths, are immutable; use a fresh directory outside them.

For `k` joint successes out of fixed `N`, the one-sided exact 95% Clopper–Pearson lower bound is `BetaQuantile(0.05; k, N-k+1)` (`0` when `k=0`). At `N=40` and a declared lower-bound floor of `0.80`, **37/40 passes and 36/40 fails**. The certificate concerns the specified reset/scenario distribution and episode event; it does not cover arbitrary terrain, perturbations or species.

CLI exit 0 means a report or preflight completed, **not that a certificate passed**. Read `report_only`, `certification_eligible`, `certified`, failures and statistics. Exit 2 indicates command usage errors; exit 3 indicates an evaluation refusal.

Mac CPU replay is fresh evidence on that runtime, not confirmation of an original GPU trajectory. Contact-sensitive runs can diverge across hardware/backends; preserve runtime and plant bindings, and do not certify unverified model/physics revisions. The observer does not modify frozen MJX code or species physics. The [historical audit's replay limitations](investigations/GAIT_AUDIT_2026_09.md) explain this distinction.

## T. rex locomotion: the enforcement step

**Prepared 2026-10-04; apply after the gait-r1 pilot passes, not before.** `configs/trex/locomotion.toml` keeps `gate_kind = "reward_and_length/v1"` through the T. rex locomotion task revision `gait-r1` (the gait reward kit, the speed cap at 1.25 m/s and a 2000-step horizon). The plan's pilot runs that revision first ([GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md) §5.5; the command-line steps are in [NEXT_STEPS.md](NEXT_STEPS.md) §3). Only when the pilot passes is the `[curriculum]` change below committed. It is a gate revision of its own, landed after the gait-r1 commit and before the training session. Its commit names the moved `gate_sha256` and `stage_config_view_sha256` lines of T. rex locomotion. From then on, reuse rule 7 refuses every T. rex locomotion verdict judged under `reward_and_length/v1`.

### Why the panel is 20 s

A panel rolls the task's own horizon. `gait/report.py` reads `max_episode_steps` from the stage's environment kwargs, and the panel's task fingerprint must equal the checkpoint's. Judging on 20 s therefore means training on 20 s, and gait-r1 sets `max_episode_steps = 2000`.

- After the 1 s settle, a 9 s window holds 8 to 10 T. rex strides (0.9 to 1.1 s each). A 19 s window holds 17 to 21.
- The off-gait budget is 15 % of the window: 1.35 s of a 9 s window, little more than one stride. A single stumble, or a slow standing start (standing time counts against the budget), can use most of it.
- The walk-first verification of 2026-10-04 ran a T. rex-cadence walk with two trips 35 % late on development data. In a 9 s window it passed 4, 3 and 3 of 6 episodes (2, 10 and 20 ms sampling), with 0.095 to 0.188 of the window off-gait. The same trips take 0.045 to 0.089 of a 19 s window, inside the budget. That figure is arithmetic, not a measured panel.
- At 37 of 40, a policy whose episodes qualify 95 % of the time already fails 14 % of panels (GQ-7). Thin per-episode margins cost a whole retrain.
- `min_complete_cycles_per_foot` (3) is met in either window. The margin that matters is the budget's.

The cost is training time. The in-training evaluations (30 episodes every 50,000 steps) double in length, so the estimate for an 8M session is 13 to 15 h, against 8h46m for the certified run on 1000-step episodes. That is still inside Colab's roughly 24 h cap.

### The block

The block replaces `reward_and_length/v1`'s gate keys in `[curriculum]`. It removes `min_avg_episode_length` and `min_avg_forward_vel`, which the schema refuses under `locomotion_gait/v2`, and keeps `timesteps`, the collapse keys and the warm-up and ramp keys as they are. Every bar is the walk-first `biped_walk` value (`provisional_gait_criteria("biped_walk")`, margin table above), except the stride floor, which is tightened for T. rex (below).

```toml
gate_schema_version = 1
gate_kind = "locomotion_gait/v2"
gait_profile = "biped_walk"                  # Froude 0.12 at the bar: a walk
measurement_protocol_sha256 = "sha256:..."   # the --protocol-only output below
min_eval_episodes = 40
gait_panel_seed_start = 3042                 # the registered certification block, 3042-3081
min_gait_success_lcb = 0.80                  # 37/40 passes, 36/40 fails
required_consecutive = 1
min_avg_reward = 100.0                       # optional panel reward rail, kept from v1
min_episode_forward_vel = 1.0                # the stage's speed bar, per episode
min_episode_duration_s = 19.0                # 2000 steps x 0.01 s, less the 1 s settle
# support (walking)
max_flight_fraction = 0.10
max_unloaded_fraction = 0.15
min_walking_duty = 0.35
max_walk_glide_stance_fraction = 0.15
max_body_support_fraction = 0.01
max_foot_foot_contact_fraction = 0.02
max_skid_fraction = 0.35
max_glide_stance_fraction = 0.10
min_trunk_height_over_leg = 0.50
# participation
min_limb_phase_coverage = 0.80
min_limb_duty = 0.10
min_relative_limb_load_share = 0.36
max_light_stance_fraction = 0.18
min_pair_load_ratio = 0.70
min_pair_duty_ratio = 0.70
min_complete_cycles_per_foot = 3
# stepping
min_valid_swing_fraction = 0.75
min_median_swing_clearance_over_leg = 0.02
min_stride_length_over_leg = 0.40            # T. rex: 0.20 for the profile (below)
max_swing_ground_fraction = 0.50
max_swing_slip_fraction = 0.10
min_step_length_over_leg = 0.05
min_step_through_stride_fraction = 0.60
min_step_symmetry = 0.10
min_body_frame_step_to_symmetry = 0.35
# coupling and persistence
min_phase_locking = 0.50
max_alternation_phase_offset = 0.15
max_alternating_overlap_index = 0.50
max_off_gait_fraction = 0.15
```

Leave `gait_report_episodes` out: on a `locomotion_gait/v2` stage it must be absent or equal `min_eval_episodes`.

The block was checked on 2026-10-04 in a scratch merge of this branch with the PR #585 plumbing fixes:

- `gate_schema.validate_gate_config` and `GaitGateThresholds.from_curriculum` accept it.
- Left in place, `min_avg_episode_length` and `min_avg_forward_vel` are refused.
- With the digest planned there, the gait preflight passes for run seed 45 and refuses run seeds 3040, 2045 and 1040, whose training, selection and replay seeds collide with the block.

**The T. rex stride floor, 0.40 `L`** (`L` = 0.877 m, so 0.351 m), tightened from the profile's 0.20:

- The audited hops stride 0.11 to 0.22 `L` (plan §4.4; the seed-42 and seed-44 hops 0.17 to 0.21 `L`). The profile's 0.20 sits among them: on its own it would pass the longest hops, which the other rails refuse.
- At the 1.0 m/s speed bar, a stride under 0.40 `L` takes more than 2.85 strides a second, a period under 0.35 s. That is shorter than T. rex's 0.39 s swing reference (1.3 √(`L`/g)) alone.
- The re-scoring's T. rex puppet walks at 1.04 m/s were designed with 0.44 and 0.51 m steps (strides of 0.88 and 1.02 m, 1.00 to 1.16 `L`), and Alexander's prediction at Froude 0.12 is 1.21 `L`. A walk at the bar sits at least 2.5 times above the floor, and the longest hop 45 % below it.
- The profile's 0.20 rests on a synthetic fast walk at a generic scale (0.293 `L`) and stays the value for the other species.
- Replayed with the gait report CLI on development seeds 9000 to 9003, the seed-42 hop measures a stride of 0.16 to 0.17 `L` (7.7 Hz).

No other bar is tightened. The swing clearance floor (0.02 `L`, 17.5 mm) sits at the T. rex hops' 17 to 25 mm, and the margin table's nearest refused value is a T. rex hop at 0.0199 `L`. No T. rex walk has been measured to calibrate a higher floor, and the hops fail the coupling, support and step-through rails regardless.

### Planning the hash

Run this on the commit that will train, with the gait-r1 commit merged (it supplies the 2000-step horizon) and the measurement code final:

```bash
python -m environments.shared.scripts.gait_report trex --stage locomotion \
  --protocol-only --episodes 40 --seed 3042 --settle-s 1.0 > planned_protocol.json
```

- The committed `configs/trex/locomotion.toml` supplies the task, so `--env-json` is not needed.
- `--protocol-only` rolls no episode. It prints `measurement_protocol_sha256` and the payload it hashes, whose `panel.horizon_control_steps` must read 2000.
- Copy the digest into the block, then check that the stage validates with the gait preflight, `python -m environments.shared.gait.preflight --check`, which CI also runs.
- Re-plan if the measurement code changes before the session: the preflight refuses a stale digest before anything is trained.
- The session's `SEED` must keep its training (`SEED`+rank), selection (`SEED`+1000) and replay seeds off 3042 to 3081. The pilot's seed, 45, does.

Do not use 3042 to 3081 for anything else in the meantime: no pilot judging, no reward design and no retries. The pilot is judged on the development block (9000 onward). The re-scoring that chose the gait-r1 weights replayed the old hops on part of the certification block, and repeated its measurements on the development block before the weights were committed ([investigations/TREX_GAIT_R1_RESCORE_2026_10.md](investigations/TREX_GAIT_R1_RESCORE_2026_10.md)).
