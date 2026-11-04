# REACH: swing and stride reach of the T. rex (physics r10) and of compsognathus (the positive control)

Stage REACH of the T. rex walking-feasibility study, main 4d5f4e7 export. This stage is analysis only. Both models were built with the export's own factory (`environments.shared.config.build_env(ModelId(species, "anatomical"), "locomotion")`; the stance stage compiles to the same physics, per the FACTS sheets).

- Kinematics were read with `mj_kinematics` on separate `MjData` objects.
- Static torques come from `mj_fwdPosition` and `mj_fwdVelocity` at qvel = 0.
- Distances come from `mj_geomDistance`.
- The only dynamic simulation is each species' zero-action, noise-free statue, run to read the settled pelvis height and pitch.
- No file in the repository or the export was changed, and no model was edited. No HYPOTHETICAL model was built. No policy was trained.
- Runs were one process at a time, with `OMP_NUM_THREADS=1`, `nice -n 10` and TMPDIR inside this folder.

Labels: **[measured: file]** means a number read from a run in this folder or from a repository file:line. **[inferred: basis]** means arithmetic or reasoning on measured numbers.

Units: m, rad or deg as stated, N, N.m. **L** is leg length: the mean world z of the two hip-pitch anchors at the home keyframe, the gait probe's definition. It is 0.8766 m for the T. rex and 0.2444 m for compsognathus.

## 0. Bottom line for this stage

1. **The T. rex's reach does not stop it walking.**
   - At the settled standing height, with the pad flat, the planted foot can sweep **0.614 L (0.538 m)** fore-aft under the hip. This figure comes from a closed form and agrees with the converged grid. Centred on the CoM the sweep is 0.584 L. [measured: tol_check_stdout.txt]
   - Rolling the foot up to 15 deg raises the sweep to about 0.80 L; nearly all of the gain is toe-off. A 5 % crouch raises it to about 0.82 L. [measured: reach3_run3_stdout.txt]
   - The GQ-8 genuine-walker band (stride 0.336-0.742 L; `docs/GAIT_QUALITY_PLAN_2026_09.md:190`) needs a planted sweep of only **0.17-0.45 L** at duty 0.5-0.6. That is 27-72 % of what the T. rex has.
   - Even the top of the band at duty 0.6 only straightens the stance knee from its 135 deg home to 152 deg interior. [measured: stride_angles_stdout.txt]
2. **Toe clearance is not a limit.**
   - Over every fore-aft position from the rearmost (PEP) to the foremost (AEP) foot position, the swing leg's lowest point can be lifted at least **0.190 m (0.22 L)**. The requirement is 2-3 cm.
   - A foot carried level 3 cm up spans the whole PEP-AEP range. [measured: reach3_run3_stdout.txt]
3. **Static ctrl and torque are not a limit. The cost is soft servos.** This is a static check only.
   - In single support with one body weight, anywhere on the flat envelope, the T. rex stance leg uses at most **18 %** of any forcerange when the ground force is vertical, and at most **27 %** when it acts through the CoM.
   - Every ctrl it needs is inside its ctrlrange.
   - Band strides need policy actions of **|a| ≤ 0.42** without load.
   - The cost is servo offset: up to 15.8 deg on hip pitch at the AEP, and 10-12 deg on hip roll. [measured: reach3_run3_stdout.txt, action_ranges_stdout.txt]
4. **Lateral room is adequate. The pad rolls with the hip.**
   - A planted foot can be placed from -0.55 L to +0.23 L about the midline (hip roll -22 to +22 deg). The midline is reached at 9.2 deg of hip roll and the CoM line at 9.7 deg.
   - The pad tilts by the same angle because there is no ankle-roll joint. [measured: reach3_run3_stdout.txt]
5. **No feet, legs or tail collide in a normal stride.**
   - With hip roll 0 the closest leg-leg gap is 0.100 m (the thighs), the closest leg-tail gap over collidable pairs is at least 0.244 m, and the closest leg-belly gap is at least 0.098 m.
   - At mid-swing the swing foot can be adducted to the midline past the stance foot with a 17 mm gap. The legs first touch at 15 deg of adduction. [measured: reach3_run3_stdout.txt]
6. **Calibration against compsognathus.** The walking positive control passes every check. [measured, and inferred from the GAIT_QUALITY_PLAN numbers]
   - Its relative reach is larger: 0.891 L converged.
   - Its stance torque reaches 100-120 % of forcerange at the far ends of its sweep under a vertical ground force.
   - Its measured walk (stride 0.37 L, duty about 0.5, on physics r1) used about 0.19 L, roughly 21 % of its sweep.
   - The T. rex has comparable relative reach and larger torque margins. **So swing and stride geometry and static ctrl reach do not explain why no T. rex alternating gait has appeared.** [inferred]
7. **What this stage does not show.** It checks kinematics, static torque and ctrl reach only. It does not test balance: neither quasi-static support (CoM over the foot) nor dynamic support (CoP/ZMP inside the foot). It does not test servo tracking, bandwidth, the 10 Hz command filter or swing timing. Those belong to the other stages (section 8).

## 1. Setup and definitions

| quantity | T. rex | compsognathus | label |
|---|---|---|---|
| L | 0.8766 m | 0.2444 m | [measured: reach3_run3.json `L_m`] |
| statue | 1000 control steps (10 s), noise 0, zero action, no termination | 500 steps (10 s), no termination | [measured: reach3_run3_stdout.txt] |
| settled hip height H0 (mean hip-pitch anchor z) | 0.8770 m = 1.0004 L | 0.2445 m = 1.0003 L | [measured: same] |
| settled pelvis pitch (nose-down +) | 0.942 deg at the end; mean 0.955 deg over steps 200-999 (the env comment says 0.95: `tyrannosaurus_rex/base_env.py:465-478`) | -0.007 deg | [measured: same] |
| CoM ahead of the hip anchors | 0.1532 m = 0.175 L | 0.0134 m = 0.055 L | [measured: same] |
| animal mass and weight | 85.72 kg, 840.9 N | 1.000 kg, 9.81 N | [measured: same] |
| hip separation (natural track width) | 0.28 m = 0.319 L | 0.094 m = 0.385 L | [measured: explore_stdout.txt anchors] |
| knee interior angle at home | 135.0 deg (straight at q = -69.8 deg; the range allows 5 deg of hyperextension) | 118.5 deg (the extension stop at -28.65 deg gives 147.1 deg, so it never straightens) | [measured: reach3_run3_stdout.txt; inferred: interior angle = home value minus the knee rotation] |
| metatarsus capsule above the sole with a flat pad | 3.0 mm | 4.4-5.0 mm at the envelope ends | [measured: reach3_run3.json `AEP/PEP_nonsupport_clearance_m`] |
| planar FK against `mj_kinematics`, 300 random configurations | max error 8.9e-16 m | 2.2e-16 m | [measured: reach3_run3_stdout.txt] |

Definitions [inferred: gait kinematics]:

- **Planted foot.** The lowest support point sits on the floor with the pelvis at hip height H. Support means the pad plus digits; for T. rex toe-off it means the pad only, because the passive digit hinges flex up onto the floor. The thigh, tibia and metatarsus may not touch the floor. The pad is either flat in world pitch or rolled about its edge by φ: heel strike φ < 0, toe-off φ > 0, |φ| ≤ 15 deg.
- **Foot position.** The flat-equivalent pad centre, relative to the hip.
- **AEP and PEP.** AEP is the most forward planted position and PEP the most rearward. Only the contiguous range that contains the CoM projection is kept.
- **Sweep E.** E = AEP - PEP.
- **Gait requirement.** A stride S at duty factor β moves the hip β·S over the planted foot, so it needs **E ≥ β·S**. Hence S_max = E/β.
  - The gait probe's stride is speed × stride period: one full cycle, two steps (`docs/investigations/gait_2026_09/gait_probe.py:105,843-845`).
  - E_sym is the sweep symmetric about the CoM projection: 2·min(AEP - x_CoM, x_CoM - PEP).
- **"Practical" envelope.** Every joint at least 5 deg from its range ends, and the knee interior angle at most 175 deg.
- **Single-leg pose.**
  - No T. rex single-leg pose exists in the export. A grep for single-leg, one-foot and single-support turned up only reward comments, for example `configs/tyrannosaurus_rex/anatomical/stance.toml:723`.
  - Compsognathus has only open-loop one-leg scripts, described in words (`configs/compsognathus_longipes/anatomical/stance.toml:79-84,311-319`).
  - The reference is therefore the home stance: pelvis level at the statue pitch and at H0, with both legs on the same pelvis. The sensitivity check (§2) rolls the stance hip until the pad sits under the CoM.
- **Grid tolerance.** Grids select configurations within ±0.0025 L of the target H: 2.2 mm for the T. rex, 0.61 mm for compsognathus.
  - Near the straight-knee singularity this inflates the T. rex E by +0.011 L at H0. The compsognathus bias is +0.008 L.
  - The converged values are in Table 2 [measured: tol_check_stdout.txt]. The other grid values carry a bias of about +0.01 L.

## 2. Planted-foot (stance) envelope at the settled height H0

Positions are measured from the hip anchor; + is forward.

| species / mode | AEP | PEP | E | E_sym about CoM | binding constraint | label |
|---|---|---|---|---|---|---|
| T. rex, flat pad, closed form (exact H0) | +0.409 m (+0.467 L) | -0.129 m (-0.147 L) | **0.614 L** (0.538 m) | **0.584 L** | knee fully straight (180 deg) at both ends: femur 0.3513 + tibia 0.3513 m against a 0.649 m hip-ankle height | [measured: tol_check_stdout.txt] |
| T. rex, flat, grid 0.05 deg, tol 0.22 mm | +0.4095 | -0.1296 | 0.615 L | 0.585 L | same | [measured: tol_check_stdout.txt] |
| T. rex, flat, grid tol 2.2 mm (used in Tables 3 and 6) | +0.414 (+0.472 L) | -0.134 (-0.153 L) | 0.625 L | 0.595 L | knee 180 deg; ankle 10 deg from its upper stop at the AEP | [measured: reach3_run3_stdout.txt, inspect_run3_stdout.txt] |
| T. rex, practical | | | 0.621 L | 0.591 L | knee 175 deg; ankle 12.7 deg from its stop | [measured: same] |
| T. rex, knee ≤ 170 / 160 / 150 deg | | | 0.609 / 0.559 / 0.465 L | 0.579 / 0.529 / 0.436 L | knee cap | [measured: reach3_run3_stdout.txt] |
| T. rex, rolling ±15 deg | +0.475 L (best heel strike, at -5 deg; flat gives +0.472 L) | -0.332 L (toe-off +15 deg) | 0.807 L | 0.600 L | heel strike adds almost nothing. The pad's rear edge is only 0.07 m ahead of the ankle, so rotating the foot about it moves the ankle back about as much as it lowers it. The whole gain is toe-off at the PEP. | [measured: same; per-φ rows in inspect_run2_stdout.txt; inferred: rotation geometry] |
| T. rex, rolled single-support pose (hip roll 9.70 deg, pad under CoM, pad roll 9.20 deg), continuation, exact H | +0.390 m | -0.110 m | 0.570 L (same method at roll 0: 0.610 L) | 0.540 L | knee straight | [measured: reach3_run3_stdout.txt] |
| compsognathus, flat, converged (grid 0.25 deg, tol 0.06 mm) | +0.1329 | -0.0849 | **0.891 L** (0.218 m) | **0.804 L** | knee on its extension stop at both ends (147.1 deg); ankle 0.5 deg from its stop at the AEP; toe 0.2 deg from its stop at the PEP | [measured: tol_check_stdout.txt, inspect_run2_stdout.txt] |
| compsognathus, flat, grid tol 0.61 mm | +0.134 (+0.548 L) | -0.0857 (-0.351 L) | 0.899 L | 0.811 L | same | [measured: reach3_run3_stdout.txt] |
| compsognathus, practical | | | 0.825 L | 0.723 L | 5-deg margins | [measured: same] |
| compsognathus, knee 10 / 20 deg below its maximum | | | 0.797 / 0.642 L | 0.713 / 0.566 L | knee cap | [measured: same] |
| compsognathus, rolling ±15 deg | +0.573 L | -0.491 L | 1.065 L | 1.037 L | | [measured: same] |
| compsognathus, rolled pose (hip roll 12.25 deg, pad roll 12.24 deg) | +0.121 m | -0.073 m | 0.795 L (same method at roll 0: 0.850 L; the ankle was scanned in 5-deg steps) | 0.705 L | knee stop | [measured: same] |

Every flat envelope at H0 is one contiguous range: there are no empty 0.005 L bins between PEP and AEP [measured: reach3_run3.json `empty_bins_between_branches` = 0]. The planted foot sweeps continuously.

## 3. Sweep against hip height

Grid values, about +0.01 L biased (§1). The pelvis z values are approximate.

| H / L | T. rex pelvis z (m) | T. rex E flat / practical / rolling (L) | compsognathus pelvis z (m) | compsognathus E flat / practical / rolling (L) |
|---|---|---|---|---|
| 1.000 (H0) | 0.926 | 0.625 / 0.621 / 0.807 | 0.244 | 0.899 / 0.825 / 1.065 |
| 0.975 | 0.904 | 0.734 / 0.730 / 0.905 | 0.238 | 0.982 / 0.907 / 1.130 |
| 0.950 | 0.882 | 0.824 / 0.820 / 0.988 | 0.232 | 1.046 / 0.972 / 1.190 |
| 0.925 | 0.860 | 0.901 / 0.893 / 1.063 | 0.226 | 1.102 / 1.029 / 1.213 |
| 0.900 | 0.838 | 0.968 / 0.955 / 1.132 | 0.220 | 1.152 / 0.876 / 1.261 |

[measured: reach3_run3_stdout.txt.] The healthy pelvis floors are 0.70 m for the T. rex and 0.147 m for compsognathus [measured: FACTS sheets], so every row is inside them. Below 0.9 L the envelopes break into disconnected branches (deep crouch). Those rows are in the JSON but are not used.

## 4. Comparison with the gait-plan bands (GQ-8)

| target | duty β | sweep needed β·S | T. rex: share of 0.614 L | compsognathus: share of 0.891 L | label |
|---|---|---|---|---|---|
| genuine walkers, low end S = 0.336 L | 0.5 / 0.6 | 0.168 / 0.202 L | 27 % / 33 % | 19 % / 23 % | [inferred: arithmetic on Table 2] |
| genuine walkers, high end S = 0.742 L | 0.5 / 0.6 | 0.371 / 0.445 L | 60 % / 72 % | 42 % / 50 % | [inferred: same] |
| compsognathus' measured walk: S 0.37 L, right-foot duty 0.49-0.52 (physics r1) | 0.49-0.52 | 0.18-0.19 L | 30 % | **21 %** | [measured: GAIT_QUALITY_PLAN_2026_09.md:93,358; inferred: product] |
| T. rex September hops: S 0.11-0.22 L, duty 0.48-0.66 | | ≤ 0.145 L | ≤ 24 % | | [measured: GAIT_QUALITY_PLAN_2026_09.md:357-358; inferred: product] |

Maximum stride S_max = E/β [inferred: arithmetic on Table 2]:

| T. rex configuration | S_max at β = 0.5 | S_max at β = 0.6 |
|---|---|---|
| flat pad at H0 | 1.23 L | 1.02 L |
| centred on the CoM | 1.17 L | 0.97 L |
| rolled single-support pose | 1.14 L | 0.95 L |
| knee ≤ 150 deg | 0.91 L | 0.76 L |
| rolling ±15 deg | 1.6 L | 1.33 L |

Compsognathus, flat: 1.78 L at β = 0.5 and 1.49 L at β = 0.6.

The top of the walker band, 0.742 L, fits in every T. rex configuration, including a knee kept at or below 150 deg. Nominal strides solved by IK, with the stance foot flat, the stride centred on the CoM and the swing foot lifted on a half-sine, stay inside every joint range for every band stride. [measured: reach3_run3_stdout.txt `stride ... out-of-range 0`]

Joint excursions over the cycle for S = 0.742 L, β = 0.6 [measured: stride_angles_stdout.txt]:

| leg | hip pitch | knee | ankle | toe (MTP) |
|---|---|---|---|---|
| T. rex stance | -13.1..+18.2 deg | -41.7..-24.8 deg (interior 135..152) | 67.3..103.2 deg | (passive digits) |
| T. rex swing | -16.1..+18.2 deg (34 deg excursion) | -41.7..-13.4 deg | 65.5..103.2 deg | (passive digits) |
| compsognathus stance | -8.5..+12.9 deg | -7.7..+0.3 deg | -1.0..+10.2 deg | -18.6..+15.6 deg |

Context [inferred: arithmetic]: the T. rex locomotion gate asks for 1.0 m/s (`configs/tyrannosaurus_rex/anatomical/locomotion.toml:139`), which is Froude 0.116.

- At S = 0.742 L (0.650 m) that speed needs a stride frequency of 1.54 Hz.
- At the flat β = 0.6 maximum (1.02 L) it needs 1.12 Hz.
- The literature stride 2.3·Fr^0.3·L, a report-only band (`GAIT_QUALITY_PLAN_2026_09.md:357`), gives 1.21 L at this Froude number. That needs β ≤ 0.5 or heel-toe rolling. Compsognathus' walk ran at 0.38 times that stride (same line).

## 5. Swing clearance

| quantity | T. rex | compsognathus | label |
|---|---|---|---|
| highest lift of the swing leg's lowest point (any leg geom), worst case over PEP..AEP at H0 | **0.190 m (0.217 L)** | **0.0586 m (0.240 L)** | [measured: reach3_run3_stdout.txt] |
| same, at the CoM's fore-aft position (mid-swing) | 0.201 m (0.229 L) | 0.075 m (0.307 L) | [measured: same] |
| same at H = 0.95 L / 0.90 L | 0.146 / 0.102 m | 0.046 / 0.034 m | [measured: same] |
| foot carried level 3 cm up: reachable fore-aft range | -0.196..+0.476 m, covers PEP..AEP (-0.134..+0.414) | -0.111..+0.181 m, covers -0.086..+0.134 | [measured: same `flat-foot swing`] |
| foot carried level 2 cm up | -0.177..+0.457 m | -0.105..+0.168 m | [measured: same] |
| L-scaled 3 cm (0.0342 L = 8.4 mm on compsognathus) | as at 3 cm | -0.096..+0.149 m, covers its range | [measured: same] |
| mid-swing joint offsets from home (foot level, 0.0342 L up, under the CoM) | hip -7.9, knee +11.4, ankle -4.5 deg | hip -3.3, knee +7.2, ankle -6.2, toe +2.4 deg | [measured: inspect_run3_stdout.txt] |
| largest action and torque that hold that swing pose | \|a\| 0.24; tau ≤ 0.8 % of forcerange | \|a\| 0.53; tau ≤ 3.6 % | [measured: same] |

The swing-envelope grid runs over the whole joint box at 1 deg for the T. rex and 2 deg for compsognathus. The T. rex digits are held at their spring rest angle; gravity sags them by about 0.3 deg [inferred: 0.22 kg × 0.1 m × g / 40 N.m/rad].

## 6. Ctrl reach and static torque in single support at H0

Method: one body weight acts at the pad centre, either vertically or through the CoM (an inverted-pendulum force with vertical component W). The other leg is at the mid-swing pose. Then tau = qfrc_bias - qfrc_passive - J^T F, ctrl = q + tau/kp, and the action comes from each species' own map inverted. The servo offset is tau/kp.

| readout | T. rex | compsognathus | label |
|---|---|---|---|
| largest \|tau\|/forcerange along the flat envelope, vertical force | 0.08 (rear and middle, ankle) to 0.18 (AEP, hip pitch) | 0.30 (middle) to 1.20 (far AEP, hip pitch) | [measured: reach3_run3_stdout.txt `profile`] |
| same, force through the CoM | 0.23-0.27 (hip roll) | 0.48-0.55 (hip roll, ankle) | [measured: same] |
| foot range (relative to CoM) with torque ≤ forcerange and ctrl inside the range, vertical / through CoM | -0.327..+0.298 L, the whole envelope, in both cases | -0.383..+0.404 L / -0.383..+0.471 L (envelope -0.405..+0.494 L) | [measured: same] |
| full-range AEP, vertical force | hip pitch 331 N.m (18 %, servo offset +15.8 deg); ankle ctrl 0.6 deg inside its stop; \|a\| 0.99 | knee and ankle are on their stops, so the needed ctrl falls outside the range | [measured: inspect_run3_stdout.txt] |
| practical AEP and PEP | \|a\| ≤ 0.93; ctrl margin ≥ 3.4 deg; ≤ 27 % | \|a\| ≤ 0.96; but 114 % of hip-pitch forcerange at the AEP with a vertical force (55 % through the CoM) | [measured: same] |
| foot under the CoM fore-aft, hip roll 0 | vertical: 8.7 % (ankle 117 N.m, offset +7.5 deg), \|a\| 0.06; through CoM: 25 % (hip roll -119 N.m, offset -11.4 deg), \|a\| 0.46 | vertical: 48 % (ankle), \|a\| 0.20; through CoM: 49 % (hip roll), \|a\| 0.34 | [measured: same] |
| rolled single-support pose: pad under the CoM in x and y, pelvis level at H0 | hip roll q +9.70 deg, tau -105 N.m (22 %), offset -10.0 deg, so **ctrl -0.33 deg (a -0.01)**; hip pitch 6.7 %, knee 0.9 %, ankle 8.7 %; largest \|a\| 0.10 | hip roll q +12.25 deg, tau -0.46 N.m (46 %), ctrl +10.9 deg, **a +0.80**; ankle 48 %; largest \|a\| 0.80 | [measured: rolled_support_torque_stdout.txt] |
| no-load actions (ctrl = q) for the band strides, S 0.742 L / 0.336 L at β 0.6 | \|a\| ≤ 0.42 (ankle; hip ≤ 0.32, knee ≤ 0.34) / ≤ 0.25 | up to 0.93 (toe; hip 0.66) / ≤ 0.69 | [measured: action_ranges_stdout.txt] |

Reading [inferred: from the table]:

- **T. rex.** Every static single-support pose on the sweep is ctrl-reachable with at least 73 % of forcerange spare.
  - What the T. rex pays is compliance. Single-support loads move the joints 5-16 deg off their commanded targets. In the rolled pose the soft hip-roll servo, commanded at home (a ≈ 0), sits 9.7 deg adducted under one body weight.
  - The no-load actions above under-state the loaded ones by these offsets, about 0.1-0.25 in action units on the T. rex.
- **Compsognathus.** It is torque-tight at the far ends of its sweep and action-hungry because of its soft-cubic map: 0.80-0.93 for a rolled stance or a top-band stride. It still walked on r1, using about ±0.09 L of sweep about its CoM.
- Doubling the external load (a 2 BW peak) would take the T. rex's worst row to about 0.5 of forcerange [inferred: the external part is linear in F].

## 7. Lateral (hip abduction and adduction) room

| quantity | T. rex | compsognathus | label |
|---|---|---|---|
| hip-roll range and action map | ±25 deg, linear, 25 deg per unit action | ±20.05 deg, soft-cubic | [measured: FACTS sheets; explore_stdout.txt] |
| hip roll that keeps the foot planted (pad flat in pitch, pad at home fore-aft, pelvis level at H0) | -22..+22 deg (no solution at ±23-25: the leg cannot reach the floor in range) | -20..+20 deg (the whole range) | [measured: inspect_run3_stdout.txt `lateral rows`] |
| pad-centre y from the midline over that roll range | **-0.483..+0.203 m (-0.551..+0.232 L)** | **-0.130..+0.036 m (-0.532..+0.147 L)** | [measured: same] |
| at hip roll 0 | -0.140 m (-0.160 L) | -0.047 m (-0.192 L) | [measured: same] |
| sensitivity | 0.014-0.017 m per deg | 0.0039-0.0042 m per deg | [measured: same, differences between 1-deg rows] |
| pad on the midline at hip roll / pad roll | about 9.2 deg / about 8.8 deg | about 11.3 deg / about 11.3 deg | [measured: same, interpolated between 1-deg rows] |
| pad under the CoM: hip roll / pad roll | 9.70 deg / 9.20 deg (FACTS: 9.72 deg with the pelvis fixed) | 12.25 deg / 12.24 deg (FACTS: 12.33) | [measured: reach3_run3_stdout.txt, rolled_support_torque_stdout.txt] |
| pad roll against hip roll | equal to within 0.4 deg everywhere. Neither species has an ankle-roll or foot-roll joint, so a laterally placed foot lands on its pad edge | equal | [measured: lateral rows] |
| swing-foot adduction at mid-swing (feet side by side, swing foot 0.0342 L up) before any leg touches the stance leg | **15 deg** (tibia on tibia); the swing pad is then at y -0.073 m, 0.084 L past the midline | **19 deg** (swing pad on stance metatarsus); swing pad at -0.028 m, 0.116 L past | [measured: reach3_run3_stdout.txt `swing adduction`] |
| gap to the stance foot when the swing pad passes over the midline | 17 mm at 10 deg (digit II on digit II) | 11 mm at 12 deg | [measured: inspect_run3_stdout.txt `adduction`] |

## 8. Collisions in a normal stride

Setup: hip roll 0; 41 phases; stance foot flat sweeping from AEP to PEP; swing foot level on a half-sine with a 0.0342 L peak. Distances are counted only for geom pairs that can collide under MuJoCo's filters: contype and conaffinity, `<exclude>` and the parent filter.

| minimum distance | T. rex | compsognathus | label |
|---|---|---|---|
| left leg against right leg, every band stride and the practical-extreme stride | 0.100 m (thighs), constant: the legs move in parallel sagittal planes | 0.050 m (thighs), constant | [measured: reach3_run3_stdout.txt] |
| leg against tail | ≥ 0.244 m in band strides (tibia vs tail_1); 0.199 m at E = 0.615 L | 0.0374 m (thigh vs tail_1, near the hip, independent of the stride) | [measured: same] |
| leg against pelvis geoms (torso, belly, ribcage) | ≥ 0.098 m (tibia vs belly); 0.095 m at 0.615 L | ≥ 0.066 m in band strides; 0.052 m at 0.817 L | [measured: same] |
| swing-leg hip extension from home to the range end, against the tail | 0.272 m at 0 deg falling to 0.110 m at 65 deg | 0.037 m constant up to the 40 deg stop | [measured: inspect_run3_stdout.txt `leg-tail ext`] |
| pairs filtered out | the T. rex thighs against tail_1 (excluded in the MJCF, which overlaps them by 18.8 mm at home: `tyrannosaurus_rex_anatomical.xml:552-563`); thighs against pelvis geoms (parent filter) | thigh against pelvis geoms only (parent filter); every leg-tail pair is live | [measured: reach3_run3.json `collision_filter`] |

So the feet do not collide with each other or with the tail in a normal stride, on either species. Leg-leg contact only becomes possible when the swing hip adducts beyond 15 deg (T. rex) or 19 deg (compsognathus) at mid-swing (§7). [measured]

## 9. Static, dynamic and actuator limits: what this stage covers

| limit | covered here? | finding |
|---|---|---|
| kinematic reach and clearance | yes | Not limiting on either species (§2-5). |
| quasi-static walk (CoM over the stance foot) | geometry only | The stance pad can be placed under the CoM: hip roll 9.7 deg, pad rolled 9.2 deg onto its edge. Its fore-aft sweep in that pose is about 0.57 L. Its static torques are 22 % of forcerange or less, with ctrl inside the range (§6). Whether this pose balances is not assessed. |
| dynamic walk (CoP/ZMP inside the stance foot) | no | Not assessed. A failed static single-support check would not prove infeasibility. |
| actuator: static torque and ctrl range | yes | Not limiting on the T. rex: ≤ 27 % of forcerange, every ctrl inside its range. The cost is large servo offsets: up to 16 deg of hip pitch and 10-12 deg of hip roll. |
| actuator: tracking, bandwidth, 10 Hz filter, swing timing | no | Inputs for that stage [measured: stride_angles_stdout.txt]: a top-band stride swings the hip through about 34 deg, and the swing knee travels 28 deg (-41.7 to -13.4). |

## 10. Limits of this analysis

- **Pelvis held level.** The pelvis is fixed at the statue pitch and one height for both feet, with no vertical or pitch motion within the stride. Pelvis pitch and vaulting would only add reach [inferred: geometry].
- **Planar sagittal envelopes.** They use hip roll 0. The rolled single-support case uses 3-D FK continuation, and its compsognathus figure under-reads because the ankle was scanned in 5-deg steps (0.850 L at roll 0 against 0.891 L from the grid). Only the rolled/unrolled ratio of about 0.93-0.94 is relied on.
- **T. rex foot assumptions.**
  - In toe-off the passive digits are assumed to flex onto the floor. The flex needed is at most 15 deg against a ±37.5 deg range.
  - Vaulting over the digit tips is excluded: one body weight there needs about 160 N.m about a digit hinge whose spring gives only 26 N.m (FACTS).
- **Grid tolerance.** Selecting by tolerance inflates the grid E by about 0.01 L. This is quantified in tol_check_stdout.txt, and the headline numbers are the converged ones.
- **Static torques are quasi-static.**
  - They assume one body weight, either vertical or through the CoM. Real walking peaks of 1.1-1.5 BW scale only the external part.
  - The double-support ends are scored as if in single support, which is conservative.
  - The root residual (toppling moment) is ignored; that belongs to the stability stages.
- **Box-box distances.** `mj_geomDistance` (MuJoCo 3.10.0) returned 0 for some well-separated box-box pairs, the two plantar pads (distcheck2_stdout.txt). Every zero on a box-box pair was replaced by a separating-axis lower bound when that bound was positive: 2 times on the T. rex and 14 on compsognathus in run 3. The leg-leg minima quoted come from capsule pairs.
- **Strides near the singularity.** The practical-extreme T. rex stride has an IK residual of 1.2 mm at a phase near the straight-knee singularity. It is used for collision distances only.
- **Positive control on a different plant.** Compsognathus' measured walk was on physics r1 with a linear residual map. This plant is r2 with a soft-cubic map (FACTS). The r2 kinematic ranges are those of r1, so its reach calibration holds [measured: FACTS, plant_versions.toml note 15 lists no range change].

## 11. Slips

1. **Early commands logged late.** About a dozen read-only commands (FACTS reads, MJCF and doc reads, greps) ran before CMDLOG.txt was set up for literal capture, between 21:52Z and 21:58Z. They are listed after the fact and flagged in CMDLOG.txt, together with one Read-tool view of reach.py.
2. **explore.py exclude decoding.** explore.py decoded `exclude_signature` wrongly: it subtracted 1, and the printed pairs are nonsense. The decoding was not used. reach2 and reach3 decode it correctly, and the filtered pairs match the MJCF excludes.
3. **explore.py statue too short.** explore.py read the statue after only 300 steps and got a pelvis pitch of 1.35 deg, still settling. From reach.py on, the statue runs 1000 steps and reads 0.94 deg (mean 0.955, which matches the env comment). explore.json's statue is superseded.
4. **reach.py run 1 errors, all superseded.** The run 1 outputs (reach_run1.json and its stdout) are kept, but every figure from them is superseded. Their errors were:
   - envelopes that took in disconnected branches at low H;
   - a key-mismatch bug that dropped the swing clearance at H0;
   - minimum distances that counted non-collidable pairs, for example -0.019 m thigh-tail_1 and -0.111 m thigh-torso;
   - spurious zero pad-pad distances;
   - an out-of-range IK at the extreme stride.
5. **reach2.py knee angle.** reach2.py took the knee interior angle from an arccos, which cannot exceed 180 deg. reach3 uses the signed angle. reach2's practical envelope was unaffected, because its 5-deg margin already excludes hyperextension.
6. **reach3 run 1 stride seeds.** reach3 run 1 seeded the band-stride IK from the grid, which put 15-23 of 41 compsognathus phases out of range. Run 2 went back to home-seeded continuation for band strides and per-phase grid seeding for the extreme stride. Run 3, which adds joint-angle recording, reproduces run 2 apart from timings.
7. **Grid E figures biased.** The grid E values reported in the stdout logs carry the +0.01 L tolerance bias, which was found afterwards with tol_check.py. Table 2 gives the converged values.
8. **Empty diff file.** tmp/diff_run2_run3.txt is empty because its filter removed every line. The meaningful run 2 / run 3 comparison is the timestamp-stripped diff recorded in CMDLOG.txt.
9. **Draft errors fixed during self-review.** The first draft of this file:
   - called the T. rex heel strike ankle-limited (the per-φ rows show the geometric reason given in Table 2);
   - gave the lateral sensitivities as 0.0151-0.0165 and 0.0042 m per deg (the 1-deg rows give 0.014-0.017 and 0.0039-0.0042);
   - cited the wrong MJCF lines for the tail_1 excludes.
   All three were corrected before delivery.

## 12. Files (all in this folder)

- Scripts:
  - explore.py: geometry and statue readout.
  - reach.py, reach2.py, reach3.py: three versions of the analysis. **reach3.py is the final one.** Patches are in tmp/patch1-5.py.
  - distcheck.py, distcheck2.py: box-box distance checks.
  - tol_check.py: tolerance convergence and the T. rex closed form.
  - rolled_support_torque.py.
  - action_ranges.py.
  - inspect_final.py: JSON readout.
- Outputs:
  - **reach3_run3.json** and **reach3_run3_stdout.txt**: the final run.
  - inspect_run3_stdout.txt, tol_check_stdout.txt, rolled_support_torque_stdout.txt, action_ranges_stdout.txt, stride_angles_stdout.txt.
  - Superseded runs: reach_run1.*, reach2_run1.*, reach3_run1.*, reach3_run2.*, plus inspect_final_stdout.txt and inspect_run2_stdout.txt (both read from run 2, which is identical to run 3).
  - explore.json and explore_stdout.txt; distcheck*_stdout.txt.
- CMDLOG.txt: literal commands with UTC times.
- Reproduce with:

  ```
  TMPDIR=<this folder>/tmp OMP_NUM_THREADS=1 MUJOCO_GL=disable nice -n 10 <optB venv python> -I -B reach3.py <export> reach3_run3.json
  ```

  This takes about 75 s and is deterministic.
