# Compsognathus Longipes: sole isolation, toe compliance and action bandwidth

Date: September 10, 2026. Base: `9557e97`. Status: **research candidates; no production plant change**.

The strongest direction is a **load-bearing sole attached to the metatarsus,
passive toe articulation, adequate distal-leg clearance, and a separately tested
action-bandwidth limit**. Fixing the entire foot was insufficient. Independent
toes improve local conformation and sole loading, but a simpler coupled-toe
mechanism matched their survival results in this small panel.

This PR provides an opt-in MJCF builder, mechanical and checkpoint replay
probes, regression tests, and [208 trial records](data/compsognathus_feet_2026_09/manifest.json).
The anatomical plant, robot plant, notebook, rewards, gates and registered
checkpoint identities retain their existing behavior. These derived XMLs are
**not registered training environments** and cannot accept the existing policy.

The follow-up [learned-balance study](COMPSOGNATHUS_LEARNED_BALANCE_STUDY_2026_09.md) now provides an opt-in Colab notebook that loads this PR directly, reads existing Drive experiments, and compares filtering × stance rewards on the current feet. This prioritizes learned quiet bilateral stance before additional mechanical redesign. The mechanical candidates and evidence below remain separate hypotheses.

## What problem is being isolated?

The canonical anatomical foot places its plantar pad and three digit capsules
on a single motorized `r_toe`/`l_toe` hinge. A command at that joint rotates the
whole contact surface. It is not three independently controlled toes. Passive
springs currently occur in three distal tail hinges, not the leg joints.

The completed run `20260909_162812` already passed its saved 40-episode stance
gate at the selected 10.85M-step checkpoint: reward **2801.62**, full horizon
**40/40**, unsupported duty **1.306%**, upper bound **1.409%**. Its bilateral
control-window duty was only **1.403%**. Passing the supported-stance gate does
not establish quiet double support or load on the plantar pads.

Tyrannosaurus Rex offers several relevant precedents:

| Feature | Tyrannosaurus Rex | Current Compsognathus Longipes |
|---|---|---|
| Toe mechanism | Passive articulated toes; toe actuators removed during the earlier investigation | One powered hinge moves each entire foot |
| Passive leg compliance | Springs and damping at hip pitch, hip roll, knee and ankle | No leg springs; much smaller passive damping |
| Action filter | 10 Hz, declared in the plant interface | Disabled |
| Stance load shaping | Bilateral-support weight 0.6 and load-balance weight 0.3 | No equivalent bilateral terms |
| Stance action smoothness weight | 2.0 | 0.02 |
| Home-pose objective | Narrow and broad Gaussian rewards for leg pose, plus settled-tail targets | Quadratic error cost over actuated joints |

See [Tyrannosaurus Rex stance config](../../configs/trex/stance.toml),
[passive-toe investigation](TREX_STAGE1_PASSIVE_TOES_RUN_2026_08.md), and
[action-filter semantics](../../environments/shared/action_filter.py).
Reward coefficients are not scale-independent calibration values to copy blindly.

## Mechanical comparisons

Four builders retain the same 1 kg animal, neutral foot geometry, contact
materials, remaining motor gains/caps and powered home targets:

| Variant | Sole | Digits | `nq / nv / nu` |
|---|---|---|---|
| `baseline` | Original powered hinge | Rigidly attached to sole | 24 / 23 / 14 |
| `fixed` | Fixed to metatarsus | Rigidly attached to sole | 22 / 21 / 12 |
| `coupled` | Fixed to metatarsus | Three digits on one passive hinge per foot | 24 / 23 / 12 |
| `independent` | Fixed to metatarsus | One passive hinge per digit | 28 / 27 / 12 |

Independent hinges use **0.15 N·m/rad**, **0.002 N·m·s/rad**, armature
**1e-6 kg·m²**, limits **−0.3 to +0.4 rad**, and zero spring reference.
The coupled hinge uses three times each spring, damping and armature value,
matching their nominal totals. This does not make the two mechanisms dynamically
equivalent: their moving masses and contact constraints differ.

Moving digits have their own body-local touch sensors. The foot total sums
the sole and digit sensors. Only intentional overlapping digit roots receive
sibling collision exclusions; opposing feet and body-ground contact remain active.

Each variant ran two reset seeds, 4010–4011, on nine scenes: flat ground;
3 mm and 6 mm bumps under either middle digit; 3 mm bumps under either outer
digit; and 3 mm bumps under either pad. The pad cases are comparison controls,
not assumed failures. Common original joint positions receive matched ±0.01 rad
noise by name. Added joints start at zero. Obstacle scenes start with a
`height + 1 mm` settling drop, identically for all mechanisms. The protocol uses
20-second horizons, 4-second settling windows, and every-physics-step contact checks.

### Original distal-leg envelope: 72 trials

| Variant | Full 20 seconds, all scenes | 3 mm middle bumps | 6 mm middle bumps | Flat-ground pad share |
|---|---:|---:|---:|---:|
| Baseline | 9/18 | 0/4 | 0/4 | 44.0% |
| Fixed whole foot | 10/18 | 0/4 | 0/4 | 44.0% |
| Coupled passive digits | 15/18 | 4/4 | 1/4 | 62.9% |
| Independent passive digits | 15/18 | 4/4 | 1/4 | 76.0% |

Both compliant variants handle the smaller middle-digit obstacle; welding the
foot alone does not. Independent toes deflect locally: the obstructed middle
digit bends about **3.2°**, while its neighbors stay within **0.04°** of zero.
All variants survive the four pad-bump trials. There is no evidence here for a
universal advantage on every obstacle type.

### Clearance failure and a separate geometry correction: another 72 trials

The six compliant-foot failures on 6 mm bumps occur **40–42 ms** into landing.
Their first disallowed contact is a metatarsal capsule, with **zero normal force**
at that instant, at distances from approximately **−0.095 to +0.099 mm**. The
strict judge latches contact-list entries, including entries within MuJoCo's
margin; these are clearance/contact-rule failures, not demonstrated collapses.
The original capsule ends only about 1 mm above the sole plane at home.

The follow-up raises each capsule's distal endpoint **2 mm in local Z**.
It retains capsule mass, leg linkage, joint anchors, sole footprint and toe
positions. Capsule inertia changes and whole-animal home COM rises **0.036 mm**;
this is explicitly a separate geometry arm, not a mass-identical inertia claim.
The same strict body-contact termination remains active.

| Variant, with clearance correction | Full 20 seconds | 6 mm middle bumps |
|---|---:|---:|
| Baseline | 9/18 | 0/4 |
| Fixed whole foot | 10/18 | 0/4 |
| Coupled passive digits | **18/18** | **4/4** |
| Independent passive digits | **18/18** | **4/4** |

The corrected independent middle digit bends about **6.4°** over the 6 mm bump;
neighbors remain nearly neutral. Across both mechanical panels, summed touch
forces match directly measured foot contact normal forces within **8e-15 N**.
This verifies sensor accounting in these scenes, not sensor realism or accuracy
on arbitrary terrain. Pad share means integrated contact-normal force share;
it is not a vector ground-reaction-force measurement.

## Control comparisons on the existing learned policy

All replays use deterministic policy predictions, the same checkpoint and frozen
VecNormalize statistics. The canonical environment and both artifacts must pass
plant-identity validation
before an external action intervention is applied. No policy weights are updated
and no altered plant is mislabeled as a compatible checkpoint.

First screen: four fresh reset seeds, 4100–4103; 10 variants, **40 trials**.
Filters use the repository's discrete RC rule, seeded with the first command.
“Mean hold” averages only commands already seen in seconds 1–4 and ramps toward
that reference over seconds 4–5. “Home hold” similarly hands off to zero residual.

| Intervention | Full horizon | Mean reward, all episodes | Unsupported windows after settling |
|---|---:|---:|---:|
| Baseline | 4/4 | 2777.8 | 1.406% |
| Feet, 10 Hz | 4/4 | 2773.0 | 1.406% |
| Feet, 5 Hz | 4/4 | 2764.4 | 1.406% |
| Feet, 2 Hz | 4/4 | 2745.8 | 1.938% |
| Ankles and feet, 10 Hz | 4/4 | 2770.2 | 0.719% |
| All actions, 10 Hz | 4/4 | 2817.5 | 0.250% |
| All actions, 5 Hz | 1/4 | 735.6 | Not comparable: 3 fail before settling |
| Feet held at causal mean | 4/4 | 2743.6 | 4.406% |
| Ankles and feet held at causal mean | 0/4 | 755.8 | Not comparable: all fail at 5.60–5.76 s |
| Feet handed to home command | 4/4 | 2688.5 | 5.094% |

The actual demeaned foot-target FFT peak is around **3 Hz** in baseline replays.
Do not interpret an earlier “effective frequency” statistic as the dominant
spectral peak or required feedback bandwidth. The current strategy needs active
ankle feedback; freezing it fails, and aggressive filtering can fail during reset.

Fresh confirmation seeds 4110–4117: three arms, **24 trials**. All finish 20 seconds:

| Mean metric, seconds 4–20 unless noted | Baseline | All actions, 10 Hz | Ankles and feet, 10 Hz |
|---|---:|---:|---:|
| Reward, entire episode | 2811.8 | 2837.2 | 2805.2 |
| Unsupported control windows | 1.297% | **0.422%** | 0.563% |
| Positive actuator mechanical work | 37.06 J | **11.40 J** | 37.72 J |
| Pelvis angular-velocity RMS | 0.836 rad/s | **0.557 rad/s** | 0.814 rad/s |
| Dominant-foot load sign changes | 7.54/s | **5.34/s** | 7.35/s |
| Foot actuator force-cap occupancy | 66.84% | 60.31% | 59.09% |
| Plantar-pad load share | 0.076% | 0.160% | 0.103% |
| Bilateral support, simultaneous physics samples | 8.77% | 8.21% | 9.91% |

All-action 10 Hz filtering reduces positive actuator work about **69%** in this
panel. This is `integral(sum(max(torque * velocity, 0)))`, not battery energy;
motor efficiency, holding current, thermal limits and regeneration are absent.
It improves several motion metrics but does **not** restore useful sole loading
or quiet bilateral stance. Less saturation alone is not the selection criterion.

Unsupported control windows use per-foot minima over ten 2 ms substeps and the
shared 0.1 N diagnostic threshold. They are conservative window diagnostics, not
necessarily periods when both feet are simultaneously airborne. Simultaneous
bilateral physics samples are reported separately. Failed episodes have blank
post-settle metrics when no samples exist; they are not treated as zero duty.

## Additional lessons from Tyrannosaurus Rex

The following findings come from the current model/reward code and the earlier
Tyrannosaurus Rex investigations. They add interpretation and proposed work;
they do not add new Compsognathus trials to the 208 records above.

### Passive leg springs are a separate design difference

A compiled-model audit confirms these parameters on both sides:

| Joint | Tyrannosaurus Rex stiffness, N·m/rad | Tyrannosaurus Rex passive damping, N·m·s/rad | Compsognathus stiffness, N·m/rad | Compsognathus passive damping, N·m·s/rad |
|---|---:|---:|---:|---:|
| Hip pitch | 40 | 45 | 0 | 0.01 |
| Hip roll | 40 | 45 | 0 | 0.01 |
| Knee | 40 | 45 | 0 | 0.01 |
| Ankle | 40 | 45 | 0 | 0.01 |

These are **passive joint parameters**, distinct from position-actuator gains
and derivative feedback. The Tyrannosaurus Rex hip-pitch, knee and ankle spring
references are the standing-pose values (−2.6°, −24.8° and 76.9° in its
degree-based XML); hip roll is zero. The geometric joint reference, spring
reference and commanded home target have different meanings. See the
[Tyrannosaurus Rex XML](../../environments/trex/assets/trex.xml) and
[Compsognathus XML](../../environments/compsognathus/assets/compsognathus.xml).
The experimental Compsognathus variants add springs at the digits only.

Test weak leg compliance as a separate hypothesis, with spring references
calibrated around the loaded standing equilibrium. Select stiffness and damping
using Compsognathus joint inertia, gravity loading and response times; mass
scaling alone does not preserve dynamics. Copying the Tyrannosaurus Rex numbers
would ignore the different geometry, inertia, actuator authority and control
interval. Springs can introduce rebound or resist deliberate stepping, so
improved quiet stance must also survive disturbance and foot-lift tests. A
zero-residual or fixed-command experiment still has active position servos;
its success is not evidence of unpowered standing.

### Fixing toes can move the exploit to another joint

The [passive-toe investigation](TREX_STAGE1_PASSIVE_TOES_RUN_2026_08.md) and
[subsequent narrow-tolerance run](TREX_STAGE1_NARROW_TOLERANCE_RUN_2026_08.md)
show that passive toes alone did not establish quiet stance. The later policy
used a knee-locked crouch with ankle pumping. Diagnose actual joint motion,
joint-limit contact and torque throughout the legs so that suppressing one
oscillation channel does not hide its replacement elsewhere.

The [gate-pass run](TREX_STAGE1_GATE_PASS_RUN_2026_08.md) combined the 10 Hz
training-time command filter with action-saturation shaping, settled-tail
home targets and a broad leg-home gradient. These changes were introduced
together. That history motivates the Compsognathus experiments, but it cannot
isolate which component caused the improvement.

One reward fix does **not** transfer directly. The current
[Compsognathus reward](../../environments/compsognathus/envs/compsognathus_env.py)
uses `-home_pose_weight * mean(square(joint_error))` over actuated joints.
Its gradient does not disappear far from home as the earlier narrow Gaussian
did. Audit joint coverage, settled targets, ranges and relative weights before
adding another home-pose term. Current support-conditioned alive, posture and
height rewards also do not explicitly price bilateral load sharing in the same
way as Tyrannosaurus Rex.

### Replication and deterministic motion matter

The [seed-43 report, including its August 16 addendum](TREX_STAGE1_SEED43_REPLICATE_2026_08.md)
records three independent training seeds on the same Tyrannosaurus Rex plant
and configuration:

| Training seed | Stance verdict | Full horizon | Unsupported duty | Duty upper bound | Reported command AC RMS |
|---|---|---:|---:|---:|---:|
| 42 | Pass | 40/40 | 0.48% | 0.80% | 0.135 |
| 43 | Fail | 37/40 | 5.97% | 7.47% | 0.329 |
| 44 | Pass | 40/40 | 0.69% | 1.17% | 0.132 |

This is **two observed passes in three runs**, not an established success
probability. Additional reset seeds for one saved policy are not independent
training replicates. Use at least three independent training seeds for
promising Compsognathus candidates as an initial replication budget, and report
every outcome rather than selecting only the best seed.

The Tyrannosaurus Rex report did not directly re-analyze the learned policy
standard-deviation trajectory. Residual command variation alone does not prove
its exploration-noise hypothesis. Compsognathus bouncing is already present in
the **deterministic** replays in this PR, so sampled exploration noise cannot
be its sole explanation. A learned closed-loop oscillation, contact response or
posture strategy remains possible. Compare deterministic and stochastic runs,
recording policy means, sampled actions, applied commands and learned action
standard deviation separately. Setting the entropy coefficient to zero does not itself
force that standard deviation to zero; a new noise penalty needs its own evidence.

## Diagnose motion and load before optimizing their summaries

“Bouncing feet” can describe different behavior. Add synchronized physics-step
traces that distinguish the following, using the same timestamps as the commands:

| Possible behavior | Measurements that distinguish it |
|---|---|
| Whole-body hopping | Pelvis/COM height and vertical velocity, both feet's height and vertical load, simultaneous loss of support |
| Heel/toe rocking | Foot pitch and angular velocity, pad versus digit contacts, contact positions and center of pressure |
| Alternating weight transfer | Left/right vertical load as fractions of body weight, simultaneous bilateral support and pelvis roll |
| Joint or actuator limit cycling | Raw and applied commands, actual joint angles/velocities, actuator force, joint-limit contacts |

Compute ground-reaction vectors from the contact frames for vertical-load and
center-of-pressure diagnostics. The current scalar touch/normal-force sums do
not provide those quantities on arbitrary terrain. Mark center of pressure
undefined when vertical support is negligible rather than dividing by near zero.
Pad load share is a geometry-dependent diagnostic, not an objective to maximize
to 100%; a stance that cannot unload a foot is a poor basis for locomotion.

Keep three different limit metrics separate: **normalized command-limit
occupancy, actual actuator force-cap occupancy, and joint hard-stop contact**.
The Tyrannosaurus Rex reports' “zero saturated” summaries refer to normalized
command/DC thresholds. They are not directly comparable to the **66.84%**
physics-step actuator force-cap occupancy reported here. Low command saturation
does not establish unused torque capacity.

For species comparisons, express loads as fractions of body weight and durations
in seconds. The same absolute 0.1 N support threshold represents different
fractions of body weight. Preserve the official gate while adding normalized
diagnostics, and keep simultaneous physics-step support separate from the
conservative per-foot minima over control windows.

## Recommended validation and training sequence

The remaining work below is **proposed, not completed**. The first choice remains
a fixed sole with passive digits and the 2 mm clearance correction, with global
10 Hz filtering tested as an independent training factor. Neither larger motors
nor a wider foot is yet established as necessary by these measurements.

### Focused screening before long training runs

| Priority | Experiment | Decision it should resolve |
|---|---|---|
| 1 | Record the synchronized motion, load and limit traces above on matched baseline and candidate scenes | Identify hopping, rocking, load transfer or a joint-level cycle before choosing a fix |
| 2 | Audit reward ordering for quiet double support, rocking, single support and airborne motion; compare component returns over equal durations and retain termination costs | Check that desired stance is preferred, and whether scaled bilateral shaping is needed without removing recovery incentives |
| 3 | Calibrate the loaded ankle/leg response, changing actuator gains, passive damping and weak leg springs separately | Reduce tracking error and rebound while retaining torque headroom, reset survival and perturbation response |
| 4 | Compare corrected coupled and independent digits; then test modest sole width, length and placement changes separately | Decide whether independent articulation or footprint changes justify their complexity; retain matched mass/controller where possible and record any COM/inertia changes |
| 5 | Compare 1 ms and 2 ms physics with the same 50 Hz control interval, then small contact-softness, damping and friction sweeps | Check that rankings persist across reasonable numerical/material settings rather than a single contact configuration |
| 6 | Use fresh obstacle positions, slopes, friction conditions and lateral/forward pushes; later test deliberate foot lifting | Reject candidates that gain quiet stance by sacrificing recovery or stepping capability |

Loaded calibration should measure settling time, overshoot, foot pitch, actual
torque clipping and joint-limit contact with identical home-controller targets
and disturbances. Record passive spring/damping torque separately from actuator
torque so added compliance is not mistaken for increased motor authority. Do not
add all promising gain, spring and reward changes to the first training comparison.

For numerical sensitivity, use 20 substeps at 1 ms versus 10 at 2 ms so the policy
still updates every 20 ms. Keep horizons, filter constants, perturbation timing
and diagnostic windows fixed in seconds, and record the changed plant identity.
The current `solref="0.006 1"` at a 2 ms timestep already satisfies MuJoCo's
guidance that the positive-format contact time constant be at least twice the
timestep; it is not evidence of an obvious violation. The
[MuJoCo solver-parameter guidance](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
motivates a sensitivity test, not arbitrary softening until the model passes.
Preserve strict non-foot-contact checks and report margin contacts separately;
do not relabel the earlier zero-force metatarsal contacts as successful trials.

Use mass- and size-aware disturbance descriptions, such as force/body weight,
impulse/mass and obstacle height/leg length, along with their physical units.
Do not transplant Tyrannosaurus Rex push magnitudes or foot dimensions directly.
Keep candidate-selection scenes separate from later confirmation scenes.

### Four matched training arms

Choose and record one corrected compliant-foot candidate before this comparison.
Coupled and independent digits both passed the existing mechanical panel;
independence adds local adaptation and higher sole share, but also four more
DOFs across the animal. If both remain contenders, add explicit extra arms
rather than pooling different mechanisms under one label.

| Arm | Foot mechanism | Global action filter |
|---|---|---|
| A: control | Current canonical powered foot | Off |
| B: filter only | Current canonical powered foot | 10 Hz |
| C: mechanism only | Selected fixed sole/passive digits with 2 mm clearance correction | Off |
| D: combined | Same mechanism as C | 10 Hz |

Start new training runs with matched training seeds, environment-step budgets,
reward settings, resets, remaining actuator gains, evaluation seeds and
checkpoint-selection rules. Match common-joint reset perturbations by name;
declare initialization for any added passive joints. Arm C is a selected
mechanical package; this matrix separates that package from filtering, while
the earlier clearance ablation separates its geometry correction. It does not
independently estimate every toe parameter's contribution.

Use the existing 40-episode stance gate for qualification and report the full
gate verdict, time/steps to qualification and failures for every seed. Supplement
it with simultaneous support, load distribution, foot/COM motion spectra,
command limits, actuator clipping, joint stops and positive mechanical work.
Compare failed and surviving episodes explicitly rather than averaging only
the successful trajectories. Keep reward rails fixed for this matched comparison;
if a later plant or reward change requires new home/theoretical reference values,
recompute and version the rails and affected percentage thresholds before that
study. A raw reward increase across different reward definitions is not progress.

Predeclare acceptable tradeoffs before selecting a winner: reproducible gate
qualification, lower unwanted motion/work or clipping, and no material loss of
disturbance tolerance or ability to unload a foot. The existing measurements do
not yet justify a numerical pad-share or bounce-frequency acceptance threshold.
Replicate promising arms across independent training seeds; fresh evaluation
seeds alone do not establish convergence. After isolating mechanism and filter
effects, evaluate reward, gain and passive-spring changes separately, followed
by recovery and locomotion qualification. Establish those results before changing
the notebook default.

Training integration must update the SB3/MJX action and observation mappings,
body-ground exclusions for moving digits, all foot sensor groups, home/reset
logic, canonical plant revisions/manifests and parity tests. Existing compiled-
plant validation correctly rejects these derived models. Filtering must be part
of the declared training interface with consistent reset/state handling; do not
silently resume an existing checkpoint under changed action semantics or mechanics.
The fixed-tail robot has different mechanics and needs its own experiment;
this research uses the
**anatomical proxy only**.

The 18-trial mechanical panels use two reset seeds and a powered home controller,
not learned walking. The replay panel uses one trained policy, not independent
training replicates. No new full curriculum gate, recovery qualification,
hardware validation or fabrication-ready toe design is claimed.

## Reproduce

Core mechanical experiments need `pip install -e ".[test]"`; replay additionally
needs `.[train]` and the checkpoint's own normalization and captured stage config.
Use the pinned MuJoCo version. Exact artifact hashes and runtime versions are
in the [data metadata](data/compsognathus_feet_2026_09/manifest.json).

```bash
python -m environments.compsognathus.scripts.probe_feet build --output /tmp/comps-feet
python -m environments.compsognathus.scripts.probe_feet mechanical \
  --output /tmp/comps-mechanical --seeds 4010 4011
python -m environments.compsognathus.scripts.probe_feet mechanical \
  --output /tmp/comps-clearance --seeds 4010 4011 --distal-clearance .002
python -m environments.compsognathus.scripts.probe_feet replay \
  --output /tmp/comps-replay-screen --seeds 4100 4101 4102 4103 \
  --model /path/to/robust_best_model.zip \
  --normalization /path/to/robust_best_model_vecnorm.pkl \
  --stage-config /path/to/stage_config.json
python -m environments.compsognathus.scripts.probe_feet replay \
  --output /tmp/comps-replay-holdout --seeds 4110 4111 4112 4113 4114 4115 4116 4117 \
  --variants baseline all_lp10 ankles_feet_lp10 \
  --model /path/to/robust_best_model.zip \
  --normalization /path/to/robust_best_model_vecnorm.pkl \
  --stage-config /path/to/stage_config.json
pytest environments/compsognathus/tests/test_foot_experiments.py
```

Each completed panel writes JSON, trial-level CSV and metadata; mechanical mode
also emits isolated XMLs. CSV nested metric names use dots; blank means unmeasured.
The recorded checkpoint contains Python 3.13 schedule closures. The loader
replaces learning-rate/clip schedule metadata for diagnostic Python 3.12 inference,
without changing weights or normalization. Local baseline rewards differ slightly
from the saved Drive panel, so all intervention comparisons use paired **local**
baselines; bitwise reproduction of the original hosted run is not claimed.

## External research informing the experiments

- [Piazza et al., SoftFoot](https://arxiv.org/abs/2401.05318): comparisons of an
  adaptive passive foot with rigid/compliant alternatives motivate matched
  footprints and obstacle-specific tests. Its tendon/pulley mechanism is more
  elaborate than these hinge proxies; its hardware results do not validate them.
- [Mysore et al., CAPS](https://arxiv.org/abs/2012.06644): training-time policy
  regularization can reduce high-frequency actions. It supports investigating
  smooth control during learning; it does not specify a Compsognathus cutoff.
- [MuJoCo contact model](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact):
  contact-list entries, soft contact and normal-force constraints must be
  distinguished when interpreting clearance failures.
- [MuJoCo solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters):
  contact stiffness/damping and timestep interact; sensitivity tests must retain
  the control interval and distinguish numerical robustness from physical validation.
- [MuJoCo touch sensors and actuators](https://mujoco.readthedocs.io/en/stable/XMLreference.html):
  moving digit bodies need their own sensor coverage; position gains and force
  limits are distinct from measured actuator bandwidth or electrical power.
