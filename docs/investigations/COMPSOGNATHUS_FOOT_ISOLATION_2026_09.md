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

Tyrannosaurus Rex offers three relevant precedents:

| Feature | Tyrannosaurus Rex | Current Compsognathus Longipes |
|---|---|---|
| Toe mechanism | Passive articulated toes; toe actuators removed during the earlier investigation | One powered hinge moves each entire foot |
| Action filter | 10 Hz, declared in the plant interface | Disabled |
| Stance load shaping | Bilateral-support weight 0.6 and load-balance weight 0.3 | No equivalent bilateral terms |
| Stance action smoothness weight | 2.0 | 0.02 |

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

All replays use the same checkpoint and frozen VecNormalize statistics. The
canonical environment and both artifacts must pass plant-identity validation
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

## Recommendation and next training experiment

1. Carry forward the **fixed sole plus passive digits**, with the **2 mm distal
   envelope correction**, as a mechanical candidate. Keep coupled and independent
   digits as alternatives: both passed the tested scenes; independence adds local
   adaptation and higher sole share but also four more DOFs across the animal.
2. Test **10 Hz filtering of all actions during new training** as a separate
   factor. The replay evidence favors it over a foot-only filter. Do not turn it
   on while silently resuming the old checkpoint: filtering changes action meaning.
3. Use matched training seeds and budgets for four primary arms: current plant;
   filter only; compliant sole/toe plant only; and both. Keep reward and reset
   settings fixed initially. Compare gate qualification, pad share, bilateral
   weight support, pitch spectrum, cap occupancy and mechanical work.
4. Then calibrate load-balance rewards, passive stiffness/damping and ankle/foot
   gains separately. Add pushes, slopes, obstacle positions and terrain not used
   to select the candidate. Establish convergence over multiple independent
   training seeds before changing the notebook default.

Training integration must update the SB3/MJX action and observation mappings,
body-ground exclusions for moving digits, all foot sensor groups, home/reset
logic, canonical plant revisions/manifests and parity tests. Existing compiled-
plant validation correctly rejects these derived models. The fixed-tail robot
has different mechanics and needs its own experiment; this research uses the
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
- [MuJoCo touch sensors and actuators](https://mujoco.readthedocs.io/en/stable/XMLreference.html):
  moving digit bodies need their own sensor coverage; position gains and force
  limits are distinct from measured actuator bandwidth or electrical power.
