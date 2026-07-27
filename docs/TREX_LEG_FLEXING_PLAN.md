# T-Rex Stage-1 Leg Flexing — Remediation Plan

**Status:** Proposed — not implemented.
**Date:** 2026-07-27
**Motivating runs:** `20260725_194916` (w=0.1), `20260726_191730` (w=0.7),
`20260727_130726` (w=2.0), all T-Rex stage 1, PPO, seed 42.

Stage-1 policies visibly flex their legs in an unnatural way. Three
`smoothness_weight` escalations reduced the effect without removing it.

**The diagnosis moved once during scoping.** The first version of this plan
blamed the action-to-joint-angle mapping and proposed a per-stage
`action_scale`. Measuring the limb geometry afterwards showed the model stands
with a near-straight knee, which makes the leg 7.3× worse at height control
than a properly flexed theropod stance — plausibly forcing the large joint
excursions rather than merely permitting them. The stance fix is now ranked
first, and `action_scale` is demoted to a mitigation that should not be sized
until the stance is settled.

---

## The measurement

`smoothness_weight` works, with returns that rule it out as a path to a smooth
gait. RMS action change per joint per step, via `compare_run_diagnostics.py`:

| `smoothness_weight` | 0.1 | 0.7 | 2.0 |
|---|---|---|---|
| RMS action change / joint | 1.018 | 0.747 | 0.628 |
| final eval reward | 2699.8 | 2516.9 | 2654.7 |
| eval std | ±40.6 | ±87.2 | **±12.2** |

Both escalations fit `r ∝ w^-0.16` to within 4% (exponents −0.159 and −0.165).
Extrapolating, `r = 0.5` needs `w ≈ 8`, halving to `r = 0.31` needs `w ≈ 144`,
and `r < 0.1` ("smooth") needs a weight in the six figures — a regime where the
smoothness term is the entire objective. The lever is real and cheap (w=2.0
scored *better* than w=0.7 on every axis) but it cannot reach the target.

### The legs are barely working

Measured on the settled home stance (zero action, 600 steps):

| joint | ctrl span | kp | static hold torque | force limit | load |
|---|---|---|---|---|---|
| hip_pitch | 130° | 1200 | 52.1 N·m | 1800 | 2.9% |
| knee | 100° | 1500 | 49.5 N·m | 2250 | 2.2% |
| ankle | 100° | 900 | 58.0 N·m | 1350 | 4.3% |

At `r = 0.628` the policy commands **31° of knee and 41° of hip every 10 ms**
(control runs at 100 Hz: `timestep 0.002` × `frame_skip 5`), producing
822–855 N·m of torque swing — 16–17× the static hold torque. Holding the stance
takes about 1.9° of equilibrium offset at the knee (49.5 N·m ÷ kp 1500) against
±50° of granted authority, roughly **26× more than the task requires**.

## The stance is not anatomical, and that is mechanical

Settled-stance limb geometry:

| | model | real *Tyrannosaurus* |
|---|---|---|
| knee interior angle | **172.1°** | flexed; reconstructions generally 110–140° |
| ankle interior angle | 155.1° | flexed |
| femur inclination from vertical | 4.8° | subvertical but clearly inclined |
| tibia / femur | **1.000** | ~0.9 in large adults; >1 in juveniles |
| metatarsus / femur | 0.613 | ~0.5 in large adults |
| digitigrade | yes (ankle 0.228 m clear) | yes ✔ |
| hip height | 0.929 m | ~2.5–3.2 m (model is ~⅓ scale) |

The limb is a near-vertical column — an elephant's posture rather than a
theropod's. The exact reconstructed knee angle varies by source and by whether
it describes standing or mid-stance, but nothing in the literature puts it near
172°. A tibia:femur of exactly 1.000 reads as two segments made equal for
convenience rather than a measured proportion; adult *T. rex* is specifically
known for a tibia *shorter* than its femur, which is central to the argument
that it was not a fast runner. Being digitigrade is correct and should be kept.

**Why it matters.** For a two-segment limb, hip-to-ankle distance is
`d = √(l₁² + l₂² − 2·l₁·l₂·cos θ)`, so `dd/dθ = l₁·l₂·sin θ / d`. At the
model's own numbers:

| knee interior angle | leg-length authority | knee travel per 1 cm of height |
|---|---|---|
| **172.1° (model)** | 0.024 m/rad | **23.7°** |
| 120° (flexed) | 0.175 m/rad | 3.3° |

**The straight-legged stance makes the leg 7.3× less effective at height
control.** Stage 1 carries a live height term (`height_weight = 1.0`, target
0.9757), and from a locked knee the only way to service it is to swing the
joint through tens of degrees. That is the 31°-per-step figure, arrived at from
geometry alone. It also explains the trivial hold torques: a columnar limb
carries load through bone alignment rather than muscle, so the actuators are
not so much oversized as idle.

This reframes the whole problem. The large excursions may be the policy doing
the only thing the geometry allows, not a control pathology to be taxed or
clamped.

## What this is not

Ruled out by direct measurement:

- **Center of mass.** Excluding the prey mocap body (65.4 kg, not part of the
  animal), the T-Rex masses 85.7 kg with its CoM at x = +0.0887, sitting 60.9%
  of the way from heel to toe within a 0.336 m support polygon — 0.131 m of
  margin to the toe edge, 0.205 m to the heel. Correct for a biped.
- **Mass distribution.** Pelvis 35 kg (41% of the animal), skull 8.7 kg, thighs
  5.0 kg each. Nothing anomalous.
- **Joint limits.** Every leg joint parks at 46–50% of its *range*. Nothing is
  pinned against a stop.

> **Correction.** An earlier revision of this document also claimed "no
> near-singular knee" on the strength of that last bullet. That was wrong.
> Sitting mid-range is not the same as being well-conditioned: at 172.1° the
> knee is 8° from full extension, which is precisely the singular region for
> leg-length control. Range position and mechanical conditioning are different
> properties, and the first is not evidence for the second.

## Prior art: the raptor

There is **no existing `action_scale` concept anywhere in the repo** — it would
be the first. The raptor's documented jitter fix was the entropy anchor, not
action scaling (`TREX_STAGE1_LEG_JITTER.md`: raptor S1 was the only stage that
annealed `algo_std`, and it did so at the *lowest* smoothness weight, 0.05).

The raptor plant is better proportioned. Same probe, both at 100 Hz:

| | T-Rex | Raptor |
|---|---|---|
| animal mass | 85.7 kg | 13.5 kg |
| knee kp | 1500 | 180 |
| hip / knee / ankle hold vs force limit | 2.9% / 2.2% / 4.3% | 4.7% / 5.6% / **23.0%** |
| **mean leg authority ratio** | **22×** | **11×** |

But the raptor is not a smoothness success story in absolute terms: at its
documented `r = 0.925` and 57.5° per unit of action it commands ~53° of knee
per step, *more* than the T-Rex's current 31°. The normalized metric flatters
it.

**Its stance is columnar too** (measured 2026-07-27, same probe):

| | T-Rex | Raptor |
|---|---|---|
| knee interior angle | 172.1° | **163.1°** |
| degrees from straight | 7.9° | 16.9° |
| femur inclination from vertical | 4.8° | 13.7° |
| tibia / femur | 1.000 | 1.110 |
| leg-length authority | 0.024 m/rad | 0.028 m/rad |
| **knee travel per 1 cm of height** | **23.7°** | **20.5°** |

The raptor is less extreme on every measure and its proportions are more
defensible — a tibia longer than the femur is the cursorial condition, correct
for a small fast theropod — but 163.1° is the same class of problem, and its
height authority is barely better than the T-Rex's. **This is a plant-wide
stance issue, not a T-Rex one.** Fix the T-Rex first as the pilot, then apply
the same treatment to the raptor rather than re-deriving the approach.

The two quadrupeds (`dibothrosuchus`, `brachiosaurus`) use different leg-joint
naming and were not measured by this probe — extend it before assuming
anything about them either way.

## Options, ranked

### 1. Correct the home-keyframe stance (recommended)

Flex the home keyframe to a defensible theropod posture and re-centre the leg
`ctrlrange` on the new pose.

- Attacks the root cause. Restores height authority (~7× per the table above),
  loads the actuators sensibly, and plausibly shrinks the excursions without
  touching the reward or the action mapping at all.
- Fixes an anatomical fidelity problem in its own right.
- **Widest blast radius of any option** — see below. This is the honest cost.

### 2. Per-stage `action_scale` (mitigation, do not size yet)

An env parameter scaling the normalized action about the ctrlrange midpoint,
exposed per stage in the TOML `[env]` block.

- Pure function of the action, so it fits the existing action-mapping contract
  (`plant_contract.py:56-58`) as a new mode. `policy_interface_revision` 6→7,
  physics untouched. Preserves zero-action behaviour, so the 1800.56 baseline
  and the 1900 gate survive.
- **Demoted because sizing it against the current stance would tune it to
  compensate for the wrong thing.** If the knee genuinely needs 24° of travel
  per centimetre of height, clamping the envelope makes balancing harder, not
  smoother. Revisit once the stance is fixed and the required envelope is known.
- Sizing floor, when it is time: a policy with RMS action change `r` must sweep
  a commanded envelope of at least `r`, so `r = 0.628` implies **≥31°
  peak-to-peak of knee travel**. `action_scale = 0.25` sits *below* what the
  current policy demonstrably uses; 0.4–0.5 is the defensible starting range.

### 3. Action rate limit / low-pass filter (deferred)

Caps per-step change while preserving reachable range — the best long-term
answer for locomotion, since stage 2 needs the full swing. Deferred because it
makes the mapping history-dependent, breaking the "pure function of action"
assumption in the contract's callable fingerprinting and forcing the MJX path
to carry filter state (`mjx_env.py:508-556`, `jax_setup.py:487-509`).

### 4. Narrow the leg `ctrlrange` alone (rejected)

`actuator_ctrlrange` feeds both the policy-interface payload
(`plant_contract.py:961`) and the physics payload (`:1107`), so it bumps both
revisions — the same cost as fixing the stance properly, without fixing it.
Also global across stages, and stage 2 reaches 6.36 m/s on that swing.

## Recommended sequence

1. ~~**Measure the raptor's stance.**~~ **Done** — it is columnar too (table
   above). Scope this as a plant-wide issue with the T-Rex as the pilot.
2. **Fix the T-Rex stance** (option 1), re-derive the dependent constants,
   re-run. Nothing below blocks this.
3. **Capture the current joint envelope** —
   `joint_excursion_report.py trex 1 <best_model.zip>` against
   `20260727_130726/stage1`, ~10 min in Colab. This is the "before" picture for
   comparing against the post-fix run. **Not a blocker:** it was originally
   pre-work for sizing `action_scale`, which is now demoted, and the model is
   saved on Drive so the measurement can be taken at any time.
4. **Re-evaluate whether `action_scale` is still needed.** It may not be.
5. **Apply the same stance treatment to the raptor**, and extend the geometry
   probe to the quadrupeds.

## Blast radius of the stance fix

| Area | Impact |
|---|---|
| `environments/trex/assets/trex.xml` home keyframe + leg `ctrlrange` | `physics_revision` 4→5 **and** `policy_interface_revision` 6→7 |
| `plant_manifest.generated.json` | Regenerate |
| `environments/trex/envs/trex_env.py` `target_z = 0.9757` | Settled height moves; must be re-measured |
| `TestHeightTargetTracksStance` | Pins `target_z` to the measured stance — will fail by design |
| `environments/trex/mjx_config.py` `target_standing_z` | Must track the SB3 value |
| `configs/trex/*.toml` `natural_pitch = 0.05` | Describes the measured neutral pose; re-measure |
| `nosedive_termination_threshold = 0.47` | Calibrated against `natural_pitch` |
| `healthy_z_range = (0.75, 1.6)` | Hip height moves; revisit both bounds |
| `min_avg_reward = 1900.0` | Zero-action floor changes; re-derive from the baseline script |
| **All existing checkpoints** | **Invalidated.** Forces a full 13h curriculum re-run. |
| CI | `plant-contract` verifies the manifest; `test-jax-cpu` verifies SB3/MJX parity |

The dependent-constant chain is the real work here, not the XML edit. Several
of those values were themselves measured against the current stance, and at
least one (`target_z`) has a test pinning it deliberately so it cannot drift
silently — that test failing is the system working.

## Validation

1. Re-run `zero_action_baseline.py trex --episodes 40`. It **will** move; the
   new value re-derives the stage gate. Confirm a real policy still clears it
   with margin before spending 13h.
2. Re-measure the settled stance; update `target_z` and `target_standing_z`
   together, and confirm `TestHeightTargetTracksStance` passes on the new
   number.
3. Regenerate the plant manifest; confirm both revisions bump.
4. `plant-contract` and `test-jax-cpu` green — the parity test is the one that
   matters and cannot be run locally.
5. Full stage-1 run, compared with `compare_run_diagnostics.py` and
   `joint_excursion_report.py` against `20260727_130726/stage1`. Success is
   materially lower commanded degrees per step **at equal or better reward**.
   The bar is 2654.7 ± 12.2, and it is a high one.

## Risks

- **The stance fix could cost reward.** The current policy is well adapted to
  its columnar stance and scores 2654.7 ± 12.2 — the best T-Rex balance policy
  to date. A flexed stance stores less passive stability, so the zero-action
  baseline will likely fall faster and stage 1 becomes a genuinely harder task.
  That is the point, but the gate must be re-derived rather than reused.
- **Anatomical targets are a judgement call.** Reconstructed knee angles vary
  by source; this plan should pick a defensible published figure and cite it
  rather than splitting the difference silently.
- **Stages 2 and 3 are untouched** and still run `smoothness_weight` 0.05 and
  0.02. Independent of everything above, worth fixing.
- **Judge `action_scale`, if it is ever used, on commanded degrees per step**,
  not on `r` — it caps amplitude, not normalized rate, so the headline metric
  may barely move while visible motion falls by the scale factor.

## Effort

The stance fix is a day or two: the XML edit is minutes, the dependent-constant
chain and its tests are the bulk, plus a 13h validation curriculum. The two
measurement steps in the recommended sequence are ~20 minutes total and should
happen first — step 2 in particular may widen this from a T-Rex plan into a
project-wide one.
