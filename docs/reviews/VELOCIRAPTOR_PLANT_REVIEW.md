# Velociraptor Plant Review — Anatomy and Mechanics

**Date:** 2026-07-27
**Scope:** `environments/velociraptor/assets/raptor.xml`, its Gymnasium env, and
its MJX registration. Nothing was changed; this is a findings document.
**Plant reviewed:** `physics_revision = 2`, `policy_interface_revision = 6`,
`visual_revision = 3` (`configs/plant_versions.toml`).

Every claim below is either a number measured on the committed plant or a
published figure with a citation. Where a recommendation rests on a judgement
call rather than a measurement, it says so.

---

## How to reproduce the measurements

All plant numbers come from loading `raptor.xml`, resetting to the `home`
keyframe, holding `key_ctrl`, and stepping 600 env steps (3000 MuJoCo steps at
`timestep 0.002`, `frame_skip 5` — i.e. 6 s of settling at 100 Hz control).
Contact forces use `mj_contactForce`; spring torques are computed directly as
`-jnt_stiffness * (qpos - qpos_spring)`; the reward figures come from stepping
the real `RaptorEnv` with `reset_noise_scale = 0` and the committed
`configs/velociraptor/stage1_balance.toml` weights.

Counterfactuals (§3.2) rebuild the model from modified MJCF text held in a
temporary file, with `mujoco.MjModel.from_xml_path` redirected for the duration
— nothing in the repository is edited, and the same stage config and seeds are
used across variants so only the one attribute differs.

This is the same instrument used for the T-Rex stance work, which reproduced
that plant's committed constants exactly (`target_z` 0.9757, `natural_pitch`
0.05, zero-action baseline 1800.56 ± 1267.66) before being trusted — so the
protocol is validated against known-good numbers. The T-Rex also serves as the
control for the §3.2 spring experiment.

---

## Summary of findings

Ranked by (evidence strength × consequence). "Free" means the change moves no
plant fingerprint and so invalidates no checkpoints — verified by regenerating
the manifest, not assumed.

| # | Finding | Evidence | Severity | Cost to fix |
|---|---|---|---|---|
| 1 | **Stage 1 is already solved by doing nothing** — a statue scores 17× the advancement gate at 98% survival | measured (§3.0) | **Critical** | config only |
| 2 | **The plant does not stand on its actuators.** Remove the (unintended) leg spring bias and it falls in 1.4 s, 0% survival | measured, 3-way counterfactual + T-Rex control (§3.2) | **Critical** | plant + gain retune |
| 3 | Foot touch sensor sees **55.6%** of transmitted floor force | measured | High — same bug class the T-Rex already fixed | plant revision bump |
| 4 | Metatarsus **78% too long** relative to femur; also bears 26% of load, extending the support base rearward | Persons & Currie 2016 + measured | High | plant revision bump |
| 5 | `natural_pitch` stale by **4.0°**, costing **~104 reward/episode** | measured | Medium | **free** |
| 6 | Claw motors are the **only unbounded actuators** (693 N at tip, 5.2× body weight) | measured | Medium | plant revision bump |
| 7 | MJX has **no torso/neck/head termination**; SB3 does | code, both paths | Medium | config only |
| 8 | Ankle parks at **20.2%** of its range, not mid-range | measured | Medium — consequence of #2 | plant |
| 9 | Stance is columnar (**163.1°** knee, 20.5°/cm) | measured | Low *for now* — see §3.5 | plant |
| 10 | No neck or head articulation at all | code | Low–Medium | plant + obs change |
| 11 | Forelimbs are 2-DOF stubs, 25% of femur+tibia+metatarsus length | measured | Low | design question |

> **Findings 1 and 2 are the review's main result and they compound.** The
> plant is held upright by a passive bias torque that no biological or robotic
> system would get, and the stage that is supposed to teach it to balance is
> already passed by a policy that does nothing. Every raptor training result to
> date — including the `r = 0.925` jitter figure that
> `docs/TREX_LEG_FLEXING_PLAN.md` compares the T-Rex against — was obtained on
> this plant.

**Correct as-is, do not change:** digit II is non-weight-bearing (§2.3),
tibia:femur is within 3.6% of the published value (§2.2), the mass budget is
close to published estimates (§2.1).

---

## 1. What the plant is

| | value |
|---|---|
| `nq` / `nv` / `nu` | 31 / 30 / 22 |
| bodies / geoms / sensors | 22 / 24 / 10 |
| timestep, frame skip | 0.002 s, 5 → **100 Hz** control |
| integrator | `implicitfast`, warmstart enabled |
| total mass (prey excluded) | **13.50 kg** |
| settled pelvis height | 0.4932 m |
| settled pitch | 24.06° nose-down |

Mass budget:

| group | mass | share |
|---|---|---|
| axial (torso + neck + head, one rigid body) | 5.30 kg | 39.3% |
| tail (5 segments) | 3.00 kg | 22.2% |
| legs (both) | 4.90 kg | 36.3% |
| forelimbs (both) | 0.30 kg | 2.2% |

---

## 2. Anatomical fidelity

### 2.1 Size and mass — good

Volumetric reconstructions of *Velociraptor mongoliensis* (holotype AMNH 6515)
give **14.1–19.7 kg**. The model is **13.5 kg** — 4% below the bottom of that
range, and clearly the right order. Hip height 0.493 m is consistent with the
commonly cited ~0.5 m. The XML's "~2 m nose-to-tail" claim is roughly borne out
(geom-centre extent 1.22 m, plus capsule half-lengths and the tail tip).

**No action.** The animal is the right size.

### 2.2 Hindlimb proportions — one segment is badly wrong

Persons & Currie (2016), *Scientific Reports* 6:19828, Table 1, specimen
**IGM 100/986** (*V. mongoliensis*): femur **238 mm**, tibia **255 mm**,
metatarsal III **99 mm**.

Measured on the plant (capsule lengths from `r_thigh_geom`, `r_tibia_geom`,
`r_metatarsus_geom`):

| segment | model | IGM 100/986 | model / real |
|---|---|---|---|
| femur | 181.1 mm | 238 mm | 0.761 |
| tibia | 201.0 mm | 255 mm | 0.788 |
| metatarsal III | **134.2 mm** | **99 mm** | **1.356** |
| total limb | 516 mm | 592 mm | 0.872 |

| ratio | model | real | error |
|---|---|---|---|
| tibia / femur | 1.110 | 1.071 | **+3.6%** ✔ |
| MT III / femur | **0.741** | **0.416** | **+78.1%** ✘ |

Share of total limb length: model femur 35.1% / tibia 38.9% / metatarsus
**26.0%**; real 40.2% / 43.1% / **16.7%**.

Norell & Makovicky (1999), *AMNH Novitates* 3282, give a second, independent
data point on a different specimen (MPC-D 100/985): femur ~220 mm, metatarsal
IV ~113 mm → MT/femur ≈ 0.51. The model's 0.741 is well above *both* published
figures, so this is not an artefact of one specimen or one measurement
convention.

**tibia:femur is genuinely good** and should be left alone — it is the
cursorial condition and the model has it within 4%. The problem is isolated to
the metatarsus, which is roughly 45–78% too long depending on which specimen
you compare against.

This matters mechanically, not just cosmetically. A long metatarsus lengthens
the moment arm from the ankle to the ground, which is exactly the joint already
carrying the plant's highest actuator load (§3.4), and it makes the model look
*more* cursorial than the animal was — Persons & Currie specifically find that
"many dromaeosaur taxa, including *Velociraptor* and *Deinonychus*, have **low**
cursorial-limb-proportion scores."

### 2.3 The foot and digit II — correct, and worth protecting

The sickle claw geoms carry `contype="2" conaffinity="2"` while the floor is
`contype="1" conaffinity="1"`, so **the claw can never contact the ground** but
*can* contact the prey (also class 2). Weight passes through digits III and IV
only.

That is the right call and it matches the literature. Fowler, Freedman,
Scannella & Kambic (2011), *PLoS ONE* 6(12):e28964, "The Predatory Ecology of
*Deinonychus* and the Origin of Flapping in Birds," argue the hypertrophied
digit II claw was functionally analogous to the enlarged digit II talon of
accipitrid hawks and eagles — a **prey-restraint** structure used to grip and
pin, not a slashing or running structure. Held clear of the ground is exactly
where it belongs.

**No action** — but see §3.3, because the claw's *actuator* is not as carefully
specified as its collision class.

### 2.4 The tail — defensible, but not graded

Dromaeosaurids are diagnosed partly by greatly elongated prezygapophyses and
chevrons spanning several caudal vertebrae. Ostrom's original interpretation
(from *Deinonychus*) was that these stiffened the tail so it could flex only at
the base and swing as one rigid lever. That has since been softened: an
articulated *V. mongoliensis* tail preserved in a long horizontal S-curve shows
real lateral flexibility in life. The modern reading is that the rods raise
stiffness so the tail works as a **dynamic stabilizer**.

The model gives all five tail joints the *same* ±15° range and the *same*
stiffness 40 / damping 15, and actuates only the proximal four DOF:

| joint | range | stiffness | actuator |
|---|---|---|---|
| tail_1_pitch | ±15° | 40 | yes |
| tail_1_yaw | ±10° | 40 | yes |
| tail_2_pitch | ±15° | 40 | yes |
| tail_3_pitch | ±15° | 40 | yes |
| tail_4_pitch | ±15° | 40 | **none (passive)** |
| tail_5_pitch | ±15° | 40 | **none (passive)** |

So mobility is uniform rather than concentrated at the base, and cumulative
deflection is ±75°. This is a reasonable engineering compromise and it is
*directly* relevant to the project's robotics aims — Libby, Moore, Chang-Siu,
Li, Cohen, Jusufi & Full (2012), *Nature* 481:181–184, "Tail-assisted pitch
control in lizards, robots and dinosaurs," show an active tail stabilising body
attitude by transferring angular momentum, using a lizard-sized robot.

For contrast, the T-Rex plant took the other route and **fused** its distal two
tail segments into rigid geometry, citing ossified tendons. The raptor keeping
them jointed-but-passive is not wrong, but the two species now model the same
anatomical structure two different ways, and neither file mentions the other.

**Low priority.** If touched, grade the stiffness distally rather than adding
range.

### 2.5 Head and neck — absent

There is **no neck or head body**. `neck` and `head` are geoms attached
directly to the `pelvis` body, so the animal's head is rigidly welded to its
torso with zero degrees of freedom.

For a balance stage this is survivable. For **stage 3 (strike)** it is a real
simplification: the task rewards bringing a claw to the prey, and the animal
cannot orient its head at all. The T-Rex, whose stage 3 is a *bite* task, has
`neck_pitch`, `neck_yaw` and `head_pitch`.

**Judgement call, not a defect.** Flagged because it is invisible from the
config layer and surprising given the T-Rex precedent.

### 2.6 Forelimbs — heavily under-scaled

Each arm is a single 0.130 m capsule, 0.15 kg, with two shoulder DOF and no
elbow or wrist — **25% of the femur+tibia+metatarsus length** (21% if digit III
is counted into the leg). Real
*Velociraptor* forelimbs are long relative to the hindlimb and, under Fowler et
al. (2011), functionally important: that paper's "stability flapping"
hypothesis has dromaeosaurs beating their forelimbs to stay balanced atop
pinned prey — precisely the stage-3 scenario this species trains for.

Measured settled load on the shoulder actuators is **0.02 N·m (0.2% of
forcerange)**, i.e. they currently do nothing, while costing 4 actuators and 8
observation dimensions.

**Judgement call.** Either commit to them (longer, with an elbow, per Fowler)
or weld them as the T-Rex did — the current state is the expensive middle.

---

## 3. Plant mechanics defects

### 3.0 Stage 1 is already solved by doing nothing

`environments/shared/scripts/zero_action_baseline.py velociraptor --episodes 40`,
committed stage-1 config:

| | velociraptor | T-Rex (post stance fix) |
|---|---|---|
| statue reward | **1704.93 ± 259.12** | 1743.73 ± 1275.54 |
| statue mean − std | **1445.81** | 468.19 |
| statue episode length | **977.5 / 1000** | 638.1 |
| statue full-horizon share | **98%** | 57% |
| `min_avg_reward` gate | **100.0** | 1840 |
| statue clears its gate by | **17×** | fails it |

A policy that outputs `np.zeros(22)` survives 98% of episodes, scores 17× the
stage's advancement gate, and is promoted into stage 2. There is almost nothing
for a stage-1 policy to learn, and a run that learns nothing is indistinguishable
from one that learns to balance.

This is the exact failure the T-Rex config already documents and fixed —
"a statue cleared the old gate eighteen times over and advanced into stage 2"
(`configs/trex/stage1_balance.toml`). The raptor has the same problem, unfixed,
at 17×.

The reset-noise calibration was not carried over either. The T-Rex config
records that `reset_noise_scale = 0.05` leaves a 93% statue floor so "the stage
rewards a statue and training can only lose ground," which is why it moved to
0.10. The raptor is still at 0.05:

| `reset_noise_scale` | statue full-horizon | statue mean − std |
|---|---|---|
| 0.01 | 100% | 1745.4 |
| **0.05 (committed)** | **97%** | **1393.4** |
| 0.10 | 80% | 774.7 |
| 0.15 | 60% | 301.7 |
| 0.20 | 37% | −102.9 |

In fairness, part of the gap to the T-Rex is the noise setting, not the plant:
at matched noise (0.10) the raptor statue is 80% against the T-Rex's 57%. But
80% is still a stage where doing nothing mostly works. The rest of the gap is
§3.2.

### 3.1 The foot touch sensor sees 55.6% of the force — the T-Rex bug, unfixed

A MuJoCo touch sensor sums only contacts on geoms belonging to **its own
site's body**. The `r_foot` / `l_foot` sites sit on the `*_toe_d3` bodies. The
other two load paths are separate bodies and are therefore invisible:

| body | floor force | sensed? |
|---|---|---|
| `r_toe_d3` | 36.79 N | **yes** |
| `r_metatarsus` | 17.36 N | no |
| `r_toe_d4` | 12.07 N | no |
| **total transmitted** | **66.22 N** | sensor reports **36.79 N** |

**The sensor captures 55.6% of the force the foot actually transmits**,
identically on both feet.

This is the *same defect, in the same repo*, that the T-Rex plant already
fixed. `configs/plant_versions.toml` item 3 records it: "At the home keyframe
each foot transmitted 500.4 N while its sensor reported 388.4 N" — 77.6%. The
raptor is **worse** (55.6%), and the fix is already designed and shipped for
the other species: give each load-bearing body its own touch site and sensor,
append them after the existing sensor block so no index moves, and sum them in
both the SB3 and MJX paths (`sensor_foot_aux_indices`).

Two observations make this concrete rather than theoretical:

- Foot contact is an **observation** the policy trains on, and it is used by
  the JAX `foot_contact_gate` / `foot_contact_weight` reward terms.
- Under load transfer the error is not a constant scale factor. If weight
  shifts onto digit IV and the metatarsus, the reading can fall while the true
  force rises — the same failure mode the T-Rex note describes ("under load
  transfer the reading could drop to zero on a foot carrying full weight").

### 3.2 `springref` is unset, and the springs are holding the animal up

The `leg_joint` default class sets `stiffness="20.0"`. **No raptor leg joint
sets `springref`**, so it defaults to 0 and every spring pulls its joint toward
zero rather than toward the stance:

| joint | home qpos | `ref` | `springref` | spring torque at home |
|---|---|---|---|---|
| `r_hip_pitch` | +38.0° | +38.0° | **0.0°** | −13.26 N·m |
| `r_knee` | −50.0° | −50.0° | **0.0°** | +17.45 N·m |
| `r_ankle` | +100.0° | +100.0° | **0.0°** | −34.91 N·m |
| | | | **total** | **65.62 N·m per leg** |

The T-Rex sets `springref` explicitly on every leg joint; its equivalent total
is **0.00 N·m**.

**The XML says this was not the intent.** The keyframe comment reads:

> Hip ref angles are set to 38 deg (= 18 deg crouch + 20 deg lean compensation)
> so **joint springs hold this pose at rest**.

In MuJoCo `ref` sets the qpos↔pose mapping and `springref` sets the spring
equilibrium; they are independent. Setting `ref` alone does not do what the
comment describes. This is an author-intent mismatch documented in the file
itself.

**This is not a tidiness problem. The plant does not stand on its actuators.**

Three-way counterfactual, zero action, committed stage-1 config, 20 episodes
each (the model is rebuilt from modified MJCF; nothing in the repo is changed):

| leg spring configuration | reward | episode length | full-horizon |
|---|---|---|---|
| **as committed** (stiffness 20, `springref` → 0) | 1671.8 ± 311.7 | 962.3 | **95%** |
| springs deleted (stiffness 0) | 173.0 ± 56.7 | 138.9 | **0%** |
| same stiffness, `springref` = stance | 161.9 ± 45.0 | 132.2 | **0%** |

Remove the spring bias and the animal falls in ~139 steps — **1.4 seconds,
every episode**. The position actuators cannot hold the pose on their own.

The detail that matters most: **both** counterfactuals collapse. Anchoring the
*same* stiffness at the pose the animal actually holds kills it as thoroughly
as deleting the springs. So the support is not coming from the stiffness — it
is coming from the **offset**. What holds the raptor up is a constant
antigravity bias torque, 65.6 N·m per leg, that falls out of an unset attribute
and is anchored at a joint coordinate with no anatomical meaning.

That is the distinction between "the model has passive tendon stiffness, which
is biologically reasonable" and "the model has an invisible prop." Passive
stiffness anchored near the neutral posture is the former. This is the latter.

**Control: the T-Rex does not behave this way.** Same experiment, same
protocol, on `trex.xml` (which sets `springref` to the stance on every leg
joint, total spring torque at home 0.00 N·m):

| | as committed | springs deleted |
|---|---|---|
| reward | 1671.7 ± 1290.4 | 1681.5 ± 1283.9 |
| full-horizon | 55% | **55%** |

Deleting the T-Rex's leg springs changes nothing measurable. It stands on its
actuators. So this is a raptor-specific defect, not a shared modelling
convention — and the fix has a working reference implementation in the same
repo.

Secondary consequence: because the servos are dragged off their commanded pose,
**the plant does not stand in the pose anyone authored.** The ankle is
commanded to 100° and settles at 80.2°. The `home` keyframe is not the stance.

`springref` and the actuator sizing have to move together, and the honest
expectation is that fixing this makes stage 1 markedly harder — which is the
point of fixing it.

This also **re-explains a number the T-Rex leg-flexing plan flagged as
notable.** That plan's comparison table gives raptor hip/knee/ankle hold at
4.7% / 5.6% / **23.0%** of force limit against the T-Rex's 2.9% / 2.2% / 4.3%,
and reads the raptor as the better-loaded plant. The measurement says
otherwise: the ankle actuator is applying 34.54 N·m against a **−28.00 N·m
spring**, so ~81% of that "load" is the plant fighting itself, not carrying the
animal.

### 3.3 The claw motors are the only unbounded actuators in the plant

Every position actuator declares an explicit `forcerange`. The two claw
actuators do not:

```
r_claw_act   forcelimited=False   forcerange=[0. 0.]   gear=50
l_claw_act   forcelimited=False   forcerange=[0. 0.]   gear=50
```

`gear="50"` with `ctrlrange="-1 1"` means ±50 N·m applied to a 0.05 kg claw on
a 0.0721 m geom — **693 N at the claw tip, 5.2× the animal's 132 N body
weight**. Because the claw is the geom that contacts the prey
(§2.3) and stage 3 rewards exactly that contact, this is the actuator most
directly coupled to the task reward and the only one with no bound.

The repo has a documented convention for this — `0.8× kp` spike caps, raised to
`1.5× kp` for the stance joints, recorded in the actuator comment block and in
`docs/investigations/STAGE2_RECOMMENDATIONS.md` R2. The claws were never
brought into it.

### 3.4 The ankle parks near the bottom of its range

| joint | range | home | settled | position in range |
|---|---|---|---|---|
| `r_hip_pitch` | [−60, 90] | 38.0 | 34.0 | 62.7% |
| `r_knee` | [−120, −5] | −50.0 | −45.2 | 65.1% |
| `r_ankle` | [60, 160] | 100.0 | **80.2** | **20.2%** |
| `r_toe_d3` | [−30, 60] | 10.0 | 5.1 | 39.0% |

The ankle settles **20.2° from its lower stop** while carrying the plant's
highest actuator load, and it sags 19.8° below its commanded 100° because it is
fighting the −28 N·m spring (§3.2). The T-Rex's equivalent joints all park at
46–50% of range by design.

This is a consequence of §3.2, not an independent defect, and it should be
re-measured after any `springref` work rather than fixed on its own.

### 3.5 The stance is columnar, but the T-Rex argument does not transfer

Measured: knee interior **163.1°**, ankle 127.9°, femur 13.7° from vertical,
leg-length authority 0.0280 m/rad → **20.5° of knee travel per centimetre** of
hip height.

That is the same class of geometry the T-Rex stance correction just fixed
(172.1° → 135.0°, 23.7°/cm → 4.3°/cm), and `TREX_LEG_FLEXING_PLAN.md` step 5
proposes porting the treatment here.

**The premise does not hold for this species.** The T-Rex argument was
specifically that stage 1 carries a live height term (`height_weight = 1.0`,
target 0.9757) which, from a locked knee, can only be serviced by large knee
excursions. The raptor has **no height term at all**:

- `configs/velociraptor/stage1_balance.toml` has no `height_weight` key.
- `raptor_env.py` references pelvis height only for logging (`info["pelvis_height"]`,
  twice) and for the shared height/tilt *termination* — there is no `target_z`,
  no `height_frac`, no `reward_height`. Five height mentions in total, none of
  them a reward term; `trex_env.py` has 21.)

So the raptor has the poor geometry but nothing pulling on it. Its documented
jitter (r = 0.925, ~53° of knee per step — *worse* than the T-Rex's 31°) is
real, but the height-term mechanism cannot be its cause.

**Recommendation: do not port the stance fix on the strength of the T-Rex
result.** Diagnose the raptor's excursions on their own terms first. And note
that §3.2 blocks a stance change anyway: you cannot meaningfully re-pose a limb
whose stance is partly held up by a spring anchored at an unrelated angle.

### 3.6 `natural_pitch` is stale by 4.0°, and it is not free reward

`RaptorEnv.natural_pitch` defaults to **0.35 rad** (20.05°) and
`mjx_config._NATURAL_PITCH` matches it. The plant settles at **0.4200 rad
(24.06°)**, sd 0.0002 — a **+4.01°** mismatch.

Unlike the T-Rex, the raptor *does* set `posture_target_forward_z`, so the
posture reward is centred on the stale angle and the animal is penalised for
standing in its own natural pose. Measured over a settled episode at the
committed stage-1 weights (`posture_weight = 1.5`, `nosedive_weight = 1.5`):

| | `natural_pitch` 0.35 | corrected to 0.42 |
|---|---|---|
| nosedive | **−97.3 / episode** | −0.1 |
| posture | −7.3 / episode | −0.0 |
| total reward | 1.745 / step | **1.850 / step (+6.0%)** |

**~104 reward per episode of pure shaping error**, dominated by the nosedive
term. It also shifts the nosedive *termination* boundary, costing ~4° of
forward-pitch headroom.

**This one is free.** I verified it: applying the change and running
`python -m environments.shared.plant_contract --check` reports "Plant manifest
is current" — no fingerprint moves, so no checkpoint is invalidated and no
curriculum re-run is forced. The raptor's stage-1 gate is the placeholder
`min_avg_reward = 100.0`, so there is nothing to re-derive either.

One caveat that is the owner's call, not mine: `mjx_config.py` deliberately
pins `natural_forward_z=-0.342` with the comment "Preserve the existing rounded
nosedive baseline so reward shaping does not move the termination boundary in
this isolated experiment." Someone froze that on purpose. Correcting
`natural_pitch` retires that freeze.

---

## 4. SB3 / MJX parity

### 4.1 The two paths terminate on different things

| | SB3 (`raptor_env.py`) | MJX (`mjx_config.py`) |
|---|---|---|
| floor contact terminates | torso, neck, head, tail_3, tail_4, tail_5 | — |
| body-height terminates | — | tail_3 0.05, tail_4 0.04, tail_5 0.03 |
| site-height terminates | — | `None` |
| healthy z | (0.3, 1.0) | (0.3, 1.0) |

The tail is covered on both paths. **The torso, neck and head are covered only
on SB3.** On MJX the raptor can put its face on the ground without terminating,
as long as the pelvis stays above 0.3 m and pitch stays inside the nosedive
gate (57° nose-down).

For contrast, the T-Rex covers the anterior body on both paths — MJX
`termination_body_heights` includes `skull: 0.45`, `torso: 0.25`, both thighs,
and a `head_tip: 0.12` site threshold.

This is the kind of divergence the shared `test_species_integration.py`
`healthy_z_range` test was written to catch for *one* field; the termination
surface has no equivalent test.

### 4.2 The nosedive termination threshold is hardcoded on one path only

`raptor_env.py:530` hardcodes it:

```python
if forward_z < self._natural_forward_z - 0.5:
```

The MJX path reads `nosedive_termination_threshold` from the stage config,
defaulting to 0.5 (`jax_setup.py`). Today the two agree, because no raptor TOML
sets the key. But the moment one does, MJX honours it and SB3 silently does
not. The T-Rex exposes it as a constructor argument and its stage-1 config sets
it explicitly.

**Low severity today, latent trap.** Cheap to close by promoting it to a
constructor argument as the T-Rex has.

---

## 5. Recommendations, ranked

| # | Change | Evidence | Blast radius |
|---|---|---|---|
| **R1** | Re-derive `min_avg_reward` from the measured statue floor, and re-calibrate `reset_noise_scale`, exactly as the T-Rex config did | §3.0, statue clears the gate 17× at 98% survival | **config only** — no checkpoint cost |
| **R2** | Set `springref` to the stance **and re-size the leg actuators together** so the plant carries itself | §3.2, 0% survival without the bias; T-Rex control | physics + policy revision; needs its own design pass |
| **R3** | Correct `natural_pitch` 0.35 → measured 0.42 on both paths | §3.6, ~104 reward/episode | **none** — verified no fingerprint moves |
| **R4** | Give digit IV and the metatarsus their own touch sites/sensors and sum them, as the T-Rex does | §3.1, sensor sees 55.6% | physics + policy revision |
| **R5** | Bound the claw motors to the repo's documented `forcerange` convention | §3.3, only unbounded actuators | physics + policy revision |
| **R6** | Add torso/neck/head termination to the MJX registration | §4.1, code-level asymmetry | config only |
| **R7** | Promote `nosedive_termination_threshold` to a constructor argument | §4.2 | config only |
| **R8** | Shorten the metatarsus toward MT III / femur ≈ 0.42–0.51 | §2.2, two independent specimens | physics + policy revision; changes mass/inertia |

**R1 first, and it is urgent rather than merely cheap.** It costs nothing, it
requires no plant edit, and until it lands every raptor stage-1 run promotes a
do-nothing policy into stage 2. It also has to be redone after R2, because R2
moves the statue floor — but doing it now stops the bleeding.

**R2 is the real work** and everything else is small beside it. It is not a
one-line change: the naive fix collapses the plant (§3.2), so the actuators
have to be re-sized to take over the load the spring bias is currently
carrying. Budget it as a design pass with its own before/after baselines, not
as a bug fix.

**R3 is free and independent** — do it whenever.

**R4 and R5 together** if the raptor's revisions are being bumped at all: both
are plant edits, both have in-repo precedent, and paying the checkpoint cost
once is cheaper than twice. Fold them into R2 if R2 is happening.

**R8 belongs with R2.** Both change the limb's static equilibrium, so doing one
without the other means measuring the stance twice.

### An honest note on ordering

The first draft of this review ranked `springref` last, as a tidiness item, and
`natural_pitch` first because it was free. That ordering was wrong, and the
experiment in §3.2 is what corrected it. The lesson generalises: "the plant has
an odd passive element" and "the plant is being held up by an odd passive
element" look identical until you delete the element and re-run. Any future
plant review here should run that counterfactual as a matter of course — it
cost minutes and it changed the conclusion.

### Explicitly not recommended

- **Do not port the T-Rex stance correction** (`TREX_LEG_FLEXING_PLAN.md`
  step 5) on the strength of the T-Rex result. The causal mechanism — a live
  height term serviced through a near-singular knee — does not exist on this
  species (§3.5).
- **Do not change tibia:femur.** At 1.110 against a published 1.071 it is
  within 3.6%, and a tibia longer than the femur is the correct cursorial
  condition.
- **Do not make digit II weight-bearing.** Its collision class is correct and
  matches the prey-restraint interpretation (§2.3).

---

## 6. Open questions this review did not settle

- **What actually causes the raptor's 53°/step knee excursions?** Ruled out:
  the height-term mechanism (§3.5). Not investigated: the entropy anchor
  (`docs/investigations/TREX_STAGE1_LEG_JITTER.md` records that raptor S1 was
  the only stage that annealed `algo_std`), the 65.6 N·m spring load, and the
  0.8×-kp toe actuators which sit at 10.7% load at rest. Note that §3.0 offers
  a simpler candidate explanation than any of these: if standing costs nothing,
  the policy is free to thrash without penalty, because there is no failure
  mode to punish it.
- **Can a retuned raptor stand at all?** §3.2 shows the springs are load-bearing
  and that removing them naively collapses the plant. It does **not** show that
  a properly re-sized plant can hold the pose — that requires the gain work in
  R2, and it is possible the actuators need more authority than they currently
  have. This is the main open risk in R2.
- **Tail mass fraction.** The tail is 22.2% of body mass. I did not find a
  published estimate for a dromaeosaurid tail mass fraction to check that
  against, so it is unassessed rather than endorsed.
- **Is the metatarsus meant to be load-bearing?** It carries 26.2% of each
  foot's floor force and contacts at the rear of the support polygon, which
  sits oddly with the "digitigrade stance" comment on the ankle joint. Whether
  that is a deliberate ball-of-foot contact or a consequence of the overlong
  metatarsus (§2.2) is unresolved.

---

## References

1. Persons, W. S. & Currie, P. J. (2016). "An approach to scoring cursorial
   limb proportions in carnivorous dinosaurs and an attempt to account for
   allometry." *Scientific Reports* 6:19828. — Table 1, *V. mongoliensis*
   IGM 100/986: femur 238 mm, tibia 255 mm, metatarsal III 99 mm; low CLP
   scores for dromaeosaurs.
2. Norell, M. A. & Makovicky, P. J. (1999). "Important features of the
   dromaeosaurid skeleton II: information from newly collected specimens of
   *Velociraptor mongoliensis*." *American Museum Novitates* 3282:1–45. —
   second specimen, femur ~220 mm, metatarsal IV ~113 mm.
3. Fowler, D. W., Freedman, E. A., Scannella, J. B. & Kambic, R. E. (2011).
   "The Predatory Ecology of *Deinonychus* and the Origin of Flapping in
   Birds." *PLoS ONE* 6(12):e28964. — digit II as an accipitrid-analogous
   prey-restraint talon; stability-flapping hypothesis for the forelimbs.
4. Libby, T., Moore, T. Y., Chang-Siu, E., Li, D., Cohen, D. J., Jusufi, A. &
   Full, R. J. (2012). "Tail-assisted pitch control in lizards, robots and
   dinosaurs." *Nature* 481:181–184. — active tails stabilising attitude by
   angular-momentum transfer.
5. Ostrom, J. H. (1969), on *Deinonychus* — origin of the stiffened-tail
   "rigid lever" interpretation, since revised by the articulated S-curved
   *Velociraptor* tail showing lateral flexibility in life.
6. Volumetric body-mass estimates for *V. mongoliensis* (holotype AMNH 6515):
   14.1–19.7 kg.

### In-repo cross-references

- `configs/plant_versions.toml` item 3 — the T-Rex foot-sensor repair this
  review's §3.1 mirrors.
- `docs/TREX_LEG_FLEXING_PLAN.md` — the stance argument §3.5 declines to port.
- `docs/investigations/STAGE2_RECOMMENDATIONS.md` R2 — the actuator
  `forcerange` convention §3.3 says the claws never joined.
