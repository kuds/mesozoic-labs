# MicroDuck lessons for gait learning and robot transfer

**Date:** 2026-10-26  
**Status:** Proposed design; no implementation or training results.  
**Scope:** A source-based comparison and staged adoption plan for Mesozoic Labs.

## 1. Recommendation and decision boundary

Adopt MicroDuck's methods for contact-aware learning, measured actuator physics,
and testing the deployed controller. Preserve Mesozoic's behavior recipes,
plant identities, normalization checks and independent certification. A new RL
backend is not the first step: both projects already use PPO.

There are two tracks:

- **Gait learning:** make declared walking behavior measurable, then test whether
  exploration and competing reward terms prevent that behavior from emerging.
- **Robot transfer:** use `compsognathus_longipes` / `robot` as the closest
  existing software benchmark for a sensor-realizable policy, calibrated servo
  model and exported-controller rehearsal.

The robot benchmark does not replace the hardware track's recommended first
physical build or approve a purchase. That choice remains in the
[hardware overview](hardware/README.md). Anatomical locomotion research and
robot engineering require different evidence: a successful simulated gait is
not validation of an extinct animal's actual locomotion.

This proposal supplements the [gait-quality plan](GAIT_QUALITY_PLAN_2026_09.md)
and [behavior-recipes plan](BEHAVIOR_RECIPES_PLAN.md); it does not adopt their
open GQ decisions, change their work order, or assign new decision IDs.
Implementation and experiment criteria remain separate decisions. In particular:

- D-D17 keeps Stable-Baselines3 as the only training backend. A future mjlab
  benchmark would require a new backend decision, not an incidental dependency.
- D-D33 places shared servo, IMU, identification and deployment facilities in a
  shared package rather than copies under individual species.
- D-D44 changes the T-Rex to physics r10 with preloaded tail springs. Earlier
  r9 checkpoints, including the proposed walking parent, are not valid parents
  for that plant. Any T-Rex work starts from the current prerequisites in
  [NEXT_STEPS.md](NEXT_STEPS.md), with criteria fixed before training.

## 2. Evidence and current capabilities

Review baseline: Mesozoic Labs `06b1a8e1fb940eec0d7a8e3af420c5b384ac0e8b`.
The upstream source revisions are pinned in section 10. The comparison is a
source/configuration review, not an independent reproduction of MicroDuck's
training or hardware results. Historical Mesozoic outcomes below are evidence
of failure modes, not evaluations of today's plant.

| Area | Existing Mesozoic capability or gap | Transferable MicroDuck lesson |
|---|---|---|
| Gait validity | The [September audit](investigations/GAIT_AUDIT_2026_09.md) found historical walkers hopping, skidding or using stacked feet. The [robot locomotion config](../configs/compsognathus_longipes/robot/locomotion.toml) still uses `reward_and_length/v1`. Floor-truth stance measurement already exists. | Separate foot-terrain support from self-contact; use clearance, air time and slip as task signals, with independent behavior evaluation. |
| Motion initiation | The [October investigation](investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md), section 15, records STILL on all three first-pilot seeds. This does not identify the cause. | Tight instantaneous posture tracking and early action penalties can make immobility preferable; inspect per-term incentives and schedule precision after skill acquisition. |
| Robot plant | The twelve-servo [robot XML](../environments/compsognathus_longipes/robot/assets/compsognathus_longipes_robot.xml) has stiffness, damping and force caps. Its 0.002 s timestep and stage `frame_skip = 10` give 50 Hz control. | Add measured torque-speed, voltage, friction, backlash and latency behavior rather than merely adding PD control. |
| Actor inputs | The [robot subclass](../environments/compsognathus_longipes/robot/env.py) retains world-frame observations, including simulator base velocity and exact target-relative information from the [species base](../environments/compsognathus_longipes/base_env.py). | Use measurements/estimates available on the robot; keep privileged simulation information out of the deployed actor. |
| Robustness | Reset noise, scheduled [pushes](../environments/shared/perturbation.py) and optional [action filtering](../environments/shared/action_filter.py) exist. Production physics randomization, sensor noise and action delay remain gaps; the [capability manifest](../configs/species_manifest.toml) marks domain randomization planned. | Randomize measured uncertainty from nominal parameters, verify the randomizers work, and avoid physically impossible ranges. |
| Artifact integrity | The [policy loader](../environments/shared/policy_loading.py) already pairs checkpoints with normalization; the [policy contract](../environments/shared/plant_contract/policy_layer.py) describes semantic identities. | Extend those contracts through export and the runtime; test the actual artifact used to command the robot. |
| Skills | The machinery for separately certifying and publishing stance, walk, hunt and recovery nodes exists. Historical certificates remain subject to current identity checks; push recovery is not equivalent to getting up from arbitrary fallen poses. | Evaluate skill entry/exit states; retain frozen specialists when experimenting with a combined policy. |

The [known-issues list](KNOWN_ISSUES.md) remains the authoritative backlog. This
design links existing gaps rather than creating a second bug list. The older
[sim-to-real plan](hardware/SIM_TO_REAL_PLAN.md) predates several current
capabilities; use executable code and current manifests to establish what exists.

## 3. Gait-learning work

### 3.1 Measure the requested behavior first

Extend the existing [gait measurement package](../environments/shared/gait/)
through the gait plan's proposed locomotion work. Do not introduce another gate
registry or a parallel publication route. Collect ground-contact truth at the
physics-substep level and aggregate explicitly to policy steps and episodes.

For the robot walker, distinguish:

- Loaded sole-to-ground support from foot-on-foot or other self-contact.
- Alternating steps from synchronized hopping, single-leg skidding and stillness.
- Foot slip while loaded, clearance during swing, and sustained commanded
  displacement from an initial lunge followed by a fall.
- Valid foot support from shin, knee, body or tail support that the task disallows.

Use morphology-specific definitions. A running task can allow flight; a walk
must not inherit a universal duck cadence or clearance height. Preserve the
existing stance and physical-failure checks. Numerical gait thresholds and
the allowed contact set must be calibrated and recorded before training, using
fixtures independent of the policies later judged.

Measurement acceptance requires fixtures for stillness, an alternating walk,
synchronous hops, a skid, stacked feet and lunge-then-fall, plus a valid running
fixture where the task allows it. Reject the invalid examples for the declared
task, report ambiguous examples, and verify invariance to irrelevant world
heading. Simulation truth used by the judge must remain separate from noisy
actor observations: an onboard touch sensor seeing self-contact must not make
the judge report ground support.

### 3.2 Diagnose motion suppression before changing the recipe

MicroDuck's velocity configuration records a posture-tightening experiment that
stopped walking, and then replaces the instantaneous error penalty with a term
on persistent bias. It also ramps action-rate penalties after gait bootstraps.
These are hypotheses for Mesozoic, not established explanations for its STILL
pilot. First compare reward components and action statistics for stillness,
attempted steps and valid motion on one fixed plant.

A proposed screen holds the plant, observation interface, failure checks,
training budget and evaluation panels constant. Compare the unchanged recipe
against one change at a time: an explicitly set exploration level at the
stance-to-walk handoff, or a staged smoothness/posture objective. Do not combine
both in the first contrast. Preserve the declared parent and its normalization;
record any reset of action-distribution parameters as part of the recipe.

Use at least three paired training seeds as an initial screen, with a fixed
checkpoint-selection rule and disjoint confirmation reset seeds. This is a
proposed screen size, not a certification rule. Report each seed's sustained
speed, gait class, valid-support fraction, falls, slip, saturation and reward
components. Keep a zero-action baseline. Freeze decision criteria before the
screen: reward improvement or survival alone cannot select an arm. A changed
plant or task invalidates an otherwise convenient parent and comparison.

Once nominal locomotion works, explicitly sample and evaluate zero command,
start/stop and turn-in-place. Uniform random velocity commands rarely exercise
exact idle or pure yaw. Their sampling fractions should be tuned to this robot's
operating requirements rather than copied from MicroDuck.

## 4. Sensor-realizable robot policy

Propose a new robot policy-interface revision using joint measurements,
IMU-derived gravity and angular velocity, previous action, and body-relative
motion commands. Reuse [heading_observation.py](../environments/shared/heading_observation.py)
where applicable; some anatomical models already use heading-free observations.
That utility does not turn simulator velocity into a sensor measurement.

For every actor field, specify its hardware source or estimator, units, frame,
rate, noise, delay and missing-data behavior. Remove true base velocity unless
an estimator is part of the declared runtime. Replace exact target geometry
with commands supplied by a planner, or define and validate the perception
source. Do not silently substitute a simulated target beacon on hardware.

MicroDuck uses an asymmetric actor/critic: the critic can see privileged state
that the actor cannot. Stock Mesozoic SB3 policies do not acquire this split just
by declaring two observation lists. The first option is a sensor-only actor
and critic; an asymmetric critic is a separately tested SB3 policy/data-path
extension, with a test proving privileged inputs cannot reach actor inference.

Changing the observation semantics requires an interface revision and fresh
training or an explicitly supported migration under the
[plant contract](PLANT_CONTRACT.md). A larger vector padded with zeros is not a
valid semantic migration. Keep the anatomical embodiments' interfaces separate.

## 5. Actuator identification and calibrated randomization

### 5.1 One-actuator benchmark

Begin with a simulation-side trace harness, then collect data from the actual
selected servo, firmware, supply and representative load. MicroDuck's paired
simulation/hardware testbench is the reference method, not a source of motor
constants. Record timestamped target, measured position/velocity, voltage,
available current/temperature, load configuration and calibration identity.
Separate commanded values from firmware-clipped values where observable.

Use steps, direction reversals and bounded frequency sweeps to identify delay,
overshoot, steady-state error, backlash, friction and load/supply dependence.
Fit on one set of traces and validate on held-out trajectories, loads and supply
conditions. Compare against the existing PD model, reporting tracking MAE,
peak error, phase lag and saturation duration. Fix acceptable error budgets
before fitting; this design does not claim an unmeasured tolerance is sufficient.

Keep the fitted model and calibration metadata in the shared robot facilities
required by D-D33. A model change follows physics-version and digest regeneration
rules. A one-servo fit does not establish whole-body contact fidelity or prove
safe walking; those remain separate validation steps.

### 5.2 Randomization after a working nominal controller

Introduce one uncertainty family at a time: actuator fit residuals, contact
friction, then sensor bias/noise and communication delay. Manufacturing
variation on the robot and reconstruction uncertainty on anatomical bodies
need separate justifications and ranges. Do not vary gravity as a substitute
for uncertain mass or motor strength.

Every randomizer must:

1. Sample relative to immutable nominal parameters, independently per environment
   and reproducibly from recorded seeds; repeated resets must not accumulate drift.
2. Preserve valid physics, including consistent positive mass/inertia and
   realizable geometry. Reject impossible parameter combinations.
3. Record the distribution, range, sampled values or reproducible sample identity,
   units and application times in run provenance.
4. Demonstrate with a probe that the intended quantity actually changes; include
   reset-restoration, isolation between environments and disabled-path equivalence.
5. Keep reward/evaluation truth separate from corrupted actor observations, and
   define delay-buffer initialization at reset and policy transitions.

Compare nominal and randomized training on the same nominal panel plus held-out
perturbation panels. Accept robustness only with a predeclared gain on the
targeted failures and bounded nominal regression. Broader ranges are not
automatically better: MicroDuck documents center-of-mass ranges that made
desirable motion physically unattainable.

## 6. Export and deployment contract

Extend the existing policy and result-bundle machinery rather than replacing it.
The deployable unit is an inference bundle, not a checkpoint file alone:

```text
sensor samples -> frame conversion / estimation -> actor normalization
               -> deterministic actor -> clipping / residual action mapping
               -> declared runtime filtering / buffering -> actuator commands

independent simulator truth -> existing gait / stance / task judges
```

This is a conceptual data path; the exact transform order must match the
training environment and be declared in the contract. The bundle records:

| Contract component | Required contents |
|---|---|
| Identity | Species/embodiment, physics and interface revisions/hashes, task/gate identity, checkpoint hash, code revision and actuator calibration identity. |
| Observation | Ordered fields, units/frames, preprocessing, normalizer mean/variance/epsilon/clipping and frozen inference behavior; estimator version where needed. |
| Action | Joint order/signs/offsets, bounds, neutral pose and residual mapping, clipping order, filter coefficients and previous-action meaning (raw, clipped or applied). |
| Timing and state | Policy period, physics/actuator cadence, filter/delay buffers, initialization, reset behavior and transition entry conditions; recurrent memory if introduced. |
| Evidence | Export/tool versions and artifact hash, parity report, closed-loop evaluation, compatibility result and target-computer timing measurements. |

Distinguish intentional runtime filters/buffers from simulated transport and
actuator latency. Reproduce modeled delay in simulation rehearsal, but do not
inject training randomization or add the modeled physical delay a second time
on hardware. Declare any actual software buffer in the runtime contract.

Export deterministic inference, embedding normalization in ONNX or packaging
it as an enforced, inseparable transform. Never normalize twice or deploy raw
observations into an actor trained on normalized inputs. The loader must reject
incompatible contracts, wrong shapes/order, non-finite values and unsupported
state semantics before activation; informational metadata alone is insufficient.

Validation has three levels:

1. **Action parity:** compare the existing SB3 inference path and exported path
   on held-out observations, including near-bound values. Check normalized
   inputs, actor outputs and final actuator targets with identical clipping,
   residual mapping and filter state. Declare numerical tolerances before the
   comparison and test reset sequences, not just isolated vectors.
2. **Closed-loop rehearsal:** run the exported controller in MuJoCo through the
   same I/O abstraction intended for hardware; reuse the current evaluation
   panels and gates. Simulation divergence makes exact long-horizon trajectory
   equality inappropriate; use predeclared behavioral non-regression criteria.
3. **Target runtime:** measure full sensor-to-command latency, deadline misses
   and jitter, including I/O and scheduling. The current robot's 50 Hz period is
   20 ms, not an inference-only latency allowance. Check transitions, stale
   sensors, invalid actions and inference failure against explicit stop behavior.

Enforce independently specified position, rate, current/torque and thermal
limits and an emergency stop for physical trials. MicroDuck's actuator-travel
clamp and skill-specific fall-reflex behavior are not a complete safety design
for another morphology. Rehearsal cannot validate motor-bus faults, real contact
or thermal behavior that it does not model.

## 7. Recovery and policy composition

Keep separately certified policies as the baseline. Add a transition matrix
covering stance-to-walk, walk-to-stop, stop-to-walk and disturbance recovery;
specify entry-state validity, command continuity, previous action, filter state
and timeouts. Physical get-up is an additional task, with collision-complete
training and evaluation on naturally generated fall poses as well as canonical
spawns. It must not be inferred from a push-recovery certificate.

Transition evaluation carries the continuous physical state across policy
switches rather than resetting the simulator. Reset or reseed controller state
only as specified by the transition contract, and test the resulting transient.

Only after specialists work should a combined policy be considered. MicroDuck
documents a recovery teacher destroying walking until a frozen walk teacher
anchors upright behavior. A Mesozoic experiment would preserve each teacher's
normalization and action semantics, define state-dependent imitation regions,
and separately judge walking, recovery and the intervening transition states.
Keep the separate-policy fallback; average reward cannot hide regression of
one specialist skill.

## 8. Proposed delivery sequence and acceptance

These are scoped follow-up work items, not adopted PR numbers or amendments to
the existing work order. The gait plan continues to own locomotion-gate adoption.

| Work item | Dependency and deliverable | Acceptance evidence |
|---|---|---|
| Gait measurement | Existing gait-plan prerequisites; extend shared floor-truth measurement and the proposed locomotion gate. | Invalid gait fixtures rejected; valid declared classes accepted; per-episode results and contact-source checks. |
| Motion-initiation screen | Fixed current plant, permitted parent and frozen judge; compare one reward or exploration change. | Paired seed table, disjoint confirmation, unchanged failure criteria; sustained gait improvement rather than reward alone. |
| Robot interface | Robot sensor/estimator inventory; versioned sensor-realizable observation/action contract. | Field-to-hardware mapping, no actor access to privileged truth, interface invalidation tests and a fresh nominal baseline. |
| Servo benchmark | Chosen actuator/controller; shared trace harness and calibrated model. Simulation scaffolding can precede hardware. | Held-out fit errors against PD baseline, reproducible traces and documented parameter uncertainty. |
| Export rehearsal | Stable declared inference contract; bundle/exporter and simulated deployment I/O. Scaffolding can proceed alongside calibration. | Deterministic action parity, contract rejection cases, closed-loop non-regression and reset/transition tests. |
| Robustness | Working nominal controller and measured uncertainty; seeded randomization and perturbation panels. | No drift/no-op randomizers, reproducibility, held-out gains and acceptable nominal regression. |
| Recovery composition | Working specialist policies and valid transitions; optional distillation experiment. | Separate walking/recovery acceptance plus transition evaluation; preserve fallback. |

Before each experimental item, choose and record the model/revisions, parent,
training seeds/budget, selection rule, untouched confirmation panel, numerical
acceptance bars and rollback criterion. Keep report-only evidence distinct from
certification in [result bundles](RESULT_BUNDLES.md). Changes to plants, tasks,
interfaces or gates use the existing revision/digest rules; this documentation
change moves none of them.

The first useful research deliverable is an independently checked gait judge
and a bounded motion-initiation experiment. The first robot-transfer deliverable
is a traceable servo benchmark plus exported-controller rehearsal. Neither
requires committing to whole-body hardware deployment or a new training backend.

## 9. Deferred choices, risks and alternatives

- **Backend:** keep SB3. If rollout cost later blocks replication, propose an
  isolated mjlab/MuJoCo Warp benchmark only after reopening D-D17 explicitly.
  Match physics, rewards, interfaces and judges; report wall-clock cost and
  behavior across seeds. MicroDuck's stated 4,096-environment training time is
  not a forecast for dinosaur models.
- **Observation architecture:** decide sensor-only actor/critic versus an
  asymmetric SB3 extension from measured learning performance. Recurrent
  policies or observation history are later options for partial observability,
  with explicit memory/reset/export contracts, not default additions.
- **Experiment policy:** select exact gait thresholds, exploration levels,
  smoothing schedules and regression tolerances before running each screen.
  This proposal supplies test structure, not uncalibrated certification bars.
- **Scope and morphology:** reuse methods, not MicroDuck's weights, 61-input /
  14-action layout, motor constants, clearance heights or reward coefficients.
  The current robot has 46 observations / 12 actions; this design does not
  prescribe the dimension of its successor interface.
- **Physical validation:** hardware is still planned. Simulation-only success
  cannot demonstrate sim-to-real transfer or protection under real faults.
- **Licensing:** MicroDuck RL code is Apache-2.0; the covered robot/testbench
  meshes are CC BY-NC-SA 4.0. Review the specific files' licenses before direct
  reuse. This design imports no code, policies or assets.

## 10. Upstream source record

Sources below are public and pinned; repository-local evidence uses relative
links above. No private Drive artifacts are needed to review this proposal.
Source comments describe upstream experience, not experiments performed here.

**MicroDuck RL:** `pollen-robotics/microduck_rl`, `develop`, commit
`b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`.

- [Pipeline, actuator model and export overview](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/README.md).
- [Foot-terrain sensor, gait terms, actor/critic observations, reward curricula and randomization](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py).
- [Single-servo simulation/hardware testbench](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/scripts/testbench_sim2real.py)
  and [actuator model](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/src/mjlab_microduck/actuator/friction_dr_bam.py).
- [Normalizer-aware exporter](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/src/mjlab_microduck/export.py).
- [Recovery failure analysis](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py)
  and [frozen walk/stand teachers](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/src/mjlab_microduck/tasks/distill.py).
- [Mesh-license scope](https://github.com/pollen-robotics/microduck_rl/blob/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3/LICENSE-MODELS.md).

**MicroDuck runtime:** `pollen-robotics/microduck`, `main`, commit
`96b89d8613f06722c1d64ce75ab0d7124fed9d8f`.

- [Production runtime against a simulated body](https://github.com/pollen-robotics/microduck/blob/96b89d8613f06722c1d64ce75ab0d7124fed9d8f/docs/robot/simulation.md).
- [Observation builder](https://github.com/pollen-robotics/microduck/blob/96b89d8613f06722c1d64ce75ab0d7124fed9d8f/duck-control/src/obs.rs)
  and [policy loading/inference](https://github.com/pollen-robotics/microduck/blob/96b89d8613f06722c1d64ce75ab0d7124fed9d8f/duck-control/src/policy.rs).
- [Runtime control and protection scope](https://github.com/pollen-robotics/microduck/blob/96b89d8613f06722c1d64ce75ab0d7124fed9d8f/docs/design/robotd-design.md)
  and [actual actuator-travel clamp](https://github.com/pollen-robotics/microduck/blob/96b89d8613f06722c1d64ce75ab0d7124fed9d8f/duck-control/src/safety.rs).

Executable configurations take precedence over stale overview comments. For
example, the reviewed VelStand configuration uses the full-collision model
despite the README's claim that no task uses it. Do not copy such summary claims
into implementation requirements without checking the pinned code.
