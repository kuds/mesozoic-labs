# Splitting Stage 1 into Stance (1a) and Recovery (1b)

Design proposal to replace the single balance stage with two: **1a — stance**, reaching and
holding a stable pose, and **1b — recovery**, holding it against external disturbance.

Follows from [investigations/TREX_REVIEW_2026_07.md](investigations/TREX_REVIEW_2026_07.md)
§F1 and §NS-1, and from the measurements recorded in
[PR #471](https://github.com/kuds/mesozoic-labs/pull/471).

Revision 2 incorporates review feedback on
[PR #474](https://github.com/kuds/mesozoic-labs/pull/474); §11 records what changed and why.
Every claim below carries a label from the ledger in §11.

## TL;DR

Stage 1 currently asks one number to answer two questions: *did the plant reach a stable
stance*, and *is that stance actively controlled rather than passively propped*. The first is
legitimately satisfied by a controller that settles and then stops working. The second is not
measurable at all without a disturbance. No gate on episode return can separate them, which is
why the gate has been either unbindable or unclearable in every configuration tried so far.

Splitting gives each question its own stage and its own gate:

| | stage 1a — stance | stage 1b — recovery |
|---|---|---|
| task | settle from randomised init, hold pose | same, plus scheduled external shoves |
| perturbation | none | `xfrc_applied`, schedule TBD |
| settle-then-passive controller | **passes, by design** | fails |
| literal zero action | **fails** — 57.5% full-horizon vs 90% required | fails — 0 of 40 |
| gate is about | stance quality, held to the horizon | recovery from disturbance |

The gate-calibration problem that has consumed this investigation becomes tractable. 1a stops
asking return to discriminate stance quality, and 1b's null is separated from any plausible
candidate by a wide margin rather than by a few percent of a noisy mean.

**Two clarifications from review.** A *statue* does not pass 1a — literal zero action reaches
the full horizon in only 57.5% of episodes, well below the 90% proposed in §2.3. What passes by
design is a controller that corrects its spawn perturbation and then becomes passive; that is a
different and weaker null. And 1b's null is not *zero* survival: 0 successes in 40 episodes
bounds zero-action survival at **≤ 7.216%** (exact one-sided 95%), not at 0. That is still
strong separation from a 70% requirement, but it is a bound, not a certainty.

## 1. Motivation

### 1.1 The two-jobs problem

`configs/trex/stage1_balance.toml` describes stage 1 as "learn to stand and balance without
falling." That is two capabilities:

* **Reach and hold a stable configuration** from a randomised initial state
  (`reset_noise_scale = 0.1`). Non-trivial — the policy must correct its own spawn
  perturbation — but achievable by a controller that converges to the home pose and stops
  working.
* **Actively reject deviation.** Not exercised at all, because the plant is passively stable at
  the home keyframe and nothing ever displaces it.

A single scalar gate cannot rank both. The evidence is in §1.2.

### 1.2 Why no reward term fixes this  `[artifact-derived]`

At any static equilibrium the centre of pressure lies exactly under the centre of mass — forced,
since at rest the ground reaction must pass through the CoM or there would be a net moment.
TREX_REVIEW_2026_07 measured this to **0.2 mm over 6126 statue steps**.

Consequently any bounded per-step function meaning "well balanced" is *maximised by standing
perfectly still*. §NS-1 built four candidate terms and had each attacked by two independent
verifiers; the three per-step candidates were all refuted on the same measurement — the statue
collects the term at least as much as the trained policy:

| candidate | statue | trained policy | winner |
|---|---|---|---|
| two-sided height error | −1.25 | −51.57 | statue by 50.32 |
| capture-point / support-polygon containment | −392.86 | −1792.58 | statue by 1399.72 |
| potential-based CoM-velocity shaping | +0.99/ep | −1.45/ep | statue by 2.44 |

Three unrelated mathematical families, one result. This is structural, not three tuning misses.

### 1.3 The current stance reward saturates against a statue  `[measured]`

`48fd90a`, zero action, `reset_noise_scale = 0`, settled window of a 1000-step episode:

```
settled pelvis_z 0.9260   head_tip_z 0.9444   foot forces R 420.5 N  L 420.5 N

term                        weight    value   frac of weight
reward_alive                  1.00   1.0000        1.000
reward_bilateral_support      0.60   0.6000        1.000
reward_height                 0.60   0.6000        1.000
reward_head_clearance         0.35   0.3500        1.000
reward_heading                0.10   0.0998        0.998
reward_leg_home_pose          0.50   0.4903        0.981
reward_neck_posture           0.20   0.1574        0.787
reward_smoothness                —   0.0000    (exactly 0)
reward_energy                    —   0.0000    (exactly 0)
TOTAL                                3.294 /step
```

Every positive term is collected essentially in full by a plant doing nothing, and the two
action-cost terms are exactly zero because a constant action has no action delta.

### 1.4 It is not a T-Rex problem  `[measured]`

Zero-action baseline, all four species, `48fd90a`, 40 episodes, seed 3042:

```
species              mean     len   gate  len gate   reward?   length?   FULL GATE
trex               1971.6   638.1   1840       750    CLEARS    blocks   blocked
velociraptor       1704.9   977.5    100       750    CLEARS    CLEARS   STATUE PASSES
brachiosaurus       108.2    98.7    100       750    CLEARS    blocks   blocked
dibothrosuchus     1702.0   674.2    100       750    CLEARS    blocks   blocked
```

Every stage-1 **reward** threshold is cleared by a statue. Velociraptor's **complete** gate is
cleared by a statue. In the other three, `min_avg_episode_length` is doing all of the gating
work and the reward threshold is decorative.

### 1.5 The observed policy appears to hop rather than stand  `[measured diagnostics, inferred behaviour]`

Final stage-1 diagnostics from run `20260729_151044` (6,004,736 steps):

| metric | start | final |
|---|---|---|
| `bilateral_support_duty` | 0.084 | 0.730 |
| `single_support_duty` | 0.223 | **0.061** |
| `unsupported_duty` | 0.692 | **0.209** |

Walking requires sustained single support. These readings show the plant alternating between
both-feet-loaded and neither-foot-loaded, with single support driven down 3.6× over training.
The reward makes single support the worst available state, which would explain it:

| state | `bilateral_support` | `foot_load_balance` | sum |
|---|---|---|---|
| both feet down, even | +0.600 | −0.000 | **+0.600** |
| airborne | 0.000 | −0.000 | **0.000** |
| one foot carries load | 0.000 | −0.300 | **−0.300** |

**Status of this claim.** The `[0, 0]` arithmetic hole in `reward_foot_load_balance` is
`[measured]` — verified directly against the shipped weights. That the policy *is* hopping, and
that the reward *caused* it, remain `[inferred]` until the touch sensors are cross-checked
against `mj_contactForce` and kinematic flight phases (§7.2). The duty figures are sensor
readings; if the sensors under-report, the behavioural conclusion changes even though the
formula defect does not.

## 2. Stage 1a — Stance

### 2.1 Objective

From a randomised initial state, converge to a stable upright pose and hold it to the horizon.

**What may legitimately pass:** a controller that corrects reset randomisation and then becomes
passive. That is the intended null for this stage and it is *not* a defect — settling is the
capability 1a exists to certify.

**What does not pass:** literal zero action, which reaches the full horizon in only 57.5% of
episodes against the 90% proposed below. The two are often conflated; they are different
policies with different scores.

### 2.2 Configuration

Unchanged from today's `stage1_balance.toml`, except:

```toml
[env]
perturbation_delta_v = 0.0        # explicit; 1a is the undisturbed control
```

Keep `reset_noise_scale = 0.10`. §NS-1 correction 1 measured that dropping to 0.05 lets the
statue win outright (2568.7 at 90% full-horizon against the checkpoint's 2498.8), reversing the
shipped config's edge to the policy. `[artifact-derived]`

### 2.3 Gate

**Stop gating 1a on beating a statue.** Episode return in this stage is ~100% collectable by a
passive controller (§1.3), so any threshold either admits one or excludes a working policy.

Proposed episode-level success event, evaluated per episode and then aggregated:

```
stance_success =
    full_horizon
    and settles_by        <= T_settle
    and tail_q95(height_error)      <= h_max
    and tail_q95(orientation_error) <= angle_max
    and tail_q95(planar_speed)      <= v_max
    and tail_q95(angular_speed)     <= omega_max
    and tail_drift_rate             <= d_max
```

Gate on a one-sided lower confidence bound of `P(stance_success)`, not on a mean of means.
Quantiles over a defined tail window rather than episode averages, because averages hide
oscillation and large transients — which is precisely the failure mode in §1.5.

**This is new machinery, not a config change.** `StageThreshold`
(`environments/shared/curriculum.py:79-84`) currently supports exactly `min_avg_reward`,
`min_avg_episode_length`, `min_avg_forward_vel`, `min_success_rate`, `min_eval_episodes` and
`required_consecutive`. `[measured]` Revision 1 of this document said the proposed quantities
"are already logged"; that conflated diagnostic logging with canonical gate evidence. Before any
of this can gate advancement, the following must be specified and implemented:

* per-step → per-episode → per-evaluation aggregation, explicitly;
* whether failed episodes contribute, and with what value;
* the settling and tail window definitions;
* whether drift means final, maximum, time-averaged or cumulative displacement;
* per-species thresholds or morphology-normalised ones — `height_error` is currently T-Rex
  instrumentation, not a four-species advancement metric;
* SB3 and MJX/JAX parity;
* persistence through reporting, result bundles, notebook and sweeps.

Provisional T-Rex values, to be calibrated rather than adopted: `T_settle` 200 steps,
`h_max` 0.03 m, `d_max` 0.5 m, `P(stance_success)` LCB ≥ 0.90. `[inferred]`

`min_avg_reward` is **unset** for 1a.

### 2.4 Budget

Provisional 3M steps, down from 6M. Run `20260729_151044` reached 1000-step episodes with
`height_error` 0.009 by ~3.5M under the *current* reward, which carries posture shaping 1a does
not need. Confirm against a pilot. `[inferred]`

## 3. Stage 1b — Recovery

### 3.1 Objective

Hold the stance from 1a against scheduled external disturbance, and demonstrate *recovery* —
returning to a safe pose/velocity/contact set after each shove — not merely survival.

### 3.2 Perturbation mechanism

Per §NS-1: a runtime write to `data.xfrc_applied[root, 0:3]`. No reward term, no observation
change.

* new `_apply_perturbation()` in `environments/shared/base_env.py`, called at the top of `step()`
* pure `external_push_force()` kernel in `environments/shared/reward_functions.py` so SB3 and
  JAX/MJX share the force arithmetic
* new `perturbation_*` keys in `[env]`, defaulting to `0.0` for every other species and stage

```toml
[env]
perturbation_delta_v      = 1.5     # multiple of capture-point velocity
perturbation_interval     = 2.0     # seconds between shoves
perturbation_jitter       = 0.5     # +/- seconds, defeats a blind clock-timed brace
perturbation_duration     = 0.20    # seconds of applied force
perturbation_direction    = "uniform_horizontal"
```

On the T-Rex plant `1.5×` capture-point velocity is roughly **150 N for 0.20 s**.
`[artifact-derived]`

**Checkpoint compatibility — narrower than revision 1 claimed.** `plant_contract.py:895-921`
fingerprints observation/action implementations and selected reset semantics; `step` appears
zero times in the interface payload, so a `step()` hook moves no `policy_interface_revision`.
`[measured]` Revision 1 concluded from this that the change "invalidates no existing
checkpoints." That is too broad. A pushed task changes the transition kernel and the evaluation
distribution: existing checkpoints stay *mechanically loadable and interface-compatible* while
being *unvalidated for the new task*. The fingerprint's silence about `step` is a provenance
gap, not evidence of task equivalence.

Add a distinct **task/evaluation fingerprint** covering perturbation implementation, schedule,
RNG protocol, force parameters, reset configuration, horizon, reward and termination semantics,
and backend. Resume must not cross that boundary silently.

**Scheduler requirements** — a shared force kernel does not give backend-neutral scheduling.
The design must specify: explicit clearing of `xfrc_applied` after each pulse and on reset;
deterministic episode-local push times and directions; pre-generated schedules so baseline and
policy receive *identical* disturbances; MJX `data.replace(xfrc_applied=...)` and auto-reset
clearing; CPU-JAX evaluation parity; one schedule unit shared by SB3 and thousands of parallel
MJX environments; and persisted/restored ramp progress on resume. Calibration and advancement
always run at frozen full strength, never partway through a ramp.

### 3.3 Ramp versus fixed — an open question, not a decision

§NS-1 correction 2 recommends a fixed impulse, attributing the problem to `set_reward_weight`
(`base_env.py:752`) being a bare `setattr`. Revision 1 of this document repeated that.
**Both are wrong about the mechanism.** `RewardRampCallback`
(`environments/shared/curriculum.py:794-877`) already computes linearly interpolated values from
global timesteps and propagates them periodically via `env_method`; the setter merely applies
what the callback computed. `[measured]`

The actual missing piece is a *dynamic perturbation-scale input* with one defined unit across
backends and defined resume behaviour. That is a real gap, but a different one.

Whether ramping prevents catastrophic forgetting at the 1a → 1b boundary is a **hypothesis**
`[inferred]`, to be settled by the transfer pilot in §8.1, not assumed.

### 3.4 Gate — thresholds provisional pending measurement

§NS-1 measured the statue under push: `[artifact-derived, stale]`

| | statue | trained checkpoint |
|---|---|---|
| no push | 1743.73, 57% full-horizon | 2489.65, 100% |
| push, noise 0.05 | **711.05 ± 403.76, 0 of 40** | 2418.38 ± 357.61, 85% |
| push, noise 0.10 | **604.18 ± 483.99, 0 of 40** | **not measured** |

Three cautions on reading this, all verified by exact binomial calculation `[measured]`:

1. **0 of 40 is a bound, not zero.** The exact one-sided 95% upper bound on zero-action survival
   is **0.07216** (equivalently `1 − 0.05^(1/40)`). Strong separation from a 70% requirement,
   but the correct statement is "bounded below ~7.2%," not "zero."
2. **The candidate evidence is thinner than it looks.** 85% is 34 of 40; its exact one-sided 95%
   lower bound is **0.72526** — only narrowly above a 0.70 gate. And it was measured at noise
   0.05, while §2.2 retains 0.10, where the checkpoint is unmeasured.
3. **Repeating a deterministic seed panel three times is process stability, not three
   independent confirmations.** `required_consecutive = 3` should not be read as statistical
   replication.

Proposed 1b gate, on the episode-level recovery event from §3.1:

```
LCB95( P(full horizon and every shove recovered) )        >= p_recovery
LCB95( mean(policy_success_i - zero_success_i) )          >= Δ_success   # paired, same schedule
```

with paired unconditional reward optionally retained as a *secondary* criterion. Per-shove
recovery = re-entering the safe set within `T_recover` and dwelling there.

All of `p_recovery`, `Δ_success`, `T_recover` and the `800`/`0.70`/`3` figures from revision 1
are **provisional** until the finalised pushed task is measured (§8.1, §8.2). The push figures
above also predate `435f35f`.

**Null suite.** Zero action alone is insufficient — survival does not prove feedback control.
Calibrate against zero action, constant/brace controllers, *and* the incoming 1a checkpoint.

### 3.5 Budget

Provisional 3M steps, warm-started from the 1a checkpoint. `[inferred]`

## 4. Stage identity — semantic IDs, not renumbering

Revision 1 proposed renumbering 2→3 and 3→4 and called it "mechanical." It is a schema
migration. Verified blockers `[measured]`:

* `environments/shared/config.py:154` — `_STAGE_FILE_PREFIX = {1: "stage1_", 2: "stage2_", 3: "stage3_"}`; stage 4 raises `KeyError`.
* `environments/shared/reporting.py:968` — result bundles reject any stage set not a subset of `{1, 2, 3}`.
* `train_base.py:1285` assumes `stage < 3`.

Renumbering also silently changes the historical meaning of "stage 2" and "stage 3" in every
existing run summary, bundle and website record.

**Prefer stable semantic identifiers** with a separate display/order field:

```
stance
recovery
locomotion
behavior
```

plus a schema-version bump and backward readers for existing three-stage artifacts. Enable
`recovery` for T-Rex only at first, and per species thereafter only once that species has
task-matched evidence.

## 5. Gate resolution

Both stages should resolve thresholds from a measured baseline rather than a TOML literal. The
lifecycle:

1. Materialise the fully effective reward/environment/perturbation/backend config.
2. Measure or validate a compatible baseline on a registered seed vector.
3. Resolve once, atomically persist `gate_resolution.json` with full provenance.
4. Put the finite resolved values into an immutable run config.
5. Pass that snapshot to SB3, JAX/MJX, notebook, reporting, visualization, bundles, sweeps.
6. No executable consumer reopens raw TOML after resolution.
7. Resume loads the frozen gate; it never recomputes in place.
8. A changed commit, config, backend **or task fingerprint** (§3.2) is a new run and recalibrates.

Missing, stale, or incompatible baseline data must **block** advancement rather than silently
falling back to a literal.

### 5.1 If a reward threshold is retained, the paired test is authoritative

An earlier revision proposed `reward_mean_standing × 1.055`. That is wrong: the policy is gated
on *unconditional* mean return while `reward_mean_standing` is conditioned on full-horizon
survival, so conditioning removes the failure mode the policy is supposed to eliminate. Measured
counterexample — over 120 seed-matched episodes the trained policy beat zero action by
**+568.02** with survival **118/120 against 68/120**, while sitting 677–775 points below the
survivor-conditioned statue mean. `[artifact-derived]` A standing-floor gate would reject a
policy that is unambiguously better than doing nothing.

Revision 1 replaced it with a *conjunction* of an unpaired scalar and a paired test:

```
G_run = max(configured_literal, UCB95(E[R_zero]) + Δ_abs)     # revision 1 — do not use
```

That has two defects. Requiring both means the unpaired scalar can reject a candidate whose
paired improvement is precise and positive, discarding the variance reduction that motivated
pairing in the first place. And `max(configured_literal, …)` carries a legacy, task-dependent
number into a new pushed task as an implicit fallback.

```
D_i         = R_policy(seed_i) - R_zero(seed_i)      # identical seeds and push schedule
pass_reward = LCB95(mean(D_i)) >= Δ_R                # authoritative

G_screen    = UCB95(E[R_zero]) + Δ_R                 # display/screening only, never overrides
```

A configured literal may be retained only as an **explicit, fingerprinted safety floor**, never
as an unexplained operand or silent fallback. With `Δ_R = 0` this establishes statistical
superiority only, not practical usefulness; any nonzero effect size needs a task-based rationale
rather than a percentage inherited from rounding.

The unconditional mean needs a confidence bound because it is genuinely noisy — across three
disjoint 40-seed blocks it moved 1971.57 / 1968.72 / 1884.71 (spread 86.9) while the standing
mean moved 3244.04 / 3250.45 / 3233.99 (spread 16.5). `[artifact-derived]`

For 1b, reward should be secondary to directly measured recovery capability regardless.

### 5.2 Blocking pre-flight

Worth shipping before the resolver: make the §3b notebook cell raise instead of print, and
evaluate the **full joint predicate** against the null suite rather than the reward
sub-threshold alone. §1.4 shows the current one-sided check reports `FAILS` for all four species
while only velociraptor's complete gate is actually statue-clearable — directionally right,
quantitatively misleading.

## 6. Diagnostics

Both stages need instruments that separate *standing* from *not yet fallen*, and *standing* from
*hopping*. Ordered by value:

1. **Every metric reported as a margin over the measured null.** Put the floor on the eval plot
   as a horizontal line and in `training_summary.txt`.
2. **Ground-reaction-force check against body weight.** Time-averaged total GRF must equal body
   weight for periodic motion — a physics invariant, so deviation is a sensor or accounting bug.
   Log `mean(total_contact_force) / (m·g)` and alarm outside `[0.95, 1.05]`. This would already
   be firing: the statue's static total is 841 N while the policy's logged mean is 1460 N.
   `[measured]`
3. **Support-state transition matrix** over `{bilateral, single-L, single-R, airborne}`. A walk
   is dominated by `single-L ↔ single-R`, a bounce by `bilateral ↔ airborne`.
4. **Centre-of-pressure position, excursion and velocity** — the biomechanical definition of
   balance, and the primary success metric for 1b.
5. **Recovery time after each shove** (1b only) — steps from impulse to CoP re-entering the
   support polygon, plus dwell. The single most legible number this stage can produce.
6. **Vertical oscillation and flight-phase count** — `std(pelvis_z)` and peak-to-peak alongside
   the mean. Mean pelvis height is 0.932 against the statue's 0.926, so the mean hides the
   entire behaviour. `[measured]`
7. **Fix `alternation_ratio`, or stop reporting it.** Verified against the shipped
   `_compute_gait_symmetry`: synchronized bounce **1.000**, true alternating walk **1.000**,
   statue **1.000**, limp 0.684. `[measured]` Root cause at `base_env.py:512-515` — a
   simultaneous two-foot landing appends `"R"` then `"L"`, so every bounce reads as a textbook
   alternation. Record a simultaneous touchdown as one `BOTH` event.

## 7. Prerequisites

### 7.1 Fix the airborne hole in `reward_foot_load_balance`  `[measured]`

`|R−L| / (R+L+1e-8)` returns 0 when both feet read zero, making airborne strictly cheaper than
single support (§1.5). Under a disturbance this matters more, not less: a policy that leaves the
ground cannot reject a shove mid-flight.

```python
total = right_force + left_force
imbalance = xp.where(total > min_support_force,
                     xp.abs(right_force - left_force) / (total + 1e-8),
                     1.0)
```

Needs the matching JAX-path edit for parity, and `[0, 0]` cases in `test_reward_functions.py`
and `test_trex_mjx_reward_parity.py`; neither covers it today.

### 7.2 Verify the foot touch sensors — gates §7.1, §6.3 and §1.5

`unsupported_duty = 0.209` on a plant holding 0.93 m pelvis height that never falls is more
consistent with sensor under-reporting than with 21% airtime. Cross-check touch-sensor sums
against `mj_contactForce` *and* kinematic flight phases during a policy rollout, per OQ-6. The
formula defect in §7.1 is real regardless; the behavioural story in §1.5 depends on this.

### 7.3 Give diagnostic tooling explicit task modes

Revision 1 said to force `perturbation_delta_v = 0.0` in every diagnostic script. That
recreates the incompatible-baseline problem it was meant to prevent: 1b's gate must be
calibrated against the *pushed* floor, and a tool that silently disables the configured task
cannot produce it.

```
plant_sanity : perturbation forced off      — nominal plant-integrity control
task_gate    : perturbation exactly matches advancement evaluation
```

The mode and the complete perturbation fingerprint must be persisted with every measurement. A
tool must never silently turn the configured task off.

Affected tools that build the env from TOML: `zero_action_baseline`, `joint_excursion_report`,
`action_bound_report`, `observation_ablation_report`. **`actuator_saturation_report` is not
affected** — it loads raw XML via `mujoco.MjModel.from_xml_string` and steps MuJoCo directly
(`environments/shared/scripts/actuator_saturation_report.py:44-76`). `[measured]` Revision 1
inherited that error from §NS-1 correction 4.

### 7.4 Decouple the collapse detector  `[measured]`

`curriculum.py:1014` falls back to `min_avg_reward` for `peak_floor` when `collapse_peak_floor`
is unset. `configs/trex/stage1_balance.toml:164` sets it to `2200.0`; the other eleven stage
configs do not. Since 1a removes `min_avg_reward` entirely, the fallback becomes undefined
there — set `collapse_peak_floor` explicitly per stage, or decouple it, **before** removing the
reward gate. Single-stage pilots still install plateau/collapse callbacks, so "advancement
disabled" does not make an inherited threshold inert.

## 8. Open questions — measure before committing

1. **Does a 1a policy transfer into 1b at all?** If the stance 1a learns is passive enough that
   the first shove destroys it, the split buys nothing over training with the push from the
   start. Test: take `robust_best_model.zip` from `20260729_151044`, enable the push, evaluate.
   Also §NS-1's own missing load-bearing number (checkpoint at noise 0.10, push on) — a
   ~10-minute eval, not a training run.
2. **Re-measure every push figure at current `main`,** on a registered seed schedule, then on a
   held-out block. All of §3.4 predates `435f35f`, which moved the undisturbed statue +227.84
   mean / +409.09 standing with byte-identical trajectories. `[measured]`
3. **Does the perturbation penalise the hop?** A policy airborne 21% of the time cannot reject a
   shove mid-flight, so 1b should select against bouncing. `[inferred]`
4. **Ramp or fixed?** §3.3 — decide by pilot.
5. **Is 3M steps enough for 1a?** §2.4.
6. **Do the §2.3 thresholds admit the current checkpoint?** Proposed from observed values, not
   calibrated.

## 9. Risks

| risk | mitigation |
|---|---|
| Catastrophic forgetting at the 1a → 1b boundary | ramp *if* the §8.1 pilot supports it; the task change is otherwise small — same reward, same plant, one force |
| Added wall-clock for a fourth stage | 1a shortened to ~3M (§2.4); partly offsets |
| Stage-identity migration across four species and the website | semantic IDs + schema bump (§4); land as a separate prep change |
| 1b unlearnable if a species' 1a is weak | per-species `perturbation_delta_v`; enable per species only after its own preflight |
| Brachiosaurus zero action never reaches the horizon (0 of 40, `n_standing = 0`) | its 1a gate will fail, which is the `CHECK PLANT` signal surfacing. Note this is a *zero-action* result: it does not by itself show that no learned controller can stand, nor prove plant corruption. Investigate before concluding either. |

## 10. Sequencing

1. §7.2 sensor verification — gates §7.1, §6.3 and the §1.5 behavioural claim.
2. §7.4 collapse decoupling — must precede removing any reward gate.
3. §3.2 task fingerprint and §4 semantic IDs with backward readers.
4. §7.1 `foot_load_balance` fix, with parity tests.
5. §2.3 episode-level gate metrics, implemented and parity-tested.
6. Deterministic perturbation scheduler, with force-off regression, clearing, seed/schedule,
   SB3/MJX and resume tests. Default `0.0` — no behaviour change until enabled.
7. §8.1 T-Rex evaluation: zero action, constant/brace controls, and the existing checkpoint at
   noise 0.10 under the finalised full push, on one registered 40-seed schedule; repeat on a
   held-out block.
8. Calibrate thresholds; decide ramp versus fixed.
9. Stage split enabled for T-Rex only; other species after their own preflights.
10. Gate resolver (§5) before, or atomically with, any executable stage depending on it.

Steps 1, 2 and 4 are measurement and small fixes. None require the split, and all should happen
regardless.

## 11. Claim ledger

| label | meaning |
|---|---|
| `measured` | reproduced directly against `48fd90a` for this document |
| `artifact-derived` | taken from run artifacts or TREX_REVIEW_2026_07 §NS-1; not re-run here |
| `inferred` | reasoning, not measurement |
| `stale` | measured before `435f35f` changed the stage-1 reward; needs re-measuring |

**`measured` here:** the four-species baseline and full-conjunction tables; the statue per-term
decomposition and settled plant constants; the foot-state cost table and the `[0, 0]` hole; the
`alternation_ratio` gait test; the `ec23125` → `48fd90a` counterfactual; the exact binomial
bounds in §3.4; and the code citations in §2.3, §3.2, §3.3, §4, §7.3 and §7.4.

**`artifact-derived`:** the §NS-1 refuted-candidate table, the CoP-under-CoM measurement, the
push-on figures (also `stale`), the anti-gaming searches, the 120-seed paired comparison, and
the three-block seed stability figures.

**Not yet reproducible from this repository:** the push-on measurements in §3.4. The
perturbation implementation, exact schedule and raw per-episode outcomes are not present, so
those numbers cannot be independently verified until §10.6 lands. They are marked `stale` for
the additional reason that they predate `435f35f`.

**Outstanding provenance work.** Each retained load-bearing number should carry its command,
exact effective config, backend, dependency versions, ordered seeds, raw episode data and
artifact hash. This document groups provenance by claim class rather than per number; a
per-measurement manifest is the next improvement.
