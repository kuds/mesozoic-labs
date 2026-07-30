# Splitting Stage 1 into Stance (1a) and Recovery (1b)

Design proposal to replace the single balance stage with two: **1a — stance**, reaching and
holding a stable pose, and **1b — recovery**, holding it against external disturbance.

Follows from [investigations/TREX_REVIEW_2026_07.md](investigations/TREX_REVIEW_2026_07.md)
§F1 and §NS-1, and from the measurements recorded in
[PR #471](https://github.com/kuds/mesozoic-labs/pull/471).

## TL;DR

Stage 1 currently asks one number to answer two questions: *did the plant reach a stable
stance*, and *is that stance actively controlled rather than passively propped*. The first is
legitimately satisfied by something statue-like. The second is not measurable at all without a
disturbance. No gate on episode return can separate them, which is why the gate has been either
unbindable or unclearable in every configuration tried so far.

Splitting gives each question its own stage and its own gate:

| | stage 1a — stance | stage 1b — recovery |
|---|---|---|
| task | settle from randomised init, hold pose | same, plus scheduled external shoves |
| perturbation | none | `xfrc_applied`, ramped |
| a statue | **passes, by design** | **fails — 0 of 40 full-horizon** |
| gate is about | survival, height, drift | survival under disturbance |
| calibration difficulty | low — stop asking it to discriminate | low — the null is zero |

The gate-calibration problem that has consumed this investigation largely dissolves. 1a stops
pretending to measure balance, and 1b's null policy scores zero survival, so its threshold needs
no statistical machinery to defend.

## 1. Motivation

### 1.1 The two-jobs problem

`configs/trex/stage1_balance.toml` describes stage 1 as "learn to stand and balance without
falling." That is two capabilities:

* **Reach and hold a stable configuration** from a randomised initial state
  (`reset_noise_scale = 0.1`). Non-trivial — the policy must correct its own spawn
  perturbation — but achievable by a controller that converges to the home pose and stops.
* **Actively reject deviation.** Not exercised at all, because the plant is passively stable at
  the home keyframe and nothing ever displaces it.

A single scalar gate cannot rank both. The evidence is in §1.2.

### 1.2 Why no reward term fixes this

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

### 1.3 The current stance reward saturates against a statue

Measured on `48fd90a`, zero action, `reset_noise_scale = 0`, settled window of a 1000-step
episode:

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
action-cost terms are exactly zero because a constant action has no action delta. Adding
posture terms raises the floor at least as much as the ceiling.

### 1.4 It is not a T-Rex problem

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

### 1.5 The observed policy is not standing

Final stage-1 diagnostics from run `20260729_151044` (6,004,736 steps):

| metric | start | final |
|---|---|---|
| `bilateral_support_duty` | 0.084 | 0.730 |
| `single_support_duty` | 0.223 | **0.061** |
| `unsupported_duty` | 0.692 | **0.209** |

Walking requires sustained single support. This policy alternates between both-feet-down and
**neither**-foot-down: a two-footed hop in place. Training drove `single_support_duty` down
3.6×, because the reward makes single support the worst available state:

| state | `bilateral_support` | `foot_load_balance` | sum |
|---|---|---|---|
| both feet down, even | +0.600 | −0.000 | **+0.600** |
| airborne | 0.000 | −0.000 | **0.000** |
| one foot carries load | 0.000 | −0.300 | **−0.300** |

(`reward_foot_load_balance` computes `|R−L| / (R+L+1e-8)`, which returns 0 when both feet read
zero. Tracked separately — see §7.1.)

This is the capability gap the split is meant to expose: a hopping controller and a balanced
one are nearly indistinguishable under the current task, and obviously different under a shove.

## 2. Stage 1a — Stance

### 2.1 Objective

From a randomised initial state, converge to a stable upright pose and hold it for the full
horizon. **A passive controller passing this stage is correct behaviour, not a defect.**

### 2.2 Configuration

Unchanged from today's `stage1_balance.toml`, except:

```toml
[env]
perturbation_delta_v = 0.0        # explicit; 1a is the undisturbed control
```

Keep `reset_noise_scale = 0.10`. §NS-1 correction 1 measured that dropping to 0.05 lets the
statue win outright (2568.7 at 90% full-horizon against the checkpoint's 2498.8), reversing the
shipped config's edge to the policy.

### 2.3 Gate

**Stop gating 1a on beating a statue.** Gate on the properties 1a is actually for:

| criterion | proposed | rationale |
|---|---|---|
| `min_avg_episode_length` | 950 of 1000 | survival is the real 1a requirement |
| `min_full_horizon_share` | 0.90 | *new field* — mean length hides a bimodal mix |
| `max_avg_height_error` | 0.03 m | *new field* — holds the target pose |
| `max_avg_drift_distance` | 0.5 m | *new field* — does not wander |
| `min_avg_reward` | unset | cannot discriminate; see §1.2 |

Rationale for dropping the reward threshold entirely rather than recalibrating it: §1.3 shows
episode return in this stage is ~100% collectable by a statue, so any value either admits a
statue or excludes a working policy. The measured proxies above are what the stage is for, and
each is already logged.

This sidesteps the standing-floor-versus-unconditional-mean dispute in PR #471 by declining to
gate 1a on return at all.

### 2.4 Budget

Provisional 3M steps, down from 6M. Run `20260729_151044` reached 1000-step episodes with
`height_error` 0.009 by ~3.5M under the *current* reward, which carries posture shaping 1a does
not need. Confirm against a pilot before fixing.

## 3. Stage 1b — Recovery

### 3.1 Objective

Hold the stance from 1a against scheduled external disturbance.

### 3.2 Perturbation mechanism

Per §NS-1: a runtime write to `data.xfrc_applied[root, 0:3]`. No reward term, no observation
change.

* new `_apply_perturbation()` in `environments/shared/base_env.py`, called at the top of `step()`
* pure `external_push_force()` kernel in `environments/shared/reward_functions.py` so SB3 and
  JAX/MJX share one implementation
* new `perturbation_*` keys in `[env]`, defaulting to `0.0` for every other species and stage

**It must go in `step()`, not `reset()`.** `plant_contract.py:916` hashes
`_callable_semantics(env.reset)` into `policy_interface_revision`; `step` appears zero times in
the interface payload. A `step()` hook therefore moves no fingerprint and **invalidates no
existing checkpoints**.

```toml
[env]
perturbation_delta_v      = 1.5     # multiple of capture-point velocity
perturbation_interval     = 2.0     # seconds between shoves
perturbation_jitter       = 0.5     # +/- seconds, defeats a blind clock-timed brace
perturbation_duration     = 0.20    # seconds of applied force
perturbation_direction    = "uniform_horizontal"
perturbation_ramp_updates = 200     # ramp magnitude 0 -> full over the first N updates
```

On the T-Rex plant `1.5×` capture-point velocity is roughly **150 N for 0.20 s**.

### 3.3 Ramp rather than step

§NS-1 correction 2 recommends shipping the impulse fixed rather than ramped, because
`set_reward_weight` (`base_env.py:752`) is a bare `setattr` and a ramp callback would produce a
step function. That is an implementation limitation, not a principle. Recomputing the force
inside `_apply_perturbation` on every call makes a true ramp possible, and a ramp is worth
having here: it removes the difficulty cliff at the 1a → 1b boundary, which is where
catastrophic forgetting would otherwise bite.

Ramp magnitude from 0 to full over `perturbation_ramp_updates`, then hold.

### 3.4 Gate

This is where "beat the null" becomes meaningful, and the null is nearly free to establish.
§NS-1 measured the statue under push:

| | statue | trained checkpoint |
|---|---|---|
| no push | 1743.73, 57% full-horizon | 2489.65, 100% |
| push, noise 0.05 | **711.05 ± 403.76, 0 of 40** | 2418.38 ± 357.61, 85% |
| push, noise 0.10 | **604.18 ± 483.99, 0 of 40** | **not measured** |

The standing statue ceases to exist. Proposed gate:

| criterion | proposed |
|---|---|
| `min_full_horizon_share` | 0.70 |
| `min_avg_episode_length` | 800 |
| `min_avg_reward` | resolved from the measured pushed floor, or unset |
| `required_consecutive` | 3 |

With the null at 0 of 40 survival, a survival-share threshold needs no confidence-interval
argument to defend. If a reward threshold is retained, resolve it per §5 rather than hardcoding.

**Note:** the push figures above predate `435f35f` and must be re-measured. See §8.

### 3.5 Budget

Provisional 3M steps, warm-started from the 1a checkpoint.

## 4. Stage renumbering

| today | proposed | id |
|---|---|---|
| stage 1 — balance | stage 1 — stance | `stage1_stance` |
| — | stage 2 — recovery | `stage2_recovery` |
| stage 2 — locomotion | stage 3 — locomotion | `stage3_locomotion` |
| stage 3 — bite/strike/snap/food_reach | stage 4 — *(unchanged names)* | `stage4_*` |

Touches `configs/*/`, `environments/shared/species_catalog.py`,
`environments/shared/curriculum.py`, `reporting.py`, `visualization.py`, `result_bundle.py`,
the sweep code, `notebooks/sb3_training.ipynb`, and `website/src/data/species.generated.json`.
Mechanical but non-trivial across four species; several call sites assume `stage < 3`
(`train_base.py:1285`).

An alternative that avoids renumbering is to keep one stage and phase the perturbation inside
it. That is cheaper but forfeits the main benefit — two separately gated, separately
checkpointed, separately demonstrable capabilities.

## 5. Gate resolution

Both stages should resolve thresholds from a measured baseline rather than a TOML literal,
and the resolver applies unchanged to the rest of the curriculum. The lifecycle:

1. Materialise the fully effective reward/environment/perturbation/backend config.
2. Measure or validate a compatible baseline on a registered seed vector.
3. Resolve once, atomically persist `gate_resolution.json` with full provenance.
4. Put the finite resolved float into an immutable run config.
5. Pass that snapshot to SB3, JAX/MJX, notebook, reporting, visualization, bundles, sweeps.
6. No executable consumer reopens raw TOML after resolution.
7. Resume loads the frozen gate; it never recomputes in place.
8. A changed commit, config, or backend is a new run and recalibrates.

Missing, stale, or incompatible baseline data must **block** advancement rather than silently
falling back to a literal.

**The reward threshold, where one is used, must compare like with like.** An earlier revision of
this plan proposed `reward_mean_standing × 1.055`; that is wrong, because the policy is gated on
*unconditional* mean return while `reward_mean_standing` is conditioned on full-horizon
survival. Conditioning removes the failure mode the policy is supposed to eliminate. Measured
counterexample: over 120 seed-matched episodes the trained policy beat zero action by **+568.02**
with survival **118/120 against 68/120**, while sitting 677–775 points below the
survivor-conditioned statue mean. A standing-floor gate would reject a policy that is
unambiguously better than doing nothing.

Use instead:

```
G_stat = UCB95(mean unconditional zero-action return) + Δ_abs
G_run  = max(configured_literal, G_stat)

D_i         = R_policy(seed_i) - R_zero(seed_i)      # paired, identical seeds
pass_paired = LCB95(mean(D_i)) >= Δ_abs
```

with `Δ_abs = 0` until there is evidence for a nonzero practical effect size. Retain
`reward_mean_standing` as a labelled diagnostic only.

The unconditional mean needs a confidence bound because it is genuinely noisy — across three
disjoint 40-seed blocks it moved 1971.57 / 1968.72 / 1884.71 (spread 86.9) while the standing
mean moved 3244.04 / 3250.45 / 3233.99 (spread 16.5).

### 5.1 Blocking pre-flight

Independent of the resolver, and worth shipping first: make the §3b notebook cell raise instead
of print, and evaluate the **full joint predicate** against zero action rather than the reward
sub-threshold alone. §1.4 shows the current one-sided check reports `FAILS` for all four species
while only velociraptor's complete gate is actually statue-clearable — the verdict is
directionally right and quantitatively misleading.

## 6. Diagnostics

Both stages need instruments that separate *standing* from *not yet fallen*, and *standing* from
*hopping*. Ordered by value:

1. **Every metric reported as a margin over the measured null.** The pre-flight already computes
   the floor; put it on the eval plot as a horizontal line and in `training_summary.txt`.
2. **Ground-reaction-force check against body weight.** Time-averaged total GRF must equal body
   weight for periodic motion — a physics invariant, so deviation is a sensor or accounting bug.
   Log `mean(total_contact_force) / (m·g)` and alarm outside `[0.95, 1.05]`. This would already
   be firing: the statue's static total is 841 N while the policy's logged mean is 1460 N.
3. **Support-state transition matrix** over `{bilateral, single-L, single-R, airborne}`. A walk
   is dominated by `single-L ↔ single-R`, a bounce by `bilateral ↔ airborne`.
4. **Centre-of-pressure position, excursion and velocity** — the actual biomechanical definition
   of balance, and the primary success metric for 1b.
5. **Recovery time after each shove** (1b only) — steps from impulse to CoP re-entering the
   support polygon. The single most legible number this stage can produce.
6. **Vertical oscillation and flight-phase count** — `std(pelvis_z)` and peak-to-peak alongside
   the mean. Mean pelvis height is 0.932 against the statue's 0.926, so the mean hides the
   entire behaviour.
7. **Fix `alternation_ratio`, or stop reporting it.** Verified against the shipped
   `_compute_gait_symmetry`: synchronized bounce **1.000**, true alternating walk **1.000**,
   statue **1.000**, limp 0.684. Root cause at `base_env.py:512-515` — a simultaneous two-foot
   landing appends `"R"` then `"L"`, so every bounce reads as a textbook alternation. Record a
   simultaneous touchdown as one `BOTH` event.

## 7. Prerequisites

### 7.1 Fix the airborne hole in `reward_foot_load_balance`

`|R−L| / (R+L+1e-8)` returns 0 when both feet read zero, making airborne strictly cheaper than
single support (§1.5). Under a disturbance this gets worse, not better: a policy that leaves the
ground cannot reject a shove.

```python
total = right_force + left_force
imbalance = xp.where(total > min_support_force,
                     xp.abs(right_force - left_force) / (total + 1e-8),
                     1.0)
```

Needs the matching JAX-path edit for parity, and `[0, 0]` cases in `test_reward_functions.py`
and `test_trex_mjx_reward_parity.py`; neither covers it today.

### 7.2 Verify the foot touch sensors

`unsupported_duty = 0.209` on a plant holding 0.93 m pelvis height that never falls is more
consistent with sensor under-reporting than with 21% airtime. Cross-check touch-sensor sums
against `mj_contactForce` during a policy rollout, per OQ-6. **This gates §7.1 and §6.3** — both
rest on those readings being true.

### 7.3 Force `perturbation_delta_v = 0.0` in every diagnostic script

`zero_action_baseline`, `joint_excursion_report`, `action_bound_report`,
`actuator_saturation_report` and `observation_ablation_report` all build the env from the TOML
and would otherwise silently measure a shoved plant.

### 7.4 Decouple the collapse detector

`curriculum.py:1014` falls back to `min_avg_reward` for `peak_floor` when `collapse_peak_floor`
is unset. `configs/trex/stage1_balance.toml:164` sets it to `2200.0`; the other eleven stage
configs do not. Set it explicitly per stage, or decouple it, before gate values move.

## 8. Open questions — measure before committing

1. **Does a 1a policy transfer into 1b at all?** If the stance 1a learns is passive enough that
   the first shove destroys it, the split buys nothing over training with the push from the
   start. Test: take `robust_best_model.zip` from `20260729_151044`, enable the push, evaluate.
   This is also §NS-1's own missing load-bearing number (checkpoint at noise 0.10, push on) — a
   ~10-minute eval, not a training run.
2. **Re-measure every push figure at current `main`.** All of §3.4 predates `435f35f`, which
   moved the undisturbed statue +227.84 mean / +409.09 standing with byte-identical
   trajectories.
3. **Does the perturbation penalise the hop?** A policy airborne 21% of the time cannot reject a
   shove mid-flight, so 1b should select against bouncing. *Inferred, not measured.*
4. **Is 3M steps enough for 1a?** See §2.4.
5. **Do the 1a gate thresholds in §2.3 admit the current checkpoint?** They are proposed from
   observed values, not calibrated.

## 9. Risks

| risk | mitigation |
|---|---|
| Catastrophic forgetting at the 1a → 1b boundary | ramp the perturbation (§3.3); the task change is small — same reward, same plant, one force |
| Added wall-clock for a fourth stage | 1a shortened to ~3M (§2.4); partly offsets |
| Renumbering churn across four species and the website | mechanical; can land as a separate prep PR |
| 1b unlearnable from scratch if a species' 1a is weak | ramp, plus per-species `perturbation_delta_v` |
| Brachiosaurus cannot stand at all today — 0% full-horizon, `n_standing = 0` | its 1a gate will fail correctly; that is the `CHECK PLANT` condition surfacing, and it must be fixed before 1b is meaningful for that species |

## 10. Sequencing

1. §7.2 sensor verification — gates the rest.
2. §7.1 `foot_load_balance` fix, with parity tests.
3. §5.1 blocking pre-flight, evaluating the full conjunction.
4. §8.1 push-transfer pilot on the existing checkpoint.
5. §6 diagnostics — at minimum the support-state matrix and CoP.
6. Perturbation mechanism (§3.2) behind a default-`0.0` flag; no behaviour change until enabled.
7. Stage split and renumbering (§4).
8. Gate resolver (§5) as its own tested change.

Steps 1–4 are measurement and small fixes, and none of them require the split. They should
happen regardless.

## Provenance

Measurements attributed as follows. Reproduced directly against `48fd90a` for this document:
the four-species baseline table, the full-conjunction table, the statue per-term decomposition,
the foot-state cost table, the `alternation_ratio` gait test, the settled plant constants, and
the `ec23125` → `48fd90a` counterfactual. Taken from run `20260729_151044` artifacts: the
eval curves and all stance diagnostics. Taken from TREX_REVIEW_2026_07 §NS-1: the refuted
candidate table, the CoP-under-CoM measurement, the push-on figures, and the anti-gaming
searches. Taken from the PR #471 review thread: the 120-seed paired comparison and the
three-block seed stability figures.

Items marked *inferred* are reasoning, not measurement.
