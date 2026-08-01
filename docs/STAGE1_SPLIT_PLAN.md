# Splitting Stage 1 into Stance (1a) and Recovery (1b)

Design proposal to replace the single balance stage with two: **1a — stance**, reaching and
holding a stable pose, and **1b — recovery**, holding it against external disturbance.

Follows from [investigations/TREX_REVIEW_2026_07.md](investigations/TREX_REVIEW_2026_07.md)
§F1 and §NS-1. The task-change rewrite of §NS-1 that originated in PR #471 was ported onto
`main` in `3774301`; that PR is closed and its measurements now live in `TREX_REVIEW_2026_07`.

Revision 3 incorporated two rounds of review feedback on
[PR #474](https://github.com/kuds/mesozoic-labs/pull/474), including its empirical addendum.
**Revision 4 rebases the document onto `34a7002`**, where every blocking prerequisite it
identified has been fixed. §11 records what changed and why. Every claim below carries a label
from the ledger in §11.

**Revision 5 (`ca56f6c`) reports the re-baseline.** It changes three things materially:

* **§7.5 fixed the wrong invariant.** The real reset defect was geometric interpenetration on
  *every* episode, not a ~1% terminal tail. With it fixed the statue baseline rose to 2243.12,
  which puts the trained stage-1 policy's best evaluation (1992.7) **below do-nothing**.
* **No reward threshold can gate 1a.** The statue collects 97.0% of the theoretical maximum
  return with zero actuation cost, so it is the optimum, and a policy's ceiling is strictly
  below it. §2.3's `stance_success` event is the only viable gate.
* **Reset noise is the 1a/1b dial**, measured: 100% statue full-horizon at ≤ 0.05, 65% at the
  configured 0.10, 22% at 0.20, with the *standing* score nearly flat throughout.

§2.3.1 carries the measurements and per-species gate proposals for review.

## Status of the blocking prerequisites — all landed, one superseded

Revision 3 opened by saying these defects "must be fixed before any gate in this document can
mean anything," and §10 called them "not optional preliminaries — they are the work." They
landed in [PR #478](https://github.com/kuds/mesozoic-labs/pull/478):

| § | defect | status on `34a7002` |
|---|---|---|
| 7.5 | ~1% of episodes terminal at generation | **superseded** — right fix, wrong invariant; see §2.3.1 and `ca56f6c` |
| — | every reset interpenetrates the floor (≤ 0.198 m, ≤ 19× body weight) | **fixed** — `ca56f6c` |
| — | two permanent home-pose self-collisions (16.3 kN trex, 25.9 kN brachiosaurus) | **fixed** — `ca56f6c` |
| 5.2 | gate plumbing fails open on both backends | **fixed** — both reject, `17ca4e5` |
| 7.4 | `collapse_peak_floor` inherits `min_avg_reward` | **fixed** — decoupled, `17ca4e5` |
| 7.1 | `[0, 0]` scores airborne as perfectly balanced | **fixed** — reports 1.0, `c667938` |
| 7.2 | foot touch sensors unverified | **done** — audited, `2ab5ae6` / `efe93dd` |

Each section below keeps its original diagnosis, because the reasoning is what justifies the
gate design, and adds a **STATUS** note recording what shipped. Two consequences propagate
outwards and are the substantive content of this revision:

* **§2.3.1's empirical argument is stale.** It rejected `LCB95 ≥ 0.90` at n=40 partly because
  the current checkpoint scored 39/40 on two panels — but the first failure, seed 3077, *was*
  the reset defect. That is fixed, so the panels must be re-measured before the operating point
  is chosen.
* **§7.2 found two defects nobody had filed**, one of them on velociraptor, which changes the
  per-species enablement order in §4 and the risk table in §9.

**What is still outstanding is the split itself** — §2.3's episode-level gate metrics, §3's
perturbation scheduler, §4's stage manifest, and §5's gate resolver. None of those has started.

## TL;DR

Stage 1 currently asks one number to answer two questions: *did the plant reach a stable
stance*, and *is that stance actively controlled rather than passively propped*. The first is
legitimately satisfied by a controller that settles and then stops working. The second is not
measurable at all without a disturbance. On the undisturbed task — where a passive and an
active controller can generate the *same trajectory* — realized on-trajectory return cannot
identify active feedback, which is why the gate has been either unbindable or unclearable in
every configuration tried so far. (This is a claim about the undisturbed task, not a universal
impossibility result about return-based gates.)

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
species              mean     len   gate  len gate   reward?   length?   REWARD+LENGTH
trex               1971.6   638.1   1840       750    CLEARS    blocks   blocked
velociraptor       1704.9   977.5    100       750    CLEARS    CLEARS   STATUE PASSES
brachiosaurus       108.2    98.7    100       750    CLEARS    blocks   blocked
dibothrosuchus     1702.0   674.2    100       750    CLEARS    blocks   blocked
```

The last column is the reward-plus-length conjunction evaluated on **one 40-episode
aggregate**. It is not the complete advancement predicate, which also involves evaluation
batching, `min_eval_episodes` and `required_consecutive`.

Every stage-1 **reward** threshold is cleared by a statue. Velociraptor's **complete** gate is
cleared by a statue. In the other three, `min_avg_episode_length` is doing all of the gating
work and the reward threshold is decorative.

> **STATUS — re-measured on `34a7002`, and the conclusion is unchanged.** `[measured]` The reset
> repair (§7.5) moved three species' reset distributions, so this table was re-taken. T-Rex is
> **1976.62** mean / 23 of 40 full-horizon (was 1971.6 / 23 of 40) and velociraptor is
> **1704.93** mean / **977.5** length / 39 of 40 (unchanged to three decimals, because its bound
> binds at 3.6σ and clips ~0.02% of draws). Velociraptor's statue still clears its complete
> reward-plus-length gate, which is the load-bearing claim of this section. Brachiosaurus and
> dibothrosuchus were not re-taken; brachiosaurus's reset is unchanged, and dibothrosuchus's
> moved only in the far tail.

**Zero-action survival is stable across panels; its reward is not.** `[artifact-derived]` Four
disjoint 40-seed T-Rex panels (3042/4042/5042/6042) gave 23, 23, 22 and 25 full-horizon
episodes — pooled **93/160 = 58.125%**, exact two-sided 95% CP interval **50.08%–65.87%** —
while per-panel reward means ranged **1884.71 to 2157.53**. The physical result is reproducible;
the reward scalar attached to it is not stable enough to gate on. This is the empirical case for
making stance a *state-capability* gate rather than a reward gate.

### 1.5 The observed policy hops rather than stands — confirmed on video  `[measured]`

> **UPGRADED from inferred to measured.** Frame-by-frame analysis of
> `trex_ppo_stage1_best.mp4` / `_final_side.mp4` from run `20260731_132102` confirms the
> behaviour this section inferred from duty ratios, and resolves it into **two** superimposed
> oscillations. Note the renders are 1000 frames at 50 fps for a 1000-step episode at a 100 Hz
> control rate, i.e. **2× slow motion** — frequencies below are real-time.
>
> | | best (4.65M) | final (6.0M) |
> |---|---|---|
> | episode survived | 20.0 s (full) | 8.5 s (fell) |
> | toe lift-downs / sec | ~15 | ~14 |
> | dominant fast frequency | 4.3 Hz | **13.5 Hz** |
> | toe-motion power > 2 Hz (real-time > 4 Hz) | 35% | **71%** |
> | unsupported duty | 0.351 | 0.300 |
>
> 1. **Fast toe chatter, ~15 lift-downs per second** — at 100 Hz control that is ~7 control
>    steps per cycle, i.e. a control-bandwidth limit cycle, not a gait.
> 2. **A slow whole-body crouch↔extend bob at ~0.6–1.0 Hz**, ~30 cm peak-to-peak in the
>    silhouette centroid.
>
> Stance breakdown at the best checkpoint: **bilateral 52% / airborne 35% / single 13%**. Single
> support is the *rarest* state. A walk alternates bilateral↔single; this alternates
> bilateral↔airborne.
>
> The collapse between the two checkpoints is the same chatter with the slow postural control
> stripped out: identical lift-down rate, double the high-frequency share.
>
> **`smoothness_weight` cannot see this.** It penalises action-delta *magnitude*, not frequency.
> From best to final, `action_delta` **fell** 12.0 → 10.5 and the smoothness penalty *improved*
> (−0.286 → −0.250) while high-frequency power doubled. The policy got smoother by the metric
> while getting buzzier in fact. Any fix needs a frequency-aware or contact-switch-rate cost —
> see §7.1's successor.
>
> Caveat carried forward: a large share of the airborne duty was **learned from the broken
> reset** (§2.3.1) — if episodes routinely opened with a catapult and ~60 ballistic steps,
> airborne-tolerant behaviour is what training rewarded. How much of the chatter survives on the
> repaired plant is unmeasured, and needs a fresh run.

Original inference, retained:

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

**Status of this claim.** The `[0, 0]` arithmetic hole in `reward_foot_load_balance` was
`[measured]` — verified directly against the shipped weights.

> **STATUS — the sensor escape route is closed; the cost table is repaired.** `[measured]`
>
> The hole is fixed (§7.1): an unsupported pair now reports maximal imbalance, so the ordering
> above becomes `both feet down +0.600 > single support = airborne −0.300`. Airborne is no
> longer the second-best state on a balance stage.
>
> The sensor caveat is discharged (§7.2), in the direction this section did not expect. T-Rex's
> touch sensors agree with `mj_contactForce` to **1.000** on a settled plant, and the duty
> classifier tracks kinematic ground truth — `mj_geomDistance` from every foot geom to the floor
> — to within **0.52%** of steps across a swept 3%–67% airborne range. The nearest regime to the
> disputed reading, low-amplitude jitter, measured **16.07% true airtime against 16.21%
> reported**. The classifier's threshold is 0.1 N against ~421 N per foot in quiet stance, far
> too low for partial unloading to manufacture a false `unsupported`.
>
> So `unsupported_duty = 0.209` can be taken at face value: **the plant really is off the ground
> about 21% of the time.** This claim moves from `[inferred]` to `[measured instrument, inferred
> behaviour]`.
>
> **What remains `[inferred]` is causation** — that the *reward* produced the hop. No sensor work
> can establish that; it needs the counterfactual run with the §7.1 repair in place. That run is
> now possible and is the cheapest open question in §8.
>
> Full method and results:
> [investigations/FOOT_SENSOR_VERIFICATION.md](investigations/FOOT_SENSOR_VERIFICATION.md).

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

> **STATUS — the setting stands, but what it *means* changed slightly.** `[measured]` The §7.5
> repair bounds the root-height draw, so at `reset_noise_scale = 0.10` T-Rex's spawn-height
> standard deviation is now **0.0980** rather than 0.1007, with the mean unchanged at 0.9261 m.
> The recommendation is unaffected — the statue still reaches the full horizon in 57.5% of
> episodes at 0.10, which is the number §2.3's 90% requirement is set against — but the reset
> distribution is no longer exactly the one the pre-`34a7002` figures were taken under, and the
> key is now a joint-angle scale that no longer doubles as an unbounded length.
>
> Note also that `perturbation_delta_v` is the name revision 2 used; §3.2 replaced it with a
> dimensionless `perturbation_capture_velocity_multiple`. The 1a config should set whichever
> name the scheduler ships with, explicitly, to `0.0`.

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

> **DECIDED 2026-08-01 — the statistic is a HYBRID, reconciling this section with §12 of
> PLANT_VALIDATION.** The two documents specified different rules: this section's binomial
> `LCB95(P(stance_success)) ≥ 0.90`, and §12's raw fractions (full-horizon ≥ 95%, unsupported
> duty ≤ 0.02). They are about five points of certified capability apart — verified by exact
> binomial calculation, a raw 38/40 certifies only `P ≥ 0.851`, while `LCB95 ≥ 0.90` at n=40
> requires **40/40** and passes a genuinely-95% policy just 12.9% of the time. `[measured]`
>
> The cliff is not an argument against confidence bounds; it is an argument against small
> panels *for a binarised metric*. Adopted rule, in three parts:
>
> 1. **Screening, every evaluation at n=40** — raw fractions, with `required_consecutive` as
>    scheduler hysteresis ONLY. Re-running a deterministic panel is not statistical
>    replication (§3.4 caution 3).
> 2. **The load-bearing bound — one-sided LCB on MEAN UNSUPPORTED DUTY at n=40.** Duty is
>    continuous with tiny measured variance (the statue sits at 0.000 on all four species), so
>    an interval on it has real power at 40 episodes where the pass/fail count has almost none.
>    This is what actually certifies stance quality.
> 3. **Confirmation — one predeclared held-out panel at n≈100–180** for the binary
>    full-horizon event, run once a candidate qualifies (96/100 certifies `P ≥ 0.911`; 168/179
>    certifies `P ≥ 0.900`). Evaluation is minutes and training is days, so this is the cheapest
>    rigour available.
>
> **Artifact requirement, verified against the current evaluator:** the per-episode outcomes
> this needs are already recoverable. `EvalResults` carries per-episode `rewards`, `lengths`
> and `successes`, and the per-step `diag_*` arrays are appended inside the sequential
> per-episode loop, so episode boundaries reconstruct exactly from `cumsum(lengths)` — for
> BIPEDS. For quadrupeds they do not; see the `diag_r_foot`/`diag_l_foot` interleaving defect
> recorded in KNOWN_ISSUES, which must be fixed before this rule can be applied to
> brachiosaurus or dibothrosuchus.

**This is new machinery, not a config change.** `StageThreshold`
(`environments/shared/curriculum/manager.py:23-30`) currently supports exactly `min_avg_reward`,
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
`h_max` 0.03 m, `d_max` 0.5 m. `[inferred]`

`min_avg_reward` is **unset** for 1a — see §5.2, which makes that safe rather than fail-open.

#### 2.3.1 The statistical operating point must be declared  `[measured]`

> **RE-BASELINED on `ca56f6c`.** Step 6 of §10 is done for the baseline half. Everything from
> here to the end of this subsection is measured on the repaired plant; the pre-`ca56f6c`
> numbers that used to head this section are retained below only as the *before* column.

##### The reset defect was misdiagnosed, and §7.5 fixed the wrong invariant  `[measured]`

§7.5 bounded the root-height draw against `healthy_z_range` and reported 43/4000 → 0/4000
already-terminal spawns. That measurement was correct and it was beside the point.
`healthy_z_range` is a **termination predicate on the root**; it says nothing about
foot-to-floor geometry. A T-Rex pelvis at 0.739 m is "healthy" while the toes are 0.198 m
underground.

Measured over 40 seeds on the pre-fix plant, spawns ranged from **0.198 m inside the floor** to
**0.18 m above it**, and the solver answered penetration with up to **19× body weight**:

```
seed 2:  t= 0  pelvis 0.746  feet 5777.6 / 6033.7 N   <- spawned inside the floor
         t=30  pelvis 1.255  feet    0.0 /    0.0 N   <- ejected 0.52 m, ballistic
         t=60  pelvis 0.879  forward_z -0.440         <- lands nose-first
         t=66         forward_z -0.530                <- nosedive termination
```

Penetration depth predicted failure: failed resets averaged −0.088 m, survivors −0.034 m.
Two independent controller sweeps confirmed the diagnosis by failing — no PD gain on trunk
pitch (70 combinations, both signs) and no static posture offset (75 combinations) beat zero
action, because for the first ~60 steps there is no ground contact to act against.

Separately, two MJCF geom pairs overlapped in the home pose and injected a constant,
**pose-independent** force into every step of every episode in every run to date:
`trex tail_1_geom` against both thighs (18.8 mm, 16.3 kN = 11.0× weight) and
`brachiosaurus torso_main` against `tail_2_geom` (66.9 mm, 25.9 kN = 12.7× weight).

Both are fixed in `ca56f6c`; `environments/shared/tests/test_reset_plant_invariants.py` asserts
the invariants for all four species. T-Rex t=0 contact force went 16990 N → 190 N.

##### Re-measured statue baseline  `[measured]`

Same protocol as before — `zero_action_baseline.py`, seed 3042, 40 episodes, noise 0.10:

| | before (broken plant) | after (`ca56f6c`) |
|---|---|---|
| reward mean | 1974.74 | **2243.12** |
| reward mean − std | 492.94 | **867.21** |
| reward standing | 3243.10 ± 23.65 | 3250.27 ± 24.49 |
| episodes reaching horizon | 23/40 (57.5%) | **26/40 (65%)** |
| episode length | 640.85 | **716.5** |
| terminations | fallen 3, nosedive 14, trunc 23 | fallen 2, nosedive 12, trunc 26 |

**The trained stage-1 policy is now worse than doing nothing.** Run `20260731_132102` peaked at
**1992.7** mean eval reward; the repaired statue scores **2243.12**. That run cleared
`min_avg_reward = 1840` sixteen times, including an eight-evaluation streak, while sitting 250
points below the do-nothing floor. `[artifact-derived]`

Nosedive remains 12 of 14 failures, so the catapult was not the only source — a residual pitch
instability in the home pose survives at noise 0.10. That is a real Stage 1a target; it was
previously masked.

##### The operating point: reset noise is the 1a/1b dial  `[measured]`

`zero_action_baseline.py trex --sweep-noise`, 40 episodes each:

| reset noise | reward | mean−std | standing | ep length | full-horizon |
|---|---|---|---|---|---|
| 0.01 | 3287.9 | 3286.3 | 3287.9 | 1000.0 | **100%** |
| 0.05 | 3271.8 | 3259.7 | 3271.8 | 1000.0 | **100%** |
| **0.10** ← stage-1 default | 2243.1 | 867.2 | 3250.3 | 716.5 | 65% |
| 0.15 | 1348.2 | −96.9 | 3209.9 | 465.7 | 38% |
| 0.20 | 845.4 | −411.7 | 3166.9 | 324.4 | 22% |

`standing` is nearly flat across the whole sweep (3288 → 3167) while `full-horizon` collapses
100% → 22%. Reset noise does not make standing *harder*; it decides how often you get to stand
at all. Stage 1 currently sits exactly on the knee, which is why survival and stance quality
have been inseparable in every measurement to date.

This is the split, quantified. **1a belongs at noise ≤ 0.05**, where a statue is at 100%
full-horizon and survival is not the binding constraint, so the gate can be about stance
quality. **1b supplies robustness through `xfrc_applied` perturbations** (§3.2) at a declared
magnitude, rather than through reset noise where a lucky draw and a good controller are
indistinguishable.

##### No reward threshold can separate a statue from a policy  `[measured]`

§1.3 asserted the stance reward "saturates against a statue". It is stronger than that: on
stage 1 as configured, **the statue is the global optimum**.

Summing the positive T-Rex stage-1 weights — `alive_bonus` 1.00, `height` 0.60,
`bilateral_support` 0.60, `leg_home_pose` 0.50, `head_clearance` 0.35, `neck_posture` 0.20,
`heading` 0.10 — gives a theoretical maximum of **3.35/step = 3350** over the horizon. The
statue collects **3250.27 = 97.0%** of it, with `energy` and `smoothness` at exactly zero. Any
active policy pays both; run `20260731_132102` paid 0.30/step on smoothness alone, which alone
is a 300-point handicap.

So a policy's realistic ceiling is *below* the statue's score, and there is no threshold that
admits a competent policy and excludes a passive one. This is not a calibration problem to be
solved with a better number — it retires reward thresholds for 1a entirely, and it is why §2.3
gates on the episode-level `stance_success` event instead.

##### Proposed per-species 1a operating point — FOR REVIEW, not adopted  `[measured baseline, inferred thresholds]`

Statue stance quality at the proposed 1a noise of 0.05, 40 episodes from seed 3042:

| species | full-horizon | all-feet duty | unsupported duty | contact switches | standing reward |
|---|---|---|---|---|---|
| trex | 100% | 0.998 | 0.000 | 0.09 /s | 3271.8 ± 12.0 |
| velociraptor | 100% | 1.000 | 0.000 | 0.00 /s | 1745.8 ± 5.0 |
| dibothrosuchus | 100% | 0.997 | 0.000 | 0.14 /s | 2598.3 ± 0.9 |
| brachiosaurus | **0%** | 0.000 | 0.885 | 0.00 /s | never reaches horizon |

These are the numbers a 1a policy must **match**, not beat — the statue defines the quality
ceiling, and 1a's job is to certify a policy has not bought stability with actuation. Proposed
per-species thresholds:

| species | noise | full-horizon | unsupported duty | contact switches | reward floor |
|---|---|---|---|---|---|
| trex | 0.05 | ≥ 95% | ≤ 0.02 | ≤ 1.0 /s | ≥ 2900 (0.89 × statue) |
| velociraptor | 0.05 | ≥ 95% | ≤ 0.02 | ≤ 1.0 /s | ≥ 1550 (0.89 × statue) |
| dibothrosuchus | 0.05 | ≥ 95% | ≤ 0.02 | ≤ 1.0 /s | ≥ 2300 (0.89 × statue) |
| brachiosaurus | — | **blocked** — see below | | | |

Three notes on these, all of which need a decision rather than adoption:

* The reward floor is a **sanity rail, not the gate**. It is set *below* the statue on purpose,
  because above it is unreachable; its only job is to reject a policy that has thrown away most
  of the available return. The `stance_success` event of §2.3 remains the actual gate. The
  0.89 multiplier is a round number chosen to sit clear of the ~0.30/step smoothness cost a
  reasonable active policy pays; it is **not** measured and should be revisited once any policy
  clears 1a.
* The unsupported-duty and switch-rate ceilings are set well above the statue's measured 0.000
  and 0.00–0.14 /s so that normal weight-shifting is not penalised. Note the switch metric
  conflates bilateral↔single with bilateral↔airborne; the repaired plant moved T-Rex's raw
  switch count *up* (0.86 → 1.00 /s) while unsupported duty went to zero, because the extra
  switches are weight-shifting. **Gate on unsupported duty; treat switch rate as diagnostic**
  until it is decomposed.
* Every current stage-1 `min_avg_reward` is cleared by a statue: trex 1840 vs 2243, and 100 vs
  1746 / 163 / 1834 for the other three. Whatever is decided for 1a, those four values are
  wrong today.

**Brachiosaurus stage 1 is not currently a balance task.** The statue scores 0/40 full-horizon,
mean length 130.7, terminations `fallen 34, tail_contact 6` — it falls every single time. There
is no "do not fall" floor to beat, so no brachiosaurus stage-1 result is interpretable. Verified
identical before and after `ca56f6c`, so this is pre-existing and not a regression. It needs its
own stance fix before any gate is set for it, and it is a new blocking item — see §10 step 7.

**Velociraptor is the best species to prototype the 1a gate on**: 40/40 at 1745.84 ± **5.03**.
That variance is small enough that a paired test against the statue has real power at n=40,
which is not true for T-Rex (± 24.49 on standing, ± 1375.90 overall).

##### Power table — arithmetic, unchanged

Interval method: **exact one-sided 95% Clopper-Pearson**. With that fixed, `LCB95 ≥ 0.90`
implies these cutoffs and these chances of a *good* policy passing:

```
    n   cutoff   P(pass | true p=0.95)   P(pass | true p=0.98)
   30    30/30                 21.464%                 54.548%
   40    40/40                 12.851%                 44.570%
   80    77/80                 42.845%                 92.315%
  100    96/100                43.598%                 94.917%
```

At n=30 and n=40 the rule permits **no failures at all**. Revision 3 argued this was "not a
theoretical concern" on the grounds that the current checkpoint scored 39/40, 39/40, 40/40,
40/40 across four 40-seed panels — pooled **158/160 = 98.75%** — and so failed two of them.
`[artifact-derived]`

> **STATUS — that evidence is stale, and the argument it supported may not survive.**
> `[measured]`
>
> §7.5 identified the first failure, **seed 3077, as the reset defect**: the pelvis spawned
> below the height floor and the episode ended on step 1 whatever the policy did. That defect is
> fixed. Re-checked on `34a7002`, seed 3077 now spawns at **0.72000 m** against the 0.70 m floor
> and survives a zero-action step.
>
> So the two 39/40 panels were an artifact of an environment bug, not a property of the policy.
> With it removed the checkpoint may well be 40/40 on all four, in which case `LCB95 ≥ 0.90` at
> n=40 is *not* the impractical rule this section makes it out to be, and the case for n=179
> weakens considerably.
>
> **AMENDED on `ca56f6c`.** The above reasoning was right in shape and wrong in magnitude. Seed
> 3077 was not one instance of a ~1% tail; it was one instance of a defect affecting **every
> reset**, which the height bound did not address because it checked the root against a
> termination range rather than against floor geometry. See the re-baseline at the top of this
> subsection. The panels still need re-measuring and the conclusion about n is still open, but
> the premise is now "the plant was wrong for every episode", not "1% of episodes were
> unwinnable".
>
> **The power table above is arithmetic and stands.** What is stale is the empirical premise
> that a good policy fails the cutoff. Nothing here should be read as settled until the panels
> are re-measured on the repaired reset.
>
> Two caveats on that re-measurement. The available checkpoint is at
> `policy_interface_revision` 7 against the current 8, and was trained under the old reset
> distribution, so it is unvalidated for this task — it can be evaluated as *evidence about the
> defect*, but a properly re-baselined policy is what should set the operating point. And the
> ~1% floor of unwinnable episodes is gone, which is precisely what makes a no-failures-permitted
> cutoff reachable in principle for the first time.

Pick one operating point and write it down — **after** the re-measurement above, not before:

* **capability target p ≈ 0.95, ~80% power** → `n = 179`, cutoff `168` (or `n = 180`, cutoff
  `169`, power 80.8%);
* **capability target p ≈ 0.98** → `n = 100`, cutoff `96`, power 94.9%.

**Do not multiply confidence by `required_consecutive`.** Re-running the *same* deterministic
panel three times adds no statistical evidence, and demanding three independent panels each
clear a low-power cutoff has worse power than one properly sized panel. Use cheap evaluations
plus `required_consecutive` as scheduler hysteresis only; once a candidate qualifies, freeze it
and run **one predeclared held-out confirmation panel** at the declared `n`.

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
perturbation_capture_velocity_multiple = 1.5   # dimensionless; see note below
perturbation_interval     = 2.0     # seconds between shoves
perturbation_jitter       = 0.5     # +/- seconds, defeats a blind clock-timed brace
perturbation_duration     = 0.20    # seconds of applied force
perturbation_direction    = "uniform_horizontal"
```

On the T-Rex plant `1.5×` capture-point velocity is roughly **150 N for 0.20 s**.
`[artifact-derived]`

Revision 2 named this key `perturbation_delta_v = 1.5`, which is dimensionally ambiguous — the
name says a velocity, the value is a multiplier. Use either the dimensionless multiple above or
an explicit `perturbation_delta_v_mps`, never a bare number whose unit depends on prose. The
**derived force and impulse must be persisted per species**, since the same multiple produces
different absolute forces on different plants.

**Checkpoint compatibility — narrower than revision 1 claimed.** `plant_contract/policy_layer.py:292-374`
fingerprints observation/action implementations and selected reset semantics; `step` appears
zero times in the interface payload, so a `step()` hook moves no `policy_interface_revision`.
`[measured]` Revision 1 concluded from this that the change "invalidates no existing
checkpoints." That is too broad. A pushed task changes the transition kernel and the evaluation
distribution: existing checkpoints stay *mechanically loadable and interface-compatible* while
being *unvalidated for the new task*. The fingerprint's silence about `step` is a provenance
gap, not evidence of task equivalence.

Add a distinct **task/evaluation fingerprint** covering perturbation implementation, schedule,
RNG protocol, force parameters, reset configuration, horizon, reward and termination semantics,
and backend.

**Two distinct load modes.** Revision 2 said "resume must not cross that boundary silently,"
which contradicts the fact that 1b is *meant* to start from a 1a checkpoint across exactly that
boundary. Separate them:

| mode | task fingerprint | requires | notes |
|---|---|---|---|
| `resume_same_stage` | must match exactly | resolved gate, scheduler/ramp state, optimizer + normalization compatibility | continuation of one run |
| `initialize_next_stage` | mismatch **expected**, recorded as lineage | policy-interface compatibility only | explicit optimizer / normalization / ramp reset behaviour |

**Narrow the reproducibility promise.** Exact mid-stage reproducibility needs more than current
checkpoints preserve — PRNG state, environment and scheduler state, registered schedule
position, global transition count, and ramp progress. Until those are persisted, promise
*reproducible stage-boundary restart*, not exact mid-stage resumption.

**Scheduler requirements** — a shared force kernel does not give backend-neutral scheduling.
The design must specify: explicit clearing of `xfrc_applied` after each pulse and on reset;
deterministic episode-local push times and directions; pre-generated schedules so baseline and
policy receive *identical* disturbances; MJX `data.replace(xfrc_applied=...)` and auto-reset
clearing; CPU-JAX evaluation parity; one schedule unit shared by SB3 and thousands of parallel
MJX environments; and persisted/restored ramp progress on resume. Calibration and advancement
always run at frozen full strength, never partway through a ramp.

### 3.3 Ramp versus fixed — an open question, not a decision

§NS-1 correction 2 recommends a fixed impulse, attributing the problem to `set_reward_weight`
(`base_env.py:759`) being a bare `setattr`. Revision 1 of this document repeated that.
**Both are wrong about the mechanism.** `RewardRampCallback`
(`environments/shared/curriculum/advancement.py:477-560`) already computes linearly interpolated values from
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
   but the correct statement is that survival is bounded **above by** ~7.2%, not that it is
   zero. (Revision 2 said "bounded below," which inverts the direction.)
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

**Null suite, and the multiplicity rule.** Zero action alone is insufficient — survival does not
prove feedback control. Calibrate against zero action, constant/brace controllers, *and* the
incoming 1a checkpoint.

The paired formula above names only zero action while the prose names three nulls; that gap has
to close one of two ways, declared in advance:

* **simultaneous** — require the paired lower bound against *every* predeclared null, with a
  multiplicity correction across the suite; or
* **select-then-confirm** — identify the strongest null on calibration seeds, then confirm once
  against it on held-out seeds.

**Pair identity is part of the estimand.** Two panels with identical marginal totals
(policy 30/40, baseline 20/40) can yield materially different paired bounds depending on *which*
seeds succeeded. `[artifact-derived]` Every canonical gate record must therefore carry, per
episode: controller ID, pair ID, episode seed, success outcome, return, and realized push
schedule. An aggregate CSV of marginal means cannot reproduce a paired decision and is not
acceptable evidence.

### 3.5 Budget

Provisional 3M steps, warm-started from the 1a checkpoint. `[inferred]`

## 4. Stage identity — semantic IDs, not renumbering

Revision 1 proposed renumbering 2→3 and 3→4 and called it "mechanical." It is a schema
migration. Verified blockers `[measured]`:

* `environments/shared/config.py:154` — `_STAGE_FILE_PREFIX = {1: "stage1_", 2: "stage2_", 3: "stage3_"}`; stage 4 raises `KeyError`.
* `environments/shared/reporting/bundles.py:90` — result bundles reject any stage set not a subset of `{1, 2, 3}`.
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

**Make this an executable manifest, not a naming convention.** A versioned, ordered per-species
manifest is what lets T-Rex carry a `recovery` stage while the other three do not, without
reinterpreting any historical artifact:

```
stage_manifest/v1  (per species, ordered)
  - id: stance      config: configs/trex/stance.toml      terminal: false   legacy_alias: 1
  - id: recovery    config: configs/trex/recovery.toml    terminal: false   legacy_alias: null
  - id: locomotion  config: configs/trex/locomotion.toml  terminal: false   legacy_alias: 2
  - id: behavior    config: configs/trex/bite.toml        terminal: true    legacy_alias: 3
```

Keep two fingerprints separate, because they answer different questions:

* **task identity** — plant and policy-interface identity, model and implementation hashes, full
  effective environment/reward/termination/perturbation config, backend and precision, relevant
  dependency versions.
* **evaluation protocol** — null-controller definitions, ordered episode seeds, pair IDs,
  episode count, confidence procedure, and both the *intended* and *realized* push schedule.

Recovery evidence emits one row per shove: push ID, actual start and end step, force vector and
impulse, schedule hash, recovery-entry step, and dwell result.

## 5. Gate resolution

**Capability requirements are normative; baselines are evidence.** Revision 2 said "both stages
resolve their thresholds from a measured baseline," which wrongly implies that safe height,
tilt, speed, drift, settling time, required recovery probability and maximum recovery time
should track whatever the null controller happens to do. They should not — those are task
requirements. A baseline exists to support the blocking preflight (§5.2) and relative-superiority
comparisons, nothing more.

Freeze three separate artifacts per run:

| artifact | kind | contents |
|---|---|---|
| `capability_spec` | **normative**, versioned | `h_max`, `angle_max`, `v_max`, `omega_max`, `d_max`, `T_settle`, `p_recovery`, `T_recover`, dwell |
| `null_manifest` | **evidential**, measured | null-controller definitions and their measured outcomes on a compatible task |
| `decision_procedure` | **predeclared** | interval method, `n`, cutoff, calibration and held-out panels, multiplicity rule |

Only the relative-superiority margins are derived from the baseline. The lifecycle:

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

### 5.2 The gate schema must fail closed  `[measured]`

**This was a blocking prerequisite, and the plumbing demonstrably failed open.** Review
constructed a composite-only gate:

```toml
[curriculum]
gate_schema_version = 1
gate_kind = "stance_success_lcb"
min_stance_success_lcb = 0.90
```

The loader preserved the unknown fields but `thresholds_from_configs` silently discarded them.
SB3 then materialised legacy permissive defaults (`min_avg_reward = -inf`, length and success
floors `0`) and **advanced** — returning `False, False, True` across three ten-episode
evaluations whose reward was deliberately `-1e12` and whose episode length was `1`. The legacy
JAX check returned `True`; the active JAX evaluation check returned `(True, [])`. Existing
focused tests passed, because they currently codify permissive missing-threshold behaviour.
`[artifact-derived]`

Independently confirmed here: `jax_curriculum.py:44-50` logs a warning and `return True` when
`min_avg_reward` is absent, and `StageThreshold` (`curriculum/manager.py:23-30`) defaults every omitted
threshold to a permissive value. `[measured]`

So removing `min_avg_reward` for 1a — as §2.3 proposes — would have converted the gate into a
no-op on both backends unless the schema landed first. Requirements, all now met:

* versioned `gate_schema_version` and `gate_kind` on every stage config;
* **unknown gate kinds and unknown fields are fatal** whenever advancement is enabled;
* absence of a gate is acceptable **only** in an explicit, recorded non-advancing
  diagnostic/pilot mode;
* config, SB3 consumer, JAX consumer and parity tests land **atomically** — the existing tests
  must be updated in the same change, since today they assert the permissive behaviour.

> **STATUS — landed in `17ca4e5`.** `[measured]` `environments/shared/curriculum/gate_schema.py`
> implements every requirement above. All twelve stage configs declare
> `gate_schema_version = 1` and `gate_kind = "reward_and_length/v1"`; unknown keys, unknown
> kinds and unsupported versions are fatal when advancement is enabled, as is a threshold field
> the declared kind does not consume. `gate_kind = "none/v1"` is the recorded non-advancing
> mode and refuses to advance rather than passing by default.
>
> The composite-only experiment from the addendum was re-run against the shipped schema: both
> backends now raise `GateSchemaError` where they previously produced permissive thresholds and
> `True`. Effective thresholds for all four species are unchanged. The tests that asserted the
> permissive behaviour were rewritten in the same commit, and the resolution logic was later
> extracted (`e3958f9`) so it is pinned in the SB3-free CI job too — the job with the least
> installed is where a fail-open regression would otherwise hide.
>
> A third fail-open path this section did not list also closed: a **misspelled** threshold key
> used to disable the threshold it meant to set, silently. It is now fatal.
>
> **§2.3 can now safely unset `min_avg_reward` for 1a.** That was the dependency this section
> existed to remove.

### 5.3 Blocking pre-flight

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
   Log `mean(total_contact_force) / (m·g)` and alarm outside `[0.95, 1.05]`.

   > **CORRECTION — this item read the invariant backwards.** `[measured]` Revision 3 wrote
   > "this would already be firing: the statue's static total is 841 N while the policy's logged
   > mean is 1460 N," implying 841 N was the anomaly. **841 N is exactly right.** The T-Rex
   > *animal* masses 85.72 kg, so its weight is 840.9 N and the measured static total of 842.2 N
   > is a ratio of **1.002** — the invariant holds essentially perfectly.
   >
   > The 1483 N that made 841 N look low is `mj_getTotalmass`, which includes the **65.45 kg
   > prey body** — 43% of the 151.17 kg model total. **Any implementation of this diagnostic must
   > divide by the mass of the animal's kinematic subtree**, or it will report a false 0.57 on a
   > plant standing in perfect equilibrium and fire on every species that carries prey or food.
   > `environments/shared/scripts/foot_sensor_report.py` does this by summing only the subtrees
   > containing actuated joints; reuse that rather than `mj_getTotalmass`.
   >
   > The diagnostic would still fire — on the **policy's** 1460 N, at 1.74× body weight. That is
   > consistent with intermittent ground contact rather than a steady stance, and so with §1.5,
   > but the figure is `[artifact-derived]` and must be re-measured with the corrected
   > denominator before it is relied on.
3. **Support-state transition matrix** over `{bilateral, single-L, single-R, airborne}`. A walk
   is dominated by `single-L ↔ single-R`, a bounce by `bilateral ↔ airborne`. The underlying
   duty classification is now validated against kinematic ground truth across a 3%–67% airborne
   range (§7.2), so this matrix can be built on it for T-Rex — but **not** for velociraptor or
   brachiosaurus, whose foot sensors report 55% and 0% of true load respectively.
4. **Centre-of-pressure position, excursion and velocity** — the biomechanical definition of
   balance, and the primary success metric for 1b.
5. **Recovery time after each shove** (1b only) — steps from impulse to CoP re-entering the
   support polygon, plus dwell. The single most legible number this stage can produce.
6. **Vertical oscillation and flight-phase count** — `std(pelvis_z)` and peak-to-peak alongside
   the mean. Mean pelvis height is 0.932 against the statue's 0.926, so the mean hides the
   entire behaviour. `[measured]`
7. **Fix `alternation_ratio`, or stop reporting it.** Verified against the shipped
   `_compute_gait_symmetry`: synchronized bounce **1.000**, true alternating walk **1.000**,
   statue **1.000**, limp 0.684. `[measured]` Root cause at `base_env.py:520-523` — a
   simultaneous two-foot landing appends `"R"` then `"L"`, so every bounce reads as a textbook
   alternation. Record a simultaneous touchdown as one `BOTH` event.
   **Still open on `34a7002`** — re-verified, a synchronized bounce still scores 1.000. The
   citation is now `base_env.py:520-523`.

## 7. Prerequisites — §7.1, §7.2, §7.4 and §7.5 landed

Those four shipped in PR #478 (`34a7002`). Each keeps its original diagnosis, because that
reasoning is what justifies the gate design downstream, and carries a **STATUS** note recording
what actually shipped and where it differed from the proposal.

**§7.3 has not landed and cannot yet**: it specifies `plant_sanity` / `task_gate` modes for
diagnostic tooling, and there is no perturbation to switch off or match until the scheduler
exists (§10.11). Its one immediately actionable correction — that
`actuator_saturation_report` is unaffected because it loads raw XML rather than building an env
from TOML — was applied to `TREX_REVIEW_2026_07` in `3774301`.

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

> **STATUS — landed in `c667938`.** `[measured]` An unsupported pair now reports maximal
> imbalance, so the §1.5 cost table becomes `both feet down +0.600 > single support = airborne
> −0.300`. The shipped form takes an optional `min_support_force`, defaulting to `0.0`, which
> closes only the exact `[0, 0]` case and leaves **every loaded state numerically unchanged**;
> raise it during gate calibration to also deny credit for a token grazing contact, against
> measured flight phases rather than by eye. Wired identically through the Gymnasium, MJX and
> JAX paths, with `[0, 0]` coverage added to both test files — neither covered it, which is how
> the hole survived, and `test_trex_rewards.py` actively asserted the old `0.0`.
>
> One residual: the two failure states are now **tied** rather than airborne being strictly
> worst. Separating them needs a term this repair does not add, and is worth revisiting once
> §1.5's causal question is settled.

### 7.2 Verify the foot touch sensors — gates §7.1, §6.3 and §1.5

`unsupported_duty = 0.209` on a plant holding 0.93 m pelvis height that never falls is more
consistent with sensor under-reporting than with 21% airtime. Cross-check touch-sensor sums
against `mj_contactForce` *and* kinematic flight phases during a policy rollout, per OQ-6. The
formula defect in §7.1 is real regardless; the behavioural story in §1.5 depends on this.

> **STATUS — done in `2ab5ae6` and `efe93dd`, and it found more than it was looking for.**
> `[measured]` Full method in
> [investigations/FOOT_SENSOR_VERIFICATION.md](investigations/FOOT_SENSOR_VERIFICATION.md);
> reproduce with `foot_sensor_report.py` and `stance_duty_validation.py`.
>
> **T-Rex is clean, so the hypothesis this section raised is refuted.** Its sensors agree with
> `mj_contactForce` to 1.000 statically, and the duty classifier tracks `mj_geomDistance`
> ground truth to within 0.52% of steps across a swept 3%–67% airborne range. §1.5's reading
> stands.
>
> **Two other species do not pass**, and neither was previously filed:
>
> | species | sensor / measured contact | contact / weight |
> |---|---|---|
> | trex | 1.000 | 1.002 |
> | dibothrosuchus | 1.000 | 1.000 |
> | velociraptor | **0.553** | 1.000 |
> | brachiosaurus | **0.000** | 0.988 |
>
> `contact / weight` is 1.00 everywhere, so the plants are in equilibrium and these are
> sensor-scope defects, not physics ones. Velociraptor's site sits on `toe_d3` alone and misses
> the `metatarsus` (17.54 N) and `toe_d4` (12.03 N) per foot; `aa3395c` fixed that site's *size*
> but not its *body scope*. Brachiosaurus sees none of its 1699.2 N.
>
> Neither reaches a stage-1 reward term today, but both reach the **observation** —
> brachiosaurus trains with four permanently zero input channels at indices 75–78 of 83.
> **This changes §4 and §9**: a stance or recovery gate that reads contact state cannot be
> trusted on either species until repaired, so T-Rex-first enablement is now forced by
> instrumentation rather than merely preferred. The repairs are per-species MJCF changes that
> move physics and policy fingerprints, and are filed rather than made.

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

`curriculum/early_stopping.py:163` falls back to `min_avg_reward` for `peak_floor` when `collapse_peak_floor`
is unset. `configs/trex/stage1_balance.toml:164` sets it to `2200.0`; the other eleven stage
configs do not. Revision 2 said that removing `min_avg_reward` makes the fallback "undefined."
It does not — `curriculum/early_stopping.py:163` chains `collapse_peak_floor` → `min_avg_reward` → **`0.0`**.
`[measured]` The real failure mode is that a `0.0` floor arms collapse detection after *any*
positive robust peak, which is more eager than intended and silently so. Set
`collapse_peak_floor` explicitly per stage, or decouple it, **before** removing the reward
gate. Single-stage pilots still install plateau/collapse callbacks, so "advancement
disabled" does not make an inherited threshold inert.

> **STATUS — landed in `17ca4e5`.** `[measured]` The fallback to `min_avg_reward` is removed
> outright: a missing floor now means **never arm**, because a backstop that is not configured
> should not abort a run. The eleven configs that relied on the fallback set the value it
> produced, so every effective floor is bit-identical to before — `100.0` everywhere except
> T-Rex stage 1's `2200.0`, verified config-by-config against `origin/main` both at the time and
> again after the resolver was extracted in `e3958f9`. `collapse_smoothing_window` was readable
> but undeclared and is now part of the gate schema.

### 7.5 Reset-validity preflight — LANDED, then SUPERSEDED by `ca56f6c`  `[measured]`

> **This section fixed the wrong invariant.** The height bound below is real and still in place,
> but `healthy_z_range` is a termination predicate on the *root* and constrains nothing about
> foot-to-floor geometry. The actual defect was geometric interpenetration on **every** reset
> (up to 0.198 m, answered by up to 19× body weight), plus two permanent home-pose
> self-collisions. Both are fixed in `ca56f6c` and asserted by
> `environments/shared/tests/test_reset_plant_invariants.py`. Read §2.3.1 for the measurement;
> the "~1%" framing below understates the problem by two orders of magnitude.

**About 1% of episodes are unwinnable at generation.** Reset randomisation
(`reset_noise_scale = 0.10`) can place the pelvis below the height termination floor, so the
episode ends before the policy acts. Verified directly:

```
seed 3077: pelvis_height 0.66230 after one zero-action step, terminated=True
scan of seeds 3042-5041: 16/2000 = 0.800% already-terminal
```

A wider independent scan of seeds 3042–7041 found **43/4000 = 1.075%** (exact two-sided 95%
interval 0.779%–1.445%), all for the same height-floor reason. `[artifact-derived]`

This interacts badly with everything in §2.3.1: a ~1% floor of unwinnable episodes makes any
"no failures permitted" cutoff unreachable for reasons that have nothing to do with the policy.
It is also the direct cause of the current checkpoint's 39/40 panels — its first failure, seed
3077, is this bug.

Requirements:

* reset must produce a **nonterminal** initial state;
* or use deterministic, constraint-aware resampling, recording the number of attempts and the
  realized initial state;
* already-terminal task generation must **not** be counted as a policy failure;
* and it must **not** be discarded after outcomes are observed — post-hoc filtering of episodes
  by their result invalidates the panel.

> **STATUS — landed in `71a91b7`, and the root cause was narrower than described.** `[measured]`
>
> The defect was not reset randomisation in general. The root-height jitter was the **only
> unbounded term in the reset** — every other one is a bounded uniform — so
> `qpos[2] += normal(0, height_scale)` had nothing stopping it leaving `healthy_z_range`. T-Rex's
> 0.926 m home pelvis sits 0.226 m above the 0.70 m floor, which at σ = 0.10 m is **2.26σ**,
> predicting **1.19%** sub-floor spawns — closed-form, and it brackets both observed scans.
>
> Every failure was already sub-floor **at spawn**; none became terminal from the step itself, so
> bounding the draw removes the whole failure mode rather than most of it. The draw is now
> truncated to the distance to the nearer end of `healthy_z_range` less a 0.02 m margin:
> **43/4000 → 0/4000**. The bound is symmetric, so the mean spawn height is unchanged
> (0.9262 → 0.9261 m) while σ tightens 0.1007 → 0.0980. It binds at 2.07σ for T-Rex (3.9% of
> draws) against 3.6–3.8σ for velociraptor and dibothrosuchus (~0.02%), so only T-Rex moves
> materially. Dibothrosuchus had hit this same defect first and fixed it by decoupling
> `reset_height_noise_scale`; T-Rex was left coupled and nobody re-checked it.
>
> Requirement 1 is met structurally rather than by resampling, which keeps the RNG draw count
> per reset fixed and the reset deterministic; the cost is a small point mass at each bound.
> Requirements 3 and 4 are moot — there are no already-terminal episodes left to count or
> discard.
>
> **Two consequences.** Reset semantics are fingerprinted through `home_reset`, so the three
> home-keyframe-residual species took a policy-interface bump (velociraptor 6 → 7, trex 7 → 8,
> dibothrosuchus 3 → 4); their existing checkpoints were trained against a different reset
> distribution and **must be re-baselined before any gate here is calibrated on them**. And
> §2.3.1's central evidence is now stale — see the STATUS note there.

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

**Added in revision 4**, both cheap and both now unblocked:

7. **Re-measure the four 40-seed panels on the repaired reset.** This is the highest-value
   remaining evaluation, because §2.3.1's operating point currently rests on evidence known to
   be an artifact of the reset defect. Cheap, and it may remove the case for n=179 entirely.
   Note the available checkpoint is interface-invalid (rev 7 against 8) and was trained on the
   old reset distribution, so it answers *"was 39/40 the bug?"* but should not itself set the
   operating point.
8. **Does the §7.1 repair change the hop?** The one remaining `[inferred]` claim in §1.5 is
   causal — that the reward produced the bouncing. Airborne is no longer the cheap state, so a
   single stage-1 run under the repaired reward is the counterfactual that settles it. Nothing
   else in this document depends on the answer, but §6.3 and §1.5 both get firmer with it.

## 9. Risks

| risk | mitigation |
|---|---|
| Catastrophic forgetting at the 1a → 1b boundary | ramp *if* the §8.1 pilot supports it; the task change is otherwise small — same reward, same plant, one force |
| Added wall-clock for a fourth stage | 1a shortened to ~3M (§2.4); partly offsets |
| Stage-identity migration across four species and the website | semantic IDs + schema bump (§4); land as a separate prep change |
| 1b unlearnable if a species' 1a is weak | per-species `perturbation_delta_v`; enable per species only after its own preflight |
| Brachiosaurus zero action never reaches the horizon (0 of 40, `n_standing = 0`) | its 1a gate will fail, which is the `CHECK PLANT` signal surfacing. Note this is a *zero-action* result: it does not by itself show that no learned controller can stand, nor prove plant corruption. Investigate before concluding either. **Revision 4:** part of the explanation is now measured — its four foot touch sensors read 0.0 N against 1699.2 N of real floor contact, so it trains with four permanently dead observation channels and no foot-contact information at all (§7.2). That is a plausible partial cause that does *not* require plant corruption, though it does not establish one either. |
| **Velociraptor's foot sensors report 55.3% of true load** (§7.2, new in revision 4) | its 1a stance gate and any §6.3 support-state diagnostic read those sensors, so both would be calibrated against a 45% under-read. Repair the MJCF before enabling either stage for this species. This is the species whose *complete* stage-1 gate a statue already clears, so it has the least instrumentation margin to spare. |

## 10. Sequencing

**Steps 1–5 are done** (PR #478, `34a7002`). They were the environment and plumbing bugs, and
they required no part of the split — which is why they went first and shipped independently.

| # | item | status |
|---|---|---|
| 1 | **§7.5 reset validity** | done, `71a91b7` — 43/4000 → 0/4000 |
| 2 | **§5.2 fail-closed gate schema** | done, `17ca4e5` — config + SB3 + JAX + tests, atomically |
| 3 | **§7.4 collapse decoupling** | done, `17ca4e5` — every effective floor preserved |
| 4 | **§7.2 sensor verification** | done, `2ab5ae6` / `efe93dd` — and found two new defects |
| 5 | **§7.1 `foot_load_balance`** | done, `c667938` — with the parity coverage that was missing |

**Remaining, in order.** Step 6 is new in revision 4 and comes first because everything
downstream calibrates against it.

6. **Re-baseline** — **baseline half done, `ca56f6c`.** The plant defects that made the old
   panels meaningless are fixed (geometric reset settling, two home-pose self-collisions), and
   the statue baselines and noise sweep are re-measured for all four species in §2.3.1. What
   remains is the *policy* half: every checkpoint is now interface-invalid and was trained on a
   different task, so the four 40-seed panels (§8.7) need a fresh stage-1 run, not an
   evaluation. Nothing below should be calibrated until that lands.
6a. **Re-derive the four stage-1 `min_avg_reward` values.** Every one is currently cleared by a
   statue — trex 1840 vs 2243, and 100 vs 1746 / 163 / 1834. §2.3.1 carries proposals for
   review. Cheap, and independent of the fresh run.
6b. **Fix brachiosaurus's stance collapse.** Its statue scores 0/40 full-horizon (mean length
   130.7, `fallen` 34), so brachiosaurus stage 1 is not a balance task and no result for it is
   interpretable. Pre-existing, not a regression. Blocks any brachiosaurus gate.
7. **Repair the velociraptor and brachiosaurus foot sensors** (§7.2) — per-species MJCF changes.
   Not on the T-Rex critical path, but they gate enabling either stage for those species, and
   both invalidate that species' checkpoints, so they are cheapest done alongside step 6b.
8. §3.2 task fingerprint and load modes; §4 executable stage manifest with backward readers.
9. §2.3 episode-level gate metrics, implemented and parity-tested.
10. **Gate resolver (§5)** — before, or atomically with, any executable stage that depends on it.
11. Deterministic perturbation scheduler, with force-off regression, clearing, seed/schedule,
    SB3/MJX and resume tests. Default off — no behaviour change until enabled.
12. §8.1 T-Rex evaluation: zero action, constant/brace controls, the 1a checkpoint and the
    candidate, at noise 0.10 under the finalised full push, on registered calibration and
    held-out panels, saving one row per episode and per shove.
13. Calibrate thresholds; decide ramp versus fixed.
14. Stage split enabled for T-Rex only; other species after step 7 plus their own plant and
    learnability preflights.

Two orderings are worth stating explicitly, because earlier revisions got them wrong. The
resolver lands *before* stage enablement, not after (revision 2 had this backwards). And the
re-baseline lands *before* the operating point is declared — revision 3 declared candidate
operating points from panels that are now known to be contaminated by the reset defect.

## 11. Claim ledger

| label | meaning |
|---|---|
| `measured` | reproduced directly against the stated commit for this document |
| `artifact-derived` | taken from run artifacts or TREX_REVIEW_2026_07 §NS-1; not re-run here |
| `inferred` | reasoning, not measurement |
| `stale` | measured before a change that invalidates it; needs re-measuring |

**Two baseline commits are now in play.** Revision 3's `measured` claims were taken against
`48fd90a`. Revision 4's STATUS notes were taken against `34a7002`, which carries the reset
repair, the fail-closed gate schema, the collapse decoupling, the `foot_load_balance` repair and
the sensor audit. Where the two disagree, `34a7002` wins; each STATUS note says which.

**`measured` against `48fd90a`:** the statue per-term decomposition and settled plant constants;
the foot-state cost table and the `[0, 0]` hole; the `alternation_ratio` gait test (re-verified
still failing on `34a7002`); the `ec23125` → `48fd90a` counterfactual; and the exact binomial
bounds in §3.4.

**`measured` against `34a7002` (new in revision 4):** the reset-repair figures in §7.5; the
re-measured §1.4 baselines for T-Rex and velociraptor; the four-species sensor audit and the
duty-metric validation in §7.2; the body-mass correction in §6.2; the fail-closed reproduction
in §5.2; the collapse-floor equivalence in §7.4; seed 3077's post-repair spawn in §2.3.1; and
every code citation in the document, each re-resolved against the package split in #472–#477.

**`artifact-derived`:** the §NS-1 refuted-candidate table, the CoP-under-CoM measurement, the
push-on figures (also `stale`), the anti-gaming searches, the 120-seed paired comparison, and
the three-block seed stability figures.

**Newly `stale` in revision 4:**

* **§2.3.1's four-panel checkpoint result (39/40, 39/40, 40/40, 40/40).** Contaminated by the
  reset defect — its first failure, seed 3077, was the bug. Re-measure before declaring an
  operating point. This is the single most consequential stale claim in the document.
* **§6.2's `1460 N` policy GRF figure.** Never reproduced here, and the denominator it was
  compared against was wrong. Re-measure against the animal's kinematic subtree mass.
* **§1.4's brachiosaurus and dibothrosuchus rows.** Not re-taken; brachiosaurus's reset is
  unchanged and dibothrosuchus's moved only in the far tail, so both are expected to hold, but
  neither has been confirmed on `34a7002`.

**Not yet reproducible from this repository:** the push-on measurements in §3.4. The
perturbation implementation, exact schedule and raw per-episode outcomes are not present, so
those numbers cannot be independently verified until the scheduler lands (§10.11). They are
`stale` for the additional reason that they predate `435f35f`.

**Surrounding documentation — the revision-3 items are now closed.**

* The **PR #474 description** described revision 1. It should be rewritten to match revision 4;
  in particular its "Blocking prerequisites found during review" section now describes work that
  has shipped, and its "Known outstanding: §NS-2" line is resolved.
* `TREX_REVIEW_2026_07` **§NS-2 is superseded on `main`** as of `3774301` — explicitly, in the
  document itself, on the measured grounds §5.1 gives. Its §NS-1 was replaced by the task-change
  rewrite in the same commit, with four of that rewrite's own corrections withdrawn or narrowed.
  The contradictory-canonical-guidance risk revision 3 flagged no longer exists.
* The pushed-task numbers in §3.4 remain unreproducible from this repository, and the checkpoint
  available today differs from the artifact behind the published `2489.65`.
  `[artifact-derived, unverified]`

**Outstanding provenance work.** Each retained load-bearing number should carry its command,
exact effective config, backend, dependency versions, ordered seeds, raw episode data and
artifact hash. This document groups provenance by claim class rather than per number; a
per-measurement manifest is the next improvement. The two scripts added in #478 —
`foot_sensor_report.py` and `stance_duty_validation.py` — are the first load-bearing numbers
here that are reproducible by a single documented command, which is the pattern the rest should
follow.
