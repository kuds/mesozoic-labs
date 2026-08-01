# Plant Validation and the Stage 1 Objective

> **Date:** 2026-07-31
> **Status:** Design — Part I/II landed in [PR #479](https://github.com/kuds/mesozoic-labs/pull/479); Parts IV–V open
> **Baseline commit:** `ca56f6c` (measurements), `0384fca` (the run analysed)
> **Related:** [STAGE1_SPLIT_PLAN.md](STAGE1_SPLIT_PLAN.md) §2.3.1 (operating point),
> [investigations/TREX_REVIEW_2026_07.md](investigations/TREX_REVIEW_2026_07.md) §NS-1,
> [investigations/FOOT_SENSOR_VERIFICATION.md](investigations/FOOT_SENSOR_VERIFICATION.md)

Every claim carries a label from the ledger in §19.

---

## 1. Summary

Three findings, in the order they have to be understood.

**The plant was invalid.** Every reset of every training run to date started with the animal
intersecting the floor — up to 0.198 m — and with geom pairs permanently overlapping inside the
body. The contact solver responded with up to 19× body weight, ejecting the model into a
ballistic arc that no policy could influence. This is fixed.

**The objective is degenerate.** On Stage 1 as configured, doing nothing is the *global optimum*:
the zero-action policy collects 97.0% of the theoretical maximum return while paying exactly zero
energy and smoothness cost. No reward threshold can admit a competent policy and exclude a
passive one, because a competent policy scores *lower*. This is not a calibration problem.

**The instrumentation could not see either.** Four separate metrics were structurally incapable
of reporting the failure they were built to catch — including two repairs that shipped as no-ops.

The consequence for the Stage 1a/1b split is not that it was wrong. It is that its motivating
evidence was measured through a broken plant, and the case for it is now both stronger and
differently shaped.

---

## Part I — The plant was invalid

### 2. Reset interpenetration  `[measured]`

`BaseDinoEnv.reset` placed the root by height alone. Joint jitter changes how far the lowest
geom sits below the root, so no fixed root height is correct for most sampled poses.

Measured over 40 T-Rex seeds on the pre-fix plant:

| quantity | value |
|---|---|
| deepest floor penetration | **−0.198 m** (median −0.022 m) |
| resets exceeding 2× body weight at t=0 | **40 / 40** |
| contact force at t=0 | median 16,990 N (**11.5× weight**), max 28,517 N (**19×**) |
| mean penetration, resets that later failed | −0.088 m |
| mean penetration, resets that survived | −0.034 m |

The failure mechanism, traced step by step on seed 2:

```
t= 0  pelvis 0.746  feet 5777.6 / 6033.7 N   spawned inside the floor
t= 6  pelvis 0.927  feet    0.0 /    0.0 N   ejected, airborne
t=30  pelvis 1.255  feet    0.0 /    0.0 N   apex, +0.52 m
t=60  pelvis 0.879  feet 1638.2 / 1334.5 N   lands
t=66  forward_z -0.530                       nosedive termination
```

The episode opens with a catapult, ~60 ballistic steps with zero ground contact, and a
nose-first landing past the 0.493 nosedive threshold.

Spawns went the other way too: seed 0 started at pelvis 1.107 m with **no floor contact at all**
— 0.18 m in the air, opening with a free fall. The reset distribution spanned "0.20 m buried" to
"0.18 m airborne".

**Two controller sweeps confirmed the diagnosis by failing.** A PD controller on trunk pitch
driving ankle and hip (70 gain combinations, both signs, with and without a derivative term) and
a static posture sweep (75 combinations on ankle/hip/knee) both failed to beat zero action at
*every* setting. That is the signature of a failure occurring where control does not exist: for
the first ~60 steps there is nothing to push against.

### 3. Permanent self-collisions  `[measured]`

Two MJCF geom pairs overlapped in the home pose, injecting a constant, **pose-independent** force
into every step of every episode ever run:

| species | pair | depth | force |
|---|---|---|---|
| trex | `tail_1_geom` ↔ `r_thigh_geom` / `l_thigh_geom` | 18.8 mm | 8,143 N each = **16.3 kN** (11.0× weight) |
| brachiosaurus | `torso_main` ↔ `tail_2_geom` | 66.9 mm | **25.9 kN** (12.7× weight) |

Both tail chains already excluded *adjacent* links (`torso`↔`tail_1`, `tail_1`↔`tail_2`); both
bodies are long enough to reach two links back. The repo already carried the identical fix and
its rationale for the sibling toe capsules — the precedent existed and was not generalised.

Beyond corrupting the solver, this pollutes every contact-force reward and diagnostic in the
system.

### 4. Why the earlier repair missed it  `[measured]`

STAGE1_SPLIT_PLAN §7.5 identified "~1% of episodes terminal at generation", bounded the
root-height draw against `healthy_z_range`, and reported 43/4000 → 0/4000. That measurement was
correct and beside the point.

`healthy_z_range` is a **termination predicate on the root**. It constrains nothing about
foot-to-floor geometry. A pelvis at 0.739 m is comfortably "healthy" while the toes are 0.198 m
underground.

The generalisable lesson: *a bound expressed in the same variable as a termination rule validates
the termination rule, not the physics.* The right invariant was geometric — where is the lowest
body geom relative to the ground — and it was never checked. The measurement that looked like
validation (43/4000 → 0/4000) was real, specific, and answered a question nobody needed answered.

### 5. The repair, and what is now asserted  `[measured]`

`reset()` shifts the root so the lowest body geom sits at the clearance **the home keyframe was
authored with** — 0.491 mm of contact on T-Rex, per species elsewhere. Settling to the authored
depth rather than to a constant keeps the noise-free reset bit-identical and preserves real
foot-floor contact at spawn, which the species static-balance suites already require. Because the
floor is a horizontal plane, one shift settles the pose exactly: no iteration, no extra RNG draws,
determinism preserved.

An earlier draft settled to a fixed +2 mm clearance and turned four existing `test_static_balance`
suites red, because they assert the home pose has real foot-floor contact. That is the existing
suite doing its job, and it is why the settle target is the authored clearance.

T-Rex, zero action, 40 seeds:

| | before | after |
|---|---|---|
| t=0 contact force | 16,990 N (11.5× weight) | **190 N** (0.13×) |
| settled clearance spread across seeds | — | **0.000 mm** |
| bilateral duty | 0.983 | 0.992 |
| single-support duty | 0.003 | 0.008 |
| unsupported duty | 0.014 | **0.000** |
| survival | 25/40 | 28/40 |
| mean episode length | 683 | 757 |

`environments/shared/tests/test_reset_plant_invariants.py` asserts 8 invariants × 4 species,
independent of reward, stage and animal: no penetration, no hover, plausible t=0 contact force,
no home-pose self-collision, real ground contact at spawn, no already-terminal reset, settling
consumes no RNG, settle target does not drift.

**On first run they failed on brachiosaurus**, surfacing the `torso_main` overlap, which had not
been reported before.

---

## Part II — What the repaired plant measures

### 6. Statue baselines, all four species  `[measured]`

`zero_action_baseline.py`, seed 3042, 40 episodes, each species' configured stage-1 noise:

| species | noise | reward | mean−std | standing | full-horizon | terminations |
|---|---|---|---|---|---|---|
| trex | 0.10 | 2243.12 ± 1375.90 | 867.21 | 3250.27 ± 24.49 | 26/40 (65%) | fallen 2, nosedive 12, trunc 26 |
| velociraptor | 0.05 | 1745.84 ± 5.03 | 1740.81 | 1745.84 ± 5.03 | **40/40 (100%)** | trunc 40 |
| dibothrosuchus | 0.30 | 1833.53 ± 1163.36 | 670.18 | 2595.03 ± 3.29 | 28/40 (70%) | trunc 28, excessive_tilt 10, tail_contact 2 |
| brachiosaurus | 0.05 | 163.35 ± 81.40 | 81.95 | **never** | **0/40 (0%)** | fallen 34, tail_contact 6 |

T-Rex before → after the repair: reward 1974.74 → **2243.12**, mean−std 492.94 → **867.21**,
full-horizon 57.5% → **65%**, episode length 640.85 → **716.5**.

Three things fall out.

**The trained policy is now below do-nothing.** Run `20260731_132102` peaked at **1992.7** mean
evaluation reward against the repaired statue's **2243.12** — while clearing
`min_avg_reward = 1840` sixteen times, including an eight-evaluation streak.
`[artifact-derived]`

**Every species' stage-1 gate is cleared by its own statue**: trex 1840 vs 2243; velociraptor,
brachiosaurus and dibothrosuchus all 100, vs 1746 / 163 / 1834.

**Brachiosaurus stage 1 is not a balance task.** Its statue falls on 40 of 40 episodes at a mean
length of 130.7. There is no "do not fall" floor to beat, so no brachiosaurus stage-1 result is
interpretable. Verified identical before and after the repair — pre-existing, not a regression.

**Velociraptor is the species to prototype gates on**: 40/40 at ± **5.03**. That variance gives a
paired test real power at n=40; T-Rex's ± 24.49 (standing) / ± 1375.90 (overall) does not.

### 7. Reset noise is the 1a/1b dial  `[measured]`

`zero_action_baseline.py trex --sweep-noise`, 40 episodes each:

| reset noise | reward | mean−std | standing | ep length | full-horizon |
|---|---|---|---|---|---|
| 0.01 | 3287.9 | 3286.3 | 3287.9 | 1000.0 | **100%** |
| 0.05 | 3271.8 | 3259.7 | 3271.8 | 1000.0 | **100%** |
| **0.10** ← stage-1 default | 2243.1 | 867.2 | 3250.3 | 716.5 | 65% |
| 0.15 | 1348.2 | −96.9 | 3209.9 | 465.7 | 38% |
| 0.20 | 845.4 | −411.7 | 3166.9 | 324.4 | 22% |

The `standing` column is nearly flat (3288 → 3167) while `full-horizon` collapses 100% → 22%.

**Reset noise does not make standing harder. It decides how often you get to stand at all.**

Stage 1 sits exactly on the knee between 0.05 and 0.10, which is why survival and stance quality
have been inseparable in every measurement to date, and why a single scalar gate over both has
never been able to say anything useful.

### 8. Statue stance quality at the proposed 1a point  `[measured]`

`stance_quality_baseline.py 0.05 40` — the companion that measures duty rather than reward:

| species | full-horizon | all-feet duty | unsupported duty | contact switches | standing reward |
|---|---|---|---|---|---|
| trex | 100% | 0.998 | 0.000 | 0.09 /s | 3271.8 ± 12.0 |
| velociraptor | 100% | 1.000 | 0.000 | 0.00 /s | 1745.8 ± 5.0 |
| dibothrosuchus | 100% | 0.997 | 0.000 | 0.14 /s | 2598.3 ± 0.9 |
| brachiosaurus | 0% | 0.000 | 0.885 | 0.00 /s | never |

---

## Part III — The objective is degenerate

### 9. The statue is the global optimum  `[measured]`

Summing the positive T-Rex stage-1 weights:

| term | weight |
|---|---|
| `alive_bonus` | 1.00 |
| `height_weight` | 0.60 |
| `bilateral_support_weight` | 0.60 |
| `leg_home_pose_weight` | 0.50 |
| `head_clearance_weight` | 0.35 |
| `neck_posture_weight` | 0.20 |
| `heading_weight` | 0.10 |
| **theoretical maximum** | **3.35 / step = 3350** |

The statue collects **3250.27 = 97.0%** of it, with `energy` and `smoothness` at exactly zero.

Any active policy pays both. Run `20260731_132102` paid **0.30/step on smoothness alone** — a
300-point handicap before anything else. A policy's realistic ceiling is therefore *below* the
statue's score.

**Consequence: no reward threshold separates a competent policy from a passive one.** Set it
above the statue and Stage 1 is unpassable; set it below and a statue passes. This retires reward
thresholds for 1a rather than asking for better numbers, and it is a stronger statement than
STAGE1_SPLIT_PLAN STAGE1_SPLIT_PLAN §1.3's "the stance reward saturates against a statue."

### 10. The expectation tie  `[measured]`

Worse than a ceiling problem, the objective is *flat* across the two behaviours we care about
distinguishing:

| | reward per surviving episode | survival | expected return |
|---|---|---|---|
| zero-action statue (pre-repair) | 3243.1 | 57.5% | **1974.7** |
| trained policy @ 4.10M | 2054.0 | 93% | **1992.7** |

**0.9% apart.** "Quiet but fragile" and "buzzy but robust" are worth the same to this objective.
There is no gradient preferring the good one, and PPO landed in the buzzy basin. As long as
chatter can buy back its cost in survival probability, the policy will keep paying — which means
an airborne penalty alone will not fix it. Breaking the tie is the design requirement.

---

## Part IV — The instrumentation could not see the failure

## 11. Metrics that cannot report their own failure mode

Four metrics were structurally incapable of reporting what they were built to catch. Two were
repairs that shipped as no-ops.

### 11.1 `foot_load_balance` shipped inert  `[measured]`

STAGE1_SPLIT_PLAN §7.1's repair set `foot_load_balance_min_support_force = 0.0`, intending airborne to cost the
same as single support. `derive_stance_info` and `reward_foot_load_balance` differ *only* in the
near-zero branch, so if the airborne branch ever fired the two would diverge. Over all **709**
logged PPO rollouts, spanning 30–67% unsupported duty:

```
max |reward_imbalance - diag_imbalance| = 0.000000
correlation                              = 1.000000
```

Neither branch ever fires. The sum of two MuJoCo touch-sensor readings is essentially never
exactly `0.0`, so `total > 0.0` never selects the airborne case. Meanwhile the *duty* metrics use
`force > 0.1 N` per foot. A foot registering 0.001 N counts as **unsupported** in the diagnostic
and **supported** in the reward.

`derive_stance_info` also still carries the original defect it was meant to expose: true-airborne
yields `imbalance = 0.0`, i.e. *perfect balance*. It simply never triggers either.

### 11.2 `smoothness_weight` is blind to frequency  `[measured]`

It penalises action-delta *magnitude*. From the best checkpoint to the final one:

| | best (4.65M) | final (6.0M) |
|---|---|---|
| `action_delta` | 12.0 | **10.5** |
| smoothness penalty | −0.286 | **−0.250** |
| toe-motion power > 4 Hz | 35% | **71%** |

The policy got *smoother by the metric* while getting buzzier in fact. A small-amplitude
high-frequency limit cycle is nearly free under this term.

### 11.3 Contact-switch rate conflates two states  `[measured]`

The plant repair moved T-Rex's raw switch count **up** (0.86 → 1.00 /s) while unsupported duty
went to **zero**. The extra switches are bilateral↔single weight-shifting, not bilateral↔airborne
chatter. The metric cannot tell them apart and must not be gated on until decomposed.

### 11.4 The collapse detector was disarmed by a reward-scale change  `[measured]`

`collapse_peak_floor = 2200` was calibrated against the 7/29 run, whose rolling-median peak
reached 2496. The reward-scale shift lowered the 7/31 run's peak to **1934.1**, below the floor.
The detector never armed, and watched a **−59%** collapse (2148.3 → 888.0, full-horizon 93% → 7%)
without firing.

Simulated against the actual series:

| config | max rolling-median peak | armed | stopped |
|---|---|---|---|
| `peak_floor=2200` (shipped) | 1934.1 | **never** | never |
| `peak_floor=1840` (pre-SPLIT_PLAN §7.4) | 1934.1 | 2.9M | never |

Note the second row: even armed, `drop_fraction=0.5` + `patience=10` would not have caught it.
**An absolute reward floor cannot survive a reward-function edit** — it needs to be relative to
the zero-action standing baseline, and the drop/patience pair needs tightening independently.

### 11.5 What the diagnostics did say, and how it misled  `[measured]`

Through the entire collapse, training-rollout diagnostics looked healthy or improving: per-step
reward flat (1.670 → 1.795 → 1.741), support duty improving (bilateral 0.482 → 0.567, unsupported
0.381 → 0.300), posture stable, and PPO textbook-healthy (`approx_kl` 0.0123 → 0.0113 against a
0.03 target, `clip_fraction` 0.138 → 0.123, `explained_variance` 0.938 → **0.983**, policy std
decaying smoothly 0.818 → 0.743).

The collapse was **entirely episode-length driven** and visible only in deterministic evaluation.
Healthy training rollouts plus failing deterministic eval is a real signature and none of the
existing dashboards surface it.

### 11.6 Video analysis  `[measured]`

Frame-by-frame measurement of `trex_ppo_stage1_best.mp4` / `_final_side.mp4`. **The renders are
1000 frames at 50 fps for a 1000-step episode at 100 Hz control — 2× slow motion.** Figures below
are real-time.

| | best (4.65M) | final (6.0M) |
|---|---|---|
| episode survived | 20.0 s (full) | 8.5 s (fell) |
| toe lift-downs / sec | ~15 | ~14 |
| dominant fast frequency | 4.3 Hz | **13.5 Hz** |
| power above 4 Hz | 35% | **71%** |
| unsupported duty | 0.351 | 0.300 |

Two superimposed oscillations: **~15 toe lift-downs per second** (~7 control steps per cycle —
a control-bandwidth limit cycle, not a gait) and a **~0.6–1.0 Hz whole-body crouch↔extend bob**,
~30 cm peak-to-peak.

Stance breakdown at the best checkpoint: **bilateral 52.1% / airborne 35.1% / single 12.8%**.
Single support is the *rarest* state. A walk alternates bilateral↔single; this alternates
bilateral↔airborne.

The posture itself is not wrong — horizontal spine, counterbalancing tail, digitigrade stance is
the correct modern reconstruction, and pelvis height sits on target. The defect is the
oscillation.

---

## Part V — Design response

### 12. What Stage 1a must be

The measurements in §7 hand the split its operating point directly.

**Run 1a at reset noise ≤ 0.05**, where a statue reaches 100% full-horizon and survival is not
the binding constraint. Only then does a stance-quality gate measure stance quality rather than
luck.

**Gate on the episode-level `stance_success` event** (STAGE1_SPLIT_PLAN §2.3), not on reward.
§9 makes this mandatory, not preferable.

**Gate on unsupported duty; treat switch rate as diagnostic** until §11.3 is resolved.

Proposed per-species thresholds — **for review, not adopted**:

| species | noise | full-horizon | unsupported duty | contact switches | reward floor |
|---|---|---|---|---|---|
| trex | 0.05 | ≥ 95% | ≤ 0.02 | ≤ 1.0 /s | ≥ 2900 (0.89 × statue) |
| velociraptor | 0.05 | ≥ 95% | ≤ 0.02 | ≤ 1.0 /s | ≥ 1550 |
| dibothrosuchus | 0.05 | ≥ 95% | ≤ 0.02 | ≤ 1.0 /s | ≥ 2300 |
| brachiosaurus | — | **blocked** — §6 | | | |

The reward floor is a **sanity rail, not the gate**. It sits *below* the statue on purpose,
because above it is unreachable; its only job is to reject a policy that has discarded most of
the available return. The 0.89 multiplier is a round number chosen to clear the ~0.30/step
smoothness cost a reasonable active policy pays. It is `[inferred]`, not measured, and should be
revisited once anything clears 1a.

Duty and switch ceilings sit well above the statue's measured 0.000 and 0.00–0.14 /s so that
normal weight-shifting is not penalised.

### 13. What Stage 1b must be

1b supplies robustness through `xfrc_applied` perturbations at a declared magnitude — **not**
through reset noise, where a lucky draw and a good controller are indistinguishable (§7).

1b is also where the *discrimination* happens. A statue passes 1a by construction, and that is
correct: 1a certifies a policy has not bought stability with actuation. A statue cannot recover
from a shove, so 1b is where a policy must demonstrate it has learned something a statue has not.
Its gate must be measured against **the statue under the same perturbation schedule**, which
requires the perturbation mechanism to land first.

### 14. What must change in the reward

Ordered by confidence, all currently unimplemented:

1. **Give `foot_load_balance_min_support_force` a real value.** As shipped it is a no-op (§11.1).
   It must be consistent with the duty metrics' 0.1 N/foot threshold, or better, a fraction of
   body weight (~40 N ≈ 5% for the T-Rex subtree excluding the 65.45 kg prey body). And the
   ordering must be **monotone** — airborne strictly worse than single support, not equal to it,
   which is a flat region with no gradient out of the air.
2. **Add a frequency-aware cost.** Penalise contact-state switching rate directly, or put
   smoothness on the second difference of actions. The current term cannot see a 15 Hz buzz
   (§11.2).
3. **Make `collapse_peak_floor` relative** to the zero-action standing baseline, and tighten
   `drop_fraction`/`patience` independently (§11.4).
4. **Fix `derive_stance_info`'s airborne branch** so the diagnostic can report the difference the
   reward is trying to create (§11.1).
5. **Unsaturate the dead terms.** `head_clearance` sits pinned at exactly its full weight (0.350)
   in every measured window; `height` 0.578 of 0.6; `neck_posture` 0.173 of 0.2. Saturated terms
   contribute no gradient.

**These should wait for the fresh run in §18**, for the reason in §16. Two confident reward-side diagnoses in this
investigation turned out to be wrong, both because the environment underneath was lying.

---

## Part VI — Method

## 15. How each probe was run

Each probe, with what makes it reproducible. The pattern worth keeping: **the negative results
were the informative ones.**

| probe | tool | what it establishes |
|---|---|---|
| statue baseline | `zero_action_baseline.py <species>` | the floor every run must clear |
| noise sweep | `zero_action_baseline.py trex --sweep-noise` | separates survival from stance quality |
| stance quality | `stance_quality_baseline.py <noise> <episodes>` | duty and switch rate, not reward |
| plant invariants | `pytest environments/shared/tests/test_reset_plant_invariants.py` | reset geometry, all species |
| controller sweeps | scratch (§2) | *negative* result localising the failure outside control |
| video measurement | scratch — silhouette + floor-reflection tracking | frequency content the duty ratios cannot show |

Three methodological notes worth carrying forward:

**A same-seed control run is worth more than any absolute number.** The 7/29 run had identical
reward weights, hyperparameters, seed, and budget — differing only in the two changes under test.
Without it, "reward 1900" means nothing. *Check that the comparator is the same stage*: an early
comparison in this investigation was accidentally made against a stage-3 bite run and produced a
confidently wrong conclusion.

**When every setting of a controller sweep fails, suspect the plant.** 70 PD gains and 75 static
postures, none beating zero action, is not a tuning problem.

**Live artifacts tear.** `evaluations.npz` downloaded mid-write produced a byte-stitched file
whose reward array was silently garbage past eval 57 while its headers and other members parsed
cleanly. Verify CRCs; do not disable them to force a read.

---

## Part VII — What is not known

## 16. Open empirical questions

Stated explicitly so it is not mistaken for settled.

**How much of the chatter survives on the repaired plant.** A substantial share of the airborne
duty may have been *learned from the broken reset* — if episodes routinely opened with a catapult
and ~60 ballistic steps, airborne-tolerant behaviour is exactly what training rewarded. The
behaviour may be a rational response to a broken environment rather than a reward-design flaw.
**Unmeasured.** This is the single highest-information open question.

**Why the 7/31 run underperformed the 7/29 control.** With `foot_load_balance` proven inert
(§11.1), the only remaining delta is the reset height clip — which folded the Gaussian tails onto
the clip boundary, redistributing exactly the penetration depth that drove failure. Plausible
mechanism, `[inferred]`, not demonstrated. n=1 seed either way.

**Whether a controller exists that is both quiet and reset-robust.** Zero action on the repaired
plant reaches 78% survival at 0.2% airborne duty, but still fails 9/40 — residual nosedive, 12 of
14 failures at noise 0.10. That residual is the legitimate 1a target, and no controller has yet
been shown to close it.

**Whether the proposed 1a thresholds are achievable.** They are derived from the statue, which is
an upper bound on quality and a *lower* bound on difficulty. Nothing has yet cleared them.

**Brachiosaurus's collapse mechanism.** Confirmed 0/40, confirmed pre-existing, not diagnosed.

---

## Part VIII — Decisions required

## 17. Decisions that need a human call

Not recommendations — these need a human call.

1. **The four `min_avg_reward` values.** Every one is cleared by its own statue. Whatever replaces
   them, the current values certify nothing. Cheap, independent of any run.
2. **1a's operating noise.** §7 argues ≤ 0.05. This changes what Stage 1 *is*, so it is a design
   decision, not a tuning one.
3. **Whether the reward floor rail (§12) is worth having at all**, given it cannot be the gate.
4. **Whether to fix brachiosaurus's stance before or after the T-Rex path clears.** It blocks
   only brachiosaurus.

## 18. Recommended sequence

1. **Fix the four gate values** and **fix brachiosaurus's stance.** Neither needs GPU time.
2. **Run a fresh T-Rex stage-1 pilot on the repaired plant — 3M steps, not 6M.** It answers §16's
   headline question: does the chatter survive? The prior run reached 93% survival by 4.6M, so
   the trend is visible well before the endgame. Judge it on **airborne duty and switch rate, not
   reward** — a good policy will still score below the statue (§9), and that is expected, not a
   failure.
3. **Then** the reward changes in §14, informed by what that run shows.
4. Perturbation mechanism, then 1b's gate against the statue-under-perturbation.

## 19. Claim ledger

| label | meaning |
|---|---|
| `measured` | reproduced directly against `ca56f6c` for this document |
| `artifact-derived` | from run `20260731_132102` / `20260729_151044` artifacts; not re-run |
| `inferred` | reasoning, not measurement |

**`measured`:** every figure in Parts I, II and III; the `foot_load_balance` inertness proof over
709 rollouts; the smoothness and switch-rate blindness; the collapse-detector simulation; the
video frequency analysis; the four-species statue and stance-quality baselines; the noise sweep;
the positive-weight sum and the 97.0% figure.

**`artifact-derived`:** the 7/31 run's evaluation series and per-term diagnostics; the 7/29
control run's series; the expectation-tie table in §10.

**`inferred`:** the proposed 1a thresholds and the 0.89 reward-floor multiplier; the reset-clip
mechanism for the 7/29→7/31 regression; the claim that airborne duty was partly learned from the
broken reset.

**Not claimed:** any statement that the repaired plant produces better *training* outcomes. No
training run has been performed on it.

---

## Addendum — 2026-07-31, after `ca56f6c`

Work landed on the review branch since this document was written, recorded here so the sections
above stay a faithful snapshot:

* **§6/§8's brachiosaurus rows are resolved** (plant_versions notes 7–8). The collapse mechanism
  §16 left undiagnosed was measured to be two defects: the midpoint action mapping never
  commanded the home pose (knees dragged up to 0.349 rad off), and the leg servos sagged
  71.6 mm under static weight, leaving the planted stance's roll stiffness at parity with
  `m·g·h` so the statue tipped over in slow roll even holding home exactly. With the residual
  mapping and doubled leg kp the statue is **40/40 full-horizon at 1739.08 ± 1.17** (was 0/40 at
  163.35 ± 81.40); with the foot-sensor repair its stance-quality row reads all-feet 0.998 /
  unsupported 0.000 (was 0.000 / 0.885). §12's "blocked" row and decision §17.4 are discharged;
  the **± 1.17** spread supersedes velociraptor's ± 5.03 as the tightest gate-prototyping
  variance.
* **The MJX reset now settles on the ground** like §5's Gymnasium repair (plant_versions note 6
  addendum). Un-settled MJX spawns measured −41.2 mm to +5.2 mm at stage-1 noise on T-Rex, and
  the brachiosaurus midpoint base pose hovered 610 mm. Part I's findings therefore applied to
  the JAX path too; they no longer do.
* **§16's HEADLINE QUESTION IS ANSWERED — and the answer is "no".** Run `20260801_021545`
  (T-Rex, 6.0M steps, this document's repaired plant, `reset_noise_scale = 0.05`) ended at
  **28.4% unsupported duty** against the statue's 0.000, versus 30–35% on the broken plant.
  §16 asked whether "a substantial share of the airborne duty was learned from the broken
  reset." It was not: the plant repair moved it by ~19% relative and left it an order of
  magnitude away from the target. **The chatter is a reward-design problem**, which promotes
  §14 from inferred to evidence-backed. Duty fell monotonically (0.657 → 0.524 → 0.322 →
  0.284) and had *not* converged at 6M — so §18's 3M pilot would have read ~0.45 and
  understated the trend. `[measured]`
* **§14 items 1–4 are now implemented** on the strength of that result: a real
  `foot_load_balance_min_support_force` (42 N = 5% of the animal's own weight, versus the
  measured no-op at 0.0), a `foot_load_balance_airborne_penalty` making the ordering strictly
  monotone (`+0.600 > −0.300 > −0.600`, ending the flat region §14.1 identified), a
  frequency-aware `action_jerk_weight` on the second difference of actions (a slow ramp scores
  0.00 where a Nyquist buzz scores 336, while the *first* difference rates the ramp as
  rougher — the §11.2 blindness, inverted), and `derive_stance_info`'s airborne branch keyed to
  the same threshold the duty metrics use.
* **§11.4 repeated, and the absolute floor is retired.** `collapse_peak_floor` failed to arm a
  second time: 2450 (0.75 × statue) against a run whose best evaluation was **2347.67**, so the
  detector watched eval degrade 2347.67 → 1666.33 without firing. Deriving the floor from the
  statue was only half a fix — the statue bounds what is *achievable*, not what a *learning*
  policy passes through. It is now relative (`collapse_peak_floor_fraction` ×
  `collapse_peak_floor_reference`, 0.45 × 3271.8 = 1472), which the failing run cleared by ~2M
  steps while still sitting well above the 888 collapse bottom.
* **§11.5 repeated too**: rollout diagnostics improved monotonically straight through the
  window where deterministic evaluation degraded. Healthy rollouts plus failing eval, still
  surfaced by no dashboard.
* **The 2026-08-01 run was not the §18 pilot** — 6M steps with advancement enabled through
  stages 2 and 3, not the 3M stage-1-only diagnostic. Its stage 1 "passed" on a transient peak
  at 5.75M while the *final* model fails both gates (1666.33 < the 1950 rail; 742.8 < the 950
  length floor). Note the 0.89 × rail this addendum superseded would have blocked that
  advancement; 0.60 × did not. The deeper point stands either way — no reward threshold
  detects 28.4% airborne duty, which is what the unbuilt `stance_success` gate is for.
* **§16's reset-height-clip hypothesis is retired going forward**: the ground settle makes the
  entire root-height jitter channel state-inert (verified to one ULP), so the clip cannot
  influence any future run. It remains a candidate explanation for the historical 7/29→7/31
  regression.
* **§17's decisions are now ADOPTED, not provisional**, following a decision review on
  2026-08-01. Each is recorded here with what actually got decided:

  1. **§17.2 — 1a operating noise: `reset_noise_scale = 0.05`, all four species.** §7's argument
     taken as written. Accepted consequence: a statue passes 1a by construction, with
     discrimination deferred to 1b's perturbations (§13). Accepted risk: if 1b never lands,
     stage 1 is trivially passable.
  2. **§17.3 — the reward rail is kept, as a gate component.** It stays in `min_avg_reward` and
     blocks advancement, because it is the only enforcement slot that exists and the collapse it
     catches is real.
  3. **§17.1 — the four rail values are 0.60 × statue, NOT §12's 0.89 ×**: trex 1950,
     velociraptor 1050, brachiosaurus 1040, dibothrosuchus 1560. §12 chose 0.89 to sit just
     below a competent policy, which is competence-bar reasoning that §9 refutes. Sized instead
     to the rail's actual job: the measured collapse bottomed at **888 = 0.27 × statue**, so
     0.60 clears it by better than 2×, whereas 0.89 = 2900 sat within ~2.4% of a competent
     policy's estimated ceiling (~2970 = statue − the 0.30/step smoothness cost, before energy
     and the posture terms a moving policy gives up) and risked rejecting the policy it was
     meant to admit. **§12's table is superseded on this point.**
  4. **§12's gate statistic is a hybrid**, reconciling §2.3 (binomial LCB) with §12 (raw
     fractions), which disagreed by ~5 points of certified capability: raw-fraction screening at
     n=40 with `required_consecutive` as scheduler hysteresis only, a one-sided bound on **mean
     unsupported duty** at n=40 where a continuous metric has real power, and one predeclared
     held-out panel at n≈100–180 for the binary full-horizon event. See STAGE1_SPLIT_PLAN §2.3.
  5. **§17.4 — brachiosaurus was fixed first** (notes 7–8 above), so no ordering question
     remains.

  `min_avg_episode_length = 950` encodes the full-horizon ≥ 95% floor and is **now enforced on
  both backends** — the JAX gate previously read only `min_avg_reward`, half-enforcing the gate
  kind named `reward_and_length/v1`. `collapse_peak_floor` is 0.75 × each statue's standing
  reward, still absolute pending §14 item 3. The `stance_success` machinery and the §14 reward
  changes remain unimplemented — the latter deliberately, per §14's "wait for the fresh run".
