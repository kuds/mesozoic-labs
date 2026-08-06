# T-Rex Stage 1: the phase-locked bounce, and what three runs taught us

**Date:** 2026-08-05
**Runs:** `20260804_143747` (FAIL), `20260805_011234` (**PASS**), `20260805_132950` (FAIL)
**Issues:** #486 (closed), #489, #491 · **PRs:** #487, #490, #492

Point-in-time record. Not rewritten — corrections are appended.

---

## 1. Summary

Three consecutive 10M-step stage-1 runs, all seed 42, same plant. One passed.
The two that failed converged to a **phase-locked vertical bounce** at an exact
integer subharmonic of the control rate, with both feet leaving and landing
together.

| run | `leg_home_pose_weight` | duty | single support | frequency | verdict |
|---|---|---|---|---|---|
| `20260804_143747` | 0.5 | **0.1668** = 1/6 | 0.0000 | 16.7 Hz | FAIL |
| `20260805_011234` | 0.5 | **0.0000** | 0.0000 | — | **PASS** |
| `20260805_132950` | 1.5 | **0.2001** = 1/5 | 0.0011 | 20.0 Hz | FAIL |

The headline result is negative and worth stating plainly: **the change made in
#490 did not work, and the instrumentation added in the same PR is what proved
it.** That is the outcome the instrumentation was built for.

---

## 2. What was tried, and what happened

**#487 — entropy decays to zero over 70% of the budget.** Diagnosed #486's
16.7 Hz bounce as an optimisation failure driven by an `ent_coef` floor holding
policy std at ~0.375 (≈20.6° of commanded joint noise). Fix: `ent_coef_end`
0.001 → 0.0, decay 3M → 7M. **This worked** — run `20260805_011234` passed with
duty 0.0000.

**#489 — the residual gap.** The passing policy scored 3004.3 against the
zero-action statue's 3274.4. Inverting the action penalties through their closed
forms gave a per-actuator DC offset of ~0.800 and an AC tremor of ~0.277 rms at
an effective 22.6 Hz, and argued the two were **one phenomenon**: the tremor is
closed-loop feedback holding up a displaced pose.

**#490 — raise `leg_home_pose_weight` 0.5 → 1.5,** on that hypothesis, plus
per-actuator instrumentation to check it.

**Run `20260805_132950` — FAIL.** Duty locked at exactly 1/5 for the entire
second half of the run.

---

## 3. Why #490 could not have worked

`leg_home_pose_joint_names` for trex is **eight joints**: r/l `hip_pitch`,
`hip_roll`, `knee`, `ankle`. Measured per actuator (4.5M+, in-flight):

| | DC | AC (tremor) |
|---|---|---|
| the 8 governed joints | 0.035 – 0.263 → **1.2% of DC power** | 0.681 rms |
| the other 13 | up to ±0.996 → **98.8%** | 0.253 rms |

The governed joints were **already at home**. The displacement lives in the
tail, neck, head and toes, which no term in this stage touches.

The final report is the cleanest possible confirmation. `reward_leg_home_pose`
reached **1409.17 of the statue's 1466.80 — 96%**. *The term did exactly what it
was asked to do.* `action_dc_rms` still finished at 0.724 against the passing
policy's 0.766, and the stage failed anyway.

**Lesson: the joint list, not the weight, was the lever.** A term's weight can
only move what the term measures.

---

## 4. The bounce is not reward-preferred

Restating both policies under the **same** reward (only `leg_home_pose`
rescales between them):

| | score |
|---|---|
| statue | 3271.8 |
| passing policy (duty 0.0000) | 3004.3 |
| **bouncing policy (duty 0.2001)** | **2554.3** |

The bounce is **450 points worse**, and loses on every term but the one whose
weight was raised:

| term | passing | bouncing | Δ |
|---|---|---|---|
| `foot_load_balance` | −16.9 | −154.9 | **−137.9** |
| `bilateral_support` | 594.2 | 477.8 | **−116.5** |
| `alive` | 995.2 | 898.1 | **−97.1** |
| `neck_posture` | 175.6 | 116.0 | −59.6 |
| `smoothness` | −65.0 | −102.4 | −37.4 |
| `action_jerk` | −41.4 | −68.0 | −26.6 |

**This rules out the obvious reward tweaks.** Every candidate is already firing,
already correct, and already losing:

- `foot_load_balance_airborne_penalty` charges the bouncer 138 points more.
- `support_conditioned_alive_fraction` costs it 97 points of the single largest
  term in the reward.
- `action_jerk_weight` charges it 64% more than the policy that passed.

None deterred it. Raising any of them deepens a trap the optimiser is already
stuck in without creating a route out.

**Lesson: this is an optimisation failure, not a reward-shaping one** — the same
category as #486, and the same mistake that would have been made twice if a new
penalty had been proposed here.

---

## 5. The tremor is load-bearing (#489's other claim held)

`stance_gate_report.py --filter-actions`, on the **passing** checkpoint, a
first-order low-pass between policy and plant:

| cutoff | mean episode length | outcome |
|---|---|---|
| none | 1000 | PASS, duty 0.0000 |
| 35 Hz | 351 | falls, `tail_contact` |
| 20 Hz | 289 | falls |
| 10 Hz | 189 | falls |
| 5 Hz | 96 | falls |

Control runs at 100 Hz, so 35 Hz is most of the way to Nyquist. The failing
policy behaves the same way (109 steps at 5 Hz, 40/40 `tail_contact`).

**An action low-pass or rate limit cannot be retrofitted.** If one is wanted for
sim-to-real it must be present during training.

**Caveat:** a causal low-pass adds phase lag as well as attenuation, and a tight
stabilising loop is delay-sensitive. This proves the policy cannot tolerate added
latency; it does not cleanly separate "needs 22 Hz of bandwidth" from "needs zero
delay". A zero-phase (`filtfilt`) offline pass would isolate that.

### The coupling hypothesis was wrong

#489 argued the tremor holds up the displaced pose. Per-actuator measurement
says otherwise: the **displaced** joints barely move (tail/neck/head/toes, AC
0.253) and the **shaking** is in the load-bearing legs (hips/knees/ankles, AC
0.681). Different joints, different phenomena. The tremor being load-bearing and
the pose being displaced are both true and **not the same fact**.

---

## 6. The control rate permits the bounce; it does not encourage it

Both failures landed on exact integer subharmonics of the 100 Hz control rate —
1/6 and 1/5. **That alone is not evidence of pathology:** in a deterministic
discrete-time closed loop every limit cycle has an integer period by definition.

What does survive is the *frequency*. Body-on-leg resonance here is ~1.1–1.4 Hz;
16.7 Hz on this mass would need ~1.66 MN/m of stiffness, and the statue holds
`bilateral 1.0000` — it does not bounce passively. **The oscillation is actively
driven**, which requires command authority in the 20 Hz band.

| | control | Nyquist | period-5 cycle | 20 Hz reachable? |
|---|---|---|---|---|
| **frame_skip 5 (current)** | **100 Hz** | 50 Hz | 20.0 Hz | yes |
| frame_skip 10 | 50 Hz | 25 Hz | 10.0 Hz | yes |
| frame_skip 16 | 31 Hz | 15.6 Hz | 6.2 Hz | **no** |

The task needs almost none of that: 100 Hz gives **71 control steps per balance
cycle** at 1.4 Hz, and even 25 Hz control would give 18.

**But the passing run ran at the same 100 Hz and did not bounce.** The control
rate is a background condition that makes the failure *reachable*, not the thing
that separates pass from fail. Changing `frame_skip` bumps
`policy_interface_revision` and invalidates every checkpoint and every measured
constant in the config — too large a change to buy on this evidence.

**Falsifiable test if it becomes worth it:** train at `frame_skip 10`. A bounce
at period 5–6 control steps (8–10 Hz) means the cycle is locked to the control
clock and lowering the rate removes the failure class. A bounce at ~20 Hz again
means it is a plant/task property and the rate is irrelevant.

---

## 7. Actuator saturation (open — see #491)

**Ten to twelve of 21 actuators sit pinned at `|action| ≥ 0.99`** — tail, neck,
head, toes — in *both* the passing and failing policies. Nothing in the reward
opposes this: `energy` penalises `mean(a²)` at weight 0.075 (~48/episode against
an alive bonus paying ~1000), and `leg_home_pose` covers 1.2% of the offset.

A saturated actuator has **no headroom in one direction**, so the recovery
envelope on those axes is one-sided.

This is a genuine gap, and it is the one reward idea whose mechanism is *not*
already firing — but the **passing** policy saturates just as hard, so it is not
what separates pass from fail. Treat it as sim-to-real (#491), not a stage-1
blocker.

---

## 8. Method lessons

**Measure before shaping.** The #489 inversion was *correct* — reproducing the
inferred tremor on the statue matched the trained policy's per-step penalties to
within 2%, and direct measurement later confirmed DC 0.766 / AC 0.359 / 22.7 Hz
against the inferred 0.800 / 0.277 / 22.6 Hz. The **causal story built on top of
it** was wrong. Being able to measure a quantity accurately is not the same as
knowing what causes it.

**Instrumentation earns its cost by refuting you.** #490 shipped a config change
and the instrumentation to check it. The instrumentation refuted the change. Had
only the config change shipped, the run would have looked like ordinary bad luck.

**n = 1 is not a result.** Three runs, all seed 42: one pass, two bounce. Nothing
here establishes whether passing is *reliable* or was *lucky*, and that question
determines everything about what to do next. **Seed replicates are the highest-
value next experiment** — higher than any reward tweak.

**Derived constants go stale silently.** `min_avg_reward` and
`collapse_peak_floor_reference` are both derived from the zero-action statue,
and the statue moves whenever a reward weight moves. Because the statue commands
`action = 0` at any weight, its trajectory never changes and only the affected
term rescales — which makes the new value exact and cheap to obtain, and makes
forgetting it silent. Any reward-weight change must re-measure with
`zero_action_baseline.py` and re-derive both.

**A probe must not be able to masquerade as a verdict.** The `--filter-actions`
report scores a *modified* policy. It gets its own filenames, prints its warning
above the verdict, and never writes `stance_panel_selected.csv` — the
per-episode evidence a bundle is certified from. All three are enforced
structurally rather than by caller discipline.

---

## 9. Recommendations

1. **Merge #492** (revert `leg_home_pose_weight` to 0.5 and the two
   statue-derived constants with it) and re-run unchanged. That is the
   configuration that produced duty 0.0000.
2. **Then run it again at a different seed.** One pass and two failures at a
   single seed cannot tell you whether the stage is solved or lucky. If two seeds
   pass, #490 was the whole problem. If one bounces, the escape is a coin flip
   and the work is making it reliable — which no reward term addresses.
3. **Do not add or reweight reward terms** on this evidence (§4).
4. **Do not raise `action_jerk_weight`** — the high-frequency content is
   load-bearing in both policies (§5).
5. **Leave saturation and near-Nyquist dependence in #491** as sim-to-real work.
6. **`frame_skip` is a later experiment,** not a fix (§6).

---

## 10. Instrumentation added along the way

All merged and running on every SB3 run, with no opt-in:

- `gate_progress.npz` — the deterministic gate criteria per evaluation, plus
  `action_dc_rms` / `action_ac_rms` / `action_delta` / `action_jerk` /
  `action_freq_hz`, the per-actuator `action_dc_per_actuator` and
  `action_ac_rms_per_actuator` matrices, and every reward term as `term_*`.
- `action_jerk` in `diagnostics.npz` — the environment always emitted it and
  `INFO_KEYS` always dropped it.
- `stance_gate_report.py --filter-actions HZ` and the automatic probe, with
  per-actuator DC/AC printed under real joint names.
- `curriculum.baseline_watch` — reports every evaluation against the run's own
  zero-action baseline, advisory only.

The frequency estimate is `jerk/delta = (2 sin πfΔt)²`, which is blind to any
constant offset — the DC/AC split is the point, because a pooled standard
deviation over actuators and time cannot separate "sitting in the wrong place"
from "shaking", and those have different causes and different fixes.

---

## Addendum, 2026-08-06 — the tremor is feedback, not bandwidth

Appended, not rewritten. This settles the question §5 left open.

### What §5 could not answer

The low-pass probe proved the tremor is load-bearing, but a low-passed policy
**still responds to what it sees** — just slowly. So a fall under the filter
shows the policy needs *bandwidth*; it cannot show whether it needs *feedback*
at all. Two readings survived, with opposite fixes:

- the pose the policy holds needs continuous active stabilisation, or
- the pose is holdable by a constant, and the tremor is waste the action
  penalties failed to suppress.

### The experiment

Replace the commanded action with the constant the policy commands **on
average** — its own post-settle per-actuator DC — and keep everything else
identical. Feedback is cut outright rather than attenuated. Two controls
bracket the measurement.

`stance_gate_report.py --hold-constant`, on the **passing** checkpoint
(`20260805_011234`), 10 episodes per variant:

| variant | handoff | ramp | ep length | full-horizon | reward | terminations |
|---|---|---|---|---|---|---|
| policy (control) | never | — | **1000.0** | 1.0000 | 3006.9 | truncated 10 |
| `hold_after_settle` | 200 | 0 | 348.9 | 0.0000 | 908.8 | tail_contact 9, fallen 1 |
| `hold_after_settle_ramped` | 200 | 50 | 330.2 | 0.0000 | 795.0 | fallen 8, tail_contact 2 |
| `hold_from_reset` | 0 | 50 | 133.3 | 0.0000 | 284.8 | tail_contact 10 |
| `hold_zero` (statue control) | 0 | 0 | **1000.0** | 1.0000 | 3271.0 | truncated 10 |

Both controls behave exactly as required — the unmodified policy reaches the
horizon, and the statue reproduces its known 3271.8 within panel noise — so the
harness is measuring the pose and not itself.

### Result

**The pose the policy holds requires continuous feedback.** Every held variant
collapses, and the ramped variant collapses too, which removes the obvious
objection: it is not a step transient at the handoff. Ramping in over 50 steps
made it *slightly worse* and shifted the termination reason from `tail_contact`
to `fallen`, which is what a slow topple looks like rather than a jolt.

### Why this matters more than it first appears

The statue stands at the home keyframe for the full horizon on a constant
command. The policy stands 0.765 rms **away** from home — 12 of 21 actuators
pinned at exactly ±1.000 with AC 0.000 — and *that* pose cannot be held by a
constant. So the tremor is not a property of the task. It is the cost of
standing where the policy chose to stand.

That reframes the whole question. It is not "why does the policy shake?" but
**"why did it walk off a pose that is free to hold, onto one that costs 267
reward points *and* requires continuous stabilisation?"** The answer is not in
the action penalties, and raising them is now positively contraindicated —
§4 already showed they are firing and losing, and this shows the thing they
would suppress is the only thing holding the animal up.

The candidate answer is that **nothing in stage 1 constrains where 13 of the 21
actuators sit.** `leg_home_pose` covers 8 joints carrying 1.2% of the offset
(§3). The tail, neck, head and toes are unconstrained, they saturate, and the
sagittal chain — hip pitch, knee, ankle, the joints with headroom left — pays
for it at 22.6 Hz.

### It does not, by itself, argue for a reward change

Two things must not be read into this:

1. **It says nothing about the bounce.** The bounce is still 450 points worse
   under the same reward (§4), still an optimisation failure, and still not
   addressable by reweighting.
2. **It does not show the reward is wrong, only that it is silent.** A term
   constraining the unconstrained joints is a *hypothesis*, and the last time a
   pose term was reweighted on a plausible mechanism (#490) it did exactly what
   it was asked to do and the stage failed anyway. Measure the mechanism before
   shaping for it.

The measurement that would test it is cheap and does not need a training run:
hold the constant pose but **release only the saturated joints back to home**,
and see whether the rest becomes holdable. If it does, the saturated pose is
what creates the stabilisation load.

> **Run, same day — and the answer is no.** See §Addendum 2 below. Releasing
> all twelve saturated actuators does *not* rescue the pose, and the tail, the
> single most conspicuous thing in the DC table, turns out to be a bystander.
> The candidate answer above is **refuted**; the conspicuousness of saturation
> is exactly why it needed testing rather than adopting.

### And a separate point about the task

Stage 1 has **no in-episode disturbance** — the only perturbation is joint-angle
noise at reset (`reset_noise_scale 0.05`); there is no external force anywhere
in the environment. So a policy that learns active postural correction earns
nothing over one that stands still, and "actively stand and correct posture"
is not underweighted in this reward, it is absent from the task. That is the
gap `STAGE1_SPLIT_PLAN.md` proposes 1a/1b for, and no reweighting substitutes
for it.

---

## Addendum 2, 2026-08-06 — which joints make the pose unholdable

Appended, not rewritten. This **refutes** the candidate answer in Addendum 1.

### The experiment

Addendum 1 established that the policy's pose cannot be held by a constant, and
proposed that the twelve saturated actuators were why. Testing that needs two
questions per group, not one, because either alone misleads:

- `release_G` — hold everything **except** G. Is G **necessary**? Does removing
  it rescue a pose that otherwise falls?
- `only_G` — hold **only** G. Is G **sufficient**? Does it alone break the
  statue, which is known to stand?

All variants are commanded **from reset**, so the endpoints are the two known
measurements: releasing everything is the statue (stands 1000), releasing
nothing is Addendum 1's `hold_from_reset` (falls at ~133).

`stance_gate_report.py --hold-release-ablation`, passing checkpoint, 8 episodes:

| variant | released | ep length | full-horizon | reward | terminations |
|---|---|---|---|---|---|
| policy (control) | — | **1000.0** | 1.0000 | 3010.1 | truncated 8 |
| `hold_all` | 0 | 128.6 | 0.0000 | 268.8 | tail_contact 8 |
| `release_saturated` | 12 | 224.6 | 0.0000 | 460.1 | fallen 5, tail_contact 3 |
| `only_saturated` | 9 | 140.5 | 0.0000 | 330.6 | tail_contact 8 |
| `release_tail` | 4 | 116.6 | 0.0000 | 250.1 | tail_contact 8 |
| **`only_tail`** | 17 | **1000.0** | **1.0000** | **3254.0** | truncated 8 |
| `release_toes` | 6 | 198.5 | 0.0000 | 397.3 | fallen 6, tail_contact 2 |
| **`only_toes`** | 15 | **125.5** | 0.0000 | 289.2 | tail_contact 8 |
| `release_head_neck` | 3 | 123.2 | 0.0000 | 256.4 | tail_contact 8 |
| **`only_head_neck`** | 18 | **1000.0** | **1.0000** | **3286.1** | truncated 8 |
| `release_hip_rolls` | 2 | 129.4 | 0.0000 | 270.4 | tail_contact 8 |
| `only_hip_rolls` | 19 | 989.8 | 0.8750 | 3190.5 | truncated 7, fallen 1 |
| `hold_zero` (statue control) | 21 | **1000.0** | 1.0000 | 3274.4 | truncated 8 |

Controls behave, and `hold_all` at 128.6 reproduces Addendum 1's 133.3 — the
same experiment reached twice by different routes.

### Result

**The toes are sufficient on their own.** Holding the six toe joints at their
commanded DC with every other actuator returned to home reproduces the full
failure almost exactly — **125.5 steps against `hold_all`'s 128.6, with
`tail_contact` 8 of 8 in both.** Six of 21 actuators account for essentially the
whole effect.

**The tail is a bystander.** All four tail joints held at `+1.000`, everything
else at home, stands the **full horizon at reward 3254.0** — within 0.6% of the
statue's 3274.4. Head and neck likewise: 1000 steps, **3286.1**, which is
*above* the statue. The two groups whose DC looks most alarming do nothing.

**No group is necessary.** `release_saturated` improves matters (128.6 → 224.6)
but still falls; so does `release_toes` (198.5). The pose is **over-determined**
— more than one subset breaks it independently, so removing any single group
leaves another still able to. "Which joint is the cause" is the wrong question.
"Where is the effect concentrated" has an answer, and it is the toes.

### What this refutes

Addendum 1 proposed that **saturation** creates the stabilisation load, on the
reasoning that twelve actuators pinned at `±1.000` with no headroom must be
doing something. Measured: releasing all twelve does not rescue the pose, and
the largest saturated group — the tail — is provably inert. `only_saturated`
falls (140.5), but it contains five of the six toes, so its sufficiency is
plausibly the toes' and nothing else's.

Saturation is *conspicuous*, not *causal*. It was worth testing precisely
because it looked obvious, and the same instinct that made #490 attractive
would have made a tail- or saturation-targeted reward term attractive here.

### Why the toes are a plausible mechanism

They are the ground contact. Five of six sit at `±1.000` — driven to their
stops — which changes the foot's contact geometry and therefore the support
polygon the whole body balances on. Unlike the tail, whose effect on a standing
animal is a static moment the position servos absorb, a toe at its limit
changes *where and how the animal touches the floor*. That is a mechanical
effect on the plant, not a reward-shaping one, and it is the strongest evidence
so far for the "something is off with the mechanics" reading.

The termination reason agrees: `only_toes` and `hold_all` both end in
`tail_contact` 8 of 8, while the partial rescues (`release_saturated`,
`release_toes`) shift toward `fallen` — a different failure, reached later.

### What this does NOT establish

- **Not a reason to add a toe reward term.** Sufficiency says the toe pose
  breaks a *statue*; it does not say the trained policy would stand if the toes
  were constrained, because the rest of the pose falls on its own too
  (`release_toes` = 198.5).
- **Nothing about the bounce.** §4 is untouched: still 450 points worse, still
  an optimisation failure.
- **Nothing about `ctrlrange`.** Whether `±1.000` on a toe is a large joint
  angle or a small one has not been measured. Read the actual ranges out of
  `trex.xml` before concluding the toes are over-driven rather than merely
  commanded to their limits.

### Method note: the control caught a broken experiment

The first run of this ablation handed off at `settle_steps` rather than from
reset, so each variant snapped a subset of joints from `±1.000` to `0` in one
step midway through the episode. The transient dominated: all thirteen variants
landed within **311–349 steps** of each other, indistinguishable from
`hold_all`. The tell was the `hold_zero` row — supposedly the statue — falling
at **323 steps** when it is known to stand for 1000.

Both endpoints on the same side of the answer means the path has no signal, and
without the statue control the flat table would have read as "no group matters",
which is a conclusion and a wrong one. The control cost 8 episodes and saved a
false negative. `TestReleaseAblationVariants` now pins the from-reset
requirement.

---

## Addendum 3, 2026-08-06 — why the ablation was needed at all

Appended. This answers the prerequisite Addendum 2 flagged as unmeasured, and
the answer means the ablation should not have been necessary to find its result.

### `|action| = 1` is not one physical event

`_scale_action` maps `[-1, 1]` linearly onto each actuator's own `ctrlrange`.
Those ranges differ by a factor of six across the T-Rex:

| group | `action = +1` | `action = −1` | deflection from home | ablation verdict |
|---|---|---|---|---|
| **toes** | **+50°** | **−25°** | **±37.5°** | **sufficient** |
| hip_roll | +25° | −25° | ±25° | inconclusive (0.875) |
| head / neck | +20…40° | −20…30° | ~±25° | bystander |
| **tail** | **+12° pitch, +8° yaw** | **−12° / −8°** | **±8–12°** | **bystander** |

The ablation's result falls straight out of that column. The tail is inert
because saturating a tail joint moves it **eight to twelve degrees**. The toes
carry the effect because saturating a toe moves it **37.5°**, and adjacent toes
on the same foot are driven to opposite ends — on the right foot d2 and d4 at
+50° while d3 sits at −25°, a **75° spread across one foot**. The foot is
splayed into a twist, asymmetrically between the two feet.

So the answer to Addendum 2's open question is **yes, very large** — and the
"toes are the ground contact" mechanism survives.

### The metric was the problem

"Ten to twelve of 21 actuators pinned at `|action| ≥ 0.99`" — quoted in §7, in
KNOWN_ISSUES and throughout this investigation — pools an 8° tail deflection
with a 37.5° toe deflection and counts them as the same event. Sorted by `|dc|`,
the per-actuator table put twelve joints at exactly `±1.000` at the top with no
way to tell them apart, and the most conspicuous of them was the inert one.

**This is the same error the DC/AC split was introduced to fix, one level
down.** §10 says a pooled standard deviation "cannot separate sitting in the
wrong place from shaking". A pooled *normalised* offset cannot separate moving
8° from moving 37.5°, and the fix is the same shape: report the physical
quantity next to the normalised one.

`stance_gate_report.py` now emits `dc_deg`, `ac_rms_deg`, `range_deg` and
`zero_offset_deg` per actuator, and **orders the table by degrees**. On the
passing checkpoint that puts the five saturated toes at the top at ±37.5° and
drops `tail_1_yaw` (±8°) out of the printed twelve entirely — the ablation's
conclusion, readable directly off the report.

**Lesson: a normalised summary is only as good as the assumption that its units
mean the same thing everywhere.** Twice now that assumption has been false, and
both times it hid the answer rather than merely blurring it.

### And `action = 0` is not exactly the home keyframe

`action = 0` is the `ctrlrange` **midpoint**, which equals home only where the
range was authored centred on it. For four of 21 actuators it is not:

| actuator | `action = 0` minus home |
|---|---|
| `r_ankle`, `l_ankle` | **+5.5°** |
| `head_pitch`, `neck_pitch` | **+5.0°** |

Everything else is exact to floating point. The magnitude is small and the
statue still stands 40 of 40, so nothing measured here is invalidated. But
"`action = 0` **is** the home keyframe" is the phrasing the residual mapping is
named for, and both statue-derived constants and every DC interpretation in
this document rest on it. It is inexact for those four, and the report now says
so rather than leaving it to be rediscovered.
