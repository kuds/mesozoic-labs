# T-Rex Stage 1 with the narrowed home-pose Gaussian: the knee-lock run

**Date:** 2026-08-10
**Run:** `20260809_155206` (seed 42, 10,002,432 steps in 10h 29m, PPO/SB3, NVIDIA L4) — FAIL at the stance gate
**Plant:** physics r7 / policy interface r10 (passive toes, action_dim 15) — unchanged from the previous run
**Config delta vs previous run:** `leg_home_pose_tolerance` 0.20 → 0.10 rad, rails re-derived (#502)
**PRs:** #501 (regrouped ablation + previous postmortem), #502 (tolerance + rails)
**Companion:** `TREX_STAGE1_PASSIVE_TOES_RUN_2026_08.md` (the previous run, hereafter "run 1")

Point-in-time record. Not rewritten — corrections are appended.

---

## 1. Summary

Second stage-1 run on the passive-toes plant, and the controlled test of the
previous postmortem's recommendation: narrow the home-pose Gaussian from 0.20
to 0.10 rad so the statue basin out-pays every chattering alternative. The
gate verdict on the robust-best checkpoint:

| criterion | required | final | verdict |
|---|---|---|---|
| full-horizon fraction | ≥ 0.95 | **0.9750** (39/40 truncated, 1 fallen) | pass |
| mean reward | ≥ 1940 | **1999.9 ± 133.0** | pass |
| mean unsupported duty | ≤ 0.02 | **0.3987** | **FAIL** |
| unsupported duty UCB | ≤ 0.02 | **0.4029** | **FAIL** |

The headline number barely moved (run 1: 0.4095), but the run underneath it is
a different animal, and the probes finally show the failure's shape clearly
enough to name the exploit. The policy converged to a **deep knee-locked
crouch**: both knees commanded to −50.0°, hard against the flexion stop, with
±1.1–1.5° of residual motion — mechanically locked — while the ankles pump
±28–33° around neutral at ~16.8 Hz and the hip pitches row at +37–40° with
±19–23° swings. Anyone watching the replay sees exactly what the numbers say:
a dinosaur standing with locked knees, flexing its feet up and down.

Three findings define this run:

1. **The pose is not statically stable, and the foot-flexing is what holds it
   up.** The new constant-hold probe (section 4) replaces the policy's output
   with its own post-settle mean action: every held variant falls within ~3
   seconds, 9 of 10 times backward onto the tail. The tremor is active
   stabilisation of a backward-leaning pose, not residual noise.
2. **The narrowed Gaussian did not pull the policy home — it never felt it.**
   Home-pose error rose monotonically from 0.20 to 0.545 rad across the entire
   10M steps (section 5). Outside ~2 tolerance widths the Gaussian's gradient
   is numerically zero; once early exploration wandered, there was no
   restoring force at all. The recommendation is falsified as a basin-capture
   mechanism.
3. **The policy discovered saturation parking.** A saturated command cannot
   oscillate, so the smoothness and jerk penalties price a joint pinned at its
   hard stop as perfectly quiet. One by one, formerly-chattering channels went
   silent by being driven to a limit: on the gate panel, five of fifteen
   actuators sit at or within 2% of a stop (both knees, head_pitch,
   tail_2_pitch, tail_1_yaw). Saturation became the cheapest way to be smooth.

---

## 2. Trajectory

Gate checks every 50k steps, 40-episode panels, seeds 3042–3081. Three acts:

| phase | steps | what happened |
|---|---|---|
| statue flash | 0–100k | init lands near home: **2520.7 ± 1116.5 at 50k — the best eval either run ever recorded** — with near-perfect duty. High variance (it falls often), but when it stands, it stands the statue's way |
| trough + rebuild | 100k–2M | exploration leaves home and never returns; duty UCB peaks at 0.756 (1.9M), worse than run 1's rebuild |
| crouch consolidation | 2M–8.6M | steady, monotone improvement on every support metric: duty UCB 0.756 → 0.40, bilateral support 0.03 → 0.55, foot-load imbalance 0.99 → 0.50, per-foot contact force 220 → 330 N; full-horizon pinned at ~1.0 for 7M straight steps; reward crosses the 1940 rail at ~7.55M |
| Goodhart collapse | 8.6M–10M | duty UCB keeps plunging (0.40 → **0.248**) but survival is spent to buy it: full-horizon crashes as low as 0.15, episode length swings 560–1000, eval reward whipsaws 1070–2020, mean tilt climbs from ~4° to ~9° |

Notable singletons:

- **Duty fell straight through the 7M entropy zero-crossing without a
  kink** — the clean confirmation of run 1's falsification. The decline is
  driven by the reward landscape, not exploration noise. It also
  *accelerated* late (~0.03/M through 7–8M, ~0.087/M after 8.25M), which is
  the collapse announcing itself: the optimizer found that leaning further
  back unloads the feet faster than balancing better does.
- **The robust-best selector earned its keep.** The final 10M policy
  evaluates at 1869.6 ± 607.0 with mean episode length 851; the selected
  checkpoint (from the ~8.3M plateau) evaluates at 2012.1 ± 57.7 with every
  episode full-length. The last 1.5M steps of "progress" were not robust, and
  the selector correctly refused to ship them.
- Mean tilt angle tells the lean story in one curve: ~15° at init, bottoming
  at ~3.6° mid-run (4.5–5.5M), then climbing steadily back to ~9° by 10M as
  the policy traded uprightness for foot-unloading, right up against the
  tail-contact termination.

---

## 3. Anatomy of the stance

From the gate report's commanded-action table (post-settle, robust-best
checkpoint, 40 episodes; DC 0.670 rms, AC 0.497 rms, ~16.8 Hz effective):

| actuator | command | swing | reading |
|---|---|---|---|
| l_knee / r_knee | −50.0° / −49.9° (of ±50°) | ±1.1° / ±1.5° | locked against the flexion stop |
| l_ankle / r_ankle | −7.7° / −7.9° | **±33.1° / ±27.6°** | the stabiliser: feet pumping |
| l_hip_pitch / r_hip_pitch | +37.2° / +39.7° | ±23.1° / ±19.4° | weight-shifting, second stabiliser |
| neck_pitch | −14.7° | ±20.7° | third stabiliser |
| head_pitch | −25.0° (stop) | ±0.9° | parked |
| tail_2_pitch | +12.0° (stop) | ±0.5° | parked |
| tail_1_yaw | −7.7° (near stop) | ±1.1° | parked |
| tail_3_pitch | −8.6° | ±8.0° | active counterweight |

The division of labour is total: three joint groups (ankles, hip pitches,
neck pitch) carry all of the stabilisation bandwidth, everything else is
parked — mostly at hard stops. The unsupported duty this stance scores (0.40)
is the pumping itself: at ~16.8 Hz the feet break full contact inside a large
fraction of control steps, and the substep-honest metric (#499) counts every
one of them.

Reward accounting per episode (gate panel): alive 761.7, height 592.4,
head_clearance 348.2, bilateral_support 317.2, foot_load_balance −280.3,
leg_home_pose 174.4 (of ~500 available), neck_posture 171.4. The support
terms recovered most of run 1's deficit; what is left on the table is the
abandoned home pose and the load-balance penalty of the crouch.

---

## 4. The probes: the pose needs its tremor, more than ever

**Constant-hold probe** (new this run): replace the policy's action with a
constant partway through the episode — its own post-settle mean — and see if
the pose survives without feedback.

| variant | handoff | ramp | ep length | full-horizon | terminations |
|---|---|---|---|---|---|
| policy (control) | never | — | 1000.0 | 1.00 | truncated 10 |
| hold_after_settle | 200 | 0 | 318.0 | 0.00 | **tail_contact 9**, nosedive 1 |
| hold_after_settle_ramped | 200 | 50 | 269.3 | 0.00 | tail_contact 7, nosedive 2, fallen 1 |
| hold_from_reset | 0 | 50 | 96.6 | 0.00 | nosedive 10 |
| hold_zero (statue control) | 0 | 0 | 1000.0 | 1.00 | truncated 10 |

No held variant reaches the horizon; the statue control does. The chosen pose
*requires* continuous feedback, and its failure direction is specific:
backward, onto the tail. The knee-locked crouch sits behind its feet, and the
ankle pumping is the only thing preventing the animal from sitting down.

**Filter probe** — and this is the run's most important negative result. The
policy's actions low-passed at eval time:

| cutoff | run 2 ep length (full-horizon) | run 1 (for comparison) |
|---|---|---|
| 35 Hz | 790 (0.40) | survived (1.00) |
| 30 Hz | 633 (0.00) | survived |
| 20 Hz | 295 (0.00) | survived |
| 10 Hz | 161 (0.00) | died |
| 5 Hz | 71 (0.00) | died |

Run 1's stance tolerated everything down to 20 Hz. Run 2's stance dies at
**every** cutoff tested, with tail-contact and falls splitting the
terminations. Despite lower AC amplitude (0.50 vs 0.61 rms) and a lower
effective frequency (16.8 vs 18.7 Hz), this policy is *more* dependent on
high-bandwidth feedback than its predecessor — the amplitude metrics measure
the tremor's size, not its necessity. The balance task itself needs ~1.1–1.4
Hz. The gap between what the task needs and what the policy uses is the
exploit's whole surface area.

---

## 5. The home-pose Gaussian: falsification, completed

Run 1's postmortem priced the crouch-vs-home tradeoff and recommended
narrowing the tolerance so the statue basin dominates. Run 2 is the verdict:

- Home-pose error rose **monotonically** from 0.20 rad (init) to 0.545 rad
  (10M). There is no episode of return, no plateau, no fight. The policy
  left home in the first 100k steps and drifted further for the remaining
  9.9M while the term flatlined at ~174/500.
- The mechanism was named in the previous postmortem's section 6 and operated
  exactly as written: a Gaussian's gradient at 5 tolerance widths is zero to
  machine precision. Narrowing the tolerance made the basin *richer* and its
  gravitational reach *shorter* — the opposite of capture. The term sees the
  pose; it just cannot pull anything toward it.
- The 50k statue flash (2520.7, the best eval either run recorded) proves the
  basin pays exactly as designed — for a policy already inside it. The
  intervention failed not on the economics but on the reach.

Consequence for the fix list: any home-pose term that must *attract* — not
merely *reward* — needs a gradient that survives distance: a long-tailed
component (e.g. Cauchy), a tolerance annealed from wide to narrow, or an
explicit curriculum that starts episodes inside the basin.

The A/B against run 1 must also be stated honestly: at every matched step
past 6M, run 2 leads on duty, chatter amplitude, and action frequency, and
its training frontier (0.248) went far below anything run 1 touched. But with
one seed per condition and a qualitatively different attractor found, none of
that is causally attributable to the tolerance change. Seed replicates before
crediting it.

---

## 6. The actuator question, considered and declined

An obvious response to a policy that stabilises with 17 Hz foot-pumping is to
ask whether the body is missing an actuator — something in the tail or torso
that would let it balance "properly." Considered, and declined, for three
reasons:

1. **The statue is the existence proof that nothing is missing.** Constant
   zero action, springs only: full horizon, 0.998 bilateral support, 3240
   reward. Stage-1 balance requires *no* actuation at all from this
   morphology. A missing-DOF deficit would present as "no statically viable
   pose exists"; we observe the opposite — the policy declines a statically
   viable pose in favour of one it must fight.
2. **The high-frequency movement is a chosen pose's life-support, not a
   control-authority gap.** Run 1 chattered the toes; after #500 removed
   them, run 2 chattered the ankles. The channel migrates to wherever
   bandwidth is free; adding actuators adds channels. The toe motors were
   this exact experiment run in reverse, and removing them was an
   improvement.
3. **This run pinned the actuators the reward prices worst.** By 10M, six of
   fifteen actuators sat at stops — head_pitch, both knees, and the entire
   tail yaw/pitch chain — while the top remaining chatter channels were
   neck_yaw (AC 0.89) and the hip rolls. The audit of what stage 1 actually
   prices explains the sorting: the tail term penalises tail-*tip angular
   velocity* only, so a tail parked motionless at a hard stop is *optimal*
   for it; the neck term covers neck_yaw and head_pitch but at weight 0.20
   with a 0.35 rad tolerance — as wide as the entire neck_yaw range — so
   yaw chatter is nearly free; and the legs sit outside the narrowed
   Gaussian's reach. A new tail or torso actuator would inherit one of
   these camps — parked at a limit or drafted as the new pump. Authority
   the reward cannot see is authority the optimizer will spend against us.

One genuine candidate is deferred, not rejected: **ankle roll** (lateral
ankle strategy) for stage-2 locomotion, where lateral balance currently
routes entirely through hip roll. That is a stage-2 question, to be answered
by stage-2 evidence, and any addition should arrive with its reward terms
already in place.

---

## 7. Recommendations → the fix plan

In priority order (items 1–3 are being implemented now):

1. **Training-time action low-pass filter (~10 Hz).** The structural fix both
   runs point at. Filter the commanded action inside the environment during
   training so high-bandwidth stabilisation is not merely priced but
   *unavailable*; poses that need 17 Hz feedback stop being attractors, and
   the statically stable basin becomes the cheap one. The eval-side probe
   already demonstrates the mechanism; this promotes it into the plant's
   training interface. The task needs ~1.1–1.4 Hz; a ~10 Hz cutoff leaves
   almost a decade of margin while sitting well below both runs' exploit
   bands (16.8, 18.7 Hz).
2. **Price saturation.** A small penalty on the fraction of actuators pinned
   near |action| = 1 removes the free-smoothness exploit that parked five
   channels at their stops.
3. **Price tail pose, not just tail-tip motion.** The tail joints are the
   one actuator group with no positional term at all — the existing tail
   term (tip angular velocity) actively prefers a tail frozen at its stops.
   A soft home term over the four tail joints closes the loophole the whole
   tail chain disappeared into.
4. **Give the home-pose term long-range gradient** (Cauchy tail or annealed
   tolerance), so the basin can attract as well as pay. Without this, items
   1–3 prevent the exploit but nothing actively routes the policy home.
5. **Seed replicates** for any future A/B; this run demonstrates how easily
   one seed's attractor discovery can masquerade as an intervention effect.
6. Minor: the foot-contact figure renders an empty "diagonal pair" panel for
   bipeds; skip it for two-footed species.

Items 2–4 change the reward function, which moves the statue baseline; the
rails (`min_avg_reward`, `collapse_peak_floor_reference`) must be re-derived
against the new terms in the same PR, per the provenance rules added in #501.

One caution for downstream stages: `robust_best_model.zip` from this run is
the knee-locked crouch. If stage 2 ever seeds from a stage-1 checkpoint,
this one imports a stance that cannot survive without ~17 Hz feedback — check
the filter probe of any checkpoint before promoting it.
