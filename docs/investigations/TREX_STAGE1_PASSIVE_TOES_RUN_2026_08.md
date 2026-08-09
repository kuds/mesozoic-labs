# T-Rex Stage 1 on the passive-toes plant: the first honest 10M steps

**Date:** 2026-08-09
**Run:** `20260808_230537` (seed 42, 10M steps, PPO/SB3, NVIDIA L4) — FAIL at the stance gate
**Plant:** physics r7 / policy interface r10 (passive toes, action_dim 15)
**PRs:** #499 (substep contact aggregation), #500 (passive-toes plant revision), #501 (regrouped ablation)

Point-in-time record. Not rewritten — corrections are appended.

---

## 1. Summary

First stage-1 run on the passive-toes plant (#500), and the first run whose
duty numbers are measured at every physics substep (#499) rather than at the
control boundary. The headline is a split verdict:

| criterion | required | final | verdict |
|---|---|---|---|
| full-horizon fraction | ≥ 0.95 | **1.0000** (locked from 4.5M to the end) | pass |
| mean reward | ≥ 1950 | **2213.7 ± 18.4** (crossed the rail at ~6.3M) | pass |
| mean unsupported duty | ≤ 0.02 | **0.4084** | **FAIL** |
| unsupported duty UCB | ≤ 0.02 | **0.4095** | **FAIL** |

By every survival measure this is the best stage-1 run the repository has
produced: no collapse, no regression, 40/40 full-horizon panels for the entire
back half, reward within 68% of the statue's 3270.3 and still climbing at the
buzzer. The gate still refuses it, for the one thing it now measures honestly:
the policy stands while unloading both feet at some instant inside **41% of
its control steps** — a high-frequency foot chatter at 18.7 Hz that the old
boundary-sampled metric could have scored as near-continuous support.

Three hypotheses from the previous investigation cycle got definitive answers:

1. **The toe mechanism is gone.** No probe implicates anything the r6 toes
   used to do; the failure moved, it did not survive by another name.
2. **The entropy hypothesis is falsified.** `ent_coef` reached exactly zero at
   7M and nothing broke: tremor amplitude stayed at AC ≈ 0.61–0.64, and duty
   kept its gentle, *decelerating* decline (0.487 at 7.1M → 0.406 at 10M) with
   no cliff. The chatter is a learned strategy, stable with zero exploration
   noise — not noise the policy was forced to carry.
3. **The stabilisation load is localizable after all** — but only with the
   right ablation groups. It lives in the **left knee and left ankle**
   (section 5), in a pose parked just inside `leg_home_pose_tolerance`'s
   penalty-free band (section 6).

---

## 2. Trajectory

Gate checks every 50k steps, 40-episode panels, seeds 3042–3081.

| phase | steps | what happened |
|---|---|---|
| init spike | 0–100k | starts near home (≈ statue): 1454 reward, 28% full-horizon at 50k |
| trough | 100k–800k | exploration pushes off home; bottom 47.8 at 350k, 0% full-horizon |
| recovery | 800k–1.55M | survival breakthrough; first ≥95% full-horizon panel at 1.55M |
| survival lock | 1.55M–4.5M | full-horizon pins to ~100% (one transient dip 3.6–3.7M, self-recovered) |
| reward climb | 4.5M–10M | 1570 → 2213, monotone within noise; duty UCB 0.65 → 0.41 |

Notable singletons:

- The collapse detector armed correctly past the 1M warmup and never fired;
  the run's peak first exceeded the 1471.6 floor (0.45 × 3270.3) around 4.45M.
- Bilateral support duty compounded steadily all run: 0.03 at 1.55M → 0.59 at
  10M, still rising at the end.
- The commanded offset (DC ≈ 0.49 rms) never returned home, and the tremor
  amplitude (AC ≈ 0.62 rms, ~18.7 Hz effective) never fell — the reward gains
  of the back half came from everything *around* the chatter.

**The entropy zero-crossing at 7M is the load-bearing observation.** Duty UCB
by segment: 0.646 → 0.626 (4.6–5.05M, ent ≈ 0.0017), 0.622 → 0.564
(5.05–5.65M, the fastest stretch), 0.487 → 0.406 (7.1–10M, ent = 0.000,
~0.027/M and decelerating). If the tremor were noise-sustained, the final 3M
would have shown a qualitative break. It showed the same crawl.

---

## 3. Where the reward gap lives

Final panel decomposition (statue reference 3270.3; gap 1057):

| term | policy | note |
|---|---|---|
| head_clearance | +350.0 | at ceiling (0.35 × 1000) |
| height | +596.6 | effectively at ceiling (0.6 × 1000) |
| neck_posture | +169.3 | near ceiling |
| leg_home_pose | +379.2 | inside tolerance — see section 6 |
| alive | +785.4 | of 1000; the missing 215 is the support-conditioned half |
| bilateral_support | +342.5 | of 600 |
| foot_load_balance | **−271.5** | largest single penalty; airborne penalty dominates |
| smoothness + jerk + energy | −123.3 | modest — the chatter is cheap under the action penalties |

Essentially the entire gap to the statue sits in the three support-linked
terms (≈ 1030 of 1057). The policy solved posture, height, heading and
survival completely, and pays precisely and only the price of rattling its
feet. The action-path penalties price that rattle at ~123 points — far less
than the ~1030 the support terms forfeit — so the pricing is not the reason
it chatters (section 4 says the tremor is load-bearing, not cheapness).

---

## 4. Probe results

All probes on `robust_best_model.zip` + its VecNormalize stats.

**Hold-constant: the pose requires feedback.** Freezing the commanded mean
falls from every handoff (141–331 steps, `tail_contact` dominant) while the
statue control stands at 3269.2. Same verdict as r6: the tremor is active
stabilisation; penalising it harder removes the only thing keeping the animal
up.

**Filter sweep: the bandwidth story flipped.** The r6 passing policy fell at
*every* cutoff from 5 to 35 Hz. This policy:

| cutoff | full-horizon | reward |
|---|---|---|
| 5 Hz | 0% (542 steps) | 837 |
| 10 Hz | 10% (704) | 1126 |
| 20 Hz | **100%** | 2029 |
| 30 Hz | 100% | 2094 |
| 35 Hz | 100% | 2110 |

Its load-bearing content is confined to the band just under 20 Hz — the
18.7 Hz chatter fundamental — and it loses only ~180 reward when filtered at
20 Hz. This is one octave away from tolerating a sim-to-real-grade action
filter, where r6 policies needed unbounded bandwidth.

**Impulse probe: first partial recovery ever recorded.** At 0.5 m/s lateral
−y the policy recovered to full horizon in 6/8 episodes (statue: 0/8). Every
stronger shove flattens policy and statue alike (+116-step margin ≈ noise).
The 1a/1b split argument is unchanged: nothing in stage 1 pays for
correction, so recovery must enter the task, not the reward.

---

## 5. The release ablation, regrouped

Over the r6-era groups (saturated trio, tail, head/neck, hip rolls) the
ablation returned **no group necessary, none sufficient** — every release
still fell, every group held alone still stood near statue reward. The old
grouping was built for the toe question and never tested the distal leg
chain as a unit; #501 added `knees_ankles`, `left_leg`, `right_leg`
(per-side because the tremor is strongly asymmetric). Result (8 eps/variant):

| variant | ep length | full-horizon | reading |
|---|---|---|---|
| only_knees_ankles | 118.6 | 0% | **sufficient** — alone breaks the statue |
| only_left_leg | 164.6 | 0% | **sufficient** |
| only_right_leg | 903.5 | 87.5% | nearly exonerated |
| only_hip_rolls | 870.9 | 75% | weakly implicated |
| only_saturated / tail / head_neck | 1000 | 100% at ~3255–3279 | bystanders |
| release_knees_ankles | 794.0 | 37.5% | the only release that rescues anything |

Intersection: **the left knee (+9.2° DC, ±26.7° AC) and left ankle (+4.1° DC,
±39.2° AC)** — also the two largest degree entries in the gate report. The
right leg's offsets (+0.9°, +1.2°) are benign. The pose is over-determined
across groupings that *include* these two joints and innocent everywhere
else. The saturated trio (head_pitch −25.0°, tail_3_pitch +12.0°, tail_1_yaw
−8.0°, all with zero AC) is conspicuous and fully exonerated — the r6 lesson
about saturation being visible rather than causal, repeated on a new plant.

---

## 6. The tolerance dead zone

`leg_home_pose_tolerance = 0.2` defines a penalty-free band around the home
pose for the eight governed leg joints. The checkpoint's statue-breaking
offsets sit at or just inside it: l_knee DC 0.185 normalised (0.16 rad =
9.2°), l_ankle 0.081, hip rolls ≈ 0.18. The policy parked its pose precisely
where the pose penalty cannot see it — `leg_home_pose` still pays +379 of its
+500 ceiling — and then pays the chatter tax forever to stabilise a pose the
passive plant cannot hold.

This is the same *shape* of finding as the r6 toes (a statue-breaking pose
living where no term prices it), one level up: the toes were ungoverned by
construction; the left knee/ankle are governed but hiding inside the
tolerance. The band was sized to keep reset-noise settling penalty-free, not
to admit standing 9° off home on one knee.

---

## 7. Recommendations, ranked

1. **Tighten or reshape `leg_home_pose`'s dead zone.** Mechanism-backed by
   sections 5–6, one config line, and statue-invariant: the statue sits at
   zero offset, so `min_avg_reward`/`collapse_peak_floor_reference` need no
   re-derivation (the freshness pin from #500 stays honest). Candidate:
   tolerance 0.2 → 0.05–0.10, or a quadratic-from-zero penalty with no band.
   Verify reset-noise settling still lives inside whatever band remains
   (reset noise is 0.05 rad ≈ 2.9°).
2. **Training-time action low-pass (~10–12 Hz).** The structural complement:
   this checkpoint already tolerates 20 Hz filtering at −180 reward, and a
   10 Hz in-loop filter makes the 18.7 Hz chatter inexpressible — the same
   "make the failure mode impossible" move as the passive toes. Requires a
   small feature (the filter exists only as an eval probe today) and a
   decision about whether filtered actions change the policy-interface
   contract.
3. **Seed replicates of this configuration.** Still the cheapest evidence
   about variance, unchanged from the previous investigation's advice, and
   now doubly interesting: is the left-leg asymmetry a seed artifact or an
   attractor?
4. **The 1a/1b split for recovery.** Unchanged; the impulse probe's first
   partial recovery does not alter the argument that correction must enter
   the task.

What this run retires: the toe mechanism (fixed by construction, confirmed by
absence), the entropy hypothesis (falsified at the 7M zero-crossing), and any
doubt about the substep-MIN metrics (the gate refused a policy the boundary
sample might have passed, which is exactly what #499 was for).
