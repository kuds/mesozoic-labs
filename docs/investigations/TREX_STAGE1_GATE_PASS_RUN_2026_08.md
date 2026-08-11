# T-Rex Stage 1 passes the stance gate: the filtered-plant run

**Date:** 2026-08-11
**Run:** `20260810_145546` (seed 42, 10,002,432 steps in 11h 30m, PPO/SB3) — **GATE: PASS**
**Plant:** physics r7 / policy interface r11 (passive toes, 10 Hz command low-pass)
**Config delta vs previous run:** the r11 filter (#503) and the shaping pack (#504:
`action_saturation` 0.5/0.9, `tail_home_pose` 0.25/0.05 at settled-droop targets,
`leg_home_pose_broad_fraction` 0.25 × scale 6); rails re-derived (statue 3495.2,
`min_avg_reward` 2100, `collapse_peak_floor_reference` 3495.2)
**Companions:** `TREX_STAGE1_PASSIVE_TOES_RUN_2026_08.md` (run 1),
`TREX_STAGE1_NARROW_TOLERANCE_RUN_2026_08.md` (run 2)

Point-in-time record. Not rewritten — corrections are appended.

---

## 1. Summary

The first stance-gate PASS in the repository's history, certified on
`robust_best_model.zip` (selected from 9.55M steps) over the standard
40-episode panel:

| criterion | required | certified | runs 1 / 2 |
|---|---|---|---|
| full-horizon fraction | ≥ 0.95 | **1.0000** (40/40) | 1.00 / 0.975 |
| mean unsupported duty | ≤ 0.02 | **0.0048** | 0.4084 / 0.3987 |
| unsupported duty UCB | ≤ 0.02 | **0.0080** | 0.4095 / 0.4029 |
| reward | rail 2100 | **3368.7 ± 46.7** | — |

The certified reward is 96.4% of the zero-action statue's 3495.2, with
bilateral support 0.9938 against the statue's own 0.998. Thirty-two of the
forty panel episodes recorded an unsupported duty of literally 0.0. The
formal pass condition (three consecutive passing panels) first completed at
**9.35M steps**, and 17 of the run's 200 panels passed outright.

The stance behind the numbers is what stages of this investigation kept
asking for and never got: a quiet, near-home, statically supported pose held
with **9.3° rms of total commanded motion** (normalised AC 0.135 — runs 1
and 2 ended at 0.61 and 0.50). Its balance runs through ±4–5° hip-roll
sways; its largest deviations from action zero are a deliberately lowered
head/neck (−20 to −22°) and a tail commanded into its measured settled
droop. **No actuator saturated at any point in the entire run** — the
per-check max |DC| never crossed 0.9.

---

## 2. Trajectory: three acts

Gate checks every 50k steps, 40-episode panels, seeds 3042–3081.

| act | steps | what happened |
|---|---|---|
| statue basin, found | 0–1.6M | after the usual spawn-fall trough, the policy converged nearly onto the statue: duty UCB **0.0047 at 1.4M** (the gate's duty criterion met, 1.5M steps into training), bilateral 0.98, reward 2657 — but full-horizon only 0.725–0.80: the quiet stance couldn't survive ~25% of noisy spawns |
| entropy exile | 1.6M–3.4M | a slow slide out of the basin into active balance: duty UCB up to ~0.42, bilateral down to 0.47, AC 0.27 → 0.64, reward down to a volatile 1650–2350; **no pinning, no crouch, still near home** throughout |
| re-descent and convergence | 3.4M–10M | every metric improves together: full-horizon locks to 0.95–1.00 (27 consecutive perfect panels from 8.0M), reward climbs through 3000 (surpassing act 1), and duty falls — slowly at first, then sharply through the 7M entropy zero-crossing (−0.10/M), settling exponentially into the basin: 0.24 at 6.6M, 0.075 at 7.8M, 0.019 at 8.9M, **0.0030 best panel at 9.15M**; first 3-consecutive pass streak at 9.35M |

**The exile-and-return is the run's scientific payload.** Act 1 proves the
redesigned landscape makes the statue basin *findable and payable*; act 2
shows what pushed every previous run's statue flash away — not the reward
(the deterministic policy was earning ~2650 in the basin) but the
**stochastic policy's fragility under its own exploration noise**: a statue
plus sampled action noise falls, an active stance absorbs it, and PPO
optimises the noisy return. Act 3 is the discriminating prediction coming
true: as `ent_coef` annealed to zero at 7M, the pressure vanished on
schedule and the policy re-descended — this time into a basin that pays it
to stay, with every alternative priced. Runs 1 and 2 never came back
because their landscapes made the exploit cheaper than the return trip.

The mid-run detour also bought something: act 3's stance survives noisy
spawns act 1's could not (full-horizon 1.00 vs 0.75 at matched duty). The
exile functioned as robustness training for the statue.

---

## 3. Anatomy of the certified stance

From the gate report's commanded-action table (post-settle, DC 0.439 rms,
AC 0.135 rms, 9.3° rms across angular position controls):

| actuator | command | swing | reading |
|---|---|---|---|
| head_pitch | −22.3° (DC −0.890) | ±1.9° | head lowered — parked **just under** the 0.9 saturation threshold |
| neck_pitch | −20.5° | ±1.6° | steadier than the authored high-headed pose |
| tail_1/2_pitch | −10.2° / −10.3° | ±2.0° / ±1.7° | commanded *into* the measured settled droop — the tail term's targets, learned |
| hip pitches | +5.8° / +6.7° | ±2.2° / ±2.5° | slight forward bias |
| knees / ankles | +1.8° to −5.6° | ±1.8°–2.1° | essentially home |
| hip rolls | −2.1° / +1.3° | **±4.7° / ±4.2°** | the working balance channel: slow lateral sways |

Reward decomposition per episode: alive 987.1, height 598.7, bilateral
584.6, leg_home_pose 420.4 (of 500), head_clearance 350.0 (max),
tail_home_pose 241.6 (of 250), neck 162.5, heading 98.6. Every cost is
single- or double-digit: saturation −24.3, load balance −23.9, energy
−15.5, smoothness −1.9, jerk −1.7. Nothing is left on the table but ~125
points of home-pose precision.

The head_pitch DC of −0.890 deserves its own sentence: the saturation
penalty's ramp starts at 0.9, and the policy rests its most-displaced
command a hair below it. The fence did not merely prevent parking — it is
visibly shaping where the policy chooses to sit.

---

## 4. The probes: a robustness profile, not an autopsy

For the first time the probe suite describes a healthy animal.

**Filter probe** — the run-over-run reversal in one table:

| cutoff | run 1 | run 2 | **run 3** |
|---|---|---|---|
| 35 Hz | survived | died (0.40) | 0.90 |
| 30 Hz | survived | died | **1.00** (reward 3324) |
| 20 Hz | survived | died | **1.00** (3341) |
| 10 Hz | died | died | **1.00** (3314) |
| 5 Hz | died | died | **1.00** (3232) |

The certified policy low-passed at **5 Hz** still stands every episode and
keeps 96% of its reward. Its feedback genuinely lives in the low band the
task requires (~1.1–1.4 Hz); the residual fast content is decoration, not
life support. (The lone 35 Hz head_contact is the sort of single-episode
noise the probe's own guidance says not to read.)

**Constant-hold probe**: the pose held with no feedback at all survives
3/10 episodes to the full horizon and averages 611 steps; failures are
gentle nosedives, not run 2's backward tail-slams (318 steps, 0/10,
9/10 tail_contact). The stance is *near*-statically stable — a small
residual forward lean that its own ±2° corrections trivially manage.

**Release ablation**: over-determined verdict — no single group is
necessary; knees_ankles and each leg alone are sufficient to tip the held
pose. Consistent with a small distributed lean rather than any joint doing
covert stabilisation work: only_tail, only_hip_rolls and (mostly)
only_head_neck each hold the full horizon on their own, exonerating the
groups with the most conspicuous DC.

---

## 5. What each intervention did (and what is attributable)

- **10 Hz command filter (r11)**: removed the exploit bandwidth. Both prior
  runs stabilised unstable poses with 16.8–18.7 Hz command content; run 3
  never built such a pose, and its final policy demonstrably does not need
  even 10 Hz (filter probe). This is the intervention with the clearest
  fingerprints on the outcome.
- **Long-range home gradient**: run 2's home error rose monotonically
  0.20 → 0.545 rad against a numerically zero gradient; run 3's *fell*
  monotonically from the first check (0.17 → 0.07 rad by 1.25M) and the
  final stance collects 84% of the leg term. The basin attracted, as
  designed.
- **Tail home-pose term (settled-droop targets)**: the final policy
  commands the tail into the droop and collects 97% of the term; the tail
  chain that run 2 parked at stops stayed live and quiet.
- **Saturation cost**: never meaningfully triggered (−24 of 3369) — a fence
  that worked by existing, with the head_pitch −0.890 as its visible mark.
- **Entropy-fragility hypothesis**: not an intervention but the run's key
  diagnostic claim, and it passed its test — the exile began while entropy
  was high, and the re-descent tracked the anneal, accelerating through
  the 7M zero-crossing exactly as predicted.

The honest caveat is unchanged: **one seed per condition** across all three
runs. The interventions changed together (filter + shaping pack), so their
individual contributions are not separable from this run alone, and seed-42
luck cannot be fully excluded. The within-run dynamics (act structure,
timing against the entropy schedule, absence of every previously observed
exploit) are strong mechanistic evidence, but a replicate seed — and
ideally a filter-only ablation run — is what would make the attribution
rigorous.

---

## 6. Three-run scoreboard

| | run 1 (passive toes) | run 2 (narrow tolerance) | run 3 (filter + shaping) |
|---|---|---|---|
| gate verdict | FAIL | FAIL | **PASS** |
| duty UCB | 0.4095 | 0.4029 | **0.0080** |
| full-horizon | 1.000 | 0.975 | **1.000** |
| certified reward vs statue | 68% | 62% | **96.4%** |
| AC (commanded) | 0.61 | 0.50 | **0.135** |
| strategy | toe-splay chatter | knee-lock + ankle pump | quiet near-home stance |
| pinned actuators | ~10 of 21 | 6 of 15 | **0 of 15** |
| filter probe | dies ≤ 10 Hz | dies at all cutoffs | **survives 5 Hz** |
| constant-hold | — | 0/10, tail-slam in ~0.3 s | 3/10 full horizon, mean 611 steps |

---

## 7. Minor findings and follow-ups

1. **The `success` column in the evaluation CSVs is not the gate verdict.**
   It records the stage-3 task event (for trex: the head_tip site reaching
   the prey — gated on `bite_bonus > 0`), which stage 1 sets to zero, so it
   reads `False` in every stage-1 row by construction — including in this
   passing run. The gate verdict lives in `stance_gate_report.{txt,json}`
   and `stage_summary.txt`. Worth renaming to `task_success` (or omitting
   in stages with no success event) — it invites exactly this misreading
   at the moment of the project's first pass.
2. The biped-empty "diagonal pair" contact panel (run-2 postmortem, item 6)
   remains unfixed; still cosmetic.
3. **Stage-2 handoff**: `robust_best_model.zip` (9.55M, r11-native) is a
   legitimate locomotion seed — low-bandwidth, near-statically stable, no
   inherited tremor. Note stage 2's TOML enables none of the new shaping
   terms yet; deciding their stage-2 weights (and whether the 10 Hz cutoff
   suits gait frequencies — a trot's stride content is well under 10 Hz,
   but this should be checked, not assumed) is stage-2 design work.
4. A seed replicate of this exact configuration would upgrade the
   attribution from mechanistic to statistical; a filter-only run would
   separate the filter's contribution from the shaping pack's.
