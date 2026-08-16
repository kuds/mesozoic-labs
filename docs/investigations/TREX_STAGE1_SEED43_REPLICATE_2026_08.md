# T-Rex Stage 1, Seed-43 Replicate — 2026-08-15

**Run**: `20260815_014118` (Colab, SB3 PPO, 4 envs, seed 43, 10,002,432 steps,
11h 30m 25s, started 2026-08-15 01:41 UTC). **Plant**: physics r7
(`sha256:72c662…`), policy interface r11 (`sha256:96ef13…`), visual r4 — the
exact plant of the gate-pass run. **Config**: the gate-pass configuration
unchanged (10 Hz command low-pass, #504 shaping pack, rails 2100 / 3495.2);
the run started before the 2026-08-15 cleanup merge, so its artifacts carry
the pre-rename `success` CSV column and the old `[5, 10, 20, 30, 35]` Hz
probe sweep. **Purpose**: the seed replicate every postmortem since the
bounce investigation has asked for — the first attempt to distinguish
"solved" from "lucky" on the certified configuration.

This is a point-in-time record. Written from the run's shipped gate report,
probe reports, `gate_progress.npz`, and summaries; the per-step stance CSVs
and `diagnostics.npz` (`algo_std` trajectory) were not re-analyzed (§7).

---

## 1. Verdict: FAIL — and the answer to "solved or lucky" is "seed-dependent"

Certification ran on `robust_best_model.zip` (selected 9.95M) over the
standard 40-episode panel (seeds 3042–3081, settle 200):

| criterion | seed 43 (this run) | seed 42 (gate pass) | required |
|---|---|---|---|
| full-horizon fraction | **0.9250** (37/40; 3 nosedives) | 1.0000 | ≥ 0.95 |
| mean unsupported duty | **0.0597** | 0.0048 | ≤ 0.02 |
| duty UCB (95%) | **0.0747** | 0.0080 | ≤ 0.02 |
| duty episodes | 37 (< 38 floor) | 40 | ≥ 38 |
| panel reward | 3093.0 ± 497.0 | 3368.7 ± 46.7 | ≥ 2100 (rail) |
| bilateral support | 0.9269 | 0.9938 | (statue 0.998, not gated) |
| panels passing, whole run | **0 of 200** | 17 of 200 | 3 consecutive |

The reward is the trap the gate exists for: best eval 3216 ± 183, panel 3093
— comfortably past the 2100 rail, 88% of the statue. By every pre-gate
standard this run "succeeded." Under `stance_quality/v1` it failed every
stance criterion, and no panel in the entire run passed even once.

With n = 2 on the identical configuration (1 PASS / 1 FAIL), the "passes or
bounces" known issue has its answer: **the certified configuration is
seed-sensitive**. Not solved, not merely lucky — the shaping pack reliably
finds the right basin (§3), and the seed decides whether the policy quiets
enough inside it to certify.

## 2. The trajectory: the same three acts, with a stalled third act

`gate_progress.npz` (200 panels at 50k intervals) reproduces the gate-pass
run's three-act structure almost beat for beat, then diverges at the end:

| steps | duty | UCB | full-horiz | AC rms | reward |
|---|---|---|---|---|---|
| 0.85–1.2M | **0.0000–0.0001** | 0.0003 | 0.125–0.55 | 0.25–0.38 | ~1400 |
| 3.0M | 0.4005 | 0.4080 | 1.000 | 0.633 | 2404 |
| 6.0M | 0.2474 | 0.2615 | 0.975 | 0.568 | 2528 |
| 7.0M | 0.0952 | 0.1124 | 0.825 | 0.398 | 2680 |
| 8.0M | 0.0790 | 0.0988 | 0.850 | 0.382 | 2874 |
| 9.0M | 0.0557 | 0.0726 | 0.850 | 0.351 | 2999 |
| 9.95M | 0.0576 | 0.0694 | 0.975 | 0.320 | 3216 |

- **Act 1 (statue flashes)**: duty exactly 0.0000 at 0.85–0.95M — but
  full-horizon only 0.125–0.55. Same as seed 42's act 1 (duty 0.0047 at
  1.4M, full-horizon 0.725–0.80): the quiet stance exists early and cannot
  survive noisy spawns under the stochastic policy.
- **Act 2 (entropy exile)**: duty climbs to 0.40 by 3M with AC 0.63 —
  seed 42 peaked at ~0.42. Survival locks in (full-horizon ~1.0) while
  stance quality is spent on exploration robustness.
- **Act 3 (the re-descent — where the seeds diverge)**: seed 43 descends on
  schedule through the entropy anneal (−0.15/M into 7M, exactly the window
  where `ent_coef` reaches zero) and then **stalls at duty ≈ 0.06–0.08 with
  AC ≈ 0.32–0.35 for the final two million steps**. Seed 42's descent
  continued through the same window to 0.019 by 8.9M and 0.005 at
  selection, with AC collapsing to 0.135.

The pre-registered prediction in the config (`ent_coef_end` comment: by ~4M,
`algo_std` well under 0.2 and duty below 0.28) **failed on this run's duty
axis**: duty at 4M was 0.345. The run still descended afterward, so the
mechanism (anneal → quieting) is intact but its timing calibration is not;
the `algo_std` half of the prediction is checkable in `diagnostics.npz` and
was not checked here (§7).

## 3. Same pose, noisier execution

The two seeds converged on the **same stance**. Seed 43's selected
checkpoint, post-settle: head_pitch DC −0.803 (−20.1°), neck_pitch −0.643
(−19.3°), tail commanded into its settled droop (tail_1_pitch −8.3°),
balance carried by hip-roll sways (l −9.2° ± 8.8°, r +7.2° ± 7.7°), **zero
of 15 actuators saturated** (max |DC| 0.803). Seed 42: head_pitch −22.3°,
neck −20.5°, tail droop, ±4–5° hip-roll sways, zero saturated. The #504
shaping pack's basin is reproducible; nothing about the *target* of
optimization differed.

What differs is the noise around it: AC 0.329 rms vs seed 42's 0.135, at a
reported ~33.9 Hz — which is the frequency estimator's white-noise
saturation point (~33.3 Hz at 100 Hz control), so this is **broadband
command noise, not a coherent tremor**, and not any of the named prior
failure modes (no 16.7/20 Hz subharmonic lock, no knee-lock crouch, no
saturation parking). Per-actuator, the largest residual AC rides the hip
rolls throughout the descent (peak actuator AC 0.47–0.56 from 6.5M to 10M,
consistently the roll pair that carries lateral balance, plus tail pitch) —
the noise lives exactly in the joints doing the balancing. Duty 0.06 is the
foot-contact cost of balancing with a hand tremor: the pose is right, the
execution is loud, and 3 of 40 episodes tip into nosedives.

## 4. Probes

- **Statue controls verify the rails are fresh**: `hold_zero` scored 3493.8
  (constant-hold) and 3497.1 (release ablation) against the configured
  statue 3495.2 ± 13.9. The panel measured a real policy deficit, not
  constant drift.
- **Constant-hold**: hold-after-settle survives 3/10 to the horizon (531
  steps mean; ramped 3/10; from-reset 2/10). Statically the pose is about
  as stable as seed 42's (3/10, 611 steps) — further evidence the *pose* is
  shared and the noise is the difference.
- **Impulse recovery** (0.5/1.0/2.0 m/s lateral, both signs, impulse at
  step 200): partial recoveries only — 4/8 at 0.5 m/s +y (+349-step margin
  over the statue), 2/8 at −y (+151), nothing at ≥ 1.0 m/s. The report's
  full-recovery envelope reads **none/none**, weaker than seed 42's
  0.50/0.00 envelope. A noisy stance has less margin to spend on
  disturbance rejection.
- **Filter probe** (old sweep; 10 episodes/cutoff): 5 Hz → 0.80
  full-horizon at 3013.9 reward; 10 Hz → 0.90; 20/30/35 Hz → 0.60, 0.60,
  1.00. No r6-style collapse at any cutoff. The ≥ 20 Hz spread is
  10-episode sampling noise around the policy's own ~0.9 survival — on the
  r11 plant those cutoffs stack a wider filter onto the plant's 10 Hz pole
  and measure nearly the unprobed checkpoint (the 2026-08-15 sweep
  re-derivation fixes this going forward). The informative rows: mild
  degradation at 5 Hz, none catastrophic — this policy, unlike every r6
  policy, does not *depend* on high-frequency content; it merely emits it.
- **Release ablation**: strongly lateralized — `only_right_leg` falls 8/8
  ("sufficient alone, not necessary"), `only_left_leg` survives 5/8; the
  releases of hip_rolls/knees_ankles/left_leg all fail to rescue the held
  pose. The stabilisation load concentrates on the right side for this
  seed (run 1's chatter localized left) — per-seed asymmetry, consistent
  with per-seed noise rather than a structural plant defect.

## 5. What this changes

1. **Seed replication is no longer optional evidence — it is the finding.**
   1/2 at the gate. The gate-pass postmortem's one-seed caveat was the
   correct instinct.
2. **The shaping pack is validated at the level it can be**: both seeds find
   the same unsaturated, near-home, tail-live stance, and the reward's
   gradient carries both through exile and re-descent. What the reward does
   not control is the **residual noise floor after the entropy anneal** —
   seed 42's std collapsed onto the pose, seed 43's did not, and no reward
   term prices broadband command noise directly (smoothness/jerk price
   sample-to-sample deltas on the *filtered* command, which the 10 Hz
   filter has already softened; nothing sees the pre-filter std).
3. **The 1b rationale strengthens**: robustness-by-lucky-seed is now
   measured, not suspected. A recovery stage with declared disturbances
   (and its paired null tests) certifies the property the undisturbed task
   provably selects for only sometimes.
4. **The certified seed-42 checkpoint remains the 1b warm-start** — it is
   unaffected by this result and is further validated downstream (its
   stage-2 locomotion continuation passed its gate at 2.16 m/s on
   2026-08-11).

## 6. Candidate responses (for discussion, not adopted)

- **More seeds first** (a third run is planned): at 1/2 the pass-rate
  estimate is uninformative; each additional seed at ~11.5 h materially
  narrows it.
- **Post-anneal noise floor**: investigate why `algo_std` fails to collapse
  for some seeds — candidates include a longer anneal tail (the decay ends
  at 7M; seed 43's stall spans exactly the post-anneal window), an
  explicit late-stage std penalty, or pricing pre-filter command noise
  (the filter currently hides it from smoothness/jerk).
- **Do nothing to stage 1 and let 1b decide**: if the recovery stage's
  training pressure quiets the stance anyway (a shove punishes a noisy
  stance more than a quiet one), stage-1 seed variance may not be worth
  fixing in isolation. This is measurable once the perturbation engine
  exists.

## 7. Not analyzed here / follow-ups

- `diagnostics.npz` (`algo_std`, `raw_action_saturation` trajectories): the
  direct test of "std never collapsed" and the second half of the 4M
  prediction. Available in the run dir.
- Per-step stance CSVs and replays: where the 3 nosedives start, and
  whether the duty concentrates in specific episodes or spreads evenly.
- The next seeded run (launching 2026-08-15): with a third seed, update the
  "passes or bounces" entry from 1/2 to x/3 and revisit whether the
  post-anneal noise floor correlates with anything observable at 4M (the
  config's prediction checkpoint).

---

**Addendum (2026-08-16).** The third seed is in: seed 44, run
`20260815_205206`, **PASS** — and the strongest certification yet
(full-horizon 40/40, duty 0.0069 / UCB 0.0117, panel reward 3408.3 ± 88.5 =
97.5% of the statue, zero non-truncated terminations). Its action signature
is the decisive datum for §5's noise-floor hypothesis: **AC 0.132 rms** —
the same quiet endpoint as seed 42 (0.135), nowhere near this run's stalled
0.329 — on the same unsaturated, head-lowered, tail-droop stance (head
−20.3°, neck −21.0°, max |DC| 0.830). Replication stands at **2/3**, split
exactly along the post-anneal noise floor: both passes quieted, the one
fail did not. "What decides the anneal's endpoint" is now the sharpest open
question stage 1 has, and §6's candidate responses are unchanged.
