# T-Rex Recovery Stage (1b), First Field Runs and P3 Calibration — 2026-08-22

**Stage**: `recovery` (semantic id; manifest position 2, no legacy number).
**Plant**: physics r7 (`sha256:72c662…`), policy interface r12
(`sha256:4890b4…`), visual r4. **Task**: `mesozoic.task-fingerprint/v1`
`sha256:97c28f29…` — stance's `[env]` verbatim plus the five
`perturbation_*` keys; identical across both recovery runs below, so their
policies and every null panel here are directly comparable without
re-measurement. **Derived disturbance** (from the plant, not configured):
165.501 N for 0.200 s, impulse 33.100 N·s, from subtree mass 85.72 kg,
support radius 0.0773 m, CoM height 0.884 m, capture velocity 0.2574 m/s at
multiple 1.5. **Purpose**: record the first successful recovery training
runs, the P3 safe-set calibration, and what the evidence does and does not
support.

This is a point-in-time record. Written from the runs' shipped summaries and
fingerprints, plus certification panels and calibration traces rolled
locally on the container (§8 lists what was not done).

---

## 1. Verdict

The recovery stage works as a **certification instrument** and is not yet
proven as a **capability stage**.

* Certification: recovery is the only task in this project where doing
  nothing is fatal. Both nulls score **0/40** under every safe set tried.
  The stance gate cannot make that separation — its own reward optimum is a
  statue.
* Capability: the trained policy reaches **21/40** (LCB95 38.5%) under the
  calibrated safe set and beats its own warm-start parent by a paired
  **+0.30 (LCB95 +0.15)**. Real, but modest, and the 5M run peaked at 1.6M
  steps and drifted down thereafter.
* The open question that decides its sim-to-real value is **schedule
  memorization vs. reactive control** (§6.1), which no run so far tests.

---

## 2. Field runs

### 2.1 Three attempts that never reached the stage

| run | stance | what happened next |
|---|---|---|
| `20260816_180603` | PASS 06:46 | proceeded to locomotion — `RUN_RECOVERY_STAGE` was still `False` |
| `20260817_165515` | PASS 05:55, closeout 06:05 | **nothing** — no recovery dir, no locomotion, no further Drive writes |
| `20260818_134249` | PASS 02:37 | **nothing** — same signature |

Root cause of the two halts, reproduced deterministically: plant identities
record `model_path` repository-relative, and `derive_stage_task_fingerprint`
passed it straight to `mujoco.MjModel.from_xml_path`, which resolves against
the **process cwd**. The test suite runs from the repository root so this
always passed; Colab adds the clone to `sys.path` without chdir-ing into it,
so the first pushed-stage fingerprint raised `ParseXML` *before* the stage
directory existed — halting run-all with nothing in Drive to explain why.
Push-free stages never load the model, which is why three full stance runs
sailed through the identical code path. Fixed in PR #508; the path now
resolves against the repository root and a regression test pins
cwd-independence.

### 2.2 `20260819_154702` — first recovery training run (3M)

Warm-started from that run's certified stance checkpoint via
`initialize_next_stage`; the checkpoint carries both its own recovery
fingerprint and the stance lineage record. Trained 3M steps in 3h 07m.
Colab eval reward 2811 ± 474. Panel results in §3.

Replays were **not** produced: `record_stage_video` seeded the replay env
with `seed + 2000 + stage`, which raises `TypeError` for the string
`"recovery"` inside the recorder's best-effort `try/except` — figures were
written, videos silently were not. Fixed in PR #509 via `replay_seed()`
(integer stages bit-identical; semantic ids get a stable crc32 offset).

### 2.3 `20260821_142144` — second recovery run (5M), full pipeline

First run on the merged stage-structure code (`repository_commit
a0c92d89…`, clean tree). All three legs completed:

| stage | window | result |
|---|---|---|
| `01_stance` | 14:21 → 03:42 (13h 20m) | 11,001,856 steps, gate **PASS** |
| `02_recovery` | 03:42 → 08:39 (**4h 56m 12s**) | 5,005,312 steps, gate FAIL-by-design |
| `03_locomotion` | 08:40 → … | running |

Recovery numbers: final eval **2907.03 ± 463.98**; best eval **3055.63 ±
208.86 at 1,600,000 steps**; best-model re-eval 2941.54 ± 422.41; mean
episode length 956.1 ± 107.7 steps. Reward variance is ~14× stance's
(±464 vs ±33) — the pushes are the variance.

Three pieces of machinery were confirmed in the field by this run:

1. **Replays**, both checkpoints: `trex_ppo_recovery_{final,selected}.mp4`
   plus `_side` / `_front` views and both per-frame stance CSVs.
2. **Pushed fingerprint**: `stage: "recovery"` with the full derived
   perturbation block — the code path that had failed three times.
3. **`NN_id` layout**: `01_stance` / `02_recovery` / `03_locomotion`, and the
   `none/v1` gate printed as informational rather than raising, so the
   notebook flowed on to locomotion as designed.

**The budget increase did not pay.** The best checkpoint landed at 1.6M of
5M and the endpoint is ~5% below it. The binding constraint is the
entropy/LR schedule for a warm-started run, not the step budget — exactly
what `recovery.toml`'s own comment anticipated. Two pilots now say so.

---

## 3. Certification panels

All panels: 40 episodes, seeds 3042–3081, the same pushed task. Pairing is
**structural** — the schedule derives from the reset seed via a
NumPy/JAX-identical integer-hash PRF, so every controller receives the same
shoves at the same steps without coordination. Provisional judging:
`t_recover` 100 steps (2.0 s), dwell 50 steps (1.0 s).

| controller | success | full horizon | per-shove | mean length | reward |
|---|---|---|---|---|---|
| statue (zero action) | 0/40 | 0/40 | 26/50 (52%) | 360 | 975 ± 217 |
| brace (policy's post-settle mean, held) | 0/40 | 0/40 | 8/35 (23%) | 260 | 680 |
| stance policy (warm-start floor) | 7/40 | 13/40 | 74/128 (58%) | 769 | 2203 |
| recovery policy, 3M | 9/40 | 27/40 | 100/152 (66%) | 929 | 2799 ± 587 |

Three findings from these panels:

**3.1 The nulls prove active feedback, not posture.** The brace holds the
trained policy's own post-settle mean action (measured on off-panel seeds
5042–5046) and dies *faster* than the statue — 260 vs 360 steps, 23% vs 52%
per-shove. The policy's flexed stance is **less passively stable** than the
home keyframe's springs; it stands only because feedback is wrapped around
it. This pre-emptively answers the "it's just a better set-point" objection
to any future recovery pass.

**3.2 Per-shove recovery alone cannot certify.** The statue passively
"recovers" 52% of judged shoves — the plant's springs and damping genuinely
re-enter the safe set with zero actuation. Only the episode-level
conjunction (full horizon **and** every shove recovered) separates
controllers. The gate's design assumption is now measured, not argued.

**3.3 §8.1 transfer pilot: no force ramp needed.** The certified stance
policy survives the real 165.5 N schedule — shortest episode 395 steps, so
every episode survives the *first* shove; failures accumulate at shoves 2–4.
The plan's ramp option (§3.3) is not required, and training starts from a
useful ~17% floor rather than zero.

*(Method note: the stance policy's VecNormalize arrived from Drive with 60
bytes missing, all inside the ±inf space constants. The normalizer was
rebuilt from the intact statistics and **validated** by re-rolling the
un-pushed stance panel — 3446 mean vs the gate report's 3431.8 ± 69.9, 10/10
full horizon — before it was allowed to drive the pilot.)*

---

## 4. P3 calibration

Thresholds were derived **only** from the certified stance policy standing
quietly (16,000 post-settle steps over 20 off-panel episodes), never from
the recovery policy — the exam is not fitted to the student. Panels were
then re-judged offline from instrumented physics traces; the offline judge
reproduces the live panels exactly (0/0/7/9), which validates it.

### 4.1 Derived thresholds

| quantity | quiet stance | provisional | calibrated (p99.9 × 1.5) |
|---|---|---|---|
| height reference | 0.9267 m (measured settled median) | per-episode reset stamp | **0.9267 m fixed** |
| height error | p99.9 = 0.0112 m | ≤ 0.10 m | **≤ 0.0168 m** |
| tilt | p99.9 = 0.055 rad | ≤ 0.35 rad | **≤ 0.0825 rad** |
| planar speed | p99.9 = 0.2136 m/s | ≤ 0.30 m/s | **≤ 0.3203 m/s** |
| min foot force | p0.1 = **0.0 N** | ≥ 0.1 N | **removed — see §4.2** |

Certified stance is far tighter than the provisional guesses: 9× on height,
6× on tilt. Speed was essentially right. `t_recover` is validated rather
than merely retained: measured re-entry times have **p90 = 84 steps**,
comfortably inside the 100-step window.

### 4.2 The support criterion fails the certification target itself

The per-step "both feet ≥ 0.1 N every step" criterion **cannot be met by the
thing it is meant to certify**: quiet certified stance transiently reads
0.0 N on a foot during weight shifts (p0.1 of min foot force is exactly
0.0). It is a touch-sensor measurement artifact, not a policy property, and
the blocker histograms show it denying more shoves than any posture
criterion for the trained policy (22 of the recovery policy's denied shoves,
vs 0 height and 0 tilt).

Any future support term must be **windowed** (e.g. dwell-averaged) and
re-derived from certified stance before it can gate.

### 4.3 Re-judged panels

Successes out of 40:

| safe set | statue | brace | stance | recovery (3M) | paired recovery − stance |
|---|---|---|---|---|---|
| provisional | 0 | 0 | 7 | 9 | +0.05, LCB −0.097 |
| calibrated, with support term | 0 | 0 | 4 | 8 | +0.10, LCB −0.018 |
| **calibrated, posture-only** | **0** | **0** | **9** | **21** | **+0.30, LCB +0.150** |

Under the defensible safe set the trained policy reaches **21/40, LCB95
38.5%**, both nulls stay at zero (UCB95 7.2%), and — the thing the
provisional thresholds could not show — **training superiority over its own
warm-start parent becomes statistically established** (paired LCB +0.15 >
0). The support artifact alone had been masking 13 of the policy's 21 real
recoveries.

Paired margins under the calibrated posture-only set: **+0.525 (LCB +0.390)**
against each null, +0.300 (LCB +0.150) against the warm-start parent.

---

## 5. Assessment

**What recovery has earned.** It is the project's only falsification
instrument for the statue problem. Stage 1a's entire history is a fight
against a reward optimum achieved by doing nothing; `stance_quality/v1`
patched that by measuring posture rather than return, but posture is a quiet
measurement — it certifies a pose, not control. Recovery is categorical:
the statue is 0/40 and topples by step 360. The brace result (§3.1) sharpens
it into a positive claim about the policy's mechanism.

**Where it is weaker than it looks.** The training half is unproven. 21/40
is a real margin over nulls but is not a robust policy; the 5M run peaked at
a third of its budget; and the disturbance model is narrow — one magnitude,
one duration, horizontal only, at the root body, on a fixed 2.0 s ± 0.5 s
cadence. Real reality-gap forces are persistent and structured (slopes,
compliant ground, mass-identification error, actuator lag), not impulsive.
Note for scale: the 10 Hz command low-pass filter added earlier is arguably
a **larger** sim-to-real intervention than pushes, because it removed a
policy behaviour (16–18 Hz chatter) no real actuator could reproduce.

**Recommendation.** Keep the stage, but bank it as a certification stage
rather than a capability stage until §6.1 comes back. Its highest current
value is that it yields a falsifiable robustness number to track as the
things that actually close the reality gap are added (actuator models,
contact/friction randomization, sensor noise, mass error). Without it, those
changes would land with no instrument able to say whether they helped.

---

## 6. Next steps, in priority order

### 6.1 Off-distribution generalization test (highest value, ~10 min)

The policy cannot see the pushes — deliberate — but it **can** learn their
timing statistics. At 2.0 s interval with only ±0.5 s jitter, a policy could
learn a periodic bracing rhythm rather than reactive recovery. The two are
indistinguishable on the training schedule and completely different for
transfer.

Roll the certification panel at schedules the policy never trained on, using
the frozen nulls (valid: task hash unchanged):

* **timing**: interval 3.5 s, jitter ±1.5 s
* **magnitude**: 120 N and 210 N (capture multiples ≈1.1 and ≈1.9)

If the margin over the nulls holds → disturbance rejection was learned, and
the stage is genuine sim-to-real progress. If it collapses → the schedule
was memorized, and the fix is **per-episode randomization of magnitude and
interval**, not a longer budget.

### 6.2 Panel the 5M policy against the 3M policy

The `20260821_142144` recovery checkpoint has not been panel-tested. Same
seeds, calibrated posture-only safe set, direct comparison against the 3M
policy's 21/40. Answers whether the extra budget helped, hurt, or did
nothing on the metric that will actually gate.

### 6.3 Re-derive the warm-started schedule (blocks further training)

Both pilots peaked early (best at 1.6M of 5M in run 2). Re-derive
`ent_coef` / `learning_rate` decay for a warm-started 3M-class budget before
spending more compute; the current values are stance's, mirrored as a
placeholder and explicitly marked pilot-dependent.

### 6.4 P5 — freeze the gate resolution

Once §6.1 and §6.2 land, write `gate_resolution.json` with:

* safe set: calibrated **posture-only** (reference 0.9267 m, height 0.0168,
  tilt 0.0825, speed 0.3203), `t_recover` 100, dwell 50
* null manifest: the frozen statue and brace panels with exact
  Clopper–Pearson UCBs
* `min_recovery_success_lcb`: **open decision.** The current policy sits at
  LCB 0.385 against a null UCB of 0.072. The plan's aspirational 0.725 (LCB
  of 34/40) needs a materially stronger policy. This is a deliberate choice
  between freezing an attainable first threshold now and extending the
  training recipe first — and it is now a choice made with measurements.

### 6.5 Smaller items

* Re-derive `collapse_peak_floor_reference` for recovery from the **pushed**
  statue (974.7 ± 216.6) rather than the un-pushed 3495.2 placeholder; the
  current value errs in the arming direction, so this is correctness, not
  safety.
* `recovery.toml`'s `collapse_peak_warmup_timesteps` comment still says "1/3
  of this stage's 3M budget" after the budget moved to 5M. Cosmetic.
* Consider a per-shove replay overlay (border flash / direction inset during
  push windows) so reviewers can see the disturbance in the video. Design
  discussed 2026-08-20; deferred.

---

## 7. Reproduction

* Panels and calibration: `roll_recovery_panel` /
  `write_recovery_evidence` (`environments/shared/recovery_evaluation.py`),
  judged by `per_push_recovery` / `episode_recovery_success`
  (`environments/shared/curriculum/recovery_gate.py`), bounds by
  `binomial_lcb` / `binomial_ucb`, paired margins by
  `stance_gate.one_sided_t95`.
* Evidence CSVs (one row per episode **and** one per shove, carrying
  controller id, seed, push vector, timing, and recovery outcome) were
  produced for all four controllers and shared with the project owner; they
  are not committed here.

## 8. Not done

* The 5M recovery checkpoint has not been panel-tested (§6.2).
* No off-distribution schedule has been rolled (§6.1).
* `diagnostics.npz` per-step series were not re-analyzed for either run.
* No MJX/JAX recovery run exists; every number here is SB3.
* Recovery runs are excluded from result bundles by design — the bundle
  schema remains integer-stage until that migration.
