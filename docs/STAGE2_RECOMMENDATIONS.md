# Getting Past Stage 2 — Review of Run 20260711_165924 and Ranked Recommendations

Review of the latest velociraptor PPO run (`20260711_165924`, 2026-07-11) against
the Drive summaries and the repo state, with corrections to the earlier analysis
and a ranked, concrete plan for clearing stage 2. Follows up on
[STAGE2_INVESTIGATION.md](STAGE2_INVESTIGATION.md) (run `20260709_185946`).

## TL;DR

Run `20260711_165924` was a **bit-for-bit replication** of the July 9 failure —
same plant, same config, same seed, and (verified) bitwise-identical stage-2
eval arrays. It confirms the collapse is deterministic but tested **none** of
the investigation's experiments: the plant A/B arm has still never run, and all
Experiment C knobs (`ent_coef_end`, `fall_penalty`, `forward_vel_max`) are still
at their old values on `main`. 3h21m of the 4h run was spent retraining stage 1
to bit-identical results.

To get past stage 2, in order of leverage:

1. **Warm-start stage 2** from the saved stage-1 checkpoint (mechanism exists
   today) — cuts iteration from ~4h to ~40min–5h of pure stage-2 signal.
2. **Raise hip/ankle `forcerange`** (0.8×kp → 1.5×kp) on a branch merged to
   `main`, updating the actuator-bounds test pins in the same PR — this is the
   still-unrun experiment the evidence points at.
3. **Recalibrate the 2.0 m/s gate** against a measured top speed (none exists);
   precedent: the brachiosaurus gate was cut 2.0 → 0.75 and passed at 1.12 m/s.
4. **Apply the cheap config changes** (`ent_coef_end = 0.001`,
   `forward_vel_max = 2.5`, `fall_penalty = -150`) — with the corrected
   break-even math below (the old −250 figure was wrong).
5. **Keep the collapse early-stop from censoring experiments** — both failures
   were killed at 1.1M/8M steps, mid-ramp, so no bounded-plant run has ever
   trained with the full speed incentive.

---

## 1. What the run showed

### 1.1 Results

| | Stage 1 (balance) | Stage 2 (locomotion) |
|---|---|---|
| Outcome | **PASSED** | **FAILED** (early-stopped 1.1M / 8M steps) |
| Final eval | 1747.65 ± 19.51 | 348.01 ± 511.54 |
| Best eval | 1758.66 ± 15.43 @ 5.8M | 1080.6 ± 341.0 @ 800k |
| Robust-best model eval | 1752.96 ± 17.15 | 948.31 ± 285.35 (len 907.9) |
| Mean fwd vel | 0.10 m/s | 0.76 ± 0.21 m/s (gate: 2.0) |
| Duration | 3h 20m 57s | 37m 57s |

### 1.2 It was an exact replication, by construction

- **Plant**: `raptor.xml` has not been touched since `156a933` (the
  forcerange/implicitfast commit, merged to `main` 3 minutes before the July 9
  run started). Both July runs trained on the identical bounded plant.
- **Config**: the only stage-2 TOML change between the runs is the *commented-out*
  `ent_coef_end` suggestion. Same seed (42), same n_envs (4).
- **Verified determinism**: the two runs' stage-2 `evaluations.npz` arrays are
  **bitwise identical** (22 checkpoints × 30 episodes). Stage 1 also reproduced
  to every printed decimal.
- The apparent stage-2 "best" difference (1068.48 → 948.31) is only the new
  `RobustBestModelCallback` (PR #433) choosing a different checkpoint to
  re-evaluate — the underlying training trajectory was identical.

So the run adds two genuine facts: the collapse is **deterministic and
reproducible**, and the **robust checkpointing worked as designed** — it
selected the healthy 400k checkpoint (mean−std peak 986.6, eval 948 ± 285 over
908-step episodes) instead of the seductive-but-bimodal 800k peak. Everything
else was already known.

### 1.3 Collapse anatomy (matches July 9 point-for-point, with new granularity)

From the decoded `evaluations.npz` / `diagnostics.npz`:

- Clean climb to 1039.5 ± 52.9 @ 400k with **zero** catastrophic episodes
  through 450k; bimodality onset at exactly 500k (std 50 → 165).
- Catastrophic fraction (eval reward < 300) climbs 1/30 → **25/30** at 1.1M —
  non-monotonically (dips to 3/30 at 800k), which is why the whole-run *mean*
  peak (1080.6 @ 800k) sits deep inside the bimodal regime.
- The bimodality is a **widening scissor**: per-checkpoint max episode reward
  climbs monotonically 915 → 1497 (survivors keep getting faster/better) while
  min crashes to 13–40. The policy trades robustness for speed.
- Failures are overwhelmingly **body_contact** terminations, preceded by
  measurable drift: drift_distance 0.19 → 2.74 m, pelvis height sagging
  0.495 → 0.486 m, tilt and spin instability rising.
- PPO internals healthy throughout (approx_kl ~0.015–0.016, explained_var
  0.86–0.94); **action std grew 1.179 → 1.489** under constant `ent_coef`
  (entropy still rising at stop). Training-rollout metrics improve to the very
  last step (forward_vel 0.02 → 0.61) — the training signal never sees the
  problem the deterministic eval sees.

### 1.4 What actually stopped the run (important, previously unexamined)

The stop at 1.1M was **not** the curriculum gate. It was
`EvalCollapseEarlyStopCallback` (eval mean >30% below peak for 5 consecutive
50k-cadence evals; hardcoded `min_evals=8, patience=5` in
`train_base.py:385`). Two consequences:

- At the stop, the `forward_vel_weight` ramp (0.2 → 2.0 over 2M steps) was only
  **55% complete** (effective weight 1.19). No bounded-plant run has ever
  trained with the full speed incentive, let alone the 6–8M steps every
  passing run used.
- The 0.76 m/s figure is a **censored policy outcome**, not a plant ceiling —
  training-rollout velocity was still climbing when the run was killed at 14%
  of budget.

---

## 2. Corrections to the previous investigation

### 2.1 The fall-penalty break-even math was wrong

The "Secondary factor" section of STAGE2_INVESTIGATION.md computed break-even
`fall_penalty ≈ −250` by weighing the speed gain against **only** the
discounted penalty (50 × 0.995³⁰⁰ ≈ 11). It omitted that falling also
**forfeits all remaining future reward**. Corrected accounting for its own
scenario (robust 1.0 m/s surviving 1000 steps vs fragile 1.5 m/s falling at
step 300, γ = 0.995):

- Falling forfeits a discounted tail of ≈ 226 (alive bonus ≈ 97 + foregone
  speed reward ≈ 129, valued at the fall time).
- The corrected break-even is ≈ **−7**, and the existing −50 already makes the
  robust gait optimal in that scenario (discounted return 231.8 vs 222.2).
- The break-even is wildly fall-timing-sensitive: a fragile 2.0 m/s gait
  falling at step 300 needs ≈ −240; falls at step 500/700 need ≈ −540/−1980;
  falls before ~step 300 need no penalty at all.

Implications: raising `fall_penalty` to −150 is still a reasonable hedge (it
targets the late-fall region where the incentive really is broken), but it is
**not** the clean fix the −250 figure implied, and no realistic penalty makes
very late falls unprofitable at γ = 0.995. Note also that eval selection and
the gate use *undiscounted* 30-episode means, where fragile gaits already lose
badly (≈ 400 vs ≈ 1167) — the bias lives in the discounted training objective
and in exploration noise, which is why entropy decay and robust checkpointing
are complementary levers, not nice-to-haves.

### 2.2 "The 2.0 m/s gate may be unreachable" — status: unmeasured

The gate margin clearly collapsed (March passes: 3.1–3.9 m/s on the unbounded
plant; bounded runs: 0.76 m/s censored at 1.1M steps; hips clip 34–40% and
ankles 22–25% of a moderate scripted gait cycle). But **no top-speed
measurement of the bounded plant exists** — `actuator_saturation_report.py`
measures clip fractions, not attainable speed, and the knees (largest
actuators) clip 0%. Treat reachability as an open measurement, not a
conclusion (§3.3).

---

## 3. Recommendations (ranked)

### R1 — Stop retraining stage 1 (do this for every experiment below)

Stage 1 is deterministic at seed 42 and its checkpoint is already on Drive.
Warm-starting stage 2 is supported today:

- **Notebook** (`sb3_training.ipynb`): skip the stage-1 cells and set, before
  the stage-2 cell:
  ```python
  path_1    = "/content/drive/MyDrive/mesozoic-labs/logs/velociraptor/ppo/20260711_165924/stage1/models/robust_best_model.zip"
  vecnorm_1 = "/content/drive/MyDrive/mesozoic-labs/logs/velociraptor/ppo/20260711_165924/stage1/models/robust_best_model_vecnorm.pkl"
  ```
  (`train_stage(stage=2, load_path=..., vecnorm_path=...)` already accepts
  arbitrary paths.)
- **CLI**: `train --stage 2 --load <path>` (vecnorm auto-derived).
- **Sweeps**: `ray_tune_sweep.ipynb` `STAGE=2` + `LOAD_PATH`, or Vertex
  `launch --stage 2 --load ...`.

Payoff: ~3h21m saved per experiment; a collapsing stage-2 run costs ~40min.
Consider adding a `START_STAGE` / checkpoint-path knob to the notebook config
cell so this doesn't require cell surgery each time.

### R2 — Run the plant experiment that never ran (highest information)

Both July runs were the *control* arm. The treatment arm — more actuator
headroom — is still untested. Recommended sizing (Experiment B of the
investigation, skipping the pure revert since the bounded plant is the
long-term intent):

```xml
<!-- raptor.xml: hips and ankles only; knees/toes measure ~0% gait clipping -->
r_hip_pitch_act / l_hip_pitch_act:  kp=150  forcerange="-225 225"   (was ±120, 0.8×kp → 1.5×kp)
r_ankle_act     / l_ankle_act:      kp=100  forcerange="-150 150"   (was ±80)
```

Operational requirements:

- **Must merge to `main`** — the Colab notebook clones `main` with no branch
  override.
- **Update the test pins in the same PR** or CI fails:
  `test_actuator_bounds.py` pins `FORCERANGE_KP_RATIO = 0.8` (±0.05) and
  asserts dynamic saturation floors (hip > 10%, ankle > 5%) that are designed
  to trip when headroom is gained.
- **Re-measure before spending GPU**: run `actuator_saturation_report.py` and
  target < 10% gait clipping on hips/ankles.

Decision rule: warm-started stage 2 on the raised-forcerange plant with the
March-proven config. If it tracks the March trajectory (no bimodal collapse,
speed well past 1 m/s by ~2M steps), the plant is confirmed as the regression
and the remaining work is gate calibration. If it still collapses identically,
suspect the reward/exploration side (R4) or the June/July training-code window
(Experiment D of the investigation).

### R3 — Measure top speed and recalibrate the gate

`min_avg_forward_vel = 2.0` was calibrated when the plant could do 3.5+ m/s.
Nobody knows what the bounded plant can do. Two cheap measurements:

1. Extend `actuator_saturation_report.py` to report achieved forward velocity
   for its scripted gaits (it already simulates them), sweeping frequency and
   amplitude — gives a scripted-gait speed floor for the plant.
2. Any uncensored RL run (R5) gives an empirical policy speed.

Then set the gate to ~60–70% of measured top speed. Precedent: the
brachiosaurus gate was cut 2.0 → 0.75 after its first stage-2 failure and its
single pass came at 1.12 m/s with 16M steps. If the bounded raptor plant tops
out near 1.5 m/s, a 2.0 gate guarantees failure no matter how good the
training is; keeping the gate honest is what makes the rest of the plan
falsifiable. If the plant fix (R2) restores 3+ m/s, keep the 2.0 gate.

### R4 — Cheap config changes for any bounded-plant rerun (Experiment C, corrected)

In `configs/velociraptor/stage2_locomotion.toml` (merge to `main`):

```toml
[env]
forward_vel_max = 2.5     # was 3.0 — saturate the speed incentive just past the gate
fall_penalty = -150.0     # was -50 — hedge for late falls; see corrected math in §2.1
                          #   (the old -250 target overstated the required penalty)

[ppo]
ent_coef_end = 0.001      # uncomment — action std grew 1.18→1.49 in BOTH collapses,
                          #   entropy still rising at stop
```

Keep `RobustBestModelCallback` defaults — it demonstrably picked the healthy
checkpoint this run. These are hedges, not a root-cause fix: the corrected math
(§2.1) shows the discounted objective only mildly favors fragile-fast gaits,
so expect these to *delay* the scissor, and rely on R2/R3 for the cure.

### R5 — Don't let the collapse detector censor experiments

`EvalCollapseEarlyStopCallback` (hardcoded `min_evals=8, patience=5,
drop_fraction=0.3`) is behaving correctly as a GPU-saver, but it has clipped
every bounded-plant run at 1.1M steps — before the reward ramp finishes — so
we have zero uncensored bounded-plant data. Two small changes
(`train_base.py:385`):

- Expose `min_evals` / `patience` / `drop_fraction` under `[curriculum]` in
  the TOML rather than hardcoding.
- For tuning runs, start the collapse window only after the reward ramp
  completes (`num_timesteps > ramp_timesteps`), so "mid-ramp transition" isn't
  judged as "collapse". At 50k cadence that means earliest stop ≈ 2.4M instead
  of 1.1M — one uncensored datapoint costs ~2h.

### R6 — Scale with the existing sweep infra once a single run survives

`configs/velociraptor/sweep_ppo.json` already has a stage-2 space (40 trials,
8M steps, sweeps `forward_vel_weight/max`, `alive_bonus`, posture, warmup/ramp,
PPO lr/entropy/batch) and warm-starts every trial from `LOAD_PATH`. Two gaps:

- `fall_penalty` is not in the search space — add `env_fall_penalty` if
  sweeping it matters after R4.
- Use the `ray_tune_sweep.ipynb` path, not `ray_orchestration.py` (zero
  callers, already diverged — see KNOWN_ISSUES).

Optional extra arm: SAC cleared stage 2 on the old plant (2.91 m/s,
`sac_20260321_170055`); auto-entropy sidesteps the manual entropy-decay tuning
and would be an informative second algorithm on the fixed plant.

### R7 — Ops / data-quality notes

- **Everything must land on `main`** before a Colab run picks it up (the July 9
  regression shipped exactly this way, 3 minutes before launch).
- **The npz artifacts on Drive fail `np.load`**: both July runs' archives have
  a 96-byte member-size defect (this analysis had to parse zip headers by
  hand; the final checkpoint's episode lengths are truncated). Run 1 predates
  the PR #432 atomic-write change, so the defect is either older or introduced
  by the Drive sync path — worth a quick `np.load` directly in Colab to
  bisect. Rewards/diagnostics were recoverable and complete.
- A rerun with zero changed variables at a fixed seed re-buys known
  information (this run's 4 GPU-hours produced bitwise-identical arrays).
  Change at least one variable per run — or vary the seed when the question is
  variance, not mechanism.

---

## 4. Suggested sequence

| # | Experiment | Cost | Question answered |
|---|---|---|---|
| 1 | Merge R2 (plant) + R4 (config) + R3 gate recalibration; warm-start stage 2 (R1) | ~1 PR + 2–5h GPU | Does headroom restore March-like locomotion? |
| 2 | If (1) collapses: same config on *current* plant with R5 relaxation | ~2h GPU | Uncensored bounded-plant speed ceiling + whether tuning alone stabilizes |
| 3 | If (1) passes but misses the gate: measure top speed (R3), set gate accordingly, rerun | ~30min + 5h GPU | Is the gate honest for this plant? |
| 4 | If a recipe survives 8M steps but sits near the gate: stage-2 sweep (R6) warm-started from stage 1 | ~40 trials | Best config within the surviving family |
| 5 | If (1) *and* (2) both collapse: bisect June/July training-code window (Experiment D of the investigation) | variable | Rules the harness in/out |

Artifacts referenced: Drive run folder
`mesozoic-labs/logs/velociraptor/ppo/20260711_165924/` (stage configs,
evaluations/diagnostics npz, videos), `logs/runs_summary.csv` and per-species
summaries (2026-07-11 export), repo state `da89959` on
`claude/stage-2-recommendations-un9fxn`.
