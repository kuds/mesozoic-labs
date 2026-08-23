# T-Rex Review — Run 20260821_142144, Merges #506–#510, Next Steps (2026-08-23)

**Scope**: repo at `1a0c048` (main, post-#510); Drive run
`logs/trex/ppo/20260821_142144` (all three stage folders: evaluations,
gate-progress, and diagnostics npz series, CSVs, gate reports, probes);
the combined diff of merged PRs #506–#509 plus direct commits `08a66b3`
(stance 10M→11M) and `0836043` (recovery 3M→5M); PR #510 (docs).
**Method**: every stage's npz series was decoded and re-analyzed
(the per-step re-analysis `TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md` §8
lists as not done); each code finding below was independently verified
three ways (code evidence, failure-scenario trace, design-intent check
against plans/comments/history) and only kept if at least two checks
confirmed it — 21 candidate findings reduced to 16 confirmed, 5 refuted.
**Validation**: full `environments/trex/tests` + `environments/shared/tests`
suite at HEAD: 2328 passed, 4 skipped, 0 failed;
`species_catalog --check` clean.

---

## 1. Run 20260821_142144 — what the data says

First full run of the merged stage-structure code (notebook path,
`RUN_RECOVERY_STAGE=True`), started hours after #509 merged. Colab, L4,
SB3 2.9.0, seed 42, 4 envs.

| stage | window (UTC) | steps | outcome |
|---|---|---:|---|
| `01_stance` | 08-21 14:21 → 08-22 03:42 | 11,001,856 | gate **PASS** |
| `02_recovery` | 03:42 → 08:39 | 5,005,312 | FAIL-by-design pilot (none/v1) |
| `03_locomotion` | 08:40 → ~14:17 | 5,488,640 of 8M | **interrupted** (~24 h Colab cap) |

### 1.1 Stance: certified, with margin

Gate panel (40 eps, seeds 3042–3081): reward 3431.8 ± 69.9 vs rail 2100;
full-horizon 40/40; unsupported duty 0.0055 / UCB95 0.0091 vs 0.02.
First in-training all-criteria pass at 6.75M; first 3-consecutive streak
at 6.85M; a streak ends exactly at the 10.8M `robust_best_model`
(3462.19 ± 36.36). The policy never beats the statue (3495.2) — by
design; gap 0.94% at best. Probe headlines: bandwidth knee between 2.5
and 5 Hz; constant-hold "Mixed" (pose needs feedback ~half the time);
no single actuator group necessary-and-sufficient; impulse recovery
envelope ≈ none (recovery is 1b's job). Panel duty mass concentrates in
4 of 40 seeds (3046: 0.0575, 3065: 0.055, 3066: 0.0275, 3055: 0.02125 —
also the 4 lowest rewards): natural stress seeds for later gates.

### 1.2 Recovery: the 5M pilot confirms the 3M pilot's shape

Warm-started from the stance handoff (`robust_best_model` + vecnorm,
`initialize_next_stage`). Eval curve (100 × 30 eps): first eval already
2297.91; **peak 3055.63 ± 208.86 at 1.6M**; trough 1910.45 ± 766.14 at
3.2M; final 2738.50 in-training, 2907.03 ± 463.98 publication (26/30
full episodes; selected model 2941.54 ± 422.41, 25/30). Only 53.8% of
all in-training eval episodes reach the 1000-step horizon; late
terminations shift toward nosedive (0.55 of last-decile terminations).
Reward variance is ~14× stance's — the pushes are the variance. The
collapse backstop armed at the first post-warmup eval and never fired
(post-warmup minimum 1910.45 vs floor 1572.8, margin +338).

Both budget-increase conclusions of the first-runs doc hold in this
data: the best checkpoint landed at 32% of budget, and the endpoint sits
~5% below the peak. Two additions from the diagnostics re-analysis:

* **The training-signal contamination in §3.4 below applies to this
  run** — `02_recovery/stage_config.json` records
  `ramp_start_value=0.1, ramp_timesteps=500000`, so the pilot's first
  500k steps trained with a forward-velocity incentive its own task
  fingerprint denies (eval/panels unaffected — they roll the declared
  task).
* **Entropy never decayed meaningfully** (`ent_coef_decay_timesteps=7M`
  against a 5M budget → ent_coef ≈ 0.0014 at stage end, above the 0.001
  floor stance's derivation record calls stand-still-unlearnable). The
  mid-run regression and the never-quiet policy are consistent with
  both; the schedule re-derivation (first-runs doc §6.3) is the fix.

### 1.3 Locomotion: interrupted while improving — but the gate was never reachable

The stage died with the Colab runtime (~23h55m after run start): last
checkpoint `stage2_5000000_steps.zip` at 13:49Z, best model saved
14:07Z, last npz flush 14:16Z. No `stage2_final`, no stage summary, no
figures/replays; run-root `artifact_manifest.json` is `partial`
(stance only). What the 109 evals (every 50k, 30 eps) show:

* Reward improving monotonically to the end: ~1100 flat through 2M
  (ramp window), then +~125/1M steps to **best 1494.47 ± 32.20 at
  5.3M**; last evals wobble down to 1435 with full-horizon fraction
  0.733 and fall terminations appearing — speed-induced late-episode
  falls, far above the collapse floor (729 effective).
* PPO internals healthy: explained_variance 0.05 → 0.986, approx_kl
  under target, LR exactly on its 8M schedule, no std inflation.
  Real gait developing: bilateral-support duty 0.97 → 0.54,
  unsupported 0.27, distance/episode up to ~2.2 m.
* **The advancement gate never had a single passing eval.** Reward,
  episode-length, and episode-count criteria passed 109/109; the
  2.0 m/s velocity criterion passed 0/109. Forward velocity grew
  0.02 → ~0.55 m/s (accelerating: 0.12 at 3M, 0.27 at 4M, 0.44 at 5M),
  but even optimistic extrapolation puts 2.0 m/s at or beyond the 8M
  budget. The 2.0 m/s value is a velociraptor copy
  (`configs/trex/locomotion.toml`, "matches raptor stage 2 gate") on a
  6× heavier plant, and `STAGE2_RECOMMENDATIONS.md` already notes it
  was calibrated when the plant could do 3.5+ m/s. `forward_vel_max=2.5`
  caps the speed incentive just past a gate the plant may not reach.

**Resume caution**: the two resume-path defects in §3.2 (vecnorm
sidecar naming; warm-up/ramp misapplied on `resume_same_stage`) both
fire on the natural resume command for this exact run. Resume from
`best_model.zip`/`robust_best_model.zip` (their `_vecnorm.pkl` sidecars
pair correctly) or fix the loader first; do not resume from the periodic
`stage2_5000000_steps.zip` until #3.2a is fixed.

---

## 2. Merges #506–#510 — shape of the change

#506–#509 (branch `claude/trex-stage1-review-1b-plan-fzkzy5`, +5528/−488
over 80 files) delivered the 1b machinery: perturbation engine (both
backends, capture-point-derived 165.5 N / 0.20 s on r7), task
fingerprints + load modes, the four-stage semantic manifest,
`recovery_quality/v1` + pushed-panel harness + frozen gate resolver,
NN_id run layout, and the two field-run fixes (cwd-independent
fingerprints #508; semantic-stage replay seeds #509). #510 is the
first-runs/P3-calibration record. The architecture held up in the field
on its first full run; the confirmed findings below are the residue.

## 3. Confirmed findings (16; each verified by ≥2 of 3 independent checks)

### 3.1 Gate-dispatch fail-opens (highest severity)

1. **`CurriculumManager.should_advance` fail-opens on
   `recovery_quality/v1`** (`environments/shared/curriculum/manager.py:250`,
   HIGH). The dispatch has arms for `none/v1` and the stance gate; every
   other schema-valid kind falls into the reward-and-length evaluator
   with `StageThreshold` defaults (`min_avg_reward=-inf`).
   `reporting/gates.py:275-285` explicitly refuses this exact
   fall-through; the live advancement engine is the fail-open side. The
   moment P5 flips `recovery.toml`'s gate kind, the SB3 curriculum would
   advance the pushed stage on reward alone. Fail closed on unknown kinds.
2. **JAX `check_stage_gate` has the same hole**
   (`environments/shared/jax_curriculum.py:183`, LOW today): a
   `recovery_quality/v1` stage either crashes with a bare `KeyError`
   after spending its whole budget, or (with the optional reward rail
   present) advances on reward alone.

### 3.2 Resume path (blocks resuming the interrupted locomotion run)

3. **Periodic-checkpoint VecNormalize sidecar never found**
   (`environments/shared/train_base.py:327`, MEDIUM). Loader probes
   `<base>_vecnorm.pkl`; SB3's CheckpointCallback writes
   `<prefix>_vecnormalize_<steps>_steps.pkl` (the run's Drive folder
   shows exactly this pair). Resuming from `stage2_5000000_steps.zip`
   warns, then trains the loaded policy under fresh normalization
   statistics — collapsed behavior, silently.
4. **Same-stage resume applies stage-entry shaping**
   (`environments/shared/train_base.py:827`, MEDIUM). The
   warm-up/ramp block keys on `stage_position > 1 and load_path`, not
   on `task_load_mode`, so a `resume_same_stage` resume that just
   passed its exact-fingerprint check trains ~500k steps on a
   ramp-modified task the fingerprint claims is unchanged.

### 3.3 Task-identity integrity

5. **Notebook recovery training injects a forward-velocity ramp the
   fingerprint denies** (`notebooks/sb3_training.ipynb` train_stage
   cell, HIGH). `train()` has the guard with a comment naming exactly
   this hazard (`train_base.py:838-849`); the notebook copy does not.
   Confirmed in the field: the 20260821 recovery pilot trained its
   first 500k steps with `forward_vel_weight` ramping 0.1 → 0.0.
   `train_curriculum` (`train_base.py:1351-1360`) also lacks the guard.
6. **Fingerprints hash only TOML-present `[env]` keys**
   (`environments/shared/task_fingerprint.py:122`, MEDIUM).
   Constructor-default task parameters (e.g. TRexEnv `healthy_z_range`)
   are outside task identity: retuning a default would let
   `resume_same_stage` silently resume across a real
   transition-kernel change. Hash the effective config
   (defaults + TOML, as `save_stage_config` already computes).

### 3.4 Schedule/config coherence

7. **Recovery entropy decay horizon exceeds its budget**
   (`configs/trex/recovery.toml:94`, HIGH). `ent_coef_decay_timesteps=7M`
   (a stance 11M-budget derivation) against a 5M stage: ent_coef never
   drops below ~0.0014, above the 0.001 floor stance's own record calls
   incompatible with quiet standing — while the recovery task's success
   event is quiet stance between pushes. Fold into the first-runs doc's
   §6.3 schedule re-derivation; nothing currently tests
   `decay ≤ budget`.
8. **Stale derivation records after the two budget bumps** (LOW):
   `recovery.toml:80-82` ("3M warm-started budget") and `:140`
   ("1/3 of this stage's 3M budget" — now 1/5 of 5M; the companion test
   tolerates any warmup ≤ 2.5M so it cannot flag this);
   `stance.toml:236` ("70% of the 10M budget" — 7M is 64% of 11M) and
   `:272` ("the budget is now 10M"). In a derivation-record discipline,
   wrong records are the documented mechanism for future
   mis-derivations.

### 3.5 Semantic-stage stragglers (recovery is the stage that loses)

9. **`plot_diagnostics_graphs` crashes on semantic stage ids**
   (`environments/shared/visualization.py:279`, HIGH):
   `stage_num % 10` raises TypeError for `"recovery"` inside the
   one-big-try artifact wrapper — which is why `02_recovery` got one
   figure where stance got five. Costs every recovery run its
   locomotion-health/behavioral/foot-contact/stance-diagnostics figures.
10. **CLI `--stage locomotion` (etc.) crashes without `--timesteps`**
    (`environments/shared/cli.py:226`): the default-timesteps lookup
    runs before semantic-id resolution; raw `KeyError` where the code
    promises "`--stage locomotion` works wherever `--stage 2` does".
11. **Eval stage auto-detect misclassifies recovery checkpoints from
    single-stage run dirs as stage 1**
    (`environments/shared/evaluation.py:379`) and the eval subcommand's
    `--stage` is `int, choices=[1,2,3]`, so it cannot be overridden —
    a recovery checkpoint gets silently evaluated on the un-pushed
    stance task.
12. **`google_drive_summary` iterates legacy stages (1,2,3) only**
    (notebook `scan_run`): `02_recovery` produces no row in any run
    summary, and a recovery-only run dir is not recognized as a run.
13. **W&B run name mints `trex-stagerecovery`**
    (`environments/shared/wandb_integration.py:96`): tags were converted
    to `stage_label`, the name was missed.

### 3.6 Coverage and docs

14. **MJX perturbation tests never run in any CI job**
    (`.github/workflows/python-ci.yml:425`, MEDIUM): the shared-test job
    has no jax so they importorskip; the jax-cpu job enumerates files and
    omits `test_perturbation.py`. The traced push path — including the
    NumPy/JAX schedule parity the paired null statistics rely on — ships
    unverified. One-line fix: add `test_perturbation.py` to the jax-cpu
    job's file list.
15. **`hyperparameters.md` documents `stance.toml`/`locomotion.toml`
    filenames for species that still use `stage1_*`/`stage2_*`**
    (only trex was renamed), says "three stage configs" while trex has
    four, and omits `recovery.toml`.
16. **JAX writers still emit the pre-manifest generation**
    (`jax_setup.py:372` `stage{N}` dirs;
    `stage_artifacts.py:1379` `stage{stage}_final.pkl` — would produce
    `stagerecovery_final.pkl` the moment the JAX path gains semantic
    stages). Harmless today; a landmine for the MJX recovery run the
    first-runs doc lists as not-done.

### 3.7 Verified-and-cleared (5)

Adversarial verification refuted five plausible candidates — recorded so
they are not re-found: the judged-push truncation filter in
`recovery_evaluation.py` (correct as written); MJX/fingerprint keyframe-0
vs SB3 reset keyframe (consistent); full-precision float hashing in
fingerprints (deliberate, with the dated transition valve);
`current+verified` provenance reachability (reachable); and the
config-guardrail tests' stage parametrization (recovery keys are
covered elsewhere).

---

## 4. Next steps, in priority order

The first-runs doc's §6 list stands; items below merge it with this
review's findings. "F#" = finding above.

**A. Cheap experiments that unblock everything else**

1. **Off-distribution schedule test** (first-runs §6.1, ~10 min).
   Unchanged as the highest-value item: schedule memorization vs
   reactive control decides the stage's sim-to-real worth and blocks P5.
   Roll the frozen panel at interval 3.5 s ± 1.5 s and at 120 N / 210 N.
2. **Panel the 5M recovery checkpoint** (first-runs §6.2) against the
   3M policy's 21/40 under the calibrated posture-only safe set. The 5M
   eval curve (peak at 1.6M, trough at 3.2M) predicts it will not beat
   the 3M policy — worth confirming cheaply before any schedule work.
3. **Decide the locomotion gate before resuming the run** (§1.3). The
   run produced 5.4M steps of healthy learning against a velocity gate
   that passed 0/109 evals and extrapolates as unreachable within
   budget. Re-derive the speed target from the trex plant (statue-style
   measured reference + margin, as stance did for reward), or
   consciously accept a long-budget stage. Resuming first spends
   ~2.6M steps against a gate already known not to bind.

**B. Fixes before the next training run** (small, each maps to a finding)

4. Resume path: fix the vecnorm sidecar probe for periodic checkpoints
   (F3) and key stage-entry shaping on
   `task_load_mode == "initialize_next_stage"` (F4). Both fire on the
   natural resume of the interrupted locomotion stage. Until then,
   resume only from `best_model`/`robust_best_model`.
5. Guard the notebook's `RewardRampCallback` with the same
   `forward_vel_weight > 0` condition as `train()` (F5), and add it to
   `train_curriculum`.
6. Fail closed on unknown gate kinds in `CurriculumManager.should_advance`
   and `jax_curriculum.check_stage_gate` (F1, F2) — prerequisite for P5
   flipping the key safely.
7. Re-derive the recovery warm-started schedule (first-runs §6.3 + F7):
   entropy/LR for a ~2M-class warm-started budget, decay anchored to the
   stage budget, plus a test relating `ent_coef_decay_timesteps` to
   `timesteps`.

**C. P5 and the instrument's paper trail**

8. **P5**: after A1/A2, freeze `gate_resolution.json` (calibrated
   posture-only safe set; frozen statue+brace nulls; the
   `min_recovery_success_lcb` decision the first-runs doc frames) and
   wire the resolver into a production entry point — today
   `build/write/evaluate_gate_resolution` have test-only callers.
9. Re-derive the recovery collapse floor pair from the pushed statue
   (974.7 ± 216.6) per first-runs §6.5, and fix the four stale
   derivation records (F8).
10. Restore recovery's lost instrumentation: the visualization crash
    (F9), Drive-summary blindness (F12), eval misdetection (F11), CLI
    resolution (F10), W&B name (F13). These are why the 5M pilot has
    one figure and no summary row.
11. Add `test_perturbation.py` to the CI jax-cpu job (F14).

**D. Structural, when the above is banked**

12. Fingerprint the effective env config (defaults + TOML) with a dated
    transition valve (F6).
13. Integer-stage bundle/catalog migration so recovery runs join result
    bundles (first-runs §8; also fixes "recovery excluded by design").
14. Bring JAX writers onto `stage_label`/NN_id (F16) ahead of any MJX
    recovery run; the MJX/SB3 push parity tests must be running in CI
    (item 11) first.
15. Decide whether locomotion should eventually warm-start from the
    gated recovery checkpoint rather than stance (today recovery's
    policy is discarded by design of the pilot); revisit once the gate
    can certify one.
16. Colab operations: three earlier attempts died at the stance→recovery
    boundary and this run died at the ~24h cap mid-locomotion. With the
    resume fixes (item 4), a documented resume-from-latest-checkpoint
    cell turns the cap from a run-killer into a checkpoint boundary.
