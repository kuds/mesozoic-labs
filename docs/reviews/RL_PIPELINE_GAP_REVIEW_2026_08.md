# RL Pipeline Gap Review — 2026-08-28

**Status: findings record. Nothing in this document has been fixed; no implementation has started.**
Reviewed at commit `ea0d339` (the P5 recovery-gate freeze) on branch
`claude/trex-review-next-steps-kn9rcj`. This document is the deliverable of the 2026-08-28
"review the RL pipeline, notebooks, stages, and gaps + cleanup opportunities" request.

## 0. Scope and method

Three instruments, all run against the clean tree at `ea0d339`:

1. **Eight-area survey.** Independent read-only reviewers over: the SB3 training core, all four
   notebooks, environment/physics, evaluation/reporting, the JAX/MJX stack, configs and sweep
   search spaces, tests/CI, and operational scripts. Reviewers ran code (smoke probes, targeted
   pytest, runtime experiments) rather than guessing where a claim was cheaply testable.
2. **Completeness critic + four cross-cutting probes.** A critic asked what the area survey
   structurally missed, then spawned probes for: end-to-end execution of the real CLI entry
   point, the Colab session-cap/ungraceful-disconnect workflow, single-seed decision validity,
   and resume/multi-session provenance blindness.
3. **Four-lens cleanup sweep** (dead code, duplication, stale markers, structure), with
   reference searches across `environments/`, `configs/`, `notebooks/` (cell sources), `docs/`,
   `website/`, `scripts/`, `.github/`, and runtime import checks.

**Verification protocol.** Every candidate finding was handed to an independent adversarial
verifier instructed to refute it: reproduce the quoted code verbatim, re-run the reference
searches, check callers/guards/tests the finder may have missed, and test the claim at runtime
where feasible. Results: **84 pipeline findings confirmed** (4 candidates refuted — Appendix A)
and **36 cleanup opportunities confirmed** (2 refuted). Verifier-corrected severity and wording
are used throughout; where a verifier narrowed a claim, the narrowed version is what is
recorded here. Where two finders independently hit the same defect, the entry is recorded once
with a cross-reference.

The P5-touched files (`recovery_evaluation.py`, the recovery harnesses, `reporting/gates.py`,
`curriculum/manager.py`, `jax_curriculum.py`, `configs/trex/recovery.toml`) were excluded from
the automated survey because they were being written during it; they were reviewed by hand at
integration time instead (§5), and the end-to-end probe exercised the merged result.

Severity calibration: **high** = would corrupt results, waste a training run, or silently
invalidate evaluation; **medium** = would mislead or cost significant debugging; **low** =
papercut. Fix effort: small / medium / large.

---

## 1. Headline: the resume path corrupts the very runs that need it

The single most consequential result of this review, because it changes what the next action on
run `20260821_142144` should be.

At observed L4 throughput, stance (11M steps) and locomotion (8M) each exceed the ~24 h Colab
session cap — the interruption of run `20260821_142144` at 5.49M/8M was not bad luck, it is the
steady state. **Every production run is therefore a multi-session run, and the resume path is
mandatory infrastructure.** That path has five independent confirmed defects that compound, plus
a provenance layer that cannot record any of it:

1. **The step counter restarts at zero on every resume** (TC1). `model.learn()` is called with
   SB3's default `reset_num_timesteps=True` everywhere. Post-resume checkpoints are renamed from
   the stride down, `CheckpointRetentionCallback` — which keeps the highest filename step counts —
   **deletes each fresh checkpoint the moment it is saved** while retaining the stale pre-crash
   set, and a second disconnect makes the resume cell's glob-max reload the stale pre-crash
   policy, silently discarding all post-resume compute. Verified end to end with a runtime probe.
2. **Best-model trackers restart at −inf** (TC2). The first post-resume evaluation
   unconditionally overwrites `best_model.zip`, `robust_best_model.zip` and their vecnorm
   sidecars, even when it scores far below the pre-crash best; `evaluations.npz` is replaced
   with only the post-resume slice; next-stage handoff prefers exactly the overwritten artifact.
3. **Hyperparameter schedules snap back** (TC3). Config-fresh kwargs passed into `PPO.load`
   overwrite the pickled decayed `ent_coef`; LR/clip schedules re-anchor to their start values
   and re-anneal over only the remaining budget. A resumed run silently trains under different
   hyperparameters than an uninterrupted one — the exact late-training destabilisation the
   stance.toml schedule comments exist to prevent.
4. **A checkpoint saved during stage-entry warm-up permanently poisons entropy decay** (EE3).
   The pickled warm-up marker is restored on resume, no warm-up callback runs to clear it, and
   `EntCoefDecayCallback` defers forever — observed live in the end-to-end probe, and the stale
   marker re-pickles into every descendant checkpoint.
5. **The CLI resume path has no remaining-budget arithmetic and a lenient sidecar guard**
   (TC4, TC5). `--load` with no `--timesteps` retrains the full TOML budget on top of the
   checkpoint's steps; a missing VecNormalize sidecar is a warn-and-continue on the CLI (the
   notebook correctly hard-fails on the identical condition).

Around this sit the session-boundary hazards (§4.2: no Drive flush for checkpoints on ungraceful
reclaim, a resume cell with no integrity fallback, ~2.2 h of re-trained compute per boundary from
`save_freq=500000`) and the provenance blindness (§4.4: a resumed run audits canonical-valid as a
clean single-session run; legitimate resumes are refused outright when anything in the unpinned
environment drifted overnight).

**Practical consequence, stated plainly: do not resume the interrupted locomotion run — and do
not start any stage whose budget exceeds one session — until at least TC1, TC2, TC3, EE3 and
TC5 are fixed.** The resume walkthrough given earlier in this session (notebook §7b /
`--load` CLI) is mechanically correct but lands on all five defects; it certifies the resumed
run's artifacts as something they are not. The fixes are individually small (one `learn()`
argument, seeding two trackers, load-kwargs hygiene, one marker clear, one fail-closed guard).

## 2. What blocks the next training run

In dependency order. Item groups reference the numbered findings in §3–§4.

**Before any resume of `20260821_142144` (or any multi-session stage):**
- TC1 — pass `reset_num_timesteps=False` on `resume_same_stage`; make retention refuse to prune
  checkpoints newer than training start.
- TC2 — seed `EvalCallback.best_mean_reward` / `RobustBestModelCallback.best_score` from existing
  artifacts; make `PublishEvalArtifactsCallback` append rather than replace `evaluations.npz`.
- TC3 — stop passing schedule-bearing kwargs into `load()`; anchor schedules to the full stage
  budget on resume.
- EE3 — clear the warm-up marker on `resume_same_stage` loads.
- TC5 — fail closed on a missing vecnorm sidecar in the CLI path (mirror the notebook).
- CO1/CO2 — resume-cell integrity fallback (skip orphaned/truncated newest pair) and a
  checkpoint-cadence Drive flush; CO4 — reconsider `save_freq` (500k ≈ 2.2 h re-trained per
  boundary; retention has made the storage rationale moot).
- RP5/CO3 — split the provenance identity check so environment drift appends a session record
  instead of refusing the resume.

**Before the recovery re-run at 3M under the frozen gate:**
- EE1 — recovery stages currently train with zero gate-relevant diagnostics (no
  `gate_progress.npz`, no `diagnostics/eval_*` tags): move the panel recording above the
  legacy-threshold early-return.
- EE2 — `_apply_stage_gate` forwards neither `stage_dir` nor panel evidence, so the frozen
  gate resolution is unreachable from the pipeline's one shared call site: plumb both, and add
  the post-stage panel roll that produces `recovery_successes_by_seed`.
- Freeze the actual `gate_resolution.json` (hand run of
  `python -m environments.shared.harnesses.freeze_recovery_gate --stage-dir <dir>`, ideally with
  a checkpoint so the brace null joins the manifest) — see §5.
- NB6/NB7 — PPO.load preflight at notebook setup; pin the pip installs (the current L4 segfault
  is precisely an unpinned-environment failure).

**Before trusting evaluation numbers used in decisions:**
- ER1 — `evaluate()` misses the periodic-checkpoint vecnorm sidecar naming and scores raw
  observations (also independently found by the CI finder; same class as OP7 in three report
  scripts).
- ER3 — `evaluate()` is the one unseeded evaluation path in the repo.
- ER4 — sweep trials fabricate velocity/success as 0.0 into gate columns.
- OP2 — the sweep pipeline evaluates pass/fail from retired legacy threshold keys (a trex
  stance sweep is gated reward-only, which the statue clears).
- CF2/SS2 — the behavior stage's 2.0 m/s episode-average gate is structurally unreachable in
  bite episodes, and its 0.5 success gate at n=30 has 43% false-block at threshold.
- SS1 — nothing can even express multi-seed certification; single-seed decisions recur by
  default (measured 2/3 stance replication).

**CI hardening (cheap, high leverage):**
- CI3 — the resume-safety tests exist but run in no CI job (sb3-gated, silently skipped).
- CI4 — the MJX parity tests are excluded from the jax job; MJX reward drift merges green.
- CI7 — root `conftest.py` (and workflows, docs subdirs) are outside the path filters.
- CI9 — notebooks trigger CI but nothing validates them; AST-parse all four.

**Before any MJX pilot on trex:** JX1 (eval composer omits the jerk term), JX5 (GAE truncation
bootstrap bias in the synchronized-reset regime), JX2 (JAX notebook resume discards opt_state),
JX6 (alive-gate/fall-penalty asymmetry vs the shared gate bar), JX3 (recovery stage
unreachable end to end on MJX — precise inventory inside).

---

## 3. Findings by area

### 3.1 Training core (SB3)

*Scope: environments/shared/train_base.py, cli.py, and the curriculum callbacks.*

**TC1. `environments/shared/train_base.py:933` — resume-safety [high / fix small]**

Resume (resume_same_stage via CLI --load or the notebook resume cell) calls model.learn() with SB3's default reset_num_timesteps=True, so the step counter restarts at 0: post-resume periodic checkpoints are renamed from the stride down, CheckpointRetentionCallback (which keeps the max_checkpoints largest filename-parsed step counts) deletes each fresh checkpoint on arrival while retaining stale pre-crash ones, and a second interruption makes the max-by-name resume cell reload the stale pre-crash policy — silently discarding all post-resume training and mis-deducting the remaining budget. Filename 'steps' also stops meaning cumulative steps, colliding with/overwriting pre-crash names once the new counter overlaps them.

*Bites when:* Probe-verified with prune_periodic_checkpoints: stage crashes at 3.7M with retained checkpoints stance_{1.5M..3.5M}_steps.zip; the notebook resume cell (or CLI --load) restarts numbering at 0, the first new checkpoint lands as stance_500000_steps.zip, and the retention callback immediately deletes it (kept set is the 5 largest step numbers, all stale). Until renamed steps exceed 1.5M the run has no surviving fresh checkpoint; a second Colab disconnect then makes the resume cell pick max-by-name stance_3500000_steps.zip -- the SAME stale pre-crash policy -- silently discarding all post-resume training and recomputing remaining budget as budget-3.5M. Where new names collide with old ones, pre-crash checkpoints are also silently overwritten, so 'steps' in filenames no longer means cumulative steps.

*Suggested fix:* On resume_same_stage, pass reset_num_timesteps=False to model.learn (SB3 then continues num_timesteps and adds total_timesteps to the counter), or offset CheckpointCallback naming by the loaded checkpoint's step count; at minimum make CheckpointRetentionCallback refuse to prune files newer by mtime than training start.

**TC2. `environments/shared/curriculum/checkpoints.py:63` — resume-safety [high / fix medium]**

On same-stage resume (notebook cell 29 / --load with resume_same_stage into the same stage dir), EvalCallback and RobustBestModelCallback are constructed fresh with -inf best trackers and no seeding from existing artifacts, so the first post-resume evaluation unconditionally overwrites best_model.zip, robust_best_model.zip and their vecnorm sidecars — even when the loaded periodic checkpoint scores below the pre-crash best — and PublishEvalArtifactsCallback replaces the stage's evaluations.npz with only the post-resume slice, whose timesteps restart near 0 (learn() uses default reset_num_timesteps=True). _select_handoff_checkpoint then promotes the possibly-degraded checkpoint and _record_stage_result/_report_hpt_metrics report best/last metrics from the truncated record. One caveat vs. the finder: obs_rms is carried forward from the sidecar, so the first post-resume eval reflects the loaded checkpoint's real quality rather than a normalization "shock" — the damage arises because the newest periodic checkpoint can genuinely be worse than the historical best (the run-20260709 bimodal-degradation pattern that motivated RobustBestModelCallback), and best_model artifacts partially self-heal only if the resumed slice later exceeds the old best, while the npz history loss is unconditional and silent.

*Bites when:* A stance run peaks at eval mean 2300 (robust_best saved), Colab disconnects, user resumes into the same stage dir per notebooks/sb3_training.ipynb cell 29. The first post-resume eval scores 900 (warm normalization shock, restarted schedules): EvalCallback and RobustBestModelCallback both see -inf best and overwrite best_model.zip/robust_best_model.zip + vecnorm with the 900-scoring policy. _select_handoff_checkpoint then promotes this degraded checkpoint to the next stage, and _report_hpt_metrics/_record_stage_result compute best/last metrics from the truncated post-resume evaluations.npz whose timesteps also restart at 0 -- the run's record no longer contains its actual best.

*Suggested fix:* On resume, seed EvalCallback.best_mean_reward and RobustBestModelCallback.best_score from the existing evaluations.npz / saved artifacts (or score the loaded checkpoint once before training), and have PublishEvalArtifactsCallback append to rather than replace the prior npz.

**TC3. `environments/shared/train_base.py:417` — resume-safety [high / fix medium]**

On any same-stage resume (--load or the notebook resume cell), train_base.py:417 passes config-fresh alg_kwargs into alg_cls.load, so SB3's kwargs-after-data update overwrites the pickled decayed ent_coef (back to 0.005) and re-anchors the LR/clip schedules to their config start values; with reset_num_timesteps left at SB3's default True, progress_remaining restarts at 1.0 (LR snaps from ~1.3e-5 back to 3e-5 and re-anneals over only the current call's budget) and EntCoefDecayCallback's num_timesteps counter restarts at 0 against the config's absolute 7M decay anchor, so a late-stage resume trains its remaining steps at 2-7x the entropy an uninterrupted run would have — exactly the late-training destabilisation the stance.toml schedule comments exist to prevent — and resumed runs are silently non-comparable to uninterrupted ones.

*Bites when:* T-Rex stance (configs/trex/stance.toml: learning_rate 3e-5 -> 1e-5, ent_coef 0.005 -> 0.0 over 7M steps) crashes at 8M of 11M with lr ~1.3e-5 and ent_coef ~0.0007. Resume via --load or the notebook cell: lr jumps 2.3x back to 3e-5 and re-anneals over only the remaining 3M; ent_coef jumps back to 0.005 and, with num_timesteps restarting at 0 against decay_timesteps=7M, never decays below 0.0028 for the rest of the stage -- exactly the late-training destabilisation/consolidation failure the TOML comments say these schedules exist to prevent. A resumed run is silently trained under different hyperparameter trajectories than an uninterrupted one, confounding seed/config comparisons.

*Suggested fix:* For resume_same_stage: keep num_timesteps (reset_num_timesteps=False) so progress_remaining continues, anchor schedules to the full stage budget rather than the current call's total, do not pass ent_coef into load() (keep the pickled decayed value), and give EntCoefDecayCallback the absolute step offset.

**TC4. `environments/shared/cli.py:259` — budget-waste [medium / fix small]**

CLI resume (train --stage N --load <periodic checkpoint>) with no --timesteps defaults to the stage's full TOML budget (cli.py:259) and train() calls learn() with reset_num_timesteps=True, so an interrupted stage trains checkpoint-steps + full-budget total experience (e.g. 8M+11M=19M for stance) while metrics.json reports only the 11M segment; only the notebook resume cell computes remaining = max(budget - steps, 0). Mitigated slightly by the CLI logging its chosen timesteps value and by --timesteps being explicitly settable.

*Bites when:* Stance crashes at 8M of its 11M budget. 'train_sb3.py train --stage 1 --load .../stance_8000000_steps.zip' passes fingerprint validation (resume_same_stage) and trains 11M MORE steps (19M total, ~8 extra GPU-hours) with no warning; metrics.json and curriculum comparisons then mix 11M-budget and 19M-budget runs. Only the notebook resume cell computes remaining = budget - steps; the CLI, the documented resume path for non-Colab runs, does not.

*Suggested fix:* When load_path matches the periodic _PERIODIC_CHECKPOINT_RE pattern and task_load_mode=resume_same_stage, default total_timesteps to max(budget - parsed_steps, 0) and log it, mirroring the notebook's arithmetic.

**TC5. `environments/shared/train_base.py:377` — silent-failure [medium / fix small]**

In the CLI launch path, a --load resume whose VecNormalize sidecar is missing (e.g. preemption between CheckpointCallback's zip and pkl writes, or a lost buffered Drive write) logs a single warning whose text claims only the eval env is affected, then trains the loaded policy under fresh mean-0/var-1 obs_rms with training=True — mis-scaled observations that degrade the policy. The notebook resume cell (sb3_training.ipynb cell 29) hard-fails on the identical condition, so the guardrail exists on only one of the two launch paths; a repo test (test_resume_load_path.py::test_missing_sidecar_still_warns_and_resets_eval_env) pins the current lenient behavior and must change with the fix.

*Bites when:* SB3's CheckpointCallback writes the .zip then the vecnormalize .pkl; a Colab preemption between the two leaves a periodic zip with no sidecar. A CLI resume --load of that zip warns once (and the message misleadingly says only the EVAL env uses defaults) then trains the policy under obs_rms mean-0/var-1 stats: observations are wildly mis-scaled relative to the weights, the policy collapses in the first updates, and per finding 2 the collapsed evals overwrite best_model/robust_best_model. notebooks/sb3_training.ipynb cell 29 raises FileNotFoundError for the identical condition ('refusing to resume under fresh normalization statistics'), so the guardrail exists but only on one of the two launch paths.

*Suggested fix:* In _load_vecnorm_into_envs, when load_path is set but the resolved sidecar does not exist, raise (or require an explicit --allow-fresh-vecnorm) instead of warn-and-continue; also fix the warning text, which understates that the TRAIN env keeps fresh stats.

**TC7. `environments/shared/train_base.py:1286` — curriculum-integrity [medium / fix small]**

train_curriculum (train_base.py:1286) iterates hardcoded range(1, 4) over stage_configs that load_all_stages keys as [1, 'recovery', 2, 3] for trex, so the 'curriculum' CLI command trains stance -> locomotion -> behavior, handing the stance handoff checkpoint straight to locomotion and never training recovery. The 3-stage scope is hinted in the CLI help text ("stages 1-3") and docstring, but at runtime the loaded 'recovery' config is silently ignored (thresholds_from_configs also drops it without error) with no skip log, and nothing will flag the omission when recovery becomes an advancing stage.

*Bites when:* load_all_stages('trex') returns keys 1, 'recovery', 2, 3 per configs/trex/stages.toml (recovery has no legacy_number). 'train_sb3.py curriculum' therefore trains stance -> locomotion -> behavior with no push-recovery training and no error or log line saying a manifest stage was skipped, while task lineage records a clean 1->2 boundary. Anyone treating the curriculum command's output as 'the full T-Rex curriculum' ships a policy that never saw the 165.5 N scheduled pushes. Recovery is currently a non-advancing pilot (notebook cell 20), so the omission may be transitional -- but it is silent, and the loop will remain wrong the day recovery becomes advancing.

*Suggested fix:* Iterate the stage manifest's curriculum order (load_stage_manifest(species).stages) instead of range(1, 4), with an explicit skip-with-log for stages marked non-advancing/pilot.

**TC8. `environments/shared/train_base.py:673` — resume-safety [medium / fix medium]**

A crash inside a boundary-crossed stage's entry-shaping window (warm-up 100k / reward ramp 500k steps) cannot be resumed continuously: the resume_same_stage path (train_base.py:673, a deliberate F4 fix pinned by tests) attaches no shaping callbacks, the env is rebuilt at the full config forward_vel_weight, reset_num_timesteps defaults to True so the counter cannot recover ramp position, and no sidecar/lineage/log records shaping progress — so the policy gets the remaining ramp delta and the removal of warm-up clamps in one step, silently.

*Bites when:* Locomotion is entered from a recovery/stance checkpoint with RewardRampCallback ramping forward_vel_weight 0.1 -> 1.0 over 500k steps; Colab disconnects at 200k (weight ~0.46). The resume (correctly resume_same_stage, exact fingerprint match) rebuilds the env from env_kwargs at the FULL weight with no ramp and no warm-up clamp, so the policy that was being eased into the walk incentive gets the remaining 0.54 of the weight in one step -- the 'catastrophic gradient updates that overwrite previously learned balance' the ramp exists to prevent, compounded by finding 3's restarted LR. Ramp/warm-up position is persisted nowhere, so no resume can continue it.

*Suggested fix:* Record ramp/warm-up progress (e.g. in the checkpoint's task lineage or a sidecar) and re-attach the shaping callbacks with an offset when a resume lands inside the entry-shaping window; or at minimum log loudly that a mid-ramp resume is discarding the remaining ramp.

**TC6. `environments/shared/curriculum/checkpoints.py:269` — resume-safety [low / fix small]**

load_vecnorm_stats (checkpoints.py:269) unconditionally discards ret_rms, and _load_vecnorm_into_envs (train_base.py:355) never passes task_load_mode, so a resume_same_stage resume re-learns return normalization from scratch even though the correct ret_rms sits in the loaded .pkl. The practical effect is a short transient (roughly the first ~100 steps per env of the first rollout: one clipped reward per env and ~30% inflated normalized-reward scale measured empirically), not a sustained value-loss shock — SB3's RunningMeanStd fresh count of 1e-4 makes it re-adapt almost immediately, and the identical reward distribution means it re-converges to the same statistics. A resume-fidelity papercut worth the small fix (carry ret_rms when task_load_mode == 'resume_same_stage'), not a training-run hazard.

*Bites when:* Resume of stance mid-stage: raw returns are in the thousands; with ret_rms reset, the first post-resume rollouts' normalized rewards are computed against var~1 and clipped at DEFAULT_CLIP_REWARD=50 (constants.py), a very different scale than the one the loaded critic was trained on, producing a value-loss/advantage shock during exactly the updates that finding 3 already runs at restarted (high) LR and entropy. The rationale in the docstring ('reward distribution changes between curriculum stages') is correct for initialize_next_stage boundaries but is applied to same-stage resumes too; the discarded ret_rms is sitting in the very .pkl being loaded.

*Suggested fix:* Add a carry_ret_rms flag to load_vecnorm_stats (copy prev_norm.ret_rms and returns when set) and pass it from _load_vecnorm_into_envs when task_load_mode == 'resume_same_stage'.

**TC9. `environments/shared/curriculum/baseline_watch.py:202` — silent-failure [low / fix small]**

Real but narrower than it may sound: the anchoring mismatch only manifests when the plain --timesteps flag (or a shortened resume budget) diverges from the TOML — the default no-flag launch, an --override curriculum.timesteps=... launch, and the train_curriculum() path all stay consistent, and the callback is advisory-only (a mistimed or missing warning, never a wrong training outcome). Fix is threading train()'s total_timesteps through _build_core_callbacks into build_baseline_progress_callback (and optionally the early_stopping.py:370 sanity log).

*Bites when:* A resumed stance run with 3M remaining of an 11M budget: num_timesteps restarts at 0 and only ever reaches 3M, below 0.35*11M=3.85M, so the never-beaten-baseline warning -- the guardrail built specifically because run 20260804_143747 burned its whole budget below the statue unnoticed -- can never fire for the resumed portion. Conversely a --timesteps 20M exploratory run on a 500k-TOML stage warns at 175k steps, crying wolf. Same anchoring mismatch applies to the collapse warm-up sanity check at early_stopping.py:370 (budget from TOML while --timesteps may differ).

*Suggested fix:* Thread the actual total_timesteps passed to train()/model.learn into build_baseline_progress_callback instead of re-reading the TOML value.

**TC10. `environments/shared/cli.py:63` — silent-failure [low / fix small]**

_apply_overrides has no stage-scoped syntax for semantic-id stages: 'recovery.*' or '1b.*' keys miss the isdigit()-gated stage branch, fall into the all-stages branch, and crash with an opaque KeyError ('recovery_kwargs'/'1b_kwargs') at launch. For single-stage runs the two-part all-stages form (--stage recovery --override env.push_force=200) is an effective workaround since train() reads only the selected stage's config; the every-stage-contamination hazard is real only for the curriculum command. The numeric stage-scoped branch also silently ignores unknown stage numbers.

*Bites when:* Probe-verified: --override recovery.env.push_force=200 partitions to section='recovery', producing kwargs_key='recovery_kwargs' and KeyError: 'recovery_kwargs' on the first stage config -- an opaque crash at launch. There is no working syntax to target only the recovery stage (the stage-scoped branch requires parts[0].isdigit()), so a push-magnitude sweep over stage 1b must either edit the TOML or apply env overrides to all four stages, silently changing every stage's task fingerprint. Adjacent to but distinct from the known 'sweep launch CLI accepts int-only stage refs' issue -- this is the training CLI's own parser.

*Suggested fix:* In the stage-scoped branch, accept any first token that is a key in configs (or resolves via the stage manifest) rather than only digits, and error loudly on an unrecognized section instead of KeyError.

### 3.2 Notebooks

*Scope: all four notebooks, cross-checked against the code they call.*

**NB1. `notebooks/sb3_training.ipynb:922` — resume-safety [high / fix small]**

The 7b resume path (and train_base's learn calls) uses SB3's default reset_num_timesteps=True, so a resumed session re-numbers periodic checkpoints from 0; CheckpointRetentionCallback keeps the 5 highest step numbers, so every resume checkpoint below the stale maximum is deleted the step it is saved (and any that tie a stale step silently overwrite it with a mislabeled policy). Since the resume budget is budget-steps_res, the resumed counter can never exceed the stale max when the interruption was past halfway — a second interruption then re-selects the stale checkpoint and discards all resume compute. Even on success, eval history (evaluations.npz), best_model selection (fresh -inf EvalCallback), and stage_config/stage_results provenance are corrupted. Minor refinement to the finder's scenario: checkpoints at colliding step numbers are overwritten (mislabeled provenance) rather than pruned; pruning hits only the strictly-lower ones — the practical outcome is the same.

*Bites when:* Run 20260821_142144 scenario: locomotion interrupted at 5.49M of 8M; models/ retains stage2_{3M..5M}_steps.zip (max_checkpoints=5). Resume loads stage2_5000000_steps.zip and trains remaining 3M with num_timesteps reset to 0. Each new checkpoint (stage2_500000_steps.zip, ...) has a LOWER step number than the five retained stale ones, so retention prunes it on the same step it lands. If Colab reclaims the runtime again at any point before the resume completes, the glob-max in cell 29 again picks the stale stage2_5000000_steps.zip — the entire resume session's compute is silently discarded and the loop can repeat forever. Side effects even on success: stage2_3000000_steps.zip is overwritten by the true-8M-step policy (mislabeled provenance), PublishEvalArtifactsCallback replaces the stage's evaluations.npz with a fresh file whose timesteps restart near 0 (first-session eval history and best_eval_timestep lost), stage_config.json is rewritten with timesteps=remaining, stage_results records actual_timesteps=3M for an 8M stage, and the fresh EvalCallback (best=-inf) unconditionally overwrites best_model.zip on its first eval even if worse than the first session's best. The CLI --load path in train_base.train() shares the same learn() default.

*Suggested fix:* In the resume path pass reset_num_timesteps=False to model.learn (and total_timesteps=budget_res so SB3 trains the remainder with a continuous counter), or offset CheckpointCallback naming; at minimum have the resume cell warn that retention will prune resumed checkpoints and disable retention (max_checkpoints=0) for the resumed stage.

**NB3. `notebooks/ray_tune_sweep.ipynb:625` — misleading-artifact [high / fix small]**

In notebooks/ray_tune_sweep.ipynb cell 23 (line 625), the sweep-level training_summary.txt is written from _stage_results, which after the per-trial eval loop holds the rank-TOP_K (worst of the top-K analyzed) trial's metrics and model/vecnorm paths, not rank 1 as the "last iteration = rank 1" comment claims; the correctly-captured _best and _best_trial_dir variables are dead stores. Whenever TOP_K > 1 (default 5), the sweep's canonical summary artifact — the one consumed by google_drive_summary.ipynb, the website results table, and anyone choosing a LOAD_PATH — silently reports the worst analyzed trial as the sweep's best result. Per-trial artifacts inside each trial_dir and the interactively printed ranking table remain correct, which is why the wrong summary looks plausible and goes unnoticed.

*Bites when:* TOP_K=5 analysis of a trex stage-1 sweep: rank 1 trial scores mean_reward 3400, rank 5 scores 2100. training_summary.txt written to SWEEP_DIR — explicitly produced so "downstream tools (google_drive_summary.ipynb, website results table) work seamlessly" — reports the rank-5 metrics (reward 2100, its model_path/vecnorm_path, its gate fields) as the sweep's summary. Anyone comparing sweeps by these summaries, or picking LOAD_PATH from them, is steered to a policy ~40% worse than the sweep actually found.

*Suggested fix:* Capture the rank-1 stage_results inside the loop (e.g. `if rank == 1: _best_stage_results = _stage_results`) or store per-trial results in a list and index [0].

**NB4. `notebooks/jax_training.ipynb:501` — duplicate of JX4.** Independently re-discovered by this finder: the JAX notebook auto-resume probing the retired stage{N} directory layout. See JX4; the double detection is itself weak evidence of how visible the defect is from multiple directions.

**NB2. `notebooks/google_drive_summary.ipynb:522` — silent-failure [medium / fix small]**

Confirmed as stated. One nuance vs the finder's "high": sweep data is intact on Drive and the sweep notebook's own analysis/plots still work, so nothing is corrupted or invalidated — the defect is that the Drive summary notebook silently shows zero sweep rows (and its layout docs describe a directory format the sweep notebook no longer produces), which misleads the user into thinking sweep results were lost/unsynced and plausibly triggers a redundant expensive sweep re-run. Medium severity (misleading artifact, significant debugging cost, possible wasted sweep budget); fix is a small regex/fallback extension plus a markdown doc update.

*Bites when:* User runs a trex PPO sweep today; it lands at logs/trex/sweeps/ppo_20260828_143000/ with collected_results.csv. In the summary notebook, _is_sweep_dir("ppo_20260828_143000") is False, the sweeps branch returns without recursing, scan_sweep_dir never runs, and the sweep appears nowhere — no row, no warning, 'No sweep results found.' The summary's own cell-6 markdown still documents the old sweeps/stage<N>_<algorithm>_<timestamp> layout, so the user concludes their sweep results were lost or not synced.

*Suggested fix:* Extend parse_sweep_dir_name with a second pattern ^(.+?)_(\d{8}_\d{6})$ for children of a sweeps/ directory (stage read from collected_results.csv rows or search_space json, which scan_sweep_dir already falls back to via csv_row.get("stage", stage)), and update the cell-6 markdown layout docs.

**NB5. `notebooks/sb3_training.ipynb:1488` — resume-safety [medium / fix small]**

The documented resume finish flow breaks partway: in a fresh runtime the stage-2 artifact cell (cell 24) NameErrors on undefined results_1 at the write_training_summary line — after gate verdict/videos/graphs are generated but before the summary refresh, bundle regeneration, and the gate-enforcement/disconnect run — so a resumed run cannot regenerate a complete validated bundle via any documented path. The claimed bundle corruption from a [results_2]-only workaround does NOT happen: save_result_bundle's contiguous-prefix preflight raises ResultBundleError before writing anything; only training_summary.txt could be overwritten stage-2-only via a hand-edited write_training_summary call.

*Bites when:* Colab reclaims the runtime mid-locomotion; user follows 7b exactly: fresh runtime, sections 1-4 with RUN_ID set, resume cell completes, then sets dir_2, results_2 = dir_res, results_res and runs the stage-2 artifact cell. It crashes with NameError: name 'results_1' is not defined (stage 1 was never re-trained in this runtime and the notebook header says cross-session reconstruction is unsupported). If the user edits the call to pass [results_2] alone, save_result_bundle idempotently rewrites collected_results.csv and the summary without the stance (and recovery-pilot) rows, corrupting the run's bundle. Either way the resumed run cannot regenerate a complete gated bundle via the documented flow.

*Suggested fix:* Have the resume path rebuild prior-stage results from disk (build_stage_results_from_eval_data over existing stage dirs, as the drive summary does) before bundle writes, or document that after a resume only generate_stage_artifacts may be run and bundle regeneration requires reconstructing all completed stages.

**NB6. `notebooks/sb3_training.ipynb:1046` — budget-waste [medium / fix small]**

There is no PPO.load preflight anywhere in the notebook: the first checkpoint load of a 'Run all' happens only AFTER stage-1 training completes, so on the user's current Colab L4 image (where PPO.load is known to segfault) the full 11M-step stance budget is spent before the crash.

*Bites when:* User starts a full trex run on the current L4 image. Stage 1 trains for its 11M-step budget (many GPU-hours), _select_handoff_checkpoint picks robust_best_model, and AlgoClass.load segfaults the kernel — losing all in-memory state (model_1, results_1, path threading) so stages 2-3 cannot run; auto-disconnect never fires and the runtime idles. A 5-second preflight (save a freshly constructed tiny model to scratch and PPO.load it back) at setup would have failed the session before any budget was spent. The segfault itself is tracked as environmental; the missing fail-fast guard is the actionable gap.

*Suggested fix:* Add to the setup or infrastructure cell: construct a minimal PPO on a dummy env, model.save to scratch, PPO.load it, and raise with a clear 'runtime image cannot load SB3 checkpoints' message if it crashes or errors.

**NB7. `notebooks/sb3_training.ipynb:55` — reproducibility [medium / fix small]**

Confirmed with one correction: the unquoted gymnasium>=0.29.0 is real and empirically discards the floor on that pip line (bare gymnasium installed, pip stdout redirected into a junk file named "=0.29.0" in cwd), but the floor is re-imposed indirectly by the subsequent editable install (pyproject.toml pins gymnasium>=0.29.0) and by SB3's own dependency constraints — so the effective defects are (a) fully floating versions for SB3/torch/jax/flax/optax across all three training notebooks making Colab runs irreproducible, (b) the quoting bug hiding pip's install output and creating a junk file, and (c) no legacy-gym/shimmy guard, though that last part is hypothetical on current Colab images.

*Bites when:* Two Colab runs weeks apart resolve different stable-baselines3/gymnasium/torch (and jax/flax/optax) versions; a behavior change between them (e.g. an SB3 minor release changing VecNormalize or load behavior — the current PPO.load segfault is exactly this class) makes runs incomparable, and the gymnasium constraint that was intended is never even passed to pip. provenance.json records versions after the fact but cannot re-create them. If a runtime image ships the legacy 'gym' package, SB3's shimmy pathway can import the broken package with no guard.

*Suggested fix:* Quote every requirement spec, pin exact versions (== for gymnasium/SB3/mediapy/jax/flax/optax) matching a tested set, and add 'pip uninstall -y gym gym-notices shimmy' (or an import-check assert) before the first stable_baselines3 import.

**NB8. `notebooks/jax_training.ipynb:53` — silent-failure [low / fix small]**

In notebooks/jax_training.ipynb setup cell (cell 2), GPU_AVAILABLE = subprocess.run("nvidia-smi").returncode == 0 raises FileNotFoundError on any runtime without the NVIDIA driver (CPU Colab, local machines), so the advertised CPU smoke-test branch and the osmesa MUJOCO_GL fallback in that same cell never execute; the failure is a loud traceback at setup, not a silent one, and cell 3's diagnostics call has the same unguarded pattern.

*Bites when:* User opens the notebook on a CPU Colab runtime (or locally without NVIDIA drivers) for the documented CPU smoke test: the setup cell dies with FileNotFoundError: [Errno 2] No such file or directory: 'nvidia-smi' before GPU_AVAILABLE is ever assigned; MUJOCO_GL is also never set for the CPU path, and every later cell referencing GPU_AVAILABLE fails.

*Suggested fix:* Wrap in try/except FileNotFoundError (or use shutil.which("nvidia-smi")) and set GPU_AVAILABLE=False on absence.

**NB9. `notebooks/sb3_training.ipynb:138` — reproducibility [low / fix small]**

Confirmed as stated: CLI SAC runs silently default to 8 parallel envs (environments/shared/cli.py:227,247-249, and again at 287 for the curriculum command) while the Colab notebook trains SAC with its unconditional N_ENVS = 4 (cell 6, used for env creation in cell 14), so default SAC baselines from the two entry points differ 2x in env count and collect-to-update ratio; the value is recorded in provenance/extra_meta but the default divergence is never surfaced to the user. Low severity: SAC is the secondary algorithm, PPO defaults are in parity, and the difference is discoverable in artifacts.

*Bites when:* A SAC baseline trained via `train_sb3.py train --algorithm sac` runs with 8 envs; the 'same' run in Colab with ALGORITHM="sac" runs with 4. Off-policy data collection rate vs gradient-update ratio differs 2x, results diverge, and neither artifact flags the difference (n_envs is recorded but nobody is told the defaults differ). PPO defaults are otherwise in parity (n_envs 4, seed 42, eval seed +1000, eval_freq 50k, save_freq 500k, EvalCallback episodes via _eval_episodes_for_stage).

*Suggested fix:* Mirror the CLI: `if ALGORITHM.lower() == "sac" and N_ENVS == 4: N_ENVS = 8` in the configuration cell, or print an explicit warning that CLI SAC defaults differ.

**NB10. `notebooks/ray_tune_sweep.ipynb:701` — misleading-artifact [low / fix small]**

Confirmed as stated, with one path correction: the sweep-side special handling lives at environments/shared/scripts/sweep/ray_tune.py:523 (not scripts/sweep/ray_tune.py). Cell 28 of notebooks/ray_tune_sweep.ipynb emits --override ppo.policy_kwargs.net_arch=<preset> with a comment claiming the training script handles it specially; cli._apply_overrides has no such handling and stores the literal key 'policy_kwargs.net_arch' in ppo_kwargs, which survives _prepare_alg_kwargs (train_base.py:270) and the policy_kwargs pop (train_base.py:412) and makes PPO(**alg_kwargs) raise TypeError. Currently latent — no shipped sweep config sweeps net_arch — and it fails loudly, not silently.

*Bites when:* Someone adds a net_arch preset dimension to configs/<species>/sweep_ppo.json (NET_ARCH_PRESETS exists for exactly this), runs a sweep, and copies the cell-28 command to retrain the winner via the CLI. _prepare_alg_kwargs copies ppo_kwargs including 'policy_kwargs.net_arch', _create_or_load_model pops only 'policy_kwargs', and PPO(**alg_kwargs) raises TypeError: unexpected keyword argument 'policy_kwargs.net_arch' — and even if the key parsed, the preset label string would not be a valid net_arch.

*Suggested fix:* Either resolve the preset in the cell (emit nothing and print a note that net_arch must be set in the TOML/policy_kwargs), or teach cli._apply_overrides a nested policy_kwargs.net_arch path with NET_ARCH_PRESETS resolution, then keep the comment truthful.

### 3.3 Environment & physics

*Scope: base_env.py, reward/observation functions, perturbation, trex envs.*

**EP1. `environments/shared/base_env.py:1101` — silent-failure [medium / fix small]**

Confirmed as stated, with two refinements: (1) the probe's warning counter is mjWARN_BADQVEL (index 4) for the qvel injection, though badqacc follows the same auto-reset path; (2) "no trace" is almost exact — MuJoCo prints one stderr warning line on the first divergence per MjData instance, then all subsequent divergences in that env are fully silent; nothing programmatic (termination, info dict, metrics) ever records it.

*Bites when:* Verified by probe: with the stance config, injecting divergent state (qvel=1e12) then stepping prints MuJoCo's 'simulation is unstable' warning, mj_step resets to qpos0 (pelvis z 0.977, healthy), and the env returns terminated=False, reward=+1.10, all-finite obs; the next step also continues normally. In a real run (165.5 N pushes, kp=1500 position servos, early-training contact spikes) a badqacc divergence mid-episode silently teleports the T-Rex back to the un-noised default pose at the spawn origin -- which also erases accumulated drift penalty and resets velocity to zero -- while the push-schedule clock, VecNormalize statistics, episode-length/reward gate metrics, and any recovery scoring ingest the physically impossible trajectory with no trace. The failure is invisible by construction, so past runs cannot confirm it never happened.

*Suggested fix:* Snapshot data.warning.number before the frame-skip loop and after it; if any counter incremented, terminate the episode with a distinct termination_reason (e.g. 'physics_divergence') and log it, or raise in debug contexts. Mirror with a divergence flag on the MJX side.

**EP2. `environments/shared/perturbation.py:113` — budget-waste [medium / fix medium]**

Requested refinement of the known magnitude-randomization gap: the exact insertion point is a third hash lane in push_schedule plus a per-push magnitude argument to external_push_force, and the critical hazard is that the existing hash-lane numbering (k*2, k*2+1) must not be renumbered or every existing schedule silently changes.

*Bites when:* Confirmed scalar-only: derive_push_parameters computes one force (165.501 N verified for multiple=1.5) and external_push_force applies it uniformly to every push. Insertion point: (a) push_schedule (lines 95-121) gains a magnitude lane -- it MUST use an index set disjoint from {k*2, k*2+1} (e.g. hash_uniform01(schedule_seed, k + 0x80000000) or a salted seed); naively renumbering lanes to k*3/k*3+1/k*3+2 changes every existing schedule for the same seed, silently re-anchoring the null-controller pairing baselines and cross-run comparability. (b) external_push_force (123-142) changes signature to take magnitudes[max_pushes] and compute sum(active[:,None]*directions*magnitudes[:,None]) -- keep static shape for JAX scan. (c) validate_push_config (145-182) gains fail-closed range validation. (d) derive_push_parameters (184-259) should return force-per-unit-multiple so per-push multiples scale it. Blast radius: base_env.py __init__ 243-273 (new kwarg, stored constants), reset 1447-1456 (derive per-episode magnitudes from the same schedule_seed so policy/null-controller pairing holds automatically), step 1063-1071 (call site), perturbation_manifest 1013-1027 (persist range); mjx_env.py config fields 256-260, EnvState carry 286-287 plus its pytree data_fields registration, _push_constants 826-855, step kernel 967-979, reset kernel 1438-1456 including the (0,)-shaped off branch, manifest 1514+; task_fingerprint.py SCHEDULE_IMPLEMENTATION = "push_schedule/v1+lowbias32+capture-point" (line 63) must be bumped because the transition kernel's meaning changes; configs/trex/recovery.toml and recovery_evaluation.py/harnesses are out of scope (in-flight) but consume the same schedule functions -- pairing survives iff magnitudes are a pure function of schedule_seed.

**EP3. `environments/shared/mjx_env.py:838` — reproducibility [low / fix small]**

MJX derive_push_parameters call (mjx_env.py:838) omits keyframe_id, defaulting to keyframe 0, while the SB3 path (base_env.py:270) passes the species' _reset_keyframe_id and MJX's own reset resolves 'home' by name — so any future asset whose home keyframe is not index 0 would get push forces calibrated from the wrong pose on MJX (diverging both from SB3 and from MJX's own reset pose). Latent today: all four species assets have 'home' as their sole keyframe (id 0).

*Bites when:* A future species (or a T-Rex asset edit that adds a second keyframe before 'home') registers a reset keyframe with id != 0. derive_push_parameters then measures CoM height and support-polygon radius from keyframe 0's pose on MJX but from the reset keyframe's pose on SB3, producing different force_n for the same capture_velocity_multiple. Recovery training on one backend and gate evaluation on the other (the exact split the MJX parity plan aims for) would silently apply different push difficulty; the divergence appears only as mismatched force_n values in the two perturbation manifests, which nothing currently cross-asserts.

*Suggested fix:* Pass the registered reset keyframe id in the MJX call (mjx_config already knows the home keyframe), and/or have the plant/task contract assert both backends' perturbation manifests agree.

**EP4. `environments/shared/base_env.py:1348` — eval-validity [low / fix small]**

Confirmed core: after a noisy reset, _settle_root_on_ground (base_env.py:1348) aligns only the lowest geom to home clearance, so one foot spawns airborne and reads 0 N for the first ~3-5 control steps of essentially every stance episode (reproduced 4/4 seeds), zeroing bilateral_support_quality, halving the alive gate, and setting imbalance=1 — reward 1.86-1.99 vs 3.52 for the settled statue (~0.3% of episode return, policy-independent). The stance-duty gate is unaffected (settle_steps=200 excludes the transient). NOT reproduced: the claimed ~1-per-600 mid-episode substep-MIN zero artifact and its ~0.17% duty contribution — 0 occurrences in 6779 statue steps across 12 seeds; substep-MIN/boundary divergence was observed only within the reset transient.

*Bites when:* Measured under the home controller with the stance config (reset_noise_scale=0.05), seeds 0-3: 3-5 steps per 600 have an aggregated per-foot MIN force of exactly 0.0 N, concentrated at t=0..3 where even the boundary sample reads 0 N (one foot spawns millimetres above the floor), plus ~1 mid-episode step per 600 where the boundary sample reads ~394 N but a substep dipped to 0 (the substep-MIN artifact). On those steps bilateral_support_quality=0, alive_gate drops to 0.5, and foot_load_imbalance=1, cutting reward from 3.53 to 1.86 for a perfect statue. Training impact is small (~0.1% of episode return) but the same r_foot_contact/l_foot_contact info keys feed the stance-duty gate (max_unsupported_duty=0.02): the settle window excludes the reset transient, but the mid-episode artifact contributes ~0.17% duty of the 2% budget as pure sensor noise, and any recovery-gate window opened right after a push (out-of-scope work in flight) will see the same transient-0 readings from the env side.

*Suggested fix:* Either settle each foot's support chain (or verify both feet within a small clearance) at reset, or exclude the first few post-reset steps from support-conditioned reward terms; document the transient-0 behaviour for gate authors.

### 3.4 Evaluation & reporting

*Scope: evaluation.py, diagnostics, result_bundle, reporting.*

**ER1. `environments/shared/evaluation.py:429` — eval-validity [high / fix small]**

CLI evaluate() (evaluation.py:429-431) probes only the curated <base>_vecnorm.pkl sidecar name, so for SB3 periodic checkpoints (<stage_label>_<steps>_steps.zip with <stage_label>_vecnormalize_<steps>_steps.pkl sidecar — the naming this project's CheckpointCallback with save_vecnormalize=True produces) the sidecar is never found and evaluation silently proceeds on unnormalized observations after a single warning (line 466). Curated final/best checkpoints are unaffected. train_base._resolve_vecnorm_sidecar already handles both schemes; evaluate() just needs to call it.

*Bites when:* User runs `train_sb3.py eval --model-path .../01_stance/models/stage1_5000000_steps.zip` (the checkpoint naming this project uses: <stage_label>_<steps>_steps.zip with sidecar <stage_label>_vecnormalize_<steps>_steps.pkl). evaluate() probes stage1_5000000_steps_vecnorm.pkl, which never exists (verified: SB3 2.9.0 CheckpointCallback writes stage1_vecnormalize_5000000_steps.pkl), logs one warning, and evaluates the policy on raw observations while it was trained on VecNormalize-standardized ones — the reported reward/velocity/duty numbers describe a different policy. train_base._resolve_vecnorm_sidecar was added for exactly this mismatch (review F3) and handles both naming schemes, but evaluate() does not call it; test_evaluation.py only exercises curated names. With PPO.load segfaulting on the current Colab image (known), CLI eval is precisely the fallback path users hit — mid-training checkpoints will all look catastrophically broken.

*Suggested fix:* In evaluate(), resolve the sidecar via train_base._resolve_vecnorm_sidecar(model_path) instead of the inline .replace(); consider making a missing sidecar an error (or requiring --allow-unnormalized) rather than a warning.

**ER3. `environments/shared/evaluation.py:492` — reproducibility [medium / fix small]**

evaluate() in environments/shared/evaluation.py (lines 402-492) never seeds its evaluation environment — no seed parameter exists, _make_eval_env constructs the env unseeded, the per-episode vec_env.reset() passes no seed, and cli.py's eval subcommand exposes no --seed — so with deterministic=True policy actions the only randomness is OS-entropy reset noise (reset_noise_scale=0.05), making CLI evaluation numbers irreproducible across invocations and checkpoint comparisons unpaired; the module's own replay path (replay_seed, line 297) seeds properly, so evaluate() is the sole unseeded outlier. Note the default is only 10 episodes (CLI --episodes default), not 30, amplifying run-to-run variance.

*Bites when:* Two invocations of `train_sb3.py eval` on the same checkpoint draw reset randomization from OS entropy (Gymnasium lazily seeds np_random on first unseeded reset) and produce different episode panels, so evaluations of two checkpoints are not paired and a re-run cannot reproduce a reported number. This contradicts the provenance discipline everywhere else in the repo (seed_roles, fixed evaluation_seeds, deterministic=True protocols): the training notebook seeds its publication envs with EVALUATION_SEED, but the CLI diagnostic path — the one used to compare checkpoints after a crashed run — has no seed at all. Note eval seeds elsewhere ARE distinct from training (SEED vs SEED+1000 selection vs SEED+3000 publication) and applied once at env construction with episodes drawn deterministically from that stream, which is sound; evaluate() is the outlier.

*Suggested fix:* Add a seed parameter to evaluate() (default e.g. training-seed-independent constant), seed the env at construction (env.reset(seed=...) or vec_env.seed), and thread --seed through cli.py's eval command.

**ER4. `environments/shared/reporting/stage_artifacts.py:143` — misleading-artifact [medium / fix small]**

build_stage_results_from_eval_data (environments/shared/reporting/stage_artifacts.py:143-145) hardcodes mean_forward_vel/std_forward_vel/mean_success_rate to 0.0 even though it already opens the metrics.json that holds the real 30-episode values; both sweep workers (environments/shared/scripts/sweep/trial.py:223 and ray_tune.py:1001 — note the correct path, not scripts/sweep/trial.py) call generate_stage_artifacts without stage_results, so _apply_stage_gate evaluates velocity/success criteria against fabricated 0.0 and every stage-2/3 sweep trial records gate_passed=False with 'best model forward vel 0.00 m/s' regardless of actual performance. Fail-closed (no false pass), but gate columns and stage summaries systematically contradict metrics.json beside them.

*Bites when:* A stage-2 locomotion sweep trial finishes with a genuinely good policy (metrics.json beside it holds a real mean_forward_vel from _report_hpt_metrics' 30-episode eval). generate_stage_artifacts builds stage_results from evaluations.npz, leaving mean_forward_vel=0.0; reporting/gates.py's _gate_metric falls through best_model_fwd_vel (absent) to mean_forward_vel=0.0, compares 0.0 < min_avg_forward_vel=1.0, and records gate_passed/publication_gate_passed=False plus a stage_summary.txt showing 'Avg fwd vel: 0.00 m/s' and a FAIL verdict citing velocity — for every trial, regardless of actual performance. Direction is fail-closed (no pass is laundered), but the gate columns and summaries for stages 2/3 sweeps are systematically wrong and contradict metrics.json in the same directory, poisoning trial triage. (Interacts with reporting/gates.py, which is out of scope; reported from the stage_artifacts/builder side: the defect is fabricating a plausible measured value instead of omitting the key — the dict's own docstring says these 'can be updated by the caller after running eval_policy', and the sweep caller never does.)

*Suggested fix:* Have build_stage_results_from_eval_data read mean_forward_vel/mean_distance_traveled/mean_success_rate from metrics.json when present, and otherwise omit the keys (or use None) so _gate_metric treats them as unmeasured instead of 0.0.

**ER5. `environments/shared/reporting/stage_artifacts.py:1363` — resume-safety [medium / fix small]**

The JAX artifact path (save_jax_stage_artifacts) writes diagnostics.npz via bare np.savez and stage_result.json via bare write_text — non-atomic on Drive/GCS FUSE despite file_io.py existing for exactly this and all SB3-side writers using it. A truncated stage_result.json then wedges every subsequent save_jax_stage_artifacts call for the run (next stage, retries, and even re-saving the same stage), raising ResultBundleError before any artifact — checkpoints included — is written, and also crashes the notebook's stage-2/3 auto-resume cell, until the file is hand-repaired. Minor overstatement in the original: trained params are usually still recoverable from jax_trainer's params.pkl and the live notebook session, so the cost is the canonical stage record/bundle and curriculum progression plus manual repair, not necessarily the weights; a truncated diagnostics.npz corrupts only that artifact and does not wedge.

*Bites when:* file_io.py exists specifically because 'np.savez in particular rewrites the whole zip on every call' on Drive/GCS FUSE and can be 'permanently left truncated', and the SB3 DiagnosticsCallback uses atomic_savez for exactly this file — but the JAX stage-artifact path calls bare np.savez and bare write_text. If the Colab runtime dies mid-flush of stage_result.json (written after each multi-hour stage), the next session's save_jax_stage_artifacts hits json.JSONDecodeError on the prior stage record and raises ResultBundleError before saving anything — including the just-trained stage's best_model.pkl/final checkpoint (saved at step 5, after the scan) — so an environment hiccup during a cheap JSON write costs the following stage its entire artifact set until someone hand-repairs the file. The cross-session curriculum flow this file exists for ('one idempotent stage record for cross-session curricula') is the exact flow that gets wedged.

*Suggested fix:* Use file_io.atomic_savez for diagnostics.npz and result_bundle.hashing._write_json (or atomic_copy) for stage_result.json; same for summaries.save_results_json's write_text.

**ER2. `environments/shared/eval_diagnostics.py:923` — silent-failure [low / fix small]**

StageGatePlateauCallback formats self.stage with %d in both the plateau warning (eval_diagnostics.py:922-936) and plateau-cleared (631-636) messages, and semantic string stages like "recovery" reach it unchanged, causing a runtime-verified logging TypeError that replaces the message with a '--- Logging error ---' traceback. But the finder's failure scenario is overstated: with the current (and committed) recovery config, the stage declares only the recovery_quality/v1 LCB gate and none of the scalar thresholds this callback tracks, so the plateau branch is unreachable on stage 1b and no diagnostic is currently lost. It is a latent bug that fires only if a semantic-id stage ever gains a scalar gate threshold (e.g. min_avg_reward added to recovery.toml).

*Bites when:* The notebook trains the current focus stage with train_stage(stage="recovery") (sb3_training.ipynb cell 21), which reaches StageGatePlateauCallback(stage="recovery") — the class signature is stage: "int | str". When the blocking-gate plateau fires, logging's getMessage() raises TypeError('%d format: a number is required'); verified in the venv: the handler prints a '--- Logging error ---' traceback and the actual message (which metric plateaued, latest vs required value, the guidance text) is lost. The one diagnostic that tells the operator which recovery-stage gate is blocked never appears, replaced by a confusing traceback; every plateau on stage 1b is invisible. test_eval_diagnostics.py only uses integer stages, so this is untested. This is a residual int-stage assumption from the semantic-stage migration (question 4).

*Suggested fix:* Change both format strings to %s for the stage field (matching evaluation.py's 'Stage %s video' style).

**ER6. `environments/shared/result_schema.py:354` — eval-validity [low / fix small]**

Accurate as stated. Minor refinements: the gap exists in two places (result_schema.validate_provenance canonical branch AND result_bundle/provenance.initialize_result_bundle lines 181-186, which duplicates the same set-equality logic and also lacks distinctness checks), so a fix must cover both to be effective at capture time as well as publish time. Runtime-verified: seed_roles with publication_evaluation == training_seed, with a selection role sharing the publication seed, or with all roles on one seed all validate as canonical, and the evidence audit only pins CSVs to the publication seed. Today's notebooks use distinct seeds, so severity low (missing guardrail, requires a future caller error to bite — but nothing would flag it when it does).

*Bites when:* A caller (or a future notebook edit) sets CHECKPOINT_SELECTION_SEED == EVALUATION_SEED, or omits the selection role entirely: seed_roles={'training': 42, 'publication_evaluation': 1042} with evaluation_seeds=[1042] validates as canonical, the evidence CSVs pass audit (they only must carry the publication seed), and the published selected_model_* numbers are then computed on the same episode panel that chose the checkpoint — classic winner's-curse inflation the seed_roles design exists to prevent, with nothing in schema, audit, or bundle save flagging it. Similarly seed_roles={'training': 42, 'publication_evaluation': 42} validates, so publication episodes can share the training seed. Today the SB3 notebook hard-codes distinct SEED/+1000/+3000 and the JAX notebook uses 0 vs 42, so this is a missing guardrail rather than an active defect — but it is the only thing standing between the provenance contract and silent selection bias (question 2).

*Suggested fix:* In canonical validation, require that seed_roles.publication_evaluation differs from training_seed and from any role matching *selection*; optionally require a recorded selection-evaluation role for stable-baselines3 bundles.

**ER7. `environments/shared/result_bundle/manifest.py:38` — audit-gap [low / fix small]**

Dot-prefixed files are excluded from both manifest hashing and the reject_unlisted scan, so files can be added to or left inside a 'complete' immutable bundle without failing audit.

*Bites when:* A crashed atomic write (hashing._write_json's temporary is named '.summary.json.tmp' in the same directory) or any tool dropping a dot-named file leaves content inside a canonical-valid, 'immutable' bundle that no hash covers and reject_unlisted never reports — e.g. a stale .summary.json.tmp containing different numbers than the audited summary.json survives inside a verified bundle, and audit_result_bundle still reports canonical-valid. More broadly for question 3, the audit's trust root is the manifest itself: a coordinated edit of an artifact plus its manifest entry (plus provenance/summary hashes for the covered files) is undetectable because nothing outside the run directory anchors artifact_manifest.json — inherent to a self-contained bundle, but worth stating in the audit's docs. What the audit does catch is solid: any post-manifest mutation of a declared file (size+sha256), stance verdicts re-derived from per-episode panels rather than trusted, and evidence aggregates recomputed against summary numbers.

*Suggested fix:* Either include dot-files in the reject_unlisted scan (while still skipping them at write time, so leftovers fail closed and get cleaned), or have write_artifact_manifest delete stray '.*.tmp' files before hashing.

### 3.5 JAX / MJX stack

*Scope: jax_* modules, mjx_env, trex mjx_config.*

**JX1. `environments/shared/jax_reward_termination.py:297` — eval-validity [high / fix small]**

The JAX CPU-eval reward composer (compute_total_reward and compute_reward_components in environments/shared/jax_reward_termination.py) omits the action-jerk term that the MJX training step pays (mjx_env.py:1149-1155) with stance weight 3.0 from configs/trex/stance.toml, so run_stage_evaluation's episode rewards, the min_avg_reward=2100 rail, and the reward component panel systematically overscore high-frequency-chatter policies relative to training and to the SB3 backend (which does pay jerk). One nuance: stance.toml documents min_avg_reward as a collapse rail, not the promotion gate itself, so "gate certifies against" is slightly overstated — but the rail comparison and all reported eval rewards are computed jerk-free, so the eval-validity and cross-backend-comparability defect stands.

*Bites when:* jax_setup.run_stage_evaluation scores episodes with compute_total_reward and feeds them to the stance gate's min_avg_reward=2100 rail and reward panel. The jerk term exists precisely to price the high-frequency chatter the stance gate is meant to reject; a buzzy policy therefore scores strictly higher in CPU eval than in training, and higher than the identically-configured SB3 eval (which runs the real env, jerk included). Best-model selection uses training returns (jerk included) while gate certification uses eval rewards (jerk excluded) — the two rank policies differently, and the same TOML min_avg_reward means different things per backend.

*Suggested fix:* Add an action_jerk_weight branch (with a prev_prev_action kwarg) to compute_total_reward and compute_reward_components, thread prev_prev_action from evaluate_policy_cpu, and pin with a step-vs-composer parity test that sets action_jerk_weight > 0.

**JX2. `notebooks/jax_training.ipynb` — resume-safety [high / fix small]**

Same-stage RESUME_FROM in notebooks/jax_training.ipynb silently breaks continuation: cell 15 loads only params/update/obs_rms (never the checkpoint's saved opt_state) and applies obs_rms count decay (default 0.01, stance TOML sets no override), and cell 22 unconditionally re-inits the optimizer — so Adam moments zero, the linear LR schedule (count lives in opt_state, verified) restarts at 3e-5, and normalization stats re-fit within ~1 update. restore_train_state (jax_checkpoint.py:158) would fix this but has no non-test callers, and train_jax exposes no init_opt_state despite JaxTrainer.train supporting it. For cross-stage auto-resume the fresh optimizer and obs decay appear intentional (documented in comments/docstrings), so the defect is specifically the same-stage resume path.

*Bites when:* Resume a stance run from checkpoint_400.pkl via RESUME_FROM: Adam moments restart from zero, the linear LR schedule (3e-5 -> 1e-5, count lives inside opt_state) snaps back to 3e-5 for the whole resumed segment, and the obs_rms count is cut 100x so the normalization statistics re-fit to the current policy's distribution within ~1 update — the resumed run is not a continuation of the original, silently, while the checkpoint's opt_state that would prevent all three sits unused on disk. On stance, where seed sensitivity is already a known pain point, a post-crash resume can look like a training regression.

*Suggested fix:* Use restore_train_state in the notebook (and add --resume/init_opt_state to train_jax); gate obs_rms_decay_on_resume on cross-stage resume only (default 1.0 for same-stage).

**JX3. `environments/shared/jax_setup.py:620` — recovery-gap [medium / fix medium]**

Verified refinement of the known MJX recovery-parity gap: MJXDinoEnv fully implements recovery pushes (derives force_n=165.5 N, 20-step duration, 200/50 interval/jitter steps, max_pushes=7 from configs/trex/recovery.toml [env]), reward terms, and termination — but the stage cannot be trained or honestly evaluated: jax_training.py --stage is int-only (recovery has no legacy number), setup_species accepts stage="recovery" yet make_reward_fns then crashes with TypeError at jax_setup.py:620 ("ctx.stage >= 3" on a str), evaluate_policy_cpu in jax_eval.py contains no push application (so a future recovery gate would certify on push-free episodes), and check_stage_gate_for_config dispatches only stance_quality/v1 and the reward/length fallback. Correction to the evidence: SpeciesContext.success_sites does not crash for trex recovery — "if self.env_config and self.stage >= 3" short-circuits on the empty env_config; the crash is in make_reward_fns (and any int comparison actually reached).

*Bites when:* Precise gap inventory for planning: (a) pushes — implemented in mjx_env step/reset with SB3-shared perturbation semantics and manifest; (b) recovery reward — every recovery.toml [env] key maps to an implemented MJX term; (c) recovery termination — same generic checks incl. per-stage nosedive threshold; (d) stage addressing — MISSING: --stage cannot express "recovery" (no legacy number) and jax_setup's int comparisons crash, so no MJX recovery training run can be launched; (e) recovery evaluation — MISSING: evaluate_policy_cpu applies no pushes, so any future gate run through run_stage_evaluation would certify recovery on push-free episodes; (f) recovery gate — MISSING on the JAX side (check_stage_gate_for_config dispatches only stance_quality/v1 and reward_and_length/v1; the new recovery gate lives in the out-of-scope in-flight curriculum files with no jax_eval/jax_setup integration). This refines the tracked 'MJX recovery-stage parity gap' item; reported from the in-scope side.

*Suggested fix:* Accept semantic stage ids through train_jax/jax_setup (replace stage>=3 with a manifest lookup of the behavior stage), add push application to evaluate_policy_cpu behind the env's perturbation config, and wire the recovery gate kind into check_stage_gate_for_config once the gate lands.

**JX4. `notebooks/jax_training.ipynb` — resume-safety [medium / fix small]**

Cell 15's stage-2/3 auto-resume resolves the previous stage as RUN_DIR/f"stage{N}", but cell 6's setup_output_dirs writes NN_id directories (01_stance) and stage_artifacts writes stage_result.json only there, so auto-resume raises FileNotFoundError for every run created under the current layout — in any session, not just fresh ones — and the manual RESUME_FROM workaround silently skips the publication-gate check.

*Bites when:* Train stance in one Colab session (artifacts land in <run>/01_stance/), start a new session with CURRENT_STAGE=2 and the same RUN_ID: cell 15 raises FileNotFoundError ('missing .../stage1/stage_result.json') even though the gated stage-1 result exists, blocking the documented multi-session JAX curriculum flow; the workaround (manual RESUME_FROM) then skips the publication-gate check that auto-resume enforces.

*Suggested fix:* Resolve the previous stage dir via stage_manifest.stage_dir_candidates (and the manifest's stage ordering rather than CURRENT_STAGE - 1, which will also be wrong once recovery is addressable).

**JX5. `environments/shared/jax_trainer.py:383` — correctness [medium / fix small]**

In both JAX trainers (functional collect_rollout, jax_trainer.py:382-383, and JaxPPOTrainer, 1046-1047), the GAE done mask is termination-only while per-step final_obs is used solely for the rollout's last step (641/1319); any time-limit truncation at a non-final rollout position therefore bootstraps gamma*V(post-auto-reset obs) instead of gamma*V(final_obs) and lets the lambda carry leak the next episode's advantages into the ended one. With synchronized resets (step_count=0, no phase randomization), max_episode_steps=1000 and rollout_len=64 (1000 mod 64 = 40), all surviving envs truncate mid-rollout simultaneously once the policy reaches the horizon — the regime the stance gate selects for. Numerically reproduced with the repo's compute_gae. Truncation at the last rollout step IS handled correctly; only mid-rollout truncations are biased.

*Bites when:* All 2048 envs reset together and stay step-synchronized while surviving, so once a stance/locomotion policy reaches the 1000-step horizon the whole fleet truncates simultaneously at rollout position 1000 mod 64 — on that update ~2048 advantage tails bootstrap V(reset-pose) instead of V(s_1000) and mix the next episode's lambda-weighted advantages into the ended episode. The docstrings claim the opposite ("final_obs ... needed to bootstrap value correctly at truncation boundaries"), and SB3's handling (terminal_observation bootstrap + done-masked GAE) does not have this bias, adding another quiet backend divergence exactly at the full-horizon regime the stance gate targets.

*Suggested fix:* Bootstrap per-step: where truncated & ~terminated, add gamma * V(final_obs_t) to the step reward (SB3-style) or use full_done as the GAE carry mask while keeping the termination-only mask for the bootstrap term.

**JX6. `environments/shared/mjx_env.py:1076` — eval-validity [medium / fix small]**

In stages 2/3 (locomotion.toml and behavior.toml leave support_conditioned_alive_fraction at 0.0), the MJX alive bonus is height-gated to ~0.27x (mjx_env.py:1069-1081) while SB3 pays it in full (envs/trex_env.py:636), and [jax] fall_penalty=-10 overrides [env] -150 (applied via jax_curriculum.py:440-441), yet both backends' curriculum gates compare raw episode returns against the same [curriculum] min_avg_reward=100 — so the shared gate encodes a materially different bar per backend (~370 points of alive contribution over a 1000-step episode) and cross-backend 'passed the same gate' comparisons are invalid. Correction: the VecNormalize prong applies to training dynamics only, not gate calibration — train_base.py:379/382 force eval_env.norm_reward=False, so SB3 gate evaluation already uses raw rewards; the jax_normalization.py 'Equivalent to SB3's VecNormalize' docstring being obs-only remains a minor documentation inaccuracy.

*Bites when:* Trex locomotion: alive contribution alone is 500/episode on SB3 vs ~133 on MJX at standing height (alive_bonus 0.5 x height_frac 0.266 x 1000 steps), before the forward-ref divergence and the -150 vs -10 fall penalty; the single min_avg_reward=100 in [curriculum] therefore encodes a materially different bar per backend, and any 'JAX run passed the same gate SB3 passed' comparison is spurious. The height-gate itself is documented in code as deliberate legacy behavior — the un-tracked hazard is the shared gate calibration and the jax_normalization docstring's claim of being 'Equivalent to SB3's VecNormalize' while reward normalization exists only on SB3.

*Suggested fix:* Either split min_avg_reward per backend in the TOML ([curriculum.jax]) or align the stage-2/3 alive gate and fall penalties, and document the norm_reward asymmetry in jax_normalization.py.

**JX7. `environments/shared/jax_viz.py:658` — misleading-artifact [medium / fix small]**

record_training_video (jax_viz.py:521-660) rolls out the raw policy mean with no command low-pass, while the MJX training kernel (mjx_env.py:907-951), CPU gate eval (jax_eval.py:308-360), and the SB3 plant all apply the trex's mandatory plant-level 10 Hz filter (mjx_config.py:43; plant_versions.toml item 10 — note current trex policy_interface_revision is 12, not 11). The notebook's evaluation.mp4 and its printed episode_reward (which also feeds the unfiltered action to reward_fn) therefore come from a different plant than training and gate eval; measured trajectory divergence for identical commands reaches ~0.19 m root height within 2 s.

*Bites when:* For a policy trained under the 10 Hz filter, any command content above the cutoff — which the filter made free during training — reaches the actuators only in the video. The evaluation.mp4 and its printed episode_reward (the human-facing evidence a stage 'looks right') can show falls, chatter, or altered gait that neither training nor the gate eval exhibits, or mask chatter the gate would care about; stage review decisions get made on rollouts from the wrong plant.

*Suggested fix:* Add action_filter_cutoff_hz to record_training_video (same seed-with-first-action carry as jax_eval) and pass env.config.action_filter_cutoff_hz from the notebook.

**JX8. `environments/shared/jax_trainer.py:1411` — silent-failure [low / fix small]**

jax_trainer.py:1411 computes eval_metrics['mean_episode_return'] with jnp.mean over the last-10 episode-return window, whose entries are NaN for updates with no completed episode (line 1300); one NaN poisons the whole mean even when the other nine updates show a passing return. The gate comparison (episode_return >= min_reward in jax_curriculum.check_stage_gate) makes NaN fail CLOSED, not open — so the harm is a spurious gate failure that stops run_curriculum early on a genuinely passing policy (logged as 'episode return=nan') plus a NaN headline metric in returned eval_metrics/JSON, inconsistent with mean_episode_length two lines below, which was deliberately NaN-hardened via _finite_mean.

*Bites when:* A short CPU/CI run or an early-collapsing policy where one of the final 10 updates completes no episodes: mean_episode_return=NaN flows to train_jax's return and the curriculum gate consumer, where 'NaN < min_avg_reward' is False in a way that reads as a metric value rather than 'unmeasured' — the run's headline metric and any JSON it lands in show NaN, and downstream threshold logic behaves inconsistently with the length metric's documented fail-closed treatment.

*Suggested fix:* Use _finite_mean (or propagate None) for mean_episode_return as well.

**JX9. `environments/shared/jax_training.py:152` — reproducibility [low / fix small]**

[jax] policy_kwargs.net_arch (declared in every trex stage TOML as [jax.policy_kwargs] net_arch = [512, 256], commented "Match SB3 PPO architecture") is inert on the entire JAX path: train_jax has no parameter for it, jax_training.py:152 calls make_actor_critic(env.action_dim) with the hardcoded (512, 256) default, and jax_curriculum's _JAX_KEY_MAP never forwards it — while validate_jax_kwargs deliberately allowlists the key so it passes validation with no runtime warning. The gap is acknowledged in a committed code comment in jax_curriculum.py ("consumed by no JAX path today ... wiring it into make_actor_critic is its own change"), so this is a known-in-code deferred wiring gap rather than an unnoticed bug, but it is not on the project's tracked issue list and today's coincidence of values means a future net_arch change would silently train the old architecture on JAX while recording the new one.

*Bites when:* The values coincide today, so nothing is visibly wrong — but a stage TOML changing net_arch (e.g. a sweep testing [256,128]) would train the old architecture on the JAX path while recording the new one in run provenance, and the run would be silently non-reproducible from its recorded config; on SB3 the same key takes effect, adding another cross-backend asymmetry.

*Suggested fix:* Pass hidden_dims=tuple(jax_kwargs.get('policy_kwargs', {}).get('net_arch', (512, 256))) at every make_actor_critic call site, or reject the key loudly like ramp_attr.

### 3.6 Configs & sweep search spaces

*Scope: stage TOMLs (except recovery.toml), sweep JSONs, config loading.*

**CF1. `configs/trex/sweep_ppo.json:91` — budget-waste [high / fix medium]**

Stage-3 sweep search spaces in configs/{trex,velociraptor,brachiosaurus,dibothrosuchus}/sweep_{ppo,sac}.json sample env_* keys (prey_distance_min/max, prey_distance_range_min/max, food_distance_range_min, food_height_range_min) that no env constructor accepts; both the Vertex HPT path (trial.py -> cli._apply_overrides) and the Ray Tune path (ray_tune.apply_sampled_config) forward them verbatim into env_kwargs, so every stage-3 trial dies with TypeError at make_env before any training, yielding a zero-result sweep launch.

*Bites when:* Vertex AI HPT samples env_prey_distance_min for a stage-3 trial; trial.py _hpt_arg_to_override converts it verbatim to 'env.prey_distance_min=1.7'; cli.py _apply_overrides writes env_kwargs['prey_distance_min']; train_base.make_env line 185 does 'env = species_cfg.env_class(**env_kwargs)'; verified empirically: TRexEnv(prey_distance_min=1.5) raises TypeError: unexpected keyword argument. The env only accepts prey_distance_range (trex_env.py line 161). Every one of the 50 PPO / 25 SAC stage-3 trials fails at startup, burning the whole sweep launch with zero results. The Ray Tune path (ray_tune.py apply_sampled_config line 527-529) has the identical hole. Systemic across species: velociraptor sweeps carry the same two keys, brachiosaurus carries food_distance_range_min/max and food_height_range_min/max, dibothrosuchus carries prey_distance_range_min/max — programmatic check against each env_entrypoint signature confirmed none exist. No code anywhere in the repo translates the min/max split back into the *_range tuple (repo-wide grep: the names appear only in the sweep JSONs).

*Suggested fix:* Either add prey_distance_min/max (etc.) kwargs to the envs that forward into the range tuple, or teach the override layer (trial.py/_apply_overrides and ray_tune.apply_sampled_config) to fold <name>_min/<name>_max pairs into <name>_range tuples; add a preflight that validates every env_* sweep param against the env constructor signature before launching paid trials.

**CF2. `configs/trex/behavior.toml:96` — eval-validity [high / fix medium]**

configs/trex/behavior.toml:96 gates behavior-stage advancement on a per-episode-mean forward velocity of 2.0 m/s, but bite success terminates episodes (trex_env.py:941-947) after as little as ~0.48 m of pelvis displacement (head reach ~1.52 m, prey 2-6 m), so bite episodes structurally cannot average 2.0 m/s from a standing start; the threshold was derived from full-horizon locomotion episodes at double the forward-velocity reward weight, and the review it cites actually prescribed it as a stage-entry requirement, not an in-stage episode-average gate. A successful hunting policy will be reported as gate-failed for the full 8M budget, and sweep stage_passed will be False for all behavior trials.

*Bites when:* eval_diagnostics.py accumulates info['forward_vel'] every step and on done appends total/count (lines 234-261): the episode average includes the acceleration-from-rest prefix. behavior.toml tightens prey_distance_range = [2.0, 6.0] (line 20) and bite_bonus=1000 terminates the episode on contact. Averaging 2.0 m/s over a 2 m dash from rest requires covering it in <=1.0 s (~4 m/s peak, Froude ~2 on this plant — far beyond the walk-run boundary the config itself places at 2.0-2.5 m/s); even a 6 m spawn needs a near-instant launch to 3+ m/s. Meanwhile min_success_rate = 0.5 (line 106) rewards exactly the short episodes that depress the average, and [env] halves forward_vel_weight to 0.5 (line 6) 'to prevent forward reward from dominating'. The threshold's cited basis (run 20260821_142144's velocity curve at 7.1M+ steps) is a full-horizon locomotion measurement that does not transfer. Concrete outcome: a policy that bites 100% of the time at healthy sprint speed still averages ~1.0-1.5 m/s per episode, the gate never passes, the stage burns its full 8M budget and the curriculum reports the capability target as failed — the same 0/109-evals-passing failure mode the locomotion 2.0 gate was just fixed for.

*Suggested fix:* Gate the 2.0 m/s capability on a windowed/peak velocity metric (e.g. mean over steps after reaching cruise, or max sustained-1s velocity) or measure it on dedicated no-prey probe episodes; alternatively re-derive the threshold from actual bite-episode velocity distributions before the next behavior run.

**CF3. `configs/trex/behavior.toml:93` — missing-guardrail [medium / fix small]**

behavior.toml:93 keeps the absolute collapse_peak_floor = 100.0 that the same-day locomotion change removed as meaningless: measured do-nothing reward for stage 3 is 557 +/- 142 (10/10 full horizon), so the floor arms on the earliest full smoothing window of every run and certifies nothing about genuine collapse. Because behavior omits collapse_min_evals/patience/drop_fraction, it inherits defaults 12/8/0.4 — tighter than locomotion's tuned 20/10/0.5 and inside the parameter region early_stopping.py's own simulation showed aborts healthy runs on this project's eval noise — during the curriculum's largest reward-landscape shift. It also lacks the reference/fraction pair and statue_constants_physics_revision pin, so test_statue_constant_freshness.py cannot protect it. The false-abort risk is plausible rather than demonstrated (prior stage-3 runs completed under equivalent-or-tighter settings); the trivially-armed, uncalibrated floor is directly measured.

*Bites when:* locomotion.toml (same commit, 2026-08-23) documents that its old absolute collapse_peak_floor = 100.0 'sat 11x BELOW the measured do-nothing reward, so it armed on the first qualifying eval of any run, healthy or not, and encoded nothing about this stage', and replaced it with the reference/fraction pair plus a physics-revision pin. behavior.toml kept the absolute 100.0: with heading_weight 0.5 alone worth ~500/episode to the loaded locomotion policy, the floor arms on the first evaluation of every run. Because behavior omits collapse_min_evals/collapse_patience/collapse_drop_fraction, early_stopping.py defaults apply (min_evals 12 = ~600k steps, patience 8, drop_fraction 0.4) — stricter than locomotion's tuned 20/10/0.5 — during the transition where alive_bonus drops 0.5->0.0, forward_vel_weight halves and the sparse bite reward is undiscovered. An early ramp-window peak followed by the normal exploration dip (>40% below the rolling-median peak for 8 evals = 400k steps) aborts a healthy stage; conversely the floor certifies nothing about genuine collapse. collapse_settings_from_config's own docstring: 'The floor should be expressed RELATIVELY. An absolute number cannot survive a reward-function edit, and has now failed to arm twice.' behavior also lacks a statue reference and statue_constants_physics_revision pin, so the freshness CI cannot protect it.

*Suggested fix:* Measure the behavior-stage zero-action baseline with zero_action_baseline.py trex:3, replace collapse_peak_floor with the collapse_peak_floor_reference/fraction pair plus statue_constants_physics_revision, and set collapse_min_evals/patience/drop_fraction explicitly (at least as loose as locomotion's 20/10/0.5 given the stage-3 warmup/ramp).

**CF4. `environments/shared/config.py:200` — silent-failure [medium / fix small]**

load_stage_config performs no table-name validation, so a misspelled top-level table ([environment], [ppo_config]) silently yields empty kwargs and the stage trains on class/SB3 defaults while save_stage_config back-fills constructor defaults into stage_config.json, masking the omission; and cli.py _apply_overrides silently drops stage-scoped overrides whose stage number is not a config key. Caveat: CI's test_config.py has-common-keys tests would catch a table typo in committed stage 1/2/3 TOMLs — the unguarded surface is the recovery stage config (tests parametrize stages [1,2,3] only), any file loaded via config_path (e.g. stance_gate_report --config, Colab-edited configs), and the override path.

*Bites when:* Verified by probe with /home/user/venv/bin/python: a config whose [env] table is misspelled [environment] loads with env_kwargs == {} and no warning — the run silently trains against TRexEnv constructor defaults (forward_vel_weight 1.0, alive_bonus 0.1, all shaping terms 0.0), an entirely different experiment. A misspelled [ppo] table likewise yields SB3's own defaults, including the learning_rate 3e-4 that stance.toml records as 'caused instability'. Individual key typos inside [env]/[ppo] do fail closed later (TypeError at construction, verified), and [curriculum] typos are caught by gate_schema — but only where validate_gate_config actually runs (CurriculumManager and the JAX paths), not at load time. Compounding: cli.py _apply_overrides lines 50-56 silently drops stage-scoped overrides whose stage number is not a config key ('if stage_num in configs:' with no else), so '--override 4.env.x=1' (position 4 = behavior, whose key is legacy 3) is a silent no-op. save_stage_config then merges constructor defaults into the JSON (config.py lines 291-305), so the recorded artifact masks the omission.

*Suggested fix:* In load_stage_config, reject top-level tables outside {stage, env, ppo, sac, jax, curriculum} and warn (or fail when the file is a declared-manifest stage config) when [env] or the active algorithm table is empty; make _apply_overrides error on a stage reference that matches no config key.

**CF5. `configs/trex/sweep_ppo.json:6` — eval-validity [medium / fix small]**

Stage-1 sweep trial budget (6M in configs/trex/sweep_ppo.json line 6, and sweep_sac.json) is 55% of stance's 11M production budget and shorter than stance's ent_coef_decay_timesteps=7M. On the Vertex path (orchestration.py -> trial.py -> train_base), every trial trains under a truncated entropy schedule ending at ent_coef~0.0007 (verified at runtime) — the exact nonzero-floor regime stance.toml documents as the historical failure — so trial rankings do not predict 11M production behavior. On the Ray notebook path the situation is worse: ray_tune.py train_trial never strips ent_coef_end/ent_coef_decay_timesteps from ppo_kwargs before PPO(**alg_kwargs), and PPO's constructor rejects unknown kwargs, so trex stage-1 Ray trials would crash with TypeError at model construction (an interacting latent bug in ray_tune.py around line 727).

*Bites when:* ray_tune_sweep.ipynb takes TIMESTEPS_PER_TRIAL from this settings block, and the Vertex path resolves args.timesteps from it too (orchestration.py lines 466-477). stance.toml sets ent_coef_decay_timesteps = 7000000 with a lengthy justification that decaying ent_coef fully to 0.0 is load-bearing ('any floor sets a non-zero std equilibrium'); train_base builds the decay callback from that absolute value regardless of the trial's shorter run ('decay_timesteps=int(ppo_kwargs.get("ent_coef_decay_timesteps", total_timesteps))', line 644). A 6M trial therefore ends mid-decay at ent_coef ~0.0007, in exactly the never-converged high-std regime stance.toml documents as the historical failure, and covers only 55% of the production budget. Hyperparameters ranked by best_mean_reward at 6M under that regime do not predict 11M production behavior — the sweep selects for a schedule that will never be run. The stance.toml history itself notes the 6M anchor era was 'never revisited when stage 1 went to 10M'. sweep_sac.json stage1 carries the same 6M.

*Suggested fix:* Update stage1 sweep timesteps toward the production budget (or explicitly rescale schedule anchors like ent_coef_decay_timesteps proportionally for sweep trials), and record in the sweep JSON which stage budget it was sized against.

**CF6. `configs/trex/sweep_ppo.json:20` — eval-validity [low / fix medium]**

configs/trex/sweep_ppo.json stage1 is stale against the current stance config: env_nosedive_weight [2.0, 4.0] excludes the incumbent 1.5 (stance.toml:131), the shaping terms that now dominate the stance reward (smoothness 2.0, action_jerk 3.0, bilateral_support 0.6, leg/tail home-pose) are not in the space, and trials train on and are ranked by best_mean_reward — a metric stance.toml documents the zero-action statue maximizes — with no duty/full-horizon metric in configs/quality_scoring.toml [stage_1]. Correction to the finder: only nosedive excludes its incumbent — the other 8 swept env terms' current values ARE inside their ranges (alive_bonus 1.0 sits exactly on the min boundary), so "the six swept env terms are mostly no-longer-binding" is overstated; and quality_scoring.toml lives at configs/, not configs/trex/. Bites only when a new stage-1 sweep is launched, at which point ~30 trials x 6M steps select statue-shaped hyperparameters the actual stance gate rejects.

*Bites when:* stance.toml now has nosedive_weight = 1.5 (line 131), so the sweep cannot even sample the incumbent configuration — every trial trains a strictly more nose-conservative reward. The sweep also predates the stance rework: smoothness_weight (now 2.0), action_jerk_weight 3.0, bilateral_support_weight, leg/tail home-pose and saturation terms are absent from the space, while the six swept env terms are mostly no-longer-binding. Trials are ranked for HPT by best_mean_reward (scoring.py: 'the training objective (which remains best_mean_reward for ASHA / Vertex AI HPT)'), yet stance.toml documents that 'the statue is the reward optimum at every noise level' and the stage's real gate is stance_quality/v1 duty/full-horizon — so a 30-trial x 6M-step sweep selects hyperparameters that produce statue-like policies the actual gate rejects. quality_scoring.toml stage_1 post-ranking likewise has no unsupported-duty metric.

*Suggested fix:* Refresh the stage-1 space around the current stance.toml operating points, include the shaping weights that are now load-bearing, and rank stage-1 trials by the stance-gate statistics (unsupported duty UCB, full-horizon fraction) rather than raw reward.

### 3.7 Tests & CI

*Scope: test coverage, workflow path filters, job selections.*

**CI1. `environments/shared/evaluation.py:429` — duplicate of ER1.** Independently re-discovered by this finder: the periodic-checkpoint VecNormalize sidecar miss in `evaluate()`. See ER1; the double detection is itself weak evidence of how visible the defect is from multiple directions.

**CI2. `environments/shared/curriculum/checkpoints.py:342` — duplicate of TC1.** Independently re-discovered by this finder: the reset_num_timesteps checkpoint renumbering / retention pruning defect. See TC1; the double detection is itself weak evidence of how visible the defect is from multiple directions.

**CI3. `.github/workflows/python-ci.yml:376` — ci-coverage [medium / fix small]**

The sb3-gated resume-safety tests execute in no CI job: test-sb3's hardcoded selection (python-ci.yml:376-384) omits test_resume_load_path.py and test_stance_gate_report.py, and the only job that runs them (test-shared, via .[dev]) has no stable-baselines3, so 14 tests skip silently — 7 in test_resume_load_path.py (the real VecNormalize sidecar round trip and all 6 stage-entry-shaping tests pinning F3/F4) and 7 sb3-gated tests in test_stance_gate_report.py (not "all 5" — that file's other ~150 tests do run in test-shared). A regression re-introducing F3 (resumed policy under fresh normalization stats) or F4 (warm-up/ramp applied on resume_same_stage) merges green despite dedicated pinning tests existing in the repo.

*Bites when:* 7 tests in test_resume_load_path.py (the real VecNormalize periodic-sidecar round trip and all 6 TestStageEntryShapingCallbacks tests keyed on task_load_mode) and all 5 tests in test_stance_gate_report.py skip in every CI job (verified: train_base/curriculum/evaluation import cleanly with sb3 blocked, so the skips are silent). A regression re-introducing F3 (resumed policy trained under fresh normalization stats) or F4 (warm-up/ramp applied to a resume_same_stage load) merges green even though tests written specifically to pin those bugs exist in the repo.

*Suggested fix:* Add test_resume_load_path.py and test_stance_gate_report.py to the test-sb3 selection (or run the whole shared dir there), and/or assert an expected skip count so newly-authored sb3-gated files outside the list cannot silently join the never-run set.

**CI4. `.github/workflows/python-ci.yml:424` — ci-coverage [medium / fix small]**

Confirmed as stated, with counts refined: the test-jax-cpu job's four filename patterns exclude test_mjx_reset_plant_invariants.py (5 functions / 20 parametrized tests, module-level importorskip at line 27), the 3 (not 2) jax-gated tests in test_action_filter.py (skips at lines 124/128/136), and all of environments/trex/tests — including test_trex_mjx_reward_parity.py's 6 collected stance-parity tests (module-level importorskip at line 8) — while every job that does run those paths installs .[dev], which contains no jax. All of these tests pass locally with jax installed and skip silently (green) without it, so MJX reward/reset drift merges green despite existing coverage.

*Bites when:* 5 MJX reset-invariant tests, 2 MJX action-filter parity tests, and the 5 trex reward-parity tests (which pass locally in ~52s of jax compile: 33.9s + 16.2s + 1.7s) skip in every CI job. An MJX-side change that breaks stance reward parity with the Gymnasium env — the exact class of drift already known to threaten the recovery stage — merges green, and the JAX pilot then trains against silently divergent reward semantics. This is distinct from the known recovery-stage parity gap: here the EXISTING stance-parity tests are CI-invisible.

*Suggested fix:* Add test_mjx_reset_plant_invariants.py, test_action_filter.py and environments/trex/tests to the test-jax-cpu job (budget a few extra minutes of compile), or install jax-cpu in one test-trex matrix cell.

**CI5. `environments/shared/curriculum/checkpoints.py:62` — duplicate of TC2.** Independently re-discovered by this finder: the -inf best-tracker overwrite on resume. See TC2; the double detection is itself weak evidence of how visible the defect is from multiple directions.

**CI6. `environments/shared/cli.py:53` — silent-failure [medium / fix small]**

_apply_overrides (environments/shared/cli.py:50-67) has two confirmed defects for stage-scoped overrides: (1) a numeric stage token not present in the config dict (e.g. '7.ppo.learning_rate=1e-4') silently no-ops with no warning, so a typo'd override trains the full stage at unmodified hyperparameters; (2) a non-numeric stage token ('recovery.…' or '1b.…') falls through to the all-stages branch, which treats the stage id as a config section and raises an uncaught KeyError ('recovery_kwargs' / '1b_kwargs') at launch — so the semantic 'recovery' stage, the current training focus, cannot be targeted by --override at all even though --stage recovery works. TestApplyOverrides covers only int-keyed happy paths.

*Bites when:* The recovery stage is the active training focus, yet `--override recovery.ppo.learning_rate=...` (or `1b.env....`, since '1b'.isdigit() is False) crashes the launch with a bare KeyError, and a typo'd `--override 4.ppo.learning_rate=1e-5` trains the full multi-hour stage budget at the unmodified learning rate with nothing in the log but the absence of an 'override' line. TestApplyOverrides (environments/shared/tests/test_cli.py:30-67) covers only int-keyed happy paths on a synthetic 2-stage dict.

*Suggested fix:* Resolve the first token through the stage manifest (semantic ids and legacy numbers), error loudly on unknown stage or section, and add tests for missing-stage and semantic-stage-scoped overrides.

**CI7. `.github/workflows/python-ci.yml:22` — ci-path-filter [medium / fix small]**

The push and pull_request path filters in .github/workflows/python-ci.yml (lines 6-19 and 22-35) omit repo-root conftest.py — the sys.path bootstrap that every CI pytest invocation imports before collecting tests — so a conftest-only PR triggers no python-ci jobs yet can break all subsequent CI runs; the same filters omit .github/workflows/deploy.yml, scripts/**, Dockerfile, and .pre-commit-config.yaml, though those files are not exercised by any python-ci job so their omission is largely inert; docs/hardware|investigations|reviews subdirectories are also unmatched (refinement of the known docs/*.md top-level-only issue).

*Bites when:* A PR editing only /conftest.py changes test collection behavior for every subsequent CI and local pytest run yet triggers no python-ci jobs and merges without a single check; same for a deploy.yml edit (deploy itself runs, python-ci does not). Also, only PRs targeting main are filtered in at all (branches: [main]).

*Suggested fix:* Add conftest.py, .github/workflows/**, and docs/** to both push and pull_request path lists.

**CI8. `.github/workflows/deploy.yml:3` — deploy-gating [low / fix small]**

deploy.yml is fully independent of python-ci: it has no paths filter (every push/PR to main builds the site) and its deploy job is gated only by needs:build + ref==main, so a direct push (or admin merge past a red check) to main with a stale website/src/data/species.generated.json deploys to the public site even while python-ci's `species_catalog --check` fails in parallel — the docusaurus build itself performs no staleness validation. Minor evidence correction: the pull_request trigger does specify branches: [main]; only the paths filter is missing.

*Bites when:* A direct push to main (or an admin merge past a red check) with a hand-edited or stale species.generated.json deploys to the public site immediately while python-ci's `python -m environments.shared.species_catalog --check` fails in parallel. Secondary waste: every python-only PR burns a full npm ci + docusaurus build, and PR builds share `concurrency: group: "pages"` with production deploys (cancel-in-progress: false), so PR builds can queue ahead of a main deploy. Deploy cannot push to the repo (contents: read) and regenerates nothing, so no stale artifact is ever written back — the risk is deploying, not committing.

*Suggested fix:* Add a website/** (+ workflow file) path filter, and gate the deploy job on the python-ci run for the same SHA (workflow_run or required checks).

**CI9. `.github/workflows/python-ci.yml:27` — notebook-validation [low / fix small]**

notebooks/** triggers the full CI matrix (python-ci.yml lines 11 and 27) but no job lints, parses, or validates any notebook: coverage is limited to substring greps of three notebooks, one exec'd preflight cell (test_zero_action_baseline.py:124), and content checks; google_drive_summary.ipynb is untested entirely, and the resume escape-hatch cell (sb3_training.ipynb cell 29, the sidecar guard) is only substring-grepped, never parsed or executed — so a symbol/kwarg rename can break the Colab entry point while every CI job stays green.

*Bites when:* A refactor renames a train_stage kwarg or a symbol used by the resume cell; every CI job passes; the break surfaces hours into the next paid Colab session — precisely the environment where recovery-stage pilots are being run. A notebook-only PR gets a green check that implies validation which never happened.

*Suggested fix:* Add a cheap test/CI step that nbformat-validates all four notebooks and ast.parses every code cell (no execution needed).

### 3.8 Operational scripts

*Scope: sweep runner, report scripts, requirements, Vertex path.*

**OP1. `environments/shared/scripts/sweep/trial.py:201` — budget-waste [high / fix small]**

Sweep trials call train() without task_load_mode, so warm-started trials validate under the default resume_same_stage; launch-all's stage-1-to-2 and 2-to-3 chaining passes the previous stage's fingerprinted best_model.zip via --load, and every cross-stage worker raises TaskFingerprintError (differing sections ['stage','env']) at startup — multi-stage sweeps fail after stage 1 with checkpoints minted by current (post-2026-08-15) code, burning Vertex trial quota, and the trial CLI has no flag to override the mode.

*Bites when:* launch-all completes the stage-1 sweep and submits stage 2 with --load /gcs/.../stage1/<best>/models/best_model.zip. That checkpoint carries the stage-1 task fingerprint (train() attaches it to every model saved since 2026-08-15). Each stage-2 worker starts, run_trial calls train() without task_load_mode, _create_or_load_model validates with mode=resume_same_stage, and validate_recorded_task raises TaskFingerprintError. Verified empirically in this session: validate_recorded_task(fp_stage1, fp_stage2, mode='resume_same_stage') raises "was trained on a different task than it is being resumed into (differing sections: ['stage', 'env'])". Every warm-started trial crashes at startup after container spin-up, the HPT job ends FAILED, GPU-hours and trial quota are burned, and the sweep can never advance past stage 1 (train_curriculum passes task_load_mode="initialize_next_stage" at train_base.py:1379 — the sweep path was never given the same treatment). The sweep trial CLI has no flag to work around it.

*Suggested fix:* In run_trial, pass task_load_mode="initialize_next_stage" when the trial's --stage differs from the loaded checkpoint's recorded stage (or add a --load-mode passthrough set by _submit_stage_sweep when chaining stages), mirroring train_curriculum's call at train_base.py:1368-1380.

**OP2. `environments/shared/scripts/sweep/results.py:85` — eval-validity [high / fix medium]**

The sweep pipeline (results.py _extract_thresholds/_evaluate_curriculum_gate, plus the offline collector and ray_tune path) evaluates stage pass/fail from the four legacy threshold keys only and has no knowledge of gate_kind/gate_schema_version. For the trex stance stage (gate_kind = "stance_quality/v1"), only min_avg_reward = 2100 survives extraction — a value the TOML explicitly labels a collapse rail, not a gate — and since min_avg_episode_length was replaced by min_full_horizon_fraction, the other three thresholds are all None, so the sweep gate is reward-only: the documented statue (3271.8) and the chatterer the gate exists to reject (2133.4) both mark stage_passed=True, and _best_trial_model_path then warm-starts stage 2 from the most statue-like trial while CSVs and pass/fail plots report a retired gate.

*Bites when:* Verified in this session: _extract_thresholds(load_stage_config('trex', 1)) returns (2100.0, None, None, None). A trex stage-1 sweep therefore marks stage_passed=True for any trial with best_mean_reward >= 2100 — the stance TOML documents that the statue scores 3271.8 and the chattering policy the gate must reject scores 2133.4, both clearing 2100. _best_trial_model_path then picks the highest-reward 'passing' trial (the most statue-like) and launch-all chains it into stage 2 as the stance winner, while the actual gate criteria (unsupported-duty UCB, full-horizon fraction, 40-episode panel) are never evaluated. Sweep CSVs and pass/fail plots report a gate the project has explicitly retired.

*Suggested fix:* Route sweep gate evaluation through the gate_kind/gate_schema_version declaration (the same resolver the training-side gates use) instead of the four hardcoded threshold keys; at minimum, refuse to compute stage_passed for a stage whose [curriculum] declares a gate_kind the sweep cannot evaluate, rather than silently substituting the reward rail.

**OP3. `environments/shared/scripts/sweep/__main__.py:38` — curriculum-coverage [medium / fix medium]**

Refinement of the known int-only stage-ref limitation, with two line-number corrections: the recovery stage ('1b'/'recovery') is unreachable from all sweep/Vertex/Ray entry points (argparse rejects it in trial and launch modes; setup_vertex_ai.sh and the Ray notebook only accept 1-3), and widening argparse alone is insufficient — orchestration.py:909 int(stg_key) crashes on a non-numeric resume key, orchestration.py:949 iterates range(1,4), submit.py:269 mints 'stage{stage}' paths (and :439 logs with %d, a hard crash on str), search_space.py/_is_per_stage and configs/trex/sweep_ppo.json key stage1-3 only, and scoring.py:95 looks up f"stage_{stage}" in configs/quality_scoring.toml (repo root, not configs/trex/), which has no recovery section. Cited lines submit.py:523 and scoring.py:1072 do not exist (files are 440 and 256 lines); the quoted code lives at submit.py:269 and scoring.py:95.

*Bites when:* Any attempt to sweep or Vertex-train the recovery stage dies at argparse ('invalid int value'). Even after widening the two argparse types, the pipeline breaks downstream: launch_all_stages hardcodes `for stage in range(1, 4)` (orchestration.py:949) and resume does `stg_num = int(stg_key)` (orchestration.py:909, crashes on a 'recovery' state key); output paths are minted as f"/gcs/{bucket}/sweeps/{species}/stage{stage}" (submit.py:523, would produce 'stagerecovery' instead of the stage_label convention); search-space files are keyed stage1/stage2/stage3 (search_space.py:_is_per_stage, configs/trex/sweep_ppo.json has no recovery block); quality scoring looks up f"stage_{stage}" in quality_scoring.toml (scoring.py:1072); setup_vertex_ai.sh:134 rejects non-[123] stage numbers (`if [[ ! "${STAGE_NUM}" =~ ^[123]$ ]]`); the Ray notebook exposes STAGE as an integer @param. With Colab's PPO.load broken and recovery being the current focus, none of the cloud launch paths can train the stage the team is working on.

*Suggested fix:* Adopt cli._parse_stage_ref for --stage in trial/launch (as collect-results already does), derive stage paths via stage_manifest.stage_label/stage_dirname, key search-space and scoring configs by stage id, iterate launch-all over the species manifest rather than range(1,4), and widen setup_vertex_ai.sh's prompt.

**OP4. `environments/shared/scripts/sweep/ray_tune.py:831` — correctness [medium / fix small]**

Latent, not currently reachable: both existing entry points (sweep CLI with --stage choices=[1,2,3] and the ray_tune_sweep notebook's integer STAGE param) only pass ints today, so no current run crashes. But the Ray trainable train_trial is the only sweep backend that could run a recovery sweep, and its warm-up guard `if stage > 1 and load_path:` (ray_tune.py:831) raises TypeError on any semantic stage ref — even with empty load_path, since `stage > 1` is evaluated first — after config load, env creation, and model creation all succeed (the shared helpers are already typed int|str). Checkpoint/final naming (lines 813, 863) would additionally mint 'stagerecovery_*' artifacts, breaking the stage_label convention downstream tooling keys on. train_base.py:915 already fixed the same guard via manifest position; ray_tune.py was missed.

*Bites when:* load_all_stages('trex') already returns a 'recovery' key (verified: keys are 1, 2, 3, 'recovery'), so wiring _stage='recovery' into run_ray_sweep gets all the way through config load, env creation and model load, then crashes with `TypeError: '>' not supported between instances of 'str' and 'int'` during callback setup — every trial of the sweep errors identically. Additionally name_prefix=f"stage{stage}" (line 813) and f"stage{stage}_final" (line 863) would mint 'stagerecovery_*' artifacts, violating the stage_label convention ('a recovery run writes recovery_final.zip', stage_manifest.py) that downstream recovery tooling keys on.

*Suggested fix:* Mirror train_base: resolve the manifest position once (`load_stage_manifest(species).resolve(stage).position`) for the warm-up guard, and use stage_label(stage) for checkpoint prefixes and final-artifact names (also update export_best_trial's glob patterns to match).

**OP5. `environments/shared/scripts/sweep/ray_tune.py:757` — missing-guardrail [medium / fix small]**

The Ray sweep path (ray_tune.py train_trial) neither validates task fingerprints on warm-start loads nor attaches the current stage's fingerprint to minted checkpoints — and, verified at runtime, a warm-started trial re-saves the source checkpoint's stale fingerprint verbatim via SB3 attribute persistence, so Ray-minted artifacts are either unfingerprinted (fresh trials; unloadable once the train_base allow_unfingerprinted valve is tightened per plan §W5) or mislabeled with the wrong stage's task fingerprint (warm-started trials), while a wrong-task warm-start loads silently with only plant-identity validation. The best_model promotion in ray_orchestration.py likewise writes a plant-identity sidecar but no task_fingerprint.json.

*Bites when:* A Ray trial warm-started from the wrong stage's (or wrong task variant's) checkpoint loads silently — the exact scenario the fingerprint layer exists to catch, and the mirror image of the Vertex path which fails closed (finding 1). Worse, best_model.zip / stageN_final.zip minted by Ray sweeps carry no task fingerprint, so later `train --load <ray best>` passes only through the allow_unfingerprinted transition valve with a warning; when that valve is tightened to fail-closed (planned, train_base.py comment 'Tighten to fail-closed once fingerprinted checkpoints are the norm... plan §W5'), every Ray-sweep checkpoint minted today becomes unloadable.

*Suggested fix:* In train_trial, derive the stage task fingerprint (derive_stage_task_fingerprint), validate warm-start loads with validate_model_task (initialize_next_stage for cross-stage), and attach_task_fingerprint alongside the existing attach_plant_identity.

**OP6. `environments/shared/train_base.py:1286` — duplicate of TC7.** Independently re-discovered by this finder: the `curriculum` command silently skipping the recovery stage. See TC7; the double detection is itself weak evidence of how visible the defect is from multiple directions.

**OP7. `environments/shared/scripts/joint_excursion_report.py:86` — eval-validity [medium / fix small]**

joint_excursion_report.py:86 and action_bound_report.py:144 silently evaluate the policy on unnormalized observations for any periodic checkpoint (<prefix>_<steps>_steps.zip), because their sidecar guess (<stem>_vecnorm.pkl) can never match CheckpointCallback's <prefix>_vecnormalize_<steps>_steps.pkl naming and no warning is emitted; observation_ablation_report.py:221 has the same never-matching guess but does print "VecNormalize: NONE — results are meaningless" in its header rather than being silent. train_base.py's _resolve_vecnorm_sidecar already implements the correct two-convention lookup (added after this exact mismatch silently broke resumes, per its docstring), and stance_gate_report.py treats the missing-sidecar case as fatal — the diagnostic scripts just never adopted either fix.

*Bites when:* Run checkpoints are saved by SB3's CheckpointCallback as <stage_label>_<steps>_steps.zip with the VecNormalize sidecar named <stage_label>_vecnormalize_<steps>_steps.pkl (run layout stated in the project context; CheckpointCallback save_vecnormalize=True naming). The guess computes <stage_label>_<steps>_steps_vecnorm.pkl, which never exists, so for every periodic checkpoint — the exact artifacts analyzed mid-run or after a Colab crash, and the primary CLI fallback now that Colab's PPO.load is broken — the script runs the policy on raw observations with zero warning. The resulting saturation/excursion/ablation tables are garbage-but-plausible (a policy fed unnormalized obs rails its actuators), and a config decision like resizing action bounds could be made from them. stance_gate_report.py:126-128 already treats this case as a hard error ('Running without --vecnorm would evaluate the policy on unnormalised...'), so the correct pattern exists in the same directory.

*Suggested fix:* Extend the guess to the CheckpointCallback naming (replace '_<steps>_steps' stem with '_vecnormalize_<steps>_steps.pkl'), and when no sidecar is found either fail like stance_gate_report or print a loud UNNORMALIZED EVAL banner in the report header.

**OP8. `environments/shared/scripts/sweep/results.py:54` — silent-failure [medium / fix small]**

A NaN best_mean_reward passes _evaluate_curriculum_gate (results.py:49,54): `best_reward is not None` is True for NaN and `NaN < threshold` is False, and the NaN guard in _get covers only aux metrics. Via collect_ray_results (ray_tune.py:1094) this marks Ray trials that errored before their first tune.report — and any row whose best_mean_reward column is NaN — as stage_passed=True, corrupting the CSV, pass/fail plot, orchestration pass counts, and persisted sweep state. Model selection and stage chaining are unaffected (NaN never beats -inf in _best_trial_model_path). Note: typical ASHA-pruned trials retain a real best_mean_reward from their last per-eval report, so the trigger is early-erroring trials rather than routine pruning; the finder's "ASHA-pruned" framing slightly overstates which trials hit it.

*Bites when:* Verified in this session: _evaluate_curriculum_gate(float('nan'), {}, 100.0, None, None, None) returns (True, []). collect_ray_results feeds pandas NaN straight into this function (`row["best_mean_reward"] = rt_row.get(metric)`), and for trex stage 1 the extracted thresholds are (2100.0, None, None, None) — reward is the only criterion — so a Ray trial that errored or was pruned before its final report gets stage_passed=True with no criteria evaluated. Pass-rate plots, CSV rows, and launch_all's 'Trials: %d total, %d passed' log overstate sweep success; _best_trial_model_path won't select it (NaN > x is False) but the recorded evidence is wrong.

*Suggested fix:* Treat a non-finite best_reward as failing: `if best_reward is None or (isinstance(best_reward, float) and math.isnan(best_reward)):` set passed=False with reason 'no finite reward reported'.

**OP9. `environments/trex/requirements.txt:2` — reproducibility [low / fix small]**

The three per-species requirements.txt files leave mujoco unpinned (>=3.0.0), so installing from them today yields mujoco 3.12.0 instead of the canonical 3.10.0; a subsequent training start dies at current_plant_identity with the misleading 'generated plant manifest is stale for trex' error (reproduced). The defect is real but softer than stated: the error's own recommended next step (the plant-contract check) names the true cause ('requires MuJoCo 3.10.0, found 3.12.0'), and the silent-training-on-non-canonical-dynamics branch does not occur with 3.12.0 because option-field fingerprinting makes the digests differ — training halts before any budget is wasted. Net impact: a confusing two-hop diagnosis and an untrustworthy install file, not corrupted results.

*Bites when:* A collaborator does `pip install -r environments/trex/requirements.txt` (the file sits beside the scripts and looks authoritative) and gets a newer mujoco. current_plant_identity recomputes layer digests at train start without the canonical-version guard (only check/write use require_canonical_mujoco=True), so training either dies with the misleading 'generated plant manifest is stale for trex; run the plant-contract check before training' (pointing at a manifest problem, not the interpreter's mujoco), or — if the digests happen to coincide — trains against non-canonical compiled dynamics. Remaining three-way drift is smaller than feared: notebooks and pyproject agree on mujoco==3.10.0 and SB3>=2.2.0/ray[tune]>=2.55,<3; torch is pinned only in CI's real-backend job (2.13.0 CPU) while the Dockerfile and Colab float it.

*Suggested fix:* Pin mujoco==3.10.0 in the three species requirements.txt files (or delete them in favor of `pip install -e .[train]`), and have current_plant_identity include mujoco.__version__ vs canonical in its stale-manifest error message.
---

## 4. Cross-cutting probes (critic round)

The completeness critic's own assessment, verbatim: the eight areas were consistent and the
defended paths held up under spot-probes (the replay selector requires matched vecnorm sidecars;
VecNormalize gamma is wired from PPO gamma; gate-failure disconnect fires only after artifacts
and bundle save). What no single area owned: nobody had executed the real CLI end to end;
nobody had connected the notebook's own admission that stages exceed the session cap to the
resume machinery's actual behavior; every certification decision is single-seed; and the
provenance layer cannot see resumes at all.

### 4.1 End-to-end execution of the real CLI

This probe actually ran `train_sb3.py train --stage recovery` (cold and warm-started) and `--stage 1` at smoke budgets — the only part of the review that executed the production entry point end to end.

**EE1. `environments/shared/eval_diagnostics.py:833` — silent-failure [high / fix small]**

For a recovery_quality/v1 stage (configs/trex/recovery.toml declares none of the six legacy threshold keys), StageGatePlateauCallback._process_evaluation's early return at eval_diagnostics.py:833-834 fires on every evaluation, so _record_stance_scalars (line 841) — the sole feeder of _record_gate_progress and of every diagnostics/eval_* TensorBoard tag — never runs: gate_progress.npz (the artifact train_base.py:508-511 designates as the mid-run check) is never written and plateau warnings can never fire for the recovery stage. Slight overstatement in "trains completely dark": SB3's standard eval/mean_reward, eval/mean_ep_length, evaluations.npz, and rollout DiagnosticsCallback output still exist; what vanishes is all gate-relevant diagnostics (duty panel, action DC/AC, reward terms, gate_met/plateau scalars, gate_progress.npz). Also note the suggested reorder restores recording but plateau warnings would remain impossible for recovery until recovery-specific metrics join _METRIC_PRIORITY.

*Bites when:* Executed and observed: two smoke runs of `train_sb3.py train --stage recovery` (cold and warm-started) produced NO gate_progress.npz and ZERO diagnostics/eval_* tags in TensorBoard (verified with EventAccumulator: stance run has eval_gate_met, eval_plateau_active, eval_duty_episodes, action stats, etc.; recovery run has an empty list) — even though StageAwareEvalCallback captured all the per-episode duty/action data. On the next production 3M-step recovery run, the artifact train_base.py itself documents as "the artifact a human (or a Drive reader) checks mid-run" (train_base.py:508-510) never appears, plateau warnings can never fire, and a stalled run is indistinguishable from a progressing one until the 3M budget is spent. Interacts with in-flight recovery-gate work (recovery.toml / gate files are out of scope), but the early-return coupling of gate-progress RECORDING to plateau-metric CONFIGURATION is in in-scope eval_diagnostics.py.

*Suggested fix:* In _process_evaluation, call _record_stance_scalars(index, panel) (and record eval_episode_count) BEFORE the `if not metrics: return` early-out, so gate_progress.npz and the eval diagnostics are written whenever panel data exists, independent of whether any plateau-followable threshold is configured; the plateau logic alone should depend on `metrics`.

**EE3. `environments/shared/curriculum/advancement.py:531` — resume-safety [high / fix small]**

A checkpoint saved during a stage-entry warm-up pickles _ent_coef_warmup_active=True on the model; a default resume_same_stage --load (train_base.py:673 returns no StageWarmupCallback, and _create_or_load_model never clears the marker) restores it, so EntCoefDecayCallback defers on every step for the whole resumed run — ent_coef stays fixed at the config base (0.005, since PPO.load kwargs override the pickled 0.02) and the configured decay (e.g. recovery 0.005→0.0 over 2M) silently never runs, with the stale marker re-pickled into every descendant checkpoint. The schedules.py header comment's claimed mitigation ("every stage>1 --load path re-adds the warm-up callback") is false for the resume_same_stage path.

*Bites when:* Executed and observed: resumed `--stage recovery --load recovery_warm/models/recovery_final.zip` (default resume_same_stage); the log shows only 'EntCoefDecay: deferring to the stage warm-up before capturing the base ent_coef' and after the run the newly saved recovery_final.zip STILL carries warmup_marker=True — the decay never captured a base and never ran, and the poison propagates into every checkpoint of the resumed lineage. Production scenario: the warm-started 3M recovery run dies inside its 100k warm-up window (Colab disconnect, or the known PPO.load segfault forcing a restart; KeyboardInterrupt also saves recovery_final.zip, and best_model/robust_best_model are minted at the 50k eval which is inside the window). The documented resume then trains the remaining ~3M steps at a fixed ent_coef with the configured 0.005→0.0 decay (2M anchor) silently never applied — different exploration schedule than every fresh run, invisible except for one INFO line. Distinct from the tracked resume-family items (reset_num_timesteps / best-model overwrite / schedule snap-back / retention).

*Suggested fix:* Clear the marker on load for resume_same_stage (e.g. in _create_or_load_model: if no warm-up callback will run, setattr(model, ENT_COEF_WARMUP_MARKER, False)), or exclude the attribute from save, or have EntCoefDecayCallback treat a marker with no live StageWarmupCallback as stale.

**EE2. `environments/shared/reporting/stage_artifacts.py:736` — misleading-artifact [medium / fix medium]**

_apply_stage_gate (stage_artifacts.py:736) forwards neither stage_dir NOR recovery_successes_by_seed to evaluate_stage_gate, so from the single shared entry point (notebook cell 21 and sweep trial worker) a recovery_quality/v1 stage always records the fail-closed 'carried no stage_dir' refusal and the frozen gate_resolution.json pipeline is unreachable. The finder's suggested fix is incomplete: forwarding stage_dir alone only moves the refusal to the next check (gates.py:327-335, no pushed-panel evidence), because neither _apply_stage_gate nor generate_stage_artifacts has any parameter to carry RecoveryPanelEvidence.successes_by_seed(); full wiring must plumb both inputs.

*Bites when:* The next recovery pilot run finishes on Colab; cell 21 calls generate_stage_artifacts(stage='recovery', stage_dir=dir_r, ...) and prints the 'Recovery pilot gate record'. Regardless of whether the in-flight resolver work has frozen a valid gate_resolution.json into dir_r, the verdict is always this no-stage_dir refusal, because the one shared gate call site drops the directory. The whole resolver pipeline is unreachable from the single entry point both the notebook and sweep worker use (test_recovery_gate_wiring.py even has a test named test_a_caller_that_passes_no_stage_dir_refuses documenting this caller class). Fail-closed, so not unsafe — but the recorded pilot verdict and stage_summary are permanently misleading and the gate can never pass. Reported from the in-scope side; evaluate_stage_gate itself lives in in-flight reporting/gates.py.

*Suggested fix:* Give _apply_stage_gate a stage_dir parameter and pass it from generate_stage_artifacts (stage_dir is already a local there); forward it to evaluate_stage_gate.

**EE4. `environments/shared/train_base.py:1152` — budget-waste [low / fix small]**

train() unconditionally calls _report_hpt_metrics after every run (train_base.py:968), which runs a hardcoded 50-episode quality eval (line 1152) plus 30-episode velocity/success eval (line 1172); no train() parameter, CLI flag, config key, or --override path can reduce or skip it — the only escapes are Ctrl-C mid-eval (model is saved beforehand, but metrics.json is lost) or not calling train(). A papercut on smoke/debug/CI invocations; negligible on real 3M-step runs, and the panel does produce the metrics.json the sweep tooling consumes.

*Bites when:* Measured on the executed smokes: the 8k-step stance smoke spent 2m21s of its 5m11s wall time (~45%) in this post-training eval, and the recovery smokes similarly (~50s of ~2m05s). Any debug, CI, or smoke invocation of the real CLI — the runs that verify a fix before committing the next 3M-step recovery budget — pays minutes of unskippable serial evaluation; the only workaround is not running train() at all. Operational-friction evidence requested by the probe: the smoke needed no eval_freq surgery (panels were affordable), but this stage had no knob when it would have needed one.

*Suggested fix:* Plumb a --post-eval-episodes (or reuse stance_report_episodes-style curriculum override) through train() into _report_hpt_metrics, with 0 skipping the quality/velocity panels while still writing metrics.json.

### 4.2 Colab session cap & ungraceful disconnect

Stance (11M) and locomotion (8M) budgets exceed the ~24 h Colab session cap at observed throughput, so every production run is a multi-session run and the ungraceful-reclaim path is the common case, not the exception.

**CO1. `notebooks/sb3_training.ipynb:1647` — resume-safety [medium / fix small]**

The resume cell (cell 29, raw lines ~1630-1660) deterministically selects the max-step checkpoint pair with no zip/pickle integrity validation and no fallback to the 4 older complete step-points retention preserves; after an ungraceful session-cap reclaim that leaves the newest pair orphaned or truncated on Drive (the graceful-only flush in disconnect_runtime never ran), every resume retry fails — loudly but with misleading errors (the sidecar-missing message blames 'fresh normalization statistics'; the truncated-zip AssertionError resembles the known Colab L4 PPO.load segfault) — until the user manually deletes the poisoned pair or edits the cell. Progress is not lost and the failure is not silent, but the mandatory multi-session resume path stalls and the recovery procedure is undocumented, risking a wasted debugging session or a from-scratch stage restart.

*Bites when:* Stance (11M) and locomotion (8M) budgets exceed the ~5.49M-steps/session cap (cell 28's run 20260821_142144), so every production run MUST pass through this cell. A reclaim/preemption during or shortly after a checkpoint write leaves either (a) newest zip present, sidecar absent -> resume raises FileNotFoundError on every retry, or (b) both present but the zip/pkl truncated by the unflushed Drive FUSE upload -> verified crashes: PPO.load on a half-truncated zip raises AssertionError 'No data found in the saved file', on a zero-byte zip ValueError "wasn't a zip-file", and VecNormalize.load on a truncated pkl raises UnpicklingError 'pickle data was truncated' (reproduced with SB3 2.9.0 in /home/user/venv). In all cases the cell never tries the 4 older complete step-points that CheckpointRetentionCallback (DEFAULT_MAX_CHECKPOINTS=5) deliberately preserved, and no error message tells the user to delete the poisoned newest pair, so the run stalls with GPU-days of progress inaccessible until manual Drive surgery.

*Suggested fix:* In cell 29, iterate candidates from newest to oldest: skip a step-point whose sidecar is missing (with a warning naming the orphan) and validate both files before use (zipfile.ZipFile(...).testzip() / namelist() contains 'data'; pickle header readable, or simply try alg_cls.load in a try/except and fall back). Same validation belongs in _resolve_vecnorm_sidecar/_create_or_load_model for the CLI --load path.

**CO2. `notebooks/sb3_training.ipynb:614` — silent-failure [medium / fix small]**

On Colab+Drive runs, Drive flushing (drive.flush_and_unmount in disconnect_runtime, notebooks/sb3_training.ipynb cell 14) runs only on graceful halts (gate-failure raises in cells 19/24/27, completion cell 39); nothing flushes during training (PeriodicTbSyncCallback is GCS-gated, train_base.py:277/604-605), and SB3's CheckpointCallback (train_base.py:529-534) streams the checkpoint zip + vecnormalize pkl directly onto the Drive FUSE mount with no local-stage/atomic-publish — unlike evaluations.npz, which got exactly that treatment for exactly this hazard (train_base.py:496-499, file_io.py docstring). An ungraceful runtime reclaim landing within DriveFS's post-write sync window can therefore lose or truncate the newest periodic checkpoint pair, the sole artifact the resume cell (cell 29) depends on. Overstatements: the window is narrow (small ~4 MB files, ~2.2h checkpoint cadence — the documented reclaim of run 20260821_142144 did NOT lose its newest pair), so this is a probabilistic hazard, not one that bites every session; and a truncated zip fails PPO.load loudly and recoverably, not a permanent resume wedge. Cost when it hits: ~500k steps (~2h GPU) of extra reverted progress or a manual-intervention resume failure.

*Bites when:* The evaluations.npz/diagnostics.npz/gate_progress.npz atomic publishes protect against a reader seeing a torn rename, but nothing forces DriveFS to upload the checkpoint pair written minutes before a reclaim: a preemption or OOM shortly after the e.g. 5.5M-step save leaves Drive holding only the 5.0M pair (or a truncated 5.5M pair, per the repo's own file_io.py warning), silently reverting the session's resumable progress by an extra 500k steps (~2.2 h of L4 GPU at the observed 5.49M steps/session) on top of the inherent snap-back, or wedging resume entirely (previous finding). Since stance needs ~3 sessions and locomotion ~2, this un-flushed window is crossed on every production run, and the design's only mitigation runs precisely on the halt paths where the runtime would have survived long enough to sync anyway.

*Suggested fix:* After each periodic checkpoint lands (e.g. a small callback placed after CheckpointCallback, on the same save_freq cadence), stage the pair via write-local + atomic_copy to Drive and/or call google.colab.drive.flush() (or os.sync()) so at most the in-progress interval is exposed; alternatively write checkpoints to local disk and atomic-publish, as already done for evaluations.npz.

**CO3. `environments/shared/result_bundle/provenance.py:234` — resume-safety [medium / fix medium]**

The SB3 notebook's mandatory cross-session resume procedure (cell 28: re-run sections 1-4 with RUN_ID set) re-runs initialize_result_bundle (provenance.py:209-234), which by intentional design rejects any drift in repository_commit, platform, python_version, or tracked dependency versions; because setup cell 3 clones unpinned main and pip-installs unpinned stable-baselines3/gymnasium/numpy, any push to main, PyPI release, or Colab image update between sessions makes section 2 raise 'run directory already belongs to a different run' for what is the same run, blocking resume before the resume cell runs. The commit-checkout remedy is documented only in RESULT_BUNDLES.md's JAX-reuse paragraph, not in the notebook, and no documented remedy exists for dependency or platform drift.

*Bites when:* A stance run spans ~3 sessions over 2-3 calendar days. Any commit pushed to main in that window (this is an actively developed repo), any SB3/gymnasium/numpy release picked up by the unpinned pip install, or a Colab image update changing platform.platform()/python_version makes the section-2 cell raise ResultBundleError 'run directory already belongs to a different run' — for what is in fact the same run in a new session. The resume path the session cap makes mandatory is thus blocked, with an error that misdiagnoses the situation and no in-notebook way to pin the recorded commit; the tempting workaround (deleting provenance.json) destroys the run's provenance, and the correct one is undocumented in the notebook that needs it.

*Suggested fix:* In cell 3, when RUN_ID names an existing run, read provenance.json's repository_commit and 'git checkout' it (and pip install that revision) before importing the package; or have initialize_result_bundle distinguish identity fields (species/algorithm/seed/plant) that must match from environment fields (commit/platform/deps) that should produce a loud warning plus a recorded environment-drift entry on an explicit resume. At minimum, cell 28's instructions must include the commit-pinning step RESULT_BUNDLES.md already prescribes.

**CO4. `environments/shared/cli.py:238` — budget-waste [medium / fix small]**

save_freq=500000 (cli.py:238/157/189, notebook cell 14) sets checkpoint-resume granularity at ~2.2 h of L4 wall clock per interval, and with the documented resume-after-session-cap workflow (notebook cell 28: resume trains 'stage timesteps minus checkpoint steps') each of the ~4 session boundaries in a full T-Rex curriculum re-trains up to one interval — observed once already (run 20260821_142144: 488,640 steps discarded). The only recorded rationale for 500k is storage (train_base.py:537-539), which CheckpointRetentionCallback (keep 5) has made moot. Minor correction to the finder: the cost is not fully invisible — notebook cell 28 records the observed loss — but no doc, comment, or config prices the cadence against the session cap or revisits it post-retention; also note lowering save_freq to 100k shrinks the manual-rollback window span from 2.5M to 0.5M steps, a trade-off the fix should state.

*Bites when:* At observed throughput each 500k interval is ~2.19 h of wall clock. Every session boundary discards up to one interval (observed: 488,640 steps = 8.9% of the session) and resume snaps back to the 500k multiple, so a full T-Rex curriculum (stance 11M -> 3 sessions, locomotion 8M -> 2, behavior 8M -> 2; recovery 3M is the only stage that fits one session) re-trains ~4-9 h of GPU per run as a matter of course. Meanwhile evaluation evidence is already written every 50k steps (eval_freq=50000, ten times finer), showing the I/O budget tolerates a finer cadence; retention (keep 5) caps storage regardless of save_freq. Nothing surfaces this cost, so the budget leaks invisibly on every production run.

*Suggested fix:* Lower save_freq (e.g. 100k, cutting worst-case discard to ~26 min while retention keeps storage constant at 5 step-points), or at least document the ~2.2 h/interval loss window beside the save_freq defaults so the cadence is a priced decision rather than an inherited one.

**CO5. `environments/shared/train_base.py:285` — silent-failure [low / fix small]**

On Colab+Drive, TensorBoard event files are streamed in-place to the FUSE mount and held open for the entire stage — the local-buffer + periodic-sync protection exists only for GCS paths — so a hard reclaim loses the un-uploaded tail (potentially most) of the stage's TensorBoard logs, the exact failure the GCS path was built to prevent.

*Bites when:* A stance session hits the ~24 h cap; the events file under <stage_dir>/tensorboard/ has been open and appended to all session, and DriveFS only guarantees upload on flush/close — which the hard reclaim skips (disconnect_runtime's flush never runs). The Drive copy ends truncated at whatever DriveFS happened to sync, losing hours of TB history for the very sessions that need post-mortem inspection. Impact is moderated because diagnostics.npz/gate_progress.npz/evaluations.npz duplicate the load-bearing series atomically, but TB-only channels (per-component diagnostics/* scalars, grad norms) are lost, and a reader comparing TB against the npz record sees inconsistent histories for the same stage.

*Suggested fix:* Treat any non-local log_path (Drive as well as GCS) as remote: buffer TensorBoard locally and reuse PeriodicTbSyncCallback (it is a plain directory copy, not GCS-specific) on the same save_freq cadence.

### 4.3 Single-seed decision validity

Every irreversible curriculum decision to date — gate raises, config locking, publication — rests on one training seed, while the repo’s own records show the stance configuration replicates 2-of-3 across seeds.

**SS1. `docs/KNOWN_ISSUES.md:666` — eval-validity [medium / fix medium]**

All configuration-level certification decisions (gate-driven disconnect/raise in the notebook, TOML config locking, publication bundles) run off exactly one training seed, and the repo's own documented 3-seed replication prescription (KNOWN_ISSUES.md Open questions #2) is enforced by no config key, CI check, notebook parameter, or standardized bundle field — the canonical provenance field is a scalar training_seed, and while the open-keyed seed_roles dict can technically carry an extra ad-hoc training role (verified at runtime), nothing produces, interprets, or requires one, so single-seed certification recurs by default and cannot be flagged. (The underlying stance seed sensitivity itself — 2/3 pass rate — is a separately tracked known issue; checkpoint-level advancement of an artifact that itself passed is valid at n=1.)

*Bites when:* The stance stage was certified, advanced into recovery, and its config locked into stance.toml off the seed-42 run alone (2026-08-11); the seed-43 replicate of the byte-identical configuration then FAILED the gate 0-of-200 panels (duty UCB 0.0747 vs ceiling 0.02). Measured base rate 2/3: the next full 'Run all' at a different seed halts the entire multi-hour curriculum via disconnect+raise with probability ~1/3, and conversely every configuration-level claim published from a single passing run (which is all of them — the pipeline offers no other mode) is the winning arm of a documented seed lottery. Nothing in the pipeline can flag or block a single-seed certification, so the violation the KNOWN_ISSUES prescription exists to prevent recurs silently on every future run.

*Suggested fix:* Add an enforced multiplicity field: e.g. a required 'training_seeds' list in result-bundle provenance with a warning/refusal when a stage is marked certified from n=1, and a notebook/CLI replicate mode; at minimum record 'n_training_seeds: 1' loudly in stance_gate_report.json and the website generator.

**SS2. `configs/trex/behavior.toml:106` — eval-validity [medium / fix small]**

Confirmed as stated, with one calibration: the statistical mechanism (raw 30-episode binomial mean vs 0.5, no interval, ~160 unadjusted in-training panels feeding a bare consecutive-3 counter, publication gate that disconnects the runtime on a single n=30 draw) is exactly as described and all quoted numbers reproduce; but its practical bite is conditional on the true bite-success rate landing in roughly the 0.35-0.65 band, whereas the only committed run scored 0.97, and the behavior stage's joint gate may in practice be bound by the co-located 2.0 m/s velocity target rather than the success-rate term.

*Bites when:* Exact binomial at n=30, threshold >=15/30: a genuinely at-threshold policy (true p=0.50) is BLOCKED 43% of the time — the notebook then disconnects the Colab runtime and raises, discarding the ~4-hour stage as failed; a true p=0.60 policy is wrongly blocked 10% of the time. Conversely a true p=0.45 policy passes the single publication check 36% of the time and is published as '>=50% bite success'. In-training, a true p=0.45 policy achieves 3 consecutive passing 30-episode panels somewhere in a full-length run with probability ~0.99 (true p=0.40: ~0.51), so the consecutive-3 rule cannot stop a below-threshold policy from qualifying under the CLI curriculum path (the manager consuming it is out of my scope; the config and panels feeding it are not). The committed 2026-03 legacy result (29/30 = 0.9667 vs then-threshold 0.25) was robust, but the threshold has since doubled to 0.5 where these numbers bite.

*Suggested fix:* Gate on a one-sided binomial LCB (Clopper-Pearson or Wilson) at a declared n, as stance_quality/v1 already does for duty; size n for the 0.5 threshold (n=30 cannot separate 0.45 from 0.60).

**SS3. `environments/shared/curriculum/stance_gate.py:49` — eval-validity [medium / fix small]**

The repo's predeclared held-out confirmation panel (n~100-180, SPLIT_PLAN section 2.3 item 3) for certifying the binary full-horizon >= 0.95 event exists only as prose — no code path, requirement, or artifact record implements it — and the first-ever certified stance PASS labeled full-horizon "certified 1.0000 (40/40)" on the n=40 screening panel (LCB95 0.928), whose seeds (3042-3081) are also the model-selection seeds, so a below-threshold policy (true rate 0.90-0.93, exactly where the seed-43 replicate measured 0.925) passes with 22-46% probability per panel and advances with an indistinguishable certification artifact. Minor refinement: stance_gate_report.py does exist and accepts --episodes, so the confirmation is a one-flag run away — what is missing is the held-out seed block, the requirement before publication_gate_passed, and any artifact recording that the step ran or was skipped.

*Bites when:* A future candidate with true full-horizon rate 0.90-0.93 (below threshold) passes the survival criterion on a resampled 40-episode certification panel with probability 0.22-0.46 per panel; the recorded verdict then reads 'certified >= 0.95' with no artifact distinguishing it from a genuinely compliant policy, and the checkpoint advances into recovery/locomotion. _apply_stage_gate certifies from the 40-episode report alone with nothing prompting or blocking on the missing confirmation step. (For seed 42 the claim happens to be rescued by ~1080 in-training episodes at 27 consecutive perfect panels — binomial LCB95 0.997 — but that evidence is selection-seed and appears in no certification artifact.)

*Suggested fix:* Implement the confirmation panel as stance_gate_report.py --episodes 179 (cutoff 168 certifies P>=0.90 per SPLIT_PLAN §2.3) on a fresh seed block, required before publication_gate_passed for stance_quality/v1 stages; or record the pooled in-training panel evidence in the certification JSON.

**SS4. `environments/shared/scripts/stance_gate_report.py:2272` — eval-validity [medium / fix small]**

Candidate certification, publication, and training-time gate checks all evaluate exclusively on the fixed seed block 3042-3081 (deterministic given the policy), and no decision path ever draws fresh evaluation seeds; a marginal policy (true full-horizon rate ~0.93) that fits this block (~46% of blocks) passes deterministically on every recheck while a fresh block would fail ~54% of the time. The project's own plan (STAGE1_SPLIT_PLAN section 2.3 item 3) predeclares a held-out confirmation panel and admits it is unimplemented. Calibration coupling is partial rather than total: the 0.95 full-horizon floor was pooled over three statue blocks and the reward rail is block-robust, but the duty ceiling's chatterer reference and both certified verdicts used only 3042-3081.

*Bites when:* A policy with true full-horizon rate ~0.93 on the spawn distribution has ~46% chance the fixed 40-spawn block is favorable; if it is, certification and every subsequent recheck of that policy pass identically (the panel is deterministic given the policy), while a fresh seed block would fail ~54% of the time — the pass is unfalsifiable within the pipeline because no decision path ever draws new evaluation seeds. The one committed cross-block measurement proves block-level variance is real: the statue scores 119/120 pooled over three independent 40-seed blocks (stance.toml:341), i.e. some blocks contain killer spawns and others do not.

*Suggested fix:* Certify on a seed block disjoint from calibration (thresholds were tuned on 3042-3081), or run the confirmation panel of finding 3 on freshly drawn seeds and record both blocks in the report JSON.

**SS5. `results/trex/ppo/summary.json:35` — misleading-artifact [low / fix small]**

The committed publication artifacts re-serve single-seed stage_passed=true verdicts earned under the retired statue-passable reward gate, displayed on the website beneath a curriculum description of the current stance_quality gate — a stance 'pass' the current gate was designed to reject.

*Bites when:* A reader of the species page (or a downstream consumer of summary.json) concludes T-Rex stance was certified and 22M-step curriculum completion is an established single-seed capability, when the only stance policy ever to pass the actual stance gate is a different, later checkpoint from a different run — and the displayed 'pass' is one the current gate would score as a statue. Mitigated but not cured by the site's banner ("These are historical experiment records...") since the per-stage pass/fail booleans carry no gate-kind provenance.

*Suggested fix:* Record gate_kind alongside stage_passed in summary schema v2+ and have the website render pre-stance-gate passes as 'passed retired reward gate', or null out stage_passed for stages whose gate has since been replaced.

### 4.4 Resume & multi-session provenance blindness

The provenance/audit layer cannot represent that a resume happened: a multi-session run audits canonical-valid as a clean single-session run, and several checks that should catch cross-session interference do not exist.

**RP1. `environments/shared/result_bundle/evidence.py:488` — eval-validity [high / fix small]**

audit_result_bundle/validate_evaluation_evidence reads stage_config.json's curriculum block (which carries the full stage budget in curriculum.timesteps, beside run.timesteps and the summary's session-local timesteps) but only for gate thresholds, never cross-checking any timesteps value or manifest checkpoint step numbers, so a resumed multi-session run — budget shortfall, truncated/renumbered eval history, stale prior-session checkpoint — audits canonical-valid indistinguishably from a clean single-session full-budget run. (One nuance: whether every cited resume corruption — best_model overwrite, schedule snap-back, ret_rms discard — occurs is a separate question; the audit blindness itself is confirmed by experiment.)

*Bites when:* Confirmed by experiment (scratchpad/provenance-probe/build_resumed_bundle.py): a synthetic trex bundle built with the real save_result_bundle whose stage-2 stage_config.json records run.timesteps=2,511,360 against curriculum.timesteps=8,000,000 in the SAME file, whose evaluations.npz history is truncated and renumbered from ~50k, and whose manifest declares a stale stage2_5000000_steps.zip (5M > the 2.5M the summary claims the stage trained) beside renumbered stage2_500000_steps.zip, audits {"status": "canonical-valid", "errors": [], "warnings": []}. So the known resume corruptions (best_model overwritten by the best of only the remaining 2.5M steps, schedule snap-back, ret_rms discard) are certified for repository promotion indistinguishably from a clean run, and cross-run comparisons treat the resumed stage's best_eval as an 8M-budget result.

*Suggested fix:* In audit_result_bundle/validate_evaluation_evidence, cross-check each stage's summary timesteps against stage_config.json curriculum.timesteps and run.timesteps (warn on shortfall), and flag manifest-declared periodic checkpoints whose step count exceeds the stage's recorded timesteps.

**RP2. `environments/shared/result_bundle/provenance.py:235` — misleading-artifact [medium / fix medium]**

Confirmed as mechanism, overstated in blast radius. On the documented cell-7b resume flow (and any second live session in a matching environment), initialize_result_bundle (environments/shared/result_bundle/provenance.py:209-235) accepts the existing run dir and returns with provenance.json byte-identical — no captured_at update, no session record — while update_provenance's whitelist (constants.py:23-33, enforced provenance.py:282-286) forbids ever adding one, and bundles.py:348 + result_schema.py:834-837 pin summary.date to session 1's captured_at, so a multi-session run is structurally recorded as single-session, with stage training_time_seconds counting only the resumed session (cell 14 per-call timing, schema-forced total at result_schema.py:866-870). However, "every resume-corruption defect becomes invisible at audit" overstates it: the exists-branch rejects resumes under a changed commit/dirty-patch/dependency-set/platform, and resume_same_stage task-fingerprint validation fails closed on task drift — what is invisible is that a resume happened, the honest finishing date, and pre-interruption wall time. That misleads artifacts and removes multi-session provenance but does not invalidate rewards/evaluation or waste budget, so severity is medium rather than high.

*Bites when:* Confirmed by experiment (scratchpad/provenance-probe/second_session.py): re-running the notebook storage cell's initialize_result_bundle against the existing run dir - the documented cell-7b resume flow, and equally an unnoticed second live session - was accepted with provenance.json byte-identical; the only timestamp-like field is captured_at from session 1. A 24h-capped run finished across three sessions publishes provenance/summary dated to day 1 with training_time_seconds counting only each stage's final session, and any downstream reader (or the schema validator itself, which would REJECT a summary honestly dated to the finishing session) is structurally guaranteed the single-session story. Every resume-corruption defect becomes invisible at audit because no artifact is allowed to say a resume happened.

*Suggested fix:* In the exists-branch of initialize_result_bundle, append a sessions/resumed_at entry (new schema-versioned list field) instead of returning silently, add it to the finalization whitelist, and have the audit surface session_count > 1 as a warning.

**RP3. `environments/shared/reporting/bundles.py:284` — eval-validity [medium / fix medium]**

Evaluation evidence (evaluation_*.csv) carries no hash of the checkpoint it evaluated, and nothing locks a run dir against a second live session; save_result_bundle hashes the selected checkpoint only at bundle-save time (hours after stage evidence was written) and the audit only re-verifies that save-time hash, so any write to best_model/robust_best_model landing between evidence generation and bundle save — a zombie session's EvalCallback or a late-flushed Drive write — yields a canonical-valid bundle whose hash-certified checkpoint never produced its certified metrics (demonstrated at runtime). One imprecision in the original scenario: session A mid-stage-2 overwrites 02_recovery/.../best_model.zip (its own stage dir), not 03_locomotion — but the mechanism is identical for whichever stage dir collides.

*Bites when:* Session A hits the ~24h cap mid-stage-2; operator starts session B per cell 7b while A's runtime (or A's buffered Drive writes) is still live. B resumes stage 2, loads best_model, writes 30-episode evaluation_selected.csv, then trains stage 3 for hours; in that window A's EvalCallback overwrites 03_locomotion/models/best_model.zip. B's save_run_bundle then hashes A's file into provenance.selected_checkpoints and the manifest; all hashes agree, evidence aggregates re-derive from B's CSV, and the bundle audits canonical-valid - demonstrated structurally by the experiment, where best_model.zip was arbitrary bytes unrelated to the evidence and the audit passed. The promoted, hash-certified checkpoint never produced its certified metrics. (A write landing in the later hash-to-manifest window instead wedges the bundle - see the low finding.)

*Suggested fix:* Record a sha256 of the evaluated checkpoint into evaluation_*.csv (or a sidecar) at evidence-generation time and have validate_evaluation_evidence compare it to the selected checkpoint hash; add a session lockfile/heartbeat in RUN_DIR that the storage cell refuses to override while fresh.

**RP4. `environments/shared/train_base.py:450` — reproducibility [medium / fix medium]**

Task-lineage/load provenance is validated at --load time but never persisted anywhere readable: the lineage record is attached only as a write-only attribute inside the SB3 checkpoint zip (stored in the zip's data entry; no production reader exists), save_stage_config and the notebook's extra_meta record neither load_path nor task_load_mode, and the result-bundle audit verifies the selected checkpoint by file sha256 only — so a stage warm-started via initialize_next_stage, or resumed from a same-task checkpoint belonging to a different run (fingerprints carry no run/seed identity, verified at runtime), publishes a bundle identical to a from-scratch stage: run A's seed/config provenance can audit clean over run B's weights, and sample-efficiency comparisons drawn from the bundle are silently corrupted.

*Bites when:* Task fingerprints are task-level, not run-level: --load pointing at a same-task stage-2 checkpoint from a DIFFERENT run (a path typo across logs/trex/ppo/<run_id> siblings) passes resume_same_stage validation, and the published bundle then carries run A's configs, seed, and provenance with run B's policy weights - audits canonical-valid, and reproduction from the recorded seed/config fails inexplicably. Likewise a stage warm-started via initialize_next_stage from 11M steps of stance pretraining is indistinguishable in summary.json/stage_config.json from a from-scratch stage, corrupting any sample-efficiency comparison drawn from the bundle.

*Suggested fix:* Persist {load_path, load_mode, parent_task_sha256, parent checkpoint sha256} into stage_config.json's run block (and thence the summary), and have the audit check the selected checkpoint's embedded lineage against it.

**RP5. `environments/shared/result_bundle/provenance.py:234` — resume-safety [medium / fix small]** (Same defect as CO3, recorded here from the provenance side; the two entries complement each other.)

initialize_result_bundle (environments/shared/result_bundle/provenance.py:211-234) refuses a reused run directory on ANY field mismatch, including environment fields (repository_commit, dependency_versions, python_version, platform, repository_dirty/patch); because the documented 7b resume flow re-runs the storage cell in a fresh Colab runtime that clones unpinned main and pip-installs unpinned >= floors, any overnight commit, dependency patch release, or Colab image bump makes the mandatory resume path fail closed with "run directory already belongs to a different run" before section 7b is reached — the only workarounds destroy the provenance anchor or abandon the partial stage. The docstring claims rejection only for species/algorithm/backend/seed, so the code exceeds its own contract; when nothing drifted, the second session is absorbed with no per-session record (tracked as a separate finding).

*Bites when:* Stage 2 interrupted at 5,488,640 of 8M steps on day 1; overnight a commit lands on main (or PyPI ships a new SB3 patch, or Colab bumps its kernel so platform.platform() changes). The day-2 resume session's storage cell calls initialize_result_bundle and raises 'run directory already belongs to a different run' before section 7b is ever reached - the mandatory resume path for every 8-11M-step stage is dead, and the operator either abandons 5.5M steps of budget or hand-edits/deletes provenance.json, destroying the bundle's identity anchor. Conversely when nothing drifted the session is absorbed with zero record (see the provenance finding) - there is no middle path that records the second session and its environment delta.

*Suggested fix:* Split the identity check: hard-fail only on species/algorithm/seed/plant mismatches; for environment/commit drift, append a per-session environment record to provenance and let the audit report it, instead of refusing.

**RP6. `environments/shared/reporting/bundles.py:429` — silent-failure [low / fix small]**

In save_result_bundle (environments/shared/reporting/bundles.py:429-431), the status=complete artifact manifest is written before the bundle's final validate_result_bundle call; if a concurrent (zombie-session) write to a selected checkpoint lands between the line-284 provenance hashing and the line-429 manifest walk, validation raises "selected checkpoint hash does not match provenance" after the complete marker hit disk. Every subsequent save_result_bundle then fails at the line 316-317 re-entry gate (complete manifest => require_complete validation) before the rebuild path, so the run stays canonical-conflict; the only recovery — manually deleting artifact_manifest.json, after which the rebuild succeeds — is undocumented (docs/RESULT_BUNDLES.md offers only "start a new run ID"). Reproduced end-to-end at runtime, including the successful rebuild after manual manifest deletion.

*Bites when:* A zombie session's best_model.zip write lands after line 284 hashed it into provenance but before line 429's manifest walk re-hashes it: line 431 raises 'selected checkpoint hash does not match provenance' AFTER a status=complete manifest hit disk. Every subsequent save_run_bundle call now takes the line-316 path and raises the same error before it can rebuild, so a finished multi-day curriculum is unpromotable; the audit presents it as generic corruption ('artifact size mismatch' / hash mismatch - reproduced in scratchpad/provenance-probe/second_session.py, which showed exactly these errors and no concurrency attribution), and the only remedies are undocumented manual deletion of artifact_manifest.json or abandoning the run_id.

*Suggested fix:* Run the final validate against an in-memory manifest before writing it (write the completion marker only after validation passes), and on re-entry allow a rebuild when the existing complete manifest fails verification, downgrading it to partial with a logged reason.
---

## 5. P5 integrator notes (the gate freeze at `ea0d339`)

Recorded here because the automated survey deliberately excluded these files while they were
being written. What merged, and what the merge left open:

- **What landed.** The calibrated posture-only safe set and fixed height reference are now
  canonical in `recovery_evaluation.py` (`roll_recovery_panel` gained an opt-in
  `height_reference`; `None` is byte-identical to the pre-P5 reset-stamp path, pinned by test).
  The off-distribution harness delegates to the stock roller and reproduces §9.1 exactly
  (0/40 episodes, 18/50 shoves, mean length 360.2). A hand-run producer
  (`harnesses/freeze_recovery_gate.py`) rolls the nulls, derives the real task fingerprint,
  writes `gate_resolution.json`, and verifies it by re-reading through
  `require_gate_resolution`. `reporting/gates.py` judges `recovery_quality/v1` solely through
  the frozen resolution, with distinct refusals for every missing input and a
  declared-vs-frozen agreement check so a TOML edited without re-freezing blocks rather than
  drifts. The in-training manager and the JAX curriculum refuse the kind explicitly, naming the
  three inputs those paths structurally cannot obtain. `configs/trex/recovery.toml` declares the
  frozen thresholds (LCB ≥ 0.30, paired Δ ≥ 0.20 vs the statue null, 100/50 steps, 40 episodes,
  budget back at the measured 3M) with full derivation records. The species catalog gained a
  `recovery_quality/v1` export + render arm (it previously would have rendered an episode-count
  shell that read as a permissive gate) and was regenerated.
- **No frozen record exists yet.** No stage directory holds a real `gate_resolution.json`; the
  committed guard tests assert that the config alone refuses. Producing the record is a hand
  run of `python -m environments.shared.harnesses.freeze_recovery_gate --stage-dir <stage dir>`
  at the default 40 episodes (~2–3 min CPU for the statue null), ideally with
  `--policy-zip/--vecnorm` so the brace null enters the manifest.
- **The checkpoint-loading path of the producer is untested end to end** — no checkpoint exists
  in this container. The pure functions (NumPy forward, normalization) and the fail-closed
  half-configuration check are tested; the first real freeze with a checkpoint should be
  watched.
- **The pipeline cannot reach the new arm yet.** `_apply_stage_gate` forwards neither
  `stage_dir` nor panel evidence — fail-closed and correct today, but the advancing path is
  unreachable until both are plumbed. This was independently confirmed as finding EE2.
- **Residual small items:** `test_result_bundle_semantic_stages.py` still pins `none/v1` in a
  synthetic fixture (harmless — it builds its own config — but it no longer mirrors the
  committed one); the `recovery_quality/v1` paired criterion is judged against the
  `zero_action` null id only (no knob for the brace null — deliberate, no unused knob was
  invented); `min_avg_reward` is deliberately NOT declared for recovery (no rail measured for
  the pushed task; the collapse backstop covers the discarded-return failure mode).
- **Stale-marker follow-through:** the cleanup sweep (§6.3) found the P5-overtaken language that
  remains in files P5 did not touch — `recovery_gate.py` and `gate_schema.py` still say
  "provisional until P5", `manager.py` still cites the pre-P5 design in two comments. SM1, SM2,
  SM3, SM8.

---

## 6. Cleanup opportunities (verified, not applied)

36 confirmed opportunities across four lenses. Every entry was adversarially verified: dead-code
claims by repo-wide word-boundary reference searches (including notebook cell sources, docs,
website, CI and string dispatch) plus runtime imports; duplication claims by diffing the copies;
stale-marker claims by locating the commit that overtook them; structure claims by checking the
seam for hidden state and external users. Two candidates were refuted and dropped (a
recovery-evaluation helper that is genuinely referenced, and a proposed `base_env` accessor
change that would have touched deliberate seams). Nothing here is applied; each entry names the
exact action so it can be picked up as a batch.

### 6.1 Dead code

**DC1. `environments/shared/tests/test_sweep.py:20` [small]** The test_sweep.py compatibility shim star-imports all nine split sweep test modules, so every sweep test is collected and executed twice in any full pytest run, including the CI shared-tests job.

*Action:* As proposed: delete environments/shared/tests/test_sweep.py. Optionally fold its module-to-topic map (the nine-bullet docstring listing what each test_sweep_* module covers) into a comment in one of the split modules or the tests' README if the maintainer wants to keep that index, but nothing functional depends on the file.

*Risk / check:* Someone's muscle-memory `pytest test_sweep.py` stops working (clear no-such-file error, trivially rediscovered). Check: run `pytest environments/shared/tests --collect-only -q` before/after and confirm the total drops by exactly 172 with the per-module counts unchanged.

**DC2. `environments/shared/scripts/sweep/constants.py:47` [small]** _DEFAULT_SEARCH_SPACES and its two component dicts (_DEFAULT_PPO_SEARCH_SPACE L31, _DEFAULT_SAC_SEARCH_SPACE L39) are in-code sweep defaults nothing reads; search spaces now load exclusively from configs/<species>/sweep_{ppo,sac}.json.

*Action:* Delete lines 26-50 of environments/shared/scripts/sweep/constants.py (the "── Default search spaces ──" comment block and _DEFAULT_PPO_SEARCH_SPACE, _DEFAULT_SAC_SEARCH_SPACE, _DEFAULT_SEARCH_SPACES; NET_ARCH_PRESETS stays), and trim the module docstring on line 1 from "Shared constants, exceptions, and default search spaces for the sweep tool." to "Shared constants and exceptions for the sweep tool." In the same change, fix website/docs/training/sweeps.md's "## Default Search Spaces" section (line 207) — it documents the deleted in-code defaults (the SAC table has already drifted from them) and implies a generic-default fallback the loaders deliberately refuse; replace it with a pointer to the per-species configs/<species>/sweep_{ppo,sac}.json files, or delete it. Verify with: rg -w '_DEFAULT' environments/shared/scripts/sweep/ (expect no matches) and cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_sweep_constants.py environments/shared/tests/test_sweep_search_space.py environments/shared/tests/test_sweep_ray_search_space.py -q (59 passed at baseline).

*Risk / check:* Near zero — module-private constants with no readers. Check: `pytest environments/shared/tests/test_sweep_constants.py test_sweep_search_space.py test_sweep_ray_search_space.py -q` still passes and `rg -w _DEFAULT` in the sweep package returns nothing.

**DC3. `environments/shared/jax_training_utils.py:269` [small]** class RolloutProfiler (the entire 'Per-operation profiling' section, L262-309, end of file) has zero references anywhere — no trainer wiring, no config key, no test, no notebook.

*Action:* Delete the 'Per-operation profiling' section at the end of /home/user/mesozoic-labs/environments/shared/jax_training_utils.py — the section-divider comment block and the RolloutProfiler dataclass (lines 263-309, plus the preceding blank lines 261-262 so the file ends cleanly after TrainingCSVLogger.__exit__ at line 260) — and delete the module-docstring bullet at line 9: '- **Profiling**: Per-operation timing within rollout steps'. Then run `cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_jax_training_utils.py environments/shared/tests/test_jax_trainer.py -q` and `/root/.local/bin/ruff check environments/shared/jax_training_utils.py`.

*Risk / check:* Someone may have meant to wire it into the rollout loop someday; if so it lives in git history. Check: `pytest environments/shared/tests/test_jax_training_utils.py test_jax_trainer.py -q` passes after removal.

**DC4. `environments/shared/tests/reward_test_helpers.py:12` [small]** Five of the thirteen shared reward assertion helpers have no caller in any test file: assert_alive_bonus_positive (L12), assert_energy_penalty_structure (L20), assert_approach_reward_zero_on_first_step (L33), assert_zero_weight_zeroes_reward (L79), assert_spin_penalty_non_positive (L100).

*Action:* Delete the five uncalled helpers (assert_alive_bonus_positive L12-17, assert_energy_penalty_structure L20-30, assert_approach_reward_zero_on_first_step L33-39, assert_zero_weight_zeroes_reward L79-89, assert_spin_penalty_non_positive L100-111) from environments/shared/tests/reward_test_helpers.py (~48 lines). Do NOT take the proposal's alternative of calling them from species tests — their invariants are already enforced in environments/shared/tests/test_species_integration.py::TestRewardConsistency (all species, L322-366) and in local spin tests in both species reward files, so wiring them in would duplicate coverage. Verify with: cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/trex/tests/test_trex_rewards.py environments/velociraptor/tests/test_raptor_rewards.py environments/shared/tests/test_species_integration.py -q. Optionally leave docs/CODE_CONSOLIDATION.md untouched (it is a historical plan whose listed signatures already diverge from the implementation).

*Risk / check:* None to production; these encode invariants nobody currently checks, so deleting them changes no test outcome. Check: `pytest environments/velociraptor/tests/test_raptor_rewards.py environments/trex/tests/test_trex_rewards.py -q` passes.

**DC5. `environments/shared/reward_functions.py:702` [small]** check_distance_contact() (L702-710, end of file) — a 'JAX-compatible alternative to contact pairs' — was never adopted by any reward path; the JAX proximity rewards compute distances inline.

*Action:* Delete check_distance_contact (environments/shared/reward_functions.py L702-710, end of file); in environments/shared/tests/test_reward_functions.py remove the check_distance_contact entry from the import block at L11 AND the whole TestDistanceContact class (the class is named TestDistanceContact, not TestCheckDistanceContact; removing only the two assertions would leave a broken import at collection time). Verify with: cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_reward_functions.py -q

*Risk / check:* Trivial — a maintainer adding an MJX bite-contact reward later can re-derive the 3-line norm check. Check: `pytest environments/shared/tests/test_reward_functions.py -q` passes.

**DC6. `environments/shared/jax_reward_termination.py:699` [small]** _check_foot_contact() (L699-707, end of file) is a private helper with zero callers anywhere, including its own module and tests; it also hard-codes a 0.1 N threshold that duplicates the per-sensor logic foot_sensor_groups-based code took over.

*Action:* Delete lines 697-707 of environments/shared/jax_reward_termination.py (the two blank separator lines plus the entire _check_foot_contact definition through end of file), so the file ends with a single newline after the `return jnp.stack(forces)` on line 696. Deleting only 699-707 as proposed would leave two trailing blank lines. Verify with: /home/user/venv/bin/python -c 'import environments.shared.jax_reward_termination' and pytest environments/shared/tests/test_jax_reward_termination.py -q.

*Risk / check:* None — private, uncalled, untested. Check: module imports cleanly (`python -c 'import environments.shared.jax_reward_termination'`) and `pytest environments/shared/tests/test_jax_reward_termination.py -q` if present, else the jax test set.

**DC7. `environments/shared/mjx_utils.py:67` [small]** unscale_action_jax() (L67-72) — the documented inverse of scale_action_jax — has zero references in code, tests, notebooks, docs, or website.

*Action:* Delete unscale_action_jax (environments/shared/mjx_utils.py lines 67-72, plus one of the adjacent blank-line pairs to keep two blank lines between top-level defs). In the report, state the justification as "zero references across code, tests, notebooks, docs, website, configs, CI, and plant-contract fingerprints (which hash only named functions)" and REMOVE the claim that it is an incorrect/stale inverse — it is a correct inverse of the still-live symmetric scale_action_jax; it is simply unused. Verify with: rg -nw unscale_action_jax (expect no hits) and cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_mjx_env.py environments/shared/tests/test_plant_contract_layers.py -q.

*Risk / check:* None; anyone needing a true inverse must write a new one against the piecewise mapping anyway (the stale one would silently give wrong answers). Check: `rg -w unscale_action_jax` returns nothing and the MJX test files still pass.

**DC8. `environments/shared/reporting/stage_layout.py:97` [small]** find_figure() (L97-110) has no caller outside its own unit test, and GENERATED_DIRNAMES (L63) has zero references anywhere; the live readers of the stage layout all use iter_figures/figures_dir/replays_dir instead.

*Action:* In environments/shared/reporting/stage_layout.py delete L62-63 (the `#: Every subdirectory this module owns...` comment plus `GENERATED_DIRNAMES: tuple[str, ...] = (FIGURES_DIRNAME, REPLAYS_DIRNAME)`) and L97-110 (the entire find_figure function). In environments/shared/tests/test_stage_layout.py delete the entire TestFindFigure class (L31-42), which is the sole container of the three find_figure asserts. Then run: cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_stage_layout.py -q (expect 31 passing, down from 34).

*Risk / check:* find_figure encodes the nested-vs-legacy shadowing convention; iter_figures (live, called by stage artifact code) encodes the same convention, so no knowledge is lost. Check: `pytest environments/shared/tests/test_stage_layout.py -q` passes after trimming.

**DC9. `environments/shared/scripts/sweep/ray_tune.py:74` [small]** _sync_to_drive() (L74-83) — the copy-everything Drive sync — has zero callers; its siblings _sync_best_model (skip periodic checkpoints) and _sync_trial_metadata took over both call paths.

*Action:* Delete def _sync_to_drive (environments/shared/scripts/sweep/ray_tune.py lines 74-83) plus one of the adjacent blank-line pairs so exactly two blank lines separate _copy_to_drive from _sync_best_model; then run ruff check and pytest environments/shared/tests/test_sweep_ray_tune.py environments/shared/tests/test_sweep_ray_plant_contract.py -q. In the report, drop or soften the REPO_REVIEW_2026_07_RL_GCP.md:100 citation — that line is about the JAX TrainingCSVLogger, not this function.

*Risk / check:* None — private, uncalled; the copy-everything behavior it implemented is exactly what the review flagged as the pattern to avoid. Check: `pytest environments/shared/tests/test_sweep_ray_tune.py test_sweep_ray_plant_contract.py -q` passes.

### 6.2 Duplicated logic & constants

**DU1. `environments/shared/train_base.py:648` [small]** The stage-entry shaping block (StageWarmupCallback + RewardRampCallback assembly) exists in three copies — the canonical train_base._stage_entry_shaping_callbacks, an inline copy in notebooks/sb3_training.ipynb's train_stage cell, and an inline copy in environments/shared/scripts/sweep/ray_tune.py — despite the canonical helper's own docstring mandating that no inline copies exist, and the copies have already drifted.

*Action:* (1) ray_tune.py:831-854 — replace as proposed: `callbacks.extend(_stage_entry_shaping_callbacks(stage_config, task_load_mode="initialize_next_stage", stage_position=stage, load_path=load_path))`, adding the helper to the existing train_base import block at ray_tune.py:609. Exactly behavior-identical. (2) Notebook cell 7 — replace the two inline blocks with the helper call BUT preserve the notebook's long-standing PPO-only warm-up gate, e.g.: `shaping = _stage_entry_shaping_callbacks(config, task_load_mode=task_load_mode, stage_position=stage_position, load_path=load_path)` then `if ALGORITHM.upper() != "PPO": shaping = [cb for cb in shaping if not isinstance(cb, StageWarmupCallback)]  # notebook has never warmed up SAC; see maintainer note` then `callbacks.extend(shaping)`. StageWarmupCallback is already imported in cell 1, and `_stage_entry_shaping_callbacks` joins the seven private train_base helpers cell 7 already imports; keep the extra_meta recording lines (cell 7 lines 185-201) untouched. (3) File a separate one-line question for the maintainer: should notebook SAC runs get the SAC warm-up (as CLI train(), train_curriculum(), and ray_tune already do), which would let the filter in (2) be deleted? (4) Optionally extend TestShapingIsWired with a notebook-source assertion mirroring tests at test_resume_load_path.py:276/283 so the copy cannot regrow. Validate with `cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_resume_load_path.py environments/shared/tests/test_curriculum_advancement.py -q` plus a notebook-cell smoke of train_stage.

*Risk / check:* Notebook runs gain the warmup_lr_scale passthrough (behavior-identical today since no committed TOML sets it and the callback default is the same 0.1); ray_tune keeps identical behavior for integer stages. Check: run environments/shared/tests/test_curriculum_advancement.py and one short notebook-cell smoke (the repo has test_zero_action_baseline-style notebook-cell executors as precedent).

**DU2. `environments/shared/species_registry.py:31` [small]** The per-species SpeciesConfig values are maintained in two live homes — environments/shared/species_registry.py and each of the four environments/<species>/scripts/train_sb3.py — with production consumers split across the copies and no parity test between them.

*Action:* As proposed, with two refinements: (1) In each of the 4 train_sb3.py scripts, replace the inline SpeciesConfig block (and the now-unneeded SpeciesConfig/env-class imports) with `from environments.shared.species_registry import get_species_config` and `SPECIES_CONFIG = get_species_config("<species>")`, keeping the sys.path fixup and `main(SPECIES_CONFIG)` entry point unchanged. (2) In trial.py, replace the 11-line if/elif chain with `from environments.shared.species_registry import get_species_config` plus `try: SPECIES_CONFIG = get_species_config(args.species)\nexcept ValueError as e: logger.error("%s", e); sys.exit(1)` — preserving the logger.error + exit(1) contract because run_trial is re-exported as public API in sweep/__init__.py (via CLI the branch is unreachable anyway, since __main__.py gates --species with argparse choices). (3) In species_registry.py:25-26, drop only the "Parity with velociraptor/scripts/train_sb3.py:" clause; keep the derivation record, e.g. "# The raptor env only emits strike_success (bite_success was impossible and was removed)." Verify with: cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_species_registry.py environments/shared/tests/test_train_entry_point.py environments/shared/tests/test_sweep_trial.py -q (currently 54 passed).

*Risk / check:* None to behavior — verified byte-equal configs; get_species_config's lazy factory keeps import costs identical. trial.py's exit-code-1 on unknown species becomes a ValueError — wrap in try/except if the exact exit behavior matters. Check: /home/user/venv/bin/python -m pytest environments/shared/tests/test_species_registry.py environments/shared/tests/test_train_entry_point.py environments/shared/tests/test_sweep_trial.py -q.

**DU3. `environments/shared/scripts/zero_action_baseline.py:40` [medium]** Six report scripts each restate the species-to-env-class mapping (SPECIES_ENVS x5 plus stance_quality_baseline's SPECIES list), five restate JAX_ONLY_ENV_KEYS, and four carry a private build_env in two already-drifted variants — while the canonical 3-line form already exists in freeze_recovery_gate.build_env on top of species_registry + load_stage_config.

*Action:* As proposed — add build_env(species, stage) to environments/shared/config.py (lazy in-function species_registry import, mirroring freeze_recovery_gate.py:164-175), have the five scripts import it and delete SPECIES_ENVS/JAX_ONLY_ENV_KEYS/stage_env_section/local build_env, convert stance_quality_baseline's SPECIES list to registry lookups, and make freeze_recovery_gate.build_env a re-export — with two additions: (a) also export a canonical four-name species tuple (e.g. SPECIES_NAMES = ("brachiosaurus", "dibothrosuchus", "trex", "velociraptor")) from the shared home and use it wherever the scripts currently write sorted(SPECIES_ENVS) for argparse choices/defaults, since sorted(species_registry.SPECIES_FACTORIES) would silently add aliases (raptor, t-rex, dibo) to the CLI surface; (b) keep build_env importable at module level in zero_action_baseline specifically (from environments.shared.config import build_env), which preserves both the sb3_training.ipynb cell-12 import and test_zero_action_baseline's monkeypatch. Verify with: pytest environments/shared/tests/test_zero_action_baseline.py environments/shared/tests/test_stance_gate_report.py -q (154 tests pass at HEAD), plus one hand-run per former build_env variant (e.g. zero_action_baseline trex, action_bound_report dibothrosuchus 1 — the latter exercises the previously-filtered nonzero foot_contact kwargs).

*Risk / check:* The consolidated path passes foot_contact_weight/foot_contact_gate to envs where the old scripts filtered them — verified accepted-and-unused on the SB3 path (constructor stores them; no other reads), and it is exactly what production training already does. Keep the module-level build_env attribute in zero_action_baseline for the monkeypatching test. Check: pytest environments/shared/tests/test_zero_action_baseline.py test_stance_gate_report.py -q, then hand-run one report script per variant.

**DU4. `environments/shared/reporting/gates.py:79` [small]** The fallback default for min_eval_episodes (the panel size below which a gate refuses to certify) is stated in nine places with two different values — 40 in the stance/JAX/bundle paths, 10 in reporting/gates.py and the SB3 CurriculumManager's StageThreshold — a live SB3-vs-JAX divergence currently masked only because the trex TOMLs set the key explicitly.

*Action:* Two-part change, split so the behavior-neutral piece is separable. (1) Naming: define DEFAULT_MIN_EVAL_EPISODES_STANCE = 40 in environments/shared/curriculum/stance_gate.py (use it as the StanceGateThresholds field default and at jax_curriculum.py:350, jax_eval.py:788, result_bundle/evidence.py:383, scripts/stance_gate_report.py:709, stage_artifacts.py:203 and :302; recovery_gate.py:180 may reference it or keep its own named default) and DEFAULT_MIN_EVAL_EPISODES = 10 beside StageThreshold in curriculum/manager.py (use it at manager.py:58, reporting/gates.py:79, eval_diagnostics.py:836). Leave freeze_recovery_gate.py's MIN_EVAL_EPISODES alone — it is the frozen registered-panel size, semantically not a fallback default, and tests pin it (test_recovery_gate_config.py:207). (2) Conscious behavior decision, flagged separately in the report: make the SB3 stance path resolve the stance default — manager's stance branch (manager.py:395) and eval_diagnostics.py:595's required_duty_episodes input should use 40 when the TOML omits the key, matching every other stance-path consumer; affects no committed config. Verify with: cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_curriculum_manager.py environments/shared/tests/test_reporting_gates.py environments/shared/tests/test_jax_curriculum.py environments/shared/tests/test_eval_diagnostics.py environments/shared/tests/test_stance_gate.py -q (add test_eval_diagnostics and test_stance_gate to the proposal's check list, since eval_diagnostics is now touched).

*Risk / check:* Pure naming is behavior-neutral; aligning manager's stance path from 10 to 40 changes behavior only for a stance-gated stage whose TOML omits min_eval_episodes — none committed today, and 40 is what every other stance-path consumer already enforces. Check: pytest environments/shared/tests/test_curriculum_manager.py test_reporting_gates.py test_jax_curriculum.py -q.

**DU5. `environments/shared/curriculum/stance_gate.py:145` [small]** The curriculum-dict-to-StanceGateThresholds mapping is copy-pasted at four call sites (plus a field-by-field re-copy in the manager), each restating the same key names and fallback literals (settle_steps 0, min_eval_episodes 40, min_avg_reward -inf, required_consecutive 3) that the dataclass partly declares already.

*Action:* As proposed, with three execution notes: (1) the classmethod should take Mapping[str, Any] and encode strict required-key access via curriculum["key"] so KeyError semantics match today's; (2) keep evidence.py's existing None-check/ResultBundleError before the call and move stance_gate_report's lenient-default rationale docstring onto the require_criteria=False branch so the documented convention survives; (3) implement the manager path as a StageThreshold.stance_thresholds() field-copy accessor (or leave it as-is), never from_curriculum — StageThreshold's min_eval_episodes default is 10, not the dict-path 40, so routing the manager through from_curriculum would change gate behavior.

*Risk / check:* None if each site's current defaults are preserved through the strict/lenient flag; the -math.inf vs -float("inf") difference is cosmetic (equal values). Check: pytest environments/shared/tests/test_stance_gate.py test_result_bundle_evidence.py test_jax_curriculum.py test_stance_gate_report.py -q.

**DU6. `environments/shared/scripts/stance_gate_report.py:86` [medium]** The SB3 checkpoint + VecNormalize loading block (PPO.load, sibling *_vecnorm.pkl guessing, VecNormalize.load with a throwaway DummyVecEnv, training/norm_reward=False, normalize-then-predict) is copy-pasted in three report scripts, with stance_gate_report carrying a fourth, richer version whose docstring says it is 'matching the sibling report scripts' — and the copies have drifted in strictness and plant validation.

*Action:* Extract a shared helper in a new environments/shared/scripts/policy_loading.py with the SIBLINGS' shape, not _load_policy's: load_sb3_checkpoint(model_path, vecnorm_path, env_factory, *, guess_sidecar=True, plant_identity=None, allow_legacy_plant=False) -> (model, normalizer | None, resolved_vecnorm_path | None), implementing PPO.load(device="cpu"), the <stem>_vecnorm.pkl sidecar guess, VecNormalize.load over DummyVecEnv([env_factory]), and training=False/norm_reward=False, with SB3 imported lazily inside the function (the repo's lazy-SB3 convention). Point the three sibling scripts at it with plant_identity=None so their current lenient, unvalidated behavior is unchanged (observation_ablation and action_bound keep using model and normalizer separately, as their rollouts require). For stance_gate_report, either leave _load_policy untouched, or have it call the helper only for the PPO.load + VecNormalize.load steps while keeping its StanceGateReportError wrapping (with the exact existing messages), Dict obs_rms rejection, plant validation placement, and the fused stats-based predict closure local — preserving the 7 direct test call sites and the not-SystemExit contract stage_artifacts depends on. Optionally also deduplicate the identical build_env triplicated in the three siblings into the same module. Do NOT touch the recovery harnesses' raw-pickle/NumPy loading path (deliberate §9 workaround) or evaluation.py's Monitor-wrapped variant. Verify with pytest environments/shared/tests/test_stance_gate_report.py -q (149 tests, currently green) and a hand-run of one sibling report against an existing checkpoint.

*Risk / check:* Preserve each script's current strictness via the flag so no report changes observable behavior; the DummyVecEnv-based load path is identical across all copies. Check: pytest environments/shared/tests/test_stance_gate_report.py -q and hand-run one sibling report against an existing checkpoint under results/.

**DU7. `environments/shared/constants.py:28` [small]** The publication-protocol first seed 3042 is restated as a bare literal in seven scripts, one notebook cell, and one named constant (freeze_recovery_gate.PANEL_SEED_START), with no shared home — even though environments/shared/constants.py exists precisely to centralise such cross-cutting magic numbers.

*Action:* Add PUBLICATION_SEED_START: int = 3042 to environments/shared/constants.py with a one-line derivation note ("publication_evaluation seed role; panel family 3042-3081 — see provenance.evaluation_protocols"). Point the six argparse defaults (zero_action_baseline.py:190, observation_ablation_report.py:201, action_bound_report.py:128, joint_excursion_report.py:75, stance_gate_report.py:2272), the stance_gate_report.py:959 keyword default, and stance_quality_baseline.py:105's range() at it. For freeze_recovery_gate.PANEL_SEED_START, prefer leaving the frozen literal in place with a cross-reference comment to PUBLICATION_SEED_START rather than aliasing — that block is the registered frozen decision procedure and its values are deliberately self-contained; the alias variant proposed is also safe (test_recovery_gate_config.py:210 pins it), so either is acceptable. Leave the notebook literal as-is (documented API naming the role). Verify with: cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_recovery_gate_config.py environments/shared/tests/test_gate_resolution_producer.py environments/shared/tests/test_constants.py -q.

*Risk / check:* None — identical value everywhere; only argparse defaults and one range() change source. test_recovery_gate_config.py:210 asserts producer.PANEL_SEED_START == 3042 and continues to pass. Check: pytest environments/shared/tests/test_recovery_gate_config.py test_gate_resolution_producer.py -q.

**DU8. `environments/shared/tests/reporting_helpers.py:13` [small]** The identical fake velociraptor PlantIdentity fixture (same 16 fields: raptor.xml, revisions 1/1/1, sha256 '1'*64..'4'*64, nq=31/nv=30/nu=22, obs 67/act 22) is copy-pasted eight times across environments/shared/tests/, in three stylistic variants that result_bundle_helpers-style sharing should own.

*Action:* Add make_plant_identity(**changes) -> PlantIdentity (dict-update + PlantIdentity(**values) style) to environments/shared/tests/reporting_helpers.py (it already imports only plant_contract, so either helper module can import it without a cycle); redefine reporting_helpers.plant_identity() as make_plant_identity() to keep existing importers (test_jax_stage_layout, test_reporting_summaries, test_reporting_stage_artifacts) untouched. In result_bundle_helpers.py, rewrite _plant_identity(species) as make_plant_identity(species=species, model_path=model_paths[species]).to_dict(), preserving the trex mapping and the exact dict shape (to_dict adds the "schema" key the dict variants carry). Delete the private copies in test_config.py, test_evaluation.py, test_jax_checkpoint.py, test_train_base.py, and test_sweep_ray_plant_contract.py, importing make_plant_identity as _plant_identity (or renaming call sites); replace test_result_summaries._canonical_plant_identity() with the shared dict builder. Verify with: cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests -q (the six directly-edited files plus the reporting/result-bundle consumers of the shared helpers).

*Risk / check:* Test-only; zero production impact. The only subtlety is dict-vs-dataclass return types per consumer — keep both entry points. Check: pytest environments/shared/tests/test_config.py test_evaluation.py test_jax_checkpoint.py test_train_base.py test_sweep_ray_plant_contract.py test_result_summaries.py -q.

### 6.3 Stale markers & overtaken text

**SM1. `environments/shared/curriculum/recovery_gate.py:24` [small]** The recovery-gate module's own docstrings still declare its thresholds 'PROVISIONAL until P5' and say the recovery config 'declares none/v1 until then' — P5 landed at HEAD (ea0d339) and froze measured thresholds, so the module that computes the gate now contradicts the config that declares it.

*Action:* As proposed, with one adjustment: recovery_gate.py lives in environments/shared/ (species-generic), so do not embed the trex-specific value 0.20 in the RecoveryGateThresholds docstring — that duplicates a per-species config value and creates a fresh staleness liability. Rewrite line 24-28 of the module docstring to past/present tense: thresholds were provisional until the P3/P5 calibration; since 2026-08-28 the trex recovery config declares recovery_quality/v1 at measured values and the W5 resolver's frozen null panels supply the paired criterion (keep the fail-closed sentence verbatim). In the RecoveryGateThresholds docstring (line 169), drop "All PROVISIONAL until P5.", keep "min_paired_success_delta_lcb is optional (None disables)" as-is (it remains true in code and for species without frozen panels), and change "once those exist it becomes the authoritative criterion per the split plan" to "with the resolver's frozen null panels it is the authoritative criterion per the split plan (each species' recovery config declares its value)". Comment-only; no code change.

*Risk / check:* None to behavior — docstrings only, nothing pins them. Only care needed: keep the fail-closed and pairing rationale sentences, which are still accurate.

**SM2. `environments/shared/curriculum/gate_schema.py:89` [small]** The GATE_KINDS entry for recovery_quality/v1 carries the same overtaken claim — 'Thresholds are provisional until the calibration runs (plan P3/P5); configs declare none/v1 until then' — in the very schema that now validates the frozen declaration in configs/trex/recovery.toml.

*Action:* In environments/shared/curriculum/gate_schema.py, replace the two overtaken sentences at lines 89-90 ("Thresholds are provisional until the calibration runs (plan P3/P5); configs declare none/v1 until then.") with the current state, e.g.: "The paired criterion consumes the resolver's frozen null panels and is authoritative when declared. Frozen for trex 2026-08-28 (plan P5) with measured thresholds (configs/trex/recovery.toml); none/v1 remains only for future non-advancing pilots." Keep the surrounding description of the binomial LCB and the pointer to :mod:`environments.shared.curriculum.recovery_gate` unchanged. Optionally drop the trex-specific sentence to just the correction if the maintainer prefers provenance to live only in the config. Run cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_recovery_gate_config.py -q as a sanity check.

*Risk / check:* None — comment inside a frozenset literal; the key set is untouched. Run environments/shared/tests/test_recovery_gate_config.py to confirm nothing readable changed.

**SM3. `environments/shared/curriculum/manager.py:466` [small]** thresholds_from_configs justifies excluding semantic-only stages with a design that no longer exists: 'until the manifest-walking curriculum lands with the recovery gate (plan W4): including one here would validate its none/v1 placeholder' — W4 landed 2026-08-16, recovery no longer declares none/v1, and P5 settled on a different mechanism entirely (post-stage verdict from the frozen resolution; the in-training manager structurally refuses the kind).

*Action:* As proposed: reword manager.py lines 464-469 to the standing rationale — the numeric curriculum walks legacy-numbered (integer-keyed) stages only; the one semantic-only stage (recovery) is judged post-stage by environments/shared/reporting/gates.py::evaluate_stage_gate from the stage's frozen gate_resolution.json, and its recovery_quality/v1 threshold keys are deliberately not carried onto StageThreshold (cross-reference the existing comment at ~line 484 and _recovery_gate_refuses). Drop the "manifest-walking curriculum", "plan W4", and "none/v1 placeholder" clauses. Leave CHANGELOG.md:546 untouched (dated historical record). If an in-training manifest-walking curriculum is still desired for future stages, record it in a plan/ROADMAP document, not this comment.

*Risk / check:* None — comment only. The isinstance(stage, int) filter is untouched; test_curriculum_manager.py covers the behavior.

**SM4. `environments/shared/train_base.py:429` [small]** Both dated transition valves in the task-fingerprint machinery cite 'plan §W5' as their planned tightening point — W5 landed 2026-08-16 without the tightening, and both valves must in fact STAY for now, so the stale pointer misstates the real expiry condition.

*Action:* In environments/shared/train_base.py:429-430 and environments/shared/task_fingerprint.py:288-289, replace the "(planned with the gate resolver, plan §W5)" / "(alongside the allow_unfingerprinted valve, plan §W5)" pointers, which cite a plan item that landed 2026-08-16 without the tightening. State the true expiry condition without inventing a schedule: for allow_unfingerprinted — "tighten to fail-closed once no live lineage resumes a pre-2026-08-15 checkpoint (today the certified stance parent 20260810_145546 still does)"; for the v1 schema valve — "tighten to exact-hash-only once no live checkpoint predates the 2026-08-23 v2 bump (today the recovery pilots 20260819_154702 / 20260821_142144 still do)"; keep the two comments cross-referencing each other as they do now. Do NOT tighten either valve. Separately, add the tightening as a tracked follow-up item in docs/STAGE1B_IMPLEMENTATION_PLAN.md (or the successor plan), since no document currently tracks it — that gap is why the pointer went stale.

*Risk / check:* None for the comment edit. Actually tightening the valves today would be a behavior change that breaks loading the certified stance checkpoint and both recovery pilot checkpoints — that is exactly why this finding is text-only.

**SM5. `docs/KNOWN_ISSUES.md:132` [medium]** KNOWN_ISSUES — a living tracker with an explicit delete-when-fixed policy, unlike the append-only investigation records — still asserts 'stage 1's real gate machinery does not exist yet', six weeks after stance_quality/v1 landed and three weeks after it certified the first gate PASS; at least three sibling entries are similarly overtaken.

*Action:* As proposed, with one addition: run the prune pass per the file's own policy (precedent: commit 2876952) — delete the line-132 gate-machinery entry and the line-122 statue-gating entry (their ask shipped as stance_quality/v1 and certified the 2026-08-11 GATE: PASS); rewrite line 98 to its residual (stance 1a is undisturbed by design now that recovery.toml is the pushed stage; point at the stage-1b/recovery records); shrink line 166 to its live residual (velociraptor/brachiosaurus/dibothrosuchus stage-1 collapse_peak_floor values are still absolute, per their "Absolute pending §14 item 3" comments; trex is done via collapse_peak_floor_fraction). ADDITION: explicitly leave untouched the two entries referenced from code — the diag_r_foot/diag_l_foot interleaving defect (~KNOWN_ISSUES lines 188–223, referenced by environments/shared/jax_eval.py:700 and the fail-closed error message in environments/shared/jax_curriculum.py:327, which depends on that entry existing for the quadruped case) and the height-jitter state-inert entry (line 248, referenced by environments/trex/tests/test_trex_env.py:418). Verify each entry against the tree at execution time, as the 2026-08-15 prune commit did.

*Risk / check:* Over-pruning could delete a still-live residual (the non-trex absolute floors, the quadruped duty limitation). Mitigate by keeping the residuals as rewritten entries; the file is not consumed by any code (grep KNOWN_ISSUES environments/ → prose references only, e.g. mjlab/jax_eval pointing INTO it).

**SM6. `website/docs/training/hyperparameters.md:13` [small]** Live website documentation misstates the config tree for three of four species and the recovery gate's status: the hyperparameters guide shows velociraptor/brachiosaurus with renamed files they never got, omits trex's recovery.toml and dibothrosuchus entirely, and trex.mdx still says recovery's gate is enabled 'once its thresholds are calibrated' — they were, at HEAD.

*Action:* In website/docs/training/hyperparameters.md: change line 11 to state per-species counts (trex has four stage configs plus a stages.toml manifest; the other species have three); fix the tree to the real filenames — velociraptor: stage1_balance.toml, stage2_locomotion.toml, stage3_strike.toml; brachiosaurus: stage1_balance.toml, stage2_locomotion.toml, stage3_food_reach.toml; trex: stance.toml, recovery.toml, locomotion.toml, behavior.toml, stages.toml — and add a dibothrosuchus block (stage1_balance.toml, stage2_locomotion.toml, stage3_snap.toml). In website/docs/models/trex.mdx: at line 29 replace "is enabled once its thresholds are calibrated" with the frozen state (recovery_quality/v1 frozen 2026-08-28 with measured thresholds; the verdict comes from the stage directory's gate_resolution.json written by freeze_recovery_gate.py); at line 36 do NOT claim recovery results are in the catalog — reword the still-forward-looking sentence to say recovery results join the tables and videos as runs certified under the now-frozen recovery_quality/v1 gate are published (the recovery stage currently has no catalog video or historical results). Docs-only; optionally run the website build if CI does.

*Risk / check:* Docs-only; no build coupling found (the tree is a fenced code block). Check the website build renders (npm build under website/) if the repo's CI does so.

**SM7. `environments/trex/envs/trex_env.py:271` [small]** Two load-bearing recalibration pointers still name configs/trex/stage1_balance.toml, a file renamed to stance.toml on 2026-08-20 — a maintainer following the trex.xml keyframe's 'must be re-measured together with any edit here' checklist is sent to a file that does not exist.

*Action:* Execute as proposed, with one precision fix: (a) In trex_env.py:271-272 and trex.xml:555, repoint "configs/trex/stage1_balance.toml" to "configs/trex/stance.toml". (b) In the trex.xml checklist, note that recovery.toml mirrors stance.toml's [env] block only — nosedive_termination_threshold propagates via the mirror pinned by test_stage_manifest.py::TestRecoveryStageConfig::test_env_mirrors_stance_plus_exactly_the_perturbation_block, while min_avg_reward is deliberately NOT declared in recovery.toml (recovery.toml:234) — so a re-measurement edits stance.toml and then re-mirrors recovery.toml's [env]. (c) Because the trex.xml byte change moves source_closure_sha256, regenerate the catalog (python -m environments.shared.species_catalog, verified in CI by --check) and re-run environments/shared/tests/test_species_catalog.py plus the plant_contract test suite; expect the only catalog diff to be trex's source_closure_sha256. No checkpoint invalidates: physics_sha256 is derived from the compiled MjModel, not XML bytes, and compatibility_errors excludes source revisions.

*Risk / check:* trex_env.py edit: none. trex.xml edit: ANY byte change moves the plant's source_closure_sha256 (source_layer.py: "any byte change in the model or its assets changes the source digest") — that digest is recorded, embedded in website/src/data/species.generated.json, but deliberately NOT checked by PlantIdentity.compatibility_errors ("Source and visual revisions are intentionally excluded") and is not pinned in configs/plant_versions.toml, so no checkpoint invalidates. Regenerate the catalog and run the plant_contract test suite to confirm.

**SM8. `environments/shared/curriculum/manager.py:258` [small]** should_advance's else-branch comment still names recovery_quality/v1 as its example of 'a kind with no evaluator here' — but the P5 commit added an explicit RECOVERY_GATE_KIND branch two lines above, so the named kind can no longer reach the branch the comment annotates.

*Action:* In /home/user/mesozoic-labs/environments/shared/curriculum/manager.py, rewrite only the else-branch comment at lines 258-265 of should_advance to describe what the branch now guards: a future gate kind added to gate_schema.GATE_KINDS (per gate_schema.py's documented extension path) without a corresponding evaluator branch in this manager. Keep the load-bearing history that this else used to be the reward_and_length fall-through which advanced stages on StageThreshold's permissive defaults (min_avg_reward = -inf), the fall-through reporting/gates.py refuses. Drop the recovery_quality/v1 example from the else comment entirely; do NOT add a pointer comment on the RECOVERY_GATE_KIND elif — should_advance's Returns docstring (lines 228-231) and _recovery_gate_refuses's docstring already cover it. Leave the logger.error text and all branch logic untouched. Do not touch jax_curriculum.py:246 — its analogous comment is accurate there.

*Risk / check:* None — comment only; the branch logic is untouched and covered by test_curriculum_manager.py.

**SM9. `configs/quality_scoring.toml:52` [small]** The repo's only actionable TODO markers (2 of them, both here) rest on a premise the tree has since falsified: they say gait_symmetry is 0.0 'in all training configs' because 'the formula rewards asymmetry' — but the formula now rewards touchdown alternation and two species train stage 2 with nonzero weights, so the marker hides a live scoring decision behind dead facts.

*Action:* Rewrite both TODO blocks in configs/quality_scoring.toml (lines 52-56 and 94-95) to the current facts: the formula was fixed in 70a42f7 to reward touchdown alternation; brachiosaurus (2.2) and dibothrosuchus (2.0) train stage 2 on it while trex/velociraptor keep 0.0; note that the candidate scoring key for the quadruped species that train on it is quad_gait_symmetry (they emit gait_symmetry only for the untrained front pair). Record as tracked follow-up — NOT done in this cleanup — whether stage_2 sweep scoring should add quad_gait_symmetry/gait_symmetry per species, since any weight addition reorders sweep rankings and needs a scoring re-run. In the same comment-only pass, also update the three stale trex config comments repeating the falsified claim (configs/trex/locomotion.toml:25 "formula is inverted", configs/trex/behavior.toml:13 "formula rewards asymmetry", configs/trex/stance.toml:174 "default formula rewards single-foot stance") to state the actual current rationale (kept at 0.0, formula is fixed but untested/unswept for trex). Do not touch velociraptor/stage2_locomotion.toml:22 — its disable rationale is sweep design, not a broken-formula claim.

*Risk / check:* Comment-only edit is riskless; the trap is 'completing' the TODO by adding a [stage_2.gait_symmetry] weight, which silently reorders sweep results — keep that as the tracked follow-up, applied with a scoring re-run.

**SM10. `environments/shared/diagnostics.py:193` [small]** DiagnosticsCallback's deprecated plateau_window/plateau_threshold parameters (deprecated 2026-07-24, ignored ever since) have zero callers anywhere in the repo — environments, notebooks, website, scripts, configs — so the deprecation window has served its purpose and the shim can be retired.

*Action:* As proposed: delete plateau_window/plateau_threshold from DiagnosticsCallback.__init__ (diagnostics.py:176-177) and the warning block (lines 193-197); replace test_legacy_plateau_params_are_accepted_but_ignored (test_diagnostics.py:56-62) with a pytest.raises(TypeError) assertion or delete it; fix the sb3_training.ipynb cell-14 print to drop "plateau detection" (or attribute it to the stage-gate eval callbacks). Keep the module-level logger (still used by logger.debug at diagnostics.py:340) and do not touch eval_diagnostics.py's live plateau_window or the diagnostics_plateau_window curriculum key. Verify with: cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_diagnostics.py environments/shared/tests/test_train_base.py -q.

*Risk / check:* An out-of-tree caller still passing the kwargs would get TypeError instead of a warning — that is the intended end state of a deprecation window, and the repo's own surfaces are clean. Run test_diagnostics.py and test_train_base.py after.

### 6.4 Structural simplification

**ST1. `.github/workflows/python-ci.yml:129` [small]** Five byte-identical CI test jobs (test-shared, test-velociraptor, test-brachiosaurus, test-trex, test-dibothrosuchus) are copy-pasted 37-line blocks that differ only in the pytest path and the coverage-file suffix; a single matrix job replaces ~184 lines with ~40.

*Action:* Replace the five jobs test-shared/test-velociraptor/test-brachiosaurus/test-trex/test-dibothrosuchus (python-ci.yml lines 129-312) with one job `test` (name is unclaimed): needs: lint; strategy: {fail-fast: false, matrix: {python-version: ["3.11", "3.12", "3.13"], suite: [shared, velociraptor, brachiosaurus, trex, dibothrosuchus]}}; identical five steps with run: `pytest environments/${{ matrix.suite }}/tests/ -v --tb=short --cov=environments --cov-report= --cov-fail-under=0` (drop the proposal's redundant `matrix.suite == 'shared' && ...` ternary — all suite names map directly to directories), env COVERAGE_FILE: `.coverage.${{ matrix.suite }}-${{ matrix.python-version }}`, MUJOCO_GL: osmesa, and artifact name `coverage-${{ matrix.suite }}-${{ matrix.python-version }}` — byte-identical to current artifact names so the coverage combine job is untouched except shrinking its needs list to [plant-contract, test, test-sb3, test-jax-cpu]. Keep python-version values as quoted strings (matching the current file) and keep fail-fast: false. In the same PR, update any branch-protection required-check names from `test-<suite> (<py>)` to the new `test (<py>, <suite>)` form (unverifiable from the repo; check repo settings), and verify on a trial run that the coverage job's download step lists the same 17 artifacts as before.

*Risk / check:* GitHub check names change from 'test-shared (3.12)' to 'test (shared, 3.12)'; if branch protection lists the old names as required checks, that setting must be updated in the same PR. Coverage combine is unaffected because artifact names are preserved — verify by diffing the 'Download coverage data' result on a trial run.

**ST2.** Duplicate of DC1 (the `test_sweep.py` star-import shim double-collecting 172 tests); counted once.

**ST3. `environments/shared/scripts/sweep/trial.py:171` [small]** trial.py hand-rolls a four-branch if/elif that imports SPECIES_CONFIG from each per-species train_sb3.py script, duplicating species_registry.get_species_config — and the four scripts each restate a SpeciesConfig literal that is field-for-field identical to the registry's.

*Action:* In environments/shared/scripts/sweep/trial.py replace lines 170-181 with: from environments.shared.species_registry import get_species_config; then `try: SPECIES_CONFIG = get_species_config(args.species)` / `except ValueError: logger.error("Unknown species: %s", args.species); sys.exit(1)` (preserving the fail-closed exit even though argparse choices in __main__.py already gate it). Shrink each environments/<species>/scripts/train_sb3.py to keep its docstring, the sys.path.insert stanza (required for direct `python train_sb3.py` invocation documented on the website), `from environments.shared.species_registry import get_species_config`, `from environments.shared.train_base import main`, `SPECIES_CONFIG = get_species_config("<species>")`, and the `if __name__ == "__main__": main(SPECIES_CONFIG)` block; delete the now-unused SpeciesConfig and env-class imports along with the four duplicated literals. Do not add the originally proposed registry-vs-script parity assert (it becomes tautological); optionally add a smoke test that each script module's SPECIES_CONFIG.species matches its directory. Validate with: cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_sweep_trial.py environments/shared/tests/test_species_registry.py environments/shared/tests/test_train_entry_point.py -q, plus one short trial dry-run (python -m environments.shared.scripts.sweep trial --species trex --timesteps 1000 ...) since Vertex HPT workers exercise this path.

*Risk / check:* The registry factory and script literal could drift in a way I missed — mitigated because I compared all four pairs verbatim and they match today; a parity assert in test_species_registry.py would lock it. Vertex trial workers exercise this path, so run test_sweep_trial.py and one trial dry-run.

**ST4. `environments/shared/train_base.py:610` [small]** _select_handoff_checkpoint is a 15-line pure-Path function stranded in train_base, forcing reporting/stage_artifacts.py into four separate function-level private cross-package imports; it belongs beside RobustBestModelCallback in curriculum/checkpoints.py.

*Action:* Move _select_handoff_checkpoint (train_base.py:610-624) to environments/shared/curriculum/checkpoints.py as public select_handoff_checkpoint, docstring intact. In train_base, replace the definition with a module-level re-export: from .curriculum.checkpoints import select_handoff_checkpoint as _select_handoff_checkpoint  # noqa: F401 — this keeps the internal call sites (1111, 1472), environments/shared/tests/test_train_base.py:31, and notebooks/sb3_training.ipynb cell 14 working unchanged per the module's stated re-export convention. In reporting/stage_artifacts.py, add one module-level import (from environments.shared.curriculum.checkpoints import select_handoff_checkpoint) and delete the lazy imports at lines 122, 228, 308; at line 949, keep the lazy import but reduce it to from environments.shared.train_base import _ensure_sb3 (train_base must never be imported at reporting module level — it pulls SB3-dependent submodules). Update the four call sites to the public name; leave the prose comments that mention the old name or update them in passing. Optionally add select_handoff_checkpoint to curriculum/__init__'s re-exports beside the other checkpoints names. Verify with: (a) pytest environments/shared/tests/test_train_base.py -k handoff; (b) an SB3-blocked import check (meta-path hook or bare venv) of environments.shared and environments.shared.reporting.stage_artifacts — do not rely on the CI wheel smoke alone, since its venv uses --system-site-packages and may see SB3.

*Risk / check:* Import-order: stage_artifacts is imported during 'import environments.shared' (via reporting/__init__), so the new module-level import must not pull SB3 — curriculum/checkpoints imports only sb3_compat (the optional-SB3 shim), verified by reading its import block; confirm with the CI wheel job's SB3-free 'import environments' smoke.

**ST5. `environments/shared/recovery_evaluation.py:223` [small]** roll_recovery_panel smuggles the height reference to _safe_step by stamping a private attribute onto the env object (env._recovery_height_reference) even though _safe_step's only caller is roll_recovery_panel itself — an explicit parameter removes the side channel.

*Action:* In /home/user/mesozoic-labs/environments/shared/recovery_evaluation.py: change _safe_step to _safe_step(env, safe_set, height_target: float); delete lines 172-174 (getattr + RuntimeError) and use the height_target parameter at line 175; in roll_recovery_panel replace the line-223 stamp with a local 'height_target = float(env.data.qpos[2] if height_reference is None else height_reference)' and pass it at the line-237 call site; reword _safe_step's docstring first paragraph (lines 159-165) from "whichever reference roll_recovery_panel stamped on the env" to "the reference roll_recovery_panel resolved after reset and passes in", keeping the §4.1/§4.2 derivation-record content verbatim. In /home/user/mesozoic-labs/environments/shared/tests/test_gate_resolution_producer.py: delete the attribute assert at line 109 (the behavioral asserts at 110-111 carry that test); delete or restate behaviorally the tests at lines 81-87 (test_none_stamps_the_per_episode_reset_height) and 113-116 (test_the_calibrated_reference_is_the_measured_settled_height), whose properties are already behaviorally pinned by lines 89-100 and 102-111; replace the line-153 stamp by passing float(env.data.qpos[2]) as the third argument to the _safe_step calls at 155-156. Then rerun environments/shared/tests/test_gate_resolution_producer.py (baseline: 21 passed) and environments/shared/tests/test_recovery_gate_wiring.py to confirm the frozen-gate evidence is unchanged.

*Risk / check:* test_gate_resolution_producer asserts the stamped value directly (lines 87/109/116), so the test edit must keep asserting the resolved reference (now via the local/parameter); rerun test_gate_resolution_producer.py and test_recovery_gate_wiring.py to confirm the frozen-gate evidence is unchanged.

**ST6. `environments/shared/jax_trainer.py:422` [medium]** jax_trainer.py contains two parallel PPO training-loop implementations — the notebook-path module-level train()+_build_jit_fns (~850 lines, consumed only by notebooks/jax_training.ipynb and two tests) and the 'Library API' JaxTrainer class (~510 lines, the CLI path) — with an explicit banner marking the seam at line 900.

*Action:* Extract TrainConfig, TrainResult, _build_jit_fns, and train (four names, NOT _finite_mean — its sole caller is JaxTrainer at jax_trainer.py:1418, so it stays put) from environments/shared/jax_trainer.py into a new environments/shared/jax_train_fn.py, re-exporting all four from jax_trainer.py per the jax_trainer_types.py precedent so the notebook (cell 23) and test imports (tests/test_jax_trainer.py:401,442) keep working unchanged. The new module's top-level imports are only dataclasses/pathlib/typing/numpy plus its own logging.getLogger; the heavy jax/* imports are already function-local and move verbatim. Also update the two living cross-references into the moved code: docs/KNOWN_ISSUES.md:284 and environments/shared/scripts/action_bound_report.py:8, both citing 'jax_trainer.py:365' for the action clip (actually now line 378, inside _build_jit_fns) — repoint them to the new file:line; leave the dated review snapshots (docs/reviews/, docs/investigations/) untouched. Touch up the 'Two APIs' module docstring (jax_trainer.py:1-23) to name the new module. Verify with: cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_jax_trainer.py -q, plus re-running the notebook's import cell. Consolidating the two loops (migrating the notebook onto JaxTrainer) remains explicitly out of scope.

*Risk / check:* Pure move-plus-re-export; risk is only a missed intra-file reference — mechanical to check since the two halves share no helpers except imports (verified: JaxTrainer methods never call _build_jit_fns/_finite_mean... verify _finite_mean specifically before moving; it sits in the function half). Check: pytest environments/shared/tests/test_jax_trainer.py and re-run the notebook's import cell.

**ST7. `environments/shared/reporting/stage_artifacts.py:243` [large]** Library code (reporting/stage_artifacts.py) imports ten report-building functions from a script module (environments/shared/scripts/stance_gate_report.py, 2504 lines) at eight call sites — inverting the package boundary and, because pyproject omits */scripts/* from coverage, hiding training-path code from the 70% coverage gate.

*Action:* As proposed — move the ten functions plus their call-graph closure (in practice: everything except main() and the argparse block, including write_stance_panel_evidence, STANCE_PANEL_FIELDNAMES, ConstantHold, RootImpulse, and the private helpers) into environments/shared/reporting/stance_report.py; scripts/stance_gate_report.py keeps main()/CLI and re-imports the moved names so the CLI, docs references, and test_result_bundle_evidence.py keep working — with four additions: (1) do NOT re-export stance_report from reporting/__init__.py: environments/shared/__init__.py eagerly imports .reporting, and stance_report's module-level SPECIES_FACTORIES import pulls species_registry -> train_base -> .cli, so an eager re-export risks a partial-initialization cycle and makes importing environments.shared heavier; (2) keep the eight import sites in stage_artifacts.py lazy, only retargeted to reporting.stance_report; (3) retarget the ~10 monkeypatch.setattr(stance_gate_report, "build_stance_gate_report", ...) sites in test_stance_gate_report.py (lines 148, 304, 1116, 1182, 1209, 1232, 1626, 2591, 2610, 2629) to the new module — via the scripts re-export they would otherwise silently stop intercepting stage_artifacts' calls and let tests attempt real rollouts; (4) drop the sys.path bootstrap from the moved code (the script shim keeps it). Verify with test_stance_gate_report.py, test_reporting_stage_artifacts.py, and test_result_bundle_evidence.py; expect the gated coverage percentage to rise (module measures 84% under its own tests).

*Risk / check:* The 2504-line script's helpers are interconnected — the move must be closed under the call graph of the ten functions, so do it mechanically (move, run test_stance_gate_report.py and test_reporting_stage_artifacts.py, chase ImportErrors). Behavior of reports is unchanged since only module location moves. The coverage percentage will shift (more measured lines) — the 70% gate should be rechecked.

**ST8. `environments/shared/jax_training_utils.py:268` [small]** Five symbols have zero references anywhere in the repository: RolloutProfiler (41-line class), jax_reward_termination._check_foot_contact, mjx_utils.unscale_action_jax, sweep/ray_tune._sync_to_drive, and compare_run_diagnostics._fmt — ~70 dead lines total.

*Action:* Delete the five definitions as proposed, with one span precision fix: in environments/shared/jax_training_utils.py remove the 'Per-operation profiling' banner comment block (lines 262-264), the @dataclass decorator (line 268), and the RolloutProfiler class body (lines 269-309); also remove jax_reward_termination.py:699-707 (_check_foot_contact), mjx_utils.py:67-72 (unscale_action_jax — note its docstring falsely claims to invert the piecewise scale_action_jax, so do not keep it as an API pair), ray_tune.py:74-83 (_sync_to_drive), and compare_run_diagnostics.py:144-145 (_fmt). No import statements need changing (verified: dataclass/field/np still used in jax_training_utils.py; _copy_to_drive still used by the surviving sync helpers). Afterwards run ruff on the five files and pytest environments/shared/tests/test_jax_reward_termination.py test_jax_training_utils.py test_mjx_env.py test_sweep_ray_tune.py.

*Risk / check:* unscale_action_jax is a public-looking inverse of a used function — if a maintainer wants to keep the pair as documented API it should instead gain a test/doc reference; today nothing in docs/ or notebooks/ names it, so by the repo's own standard it is dead. Check: run ruff and the jax test files (test_jax_reward_termination.py, test_mjx_env.py) after deletion.

**ST9. `environments/shared/train_base.py:1639` [small]** In train_base's backward-compat tail, the DiagnosticsCallback re-export (line 1639) and the 'evaluate' and 'record_stage_video' names on line 1640 have zero importers anywhere — only eval_policy on that line is actually used (internally, at line 1168).

*Action:* In environments/shared/train_base.py: (1) delete line 1639 ("from .diagnostics import DiagnosticsCallback  # noqa: E402, F401"); (2) replace line 1640 with "from .evaluation import eval_policy  # noqa: E402" (drop evaluate and record_stage_video; drop F401 since eval_policy is used at line 1168); (3) keep line 1638 (.cli re-exports — live via train.py, all four species train_sb3.py scripts, sweep/trial.py, and tests) and line 45 (tb_sync — used internally at 277/278/721 and imported from train_base by test_train_base.py) untouched; (4) update the module docstring: remove "DiagnosticsCallback" from the line-11 bullet and "evaluate, record_stage_video" from the lines-14-15 bullet, AND reword the lines-20-23 sentence from "All public names are re-exported here" to state that only the still-exported names (main/_apply_overrides/_cast_value, eval_policy, tb_sync helpers) are kept for backward compat; (5) optionally drop the redundant "# noqa: F401  (re-exported for backward compat)" on line 45 or reword it to note the names are used internally and imported by tests — but do not remove the import itself. Verify with: ruff check environments/shared/train_base.py; cd /home/user/mesozoic-labs && /home/user/venv/bin/python -m pytest environments/shared/tests/test_train_base.py environments/shared/tests/test_train_entry_point.py environments/shared/tests/test_cli.py -q (95 passed at baseline).

*Risk / check:* Near zero: the search covered every tree including notebook JSON; the only conceivable breakage is an out-of-repo consumer of 'from environments.shared.train_base import evaluate', which the repo cannot see. Check: pytest environments/shared/tests/test_train_base.py test_train_entry_point.py test_cli.py.
---

## 7. Suggested execution order

Phased so that each batch is independently shippable and testable; nothing here is started.

- **Phase R — resume safety** (before any resume; all small/medium): TC1, TC2, TC3, EE3, TC5,
  TC4, CO1, CO2, CO4, RP5. Regression tests: a two-session resume harness that asserts
  continuous checkpoint numbering, preserved best trackers, appended eval history, continued
  schedules, cleared warm-up marker. CI3 belongs here too (the existing resume tests must
  actually run in CI).
- **Phase G — recovery gate reachability** (before the 3M recovery re-run): EE1, EE2, the hand
  freeze of `gate_resolution.json` (§5), NB6, NB7, and TC10/CI6 (stage-scoped `--override` for
  semantic stage ids, so a 1b push-magnitude sweep does not have to edit the TOML).
- **Phase V — evaluation validity**: ER1, ER3, ER4, OP2, OP7, ER2 (recovery plateau logging
  crash — latent), CF2, CF3, SS2, SS3/SS4 (held-out confirmation panel), SS1 (seed-multiplicity
  field), SS5 (retired-gate labeling in published summary), RP1 (audit cross-checks), RP2
  (session records), RP4 (persist load lineage into stage_config).
- **Phase C — CI hardening**: CI3, CI4, CI7, CI9, CI8 (deploy gating), ST1 (matrix
  consolidation), DC1 (double-collected sweep tests).
- **Phase J — JAX/MJX parity** (before any MJX pilot): JX1, JX5, JX2, JX4, JX6, JX8, JX9, JX7,
  JX3 (recovery reachability inventory), EP3 (MJX keyframe id).
- **Phase K — cleanup batches**: quick wins first (all DC entries, SM entries, DU4, DU7, ST5,
  ST9 — each < 1 h), then the consolidations (DU1, DU2, DU3, DU5, DU6, DU8, ST3, ST4, ST6),
  and the one large move last (ST7, `stance_gate_report` → `reporting/stance_report.py`).

Two standing observations that are not fixes but should shape planning: the sweep search
spaces are stale against the current stance design (CF6) and the sweep gate logic predates the
gate-kind system (OP2) — any new sweep should wait for Phase V; and the behavior stage's gate
design (CF2, CF3, SS2) needs a decision, not just a patch, since its 2.0 m/s target now lives
in the wrong metric for the episodes the stage actually produces.

## Appendix A — refuted candidates (for the record)

Four area-level candidates were killed by adversarial verification; they are recorded so the
same false leads are not re-chased:

- `recovery_evaluation.paired_success_differences` "duplicates the resolver's pairing logic" —
  refuted: it pairs two live panels, a deliberately different job than pairing against a frozen
  manifest.
- `base_env` "is_success is always False outside behavior" — refuted: success semantics are
  stage-scoped by design and the gate paths never consume `is_success` outside stage 3.
- MJX "forward-velocity uses current heading vs SB3's episode-fixed heading" — refuted: both
  backends use the same reference; the finder misread a helper.
- The stance-panel fail-closed path "is untested" — refuted: a dedicated test exists and runs.

Two cleanup candidates were likewise refuted (a claimed-dead helper with a live caller; a
proposed collapse of the `base_env` underscore seams that external modules legitimately read —
the verifier judged the existing seam deliberate).

---

*Method note: this review was produced by three orchestrated multi-agent workflows (101 + 42
agents) with adversarial verification of every finding, plus a by-hand integrator review of the
P5 merge. The full per-finding verification transcripts (including the runtime probes and their
scripts) are preserved in the session working directory; the findings above stand on the
verifier-corrected wording.*
