# Known Issues & Open Recommendations

The single living list of verified-but-unfixed findings and standing
recommendations, consolidated from the dated code reviews in
[`docs/reviews/`](reviews/). When an item here gets fixed, delete it (the
full context stays in the archived review it came from). When a new review
lands, fold its open items in here and archive the review document.

**Review history:**

| Review | Scope | Outcome |
|---|---|---|
| [reviews/CODE_REVIEW.md](reviews/CODE_REVIEW.md) (2026-03) | Duplication + code quality | Consolidation done in v0.3.0; bugs fixed except thread-unsafe CSV writes (below) |
| [reviews/REPO_REVIEW_2026_06.md](reviews/REPO_REVIEW_2026_06.md) | Full repo: SB3 + JAX RL correctness, sweeps, configs, docs | ~25 verified bugs fixed in PRs #423–#425 |
| [reviews/REPO_REVIEW_2026_07_RL_GCP.md](reviews/REPO_REVIEW_2026_07_RL_GCP.md) | GCP/Vertex integration, SB3/JAX/sweep delta pass, notebooks | ~30 verified bugs fixed in PR #426 (incl. the JAX eval/CLI follow-up pass) |
| [reviews/VELOCIRAPTOR_PLANT_REVIEW.md](reviews/VELOCIRAPTOR_PLANT_REVIEW.md) (2026-07-27) | Raptor plant: anatomy vs published *Velociraptor* material, and mechanics | 11 findings, all open — **execution deferred until the T-Rex clears stages 1–3**; see below |

Severity: **HIGH** = wrong results in common cases, **MEDIUM** = edge cases /
robustness, **LOW** = cosmetic / QoL.

---

## Known SB3 ↔ JAX divergences (documented, deliberate for now)

- **Forward-velocity reference frame** — SB3 envs project velocity onto the
  *fixed initial* agent→target direction; the MJX env (and, since PR #426,
  the JAX CPU eval) use the *current* direction each step. (June §2.5)
- **Eval target placement** — JAX CPU eval evaluates against the model's
  fixed target-body position; training randomizes a virtual target 3–8 m
  ahead. Consider sampling eval targets per episode. (July §3)
- **Stage-3 success semantics** — the SB3 Velociraptor and T-Rex environments
  detect geom contact, while MJX uses claw-tip/head-tip distance thresholds.
  The generated species catalog documents both definitions; parity is still
  open.
- **Curriculum gates** — SB3 can advance early after consecutive passing
  evaluations. The JAX CLI evaluates reward once after a full stage; the JAX
  notebook checks reward, episode length, velocity, and success once. These
  paths are documented but not behaviorally equivalent.
- **PPO advantage normalization** — per-minibatch in JAX vs per-batch in
  SB3; acceptable, documented in `jax_ppo.py`. (June §2.7)

Targeted tests now pin the shared NumPy/JAX Velociraptor natural-lean posture
primitive and its per-path runtime routing. A comprehensive per-component
SB3↔JAX reward **parity test** (one fixed state, assert every component within
tolerance) remains the standing recommendation for the divergences above.
(June §6.8; see the
[Stage-1 basin investigation](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md))

## Training / RL

<!-- The three items below come from the 2026-08-05 stage-1 bounce
     investigation; full evidence in
     investigations/TREX_STAGE1_BOUNCE_2026_08.md. -->

- **HIGH** — **T-Rex stage 1 passes or bounces depending on the run, and we
  cannot yet say which is typical.** Three 10M runs, all seed 42: one passed
  with duty 0.0000, two converged to a phase-locked vertical bounce at exact
  integer subharmonics of the 100 Hz control rate (duty 1/6 = 16.7 Hz, and
  1/5 = 20.0 Hz), both with single-support ≈ 0. **The bounce is not
  reward-preferred** — scored under the same reward it is 450 points *worse*
  than the policy that passed, and loses on every term. Every candidate reward
  tweak (`foot_load_balance_airborne_penalty`, `support_conditioned_alive_fraction`,
  `action_jerk_weight`) is already firing, already correct, and already losing,
  so this is an optimisation failure rather than a shaping one and further
  reward changes are not indicated. **The next experiment should be seed
  replicates of the passing configuration**, which is the only thing that
  distinguishes "solved" from "lucky".
- **MEDIUM** — **half the actuators sit saturated and nothing opposes it.**
  Ten to twelve of 21 actuators are pinned at `|action| ≥ 0.99` — tail, neck,
  head, toes — in *both* passing and failing policies. `energy` charges
  ~48/episode against an alive bonus paying ~1000, and `leg_home_pose` governs
  only 8 joints carrying 1.2% of the commanded offset. A saturated actuator has
  no headroom in one direction, so the recovery envelope on those axes is
  one-sided. Not a stage-1 blocker (the passing policy saturates too) —
  tracked as sim-to-real in #491.
- **MEDIUM** — **a passing policy cannot stand if its actions are filtered, at
  any cutoff.** Low-passing the checkpoint that passed the gate collapses it at
  every cutoff from 5 to 35 Hz against a 100 Hz control rate (96 → 351 steps of
  a 1000-step horizon). The high-frequency content is load-bearing closed-loop
  stabilisation, so an action filter or rate limit **cannot be retrofitted** —
  if one is wanted for sim-to-real it has to be present during training. The
  probe conflates bandwidth-dependence with delay-sensitivity; a zero-phase
  offline filter would separate them. Tracked in #491.

<!-- The six items below come from the 2026-07-31 plant validation pass; full
     evidence in PLANT_VALIDATION_AND_STAGE1_OBJECTIVE.md. The reset and
     self-collision defects that pass also found are FIXED in PR #479 and so
     are deliberately not listed here. -->

- **HIGH** — **the stage-1 objective's global optimum is the zero-action
  policy, so no reward threshold can gate it.** Summing the positive T-Rex
  stage-1 weights gives 3.35/step = 3350; the statue collects 3250.27 = **97.0%**
  of it with `energy` and `smoothness` at exactly zero. Every active policy pays
  both — the 7/31 run paid 0.30/step on smoothness alone — so a policy's ceiling
  sits *below* the statue's score. Set a threshold above the statue and stage 1
  is unpassable; below, and a statue passes. Needs the episode-level
  `stance_success` gate (STAGE1_SPLIT_PLAN §2.3), not a better number.
  (PLANT_VALIDATION §9)

- **HIGH → MEDIUM** — **stage 1's real gate machinery does not exist yet.** The
  four `min_avg_reward` values are no longer arbitrary — each is now the §12
  sanity rail (0.89 × its statue's standing reward at the 0.05 operating
  point), `min_avg_episode_length = 950` encodes the full-horizon ≥ 95% floor,
  and stage-1 `reset_noise_scale` sits at 0.05 for every species so survival
  is no longer the binding constraint. But a statue still clears the rail *by
  §12's own design* — the rail only rejects return-discarding policies. What
  separates a competent policy from a statue is the episode-level
  `stance_success` event over unsupported duty (STAGE1_SPLIT_PLAN §2.3), which
  is new machinery: per-episode aggregation, an LCB gate, duty metrics fed to
  both backends' evaluators, and a `stance_quality/v1` gate kind. Until it
  lands, passing stage 1a certifies "did not discard return and did not fall",
  not stance quality. (PLANT_VALIDATION §12/§14)

- **HIGH** — **`foot_load_balance_min_support_force = 0.0` makes the §7.1
  airborne repair a no-op.** `derive_stance_info` and `reward_foot_load_balance`
  differ only in the near-zero branch, yet produced bit-identical output over all
  709 logged rollouts (`max |diff| = 0.000000`, correlation `1.000000`) spanning
  30–67% unsupported duty. The sum of two touch-sensor readings is essentially
  never exactly `0.0`, so the airborne branch never fires. The duty metrics use
  `> 0.1 N` per foot, so a foot at 0.001 N is *unsupported* in the diagnostic and
  *supported* in the reward. Needs a real threshold and a **monotone** ordering
  (airborne strictly worse than single support, not equal). `derive_stance_info`
  still scores true-airborne as perfect balance for the same reason.
  (PLANT_VALIDATION §11.1)

- **MEDIUM** — **`smoothness_weight` penalises action-delta magnitude, not
  frequency, and cannot see a high-frequency limit cycle.** From the 7/31 run's
  best to final checkpoint, `action_delta` *fell* 12.0 → 10.5 and the smoothness
  penalty *improved* −0.286 → −0.250, while toe-motion power above 4 Hz
  **doubled** 35% → 71%. The policy got smoother by the metric while getting
  buzzier in fact. Needs a contact-switch-rate cost or smoothness on the second
  difference of actions. (PLANT_VALIDATION §11.2)

- **MEDIUM** — **`collapse_peak_floor` is an absolute reward value and cannot
  survive a reward-function edit.** Calibrated at 2200 against the 7/29 run
  (rolling-median peak 2496), it left the 7/31 run (peak **1934.1**) permanently
  disarmed through a **−59%** collapse (2148.3 → 888.0; full-horizon 93% → 7%).
  Simulation confirms it never armed; it also confirms that even armed at the old
  inherited 1840 floor, `drop_fraction=0.5` + `patience=10` would not have fired.
  Make the floor relative to the zero-action standing baseline and tighten the
  drop/patience pair independently. (PLANT_VALIDATION §11.4)

- **LOW** — **contact-switch rate conflates bilateral↔single with
  bilateral↔airborne.** The PR #479 plant repair moved T-Rex's raw switch count
  *up* (0.86 → 1.00 /s) while unsupported duty went to **zero** — the extra
  switches are ordinary weight-shifting. Do not gate on it until decomposed;
  gate on unsupported duty instead. (PLANT_VALIDATION §11.3)

- **LOW** — **four stage-1 reward terms are saturated and contribute no
  gradient**: `head_clearance` pinned at exactly its full 0.350 weight in every
  measured window, `height` 0.578 of 0.6, `neck_posture` 0.173 of 0.2,
  `leg_home_pose` 0.312 of 0.5. (PLANT_VALIDATION §14)

- **MEDIUM** — **JAX evaluation cannot produce per-episode foot duty for
  quadrupeds.** `jax_eval` routes per-foot force with
  `results.diag_r_foot if i % 2 == 0 else results.diag_l_foot`, so on a
  four-footed species feet 0 and 2 both land in `diag_r_foot` and feet 1 and 3
  in `diag_l_foot`: the arrays carry two feet interleaved at twice the step
  count, under labels that no longer mean right and left. Bipeds are correct
  (foot 0 → r, foot 1 → l, one entry per step), which is why episode
  boundaries reconstruct exactly from `cumsum(lengths)` there and not for
  quadrupeds. This blocks the adopted 1a duty bound (STAGE1_SPLIT_PLAN §2.3)
  on brachiosaurus and dibothrosuchus, and it became load-bearing when the
  brachiosaurus stance and sensor repairs made its §8 stance-quality row
  interpretable for the first time. The T-Rex pilot is unaffected.

  **Contained, not fixed (2026-08-02).** `jax_eval.stance_panel_from_eval_results`
  refuses to reconstruct a panel unless `len(diag_r_foot) == sum(lengths)` —
  one reading per side per step, which holds for bipeds and gives exactly 2x
  for a four-footed species. A quadruped therefore fails the stance gate
  closed with that ratio named, rather than being scored on mis-paired feet.
  The routing defect itself is unchanged; fixing it still means keying feet by
  sensor identity instead of `i % 2`.

- **MEDIUM** — **the JAX backend cannot finalise a stance-gated result bundle.**
  `result_bundle.evidence` certifies a `stance_quality/v1` stage by re-deriving
  its criteria from `stage<N>/stance_panel_selected.csv`, the per-episode duty
  record `write_stance_gate_report` emits. Only the SB3 path writes it:
  `generate_stage_artifacts` calls `_write_stance_gate_report`, and
  `save_jax_stage_artifacts` has no equivalent. A JAX run whose stage 1
  declares the stance gate will therefore train all three stages and then fail
  bundle finalisation with `stance_panel_selected.csv is missing`.

  This is a **fail-closed** limitation, not a wrong verdict — the bundle
  refuses rather than certifying stance quality nobody recorded — and it is
  not a regression: the same bundle previously refused unconditionally, on
  every backend. What changed is that the SB3 path is now unblocked and the
  JAX path is not.

  The measurements exist on the JAX side already:
  `jax_eval.stance_panel_from_eval_results` reduces `diag_r_foot`/`diag_l_foot`
  into per-episode duties before summarising them into a `StancePanel`. Fixing
  this means returning those per-episode duties alongside the panel and having
  `save_jax_stage_artifacts` write them through the same
  `write_stance_panel_evidence` the SB3 path uses — deliberately the same
  writer, so the two backends cannot disagree about the evidence format the
  auditor reads. Note the quadruped restriction above applies to that
  reconstruction too.

- **LOW** — **collidable necks are deferred until terrain lands.** Velociraptor
  is the reference: its neck geom collides *and* sits in `_body_ground_geoms`,
  so hitting the ground with it terminates the episode. The other three carry
  `contype=0` necks (plus cosmetic `brow_ridge` / `crest` / `sagittal_crest`
  and dibothrosuchus' twelve `scute`s), and brachiosaurus documents the choice
  explicitly, using the collidable head as the termination proxy. On a flat
  floor this is unobservable — an animal whose neck reaches the ground has
  already tripped tilt, height or head-contact termination — so the decision
  was to leave physics alone and revisit when heightfield terrain arrives, at
  which point the raptor's pattern is the template. Note the MJX settle
  currently *raises* on a heightfield floor and would need an iterative settle,
  and newly-colliding long neck capsules must be checked for home-pose
  self-collision (the defect class fixed twice in the PR #480 series). The
  cosmetic geoms should stay non-collidable permanently; they are already
  excluded from the ground-settle probe.

- **LOW** — **the reset's root-height jitter channel is state-inert but still
  present.** The PR #479 ground settle overwrites the root height as a pure
  function of the sampled joint pose, so `reset_height_noise_scale` and
  `_bounded_reset_height_delta` no longer reach the post-reset state (verified
  to one ULP; pinned by `TestHeightJitterIsInertSinceGroundSettling`). The RNG
  draw is deliberately kept — removing it would shift every subsequent draw and
  re-anchor all seeded baselines, including the PLANT_VALIDATION §6 tables.
  Remove the whole channel (draw, knob, bound) at the next policy-interface
  revision. Note this also retires PLANT_VALIDATION §16's reset-height-clip
  hypothesis *going forward*: the clip cannot influence any future run.

- **MEDIUM** — **the policy saturates its action bound, and the
  `diagnostics/action_*` family mixes pre-clip and post-clip quantities.**
  Measured on T-Rex stage-1 run `20260727_130726` (PPO, 6.0M steps), from its
  own tensorboard:

  | `diagnostics/` scalar | first | last | mean | max | measured on |
  |---|---|---|---|---|---|
  | `action_abs_max` | 4.55 | 6.71 | **6.11** | **7.77** | pre-clip |
  | `action_std` | 1.00 | 1.93 | 1.42 | 1.93 | pre-clip |
  | `action_saturation` | 0.32 | **0.68** | 0.49 | 0.68 | pre-clip |
  | `action_delta` | 21.27 | 6.24 | 15.04 | 21.43 | post-clip |

  `action_saturation` (fraction of components at or beyond 0.99) rose
  **monotonically** across all 6M steps — 0.32 → 0.68. The policy's raw output
  peaks at 7.77 against a `Box(-1, 1)` action space. Note `train/std` *fell*
  over the run (1.00 → 0.72) while empirical `action_std` nearly doubled: the
  policy is pushing its **mean** out of bounds, not widening its exploration.

  **What this is not: the reward is not inflated.** Both training paths clip
  before stepping the environment, so `_get_reward_info` receives an in-bound
  action and both penalties are computed on it:

  - SB3 `on_policy_algorithm.py:214-218` — `clipped_actions = np.clip(actions,
    low, high)` immediately before `env.step(clipped_actions)`; `policies.py:379`
    does the same inside `predict()`, which is what both eval loops use.
  - `jax_trainer.py:365` — `actions = jnp.clip(raw_actions, -1.0, 1.0)` before
    `env.step`; `jax_ppo.sample_action`'s docstring states the contract
    ("returns the **unclipped** action… callers must clip before sending to the
    environment").
  - `base_env.py:_scale_action` says so directly: "SB3 already clips before
    stepping, but direct callers… would otherwise command out-of-range ctrl."

  **The metric hazard.** `action_mean`, `action_std`, `action_abs_max` and
  `action_saturation` come from `diagnostics.py:210`, which reads
  `self.locals["actions"]` — SB3's **pre-clip** Gaussian sample. `action_delta`
  arrives by a different route: it is returned by `reward_action_smoothness`
  from *inside* the env, so it is computed on the **post-clip** action. Four
  scalars in one namespace describe the policy's raw output; the fifth
  describes what the plant received. Reading the group as one space is an easy
  and consequential mistake.

  **What is real.** PPO stores the raw action and its `log_prob`, while the
  environment responds to the clipped one. With 68% of components saturated,
  most of the policy's output distribution sits where moving the mean further
  changes the plant not at all — the standard bias from sampling an unbounded
  Gaussian into a bounded action space, and a plausible contributor to the
  bang-bang envelope measured on the same run (`used` = 100% of range on 20 of
  21 actuators). Worth watching `action_saturation` as a first-class health
  metric rather than as evidence about the reward.

  **Latent trap.** `base_env.py:783` passes the raw `action` to
  `_get_reward_info` while `ctrl` is clipped separately at line 765. Harmless
  under SB3 and the JAX trainer today, but any direct caller — a notebook, a
  custom rollout loop, a diagnostic script — is silently charged energy and
  smoothness penalties for magnitude the plant never sees. Clipping at line
  783 would make the two paths agree and moves no fingerprint.

  **Explicitly retracted.** An earlier version of this entry claimed the energy
  term was inflated ~3.8× (~209/episode, ~7.9% of return) and that the
  `r ∝ w^-0.16` smoothness fit in
  [TREX_LEG_FLEXING_PLAN.md](TREX_LEG_FLEXING_PLAN.md) was therefore unsound.
  Both are wrong: the reward saw clipped actions, and `r` derives from
  `action_delta`, which is post-clip and so is coupled to the physics. That
  study stands, no `min_avg_reward` gate needs re-deriving, and no historical
  reward comparison is invalidated. (2026-07 T-Rex telemetry review)

- **LOW** — `BaseDinoEnv.reset` still applies one `reset_noise_scale` scalar to
  the whole of `qvel`, which mixes root linear velocity (m/s), root angular
  velocity (rad/s) and joint velocities (rad/s). This is the same
  units-conflation that made the root-height jitter wrong, but far less severe:
  a velocity kick has to actually defeat the controller, whereas the height
  jitter could spawn an episode already outside `healthy_z_range`. Worth
  separating if a species much smaller than Dibothrosuchus is ever added.
  (2026-07 Dibothrosuchus review)
- **MEDIUM** — two code paths decide "is this species a quadruped?" by
  different means and can disagree. `mjx_env.build_mjx_observation` tests
  `"torso" in body_ids`; `plant_contract._policy_interface_payload` tests
  `observation_schema == "quadrupedal-target/v1"`. A registration declaring a
  bipedal schema with a `torso` root would error in the plant contract (so CI
  catches it) but silently pick the torso root at runtime. Give the MJX
  registration the observation schema, or assert exactly one of
  `{"torso", "pelvis"}` in `body_ids`. (2026-07 Dibothrosuchus review)
- **LOW** — `plant_contract._mocap_target_name` now requires *every* plant to
  declare exactly one mocap body. All four comply and it fails loudly, but the
  constraint was introduced to derive a segment label, not because the contract
  needs uniqueness. (2026-07 Dibothrosuchus review)

- **LOW** — T-Rex SB3 env silently accepts `foot_contact_weight` /
  `foot_contact_gate` (JAX-only params) without using them; typo'd weights
  do nothing. Reject unknown env kwargs loudly. (June §1.6)
- **LOW** — `CurriculumCallback` / `LocomotionMetrics` hardcode success keys
  (`bite_success`, `strike_success`, `food_reached`) instead of using
  `SpeciesConfig.success_keys`. (June §6.4)
- **LOW** — `curriculum/advancement.py` `_read_latest_eval`: the
  `successes.shape[0] == n_evals` guard permanently discards npz successes
  if SB3 starts recording them one eval late. (July §2)
- **LOW (JAX)** — two same-named `check_stage_gate` functions with different
  signatures (`jax_eval` vs `jax_curriculum`); logged `learning_rate` decays
  faster than the real schedule (display-only); KL early-stop inside
  `lax.scan` still computes-then-discards remaining minibatch gradients;
  `StabilityMonitor` default `kl_warn=100` only fires after total collapse;
  eval per-step reward diagnostics decompose forward velocity in world-X
  rather than the agent→target frame the total now uses. (June §2.7, §6.7;
  July §3)
- **MEDIUM (perf)** — `EvalCallback` runs 30 serial episodes every 50k steps
  plus supplementary + post-stage evals — up to ~3.6M serial eval steps per
  6M-step stage. Vectorize the eval env or trim episodes. (June §6.1)
- **Experiment** — consider tightening Brachiosaurus
  `head_proximity_max_dist` (~2.0 m) to concentrate the last-mile
  food-reach gradient. (June §6.10; the rest of that finding is resolved —
  the published summary is now the 2026-07-18 run, which passes its
  stage-3 gate with success rate 1.0.)
- **Watch item** — the biped home-residual action mapping has a
  slope discontinuity at action 0 (the two piecewise segments span
  home→min and home→max, which are unequal), so zero-mean Gaussian
  exploration produces a physically biased mean command away from home
  wherever the home control is off-midpoint. The listed Velociraptor effect
  is hip pitch ≈ −0.29 rad toward flexion, knee ≈ −0.16, ankle ≈ +0.13 at
  σ=1; the T-Rex lower-body home controls are midpoint-aligned, so its
  asymmetry is limited to neck/head controls. If a fresh run's early training
  looks persistently off-home while `algo_std` is still ≈1.0, suspect this
  bias before suspecting rewards; possible mitigations (smaller
  `log_std_init`, a smoothed mapping) are interface experiments and must be
  run in isolation. (Stage-1 basin investigation follow-up)

## Sweeps / infrastructure

- **MEDIUM** — `env_*_range_min` / `env_*_range_max` sweep parameters crash
  the trial runner. `configs/brachiosaurus/sweep_{ppo,sac}.json` sweep
  `env_food_distance_range_min` / `_max` (and `env_food_height_range_*`), and
  `configs/dibothrosuchus/sweep_ppo.json` copies the pattern with
  `env_prey_distance_range_*`. Nothing in `scripts/sweep/` reassembles the
  `_min`/`_max` suffix into the tuple the constructor wants, so each becomes
  `env_kwargs["food_distance_range_min"]` and the env raises
  `TypeError: unexpected keyword argument` (verified for both species). Either
  implement suffix pairing in `_apply_overrides`, or drop those six keys.
  (2026-07 Dibothrosuchus review)

- **MEDIUM (cleanup)** — `ray_orchestration.py` (759 lines) has zero
  callers; `ray_tune_sweep.ipynb` re-implements it inline and the copies
  have already diverged once. Wire the notebook to the module or delete the
  module. (July §4)
- **LOW** — quality scoring weights `cost_of_transport` / `vel_consistency`
  that only the notebook path exports, so scores aren't comparable across
  paths; a single trial missing a metric drops that metric for the whole
  set. (July §4)
- **LOW** — `_handle_stage_failure` uses `os._exit(1)` (skips
  atexit/W&B finalizers); `scoring.compute_quality_scores` sorts the
  caller's list in place; `plot_sweep_results` uses deprecated
  `tempfile.mktemp`; `collect_ray_results` may emit empty `trial_id`s when
  Ray returns ids as the index; `metrics.py` `velocity_consistency`
  explodes when mean velocity ≈ 0; `load_resume_settings` reads a
  `gpu_model` key never written and can't store `seed=0`;
  `_is_retryable_gcp_error` treats the generic `GoogleAPICallError` name as
  retryable; thread-unsafe CSV appends under concurrent local runs.
  (June §3.3; July §1/§4; CODE_REVIEW §2.1#1)
- **LOW** — JAX `TrainingCSVLogger` flushes per update directly to the
  output path — one network write per update on `/gcs` FUSE; buffer locally
  like `tb_sync` when `_is_gcs_path(path)`. (July §1)

## Post-training artifacts (recommended additions)

1. Vertex-side sweep manifest at submit time (resolved search space, image,
   machine type, HPT job resource name/console URL) for parity with the Ray
   path's `save_search_space`. (July §4)
2. `best_trial_config.json` per stage on the Vertex path + a
   `model_manifest.json` next to every exported `best_model.zip`
   (trial id, hyperparameters, seed, eval metrics, library version,
   VecNormalize pairing). (July §4)
3. Persist the curriculum gate history (`CurriculumManager.summary()` →
   `curriculum_state.json`) and the achieved gate metrics in
   `curriculum_results.csv`; include CLI overrides in the saved effective
   config. (July §4; June §6.5)
4. CLI curriculum parity with the notebook: call `generate_stage_artifacts`
   per stage and run the quality eval so `metrics.json` exists on that path.
   (July §4)
5. W&B: pass a render-capable eval env into `WandbCallback` (its video path
   is dead code), log final metrics before `finish()`, upload
   `best_model.zip` + vecnorm as W&B Artifacts, log the run URL into
   `stage_config.json`. (July §4; June §6.6)
6. Sweep CSV rows lack a run timestamp / `resume_run` id, so rows merged
   across resume cycles are indistinguishable. (July §4)

## MuJoCo models (July 2026 model review)

Fixed on this branch: the brachiosaurus could not physically hold any torso
height in its alive region (home-keyframe-controlled settle z≈0.68,
straight-leg command z≈0.70, vs `healthy_z_range` floor 1.0 and height target
1.2) — leg springs now reference the stance angles and the leg servos are
stronger with bounded force, giving an actively servo-held home stand at
z≈1.13, level, all four feet grounded. The T-Rex's former step-77 neutral
nosedive is also fixed: a shallow plantar contact, stance-referenced springs,
mass-scaled gait servos, named-home residual actions, and live load-bearing
foot sensors give it a full-horizon neutral stand; see
[the home-equilibrium investigation](investigations/TREX_HOME_EQUILIBRIUM.md).
All three models now use
`integrator="implicitfast"` and bounded
`forcerange` on position actuators, sized to clip impact/reset spikes only,
with gait-critical actuators at 1.5×kp on every species (raptor hip
pitch/knee/ankle; trex hip pitch/knee/ankle; brachiosaurus all four hip
pitches — the raptor knee joined the 1.5× set in July 2026 after measuring
0 % clip at the moderate 2.5 Hz/0.8-amplitude regime but 30–46 % at
sprint-like 3–4 Hz full-amplitude excitation while still capped at
0.8×kp). The original home-control-only sizing clipped 20–50 % of
gait-cycle torque per species and collapsed velociraptor stage-2 twice — see
[investigations/STAGE2_RECOMMENDATIONS.md](investigations/STAGE2_RECOMMENDATIONS.md)
§5. Post-fix clipping on the pinned gait-critical hip/knee/ankle actuators is
≤0.5 %, covered by each species' `tests/test_actuator_bounds.py`; re-measure
any re-sizing with
`environments/shared/scripts/actuator_saturation_report.py`.

Neutral-action stability and truly actuator-disabled passive behavior are now
separate test contracts. Layered policy, physics, visual, and source identities
are documented in [PLANT_CONTRACT.md](PLANT_CONTRACT.md).

### Foot touch sensors under-report on two species — open (July 2026 sensor audit)

A MuJoCo touch sensor sums only contacts on geoms belonging to its site's **own body**, so a
site on a parent segment silently misses whatever the child geoms carry. Auditing all four
species against `mj_contactForce` found T-Rex and dibothrosuchus correct (ratio 1.000) and two
species wrong:

| species | sensor / measured contact | missing |
|---|---|---|
| velociraptor | **0.553** | `metatarsus` 17.54 N + `toe_d4` 12.03 N per foot; the site is on `toe_d3` only |
| brachiosaurus | ~~0.000~~ **FIXED** — now 1.000 | was everything; repaired per plant_versions note 8 (pad sites enlarged, meta sensors appended, pad + meta summed on both backends) |

Total floor reaction equals body weight on all four species, so the contacts are real and the
plants are in equilibrium; these are sensor-scope defects, not physics ones. `aa3395c` fixed the
raptor site's *size* but not its *body scope*, one repair short of `aa87445`.

The remaining raptor defect does not reach a stage-1 reward term today — its stage-1 config sets
none of `foot_contact_gate`, `foot_contact_weight`, `bilateral_support_weight` or
`foot_load_balance_weight`, and its `gait_symmetry_weight` is 0.0. It does reach the
**observation**: the raptor's policy sees 55% of true per-foot load. (Brachiosaurus's four
permanently-zero input channels were revived by the note-8 repair.)

Repair is an MJCF change of the `aa87445` shape — per-geom touch sites and sensors, appended so
existing sensor indices keep their positions, summed per foot on both backends — and moves that
species' physics and policy fingerprints. Full evidence, method and reproduction in
[investigations/FOOT_SENSOR_VERIFICATION.md](investigations/FOOT_SENSOR_VERIFICATION.md);
re-check any repair with `environments/shared/scripts/foot_sensor_report.py`.

### Velociraptor plant — open (July 2026 raptor review)

**The stance-referenced-spring migration above never reached the raptor.** It
is the only species still carrying the pre-fix arrangement, and the
consequences compound. Full evidence and method in
[reviews/VELOCIRAPTOR_PLANT_REVIEW.md](reviews/VELOCIRAPTOR_PLANT_REVIEW.md).
**Execution is deferred until the T-Rex clears stages 1–3** on the corrected
stance (PR #464).

| species | \|leg spring torque\| at home | `springref` outside the joint limit |
|---|---|---|
| **velociraptor** | **145.21 N·m** | **4 joints** |
| trex | 0.00 N·m | 0 |
| brachiosaurus | 0.47 N·m | 0 |
| dibothrosuchus | 0.00 N·m | 0 |

- **HIGH — stage 1 is already solved by doing nothing.** A zero-action policy
  scores 1704.93 ± 259.12 at 98% full-horizon survival against
  `min_avg_reward = 100.0`; it clears the gate **17×** and is promoted into
  stage 2. Same failure the T-Rex config fixed by re-deriving its gate from the
  measured statue floor. The reset-noise calibration was not carried over
  either — the raptor is still at 0.05, measured at 97% statue survival, where
  0.10 gives 80%. *Config-only fix, no checkpoint cost.*
- **HIGH — the plant does not stand on its actuators.** No raptor leg joint
  sets `springref`, so the springs are neutral at `qpos = 0` — which is
  *outside the legal range* for the knee and ankle, making them a permanent
  one-directional bias rather than a restoring element. Zero-action survival is
  95% as committed, **0% with the springs deleted, and 0% with the same
  stiffness anchored at the stance** (falls in ~1.4 s either way). The support
  comes from the offset, not the stiffness. Deleting the T-Rex's leg springs,
  by contrast, changes nothing (55% → 55%). Fixing this requires re-sizing the
  leg actuators at the same time — exactly the pairing the brachiosaurus fix
  needed.
- **HIGH — foot touch sensors report 55.6% of transmitted force.** The
  `r_foot`/`l_foot` sites sit on the `toe_d3` bodies, so digit IV (12.07 N) and
  the metatarsus (17.36 N) are invisible against 36.79 N sensed of 66.22 N
  real. This is the *same defect* as the T-Rex foot-contact repair (which was
  at 77.6%); the raptor is worse and was never brought along. Foot contact is
  a trained observation and feeds the JAX `foot_contact_gate`.
- **HIGH (fidelity) — the metatarsus is 78% too long** relative to the femur:
  model MT III/femur 0.741 against 0.416 (Persons & Currie 2016, *Sci Rep*
  6:19828, Table 1, IGM 100/986) and ~0.51 from a second specimen (Norell &
  Makovicky 1999, *AMNH Novitates* 3282). It also bears 26.2% of each foot's
  load and forms the *rear* edge of the support polygon, so the "digitigrade"
  foot is functionally part-plantigrade. tibia:femur is within 3.6% and correct
  — leave it alone.
- **MEDIUM — `natural_pitch` is stale by 4.0°.** Configured 0.35, the plant
  settles at 0.4200. Because the raptor centres its posture reward on that
  angle, standing naturally costs **~104 reward/episode** (1.745 → 1.850 per
  step). *Verified free — the plant manifest stays current, so no checkpoint is
  invalidated.*
- **MEDIUM — the two claw motors are the only unbounded actuators** in any
  plant (`forcelimited=False`, `gear=50`): 693 N at the claw tip, 5.2× body
  weight, on the geom that scores stage 3. The July 2026 `forcerange` sweep
  missed them.
- **MEDIUM — SB3/MJX termination asymmetry.** SB3 terminates on floor contact
  of torso, neck, head and tail_3/4/5; the MJX registration lists only the
  three tail bodies and no `termination_site_heights`. On MJX the raptor can
  put its face on the ground without terminating.
- **LOW — `nosedive_termination_threshold` is hardcoded** at
  `raptor_env.py:530` while the MJX path reads it from stage config. They agree
  today only because no raptor TOML sets the key.
- **Not recommended:** porting the T-Rex stance correction here. That argument
  rests on a live stage-1 height term forcing knee travel through a
  near-singular joint, and the raptor env has **no height reward at all** —
  five height mentions, none of them a reward term, against 21 in
  `trex_env.py`.
- **Note for the hardware track:** because the springs are load-bearing, the
  raptor's true actuator requirement is *higher* than its sim actuator forces
  suggest, which pushes against the torque crux already flagged in
  [hardware/HARDWARE_BOM.md](hardware/HARDWARE_BOM.md) §2.1. Magnitude needs
  the retune; only the direction is known.

> **Note:** these changes alter the physics plant. Policies trained before the
> change are incompatible by contract, including when a change seems marginal;
> use an explicit legacy override only for deliberate historical evaluation.

Still open:

- **LOW** — the raptor's toe-clipping margin now sits at the toes: at
  sprint-like excitation (3–4 Hz, full amplitude) the 0.8×kp toe caps clip
  10–16 % and the 1.5×kp hip pitch ~11–16 % (its physical envelope); the
  knee measures 0 % after its 1.5× bump. Re-measure with
  `environments/shared/scripts/actuator_saturation_report.py` before any
  faster-gait (stage-3 sprint) work and consider 1.5× toes if they bind.
- **LOW** — scene boilerplate (skybox/grid/floor/option) is copy-pasted
  across the three XMLs → extract a shared `scene.xml` include; limbs are
  hand-mirrored sign-flips → generate via script/PyMJCF or add a left/right
  symmetry test.
- **LOW** — raptor claw `motor gear="50"` on a 0.05 kg claw (huge
  torque-to-inertia; slams its limits); contype-0 neck geoms can visually
  clip the floor; no `<light>`/`<visual>` block for nicer renders; brachio
  food has no collision partner.
- ~~HIGH — the brachiosaurus cannot hold its home stance.~~ **FIXED**
  (plant_versions notes 7–8): the collapse decomposed into the midpoint action
  mapping never commanding the home pose (knees dragged up to 0.35 rad off)
  and leg servos that sagged 71.6 mm under static weight, leaving the planted
  stance's roll stiffness at parity with `m·g·h`. Brachiosaurus now uses the
  home-keyframe-residual mapping like the other species, leg kp is doubled
  (sag 11.1 mm), and the zero-action baseline is 40/40 full-horizon at
  1739.08 ± 1.17 (was 0/40 at 163.35 ± 81.40). The full-horizon neutral test
  this entry asked for now exists on the brachiosaurus
  `TestNeutralActionStability` subclass.
- **LOW** — the T-Rex `tail_1_geom` overlaps both thigh capsules by 18.8 mm at
  the home keyframe, injecting a constant self-contact force into the stance
  (pre-existing; unchanged by the July 2026 plant revision, which measured it
  rather than fixing it). Either exclude the pair like the sibling toes, or
  reshape `tail_1` so the overlap is gone; re-measure the home stance after.
- **Experiment** — with `implicitfast`, a `timestep` 0.002→0.004 A/B is
  worth running (halves sim cost if stable).

## Configs, docs & website

- Brachio stage-2 `natural_pitch = -0.15` while stages 1/3 use 0.0 —
  intentional? (June §4)
- `pyproject.toml` gymnasium entry-point groups are likely dead config
  (envs self-register at import); `[all]` omits `[mjlab]`. (June §4)
- `docs/investigations/REWARD_SCALE_REDESIGN.md` uses `*_bonus_weight` key
  names that don't exist. (June §5)

## Notebooks

- `ray_tune_sweep.ipynb` duplicates `ray_orchestration.py` (see above); its
  `EVAL_EPISODES` knob doesn't affect the in-trial eval episode count. (July §5)
- The notebooks now pin the plant compiler (`mujoco`/`mujoco-mjx`) and Ray
  Tune compatibility range, but still use broad `stable-baselines3`, `jax`,
  Flax, and Optax ranges; pin complete lockfiles for reproducible training.
  (July §5)

## Testing / CI

- **LOW** — `hw_chassis_study.py`'s excitation drives hip pitch and knee in
  phase, unlike a real trot where knee flexion leads swing. This contributes to
  the knee pinning at its 22 N·m cap in all 18 runs, so that column is reported
  as "not a configuration discriminator"; the ab/ad conclusions are unaffected
  (that joint is driven at 0.25x and does discriminate).
  (2026-07 Dibothrosuchus review)
- TOML→env round-trip test: construct each env with each stage's
  `env_kwargs`, assert no unknown/unused keys. (June §6.8)
- SB3↔JAX reward parity test (see divergences section above). (June §6.8)
- The four CI test jobs are near-identical — a matrix would halve the YAML.

## Open questions

1. Is SB3↔JAX reward parity a hard goal? If yes, the parity test should
   gate CI; if the JAX path is a research spike, keep the divergence table
   above authoritative. (June §7)
2. Published sweeps ran with identical seeds per trial (fixed going
   forward); re-run top-3 configs with 3 seeds before locking them into the
   TOMLs. (June §7)
