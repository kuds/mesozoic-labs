# Full Repository Review — June 2026

Scope: all documentation, training code (SB3 + JAX/MJX), sweep/eval/reporting
infrastructure, configs, and shipped experiment results. Focus: RL-correctness
bugs, robustness, and training/debugging quality-of-life.

Severity legend: **CRITICAL** = corrupts results or breaks a documented workflow,
**HIGH** = wrong results in common cases, **MEDIUM** = wrong in edge cases /
robustness, **LOW** = cosmetic / QoL. Items marked *(verified)* were confirmed
by reading the exact code paths and/or the shipped result files.

---

## 1. RL-correctness bugs — SB3 path (the path that produced the published results)

### 1.1 `LocomotionMetrics._dt = 0.02` is 2× the real control dt — **HIGH** *(verified)*
`environments/shared/metrics.py:58` hardcodes `_dt = 0.02` ("default timestep ×
frame_skip"), but every species XML uses `timestep="0.002"` with `frame_skip=5`,
so the real dt is **0.01 s**. Consequences (metrics only — rewards are unaffected):

- `total_distance` (`metrics.py:187`) overestimated 2×
- `cost_of_transport` underestimated 2×
- `stride_frequency` (`metrics.py:342`) underestimated 2×
- `time_to_target` (`metrics.py:248`) overestimated 2×

These flow into `metrics.json` (quality eval), the `eval` CLI output, and the
curriculum locomotion logs. **Fix:** accept `dt` in `LocomotionMetrics` and plumb
`env.dt` (already computed in `BaseDinoEnv.__init__`) through
`eval_policy_quality`, `CurriculumCallback`, and `evaluate`.

### 1.2 `LocomotionMetrics` success_rate is per-step, not per-episode — **HIGH** *(verified)*
`metrics.py:265`: `result["success_rate"] = float(np.mean(self._success_events))`
where `_success_events` is appended every step. An episode that succeeds on its
terminal step of a 200-step episode reports `success_rate = 0.005`. The `eval`
CLI then prints "Success rate: 0.5%" (`evaluation.py:406-410`) for a policy that
succeeds 100% of the time. Meanwhile the *episode-level* success rate (correct)
is computed separately in `eval_policy` / `CurriculumCallback`. Two metrics share
one name with different semantics. **Fix:** in `compute()`, report
`success_rate = 1.0 if any(success_events) else 0.0` (episode-level), keep
`total_successes` if the step count is wanted.

### 1.3 SAC stage-warmup entropy manipulation is a no-op, and the restore is harmful — **HIGH** *(verified)*
`curriculum.py:684-686`: `StageWarmupCallback` sets `self.model.ent_coef = 0.02`
for SAC. With `ent_coef="auto"` (all SAC configs), SB3's `SAC.train()` reads
`self.log_ent_coef` via `ent_coef_optimizer` and **ignores the `ent_coef`
attribute** — so the warmup does not fix the entropy coefficient at all; auto-
tuning continues. Then at restore (`curriculum.py:713-717`)
`log_ent_coef.data.fill_(self._original_log_ent_coef)` **rolls the learned
entropy coefficient back to its stage-start value**, discarding 100k–300k steps
of tuning. Net effect: the SAC warmup is "LR reduction + entropy rollback".

Also: the docstring says "The critic learning rate is unchanged so Q-values
converge quickly", but the `lr_schedule` override is applied by SB3's
`_update_learning_rate` to the **actor, critic, and entropy optimizers alike**.

**Fix:** to actually freeze entropy during warmup, set
`log_ent_coef.data.fill_(log(warmup_ent_coef))` and zero/freeze its optimizer
(or simply don't touch entropy for SAC); drop the restore-fill; correct the
docstring or implement per-optimizer LR scaling.

### 1.4 Stage-advancement gates rely on a 5-episode success sample — **HIGH** *(verified)*
`CurriculumCallback` defaults `supplementary_episodes=5` (`curriculum.py:367`)
and `train_curriculum` doesn't override it. Stage-3 gates use
`min_success_rate = 0.5` with `required_consecutive = 3`:

- success measured on 5 episodes → granularity 0.2;
- a policy with true 50% success passes one check with p≈0.5 → passes 3
  consecutive with p≈0.125;
- a 70% policy passes 3 consecutive with only p≈0.59.

So advancement on stage 3 is substantially a coin flip, and `min_avg_forward_vel`
(stage 2) is estimated from the same 5 episodes. Note the
`StageThreshold.min_eval_episodes = 10` guard is satisfied by the **30 reward
episodes** from EvalCallback, so it never protects the success/velocity sample.
**Fix:** raise `supplementary_episodes` to ≥20 for stages with success/velocity
gates, or extract per-episode success from EvalCallback's eval loop (via a
callback collecting `info`) so all gate metrics share the same 30 episodes.

### 1.5 Brachiosaurus success termination is not gated on the bonus — **MEDIUM** *(verified)*
`brachio_env.py:497` terminates with `success=True` whenever
`head_food_dist < food_reach_threshold`, in **all stages**. Raptor and T-Rex
gate the equivalent check on `strike_bonus > 0` / `bite_bonus > 0`
(`raptor_env.py:497`, `trex_env.py:505`). In brachio stages 1–2
(`food_reach_bonus = 0`), accidental head-food proximity ends the episode early
and records success — visible in the shipped results
(`results/brachiosaurus/ppo/collected_results.csv` stage 2:
`mean_success_rate = 0.0333`). It also slightly caps stage-2 episode
length/reward. **Fix:** add `if self.food_reach_bonus > 0 and ...` to match the
other species (the velociraptor stage-2 0.0333 is different and harmless — a
claw graze sets the info flag without terminating).

### 1.6 T-Rex env silently accepts unused reward params — **LOW** *(verified)*
`trex_env.py:97-98,128-129`: `foot_contact_weight` / `foot_contact_gate` are
stored but never used in `_get_reward_info`. The TOML comments document them as
JAX-only, but accepting-and-ignoring kwargs in the SB3 env is a footgun (a typo'd
or misunderstood weight silently does nothing). **Fix:** either implement them in
the SB3 reward, or move them to the `[jax]` section only and have the env reject
unknown kwargs.

### 1.7 `_scale_action` doesn't clip incoming actions — **LOW** *(verified)*
`base_env.py:777-785` linearly maps action → ctrl without clipping to [-1, 1].
Safe under SB3 (which clips in `collect_rollouts`/`predict`), but any direct
`env.step(raw_gaussian_action)` from a custom script commands out-of-range ctrl.
One-line `np.clip(action, -1.0, 1.0)` makes the env self-defending.

### What's right (worth keeping as-is)
- Termination vs truncation semantics are correct (Gymnasium API + SB3 ≥ 2.2
  bootstraps truncations properly).
- VecNormalize handling is unusually careful: SAC `norm_reward=False`
  (replay-scale drift), `obs_rms` carried across stages with `ret_rms` reset,
  best-model checkpoints paired with matched `best_model_vecnorm.pkl`, eval envs
  consistently set `training=False, norm_reward=False` (with try/finally), and
  SB3's `EvalCallback` auto-syncs train→eval normalization stats.
- Stage-3 reward design (alive_bonus=0 + terminal bonus ≫ discounted future
  value) correctly avoids the "don't-end-the-episode" trap, and the configs
  document the arithmetic.

---

## 2. RL-correctness bugs — JAX/MJX path

### 2.1 JAX curriculum gate reads the wrong config key → never passes — **CRITICAL** *(verified)*
`jax_curriculum.py:33`: `check_stage_gate` reads `stage_config.get("curriculum", {})`,
but `load_stage_config` (`config.py:196`) returns the section under
`"curriculum_kwargs"`. `min_reward` therefore defaults to `float("inf")` and the
gate **always fails** — `run_curriculum` stops after stage 1 every time.
**Fix:** read `"curriculum_kwargs"`; add a round-trip test that loads a real TOML
and asserts the gate can pass.

### 2.2 TOML→MJX reward-weight key translation gaps — **HIGH** *(verified)*
`mjx_env.py:381-383` reads the approach weight as
`weights.get("bite_approach_weight", weights.get("approach_weight", 0.0))`, and
TOML `[env]` keys are merged over the species registry defaults
(`mjx_env.py:200-213`). But:

- velociraptor TOMLs use `strike_approach_weight`, brachiosaurus use
  `food_approach_weight` — **neither is ever read**;
- both species registries default `approach_weight = 1.0`
  (`velociraptor/mjx_config.py`, `brachiosaurus/mjx_config.py`), so approach
  shaping **leaks into JAX stage 1/2 at weight 1.0** while SB3 has it at 0, and
  the stage-3 values (3.0) silently don't apply.

The trex stage-1 TOML comment ("Must be explicit — species default was leaking
into stage 1 via key mismatch bug") shows this bug class was found for trex and
patched *in the TOML* instead of fixed in code; velociraptor/brachio are still
exposed. Related: MJX success thresholds are static registry values (brachio
`success_threshold=0.50`) while the SB3 stage-3 TOML now uses
`food_reach_threshold = 0.8` — the two backends use different success
definitions. **Fix:** one explicit per-species key-translation map (e.g.
`strike_approach_weight → approach_weight`), and **fail loudly** when a TOML
`[env]` key is neither a known env param nor a known reward weight.

### 2.3 Height-maintenance reward: termination ceiling used as target height — **HIGH** *(verified)*
`mjx_env.py:423-424` calls
`reward_height_maintenance(z, healthy_z_range[0], healthy_z_range[1], w)` — the
third parameter of the shared function is `target_z` (the standing height). SB3
uses 0.90 m (trex) / 1.2 m (brachio); JAX uses the termination ceiling (1.6 /
3.5 m), making the gradient ~5–10× flatter. `jax_reward_termination.py:53-54,140`
names the parameter `healthy_z_max`, and `jax_eval.py` hardcodes 0.90 for all
species. Three inconsistent conventions for one function. **Fix:** add a
per-species `target_standing_z` to `MJXEnvConfig` and use it everywhere.

### 2.4 JAX checkpoints don't save optimizer state (or RNG) — **HIGH** *(verified)*
`jax_trainer.py:762` (periodic) and `jax_trainer.py:812` (final) save only
`params + obs_rms + history`. `save_checkpoint` supports `opt_state` but it's
never passed; resuming reinitializes Adam moments and the LR-schedule step,
silently degrading resumed runs. **Fix:** save/restore `opt_state`, the PRNG
key, and the update counter; add `JaxTrainer.resume_from_checkpoint()`.

### 2.5 MJX forward-velocity reference frame diverges from SB3 — **MEDIUM** *(verified)*
`mjx_env.py:341-344` projects velocity onto the **current** agent→target
direction each step. The SB3 envs deliberately use the **fixed initial**
direction (`raptor_env.py:143-147`) to stop the reward flipping sign when the
agent passes the prey — the MJX path reintroduces exactly that behavior.
**Fix:** store the initial direction in `EnvState` at reset and use it.

### 2.6 Hardcoded eval config in `jax_setup.run_stage_evaluation` — **MEDIUM** *(verified)*
`jax_setup.py:541,545`: `EvalConfig(target_body="prey", ..., forward_vel_max=8.0)`
for every species — brachiosaurus' target body is food and its
`forward_vel_max` is 1.0 (velociraptor 10.0), so stage-3 success detection and
the eval reward decomposition are wrong for non-trex species.

### 2.7 Other JAX findings (from the deep-dive, spot-checked)
- **MEDIUM** — `jax_eval.evaluate_policy_cpu` uses unseeded global `np.random`
  for reset noise → non-reproducible gate evaluations. Accept a seed.
- **MEDIUM** — the hook-based `JaxTrainer` rollout (`jax_trainer.py:937-966`)
  never passes `forward_vel_scale`, so TOML `ramp_*` settings are inert on the
  CLI path (the notebook path wires them correctly).
- **LOW** — logged `learning_rate` decays `n_minibatches`× faster than the real
  optimizer schedule (`jax_trainer.py:644-649`); display-only.
- **LOW** — PPO advantage normalization happens per-minibatch
  (`jax_ppo.py:220`) vs SB3's per-batch; acceptable, worth a comment.
- **LOW** — KL early-stop inside `lax.scan` keeps computing (then discarding)
  gradients for the remaining minibatches; wasted compute when KL trips early.
- **QoL** — `decay_running_stats` exists for stage transitions but
  `run_curriculum` never applies it (`obs_rms_decay_on_resume` is honored only
  on the `jax_setup` path — worth unifying).
- Positive: GAE handles truncation-vs-termination correctly (separate
  `gae_done`/`full_done`, `final_obs` bootstrap), obs-normalization
  update-ordering is correct, and the PPO loss (clipping, entropy sign, ratio)
  is self-consistent.

---

## 3. Sweep / evaluation / reporting infrastructure

### 3.1 Quality scoring is silently disabled by a metric-name mismatch — **HIGH** *(verified empirically)*
`configs/quality_scoring.toml` keys (`"fwd_vel_m/s"`, `"distance_m"`,
`"ep_length"`, …) don't match the row keys produced by the training/collection
pipeline (`mean_forward_vel`, `mean_distance_traveled`,
`best_mean_episode_length`, …). Result: `quality_score` / `quality_rank` are
**empty in every shipped `collected_results.csv`** (all four runs). **Fix:** add
an alias table in `scoring.py` (or rename the TOML keys), and add a regression
test that scores one of the shipped CSVs and asserts a non-empty score.

### 3.2 W&B resume produces split runs — **MEDIUM** *(verified)*
`wandb_integration.py:113` uses `reinit=True` and no stable `id`; any
reconnect/resume creates a new run. Use `id=<deterministic>` +
`resume="allow"`.

### 3.3 Smaller items (agent-reported, spot-checked where noted)
- `_handle_stage_failure`'s bare `raise` (`orchestration.py:202`) **is correct**
  (called inside the `except` block) — but `raise exc` would be clearer; the
  preceding `os._exit(1)` skips `atexit`/W&B finalizers — prefer `sys.exit(1)`.
- `orchestration.py` `launch_sweep`: `remaining_trials` is not recomputed after
  partial-trial reconnect recovery (the `launch_all_stages` path does recompute).
- `scoring.compute_quality_scores` sorts the caller's row list in place —
  surprising side effect for `write_results_csv` callers.
- `metrics.py:184` `velocity_consistency` explodes to large negative values when
  mean velocity ≈ 0 (stage 1); harmless today (weight 0 in stage-1 scoring) but
  worth clamping.
- `plot_sweep_results` uses deprecated/racy `tempfile.mktemp`.
- `collect_ray_results` may emit empty `trial_id`s when Ray returns the id as
  the index rather than a column.
- All sweep trials run with the same `seed=42` — per-trial seeds
  (`seed + trial_index`) would decorrelate trials; record the seed per row.
- `DiagnosticsCallback` termination-fraction histories don't NaN-backfill the
  way algo metrics do (`diagnostics.py:246` vs `:354-358`), so
  `term_*` series in `diagnostics.npz` misalign with `term_timesteps` whenever a
  reason skips a rollout *(verified)*.
- `train_curriculum`'s CSV always writes `training_duration_seconds: ""`
  (`train_base.py:1118`) even though the single-stage path measures it.

---

## 4. Configs & results consistency

- **Comment/value mismatches** *(verified)*: all three stage-3 TOMLs say
  "At least 25% … success" next to `min_success_rate = 0.5`; trex stage-2 says
  "Reduce from 0.4" next to `min_avg_forward_vel = 2.0`.
- **Brachio stage-2 `natural_pitch = -0.15`** while stages 1/3 use the default
  0.0 — the nosedive reference frame changes between stages; intentional?
- **Config drift vs shipped results** (expected but worth labeling): brachio
  stage-3 TOML now says 12M steps / threshold 0.8 while
  `results/brachiosaurus/ppo/summary.json` reflects the 8M-step run (30M total,
  matching the README).
- `summary.json` vs `collected_results.csv` best-reward deltas (e.g. trex stage 1
  3008.66 vs 2994.34) — two writers, different eval moments; document which is
  authoritative in `results/README.md`.
- `pyproject.toml` declares both `gymnasium.envs.__root__` and
  `gymnasium.envs.MesozoicLabs` entry-point groups; Gymnasium no longer loads
  plugin entry points, and the envs already self-register at import — likely
  dead config.
- `[all]` extra omits `[mjlab]`; Dockerfile installs `[train,viz]` only (no
  `[jax]`) — fine if intentional, but document that the image is SB3-only.

## 5. Documentation drift (website + docs/)

The website is significantly out of date relative to configs/results:

- `website/docs/training/ppo.md` and `sac.md`: hyperparameter tables and result
  tables reflect an old single-stage run (e.g. LR 3e-4 vs actual 3e-5, 2.6M/3.6M
  steps vs actual 22M curricula).
- `website/docs/models/custom-models.md` reference table: every species row is
  wrong (e.g. T-Rex "14 act / 77 obs" vs actual 21 / 83);
  `brachiosaurus.mdx` says 30 actuators vs actual 26;
  `vertex-ai.md` troubleshooting repeats the stale dims.
- `environments/trex/README.md` curriculum lists 500K/1M/2M steps vs actual
  6M/8M/8M; `environments/velociraptor/README.md` stage-3 table says
  `strike_bonus=+500`, `alive_bonus=0.1` vs actual 1000 / 0.0, and still
  references a Brax migration that became MJX.
- `docs/CODE_REVIEW.md` lists "critical issues" that are already fixed in code (e.g.
  VecNormalize flag restoration now uses try/finally; the helper consolidation
  it requests is done) — stale docs make every future review re-litigate them.
  Mark resolved items or archive the doc.
- `docs/REWARD_SCALE_REDESIGN.md` uses `*_bonus_weight` key names that don't
  exist (`strike_bonus` etc. are the real keys).

**Recommendation:** generate the dims/hyperparameter tables from
`configs/*.toml` + env introspection (a tiny script writing MDX partials) so
they can't drift again, and add a CI check.

## 6. Quality-of-life recommendations (training execution & debugging)

1. **Cut eval overhead.** `EvalCallback` runs 30 episodes × ≤1000 steps on a
   single env every 50k steps (plus the 5-episode supplementary eval, plus 80
   episodes post-stage). On a 6M-step stage that's up to ~3.6M serial eval steps
   against 6M (parallelized) training steps. Use a vectorized eval env (4–8
   envs), or 10–15 episodes per eval; both preserve the gating statistics per
   wall-clock far better.
2. **`train --stage N` should default `--timesteps` from the TOML**
   (`curriculum.timesteps`) instead of a hardcoded 500k — the current default
   silently trains 8–16× fewer steps than the configs intend.
3. **Make `--override` bool-aware** (`cli.py:_cast_value` returns the string
   `"true"`, which is truthy but wrong as a float weight) and reject unknown
   sections/keys loudly — today a typo'd override is applied to a dict nobody
   reads.
4. **Single source of truth for success keys** — `SpeciesConfig.success_keys`
   exists but `CurriculumCallback`/`LocomotionMetrics` hardcode
   `("bite_success", "strike_success", "food_reached")`; raptor's config also
   lists `bite_success` it can never emit.
5. **Persist `training_duration_seconds`** in the curriculum CSV, and write the
   effective config (post-override, post-ramp targets) into `stage_config.json`
   (it currently captures TOML+defaults, which is good — add the CLI overrides).
6. **W&B**: stable run ids + `resume="allow"`; log the run URL into
   `stage_config.json` / sweep state.
7. **JAX**: add `resume_from_checkpoint`, seedable eval, and tighter
   `StabilityMonitor` defaults (`kl_warn=100` only fires after total collapse).
8. **Tests to add** (cheap, high-leverage):
   - TOML→env round-trip: construct each env with each stage's `env_kwargs`
     and assert no unknown/unused keys (catches §1.6, §2.2).
   - JAX/SB3 reward parity: one fixed state, assert per-component rewards match
     within tolerance (catches §2.2/2.3/2.5 classes).
   - Gate logic: `check_stage_gate` against a real loaded TOML (catches §2.1).
   - Scoring: `compute_quality_scores` on a shipped CSV → non-empty scores.
9. **CI**: add a CPU-only JAX job (`pip install ".[jax]"`,
   `JAX_PLATFORMS=cpu`) that runs the `test_jax_*` suite; currently the entire
   JAX stack is untested in CI.
10. **Brachio stage 3** (16.7% vs 50% target): before burning another 12M steps,
    fix §1.4 (gate noise), §1.5 (success-gating consistency), and consider that
    at `gamma=0.995` with `food_reach_bonus=1000` the math is fine, but the
    head-proximity gradient (`head_proximity_max_dist=5.0`, weight 2.0) is
    nearly flat over the last meter — a tighter shaping range (e.g. 2.0 m, like
    the raptor claw) concentrates the gradient where the failure happens
    (last-mile neck positioning).

## 7. Open questions

1. **Is SB3↔JAX reward parity a hard goal?** If yes, §2.2/2.3/2.5/2.6 are the
   priority list and a parity test should gate CI. If the JAX path is a research
   spike, document the known divergences in one table instead.
2. **Were the published sweeps run with identical seeds per trial?** If so,
   ranking confidence is lower than it appears; worth re-running the top-3
   configs with 3 seeds each before locking them into the TOMLs.
3. `docs/CODE_REVIEW.md` / `REFACTORING.md` contain completed-but-open-looking
   items — OK to mark resolved/archive?
