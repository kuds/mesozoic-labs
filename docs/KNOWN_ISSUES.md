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

- **LOW** — T-Rex SB3 env silently accepts `foot_contact_weight` /
  `foot_contact_gate` (JAX-only params) without using them; typo'd weights
  do nothing. Reject unknown env kwargs loudly. (June §1.6)
- **LOW** — `CurriculumCallback` / `LocomotionMetrics` hardcode success keys
  (`bite_success`, `strike_success`, `food_reached`) instead of using
  `SpeciesConfig.success_keys`. (June §6.4)
- **LOW** — `curriculum.py` `_read_latest_eval`: the
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
- **Watch item** — the Velociraptor home-residual action mapping has a
  slope discontinuity at action 0 (the two piecewise segments span
  home→min and home→max, which are unequal), so zero-mean Gaussian
  exploration produces a physically biased mean command away from home
  (hip pitch ≈ −0.29 rad toward flexion, knee ≈ −0.16, ankle ≈ +0.13 at
  σ=1). If a fresh run's early training looks persistently crouched or
  off-home while `algo_std` is still ≈1.0, suspect this bias before
  suspecting rewards; possible mitigations (smaller `log_std_init`, a
  smoothed mapping) are interface experiments and must be run in
  isolation. (Stage-1 basin investigation follow-up)

## Sweeps / infrastructure

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
z≈1.13, level, all four feet grounded. All three models now use
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
§5. Post-fix gait clipping is ≤0.5 % everywhere, pinned by each species'
`tests/test_actuator_bounds.py`; re-measure any re-sizing with
`environments/shared/scripts/actuator_saturation_report.py`.

Neutral-action stability and truly actuator-disabled passive behavior are now
separate test contracts. Layered policy, physics, visual, and source identities
are documented in [PLANT_CONTRACT.md](PLANT_CONTRACT.md).

> **Note:** these changes alter the physics plant. Policies trained before the
> change are incompatible by contract, including when a change seems marginal;
> use an explicit legacy override only for deliberate historical evaluation.

Still open:

- **MEDIUM** — the T-Rex foot touch sensors read 0 during settled stance
  (verified empirically: both `r/l_foot_contact` are 0.0 after a settle),
  the same defect fixed on the velociraptor: a lying toe capsule contacts
  the floor near its ends, and the r=0.06 site at the toe midpoint misses
  those contact points. The two foot-contact observation dims and any
  foot-contact reward gating are dead. Fix as on the raptor: enlarge the
  site to envelop the toe capsule, check for adjacent-digit
  interpenetration (add contact excludes if the digits overlap at rest),
  and bump the trex plant revisions.
- **LOW** — the raptor's toe-clipping margin now sits at the toes: at
  sprint-like excitation (3–4 Hz, full amplitude) the 0.8×kp toe caps clip
  10–16 % and the 1.5×kp hip pitch ~11–16 % (its physical envelope); the
  knee measures 0 % after its 1.5× bump. Re-measure with
  `environments/shared/scripts/actuator_saturation_report.py` before any
  faster-gait (stage-3 sprint) work and consider 1.5× toes if they bind.
- **MEDIUM** — the T-Rex home keyframe (pitch 0) is ~35° from its passive
  equilibrium (settles to forward_z −0.583), which is *past* the stage-1
  nosedive termination line (−0.519 with the TOML's 0.35 threshold): every
  episode opens with a dive the policy must catch, and pure passivity is
  death. Move the passive equilibrium near the intended ~10° natural pitch
  (tail `springref`, neck stiffness, or hip `ref`/keyframe lean).
- **LOW** — scene boilerplate (skybox/grid/floor/option) is copy-pasted
  across the three XMLs → extract a shared `scene.xml` include; limbs are
  hand-mirrored sign-flips → generate via script/PyMJCF or add a left/right
  symmetry test.
- **LOW** — raptor claw `motor gear="50"` on a 0.05 kg claw (huge
  torque-to-inertia; slams its limits); trex passive ball-joint arms are
  8 of 37 DOF doing nothing (weld to cut MJX cost ~20%); contype-0 neck
  geoms can visually clip the floor; no `<light>`/`<visual>` block for
  nicer renders; brachio food has no collision partner.
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
