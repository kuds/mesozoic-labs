# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Reproducible Runs & Velociraptor Stage-1 Diagnosis (v0.3.5)

### Changed
- **The stage-1 reward rails are sized to reject collapse, not to approximate
  competence** (0.60 × each statue's standing reward, superseding
  PLANT_VALIDATION §12's 0.89 ×): trex 2900 → **1950**, velociraptor and
  brachiosaurus 1550 → **1050 / 1040**, dibothrosuchus 2300 → **1560**. §12
  picked 0.89 to sit just below a competent policy, which is competence-bar
  reasoning that §9 refutes — the statue is the reward optimum, so no
  threshold separates competent from passive. Sized instead to the rail's one
  real job: the measured collapse (full-horizon 93% → 7%) bottomed at
  **888 = 0.27 × statue**, so 0.60 clears it by better than 2×, whereas 0.89 =
  2900 sat within ~2.4% of a competent policy's estimated ceiling (~2970 =
  statue − the measured 0.30/step smoothness cost, before energy and the
  posture terms a moving policy gives up) and risked rejecting the very policy
  it was meant to admit. `collapse_peak_floor` stays at 0.75 ×, which *should*
  sit near a good level since its job is arming a detector rather than
  admitting a policy.
- **`min_avg_episode_length` is now enforced on the JAX path.**
  `check_stage_gate` read only `min_avg_reward`, and the trainer's
  `eval_metrics` carried no length key at all — so the gate kind named
  `reward_and_length/v1` was fully enforced on SB3 and half-enforced on JAX,
  the exact "one backend silently ignores a gate the other enforces"
  divergence `gate_schema` exists to prevent. It became load-bearing when
  stage 1 began encoding its full-horizon ≥ 95% floor in that field. The
  trainer now tracks `episode_length_history` and emits `mean_episode_length`
  (NaN windows collapse to 0.0, which fails a length gate rather than passing
  it), and the gate enforces both halves; declaring the threshold without the
  metric raises rather than passing on reward alone.
- **The ground settle no longer probes geoms that cannot touch the ground.**
  Both backends included `contype=0, conaffinity=0` geoms — 2 on T-Rex, 5 on
  brachiosaurus, **14 of dibothrosuchus's 39** — when deciding where the floor
  is. Settling to one would hold the animal's real feet above the ground,
  which is the hover the settle exists to prevent. Measured latent (no phantom
  is currently the lowest geom at any shipped noise), so the filter is
  behaviour-preserving: resets are verified **bit-identical across all four
  species** after the change. A new invariant test pins that only collidable
  geometry can drive the settle.
- **Stage 1 moved to the 1a operating point and the reward gates were
  re-founded as sanity rails** (PLANT_VALIDATION §7/§9/§12; changes what
  stage 1 *is*, per decision §17.2). Every species' stage-1
  `reset_noise_scale` is now **0.05** (T-Rex was 0.10, dibothrosuchus 0.30):
  the noise sweep on the repaired plant showed reset noise does not make
  standing harder (statue standing reward nearly flat, 3288 → 3167 across
  0.01–0.20 on T-Rex) — it decides how often you get to stand at all
  (full-horizon 100% at ≤ 0.05, 65% at 0.10, 22% at 0.20). At the old
  operating points survival and stance quality were inseparable in every
  metric, which is why no scalar gate ever said anything useful; at 0.05 a
  statue survives 100% and stance-quality gates can measure stance quality.
  Robustness becomes stage 1b's job via declared `xfrc_applied`
  perturbations, where a lucky draw and a good controller are
  distinguishable — not reset noise. The four `min_avg_reward` values (all
  previously cleared by their own species' statue, certifying nothing) are
  now §12 sanity **rails** at 0.89 × each statue's standing reward at 0.05:
  trex 2900 (statue 3271.8 ± 12.0), velociraptor 1550 (1745.8 ± 5.0),
  brachiosaurus 1550 (1739.1 ± 1.2, on the repaired plant), dibothrosuchus
  2300 (2598.3 ± 0.9). A rail sits *below* the statue on purpose — above it
  is unreachable, since the statue is the reward optimum — and only rejects
  a policy that discarded most of the available return; `min_avg_episode_length`
  rises 750 → **950** on all four to encode §12's full-horizon ≥ 95% floor,
  and `collapse_peak_floor` moves to 0.75 × each statue's standing reward
  (still absolute; relative floors are §14 item 3 and wait for the fresh
  run). All values are provisional per §12 ("for review, not adopted") and
  the real 1a gate — the episode-level `stance_success` event over
  unsupported duty — remains unbuilt (tracked in KNOWN_ISSUES).
- **T-Rex home stance corrected to a flexed theropod limb** (breaking — plant
  change, physics revision 4 → 5 **and** policy interface revision 6 → 7; all
  existing T-Rex checkpoints are invalidated): the home keyframe stood the
  animal on a 172.1° interior knee, 7.9° from full extension and inside the
  singular region for leg-length control. Hip-to-ankle distance changed at
  only 0.024 m/rad there, so servicing stage 1's live height term
  (`height_weight = 1.0`) cost 23.7° of knee travel per centimetre of pelvis
  height — geometry alone accounting for the 31°-per-step knee excursions that
  three `smoothness_weight` escalations (0.1 → 0.7 → 2.0) reduced but could
  not remove. The keyframe now flexes to a **135.0°** interior knee: 45° of
  flexion from a fully columnar limb, the figure Hutchinson, Anderson, Blemker
  & Delp (2005), "Analysis of hindlimb muscle moment arms in *Tyrannosaurus
  rex* using a three-dimensional musculoskeletal computer model",
  *Paleobiology* 31(4):676–701, use for *T. rex* in Fig. 8, converted by their
  Table 1 rule ("for the knee subtract the angle here from 180°"). Leg-length
  authority rises to 0.134 m/rad, i.e. **4.3° of knee per centimetre**. Femur
  and tibia sit symmetrically 22.5° either side of vertical, the one
  inclination that holds the hip over the ankle at this plant's tibia:femur of
  1.000, so the CoM still sits 33.8% of the way heel-to-toe through the
  support polygon against 34.3% before. The three rotations sum to zero, so
  the metatarsus keeps its 21.8° digitigrade slope and the plantar pads and
  digits keep their 0.5 mm contact; segment vectors, masses and inertias are
  untouched. Leg `ctrlrange`s were re-centred on the new pose at unchanged
  width, preserving the `home-keyframe-residual/v1` invariant that the range
  midpoint *is* the home control — which incidentally cuts reachable knee
  hyperextension from 40.2° to 5.0°. Every dependent constant was re-measured
  on the new plant rather than adjusted by eye: `target_z` and
  `target_standing_z` 0.9757 → 0.9260, `natural_pitch` 0.05 → 0.027,
  `nosedive_termination_threshold` 0.47 → 0.493 (holding the absolute −0.520
  forward_z envelope), `healthy_z_range` (0.75, 1.6) → (0.70, 1.55), and
  `min_avg_reward` 1900 → 1840 from a re-run zero-action baseline of
  1743.73 ± 1275.54 (was 1800.56 ± 1267.66). `visual_revision` is deliberately
  unchanged: that layer fingerprints body-local geom/site/material/camera
  definitions, which a pose edit does not touch. See
  `docs/TREX_LEG_FLEXING_PLAN.md`.

- **Velociraptor knee actuators raised to 1.5×kp** (breaking — plant change,
  physics revision 1 → 2): the knee measured 0 % clip at the moderate
  2.5 Hz/0.8-amplitude contract gait but 30–46 % at sprint-like 3–4 Hz
  full-amplitude excitation while still capped at 0.8×kp (`forcerange`
  ±145 on kp=180) — the same clipped-torque signature that collapsed
  stage-2 locomotion at the hips. At ±270 the same sprint excitation
  measures 0 % knee clip; the moderate-regime contract and the
  home-control no-saturation guarantee are unchanged, and a new
  sprint-regime contract test pins the knee headroom.

- **Eval-collapse early-stop now uses a robust peak, raw-eval patience, a
  gate floor, and looser stage-1 settings**: the backstop takes the maximum
  full-window rolling median (default 5 evals,
  `collapse_smoothing_window`) of the per-evaluation mean rewards as its
  reference peak, then counts each newly arrived raw per-evaluation mean
  below `(1 − drop_fraction)` of that peak once toward `collapse_patience`.
  Detection waits for both `collapse_min_evals` and a full median window,
  and only arms once the robust peak clears the stage's `min_avg_reward`
  gate (overridable via `collapse_peak_floor`). Separating the peak
  estimator from patience rejects noise without multiplying it across
  overlapping windows: a variance-inflated early evaluation (run
  `20260720_203454`'s 50k eval, 261.79 ± 261.72) cannot dominate the median
  peak, while one low evaluation contributes exactly one patience strike.
  A healthy but bimodal gait transition is judged by its per-evaluation
  mean rather than `mean − std`, so intermittent healthy evaluations reset
  patience instead of reading as a sustained collapse.
  All three stage-1 TOMLs also gain the explicit
  `collapse_min_evals = 20` / `collapse_patience = 10` /
  `collapse_drop_fraction = 0.5` overrides stage 2 already had.
  (This supersedes the intermediate `mean − std` peak/current rule shipped
  earlier in this release, which correctly fixed the stage-1 abort but then
  aborted run `20260722_124556`'s stage 2 at 1.75M — 200k steps before it
  recovered to 2707 — because a bimodal transition deflates `mean − std` on
  the current side. The regression is pinned by tests built from that run's
  trace and the identical recovering trace `20260721_141523`.)

- **Velociraptor stage-3 budget raised 8M → 12M timesteps**: in run
  `20260721_141523` the best strike checkpoint landed at 7.9M of 8.01M
  with success still climbing (last evals 0.63 → 0.87 → 0.77); strike
  discovery produced no successes until ~5.5M, leaving only ~2.5M of
  effective hunting curriculum. 12M matches the brachiosaurus stage-3
  budget.

- **The four largest `environments/shared` modules are now packages**, each
  split along the seams its own docstring already described. Public surfaces
  are unchanged — every package `__init__` re-exports exactly what the module
  exported, so no import site outside the packages moved:
  - `reporting.py` (1,929 lines) → `formatting`, `gates`, `csv_output`,
    `text_summaries`, `summaries`, `bundles`, `stage_artifacts`
  - `plant_contract.py` (1,822) → one module per contract layer
    (`source_layer`, `policy_layer`, `physics_layer`, `visual_layer`) over
    `constants`/`errors`/`digests`/`identity`/`versions`/`introspection`, with
    `manifest`, `validation`, and a `__main__` for the existing
    `python -m environments.shared.plant_contract --check` entry point
  - `curriculum.py` (1,316) → `sb3_compat`, `manager`, `schedules`,
    `advancement`, `early_stopping`, `checkpoints`; the manager no longer
    shares a module with the SB3 integration
  - `result_bundle.py` (1,201) → `constants`, `errors`, `naming`, `hashing`,
    `provenance`, `manifest`, `evidence`, `audit`

  Three test patch points moved with the code they belong to: the plant
  contract's `configs/` paths are read through its `constants` module,
  `_SB3_AVAILABLE` through `curriculum.sb3_compat`, and the provenance helpers
  through `result_bundle.provenance`. Each is now a single patch point rather
  than one per consuming module.

- **Fixed: default provenance capture resolved the repository root one level
  too shallow.** `initialize_result_bundle` derived it as
  `Path(__file__).resolve().parents[2]`, correct while the code lived in
  `environments/shared/result_bundle.py` but off by one once it moved into
  `result_bundle/provenance.py`, where it resolved to `<repo>/environments`.
  `git status` reports repo-wide while `git ls-files --others` is scoped to its
  working directory, so an untracked file under `configs/` or the repository
  root set `repository_dirty` without entering `repository_patch_sha256` —
  materially different dirty trees could share one provenance hash. The root is
  now `REPOSITORY_ROOT` in `result_bundle/constants.py`, derived from a named
  `_SHARED_ROOT` anchor rather than a bare parent count, read through the
  `constants` module so there is one authoritative binding, and pinned by three
  tests — one on the constant, one on the initializer's default wiring, and one
  asserting a root-level untracked file changes the patch hash. Note the expression was byte-identical across the move, so an AST-level
  verbatim check could not see it; only evaluating it could.

- **The per-species script harnesses moved to `environments/shared/harnesses/`**
  and no longer use `test_` names: `test_env_base.py` → `harnesses/env_smoke.py`,
  `test_actuators_base.py` → `harnesses/actuators.py`, `view_model_base.py` →
  `harnesses/viewer.py`, with `test_actuators()` → `run_actuator_test()` and the
  internal `test_*` checks → `check_*`. These are hand-run smoke checks and
  viewers, but pytest collected them as tests — and errored on their unfillable
  `env_class`/`cfg` arguments — for any invocation that reached them, which the
  repository's `testpaths`/`norecursedirs` settings were the only thing
  preventing. The coverage `omit` entry added to work around the old naming is
  replaced by one covering the package.

- **CI now measures the coverage it actually produces.** Three jobs ran tests
  without `--cov` — `test-jax-cpu`, `test-sb3` and `plant-contract` — and those
  are precisely the jobs carrying the optional dependencies. The only job
  reporting a number for `environments/shared` was `test-shared`, which installs
  neither JAX nor SB3, so the modules needing them were omitted from the report
  as "not available in the standard CI test environment". That stopped being
  true when `test-jax-cpu` was added: 1,840 statements of JAX/MJX and 495 of
  `train_base.py` — 23% of the production code in `environments/shared` — were
  being exercised by CI and excluded from its number, which read 79.57% over the
  remaining 7,652 statements.

  Deleting the omits alone does not work; it drops `test-shared` to 66%, under
  the `fail_under = 70` gate, because that job genuinely cannot import those
  modules. So every test job now writes its own `.coverage.<job>` and uploads
  it, and a new `coverage` job combines them and gates the union. Per-job
  reports are suppressed with `--cov-fail-under=0`, since no single job's slice
  is meaningful alone. `relative_files` is enabled so the combine lines up
  across jobs. `harnesses/` and the mjlab adapter stay omitted, now with
  accurate reasons — hand-run, and GPU-only with no job able to reach it.

  The combined gate reports **80% over 11,071 statements**. On the comparable
  `environments/shared` slice that is 78.07% over 9,987 statements against the
  old 79.57% over 7,652 — the figure barely moves while the denominator grows
  31%, which was the point. `fail_under` stays at 70 rather than being tuned to
  the result. The combine log reads `Combined 15 files, skipped 3` against 18
  artifacts: coverage.py content-hashes each data file and skips one whose
  bytes match a file already combined, so three matrix runs that produced
  identical data to a sibling Python version contributed nothing new. No data
  is dropped.

### Removed
- **The top-level `Images/` directory** (25 MB): its three GIFs were
  byte-identical to the copies under `website/static/img/`, which are the ones
  the site and README actually reference. Nothing in the repository referred to
  `Images/` except the `.dockerignore` entry excluding it, now also dropped.

### Added
- **`stance_duty_validation.py`, which shows the support-duty metrics measure
  what they claim.** `unsupported_duty` and its siblings classify each step by
  thresholding the foot touch sensors at 0.1 N, and that reading is the evidence
  behind the claim that the stage-1 policy is off the ground ~21% of the time —
  a sensor that under-reports contact looks exactly like a foot in flight. The
  static check above covers one operating point; this sweeps driver policies
  producing **3% to 67%** airtime and compares the sensors against both
  `mj_contactForce` over foot-geom contacts and `mj_geomDistance` from every
  foot geom to the floor, the latter computed from geometry alone and so
  independent of the sensor path. The metric tracks kinematic truth to within
  **0.52%** of steps across the whole range (2.6% only under uniform-random
  actuation, which no policy occupies), and the two error directions nearly
  cancel. The decisive row is low-amplitude jitter — **16.07% true airtime
  against 16.21% reported**, straddling the figure under dispute. Combined with
  the 0.1 N threshold sitting against ~421 N per foot in quiet stance, the
  sensor-artifact explanation for `STAGE1_SPLIT_PLAN.md` §1.5 is exhausted and
  its hopping reading can be relied on. This validates the *instrument*, not the
  policy: the checkpoint was not replayed, and the causal half of §1.5 ("the
  reward caused the hop") still needs a counterfactual run.
- **`foot_sensor_report.py`, and a four-species audit of what the foot touch
  sensors actually measure.** A MuJoCo touch sensor sums only contacts on geoms
  belonging to its site's own body, so a site on a parent segment silently
  misses whatever the child geoms carry — a failure invisible from the reward
  trace, because a sensor reading zero looks exactly like a foot off the ground.
  Checking every species against `mj_contactForce` found T-Rex and dibothrosuchus
  correct (ratio **1.000**) and two species wrong: **velociraptor at 0.553**,
  missing its `metatarsus` (17.54 N) and lateral `toe_d4` (12.03 N) per foot
  because the site sits on `toe_d3` alone, and **brachiosaurus at 0.000**, blind
  to all 1699.2 N of its floor contact. Total floor reaction equals body weight
  on all four, so the contacts are real and these are sensor-scope defects, not
  physics ones. Neither reaches a stage-1 reward term today, but both reach the
  **observation**: brachiosaurus trains with four permanently zero input
  channels (indices 75–78 of 83) and the raptor's policy sees 55% of true
  per-foot load. Repairs are per-species MJCF changes and are *not* included
  here. Recorded in `docs/investigations/FOOT_SENSOR_VERIFICATION.md`, which
  also corrects the ground-reaction-force reading in `STAGE1_SPLIT_PLAN.md` §6.2:
  the T-Rex statue's 841 N is *exactly* body weight (840.9 N, ratio 1.002), and
  the 1483 N that makes it look low is `mj_getTotalmass` counting the 65.45 kg
  **prey body** — 43% of the model total. A GRF diagnostic must divide by the
  animal's kinematic subtree or it reports a false 0.57 on a plant standing in
  perfect equilibrium.

### Fixed
- **The stage-1 reward now charges the two things run `20260801_021545`
  proved it could not see** (PLANT_VALIDATION §14 items 1, 2 and 4). That run
  — T-Rex, 6.0M steps on the repaired plant at the 1a operating point — ended
  at **28.4% unsupported duty** against the statue's 0.000, versus 30–35% on
  the broken plant. §16's headline question ("was the airborne duty learned
  from the broken reset?") is therefore answered **no**: the plant repair
  moved it ~19% relative, so the chatter is a reward-design problem. Three
  changes follow:
  - `foot_load_balance_min_support_force` is **42.0** on T-Rex stage 1, 5% of
    the animal's own weight (85.72 kg excluding the prey prop = 840.9 N). The
    previous `0.0` was a *measured* no-op — two touch readings essentially
    never sum to exactly zero, so the airborne branch never fired once across
    709 logged rollouts. It now sits well above the duty metrics' 0.1 N/foot,
    so reward and diagnostics agree about what "unsupported" means.
  - A new `foot_load_balance_airborne_penalty` (0.3) makes the ordering
    **strictly monotone**: `both feet even +0.600 > single support −0.300 >
    airborne −0.600`. Previously airborne and single support both scored
    −0.300 — a flat region with no gradient out of the air, on the stage whose
    whole job is staying on the ground.
  - A new frequency-aware `action_jerk_weight` (1.0) penalises the **second**
    difference of actions. `smoothness_weight` charges first-difference
    magnitude and is blind to frequency: from the best to the final checkpoint
    of run `20260731_132102`, `action_delta` *fell* 12.0 → 10.5 and its penalty
    *improved* while toe-motion power above 4 Hz doubled. The new term inverts
    that — a slow ramp scores jerk 0.00 (though it has the *higher*
    `action_delta`) while a Nyquist-rate buzz scores 336.
  - `derive_stance_info` keys its airborne branch to the same
    `> contact_threshold` test the duty metrics use, so a foot at 0.001 N is
    unsupported in both instead of supported in one. It previously reported
    *perfect balance* for a truly airborne pair, and never fired anyway.
  All wired through the Gymnasium, MJX and JAX paths with matching parameters;
  every new weight defaults to `0.0`, so untouched species and stages are
  numerically unchanged.
- **`collapse_peak_floor` can be relative, because the absolute one failed to
  arm a second time.** Set to 2450 (0.75 × the statue) it sat above run
  `20260801_021545`'s best evaluation of **2347.67**, so the detector never
  armed and watched eval degrade 2347.67 → 1666.33 — the identical failure to
  the 2200-vs-1934.1 case in §11.4. Deriving the number from the statue was
  only half a fix: the statue bounds what is *achievable*, not what a
  *learning* policy passes through. `collapse_peak_floor_fraction` ×
  `collapse_peak_floor_reference` (0.45 × 3271.8 = 1472 on T-Rex stage 1)
  re-anchors whenever the reward changes; the failing run cleared it by ~2M
  steps while it still sits well above the 888 collapse bottom. An explicit
  `collapse_peak_floor` still takes precedence, so existing configs are
  unchanged, and a half-declared pair resolves to "never arm" rather than to a
  silently low floor.
- **Brachiosaurus stage 1 is a balance task again** (breaking — plant change,
  physics revision 2 → 4, policy interface revision 4 → 6, visual revision
  1 → 2; all existing brachiosaurus checkpoints are invalidated). The
  zero-action statue fell on 40 of 40 episodes at mean length 130.7, so no
  stage-1 result was interpretable (PLANT_VALIDATION §6, KNOWN_ISSUES). The
  collapse decomposed into two measured defects. First, BrachioEnv was the
  only species still on the inherited midpoint action mapping, and its home
  stance is not the ctrlrange midpoint: "do nothing" dragged the rear knees
  0.349 rad, the front knees 0.262 rad and all four ankles 0.175 rad away
  from the standing pose on every step; commanding the actual home controls
  tripled statue survival (mean length 141 → 455) on its own. Both backends
  now use `home-keyframe-residual/v1` like the other three species — on MJX
  this also moves the reset base pose from `qpos0`, which hovered 610 mm
  above the floor, to the home keyframe. Second, the leg position servos
  could not statically carry the animal: it sagged 71.6 mm under its own
  weight, putting the planted stance's lateral (roll) stiffness (~1.9·10³
  N·m/rad) at parity with the destabilising gravity stiffness m·g·h
  (~1.77·10³), so the statue tipped over in slow roll on 9 of 10 resets even
  when commanding the exact home pose — the same-signed slow roll to
  `fallen` visible in every trace. Leg kp is doubled (sag 11.1 mm), with
  forceranges scaled so each actuator keeps its documented
  force-to-stiffness sizing. Zero action, 40 episodes at stage-1 noise 0.05,
  before → after: reward 163.35 ± 81.40 → **1739.08 ± 1.17**, full-horizon
  **0/40 → 40/40**, terminations (fallen 34, tail_contact 6) → (truncated
  40). The ± 1.17 spread is the tightest of any species, making
  brachiosaurus the best-conditioned gate-prototyping plant — the property
  PLANT_VALIDATION §6 wanted. The full-horizon neutral-action test the
  KNOWN_ISSUES entry prescribed now exists on the brachiosaurus static
  balance suite.
- **Brachiosaurus can feel its feet.** All four foot touch sensors read
  exactly 0.0 N while `mj_contactForce` reported 98.8% of body weight over
  four real floor contacts (FOOT_SENSOR_VERIFICATION §4) — the policy
  trained with four permanently dead input dimensions, and the support-duty
  diagnostics scored the standing statue as 88.5% airborne (PLANT_VALIDATION
  §8). Two scope defects: the pad sites were r=0.03 spheres that did not
  contain the pad's floor contact points, and the metacarpus/metatarsus
  carries part of the load on the pad's *parent* body, invisible to any site
  on the pad body. The pad sites are now boxes enclosing the pad's contact
  zone, four meta touch sensors are appended at sensordata 26–29 so existing
  indices keep their positions, and both backends sum pad + meta per leg —
  the same repair shape as the T-Rex digit sensors (`aa87445`). Verified:
  summed sensors report **100.0%** of the independently measured
  `mj_contactForce` floor total in settled stance (fr/fl 313 N, rr/rl
  535 N), and statue stance quality at noise 0.05 moved from all-feet 0.000
  / unsupported 0.885 to **all-feet 0.998 / unsupported 0.000** — in line
  with the other three species. On attribution: with the stance repair above
  in place, the **pad-site enlargement alone** accounts for that 100.0% —
  the meta capsules no longer reach the floor and the meta sensors read
  0.0 N at home. They were carrying real load on the pre-repair sagging
  plant and remain correct cover for poses that put the meta back down, but
  nothing should read them as load-bearing today. Two gaps stay open:
  velociraptor's 45% under-read (same doc, §3), and the fact that only pads
  and metas are instrumented — a brachiosaurus kneeling on its shins
  (measured: shins carrying the full 1695 N with the feet clear) reads zero
  on every foot sensor, so support-duty scores it as airborne. That errs
  safely for a stance gate, but distinguishing "kneeling" from "airborne"
  would need shin instrumentation. With this and the stance repair, every species' §8
  stance-quality row is now interpretable, unblocking the §12 brachiosaurus
  gate row.
- **The MJX reset now settles the animal on the ground, closing the plant gap
  the PR #479 review found.** The `ca56f6c` ground settle landed only in
  `BaseDinoEnv.reset`; the MJX reset kept applying joint and yaw jitter with
  the root height fixed at the keyframe value — the same defect class, measured
  on T-Rex at the stage-1 noise (0.10) as spawns from **41.2 mm inside the
  floor to 5.2 mm above it**. The midpoint action mapping was worse: its reset
  base pose is `qpos0`, which on brachiosaurus hovers **610 mm** over the
  floor, so every MJX episode opened with a half-metre free fall. MJX has no
  `mj_geomDistance`, so the settle probes an analytic support extent per geom
  — exact for the sphere/capsule/cylinder/box/ellipsoid primitives the species
  use over a horizontal plane floor, and verified against `mj_geomDistance` to
  ~1e-9 across random poses on all four species. The settle target is the home
  keyframe's authored clearance (the `BaseDinoEnv.home_ground_clearance`
  semantics), probed once at construction through the reset's own kinematics
  path, so a noise-free home-residual reset computes a shift of exactly `0.0`
  and stays bit-identical to the pre-settle behaviour; only jittered and
  midpoint spawns move. Jittered spawns now land within 0.4 µm of the authored
  contact, deterministically in the PRNG key, with no extra RNG consumed. A
  new `test_mjx_reset_plant_invariants.py` suite pins penetration, hover,
  determinism, the bit-exact no-op, the brachiosaurus hover fix, and
  cross-backend agreement of the settle target — every assertion cross-checked
  with CPU `mj_geomDistance` rather than the implementation's own formula.
  Cost: one extra `mjx.kinematics` pass (the cheap position stage, no
  collision or constraint work) per reset, which the fused auto-reset step
  computes every step; the final full forward still runs once, on the settled
  pose. No plant revision bump: entry 6's bump already declared reset placement moved
  for every species, and no MJX checkpoint or baseline was produced between
  the two landings (addendum recorded in `configs/plant_versions.toml`).
- **A declared reward gate must now carry its reward threshold.** The gate
  schema only rejected *misplaced* threshold keys, so a config declaring
  `gate_kind = "reward_and_length/v1"` with no thresholds at all passed
  validation, produced no `threshold_fields`, and fell through to
  `StageThreshold`'s permissive defaults (`min_avg_reward = -inf`) on the SB3
  path — advancing on any evaluation — while the JAX path raised on the same
  config. That is precisely the backend divergence the schema exists to
  prevent. Each gate kind now declares required threshold fields
  (`reward_and_length/v1` requires `min_avg_reward`), enforced in
  `validate_gate_config` whenever the kind is declared, so both backends
  reject the shape identically; the JAX path's now-redundant local check is
  removed.
- **`BaseDinoEnv.lowest_ground_clearance` honours its `data` argument.** The
  parameter was accepted and silently ignored — the probe always measured
  `self.data`, and `home_ground_clearance` had to swap `self.data` out to use
  a scratch buffer. A caller passing a scratch pose got the live pose's
  clearance with no error. The probe now reads the passed `MjData` (pinned by
  a test probing the home pose shifted 0.1 m down), the swap hack is gone, and
  the noise-free reset state is verified bit-identical across the change.
- **The reset's root-height jitter is now documented and pinned as
  state-inert.** The ground settle overwrites the root height as a pure
  function of the sampled joint pose, which silently made
  `reset_height_noise_scale` and PR #478's `_bounded_reset_height_delta`
  dead — verified to one ULP with RNG streams identical across height scales.
  Behaviour is unchanged: the draw is deliberately kept so seeded resets and
  the freshly re-measured baselines stay anchored, the inertness is now stated
  at every definition site, `TestHeightJitterIsInertSinceGroundSettling`
  pins both the inertness and the stream alignment, and a KNOWN_ISSUES entry
  schedules the channel's removal for the next policy-interface revision.
- **Being airborne is no longer cheaper than honest single support.**
  `reward_foot_load_balance` computed `|R − L| / (R + L + 1e-8)`, which
  evaluates to **zero** when both feet read zero — so a plant off the ground
  scored the same as one standing evenly on both feet, and strictly better than
  one carrying its weight on a single foot, which pays the full penalty. On the
  stage-1 weights that ordering was `both feet down +0.600 > airborne 0.000 >
  single support −0.300`: flight was the second-best available state on the
  stage whose entire job is to stand still, and a policy off the ground cannot
  reject a disturbance at all. An unsupported pair now reports maximal
  imbalance, giving `both feet down +0.600 > single support = airborne −0.300`.
  A new optional `foot_load_balance_min_support_force` sets the total force
  below which the pair counts as unsupported; it defaults to `0.0`, which closes
  only the exact `[0, 0]` case and leaves **every loaded state numerically
  unchanged**, and can be raised during gate calibration to also deny credit for
  a token grazing contact. Wired identically through the Gymnasium, MJX and
  JAX paths, with the `[0, 0]` case now covered in both `test_reward_functions`
  and `test_trex_mjx_reward_parity` — neither covered it before, which is how
  the hole survived. Note the two failure states are now *tied* rather than
  airborne being strictly worst; separating them needs a term this fix does not
  add.
- **The curriculum advancement gate now fails closed** (breaking — every stage
  config must declare `gate_schema_version` and `gate_kind`). The gate plumbing
  failed *open* in three independent ways, so a stage could advance on evidence
  nobody had checked. `thresholds_from_configs` copied six known keys out of
  `[curriculum]` and **silently discarded everything else**, so a config
  carrying only new-style gate fields produced no thresholds at all and SB3 fell
  back to `StageThreshold`'s permissive defaults (`min_avg_reward = -inf`,
  length and success floors `0`) — which advance on any evaluation whatsoever.
  `jax_curriculum.check_stage_gate` logged a warning and returned `True` when
  `min_avg_reward` was absent. And neither backend rejected an unrecognised key,
  so a misspelled threshold name *disabled* that threshold instead of failing.
  Together these made "no gate" indistinguishable from "gate satisfied", with
  the permissive reading always winning — which is the wrong default for the
  mechanism that decides whether a policy is good enough to build the next stage
  on. A new `environments/shared/curriculum/gate_schema.py` makes the
  declaration explicit and versioned: unknown keys, unknown gate kinds and
  unsupported schema versions are **fatal** whenever advancement is enabled, and
  a threshold field its declared kind does not consume is fatal too, since it
  implies a gate that is not actually enforced. Running without a gate is still
  possible, but only by declaring `gate_kind = "none/v1"`, which is recorded in
  the config and *refuses* to advance rather than passing by default. All twelve
  stage configs declare `reward_and_length/v1`; the existing tests that asserted
  the permissive behaviour were updated in the same change, as they were the
  reason the defect survived. Effective thresholds for all four species are
  unchanged.
- **`collapse_peak_floor` no longer inherits `min_avg_reward`**, which coupled
  an early-stop backstop to an unrelated advancement threshold. The builder
  chained `collapse_peak_floor` → `min_avg_reward` → `0.0`, and only
  `configs/trex/stage1_balance.toml` set the key explicitly (1 of 12), so
  removing a stage's reward gate — which a state-capability gate would do —
  silently dropped its arming floor to `0.0` and armed collapse detection after
  *any* positive robust peak. A missing floor now means **never arm**, because a
  backstop that is not configured should not abort a run; the eleven configs
  that were relying on the fallback now set the value it produced, so every
  effective floor is bit-identical (`100.0` everywhere except T-Rex stage 1's
  `2200.0`). `collapse_smoothing_window` was also readable but undeclared, and
  is now part of the schema.
- **Reset can no longer generate an already-terminal episode** (breaking —
  policy interface revision bumps for the three home-keyframe-residual species:
  velociraptor 6 → 7, T-Rex 7 → 8, dibothrosuchus 3 → 4; brachiosaurus does not
  carry `home_reset` and is unchanged). The root-height jitter was the **only
  unbounded term in the reset** — every other one is a bounded uniform — so
  `BaseDinoEnv.reset` drew `normal(0, height_scale)` with nothing stopping it
  from placing the root outside `healthy_z_range` before the policy acted.
  T-Rex is the species it actually broke: a 0.926 m home pelvis sits 0.226 m
  above the 0.70 m floor, which at σ = 0.10 m is only **2.26σ**, so a predicted
  **1.19%** of spawns started sub-floor. Measured over seeds 3042–5041, 18/2000
  spawned below the floor and **16 of those terminated on step 1 whatever the
  policy did**; a wider scan of 3042–7041 found 43/4000. The draw is now
  truncated to the distance to the nearer end of `healthy_z_range` less a 0.02 m
  margin, which takes sub-floor spawns to **0/4000**. The bound is symmetric, so
  the mean spawn height is unchanged (0.9262 → 0.9261 m over 2000 seeds) while
  the standard deviation tightens 0.1007 → 0.0980. It binds at 2.07σ for T-Rex
  (3.9% of draws) against 3.6–3.8σ for velociraptor and dibothrosuchus (~0.02%),
  so those two move only in the far tail. Dibothrosuchus hit this same defect
  first and fixed it by decoupling `reset_height_noise_scale`; T-Rex was left
  coupled and nobody re-checked it.

  **This does not meaningfully move the zero-action baseline, and that is the
  point.** Re-measured on seeds 3042–3081 the statue is unchanged where it
  matters — still 23/40 full-horizon, `reward standing` still 3244.04 ± 23.55,
  unconditional mean 1971.57 → 1976.62 — because a statue fails those seeds
  anyway. What the defect capped was the **competent** policy: an episode that
  ends on step 1 regardless of the action is not a policy failure, and counting
  it as one put an unreachable ceiling on any reliability gate. Seed 3077 is
  precisely the first failure in the 39/40 evaluation panels that
  `docs/STAGE1_SPLIT_PLAN.md` §2.3.1 reports, and it is this bug rather than the
  policy. Existing checkpoints for the three bumped species were trained against
  a different reset distribution and must be re-baselined before any stage-1
  gate is calibrated against them.
- **T-Rex stages 2 and 3 no longer terminate at a different pitch angle on the
  MJX path than on the Gymnasium one**: `nosedive_termination_threshold` is the
  per-stage tunable pitch gate, and only `stage1_balance.toml` sets it. When a
  stage leaves it unset the two backends reached for different fallbacks —
  `TRexEnv.__init__`'s **0.62** on the Gymnasium path against the generic
  `weights.get("nosedive_termination_threshold", 0.5)` literal in `mjx_env` on
  the MJX one. With `natural_forward_z = -sin(0.027)` that is a termination at
  40.3° nose-down against 31.8°: MJX killed the episode **8.5° earlier**, in
  exactly the two stages whose configs deliberately ask for a head-forward
  running posture (`nosedive_weight` 1.5 → 0.5 → 0.2, `posture_weight` 1.5 →
  0.2 → 0.1). The comment at `environments/trex/mjx_config.py` asserted the two
  paths already agreed on 0.62, and the parity class that pins this category of
  parameter (`TestSB3MJXEnvelopeParity`, added for `healthy_z_range` and
  `max_tilt_angle`) did not cover it. The T-Rex registry now carries the value,
  making the generic fallback unreachable; registry weights are merged before
  the TOML overlay, so stage 1's calibrated 0.493 still wins. No plant
  fingerprint moves — reward and termination code are deliberately outside the
  policy-interface payload — so no revision bump is required and existing
  checkpoints stay valid. Found during the July 2026 T-Rex review; see
  `docs/investigations/TREX_REVIEW_2026_07.md` finding F2. The same shared 0.5
  fallback exists for the other three species, whose Gymnasium defaults also
  differ (Dibothrosuchus 0.55); those were out of the review's scope and are
  left for a general fix.

- **T-Rex neutral stance now has a viable training basin** (breaking —
  policy-interface revision 1 → 2, physics revision 1 → 2, visual revision
  1 → 2): the old `home` state embedded the feet as much as 9 cm into the
  floor, loaded the rear metatarsus behind the whole-body COM, made each leg
  spring pull toward numeric zero instead of the stance, and used raptor-scale
  servo gains on an 85 kg plant. A shallow plantar pad now supplies a real
  support surface; the reset is a 0.5 mm loaded contact; every leg spring
  references the stance; lower-body joint references preserve a small ankle
  gravity preload; and the hip/knee/ankle gains use mass scaling plus a tested
  stance margin while retaining bounded 1.5×kp headroom. The complete policy
  now uses the named-home residual mapping in both Gymnasium and MJX, so action
  zero commands all 21 home controls while ±1 still reach the actuator limits.
  The load-bearing plantar pads also own hidden box-shaped touch volumes:
  settled readings are ~419 N per foot, airborne readings are exactly zero,
  and four contact exclusions remove the former 28 mm adjacent-toe overlaps
  without changing sensor order or observation dimensions.
  The JAX factory also leaves the CPU evaluation model's XML solver and
  parent-contact options unchanged, eliminating evaluation-only plantar/toe
  self-contact and keeping checkpoint gates aligned with MJX training.
  The old plant nosedived at step ~77 even under its XML home command. The
  corrected plant survives all 1,000 Stage-1 steps in 50/50 noise-free probes,
  50/50 JAX-style probes at 0.05 rad joint noise, and 47/50 full SB3 resets
  (the raptor reference is 48/50), with peak actuator force at 48.1% of its
  bound. CPU-MJX tests pin the same complete home reset, residual action
  mapping, and live contact observations. Root-pitch, tail-spring,
  neck-stiffness, COM-shift, and static-control A/Bs were rejected before
  changing the stance mechanics. See
  `docs/investigations/TREX_HOME_EQUILIBRIUM.md`.

- **`metrics.json` now describes the promoted checkpoint**: the post-stage
  quality/velocity/success evaluation loaded SB3's mean-reward
  `best_model` while next-stage training loads `robust_best_model` when
  present, so the sidecar could describe a policy the curriculum never
  promoted (in run `20260720_203454`, the degenerate lucky-peak 50k
  checkpoint). Both paths now share one selection helper
  (`robust_best_model` → `best_model`, each only with its matched
  VecNormalize stats), and the sidecar records which checkpoint it
  evaluated as `quality_eval_checkpoint`.
- **Velociraptor foot touch sensors were dead during normal stance**
  (breaking — policy-interface revision 2 → 3, visual revision 1 → 2): the
  raptor is digitigrade and a lying toe capsule contacts the floor near its
  ends, but the r=0.02 touch sites sat at the toe midpoint, so both sensors
  — and the two foot-contact observation dims fed from them — read 0 while
  standing (verified: six active floor contacts, zero sensor signal). The
  sites now envelop the whole toe_d3 capsule. Fixing them exposed a second
  defect: the adjacent toe digits (d3/d4) interpenetrate at rest and the
  solver held a constant ~300 N phantom repulsion between them, which the
  enlarged sites would have summed even airborne; new contact excludes
  remove it. Verified post-fix: ~37 N per foot at settled stance, exactly
  0 N airborne, 48/50 standing-probe survival unchanged, and regression
  tests pin stance/airborne sensor behavior.
  Follow-up (visual revision 2 → 3): the enlarged sites rendered as large
  gray balls on the feet in run videos (first visible in run
  `20260721_141523`'s stage-1 video); they now live in site group 4,
  which default rendering hides, so they no longer draw — toggle site
  group 4 in an interactive viewer to visualize the touch volumes.

- **`BaseDinoEnv` reset-height jitter is separable from joint jitter**
  (`reset_height_noise_scale`, breaking — policy interface revision for the
  home-keyframe-residual species: velociraptor 4 → 5, trex 4 → 5,
  dibothrosuchus 1 → 2): `reset()` applied `reset_noise_scale` — a joint-angle
  scale in **radians** — as a root-height jitter in **metres**. That is nearly
  harmless on a metre-tall plant (T-Rex: 10 % of stance) and badly wrong on a
  short one. On Dibothrosuchus's 0.313 m stance the committed 0.14 spawned
  trunk heights across 0.050–0.691 m, putting **25 % of stage-1 episodes
  outside `healthy_z_range` before the first step**; all `fallen` and
  `too_high` terminations fired at step 1. The stage-1 baseline therefore read
  as "62 % full-horizon" while measuring unlearnable spawn noise rather than
  balance — genuine balance failures were only ~13 %.

  The new argument defaults to `None`, which reproduces the previous
  arithmetic exactly, so velociraptor, trex and brachiosaurus reset
  distributions are unchanged and their checkpoints remain valid; the revision
  bumps reflect the shared `home_reset` source moving, not a behaviour change.
  `reset_noise_scale` remains the master switch, so zero still gives a fully
  deterministic reset. Dibothrosuchus sets 0.03 m (~10 % of stance, 4.5σ clear
  of its healthy-height floor) and its stage-1 noise is recalibrated from 0.14
  to 0.30 rad on that basis. A regression test pins that no stage-1 seed spawns
  already terminated.

### Added
- **Tests for the unified training entry point** (`environments/shared/train.py`),
  which the newly-honest combined coverage showed at **0%** — the module was
  never imported by any test. It is the `--species` shim in front of
  `train_base.main`, so a break there takes out every documented
  `python -m environments.shared.train ...` invocation while the species
  packages stay green. 13 tests cover both validation exits, the argv rewrite
  (species pair at the front, middle and end; `argv[0]` preserved), the
  hand-off to `get_species_config`/`main`, and an unknown species propagating
  the registry's `ValueError` rather than being swallowed. The `sys.path`
  bootstrap is pinned against repository landmarks rather than its
  `parents[2]` arithmetic — the same depth-counting failure that produced the
  provenance-root bug on this branch — and the `__main__` guard is exercised
  end to end as a subprocess. 0% → 95%, the remainder being the guard body
  itself, which only runs out of process. All nine mutations of the module
  (both exit codes, the argv slice bounds, the parent depth, the guarded
  `sys.path` insert, and swallowing the registry error) turn the suite red.

- **New species: *Dibothrosuchus elaphros*** (77 obs / 27 actuators / 8.65 kg
  / nq=35, nv=34) — a sphenosuchian crocodylomorph and the first non-dinosaur
  in the project. It is a small, gracile quadruped that held its limbs erect
  and parasagittal rather than sprawled, with hindlimbs longer than its
  forelimbs, paired paramedian rows of dorsal osteoderms, and a narrow crested
  skull whose snout drives the Stage 3 "snap" contact proxy (no jaw
  articulation). Ships the MJCF plant, `DibothrosuchusEnv`, MJX registration,
  three curriculum stages plus PPO/SAC sweep spaces, 65 species tests, a
  `test-dibothrosuchus` CI job, and the model documentation page.

  Plant characterization, all measured rather than assumed: the stance is
  authored into the body offsets so **every hinge joint and the whole home
  control vector are 0** at the keyframe; holding that control for 1500 steps
  settles the trunk at 0.3129 m with forward_z +0.003, so `natural_pitch` is 0
  and posture/nosedive keep their world-vertical reference. All four pads
  carry the full 84.9 N of weight, split 71/29 rear/front. Hip pitch, knee and
  ankle carry 1.5×kp `forcerange` on **every** leg (a 1.24 m animal strides
  fast relative to its body length, so all three stance joints see gait-scale
  torque); measured clipping is 0.0 % under home-control settling, a 1.5 Hz
  walk and a 2.5 Hz trot, and 0.5 % worst case under 4 Hz full-amplitude
  excitation. Stage-1 `reset_noise_scale` is 0.30 rad, calibrated against the
  zero-action baseline with root-height jitter decoupled (see below): 0.05,
  0.10 and 0.14 all leave a do-nothing policy at 100 % full-horizon, while
  0.30 leaves 65 % with every failure a genuine topple (earliest step 40,
  median 58) and nothing spawning out of bounds.

- **Canonical result bundles for Colab/Google Drive training**: schema-v2
  summaries, runtime-captured provenance, immutable completed bundles,
  selected/terminal evaluation evidence, plant/config/model/VecNormalize
  hashes, partial-versus-promotion validation, idempotent manifests, and
  strict exporter tests now make failed runs auditable without presenting
  them as publishable results.
- **Velociraptor Stage-1 basin investigation**: records the pre-training
  50-seed physics probe, Commit A's A-only PPO run (`20260720_203454`), the
  natural-lean diagnosis, Commit B's reward geometry, targeted backend-routing
  lessons, and the falsifiable next-run plan.

### Changed
- **Velociraptor actions are residuals around the XML `home` controls**:
  `-1`/`0`/`+1` map to actuator minimum/home/maximum through a piecewise-linear
  transform. This is a breaking policy-interface revision (1 → 2); older PPO,
  SAC, and JAX checkpoints remain historical and cannot be resumed as current
  policies. XML physics and actuator reach are unchanged.
- **Velociraptor posture shaping targets its natural forward pitch** (0.35
  rad) instead of world vertical. The yaw-invariant, direction- and roll-aware
  squared-chord formulation keeps finite JAX gradients at the exact target;
  T-Rex and Brachiosaurus retain vertical posture targets.

### Fixed
- **Targeted natural-posture consistency across training and evaluation
  paths**: the shared NumPy/JAX primitive, direct MJX, JAX total and detailed
  reward paths, CPU evaluation, diagnostics, and runtime `natural_pitch`
  overrides resolve the same reward-only target. Absolute-tilt termination and
  the existing nosedive boundary remain unchanged, and the JAX notebook now
  binds reward functions only after environment creation. Comprehensive
  fixed-state, all-component SB3↔JAX parity remains open.

## [Unreleased] — RL/GCP Review Fixes & Model Physics (v0.3.2)

### Changed
- **Plateau diagnostics now follow deterministic stage-gate evaluations**:
  rollout return warnings (which fired at different rates for PPO and SAC and
  repeated every rollout) are replaced by a warn-once/re-arm state machine.
  It monitors the currently blocking configured gate — balance duration,
  forward velocity, task success, then reward — and explains that training
  continues independently. SB3 evaluation now reports task success as `N/A`
  when the current stage has no success-rate gate, while preserving genuine
  Stage 3 `0%` results and success arrays.
- **Trex and brachiosaurus gait actuator headroom raised to 1.5×kp**
  (breaking — plant change): the same static-only forcerange sizing that
  broke velociraptor stage-2 measured 44–50 % hip / ~31 % ankle / 10–14 %
  knee gait-cycle clipping on trex (2.5 Hz) and 20–33 % hip clipping on
  brachiosaurus *at a 1.5 Hz walk*. Raised the clipped groups only (trex
  hip pitch/knee/ankle; brachio's four hip pitches) — post-fix clipping
  ≤0.5 %, root-z divergence vs an unbounded plant ≤2 mm. New
  `test_actuator_bounds.py` for both species pins the contract, built on
  shared helpers (`environments/shared/tests/actuator_bounds_helpers.py`);
  the velociraptor test now uses the same helpers. A species-generic
  saturation report lives at
  `environments/shared/scripts/actuator_saturation_report.py` (the
  velociraptor script delegates to it). Policies trained on the old caps
  will not transfer.
- **Entropy decay enabled for velociraptor stage 1** (`ent_coef_end =
  0.001`): run `20260711_235303` — which cleared stages 1–3 and set the
  stage-2 record (2705.93 ± 7.84 @ 2.75 m/s) — showed stage 1 destabilizing
  late (peak 1524 @ 2.45M → 385 @ 3.95M, bimodal falls) under constant
  `ent_coef`, the same pathology entropy decay fixed in stage 2. Staged
  (commented) `ent_coef_end` suggestions added to the trex and
  brachiosaurus stage 1/2 TOMLs for their next runs.
- **Raptor hip-pitch/ankle actuator headroom raised to 1.5×kp** (breaking —
  plant change): the blanket 0.8×kp `forcerange` from the July 9 model
  hardening clipped 34–40 % of hip and 22–25 % of ankle torque during a
  moderate gait cycle and collapsed stage-2 locomotion twice (runs
  `20260709_185946` and `20260711_165924`, bitwise-identical failures). At
  1.5×kp the same excitation measures 0 % clipping and zero pelvis-z
  divergence from an unbounded plant; `test_actuator_bounds.py` now pins
  the per-group ratios and asserts the gait regime stays *unclipped*.
  Velociraptor policies trained on the 0.8×kp plant will not transfer.
- **Velociraptor stage-2 config hardened against the fragile-fast-gait
  collapse**: `ent_coef_end = 0.001` (entropy decay — action std grew
  1.18→1.49 in both collapses), `forward_vel_max` 3.0 → 2.5 (saturate the
  speed incentive just past the 2.0 m/s gate), `fall_penalty` −50 → −150
  (late-fall hedge; the corrected break-even math is in
  `docs/investigations/STAGE2_RECOMMENDATIONS.md` §2.1).
- **Docs reorganized**: dated run analyses and reward investigations moved
  to `docs/investigations/` (TRAINING_REVIEW, TRAINING_REVIEW_JAX_STAGE1,
  REWARD_DISCREPANCY_INVESTIGATION, REWARD_SCALE_REDESIGN,
  STAGE2_INVESTIGATION, STAGE2_RECOMMENDATIONS); `docs/README.md` now maps
  the four doc categories (living reference / plans / investigations /
  reviews) and their lifecycles.
- **Python 3.11+ required; Python 3.13 supported** (breaking): dropped
  Python 3.10 support (`requires-python >= 3.11`), added 3.13 to the
  trove classifiers and the CI test matrix (3.11 + 3.13), and bumped the
  Docker base image to `python:3.13-slim`.  The `tomli` conditional
  dependency and import fallbacks are gone — stdlib `tomllib` is always
  available on 3.11+.  Ruff now targets py311; mypy targets 3.12 because
  numpy ≥ 2.3 stubs use PEP 695 syntax that mypy rejects for older targets.
- **Homepage light mode actually renders light**: the landing page palette
  (`index.module.css`) gained a `[data-theme='light']` variant — until now
  every section was hardcoded dark, so toggling the theme changed almost
  nothing.  Accent colors are darkened for WCAG AA contrast on light
  surfaces, the curriculum/feature icon colors follow the theme, and the
  forced-dark homepage navbar override (plus its logo-swap hack) in
  `custom.css` is removed.  The footer now follows the theme in both modes
  (Infima's `.footer--dark` element-level variables had silently overridden
  the intended colors, so it rendered `#303846` even in dark mode).  The
  code terminal mock-up intentionally stays dark in both themes.

### Fixed
- **Trex/brachio stage TOMLs: restored `[ppo.policy_kwargs]` headers lost
  in the entropy-decay staging** (PR #436): the dropped headers put
  `net_arch` into the top-level `[ppo]` table, crashing `PPO.__init__()`
  at stage start (trex run `20260712_185931`). New config regression
  tests now pin the structure for every species/stage: `net_arch` only
  under `policy_kwargs`, `[ppo]` keys validated against the PPO
  constructor signature plus harness schedule keys, and `[env]` keys
  validated against each species' env constructor signature.
- **JAX pipeline correctness** (July 2026 review, PR #426): curriculum gate
  now compares episode-level return against the TOML threshold (it compared
  per-step reward and never advanced past stage 1); observation-normalization
  stats carry across curriculum stages; `MJXDinoEnv` no longer mutates the
  live species registry (stage config leaked across stages); trex TOML
  `approach_weight` was shadowed by a legacy registry key; `vf_clip_range`
  now reaches the fused PPO update; CPU evaluation observes the real target
  body (was hardcoded to the world origin) and its reward includes the same
  approach/proximity/success/fall components as training; eval records
  per-episode success and gates on TOML `min_avg_forward_vel` /
  `min_success_rate`
- **SB3 pipeline**: eval-collapse callback re-read `evaluations.npz` every
  training step (a GCS network read on Vertex); `diagnostics.npz` was fully
  rewritten up to 3× per rollout-end (O(n²) I/O, severe for SAC); final
  model now saved before the post-training eval; `time_to_target` no longer
  averages a `-1` sentinel into the mean
- **Sweeps / Google Cloud**: sweep resume no longer submits duplicate Vertex
  HPT jobs after a swallowed timeout; Ray Tune stage-2/3 curriculum gates no
  longer fail unconditionally (metric-alias mismatch; NaN treated as
  missing); `--wandb` with a missing API key no longer kills headless
  training and relaunches resume the same W&B run; TensorBoard events sync
  to GCS periodically (spot-VM preemption no longer loses a stage's logs);
  `_wait_for_job` honors `--stage-timeout` during persistent API failures;
  effective per-trial seed and run identity recorded in `metrics.json`
- `ray_tune_sweep.ipynb` shipped with a `SyntaxError` in the
  save-search-space cell
- Broken sweep commands in `vertex-ai.md`; stale `configs/sweep_*.json`
  paths; nonexistent `wandb_project`/`WandbHook` in `jax.md`

### Changed
- **Brachiosaurus model physics (breaking for trained policies)**: leg
  springs now reference stance angles with stronger, force-bounded servos —
  the model can now statically stand at torso z≈1.13 (previously collapsed
  to z≈0.70, below the z=1.0 alive floor, even when commanded straight).
  Pre-existing brachiosaurus checkpoints will not transfer.
- All three models use `integrator="implicitfast"` and bounded `forcerange`
  on position actuators (raptor/trex bounds sized to leave settle behavior
  unchanged)
- Headless JAX runs with `--checkpoint-dir` now produce a per-update
  training CSV, rotating + final checkpoints, and a best-episode-return
  snapshot; `--curriculum` forwards the checkpoint dir per stage; the
  single-stage CLI passes previously-dropped TOML `[jax]` keys
  (`minibatch_size`, `warmup_*`, `ramp_*`, `num_envs`, `vf_coef`)
- New `gcp` extra (`google-cloud-aiplatform`, `google-cloud-storage`),
  installed in the Docker image
- Code reviews consolidated: `docs/KNOWN_ISSUES.md` is the living list of
  open findings; dated reviews archived under `docs/reviews/`

## [Unreleased] — Codebase Consolidation & Training Results (v0.3.0)

### Added
- Velociraptor SAC training results — all 3 stages passed (90.0% strike success, 22M steps, 22:59:18)
- Brachiosaurus PPO training results — Stages 1-2 passed, Stage 3 (food_reach) at 16.7% success (target: 50%)
- Updated training review with Brachiosaurus Stage 2/3 analysis and Velociraptor SAC comparison
- `environments/shared/train_base.py` — shared training logic with `SpeciesConfig` dataclass (~1,100 lines), reducing each species' `train_sb3.py` from ~950 lines to ~44 lines
- `environments/shared/test_env_base.py` — shared test utilities for environment validation (~214 lines)
- `environments/shared/constants.py` — centralized simulation-wide constants (sensor layout, VecNormalize defaults, physics defaults)
- `environments/shared/tests/reward_test_helpers.py` — reusable reward assertion functions for cross-species test consistency
- Expanded T-Rex reward tests (nosedive, height, heading, spin, drift, backward velocity)
- Expanded Brachiosaurus reward tests (gait instability, speed penalty, food reach threshold)
- Consolidated training notebooks into single parameterized `notebooks/training.ipynb`
- Algorithm-specific training diagnostics persisted to `diagnostics.npz` for **both PPO and SAC** (previously only on TensorBoard): captured SB3 `train/*` metrics (PPO `clip_fraction`/`approx_kl`/`explained_variance`/`entropy_loss`, SAC `critic_loss`/`actor_loss`/`ent_coef`) under `algo_*` keys
- `diagnostics/action_saturation` and `diagnostics/action_abs_max` — fraction of action components pinned at the control limit and the peak magnitude, computed for both PPO and SAC (SAC previously had no action logging at all)
- `diagnostics/grad_norm` — PPO post-update (clipped) gradient norm, surfacing when the clip ceiling binds or the policy stalls
- `explained_variance` in the JAX PPO loss info dict, per-update CSV log, and console output (value-function fit diagnostic)

### Changed
- Species training scripts (`train_sb3.py`) are now thin wrappers around shared `train_base.py`
- `DiagnosticsCallback` plateau detection now averages every episode that completed during a rollout (accumulated per-step) instead of only episodes that happened to end on the rollout's final step — the old sampling was biased and sparse
- JAX PPO clamps the Gaussian policy's log-std to `[-5, 2]` (applied identically in `sample_action` and `ppo_loss`, preserving the importance ratio) so a collapsing/exploding policy can no longer drive the action std to 0/inf and NaN the log-prob and entropy terms
- Species test scripts use shared `test_env_base.py` utilities
- `BaseDinoEnv` now provides concrete helper methods for common reward computations, gait symmetry, and termination checks
- Raptor, T-Rex, and Brachio reward tests refactored to use shared helpers
- `CookieResetButton` now uses shared `resetConsentStatus()` instead of duplicating localStorage logic

### Removed
- ~3,000+ lines of duplicated code across training scripts, test utilities, and environment methods
- Unused `getConsentStatus()` export from CookieConsent component

## [0.2.0] - 2026-02-09

### Added
- TOML-based configuration for curriculum stage reward weights and hyperparameters (`configs/` directory)
- Config loader utility (`environments/shared/config.py`) with `load_stage_config()` and `load_all_stages()`
- Gymnasium namespace registration (`MesozoicLabs/Raptor-v0`, `MesozoicLabs/Brachio-v0`, `MesozoicLabs/TRex-v0`)
- Pre-commit hooks with Ruff (format + lint), mypy, and standard checks
- `pytest-cov` for test coverage reporting with 70% threshold
- `CONTRIBUTING.md` with development workflow guidelines
- `CHANGELOG.md` (this file)
- GitHub issue templates (bug report, feature request, new species proposal)
- Reward function unit tests and curriculum stage transition tests
- Package metadata in `pyproject.toml` (authors, license, classifiers, URLs)
- `mypy` and `ruff` configuration in `pyproject.toml`
- `CurriculumManager` class for automated multi-stage training (`environments/shared/curriculum.py`)
- `CurriculumCallback` SB3 callback that monitors evaluation and stops training when advancement thresholds are met
- `thresholds_from_configs()` helper to extract curriculum thresholds from TOML configs
- `[curriculum]` section in all TOML stage configs with `timesteps`, `min_avg_reward`, `min_avg_episode_length`, and `required_consecutive` fields
- `curriculum` subcommand in all three species' training scripts for automated end-to-end 3-stage training
- `LocomotionMetrics` class with gait symmetry, cost of transport, stride frequency, and time-to-target (`environments/shared/metrics.py`)
- `WandbCallback` for SB3 with per-component reward logging and config snapshots (`environments/shared/wandb_integration.py`)
- `WandbCallback` video recording of evaluation episodes (`video_env`, `video_freq` parameters)
- `wandb` added to `[train]` optional dependencies

### Changed
- Training scripts now load stage configs from TOML files instead of hardcoded dictionaries
- Config loader now parses `[curriculum]` section from TOML files into `curriculum_kwargs`
- Bumped version from 0.1.0 to 0.2.0
- Dev dependencies expanded: `pytest-cov`, `mypy`, `ruff`, `pre-commit`
- All `print()` calls in training scripts replaced with `logging` module
- Gymnasium environments auto-register on `import environments` (no longer requires `register_all()`)

## [0.1.0] - 2025-01-01

### Added
- Initial release
- Velociraptor bipedal locomotion environment with sickle claw strike
- Brachiosaurus quadrupedal locomotion environment with food reaching
- T-Rex bipedal locomotion environment with jaw bite
- 3-stage curriculum learning (balance, locomotion, behavior)
- PPO and SAC training via Stable-Baselines3
- MuJoCo MJCF models for all three species
- Colab-ready training notebooks
- Docusaurus documentation site
- GitHub Actions CI with per-species test jobs
