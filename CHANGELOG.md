# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Reproducible Runs & Velociraptor Stage-1 Diagnosis (v0.3.7)

### Changed
- **Stage-1 review cleanup, final batch (§2.3)** — the cosmetic and
  small-code items closing out the 2026-08-15 review's cleanup list:
  - **Evaluation CSVs: `success` column renamed `task_success`** (gate-pass
    postmortem, follow-up 1). The column records the stage's TASK event
    (bite/strike/food-reached), which a stance-gated stage can never emit —
    it reads False in every stage-1 row by construction and was repeatedly
    misread as the gate verdict, which lives only in
    `stance_gate_report.{txt,json}` and `stage_summary.txt`. The bundle
    reader accepts the legacy `success` header so pre-rename bundles still
    audit (pinned by a test); the writer emits only the new name.
  - **The biped-empty "diagonal pair" contact panel is fixed**
    (narrow-tolerance postmortem, item 6). `plot_foot_contacts` keyed its
    quadruped layout off key *presence* in `diagnostics.npz` — but
    `DiagnosticsCallback` saves every info key for every species, so a
    biped's file contains `rr/rl_foot_contact` as all-NaN series: the figure
    grew an empty diagonal panel and routed the two real feet through the
    four-foot branch under FR/FL labels. Detection now requires finite data;
    a regression test pins the all-NaN-keys biped case the existing
    omitted-keys fixture could never catch.
  - **`stage1_balance.toml` comment corrections**: the release-ablation cost
    justification updated from the r6-era "13 panels" to the current 17
    (3 controls + 2 × 7 matched groups; toes skips); the self-contradictory
    `posture_weight` ("Increased from 1.5") and `nosedive_weight`
    ("Increased from 3.0", a decrease) provenance claims replaced with
    honest notes — both date from the file's first commit and compare
    against unrecoverable pre-repo values; the `[sac]` block gains a
    staleness warning and corrected cross-references (its "match PPO"
    lr/gamma/budget comparisons were written against a [ppo] block that has
    since moved to 3e-5 / 0.98 / 10M — the block itself has never been run);
    `ent_coef_end`'s entropy-floor analysis is era-marked as the r6
    21-actuator measurement, with a note that the figures still hold on r7.
  - **`_actuator_pose_mapping`'s docstring** now describes the deleted toe
    actuators in the past tense with the r6/r7 framing
    (plant_versions.toml note 9) instead of presenting an impossible
    example as current.
- **KNOWN_ISSUES pruned against the r7/r11 campaign** — the §2.2 batch of the
  2026-08-15 review (`docs/STAGE1B_IMPLEMENTATION_PLAN.md`), applying the
  file's own policy ("when an item here gets fixed, delete it"). Three
  T-Rex stage-1 entries deleted, each fixed or falsified by commits already
  on `main`; their evidence lives on where cited:
  - **"A passing policy cannot stand if its actions are filtered"** — the r6
    measurement behind "an action filter cannot be retrofitted." The r11
    10 Hz command low-pass (#503) did what the entry prescribed — filter
    present during training, unfiltered checkpoints invalidated — and the
    certified policy survives every probed cutoff down to 5 Hz at 96% of
    reward (`TREX_STAGE1_GATE_PASS_RUN_2026_08.md` §4; original analysis
    archived in `TREX_STAGE1_BOUNCE_2026_08.md` §5).
  - **"The passing policy stands in a pose it cannot hold without continuous
    feedback"** (HIGH) — its stated closure condition (re-run the
    hold-constant probe on an r7-trained policy) was met: 3/10 episodes to
    the full horizon, mean 611 steps, gentle nosedives — near-statically
    stable, measured on one seed (gate-pass postmortem §4). The entry's
    bolded advice that raising `smoothness`/`action_jerk` is contraindicated
    described the deleted r6 pose and was empirically falsified by the
    0.1→2.0 smoothness escalation the certified run trained under.
  - **"Half the actuators sit saturated and nothing opposes it"** — the #504
    `action_saturation` penalty (0.5 / 0.9) now opposes it, and the r7
    counts were re-measured: 5 of 15 parked in the narrow-tolerance run,
    **0 of 15** in the gate-pass run (max |DC| 0.890, parked under the
    ramp). One-sided-headroom residue stays tracked in #491.
  - The surviving **"passes or bounces"** entry gained a dated update: its
    numbers are r6-era, and the seed-replicate ask now targets the r11
    gate-pass configuration (n = 1).
  - The **latent raw-vs-clipped reward trap** paragraph was rewritten with
    function-level anchors (its `base_env.py` line numbers had rotted by
    ~250 lines) and scoped to filter-free species — r11's `_filter_action`
    clips before the reward terms read the action, closing it for the T-Rex.
  - Same-family staleness in `docs/hardware/SIM_TO_REAL_PLAN.md` §3.3 fixed:
    two rotted `base_env.py` line anchors re-pointed at
    `BaseDinoEnv._scale_action` / `__init__`, and "no action filtering
    anywhere in the step loop" corrected — the T-Rex has filtered in-loop
    since r11; the other species still do not.
- **Stage-1 closeout cleanup** — the four pre-training-run items from the
  2026-08-15 review (`docs/STAGE1B_IMPLEMENTATION_PLAN.md` §2.1):
  - **`DiagnosticsCallback`'s raw saturation metric renamed
    `diagnostics/raw_action_saturation`** (was `diagnostics/action_saturation`).
    The #504 pack gave the env an `action_saturation` info key — the
    post-filter ramp fraction behind `reward_action_saturation` — which the
    callback recorded at rollout end and then **overwrote in the same pass**
    with its pre-clip |a| ≥ 0.99 fraction under the identical TB key. The two
    disagree by construction (pre-clip raw sample vs filtered command; 0.99
    hard threshold vs 0.9 ramp). `diagnostics/action_saturation` is now
    unambiguously the env's; dashboards tracking the raw quantity should
    follow the rename (TB traces from runs between #504 and this fix carry
    the env value only until the first rollout end, then the raw overwrite).
    `action_bound_report.py`'s cross-references updated to match.
  - **T-Rex filter-probe sweep re-derived for the r11 plant**:
    `stance_probe_filter_hz` `[5, 10, 20, 30, 35]` → `[1.0, 2.5, 5.0, 8.0,
    10.0]`. The old sweep predates the plant's own 10 Hz command low-pass —
    probe cutoffs above 10 Hz stack a wider filter onto a narrower one and
    measured approximately the unprobed checkpoint (30/35 Hz were
    near-duplicate rows of the gate panel at 10 episodes each). The gate-pass
    checkpoint also survives every old cutoff, so the informative region
    moved **below** the plant cutoff; the new sweep locates the degradation
    knee there, keeping 10.0 as the cross-run continuity point and
    stack-sanity row. Same five-panel cost.
  - **Release-ablation group test extended to the r7 groups**:
    `test_every_group_gets_both_directions` now covers `knees_ankles`,
    `left_leg`, and `right_leg` — the #501 groups the passive-toes run's
    knee-localization conclusion rests on, which had no coverage — and the
    `_TREX_ACTUATORS` fixture gains `l_knee`/`l_ankle` so the per-side
    groups resolve more than a hip roll.
  - **Shaping-pack constants pinned exactly in the stage-1 factory test**:
    `tail_home_pose_weight` 0.25 / `tolerance` 0.05, `action_saturation_weight`
    0.5 / `threshold` 0.9, `leg_home_pose_broad_fraction` 0.25 /
    `broad_scale` 6.0 (was `> 0.0` for three of them, while every neighboring
    assertion pinned exact values). The statue rails (2100 / 3495.2) were
    measured at exactly these constants, so a silent TOML drift on any of
    them invalidated the rails without failing a test.
- **T-Rex stage-1 reward shaping pack** — three measured responses to the
  knee-lock run (`docs/investigations/TREX_STAGE1_NARROW_TOLERANCE_RUN_2026_08.md`),
  each opt-in with zero defaults in both backends:
  - **`action_saturation` penalty** (weight 0.5, threshold 0.9): prices
    commands parked past 90% of their range. A saturated command cannot
    oscillate, so it was invisible to smoothness/jerk — the run parked six
    of fifteen actuators at hard stops. The parked stance now pays
    ~−200/episode; the statue commands zero and pays nothing.
  - **`tail_home_pose` term** (weight 0.25, tolerance 0.05 rad): prices tail
    *joint positions*; the existing tail term reads only tail-tip angular
    velocity, which a tail frozen against a stop satisfies perfectly.
    Targets are the measured **settled droop**, not the keyframe — the
    passive tail rests on its ventral stops under gravity
    (−0.2107/−0.2029/−0.0926 rad with sub-milliradian spread, 40 seeds),
    so the authored zeros are unreachable and would cap the ideal statue at
    ~0.25 quality (`TRexEnv._TAIL_SETTLED_QPOS` = MJX registry
    `tail_home_pose_targets`, pinned equal by the stage-1 factory test).
  - **Long-range gradient for `leg_home_pose`** (`broad_fraction` 0.25,
    `broad_scale` 6): mixes a 6×-wider Gaussian into the term so distant
    poses still feel a pull. The narrow Gaussian's gradient is zero a few
    widths out, which the run demonstrated by drifting monotonically from
    0.20 to 0.545 rad of home error without ever being pulled back; at the
    measured crouch distance the mixture still pays 0.44 quality with live
    gradient. At home both components are 1, so the maximum is unchanged.
  Statue rails re-derived with the pack (zero_action_baseline.py seed 3042,
  40 episodes, noise 0.05; stance_quality_baseline.py agrees): statue
  3241.3 → **3495.2 ± 13.9** (+250 tail term at its settled targets, ~+4
  broad-leg at the statue's own sag, +0 saturation), `min_avg_reward`
  1940 → **2100** (0.60×, nearest 10), `collapse_peak_floor_reference`
  3241.3 → **3495.2**. Physics stays r7; `statue_constants_physics_revision`
  unchanged.
- **T-Rex commands are low-passed at 10 Hz during training and evaluation**
  (breaking — plant change, policy interface revision 10 → 11; all existing
  T-Rex checkpoints are invalidated; physics revision stays 7 and the
  statue-derived constants carry, since the statue's zero action filters to
  zero). Both backends now run the commanded action through a first-order
  low-pass (`environments/shared/action_filter.py`, the same discretization
  as the stance-gate probe filter), seeded with the first post-reset action;
  the dynamics **and** the action-derived reward terms (energy, smoothness,
  jerk) consume the filtered command. The cutoff is plant-level by
  construction — an SB3 class attribute and a `register_species_mjx` field
  guarded by `_PLANT_INTERFACE_CONFIG_FIELDS` — so stage TOMLs cannot tune
  it, and the plant contract records the cutoff plus the filter fingerprint
  and asserts backend agreement. Motivation
  (`docs/investigations/TREX_STAGE1_NARROW_TOLERANCE_RUN_2026_08.md`): both
  2026-08 stage-1 runs converged on statically unstable poses stabilised by
  16.8–18.7 Hz command chatter — the narrow-tolerance run locked both knees
  against their stops and pumped the ankles ±28–33°, and its constant-hold
  probe fell backward onto the tail in 9/10 episodes within ~3 s — while
  the filter probe puts the balance task's real bandwidth at ~1.1–1.4 Hz.
  Amplitude penalties cannot price the strategy out (a saturated command is
  perfectly smooth); removing the bandwidth makes such poses unreachable
  attractors instead of cheap ones. Other species keep the exact legacy
  step arithmetic (cutoff 0.0, byte-identical policy fingerprints).
- **T-Rex stage 1 `leg_home_pose_tolerance` narrowed 0.20 → 0.10 rad**, and the
  two statue-derived constants re-derived with it (`min_avg_reward`
  1950 → 1940, `collapse_peak_floor_reference` 3270.3 → 3241.3, both from a
  fresh 40-episode zero-action baseline at seed 3042). Motivated by the
  passive-toes run postmortem: the 10M checkpoint stands on a pose the
  release ablation convicts (left knee 0.21 rad, left ankle 0.17 rad off
  home) that the 0.20-wide Gaussian priced at only ~29 of the term's 500
  points, while the chatter stabilising that pose forfeits ~1030 in the
  support-linked terms. The width was chosen from measurement, not taste:
  0.10 doubles the statue-vs-chatterer pricing differential (109 → 180
  points) while staying 2× above the statue's own p99 settle error
  (~0.05 rad hip-pitch gravity sag); 0.075 and 0.05 sit past the knee of
  the curve, paying −26/−83 more statue reward for +23/+33 differential.
  The statue remains the reward optimum with a wider margin over the
  measured chatterer (~1128 vs ~1057 points).
- **T-Rex toes made passive** (breaking — plant change, physics revision 6 → 7
  **and** policy interface revision 9 → 10; all existing T-Rex checkpoints are
  invalidated). The six per-digit `<position>` actuators (kp 60, forcerange
  ±50 N·m, ctrlrange −0.4363..0.8727) are deleted; the digits keep their
  hinges, and the `leg_joint` class already supplies the passive stance —
  stiffness 40 with `springref` 12.5° holds the old commanded home pose,
  damping 15 dissipates strike energy. Action dimension drops **21 → 15**
  (3 neck/head + 8 legs + 4 tail); the home-keyframe ctrl vector shrinks with
  it. The observation is **unchanged at 61** — toe joints stay in qpos/qvel,
  and all 16 sensors (including the six per-digit touch sensors) keep their
  indices. `visual_revision` deliberately stays 4: no geom, site, material,
  camera or light moved, and the visual layer carries no actuator or keyframe
  fields.
  Motivation is the stage-1 bounce postmortem
  (`docs/investigations/TREX_STAGE1_BOUNCE_2026_08.md`): the release ablation
  showed holding **only the six toes** at their commanded DC reproduces the
  whole constant-hold failure (125.5 steps vs the full pose's 128.6) while
  tail, neck and head held alone all stand the full horizon — the policy's
  learned 0.8727 rad claw curl splayed adjacent digits to opposite ends of a
  75° range, changing the support polygon. A splayed digit is a pose the
  passive spring cannot hold, so the failure mode is now structurally
  impossible rather than merely unrewarded.
  Statue re-measured on the new plant (`zero_action_baseline.py` seed 3042
  and `stance_quality_baseline.py 0.05 40` agree): standing reward
  **3270.3 ± 12.3**, 40/40 full horizon, unsupported duty 0.000 — within one
  standard error of the r6 figure 3271.8, as expected with the springs
  holding the same pose the servos did. `collapse_peak_floor_reference`
  updates to the measured 3270.3; `min_avg_reward` stays 1950 (0.60 × the
  statue, not re-rounded over a within-noise move). The settled pelvis height
  is unchanged at 0.9260, so `target_z`, `natural_pitch` and the nosedive
  threshold keep their pins.
  Also folded in, completing the substep-aggregation lockstep: the SB3
  site/body **height terminations** (T-Rex `head_tip_z` < 0.12 / `skull_z` <
  0.45, dibothrosuchus `snout_tip_z` < 0.04) now consume the per-substep
  MINIMUM height recorded by the step loop, so a head dip that recovers
  between control-boundary samples terminates exactly as MJX's any-substep
  height emulation does; the `head_tip_z`/`snout_tip_z` info keys keep
  reporting the boundary sample.
  The statue constants get the freshness guard their comments have begged
  for since #491: the stage TOML now records
  `statue_constants_physics_revision` = the plant revision the constants
  were measured on, and `test_statue_constant_freshness.py` cross-checks it
  against the manifest — so the NEXT plant bump cannot land without either
  re-measuring the statue or consciously updating the pin where a reviewer
  sees the constants did not move. And the action-filter probe sweep gains
  the 30/35 Hz cutoffs (`stance_probe_filter_hz`), closing the survival
  curve at the control Nyquist's edge instead of stopping at 20 Hz with the
  measured 22.7 Hz tremor unsampled above it.
- **Reverted T-Rex stage 1 `leg_home_pose_weight` to 0.5**, and the two statue-
  derived constants with it (`min_avg_reward` 2550 → 1950,
  `collapse_peak_floor_reference` 4250.4 → 3271.8). The 1.5 experiment **could
  not have worked**, and the instrumentation added alongside it says why
  (issue #491).
  `leg_home_pose` governs eight joints — r/l `hip_pitch`, `hip_roll`, `knee`,
  `ankle` — and measured per actuator, those eight carry **1.2% of the
  policy's commanded pose offset**. The other 98.8% is in the tail, neck, head
  and toes, which no term in this stage touches; 10–12 of 21 actuators sit
  pinned at `|action| = 1.000`. The governed joints were **already** near home
  (|DC| 0.035–0.263), so tripling their weight had nothing to pull on.
  Measured, it didn't: commanded DC moved **0.766 → 0.738**, 3.7%, while the
  run took roughly 2M extra steps to escape its early collapse. The joint
  *list*, not the weight, is the problem.
  What survives the revert is the discipline: both constants are documented as
  derived from the statue and neither updates itself, so any reward-weight
  change has to re-measure with `zero_action_baseline.py` and re-derive both.
  Because the statue commands `action = 0` at any weight its trajectory never
  changes and only the affected term rescales — which makes the new value
  exact and cheap to obtain, and makes forgetting it silent.
  `stance_probe_filter_hz` is **kept**, and widened from a single cutoff to a
  **sweep** — `[5.0, 10.0, 20.0]` on trex 1a. A single cutoff answers a yes/no
  that is already answered: the checkpoint that PASSED the gate falls at every
  cutoff from 5 to 35 Hz against a 100 Hz control rate, so its PASS/FAIL
  carries no information. What can move is how long the filtered policy
  survives — measured 101 steps at 5 Hz, 199 at 10, 288 at 20, against a
  1000-step horizon — and reading that against cutoff measures *how much*
  high-frequency content the policy depends on rather than merely that it
  depends on some (#491).
  Balance correction on this plant is a ~1.1–1.4 Hz phenomenon and every cutoff
  passes it essentially untouched (gain 0.963 at 1.4 Hz even at 5 Hz), while
  the measured 22.7 Hz tremor is cut 13.3 dB at 5 Hz and 3.6 dB at 20 — so a
  short episode means dependence on content well above what the task requires.
  `stance_probe_filter_hz` now accepts a number or a list; unusable entries are
  dropped with a warning rather than costing the cutoffs that are fine.
  The sweep lands in `stance_gate_probe_filtered.{txt,json}` as one curve
  sorted by cutoff, and is written even if a cutoff raises — a partial curve is
  still a curve. It rolls **10 episodes per cutoff** rather than the gate's 40:
  it certifies nothing, and the effect it measures is enormous (101 steps
  against 1000), so it does not need the sample size the bound's power is
  specified at. The whole sweep costs about what the old single 40-episode
  probe did.

### Fixed
- **Six readers the NN_id layout would have broken, found by the pre-PR
  adversarial review and each reproduced by execution before fixing.**
  The worst: the complete-bundle audit hardcoded its nine required
  `stage{N}/...` paths, so a full curriculum run in the new layout passed
  every save-side check, wrote a `status="complete"` manifest, then
  failed its own final validation as canonical-conflict — permanently
  wedged, since retries die in preflight against the complete status.
  Required paths now resolve to the run's actual directory names. Also:
  `sweep collect-results` discovered stages only by the `stage` name
  prefix (new runs collected zero rows — the legacy number now comes from
  the id suffix, never the position digits); `evaluate()`'s stage
  auto-detect silently fell back to stage 1 for NN_id paths (extracted as
  `detect_stage_from_path`, id-aware, recovery passes through as itself);
  `save_jax_stage_artifacts` accepted NN_id directory names its own
  prior-stage `stage*/` globs could not see; the Drive-summary notebook's
  collector walked only `stage{N}` dirs; and the GCS sync iterated stages
  1–3 by number, which would have silently never uploaded recovery's
  replays — it now mirrors every stage directory the run actually wrote,
  in any generation. Regression tests pin each fix.

### Changed
- **Run directories now name their stages `NN_id`** — `01_stance`,
  `02_recovery`, `03_locomotion`, `04_behavior` — so a run's folders sort
  in curriculum order and say what they trained (project decision,
  2026-08-20). The id suffix is the key and the numeric prefix is
  provenance: it records the stage's manifest position when the run
  happened, and it is deliberately NOT `stage{position}` — `stage2`
  already means locomotion to every pre-manifest artifact, and a
  `stage2_recovery` folder would smuggle the renumbering hazard back in
  through the filesystem. `stage_dirname()` is the writer-side authority;
  readers (bundle validation/evidence/audit, GCS sync) go through
  `find_stage_dir()`, which accepts every generation — `stage{N}`, bare
  ids, and the new form — so existing runs keep collecting. File-level
  prefixes (checkpoints, `*_final`, videos) stay on `stage_label()`.
- **Trex stage configs are named by stage id**: `stance.toml`,
  `locomotion.toml`, `behavior.toml` (recovery already was), reversing the
  earlier keep-historical-names choice — the manifest's `config` fields
  are the single source of truth, and every resolution path (integer refs
  included) now goes through the manifest instead of the `stage{N}_*`
  glob, which survives only inside the synthesizer for manifest-less
  species. Diagnostic scripts that carried their own filename maps
  (zero-action/stance-quality baselines, foot-sensor, action-bound,
  joint-excursion, observation-ablation reports) resolve through the
  manifest too, so they keep working for every species regardless of
  naming era. Catalog and historical investigation documents keep their
  recorded names — history is not rewritten.

### Fixed
- **The recovery stage records its replay videos.** `record_stage_video`
  seeded the replay environment with `seed + 2000 + stage` — integer
  arithmetic that raises `TypeError` for a semantic stage id, inside the
  replay recorder's best-effort try/except, so the first field recovery run
  produced figures but silently no `replays/`. Replay seeding now goes
  through `replay_seed()`: integer stages keep the historical arithmetic
  bit-for-bit, semantic ids map to a stable crc32 offset (not `hash()`,
  which varies per process), and tests pin both. Recovery replays also
  carry the stance side/front camera views and per-frame stance CSV — the
  same task plus pushes, and the side view is where a shove is visible.
- **The recovery stage can now start from Colab** (found by three field
  runs, each dying silently at the stance/recovery boundary). Plant
  identities record `model_path` repository-relative, and
  `derive_stage_task_fingerprint` passed it straight to
  `mujoco.MjModel.from_xml_path`, which resolves against the process cwd —
  correct in the test suite (cwd = repo root), fatal in Colab, whose
  notebook adds the clone to `sys.path` without chdir-ing into it. The
  first pushed-stage fingerprint therefore raised `ParseXML` before the
  stage directory existed, halting run-all with nothing in Drive to show
  why; push-free stages never load the model, which is exactly why three
  full stance runs sailed through the same code. The path now resolves
  against the repository root (absolute paths pass through), a missing
  model raises a `TaskFingerprintError` that names both paths, and a
  regression test pins that the fingerprint is cwd-independent.
- **The plant contract now records the perturbation engine** (CI caught
  what the pre-merge review missed). Every policy-interface fingerprint
  moved when `BaseDinoEnv.reset`/`step` learned to derive and apply push
  schedules — the interface hash covers `reset`'s source through
  `home_reset`, which all four species carry — but the manifest was never
  regenerated, so `plant-contract`, `test-sb3`, `test-shared`, and
  `test-jax-cpu` all failed on the same staleness error. Recorded as
  `plant_versions.toml` note 11 with a policy-interface revision bump for
  all four species (velociraptor 8→9, trex 11→12, brachiosaurus 6→7,
  dibothrosuchus 5→6; physics and visual untouched — no MJCF edit), then
  regenerated the manifest and species catalog. The note pins the
  compatibility facts: with perturbation off the new code is inert end to
  end (no extra RNG draw, no force written), so a perturbation-free
  episode is bit-identical to the previous plant and existing checkpoints
  remain valid on their own stages. This also retires the "7 pre-existing
  container failures" claim from the review round: those local
  `test_stage_layout`/`test_sweep_reporting` failures were this branch's
  own staleness (a stash bisect could not catch it — the perturbation
  changes were already committed), and all pass after regeneration.
- **The recovery stage arms its collapse backstop.** `recovery.toml`
  configured no collapse floor, so the backstop could resolve only to
  `inf` and never arm — exactly the silent failure
  `test_curriculum_early_stopping` exists to prevent, and it correctly
  refused the new stage. Recovery now mirrors stance's relative pair
  (`collapse_peak_floor_fraction = 0.45` ×
  `collapse_peak_floor_reference = 3495.2`) plus the 1M-step arming
  delay, with a comment recording why the **un-pushed** statue is the
  right reference for now (it errs in the arming direction; the pushed
  task's own zero-action baseline is what the P3 null panels measure, and
  P3 re-derives the pair).
- **Ten findings from the pre-merge review of the 1b branch**, the four
  severe ones first: (1) `recovery_quality/v1` was **fail-open** through
  the shared `evaluate_stage_gate` dispatch — it fell through to the
  reward-and-length gate, certifying a pushed stage on return alone; it now
  fails closed with a pointer at the resolver, the only supported verdict
  path. (2) Horizon-adjacent pushes whose recovery window could not fit
  were counted as failed shoves, structurally capping panel success below
  any threshold for every controller; the harness now judges only pushes
  that were **delivered** (window opened before episode end — undelivered
  phantom shoves are no longer recorded either) **and judgeable** (push +
  dwell fits the horizon). (3) The documented recovery warm-start
  (`--load` a stance checkpoint) would have hard-failed under
  `resume_same_stage` once checkpoints carry fingerprints; `train()` and
  the CLI gain `--load-mode initialize_next_stage`, and the notebook
  chooses the mode from manifest position. (4) The stage-entry warm-up
  injected a `forward_vel_weight` ramp (0.1 → 0.0) into the recovery task,
  whose config zeroes that weight; the ramp is now skipped when the
  stage's target weight is 0. Also: the notebook now actually computes,
  persists, attaches, and mode-validates task fingerprints (its recovery
  cell's lineage claim was previously unbacked); `require_gate_resolution`
  now verifies the resolution's own integrity hash, so a hand-edited
  frozen record is detected instead of trusted; twenty `Stage %d` log
  formats widened to `%s` (string stages raised inside logging on the
  recovery path); the safe-set docstring now describes the reference the
  code implements (panel-start height, P3 recalibrates); `--stage
  locomotion`-style semantic ids for legacy stages resolve through the
  manifest instead of erroring; and the no-subcommand CLI default path
  tolerates the absent `--load-mode`.

### Added
- **Recovery stage in the notebook, docs, and website**: the training
  notebook gains an opt-in "5b. Recovery Stage" cell pair —
  `RUN_RECOVERY_STAGE = False` by default — that warm-starts
  `train_stage("recovery", ...)` from the Stage 1 checkpoint, generates
  stage artifacts, and **records** the `none/v1` pilot verdict instead of
  enforcing a gate (contrast the numbered stage cells, which raise); the
  run is deliberately excluded from the result bundle until the
  integer-stage bundle schema migrates. `train_stage` itself is
  manifest-aware: artifact directories via `stage_label` (semantic runs
  land in `recovery/`, integers keep `stage{N}/`) and warm-up keyed off
  manifest position. Two library sites gained label-safe naming so
  recovery artifacts read `trex_ppo_recovery_*` / `recovery_final` rather
  than a fabricated number (`evaluation.record_stage_video`,
  `_record_stage_replays`; integer naming byte-identical). Docs: the
  README explains the stage manifest and the T-Rex four-stage curriculum
  with the `--stage recovery` invocation; the trex environment README
  gains the recovery example; the website's T-Rex page gains a Curriculum
  section (stance → recovery → locomotion → behavior, what each certifies,
  and why integers keep their historical meaning) and notes the toes are
  passive since r7; quick-start wording is species-neutral. The catalog
  tables / generated species data stay numbered-curriculum until the
  manifest migration's final part (recorded in the plan's status block).
- **The pushed-panel harness and the gate resolver (stage 1b, W4 part 2 +
  W5)** (`environments/shared/recovery_evaluation.py`,
  `environments/shared/curriculum/gate_resolver.py`): the evaluation
  machinery that puts real evidence behind `recovery_quality/v1`.
  `roll_recovery_panel` rolls any controller over the seeded pushed panel,
  measures the per-step safe set from **physical state** (pelvis height vs
  the panel-start reference, tilt, planar speed, bilateral foot load —
  provisional thresholds recorded in every evidence file), reads the push
  schedule from the environment itself, and writes one row per episode AND
  one per shove (controller, seed, push vector/timing, recovery step) —
  the split plan's pair-identity-is-part-of-the-estimand requirement.
  Pairing is structural and test-pinned: identical seeds produce identical
  push schedules across controllers with no coordination. Null controllers
  (zero-action statue, constant brace) ship as plain callables; a
  push-free environment is refused. The resolver freezes capability spec +
  null manifest (per-seed outcomes, exact-UCB headline) + decision
  procedure into an atomic, hashed `gate_resolution.json`; a **missing
  resolution blocks advancement**, a **stale one** (task-fingerprint
  mismatch) demands recalibration by name, and
  `evaluate_recovery_gate_from_resolution` is the only supported path to
  an advancing recovery verdict — thresholds from the frozen spec, paired
  differences from the frozen null panel, never a fresh roll. What
  remains before the first gated recovery run is measurement, not
  machinery: the P3 calibration panels (real null baselines, safe-set and
  threshold calibration) and the §8.1 transfer pilot.
- **The recovery gate statistic: `recovery_quality/v1` (stage 1b, W4
  part 1)** (`environments/shared/curriculum/recovery_gate.py`, registered
  fail-closed in the gate schema): certifies **per-shove recovery** —
  after each scheduled push the body must re-enter the safe set within
  `recovery_t_recover_steps` and dwell there for `recovery_dwell_steps`
  (touching the safe set mid-fall is not recovery); an episode succeeds by
  reaching the horizon AND recovering every push. The gate bounds episode
  success with an **exact Clopper-Pearson LCB** (scipy-free bisection on
  the binomial tail — the right shape for a binary event, where the
  stance gate's Student-t is not), reproducing the split plan's own
  pinned arithmetic to the digit as regression tests: LCB95(34/40) =
  0.72526, one-sided 95% upper bound of 0/40 = 7.216%. The paired
  null-superiority statistic (same seeds, same schedules, t-bounded mean
  difference — the pairing the schedule PRF exists for) ships as an
  optional criterion that **fails closed when declared without its null
  panel** rather than skipping; it becomes authoritative once the
  resolver (W5) freezes the baselines. All thresholds provisional until
  P3/P5 calibration; `recovery.toml` keeps `none/v1` until then, its
  comment now pointing at the waiting kind. Not yet built (W4 part 2,
  with W5): the evaluation harness that rolls pushed panels, computes the
  per-step safe mask, runs the null suite, and writes per-shove evidence
  rows.
- **Stage identity through the consumers (stage 1b, W3 part 2)**: the
  single-stage SB3 train path now accepts semantic stage references
  end-to-end — `--stage recovery` parses at the CLI (digits stay legacy
  numbers), `load_all_stages` walks the manifest (trex gains a
  `"recovery"` key in curriculum order; the integer keys and every other
  species are untouched, test-pinned), artifacts and directories label
  semantic runs by ID via `stage_label` (`recovery_final.zip` under a
  `recovery_*` dir — never a fabricated number; legacy integers keep
  their historical `stage{N}` form everywhere), the task fingerprint
  records the semantic ID as its stage, and stage-entry warm-up now keys
  off manifest *position* (identical behavior for integers; correct for
  `recovery` at position 2). The legacy numeric curriculum deliberately
  ignores semantic-only stages (`thresholds_from_configs` filters to
  integer keys) — including recovery there would validate its `none/v1`
  placeholder under advancement and correctly-but-prematurely refuse the
  whole run; the manifest-walking curriculum lands with the W4 gate.
- **The seed-44 run recorded — replication now 2/3** (addendum in
  `docs/investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md`, KNOWN_ISSUES
  update 3): run `20260815_205206`, gate **PASS**, the strongest yet —
  full-horizon 40/40, duty 0.0069 / UCB 0.0117, panel reward 3408.3 ± 88.5
  (97.5% of the statue), zero non-truncated terminations, same unsaturated
  stance. The decisive datum: **AC 0.132** — the same quiet post-anneal
  endpoint as seed 42 (0.135), nowhere near seed 43's stalled 0.329. The
  2/3 split tracks the noise floor exactly; "what decides the anneal's
  endpoint" is now stage 1's sharpest open question.
- **Semantic stage manifest and the recovery stage config (stage 1b, W3
  part 1)** (`environments/shared/stage_manifest.py`,
  `configs/trex/stages.toml`, `configs/trex/recovery.toml`): the T-Rex
  curriculum is now FOUR stages — stance → recovery → locomotion →
  behavior — identified by stable semantic IDs, not numbers
  (STAGE1_SPLIT_PLAN §4). The no-silent-renumbering guarantee is
  load-bearing and test-pinned: **an integer stage reference means the
  legacy number forever** (`resolve(2)` is locomotion even though
  locomotion's position is now 3), recovery — having no numeric history —
  is reachable only by ID, and a manifest that tries to reassign or
  reorder legacy numbers is rejected at load. Manifest-less species
  synthesize their legacy three-stage manifest, which is how
  "recovery for the T-Rex only" is expressed. `load_stage_config` accepts
  semantic IDs (`load_stage_config("trex", "recovery")`); integer loading
  is byte-for-byte untouched. The recovery config is stage 1's `[env]`
  verbatim plus exactly the five `perturbation_*` keys at the adopted
  §3.3 values (165.5 N derived on r7), with a freshness test pinning the
  mirror so stage-1 shaping changes cannot silently strand it; its gate
  is `none/v1` — the schema's own non-advancing-pilot rule refuses to
  advance through it, fail-closed, until `recovery_quality/v1` (W4) and
  the frozen null baselines (W5) exist. Not yet migrated (W3 part 2):
  the curriculum loop, CLIs, and sweep still iterate legacy integers;
  artifacts do not yet carry stage IDs.
- **Task/evaluation fingerprint and checkpoint load modes (stage 1b, W2)**
  (`environments/shared/task_fingerprint.py`): closes the provenance gap
  STAGE1_SPLIT_PLAN §3.2 names — a `step()`-level task change like the
  scheduled pushes moves no `policy_interface_revision`, so checkpoints
  stay mechanically loadable while being unvalidated for the new task. The
  fingerprint hashes the task-defining configuration (species, stage,
  backend, plant physics/interface hashes, the full `[env]` kwargs, and
  the perturbation block with its per-species **derived** newtons plus the
  schedule-PRF implementation name). Every SB3 stage now writes
  `task_fingerprint.json` beside `plant_identity.json`, embeds it in
  `stage_config.json`, uploads it to GCS, and attaches it to the model so
  SB3 persists it in the checkpoint ZIP. On load there are exactly two
  modes: `resume_same_stage` (exact match or a fatal error naming the
  differing sections — a changed task can never be resumed silently) and
  `initialize_next_stage` (the curriculum handoff: a boundary crossing is
  expected once and recorded as a parent/child lineage that travels on
  the new checkpoint). `train()`'s user `--load` is same-stage; the
  curriculum loop's only loads are handoffs. Fail-closed core with one
  dated transition valve: checkpoints minted before 2026-08-15 carry no
  fingerprint, so the train paths pass `allow_unfingerprinted=True`
  (warn, not fail) until fingerprinted checkpoints are the norm —
  tightening lands with the gate resolver (W5). The JAX training path
  does not attach fingerprints yet; `derive_stage_task_fingerprint` is
  backend-tagged and env-free, so wiring it there is mechanical.
- **Species-generic scheduled pushes, SB3 path (stage 1b, W1)**: every
  species' environment now accepts five `perturbation_*` parameters,
  default **off** — with the multiple at 0.0 no schedule exists, no RNG is
  drawn, and trajectories are byte-identical to the pre-perturbation code
  (pinned by test). When enabled, `BaseDinoEnv` derives the push force from
  the plant itself (`environments/shared/perturbation.py`: root-subtree
  mass, home-keyframe CoM height, floor-contact support hull → capture-point
  velocity; on the r7 trex, multiple 1.5 over 0.20 s derives to 165.5 N,
  reproducing STAGE1_SPLIT_PLAN §3.3's ~150 N from first principles),
  generates a deterministic per-episode schedule from a `lowbias32` hash
  (bit-identical on NumPy and JAX, so paired policy-vs-null evaluation gets
  identical pushes from identical seeds), and writes the self-clearing
  force to `xfrc_applied` at the root each control step. Reward and
  observation are untouched — the push changes the task, not the interface.
  `perturbation_manifest()` exposes the derived per-species constants for
  run provenance.
- **Scheduled pushes, MJX/JAX path (stage 1b, W1 completion)**: the five
  `perturbation_*` keys are `MJXEnvConfig` fields (so shared TOMLs flow to
  both backends), deliberately outside the versioned plant interface — a
  push changes the task, not what a checkpoint's actions mean; the coming
  task fingerprint (W2), not the policy interface, distinguishes pushed
  runs. The step kernel gates at trace time exactly like the r11 action
  filter: multiple 0.0 bakes the push-free trace with an empty `(0,)`
  schedule carry, so existing training is untouched. When enabled, the
  per-episode schedule is carried through `EnvState` (regenerated by the
  fused auto-reset from a `fold_in` of the reset key, leaving the existing
  draw sequence intact), and the self-clearing force is written into
  `xfrc_applied` at the root before the substep loop. Both backends derive
  from the same host model, so the per-species constants are identical to
  machine precision (pinned by a cross-backend test); schedule *content*
  differs across backends only through the seed draw, never through the
  PRF, which is bit-identical by construction.
- **The seed-43 replicate postmortem**
  (`docs/investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md`): the first
  replicate of the certified gate-pass configuration, and a gate **FAIL** —
  duty 0.0597 / UCB 0.0747, full-horizon 0.925, 0 of 200 panels passing,
  behind a deceptively healthy 3093 panel reward. The run reproduces the
  gate-pass trajectory's three acts and converges on the **same stance**
  (zero saturated actuators, head lowered ~−20°, tail in its settled droop,
  hip-roll balance), but the act-3 re-descent stalls at a broadband
  command-noise floor (AC 0.329 at the estimator's white-noise saturation,
  concentrated on the hip rolls) instead of quieting to seed 42's 0.135.
  Statue controls read 3493.8/3497.1 against the configured 3495.2, so the
  rails are fresh and the deficit is real. **n = 2 on the identical config:
  1 pass / 1 fail — seed-sensitive**, which resolves the "solved or lucky"
  question ("neither") and strengthens the stage-1b rationale: robustness by
  lucky seed is now measured. The config's ~4M prediction failed on the duty
  axis (0.345 vs < 0.28 predicted); the `algo_std` half remains checkable in
  the run's `diagnostics.npz`.
- **Stage 1b implementation plan**
  (`docs/STAGE1B_IMPLEMENTATION_PLAN.md`): maps the unbuilt recovery half of
  `docs/STAGE1_SPLIT_PLAN.md` onto the tree as it stands after the first
  certified stance-gate pass. Records the stage-1 closeout review's verdict —
  no blocker to starting the 1b build, with a verified cleanup list (a
  `diagnostics/action_saturation` logger-key collision, a filter-probe sweep
  that predates the plant's own 10 Hz filter, untested release-ablation
  groups, unpinned pack weights in the parity factory test, and four stale
  KNOWN_ISSUES entries falsified by the r11/gate-pass commits) — then lays
  out the build as six workstreams (perturbation engine, task fingerprint and
  load modes, stage identity, `recovery_quality/v1` gate, gate resolver,
  diagnostic tool modes), a run plan starting with the seed replicate the
  postmortems keep asking for, and seven decisions that need review before
  their workstreams start.
- **The passive-toes run postmortem**
  (`docs/investigations/TREX_STAGE1_PASSIVE_TOES_RUN_2026_08.md`): the first
  10M stage-1 run on the r7 plant under substep-honest metrics. Best survival
  ever recorded (100% full-horizon locked from 4.5M, reward 2213 vs the 1950
  rail, no collapse) and still a gate FAIL at duty 0.41 vs 0.02 — an 18.7 Hz
  foot chatter the boundary-sampled metric could have passed. The run
  falsifies the entropy hypothesis (ent_coef reached exactly zero at 7M with
  no break in tremor or duty), and the regrouped release ablation localizes
  the unholdable pose to the **left knee and ankle**, priced at ~29 of the
  home-pose term's 500 points while the chatter stabilising it forfeits
  ~1030 in support terms. The release ablation gains
  `knees_ankles` and per-side groups, and the impulse probe's prose no longer
  attributes asymmetry to toe splay the r7 plant cannot produce.
- **A constant-hold probe, which separates "needs bandwidth" from "needs
  feedback".** `stance_gate_report.py --hold-constant` replaces the policy's
  commanded action with the constant it commands *on average* — its own
  post-settle per-actuator DC — partway through each episode, and scores that.
  Runs automatically after the gate report when a stage sets
  `stance_probe_hold_constant`, writing
  `stance_gate_probe_constant.{txt,json}`.
  The existing low-pass probe could not answer this. A filtered policy still
  responds to what it sees, just slowly, so a fall under the filter shows the
  policy needs *bandwidth* without showing whether it needs *feedback* at all —
  and the two have opposite fixes. Cutting the loop outright decides it.
  Five variants bracketed by two controls: the unmodified policy (expressed as
  a handoff at the horizon, so it still carries the probe marker and can never
  land on the certification filenames) and the zero hold, which is the statue.
  Between them the measured pose is held from settle, hard and ramped, and from
  reset. The ramped variant exists to answer the obvious objection to a hard
  switch — that a step transient rather than the loss of feedback knocked the
  animal over — with a measurement instead of an argument.
  Measured on the passing `20260805_011234` checkpoint: **both controls reach
  the 1000-step horizon** (policy 3006.9, statue 3271.0) and **every held
  variant collapses** — 348.9 steps from settle, 330.2 ramped, 133.3 from
  reset, all at full-horizon 0.0000. The pose that policy holds requires
  continuous feedback, so the tremor is stabilisation rather than waste and
  raising `smoothness`/`action_jerk` is contraindicated. Full analysis in the
  2026-08-06 addendum to `TREX_STAGE1_BOUNCE_2026_08.md`.
- **All four stance probes now run automatically on every SB3 run.** The
  notebook calls `generate_stage_artifacts` and nothing else, so a probe
  reachable only from the CLI produces nothing on a real run — and a diagnostic
  nobody runs is a diagnostic that does not exist. `_run_stance_probes`
  dispatches to all four, each gated by its own `[curriculum]` key:
  `stance_probe_filter_hz`, `stance_probe_hold_constant`,
  `stance_probe_release_ablation`, `stance_probe_impulse_speeds`. All four are
  on for trex stage 1; every other stage and species is unaffected.
  The probes run **dead last** in `generate_stage_artifacts` — after the gate
  verdict is recorded, the summary written, and the graphs and replays saved —
  because they are the most expensive artifact step and nothing downstream
  reads them: a runtime lost mid-probe costs the probes alone, and a
  probe-wiring failure has nothing left to sink. (They briefly ran inside the
  gate report's own `try`, where a caller-side exception would have been
  caught by the report's handler and recorded a FAIL for a stage whose
  on-disk report says PASS — the founding defect of `_apply_stage_gate`, one
  layer up.) Pinned by behaviour tests through the real call sites — kwarg
  drift fails them, not just a deleted call — plus a source-order check that
  the verdict precedes the probes, so a fifth probe added beside them cannot
  quietly go unwired the way these two nearly did. Each is individually
  non-fatal at both levels: the helpers guard their own rollouts, and the
  runner guards each call site.
  Cost, per stage, at the trex settings: 3 filter panels at 10 episodes, 5 hold
  panels at 10, 13 ablation panels at 8, and 14 impulse panels at 8 — the
  impulse sweep doubled because the statue control is rolled over the same
  magnitudes and is not optional.
- **An impulse recovery probe**, `--impulse-probe`, which answers the one
  question no training artifact can: does a stage-1 policy actually *correct*,
  or has it only learned to stand still? Stage 1 declares no in-episode
  disturbance — the sole perturbation is joint-angle noise at reset — so the
  gate, the reward and all three earlier probes are silent on it by
  construction. This applies a step change to the root's linear velocity
  mid-episode (a shove, fully specified by `delta_v`) and sweeps magnitude in
  **both lateral directions**, with the **zero-action statue as the control**:
  it commands a constant and cannot respond, so its survival is the plant's
  passive robustness and only the policy's margin over it is active control.
  Measured on the passing checkpoint: **recovery envelope 0.50 m/s one way,
  0.00 the other.** It takes a 0.5 m/s shove to the horizon 8/8 pushed one
  direction and falls 0/8 pushed the other (424 steps against the statue's
  337); at 1.0 m/s and above it is within noise of the statue on both sides.
  Narrow *and* asymmetric — which is what a pose with differently-splayed toes
  predicts. So stage 1's gap is a **metric** gap: the capability partly exists
  and the task cannot see it, which makes the envelope a ready-made acceptance
  metric for `STAGE1_SPLIT_PLAN`'s 1b.
  The read-out is keyed off that envelope rather than a mean step margin. The
  first version averaged the margin across rows and reported +135, of which 82%
  came from one row — a number equally produced by a policy that recovers
  everywhere. That is the **third** pooled statistic in this investigation to
  hide the structure it summarised, after `action_std` over actuators-and-time
  and the saturation count over differing `ctrlrange`s. The pooled margin is
  still printed, second, with its caveat attached.
- **Per-actuator pose reported in degrees, and the table ordered by them.**
  `|action| = 1` is not one physical event: `_scale_action` maps `[-1, 1]` onto
  each actuator's own control span, and on the T-Rex those differ by 6× — a
  saturated tail joint is **8–12°** of deflection while a saturated toe is
  **37.5°**. Sorted by normalised `|dc|`, the per-actuator table put twelve
  joints at exactly `±1.000` at the top with no way to tell them apart, and the
  most conspicuous of them (the tail) was the one later measured to be
  mechanically inert.
  This is the same error the DC/AC split was introduced to fix, one level down:
  a pooled *normalised* offset cannot separate moving 8° from moving 37.5° any
  more than a pooled standard deviation can separate "sitting in the wrong
  place" from "shaking". Report schema v2 now records applied `ctrl_mean` and
  `ctrl_ac_rms` in native units, preserves the separate `action_zero_ctrl` and
  `home_ctrl` anchors, and marks which actuators are direct angular position
  controls. Only those receive `dc_deg`, `ac_rms_deg`, `range_deg`,
  `zero_offset_deg`, `home_qpos`, and `home_preload_deg`; geared motors are not
  mislabeled as radians. It adds `dc_rms_deg` across the compatible controls
  and **orders the table by degrees** — which puts the five
  saturated toes on top at ±37.5° and drops `tail_1_yaw` (±8°) out of the
  printed twelve. The normalised `dc` is kept alongside, because that is what
  inverts through the action penalties; the degree fields are reduced from the
  controls actually applied during rollout.
  This distinction matters for `home-keyframe-residual/v1`: T-Rex maps the two
  halves piecewise around `key_ctrl[home]`, so neither the mean nor RMS physical
  target can be reconstructed from normalised moments with one full-range
  scale. Action zero maps exactly to the home control for all 21 actuators. The
  ankles' +5.5° `key_ctrl - key_qpos` is an intentional gravity preload, now
  reported separately instead of being mislabeled as policy displacement or a
  reason to re-centre the control range.
  The same correction is now swept through the prose that taught the old
  identity: docstrings in `eval_diagnostics.py` and the hold/ablation helpers,
  the rendered ablation header, the CLI help, and the KNOWN_ISSUES saturation
  entry all say "home control" where they said "home keyframe" for what
  `action = 0` commands — the keyframe's *pose* keeps its name. The
  constant-hold docstring also now states its one deliberate approximation:
  the held command is `f(mean(action))`, not `mean(f(action))`, which differs
  only on zero-crossing actuators with asymmetric spans (neck/head pitch on
  this plant, bias ~0.01°).
- **A release ablation on top of it**, `--hold-release-ablation`, which asks
  *which* joints make a held pose unholdable. Each actuator group gets two
  variants: `release_G` holds everything except G (is G **necessary** — does
  removing it rescue the pose?) and `only_G` holds G alone (is G
  **sufficient** — does it break the statue by itself?). One side alone
  misleads, and a group can look implicated purely because it is conspicuous.
  Measured on the passing checkpoint: **holding only the six toe joints
  reproduces the entire failure** (125.5 steps against the full pose's 128.6,
  `tail_contact` 8 of 8 in both), while **holding only the four tail joints at
  ±1.000 stands the full horizon** at reward 3254.0 — within 0.6% of the statue
  — and head/neck likewise at 3286.1. No group is necessary (releasing all
  twelve saturated actuators still falls at 224.6), so the pose is
  over-determined, which the artifact now says in as many words.
  **This refutes the saturation hypothesis** recorded the same day: saturation
  is conspicuous, not causal, and the largest saturated group is provably inert.
  Every ablated variant is commanded **from reset**, which is load-bearing and
  was wrong in the first run: handing off at `settle_steps` snapped joints from
  ±1.000 to 0 mid-episode, the transient dominated, all thirteen variants landed
  within 311–349 steps of each other, and the `hold_zero` row — the statue,
  known to stand 1000 — fell at 323. The control caught it; without it the flat
  table would have read as "no group matters", which is a conclusion and a wrong
  one. `TestReleaseAblationVariants` now pins the from-reset requirement.
  Probe branding is now a registry (`_PROBE_MARKERS`) rather than a chain of
  `is not None` tests. The filter probe's guard was a single such test, and a
  second probe added beside it would have silently inherited the certification
  filenames and the `stance_panel_selected.csv` evidence a bundle is certified
  from.
- **The deterministic policy's action, measured instead of inferred.** Every
  action number on disk came from *training* rollouts, so all of it carried
  exploration noise; diagnosing issue #489 meant recovering the commanded pose
  and the tremor about it by inverting per-episode reward totals under a
  narrowband assumption. `StageAwareEvalCallback` now measures them directly
  during evaluation and `gate_progress.npz` carries them:
  `action_dc_rms` (the static distance from the home keyframe, which under
  `home-keyframe-residual/v1` is exactly `action = 0`), `action_ac_rms` (the
  tremor about it), `action_delta`, `action_jerk`, and `action_freq_hz`.
  The DC/AC split is the point: a pooled standard deviation over actuators and
  time — which is what `diagnostics.action_std` computes — cannot separate
  "sitting in the wrong place" from "shaking", and those have different causes
  and different fixes. Measured over the post-settle window, the same one the
  duty uses, so the reset transient does not inflate the AC term.
- **`action_dc_per_actuator` / `action_ac_rms_per_actuator`**, the same two
  quantities as `(n_evals, n_actuators)` matrices. Which actuators carry the
  offset decides whether a leg-pose weight can reach it at all: `leg_home_pose`
  covers only the leg joints, so a displacement living in the tail or neck is
  invisible to it and unfixable by it. Logged per actuator rather than
  per group because grouping is an analysis decision, and resolving joint names
  through the VecEnv wrappers is exactly the kind of best-effort lookup that
  silently returns nothing.
- **`term_*` in `gate_progress.npz`** — every reward term, per evaluation.
  Only `mean_reward` was kept, so comparing two runs meant comparing their
  final checkpoints and nothing in between, with no way to ask *when* a policy
  adopted the pose it ended up with.
- **`action_jerk` in `diagnostics.npz`.** The environment has always emitted it
  (it is what `reward_action_jerk` charges) and `INFO_KEYS` has always dropped
  it, so the signal was computed every step and discarded. With both
  differences recorded the effective frequency is free:
  `jerk/delta = (2 sin πfΔt)²`, which unlike either alone is blind to a
  constant offset.
- **`stance_gate_report.py --filter-actions HZ`**, a probe for whether a
  policy's high-frequency action content is load-bearing or waste. It
  low-passes the action between the policy and the plant and scores *that*:
  if the policy still stands, the tremor was waste and the fix belongs on the
  action path; if it falls, the tremor is closed-loop stabilisation and the fix
  belongs elsewhere (#489). Open-loop experiments on the statue bound what a
  tremor *can* do; only filtering the real policy answers it for that policy.
  A filtered rollout scores a **modified** policy, so the report records
  `filter_actions_hz` and the text form prints the warning *above* the verdict
  — a PASS obtained this way must never be mistakable for a gate result.
  **Wired into the training path**, not left as a manual script: with
  `stance_probe_filter_hz` set in `[curriculum]` (5.0 on trex 1a) the artifact
  writer re-scores the selected checkpoint after the gate report and emits
  `stance_gate_probe_filtered.{txt,json}`, so every run answers the question
  without anyone remembering to. Unset elsewhere, because it costs a second
  40-episode panel per stage and per sweep trial.
  The probe can corrupt the record two ways, and both are made structurally
  impossible rather than left to the caller: it gets its own filenames derived
  from the report itself, and it never writes `stance_panel_selected.csv` —
  that file is the per-episode evidence `result_bundle.evidence` certifies a
  stance-gated stage from, and a filtered panel there would certify a policy
  that was never run. `stance_probe_filter_hz` is registered in
  `gate_schema._DIAGNOSTIC_KEYS`, without which the fail-closed unknown-key
  check would make the setting unreachable.
- Every new evaluation series is recorded to the **SB3 logger as well as the
  npz**, so TensorBoard and W&B carry them live: `diagnostics/eval_action_*`
  and `reward_terms/eval_*`. A diagnostic that exists only in a file nobody
  opens mid-run is how the stance gate criteria stayed invisible for the whole
  of run `20260804_143747`.
- **`stance_gate_report.py` now reports the commanded action per actuator** —
  DC, AC rms, and the effective frequency, over the post-settle window, with
  real joint names (`r_hip_pitch`, `r_knee`, …). This script holds the
  environment directly rather than through VecEnv wrappers, so the
  actuator→joint mapping is exact rather than a best-effort lookup.
  It means **one five-minute run against an existing checkpoint** tests both
  premises of the `leg_home_pose_weight` change: whether the tremor is
  load-bearing (`--filter-actions`) and whether the static offset is even in
  the leg joints, which is the other way that change can fail by construction.
  Neither previously needed a training run to answer — they needed instrumentation
  that did not exist.

### Changed
- **T-Rex stage 1 `leg_home_pose_weight` 0.5 → 1.5, and the two constants
  derived from the statue re-measured with it.** Run `20260805_011234` **passed**
  `stance_quality/v1` — unsupported duty 0.0000 against a 0.0200 ceiling, 40/40
  episodes at the horizon, bilateral support 1.0000, confirming the entropy
  diagnosis in #487 — but scored **3004.3 against the zero-action statue's
  3274.4** and never crossed it in 200 evaluations (issue #489).
  The whole 270-point gap has one cause. Inverting the action penalties through
  their closed forms gives a per-actuator DC offset of **0.800** with an AC
  tremor of **rms 0.277 at an effective 22.6 Hz** (from `J/D = (2 sin πfΔt)²`,
  which is DC-blind); reproducing that tremor on the statue matches the trained
  policy's per-step penalties to within 2%. Both halves are open-loop fatal:
  a static offset of just **0.10 rms falls in every direction tested**
  (107–268 steps), and a 0.277 rms tremor collapses the statue at **every**
  frequency from 2 to 40 Hz (38–80 steps), while `action = 0` stands
  indefinitely. So the policy is holding a displaced pose upright with
  closed-loop feedback that the home keyframe does not need — the tremor is the
  cost of the pose, not a separate defect.
  That is why the fix is not on `action_jerk_weight`, whose penalty sits at
  1.4% of its normaliser and looks like the obvious lever. Penalising the
  tremor attacks load-bearing feedback, and the cheapest response to a bigger
  jerk penalty is to fall over. Raising the pose weight instead makes the
  displaced operating point uncompetitive — the pose gap goes 85 → 255 points,
  dominating the tremor's 112-point cost by 2.3× — and if the policy returns to
  the home keyframe the feedback becomes unnecessary and its cost goes too:
  **245 points, not the 112 a smoothness penalty could reach.**
  `min_avg_reward` 1950 → **2550** and `collapse_peak_floor_reference` 3271.8 →
  **4250.4**. Both are documented as derived from the statue (0.60× and 1.00×),
  and neither updates itself — leaving them would arm the collapse backstop
  against the wrong scale, the exact staleness the config's own comments warn
  about. The new baseline is measured, not scaled: 4250.40 ± 13.86 over 40
  episodes at noise 0.05, 100% full-horizon, from `zero_action_baseline.py`.
  The ratios are unchanged (rail 0.60×, floor 0.45× → 1913), which is what
  carries across a rescale; the absolute numbers do not, and neither do the
  historical run figures quoted in the config, now flagged as old-scale.

### Added
- **A watch on the do-nothing baseline.** Run `20260804_143747` trained T-Rex
  stage 1 for its full 10,002,432 steps — 8h 13m — and finished at **2686.9
  against the zero-action statue's 3271.8**: below the do-nothing policy on
  every major reward term, for the entire run. Nothing reported it. Every
  logged signal looked healthy (reward climbing, full-horizon 100%, evaluation
  sd down to 5, the collapse backstop correctly silent), and the run was only
  found to be in a worse-than-trivial local optimum by scoring `action = 0` by
  hand afterwards (issue #486).
  `curriculum.baseline_watch.BaselineProgressCallback` now reports every
  evaluation against the run's own captured `zero_action_baseline.json` — one
  float already written to the run directory before training starts, so the
  comparison is free — and warns **once** if the policy has still never beaten
  it past `baseline_warn_after_budget_fraction` of the budget (0.35 on trex
  1a, i.e. 3.5M steps with ~5.5 hours still to save).
  Deliberately **advisory**: it logs and warns, never stops. On stage 1 the
  statue *is* the reward optimum, so a learning policy legitimately sits below
  it for millions of steps; aborting on that would kill healthy runs for the
  same reason tightening the collapse detector does. A run that beats the
  baseline at any point is never warned about again — falling back under it is
  what the collapse backstop is for.
  `_build_core_callbacks` takes `species` to construct it, and a test pins that
  `sb3_training.ipynb` passes it, because an optional argument the trainer
  forgets is exactly how the stance gate became dead code on the Colab path.
- **`gate_progress.npz`, the gate criteria per evaluation.** The deterministic
  panel-estimator numbers the stance gate actually tests — `duty_episodes`,
  `unsupported_duty`, `unsupported_duty_ucb`, `bilateral_support_duty`,
  `mean_reward` and `full_horizon_fraction`, against `timesteps` — were computed
  every evaluation and recorded only to the SB3 logger. Diagnosing run
  `20260804_143747` therefore had to substitute the *training-rollout* duty
  from `diagnostics.npz`, which is contaminated by exploration noise; it
  happened to agree to three decimals, but that was luck.
  `StageGatePlateauCallback` now writes the file to the stage directory (beside
  `diagnostics.npz`, not the scratch eval dir) after each evaluation, so the
  gate's own view of a run is readable from Drive mid-run without TensorBoard.
  A separate file rather than columns in `diagnostics.npz` because the two are
  on different clocks — per evaluation versus per training rollout — and
  merging them would force one series to be NaN-padded against the other's
  timeline.
- **`action_abs_mean` and `action_std` in `diagnostics.npz`.** Mean |action| is
  the direct distance from the zero-action statue, which under
  `home-keyframe-residual/v1` is exactly `action = 0`. Neither existing series
  gives it: signs cancel in `action_mean`, so it sits near zero even for a
  large-magnitude policy, and `action_abs_max` is dominated by whichever single
  joint is most saturated. It separates the two ways an entropy collapse can
  end — std falls and the mean settles *on* the statue, versus std falls and
  the mean settles on some other committed pose — which are indistinguishable
  in `algo_std` alone and mean opposite things for what to do next.
### Changed
- **T-Rex stage 1 entropy now decays to zero, over 70% of the budget rather
  than 30%** (`ent_coef_end` 0.001 → 0.0, `ent_coef_decay_timesteps` 3M → 7M).
  Run `20260804_143747` trained the full 10M and finished at 2686.9 against
  the zero-action statue's 3271.8 — **588 below the do-nothing policy** — with
  unsupported duty pinned at 0.1668, 8.3× the gate ceiling (issue #486).
  The cause is measured, not guessed: `ent_coef` reached its 0.001 floor at 3M
  and held for the remaining 7M, which across 21 action dims holds the policy
  std at equilibrium ~0.375 (`algo_std` 0.47 at 5.9M, 0.375 at 8.4M). Through
  `_scale_action`'s residual mapping and this model's ctrlranges that std is a
  1σ command noise of **24.4° at the hips and 18.8° at knee and ankle**, mean
  **20.6°** across the major leg joints, resampled every control step at
  100 Hz. A policy cannot learn to hold a pose it is being commanded to shake
  by 20°; PPO optimises expected return *under* that noise, so it converged to
  a 16.7 Hz limit cycle that survives its own tremor rather than to a still
  stance.
  The target is known reachable and known still: `action = 0` **is** the home
  keyframe under `home-keyframe-residual/v1` and scores 3274.4 ± 7.6 with duty
  0.0000 (measured locally with `stance_gate_report.py trex --stage 1
  --zero-action`). So this is a convergence failure, not a reward-shaping one,
  and the usual risk of zeroing entropy — premature convergence to a bad
  optimum — is unusually low here.
  Early exploration is deliberately unchanged: `ent_coef` still starts at
  0.005, and the first 3M now decays more slowly than before. The 3M anchor
  was sized for a 6M stage and never revisited when stage 1 became 10M.

### Fixed
- **Contact is now measured at every physics substep, not one in five — the
  stance-duty gate certifies airborne time instead of a sampled duty.**
  Physics runs at 500 Hz and control at 100 Hz (`frame_skip = 5`), and every
  contact-derived quantity — the `r/l_foot_contact` info keys the
  `stance_quality/v1` gate certifies from, the bilateral-support /
  foot-load-balance / support-conditioned-alive rewards, gait-symmetry
  touchdown detection, and floor-strike termination — read only the **last**
  substep's state. The seed-43 stage-1 run exposed the consequence
  mechanically: a 20 Hz control-clock-locked hop put exactly one unloaded
  sample in every five (`bilateral_support_duty` exactly 0.8000 in all 40
  panel episodes, every duty a multiple of 1/800), so the gated 0.1958 was a
  sample-phase statistic, not airborne time — and the false-PASS direction
  was open: a policy sharpening its unloading to fall *between* the 10 ms
  samples would have read duty 0.000 and passed the 0.02 ceiling while
  unloading every cycle. Reward, obs, and gate all shared the stroboscope,
  and the run proved policies phase-lock to it.
  Both backends now aggregate inside the substep loop, in lockstep: per-foot
  **MIN** touch force across the substeps (min of per-substep per-foot
  *sums* — never per-sensor minima, which under-report when load shifts
  between pad and digits) feeds the rewards and info keys, and the first
  floor strike is latched (SB3: contact-pair latch in `BaseDinoEnv.step`;
  MJX: the height-emulation checks OR-ed per substep through the `fori_loop`
  carry). One residual asymmetry is deliberate and documented rather than
  closed: SB3's explicit site/body **height** terminations (T-Rex
  `head_tip_z`/`skull_z`, dibothrosuchus `snout_tip_z`) remain
  boundary-sampled — SB3's any-substep coverage is the *contact* latch,
  MJX's is the *height* emulation, so a transient head dip that touches
  nothing can terminate on MJX/eval while surviving SB3. The divergence
  direction is MJX-stricter (conservative for training); folding SB3's
  height checks into the substep loop is left for the plant-revision PR.
  `evaluate_policy_cpu` aggregates identically and passes the
  substep-MIN through a new `aggregated_foot_forces` override on the reward
  composers, so stage-gate evaluation scores the same quantity training
  optimizes; the override defaults to single-state semantics, which is what
  the SB3-vs-JAX parity test pins. **Observations deliberately stay on the
  boundary sample**: the obs builders are source-fingerprinted by the plant
  contract, so aggregating them would bump every species' policy interface —
  and the defect lives in what the gate and reward *price*, not in what the
  policy senses.
  `stance_duty_validation.py` gains the regime that certifies this: a
  control-clock-locked 20 Hz hop (period 5 = the aliasing regime its 2.5 Hz
  driver could never express), with per-substep kinematic ground truth
  recorded through a new step-loop probe hook and a within-control-step
  aliasing column. A regression test drives the same hop and pins both that
  `info` equals the min of the per-substep sums exactly and that steps exist
  where the boundary sample read supported while a substep was airborne —
  structurally impossible to catch before this change. The statue is
  measurably unaffected (settled-stance substep ripple is below the
  1e-6-relative equality tests' tolerance; a statue-duty test pins
  unsupported 0.0 / bilateral 1.0), but **duty histories are pre/post
  apples-to-oranges for hopping policies** — the chatterer's 0.319 and the
  seed runs' 0.1958 were boundary-sampled and would measure higher now; the
  statue-derived constants are re-derived with the passive-toes plant
  revision that follows.
- **`obs_rms_decay_on_resume` and `ramp_attr` were silently dropped by both
  JAX library entry points — and the `[jax]` table now rejects unknown keys.**
  All eight stage-2/3 TOMLs set `obs_rms_decay_on_resume = 0.01` ("lets stats
  adapt to Stage 2's velocity distribution within ~2 updates"), and only the
  notebook honored it: `run_curriculum` carried the prior stage's observation
  statistics into the next stage with their full sample count — millions —
  so `update_running_stats` was nearly a no-op exactly when the obs
  distribution shifts, which is the pathology the key exists to prevent.
  `run_curriculum` now decays the carried count through
  `decay_running_stats` with the entered stage's configured factor (same
  0.01 default and `1.0` opt-out as the notebook's resume cell).
  `ramp_attr` was likewise unread by `_JAX_KEY_MAP` and the CLI; it now
  reaches `train_jax` → `JaxTrainer`, which honors the one value the MJX
  path can ramp (`forward_vel_weight` is wired through `env.step` as a
  runtime scale; other weights are baked into the jitted reward at trace
  time) and **refuses anything else at construction** — the same contract
  the notebook's `train()` path already enforced — instead of silently
  ramping nothing.
  The mechanism that let both keys rot is closed the way `[curriculum]`
  closed it: `validate_jax_kwargs` rejects unknown `[jax]` keys fail-closed
  on both library paths ("silently dropping a misspelled threshold disables
  it"), `_JAX_KEY_MAP` is hoisted to module level as the single source of
  truth, and a test pins every mapped name against `train_jax`'s real
  signature so the map and the function cannot drift apart. One key is
  grandfathered as known-but-inert: `[jax.policy_kwargs]` is declared in
  every TOML but no JAX path reads it — the network factory is fixed —
  which is now documented at the known-key set instead of discovered.
- **`warmup_ent_coef` never reached a gradient update: `EntCoefDecayCallback`
  overwrote the stage-entry boost on the very next step.** Every PPO stage-2/3
  config sets both mechanisms, and they fought: `StageWarmupCallback` set the
  configured boost (0.02) exactly once at training start and logged "warm-up
  active … ent_coef=0.020", while the decay callback reassigned
  `model.ent_coef` from its own schedule on **every** step — from an initial
  value captured before the warm-up ran — so the boost was gone long before
  PPO's `train()` first read the attribute, and the warm-up's restore at the
  end of the window was clobbered identically. Callback order could not save
  it: a per-step overwrite beats a once-at-start set in either order. The
  documented anti-catastrophic-forgetting boost therefore never happened on
  any species' stage-2/3 transition, while the log claimed it was active.
  The two now hand off through a marker on the model
  (`ENT_COEF_WARMUP_MARKER`): the warm-up stamps it while it owns `ent_coef`
  and clears it when it restores, and the decay callback stands aside while
  it is set, capturing its decay base on the first step after release — so
  the boost holds for the whole warm-up window and the configured decay
  continues exactly as before from there. On the model rather than between
  the callbacks because the two are constructed independently, in no
  guaranteed order, in all three launch paths (CLI `train`,
  `train_curriculum`, and the notebook) — which is also why the fix needs no
  wiring change anywhere. Pinned by tests that drive the two real callbacks
  together, in both construction orders. SAC is unaffected (no decay
  callback is built for it, and its warm-up writes `log_ent_coef`);
  stage-1 runs are unaffected (no warm-up).
- **A stance-gated run that passed every gate still could not publish.**
  `result_bundle.evidence` refused any complete bundle whose stage declares
  `stance_quality/v1`, because `evaluation_selected.csv` records per-episode
  reward and length but no unsupported duty, and certifying on `min_avg_reward`
  alone would pass the zero-action statue. The refusal was correct and it also
  made the stage-1 milestone unreachable: the refusal fires only once all three
  stages have passed, so the first genuinely successful run would train for ten
  hours and then fail at finalisation. Observed on run `20260803_012355`, whose
  `collected_results.csv` and `artifact_manifest.json` stop at stage 2 while
  stage 3's own artifacts (`stage_summary.txt`, `figures/`, `replays/`, both
  evaluation CSVs) were written in full.
  The stance panel now persists its per-episode measurements as
  `stance_panel_selected.csv` beside the report it already wrote, and the
  auditor **re-derives** the verdict from those episodes rather than trusting
  the `passed` recorded in `stance_gate_report.json` — an evidence file whose
  claim is not reproducible from the episodes behind it certifies nothing, and
  run `20260802_203215` is the counterexample that motivated it. Reduction and
  scoring both go through `curriculum.stance_gate`, so the auditor cannot drift
  from the gate the trainer applied. An unmeasurable duty is written blank and
  read as unmeasured, never as `0.0` — zero is the statue's score and the best
  attainable one, so coercing would turn missing evidence into a perfect
  result. A separate file rather than new columns because the panel is a
  different evaluation from the publication one: 40 episodes at the panel seeds
  against 30 at the publication seed, and `min_eval_episodes` is the sample
  size the bound's power is specified at.
- **The JAX gate cleared its thresholds on metrics it never measured.**
  `nan < threshold` is `False`, so `jax_eval.check_stage_gate` returned
  `(True, [])` for an evaluation that produced no usable episodes — and
  `jax_setup` writes that verdict straight into `publication_gate_passed`.
  This is the same fail-open closed on the SB3 path, in the function that
  gates JAX stages 2 and 3. The rule now lives once as
  `gate_schema.finite_gate_metric` and both backends read it. Confined to
  declared thresholds: an unset rail (`-inf`) and an unset floor (`0`) could
  not be failed before and still cannot, so no legitimate config changes
  verdict.

### Changed
- **The curriculum gate verdict is now durable.** `gate_failures` was computed
  and persisted nowhere — not in `stage_summary.txt`, not in
  `collected_results.csv`, not in the `stage_result.json` allowlist — so on a
  failure the notebook disconnected the runtime and raised, and the reasons
  went with it. `stage_summary.txt` now states the verdict and every failing
  criterion, and the reasons are persisted on both backends.
  `generate_stage_artifacts` writes the summary *after* evaluating the gate,
  which it previously did not, so the summary could never contain the verdict.
- **The collapse backstop says what it is doing.** The resolved settings are
  logged once at construction, and the disarmed → armed transition is announced
  once with the peak, the floor and the drop threshold. Neither appeared
  anywhere before, which is why diagnosing runs `20260802_203215` and
  `20260803_012355` required replaying their evaluation series against the TOML
  at each run's commit. A `collapse_peak_warmup_timesteps` at or beyond the
  stage's own budget — a plausible extra-zero typo — now warns rather than
  silently disabling the backstop for the whole run.
- **The collapse backstop armed on the untrained policy and killed stage 1 at
  14.5% of its budget.** With `home-keyframe-residual/v1` and
  `log_std_init = 0`, action = 0 commands the nominal stance, so an *untrained*
  T-Rex already scores near the reward optimum. Run `20260803_012355` peaked at
  2469.4 on its **second** evaluation (100k steps) — 75% of the 3271.8
  zero-action baseline — which cleared the 0.45 × 3271.8 = 1472.3 arming floor
  on initialisation. The ordinary exploration dip that follows then read as a
  collapse: training stopped at 1.45M of a 10M budget, and stages 2 and 3 spent
  9¼ hours on the near-statue checkpoint it left. The 6M run at the same seed
  passed through that same dip to new bests at 2.8M, 3.3M and 4.55M, so there
  was nothing to catch.
  No `peak_floor` fixes this, because on stage 1 "already good" and "hasn't
  started learning" are the same number: set the floor above initialisation
  (>0.75×) and it lands above what a learning policy passes through — exactly
  how the two absolute floors documented in `collapse_settings_from_config`
  failed to arm. `collapse_peak_warmup_timesteps` is a different axis: rolling
  windows containing any evaluation before it cannot **set** the peak.
  Eligibility is by window *start*, so no surviving window straddles the
  boundary — a median absorbs one contaminating sample out of five, not five.
  While nothing qualifies, the backstop stays disarmed, which is the fail-safe
  direction; if the evaluation timesteps are unavailable it also stays
  disarmed rather than aborting a multi-hour run on a signal it cannot read.
  Expressed in timesteps rather than reward, so unlike an absolute floor it
  survives a reward-function edit. Defaults to `0.0` — every existing stage
  behaves exactly as before; only trex 1a sets it, to **1.0M**.
  Note this is **not** a tightening of `drop_fraction`/`patience`, which
  `collapse_settings_from_config` documents at length as aborting healthy runs.
  The value and the diagnosis are both measured against that run's real
  29-evaluation series: replaying the detector over it reproduces the stopping
  point of exactly 1,450,000, and any warm-up from 150k up never arms. The
  margin was **2.3%** — the first rolling-median window came to 1506.5 against
  the 1472.3 floor, and the best of the other 24 windows is 580.5, so a
  35-point difference in one early evaluation decided a ten-hour run. The
  policy was also *recovering* when it was stopped (means rise 59.4 → 350.7
  from evaluation 14 to 27) against a threshold of 753.3 anchored to an
  untrained policy. 1.0M rather than a larger value because the floor already
  blocks arming through the trough, so the warm-up only has to clear the
  initialisation spike, which is spent by 200k; 1.0M leaves 88% of a 10M budget
  under the backstop where 2.5M would leave 72%.
- **The training notebook never enforced the stance gate, and advanced a
  failing stage on its reward rail.** `notebooks/sb3_training.ipynb` computed
  `publication_gate_passed` from its own checklist over `min_avg_reward` /
  `min_avg_episode_length` / `min_avg_forward_vel` / `min_success_rate`. It
  never called `CurriculumManager.should_advance`, where `stance_quality/v1`
  is evaluated — `_build_core_callbacks` constructs no `CurriculumCallback`,
  so on the Colab path the stance gate was **dead code**. Retiring
  `min_avg_episode_length` from the trex 1a config then made the checklist's
  only other criterion vacuous (`.get(..., 0.0)`), leaving *reward alone* —
  the one threshold the zero-action statue clears by 68%, and the exact
  reading `stance_quality/v1` exists to refute. Run `20260802_203215`
  recorded `stage_passed = publication_gate_passed = True` in
  `collected_results.csv` beside its own `stance_gate_report.txt` reading
  `GATE: FAIL`, `mean_unsupported_duty 0.2120 > 0.0200` — 10.6x the ceiling —
  and started stage 2 on that checkpoint. The rule now exists once, as
  `reporting.gates.evaluate_stage_gate`, dispatched on the declared
  `gate_kind` and called from `generate_stage_artifacts`: the entry point the
  notebook and the sweep trial worker already share, so no caller can advance
  on a checklist that has drifted away from the schema. It is fail-closed at
  every branch — an undeclared or unknown kind, `none/v1`, a missing stance
  panel, a verdict scored for a different gate, and a non-boolean verdict all
  fail with a stated reason, because "we could not check" must never read as
  "it passed". Replayed against the run's own config and measurements, the
  old checklist evaluates one criterion and returns `True`; the new gate
  returns `False` naming both duty failures. Stage 2 and 3 semantics are
  unchanged: thresholds still apply only when declared, and the velocity and
  success floors only when positive. Three fail-open holes were found by
  probing the new code and closed before it shipped — a NaN metric cleared
  every floor (`nan < threshold` is `False`, measured returning
  `(True, [])`), a `reward_and_length/v1` block with no threshold set was a
  vacuously-true empty conjunction, and a `failures` string was shredded into
  one-character reasons.
- **The evaluation diagnostic no longer contradicts the gate it reports on.**
  `evaluate_stance_gate` certifies on
  `ceil(min_eval_episodes x min_full_horizon_fraction)` duty episodes — 38 of
  40 for trex 1a — but `StageGatePlateauCallback` marked the duty metric
  usable only at the full panel size, so a 39-of-40 panel **passed the gate
  while `eval_gate_met` read 0**. That is the same curve-vs-gate disagreement
  the stance scalars were fixed for, reintroduced one layer up. Both now
  share `stance_gate.required_duty_episodes`. Two smaller companions:
  `iter_replay_files` guarded `is_file()` on the nested branch but not the
  legacy one, so a directory named `*.mp4` was handed to the GCS upload for
  one layout and not the other; and the report's `plant_validated` was
  `not allow_legacy_plant`, but the rollout environment is validated
  unconditionally and the flag only relaxes the *checkpoint* check — so for
  `--zero-action`, which has no checkpoint, `False` claimed something untrue.
  It is now `checkpoint_plant_validated`, `None` when there is no checkpoint.
- **Six smaller defects in the stance gate report.** (1) Its JSON now carries
  `episode_evidence` — per-episode length, reward, duty and the two ungated
  shares — which `write_stance_gate_report`'s docstring already claimed it
  did. `run_panel` computed those and threw them away, so the file held only
  a summary that cannot be re-checked. Note this does **not** by itself lift
  `result_bundle.evidence`'s refusal of stance-gated bundles, which reads
  `evaluation_selected.csv`; that docstring now says so instead of implying
  otherwise. (2) `run_panel` closes its environment, including when the
  rollout raises — a MuJoCo env holds native handles and the artifact path
  builds one per stage. (3) Its unused `stage` parameter is gone. (4)
  `stance_report_episodes` in `[curriculum]` overrides the report's panel
  size, and `0` skips the report: the 40-episode rollout costs a few minutes
  per stage *and* per sweep trial, which a fifty-trial sweep paid fifty times
  with no way to decline. Overriding downward is logged as not certifying
  what the gate claims. (5) `--episodes 0` is rejected rather than falling
  through `episodes or min_eval_episodes` to a full-size panel while also
  skipping the under-powered warning — it read as an override that silently
  did nothing. (6) The checkpoint and the rollout environment are validated
  against the species' current plant identity, as every other artifact path
  already does; `--allow-legacy-plant` scores a checkpoint that predates the
  contract — a real use, since the script exists partly to judge finished
  runs — and the report records `plant_validated` so the flag travels with
  the number.
- **`[curriculum]` keys that were read but unregistered are now settable.**
  `diagnostics_plateau_window`, `diagnostics_plateau_min_relative_variation`
  and `supplementary_episodes` are each read with a default in the trainer,
  which means each was intended to be configurable — but `validate_gate_config`
  rejects any key it does not know, so setting one in a TOML was fatal. The
  same trap `max_checkpoints` hit. Registered alongside the new
  `stance_report_episodes`.
- **A missing `mean_episode_return` is fatal instead of silently substituting
  the per-step reward.** `min_avg_reward` is an episode-level threshold shared
  with the SB3 TOMLs — trex stage 1 sets 1950.0 — while the MJX trainer's
  `mean_reward` is the mean *per-step* rollout reward, ~3.3 for a standing
  T-Rex. Three call sites substituted one for the other with three different
  behaviours: `reward_and_length` warned and defaulted to `0.0`, the stance
  branch fell back **silently** and defaulted to `-inf`, and
  `run_curriculum`'s log line printed a third variant *labelled "episode
  return"*. Substituting is always wrong — it compares numbers three orders
  of magnitude apart — and merely happens to give the right verdict
  sometimes; against the stance rail it gives 3.3 < 1950, so the stage never
  advances and the only clue is a rail failure indistinguishable from a
  policy that genuinely threw away its return. One resolver,
  `episode_return_for_gate`, now raises `GateSchemaError` naming both numbers
  when a finite reward criterion is configured and the episode return is
  absent — matching how this module already treats a declared
  `min_avg_episode_length` with no `mean_episode_length`. The rail is
  optional for `stance_quality/v1`, so an absent return with no rail declared
  stays fine. The live path is unaffected: `jax_trainer` always emits
  `mean_episode_return`, so only a hand-written `train_fn` can reach the
  raise — which is exactly who needs to be told which key to emit.
- **The JAX/MJX publication gate no longer certifies a stance-gated stage on
  the reward rail alone.** `jax_setup.run_stage_evaluation` called
  `jax_eval.check_stage_gate`, which reads four fixed thresholds and knows
  nothing about `gate_kind` — so for `stance_quality/v1` it certified on
  whichever of the four happened to be set. On trex stage 1 that is
  `min_avg_reward = 1950` alone, since `min_avg_episode_length` was retired
  when the stance gate replaced it, and the verdict is written straight into
  `stage_results["publication_gate_passed"]`. Measured: the zero-action
  statue (3271.8) and the chattering policy the gate exists to reject
  (2133.4, duty 0.319) **both passed**. The new
  `check_stage_gate_for_config` dispatches on the declared kind and
  reconstructs the stance panel from the CPU-eval foot traces, so the JAX
  path now enforces the same criteria the SB3 one does. Reconstruction is
  valid only where each step contributed one reading per side — the biped
  case — and the guard is a measurement of the data in hand
  (`len(diag_r_foot) == sum(lengths)`) rather than a species allow-list, so
  the known `diag_r_foot`/`diag_l_foot` interleaving defect fails the gate
  closed with its reason instead of silently mis-pairing feet.
- **One unmeasurable metric no longer switches the whole evaluation
  diagnostic off.** `StageGatePlateauCallback._process_evaluation` returned
  early if *any* configured metric was short of samples. Adding unsupported
  duty to the priority list then silenced the callback entirely for the case
  it is most needed: early stage-1 training, where episodes end before
  `settle_steps` and every duty is NaN. Measured — a flat, dying run produced
  one plateau warning under `reward_and_length/v1` and **zero** under the
  stance gate, with `eval_gate_met` and `eval_plateau_active` never recorded.
  It now follows the metrics the evaluation can support and names the ones it
  could not measure in the warning, so a partial panel cannot imply the
  metric it reports is the whole story. `eval_gate_met` is still only 1.0
  when every criterion was checkable.
- **The TensorBoard duty scalar is the number the gate compares.**
  `diagnostics/eval_unsupported_duty` averaged every episode while the gate
  averages full-horizon episodes only: on a panel of 35 clean episodes at
  0.01 plus 5 early failures at 0.60 the curve read **0.084 against a 0.02
  ceiling the policy was actually clearing**. It is now built from the same
  `stance_gate` reduction the curriculum runs. The three quantities that
  actually decide advancement reached no logger at all and are now published
  as `eval_unsupported_duty_ucb`, `eval_full_horizon_fraction` and
  `eval_duty_episodes` — the last even when zero, which is the early-training
  signal that nothing survived the settling window and otherwise reads
  identically to "not evaluated". `eval_bilateral_support_duty` moved to the
  same full-horizon episode set so the shares stay comparable.
- **A failed stance gate report no longer costs a finished run its
  artifacts.** `_load_policy` raised `SystemExit`, which derives from
  `BaseException` and sailed through the `except Exception` guard in
  `_write_stance_gate_report` that exists precisely so a diagnostic cannot
  sink a run — a truncated VecNormalize `.pkl` aborted artifact generation
  before the graphs and videos were written. It now raises
  `StanceGateReportError`, and `main` converts that to `SystemExit` at the
  CLI boundary where the behaviour belongs.
- **The report's three stance shares sum to 1 again.** Bilateral and single
  support were averaged over every episode that measured anything while the
  gated duty uses full-horizon episodes only, so the identity broke whenever
  an episode failed early — measured 0.90 with 5 failures in 40 — and it
  broke in the direction that hides the problem, folding the flailing
  episodes into bilateral and single but excluding them from unsupported.
  That identity is the stated reason the report prints those shares at all.
- **`stance_gate_report.json` is valid JSON.** `json.dumps` emitted bare
  `NaN` and `Infinity` for the gate's unmeasurable sentinels — Python reads
  them back, `jq`, `JSON.parse` and Go's `encoding/json` do not — so it was
  the *failing* panel, the one worth inspecting, that produced an unparseable
  file, while the docstring promises this form exists for tooling that does
  not parse prose. Non-finite values now serialize as `null` (with
  `allow_nan=False` as a backstop); the text form still prints `inf`.

### Changed
- **The replay video now shows the checkpoint the evidence CSV is evidence
  for.** `generate_stage_artifacts` recorded it from `best_model` and labelled
  it `best`, while `evaluation_selected.csv`, the next-stage handoff, and the
  stance gate report all resolve through `_select_handoff_checkpoint`, which
  prefers the risk-adjusted `robust_best_model`. Both checkpoints normally
  exist, so a stage directory held a video and an evidence CSV describing
  **different policies**, with nothing in either name saying so — the kind of
  mismatch that makes a published figure not match the numbers beside it.
  There were four private copies of the preference order (the artifact
  generator, `build_stage_results_from_eval_data`'s recorded `model_path`
  — which is what the sweep trial worker records, with nothing else to
  re-derive it — the stance gate report, and the SB3 notebook's
  `train_stage`); all four now call the one selector, and the replay is
  labelled `selected`
  (`<species>_<algo>_stage<N>_selected.mp4`). Because that selector requires
  the matched VecNormalize statistics, two silent-wrong-answer paths close
  with it: the stance gate report no longer passes `vecnorm_path=None` when
  the statistics are missing (scoring the policy on unnormalised observations
  — a different policy — and reporting the verdict as this one's), and the
  replay is skipped with a reason rather than showing footage that
  misrepresents the checkpoint. The notebook's "robust weights, best_model's
  statistics" hybrid case now falls back to `best_model` the way next-stage
  loading always did, instead of raising.
- **SB3 keeps only the newest `max_checkpoints` periodic checkpoints
  (default 5).** SB3's `CheckpointCallback` never deletes anything: at the
  shipped `save_freq = 500_000` a 6M-step stage 1 left **twelve** 4.02 MB
  policy zips plus their VecNormalize sidecars — 48 MB of the 60 MB `models/`
  measured on run `20260801_021545`, and ~200 MB for a three-stage run — on a
  Drive mount, for files nothing in this repository reads. They exist for
  manual rollback, which needs recent history, not all of it. The JAX backend
  already capped this at `max_checkpoints = 5`; the default here matches so a
  stage keeps the same rollback depth whichever backend produced it. At that
  default the measured stage-1 `models/` goes **60.4 MB → 32.2 MB** (30 files
  → 16), a three-stage run 181 MB → 97 MB. Note this reclaims *storage*, not
  write bandwidth: all twelve checkpoints are still written and seven are
  deleted again. Writing fewer would mean raising `save_freq`, which is a
  different and lossier decision. Pruning
  matches only the periodic `stage<N>_{steps}_steps.*` pattern, so
  `best_model`, `robust_best_model` and `stage<N>_final` are unreachable from
  it by construction rather than by an exclusion list that could fall out of
  date; a pruned step takes its matched `vecnormalize_` statistics with it;
  and ordering is by the step count parsed from the filename, not mtime,
  which on a Drive mount records when the upload finished rather than when
  the checkpoint was produced. It fires on the `CheckpointCallback` cadence,
  not every step — `_on_step` runs millions of times a stage and globbing a
  Drive mount that often would cost far more than the storage it reclaims.
  Set `max_checkpoints = 0` in `[curriculum]` to keep every one.
- **A stage's generated figures and replays are grouped into `figures/` and
  `replays/`, and render to local scratch before publishing.** A stage
  directory held 20 loose entries — 5 PNGs, 6 MP4s and 2 per-frame stance CSVs
  intermixed with the config, summary and npz files — written by three
  unrelated call sites, with the only reader
  (`config.upload_curriculum_artifacts`) finding the videos through a
  hand-written `glob("*.mp4")` that would have silently stopped uploading them
  the moment either side moved. `reporting.stage_layout` now owns the layout
  for both backends, and every accessor falls back to the legacy flat location
  so the existing runs on Drive keep resolving; nothing rewrites history.
  Rendering happens in local scratch and publishes in one batched pass of
  atomic copies, because a stage directory is normally a Drive or GCS-FUSE
  mount and both matplotlib's `savefig` and mediapy's encoder write
  incrementally — every flush of a 700 KB mp4 was a separate round trip
  interleaved with the encode. On run `20260801_021545` stage 1 that is
  **130 writes against the mount reduced to 13** for the same 8.15 MB; the
  time saved is a fraction of a second, so the durability is the real gain,
  and the ~30 model and vecnorm files under `models/` remain the dominant
  Drive cost and are untouched here. It also means a runtime that dies mid-encode
  leaves the previous complete artifact set in place rather than a truncated
  video, and that the replays which *did* render still land when a later one
  raises. `evaluations.npz`, `diagnostics.npz` and `evaluation_{selected,final}.csv`
  deliberately do **not** move: the first two are written on the training hot
  path and read by a dozen call sites including the sweep tooling, and the
  evaluation CSVs are the publication evidence contract `result_bundle.audit`
  names by fixed relative path. Filenames inside `replays/` are unchanged and
  still carry the redundant `<species>_<algo>_stage<N>` prefix the path already
  states; dropping it is a separate rename.
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
