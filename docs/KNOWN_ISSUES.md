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
  evaluations. The JAX CLI checks reward and episode length once after a full
  stage and ignores `min_avg_forward_vel` (a defect, not a deliberate
  divergence: see "the JAX command-line curriculum advances a stage without
  checking..." under Training / RL); it never evaluates the final stage's
  gate, so `min_success_rate` is never consulted. The JAX notebook checks
  reward, episode length, velocity and success once on a CPU evaluation.
  These paths are documented but not behaviorally equivalent.
- **PPO advantage normalization** — per-minibatch in JAX vs per-batch in
  SB3; acceptable, documented in `jax_ppo.py`. (June §2.7)

Targeted tests now pin the shared NumPy/JAX Velociraptor natural-lean posture
primitive and its per-path runtime routing. A comprehensive per-component
SB3↔JAX reward **parity test** (one fixed state, assert every component within
tolerance) remains the standing recommendation for the divergences above.
(June §6.8; see the
[Stage-1 basin investigation](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md))

## Training / RL

<!-- The two items below come from the 2026-08-05 stage-1 bounce
     investigation and its addenda; full evidence in
     investigations/TREX_STAGE1_BOUNCE_2026_08.md. Three sibling items
     (actuator saturation unopposed; a filter cannot be retrofitted; the pose
     needs continuous feedback) were deleted 2026-08-15 after the r7/r11
     campaign fixed or falsified each — see the CHANGELOG entry of that date
     for where their evidence now lives. -->

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
  **Update (2026-08-15):** every number above is r6-era; the plant, interface,
  and reward have all moved since (r7 passive toes, r11 10 Hz command filter,
  the #504 shaping pack), and the campaign ran three more 10M runs — two gate
  FAILs by different mechanisms (18.7 Hz foot chatter at duty 0.41; a
  knee-locked crouch at 0.40) and then the first certified PASS (duty 0.0048,
  UCB 0.0080;
  [investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md](investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md)).
  The ask stands unchanged with the target moved: seed replicates of the
  **r11 gate-pass configuration**, which is n = 1.
  **Update 2 (2026-08-15, evening):** the first replicate is in — seed 43,
  run `20260815_014118`, identical configuration: gate **FAIL** at duty
  0.0597 / UCB 0.0747, full-horizon 0.925, 0 of 200 panels passing, despite
  a healthy 3093 panel reward. Same three-act trajectory and the **same
  stance** (unsaturated, near-home, tail-live) — the act-3 re-descent
  stalled at a broadband-noise floor (AC 0.329 vs seed 42's 0.135) instead
  of quieting. **n = 2: 1 pass / 1 fail — the configuration is
  seed-sensitive.** A third seed is queued; full record in
  [investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md](investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md).
  **Update 3 (2026-08-16):** seed 44 (run `20260815_205206`) — **PASS**,
  the strongest yet: full-horizon 40/40, duty 0.0069 / UCB 0.0117, reward
  3408.3 ± 88.5 (97.5% of the statue), AC 0.132, same unsaturated stance.
  **n = 3: 2 pass / 1 fail**, and the split tracks the post-anneal noise
  floor exactly — both passes quieted to AC ≈ 0.13, the one fail stalled
  at 0.33. The open question is no longer whether the configuration can
  pass (it usually does) but what decides the anneal's endpoint; the
  seed-43 postmortem's candidate responses stand.
- **LOW** — **stage 1a (stance) contains no in-episode disturbance, so a
  stance-gate PASS certifies stance quality, not active balance control.** The
  only perturbation is joint-angle noise at reset (`reset_noise_scale 0.05`).
  That is by design since the 1a/1b split: the disturbance lives in the
  recovery stage (`configs/trex/recovery.toml` — stance's `[env]` plus the
  scheduled pushes, warm-started from the certified stance checkpoint), whose
  `recovery_quality/v1` gate was frozen 2026-08-28 with measured thresholds.
  The pre-split `--impulse-probe` envelope (2026-08-06: 0.50 m/s one way,
  0.00 the other) is superseded by the recovery records —
  [STAGE1B_IMPLEMENTATION_PLAN.md](STAGE1B_IMPLEMENTATION_PLAN.md) and
  [investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md](investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md).
- **MEDIUM (operational)** — **every checkpoint trained before the Phase C
  interface revision is refused as a trunk, every `gate_resolution.json`
  frozen before it is stale, and the two recipes below (the D-A22 re-judge
  and the D-B16 republish) are DEAD for pre-Phase-C runs — the only path is
  the command-line `widen_checkpoint` into a NEW run id that the notebook then
  judges (decision D-D14; the notebook's widen cell and knobs, which the two
  widen sessions used, left with consolidation PR-14a); the one certified
  stance parent that needed widening (`20260815_205206`, seed 44, an r11
  archive two revisions behind r13, hence a revision gap of 2, decision
  D-C17) was widened on 2026-09-20 as `20260920_010912`; seed 42
  is already certified at r13 by `20260914_123816`** (BEHAVIOR_RECIPES_PLAN
  §4.6, decisions D-C8–D-C14 and D-C17; PLANT_CONTRACT.md "Widening a
  checkpoint across a policy-interface bump"). Phase C appended a 3-dim command segment to every
  species' observation (`policy_interface_revision` trex 12 → 13, velociraptor
  9 → 10, brachiosaurus 7 → 8, dibothrosuchus 6 → 7, compsognathus and
  compsognathus_robot 1 → 2). The task payload carries the plant's
  `policy_interface_sha256` (`task_fingerprint.py`, the `plant_identity`
  section), so every stage's `task_sha256` moved with it and reuse rule 3
  (the verdict's `task_sha256` must equal the fingerprint derived from the
  CURRENT stage config; `environments/shared/ancestors.py`) refuses every
  pre-Phase-C verdict; were the hash to match, rule 6 (the checkpoint's
  recorded plant identity must validate against the current plant, no legacy
  allowance) refuses the pre-Phase-C archive too (rules are evaluated 1, 2,
  3, 7, 4, 5, 6, so a pre-Phase-C candidate never reaches 6). As with rule 7
  the refusal is a
  log line and the node then TRAINS in the new run. Every existing recovery
  `gate_resolution.json` — the 2026-08-28 trex freeze included — records a
  pre-Phase-C `task_sha256`, and `require_gate_resolution` refuses a
  resolution whose recorded task differs from the current one ("Recalibrate —
  re-measure the null panels under the current task"), so no recovery run
  under the new plant can consume one: each is re-frozen from a widened
  handoff. The two recipes below are dead for pre-Phase-C runs for two
  concrete reasons: the re-judge path (set `RUN_ID` to the old run, remove
  the refused verdict, let the JUDGE branch re-panel) dies first in the
  storage cell — `initialize_result_bundle` compares the existing
  `provenance.json` against this session's identity, `plant_identity`
  (now r13) included, and raises `run directory already belongs to a
  different run: {'plant_identity': …}` before the infra cell exists (the
  missing `certification_panel` role refuses these two runs the same way,
  D-B17) — and with `provenance.json` removed to get past that, the JUDGE
  branch's `evaluate_stage_checkpoints` loads `<stage_label>_final.zip` and
  calls `validate_model_plant(model, PLANT_IDENTITY, ...)` against the
  checkout's r13 identity, so the pre-Phase-C archive is refused before any
  panel rolls (and `backfill_gate_verdict.py --force` could at best mint a
  verdict under the old task hash, which rule 3 refuses); the republish path
  (remove
  `provenance.json`, set `RUN_ID` to the run, re-run the setup and
  publication cells) dies in the audit, because the storage cell mints the
  provenance with `current_plant_identity(SPECIES)` (r13) and
  `validate_result_bundle` compares every stage config's recorded
  `plant_identity` against it (`stage <N> config plant_identity does not
  match provenance.json`). The remedy is a NEW run, widened on the command
  line and judged in the notebook (decision D-D14): `python -m
  environments.shared.scripts.widen_checkpoint --species <species> --stage
  stance --from-stage-dir <parent run>/<its stance directory> --to-stage-dir
  <LOG_BASE>/<species>/<algo>/<new run id>/01_stance [--max-revision-gap N]
  [--label L]`, with `<new run id>` a new timestamp id `YYYYMMDD_HHMMSS`, run
  before the notebook's storage cell has opened that run id (on Colab, the
  three steps of
  [PLANT_CONTRACT.md](PLANT_CONTRACT.md#widening-a-checkpoint-across-a-policy-interface-bump):
  section 1, a scratch cell that mounts Drive, never the storage cell, then
  the tool from `/content/mesozoic-labs`), widens the parent's certified stance handoff
  into the new run's root stage directory (zero columns for the new dims, the
  run block's `WIDEN_LINEAGE_KEYS` with the parent's `run.seed`, the identity
  and task fingerprint re-stamped, no verdict, no `provenance.json`); then the
  notebook, with `RUN_ID` set to the new run id, `SEED` to the parent's
  recorded `run.seed` and `TRUNK_FROM = ""`, mints an r13 provenance for it
  (the storage cell refuses any other `SEED` before it writes anything,
  D-C14: the provenance publishes `training_seed = SEED` and replication
  counts distinct seeds; the resolve cell refuses a trunk until the widened
  root holds a verdict, D-C13), and the chain loop refuses — loudly — to
  reuse the verdict-less directory, finds the `<stage_label>_final.*` pair
  and JUDGES it: a fresh 40-episode panel (seeds 3042–3081) under the
  current gate, a `gate_verdict.json` minted under the new task hash. Never by pointing `RUN_ID` at the old run: for a
  pre-Phase-C run the storage cell refuses the directory outright (above),
  and for a same-plant run whose verdict rule 3 or 7 refuses the chain loop
  raises "mint a fresh RUN_ID" (D-C13).
  Inventory: the two certified stance parents are `20260810_145546` (seed
  42) and `20260815_205206` (seed 44) — and BOTH are policy-interface **r11**
  archives (`sha256:96ef13…`;
  [investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md](investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md),
  [investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md](investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md)),
  not r12: trex went r11 → r12 on 2026-08-16 (commit `8795e28`, the
  perturbation engine entering the interface fingerprint; `observation_dim`
  stayed 61, physics r7 and `action_dim` 15 unchanged, no tensor moved) after
  both had trained, and an archive's identity is stamped at training time.
  The 2026-09-17 Drive survey
  ([investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md))
  adds two facts: neither r11 parent holds a `gate_verdict.json` in its
  `stage1/` directory (the 2026-08 verdicts live only in the run-level
  records), and widening does not need one — `widen_checkpoint` reads the
  parent's `stage_config.json` run block, its handoff pair and the stamped
  VecNormalize sidecar, and a parent verdict is optional (it must record
  `passed = true` only if present), so no backfill precedes a widen; and
  seed 42 no longer needs widening at all, because the fresh r13 run
  `20260914_123816` certified its stance (and its locomotion) on
  2026-09-15, which leaves the seed-44 parent `20260815_205206` as the one
  remaining widen candidate.
  `widen_checkpoint`'s gate admits a parent at most `max_revision_gap`
  interface-only revisions behind (`identity_gate_errors`: `1 <= current -
  parent.policy_interface_revision <= max_revision_gap`; the default 1 is the
  Phase C bump alone, fail closed), so under the default it refuses both
  parents with `policy_interface_revision: parent=11, current=13 (gap 2
  exceeds max_revision_gap=1; pass --max-revision-gap 2 / max_revision_gap=2
  …)`, and `--allow-legacy-plant` does not apply (it covers only an archive
  with NO recorded identity). **The remedy is decision D-C17**: pass
  `--max-revision-gap 2` to the command-line tool (`max_revision_gap=2` in
  the API; session 1 set the notebook's revision-gap knob to 2, a knob that
  left with consolidation PR-14a, D-D14). Every other gate field is checked
  whatever the bound — same species, `physics_sha256`, `nq` / `nv` / `nu`,
  `action_dim` and `observation_dim + 3 == current` — so only fingerprint-only
  intermediate bumps can be crossed, and opting in asserts, from
  `configs/plant_versions.toml`'s numbered notes, that the crossed bumps
  changed nothing the widening cannot bridge (for trex r11 → r13: note 11
  recorded the perturbation engine, note 12 appended the command segment).
  The widened stage's `widen_report.json` then records `revision_gap: 2` /
  `max_revision_gap: 2` and its run block
  `widened_from_policy_interface_revision: 11`; a hand re-stamp of the archive
  is never the path. The three r12 stance PASSes on the log tree —
  `20260816_180603`, `20260817_165515`, `20260818_134249`
  ([investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md](investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md)
  §2.1) — carry r12 identities and widen under the default bound, but they
  are not the certified, published parents; widening one is a first
  certification under the new hash with its own `run.seed`, not a re-panel
  of a certificate. After the first certified parent is
  widened and re-paneled trex stance publishes as `1 run of 2 seeds;
  provisional`
  (replicates are discovered among siblings judged under the SAME
  `task_sha256`, so the un-widened sibling does not count), and it reads
  `2 runs of 2 seeds` only once BOTH are widened and re-paneled in two
  sessions, each with `SEED` set to its parent's seed (42, then 44) —
  since the 2026-09-17 survey the seed-42 slot is already filled by
  `20260914_123816`, so one widen session (`SEED = 44`) closes the bar, and
  it did: session 1 of [NEXT_STEPS.md](NEXT_STEPS.md) §3 ran on 2026-09-20 as
  `20260920_010912` (re-panel identical to the r11 certificate, recovery
  certified 2026-09-21, bundle `complete` with trex stance at replication 2).
  The outcome table is
  [investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md](investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md)
  (its §1–§3 seed-44 columns filled, §7 and §8 appended; its §6, appended
  2026-09-19, marks Session 1 (the seed-42 widen) and Session 2 superseded by
  `20260914_123816`). Three neighbours: an
  in-flight Ray sweep experiment cannot be resumed under the new plant — the
  sweep notebook validates the recorded `plant_identity.json` of the sweep
  and experiment directories against the current identity on resume
  (`validate_recorded_identity`) and trials are not widened (sweeps stay
  trunk-only, plan A6) — so it restarts under a new experiment directory.
  JAX checkpoints are not widened: `jax_checkpoint.load_checkpoint`
  validates the recorded identity against `current_plant` and has no widen
  path, so a pre-Phase-C JAX checkpoint fails closed (plan A7: SB3 is the
  evidence backend). The two compsognathus recovery calibrations were
  restamped, not re-measured, which moved their `profile_sha256`, so every
  compsognathus / compsognathus_robot recovery freeze made before Phase C is
  refused and must be re-frozen from the restamped profile
  (`environments/compsognathus/RECOVERY_CALIBRATION.md`).
- **MEDIUM (operational)** — **SB3 archives are bound to the interpreter that
  saved them; only `policy_loading.load_sb3_model` opens one safely, and the
  Colab image moves without notice.** SB3 stores a model's `learning_rate`,
  `lr_schedule` and `clip_range` members through cloudpickle. A closure — the
  `linear_schedule` every stage TOML with `learning_rate_end` produced before
  2026-09-19, and the nested schedule functions older SB3 releases built for a
  float learning rate — is pickled by
  value with its code object, and `PPO.load` / `SAC.load` execute it while
  rebuilding the optimizer (`_setup_model` → `lr_schedule(1)`). Bytecode
  compiled by Python 3.12 run by 3.13, or the reverse, segfaults the process
  with no Python traceback: reproduced both ways in the review container
  with torch held constant at 2.14 (`faulthandler` places the crash in the
  closure body, `train_base.py` `linear_schedule.<locals>.schedule`, called
  from SB3's `FloatSchedule.__call__` inside `policies._build`), so torch is
  not the cause the 2026-08-28 note
  ([investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md](investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md)
  §9, "Python 3.13 / torch 2.11") took it for. Merely unpickling the data
  (what the widen tool's archive read does) does not crash; calling the
  schedule does. **The incident:** Colab's L4 image moved from Python 3.12.13
  / numpy 2.0.2 / jax 0.7.2 (the r11 parent `20260815_205206`, 2026-08-15)
  to Python 3.13.15 / numpy 2.1.3 / jax 0.11.1 (already the image of the
  runs of 2026-09-14/15, `20260914_123816` and `20260915_160239`). Two
  trex seed-44 widen sessions of 2026-09-18 at `22c1fc8` (runs
  `20260918_230155` and `20260918_230335`, on Drive, recorded 2026-09-23)
  were the first to load an archive saved under the older image and left
  the same four-file stray, and both first attempts at NEXT_STEPS.md session 1 (runs
  `20260919_170528` at `22c1fc8` and `20260919_190251` at `ab35dbd`,
  `BEHAVIOR="stand"`, `WIDEN_FROM="20260815_205206"`,
  `WIDEN_MAX_REVISION_GAP=2`, `SEED=44`) died with
  `AsyncIOLoopKernelRestarter: restarting kernel` right after the widen tool
  wrote `robust_best_model.zip`, `robust_best_model_vecnorm.pkl`,
  `stage1_final.zip` and `stage1_final_vecnorm.pkl` into `01_stance/models/`
  and before `widen_report.json`, `plant_identity.json`,
  `task_fingerprint.json` or `stage_config.json`: the self-verification's
  first load of the r11 parent, saved under an earlier image, was a bare
  `alg_cls.load`. The notebook's `PPO.load` preflight of the time ran in the
  infrastructure cell, after the widen cell, on a throwaway model saved by
  the same interpreter, so it protected nothing. **Fixed on 2026-09-19 (the
  loader change; CHANGELOG "Fixed"):** every archive load in the repository
  and both notebooks goes through `load_sb3_model`, which supplies the
  schedule members through SB3's `custom_objects` instead of unpickling
  them and refuses an archive whose other members carry another
  interpreter's bytecode; `linear_schedule` / `cosine_schedule` are
  picklable-by-reference classes, so archives saved from now on carry no
  bytecode; the widen tool re-states a parent's schedules from its recorded
  `hyperparameters` block, or from the current stage config's block when the
  parent's `stage_config.json` predates that block (`widen_report.json`
  names the source; the r11 parent's carries the block, and its 2026-09-20
  widen report reads `parent_stage_config`); and the notebook's load preflight runs right
  after the resolve cell on a real archive, the trunk run's root handoff when
  there is one (until consolidation PR-14a moved widening to the command line,
  the notebook's widen parent's handoff first)
  (`test_policy_loading.py`, with fixture archives saved under 3.12 and
  3.13). **What stays true and is why this entry stands:** every archive on
  Drive trained before that date — every trex and compsognathus stage
  checkpoint, both r11 stance parents and the compsognathus r1 parent
  included — embeds its saving interpreter's bytecode in those three
  members for good, so any path outside the loader (an ad-hoc `PPO.load` in
  a notebook cell, the SB3 CLI, a third-party tool, an older checkout) still
  dies on a foreign image with no traceback; a run's `provenance.json`
  `python_version` names the interpreter its archives belong to, and the
  image will move again. The four stray run directories `20260918_230155`,
  `20260918_230335`, `20260919_170528` and `20260919_190251` hold only
  `provenance.json` and four unverified
  model files whose archives re-pickled the parent's 3.12 bytecode under a
  3.13 `system_info.txt`, with no report, identity or fingerprint; nothing
  can reuse them (`select_trunk` lists them as refused); session 1 re-ran
  beside them on 2026-09-20 without harm, and they are still to be deleted
  (housekeeping in [NEXT_STEPS.md](NEXT_STEPS.md) §3).
- **MEDIUM (operational)** — **run-level records go stale when a run is
  continued in a later session.** The certified r13 trex walker
  `20260914_123816` shows it: its run-level `summary.json`,
  `provenance.json` and `artifact_manifest.json` were written 2026-09-15
  02:08 UTC, after the stance verdict (judged 01:57 UTC), and the
  locomotion verdict judged 11:39 UTC the same day was never folded in
  (only `training_summary.txt` was refreshed), so the run-level records
  under-report the run as stance-only. Nothing canonical breaks: reuse
  reads the per-node files (`gate_verdict.json`, `task_fingerprint.json`,
  `plant_identity.json`, `stage_config.json`), which are written the moment
  a node is judged, so `TRUNK_FROM = "auto"` still selects the run and
  reuses both nodes for a hunt session, whose chain runs through both
  (2026-09-17 Drive survey,
  [investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md));
  a stand or walk session considers stance only, which the newer
  `20260920_010912` also covers since 2026-09-20, so the tie goes to it.
  Remedy for that run: none in place (corrected 2026-09-23). Its bundle is
  `complete` under a stance target (`target_deliverable "1"`), not
  `partial`; a re-entry that reuses every node never reaches the chain
  loop's `save_run_bundle`, and a direct save is refused because
  `03_locomotion/` appeared after the publication (a complete bundle is
  immutable, [RESULT_BUNDLES.md](RESULT_BUNDLES.md); since consolidation
  PR-14a the notebook refuses, before anything is trained or written, a
  session that would judge or train a node into a complete run). The
  records stay stance-only, with stance at replication 1, although the
  seed-44 sibling `20260920_010912` counts this run at 2; nothing reads them
  for reuse. Remedy in general: a `partial` bundle is rebuilt by the next
  session that trains or judges a node in the run; PR-14a's complete-run
  refusal does not change this
  ([CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) §8).
- **LOW (operational)** — **a re-entry that only reuses a `partial` or
  `failed` run's nodes stops at the bundle-verification cell (verified
  2026-09-24).** In a new runtime the storage cell's
  `initialize_result_bundle` appends the session to the run's
  `provenance.json` (`sessions`), and the zero-action cell refreshes the
  run's `zero_action_baseline.json`; the run's `artifact_manifest.json`
  declares `provenance.json`. With every chain node reused in place nothing
  rebuilds the bundle (only a node trained or judged in the run does), so the
  verification cell's (§8, "Verify the result bundle"; §10 "Cleanup" before
  #558) `validate_result_bundle` raises
  (`artifact size mismatch: provenance.json; summary provenance does not
  match provenance.json`) and "Run all" stops before the auto-disconnect.
  Verified against the library: `initialize_result_bundle` from a new
  process on a run sealed `partial` makes `validate_result_bundle` refuse
  it, while on a run sealed `complete` it records no session and the bundle
  still verifies (and since consolidation PR-14a the zero-action cell keeps
  a complete run's copy). Nothing certified is touched, and the next node
  trained or judged in the run rebuilds the bundle over both files. Remedy:
  run the auto-disconnect cell (§9) by hand.
- **MEDIUM (operational)** — **the best and robust-best handoff pairs are
  written straight to the mount, and one a reclaim cuts short leaves the node
  unjudgeable (reproduced 2026-09-26 at the #558 follow-up, #559).**
  On a Drive/GCS mount `train()` stages only the periodic pairs locally and
  publishes them atomically (`_build_core_callbacks`, `train_base.py:726-760`).
  SB3's `EvalCallback` saves `best_model.zip` into `models/` directly
  (`best_model_save_path`, :710) and `SaveVecNormalizeCallback` then its
  sidecar; `RobustBestModelCallback` saves `robust_best_model.zip`, then its
  sidecar, the same way (`curriculum/checkpoints.py:80-84`). The final pair is
  written the same way (`_save_final_and_sync_tb`, `train_base.py:918-932`),
  but since #559 the RESUME cell and the chain loop check it with
  `curriculum.checkpoint_pair_problem`. Nothing checks the handoff pairs:
  `select_handoff_checkpoint` (`curriculum/checkpoints.py:96`) tests only
  that both files exist, and the JUDGE branch's `evaluate_stage_checkpoints`
  loads the selected pair after evaluating the final one
  (`reporting/stage_artifacts.py:1292-1312`). With an intact final pair and
  `robust_best_model.zip` cut in half, the RESUME cell prints "Nothing to
  resume", the selector still returns `robust_best_model`, and
  `load_sb3_model` raises `AssertionError: No data found in the saved file`,
  so every Run all fails in JUDGE with a bare load error. A resume does not
  repair it: `seed_resume_eval_state` (`train_base.py:1344`) seeds the best
  trackers from `evaluations.npz`, so the pair is rewritten only when a later
  evaluation beats the old best. A reclaim between a zip and its sidecar can
  also pair a new zip with the previous best's sidecar, which no integrity
  check can see (read from the write order, not reproduced). Workaround:
  before judging, run `checkpoint_pair_problem` over `robust_best_model.*` and
  `best_model.*`; moving a broken pair aside lets the selector fall back to
  the other (the verdict then binds that checkpoint), otherwise train the node
  again in a fresh `RUN_ID`. Plan: stage the final and best pairs like the
  periodic ones (CU-3 in [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md);
  names and bytes are unchanged, so no digest moves). (#558 reviews)
- **LOW (operational)** — **nothing on disk records the trunk a session
  resolved, so a resume must re-supply it by hand, and a wrong one fails late
  or discards the resumed node (read from the code at the #558 follow-up,
  #559).** `TRUNK_FROM = "auto"` is resolved in the kernel (`select_trunk` in
  the resolve cell, D-A25) and only printed. Each reused
  node leaves `ancestors/<id>/ancestor.json`, but that names the run that
  certified the node after following records (`_ancestor_record`,
  `ancestors.py:648-671`), not the trunk, and the chain loop never follows
  this run's own records (`follow_records=candidate is not RUN_DIR`). So the
  resume recipe (section 5) has the operator pin `TRUNK_FROM` to the trunk the
  interrupted session printed, or derive it from the nearest ancestor record.
  A wrong trunk goes one of three ways. (a) `"auto"` picks a newer run
  holding a different certified copy of a reused ancestor: `record_ancestor`
  refuses ("a run cannot reuse two parents for one node",
  `ancestors.py:694-704`), but only in the chain loop, after the RESUME cell
  has trained the remaining budget; re-running with the right trunk then
  judges the node. (b) `"auto"` picks a newer run that certifies the resumed
  node itself (an ancestor of `BEHAVIOR`'s target, since the target is looked
  for only in this run): the loop reuses that copy, prints it as a reuse, and
  never judges the resumed node. (c) `TRUNK_FROM = ""`: ancestors this run
  holds only as records are trained again here (found by the review of
  #559's first round, `ee91f51`; review 2 in CLEANUP_PLAN_2026_09.md §5.4). A node `RETRAIN_FROM` covered has the shape of (b); #559
  refuses its resume and names the route (`BEHAVIOR` set to that node, then a
  fresh `RUN_ID` trunked from this run). Plan: two
  pending decisions in [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §2,
  a run-level record of the resolved trunk that the RESUME cell reads, and
  judging an unjudged `RUN_DIR` node before any trunk reuse. (#558 reviews)
- **MEDIUM (operational)** — **`train --load <checkpoint>` (default
  `--load-mode resume_same_stage`) writes into a stage directory that already
  holds `gate_verdict.json` (guard executed 2026-09-26).** The D-A20 guard
  `config.refuse_occupied_stage_dir` (`config.py:403-427`, called at
  `train_base.py:1143`) lets any same-stage resume through. Against a
  directory holding `stage_config.json` and a passed `gate_verdict.json`, it
  returns for `resume_same_stage` and raises only for `initialize_next_stage`
  or no load. `train()` has no complete-bundle refusal either. Read from the
  code, such a resume re-saves `stage_config.json` and always rewrites the
  final pair. It rewrites the handoff pair the verdict hashes whenever a
  post-resume evaluation beats the seeded best. When that happens the verdict
  no longer matches its checkpoint, reuse rule 5 refuses the node, and a
  `complete` bundle stops verifying. The SB3 notebook refuses this since D-D16
  (its RESUME cell trains nothing for a node that holds a verdict); the CLI
  and the library do not, and the website's recipes and quick-start pages
  state the resume exception without the judged case. Workaround: never point
  `--output-dir` at a judged stage directory; a new attempt is a new run
  directory. Plan: D-D16 left the library guard open because it amends D-A20;
  pending in [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §2. (#558
  reviews)
- **MEDIUM (operational)** — **every `gate_verdict.json` written before the
  gate-configuration digest (decision D-A22, Phase B WS-B3) is refused as a
  trunk until it is re-judged.** Reuse rule 7 compares the verdict's
  `gate_sha256` (the digest of the gate it was judged under) with the gate
  the reusing run declares; a verdict without the field certifies an unknown
  gate and `--trunk-from` / `TRUNK_FROM` refuses it naming the re-judge
  paths — and then TRAINS the node in the new run (the refusal is only in
  the log: `Not reusing 'stance' from --trunk-from ...: ... Training it in
  this run instead.`), so the inventory below must be done before the first
  trunked run or the stance retrains for hours. **Superseded for every
  pre-Phase-C run (2026-09-14): the two re-judge paths that follow, and the
  digest measurement at the end of this entry, are dead for a checkpoint
  minted under the previous policy interface — see the Phase C entry above;
  the widened copy is re-paneled under the current gate in a new run, which
  settles the digest question for it. The text is kept as history.** Two
  re-judge paths existed:
  the notebook JUDGE branch / `generate_stage_artifacts` for a directory
  holding no verdict (set `RUN_ID` to the run, remove the refused
  `gate_verdict.json` from the stage directory, and the chain loop judges
  the held checkpoints under this session's gate, measuring a fresh stance
  panel), and `backfill_gate_verdict.py --force` for one that holds a
  pre-D-A22 verdict. This covers the certified trex stance run `20260810_145546` and the
  seed-44 replicate `20260815_205206` (assumed backfilled under decision D-A6
  before the digest existed; the 2026-09-17 Drive survey found no
  `gate_verdict.json` in either `stage1/`, see the Phase C entry above), and
  every other backfilled or Phase-A-judged stage directory. Inventory the
  log tree with
  `find <LOG_BASE> -name gate_verdict.json -exec grep -L gate_sha256 {} +`
  and re-backfill each hit with
  `python -m environments.shared.scripts.backfill_gate_verdict <stage_dir> --force`;
  when the rule-7 refusal then names a differing threshold — stance's
  `min_avg_reward` rail moved 1940.0 → 2100.0 on 2026-08-10, the day
  `20260810_145546` ran, so its recorded block may not digest to the
  checkout's — the directory must be re-judged under the current gate (a
  moved rail or a statue re-measure is a RE-JUDGE of every certified trunk,
  never a retrain; decisions D-B7/D-B8). For a `reward_and_length/v1`
  directory that is `--gate current`, which re-aggregates the evidence rows
  under the checkout's block and records that gate. For a stance directory
  it is NOT: a `stance_quality/v1` verdict is read off
  `stance_gate_report.json`, which certifies only the thresholds it scored,
  so the tool refuses a report scored under the old rail under either
  `--gate` (a pass at 1950 under the 1940 rail must never be minted as a
  pass under 2100) — re-judge both stance directories through the notebook
  JUDGE branch, which measures a fresh panel under the current gate.
  Recovery verdicts cannot be backfilled and need a notebook re-roll. Measurement still owed before the first Phase-B
  trunked run: for both stance directories compute
  `gate_config_sha256(gate_config_view(stage_config.json["curriculum"]))`
  and compare it with `load_all_stages("trex")[1]["curriculum_kwargs"]`'s
  digest, recording which of the two backfill paths each needed.
- **MEDIUM (operational)** — **pre-Phase-B stance bundles need republishing
  with the `certification_panel` seed role, and pre-D-A22 siblings join a
  deliverable's replication count only after re-backfill** (decisions
  D-B16/D-B17, Phase B WS-B4; plan §4.5). The publication audit now binds
  `stance_panel_selected.csv` row `i` to `seed_roles.certification_panel + i`
  and a recovery `gate_resolution.json`'s `decision_procedure.panel_seed_start`
  to the role, and REFUSES a recorded `stance_quality/v1` pass whose
  provenance declares no such role. No committed bundle is affected, but
  every Drive bundle published before this change — the certified trex
  stance run `20260810_145546` (the bundle
  `TREX_STAGE1_GATE_PASS_RUN_2026_08.md` audits) and the seed-44 replicate
  `20260815_205206` — carries a `seed_roles` without it and fails its next
  audit or republish; `initialize_result_bundle` also refuses to resume such
  a run directory from the new notebook (the seed roles are identity, and
  the mismatch reads as "already belongs to a different run"). **Superseded
  for every pre-Phase-C run (2026-09-14): the republish recipe that follows
  is dead for these two runs — the storage cell now mints an r13 provenance
  the audit rejects against their r11 stage configs; the seed-44 run is
  widened into a new run instead and the seed-42 widen is superseded by the
  fresh r13 run `20260914_123816` (the Phase C entry above), whose bundles carry the
  `certification_panel` role from the start. Kept as history.** To
  republish a bundle under the SAME plant, one would:
  remove the run's `provenance.json` (a regenerated artifact — the manifest
  may disagree only on those), set `RUN_ID` to the run and re-run the setup
  and publication cells, which re-capture the provenance with the role under
  the same run id and re-audit the stance panel against it. Replication is
  writer-recorded from `LOG_BASE/<species>/<algo>/` siblings whose verdict
  carries `gate_sha256`, so the seed-44 run counts as the certified run's
  replicate (and vice versa) only once BOTH directories are re-backfilled
  per the D-A22 inventory above and the counting run's publication cell is
  re-run with the sibling present (a `partial` bundle is rebuilt; a
  `complete` one regenerates its derived artifacts when the replication
  record is the only change, and stays immutable in everything else). Until
  then trex stance
  publishes at n = 1 against `certification_seeds = 2` and the catalog
  labels it provisional — which is the honest reading of KNOWN_ISSUES'
  own 2 pass / 1 fail record. Overtaken for the r13 pair on 2026-09-21: the
  widened seed-44 run `20260920_010912` and the fresh seed-42 run
  `20260914_123816` share the r13 stance `task_sha256`, the seed-44 bundle
  records replication 2, and the seed-42 run's own records stay at
  replication 1: its `complete` bundle cannot be rewritten in place (the
  MEDIUM entry "run-level records go stale" above; a complete bundle is
  immutable, [RESULT_BUNDLES.md](RESULT_BUNDLES.md)).
- **MEDIUM (provisional threshold)** — **the trex hunting bar
  `min_success_lcb = 0.5` in `configs/trex/behavior.toml` is PROVISIONAL
  (decision D-B2, Phase B WS-B2).** It was frozen BEFORE any Phase-B pilot,
  on the strength of the schema-2 committed result (`results/trex/ppo/
  summary.json` stage 3: mean 0.9667 → 29/30 → LCB 0.8514, a recomputation
  from a rounded mean with no per-episode evidence, judged at
  `min_success_rate` 0.25 with no velocity term under a different `[env]`
  block) — freeze-then-revise, not the pilot-then-freeze precedent the
  recovery gate set. Follow-up: after the first hunting pilot under
  `task_success/v1`, re-freeze the bar attainable-not-aspirational from its
  `evaluation_selected.csv` (20/30 clears 0.5 at 0.5006; 19/30 does not),
  update the TOML comment and plan §4.4, and re-judge — never retrain — any
  verdict minted under the provisional bar (a moved threshold is a re-judge
  under D-B7/D-B8).
- **LOW (calibration owed)** — **`collapse_peak_warmup_timesteps =
  1_000_000` on the trex behavior stage is a judgment, not a measurement
  (decision D-B5).** Stance's 1.0M was measured by replaying run
  `20260803_012355`'s evaluation series through the detector; no
  behavior-stage series under the current config exists to replay, so the
  value is the stance/recovery number bounded below by the 600k stage-entry
  window (`warmup_timesteps` + `ramp_timesteps`, where a warm-started walker
  evaluates near the 602.13 statue and would arm the 270.9 floor at once)
  and above by the ≤ 0.5× budget pin in `test_curriculum_early_stopping.py`.
  Follow-up: replay the first full hunting run's `evaluations.npz` through
  `EvalCollapseEarlyStopCallback` (as STAGE1_SPLIT_PLAN §12.4 did) and
  re-derive; re-measure the statue (`zero_action_baseline.py trex:3
  --episodes 40 --seed 3042`) whenever a behavior reward weight or the plant
  moves, and re-derive the 361 rail and the 602.0 reference together.
- **MEDIUM (operational)** — **a CLI-certified hunt (`curriculum --target
  hunt`) is judged IN-TRAINING on the last EvalCallback panel and cannot be
  backfilled.** `train_curriculum` has no post-stage judge and writes no
  `evaluation_selected.csv`; its `task_success/v1` verdict is the
  `CurriculumManager`'s own bound over the EvalCallback successes
  (persisted as `success_count` / `n_success_samples` in the verdict's
  `stage_result`, with `required_consecutive` hysteresis), which is a
  different estimator from the post-stage arm's single hash-bound panel on
  the selected handoff. `backfill_gate_verdict.py` refuses such a directory
  (no `evaluation_selected.csv` hash-bound to the handoff), and publication
  needs the notebook, whose chain loop writes the evidence CSV and judges
  it through `generate_stage_artifacts`. Treat a CLI hunt verdict as a
  training-time signal, not a certification.

<!-- The items below come from the 2026-07-31 plant validation pass; full
     evidence in PLANT_VALIDATION_AND_STAGE1_OBJECTIVE.md. The reset and
     self-collision defects that pass also found are FIXED in PR #479 and so
     are deliberately not listed here. Two further entries — the stage-1
     statue-optimum-cannot-be-reward-gated finding (§9) and "stage 1's real
     gate machinery does not exist yet" (§12/§14) — were deleted 2026-09-05:
     their ask shipped as the stance_quality/v1 gate kind
     (environments/shared/curriculum/gate_schema.py, stance_gate.py), which
     certified the first GATE: PASS on 2026-08-11
     (investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md). -->

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

- **MEDIUM** — **`collapse_peak_floor` is still an absolute reward value on
  three species' stage-1 configs and cannot survive a reward-function edit.**
  The T-Rex is done: `stance.toml` and `recovery.toml` use the relative
  `collapse_peak_floor_fraction` (PLANT_VALIDATION §14 item 3) after the
  absolute floor failed to arm twice. `velociraptor`, `brachiosaurus` and
  `dibothrosuchus` `stage1_balance.toml` still carry absolute floors
  (1300 / 1300 / 1950, each commented "Absolute pending §14 item 3") derived
  from their statues' standing reward; re-derive them as fractions the same
  way. (PLANT_VALIDATION §11.4) **Update 2026-09-23:** an absolute floor at
  0.75 x the statue also sits below the UNTRAINED policy (action 0 is the
  nominal stance under home-keyframe-residual/v1), so without a peak
  warm-up the backstop arms on initialisation: dibothrosuchus run
  `20260923_020654` stopped both nodes at 1.45M (stance armed on the 2592.9
  statue plateau; locomotion on the 2246.9 standing level, 22x its absolute
  floor of 100). `collapse_peak_warmup_timesteps` now separates the two on
  dibothrosuchus and brachiosaurus stages 1-2 (1.0M on stance; on
  locomotion the D-B5 bound `warmup_timesteps + ramp_timesteps`, 3.3M and
  4.0M, conservative since the two shaping schedules run concurrently),
  replayed on that run's series in
  `test_curriculum_early_stopping.py`. Still open: the four floors stay
  absolute (convert them to the relative pair, with the
  `statue_constants_physics_revision` pin, once the planned dibothrosuchus
  and brachiosaurus plant updates have settled; the locomotion statues
  re-measured 2026-09-23 are 2196.9 +/- 217.7 and 2242.7 +/- 7.8,
  `zero_action_baseline.py <species>:2 --episodes 40 --seed 3042`), and
  velociraptor stage 1 has the same shape with no warm-up (its session-3 run
  `20260922_125248` trained the full 6M by its trajectory, not by protection).
- **LOW** — **an early stop by the collapse backstop is invisible in the run
  records (verified 2026-09-23).** Run `20260923_020654` stopped both nodes
  at 1,450,000 steps; `stage_config.json` keeps the budget in `run.timesteps`
  (6,000,000 / 12,000,000), `gate_verdict.json` records
  `stage_result.timesteps = 1450000`, and no field names a stop or its
  reason (`collected_results.csv`'s `gate_reason` / `gate_evaluable` are
  empty). The callback logs the arming and the stop, but only to the
  session log; the cause was established by replaying `evaluations.npz`
  through `EvalCollapseEarlyStopCallback`. Fix: record `early_stopped`,
  the stop step and the armed peak in the stage result.
- **MEDIUM** — **the quadruped gait-symmetry reward pays a motionless
  statue its full weight on every step (verified 2026-09-23).**
  `BaseDinoEnv._compute_quadruped_gait_symmetry` (`base_env.py`) scores the
  alternation ratio of a touchdown history that never decays: a statue logs
  one touchdown per diagonal pair at reset, the ratio is 1.0 from then on,
  and `weight x 1.0` is paid every step. On the locomotion stages it is
  most of a typical do-nothing episode: dibothrosuchus 1994.0 of 2244.2
  (`gait_symmetry_weight` 2.0 x 997 steps, 89 percent), brachiosaurus
  2200.0 of 2244.0 (2.2 x 1000, 98 percent), measured with
  `zero_action_baseline.py <species>:2`. Standing still is therefore a
  strong local optimum of the walk stage, and dibothrosuchus run
  `20260923_020654` judged a stand-still locomotion checkpoint
  (2249.86, 0.0012 m/s; gate FAIL on the forward rail). In two of 40
  dibothrosuchus statue episodes (seeds 3045, 3056) one diagonal pair touches
  down twice while the statue settles after reset (by step 3-4), the history
  freezes at AAB / BBA, the ratio is 0.5 for the whole episode and it scores
  about 1248. The biped `_compute_gait_symmetry` shares the
  history-ratio shape (not measured here). A reward change moves the
  task fingerprint, so it belongs with the planned plant and reward work on
  these species, not with a backstop setting.

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

- **MEDIUM (JAX)** — **the JAX command-line curriculum advances a stage
  without checking `min_avg_forward_vel` (verified 2026-09-26).** The
  `reward_and_length/v1` arm of `jax_curriculum.check_stage_gate`
  (`jax_curriculum.py:455-486`) checks `min_avg_reward` and
  `min_avg_episode_length` only. The SB3
  `CurriculumManager` enforces both further thresholds when they are set
  (`curriculum/manager.py:340-348`), and so does the JAX notebook's CPU
  evaluation. Executed with a huge return and length and no velocity or
  success metric, it passes locomotion for velociraptor (bar 2.0 m/s),
  brachiosaurus (0.75), dibothrosuchus (0.9) and trex (1.0). It would also pass
  the stage-3 success bar (0.5) of the first three, but its one caller,
  `run_curriculum` (reached by `python -m environments.shared.jax_training
  --curriculum`), never checks the final stage's gate (`if stage !=
  stages[-1]`, `jax_curriculum.py:654`), so only the missing velocity check
  bites; the missing success check is latent in `check_stage_gate`. Every
  stage of velociraptor, brachiosaurus and dibothrosuchus is
  `reward_and_length/v1`, so that command can advance a standing policy to
  stage 3; trex's is refused up front (D-B13). No certified run comes from
  this path. Plan: retired with the JAX runtime (PR-B in
  [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md), proposed D-D17); until
  then read a JAX CLI stage advance as reward and length only. (2026-09
  cleanup survey)

- **MEDIUM (JAX)** — **MJX training never pays dibothrosuchus
  `snap_snout_proximity_weight`, while the JAX CPU evaluation that gates the
  stage does (verified 2026-09-26).** The MJX step kernel looks the proximity
  weight up under `bite_head_proximity_weight`,
  `strike_claw_proximity_weight` and `food_head_proximity_weight` only
  (`mjx_env.py:1303-1306`). `compute_total_reward`, which the CPU evaluation
  uses, also reads `snap_snout_proximity_weight`
  (`jax_reward_termination.py:348-356`). For dibothrosuchus stage 3
  (`stage3_snap.toml:21`, weight 2.0) the built MJX env's config carries 2.0
  while the kernel's lookup returns 0.0. `_KNOWN_REWARD_KEYS`
  (`mjx_env.py:61-111`) lists the key as read by the MJX step, so no
  unknown-key warning fires. MJX training therefore never optimises a term
  that the stage's `min_avg_reward = 100.0` is judged on. Neither JAX side
  matches SB3, which pays it over a fixed 1.5 m range
  (`dibothrosuchus_env.py:453-456`) where the composer uses `forward_vel_max`.
  Plan: retired with the JAX runtime (PR-B); a re-add builds one reward
  composition for the kernel and the evaluation. (2026-09 cleanup survey)

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
  excluded from the ground-settle probe. On behavior terrain, T. rex's
  non-colliding neck is probed through `_terrain_contact_probe_geoms`
  (#556, consolidation PR-7) and ends the episode as `neck_ground_contact`;
  brachiosaurus and dibothrosuchus can opt in by declaring their neck geoms
  there, without changing canonical physics.

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
  - `jax_train_fn.py` (step_fn) — `actions = jnp.clip(raw_actions, -1.0, 1.0)` before
    `env.step`; `jax_ppo.sample_action`'s docstring states the contract
    ("returns the **unclipped** action… callers must clip before sending to the
    environment").
  - `base_env.py:_scale_action` says so directly: "SB3 already clips before
    stepping, but direct callers… would otherwise command out-of-range ctrl."

  **The metric hazard.** `action_mean`, `action_std`, `action_abs_max`,
  `action_abs_mean` and `raw_action_saturation` come from
  `DiagnosticsCallback._on_step`'s read of `self.locals["actions"]` — SB3's
  **pre-clip** Gaussian sample. `action_delta` arrives by a different route:
  it is returned by `reward_action_smoothness` from *inside* the env, so it is
  computed on the **post-clip** action. Five scalars in one namespace describe
  the policy's raw output; the sixth describes what the plant received.
  Reading the group as one space is an easy and consequential mistake.
  It was briefly sharper than that: the 2026-08-10 shaping pack gave the env
  an `action_saturation` info key (the ramp fraction behind
  `reward_action_saturation`, measured on the filtered command the plant
  integrates), and the callback recorded its pre-clip 0.99-threshold fraction
  under the **same** `diagnostics/action_saturation` key later in the same
  rollout-end pass, silently overwriting the env's value every rollout. Fixed
  2026-08-15 by renaming the callback's metric to
  `diagnostics/raw_action_saturation`; `diagnostics/action_saturation` is now
  unambiguously the env's. The table above predates the rename — its
  `action_saturation` row is the pre-clip quantity now named
  `raw_action_saturation`.

  **What is real.** PPO stores the raw action and its `log_prob`, while the
  environment responds to the clipped one. With 68% of components saturated,
  most of the policy's output distribution sits where moving the mean further
  changes the plant not at all — the standard bias from sampling an unbounded
  Gaussian into a bounded action space, and a plausible contributor to the
  bang-bang envelope measured on the same run (`used` = 100% of range on 20 of
  21 actuators). Worth watching `raw_action_saturation` as a first-class health
  metric rather than as evidence about the reward.

  **Latent trap — now scoped to filter-free species.** `BaseDinoEnv.step`
  passes the raw `action` to `_get_reward_info` while `_scale_action` clips
  separately on the way to `ctrl` (anchor on the function names; the line
  numbers this paragraph used to carry rotted by ~250 lines in the August
  rewrites). Harmless under SB3 and the JAX trainer today, but any direct
  caller — a notebook, a custom rollout loop, a diagnostic script — is
  silently charged energy and smoothness penalties for magnitude the plant
  never sees. Since r11 this applies only to species with
  `action_filter_cutoff_hz = 0` — today, every species except the T-Rex:
  `_filter_action` clips before the reward terms read the action, so the
  T-Rex's two paths already agree. Clipping once at the top of `step` for the
  filter-free species would close it everywhere and moves no fingerprint.

  **Explicitly retracted.** An earlier version of this entry claimed the energy
  term was inflated ~3.8× (~209/episode, ~7.9% of return) and that the
  `r ∝ w^-0.16` smoothness fit in
  [TREX_LEG_FLEXING_PLAN.md](TREX_LEG_FLEXING_PLAN.md) was therefore unsound.
  Both are wrong: the reward saw clipped actions, and `r` derives from
  `action_delta`, which is post-clip and so is coupled to the physics. That
  study stands, no `min_avg_reward` gate needs re-deriving, and no historical
  reward comparison is invalidated. (2026-07 T-Rex telemetry review)

- **MEDIUM** — **`render_mode="human"` crashes on the first step, so
  `train_sb3.py eval` without `--no-render` fails (reproduced 2026-09-26).**
  `BaseDinoEnv.render` calls `mujoco.viewer.launch_passive`
  (`base_env.py:1558`), but `base_env.py` imports only `mujoco` (:19-21). After
  importing `cli`, `evaluation` and `train_base` under mujoco 3.10.0 the
  `mujoco.viewer` submodule is still absent, and
  `TRexEnv(render_mode="human")` raises `AttributeError: module 'mujoco' has no
  attribute 'viewer'` from `step` (:1249). `evaluate()` defaults to
  `render=True` (`evaluation.py:415-463`), and the documented eval commands
  (`environments/velociraptor/README.md:102`, the website quick start and API
  overview) omit `--no-render`; `test_env.py --render` (`harnesses/env_smoke.py`)
  takes the same path. Only standalone scripts (`harnesses/viewer.py`,
  `harnesses/actuators.py`, `compsognathus/scripts/view_model.py`) import the
  viewer; nothing on the env or evaluation path does. Workaround: pass `--no-render`, or `import mujoco.viewer`
  before creating the env. Fix: a lazy import in the human branch that shares
  `_make_camera`'s camera setup (CU-2 in
  [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md)); `render` enters no
  digest. (2026-09 cleanup survey)

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
- **MEDIUM** — **the direction/terrain pilots are a second pipeline
  (#540/#541).** The 2026-09-17 review found that the pilots delivered new
  content (a direction controller, a tracking reward, a heightfield terrain
  generator, a replay recorder with terrain maps) beside every canonical
  concept instead of through it: behavior env classes (two until PR-7 deleted
  `TRexBehaviorEnv`) that bypass the
  reserved `BaseDinoEnv._draw_episode_command` hook and write
  `self._command` directly; a second PPO trainer
  (`environments/shared/train_behaviors.py`, one CPU env) with its own
  recipe dialect; 66 behavior TOMLs under `configs/<species>/behaviors/`
  (11 templates × 6 species; the 8 trex `[pilot]` twins under
  `configs/trex/behavior_pilots/` and the `[pilot]` dialect left with PR-6);
  a second gate outside `GATE_KINDS` (`configs/behavior_certification.toml`,
  judged by `judge_behavior_panel` and reached only from tests since PR-5
  deleted its `certification/certificate.json` writer; never a
  `gate_verdict.json`); a checkpoint identity keyed on source-file hashes (any
  edit to `behavior_env.py` strands exact resume) — about 7,000 lines of
  modules at 22c1fc8, tests excluded, with the notebook `COMMAND_TERRAIN_BEHAVIOR`
  mode switch that 19 of the notebook's 22 code cells referenced (the certified library
  left with PR-4/PR-5, the mode switch, its ten `BEHAVIOR_*` knobs and
  `behavior_notebook.py` with the notebook-only PR-12 slice). Pilot outputs
  (`logs/<species>/ppo/behaviors/<behavior>/<run-id>/`) are evaluation-only
  (decision D-D9): none is reusable through `find_certified_ancestor` or
  carried forward as a training parent. The fold-back — manifest nodes
  under `locomotion`, the controller as the body of the reserved hook, one
  generic opt-in terrain env subclass, a registered gate kind, the library,
  the mode switch and the second trainer deleted — is
  [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md)
  PR-3..PR-15, released 2026-09-20 in the notebook-first order of D-D13
  (PR-3, bounding the SB3 CI job, landed as #546; PR-4, the canonical library
  wrapper and the notebook's library knobs, landed as #547; PR-5, the
  certified library and the trainer's library path, landed as #548; PR-6, the T. rex
  pilot recipes, the `[pilot]` dialect and the trex shim, landed as #549;
  the notebook-only PR-12 slice, the notebook's direction/terrain mode and
  `behavior_notebook.py`, landed as #552 on 2026-09-24; PR-14, split into
  PR-14a, PR-14b and PR-14c (D-D15), landed as #553, #554 and #555 on
  2026-09-24; PR-7, the ground-height hook and the deletion of
  `TRexBehaviorEnv`, landed as #556 on 2026-09-25); PR-1 (#542), the
  auto-trunk PR (#543) and PR-2 (#544) landed. Until PR-11 adds the follow and
  terrain manifest nodes, the pilots run from the command line only
  (`python -m environments.shared.train_behaviors`, D-D13), with
  [TRAIN_DIRECTION_AND_TERRAIN.md](TRAIN_DIRECTION_AND_TERRAIN.md) as the
  operator guide and an explicit `--checkpoint` / `--vecnormalize` pair in
  every load mode and for evaluation (the certified library and
  `SOURCE_SELECTION` left with PR-5).
- **HIGH (terrain blocker)** — **every certified walker survives the plane and
  falls on a flat heightfield (measured 2026-09-25).** An eval-only run (seed
  1, nothing trained or written under `logs/`) of the `robust_best_model` pair
  each node's `gate_verdict.json` names — trex `20260914_123816/03_locomotion`,
  velociraptor `20260922_125248/02_locomotion`, compsognathus
  `20260921_203149/03_locomotion` — on four direction/terrain recipes kept all
  23 plane episodes per walker at full horizon. On the `terrain_contact`
  family, an all-zero heightfield (`mode = "flat"`, `terrain.py:391`) with the
  plane's geometry, full horizon fell to 1/13 for trex (fallen 11,
  head_contact 1), 0/13 for velociraptor (tail_contact 3, fallen 1; the other
  9 left the map) and 0/13 for compsognathus (excessive_tilt 12, body_contact
  1); the sloped, bumps, depressions and mixed families gave 3/24, 0/24 and
  0/24. Cells are the committed 200 mm (trex), 137.5 mm (velociraptor) and
  20 mm (compsognathus); trex at 100 mm fell 7/7, although its zero-action
  statue survives 3/4 there, so statue results do not predict the walker. The
  cause is open. The two precompiled scenes
  (`SpeciesBehaviorMixin._select_contact_model`, `behavior_env.py:249`) give
  the `floor` geom the same `solref`, `solimp`, friction, margin, gap and
  condim and differ only in its type (plane vs hfield; checked 2026-09-26).
  That leaves the hfield collision and the terrain spawn settle
  (`behavior_env.py:395-413`, which reproduces velociraptor's authored −44.6 mm
  toe penetration) as the suspects. Compsognathus-pair foot contact also
  flickers on a heightfield (a statue measurement from the 2026-09-25 readiness
  review). Velociraptor's map exits are a separate mismatch: its recipes cruise
  at 2.0 m/s, the walker runs at 3.64–3.66 m/s, and the maps fit 25 s at
  cruise. No terrain pilot or terrain node should start from these parents,
  and consolidation PR-11 copies the terrain keys verbatim. Plan: a dated
  heightfield-contact investigation before PR-11, decided together with the
  recipe speeds and map sizes
  ([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §2 and §5.1).

## Sweeps / infrastructure

- **MEDIUM** — **every Ray Tune PPO trial raises `TypeError` before it trains
  (reproduced 2026-09-26).** `ray_tune.train_trial` copies
  `stage_config["ppo_kwargs"]` (`scripts/sweep/ray_tune.py:714`) and pops only
  `learning_rate_end`, `lr_schedule`, `clip_range_end` and `policy_kwargs`
  (:718-733) before `PPO("MlpPolicy", ..., **alg_kwargs)` (:759). The
  canonical builder also pops the callback-driven `ent_coef_end` and
  `ent_coef_decay_timesteps` (`train_base._prepare_alg_kwargs`,
  `train_base.py:354-355`) and adds the decay callback
  (`_maybe_ent_coef_decay_callback`, :841). All 21 PPO stage configs carry
  `ent_coef_end` (20 also carry `ent_coef_decay_timesteps`), and replaying
  :714-733 for each against SB3 2.9.0 raised `PPO.__init__() got an unexpected
  keyword argument 'ent_coef_end'` 21 times out of 21. The sweep notebook's
  default `ALGORITHM = "ppo"` hits it. A warm-started trial (:749) does not
  raise, but SB3's `load` stores the keys as plain attributes, so entropy
  never decays (read from the code). SAC configs carry no such keys, and the
  Vertex trial trains through `train()`, unaffected. No test builds the
  trial's model. The 2026-09 survey found the defect already present at the
  oldest reachable commit (2026-08-09). Plan: PR-A of the backend retirement
  (proposed D-D17, [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md))
  deletes the Ray worker; a re-add wraps `train_base.train()` instead of
  copying it. (2026-09 cleanup survey)

- **MEDIUM** — **7 of the 12 sweep configs crash every stage-3 trial: they
  sample 18 env keys that no constructor accepts (re-counted 2026-09-26).**
  brachiosaurus `sweep_{ppo,sac}.json` sample `env_food_distance_range_min` /
  `_max` and `env_food_height_range_min` / `_max` (the constructor takes the
  tuples `food_distance_range` / `food_height_range`); dibothrosuchus
  `sweep_ppo.json` samples `env_prey_distance_range_min` / `_max`; trex and
  velociraptor `sweep_{ppo,sac}.json` sample `env_prey_distance_min` / `_max`,
  which match no constructor parameter at all. Both sweep paths put the
  prefix-stripped name into `env_kwargs` (Ray: `scripts/sweep/ray_tune.py:499-517`;
  Vertex: `scripts/sweep/trial.py:16-31`, then `_apply_overrides`), and each
  of the 18 raised `TypeError: ... unexpected keyword argument` when its
  stage-3 env was constructed. Suffix pairing in the override code would fix
  only the brachiosaurus and dibothrosuchus keys; the trex and velociraptor
  keys must go. Plan: PR-A of the backend retirement deletes the sweep JSONs;
  a re-add regenerates them from the constructor signatures. (2026-07
  Dibothrosuchus review; 2026-09 cleanup survey)

- **MEDIUM (cleanup)** — `ray_orchestration.py` (1,006 lines) is wired to
  `ray_tune_sweep.ipynb` only through `export_best_trial`; `create_ray_tuner`,
  `run_ray_sweep`, `discover_and_rank_trials` and `evaluate_trials_parallel`
  (about 740 lines) have no production caller, and the notebook keeps its
  inline Tuner and ranking copies, which have already diverged once. PR-A of
  the backend retirement deletes both. (July §4)
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
- **LOW** — **stage summaries built from `evaluations.npz` assume a 0.01 s
  control step, so both compsognathus species report half their sim time
  (reproduced 2026-09-26).** `build_stage_results_from_eval_data` records
  `sim_dt = stage_config["env_kwargs"].get("sim_dt", 0.01)`
  (`reporting/stage_artifacts.py:151`), and no stage config sets `sim_dt`.
  compsognathus and compsognathus_robot step at 0.02 s, the other four
  species at 0.01 s. `write_stage_summary` and `write_training_summary`
  multiply episode length by it, printing 1,000 steps as "10.00s sim time"
  instead of 20 s. It shows where `generate_stage_artifacts` builds its own
  results (`stage_results=None`, :1463): the Ray and Vertex sweep trials
  (`ray_tune.py:1025`, `trial.py:233`); `ray_tune_sweep.ipynb` also keeps its
  own copy of the default (`_sim_dt`, passed as `stage_results`) beside a
  `LocomotionMetrics()` that defaults to the same 0.01 s. The notebook's TRAIN and JUDGE paths
  overwrite `sim_dt` with the env's `dt` (`evaluate_stage_checkpoints`,
  :1242, :1277). `backfill_gate_verdict.py:293` builds these results, but no
  field it persists uses `sim_dt` (the verdict's `stage_result` projection
  omits it). No gate, verdict or digest reads it. Plan: take `dt` from a probe
  env (CU-2 in [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md)); once PR-A
  retires the sweep trials and that notebook, nothing in the repository shows
  it, and the default is latent until CU-2.
  (2026-09 cleanup survey)

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
- `pyproject.toml`'s gymnasium entry-point groups (`gymnasium.envs.__root__`,
  `gymnasium.envs.MesozoicLabs`) are dead: gymnasium 1.3.0 has no plugin
  loader, so `gym.make("MesozoicLabs/Raptor-v0")` without `import
  environments` raises `NamespaceNotFound` even with the groups installed
  (verified 2026-09-26); the envs self-register on `import environments`.
  Delete the block (CU-7). `[all]` omits `[mjlab]` (moot once PR-A removes
  `[mjlab]`). (June §4)
- `docs/investigations/REWARD_SCALE_REDESIGN.md` uses `*_bonus_weight` key
  names that don't exist. (June §5)

## Notebooks

- `ray_tune_sweep.ipynb` duplicates `ray_orchestration.py` (see above); its
  `EVAL_EPISODES` knob doesn't affect the in-trial eval episode count; and its
  post-sweep analysis writes the sweep's `training_summary.txt` from the
  last-ranked of the top-`TOP_K` trials (default 5), not rank 1: the loop runs
  ranks 1..`TOP_K` and the summary takes the last iteration (cell 23; gap
  review NB3). PR-A deletes the notebook. (July §5)
- **LOW** — **the Drive summary skips every sweep folder written under the
  current naming (verified 2026-09-26; gap review NB2).** `ray_tune_sweep.ipynb`
  names a sweep `<species>/sweeps/<algorithm>_<YYYYMMDD_HHMMSS>` (cell 7;
  present at the oldest reachable commit, 2026-08-09), while
  `google_drive_summary.ipynb`'s `parse_sweep_dir_name` (cell 8) matches only
  `^stage(\d+)_(.+?)_(\d{8}_\d{6})$`, and `discover_runs` (cell 11) keeps only
  matching children of `sweeps/` without recursing. A current sweep is
  dropped with no warning. A read-only listing of the project Drive
  (2026-09-26) found three `sweeps/` folders, created 2026-03-25..28, holding
  only ten legacy `stage<N>_<algo>_<ts>` folders, which the summary reads, and
  no current-layout sweep. Plan: PR-A stops all sweep writes and keeps this
  reader for the March folders; drop the entry then, and fix the pattern only
  if a current-layout folder turns up.
  ([reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md](reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md)
  NB2)
- The notebooks pin the plant compiler (`mujoco==3.10.0`, `mujoco-mjx`),
  `stable-baselines3[extra]==2.9.0` (SB3 and Ray notebooks) and
  `jax[cuda12]==0.10.2` / `flax==0.12.8` / `optax==0.2.8` (JAX notebook), but
  torch is unpinned, Ray is a range (`>=2.55.0,<3`), and the `pyproject.toml`
  extras stay ranges; pin complete lockfiles for reproducible training.
  (July §5)

## Testing / CI

- **LOW** — `hw_chassis_study.py`'s excitation drives hip pitch and knee in
  phase, unlike a real trot where knee flexion leads swing. This contributes to
  the knee pinning at its 22 N·m cap in all 18 runs, so that column is reported
  as "not a configuration discriminator"; the ab/ad conclusions are unaffected
  (that joint is driven at 0.25x and does discriminate).
  (2026-07 Dibothrosuchus review)
- **LOW** — **pre-commit pins ruff 0.4.4 while CI installs the latest ruff,
  and the two disagree on 22 files (verified 2026-09-26).**
  `.pre-commit-config.yaml` pins `ruff-pre-commit` `v0.4.4` (and
  `mirrors-mypy` `v1.15.0`), while the CI lint job runs `pip install ruff mypy
  gymnasium numpy` unpinned (`.github/workflows/python-ci.yml:82`). On
  `be63a58`, as on `f850815`, ruff 0.4.4 `format --check environments/`
  would reformat 22 files, and its `ruff check` reports E721 at
  `shared/tests/test_widen_checkpoint.py:1344`. Ruff 0.16.8 passes both.
  Formatting the tree with 0.4.4, as the hook does, leaves 22 files that
  0.16.8's `ruff format --check` then rejects. `CONTRIBUTING.md` tells
  contributors to install and pass the hooks. Workaround: skip the ruff hooks
  (`SKIP=ruff,ruff-format`) and run a current ruff as CI does. Plan: pin one
  ruff version for both (CU-1 in [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md)).
  (2026-09 cleanup survey)
- **LOW** — **CI's mypy never sees Stable-Baselines3 or torch types; with
  them installed the tree has 21 type errors (measured 2026-09-26).** The
  lint job installs only `ruff mypy gymnasium numpy`
  (`.github/workflows/python-ci.yml:82`) and runs `mypy environments/
  --ignore-missing-imports` (:91). That reports no issues in 358 files,
  because SB3 and torch resolve to `Any`, and the pre-commit mypy hook has
  the same blind spot. With stable-baselines3 2.9.0 and torch 2.14.0+cpu
  installed, the same command finds 21 errors in 6 files, both on `f850815`
  and on `be63a58` (#559): `curriculum/advancement.py` 7,
  `scripts/widen_checkpoint.py` 5, `diagnostics.py` 4,
  `curriculum/schedules.py` 2, `tests/test_widen_checkpoint.py` 2,
  `command_frame.py` 1. All are type-only. `BaseAlgorithm` lacks `ent_coef`,
  `clip_range` and `log_ent_coef`; the SB3-absent fallback assigns read-only
  `BaseCallback` properties (`diagnostics.py:222-223`); state dicts are typed
  as `Tensor`; and some Optional values go unchecked. A new type error in
  SB3-facing code passes CI unseen. Plan: fix the 21 with casts and
  annotations (in `advancement.py`, casts rather than runtime guards), then
  add a mypy step to the SB3 job with `stable-baselines3==2.9.0` and
  `torch==2.13.0` pinned, so an upstream release cannot turn an unrelated PR
  red (CU-1 in [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md)).
- TOML→env round-trip test: construct each env with each stage's
  `env_kwargs`, assert no unknown/unused keys. (June §6.8)
- SB3↔JAX reward parity test (see divergences section above). (June §6.8)

## Open questions

1. Is SB3↔JAX reward parity a hard goal? If yes, the parity test should
   gate CI; if the JAX path is a research spike, keep the divergence table
   above authoritative. (June §7)
2. Published sweeps ran with identical seeds per trial (fixed going
   forward); re-run top-3 configs with 3 seeds before locking them into the
   TOMLs. (June §7)
