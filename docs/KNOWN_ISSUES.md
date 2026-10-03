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
| [reviews/VELOCIRAPTOR_PLANT_REVIEW.md](reviews/VELOCIRAPTOR_PLANT_REVIEW.md) (2026-07-27) | Raptor plant: anatomy vs published *Velociraptor* material, and mechanics | 11 findings; finding 7 (MJX termination) retired with the JAX/MJX runtime (D-D17, cleanup PR-B), the other 10 open — **execution deferred until the T-Rex clears stages 1–3**; see below |
| [reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md](reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md) (2026-08-28) | SB3 training core, notebooks, env/physics, evaluation, JAX/MJX, configs and sweeps, CI, scripts; cleanup opportunities | 120 ids: 78 fixed (#514–#517, #519, #530, #534, #535), 25 retired by D-D17 (16 of them after a fix), 3 fixed with a residue, 6 duplicates; the 8 open ones and the three residues (SM7, DU4, CI8) are below (its appendix B, 2026-09-30) |

Severity: **HIGH** = wrong results in common cases, **MEDIUM** = edge cases /
robustness, **LOW** = cosmetic / QoL.

---

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
  **Update (2026-09-28):** nor does a PASS always certify a clean stance: the
  certified seed-44 stance `20260920_010912` hops to rebalance in 6 of 40
  episodes (the stance-gate HIGH at the end of this section).
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
  `20260914_123816`). One neighbour: the two compsognathus recovery
  calibrations were restamped, not re-measured, which moved their
  `profile_sha256`, so every compsognathus / compsognathus_robot recovery
  freeze made before Phase C is refused and must be re-frozen from the
  restamped profile
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
- **LOW (operational)** — **nothing on disk records the trunk a session
  resolved, so a resume must re-supply it by hand, and a wrong one fails late
  or discards the resumed node (read from the code at the #558 follow-up,
  #559).** `TRUNK_FROM = "auto"` is resolved in the kernel (`select_trunk` in
  the resolve cell, D-A25) and only printed. Each reused
  node leaves `ancestors/<id>/ancestor.json`, but that names the run that
  certified the node after following records (`_ancestor_record`,
  `ancestors.py:644-667`), not the trunk, and the chain loop never follows
  this run's own records (`follow_records=candidate is not RUN_DIR`). So the
  resume recipe (section 5) has the operator pin `TRUNK_FROM` to the trunk the
  interrupted session printed, or derive it from the nearest ancestor record.
  A wrong trunk goes one of three ways. (a) `"auto"` picks a newer run
  holding a different certified copy of a reused ancestor: `record_ancestor`
  refuses ("a run cannot reuse two parents for one node",
  `ancestors.py:690-700`), but only in the chain loop, after the RESUME cell
  has trained the remaining budget; re-running with the right trunk then
  judges the node. (b) `"auto"` picks a newer run that certifies the resumed
  node itself (an ancestor of `BEHAVIOR`'s target, since the target is looked
  for only in this run): the loop reuses that copy, prints it as a reuse, and
  never judges the resumed node. (c) `TRUNK_FROM = ""`: ancestors this run
  holds only as records are trained again here (found by the review of
  #559's first round, `ee91f51`; review 2 in CLEANUP_PLAN_2026_09.md §5.4). A node `RETRAIN_FROM` covered has the shape of (b); #559
  refuses its resume and names the route (`BEHAVIOR` set to that node, then a
  fresh `RUN_ID` trunked from this run). Plan: two decisions in
  [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §2 (rows 4 and 6), which
  the maintainer took on 2026-10-02 and the notebook PR ROW-4/6 carries out: a
  run-level record of the resolved trunk that the RESUME cell reads, and
  judging an unjudged `RUN_DIR` node before any trunk reuse. (#558 reviews)
- **MEDIUM (operational)** — **`train --load <checkpoint>` (default
  `--load-mode resume_same_stage`) writes into a stage directory that already
  holds `gate_verdict.json` (guard executed 2026-09-26).** The D-A20 guard
  `config.refuse_occupied_stage_dir` (`config.py:550-573`, called at
  `train_base.py:1101`) lets any same-stage resume through. Against a
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
  **Update (2026-09-28, gait audit):** the biped `_compute_gait_symmetry` has
  been measured: a synchronous two-foot landing appends `"R"` then `"L"`
  (`base_env.py:745-748`), so a bounce scores 1.000, as do a true alternating
  walk and the statue ([STAGE1_SPLIT_PLAN.md](STAGE1_SPLIT_PLAN.md) §6
  item 7). Its weight is 0.0 in every biped stage, so only its
  `alternation_ratio` diagnostic misleads. The quadruped version likewise pays
  a pronk, bound or pace in full, since a simultaneous landing appends both
  pairs (:845-848; [gait audit](investigations/GAIT_AUDIT_2026_09.md) §3). The
  gait plan keeps the code byte-identical and sets the weight to 0 in the
  quadruped `gait-r1` revisions ([gait plan](GAIT_QUALITY_PLAN_2026_09.md)
  §5.3, PR-G7); this entry stays while any TOML uses the term.

- **LOW** — **contact-switch rate conflates bilateral↔single with
  bilateral↔airborne.** The PR #479 plant repair moved T-Rex's raw switch count
  *up* (0.86 → 1.00 /s) while unsupported duty went to **zero** — the extra
  switches are ordinary weight-shifting. Do not gate on it until decomposed;
  gate on unsupported duty instead. (PLANT_VALIDATION §11.3)

- **LOW** — **four stage-1 reward terms are saturated and contribute no
  gradient**: `head_clearance` pinned at exactly its full 0.350 weight in every
  measured window, `height` 0.578 of 0.6, `neck_posture` 0.173 of 0.2,
  `leg_home_pose` 0.312 of 0.5. (PLANT_VALIDATION §14)

- **LOW** — **collidable necks are deferred until terrain lands.** Velociraptor
  is the reference: its neck geom collides *and* sits in `_body_ground_geoms`,
  so hitting the ground with it terminates the episode. The other three carry
  `contype=0` necks (plus cosmetic `brow_ridge` / `crest` / `sagittal_crest`
  and dibothrosuchus' twelve `scute`s), and brachiosaurus documents the choice
  explicitly, using the collidable head as the termination proxy. On a flat
  floor this is unobservable — an animal whose neck reaches the ground has
  already tripped tilt, height or head-contact termination — so the decision
  was to leave physics alone and revisit when heightfield terrain arrives, at
  which point the raptor's pattern is the template. Newly-colliding long neck
  capsules must be checked for home-pose self-collision (the defect class
  fixed twice in the PR #480 series). The
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

  **What this is not: the reward is not inflated.** SB3 clips before
  stepping the environment, so `_get_reward_info` receives an in-bound
  action and both penalties are computed on it:

  - SB3 `on_policy_algorithm.py:214-218` — `clipped_actions = np.clip(actions,
    low, high)` immediately before `env.step(clipped_actions)`; `policies.py:379`
    does the same inside `predict()`, which is what both eval loops use.
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
  rewrites). Harmless under SB3 today, but any direct
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

- **LOW** — `BaseDinoEnv.reset` still applies one `reset_noise_scale` scalar to
  the whole of `qvel`, which mixes root linear velocity (m/s), root angular
  velocity (rad/s) and joint velocities (rad/s). This is the same
  units-conflation that made the root-height jitter wrong, but far less severe:
  a velocity kick has to actually defeat the controller, whereas the height
  jitter could spawn an episode already outside `healthy_z_range`. Worth
  separating if a species much smaller than Dibothrosuchus is ever added.
  (2026-07 Dibothrosuchus review)
- **LOW** — `plant_contract._mocap_target_name` now requires *every* plant to
  declare exactly one mocap body. All four comply and it fails loudly, but the
  constraint was introduced to derive a segment label, not because the contract
  needs uniqueness. (2026-07 Dibothrosuchus review)

- **LOW** — the T-Rex and Dibothrosuchus SB3 envs accept
  `foot_contact_weight` / `foot_contact_gate` (`trex_env.py:141-142`,
  `dibothrosuchus_env.py:108-109`) and no SB3 reward reads them: they were
  knobs of the MJX reward, which left with the JAX/MJX runtime (D-D17, cleanup
  PR-B). They stay, with their six `[env]` keys (`configs/trex/stance.toml:13-14`,
  `configs/trex/recovery.toml:57-58`, `configs/dibothrosuchus/stage1_balance.toml:13-14`),
  because `task_sha256` hashes the constructor defaults overlaid with `[env]`:
  removing either moves the trex and dibothrosuchus task digests (stripping the
  dibothrosuchus keys moved its certified stance task from `083e2966` to
  `abfb339f`, measured 2026-09-26). Do not delete them
  ([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §7). A typo'd weight
  still does nothing; a loud rejection of unknown env kwargs must keep these two
  names. (June §1.6)
- **LOW** — `CurriculumCallback` / `LocomotionMetrics` hardcode success keys
  (`bite_success`, `strike_success`, `food_reached`) instead of using
  `SpeciesConfig.success_keys`. (June §6.4)
  **Update (2026-09-28, gait audit):** the supplementary evaluation's flags
  (`curriculum/advancement.py:262,402`) read 0.0 on an upright dibothrosuchus
  or compsognathus success, as the tuple lacks `snap_success` and
  `target_success`; the `evaluations.npz` sample is preferred where it exists
  (:337), so in that check the manager's success rate still read 1.0.
  `LocomotionMetrics` (`metrics.py:154`) now lists `target_success` but not
  `snap_success`.
- **LOW** — `curriculum/advancement.py` `_read_latest_eval`: the
  `successes.shape[0] == n_evals` guard permanently discards npz successes
  if SB3 starts recording them one eval late. (July §2)
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
  **Update (2026-09-28, gait audit):** the certificate has no contact
  criteria. Its rules (`configs/behavior_certification.toml`) are survival,
  success, commanded speed and yaw-rate tracking, settling and course
  progress, so the per-episode statistic PR-13 registers as
  `terrain_command/v1` (D-D6) would certify a hop, scoot or slide that tracks
  the command ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3). The
  gait plan recommends a per-episode gait clause in that kind
  ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) PR-G10; GQ-18, open).
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

<!-- The items below come from the 2026-09-28 gait audit: CPU replays of
     the certified nodes and zero-action rollouts of every stage gate, with
     contact read two ways, "touch" (the env's own foot sensors) and "floor
     contact" (the floor's normal force on a leg above 0.1 N on at least
     half of a control step's substeps). Full evidence in
     investigations/GAIT_AUDIT_2026_09.md; the plan that acts on them is
     GAIT_QUALITY_PLAN_2026_09.md, whose decisions (GQ-1..GQ-18) are all
     open and whose code PRs build on 0.3.9. The lunge-then-fall and
     falling-step hunt items were executed through the repository's own
     code before they were entered. -->

- **HIGH** — **no locomotion gate reads a foot contact, and three of the five
  certified walkers hop (replayed 2026-09-28).** All six locomotion stages
  judge `reward_and_length/v1` (`reporting/gates.py:617-682`) on mean reward,
  length and forward velocity, with reward floors 3–22× below the zero-action
  statue, so beyond mean length the only test is mean root speed. On floor
  contact, trex seed 42 `20260914_123816` and seed 44 `20260925_033501` hop on
  both feet at 7.7 and 9.2 Hz, airborne 35% and 43% of steps, and
  compsognathus_robot `20260924_031815` micro-hops at 9.2 Hz, airborne 48%;
  only compsognathus `20260921_203149` walks and velociraptor
  `20260922_125248` runs. The in-training dibothrosuchus re-run
  `20260928_012318` skids on three legs at 4.3M and clears its gate as written
  ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3–§4). Plan:
  `locomotion_gait/v1` in the six locomotion TOMLs, then retrains
  ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) PR-G5 and PR-G7; GQ-8, open).

- **HIGH (hardware target)** — **compsognathus_robot stands and walks on
  stacked feet, and its touch sensors count sole-on-sole force as support
  (replayed 2026-09-28).** Its 10 × 9 cm soles collide with each other
  (`compsognathus_robot.xml:181,254`), and both certified nodes of run
  `20260924_031815` rest the right sole on the left foot: the stance carries
  81% of its floor load on the left foot, and its reported two-foot support of
  0.9988 is 0.62 on floor contact. The walker's feet touch on 96% of substeps,
  pushing 1.12 body weights into each touch sensor, so the support flag that
  scales its alive, posture and height rewards (touch sum above 4% of body
  weight, `compsognathus_env.py:189,239`) is on 99.6% of steps by touch and
  34.3% by floor: about 838 of its about 2627 reward pays for support the
  floor never gives ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §5).
  Plan: the robot's `gait-r1` revisions, with floor-contact support and a
  foot-collision penalty ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) PR-G7;
  GQ-14, open); the touch observation still counts that force until the
  robot's next policy-interface revision.

- **HIGH** — **the stance gates admit stances that chatter, march, stand on
  one foot or hop (replayed 2026-09-28).** `stance_quality/v1` counts only
  steps with neither foot above 0.1 N, averaged over the panel, and the other
  three stances gate on a reward rail at 0.60× the statue; the zero-action
  statue passes all six by design (`stance_gate.py:13-20`), as it would the
  planned v2. Of the six certified stances only trex seed 42 `20260914_123816`
  is clean: trex seed 44 `20260920_010912` hops to rebalance in 6 of 40
  episodes (duty 0.02–0.07 each, drift up to 0.57 m) under a panel mean of
  0.0069; compsognathus `20260921_203149` marches in place at 3.05 Hz, both
  feet down 10% of steps; velociraptor `20260922_125248` chatters at about
  10 Hz and slides 0.65 m; the robot stands on one foot; dibothrosuchus
  `20260923_020654` is the statue, three-legged in some episodes
  ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3–§4). Plan:
  `stance_quality/v2` per species inside its `gait-r1`
  ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) PR-G8; GQ-10 and GQ-12, open);
  compsognathus and velociraptor stay open until their optional revisions.

- **MEDIUM** — **locomotion gates average per-episode means, so a policy that
  lunges and falls passes (executed 2026-09-28).** An episode's speed is the
  mean of its per-step `info["forward_vel"]` (`evaluation.py:97-105`), the
  panel's the unweighted mean of those (`reporting/stage_artifacts.py:1356`,
  `curriculum/manager.py:197`), and length a panel mean, so a 250-step episode
  weighs as much as a 1,000-step one. Through the repository's own evaluation
  and gate code on trex locomotion (length 750, speed 1.0 m/s), three scripted
  episodes at 0.7 m/s for 1,000 steps and one at 2.0 m/s that falls at step
  250 give mean length 812.5 and speed 1.025 m/s (recorded 1.02) and pass,
  though distance over time is 0.80 m/s
  ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3); on a 30-episode
  panel, 23 walks and 7 lunges pass both the post-training judge and the
  in-training `CurriculumManager`, while 24 and 6 fail. The post-training
  judge also compares the speed rounded to two decimals (:1356): a 0.996 m/s
  panel passed the 1.0 bar. Plan: per-episode qualification under
  `locomotion_gait/v1` ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) §4.2, PR-G5;
  GQ-7, open).

- **MEDIUM** — **the recovery safe set has no support clause, and the
  certified trex recovery answers forward pushes with two-footed hops
  (replayed 2026-09-28).** The calibrated judge is posture-only on purpose,
  `min_foot_force_n = 0.0` (`recovery_evaluation.py:74-79`; the compsognathus
  pair's calibrations the same), because quiet certified stance reads 0.0 N on
  a foot during weight shifts. On a CPU replay of the seed-44 recovery in
  `20260920_010912` (certified 28/40), none of 41 forward pushes is absorbed
  in place: 27 mix two-footed hops with single-foot touches, 13 are pure
  two-footed hops and 1 falls; 37% of its touchdowns fall outside push
  windows, and it drifts 0.67 m per episode
  ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §4). Plan:
  `recovery_quality/v2`, with windowed support clauses, registered and then
  adopted per species ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) PR-G9; GQ-13,
  open); allowing no two-footed touchdown in a push window, seed 44 scores
  5/40.

- **MEDIUM** — **hunt success counts on a step that ends in a fall, and the
  panel and training definitions disagree (executed 2026-09-28).** On trex,
  velociraptor, brachiosaurus and dibothrosuchus the reward info sets the
  success flag and pays the 1,000 bonus whenever the contact or reach holds,
  but `_is_terminated` runs every fall check except floor contact before its
  success check, so such a step ends as a fall with `is_success` False
  (`base_env.py:1291`). The post-training panel reads the flag on every step
  (`evaluation.py:100`) and counts the episode, for trex in the `task_success`
  that `task_success/v1` judges; training reads `is_success` and does not.
  Executed on each hunt env with the body rolled past `max_tilt_angle` and the
  prey on the success geometry: `excessive_tilt`, 991–992 reward, a panel
  success (trex 30/30, gate PASS) and 0 in training; a floor-contact fall on
  the contact step ends as a success in both. The compsognathus pair reads
  success after its fall checks (`compsognathus_env.py:240-241`) and is
  unaffected. Leaves with each hunt's task revision, which the gait plan does
  not schedule ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) §9;
  [gait audit](investigations/GAIT_AUDIT_2026_09.md) §3).

- **MEDIUM** — **the brachiosaurus zero-action statue reaches the food in 11
  of 40 hunt episodes (executed 2026-09-28).** The food spawns 1.5–4.0 m ahead
  and 2.0–3.5 m up and counts within 0.8 m of the head tip
  (`configs/brachiosaurus/stage3_food_reach.toml:18-23`), so the statue
  reaches it on 27.5% of the audit's seeds (3042–3081; a spawn draw, about
  ±7%) against a `min_success_rate` of 0.5: more than half the bar is free
  ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3). Leaves with the
  brachiosaurus hunt's task revision (a paired comparison with the statue, or
  food spawned out of resting reach), which the gait plan does not schedule
  ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) §9).

- **MEDIUM** — **knee, shin and proximal-tail floor contact never ends an
  episode on trex, velociraptor, brachiosaurus or dibothrosuchus (read from
  the code, clearances measured 2026-09-28).** Their `_body_ground_geoms` hold
  only the torso (and belly), head and distal tail (`trex_env.py:361-368`,
  `raptor_env.py:219-226`, `brachio_env.py:231-237`,
  `dibothrosuchus_env.py:247-254`; velociraptor adds its neck), so a policy
  may kneel or crawl while the root stays above the height floor. Settled
  clearance of the lowest such geom, then the root drop allowed: trex tibia
  0.157 m, 0.226 m; velociraptor metatarsus already touching, 0.194 m;
  brachiosaurus shins 0.067 m, 0.186 m (0.336 m on locomotion); dibothrosuchus
  shins 0.047 m, 0.133–0.153 m. The compsognathus pair ends on any non-foot
  geom ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3). Plan: the
  reward kit's `terminate_on_leg_contact`, set in the dibothrosuchus and
  brachiosaurus `gait-r1` revisions ([gait plan](GAIT_QUALITY_PLAN_2026_09.md)
  PR-G6 and PR-G7); none is planned for trex or velociraptor.

- **LOW** — **the shared stance diagnostic reads only the forefeet on
  quadrupeds (read from the code 2026-09-28).** `derive_stance_info`
  (`stance_diagnostics.py:75`) takes `r_foot_contact` and `l_foot_contact`,
  which the brachiosaurus and dibothrosuchus envs set to their front feet
  (`brachio_env.py:410-411`, `dibothrosuchus_env.py:449-450`), so its
  unsupported duty and balance ignore the hind feet, which carry 79% of the
  certified dibothrosuchus stance's floor load
  ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §3). No quadruped gate
  reads it today. Plan: a four-foot `derive_stance_info`
  ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) PR-G2).

- **LOW** — **`zero_action_baseline.py`'s verdict judges a `stance_quality`
  stage by its reward rail alone (read from the code 2026-09-28).** It
  compares the statue with `min_avg_reward` whatever the gate kind
  (`scripts/zero_action_baseline.py:176-190`), so the trex, compsognathus and
  compsognathus_robot stances print "FAILS — a statue clears this gate"
  because the statue clears the collapse rail, not because of the stance
  gate's criteria, which the statue passes by design. Plan: PR-G2 prints the
  stance-gate verdict for `stance_quality` stages
  ([gait plan](GAIT_QUALITY_PLAN_2026_09.md)).

<!-- The items below come from the 2026-08-28 RL pipeline gap review
     (reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md, whose appendix B records
     what became of every finding). Each was re-checked against the code on
     2026-09-30; EP1 and EP4 were executed. -->

- **MEDIUM** — **a same-stage resume inside a stage's entry window drops the
  rest of its reward ramp and warm-up (read from the code 2026-09-30).** A
  node entered from its parent (`initialize_next_stage`) trains under
  `StageWarmupCallback` (default 100k steps) and, when its
  `forward_vel_weight` is positive, `RewardRampCallback` (from 0.1 to the
  stage's weight over 500k steps by default). A `resume_same_stage` load
  attaches neither: `_stage_entry_shaping_callbacks` returns nothing for that
  mode (`train_base.py:839-840`, called at :1316-1323), so the env trains at
  the stage's full `forward_vel_weight` from the first resumed step. Since the
  gap review's TC1 fix a continuation keeps the checkpoint's step counter
  (`train_base.py:1223-1237`, `reset_num_timesteps=not resuming` at :1342),
  and the ramp reads that counter (`curriculum/advancement.py:640-659`), so
  its position is recoverable; nothing re-applies it. The warm-up marker is
  cleared with a warning that the rest of the warm-up is not re-applied
  (`train_base.py:549-563`); the ramp's remainder goes without a log line. It bites when a
  session dies early in a locomotion or behavior stage and the RESUME cell
  continues it: 200k steps into a 0.1 → 1.0 ramp, the policy meets the other
  0.54 of the weight in one step. Fix: re-attach the shaping on a
  continuation, offset by the counter (not on a curated `--load`, which
  `_is_resume_continuation` tells apart), or at least warn. No cleanup PR
  owns it. (2026-08 gap review TC8)

- **MEDIUM** — **certification is judged on one fixed 40-seed block, and the
  held-out confirmation panel the stance design calls for does not exist
  (read from the code 2026-09-30).** `stance_quality/v1` implements items 1
  and 2 of its adopted rule; item 3, one predeclared held-out panel at
  n ≈ 100–180 for the full-horizon event, "is offline and deliberately lives
  outside this module" (`curriculum/stance_gate.py:49-51`), and no code runs
  it ([STAGE1_SPLIT_PLAN.md](STAGE1_SPLIT_PLAN.md) §2.3: "neither is item 3's
  held-out panel"). Every certifying panel runs on the registered block
  3042–3081 (`constants.py:34`, `PUBLICATION_SEED_START`): publication
  refuses a `certification_panel` role that starts anywhere else
  (`result_bundle/evidence.py:682-686`) and binds each stance panel row to it
  (:441-448). The panel is deterministic given the policy, so a marginal
  policy that happens to fit the block passes every re-check; by the review's
  arithmetic, one with a true full-horizon rate near 0.93 fits about 46% of
  blocks and would fail a fresh one about 54% of the time. The blocks do
  differ: the statue scores 119/120 over three of them
  (`configs/trex/stance.toml:318`). Seed replication (`certification_seeds`,
  the review's SS1 fix) counts training seeds, not fresh evaluation seeds.
  Fix: run the confirmation panel on a disjoint seed block before a stance is
  certified and record both blocks (`stance_gate_report.py --episodes`
  already sizes such a panel). No cleanup PR owns it. (2026-08 gap review SS3
  and SS4)

- **MEDIUM** — **a step that diverges in MuJoCo returns as an ordinary step
  (executed 2026-09-30).** `BaseDinoEnv.step` (`base_env.py:1173-1297`, its
  frame-skip loop at :1224) never reads `data.warning`, and on a bad `qvel`
  or `qacc` MuJoCo resets the state to the model's default pose (`qpos0`) and
  carries on. On the trex stance config, setting `qvel` to 1e12 and taking
  one zero-action step printed MuJoCo's "The simulation is unstable" warning
  and returned `terminated=False`, reward 1.101 and a finite observation, with the pelvis
  at 0.977 m (`qpos0`, not the settled stance) and
  `warning[mjWARN_BADQVEL].number` at 1, but no `termination_reason`. In
  training, a mid-episode divergence silently moves the animal back to that
  pose (clearing its drift and velocity) while the push clock, VecNormalize's
  statistics and the gate's length and reward metrics take in the impossible
  trajectory. Fix: compare the warning counters around the frame-skip loop
  and end the episode with its own `termination_reason`, logged. No cleanup PR
  owns it. (2026-08 gap review EP1)

- **MEDIUM (design gap)** — **every push of a recovery stage has the same
  magnitude (read from the code 2026-09-30).** `push_schedule` draws each
  push's start and heading from hash lanes `k*2` and `k*2+1`
  (`perturbation.py:95-120`), and `external_push_force` applies one scalar
  `force_newtons` to every push (:123-142), while the recovery design
  pre-generated push "times, directions, and magnitudes" per episode
  ([STAGE1B_IMPLEMENTATION_PLAN.md](STAGE1B_IMPLEMENTATION_PLAN.md) W1), so a
  certified recovery policy has met one force. Adding magnitudes needs a hash
  lane disjoint from {`k*2`, `k*2+1`} (renumbering the lanes would silently
  change every existing schedule and its null-controller pairing), and it
  changes the transition, so `SCHEDULE_IMPLEMENTATION`
  (`task_fingerprint.py:75`) and every recovery task digest move. No cleanup
  PR owns it. (2026-08 gap review EP2)

- **LOW** — **after a noisy reset one foot spawns just above the floor and
  reads 0 N for the first few steps (executed 2026-09-30).**
  `_settle_root_on_ground` (`base_env.py:1469-1500`) shifts the root so the
  lowest geom sits at the home clearance, which grounds one foot and can
  leave the other millimetres up. On the trex stance config
  (`reset_noise_scale` 0.05, zero action), seeds 0–3 read 0.0 N on one foot
  for the first 5, 3, 3 and 4 control steps, with a step reward of 1.86–1.99
  on those steps against 3.19–3.38 within two steps of the second foot's
  touchdown. The stance duty gate is unaffected
  (`settle_steps = 200` excludes the transient); the reward lost is small and
  the same for every policy, but any measurement window that opens at a
  reset reads the same zeros. Fix: settle each foot, or leave the first
  post-reset steps out of the support-conditioned terms; either one moves the
  digest-snapshot golden's `reward` lines, which capture noisy resets. No
  cleanup PR owns it. (2026-08 gap review EP4)

- **LOW** — **a `stance_quality/v1` stage that leaves out `min_eval_episodes`
  gets a 10-episode panel floor in training and in the recorded-gate reader,
  and a 40-episode one in the stance report and publication (read from the
  code 2026-09-30).** The SB3 curriculum manager's `StageThreshold` falls
  back to `DEFAULT_MIN_EVAL_EPISODES` = 10 (`curriculum/manager.py:29-36,73`)
  and hands it to the stance gate as a field copy (`stance_thresholds`,
  :76-92), the in-training diagnostics callback sizes the gate's duty count
  from the same 10 (`eval_diagnostics.py:636-639`), the recorded-gate reader
  the Drive summary uses falls back to it too (`reporting/gates.py:105`), and
  the species catalog publishes it (`species_catalog.py:467`), while the
  stance report, its probes and publication fall back to
  `DEFAULT_MIN_EVAL_EPISODES_STANCE` = 40 (`curriculum/stance_gate.py:142`,
  `reporting/stage_artifacts.py:226,515`, `result_bundle/evidence.py:509`).
  The gate schema requires `min_eval_episodes` for `task_success/v1` only
  (`curriculum/gate_schema.py:148-179`), so nothing stops a new
  `stance_quality/v1` TOML from leaving it out; the three that exist (trex,
  compsognathus and compsognathus_robot `stance.toml`) set 40, so it is latent. #519 named both
  values and kept them apart on purpose, because unifying them would change
  which panels the SB3 manager certifies (gap review DU4, part 2 not applied).
  Fix: require `min_eval_episodes` for `stance_quality/v1` in the gate schema,
  as `task_success/v1` does (the three stance TOMLs already set it), or unify
  the two fallbacks. No cleanup PR owns it. (2026-08 gap review DU4)

## Infrastructure

- **LOW** — `metrics.py` `velocity_consistency` explodes when mean velocity
  ≈ 0; thread-unsafe CSV appends under concurrent local runs. (June §3.3;
  CODE_REVIEW §2.1#1)
- **LOW** — **a node re-entered from its own `gate_verdict.json` prints a
  0.01 s control step in `training_summary.txt` (reproduced 2026-09-28).**
  When the SB3 notebook's chain loop re-runs in the same `RUN_DIR`, it takes
  a certified node's results from its verdict
  (`NODE_RESULTS[NODE.id] = dict(stage_result)`, the `same_run` branch). The
  verdict's `stage_result` projection (`_PERSISTED_STAGE_RESULT_KEYS`,
  `result_bundle/gate_verdict.py:67`) omits `sim_dt`, so
  `write_training_summary` falls back to 0.01 s
  (`reporting/text_summaries.py:156`): a reused compsognathus or
  compsognathus_robot node shows 1,000 steps as "10.00s sim time" instead of
  20 s, while a freshly trained one shows 20 s. No gate, verdict or digest
  reads it. CU-2 fixed the other 0.01 s fallback, in summaries built from
  `evaluations.npz`, but not this one. Fix: persist `sim_dt` in the
  projection (new verdicts only), or give `write_training_summary` the
  node's control step; no cleanup PR owns it yet. (2026-09 CU-2 review)
- **LOW** — **the final-checkpoint replay runs on raw observations, with no
  warning, when the final pair has lost its sidecar (reproduced
  2026-10-01).** `_record_stage_replays` records the selected checkpoint only
  from a pair whose `_vecnorm.pkl` exists and otherwise skips it with a
  warning (`reporting/stage_artifacts.py:1668-1676`). The final replay checks
  only the `.zip` (`:1705`) and passes `<stage>_final_vecnorm.pkl` (`:1646`,
  `:1723`), which `evaluation.record_stage_video` ignores when the file is
  missing (`evaluation.py:289`): the policy is replayed on unnormalised
  observations, which makes it a different policy, and its `_final.mp4` is
  written as usual. In a probe through the unchanged library code (the SB3
  classes, the loader, the plant checks and mediapy stubbed; the same at
  `02d98ca`), the selected replay's policy saw normalised observations, the
  final one's raw, and nothing was logged at WARNING or above. It is latent:
  every caller today hands the replays an intact final pair (the SB3
  notebook's chain loop and manual cell call `generate_stage_artifacts` right
  after training saved both files, and its JUDGE branch judges only an
  intact final pair, `checkpoint_pair_problem`). A direct call on a stage
  directory whose final `.pkl` is missing, for example one copied without
  it, would publish the misleading video. Fix: skip the final replay with a
  warning when its sidecar is missing, as the selected replay does, or have
  `record_stage_video` refuse a named `vecnorm_path` that does not exist; no
  cleanup PR owns it yet. (2026-10 CU-8b scouting)
- **LOW** — **a result-bundle JSON file that is not UTF-8 escapes as a raw
  `UnicodeDecodeError` instead of the reader's own refusal (reproduced
  2026-10-01).** These readers read with `encoding="utf-8"` but catch only
  `OSError` and `json.JSONDecodeError` (and their own errors), so the
  decode error, a `ValueError`, passes through: `result_bundle/audit.py:193`
  (the summary), `:230` (the artifact manifest) and `:343` (the plant
  identity), `reporting/bundles.py:530` (the previous manifest in
  `save_result_bundle`), `result_bundle/evidence.py:530` (the gate
  resolution), `result_bundle/manifest.py:216` (`verify_artifact_manifest`),
  `result_bundle/provenance.py:368` (`load_provenance`) and
  `curriculum/baseline_watch.py:66` (`read_zero_action_baseline`, which
  training calls when it builds its callbacks, so there a bad file stops
  the stage before it trains instead of skipping the advisory watch).
  `initialize_result_bundle` reads an existing `provenance.json` with no
  handler at all (`result_bundle/provenance.py:251`), so a file there that
  is not UTF-8, or not JSON, escapes raw too, also from
  `save_result_bundle` given a `run_id`. In a probe, a run directory
  holding only a `summary.json` with a 0xff byte inside a string makes
  `audit_result_bundle` raise `UnicodeDecodeError` instead of returning an
  audit error, and `load_provenance`, `verify_artifact_manifest` and
  `read_zero_action_baseline` do the same on such a file of their own; the
  same at `08f7bdd`. It is latent: the project's writers write UTF-8, so
  only a corrupt or hand-edited file reaches it. CU-8c closed the leak for
  `stage_config.json` alone, by catching `ValueError` in
  `save_result_bundle`, `audit_result_bundle` and
  `validate_evaluation_evidence`. Fix: catch `ValueError` in these
  handlers too, and give `initialize_result_bundle`'s read its own
  refusal; no cleanup PR owns it yet. (2026-10 CU-8c review)
- **LOW** — **the species `requirements.txt` files take any MuJoCo from
  3.0.0, and the stale-manifest error does not name the version (read from
  the code 2026-09-30).** `environments/trex/requirements.txt:2`,
  `environments/velociraptor/requirements.txt:2` and
  `environments/brachiosaurus/requirements.txt:1` ask for `mujoco>=3.0.0`,
  while the package pins `mujoco==3.10.0` (`pyproject.toml:25`), the version
  the plant digests are built on. An install from one of them gets a newer
  MuJoCo, and training then stops in `current_plant_identity` with
  "generated plant manifest is stale for <species>; run the plant-contract
  check before training" (`plant_contract/manifest.py:365-370`); only the
  check it points to names the cause ("plant manifest generation requires
  MuJoCo 3.10.0", :168-171). In the review's reproduction (MuJoCo 3.12.0)
  training stopped before its first step, so the cost is a two-step
  diagnosis, not a wrong result. Fix: pin or delete the three files
  (`pip install -e ".[train]"` is the documented install) and name the
  MuJoCo version in the stale-manifest error. No cleanup PR owns it.
  (2026-08 gap review OP9)

## Post-training artifacts (recommended additions)

1. Persist the curriculum gate history (`CurriculumManager.summary()` →
   `curriculum_state.json`) and the achieved gate metrics in
   `curriculum_results.csv`; include CLI overrides in the saved effective
   config. (July §4; June §6.5)
2. CLI curriculum parity with the notebook: call `generate_stage_artifacts`
   per stage and run the quality eval so `metrics.json` exists on that path.
   (July §4)
3. W&B: pass a render-capable eval env into `WandbCallback` (its video path
   is dead code), log final metrics before `finish()`, upload
   `best_model.zip` + vecnorm as W&B Artifacts, log the run URL into
   `stage_config.json`. (July §4; June §6.6)

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
existing sensor indices keep their positions, summed per foot in the SB3 env (for velociraptor the
frozen MJX registration is not edited, so under D-D17 that policy-interface revision is where it
declares itself SB3-only, [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §4.1) — and moves that
species' physics and policy fingerprints. Full evidence, method and reproduction in
[investigations/FOOT_SENSOR_VERIFICATION.md](investigations/FOOT_SENSOR_VERIFICATION.md);
re-check any repair with `environments/shared/scripts/foot_sensor_report.py`.

**Update (2026-09-28, gait audit):** a second under-read, shared by every species, comes from how
substeps are combined rather than from sensor scope. The contact-shaped rewards and foot-contact
info keys read `_aggregated_foot_contact_forces()`, the per-foot **minimum** over a control step's
physics substeps (`base_env.py:956-969`), so a foot touching on only some substeps reads as
airborne. Against the floor's normal force on the leg (above 0.1 N on at least half the substeps),
the velociraptor run `20260922_125248` reads 49% flight by touch against 33% (its stance 25% against
15%), and the in-training dibothrosuchus re-run `20260928_012318` reads 0.39 body weights against
1.00 with all of its floor load inside the touch-site volumes, its 10–30 ms contacts lost to the
minimum ([gait audit](investigations/GAIT_AUDIT_2026_09.md) §2.2). The gait plan measures gait on
floor contact instead ([gait plan](GAIT_QUALITY_PLAN_2026_09.md) §3.2; GQ-6, open). The opposite
error, touch over-reporting through compsognathus_robot's stacked soles, is under Training / RL.

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
  **Update (2026-09-28):** the stage now declares a collapse rail of 1050,
  0.60× the statue's 1745.8 (`configs/velociraptor/stage1_balance.toml:67`),
  which the statue still clears by design: it passes the gate on the certified
  stance's 30 resets (1694 ± 277), and that stance, `20260922_125248`,
  chatters and slides (the stance-gate HIGH under Training / RL;
  [gait audit](investigations/GAIT_AUDIT_2026_09.md) §4).
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
  a trained observation.
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
- **Not recommended:** porting the T-Rex stance correction here. That argument
  rests on a live stage-1 height term forcing knee travel through a
  near-singular joint, and the raptor env has **no height reward at all** —
  `height` appears in `raptor_env.py` only in the `pelvis_height` diagnostic
  and the shared height/tilt termination, never in a reward term.
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
- **LOW** — the T-Rex home keyframe's re-measurement checklist
  (`environments/trex/assets/trex.xml:552-556`) points at two things that no
  longer hold what it names: `configs/trex/stage1_balance.toml` (renamed
  `stance.toml`) and `mjx_config.py`'s `target_standing_z`, `_NATURAL_PITCH`
  and `healthy_z_range` (the frozen MJX registration keeps none of them since
  D-D17, cleanup PR-B). `trex_env.py`'s pointer was fixed (gap review SM7, #519);
  this one was left because any byte change to `trex.xml` moves the plant's
  source digest (`source_closure_sha256`), which the plant manifest, the
  species catalog and the digest-snapshot golden record, so the PR that edits
  the file regenerates all three (no checkpoint is invalidated: plant
  compatibility ignores source revisions). Fix it with the next deliberate
  `trex.xml` edit; the same stale `mjx_config.py` pointers in `trex_env.py`
  were reworded by cleanup CU-12, which folded in optional PR-C
  ([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §3.2). (read from the
  model 2026-09-30; 2026-08 gap review SM7)
- **Experiment** — with `implicitfast`, a `timestep` 0.002→0.004 A/B is
  worth running (halves sim cost if stable).

## Configs, docs & website

- Brachio stage-2 `natural_pitch = -0.15` while stages 1/3 use 0.0 —
  intentional? (June §4)
- `docs/investigations/REWARD_SCALE_REDESIGN.md` uses `*_bonus_weight` key
  names that don't exist. (June §5)
- **LOW** — `website/static/img/logo.svg` is 38,405 bytes because it is not a
  vector drawing: it is a 1000×1000 SVG wrapping one base64 PNG (500×500,
  28,552 bytes, with a 418-byte XMP text chunk) and has no paths and no
  SVG editor metadata, so SVGO or path simplification, the remedy
  [WEBSITE_PLAN.md](WEBSITE_PLAN.md) proposed, would not shrink it; the
  navbar's dark-mode `logo-dark.svg` (25,293 bytes) is built the same way
  (`website/docusaurus.config.ts:108-112`), its PNG (18,720 bytes) without
  the text chunk. Fix: recompress both PNGs, dropping `logo.svg`'s XMP chunk,
  or redraw the logo as a vector. (measured 2026-09-30; moved
  from WEBSITE_PLAN.md's "Optimize Logo SVG")

## Notebooks

- The SB3 notebook pins the plant compiler (`mujoco==3.10.0`) and
  `stable-baselines3[extra]==2.9.0`, but torch is unpinned and the
  `pyproject.toml` extras stay ranges; pin complete lockfiles for reproducible
  training. (July §5)
- **LOW** — the SB3 notebook trains SAC on 4 environments where the command
  line trains it on 8: `cli.py` raises `n_envs` from 4 to 8 for SAC unless
  `--n-envs` says otherwise (`cli.py:430-432`, applied at :454-456 for
  `train` and :515-517 for `curriculum`), while the configuration cell of
  [notebooks/sb3_training.ipynb](../notebooks/sb3_training.ipynb) sets
  `N_ENVS = 4` whatever `ALGORITHM` is and says nothing about SAC. The two
  entry points' default SAC runs differ twofold in collection against
  gradient updates; `n_envs` is recorded, so the difference can be found but
  is never flagged. Fix: mirror the CLI in the configuration cell, or say so
  there. (read from the code 2026-09-30; 2026-08 gap review NB9)
- **LOW** — **the Drive summary's `library_version` column holds two
  different versions (read from the code and probed 2026-10-02).** For a
  run with a result bundle it is the training backend's version:
  `_bundle_fields` (cell 9 of
  [notebooks/google_drive_summary.ipynb](../notebooks/google_drive_summary.ipynb),
  `:763`) takes `summary.json`'s `backend_version`, else `provenance.json`'s,
  and both record the Stable-Baselines3 version (`reporting/bundles.py:354`,
  `:643`); a partial or failed run, which `_canonical_csv_rows` reads
  without a summary (`:784-796`), takes the provenance one. For an older
  run without a bundle it is the mesozoic-labs version, which the legacy
  `scan_run` (cell 8, `:526`) reads from `stage_config.json`
  (`config.py:948`). The summary tables (cells 14, 16 and 18) and the
  export to `runs_summary.csv` (cell 24) show both under the one name. In
  a probe of the reader cells, a complete and a failed run read the
  backend version their bundles record and a flat run its stage config's
  mesozoic-labs version. Nothing in the repository reads the column back.
  Fix: export the bundle value as `backend_version`, and fill
  `library_version` from `provenance.json`'s
  `dependency_versions.mesozoic_labs`, which changes what
  `runs_summary.csv` says for those rows; no cleanup PR owns it. (2026-10
  CU-15 scouting)

## Testing / CI

- **LOW** — `hw_chassis_study.py`'s excitation drives hip pitch and knee in
  phase, unlike a real trot where knee flexion leads swing. This contributes to
  the knee pinning at its 22 N·m cap in all 18 runs, so that column is reported
  as "not a configuration discriminator"; the ab/ad conclusions are unaffected
  (that joint is driven at 0.25x and does discriminate).
  (2026-07 Dibothrosuchus review)
- TOML→env round-trip test: construct each env with each stage's
  `env_kwargs`, assert no unknown/unused keys. (June §6.8)
- **LOW** — **the site deploy waits on nothing but its own build (read from
  the code 2026-09-30).** `.github/workflows/deploy.yml` runs on a push to
  `main` that touches `website/**`, a `summary.json` or the workflow itself
  (:3-15), and its `deploy` job needs only the `build` job (`npm ci`, `tsc`,
  the site build) and the `main` ref (:72-78). A push with a hand-edited or
  stale `website/src/data/species.generated.json` therefore deploys while
  python-ci's `species_catalog --check` (`python-ci.yml:122-125`) fails on the
  same commit, and `main` has no required checks
  ([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §4.8 item 2). The
  workflow's comment (:7-11) leaves the wait to a required-checks setting, but
  it can be built in the tree. Fix: a `species_catalog --check` step in the
  build job, or a `workflow_run` trigger on python-ci that deploys only when
  it succeeds, with python-ci's path filters widened to all of `website/`
  (today a push that touches only the site config never runs python-ci); turning on required checks (the cleanup plan's decision 18 (a))
  closes it for everyone who cannot bypass them. No cleanup PR owns it.
  (2026-08 gap review CI8)
