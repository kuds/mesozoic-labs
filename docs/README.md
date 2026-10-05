# Documentation Map

The docs fall into a few categories with different lifecycles. Knowing which
category a document belongs to tells you whether to trust it as current,
read it as history, or update it when things change. Multi-doc topics that grow
past a couple of files get their own subdirectory (`investigations/`,
`reviews/`, `hardware/`), indexed below; `hardware/` also has a short README.

| Category | Lifecycle | Where |
|---|---|---|
| **Living reference** | Continuously updated; always current | `docs/` root ([KNOWN_ISSUES.md](KNOWN_ISSUES.md)) |
| **Plans & designs** | Updated until executed, then marked complete | `docs/` root |
| **Hardware & sim-to-real** | Plans/designs, grouped as a topical track | [`hardware/`](hardware/) |
| **Investigations & run analyses** | Point-in-time, dated; never rewritten (corrections are appended or cross-linked) | [`investigations/`](investigations/) |
| **Code & repo reviews** | Archived records; findings migrate to KNOWN_ISSUES | [`reviews/`](reviews/) |

## Living reference

- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — the single list of verified-but-unfixed
  findings and standing recommendations. Fixed items are deleted; full context
  stays in the archived review or investigation they came from.
- [NEXT_STEPS.md](NEXT_STEPS.md) — the entry point for the behavior-recipes
  program state, the run plan and the open items.
- [PLANT_CONTRACT.md](PLANT_CONTRACT.md) — versioned policy-interface,
  physics, visual, and source identities for MuJoCo models and checkpoints.
- [RESULT_BUNDLES.md](RESULT_BUNDLES.md) — canonical Colab/Google Drive result
  artifacts, per-deliverable publication (result schema v4: bundle status,
  `gate_verdict.json`, `ancestors/` records), provenance capture, and
  promotion validation.
- [TRAIN_DIRECTION_AND_TERRAIN.md](TRAIN_DIRECTION_AND_TERRAIN.md) — operator
  guide for the #540/#541 pilot pipeline (2026-09-15), command line only since
  the notebook-only PR-12 slice (D-D13); pilot outputs are evaluation-only
  (D-D9) until the nodes are re-homed as manifest nodes (consolidation PR-11).
- [SPECIES_CATALOG.md](SPECIES_CATALOG.md) — the generated species catalog:
  every species' specifications, current stages with their budgets and gates,
  success definitions, and the provenance-labelled historical run summaries;
  written whole by `python -m environments.shared.species_catalog`, never by hand.
- [SPECIES_NAMING.md](SPECIES_NAMING.md) — the display names that
  `configs/species_manifest.toml` owns, the stable IDs and the accepted
  aliases of the six species selections.

## Plans & designs

| Document | Date | Status |
|---|---|---|
| [ROADMAP.md](ROADMAP.md) | 2026-10-05 | Active — phased project timeline |
| [RECOMMENDATIONS.md](RECOMMENDATIONS.md) | 2026-03-25 | Dated snapshot — the codebase-wide improvement backlog as of 2026-03-25 (one item added 2026-07-27, §6.1 reworded 2026-07-13), not maintained; ROADMAP.md holds the phased plan and KNOWN_ISSUES.md the open problems, and D-D17 retired the JAX/MJX migration it recommends |
| [RL_TRAINING_PLAN.md](RL_TRAINING_PLAN.md) | 2026-03-26 | Dated planning snapshot (update notes 2026-04-18, 2026-09-13 and 2026-09-27) — trial list for the remaining SAC/PPO runs; its notebook setup predates the behavior chain loop (`BEHAVIOR`) and the id-named T-Rex stage configs, which the 2026-09-13 note maps; its trials 2 and 4 were Ray Tune sweeps, retired by D-D17 (the notebook and sweep JSONs are recoverable from the `0.3.8` tag) |
| [TREX_LEG_FLEXING_PLAN.md](TREX_LEG_FLEXING_PLAN.md) | 2026-07-27 | Option 1 (stance correction) implemented; options 2–4 open, step 5 (port to the raptor) withdrawn — see the raptor review |
| [MJX_CONVERSION_PLAN.md](MJX_CONVERSION_PLAN.md) | 2026-07-13 | Retired design record (D-D17) — cleanup PR-B removed the JAX/MJX runtime it built; a frozen MJX interface core stays for four species' policy-interface digests ([PLANT_CONTRACT.md](PLANT_CONTRACT.md)) |
| [WEBSITE_PLAN.md](WEBSITE_PLAN.md) | — | **Complete** (2026-09-30) — Docusaurus site improvements; of its two remaining items, the unused apex GIFs stay (maintainer decision, 2026-09-30) and the logo size went to KNOWN_ISSUES.md |
| [BALANCE_REWARD_METRICS.md](BALANCE_REWARD_METRICS.md) | 2026-03-16 | Withdrawn: Ray Tune retired (D-D17) — composite ASHA metric for stage-1 sweeps, never implemented |
| [PLANT_VALIDATION_AND_STAGE1_OBJECTIVE.md](PLANT_VALIDATION_AND_STAGE1_OBJECTIVE.md) | 2026-07-31 | Active — **read before any stage-1 work.** Why every reset was geometrically invalid, why the stage-1 objective's optimum is the zero-action policy, and what replaces the reward gate |
| [STAGE1_SPLIT_PLAN.md](STAGE1_SPLIT_PLAN.md) | 2026-07-31 | Implemented (rev 5; status 2026-09-30) — design record of the split of balance into 1a stance / 1b recovery: `stance_quality/v1` shipped 2026-08-01 and the recovery stage through STAGE1B_IMPLEMENTATION_PLAN.md by 2026-08-16; its held-out confirmation panel is open in KNOWN_ISSUES.md |
| [STAGE1B_IMPLEMENTATION_PLAN.md](STAGE1B_IMPLEMENTATION_PLAN.md) | 2026-08-15 | Dated planning snapshot (updates 2026-08-22, 2026-08-23 and 2026-08-28) — companion to STAGE1_SPLIT_PLAN.md mapping the recovery stage (1b) onto the tree; W1–W5 landed 2026-08-16 and P5 froze `recovery_quality/v1` on 2026-08-28 |
| [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) | 2026-09-05 | Active — adopted design turning the stage ladder into a DAG of behavior recipes (stand / walk / hunt / follow direction), each a separately certified, separately published policy. Implemented: Phase A (manifest v2, per-node gate verdicts, cross-run ancestor reuse, per-deliverable publication, the notebook chain loop; #528–#531), Phase B (the measured hunting gate `task_success/v1`, gate-configuration digests with reuse rule 7, seed replication as provenance; #532–#536), Phase C (the 3-dim command segment on all six species and `widen_checkpoint`; #537–#539) and Phase C½ (the walker, certified by the r13 run `20260914_123816`); Phase D's pilot pipeline (#540/#541) is folded back by the consolidation plan; Phase E (follow-on) is pending. §6.1 records the D-A, D-B and D-C decisions and §6.2 the D-D and G series, append-only |
| [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) | 2026-09-17 | Active — fifteen-PR sequence folding the #540/#541 pilots and the certified library into the recipes machinery, released 2026-09-20 in the notebook-first order of D-D13, with PR-14 split in three (D-D15); its status table is the landing record of every PR in the sequence and of the PRs outside it, the cleanup's included (Conventions below), and the sections of the PRs landed by `0.3.9` point to their full text at commit `20ab100`; paused after PR-10 on 2026-10-05 until one or two species walk well (its status table) |
| [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) | 2026-09-26 | Active — the cleanup left after #558 and its follow-up (#559): the retirement of Ray Tune, the Vertex AI tuning sweeps, the single-job Vertex AI route and GCS upload, mjlab and JAX/MJX behind a frozen MJX interface core, so no digest moves (D-D17); the smaller cleanup PRs CU-1..CU-17 and their order against consolidation PR-8..PR-15, the deferred PRs first (§3.1 item 5); the decisions they need, each with its outcome (§2); the digest-snapshot harness and its golden (D-D22); the 2026-09-25 eval-only check of the certified walkers on the direction/terrain recipes; and the lessons of the #558 reviews. §3.1 keeps the cleanup's records up to CU-17 as its history |
| [GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md) | 2026-09-28 | Proposed, nothing decided — acts on the 2026-09-28 gait audit ([investigations/GAIT_AUDIT_2026_09.md](investigations/GAIT_AUDIT_2026_09.md)): gait measured on floor-truth contact through the per-substep hook, per-episode gait gates (`locomotion_gait/v1`, later `stance_quality/v2` and `recovery_quality/v2`), per-species task revisions (`gait-r1`) that change what the rewards pay, and the PR-G0..PR-G10 sequence; eighteen open decisions (GQ-1..GQ-18); its code PRs come after the 0.3.9 cut and, by the maintainer's choice of 2026-10-02, after the deferred cleanup; §10 holds a prompt for continuing the work in a fresh session |
| [REFACTORING.md](REFACTORING.md) | 2026-03-19 | **Complete** — v0.3.0 consolidation plan |
| [CODE_CONSOLIDATION.md](CODE_CONSOLIDATION.md) | 2026-03-19 | **Complete** — v0.3.0 implementation record |

## Hardware & sim-to-real

Plans and feasibility studies for physical robots, grouped in
[`hardware/`](hardware/) with a reading order in its
[README](hardware/README.md). Nothing here is built yet.

**Headline conclusion:** there is no buildable-today, sub-$10k, two-legged
*running* recipe — the limit is control, not motors. The recommended sub-$5k
first physical platform is a juvenile-Psittacosaurus shell on the open Pupper V3
quadruped; reliable walking/trotting comes before speed experiments. A
clean-sheet, catalog-actuated _Dibothrosuchus elaphros_-inspired quadruped
becomes credible near a $10k parts cap, with 2 m/s as repeatable acceptance and
3 m/s as a short-sprint stretch target.

| Document | Date | One-liner |
|---|---|---|
| [hardware/SIM_TO_REAL_PLAN.md](hardware/SIM_TO_REAL_PLAN.md) | 2026-07-19 | Sim-to-real feasibility & phased plan — the six sim↔hardware gaps |
| [hardware/HARDWARE_BOM.md](hardware/HARDWARE_BOM.md) | 2026-07-19 | Parts/cost & buildability for *walking* Compsognathus (~$950) + Velociraptor (~$8.7k) |
| [hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md](hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md) | 2026-07-24 | *Running* robots: raptor scaling, a sourced sub-$5k Psittacosaurus/Pupper V3 starter, and a custom ~$10k Dibothrosuchus-inspired speed tier |

## Investigations & run analyses

Dated, evidence-driven analyses of training runs and reward behavior, in
chronological order. Each is frozen at its date; follow-ups cross-link rather
than rewrite. Notes dated before D-D17 cite code it retired (the sweeps,
mjlab, the single-job Vertex AI route and GCS upload, and the JAX/MJX runtime); the release tag `0.3.8`
(`afad625`), which predates the retirement, and git history keep that code
reachable, so check out the tag or the commit a note names to reproduce one.

| Document | Date | One-liner |
|---|---|---|
| [TRAINING_REVIEW.md](investigations/TRAINING_REVIEW.md) | 2026-03-25 | Review of 140+ SB3 runs (Feb–Mar); empirical basis for the low-penalty stage-2 reward recipe |
| [TRAINING_REVIEW_JAX_STAGE1.md](investigations/TRAINING_REVIEW_JAX_STAGE1.md) | 2026-04-01 | T-Rex JAX/MJX stage-1 run review (KL-halt behavior) |
| [REWARD_DISCREPANCY_INVESTIGATION.md](investigations/REWARD_DISCREPANCY_INVESTIGATION.md) | 2026-04-02 | T-Rex stage-1 reward vs episode-length inconsistency root cause |
| [REWARD_SCALE_REDESIGN.md](investigations/REWARD_SCALE_REDESIGN.md) | 2026-04-18 | Stage-3 terminal-bonus rescale analysis (implementation deferred) |
| [STAGE2_INVESTIGATION.md](investigations/STAGE2_INVESTIGATION.md) | 2026-07-09 | Root cause of the velociraptor stage-2 locomotion collapse (bounded-plant actuator clipping) |
| [STAGE2_RECOMMENDATIONS.md](investigations/STAGE2_RECOMMENDATIONS.md) | 2026-07-11 | Replication review of run 20260711_165924, corrected fall-penalty math, ranked plan — validated 2026-07-12 by run 20260711_235303 (stages 1–3 cleared, stage-2 record); §5 has the outcome and cross-species lessons |
| [VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md) | 2026-07-20 | Commit A physics probe and A-only PPO run 20260720_203454; natural-lean reward-conflict diagnosis, pending Commit B validation, and fresh-run decision rules |
| [TREX_HOME_EQUILIBRIUM.md](investigations/TREX_HOME_EQUILIBRIUM.md) | 2026-07-23 | T-Rex neutral-stance basin repair, home-residual action mapping, and live plantar contact sensors |
| [TREX_STAGE1_LEG_JITTER.md](investigations/TREX_STAGE1_LEG_JITTER.md) | 2026-07-24 | Jittery leg movement in run `20260723_204941` stage 1 (SB3 PPO, 6.0M steps): root cause in the exploration/entropy schedule (`algo_std` stuck near 1.0), not the reward shaping; one-line config fix applied pending a validation run |
| [TREX_REVIEW_2026_07.md](investigations/TREX_REVIEW_2026_07.md) | 2026-07-28 | T-Rex simulation review of tree `081a2020`: `environments/trex/**`, `configs/trex/**` and the shared code the T-Rex executes, with findings copied to KNOWN_ISSUES |
| [FOOT_SENSOR_VERIFICATION.md](investigations/FOOT_SENSOR_VERIFICATION.md) | 2026-07-31 | Foot touch sensors versus `mj_contactForce`: discharges the STAGE1_SPLIT_PLAN §7.2 sensor-verification prerequisite — whether `unsupported_duty = 0.209` on a plant that never falls is real airtime or a sensor under-reporting |
| [TREX_STAGE1_BOUNCE_2026_08.md](investigations/TREX_STAGE1_BOUNCE_2026_08.md) | 2026-08-05 | Three 10M stage-1 runs (one pass, two phase-locked bounces at 1/6 and 1/5 duty). Why raising `leg_home_pose_weight` could not work, why the bounce is 450 points *worse* than not bouncing and so is not a reward problem, the filter probe confirming the tremor is load-bearing, and why seed replicates outrank any further reward tweak |
| [TREX_STAGE1_PASSIVE_TOES_RUN_2026_08.md](investigations/TREX_STAGE1_PASSIVE_TOES_RUN_2026_08.md) | 2026-08-09 | Run `20260808_230537` (seed 42, 10M steps) on the passive-toes plant (physics r7 / policy interface r10, action_dim 15) — the first honest 10M steps; FAIL at the stance gate; PRs #499–#501 |
| [TREX_STAGE1_NARROW_TOLERANCE_RUN_2026_08.md](investigations/TREX_STAGE1_NARROW_TOLERANCE_RUN_2026_08.md) | 2026-08-10 | Run `20260809_155206` (seed 42, 10M steps) with `leg_home_pose_tolerance` 0.20 → 0.10 rad and re-derived rails (#502): the knee-lock run, FAIL at the stance gate |
| [TREX_STAGE1_GATE_PASS_RUN_2026_08.md](investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md) | 2026-08-11 | Run `20260810_145546` (seed 42, 10M steps) on the filtered plant (policy interface r11: 10 Hz command low-pass, #503; the #504 shaping pack; rails re-derived) — the first certified stance GATE: PASS |
| [TREX_STAGE1_SEED43_REPLICATE_2026_08.md](investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md) | 2026-08-15 | Run `20260815_014118` (seed 43, identical gate-pass configuration): FAIL at duty 0.0597 / UCB 0.0747 — the first seed replicate; n = 2 says the configuration is seed-dependent, not solved |
| [TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md](investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md) | 2026-08-22 | The recovery stage (1b): first successful recovery training runs, the P3 safe-set calibration, the derived disturbance (165.5 N for 0.2 s from the plant), and what the evidence does and does not support |
| [TREX_STANCE_WIDENED_INTERFACE_2026_09.md](investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md) | 2026-09-14 | **(template — re-scoped 2026-09-19 in its appended §6; seed-44 results filled 2026-09-20 and 2026-09-23, §7–§8 appended, §9 correction 2026-09-23)** The certified stance parents `20260810_145546` / `20260815_205206` widened to the Phase C interface (policy interface r13) with `widen_checkpoint` + `WIDEN_FROM`, re-paneled under the new task hash, one recovery freeze re-rolled from the widened handoff, and the C½ walker; the sessions' exact knobs and commands are written. The seed-42 widen and the C½ walker sessions are superseded by the fresh r13 run `20260914_123816` (stance and locomotion both certified); the seed-44 widen of `20260815_205206` (an r11 archive two revisions behind r13, `WIDEN_MAX_REVISION_GAP = 2`, decision D-C17; KNOWN_ISSUES) ran 2026-09-20 as `20260920_010912`: re-panel identical to the r11 certificate, recovery certified 2026-09-21, trex stance at replication 2 (the notebook knobs named there left with consolidation PR-14a, D-D14; a new widen is the command-line recipe in [PLANT_CONTRACT.md](PLANT_CONTRACT.md), per the note's appended §10) |
| [DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md) | 2026-09-17 | Per-species survey of the maintainer's Drive under the Phase C interface: trex run `20260914_123816` (seed 42, r13; stance and locomotion both `gate_verdict.json` PASS) is the certified walker `TRUNK_FROM = "auto"` reuses, seed 43 (`20260915_160239`) failed the stance duty rail, the seed-44 r11 stance `20260815_205206` and the compsognathus r1 stance `20260909_162812` are widen candidates, the July velociraptor and brachiosaurus runs are fresh starts; ends with the six recommended training sessions; a 2026-09-23 correction is appended (the seed-42 run's records cannot be rebuilt in place) |
| [GAIT_AUDIT_2026_09.md](investigations/GAIT_AUDIT_2026_09.md) | 2026-09-28 | Replays of fifteen nodes (every certified node on Drive, the failed dibothrosuchus walk, the in-training dibothrosuchus re-run and the July brachiosaurus walker) against floor-truth foot contact: two of the five certified walkers are genuine (compsognathus walks, velociraptor runs) and three hop on both feet (trex `20260914_123816` and `20260925_033501`, compsognathus_robot `20260924_031815`); one of the six certified stances is clean; the zero-action statue passes every stance gate; the robot's touch sensors count sole-on-sole contact as floor support; the in-training dibothrosuchus re-run `20260928_012318` is a three-legged skid its gate would pass; evidence files (the probe, a per-node CSV, sha256 sums) in [gait_2026_09/](investigations/gait_2026_09/README.md) |

## Code & repo reviews

Archived point-in-time reviews in [`reviews/`](reviews/) —
[VELOCIRAPTOR_PLANT_REVIEW.md](reviews/VELOCIRAPTOR_PLANT_REVIEW.md)
(2026-07-27, anatomy and mechanics audit of the raptor plant against published
*Velociraptor* measurements; findings open, execution deferred until the T-Rex
clears stages 1–3),
[REPO_REVIEW_2026_06.md](reviews/REPO_REVIEW_2026_06.md),
[REPO_REVIEW_2026_07_RL_GCP.md](reviews/REPO_REVIEW_2026_07_RL_GCP.md),
[TREX_REVIEW_2026_08_MERGES_AND_NEXT_STEPS.md](reviews/TREX_REVIEW_2026_08_MERGES_AND_NEXT_STEPS.md)
(2026-08-23, run `20260821_142144` and merges #506–#510),
[RL_PIPELINE_GAP_REVIEW_2026_08.md](reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md)
(2026-08-28, pipeline gaps and cleanup findings at `ea0d339`; its appended
appendix B records what became of every finding as of 2026-09-30, and its open
findings are in KNOWN_ISSUES),
[CODE_REVIEW.md](reviews/CODE_REVIEW.md) (superseded). Open findings live in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md), not here.

## Conventions

- New run analysis or root-cause doc → `investigations/`, dated in the header,
  linked from the table above.
- New code/repo review → `reviews/`, findings copied into KNOWN_ISSUES.
- The root README is the repository's front page: what the project is, how
  to install it, one quick start and where to go next. Run metrics, stage
  budgets, flag references and status narrative live on the pages it links
  to; its species and notebook tables, and the species catalog, are generated.
- Don't rewrite a dated document when conclusions change — append a correction
  note that links to the newer analysis (the appended §6–§10 of
  [investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md](investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md),
  or appendix B of
  [reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md](reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md), is the pattern).
- Decision ids (D1–D5, the D-A, D-B, D-C, D-D, G and GQ series, and the cleanup
  plan's §2 row numbers) are used exactly as recorded and never renumbered; a
  decision row changes only by an appended, dated amendment.
- Relative Markdown links only, and every link must resolve.
- **One landing record.** A PR's landing (its number, merge date and CI
  result) is recorded by the PR after it, in two places only: the PR's row in
  the status table of
  [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) (a PR outside
  the consolidation sequence goes in the cleanup row or a row of its own) and
  its number in its entry in [CHANGELOG.md](../CHANGELOG.md). Every other
  document, a plan's own PR sections included, points there instead of
  restating it, and states order without tense ("the deferred cleanup comes
  before PR-8"). The rule governs landings from cleanup CU-17 on (the
  maintainer's ruling of 2026-10-02). The records written before it are
  history and take no new landing notes. The cleanup plan (§3.1 above all),
  the decision rows (append-only) and dated notes keep theirs; cleanup CU-17
  cut running accounts and landing lists, among them the consolidation
  sections landed by `0.3.9`, NEXT_STEPS.md's landing narrative and table,
  the cleanup plan's header chronicle and §1 items 1 and 2, this file's plan
  rows and the list in the KNOWN_ISSUES pilots entry, pointing to the status
  table or to a commit that holds the old text (`20ab100` for those
  consolidation sections).
  `environments/shared/tests/test_landing_records.py` fails a later PR's
  landing record ("landed as #NNN", "#NNN merged", "#NNN's CI", a row of a
  landed-PR table) in the root README, CONTRIBUTING, `results/README.md`, the
  species READMEs and every document under `docs/` outside `investigations/`
  and `reviews/`.
