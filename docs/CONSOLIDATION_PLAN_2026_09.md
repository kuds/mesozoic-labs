# Consolidation plan: behavior recipes and the direction/terrain pilots (2026-09-17)

**Date**: 2026-09-17 (the review session); migrated into the repository by the
2026-09-19 documentation pass. **Assessed at**: `main` @ 723f58f (the merge
of #541). **Scope**: the recipes program (PRs #528–#539) and the direction/terrain
pilots with their certified library (PRs #540–#541). **Companions**:
[BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) (the design of record this
plan folds the pilots back into; its §6.2 carries the D-D and G series recorded
in §6 and §7 below), [NEXT_STEPS.md](NEXT_STEPS.md) (the training sessions that
run in parallel with this sequence, and the Drive state they rest on),
[investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md)
(the 2026-09-17 Drive survey, the note of record for every run fact cited
here),
[TRAIN_DIRECTION_AND_TERRAIN.md](TRAIN_DIRECTION_AND_TERRAIN.md) and
`CERTIFIED_MODELS.md` (the pilots' own guides; the sequence
shortens the first and PR-5 deleted the second on 2026-09-20). Every `file:line` below was re-read
at 723f58f while the plan was written and drifts after it (with every merge
from #542 on); read them as anchors, not contracts. Where a first estimate was
corrected on re-reading during the review, the corrected figure is used.

## Status (2026-10-06)

This table and each PR's CHANGELOG entry are the only record of a PR's landing, the cleanup's PRs included
(the one-landing-record rule, [README.md](README.md#conventions), from cleanup CU-17 on); other documents point here.

| Item | State |
|---|---|
| Baseline | `main` @ 723f58f. Line numbers in this document were re-read at that commit and drift after it. |
| PR-1 (stop the live disconnect) | **Landed** as #542 on 2026-09-16: notebook default `PUBLISH_CERTIFIED = False`, pin flipped. |
| Automatic trunk selection (D-A25; settles D-D4) | **Landed** as #543 on 2026-09-16: `environments/shared/ancestors.select_trunk`, notebook default `TRUNK_FROM = "auto"`, CLI `curriculum --trunk-from auto`. Canonical chains no longer consult the certified library; widen sessions select no trunk. |
| PR-2 (record the decisions, fix the stale docs) | **Executed by the 2026-09-19 documentation pass**: this document, the D-D and G rows in BEHAVIOR_RECIPES_PLAN.md §6.2, the docs index, the README roadmap bullet, the CHANGELOG, NEXT_STEPS.md, the Drive survey note (investigations/DRIVE_RUN_SURVEY_2026_09.md), four KNOWN_ISSUES.md entries, the template note's appended §6 and one paragraph in website/docs/training/recipes.md. |
| PR-3 .. PR-15 | **Released 2026-09-20** in the notebook-first order of decision D-D13 (§6): PR-3, PR-4, PR-5, PR-6, a notebook-only slice of PR-12, PR-14, then PR-7 .. PR-11, the rest of PR-12, PR-13, PR-15. PR-14 lands as PR-14a, PR-14b and PR-14c, in that order (decision D-D15, 2026-09-24). |
| PR-3 (bound the SB3 CI job) | **Landed** as #546 on 2026-09-20, together with the widened-root bundle-write fix: the SB3-free suites leave the `test-sb3` lists (the `test` matrix runs them; verified locally with SB3, torch and ray blocked), the full six-species and four-notebook-parameter sets run on the nightly schedule and under the `full-ci` label, one real-PPO smoke per body of work stays on every PR, `walker` is module-scoped. Measured on the #544 merge run: the job took 52 minutes (notebook smoke 6, behaviors 14, integration 31); measured on #546's two CI runs: the lean job 48:02 (smoke 2:56, behaviors 4:59, integration 38:00 for 1,384 tests), the labelled full job 45:21 (5:51, 9:01, 28:39), coverage 90 percent against the 70 gate. The integration step is now the whole cost and swung by ten minutes between two runs of the same commit, and the JAX job (39 to 51 minutes) is the longest pull-request job: both are follow-up candidates, not PR-3's. |
| PR-4 (delete the canonical library wrapper and the notebook hooks) | **Landed** as #547 on 2026-09-20 (PR-3 landed as #546): `certified_canonical.py`, its test and `test_sb3_notebook_certified.py` deleted; the notebook loses the three library knobs, the stamp, copy and publish blocks (about 90 cell lines; 41 cells, chain loop at index 23) and never publishes; `certified_comparison.py` moves to PR-5 (its only importer is PR-5 code); pins re-pointed, two ported; `train_behaviors --auto-source` without `--resume` refuses with an explicit message; the stamp's post-construction `set_random_seed` call left with it, so a run is seeded once, at model construction, and is not bit-reproducible against a pre-PR-4 run. |
| PR-5 (delete the certified library and the trainer's library path) | **Landed** as #548 on 2026-09-20 (PR-4 landed as #547): `certified_library.py`, `certified_comparison.py`, their tests, `test_behavior_publication.py` and `docs/CERTIFIED_MODELS.md` deleted (1,602 whole-file lines); `train_behaviors` takes an explicit `--checkpoint` / `--vecnormalize` pair in every mode; `certify_and_publish_behavior` and the behavior certificate writer gone (no run writes `certification/certificate.json` until PR-13); the notebook loses `SOURCE_SELECTION` and its prose (2,463 lines); about 1,880 net code and configuration lines removed. |
| PR-6 (delete the T. rex pilots, the `[pilot]` dialect and the shim) | **Landed** as #549 on 2026-09-20 (PR-5 landed as #548): `configs/trex/behavior_pilots/` (8 TOMLs, 239 lines), pyproject.toml's package-data line and `environments/trex/scripts/train_behaviors.py` deleted; `read_recipe` reads `[behavior]` only and refuses a recipe without one instead of defaulting to trex; `behavior_notebook` loses the six pilot aliases and the dead `mesozoic.behavior-pilot-run/v1` reader; the trex suite reads `configs/trex/behaviors/`; `mesozoic.trex-command-terrain/v1` stays accepted until PR-7 deletes its emitter with `TRexBehaviorEnv`; measured −256 code and configuration lines. |
| PR-12, notebook-only slice (D-D13) | **Landed** as #552 on 2026-09-24: the notebook loses `COMMAND_TERRAIN_BEHAVIOR`, the ten `BEHAVIOR_*` knobs, the eleven direction/terrain dropdown values, the six behavior cells (7/19/20/33/34/39 in the 22c1fc8 numbering, 7/20/21/34/35/40 at 2e77150) and the 15 guard sites (the plan's 14 plus the guarded archive-load preflight; the guarded code dedented, two lint fixes aside): 41 cells, 23 code, 2,463 lines → 35 cells, 19 code, 2,329 lines; `behavior_notebook.py` (294 lines) and `test_behavior_notebook.py` (815) deleted, their canonical halves ported to `test_sb3_notebook_pins.py` (the configuration defaults and free-form stage ids, the dropdown's JSON annotations, the setup cell's `REPO_REF` safety) with `_canonical_source` removed; CI drops the deleted test from the SB3 list and the wheel step names the eleven recipe files itself. `train_behaviors.py` stays the pilots' command-line path until the rest of PR-12. Measured: −996 code, test and CI lines, −134 notebook source lines. The same PR carries one bug fix found while mapping PR-14: the training-curves cell no longer writes PNGs into the sealed bundle, which had stopped every completed "Run all" at the cleanup cell before the auto-disconnect (CHANGELOG "Fixed"; +1 notebook line, one executed pin). |
| PR-14a (the storage path, D-D15) | **Landed** as #553 on 2026-09-24: the widen cell, `WIDEN_FROM`, `WIDEN_MAX_REVISION_GAP` and `select_trunk(widen_from=)` deleted (D-D14; `widen_checkpoint` stays the command-line widen path, and its seed and verdict guards become on-disk refusals in the storage and resolve cells); `RUN_ID` a configuration-cell knob resolved into the `_ACTIVE_RUN_ID` memo; a session that would judge or train a node into a complete run refused at the end of the resolve cell, before anything is trained or written (the KNOWN_ISSUES bug of 2026-09-23, CHANGELOG "Fixed"), with the chain loop refusing, before the write, what the resolve cell cannot predict, the manual and resume cells refusing the same write and the zero-action cell no longer rewriting a complete run's copy: 35 cells, 19 code, 2,330 lines → 34 cells, 18 code, 2,271 lines. Measured (`git diff --numstat` against `2ebed89`): code +380 / −20, tests +894 / −680, notebook JSON +163 / −230, docs +543 / −217. *Since 0.3.9: cleanup ROW-4/6 replaced the resolve cell's D-C13 refusal of a trunk over an unjudged widened root (D-D15's amendment of 2026-10-03): the chain loop judges a node the run holds trained but unjudged before it consults any trunk, which generalises the chain-loop repeat of that refusal PR-14a left out, and the resolve cell now ends with the complete-run refusal and `record_trunk_run` (`trunk_run.json`, decision 4 (a)).* |
| PR-14b (one disconnect path, videos, the baseline cells; D-D15) | **Landed** as #554 on 2026-09-24: `disconnect_runtime`, a new `halt` and `display_stage_videos` move from the infrastructure cell into `environments/shared/notebook_runtime.py` with the knobs passed at call time; the chain loop's gate refusal calls `halt`; videos play through IPython's `Video` (the `_HAS_MEDIAPY` probe gone); the random-baseline cell is deleted; the zero-action cell becomes its knobs and one call to `zero_action_baseline.preflight` (payload, run copy and complete-bundle skip byte-identical): 34 cells, 18 code, 2,271 lines → 33 cells, 17 code, 2,079 lines. Measured (`git diff --numstat` against `3571b24`, new files counted whole): code +183 / −1, tests +245 / −50, notebook JSON +27 / −229, docs +131 / −37. |
| PR-14c (`train_stage` over `train_base.train`; D-D7, D-D11, D-D15) | **Landed** as #555 on 2026-09-24: `train_stage` becomes its argument refusals, the node banner, one `train_base.train(..., report_metrics=False, save_on_interrupt=False)` call and the evaluation; `train()` seeds construction (an algorithm-block seed kept), records `run.duration_seconds` at the final save and takes `parent_run_id` and an explicit `vecnorm_path`; four loads of seeded archives pass `seed=None` (the CLI's two panel loads, the task-success re-roll, the Ray warm start); `evaluate_stage_checkpoints` moves beside `generate_stage_artifacts`: 33 cells, 17 code, 2,079 lines → 33 cells, 17 code, 1,492 lines. Measured (`git diff --numstat` against `d63faff`): code +356 / −31, tests +402 / −278, notebook JSON +45 / −632, docs +168 / −46. *Since 0.3.9: cleanup CU-10b aligned `train_curriculum`, the `curriculum` subcommand, which PR-14c left out (D-D7): each node it trains seeds model construction and records `run.duration_seconds` (D-D11's amendment of 2026-10-03).* |
| PR-7 (one behavior env, part 1: the ground-height hook; `TRexBehaviorEnv` deleted) | **Landed** as #556 on 2026-09-25 (measured on its CI: SB3 job 47:09, JAX job 49:05, coverage 90 percent): `BaseDinoEnv._ground_height_at` / `_clearance`, every species height reward, head/snout clearance and height termination and the step loop's substep head/snout minima read through it (velociraptor's root termination added to the plan's list); the mixin's four height overrides and re-derived height reward go; the step loop aggregates no root min/max (the plan's text asked for one; the reason is at the end of this row); `environments/trex/envs/behavior_env.py` (497 lines) and the `mesozoic.trex-command-terrain/v1` schema deleted, the neck probe behind `TRexEnv._terrain_contact_probe_geoms`; the trex behavior tests fold into the shared suites, parametrised over the six species. Canonical plane rollouts bit-identical, `plant_contract --check` current. Measured (`git diff --numstat` against `d1b8120`): code +192 / −595, tests +1,016 / −1,039, CI +1 / −2, docs +186 / −25. *Moved here from the PR-7 section by cleanup CU-17 (2026-10-04), its only copy, on why the step loop aggregates no root min/max:* "A canonical consumer would change terminations with no digest moving and break SB3/MJX parity (MJX compares the boundary sample, mjx_env.py:1310-1311); a behavior-only consumer would keep the override the plan deletes." The behavior env's root height check therefore became the canonical boundary sample of the root clearance (behavior-only; D-D9). *Corrected the same day:* both premises of the first half have lapsed. Cleanup PR-B retired the MJX runtime (of `mjx_env.py` only its 90-line frozen part remains, and `mjx_env.py:1310-1311` no longer exists), and since cleanup CU-11 (#573) the digest golden's `reward` lines record each stage's termination reasons, so a canonical termination change now moves the golden. |
| Notebook follow-up to PR-7 (outside this sequence) | **Landed** as #557 on 2026-09-25 (measured on its CI: SB3 job 30:24, JAX job 47:26, coverage 90 percent): the RESUME cell resolves `RESUME_STAGE` through the manifest as the manual cell does (`"locomotion"` raised `KeyError`; `2` and `"locomotion"` now resume the same node), the section 6 prose names both forms and the chain-loop step after it, a node off `BEHAVIOR`'s chain is refused before anything is trained (the chain loop would never judge it), the storage cell's trunk comment drops "finished bundle", and the auto-trunk tie-break is described as the greatest run directory name in the notebook, the `--trunk-from` help, the docstrings and the website recipes page (D-A25 clarified); trunk selection is unchanged, and a node resumed by its number trains exactly as before. |
| Notebook safety (outside this sequence; D-D16) | **Landed** as #558 on 2026-09-25 (measured on its CI: SB3 job 46:19, JAX job 50:21): the RESUME cell trains nothing for a node that holds a gate verdict or its final pair and refuses a spent budget without the final pair; `QUICK_TEST` runs move to `<algo>_quick_test/`, out of the trunk selection and a real run's replicate scan; the RESUME cell moves ahead of the chain loop (a resume is `RUN_ID` + `RESUME_STAGE` + Run all) and the sections regroup (33 cells, 17 code, 1,557 lines). No digest moves. The first PR of the cleanup the maintainer put before the direction/terrain path on 2026-09-25. |
| #558 follow-up (outside this sequence; D-D16 amended) | **Landed** as #559 on 2026-09-26 (measured on its CI: SB3 job 46:51, JAX job 48:10): the RESUME cell and the chain loop check the final pair like a periodic one through `curriculum.checkpoint_pair_problem` (a pair cut short during the final save is resumed over, never judged), the run memo counts only in the tree `SPECIES`, `ALGORITHM` and `QUICK_TEST` select, a resume `RETRAIN_FROM` covers is refused, and the docs the post-merge reviews of #558 found wrong are corrected (33 cells, 17 code, 1,599 lines). No digest moves. |
| Cleanup and backend retirement (outside this sequence; D-D17) | **Planned** in [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md), which landed as #560 on 2026-09-26: a CI-signal PR (mypy with SB3 installed), then PR-A (Ray Tune, the Vertex AI tuning sweeps, mjlab) and PR-B (JAX/MJX, keeping a frozen MJX interface core so no digest moves), then sixteen smaller cleanup PRs. The maintainer took D-D17 on 2026-09-26 and widened it to the single-job Vertex AI route and GCS artifact upload too, which leave in a PR of their own, PR-A2, with an end-to-end test of the command-line curriculum path (the cleanup plan lands it after PR-A). The CI-signal PR (CU-1, D-D18) landed as #561 on 2026-09-26 (measured on its CI: SB3 job 48:54, JAX job 51:20, coverage 90 percent); CU-3 (atomic run-tree records and the final and handoff checkpoint pairs, D-D20) landed as #562 on 2026-09-26 (measured on its CI: SB3 job 47:57, JAX job 50:05, coverage 90 percent), and the CHANGELOG release cut (D-D19) landed as #563 on 2026-09-27 (measured on its CI: SB3 job 46:21, JAX job 35:47, coverage 90 percent), tagged `0.3.8` by the maintainer; PR-A (Ray Tune, the Vertex AI tuning sweeps and mjlab) landed as #564 on 2026-09-27 (measured on its CI: SB3 job 47:01, JAX job 39:11, coverage 91 percent), PR-A2 (the single-job Vertex AI route and GCS upload) landed as #565 on 2026-09-28 (measured on its CI: SB3 job 34:52, JAX job 34:43, coverage 91 percent), and PR-B (the JAX/MJX runtime, behind the frozen MJX interface core) landed as #566 on 2026-09-28 (measured on its CI: SB3 job 37:03, coverage 91 percent; PR-B removed the JAX job), which completes D-D17's removals; the gait audit and its plan (PR-G0 of [GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md), docs only and outside the cleanup and D-D21's gate) landed as #567 on 2026-09-28 (measured on its CI: SB3 job 46:07, coverage 91 percent), and CU-2 (the `render_mode="human"` crash and the 0.01 s control step stage summaries built from `evaluations.npz` assumed) landed as #568 on 2026-09-29 (measured on its CI: SB3 job 37:11, coverage 91 percent), and CU-4 (the test helpers: one notebook-cell reader for the tests and CI's notebook check, the four duplicated library-only pins deleted) landed as #569 on 2026-09-29 (measured on its CI: SB3 job 48:37, coverage 91 percent), and CU-14b (the first part of CU-14: the plant identity cached per process and species, the cleanup plan's decision 10 (c), which the maintainer settled on 2026-09-29 without a D-D id) landed as #570 on 2026-09-29 (measured on its CI: SB3 job 17:31, coverage 91 percent), and the digest-snapshot check of D-D22 (ROW-16, outside D-D21's gate) landed as #571 on 2026-09-29 (measured on its CI: SB3 job 13:09, coverage 91 percent), and CU-7a (the first part of CU-7: the dead code, the unread gym entry points and the package re-exports) landed as #572 on 2026-09-30 (measured on its CI: SB3 job 17:30, coverage 91 percent), and CU-11 (the reward, info and termination golden, the last PR of wave 1) landed as #573 on 2026-09-30 (measured on its CI: SB3 job 17:47, coverage 91 percent), which completes wave 1, and CU-9 (coverage of the certification code, the first PR of wave 2) landed as #574 on 2026-09-30 (measured on its CI: SB3 job 13:47, coverage 88 percent), and CU-5 (the SB3 notebook's stale text and dead parameters, and one budget derivation) landed as #575 the same day (measured on its CI: SB3 job 17:07, coverage 88 percent), and CU-7b (the second part of CU-7: the retired-backend wording in code and the stage-TOML comments, no digest moved) landed as #576 the same day (measured on its CI: SB3 job 15:25, coverage 88 percent), which completes CU-7, and CU-16a (the docs text of CU-16: the gap review's status appendix, its open findings moved to KNOWN_ISSUES, and the living-doc corrections) landed as #577 the same day (measured on its CI: SB3 job 10:42, coverage 88 percent), which completes wave 2, and CU-16b (the second part of CU-16: the orphan assets and the landing page's milestones) landed as #578 the same day (measured on its CI: SB3 job 17:20, coverage 88 percent), which completes CU-16, and CU-8a (the first part of CU-8: one stage-level task-fingerprint derivation, one backend constant and a public ignored-edits check) landed as #579 on 2026-10-01 (measured on its CI, at full depth: SB3 job 13:11, coverage 88 percent), and CU-12 (the species env dedup: shared reward-term, contact, termination-prefix and keyframe helpers for the four dual species) landed as #580 the same day (measured on its CI, at full depth: SB3 job 22:15, coverage 88 percent), which completes CU-12, and CU-14a (the second part of CU-14: six test jobs instead of eighteen, one path list, no duplicate steps, and the robot's six real-training runs left to the nightly schedule, the cleanup plan's decision 10 (a), which the maintainer settled on 2026-09-30 without a D-D id) landed as #581 on 2026-10-01 (measured on its CI: SB3 job 8:09, coverage 88 percent; 10 CI jobs instead of 22), which completes CU-14 and wave 3, and CU-8b (the second part of CU-8: `policy_loading` owns the SB3 import helper and the sidecar resolver) landed as #582 the same day (measured on its CI, at full depth: SB3 job 21:29, coverage 88 percent), and CU-8c (the third and last part of CU-8: one `stage_config.json` reader, one repository root, one sha256 pattern and one set of field validators) landed as #583 the same day (measured on its CI, at full depth, in a run that finished green two minutes after the merge: SB3 job 21:56, coverage 88 percent), which completes CU-8 and D-D21's gate, and the 0.3.9 cut (D-D21) landed as #584 on 2026-10-02 (measured on its CI, at reduced depth, on the tagged tree and on the head: SB3 job 14:26 and 10:37, coverage 88 percent), tagged `0.3.9` by the maintainer and published as a GitHub pre-release; the archive points were settled with no archive tags, and 0.3.9, cut after the retirement and the structural cleanup, is the clean base this sequence builds on, and the cut's records PR landed as #586 the same day (measured on its CI, at reduced depth: SB3 job 11:18, coverage 88 percent); on 2026-10-02 the maintainer chose to finish the deferred cleanup first, which goes in the order CU-10a, CU-13, CU-15, CU-10b, ROW-4/6 (the notebook PR for the cleanup plan's decisions 4 and 6), CU-6 and CU-17 (the cleanup plan's §3.1 item 5), and CU-10a (the curriculum horizon fix, split from CU-10) landed as #587 the same day (measured on its CI, at reduced depth: SB3 job 7:27, coverage 88 percent), and CU-13 (stage-TOML `extends` for recovery ← stance, in the per-table form of D-D5's amendment of 2026-10-02) landed as #588 the same day (measured on its CI, at reduced depth: SB3 job 11:53, coverage 88 percent), and CU-15 (reduced: the Drive summary's `REPO_REF` bootstrap and reader tests) landed as #589 on 2026-10-03 (measured on its CI, at reduced depth: SB3 job 11:23, coverage 88 percent), and CU-10b (one stage body for `train()` and the command-line curriculum, `eval_env_seed`, and the curriculum's D-D11 alignment, with D-D11's amendment) landed as #590 on 2026-10-03 (measured on its CI, at full depth: SB3 job 20:53, coverage 88 percent), which completes CU-10, and ROW-4/6 (the notebook PR for the cleanup plan's decisions 4 and 6: the chain loop judges an unjudged `RUN_DIR` node before the trunk, and `trunk_run.json` records the trunk, which the RESUME cell checks a resume against) landed as #591 on 2026-10-03 (measured on its CI, at full depth: SB3 job 21:43, coverage 89 percent), and CU-6 (the resume slice: the RESUME cell's periodic-pair walk and the archive-load preflight are library functions, and the RESUME cell no longer evaluates the node it trains, with D-D7's amendment) landed as #592 on 2026-10-04 (measured on its CI, at full depth: SB3 job 21:53, coverage 89 percent), and CU-17 (the docs shrink, the last of the deferred PRs: the root README a front page beside the generated `docs/SPECIES_CATALOG.md`, the one-landing-record rule, which makes this table and the CHANGELOG the only landing record, and the landed sections and NEXT_STEPS's landing records turned into pointers) landed as #593 on 2026-10-04 (measured on its CI, at reduced depth: SB3 job 7:31, coverage 89 percent), which completes the deferred cleanup; the sequence resumes with consolidation PR-8. Prerequisites it proposes for this sequence: the reward/termination golden trace before PR-8 (CU-11, landed as #573, 2026-09-30); one task-fingerprint derivation and the species env dedup before PR-9; stage-TOML `extends` for recovery ← stance, the heightfield-contact investigation and re-derived recipe speeds and map sizes before PR-11; atomic evidence writes and final/best checkpoint pairs before PR-13 (both landed with CU-3, #562). PR-B superseded PR-3b's JAX-job item. The digest-snapshot harness its acceptance checks use landed with the plan (`environments/shared/harnesses/digest_snapshot.py`). Under D-D22 (taken 2026-09-29) its output becomes a golden that CI checks, with the full harness run on pull requests; its own PR builds the check, before CU-7b, CU-8a, CU-12 and CU-13. |
| PR-8 (one behavior env, part 2: one terrain selector, one command-constant source, one normalisation; D-D3) | **Landed** as #594 on 2026-10-04 (measured on its CI, at full depth: SB3 job 20:52, coverage 89 percent): `SpeciesBehaviorMixin` takes a `terrain_sampler` and its `reset()` selects each episode's family with `select_terrain_family`; the `terrain_family` reset option and a `terrain_families` property serve every behavior env; `TerrainSamplingMixin`, `get_sampled_behavior_env_class`, `_PanelTerrainMixin` and the panel's dynamic class are deleted, and the sampler gains the `terrain_contact` family (listed last, weight 0 by default), each family setting its own surface on the `[terrain]` map. The 42 single-template recipes state `[terrain_sampler]` blocks of four (`flat = 1`, their own family at 3), every sampler table states all six families, and `env.flat_probability` is retired (refused with the migration in its message; out of the identity and the transition settings): the Bernoulli-to-blocks change of §8, measured from the harness's reset seed 1042 as episodes 1, 6 and 9 of 0-9 changing surface in each of the 42, with the other 414 resets of the 54 terrain recipes bit-identical in surface, command seed, observation and joint positions. Rewriting the 54 recipe TOMLs, rather than mapping `flat_probability` to blocks inside `read_recipe`, is the maintainer's choice of 2026-10-04. Evaluation visits every recipe's families in turn, as the panel did. `direction_commands.py` takes `COMMAND_WIDTH`, `COMMAND_COMPONENTS` and `COMMAND_RANGE` from `command_frame`. D-D3: `BehaviorVecNormalize`, the class swap and the type gate are deleted; behavior sidecars are plain `VecNormalize` files whose command slice is reseeded at preparation and then updates, and a sidecar that pickles the deleted class is refused by name. `lateral_speed_scale` waits for PR-11 (D-D12). The digest golden moves only in its behavior section, 932 to 974 lines, and its reward, stage, plant, policy and recovery lines are byte-identical; `--exact` with the behavior recipes moves 594 of 32,389 lines, every one a terrain recipe's reset record. Measured (`git diff --numstat` against `0520083`, the change commit): library +136 / −230 in 7 files, tests +468 / −130 in 7 files, recipes +390 / −54 in 54 files, the golden +306 / −264, docs +107 / −22 in 2 files, CHANGELOG +74. |
| PR-9 (Phase D through the reserved hook; identity = task fingerprint; D-D2) | **Landed** as #595 on 2026-10-05 (measured on its CI, at full depth: SB3 job 21:40, coverage 89 percent): `BaseDinoEnv` and the five species constructors take `command_mode` and one `command_config: DirectionCommandConfig \| None` in place of the five unread numeric kwargs (D-D2). A live mode builds the `DirectionCommandController` in `__init__` (under `"none"` nothing is built or drawn; a live mode without a config, a config under `"none"` and speed variation under `"heading"`, which holds `cruise_speed`, are refused), the reserved hook seeds it with one draw appended to the reset stream (`_command_rng()`), a new `_update_command()` advances it at the end of `step()`, after the reward, and the observation is re-read; `command_manifest()` is the controller's manifest and `_heading()` the base env's. `validate_command_mode` loses its SB3 branch and `SB3_COMMAND_REFUSAL` (the MJX refusal stays, invariant 9); `COMMAND_ENV_KEYS` names the two keys, so the carve-out pops both under `"none"`, and `_canonical` records a dataclass field by field; the payload's `command` section is passed exactly when the effective mode is live (either half alone is refused), and `stage_task_fingerprint` takes an optional `command_manifest`, forwarded only when given (no caller passes one until a live stage exists, PR-11). Following D-D2 strictly is the maintainer's choice of 2026-10-05: the constructors refuse the five (`TypeError`), and runs recorded before PR-9 are not kept rebuildable from their recorded constructor kwargs (a reader must drop `command_frame.RETIRED_COMMAND_ENV_KEYS`, the five names, first, and no path on main does so today: the gait audit's hand-run probe, `docs/investigations/gait_2026_09/gait_probe.py` run with `--use-recorded-env-kwargs`, is one such reader and now refuses a pre-PR-9 record, with the constructor's `TypeError`; it is frozen evidence and is not edited here), while every committed stage fingerprint, and so the reuse of certified walkers as trunks, is unchanged (all 21 derived identically at base and head). The task fingerprint refuses each of the five by name, in every mode (the PR's own default, so a break is loud), rather than hash a task that matches nothing; no committed file or code path on main hands one to the fingerprint. The recipe env passes `"heading_and_speed"` with its `commands` (`create_behavior_env` refuses `command_mode` and `command_config` by name), keeps its own command stream by overriding `_command_rng()`, and loses its refusal, its three direct writes, its `command_manifest` override and `_heading()`. Its identity is its task fingerprint: stage `command-terrain/v2`, a versioned name in place of the deleted `sources` hashes and `sampler_source_identity`, never a stage id; terrain, sampler, tracking weight and course distance in the `env` section as constructor kwargs, read live as the env reads them; the controller's manifest as the `command` section. Behavior checkpoints carry the plant identity, that fingerprint (preparation and adaptation refuse one without its `task_sha256`, by name, before loading anything, and preparation one whose stage is a stage id) and the preparation report as task lineage (an adaptation moves its `child_task_sha256` to the adapted task), and a checkpoint without a fingerprint or a `mesozoic.behavior-bundle/v1` bundle is refused by name (D-D9); `behavior_identity`, `BEHAVIOR_IDENTITY_SCHEMA`, `PREPARATION_ATTRIBUTE` and `canonical_env_parameters` are deleted (`read_recipe` checks `[env]` keys against the species signature, less the two command keys). The zeroing of the 23 species reward weights and the tracking reward stay in the recipe env until PR-11, and the "Phase D" sentences in `train_base.py` and `policy_loading.py` stay for PR-10 and PR-13. The 20 KNOWN_ISSUES citations of the edited files follow their lines (one, already 8 lines off at the base, now names the lines it describes). No committed `task_sha256` moves, `plant_contract --check` is current, and `--exact` with the behavior recipes is byte-identical on base and head (32,389 lines). The digest golden goes from 974 to 656 lines: the 42 `stage_config_view_sha256` lines move, a measured consequence of D-D2 and D-D22 (`stage_config.json` records `command_config` in place of the five keys, and the golden pins each stage's recorded view), and the behavior section's 66 identity digests and 318 source lines give way to 66 `task_sha256` lines; the reward, plant, policy and recovery lines and the stages' task, gate, hyperparameter and config-file lines are byte-identical. Measured (`git diff --numstat` against `518c877`, the change commit): library +368 / −331 in 15 files, tests +904 / −192 in 12 files, the golden +108 / −426, docs +84 / −40 in 8 files, CHANGELOG +72. |
| PR-10 (one command-column primitive in the canonical warm-start path; D-D3) | **Landed** as #597 on 2026-10-05 (measured on its CI, at full depth: SB3 job 23:57, coverage 89 percent): `policy_loading.neutralize_command_columns(model, *, observation_dim)` zeroes the observation's command columns in every first layer that reads it (PPO's actor and critic; SAC's actor and every Q network of its critic and target critic, whose command columns sit before the action, not at the end of the row) and the same block of every per-element tensor the optimizers of `_OPTIMIZER_MEMBERS` hold for those weights, checks every layer and moment before any tensor changes, and returns the zeroed names: the behavior preparation's `_zero_command_connections`, generalised over the widen tool's optimizer members. `policy_loading.assert_command_blind(model, normalizer, *, seed=3042, atol=1e-6, prepare=None)` is the preparation's probe (64 raw rows from `default_rng(seed)` around the normalizer's statistics; the actions, and PPO's values or SAC's Q-values, on zero commands; then on uniform commands from the same generator and on the widen tool's `COMMAND_PROBE_VECTOR`, each within `atol`; the preparation's two messages). The plan's signature sees one model, so the optional `prepare` (the PR's own default) takes the reference before the preparation runs and keeps the probe the preparation ran, the parent against the prepared policy: a zeroing that also changed a non-command weight is refused, not passed as command-blind. `VERIFICATION_ROLLOUT_SEED`, `ACTION_DELTA_ATOL` and `_OPTIMIZER_MEMBERS` move from the widen tool into `policy_loading` as literals (the module stays standard-library-only at import), and the widen tool binds the same objects; that constant block is the widen tool's only change, in place of the §2 row's "widen_checkpoint.py unchanged" (`test_widen_checkpoint.py` untouched). Under `initialize_next_stage`, `train_base._create_or_load_model` calls both, on the child's normalizer and before the first update, when the child's task fingerprint has a `command` section and the parent's recorded one has none (or none is recorded), and records `zeroed_command_parameters` in the task lineage; the child's mode is read from its fingerprint (exact by construction since PR-9's guard), so the function takes no new parameter, and a live parent keeps its columns. The behavior preparation calls the two functions instead of its copies (`_zero_command_connections` is deleted; `ACTION_EQUIVALENCE_ATOL` stays, now the shared object): its prepared model, optimizer state, normalizer, report and global RNG state, and its answer to every case provoked on it (its own refusals, parents outside SB3's default layouts, and zeroings that do too little or too much; 13 cases, by type and message), are byte-identical at base and head; the one refusal it gains is the probe vector's, which an exact zeroing never triggers. Every committed stage is `"none"`, so nothing trains differently: for PPO and SAC, a fresh model, the same-stage resume and the next-stage warm start (from a fingerprinted or an unfingerprinted parent, or without a child fingerprint) are byte-identical at base and head (every policy and optimizer tensor, both normalizers, the task fingerprint, lineage and plant stamps and the global RNG states, after the load, after training and in the saved archive). No `train()` or `train_curriculum` run builds a live child before PR-11 (no caller passes a `command_manifest`), so the tests load as `_train_stage_body` does, from a briefly trained walker-shaped model (64 steps under `"none"` on the real T. rex env at interface revision 13, with the stance stage's `net_arch`; the plan's "real r13 walker fixture" is a test-time model, and none is committed): the command columns and their moments start at zero, the actions are the same with and without a live command and equal to the parent's own (exactly, over a 20-step real-env rollout in the tests; over a 200-step rollout measured for this record, 0.0, where the unneutralized parent's actions move by 9.4e-4 under PPO and 2.6e-2 under SAC), and training then moves the columns. The command-slice reseed in `_load_vecnorm_into_envs` still follows the child only, so a live parent's edge (PR-11's first) would reseed the statistics that parent trained under while keeping its columns. Whether the reseed should follow the parent, as the zeroing does, is PR-11's to settle with that edge (the D-D3 rule is unchanged here); the docstring says so, and the two "Phase D" sentences now describe what is true. The four KNOWN_ISSUES lines citing `train_base.py` follow their lines. The plan's line anchors, its §2 row's positional `observation_dim` (keyword-only, as its PR-10 section writes it) and its estimate of about 80 test lines (two algorithms, the real-env warm start and the preparation's pins) are superseded. No committed `task_sha256` moves (all 21 derived identically at base and head), the digest golden is current at 656 lines, `plant_contract --check` is current, and `--exact` with the behavior recipes is byte-identical on base and head (32,389 lines). Measured (`git diff --numstat` against `ef21e87`, the change commit): library +298 / −79 in 4 files, tests +857 / −2 in 3 files, docs +17 / −7 in 3 files, CHANGELOG +25. |
| Pause after PR-10 (the maintainer, 2026-10-05) | **Landed** as #598 on 2026-10-05 (measured on its CI, at reduced depth: SB3 job 13:51, coverage 89 percent; these records only, no change commit): the consolidation pauses after PR-10. PR-11 .. PR-13, the rest of PR-12 and PR-15 wait until one or two species walk well (walk, not hop or limp), so that the follow node starts from a real gait and its gate and commanded speeds are set from measured data; the next work is training the walkers, and PR-11 resumes in a later session. The maintainer's proposals of 2026-10-05 for PR-11, to be confirmed when it starts (proposals, not decisions taken): `follow_direction` only in the notebook (`stand` → `walk` → `follow`, which `BEHAVIOR = "follow"` would select), with difficult terrain kept in the code base (the behavior recipes and the command-line runner `environments.shared.train_behaviors` stay until a terrain node exists, which narrows PR-12; it would amend G1 and D-D5, whose PR-11 node set is three new stage files per species with `follow_direction_difficult_terrain` the target deliverable); and, by default, a child of a parent that already trained with commands keeps that parent's command statistics, with an option to reset them (an amendment of D-D3's reseed rule for live parents, the question PR-10's row leaves to PR-11). The open items PR-11 meets, pointed to here and not settled: the gate (the new nodes' `none/v1` always refuses, and the cleanup plan's decision 12, recommended, not taken, would land PR-13's gate registration right after PR-11), the commanded speeds (its decision 13, "Recipe speeds and map sizes", recommended, not taken; [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §2) and the per-episode gait clause the gait plan recommends for that gate (its GQ-18, open; [GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md) PR-G10, and the KNOWN_ISSUES entry "the direction/terrain pilots are a second pipeline", [KNOWN_ISSUES.md](KNOWN_ISSUES.md)). The living statements the pause made false each take a dated sentence that points here: NEXT_STEPS.md's section 4 status and its first steps in a new session, ROADMAP.md's current focus, the docs index's row for this plan and TRAIN_DIRECTION_AND_TERRAIN.md's terrain status. No decision row, code, test, configuration, notebook or golden changes; the CHANGELOG has no entry for the pause (nothing that ships changes). These records carry no number of their own: the next PR records their landing in this row, the one place the one-landing-record rule leaves them. |
| Floor-truth stance gate (outside this sequence; D-D23) | **Landed** as #599 on 2026-10-06 (measured on its CI, at reduced depth: SB3 job 9:04, coverage 89 percent): the measurement library `environments/shared/gait/` (the substep floor-contact recorder, the six-species support-geom registry, `MEASUREMENT_VERSION = "floor-truth/v1"` and the per-episode stance metrics) and the gate kind `stance_quality/v2` (`curriculum/stance_gate_v2.py`) with its report, judge, publication, backfill, recorded-gate, catalog, verdict and baseline arms, the manager's refusal and the command-line curriculum's post-training judge. No stage declares the kind, so no digest moves (the golden is current at 656 lines). The in-training screen is not built. The evidence and the 40-episode validation, every statue clean on 40/40 and every audited stance checkpoint failing, are [investigations/STANCE_HACK_AUDIT_2026_10.md](investigations/STANCE_HACK_AUDIT_2026_10.md); the gait plan records the order change in its §11. The consolidation stays paused (the row above). |
| T. rex physics r8 and the stance on `stance_quality/v2` (outside this sequence; D-D24) | **Landed** as #600 on 2026-10-06 (measured on its CI, at reduced depth: SB3 job 13:53, coverage 89 percent): `trex.xml`'s hip-roll servos kp 150 → 600 and forcerange ±480 N·m (physics r7 → r8, note 13), three stance reward kwargs (the settled neck target, pad flatness, stance width) and the trex stance on `stance_quality/v2`, with every trex statue constant re-measured and the trex reset-golden captures re-taken. 39 digest-golden lines move, all trex. Every trex checkpoint is refused by the plant contract; the validation, the r8 statue clean on 40/40 and the four audited r7 stances on 0/40 each, is §7 of [investigations/STANCE_HACK_AUDIT_2026_10.md](investigations/STANCE_HACK_AUDIT_2026_10.md). The consolidation stays paused. |
| Velociraptor physics r3, SB3-only, and the stance on `stance_quality/v2` (outside this sequence; D-D25) | **Carried out** (2026-10-06; no PR number yet, and the next PR records its landing in this row): `raptor.xml`'s leg springs anchored at the standing pose, a flat-footed keyframe whose home ctrl carries the gravity preload, and metatarsus and digit-IV touch sensors summed per foot (physics r2 → r3, policy interface r10 → r11, visual r3 → r4, note 14); the velociraptor declared SB3-only, its `mjx_config.py` deleted; eight stance reward kwargs (five terms) and the velociraptor stance on `stance_quality/v2`, with its statue constants re-measured. The golden goes from 656 to 649 lines and every moved line is a velociraptor line. Every velociraptor checkpoint is refused by the plant contract; the validation, the r3 statue clean on 40/40 and the two audited checkpoints of the `20260922_125248` stance on 0/40 each, is §8 of [investigations/STANCE_HACK_AUDIT_2026_10.md](investigations/STANCE_HACK_AUDIT_2026_10.md). The consolidation stays paused. |
| Net removal from here | about 4,600 lines by the §3 per-PR estimates for PR-8 .. PR-13 and PR-15 (PR-12's −5,300 less the about 1,130 its notebook-only slice removed, measured); the landed PRs' measured sizes are in their rows above. The assessment counted about 10,500 from 723f58f; PR-1 was net zero and #543 added about 1,200 lines including tests. |
| Training | Not on hold. The walker sessions in NEXT_STEPS.md run on the current notebook in parallel with the sequence (G3). Sessions 1–3 ran 2026-09-20 .. 2026-09-23 (the trex seed-44 widen + recovery `20260920_010912`, the compsognathus widen + walker `20260921_203149`, the velociraptor fresh chain `20260922_125248`; every node certified, every bundle `complete`), which meets D-D14's condition. Session 4 (dibothrosuchus, `20260923_020654`, 2026-09-23) was cut short by the collapse backstop at 1.45M steps on both nodes; its re-run after the backstop fix below was started on 2026-09-28 and is recorded when it finishes; session 5 remains; session 6 (compsognathus_robot, `20260924_031815`) passed its stance gate on 2026-09-24 and, resumed on 2026-09-28 after the Colab cap (2.8M of 3M), its locomotion gate the same day (0.227 m/s, every evaluation episode the full 1,000 steps; bundle `complete`); session 7 (the trex seed-44 walker, `20260925_033501`) passed its locomotion gate on 2026-09-25. The maintainer paused the PR sequence after PR-6 on 2026-09-20 and lifted the pause on 2026-09-23: the notebook-only PR-12 slice landed as #552 and PR-14a as #553 and PR-14b as #554 and PR-14c as #555, all on 2026-09-24 (PR-14 complete), and PR-7 as #556 on 2026-09-25; PR-8 follows the deferred cleanup (the maintainer's choice of 2026-10-02). The gait audit of 2026-09-28 ([investigations/GAIT_AUDIT_2026_09.md](investigations/GAIT_AUDIT_2026_09.md)) found that three of the five certified walkers hop (trex seed 42 `20260914_123816` and seed 44 `20260925_033501`, and compsognathus_robot `20260924_031815`; only compsognathus `20260921_203149` walks and velociraptor `20260922_125248` runs), and the gait plan's GQ-3 ([GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md); open, like all its decisions) recommends holding new walker sessions meanwhile: session 5 until brachiosaurus's `gait-r1` revision, and no walker session on today's rewards (it would amend G3). |
| Collapse-backstop fix (outside this sequence) | **Landed** as #551 on 2026-09-23 (measured on its CI: SB3 job 35:37, JAX job 53:34, coverage 90 percent): `collapse_peak_warmup_timesteps` on dibothrosuchus and brachiosaurus stages 1–2 (1.0M on stance; on locomotion the D-B5 bound `warmup_timesteps + ramp_timesteps`, 3.3M and 4.0M, conservative since the clip/entropy warm-up and the forward ramp run concurrently), replayed on session 4's evaluation series in `test_curriculum_early_stopping.py`; no task, gate or hyperparameter digest moves. It touches four stage TOMLs and one test file, none of which this sequence edits, and the docs pass of the same PR corrected the living docs (KNOWN_ISSUES gained three entries). |
| Loader change (2026-09-19, outside this sequence) | The Colab image moved to Python 3.13 and both first attempts at NEXT_STEPS.md session 1 died inside the widen tool's self-verification (KNOWN_ISSUES, "SB3 archives are bound to the interpreter that saved them"). `policy_loading.load_sb3_model` is now the one archive loader, `linear_schedule` / `cosine_schedule` are picklable classes, and the notebook's load preflight is a cell right before the widen cell. Consequences for this plan: PR-14 item (d) has one disconnect-before-raise site left (cell 22's gate refusal; PR-4 then removed the publish block's), not three, and the notebook target of §4 gains one ~75-line code cell (preflight) right after the resolve cell, where the widen row PR-14a deletes used to follow it (it reads `TRUNK_DIR`, which the storage row binds for a pinned trunk and the resolve cell under `"auto"`); the disconnect-site numbers are in the plan's `22c1fc8` cell numbering, and PR-14b turns that one site (cell 18 after it) into a `halt` call (§3). |

Decisions: the maintainer took D-D1..D-D10 and G1..G4 on 2026-09-17 (§6, §7)
and confirmed D-D11 and D-D12 on 2026-09-20, when D-D13 (the notebook-first
order) and D-D14 (the widen path becomes CLI-only after sessions 1 and 2) were
taken; D-D15 (PR-14 lands as PR-14a, PR-14b and PR-14c) was taken on
2026-09-24 and D-D16 (the notebook-safety PR, #558) on 2026-09-25; D-D17 (the
backend retirement), D-D18 (mypy with SB3 in CI), D-D19 (the CHANGELOG
release cut) and D-D20 (atomic final and handoff checkpoint pairs) were taken
on 2026-09-26 for the cleanup plan, and D-D21 (0.3.9 as the clean base release)
on 2026-09-27. D-D22 (the digest snapshot as a CI check) was taken on
2026-09-29 and carried out by its own PR, ROW-16, which landed as #571 the same day. D-D23 (the floor-truth stance
gate) was taken on 2026-10-06 for the gait plan, outside this sequence, and D-D24 (the T. rex physics revision r8
and the T. rex stance on that gate) and D-D25 (the velociraptor physics revision r3, its SB3-only exit and the
velociraptor stance on that gate) the same day, also outside it. Ids follow the series recorded
in BEHAVIOR_RECIPES_PLAN.md §6.2; the review's working labels D1..D12 map
one-to-one onto D-D1..D-D12.

The hold lifted on 2026-09-20 with the order of D-D13: PR-3 (no prerequisites,
bounds the CI job that every later PR's validation runs through), then PR-4
and PR-5 in that order (the canonical wrapper imports the library, not the
reverse), PR-6, the notebook-only slice of PR-12 (pulled ahead of PR-11 so the
notebook shrinks first), PR-14 (as PR-14a, PR-14b, PR-14c; D-D15), then
PR-7 .. PR-11, the rest of PR-12, PR-13 and PR-15. Every later PR names its prerequisites and the decisions it rests
on; none of the decisions it needs is still open.

## 1. Headline

Two bodies of work landed in four days. The recipes program (#528–#539, ~31k
lines) built the canonical machinery: stage manifest v2, per-deliverable result
bundles with `gate_verdict.json` and `ancestors/`, the notebook chain loop with
`TRUNK_FROM` / `WIDEN_FROM` / `RETRAIN_FROM`, gate kinds behind one closed
dispatch, the Phase C interface bump that appended three zero command dims to
every observation, and `widen_checkpoint.py`; it also reserved, on purpose, the
exact hook and kwargs that Phase D was to fill
(`BaseDinoEnv._draw_episode_command`,
`command_mode`, `command_manifest()`, the `command_tracking/v1` gate, two
manifest nodes warm-started from the certified walker;
[BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §4.6). The
direction/terrain
PRs (#540–#541, 117 files, 18,400 lines) delivered the direction sampler,
tracking reward and a real heightfield generator, which are genuinely new and
worth keeping, but delivered them as a second pipeline beside every canonical
concept: two behavior env classes that refuse the reserved `command_mode` and
write `self._command` directly after `super().reset()`
(environments/shared/behavior_env.py:86-87, 397), a second checkpoint preparer
that overwrites the plant identity attribute with its own schema
(environments/shared/behavior_checkpoint.py:259-261), a second PPO trainer with
its own recipe dialect (environments/shared/train_behaviors.py:45-150, 540), 66
TOMLs that are an 11-behavior by 6-species outer product, a second gate outside
`GATE_KINDS` (environments/shared/behavior_certification.py:227-338), a third
identity keyed on source-file hashes, and a "certified library" that re-verifies
chains `find_certified_ancestor` already verifies, recounts replication
`discover_replicates` already counts, copies every parent checkpoint into every
child run, and, under the default `PUBLISH_CERTIFIED = True` that #542 has since
flipped, would have disconnected the Colab runtime the moment a widened stance
passed its gate (no widen session had run by then; #542 closed the path first; cell
22:298-321; environments/shared/certified_canonical.py:103-107). The
notebook grew to 40 cells and 2,500 lines with a `COMMAND_TERRAIN_BEHAVIOR`
switch guarding 12 whole cells and two partial ones.

The end state is the plan of record, filled in with the new content:
direction-following and terrain traversal are ordinary manifest nodes under
`locomotion`, the direction controller is the body of the reserved hook in
`BaseDinoEnv`, terrain is one generic opt-in env subclass (it cannot enter
`reset()`, whose source is fingerprinted), command-column zeroing is one
primitive in `policy_loading` used by `initialize_next_stage`, nodes are stage
TOMLs judged by a registered gate kind and reused through
`find_certified_ancestor`, the certified library is deleted, and the notebook
loses the mode switch and every library knob while keeping every knob the
maintainer already uses. Fifteen PRs, ordered so that the live workflow is
protected first and each PR stays reviewable; net about 10,500 lines removed
from 723f58f (roughly 60% of what #540–#541 added) with the uncertainty
concentrated in the stage-TOML form (D-D5) and the gate work (D-D6).

Two things the assessment did not know. The 2026-09-17 Drive survey
([investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md),
summarised in NEXT_STEPS.md) established that the certified r13 trex walker the
plan's Phase C½ asked for already exists (run 20260914_123816, stance and
locomotion both `gate_verdict.json` PASS with task and gate digests equal to
those derived from the current stage TOMLs, so `TRUNK_FROM = "auto"` reuses
both nodes), obtained from a from-scratch r13 stance rather than a widened r11
one. And nothing relies on a `mesozoic-labs/certified` library directory on
Drive since #542/#543 (the survey did not inspect it). Both narrow the
consequences noted under PR-4, PR-11 and D-D1 below.

## 2. Target architecture

| Concept | Today: recipes implementation | Today: #540/#541 implementation | After |
|---|---|---|---|
| Env command source | Reserved hook `BaseDinoEnv._draw_episode_command` returns zeros, `command_manifest()` returns None (environments/shared/base_env.py:1063-1083), called at base_env.py:1514 after every reset draw; six `command_*` kwargs stored but never read (base_env.py:195-200, 301-306); `command_frame.validate_command_mode` fails closed on SB3 (command_frame.py:77-94) | `DirectionCommandController` (environments/shared/direction_commands.py, 442 lines) driven from `SpeciesBehaviorMixin.reset/step/set_direction`, which refuse `command_mode != "none"` (behavior_env.py:86-87) and assign `self._command` directly (397, 417, 443); `command_manifest()` overridden (308) but never fed to `compute_task_fingerprint` | `BaseDinoEnv` owns the controller under `command_mode in {"heading","heading_and_speed"}` with one `command_config` kwarg replacing the five dead numeric kwargs (D-D2); hook body = `controller.reset(...)`, new `_update_command()` at the end of `step`, `command_manifest()` = `controller.manifest()` into the task fingerprint. Deleted: the refusals, the three direct writes, the SB3 branch of `validate_command_mode`, the five numeric kwargs in five species constructors and `MJXEnvConfig`. direction_commands.py survives unchanged in substance (imports `COMMAND_WIDTH`/`COMMAND_COMPONENTS`/`COMMAND_RANGE` from command_frame). |
| Terrain | None (base_env.py:1332-1344 documents plane-only settling) | terrain.py (522) + terrain_sampling.py (183) + two model/data pools swapped by `_select_contact_model` (behavior_env.py:177-193); four stacked layers decide an episode's ground: `flat_probability` draw (373-378), `TerrainSamplingMixin.reset` try/finally mutation (terrain_sampling.py:148-163), `_sampled_behavior_env_class` (171-183), `_PanelTerrainMixin` in behavior_certification.py:63-107 | terrain.py kept as the single injection point (`build_terrain_model`, plane named `floor`). One generic opt-in subclass (the surviving `SpeciesBehaviorMixin` minus its command half, built by `make_env` when `[env]` carries terrain keys) holds the two pools, the terrain settle, and ONE per-episode family selector (`select_terrain_family` with a `terrain_contact` family and a `terrain_family` reset option) (D-D10). Deleted: environments/trex/envs/behavior_env.py (497), `TerrainSamplingMixin`, `_sampled_behavior_env_class`, `get_sampled_behavior_env_class`, `_PanelTerrainMixin`, `flat_probability`. `BaseDinoEnv` gains `_ground_height_at(xy)` so species rewards/terminations are terrain-relative without re-derivation. |
| Checkpoint preparation | widen_checkpoint.py (1308): inserts zero columns into an r11/r12 archive, pads Adam moments, `pad_running_stats`, restamps, self-verifies with seed 3042 / atol 1e-6; `load_vecnorm_stats(reseed_command_slice=...)`; `_create_or_load_model` under `resume_same_stage` / `initialize_next_stage` (train_base.py:552-637), which does NOT zero existing command columns | behavior_checkpoint.py (457): `_zero_command_connections` on the same two layers and moments (145-162), same probe (208-231), `BehaviorVecNormalize` passthrough installed by `__class__` swap (58-75, 214) while still calling `reseed_command_slice` (216), `load_behavior_checkpoint` = resume, `adapt_behavior_checkpoint` + three hand-listed frozensets = initialize_next_stage (315-457); marker written into both `MODEL_IDENTITY_ATTRIBUTE` and `MODEL_TASK_ATTRIBUTE` (259-261) | `policy_loading.neutralize_command_columns(model, observation_dim)` (~35 lines) + `assert_command_blind(...)` (~25) sharing widen's probe constants; called by `_create_or_load_model` on `initialize_next_stage` when the parent fingerprint has no `command` section and the child's mode is live; resume = `resume_same_stage`, adapt = `initialize_next_stage` with lineage; command-slice normalisation follows the reseed rule (D-D3). Deleted: behavior_checkpoint.py and its 420-line test. widen_checkpoint.py unchanged. |
| Training entry point | `train_base.train` (980), `train_curriculum` (1999), `make_env` builds `species_cfg.env_class(**env_kwargs)` (204-228); notebook cell 16 `train_stage` (410 lines) re-runs train()'s body | `train_behaviors.main` (601 lines): own `read_recipe`, own env factory, own `model.learn` (540), own run.json/bundle.json, one CPU env; `behavior_notebook.NotebookBehaviorPlan.argv` (94-128) serialises 15 knobs into argv and calls `main()` in-process; trex shim (16) | `train_base.train` for everything; the notebook `train_stage` becomes a wrapper over `train()` (D-D7; as built by PR-14c, about 100 lines: its four argument refusals, the node banner, one `train()` call and the evaluation; `train()` gains the seed line, duration recording, `parent_run_id`, an explicit `vecnorm_path` and the `report_metrics` / `save_on_interrupt` switches and still returns the model; no eval-seed parameter, since the notebook's checkpoint-selection seed is `train()`'s `seed + 1000`). Deleted: train_behaviors.py, behavior_notebook.py, the shim, cells 7/20/34/39. |
| Recipe / stage configuration | Stage TOMLs via `load_stage_config` (config.py:184), `stages.toml` v2 edges, `StageEntry.warm_start_from/deliverable/recipe` (stage_manifest.py:108-128) | 66 `configs/<species>/behaviors/*.toml` (3,030 lines) whose only cross-species differences are 20 keys with one value per species, and whose within-species differences are 7 keys; 8 `configs/trex/behavior_pilots/*.toml` (239) with a `[pilot]` dialect branch in `read_recipe` (train_behaviors.py:51-66, 76, 112) referenced only by pyproject.toml:114, environments/trex/tests/test_behavior_training.py:17 and one doc line (deleted by PR-6 on 2026-09-20) | Per species three new stage TOMLs (`follow_direction`, `follow_direction_difficult_terrain`, `difficult_terrain`; the assessment listed a fourth, `follow_direction_speed`, which G1 folds into `follow_direction`) with `[env] command_mode`, `command_config`, `terrain_*` scalars, `[ppo]`, `[curriculum]`, plus `[[stages]]` entries after `behavior`; the nine single-template variants become documented `[env]` overrides for the manual cell. `load_stage_config` gains a ~20-line `extends` key (D-D5). Deleted: all 74 TOMLs, the `[pilot]` branch, the pyproject globs. |
| Gate / certificate | Closed `GATE_KINDS` + `_REQUIRED_THRESHOLD_KEYS` (curriculum/gate_schema.py:53-191), `evaluate_stage_gate` (reporting/gates.py:688-778), `_apply_stage_gate` -> `gate_verdict.json` (stage_artifacts.py:1022-1134), `binomial_lcb`/`paired_difference_lcb` | `judge_behavior_panel` (behavior_certification.py:227-338) reading configs/behavior_certification.toml, its own `gate_sha256` from source-file hashes and package versions (35-61), output `certification/certificate.json`, never `gate_verdict.json`; imports only `binomial_lcb` | One new gate kind, `terrain_command/v1` (D-D6), registered in `GATE_KINDS` with an evidence writer in the `write_recovery_evidence` pattern, judged by `evaluate_stage_gate`, written as `gate_verdict.json` so reuse rules 1-7 apply, first thresholds from G4; first pilots run `none/v1` recorded-not-enforced (gate_schema.py:128). Deleted: behavior_certification.py, the TOML, the certificate schema. |
| Publication and reuse | `gate_verdict.json` hash-bound to the handoff pair, `provenance.deliverables[*].replication/provisional`, `ancestors/` records that never copy checkpoints (ancestors.py:678-686, plan A10), `find_certified_ancestor` rules 1-7, `discover_replicates` | certified_library.py (671), certified_canonical.py (917), certified_comparison.py (153): immutable store re-hashed on every read, second replication counter (258-289), second paired statistic (291-346), flock, complete copies into `certified_inputs/`, `copy_canonical_ancestor` reversing A10, `stamp_canonical_training` written only by the notebook | The recipes publication as it stands, with automatic parent selection provided by `ancestors.select_trunk` (D-A25, landed as #543). Deleted: all three modules, four test files, test_behavior_publication.py, docs/CERTIFIED_MODELS.md, the four notebook knobs and three notebook blocks, `--auto-source/--publish-certified/--certified-library`. No recommendation pointer survives (D-D4). |
| Evaluation | `eval_policy_quality` on `LocomotionMetrics` (evaluation.py:111), `load_sb3_checkpoint` (policy_loading.py:96) | `evaluate_behavior` (behavior_evaluation.py:401-640) importing nothing from evaluation/metrics, `evaluate_saved_panel` wrapper, `benchmark_canonical_stage` fourth rollout loop, three `_sha` helpers and two `_json_value` coercers | `summarize_episode`, `terrain_family_from_reset`, `by_terrain_family` kept as the behavior-specific statistic feeding the gate's evidence writer; one `sha256_file` and one `json_ready`. Deleted: `benchmark_canonical_stage`, `evaluate_saved_panel`, the helper copies, `_PanelTerrainMixin`. No shared rollout generator (refuted as a non-simplification). |
| Replay / video | `record_stage_video` (evaluation.py:221-370): fresh render env at `replay_seed`, mediapy, camera presets, stance CSV the r11/r13 reviews read | `BehaviorReplayRecorder` gym.Wrapper (behavior_replay.py:305-500): same scored trajectory, heightfield snapshot, terrain maps, decoded-MP4 check | Both kept (keep verdict). Only the `_sha`/`_json_value` helpers unify; mediapy stays in the Colab install line (evaluation.py:257, 354 record the canonical replays). |
| Notebook routing | Chain loop (cell 22, 339 lines) + `train_stage` (cell 16:201-610) + widen (11) + resume (26) | `COMMAND_TERRAIN_BEHAVIOR` membership switch (cell 6:64-84) guarding cells 8,10,11,12,14,24,26,28,30,32,36,38 and partials 16:928 / 22:38-47; second storage cell (7), run/display/disconnect cells (20/34/39); 10 `BEHAVIOR_*` + 4 `CERTIFIED_*`/`SOURCE_SELECTION` knobs; second output tree without provenance | One notebook, no switch, no second tree: follow/terrain nodes are values of `BEHAVIOR` walked by the existing chain loop; `train_stage` wraps `train_base.train`; no widen cell (D-D14: a root widened on the command line has its seed checked before the provenance is minted and is judged before any trunk may stand in for it); `RUN_ID` is a knob; a complete run takes no new node. Knobs kept: SPECIES, ALGORITHM, N_ENVS, SEED, VERBOSE, QUICK_TEST, USE_GOOGLE_DRIVE, AUTO_DISCONNECT, BEHAVIOR, TRUNK_FROM, RETRAIN_FROM, RUN_LABEL, RUN_ID, REPO_REF. |
| Vocabulary | "behavior" = recipe label and the hunt node id `behavior` (RESERVED_STAGE_IDS, stage_manifest.py:79); "certified" = gate passed with every ancestor passed, hash-bound (`CertifiedAncestor`); "identity" = `PlantIdentity` + task fingerprint | "behavior" = 28 public symbols, 4 checkpoint stamps, 14 knobs, two config trees; "certified" = a recommendation store; a third `behavior_identity` schema (`mesozoic.command-terrain/v1`) hashing five source files; checkpoint stamps 4 -> 12 | "behavior" = recipe label/node only; the surviving env feature is named "command" and "terrain"; "certified" keeps the plan §2 meaning; identity = plant identity + task fingerprint (`command` section via `command_manifest`, terrain via `[env]` kwargs); stamps back to the four recipes attributes. |

## 3. Ranked PR sequence

Sizes: S < 200 changed lines, M < 800, L < 2000, XL above. "Validation" names
the checks that exist today: ruff, mypy on environments/, the shared suite
(`pytest environments/shared/tests/`), the species suites, the notebook
round-trip/parse step (python-ci.yml:81-111) and the SB3 job. "Folds in" names,
in plain words, which of the review's proposals each PR absorbs, so that a
reader of the review's working notes can find where each idea went.

The sections of PR-1 .. PR-7, the notebook-only slice of PR-12 and PR-14a .. PR-14c (landed by `0.3.9`) keep
their headings and point to their full text (goal, folds, breaks, validation, the as-executed
record and every amendment made before the tag) at commit `20ab100`, the `0.3.9` release:
`git show 20ab100:docs/CONSOLIDATION_PLAN_2026_09.md`. Cite the commit, not the tag name: the tag first
pointed at #584's merge commit (the cleanup plan's §2 row 20). The two notes PR-14a and PR-14c gained after the
tag are summarized in their status rows; their text is at commit `ace8112`. Each one's landing record is its
row in the status table above or, for PR-2, its heading (cleanup CU-17, 2026-10-04).

### PR-1. Stop the live disconnect (S, 4 files, +10/-7, no decision needed) — LANDED as #542, 2026-09-16
Landed (status table above); full text at `20ab100` (`0.3.9`), as §3's note says.

### PR-2. Record the decisions in the design of record (S, docs only; net about +100 lines) — LANDED as #544, 2026-09-19
Landed (status table above); full text at `20ab100` (`0.3.9`), as §3's note says.

### PR-3. Bound the SB3 CI job (S, about +10 to +40 workflow lines; about 40 minutes off every PR run) — LANDED as #546, 2026-09-20
Landed (status table above); full text at `20ab100` (`0.3.9`), as §3's note says. Its follow-up list
(PR-3b) is superseded by the cleanup plan (D-D17, D-D18; its §2 row 10).

### PR-4. Delete certified_canonical.py, certified_comparison.py and the notebook library hooks (L by count, mostly file deletion; about -2,400) — LANDED as #547, 2026-09-20
Landed (status table above); full text at `20ab100` (`0.3.9`), as §3's note says.

### PR-5. Delete certified_library.py and its consumers in the behavior trainer (M/L, about -1,400) — LANDED as #548, 2026-09-20
Landed (status table above); full text at `20ab100` (`0.3.9`), as §3's note says.

### PR-6. Delete the T. rex pilots, the `[pilot]` dialect and the shim (S, about -275) — LANDED as #549, 2026-09-20
Landed (status table above); full text at `20ab100` (`0.3.9`), as §3's note says.

### PR-7. One behavior env, part 1: ground-height hook and delete TRexBehaviorEnv (M, about -600; measured −427 net code, test and CI lines) — LANDED as #556, 2026-09-25
Landed (status table above); full text at `20ab100` (`0.3.9`), as §3's note says.

### PR-8. One behavior env, part 2: one terrain selector, one command-constant source, one normalisation (M, about -170)
Goal: (a) fold `TerrainSamplingMixin` (terrain_sampling.py:89-168),
`_sampled_behavior_env_class`, `get_sampled_behavior_env_class`, the
factory-by-key logic (train_behaviors.py:161, behavior_certification.py:105) and
`_PanelTerrainMixin` + the `type("BehaviorCertificationPanelEnv", ...)`
(behavior_certification.py:63-107) into one `terrain_sampler` kwarg on the mixin
whose `reset()` calls `select_terrain_family` directly, keeps the
`options={"terrain_family": ...}` override, and gains a `terrain_contact`
(flat-heightfield, `mode = "flat"`) family the sampler cannot express today
(terrain_sampling.py:108-109); map `[terrain]`-without-sampler recipes to block
weights (e.g. `flat=1, <template>=3` for `flat_probability = 0.25`; Bernoulli ->
balanced blocks is a distribution change, stated in §8). Remove
`flat_probability` from `_TRANSITION_TOP_SETTINGS` (behavior_checkpoint.py:350-351).
(b) direction_commands.py imports
`COMMAND_WIDTH`/`COMMAND_COMPONENTS`/`COMMAND_RANGE` from command_frame.py
instead of re-stating them (137, 202, 334). (c) D-D3 (taken): delete
`BehaviorVecNormalize`, the `__class__` swap and the `isinstance` gate
(behavior_checkpoint.py:58-75, 214, 299) so behavior sidecars are plain
`VecNormalize` files and the plan's invariant 8 (reseed) is the one rule; the
passthrough alternative (keep the subclass, delete the redundant
`reseed_command_slice` call at :216, document the pickled-class dependency) is
closed. (d) drop the dead `lateral_speed_scale` (direction_commands.py:321
always emits `v_y = 0`) when the TOMLs are rewritten in PR-11 (D-D12,
recommended).
Folds in: the terrain-selector finding (corrected), the reduced form of the
reward-helpers finding (the move into reward_functions.py is dropped), and the
single-probe-env / terrain-family-property finding.
Breaks: test_terrain_sampling.py mixin cases rewritten against the env;
test_behavior_checkpoint.py passthrough cases; identity of single-template
recipes (already stranded). Validation: shared suite, test_behavior_env
parametrised suite. Prerequisites: PR-7; D-D3.
*Cleanup CU-11 (landed as #573, 2026-09-30): validation adds the digest snapshot's `reward` lines, which must not
move (the 21 stage envs never import PR-8's modules), and the harness's `--exact` base/head diff with the behavior
recipes, where PR-8's numbers are (the cleanup plan's §3.4).*

### PR-9. Phase D through the reserved hook; identity = task fingerprint (M, about -150 net here, more in PR-12)
Goal: `BaseDinoEnv.__init__` takes `command_mode` and ONE `command_config:
DirectionCommandConfig | None` (D-D2), replacing `command_speed_range`,
`command_lateral_range`, `command_yaw_rate_max`, `command_switch_interval`,
`command_switch_jitter` in base_env.py:195-200/301-306, `MJXEnvConfig`
(mjx_env.py:274-279) and the five species constructors
(trex_env.py:195-199/323-327, raptor_env.py:113-117/186-190,
brachio_env.py:109-113/196-200, dibothrosuchus_env.py:131-135/213-217,
compsognathus_env.py:71-75/154-158); the controller is constructed and seeded
only when `command_mode != "none"`; `_draw_episode_command()` returns
`controller.reset(rng, self._heading()).normalized` (the hook already runs after
the settle and the push draw, base_env.py:1514); a new `_update_command()` at
the end of `BaseDinoEnv.step` returns `controller.update(t,
heading).normalized`; `command_manifest()` returns `controller.manifest()` and
`derive_stage_task_fingerprint(command_manifest=...)` carries it; extend the
carve-out at task_fingerprint.py:161-170 to pop `command_config` while the
effective mode is `none` (otherwise every canonical `task_sha256` moves,
breaking test_phase_c_interface.py:348 and the r13 run's identity) and record it
via `asdict` (a ~6-line dataclass branch in `_canonical`, :100-113). Delete the
SB3 branch of `validate_command_mode` (command_frame.py:89-92; MJX keeps failing
closed at mjx_env.py:148-154), the two refusals, the three direct writes, the
mixin's `command_manifest` override, `behavior_identity`
(behavior_env.py:194-221), `BEHAVIOR_IDENTITY_SCHEMA`, `PREPARATION_ATTRIBUTE`,
the source-hash `sources` block, `sampler_source_identity`
(terrain_sampling.py:83-86) and `_evaluation_contract`
(behavior_certification.py:40-48; deleted by PR-5 on 2026-09-20) in favour of versioned strings like
`SCHEDULE_IMPLEMENTATION` (task_fingerprint.py:71). Terrain kwargs
(`TerrainConfig`, `TerrainSamplerConfig`) enter the fingerprint's `env` section
as constructor kwargs of the opt-in subclass. The mixin's zeroing of 23 species
reward weights (behavior_env.py:101-137; 20 until PR-7 folded in trex's three
bite weights) becomes explicit `[env]` keys in the PR-11 TOMLs. `_heading()`
(behavior_env.py:223-225) moves to `BaseDinoEnv`.
Folds in: the reserved-hook finding (corrected), the third-identity finding, and
parts (a)-(c) of the cross-cutting identity proposal (part (d), terrain into
`reset()`, is rejected: `home_reset` fingerprints
`inspect.getsource(env.reset)`, policy_layer.py:399 and digests.py:154-166, and
the model swap must precede `mj_resetDataKeyframe` at base_env.py:1452, so it
stays in the subclass; D-D10).
Breaks: `test_sb3_refuses_command_mode_other_than_none_in_phase_c`
(test_phase_c_interface.py:321) and `SB3_COMMAND_REFUSAL` cases in
test_command_frame.py are replaced; test_mjx_phase_c_interface.py loses five
fields; D-D2 amends D-C3 ("(mode, config) replaces six kwargs"). Plant identity,
golden reset stream, the r11/r13 artifacts and every committed stage fingerprint
are unchanged because `command_mode` stays `none` and nothing fingerprinted is
edited. Validation: `plant_contract --check`, golden fixture, shared suite, MJX
suite, species suites, on every species (§8). Prerequisites: PR-7, PR-8; D-D1,
D-D2.
Amended by D-D17 (cleanup PR-B, 2026-09-28): the `MJXEnvConfig` dataclass,
`canonicalize_env_kwargs`, test_mjx_phase_c_interface.py and the MJX suite are
deleted (a type-checking-only `MJXEnvConfig` alias keeps the frozen signature
type-checkable), so PR-9 edits no MJX config, loses no MJX test fields and runs no
MJX suite; `command_frame.py` keeps its MJX refusal until PR-9 rewrites the
file. PR-9 still leaves the frozen MJX interface core untouched and passes
`plant_contract --check` (the cleanup plan's §3.4).
*Cleanup CU-11 (landed as #573, 2026-09-30): validation adds the digest snapshot's `reward` lines, which must not
move (the `none` command path), and the harness's `--exact` base/head diff with the behavior recipes.*
*Cleanup CU-8a (landed as #579, 2026-10-01): the one derivation is `task_fingerprint.stage_task_fingerprint(species,
stage, *, stage_config=None, env_kwargs=None, plant_identity=None)`, which every site calls (training, the
curriculum walk, `select_trunk`, the recovery freeze, calibration and restamp, `widen_checkpoint` and the notebook's
chain loop); the `command_manifest` this PR adds goes through it, forwarded to `derive_stage_task_fingerprint` only
when given, since a test pins the five keywords it passes today.*
*Cleanup CU-12 (landed as #580, 2026-10-01): the four dual species share `BaseDinoEnv`'s reward-term
(`_progress_terms`, `_nosedive_term`, `_heading_terms`, `_speed_terms`), contact (`_contact_geom`),
termination-prefix (`_root_termination`) and keyframe (`_cache_home_keyframe`) helpers, and the foot forces use one
summation order; this PR's constructors edit the same files and start from them.*
*Cleanup CU-8c (landed as #583 on 2026-10-01): one `stage_config.json` reader (`config.read_recorded_stage_config`), one
`REPOSITORY_ROOT` (`environments/shared/paths.py`, which `plant_contract.constants` binds and stays the patch point
for), and one sha256 pattern and one set of field validators (`environments/shared/record_fields.py`); the
repository-relative paths this PR's identity hashes keep their values.*

### PR-10. One command-column primitive in the canonical warm-start path (S/M, about +180 now; enables PR-12's -720)
Goal: `policy_loading.neutralize_command_columns(model, *, observation_dim)`
(body of `_zero_command_connections`, behavior_checkpoint.py:145-162,
generalised over widen's `_OPTIMIZER_MEMBERS`) and
`policy_loading.assert_command_blind(model, normalizer, *, seed=3042,
atol=1e-6)` (the probe at 202-230) sharing `VERIFICATION_ROLLOUT_SEED`,
`ACTION_DELTA_ATOL`, `COMMAND_PROBE_VECTOR` with widen's `_verify`.
`_create_or_load_model` (train_base.py:552-637) calls it under
`initialize_next_stage` when the loaded checkpoint's recorded fingerprint has no
`command` section and the child's `command_mode` is live, recording
`zeroed_command_parameters` in the task lineage. This is needed regardless of
consolidation: the from-scratch r13 stance run 20260914_123816 was initialised
with random command columns that never received a gradient, and the plan
requires a zero action delta on non-zero commands at the follow warm-start
(plan:780). Widen's insert-then-verify stays separate (identity gate refuses gap
< 1 by D-C17; it rewrites the serialized archive).
Folds in: the primitive half of the checkpoint-preparer finding (the widen-mode
variant is not adopted: it needs its own identity gate, and the primitive is the
smaller change).
Breaks: nothing today (every current stage TOML has `command_mode = none`).
Validation: ~80 lines of tests in test_policy_loading/test_train_base (zero
columns, zero moments, action equality on a real r13 walker fixture);
test_widen_checkpoint.py untouched. Prerequisites: PR-9 (fingerprint carries the
`command` section); D-D3.
*Cleanup CU-8b (landed as #582 on 2026-10-01): `policy_loading` owns the SB3 import helper (`_ensure_sb3`) and the
VecNormalize sidecar resolver (`_resolve_vecnorm_sidecar`) and no longer imports `train_base`; this PR's
`neutralize_command_columns` and `assert_command_blind` join that module.*

### PR-11. Manifest nodes: stage TOMLs and `[[stages]]` entries, trained by train_base (M, about +370)
Goal: per species, three new stage TOMLs under G1/G2 (the assessment listed
four, with a separate `follow_direction_speed.toml`; G1 folds it into
`follow_direction`): `configs/<species>/follow_direction.toml` (`[env]
command_mode` for the full command set of G2, `command_config` table with
`speed_range` as a fraction `[0.5, 1.0]` of `cruise_speed`, verified identical
on all six species, `[ppo]` = the recipe's learning_rate/ent_coef/target_kl,
`[curriculum] timesteps/warmup_timesteps/warmup_clip_range/gate_kind =
"none/v1"`), `follow_direction_difficult_terrain.toml` (the target deliverable:
the same commands plus terrain_* scalars and `terrain_families`), and
`difficult_terrain.toml` (commands-free, terrain_* scalars, `terrain_families`;
the optional diagnostic sibling); `[[stages]]` entries after `behavior` with
`warm_start_from = "locomotion"` / `"follow_direction"`, `deliverable = true`,
`recipe = "follow"` / `"terrain"` (the manifest allows a second child of
locomotion: `parent_of`, stage_manifest.py:232-237; `resolve_behavior` picks the
deepest deliverable per label, 253-262, so `follow` resolves to
`follow_direction_difficult_terrain`). The 19 per-species scale values
(`cruise_speed`, three scales, `course_distance`, `max_episode_steps`, 13
terrain size keys) come from the TOMLs verbatim. D-D5 takes `extends =
"locomotion"` (~20 lines in `load_stage_config`; no inheritance exists today)
over self-contained `[env]` blocks like configs/trex/behavior.toml. The nodes
are non-advancing (no `legacy_number`), so `train_curriculum --target
follow_direction` refuses them by design (train_base.py:1937-1955) and they run
through the notebook chain loop or `train --stage <id> --load-mode
initialize_next_stage`; the notebook needs no new cell. The final node is
SB3-only (G1). ~40 lines of tests asserting the entries load through
`load_stage_config` and resolve their parent.
Folds in: the target half of the trainer finding (corrected), the stage-TOML
form of the config-dialect finding (not a third dialect), step (b) of the
config-generator proposal (the generator-script variant is superseded by D-D1),
and the target half of the cross-cutting trainer finding.
Breaks: nothing yet (the trainer still reads the 66 files until PR-12). Workflow
consequence to state: under the manifest a follow/terrain node needs a CERTIFIED
locomotion ancestor at the Phase C interface (r13 for trex; each other species
at its own current `policy_interface_revision` in configs/plant_versions.toml).
At the time of the assessment none existed; the
2026-09-17 Drive survey found one for trex (20260914_123816, both nodes reused
by `TRUNK_FROM = "auto"`), and the walker sessions in NEXT_STEPS.md produce the
other five species' in parallel with this sequence (G3). The explicit-load
escape is the manual single-node cell (cell 24). Validation: `load_stage_config`
tests; one real-PPO smoke `train --stage follow_direction` for
compsognathus_robot and trex through the SB3 job. Prerequisites: PR-9, PR-10;
D-D1, D-D5, G1, G2.
*Cleanup CU-13 (landed as #588 on 2026-10-02): `load_stage_config` resolves `extends`, so inheritance exists: in the
per-table form of D-D5's amendment of 2026-10-02, `extends = { stage = "<parent id>", tables = [...] }`, one level
deep, with no chains and never `curriculum`, which the three recovery stages use to extend stance with every resolved
stage config byte-identical; this PR's nodes name `locomotion` the same way, listing the tables each inherits.*

### PR-12. Delete the parallel trainer, router, checkpoint module, 66 TOMLs, notebook mode and their tests (XL by count, almost all deletion; about -5,300) — notebook-only slice LANDED as #552, 2026-09-24 (D-D13)
Goal: delete environments/shared/train_behaviors.py (601), behavior_notebook.py
(351), behavior_checkpoint.py (457), configs/*/behaviors/ (66 files, 3,030
lines), pyproject.toml:113 glob; tests test_behavior_checkpoint.py (420),
test_behavior_notebook.py (792), test_behavior_species_training.py (242;
replaced by one real-PPO smoke per species through the manifest node, ~200
lines), test_behavior_recipes.py (179; replaced by PR-11's manifest tests), the
`CommandEnv` fixture; notebook cells 7, 19, 20, 33, 34, 39, the 14 guard sites
(dedent cells 8,10,11,12,14,24,26,28,30,32,36,38; cell 16:928-934 and 22:38-47
keep one print each), cell 6:48-84 (10 `BEHAVIOR_*` knobs,
`COMMAND_TERRAIN_BEHAVIOR`, the `ModuleNotFoundError` shim at 70-76, the
N_ENVS/SEED caveat comments at 14-15), cell 3:7 stale comment; `BEHAVIOR`
dropdown becomes `stand | walk | hunt | follow | terrain` plus stage ids; the
guard-stripping helper `_canonical_source` (test_sb3_notebook_pins.py:78-88) and
the 46 pin lines #540/#541 added; CI collapses the two behavior steps
(python-ci.yml:300-324) into the SB3 job and drops the wheel-step recipe loop
(182-187). docs/TRAIN_DIRECTION_AND_TERRAIN.md shrinks to an operator guide
(~150 lines: behavior table, species-scale table, terrain templates, replay)
with the CLI walkthroughs and notebook-knob paragraph removed.
Folds in: the delete halves of the trainer, checkpoint and cross-cutting
findings, the notebook mode-switch findings (no interim behaviors notebook,
D-D8), the `BEHAVIOR_*` half of the knob finding, the checkpoint-suite collapse,
and the operator-guide cut.
Breaks: every behavior bundle from #540/#541
(bundle.json/run.json/`checkpoints/step-*`) stops loading anywhere (D-D9); pilot
output under `logs/<species>/ppo/behaviors/` becomes orphaned (leave it; none is
on Drive, [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §5.2). The
canonical chain, both r11 parents, the r13 run,
TRUNK_FROM/RETRAIN_FROM/RUN_ID and the resume cell are untouched (WIDEN_FROM
left with PR-14a, D-D14).
Validation: full shared and species suites, notebook parse and pins, the SB3 job
with the new smoke, wheel step. Prerequisites: PR-11.

The notebook items of the Goal left with the notebook-only slice (status table above); everything else in
the Goal is this PR's. The slice's as-executed record, and the curves-cell fix it carried, are at `20ab100`
(`0.3.9`), as §3's note says.

### PR-13. The gate kind, its evidence writer, and the end of the second gate system (L, about -400 net)
Goal: register `terrain_command/v1` (D-D6) in `GATE_KINDS` and
`_REQUIRED_THRESHOLD_KEYS` in the same commit (parity raise at
gate_schema.py:186-190) with thresholds from configs/behavior_certification.toml
carried once into a `[curriculum]` block per node (never 66 copies), the first
values set by G4; move `episode_measurements`
(behavior_certification.py:181-224) and `summarize_episode`
(behavior_evaluation.py:72-300) into a `curriculum/<kind>_gate.py` as the
per-episode statistic; write an evidence writer in the `write_recovery_evidence`
pattern producing per-episode rows (`command_v_x`, `command_yaw_rate`,
`tracking_error_v`, `tracking_error_yaw`, `terrain_family`, `course_progress_m`,
survival), hash-bound to model.zip/vecnormalize.pkl the way
`_task_success_stage_gate` binds its CSV (gates.py:463-); judge in
reporting/gates.py; the notebook's JUDGE/TRAIN branches then write
`gate_verdict.json` through `_apply_stage_gate` with no new cell. The name is
honest by decision: the certificate samples per EPISODE and lacks the plan's
per-event settle/dwell statistic, worst-of-eight-heading-bins floor and
`paired_difference_lcb` against the command-blind walker (plan:803-815), so it
is registered for what it is; the plan's `command_tracking/v1` with the paired
null is a later second kind, outside this sequence. Delete
behavior_certification.py (558), configs/behavior_certification.toml,
test_behavior_certification.py (212; panel-judging cases re-homed as gate cases
in test_gate_dispatch_fail_closed.py / test_reporting_gates.py),
`evaluate_saved_panel`, the `evaluate_behavior` wrapper layers, the three `_sha`
helpers (`train_behaviors._sha` is gone; `behavior_replay._sha` :50 and
`certified_library._hash_file` (left with PR-5 on 2026-09-20) are replaced by
`result_bundle.hashing.sha256_file`) and the second `_json_value`
(behavior_replay.py:36-47 vs behavior_evaluation.py:301-314). Keep
`BehaviorReplayRecorder`, `capture_terrain_snapshot`, `write_terrain_maps` (keep
verdict; the canonical `record_stage_video` rolls a different episode on purpose
and carries the stance CSV the r11/r13 reviews read).
Folds in: the gate-kind finding (corrected), the replay-recorder finding
(corrected; recorder deletion rejected), the helper half of the evaluation
finding, and invariant 10 (the fail-closed dispatch test gains a case).
Breaks: nothing on Drive: no `certification/certificate.json` exists there (no
behavior has passed the certificate per #541, and its writer left with PR-5).
Validation: gate schema tests,
dispatch fail-closed test, shared suite, one notebook smoke with `gate_kind`
set. Prerequisites: PR-11, PR-12; D-D6, G4.

Open since 2026-09-28: the gait-quality plan
([GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md)) proposes, as its
GQ-18 (open), that `terrain_command/v1` gain a per-episode gait clause,
appended to D-D6, because the certificate tracks pelvis velocity and would
certify a hop or a slide; its PR-G10 would add the clause inside PR-13.
Nothing is decided, and D-D6 is unchanged.

### PR-14. Notebook: one storage path, one disconnect path, `train_stage` over `train_base.train`, `RUN_ID` as a knob — split into PR-14a, PR-14b and PR-14c (decision D-D15)
All three parts landed (status table above); full text at `20ab100` (`0.3.9`), as §3's note says.

#### PR-14a. The storage path: the widen cell out (D-D14), `RUN_ID` as a knob, complete runs refused before training (L by count: code +380 / −20, tests +894 / −680, notebook 2,330 → 2,271 source lines) — LANDED as #553, 2026-09-24
Landed (status table above); full text at `20ab100` (`0.3.9`), as §3's note says.

#### PR-14b. One disconnect path, video display and the baseline cells (M by count: code +183 / −1, tests +245 / −50, notebook 2,271 → 2,079 source lines) — LANDED as #554, 2026-09-24
Landed (status table above); full text at `20ab100` (`0.3.9`), as §3's note says.

#### PR-14c. `train_stage` over `train_base.train` (L by count: code +356 / −31, tests +402 / −278, notebook 2,079 → 1,492 source lines) — LANDED as #555, 2026-09-24
Landed (status table above); full text at `20ab100` (`0.3.9`), as §3's note says.

### PR-15. Docs fold, CHANGELOG Changed/Removed, test helpers and pin budget (M, about -290)
Goal (restated 2026-10-04 by cleanup CU-17; the earlier text is at commit `20ab100`): a residual docs pass for
PR-8..PR-13 under the one-landing-record rule (docs/README.md, Conventions): their landed sections here become
pointers like PR-1..PR-7's, and NEXT_STEPS.md §4 drops their rows; website/docs/training/recipes.md: replace
the 2026-09-19 pilot paragraph with the landed node description and fix the Leaf row (still "today
`behavior` (hunt)", :27) once PR-11's nodes exist; docs/RESULT_BUNDLES.md canonical layout stays as is (no
`certified_inputs/` after PR-4); layout literals that survive become named constants next to
`result_bundle.constants.ANCESTORS_DIRNAME`; a pin budget proposed afresh, since the cleanup plan's §7 rejects
"the pin budget as proposed" (it overlooks four tombstone tables and narrows checks to code cells). The exact
PPO update tuple and `mesozoic_behavior_stage_start` pins (`test_behavior_checkpoint.py:764-767`) leave with
that file in the rest of PR-12. Keep the D-C17 `DEFAULT_MAX_REVISION_GAP == 1` pin (test_widen_checkpoint.py),
the `TRUNK_FROM = "auto"` default pin (D-A25) and the empty-default pins for RETRAIN_FROM and RUN_LABEL. Done
before the restatement: the operator guide is under docs/README.md's Living reference (PR-2);
`notebook_cells.py` replaced the extractor copies (test_sb3_notebook_pins.py:75-97,
test_compsognathus_training.py:386-392; CU-4, the amendment below); and every landed PR from PR-4 on wrote its
own CHANGELOG entry, released. This document is then marked complete in the docs index.
Folds in: the remainder of the documentation findings and the pin budget.
Breaks: nothing at runtime. Validation: workflow run. Prerequisites: PR-14c.
Amended 2026-09-29 (cleanup CU-4, the cleanup plan's §2 row 9): the
`notebook_cells` slice left with CU-4.
`environments/shared/tests/notebook_cells.py` (standard library only)
replaced the 16 notebook-extraction sites in nine test files, the two cited
above among them, and CI's lint job runs it by path to parse every notebook
code cell. The cross-file duplicates of notebook pins that CU-4 left (the
`RUN_RECOVERY_STAGE` pin three times, the no-inference pin and the
`chain_results` routing pin twice each) stay with PR-15's pin budget.

Running totals (net, using the corrected figures; moves between files count
zero): PR-1 0; PR-2 +100; PR-3 +20; PR-4 -2,400; PR-5 -1,400; PR-6 -275; PR-7
-600 (measures −427 net code, test and CI lines: code −403, tests −23, CI −1, since
the trex env cases now run for all six species); PR-8 -170; PR-9 -150; PR-10 +180; PR-11 +370; PR-12 -5,300; PR-13 -400;
PR-14 -450 (the pre-split estimate; PR-14a measures about +290 net code, test and notebook lines, since it adds the complete-run
refusal, the on-disk widen guards and their tests while deleting the widen cell; PR-14b measures +185 (code +182,
tests +195, notebook −192 source lines), since it moves the notebook's helpers and zero-action body into the
package with new tests; PR-14c measures −138 (code +325, tests +124, notebook −587 source lines)); PR-15 -290. Net about -10,750 from 723f58f; stated as about 10,500
with a plausible band of 9,000-12,000 (D-D5's TOML form is worth ~700 either
way; D-D7 could remove ~700 more notebook lines while adding them to the
package). From 22c1fc8, with PR-1 landed and #543's ~1,200 lines added, about
9,500.

## 4. Notebook target

Before: 40 cells, 2,500 lines (2,294 code, 206 markdown; 22 code cells; 12 whole
cells and 2 partial cells behind `COMMAND_TERRAIN_BEHAVIOR`). After PR-12 and
PR-14 (the taken form under D-D7: chain loop and resume logic still in
cells as the plan §4.7 pins them): 33 cells, about 1,400 lines. Under the full
package move that D-D7 defers until after PR-14, the same notebook is about 700
lines in 19 sections. The rows keep the plan's order; since the notebook-safety
PR (D-D16) the RESUME cell (row 14) runs ahead of the chain loop (row 12). The
sizes after each PR, as this paragraph listed them before cleanup CU-17, are at
commit `ace8112`, and each is also in the status table, the CHANGELOG or the
cleanup plan's §3.1.

| # | Title | Type | ~lines | Source cells | What moves into the package |
|---|---|---|---|---|---|
| 1 | Title and intro | md | 13 | 0-2 | - |
| 2 | Setup and install | code | 59 | 3 | stale comment 3:7 removed; mediapy stays |
| 3 | Imports and repo root | code | 39 | 4 | - |
| 4 | Configuration | md | 12 | 5 (minus 13-35) | library and behavior prose to the operator guide |
| 5 | Knobs | code | 30 | 6 | drops `SOURCE_SELECTION`, `CERTIFIED_LIBRARY_ROOT`, `PUBLISH_CERTIFIED`, `CERTIFIED_COMPARISON_EPISODES`, ten `BEHAVIOR_*`, `COMMAND_TERRAIN_BEHAVIOR`; gains `RUN_ID` |
| 6 | Storage, provenance, trunk | code | 131 | 8 | `CERTIFIED_LIBRARY` line gone; `RUN_ID` resolved into `_ACTIVE_RUN_ID` (a path or surrounding whitespace refused; new run or re-entry printed); a command-line-widened root's seed checked before `initialize_result_bundle` runs, and `RUN_DIR` and the memo bound only once it accepted the run (PR-14a) |
| 7 | Resolve chain, table, trunk selection | code | 89 | 10 | stays after storage (its auto-trunk half reads `LOG_BASE`, `PLANT_IDENTITY`, `RUN_DIR`); ends with the complete-run and widened-root refusals (PR-14a); since cleanup ROW-4/6 it ends with the complete-run refusal and `record_trunk_run`, which writes `RUN_DIR/trunk_run.json` (decision 4 (a)), and D-C13 holds through row 12 instead of the widened-root refusal |
| 8 | Archive-load preflight | code | 65 | (2026-09-19) | the `WIDEN_FROM` branch gone: the trunk's root handoff, else a throwaway (PR-14a); the widen cell (old 11) deleted, D-D14; the cell's body moves into `policy_loading.sb3_archive_load_preflight`, and the cell is that one call, 15 lines (cleanup CU-6, carried out 2026-10-03) |
| 9 | Explore + zero-action baseline | md+code | 1+65+40 | 9, 13, 14 | the zero-action body after its knobs folded into `zero_action_baseline.preflight` (PR-14b); cell 12, the random baseline, deleted (PR-14b) |
| 10 | Training infrastructure | code | 211 | 16 | `train_stage` -> a wrapper over `train_base.train` (refusals, banner, one `train()` call, the evaluation; about 100 lines, PR-14c); `evaluate_stage_checkpoints` -> reporting/stage_artifacts, called with the session's globals (PR-14c); stamp block and mediapy probe deleted; `disconnect_runtime` (explicit parameters), `halt` and `display_stage_videos` in `notebook_runtime` (PR-14b); `train_stage` takes `evaluate=True`, which the RESUME cell sets to `False` (cleanup CU-6, carried out 2026-10-03) |
| 11 | Visualization | code | 46 | 18 | - |
| 12 | Chain loop | md+code | 10+270 | 21, 22 | guard, library lookup and publish block deleted (-70); refuses a write into a complete run the resolve cell could not predict (PR-14a); loop stays AST-pinned; judges a node `RUN_DIR` holds trained but unjudged before it consults any trunk (cleanup ROW-4/6, decision 6 (b)) |
| 13 | Manual single node | md+code | 3+98 | 23, 24 | guard removed |
| 14 | Resume interrupted node | md+code | 45+106 | 25, 26 | remedy prose deleted; guard removed; its periodic-pair walk is `curriculum.newest_intact_periodic_pair`, and it trains with `evaluate=False`, the chain loop's JUDGE branch evaluating the node from disk (cleanup CU-6, carried out 2026-10-03); checks a resume against the run's `trunk_run.json` (cleanup ROW-4/6) |
| 15 | Evaluate | code | 1+20 | 27, 28 | guard removed |
| 16 | Training curves | code | 1+12 | 29, 30 | guard removed |
| 17 | Replay videos | md+code | 5+11 | 31, 32 | guard removed; IPython Video (PR-14b) |
| 18 | Cleanup | code | 1+31 | 35, 36 | guard removed |
| 19 | Auto-disconnect | md+code | 3+2 | 37, 38 | one cell |

Deleted outright: cells 7, 12, 19, 20, 33, 34, 39 and all 14 guard sites (the
notebook-only PR-12 slice took all but cell 12, the random baseline that PR-14b
deletes, and found 15 guard sites with the preflight). Every
knob the maintainer uses keeps its name and meaning; `BEHAVIOR` gains `follow`
and `terrain`.

## 5. What stays as-is and why

Keep verdicts (each checked against the code during the review):
- The two model/data pools and terrain outside the plant identity. A heightfield
  model is a different plant by construction (physics digest hashes `geom_type`,
  `geom_dataid`, hfield data; plant_contract/physics_layer.py:133-139, 240-246),
  the plane and a flat hfield differ in narrow-phase contact (5/5 vs 0/5 in
  #540), and `build_terrain_model` is already the single injection point.
- `BehaviorReplayRecorder` and the terrain maps stay a distinct exporter;
  `record_stage_video` rolls a fresh episode at `replay_seed` with camera
  presets and the stance CSV the r11/r13 reviews depend on.
- widen_checkpoint, its identity gate (gap >= 1, D-C17), `WIDEN_LINEAGE_KEYS`
  and the reseed plumbing are the single mechanism the maintainer's r11 -> r13
  sessions use; only the probe constants are shared.
- The chain loop is not `train_curriculum` (no JUDGE branch, no frozen-null
  recovery flow, no `generate_stage_artifacts`, no library publication in the
  CLI); the behaviors runner cannot be a branch of the loop today; the
  widen/resume/preflight guards encode real failures (run 20260821_142144; the
  PPO.load segfault).
- direction_commands.py and terrain.py are the genuinely new content; do not
  merge the controller into command_frame.py (amendment A3 keeps command_frame
  numpy-only).
- test_terrain.py, test_terrain_sampling.py, test_direction_commands.py, the
  widen golden pins, the AST notebook pins and (until PR-5) the
  library-immutability tests are distinct; they relocate out of the SB3 lists
  only.

Proposal parts dropped or rejected during the review (no proposal was refuted as
a whole):
- Moving `body_frame_velocity`/`tracking_metrics`/`gaussian_tracking_reward`
  into reward_functions.py deletes nothing; dropped.
- Deleting `_settle_root_on_ground` and the `lowest_ground_clearance` raise;
  rejected (different algorithm; hfield guard).
- A `starting_updates` / `recommend()` pointer in place of the library: a new
  mechanism; dropped under D-D4 (pure deletion, with `select_trunk` as the
  replacement).
- A `[curriculum]` block in all 66 TOMLs (+800 lines) and the name
  `command_tracking/v1` for a gate that does not implement the plan's statistic;
  rejected.
- A shared rollout generator: nets ~0 lines and touches the canonical `evaluate`
  CLI; dropped.
- `train_curriculum --target follow_direction`: refused by
  `_resolve_curriculum_target` for non-advancing nodes; dropped from the target.
- The 11-template + 6-profile dialect and the generator script; superseded by
  stage TOMLs under D-D1.
- Terrain in `BaseDinoEnv.reset()`; rejected (reset source is fingerprinted; it
  would be an r14 bump for all six species; D-D10).
- A `--zero-command-columns` widen mode; replaced by the `policy_loading`
  primitive (widen's gate refuses same-width parents).
- Deleting the replay recorder; rejected (keep verdict above).
- An interim `sb3_behaviors.ipynb`; not built (D-D8): the mode is deleted in
  PR-12 and the split would be maintained for weeks then thrown away.
- Removing the `_ACTIVE_RUN_ID` memo; optional
  (test_sb3_notebook_pins.py's `TestStorageCellRerun` pins the rerun behaviour since PR-4 and
  jax_training.ipynb keeps the same memo). The JAX notebook left with cleanup
  PR-B (D-D17, 2026-09-28), so only the SB3 notebook keeps the memo.
- Two of the CI-length items: already the case / not a large contributor.
- The CHANGELOG `Added`/`Migration` block is an addition (+70), kept for its
  content, not counted as removal.

## 6. Decisions (the D-D series, taken 2026-09-17)

Recorded in [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2 under the
same ids. Ordered by how many PRs each blocks, as the review posed them. "Taken"
means decided by the maintainer on 2026-09-17; the last two of that day (D-D11,
D-D12) were recommended then and confirmed on 2026-09-20, when D-D13 and D-D14
were taken; D-D15 was taken on 2026-09-24 and D-D16 on 2026-09-25. D-D17,
D-D18, D-D19 and D-D20 were taken on 2026-09-26 from the decisions
[CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §2 lists. D-D21 was taken
on 2026-09-27 from row 20 of the same list. D-D22 was taken on 2026-09-29 from
row 16 of the same list. D-D23 was taken on 2026-10-06 from the gait plan's
GQ-6 and GQ-7 ([GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md)
§2), for stance only; it is outside this sequence. D-D24 was taken the same
day, overtaking the same plan's GQ-12 for the T. rex; it is outside this
sequence too. D-D25 was taken the same day as well, overtaking KNOWN_ISSUES'
deferral of the velociraptor plant review and the same plan's §9 for the
velociraptor foot sensor; it is outside this sequence too.

| Id | Question | Decision | Unblocks |
|---|---|---|---|
| D-D1 | Are direction-following and terrain traversal ordinary manifest nodes under `locomotion` (the plan's Phase D), or a standalone pipeline that only dedupes inside itself? | Taken: manifest nodes, stage TOMLs, gate kinds, ancestors reuse. Consequence accepted: a follow/terrain node needs a certified locomotion ancestor at the Phase C interface (r13 for trex; each other species at its own current revision). Trex has one since the Drive survey (20260914_123816); the other five species get theirs from the NEXT_STEPS.md walker sessions; the manual cell is the escape hatch until then. | PR-9 to PR-15 |
| D-D2 | Fill the six reserved `command_*` kwargs literally (D-C3) or replace the five numeric ones with one `command_config: DirectionCommandConfig \| None`? | Taken: replace. The five (`command_speed_range`, `command_lateral_range`, `command_yaw_rate_max`, `command_switch_interval`, `command_switch_jitter`) have no reader anywhere and the dataclass is already validated and JSON-able; the task-fingerprint carve-out is extended so no canonical `task_sha256` moves while `command_mode` is `"none"`. Amends D-C3. | PR-9 |
| D-D3 | Command-slice normalisation: the plan's invariant 8 (reseed to mean 0 / var 1, statistics keep updating) or the pilots' exact passthrough (`BehaviorVecNormalize`, a pickled class in every sidecar)? | Taken: reseed. The passthrough subclass is deleted; #540/#541-trained policies saw different inputs and are not continuations. | PR-8, PR-10 |
| D-D4 | Keep automatic parent selection when the library goes? | Taken: keep it, as `ancestors.select_trunk` (D-A25, landed as #543): scans the runs under `logs/<species>/<algo>/`, applies the seven reuse rules root-first and picks the run covering the most of the chain; `TRUNK_FROM = "auto"` is the notebook default and `--trunk-from auto` the CLI form. It landed before PR-4/PR-5 delete the library trio, so the capability never lapses. The library is deleted outright; no recommendation pointer survives. | PR-4, PR-5 |
| D-D5 | Stage TOML form for the new nodes: a ~20-line `extends` key in `load_stage_config` (small files, new mechanism) or self-contained `[env]` blocks like configs/trex/behavior.toml (no new mechanism, ~30 restated keys per file)? And the node set: 3-4 per species or all 11? | Taken: `extends`, with four base stage TOMLs per species (`follow_direction`, `follow_direction_difficult_terrain`, `difficult_terrain`, and the existing `locomotion` as the `extends` parent); the terrain templates become a `terrain_families` list inside `difficult_terrain`; single-template runs are documented `[env]` overrides, not files. Refined by G1: three new files per species, `follow_direction_speed` folded into `follow_direction`. **Amended 2026-10-02 (the maintainer, as the cleanup plan's decision 9), carried out by its CU-13:** CU-13 pulls this decision's `extends` forward for recovery ← stance in trex, compsognathus and compsognathus_robot, in the form the maintainer chose on 2026-10-02: a per-table list, `extends = { stage = "stance", tables = [...] }`, one level deep, with no chains, never naming `curriculum`, and trex recovery not inheriting `[sac]`. As CU-13 carries it out, `extends = { stage = "<parent id>", tables = [...] }` is a top-level key naming a stage of the same species (resolved through the stage manifest of the file's own directory) and the tables it inherits, each the parent's with the file's own keys overriding in place or appended (a nested table the file declares replaces the parent's whole); trex recovery lists `stage`, `env` and `ppo` and stays PPO-only, and the compsognathus pair lists `sac` too. All 21 resolved stage configs are byte-identical, key order included, so no digest moves. CU-13's step 2 (compsognathus_robot ← compsognathus) waits for PR-12 and for an amendment of this form: a parent id resolves only through the stage manifest of the file's own directory, and the robot's recovery already extends the robot's stance, so a robot stance that extended compsognathus's would put that recovery in a chain. This decision's own nodes wait for PR-11, where each names `locomotion` in this form. CU-13 landed as #588 on 2026-10-02. | PR-11, PR-12 |
| D-D6 | Gate: implement the plan's `command_tracking/v1` (per-event settle/dwell, heading-bin floor, paired null against the command-blind walker) or register the certificate's per-episode statistic under an honest name? | Taken: honest name first. Pilots run `none/v1` (recorded, not enforced); then the certificate's per-episode statistic is registered as `terrain_command/v1` with one shared threshold block (first values from G4); the plan's `command_tracking/v1` with the paired null is a later second kind, new work outside this sequence. | PR-13 |
| D-D7 | Notebook depth: only the `train_stage` wrapper (the plan §4.7 AST pins on the chain loop stay) or also move cells 11/22/24/26 verbatim into a package module behind a `NotebookSession` (rewrites most of the 2,125-line pin file and plan §4.7:882)? | Taken: wrapper now (PR-14); whether to move the chain loop / widen / resume cells into a package module is decided after PR-14 has settled. **Amended 2026-10-03 (the cleanup plan's §2 row 9), carried out by its CU-6:** a slice of the resume cell and the archive-load preflight move into the package ahead of the deferred full move. The RESUME cell's periodic-pair walk is `curriculum.newest_intact_periodic_pair`, which reads `policy_loading._PERIODIC_CHECKPOINT_RE`; the archive-load preflight cell is one call of `policy_loading.sb3_archive_load_preflight`; and `train_stage` takes a keyword-only `evaluate` (default `True`; the RESUME cell passes `False`, since the chain loop's JUDGE branch evaluates, from disk, every node the RESUME cell trains, cleanup ROW-4/6). The chain loop, the rest of the RESUME cell (its refusals, warnings and report), `node_budget` (in the infrastructure cell, read by the chain loop, the RESUME cell and the manual cell), and the storage and resolve cells stay in the notebook, and the §4.7 AST pins on the chain loop stay, as this decision says; the full package move stays deferred (the cleanup plan's §7). | PR-14's scope |
| D-D8 | Build an interim behaviors notebook now, or tolerate the mode switch until PR-12? | Taken: tolerate. #542 removed the dangerous default; the switch is deleted in PR-12. | PR-12 |
| D-D9 | Are any #540/#541 behavior bundles on Drive worth carrying forward? Their identity hashes environments/shared/behavior_env.py itself, so exact resume already breaks on any edit; #540 calls them pilots. | Taken: none. The bundles are evaluation-only; no bundle is carried forward as a training parent. | PR-6, PR-7, PR-9, PR-12 acceptance |
| D-D10 | Terrain in the env: one generic opt-in subclass, or an r14 interface bump that puts the model swap into `reset()`, batched with the queued height-channel removal (plan:668-673)? | Taken: opt-in subclass; no r14 bump (the reset source is fingerprinted). | PR-7, PR-9 |
| D-D11 | May CLI runs record stage duration and seed model construction like the notebook does? | Confirmed 2026-09-20: yes (PR-14 aligns `train()` with the notebook's `alg_kwargs["seed"]` line and its duration recording). Amended by PR-14c: implemented in `train()`, so for the `train` subcommand and Vertex sweep trials (`sweep/trial.py`), with a seed the algorithm block names kept; `train_curriculum` (`curriculum`) and the Ray Tune worker are not aligned (neither seeds construction nor records a duration), outside PR-14c's scope (D-D7). Amended by D-D17: the Vertex sweep trials and the Ray Tune worker are retired (cleanup PR-A); `train()` serves the `train` subcommand and the notebook's `train_stage`, and `train_curriculum` stays unaligned. **Amended 2026-10-03:** CLI curriculum runs now seed model construction and record stage duration as the notebook and `train` do (cleanup CU-10b, as the maintainer chose on 2026-10-02; the cleanup plan's §2 row 20): `train()` and `train_curriculum` share one stage body, which seeds construction with `alg_kwargs.setdefault("seed", seed)` (an algorithm-block seed kept), and each node the `curriculum` subcommand trains records `run.duration_seconds`. The full text is in BEHAVIOR_RECIPES_PLAN.md §6.2. CU-10b landed as #590 on 2026-10-03. | PR-14 |
| D-D12 | Drop the dead `lateral_speed_scale` field (always divides a zero) when the TOMLs are rewritten? | Confirmed 2026-09-20: drop in PR-11/PR-12 (PR-8 item (d)). | PR-11, PR-12 |
| D-D14 | What happens to the widen path (`WIDEN_FROM`, `WIDEN_MAX_REVISION_GAP`, the widen cell, `widen_checkpoint`) after the two pending parents are widened? | Taken 2026-09-20: keep it for NEXT_STEPS.md sessions 1 and 2, then CLI-only — the notebook refactor (PR-14, or a PR right after it once both sessions are decided) deletes the widen cell and both knobs; `widen_checkpoint` stays a command-line tool for the next interface bump. Amends the §4 "knobs kept" list. Both sessions decided PASS by 2026-09-21 (`20260920_010912`, `20260921_203149`), so PR-14 deletes them. **Scheduled by D-D15:** PR-14a is the PR that deletes them. | PR-14 |
| D-D13 | In which order do PR-3 .. PR-15 land now that the hold is lifted? | Taken 2026-09-20: notebook-first. PR-3, PR-4, PR-5, PR-6, then a notebook-only slice of PR-12 (the `COMMAND_TERRAIN_BEHAVIOR` switch, the ten `BEHAVIOR_*` knobs, cells 7/19/20/33/34/39 and the guard sites, `behavior_notebook.py` with its tests and pins; `train_behaviors.py` stays a CLI-only path) pulled ahead of PR-11, then PR-14, then PR-7 .. PR-11, the rest of PR-12, PR-13, PR-15. Amends D-D8: the switch is tolerated only until that slice, and the direction/terrain pilots have no notebook path between the slice and PR-11 (evaluation-only under D-D9). | the whole sequence |
| D-D15 | How does PR-14 land, now that D-D14 deletes the widen path in it and the complete-run bug of 2026-09-23 (a node trained in place into a `complete` bundle fails its bundle write after training) falls in its storage path? | Taken 2026-09-24: as three PRs, in the order PR-14a, PR-14b, PR-14c. PR-14a, the storage path: D-D14's deletion of the widen cell, `WIDEN_FROM`, `WIDEN_MAX_REVISION_GAP` and `select_trunk(widen_from=)`; the old item (b) without its resolve-before-storage reorder (no widen-seed read is left to move; a root widened on the command line keeps D-C14 through an on-disk seed check before `initialize_result_bundle` and D-C13 through a resolve-cell refusal of a trunk until it holds a verdict), the living copies of the restart / memo-reset remedy deleted and the decision rows amended append-only; item (c), `RUN_ID` as a configuration-cell knob resolved into the `_ACTIVE_RUN_ID` memo every later cell reads; and the complete-run refusal before anything is trained or written, with the chain loop refusing what the resolve cell cannot predict, the manual and resume cells refusing the same write and the zero-action cell's companion fix. PR-14b: item (d), the one disconnect path, IPython video display and the baseline cells. PR-14c: item (a), `train_stage` over `train_base.train` (D-D7, D-D11). Amends D-D13's order. **Amended 2026-10-03 (the cleanup plan's §2 row 6, ROW-4/6; not a D-D row):** the resolve cell no longer refuses a trunk over a widened root without a verdict: D-C13 holds through the chain loop, which judges a node `RUN_DIR` holds trained but unjudged, the widened root among them, before it consults any trunk (a widened pair cut short is refused as an interrupted node), and refuses a root widened into a run that already holds it as an `ancestors/` record when the root is an ancestor of `BEHAVIOR`'s node, naming a new run id; `result_bundle.refuse_trunk_over_unjudged_widened_root` stays exported and uncalled. The D-C14 seed check is unchanged. The full text is in BEHAVIOR_RECIPES_PLAN.md §6.2. ROW-4/6 landed as #591 on 2026-10-03. | PR-14a, PR-14b, PR-14c |
| D-D16 | May the SB3 notebook's RESUME cell train into a node that is already judged or finished, and may a `QUICK_TEST` run sit where trunk selection and seed replication look? | Taken 2026-09-25: no to both. The RESUME cell trains nothing for a node holding `gate_verdict.json` or its final pair and refuses a spent budget without that pair; `QUICK_TEST` runs live under `<algo>_quick_test/`; the RESUME cell moves ahead of the chain loop and the sections regroup. The full text is in BEHAVIOR_RECIPES_PLAN.md §6.2. **Amended 2026-09-25 (the #558 follow-up):** only an intact final pair marks a finished node, in the RESUME cell and the chain loop alike; the run memo counts only in its tree; a resume `RETRAIN_FROM` covers is refused. **Amended 2026-10-03 (the cleanup plan's §2 rows 4 and 6, ROW-4/6; not a D-D row):** a resume `RETRAIN_FROM` covers is still refused, and the route the #558 follow-up named for such a node (`BEHAVIOR` set to it, then a fresh `RUN_ID`) is retired: with `RETRAIN_FROM` unset it is resumed in place, and the chain loop judges it before it consults the trunk; the RESUME cell also refuses a node the run holds as an `ancestors/` record and, once the run holds one, a resume under another trunk than `trunk_run.json` records. The full text is in BEHAVIOR_RECIPES_PLAN.md §6.2. ROW-4/6 landed as #591 on 2026-10-03. | The notebook-safety PR (#558) and its follow-up |
| D-D17 | Do Ray Tune, the Vertex AI tuning sweeps, mjlab and JAX/MJX stay while SB3 has certified only part of the species and behaviors, and does the single-job Vertex route stay? | Taken 2026-09-26: all retired, the single-job Vertex AI route and GCS artifact upload too; Stable-Baselines3 is the only training, evaluation and evidence backend until every species' stage chain and behavior is certified. PR-A removes the sweeps, mjlab, the Vertex route and GCS upload, PR-B the JAX/MJX runtime; a frozen MJX interface core stays so that no digest moves. The archive points (the cleanup plan's decision 2) are still open. The full text, with the rows it amends, is in BEHAVIOR_RECIPES_PLAN.md §6.2. **Amended 2026-09-26:** PR-A is split into two PRs under the same D-D17, each with its own review: PR-A deletes the sweeps and mjlab and stays pure deletion; PR-A2 removes the single-job Vertex AI route and GCS upload, with an end-to-end test of the command-line curriculum path, so the edits to live SB3 code (`cli.py`, `config.py`, `train_curriculum`, `reporting/csv_output.py`) get their own review (the cleanup plan lands PR-A2 after PR-A). **Archive points settled 2026-09-27** (decision 2, option (c)): no archive tags; the retired code stays reachable from the release tag `0.3.8` (`afad625`) and git history. As carried out by PR-A (2026-09-27): Ray Tune, the Vertex AI tuning sweeps and mjlab removed (42 files, 13,222 lines); D-A12, D-A15, D-B1, D-B12 and D-D11 amended and A6 superseded; no digest moves. Landed as #564 on 2026-09-27. As carried out by PR-A2 (2026-09-27): the single-job Vertex AI route and GCS artifact upload removed (4 files and 957 lines, plus the upload code in `config.py`, `cli.py`, `train_curriculum` and `reporting/csv_output.py`, and the `[gcp]` extra), with an end-to-end test of the command-line curriculum path; no digest moves. Landed as #565 on 2026-09-28. As carried out by PR-B (2026-09-28): the JAX/MJX runtime removed (30 files and 15,665 lines: the 13 JAX modules (trainer, curriculum, evaluator and their helpers), the JAX notebook and guide, and their tests; plus the MJX environment, the stage writer `save_jax_stage_artifacts`, `validate_mjx_environment_plant`, the `[jax]` tables of the 12 stage TOMLs, the `jax` and `jax-cpu` extras and the `test-jax-cpu` job), and `load_stage_config` now refuses a `[jax]` table; the 549-line frozen MJX interface core stays, pinned by `test_plant_contract_frozen_mjx.py`; A7 narrowed and D-A5, D-B13, D-C3, D-C4, D-C7, D-C16 and G1 amended; no digest moves. PR-B landed as #566 on 2026-09-28. | PR-A and PR-B of the cleanup plan (its §4); PR-A2 too (split 2026-09-26) |
| D-D18 | Should CI type-check the tree with SB3 and torch installed? | Taken 2026-09-26: yes, in the SB3 job, with SB3 pinned as the notebook pins it and CPU `torch==2.13.0`; mypy must report no errors there, and ruff and mypy are each pinned to one version for CI and pre-commit. As carried out by CU-1: ruff 0.16.9 in the lint job, pre-commit and the `dev` extra, mypy 2.3.1 there and in the SB3 job's mypy step, the pre-commit ruff hooks scoped to `environments/` as CI's ruff is, and `test_ci_tool_pins.py` keeping them in agreement. Landed as #561 on 2026-09-26. | The cleanup plan's CU-1; PR-10 |
| D-D19 | When is the CHANGELOG's release cut, now that `[Unreleased] (v0.3.8)` holds most of the file? | Taken 2026-09-26: before PR-A. The three undated `[Unreleased]` headings are dated, an empty one opens, the version moves to `0.3.9.dev0`, and the maintainer tags `v0.3.8`; v0.4.0 stays ROADMAP's milestone. As carried out by the release PR: its first commit (the one tagged `v0.3.8`) reads `0.3.8` and dates `[0.3.8] - 2026-09-27`, `[0.3.2] - 2026-07-21` and `[0.3.0] - 2026-07-09`; its second opens a bare `## [Unreleased]` and sets `0.3.9.dev0`; no digest moves. The full text is in BEHAVIOR_RECIPES_PLAN.md §6.2. Landed as #563 on 2026-09-27; the maintainer tagged `afad625` as `0.3.8` (lightweight, without the `v` prefix) and published it as a GitHub pre-release. | The release PR, before PR-A |
| D-D20 | May a Colab reclaim leave the final pair or a handoff pair truncated, or mixed with the previous pair so that its checks accept it? | Taken 2026-09-26: no. On a Drive/GCS mount `train()` stages the final, best and robust-best pairs locally and publishes them like the periodic pairs, removing the destination file it publishes last before publishing, and puts an empty placeholder in the final zip's place before the final save begins, so a reclaim leaves at worst an incomplete pair its readers reject (a handoff sidecar without its zip, or a final zip `checkpoint_pair_problem` rejects), never a mixed pair they accept, as with a pair a save straight to the mount cut short; off a mount nothing changes, and no digest moves. The full text is in BEHAVIOR_RECIPES_PLAN.md §6.2. Landed as #562 on 2026-09-26. | The cleanup plan's CU-3; PR-13 |
| D-D21 | Which state is the clean base this sequence builds on, and when is it released? | Taken 2026-09-27: 0.3.9, cut once its gate has landed (PR-A, PR-A2, PR-B, CU-2, CU-4, CU-5, CU-7, CU-8, CU-9, CU-11, CU-12, CU-14 and CU-16); 0.3.8 stays the pre-cleanup release; CU-6, CU-10, CU-13, CU-15, CU-17 and PR-8..PR-15 are deferred, not dropped. The full text is in BEHAVIOR_RECIPES_PLAN.md §6.2. **Amended 2026-09-29:** the maintainer accepted splitting CU-7 into CU-7a and CU-7b (CU-7's sha256 regex and validators go to CU-8c), CU-8 into CU-8a, CU-8b and CU-8c, CU-14 into CU-14b (first) and CU-14a, and CU-16 into CU-16a and CU-16b; the gate's names cover all their parts, so the gate itself is unchanged. As carried out by the 0.3.9 cut PR (2026-10-02), once the gate had landed with CU-8c (#583, 2026-10-01), with the details the maintainer chose on 2026-10-01 (the title kept without its "(v0.3.9)" suffix, the date rule and `0.4.0.dev0` as the next development version): its first commit, the one the maintainer tags, sets `version = "0.3.9"` and dates the section `## [0.3.9] - 2026-10-02 — Backend Retirement & Cleanup`, by that commit's UTC author date (re-dated if it is rebuilt on a later UTC day); its second opens a bare `## [Unreleased]` above it, as the 0.3.8 cut did, and sets `0.4.0.dev0`. The digest-snapshot harness reports the golden current (932 lines, 0 errors) on the base, the tagged commit and the head. From `## [0.3.9]` to the end, `CHANGELOG.md` has sha256 `88c579c35fb4ee3a8a29ce1807e0e98210f5861a4e0fd3e12e8800a43e398dca` at the tagged commit and the head; from `## [0.3.8]` it is unchanged (`c750a3fd…773b`). The cut PR merges with a merge commit, so the tagged commit keeps its SHA on `main`, and never through "Update branch": if `main` moves first, both commits are rebuilt on it. Landed as #584 on 2026-10-02; the maintainer tagged `20ab100` as `0.3.9` (lightweight) and published it as a GitHub pre-release. **Amended 2026-10-02 (the maintainer):** the deferred PRs land before consolidation PR-8 and the gait plan's code PRs. With the maintainer's answers of that day they go one at a time, in the order CU-10a, CU-13, CU-15 (reduced scope), CU-10b, ROW-4/6 (the notebook PR for the cleanup plan's decisions 4 and 6, which the maintainer took the same day), CU-6 and CU-17, then PR-8. CU-10 is split into CU-10a (the curriculum horizon fix chosen on 2026-09-27) and CU-10b (the stage body on a private helper shared with `train()`, with `eval_env_seed`, then D-D11's alignment for the curriculum as its own last commit), and CU-10's optional split of `train_base.py` into modules is not taken (the cleanup plan's §3.1 item 5). *2026-10-03 (the cleanup plan's §3.1 item 5): of the deferred PRs, CU-10a, CU-13, CU-15, CU-10b and ROW-4/6 have landed (#587 to #591), and CU-6 is carried out; CU-17 is the last of them, before PR-8.* | The 0.3.9 cut; PR-8 and PR-9 (their CU-11, CU-8 and CU-12 prerequisites are in the gate) |
| D-D22 | Should CI check the digest snapshot, and with which harness run on pull requests? | Taken 2026-09-29 (the cleanup plan's decision 16, option (a)): yes. The digest-snapshot output is committed as a golden and CI checks it with the full harness run on pull requests (not `--skip-behaviors`), in the plant-contract job; a PR that deliberately moves a digest updates the golden in its own diff. Its own PR builds the check, not CU-4, before CU-7b, CU-8a, CU-12 and CU-13. The full text is in BEHAVIOR_RECIPES_PLAN.md §6.2. As carried out by the ROW-16 PR: the golden is `configs/digest_snapshot.generated.txt` (848 lines at `a919985`), which the plant-contract job's step "Verify the digest snapshot golden" checks with the full run (`--block-optional-backends --check`) on every event; `--write` regenerates it. Landed as #571 on 2026-09-29. | The check's own PR; CU-7b, CU-8a, CU-12 and CU-13, which follow it |
| D-D23 | Is a stance certified on floor truth, episode by episode, and with which criteria? | Taken 2026-10-06 (the maintainer; the gait plan's GQ-6 (a) and GQ-7 (a), for stance only): yes. `stance_quality/v2` is registered beside v1 under `GATE_SCHEMA_VERSION` 1: each episode of the 40-episode certification panel (seeds 3042 + i) is clean when it reaches the horizon and every declared criterion holds on a finite floor-truth metric (`environments/shared/gait/`, `MEASUREMENT_VERSION = "floor-truth/v1"`), and the panel passes when the exact Clopper-Pearson bound on clean episodes clears `min_clean_stance_lcb` (0.80: 37/40) and the declared rails hold. The required keys add actuator saturation and the settle window's hop and impact (after a 0.1 s spawn grace) to the plan's list; flatness, coverage, foot-on-foot, phantom, non-foot and two statue-relative ratios are optional. The certificate is the post-stage stance report on the handoff pair, re-derived by the judge and at publication; the in-training manager refuses the kind. No stage adopts it with this decision, so no digest moves; each adoption is a gate revision of its own. The full text is in BEHAVIOR_RECIPES_PLAN.md §6.2; the evidence and validation are [investigations/STANCE_HACK_AUDIT_2026_10.md](investigations/STANCE_HACK_AUDIT_2026_10.md). | A stance stage's adoption of the kind (outside this sequence) |
| D-D24 | Do the T. rex plant revision and the T. rex stance's adoption of `stance_quality/v2` land now, retiring every T. rex certificate? | Taken 2026-10-06 (the maintainer; overtakes the gait plan's GQ-12 and PR-G8, which kept the trex stance on v1 until a retrain, and its §5.4 trex `TRUNK_FROM`): yes. Physics r7 → r8 (the hip-roll servos kp 150 → 600, forcerange ±480 N·m; `configs/plant_versions.toml` note 13; policy interface and visual layer unchanged), three stance reward kwargs inert at their defaults and set in the stance TOML, and the stance on `stance_quality/v2` with bars set on the r8 statue: clean on 40/40, the four audited r7 stance checkpoints on 0/40 each. The plant contract refuses every T. rex checkpoint and none can be widened; the stances, the recovery, both walkers and the hunt retrain. The full text is in BEHAVIOR_RECIPES_PLAN.md §6.2; the validation is [investigations/STANCE_HACK_AUDIT_2026_10.md](investigations/STANCE_HACK_AUDIT_2026_10.md) §7. | The T. rex stance retrain and every T. rex chain (outside this sequence) |
| D-D25 | Do the velociraptor plant revision, its SB3-only exit and the velociraptor stance's adoption of `stance_quality/v2` land now, retiring every velociraptor certificate? | Taken 2026-10-06 (the maintainer; overtakes KNOWN_ISSUES' deferral of the velociraptor plant review until the T-Rex clears stages 1–3, and the gait plan's §9, which left the velociraptor foot-sensor repair out): yes. Physics r2 → r3, policy interface r10 → r11, visual r3 → r4 (`configs/plant_versions.toml` note 14): leg springs anchored at the standing pose, a flat-footed keyframe with the gravity preload in its home ctrl (so the nominal leg servos hold the stance; the review's "re-size the leg actuators" is refuted by measurement), and metatarsus and digit-IV touch sensors summed per foot, which the frozen MJX registration cannot mirror, so the velociraptor declares itself SB3-only and its `mjx_config.py` is deleted. Eight stance reward kwargs (five terms) inert at their defaults and set in the stance TOML, and the stance on `stance_quality/v2` with bars set on the r3 statue: clean on 40/40, the two audited checkpoints of the `20260922_125248` stance on 0/40 each. The plant contract refuses every velociraptor checkpoint and none can be widened; the stance, the walker and the hunt retrain. The full text is in BEHAVIOR_RECIPES_PLAN.md §6.2; the validation is [investigations/STANCE_HACK_AUDIT_2026_10.md](investigations/STANCE_HACK_AUDIT_2026_10.md) §8. | The velociraptor stance retrain and every velociraptor chain (outside this sequence) |

## 7. Goal decisions G1–G4 (taken 2026-09-17)

The target behavior is: every species follows a direction on difficult terrain.
These four decisions fix what that means for the manifest and the gate; they are
recorded in BEHAVIOR_RECIPES_PLAN.md §6.2 beside the D-D series.

| Id | Decision | What it changes |
|---|---|---|
| G1 | Chain shape `walk -> follow_direction (full command set on flat ground) -> follow_direction_difficult_terrain` (the target deliverable); a commands-free `difficult_terrain` node stays as an optional diagnostic sibling; the recipe label `follow` resolves to the deepest deliverable; `follow_direction_speed` is folded into `follow_direction`; the final node is SB3-only (MJX fails closed on live commands and has no terrain). Amended by D-D17: the JAX/MJX runtime is retired (cleanup PR-B), so every node trains on SB3 only; the frozen MJX interface core only feeds the plant contract's digests and probe. | PR-11's node set becomes `follow_direction` and `follow_direction_difficult_terrain` (deliverables, `recipe = "follow"`) plus `difficult_terrain` (optional, `recipe = "terrain"`): three new stage files per species, not four; no `follow_direction_speed.toml`. All three new files carry `extends = "locomotion"` (D-D5); only `warm_start_from` differs (`locomotion` / `follow_direction`). The §2 table row for stage configuration reads accordingly. |
| G2 | Command set = heading, speed (half to full cruise), stops and restarts, switching every few seconds: the pilots' combined recipe. | PR-11's `follow_direction` carries the full command set from the start (the assessment had listed a heading-only first node and a separate speed node): `command_config` with `speed_range = [0.5, 1.0]` of `cruise_speed`, stops and restarts, and a switch interval of a few seconds; `follow_direction_difficult_terrain` inherits it. |
| G3 | Walker sessions start now on the current notebook, trex first (its r13 chain 20260914_123816 is selected automatically), the other five species one at a time, in parallel with the consolidation PRs. | Nothing in the PR order changes; it fixes the constraint every PR's "Breaks" line already honours: the notebook chain loop, `TRUNK_FROM = "auto"`, `WIDEN_FROM`, `RETRAIN_FROM` and the resume cell keep working at every step, because the sessions in NEXT_STEPS.md run on whatever `main` is at the time. Those sessions are what give PR-11's nodes their certified locomotion ancestors on the five species that lack one. **Amended by D-D14 and D-D15 (2026-09-24):** `WIDEN_FROM` leaves the notebook with PR-14a, once both widen sessions were decided; widening stays available on the command line. |
| G4 | The first `terrain_command/v1` gate adopts the pilots' certificate thresholds: 20 episodes per terrain family, 20 s minimum horizon, survival LCB 0.80, success LCB 0.60, tracking and settle fractions 0.60; tightened after the first certified species. | PR-13's `[curriculum]` threshold block is fixed at these values, carried once (never 66 copies) from configs/behavior_certification.toml before that file is deleted; tightening later is a gate-digest change (rule 7), so a node certified under the first thresholds is re-judged, not silently reused, after the change. |

## 8. Risks and open questions

Carried from the review, with the 2026-09-17 additions.

- The stop-gap in PR-1 (#542) is the only change that protected the maintainer's
  next Colab session; everything else waited for the hold, which lifted on
  2026-09-20. Since #543
  canonical chains never consult the library; `SOURCE_SELECTION = "auto"`
  applied only to the direction/terrain path (`train_behaviors --auto-source`;
  both left with PR-5 on 2026-09-20), which would have copied a library version
  into the pilot bundle's `certified_inputs/` (certified_library.py:613 at
  723f58f; that tree wrote bundle.json, never `artifact_manifest.json`); the
  survey did not inspect the library directory and nothing relies on one, so
  that half is gone. What
  canonical chains did until PR-4 (2026-09-20, the session branch) was copy a cross-run trunk ancestor's
  bundle into `certified_inputs/` through `copy_canonical_ancestor` (cell 22;
  certified_canonical.py:882), and `artifact_manifest.json` hashes those copies
  into every bundle write (manifest.py:89, 158).
- Does a widened stance reproduce its panel under r13? Answered 2026-09-20/21,
  yes to every printed digit, by NEXT_STEPS.md sessions 1 (trex seed 44, gap 2:
  3408.3 ± 88.5, duty 0.0069 / UCB 0.0117) and 2 (compsognathus seed 42, gap
  1: 2801.6 ± 51.2, duty 0.0131 / UCB 0.0141); the SEED = 45 fallback was not
  needed.
- The `ConstantSchedule` `custom_objects` guard in `_load_ppo`
  (behavior_checkpoint.py:108-121, for cloudpickled py3.13 schedule bytecode) is
  deleted with PR-12. Settled 2026-09-19: the canonical `alg_cls.load` path had
  never crossed an interpreter boundary, the first cross-interpreter load (the
  widen tool's self-verification of the r11 parent on the Python 3.13 image)
  killed the kernel, and every load now goes through
  `policy_loading.load_sb3_model`, which makes the guard redundant (KNOWN_ISSUES,
  "SB3 archives are bound to the interpreter that saved them").
- Not verified during the review (flagged, unverified): (i) the two dated
  task-fingerprint valves (`allow_unfingerprinted=True`, train_base.py:593-604;
  the schema-v1 valve in `validate_recorded_task`) may already be dead because
  the plant check runs first on r13 species; deleting them is a ~60-line
  follow-up once the r11 parents are widened or superseded (both are: seed
  44's ran 2026-09-20 as `20260920_010912`, seed 42's was superseded by
  `20260914_123816`); (ii) `EpisodeManifestRecorder`
  (train_behaviors.py:165-176) becomes an info key for the existing
  Monitor/DiagnosticsCallback under train_base, so per-episode terrain manifests
  must be checked to survive PR-12; (iii) `canonical_env_parameters`
  (behavior_env.py:44-55) duplicates `_effective_env_kwargs`; dies with PR-9 but
  its use for the allowed-`[env]`-keys check in `read_recipe` must be re-homed
  in `load_stage_config` (PR-9/PR-11); (iv) the four provenance records of
  seed/parent facts (run block `LOAD_LINEAGE_KEYS`, task lineage,
  `mesozoic_canonical_training`, the four `mesozoic_behavior_training_*` stamps)
  collapse to the first two after PR-4/PR-12, but the task lineage may need
  `parent_normalization_sha256` added (~15 lines) once the pilot stamps go.
- Distribution change in PR-8: single-template recipes move from a Bernoulli
  `flat_probability` draw to balanced blocks; acceptable for pilots
  (evaluation-only, D-D9), stated here as the decision record requires.
- PR-9 touches five species constructors, `MJXEnvConfig` and the fingerprint
  carve-out; the acceptance test is that no committed `task_sha256` moves
  (test_phase_c_interface.py:348) and that `plant_contract --check` reports no
  interface change, run on every species, not only trex. Since cleanup PR-B
  (D-D17, 2026-09-28) the `MJXEnvConfig` dataclass is gone: PR-9 edits the five
  constructors and the carve-out only, and the frozen MJX interface core stays
  untouched under the same `plant_contract --check`.
- MJX reward kernels stay world-z after PR-7 (MJX has no terrain and fails
  closed on live command modes); PR-7 notes the divergence in the comment block
  above mjx_env's `_height_strike`.
  Under G1 the final node is SB3-only for the same reason. Cleanup PR-B (D-D17,
  2026-09-28) deleted the MJX reward kernels with the rest of the JAX/MJX
  runtime, so the divergence and PR-7's comment above `_height_strike` are gone
  (G1 amended).
- CI coverage `fail_under=70` will need re-measuring after the large deletions
  in PR-12/PR-13 (after PR-3 it read 90 percent on #546's CI runs); the
  excluded trainings mostly cover code that is deleted.
- Test-to-test coupling must be untangled in order:
  environments/trex/tests/test_behavior_training.py imports `CommandEnv` from
  test_behavior_checkpoint.py (:384) (resolved: PR-7 moved its command-line
  cases into test_behavior_checkpoint.py beside `CommandEnv`);
  test_behavior_publication.py imports `_identity`/`_report` from
  test_behavior_certification.py (:14) (resolved: the file left with PR-5 on
  2026-09-20).
- jax_training.ipynb carries the same Drive-mount block and `_ACTIVE_RUN_ID`
  memo (:250, :296); if the D-D7 package move or the memo removal is taken, take
  it for both notebooks so the two Colab drivers do not diverge on the same
  footgun. PR-14a keeps the memo and makes `RUN_ID` a configuration-cell knob in
  the SB3 notebook only; the JAX notebook keeps its storage-cell `RUN_ID`.
  Moot since cleanup PR-B (D-D17, 2026-09-28): the JAX notebook is deleted, so
  only the SB3 notebook carries the memo.
- The plan's per-event `command_tracking/v1` statistic and the
  worst-of-heading-bins floor have no implementation anywhere yet; D-D6's second
  kind is new work outside this consolidation.
- Uncertainty on the totals: the largest single figure (PR-12, about -5,300)
  depends on D-D5 (self-contained TOMLs would add ~700 lines back) and on how
  much of test_behavior_species_training.py survives as the real-PPO smoke; the
  band 9,000-12,000 reflects that.
- Velociraptor:
  [investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md)
  (2026-07-20) left a pending Commit B validation; the fresh r10 stance run in
  NEXT_STEPS.md session 3 is the first data since.
- The r13 trex run's run-level summary.json, provenance.json and
  artifact_manifest.json predate its locomotion verdict (written 2026-09-15
  02:08 UTC, verdict 11:39 UTC); reuse reads per-node files so nothing breaks,
  but the run-level records under-report the run and cannot be refreshed in
  place: the bundle is `complete` under a stance target, a re-entry that
  reuses every node writes no bundle, and a direct save is refused because
  `03_locomotion/` appeared after publication. The bug behind it (a node
  trained in place into a complete bundle failed its bundle write after
  training) is fixed by PR-14a's refusal before training (D-D15; the
  alternative, a bundle writer that extends a complete bundle by new nodes,
  would reopen the immutability contract and is not taken); the run's
  records stay as they are.
- Phase B items the maintainer deferred on 2026-09-13 stay deferred
  (BEHAVIOR_RECIPES_PLAN.md §10); nothing in this sequence reopens them.
