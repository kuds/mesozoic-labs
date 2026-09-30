# Cleanup and backend retirement plan (2026-09)

**Status**: living plan, updated 2026-09-30. `main` = `ae4d651` (#574, CU-9, merged 2026-09-30 15:39 UTC). Written at `be63a58` (#559, the
#558 follow-up, merged 2026-09-26 03:42 UTC; commits `ee91f51`, `634e2b3`, `ad4e621`). #558 (notebook safety, D-D16) merged 2026-09-25 22:44 UTC as `f850815`.
Line numbers are at `f850815` unless marked "at the follow-up", which equals `be63a58` for every file #559 touched
(`sb3_training.ipynb`, `curriculum/__init__.py`, `curriculum/checkpoints.py`, `test_curriculum_checkpoints.py`,
`test_sb3_notebook_pins.py`, `CHANGELOG.md`, `docs/NEXT_STEPS.md`, `docs/CONSOLIDATION_PLAN_2026_09.md`,
`docs/BEHAVIOR_RECIPES_PLAN.md`, `docs/README.md` and `website/docs/training/recipes.md`); every other file is
byte-identical at `f850815` and `be63a58`. Line numbers drift with every merge, so re-read before editing. This plan
landed as #560 (merged 2026-09-26 05:03 UTC as `8e03483`). On 2026-09-26 the maintainer took three of §2's decisions
as D-D17 (row 1, widened), D-D18 (row 8) and D-D19 (row 3), and closed #527 and #498 (row 19); CU-1 is the first PR
after it (§3.1). CU-1 landed as #561 the same day; the maintainer then took decision 7 as D-D20, carried out by CU-3,
and moved the Vertex route and GCS upload out of PR-A into a PR of their own, PR-A2, with an end-to-end test of the
command-line curriculum path (D-D17 amended; §4.5). CU-3 landed as #562 the same day (§3.1). The release cut
(D-D19) landed as #563 on 2026-09-27, and the maintainer tagged its first commit, `afad625`, as `0.3.8` (a
lightweight tag, published as a GitHub pre-release). The same day the maintainer settled §2 row 2 as (c), no
archive tags, and took row 20 as D-D21: 0.3.9 is the clean, refactored base release, cut once its gate has landed.
PR-A landed as #564 the same day, PR-A2 as #565 on 2026-09-28 and PR-B as #566 the same day, which completes D-D17's
removals (§3.1). The gait audit and its plan (PR-G0, docs only and outside the cleanup; §1 item 2) landed as #567
the same day, and CU-2 as #568 on 2026-09-29 (§3.1). The same day the maintainer took §2 row 16 as D-D22, accepted
splitting CU-7, CU-8, CU-14 and CU-16 into parts (row 20) and the order of the rest of the gate (§3.1 item 4), and
settled row 10 as (c) first; CU-4 landed as #569 the same day, and CU-14b, decision 10 (c), as #570 (§3.1).
ROW-16, the digest-snapshot check of D-D22, landed as #571 the same day (§3.1 item 4), and CU-7a, the first
part of CU-7, as #572 on 2026-09-30 (§3.1 item 4, §3.2). CU-11, the reward, info and termination
golden and the last PR of wave 1, landed as #573 on 2026-09-30, which completes wave 1 (§3.1 item 4, §3.2). CU-9,
coverage of the certification code and the first PR of wave 2, landed as #574 on 2026-09-30 (§3.1 item 4, §3.2).
CU-5, the notebook text and dead parameters, is carried out (§3.1 item 4, §3.2).

## How to use this document

This plan lists the cleanup that remains, the order to do it in, and what the maintainer must decide first. It absorbs
three working notes from 2026-09-25 that are not in the repository: the cleanup survey (waves C1–C21), the plan to
retire the secondary backends, and the direction/terrain readiness review. Each note had a critic; where a note and
its critic disagree, this plan follows the critic. Companions: [NEXT_STEPS.md](NEXT_STEPS.md) (training sessions,
Drive state), [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) (PR-7..PR-15, the D-D series),
[BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) (design of record; §6.1 holds the D-A/D-B/D-C rows and §6.2 the
D-D and G rows, where new D-D rows go) and [KNOWN_ISSUES.md](KNOWN_ISSUES.md) (verified, unfixed problems only). Read
§2 (decisions) first, then §3 (PR order); §3.5 lists the KNOWN_ISSUES entries added with this plan and the PR that
closes each. §4 is detailed enough to carry out the retirement; §5–§7 hold the evidence, the lessons and the
do-not-do list.
Notebook paths here omit the `notebooks/` prefix on purpose: `test_species_catalog.py:618-636` fails any `docs/*.md`
that names a missing `notebooks/<name>.ipynb`, and PR-A and PR-B each delete a notebook.

## 1. Bottom line

1. **What is left.** The #558 follow-up landed as #559 on 2026-09-26. The backend retirement has landed (PR-A as
   #564, PR-A2 as #565 and PR-B as #566, §4.6). What remains is eleven smaller PRs (CU-5..CU-17 less CU-9 and CU-11,
   §3); the CI-signal PR (CU-1, D-D18) landed as #561, CU-3 (D-D20) as #562, the release cut (D-D19) as #563, CU-2
   as #568, CU-4 as #569, CU-14b as #570, ROW-16 (D-D22, outside the gate) as #571, CU-7a as #572, CU-11 as #573 and
   CU-9 as #574. D-D21 (§2 row 20) gates the 0.3.9 release on thirteen of these PRs: PR-A, PR-A2, PR-B, CU-2, CU-4,
   CU-5, CU-7, CU-8, CU-9, CU-11, CU-12, CU-14 and CU-16, of which six remain now that the retirement, CU-2 (#568,
   2026-09-29), CU-4 (#569, 2026-09-29), CU-11 (#573, 2026-09-30) and CU-9 (#574, 2026-09-30) have landed: CU-5,
   CU-7, CU-8, CU-12, CU-14 and CU-16 (on 2026-09-29 the maintainer split CU-7, CU-8, CU-14 and CU-16 into parts
   that the gate's names cover, §2 row 20 and §3.1 item 4; CU-14b, the first part of CU-14, landed as #570,
   2026-09-29; CU-7a, the first part of CU-7, landed as #572, 2026-09-30; CU-11 landed as #573, 2026-09-30; CU-9
   landed as #574, 2026-09-30; CU-5 is carried out); the other five CUs are deferred, not dropped. The retirement
   makes four of the survey's 21 waves wholly moot, most of C15 and half of C1. It also deletes five of the survey's
   nine live defects along with their code, and #558 already fixed two more; CU-2 fixed the render crash (§5.3
   defect 2; landed as #568 on 2026-09-29).
2. **Order.** CU-1 (mypy with SB3, readable CI logs; D-D18) came first, as #561, and CU-3 (atomic run-tree records
   and checkpoint pairs; D-D20) second, as #562, and the CHANGELOG release cut (D-D19) third, as #563. PR-A (Ray Tune, the Vertex AI
   tuning sweeps and mjlab; its acceptance runs the digest-snapshot harness this plan adds, §4.4) came fourth, as #564, and PR-A2 (the
   single-job Vertex route and GCS upload, which D-D17 also retires) fifth, as #565, and PR-B (JAX/MJX, keeping a frozen
   interface core) sixth, as #566. Next comes the rest of D-D21's 0.3.9 gate (§2 row 20), among it the golden-trace (CU-11),
   one-derivation (CU-8) and env-dedup (CU-12) PRs that consolidation PR-8 and PR-9 need, the CI structure (CU-14)
   and docs correctness (CU-16), and then the 0.3.9 cut. The deferred PRs follow at their §3.2 points: `extends`
   (CU-13) before PR-11, and the docs shrink (CU-17) last, before PR-15. The gait audit of 2026-09-28
   ([investigations/GAIT_AUDIT_2026_09.md](investigations/GAIT_AUDIT_2026_09.md)) found that three of the five
   certified walkers hop, and its plan ([GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md), whose
   decisions GQ-1..GQ-18 are all open) places the gait-check code after the 0.3.9 cut, as the maintainer asked;
   nothing in D-D21's gate changes. Both landed with PR-G0, the docs-only records PR, as #567 on 2026-09-28.
   CU-2 landed as #568 on 2026-09-29, and the same day the maintainer accepted the order of the rest of the gate,
   in four waves that open with the digest-snapshot check (D-D22, §2 row 16), CU-4, CU-7a, CU-11 and CU-14b
   (§3.1 item 4). CU-4, CU-14b, the digest-snapshot check (ROW-16), CU-7a and CU-11 have landed (#569-#573), which
   completes wave 1; CU-9, the first of wave 2, landed as #574 on 2026-09-30, and CU-5, the second, is carried out.
3. **Measured payoff.** PR-A and PR-B delete 72 whole files and 28,919 lines (PR-A's and PR-B's, measured before
   D-D17 was taken; the Vertex route and GCS upload leave in PR-A2, which derives its own). That covers about 17,459 of the 71,682
   non-test library lines (24%) and about 12,090 test lines. As carried out: PR-A deleted 42 files and 13,222 lines,
   PR-A2 4 files and 957 lines, and PR-B 30 files and 15,665 lines (whole files; PR-B's whole diff is +489 / −21,178).
4. **CI savings (estimated from 18 runs, #1210–#1227).** Jobs: 23 → 22. Runner time per PR/push run: 165.7 → 116.7 min
   (−30%, about 1,900 runner-minutes a week). Median wall time: 52.3 → 46.0 min. The nightly run is unchanged. Nearly
   all of this comes from PR-B; PR-A (Ray, Vertex and mjlab) saves tens of seconds.
5. **The larger payoff is no longer mirroring core changes into untrained code.** 53–54 of the 128 first-parent units
   on `main` since 2026-07-01 touched backend files. Four mirrored copies had already drifted into live defects.
6. **The surviving cleanup is mostly about correctness, not size.** It removes roughly 830–900 lines of code, tests
   and notebook (about 530–750 if the optional CU-15 reader move is done), and about 900 lines of docs (survey
   estimates, re-summed without the moot items). Its most important items are the `render_mode='human'` crash, atomic
   final and best checkpoint pairs, and a golden trace for reward and termination, which no digest covered until CU-11 (landed as #573, 2026-09-30: the digest
   snapshot's `reward` section, §4.4).
7. **The overriding constraint: no digest may move.** The policy-interface digests of trex, velociraptor,
   brachiosaurus and dibothrosuchus hash MJX source tokens and run an MJX probe. A frozen MJX interface core (473 lines
   planned, 549 as PR-B built it with its FROZEN notices; §4.3) therefore stays, unedited, until each of those
   species reaches its next deliberate policy-interface revision.
8. **No certified run is affected.** With the core kept, two independent prototypes produced byte-identical snapshots
   of every plant, stage, behavior and recovery digest. Every certified run on Drive stays valid, including the 18
   certified stage/digest prefixes in [NEXT_STEPS.md](NEXT_STEPS.md).
9. **Archive points settled (§2 row 2, 2026-09-27): no archive tags.** The retired code stays reachable at the
   `0.3.8` tag (`afad625`) and in git history (§4.8). D-D17 is taken (2026-09-26), and the single-job Vertex route and GCS upload go with PR-A2 (§2, row 1; §4.5; carried out by PR-A2, 2026-09-27). The digest-snapshot harness is already in the repository
   (added with this plan, 2026-09-26; §4.4), and its output is a committed golden that CI checks (D-D22, carried
   out by ROW-16, landed as #571). The frozen-core reference copies are not: PR-B rebuilds them from
   `f850815` (§4.2; carried out by PR-B, 2026-09-28, which pins them with a test, §4.3).
10. **Direction/terrain is now gated by physics, not cleanup.** All three certified walkers survive the plane in every
    episode, but only 1 of 39 flat-heightfield episodes reached full horizon (§5.1). The heightfield contact
    investigation and the speed and map re-derivation must come before PR-11.

## 2. Decisions needed before continuing

D-D17 was reserved for the retirement, because PR-A's text must cite it. The other rows take D-D18 onward, in the
order taken, even if one is taken before PR-A merges. On 2026-09-26 the maintainer took row 1 as D-D17 (amended the same
day to split the Vertex route and GCS upload into PR-A2), row 8 as D-D18, row 3 as D-D19 and row 7 as D-D20, and
recorded row 19 (closing the stale PRs) as an operational choice rather than a D-D row; each outcome is appended to its
row below. On 2026-09-27 the maintainer settled row 2 as (c), no archive tags, and took row 20 as D-D21. On
2026-09-29 the maintainer took row 16 as D-D22 and accepted splitting four of D-D21's gate PRs (row 20), and CU-4
appended its row 9 amendment. The same day the maintainer settled row 10 as (c) first, without a D-D id, as row 2
was; CU-14b carried it out (#570). ROW-16 carried out row 16 (D-D22; #571). Record every row append-only in
[BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2 and in the consolidation plan's decision table.

| # | Decision | Options | Recommendation (why) | Blocks |
|---|---|---|---|---|
| 1 | **Proposed D-D17: retire the backends.** The maintainer's direction is to remove Ray Tune, the Vertex AI hyperparameter-tuning sweeps, mjlab and JAX/MJX now, and add them back once every species and behavior is learned. **Open scope question** (removal critic item 12): does the single-job Vertex route stay? That route is `scripts/setup_vertex_ai.sh` (184 lines; submits SB3 `train_sb3.py train`/`curriculum` jobs at :127-162; one sweep hint at :183) and the `Dockerfile` (44 lines; `.[train,viz,gcp]` at :31). | (a) D-D17 as drafted (§4.5), keeping the route. (b) Also delete `setup_vertex_ai.sh`, the `Dockerfile`, `.dockerignore`, the kept parts of `vertex-ai.md`, and `google-cloud-aiplatform` from `[gcp]`. | **(a).** The route runs SB3 only and references no sweep code, and removing it later is a separate, reversible choice. It still needs the maintainer's explicit yes, because the original scope named both files. Also confirm the appended note to [investigations/TREX_REVIEW_2026_07.md](investigations/TREX_REVIEW_2026_07.md) (§4.5). **Taken 2026-09-26 as D-D17, wider than (b):** the maintainer also retires GCS artifact upload (`curriculum --gcs-bucket` / `--gcs-project`, the helpers they reach and the whole `[gcp]` extra), so PR-A's delete list grows beyond §4.5's (see its note). **Split the same day:** those two leave in a PR of their own, PR-A2, with an end-to-end test of the command-line curriculum path; this plan lands it after PR-A (D-D17 amended). **Carried out in part by PR-A (2026-09-27; §4.5):** Ray Tune, the Vertex AI tuning sweeps and mjlab removed (42 files, 13,222 lines); D-A12, D-A15, D-B1, D-B12 and D-D11 amended and A6 superseded; no digest moves. PR-A2 and PR-B carry out the rest. **PR-A landed as #564 on 2026-09-27.** **Carried out by PR-A2 (2026-09-27; §4.5):** the single-job Vertex AI route and GCS artifact upload removed (4 files and 957 lines, the upload code, the `--gcs-*` flags and `[gcp]`), with an end-to-end test of the command-line curriculum path; no digest moves. PR-B carries out the rest. **PR-A2 landed as #565 on 2026-09-28.** **Carried out by PR-B (2026-09-28; §4.6):** the JAX/MJX runtime removed (30 whole files and 15,665 lines: the 13 `jax_*` modules, 15 JAX/MJX test files, the notebook and the JAX guide; also the stage writer `save_jax_stage_artifacts`, `validate_mjx_environment_plant`, the `[jax]` stage tables, the `jax`/`jax-cpu` extras and the `test-jax-cpu` job) behind the 549-line frozen MJX interface core and its pin test; A7 narrowed and D-A5, D-B13, D-C3, D-C4, D-C7, D-C16 and G1 amended; no digest moves. This completes D-D17's removals. **PR-B landed as #566 on 2026-09-28.** | PR-A and PR-B. Their FROZEN docstrings, amendments and CHANGELOG entries cite D-D17. PR-A2 too (split 2026-09-26). |
| 2 | **Archive points.** | (a) One tag at PR-A's first parent. It misses any JAX change that lands between the two merges. (b) Two annotated tags: `archive/secondary-backends-2026-09` at PR-A's first parent and one at PR-B's first parent (e.g. `archive/jax-mjx-2026-09`), each pushed before its PR merges, with both SHAs in D-D17 (critic item 5). (c) No tags; merge commits keep the SHAs reachable. | **(b).** It is cheap, and each half can be recovered from a named point. The remote had 0 tags on 2026-09-25. Do not reuse the stale `JAX` branch (`f900d3a`, 2026-02-03), which is not an ancestor of `main`. **Decided 2026-09-27: (c), no archive tags.** The project is in alpha, and the maintainer had already tagged the tree before the retirement as the release `0.3.8` (`afad625`, row 3). Every file PR-A deletes is byte-identical there, and anything that changes later stays reachable through PR-A's and PR-B's merge commits. §4.8 item 1 no longer applies. | The PR-A and PR-B merges. Maintainer action. |
| 3 | **Cut the CHANGELOG release?** `[Unreleased] (v0.3.8)` spans `CHANGELOG.md:8-3805`: 3,798 of 4,009 lines (3,860 of 4,071 at the follow-up). Two more undated `[Unreleased]` headings sit at :3806 and :3934. The last dated release is 0.2.0 (2026-02-09). Since `c0e9b52` (2026-08-02), `.devN` in `pyproject.toml:7` and the heading move together, with no tags. | (a) Before PR-A: date the three headings, open an empty `[Unreleased]`, bump to `0.3.9.dev0` (or `0.4.0.dev0` if the terrain path is ROADMAP's v0.4.0), and tag `v0.3.8`. (b) The same, without a release tag. (c) Keep growing the section. | **(a) or (b), before PR-A**, so the retirement's Removed and Migration entries head a short section. A release tag is the maintainer's call; archive tags are not release tags. The version is recorded in `stage_config.json` (`config.py:788`), in the HPT metrics payload (`train_base.py:1503`) and in `provenance.json`'s `dependency_versions` (`"mesozoic_labs"`, `result_bundle/constants.py:55`). It enters no digest; a resume after the bump only records it as environment drift, as a new `repository_commit` already does. **Taken 2026-09-26 as D-D19: (a), with `0.3.9.dev0` next; the maintainer tags `v0.3.8`.** **Carried out by the release PR (§3.1).** **Landed as #563 on 2026-09-27** (merged 03:02 UTC as `04d107a`); the maintainer tagged `afad625` as `0.3.8`, a lightweight tag without the `v` prefix, and published it as a GitHub pre-release. | Nothing hard. It only decides where PR-A's entries go. |
| 4 | **Persist the resolved trunk run on disk.** The chain loop never follows the run's own `ancestors/` records, so a resume must pin `TRUNK_FROM` to the trunk that the interrupted session's resolve cell *printed*. If that output is lost, the operator has to hunt through the ancestor records (`CHANGELOG.md:1712-1726` at the follow-up; round-3 check). | (a) The resolve cell writes the resolved trunk into a run-level sidecar, and the RESUME cell and the recipe read it. (b) Keep the printed output plus the recipe. (c) Put it in `provenance.json`. | **(a), as a small notebook PR.** First check how the run manifest treats a new run-level file: `result_bundle/manifest.py` hashes the run tree, and a `complete` bundle is immutable. Not (c): provenance keys drive the drift records and the audit. | Nothing. It removes an operator error mode (KNOWN_ISSUES, §3.5). |
| 5 | **Library guard against `resume_same_stage` into a judged directory.** `config.refuse_occupied_stage_dir` (`config.py:403-427`) lets any same-stage resume through, even with `gate_verdict.json` present. The SB3 notebook has guarded against this since D-D16; the CLI (`--load-mode resume_same_stage`) has not. | (a) Refuse when `gate_verdict.json` exists. This amends D-A20 and covers the CLI. (b) Keep the notebook-only guard. | **(a)**, once you have confirmed that no legitimate writer resumes into a judged directory: `train_curriculum`'s in-training verdict (D-A5), backfill, widen. This is not the do-not-do short-budget guard (§7). | Nothing on the terrain path. It closes a CLI hole (KNOWN_ISSUES, §3.5). |
| 6 | **Judge an unjudged `RUN_DIR` node before any trunk reuse.** When a trunk certifies a node, `RUN_DIR`'s own copy is bypassed even if it holds an intact final pair and no verdict. One case is a node that `RETRAIN_FROM` covered and that was then resumed. The follow-up documents a manual route: set `BEHAVIOR=<node>`, then start a fresh `RUN_ID` trunked from this run. D-C13's `refuse_trunk_over_unjudged_widened_root` (`result_bundle/reentry.py:255`) covers only a widened root. | (a) Generalise that refusal to any such node, with the `BEHAVIOR=<node>` remedy. Not `TRUNK_FROM = ""`, which retrains ancestors the run only reused (review 2). (b) The chain loop judges such a node before consulting the trunk. This amends the loop order fixed by D-A17/D-C13 and the BEHAVIOR_RECIPES_PLAN §4.7 pins. (c) Keep the manual route. | **(b) if the maintainer accepts amending the loop order; otherwise (a).** (b) removes a two-session manual route and makes D-C13 a special case. | Land before PR-13 edits the chain-loop cell. The KNOWN_ISSUES trunk entry covers this case (§3.5). |
| 7 | **Stage the final and best checkpoint pairs atomically.** Only the periodic pairs are staged and published atomically (`train_base.py:727-760`). `best_model` (:691, :710), `robust_best_model` (:724) and the final pair (`_save_final_and_sync_tb`, :918-932) are written straight to the mount. Since the follow-up, the RESUME cell and the chain loop check the final pair with `checkpoint_pair_problem`; nothing checks the handoff pairs. If a reclaim cuts a best pair short while the final pair is intact, RESUME treats the node as finished and JUDGE fails to load the handoff. | (a) Stage them like the periodic pairs, in CU-3. (b) Extend `checkpoint_pair_problem` to the handoff pair. (c) Only a KNOWN_ISSUES entry. | **(a).** The failure was reproduced 2026-09-26 at the follow-up: with an intact final pair and `robust_best_model.zip` cut in half, the RESUME cell prints "Nothing to resume", `select_handoff_checkpoint` still returns the pair, and loading it raises. It is recorded in KNOWN_ISSUES until CU-3 lands (§3.5). File bytes and names are unchanged, so no digest moves. **Taken 2026-09-26 as D-D20 and carried out by CU-3 (§3.2).** **Landed as #562 on 2026-09-26**, which deleted the KNOWN_ISSUES entry (§3.5). | Nothing. It lowers reclaim risk on Colab. |
| 8 | **mypy with SB3 in CI.** The lint job installs only `ruff mypy gymnasium numpy` (`python-ci.yml:82`; mypy at :90-91) and reports no issues. With SB3 installed, mypy finds 21 errors (§5.5; KNOWN_ISSUES, §3.5). | (a) Add a mypy step to `test-sb3`, pinning `stable-baselines3==2.9.0` (as the notebook's install does) and `torch==2.13.0` (as `python-ci.yml:266` already does). (b) Install `.[train]` with CPU torch in the lint job. (c) Keep the hand-kept baseline. | **(a).** Before adding the step, run mypy once in `test-sb3`'s own environment (SB3 2.9.0, torch 2.13.0 CPU, ray and wandb installed, no JAX; `python-ci.yml:266-267`) and fix what it reports. The 21 were measured with mypy 2.3.1, torch 2.14.0+cpu, JAX installed and ray/wandb absent, so the count in the job's environment is unmeasured; five of the errors come from torch's `Tensor` typing. Pin mypy in the step and in the lint job (`python-ci.yml:82` installs it unpinned), and give pre-commit the same version (`.pre-commit-config.yaml:10` pins v1.15.0). The pins stop an SB3, torch or mypy release from turning an unrelated PR red. `test-sb3` is off the critical path until PR-B. **Taken 2026-09-26 as D-D18 and carried out by CU-1 (§3.2).** Measured in `test-sb3`'s environment on `8e03483` (Python 3.12, numpy 2.5.3, torch 2.13.0+cpu, ray 2.58.0, wandb, no JAX): 23 errors in 8 files, the 21 plus `harnesses/freeze_recovery_gate.py:469` (numpy 2.5's `ndarray` typing; numpy 2.5 needs Python 3.12, so the 3.11 lint job resolves 2.4.6) and `scripts/sweep/ray_orchestration.py:247` (seen only with ray installed). **Landed as #561 on 2026-09-26**; in CI the step printed "Success: no issues found in 359 source files". | PR-10 (torch-facing code). It also gives PR-A and PR-B an enforced mypy check. |
| 9 | **Amendments the surviving waves need.** CU-6 moves a slice of D-D7 into the package. CU-13 extends D-D5's `extends` to recovery ← stance. CU-4 deletes the pin that `CONSOLIDATION_PLAN_2026_09.md:247-249` says STAYS. | Append the amendments, or drop those slices. | **Append each amendment when its PR opens.** The consolidation plan's §8 risk "take it for both notebooks" (`CONSOLIDATION_PLAN_2026_09.md:1318-1322`) becomes moot after PR-B. **CU-4 appended its amendment on 2026-09-29:** a dated note on the consolidation plan's landed PR-4 body says that `test_the_library_rule_the_notebook_relies_on`, the pin that body says STAYS, is deleted as a duplicate of `test_ancestors.py` (invariant 6's own pin), and a note on PR-15 says that its `notebook_cells` slice left with CU-4 (§3.2). It is not a D-D row. | CU-4, CU-6, CU-13. |
| 10 | **After PR-B, `test-sb3` is the critical path.** `test_compsognathus_training.py` takes 13.8–21.6 min of the integration step. The survey's do-not-do against gating its 12 real-training parametrisations rested on the JAX job setting the wall time, and that ends with PR-B. Measured (survey, locally, 12 passed in 618 s): the six compsognathus_robot parametrisations take 484 s (78%), because `current_plant_identity("compsognathus_robot")` rebuilds the robot plant (147 geoms) in 7.8–9.5 s per call, several times per test; compsognathus takes 0.4–0.5 s per call. Runner variance is about ±40%. | (a) Gate them behind the depth switch; nightly and `full-ci` still run them. (b) Keep them on every PR. (c) Cache the plant identity per process and species (no digest change; it also speeds up behavior env construction, where the robot's identity rebuild costs about 10 s). | **(c) first, then decide (a) after PR-B** from CU-1's `--durations` output. If (a), keep compsognathus rather than the robot on pull requests. **Settled 2026-09-29: (c) first, carried out by CU-14b (§3.2, CU-14 row).** The maintainer accepted the recommended order of the gate, whose first CU-14 part is CU-14b (row 20, §3.1 item 4); (a) is decided in CU-14a from the durations CU-14b's CI produces. It is settled without a D-D id, as row 2 was. | CU-14. |
| 11 | **Heightfield contact parity.** All three certified walkers survive the plane 23/23 and fail on a flat heightfield: full horizon 1/13 for trex, 0/13 for velociraptor and 0/13 for compsognathus (§5.1). Finer cells did not help the trex walker (0/7 at 100 mm), though they helped its statue. | (a) Investigate first, as a dated note under `docs/investigations/`. Cover plane vs heightfield contact (solref/solimp, margins, collision pairs), the terrain settle and spawn clearance (velociraptor's authored −44.6 mm toe penetration, `behavior_env.py:403`), and the compsognathus contact flicker. (b) Proceed and treat terrain later. (c) Train terrain as it is. | **(a).** It can run in parallel with PR-8..PR-10, but it must finish before PR-11, which copies the terrain keys verbatim (`CONSOLIDATION_PLAN_2026_09.md:655-656`). | PR-11's terrain nodes, any terrain pilot, and PR-13's terrain thresholds. |
| 12 | **Split PR-13.** The plan requires PR-11 and PR-12 before PR-13 (`CONSOLIDATION_PLAN_2026_09.md:793`). Its gate-registration half appears not to need PR-12's deletions (inferred from the plan text). | (a) Land the gate-registration half right after PR-11. (b) Keep PR-8 → PR-9 → PR-10 → PR-11 → PR-12 → PR-13. | **(a).** The notebook chain then halts at `follow_direction` for less time. The PR-13 row needs an amendment. | PR-11 scope. |
| 13 | **Recipe speeds and map sizes.** Cruise is set to the locomotion gate minimum, not to the learned gait. Velociraptor: 2.0 m/s against 3.17–3.30 certified (3.5–3.7 measured on the recipes). Compsognathus: 0.08 against 0.34 (4.3×; 0.38 measured). Trex: 1.05 fits the seed-42 walker (1.07) but not seed 44 (1.57). Commands span half to full cruise, so at half cruise the gap is 3.3× (velociraptor) and 8.5× (compsognathus), and velociraptor's tracking reward at gait speed is 4e-4 to 6e-5. Maps are sized for 25 s at cruise, so a walker at its own speed leaves after about 17 s (velociraptor) or 10–12 s (compsognathus), which counts as a non-survival; velociraptor left the map in 9 of 13 flat-heightfield episodes. | (a) Re-derive `cruise_speed`, the scales, extent, grid, `course_distance` and apron from each certified walker. Compsognathus at ~0.34 m/s needs an extent of ~9.1 m. Velociraptor at ~3.2 m/s needs an extent above ~81 m and at least ~1,081 columns. (b) Deliberately command below the gait. | **(a)** for each species that has a walker. Decide the robot once its walker is certified. | PR-11, which copies the 19 per-species scale values verbatim. |
| 14 | **Compsognathus terrain feasibility.** As written, compsognathus_robot cannot pass the certificate on the commanded terrain recipes (`follow_direction_difficult_terrain`, `combined_terrain`, `combined_mixed_terrain`) even with perfect tracking: on `follow_direction_difficult_terrain` it clears the departure rule in 1/5/4/4 of 20 episodes per non-flat family, where 17 are needed for the success bound (10/11/17/15 at the top of the velocity tolerance). Compsognathus passes only if it runs at the top of the ±0.016 m/s tolerance (18/18/19/18; `combined_terrain` 18/20), so it is a risk, not a blocker; `combined_mixed_terrain` fails either way (mixed 14/20). The command-free recipes (`difficult_terrain` and the four single-template presets) clear departure, and `terrain_contact` is exempt from it (`behavior_certification.py:213`). The rule uses apron + blend width (:230), but the generator ignores blend width for three families (`terrain.py:278-282`). Even with a per-family radius, an exact-tracking executor fails sloped (compsognathus 13/20, robot 1/20). The walker also tips over on a flat heightfield (0/13). | (a) Define departure per family from the terrain config, and rescale the maps (row 13). (b) Leave the pair out of the first terrain certification. (c) Drop the rule; G4 does not mention it. | **(a), decided together with row 11**, which may find a physics cause that no rule change fixes. | PR-13's certificate rules and the compsognathus terrain nodes. |
| 15 | **Remaining PR-11/PR-13 choices** (from the readiness review and its critic). Seed terrain per episode (`behavior_env.py:430` omits `_episode_index`, which :432 and :439 include), or document one layout per run. Keep the learning rate at or below the parent's final rate (a constant 5e-5 today against the walkers' 1e-5; compsognathus ≤ 3e-5). Zero `gait_symmetry_weight` for brachiosaurus and dibothrosuchus. Take the family list from the node config, not the report. Use 40 episodes per family with fixed seeds. Make the tolerances and settle/dwell constants threshold keys. Report heading coverage. Add terrain probes for brachiosaurus and dibothrosuchus. PR-11 also owns `make_env` routing to the terrain subclass (`train_base.py:204-229`; the `env_class(**env_kwargs)` call is at :218), the terrain/command TOML-to-dataclass conversion, and a real-PPO smoke of `follow_direction_difficult_terrain` (the only planned one is flat). Raise the compsognathus pair's `n_steps` (1,024) to at least one 1,250-step episode, or record why not. Decide whether behaviors keep 25 s episodes when the parents were certified on 10 s (20 s for compsognathus; velociraptor's certified checkpoint averages 968.6 of 1,000 steps). | — | Take them in the PR-11 and PR-13 reviews. At 20 episodes per family, a 99%-survival policy passes all five families with probability 0.37; at 40 per family with up to 3 falls it passes with 0.997. | PR-11, PR-13. |
| 16 | **The digest snapshot as a CI check.** Nothing in CI pins behavior identities or stage-config views (§4.6 risks), yet every later CU claims "no digest". | (a) Commit the golden output plus a `--check` step: `--skip-behaviors` on pull requests, the full run nightly. (b) Run the harness by hand in each acceptance. | **(a), after PR-B and before CU-8 and CU-11..CU-13.** **Taken 2026-09-29 as D-D22: (a), with the full harness run on pull requests (not `--skip-behaviors`), in the plant-contract job.** With `--skip-behaviors`, a PR that moves a behavior identity (a byte edit to a species env file, as in CU-12) would pass its own CI. Its own PR (ROW-16, outside D-D21's gate) builds the check, not CU-4; it lands before CU-7b, CU-8a, CU-12 and CU-13 (§3.1 item 4). **Carried out by the ROW-16 PR (2026-09-29; §3.1 item 4, §4.4).** The full run at `a919985` (848 lines, 0 errors) is committed as `configs/digest_snapshot.generated.txt`, and the plant-contract job's step "Verify the digest snapshot golden" runs `python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --check` on every event, with no `if:`; a mismatch names every moved line and prints the `--write` command that regenerates the golden. **Landed as #571 on 2026-09-29** (merged 22:44 UTC as `19762d5`). | Nothing; it makes the "no digest moves" claims checkable. |
| 17 | **Retired-backend metric lines.** After the retirement the generated catalog in the root `README.md` (:105-215) and the website still advertise JAX/MJX success metrics for four species that nothing can train. `species_catalog.py:949-954` requires those rows while the species stay dual. | (a) A `species_manifest.toml` field separating the declared interface backends (`plant_contract/manifest.py:88-93`) from the advertised training backends, so the catalog stops rendering JAX/MJX metrics. (b) Keep them with a note. | **(a), as a catalog PR after PR-B.** The removal inventory found the declared backends list is not a bundle-digest input; confirm with the harness. | Nothing; living docs stay true. |
| 18 | **Branch protection.** On 2026-09-25 `main` had no required checks, although `python-ci.yml:210-212,316` assume them (§4.8). | (a) Turn on required checks once the job names settle (after CU-14). (b) Reword the workflow comments. | **(a)**, a maintainer action. | Nothing. |
| 19 | **Stale open PRs.** #527 ("Research Compsognathus feet and add resumable Colab balance sessions", 2026-09-10) edits `python-ci.yml` and `harnesses/freeze_recovery_gate.py` and adds a notebook. #498 ("Record the August 2026 RL pipeline review", 2026-08-08) adds a review that was never merged. | Close, or rebase and merge. | **Decide before PR-A.** #527 conflicts with PR-A/PR-B in `python-ci.yml` and with CU-9. **Decided 2026-09-26: both closed, each with a comment (an operational choice, not a D-D row).** | PR-A (rebase cost). |
| 20 | **Which release is the clean base.** Added 2026-09-27, after the release cut: the maintainer asked whether to finish the cleanup and refactor first, so that the tagged version is the clean base. `0.3.8` was already tagged and published on `afad625`, before any retirement PR. | (a) Re-open `0.3.8`: move the tag to the cleaned tree. (b) Keep `0.3.8` as the pre-cleanup release, and make 0.3.9 the clean, refactored base, cut once a named set of cleanup PRs has landed. | **(b).** A published tag should not move, and `0.3.8` doubles as the archive point for the retired code (row 2). **Taken 2026-09-27 as D-D21: (b).** The 0.3.9 gate is thirteen PRs: PR-A, PR-A2, PR-B, CU-2, CU-4, CU-5, CU-7, CU-8, CU-9, CU-11, CU-12, CU-14 and CU-16. Deferred, not dropped, each at the point §3.2 and §3.4 give it: CU-6 (after session 6's resume and CU-4; amends D-D7), CU-13 (before PR-11; amends D-D5), CU-10 (lowest priority), CU-15 (optional; after PR-A), CU-17 (last, before PR-15), and consolidation PR-8..PR-15, which build on 0.3.9. CU-4 and CU-14 still take decisions 9 and 10 when they open. *Session 6's resume finished on 2026-09-28, so CU-6 waits only for CU-4.* **Split 2026-09-29:** the maintainer accepted splitting CU-7 into CU-7a (dead code and import cost) and CU-7b (retired-backend wording, with CU-16's stage-TOML comments folded in), with CU-7's sha256 regex and validators moving to CU-8c; CU-8 into CU-8a (one derivation), CU-8b (`policy_loading` owns the SB3 import and the sidecar resolver) and CU-8c (one reader, one root, one regex, one set of validators); CU-14 into CU-14b (the plant-identity cache, decision 10 (c)), first, and CU-14a (the workflow YAML); and CU-16 into CU-16a (docs text) and CU-16b (orphan assets). The gate's names cover their parts, so the gate itself is unchanged (§3.1 item 4, §3.2). *Corrected 2026-09-29: CU-6 also follows CU-5, CU-8b and the notebook PR for decisions 4 and 6 (§3.2, CU-6 row), not CU-4 alone.* | The 0.3.9 release cut. |

**Made moot by the retirement:** whether the dibothrosuchus MJX kernel should pay `snap_snout_proximity_weight`; JAX
`ppo_epochs` (4 in the notebook, 10 in the CLI); whether to keep Vertex HPT or mjlab; a Ray sweep-backend decision;
routing the Ray worker through `train()` (it also carried a D-A20 hazard on `Tuner.restore`); the consolidation plan's
§8 risk "take it for both notebooks" (the D-D7 package move); and PR-3b's "look at the JAX job".

**Already answered:** the survey's "RESUME on a finished node" became D-D16 (#558, amended by #559). A further
attempt after an early stop is a fresh `RUN_ID`.

## 3. The remaining cleanup as a PR sequence

### 3.1 Order

0. **The #558 follow-up: landed as #559 on 2026-09-26.** It added `curriculum.checkpoint_pair_problem`
   (`environments/shared/curriculum/checkpoints.py:113` at the follow-up), which the RESUME cell and the chain loop
   use to check the final pair. It keeps the run memo in its tree, refuses a resume that `RETRAIN_FROM` covers, and
   amended D-D16. It went first because PR-B also edits `test_sb3_notebook_pins.py`. The docs PR that adds this plan
   marks its status row (`CONSOLIDATION_PLAN_2026_09.md:41` at the follow-up) landed, with its CI times (SB3 job
   46:51, JAX job 48:10, at `ad4e621`).
1. **CU-1 (CI signal), before PR-A.** CU-1, PR-A and PR-B all edit `python-ci.yml`. PR-A changes the `test-sb3`
   install and verify steps (:267-306, :377). PR-B changes `test-jax-cpu` (:391-434) and the coverage `needs` (:445).
   Landing the small PR first means the large PRs rebase over it once, and "mypy shows the same 21 errors" becomes a
   green CI step. If PR-A is ready first, land PR-A and rebase CU-1; only the size of the conflict changes.
   **Carried out by the CU-1 PR (2026-09-26, D-D18):** the SB3 job's new mypy step reports no errors, from 23 in
   `test-sb3`'s environment and 21 locally (§3.2). **Landed as #561 on 2026-09-26** (measured on its CI: SB3 job
   48:54, whose mypy step printed "Success: no issues found in 359 source files" in 77 s; JAX job 51:20; coverage 90
   percent).
2. **CU-3 (D-D20), the release cut (D-D19), then PR-A (§4.5), PR-A2 (the Vertex route and GCS upload), then PR-B
   (§4.6).** **CU-3 landed as #562 on 2026-09-26** (measured on its CI: SB3 job 47:57, whose mypy step printed
   "Success: no issues found in 361 source files" in 75 s; JAX job 50:05; coverage 90 percent). The release PR dates the three `[Unreleased]`
   headings, opens an empty one and moves the version to `0.3.9.dev0`; the maintainer tags `v0.3.8` on the commit that
   dates its section; like any change that claims to move no digest, the release PR runs the digest-snapshot harness
   on its base and head. **As carried out by the release PR** (the maintainer chose the details on 2026-09-26): its first
   commit, the one the maintainer tags `v0.3.8`, reads `version = "0.3.8"` and dates `[0.3.8] - 2026-09-27`,
   `[0.3.2] - 2026-07-21` and `[0.3.0] - 2026-07-09`, each older section by the UTC date of the commit that opened the
   heading above it (`db153b2`, `3f176ce`); its second opens a bare `## [Unreleased]` and sets `0.3.9.dev0`. The harness
   printed 848 lines with 0 errors, byte-identical on the base `89d814a`, the tagged commit and the head. Because PRs
   here merge with a merge commit, the tagged commit keeps its SHA on `main`. **The release cut landed as #563 on
   2026-09-27** (measured on its CI at `29cbc2d`: SB3 job 46:21, whose mypy step printed "Success: no issues found
   in 361 source files" in 70 s; JAX job 35:47; all 23 checks green; coverage 90 percent). The maintainer tagged
   `afad625` as `0.3.8`, without the `v` prefix, and published it as a GitHub pre-release. **As carried out by PR-A
   (2026-09-27; §4.5):** 42 files and 13,222 lines deleted; the digest-snapshot harness printed 848 lines with 0
   errors, byte-identical on the base `04d107a` and the head; mypy reports no issues in 334 source files (from 361)
   in all three environments; `pytest --collect-only` collects 4,344 tests locally (from 4,599) and 4,294 in
   CI's SB3 environment (from 4,549), with no errors. **PR-A landed as #564 on 2026-09-27** (merged 20:45 UTC
   as `9369d6b`; measured on its CI at `3c5eab7`: SB3 job 47:01, whose mypy step printed "Success: no issues found in 334 source files" in 68 s; JAX job 39:11; all 23 checks green; coverage 91 percent). The run was at reduced depth,
   because the `full-ci` label was missing; the full-depth selections (the 4 notebook-training parameters and
   the 12 behavior-training cases) were then run by hand in CI's SB3 environment on `9369d6b` and passed.
   **As carried out by PR-A2 (2026-09-27; §4.5):** its first commit adds the end-to-end test of the
   command-line curriculum path, which passes on the base; the next two remove the GCS upload and the route
   (4 files and 957 lines deleted) with the test unchanged; the digest-snapshot harness printed 848 lines with
   0 errors, byte-identical on the base `9369d6b` and the head; mypy reports no issues in 335 source files
   (from 334; the new test) in all three environments; `pytest --collect-only` collects 4,339 tests locally
   (from 4,344) and 4,289 in CI's SB3 environment (from 4,294), with no errors. **PR-A2 landed as #565 on
   2026-09-28** (merged 01:02 UTC as `7ae0a19`, whose tree equals `647ca1f`; measured on its CI at `647ca1f`: SB3
   job 34:52, whose mypy step printed "Success: no issues found in 335 source files" in 41 s; JAX job 34:43; all 23
   checks green; coverage 91 percent; at full depth, with the `full-ci` label). **As carried out by PR-B
   (2026-09-28; §4.6):** 30 files and 15,665 lines deleted as whole files (the whole diff is +489 / −21,178 in 113
   files); the digest-snapshot harness printed 848 lines with 0 errors, byte-identical on the base `7ae0a19` and the
   head, with the optional backends blocked and unblocked; mypy reports no issues in 309 source files (from 335) in
   all three environments; `pytest --collect-only environments` collects 3,943 tests both locally (from 4,339) and
   in CI's SB3 environment (from 4,289), with no errors; the plant-contract tests are 69 (54 + the 15 pin tests);
   and the wheel, installed without JAX, reproduces all six `policy_interface_sha256` values. **PR-B landed as #566
   on 2026-09-28** (merged 05:13 UTC as `2b9219d`, whose tree equals `c8b66a6`; measured on its CI at `c8b66a6`, run
   36376798324: SB3 job 37:03, whose mypy step printed "Success: no issues found in 309 source files" in 53 s; all
   22 CI jobs green, with no JAX job any more; coverage 91 percent). The run was at reduced depth, because the
   `full-ci` label was missing (its log printed "SB3 depth: one real-PPO smoke per body of work"); the full-depth
   selections (the 4 notebook-training parameters and the 12 behavior-training cases) were then run by hand in CI's
   SB3 environment on `2b9219d` and passed. PR-B follows PR-A (§4.7). Both use the digest-snapshot harness, which is already in the repository (§4.4), for their acceptance.
3. **The surviving waves (§3.2).** CU-7 is independent of the retirement (CU-7a landed as #572 on 2026-09-30; CU-7b follows
   CU-7a and ROW-16, item 4) (CU-3 and CU-2, also
   independent, landed as #562 and #568). Work
   that edits CI structure, or docs the retirement also edits, waits for PR-B (landed as #566, 2026-09-28).
   **Carried out by the CU-2 PR (2026-09-28):** `render_mode="human"` loads the viewer itself instead of crashing
   on the first step, and `generate_stage_artifacts`, when it builds its own results, takes the node's own control
   step (0.02 s for the compsognathus pair) instead of 0.01 s; both KNOWN_ISSUES entries are deleted (§3.2, §3.5).
   The digest-snapshot harness, run with the optional backends blocked, printed 848 lines with 0 errors,
   byte-identical on the base `1ce42f6` and the head. **CU-2 landed as #568 on 2026-09-29** (merged 00:03 UTC as
   `7012483`, whose tree equals `d8ea2a6`; measured on its CI at `d8ea2a6`, run 36486171082: SB3 job 37:11, whose
   mypy step printed "Success: no issues found in 309 source files" in about 51 s; all 22 CI jobs green; coverage
   91 percent). The run was at reduced depth, without the `full-ci` label (its log printed "SB3 depth: one real-PPO
   smoke per body of work"); nothing in CU-2 changes training, a config or a digest input.
4. **The rest of the gate, in the order the maintainer accepted on 2026-09-29** (§2 rows 16 and 20). A PR opens once
   the PRs listed with it have landed, and the PRs of one wave can run side by side. ROW-16, the digest-snapshot
   check of D-D22, is a PR of its own outside the gate and a hard predecessor only of CU-7b, CU-8a, CU-12 and CU-13;
   CU-4 at most rebases over it (both edit `python-ci.yml`, in different jobs). The longest chains are five PRs, all
   ending CU-8a → CU-8b → CU-8c (from CU-4 through CU-5, from CU-7a through CU-7b or CU-9, and from ROW-16 through
   CU-7b).
   - **Wave 1:** ROW-16, CU-4, CU-7a, CU-11 and CU-14b.
   - **Wave 2:** CU-9 after CU-7a; CU-5 after CU-4; CU-7b after CU-7a and ROW-16; CU-16a after CU-7a.
   - **Wave 3:** CU-12 after CU-11, CU-7b and ROW-16, with the `full-ci` label; CU-14a after CU-4 and CU-14b; CU-16b
     after CU-16a; CU-8a after CU-4, CU-5, CU-7b and CU-9.
   - **Wave 4:** CU-8b, then CU-8c; then the 0.3.9 cut (D-D21).

   **Carried out by the CU-4 PR (2026-09-29):** one standard-library notebook reader serves the tests and CI's
   notebook check, a new test resolves every `environments` import of both notebooks, the four duplicated
   library-only pins are deleted, and shared helpers replace the test-to-test imports and the TinyEnv copies (§3.2).
   The digest-snapshot harness, run with the optional backends blocked, printed 848 lines with 0 errors,
   byte-identical on the base `7012483` and the head. **CU-4 landed as #569 on 2026-09-29** (merged 11:02 UTC as
   `c69a169`, whose tree equals `9d8b53f`; measured on its CI at `9d8b53f`, run 36515741479: SB3 job 48:37, whose
   mypy step printed "Success: no issues found in 313 source files" in about 71 s; all 22 CI jobs green; coverage
   91 percent). The run was at reduced depth, without the `full-ci` label (its log printed "SB3 depth: one real-PPO
   smoke per body of work"); the lint job's new step, `python environments/shared/tests/notebook_cells.py`, printed
   "all notebook code cells parse".

   **Carried out by the CU-14b PR (2026-09-29; §2 row 10 (c)):** `current_plant_identity` keeps one build per
   process and species, reused only while everything the build reads that can change in a process, Python code
   aside (a known limit, §3.2), is unchanged and the policy layer, re-run on a fresh environment, still matches; the `verify_generated` comparison with the committed manifest runs
   on every call, hit or miss, and the manifest build and CI's `--check` never read the cache (§3.2, CU-14 row); a
   review follow-up checks the compiled physics of every env `config.build_env` builds, which the frozen recovery gate,
   its calibration and the zero-action baseline's preflight had left to the identity alone (§3.2, CU-14 row). Measured
   locally, CI's three SB3 pytest steps took 855 s with the cache against 2,609 s at `c69a169`; runners differ, so
   CU-14a decides 10 (a) from the PR's own CI durations. The digest-snapshot harness, run with the optional backends
   blocked, printed 848 lines with 0 errors, byte-identical on the base `c69a169` and the head. **CU-14b landed as
   #570 on 2026-09-29** (merged 18:44 UTC as `a919985`, whose tree equals `e62ce6d`; measured on its CI at
   `e62ce6d`, run 36611580600: SB3 job 17:31, whose mypy step printed "Success: no issues found in 314 source
   files" in about 65 s; all 22 CI jobs green; coverage 91 percent). The run was at reduced depth, without the
   `full-ci` label (its log printed "SB3 depth: one real-PPO smoke per body of work"); its three SB3 pytest steps
   took 879 s (#569: 2,731 s) and the robot's freeze rehearsals, the `build_env` physics check, 30.4 s each (181 s
   at #569). CU-14a decides 10 (a) from these durations.

   **Carried out by the ROW-16 PR (2026-09-29; D-D22, §2 row 16):** the harness's full output at `a919985`, 848
   lines with 0 errors, is committed as `configs/digest_snapshot.generated.txt` (not shipped in the wheel), and
   the plant-contract job's step "Verify the digest snapshot golden" runs `python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --check`
   after the monotonicity step, on every event (about 45 s in a CI-like install). The harness gains `--check
   [PATH]`, which names every moved line and prints the diff and the regeneration command, and `--write [PATH]`,
   which refuses a run with an ERROR line; both imply `--block-optional-backends` and refuse `--skip-behaviors`.
   `test_plant_contract_digest_snapshot.py` pins the whole step (no `if:`, `continue-on-error:`, `shell:` or run
   defaults can skip, tolerate or replace it), the golden's shape and `main()`'s `--check`/`--write` wiring.
   `.gitattributes` joins both path filters, since it sets the checkout's line endings and the digests hash file
   bytes; the readers of `python-ci.yml` move from `test_ci_tool_pins.py` into `ci_workflow_helpers.py`, so no test
   module imports another. A PR that moves a digest on purpose regenerates the golden with `--write` in its own
   diff. No digest moves: the golden equals the harness's plain output on the base. Validated locally: in a pip install
   built as the plant-contract job builds it (Python 3.12, `.[test]`, MuJoCo 3.10.0, no SB3), the exact CI command
   reports the golden current (848 lines, about 40-45 s) and the plant-contract pytest step passes (186 tests); mypy
   finds no issues in 316 source files in all three environments; the suite collects 3,991 tests and passes (3,990
   passed, 1 skipped). *Review follow-up (2026-09-29):* an adversarial review confirmed six minor findings, all fixed:
   the regeneration command names a custom PATH, quoted; a byte-order mark or extra final newlines are reported as the
   file's form, not as moved digests, and moved lines are counted once each; the trigger pins refuse `!` path
   patterns in either YAML quote style, since one could exclude a pinned file; the lint job the plant-contract job needs must also always run; the
   refusal tests prove that nothing ran first; and the golden test compares bytes, so a lone CR line ending fails it.
   **ROW-16 landed as #571 on 2026-09-29** (merged 22:44 UTC as `19762d5`, whose tree equals `6ae87e4`; measured on
   its CI at `6ae87e4`, run 36639964231: the new step took 50 s and printed "digest_snapshot: 848 lines, 0 errors,
   49.6 s" and "Digest snapshot is current: configs/digest_snapshot.generated.txt (848 lines)"; the plant-contract
   job's pytest step passed 186 tests; SB3 job 13:09, whose mypy step printed "Success: no issues found in 316
   source files"; all 22 CI jobs green; coverage 91 percent). The run was at reduced depth, without the `full-ci`
   label (its log printed "SB3 depth: one real-PPO smoke per body of work"). The first push run on `main` after the
   merge (run 36641336583) reported the golden current too, in 49 s.

   **Carried out by the CU-7a PR (2026-09-30; §3.2, CU-7 row):** the dead code goes with the tests that covered
   only it, among it `gate_schema`'s two backend-override merge helpers, a departure from §4.9 recorded there;
   the gymnasium entry-point groups, which no supported gymnasium version reads, go with their KNOWN_ISSUES
   bullet (§3.5); `environments/shared/__init__.py` keeps only its docstring, so `import environments` no longer
   loads SB3, torch and wandb where they are installed; `imageio` joins the `viz` extra; and the plant contract's
   MJX probe requires exactly one root body, which closes the quadruped-detection MEDIUM (§4.6). The digest-snapshot
   check, run with CI's command, reports the golden current (848 lines, 0 errors), so no digest moves.
   **CU-7a landed as #572 on 2026-09-30** (merged 01:55 UTC as `fd8ba78`, whose tree equals `79af481`; measured on
   its CI at `79af481`, run 36653953530: the digest step took 50 s and printed "digest_snapshot: 848 lines, 0 errors,
   49.7 s" and "Digest snapshot is current: configs/digest_snapshot.generated.txt (848 lines)"; the plant-contract
   job's pytest step passed 187 tests; SB3 job 17:30, whose mypy step printed "Success: no issues found in 316
   source files"; all 22 CI jobs green; coverage 91 percent). The run was at reduced depth, without the `full-ci`
   label (its log printed "SB3 depth: one real-PPO smoke per body of work"). The first push run on `main` after the
   merge (run 36657404412) passed.

   **Carried out by the CU-11 PR (2026-09-30; §3.2, CU-11 row):** the digest-snapshot harness gains a `reward`
   section and `--exact` (§4.4). CI's existing digest step checks the four new lines per stage, and the
   digest-snapshot tests pin their completeness, the ends the probes reach and the probes' distance from the tilt,
   height and nosedive thresholds. The first 848 lines of the golden are unchanged, so no existing digest moves: CI's command reports the
   golden current (932 lines, 0 errors). Once it lands, wave 1 is complete.
   **CU-11 landed as #573 on 2026-09-30** (merged 12:14 UTC as `89fd117`, whose tree equals `d4496a5`;
   measured on its CI at `d4496a5`, run 36680247363: the digest step took 73 s and printed "digest_snapshot: 932
   lines, 0 errors, 72.2 s" and "Digest snapshot is current: configs/digest_snapshot.generated.txt (932 lines)";
   the plant-contract job took 5:53 and its pytest step passed 195 tests; SB3 job 17:47, whose mypy step printed
   "Success: no issues found in 316 source files"; all 22 CI jobs green; coverage 91 percent). The run was at
   reduced depth, without the `full-ci` label (its log printed "SB3 depth: one real-PPO smoke per body of work").
   The first push run on `main` after the merge (run 36713571425) passed. Wave 1 is complete.

   **Carried out by the CU-9 PR (2026-09-30; §3.2, CU-9 row):** coverage's blanket `*/scripts/*` and
   `environments/shared/harnesses/*` omits become 21 named entries (30 files), each group with its reason;
   `digest_snapshot.py` stays omitted by name, and `test_coverage_config.py` pins the list (no blanket glob, a `*`
   only for the species directory, no stale entry, the certification code and the statue baselines measured, no
   other file that a test imports by name omitted, and no other way out of the union in coverage's configuration:
   no report-time omit, no `include`, no other coverage config file). `brace_controller` and its two settle
   constants move from the hand-run `harnesses/recovery_offdist_panel.py` to `recovery_evaluation.py`, so the
   freeze producer no longer imports the panel that imports it; new tests cover the brace (unit, fake env) and the
   producer's T-Rex brace branch, which no test reached. Measured locally on the CI-like union (the plant-contract
   selection and the six suites on 3.12 without SB3, the SB3 job's three steps with it): 19,626 statements, 2,405
   missed, 87.75 percent against 17,447 statements, 1,528 missed, 91.24 percent before (#573's CI: 91 percent).
   CI's command reports the digest golden current (932 lines, 0 errors), so no digest moves; `plant_contract
   --check` is current.
   **CU-9 landed as #574 on 2026-09-30** (merged 15:39 UTC as `ae4d651`, whose tree equals `8465ace`;
   measured on its CI at `8465ace`, run 36715850832: the digest step took 72 s and printed "digest_snapshot: 932
   lines, 0 errors, 72.1 s" and "Digest snapshot is current: configs/digest_snapshot.generated.txt (932 lines)"; the
   plant-contract job took 5:51 and its pytest step passed 195 tests; SB3 job 13:47, whose mypy step printed
   "Success: no issues found in 317 source files"; all 22 CI jobs green; coverage 88 percent, 19,626 statements with
   2,405 missed, exactly the local measurement). The run was at reduced depth, without the `full-ci` label (its log
   printed "SB3 depth: one real-PPO smoke per body of work"). The first push run on `main` after the merge (run
   36738456084) passed.

   **Carried out by the CU-5 PR (2026-09-30; §3.2, CU-5 row):** the SB3 notebook's markdown goes from 223 to 111 source
   lines with every pinned phrase and D-D16 rule kept, and a new pin keeps the resume recipe's trunk-recovery text
   and D-D16 rules; the deleted `trex/stance.toml` quote, "all four" species, the "review F3" note and the chain
   loop's plan-numbered branch labels are fixed; the dead plotting, summary and bundle parameters go and
   `train_stage`'s `run_dir` is required; `node_budget()` is the one budget derivation of the chain loop, the RESUME
   cell and the manual cell. 33 cells (17 code), 1,599 → 1,468 source lines; no library, configuration or digest
   change: CI's command reports the digest golden current (932 lines, 0 errors). Cells 15 and 20 change, so the PR
   needs the `full-ci` label; the real-PPO notebook training at full depth passed locally (4 cases).

### 3.2 The surviving waves (relabelled CU-n; "was" names the survey item)

Sizes are survey estimates of net lines unless marked as measured. "No digest" means the PR touches no hashed
callable, no byte-frozen file and no digest input (§5.6).

| PR | Was | What | Why | Size | Risk | When |
|---|---|---|---|---|---|---|
| CU-1 CI signal | C2 (trimmed) | Re-measure the mypy errors in `test-sb3`'s environment (decision 8), then fix them with casts and annotations (the 21 local ones are listed in §5.5). The `advancement.py` fixes must be casts, not runtime guards. The SB3-absent fallback at `diagnostics.py:218-223` gets `# type: ignore[misc,assignment]`. Add the pinned mypy step (decision 8). Keep CI's `-v` and add `--durations=30 -rs`; where test ids are wanted, pass `-vv` (or `-o addopts="--tb=short"`), because `pyproject.toml:132` `addopts = "--tb=short -q"` cancels one `-v`. Pre-commit pins ruff `v0.4.4` and mypy `v1.15.0` (`.pre-commit-config.yaml:3,10`) while CI installs both unpinned (`python-ci.yml:82`); pin one ruff version and one mypy version for both. Closes the two Testing / CI KNOWN_ISSUES entries (§3.5). **As carried out (2026-09-26, D-D18):** 23 errors in 8 files in `test-sb3`'s environment on `8e03483` (§2 row 8), fixed with casts and annotations and no runtime change (the four in the SB3-absent diagnostics fallback with the planned `# type: ignore`); the SB3 job pins `stable-baselines3[extra]==2.9.0` and runs mypy 2.3.1; ruff 0.16.9 and mypy 2.3.1 are pinned in the lint job, pre-commit and the `dev` extra, and `test_ci_tool_pins.py` keeps them, with the SB3 pin, in agreement. The pre-commit ruff hooks are scoped to `environments/`, because this ruff also formats notebooks and the Python blocks in Markdown files: unscoped, it would rewrite all four notebooks and 15 Markdown files, two of them frozen investigations. Every pytest step passes `-vv -rfEs --durations=30`, not `-rs`, which replaces pytest's default `-rfE` and would drop the failure lines. The review of the CU-1 PR added two things: `.pre-commit-config.yaml` joins both path filters of `python-ci.yml`, so a pre-commit-only PR runs the pin checks, and a top-level pre-commit `exclude` keeps every hook off the digest data files (MJCF sources and meshes, recipe TOMLs, plant manifests, `plant_versions.toml`, recovery calibrations; the byte-hashed Python modules stay under the hooks, which CI's pinned ruff keeps from changing them): `pre-commit run --all-files` would have let `end-of-file-fixer` append a newline to three compsognathus MJCF files and move both compsognathus plant identities (reproduced in a scratch worktree; `plant_contract --check` then reports the manifest stale). | CI cannot see SB3 types. Logs show no test ids, durations or skip reasons. Pre-commit reformats 22 files that CI accepts. A pinned ruff also keeps a formatter release from reformatting the token-hashed callables of every species (§5.6). | +40–60 | LOW | Landed as #561 (2026-09-26) |
| CU-2 Live library bugs | C1 (viewer half) + critic item | Add a lazy `import mujoco.viewer` in `render()`'s human branch (`base_env.py:1556-1566`; the imports at :19-21 lack it), one camera helper shared with `_make_camera`, and a test that monkeypatches `launch_passive`. Fix the `sim_dt` default of 0.01 at `reporting/stage_artifacts.py:151` by taking `dt` from a probe env. Both compsognathus species step at 0.02 s, so their summaries print half the sim time, but only where `generate_stage_artifacts` builds its own results (`stage_results=None`, :1463): the Ray and Vertex sweep trials (`ray_tune.py:1025`, `trial.py:233`). The notebook path overwrites `sim_dt` with the env's `dt` (:1242, :1277), and `backfill_gate_verdict.py` persists nothing that uses it, so after PR-A the default is latent. Closes the render and `sim_dt` KNOWN_ISSUES entries (§3.5). **As carried out (2026-09-28):** the human branch imports the viewer itself (`from mujoco import viewer as mujoco_viewer`), since `import mujoco` does not load `mujoco.viewer`; it is imported under its own name because a bare `import mujoco.viewer` would make `mujoco` local to `render` and break the rgb_array branch. One `_configure_camera` helper aims both the viewer's camera and `_make_camera`'s, the viewer's under the viewer's lock (the CU-2 review found that its render thread reads the camera on every frame, and that an unlocked tracking camera with no body yet can end the process). `generate_stage_artifacts`, when it builds its own results (`stage_results=None`), takes `sim_dt` from a bare env of the node's task (`species_cfg.env_class(**env_kwargs).dt`, closed afterwards), as `evaluate_stage_checkpoints` already did; `build_stage_results_from_eval_data` gains a `sim_dt` keyword that both callers pass, and its fallback for other direct callers stays `env_kwargs["sim_dt"]` or 0.01 s, as its docstring now says. `backfill_gate_verdict.py` still calls it without `sim_dt`, and nothing it persists reads `sim_dt`. Five new tests: a human render with `mujoco.viewer` removed from the package and a fake in `sys.modules` launches once, aims the camera like `_make_camera` and under the lock, syncs every step and closes; an rgb_array render works with a fake `mujoco.Renderer` (the test that guards against the bare import); the probe env's `dt` reaches the results and the env is closed; an explicit `sim_dt` wins over the fallback; both compsognathus envs step at 0.02 s. Each of six mutations fails at least one of them (the old attribute access, a bare import, no camera setup, no lock, a generate that ignores the probe's `dt`, a build that ignores `sim_dt`). Measured (`git diff --numstat` against `1ce42f6`): 5 files, code +36 / −12 in `base_env.py` and `stage_artifacts.py`, tests +167. Both KNOWN_ISSUES entries are deleted (§3.5). The CU-2 review found a second 0.01 s fallback, in `write_training_summary` for a node the notebook re-enters from its own verdict, whose `stage_result` projection omits `sim_dt`; it is a new KNOWN_ISSUES LOW, unowned. The digest-snapshot harness, run with the optional backends blocked, printed 848 lines with 0 errors, byte-identical on the base `1ce42f6` and the head. | `evaluate` without `--no-render` crashes on step 1. Re-checked 2026-09-26: `mujoco.viewer` is not imported after `base_env`, `train_base` and `evaluation` load (mujoco 3.10.0). | about +35 | LOW (`render` is not hashed; `base_env.py` is not byte-hashed) | Landed as #568 (2026-09-29) |
| CU-3 Atomic writes and handoff pairs | C8 + decision 7 | Add `file_io.atomic_write_json` and `atomic_write_csv`. Use them for `stage_config.json` (`config.py:823`), `metrics.json`, the stance JSON/text writers and the three evidence CSVs, keeping each site's `json.dumps` arguments so the bytes stay identical. Three hand-rolled writers leave `<name>.json.tmp` files that the manifest's `.*.tmp` cleanup misses. That happens only after a crash between write and replace, and the next write overwrites the file (survey critic). Also stage the final and best pairs (decision 7), which closes the handoff-pair KNOWN_ISSUES entry (§3.5). **As carried out (2026-09-26, D-D20):** `file_io` gains `atomic_write_json` and `atomic_write_csv` (and `atomic_write_text` a `newline` argument). `stage_config.json`, `metrics.json` (still without a trailing newline), the five stance text/JSON pairs, the three evidence-CSV writers (`save_evaluation_episodes`, `write_stance_panel_evidence`, `write_recovery_evidence`) and the three hand-rolled sidecar writers (`gate_resolution.json`, `task_fingerprint.json`, `plant_identity.json`, all written inside run trees) go through them with their bytes unchanged, so a stranded temporary is now dot-named and discarded by the manifest. On a mount the final, best and robust-best pairs are saved to local scratch and published by `curriculum.publish_staged_pair`: handoff pairs zip last, the final pair sidecar last, each after removing the destination file it publishes last, so no reclaim leaves a truncated file or a mixed pair the pair's checks accept. Before the final pair is saved, an empty placeholder takes the final zip's place and then the final sidecar is removed, so a reclaim at any point of the final save leaves a final zip `checkpoint_pair_problem` rejects, as a save straight to the mount did; without it, a reclaim before the zip landed left no final zip, and the RESUME cell trained an early-stopped node further in place (found by the CU-3 review and reproduced; `test_sb3_notebook_pins.py` now runs the RESUME cell on each state the staged save can leave). Off a mount the pairs are written in place, as before. Not converted, among other writers of the same pattern outside CU-3's list: `stage_summary.txt` and `training_summary.txt`, `collected_results.csv` and `curriculum_results.csv` (append mode), the stance diagnostics CSV, the run's `zero_action_baseline.json`, the stage's `wandb_run_id.txt`, `ancestors.py`'s record copy, the widen tool's pairs, the behavior pilot's outputs and the repository's `plant_manifest.generated.json`. Measured (`git diff --numstat` against `74ba16c`): code +270 / −99, tests +1,039 / −3, notebook JSON +2 / −2; the size estimate above counted only tests. | Otherwise PR-13's evidence writer copies the non-atomic pattern. | about +60 (tests) | LOW | Landed as #562 (2026-09-26) |
| CU-4 Test helpers | C19 | One `notebook_cells.py` for the notebook extractors (fewer after PR-A and PR-B), which the CI notebook validator uses too. Delete the duplicated library-only pins (decision 9) and the dead `_clean_repository_state`. `ancestors_helpers.py` replaces the test-to-test imports. One TinyEnv. Add a notebook import-resolution check and a canonical-JSON pin for the two notebooks that remain; CI only AST-parses notebooks (`python-ci.yml:93-124`). **As carried out (2026-09-29):** `environments/shared/tests/notebook_cells.py` (111 lines, standard library only, no relative imports) holds the notebook readers, and the 16 extraction sites in 9 test files import from it (the pins file keeps thin wrappers). Run by path with no arguments, it parses every code cell of the repository's notebooks and fails if it finds none, so CI's lint step "Validate notebooks (parse every code cell)" is now `python environments/shared/tests/notebook_cells.py` in place of the inline script, with the same messages. `test_notebook_cells.py` (155 lines, 5 tests) adds the import-resolution check: every `environments` import in both notebooks (66 name imports in the SB3 notebook, 52 of them distinct, and 12 in the Drive summary, from 21 modules; names bound only under `if TYPE_CHECKING:` do not count) resolves without importing the package; two more tests run `notebook_cells.py` by path with `python -I -S`, on the real notebooks and on a broken one, and one checks that a failure names a repository notebook as `notebooks/<name>.ipynb`. The canonical-JSON pin covers both notebooks (ids `sb3` and `drive_summary`). The four duplicated library-only pins are deleted, each with a twin that runs in the shared (non-SB3) matrix: `test_the_manifest_mechanism_the_notebook_relies_on` (twin in `test_stage_manifest.py`), `test_the_library_rule_the_notebook_relies_on` (decision 9's pin; twin `test_ancestors.py`'s `TestFindCertifiedAncestor`, invariant 6's own pin), `test_the_selection_it_delegates_to` (twin `TestSelectTrunk`) and `test_the_gate_kind_set_the_notebook_relies_on` (twin in `test_recovery_gate_config.py`); `StageManifestError` leaves the pins file's imports. The manifest pin asserted things its twin did not, so those assertions moved into `test_stage_manifest.py`: trex's `chain_for("recovery")` and `chain_for("locomotion")`, trex's `resolve_behavior("stance")`, and the recipe-label checks in a new loop over all six committed species (the old twin checked trex and the three integer species, not the compsognathus pair) (dropping `recipe = "stand"` from compsognathus's recovery stage fails the moved test and passed the old twin). `ancestors_helpers.py` (210 lines) holds the trunk builders and constants, moved byte-identically, and `test_ancestors`, `test_replication` and `test_train_base` import from it; `_canonical_summary`, with `_load_summary`, `_published_summary`, `_canonical_plant_identity` and `REPOSITORY_ROOT`, moves to `result_bundle_helpers.py`; no test module imports another. `tiny_env_helpers.tiny_env_class(obs_dim=1, info=None)` replaces the eight TinyEnv copies, which differed only in an observation shape (one copy) and an `info` dict (two), and the subprocess script in `test_curriculum_staged_pairs.py` imports it too. `test_sb3_notebook_pins.py` goes from 118 tests to 115 (four pins out, one Drive-summary case in). Kept on purpose: the cross-file notebook-pin duplicates (`RUN_RECOVERY_STAGE` three times, the no-inference pin twice, `chain_results` routing twice), which stay for PR-15; the widen-help pin (`test_widen_checkpoint.py` runs only with SB3); the `TestCommandSliceReseed` pins; and `TinyEvalEnv`. Measured (`git diff --numstat` against `7012483`): 25 files, +726 / −642, net +84, all under `environments/shared/tests/` except `python-ci.yml` (+5 / −27). About 300 of those lines are moves (`ancestors_helpers.py` +210 against `test_ancestors.py` −190 / +13, and +97 in `result_bundle_helpers.py` against `test_result_summaries.py` −95 / +2). The net is +84, not the estimate of about −200: the new check and its tests (`test_notebook_cells.py`, 155 lines) and `notebook_cells.py` (111) outweigh the deletions, and the estimate also counted `_clean_repository_state`, which PR-B had already deleted. It touches tests, test helpers and the lint step only. Validated on the code commit: the digest-snapshot harness, run with the optional backends blocked, printed 848 lines with 0 errors, byte-identical on the base `7012483` and the head; mypy finds no issues in 313 source files (from 309) in all three environments; `pytest --collect-only environments` collects 3,951 tests (from 3,949: four pins fewer, one canonical-JSON case and five `test_notebook_cells.py` tests more), and the full suite passes (3,950 passed, 1 skipped). | CU-5 and CU-6 rewrite the same extractors and pins. This is PR-15's test slice. | about −200 | None | Landed as #569 (2026-09-29) |
| CU-5 Notebook text and dead parameters | C5 | Re-check the survey's text fixes against the follow-up's notebook: the deleted `trex/stance.toml` quote, "all four" species (there are six), the stale "review F3" note, and the branch numbering. Drop `save_path`/`save_dir`/`_show`/`fig1`/`fig2`, `species=None` and `run_dir=None`. Add `node_budget()`. Cut the markdown toward about 95 lines, most of this PR's reduction. Measured: the SB3 notebook has 33 cells and 1,557 source lines at `f850815` (207 of them markdown), and 1,599 at the follow-up. The survey projected about 29 cells and 1,290 source lines after C4, C5, C6 and C9, counted from 1,515; re-derive it from 1,599. **As carried out (2026-09-30):** the row's four text items: the zero-action cell drops its quote of a `configs/trex/stance.toml` comment that `435f35f` deleted and says directly that the floor goes stale when the plant or the stage-1 reward weights change; "all four" species becomes "all six" (the cell and a code comment), and the four-species bash recipe becomes one sentence naming the script; the resume section's "review F3" / "this branch" history goes and its rule stays (a policy is never resumed without its sidecar); the chain cell's labels become `(1) REUSE`, `(2) JUDGE`, `(3) TRAIN`, the loop's order, which section 6 already used (`BEHAVIOR_RECIPES_PLAN.md` §4.7 keeps its own numbering, so the library's references to its "branch 3", the Judge, stay true). Also stale and fixed: the title's legacy `stage*.toml`, the configuration cell's three-stage list (the compsognathus manifests have four nodes) and its "not demonstrated walking" (the gait audit has compsognathus `20260921_203149` walking; the robot caveat stays in the README's words, "cannot be deployed as-is on the robot"), relative links that Colab cannot open (now absolute GitHub links, as the Colab badge's), and a code comment's "cleanup cell". Dead parameters, none passed at any call site: `plot_training_curves`'s `save_path`, `plot_diagnostics_graphs`'s `save_dir` and `_show` and its unused `fig1`/`fig2`, `write_training_summary`'s `species=None` and `save_run_bundle`'s `run_dir=None` (the wrappers forward `SPECIES` and `RUN_DIR`); `train_stage`'s `run_dir` has no default any more, since its fallback, `logs/` under the repository root, would have trained a node outside the run, and every caller passes it. `node_budget(stage)` in the infrastructure cell (the stage TOML's `curriculum.timesteps`, or 50,000 under `QUICK_TEST`, read when it is called) replaces the three copies in the chain loop (TRAIN and JUDGE), the RESUME cell and the manual cell, which only a comment kept in step, while D-D16's refusals are right only when they agree; CU-6 can take it into the package. Measured: 33 cells (17 code), 1,599 → 1,468 source lines (−131; code 1,376 → 1,357), markdown 223 → 111 lines and 3,286 → 1,914 words (−42 percent, so the cut is not line-joining). Why 111 and not about 95: the survey's 95 is its cut of 112 from 207 lines at `f850815`; #559 then added 16 lines of D-D16 text to the resume section, which stay, and the same cut from 223 is 111. Re-derived projection: after CU-6's notebook −115, about 1,350 lines, not 1,290 (the gap is the 84 lines of D-D16 safety code and prose), and 33 cells (29 would need four markdown cells merged or deleted, which no row asks for). Every pinned phrase stays on one line and no pin is weakened. The pins grow from 115 to 119 tests: `node_budget` at every budget site, and executed for every node of all six species under both `QUICK_TEST` values; the branch numbering in the code and the prose; the resume recipe's trunk-recovery and D-D16 phrases (13, among them "not necessarily the trunk", "never follows this run's own ancestor records" and "spent budget without an intact final pair"), which no pin read before; three strengthened pins (`run_dir` has no default; the bundle and summary are written for `RUN_DIR` and `SPECIES`, and neither wrapper takes a parameter it ignores; the plotting wrappers take and forward no save path); one re-pin of the same strength (`discover_replicates_for_run(RUN_DIR, ...)`); and `notebook_cells.exec_top_level_def`, with which the executed tests define the real `node_budget`. Measured (`git diff --numstat` against `8465ace`, #574's head): 4 files, +232 / −246 (the notebook +109 / −240, the pins +106 / −4, `test_compsognathus_training.py` +4 / −1, `notebook_cells.py` +13 / −1). Validated on the code commit: the digest-snapshot check, run with CI's command, reports the golden current (932 lines, 0 errors), and nothing reads the notebook into a digest; mypy finds no issues in 317 source files in all three environments; the full suite passes (4,005 passed, 1 skipped); the real-PPO notebook training at full depth passes (4 cases, 161 s), which CI runs on a pull request only under the `full-ci` label, so the PR needs it (cells 15 and 20 change). *Review follow-up (2026-09-30):* an adversarial review confirmed five findings, one minor and four nits, all fixed: cell 1 now says the run directory keeps the checkpoints, verdict and artifacts of every node this run trains or judges, and the bundle publishes the certified ones (a reused ancestor is only recorded under `ancestors/`); cell 5, that the resolve cell prints the first 20 runs it refused (`TRUNK_SELECTION.candidates` holds every run scanned); cell 17, that a node cut off at the runtime cap has no final pair; and the resume-prose pin also holds "`gate_verdict.json` or an intact final pair" and "`BEHAVIOR` set to it". Dropping either phrase, or restoring `run_dir=None`, fails a pin. | Moves toward the consolidation plan's §4 notebook targets. | about −110 | None (keep the pinned phrases) | After CU-4. *Carried out (2026-09-30).* |
| CU-6 Resume slice into the package | C6 (remainder) | The follow-up already moved the pair check. What remains: `newest_intact_periodic_pair` sharing `train_base`'s regex, with one test per skip reason; `train_stage(evaluate=False)`, which drops a 60-episode evaluation that JUDGE always redoes; and moving the archive preflight into `policy_loading`. *2026-09-30 (CU-5): the budget derivation is `node_budget` in the notebook's infrastructure cell, for the resume slice to take along; the notebook is 1,468 lines after CU-5.* | Resume becomes tested library code. | notebook −115, library and tests +200 | LOW | After session 6's resume and CU-4; amend D-D7. *Session 6's resume finished on 2026-09-28, so CU-6 waits only for CU-4.* *Corrected 2026-09-29: it also follows CU-5 (the same notebook), CU-8b (`policy_loading`) and the notebook PR for decisions 4 and 6 (§2 rows 4 and 6), and no Colab resume may be pending when it merges.* |
| CU-7 Dead code and import cost | C7 (minus JAX) + critic items | Delete 6 dead defs, `FIGURE_NAMES`, the stale "funnel through `summarize_stance_panel`" text, the test-only `recovery_evaluation.paired_success_differences`, and the dead gym entry points (`pyproject.toml:77-88` after PR-A2; gymnasium 1.3.0 loads no plugins). Reduce `environments/shared/__init__.py` to its docstring; the survey measured `import environments` falling from 1.9 s to 0.3 s and from 1,676 modules to 390. Script hygiene. One sha256 regex and one set of validators. Declare `imageio` (used at `compsognathus/scripts/view_model.py:117`). Delete what PR-A leaves of the KNOWN_ISSUES gym entry-point bullet (under Configs, docs & website) and fix `docs/ROADMAP.md:48-52`, which ticks "Register Gymnasium entry points" and names `MesozoicLabs/Velociraptor-v0` (the id is `Raptor-v0`). Correct the fingerprint docstring (`task_fingerprint.py:88` says "all four species constructors"; there are five) and, unless PR-B has already deleted it, the `foot_contact_*` test comment (`test_species_integration.py:454-456`) (§7). Optional: make `curriculum/__init__.py` lazy (a PEP 562 `__getattr__` over the same names), so the pure gate modules stop importing SB3 and torch; the survey measured `curriculum.gate_schema` falling from 1.73 s to 0.28 s that way. **Also left by PR-A (2026-09-27)**, sweep and HPT wording in code that PR-A did not otherwise touch (line numbers at PR-A's head): in `train_base.py`, the `# ── HPT metric reporting` banner (:1497), the `HPT metric reported` and `HPT eval` log strings, the "HPT report" in `train()`'s `report_metrics` docstring (:1116), the "sweep warm-starts" in the load-mode docstring (:399), the GCS comment at :1637-1638 (PR-A2's), the Ray Tune worker in `run_success_panel`'s docstring (:1664-1666), the sweep-row comments and docstrings at :1733, :1765-1770 and :1889, and the sweep-CSV comments at :2822 and :2832; in `reporting/stage_artifacts.py`, the sweep-trial docstrings and comments at :52-53, :113-114, :137-138, :197-204, :302-311, :1037-1038, :1432-1433 and :1479; the "alias used by existing sweep analysis" comments at `train_base.py:1871` and :1881; in `curriculum/task_success_gate.py:22-23`, the sweep's offline row verdict, which D-B12's D-D17 amendment retires; in `stage_manifest.py`, the sweeps (:43, :209) and the sweep collector (:76), which D-A12's amendment retires; in `reporting/csv_output.py`, :4, :21-22 (it names the deleted `sweep/results.write_results_csv`), :38-39, :87, :249, :258 and :335; `reporting/gates.py:6`; `reporting/stage_layout.py:27`; `config.py:175`; `cli.py:21`; and `tb_sync.py:4`. `config.py:988` goes with PR-A2's GCS code, and `result_bundle/evidence.py:565` and `result_bundle/hashing.py:38` wait until after PR-B, whose acceptance keeps `result_bundle/` unchanged. The `best_mean_*` alias keys stay (§4.9). *As carried out by PR-A2 (2026-09-27):* `config.py:988` went with the GCS code and `train_base.py:1637-1638` was reworded. **Also left by PR-A2**, Vertex-route and GCS wording in code it did not otherwise touch: `curriculum/early_stopping.py:168`, `file_io.py:4`, `wandb_integration.py:84-85` and `tb_sync.py:112` (with :4 above). Line numbers in this row are at PR-A's head unless marked; after PR-A2 those in `train_base.py` past :2084 and in `reporting/csv_output.py` past :95 moved, so re-derive them. *As carried out by PR-B (2026-09-28):* the `foot_contact_*` test comment went with `test_species_integration.py`'s MJX block, and PR-B rewrote the `config.py:175` sentence, so both items are done. **Also left by PR-B**, JAX and MJX wording in code it did not otherwise touch (line numbers at PR-B's head): in files its acceptance keeps unchanged, `result_bundle/gate_verdict.py:18,62-63` (they name the retired `save_jax_stage_artifacts`), `action_filter.py:12-13` (it names the deleted `_PLANT_INTERFACE_CONFIG_FIELDS`) and `result_bundle/evidence.py:179,762`, beside `evidence.py:565` and `hashing.py:38` above; elsewhere, the `base_env.py` comments that describe a live MJX step or settle (:99, :109, :133, :884, :1116, :1160, :1359-1366), the `reward_functions.py` module docstring (:6-18), `test_widen_checkpoint.py:20` and, optionally, the `jax_training` pair at `species_catalog.py:1112`. Dead-code candidates after PR-B, uncalled or reachable only from tests: `gate_schema.apply_backend_overrides` and `has_backend_overrides` (inside §4.9's keep range, so CU-7 decides), `stance_gate.episode_unsupported_duty`, and `reward_functions.check_nosedive_termination` and `reward_height_maintenance`. Also from PR-B's review: the quadruped-detection MEDIUM in KNOWN_ISSUES closes with a one-root assertion (exactly one of `torso` / `pelvis` in `body_ids`) in the plant contract's MJX probe, `policy_layer._jax_policy_interface_payload`, which is not hashed, so no digest moves (checked with `plant_contract --check`). *Split 2026-09-29 (§2 row 20):* CU-7a takes the dead code (with the gym entry points, their KNOWN_ISSUES bullet and the ROADMAP fix), the import cost, `imageio` and the one-root assertion that closes the quadruped-detection MEDIUM; CU-7b takes the retired-backend wording that PR-A, PR-A2 and PR-B left and the fingerprint docstring, with CU-16's stage-TOML comments folded in; the sha256 regex and the validators go to CU-8c. **As carried out (2026-09-30): CU-7a.** Deleted, with the 13 tests that covered only them: `gate_schema.apply_backend_overrides` and `has_backend_overrides`; `stance_gate.summarize_stance_panel` and `episode_unsupported_duty` (the live reduction is `stance_panel_from_episode_duties`, and the stale references in `manager.py` and the "funnel through" text in `stance_gate_report.py` now name it, while `stance_report.py`, which already did, drops its claim that `environments.shared` imports the reporting package eagerly; the stance tests of live behavior feed it the same post-settle reduction through a local helper, and `TestSettlingWindow`, which tested only `episode_unsupported_duty`, goes); `recovery_evaluation.paired_success_differences`; `reward_functions.reward_height_maintenance` and `check_nosedive_termination`; `stage_layout.FIGURE_NAMES` with `iter_figures` and `iter_generated_artifacts`, which only read it; `StageManifest.by_position`, `stage_manifest.STAGE_MANIFEST_SCHEMA`, `constants.DEFAULT_FRAME_SKIP` and the no-op `environments.register_all`. The two override helpers sit inside §4.9's keep range, whose reason covers the validation of recorded blocks only: they had no caller after D-D17, no committed stage TOML in the full history ever held a `[curriculum.jax]` table, and the validation stays (§4.9). The gym entry-point groups go with their KNOWN_ISSUES bullet, the CONTRIBUTING step that told a new species to add one, and the ROADMAP item's wrong id: gymnasium 0.29's plugin loader reads only the `gymnasium.envs` group, never these two, and 1.0 onward has no loader (checked in the published wheels of 0.29.0, 0.29.1, 1.0.0, 1.1.1, 1.2.0, 1.2.2 and 1.3.0; the floor is `gymnasium>=0.29.0`); the envs still register on `import environments`. `environments/shared/__init__.py` keeps only its docstring: it re-exported 19 names that nothing imports through the package, and importing any `shared` submodule runs it first, as every species env does. Measured in CI's SB3 environment, `import environments` took 3.45-3.55 s and loaded 2,480 modules (SB3, torch and wandb among them), against about 0.36 s and 569 modules after; without SB3 the time is within noise (638 modules to 568). The survey's 1.9 s to 0.3 s was measured in a different environment. `imageio>=2.16.1` joins the `viz` extra (compsognathus `view_model.py --video` imports `imageio.v2`, which needs 2.16; 2.16.0 was yanked), and the compsognathus README and the script's docstring point to the extra. The plant contract's MJX probe requires the registration to map exactly one root body, the one the observation schema names, and a new test fails without it; the probe is not hashed. Script hygiene: the usage docstrings of seven species scripts whose old usage (`python <name>.py`) fails without an installed package (`view_model.py` for brachiosaurus, dibothrosuchus, trex and velociraptor, `test_actuators.py` for dibothrosuchus, trex and velociraptor) give `python -m environments.<species>.scripts.<name>`. Not taken: the optional lazy `curriculum/__init__.py`, which with the reduction takes `import environments.shared.curriculum.gate_schema` in the SB3 environment from about 3.5 s to 0.4 s, but needs three of the SB3 notebook's imports pointed at their submodules, because CU-4's import check does not count names bound under `if TYPE_CHECKING:`, so it stays optional; `StageEntry.has_parent` (its test is the bit-identity proof of the live `_derive_legacy_edges`), `stance_report._commanded_angle` (the oracle of seven tests of the live `_reduce_action_stats`), `config.RESUME_LINEAGE_KEYS` and `ancestors.ANCESTOR_RECORD_KEYS` (schema vocabulary that tests pin); `TRexEnv._foot_load_imbalance` (in a byte-hashed species env file, so deleting it moves 22 behavior lines; CU-12's), compsognathus `render_head_camera` and `onboard_readings` (documented API) and `behavior_certification.py` (PR-13's). Measured (`git diff --numstat` against `19762d5`, the code commit): 37 files, +103 / −434 (library +31 / −265 in 13 files, tests +40 / −104 in 9, scripts +18 / −18 in 9, docs +12 / −34 in 5, `pyproject.toml` +2 / −13). The digest-snapshot check, run with CI's command, reports the golden current (848 lines, 0 errors), and `plant_contract --check` reports the manifest current, so no digest moves. | Pure deletion. | about −330 | LOW | Any time. *2026-09-29: CU-7a any time; CU-7b after CU-7a and ROW-16 (§3.1 item 4).* *CU-7a landed as #572 (2026-09-30).* |
| CU-8 One derivation, one reader | C9 | Generalise `stage_task_fingerprint` to all 6 sites, which retires the text pin at pins:487-506. Make `ignored_hyperparameter_edits` public; the notebook copy lacks its guard. One `stage_config.json` reader. Move the sidecar resolver and `_ensure_sb3` into `policy_loading`. Keep one `FINGERPRINT_BACKEND` constant (today at `freeze_recovery_gate.py:127` and `widen_checkpoint.py:153`), one constant for the 33 `"stable-baselines3"` literals, and one `REPOSITORY_ROOT`/`_SHARED_ROOT` (defined in both `result_bundle/constants.py` and `plant_contract/constants.py`). Keep the values unchanged: the backend string enters `task_sha256`, and repo-relative paths enter `behavior_identity`. *Split 2026-09-29 (§2 row 20):* CU-8a takes the one derivation (the stage-level fingerprint helper for the six sites, one `FINGERPRINT_BACKEND` and the public ignored-edits check), CU-8b moves `_ensure_sb3` and the sidecar resolver into `policy_loading`, and CU-8c takes the one `stage_config.json` reader and one `REPOSITORY_ROOT`, with CU-7's one sha256 regex and one set of validators. *2026-09-30 (CU-9, #574): `freeze_recovery_gate.py` and `widen_checkpoint.py`, which hold CU-8a's two `FINGERPRINT_BACKEND` sites (now :128 and :153), count toward coverage.* | PR-9 gets a single derivation, and PR-10 gets `policy_loading`. | about −100 | LOW; acceptance is the committed `task_sha256` tests | After PR-A; before PR-9 |
| CU-9 Coverage of certification code | critic item | `*/scripts/*` and `environments/shared/harnesses/*` are coverage-omitted (`pyproject.toml:164-169`). That hides `freeze_recovery_gate.py` (892 lines), `widen_checkpoint.py` (1,412) and `backfill_gate_verdict.py` (469). Replace the blanket omits with explicit entries, and move `brace_controller` to break the cycle. Keep `harnesses/digest_snapshot.py` (added with this plan) in the explicit omit list. Close or rebase #527 first (decision 19); it edits `freeze_recovery_gate.py`. **As carried out (2026-09-30):** the blanket omits become 21 named entries (30 files, `pyproject.toml:139-183`), each group with its reason; 25 files that the blanket omits hid now count, adding 2,165 statements (877 missed): `freeze_recovery_gate.py` 310 / 56 (81.9 percent), `widen_checkpoint.py` 567 / 73 (87.1), `backfill_gate_verdict.py` 181 / 19 (89.5); the union reads 19,626 statements, 2,405 missed, 87.75 percent (was 91.24, CI 91). Stay omitted, by name: the display-bound viewers, actuator drivers and env smoke checks with their per-species wrappers; `digest_snapshot.py` (its own CI step); the Compsognathus MJCF generators and review tools; the PPO probe; the investigation scripts that feed no gate (five shared ones and the velociraptor wrapper). `calibrate_recovery.py` (0 percent) counts, since it reproduces the recovery judge, and so does `stance_quality_baseline.py` (0 percent), which measures the statue-derived stage-1 gate constants. `brace_controller` moved to `recovery_evaluation.py` (3 SCCs of the import graph become 2). `.coveragerc`, `.coveragerc.toml`, `setup.cfg` and `tox.ini`, which coverage reads before `pyproject.toml`, join both path filters of `python-ci.yml`, so a PR that only adds one runs the pin. #527 was closed on 2026-09-26 (§2 row 19), so there was nothing to rebase. Line drift: the row's `pyproject.toml:164-169` (at `f850815`) was :139-145 at `d4496a5`, and `widen_checkpoint.py` is 1,417 lines, not 1,412 (CU-1 added 5). | Certification code should count toward `fail_under = 70`. | small | LOW (re-measure) | After PR-B. *Landed as #574 (2026-09-30).* |
| CU-10 `train_curriculum` body onto `train()`'s helper | C10 | Name `eval_env_seed`. Validate old against new under a frozen clock. Optional second PR: split `train_base.py` into trainer, curriculum runner and post-training panels. **Also fixes the curriculum horizon defect** (added 2026-09-27 by the maintainer, when cleanup PR-A2 recorded it in KNOWN_ISSUES under Training / RL): `CurriculumCallback._eval_horizon` reads `max_episode_steps` from the TOMLs `CurriculumManager` re-reads, not from the overridden stage config the panel runs, so hand it the overridden horizon, with a test that a `--override env.max_episode_steps` stance stage can pass. | Removes the second copy of the stage body, and the horizon defect with it. | about −100 | LOW-MED | Lowest priority |
| CU-11 Reward/info/termination golden | C11 | 21 species×stage captures, stored as digests plus a summary. Quantise the values, or run a same-machine A/B. Reuse the `reset_golden` helpers. **As carried out (2026-09-30):** a `reward` section of the digest-snapshot harness (§4.4). The committed golden gains four lines per stage, 84 in all (the first 848 lines are unchanged): a summary (steps, end and reward sum per stepped part), the ends of the state probes, a digest of the discrete records (info keys and their order, flags, reasons, the resets' generator state) and a digest of every value rounded to 6 decimals (7 significant digits from 10 up). The capture, per stage from `reset(seed=1042)`: 300 steps of uniform float32 noise (0.2 for brachiosaurus and dibothrosuchus, 0.1 for trex and velociraptor, 0.01 for the compsognathus pair; `default_rng(7)`), then a held-sign kick; the roll ends every stage's episode (compsognathus recovery's during the noise, at step 162); an unseeded second reset; zero action through the pushes of the three recovery stages; one-step probes (the target moved onto the effector, the root lifted, rolled, both) and a three-step horizon; and twelve root and neck poses sized by the stage's own thresholds (applied on top of the reset pose) plus a non-finite velocity, scored without a physics step, which reach tail and head contact, body contact through the live contact scan, dibothrosuchus's nosedive and `nonfinite_state`, none of which a rollout reaches. The reset record is `reset_golden._reset_record`. CI's existing digest step checks the new lines (the full harness took 56 s locally, against 40 s before). `--exact` prints one bit-exact digest per part and stream for the 21 stages and the 66 behavior recipes (a zero-action episode of up to 650 steps, nine unseeded resets, and on terrain the root moved off the flat apron): the same-machine base/head diff of CU-12, PR-8 and PR-9, run by hand. The rounded golden cannot see an ulp (measured: CU-12's foot-force deletion in the naive sum order moves 87 `--exact` lines (50 in the stages, 37 in the trex behavior recipes) and no golden line; with the base order pinned, none). Measured for the design: the 84 lines were identical under Python 3.11 with numpy 2.4.6 and 3.12 with 2.5.3, with and without SB3, under an AMD-like numpy/OpenBLAS dispatch and with a libm without FMA; `--exact` differs across the Python and numpy versions and the CPU and libm dispatch (not with and without SB3) and is bit-identical across processes on one machine (the base/head diff of the stages printed 21,251 identical lines). Departures: the row's +150–200 becomes +548 / −17 lines of harness and tests (`git diff --numstat` against `fd8ba78`, with the review's fixes: harness +373 / −12, tests +175 / −5, and the golden's 84 lines), about 145 of them the state probes (the design without them measured +372), which alone catch a slip in the contact-category scan and a removed non-finite check; the behavior recipes are `--exact` only, so PR-8's check is `--exact` with the recipes (§3.4). | No digest covers reward or termination code, and PR-8/PR-9 need a "no number moved" check. | +150–200 | LOW | Before PR-8. *Landed as #573 (2026-09-30).* |
| CU-12 Species env dedup | C12 (minus MJX scalars) | Reward-term helpers replace the 8 thin `_compute_*` wrappers. One contact query and one height/tilt termination prefix. Drop the foot-force overrides, pinning the base order `group[0]+sum(group[1:])`. A home-keyframe helper. Delete the test-only trex accessor. The cross-backend scalars need no work: PR-B deletes their MJX copies, and the 7–9 keys left in the frozen `mjx_config.py` registrations are not edited. Leave every token-hashed method alone (§5.6). | PR-9's constructors edit the same files. | about −300 | LOW (CU-11 proves it). A byte edit to a species env file moves the pilot `behavior_identity`; those bundles are evaluation-only (D-D9), none is on Drive, and PR-9 deletes that identity. Acceptance: CU-11's golden, plus a harness diff limited to the `behavior` identity lines of the species touched. *2026-09-29 (D-D22): that diff is CU-12's own update of `configs/digest_snapshot.generated.txt`, and CI's `--check` shows it.* *2026-09-30 (CU-11, #573): CU-11's golden is the `reward` lines of that file, which CU-12's `--write` must leave unchanged; its "pinning the base order" claim is checked by the harness's `--exact` on the base and the head, with the behavior recipes (see its `--help`), which the rounded lines cannot see (measured: the naive order moves 87 exact lines, 50 of them in the stages, and no golden line).* | After CU-11; before PR-9 |
| CU-13 Stage-TOML `extends`, step 1 | C13 | D-D5 pulled forward: recovery extends stance for `[env]`, `[ppo]`, `[sac]` and `[stage]` in trex, compsognathus and compsognathus_robot. The survey critic confirmed that `{**stance, **recovery}` reproduces those tables with identical key order and that only `[curriculum]` differs. Never inherit `[curriculum]`. Keep a permanent 21-stage digest snapshot, which the digest-snapshot harness (§4.4) produces. *2026-09-29 (D-D22): the committed golden, `configs/digest_snapshot.generated.txt`, holds it, and CU-13 leaves its `stage` lines unchanged.* Step 2 (compsognathus_robot ← compsognathus, after rerouting the direct TOML readers) waits for PR-12 and session 6. *Session 6 finished on 2026-09-28 (compsognathus_robot's walk certified), so step 2 waits only for PR-12.* *Added 2026-09-29:* CU-13 must keep the `foot_contact_*` lines of `configs/trex/recovery.toml` (`foot_contact_gate` and `foot_contact_weight`, two of the six keys §4.2 keeps, and `foot_contact_saturation_force`) as explicit redeclarations, equal to stance's, rather than inheriting them. For trex it cannot inherit `[sac]` without moving two digests: `configs/trex/recovery.toml` is the only stage TOML without a `[sac]` table, and giving it stance's moves trex recovery's `hyperparameters_sha256.SAC` and `stage_config_view_sha256.SAC` (measured 2026-09-29 at `7012483` with the harness and `--skip-behaviors`: those two of 440 lines differ, the rest are byte-identical), so for trex the critic's check above holds for `[env]`, `[ppo]` and `[stage]` only. | Tests PR-11's mechanism against a known answer. | about −130 | LOW | Before PR-11; amend D-D5 |
| CU-14 Workflow structure | critic items + decision 10 | Collapse the 18 matrix jobs to 6 (`python-ci.yml:216-220`). Anchor the byte-identical `paths` lists (:6-29, :38-61). Install `.[test]` rather than `.[dev]` in the matrix. Drop the plant-contract job's duplicate pytest step (192 s), keeping its `--check`, baseline, digest-snapshot (D-D22, added by ROW-16) and wheel steps. Drop `test_phase_c_interface.py` from the SB3 list (145 s). Apply decision 10, and fix the "smallest plant" comment at `python-ci.yml:313-314` (the robot is the costliest plant for those tests). Update branch protection if job names change (decision 18). After the plant-contract pytest step goes, the frozen-core pin test runs only in the `test (shared, …)` matrix (§4.3). *Split 2026-09-29 (§2 row 20):* CU-14b goes first and takes decision 10 (c), the per-process plant-identity cache; CU-14a, after CU-4 and CU-14b, takes the workflow YAML and the test pins it breaks, and decides 10 (a) from CU-14b's durations. **As carried out (2026-09-29): CU-14b** (§2 row 10 (c)). `current_plant_identity` (`plant_contract/manifest.py`) keeps one slot per species, `_IDENTITY_CACHE`, consulted after the plant versions and species entries are read and an unknown species is refused. Its key (`_identity_cache_key`) holds everything the build reads that can change in a process: the species entry (as sorted JSON), the `PlantVersion`, `constants.REPOSITORY_ROOT`, the resolved model path, the source-closure digest, the environment class and `mujoco.__version__`; when the closure cannot be read the key is `None` and the call builds uncached, raising its own error. A hit also re-runs the policy layer on a fresh environment (`_policy_interface_unchanged`: the policy-interface digest, the observation and action dimensions and the training backends) and rebuilds on any difference. A build is stored only when it hashed the same closure bytes the key read and its entry equals the committed manifest's (`_is_committed_entry`, which never raises), so a stale plant rebuilds on every call; the `verify_generated` comparison with the committed manifest runs on every call, hit or miss. `build_plant_manifest`, `--write` and `check_plant_manifest` (CI's `--check`) never read the cache, so the check always recomputes; `clear_plant_identity_cache()` is exported from `plant_contract`. Kept on purpose, a known limit: Python code is not in the key. The policy layer is re-checked on every hit, but a hit does not see an environment-code patch, made in a live process, that changes the compiled model (not the MJCF); the live checks compare physics instead, `validate_environment_plant` for every env training's `make_env` builds with a plant identity and those the evaluation paths build, and `validate_compiled_plant` for every behavior env and every env `config.build_env` builds (the frozen recovery gate's nulls, brace and policy panel, the recovery calibration, the zero-action baseline, `widen_checkpoint` and the report scripts; the review follow-up below), and a process that patches environment code and builds envs another way (a `make_env` without a plant identity, as `compsognathus/scripts/probe_ppo_updates.py` does) calls `clear_plant_identity_cache()`. A physics re-check on every hit was measured at 1.2 s of the robot's 7.4 s build, so it was left out. Measured locally in CI's SB3 environment, without coverage: compsognathus_robot's first call 7,390 ms and repeat calls 109–121 ms; compsognathus 429 ms → 34 ms. CI's three SB3 pytest steps, run locally with CI's flags (under coverage, at reduced depth), took 2,609 s at `c69a169` and 855 s with the cache (the integration step 36:08 → 11:05; the robot's PPO freeze rehearsal, `test_profile_backed_recovery_freeze_rehearsal_cannot_certify[ppo-compsognathus_robot]`, 176.6 s → 15.3 s), and `digest_snapshot.py` 207.9 s → 54.3 s; CI's runners differ, so the PR's own CI durations are the record, and CU-14a takes decision 10 (a) from them. Each guard is caught by a mutation: a cache that never hits, hits that skip the manifest comparison, storing every build, a key without the species entry or with three fields only, a key without the plant version, the root and the resolved model path together (either alone is redundant: the resolved path holds the root, and the entry and the root give the path), the closure, the environment class or the MuJoCo version, the key's error handling, a manifest build that reads the cache, no policy re-check and no same-bytes guard. Measured (`git diff --numstat` against `c69a169`): 4 files, +376 / −2: `manifest.py` +93 / −2, `plant_contract/__init__.py` +2 (the export), `test_plant_contract_identity_cache.py` +269 (15 tests) and a `docs/PLANT_CONTRACT.md` paragraph (+12). CU-14b is code, not YAML (the Size cell is CU-14a's), and it edits `manifest.py` outside any hashed payload (§4.2). Validated on the code commit: the digest-snapshot harness, run with the optional backends blocked, printed 848 lines with 0 errors, byte-identical on the base `c69a169` and the head; `plant_contract --check` and `--check --baseline` (against `c69a169`'s manifest) report "Plant manifest is current"; mypy finds no issues in 314 source files (from 313) in all three environments; `pytest --collect-only environments` collects 3,966 tests (from 3,951), and the full suite passes (3,965 passed, 1 skipped). *Review follow-up (2026-09-29):* the envs `config.build_env` builds were never live-checked: the frozen recovery gate, the recovery calibration, the zero-action baseline's `preflight` and `widen_checkpoint` relied on `current_plant_identity` alone, and the report scripts (and the zero-action baseline's command-line `report`) had no plant check at all, so with a warm cache a mid-process patch of `CompsognathusEnv.__init__` (`functools.wraps`, the pelvis mass times 1.5) was accepted by `build_env`, `load_recovery_calibration`, `freeze_recovery_gate` (which wrote its resolution), the brace derivation, `roll_policy_panel` and the zero-action baseline's `preflight`; at `c69a169` the identity rebuild refused the calibration, the freeze, the panel and the preflight (the brace runs only inside the freeze). `build_env` now compares each env's compiled model with the identity (`validate_compiled_plant`: the physics digest and nq/nv/nu; the identity read with `verify_generated=False`, since the callers that certify compare it with the committed manifest in their own calls), and all six refuse. Measured locally, a warm `build_env` takes 0.03–0.05 s for the five small plants (from about 0.01 s) and 1.2–1.5 s for the robot (from 0.06–0.09 s), and the first `build_env` per species in a process that has not built the identity also pays that build (about 7.4 s for the robot); all 21 species-stage envs are accepted unpatched, and the robot's freeze rehearsal takes 13.5–14.1 s (from 8.5–8.8 s). A second new test covers `_is_committed_entry`'s error handling (an undecodable manifest and one of the wrong schema, read with `verify_generated=False`). `git diff --numstat` of the follow-up: `config.py` +14 / −2 and `test_plant_contract_identity_cache.py` +43 / −1 (18 tests; the suite collects 3,969 and passes, 3,968 passed and 1 skipped; mypy finds no issues in 314 source files). The `build_env` test fails on the cache commit's `build_env`, with the check removed, with the declared model checked in place of the built env and with the refusal swallowed; the error-handling test fails with the `try` removed. The digest-snapshot harness, run with the optional backends blocked, printed 848 lines with 0 errors, byte-identical on the base `c69a169` and the follow-up. | `test-sb3` is the critical path after PR-B. | YAML only | LOW | After PR-B |
| CU-15 Drive summary reader | C16 (reduced) + the rest of C15 | No sweeps are written after PR-A. NB2 (current-layout `sweeps/<algo>_<ts>` folders are skipped) has no present impact: a read-only Drive listing on 2026-09-26 found three `sweeps/` folders (created 2026-03-25..28) holding ten legacy `stage<N>_<algo>_<ts>` folders, which the summary reads, and no current-layout folder. PR-A drops its KNOWN_ISSUES entry. Give the Drive summary's setup cell the SB3 bootstrap: `REPO_REF` with fetch-and-detach instead of a `--depth 1` clone of the default branch, the three-clause `IN_COLAB`, and a guarded Drive mount. Optional: move the 786-line parser into a tested `reporting/run_index.py`, with pandas imported lazily and declared. | Seven reader patches since August. The bootstrap records which code the summary ran. | +150–300, all tested (optional), plus the small bootstrap edit | LOW | After PR-A |
| CU-16 Docs correctness | C20 | Add a status appendix to [reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md](reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md) and move its open findings (CF6, OP3, SS3/4, TC8) into KNOWN_ISSUES; mark JX2, JX4, JX7, JX9, NB3 and CF1 "retired by D-D17". List all 6 env ids in the API overview. Fix the backfill tool path. Mark STAGE1_SPLIT_PLAN's status, re-label RECOMMENDATIONS.md as a dated 2026-03 snapshot (`docs/README.md:43` says "Active"), and mark WEBSITE_PLAN complete (:47), moving its logo-SVG item to KNOWN_ISSUES. Update ROADMAP, the README and the website milestones. Correct the living-doc sentences that say pilot or certificate data exists on Drive (none does); the CHANGELOG copies are history, but add a one-line correction to the `Images/` entry (`CHANGELOG.md:3326-3329`, :3388-3391 at the follow-up), which is wrong about where the removed GIFs survive, in the new `[Unreleased]`. Assets: the three `website/static/videos/raptor_stage*.mp4` videos and their three posters (1,974,435 + 19,787 B) are unreferenced and have no other copy. `sac_apex.gif`/`ppo_apex.gif` share blobs with `results/velociraptor/{sac,ppo}/stage3_strike.gif` (23,558,176 B), which only `results/README.md:12-17` lists, so delete both copies or neither. `results/velociraptor/ppo/stage1_balance.gif` and `raptor_balance_ppo.gif` share a blob, but both are referenced: no action. The orphan PWA icons (62,739 B) and `compsognathus/data/robot_camera_view.png` (32,616 B) can go. The heightfield and render-crash entries are already added with this plan (§3.5). *After PR-A2:* CF6's file (`configs/trex/sweep_ppo.json`) left with PR-A and OP3's last entry point (`scripts/setup_vertex_ai.sh`'s stage prompt) with PR-A2, so both are retired by D-D17 and do not move into KNOWN_ISSUES. *After PR-B (2026-09-28):* the stage-TOML comments that still mention JAX or MJX outside the deleted `[jax]` tables are CU-16's (line numbers at PR-B's head): brachiosaurus `stage1_balance.toml:48`, `stage2_locomotion.toml:30,49` and `stage3_food_reach.toml:26`; dibothrosuchus `stage1_balance.toml:12-14,59`; trex `behavior.toml:25`, `locomotion.toml:35,65` and `stance.toml:13,110,255,411`; velociraptor `stage2_locomotion.toml:31,50` and `stage3_strike.toml:27`. PR-B's acceptance limited its stage-TOML diff to the `[jax]` tables, so they wait. Edit the comments only (three of those lines carry `foot_contact_*` keys, which stay; §4.2), and run the harness on base and head. The recipes plan's vocabulary row for a node (`BEHAVIOR_RECIPES_PLAN.md:135`) still lists `[jax]` among a stage TOML's tables, which `load_stage_config` now refuses; correct it here. *Split 2026-09-29 (§2 row 20):* CU-16a, after CU-7a, takes the docs text (the review appendix, the KNOWN_ISSUES moves and the living-doc corrections above, the CHANGELOG line and `BEHAVIOR_RECIPES_PLAN.md:135` included), CU-16b takes the orphan assets (and the GIF pair, if decided), and the stage-TOML comments go to CU-7b. | Living docs must stay true. About 2.1 MB is freed without the GIF decision. | about +150 | None | After PR-B |
| CU-17 Docs shrink | C21 | First move the only copies of unique text: PR-7's parity reason, PR-4's reproducibility note and the NEXT_STEPS §5 operational choices. PR-3b's list is superseded; say so. Then: landed consolidation-plan PR bodies become pointers (−500), NEXT_STEPS shrinks (−185), KNOWN_ISSUES entries over 40 lines shrink (−275), and the README Quick Start folds into the recipe pages (−90). Rewrite PR-15's text. Adopt the rule that PR landing status lives only in the consolidation status table and the CHANGELOG (a `docs/README.md` convention). *Since PR-B (2026-09-28) the consolidation plan's PR-3b paragraph says it is superseded.* | Shorter living docs. | about −1,050 (docs) | None | Last |

**Optional, unscheduled:** split the diagnostic stance probes into `reporting/stance_probes.py` (about 1,216 lines
move, no net change).

**Optional PR-C** is the comment-only MJX cleanup in the species env files:
`trex_env.py:93-94,167,182,284,384,410,450-451,772`, `brachio_env.py:283`, `dibothrosuchus_env.py:103,168,293` and
`raptor_env.py:252`, plus the `foot_contact_*` JAX-only note in `TRexEnv` and the stale observation list in the
`brachio_env.py` module docstring. `behavior_identity` hashes these files' bytes, so PR-C moves the pilot identities
of the species it touches. It can fold into CU-12, which moves the same identities under the same acceptance, or wait
for PR-9, which deletes the `behavior_identity` sources block. It can also be folded into PR-15.

**Done since the survey:** C4 (RESUME safety, the quick-test tree, resume before the chain loop) landed as #558 and was
hardened by #559.

### 3.3 Survey items made moot by the retirement

- **Whole waves:** C3, C14 (including NB3 and the Ray "Apply Best Hyperparameters" printer), C17 and C18.
- **Most of a wave:** C15, except the Drive-summary bootstrap (`REPO_REF`, the three-clause `IN_COLAB`, a guarded
  Drive mount), which moves to CU-15.
- **Parts of waves:** C1's Ray half; C2's job-list additions and `google.auth` stub; C7's
  `Transition`/`batched_sample`; C12's cross-backend scalars (PR-B deletes their MJX copies); C16's NB2 fix, which
  now affects historical folders only.
- **Critic and survey items outside the waves:** the sweep-row builder; the Ray worker via `train()`; the duplicate
  `PLANT_IDENTITY_FILENAME`; "C9 before C14"; C3's trex-directory narrowing; C17's `jax.md` examples;
  `cloudml-hypertune`; the `evaluate_policy_cpu` rollout loop and its video hook; the Vertex launcher's stage-runner
  dedup.
- **Gap-review findings:** CU-16's appendix marks JX2, JX4, JX7, JX9, NB3 and CF1 "retired by D-D17".
- **Live defects (§5.3):** five disappear with PR-A/PR-B and #558 fixed two; `render_mode='human'` survives the
  retirement (CU-2 fixed it; landed as #568 on 2026-09-29) and NB2 shrinks to historical folders (CU-15).

### 3.4 Prerequisites of consolidation PR-8..PR-15 (updated for the retirement)

| Planned | Needed for terrain? | Land first | Effect of the retirement |
|---|---|---|---|
| PR-8 selector, constants, normalisation | Yes | CU-11 (golden as acceptance) | Unchanged. *CU-11 (#573, 2026-09-30): its golden covers the 21 stage envs, which never import PR-8's modules; PR-8's numbers are checked by `--exact` with the behavior recipes (the terrain family of episodes 0-9, the command stream, the clearance off the apron).* |
| PR-9 phase D hook, identity = fingerprint | Yes | CU-8, CU-12 | Smaller: no `MJXEnvConfig` edit (`mjx_env.py:274-279`), no MJX suite. It must still leave the frozen core untouched and pass `plant_contract --check`. *Since PR-B (2026-09-28):* the `MJXEnvConfig` dataclass and the MJX suite are gone; the `TYPE_CHECKING`-only `MJXEnvConfig` alias in the frozen core is not PR-9's to edit (§4.3) |
| PR-10 command-column warm start | Yes | CU-8, CU-1 (SB3-aware mypy) | Unchanged |
| PR-11 follow/terrain nodes | Yes | CU-13; PR-11's Breaks line must name `test_every_committed_stage_is_command_mode_none_in_phase_c` (`test_sb3_notebook_pins.py:2333`, :2559 at the follow-up); decisions 11, 13, 14 and 15 | New TOMLs carry no `[jax]` (after PR-B, `config.py:224-231` rejects it) and need no MJX mirror. *Since PR-B (2026-09-28):* the rejection is live (`config.py:225-231` at PR-B's head), and `test_config.py`'s `test_a_retired_jax_table_is_rejected` pins it |
| PR-12 rest (delete the pilot) | No; it is cleanup that follows PR-11 by design | — | It updates the harness's behavior section when the 66 recipe TOMLs go, and conflicts with the retirement only in `python-ci.yml` |
| PR-13 terrain_command gate | Yes, to certify terrain | CU-3 (landed as #562 on 2026-09-26, D-D20); decisions 6 and 12 | Invariant 10's new case covers only the SB3 manager. Land PR-B first, because it deletes `test_gate_dispatch_fail_closed.py:161-351`. *Deleted by PR-B (2026-09-28).* |
| PR-15 | Mostly cleanup | CU-4, CU-17 | No JAX or sweep docs to fold. CHANGELOG Removed inherits the D-D17 entries |
| PR-3b (JAX/SB3 job) | No | — | Superseded by PR-B. The next CI lever is inside `test-sb3` (decision 10). *PR-B deleted the JAX job (`test-jax-cpu`), 2026-09-28.* |

### 3.5 KNOWN_ISSUES entries added with this plan

The docs PR that adds this plan also adds these entries to [KNOWN_ISSUES.md](KNOWN_ISSUES.md) and corrects seven
existing ones. Each was checked on 2026-09-26 at `f850815` and at the follow-up (KNOWN_ISSUES is byte-identical at
both), and each key claim was re-checked against `be63a58` before the entries were added (executed for the
heightfield floor comparison, the handoff pair, the `train --load` guard, the render crash, the JAX thresholds, the
snout proximity, the Ray `ent_coef_end` replay, the `sim_dt` default, the gym entry points, the ruff pin and mypy;
read for the rest); all twelve still hold. Each entry says whether its failure was reproduced, executed or read from
the code, and cites code lines at `be63a58`. The PR that removes a cause also deletes its entry.

| Entry (section, severity) | Plan item | Leaves KNOWN_ISSUES with |
|---|---|---|
| Every certified walker survives the plane and falls on a flat heightfield; it also records velociraptor's map exits (Training / RL, HIGH, terrain blocker) | Decisions 11 and 13; §5.1 | The heightfield-contact fix that follows the investigation |
| The best and robust-best handoff pairs are written straight to the mount (Training / RL, MEDIUM) | Decision 7 | CU-3 (deleted by it, #562, 2026-09-26; D-D20) |
| Nothing on disk records the trunk a session resolved (Training / RL, LOW) | Decisions 4 and 6 | The PRs that take both decisions |
| `train --load` with the default `resume_same_stage` writes into a judged stage directory (Training / RL, MEDIUM) | Decision 5 | The library guard |
| `render_mode="human"` crashes on the first step (Training / RL, MEDIUM) | Defect 2 | CU-2 (deleted by it, #568, 2026-09-29) |
| The JAX command-line curriculum ignores `min_avg_forward_vel` (Training / RL, MEDIUM (JAX)) | Defect 3 | PR-B (deleted by it, 2026-09-28; D-D17) |
| MJX training never pays dibothrosuchus `snap_snout_proximity_weight` (Training / RL, MEDIUM (JAX)) | Defect 4 | PR-B (deleted by it, 2026-09-28; D-D17) |
| Every Ray Tune PPO trial raises `TypeError` on `ent_coef_end` (Sweeps / infrastructure, MEDIUM) | Defect 1 | PR-A (deleted by it, 2026-09-27; D-D17) |
| Stage summaries built from `evaluations.npz` assume a 0.01 s control step (Sweeps / infrastructure, LOW) | CU-2 | CU-2 (latent after PR-A, which rewords it, 2026-09-27; deleted by CU-2, #568, 2026-09-29) |
| The Drive summary skips current-layout sweep folders, gap-review NB2 (Notebooks, LOW) | Defect 6; CU-15 | PR-A (deleted by it, 2026-09-27; D-D17) |
| Pre-commit pins ruff 0.4.4 while CI installs the latest ruff (Testing / CI, LOW) | CU-1 | CU-1 (deleted by it, #561, 2026-09-26) |
| CI's mypy never sees SB3 or torch types (Testing / CI, LOW) | Decision 8 | CU-1 (deleted by it, #561, 2026-09-26) |
| The command-line curriculum judges `stance_quality/v1`'s full horizon against the stage TOML, not an overridden `env.max_episode_steps` (Training / RL, MEDIUM; added by PR-A2, #565, 2026-09-27) | — | CU-10 (the maintainer's choice, 2026-09-27) |

Corrected existing entries:
- The "Curriculum gates" divergence bullet: the JAX CLI checks reward and episode length, and ignoring velocity and
  success is a defect, not a divergence. PR-B deletes it with its section. Deleted by PR-B (2026-09-28).
- The LOW re-entry entry now names the bundle-verification cell, which #558 renamed from "Cleanup". It stays.
- The stage-3 sweep-keys entry is replaced: 18 keys in 7 files (defect 9). PR-A deletes it.
- The `ray_orchestration.py` entry (1,006 lines; `export_best_trial` has a caller) and the `ray_tune_sweep.ipynb`
  bullet, which gains gap-review NB3. PR-A deletes both.
- The notebook-pins bullet: the notebooks now pin SB3 and JAX. PR-A drops its Ray clauses and PR-B its JAX clauses.
  Both done (#564; PR-B, 2026-09-28).
- The gym entry-point line: verified dead. CU-7 deletes it. Deleted by CU-7a (#572, 2026-09-30).

The `foot_contact_*` LOW (the JAX-only knobs entry under Training / RL) is not edited now; PR-B rewords it (§4.6, §7).
Reworded by PR-B (2026-09-28). The speed and map mismatch of decision 13 has no entry of its own; the heightfield
entry records velociraptor's map exits.

## 4. Backend retirement plan

### 4.1 Why a frozen core stays

For trex, velociraptor, brachiosaurus and dibothrosuchus, the policy-interface payload does two things: it hashes the
source tokens of MJX functions, and it runs an MJX observation probe. In `plant_contract/policy_layer.py`:
- the declared backends are read at :355;
- the payload is built at :364-391 (`_jax_policy_interface_payload`, :152-296);
- the home reset is hashed at :398-404.

A plain delete makes `plant_contract --check` exit 2 with "cannot import MJX plant registration
environments.brachiosaurus.mjx_config" (measured), and it moves four species' digests. The compsognathus pair is
SB3-only (`compsognathus_env.py:29`), so no JAX file feeds its digests.

A species leaves the core only by declaring itself SB3-only inside its next deliberate `policy_interface_revision`
bump, which switches it to the literal branch (`policy_layer.py:392-397`). Once all four have done so, delete the core
and its pin test.

One such revision is already queued: the reset height-channel removal, held for "a revision that already plans a trunk
re-certification" ([BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md):666-671; `configs/plant_versions.toml:154-156`;
D3). When it is taken, take it once, as one decision, and batch into it per species the survey's deferred checklist:
the SB3-only declaration (which takes that species off the frozen core), one `_scale_action` and one `_get_obs` in the
base class, removal of the midpoint mapping and the inert reset height channel, then a regenerated
`plant_manifest.generated.json` and reset golden (for trex, also the stale comment at
`environments/trex/assets/trex.xml:554`, added by PR-B, 2026-09-28: it names `mjx_config.py`'s `target_standing_z`,
`_NATURAL_PITCH` and `healthy_z_range`, which the frozen registration no longer holds, and MJCF bytes enter the plant
identity, so only a trex plant or policy-interface revision may edit it). It moves every certified run's plant
identity, so it is not cleanup (§7); decide it with the maintainer once the walkers it would strand are known.

### 4.2 What stays

| Kept | Location at `f850815` | Why | Frozen size |
|---|---|---|---|
| `build_mjx_observation` (verbatim, top level), `_SPECIES_CONFIGS`, `register_species_mjx` | `environments/shared/mjx_env.py:342,362-419` | The probe executes it (`policy_layer.py:245-260`), and it is token-hashed as `training_reset_and_step` (:387). Its call-time `from .obs_functions import …` (`mjx_env.py:385`) pins it to `environments/shared/`, because relative-import dots are hashed | 1,645 → 76 lines |
| `make_obs_fn` (verbatim, signature included) | `jax_setup.py:524-562` | Hashed as `cpu_evaluation` (`policy_layer.py:388`); never executed | 1,003 → 54 lines |
| `scale_action_around_nominal_jax`, `reset_mujoco_data_to_home`, `scale_action_jax` | `mjx_utils.py:22-83` | The first two are hashed (`policy_layer.py:382-385,400-404`). `policy_layer.py:34` names `scale_action_jax`. Only `check_jax` (:11-19) goes | 83 → about 72 lines |
| `obs_functions.py` (whole file) | :1-180 | `_array_mod`, `build_bipedal_obs` and `build_quadruped_obs` are hashed (`policy_layer.py:378-381`); the probe executes `SensorLayout` | 180, unchanged |
| Per-species `mjx_config.py`, reduced to the registration | trex 155, velociraptor 65, brachiosaurus 81, dibothrosuchus 78 lines | Read at `policy_layer.py:181-241`; the module name is a payload literal. Key sets differ: trex 9 (`action_mapping`, `action_filter_cutoff_hz`, `frame_skip`, `sensor_foot_indices`, `sensor_foot_aux_indices`, `sensor_gyro_start`, `sensor_accel_start`, `sensor_quat_start`, `body_ids`), brachiosaurus 8 (no cutoff), velociraptor and dibothrosuchus 7 (no cutoff, no `sensor_foot_aux_indices`) | 379 → 91 lines |
| `plant_contract/policy_layer.py`, `digests.py`, `manifest.py` | whole files | They are the hasher | zero-line diff. *2026-09-29: CU-14b edits `manifest.py` (the per-process plant-identity cache, §3.2 CU-14 row) outside any hashed payload: no digest hashes `manifest.py` itself (`policy_layer.py:355-366` hashes named callables), and the frozen MJX core does not include it; the digest-snapshot harness, run with the optional backends blocked, printed 848 lines with 0 errors, byte-identical on the base `c69a169` and the head.* |
| `action_filter.low_pass_alpha`, `apply_low_pass` | `action_filter.py:47-70` | Hashed for trex (`policy_layer.py:405-418`) | unchanged |
| `command_frame` constants | `command_frame.py` | Enter the policy payload (`policy_layer.py:15`) and `task_sha256` | unchanged |
| `foot_contact_weight`/`foot_contact_gate` params and their six `[env]` keys | `trex_env.py:141-142`, `dibothrosuchus_env.py:108-109`; `configs/dibothrosuchus/stage1_balance.toml:13-14`, `configs/trex/stance.toml:13-14`, `configs/trex/recovery.toml:57-58` | `task_sha256` hashes the constructor defaults overlaid with `[env]`. Stripping the dibothrosuchus keys moved its certified stance task from `083e2966` to `abfb339f` (measured) | unchanged |
| Byte-frozen files: `behavior_env.py`, the five species env modules, `direction_commands.py`, `terrain.py`, `terrain_sampling.py` | whole files | `behavior_identity` hashes their raw bytes, comments included | zero-line diff in PR-A and PR-B (their acceptance step 9). Any other PR that edits these files says so, and its harness diff shows only the `behavior` identity lines of the species it touches (CU-12, PR-C) |
| TOML digest inputs | `[stage]`, `[env]`, `[ppo]`, `[sac]`, `[curriculum]`; `configs/*/behaviors/*.toml`; `plant_versions.toml`, `plant_manifest.generated.json`, `recovery_calibration.json`; the `[[species]]` rows, with no `training_backends` added for the four dual species (`plant_contract/manifest.py:88-93` raises on a mismatch) | Only `[jax]` and `[jax.policy_kwargs]` may go. Three prototypes confirmed they enter no digest | — |

The frozen code totals 76 + 54 + 72 + 180 + 91 = **473 lines**. *As built by PR-B (2026-09-28):* 90 + 60 + 79 +
180 + 140 = 549 lines, 464 without their module docstrings; the difference is mostly the FROZEN (D-D17) notices (§4.3).

The tokenizer (`digests.py:98-137`) drops comments, docstrings and indentation width. It keeps identifiers,
annotations, parentheses, commas and relative-import dots, and it refuses f-strings. Module paths are not hashed
(`digests.py:154-166`). So module docstrings can say anything, but function bodies and signatures must not change.

**Anchor token digests** (`_callable_semantics(...)["tokens_sha256"]`), re-computed 2026-09-26 at `f850815`:

| Function | Token digest |
|---|---|
| `mjx_env.build_mjx_observation` | `sha256:d1a8ac56e6f533670690897f47756ebf07cc2e499d6f56da89a5bea7d2446ac4` |
| `jax_setup.make_obs_fn` | `sha256:66d4e404736f5c23159eec010e125a3a9a5c0981fae300897040c16b6fa923c5` |
| `mjx_utils.scale_action_around_nominal_jax` | `sha256:f01be749e8a78ac9c9266d462f31511a06e9f0c0fef2b7d7292c4b8c431b2eae` |
| `mjx_utils.reset_mujoco_data_to_home` | `sha256:f780f506759e615e1a0e9292deeae31fa8113a6feef5442993bd26bfbec28eb0` |
| `obs_functions._array_mod` | `sha256:cbcfa3fe9292bf98e8d8c70d88d4f1ef03e49e11cfcbd50d0d6f87e6967d48f4` |
| `obs_functions.build_bipedal_obs` | `sha256:8b46a7e02e14059c38d723ff46f60eae31b633bb3b61bb2455caf862dc6f75e4` |
| `obs_functions.build_quadruped_obs` | `sha256:7b4e74c4c97a78b143f80902d6e4eb542ecc8a3830579182c1921ad5065dbded` |
| `action_filter.low_pass_alpha` (trex) | `sha256:f32bb3f1cda4b52493b8bbd4c61fff9c86231f705ddcf9d462900f97af7f0908` |
| `action_filter.apply_low_pass` (trex) | `sha256:8c619d7021723c3f74bfc3d5ccb45c27738c931d176c487a3a76a18b1affaa19` |

### 4.3 Labelling and the pin test (with the critic's corrections)

1. **Module docstrings.** `mjx_env.py`, `jax_setup.py`, `mjx_utils.py` and the four `mjx_config.py` files each open
   with a FROZEN (D-D17) notice. The notice must not say "never executed for training"; the critic found that false.
   It states that: these tokens are hashed into the policy-interface digests of trex, velociraptor, brachiosaurus and
   dibothrosuchus; the plant-contract MJX probe executes `build_mjx_observation` and the `mjx_config` registrations on
   every SB3 training and evaluation run (`validate_environment_plant(require_backend_parity=True)`,
   `validation.py:81`, called from `train_base.py:220` and `evaluation.py:279,465`); `make_obs_fn` and the `mjx_utils`
   functions are hashed and never executed; nothing trains on them, and they must not be edited, reformatted or moved.
2. **Pin test.** Add `environments/shared/tests/test_plant_contract_frozen_mjx.py` (about 40–60 lines); the
   plant-contract job's glob `test_plant_contract_*.py` (`python-ci.yml:163`) picks it up; after CU-14 drops that job's
   pytest step, only the `test (shared, …)` matrix runs it. It asserts the seven anchor digests plus the two trex
   low-pass digests, so a failure names the function and D-D17 instead of a stale manifest.
   It also asserts each species' exact key set (9/8/7/7) rather than a common nine. Delete it together with the core.
3. **Formatter.** Exclude the frozen modules from `ruff format` and `ruff check --fix`. CI's ruff was unpinned when
   this was written, and a release that re-parenthesises code would move a digest. Since CU-1, CI and pre-commit pin
   ruff 0.16.9, and the pre-commit ruff hooks cover `environments/`, which holds these modules; a later pin bump
   would move a digest the same way, so the exclusion still stands.
4. **Docs.** Rewrite "Backend parity and runtime binding" in [PLANT_CONTRACT.md](PLANT_CONTRACT.md) (:152-167).
   `CONTRIBUTING.md` step 4 (:101-107) becomes "declare `supported_training_backends = ("stable-baselines3",)`". Step
   8 (:132-140) adds `training_backends = ["stable-baselines3"]` to `species_manifest.toml`; without it,
   `plant_contract/manifest.py:88-93` raises. D-D17 records the exit rule (§4.1).

*As carried out by PR-B (2026-09-28):* the eight files total 549 lines (§4.2). The FROZEN (D-D17) notices open the
seven reduced files, and `obs_functions.py` has a zero-line diff; `jax_setup.py`'s Docker wording went with its old
docstring, and the harness output is unchanged. Two `TYPE_CHECKING`-only aliases keep the hashed signatures
type-checkable without the retired classes (`from __future__ import annotations` keeps the modules importable): `MJXEnvConfig = Any` in `mjx_env.py` and `SpeciesContext = Any` in
`jax_setup.py`; the annotation tokens keep their names (§4.10). The pin test is 89 lines and 15 tests: the nine
token digests, each registration's exact key set (four tests), a tripwire that the registrations are exactly the
species declaring `jax-mjx`, and a check of the formatter exclusion. The plant-contract job's glob (`python-ci.yml:177`
at PR-B's head) and the `test (shared, …)` matrix run it. The exclusion is `[tool.ruff]` `extend-exclude`, listing
the eight files, with `force-exclude = true`, so ruff skips them even when a path is named on the command line.

### 4.4 The digest-snapshot harness (in the repository since 2026-09-26)

The harness is committed with this plan (2026-09-26) as `environments/shared/harnesses/digest_snapshot.py`, a guarded
rebuild (356 lines, lint-clean under the repository's ruff settings) of a 237-line prototype that never entered the
repository. It is listed in the module list of `environments/shared/harnesses/__init__.py` (:15-24), and coverage
already omits `harnesses/*` (`pyproject.toml:169`; keep it omitted when CU-9 lands). *CU-9 (#574, 2026-09-30):
the blanket omit goes; `digest_snapshot.py` stays omitted by name (`pyproject.toml:160`), and
`test_coverage_config.py` pins it; the module list is now `harnesses/__init__.py:15-33`.* PR-A and PR-B use it for
acceptance step 1 (§4.5, §4.6). Decision 16 (commit a golden output and add a `--check` step) stays proposed: until it
is taken, the harness is run by hand. *Taken 2026-09-29 as D-D22 (§2 row 16): the full run becomes a golden that the
plant-contract job checks on pull requests; its own PR builds the check (§3.1 item 4), and until that PR lands the
harness is still run by hand.* *As carried out by the ROW-16 PR (2026-09-29):* the full run is committed as
`configs/digest_snapshot.generated.txt`, the plant-contract job runs `--check` against it on every event, and
`--write` regenerates it (§3.1 item 4; landed as #571, 2026-09-29). The frozen-core reference copies are not in the repository; PR-B rebuilds them
from `f850815` per §4.2. *As carried out by PR-B (2026-09-28):* the eight files keep §4.2's functions verbatim (their
token digests equal the `f850815` values), and `test_plant_contract_frozen_mjx.py` pins them (§4.3).

*CU-11 (landed as #573, 2026-09-30):* a sixth section, `reward`, adds four lines per stage (a summary, the ends of the
state probes, a digest of the discrete records and one of the rounded values), and `--exact` prints bit-exact
per-stream digests of the same captures, and of every behavior recipe's, for a same-machine base/head diff (§3.2,
CU-11 row).

**Output.** The harness prints one tab-separated line per value, in a fixed order, with no timestamps or paths. It
covers every plant identity and policy sub-digest (`plant_contract.current_plant_identity`, `plant_contract/manifest.py:267`). For
each of the 21 stages it covers the task fingerprint (`task_fingerprint.derive_stage_task_fingerprint`, :503), the
gate digest (`gate_schema.gate_config_view`/`gate_config_sha256`, :302/:328), the PPO and SAC `hyperparameters_sha256`
(`config.py:528`) and the stage-config-view digests. It also covers the 2 recovery calibrations
(`load_recovery_calibration`, :82) and the 66 recipe and behavior-identity digests
(`train_behaviors.read_recipe`/`create_behavior_env`, :45/:144). The critic's base snapshot was 848 lines. In the
harness, the stage-config-view digest comes from the real `save_stage_config` writing into a temporary directory, with
the run-specific keys dropped.

**Flags.** `--block-optional-backends` makes jax, flax, optax, `mujoco.mjx`, ray, mjlab, hypertune and `google.cloud`
raise `ImportError`, and the harness refuses to start if one is already imported. `--skip-behaviors` cuts a run from
about 3.6 min (848 lines) to about 50 s (440 lines), measured 2026-09-26 on `be63a58`; the behavior section is most of
the time. `--check [PATH]` and `--write [PATH]` (ROW-16) compare the full run with the committed golden or rewrite
it; both imply `--block-optional-backends` and refuse `--skip-behaviors`. The full run took about 45 s on
2026-09-29 at `a919985` (Python 3.12, `.[test]` and MuJoCo 3.10.0, as the plant-contract job installs them).

**Guards** (critic item 6). Run the harness by file path with `PYTHONPATH=<checkout>`: a base older than this plan has
no harness, and `python -m` imports the head's package. The prototype's `--repo` mixed the head's plant and stage
digests with the base's behavior configs, so it could print a false "no diff". The harness asserts that
`environments/__init__.py` is exactly `--repo`'s own package (an editable install resolves `environments` to the
checkout it was installed from, and a checkout nested inside `--repo` must not pass either); keep it lint-clean. It exits 2 when it refuses (`environments` imported from outside `--repo`, a `--repo` that is not a
checkout, or a blocked backend already imported), 1 when any step printed an ERROR line, and 0 when clean; error lines
print `<repo>` instead of the path, so the same failure on two checkouts diffs clean. Measured before it was committed
(2026-09-26): two runs on a checkout of `f850815` and one on a clean worktree of it gave byte-identical 848-line
output with 0 errors, identical to the prototype's snapshot, and all four refusal cases refused.

**Measured on `be63a58`** (2026-09-26; the committed copy, run by file path on a `be63a58` checkout carrying this
plan's changes, with `PYTHONPATH` set to it): with `--block-optional-backends`, the full run printed 848 lines with 0
errors in 215.7 s and exited 0, and its output is byte-identical to the `f850815` snapshot (blocked backends): #559
moved no digest. With `--skip-behaviors` added, it printed 440 lines, the first 440 of the full run, with 0 errors in
50.1 s. A `--repo` that is not a checkout and `python -m` with `--repo` naming another checkout were both refused with
exit 2.

### 4.5 PR-A: retire Ray Tune, the Vertex HPT sweeps and mjlab (D-D17, widened)

PR-A is digest-neutral by construction. No hasher reads the sweep, Vertex or mjlab code, and its verifier measured
identical snapshots before and after.

**Delete (42 files, 13,221 lines):**

| What | Files | Lines |
|---|---|---|
| `environments/shared/scripts/sweep/` (Ray-only 2,489, Vertex-only 2,620, shared 1,757) | 13 | 6,866 |
| mjlab: `shared/mjlab_env.py`, `velociraptor/mjlab_config.py`, `velociraptor/scripts/train_mjlab.py` | 3 | 322 |
| `configs/*/sweep_{ppo,sac}.json` | 12 | 1,064 |
| `configs/quality_scoring.toml` (read only by `sweep/scoring.py:26`) | 1 | 145 |
| `ray_tune_sweep.ipynb` | 1 | 777 |
| `test_sweep_*.py`, except `test_sweep_reporting.py` | 11 | 3,494 |
| `website/docs/training/sweeps.md` | 1 | 553 |

**Widened by D-D17 (2026-09-26), then split.** The maintainer retired the single-job Vertex AI route and GCS artifact
upload as well, and the same day moved them into a PR of their own, PR-A2 (this plan lands it after PR-A; §3.1). PR-A2 deletes
`scripts/setup_vertex_ai.sh`, the `Dockerfile`, `.dockerignore` and
`website/docs/training/vertex-ai.md` (with its sidebar entry and the links into it), and the GCS upload path:
`curriculum --gcs-bucket` / `--gcs-project` with `config.upload_curriculum_artifacts`, the `gs://` branch of
`reporting/csv_output.py`, and the `[gcp]` extra. The counts above are PR-A's; PR-A2 derives its own list and
counts when it opens, with an end-to-end test of the command-line curriculum path it edits. The bullets below that
keep the route or `[gcp]` are superseded where marked, by PR-A2. `tb_sync.py` stays (§4.9) even though its `/gcs/` branch served the Vertex FUSE mount.
*As carried out by PR-A2 (2026-09-27):* the four files are deleted (957 lines) with the sidebar entry and the three links into the page (README, `recipes.md`, `jax.md`, whose "Vertex AI with JAX" section goes too), and so are the Docker sections of the README and the installation and quick-start pages and the landing page's "Docker support". In live SB3 code: `cli.py` loses the two flags, `train_curriculum` its two parameters and its final upload call, `config.py` `_upload_to_gcs` and `upload_curriculum_artifacts` (213 lines), and `write_results_csv` its `gs://` branch, so a `gs://` path now raises `ValueError` in both modes; `[gcp]` leaves with them and `[all]` is `[train,jax,viz,dev]`. The end-to-end test (`environments/shared/tests/test_curriculum_cli_end_to_end.py`) runs velociraptor's `train_sb3.py curriculum` in a subprocess, landed first and passes before and after. `tb_sync.py` and the `/gcs/` mount detection (`train_base._is_remote_mount_path`) stay.

**Edit:**
- **Code.** Delete `visualization.py:789-903` and the hypertune `try/except` at `train_base.py:1515-1532`. Keep
  `_report_hpt_metrics` (:1453-1619) and its name: it writes `metrics.json`, which `stage_artifacts.py:106` reads, and
  `test_train_base.py:2404` monkeypatches it. Reword `stage_artifacts.py:3`, the call-site comment at
  `train_base.py:1419` and the `_report_hpt_metrics` docstring (:1468-1473): it writes `metrics.json` for
  `stage_artifacts` and the Drive summary. *Corrected 2026-09-27 (PR-A):* only the CLI `train` subcommand writes
  `metrics.json`; the SB3 notebook's `train_stage` passes `report_metrics=False`, and the Drive summary does not
  read the file. `build_stage_results_from_eval_data` reads it back for `backfill_gate_verdict.py` and for
  `generate_stage_artifacts` called without `stage_results`; PR-A's docstring says so.
- **CI (`python-ci.yml`).** :267 becomes `".[train,test,viz]"`, keeping CU-1's `stable-baselines3[extra]==2.9.0` pin
  (`test_ci_tool_pins.py` checks it). :269-306 shrink to `import stable_baselines3, torch`
  plus the CPU-wheel assertion (:284). :377 drops `test_sweep_ray_plant_contract.py`. Job names are unchanged.
- **`pyproject.toml`.** Remove `cloudml-hypertune` (:41), `[ray]` (:56-61), `[mjlab]` (:62-67) and the mjlab coverage
  omits (:170-173). Keep `[gcp]`, rewording its comment. `[all]` (:91) becomes `[train,jax,viz,gcp,dev]`, because
  `curriculum --gcs-bucket` needs `google-cloud-storage` (`config.py:881,952`). *Superseded by D-D17:* PR-A2
  removes `[gcp]` with the GCS upload code, and `[all]` then becomes `[train,jax,viz,dev]`. Done by PR-A2.
- **Vertex single-job route** (if decision 1 keeps it; *superseded by D-D17*, which deletes the route instead). Drop `setup_vertex_ai.sh:183`. Delete `vertex-ai.md:570-679`
  and rewrite its Next Steps (:692-696). Add a `pip install google-cloud-aiplatform` line; the only one today, at
  :613, is deleted with that range. Reword :368. *Moot: PR-A2 deleted the route.*
- **Tests.** Edit `compsognathus/tests/test_training_env.py:12,128-135`, `test_policy_loading.py:597,605-606`,
  `test_resume_load_path.py:316-341`, `test_species_catalog.py:652-656`, `test_species_names.py:105` and
  `test_phase_c_interface.py:35-36,520-534` (the last range includes the banner). Rename `test_sweep_reporting.py`
  (108 lines, which cover `generate_stage_artifacts`), for example to `test_stage_artifacts_generation.py`.
- **Catalog.** Delete the `ray_tune_sweep` block (`species_manifest.toml:53-59`). Renumbering the Drive summary's
  `display_order` (4 to 3) is cosmetic, because `species_catalog.py:1060` only sorts; skip it, or renumber once in
  PR-B, which leaves another gap. Run `python -m environments.shared.species_catalog`, which regenerates `README.md:501`
  and `species.generated.json`.
- **Docs the tests and the site build force.** Remove the Ray notebook's path tokens at
  `docs/BALANCE_REWARD_METRICS.md:147,161`, `docs/RL_TRAINING_PLAN.md:5,69,121`, `hyperparameters.md:193,272` and
  `installation.md:77`. Remove `'training/sweeps'` from `website/sidebars.ts:19`. Retarget the `sweeps.md` links at
  `hyperparameters.md:194,276`, `recipes.md:618` and `vertex-ai.md:380,696`.
- **Other docs.** `recipes.md:609-618`, `README.md:514`, `CONTRIBUTING.md:116`,
  `environments/compsognathus/README.md:189-191`, `docs/ROADMAP.md:25,496-515,610` and
  `docs/SPECIES_NAMING.md:32-33` (the Ray Tune notebook sentence). KNOWN_ISSUES :879-888,
  :890-897, the sweep items of :898-907, :914-916, :932-933, :1117-1118, :1141-1143, the Ray neighbour at :249-257,
  and the `[mjlab]` clause at :1110-1111. Of the entries this plan adds (§3.5), also delete the Ray PPO `ent_coef_end`
  entry and the NB2 entry, and drop the Ray clauses of the notebook-pins bullet (:1119-1122); the replaced sweep-keys,
  `ray_orchestration.py` and `ray_tune_sweep.ipynb` entries are in the ranges above. These KNOWN_ISSUES line numbers
  are at `f850815`, before this plan's entries were added; locate entries by heading and title.
  [PLANT_CONTRACT.md](PLANT_CONTRACT.md):184-186. The Drive summary's markdown cells 1, 6 and 13 and its cell-8
  docstring. Keep the reader code: March 2026 sweep folders remain on Drive.
- **Decision record (append-only).** Add D-D17 after D-D16 in [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md)
  §6.2 and in the consolidation plan's §6 table, plus a status row. Update the id lists in [README.md](README.md) and
  NEXT_STEPS. (Done on 2026-09-26, when D-D17 was taken: the rows, the id lists and the status rows exist, so PR-A
  appends to the D-D17 rows rather than adding them; the rest of this bullet is still PR-A's.) Append "Amended by D-D17" to D-A12, D-A15, D-B1 and D-B12 (BEHAVIOR_RECIPES_PLAN.md §6.1, :1037,
  :1040, :1051, :1062), and to D-D11 in both plans (BEHAVIOR_RECIPES_PLAN.md:1121 and
  CONSOLIDATION_PLAN_2026_09.md:1229) (critic item 7). D-D17 supersedes A6; append "Superseded by D-D17" to A6
  (:1005) so the list stops stating a retired sweep rule as current. Mark BALANCE_REWARD_METRICS "Withdrawn: Ray Tune
  retired (D-D17)" in the index (`docs/README.md:48`), and note on RL_TRAINING_PLAN's row (:44) that its trials 2
  and 4 were Ray sweeps. Add CHANGELOG Removed and Migration entries. They supersede the "keep cloudml-hypertune"
  advice at [investigations/TREX_REVIEW_2026_07.md](investigations/TREX_REVIEW_2026_07.md):1172-1180, which gets an
  appended dated note and nothing else (the index allows appended corrections, `docs/README.md:14`; precedent
  `ce29548`). Add
  one sentence under the index's "Investigations & run analyses": notes dated before D-D17 cite retired paths, and the
  two archive tags reproduce them (the JAX tag for notes such as TRAINING_REVIEW_JAX_STAGE1). *Superseded
  2026-09-27 (decision 2 = (c)):* the sentence names the `0.3.8` tag (`afad625`) and git history instead, and
  the living docs that name a removed path cite that tag. Living docs that must
  name a removed path use a backticked path plus the tag (precedent `NEXT_STEPS.md:35`), never a link. No relative
  link from `docs/investigations/`, `docs/reviews/` or `docs/hardware/` points into the removal set.

**Draft D-D17**, kept as proposed. *Recorded 2026-09-26:* the maintainer took D-D17 with the single-job route and GCS
upload retired too; the row is in [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2. The archive points (§2
row 2) are still open and are not part of D-D17; once they are settled, the outcome, with any tag SHAs, is appended to
that row rather than filled into this draft. *Superseded 2026-09-27 (decision 2 = (c)):* there are no archive tags;
the retired code is recoverable from the `0.3.8` tag (`afad625`) and git history, as the D-D17 rows now record.

> **D-D17** | Taken `<date>`: Stable-Baselines3 (PPO and SAC) is the only training, evaluation and evidence backend until every species' stage chain and behavior is certified. Retired, recoverable at the annotated tags `archive/secondary-backends-2026-09` (`<SHA>`, the first parent of `<PR-A merge>`) and `<second tag>` (`<SHA>`, the first parent of `<PR-B merge>`): the Ray Tune sweep, the Vertex AI hyperparameter-tuning sweep (`environments/shared/scripts/sweep/`, `configs/*/sweep_*.json`, `configs/quality_scoring.toml`, `cloudml-hypertune` and its report in `train_base`) and the mjlab scaffold (`<PR-A>`); the JAX/MJX trainer, evaluator, stage writer, notebook, `[jax]`/`[jax-cpu]` extras, `[jax]` stage tables and `test-jax-cpu` job (`<PR-B>`). Single SB3 jobs on Vertex AI (Dockerfile, `scripts/setup_vertex_ai.sh`, `[gcp]`) stay [per decision 1]. No plant, task, gate, hyperparameter, stage-config, recipe, behavior-identity or recovery digest moves: trex, velociraptor, brachiosaurus and dibothrosuchus keep their dual-backend declaration and a frozen MJX interface core whose tokens their policy-interface digests hash; it is never edited, and each species drops it only by declaring itself SB3-only inside its next deliberate `policy_interface_revision` bump. Supersedes A6, narrows A7 to "SB3 is the evidence backend", and amends D-A5, D-A12, D-A15, D-B1, D-B12, D-B13, D-C3, D-C4, D-C7, D-C16, D-D11 and G1. Adding any backend back is a new decision; its entry conditions are D-B13, invariant 9, the drift list in the `<PR-B>` CHANGELOG entry, and a CI job that runs one real training trial.

**Acceptance (PR-A):**
1. The harness, run by path with `--block-optional-backends` on base and head, produces an empty `diff`.
2. `plant_contract --check`, `plant_contract --check --baseline <base manifest>` and `species_catalog --check` pass.
3. `pytest --collect-only environments` reports 0 errors (the verifier measured 4,494 → 4,239 tests collected).
4. Full CI passes with the `full-ci` label, and `ruff check` and `ruff format --check` are clean (since CU-1, `ruff
   format --check environments/`, CI's scope: the pinned ruff also flags notebooks and Markdown elsewhere).
5. mypy with SB3 reports the same 21 errors in the same 6 files (358 → 331 files checked; the verifier measured
   357 → 330 before the harness was added), or 0 once CU-1 has landed (CU-1's `test_ci_tool_pins.py` and CU-3's two
   test files make it 361 files before PR-A; PR-A re-derives its count).
6. The notebook AST parse, `test_sb3_notebook_pins.py` and `test_species_names.py` pass.
7. An import walk of every `environments.*` module (`pkgutil.walk_packages`) succeeds. `--ignore-missing-imports`
   hides dangling first-party imports, so mypy alone does not catch them.
8. `npm ci && npm run build` and `npx tsc --noEmit` pass in `website/`.
9. `git diff --exit-code origin/main --` is empty for: `environments/shared/plant_contract/`, `result_bundle/`,
   `result_schema.py`, `task_fingerprint.py`; `obs_functions.py`, `action_filter.py`, `command_frame.py`,
   `perturbation.py`; `behavior_env.py`, `direction_commands.py`, `terrain.py`, `terrain_sampling.py`;
   `environments/*/envs/`, `configs/*/behaviors/`, `configs/plant_*` and the stage TOMLs.

**Risks (PR-A):**
- Dropping `gcp` from `[all]` would silently disable GCS upload; the only sign is a warning. (Under D-D17 PR-A2
  removes the upload code with `[gcp]`, so a `--gcs-bucket` caller gets an argparse error instead. As carried
  out by PR-A2: argparse exits 2 with "unrecognized arguments", and `test_cli.py` pins it.)
- Removing the Drive summary's sweep branch would make `discover_runs` walk into `sweeps/`.
- `/docs/training/sweeps` will return 404, because the site has no redirect plugin.

### 4.6 PR-B: retire the JAX/MJX runtime; keep the frozen core

**Delete (30 files, 15,698 lines):**

| Files | Count | Lines |
|---|---|---|
| `environments/shared/jax_{checkpoint,curriculum,eval,hooks,normalization,ppo,reward_termination,train_fn,trainer,trainer_types,training,training_utils,viz}.py` | 13 | 6,777 |
| `shared/tests/test_jax_*.py`, `test_mjx_*.py`, `trex/tests/test_trex_mjx_reward_parity.py` | 15 | 7,256 |
| `jax_training.ipynb` | 1 | 1,372 |
| `website/docs/training/jax.md` | 1 | 293 (260 after PR-A2 removed its Vertex section) |

*As carried out by PR-B (2026-09-28):* the 30 files are deleted with the sidebar entry (15,665 lines, `jax.md` at
260; the whole diff is +489 / −21,178 in 113 files). The `[jax]` and `[jax.policy_kwargs]` tables of the 12 stage
TOMLs go (304 lines deleted, none added, no `foot_contact_*` line touched), and `load_stage_config` now refuses a
`[jax]` table with `ValueError` ("unknown top-level table(s) ['jax']"), which `test_config.py`'s new
`test_a_retired_jax_table_is_rejected` pins. Also removed: the `jax` and `jax-cpu` extras (`[all]` is
`[train,viz,dev]`), `reporting.save_jax_stage_artifacts`, `plant_contract.validate_mjx_environment_plant`, the
notebook's `species_manifest.toml` entry (the README catalog and `species.generated.json` regenerated), the JAX parts
of kept tests, `velociraptor/requirements.txt`'s JAX lines, and the `test-jax-cpu` job with its coverage `needs`
entry (23 jobs → 22). Kept, as §4.9 says: the `foot_contact_*` parameters and `[env]` keys, the readers of recorded
JAX runs, the `jax-mjx` catalog rows (decision 17), `command_frame.py` whole and `gate_schema`'s backend-override
validation. Where it departs from the lists below: `test_stance_gate.py:401-470` becomes an SB3-only class (it held the
only manager assertions on a failing stance panel) rather than going; `RESULT_BUNDLES.md`'s run-reuse paragraph is
reworded for SB3 rather than deleted, because it still describes SB3 resume and provenance drift; the KNOWN_ISSUES
quadruped-detection MEDIUM is kept, reworded as latent (CU-7a closes it; landed as #572, 2026-09-30, §3.2); and `result_bundle/gate_verdict.py:18,62-63` and
`action_filter.py:12-13`, which acceptance step 9 keeps unchanged, are not reworded but handed to CU-7 (§3.2).

**Library:**
- Reduce the frozen core per §4.2.
- In `reporting/stage_artifacts.py`, delete :1725-2200 and the mention at :4, and remove the export at
  `reporting/__init__.py:52,70`.
- In `plant_contract/validation.py`, delete :104-171 and the imports that become unused (:8, :19), and remove the
  export at `plant_contract/__init__.py:73,109`.
- Strip `[jax]`/`[jax.policy_kwargs]` from the 12 stage TOMLs (304 lines), but keep the six `foot_contact_*` `[env]`
  keys. **In the same commit**, drop `"jax"` from `config.py:180-181` and `jax_kwargs` (:249, :266), and update
  `test_config.py:101`. Doing these in any other order makes `load_stage_config` raise (`config.py:224-231`).
- Keep `"JAX_PPO": "jax_kwargs"` (`config.py:494`) for old records.
- In `species_manifest.toml`, delete :46-52 and `"jax_training"` at :77, :157, :241 and :299, then regenerate. Keep
  the `jax-mjx` metric rows, which `species_catalog.py:949-954` requires while the species stay dual.
- Reword `result_bundle/gate_verdict.py:18,62`, `reporting/summaries.py:107`, `curriculum/gate_schema.py:13,384` and
  `config.py:176,204`.

**Tests** (about 1,285 lines in kept files):
- Delete `test_plant_contract_validation.py:19,46-69`.
- In `test_reporting_stage_artifacts.py`, edit the :13 import (it still needs `build_stage_results_from_eval_data`)
  and delete :218-778 and :952-965, but move the :723-735 test (an SB3 `save_stage_config` test) out of the deleted
  `TestSaveJaxStageArtifacts` class, together with the helpers it calls (`_override_config`, :652-665, and
  `_gated_config`, :618-630, without its `jax_kwargs` key), or inline its config.
- Delete `test_gate_dispatch_fail_closed.py:26,161-351` and `test_recovery_gate_wiring.py:48,708-765`, whose imports
  would otherwise break collection. Also delete `test_stance_gate.py:401-470`, `test_species_integration.py:447-551`,
  `trex/tests/test_trex_env.py:374-401`, `test_action_filter.py:113-157` and `test_perturbation.py:59-76,269-343`.
- In `test_reward_functions.py`, delete **426-489** (with its banner) and **628-646**. Deleting only 631-646 would
  leave the `skipif` at :630 dangling, which is a `SyntaxError` (critic item 1).
- Delete `test_species_catalog.py:658-660,663-671`, `test_species_names.py:105-109` and
  `test_sb3_notebook_pins.py:15,37,2620,2629` (:15, :37, :2846, :2855 at the follow-up).
- Keep `trex/tests/test_trex_env.py:677-724` and the `jax` parameter of `test_obs_functions.py:59-84`.

**CI and packaging:**
- Delete `test-jax-cpu` (`python-ci.yml:391-434`) and, in the same commit, its entry in the coverage `needs` (:445).
- Delete the `jax` (:43-48) and `jax-cpu` (:49-55) extras; `[all]` becomes `[train,viz,gcp,dev]` (`[train,viz,dev]`
  under D-D17, since PR-A2 removes `[gcp]`).
- Delete `velociraptor/requirements.txt:11-14`.

**Docs:**
- **Notebook tokens.** `installation.md:76`, `quick-start.md:14`, `docs/CODE_CONSOLIDATION.md:326` and
  `docs/NEXT_STEPS.md:572`. In [MJX_CONVERSION_PLAN.md](MJX_CONVERSION_PLAN.md), drop only the prefix (:378, :398,
  :569) and add a "retired design record" banner at :3-6.
- **Sidebar.** `'training/jax'` in `sidebars.ts:19`.
- **Broken links.** `intro.md:52`, `hyperparameters.md:133,237,265`, `recipes.md:626`,
  `environments/velociraptor/README.md:162-166` and `MJX_CONVERSION_PLAN.md:5`. Also fix `vertex-ai.md:67`, which
  links to `jax.md#three-stage-task-sequence` and was the one broken link in the critic's site build. Reword
  `vertex-ai.md:64-67,302,308` as well. *Moot after PR-A2, which deletes `vertex-ai.md`.* The
  `installation.md` line numbers in this section are 24 lower after PR-A2 removed its Docker section, and
  `README.md:513` (the JAX roadmap bullet) is now :489; re-derive the rest before editing. PR-B's
  FROZEN notice for `jax_setup.py` (§4.3 item 1) can also drop its Docker wording (:6, :26-28), if the
  harness shows the module docstring is outside the hashed tokens.
- **Text that becomes wrong once `[jax]` is rejected.** `hyperparameters.md:50,260-266` and `recipes.md:24,620-629`.
- **Other text.** [PLANT_CONTRACT.md](PLANT_CONTRACT.md):149-150, 152-167, 179-183;
  `docs/RESULT_BUNDLES.md:183-186,313-342,363-365` (keep :375-379); ROADMAP Phase 5 (:453-516);
  `README.md:18,63-65,513`; `docs/SPECIES_NAMING.md:33-34` (the JAX sentence); the MJX_CONVERSION_PLAN row
  (`docs/README.md:46`), which becomes "Retired design record (D-D17)", because it points at KNOWN_ISSUES divergences
  and a JAX guide that PR-B deletes; `intro.md:19`; `installation.md:52-55,97-104`;
  `website/src/pages/index.tsx:648`; `website/src/components/SpeciesCatalog/index.tsx:214-215`.
- **KNOWN_ISSUES** (relocate by heading after PR-A). Delete :23-48, :613-658, :803-810, :908-910, :1050-1056, :1134
  and :1138-1140. Partly edit :249-257, :668-670, :715-718, :755 and :1033. Keep :781-788, because the probe still
  runs `build_mjx_observation`. Reword :794-796, because the knobs are still digest inputs. Of the entries this plan
  adds (§3.5), also delete the JAX curriculum-threshold and snout-proximity entries, and drop the JAX clauses of the
  notebook-pins bullet; :23-48 includes the corrected "Curriculum gates" bullet.
- **Amendments (append-only).** A7 (:1006): "Narrowed by D-D17: SB3 is the evidence backend; nothing trains on MJX,
  and the frozen core only feeds the probe." D-A5. D-B13, which becomes the re-add entry condition. D-C3. D-C4: the
  probe still runs on the frozen core, and nothing trains on it. D-C7 and D-C16. G1 in **both** plans; the consolidation plan's G1
  says "MJX fails closed". New paragraphs after the recipes plan's risk bullet and after invariant 9. The
  consolidation plan's PR-3b, PR-9 and §8 risks. *Done by PR-B (2026-09-28).*
- **CHANGELOG.** Record both archive SHAs, the drift list (§5.3), the titles of the dropped KNOWN_ISSUES entries (the
  re-add checklist, including the Ray PPO `ent_coef_end`, JAX-threshold, snout-proximity and sweep-keys entries) and
  the seven anchor digests. *Superseded in part 2026-09-27 (decision 2 = (c)):* there are no archive SHAs to
  record; PR-B's entry cites the `0.3.8` tag and its own first parent. *As carried out by PR-A:* PR-A's CHANGELOG
  entry lists the sweep-side titles (defects 1 and 9 among them), so PR-B's drift list cites it for those. *As
  carried out by PR-B (2026-09-28):* its entry cites the `0.3.8` tag, where every file it deletes except `jax.md` is
  byte-identical, and its own first parent, which holds `jax.md` as PR-A2 left it; it lists the drift list, the
  titles of the KNOWN_ISSUES entries it deletes and the nine anchor token digests (§4.2).

**Acceptance (PR-B)** (restated per critic item 3):
- Run PR-A's steps 1–8, with the snapshot diff both blocked and unblocked.
- Step 9 becomes: the only `plant_contract/` change is removing `validate_mjx_environment_plant` and its export; the
  TOML diff is `[jax]`/`[jax.policy_kwargs]` deletions only, with the `foot_contact_*` lines untouched; nothing else
  on the list changes.
- The pin test passes, and so do the plant-contract tests (a prototype measured 54).
- Collection shows 0 errors (the removal critic measured 3,812 tests collected after both PRs), and mypy checks 303
  files (331 after PR-A; 302 and 330 before the harness was added) with the same 21 errors, or 0 after CU-1 (CU-1's
  and CU-3's test files add three to each count). *As carried out by PR-B (2026-09-28): 3,943 collected, mypy 309
  files, 69 plant-contract tests (§3.1).*
- The wheel step (`python-ci.yml:174-206`) passes, which proves the core ships without JAX.
- Coverage stays at or above `fail_under = 70`. The #1226 artifacts gave 87.48% with the frozen files whole;
  re-measure on the PR.
- `ruff check --fix` runs only on non-frozen files.

**Risks (PR-B):**
- **Any token edit to the frozen core moves four species' digests.** That includes a refactor, annotation, moved
  function, f-string or formatter release. `plant_contract --check` (`python-ci.yml:147`) catches it, and the pin test
  names the function.
- **Well-meant follow-ups have the same effect:** declaring a dual species SB3-only, adding `training_backends` for
  one, "cleaning" `obs_functions._array_mod`, or editing comments in byte-frozen files.
- **Coverage of the frozen code falls** (`mjx_env` 88% → 22%). Only source drift stays guarded, which is acceptable
  for frozen code. *Corrected by PR-B (2026-09-28):* those figures describe the unreduced file. The reduced
  `mjx_env.py` is covered but for one statement (17 of 18: the non-Mapping branch of `value`, which the probe, passing a
  dict, never takes), because the plant-contract probe runs it; `jax_setup.py` and `mjx_utils.py` are about
  20 percent covered, because their bodies are hashed and never executed.
- **Behavior identities are not pinned in CI.** Nothing in CI pins them; the harness closes that gap only when run
  (decision 16; D-D22, taken 2026-09-29, closes it when its own PR lands). *Carried out by the ROW-16 PR
  (2026-09-29): the plant-contract job checks every behavior identity against the committed golden on every
  pull request (§4.4). Landed as #571 the same day.*

### 4.7 Ordering and shared lines

PR-A goes first. PR-B needs two things PR-A adds (critic item 4): D-D17, which the FROZEN docstrings and every
amendment cite, and the first tag. (Since 2026-09-26 D-D17 is recorded already, so PR-B can cite it whichever PR lands
first; what PR-B still takes from PR-A is the archive point, if decision 2, still open, calls for tags. *Superseded
2026-09-27:* decision 2 was settled as (c), so PR-B takes no archive point from PR-A.) The harness, PR-B's acceptance step 1, is already in the repository (§4.4). The
code itself is independent in both directions (checked with grep). Whichever PR lands second rebases over the lines
both edit: `pyproject.toml:91`, `python-ci.yml`, `sidebars.ts:19`, `species_manifest.toml:46-59`,
`test_species_catalog.py:652-671`, `test_species_names.py:105-109`, `README.md:500-501,513-514`,
`species.generated.json`, KNOWN_ISSUES :249-257, ROADMAP :453-516 (which contains PR-A's :496-515),
`installation.md:76-77`, `recipes.md:618,626`, `hyperparameters.md`, `docs/SPECIES_NAMING.md:32-34`, KNOWN_ISSUES
:1119-1122, NEXT_STEPS, the CHANGELOG, both plans and PLANT_CONTRACT.md. PR-A2 (the Vertex route and GCS upload,
split from PR-A on 2026-09-26) lands between them and also edits `pyproject.toml:91` (`[all]` loses `gcp`), so it
rebases over PR-A and PR-B over it. PR-A retargets three `sweeps.md` links inside `website/docs/training/vertex-ai.md`
(:380, :632, :696) and marks its GCE sweep section retired, and drops the sweep hint at `scripts/setup_vertex_ai.sh:183`;
it changes nothing else in either file. PR-A2 deletes both, so its rebase meets two modify/delete conflicts, which
`git rm` resolves. *As carried out:* PR-A landed first (#564) and PR-A2 branched from `9369d6b`, so
those conflicts never arose. PR-B rebases over PR-A2 on `sidebars.ts:19` (the same line), `pyproject.toml`'s
`[all]` (`[train,jax,viz,dev]` after PR-A2; PR-B makes it `[train,viz,dev]`), `jax.md` (modify/delete, which
`git rm` resolves), `installation.md`, `quick-start.md` (PR-B's :14 sits just above PR-A2's edit at :18),
`recipes.md`, the CHANGELOG, KNOWN_ISSUES and the plans. *As carried out by PR-B (2026-09-28):* PR-A2 landed as
#565, and PR-B was built on #565's head `647ca1f`, whose tree the merge `7ae0a19` equals, so it applied to `main`
unchanged, with no rebase conflict.

### 4.8 Maintainer actions outside the repository

1. **Tags (decision 2).** Before PR-A merges, create and push an annotated tag named
   `archive/secondary-backends-2026-09` on PR-A's first parent (`git tag -a`, then `git push origin <tag>`). Tag
   PR-B's first parent the same way before PR-B merges. **Not needed:** decision 2 was settled on 2026-09-27 as
   (c), no archive tags; the `0.3.8` release tag (`afad625`) and git history keep the retired code.
2. **Branch protection (decision 18).** On 2026-09-25 the public API showed `main` with `protected: false`, no
   required checks and no rulesets, although the comments at `python-ci.yml:210-212,316` assume required checks exist. Check Settings →
   Branches/Rules. If `test-jax-cpu` is a required check, remove that requirement when PR-B merges; otherwise every PR
   will wait for a check that never reports. *Checked 2026-09-28 (public API): `main` still has no protection, no
   required checks and no rulesets, so no required check waits for `test-jax-cpu`; the maintainer re-checks Settings →
   Branches/Rules before merging PR-B.*
3. **Before merging, check outside the repository:** put the `full-ci` label on PR-A, PR-A2 and PR-B; confirm that no Vertex
   tuning job, GCS sweep state (`gs://<bucket>/sweeps/…`) or Ray experiment is still live; leave Drive untouched: the
   27 JAX run folders (2026-03-30..04-03, none a certified parent) and the March 2026 sweep folders stay.
   PR-A merged without the label (#564); its full-depth selections were then run by hand in CI's SB3
   environment and passed. For PR-A2, also confirm that no single-job Vertex AI custom job is running, copy
   to Drive any run under `gs://<bucket>/` still wanted as a trunk, and decide whether to delete the
   Artifact Registry repository `setup_vertex_ai.sh` created (`mesozoic-labs` by default, image `trainer`)
   and any bucket it used, which nothing in the repository reads any more. For PR-B, nothing on Drive moves: the
   JAX run folders counted above stay (this plan's count, not re-checked), and §4.9's readers of recorded JAX runs
   are kept. PR-B merged without the label too (#566); its full-depth selections were then run by hand in CI's SB3
   environment on `2b9219d` and passed.

### 4.9 What not to remove

- **Reader back-compat,** so old JAX records keep validating: `result_schema.py:117,534,612,932,1032,1429` and
  `result_bundle/naming.py:15-39`; the `jax-mjx` branches in `reporting/summaries.py:23,220`, `csv_output.py:252-253`
  and `bundles.py:355,409-423`; `ancestors.py:1031-1033` and `config.py:494`; Drive summary cells 8, 9, 11 and 12.
- **The provenance keys** `_DEPENDENCY_PACKAGES` (`result_bundle/constants.py:54-64`) and their fixture
  (`conftest.py:31-45`). Changing the key set records `environment_drift` on the next resume of an incomplete run,
  such as session 6's. *Session 6 finished on 2026-09-28: its resume, on `main` = `7ae0a19`, recorded the expected
  drift (the commit and the `mesozoic_labs` version), which moves no identity.*
- **`gate_schema` backend-override validation** (:274-295, 466, 538, 544-600). It still validates recorded blocks, and
  a shallow clone could not rule out a historical `jax` sub-table. *CU-7a (#572, 2026-09-30) departs from this bullet
  for the two merge helpers only: `apply_backend_overrides` and `has_backend_overrides` had no caller after D-D17
  and are deleted, while the validation stays; a full clone shows that no committed stage TOML ever held a
  `[curriculum.jax]` table.*
- **`command_frame.py`,** including its MJX refusal. PR-9 rewrites that file.
- **All of SAC.**
- **The frozen core,** the dual declarations, the `foot_contact_*` keys and params, and the byte-frozen files.
- **What the SB3 notebook and the Drive summary use:** `build_stage_results_from_eval_data`,
  `backfill_gate_verdict.py`, the `metrics.json` writer and its alias keys (`train_base.py:1839-1850`), `tb_sync.py`,
  `cli._apply_overrides` and the `quality_score` column. *Corrected 2026-09-27 (PR-A):* neither of them uses the
  `metrics.json` writer (§4.5); it stays for the CLI `train` subcommand, whose file
  `build_stage_results_from_eval_data` reads back. The alias keys are at `train_base.py:1871-1872,1881-1882` at
  PR-A's head.
- **The generated JAX/MJX metric lines** in the README and on the website. Changing those is a separate catalog PR
  (decision 17).
- **`obs_functions._array_mod`,** which is hashed. The NumPy-only simplification of the other two `_array_mod` copies
  is digest-safe but optional, so defer it.

### 4.10 How to add a backend back

Adding a backend back is a new decision, taken against D-B13 and invariant 9. Start from the archive tags.
*Superseded 2026-09-27 (decision 2 = (c)):* start from the `0.3.8` tag (`afad625`) for Ray, Vertex HPT and mjlab,
or from PR-A's, PR-A2's or PR-B's first parent for anything that changed after it. *As carried out by PR-B
(2026-09-28):* every file it deleted except `website/docs/training/jax.md` is byte-identical at `0.3.8`, and so are
the `jax`/`jax-cpu` extras, the `[jax]` stage tables and the `test-jax-cpu` job; `jax.md` as PR-A2 left it is in
PR-B's first parent. A re-add replaces the frozen core's two `TYPE_CHECKING`-only aliases (`MJXEnvConfig`,
`SpeciesContext`) with the restored classes (§4.3).

**JAX:**
1. Restore the generic stack almost as-is (3,969 lines), from `jax_ppo` through `jax_viz`.
2. Rebuild the per-species parts (5,918 lines) against the final SB3 rewards, with a single reward composition. Today
   the reward is composed three times: `mjx_env.py:905-1366`, `jax_reward_termination.py:72-384` and `:387-641`.
3. Add a per-component parity test from the tag's template (`0.3.8`, or PR-B's first parent), and fix the drift
   list (§5.3).
4. Build the frozen core forward without changing its tokens. A species newly gaining JAX needs a
   `policy_interface_revision` bump.
5. Restore the CI job.
6. Cover what the old backend lacked: live command modes (`command_frame.py:62-66`), terrain, the
   `recovery_quality/v1`/`task_success/v1` evaluators, and compsognathus.

**Ray:** wrap `train_base.train()`, as the Vertex trial did, instead of restoring the copy. Regenerate the sweep JSONs
from the constructor signatures, and run one real trial in CI; the removed smoke used `train_fn=lambda config: None`.
Restore `visualization.plot_trial_comparison` if the notebook's analysis comes back.

**Vertex HPT:** restore the Vertex half of `scripts/sweep/`, `cloudml-hypertune` and the 18-line report.

**Vertex single-job route and GCS upload:** restore the `Dockerfile`, `.dockerignore`, the upload code, the `--gcs-*`
flags and the `gs://` branch from `0.3.8` (byte-identical there), and `[gcp]` (PR-A reworded its comment),
`scripts/setup_vertex_ai.sh` and `vertex-ai.md` with its sidebar entry from PR-A2's first parent.

**mjlab:** restore its 3 files, the `[mjlab]` extra and its two coverage omits, and the Phase C obs-dim test. It never ran; every factory raised `NotImplementedError`.

## 5. Findings

### 5.1 Walker evaluation on the direction/terrain recipes (2026-09-25)

The maintainer ran this check in eval-only mode with seed 1 and copied back a 19.6 KB summary: nothing trained and
nothing was written under `logs/`. Each certified walker ran from the handoff pair named in its node's
`gate_verdict.json` (`robust_best_model`): trex `20260914_123816/03_locomotion` (seed 42), velociraptor `20260922_125248/02_locomotion`
and compsognathus `20260921_203149/03_locomotion`.

**Episodes per recipe:** `terrain_contact` 20, `follow_direction_speed` 10, `difficult_terrain` 10, and
`follow_direction_difficult_terrain` (`fddt`) 20. Trex also ran 10 episodes on a 100 mm-cell copy of `terrain_contact`
(a 701×701 grid).

How to read the table:
- **Plane** pools the plane episodes of all four recipes. Single-template recipes draw the plane at random, and 7 of
  the 20 `terrain_contact` episodes did.
- **Flat heightfield** is the `terrain_contact` family.
- **Varied** pools sloped, bumps, depressions and mixed (8 + 16 episodes).

The committed cell sizes are 200 mm (trex), 137.5 mm (velociraptor) and 20 mm (compsognathus). Each recipe took
0.4–2.4 min, and each walker under 10 min in total.

| Walker (cruise) | Plane: full horizon, falls | Flat heightfield: full horizon, falls (reasons) | Varied: full horizon, falls | Speed after 1 s (plane) | Mean max progress, heightfield |
|---|---|---|---|---|---|
| trex (1.05 m/s) | 23/23, 0 | 1/13, 12 (fallen 11, head_contact 1). At 100 mm cells: 0/7, 7 (fallen 3, head_contact 2, nosedive 2) | 3/24, 21 (`difficult_terrain` 0/8; `fddt` sloped 2/4, bumps 1/4, depressions 0/4, mixed 0/4) | 1.09–1.11 m/s | 8.6 m |
| velociraptor (2.0 m/s) | 23/23, 0 | 0/13, 4 (tail_contact 3, fallen 1); the other 9 left the map (`terrain_boundary`) | 0/24, 9; 15 left the map | 3.64–3.66 m/s (1.8×) | 37.8 m |
| compsognathus (0.08 m/s) | 23/23, 0 | 0/13, 13 (excessive_tilt 12, body_contact 1) | 0/24, 24 (excessive_tilt 23, body_contact 1) | 0.38 m/s (4.7×) | 0.96 m |

Tracking fractions on the plane were 0.34–0.35 for trex on the straight-cruise recipes and 0.075 under
`follow_direction_speed`, whose commands the walker cannot see. They were about 0.001 for velociraptor and 0 for
compsognathus, whose gaits are far above cruise.

**What it implies:**
1. **Contact is the first terrain blocker (decision 11).** Every walker survives the plane in every episode. On a flat
   heightfield, trex falls in 12 of 13 episodes and compsognathus in 13 of 13, so for them the first blocker is
   heightfield contact, not gait or terrain shape. Velociraptor's 13 non-survivals are 9 map exits and 4 falls
   (item 3).
2. **Finer cells are not the trex fix.** The trex statue survived 3/4 at 100 mm, but the walker fell 7/7 at 100 mm and
   12/13 at 200 mm. Statue results do not predict the policy.
3. **Velociraptor mostly runs off the map.** Its 9 map-edge exits confirm the map-size mismatch. Its 4 falls are
   consistent with the authored toe penetration that the terrain settle reproduces, but they do not prove it.
4. **Compsognathus tips over almost at once,** covering under 1 m on a 20 mm-cell heightfield.
5. **Only trex seed 42 matches its recipe speed.** Velociraptor and compsognathus track at near zero (decision 13).
6. **No terrain pilot until decisions 11 and 13.** No terrain pilot should start, and PR-11 should not copy the
   terrain or speed values, until both are decided. The one pilot whose inputs fit is a flat `follow_direction_speed`
   run on trex seed 42. It is optional: its value is data for PR-13's tracking, heading and survival thresholds, not a
   parent (D-D9). Score it by hand at 40 episodes of its one flat family (about 10–14 min), and do not adapt it to
   terrain before decision 11.
7. **Two walkers were not in this check.** Repeat it on the trex seed-44 walker (`20260925_033501`, locomotion PASS at
   1.57 m/s) and on compsognathus_robot once its locomotion is judged. *Judged PASS on 2026-09-28 (session 6, run
   `20260924_031815`, 0.23 m/s).*

### 5.2 Direction/terrain readiness (with the critic's corrections)

- **Mechanics only.** All 66 recipes build, reset and pass the transition check. Real PPO runs on only 2 of the 11
  recipes, from fixture parents, and nothing has been trained beyond 4,096-step smoke runs. None of it is on Drive.
- **No certification path yet.** The judge is called only from tests, `terrain_command/v1` is not registered, and
  `none/v1` always refuses. The first-class path needs four PRs: PR-8 (a), because the sampler cannot express
  `terrain_contact`; PR-9, because every live command mode is refused (`command_frame.py:89-92`); PR-10, because the
  warm start does not zero the command columns; PR-11, for the new nodes.
- **Parents** (per [NEXT_STEPS.md](NEXT_STEPS.md)): certified walkers exist for trex seed 42 (1.07 m/s), trex seed 44
  (1.57 m/s), velociraptor and compsognathus; compsognathus_robot's locomotion stopped at the Colab cap at 2.8M of
  3.0M steps, so it needs a resume (session 6); dibothrosuchus locomotion failed (session 4); brachiosaurus has no run
  on current physics (session 5). *Updated 2026-09-28:* session 6's resume finished compsognathus_robot's
  locomotion (3,000,704 steps; PASS, 0.23 m/s; run `20260924_031815`), so its walker is certified; the maintainer
  started session 4's re-run the same day, and its result is recorded in [NEXT_STEPS.md](NEXT_STEPS.md) when it
  finishes.
- **Heightfield evidence before §5.1** (statues and #540): the trex statue survived a flat heightfield 0/10 at 200 mm
  cells, 3/4 at 100 mm and 4/4 at 50 mm; the velociraptor statue survived 2/10–4/8, and finer cells made it worse;
  compsognathus-pair contact flickers, so the support-gated reward is 0.20–0.49 per step (0.61–0.92 for the robot),
  against 1.30 on the plane; #540 had measured the trex walker at 0/5.
- **Certificate.** It is infeasible for compsognathus_robot on the commanded terrain recipes, even with perfect
  tracking. For compsognathus it is a risk: it passes only at the top of the velocity tolerance (decision 14).
  Survival must be 20/20 in each of 5 families. The judge takes the family list from the report, so a flat-only panel certifies a
  terrain recipe. Nothing reads the panel seed keys. The recipes call `course_distance` "not a success gate", yet the
  judge gates on it.
- **Other risks:** each training run sees one terrain layout; observations are in world coordinates; stops reward
  standing still (statue 0.151 vs blind walker 0.145, trex); prepare never reads the parent's verdict; pilots train
  on one CPU env; routing `make_env` to the terrain subclass and a real-PPO terrain smoke had no owner (decision 15
  assigns both to PR-11).
- **Carry into PR-11/PR-13, do not patch (PR-12 deletes the runner):** evaluation resets carry no phase tag;
  `--eval-only` hashes a re-saved copy, not the evaluated file; the CSV logger is never closed; `run.json` writes
  `certification_*` keys beside `canonical_certification false`; `lateral_speed_scale` is dead (v_y is always 0);
  replays need imageio-ffmpeg and a documented `MUJOCO_GL`. Cost notes: the robot's plant identity takes about 10 s to
  rebuild on every env construction (decision 10), and the trex neck probe costs 20–25% of step time (readiness
  review, at `b358a46`). *2026-09-29: CU-14b keeps one identity build per process and species, so only the first
  construction in a process rebuilds it (§3.2, CU-14 row).*
- **Cost (estimates):** a trex 3M-step pilot on one env takes about 2.5 h flat or 4.3 h on terrain; these rates are
  reset-bound, measured from a falling fixture on 4 cores; G1's deliverable pair takes 4–14 h per species; the
  recommended chain takes 9–28 h, as an upper bound.

### 5.3 The survey's nine live defects

| # | Defect | Status |
|---|---|---|
| 1 | Every Ray PPO trial crashes on `ent_coef_end`: `ray_tune.py:714-759` lacks the pops at `train_base.py:354-355`. Present since at least 2026-08-09 | Disappears with PR-A |
| 2 | `render_mode='human'` crashes on its first step (`base_env.py:1558`; `mujoco.viewer` is never imported) | Live when re-checked on 2026-09-26; fixed by CU-2 (landed as #568 on 2026-09-29, with its KNOWN_ISSUES entry deleted) |
| 3 | The JAX in-training gate ignores `min_avg_forward_vel` (`jax_curriculum.py:455-486` vs `curriculum/manager.py:340-348`); it would ignore `min_success_rate` too, but the CLI never gates the final stage, where that key is set | Disappears with PR-B (deleted by it, 2026-09-28, with its KNOWN_ISSUES entry; D-D17) |
| 4 | The MJX step never pays dibothrosuchus `snap_snout_proximity_weight` (`mjx_env.py:1303-1306`), although the CPU eval that gates it does | Disappears with PR-B (deleted by it, 2026-09-28, with its KNOWN_ISSUES entry; D-D17) |
| 5 | Ray post-sweep metrics for compsognathus are 2× off: the notebook's `LocomotionMetrics()` defaults dt to 0.01, but compsognathus runs at 0.02 | Disappears with PR-A. The related `sim_dt` default (`stage_artifacts.py:151`) survived it; fixed by CU-2 (landed as #568 on 2026-09-29, with its KNOWN_ISSUES entry deleted) |
| 6 | The Drive summary drops every current-layout sweep (NB2) | The reader stays, and no new sweeps are written. A read-only Drive listing on 2026-09-26 found no current-layout sweep folder, so there is no present impact; its KNOWN_ISSUES entry goes with PR-A (CU-15) |
| 7 | RESUME can retrain over a node that already has a verdict | Fixed by #558 (D-D16), hardened by #559 |
| 8 | A `QUICK_TEST` stance can certify and become a `TRUNK_FROM = "auto"` parent | Fixed by #558: quick tests live under `<algo>_quick_test/` |
| 9 | 7 of the 12 sweep configs crash every stage-3 trial: they sample **18** env keys that no constructor accepts, e.g. `env_prey_distance_min`. Re-counted 2026-09-26; the survey critic's "22" was a miscount | Disappears with PR-A, which also deletes the replaced KNOWN_ISSUES entry (the stage-3 sweep-keys entry under Sweeps / infrastructure; §3.5) |

Defects 1, 3, 4 and 9, together with the never-executed notebook copies behind 5, make up the drift list PR-B records.
PR-B's CHANGELOG entry lists it (2026-09-28).

### 5.4 What the #558 reviews found

PR #558 merged before its validation and review finished; the reviews ran on the merged code and on each follow-up round:

| Review | Raised | Confirmed | Severity | Classes of defect |
|---|---|---|---|---|
| 1, of #558 after the merge | 14 (plus 4 lenses that found nothing) | **8** (6 refuted) | 1 should-fix, 7 minor | **Resume logic:** a truncated final pair counted as finished, so JUDGE crashed on every Run all. **Quick-test tree:** with `RUN_ID = ""`, a `QUICK_TEST` toggle reused the memo id in the other tree, and quick tests count one another as replicates. **Docs truth:** the CHANGELOG and a test docstring said the verdict hashes the final pair; a message promised a budget the cell now refuses; the resume recipe was false under `RETRAIN_FROM`; D-D16 was missing from the decision lists. **Tests:** a reordering mutant survived |
| 2, of follow-up round 1 (`ee91f51`) | 16 (plus 3 lenses that found nothing) | **14** (2 refuted); 12 distinct, because two findings were raised twice | As verified, 2 medium and 12 low; the two mediums are one finding (the `TRUNK_FROM = ""` advice) raised twice. Raised as 5 medium, 9 low | **Code:** the chain loop still judged a broken final pair, and the advice "set `TRUNK_FROM = ""`" would retrain ancestors the run only reused (found by two reviewers). **Wording:** "the final pair is the one checkpoint written straight to the mount" was false, because the best pairs are too; the interrupted-node message; a status row; the docs index. **Tests:** six mutants survived: a toy zip only, a check pinned by its text, a refusal that could not see training, no negative case, a memo test toggling one of three knobs, and no "warning absent" case |
| 3, a fix check of round 2 (`634e2b3`) | — | **4** (and all 12 fixes it covered confirmed) | — | Resuming over a broken final pair trained an early-stopped node's remaining budget in place, against D-D16. The recipe named the wrong record as the trunk. No test showed that an intact final pair still reaches JUDGE. The D-D16 amendment lacked the per-run replicate scan. Round 3 (`ad4e621`) fixed all four, and each of 15 mutants of the new guards now fails a test |

None of the findings moved a digest. What the reviews left open is decisions 4–7.

### 5.5 CI measurements

| What | Value | Source |
|---|---|---|
| #558's CI (run 36196949228) | SB3 job 46:19; JAX job 50:21 | job timestamps |
| #559's CI (at `ad4e621`) | SB3 job 46:51; JAX job 48:10; all 25 checks green | job timestamps |
| #561's CI (CU-1, at `6ccae23`) | SB3 job 48:54, its new mypy step "Success: no issues found in 359 source files" in 77 s; JAX job 51:20; all 23 checks green; coverage 90 percent | job timestamps and logs |
| #562's CI (CU-3, at `db854c1`) | SB3 job 47:57, its mypy step "Success: no issues found in 361 source files" in 75 s; JAX job 50:05; all 23 checks green; coverage 90 percent | job timestamps and logs |
| #563's CI (the release cut, at `29cbc2d`) | SB3 job 46:21, its mypy step "Success: no issues found in 361 source files" in 70 s; JAX job 35:47; all 23 checks green; coverage 90 percent | job timestamps and logs |
| #564's CI (PR-A, at `3c5eab7`) | SB3 job 47:01, its mypy step "Success: no issues found in 334 source files" in 68 s; JAX job 39:11; all 23 checks green; coverage 91 percent; reduced depth, without `full-ci` | job timestamps and logs |
| #565's CI (PR-A2, at `647ca1f`, run 36360108931) | SB3 job 34:52, its mypy step "Success: no issues found in 335 source files" in 41 s; JAX job 34:43; all 23 checks green; coverage 91 percent; full depth, with `full-ci` | job timestamps and logs |
| #566's CI (PR-B, at `c8b66a6`, run 36376798324) | SB3 job 37:03, its mypy step "Success: no issues found in 309 source files" in 53 s; no JAX job; all 22 CI jobs green; coverage 91 percent; reduced depth, without `full-ci` | job timestamps and logs |
| #567's CI (PR-G0, docs only, at `fcf9f2d`, run 36453930456) | SB3 job 46:07, its mypy step "Success: no issues found in 309 source files" in about 62 s; no JAX job; all 22 CI jobs green; coverage 91 percent; reduced depth, without `full-ci` | job timestamps and logs |
| #568's CI (CU-2, at `d8ea2a6`, run 36486171082) | SB3 job 37:11, its mypy step "Success: no issues found in 309 source files" in about 51 s; no JAX job; all 22 CI jobs green; coverage 91 percent; reduced depth, without `full-ci` | job timestamps and logs |
| #569's CI (CU-4, at `9d8b53f`, run 36515741479) | SB3 job 48:37, its mypy step "Success: no issues found in 313 source files" in about 71 s; no JAX job; all 22 CI jobs green; coverage 91 percent; reduced depth, without `full-ci` | job timestamps and logs |
| #570's CI (CU-14b, at `e62ce6d`, run 36611580600) | SB3 job 17:31, its mypy step "Success: no issues found in 314 source files" in about 65 s; no JAX job; all 22 CI jobs green; coverage 91 percent; reduced depth, without `full-ci` | job timestamps and logs |
| #571's CI (ROW-16, at `6ae87e4`, run 36639964231) | SB3 job 13:09, its mypy step "Success: no issues found in 316 source files"; the plant-contract job 5:20, its new step "Verify the digest snapshot golden" 50 s ("848 lines, 0 errors", current); no JAX job; all 22 CI jobs green; coverage 91 percent; reduced depth, without `full-ci` | job timestamps and logs |
| #572's CI (CU-7a, at `79af481`, run 36653953530) | SB3 job 17:30, its mypy step "Success: no issues found in 316 source files"; the plant-contract job 5:25, its digest step 50 s ("848 lines, 0 errors", current), its pytest step 187 passed; no JAX job; all 22 CI jobs green; coverage 91 percent; reduced depth, without `full-ci` | job timestamps and logs |
| #573's CI (CU-11, at `d4496a5`, run 36680247363) | SB3 job 17:47, its mypy step "Success: no issues found in 316 source files"; the plant-contract job 5:53, its digest step 73 s ("932 lines, 0 errors", current), its pytest step 195 passed; no JAX job; all 22 CI jobs green; coverage 91 percent; reduced depth, without `full-ci` | job timestamps and logs |
| CU-9's local measurement (at `d4496a5` with the CU-9 patch) | the CI-like union 19,626 statements, 2,405 missed, 87.75 percent against 91.24 before; plant-contract selection 3:14, shared suite 7:44, SB3 integration 11:36, run in parallel on 4 CPUs | local runs |
| #574's CI (CU-9, at `8465ace`, run 36715850832) | SB3 job 13:47, its mypy step "Success: no issues found in 317 source files"; the plant-contract job 5:51, its digest step 72 s ("932 lines, 0 errors", current), its pytest step 195 passed; no JAX job; all 22 CI jobs green; coverage 88 percent (19,626 statements, 2,405 missed, as measured locally); reduced depth, without `full-ci` | job timestamps and logs |
| Earlier PRs (SB3 / JAX job) | #551 35:37 / 53:34; #556 47:09 / 49:05; #557 30:24 / 47:26 | consolidation plan status rows |
| Job that sets the finish time | JAX, in 12 of 15 PR/push runs (18 runs, #1210–#1227) | CI-cost inventory |
| Median PR/push wall time | 52.3 min now; estimated 46.0 min after PR-B (per-run saving 0–18.9 min, median 5.9) | same |
| Runner time per PR/push run | 165.7 → 116.7 min (−30%, about 1,900 runner-min a week) | same |
| Nightly wall time | 58.1 min, unchanged; the full SB3 job (57.5 min) is already the longest | same |
| `test-sb3` median | 45.25 min; it becomes the critical path | same |
| `test_jax_trainer.py` | About 20 min for 56 tests (the whole JAX job: 566 passed in 46:31) | survey, job 107921660985 |
| `test_compsognathus_training.py` | 13.8–21.6 min of the SB3 integration step | removal plan |
| `test_phase_c_interface.py` in the SB3 list | 145 s | survey critic |
| Plant-contract job's pytest step | 192 s, re-running files the shared matrix already runs three times | survey critic |
| #557's PR run (36087176501) | 23 jobs, 146.1 job-min. The 15 species matrix jobs took 0.8–1.5 min each (16.3 job-min), about 35 s of it setup | survey |
| mypy | The lint step takes about 13 s and reports no issues, because SB3 is absent. With SB3 2.9.0 and torch 2.14.0+cpu, re-measured 2026-09-26 at `f850815`: "Found 21 errors in 6 files (checked 357 source files)", 46–57 s cold in two local runs (mypy 2.3.1, JAX installed, ray and wandb absent; decision 8). Per file: `curriculum/advancement.py` 7, `scripts/widen_checkpoint.py` 5, `diagnostics.py` 4, `curriculum/schedules.py` 2, `tests/test_widen_checkpoint.py` 2, `command_frame.py` 1. All are type-only: `BaseAlgorithm` lacks `ent_coef`/`clip_range`/`log_ent_coef`; read-only callback properties are set in the SB3-absent fallback; state dicts are typed as `Tensor`; some Optional values are unchecked. The hand-kept count has drifted from 13 (`5f7318d`) to 21, and one PR saw 22. At `be63a58` with the harness added (§4.4): "Found 21 errors in 6 files (checked 358 source files)", at the same locations; the harness adds none. Locations at `f850815` (unchanged at the follow-up, which also reports 21): `command_frame.py:156`; `diagnostics.py:222` (×2) and :223 (×2), inside the SB3-absent fallback at :218-223; `curriculum/schedules.py:180,210`; `curriculum/advancement.py:519,537,539,555,559,560,617`; `scripts/widen_checkpoint.py:762,804,807,817` (×2 at :817); `tests/test_widen_checkpoint.py:254,330`. Re-measured 2026-09-26 at `8e03483` (CU-1): 21 errors in 6 files locally, as above; 23 errors in 8 files in `test-sb3`'s environment (Python 3.12, numpy 2.5.3, torch 2.13.0+cpu, ray 2.58.0, 66 s cold), the extra two at `harnesses/freeze_recovery_gate.py:469` and `scripts/sweep/ray_orchestration.py:247`; a lint-job mirror (Python 3.11, ruff 0.16.9, mypy 2.3.1, no SB3) reports none. After CU-1, all three report no issues in 359 files; after CU-3 (#562), 361, in CI's SB3 job too; after PR-A, 334 in all three; after PR-A2, 335; after PR-B, 309 | local runs |
| Tests that run in no job | 6: 3 pandas-gated (`test_sweep_results.py`), 2 `google.auth` (`test_sweep_orchestration.py`), 1 JAX parameter (`test_obs_functions.py`). PR-A deletes the first five; the JAX parameter stays skipped by design | survey |
| Log readability | `addopts = "--tb=short -q"` (`pyproject.toml:132`) cancels all six CI `-v` flags, so logs show no test ids, durations or skip reasons. Since CU-1 every CI pytest step passes `-vv -rfEs --durations=30`, so the logs show each test id, the failure, error and skip lines, and the 30 slowest tests | verified |

### 5.6 Digest facts that constrain cleanup

| Digest | What enters it | Consequence |
|---|---|---|
| `policy_interface_sha256` (plant identity) | Source tokens of each env's `_get_obs` and `_scale_action` (`policy_layer.py:360-361`); of `_cache_ids` for SB3-only species (:397), so declaring a dual species SB3-only adds it to the payload; and of `reset` under the home-keyframe-residual action mapping (:398-399). For the four dual species, also the MJX functions and the live probe (:364-391, :400-404). Trex's low-pass functions (:405-418). The `command_frame` constants (:15). The declared backends (:355) | A token edit (refactor, annotation, moved function or formatter change) moves the plant identity and every task digest built on it. Comments, docstrings, indentation width and module paths do not count (`digests.py:98-166`) |
| `task_sha256` | Constructor defaults overlaid with `[env]` (`task_fingerprint.py:117-171,219`), including the `foot_contact_*` keys. The policy-interface digest. For pushed stages, `SCHEDULE_IMPLEMENTATION` (:71, :212) and the push parameters. The fingerprint backend value (`"stable-baselines3"`, defined twice) | Deleting SB3-dead kwargs moves it (measured `083e2966` → `abfb339f`) |
| Gate digest | The gate view of `[curriculum]` (`gate_schema.py:302`) | `extends` must never inherit `[curriculum]` |
| `hyperparameters_sha256` | `[ppo]`/`[sac]` (`config.py:528`) | An inherited table must reproduce the key order |
| `recipe_sha256` | The recipe file's bytes (`train_behaviors.py:363`) | `configs/*/behaviors/*.toml` are byte-frozen |
| `behavior_identity` | Raw bytes of `behavior_env.py`, the species env module, `direction_commands.py` and `terrain.py` (`behavior_env.py:277-302`). `terrain_sampling.py` has its own byte identity (:83-86) | Comment-only edits move it: comment-only MJX edits in four env files moved all 44 of their identities. PR-9 deletes this identity |
| Reward capture (CU-11, #573) | Each stage's reward, info values, flags and termination reasons under a fixed capture (a roll, one-step probes and posed states), rounded | An `[env]`, reward, termination or MJCF change moves it: regenerate the golden in the same PR; a refactor proves "no number moved" with `--exact` on the base and the head |
| Nothing | `[jax]` tables; sweep, Vertex and mjlab code; the library version (`config.py:788`) | Safe to remove or change |
| Manifest consistency | `training_backends` must equal the env's `supported_training_backends` (`plant_contract/manifest.py:88-93`) | Do not declare a dual species SB3-only outside a revision bump |
| Provenance (not a digest) | The `_DEPENDENCY_PACKAGES` key set | A changed key set records `environment_drift` on resume |

## 6. Lessons learned

### 6.1 Process

- **Validate, review, then open, and do not merge while validation is still running.** #558 merged before its
  validation and review finished. The post-merge reviews found 8 issues, then 14 confirmed findings (12 distinct) in
  the first fix round and 4 in the second. Each fix round also introduced defects: round 1's "set `TRUNK_FROM = ""`"
  advice would have retrained ancestors that the run only reused. If a PR must open early, its body should say what is still running.
- **Mutation-check new guards.** The second review (of round 1, `ee91f51`) found six test gaps that let mutants
  survive: a toy zip fixture, a check pinned only by its source text, a refusal test that could not see training, and
  missing negative, memo-knob and warning-absent cases. In round 3 (`ad4e621`), each of 15 mutants of the new guards
  failed a test: unit tests of `checkpoint_pair_problem` (`test_curriculum_checkpoints.py`) caught the library-check
  mutants, and tests that run the cells against checkpoint files on disk caught the rest.
- **Re-derive numbers before they go into a living doc.** The survey critic corrected the survey's code shrink from
  −1.5–2k to −1.0–1.2k lines, and found that 23.6 of the "25.6 MB freed" frees nothing, because those GIFs share
  blobs with `results/`. Its own count of 22 bad sweep keys was in turn wrong: the removal plan and a 2026-09-26
  recount find 18 entries in 7 JSONs. The readiness review found that "pilot data exists on Drive" is false. The
  harness's `--skip-behaviors` run measured 51 s, not the 38 s first reported.
- **Know which checkout is under test.** An editable install resolves `environments` to the checkout it was installed
  from, so a test or harness run from another worktree without `PYTHONPATH` mixes two checkouts. The harness prototype's
  `--repo` did exactly that and could print a false "no diff". Untracked `build/lib` copies also misled plain `grep`.
  Assert `environments.__file__`, and prefer `git grep`.
- **Eval-only checks are cheap and decisive; run them where the data lives.** The walker check took minutes per walker
  on a CPU runtime and settled what statue results and an anecdote had left open. Large files (checkpoints, logs)
  cannot be pulled out of Drive for review, so run eval-only checks on Colab next to the data and copy back a small
  summary; the walker summary was 19.6 KB.

### 6.2 Technical

- **CI blind spots hide real errors.** mypy runs without SB3; notebooks are only AST-parsed; three `ray_orchestration`
  functions were never executed; six tests ran in no job; logs hide test ids. Make each signal visible (CU-1, CU-4)
  before trusting it.
- **Digests over source tokens or file bytes turn harmless-looking edits into identity changes.** A comment in a
  byte-hashed file, a ruff release, or removing a dead kwarg can each move a digest. Know what enters each digest
  (§5.6), and run the snapshot harness before and after any cleanup that claims to move nothing.
- **Mirrored backends drift silently.** Four copies drifted into live defects with no CI failure: Ray PPO (broken
  since at least 2026-08-09), the JAX thresholds, the snout proximity and the sweep keys. Code that no job actually
  trains with rots. A re-add needs one shared definition and a real training trial in CI (§4.10).
- **Statue measurements do not predict the policy.** The trex statue survived at 100 mm cells; the walker did not.
  When the question is about the walker, measure the certified walker.
- **Writes straight to a FUSE mount are not atomic.** Periodic pairs are staged, but the final and best pairs are not.
  A file whose presence marks a state (finished, judged) must be published atomically, or checked where it is read.
  Since CU-3 (D-D20) the final and best pairs are staged too, and a fixed-name pair is published so that a reclaim
  leaves at worst an incomplete pair its readers reject, never a mixed pair they accept. Staging
  also moved when the final zip first appears: without a placeholder at the start of the save, a reclaim before the
  zip landed left no final zip, which the RESUME cell reads as an interruption rather than an early stop (the CU-3
  review, 2026-09-26). A marker file must appear when the state it marks begins, as before the change.
- **A pinned tool can widen its own reach.** Ruff 0.16 formats notebooks and the Python blocks in Markdown files, so
  pre-commit hooks pinned to it without a `files:` scope would have rewritten all four notebooks and 15 Markdown files
  (two of them frozen investigations) that CI never checks (CU-1).
- **Measure in the job's own environment.** numpy 2.5, which needs Python 3.12, reports a type error that numpy 2.4.6
  under Python 3.11 does not, and ray adds one of its own: the 21 local mypy errors were 23 in `test-sb3` (CU-1).
- **A whitespace hook can move a plant digest.** The plant identity hashes MJCF bytes, and three compsognathus MJCF
  files end without a newline, so `end-of-file-fixer` under `pre-commit run --all-files` would move both compsognathus
  identities. Pre-commit now excludes the digest data files, and `test_ci_tool_pins.py` checks the exclusion against
  the plant manifest's recorded sources (CU-1). The byte-hashed Python modules stay under the hooks: CI's pinned ruff
  keeps them from changing, and any edit to them moves the behavior identities anyway.
- **`-rs` replaces pytest's default `-rfE`; it does not add to it.** The plan's own "add `-rs`" would have dropped the
  failure lines from CI's short summary; CI passes `-rfEs` (CU-1).

## 7. Do-not-do list

- **One `_scale_action`/`_get_obs` in the base class, or removing the reset height channel.** It moves every species'
  policy-interface digest and task fingerprint, stranding the certified walkers and the robot stance. No tool can
  widen across a same-width interface revision. Batch it into the queued revision instead (§4.1).
- **Deleting the SB3-dead `foot_contact_*` kwargs or TOML keys.** It moves the trex and dibothrosuchus task digests.
  Document them instead: PR-B rewords the `foot_contact_*` LOW under Training / RL in KNOWN_ISSUES (JAX-only knobs that stay because they are
  `task_sha256` inputs); the notes in `trex_env.py` and `dibothrosuchus_env.py` go in PR-C (byte-frozen until then);
  the fingerprint docstring and, unless PR-B deletes it first, the test comment go in CU-7. (PR-B reworded the LOW and
  deleted the test comment with its MJX block, 2026-09-28; the fingerprint docstring stays CU-7's.)
- **An `extends` that inherits `[curriculum]`.** It moves the gate digest. Inherit another table only where the
  stage-digest snapshot shows it unchanged, key order included. (`[jax]` is rejected after PR-B anyway.)
- **A library guard refusing reuse when `run.timesteps` < `curriculum.timesteps`.** It would reclassify existing Drive
  runs. This is not decision 5.
- **Editing, reformatting or moving the frozen MJX core, or declaring a dual species SB3-only (or giving it
  `training_backends`) outside a revision bump.** Each either moves four species' digests or makes
  `plant_contract/manifest.py` raise.
- **The tombstone alternative** (frozen literals in place of the core). It edits the hasher, stops the live probe,
  broke 15 plant-contract tests, and its verifier did not reproduce it.
- **Removing anything §4.9 keeps, dropping `gcp` from `[all]` or the Drive summary's sweep branch, or reusing the
  stale `JAX` branch (`f900d3a`) as the archive point.** Removing §4.9 items stops old records validating or logs
  drift on resume. Dropping `gcp` silently disables GCS upload. Dropping the sweep branch makes `discover_runs` walk
  into `sweeps/`. The `JAX` branch is not an ancestor of `main`. (D-D17 retires GCS upload itself: `[gcp]` leaves in
  PR-A2 together with the upload code, never on its own; carried out that way.)
- **The full D-D7 package move now.** It is line-neutral XL work that rewrites 1.3–1.9k lines of pins while walker
  sessions 4–6 are pending. Decide it after PR-13. (Session 6 finished on 2026-09-28.)
- **The session-half move (cells 7, 9 and 10).** It adds 50 to 100 lines net. Only its preflight slice is in CU-6.
- **A `RUN_PROVENANCE` dict.** It turns the loud refusal of `SEED`/`N_ENVS` drift into silently wrong provenance. That
  is also why decision 4 avoids `provenance.json`.
- **The frozen-null hook merge as proposed.** It blocks the working trex manual freeze. Only the panel-roll sequence
  is worth sharing.
- **Collapsing the harness shims** (low value; breaks a command path a frozen investigation cites), or **the pin
  budget as proposed** (it overlooks four tombstone tables and narrows checks to code cells).
- **A separate rewording pass over every Drive pilot-data sentence, CHANGELOG included.** The prescribed action is the
  same either way; CU-16 fixes only the false existence claims in living docs.
- **Unifying velociraptor's −0.342 or the brachiosaurus 1.2 target** (both are behavior changes), or **trimming the
  reporting re-exports** (`backfill_gate_verdict` uses them).
- **Deleting only the website copies of the unused GIFs.** They share blobs with the `results/` copies, so nothing is
  freed; delete both copies or neither.
- **Moved off this list:** gating compsognathus real training behind the depth switch is now decision 10, because its
  reason (the JAX job sets the wall time) ends with PR-B. "Merging the JAX PPO loops or reward composer" and "the JAX
  artifacts module move" are moot.
