# Cleanup and backend retirement plan (2026-09)

**Status**: living plan, updated 2026-09-26. `main` = `74ba16c` (#561, CU-1, merged 2026-09-26 15:46 UTC). Written at `be63a58` (#559, the
#558 follow-up, merged 2026-09-26 03:42 UTC; commits `ee91f51`, `634e2b3`, `ad4e621`). #558 (notebook safety, D-D16) merged 2026-09-25 22:44 UTC as `f850815`.
Line numbers are at `f850815` unless marked "at the follow-up", which equals `be63a58` for every file #559 touched
(`sb3_training.ipynb`, `curriculum/__init__.py`, `curriculum/checkpoints.py`, `test_curriculum_checkpoints.py`,
`test_sb3_notebook_pins.py`, `CHANGELOG.md`, `docs/NEXT_STEPS.md`, `docs/CONSOLIDATION_PLAN_2026_09.md`,
`docs/BEHAVIOR_RECIPES_PLAN.md`, `docs/README.md` and `website/docs/training/recipes.md`); every other file is
byte-identical at `f850815` and `be63a58`. Line numbers drift with every merge, so re-read before editing. This plan
landed as #560 (merged 2026-09-26 05:03 UTC as `8e03483`). On 2026-09-26 the maintainer took three of §2's decisions
as D-D17 (row 1, widened), D-D18 (row 8) and D-D19 (row 3), and closed #527 and #498 (row 19); CU-1 is the first PR
after it (§3.1). CU-1 landed as #561 the same day; the maintainer then took decision 7 as D-D20, carried out by CU-3,
and moved the Vertex route and GCS upload out of PR-A into a PR of their own, PR-A2 (D-D17 amended; §4.5).

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

1. **What is left.** The #558 follow-up landed as #559 on 2026-09-26. What remains is the backend retirement in three
   PRs (PR-A, PR-A2, then PR-B), the release cut (D-D19), and sixteen smaller PRs (CU-2..CU-17, §3); the CI-signal
   PR (CU-1, D-D18) landed as #561. The retirement makes four of
   the survey's 21 waves wholly moot, most of C15 and half of C1. It also deletes five of the survey's nine live
   defects along with their code, and #558 already fixed two more.
2. **Order.** CU-1 (mypy with SB3, readable CI logs; D-D18) came first, as #561; CU-3 (atomic run-tree records and
   checkpoint pairs; D-D20) follows, then the CHANGELOG release cut (D-D19). Next is PR-A (Ray Tune, the Vertex AI
   tuning sweeps and mjlab; its acceptance runs the digest-snapshot harness this plan adds, §4.4), then PR-A2 (the
   single-job Vertex route and GCS upload, which D-D17 also retires), then PR-B (JAX/MJX, keeping a frozen interface
   core). After that come the golden-trace, one-derivation and `extends` PRs
   that consolidation PR-8, PR-9 and PR-11 need, then the CI structure, and docs last.
3. **Measured payoff.** PR-A and PR-B delete 72 whole files and 28,919 lines (PR-A's and PR-B's, measured before
   D-D17 was taken; the Vertex route and GCS upload leave in PR-A2, which derives its own). That covers about 17,459 of the 71,682
   non-test library lines (24%) and about 12,090 test lines.
4. **CI savings (estimated from 18 runs, #1210–#1227).** Jobs: 23 → 22. Runner time per PR/push run: 165.7 → 116.7 min
   (−30%, about 1,900 runner-minutes a week). Median wall time: 52.3 → 46.0 min. The nightly run is unchanged. Nearly
   all of this comes from PR-B; PR-A (Ray, Vertex and mjlab) saves tens of seconds.
5. **The larger payoff is no longer mirroring core changes into untrained code.** 53–54 of the 128 first-parent units
   on `main` since 2026-07-01 touched backend files. Four mirrored copies had already drifted into live defects.
6. **The surviving cleanup is mostly about correctness, not size.** It removes roughly 830–900 lines of code, tests
   and notebook (about 530–750 if the optional CU-15 reader move is done), and about 900 lines of docs (survey
   estimates, re-summed without the moot items). Its most important items are the `render_mode='human'` crash, atomic
   final and best checkpoint pairs, and a golden trace for reward and termination, which no digest covers today.
7. **The overriding constraint: no digest may move.** The policy-interface digests of trex, velociraptor,
   brachiosaurus and dibothrosuchus hash MJX source tokens and run an MJX probe. A 473-line frozen MJX interface core
   therefore stays, unedited, until each of those species reaches its next deliberate policy-interface revision.
8. **No certified run is affected.** With the core kept, two independent prototypes produced byte-identical snapshots
   of every plant, stage, behavior and recovery digest. Every certified run on Drive stays valid, including the 18
   certified stage/digest prefixes in [NEXT_STEPS.md](NEXT_STEPS.md).
9. **Before PR-A merges,** the maintainer settles the archive points (§2 row 2, still open) and pushes any tag they
   call for (§4.8). D-D17 is taken (2026-09-26), and the single-job Vertex route and GCS upload go with PR-A2 (§2, row 1; §4.5). The digest-snapshot harness is already in the repository
   (added with this plan, 2026-09-26; §4.4). The frozen-core reference copies are not: PR-B rebuilds them from
   `f850815` (§4.2).
10. **Direction/terrain is now gated by physics, not cleanup.** All three certified walkers survive the plane in every
    episode, but only 1 of 39 flat-heightfield episodes reached full horizon (§5.1). The heightfield contact
    investigation and the speed and map re-derivation must come before PR-11.

## 2. Decisions needed before continuing

D-D17 was reserved for the retirement, because PR-A's text must cite it. The other rows take D-D18 onward, in the
order taken, even if one is taken before PR-A merges. On 2026-09-26 the maintainer took row 1 as D-D17, row 8 as D-D18
and row 3 as D-D19, and recorded row 19 (closing the stale PRs) as an operational choice rather than a D-D row; each
outcome is appended to its row below. Record every row append-only in
[BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2 and in the consolidation plan's decision table.

| # | Decision | Options | Recommendation (why) | Blocks |
|---|---|---|---|---|
| 1 | **Proposed D-D17: retire the backends.** The maintainer's direction is to remove Ray Tune, the Vertex AI hyperparameter-tuning sweeps, mjlab and JAX/MJX now, and add them back once every species and behavior is learned. **Open scope question** (removal critic item 12): does the single-job Vertex route stay? That route is `scripts/setup_vertex_ai.sh` (184 lines; submits SB3 `train_sb3.py train`/`curriculum` jobs at :127-162; one sweep hint at :183) and the `Dockerfile` (44 lines; `.[train,viz,gcp]` at :31). | (a) D-D17 as drafted (§4.5), keeping the route. (b) Also delete `setup_vertex_ai.sh`, the `Dockerfile`, `.dockerignore`, the kept parts of `vertex-ai.md`, and `google-cloud-aiplatform` from `[gcp]`. | **(a).** The route runs SB3 only and references no sweep code, and removing it later is a separate, reversible choice. It still needs the maintainer's explicit yes, because the original scope named both files. Also confirm the appended note to [investigations/TREX_REVIEW_2026_07.md](investigations/TREX_REVIEW_2026_07.md) (§4.5). **Taken 2026-09-26 as D-D17, wider than (b):** the maintainer also retires GCS artifact upload (`curriculum --gcs-bucket` / `--gcs-project`, the helpers they reach and the whole `[gcp]` extra), so PR-A's delete list grows beyond §4.5's (see its note). **Split the same day:** those two leave in a PR of their own, PR-A2, after PR-A (D-D17 amended). | PR-A and PR-B. Their FROZEN docstrings, amendments and CHANGELOG entries cite D-D17. |
| 2 | **Archive points.** | (a) One tag at PR-A's first parent. It misses any JAX change that lands between the two merges. (b) Two annotated tags: `archive/secondary-backends-2026-09` at PR-A's first parent and one at PR-B's first parent (e.g. `archive/jax-mjx-2026-09`), each pushed before its PR merges, with both SHAs in D-D17 (critic item 5). (c) No tags; merge commits keep the SHAs reachable. | **(b).** It is cheap, and each half can be recovered from a named point. The remote had 0 tags on 2026-09-25. Do not reuse the stale `JAX` branch (`f900d3a`, 2026-02-03), which is not an ancestor of `main`. | The PR-A and PR-B merges. Maintainer action. |
| 3 | **Cut the CHANGELOG release?** `[Unreleased] (v0.3.8)` spans `CHANGELOG.md:8-3805`: 3,798 of 4,009 lines (3,860 of 4,071 at the follow-up). Two more undated `[Unreleased]` headings sit at :3806 and :3934. The last dated release is 0.2.0 (2026-02-09). Since `c0e9b52` (2026-08-02), `.devN` in `pyproject.toml:7` and the heading move together, with no tags. | (a) Before PR-A: date the three headings, open an empty `[Unreleased]`, bump to `0.3.9.dev0` (or `0.4.0.dev0` if the terrain path is ROADMAP's v0.4.0), and tag `v0.3.8`. (b) The same, without a release tag. (c) Keep growing the section. | **(a) or (b), before PR-A**, so the retirement's Removed and Migration entries head a short section. A release tag is the maintainer's call; archive tags are not release tags. The version is recorded in `stage_config.json` (`config.py:788`), in the HPT metrics payload (`train_base.py:1503`) and in `provenance.json`'s `dependency_versions` (`"mesozoic_labs"`, `result_bundle/constants.py:55`). It enters no digest; a resume after the bump only records it as environment drift, as a new `repository_commit` already does. **Taken 2026-09-26 as D-D19: (a), with `0.3.9.dev0` next; the maintainer tags `v0.3.8`.** | Nothing hard. It only decides where PR-A's entries go. |
| 4 | **Persist the resolved trunk run on disk.** The chain loop never follows the run's own `ancestors/` records, so a resume must pin `TRUNK_FROM` to the trunk that the interrupted session's resolve cell *printed*. If that output is lost, the operator has to hunt through the ancestor records (`CHANGELOG.md:1712-1726` at the follow-up; round-3 check). | (a) The resolve cell writes the resolved trunk into a run-level sidecar, and the RESUME cell and the recipe read it. (b) Keep the printed output plus the recipe. (c) Put it in `provenance.json`. | **(a), as a small notebook PR.** First check how the run manifest treats a new run-level file: `result_bundle/manifest.py` hashes the run tree, and a `complete` bundle is immutable. Not (c): provenance keys drive the drift records and the audit. | Nothing. It removes an operator error mode (KNOWN_ISSUES, §3.5). |
| 5 | **Library guard against `resume_same_stage` into a judged directory.** `config.refuse_occupied_stage_dir` (`config.py:403-427`) lets any same-stage resume through, even with `gate_verdict.json` present. The SB3 notebook has guarded against this since D-D16; the CLI (`--load-mode resume_same_stage`) has not. | (a) Refuse when `gate_verdict.json` exists. This amends D-A20 and covers the CLI. (b) Keep the notebook-only guard. | **(a)**, once you have confirmed that no legitimate writer resumes into a judged directory: `train_curriculum`'s in-training verdict (D-A5), backfill, widen. This is not the do-not-do short-budget guard (§7). | Nothing on the terrain path. It closes a CLI hole (KNOWN_ISSUES, §3.5). |
| 6 | **Judge an unjudged `RUN_DIR` node before any trunk reuse.** When a trunk certifies a node, `RUN_DIR`'s own copy is bypassed even if it holds an intact final pair and no verdict. One case is a node that `RETRAIN_FROM` covered and that was then resumed. The follow-up documents a manual route: set `BEHAVIOR=<node>`, then start a fresh `RUN_ID` trunked from this run. D-C13's `refuse_trunk_over_unjudged_widened_root` (`result_bundle/reentry.py:255`) covers only a widened root. | (a) Generalise that refusal to any such node, with the `BEHAVIOR=<node>` remedy. Not `TRUNK_FROM = ""`, which retrains ancestors the run only reused (review 2). (b) The chain loop judges such a node before consulting the trunk. This amends the loop order fixed by D-A17/D-C13 and the BEHAVIOR_RECIPES_PLAN §4.7 pins. (c) Keep the manual route. | **(b) if the maintainer accepts amending the loop order; otherwise (a).** (b) removes a two-session manual route and makes D-C13 a special case. | Land before PR-13 edits the chain-loop cell. The KNOWN_ISSUES trunk entry covers this case (§3.5). |
| 7 | **Stage the final and best checkpoint pairs atomically.** Only the periodic pairs are staged and published atomically (`train_base.py:727-760`). `best_model` (:691, :710), `robust_best_model` (:724) and the final pair (`_save_final_and_sync_tb`, :918-932) are written straight to the mount. Since the follow-up, the RESUME cell and the chain loop check the final pair with `checkpoint_pair_problem`; nothing checks the handoff pairs. If a reclaim cuts a best pair short while the final pair is intact, RESUME treats the node as finished and JUDGE fails to load the handoff. | (a) Stage them like the periodic pairs, in CU-3. (b) Extend `checkpoint_pair_problem` to the handoff pair. (c) Only a KNOWN_ISSUES entry. | **(a).** The failure was reproduced 2026-09-26 at the follow-up: with an intact final pair and `robust_best_model.zip` cut in half, the RESUME cell prints "Nothing to resume", `select_handoff_checkpoint` still returns the pair, and loading it raises. It is recorded in KNOWN_ISSUES until CU-3 lands (§3.5). File bytes and names are unchanged, so no digest moves. **Taken 2026-09-26 as D-D20 and carried out by CU-3 (§3.2).** | Nothing. It lowers reclaim risk on Colab. |
| 8 | **mypy with SB3 in CI.** The lint job installs only `ruff mypy gymnasium numpy` (`python-ci.yml:82`; mypy at :90-91) and reports no issues. With SB3 installed, mypy finds 21 errors (§5.5; KNOWN_ISSUES, §3.5). | (a) Add a mypy step to `test-sb3`, pinning `stable-baselines3==2.9.0` (as the notebook's install does) and `torch==2.13.0` (as `python-ci.yml:266` already does). (b) Install `.[train]` with CPU torch in the lint job. (c) Keep the hand-kept baseline. | **(a).** Before adding the step, run mypy once in `test-sb3`'s own environment (SB3 2.9.0, torch 2.13.0 CPU, ray and wandb installed, no JAX; `python-ci.yml:266-267`) and fix what it reports. The 21 were measured with mypy 2.3.1, torch 2.14.0+cpu, JAX installed and ray/wandb absent, so the count in the job's environment is unmeasured; five of the errors come from torch's `Tensor` typing. Pin mypy in the step and in the lint job (`python-ci.yml:82` installs it unpinned), and give pre-commit the same version (`.pre-commit-config.yaml:10` pins v1.15.0). The pins stop an SB3, torch or mypy release from turning an unrelated PR red. `test-sb3` is off the critical path until PR-B. **Taken 2026-09-26 as D-D18 and carried out by CU-1 (§3.2).** Measured in `test-sb3`'s environment on `8e03483` (Python 3.12, numpy 2.5.3, torch 2.13.0+cpu, ray 2.58.0, wandb, no JAX): 23 errors in 8 files, the 21 plus `harnesses/freeze_recovery_gate.py:469` (numpy 2.5's `ndarray` typing; numpy 2.5 needs Python 3.12, so the 3.11 lint job resolves 2.4.6) and `scripts/sweep/ray_orchestration.py:247` (seen only with ray installed). **Landed as #561 on 2026-09-26**; in CI the step printed "Success: no issues found in 359 source files". | PR-10 (torch-facing code). It also gives PR-A and PR-B an enforced mypy check. |
| 9 | **Amendments the surviving waves need.** CU-6 moves a slice of D-D7 into the package. CU-13 extends D-D5's `extends` to recovery ← stance. CU-4 deletes the pin that `CONSOLIDATION_PLAN_2026_09.md:247-249` says STAYS. | Append the amendments, or drop those slices. | **Append each amendment when its PR opens.** The consolidation plan's §8 risk "take it for both notebooks" (`CONSOLIDATION_PLAN_2026_09.md:1318-1322`) becomes moot after PR-B. | CU-4, CU-6, CU-13. |
| 10 | **After PR-B, `test-sb3` is the critical path.** `test_compsognathus_training.py` takes 13.8–21.6 min of the integration step. The survey's do-not-do against gating its 12 real-training parametrisations rested on the JAX job setting the wall time, and that ends with PR-B. Measured (survey, locally, 12 passed in 618 s): the six compsognathus_robot parametrisations take 484 s (78%), because `current_plant_identity("compsognathus_robot")` rebuilds the robot plant (147 geoms) in 7.8–9.5 s per call, several times per test; compsognathus takes 0.4–0.5 s per call. Runner variance is about ±40%. | (a) Gate them behind the depth switch; nightly and `full-ci` still run them. (b) Keep them on every PR. (c) Cache the plant identity per process and species (no digest change; it also speeds up behavior env construction, where the robot's identity rebuild costs about 10 s). | **(c) first, then decide (a) after PR-B** from CU-1's `--durations` output. If (a), keep compsognathus rather than the robot on pull requests. | CU-14. |
| 11 | **Heightfield contact parity.** All three certified walkers survive the plane 23/23 and fail on a flat heightfield: full horizon 1/13 for trex, 0/13 for velociraptor and 0/13 for compsognathus (§5.1). Finer cells did not help the trex walker (0/7 at 100 mm), though they helped its statue. | (a) Investigate first, as a dated note under `docs/investigations/`. Cover plane vs heightfield contact (solref/solimp, margins, collision pairs), the terrain settle and spawn clearance (velociraptor's authored −44.6 mm toe penetration, `behavior_env.py:403`), and the compsognathus contact flicker. (b) Proceed and treat terrain later. (c) Train terrain as it is. | **(a).** It can run in parallel with PR-8..PR-10, but it must finish before PR-11, which copies the terrain keys verbatim (`CONSOLIDATION_PLAN_2026_09.md:655-656`). | PR-11's terrain nodes, any terrain pilot, and PR-13's terrain thresholds. |
| 12 | **Split PR-13.** The plan requires PR-11 and PR-12 before PR-13 (`CONSOLIDATION_PLAN_2026_09.md:793`). Its gate-registration half appears not to need PR-12's deletions (inferred from the plan text). | (a) Land the gate-registration half right after PR-11. (b) Keep PR-8 → PR-9 → PR-10 → PR-11 → PR-12 → PR-13. | **(a).** The notebook chain then halts at `follow_direction` for less time. The PR-13 row needs an amendment. | PR-11 scope. |
| 13 | **Recipe speeds and map sizes.** Cruise is set to the locomotion gate minimum, not to the learned gait. Velociraptor: 2.0 m/s against 3.17–3.30 certified (3.5–3.7 measured on the recipes). Compsognathus: 0.08 against 0.34 (4.3×; 0.38 measured). Trex: 1.05 fits the seed-42 walker (1.07) but not seed 44 (1.57). Commands span half to full cruise, so at half cruise the gap is 3.3× (velociraptor) and 8.5× (compsognathus), and velociraptor's tracking reward at gait speed is 4e-4 to 6e-5. Maps are sized for 25 s at cruise, so a walker at its own speed leaves after about 17 s (velociraptor) or 10–12 s (compsognathus), which counts as a non-survival; velociraptor left the map in 9 of 13 flat-heightfield episodes. | (a) Re-derive `cruise_speed`, the scales, extent, grid, `course_distance` and apron from each certified walker. Compsognathus at ~0.34 m/s needs an extent of ~9.1 m. Velociraptor at ~3.2 m/s needs an extent above ~81 m and at least ~1,081 columns. (b) Deliberately command below the gait. | **(a)** for each species that has a walker. Decide the robot once its walker is certified. | PR-11, which copies the 19 per-species scale values verbatim. |
| 14 | **Compsognathus terrain feasibility.** As written, compsognathus_robot cannot pass the certificate on the commanded terrain recipes (`follow_direction_difficult_terrain`, `combined_terrain`, `combined_mixed_terrain`) even with perfect tracking: on `follow_direction_difficult_terrain` it clears the departure rule in 1/5/4/4 of 20 episodes per non-flat family, where 17 are needed for the success bound (10/11/17/15 at the top of the velocity tolerance). Compsognathus passes only if it runs at the top of the ±0.016 m/s tolerance (18/18/19/18; `combined_terrain` 18/20), so it is a risk, not a blocker; `combined_mixed_terrain` fails either way (mixed 14/20). The command-free recipes (`difficult_terrain` and the four single-template presets) clear departure, and `terrain_contact` is exempt from it (`behavior_certification.py:213`). The rule uses apron + blend width (:230), but the generator ignores blend width for three families (`terrain.py:278-282`). Even with a per-family radius, an exact-tracking executor fails sloped (compsognathus 13/20, robot 1/20). The walker also tips over on a flat heightfield (0/13). | (a) Define departure per family from the terrain config, and rescale the maps (row 13). (b) Leave the pair out of the first terrain certification. (c) Drop the rule; G4 does not mention it. | **(a), decided together with row 11**, which may find a physics cause that no rule change fixes. | PR-13's certificate rules and the compsognathus terrain nodes. |
| 15 | **Remaining PR-11/PR-13 choices** (from the readiness review and its critic). Seed terrain per episode (`behavior_env.py:430` omits `_episode_index`, which :432 and :439 include), or document one layout per run. Keep the learning rate at or below the parent's final rate (a constant 5e-5 today against the walkers' 1e-5; compsognathus ≤ 3e-5). Zero `gait_symmetry_weight` for brachiosaurus and dibothrosuchus. Take the family list from the node config, not the report. Use 40 episodes per family with fixed seeds. Make the tolerances and settle/dwell constants threshold keys. Report heading coverage. Add terrain probes for brachiosaurus and dibothrosuchus. PR-11 also owns `make_env` routing to the terrain subclass (`train_base.py:204-229`; the `env_class(**env_kwargs)` call is at :218), the terrain/command TOML-to-dataclass conversion, and a real-PPO smoke of `follow_direction_difficult_terrain` (the only planned one is flat). Raise the compsognathus pair's `n_steps` (1,024) to at least one 1,250-step episode, or record why not. Decide whether behaviors keep 25 s episodes when the parents were certified on 10 s (20 s for compsognathus; velociraptor's certified checkpoint averages 968.6 of 1,000 steps). | — | Take them in the PR-11 and PR-13 reviews. At 20 episodes per family, a 99%-survival policy passes all five families with probability 0.37; at 40 per family with up to 3 falls it passes with 0.997. | PR-11, PR-13. |
| 16 | **The digest snapshot as a CI check.** Nothing in CI pins behavior identities or stage-config views (§4.6 risks), yet every later CU claims "no digest". | (a) Commit the golden output plus a `--check` step: `--skip-behaviors` on pull requests, the full run nightly. (b) Run the harness by hand in each acceptance. | **(a), after PR-B and before CU-8 and CU-11..CU-13.** | Nothing; it makes the "no digest moves" claims checkable. |
| 17 | **Retired-backend metric lines.** After the retirement the generated catalog in the root `README.md` (:105-215) and the website still advertise JAX/MJX success metrics for four species that nothing can train. `species_catalog.py:949-954` requires those rows while the species stay dual. | (a) A `species_manifest.toml` field separating the declared interface backends (`plant_contract/manifest.py:88-93`) from the advertised training backends, so the catalog stops rendering JAX/MJX metrics. (b) Keep them with a note. | **(a), as a catalog PR after PR-B.** The removal inventory found the declared backends list is not a bundle-digest input; confirm with the harness. | Nothing; living docs stay true. |
| 18 | **Branch protection.** On 2026-09-25 `main` had no required checks, although `python-ci.yml:210-212,316` assume them (§4.8). | (a) Turn on required checks once the job names settle (after CU-14). (b) Reword the workflow comments. | **(a)**, a maintainer action. | Nothing. |
| 19 | **Stale open PRs.** #527 ("Research Compsognathus feet and add resumable Colab balance sessions", 2026-09-10) edits `python-ci.yml` and `harnesses/freeze_recovery_gate.py` and adds a notebook. #498 ("Record the August 2026 RL pipeline review", 2026-08-08) adds a review that was never merged. | Close, or rebase and merge. | **Decide before PR-A.** #527 conflicts with PR-A/PR-B in `python-ci.yml` and with CU-9. **Decided 2026-09-26: both closed, each with a comment (an operational choice, not a D-D row).** | PR-A (rebase cost). |

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
   (§4.6).** The release PR dates the three `[Unreleased]`
   headings, opens an empty one and moves the version to `0.3.9.dev0`; the maintainer tags `v0.3.8` on the commit that
   dates its section; like any change that claims to move no digest, the release PR runs the digest-snapshot harness
   on its base and head. PR-B follows PR-A (§4.7). Both use the digest-snapshot
   harness, which is already in the repository (§4.4), for their acceptance.
3. **The surviving waves (§3.2).** CU-2, CU-3 and CU-7 are independent of the retirement and can land any time. Work
   that edits CI structure, or docs the retirement also edits, waits for PR-B.

### 3.2 The surviving waves (relabelled CU-n; "was" names the survey item)

Sizes are survey estimates of net lines unless marked as measured. "No digest" means the PR touches no hashed
callable, no byte-frozen file and no digest input (§5.6).

| PR | Was | What | Why | Size | Risk | When |
|---|---|---|---|---|---|---|
| CU-1 CI signal | C2 (trimmed) | Re-measure the mypy errors in `test-sb3`'s environment (decision 8), then fix them with casts and annotations (the 21 local ones are listed in §5.5). The `advancement.py` fixes must be casts, not runtime guards. The SB3-absent fallback at `diagnostics.py:218-223` gets `# type: ignore[misc,assignment]`. Add the pinned mypy step (decision 8). Keep CI's `-v` and add `--durations=30 -rs`; where test ids are wanted, pass `-vv` (or `-o addopts="--tb=short"`), because `pyproject.toml:132` `addopts = "--tb=short -q"` cancels one `-v`. Pre-commit pins ruff `v0.4.4` and mypy `v1.15.0` (`.pre-commit-config.yaml:3,10`) while CI installs both unpinned (`python-ci.yml:82`); pin one ruff version and one mypy version for both. Closes the two Testing / CI KNOWN_ISSUES entries (§3.5). **As carried out (2026-09-26, D-D18):** 23 errors in 8 files in `test-sb3`'s environment on `8e03483` (§2 row 8), fixed with casts and annotations and no runtime change (the four in the SB3-absent diagnostics fallback with the planned `# type: ignore`); the SB3 job pins `stable-baselines3[extra]==2.9.0` and runs mypy 2.3.1; ruff 0.16.9 and mypy 2.3.1 are pinned in the lint job, pre-commit and the `dev` extra, and `test_ci_tool_pins.py` keeps them, with the SB3 pin, in agreement. The pre-commit ruff hooks are scoped to `environments/`, because this ruff also formats notebooks and the Python blocks in Markdown files: unscoped, it would rewrite all four notebooks and 15 Markdown files, two of them frozen investigations. Every pytest step passes `-vv -rfEs --durations=30`, not `-rs`, which replaces pytest's default `-rfE` and would drop the failure lines. The review of the CU-1 PR added two things: `.pre-commit-config.yaml` joins both path filters of `python-ci.yml`, so a pre-commit-only PR runs the pin checks, and a top-level pre-commit `exclude` keeps every hook off the digest data files (MJCF sources and meshes, recipe TOMLs, plant manifests, `plant_versions.toml`, recovery calibrations; the byte-hashed Python modules stay under the hooks, which CI's pinned ruff keeps from changing them): `pre-commit run --all-files` would have let `end-of-file-fixer` append a newline to three compsognathus MJCF files and move both compsognathus plant identities (reproduced in a scratch worktree; `plant_contract --check` then reports the manifest stale). | CI cannot see SB3 types. Logs show no test ids, durations or skip reasons. Pre-commit reformats 22 files that CI accepts. A pinned ruff also keeps a formatter release from reformatting the token-hashed callables of every species (§5.6). | +40–60 | LOW | Landed as #561 (2026-09-26) |
| CU-2 Live library bugs | C1 (viewer half) + critic item | Add a lazy `import mujoco.viewer` in `render()`'s human branch (`base_env.py:1556-1566`; the imports at :19-21 lack it), one camera helper shared with `_make_camera`, and a test that monkeypatches `launch_passive`. Fix the `sim_dt` default of 0.01 at `reporting/stage_artifacts.py:151` by taking `dt` from a probe env. Both compsognathus species step at 0.02 s, so their summaries print half the sim time, but only where `generate_stage_artifacts` builds its own results (`stage_results=None`, :1463): the Ray and Vertex sweep trials (`ray_tune.py:1025`, `trial.py:233`). The notebook path overwrites `sim_dt` with the env's `dt` (:1242, :1277), and `backfill_gate_verdict.py` persists nothing that uses it, so after PR-A the default is latent. Closes the render and `sim_dt` KNOWN_ISSUES entries (§3.5). | `evaluate` without `--no-render` crashes on step 1. Re-checked 2026-09-26: `mujoco.viewer` is not imported after `base_env`, `train_base` and `evaluation` load (mujoco 3.10.0). | about +35 | LOW (`render` is not hashed; `base_env.py` is not byte-hashed) | Any time |
| CU-3 Atomic writes and handoff pairs | C8 + decision 7 | Add `file_io.atomic_write_json` and `atomic_write_csv`. Use them for `stage_config.json` (`config.py:823`), `metrics.json`, the stance JSON/text writers and the three evidence CSVs, keeping each site's `json.dumps` arguments so the bytes stay identical. Three hand-rolled writers leave `<name>.json.tmp` files that the manifest's `.*.tmp` cleanup misses. That happens only after a crash between write and replace, and the next write overwrites the file (survey critic). Also stage the final and best pairs (decision 7), which closes the handoff-pair KNOWN_ISSUES entry (§3.5). **As carried out (2026-09-26, D-D20):** `file_io` gains `atomic_write_json` and `atomic_write_csv` (and `atomic_write_text` a `newline` argument). `stage_config.json`, `metrics.json` (still without a trailing newline), the five stance text/JSON pairs, the three evidence-CSV writers (`save_evaluation_episodes`, `write_stance_panel_evidence`, `write_recovery_evidence`) and the three hand-rolled sidecar writers (`gate_resolution.json`, `task_fingerprint.json`, `plant_identity.json`, all written inside run trees) go through them with their bytes unchanged, so a stranded temporary is now dot-named and discarded by the manifest. On a mount the final, best and robust-best pairs are saved to local scratch and published by `curriculum.publish_staged_pair`: handoff pairs zip last, the final pair sidecar last, each after removing the destination file it publishes last, so no reclaim leaves a truncated file or a mixed pair; off a mount they are written in place, as before. Not converted, the same pattern outside CU-3's list: `stage_summary.txt` and `training_summary.txt`, `collected_results.csv` and `curriculum_results.csv` (append mode), the stance diagnostics CSV, `ancestors.py`'s record copy, the widen tool's pairs and the repository's `plant_manifest.generated.json`. | Otherwise PR-13's evidence writer copies the non-atomic pattern. | about +60 (tests) | LOW | Any time; before PR-13 |
| CU-4 Test helpers | C19 | One `notebook_cells.py` for the notebook extractors (fewer after PR-A and PR-B), which the CI notebook validator uses too. Delete the duplicated library-only pins (decision 9) and the dead `_clean_repository_state`. `ancestors_helpers.py` replaces the test-to-test imports. One TinyEnv. Add a notebook import-resolution check and a canonical-JSON pin for the two notebooks that remain; CI only AST-parses notebooks (`python-ci.yml:93-124`). | CU-5 and CU-6 rewrite the same extractors and pins. This is PR-15's test slice. | about −200 | None | After PR-B |
| CU-5 Notebook text and dead parameters | C5 | Re-check the survey's text fixes against the follow-up's notebook: the deleted `trex/stance.toml` quote, "all four" species (there are six), the stale "review F3" note, and the branch numbering. Drop `save_path`/`save_dir`/`_show`/`fig1`/`fig2`, `species=None` and `run_dir=None`. Add `node_budget()`. Cut the markdown toward about 95 lines, most of this PR's reduction. Measured: the SB3 notebook has 33 cells and 1,557 source lines at `f850815` (207 of them markdown), and 1,599 at the follow-up. The survey projected about 29 cells and 1,290 source lines after C4, C5, C6 and C9, counted from 1,515; re-derive it from 1,599. | Moves toward the consolidation plan's §4 notebook targets. | about −110 | None (keep the pinned phrases) | After CU-4 |
| CU-6 Resume slice into the package | C6 (remainder) | The follow-up already moved the pair check. What remains: `newest_intact_periodic_pair` sharing `train_base`'s regex, with one test per skip reason; `train_stage(evaluate=False)`, which drops a 60-episode evaluation that JUDGE always redoes; and moving the archive preflight into `policy_loading`. | Resume becomes tested library code. | notebook −115, library and tests +200 | LOW | After session 6's resume and CU-4; amend D-D7 |
| CU-7 Dead code and import cost | C7 (minus JAX) + critic items | Delete 6 dead defs, `FIGURE_NAMES`, the stale "funnel through `summarize_stance_panel`" text, the test-only `recovery_evaluation.paired_success_differences`, and the dead gym entry points (`pyproject.toml:94-105`; gymnasium 1.3.0 loads no plugins). Reduce `environments/shared/__init__.py` to its docstring; the survey measured `import environments` falling from 1.9 s to 0.3 s and from 1,676 modules to 390. Script hygiene. One sha256 regex and one set of validators. Declare `imageio` (used at `compsognathus/scripts/view_model.py:117`). Delete what PR-A leaves of the KNOWN_ISSUES gym entry-point bullet (under Configs, docs & website) and fix `docs/ROADMAP.md:48-52`, which ticks "Register Gymnasium entry points" and names `MesozoicLabs/Velociraptor-v0` (the id is `Raptor-v0`). Correct the fingerprint docstring (`task_fingerprint.py:88` says "all four species constructors"; there are five) and, unless PR-B has already deleted it, the `foot_contact_*` test comment (`test_species_integration.py:454-456`) (§7). Optional: make `curriculum/__init__.py` lazy (a PEP 562 `__getattr__` over the same names), so the pure gate modules stop importing SB3 and torch; the survey measured `curriculum.gate_schema` falling from 1.73 s to 0.28 s that way. | Pure deletion. | about −330 | LOW | Any time |
| CU-8 One derivation, one reader | C9 | Generalise `stage_task_fingerprint` to all 6 sites, which retires the text pin at pins:487-506. Make `ignored_hyperparameter_edits` public; the notebook copy lacks its guard. One `stage_config.json` reader. Move the sidecar resolver and `_ensure_sb3` into `policy_loading`. Keep one `FINGERPRINT_BACKEND` constant (today at `freeze_recovery_gate.py:127` and `widen_checkpoint.py:153`), one constant for the 33 `"stable-baselines3"` literals, and one `REPOSITORY_ROOT`/`_SHARED_ROOT` (defined in both `result_bundle/constants.py` and `plant_contract/constants.py`). Keep the values unchanged: the backend string enters `task_sha256`, and repo-relative paths enter `behavior_identity`. | PR-9 gets a single derivation, and PR-10 gets `policy_loading`. | about −100 | LOW; acceptance is the committed `task_sha256` tests | After PR-A; before PR-9 |
| CU-9 Coverage of certification code | critic item | `*/scripts/*` and `environments/shared/harnesses/*` are coverage-omitted (`pyproject.toml:164-169`). That hides `freeze_recovery_gate.py` (892 lines), `widen_checkpoint.py` (1,412) and `backfill_gate_verdict.py` (469). Replace the blanket omits with explicit entries, and move `brace_controller` to break the cycle. Keep `harnesses/digest_snapshot.py` (added with this plan) in the explicit omit list. Close or rebase #527 first (decision 19); it edits `freeze_recovery_gate.py`. | Certification code should count toward `fail_under = 70`. | small | LOW (re-measure) | After PR-B |
| CU-10 `train_curriculum` body onto `train()`'s helper | C10 | Name `eval_env_seed`. Validate old against new under a frozen clock. Optional second PR: split `train_base.py` into trainer, curriculum runner and post-training panels. | Removes the second copy of the stage body. | about −100 | LOW-MED | Lowest priority |
| CU-11 Reward/info/termination golden | C11 | 21 species×stage captures, stored as digests plus a summary. Quantise the values, or run a same-machine A/B. Reuse the `reset_golden` helpers. | No digest covers reward or termination code, and PR-8/PR-9 need a "no number moved" check. | +150–200 | LOW | Before PR-8 |
| CU-12 Species env dedup | C12 (minus MJX scalars) | Reward-term helpers replace the 8 thin `_compute_*` wrappers. One contact query and one height/tilt termination prefix. Drop the foot-force overrides, pinning the base order `group[0]+sum(group[1:])`. A home-keyframe helper. Delete the test-only trex accessor. The cross-backend scalars need no work: PR-B deletes their MJX copies, and the 7–9 keys left in the frozen `mjx_config.py` registrations are not edited. Leave every token-hashed method alone (§5.6). | PR-9's constructors edit the same files. | about −300 | LOW (CU-11 proves it). A byte edit to a species env file moves the pilot `behavior_identity`; those bundles are evaluation-only (D-D9), none is on Drive, and PR-9 deletes that identity. Acceptance: CU-11's golden, plus a harness diff limited to the `behavior` identity lines of the species touched. | After CU-11; before PR-9 |
| CU-13 Stage-TOML `extends`, step 1 | C13 | D-D5 pulled forward: recovery extends stance for `[env]`, `[ppo]`, `[sac]` and `[stage]` in trex, compsognathus and compsognathus_robot. The survey critic confirmed that `{**stance, **recovery}` reproduces those tables with identical key order and that only `[curriculum]` differs. Never inherit `[curriculum]`. Keep a permanent 21-stage digest snapshot, which the digest-snapshot harness (§4.4) produces. Step 2 (compsognathus_robot ← compsognathus, after rerouting the direct TOML readers) waits for PR-12 and session 6. | Tests PR-11's mechanism against a known answer. | about −130 | LOW | Before PR-11; amend D-D5 |
| CU-14 Workflow structure | critic items + decision 10 | Collapse the 18 matrix jobs to 6 (`python-ci.yml:216-220`). Anchor the byte-identical `paths` lists (:6-29, :38-61). Install `.[test]` rather than `.[dev]` in the matrix. Drop the plant-contract job's duplicate pytest step (192 s), keeping its `--check`, baseline and wheel steps. Drop `test_phase_c_interface.py` from the SB3 list (145 s). Apply decision 10, and fix the "smallest plant" comment at `python-ci.yml:313-314` (the robot is the costliest plant for those tests). Update branch protection if job names change (decision 18). After the plant-contract pytest step goes, the frozen-core pin test runs only in the `test (shared, …)` matrix (§4.3). | `test-sb3` is the critical path after PR-B. | YAML only | LOW | After PR-B |
| CU-15 Drive summary reader | C16 (reduced) + the rest of C15 | No sweeps are written after PR-A. NB2 (current-layout `sweeps/<algo>_<ts>` folders are skipped) has no present impact: a read-only Drive listing on 2026-09-26 found three `sweeps/` folders (created 2026-03-25..28) holding ten legacy `stage<N>_<algo>_<ts>` folders, which the summary reads, and no current-layout folder. PR-A drops its KNOWN_ISSUES entry. Give the Drive summary's setup cell the SB3 bootstrap: `REPO_REF` with fetch-and-detach instead of a `--depth 1` clone of the default branch, the three-clause `IN_COLAB`, and a guarded Drive mount. Optional: move the 786-line parser into a tested `reporting/run_index.py`, with pandas imported lazily and declared. | Seven reader patches since August. The bootstrap records which code the summary ran. | +150–300, all tested (optional), plus the small bootstrap edit | LOW | After PR-A |
| CU-16 Docs correctness | C20 | Add a status appendix to [reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md](reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md) and move its open findings (CF6, OP3, SS3/4, TC8) into KNOWN_ISSUES; mark JX2, JX4, JX7, JX9, NB3 and CF1 "retired by D-D17". List all 6 env ids in the API overview. Fix the backfill tool path. Mark STAGE1_SPLIT_PLAN's status, re-label RECOMMENDATIONS.md as a dated 2026-03 snapshot (`docs/README.md:43` says "Active"), and mark WEBSITE_PLAN complete (:47), moving its logo-SVG item to KNOWN_ISSUES. Update ROADMAP, the README and the website milestones. Correct the living-doc sentences that say pilot or certificate data exists on Drive (none does); the CHANGELOG copies are history, but add a one-line correction to the `Images/` entry (`CHANGELOG.md:3326-3329`, :3388-3391 at the follow-up), which is wrong about where the removed GIFs survive, in the new `[Unreleased]`. Assets: the three `website/static/videos/raptor_stage*.mp4` videos and their three posters (1,974,435 + 19,787 B) are unreferenced and have no other copy. `sac_apex.gif`/`ppo_apex.gif` share blobs with `results/velociraptor/{sac,ppo}/stage3_strike.gif` (23,558,176 B), which only `results/README.md:12-17` lists, so delete both copies or neither. `results/velociraptor/ppo/stage1_balance.gif` and `raptor_balance_ppo.gif` share a blob, but both are referenced: no action. The orphan PWA icons (62,739 B) and `compsognathus/data/robot_camera_view.png` (32,616 B) can go. The heightfield and render-crash entries are already added with this plan (§3.5). | Living docs must stay true. About 2.1 MB is freed without the GIF decision. | about +150 | None | After PR-B |
| CU-17 Docs shrink | C21 | First move the only copies of unique text: PR-7's parity reason, PR-4's reproducibility note and the NEXT_STEPS §5 operational choices. PR-3b's list is superseded; say so. Then: landed consolidation-plan PR bodies become pointers (−500), NEXT_STEPS shrinks (−185), KNOWN_ISSUES entries over 40 lines shrink (−275), and the README Quick Start folds into the recipe pages (−90). Rewrite PR-15's text. Adopt the rule that PR landing status lives only in the consolidation status table and the CHANGELOG (a `docs/README.md` convention). | Shorter living docs. | about −1,050 (docs) | None | Last |

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
- **Live defects (§5.3):** five disappear with PR-A/PR-B and #558 fixed two; `render_mode='human'` survives (CU-2) and
  NB2 shrinks to historical folders (CU-15).

### 3.4 Prerequisites of consolidation PR-8..PR-15 (updated for the retirement)

| Planned | Needed for terrain? | Land first | Effect of the retirement |
|---|---|---|---|
| PR-8 selector, constants, normalisation | Yes | CU-11 (golden as acceptance) | Unchanged |
| PR-9 phase D hook, identity = fingerprint | Yes | CU-8, CU-12 | Smaller: no `MJXEnvConfig` edit (`mjx_env.py:274-279`), no MJX suite. It must still leave the frozen core untouched and pass `plant_contract --check` |
| PR-10 command-column warm start | Yes | CU-8, CU-1 (SB3-aware mypy) | Unchanged |
| PR-11 follow/terrain nodes | Yes | CU-13; PR-11's Breaks line must name `test_every_committed_stage_is_command_mode_none_in_phase_c` (`test_sb3_notebook_pins.py:2333`, :2559 at the follow-up); decisions 11, 13, 14 and 15 | New TOMLs carry no `[jax]` (after PR-B, `config.py:224-231` rejects it) and need no MJX mirror |
| PR-12 rest (delete the pilot) | No; it is cleanup that follows PR-11 by design | — | It updates the harness's behavior section when the 66 recipe TOMLs go, and conflicts with the retirement only in `python-ci.yml` |
| PR-13 terrain_command gate | Yes, to certify terrain | CU-3 (carried out 2026-09-26, D-D20); decisions 6 and 12 | Invariant 10's new case covers only the SB3 manager. Land PR-B first, because it deletes `test_gate_dispatch_fail_closed.py:161-351` |
| PR-15 | Mostly cleanup | CU-4, CU-17 | No JAX or sweep docs to fold. CHANGELOG Removed inherits the D-D17 entries |
| PR-3b (JAX/SB3 job) | No | — | Superseded by PR-B. The next CI lever is inside `test-sb3` (decision 10) |

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
| The best and robust-best handoff pairs are written straight to the mount (Training / RL, MEDIUM) | Decision 7 | CU-3 (deleted by it, 2026-09-26; D-D20) |
| Nothing on disk records the trunk a session resolved (Training / RL, LOW) | Decisions 4 and 6 | The PRs that take both decisions |
| `train --load` with the default `resume_same_stage` writes into a judged stage directory (Training / RL, MEDIUM) | Decision 5 | The library guard |
| `render_mode="human"` crashes on the first step (Training / RL, MEDIUM) | Defect 2 | CU-2 |
| The JAX command-line curriculum ignores `min_avg_forward_vel` (Training / RL, MEDIUM (JAX)) | Defect 3 | PR-B |
| MJX training never pays dibothrosuchus `snap_snout_proximity_weight` (Training / RL, MEDIUM (JAX)) | Defect 4 | PR-B |
| Every Ray Tune PPO trial raises `TypeError` on `ent_coef_end` (Sweeps / infrastructure, MEDIUM) | Defect 1 | PR-A |
| Stage summaries built from `evaluations.npz` assume a 0.01 s control step (Sweeps / infrastructure, LOW) | CU-2 | CU-2 (latent after PR-A) |
| The Drive summary skips current-layout sweep folders, gap-review NB2 (Notebooks, LOW) | Defect 6; CU-15 | PR-A |
| Pre-commit pins ruff 0.4.4 while CI installs the latest ruff (Testing / CI, LOW) | CU-1 | CU-1 (deleted by it, #561, 2026-09-26) |
| CI's mypy never sees SB3 or torch types (Testing / CI, LOW) | Decision 8 | CU-1 (deleted by it, #561, 2026-09-26) |

Corrected existing entries:
- The "Curriculum gates" divergence bullet: the JAX CLI checks reward and episode length, and ignoring velocity and
  success is a defect, not a divergence. PR-B deletes it with its section.
- The LOW re-entry entry now names the bundle-verification cell, which #558 renamed from "Cleanup". It stays.
- The stage-3 sweep-keys entry is replaced: 18 keys in 7 files (defect 9). PR-A deletes it.
- The `ray_orchestration.py` entry (1,006 lines; `export_best_trial` has a caller) and the `ray_tune_sweep.ipynb`
  bullet, which gains gap-review NB3. PR-A deletes both.
- The notebook-pins bullet: the notebooks now pin SB3 and JAX. PR-A drops its Ray clauses and PR-B its JAX clauses.
- The gym entry-point line: verified dead. CU-7 deletes it.

The `foot_contact_*` LOW (the JAX-only knobs entry under Training / RL) is not edited now; PR-B rewords it (§4.6, §7). The speed and map
mismatch of decision 13 has no entry of its own; the heightfield entry records velociraptor's map exits.

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
`plant_manifest.generated.json` and reset golden. It moves every certified run's plant identity, so it is not cleanup
(§7); decide it with the maintainer once the walkers it would strand are known.

### 4.2 What stays

| Kept | Location at `f850815` | Why | Frozen size |
|---|---|---|---|
| `build_mjx_observation` (verbatim, top level), `_SPECIES_CONFIGS`, `register_species_mjx` | `environments/shared/mjx_env.py:342,362-419` | The probe executes it (`policy_layer.py:245-260`), and it is token-hashed as `training_reset_and_step` (:387). Its call-time `from .obs_functions import …` (`mjx_env.py:385`) pins it to `environments/shared/`, because relative-import dots are hashed | 1,645 → 76 lines |
| `make_obs_fn` (verbatim, signature included) | `jax_setup.py:524-562` | Hashed as `cpu_evaluation` (`policy_layer.py:388`); never executed | 1,003 → 54 lines |
| `scale_action_around_nominal_jax`, `reset_mujoco_data_to_home`, `scale_action_jax` | `mjx_utils.py:22-83` | The first two are hashed (`policy_layer.py:382-385,400-404`). `policy_layer.py:34` names `scale_action_jax`. Only `check_jax` (:11-19) goes | 83 → about 72 lines |
| `obs_functions.py` (whole file) | :1-180 | `_array_mod`, `build_bipedal_obs` and `build_quadruped_obs` are hashed (`policy_layer.py:378-381`); the probe executes `SensorLayout` | 180, unchanged |
| Per-species `mjx_config.py`, reduced to the registration | trex 155, velociraptor 65, brachiosaurus 81, dibothrosuchus 78 lines | Read at `policy_layer.py:181-241`; the module name is a payload literal. Key sets differ: trex 9 (`action_mapping`, `action_filter_cutoff_hz`, `frame_skip`, `sensor_foot_indices`, `sensor_foot_aux_indices`, `sensor_gyro_start`, `sensor_accel_start`, `sensor_quat_start`, `body_ids`), brachiosaurus 8 (no cutoff), velociraptor and dibothrosuchus 7 (no cutoff, no `sensor_foot_aux_indices`) | 379 → 91 lines |
| `plant_contract/policy_layer.py`, `digests.py`, `manifest.py` | whole files | They are the hasher | zero-line diff |
| `action_filter.low_pass_alpha`, `apply_low_pass` | `action_filter.py:47-70` | Hashed for trex (`policy_layer.py:405-418`) | unchanged |
| `command_frame` constants | `command_frame.py` | Enter the policy payload (`policy_layer.py:15`) and `task_sha256` | unchanged |
| `foot_contact_weight`/`foot_contact_gate` params and their six `[env]` keys | `trex_env.py:141-142`, `dibothrosuchus_env.py:108-109`; `configs/dibothrosuchus/stage1_balance.toml:13-14`, `configs/trex/stance.toml:13-14`, `configs/trex/recovery.toml:57-58` | `task_sha256` hashes the constructor defaults overlaid with `[env]`. Stripping the dibothrosuchus keys moved its certified stance task from `083e2966` to `abfb339f` (measured) | unchanged |
| Byte-frozen files: `behavior_env.py`, the five species env modules, `direction_commands.py`, `terrain.py`, `terrain_sampling.py` | whole files | `behavior_identity` hashes their raw bytes, comments included | zero-line diff in PR-A and PR-B (their acceptance step 9). Any other PR that edits these files says so, and its harness diff shows only the `behavior` identity lines of the species it touches (CU-12, PR-C) |
| TOML digest inputs | `[stage]`, `[env]`, `[ppo]`, `[sac]`, `[curriculum]`; `configs/*/behaviors/*.toml`; `plant_versions.toml`, `plant_manifest.generated.json`, `recovery_calibration.json`; the `[[species]]` rows, with no `training_backends` added for the four dual species (`plant_contract/manifest.py:88-93` raises on a mismatch) | Only `[jax]` and `[jax.policy_kwargs]` may go. Three prototypes confirmed they enter no digest | — |

The frozen code totals 76 + 54 + 72 + 180 + 91 = **473 lines**.

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

### 4.4 The digest-snapshot harness (in the repository since 2026-09-26)

The harness is committed with this plan (2026-09-26) as `environments/shared/harnesses/digest_snapshot.py`, a guarded
rebuild (356 lines, lint-clean under the repository's ruff settings) of a 237-line prototype that never entered the
repository. It is listed in the module list of `environments/shared/harnesses/__init__.py` (:15-24), and coverage
already omits `harnesses/*` (`pyproject.toml:169`; keep it omitted when CU-9 lands). PR-A and PR-B use it for
acceptance step 1 (§4.5, §4.6). Decision 16 (commit a golden output and add a `--check` step) stays proposed: until it
is taken, the harness is run by hand. The frozen-core reference copies are not in the repository; PR-B rebuilds them
from `f850815` per §4.2.

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
the time.

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
upload as well, and the same day moved them into a PR of their own, PR-A2, after PR-A. PR-A2 deletes
`scripts/setup_vertex_ai.sh`, the `Dockerfile`, `.dockerignore` and
`website/docs/training/vertex-ai.md` (with its sidebar entry and the links into it), and the GCS upload path:
`curriculum --gcs-bucket` / `--gcs-project` with `config.upload_curriculum_artifacts`, the `gs://` branch of
`reporting/csv_output.py`, and the `[gcp]` extra. The counts above are PR-A's; PR-A2 derives its own list and
counts when it opens, with an end-to-end test of the command-line curriculum path it edits. The bullets below that
keep the route or `[gcp]` are superseded where marked, by PR-A2. `tb_sync.py` stays (§4.9) even though its `/gcs/` branch served the Vertex FUSE mount.

**Edit:**
- **Code.** Delete `visualization.py:789-903` and the hypertune `try/except` at `train_base.py:1515-1532`. Keep
  `_report_hpt_metrics` (:1453-1619) and its name: it writes `metrics.json`, which `stage_artifacts.py:106` reads, and
  `test_train_base.py:2404` monkeypatches it. Reword `stage_artifacts.py:3`, the call-site comment at
  `train_base.py:1419` and the `_report_hpt_metrics` docstring (:1468-1473): it writes `metrics.json` for
  `stage_artifacts` and the Drive summary.
- **CI (`python-ci.yml`).** :267 becomes `".[train,test,viz]"`, keeping CU-1's `stable-baselines3[extra]==2.9.0` pin
  (`test_ci_tool_pins.py` checks it). :269-306 shrink to `import stable_baselines3, torch`
  plus the CPU-wheel assertion (:284). :377 drops `test_sweep_ray_plant_contract.py`. Job names are unchanged.
- **`pyproject.toml`.** Remove `cloudml-hypertune` (:41), `[ray]` (:56-61), `[mjlab]` (:62-67) and the mjlab coverage
  omits (:170-173). Keep `[gcp]`, rewording its comment. `[all]` (:91) becomes `[train,jax,viz,gcp,dev]`, because
  `curriculum --gcs-bucket` needs `google-cloud-storage` (`config.py:881,952`). *Superseded by D-D17:* PR-A2
  removes `[gcp]` with the GCS upload code, and `[all]` then becomes `[train,jax,viz,dev]`.
- **Vertex single-job route** (if decision 1 keeps it; *superseded by D-D17*, which deletes the route instead). Drop `setup_vertex_ai.sh:183`. Delete `vertex-ai.md:570-679`
  and rewrite its Next Steps (:692-696). Add a `pip install google-cloud-aiplatform` line; the only one today, at
  :613, is deleted with that range. Reword :368.
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
  two archive tags reproduce them (the JAX tag for notes such as TRAINING_REVIEW_JAX_STAGE1). Living docs that must
  name a removed path use a backticked path plus the tag (precedent `NEXT_STEPS.md:35`), never a link. No relative
  link from `docs/investigations/`, `docs/reviews/` or `docs/hardware/` points into the removal set.

**Draft D-D17**, kept as proposed. *Recorded 2026-09-26:* the maintainer took D-D17 with the single-job route and GCS
upload retired too; the row is in [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2. The archive points (§2
row 2) are still open and are not part of D-D17; once they are settled, the outcome, with any tag SHAs, is appended to
that row rather than filled into this draft:

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
  removes the upload code with `[gcp]`, so a `--gcs-bucket` caller gets an argparse error instead.)
- Removing the Drive summary's sweep branch would make `discover_runs` walk into `sweeps/`.
- `/docs/training/sweeps` will return 404, because the site has no redirect plugin.

### 4.6 PR-B: retire the JAX/MJX runtime; keep the frozen core

**Delete (30 files, 15,698 lines):**

| Files | Count | Lines |
|---|---|---|
| `environments/shared/jax_{checkpoint,curriculum,eval,hooks,normalization,ppo,reward_termination,train_fn,trainer,trainer_types,training,training_utils,viz}.py` | 13 | 6,777 |
| `shared/tests/test_jax_*.py`, `test_mjx_*.py`, `trex/tests/test_trex_mjx_reward_parity.py` | 15 | 7,256 |
| `jax_training.ipynb` | 1 | 1,372 |
| `website/docs/training/jax.md` | 1 | 293 |

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
  `vertex-ai.md:64-67,302,308` as well.
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
  consolidation plan's PR-3b, PR-9 and §8 risks.
- **CHANGELOG.** Record both archive SHAs, the drift list (§5.3), the titles of the dropped KNOWN_ISSUES entries (the
  re-add checklist, including the Ray PPO `ent_coef_end`, JAX-threshold, snout-proximity and sweep-keys entries) and
  the seven anchor digests.

**Acceptance (PR-B)** (restated per critic item 3):
- Run PR-A's steps 1–8, with the snapshot diff both blocked and unblocked.
- Step 9 becomes: the only `plant_contract/` change is removing `validate_mjx_environment_plant` and its export; the
  TOML diff is `[jax]`/`[jax.policy_kwargs]` deletions only, with the `foot_contact_*` lines untouched; nothing else
  on the list changes.
- The pin test passes, and so do the plant-contract tests (a prototype measured 54).
- Collection shows 0 errors (the removal critic measured 3,812 tests collected after both PRs), and mypy checks 303
  files (331 after PR-A; 302 and 330 before the harness was added) with the same 21 errors, or 0 after CU-1 (CU-1's
  and CU-3's test files add three to each count).
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
  for frozen code.
- **Behavior identities are not pinned in CI.** Nothing in CI pins them; the harness closes that gap only when run
  (decision 16).

### 4.7 Ordering and shared lines

PR-A goes first. PR-B needs two things PR-A adds (critic item 4): D-D17, which the FROZEN docstrings and every
amendment cite, and the first tag. (Since 2026-09-26 D-D17 is recorded already, so PR-B can cite it whichever PR lands
first; what PR-B still takes from PR-A is the archive point, if decision 2, still open, calls for tags.) The harness, PR-B's acceptance step 1, is already in the repository (§4.4). The
code itself is independent in both directions (checked with grep). Whichever PR lands second rebases over the lines
both edit: `pyproject.toml:91`, `python-ci.yml`, `sidebars.ts:19`, `species_manifest.toml:46-59`,
`test_species_catalog.py:652-671`, `test_species_names.py:105-109`, `README.md:500-501,513-514`,
`species.generated.json`, KNOWN_ISSUES :249-257, ROADMAP :453-516 (which contains PR-A's :496-515),
`installation.md:76-77`, `recipes.md:618,626`, `hyperparameters.md`, `docs/SPECIES_NAMING.md:32-34`, KNOWN_ISSUES
:1119-1122, NEXT_STEPS, the CHANGELOG, both plans and PLANT_CONTRACT.md.

### 4.8 Maintainer actions outside the repository

1. **Tags (decision 2).** Before PR-A merges, create and push an annotated tag named
   `archive/secondary-backends-2026-09` on PR-A's first parent (`git tag -a`, then `git push origin <tag>`). Tag
   PR-B's first parent the same way before PR-B merges.
2. **Branch protection (decision 18).** On 2026-09-25 the public API showed `main` with `protected: false`, no
   required checks and no rulesets, although the comments at `python-ci.yml:210-212,316` assume required checks exist. Check Settings →
   Branches/Rules. If `test-jax-cpu` is a required check, remove that requirement when PR-B merges; otherwise every PR
   will wait for a check that never reports.
3. **Before merging, check outside the repository:** put the `full-ci` label on both PRs; confirm that no Vertex
   tuning job, GCS sweep state (`gs://<bucket>/sweeps/…`) or Ray experiment is still live; leave Drive untouched: the
   27 JAX run folders (2026-03-30..04-03, none a certified parent) and the March 2026 sweep folders stay.

### 4.9 What not to remove

- **Reader back-compat,** so old JAX records keep validating: `result_schema.py:117,534,612,932,1032,1429` and
  `result_bundle/naming.py:15-39`; the `jax-mjx` branches in `reporting/summaries.py:23,220`, `csv_output.py:252-253`
  and `bundles.py:355,409-423`; `ancestors.py:1031-1033` and `config.py:494`; Drive summary cells 8, 9, 11 and 12.
- **The provenance keys** `_DEPENDENCY_PACKAGES` (`result_bundle/constants.py:54-64`) and their fixture
  (`conftest.py:31-45`). Changing the key set records `environment_drift` on the next resume of an incomplete run,
  such as session 6's.
- **`gate_schema` backend-override validation** (:274-295, 466, 538, 544-600). It still validates recorded blocks, and
  a shallow clone could not rule out a historical `jax` sub-table.
- **`command_frame.py`,** including its MJX refusal. PR-9 rewrites that file.
- **All of SAC.**
- **The frozen core,** the dual declarations, the `foot_contact_*` keys and params, and the byte-frozen files.
- **What the SB3 notebook and the Drive summary use:** `build_stage_results_from_eval_data`,
  `backfill_gate_verdict.py`, the `metrics.json` writer and its alias keys (`train_base.py:1839-1850`), `tb_sync.py`,
  `cli._apply_overrides` and the `quality_score` column.
- **The generated JAX/MJX metric lines** in the README and on the website. Changing those is a separate catalog PR
  (decision 17).
- **`obs_functions._array_mod`,** which is hashed. The NumPy-only simplification of the other two `_array_mod` copies
  is digest-safe but optional, so defer it.

### 4.10 How to add a backend back

Adding a backend back is a new decision, taken against D-B13 and invariant 9. Start from the archive tags.

**JAX:**
1. Restore the generic stack almost as-is (3,969 lines), from `jax_ppo` through `jax_viz`.
2. Rebuild the per-species parts (5,918 lines) against the final SB3 rewards, with a single reward composition. Today
   the reward is composed three times: `mjx_env.py:905-1366`, `jax_reward_termination.py:72-384` and `:387-641`.
3. Add a per-component parity test from the tag's template, and fix the drift list (§5.3).
4. Build the frozen core forward without changing its tokens. A species newly gaining JAX needs a
   `policy_interface_revision` bump.
5. Restore the CI job.
6. Cover what the old backend lacked: live command modes (`command_frame.py:62-66`), terrain, the
   `recovery_quality/v1`/`task_success/v1` evaluators, and compsognathus.

**Ray:** wrap `train_base.train()`, as the Vertex trial did, instead of restoring the copy. Regenerate the sweep JSONs
from the constructor signatures, and run one real trial in CI; the removed smoke used `train_fn=lambda config: None`.

**Vertex HPT:** restore the Vertex half of `scripts/sweep/`, `cloudml-hypertune` and the 18-line report.

**mjlab:** restore its 3 files and the Phase C obs-dim test. It never ran; every factory raised `NotImplementedError`.

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
   1.57 m/s) and on compsognathus_robot once its locomotion is judged.

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
  on current physics (session 5).
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
  review, at `b358a46`).
- **Cost (estimates):** a trex 3M-step pilot on one env takes about 2.5 h flat or 4.3 h on terrain; these rates are
  reset-bound, measured from a falling fixture on 4 cores; G1's deliverable pair takes 4–14 h per species; the
  recommended chain takes 9–28 h, as an upper bound.

### 5.3 The survey's nine live defects

| # | Defect | Status |
|---|---|---|
| 1 | Every Ray PPO trial crashes on `ent_coef_end`: `ray_tune.py:714-759` lacks the pops at `train_base.py:354-355`. Present since at least 2026-08-09 | Disappears with PR-A |
| 2 | `render_mode='human'` crashes on its first step (`base_env.py:1558`; `mujoco.viewer` is never imported) | **Live** (re-checked 2026-09-26); CU-2 |
| 3 | The JAX in-training gate ignores `min_avg_forward_vel` (`jax_curriculum.py:455-486` vs `curriculum/manager.py:340-348`); it would ignore `min_success_rate` too, but the CLI never gates the final stage, where that key is set | Disappears with PR-B |
| 4 | The MJX step never pays dibothrosuchus `snap_snout_proximity_weight` (`mjx_env.py:1303-1306`), although the CPU eval that gates it does | Disappears with PR-B |
| 5 | Ray post-sweep metrics for compsognathus are 2× off: the notebook's `LocomotionMetrics()` defaults dt to 0.01, but compsognathus runs at 0.02 | Disappears with PR-A. The related `sim_dt` default (`stage_artifacts.py:151`) survives; CU-2 |
| 6 | The Drive summary drops every current-layout sweep (NB2) | The reader stays, and no new sweeps are written. A read-only Drive listing on 2026-09-26 found no current-layout sweep folder, so there is no present impact; its KNOWN_ISSUES entry goes with PR-A (CU-15) |
| 7 | RESUME can retrain over a node that already has a verdict | Fixed by #558 (D-D16), hardened by #559 |
| 8 | A `QUICK_TEST` stance can certify and become a `TRUNK_FROM = "auto"` parent | Fixed by #558: quick tests live under `<algo>_quick_test/` |
| 9 | 7 of the 12 sweep configs crash every stage-3 trial: they sample **18** env keys that no constructor accepts, e.g. `env_prey_distance_min`. Re-counted 2026-09-26; the survey critic's "22" was a miscount | Disappears with PR-A, which also deletes the replaced KNOWN_ISSUES entry (the stage-3 sweep-keys entry under Sweeps / infrastructure; §3.5) |

Defects 1, 3, 4 and 9, together with the never-executed notebook copies behind 5, make up the drift list PR-B records.

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
| mypy | The lint step takes about 13 s and reports no issues, because SB3 is absent. With SB3 2.9.0 and torch 2.14.0+cpu, re-measured 2026-09-26 at `f850815`: "Found 21 errors in 6 files (checked 357 source files)", 46–57 s cold in two local runs (mypy 2.3.1, JAX installed, ray and wandb absent; decision 8). Per file: `curriculum/advancement.py` 7, `scripts/widen_checkpoint.py` 5, `diagnostics.py` 4, `curriculum/schedules.py` 2, `tests/test_widen_checkpoint.py` 2, `command_frame.py` 1. All are type-only: `BaseAlgorithm` lacks `ent_coef`/`clip_range`/`log_ent_coef`; read-only callback properties are set in the SB3-absent fallback; state dicts are typed as `Tensor`; some Optional values are unchecked. The hand-kept count has drifted from 13 (`5f7318d`) to 21, and one PR saw 22. At `be63a58` with the harness added (§4.4): "Found 21 errors in 6 files (checked 358 source files)", at the same locations; the harness adds none. Locations at `f850815` (unchanged at the follow-up, which also reports 21): `command_frame.py:156`; `diagnostics.py:222` (×2) and :223 (×2), inside the SB3-absent fallback at :218-223; `curriculum/schedules.py:180,210`; `curriculum/advancement.py:519,537,539,555,559,560,617`; `scripts/widen_checkpoint.py:762,804,807,817` (×2 at :817); `tests/test_widen_checkpoint.py:254,330`. Re-measured 2026-09-26 at `8e03483` (CU-1): 21 errors in 6 files locally, as above; 23 errors in 8 files in `test-sb3`'s environment (Python 3.12, numpy 2.5.3, torch 2.13.0+cpu, ray 2.58.0, 66 s cold), the extra two at `harnesses/freeze_recovery_gate.py:469` and `scripts/sweep/ray_orchestration.py:247`; a lint-job mirror (Python 3.11, ruff 0.16.9, mypy 2.3.1, no SB3) reports none. After CU-1, all three report no issues in 359 files | local runs |
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
  leaves half of the new pair, never a new file beside the old pair's other half.
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
  the fingerprint docstring and, unless PR-B deletes it first, the test comment go in CU-7.
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
  PR-A2 together with the upload code, never on its own.)
- **The full D-D7 package move now.** It is line-neutral XL work that rewrites 1.3–1.9k lines of pins while walker
  sessions 4–6 are pending. Decide it after PR-13.
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
