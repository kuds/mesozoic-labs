# Next steps and program state (2026-09-24)

**Status**: living reference — updated 2026-09-24; `main` = `03d3a54` (merged 2026-09-24 02:39 UTC).

Read this first when starting a new session on the behavior-recipes program: what
has landed, what is certified on Drive, which training sessions to run next, where
the consolidation stands, and which decisions bind. Repository facts were verified at
`22c1fc8` and updated through #549 (`25132fc`); Drive facts date from the 2026-09-17
survey ([investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md))
plus the section 3 session results recorded since, and the maintainer's decisions
from 2026-09-17, 2026-09-20 and 2026-09-23.
Companions: [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) (design of
record), [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) (per-PR
sequence, file lists, rationale), [KNOWN_ISSUES.md](KNOWN_ISSUES.md). Update this
file in place when the state changes; it is not a dated investigation.

---

## 1. Where things stand

### Landed on `main` (all merged since 2026-09-12)

| PRs | Merged | What landed |
|---|---|---|
| #528–#531 | 2026-09-12 | Recipes Phase A: manifest v2, edge-keyed orchestration, per-deliverable publication, `gate_verdict.json`, `ancestors/` records, the notebook chain loop |
| #532–#536 | 2026-09-13 | Phase B: gate-configuration digest + reuse rule 7 + `backfill_gate_verdict --gate`; the `task_success/v1` hunting gate; seed replication as provenance; decisions D-B1–D-B17 (#536) |
| #537–#539 | 2026-09-13/14 | Phase C: the 3-dim command segment on all six species, `command_frame.py`, `widen_checkpoint.py`, the notebook `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP` knobs; decisions D-C1–D-C17 |
| #540, #541 | 2026-09-15 | T. rex direction-following and randomized-terrain pilots; all-species terrain behaviors and the certified library — a **separate pipeline** beside the recipes machinery (next paragraph) |
| #542 | 2026-09-16 | `PUBLISH_CERTIFIED` notebook default flipped to `False` (consolidation PR-1): library publication required a training-origin stamp a widened root handoff lacks, so a `WIDEN_FROM` session would have disconnected the Colab runtime right after the stance passed its gate |
| #543 | 2026-09-16 | Automatic trunk selection, decision D-A25: `environments/shared/ancestors.select_trunk`, notebook `TRUNK_FROM = "auto"` default, CLI `curriculum --trunk-from auto`. Canonical chains no longer consult the certified library; widen sessions select no trunk |
| #544 | 2026-09-19 | Consolidation PR-2: this file, the consolidation plan, the Drive survey note, decisions D-D1..D-D12 and G1..G4 in the plan's §6.2, the docs index and CHANGELOG |
| #545 | 2026-09-19 | The version-safe SB3 archive loader (`policy_loading.load_sb3_model`; picklable `LinearSchedule` / `CosineSchedule`; the widen tool re-states parent schedules) and the notebook's archive-load preflight cell before the widen cell, after two widen sessions died on the Python 3.13 image (KNOWN_ISSUES, "SB3 archives are bound to the interpreter that saved them") |
| #546 | 2026-09-20 | Consolidation PR-3: the bounded `test-sb3` job (one real-PPO smoke per body of work on pull requests and pushes; the six-species and four-notebook-parameter sets nightly at 05:17 UTC, on `workflow_dispatch` and under the `full-ci` label, which now starts a run when added); the widened-root bundle-write fix (a null `best_eval_reward` for a stage whose deliverable record names `widened_from_run_id`); decisions D-D11..D-D14 recorded. Measured on its CI: lean SB3 job 48:02, labelled full job 45:21, coverage 90 percent |
| #547 | 2026-09-20 | Consolidation PR-4: `certified_canonical.py`, its test and `test_sb3_notebook_certified.py` deleted; the notebook loses the three library knobs, the stamp, copy and publish blocks (2,563 → 2,477 lines) and never publishes; a reused trunk ancestor is loaded from the run that certified it (A10); `certified_comparison.py` deferred to PR-5; `train_behaviors --auto-source` without `--resume` refuses |
| #548 | 2026-09-20 | Consolidation PR-5: `certified_library.py`, `certified_comparison.py`, their tests, `test_behavior_publication.py` and `docs/CERTIFIED_MODELS.md` deleted; `train_behaviors` takes an explicit `--checkpoint` / `--vecnormalize` pair in every mode; the behavior certificate writer gone until PR-13 registers the gate kind; the notebook loses `SOURCE_SELECTION` (2,477 → 2,463 lines) and refuses blank source paths in the configuration cell; about −1,880 net code and configuration lines. Measured on its CI: SB3 job 46:12, JAX job 53:46, coverage 90 percent |
| #549 | 2026-09-20 | Consolidation PR-6: `configs/trex/behavior_pilots/` (8 `[pilot]` TOMLs), its package-data line and the trex shim deleted; `read_recipe` reads `[behavior]` only and refuses a recipe without one; the six pilot aliases and the dead pilot-run schema reader gone; `mesozoic.trex-command-terrain/v1` stays accepted until PR-7 deletes its emitter; −256 net code and configuration lines. Measured on its CI: SB3 job 34:49, JAX job 52:42, coverage 90 percent |
| #551 | 2026-09-23 | The collapse-backstop fix outside the consolidation sequence: `collapse_peak_warmup_timesteps` on dibothrosuchus and brachiosaurus stages 1–2 (1.0M on stance; on locomotion the D-B5 bound `warmup_timesteps + ramp_timesteps`, 3.3M and 4.0M), with replay tests on session 4's evaluation series and no digest moved; session 4 recorded; KNOWN_ISSUES gains three entries (a complete run cannot take a new node in place, an invisible early stop, the gait-symmetry statue payment); a truth pass over the living docs; the consolidation pause lifted. Measured on its CI: SB3 job 35:37, JAX job 53:34, coverage 90 percent |
| #552 | 2026-09-24 | Consolidation PR-12, the notebook-only slice (D-D13): the notebook's `COMMAND_TERRAIN_BEHAVIOR` switch, the ten `BEHAVIOR_*` knobs, the eleven direction/terrain dropdown values, six behavior cells and 15 guard sites gone (2,463 → 2,329 lines), `behavior_notebook.py` and its test deleted, `train_behaviors.py` the pilots' command-line path until PR-11; −996 code, test and CI lines. It carried the curves-cell fix (a completed "Run all" reaches the auto-disconnect again; 2,330 lines) and recorded the Drive cleanup of the three session bundles. Measured on its CI: SB3 job 47:56, JAX job 31:40, coverage 90 percent |

The notebook at `22c1fc8` ([notebooks/sb3_training.ipynb](../notebooks/sb3_training.ipynb))
has 40 cells (22 code), 2,526 lines; 19 code cells reference the
`COMMAND_TERRAIN_BEHAVIOR` mode switch (the 2026-09-19 loader change adds one
guarded code cell, the SB3 archive-load preflight before the widen cell:
41 cells, 23 code, 2,563 lines; consolidation PR-4 removes the library hooks,
about 90 lines: 41 cells, 23 code, 2,477 lines, the chain loop at index 23;
PR-5 removes `SOURCE_SELECTION` and its prose: 41 cells, 23 code, 2,463 lines;
the notebook-only PR-12 slice (#552, 2026-09-24) removed the mode switch, its
six cells and the 15 guard sites, the plan's 14 plus the guarded preflight:
35 cells, 19 code, 2,329 lines, the chain loop at index 20; 2,330 with the
curves-cell fix that rode along, CHANGELOG "Fixed").
Configuration-cell defaults:
`BEHAVIOR = "hunt"` (dropdown: `stand`, `walk`, `hunt`, stage ids by free input;
the eleven direction/terrain values leave with the PR-12 slice),
`TRUNK_FROM = "auto"`, `RETRAIN_FROM = ""`, `RUN_LABEL = ""`, `RUN_ID = ""`
(a configuration-cell knob since consolidation PR-14a; it was hard-coded in the
storage cell), `SEED = 42`; the library knobs (`CERTIFIED_LIBRARY_ROOT`,
`PUBLISH_CERTIFIED`, `CERTIFIED_COMPARISON_EPISODES`) left with PR-4,
`SOURCE_SELECTION` with PR-5, `COMMAND_TERRAIN_BEHAVIOR` with the ten
`BEHAVIOR_*` knobs with the PR-12 slice, and `WIDEN_FROM` /
`WIDEN_MAX_REVISION_GAP` with the widen cell in PR-14a (D-D14; 34 cells, 18
code).

### The pilot pipeline (#540/#541) — exists, evaluation-only

The pilots delivered genuinely new content (a direction controller, a tracking
reward, a heightfield terrain generator, a replay recorder with terrain maps) but
as a second copy of every canonical concept: two behavior env classes that bypass
the reserved `BaseDinoEnv._draw_episode_command` hook and write `self._command`
directly, a second checkpoint preparer, a second PPO trainer
(`environments/shared/train_behaviors.py`, one CPU env, its own recipe dialect),
66 behavior TOMLs under `configs/<species>/behaviors/` (11 templates x 6 species)
(the 8 trex `[pilot]` twins under `configs/trex/behavior_pilots/` and the
`[pilot]` dialect left with PR-6), a second gate outside `GATE_KINDS`
(`configs/behavior_certification.toml`, judged by `judge_behavior_panel`; its
`certification/certificate.json` writer left with PR-5, so it is reached only
from tests until PR-13; never a `gate_verdict.json`), a third identity keyed on
source-file hashes (any edit to `behavior_env.py` strands exact resume), and the
notebook mode switch (removed by the notebook-only PR-12 slice, #552) —
about 7,000 lines of modules at 22c1fc8, tests excluded (the certified library
left with PR-4/PR-5). Notebook pilot runs wrote to
`logs/<species>/ppo/behaviors/<behavior>/<run-id>/` (on Drive when it was mounted,
otherwise in the checkout's `logs/`); from the slice on a
pilot runs from the command line only (`python -m environments.shared.train_behaviors`,
D-D13) and, the certified library having left with PR-5, takes an explicit
`--checkpoint` / `--vecnormalize` pair in every load mode and for evaluation.
Guide: [TRAIN_DIRECTION_AND_TERRAIN.md](TRAIN_DIRECTION_AND_TERRAIN.md). Under D-D9 every pilot bundle on
Drive is evaluation-only; none is a training parent.

### The consolidation (PR-3 .. PR-15): released 2026-09-20, notebook-first

The 2026-09-17 review produced a fifteen-PR sequence folding the pilot pipeline
back into the recipes machinery ([section 4](#4-consolidation-the-remaining-prs)).
PR-1 (#542), the auto-trunk PR (#543) and PR-2 (#544) landed. On 2026-09-20 the
maintainer released the hold with a **notebook-first order** (decision D-D13):
PR-3, then PR-4 and PR-5, then PR-6, then a notebook-only slice of PR-12 (the
mode switch, the `BEHAVIOR_*` knobs, the direction/terrain cells and guard
sites and `behavior_notebook.py` go; `train_behaviors.py` stays CLI-only until
the rest of PR-12, after PR-11), then PR-14 (split on 2026-09-24 into PR-14a,
PR-14b and PR-14c, decision D-D15), then PR-7 .. PR-11, the rest of PR-12, PR-13 and PR-15.
D-D1..D-D15 are taken and recorded (D-D11/D-D12 confirmed and D-D13/D-D14 taken
on 2026-09-20, D-D15 on 2026-09-24; [section 5](#5-decisions-taken-2026-09-17)). PR-3 .. PR-6 landed
the same day (#546–#549); the maintainer then paused the sequence while the
training sessions ran (G3) and lifted the pause on 2026-09-23: the notebook-only
PR-12 slice landed as #552 on 2026-09-24, and PR-14a is in review on the session branch
([section 4](#4-consolidation-the-remaining-prs)).

### The final goal and the goal decisions

The target behavior is **every species follows a direction on difficult
terrain**. The maintainer fixed its shape on 2026-09-17:

- **G1** chain shape `walk -> follow_direction` (full command set on flat ground)
  `-> follow_direction_difficult_terrain` (the target deliverable); a commands-free
  `difficult_terrain` node stays as an optional diagnostic sibling; recipe label
  `follow` resolves to the deepest deliverable; `follow_direction_speed` is folded
  into `follow_direction`; three new stage files per species; the final node is
  SB3-only (MJX fails closed on live commands and has no terrain).
- **G2** command set = heading, speed (half to full cruise), stops and restarts,
  switching every few seconds (the pilots' combined recipe).
- **G3** walker sessions start now on the current notebook, trex first (its r13
  chain `20260914_123816` is selected automatically), the other five species one
  at a time, in parallel with the consolidation PRs.
- **G4** the first `terrain_command/v1` gate adopts the pilots' certificate
  thresholds (20 episodes per terrain family, 20 s minimum horizon, survival LCB
  0.80, success LCB 0.60, tracking and settle fractions 0.60), to be tightened
  after the first certified species.

Every follow/terrain node needs a certified locomotion ancestor at the current
policy interface (r13 for trex; r2 / r10 / r8 / r7 / r2 for compsognathus /
velociraptor / brachiosaurus / dibothrosuchus / compsognathus_robot), so the
walker sessions of [section 3](#3-recommended-training-sessions) come before any
of it.

---

## 2. Certified checkpoints on Drive

Drive layout `mesozoic-labs/logs/<species>/<algo>/<run>/`, surveyed 2026-09-17 and
updated with the section 3 session runs since.
Plant identities at `22c1fc8` (revisions from `configs/plant_versions.toml`,
dimensions from the generated plant manifest; policy interface / physics /
obs): trex r13 / 7 / 64, compsognathus r2 / 1 / 56, velociraptor r10 / 2 / 70,
brachiosaurus r8 / 4 / 86, dibothrosuchus r7 / 1 / 80, compsognathus_robot
r2 / 1 / 46. Reuse is decided per node from
`gate_verdict.json`, `task_fingerprint.json`, `plant_identity.json` and
`stage_config.json` (rules 1–7 in `environments/shared/ancestors.py`), never from
run-level summaries.

| Species | Run id | Seed | Interface | Stance | Locomotion | What it needs next |
|---|---|---|---|---|---|---|
| trex | `20260914_123816` | 42 | r13 (commit `35dd44c`) | **PASS** — `01_stance`, 11,001,856 steps, 13h13m, final eval 3460.6 ± 19.2, `gate_verdict.json` judged 2026-09-15 01:57 UTC, `gate_sha256 ff2494ba…`, handoff `robust_best_model.zip` | **PASS** — `03_locomotion`, 8,011,776 steps, 8h46m, 1.07 m/s, mean length 1000, reward 1940.8, verdict 2026-09-15 11:39 UTC, `gate_sha256 02602cb0…`, `task_sha256 31383192…` | Nothing for the two-seed bar: the second stance seed is `20260920_010912` (seed 44, certified 2026-09-20), whose bundle counts this run (replication 2 of 2). Its own run-level records stay stance-only (bundle `complete` under a stance target, replication 1): a complete bundle is immutable and not rebuilt in place ([RESULT_BUNDLES.md](RESULT_BUNDLES.md)), and reuse reads the per-node files, so nothing depends on them. A recovery node on this run's stance would be a `BEHAVIOR="stand"` session under `TRUNK_FROM = "20260914_123816"` (`"auto"` now picks the newer seed-44 run for a stand or walk session, whose chain consults stance alone; optional: the seed-44 run already certified recovery at r13); follow nodes once PR-11 exists |
| trex | `20260915_160239` | 43 | r13 (`35dd44c`) | **FAIL** — `01_stance` only, 11M steps (13h17m), final eval 3209.7 ± 284.3, best 3335.1; the 40-episode panel failed unsupported duty (mean 0.0323, UCB 0.0350, rail 0.02), reward and full-horizon passed; provenance `certified false`, `provisional true` | none | Nothing; a measured deficit, kept as history |
| trex | `20260920_010912` | 44 | r13 (commit `ac409f8`), widened from `20260815_205206` (r11, gap 2) by `widen_checkpoint/v1` | **PASS** — `01_stance`, judged 2026-09-20 01:15 UTC by `generate_stage_artifacts` on the widened handoff `robust_best_model.zip` (`checkpoint_sha256 7f4284ad…`, inherited 10,000,000 steps, no training here): panel reward 3408.3 ± 88.5, full-horizon 1.0000, duty 0.0069, UCB 0.0117, 40 episodes seeds 3042–3081, identical to the r11 certificate; final eval 3418.22 ± 87.88; `gate_sha256 ff2494ba…`, `task_sha256 82528a2e…` (= the seed-42 run's stance digest) | none; **recovery PASS** instead — `02_recovery`, 3,006,464 steps, 3h30m (2026-09-20 21:42 → 2026-09-21 01:12 UTC, the in-place continuation), 28/40 panel successes (Clopper-Pearson one-sided LCB 0.56 ≥ 0.30), paired success delta against the 0/40 statue null 0.70 (Student-t one-sided LCB 0.58 ≥ 0.20), 140/155 pushes recovered, 34/40 full horizon, panel reward 2925.4 ± 404.8; training final eval 2911.16 ± 556.35, best eval 3031.38 ± 159.07 at 2.9M; verdict 2026-09-21 01:16 UTC, `gate_sha256 a27ce071…`, `task_sha256 2c6f4a47…`, checkpoint `a8e41b98…` | A locomotion node, optionally (session 7: `BEHAVIOR="walk"`, `SEED=44`, `TRUNK_FROM="20260920_010912"`, a fresh run, section 3; not in place, because this run's bundle is `complete` and a complete bundle is immutable: since consolidation PR-14a the notebook refuses such a session before training); otherwise nothing: bundle `complete` (written after the recovery verdict), stance deliverable `certified true, provisional false`, `replication count 2` (`20260920_010912` seed 44, `20260914_123816` seed 42), recovery certified at replication 1 |
| trex | `20260815_205206` | 44 | r11 (legacy `stage1/` + `stage2/`) | PASS on `stance_gate_report.txt`: reward 3408.3 ± 88.5, full-horizon 1.0000, duty 0.0069, UCB 0.0117, 40 episodes seeds 3042–3081; **no `gate_verdict.json`**; `stage1/` holds `stage_config.json` (run block) and `models/` | not certified | Nothing: widened to r13 and re-paneled as `20260920_010912` (session 1, 2026-09-20) |
| trex | `20260810_145546` | 42 | r11 (legacy `stage1/2/3`) | PASS on the 2026-08 records; no `gate_verdict.json` in `stage1` | not certified | Nothing: seed 42 is already certified at r13 by `20260914_123816`; the template note's Session 1 (widen this run) is superseded |
| compsognathus | `20260909_162812` | 42 | r1 (obs 53, commit `9557e97`) | PASS on `stance_gate_report.txt`: reward 2801.6 ± 51.2, full-horizon 1.0000, duty 0.0131, UCB 0.0141, 40 episodes seeds 3042–3081; **no `gate_verdict.json`**; run block seed 42, n_envs 4, 11,000,000 steps; `physics_sha256 08a5fbf7…` unchanged at r2 | none | Nothing: widened to r2 and re-paneled as `20260921_203149` (session 2, 2026-09-21); stays on the log tree as history |
| compsognathus | `20260921_203149` | 42 | r2 (obs 56, commit `25132fc`), widened from `20260909_162812` (r1, gap 1) by `widen_checkpoint/v1` | **PASS** — `01_stance`, judged 2026-09-21 20:38 UTC by `generate_stage_artifacts` on the widened handoff (`checkpoint_sha256 1d46747f…`, inherited `num_timesteps` 10,850,000, no training here): panel reward 2801.6 ± 51.2, full-horizon 1.0000, duty 0.0131, UCB 0.0141, 40 episodes seeds 3042–3081, identical to the r1 report; `gate_sha256 62930e3f…`, `task_sha256 19837f77…` | **PASS** — `03_locomotion`, 3,002,368 steps, 3h49m, 0.34 m/s, mean length 1000, final eval 3308.6 ± 13.0, best eval 3337.16 ± 8.11 at 2.6M, verdict 2026-09-22 00:31 UTC, `gate_sha256 33e185d3…`, `task_sha256 63195040…` | Nothing: bundle `complete` (2026-09-22 00:31 UTC); the stance deliverable records `widened_from_run_id 20260909_162812` and a null `best_eval_reward` (the rule of #546); this is the certified compsognathus walker `TRUNK_FROM = "auto"` selects |
| velociraptor | `20260723_005740` (July) | 42 | r3 (obs 67); physics r2 = current | stage1 balance 6M, 3h41m, final eval 1767, 1000-step episodes; run-level `publication_gate_passed` only, no per-node verdict; sidecar predates identity stamping | stage2 8M, 4h50m, 3.29 m/s; stage3 strike 12M | **Not a widen candidate**: seven revisions behind r10 and the crossed revisions include the reset-settling change (`plant_versions.toml` note 6). Fresh chain ran as `20260922_125248` (session 3, 2026-09-22) |
| velociraptor | `20260922_125248` | 42 | r10 (commit `25132fc`); physics r2 | **PASS** — `01_stance`, 6,004,736 steps, 4h46m; certified checkpoint (30-episode selection eval) reward 1740.03 ± 301.5, mean length 970.3 ± 159.9 against the `reward_and_length/v1` rails 1050 / 950; training final eval 1666.19 ± 424.23, length 943.2; best eval 1797.43 at 4.35M; verdict 2026-09-22 17:42 UTC, `task_sha256 07d6a5af…`, checkpoint `732b5a71…` | **PASS** — `02_locomotion`, 8,011,776 steps, 6h30m; final eval 2683.64 ± 8.56, length 1000, 3.30 m/s; certified checkpoint 2591.88 ± 493.74, length 968.6, 3.17 m/s; best eval 2687.56 at 6.9M; verdict 2026-09-23 00:15 UTC, `task_sha256 6178a6ef…`, checkpoint `7c28eda2…` | Nothing for the chain: bundle `complete`, both deliverables certified (replication 1 each). The stance margin is thin: the certified checkpoint clears the reward rail by 690 and the length rail by 20 steps, and the training final eval's mean length (943.2) is below the 950 rail. A second stance seed (`SEED = 43`) is the cheap check if velociraptor stance is ever declared at `certification_seeds = 2` |
| brachiosaurus | `20260717_162659` (July) | 42 | r1; physics r1 (current r4) | stage1 balance 6M, 4h24m, 1739.8 | stage2 16M, 9h48m, 1.42 m/s; stage3 food_reach 12M | Physics digest differs, so widen is refused by construction. Fresh chain (session 5) |
| dibothrosuchus | `20260923_020654` | 42 | r7 (commit `25132fc`); physics r1 | **PASS, statue-level** — `01_stance`, 1,450,000 of 6,000,000 steps (1h00m30s), stopped by the collapse backstop; the judged `robust_best_model` is the 50k evaluation's: 2597.49 ± 1.95, length 1000, no forward motion, against the `reward_and_length/v1` rails 1560 / 950, while the run's zero-action statue scores 2598.29 ± 0.86 (40/40 full horizon; `zero_action_baseline.json` reads "FAILS — a statue clears this gate", which is why the plan's §4.8 labels this species' stance by gate kind); the training final checkpoint had collapsed to 64.43 ± 55.23, length 77.1; verdict 2026-09-23 03:09:56 UTC, `gate_sha256 4d88037e…`, `task_sha256 083e2966…`, checkpoint `1d3527cc…` | **FAIL** — `02_locomotion`, 1,450,000 of 12,000,000 steps (58m22s), stopped by the same backstop; the judged `robust_best_model` (`bd32442b…`) stands still: 2249.86 ± 3.96, length 1000, 0.0012 m/s against the 0.9 m/s rail (the reward 100 and length 750 rails pass); best eval 2248.89 at 150k; after the clip release at 800k the policy lunged and fell (557.6 at 850k; 39.94 ± 32.43, length 28.2 at the stop); verdict 2026-09-23 04:09:55 UTC, `gate_sha256 30762cbb…`, `task_sha256 1e348620…` | Re-run session 4 on a `main` that carries the backstop fix (section 3). Bundle `partial` (04:10 UTC): stance certified at replication 1, locomotion not certified. Both nodes' digests equal the current derivation (the fix moves none), so `TRUNK_FROM = "auto"` would reuse the statue-level stance `1d3527cc`: the re-run sets `RETRAIN_FROM = "stance"` |
| compsognathus_robot | none | — | — | — | — | Fresh chain (session 6) |

Verified reuse: at `22c1fc8` both nodes of `20260914_123816` carry a
`task_sha256` (stance `82528a2e…`, locomotion `31383192…`) and a `gate_sha256`
equal to the digests derived from the current stage TOMLs, so
`TRUNK_FROM = "auto"` selects this run and reuses **both** nodes for a trex
session whose chain runs past locomotion (hunt; follow once PR-11 exists); the
target is never reused, so a `stand` or `walk` session consults stance alone and,
since 2026-09-20, picks the newer seed-44 run `20260920_010912` on the tie. It is
the certified r13 walker the plan's Phase C½ asked for, from a from-scratch r13
stance rather than a widened r11 one. Cosmetic staleness: its
run-level `summary.json`, `provenance.json` and `artifact_manifest.json` were
written 2026-09-15 02:08 UTC (after the stance) and never refreshed after the
locomotion verdict (11:39); `training_summary.txt` was refreshed. Reuse reads
per-node files, so nothing breaks, but the run-level records under-report the
run and cannot be rebuilt in place: a re-entry that reuses every node writes no
bundle, and `save_result_bundle` refuses to rewrite a `complete` bundle once
`03_locomotion/` has appeared after its publication (section 3).

Trex stance seed inventory at r13: seed 42 certified, seed 43 failed, seed 44
certified on 2026-09-20 by the widened run `20260920_010912`, whose bundle
(`complete`, written 2026-09-21 after the recovery verdict) records the stance
deliverable as `certified true, provisional false, replication count 2`
(`20260920_010912` seed 44 and `20260914_123816` seed 42). The
`certification_seeds = 2` bar (trex stance alone declares it) is therefore met
as of 2026-09-21. The seed-42 run's own bundle still reads replication 1 (the
count is writer-recorded from the siblings' verdicts, and that `complete`
bundle cannot be rewritten in place). Nothing relies on a `mesozoic-labs/certified` library
directory.

---

## 3. Recommended training sessions

All on `main`, `notebooks/sb3_training.ipynb`, notebook defaults unless stated
(`N_ENVS = 4`, `TRUNK_FROM = "auto"`). Times are
the measured Colab wall clock of the section 2 runs or scaled from them.

Status 2026-09-23: sessions 1, 2 and 3 are done and certified (rows below;
section 2 has the numbers). Session 4 ran on 2026-09-23 as `20260923_020654`
and was cut short: the collapse backstop stopped both nodes at 1.45M steps, a
statue-level stance PASS and a stand-still locomotion FAIL (section 2). The
backstop settings of dibothrosuchus and brachiosaurus stages 1–2 were fixed the
same day (`collapse_peak_warmup_timesteps`; CHANGELOG "Fixed"), so session 4 is
re-run, then sessions 5 and 6 follow, in that order, plus the optional session
7. Sessions 1–3 each ended at the cleanup cell, not the auto-disconnect: the
curves cell wrote three undeclared PNGs into every trained stage directory
after the bundle was sealed (CHANGELOG "Fixed"; the 18 files went to Drive's
trash on 2026-09-23 and the three trees match their manifests again). The fix
landed with the notebook-only PR-12 slice (#552), so a `main` at `03d3a54` or
later releases the runtime at the end of a completed "Run all". Housekeeping: delete the four stray trex directories `20260918_230155`,
`20260918_230335`, `20260919_170528` and `20260919_190251` (first note below). The earlier second step, re-entering the
seed-42 walker `20260914_123816` in place so its bundle cell rebuilds the
run-level records, is dropped: with every node reused the chain loop writes no
bundle, and the run's `complete` stance-target bundle (not `partial`) refuses a
rebuild because `03_locomotion/` appeared after its publication (a complete
bundle is immutable, [RESULT_BUNDLES.md](RESULT_BUNDLES.md)). Reuse reads the per-node files, so nothing depends
on those records.

| # | Species | Settings | What happens | Rough time |
|---|---|---|---|---|
| 1 | trex | `BEHAVIOR="stand"`, `WIDEN_FROM="20260815_205206"`, `WIDEN_MAX_REVISION_GAP=2`, `SEED=44` | **Done.** Ran 2026-09-20 as `20260920_010912`: widen and re-panel PASSED in 15 minutes, the session died at the stance node's bundle write (fixed the same day, #546), and the in-place continuation from 21:38 UTC froze the recovery resolution, trained recovery 3M and PASSED its gate (28/40, LCB 0.56) at 01:16 UTC on 2026-09-21; bundle `complete`, trex stance at replication 2 | panel 15 min, recovery 3h30m (measured) |
| 2 | compsognathus | `BEHAVIOR="walk"`, `WIDEN_FROM="20260909_162812"`, `WIDEN_MAX_REVISION_GAP=1`, `SEED=42` | **Done.** Ran 2026-09-21 as `20260921_203149` on `main` = `25132fc`: widen and re-panel PASSED (verdict 20:38 UTC, about seven minutes after minting), locomotion 3M PASSED (00:31 UTC on 2026-09-22); bundle `complete` in one pass | panel 7 min, walk 3h49m (measured) |
| 3 | velociraptor | `BEHAVIOR="walk"`, `SEED=42` | **Done.** Ran 2026-09-22 as `20260922_125248`: stance 6M PASSED (17:42 UTC; thin length margin, section 2), locomotion 8M PASSED (00:15 UTC on 2026-09-23); bundle `complete` | 4h46m + 6h30m (measured) |
| 4 | dibothrosuchus | `BEHAVIOR="walk"`, `SEED=42`, `RETRAIN_FROM="stance"` | **Ran 2026-09-23 as `20260923_020654`, cut short**: the collapse backstop stopped both nodes at 1.45M (statue-level stance PASS, locomotion FAIL at 0.0012 m/s; section 2). Re-run on a `main` carrying the backstop fix; `RETRAIN_FROM = "stance"` keeps auto-trunk from reusing the statue-level stance (the resolve cell prints `RETRAIN_FROM 'stance': it and every node below it train here (no reuse)`): fresh stance 6M then locomotion 12M. Risk: the locomotion reward pays a motionless statue about 2200, 89 percent of it gait symmetry (KNOWN_ISSUES), the optimum the first run's locomotion settled on | ~4.2 h + ~8.1 h (scaled from the measured 399 and 414 steps/s) |
| 5 | brachiosaurus | `BEHAVIOR="stand"` then, in a second session, `BEHAVIOR="walk"` | stance 6M; the walk session reuses the certified stance through auto-trunk and trains locomotion 16M (both stages carry the 2026-09-23 backstop fix: peak warm-ups of 1.0M and 4.0M). Risk: the locomotion reward pays a motionless statue 2242.7, 98 percent of it gait symmetry (KNOWN_ISSUES) | ~4.5 h then ~10 h |
| 6 | compsognathus_robot | `BEHAVIOR="walk"`, `SEED=42` | fresh stance 11M then locomotion 3M | ~13 h + ~4 h |
| 7 (optional) | trex | `BEHAVIOR="walk"`, `SEED=44`, `TRUNK_FROM="20260920_010912"` (a fresh run) | reuses the seed-44 run's certified stance across runs (recorded under `ancestors/`), trains locomotion 8M and rolls its gate: a second r13 walker seed beside `20260914_123816`. Not in place: `20260920_010912`'s bundle is `complete`, and a complete bundle is immutable, so the notebook refuses an in-place session that would train into it before anything is trained (consolidation PR-14a; before it, the bundle write failed after training) | ~8h46m (the seed-42 walker's measured time) |

Notes:

- **The runtime image moved (by 2026-09-14).** Colab's L4 image went from Python
  3.12.13 / numpy 2.0.2 / jax 0.7.2 (the r11 parent `20260815_205206`, 2026-08-15)
  to Python 3.13.15 / numpy 2.1.3 / jax 0.11.1 (already the image of the
  2026-09-14/15 runs `20260914_123816` and `20260915_160239`), and both first
  attempts at session 1 (`20260919_170528`, `20260919_190251`) died with a kernel
  restart
  inside the widen tool's self-verification: an SB3 archive's schedule members
  embed the saving interpreter's bytecode and a bare `PPO.load` executes it
  (KNOWN_ISSUES, "SB3 archives are bound to the interpreter that saved them").
  Since the loader change of 2026-09-19 every load goes through
  `policy_loading.load_sb3_model`, and a preflight cell loads a real archive
  before anything is trained (at the time right before the widen cell, on the
  `WIDEN_FROM` parent's root handoff; since consolidation PR-14a right after the
  resolve cell, on the trunk run's root handoff when there is one). Two earlier
  trex seed-44 widen sessions of 2026-09-18 at `22c1fc8` (`20260918_230155`,
  `20260918_230335`) were the first to load the older-image archive and left
  the same stray. The four stray run directories are still on Drive
  (2026-09-23; each holds only
  `provenance.json` and four unverified model files under `01_stance/models/`)
  and should be deleted: `select_trunk` lists them as refused, which is
  harmless but noisy (session 1 re-ran beside them without harm). What the
  2026-09-20 run printed, in order: the preflight line `SB3 archive load preflight: loading
  the WIDEN_FROM parent's root handoff .../20260815_205206/stage1/models/robust_best_model.zip
  (saved by Python 3.x; this runtime is Python 3.13; bytecode members:
  clip_range, learning_rate, lr_schedule) ... a kernel death HERE means this
  image cannot load SB3 archives` followed by `SB3 archive load preflight
  passed: archives load back on this runtime through load_sb3_model.`; the
  widen cell's `Widened 'stance' (PPO) into .../01_stance` block, whose
  `num_timesteps inherited` line names `report: .../01_stance/widen_report.json`
  and which closes with `The chain loop will JUDGE this node (no verdict yet);
  parent run '20260815_205206' is untouched.` (the two dead sessions never
  reached it; that report's `schedule_members_source` read
  `parent_stage_config` and its `schedule_members_restated` named
  `learning_rate` and `lr_schedule`, the parent's `clip_range` being a
  constant); the chain loop's `JUDGE` branch
  rolling the 40-episode panel (seeds 3042–3081) and writing
  `01_stance/gate_verdict.json`; then the recovery node's freeze and 3M
  training. The settings are unchanged: `BEHAVIOR="stand"`,
  `WIDEN_FROM="20260815_205206"`, `WIDEN_MAX_REVISION_GAP=2`, `SEED=44`,
  `REPO_REF="main"` (`ac409f8`, which carries the loader change).
- **Continuing session 1 (2026-09-20; done 2026-09-21).** On 2026-09-20 run
  `20260920_010912` held the widened seed-44 stance with a passed
  `gate_verdict.json` and no bundle: the
  chain loop's `save_run_bundle` raised `ResultBundleError: best_eval_reward
  must be a finite number for canonical stage 1` right after the verdict,
  because a widened root never trained in its run and so has no
  `evaluations.npz` curve for `best_eval_reward` to summarize (the JUDGE
  branch leaves it unmeasured). The result schema now accepts that null for a
  stage whose deliverable record names `widened_from_run_id` (CHANGELOG
  2026-09-20, "A widened root's run bundle writes"; on `main` since #546, so
  `REPO_REF = "main"` carries it). The run was finished in place on
  2026-09-20 from 21:38 UTC (provenance `sessions[1]`, commit drift
  `ac409f8` → `25132fc`) with, in a fresh runtime: `SEED = 44`,
  `BEHAVIOR = "stand"`, `WIDEN_FROM = ""`
  (the widened stance already exists; the widen cell must not run again),
  `TRUNK_FROM = ""` (the reuse candidate is this run itself), and in the
  storage cell `RUN_ID = "20260920_010912"` in place of `""` (a
  configuration-cell knob since consolidation PR-14a), so the storage
  cell re-enters the run directory. The chain loop then printed `Reusing this
  run's certified 'stance': robust_best_model (...)`, froze the recovery
  resolution from that handoff (`Freezing the recovery_quality/v1 resolution
  for 'recovery' ...`; `02_recovery/gate_resolution.json` at 21:42 UTC),
  trained recovery 3M (verdict PASS at 01:16 UTC on 2026-09-21) and wrote the
  bundle with the stance's `best_eval_reward` as `null`. A fresh run with
  `TRUNK_FROM = "20260920_010912"` instead would reuse the stance across runs
  and record it under `ancestors/`, leaving `20260920_010912` without a
  bundle; prefer the in-place continuation while the run has no `complete`
  bundle. Once it has one, a later node goes in a fresh run whose `TRUNK_FROM`
  names it (the optional session 7; since consolidation PR-14a the notebook
  refuses an in-place session into a complete run before training). Session 2 (compsognathus widen)
  ran on `25132fc`, which carries the fix, and wrote its bundle in one pass.
- Sessions 1 and 2 widened through the notebook's `WIDEN_FROM` knob and widen
  cell, which left with consolidation PR-14a (D-D14). A widen is now a
  command-line step into a run id the notebook has not opened yet: `python -m
  environments.shared.scripts.widen_checkpoint ... --to-stage-dir
  <LOG_BASE>/<species>/<algo>/<new run id>/01_stance` (`<new run id>` a new
  timestamp id `YYYYMMDD_HHMMSS` that no run uses yet, `--max-revision-gap N`
  for a parent more than one revision behind, `--label` when the session sets
  `RUN_LABEL`; on Colab, the three steps of
  [PLANT_CONTRACT.md](PLANT_CONTRACT.md#widening-a-checkpoint-across-a-policy-interface-bump)
  — section 1, a scratch cell that mounts Drive, never the storage cell, then
  the tool from `/content/mesozoic-labs` — before the storage cell runs for
  that id), then the notebook with `RUN_ID` set to that new run id, `SEED` to the
  parent's recorded `run.seed` and `TRUNK_FROM = ""`. The storage cell
  refuses any other `SEED` before it writes anything (D-C14), so nothing is
  minted under a wrong seed, and its `Run directory:` line reads "re-entering
  run" with the widened root's directory; the resolve cell refuses a trunk
  until the widened root holds a verdict (D-C13).
- A widened root is judged in its new run and every node below it trains
  there (trex `stand` = re-panel stance, then recovery 3M; trex `walk` would
  train locomotion 8M instead). For velociraptor, brachiosaurus and
  dibothrosuchus `stand` is stance only (no recovery node).
- Otherwise leave `TRUNK_FROM = "auto"`.
- A parent `gate_verdict.json` is optional for widening (the r11 and r1 parents
  have none; backfilling first is NOT needed); the parent stage directory must
  hold `stage_config.json` with a run block and a stamped VecNormalize sidecar.
  The widened node gets no verdict, so the chain loop JUDGES it in the new run.
- Any chain can be split across Colab sessions on purpose: a later `walk`
  session reuses a certified stance automatically (session 5 relies on this).
- **Fallback (not needed)**: the widened seed-44 panel passed the duty rail
  (0.0069 / UCB 0.0117), so the fresh trex stance with `SEED = 45` held in
  reserve (about 13 h for the stance alone) is not run.

Not recommended yet: direction/terrain pilots (evaluation only, D-D9; command
line only from the notebook-only PR-12 slice on, with `--checkpoint` /
`--vecnormalize` naming the pair, [TRAIN_DIRECTION_AND_TERRAIN.md](TRAIN_DIRECTION_AND_TERRAIN.md);
one trex `follow_direction` pilot pointed at the September walker's locomotion
checkpoint pair is harmless but disposable) and trex hunting (off the direction path). Never
re-judge or republish a pre-Phase-C run in place (KNOWN_ISSUES, Phase C entry).

---

## 4. Consolidation: the remaining PRs

**Status: released 2026-09-20 in the notebook-first order of decision D-D13**
(PR-3, PR-4, PR-5, PR-6, the notebook-only PR-12 slice, PR-14, then PR-7 .. PR-11,
the rest of PR-12, PR-13, PR-15).
[CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) carries the
per-PR file lists, the breaks / mitigation / validation blocks and the target
architecture table. PR-1 landed as #542, automatic trunk selection as #543,
PR-2 as #544, PR-3 as #546, PR-4 as #547, PR-5 as #548 and PR-6 as #549 (2026-09-20). The maintainer paused the sequence after PR-6 on 2026-09-20 while the section 3 training sessions ran and lifted the pause on 2026-09-23, after the collapse-backstop fix and docs pass of that day (#551). The notebook-only PR-12 slice landed as #552 on 2026-09-24, and PR-14a (2026-09-24) is in review on the session branch; PR-14 lands as PR-14a, PR-14b and PR-14c (D-D15), and its D-D14 condition (sessions 1 and 2 decided) is met. Sizes: S < 200 changed lines, M < 800, L < 2,000, XL above.
Net removal from here (PR-7 .. PR-15, the table's estimates) about 6,800 lines,
of which the notebook-only PR-12 slice removes about 1,130 (measured: −996 code,
test and CI lines, −134 notebook source lines);
the plan's about 9,500 (band 9,000–12,000) was counted from `22c1fc8`, before
PR-3 .. PR-6. No PR changes the on-disk format or the reuse of the canonical
chain, both r11 parents or the r13 run `20260914_123816`; `TRUNK_FROM` (default
`"auto"`) and `RETRAIN_FROM` (empty default) keep their names and defaults
(PR-12 Breaks, PR-14a Breaks), while `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP`
leave with the widen cell in PR-14a (D-D14), which also makes `RUN_ID` a
configuration-cell knob and refuses an in-place session into a complete run
before training; the old widen-seed reorder is dropped (D-D15).

| PR | One-line goal | Size (net) | Prerequisites / decision |
|---|---|---|---|
| PR-3 | Bound the SB3 CI job: drop the SB3-free suites the shared/trex matrix already runs, keep one real-PPO smoke per body of work, move the full six-species set to a schedule (test-sb3 went 43 → 69 min from #539 to #541) | S (+10..+40) | none; land before PR-4 so later deletions edit one list; coverage `fail_under = 70` may need a re-baseline; landed as #546 on 2026-09-20 (coverage 90 percent, the floor unchanged) |
| PR-4 | Delete `certified_canonical.py`, `certified_comparison.py`, their tests and the notebook library hooks (stamp block, the `copy_canonical_ancestor` branch of the chain loop, publish block, and three of the four library knobs; `SOURCE_SELECTION` goes with PR-5) | L (about −2,400) | #542 landed; D-D4; land before PR-5; landed as #547 on 2026-09-20, `certified_comparison.py` and its test moved to PR-5 |
| PR-5 | Delete `certified_library.py`, its consumers in the behavior trainer (`--auto-source`, `--publish-certified`, `--certified-library`), `docs/CERTIFIED_MODELS.md`, the `.gitignore` line | M/L (about −1,400; measured about −1,880) | PR-4 (#547); landed as #548 on 2026-09-20 |
| PR-6 | Delete the T. rex pilots (`configs/trex/behavior_pilots/`), the `[pilot]` recipe dialect and the trex shim script | S (about −275; measured −256 code and configuration lines) | none; D-D9; landed as #549 on 2026-09-20 |
| PR-7 | One behavior env, part 1: `BaseDinoEnv._ground_height_at` / `_clearance`, species rewards and terminations terrain-relative, delete `TRexBehaviorEnv` | M (about −600) | PR-6; D-D10 |
| PR-8 | One behavior env, part 2: one terrain selector (`terrain_sampler` kwarg, `terrain_contact` family), command constants imported from `command_frame`, delete `BehaviorVecNormalize` | M (about −170) | PR-7; D-D3 |
| PR-9 | Phase D through the reserved hook: `command_config` replaces the five numeric kwargs, the controller is owned by `BaseDinoEnv`, identity = task fingerprint (source-hash identity deleted) | M (about −150) | PR-7, PR-8; D-D1, D-D2; acceptance = no committed `task_sha256` moves and `plant_contract --check` clean on every species |
| PR-10 | One command-column primitive in the canonical warm-start path (`policy_loading.neutralize_command_columns` + `assert_command_blind`, called by `_create_or_load_model` on `initialize_next_stage`) | S/M (about +180) | PR-9; D-D3 |
| PR-11 | Manifest nodes: `follow_direction`, `follow_direction_difficult_terrain`, `difficult_terrain` stage TOMLs with `extends`, `[[stages]]` entries after `behavior`, trained by `train_base` | M (about +370) | PR-9, PR-10; D-D1, D-D5, G1 |
| PR-12 (notebook-only slice) | Delete the notebook's `COMMAND_TERRAIN_BEHAVIOR` switch, the ten `BEHAVIOR_*` knobs, the eleven direction/terrain dropdown values, the six behavior cells and the 15 guard sites (guarded code dedented, two lint fixes aside), `behavior_notebook.py` with its tests and pins; `train_behaviors.py` stays the pilots' command-line path | L by count (measured −996 code, test and CI lines, −134 notebook source lines) | PR-6; D-D8 as amended by D-D13; landed as #552 on 2026-09-24 |
| PR-12 (rest) | Delete the parallel trainer, checkpoint module, the 66 TOMLs and their tests; `BEHAVIOR` dropdown gains `follow \| terrain` | XL (about −5,300 less the slice) | PR-11; D-D8, D-D9; `EpisodeManifestRecorder` must survive as an info key |
| PR-13 | Register the gate kind (`none/v1` for pilots, then `terrain_command/v1`) with an evidence writer in the `write_recovery_evidence` pattern; delete `behavior_certification.py` and the certificate schema | L (about −400) | PR-11, PR-12; D-D6, G4 |
| PR-14a | Notebook storage path: the widen cell and `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP` go (D-D14; `widen_checkpoint` stays a command-line tool, its seed and verdict guards become on-disk refusals in the storage and resolve cells), `RUN_ID` becomes a configuration-cell knob resolved into the `_ACTIVE_RUN_ID` memo, and a session that would judge or train a node into a complete run is refused before anything is written (the KNOWN_ISSUES bug of 2026-09-23), with the zero-action cell's run copy skipped on such a run | L by count (measured: code +380 / −20, tests +894 / −680, notebook 2,330 → 2,271 source lines) | the notebook-only PR-12 slice; D-D14, D-D15; in review 2026-09-24 |
| PR-14b | Notebook: one disconnect path (`halt`, explicit-parameter `disconnect_runtime`), `display_stage_videos` on IPython Video, the random-baseline cell deleted and the zero-action table folded into its script | sized when planned | PR-14a; D-D15 |
| PR-14c | Notebook: `train_stage` becomes a wrapper over `train_base.train` (seed line, `parent_run_id`, eval seed, duration), `evaluate_stage_checkpoints` beside `generate_stage_artifacts` | sized when planned (the old PR-14's about −450 was mostly this) | PR-14a; D-D7, D-D11, D-D15 |
| PR-15 | Docs fold, CHANGELOG `Changed` / `Removed`, one notebook-cell test helper, pin budget | M (about −290) | PR-14c |

---

## 5. Decisions taken 2026-09-17

Recorded in [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2 (the D-D and
G series; the D-A/D-B/D-C series keep their numbers). Confirmed by the maintainer:

- **D-D1** Direction-following and terrain traversal are ordinary manifest nodes
  under locomotion (the plan's Phase D shape), not a standalone pipeline.
- **D-D2** One `command_config: DirectionCommandConfig | None` kwarg replaces the
  five numeric reserved kwargs; the task-fingerprint carve-out is extended so no
  canonical `task_sha256` moves while `command_mode` is `"none"`. Amends D-C3.
- **D-D3** Command-slice normalisation follows invariant 8 (reseed to mean 0 /
  var 1, statistics keep updating); `BehaviorVecNormalize` is deleted;
  #540/#541-trained policies are not continuations.
- **D-D4** Automatic parent selection is kept as `select_trunk` (D-A25, #543);
  the certified library is deleted outright (PR-4/PR-5), no pointer survives.
- **D-D5** Stage TOMLs per species with a ~20-line `extends` key in
  `load_stage_config` (locomotion as the parent); terrain templates become a
  `terrain_families` list inside `difficult_terrain`; single-template runs are
  documented `[env]` overrides, not files. Refined by G1 to three new files.
- **D-D6** Honest gate name first: pilots run `none/v1` (recorded, not enforced),
  then the certificate's per-episode statistic is registered as
  `terrain_command/v1` with one shared threshold block; the plan's paired-null
  `command_tracking/v1` (per-event settle/dwell, heading-bin floor) comes later.
- **D-D7** Notebook depth: only the `train_stage` wrapper over `train_base.train`
  now (PR-14); moving the chain loop / widen / resume cells into a package module
  is decided after PR-14.
- **D-D8** No interim behaviors notebook; the mode switch is tolerated until
  PR-12 deletes it.
- **D-D9** #540/#541 behavior bundles on Drive are evaluation-only; none is a
  training parent.
- **D-D10** Terrain stays an opt-in env subclass; no r14 interface bump to move
  the model swap into `reset()` (the reset source is fingerprinted).

Confirmed by the maintainer on 2026-09-20 (recommended on 2026-09-17):

- **D-D11** CLI runs record stage duration and seed model construction like the
  notebook does (PR-14).
- **D-D12** The dead `lateral_speed_scale` field is dropped when the TOMLs are
  rewritten (PR-11/PR-12).

Taken on 2026-09-20, when the consolidation hold lifted:

- **D-D13** The sequence lands notebook-first: PR-3, PR-4, PR-5, PR-6, a
  notebook-only slice of PR-12, PR-14, then PR-7 .. PR-11, the rest of PR-12,
  PR-13, PR-15 (amends D-D8; between the slice and PR-11 the direction/terrain
  pilots have no notebook path).
- **D-D14** The widen path stays for sessions 1 and 2 of [section 3](#3-recommended-training-sessions),
  then becomes CLI-only: the notebook refactor deletes the widen cell and the
  `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP` knobs, and `widen_checkpoint` stays a
  command-line tool for the next interface bump.
- Operational choices taken the same day: the full six-species and
  four-notebook-parameter SB3 sets run nightly and under the `full-ci` label
  (PR-3); if a PR run drops the union coverage gate below 70, the measured
  number and a proposed floor are reported rather than the floor lowered; the
  session results (`gate_verdict.json`, `widen_report.json`,
  `stance_gate_report.json`) are read from Drive through the maintainer's Drive
  connector when a session ends.

Taken on 2026-09-24:

- **D-D15** PR-14 lands as three PRs, in the order PR-14a (the storage path:
  D-D14's widen deletion, `RUN_ID` as a knob, the complete-run refusal), PR-14b
  (the one disconnect path, videos and the baseline cells) and PR-14c
  (`train_stage` over `train_base.train`). The old item (b)'s
  resolve-before-storage reorder is dropped: no widen-seed read is left to
  move, and a root widened on the command line keeps D-C14 and D-C13 through
  on-disk refusals in the storage and resolve cells (amends D-D13's order).

Goal decisions **G1–G4** (chain shape and node set, command set, session order,
first gate thresholds) are in [section 1](#the-final-goal-and-the-goal-decisions).

---

## 6. Lessons learned (2026-09-12 .. 2026-09-23)

1. A publication step that can disconnect the runtime stays off by default until
   every input it needs exists: `PUBLISH_CERTIFIED = True` would have killed the
   first widen session after the stance passed (fixed by #542).
2. Reuse is decided by per-node files, not run-level summaries (the r13 run's
   run-level records are stale, yet auto-trunk reuses it). Offline check: compare
   the node's `task_sha256` and `gate_sha256` with `derive_stage_task_fingerprint`
   and `gate_config_sha256(gate_config_view(...))` at the checkout.
3. The stance duty rail is the binding constraint at r13: seed 43 passed reward
   and horizon and failed unsupported duty (0.0323 / UCB 0.0350 vs 0.02). A failed
   panel is a measured deficit; re-rolling it does not help, another seed does.
4. Widening beats retraining whenever the parent's physics digest, `nq`/`nv`/`nu`,
   `action_dim` and obs + 3 match and the crossed revisions were fingerprint-only;
   the July velociraptor and brachiosaurus runs fail those conditions and are
   fresh starts. Check the identity gate offline before booking Colab time.
5. Split long chains across Colab sessions on purpose now that
   `TRUNK_FROM = "auto"` picks the certified stance up (brachiosaurus: 6M + 16M).
6. A feature built beside reserved hooks instead of through them (the pilots
   ignored `BaseDinoEnv._draw_episode_command`, `GATE_KINDS`,
   `find_certified_ancestor` and the stage TOML loader) costs more to fold back
   than it saved; the design of record had already named hook, kwargs and gate.
7. A node added in place to a run whose bundle is already `complete` never
   reaches the run-level records: the chain loop's bundle write refused it
   after the node had trained and halted "Run all" (2026-09-23); since
   consolidation PR-14a the resolve cell (or, for a node it cannot predict,
   the chain loop) refuses such a session before anything is trained or
   written, and names the fix. Train later
   nodes in a fresh run whose `TRUNK_FROM` names that run. A `partial` bundle
   is rebuilt by the next node trained or judged in it; a session that only
   reuses a complete run's nodes writes nothing into it (the zero-action cell
   no longer rewrites a complete run's copy either), while one that only
   reuses a `partial` run's nodes in a new runtime records a session in its
   `provenance.json` and stops at the cleanup cell (KNOWN_ISSUES, LOW); and
   per-node artifacts are written immediately.
8. Two seeds at a stance bar of two is reachable only by widening the r11
   parents or training fresh r13 seeds; backfilling the r11 verdicts does
   nothing for reuse (rules 3 and 6 refuse pre-Phase-C archives).
9. A collapse backstop whose floor sits below what the untrained or the
   warm-started policy already scores arms on initialisation and ends the stage
   at the first exploration dip, and no run record says it stopped early
   (dibothrosuchus `20260923_020654`, 2026-09-23). Replay the shipped
   `evaluations.npz` through `EvalCollapseEarlyStopCallback` before reading an
   early stop as a collapse.
10. A cell that runs after the bundle is sealed must not write under
   `RUN_DIR`. The curves cell's saved PNGs stopped every completed "Run all" of
   sessions 1–3 at the cleanup cell's bundle check, so no runtime was
   auto-released and the three bundles stopped validating, unnoticed for three
   sessions (found 2026-09-23 while mapping PR-14). The pin that executes the
   curves cell covers it; a new post-seal cell needs the same.

---

## 7. Open questions and risks

- Does a widened stance reproduce its panel under the new interface? Answered
  yes, to every printed digit: trex seed 44 (r11 → r13, session 1, 2026-09-20:
  3408.3 ± 88.5, full-horizon 1.0000, duty 0.0069 / UCB 0.0117 on both sides)
  and compsognathus seed 42 (r1 → r2, session 2, 2026-09-21: 2801.6 ± 51.2,
  full-horizon 1.0000, duty 0.0131 / UCB 0.0141 on both sides).
- Settled 2026-09-19: the canonical `alg_cls.load` path had never crossed an
  interpreter boundary (the r13 run warm-started from its own stance under one
  image), and the first cross-interpreter load — the widen tool's
  self-verification of the r11 parent under the 3.13 image — killed the
  kernel. Every load now goes through `policy_loading.load_sb3_model`;
  `behavior_checkpoint._load_ppo`'s own `custom_objects` guard is redundant
  with it and goes with PR-12 as planned.
- Two dated task-fingerprint valves (`allow_unfingerprinted`, the schema-v1 valve)
  may be dead on r13 species (~60-line follow-up once no r11 parent is left to
  widen, which holds since 2026-09-20: the seed-44 parent is widened and the
  seed-42 parent's widen is superseded); task lineage may need
  `parent_normalization_sha256` (~15 lines).
- `EpisodeManifestRecorder` (per-episode terrain manifests) must survive PR-12 as
  an info key under `train_base`'s Monitor/DiagnosticsCallback;
  `canonical_env_parameters`' allowed-`[env]`-keys check must be re-homed in
  `load_stage_config` when `read_recipe` goes (PR-9/PR-11).
- PR-8 moves single-template recipes from a Bernoulli `flat_probability` draw to
  balanced blocks: a distribution change to state in the decision record.
- PR-9 touches five species constructors, `MJXEnvConfig` and the fingerprint
  carve-out; acceptance = no committed `task_sha256` moves
  (`test_phase_c_interface.py`) and `plant_contract --check` reports no interface
  change, on every species. MJX reward kernels stay world-z after PR-7 (MJX has
  no terrain and fails closed on live commands); note the divergence in `mjx_env`.
- CI: PR-3 (#546) dropped the SB3-free suites from the `test-sb3` job (the `test`
  matrix already runs them) and moved the full six-species and
  four-notebook-parameter sets to the nightly schedule and the `full-ci` label,
  keeping one real-PPO smoke per body of work on every PR; the union coverage
  gate `fail_under = 70` read 90 percent on its CI runs and is re-measured again
  after PR-12/PR-13.
- Test-to-test coupling to untangle in order: trex `test_behavior_training`
  imports `CommandEnv` from `test_behavior_checkpoint` (the
  `test_behavior_publication` import left with PR-5).
- `notebooks/jax_training.ipynb` carries the same Drive-mount block and
  `_ACTIVE_RUN_ID` memo; since consolidation PR-14a the SB3 notebook's `RUN_ID`
  is a configuration-cell knob while the JAX notebook keeps it in its storage
  cell; take D-D7-style moves for both notebooks.
- The plan's per-event `command_tracking/v1` statistic and the
  worst-of-heading-bins floor have no implementation anywhere (D-D6's second
  kind is new work).
- Totals uncertainty: PR-12's −5,300 depends on D-D5 and on how much of
  `test_behavior_species_training` survives; band 9,000–12,000 lines.
- Velociraptor: [investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md)
  (2026-07-20) left a pending Commit B validation; the fresh r10 stance run
  (session 3, `20260922_125248`, 2026-09-22) is the first data since: PASS
  under `reward_and_length/v1` with a thin length margin (section 2). The
  Commit B validation itself is still unrecorded.
- Dibothrosuchus and brachiosaurus walkers (2026-09-23): the locomotion reward
  pays a motionless statue about 2200, mostly gait symmetry (KNOWN_ISSUES), and
  the first dibothrosuchus locomotion node settled on standing before the
  backstop cut it. Whether a gait emerges within the 12M / 16M budgets is open;
  the session 4 re-run answers it for dibothrosuchus, and the reward fix belongs
  with the planned plant and reward work on both species.
- Phase B items deferred by the maintainer on 2026-09-13 stay deferred
  ([BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §10).

---

## 8. How to continue

### First steps in a new session

1. Read this file; then [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md)
   if touching code, [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §4–§6
   for the design and decision ids, and the Phase C entry of
   [KNOWN_ISSUES.md](KNOWN_ISSUES.md) ("Training / RL") before any widen session.
2. The consolidation hold lifted on 2026-09-20 (D-D13 order); the maintainer
   paused the sequence after PR-6 the same day and lifted the pause on
   2026-09-23. Continue with the next PR of
   [section 4](#4-consolidation-the-remaining-prs) (PR-14a is in review; then PR-14b and PR-14c;
   the notebook-only PR-12 slice landed as #552) on the session branch, one PR at a time,
   restarting the branch from `main` after each merge.
3. Check Drive for run directories newer than 2026-09-17 (through the Drive
   connector when the maintainer has attached one) and update
   [section 2](#2-certified-checkpoints-on-drive) here (the survey stays frozen).
4. The docs that were stale at `22c1fc8` (the `README.md` roadmap bullet that
   read "only the command-interface bump and the follow-direction leaf remain
   pending", the `docs/README.md` plan row and its missing rows for
   `CERTIFIED_MODELS.md` / `TRAIN_DIRECTION_AND_TERRAIN.md`, the plan's §5 rows
   C½ / D / E, §7, §9 SS1 and §10, the CHANGELOG entry for #540/#541, and the
   KNOWN_ISSUES gaps) were corrected by the PR-2 docs pass (#544, 2026-09-19).

### Branch and validation rules

- Automated sessions develop on the branch the session names, restart it from
  `origin/main` after each merge, commit with clear messages, push to that
  branch when the work is complete, and never push elsewhere.
- Before a push: `ruff check .` and `ruff format --check .`;
  `mypy environments/ --ignore-missing-imports`; `pytest environments/shared/tests/`
  in chunks; `pytest environments/<species>/tests/`; the notebook checks (every
  code cell parses, as `.github/workflows/python-ci.yml` does; an edited notebook
  round-trips through `json.dump` with `indent=1`, the plan's §5 PR process); the
  SB3 job for trainer changes. The review container is a venv with SB3 2.9.0,
  torch, jax, mujoco 3.10.0 and `JAX_PLATFORMS=cpu`, without IPython; since the
  notebook-only PR-12 slice deleted `test_behavior_notebook.py` no test needs it.
- Never write an AI model or vendor name into repository files; cite PR numbers,
  branch names or "the maintainer".

### Doc conventions (from [README.md](README.md))

- `docs/` root = living reference and plans (this file is living reference);
  dated investigations under `investigations/` are frozen at their date, with
  corrections appended, and get a dated header and a row in `docs/README.md`;
  `KNOWN_ISSUES.md` is the single list of verified-but-unfixed findings (fixed
  items are deleted, context stays in the archived review or investigation).
- Decision ids are used exactly as they exist in the plan (D1–D5 original,
  D-A1..D-A25, D-B1..D-B17, D-C1..D-C17, D-D1..D-D15, G1..G4); never renumber.
  Relative markdown links only; every link must resolve.

### The widened-interface template note

[investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md](investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md)
is a template with placeholders for Sessions 1 (seed-42 widen of
`20260810_145546`), 1b (recovery freeze re-roll), the seed-44 repeat, 2 (the C½
walker) and 3 (stand). The Drive survey changed what was still
needed: Session 1 is **superseded** (seed 42 is already certified at r13 by
`20260914_123816`), Session 2 is **superseded** by that run's certified
locomotion, the seed-44 repeat is section 3's session 1 (`BEHAVIOR="stand"`, so
recovery trains in the same run), and Sessions 1b and 3 are **covered** by that
recovery node (the chain loop freezes the resolution from the widened handoff
first). Its §6 (appended 2026-09-19) records these supersessions; its §7
(2026-09-20) and §8 (2026-09-23) record the seed-44 run `20260920_010912`, whose
results fill the §1–§3 seed-44 columns (from the shipped `widen_report.json`,
`gate_verdict.json`, `stance_gate_report.txt` and `02_recovery/gate_resolution.json`);
the seed-42 columns stay empty with a pointer to `20260914_123816`.

### Where things live

| What | Where |
|---|---|
| Design of record, decision series | [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) |
| Consolidation sequence, per-PR file lists, target architecture | [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) |
| Drive state as surveyed 2026-09-17 | [investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md) |
| Bundle layout, `gate_verdict.json`, `ancestors/` records | [RESULT_BUNDLES.md](RESULT_BUNDLES.md) |
| Plant identities and the widen contract | [PLANT_CONTRACT.md](PLANT_CONTRACT.md), `configs/plant_versions.toml` |
| Reuse rules 1–7, trunk selection, widen and backfill tools | `environments/shared/ancestors.py` (`find_certified_ancestor`, `select_trunk`); `environments/shared/scripts/widen_checkpoint.py`, `environments/shared/scripts/backfill_gate_verdict.py` |
| Gate digests and task fingerprints | `environments/shared/curriculum/gate_schema.py` (`gate_config_view`, `gate_config_sha256`), `environments/shared/task_fingerprint.py` (`derive_stage_task_fingerprint`) |
| The notebook, its pins, the changelog | [notebooks/sb3_training.ipynb](../notebooks/sb3_training.ipynb); `environments/shared/tests/test_sb3_notebook_pins.py`; [CHANGELOG.md](../CHANGELOG.md) |
