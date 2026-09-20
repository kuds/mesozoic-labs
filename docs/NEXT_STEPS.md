# Next steps and program state (2026-09-20)

**Status**: living reference — updated 2026-09-20; `main` = `a5ed12f` (2026-09-20).

Read this first when starting a new session on the behavior-recipes program: what
has landed, what is certified on Drive, which training sessions to run next, where
the consolidation stands, and which decisions bind. Repository facts were verified at
`22c1fc8`; Drive facts and the maintainer's decisions date from 2026-09-17
([investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md)).
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
| #532–#535 | 2026-09-13 | Phase B: gate-configuration digest + reuse rule 7 + `backfill_gate_verdict --gate`; the `task_success/v1` hunting gate; seed replication as provenance |
| #536–#539 | 2026-09-13/14 | Phase C: the 3-dim command segment on all six species, `command_frame.py`, `widen_checkpoint.py`, the notebook `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP` knobs; decisions D-C1–D-C17 |
| #540, #541 | 2026-09-15 | T. rex direction-following and randomized-terrain pilots; all-species terrain behaviors and the certified library — a **separate pipeline** beside the recipes machinery (next paragraph) |
| #542 | 2026-09-16 | `PUBLISH_CERTIFIED` notebook default flipped to `False` (consolidation PR-1): library publication required a training-origin stamp a widened root handoff lacks, so a `WIDEN_FROM` session would have disconnected the Colab runtime right after the stance passed its gate |
| #543 | 2026-09-16 | Automatic trunk selection, decision D-A25: `environments/shared/ancestors.select_trunk`, notebook `TRUNK_FROM = "auto"` default, CLI `curriculum --trunk-from auto`. Canonical chains no longer consult the certified library; widen sessions select no trunk |
| #544 | 2026-09-19 | Consolidation PR-2: this file, the consolidation plan, the Drive survey note, decisions D-D1..D-D12 and G1..G4 in the plan's §6.2, the docs index and CHANGELOG |
| #545 | 2026-09-20 | The version-safe SB3 archive loader (`policy_loading.load_sb3_model`; picklable `LinearSchedule` / `CosineSchedule`; the widen tool re-states parent schedules) and the notebook's archive-load preflight cell before the widen cell, after two widen sessions died on the Python 3.13 image (KNOWN_ISSUES, "SB3 archives are bound to the interpreter that saved them") |
| #546 | 2026-09-20 | Consolidation PR-3: the bounded `test-sb3` job (one real-PPO smoke per body of work on pull requests and pushes; the six-species and four-notebook-parameter sets nightly at 05:17 UTC, on `workflow_dispatch` and under the `full-ci` label, which now starts a run when added); the widened-root bundle-write fix (a null `best_eval_reward` for a stage whose deliverable record names `widened_from_run_id`); decisions D-D11..D-D14 recorded. Measured on its CI: lean SB3 job 48:02, labelled full job 45:21, coverage 90 percent |

The notebook at `22c1fc8` ([notebooks/sb3_training.ipynb](../notebooks/sb3_training.ipynb))
has 40 cells (22 code), 2,526 lines; 19 code cells reference the
`COMMAND_TERRAIN_BEHAVIOR` mode switch (the 2026-09-19 loader change adds one
guarded code cell, the SB3 archive-load preflight before the widen cell:
41 cells, 23 code, 2,563 lines; consolidation PR-4 removes the library hooks,
about 90 lines: 41 cells, 23 code, 2,477 lines, the chain loop at index 23;
PR-5 removes `SOURCE_SELECTION` and its prose: 41 cells, 23 code, 2,463 lines).
Configuration-cell defaults:
`BEHAVIOR = "hunt"` (dropdown: `stand`, `walk`, `hunt`, eleven direction/terrain
values, stage ids by free input), `TRUNK_FROM = "auto"`, `WIDEN_FROM = ""`,
`WIDEN_MAX_REVISION_GAP = 1`, `RETRAIN_FROM = ""`, `SEED = 42`; the library
knobs (`CERTIFIED_LIBRARY_ROOT`, `PUBLISH_CERTIFIED`,
`CERTIFIED_COMPARISON_EPISODES`) left with PR-4 and `SOURCE_SELECTION` with PR-5.

### The pilot pipeline (#540/#541) — exists, evaluation-only

The pilots delivered genuinely new content (a direction controller, a tracking
reward, a heightfield terrain generator, a replay recorder with terrain maps) but
as a second copy of every canonical concept: two behavior env classes that bypass
the reserved `BaseDinoEnv._draw_episode_command` hook and write `self._command`
directly, a second checkpoint preparer, a second PPO trainer
(`environments/shared/train_behaviors.py`, one CPU env, its own recipe dialect),
66 behavior TOMLs under `configs/<species>/behaviors/` (11 templates x 6 species)
plus 8 under `configs/trex/behavior_pilots/`, a second gate outside `GATE_KINDS`
(`configs/behavior_certification.toml`, judged by `judge_behavior_panel`; its
`certification/certificate.json` writer left with PR-5, so it is reached only
from tests until PR-13; never a `gate_verdict.json`), a third identity keyed on
source-file hashes (any edit to `behavior_env.py` strands exact resume), and the
notebook mode switch — about 7,000 lines of modules at 22c1fc8, tests excluded
(the certified library left with PR-4/PR-5). Outputs go
to `logs/<species>/ppo/behaviors/<behavior>/<run-id>/` on Drive; the certified
library left with PR-5, so a pilot needs explicit `BEHAVIOR_CHECKPOINT` /
`BEHAVIOR_VECNORMALIZE` paths in every load mode and for evaluation.
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
PR-11), then PR-14, then PR-7 .. PR-11, the rest of PR-12, PR-13 and PR-15.
D-D1..D-D14 are taken and recorded (D-D11/D-D12 confirmed and D-D13/D-D14 taken
on 2026-09-20; [section 5](#5-decisions-taken-2026-09-17)). Training sessions
run in parallel (G3).

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

Drive layout `mesozoic-labs/logs/<species>/<algo>/<run>/`, surveyed 2026-09-17.
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
| trex | `20260914_123816` | 42 | r13 (commit `35dd44c`) | **PASS** — `01_stance`, 11,001,856 steps, 13h13m, final eval 3460.6 ± 19.2, `gate_verdict.json` judged 2026-09-15 01:57 UTC, `gate_sha256 ff2494ba…`, handoff `robust_best_model.zip` | **PASS** — `03_locomotion`, 8,011,776 steps, 8h46m, 1.07 m/s, mean length 1000, reward 1940.8, verdict 2026-09-15 11:39 UTC, `gate_sha256 02602cb0…`, `task_sha256 31383192…` | A second stance seed at r13 (`certification_seeds = 2`; session 1); the recovery node (never trained at r13; session 1 trains one from the widened seed-44 stance; a `BEHAVIOR="stand"` session under `TRUNK_FROM = "auto"` would train one on this run's stance); follow nodes once PR-11 exists |
| trex | `20260915_160239` | 43 | r13 (`35dd44c`) | **FAIL** — `01_stance` only, 11M steps (13h17m), final eval 3209.7 ± 284.3, best 3335.1; the 40-episode panel failed unsupported duty (mean 0.0323, UCB 0.0350, rail 0.02), reward and full-horizon passed; provenance `certified false`, `provisional true` | none | Nothing; a measured deficit, kept as history |
| trex | `20260920_010912` | 44 | r13 (commit `ac409f8`), widened from `20260815_205206` (r11, gap 2) by `widen_checkpoint/v1` | **PASS** — `01_stance`, judged 2026-09-20 01:15 UTC by `generate_stage_artifacts` on the widened handoff `robust_best_model.zip` (`checkpoint_sha256 7f4284ad…`, inherited 10,000,000 steps, no training here): panel reward 3408.3 ± 88.5, full-horizon 1.0000, duty 0.0069, UCB 0.0117, 40 episodes seeds 3042–3081, identical to the r11 certificate; final eval 3418.22 ± 87.88; `gate_sha256 ff2494ba…`, `task_sha256 82528a2e…` (= the seed-42 run's stance digest) | none | The recovery node and the run bundle: the session died at the stance node's bundle write (fixed 2026-09-20, section 3 "Continuing session 1"); `summary.json` / `artifact_manifest.json` are absent until the continuation writes them |
| trex | `20260815_205206` | 44 | r11 (legacy `stage1/` + `stage2/`) | PASS on `stance_gate_report.txt`: reward 3408.3 ± 88.5, full-horizon 1.0000, duty 0.0069, UCB 0.0117, 40 episodes seeds 3042–3081; **no `gate_verdict.json`**; `stage1/` holds `stage_config.json` (run block) and `models/` | not certified | Nothing: widened to r13 and re-paneled as `20260920_010912` (session 1, 2026-09-20) |
| trex | `20260810_145546` | 42 | r11 (legacy `stage1/2/3`) | PASS on the 2026-08 records; no `gate_verdict.json` in `stage1` | not certified | Nothing: seed 42 is already certified at r13 by `20260914_123816`; the template note's Session 1 (widen this run) is superseded |
| compsognathus | `20260909_162812` | 42 | r1 (obs 53, commit `9557e97`) | PASS on `stance_gate_report.txt`: reward 2801.6 ± 51.2, full-horizon 1.0000, duty 0.0131, UCB 0.0141, 40 episodes seeds 3042–3081; **no `gate_verdict.json`**; run block seed 42, n_envs 4, 11,000,000 steps; `physics_sha256 08a5fbf7…` unchanged at r2 | none | Widen r1 → r2 under the default gap 1 (53 + 3 = 56), `SEED = 42`, re-panel, then locomotion 3M (session 2) |
| velociraptor | `20260723_005740` (July) | 42 | r3 (obs 67); physics r2 = current | stage1 balance 6M, 3h41m, final eval 1767, 1000-step episodes; run-level `publication_gate_passed` only, no per-node verdict; sidecar predates identity stamping | stage2 8M, 4h50m, 3.29 m/s; stage3 strike 12M | **Not a widen candidate**: seven revisions behind r10 and the crossed revisions include the reset-settling change (`plant_versions.toml` note 6). Fresh chain (session 3) |
| brachiosaurus | `20260717_162659` (July) | 42 | r1; physics r1 (current r4) | stage1 balance 6M, 4h24m, 1739.8 | stage2 16M, 9h48m, 1.42 m/s; stage3 food_reach 12M | Physics digest differs, so widen is refused by construction. Fresh chain (session 5) |
| dibothrosuchus | none | — | — | — | — | Fresh chain (session 4) |
| compsognathus_robot | none | — | — | — | — | Fresh chain (session 6) |

Verified reuse: at `22c1fc8` both nodes of `20260914_123816` carry a
`task_sha256` (stance `82528a2e…`, locomotion `31383192…`) and a `gate_sha256`
equal to the digests derived from the current stage TOMLs, so
`TRUNK_FROM = "auto"` selects this run and reuses **both** nodes for any trex
session: it is the certified r13 walker the plan's Phase C½ asked for, from a
from-scratch r13 stance rather than a widened r11 one. Cosmetic staleness: its
run-level `summary.json`, `provenance.json` and `artifact_manifest.json` were
written 2026-09-15 02:08 UTC (after the stance) and never refreshed after the
locomotion verdict (11:39); `training_summary.txt` was refreshed. Reuse reads
per-node files, so nothing breaks, but the run-level records under-report the
run until the bundle cell is re-run in a later session.

Trex stance seed inventory at r13: seed 42 certified, seed 43 failed, seed 44
certified on 2026-09-20 by the widened run `20260920_010912` (its
`gate_verdict.json` is written; its run bundle is not yet, see section 3). The
`certification_seeds = 2` bar (trex stance alone declares it) is therefore met
once that bundle exists and the seed-42 run's bundle cell is re-run beside it
(the replication count reads its siblings' verdicts). Nothing relies on a
`mesozoic-labs/certified` library directory.

---

## 3. Recommended training sessions

All on `main`, `notebooks/sb3_training.ipynb`, notebook defaults unless stated
(`N_ENVS = 4`, `TRUNK_FROM = "auto"`). Times are
the measured Colab wall clock of the section 2 runs or scaled from them.

| # | Species | Settings | What happens | Rough time |
|---|---|---|---|---|
| 1 | trex | `BEHAVIOR="stand"`, `WIDEN_FROM="20260815_205206"`, `WIDEN_MAX_REVISION_GAP=2`, `SEED=44` | widens the seed-44 r11 stance to r13, re-panels it (40 episodes), then trains recovery 3M. **Ran 2026-09-20 as `20260920_010912`**: widen and re-panel PASSED in 15 minutes, then the session died at the stance node's bundle write (fixed the same day); the recovery node is owed by the continuation below | panel ~1 h (measured 15 min), recovery ~3.5 h |
| 2 | compsognathus | `BEHAVIOR="walk"`, `WIDEN_FROM="20260909_162812"`, `WIDEN_MAX_REVISION_GAP=1`, `SEED=42` | widens the r1 stance to r2, re-panels it, trains locomotion 3M | panel ~1 h, walk ~4 h |
| 3 | velociraptor | `BEHAVIOR="walk"`, `SEED=42` | fresh stance 6M then locomotion 8M | ~4 h + ~5 h |
| 4 | dibothrosuchus | `BEHAVIOR="walk"`, `SEED=42` | fresh stance 6M then locomotion 12M | ~4 h + ~7 h |
| 5 | brachiosaurus | `BEHAVIOR="stand"` then, in a second session, `BEHAVIOR="walk"` | stance 6M; the walk session reuses the certified stance through auto-trunk and trains locomotion 16M | ~4.5 h then ~10 h |
| 6 | compsognathus_robot | `BEHAVIOR="walk"`, `SEED=42` | fresh stance 11M then locomotion 3M | ~13 h + ~4 h |

Notes:

- **The runtime image moved (2026-09-19).** Colab's L4 image went from Python
  3.12.13 / numpy 2.0.2 / jax 0.7.2 (the 2026-09-14/15 runs) to Python 3.13.15 /
  numpy 2.1.3 / jax 0.11.1, and both first attempts at session 1
  (`20260919_170528`, `20260919_190251`) died with a kernel restart inside the
  widen tool's self-verification: an SB3 archive's schedule members embed the
  saving interpreter's bytecode and a bare `PPO.load` executes it (KNOWN_ISSUES,
  "SB3 archives are bound to the interpreter that saved them"). Since the
  loader change of 2026-09-19 every load goes through
  `policy_loading.load_sb3_model`, and a preflight cell right before the widen
  cell loads the `WIDEN_FROM` parent's real root handoff first. Before re-running
  session 1 delete the two stray run directories (each holds only
  `provenance.json` and four unverified model files under `01_stance/models/`;
  `select_trunk` would list them as refused, which is harmless but noisy). What
  to look for, in order: the preflight line `SB3 archive load preflight: loading
  the WIDEN_FROM parent's root handoff .../20260815_205206/stage1/models/robust_best_model.zip
  (saved by Python 3.x; this runtime is Python 3.13; bytecode members:
  clip_range, learning_rate, lr_schedule) ... a kernel death HERE means this
  image cannot load SB3 archives` followed by `SB3 archive load preflight
  passed: archives load back on this runtime through load_sb3_model.`; the
  widen cell's `Widened 'stance' (PPO) into .../01_stance` block, whose
  `num_timesteps inherited` line names `report: .../01_stance/widen_report.json`
  and which closes with `The chain loop will JUDGE this node (no verdict yet);
  parent run '20260815_205206' is untouched.` (the two dead sessions never
  reached it; that report's `schedule_members_source` will most likely read
  `current_stage_config`, because the parent's `stage_config.json` predates
  the `hyperparameters` block by a day, and its `schedule_members_restated`
  names the three re-stated members); the chain loop's `JUDGE` branch
  rolling the 40-episode panel (seeds 3042–3081) and writing
  `01_stance/gate_verdict.json`; then the recovery node's freeze and 3M
  training. The settings are unchanged: `BEHAVIOR="stand"`,
  `WIDEN_FROM="20260815_205206"`, `WIDEN_MAX_REVISION_GAP=2`, `SEED=44`,
  `REPO_REF="main"` once the loader change has merged.
- **Continuing session 1 (2026-09-20).** Run `20260920_010912` holds the
  widened seed-44 stance with a passed `gate_verdict.json` and no bundle: the
  chain loop's `save_run_bundle` raised `ResultBundleError: best_eval_reward
  must be a finite number for canonical stage 1` right after the verdict,
  because a widened root never trained in its run and so has no
  `evaluations.npz` curve for `best_eval_reward` to summarize (the JUDGE
  branch leaves it unmeasured). The result schema now accepts that null for a
  stage whose deliverable record names `widened_from_run_id` (CHANGELOG
  2026-09-20, "A widened root's run bundle writes"; on `main` since #546, so
  `REPO_REF = "main"` carries it). To finish the run in place,
  in a fresh runtime: `SEED = 44`, `BEHAVIOR = "stand"`, `WIDEN_FROM = ""`
  (the widened stance already exists; the widen cell must not run again),
  `TRUNK_FROM = ""` (the reuse candidate is this run itself), and in the
  storage cell `RUN_ID = "20260920_010912"` in place of `""`, so the storage
  cell re-enters the run directory. The chain loop then prints `Reusing this
  run's certified 'stance': robust_best_model (...)`, freezes the recovery
  resolution from that handoff (`Freezing the recovery_success/v1 resolution
  for 'recovery' ...`), trains recovery 3M, rolls its panel and writes the
  bundle with the stance's `best_eval_reward` as `null`. A fresh run with
  `TRUNK_FROM = "20260920_010912"` instead would reuse the stance across runs
  and record it under `ancestors/`, leaving `20260920_010912` without a
  bundle; prefer the in-place continuation. Session 2 (compsognathus widen)
  would have died at the same line; it needs the fix as well.
- Sessions 1 and 2 must set `SEED` to the parent's seed **before the storage cell
  mints `RUN_ID`**: the widen cell refuses `SEED != ` the parent's recorded
  `run.seed` (D-C14) and a directory minted under the wrong seed is not
  re-minted (correct `SEED`, restart the runtime or `del _ACTIVE_RUN_ID`, delete
  the stray directory, which holds only `provenance.json`).
- With `WIDEN_FROM` set the chain loop uses no trunk: every node below the
  widened root trains in that run (trex `stand` = widen + re-panel stance, then
  recovery 3M; trex `walk` would train locomotion 8M instead). For velociraptor,
  brachiosaurus and dibothrosuchus `stand` is stance only (no recovery node).
- Leave `TRUNK_FROM = "auto"`.
- A parent `gate_verdict.json` is optional for widening (the r11 and r1 parents
  have none; backfilling first is NOT needed); the parent stage directory must
  hold `stage_config.json` with a run block and a stamped VecNormalize sidecar.
  The widened node gets no verdict, so the chain loop JUDGES it in the new run.
- Any chain can be split across Colab sessions on purpose: a later `walk`
  session reuses a certified stance automatically (session 5 relies on this).
- **Fallback**: if the widened seed-44 panel fails the duty rail as seed 43 did,
  run a fresh trex stance with `SEED = 45` (about 13 h for the stance alone).

Not recommended yet: direction/terrain pilots (evaluation only, D-D9; one trex
`follow_direction` pilot pointed at the September walker's locomotion checkpoint
pair is harmless but disposable) and trex hunting (off the direction path). Never
re-judge or republish a pre-Phase-C run in place (KNOWN_ISSUES, Phase C entry).

---

## 4. Consolidation: the remaining PRs

**Status: released 2026-09-20 in the notebook-first order of decision D-D13**
(PR-3, PR-4, PR-5, PR-6, the notebook-only PR-12 slice, PR-14, then PR-7 .. PR-11,
the rest of PR-12, PR-13, PR-15).
[CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) carries the
per-PR file lists, the breaks / mitigation / validation blocks and the target
architecture table. PR-1 landed as #542, automatic trunk selection as #543,
PR-2 as #544, PR-3 as #546 (2026-09-20); PR-4 (#547) and PR-5 are in review on the session branch. Sizes: S < 200 changed lines, M < 800, L < 2,000, XL above.
Net removal from here about 9,500 lines (band 9,000–12,000). No PR changes the
on-disk format or the reuse of the canonical chain, both r11 parents or the r13
run `20260914_123816`; `WIDEN_FROM` / `TRUNK_FROM` / `RETRAIN_FROM` keep their
names and empty defaults (PR-12 Breaks, PR-14 Breaks, PR-15 pins), though PR-14
reorders the widen-seed check and edits the resume cell's prose.

| PR | One-line goal | Size (net) | Prerequisites / decision |
|---|---|---|---|
| PR-3 | Bound the SB3 CI job: drop the SB3-free suites the shared/trex matrix already runs, keep one real-PPO smoke per body of work, move the full six-species set to a schedule (test-sb3 went 43 → 69 min from #539 to #541) | S (+10..+40) | none; land before PR-4 so later deletions edit one list; coverage `fail_under = 70` may need a re-baseline |
| PR-4 | Delete `certified_canonical.py`, `certified_comparison.py`, their tests and the notebook library hooks (stamp block, the `copy_canonical_ancestor` branch of the chain loop, publish block, and three of the four library knobs; `SOURCE_SELECTION` goes with PR-5) | L (about −2,400) | #542 landed; D-D4; land before PR-5 |
| PR-5 | Delete `certified_library.py`, its consumers in the behavior trainer (`--auto-source`, `--publish-certified`, `--certified-library`), `docs/CERTIFIED_MODELS.md`, the `.gitignore` line | M/L (about −1,400; measured about −1,880) | PR-4 (#547); in review 2026-09-20 |
| PR-6 | Delete the T. rex pilots (`configs/trex/behavior_pilots/`), the `[pilot]` recipe dialect and the trex shim script | S (about −275) | none; D-D9 |
| PR-7 | One behavior env, part 1: `BaseDinoEnv._ground_height_at` / `_clearance`, species rewards and terminations terrain-relative, delete `TRexBehaviorEnv` | M (about −600) | PR-6; D-D10 |
| PR-8 | One behavior env, part 2: one terrain selector (`terrain_sampler` kwarg, `terrain_contact` family), command constants imported from `command_frame`, delete `BehaviorVecNormalize` | M (about −170) | PR-7; D-D3 |
| PR-9 | Phase D through the reserved hook: `command_config` replaces the five numeric kwargs, the controller is owned by `BaseDinoEnv`, identity = task fingerprint (source-hash identity deleted) | M (about −150) | PR-7, PR-8; D-D1, D-D2; acceptance = no committed `task_sha256` moves and `plant_contract --check` clean on every species |
| PR-10 | One command-column primitive in the canonical warm-start path (`policy_loading.neutralize_command_columns` + `assert_command_blind`, called by `_create_or_load_model` on `initialize_next_stage`) | S/M (about +180) | PR-9; D-D3 |
| PR-11 | Manifest nodes: `follow_direction`, `follow_direction_difficult_terrain`, `difficult_terrain` stage TOMLs with `extends`, `[[stages]]` entries after `behavior`, trained by `train_base` | M (about +370) | PR-9, PR-10; D-D1, D-D5, G1 |
| PR-12 | Delete the parallel trainer, router, checkpoint module, the 66 TOMLs, the notebook mode switch and their tests; `BEHAVIOR` dropdown becomes `stand \| walk \| hunt \| follow \| terrain` | XL (about −5,300) | PR-11; D-D8, D-D9; `EpisodeManifestRecorder` must survive as an info key |
| PR-13 | Register the gate kind (`none/v1` for pilots, then `terrain_command/v1`) with an evidence writer in the `write_recovery_evidence` pattern; delete `behavior_certification.py` and the certificate schema | L (about −400) | PR-11, PR-12; D-D6, G4 |
| PR-14 | Notebook: `train_stage` becomes a ~30-line wrapper over `train_base.train`, widen-seed check before minting, one storage and one disconnect path, `RUN_ID` as a knob | M (about −450) | PR-4, PR-12; D-D7, D-D11; the widen cell and knobs leave here or right after (D-D14) |
| PR-15 | Docs fold, CHANGELOG `Changed` / `Removed`, one notebook-cell test helper, pin budget | M (about −290) | PR-14 |

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

Goal decisions **G1–G4** (chain shape and node set, command set, session order,
first gate thresholds) are in [section 1](#the-final-goal-and-the-goal-decisions).

---

## 6. Lessons learned (2026-09-12 .. 2026-09-17)

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
7. A run continued in a later Colab session leaves the run-level
   summary/provenance stale unless the bundle cell is re-run; per-node artifacts
   are written immediately.
8. Two seeds at a stance bar of two is reachable only by widening the r11
   parents or training fresh r13 seeds; backfilling the r11 verdicts does
   nothing for reuse (rules 3 and 6 refuse pre-Phase-C archives).

---

## 7. Open questions and risks

- Does a widened stance reproduce its panel under r13? Unanswered (no widen
  session has run yet); sessions 1 and 2 answer it for trex and compsognathus.
- Settled 2026-09-19: the canonical `alg_cls.load` path had never crossed an
  interpreter boundary (the r13 run warm-started from its own stance under one
  image), and the first cross-interpreter load — the widen tool's
  self-verification of the r11 parent under the 3.13 image — killed the
  kernel. Every load now goes through `policy_loading.load_sb3_model`;
  `behavior_checkpoint._load_ppo`'s own `custom_objects` guard is redundant
  with it and goes with PR-12 as planned.
- Two dated task-fingerprint valves (`allow_unfingerprinted`, the schema-v1 valve)
  may be dead on r13 species (~60-line follow-up after the r11 parents are
  widened); task lineage may need `parent_normalization_sha256` (~15 lines).
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
- CI: PR-3 drops the SB3-free suites from the `test-sb3` job (the `test` matrix
  already runs them) and moves the full six-species and four-notebook-parameter
  sets to the nightly schedule and the `full-ci` label, keeping one real-PPO
  smoke per body of work on every PR; the union coverage gate `fail_under = 70`
  is re-measured on its first CI run and again after PR-12/PR-13.
- Test-to-test coupling to untangle in order: trex `test_behavior_training`
  imports `CommandEnv` from `test_behavior_checkpoint` (the
  `test_behavior_publication` import left with PR-5).
- `notebooks/jax_training.ipynb` carries the same Drive-mount block and
  `_ACTIVE_RUN_ID` memo; take D-D7-style moves for both notebooks.
- The plan's per-event `command_tracking/v1` statistic and the
  worst-of-heading-bins floor have no implementation anywhere (D-D6's second
  kind is new work).
- Totals uncertainty: PR-12's −5,300 depends on D-D5 and on how much of
  `test_behavior_species_training` survives; band 9,000–12,000 lines.
- Velociraptor: [investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md)
  (2026-07-20) left a pending Commit B validation; a fresh r10 stance run
  (session 3) is the first data since.
- Phase B items deferred by the maintainer on 2026-09-13 stay deferred
  ([BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §10).

---

## 8. How to continue

### First steps in a new session

1. Read this file; then [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md)
   if touching code, [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §4–§6
   for the design and decision ids, and the Phase C entry of
   [KNOWN_ISSUES.md](KNOWN_ISSUES.md) ("Training / RL") before any widen session.
2. The consolidation hold lifted on 2026-09-20 (D-D13 order): continue with the
   next PR of [section 4](#4-consolidation-the-remaining-prs) on the session
   branch, one PR at a time, restarting the branch from `main` after each merge.
3. Check Drive for run directories newer than 2026-09-17 (through the Drive
   connector when the maintainer has attached one) and update
   [section 2](#2-certified-checkpoints-on-drive) here (the survey stays frozen).
4. The docs that were stale at `22c1fc8` (the `README.md` roadmap bullet that
   read "only the command-interface bump and the follow-direction leaf remain
   pending", the `docs/README.md` plan row and its missing rows for
   `CERTIFIED_MODELS.md` / `TRAIN_DIRECTION_AND_TERRAIN.md`, the plan's §5 rows
   C½ / D / E, §7, §9 SS1 and §10, the CHANGELOG entry for #540/#541, and the
   KNOWN_ISSUES gaps) were corrected by the PR-2 docs pass; if that pass has not
   merged yet, read them on its branch.

### Branch and validation rules

- Automated sessions develop on the branch the session names, restart it from
  `origin/main` after each merge, commit with clear messages, push to that
  branch when the work is complete, and never push elsewhere.
- Before a push: `ruff check .` and `ruff format --check .`;
  `mypy environments/ --ignore-missing-imports`; `pytest environments/shared/tests/`
  in chunks; `pytest environments/<species>/tests/`; the notebook checks (every
  code cell parses, as `.github/workflows/python-ci.yml` does; an edited notebook
  round-trips through `json.dump` with `indent=1`, the plan's §5 PR process); the
  SB3 job for trainer changes. IPython is absent in the review container (a venv
  with SB3 2.9.0, torch, jax, mujoco 3.10.0, `JAX_PLATFORMS=cpu`), so 12
  `test_behavior_notebook` display tests fail there identically on `main`.
- Never write an AI model or vendor name into repository files; cite PR numbers,
  branch names or "the maintainer".

### Doc conventions (from [README.md](README.md))

- `docs/` root = living reference and plans (this file is living reference);
  dated investigations under `investigations/` are frozen at their date, with
  corrections appended, and get a dated header and a row in `docs/README.md`;
  `KNOWN_ISSUES.md` is the single list of verified-but-unfixed findings (fixed
  items are deleted, context stays in the archived review or investigation).
- Decision ids are used exactly as they exist in the plan (D1–D5 original,
  D-A1..D-A25, D-B1..D-B17, D-C1..D-C17, D-D1..D-D14, G1..G4); never renumber.
  Relative markdown links only; every link must resolve.

### The widened-interface template note

[investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md](investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md)
is a template with placeholders for Sessions 1 (seed-42 widen of
`20260810_145546`), 1b (recovery freeze re-roll), the seed-44 repeat, 2 (the C½
walker) and 3 (stand). None has run, and the Drive survey changed what is still
needed: Session 1 is **superseded** (seed 42 is already certified at r13 by
`20260914_123816`), Session 2 is **superseded** by that run's certified
locomotion, the seed-44 repeat is section 3's session 1 (`BEHAVIOR="stand"`, so
recovery trains in the same run), and Sessions 1b and 3 are **covered** by that
recovery node (the chain loop freezes the resolution from the widened handoff
first). Its §6 (appended 2026-09-19) records these supersessions; after session 1
runs, fill the note's §1–§3 seed-44 columns from the shipped `widen_report.json`,
`gate_verdict.json`, `stance_gate_report.json` and `02_recovery/gate_resolution.json`.

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
