# Next steps and program state (2026-10-04)

**Status**: living reference — updated 2026-10-04. Landing status is not kept here (section 1).

Read this first when starting a new session on the behavior-recipes program: what
is certified on Drive, which training sessions to run next, what the consolidation
has left, and which decisions bind. Drive facts date from the 2026-09-17 survey
([investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md))
plus the section 3 session results recorded since.
Companions: [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) (design of
record), [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) (per-PR
sequence, file lists, rationale), [KNOWN_ISSUES.md](KNOWN_ISSUES.md). Update this
file in place when the state changes; it is not a dated investigation.

---

## 1. Where things stand

Landing status (which PR landed, as what number, when, with what CI result) is
kept in two places only: the status table of
[CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md), whose "Cleanup
and backend retirement" row carries the cleanup's PRs, and each PR's entry in
[CHANGELOG.md](../CHANGELOG.md) (the one-landing-record rule of
[README.md](README.md#conventions), since cleanup CU-17). The cleanup plan's §3.1
keeps the cleanup's records up to CU-17 as its history. This section's landing
narrative and its "Landed on `main`" table (#528 .. #591, with the CI
measurements of most PRs from #546 on), as they stood before CU-17, are at
commit `ace8112`: `git show ace8112:docs/NEXT_STEPS.md`.

In order: 0.3.9 (`20ab100`) is the clean base release (D-D21), and
Stable-Baselines3 is the only training backend (D-D17); the deferred cleanup
(the cleanup plan's §3.1 item 5) comes before consolidation PR-8..PR-15
([section 4](#4-consolidation-the-remaining-prs)) and before the gait plan's
code PRs; the training sessions are [section 3](#3-recommended-training-sessions).

The gait audit of 2026-09-28
([investigations/GAIT_AUDIT_2026_09.md](investigations/GAIT_AUDIT_2026_09.md))
replayed every certified node on Drive: three of the five certified walkers
hop on both feet (trex `20260914_123816` and `20260925_033501`,
compsognathus_robot `20260924_031815`), and only compsognathus
`20260921_203149` walks and velociraptor `20260922_125248` runs; the
zero-action statue passes every stance gate; the robot's touch sensors count
sole-on-sole contact as floor support; and the in-progress dibothrosuchus
re-run `20260928_012318` is, at 4.3M, a three-legged skid its gate would pass.
No locomotion gate reads a foot contact.
[GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md) proposes the fix
(floor-truth gait measurement, per-episode gait gates, per-species reward
revisions) and puts all of its code after the 0.3.9 cut and, by the
maintainer's choice of 2026-10-02, after the deferred cleanup; every one of
its decisions (GQ-1..GQ-18) is open. The audit and the plan came with the plan's
docs-only PR-G0, outside D-D21's gate. Section 2 labels each
audited node except the in-training dibothrosuchus re-run, which section 3's
session-4 row describes, and the plan's §10 holds a prompt for continuing the
gait work in a fresh session.

The notebook ([notebooks/sb3_training.ipynb](../notebooks/sb3_training.ipynb)) is the one SB3 training
notebook; the consolidation plan's §4 maps its cells. Its sizes from consolidation PR-3 on are in the
consolidation plan's status table, the CHANGELOG or the cleanup plan's §3.1; the earlier ones (at `22c1fc8` and
after the 2026-09-19 loader change) are in section 1 at `ace8112`.

Configuration-cell defaults:
`BEHAVIOR = "hunt"` (dropdown: `stand`, `walk`, `hunt`, stage ids by free input;
the eleven direction/terrain values leave with the PR-12 slice),
`TRUNK_FROM = "auto"`, `RETRAIN_FROM = ""`, `RUN_LABEL = ""`, `RUN_ID = ""`
(a configuration-cell knob since consolidation PR-14a), `SEED = 42`. The knobs the
consolidation retired are named in the CHANGELOG entries of PR-4, PR-5, the PR-12
slice and PR-14a.

### The pilot pipeline (#540/#541) — exists, evaluation-only

The direction/terrain pilots are a second pipeline beside every canonical concept; the KNOWN_ISSUES entry
"the direction/terrain pilots are a second pipeline" ([KNOWN_ISSUES.md](KNOWN_ISSUES.md)) lists its parts, and
the consolidation folds them back. No pilot output is on Drive: nothing was trained beyond 4,096-step smoke runs
([CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §5.2). A pilot runs from the command line only
(`python -m environments.shared.train_behaviors`, D-D13) with an explicit `--checkpoint` / `--vecnormalize`
pair in every load mode and for evaluation; guide:
[TRAIN_DIRECTION_AND_TERRAIN.md](TRAIN_DIRECTION_AND_TERRAIN.md). Under D-D9 any pilot bundle is
evaluation-only; none is a training parent.

### The consolidation (PR-3 .. PR-15): released 2026-09-20, notebook-first

The 2026-09-17 review produced a fifteen-PR sequence folding the pilot pipeline
back into the recipes machinery ([section 4](#4-consolidation-the-remaining-prs)).
On 2026-09-20 the maintainer released the hold with a **notebook-first order**
(decision D-D13): PR-3, then PR-4 and PR-5, then PR-6, then a notebook-only
slice of PR-12 (the mode switch, the `BEHAVIOR_*` knobs, the direction/terrain
cells and guard sites and `behavior_notebook.py` go; `train_behaviors.py` stays
CLI-only until the rest of PR-12, after PR-11), then PR-14 (split on 2026-09-24
into PR-14a, PR-14b and PR-14c, decision D-D15), then PR-7 .. PR-11, the rest of
PR-12, PR-13 and PR-15, with the training sessions in parallel (G3). On
2026-10-02 the maintainer chose to finish the deferred cleanup before PR-8 (the
cleanup plan's §3.1 item 5). Which of these PRs have landed is in the
consolidation plan's status table (above); the decisions are recorded where
[section 5](#5-decisions-taken-2026-09-17) points.

### The final goal and the goal decisions

The target behavior is **every species follows a direction on difficult
terrain**. The maintainer fixed its shape on 2026-09-17:

- **G1** chain shape `walk -> follow_direction` (full command set on flat ground)
  `-> follow_direction_difficult_terrain` (the target deliverable); a commands-free
  `difficult_terrain` node stays as an optional diagnostic sibling; recipe label
  `follow` resolves to the deepest deliverable; `follow_direction_speed` is folded
  into `follow_direction`; three new stage files per species; the final node is
  SB3-only (MJX fails closed on live commands and has no terrain). Amended by
  D-D17 (cleanup PR-B): nothing trains on MJX any more, so every node is SB3-only.
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
run-level summaries. A cell ending "gait audit 2026-09-28: …" gives, in plain
words, what the 2026-09-28 replays show that node doing
([investigations/GAIT_AUDIT_2026_09.md](investigations/GAIT_AUDIT_2026_09.md)
§4); the label changes no certification fact, and a verdict still certifies
only what its gate checks (no locomotion gate reads a foot contact).

| Species | Run id | Seed | Interface | Stance | Locomotion | What it needs next |
|---|---|---|---|---|---|---|
| trex | `20260914_123816` | 42 | r13 (commit `35dd44c`) | **PASS** — `01_stance`, 11,001,856 steps, 13h13m, final eval 3460.6 ± 19.2, `gate_verdict.json` judged 2026-09-15 01:57 UTC, `gate_sha256 ff2494ba…`, handoff `robust_best_model.zip`; gait audit 2026-09-28: a quiet, wide two-footed stance, the only clean one of the six certified stances | **PASS** — `03_locomotion`, 8,011,776 steps, 8h46m, 1.07 m/s, mean length 1000, reward 1940.8, verdict 2026-09-15 11:39 UTC, `gate_sha256 02602cb0…`, `task_sha256 31383192…`; gait audit 2026-09-28: two-footed hop, not a walk | Nothing for the two-seed bar: the second stance seed is `20260920_010912` (seed 44, certified 2026-09-20), whose bundle counts this run (replication 2 of 2). Its own run-level records stay stance-only (bundle `complete` under a stance target, replication 1): a complete bundle is immutable and not rebuilt in place ([RESULT_BUNDLES.md](RESULT_BUNDLES.md)), and reuse reads the per-node files, so nothing depends on them. A recovery node on this run's stance would be a `BEHAVIOR="stand"` session under `TRUNK_FROM = "20260914_123816"` (`"auto"` now picks the newer seed-44 run for a stand or walk session, whose chain consults stance alone; optional: the seed-44 run already certified recovery at r13); follow nodes once PR-11 exists |
| trex | `20260915_160239` | 43 | r13 (`35dd44c`) | **FAIL** — `01_stance` only, 11M steps (13h17m), final eval 3209.7 ± 284.3, best 3335.1; the 40-episode panel failed unsupported duty (mean 0.0323, UCB 0.0350, rail 0.02), reward and full-horizon passed; provenance `certified false`, `provisional true` | none | Nothing; a measured deficit, kept as history |
| trex | `20260920_010912` | 44 | r13 (commit `ac409f8`), widened from `20260815_205206` (r11, gap 2) by `widen_checkpoint/v1` | **PASS** — `01_stance`, judged 2026-09-20 01:15 UTC by `generate_stage_artifacts` on the widened handoff `robust_best_model.zip` (`checkpoint_sha256 7f4284ad…`, inherited 10,000,000 steps, no training here): panel reward 3408.3 ± 88.5, full-horizon 1.0000, duty 0.0069, UCB 0.0117, 40 episodes seeds 3042–3081, identical to the r11 certificate; final eval 3418.22 ± 87.88; `gate_sha256 ff2494ba…`, `task_sha256 82528a2e…` (= the seed-42 run's stance digest); gait audit 2026-09-28: quiet in 34 of 40 episodes, two-footed hops to rebalance in the other 6 | none; **recovery PASS** instead — `02_recovery`, 3,006,464 steps, 3h30m (2026-09-20 21:42 → 2026-09-21 01:12 UTC, the in-place continuation), 28/40 panel successes (Clopper-Pearson one-sided LCB 0.56 ≥ 0.30), paired success delta against the 0/40 statue null 0.70 (Student-t one-sided LCB 0.58 ≥ 0.20), 140/155 pushes recovered, 34/40 full horizon, panel reward 2925.4 ± 404.8; training final eval 2911.16 ± 556.35, best eval 3031.38 ± 159.07 at 2.9M; verdict 2026-09-21 01:16 UTC, `gate_sha256 a27ce071…`, `task_sha256 2c6f4a47…`, checkpoint `a8e41b98…`; gait audit 2026-09-28 (recovery): keeps its posture after pushes but answers forward pushes with two-footed hops | Nothing: its stance is the parent of the seed-44 walker `20260925_033501` (next row, session 7, a fresh run; not in place, because this run's bundle is `complete` and a complete bundle is immutable); otherwise nothing: bundle `complete` (written after the recovery verdict), stance deliverable `certified true, provisional false`, `replication count 2` (`20260920_010912` seed 44, `20260914_123816` seed 42), recovery certified at replication 1 |
| trex | `20260815_205206` | 44 | r11 (legacy `stage1/` + `stage2/`) | PASS on `stance_gate_report.txt`: reward 3408.3 ± 88.5, full-horizon 1.0000, duty 0.0069, UCB 0.0117, 40 episodes seeds 3042–3081; **no `gate_verdict.json`**; `stage1/` holds `stage_config.json` (run block) and `models/` | not certified | Nothing: widened to r13 and re-paneled as `20260920_010912` (session 1, 2026-09-20) |
| trex | `20260810_145546` | 42 | r11 (legacy `stage1/2/3`) | PASS on the 2026-08 records; no `gate_verdict.json` in `stage1` | not certified | Nothing: seed 42 is already certified at r13 by `20260914_123816`; the template note's Session 1 (widen this run) is superseded |
| compsognathus | `20260909_162812` | 42 | r1 (obs 53, commit `9557e97`) | PASS on `stance_gate_report.txt`: reward 2801.6 ± 51.2, full-horizon 1.0000, duty 0.0131, UCB 0.0141, 40 episodes seeds 3042–3081; **no `gate_verdict.json`**; run block seed 42, n_envs 4, 11,000,000 steps; `physics_sha256 08a5fbf7…` unchanged at r2 | none | Nothing: widened to r2 and re-paneled as `20260921_203149` (session 2, 2026-09-21); stays on the log tree as history |
| compsognathus | `20260921_203149` | 42 | r2 (obs 56, commit `25132fc`), widened from `20260909_162812` (r1, gap 1) by `widen_checkpoint/v1` | **PASS** — `01_stance`, judged 2026-09-21 20:38 UTC by `generate_stage_artifacts` on the widened handoff (`checkpoint_sha256 1d46747f…`, inherited `num_timesteps` 10,850,000, no training here): panel reward 2801.6 ± 51.2, full-horizon 1.0000, duty 0.0131, UCB 0.0141, 40 episodes seeds 3042–3081, identical to the r1 report; `gate_sha256 62930e3f…`, `task_sha256 19837f77…`; gait audit 2026-09-28: marches in place, not a still stance | **PASS** — `03_locomotion`, 3,002,368 steps, 3h49m, 0.34 m/s, mean length 1000, final eval 3308.6 ± 13.0, best eval 3337.16 ± 8.11 at 2.6M, verdict 2026-09-22 00:31 UTC, `gate_sha256 33e185d3…`, `task_sha256 63195040…`; gait audit 2026-09-28: a genuine alternating walk | Nothing: bundle `complete` (2026-09-22 00:31 UTC); the stance deliverable records `widened_from_run_id 20260909_162812` and a null `best_eval_reward` (the rule of #546); this is the certified compsognathus walker `TRUNK_FROM = "auto"` selects |
| velociraptor | `20260723_005740` (July) | 42 | r3 (obs 67); physics r2 = current | stage1 balance 6M, 3h41m, final eval 1767, 1000-step episodes; run-level `publication_gate_passed` only, no per-node verdict; sidecar predates identity stamping | stage2 8M, 4h50m, 3.29 m/s; stage3 strike 12M | **Not a widen candidate**: seven revisions behind r10 and the crossed revisions include the reset-settling change (`plant_versions.toml` note 6). Fresh chain ran as `20260922_125248` (session 3, 2026-09-22) |
| velociraptor | `20260922_125248` | 42 | r10 (commit `25132fc`); physics r2 | **PASS** — `01_stance`, 6,004,736 steps, 4h46m; certified checkpoint (30-episode selection eval) reward 1740.03 ± 301.5, mean length 970.3 ± 159.9 against the `reward_and_length/v1` rails 1050 / 950; training final eval 1666.19 ± 424.23, length 943.2; best eval 1797.43 at 4.35M; verdict 2026-09-22 17:42 UTC, `task_sha256 07d6a5af…`, checkpoint `732b5a71…`; gait audit 2026-09-28: a crouch with the feet chattering and the body sliding, not a stance (the zero-action statue passes the same gate) | **PASS** — `02_locomotion`, 8,011,776 steps, 6h30m; final eval 2683.64 ± 8.56, length 1000, 3.30 m/s; certified checkpoint 2591.88 ± 493.74, length 968.6, 3.17 m/s; best eval 2687.56 at 6.9M; verdict 2026-09-23 00:15 UTC, `task_sha256 6178a6ef…`, checkpoint `7c28eda2…`; gait audit 2026-09-28: a genuine alternating run | Nothing for the chain: bundle `complete`, both deliverables certified (replication 1 each). The stance margin is thin: the certified checkpoint clears the reward rail by 690 and the length rail by 20 steps, and the training final eval's mean length (943.2) is below the 950 rail. A second stance seed (`SEED = 43`) is the cheap check if velociraptor stance is ever declared at `certification_seeds = 2` |
| brachiosaurus | `20260717_162659` (July) | 42 | r1; physics r1 (current r4) | stage1 balance 6M, 4h24m, 1739.8 | stage2 16M, 9h48m, 1.42 m/s; stage3 food_reach 12M; gait audit 2026-09-28 (stage2): a neck-pumping asymmetric bound, not a walk | Physics digest differs, so widen is refused by construction. Fresh chain (session 5) |
| dibothrosuchus | `20260923_020654` | 42 | r7 (commit `25132fc`); physics r1 | **PASS, statue-level** — `01_stance`, 1,450,000 of 6,000,000 steps (1h00m30s), stopped by the collapse backstop; the judged `robust_best_model` is the 50k evaluation's: 2597.49 ± 1.95, length 1000, no forward motion, against the `reward_and_length/v1` rails 1560 / 950, while the run's zero-action statue scores 2598.29 ± 0.86 (40/40 full horizon; `zero_action_baseline.json` reads "FAILS — a statue clears this gate", which is why the plan's §4.8 labels this species' stance by gate kind); the training final checkpoint had collapsed to 64.43 ± 55.23, length 77.1; verdict 2026-09-23 03:09:56 UTC, `gate_sha256 4d88037e…`, `task_sha256 083e2966…`, checkpoint `1d3527cc…`; gait audit 2026-09-28: a statue, on three legs in a few episodes | **FAIL** — `02_locomotion`, 1,450,000 of 12,000,000 steps (58m22s), stopped by the same backstop; the judged `robust_best_model` (`bd32442b…`) stands still: 2249.86 ± 3.96, length 1000, 0.0012 m/s against the 0.9 m/s rail (the reward 100 and length 750 rails pass); best eval 2248.89 at 150k; after the clip release at 800k the policy lunged and fell (557.6 at 850k; 39.94 ± 32.43, length 28.2 at the stop); verdict 2026-09-23 04:09:55 UTC, `gate_sha256 30762cbb…`, `task_sha256 1e348620…`; gait audit 2026-09-28: a standing statue, paid 1997.3 of its 2249.9 by `gait_symmetry` | Re-run session 4 on a `main` that carries the backstop fix (section 3; the maintainer started it on 2026-09-28). Bundle `partial` (04:10 UTC): stance certified at replication 1, locomotion not certified. Both nodes' digests equal the current derivation (the fix moves none), so `TRUNK_FROM = "auto"` would reuse the statue-level stance `1d3527cc`: the re-run sets `RETRAIN_FROM = "stance"` |
| trex | `20260925_033501` | 44 | r13 (commit `9d729e8`); physics r7 | Reused from `20260920_010912` (`ancestors/stance`, `checkpoint_sha256 7f4284ad…`, `task_sha256 82528a2e…`) | **PASS** — `03_locomotion`, 8,011,776 steps, 8h24m (2026-09-25 03:36 → 12:03 UTC), 1.57 m/s, mean length 1000, reward 2317.5 ± 25.2 (selected checkpoint: 30 episodes, 2316.8 ± 24.4, 1.57 ± 0.03 m/s, 15.8 m), best eval 2309.7 at 8.0M; verdict 2026-09-25 12:03 UTC under the seed-42 walker's gate (`gate_sha256 02602cb0…`, `task_sha256 31383192…`), handoff `robust_best_model.zip` (`checkpoint_sha256 e6de9c92…`, the 8.0M checkpoint); bundle `complete`; gait audit 2026-09-28: the seed-42 walker's two-footed hop, faster and wider; not a walk | Nothing: a second r13 trex walker seed beside `20260914_123816` (seed 42, 1.07 m/s). It walks about 1.5× faster than the trex follow recipes' 1.05 m/s cruise, so the direction/terrain speed fit holds for the seed-42 walker only |
| compsognathus_robot | `20260924_031815` | 42 | policy interface r2 (commit `03d3a54`); physics r1 | **PASS** — `01_stance`, 11,001,856 steps, 19h31m; final eval 2971.57 ± 5.62, best 2980.26 at 10.15M; `stance_quality/v1` panel (40 episodes, seeds 3042–3081) reward 2979.2 ± 4.0, full-horizon 1.0000, duty 0.0000, UCB 0.0000, bilateral support 0.9984 (statue 0.998, not gated); verdict 2026-09-24 22:57 UTC, `task_sha256 26a8ec2b…`, `gate_sha256 62930e3f…`, handoff `robust_best_model.zip` (`checkpoint_sha256 b4093e69…`); gait audit 2026-09-28: stands on the left foot with the right sole resting on it; still, but not two-footed | **PASS** — `03_locomotion`, 3,000,704 steps: about 4h19m until the Colab cap stopped it after `stage2_2800000_steps` (2026-09-24 22:58 → 2026-09-25 03:17 UTC), then the last ~200k steps in 18m34s by the RESUME cell (2026-09-28, provenance session 2 from 01:08:38 UTC, `main` = `7ae0a19`); final eval 2605.91 ± 11.91, every episode the full 1,000 steps, 0.227 ± 0.002 m/s; best eval 2625.02 ± 10.67 at 2.8M; handoff `robust_best_model.zip` (`checkpoint_sha256 dbbfb709…`) 2624.38 ± 11.12 over 30 episodes, 0.23 m/s, 4.56 m; `reward_and_length/v1` rails 500 / 900 / 0.04 m/s (20 episodes); verdict 2026-09-28 01:36 UTC by `generate_stage_artifacts`, `task_sha256 ae9b535c…`, `gate_sha256 1dd4e1bc…`; gait audit 2026-09-28: a synchronous micro-hop from that stacked stance, not a walk | Nothing: bundle `complete`, stance and locomotion certified, not provisional (replication 1 each). `provenance.json` records the resume's expected environment drift (commit `03d3a54` → `7ae0a19`, `mesozoic_labs` 0.3.8.dev0 → 0.3.9.dev0), which moves no identity. The bundle's `total_training_time` 19:49:58 counts the stance (19:31:23) and the resume (18:34) only, not the ~4h19m locomotion segment of the capped session. The maintainer's look at the locomotion video (2026-09-28) questions whether it really walks (it may hop or scoot with a staggered stance): `reward_and_length/v1` checks no foot pattern, so a gait-quality check is planned for after the cleanup ([GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md)); the 2026-09-28 audit found a hop (the locomotion cell) |

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
later releases the runtime at the end of a completed "Run all". Session 6 started
first, on 2026-09-24 at 03:18 UTC, as `20260924_031815` on `main` = `03d3a54`; its stance passed at 22:57 UTC after 19h31m, not the ~13 h estimated, so locomotion ran into Colab's ~24 h session cap; the maintainer resumed it on 2026-09-28 and locomotion passed (the table's row 6, section 2), and the same day started the session 4 re-run. Housekeeping: delete the four stray trex directories `20260918_230155`,
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
| 4 | dibothrosuchus | `BEHAVIOR="walk"`, `SEED=42`, `RETRAIN_FROM="stance"` | **Ran 2026-09-23 as `20260923_020654`, cut short**: the collapse backstop stopped both nodes at 1.45M (statue-level stance PASS, locomotion FAIL at 0.0012 m/s; section 2). Re-run on a `main` carrying the backstop fix; `RETRAIN_FROM = "stance"` keeps auto-trunk from reusing the statue-level stance (the resolve cell prints `RETRAIN_FROM 'stance': it and every node below it train here (no reuse)`): fresh stance 6M then locomotion 12M. The maintainer started the re-run on 2026-09-28 (a fresh run); its result is recorded here and in section 2 when it finishes. The gait audit of 2026-09-28 replayed that run, `20260928_012318` (on `main` = `7ae0a19`), while it trained: its 4.3M robust-best checkpoint skids on three legs at 2.20 m/s, and its gate as written would pass it (1.95 m/s, mean length 854); by 4.8M it rolls and tips. The replay read the run as reusing the statue-level stance `1d3527cc…` although this row sets `RETRAIN_FROM = "stance"`; that reuse needs confirming from the run's `ancestors/stance/ancestor.json` (the gait plan's GQ-2, open; the plan recommends letting the run finish and be judged). Risk: the locomotion reward pays a motionless statue about 2200, 89 percent of it gait symmetry (KNOWN_ISSUES), the optimum the first run's locomotion settled on. A resume of `20260928_012318` on `main` since the 0.3.9 cut merged (#584) also records `mesozoic_labs` 0.3.9.dev0 → 0.4.0.dev0 in `provenance.json` as environment drift, beside the commit drift, which is expected and moves no identity; only an install from the `0.3.9` tag (`20ab100`) reports `0.3.9`, since `main` never pointed at a commit that reads it | ~4.2 h + ~8.1 h (scaled from the measured 399 and 414 steps/s) |
| 5 | brachiosaurus | `BEHAVIOR="stand"` then, in a second session, `BEHAVIOR="walk"` | stance 6M; the walk session reuses the certified stance through auto-trunk and trains locomotion 16M (both stages carry the 2026-09-23 backstop fix: peak warm-ups of 1.0M and 4.0M). Risk: the locomotion reward pays a motionless statue 2242.7, 98 percent of it gait symmetry (KNOWN_ISSUES). The gait plan recommends holding this session until the brachiosaurus task revision `gait-r1` (GQ-3, open): the zero-action statue also passes the brachiosaurus stance gate (1739.1 against the 1040 rail) | ~4.5 h then ~10 h |
| 6 | compsognathus_robot | `BEHAVIOR="walk"`, `SEED=42` | **Done.** Ran from 2026-09-24 03:18 UTC as `20260924_031815` (`main` = `03d3a54`): stance 11M PASSED (22:57 UTC, section 2); locomotion trained from 22:58 UTC until the ~24 h Colab cap stopped it after `stage2_2800000_steps` (2026-09-25 03:17 UTC), about 200k steps short of its 3M. The maintainer resumed it in place on 2026-09-28 (provenance session 2 from 01:08:38 UTC, `main` = `7ae0a19`): the RESUME cell trained the last ~200k steps in 18m34s (to 3,000,704) and locomotion PASSED (01:36 UTC, 0.23 m/s, section 2); bundle `complete`, stance and locomotion certified. `provenance.json` records the expected environment drift (commit `03d3a54` → `7ae0a19`, `mesozoic_labs` 0.3.8.dev0 → 0.3.9.dev0), which moves no identity | 19h31m + ~4h19m to the cap + 18m34s resume (measured; the bundle's `total_training_time`, 19:49:58, omits the capped segment) |
| 7 (optional) | trex | `BEHAVIOR="walk"`, `SEED=44`, `TRUNK_FROM="20260920_010912"` (a fresh run) | **Done.** Ran 2026-09-25 03:35 → 12:03 UTC as `20260925_033501` (`main` = `9d729e8`): stance reused from `20260920_010912` (`ancestors/stance`; the verdict's task, gate and plant digests matched), locomotion 8M PASSED (12:03 UTC, 1.57 m/s, section 2); bundle `complete`. The plan was: reuses the seed-44 run's certified stance across runs (recorded under `ancestors/`), trains locomotion 8M and rolls its gate: a second r13 walker seed beside `20260914_123816`. Not in place: `20260920_010912`'s bundle is `complete`, and a complete bundle is immutable, so the notebook refuses an in-place session that would train into it before anything is trained (consolidation PR-14a; before it, the bundle write failed after training) | 8h24m (measured) |

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
  run" with the widened root's directory; since ROW-4/6 the chain loop judges
  the widened root before it consults any trunk (D-C13, amended by the cleanup
  plan's decision 6 (b); the resolve cell no longer refuses a trunk), and
  refuses a root widened into a run that already holds it as an `ancestors/`
  record when the root is an ancestor of `BEHAVIOR`'s node, naming a new run
  id.
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

**T. rex gait-r1 pilot (prepared 2026-10-04, not run).** The T. rex locomotion task revision `gait-r1` is a
commit of its own, merged just before its training session ([GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md)
§5.2 item 5; the evidence for its values is
[investigations/TREX_GAIT_R1_RESCORE_2026_10.md](investigations/TREX_GAIT_R1_RESCORE_2026_10.md)). Before it
merges, a 1.5M-step pilot runs the same values from the command line (the plan's §5.5 step 3).

**Setting up.** On Colab, run notebook section 1 only, with `REPO_REF` set to a pushed branch carrying the gait
reward kit, the walk-first gait checker and its plumbing fixes. Then mount Drive in a scratch cell (never the
storage cell) and run from `/content/mesozoic-labs`:

```bash
python environments/trex/scripts/train_sb3.py curriculum --target walk \
  --trunk-from /content/drive/MyDrive/mesozoic-labs/logs/trex/ppo/20260914_123816 \
  --seed 45 --n-envs 4 --label gait-r1-pilot \
  --output-dir /content/drive/MyDrive/mesozoic-labs/scratch/trex_gait_r1_pilot/<UTC timestamp> \
  --override locomotion.env.support_source=floor locomotion.env.gait_phase_weight=0.5 \
    locomotion.env.flight_penalty_weight=1.0 locomotion.env.foot_slip_penalty_weight=0.2 \
    locomotion.env.forward_vel_max=1.25 locomotion.env.forward_vel_weight=1.0 \
    locomotion.env.max_episode_steps=2000 locomotion.curriculum.min_avg_episode_length=1500 \
    locomotion.curriculum.collapse_peak_floor_reference=2186.8 locomotion.curriculum.timesteps=1500000 \
    locomotion.ppo.learning_rate_end=4.25e-05
```

- **The trunk is pinned.** `auto` picks the seed-44 run on the tie, and its stance hops in 6 of 40 episodes.
  The log must show `Reusing certified 'stance' from run 20260914_123816`. A `Not reusing 'stance'` line means
  interrupt at once: the CLI would train a new stance for 11M steps.
- **Every override is stage-qualified,** so the stance task, and with it the reuse, is untouched.
- **Seed 45 is new.** Seed 42 would replay the hop run's seeds. Seed 45's training, selection and replay seeds
  stay off both the certification block and the development block.
- **The run directory lies outside `logs/trex/ppo/`,** so a pilot is never a trunk.
- **`learning_rate_end` keeps the 8M schedule's first 1.5M steps.**
- **The pilot's `task_sha256` (`3ce1d7e9…`) differs from the revision's,** because the CLI casts `1.0` to the
  integer `1`. The reward arithmetic is identical, and the difference keeps a pilot verdict from ever standing
  in for the session's.
- **Wall time is estimated at about 3 h.**

**Judging.** Judge the handoff on the development block, never on 3042-3081:

```bash
python -m environments.shared.scripts.gait_report trex --stage locomotion \
  --model <run>/03_locomotion/models/robust_best_model.zip \
  --vecnorm <run>/03_locomotion/models/robust_best_model_vecnorm.pkl \
  --env-json <the "env" object of <run>/03_locomotion/task_fingerprint.json, saved as JSON> \
  --episodes 20 --seed 9000 --settle-s 1.0 --out-dir <run>/gait_dev_9000
```

The plan's §5.5 T. rex row, read on the walk-first measurement, decides what follows. The seed-42 hop's
development-seed values are in brackets.

**Proceed** if either holds:

- At least 11 of 20 episodes walk (they qualify under the provisional `biped_walk` bars, or fail only the speed
  rail, since the pilot ends inside the forward ramp), with a median speed of at least 0.5 m/s.
- The medians read as a walk, and at least 18 of 20 episodes complete:
  - `alternation_phase_offset_max` at most 0.2 cycle [0.48];
  - `off_gait_fraction` at most 0.3 [0.999];
  - `step_through_stride_fraction_min` at least 0.5 [0.35];
  - flight and unloaded time each at most 0.2 [0.35 and 0.40];
  - stride at least 0.40 `L` [0.17].

**Stop** if any holds:

- The median speed is under 0.5 m/s.
- It is still a hop: phase offset above 0.3, or flight above 0.3.
- A new exploit appears: a shuffle, a one-legged gait, a step-to, a run at the cap (flight 0.1 to 0.3 above
  1.2 m/s), or the body on the floor.
- Fewer than 16 of 20 episodes complete.

**After a proceed:**

1. Merge `gait-r1`.
2. Merge the enforcement block of [GAIT_CERTIFICATION.md](GAIT_CERTIFICATION.md) ("T. rex locomotion: the
   enforcement step").
3. Train the 8M session from the notebook with `BEHAVIOR = "walk"`, `TRUNK_FROM = "20260914_123816"` and
   `SEED = 45`, in about 13-15 h.

In between, or on a stop, bring the report back: a re-scoring comes before any weight change. Record the
pilot's run, commit, wall time and verdict here.

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
per-PR file lists, the breaks / mitigation / validation blocks, the target
architecture table and, in its status table, which PRs have landed and the net
removal still ahead; PR-8 follows the deferred cleanup (the maintainer's choice
of 2026-10-02). Sizes: S < 200 changed lines, M < 800, L < 2,000, XL above. No
remaining PR changes the on-disk format or the reuse of the canonical chain, both
r11 parents or the r13 run `20260914_123816`; `TRUNK_FROM` (default `"auto"`) and
`RETRAIN_FROM` (empty default) keep their names and defaults (PR-12 Breaks).

**Cleanup outside the sequence.** [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) orders the cleanup left
after #558 and its follow-up (#559): the backend retirement behind a frozen MJX interface core (D-D17), the rest
of D-D21's 0.3.9 gate, and the deferred PRs, which the maintainer chose on 2026-10-02 to finish before PR-8
(its §3.1 item 5). Its §2 lists the decisions, with the outcome of each one taken, and its §3.4 the cleanup each
later consolidation PR needs first. Its §5.1 records the 2026-09-25 eval-only check of the three certified
walkers: each survived the plane in every episode, but only 1 of 39 flat-heightfield episodes reached full
horizon, so no terrain pilot or terrain node should start before the heightfield-contact investigation.

The plan's status table says which PRs below have landed; it holds the rows of PR-3 .. PR-7, the PR-12 slice
and PR-14a .. PR-14c (landed by `0.3.9`), whose sections the plan's §3 keeps as pointers to `20ab100`.

| PR | One-line goal | Size (net) | Prerequisites / decision |
|---|---|---|---|
| PR-8 | One behavior env, part 2: one terrain selector (`terrain_sampler` kwarg, `terrain_contact` family), command constants imported from `command_frame`, delete `BehaviorVecNormalize` | M (about −170) | PR-7; D-D3 |
| PR-9 | Phase D through the reserved hook: `command_config` replaces the five numeric kwargs, the controller is owned by `BaseDinoEnv`, identity = task fingerprint (source-hash identity deleted) | M (about −150) | PR-7, PR-8; D-D1, D-D2; acceptance = no committed `task_sha256` moves and `plant_contract --check` clean on every species |
| PR-10 | One command-column primitive in the canonical warm-start path (`policy_loading.neutralize_command_columns` + `assert_command_blind`, called by `_create_or_load_model` on `initialize_next_stage`) | S/M (about +180) | PR-9; D-D3 |
| PR-11 | Manifest nodes: `follow_direction`, `follow_direction_difficult_terrain`, `difficult_terrain` stage TOMLs with `extends`, `[[stages]]` entries after `behavior`, trained by `train_base` | M (about +370) | PR-9, PR-10; D-D1, D-D5, G1 |
| PR-12 (rest) | Delete the parallel trainer, checkpoint module, the 66 TOMLs and their tests; `BEHAVIOR` dropdown gains `follow \| terrain` | XL (about −5,300 less the slice) | PR-11; D-D8, D-D9; `EpisodeManifestRecorder` must survive as an info key |
| PR-13 | Register the gate kind (`none/v1` for pilots, then `terrain_command/v1`) with an evidence writer in the `write_recovery_evidence` pattern; delete `behavior_certification.py` and the certificate schema | L (about −400) | PR-11, PR-12; D-D6, G4 |
| PR-15 | Docs fold (a residual pass for PR-8 .. PR-13), pin budget; the CHANGELOG half and the notebook-cell helper are done (the plan's PR-15 section) | M (about −290) | PR-14c |

---

## 5. Decisions taken 2026-09-17

The decisions of record are the D-D and G rows of [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2
(the D-A, D-B and D-C series keep their rows in §6.1), mirrored with the consolidation's rationale in
[CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md) §6 and §7; the cleanup plan's own decisions are
the rows of its §2. Each row carries its dated amendments. This section restated them until cleanup CU-17;
that text is at commit `ace8112`. The operational choices of 2026-09-20 are under
[Branch and validation rules](#branch-and-validation-rules), the one of 2026-09-26 (the stale PRs #527 and #498
closed, each with a comment) is the cleanup plan's §2 row 19, and G1..G4 are summarized under
[The final goal and the goal decisions](#the-final-goal-and-the-goal-decisions).

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
   `provenance.json` and stops at the bundle-verification cell (§8, "Verify the
  result bundle"; KNOWN_ISSUES, LOW); and
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
- PR-9 touches five species constructors and the fingerprint carve-out
  (`MJXEnvConfig` and the MJX reward kernels left with cleanup PR-B, D-D17);
  acceptance = no committed `task_sha256` moves (`test_phase_c_interface.py`)
  and `plant_contract --check` reports no interface change, on every species.
  PR-9 leaves the frozen MJX interface core untouched.
- CI: PR-3 (#546) dropped the SB3-free suites from the `test-sb3` job (the `test`
  matrix already runs them) and moved the full six-species and
  four-notebook-parameter sets to the nightly schedule and the `full-ci` label,
  keeping one real-PPO smoke per body of work on every PR; the union coverage
  gate `fail_under = 70` read 90 percent on its CI runs and is re-measured again
  after PR-12/PR-13. *2026-09-30 (cleanup CU-9, #574): with the certification
  scripts and harnesses counted, the union reads 19,626 statements, 2,405
  missed, 88 percent on #574's CI (87.75 measured locally; 91.24 before).*
  *2026-10-01 (cleanup CU-14a, #581): the `test` matrix is six jobs, three
  Pythons × {shared, species}; the plant-contract job no longer re-runs its
  tests; pull requests and pushes skip the robot's six real-training
  parametrisations of `test_compsognathus_training.py` (the cleanup plan's
  decision 10 (a)), which the nightly schedule and `full-ci` still run.*
- The SB3 notebook's Drive-mount block and `_ACTIVE_RUN_ID` memo are still
  candidates for D-D7-style moves (the JAX notebook, which carried copies of
  both, left with cleanup PR-B).
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
  with the planned plant and reward work on both species. At 4.3M the re-run
  moves by a three-legged skid that its gate would pass (section 3, session 4).
- A gait gate without reward changes (2026-09-28): today's rewards pay the
  hoppers (trex's walk reward pays forward speed and an unconditional alive
  bonus while airborne, and nothing pays alternation; the audit's §5), so the
  gait plan warns that a gait gate alone turns each retrain into an 8 h FAIL,
  and recommends pairing the gate with per-species reward revisions
  ([GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md) GQ-5, open).
- Phase B items deferred by the maintainer on 2026-09-13 stay deferred
  ([BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §10).

---

## 8. How to continue

### First steps in a new session

1. Read this file; then [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md)
   if touching code, [BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §4–§6
   for the design and decision ids, and the Phase C entry of
   [KNOWN_ISSUES.md](KNOWN_ISSUES.md) ("Training / RL") before any widen session.
2. Continue with the next PR in the order the plans set: the deferred cleanup of
   [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §3.1 item 5, then the consolidation sequence of
   [section 4](#4-consolidation-the-remaining-prs) from PR-8, on the session branch, one PR at a time,
   restarting the branch from `main` after each merge; the consolidation plan's status table says which have
   landed. A PR records the landing of the one before it only in the two places the one-landing-record rule
   names ([README.md](README.md#conventions)): that table and the CHANGELOG entry.
3. Check Drive for run directories newer than 2026-09-17 (through the Drive
   connector when the maintainer has attached one) and update
   [section 2](#2-certified-checkpoints-on-drive) here (the survey stays frozen).
4. Before any cleanup PR, read [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md): §2 for the decisions,
   with the outcome of each one taken, §3 for the PR order and §3.5 for the KNOWN_ISSUES entries each PR
   closes, §4 for the frozen MJX interface core and the retirement's acceptance checks, and §7 for the
   do-not-do list. On any change that claims to move no digest,
   run `python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --check`
   from the repository root, with the canonical MuJoCo: it compares the full
   run with the committed golden, `configs/digest_snapshot.generated.txt`,
   and names every line that moved; CI's plant-contract job runs the same
   check on every pull request (D-D22). A PR that moves a digest on purpose
   regenerates the golden with `--write` and commits it in its own diff. To
   compare a checkout older than the golden, run the harness by file path,
   with `PYTHONPATH` set to the checkout it measures, on the base and the
   head, and `diff` the two outputs (the cleanup plan's §4.4). A refactor of
   reward, info or termination code also diffs `--exact` (bit-exact: the 21
   stages and the behavior recipes) on the base and the head, on one machine:
   the golden's `reward` lines are rounded and cannot see an ulp (CU-11).
5. Before any gait work, or any training session that builds on an audited
   node (section 2) or is session 4 or 5, read
   [GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md) and its evidence,
   [investigations/GAIT_AUDIT_2026_09.md](investigations/GAIT_AUDIT_2026_09.md).
   The plan's §2 lists the decisions GQ-1..GQ-18, all open on 2026-09-28; its
   §6 puts every gait code PR after the 0.3.9 cut and, by the maintainer's
   choice of 2026-10-02, after the deferred cleanup (the cleanup plan's §3.1
   item 5); its §10 is a prompt for continuing the gait work in a fresh
   session.

### Branch and validation rules

- Automated sessions develop on the branch the session names, restart it from
  `origin/main` after each merge, commit with clear messages, push to that
  branch when the work is complete, and never push elsewhere.
- Before a push: `ruff check .` and `ruff format --check environments/`, with the
  pinned ruff (0.16.9; on the whole tree it would also format the notebooks and
  the Python blocks in Markdown files outside `environments/`, which CI never
  checks);
  `mypy environments/ --ignore-missing-imports` with SB3 and torch installed, as
  the SB3 job runs it, reporting no errors (D-D18); `pytest environments/shared/tests/`
  in chunks; `pytest environments/<species>/tests/`; the notebook checks (every
  code cell parses, as `.github/workflows/python-ci.yml` does; an edited notebook
  round-trips through `json.dump` with `indent=1`, the plan's §5 PR process); the
  SB3 job for trainer changes. The review container is a venv with SB3 2.9.0,
  torch and mujoco 3.10.0, without JAX (since cleanup PR-B only the kept `jax`
  parameter of `test_obs_functions.py` uses it, and it skips without JAX) and
  without IPython (since the notebook-only PR-12 slice deleted
  `test_behavior_notebook.py` no test needs it).
- Operational choices of 2026-09-20 (moved from section 5 by cleanup CU-17): the
  full six-species and four-notebook-parameter SB3 sets run nightly and under
  the `full-ci` label (PR-3); if a PR run drops the union coverage gate below
  70, the measured number and a proposed floor are reported rather than the
  floor lowered; the session results (`gate_verdict.json`, `widen_report.json`,
  `stance_gate_report.json`) are read from Drive through the maintainer's Drive
  connector when a session ends.
- Never write an AI model or vendor name into repository files; cite PR numbers,
  branch names or "the maintainer".

### Doc conventions (from [README.md](README.md))

The conventions are kept once, in [README.md](README.md#conventions): where a new
document goes, how a dated one is corrected, how decision ids are used, links,
the root README's scope and the one-landing-record rule.

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
| Remaining cleanup, the backend retirement and its frozen core, open cleanup decisions | [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) |
| Gait audit of the certified nodes (2026-09-28), the gait-quality plan and its open decisions GQ-1..GQ-18 | [investigations/GAIT_AUDIT_2026_09.md](investigations/GAIT_AUDIT_2026_09.md), its evidence files in [investigations/gait_2026_09/](investigations/gait_2026_09/README.md); [GAIT_QUALITY_PLAN_2026_09.md](GAIT_QUALITY_PLAN_2026_09.md) |
| Digest snapshot (every plant, policy, stage, recovery and behavior digest and the 21 stages' reward, info and termination captures, one value per line; `--exact` for a same-machine A/B) | `environments/shared/harnesses/digest_snapshot.py`; its committed golden, which CI checks, `configs/digest_snapshot.generated.txt` |
| Drive state as surveyed 2026-09-17 | [investigations/DRIVE_RUN_SURVEY_2026_09.md](investigations/DRIVE_RUN_SURVEY_2026_09.md) |
| Bundle layout, `gate_verdict.json`, `ancestors/` records | [RESULT_BUNDLES.md](RESULT_BUNDLES.md) |
| Plant identities and the widen contract | [PLANT_CONTRACT.md](PLANT_CONTRACT.md), `configs/plant_versions.toml` |
| Reuse rules 1–7, trunk selection, widen and backfill tools | `environments/shared/ancestors.py` (`find_certified_ancestor`, `select_trunk`); `environments/shared/scripts/widen_checkpoint.py`, `environments/shared/scripts/backfill_gate_verdict.py` |
| Gate digests and task fingerprints | `environments/shared/curriculum/gate_schema.py` (`gate_config_view`, `gate_config_sha256`), `environments/shared/task_fingerprint.py` (`stage_task_fingerprint`, the stage-level helper every derivation site calls, and `FINGERPRINT_BACKEND`; `derive_stage_task_fingerprint` beneath it) |
| Policy loading: SB3 archive loads, the VecNormalize sidecar lookup and the SB3 import helper | `environments/shared/policy_loading.py` (`load_sb3_model`, `load_sb3_checkpoint`, `resolve_vecnorm_path`) |
| One `stage_config.json` reader, the repository root, the sha256 pattern and the record field validators | `environments/shared/config.py` (`read_recorded_stage_config`, over `file_io.read_json_object`); `environments/shared/paths.py` (`REPOSITORY_ROOT`; `plant_contract.constants` binds it and is the patch point); `environments/shared/record_fields.py` (`SHA256_DIGEST_PATTERN`, `is_sha256_digest`, the validators) |
| The notebook, its pins, the changelog | [notebooks/sb3_training.ipynb](../notebooks/sb3_training.ipynb); `environments/shared/tests/test_sb3_notebook_pins.py`; [CHANGELOG.md](../CHANGELOG.md) |
