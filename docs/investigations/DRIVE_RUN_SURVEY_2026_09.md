# Google Drive run survey - what is certified under the Phase C interface (2026-09-17)

**Date**: 2026-09-17 (the survey; written into the repository by the 2026-09-19
documentation pass). **Checkout**: `main` = `22c1fc8` (2026-09-16). **Drive
layout**: `mesozoic-labs/logs/<species>/<algo>/<run>/`; every run below is
`ppo`. **Purpose**: state, per species, which run directories on the
maintainer's Drive hold a node that reuse rules 1–7 accept under the Phase C
interface, which pre-Phase-C runs are widen candidates, which are history, and
which training sessions follow. **Companions**: [NEXT_STEPS.md](../NEXT_STEPS.md)
(the living run plan, updated in place as Drive changes),
[BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) (the design of record:
§4.6 Phase C, §5 the phase table with the C½ walker, §6 the decision series),
[TREX_STANCE_WIDENED_INTERFACE_2026_09.md](TREX_STANCE_WIDENED_INTERFACE_2026_09.md)
(the template note whose sessions §5 below re-scopes) and the Phase C entry of
[KNOWN_ISSUES.md](../KNOWN_ISSUES.md) ("Training / RL").

Point-in-time record, frozen at its date. Runs that appear on Drive after
2026-09-17 go into NEXT_STEPS.md §2; corrections to this note are appended,
never rewritten.

---

## 1. Method

Per run directory, read in this order:

1. Run level: `provenance.json` and `summary.json` / `training_summary.txt` for
   species, seed, commit, the plant identity (`policy_interface_revision`,
   `physics_sha256`, `observation_dim`), the deliverables block (`certified`,
   `provisional`, replication count) and stage durations. Run-level records lag
   a run continued in a later Colab session (finding 2), so they set the frame
   and decide nothing.
2. Per stage directory (`01_stance`, `03_locomotion`, ... for runs since
   2026-08-20; `stage1` / `stage2` / `stage3` on legacy runs; readers accept
   both through `stage_dir_candidates`): `gate_verdict.json` (`passed`,
   `judged_by`, `task_sha256`, `gate_sha256`, the handoff pair's digests),
   `stance_gate_report.txt` (the 40-episode panel: reward, full-horizon
   fraction, mean unsupported duty and its UCB, panel seeds),
   `task_fingerprint.json`, `plant_identity.json`, `stage_config.json` (the
   `run` block: `seed`, `n_envs`, `timesteps`) and the `models/` handoff pair.
3. Offline reuse check, against the checkout rather than in Colab: the node's
   `task_sha256` is compared with `derive_stage_task_fingerprint(...)`
   (`environments/shared/task_fingerprint.py`) evaluated for the current stage
   config and plant, and its `gate_sha256` with
   `gate_config_sha256(gate_config_view(<stage TOML [curriculum] block>))`
   (`environments/shared/curriculum/gate_schema.py`). Equality on both is what
   reuse rules 3 and 7 test (`environments/shared/ancestors.py`); rule 6 then
   validates the checkpoint's embedded plant identity (the archive's identity
   attribute) against the current plant, with no legacy allowance
   (`plant_identity.json` is one of the records copied beside `ancestor.json`
   and is what `select_trunk`'s older-interface scan reads; rule 6 never reads
   it). A node that passes is what `TRUNK_FROM = "auto"` (`select_trunk`,
   decision D-A25) picks up.
4. Widen eligibility of a pre-Phase-C parent, offline against
   `widen_checkpoint.identity_gate_errors` (`environments/shared/scripts/widen_checkpoint.py`):
   same species, same `physics_sha256`, same `nq` / `nv` / `nu`, same
   `action_dim`, parent `observation_dim + 3 == current`, and
   `1 <= current − parent policy_interface_revision <= max_revision_gap`
   (default 1; the notebook's `WIDEN_MAX_REVISION_GAP`). The parent stage
   directory must hold `stage_config.json` with a run block and a stamped
   VecNormalize sidecar; a parent `gate_verdict.json` is optional and, if
   present, must record `passed = true`.

Nothing on Drive was modified. The `mesozoic-labs/certified` library directory
(#541) was not surveyed: nothing relies on it now (`PUBLISH_CERTIFIED = False`
since #542; #543 took the library out of canonical trunk selection).

Plant identities at `22c1fc8` (revisions from `configs/plant_versions.toml`;
the dimensions from `current_plant_identity` and the generated
`configs/plant_manifest.generated.json`), the "current" side of every
comparison below:

| species | policy interface | physics | obs dim | action dim | nq / nv / nu |
|---|---|---|---|---|---|
| trex | r13 | r7 | 64 | 15 | 28 / 27 / 15 |
| compsognathus | r2 | r1 | 56 | 14 | 24 / 23 / 14 |
| velociraptor | r10 | r2 | 70 | 22 | 31 / 30 / 22 |
| brachiosaurus | r8 | r4 | 86 | 30 | 38 / 37 / 30 |
| dibothrosuchus | r7 | r1 | 80 | 27 | 35 / 34 / 27 |
| compsognathus_robot | r2 | r1 | 46 | 12 | 19 / 18 / 12 |

## 2. Per-species findings

### 2.1 Tyrannosaurus rex (`logs/trex/ppo/`)

| Run | Started | Seed | Interface | Layout | Stages present | Verdicts | What it means |
|---|---|---|---|---|---|---|---|
| `20260810_145546` | 2026-08-10 | 42 | r11 | legacy `stage1` / `stage2` / `stage3` | `stage1` (stance) plus `stage2`, `stage3`; only `stage1` surveyed | stance panel PASS on the 2026-08 records ([TREX_STAGE1_GATE_PASS_RUN_2026_08.md](TREX_STAGE1_GATE_PASS_RUN_2026_08.md)); no `gate_verdict.json` in `stage1` (verdict only in run-level records) | The certified parent the template note's Session 1 would widen. Superseded: seed 42 is certified at r13 by `20260914_123816`. History |
| `20260815_205206` | 2026-08-15 | 44 | r11 | legacy `stage1` + `stage2` | `stage1` (stance) plus `stage2`; only `stage1` surveyed | `stance_gate_report.txt`: reward 3408.3 ± 88.5, full-horizon 1.0000, mean unsupported duty 0.0069, duty UCB 0.0117, 40 episodes seeds 3042–3081, PASS; no `gate_verdict.json`; `stage1/` holds `stage_config.json` (run block) and `models/` | Widen-eligible under `WIDEN_MAX_REVISION_GAP = 2`, `SEED = 44` (session 1 of §4) |
| `20260914_123816` | 2026-09-14 | 42 | r13 (obs 64), commit `35dd44c` (the merged Phase C tree, before the pilots) | `01_stance` + `03_locomotion` (no recovery) | stance, locomotion | both `gate_verdict.json` PASS (below) | **The certified r13 walker.** `TRUNK_FROM = "auto"` reuses both nodes |
| `20260915_160239` | 2026-09-15 | 43 | r13, commit `35dd44c` | `01_stance` only | stance | stance panel FAIL on unsupported duty (below) | A measured deficit; history |

The certified run, node by node:

| node | steps / wall clock | evaluation | verdict |
|---|---|---|---|
| `01_stance` | 11,001,856 / 13h13m | final eval 3460.6 ± 19.2 | `gate_verdict.json` PASS, judged 2026-09-15 01:57:37 UTC by `generate_stage_artifacts`, `gate_sha256 ff2494ba…`, `task_sha256 82528a2e…`, handoff `robust_best_model.zip` |
| `03_locomotion` | 8,011,776 / 8h46m | mean forward velocity 1.07 m/s, mean episode length 1000, mean reward 1940.8 | `gate_verdict.json` PASS, judged 2026-09-15 11:39:57 UTC, `gate_sha256 02602cb0…`, `task_sha256 31383192…` |

Verified at `22c1fc8` by the offline check of §1 step 3: both nodes'
`task_sha256` and `gate_sha256` equal the digests derived from the current
`configs/trex/stance.toml` and `configs/trex/locomotion.toml`, so rules 3 and 7
pass and `select_trunk` picks this run for any trex session, reusing both
nodes. Its run-level `summary.json`, `provenance.json` and
`artifact_manifest.json` were written 2026-09-15 02:08 UTC, after the stance,
and never refreshed after the locomotion verdict at 11:39; `training_summary.txt`
was refreshed (finding 2).

The seed-43 replicate `20260915_160239`: stance trained 11M steps (13h17m),
final eval 3209.7 ± 284.3, best eval 3335.1, mean episode length 989.6 ± 56.2;
the 40-episode panel failed `stance_quality/v1` on unsupported duty (mean
0.0323, UCB 0.0350, both above the 0.02 rail) while reward and full-horizon
fraction passed. `provenance.json` deliverables: `certified` false,
`provisional` true, replication count 1. The same seed also failed the r11
configuration in August (run `20260815_014118`), there on every stance
criterion (full-horizon 0.9250, duty 0.0597 / UCB 0.0747;
[TREX_STAGE1_SEED43_REPLICATE_2026_08.md](TREX_STAGE1_SEED43_REPLICATE_2026_08.md)
§1); the r13 failure is narrower, on the duty rail alone.

Trex stance seed inventory at r13 (`configs/trex/stance.toml` is the only stage
declaring `certification_seeds = 2`; every other stage defaults to 1):

| seed | r13 status | source |
|---|---|---|
| 42 | certified | `20260914_123816` |
| 43 | failed (duty) | `20260915_160239` |
| 44 | not at r13; exists only as the r11 parent | `20260815_205206`, widen candidate |

The bar of two is met when a second seed certifies at r13. Widening seed 44 is
the cheapest route (re-panel about 1 h); a fresh seed-45 stance is the fallback
(about 13 h, the measured stance wall clock).

### 2.2 Compsognathus (`logs/compsognathus/ppo/`)

| Run | Started | Seed | Interface | Layout | Stages present | Verdicts | What it means |
|---|---|---|---|---|---|---|---|
| `20260909_162812` | 2026-09-09 16:29 UTC (stance finished 2026-09-10 06:05 UTC, about 13.5 h) | 42 | r1 (obs 53, pre-Phase-C), commit `9557e97` | `01_stance` only | stance | `stance_gate_report.txt`: reward 2801.6 ± 51.2 (rail 1800), full-horizon 1.0000, mean unsupported duty 0.0131, duty UCB 0.0141, 40 episodes seeds 3042–3081, GATE: PASS; no `gate_verdict.json` | Widen-eligible under the default gap 1, `SEED = 42` (session 2 of §4) |

Identity gate, offline: `physics_sha256 08a5fbf7…` is unchanged at r2 (physics
revision 1); `stage_config.json` carries the run block (seed 42, `n_envs` 4,
`timesteps` 11,000,000); 53 + 3 = 56 = the current width; revision gap r1 → r2
= 1, inside the default bound. Every field passes.

### 2.3 Velociraptor (`logs/velociraptor/ppo/`)

| Run | Started | Seed | Interface | Layout | Stages present | Verdicts | What it means |
|---|---|---|---|---|---|---|---|
| `20260723_005740` (the latest run) | 2026-07-23 | 42 | r3 (obs 67); physics r2, `physics_sha256 c10e6102…` = current | legacy `stage1` / `stage2` / `stage3` | stage1 balance 6M (3h41m, final eval 1767, 1000-step episodes), stage2 locomotion 8M (4h50m, 3.29 m/s), stage3 strike 12M | run-level `publication_gate_passed` true for all three; no per-node verdicts; the VecNormalize sidecar predates identity stamping | Not a widen candidate; fresh chain (session 3 of §4) |

Identity gate, field by field: species passes; `physics_sha256` passes (the
digest is the current one, so `nq` / `nv` / `nu` 31 / 30 / 22 and `action_dim`
22 are the same plant's); `observation_dim` passes (67 + 3 = 70). The revision
gap fails: r3 → r10 is 7 revisions, refused under the default bound, and a
bound of 7 would assert that every crossed bump was fingerprint-only, which
`configs/plant_versions.toml` note 6 contradicts (the reset now settles the
animal on the ground; "Every existing checkpoint is trained against a different
task"). Separately, the July VecNormalize sidecar predates identity stamping:
`widen_checkpoint` checks the sidecar's stamp after the identity gate
(`validate_recorded_identity`, which refuses a missing stamp without the
legacy-plant allowance the notebook's widen cell never passes), so the parent
is refused either way — on the revision gap at the gate, or on the sidecar
after it. Decision: fresh chain. A fresh r10 stance is also
the first data since the pending Commit B validation of
[VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md](VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md).

### 2.4 Brachiosaurus (`logs/brachiosaurus/ppo/`)

| Run | Started | Seed | Interface | Layout | Stages present | Verdicts | What it means |
|---|---|---|---|---|---|---|---|
| `20260717_162659` (the latest run) | 2026-07-17 | 42 | r1; physics r1 (current r4) | legacy `stage1` / `stage2` / `stage3` | stage1 balance 6M (4h24m, 1739.8), stage2 locomotion 16M (9h48m, 1.42 m/s), stage3 food_reach 12M | run-level only; no per-node verdicts | Widen refused by construction; fresh chain (session 5 of §4) |

Identity gate: `physics_sha256` differs (physics r1 against the current r4;
among the crossed changes, note 6 excluded a permanent torso / tail
self-collision), and that field is
checked whatever the revision bound, so the parent is refused before the r1 →
r8 gap (7 revisions) is even weighed. Decision: fresh chain, split in two
sessions (stance 6M, then locomotion 16M through auto-trunk).

### 2.5 Dibothrosuchus and compsognathus_robot

No run directories on Drive for either species. Both start fresh chains
(sessions 4 and 6 of §4).

## 3. Findings

1. **The certified r13 walker exists.** `20260914_123816` holds a certified
   stance and a certified locomotion node whose `task_sha256` / `gate_sha256`
   match the current stage TOMLs at `22c1fc8`; `TRUNK_FROM = "auto"` selects it
   for every trex session. It is the deliverable the plan's Phase C½ asked for,
   obtained from a from-scratch r13 stance rather than a widened r11 one, so
   the plan's §7 risk "No certified walking parent exists yet" is discharged.
2. **The run-level records of `20260914_123816` are stale.** `summary.json`,
   `provenance.json` and `artifact_manifest.json` describe the run as of the
   stance (02:08 UTC) and never learned of the locomotion verdict (11:39 UTC);
   only `training_summary.txt` was refreshed. Reuse reads per-node files, so
   nothing breaks, but the run-level records under-report the run until the
   bundle cell is re-run in a later session. Cosmetic.
3. **Seed 43 fails the duty rail at r13.** `20260915_160239` passed reward and
   full-horizon fraction and failed unsupported duty (mean 0.0323, UCB 0.0350
   against 0.02). A failed panel is a measured deficit of the policy, not panel
   noise: re-rolling the panel does not help, another seed does.
4. **The r11 parents lack `gate_verdict.json`, and widening does not need it.**
   `widen_checkpoint` reads the parent's `stage_config.json` run block, its
   `models/` handoff pair and the stamped sidecar; a verdict is optional (must be
   `passed = true` if present). The widened node gets no verdict, so the chain
   loop JUDGES it in the new run (40-episode stance panel, seeds 3042–3081).
   Backfilling the r11 verdicts (`backfill_gate_verdict.py`) is not needed
   before widening and does nothing for reuse: rules 3 and 6 refuse every
   pre-Phase-C archive.
5. **The July velociraptor and brachiosaurus runs are fresh starts.**
   Velociraptor fails the revision-gap condition for a substantive reason
   (note 6); brachiosaurus fails the physics digest. Neither is an r10 / r8
   parent. Checking the identity gate offline (§1 step 4) before booking Colab
   time is the cheap step.
6. **No runs exist for dibothrosuchus and compsognathus_robot.** Their first
   chains are fresh.

## 4. Recommended sessions (2026-09-17)

All on `main`, `notebooks/sb3_training.ipynb`, notebook defaults unless stated
(`N_ENVS = 4`, `TRUNK_FROM = "auto"`, `PUBLISH_CERTIFIED = False`).

| # | Species | Settings | What happens | Rough time |
|---|---|---|---|---|
| 1 | trex | BEHAVIOR="stand", WIDEN_FROM="20260815_205206", WIDEN_MAX_REVISION_GAP=2, SEED=44 | widens the seed-44 r11 stance to r13, re-panels it (40 episodes), then trains recovery 3M | panel ~1 h, recovery ~3.5 h |
| 2 | compsognathus | BEHAVIOR="walk", WIDEN_FROM="20260909_162812", WIDEN_MAX_REVISION_GAP=1, SEED=42 | widens the r1 stance to r2, re-panels it, trains locomotion 3M | panel ~1 h, walk ~4 h |
| 3 | velociraptor | BEHAVIOR="walk", SEED=42 | fresh stance 6M then locomotion 8M | ~4 h + ~5 h |
| 4 | dibothrosuchus | BEHAVIOR="walk", SEED=42 | fresh stance 6M then locomotion 12M | ~4 h + ~7 h |
| 5 | brachiosaurus | BEHAVIOR="stand" then, in a second session, BEHAVIOR="walk" | stance 6M; the walk session reuses the certified stance through auto-trunk and trains locomotion 16M | ~4.5 h then ~10 h |
| 6 | compsognathus_robot | BEHAVIOR="walk", SEED=42 | fresh stance 11M then locomotion 3M | ~13 h + ~4 h |

Notes: sessions 1 and 2 must set `SEED` to the parent's seed before the storage
cell mints `RUN_ID` (the widen cell refuses any other value, D-C14); leave
`TRUNK_FROM = "auto"`; `PUBLISH_CERTIFIED` stays `False`. With `WIDEN_FROM` set
the chain loop uses no trunk, so every node below the widened root trains in
that run: trex `stand` = widen + re-panel, then recovery 3M (`walk` would train
locomotion 8M instead); for velociraptor, brachiosaurus and dibothrosuchus
`stand` is stance only. If the widened seed-44 panel fails the duty rail as
seed 43 did, the fallback is a fresh trex stance with `SEED = 45`. Any chain can
be split across Colab sessions because a later `walk` session reuses a certified
stance automatically. Not recommended yet: direction / terrain pilots
(evaluation only, D-D9; one trex `follow_direction` pilot pointed at the
September walker's locomotion checkpoint pair is harmless but disposable) and
trex hunting (off the direction path). Times are the measured Colab wall clock
of the runs above or scaled from them. The living copy of this table, with the
session status, is [NEXT_STEPS.md](../NEXT_STEPS.md) §3.

## 5. What this changes for the template note

[TREX_STANCE_WIDENED_INTERFACE_2026_09.md](TREX_STANCE_WIDENED_INTERFACE_2026_09.md)
(2026-09-14) planned five sessions against the two r11 parents. None has run,
and the survey re-scopes them:

| Template session | Status after this survey |
|---|---|
| 1 — seed-42 widen of `20260810_145546` | **Superseded**: seed 42 is certified at r13 by the fresh run `20260914_123816`; a widened copy would add nothing to the seed inventory |
| 1b — recovery freeze re-roll | **Covered** by session 1 of §4: the chain loop freezes the recovery resolution from the widened seed-44 handoff before training recovery in the same run |
| 1 repeat — seed-44 widen of `20260815_205206` | **Still owed**; it is session 1 of §4, now with `BEHAVIOR = "stand"` instead of `"stance"` so recovery trains from the widened handoff in the same run |
| 2 — the C½ walker | **Superseded** by the certified locomotion of `20260914_123816` (1.07 m/s, mean length 1000 under `reward_and_length/v1`) |
| 3 — stand | **Covered** by session 1 of §4 |

After session 1 runs, fill the template's §1–§3 seed-44 columns from the
shipped `widen_report.json`, `gate_verdict.json` and `stance_gate_report.json`;
the seed-42 columns stay empty with a pointer to `20260914_123816`. The
template's §5 inventory of pre-bump directories is only partly filled by §2.1
above (the survey read the four trex runs there: `20260810_145546` and
`20260815_205206` stay on the log tree as history, the second one as the widen
parent); the r12 stance PASSes of 2026-08-16..18, the August recovery runs and
the interrupted 2026-08-21 locomotion leg were not read, so whoever runs
session 1 fills that inventory from the log-tree listing. The template's
session table is not rewritten; its own §6 (appended 2026-09-19) records the
supersessions.
