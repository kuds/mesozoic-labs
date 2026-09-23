# T-Rex Stance Widened to the Phase C Interface — 2026-09 (template)

**Status: TEMPLATE — pending the maintainer's Colab sessions (BEHAVIOR_RECIPES_PLAN
§4.6 Phase C, WS-C4).** See section 6 (appended 2026-09-19) for which
sessions are superseded. Every `<to be filled after Session N>` below is a
placeholder; the commands, knobs and seeds are exact for the Phase C tree
and are the only parts written in advance. Fill each section from the run's
shipped artifacts, never from memory, and delete this status line when the
last placeholder is gone. **Sessions 1 and the seed-44 repeat set
`WIDEN_MAX_REVISION_GAP = 2` (decision D-C17, 2026-09-14):** both certified
parents are policy-interface r11 archives (trex r11 → r12 landed 2026-08-16
after they trained, a fingerprint-only bump: `observation_dim` 61 on both
sides, physics r7 and `action_dim` 15 unchanged), two revisions behind this
checkout's r13, and `widen_checkpoint`'s default bound of one revision
refuses them with `policy_interface_revision: parent=11, current=13 (gap 2
exceeds max_revision_gap=1; pass --max-revision-gap 2 / max_revision_gap=2
…)`. Under the bound of 2 the gate still checks the physics digest, `nq` /
`nv` / `nu`, `action_dim` and the widths; setting it asserts, from
`configs/plant_versions.toml`'s notes 11 and 12, that the crossed bump
changed nothing the widening cannot bridge (KNOWN_ISSUES, the Phase C
entry). The sessions are runnable as written, pending the maintainer's Colab
time. Quote run facts (`seed`, `n_envs`,
`timesteps`) from the widened stage's `stage_config.json` run block, which
keeps the parent's; the session's `provenance.json` / summary record this
session's `N_ENVS`, which only the nodes trained here used.

**Run**: `<to be filled after Session 1: RUN_ID, e.g. 202609DD_HHMMSS>` (Colab, SB3
PPO, 4 envs, seed 42 — the widened copy of `20260810_145546`; no training
steps of its own, `num_timesteps` inherited: 10,002,432) and
`20260920_010912` (seed 44 — the widened copy
of `20260815_205206`; both parents were trained under policy interface r11,
`sha256:96ef13…`, and are widened under `WIDEN_MAX_REVISION_GAP = 2` —
`revision_gap` 2 in each `widen_report.json`). **Plant**: physics r7 (`sha256:72c662…`), policy
interface r13 (`sha256:0b43de…` — the Phase C revision: the 3-dim
body-relative command segment appended LAST, 61 → 64 dims, `command_mode =
"none"`), visual r4. **Config**: `configs/trex/stance.toml` unchanged
(`stance_quality/v1`; the widened run block carries the parent's `seed` /
`n_envs` / `timesteps` and the eight `WIDEN_LINEAGE_KEYS`, no
`LOAD_LINEAGE_KEYS`). **Purpose**: record what the Phase C interface bump did
to the certified stance — the widen itself (`widen_checkpoint`, decisions
D-C8–D-C12), the re-panel of both certified parents under the new task hash
(D-C14), one recovery freeze re-rolled from the widened handoff, and the C½
walker — and state, for every pre-bump verdict on the log tree, what the
remedy is.

This is a point-in-time record. Write it from the runs' shipped
`widen_report.json`, `gate_verdict.json`, `stance_gate_report.json`,
`gate_resolution.json`, summaries and bundle manifests; not rewritten —
corrections are appended.

---

## 0. The sessions (knobs and commands, exact for the Phase C tree)

All sessions run `notebooks/sb3_training.ipynb` on the merged Phase C tree
(PR-C1 + PR-C2 + PR-C3) with `SPECIES = "Tyrannosaurus Rex"`, `ALGORITHM =
"ppo"`, `N_ENVS = 4`. The session order is the maintainer's priority
(D-C15): stance → recovery → walking before hunting.

| Session | Configuration cell | What runs | Wall clock |
|---|---|---|---|
| 1 — widen + re-panel, seed 42 (the parent is an r11 archive, two revisions behind r13: `WIDEN_MAX_REVISION_GAP = 2`, D-C17) | `SEED = 42`, `BEHAVIOR = "stance"`, `WIDEN_FROM = "20260810_145546"`, `WIDEN_MAX_REVISION_GAP = 2`, `TRUNK_FROM = ""`, `RETRAIN_FROM = ""`, `RUN_LABEL = "phase-c widened stance seed42"` | Run all. The storage cell mints `RUN_ID` and an r13 `provenance.json` (`certification_panel` role 3042). The widen cell prints `policy-interface revisions crossed: 2 (WIDEN_MAX_REVISION_GAP=2)` (under the default 1 it halts instead with the tool's refusal naming the gap and the flag) and writes `<RUN_DIR>/01_stance/{models/robust_best_model.zip, models/robust_best_model_vecnorm.pkl, models/stage1_final.zip, models/stage1_final_vecnorm.pkl, stage_config.json, plant_identity.json, task_fingerprint.json, widen_report.json}` — no verdict. The chain loop refuses to reuse the verdict-less directory (printed reason), finds the `stage1_final.*` pair and takes JUDGE: `evaluate_stage_checkpoints` (30-episode evidence CSVs) then `generate_stage_artifacts` (the 40-episode stance panel, seeds 3042–3081, `stance_gate_report.json`, `gate_verdict.json` with `judged_by = "generate_stage_artifacts"`), the summary, a `complete` or `failed` bundle, and the verdict is enforced. | ~1–2 h |
| 1b — recovery freeze re-roll (same runtime, terminal) | — | `python -m environments.shared.harnesses.freeze_recovery_gate --stage-dir <RUN_DIR>/02_recovery --species trex --stage recovery --policy-zip <RUN_DIR>/01_stance/models/robust_best_model.zip --vecnorm <RUN_DIR>/01_stance/models/robust_best_model_vecnorm.pkl` — statue null plus the brace null held at the widened stance's post-settle mean action, frozen under the recovery stage's Phase C task hash and read back through `require_gate_resolution`. (T-Rex keeps the historical P3 judge: `_species_calibration` returns `None` for trex, so the harness loads the brace checkpoint through `PPO.load` without the calibrated species' `_validate_checkpoint_source` step; the widened archive loads because its saved observation space and identity are the current plant's.) This stand-alone freeze is the §3 record; Session 3 runs in a new run directory and freezes its own copy from the same handoff. | ~20 min |
| 1 repeat — seed 44 (an r11 archive too: `WIDEN_MAX_REVISION_GAP = 2`) | `SEED = 44`, `BEHAVIOR = "stance"`, `WIDEN_FROM = "20260815_205206"`, `WIDEN_MAX_REVISION_GAP = 2`, `RUN_LABEL = "phase-c widened stance seed44"` | As Session 1, in a NEW `RUN_ID` and a fresh runtime. Set `SEED = 44` BEFORE the storage cell runs: the widen cell refuses any other value (D-C14), and by then the storage cell has minted `RUN_DIR` with `training_seed = SEED` and will not re-mint it under another seed — after a refusal, correct `SEED`, restart the runtime (or `del _ACTIVE_RUN_ID`) and delete the stray directory (it holds only `provenance.json`). Re-run the seed-42 run's publication cell afterwards with the sibling present so its bundle counts the replicate: trex stance then reads `2 runs of 2 seeds` instead of `1 run of 2 seeds; provisional`. | ~1–2 h |
| 2 — Phase C½ walker | `SEED = 42`, `BEHAVIOR = "walk"`, `TRUNK_FROM = "<Session-1 RUN_ID>"`, `WIDEN_FROM = ""` | Run all. Stance is reused from the trunk under reuse rules 1–7 (rule 3 passes because the Session-1 verdict's `task_sha256` was minted under the new plant; rule 6 validates the widened archive's re-stamped identity), locomotion trains its 8M budget under `initialize_next_stage` from the widened handoff (stage-entry shaping fires), and `reward_and_length/v1` judges it (≥ 1.0 m/s, ≥ 750 steps). **If the Session-1 re-panel FAILED**: `RETRAIN_FROM = "stance"` with seed replicates instead — the §4.6 fallback; the widened checkpoint is then not certified. | ~9 h |
| 3 — stand (optional for Phase C; on the stance → recovery path) | `SEED = 42`, `BEHAVIOR = "stand"`, `TRUNK_FROM = "<Session-1 RUN_ID>"`, `WIDEN_FROM = ""` | A new `RUN_ID` (the Session-1 bundle is `complete` and is not re-targeted in place). The chain reuses stance from the trunk (rules 1–7, as in Session 2); `<RUN_DIR>/02_recovery` holds no `gate_resolution.json`, so the chain loop freezes one from the reused stance handoff before training (statue + brace nulls — the same handoff digests as Session 1b, so the null rates should reproduce §3), trains recovery (3M) under the new interface, then rolls the policy panel on the frozen seeds and judges `recovery_quality/v1`. | ~3.5 h |

Do NOT bring a widened checkpoint in by setting `RUN_ID` to the old run: the
storage cell's `initialize_result_bundle` refuses the old directory outright
(`run directory already belongs to a different run: {'plant_identity': …}`,
the recorded identity being r11 against this checkout's r13), and with the
old `provenance.json` removed the JUDGE branch's `validate_model_plant`
refuses the archive and the audit rejects the r13 provenance against the old
run's stage configs (KNOWN_ISSUES, the Phase C entry).

## 1. The widen

`<to be filled after Session 1 and the seed-44 repeat>`

Record, per parent, from `<RUN_DIR>/01_stance/widen_report.json` and the
parent's `gate_verdict.json`:

| field | seed-42 parent `20260810_145546` | seed-44 parent `20260815_205206` |
|---|---|---|
| widened run id | superseded (seed 42 is certified at r13 by `20260914_123816`) | `20260920_010912` (2026-09-20 01:09 UTC, Colab L4, Python 3.13.15) |
| parent handoff (`handoff_name`) | — | `robust_best_model` |
| parent `checkpoint_sha256` / `normalization_sha256` | — | `ca1a17a7…` / `51838dde…` |
| parent `task_sha256` (r11) → widened `task_sha256` (r13) | — | `null` (the r11 `stage_config.json` predates task fingerprints) → `82528a2e…` (= the seed-42 r13 stance digest) |
| widened `checkpoint_sha256` / `normalization_sha256` | — | `7f4284ad…` / `963ac712…` |
| `from_observation_dim` → `to_observation_dim` | 61 → 64 (expected) | 61 → 64 |
| `revision_gap` / `max_revision_gap` (r11 → r13 under `WIDEN_MAX_REVISION_GAP = 2`) | 2 / 2 (expected) | 2 / 2 |
| run block `widened_from_policy_interface_revision` (`stage_config.json`) | 11 (expected) | 11 (`widened_from_run_id 20260815_205206`; run block seed 44, n_envs 4, timesteps 10,000,000, `hyperparameters_sha256 8c8b9ab3…`) |
| `padded_tensors` | `mlp_extractor.policy_net.0.weight`, `mlp_extractor.value_net.0.weight` (expected; columns 61–63) | as expected: both tensors, columns 61, 62, 63 |
| `optimizer_members_padded` | `policy.optimizer` (expected) | `policy.optimizer` (states 1 and 5) |
| `max_padded_column_abs` (pin: exactly 0) | — | 0.0 (`padded_columns_exactly_zero true`, `hashes_stable_after_verification true`) |
| `max_action_delta_zero_command` / `max_action_delta_probe_command` (tolerance 1e-6, seed 3042, 200 steps) | — | 0.0 / 0.0 |
| `num_timesteps` inherited | 10,002,432 (expected) | 10,000,000 (the parent's run block records 10,000,000, not the 10,002,432 this template expected) |
| `widened_by` (tool@commit) | — | `widen_checkpoint/v1@ac409f8`; `schedule_members_restated` `learning_rate` and `lr_schedule` = `LinearSchedule(3e-05, 1e-05)` from `parent_stage_config` (the parent's `clip_range` is a constant); torch 2.11.0+cu128, stable-baselines3 2.9.0, gymnasium 1.3.0, mujoco 3.10.0 |

## 2. The re-panel: widened stance vs its pre-bump certificate

`<to be filled after Session 1 and the seed-44 repeat>`

The same 40-episode panel (seeds 3042–3081, settle 200) under
`stance_quality/v1`, judged by `generate_stage_artifacts`. Pre-bump columns
from the parents' records (`TREX_STAGE1_GATE_PASS_RUN_2026_08.md`,
KNOWN_ISSUES update 3); the widened columns from each run's
`stance_gate_report.json` / `gate_verdict.json`.

| criterion | seed 42 pre-bump (r11 certificate) | seed 42 widened (r13) | seed 44 pre-bump | seed 44 widened (r13) | required |
|---|---|---|---|---|---|
| full-horizon fraction | 1.0000 (40/40) | — (superseded) | 1.0000 (40/40) | 1.0000 (40/40) | ≥ 0.95 |
| mean unsupported duty | 0.0048 | — | 0.0069 | 0.0069 | ≤ 0.02 |
| duty UCB (95%) | 0.0080 | — | 0.0117 | 0.0117 | ≤ 0.02 |
| panel reward | 3368.7 ± 46.7 | — | 3408.3 ± 88.5 | 3408.3 ± 88.5 | ≥ 2100 (rail) |
| bilateral support | 0.9938 | — | 0.9904 (single support 0.0027) | 0.9904 (single support 0.0027) | (statue 0.998, not gated) |
| verdict | PASS | — | PASS | **PASS** (`gate_verdict.json` judged 2026-09-20 01:15:02 UTC by `reporting.stage_artifacts.generate_stage_artifacts`; `gate_sha256 ff2494ba…`; `task_sha256 82528a2e…` = the widened digest of §1; final eval 3418.22 ± 87.88 over 30 episodes) | — |

Acceptance for this section: `gate_verdict.json` with `passed = true`,
`judged_by = "generate_stage_artifacts"` and `task_sha256` equal to the
widened `task_sha256` of §1; a `complete` stance bundle. State the
replication label the seed-42 bundle publishes after the seed-44 sibling is
counted (`<1 run of 2 seeds; provisional / 2 runs of 2 seeds>`). If a
re-panel failed, say so here and record the retrain decision in §4.

## 3. The recovery freeze re-roll

Filled 2026-09-23 from `20260920_010912/02_recovery/gate_resolution.json`,
frozen 2026-09-20 21:42 UTC by the chain loop from the widened handoff (the
`stand` chain of §6, in place of the stand-alone Session 1b freeze;
`resolution_sha256 b0e34bab…`, `task_sha256 2c6f4a47…`, `panel_seed_start`
3042, success bound Clopper-Pearson one-sided, paired bound Student-t
one-sided, alpha 0.05).

From `<RUN_DIR>/02_recovery/gate_resolution.json` (`null_manifest`,
`task_sha256`, `decision_procedure.panel_seed_start = 3042`) against the
2026-08-28 freeze (`TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md`; KNOWN_ISSUES,
the stance-gate LOW entry):

| null controller | 2026-08-28 freeze (pre-bump task hash `97c28f29…`, r12) | re-roll (Phase C task hash `2c6f4a47…`, r13) |
|---|---|---|
| statue (`zero_action`) recovery rate | 0/40 (0/40 full horizon, 26/50 per-shove, mean length 360) | 0/40 (`success_ucb95` 0.0722; every seed 3042–3081 `false`) |
| brace recovery rate | 0/40 (0/40 full horizon, 8/35 per-shove, mean length 260) | 0/40 (`success_ucb95` 0.0722; every seed `false`) |
| brace source (`checkpoint_sha256`) | the pre-bump stance handoff the 2026-08 recovery runs warm-started from (that note, §3) | the widened seed-44 handoff of §1, `7f4284ad…` |
| safe set (height error / tilt / planar speed) | 0.0168 m / 0.0825 rad / 0.3203 m/s (the P3 calibration) | 0.0168 m / 0.0825 rad / 0.3203 m/s (identical) |

Expected: the statue null is identical up to the reset stream (the golden
fixture pins the seeded reset draws; `command_mode = "none"` draws nothing),
and the brace differs only if the widened policy's post-settle mean action
differs from the parent's (pin: ≤ 1e-6 per action dim on the seeded rollout).

## 4. The C½ walker (or the retrain decision)

`<to be filled after Session 2>`

| item | value |
|---|---|
| run id / `TRUNK_FROM` | `<…>` / `<Session-1 RUN_ID>` |
| stance reuse | `<reused from the trunk under rules 1–7 — quote the ancestor record>` |
| locomotion budget / duration | 8M / `<…>` |
| `reward_and_length/v1` verdict | `<PASS / FAIL>`: mean forward velocity `<…>` m/s (≥ 1.0), mean length `<…>` (≥ 750) |
| bundle status | `<complete / partial / failed>` |

If Session 1 failed and the fallback ran: `RETRAIN_FROM = "stance"`, the
seeds tried, and which stance certified.

## 5. What this means for every pre-bump verdict

`<to be filled after Session 1>` — confirm against the log tree, then keep:

Every `gate_verdict.json` minted before the Phase C revision hashes a task
whose `policy_interface_sha256` is a pre-bump one (r11 for the two certified
parents, r12 for the 2026-08-16..18 stance PASSes and the recovery runs), so
reuse rule 3 refuses it as a trunk, and rule 6 would refuse the pre-bump
archive's identity even if the hash matched; every pre-bump
`gate_resolution.json` records a pre-bump `task_sha256` and is stale under
`require_gate_resolution`. The remedy is never a re-judge in place (the
storage cell refuses the old directory; past that, the JUDGE branch's
`validate_model_plant` refuses the archive) and never a republish (the
storage cell mints an r13 provenance the audit rejects): it is a widened
copy in a new run, re-paneled
there — Sessions 1 and the seed-44 repeat above, under
`WIDEN_MAX_REVISION_GAP = 2` for the r11 parents (D-C17; an r12 archive
widens under the default 1) — and, for recovery, a fresh
freeze from the widened handoff (Session 1b). The pre-bump directories stay
on the log tree as history. Record here which log-tree directories were
inventoried and what each now is (`<…>`).

## 6. Status 2026-09-19 (appended)

No session of this note has run. The Drive survey of 2026-09-17
([DRIVE_RUN_SURVEY_2026_09.md](DRIVE_RUN_SURVEY_2026_09.md)) changed what is
still needed; the table and placeholders above are kept as written.

- **Sessions 1 and 2 are superseded.** The fresh r13 run `20260914_123816`
  (seed 42, commit `35dd44c`, `01_stance` + `03_locomotion`) certified stance
  and locomotion from scratch on 2026-09-15 (stance `gate_verdict.json` PASS at
  01:57 UTC, final eval 3460.6 ± 19.2; locomotion PASS at 11:39 UTC, 1.07 m/s,
  mean episode length 1000). Its `task_sha256` / `gate_sha256` match the
  current stage TOMLs at `22c1fc8`, so `TRUNK_FROM = "auto"` selects it for any
  trex session. Seed 42 therefore needs no widen (Session 1) and the C½ walker
  exists without one (Session 2).
- **Seed 43 failed at r13.** The fresh replicate `20260915_160239` (11M steps,
  13h17m, final eval 3209.7 ± 284.3) failed the duty rail on its 40-episode
  panel: mean unsupported duty 0.0323, UCB 0.0350 against 0.02; reward and
  full-horizon fraction passed.
- **The seed-44 widen (the "1 repeat" row) is the one session still owed.** It
  now runs with `BEHAVIOR = "stand"` instead of `"stance"` (`WIDEN_FROM =
  "20260815_205206"`, `WIDEN_MAX_REVISION_GAP = 2`, `SEED = 44` set before the
  storage cell mints `RUN_ID`), so that after the widen and the 40-episode
  re-panel the chain loop freezes the recovery resolution from the widened
  handoff and trains recovery (3M) in the same run. That covers Sessions 1b
  and 3. Fill the seed-44 columns of §1–§3 from that run's `widen_report.json`,
  `gate_verdict.json`, `stance_gate_report.json` and
  `02_recovery/gate_resolution.json`; the seed-42 columns stay empty with a
  pointer to `20260914_123816`. If the re-panel fails the duty rail as seed 43
  did, the fallback is a fresh trex stance with `SEED = 45`.
- Numbers and the per-run evidence: the survey note. The run plan and the
  session status: [../NEXT_STEPS.md](../NEXT_STEPS.md) §3.

## 7. Status 2026-09-20 (appended)

- **The seed-44 widen ran** as `20260920_010912` (`BEHAVIOR = "stand"`,
  `WIDEN_FROM = "20260815_205206"`, `WIDEN_MAX_REVISION_GAP = 2`, `SEED = 44`,
  `REPO_REF = "main"` at `ac409f8`, Python 3.13.15): the preflight and the
  widen passed, and the 40-episode re-panel reproduced the r11 certificate
  exactly (the widened policy is the parent's under zero command columns; the
  §2 seed-44 column equals the pre-bump column to every printed digit). The
  seed-44 columns of §1 and §2 above are filled from the run's
  `widen_report.json`, `stage_config.json`, `stance_gate_report.txt` and
  `gate_verdict.json`.
- **The session died after the verdict.** The chain loop's `save_run_bundle`
  raised `ResultBundleError: best_eval_reward must be a finite number for
  canonical stage 1`: a widened root never trained in its run, so the JUDGE
  branch found no `evaluations.npz` and left `best_eval_*` unmeasured, which
  the canonical summary rule rejected. Fixed the same day (CHANGELOG, "A
  widened root's run bundle writes": the schema accepts the null for a stage
  whose deliverable record names `widened_from_run_id`). The §2 acceptance
  ("a `complete` stance bundle") and §3 (the recovery freeze and 3M training)
  are owed by the in-place continuation described in
  [../NEXT_STEPS.md](../NEXT_STEPS.md) §3 ("Continuing session 1").
- Two template expectations were off: `num_timesteps` inherited is
  10,000,000 (the parent's run block), not 10,002,432; and the parent's
  `task_sha256` is `null` because its `stage_config.json` predates task
  fingerprints, so the widen report's parent block records `null` and the
  widened node's digest comes from the current stage TOML.

## 8. Status 2026-09-23 (appended)

- **Session 1 is complete.** The in-place continuation of `20260920_010912`
  (2026-09-20 from 21:38 UTC, `REPO_REF = "main"` at `25132fc`, the same
  Python 3.13.15 image; provenance `sessions[1]` records the commit drift
  `ac409f8` → `25132fc`) reused the widened stance, froze the recovery
  resolution from its handoff (§3, filled above), trained recovery for
  3,006,464 steps (3h30m) and PASSED `recovery_quality/v1` at 01:16:47 UTC on
  2026-09-21: 28/40 panel successes (Clopper-Pearson one-sided LCB 0.56
  against `min_recovery_success_lcb` 0.3), paired success delta against the
  0/40 statue null 0.70 (Student-t one-sided LCB 0.58 against
  `min_paired_success_delta_lcb` 0.2), 140 of 155 pushes recovered, 34/40
  full horizon, panel reward 2925.4 ± 404.8 (`evidence/policy.csv`);
  training final eval 2911.16 ± 556.35, best eval 3031.38 ± 159.07 at 2.9M;
  `gate_sha256 a27ce071…`, `task_sha256 2c6f4a47…` (= the resolution's),
  checkpoint `a8e41b98…`. The bundle is `complete`: the stance deliverable
  records `widened_from_run_id 20260815_205206`, `best_eval_reward` null (the
  rule of #546), `certified true`, `provisional false` and `replication count
  2` (`20260920_010912` seed 44, `20260914_123816` seed 42); the recovery
  deliverable is certified at replication 1. The §2 acceptance ("a `complete`
  stance bundle") is met, and trex stance reads 2 runs of 2 seeds in this
  run's records (the seed-42 run's own records read 1 until its bundle cell
  is re-run beside this sibling).
- **§3 holds the chain loop's freeze, not a stand-alone Session 1b run**: the
  `stand` chain froze `02_recovery/gate_resolution.json` from the widened
  handoff before training (the 2026-09-19 re-scoping of §6). Both nulls
  reproduce the 2026-08-28 freeze's 0/40 exactly and the safe set is the P3
  calibration to every digit, so the §3 expectation holds; the r12-hashed
  freeze of the 2026-08 note stays on the log tree as history.
- **The same question was answered for compsognathus** on 2026-09-21: the
  r1 → r2 widen of `20260909_162812` (gap 1, run `20260921_203149`)
  re-paneled to 2801.6 ± 51.2, full-horizon 1.0000, duty 0.0131 / UCB
  0.0141, the parent's report to every printed digit, and then trained a
  certified locomotion node (3,002,368 steps, 3308.6 ± 13.0, 0.34 m/s). With
  both widen sessions decided, decision D-D14's condition for retiring the
  notebook widen path is met.
- §4 stays superseded (`20260914_123816` is the walker) and §5's log-tree
  inventory is not done here; the template status line at the top stays
  until §5 is filled. The run plan and what remains:
  [../NEXT_STEPS.md](../NEXT_STEPS.md) §3.

## §9 Correction 2026-09-23 (appended)

§8's "until its bundle cell is re-run beside this sibling" does not hold, and
the seed-42 rebuild step it implies is dropped from
[../NEXT_STEPS.md](../NEXT_STEPS.md) §3. `20260914_123816`'s bundle is
`complete` under a stance target (`summary.json` written 2026-09-15 02:08
UTC; `03_locomotion/` created later), a re-entry that reuses every node
writes no bundle, and a direct save is refused as immutable because files
appeared after publication. Its run-level records stay stance-only at
replication 1, while this run's bundle counts it at 2; reuse reads the
per-node files, so nothing depends on them. The same rule sends a locomotion
node for this run (NEXT_STEPS.md session 7) to a fresh run with
`TRUNK_FROM = "20260920_010912"`, since this run's bundle is `complete` too
([../KNOWN_ISSUES.md](../KNOWN_ISSUES.md), "A complete run cannot take a new
node in place").
