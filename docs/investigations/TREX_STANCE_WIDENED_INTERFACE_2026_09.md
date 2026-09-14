# T-Rex Stance Widened to the Phase C Interface — 2026-09 (template)

**Status: TEMPLATE — pending the maintainer's Colab sessions (BEHAVIOR_RECIPES_PLAN
§4.6 Phase C, WS-C4).** Every `<to be filled after Session N>` below is a
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
`<to be filled after the seed-44 repeat: RUN_ID>` (seed 44 — the widened copy
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
| widened run id | `<Session 1>` | `<seed-44 repeat>` |
| parent handoff (`handoff_name`) | `<robust_best_model / best_model>` | `<…>` |
| parent `checkpoint_sha256` / `normalization_sha256` | `<…>` | `<…>` |
| parent `task_sha256` (r11) → widened `task_sha256` (r13) | `<…>` → `<…>` | `<…>` → `<…>` |
| widened `checkpoint_sha256` / `normalization_sha256` | `<…>` | `<…>` |
| `from_observation_dim` → `to_observation_dim` | 61 → 64 (expected) | 61 → 64 (expected) |
| `revision_gap` / `max_revision_gap` (r11 → r13 under `WIDEN_MAX_REVISION_GAP = 2`) | 2 / 2 (expected) | 2 / 2 (expected) |
| run block `widened_from_policy_interface_revision` (`stage_config.json`) | 11 (expected) | 11 (expected) |
| `padded_tensors` | `mlp_extractor.policy_net.0.weight`, `mlp_extractor.value_net.0.weight` (expected; columns 61–63) | same |
| `optimizer_members_padded` | `policy.optimizer` (expected) | same |
| `max_padded_column_abs` (pin: exactly 0) | `<…>` | `<…>` |
| `max_action_delta_zero_command` / `max_action_delta_probe_command` (tolerance 1e-6, seed 3042, 200 steps) | `<…>` / `<…>` | `<…>` / `<…>` |
| `num_timesteps` inherited | 10,002,432 (expected) | 10,002,432 (expected) |
| `widened_by` (tool@commit) | `<…>` | `<…>` |

## 2. The re-panel: widened stance vs its pre-bump certificate

`<to be filled after Session 1 and the seed-44 repeat>`

The same 40-episode panel (seeds 3042–3081, settle 200) under
`stance_quality/v1`, judged by `generate_stage_artifacts`. Pre-bump columns
from the parents' records (`TREX_STAGE1_GATE_PASS_RUN_2026_08.md`,
KNOWN_ISSUES update 3); the widened columns from each run's
`stance_gate_report.json` / `gate_verdict.json`.

| criterion | seed 42 pre-bump (r11 certificate) | seed 42 widened (r13) | seed 44 pre-bump | seed 44 widened (r13) | required |
|---|---|---|---|---|---|
| full-horizon fraction | 1.0000 (40/40) | `<…>` | 1.0000 (40/40) | `<…>` | ≥ 0.95 |
| mean unsupported duty | 0.0048 | `<…>` | 0.0069 | `<…>` | ≤ 0.02 |
| duty UCB (95%) | 0.0080 | `<…>` | 0.0117 | `<…>` | ≤ 0.02 |
| panel reward | 3368.7 ± 46.7 | `<…>` | 3408.3 ± 88.5 | `<…>` | ≥ 2100 (rail) |
| bilateral support | 0.9938 | `<…>` | `<from the seed-44 report>` | `<…>` | (statue 0.998, not gated) |
| verdict | PASS | `<PASS / FAIL>` | PASS | `<PASS / FAIL>` | — |

Acceptance for this section: `gate_verdict.json` with `passed = true`,
`judged_by = "generate_stage_artifacts"` and `task_sha256` equal to the
widened `task_sha256` of §1; a `complete` stance bundle. State the
replication label the seed-42 bundle publishes after the seed-44 sibling is
counted (`<1 run of 2 seeds; provisional / 2 runs of 2 seeds>`). If a
re-panel failed, say so here and record the retrain decision in §4.

## 3. The recovery freeze re-roll

`<to be filled after Session 1b>`

From `<RUN_DIR>/02_recovery/gate_resolution.json` (`null_manifest`,
`task_sha256`, `decision_procedure.panel_seed_start = 3042`) against the
2026-08-28 freeze (`TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md`; KNOWN_ISSUES,
the stance-gate LOW entry):

| null controller | 2026-08-28 freeze (pre-bump task hash `<…>`) | re-roll (Phase C task hash `<…>`) |
|---|---|---|
| statue (`zero_action`) recovery rate | `<…>` | `<…>` |
| brace recovery rate | `<…>` | `<…>` |
| brace source (`checkpoint_sha256`) | `<pre-bump handoff>` | `<widened handoff, §1>` |

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
