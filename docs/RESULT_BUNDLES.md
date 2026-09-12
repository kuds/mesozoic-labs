# Result Bundles

Mesozoic Labs training runs are produced in Google Colab, persisted to Google
Drive, and promoted into `results/` later. A result bundle preserves the
evidence needed to make that hand-off reproducible.

## Two validation levels

- **Drive validation** accepts partial and gate-failed curricula. These runs
  retain their configs, checkpoints, raw evaluation episodes, and provenance
  even though they may not be publishable.
- **Promotion validation** (`validate_result_bundle(..., require_publishable=True)`)
  requires at least one *certified deliverable*, a selected checkpoint (and
  SB3 VecNormalize sidecar) for every recorded stage, every resolved stage
  config with plant identity, recorded selected- and terminal-policy
  evaluation episodes for every recorded stage, a canonical `summary.json`,
  matching CSV metrics, and valid artifact hashes.

Publication is **per deliverable** (result schema v4,
[BEHAVIOR_RECIPES_PLAN §4.3](BEHAVIOR_RECIPES_PLAN.md)). The species' stage
manifest flags which nodes are deliverables and which node each one
warm-starts from. A deliverable is *certified* when its own gate passed and
every `warm_start_from` ancestor is present — trained in the bundle, or
carried by an `ancestors/<stage_id>/` record of a node reused from another run
— with a passed gate. Absence never reads as a pass.

Every bundle save names a **target deliverable** (`save_result_bundle(...,
target_deliverable=...)`; the manifest's last deliverable by default, the
notebook's `BEHAVIOR` node in the chain loop). The bundle status is:

| status | meaning |
|---|---|
| `complete` | the target is present and certified **and** every deliverable present in the run is certified — the bundle is then immutable |
| `partial` | at least one deliverable is certified, but not the target or not every present one — a stance-only run targeting hunt, a walk-only run targeting hunt, a run whose hunt leaf failed above a certified trunk |
| `failed` | no deliverable is certified — a failed root, or a trunk whose gate failed below every leaf |

`summary.json` is written whenever at least one deliverable is certified, so a
failed leaf still publishes its certified trunk and a walk-only run writes a
valid bundle. The **primary deliverable** — `provenance.selected_model_path`
/ `model_hash` and the headline `final_avg_reward` — is the target when it is
certified, else the deepest certified deliverable in manifest order; it is
always a certified deliverable, never a failed leaf.

`require_complete=True` (the notebook's completion cell) demands
`bundle_status == "complete"`, audited as `canonical-valid`;
`require_publishable=True` (the audit, the bundle writer, the species
catalog, the notebook's gate-failure branch) also accepts a partial bundle
with its summary, audited as `canonical-partial`. A partial bundle must not be
presented as a complete result. Run `20260720_203454` is a concrete
failed-but-auditable example; its role in the Velociraptor diagnosis is
documented in the
[Stage-1 basin investigation](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md).

Under a v1 or synthesized stage manifest (one deliverable — the last
advancing node — with edges to the previous advancing node) these rules
collapse to the historical "every advancing stage present and passed /
terminal = last advancing stage" rule, so every pre-Phase-A bundle audits
exactly as before. Schema-2 and schema-3 summaries keep their own rules.

## Canonical layout

```text
<run-id>/
├── provenance.json
├── plant_identity.json
├── collected_results.csv
├── summary.json                 # written whenever >= 1 deliverable is certified
├── artifact_manifest.json       # status marker (complete / partial / failed); written last
├── training_summary.txt
├── 01_stance/                   # stage{N} historically, NN_<id> from 2026-08-20 on
│   ├── stage_config.json
│   ├── gate_verdict.json        # per-node verdict, hash-bound to the handoff pair
│   ├── evaluation_final.csv
│   ├── evaluation_selected.csv
│   └── models/
├── 03_locomotion/
├── 04_behavior/
└── ancestors/                   # nodes reused from another run — records, never checkpoints
    └── stance/
        ├── ancestor.json        # mesozoic.ancestor-record/v1
        ├── gate_verdict.json    # verbatim copies of the ancestor stage's files
        ├── stage_config.json
        ├── task_fingerprint.json
        └── plant_identity.json
```

`gate_verdict.json` (`mesozoic.gate-verdict/v1`) lives in the stage
directory root beside `stage_config.json` and records the node's verdict with
the SHA-256 of the handoff checkpoint and its VecNormalize sidecar; it is what
makes a node reusable as an ancestor by a later run. Pre-Phase-A bundles have
none, and the audit does not require it.

`ancestors/<stage_id>/ancestor.json` records `{schema, stage_id, stage_key,
parent_run_id, source_run_dir, source_stage_dir, handoff {name, model_path,
model_sha256, normalization_path, normalization_sha256}, task_sha256,
judged_by, reused_at}`. The reader
(`result_bundle.load_ancestor_records`) fails closed on an undeclared stage
id, a missing file, a verdict whose hashes disagree with the record's, or a
task fingerprint that disagrees between `ancestor.json`, `gate_verdict.json`
and `stage_config.json`. A stage that warm-started from a reused ancestor
records `parent_run_id` in its `stage_config.json` run block, and the audit
binds that lineage to the record's checkpoint hash.

JAX/MJX also writes `stage_result.json` so stages completed in separate Colab
sessions can be combined idempotently under one run ID.

Completed bundles are immutable. Start a new run ID rather than replacing a
checkpoint, stage result, seed role, or other captured experiment setting.
Repeating an identical export is a write-free no-op, so a transient Drive
failure cannot remove the completion marker from an already valid bundle. A
partial marker — a chain whose target has not certified yet — is rebuilt
over by the next node's save, which is how the notebook's per-node saves grow
one bundle.

A completion marker that no longer verifies is not the end of the run when
only the artifacts the export itself regenerates (`summary.json`,
`collected_results.csv`, `provenance.json`, `plant_identity.json`) disagree
with it — an export interrupted mid-rewrite, say: the next export rebuilds
them in place, and only after the rebuilt bundle has passed every check does
it replace the marker. If a certified artifact — a checkpoint or its
VecNormalize sidecar, evaluation evidence, a stage config, a gate verdict, an
ancestor record — changed after publication, the export refuses and names the
file; that bundle needs a new run ID.

## Source of truth

`summary.json` is the canonical public result. `collected_results.csv` is
generated from the same normalized stage values, with resolved
hyperparameters added for analysis. README and website tables are generated
from the summary.

Do not manually copy metrics among JSON, CSV, and documentation. The bundle
validator treats disagreements as conflicts.

## Provenance captured at run time

`provenance.json` is initialized before training and records:

- run ID, species, algorithm, and backend;
- repository URL, exact commit, dirty state, and patch hash when dirty;
- training and evaluation seed roles;
- deterministic evaluation protocols and episode counts;
- Python, platform, and dependency versions;
- hardware and parallel environment count;
- plant identity;
- a `sessions` record — one entry per process that touched the run: the
  creating session's `started_at`, then a `resumed_at` entry for every later
  session, carrying an `environment_drift` map when that session's
  environment differed from the original capture.

The bundle writer adds the **finalization** fields (schema v4):

- every stage's selected checkpoint hash, matching SB3 VecNormalize path/hash,
  and resolved-config hashes (`selected_checkpoints`, `config_hash`);
- `deliverables` — `{stage_key: {model_path, model_hash, normalization_hash,
  gate_kind, certified, replication: {count, runs: [{run_id, training_seed}]}}}`
  for every deliverable present in the run (Phase A writes one run per
  record);
- `ancestors` — `{stage_key: {run_id, model_hash, normalization_hash,
  gate_kind, passed, task_sha256}}`, the projection of the on-disk
  `ancestors/` records;
- `target_deliverable` and `primary_deliverable` (stage keys, the primary
  `null` when nothing is certified), with `selected_model_path` /
  `model_hash` pointing at the primary.

These are finalization fields, not identity: re-running the notebook's setup
cell under the same run ID with a different `BEHAVIOR` is not a rejected run.

Dirty source trees remain auditable as partial or failed runs, but cannot be
promoted. Commit the intended source state and start a new run ID.

Missing historical values are never guessed. Legacy Drive directories remain
readable but are labelled `legacy-unverified` or `legacy-conflict`.

## Colab workflow

The SB3 and JAX notebooks call the shared bundle functions. For JAX curricula,
set and reuse the same `RUN_ID` for stages 1–3. Stages 2 and 3 automatically
load the preceding stage's `models/best_model.pkl` only after its recorded gate
passes. A new run ID starts an independent experiment. SB3's chain runs in one
ordered notebook session because its stage objects are kept in memory; the
bundle is saved after every node, growing from `partial` to `complete` when
the target certifies.
JAX evaluates the selected and terminal parameters separately and saves both
episode files; a training-rollout selection score is not reported as an
evaluation reward.

`publication_gate_passed` is the promotion decision recomputed from the fixed
publication evaluation and the frozen stage thresholds. A recorded pass must
be reproducible from the evidence; a recorded failure is bound to its
evidence but is not re-judged, and can never be certified. Training-time
`required_consecutive` settings describe chronological evaluation batches and
are not inferred from episodes in the publication evaluation.

Reusing a JAX run in a fresh Colab session revalidates the run's identity:
species, algorithm, backend, seed roles, evaluation protocols and seeds,
episode and parallel-environment counts, plant identity, and the run ID must
match the captured provenance exactly, and the initializer refuses a mismatch
instead of combining stages from different experiments. Environment fields —
Python, platform, and dependency versions, the Git commit/dirty state and
patch hash, and hardware — may drift between sessions: the resume is accepted,
the top-level provenance keeps the values the first session captured, and the
drift is recorded (with a logged warning) on that session's entry in the
per-session `sessions` record. For faithful reproduction, still prefer
checking out the captured `repository_commit` and reinstalling that revision
before continuing.

The Google Drive summary notebook audits canonical bundles first and uses its
older CSV/text/NPZ reconstruction only as a historical fallback; it reads
`summary.json` for both `canonical-valid` and `canonical-partial` bundles.
Before promotion, run strict validation:

```python
from environments.shared.result_bundle import validate_result_bundle

validate_result_bundle(RUN_DIR, require_complete=True)       # the target and every present deliverable certified
validate_result_bundle(RUN_DIR, require_publishable=True)    # at least one certified deliverable, summarised
```

The audit statuses are `canonical-valid` (summary + complete manifest),
`canonical-partial` (summary + partial manifest), `partial` / `failed` (no
summary), and `canonical-conflict` (any error — including a summary whose
`bundle_status` disagrees with the manifest, or a summary under a failed
manifest).

The manifest hashes files present when the exporter runs, and promotion rejects
undeclared files. The JAX notebook writes an early marker after the core
artifacts, then refreshes and validates it after optional plots and videos.
If artifacts are added manually, regenerate the manifest before promotion.
Both notebooks flush Google Drive before an automatic runtime disconnect.

## Public result paths

The public algorithm remains `PPO` or `SAC`; backend identity is separate:

```text
results/<species>/ppo/summary.json       # SB3 PPO
results/<species>/sac/summary.json       # SB3 SAC
results/<species>/jax_ppo/summary.json   # JAX/MJX PPO
```

This allows SB3 and JAX results for the same algorithm to coexist without
mislabeling JAX as a different reinforcement-learning algorithm. The paths are
unchanged by schema v4; the summary carries the deliverables map.
