# Result Bundles

Mesozoic Labs training runs are produced in Google Colab, persisted to Google
Drive, and promoted into `results/` later. A result bundle preserves the
evidence needed to make that hand-off reproducible.

## Two validation levels

- **Drive validation** accepts partial and gate-failed curricula. These runs
  retain their configs, checkpoints, raw evaluation episodes, and provenance
  even though they are not publishable.
- **Promotion validation** requires all three stages, the selected Stage 3
  checkpoint, a passed gate for every stage, all resolved stage configs,
  plant identity, recorded selected- and terminal-policy evaluation episodes,
  a canonical `summary.json`, matching CSV metrics, and valid artifact hashes.

A partial bundle must not be presented as a public three-stage result.
Run `20260720_203454` is a concrete failed-but-auditable example; its role in
the Velociraptor diagnosis is documented in the
[Stage-1 basin investigation](investigations/VELOCIRAPTOR_STAGE1_BASIN_INVESTIGATION.md).

## Canonical layout

```text
<run-id>/
├── provenance.json
├── plant_identity.json
├── collected_results.csv
├── summary.json                 # written only after stages 1–3
├── artifact_manifest.json       # completion marker; written last
├── training_summary.txt
├── stage1/
│   ├── stage_config.json
│   ├── evaluation_final.csv
│   ├── evaluation_selected.csv
│   └── models/
├── stage2/
└── stage3/
```

JAX/MJX also writes `stage_result.json` so stages completed in separate Colab
sessions can be combined idempotently under one run ID.

Completed bundles are immutable. Start a new run ID rather than replacing a
checkpoint, stage result, seed role, or other captured experiment setting.
Repeating an identical export is a write-free no-op, so a transient Drive
failure cannot remove the completion marker from an already valid bundle.

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
- plant identity; and
- every stage's selected checkpoint hash, matching SB3 VecNormalize path/hash,
  and resolved-config hashes when available.

Dirty source trees remain auditable as partial or failed runs, but cannot be
promoted. Commit the intended source state and start a new run ID.

Missing historical values are never guessed. Legacy Drive directories remain
readable but are labelled `legacy-unverified` or `legacy-conflict`.

## Colab workflow

The SB3 and JAX notebooks call the shared bundle functions. For JAX curricula,
set and reuse the same `RUN_ID` for stages 1–3. Stages 2 and 3 automatically
load the preceding stage's `models/best_model.pkl` only after its recorded gate
passes. A new run ID starts an independent experiment. SB3's three stages run
in one ordered notebook session because its stage objects are kept in memory.
JAX evaluates the selected and terminal parameters separately and saves both
episode files; a training-rollout selection score is not reported as an
evaluation reward.

`publication_gate_passed` is the promotion decision recomputed from the fixed
publication evaluation and the frozen stage thresholds. Training-time
`required_consecutive` settings describe chronological evaluation batches and
are not inferred from episodes in the publication evaluation.

Reusing a JAX run in a fresh Colab session also requires the exact captured Git
state and dependency versions. If `main` has moved, inspect
`provenance.json`, check out its `repository_commit`, and reinstall that
revision before continuing. The initializer rejects a mismatched environment
instead of combining stages from different experiments.

The Google Drive summary notebook audits canonical bundles first and uses its
older CSV/text/NPZ reconstruction only as a historical fallback. Before
promotion, run strict validation:

```python
from environments.shared.result_bundle import validate_result_bundle

validate_result_bundle(RUN_DIR, require_complete=True)
```

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
mislabeling JAX as a different reinforcement-learning algorithm.
