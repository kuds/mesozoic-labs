# Training Results

Curated training results (GIFs and metrics) organized by model (species and
embodiment, [docs/SPECIES_NAMING.md](../docs/SPECIES_NAMING.md)) and algorithm,
and the ledger of judgements made outside training
([`offline_judgements.csv`](#offline-judgements-offline_judgementscsv)).

## Directory Structure

```
results/
├── velociraptor_mongoliensis/
│   └── anatomical/
│       └── ppo/
│           └── stance.gif
├── offline_judgements.csv
└── README.md
```

No result summary is published here at present. The species catalog reads
every `results/<species>/<embodiment>/<algo>/summary.json` on disk, and each
must be listed in its model's `result_summary_paths` in
`configs/species_manifest.toml`; today every model lists none.

## Naming Conventions

- **Paths**: `results/<species>/<embodiment>/<algo>/`
- **GIFs**: `<stage>.gif`, the stage config's name under `configs/<species>/<embodiment>/`

A summary's top-level `backend` identifies the training implementation, and
its `provenance` block says whether it measures the current model revision.
A run may be labeled `current` or `verified` only when its repository commit,
model hash, configuration hash, backend version, and evaluation-episode count
are recorded; summary schema version 2 made this backend and provenance
metadata mandatory.

## Removed historical summaries (decision D-D59)

Until 2026-11-23 this directory held four historical ladder summaries, each
with its `collected_results.csv`: the velociraptor PPO and SAC runs and the
T-Rex PPO run, all from March 2026, and the Brachiosaurus PPO run of
2026-07-18, plus a `strike.gif` for each velociraptor run. The last stage of every one was a
hunt (the strike, the bite and the food reach), and decision D-D59 removed
the hunt nodes, so stage 3 of those summaries no longer resolves to any
stage and the summaries were deleted with them rather than edited (dropping
the stage-3 rows would have left recorded totals that no longer add up).
The last commit that holds them is
`6c9982d81c8a34c0402905feb2724cffde04e974`; `git show 6c9982d81c8a:results/<path>`
reads any of them. They were schema 2, marked `historical` and `unverified`:
they predated model and configuration hashes, so they were never evidence
for a current model revision. The readers accept schema 5 alone since
decision D-D63 retired schema 2; the tree at that commit (version
0.4.0.dev2) still reads them, while a later tree, even one that still
declares 0.4.0.dev2, refuses their stage 3. The website's two unused apex
GIFs, `website/static/img/ppo_apex.gif` and `sac_apex.gif`, were byte-identical
copies of the two `strike.gif` and were deleted with them; the same commit
holds them (`git show 6c9982d81c8a:website/static/img/<name>`).

`velociraptor_mongoliensis/anatomical/ppo/stance.gif` is the stance stage of
the March 2026 velociraptor PPO run whose summary was removed. It stays as
the root README's illustration, a historical recording rather than a result
of the current model.

## How Results Are Generated

SB3 training runs are saved to Google Drive as portable result bundles (so were
the JAX/MJX runs before decision D-D17 retired that backend; they still validate).
The shared exporter captures provenance at run time and saves the bundle after
every node the training notebook trains or judges. Each save writes the
canonical `summary.json` whenever at least one deliverable is certified,
derives `collected_results.csv` from the same normalized metrics, binds
selected and terminal claims to episode-level evidence, records the selected
checkpoint (and SB3 normalization state) for every stage, and writes an
artifact manifest last. See
[`docs/RESULT_BUNDLES.md`](../docs/RESULT_BUNDLES.md), and
[`docs/DRIVE_LAYOUT.md`](../docs/DRIVE_LAYOUT.md) for where each file sits on
Drive.

Files are promoted into this curated directory deliberately rather than by
every notebook automatically. The catalog consumes `summary.json`; CSV and
documentation are derived views and must agree with it, and
`environments/shared/tests/test_result_summaries.py` fails on any
`collected_results.csv` that contradicts its `summary.json`. Legacy artifacts
that predate bundles remain historical/unverified and may be reported as
conflicting instead of being silently rewritten.

## Offline judgements (`offline_judgements.csv`)

`offline_judgements.csv` is the repository's ledger of judgements made outside
training, one row per judged checkpoint: a gate certificate scored on a saved
checkpoint after its run (such as a six-heading stance certificate by
`environments/shared/scripts/stance_gate_report.py`), or a pilot's fixed
criteria applied to a checkpoint by a script. A training run writes its own
verdicts to its folder on Drive; a judgement made where Drive cannot be written
(the session agent never writes to Drive) is recorded here instead, by pull
request. Nothing writes this file. The Drive summary notebook
(`notebooks/google_drive_summary.ipynb`, section 9) adds its rows, with `source`
`offline`, to the cross-run table it saves on Drive as `logs/runs_summary.csv`
([RESULT_BUNDLES.md](../docs/RESULT_BUNDLES.md#the-cross-run-table-and-offline-judgements)).

A row records a judgement and nothing more: it certifies no run, writes no
result bundle, and promotes nothing into this directory.

`environments.shared.reporting.load_offline_judgements` reads and validates the
file, and `test_reporting_runs_summary.py` runs it on the committed file in CI.
The header is exactly the columns below, in this order: an unknown, missing,
misspelt or reordered column is refused, as is an empty required cell or a
repeated row, and every problem is reported with its line. Quote a cell that
holds a comma.

| Column | Required | Value |
|---|---|---|
| `judged_on` | yes | The date of the judgement, `YYYY-MM-DD`. |
| `species` | yes | The model's species ID ([SPECIES_NAMING.md](../docs/SPECIES_NAMING.md)). |
| `embodiment` | yes | Its embodiment. The pair must be a model of `configs/species_manifest.toml` (decision D-D33). |
| `algorithm` | yes | The public algorithm label, `PPO` or `SAC` (mirror-symmetric PPO is `PPO`). |
| `run_id` | one of the two | The run's `run_id`, as its `provenance.json` records it. |
| `run_path` | one of the two | The run's directory relative to `logs/<species>/<embodiment>/` on Drive (`ppo/<run directory>`), as the table's `run_path` reads; for a checkpoint judged elsewhere, where it was taken from. |
| `stage` | yes | The stage as `collected_results.csv` writes it: the legacy number (`1` for the stance stage) or the semantic ID (`recovery`). |
| `checkpoint_step` | yes | The checkpoint's training step, in timesteps. |
| `checkpoint_sha256` | yes | The digest of the checkpoint file judged, as the bundle and evidence files record one: `sha256:` and 64 lowercase hexadecimal digits (`result_bundle.sha256_file`). |
| `gate_kind` | yes | The gate kind applied (`stance_quality_heading/v1`, `stance_quality/v2`, ...), or for a judgement that is no gate's, the name of the fixed criteria applied. One word. |
| `verdict` | yes | `PASS` or `FAIL` for a gate; otherwise the criteria's own outcome (`WALKS`, `MIXED`, ...). Upper-case letters, digits and `_`. In the table, `PASS` reads `stage_passed` true, `FAIL` false, and any other verdict leaves it empty. |
| `passing_count` | no | How many units of the panel met the criterion: clean seeds for a stance certificate, WALK episodes for a gait judgement. |
| `panel_size` | no | The panel's size, n; at least `passing_count`. |
| `lower_bound` | no | The one-sided lower confidence bound on the passing rate that the verdict was decided on, from 0 to 1. |
| `lower_bound_bar` | no | The bar that bound had to reach (a stance gate's `min_clean_stance_lcb`), from 0 to 1. |
| `binding_bar` | no | The criterion that bound the verdict, the one the failing units failed, with its bar (`max_sole_cop_fore_aft 0.80`). |
| `panel_seeds` | yes | The evaluation seeds or panel (`3042-3081`). |
| `judging_commit` | yes | The commit of the tree the judging tool ran from, 7 to 40 lowercase hexadecimal digits. |
| `judged_by` | yes | The tool, with any options that change the judgement. |
| `evidence` | yes | Where the judgement's output is kept: a repository path or a link. |
| `note` | no | Anything else a reader needs, such as why the checkpoint was judged offline. |
| `source` | yes | `offline`, always. |
