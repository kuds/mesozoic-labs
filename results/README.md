# Training Results

Curated training results (GIFs and metrics) organized by model (species and
embodiment, [docs/SPECIES_NAMING.md](../docs/SPECIES_NAMING.md)) and algorithm.

## Directory Structure

```
results/
├── velociraptor_mongoliensis/
│   └── anatomical/
│       ├── ppo/
│       │   ├── collected_results.csv
│       │   ├── stance.gif
│       │   ├── strike.gif
│       │   └── summary.json
│       └── sac/
│           ├── collected_results.csv
│           ├── strike.gif
│           └── summary.json
├── tyrannosaurus_rex/
│   └── anatomical/
│       └── ppo/
│           ├── collected_results.csv
│           └── summary.json
├── brachiosaurus_altithorax/
│   └── anatomical/
│       └── ppo/
│           ├── collected_results.csv
│           └── summary.json
└── README.md
```

## Naming Conventions

- **Paths**: `results/<species>/<embodiment>/<algo>/`
- **GIFs**: `<stage>.gif`, the stage config's name under `configs/<species>/<embodiment>/`

Task filenames and metric keys are shorthand for implemented simulation
criteria. T-Rex `bite` means contact between a fixed head geom and prey (there
is no articulated jaw), while Brachiosaurus `food_reach` means the head tip is
within a configured distance of the target (physical food contact is not
required).

The checked-in summaries predate model/config hash tracking. Their
`provenance` blocks therefore mark them as `historical` and `unverified`; they
should not be presented as measurements of the current model revision. The
top-level `backend` identifies the training implementation; the legacy
`backend_version` and evaluation-episode counts are `null` because they were
not recorded. A run may be labeled `current` or `verified` only when its
repository commit, model hash, configuration hash, backend version, and
evaluation-episode count are recorded. Summary schema version 2 makes this
backend and provenance metadata mandatory. On 2026-10-11 decision D-D33 moved
the four summaries to `results/<species>/anatomical/<algo>/` and amended only
their identity in place: `species` is the binomial, `"embodiment":
"anatomical"` is added, and each CSV gains an `embodiment` column. They stay
schema 2, every other value is as recorded, and the GIFs were renamed by
stage with their bytes unchanged.

## How Results Are Generated

SB3 training runs are saved to Google Drive as portable result bundles (so were
the JAX/MJX runs before decision D-D17 retired that backend; they still validate).
The shared exporter captures provenance at run time, writes one canonical
`summary.json` after all three stages, derives `collected_results.csv` from the
same normalized metrics, binds selected and terminal claims to episode-level
evidence, records the selected checkpoint (and SB3 normalization state) for
every stage, and writes an artifact manifest last. See
[`docs/RESULT_BUNDLES.md`](../docs/RESULT_BUNDLES.md).

Files are promoted into this curated directory deliberately rather than by
every notebook automatically. The catalog consumes `summary.json`; CSV and
documentation are derived views and must agree with it. Legacy artifacts that
predate bundles remain historical/unverified and may be reported as conflicting
instead of being silently rewritten.
