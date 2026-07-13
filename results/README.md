# Training Results

Curated training results (GIFs and metrics) organized by species and algorithm.

## Directory Structure

```
results/
├── velociraptor/
│   ├── ppo/
│   │   ├── collected_results.csv
│   │   ├── stage1_balance.gif
│   │   ├── stage3_strike.gif
│   │   └── summary.json
│   └── sac/
│       ├── collected_results.csv
│       ├── stage3_strike.gif
│       └── summary.json
├── trex/
│   └── ppo/
│       ├── collected_results.csv
│       └── summary.json
├── brachiosaurus/
│   └── ppo/
│       ├── collected_results.csv
│       └── summary.json
└── README.md
```

## Naming Conventions

- **GIFs**: `stage<N>_<task>.gif` — matches the TOML config filenames in `configs/<species>/`

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
backend and provenance metadata mandatory.

## How Results Are Generated

The SB3 training workflow can generate a run-level `collected_results.csv`;
the JAX and sweep workflows have their own result-export sections. Files are
promoted into this curated directory deliberately rather than by every notebook
automatically. The `summary.json` files are maintained as the public,
provenance-labelled snapshots consumed by the catalog generator.

The `collected_results.csv` contains per-stage rows with all hyperparameters,
metrics, curriculum thresholds, and pass/fail status — the same format used by
sweep results, so single runs and sweeps can be compared with the same tooling.
