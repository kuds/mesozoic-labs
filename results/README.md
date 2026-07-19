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
│   ├── sac/
│   │   ├── collected_results.csv
│   │   ├── stage3_strike.gif
│   │   └── summary.json
│   └── jax_ppo/                    # when a JAX/MJX PPO result is promoted
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

SB3 and JAX training runs are saved to Google Drive as portable result bundles.
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
