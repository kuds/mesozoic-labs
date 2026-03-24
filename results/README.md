# Training Results

Curated training results (GIFs and metrics) organized by species and algorithm.

## Directory Structure

```
results/
├── velociraptor/
│   ├── ppo/
│   │   ├── stage1_balance.gif
│   │   ├── stage2_locomotion.gif
│   │   └── stage3_strike.gif
│   └── sac/
│       ├── stage1_balance.gif
│       ├── stage2_locomotion.gif
│       └── stage3_strike.gif
├── trex/
│   ├── ppo/
│   │   ├── stage1_balance.gif
│   │   ├── stage2_locomotion.gif
│   │   └── stage3_bite.gif
│   └── sac/
│       └── ...
├── brachiosaurus/
│   ├── ppo/
│   │   ├── stage1_balance.gif
│   │   ├── stage2_locomotion.gif
│   │   └── stage3_food_reach.gif
│   └── sac/
│       └── ...
└── README.md
```

## Naming Conventions

- **GIFs**: `stage<N>_<task>.gif` — matches the TOML config filenames in `configs/<species>/`

## How Results Are Generated

The Jupyter notebooks in `notebooks/` automatically generate `collected_results.csv`
and copy stage GIFs into this directory at the end of each training run. See the
"Save Results" section in each notebook.

The `collected_results.csv` contains per-stage rows with all hyperparameters,
metrics, curriculum thresholds, and pass/fail status — the same format used by
sweep results, so single runs and sweeps can be compared with the same tooling.
