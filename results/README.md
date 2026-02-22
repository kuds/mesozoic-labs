# Training Results

Curated training results (GIFs and metrics) organized by species and algorithm.

## Directory Structure

```
results/
├── velociraptor/
│   ├── ppo/
│   │   ├── stage1_balance.gif
│   │   ├── stage2_locomotion.gif
│   │   ├── stage3_strike.gif
│   │   └── summary.json
│   └── sac/
│       ├── stage1_balance.gif
│       ├── stage2_locomotion.gif
│       ├── stage3_strike.gif
│       └── summary.json
├── trex/
│   ├── ppo/
│   │   ├── stage1_balance.gif
│   │   ├── stage2_locomotion.gif
│   │   ├── stage3_bite.gif
│   │   └── summary.json
│   └── sac/
│       └── ...
├── brachiosaurus/
│   ├── ppo/
│   │   ├── stage1_balance.gif
│   │   ├── stage2_locomotion.gif
│   │   ├── stage3_food_reach.gif
│   │   └── summary.json
│   └── sac/
│       └── ...
└── README.md
```

## Naming Conventions

- **GIFs**: `stage<N>_<task>.gif` — matches the TOML config filenames in `configs/<species>/`
- **Metrics**: `summary.json` — machine-readable training results per algorithm run

## summary.json Format

Each `summary.json` contains:

```json
{
  "species": "velociraptor",
  "algorithm": "PPO",
  "hardware": "Google Colab T4 GPU",
  "seed": 42,
  "date": "2026-02-22",
  "stages": {
    "1": {
      "name": "balance",
      "timesteps": 1000000,
      "avg_reward": 45.2,
      "std_reward": 5.1,
      "training_time_seconds": 2535,
      "training_time": "0:42:15"
    },
    "2": { ... },
    "3": { ... }
  },
  "total_timesteps": 6000000,
  "total_training_time_seconds": 13127,
  "total_training_time": "3:38:47",
  "final_avg_reward": 118.37
}
```

## How Results Are Generated

The Jupyter notebooks in `notebooks/` automatically generate `summary.json` and
copy stage GIFs into this directory at the end of each training run. See the
"Save Results" section in each notebook.

## Adding Results Manually

If training outside the notebooks (e.g., via `scripts/train_sb3.py`), you can
create the `summary.json` by hand or use the data from `training_summary.txt`
and `evaluations.npz` in your run's log directory.
