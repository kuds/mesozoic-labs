The summaries below are historical experiment records generated from the versioned JSON files under
`results/`. They are not evidence for the current model revision unless provenance is marked both current
and verified. Current stage budgets may therefore differ from the steps reported here.

### Velociraptor Mongoliensis (PPO · Stable-Baselines3) — 2026-03-15

**Provenance:** Historical model; unverified; evaluation episode count not recorded; Stable-Baselines3 (version not recorded). **Run total:** 22M steps; 11:25:15. [Source summary](results/velociraptor/ppo/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 1964.43 | 0.11 m/s | — | 6M | passed retired gate (reward gate) |
| 2 — Locomotion | 2678.68 | 3.47 m/s | — | 8M | passed retired gate (reward gate) |
| 3 — Strike | 1366.19 | 2.02 m/s | 93.3% | 8M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** A left or right sickle-claw geom contacts the prey geom while the strike reward is enabled.

### Velociraptor Mongoliensis (SAC · Stable-Baselines3) — 2026-03-21

**Provenance:** Historical model; unverified; evaluation episode count not recorded; Stable-Baselines3 (version not recorded). **Run total:** 22M steps; 22:59:18. [Source summary](results/velociraptor/sac/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 970.19 | -0.64 m/s | — | 6M | passed retired gate (reward gate) |
| 2 — Locomotion | 2078.62 | 2.91 m/s | — | 8M | passed retired gate (reward gate) |
| 3 — Strike | 1195.43 | 1.63 m/s | 90.0% | 8M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** A left or right sickle-claw geom contacts the prey geom while the strike reward is enabled.

### Tyrannosaurus Rex (PPO · Stable-Baselines3) — 2026-03-18

**Provenance:** Historical model; unverified; evaluation episode count not recorded; Stable-Baselines3 (version not recorded). **Run total:** 22M steps; 13:02:32. [Source summary](results/trex/ppo/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 3008.66 | 0.02 m/s | — | 6M | passed retired gate (reward gate) |
| 2 — Locomotion | 1936.01 | 3.47 m/s | — | 8M | passed retired gate (reward gate) |
| 3 — Bite | 1294.28 | 1.68 m/s | 96.7% | 8M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** The head-bite geom contacts the prey geom while the bite reward is enabled; the model has no articulated jaw.

### Brachiosaurus Altithorax (PPO · Stable-Baselines3) — 2026-07-18

**Provenance:** Historical model; unverified; 30 evaluation episodes; Stable-Baselines3 (version not recorded). **Run total:** 34.0214M steps; 19:46:16. [Source summary](results/brachiosaurus/ppo/summary.json).

| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |
|---|---:|---:|---:|---:|---:|
| 1 — Balance | 1740.53 | 0.01 m/s | — | 6.00474M | passed retired gate (reward gate) |
| 2 — Locomotion | 6634.60 | 1.42 m/s | 3.3% | 16.0072M | passed retired gate (reward gate) |
| 3 — Food Reach | 1368.13 | 0.71 m/s | 100.0% | 12.0095M | passed retired gate (reward gate) |

**Current Stable-Baselines3 catalog definition for this task label:** The head-tip site comes within the configured food-reach threshold of the food target while the food-reach bonus is enabled.
