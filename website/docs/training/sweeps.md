---
sidebar_position: 4
---

# Hyperparameter Sweeps

This page explains how to find the best hyperparameters for each curriculum stage without running one job per combination.

## The Short Answer: One Command

Yes — there is a single command that sweeps hyperparameters across **all three curriculum stages** end-to-end:

```bash
python environments/shared/scripts/sweep.py launch-all \
  --species velociraptor --algorithm ppo \
  --project YOUR_GCP_PROJECT \
  --bucket YOUR_GCS_BUCKET \
  --image ${IMAGE_URI} \
  --trials 20 --parallel 5 \
  --timesteps-stage1 500000 \
  --timesteps-stage2 1000000 \
  --timesteps-stage3 1500000
```

`launch-all` orchestrates the full three-stage sweep automatically:

1. Submits a Stage 1 Hyperparameter Tuning job and **waits** for it to complete.
2. Identifies the best Stage 1 trial (by `best_mean_reward`).
3. Submits a Stage 2 sweep, **automatically passing the best Stage 1 checkpoint** as the warm-start model.
4. Identifies the best Stage 2 trial.
5. Submits a Stage 3 sweep, loading the best Stage 2 checkpoint.

You submit one command and come back when it's done — no manual chaining required.

## The Strategy: Why Sweep Stages Sequentially

Each stage builds on what the previous stage learned, so the optimal hyperparameters for Stage 2 depend on having a good Stage 1 policy. Sweeping all three stages simultaneously would be wasteful — Stage 2 hyperparameters don't matter much if Stage 1 policy was poor.

The `launch-all` command reduces the problem from an exponential number of combinations to three sequential Bayesian searches, each using the best model from the previous stage:

1. **Stage 1 sweep** — Run N parallel trials, each with different hyperparameters. Vertex AI uses Bayesian optimisation to find the best settings for balance.
2. **Auto-chain** — `launch-all` finds the best Stage 1 trial and passes its checkpoint to Stage 2 automatically.
3. **Stage 2 sweep** — Load the best Stage 1 model, then sweep Stage 2 hyperparameters for locomotion.
4. **Auto-chain** — `launch-all` finds the best Stage 2 trial and passes its checkpoint to Stage 3.
5. **Stage 3 sweep** — Load the best Stage 2 model, then sweep Stage 3 hyperparameters for behavior (strike/bite/food).

Each stage uses [Vertex AI Hyperparameter Tuning](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview) — N parallel trials with Bayesian optimisation, not grid search, so you get good coverage in 20–30 trials instead of hundreds.

## Quick Start

### 1. Build and push the Docker image

```bash
export PROJECT_ID=$(gcloud config get project)
export REGION=us-central1
export IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/mesozoic-labs/trainer:latest

docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

### 2. Launch the all-stages sweep

```bash
python environments/shared/scripts/sweep.py launch-all \
  --species velociraptor --algorithm ppo \
  --project YOUR_GCP_PROJECT \
  --bucket YOUR_GCS_BUCKET \
  --image ${IMAGE_URI} \
  --trials 20 --parallel 5 \
  --timesteps-stage1 500000 \
  --timesteps-stage2 1000000 \
  --timesteps-stage3 1500000
```

Vertex AI runs 20 trials per stage (5 in parallel), waiting for each stage to finish before starting the next. The final best checkpoints from each stage are saved to:

```
gs://YOUR_BUCKET/sweeps/velociraptor/stage1/<best_trial_id>/models/stage1_final.zip
gs://YOUR_BUCKET/sweeps/velociraptor/stage2/<best_trial_id>/models/stage2_final.zip
gs://YOUR_BUCKET/sweeps/velociraptor/stage3/<best_trial_id>/models/stage3_final.zip
```

### 3. Monitor progress

```bash
# List all HPT jobs in your project
gcloud ai hp-tuning-jobs list --region=us-central1

# Or view in the Console:
# https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs
```

Because `launch-all` runs synchronously (each stage blocks until the previous is done), you can monitor three sequential jobs appearing one after another in the console.

> **Long sweeps (>24 hours):** Since `launch-all` blocks until all stages complete, use a persistent environment like a GCE VM with `tmux` instead of a notebook. See [Running Long Sweeps from a GCE VM](vertex-ai.md#running-long-sweeps-from-a-gce-vm) for step-by-step setup.

## Single-Stage Sweep

If you want to sweep only one stage — for example, to re-sweep Stage 2 after finding better Stage 1 weights — use `launch` instead:

```bash
python environments/shared/scripts/sweep.py launch \
  --species velociraptor --stage 2 --algorithm ppo \
  --project YOUR_GCP_PROJECT \
  --bucket YOUR_GCS_BUCKET \
  --image ${IMAGE_URI} \
  --trials 20 --parallel 5 \
  --timesteps 1000000
```

`launch` submits the job and returns immediately (non-blocking). Use this when you want to monitor the job interactively or script your own stage-chaining logic.

## Default Search Spaces

### PPO

| Parameter | Type | Range / Values |
|---|---|---|
| `ppo_learning_rate` | log-uniform | `1e-5` to `3e-4` |
| `ppo_ent_coef` | log-uniform | `1e-4` to `0.05` |
| `ppo_batch_size` | discrete | `64, 128, 256, 512` |
| `ppo_gamma` | uniform | `0.97` to `0.999` |
| `ppo_n_steps` | discrete | `1024, 2048, 4096` |

### SAC

| Parameter | Type | Range / Values |
|---|---|---|
| `sac_learning_rate` | log-uniform | `1e-5` to `3e-4` |
| `sac_batch_size` | discrete | `128, 256, 512` |
| `sac_gamma` | uniform | `0.97` to `0.999` |

## Customising the Search Space

There are two ways to customise the search space: inline JSON or a JSON file.

### Inline JSON (same space for all stages)

Pass a JSON string to `--search-space` to override the defaults. This applies the same search space to all stages:

```bash
python environments/shared/scripts/sweep.py launch-all \
  --species trex --algorithm ppo \
  --project YOUR_PROJECT --bucket YOUR_BUCKET --image IMAGE_URI \
  --trials 30 --parallel 5 \
  --search-space '{
    "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef":      {"type": "double", "min": 0.001, "max": 0.05, "scale": "log"},
    "ppo_batch_size":    {"type": "discrete", "values": [64, 128, 256]},
    "env_alive_bonus":   {"type": "double", "min": 1.0, "max": 5.0, "scale": "linear"}
  }'
```

### JSON file with per-stage search spaces (recommended)

Use `--search-space-file` to load the search space from a JSON file. The file can define different parameters per stage using `"stage1"`, `"stage2"`, `"stage3"` top-level keys:

```bash
python environments/shared/scripts/sweep.py launch-all \
  --species trex --algorithm ppo \
  --project YOUR_PROJECT --bucket YOUR_BUCKET --image IMAGE_URI \
  --trials 20 --trials-stage1 10 --parallel 5 \
  --search-space-file configs/sweep_ppo.json
```

Example per-stage file (`configs/sweep_ppo.json`):

```json
{
  "stage1": {
    "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef":      {"type": "double", "min": 1e-4, "max": 0.05, "scale": "log"},
    "ppo_batch_size":    {"type": "discrete", "values": [64, 128, 256, 512]},
    "ppo_gamma":         {"type": "double", "min": 0.97, "max": 0.999, "scale": "linear"},
    "ppo_n_steps":       {"type": "discrete", "values": [1024, 2048, 4096]},
    "ppo_net_arch":      {"type": "categorical", "values": ["small", "medium", "large", "deep"]},
    "env_alive_bonus":   {"type": "double", "min": 1.0, "max": 5.0, "scale": "linear"}
  },
  "stage2": {
    "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef":      {"type": "double", "min": 1e-4, "max": 0.05, "scale": "log"},
    "ppo_batch_size":    {"type": "discrete", "values": [64, 128, 256, 512]},
    "ppo_gamma":         {"type": "double", "min": 0.97, "max": 0.999, "scale": "linear"},
    "ppo_n_steps":       {"type": "discrete", "values": [1024, 2048, 4096]},
    "ppo_net_arch":      {"type": "categorical", "values": ["small", "medium", "large", "deep"]},
    "env_alive_bonus":   {"type": "double", "min": 0.5, "max": 3.0, "scale": "linear"}
  },
  "stage3": {
    "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef":      {"type": "double", "min": 1e-4, "max": 0.05, "scale": "log"},
    "ppo_batch_size":    {"type": "discrete", "values": [64, 128, 256, 512]},
    "ppo_gamma":         {"type": "double", "min": 0.97, "max": 0.999, "scale": "linear"},
    "ppo_n_steps":       {"type": "discrete", "values": [1024, 2048, 4096]},
    "ppo_net_arch":      {"type": "categorical", "values": ["small", "medium", "large", "deep"]}
  }
}
```

Notice that `env_alive_bonus` is swept in stages 1-2 (where it's a meaningful reward signal) but omitted from stage 3 (where the bite/strike bonus dominates). A flat file (no `stageN` keys) applies the same space to all stages.

Pre-built search space files for PPO and SAC are included in the repo at `configs/sweep_ppo.json` and `configs/sweep_sac.json`.

### Parameter types

| Type | JSON fields | Example |
|---|---|---|
| `double` | `min`, `max`, `scale` (`"log"` or `"linear"`) | `{"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"}` |
| `discrete` | `values` (list of numbers) | `{"type": "discrete", "values": [64, 128, 256]}` |
| `categorical` | `values` (list of strings) | `{"type": "categorical", "values": ["small", "medium"]}` |

### Parameter naming convention

- `ppo_X` → sets `ppo.X` in the config (e.g. `ppo_learning_rate`)
- `sac_X` → sets `sac.X` in the config (e.g. `sac_batch_size`)
- `env_X` → sets `env.X` in the config (e.g. `env_alive_bonus`)

## Testing Locally Before Launching

Run a single trial locally to verify the setup before burning cloud credits:

```bash
# Quick smoke-test: 10 000 steps, 1 env, specific hyperparameters
python environments/shared/scripts/sweep.py trial \
  --species velociraptor --stage 1 --algorithm ppo \
  --timesteps 10000 --n-envs 1 \
  --ppo_learning_rate 1e-4 --ppo_ent_coef 0.02 --ppo_batch_size 128
```

The `trial` subcommand is also what each Vertex AI worker runs — it accepts the HPT-injected `--ppo_learning_rate X` style args and converts them to `--override ppo.learning_rate=X` automatically.

## Stage-Scoped Overrides with `--override`

Once you've found the best configs via `launch-all`, you can lock them in for production runs using the stage-scoped override syntax:

```bash
# Lock stages at best-found values for a final production run
python environments/velociraptor/scripts/train_sb3.py curriculum \
  --algorithm ppo \
  --override 1.ppo.learning_rate=3e-4 1.ppo.ent_coef=0.005 \
             2.ppo.learning_rate=1e-4 2.ppo.ent_coef=0.01 \
             3.ppo.learning_rate=5e-5 3.ppo.ent_coef=0.001
```

The `N.section.key=value` format targets a single stage; plain `section.key=value` applies to all stages. Both formats can be mixed in the same `--override` list.

## W&B Integration

Add `--wandb` to log all trials to Weights & Biases. Each trial appears as a separate run so you can compare them side-by-side on the W&B dashboard:

```bash
python environments/shared/scripts/sweep.py launch-all \
  --species velociraptor --algorithm ppo \
  --project YOUR_PROJECT --bucket YOUR_BUCKET --image IMAGE_URI \
  --trials 20 --parallel 5 \
  --wandb
```

Add `WANDB_API_KEY` as an environment variable in your Docker image or as a GCP Secret (see [Vertex AI guide](vertex-ai.md)).

## How the Metric Flows to Vertex AI

Each trial's `train()` call reports `best_mean_reward` (the highest mean evaluation reward seen during training) to Vertex AI HPT via `cloudml-hypertune`. `launch-all` reads these metrics from the completed job to identify the best trial:

```
trial training loop
    └─ EvalCallback (every --eval-freq steps)
           └─ records best_mean_reward
    └─ train() finishes
           └─ hypertune.HyperTune().report_hyperparameter_tuning_metric(
                  "best_mean_reward", eval_callback.best_mean_reward
              )
           └─ Vertex AI reads this and updates the Bayesian model

launch-all (after stage N completes)
    └─ reads hpt_job.trials
    └─ picks trial with highest best_mean_reward
    └─ constructs checkpoint path: /gcs/<bucket>/sweeps/<species>/stageN/<trial_id>/models/stageN_final.zip
    └─ passes it as --load to stage N+1 trials
```

Vertex AI uses trial results to decide which hyperparameter regions to explore next. Trials in promising areas get more follow-up trials; poor regions are avoided. This is why Bayesian optimisation needs far fewer trials than grid search.

## Recommended Trial Counts

| Stage | Recommended Trials | Parallel | Why |
|---|---|---|---|
| Stage 1 (Balance) | 20–30 | 5 | Simple task, converges quickly — need broad coverage |
| Stage 2 (Locomotion) | 20–30 | 5 | Medium complexity, load Stage 1 weights |
| Stage 3 (Behavior) | 15–20 | 5 | Complex sparse rewards — fewer trials needed since Stage 1+2 are locked |

## Cost Estimate

Each trial trains for the configured number of timesteps on an `n1-standard-8 + T4` machine (approximate costs as of early 2026 in `us-central1`; check [current GCP pricing](https://cloud.google.com/vertex-ai/pricing) before running large sweeps — costs vary by region and machine type):

| Timesteps per trial | Trial cost | 20 trials (5 parallel) | Total wall time |
|---|---|---|---|
| 100 000 | ~$0.07 | ~$1.40 | ~1 hour |
| 500 000 | ~$0.37 | ~$7.40 | ~5 hours |
| 1 000 000 | ~$0.73 | ~$14.60 | ~10 hours |

For a full `launch-all` (3 stages at 500k/1M/1.5M steps per trial, 20 trials each): roughly **$45–60 total** for a complete sweep.

**Tip:** Start with 100 000–200 000 timesteps per trial to get a rough ranking, then run longer trials for the top 3–5 configurations.

