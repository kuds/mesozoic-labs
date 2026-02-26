---
sidebar_position: 4
---

# Hyperparameter Sweeps

This page explains how to find the best hyperparameters for each curriculum stage without running one job per combination.

## The Strategy: Sweep Stages Independently

The key insight is to **sweep one stage at a time**, not the full curriculum. This reduces the problem from an exponential number of combinations to three sequential searches:

1. **Stage 1 sweep** — Run N parallel trials, each with different hyperparameters. Vertex AI uses Bayesian optimisation to find the best settings for balance.
2. **Pick the best Stage 1 config** — Download the model checkpoint from the best trial.
3. **Stage 2 sweep** — Load the best Stage 1 model, then sweep Stage 2 hyperparameters for locomotion.
4. **Pick the best Stage 2 config** — Download that model checkpoint.
5. **Stage 3 sweep** — Load the best Stage 2 model, then sweep Stage 3 hyperparameters for behavior (strike/bite/food).

Each sweep is a [Vertex AI Hyperparameter Tuning Job](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview) that:
- Runs N parallel trials simultaneously (e.g. 5 at a time)
- Uses **Bayesian optimisation** to explore the search space efficiently — not grid search, so you get good coverage in 20–30 trials instead of hundreds
- Reports `best_mean_reward` from each trial so Vertex AI knows which ones to focus on

You only need to run **3 HPT jobs total** (one per stage) to fully sweep a species, not one job per parameter combination.

## Quick Start

### 1. Build and push the Docker image

```bash
export PROJECT_ID=$(gcloud config get project)
export REGION=us-central1
export IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/mesozoic-labs/trainer:latest

docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

### 2. Launch a Stage 1 sweep

```bash
python environments/shared/scripts/sweep.py launch \
  --species velociraptor --stage 1 --algorithm ppo \
  --project YOUR_GCP_PROJECT \
  --bucket YOUR_GCS_BUCKET \
  --image ${IMAGE_URI} \
  --trials 20 --parallel 5 \
  --timesteps 500000
```

Vertex AI submits 20 trials, running 5 in parallel. Each trial trains Stage 1 with a different hyperparameter combination chosen by the Bayesian optimiser.

### 3. Monitor and pick the best trial

```bash
# List all trials for the job (sorted by best_mean_reward)
gcloud ai hp-tuning-jobs list --region=us-central1

# Or view in the Console:
# https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs
```

The best trial's model checkpoint is saved to:
```
gs://YOUR_BUCKET/sweeps/velociraptor/stage1/models/stage1_final.zip
```

### 4. Launch a Stage 2 sweep loading the best Stage 1 model

```bash
# Download the best Stage 1 model first
gcloud storage cp \
  gs://YOUR_BUCKET/sweeps/velociraptor/stage1/models/stage1_final.zip \
  ./best_stage1.zip

# Launch Stage 2 sweep, loading the Stage 1 model as the starting point
# (Pass --load via --search-space or handle in the trial args)
python environments/shared/scripts/sweep.py launch \
  --species velociraptor --stage 2 --algorithm ppo \
  --project YOUR_GCP_PROJECT \
  --bucket YOUR_GCS_BUCKET \
  --image ${IMAGE_URI} \
  --trials 20 --parallel 5 \
  --timesteps 1000000
```

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

Pass a JSON string to `--search-space` to override the defaults. You can narrow the range, add env reward weights, or remove parameters you don't want to sweep:

```bash
python environments/shared/scripts/sweep.py launch \
  --species trex --stage 1 --algorithm ppo \
  --project YOUR_PROJECT --bucket YOUR_BUCKET --image IMAGE_URI \
  --trials 30 --parallel 5 \
  --search-space '{
    "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef":      {"type": "double", "min": 0.001, "max": 0.05, "scale": "log"},
    "ppo_batch_size":    {"type": "discrete", "values": [64, 128, 256]},
    "env_alive_bonus":   {"type": "double", "min": 1.0, "max": 5.0, "scale": "linear"}
  }'
```

Parameter naming convention:
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

Once you've found the best Stage 1 config, you can lock it in and only sweep Stage 2 by using the stage-scoped override syntax when running `curriculum`:

```bash
# Lock Stage 1 at best-found values, sweep Stage 2 manually
python environments/velociraptor/scripts/train_sb3.py curriculum \
  --algorithm ppo \
  --override 1.ppo.learning_rate=3e-4 1.ppo.ent_coef=0.005 \
             2.ppo.learning_rate=1e-4 2.ppo.ent_coef=0.01
```

The `N.section.key=value` format targets a single stage; plain `section.key=value` applies to all stages. Both formats can be mixed in the same `--override` list.

## W&B Integration

Add `--wandb` to log all trials to Weights & Biases. Each trial appears as a separate run so you can compare them side-by-side on the W&B dashboard:

```bash
python environments/shared/scripts/sweep.py launch \
  --species velociraptor --stage 1 --algorithm ppo \
  --project YOUR_PROJECT --bucket YOUR_BUCKET --image IMAGE_URI \
  --trials 20 --parallel 5 \
  --wandb
```

Add `WANDB_API_KEY` as an environment variable in your Docker image or as a GCP Secret (see [Vertex AI guide](vertex-ai.md)).

## How the Metric Flows to Vertex AI

Each trial's `train()` call reports `best_mean_reward` (the highest mean evaluation reward seen during training) to Vertex AI HPT via `cloudml-hypertune`:

```
trial training loop
    └─ EvalCallback (every --eval-freq steps)
           └─ records best_mean_reward
    └─ train() finishes
           └─ hypertune.HyperTune().report_hyperparameter_tuning_metric(
                  "best_mean_reward", eval_callback.best_mean_reward
              )
           └─ Vertex AI reads this and updates the Bayesian model
```

Vertex AI uses these results to decide which hyperparameter regions to explore next. Trials in promising areas get more follow-up trials; poor regions are avoided. This is why Bayesian optimisation needs far fewer trials than grid search.

## Recommended Trial Counts

| Stage | Recommended Trials | Parallel | Why |
|---|---|---|---|
| Stage 1 (Balance) | 20–30 | 5 | Simple task, converges quickly — need broad coverage |
| Stage 2 (Locomotion) | 20–30 | 5 | Medium complexity, load Stage 1 weights |
| Stage 3 (Behavior) | 15–20 | 5 | Complex sparse rewards — fewer trials needed since Stage 1+2 are locked |

## Cost Estimate

Each trial trains for `--timesteps` steps on an `n1-standard-8 + T4` machine (approximate costs as of early 2026 in `us-central1`; check [current GCP pricing](https://cloud.google.com/vertex-ai/pricing) before running large sweeps — costs vary by region and machine type):

| Timesteps per trial | Trial cost | 20 trials (5 parallel) | Total wall time |
|---|---|---|---|
| 100 000 | ~$0.07 | ~$1.40 | ~1 hour |
| 500 000 | ~$0.37 | ~$7.40 | ~5 hours |
| 1 000 000 | ~$0.73 | ~$14.60 | ~10 hours |

**Tip:** Start with 100 000–200 000 timesteps per trial to get a rough ranking, then run longer trials for the top 3–5 configurations.
