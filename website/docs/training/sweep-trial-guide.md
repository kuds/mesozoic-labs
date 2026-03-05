---
sidebar_position: 5
---

# Running a Stage 1 Sweep Trial (Vertex AI HPT)

This guide walks through running a trial of the sweep script for **Stage 1 (Balance)** using Vertex AI Hyperparameter Tuning (HPT). It covers local validation, single-stage cloud submission, and monitoring.

## Overview

The sweep system uses three modes:

| Mode | Command | Purpose |
|---|---|---|
| `trial` | `python -m environments.shared.scripts.sweep trial` | Run a single trial locally or inside a Vertex AI worker |
| `launch` | `python -m environments.shared.scripts.sweep launch` | Submit a single-stage HPT job to Vertex AI |
| `launch-all` | `python -m environments.shared.scripts.sweep launch-all` | Sweep all three stages end-to-end |

For a Stage 1 trial, you will use `trial` (local testing) and `launch` (cloud submission).

## Prerequisites

1. **GCP project** with billing enabled
2. **Google Cloud CLI** (`gcloud`) installed and authenticated
3. **Docker** installed locally
4. **APIs enabled:**
   ```bash
   gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com
   ```
5. **Artifact Registry repository:**
   ```bash
   gcloud artifacts repositories create mesozoic-labs \
     --repository-format=docker \
     --location=us-central1 \
     --description="Mesozoic Labs training containers"
   ```
6. **GCS bucket:**
   ```bash
   gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=us-central1
   ```

> **Shortcut:** Run `bash scripts/setup_vertex_ai.sh` to set up all prerequisites interactively.

## Step 1: Run a Local Smoke Test

Before using cloud resources, verify the training pipeline works locally with a short trial:

```bash
python -m environments.shared.scripts.sweep trial \
  --species velociraptor \
  --stage 1 \
  --algorithm ppo \
  --timesteps 10000 \
  --n-envs 1 \
  --ppo_learning_rate 1e-4 \
  --ppo_ent_coef 0.02 \
  --ppo_batch_size 128
```

This runs 10,000 timesteps with specific hyperparameters. It validates that:

- The Stage 1 environment loads correctly
- The training loop runs without errors
- Checkpoints are saved to `./models/best_model.zip`
- Metrics (`best_mean_reward`) are reported

The `trial` subcommand accepts Vertex AI HPT-style args (`--ppo_learning_rate 0.0003`) and converts them to `--override` format internally (`ppo.learning_rate=0.0003`).

## Step 2: Build and Push the Docker Image

```bash
export PROJECT_ID=$(gcloud config get project)
export REGION=us-central1
export IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/mesozoic-labs/trainer:latest

# Authenticate Docker with Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push
docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

### Verify the container locally (optional)

```bash
docker run --rm ${IMAGE_URI} \
  environments/velociraptor/scripts/train_sb3.py \
  train --stage 1 --timesteps 1000
```

## Step 3: Launch a Stage 1 Sweep on Vertex AI

### Option A: Using the default search space

The default PPO search space sweeps learning rate, entropy coefficient, batch size, gamma, and n_steps:

```bash
python -m environments.shared.scripts.sweep launch \
  --species velociraptor \
  --stage 1 \
  --algorithm ppo \
  --timesteps 500000 \
  --n-envs 4 \
  --project ${PROJECT_ID} \
  --bucket YOUR_BUCKET_NAME \
  --image ${IMAGE_URI} \
  --trials 20 \
  --parallel 5 \
  --machine-type n1-standard-8 \
  --accelerator-type NVIDIA_TESLA_T4
```

### Option B: Using the per-stage search space file (recommended)

The `configs/sweep_ppo.json` file includes Stage 1-specific parameters including environment reward weights and network architecture:

```bash
python -m environments.shared.scripts.sweep launch \
  --species velociraptor \
  --stage 1 \
  --algorithm ppo \
  --project ${PROJECT_ID} \
  --bucket YOUR_BUCKET_NAME \
  --image ${IMAGE_URI} \
  --search-space-file configs/sweep_ppo.json
```

The file defines trials, timesteps, parallel count, and the full search space per stage. CLI flags override file settings when both are provided.

### Option C: Inline custom search space

```bash
python -m environments.shared.scripts.sweep launch \
  --species velociraptor \
  --stage 1 \
  --algorithm ppo \
  --timesteps 500000 \
  --project ${PROJECT_ID} \
  --bucket YOUR_BUCKET_NAME \
  --image ${IMAGE_URI} \
  --trials 10 \
  --parallel 5 \
  --search-space '{
    "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef":      {"type": "double", "min": 1e-4, "max": 0.05, "scale": "log"},
    "ppo_batch_size":    {"type": "discrete", "values": [64, 128, 256]},
    "env_alive_bonus":   {"type": "double", "min": 1.0, "max": 5.0, "scale": "linear"}
  }'
```

## Stage 1 Search Space

The `configs/sweep_ppo.json` file defines these Stage 1 parameters:

| Parameter | Type | Range / Values | Description |
|---|---|---|---|
| `ppo_learning_rate` | log-uniform | `1e-5` to `3e-4` | Learning rate |
| `ppo_ent_coef` | log-uniform | `1e-4` to `0.05` | Entropy coefficient |
| `ppo_batch_size` | discrete | `64, 128, 256, 512` | Mini-batch size |
| `ppo_gamma` | uniform | `0.97` to `0.999` | Discount factor |
| `ppo_n_steps` | discrete | `1024, 2048, 4096` | Steps per rollout |
| `ppo_net_arch` | categorical | `small, medium, large, deep, tapered, deep_tapered` | Network architecture |
| `env_alive_bonus` | uniform | `1.0` to `5.0` | Reward for staying alive |
| `env_posture_weight` | uniform | `0.5` to `3.0` | Upright posture reward weight |
| `env_nosedive_weight` | uniform | `0.5` to `3.0` | Falling-forward penalty weight |

Vertex AI uses **Bayesian optimization** to explore this space, maximizing `best_mean_reward`.

## How a Trial Executes

When Vertex AI launches a trial worker, the following happens:

1. **Vertex AI injects hyperparameters** as CLI args (e.g. `--ppo_learning_rate 0.0003 --env_alive_bonus 2.5`)
2. **`trial.py` converts** these to `--override` format (`ppo.learning_rate=0.0003`, `env.alive_bonus=2.5`)
3. **Stage 1 config** is loaded from `configs/<species>/stage1_balance.toml` with overrides applied
4. **Training runs** for the configured timesteps with evaluation every `--eval-freq` steps
5. **`cloudml-hypertune`** reports `best_mean_reward` back to Vertex AI
6. **Vertex AI updates** its Bayesian model and selects hyperparameters for the next trial

Each trial writes checkpoints to:
```
gs://YOUR_BUCKET/sweeps/<species>/stage1/<trial_id>/models/best_model.zip
```

## Step 4: Monitor the Sweep

### From the CLI

```bash
# List HPT jobs
gcloud ai hp-tuning-jobs list --region=us-central1

# Get details on a specific job
gcloud ai hp-tuning-jobs describe JOB_ID --region=us-central1
```

### From the Vertex AI Console

Navigate to:
```
https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs
```

### From the sweep script output

The `launch` command prints status updates while polling (every 120 seconds by default), showing elapsed time, trial count, and best reward found so far.

## Step 5: Retrieve Results

### Download the results CSV

```bash
gsutil cp gs://YOUR_BUCKET/sweeps/velociraptor/_stage1_results.csv ./
```

### Download the best model

```bash
# List completed trials
gsutil ls gs://YOUR_BUCKET/sweeps/velociraptor/stage1/

# Download a specific trial's model
gsutil cp gs://YOUR_BUCKET/sweeps/velociraptor/stage1/TRIAL_ID/models/best_model.zip ./
```

## Cost Estimate

Approximate costs for Stage 1 on `n1-standard-8 + T4` in `us-central1`:

| Timesteps per trial | Cost per trial | 20 trials (5 parallel) | Wall time |
|---|---|---|---|
| 100,000 | ~$0.07 | ~$1.40 | ~1 hour |
| 500,000 | ~$0.37 | ~$7.40 | ~5 hours |

**Tip:** Start with 100,000–200,000 timesteps per trial to get a rough ranking, then run longer trials for the top configurations.

Stage 1 can also run without a GPU (`--accelerator-type None`) on `n1-standard-8` since the balance task is simple and CPU-bound.

## Running a Small Trial (Budget-Friendly)

For an initial trial run with minimal cost:

```bash
python -m environments.shared.scripts.sweep launch \
  --species velociraptor \
  --stage 1 \
  --algorithm ppo \
  --timesteps 100000 \
  --n-envs 4 \
  --project ${PROJECT_ID} \
  --bucket YOUR_BUCKET_NAME \
  --image ${IMAGE_URI} \
  --trials 5 \
  --parallel 3 \
  --machine-type n1-standard-8 \
  --accelerator-type None
```

This runs 5 trials (3 in parallel) at 100K steps each on CPU-only machines. Total cost: approximately $0.35.

## Next Steps

After Stage 1 completes:

- **Inspect results** to identify the best hyperparameters and reward achieved
- **Run a full `launch-all`** to automatically chain Stage 1 → 2 → 3 with the best checkpoint propagated between stages (see [Hyperparameter Sweeps](sweeps.md))
- **Lock in best values** for production training using `--override` syntax (see [Vertex AI training](vertex-ai.md))

## Troubleshooting

| Issue | Solution |
|---|---|
| `ResourceExhausted` quota error | Retried automatically (3 attempts). Reduce `--parallel` or request quota increase. |
| MuJoCo rendering errors | Ensure the Docker image sets `MUJOCO_GL=osmesa` (the included Dockerfile handles this). |
| Out of memory | Reduce `--n-envs` or use a machine type with more RAM. |
| Job times out | Re-run the same command — completed stages are skipped automatically via sweep state persistence. |
| `ServiceUnavailable` | Transient GCP error, retried automatically. |
