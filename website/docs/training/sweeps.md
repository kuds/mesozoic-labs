---
sidebar_position: 4
---

# Hyperparameter Sweeps

This page explains how to find the best hyperparameters for each curriculum stage without running one job per combination. For GCP setup, Docker image building, and machine type selection, see [Training on Vertex AI](vertex-ai.md).

## The Short Answer: One Command

Yes — there is a single command that sweeps hyperparameters across **all three curriculum stages** end-to-end:

```bash
python -m environments.shared.scripts.sweep launch-all \
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
python -m environments.shared.scripts.sweep launch-all \
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
python -m environments.shared.scripts.sweep launch \
  --species velociraptor --stage 2 --algorithm ppo \
  --project YOUR_GCP_PROJECT \
  --bucket YOUR_GCS_BUCKET \
  --image ${IMAGE_URI} \
  --trials 20 --parallel 5 \
  --timesteps 1000000
```

`launch` submits the job and returns immediately (non-blocking). Use this when you want to monitor the job interactively or script your own stage-chaining logic.

## Running a Stage 1 Trial

This section walks through running a trial sweep specifically for **Stage 1 (Balance)** — useful for validating your setup or exploring Stage 1 hyperparameters before committing to a full three-stage sweep.

### Stage 1 search space

The `configs/sweep_ppo.json` file defines these Stage 1-specific parameters:

| Parameter | Type | Range / Values | Description |
|---|---|---|---|
| `ppo_learning_rate` | log-uniform | `1e-5` to `3e-4` | Learning rate |
| `ppo_ent_coef` | log-uniform | `1e-4` to `0.05` | Entropy coefficient |
| `ppo_batch_size` | discrete | `64, 128, 256, 512` | Mini-batch size |
| `ppo_gamma` | uniform | `0.97` to `0.999` | Discount factor |
| `ppo_n_steps` | discrete | `1024, 2048, 4096` | Steps per rollout |
| `ppo_net_arch` | categorical | `small, medium, large, deep, tapered, deep_tapered` | Network architecture preset |
| `env_alive_bonus` | uniform | `1.0` to `5.0` | Reward for staying alive |
| `env_posture_weight` | uniform | `0.5` to `3.0` | Upright posture reward weight |
| `env_nosedive_weight` | uniform | `0.5` to `3.0` | Falling-forward penalty weight |

Stage 1 sweeps balance-specific reward weights (`alive_bonus`, `posture_weight`, `nosedive_weight`) and the network architecture. The winning `ppo_net_arch` is automatically propagated to Stages 2 and 3 when using `launch-all`.

### How a trial executes

When Vertex AI launches each HPT trial worker, this is what happens end-to-end:

1. **Vertex AI injects hyperparameters** as CLI args (e.g. `--ppo_learning_rate 0.0003 --env_alive_bonus 2.5`).
2. **`trial.py` converts** HPT-style args to `--override` format (`ppo.learning_rate=0.0003`, `env.alive_bonus=2.5`).
3. **Stage config is loaded** from `configs/<species>/stage1_balance.toml` with overrides applied.
4. **Training runs** for the configured timesteps. Every `--eval-freq` steps (default: 10,000), the agent is evaluated and `best_mean_reward` is tracked.
5. **`cloudml-hypertune` reports** `best_mean_reward` back to Vertex AI when training finishes.
6. **Vertex AI updates** its Bayesian model and selects hyperparameters for the next trial.

Each trial writes checkpoints to `gs://YOUR_BUCKET/sweeps/<species>/stage1/<trial_id>/models/best_model.zip`.

### Budget-friendly trial run

For an initial test with minimal cost (~$0.35 total), run a small sweep on CPU-only machines:

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

Short accelerator aliases are accepted: `T4` → `NVIDIA_TESLA_T4`, `V100` → `NVIDIA_TESLA_V100`, `A100` → `NVIDIA_TESLA_A100`, `L4` → `NVIDIA_L4`. Use `None` for CPU-only.

This runs 5 trials (3 in parallel) at 100K steps each. Stage 1 (balance) is CPU-bound and doesn't need a GPU. Once you're confident the pipeline works, scale up to 20 trials at 500K steps with `--search-space-file configs/sweep_ppo.json`.

### Full Stage 1 sweep

```bash
python -m environments.shared.scripts.sweep launch \
  --species velociraptor \
  --stage 1 \
  --algorithm ppo \
  --project ${PROJECT_ID} \
  --bucket YOUR_BUCKET_NAME \
  --image ${IMAGE_URI} \
  --search-space-file configs/sweep_ppo.json \
  --machine-type n1-standard-8 \
  --accelerator-type NVIDIA_TESLA_T4
```

The file provides all job settings (trials, timesteps, parallel count) and search space parameters for Stage 1. CLI flags override file settings if both are provided.

### Retrieving Stage 1 results

```bash
# Download the results CSV
gsutil cp gs://YOUR_BUCKET/sweeps/velociraptor/_stage1_results.csv ./

# List completed trial checkpoints
gsutil ls gs://YOUR_BUCKET/sweeps/velociraptor/stage1/

# Download the best model from a specific trial
gsutil cp gs://YOUR_BUCKET/sweeps/velociraptor/stage1/TRIAL_ID/models/best_model.zip ./
```

For prerequisites (GCP project, Docker image, GCS bucket), see [Training on Vertex AI](vertex-ai.md#prerequisites).

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
python -m environments.shared.scripts.sweep launch-all \
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
python -m environments.shared.scripts.sweep launch-all \
  --species trex --algorithm ppo \
  --project YOUR_PROJECT --bucket YOUR_BUCKET --image IMAGE_URI \
  --trials 20 --trials-stage1 10 --parallel 5 \
  --search-space-file configs/sweep_ppo.json
```

Example per-stage file (`configs/sweep_ppo.json`):

```json
{
  "stage1": {
    "trials": 10,
    "timesteps": 500000,
    "parallel": 5,
    "n_envs": 4,
    "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef":      {"type": "double", "min": 1e-4, "max": 0.05, "scale": "log"},
    "ppo_batch_size":    {"type": "discrete", "values": [64, 128, 256, 512]},
    "ppo_gamma":         {"type": "double", "min": 0.97, "max": 0.999, "scale": "linear"},
    "ppo_n_steps":       {"type": "discrete", "values": [1024, 2048, 4096]},
    "ppo_net_arch":      {"type": "categorical", "values": ["small", "medium", "large", "deep", "tapered", "deep_tapered"]},
    "env_alive_bonus":   {"type": "double", "min": 1.0, "max": 5.0, "scale": "linear"},
    "env_posture_weight":  {"type": "double", "min": 0.5, "max": 3.0, "scale": "linear"},
    "env_nosedive_weight": {"type": "double", "min": 0.5, "max": 3.0, "scale": "linear"}
  },
  "stage2": {
    "trials": 20,
    "timesteps": 1000000,
    "parallel": 5,
    "n_envs": 4,
    "ppo_learning_rate":            {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef":                 {"type": "double", "min": 1e-4, "max": 0.05, "scale": "log"},
    "ppo_batch_size":               {"type": "discrete", "values": [64, 128, 256, 512]},
    "ppo_gamma":                    {"type": "double", "min": 0.97, "max": 0.999, "scale": "linear"},
    "ppo_n_steps":                  {"type": "discrete", "values": [1024, 2048, 4096]},
    "env_alive_bonus":              {"type": "double", "min": 0.5, "max": 3.0, "scale": "linear"},
    "curriculum_warmup_timesteps":  {"type": "discrete", "values": [50000, 100000, 200000, 300000]},
    "curriculum_warmup_clip_range": {"type": "double", "min": 0.01, "max": 0.05, "scale": "linear"},
    "curriculum_warmup_ent_coef":   {"type": "double", "min": 0.005, "max": 0.05, "scale": "log"},
    "curriculum_ramp_timesteps":    {"type": "discrete", "values": [200000, 500000, 1000000, 2000000]},
    "curriculum_ramp_start_value":  {"type": "double", "min": 0.05, "max": 0.3, "scale": "linear"}
  },
  "stage3": {
    "trials": 20,
    "timesteps": 1500000,
    "parallel": 5,
    "n_envs": 4,
    "ppo_learning_rate":            {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef":                 {"type": "double", "min": 1e-4, "max": 0.05, "scale": "log"},
    "ppo_batch_size":               {"type": "discrete", "values": [64, 128, 256, 512]},
    "ppo_gamma":                    {"type": "double", "min": 0.97, "max": 0.999, "scale": "linear"},
    "ppo_n_steps":                  {"type": "discrete", "values": [1024, 2048, 4096]},
    "env_strike_bonus":             {"type": "double", "min": 10.0, "max": 100.0, "scale": "log"},
    "env_strike_approach_weight":   {"type": "double", "min": 1.0, "max": 5.0, "scale": "linear"},
    "env_strike_proximity_weight":  {"type": "double", "min": 0.1, "max": 1.0, "scale": "linear"},
    "curriculum_warmup_timesteps":  {"type": "discrete", "values": [50000, 100000, 200000, 300000]},
    "curriculum_warmup_clip_range": {"type": "double", "min": 0.01, "max": 0.05, "scale": "linear"},
    "curriculum_warmup_ent_coef":   {"type": "double", "min": 0.005, "max": 0.05, "scale": "log"},
    "curriculum_ramp_timesteps":    {"type": "discrete", "values": [200000, 500000, 1000000]},
    "curriculum_ramp_start_value":  {"type": "double", "min": 0.05, "max": 0.3, "scale": "linear"}
  }
}
```

Each stage block can include both **job settings** and **search space parameters**. The parser distinguishes them automatically: entries with a `"type"` key are search space parameters; scalar values (`trials`, `timesteps`, `parallel`, `n_envs`) are job settings.

| Setting | Description | Default |
|---|---|---|
| `trials` | Max number of HPT trials for this stage | `--trials` CLI flag (20) |
| `timesteps` | Training timesteps per trial | `--timesteps-stageN` CLI flag |
| `parallel` | Concurrent trials | `--parallel` CLI flag (5) |
| `n_envs` | Parallel environments per trial worker | `--n-envs` CLI flag (4) |

CLI flags always override file settings. This means you can set your baseline config in the file and tweak individual values from the command line without editing JSON:

```bash
# File says trials=10 for stage 1, but override to 15 from the CLI
python -m environments.shared.scripts.sweep launch-all \
  --species trex --algorithm ppo \
  --project YOUR_PROJECT --bucket YOUR_BUCKET --image IMAGE_URI \
  --search-space-file configs/sweep_ppo.json \
  --trials-stage1 15
```

Notice that each stage sweeps different reward signals: Stage 1 sweeps `env_alive_bonus`, `env_posture_weight`, and `env_nosedive_weight` (the key balance rewards), Stage 2 sweeps `env_alive_bonus` alongside curriculum schedule parameters, and Stage 3 replaces those with `env_strike_bonus`, `env_strike_approach_weight`, and `env_strike_proximity_weight` (where the strike reward dominates). `ppo_net_arch` is only swept in Stage 1 — the winning architecture is automatically propagated to stages 2 and 3. A flat file (no `stageN` keys) applies the same space to all stages.

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
python -m environments.shared.scripts.sweep trial \
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
python -m environments.shared.scripts.sweep launch-all \
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

## Resuming a Sweep

`launch-all` automatically saves progress after each stage completes. If the process is interrupted (network failure, quota exhaustion, machine crash), re-run the **exact same command** and the completed stages will be skipped:

```bash
# First run — gets interrupted during Stage 2
python -m environments.shared.scripts.sweep launch-all \
    --species velociraptor --algorithm ppo \
    --project MY_PROJECT --bucket MY_BUCKET --image IMAGE_URI \
    --trials 20 --parallel 5

# Re-run — Stage 1 is skipped, resumes from Stage 2
python -m environments.shared.scripts.sweep launch-all \
    --species velociraptor --algorithm ppo \
    --project MY_PROJECT --bucket MY_BUCKET --image IMAGE_URI \
    --trials 20 --parallel 5
```

State is saved to both a local file (`sweep_state_<species>_<algorithm>.json`) and GCS (`gs://<bucket>/sweeps/<species>/_sweep_state.json`). On resume, GCS is checked first, then the local file.

To **start fresh** and ignore any saved state, pass `--no-resume`:

```bash
python -m environments.shared.scripts.sweep launch-all \
    --species velociraptor --algorithm ppo \
    --project MY_PROJECT --bucket MY_BUCKET --image IMAGE_URI \
    --trials 20 --parallel 5 --no-resume
```

## Collecting Results

After a sweep completes, use `collect-results` to scan all trial directories and produce a combined CSV:

```bash
python -m environments.shared.scripts.sweep collect-results \
  gs://YOUR_BUCKET/sweeps/velociraptor
```

This downloads every `metrics.json` and `stage_config.json` from the GCS prefix, evaluates pass/fail against curriculum thresholds, and writes a CSV to the same directory (`collected_results.csv` by default).

### Options

| Flag | Description | Default |
|---|---|---|
| `--csv PATH` | Output CSV path | `<output_dir>/collected_results.csv` |
| `--species NAME` | Species name (log messages & plot titles) | directory name |
| `--algorithm NAME` | Algorithm name (plot titles) | `unknown` |
| `--stages N [N ...]` | Only collect these stage numbers | all stages found |
| `--plot` | Generate PNG visualisation graphs | off |

### Example with plots

```bash
python -m environments.shared.scripts.sweep collect-results \
  gs://YOUR_BUCKET/sweeps/velociraptor \
  --species velociraptor \
  --algorithm ppo \
  --plot
```

This produces:

- **collected_results.csv** — one row per trial with hyperparameters, metrics (`best_mean_reward`, `best_mean_episode_length`, etc.), curriculum thresholds, and a `stage_passed` flag.
- **sweep_trial_metrics.png** — 2x2 grid: reward per trial, episode length per trial, training stability scatter, and pass/fail summary.
- **sweep_hyperparameter_analysis.png** — scatter plots of each hyperparameter vs `best_mean_reward`, colour-coded by stage.

### Expected directory structure

The command looks for `stage<N>/` sub-directories under the path you provide:

```
gs://YOUR_BUCKET/sweeps/velociraptor/     ← pass this path
  stage1/
    1/metrics.json
    2/metrics.json
    ...
  stage2/
    1/metrics.json
    ...
```

Single-trial (curriculum) layouts where `metrics.json` sits directly in each `stage<N>/` directory are also supported.

### Filtering by stage

```bash
# Only collect results for stages 1 and 2
python -m environments.shared.scripts.sweep collect-results \
  gs://YOUR_BUCKET/sweeps/velociraptor \
  --stages 1 2
```

## Resource Errors and Retries

When submitting a job to Vertex AI, transient errors (quota exhaustion, service unavailability) are automatically retried up to 3 times with increasing delays (60 s, 180 s, 300 s). If all retries fail:

1. Progress for already-completed stages is saved.
2. The script exits with a clear error message.
3. Re-running the same command resumes from where it left off.

Non-transient errors (invalid parameters, authentication failures) are **not** retried and fail immediately.

