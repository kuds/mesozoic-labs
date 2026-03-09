# Training on Google Cloud Vertex AI

This guide covers how to run Mesozoic Labs training jobs on [Vertex AI](https://cloud.google.com/vertex-ai), Google Cloud's managed ML platform. Vertex AI lets you run the full 3-stage curriculum on cloud GPUs without managing infrastructure.

## How Multi-Stage Curriculum Works in a Single Job

**You do not need to submit one job per stage.** The `curriculum` subcommand runs all three stages end-to-end inside a single Docker container:

```
Stage 1 (Balance) → Stage 2 (Locomotion) → Stage 3 (Behavior)
        └──────────── single Vertex AI job ────────────────┘
```

### Per-stage hyperparameters

Each stage has its **own TOML config** (`configs/<species>/stage1_*.toml`, `stage2_*.toml`, `stage3_*.toml`). When the curriculum advances, the script loads that stage's config and re-initialises the model with those hyperparameters. This means you get a full hyperparameter shift at each stage automatically — no manual intervention required.

The pattern across all three species follows a deliberate progression:

| Hyperparameter | Stage 1 (Balance) | Stage 2 (Locomotion) | Stage 3 (Behavior) | Rationale |
|---|---|---|---|---|
| `learning_rate` | `3e-4` | `1e-4` | `5e-5` | Coarser search early, fine-tune late |
| `ent_coef` | `0.005–0.01` | `0.005–0.01` | `0.001` | More exploration early, exploit late |
| `clip_range` | `0.2` | `0.2` | `0.1` | Conservative updates for sparse rewards |
| `gamma` | `0.99–0.998` | `0.99` | `0.995` | More farsighted for sparse strike/bite/food |
| `n_steps` / `batch_size` | Smaller | Medium | Larger | Larger buffer for complex behaviour |

The `env_kwargs` (reward weights, `alive_bonus`, `forward_vel_weight`, etc.) also change dramatically between stages — that is the whole point of curriculum learning.

### Stage advancement logic

The `CurriculumManager` checks thresholds after every `--eval-freq` steps. Both must be exceeded for `required_consecutive` evaluations in a row before the stage advances:

| Threshold | What it checks |
|---|---|
| `min_avg_reward` | Mean episode reward over the evaluation window |
| `min_avg_episode_length` | Mean episode length over the evaluation window |

If the per-stage `timesteps` budget is exhausted before thresholds are met, the curriculum advances anyway — so the job always completes, regardless of agent performance. Checkpoints and VecNormalize stats are saved at the end of each stage.

### When to use `curriculum` vs `train`

| Command | Use when |
|---|---|
| `curriculum` | Full automated run — one job, stages 1–3 with per-stage hyperparameters applied automatically |
| `train --stage N` | Re-running a single stage, loading from a specific checkpoint, or manually controlling stage budgets |

## Quick Start with Google Cloud Shell

If you have access to [Google Cloud Shell](https://shell.cloud.google.com), you can set up everything — Artifact Registry, Docker image, GCS bucket, and optionally submit a training job — with a single interactive script:

```bash
git clone https://github.com/kuds/mesozoic-labs.git
cd mesozoic-labs
bash scripts/setup_vertex_ai.sh
```

Cloud Shell comes with `gcloud`, `docker`, and `git` pre-installed, so no local setup is required. The script will prompt you for your project ID, region, and other settings.

If you prefer to run the steps manually, follow the sections below.

## Prerequisites

- A Google Cloud project with billing enabled
- [Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed and authenticated
- [Docker](https://docs.docker.com/get-docker/) installed locally
- The Vertex AI API enabled on your project:
  ```bash
  gcloud services enable aiplatform.googleapis.com
  ```
- An Artifact Registry repository for Docker images:
  ```bash
  gcloud artifacts repositories create mesozoic-labs \
    --repository-format=docker \
    --location=us-central1 \
    --description="Mesozoic Labs training containers"
  ```
- A GCS bucket for training artifacts:
  ```bash
  gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=us-central1
  ```

## 1. Build and Push the Docker Image

The repository includes a `Dockerfile` that packages the training code, MuJoCo, and Stable-Baselines3 into a container suitable for headless training.

```bash
# Set variables
export PROJECT_ID=$(gcloud config get project)
export REGION=us-central1
export IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/mesozoic-labs/trainer:latest

# Authenticate Docker with Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build the image
docker build -t ${IMAGE_URI} .

# Push to Artifact Registry
docker push ${IMAGE_URI}
```

### Test locally first

Before pushing to the cloud, verify the container works (use the `IMAGE_URI` variable from the build step above):

```bash
docker run --rm ${IMAGE_URI} \
  environments/velociraptor/scripts/train_sb3.py \
  train --stage 1 --timesteps 1000
```

> **Note:** If `IMAGE_URI` is not set, Docker will interpret the script path as the image name and fail with a "not found" error. Make sure you've exported `IMAGE_URI` as shown in the build step, or substitute it with the full image URI directly.

## 2. Submit a Training Job

### Option A: Full Curriculum — One Job (Recommended)

Run all three stages end-to-end in a single job. Use `--output-dir` to write checkpoints directly to a mounted GCS path:

```python
from google.cloud import aiplatform

aiplatform.init(
    project="YOUR_PROJECT_ID",
    location="us-central1",
    staging_bucket="gs://YOUR_BUCKET_NAME",
)

SPECIES = "velociraptor"
IMAGE_URI = "us-central1-docker.pkg.dev/YOUR_PROJECT/mesozoic-labs/trainer:latest"

job = aiplatform.CustomJob(
    display_name=f"{SPECIES}-curriculum-full",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE_URI,
                "command": ["python"],
                "args": [
                    f"environments/{SPECIES}/scripts/train_sb3.py",
                    "curriculum",
                    "--n-envs", "4",
                    "--output-dir", f"/gcs/YOUR_BUCKET/training/{SPECIES}",
                ],
                "env": [
                    {"name": "WANDB_API_KEY", "value": "YOUR_WANDB_KEY"},
                ],
            },
        }
    ],
    base_output_dir=f"gs://YOUR_BUCKET/training/{SPECIES}",
)

job.run(sync=False)
print(f"Job submitted: {job.resource_name}")
```

Vertex AI automatically mounts the `base_output_dir` bucket at `/gcs/YOUR_BUCKET/` inside the container.

### Option B: Single-Stage Training

Use this when you want to re-run one specific stage, pick up from a checkpoint, or manually control the timestep budget per stage:

```python
job = aiplatform.CustomJob(
    display_name="raptor-stage1-balance",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE_URI,
                "command": ["python"],
                "args": [
                    "environments/velociraptor/scripts/train_sb3.py",
                    "train",
                    "--stage", "1",
                    "--timesteps", "1000000",
                    "--n-envs", "4",
                    "--output-dir", "/gcs/YOUR_BUCKET/training/velociraptor/stage1",
                ],
            },
        }
    ],
)
job.run(sync=False)
```

To chain stages manually, pass the previous stage's final model to `--load` in the next job:

```python
# Stage 2 picks up the Stage 1 final model
"--load", "/gcs/YOUR_BUCKET/training/velociraptor/stage1/models/stage1_final.zip",
"--stage", "2",
```

### Option C: Using `gcloud` CLI

If you prefer the command line over the Python SDK:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="raptor-curriculum" \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=${IMAGE_URI} \
  --args="environments/velociraptor/scripts/train_sb3.py,curriculum,--n-envs,4,--output-dir,/gcs/YOUR_BUCKET/training/velociraptor"
```

## 3. Machine Type Selection

Choose your machine type based on budget and training needs:

| Machine Type | vCPUs | RAM | GPU | Use Case |
|---|---|---|---|---|
| `n1-standard-4` | 4 | 15 GB | None | Quick tests, debugging |
| `n1-standard-8` + T4 | 8 | 30 GB | 1x NVIDIA T4 | Standard training |
| `n1-standard-16` + T4 | 16 | 60 GB | 1x NVIDIA T4 | Multi-env SubprocVecEnv |
| `n1-standard-8` + V100 | 8 | 30 GB | 1x NVIDIA V100 | Faster training |
| `a2-highgpu-1g` | 12 | 85 GB | 1x NVIDIA A100 | Large-scale training, future MJX/JAX |

**Recommendations:**
- **Stage 1 (balance):** `n1-standard-8` without GPU is sufficient. MuJoCo CPU simulation with SB3 PPO doesn't benefit much from GPU at small batch sizes.
- **Stages 2-3 (locomotion, behavior):** `n1-standard-8` + T4 gives a good cost/performance balance for longer training runs.
- **Full curriculum runs:** `n1-standard-16` + T4 to support `--subproc` with more parallel environments.
- **Future JAX/MJX training:** A100 GPUs become essential for batch simulation.

> **Tip:** For CPU-bound training (no GPU), consider using [`c2-standard-*` machine types](https://cloud.google.com/compute/docs/compute-optimized-machines#c2_machine_types) instead of `n1-standard-*`. C2 machines offer higher per-core performance (3.1 GHz sustained all-core turbo), which benefits MuJoCo's single-threaded simulation stepping and SB3's CPU-side rollout collection. For example, `c2-standard-8` typically completes Stage 1 faster than `n1-standard-8` at a comparable price point.

## 4. Saving Checkpoints to GCS

Use `--output-dir` (the preferred flag for cloud training) to point the script at a GCS-mounted path. Vertex AI mounts the job's `base_output_dir` bucket at `/gcs/<bucket>/` inside the container, so all outputs — models, VecNormalize stats, TensorBoard logs — land in cloud storage automatically.

```python
job = aiplatform.CustomJob(
    display_name="raptor-curriculum-gcs",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE_URI,
                "command": ["python"],
                "args": [
                    "environments/velociraptor/scripts/train_sb3.py",
                    "curriculum",
                    "--n-envs", "4",
                    "--output-dir", "/gcs/YOUR_BUCKET_NAME/training/velociraptor",
                ],
                "env": [
                    {"name": "WANDB_API_KEY", "value": "YOUR_WANDB_KEY"},
                ],
            },
        }
    ],
    base_output_dir="gs://YOUR_BUCKET_NAME/training/velociraptor",
)
```

> **`--output-dir` vs `--log-dir`:** `--output-dir` takes precedence when both are specified. Use `--output-dir` for cloud training (GCS mounts, Vertex AI); use `--log-dir` for local runs where you want to pin the output path. If neither is provided, a timestamped subdirectory is created automatically.

## 5. Algorithm Selection and Hyperparameter Overrides

### Choosing an algorithm

Pass `--algorithm sac` or `--algorithm ppo` to either subcommand. Each stage's TOML config has a `[ppo]` and a `[sac]` section, so per-stage hyperparameters are respected regardless of which algorithm you pick:

```python
"args": [
    "environments/velociraptor/scripts/train_sb3.py",
    "curriculum",
    "--algorithm", "sac",
    "--n-envs", "4",
    "--output-dir", "/gcs/YOUR_BUCKET/training/velociraptor",
],
```

### Overriding hyperparameters from the command line

Use `--override` to change TOML config values without editing files. This is designed for Vertex AI hyperparameter sweep jobs. Keys use dot notation: `ppo.X`, `sac.X`, or `env.X`. Values are auto-cast to `int`, `float`, or `str`.

```python
"args": [
    "environments/velociraptor/scripts/train_sb3.py",
    "curriculum",
    "--algorithm", "ppo",
    "--output-dir", "/gcs/YOUR_BUCKET/training/velociraptor",
    "--override", "ppo.learning_rate=1e-4", "ppo.ent_coef=0.02", "env.alive_bonus=3.0",
],
```

> **Important:** `--override` applies the same value to **all three stages**. This is intentional for sweep jobs where you want a consistent adjustment. If you need different values per stage, use separate `train --stage N` jobs instead.

## 6. W&B Integration on Vertex AI

To enable Weights & Biases logging from cloud training jobs, add `--wandb` and pass your API key as an environment variable:

```python
"container_spec": {
    "image_uri": IMAGE_URI,
    "command": ["python"],
    "args": [
        "environments/velociraptor/scripts/train_sb3.py",
        "curriculum",
        "--n-envs", "4",
        "--output-dir", "/gcs/YOUR_BUCKET/training/velociraptor",
        "--wandb",
    ],
    "env": [
        {"name": "WANDB_API_KEY", "value": "YOUR_WANDB_KEY"},
        {"name": "WANDB_PROJECT", "value": "mesozoic-labs"},
    ],
},
```

If `wandb` is not installed or `WANDB_API_KEY` is not set, the flag is silently ignored — training continues without W&B logging.

**Security tip:** Use [Google Cloud Secret Manager](https://cloud.google.com/secret-manager) for production deployments instead of passing API keys directly:

```bash
echo -n "YOUR_WANDB_KEY" | gcloud secrets create wandb-api-key --data-file=-
```

Then reference the secret in your job configuration.

## 7. Training All Species in Parallel

Submit training jobs for all three species simultaneously:

```python
from google.cloud import aiplatform

aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

SPECIES_LIST = ["velociraptor", "brachiosaurus", "trex"]
IMAGE_URI = "us-central1-docker.pkg.dev/YOUR_PROJECT/mesozoic-labs/trainer:latest"

jobs = []
for species in SPECIES_LIST:
    job = aiplatform.CustomJob(
        display_name=f"{species}-curriculum",
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "n1-standard-8",
                    "accelerator_type": "NVIDIA_TESLA_T4",
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": IMAGE_URI,
                    "command": ["python"],
                    "args": [
                        f"environments/{species}/scripts/train_sb3.py",
                        "curriculum",
                        "--n-envs", "4",
                        "--output-dir", f"/gcs/YOUR_BUCKET/training/{species}",
                        "--wandb",
                    ],
                    "env": [
                        {"name": "WANDB_API_KEY", "value": "YOUR_WANDB_KEY"},
                    ],
                },
            }
        ],
        base_output_dir=f"gs://YOUR_BUCKET/training/{species}",
    )
    job.run(sync=False)
    jobs.append(job)
    print(f"Submitted {species}: {job.resource_name}")
```

## 8. Monitoring Jobs

### From the Console

Visit the [Vertex AI Training page](https://console.cloud.google.com/vertex-ai/training/custom-jobs) to see job status, logs, and resource usage.

### From the CLI

```bash
# List running jobs
gcloud ai custom-jobs list --region=us-central1 --filter="state=JOB_STATE_RUNNING"

# Stream logs from a specific job
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

### From Python

```python
# Check job status
print(job.state)

# Wait for completion
job.wait()
```

## 9. Downloading Results

After training completes, download checkpoints from GCS:

```bash
# Download all artifacts for a species
gcloud storage cp -r gs://YOUR_BUCKET/training/velociraptor/ ./results/

# Download just the final stage 3 model
gcloud storage cp \
  gs://YOUR_BUCKET/training/velociraptor/stage3/models/stage3_final.zip \
  ./models/
```

## 9. Cost Estimation

Approximate costs per training run (as of early 2026, `us-central1`):

| Configuration | Per-Hour Cost | Stage 1 (500K) | Full Curriculum (3.5M) |
|---|---|---|---|
| `n1-standard-8` (CPU only) | ~$0.38 | ~$0.50 | ~$4.00 |
| `n1-standard-8` + T4 | ~$0.73 | ~$0.40 | ~$3.00 |
| `n1-standard-8` + V100 | ~$2.86 | ~$0.80 | ~$6.00 |
| `a2-highgpu-1g` (A100) | ~$4.00 | ~$1.00 | ~$8.00 |

*Actual costs depend on training speed, which varies by species complexity and stage. GPU runs are often cheaper overall because they finish faster.*

**Cost-saving tips:**
- Start with CPU-only for Stage 1 (balance). It's a simple task and usually converges quickly.
- Use [preemptible/spot VMs](https://cloud.google.com/vertex-ai/docs/training/create-custom-job#spot-vms) for up to 60-91% savings on compute.
- Set the `--timesteps` flag conservatively and check results before running longer.

## 10. Using Spot (Preemptible) VMs

For significant cost savings on non-urgent training:

```python
job = aiplatform.CustomJob(
    display_name="raptor-curriculum-spot",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE_URI,
                "command": ["python"],
                "args": [
                    "environments/velociraptor/scripts/train_sb3.py",
                    "curriculum",
                    "--n-envs", "4",
                    "--save-freq", "25000",  # Save more frequently for preemption
                    "--output-dir", "/gcs/YOUR_BUCKET/training/velociraptor",
                ],
            },
        }
    ],
)

# Enable spot VMs via the scheduling config
job.run(
    sync=False,
    restart_job_on_worker_restart=True,  # Auto-restart on preemption
)
```

## Running Long Sweeps from a GCE VM

The `launch-all` command in `sweep.py` blocks while it orchestrates three sequential HPT jobs. For large sweeps (20+ trials across 3 stages), the total wall-clock time can exceed 24 hours. Notebook environments like Colab may disconnect before all stages complete.

The recommended approach is to run the orchestrator from a small **GCE VM** with `tmux` so the process persists indefinitely. The VM only orchestrates — all GPU training happens on Vertex AI worker nodes.

### 1. Create a small orchestrator VM

An `e2-micro` (2 vCPU, 1 GB) is sufficient since it only runs the Python SDK client:

```bash
export PROJECT_ID=$(gcloud config get project)
export ZONE=us-central1-a

gcloud compute instances create sweep-orchestrator \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --machine-type=e2-micro \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --scopes=cloud-platform \
  --metadata=startup-script='#!/bin/bash
    apt-get update -qq && apt-get install -y -qq python3-pip python3-venv tmux git
  '
```

The `cloud-platform` scope gives the VM access to Vertex AI and GCS APIs using its service account — no manual authentication required.

### 2. SSH in and set up the environment

```bash
gcloud compute ssh sweep-orchestrator --zone=${ZONE}
```

On the VM:

```bash
# Clone the repo and install the sweep orchestrator dependencies
git clone https://github.com/kuds/mesozoic-labs.git
cd mesozoic-labs

python3 -m venv .venv
source .venv/bin/activate
pip install google-cloud-aiplatform
```

### 3. Start the sweep in tmux

```bash
tmux new -s sweep
source .venv/bin/activate

# Example: T-Rex sweep — all settings (trials, timesteps, parallel,
# n_envs, search space) are defined per stage in the JSON file
python environments/shared/scripts/sweep.py launch-all \
  --species trex --algorithm ppo \
  --project ${PROJECT_ID} \
  --bucket YOUR_GCS_BUCKET \
  --image us-central1-docker.pkg.dev/${PROJECT_ID}/mesozoic-labs/trainer:latest \
  --search-space-file configs/sweep_ppo.json
```

The `--search-space-file` flag loads per-stage search spaces from a JSON file (see [Customising the Search Space](sweeps.md#customising-the-search-space) for the file format). Pre-built files are included at `configs/sweep_ppo.json` and `configs/sweep_sac.json`.

Detach from tmux with `Ctrl+B` then `D`. The sweep continues running.

To run multiple species in parallel, open additional tmux windows:

```bash
# In the same tmux session, create a new window for velociraptor
tmux new-window -t sweep

python environments/shared/scripts/sweep.py launch-all \
  --species velociraptor --algorithm ppo \
  --project ${PROJECT_ID} \
  --bucket YOUR_GCS_BUCKET \
  --image us-central1-docker.pkg.dev/${PROJECT_ID}/mesozoic-labs/trainer:latest \
  --search-space-file configs/sweep_ppo.json
```

### 4. Reconnect and monitor

```bash
# SSH back in at any time
gcloud compute ssh sweep-orchestrator --zone=${ZONE}
tmux attach -t sweep
```

Monitor individual HPT jobs from any machine:

```bash
# List running HPT jobs
gcloud ai hp-tuning-jobs list --region=us-central1 --project=${PROJECT_ID} \
  --filter="state=JOB_STATE_RUNNING"

# Or check the console
# https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs
```

### 5. Clean up the orchestrator VM

After all sweeps finish, delete the VM to stop incurring costs (~$7/month for e2-micro):

```bash
gcloud compute instances delete sweep-orchestrator \
  --zone=${ZONE} --project=${PROJECT_ID} --quiet
```

Your training artifacts remain safely in GCS at `gs://YOUR_BUCKET/sweeps/`.

## Troubleshooting

### MuJoCo rendering errors

The Dockerfile sets `MUJOCO_GL=osmesa` for headless rendering. If you see OpenGL errors, ensure the base image includes `libosmesa6`. The provided Dockerfile handles this.

### Out of memory

If training crashes with OOM, reduce `--n-envs` or switch to a machine type with more RAM. The quadrupedal Brachiosaurus environment uses more memory than the bipedal species due to its larger observation space (75D vs 69D/77D).

### Job gets preempted frequently

Increase `--save-freq` to save checkpoints more often. Consider switching to on-demand VMs for the final stage (Stage 3) where you don't want to risk losing a long training run.

## Next Steps

Once you have a working training run, use Vertex AI's built-in Hyperparameter Tuning to automatically find the best learning rate, entropy coefficient, batch size, and more — without manually submitting one job per combination. See [Hyperparameter Sweeps](sweeps.md) for the full guide, or jump to [Running a Stage 1 Trial](sweeps.md#running-a-stage-1-trial) for a focused walkthrough of running a single-stage sweep.
