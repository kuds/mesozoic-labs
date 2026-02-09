# Training on Google Cloud Vertex AI

This guide covers how to run Mesozoic Labs training jobs on [Vertex AI](https://cloud.google.com/vertex-ai), Google Cloud's managed ML platform. Vertex AI lets you run the full 3-stage curriculum on cloud GPUs without managing infrastructure.

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

Before pushing to the cloud, verify the container works:

```bash
docker run --rm ${IMAGE_URI} \
  environments/velociraptor/scripts/train_sb3.py \
  train --stage 1 --timesteps 1000
```

## 2. Submit a Training Job

### Option A: Single-Stage Training

Use the Vertex AI Python SDK to submit a single-stage training job:

```python
from google.cloud import aiplatform

aiplatform.init(
    project="YOUR_PROJECT_ID",
    location="us-central1",
    staging_bucket="gs://YOUR_BUCKET_NAME",
)

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
                "image_uri": "us-central1-docker.pkg.dev/YOUR_PROJECT/mesozoic-labs/trainer:latest",
                "command": ["python"],
                "args": [
                    "environments/velociraptor/scripts/train_sb3.py",
                    "train",
                    "--stage", "1",
                    "--timesteps", "500000",
                    "--n-envs", "4",
                ],
            },
        }
    ],
)

job.run(sync=False)  # Submit and return immediately
print(f"Job submitted: {job.resource_name}")
```

### Option B: Full Curriculum (Recommended)

Run all three stages end-to-end with automatic advancement:

```python
from google.cloud import aiplatform

aiplatform.init(
    project="YOUR_PROJECT_ID",
    location="us-central1",
    staging_bucket="gs://YOUR_BUCKET_NAME",
)

SPECIES = "velociraptor"

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
                "image_uri": "us-central1-docker.pkg.dev/YOUR_PROJECT/mesozoic-labs/trainer:latest",
                "command": ["python"],
                "args": [
                    f"environments/{SPECIES}/scripts/train_sb3.py",
                    "curriculum",
                    "--n-envs", "4",
                ],
            },
        }
    ],
)

job.run(sync=False)
```

### Option C: Using `gcloud` CLI

If you prefer the command line over the Python SDK:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="raptor-curriculum" \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=${IMAGE_URI} \
  --args="environments/velociraptor/scripts/train_sb3.py,curriculum,--n-envs,4"
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

## 4. Saving Checkpoints to GCS

By default, training saves checkpoints to local disk inside the container. To persist them to GCS, mount a GCS bucket as the output directory.

### Modify the training command to write to `/gcs/`:

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
                    "--log-dir", "/gcs/YOUR_BUCKET_NAME/training/velociraptor",
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

Vertex AI automatically mounts the base output directory at `/gcs/YOUR_BUCKET_NAME/` inside the container.

## 5. W&B Integration on Vertex AI

To enable Weights & Biases logging from cloud training jobs, pass your API key as an environment variable:

```python
"container_spec": {
    "image_uri": IMAGE_URI,
    "command": ["python"],
    "args": [...],
    "env": [
        {"name": "WANDB_API_KEY", "value": "YOUR_WANDB_KEY"},
        {"name": "WANDB_PROJECT", "value": "mesozoic-labs"},
    ],
},
```

**Security tip:** Use [Google Cloud Secret Manager](https://cloud.google.com/secret-manager) for production deployments instead of passing API keys directly:

```bash
echo -n "YOUR_WANDB_KEY" | gcloud secrets create wandb-api-key --data-file=-
```

Then reference the secret in your job configuration.

## 6. Training All Species in Parallel

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
                        "--log-dir", f"/gcs/YOUR_BUCKET/training/{species}",
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

## 7. Monitoring Jobs

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

## 8. Downloading Results

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
                    "--log-dir", "/gcs/YOUR_BUCKET/training/velociraptor",
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

## Troubleshooting

### MuJoCo rendering errors

The Dockerfile sets `MUJOCO_GL=osmesa` for headless rendering. If you see OpenGL errors, ensure the base image includes `libosmesa6`. The provided Dockerfile handles this.

### Out of memory

If training crashes with OOM, reduce `--n-envs` or switch to a machine type with more RAM. The quadrupedal Brachiosaurus environment uses more memory than the bipedal species due to its larger observation space (75D vs 69D/77D).

### Job gets preempted frequently

Increase `--save-freq` to save checkpoints more often. Consider switching to on-demand VMs for the final stage (Stage 3) where you don't want to risk losing a long training run.
