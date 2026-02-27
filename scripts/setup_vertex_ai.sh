#!/usr/bin/env bash
# setup_vertex_ai.sh — Run this in Google Cloud Shell to build and push the
# Mesozoic Labs training container to Artifact Registry, then optionally
# submit a Vertex AI curriculum training job.
#
# Usage:
#   # Clone the repo and run the script
#   git clone https://github.com/kuds/mesozoic-labs.git
#   cd mesozoic-labs
#   bash scripts/setup_vertex_ai.sh
#
# The script is interactive and will prompt you for any values it cannot
# auto-detect.  Google Cloud Shell already has gcloud, docker, and git
# pre-installed, so no extra tooling is required.

set -euo pipefail

# ---------- colours (no-op when not in a terminal) ----------
if [ -t 1 ]; then
  BOLD="\033[1m"
  GREEN="\033[0;32m"
  YELLOW="\033[0;33m"
  RED="\033[0;31m"
  RESET="\033[0m"
else
  BOLD="" GREEN="" YELLOW="" RED="" RESET=""
fi

info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*"; }

# ---------- 1. Detect / prompt for project settings ----------
echo ""
echo -e "${BOLD}===== Mesozoic Labs — Vertex AI Setup =====${RESET}"
echo ""

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
if [ -z "${PROJECT_ID}" ]; then
  read -rp "Enter your Google Cloud project ID: " PROJECT_ID
fi
info "Project ID: ${PROJECT_ID}"

REGION="${REGION:-us-central1}"
read -rp "Region [${REGION}]: " INPUT_REGION
REGION="${INPUT_REGION:-${REGION}}"
info "Region: ${REGION}"

REPO_NAME="${REPO_NAME:-mesozoic-labs}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/trainer:${IMAGE_TAG}"
info "Image URI: ${IMAGE_URI}"

# ---------- 2. Enable required APIs ----------
echo ""
info "Enabling Vertex AI and Artifact Registry APIs (may already be enabled)..."
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  --project="${PROJECT_ID}" --quiet

# ---------- 3. Create Artifact Registry repo (idempotent) ----------
echo ""
info "Creating Artifact Registry repository '${REPO_NAME}' (if it doesn't exist)..."
if gcloud artifacts repositories describe "${REPO_NAME}" \
     --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
  info "Repository '${REPO_NAME}' already exists — skipping."
else
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Mesozoic Labs training containers" \
    --project="${PROJECT_ID}"
  info "Repository created."
fi

# ---------- 4. Authenticate Docker ----------
echo ""
info "Configuring Docker authentication for ${REGION}-docker.pkg.dev..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ---------- 5. Build the Docker image ----------
echo ""
info "Building Docker image..."
docker build -t "${IMAGE_URI}" .

# ---------- 6. Push to Artifact Registry ----------
echo ""
info "Pushing image to Artifact Registry..."
docker push "${IMAGE_URI}"

echo ""
info "Image pushed successfully: ${IMAGE_URI}"

# ---------- 7. Create GCS bucket (optional) ----------
echo ""
read -rp "Enter a GCS bucket name for training artifacts (or press Enter to skip): " BUCKET_NAME
if [ -n "${BUCKET_NAME}" ]; then
  BUCKET_NAME="${BUCKET_NAME#gs://}"  # strip gs:// prefix if provided
  if gcloud storage buckets describe "gs://${BUCKET_NAME}" &>/dev/null; then
    info "Bucket gs://${BUCKET_NAME} already exists."
  else
    gcloud storage buckets create "gs://${BUCKET_NAME}" \
      --location="${REGION}" --project="${PROJECT_ID}"
    info "Bucket gs://${BUCKET_NAME} created."
  fi
fi

# ---------- 8. Submit a training job (optional) ----------
echo ""
read -rp "Submit a training job now? [y/N]: " SUBMIT_JOB
if [[ "${SUBMIT_JOB}" =~ ^[Yy]$ ]]; then
  echo ""
  echo "Available species: velociraptor, brachiosaurus, trex"
  read -rp "Species [velociraptor]: " SPECIES
  SPECIES="${SPECIES:-velociraptor}"

  if [ -z "${BUCKET_NAME:-}" ]; then
    read -rp "GCS bucket name for output (without gs://): " BUCKET_NAME
  fi

  OUTPUT_DIR="/gcs/${BUCKET_NAME}/training/${SPECIES}"

  echo ""
  echo "Select training mode:"
  echo "  1) Full curriculum (stages 1-3 in one job)  [default]"
  echo "  2) Single stage"
  read -rp "Choice [1]: " MODE_CHOICE
  MODE_CHOICE="${MODE_CHOICE:-1}"

  if [ "${MODE_CHOICE}" = "2" ]; then
    read -rp "Stage number (1, 2, or 3): " STAGE_NUM
    DISPLAY_NAME="${SPECIES}-stage${STAGE_NUM}"
    ARGS="environments/${SPECIES}/scripts/train_sb3.py,train,--stage,${STAGE_NUM},--n-envs,4,--output-dir,${OUTPUT_DIR}/stage${STAGE_NUM}"
  else
    DISPLAY_NAME="${SPECIES}-curriculum"
    ARGS="environments/${SPECIES}/scripts/train_sb3.py,curriculum,--n-envs,4,--output-dir,${OUTPUT_DIR}"
  fi

  echo ""
  read -rp "Machine type [n1-standard-8]: " MACHINE_TYPE
  MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"

  read -rp "Add a GPU? (none / NVIDIA_TESLA_T4 / NVIDIA_TESLA_V100) [NVIDIA_TESLA_T4]: " GPU_TYPE
  GPU_TYPE="${GPU_TYPE:-NVIDIA_TESLA_T4}"

  if [ "${GPU_TYPE}" = "none" ]; then
    WORKER_POOL="machine-type=${MACHINE_TYPE},replica-count=1,container-image-uri=${IMAGE_URI}"
  else
    WORKER_POOL="machine-type=${MACHINE_TYPE},accelerator-type=${GPU_TYPE},accelerator-count=1,replica-count=1,container-image-uri=${IMAGE_URI}"
  fi

  info "Submitting job '${DISPLAY_NAME}'..."
  gcloud ai custom-jobs create \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --display-name="${DISPLAY_NAME}" \
    --worker-pool-spec="${WORKER_POOL}" \
    --args="${ARGS}"

  echo ""
  info "Job submitted! Monitor it with:"
  echo "  gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT_ID} --filter='state=JOB_STATE_RUNNING'"
else
  echo ""
  info "Skipping job submission."
fi

# ---------- Done ----------
echo ""
echo -e "${BOLD}===== Setup Complete =====${RESET}"
echo ""
echo "Your image is available at:"
echo "  ${IMAGE_URI}"
echo ""
echo "Next steps:"
echo "  - Submit a training job (see docs/training/vertex-ai.md)"
echo "  - Run a hyperparameter sweep (see docs/training/sweeps.md)"
echo ""
