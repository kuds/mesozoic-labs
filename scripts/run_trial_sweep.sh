#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Trial Sweep — Lightweight end-to-end validation of the sweep pipeline.
#
# This script runs a minimal sweep (3 trials per stage, 100k-300k timesteps)
# to verify that the full three-stage sweep pipeline works before committing
# to a production-sized run (~20 trials, 500k-1.5M timesteps).
#
# Expected cost:  ~$2-4 total  (3 trials × 3 stages at 100k/200k/300k steps)
# Expected time:  ~1-2 hours wall clock
#
# What it validates:
#   - Docker image builds and runs on Vertex AI
#   - Search space JSON is parsed correctly
#   - Stage 1 → Stage 2 → Stage 3 auto-chaining works
#   - Best trial selection and checkpoint passing work
#   - GCS output paths are created correctly
#   - Resume state is saved after each stage
#
# Prerequisites:
#   1. GCP project with Vertex AI API enabled
#   2. Docker image built and pushed (see Step 1 below)
#   3. GCS bucket created
#   4. gcloud authenticated: gcloud auth application-default login
#
# Usage:
#   # 1. Set your GCP variables
#   export PROJECT_ID=$(gcloud config get project)
#   export BUCKET=my-mesozoic-bucket
#   export REGION=us-central1
#   export IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/mesozoic-labs/trainer:latest
#
#   # 2. Build and push the Docker image (if not already done)
#   docker build -t ${IMAGE_URI} .
#   docker push ${IMAGE_URI}
#
#   # 3. Run the trial sweep
#   bash scripts/run_trial_sweep.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Required environment variables ───────────────────────────────────────────
: "${PROJECT_ID:?Set PROJECT_ID to your GCP project (export PROJECT_ID=\$(gcloud config get project))}"
: "${BUCKET:?Set BUCKET to your GCS bucket name without gs:// prefix (export BUCKET=my-bucket)}"
: "${IMAGE_URI:?Set IMAGE_URI to your Docker image URI (export IMAGE_URI=us-central1-docker.pkg.dev/\$PROJECT_ID/mesozoic-labs/trainer:latest)}"

REGION="${REGION:-us-central1}"
SPECIES="${SPECIES:-velociraptor}"
ALGORITHM="${ALGORITHM:-ppo}"

echo "============================================================"
echo " Trial Sweep — Validation Run"
echo "============================================================"
echo " Project:    ${PROJECT_ID}"
echo " Bucket:     ${BUCKET}"
echo " Region:     ${REGION}"
echo " Image:      ${IMAGE_URI}"
echo " Species:    ${SPECIES}"
echo " Algorithm:  ${ALGORITHM}"
echo ""
echo " Config:     configs/sweep_trial_ppo.json"
echo " Trials:     3 per stage  (production: 20)"
echo " Parallel:   2             (production: 5)"
echo " Timesteps:  100k / 200k / 300k  (production: 500k / 1M / 1.5M)"
echo "============================================================"
echo ""

# ── Step 0: Optional local smoke test ────────────────────────────────────────
# Uncomment the block below to run a single local trial first (no cloud cost).
# This verifies that the training code, environment, and config loading work
# before submitting anything to Vertex AI.
#
# echo ">> Step 0: Local smoke test (10k steps, no cloud)..."
# python "${REPO_ROOT}/environments/shared/scripts/sweep.py" trial \
#     --species "${SPECIES}" --stage 1 --algorithm "${ALGORITHM}" \
#     --timesteps 10000 --n-envs 1 \
#     --ppo_learning_rate 1e-4 --ppo_ent_coef 0.01 --ppo_batch_size 128
# echo ">> Local smoke test passed."
# echo ""

# ── Step 1: Launch the trial sweep (all 3 stages) ───────────────────────────
echo ">> Launching trial sweep (launch-all) with minimal config..."
echo "   This will run 3 stages sequentially, ~3 trials each."
echo ""

python "${REPO_ROOT}/environments/shared/scripts/sweep.py" launch-all \
    --species "${SPECIES}" \
    --algorithm "${ALGORITHM}" \
    --project "${PROJECT_ID}" \
    --bucket "${BUCKET}" \
    --image "${IMAGE_URI}" \
    --location "${REGION}" \
    --search-space-file "${REPO_ROOT}/configs/sweep_trial_ppo.json" \
    --no-resume \
    --force-continue

echo ""
echo "============================================================"
echo " Trial sweep complete!"
echo ""
echo " Check results at:"
echo "   gs://${BUCKET}/sweeps/${SPECIES}/"
echo ""
echo " If everything looks good, run the full sweep with:"
echo "   python environments/shared/scripts/sweep.py launch-all \\"
echo "     --species ${SPECIES} --algorithm ${ALGORITHM} \\"
echo "     --project ${PROJECT_ID} --bucket ${BUCKET} \\"
echo "     --image ${IMAGE_URI} \\"
echo "     --search-space-file configs/sweep_ppo.json \\"
echo "     --trials 20 --parallel 5"
echo "============================================================"
