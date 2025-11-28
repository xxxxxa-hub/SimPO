#!/bin/bash

# K-shot inference evaluation script
# Edit the parameters below to match your setup

#######################################
# Configuration
#######################################

# Inference backend: "vllm", "openai", or "huggingface"
INFERENCE_BACKEND="openai"

# Model configuration
MODEL_NAME="gpt-4o-mini"
VLLM_URL="http://localhost:8000"  # Only needed for vllm backend

# API key (set if using OpenAI or HuggingFace)
# export OPENAI_API_KEY="your_key_here"
# export HF_TOKEN="your_token_here"
API_KEY="token"  # Or set via environment variable

# Dataset configuration
DATASET_NAME="sher222/persona-iterative-responses"
PERSONA_ID="cdf7cefb-7341-41d7-a193-ff0f2f962cf9"

# Input npz file with candidate_demo_indices
NPZ_FILE="./icl_snr_results/icl_snr_k4_results_${PERSONA_ID}.npz"  # CHANGE THIS

# K-shot parameters
K=2  # Number of demonstrations
NUM_SAMPLES=1  # Number of random k-shot samples per test example

# Multiprocessing configuration
NUM_WORKERS=16  # Number of parallel workers (1 = sequential, >1 = parallel)

# Output directory
OUTPUT_DIR="./kshot_inference_results"

# Random seed
SEED=42

#######################################
# Run the evaluation
#######################################

echo "================================================"
echo "K-shot Inference Evaluation"
echo "================================================"
echo "Backend: ${INFERENCE_BACKEND}"
echo "Model: ${MODEL_NAME}"
echo "K: ${K}"
echo "Samples per test example: ${NUM_SAMPLES}"
echo "Number of workers: ${NUM_WORKERS}"
echo "NPZ file: ${NPZ_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "================================================"
echo ""

# Build command based on backend
CMD="python test_kshot_inference_client.py \
    --inference_backend ${INFERENCE_BACKEND} \
    --model_name ${MODEL_NAME} \
    --npz_file ${NPZ_FILE} \
    --k ${K} \
    --num_samples ${NUM_SAMPLES} \
    --num_workers ${NUM_WORKERS} \
    --dataset_name ${DATASET_NAME} \
    --persona_id ${PERSONA_ID} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED}"

# Add backend-specific parameters
if [ "${INFERENCE_BACKEND}" = "vllm" ]; then
    CMD="${CMD} --vllm_url ${VLLM_URL}"
fi

if [ -n "${API_KEY}" ]; then
    CMD="${CMD} --api_key ${API_KEY}"
fi

# Print command
echo "Running command:"
echo "${CMD}"
echo ""

# Execute
eval ${CMD}

echo ""
echo "================================================"
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "================================================"
