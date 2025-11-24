#!/bin/bash

# Iterative context optimization
# Optimizes contexts by replacing demonstrations one at a time
# to maximize validation log probability

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME="sher222/persona-iterative-responses"
PERSONA_ID="cdf7cefb-7341-41d7-a193-ff0f2f962cf9"
VALIDATION_RESULTS_FILE="./icl_snr_results/icl_snr_k4_results_${PERSONA_ID}.npz"
OUTPUT_DIR="./icl_optimized_results"
BATCH_SIZE=1
TOP_N=1  # Number of best contexts to optimize
N_CANDIDATES=50  # Number of test examples to try as replacements
SEED=42

accelerate launch --num_processes 1 \
    test_iterative_context_optimization_accelerate.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --persona_id ${PERSONA_ID} \
    --validation_results_file ${VALIDATION_RESULTS_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --top_n ${TOP_N} \
    --n_candidates ${N_CANDIDATES} \
    --seed ${SEED}
