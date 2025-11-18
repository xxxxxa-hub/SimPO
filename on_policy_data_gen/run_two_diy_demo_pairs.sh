#!/bin/bash

# Test selected demonstration pairs (5 best, 5 worst, 5 random) on test set
# This script loads existing pair validation results and tests them on the test set

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME="sher222/persona-iterative-responses"
PERSONA_ID="cdf7cefb-7341-41d7-a193-ff0f2f962cf9"
PAIR_RESULTS_FILE="./icl_snr_results/icl_snr_pairs_results_${PERSONA_ID}.npz"  # Update this path
OUTPUT_DIR="./icl_diy_pair_test_results"
BATCH_SIZE=1
SEED=42

accelerate launch test_two_diy_demo_pairs_accelerate.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --persona_id ${PERSONA_ID} \
    --pair_results_file ${PAIR_RESULTS_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED}
