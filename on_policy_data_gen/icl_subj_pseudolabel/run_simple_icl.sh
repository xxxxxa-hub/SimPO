#!/bin/bash

# Iterative Context Optimization with Pseudo-Labeling
# Tests in-context learning on SetFit/subj binary classification task
# Algorithm:
#   1. Sample demonstration pool from training
#   2. Generate all k-shot permutations and evaluate on validation
#   3. Select top N contexts by SNR
#   4. Iteratively optimize by replacing with pseudo-labeled test examples

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
K_SHOT=2  # Number of demonstrations to use (k)
POOL_SIZE=10  # Size of demonstration pool to sample from training
TEST_SIZE=500  # Number of test examples to evaluate
VAL_SIZE=50  # Number of validation examples
TOP_N=3  # Number of best contexts to select and optimize
N_CANDIDATES=100  # Number of test examples to try as replacements
OUTPUT_DIR="./icl_optimized_results"
SEED=42

python simple_icl_pseudolabel.py \
    --model_name ${MODEL_NAME} \
    --k_shot ${K_SHOT} \
    --pool_size ${POOL_SIZE} \
    --test_size ${TEST_SIZE} \
    --val_size ${VAL_SIZE} \
    --top_n ${TOP_N} \
    --n_candidates ${N_CANDIDATES} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED}

echo ""
echo "Results saved to ${OUTPUT_DIR}/"
echo "  - optimized_contexts_k${K_SHOT}.npz  (numpy format)"
