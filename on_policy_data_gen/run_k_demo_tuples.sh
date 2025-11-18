#!/bin/bash
#
# Script to run test_k_demo_tuples_accelerate.py with 3 GPUs using accelerate
#
# Usage: bash run_k_demo_tuples.sh

# Set the path to your ICL gain results file
ICL_GAIN_RESULTS="icl_gain_results/icl_gain_results_persona_cdf7cefb-7341-41d7-a193-ff0f2f962cf9.npy"

# Run with accelerate on 3 GPUs
# k=4: 4 demonstrations per context
# n_best=4, n_worst=4: 8 total candidates
# P_4^8 = 8*7*6*5 = 1,680 ordered 4-tuples
# Each tested on 50 validation × 2 orderings = 168,000 examples
accelerate launch \
    --num_processes 3 \
    --multi_gpu \
    test_k_demo_tuples_accelerate.py \
    --icl_gain_results_file "$ICL_GAIN_RESULTS" \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset_name "sher222/persona-iterative-responses" \
    --persona_id "cdf7cefb-7341-41d7-a193-ff0f2f962cf9" \
    --k 4 \
    --n_best 4 \
    --n_worst 4 \
    --batch_size 1 \
    --top_n 3 \
    --rank_by "snr_prob_gain" \
    --output_dir "./icl_snr_results" \
    --seed 42
