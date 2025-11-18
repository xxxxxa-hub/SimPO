#!/bin/bash
#
# Script to run test_two_demo_pairs_accelerate.py with 3 GPUs using accelerate
#
# Usage: bash run_two_demo_pairs.sh

# Set the path to your ICL gain results file
ICL_GAIN_RESULTS="icl_gain_results/icl_gain_results_persona_cdf7cefb-7341-41d7-a193-ff0f2f962cf9.npy"

# Run with accelerate on 3 GPUs
accelerate launch \
    --num_processes 3 \
    --multi_gpu \
    test_two_demo_pairs_accelerate.py \
    --icl_gain_results_file "$ICL_GAIN_RESULTS" \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset_name "sher222/persona-iterative-responses" \
    --persona_id "cdf7cefb-7341-41d7-a193-ff0f2f962cf9" \
    --batch_size 1 \
    --top_k 5 \
    --rank_by "snr_prob_gain" \
    --output_dir "./icl_snr_results" \
    --seed 42
