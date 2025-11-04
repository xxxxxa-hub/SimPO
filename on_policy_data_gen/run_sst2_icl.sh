#!/bin/bash
# Example script to run SST2 semantic analysis in-context learning experiment

# Basic example: Multiple k-shots and seeds on validation set
accelerate launch --num_processes 3 random_seed_pairwise_preference_comparison_sst2.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name stanfordnlp/sst2 \
  --train_split train \
  --test_split validation \
  --k_shots 128 \
  --seeds 42 123 999 3407 114514 \
  --batch_size 1 \
  --output_dir ./sst2_results/llama-3.1-8b-instruct \

# Example with limited samples for testing
# accelerate launch --num_processes 3 random_seed_pairwise_preference_comparison_sst2.py \
#   --model_name meta-llama/Llama-3.1-8B-Instruct \
#   --dataset_name stanfordnlp/sst2 \
#   --train_split train \
#   --test_split validation \
#   --k_shots 0 4 8 \
#   --seeds 42 123 \
#   --max_samples 100 \
#   --batch_size 8 \
#   --output_dir ./sst2_results/test_run

# Multi-GPU example with flash attention
# accelerate launch --num_processes 4 random_seed_pairwise_preference_comparison_sst2.py \
#   --model_name meta-llama/Llama-3.1-8B-Instruct \
#   --dataset_name stanfordnlp/sst2 \
#   --train_split train \
#   --test_split validation \
#   --k_shots 0 2 4 8 16 \
#   --seeds 42 123 999 \
#   --batch_size 4 \
#   --use_flash_attention \
#   --output_dir ./sst2_results/llama-3.1-8b-instruct-multi-gpu
