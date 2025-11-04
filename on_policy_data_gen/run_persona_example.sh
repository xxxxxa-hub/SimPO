#!/bin/bash
# Example script to run persona-based in-context learning experiment

# Basic example: Use first persona found, 4-shot, test on 100 samples
accelerate launch --num_processes 1 random_seed_pairwise_preference_comparison_persona.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name sher222/persona-iterative-responses \
  --split train \
  --k_shots 2 \
  --max_samples 512 \
  --seeds 42 123 999 3407 114514 \
  --batch_size 1 \
  --output_dir ./persona_results/llama-3.1-8b-instruct-prompt-2 \
  --persona_ids cdf7cefb-7341-41d7-a193-ff0f2f962cf9 309fa18d-e081-481f-9957-80af59494c12
  # --persona_ids 309fa18d-e081-481f-9957-80af59494c12 cdf7cefb-7341-41d7-a193-ff0f2f962cf9 
  # --persona_ids 7340e022-b9ec-4e74-b365-becb1020feee
  # --persona_ids 309fa18d-e081-481f-9957-80af59494c12 cdf7cefb-7341-41d7-a193-ff0f2f962cf9 

# Example with specific persona ID
# accelerate launch --num_processes 1 random_seed_pairwise_preference_comparison_persona.py \
#   --model_name meta-llama/Llama-3.1-8B-Instruct \
#   --dataset_name sher222/persona-iterative-responses \
#   --split train \
#   --persona_id "309fa18d-e081-481f-9957-80af59494c12" \
#   --k_shot 8 \
#   --max_samples 200 \
#   --seed 42 \
#   --batch_size 4 \
#   --output_dir ./persona_results_specific

# Multi-GPU example
# accelerate launch --num_processes 4 random_seed_pairwise_preference_comparison_persona.py \
#   --model_name meta-llama/Llama-3.1-8B-Instruct \
#   --dataset_name sher222/persona-iterative-responses \
#   --split train \
#   --k_shot 10 \
#   --max_samples 500 \
#   --seed 42 \
#   --batch_size 2 \
#   --use_flash_attention \
#   --output_dir ./persona_results_multi_gpu
