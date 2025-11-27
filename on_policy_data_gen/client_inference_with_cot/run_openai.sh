#!/bin/bash

# Run evaluation with OpenAI API
# Usage: ./run_openai.sh --npz-file path/to/demos.npz [--options]
PERSONA_ID=cdf7cefb-7341-41d7-a193-ff0f2f962cf9

python main.py \
    --npz_file ../icl_snr_results/icl_snr_k4_results_${PERSONA_ID}.npz \
    --inference_backend openai \
    --model_name gpt-4o-mini \
    --dataset_name sher222/persona-iterative-responses \
    --persona_id ${PERSONA_ID} \
    --k 2 \
    --reasoning_output_file ./reasoning_results_${PERSONA_ID}_seed_42_k_2.json \
    --output_dir ./kshot_inference_results \
    --seed 42 \
    --num_samples 5 \
    --num_workers 1 \
    "${@:2}"
