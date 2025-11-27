#!/bin/bash

# Run evaluation with OpenAI API
# Usage: ./run_openai.sh --npz-file path/to/demos.npz [--options]
PERSONA_ID=cdf7cefb-7341-41d7-a193-ff0f2f962cf9
SEED=42
k=2

python main.py \
    --npz_file ../icl_snr_results/icl_snr_k4_results_${PERSONA_ID}.npz \
    --inference_backend openai \
    --model_name gpt-4o-mini \
    --dataset_name sher222/persona-iterative-responses \
    --persona_id ${PERSONA_ID} \
    --k ${k} \
    --reasoning_output_file ./reasoning_results_${PERSONA_ID}_seed_${SEED}_k_${k}.json \
    --persona_output_file ./persona_description_${PERSONA_ID}_seed_${SEED}_k_${k}.json \
    --output_dir ./kshot_inference_results \
    --seed ${SEED} \
    --num_workers 1 \
    "${@:2}"
