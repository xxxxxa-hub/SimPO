#!/bin/bash
#SBATCH --job-name=test_best_worst_demo_multigpu
#SBATCH --output=logs/test_best_worst_demo_multigpu_%j.out
#SBATCH --error=logs/test_best_worst_demo_multigpu_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Test accuracy on 512 test examples using top-k best and worst demonstrations (multi-GPU)
# Now uses SNR (signal-to-noise ratio) to rank demonstrations

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
PERSONA_ID="cdf7cefb-7341-41d7-a193-ff0f2f962cf9"
ICL_GAIN_RESULTS_FILE="./icl_gain_results/icl_gain_results_persona_cdf7cefb-7341-41d7-a193-ff0f2f962cf9.npy"
OUTPUT_DIR="./icl_gain_results"
BATCH_SIZE=1
TOP_K=5
DEMO_TYPE="all"  # Options: "best", "worst", "no_context", "all"
RANK_BY="snr_prob_gain"  # Options: "snr_prob_gain", "mean_prob_gain", "snr_acc_gain", "mean_acc_gain", "snr_prob_gain_filtered"

# Create logs directory if it doesn't exist
mkdir -p logs

accelerate launch --num_processes 3 test_one_demo_accelerate.py \
    --model_name ${MODEL_NAME} \
    --persona_id ${PERSONA_ID} \
    --icl_gain_results_file ${ICL_GAIN_RESULTS_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --top_k ${TOP_K} \
    --demo_type ${DEMO_TYPE} \
    --rank_by ${RANK_BY}

echo "Testing complete!"
