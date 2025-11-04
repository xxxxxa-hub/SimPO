#!/bin/bash
# Script to run sentence embedding analysis on elix_generations dataset

# Dataset type: elix (for elix preference format with response_x/response_y)
DATASET_TYPE="elix"

# HuggingFace dataset configuration
DATASET_NAME="Asap7772/elix_generations_gpt4omini_pref"

# Filter parameters for the elix dataset
# Only use samples where:
# - level_id_x == 1
# - level_id_y == 4
# - scorer_level_id == 4
LEVEL_ID_X=1
LEVEL_ID_Y=4
SCORER_LEVEL_ID=4

# Output directory for results
OUTPUT_DIR="./embedding_analysis_elix"

# Sentence embedding model options:
# - "all-MiniLM-L6-v2" (fast, 384 dims)
# - "all-mpnet-base-v2" (better quality, 768 dims)
# - "all-MiniLM-L12-v2" (medium, 384 dims)
MODEL_NAME="all-mpnet-base-v2"

# Dimensionality reduction method: tsne, umap, or pca
REDUCTION_METHOD="tsne"

# Number of samples to analyze (leave empty for all samples)
MAX_SAMPLES=""

echo "Running sentence embedding analysis for elix preference data..."
echo "Dataset: $DATASET_NAME"
echo "Dataset type: $DATASET_TYPE"
echo "Filter: level_id_x=$LEVEL_ID_X, level_id_y=$LEVEL_ID_Y, scorer_level_id=$SCORER_LEVEL_ID"
echo "Model: $MODEL_NAME"
echo "Method: $REDUCTION_METHOD"
echo ""

# Build command with optional max_samples
CMD="python analyze_sentence_embeddings.py \
    --dataset_type \"$DATASET_TYPE\" \
    --dataset_name \"$DATASET_NAME\" \
    --level_id_x $LEVEL_ID_X \
    --level_id_y $LEVEL_ID_Y \
    --scorer_level_id $SCORER_LEVEL_ID \
    --model_name \"$MODEL_NAME\" \
    --reduction_method \"$REDUCTION_METHOD\" \
    --output_dir \"$OUTPUT_DIR\" \
    --batch_size 32 \
    --n_clusters 2 \
    --random_seed 42"

# Add max_samples if specified
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Execute command
eval $CMD

echo ""
echo "Analysis complete! Check $OUTPUT_DIR for results:"
echo "  - embeddings_${REDUCTION_METHOD}.png: Main visualization"
echo "  - embeddings_${REDUCTION_METHOD}_detailed.png: Detailed analysis"
echo "  - embeddings.npz: Saved embeddings"
echo "  - analysis_summary.json: Metrics and results"
echo ""
echo "For the elix dataset:"
echo "  - When det_choice==1: response_y is preferred (blue)"
echo "  - When det_choice==0: response_x is preferred (blue)"
echo "  - Non-preferred responses are shown in red"
