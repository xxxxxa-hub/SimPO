#!/bin/bash
# Example script to run sentence embedding analysis on HuggingFace dataset

# HuggingFace dataset configuration
DATASET_NAME="sher222/persona-iterative-responses"
LEVEL_FILTER="309fa18d-e081-481f-9957-80af59494c12"

# Output directory for results
OUTPUT_DIR="./embedding_analysis_results"

# Sentence embedding model options:
# - "all-MiniLM-L6-v2" (fast, 384 dims)
# - "all-mpnet-base-v2" (better quality, 768 dims)
# - "all-MiniLM-L12-v2" (medium, 384 dims)
MODEL_NAME="all-mpnet-base-v2"

# Dimensionality reduction method: tsne, umap, or pca
REDUCTION_METHOD="tsne"

# Number of samples to analyze (leave empty for all samples)
MAX_SAMPLES=""

echo "Running sentence embedding analysis..."
echo "Dataset: $DATASET_NAME"
echo "Level: $LEVEL_FILTER"
echo "Model: $MODEL_NAME"
echo "Method: $REDUCTION_METHOD"
echo ""

# Build command with optional max_samples
CMD="python analyze_sentence_embeddings.py \
    --dataset_name \"$DATASET_NAME\" \
    --level_filter \"$LEVEL_FILTER\" \
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
