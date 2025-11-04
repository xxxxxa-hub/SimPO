#!/bin/bash
# Script to run sentence embedding analysis on glue-sst2 dataset

# Dataset type: classification (for single text + label format)
DATASET_TYPE="classification"

# HuggingFace dataset configuration
DATASET_NAME="stanfordnlp/sst2"
TEXT_FIELD="sentence"
LABEL_FIELD="label"

# Output directory for results
OUTPUT_DIR="./embedding_analysis_sst2"

# Sentence embedding model options:
# - "all-MiniLM-L6-v2" (fast, 384 dims)
# - "all-mpnet-base-v2" (better quality, 768 dims)
# - "all-MiniLM-L12-v2" (medium, 384 dims)
MODEL_NAME="all-mpnet-base-v2"

# Dimensionality reduction method: tsne, umap, or pca
REDUCTION_METHOD="tsne"

# Number of samples to analyze (leave empty for all samples)
MAX_SAMPLES=""

echo "Running sentence embedding analysis for classification data..."
echo "Dataset: $DATASET_NAME"
echo "Dataset type: $DATASET_TYPE"
echo "Text field: $TEXT_FIELD"
echo "Label field: $LABEL_FIELD"
echo "Model: $MODEL_NAME"
echo "Method: $REDUCTION_METHOD"
echo ""

# Build command with optional max_samples
CMD="python analyze_sentence_embeddings.py \
    --dataset_type \"$DATASET_TYPE\" \
    --dataset_name \"$DATASET_NAME\" \
    --text_field \"$TEXT_FIELD\" \
    --label_field \"$LABEL_FIELD\" \
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
echo "For the glue-sst2 dataset:"
echo "  - Label 0: Negative sentiment"
echo "  - Label 1: Positive sentiment"
