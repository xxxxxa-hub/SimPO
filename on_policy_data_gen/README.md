# On-Policy Preference Data Generation

We provide the code to generate on-policy preference data (e.g., [princeton-nlp/llama3-ultrafeedback-armorm](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm) and [princeton-nlp/gemma2-ultrafeedback-armorm](https://huggingface.co/datasets/princeton-nlp/gemma2-ultrafeedback-armorm)) used in our experiments.

## Requirements

You will need to install the [`vllm`](https://github.com/vllm-project/vllm) package for decoding. Moreover, if you are running decoding with `gemma-2` models, you will need to also install `flashinfer`.

## On-Policy Preference Data Generation Process

1. Generate multiple responses using the language model:

```
python decode.py --data_dir $DATASET_DIR --seed $SEED
```
This will generate one response per prompt under the specified seed. You need to provide a dataset containing prompts (by default, we use `HuggingFaceH4/ultrafeedback_binarized`). You can also set decoding hyperparameters by passing in corresponding arguments (by default, we use a temperature of `0.8` for sampling).

Note that you will need to run the above command under **multiple different** seeds (by default, we use `13, 21, 42, 79, 100`) to obtain different responses for each prompt.

2. Post-process the generations

```
python post_process.py
```

This will combine the generated responses under each seed and filter out samples with identical responses across all seeds.

3. Annotate the preference labels with a reward model

```
python reward_model_annotate.py --reward_model $MODEL
```

This will score the generations using a reward model (by default, we use `RLHFlow/ArmoRM-Llama3-8B-v0.1`) and binarize the dataset by taking the highest-scoring response as the winning response and the lowest-scoring one as the losing.


# Pairwise Preference Comparison and Dataset Conversion Pipeline

This directory contains a complete pipeline for performing pairwise preference comparisons and converting the results into preference datasets. The pipeline uses Hugging Face Accelerate for distributed processing and produces datasets compatible with preference learning frameworks like SimPO.

## Overview

The pipeline consists of two main stages:

1. **Pairwise Preference Comparison**: Compare all response pairs using distributed processing to build preference matrices
2. **Dataset Conversion**: Convert preference matrices to training-ready datasets compatible with frameworks like SimPO

## Quick Start

### Complete Pipeline Example

```bash
# 1. Configure accelerate (first time only)
accelerate config

# 2. Run pairwise preference comparison
accelerate launch pairwise_preference_comparison_accelerate.py \
    --model_name google/gemma-2-9b-it \
    --output_file pairwise_preferences.json

# 3. Convert to preference dataset
python convert_to_preference_dataset.py \
    --train_file pairwise_preferences.json \
    --test_file pairwise_preferences.json \
    --repo_id username/my-preference-dataset
```

## Stage 1: Pairwise Preference Comparison

The pipeline uses Hugging Face Accelerate for distributed data parallel processing across multiple GPUs.

### Files

- `pairwise_preference_comparison_accelerate.py` - Main script with Accelerate integration
- `accelerate_config_example.yaml` - Example Accelerate configuration

### Key Features

#### Key Features
- **Multi-GPU Data Parallel**: Distributes comparisons across multiple GPUs using Accelerate
- **Batched Processing**: Processes multiple comparisons simultaneously per GPU
- **Custom Dataset**: Efficient dataset class that pre-generates all comparison pairs
- **Left Padding**: Proper left padding for causal language models (required for generation)
- **No Truncation**: Supports full context length without truncation (configurable)
- **Flash Attention 2**: Optional support for faster attention computation
- **Torch Compilation**: Optional model compilation for additional speedup
- **Automatic GPU Detection**: Automatically uses all available GPUs
- **Load Balancing**: Evenly distributes work across processes
- **Result Gathering**: Efficiently collects and synchronizes results from all processes
- **Progress Tracking**: Coordinated progress reporting across processes

### Installation

```bash
# Install required packages
pip install accelerate transformers torch datasets numpy tqdm

# For Flash Attention 2 (optional, for better performance)
pip install flash-attn --no-build-isolation
```

### Usage

#### Configuration Setup (First Time Only)

```bash
# Configure accelerate interactively
accelerate config

# Or use the provided example configuration
cp accelerate_config_example.yaml ~/.cache/huggingface/accelerate/default_config.yaml
```

#### Basic Usage

```bash
# Run with default settings (uses all available GPUs)
accelerate launch pairwise_preference_comparison_accelerate.py
```

#### Custom Configuration

```bash
# Run with custom parameters
accelerate launch pairwise_preference_comparison_accelerate.py \
    --model_name google/gemma-2-9b-it \
    --dataset_name Alligator123/gemma2-ultrafeedback-armorm-false_qa \
    --output_file my_results.json \
    --batch_size 16 \
    --max_samples 100
```

#### Using Custom Accelerate Config

```bash
# Use a specific configuration file
accelerate launch --config_file accelerate_config_example.yaml \
    pairwise_preference_comparison_accelerate.py \
    --model_name google/gemma-2-9b-it \
    --batch_size 32 \
    --output_file results.json
```

### Configuration Options

#### Script Arguments

- `--model_name`: Model to use (default: google/gemma-2-9b-it)
- `--dataset_name`: HuggingFace dataset name (default: Alligator123/gemma2-ultrafeedback-armorm-false_qa)
- `--output_file`: Output JSON file (default: pairwise_preferences_accelerate.json)
- `--batch_size`: Batch size per GPU (default: 32)
- `--max_samples`: Limit number of samples for testing (default: all)
- `--use_compilation`: Enable torch compilation (default: True)
- `--use_flash_attention`: Enable Flash Attention 2 (default: True)
- `--max_length`: Maximum sequence length (default: None, no truncation)


### Performance Expectations

- **Single GPU**: ~50-100 samples/hour depending on model size
- **4 GPUs**: ~200-400 samples/hour (near-linear scaling)
- **8 GPUs**: ~400-800 samples/hour (with sufficient data)

The pipeline scales nearly linearly with the number of GPUs, making it efficient for large-scale preference data generation.

### Output Format

The output JSON file contains:

```json
[
  {
    "sample_id": 0,
    "prompt": "Original question/prompt",
    "responses": ["response1", "response2", "response3", "response4", "response5"],
    "preference_matrix": [
      [null, 0.7, 0.8, 0.6, 0.9],
      [0.3, null, 0.5, 0.4, 0.7],
      [0.2, 0.5, null, 0.3, 0.6],
      [0.4, 0.6, 0.7, null, 0.8],
      [0.1, 0.3, 0.4, 0.2, null]
    ],
    "detailed_comparisons": {
      "0_vs_1": {
        "response_a": "response1",
        "response_b": "response2",
        "prob_a_over_b": 0.7,
        "logit_a": 2.1,
        "logit_b": 1.4,
        "prompt": "Full comparison prompt..."
      }
    },
    "original_rm_scores": [0.8, 0.6, 0.4, 0.7, 0.2]
  }
]
```

## Stage 2: Preference Dataset Conversion

Convert pairwise preference comparison results into a preference dataset format compatible with [princeton-nlp/gemma2-ultrafeedback-armorm](https://huggingface.co/datasets/princeton-nlp/gemma2-ultrafeedback-armorm).

### Files

- `convert_to_preference_dataset.py` - Main conversion script

### Conversion Process

1. **Loads** your pairwise preference comparison JSON results
2. **Corrects** raw preference probabilities using the formula:
   ```
   Corrected P(A>B) = (Raw P(A>B) + 1 - Raw P(B>A)) / 2
   ```
3. **Finds** the most extreme preference for each prompt (closest to 0 or 1)
4. **Creates** chosen/rejected pairs from the most confident preferences
5. **Formats** the data like the Princeton dataset
6. **Pushes** to your Hugging Face Hub repository

### Installation

Install additional dependencies:

```bash
pip install datasets huggingface_hub
```

### Usage

```bash
# Basic conversion and upload
python convert_to_preference_dataset.py \
    --train_file train_preferences.json \
    --test_file test_preferences.json \
    --repo_id username/my-preference-dataset

# With custom settings
python convert_to_preference_dataset.py \
    --train_file pairwise_preferences.json \
    --test_file pairwise_preferences.json \
    --repo_id username/my-preference-dataset \
    --min_confidence 0.15 \
    --output_dir my_dataset \
    --private

# Dry run (don't upload)
python convert_to_preference_dataset.py \
    --train_file train_preferences.json \
    --test_file test_preferences.json \
    --repo_id username/my-preference-dataset \
    --dry_run
```

### Configuration Options

#### Script Arguments

- `--train_file`: **Required** - Train split JSON file with preference matrices
- `--test_file`: **Required** - Test split JSON file with preference matrices
- `--repo_id`: **Required** - HuggingFace repository ID (`username/dataset-name`)
- `--output_dir`: Local directory to save dataset (optional)
- `--private`: Make repository private (default: True)
- `--dry_run`: Create dataset locally but don't upload

#### Understanding Confidence

Confidence = |preference_probability - 0.5|

- **High confidence (>0.4)**: Very clear preference (prob >0.9 or <0.1)
- **Medium confidence (0.2-0.4)**: Clear preference (prob 0.7-0.9 or 0.1-0.3)
- **Low confidence (<0.2)**: Weak preference (prob 0.5-0.7 or 0.3-0.5)

Recommended thresholds:
- `0.1`: Include most pairs (more data, some noise)
- `0.2`: Balanced (good data quality)
- `0.3`: High quality only (less data, very clear preferences)

### Output Dataset Format

The created dataset follows the same structure as [princeton-nlp/gemma2-ultrafeedback-armorm](https://huggingface.co/datasets/princeton-nlp/gemma2-ultrafeedback-armorm):

#### Core Fields (Compatible)
```python
{
    "prompt_id": "sha256_hash_of_prompt",
    "prompt": "Original question/prompt",
    "all_generated_responses": ["response1", "response2", ...],
    "all_rm_scores": [0.22, 0.21, 0.23, 0.21, 0.21],
    "chosen": [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "preferred_response"}
    ],
    "rejected": [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "rejected_response"}
    ]
}
```

#### Additional Metadata
```python
{
    "preference_probability": 0.85,           # Corrected preference probability
    "confidence": 0.35,                       # Distance from random (0.5)
    "chosen_response_index": 2,               # Index in all_generated_responses
    "rejected_response_index": 4,             # Index in all_generated_responses
    "corrected_preference_matrix": [[...]]    # Full 5x5 corrected matrix
}
```

## Example Usage Scenarios

### 1. Research Dataset Creation

```bash
# High-quality dataset for training
accelerate launch pairwise_preference_comparison_accelerate.py \
    --model_name google/gemma-2-9b-it \
    --output_file research_preferences.json

python convert_to_preference_dataset.py \
    --train_file research_preferences.json \
    --test_file research_preferences.json \
    --repo_id myorg/high-quality-preferences \
    --min_confidence 0.3 \
    --private
```

### 2. Testing and Development

```bash
# Quick test with small dataset
accelerate launch pairwise_preference_comparison_accelerate.py \
    --max_samples 50 \
    --batch_size 8 \
    --output_file test_preferences.json

python convert_to_preference_dataset.py \
    --train_file test_preferences.json \
    --test_file test_preferences.json \
    --repo_id myusername/test-preferences \
    --min_confidence 0.1 \
    --dry_run
```

### 3. Public Dataset Release

```bash
# Balanced public dataset
accelerate launch pairwise_preference_comparison_accelerate.py \
    --output_file public_preferences.json

python convert_to_preference_dataset.py \
    --train_file public_preferences.json \
    --test_file public_preferences.json \
    --repo_id myusername/llama-preferences \
    --min_confidence 0.2 \
    --output_dir final_dataset
```

## Dataset Statistics

After conversion, you'll see statistics like:

```
=== Dataset Statistics ===
Total samples: 1873
Average confidence: 0.312
Average preference probability: 0.742
```

This helps you understand:
- **Total samples**: How many preference pairs were created
- **Average confidence**: How extreme the preferences are on average
- **Average preference probability**: How often the chosen response was preferred

## Troubleshooting

### Common Issues

#### Pairwise Comparison Issues

1. **Out of Memory**
   - Reduce `--batch_size`
   - Add `--max_length` with reasonable limit (e.g., 4096, 8192)
   - Use fewer GPUs

2. **Slow Performance**
   - Enable Flash Attention: `--use_flash_attention`
   - Enable compilation: `--use_compilation`
   - Increase batch size if memory allows

3. **Accelerate Configuration**
   ```bash
   # Reconfigure accelerate
   accelerate config

   # Check current config
   accelerate env
   ```

4. **CUDA Errors**
   - Ensure CUDA is properly installed
   - Check GPU memory with `nvidia-smi`
   - Restart if GPUs are in a bad state

#### Dataset Conversion Issues

1. **Low sample count**: Try reducing `--min_confidence`
2. **Authentication errors**: Set your HF token with `huggingface-cli login`
3. **Out of memory**: Process smaller chunks of data
4. **Upload failures**: Check internet connection and repository permissions

### Debugging

```bash
# Debug pairwise comparison
ACCELERATE_DEBUG_MODE=1 accelerate launch pairwise_preference_comparison_accelerate.py \
    --max_samples 10 \
    --batch_size 4 \
    --output_file debug_preferences.json

# Debug dataset conversion
python convert_to_preference_dataset.py \
    --train_file debug_preferences.json \
    --test_file debug_preferences.json \
    --repo_id test/debug \
    --dry_run
```

### Quality Control

```python
# Load and inspect the dataset
from datasets import load_dataset
import numpy as np

dataset = load_dataset("username/your-dataset")

print(f"Dataset size: {len(dataset['train'])}")
print(f"Average confidence: {np.mean(dataset['train']['confidence'])}")
print(f"Confidence distribution:")
print(np.histogram(dataset['train']['confidence'], bins=10))
```

## Technical Details

### Architecture

The pipeline uses Hugging Face Accelerate for distributed processing:

1. **Custom Dataset**: `PairwiseComparisonDataset` pre-generates all comparison pairs
2. **Batched Inference**: `get_preference_probabilities_batch_accelerate` processes batches efficiently
3. **Result Gathering**: `gather_object` synchronizes results from all processes
4. **Matrix Reconstruction**: `reconstruct_preference_matrices` rebuilds the final format

### Memory Management

- Uses `torch.bfloat16` for reduced memory usage
- Implements dynamic padding to minimize memory waste
- Periodically clears GPU cache to prevent memory leaks
- Supports gradient checkpointing (though not used in inference)

### Scaling Considerations

- **Linear Scaling**: Performance scales nearly linearly with GPU count
- **Memory Distribution**: Each GPU processes a subset of comparisons
- **Communication Overhead**: Minimal due to inference-only workload
- **Load Balancing**: Automatic through DataLoader distribution

## Integration with Training

The output dataset can be directly used with preference learning frameworks:

```python
# Load your converted dataset
from datasets import load_dataset
dataset = load_dataset("username/your-preference-dataset")

# Use with DPO, SimPO, etc.
# The format matches princeton-nlp/gemma2-ultrafeedback-armorm
```

## Contributing

When modifying the pipeline:

1. Test with single GPU first
2. Verify multi-GPU scaling
3. Check memory usage across all GPUs
4. Ensure result consistency with original version
5. Update documentation for any new features

## Citation

If you use datasets created with this pipeline, please cite the original preference comparison methodology and the underlying data sources.