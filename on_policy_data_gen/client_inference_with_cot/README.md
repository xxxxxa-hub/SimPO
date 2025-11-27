# Client Inference with Chain-of-Thought (CoT) Reasoning

A modular pipeline for k-shot prompt evaluation with difference analysis and user persona inference.

## Architecture

The pipeline is organized into separate, focused modules:

### Core Modules

1. **`demo_reasoning_utils.py`**
   - `generate_demo_reasoning()`: Generate difference analysis for a single demonstration
   - `preprocess_demonstrations()`: Generate and save difference analysis for all demonstrations
   - Analyzes differences between response pairs from multiple perspectives
   - Saves analysis results to JSON format
   - **Purpose**: Extract reasoning patterns from demonstration pairs

2. **`persona_inference.py`**
   - `infer_persona_description()`: Generate inferred persona from demonstration patterns
   - `create_persona_context()`: Format persona for prompt inclusion
   - Uses differences analysis to infer user preferences and characteristics
   - **Purpose**: Understand what kind of user would make these preference choices

3. **`inference_clients.py`**
   - `InferenceClient`: Base class for all inference backends
   - `VLLMInferenceClient`: OpenAI-compatible VLLM server
   - `OpenAIInferenceClient`: OpenAI API client
   - `HuggingFaceInferenceClient`: HuggingFace Inference API client
   - `create_inference_client()`: Factory function
   - **Purpose**: Unified interface for different inference backends

4. **`prompt_utils.py`**
   - `create_prompt_template()`: Build prompts with k-shot demonstrations
   - Formats demonstrations with pre-generated reasoning
   - Structures test queries for evaluation
   - **Purpose**: Consistent prompt construction

5. **`main.py`**
   - Entry point for the evaluation pipeline
   - Orchestrates data loading, preprocessing, and evaluation
   - Handles multiprocessing for parallel test evaluation
   - **Purpose**: Complete workflow from data to results

## Key Concepts

### Difference Analysis vs Quality Judgment

The pipeline generates reasoning that **analyzes differences** between responses rather than judging which is "better". This is important because:
- Preferences are personalized and subjective
- Different users value different response characteristics
- The differences inform what kind of persona would prefer one over the other

### Two-Stage Processing

1. **Preprocessing Stage** (One-time):
   - Generate difference analysis for all demonstrations
   - Infer user persona from preference patterns
   - Save analysis to files

2. **Evaluation Stage** (Multiple samples):
   - Use pre-computed reasoning in prompts
   - Evaluate test examples with k-shot demonstrations
   - Extract probability predictions

## Usage

### Basic Usage

```bash
python main.py \
    --npz_file path/to/candidate_demos.npz \
    --persona_id "your-persona-uuid" \
    --inference_backend vllm \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --vllm_url http://localhost:8000
```

### With Output Files

```bash
python main.py \
    --npz_file path/to/candidate_demos.npz \
    --persona_id "your-persona-uuid" \
    --inference_backend vllm \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --reasoning_output_file ./demo_reasoning_analysis.json \
    --persona_output_file ./inferred_persona.json
```

### Arguments

**Required:**
- `--npz_file`: Path to NPZ file with `candidate_demo_indices`

**Model Configuration:**
- `--inference_backend`: Backend choice (vllm, openai, huggingface)
- `--model_name`: Model name/ID
- `--vllm_url`: VLLM server URL (default: http://localhost:8000)
- `--api_key`: API key if required

**Dataset:**
- `--dataset_name`: HuggingFace dataset (default: sher222/persona-iterative-responses)
- `--persona_id`: Specific persona UUID to filter for

**Evaluation:**
- `--k`: Number of demonstrations to sample (default: 4)
- `--num_samples`: Random samples to evaluate per test (default: 5)
- `--num_workers`: Parallel workers (default: 1)

**Output:**
- `--output_dir`: Directory for results (default: ./kshot_inference_results)
- `--reasoning_output_file`: Save difference analysis to JSON
- `--persona_output_file`: Save inferred persona to JSON

## Output Files

### Main Results (`output_dir/kshot_results_*.npz`)
Contains:
- `all_log_probs`: Log probabilities for all samples and orderings
- `all_probs`: Probabilities for all samples and orderings
- `avg_probs`: Average probability across samples (per test example)
- `std_probs`: Standard deviation across samples
- `avg_accuracy`: Accuracy for each test example
- Metadata: k, persona_id, model_name, etc.

### Difference Analysis (`reasoning_output_file`)
JSON format with:
```json
{
  "metadata": {
    "purpose": "Personalized user preference analysis",
    "analysis_type": "Response differences (not quality judgment)",
    "num_demonstrations": 123
  },
  "demonstrations": [
    {
      "demo_idx": 0,
      "question": "...",
      "response_a": "...",
      "response_b": "...",
      "differences_analysis": "..."
    }
  ]
}
```

### Inferred Persona (`persona_output_file`)
JSON format with:
```json
{
  "persona_description": "Based on the examples above, this user appears to be..."
}
```

## Module Dependencies

```
main.py
├── demo_reasoning_utils.py (preprocess_demonstrations)
├── persona_inference.py (infer_persona_description)
├── inference_clients.py (create_inference_client)
├── prompt_utils.py (create_prompt_template)
└── demo_selection_utils.py (parent dir)
```

## Data Flow

```
Input Dataset
    ↓
[Split: test/validation/training]
    ↓
Candidate Demonstrations (from NPZ)
    ↓
[Combine with Validation Set → Demo Pool]
    ↓
PREPROCESSING:
  ├→ Generate Difference Analysis for each demo
  ├→ Save Analysis to JSON (optional)
  ├→ Infer Persona from patterns
  └→ Save Persona to JSON (optional)
    ↓
EVALUATION (per sample):
  ├→ Sample k demonstrations from pool
  ├→ Evaluate all test examples with sampled demos
  └→ Extract probability predictions
    ↓
Results
    ├→ Average probabilities across samples
    ├→ Compute accuracy (prob > 0.5)
    └→ Save to NPZ
```

## Example Workflow

1. **Load demonstrations and preprocess once:**
   ```python
   from demo_reasoning_utils import preprocess_demonstrations
   from inference_clients import VLLMInferenceClient

   client = VLLMInferenceClient("http://localhost:8000", "meta-llama/Llama-3.1-8B-Instruct")
   reasonings = preprocess_demonstrations(
       client, demo_pool, flip_flags,
       output_file="reasoning_analysis.json"
   )
   ```

2. **Infer persona from demonstrations:**
   ```python
   from persona_inference import infer_persona_description

   persona = infer_persona_description(client, demo_pool, flip_flags, reasonings)
   ```

3. **Evaluate test examples using pre-computed reasoning:**
   ```python
   from prompt_utils import create_prompt_template

   for test_example in test_set:
       prompt = create_prompt_template(
           test_example['prompt'],
           test_example['response_a'],
           test_example['response_b'],
           demo_examples=sampled_demos,
           demo_flips=sampled_flips,
           demo_reasonings=sampled_reasonings  # Pre-computed
       )
   ```

## Notes

- The pipeline emphasizes **difference analysis** over quality judgment to respect that user preferences are personalized
- Pre-generating reasoning once and reusing it across evaluations is more efficient than regenerating for each test
- Persona inference helps understand what characteristics the user values, enabling better preference modeling
