
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import argparse
import tqdm
import numpy as np
import datasets
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--input_dataset", type=str, default=None, help="HuggingFace dataset ID to load (e.g., 'username/dataset-name')")
parser.add_argument("--generation_file", type=str, default=None, help="Path to the output generation file (alternative to --input_dataset)")
parser.add_argument("--reward_model", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1", help="Path to reward model")
parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory (for local saving)")
parser.add_argument("--repo_id", type=str, default=None, help="HuggingFace repository ID to push to (e.g., 'username/dataset-name')")
parser.add_argument("--private", action="store_true", default=False, help="Make the repository private")
args = parser.parse_args()

# Validate arguments
if not args.input_dataset and not args.generation_file:
    parser.error("Either --input_dataset or --generation_file must be provided")
if args.input_dataset and args.generation_file:
    parser.error("Cannot specify both --input_dataset and --generation_file")

print(args)

# Load data from HuggingFace or local file
if args.input_dataset:
    print(f"Loading dataset from HuggingFace: {args.input_dataset}")
    hf_dataset = load_dataset(args.input_dataset)
    # Keep splits separate
    output_data_by_split = {}
    for split_name in hf_dataset.keys():
        output_data_by_split[split_name] = list(hf_dataset[split_name])
        print(f"Loaded {len(output_data_by_split[split_name])} samples from split '{split_name}'")

    # Also create a flat list for processing
    output_data = []
    for split_data in output_data_by_split.values():
        output_data.extend(split_data)
else:
    generation_file = args.generation_file
    with open(generation_file, 'r') as f:
        output_data = json.load(f)
    print(f"Loaded {len(output_data)} samples from local file")
    output_data_by_split = None

inputs = [data["prompt"] for data in output_data]
candidates_texts = [data["all_generated_responses"] for data in output_data]

model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, 
                                                           device_map="cuda", 
                                                           trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)

for data in tqdm.tqdm(output_data):
    prompt = data["prompt"]
    candidates = data["all_generated_responses"]
    scores = []
    for candidate in candidates:
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": candidate}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model(input_ids)
            score = output.score.float().item()
            scores.append(score)
    data["all_rm_scores"] = scores

# Save annotated outputs locally if output_dir is provided
if args.output_dir:
    if args.generation_file:
        file_name = os.path.basename(args.generation_file).split('.json')[0] + "_rm.json"
    else:
        file_name = "annotated_rm.json"

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, file_name), 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Annotated outputs saved to {os.path.join(args.output_dir, file_name)}")

# Binarize data: win = highest scoring reponse; lose = lowest scoring response
for data in output_data:
    chosen_idx = np.argmax(data["all_rm_scores"])
    rejected_idx = np.argmin(data["all_rm_scores"])
    chosen = []
    chosen.append({
        "role": "user",
        "content": data["prompt"]
    })
    chosen.append({
        "role": "assistant",
        "content": data["all_generated_responses"][chosen_idx]
    })
    rejected = []
    rejected.append({
        "role": "user",
        "content": data["prompt"]
    })
    rejected.append({
        "role": "assistant",
        "content": data["all_generated_responses"][rejected_idx]
    })
    data.update({
        "chosen": chosen,
        "rejected": rejected,
    })

# Save binarized outputs locally if output_dir is provided
if args.output_dir:
    if args.generation_file:
        output_file = os.path.basename(args.generation_file).split('.json')[0] + "_bin.json"
    else:
        output_file = "binarized.json"

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, output_file), 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Binarized outputs saved to {os.path.join(args.output_dir, output_file)}")

# Convert the data to Hugging Face datasets format
if output_data_by_split:
    # Reconstruct the splits by tracking indices
    # Create a mapping to track which split each processed sample belongs to
    split_indices = {}
    idx = 0
    for split_name, split_data in output_data_by_split.items():
        split_indices[split_name] = (idx, idx + len(split_data))
        idx += len(split_data)

    # Create datasets for each split
    dataset_dict = {}
    for split_name, (start_idx, end_idx) in split_indices.items():
        split_output_data = output_data[start_idx:end_idx]
        dataset_dict[split_name] = datasets.Dataset.from_list(split_output_data)
        print(f"Created dataset for split '{split_name}' with {len(split_output_data)} samples")

    # Create DatasetDict
    dataset_dict = datasets.DatasetDict(dataset_dict)

    # Save to disk if output_dir is provided
    if args.output_dir:
        dataset_dict.save_to_disk(args.output_dir)
        print(f"Binarized dataset with splits saved to {args.output_dir}")

    # Push to HuggingFace Hub if repo_id is provided
    if args.repo_id:
        print(f"Pushing dataset to HuggingFace Hub: {args.repo_id}")
        dataset_dict.push_to_hub(
            repo_id=args.repo_id,
            private=args.private
        )
        print(f"Successfully pushed dataset with splits {list(dataset_dict.keys())} to https://huggingface.co/datasets/{args.repo_id}")
else:
    # Single dataset (no splits to preserve)
    dataset = datasets.Dataset.from_list(output_data)

    # Save to disk if output_dir is provided
    if args.output_dir:
        dataset.save_to_disk(args.output_dir)
        print(f"Binarized dataset saved to {args.output_dir}")

    # Push to HuggingFace Hub if repo_id is provided
    if args.repo_id:
        print(f"Pushing dataset to HuggingFace Hub: {args.repo_id}")
        dataset.push_to_hub(
            repo_id=args.repo_id,
            private=args.private
        )
        print(f"Successfully pushed dataset to https://huggingface.co/datasets/{args.repo_id}")
