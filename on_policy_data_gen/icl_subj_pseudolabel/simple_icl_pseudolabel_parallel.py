#!/usr/bin/env python3
"""
Iterative Context Optimization with Pseudo-Labeling for Binary Classification
(Accelerated Version - Fixed Test/Val Split)

Dataset: SetFit/subj (subjective vs objective text classification)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import argparse
import os
from tqdm import tqdm
import random
from accelerate import Accelerator
import sys

# CONSTANT SEED FOR DATA SPLITTING
# This ensures Test and Validation sets are IDENTICAL across all runs
# regardless of the experimental args.seed
FIXED_SPLIT_SEED = 42

def create_prompt(text, label=None, demonstrations=None):
    """Create ICL prompt for binary classification."""
    label_map = {0: "objective", 1: "subjective"}
    prompt = ""

    if demonstrations:
        for i, (demo_text, demo_label) in enumerate(demonstrations):
            prompt += f"Sentence: {demo_text}\n"
            prompt += f"Answer: {label_map[demo_label]}\n\n"

    prompt += f"Sentence: {text}\n"
    prompt += "Answer:"
    return prompt


def get_label_logits(model, tokenizer, prompt, device):
    """
    Get logits for 'objective' and 'subjective' tokens.
    Returns: log_probs tensor on device [2] (objective, subjective)
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]

    objective_id = tokenizer.encode(" objective", add_special_tokens=False)[0]
    subjective_id = tokenizer.encode(" subjective", add_special_tokens=False)[0]

    label_logits = torch.stack([
        logits[:, objective_id],
        logits[:, subjective_id]
    ], dim=1)

    return F.log_softmax(label_logits, dim=1)[0]


def sample_demonstration_pool(train_data, pool_size, seed):
    """
    Sample demonstrations using the EXPERIMENT seed, not the split seed.
    """
    # Use a local RandomState to ensure we use the specific args.seed
    # regardless of global state, though global is also set in main.
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(train_data), pool_size, replace=False)
    return [train_data[int(i)] for i in indices]


def generate_k_shot_permutations(demo_pool, k_shot):
    if k_shot % 2 != 0:
        raise ValueError(f"k_shot must be even for balanced labels, got {k_shot}")

    label_0_indices = [i for i, ex in enumerate(demo_pool) if ex['label'] == 0]
    label_1_indices = [i for i, ex in enumerate(demo_pool) if ex['label'] == 1]
    k_per_class = k_shot // 2

    from itertools import combinations, permutations
    combos_0 = list(combinations(label_0_indices, k_per_class))
    combos_1 = list(combinations(label_1_indices, k_per_class))

    all_perms = []
    for combo_0 in combos_0:
        for combo_1 in combos_1:
            combined = list(combo_0) + list(combo_1)
            for perm in permutations(combined):
                all_perms.append(perm)
    return all_perms


def evaluate_context_on_validation(model, tokenizer, demo_pool, demo_indices,
                                   validation_set, device, demo_labels=None):
    """
    Evaluate a single context on validation set. 
    Returns: Tensor of log_probs for the CORRECT class [val_size]
    """
    if demo_labels is None:
        demonstrations = [(demo_pool[i]['text'], demo_pool[i]['label']) for i in demo_indices]
    else:
        demonstrations = [(demo_pool[i]['text'], demo_labels[j]) for j, i in enumerate(demo_indices)]

    log_probs_correct = []

    for val_example in validation_set:
        text = val_example['text']
        true_label = val_example['label']
        prompt = create_prompt(text, demonstrations=demonstrations)
        
        log_probs = get_label_logits(model, tokenizer, prompt, device)
        log_probs_correct.append(log_probs[true_label])

    return torch.stack(log_probs_correct)


def compute_no_context_baseline(model, tokenizer, validation_set, device):
    """Compute baseline log probabilities without context."""
    log_probs_correct = []
    for val_example in validation_set:
        text = val_example['text']
        true_label = val_example['label']
        prompt = create_prompt(text, demonstrations=None)
        log_probs = get_label_logits(model, tokenizer, prompt, device)
        log_probs_correct.append(log_probs[true_label])
    return torch.stack(log_probs_correct)


def compute_snr(log_probs_with_context, log_probs_no_context):
    """Compute SNR using tensors."""
    gain = log_probs_with_context - log_probs_no_context
    mean_gain = gain.mean()
    std_gain = gain.std()
    snr = mean_gain / (std_gain + 1e-10)
    return snr


def run_icl_inference_distributed_logits(model, tokenizer, test_data_shard, demonstrations, device):
    """
    Run ICL inference on a shard.
    Returns: Tensor of shape [N, 2] containing log_probs for both classes.
    """
    log_probs_list = []
    
    for example in test_data_shard:
        text = example['text']
        prompt = create_prompt(text, demonstrations=demonstrations)
        log_probs = get_label_logits(model, tokenizer, prompt, device)
        log_probs_list.append(log_probs)

    if not log_probs_list:
        return torch.tensor([], device=device)
        
    return torch.stack(log_probs_list)


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative context optimization (Multi-GPU)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--k_shot", type=int, default=4)
    parser.add_argument("--pool_size", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=50)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--n_candidates", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./icl_optimized_results")
    parser.add_argument("--seed", type=int, default=42, help="Seed for demo pool and optimization (Data split is fixed)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"{'='*80}")
        print(f"Iterative Context Optimization (Fixed Split / Var Seed)")
        print(f"GPUs: {accelerator.num_processes}")
        print(f"{'='*80}")
        print(f"Model: {args.model_name}")
        print(f"Experimental Seed (Demo/Opt): {args.seed}")
        print(f"Data Split Seed (Fixed): {FIXED_SPLIT_SEED}")

    # Load model and tokenizer
    if is_main: print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    if is_main: print(f"Model loaded on {device}")

    # -------------------------------------------------------------------------
    # DATA LOADING & SPLITTING (FIXED SEED)
    # -------------------------------------------------------------------------
    dataset = load_dataset("SetFit/subj")
    all_data = list(dataset['train']) + list(dataset['test'])
    
    # Use a dedicated Random instance with the FIXED seed for splitting
    # This ensures test_data is identical across different runs of args.seed
    split_rng = random.Random(FIXED_SPLIT_SEED)
    split_rng.shuffle(all_data)
    
    test_data = all_data[:args.test_size]
    validation_data = all_data[args.test_size:args.test_size + args.val_size]
    train_data = all_data[args.test_size + args.val_size:]
    
    if is_main:
        print(f"Test set: {len(test_data)} examples")
        print(f"Validation set: {len(validation_data)} examples")
        print(f"Training set: {len(train_data)} examples")

    # -------------------------------------------------------------------------
    # SET GLOBAL SEEDS (VARIABLE EXPERIMENT SEED)
    # -------------------------------------------------------------------------
    # Now we set the global seeds using args.seed
    # This affects demo pool sampling and candidate optimization
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Sample pool using the experimental seed
    breakpoint()
    demo_pool = sample_demonstration_pool(train_data, args.pool_size, seed=args.seed)

    if is_main:
        print(f"\nSampling {args.pool_size} examples as demonstration pool (Seed {args.seed})...")

    # Generate permutations
    if is_main:
        print(f"\n{'='*80}")
        print(f"Step 1: Generate all {args.k_shot}-shot permutations")
        print(f"{'='*80}")

    all_permutations = generate_k_shot_permutations(demo_pool, args.k_shot)
    if is_main:
        print(f"Total permutations: {len(all_permutations)}")
        print(f"\nComputing no-context baseline on validation set...")
        
    log_probs_no_context = compute_no_context_baseline(model, tokenizer, validation_data, device)

    # Evaluate permutations (Distributed)
    if is_main:
        print(f"\n{'='*80}")
        print(f"Step 2: Evaluate all permutations on validation set (Distributed)")
        print(f"{'='*80}")

    with accelerator.split_between_processes(all_permutations) as my_perms:
        my_val_log_probs = []
        my_snr_scores = []
        
        iterator = tqdm(my_perms, desc="Evaluating permutations") if accelerator.is_local_main_process else my_perms
        
        for perm_indices in iterator:
            log_probs_with_context = evaluate_context_on_validation(
                model, tokenizer, demo_pool, perm_indices, validation_data, device
            )
            snr = compute_snr(log_probs_with_context, log_probs_no_context)
            my_val_log_probs.append(log_probs_with_context)
            my_snr_scores.append(snr)
        
        if len(my_val_log_probs) > 0:
            my_val_probs_tensor = torch.stack(my_val_log_probs) 
            my_snr_tensor = torch.stack(my_snr_scores)
        else:
            my_val_probs_tensor = torch.tensor([], device=device)
            my_snr_tensor = torch.tensor([], device=device)

    all_val_log_probs_tensor = accelerator.gather(my_val_probs_tensor)
    all_snr_tensor = accelerator.gather(my_snr_tensor)
    
    all_val_log_probs_cpu = all_val_log_probs_tensor.cpu().numpy()[:len(all_permutations)]
    all_snr_cpu = all_snr_tensor.cpu().numpy()[:len(all_permutations)]

    # Select top N
    if is_main:
        print(f"\n{'='*80}")
        print(f"Step 3: Select top {args.top_n} contexts by SNR")
        print(f"{'='*80}")

    top_indices = np.argsort(all_snr_cpu)[::-1][:args.top_n]
    initial_contexts = []

    for i, idx in enumerate(top_indices):
        perm_indices = list(all_permutations[idx])
        initial_contexts.append(perm_indices)
        
        if is_main:
            snr = all_snr_cpu[idx]
            avg_log_prob = np.mean(all_val_log_probs_cpu[idx])
            print(f"\nContext {i+1} (rank {i+1}):")
            print(f"  Demo indices: {perm_indices}")
            print(f"  SNR: {snr:.4f}")
            print(f"  Avg Log Prob: {avg_log_prob:.4f}")

    # Prepare Candidates
    if is_main:
        print(f"\n{'='*80}")
        print(f"Step 4: Prepare test examples as replacement candidates")
        print(f"{'='*80}")

    # Candidates are selected using the args.seed (since global random is set to it now)
    # This implies optimization stochasticity is tied to the experiment seed
    test_subset_indices = random.sample(range(len(test_data)), min(args.n_candidates, len(test_data)))
    test_subset = [test_data[i] for i in test_subset_indices]
    
    candidate_examples = demo_pool + test_subset
    test_offset = args.pool_size
    
    if is_main:
        print(f"Total candidate pool: {len(candidate_examples)}")

    # Optimization
    if is_main:
        print(f"\n{'='*80}")
        print(f"Step 5: Iterative optimization (Distributed)")
        print(f"{'='*80}")

    optimized_contexts = []
    optimized_labels = []
    all_candidate_indices = list(range(test_offset, len(candidate_examples)))

    for ctx_num, initial_indices in enumerate(initial_contexts):
        if is_main:
            print(f"\nOptimizing context {ctx_num + 1}/{args.top_n}")

        initial_labels = [demo_pool[i]['label'] for i in initial_indices]
        current_indices = initial_indices.copy()
        current_labels = initial_labels.copy()

        base_log_probs = evaluate_context_on_validation(
            model, tokenizer, candidate_examples, current_indices, validation_data, device, demo_labels=current_labels
        )
        current_score = compute_snr(base_log_probs, log_probs_no_context).item()

        if is_main:
            print(f"  Initial SNR: {current_score:.4f}")

        for pos in range(args.k_shot):
            target_label = current_labels[pos]
            lbl_name = "objective" if target_label == 0 else "subjective"
            
            with accelerator.split_between_processes(all_candidate_indices) as my_candidates:
                my_best_score = -9999.0
                my_best_idx = current_indices[pos]

                iterator = tqdm(my_candidates, desc="Search", leave=False) if accelerator.is_local_main_process else my_candidates

                for test_idx in iterator:
                    candidate_indices = current_indices.copy()
                    candidate_indices[pos] = test_idx
                    
                    c_log_probs = evaluate_context_on_validation(
                        model, tokenizer, candidate_examples, candidate_indices, 
                        validation_data, device, demo_labels=current_labels
                    )
                    c_score = compute_snr(c_log_probs, log_probs_no_context).item()

                    if c_score > my_best_score:
                        my_best_score = c_score
                        my_best_idx = test_idx
                
                my_result = torch.tensor([my_best_score, float(my_best_idx)], device=device)

            gathered = accelerator.gather(my_result)
            flat = gathered.view(-1, 2)
            best_row = torch.argmax(flat[:, 0])
            global_best_score = flat[best_row, 0].item()
            global_best_idx = int(flat[best_row, 1].item())

            if global_best_score > current_score:
                if is_main:
                    print(f"    Pos {pos}: Improved {current_score:.4f} -> {global_best_score:.4f}")
                current_indices[pos] = global_best_idx
                current_score = global_best_score

        optimized_contexts.append(current_indices)
        optimized_labels.append(current_labels)

    # Testing
    if is_main:
        print(f"\n{'='*80}")
        print(f"Step 6: Testing contexts on full test set (Distributed)")
        print(f"{'='*80}")

    def run_distributed_test_probs(demonstrations):
        with accelerator.split_between_processes(test_data) as my_test_shard:
            my_logits = run_icl_inference_distributed_logits(model, tokenizer, my_test_shard, demonstrations, device)
        
        all_logits = accelerator.gather(my_logits).cpu().numpy()[:len(test_data)]
        ground_truth = np.array([ex['label'] for ex in test_data])
        correct_log_probs = all_logits[np.arange(len(all_logits)), ground_truth]
        return correct_log_probs

    if is_main: print(f"Testing initial contexts...")
    initial_test_probs = []

    for i, context_indices in enumerate(initial_contexts):
        ctx_labels = [demo_pool[idx]['label'] for idx in context_indices]
        demos = [(candidate_examples[idx]['text'], ctx_labels[j]) for j, idx in enumerate(context_indices)]
        
        probs = run_distributed_test_probs(demos)
        initial_test_probs.append(probs)
        
        if is_main:
            print(f"  Context {i+1}: Mean Log Prob = {np.mean(probs):.4f}")

    if is_main: print(f"Testing optimized contexts...")
    optimized_test_probs = []

    for i, (context_indices, context_labels) in enumerate(zip(optimized_contexts, optimized_labels)):
        demos = [(candidate_examples[idx]['text'], context_labels[j]) for j, idx in enumerate(context_indices)]
        
        probs = run_distributed_test_probs(demos)
        optimized_test_probs.append(probs)
        
        if is_main:
            print(f"  Context {i+1}: Mean Log Prob = {np.mean(probs):.4f}")

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"optimized_contexts_k{args.k_shot}_probs.npz")

        demo_pool_serial = [{'text': ex['text'], 'label': int(ex['label'])} for ex in demo_pool]

        np.savez(
            output_file,
            all_permutations_val_log_probs=all_val_log_probs_cpu,
            initial_test_log_probs=np.array(initial_test_probs),
            optimized_test_log_probs=np.array(optimized_test_probs),
            initial_contexts=initial_contexts,
            initial_context_labels=initial_context_labels,
            optimized_contexts=optimized_contexts,
            optimized_labels=optimized_labels,
            test_subset_indices=test_subset_indices,
            demo_pool=demo_pool_serial,
            k_shot=args.k_shot,
            pool_size=args.pool_size
        )

        print(f"\nResults saved to: {output_file}")
        print(f"  - all_permutations_val_log_probs: {all_val_log_probs_cpu.shape}")
        print(f"  - initial_test_log_probs: {np.array(initial_test_probs).shape}")
        print(f"  - optimized_test_log_probs: {np.array(optimized_test_probs).shape}")
        print(f"\n{'='*80}")
        print(f"Done!")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()