#!/usr/bin/env python3
"""
Script to analyze sentence embeddings for question+response pairs.

This script:
1. Loads preference data with questions and chosen/rejected responses
2. Computes sentence embeddings for question+response concatenations
3. Performs dimensionality reduction (t-SNE and UMAP)
4. Performs clustering analysis
5. Visualizes embeddings to check if winning vs losing responses are separable
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP if available
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")


def load_preference_data(
    dataset_name: str = "sher222/persona-iterative-responses",
    level_filter: str = "309fa18d-e081-481f-9957-80af59494c12",
    max_samples: int = None
) -> List[Dict]:
    """
    Load preference data from HuggingFace dataset.

    Args:
        dataset_name: Name of the HuggingFace dataset
        level_filter: Value to filter the 'level' column by
        max_samples: Maximum number of samples to load

    Returns:
        List of dictionaries with question, chosen, rejected responses
    """
    print(f"Loading preference data from HuggingFace: {dataset_name}")
    print(f"Filtering for level: {level_filter}")

    # Load dataset
    dataset = load_dataset(dataset_name, split='validation')
    print(f"Total dataset size: {len(dataset)}")

    # Filter by level
    if level_filter:
        dataset = dataset.filter(lambda x: x['level'] == level_filter)
        print(f"Filtered dataset size: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError(f"No data found for level '{level_filter}'")

    # Limit samples if requested
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"Using {max_samples} samples")

    preference_pairs = []

    for idx, sample in enumerate(dataset):
        # Extract fields with strip()
        prompt = sample['x'].strip() if sample.get('x') else ''
        chosen = sample['yw'].strip() if sample.get('yw') else ''
        rejected = sample['yl'].strip() if sample.get('yl') else ''

        # Skip if any field is empty
        if not prompt or not chosen or not rejected:
            print(f"Warning: Skipping sample {idx} due to empty fields")
            continue

        preference_pairs.append({
            'question': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'sample_id': idx
        })

    print(f"Loaded {len(preference_pairs)} preference pairs")
    return preference_pairs


def load_classification_data(
    dataset_name: str = "gimmaru/glue-sst2",
    text_field: str = "sentence",
    label_field: str = "label",
    max_samples: int = None
) -> List[Dict]:
    """
    Load classification data from HuggingFace dataset.

    Args:
        dataset_name: Name of the HuggingFace dataset
        text_field: Name of the field containing text
        label_field: Name of the field containing labels (0 or 1)
        max_samples: Maximum number of samples to load

    Returns:
        List of dictionaries with text and label
    """
    print(f"Loading classification data from HuggingFace: {dataset_name}")

    # Load dataset
    dataset = load_dataset(dataset_name, split='validation')
    print(f"Total dataset size: {len(dataset)}")

    # Limit samples if requested
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"Using {max_samples} samples")

    classification_data = []

    for idx, sample in enumerate(dataset):
        # Extract fields with strip()
        text = sample[text_field].strip() if sample.get(text_field) else ''
        label = sample[label_field] if sample.get(label_field) is not None else None

        # Skip if any field is empty or invalid
        if not text or label is None:
            print(f"Warning: Skipping sample {idx} due to empty fields")
            continue

        classification_data.append({
            'text': text,
            'label': int(label),  # 0 or 1
            'sample_id': idx
        })

    print(f"Loaded {len(classification_data)} classification samples")

    # Print label distribution
    labels_array = np.array([d['label'] for d in classification_data])
    unique, counts = np.unique(labels_array, return_counts=True)
    print(f"Label distribution:")
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} samples ({count/len(classification_data)*100:.1f}%)")

    return classification_data


def load_elix_preference_data(
    dataset_name: str = "Asap7772/elix_generations_gpt4omini_pref",
    level_id_x: int = 1,
    level_id_y: int = 4,
    scorer_level_id: int = 4,
    max_samples: int = None
) -> List[Dict]:
    """
    Load elix preference data from HuggingFace dataset.

    Args:
        dataset_name: Name of the HuggingFace dataset
        level_id_x: Value to filter level_id_x by
        level_id_y: Value to filter level_id_y by
        scorer_level_id: Value to filter scorer_level_id by
        max_samples: Maximum number of samples to load

    Returns:
        List of dictionaries with question, chosen, rejected responses
    """
    print(f"Loading elix preference data from HuggingFace: {dataset_name}")
    print(f"Filtering: level_id_x={level_id_x}, level_id_y={level_id_y}, scorer_level_id={scorer_level_id}")

    # Load dataset
    dataset = load_dataset(dataset_name, split='train')
    print(f"Total dataset size: {len(dataset)}")

    # Filter by level_id_x, level_id_y, and scorer_level_id
    dataset = dataset.filter(
        lambda x: x.get('level_id_x') == level_id_x and
                  x.get('level_id_y') == level_id_y and
                  x.get('scorer_level_id') == scorer_level_id
    )
    print(f"Filtered dataset size: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError(f"No data found for the specified filters")

    # Limit samples if requested
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"Using {max_samples} samples")

    preference_pairs = []

    for idx, sample in enumerate(dataset):
        # Extract fields
        prompt = sample.get('prompt', '').strip() if sample.get('prompt') else ''
        response_x = sample.get('response_x', '').strip() if sample.get('response_x') else ''
        response_y = sample.get('response_y', '').strip() if sample.get('response_y') else ''
        det_choice = sample.get('det_choice')

        # Skip if any field is empty or invalid
        if not prompt or not response_x or not response_y or det_choice is None:
            print(f"Warning: Skipping sample {idx} due to empty fields")
            continue

        # Determine chosen and rejected based on det_choice
        # det_choice==1: response_y is preferred (chosen)
        # det_choice==0: response_x is preferred (chosen)
        if det_choice == 1:
            chosen = response_y
            rejected = response_x
        else:  # det_choice == 0
            chosen = response_x
            rejected = response_y

        preference_pairs.append({
            'question': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'sample_id': idx
        })

    print(f"Loaded {len(preference_pairs)} preference pairs")

    # Print choice distribution
    choices = [sample.get('det_choice') for sample in dataset if sample.get('det_choice') is not None]
    if choices:
        unique, counts = np.unique(choices, return_counts=True)
        print(f"det_choice distribution:")
        for choice, count in zip(unique, counts):
            if choice == 1:
                print(f"  det_choice={choice} (response_y preferred): {count} samples ({count/len(choices)*100:.1f}%)")
            else:
                print(f"  det_choice={choice} (response_x preferred): {count} samples ({count/len(choices)*100:.1f}%)")

    return preference_pairs


def compute_embeddings(
    preference_pairs: List[Dict],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    device: str = None
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Compute sentence embeddings for question+response pairs.

    Args:
        preference_pairs: List of preference pair dictionaries
        model_name: Name of sentence-transformers model
        batch_size: Batch size for encoding
        device: Device to use (cuda/cpu)

    Returns:
        Tuple of (embeddings_array, labels_array, sample_ids)
        - embeddings_array: (2N, embedding_dim) array
        - labels_array: (2N,) array with 0=rejected, 1=chosen
        - sample_ids: (2N,) list of sample IDs
    """
    print(f"\nLoading sentence embedding model: {model_name}")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SentenceTransformer(model_name, device=device)
    print(f"Model loaded on {device}")

    # Prepare texts: question + response concatenations
    texts_chosen = []
    texts_rejected = []
    sample_ids = []

    for pair in preference_pairs:
        question = pair['question']
        chosen = pair['chosen']
        rejected = pair['rejected']
        sample_id = pair['sample_id']

        # Concatenate question with response
        text_chosen = f"Question: {question}\n\nResponse: {chosen}"
        text_rejected = f"Question: {question}\n\nResponse: {rejected}"

        texts_chosen.append(text_chosen)
        texts_rejected.append(text_rejected)
        sample_ids.extend([sample_id, sample_id])

    print(f"\nComputing embeddings for {len(texts_chosen)} chosen responses...")
    embeddings_chosen = model.encode(
        texts_chosen,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Computing embeddings for {len(texts_rejected)} rejected responses...")
    embeddings_rejected = model.encode(
        texts_rejected,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Combine embeddings and create labels
    # Interleave chosen and rejected for each sample
    embeddings_list = []
    labels_list = []

    for i in range(len(embeddings_chosen)):
        embeddings_list.append(embeddings_chosen[i])
        labels_list.append(1)  # 1 for chosen
        embeddings_list.append(embeddings_rejected[i])
        labels_list.append(0)  # 0 for rejected

    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Chosen responses: {np.sum(labels == 1)}")
    print(f"Rejected responses: {np.sum(labels == 0)}")

    return embeddings, labels, sample_ids


def compute_embeddings_classification(
    classification_data: List[Dict],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    device: str = None
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Compute sentence embeddings for classification data.

    Args:
        classification_data: List of classification data dictionaries
        model_name: Name of sentence-transformers model
        batch_size: Batch size for encoding
        device: Device to use (cuda/cpu)

    Returns:
        Tuple of (embeddings_array, labels_array, sample_ids)
        - embeddings_array: (N, embedding_dim) array
        - labels_array: (N,) array with classification labels (0 or 1)
        - sample_ids: (N,) list of sample IDs
    """
    print(f"\nLoading sentence embedding model: {model_name}")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SentenceTransformer(model_name, device=device)
    print(f"Model loaded on {device}")

    # Prepare texts and labels
    texts = []
    labels = []
    sample_ids = []

    for data in classification_data:
        texts.append(data['text'])
        labels.append(data['label'])
        sample_ids.append(data['sample_id'])

    print(f"\nComputing embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    labels = np.array(labels)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Print label distribution
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {label}: {count} samples ({count/len(labels)*100:.1f}%)")

    return embeddings, labels, sample_ids


def perform_dimensionality_reduction(
    embeddings: np.ndarray,
    method: str = 'tsne',
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embeddings to 2D for visualization.

    Args:
        embeddings: High-dimensional embeddings
        method: 'tsne', 'umap', or 'pca'
        n_components: Number of components (typically 2)
        random_state: Random seed

    Returns:
        2D embeddings
    """
    print(f"\nPerforming {method.upper()} dimensionality reduction...")

    if method == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=30,
            n_iter=1000,
            verbose=1
        )
    elif method == 'umap':
        if not UMAP_AVAILABLE:
            print("UMAP not available, falling back to t-SNE")
            return perform_dimensionality_reduction(embeddings, 'tsne', n_components, random_state)
        reducer = UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
            verbose=True
        )
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    embeddings_2d = reducer.fit_transform(embeddings)
    print(f"Reduced to shape: {embeddings_2d.shape}")

    return embeddings_2d


def perform_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 42
) -> Dict:
    """
    Perform k-means clustering and compute metrics.

    Args:
        embeddings: Embeddings to cluster
        labels: True labels (0=rejected, 1=chosen)
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Dictionary with clustering results and metrics
    """
    print(f"\nPerforming k-means clustering with {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Compute clustering metrics
    silhouette = silhouette_score(embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)

    # Compute purity with respect to true labels
    # For each cluster, find the most common true label
    purity_scores = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if np.sum(cluster_mask) == 0:
            continue
        cluster_true_labels = labels[cluster_mask]
        # Purity: fraction of most common label in cluster
        unique, counts = np.unique(cluster_true_labels, return_counts=True)
        purity = np.max(counts) / np.sum(counts)
        purity_scores.append(purity)

    avg_purity = np.mean(purity_scores) if purity_scores else 0

    # Compute accuracy of clustering vs true labels
    # Try both assignments (cluster 0->label 0 and cluster 0->label 1)
    acc1 = np.mean(cluster_labels == labels)
    acc2 = np.mean(cluster_labels != labels)
    clustering_accuracy = max(acc1, acc2)

    results = {
        'cluster_labels': cluster_labels,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'purity': avg_purity,
        'clustering_accuracy': clustering_accuracy
    }

    print(f"\nClustering Metrics:")
    print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range [-1, 1])")
    print(f"  Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz:.4f} (higher is better)")
    print(f"  Average Purity: {avg_purity:.4f}")
    print(f"  Clustering Accuracy: {clustering_accuracy:.4f}")

    return results


def visualize_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    clustering_results: Dict,
    output_path: str,
    method: str = 'tsne',
    label_names: Dict[int, str] = None
):
    """
    Create visualization of embeddings with true labels and cluster assignments.

    Args:
        embeddings_2d: 2D embeddings
        labels: True labels (0 or 1)
        clustering_results: Results from clustering
        output_path: Path to save figure
        method: Dimensionality reduction method name
        label_names: Optional dict mapping label values to names (e.g., {0: 'Negative', 1: 'Positive'})
    """
    print(f"\nCreating visualization...")

    # Default label names if not provided
    if label_names is None:
        label_names = {0: 'Label 0', 1: 'Label 1'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: True labels
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        embeddings_2d[labels == 0, 0],
        embeddings_2d[labels == 0, 1],
        c='red',
        alpha=0.6,
        s=50,
        label=label_names[0],
        edgecolors='black',
        linewidth=0.5
    )
    scatter2 = ax1.scatter(
        embeddings_2d[labels == 1, 0],
        embeddings_2d[labels == 1, 1],
        c='blue',
        alpha=0.6,
        s=50,
        label=label_names[1],
        edgecolors='black',
        linewidth=0.5
    )

    ax1.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax1.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax1.set_title(f'Sentence Embeddings: True Labels', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cluster assignments
    ax2 = axes[1]
    cluster_labels = clustering_results['cluster_labels']
    scatter3 = ax2.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=cluster_labels,
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )

    ax2.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax2.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax2.set_title(
        f'K-Means Clustering (k={len(np.unique(cluster_labels))})\n'
        f'Silhouette: {clustering_results["silhouette_score"]:.3f}, '
        f'Accuracy: {clustering_results["clustering_accuracy"]:.3f}',
        fontsize=14,
        fontweight='bold'
    )
    cbar = plt.colorbar(scatter3, ax=ax2)
    cbar.set_label('Cluster ID', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

    # Also create a detailed separability analysis plot
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))

    # Plot 1: Overlay of both labels with larger size
    ax = axes2[0, 0]
    ax.scatter(
        embeddings_2d[labels == 0, 0],
        embeddings_2d[labels == 0, 1],
        c='red',
        alpha=0.4,
        s=80,
        label=label_names[0],
        edgecolors='darkred',
        linewidth=0.8
    )
    ax.scatter(
        embeddings_2d[labels == 1, 0],
        embeddings_2d[labels == 1, 1],
        c='blue',
        alpha=0.4,
        s=80,
        label=label_names[1],
        edgecolors='darkblue',
        linewidth=0.8
    )
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title('Label Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Density plot for label 0
    ax = axes2[0, 1]
    label0_points = embeddings_2d[labels == 0]
    ax.hexbin(label0_points[:, 0], label0_points[:, 1], gridsize=30, cmap='Reds', alpha=0.7)
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(f'Density: {label_names[0]}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Density plot for label 1
    ax = axes2[1, 0]
    label1_points = embeddings_2d[labels == 1]
    ax.hexbin(label1_points[:, 0], label1_points[:, 1], gridsize=30, cmap='Blues', alpha=0.7)
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(f'Density: {label_names[1]}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Metrics summary
    ax = axes2[1, 1]
    ax.axis('off')

    # Calculate additional separability metrics
    from scipy.spatial.distance import cdist

    chosen_center = np.mean(embeddings_2d[labels == 1], axis=0)
    rejected_center = np.mean(embeddings_2d[labels == 0], axis=0)
    center_distance = np.linalg.norm(chosen_center - rejected_center)

    chosen_std = np.std(embeddings_2d[labels == 1], axis=0)
    rejected_std = np.std(embeddings_2d[labels == 0], axis=0)
    avg_std = np.mean(np.concatenate([chosen_std, rejected_std]))

    metrics_text = f"""
    Separability Analysis
    {'=' * 40}

    Clustering Metrics:
      • Silhouette Score: {clustering_results['silhouette_score']:.4f}
      • Davies-Bouldin: {clustering_results['davies_bouldin_score']:.4f}
      • Calinski-Harabasz: {clustering_results['calinski_harabasz_score']:.4f}
      • Purity: {clustering_results['purity']:.4f}
      • Accuracy: {clustering_results['clustering_accuracy']:.4f}

    Geometric Metrics:
      • Center Distance: {center_distance:.4f}
      • Average Std Dev: {avg_std:.4f}
      • Separation Ratio: {center_distance / avg_std:.4f}

    Data Summary:
      • Total Points: {len(labels)}
      • Label 1 ({label_names[1]}): {np.sum(labels == 1)}
      • Label 0 ({label_names[0]}): {np.sum(labels == 0)}

    Interpretation:
      • Higher Silhouette (→ 1): Better separation
      • Lower Davies-Bouldin (→ 0): Better separation
      • Higher C-H Score: More distinct clusters
      • Higher Separation Ratio: More separable
    """

    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    detailed_path = output_path.replace('.png', '_detailed.png')
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
    print(f"Detailed visualization saved to {detailed_path}")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Analyze sentence embeddings for preference or classification data")
    parser.add_argument("--dataset_type", type=str, default="preference",
                       choices=['preference', 'classification', 'elix'],
                       help="Type of dataset: 'preference' (question+chosen/rejected), 'classification' (text+label), or 'elix' (elix preference format)")
    parser.add_argument("--dataset_name", type=str, default="sher222/persona-iterative-responses",
                       help="HuggingFace dataset name")
    parser.add_argument("--level_filter", type=str, default="309fa18d-e081-481f-9957-80af59494c12",
                       help="Value to filter the 'level' column by (only for preference datasets)")
    parser.add_argument("--text_field", type=str, default="sentence",
                       help="Name of text field (only for classification datasets)")
    parser.add_argument("--label_field", type=str, default="label",
                       help="Name of label field (only for classification datasets)")
    parser.add_argument("--level_id_x", type=int, default=1,
                       help="Value to filter level_id_x by (only for elix datasets)")
    parser.add_argument("--level_id_y", type=int, default=4,
                       help="Value to filter level_id_y by (only for elix datasets)")
    parser.add_argument("--scorer_level_id", type=int, default=4,
                       help="Value to filter scorer_level_id by (only for elix datasets)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                       help="Sentence-transformers model name")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for encoding")
    parser.add_argument("--reduction_method", type=str, default="tsne",
                       choices=['tsne', 'umap', 'pca'],
                       help="Dimensionality reduction method")
    parser.add_argument("--n_clusters", type=int, default=2,
                       help="Number of clusters for k-means")
    parser.add_argument("--output_dir", type=str, default="./embedding_analysis",
                       help="Directory to save results")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Sentence Embedding Analysis for {args.dataset_type.capitalize()} Data")
    print("=" * 80)
    print(f"Dataset type: {args.dataset_type}")
    print(f"Dataset: {args.dataset_name}")
    if args.dataset_type == 'preference':
        print(f"Level filter: {args.level_filter}")
    elif args.dataset_type == 'classification':
        print(f"Text field: {args.text_field}")
        print(f"Label field: {args.label_field}")
    elif args.dataset_type == 'elix':
        print(f"level_id_x filter: {args.level_id_x}")
        print(f"level_id_y filter: {args.level_id_y}")
        print(f"scorer_level_id filter: {args.scorer_level_id}")
    print(f"Model: {args.model_name}")
    print(f"Reduction method: {args.reduction_method}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Load data based on dataset type
    if args.dataset_type == 'preference':
        data = load_preference_data(
            dataset_name=args.dataset_name,
            level_filter=args.level_filter,
            max_samples=args.max_samples
        )
        num_samples = len(data)

        # Compute embeddings
        embeddings, labels, sample_ids = compute_embeddings(
            data,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device
        )
    elif args.dataset_type == 'classification':
        data = load_classification_data(
            dataset_name=args.dataset_name,
            text_field=args.text_field,
            label_field=args.label_field,
            max_samples=args.max_samples
        )
        num_samples = len(data)

        # Compute embeddings
        embeddings, labels, sample_ids = compute_embeddings_classification(
            data,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device
        )
    else:  # elix
        data = load_elix_preference_data(
            dataset_name=args.dataset_name,
            level_id_x=args.level_id_x,
            level_id_y=args.level_id_y,
            scorer_level_id=args.scorer_level_id,
            max_samples=args.max_samples
        )
        num_samples = len(data)

        # Compute embeddings
        embeddings, labels, sample_ids = compute_embeddings(
            data,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device
        )

    # Save embeddings
    embeddings_path = output_dir / "embeddings.npz"
    np.savez(
        embeddings_path,
        embeddings=embeddings,
        labels=labels,
        sample_ids=sample_ids
    )
    print(f"\nEmbeddings saved to {embeddings_path}")

    # Dimensionality reduction
    embeddings_2d = perform_dimensionality_reduction(
        embeddings,
        method=args.reduction_method,
        random_state=args.random_seed
    )

    # Clustering
    clustering_results = perform_clustering(
        embeddings_2d,
        labels,
        n_clusters=args.n_clusters,
        random_state=args.random_seed
    )

    # Visualization
    output_path = output_dir / f"embeddings_{args.reduction_method}.png"

    # Set label names based on dataset type
    if args.dataset_type == 'preference':
        label_names = {0: 'Rejected (yl)', 1: 'Chosen (yw)'}
    elif args.dataset_type == 'elix':
        label_names = {0: 'Non-preferred', 1: 'Preferred'}
    else:  # classification
        # For classification datasets, use more generic names
        # You can customize this based on the specific dataset
        if args.dataset_name == "gimmaru/glue-sst2" or args.dataset_name == "stanfordnlp/sst2":
            label_names = {0: 'Negative', 1: 'Positive'}
        else:
            label_names = {0: 'Label 0', 1: 'Label 1'}

    visualize_embeddings(
        embeddings_2d,
        labels,
        clustering_results,
        str(output_path),
        method=args.reduction_method,
        label_names=label_names
    )

    # Save results summary
    results_summary = {
        'dataset_type': args.dataset_type,
        'dataset_name': args.dataset_name,
        'model_name': args.model_name,
        'num_samples': num_samples,
        'num_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'reduction_method': args.reduction_method,
        'clustering_metrics': {
            'silhouette_score': float(clustering_results['silhouette_score']),
            'davies_bouldin_score': float(clustering_results['davies_bouldin_score']),
            'calinski_harabasz_score': float(clustering_results['calinski_harabasz_score']),
            'purity': float(clustering_results['purity']),
            'clustering_accuracy': float(clustering_results['clustering_accuracy'])
        }
    }

    # Add dataset-specific fields to summary
    if args.dataset_type == 'preference':
        results_summary['level_filter'] = args.level_filter
    elif args.dataset_type == 'elix':
        results_summary['level_id_x'] = args.level_id_x
        results_summary['level_id_y'] = args.level_id_y
        results_summary['scorer_level_id'] = args.scorer_level_id
    else:  # classification
        results_summary['text_field'] = args.text_field
        results_summary['label_field'] = args.label_field

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults summary saved to {summary_path}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"\nKey Finding: Clustering Accuracy = {clustering_results['clustering_accuracy']:.2%}")
    if clustering_results['clustering_accuracy'] > 0.7:
        print("  → Labels are WELL SEPARATED by sentence embeddings")
    elif clustering_results['clustering_accuracy'] > 0.6:
        print("  → Labels are MODERATELY SEPARATED by sentence embeddings")
    else:
        print("  → Labels are POORLY SEPARATED by sentence embeddings")
    print("=" * 80)


if __name__ == "__main__":
    main()
