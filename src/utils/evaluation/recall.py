"""
Module for computing recall metrics between audio and preset embeddings.

This module provides functions to compute cross-modal retrieval metrics:
- Recall@k: Proportion of queries for which the correct item is in the top-k retrieved items
- Mean Normalized Rank (MNR): Average normalized rank (rank/total_items)
- Median Normalized Rank (MdNR): Median normalized rank

Supports both directions:
- A->P: Audio to Preset retrieval (query audio, retrieve presets)
- P->A: Preset to Audio retrieval (query preset, retrieve audio)
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np

log = logging.getLogger(__name__)


def compute_recall_at_k(
    ranks: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute recall@k for given ranks.

    Args:
        ranks (torch.Tensor): Tensor of ranks (0-indexed) for each query
        k_values (List[int]): List of k values to compute recall for

    Returns:
        Dict[str, float]: Dictionary with recall@k values
    """
    recall_results = {}

    for k in k_values:
        # Recall@k is the proportion of queries where rank < k
        recall_k = (ranks < k).float().mean().item()
        recall_results[f"recall@{k}"] = recall_k

    return recall_results


def compute_normalized_ranks(
    ranks: torch.Tensor,
    total_items: int
) -> Dict[str, float]:
    """
    Compute normalized rank statistics.

    Args:
        ranks (torch.Tensor): Tensor of ranks (0-indexed) for each query
        total_items (int): Total number of items in the collection

    Returns:
        Dict[str, float]: Dictionary with MNR and MdNR values
    """
    # Normalize ranks to [0, 1] range
    normalized_ranks = (ranks.float() + 1) / total_items

    mnr = normalized_ranks.mean().item()
    mdnr = normalized_ranks.median().item()

    return {
        "mean_normalized_rank": mnr,
        "median_normalized_rank": mdnr
    }


def compute_distance_matrix(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
    distance_metric: str = "cosine"
) -> torch.Tensor:
    """
    Compute distance matrix between query and key embeddings.

    Args:
        query_embeddings (torch.Tensor): Query embeddings of shape (n_queries, dim)
        key_embeddings (torch.Tensor): Key embeddings of shape (n_keys, dim)
        distance_metric (str): Distance metric to use ("cosine", "euclidean", "l1")

    Returns:
        torch.Tensor: Distance matrix of shape (n_queries, n_keys)
    """
    if distance_metric == "cosine":
        # Cosine distance = 1 - cosine similarity
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        key_norm = F.normalize(key_embeddings, p=2, dim=1)
        similarities = torch.mm(query_norm, key_norm.T)
        distances = 1 - similarities

    elif distance_metric == "euclidean":
        # Euclidean distance
        distances = torch.cdist(query_embeddings, key_embeddings, p=2)

    elif distance_metric == "l1":
        # L1 (Manhattan) distance
        distances = torch.cdist(query_embeddings, key_embeddings, p=1)

    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    return distances


def compute_ranks_from_distances(
    distance_matrix: torch.Tensor,
    query_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute ranks from distance matrix.

    Args:
        distance_matrix (torch.Tensor): Distance matrix of shape (n_queries, n_keys)
        query_indices (Optional[torch.Tensor]): Indices of correct keys for each query.
                                               If None, assumes diagonal matching (query i -> key i)

    Returns:
        torch.Tensor: Ranks (0-indexed) for each query
    """
    # Sort distances to get rankings
    sorted_indices = distance_matrix.argsort(dim=1)

    # If no query_indices provided, assume diagonal matching
    if query_indices is None:
        query_indices = torch.arange(distance_matrix.shape[0])

    # Find the rank of the correct item for each query
    ranks = torch.zeros(distance_matrix.shape[0], dtype=torch.long)

    for i, correct_idx in enumerate(query_indices):
        # Find where the correct index appears in the sorted list
        rank = torch.where(sorted_indices[i] == correct_idx)[0][0]
        ranks[i] = rank

    return ranks


def compute_audio_to_preset_recall(
    audio_embeddings: torch.Tensor,
    preset_embeddings: torch.Tensor,
    distance_metric: str = "cosine",
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute recall metrics for Audio->Preset retrieval.

    Args:
        audio_embeddings (torch.Tensor): Audio embeddings of shape (N, D)
        preset_embeddings (torch.Tensor): Preset embeddings of shape (N, D)
        distance_metric (str): Distance metric to use
        k_values (List[int]): List of k values for recall@k

    Returns:
        Dict[str, float]: Dictionary containing all recall metrics
    """
    assert audio_embeddings.shape[0] == preset_embeddings.shape[0], \
        "Audio and preset embeddings must have same number of samples"

    # Compute distance matrix (audio queries, preset keys)
    distances = compute_distance_matrix(audio_embeddings, preset_embeddings, distance_metric)

    # Compute ranks (assuming paired data: audio[i] should retrieve preset[i])
    ranks = compute_ranks_from_distances(distances)

    # Compute metrics
    recall_metrics = compute_recall_at_k(ranks, k_values)
    norm_rank_metrics = compute_normalized_ranks(ranks, preset_embeddings.shape[0])

    # Combine results with A2P prefix
    results = {}
    for key, value in recall_metrics.items():
        results[f"a2p_{key}"] = value
    for key, value in norm_rank_metrics.items():
        results[f"a2p_{key}"] = value

    return results


def compute_preset_to_audio_recall(
    audio_embeddings: torch.Tensor,
    preset_embeddings: torch.Tensor,
    distance_metric: str = "cosine",
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute recall metrics for Preset->Audio retrieval.

    Args:
        audio_embeddings (torch.Tensor): Audio embeddings of shape (N, D)
        preset_embeddings (torch.Tensor): Preset embeddings of shape (N, D)
        distance_metric (str): Distance metric to use
        k_values (List[int]): List of k values for recall@k

    Returns:
        Dict[str, float]: Dictionary containing all recall metrics
    """
    assert audio_embeddings.shape[0] == preset_embeddings.shape[0], \
        "Audio and preset embeddings must have same number of samples"

    # Compute distance matrix (preset queries, audio keys)
    distances = compute_distance_matrix(preset_embeddings, audio_embeddings, distance_metric)

    # Compute ranks (assuming paired data: preset[i] should retrieve audio[i])
    ranks = compute_ranks_from_distances(distances)

    # Compute metrics
    recall_metrics = compute_recall_at_k(ranks, k_values)
    norm_rank_metrics = compute_normalized_ranks(ranks, audio_embeddings.shape[0])

    # Combine results with P2A prefix
    results = {}
    for key, value in recall_metrics.items():
        results[f"p2a_{key}"] = value
    for key, value in norm_rank_metrics.items():
        results[f"p2a_{key}"] = value

    return results


def compute_bidirectional_recall(
    audio_embeddings: torch.Tensor,
    preset_embeddings: torch.Tensor,
    distance_metric: str = "cosine",
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute recall metrics for both Audio->Preset and Preset->Audio retrieval.

    Args:
        audio_embeddings (torch.Tensor): Audio embeddings of shape (N, D)
        preset_embeddings (torch.Tensor): Preset embeddings of shape (N, D)
        distance_metric (str): Distance metric to use
        k_values (List[int]): List of k values for recall@k

    Returns:
        Dict[str, float]: Dictionary containing all recall metrics for both directions
    """
    # Compute A->P metrics
    a2p_metrics = compute_audio_to_preset_recall(
        audio_embeddings, preset_embeddings, distance_metric, k_values
    )

    # Compute P->A metrics
    p2a_metrics = compute_preset_to_audio_recall(
        audio_embeddings, preset_embeddings, distance_metric, k_values
    )

    # Combine results
    results = {**a2p_metrics, **p2a_metrics}

    # Add average metrics
    for k in k_values:
        avg_recall = (results[f"a2p_recall@{k}"] + results[f"p2a_recall@{k}"]) / 2
        results[f"avg_recall@{k}"] = avg_recall

    avg_mnr = (results["a2p_mean_normalized_rank"] + results["p2a_mean_normalized_rank"]) / 2
    avg_mdnr = (results["a2p_median_normalized_rank"] + results["p2a_median_normalized_rank"]) / 2

    results["avg_mean_normalized_rank"] = avg_mnr
    results["avg_median_normalized_rank"] = avg_mdnr

    return results


def compute_cross_modal_retrieval_metrics(
    audio_embeddings: torch.Tensor,
    preset_embeddings: torch.Tensor,
    distance_metric: str = "cosine",
    k_values: List[int] = [1, 5, 10],
    compute_both_directions: bool = True
) -> Dict[str, float]:
    """
    Comprehensive function to compute cross-modal retrieval metrics.

    Args:
        audio_embeddings (torch.Tensor): Audio embeddings of shape (N, D)
        preset_embeddings (torch.Tensor): Preset embeddings of shape (N, D)
        distance_metric (str): Distance metric to use ("cosine", "euclidean", "l1")
        k_values (List[int]): List of k values for recall@k computation
        compute_both_directions (bool): Whether to compute both A->P and P->A metrics

    Returns:
        Dict[str, float]: Dictionary containing all computed metrics

    Raises:
        ValueError: If embeddings have mismatched dimensions or sizes
    """
    if audio_embeddings.shape != preset_embeddings.shape:
        raise ValueError(
            f"Audio and preset embeddings must have same shape: "
            f"{audio_embeddings.shape} vs {preset_embeddings.shape}"
        )

    if compute_both_directions:
        return compute_bidirectional_recall(
            audio_embeddings, preset_embeddings, distance_metric, k_values
        )
    else:
        # Default to A->P only
        return compute_audio_to_preset_recall(
            audio_embeddings, preset_embeddings, distance_metric, k_values
        )