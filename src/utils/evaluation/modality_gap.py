"""
Module for computing modality gap metrics between audio embeddings and preset embeddings.

This module provides functions to compute two main modality gap metrics:
1. Centroid distance: Euclidean distance between centroids of audio and preset embeddings
2. Logistic regression accuracy: Binary classification accuracy to distinguish modalities

Adapted from the research on modality gaps in multimodal representations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def compute_centroid_distance(
    audio_embeddings: torch.Tensor,
    preset_embeddings: torch.Tensor,
    normalize: bool = True
) -> float:
    """
    Compute Euclidean distance between centroids of audio and preset embeddings.

    Args:
        audio_embeddings (torch.Tensor): Audio embeddings of shape (N, D)
        preset_embeddings (torch.Tensor): Preset embeddings of shape (N, D)
        normalize (bool): Whether to normalize embeddings to unit length before computing centroids.
                         Default is True for consistency with cosine distance used in recall metrics.

    Returns:
        float: Euclidean distance between centroids

    Raises:
        ValueError: If embedding dimensions don't match
    """
    if audio_embeddings.shape[1] != preset_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimensions must match: {audio_embeddings.shape[1]} vs {preset_embeddings.shape[1]}"
        )

    # Move to CPU if needed
    audio_embeddings = audio_embeddings.cpu()
    preset_embeddings = preset_embeddings.cpu()

    # Normalize to unit length if requested
    if normalize:
        audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
        preset_embeddings = torch.nn.functional.normalize(preset_embeddings, p=2, dim=1)

    audio_centroid = torch.mean(audio_embeddings, dim=0)
    preset_centroid = torch.mean(preset_embeddings, dim=0)

    distance = torch.norm(audio_centroid - preset_centroid, p=2).item()
    return distance


def compute_logistic_regression_accuracy(
    audio_embeddings: torch.Tensor,
    preset_embeddings: torch.Tensor,
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True
) -> float:
    """
    Train logistic regression to distinguish audio vs preset embeddings.

    Args:
        audio_embeddings (torch.Tensor): Audio embeddings of shape (N, D)
        preset_embeddings (torch.Tensor): Preset embeddings of shape (N, D)
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random state for reproducibility
        normalize (bool): Whether to normalize embeddings to unit length before classification.
                         Default is True to focus on directional differences rather than magnitude.

    Returns:
        float: Classification accuracy on test set

    Raises:
        ValueError: If embedding dimensions don't match
    """
    if audio_embeddings.shape[1] != preset_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimensions must match: {audio_embeddings.shape[1]} vs {preset_embeddings.shape[1]}"
        )

    # Move to CPU if needed
    audio_embeddings = audio_embeddings.cpu()
    preset_embeddings = preset_embeddings.cpu()

    # Normalize to unit length if requested
    if normalize:
        audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
        preset_embeddings = torch.nn.functional.normalize(preset_embeddings, p=2, dim=1)

    # Convert to numpy
    audio_np = audio_embeddings.numpy()
    preset_np = preset_embeddings.numpy()

    # Create labels (0 for audio, 1 for preset)
    audio_labels = np.zeros(audio_np.shape[0])
    preset_labels = np.ones(preset_np.shape[0])

    # Combine data
    X = np.vstack([audio_np, preset_np])
    y = np.concatenate([audio_labels, preset_labels])

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train logistic regression
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict and compute accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def compute_modality_gap_metrics(
    audio_embeddings: torch.Tensor,
    preset_embeddings: torch.Tensor,
    compute_centroid: bool = True,
    compute_lr_accuracy: bool = True,
    normalize: bool = True,
    **lr_kwargs
) -> Dict[str, Optional[float]]:
    """
    Compute modality gap metrics between audio and preset embeddings.

    Args:
        audio_embeddings (torch.Tensor): Audio embeddings of shape (N, D)
        preset_embeddings (torch.Tensor): Preset embeddings of shape (N, D)
        compute_centroid (bool): Whether to compute centroid distance
        compute_lr_accuracy (bool): Whether to compute LR accuracy
        normalize (bool): Whether to normalize embeddings for centroid distance computation
        **lr_kwargs: Additional arguments for logistic regression

    Returns:
        Dict[str, Optional[float]]: Dictionary containing computed metrics
            - "centroid_distance": Euclidean distance between centroids (or None)
            - "logistic_regression_accuracy": LR classification accuracy (or None)
    """
    results = {}

    # Check if dimensions match
    dimension_mismatch = audio_embeddings.shape[1] != preset_embeddings.shape[1]

    # Compute centroid distance
    if compute_centroid:
        if dimension_mismatch:
            results["centroid_distance"] = None
        else:
            results["centroid_distance"] = compute_centroid_distance(
                audio_embeddings, preset_embeddings, normalize=normalize
            )

    # Compute logistic regression accuracy
    if compute_lr_accuracy:
        if dimension_mismatch:
            results["logistic_regression_accuracy"] = None
        else:
            results["logistic_regression_accuracy"] = compute_logistic_regression_accuracy(
                audio_embeddings, preset_embeddings, normalize=normalize, **lr_kwargs
            )

    return results


def load_embeddings_from_dataset(dataset_path: Path) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Load audio embeddings and preset embeddings from a dataset directory.

    Args:
        dataset_path (Path): Path to dataset directory

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Audio embeddings and dict of preset embeddings

    Raises:
        FileNotFoundError: If no audio embeddings found
    """
    # Try to load audio embeddings (check different naming patterns)
    audio_embeddings = None
    split_used = None

    # First try the standard naming
    if (dataset_path / "audio_embeddings.pkl").exists():
        audio_embeddings = torch.load(
            dataset_path / "audio_embeddings.pkl",
            map_location="cpu",
            weights_only=False
        )
        split_used = ""
    # If not found, try test split
    elif (dataset_path / "audio_embeddings_test.pkl").exists():
        audio_embeddings = torch.load(
            dataset_path / "audio_embeddings_test.pkl",
            map_location="cpu",
            weights_only=False
        )
        split_used = "_test"
    # If not found, try train split
    elif (dataset_path / "audio_embeddings_train.pkl").exists():
        audio_embeddings = torch.load(
            dataset_path / "audio_embeddings_train.pkl",
            map_location="cpu",
            weights_only=False
        )
        split_used = "_train"
    else:
        raise FileNotFoundError(f"No audio embeddings found in {dataset_path}")

    # Load preset embeddings
    preset_embeddings = {}
    preset_embeddings_dir = dataset_path / "preset_embeddings"

    if preset_embeddings_dir.exists():
        for pkl_file in preset_embeddings_dir.glob("*.pkl"):
            model_name = pkl_file.stem.replace("_embeddings", "")
            embeddings = torch.load(pkl_file, map_location="cpu", weights_only=False)
            preset_embeddings[model_name] = embeddings

    # If no preset embeddings found, try to load raw synth parameters
    if not preset_embeddings:
        # Try different naming patterns for synth parameters
        synth_params_path = None
        if (dataset_path / f"synth_parameters{split_used}.pkl").exists():
            synth_params_path = dataset_path / f"synth_parameters{split_used}.pkl"
        elif (dataset_path / "synth_parameters.pkl").exists():
            synth_params_path = dataset_path / "synth_parameters.pkl"

        if synth_params_path:
            synth_params = torch.load(synth_params_path, map_location="cpu", weights_only=False)
            preset_embeddings["raw_parameters"] = synth_params

    return audio_embeddings, preset_embeddings