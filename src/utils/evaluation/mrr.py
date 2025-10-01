# pylint: disable=W1203:logging-fstring-interpolation
"""
Module for the computing the Mean Reciprocal Rank (MRR) metric used for validation and evaluation.
"""
import math
import logging
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
import torch.nn.functional as F

from .recall import compute_bidirectional_recall
from .modality_gap import compute_modality_gap_metrics

log = logging.getLogger(__name__)


def compute_mrr(preds: torch.Tensor, targets: torch.Tensor, index: int = 0, p: int = 1) -> float:
    """
    Function computing the Mean Reciprocal Rank (MRR) metric using `torch.cdist`.

    Args:
        preds (torch.Tensor): predictions of shape (num_eval, num_preds_per_eval, preds_dim),
        where num_eval corresponds to the number reciprocal ranks to be computed, num_preds_per_eval corresponds
        to the number of predictions per evaluation, and preds_dim is the dimension of the predictions
        targets (torch.Tensor): target of shape (num_eval, 1, preds_dim). Note that targets.shape[1] should
        be 1 since there is a single target per evaluation.
        index (int): index of the prediction to be considered as the target in each evaluation. (default: 0)
        p (int): p value for the p-norm used to compute the distances between each prediction and the target
        in each evaluation. (default: 1)

    Returns:
        MRR score as float between 0 and 1
    """
    ranks, _ = _compute_ranks(preds, targets, index=index, p=p)
    return (1 / (ranks + 1)).mean().item()


def _compute_ranks(
    preds: torch.Tensor, targets: torch.Tensor, index: int = 0, p: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert preds.shape[0] == targets.shape[0]
    assert preds.shape[2] == targets.shape[2]
    assert targets.shape[1] == 1
    assert preds.shape[1] > 1, "There should be more than 1 prediction per evaluation"
    assert (
        index < preds.shape[1]
    ), "Prediction index should be less than the number of predictions per evaluation"

    # For each evaluation: compute the distances between each prediction and the target,
    # then sort the distances in ascending order and get the rank of the prediction matching the target
    distances = torch.cdist(preds, targets, p=p).squeeze()  # shape: (num_eval, num_preds_per_eval)
    ranks = distances.argsort(dim=-1)
    ranks = torch.where(ranks == index)[1]
    return ranks, distances


def one_vs_all_eval(
    model: nn.Module,
    dataset: Union[Dataset, Subset],
    subset_size: int = -1,
    seed: int = 0,
    p: int = 1,
    batch_size: int = 512,
    device: str = "cpu",
    log_results: bool = True,
    distance_metric: str = "cosine",
    k_values: List[int] = [1, 5, 10],
) -> Tuple[float, List, Dict, float, Dict, Dict]:
    """
    Function computing the Mean Reciprocal Rank (MRR) of each samples in the dataset against all other
    samples. It additionally computes the average L1 loss, bidirectional recall metrics, and modality gap metrics.

    Args:
        model (nn.Module): model to be evaluated
        dataset (Union[Dataset, Subset]): dataset used for evaluation.
        subset_size (int): number of (shuffled) samples to be taken from the dataset. If -1 or 0 are
        provided or if subset_size > len(dataset), all samples are taken. (Default: -1)
        seed (int): seed for the RNG. (default: 0)
        p (int): p value for the p-norm used to compute the distances between each prediction and
        the target in each evaluation. (default: 1)
        batch_size (int): number of samples to to be processed at once to avoid memory issues for
        large datasets. This includes: computing the preset embeddings, computing the pairwise distances,
        and retrieving the rank of the prediciton matching the target. (default: 512)
        device (str): device on which the model will be evaluated. (default: "cpu")
        log_results (bool): whether or not to log the results. (default: True)
        distance_metric (str): distance metric for recall computation ("cosine", "euclidean", "l1"). (default: "cosine")
        k_values (List[int]): list of k values for recall@k computation. (default: [1, 5, 10])

    Returns:
        A tuple containing:
        - MRR score as float between 0 and 1
        - a list containig the top-k MRR for k=1, 2, ..., 5
        - a dictionary {rank: [preset_ids]} where each key is a rank and each value is a list of
        preset ids that ended up with that rank
        - L1 loss as float
        - recall metrics as dictionary containing bidirectional recall metrics
        - modality gap metrics as dictionary containing centroid distance and LR accuracy
    """
    if 0 < subset_size < len(dataset):
        subset_idx = _get_rnd_non_repeating_integers(N=len(dataset), K=subset_size, seed=seed)
        dataset = Subset(dataset, subset_idx)

    rng_loader = torch.Generator(device="cpu")
    rng_loader.manual_seed(seed * 100)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False, generator=rng_loader
    )

    # compute preset embeddings and retrieve audio embeddings
    preset_embeddings = []
    audio_embeddings = []
    losses = []
    model.eval()
    with torch.no_grad():
        for i, (preset, audio_emb) in enumerate(dataloader):
            batch_preset_embedding = model(preset.to(device))
            batch_audio_embedding = audio_emb.to(device)
            loss = F.l1_loss(batch_preset_embedding, batch_audio_embedding).item()

            preset_embeddings.append(batch_preset_embedding)
            audio_embeddings.append(batch_audio_embedding)
            losses.append(loss)

    audio_embeddings = torch.cat(audio_embeddings, dim=0)
    preset_embeddings = torch.cat(preset_embeddings, dim=0)

    # iteratively compute a slice of the distance matrix and get the according ranks
    # (distance matrix's diagonal entries) which correspond to the rank of the prediction
    # matching the target
    ranks = torch.empty(len(dataset), device=device)
    for i in range(math.ceil(len(dataset) / batch_size)):
        slice_ = (
            i * batch_size,
            (i + 1) * batch_size if (i + 1) * batch_size < len(dataset) else len(dataset),
        )
        batch_distances = torch.cdist(preset_embeddings[slice_[0] : slice_[1]], audio_embeddings, p=p)
        batch_ranks = batch_distances.argsort(dim=-1)
        matching_preset_id = torch.arange(slice_[0], slice_[1], device=device)
        ranks[slice_[0] : slice_[1]] = (batch_ranks == matching_preset_id.view(-1, 1)).nonzero()[:, 1]

    # compute the log the MRR
    mrr_score = (1 / (ranks + 1)).mean().item()
    if log_results:
        log.info(f"MRR score: {mrr_score:.4f}")

    # Create a dictionary in which the keys are ranks and the values are lists of
    # preset ids that ended up with that rank
    ranks_dict = _create_rank_dict(ranks)

    # compute and log the top-k MRR
    top_k_mrr = [top_k_mrr_score(ranks_dict, k=k) for k in range(1, 6)]
    if log_results:
        for k, mrr in enumerate(top_k_mrr):
            log.info(f"Top-{k+1} MRR score: {mrr:.4f}")

    # compute the L1 loss
    l1_loss = sum(losses) / len(losses)  # compute_l1(model, dataset, batch_size=batch_size)
    if log_results:
        log.info(f"L1 loss: {l1_loss:.4f}")  # pylint: disable=W1203

    # compute bidirectional recall metrics
    recall_metrics = compute_bidirectional_recall(
        audio_embeddings, preset_embeddings, distance_metric=distance_metric, k_values=k_values
    )
    if log_results:
        for k in k_values:
            log.info(f"A->P Recall@{k}: {recall_metrics[f'a2p_recall@{k}']:.4f}")
        log.info(f"A->P MNR: {recall_metrics['a2p_mean_normalized_rank']:.4f}")
        for k in k_values:
            log.info(f"P->A Recall@{k}: {recall_metrics[f'p2a_recall@{k}']:.4f}")
        log.info(f"P->A MNR: {recall_metrics['p2a_mean_normalized_rank']:.4f}")

    # compute modality gap metrics
    modality_gap_metrics = compute_modality_gap_metrics(audio_embeddings, preset_embeddings)
    if log_results:
        log.info(f"Centroid Distance: {modality_gap_metrics['centroid_distance']:.4f}")
        log.info(f"Logistic Regression Accuracy: {modality_gap_metrics['logistic_regression_accuracy']:.4f}")

    return mrr_score, top_k_mrr, ranks_dict, l1_loss, recall_metrics, modality_gap_metrics


def _create_rank_dict(ranks: torch.Tensor) -> Dict:
    """
    Create a dictionary in which the keys are ranks and the values are lists of
    preset ids that ended up with that rank.
    """
    ranks_dict = {}
    for i, rank in enumerate(ranks):
        rank = rank.item() + 1  # use 1-based ranks for analysis
        if rank not in ranks_dict:
            ranks_dict[rank] = [i]
        else:
            ranks_dict[rank].append(i)
    # sort dict key by ascending rank
    ranks_dict = dict(sorted(ranks_dict.items()))
    return ranks_dict


def non_overlapping_eval(
    model: nn.Module,
    dataset: Dataset,
    subset_size: int = -1,
    num_ranks: int = 256,
    seed: int = 0,
    p: int = 1,
    device: str = "cpu",
    log_results: bool = True,
) -> Tuple[float, float, List, Dict]:
    """
    Function computing the Mean Reciprocal Rank (MRR) metric over a given dataset.
    Selected samples are used only once (i.e., only in a given evaluation, either as a target or canditate).
    The number of samples per evaluation is given by dataset_size // num_ranks.
    E.g., num_ranks = 256 and dataset_size = 131072, corresponds to 256 ranking
    evaluations, each containing 1 target amongst 512 candidates.
    It additionally compute the average L1 loss of each preset and audio embeddings.

    Args:
        model (nn.Module): model to be evaluated
        dataset (Dataset): dataset used for evaluation. Note that the first batch taken from this
        dataset will be used as targets in each evaluation
        subset_size (int): number of (shuffled) samples to be taken from the dataset. If -1 or 0 are
        provided or if subset_size > len(dataset), all samples are taken. (Default: -1)
        num_evals (int): number of ranking evaluations to compute the mean from. (default: 256)
        seed (int): seed used to shuffle the dataset. (default: 0)
        p (int): p value for the p-norm used to compute the distances between each prediction and
        the target in each evaluation. (Default: 1)
        device (str): device to be used for evaluation. (Default: "cpu")
        log_results (bool): whether or not to log the results. (Default: True)

    Returns:
        A tuple containing:
        - MRR score as float between 0 and 1
        - a list containig the top-k MRR for k=1, 2, ..., 5
        - a dictionary {rank: [preset_ids]} where each key is a rank and each
        value is a list of preset ids that ended up with that rank
        - L1 loss as float
    """
    if 0 < subset_size < len(dataset):
        subset_idx = _get_rnd_non_repeating_integers(N=len(dataset), K=subset_size, seed=seed)
        dataset = Subset(dataset, subset_idx)

    rng_loader = torch.Generator(device="cpu")
    rng_loader.manual_seed(seed)
    dataloader = DataLoader(
        dataset, batch_size=num_ranks, shuffle=False, num_workers=0, drop_last=True, generator=rng_loader
    )

    mrr_preds = []
    mrr_targets = None
    losses = []

    # compute preset embeddings and retrieve audio embeddings
    model.eval()
    with torch.no_grad():
        for i, (preset, audio_emb) in enumerate(dataloader):
            audio_embeddings = audio_emb.to(device)
            preset_embeddings = model(preset.to(device))
            loss = F.l1_loss(preset_embeddings, audio_embeddings).item()
            losses.append(loss)
            if i == 0:
                mrr_targets = audio_embeddings
            mrr_preds.append(preset_embeddings)

    # Compute MRR, top-K MRR, and generate rank dictionary
    num_eval, preds_dim = mrr_targets.shape
    # unsqueeze for torch.cdist (one target per eval) -> shape: (num_eval, 1, dim)
    targets = mrr_targets.unsqueeze_(1)
    # concatenate and reshape for torch.cdist-> shape (num_eval, num_preds_per_eval, dim)
    preds = torch.cat(mrr_preds, dim=1).view(num_eval, -1, preds_dim)
    ranks, _ = _compute_ranks(preds, targets, index=0, p=p)
    mrr_score = (1 / (ranks + 1)).mean().item()
    if log_results:
        log.info(f"MRR score: {mrr_score:.4f}")

    ranks_dict = _create_rank_dict(ranks)

    top_k_mrr = [top_k_mrr_score(ranks_dict, k=k) for k in range(1, 6)]
    if log_results:
        for k, mrr in enumerate(top_k_mrr):
            log.info(f"Top-{k+1} MRR score: {mrr:.4f}")

    # Compute L1 loss
    loss = sum(losses) / len(losses)
    if log_results:
        log.info(f"L1 loss: {loss:.4f}")

    return mrr_score, top_k_mrr, ranks_dict, loss


def top_k_mrr_score(ranks: Dict, k: int = 5) -> float:
    """
    Calculate the top-k Mean Reciprocal Rank (MRR) score for the given ranks.
    If a target has no matches in the top-k ranks, it will be assigned a score of 0.

    Args:
        ranks (Dict): A dictionary containing ranks for different targets.
        k (int): An integer representing the top 'k' ranks to consider (Default: 5).

    Returns:
        A float representing the mean reciprocal rank score.
    """
    # get the total number of targets, i.e., number of ranking evaluation
    num_targets = sum(len(val) for val in ranks.values())
    # compute the reciprocal rank at k:
    # 1) get the number of relevant items that ended up with rank i and
    #    divide them by the rank i for each rank i \in {1,k}
    # 2) sum over all ranks i \in {1,k}
    recriprocal_rank_at_k = sum(len(ranks.get(rank, [])) / rank for rank in range(1, k + 1))
    # divide the reciprocal rank at k by the number of targets.
    # This has the same effect as assigning a score of 0 for relevant items that did not end up in the top-k ranks
    return recriprocal_rank_at_k / num_targets


def _get_rnd_non_repeating_integers(N: int, K: int, seed: int = 0) -> List[float]:
    """
    Return a list of K random non-repeating integers between 0 and N.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    assert K <= N
    arr = torch.arange(0, N)

    # Shuffle the array
    arr_shuffled = arr[torch.randperm(N, generator=rng)]

    # Take only the first K entries
    result = arr_shuffled[:K]

    return result.tolist()
