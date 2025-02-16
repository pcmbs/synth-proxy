"""
Script to build custom schedulers
"""

from copy import deepcopy
import math
from pathlib import Path
from typing import Optional, Union
import torch
import torch.optim.lr_scheduler as S


def reduce_on_plateau(
    optimizer: torch.optim.Optimizer, mode: str, factor: float, patience: int, **kwargs
) -> S._LRScheduler:
    """
    Wrapper for ReduceLROnPlateau scheduler.
    """
    return S.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, **kwargs)


def lin_cos_scheduler_builder(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    milestone: int,
    final_lr: float,
    linear_start_factor: float = 1,
    linear_end_factor: float = 1,
) -> S._LRScheduler:
    """
    Builds a cosine LR scheduler preceded by a linear LR scheduler,
    which can either act as a linear warmup or to keep the learning rate constant allowing for a longer
    exploration phase.

    Args:
        optimizer (torch.optim.Optimizer): the optimizer for which the LR scheduler is built
        total_steps (int): the total number of training steps.
        milestone (int): the number of steps after which the cosine LR scheduler is triggered.
        final_lr (float): the final learning rate of the cosine LR scheduler.
        linear_start_factor (float): the starting factor of the linear LR scheduler, to be multiplied with
        the initial LR. (Default: 1)
        linear_end_factor (float): the ending factor of the linear LR scheduler, to be multiplied with
        the initial LR. (Default: 1)

    Returns:
        A torch.optim.lr_scheduler._LRScheduler LR scheduler with the linear LR scheduler as
        a torch sequential LR scheduler.
    """
    # Build the linear LR scheduler
    linear_scheduler = S.LinearLR(
        optimizer, start_factor=linear_start_factor, end_factor=linear_end_factor, total_iters=milestone
    )
    # Build the cosine LR scheduler
    cosine_scheduler = S.CosineAnnealingLR(optimizer, T_max=total_steps - milestone, eta_min=final_lr)

    # # Combine both schedulers
    scheduler = S.SequentialLR(
        optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[milestone]
    )
    return scheduler


def wcrc_scheduler_builder(
    optimizer: torch.optim.Optimizer,
    min_lr: float,
    num_warmup_steps: Optional[int] = None,
    warmup_factor: float = 1.0,
    num_decay_steps: int = 200_000,
    num_restart_steps: Optional[int] = None,
    restart_factor: float = 0.1,
) -> S._LRScheduler:
    """
    A function to build a learning rate scheduler which sequentially applies
    - 1) a linear Warmup (optional),
    - 2) a Cosine decay,
    - 3) a Restart (optional),
    - 4) a Constant final learning rate.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to build the scheduler.
        min_lr (float): The minimum learning rate for the scheduler.
        num_warmup_steps (Optional[int]): The number of warmup steps. (Defaults: None).
        warmup_factor (float): The warmup factor which multiplies the initial learning rate. (Defaults: 1.0).
        num_decay_steps (int): The number of (cosine) decay steps. (Defaults: 200_000).
        num_restart_steps (Optional[int]): Mumber of steps for the second cosine decay. (Defaults: None).
        restart_factor (float): The restart factor which multiplies the initial learning rate. (Defaults: 1.0).

    Returns:
        A S._LRScheduler LR scheduler.
    """
    assert num_warmup_steps is None or num_warmup_steps >= 0
    assert num_restart_steps is None or num_restart_steps >= 0

    initial_lr = optimizer.param_groups[0]["lr"]
    min_factor = min_lr / initial_lr
    milestones = []

    # warmup (optional)
    num_warmup_steps = num_warmup_steps or 0
    warmup = S.LinearLR(optimizer, start_factor=warmup_factor, end_factor=1, total_iters=num_warmup_steps)
    milestones.append(num_warmup_steps)

    # cosine
    cosine = S.CosineAnnealingLR(optimizer, T_max=num_decay_steps, eta_min=min_lr)
    milestones.append(num_warmup_steps + num_decay_steps)

    # restart (optional)
    if num_restart_steps == 0 or num_restart_steps is None:
        # if no restart is needed we simply set the LR to min_LR for one step and move on
        num_restart_steps = 1
        restart_factor = min_factor

    restart = S.LambdaLR(
        optimizer,
        lambda epoch: restart_factor
        + 0.5 * (min_factor - restart_factor) * (1 - math.cos(math.pi * epoch / num_restart_steps)),
    )
    milestones.append(milestones[-1] + num_restart_steps)

    # final constant LR (1e12 steps should be long enough)
    final = S.ConstantLR(optimizer=optimizer, factor=min_factor, total_iters=1e12)

    # Return a sequence of all schedulers
    return S.SequentialLR(optimizer, schedulers=[warmup, cosine, restart, final], milestones=milestones)


def add_wcrc_scheduler_to_ckpt(
    ckpt_path: Union[Path, str], num_restart_steps: int, filename: str = "last_wcrc"
) -> None:
    """
    Hack function to add the Restart and Constant part of the WCRC_scheduler
    to a lin_cos_scheduler from a Lightning checkpoint.

    Args:
        ckpt_path (Union[Path, str]): path to the checkpoint file
        num_restart_steps (int): number of steps for the second cosine decay.
        filename (str): filename of the new checkpoint file
    """

    ckpt_path = Path(ckpt_path)
    with open(ckpt_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu")

    ckpt = _add_wcrc_scheduler_to_ckpt(ckpt, num_restart_steps)

    with open(ckpt_path.parent / f"{filename}.ckpt", "wb") as f:
        torch.save(ckpt, f)


def _add_wcrc_scheduler_to_ckpt(ckpt: dict, num_restart_steps: int) -> dict:
    if ckpt.get("lr_schedulers") is None:
        raise KeyError("No lr_schedulers in checkpoint")

    ckpt = deepcopy(ckpt)

    # append two last milestones (cosine -> restart -> final constant)
    # assuming that the restart begins as soon as we resume training
    global_step = ckpt["global_step"]
    ckpt["lr_schedulers"][0]["_milestones"].extend([global_step, global_step + num_restart_steps])

    # get the base lr and min lr from the cosine scheduler
    # to add them to the restart and constant schedulers
    base_lr = ckpt["lr_schedulers"][0]["_schedulers"][1]["base_lrs"][0]
    min_lr = ckpt["lr_schedulers"][0]["_schedulers"][1]["eta_min"]

    # append two extra schedulers
    extra_schedulers = [
        {
            "base_lrs": [base_lr],
            "last_epoch": -1,
            "verbose": False,
            "_step_count": 1,
            "_get_lr_called_within_step": False,
            "_last_lr": [min_lr],
            "lr_lambdas": [None],
        },
        {
            "factor": min_lr / base_lr,  # we want constant to min LR
            "total_iters": 1e12,
            "base_lrs": [base_lr],
            "last_epoch": -1,
            "verbose": False,
            "_step_count": 1,
            "_get_lr_called_within_step": False,
            "_last_lr": [min_lr],
        },
    ]

    ckpt["lr_schedulers"][0]["_schedulers"].extend(extra_schedulers)

    return ckpt


if __name__ == "__main__":
    pass
