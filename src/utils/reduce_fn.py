"""
Reduction functions.
"""

import torch


def flatten(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten the input tensor along along the batch axis.

    Args:
        x (torch.Tensor): The input tensor to be flattened.

    Returns:
        the flattened torch.Tensor.
    """
    return x.flatten(start_dim=1)


def avg_channel_pooling(x: torch.Tensor) -> torch.Tensor:
    """
    Calculates the average pooling over the channel dimension of the input tensor.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        Output torch.Tensor.
    """
    return x.mean(-2)


def avg_time_pool(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the average pooling of the input tensor.
    Note that when applied to a ViT-based model,
    this operation is equivalent to the average pooling over the patches.

    Args:
        x (torch.Tensor): The input x tensor.

    Returns:
        Output torch.Tensor.
    """
    return x.mean(-1)


def max_channel_pool(x: torch.Tensor) -> torch.Tensor:
    """
    Calculates the average pooling over the channel dimension of the input tensor.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        Output torch.Tensor.
    """
    return x.amax(-2)


def max_time_pool(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the max pooling of the input tensor.
    Note that when applied to a ViT-based model,
    this operation is equivalent to the max pooling over the patches.

    Args:
        x (torch.Tensor): The input x tensor.

    Returns:
        Output torch.Tensor.
    """
    return x.amax(-1)


if __name__ == "__main__":
    x = torch.rand((10, 128, 500))

    print("original:", x.shape)
    print("flatten:", flatten(x).shape)
    print("global_avg_pool_channel:", avg_channel_pooling(x).shape)
    print("global_avg_pool_time:", avg_time_pool(x).shape)
    print("global_max_pool_channel:", max_channel_pool(x).shape)
    print("global_max_pool_time:", max_time_pool(x).shape)
