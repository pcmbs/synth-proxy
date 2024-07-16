"""
Abstract base class for all audio models.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class AudioModel(nn.Module, ABC):
    """
    Abstract base class for all audio models.
    """

    @abstractmethod
    def __init__(self):
        """
        This CTOR and input arguments are responsible for:
        - initializing the model
        - loading the checkpoint
        - setting a reduction function used to reduce the model outputs to a single vector per
        audio input. It is recommended to import `from utils import reduce_fn`, and later use getattr
        to get the reduction function, i.e., `self.reduce_fn = getattr(reduce_fn, reduction)`
        """
        super().__init__()

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """This method should return the required input audio signal's sample rate."""

    @property
    @abstractmethod
    def name(self) -> str:
        """This method should return the name of the model."""

    @property
    @abstractmethod
    def in_channels(self) -> int:
        """This method should return the number of input audio channels."""

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters of the model."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    @abstractmethod
    def out_features(self) -> int:
        """
        This method should return the number of output features of the model
        assuming a reduction function is used.
        """

    @property
    @abstractmethod
    def includes_mel(self) -> bool:
        """
        This method should return whether the model concatenate the reduced
        mel spectrogram to its output.
        """

    @abstractmethod
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the audio model's forward pass. The last statement would usually (but must not)
        be a call to self.reduce_fn in order to reduce the number of output dimension.

        Args:
            audio (torch.Tensor): Input audio signals of shape (n_sounds, n_channels, n_samples)
            in the range [-1,1], which should be resampled to `self.sample_rate` if necessary.

        Returns:
            torch.Tensor: audio embedding of shape (n_sounds, *) where * depends on the reduction function.
        """
