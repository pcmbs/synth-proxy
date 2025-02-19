"""
Wrapper class around a torchopenl3 model for integration into the current pipeline.

[1] Gyanendra Das, Humair Raj Khan, Joseph Turian (2021). torchopenl3 (version 1.0.1). 
DOI 10.5281/zenodo.5168808, https://github.com/torchopenl3/torchopenl3.
"""

import os
from functools import partial

import torch
import torchopenl3
from dotenv import load_dotenv

from models.audio.abstract_audio_model import AudioModel
from utils import reduce_fn

load_dotenv()  # take environment variables from .env for checkpoints folder
torch.hub.set_dir(os.environ["PROJECT_ROOT"])  # path to download/load checkpoints


class TorchOpenL3Wrapper(AudioModel):
    """
    Wrapper class around a torchopenl3 model for integration into the current pipeline.

    [1] Gyanendra Das, Humair Raj Khan, Joseph Turian (2021). torchopenl3 (version 1.0.1).
    DOI 10.5281/zenodo.5168808, https://github.com/torchopenl3/torchopenl3.
    """

    def __init__(
        self,
        input_repr: str,
        content_type: str,
        embedding_size: int,
        hop_size: float,
        center: bool,
        reduction: str,
    ) -> None:
        """
        Args:
            input_repr (str): Input representation. Should be one of ['linear', 'mel128', 'mel256'].
            content_type (str): Content type. Shopuld be one of ['music', 'env'].
            embedding_size (int): Embedding size. Should be one of [512, 6144].
            hop_size (float): Hop size in seconds.
            center (bool): If True, pads beginning of signal so timestamps correspond
            to center of window.
            reduction (str): the method used to reduce the model outputs to a single vector per
            audio input. Available reduction methods are available in utils/reduce_fn.py
        """
        super().__init__()
        self.input_repr = input_repr
        self.content_type = content_type
        self.embedding_size = embedding_size
        self.hop_size = hop_size
        self.center = center

        # load the model
        self.net = torchopenl3.core.load_audio_embedding_model(
            input_repr=self.input_repr,
            content_type=self.content_type,
            embedding_size=self.embedding_size,
        )

        # partial init such that only the audio must be passed in the forward function
        self.encoder = partial(
            torchopenl3.core.get_audio_embedding,
            model=self.net,
            hop_size=self.hop_size,
            center=self.center,
        )

        self.reduce_fn = getattr(reduce_fn, reduction)
        if reduction == "identity":
            raise NotImplementedError("Non-reduced outputs are not supported yet.")

    @property
    def sample_rate(self) -> int:
        """Return the required input audio signal's sample rate."""
        return 48_000

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return f"openl3_{self.input_repr}_{self.content_type}_{self.embedding_size}"

    @property
    def in_channels(self) -> int:
        """Return the required number of input audio channels."""
        return 1

    @property
    def out_features(self) -> int:
        """Return the number of output features of the model (assuming `{avg,max}_time_pool`)."""
        return self.embedding_size

    @property
    def includes_mel(self) -> bool:
        """Return whether the model concatenate the reduced mel spectrogram to its output."""
        return False

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio (torch.Tensor): mono input sounds @48khz of shape (n_sounds, n_samples, n_channels=1) in the range [-1, 1]

        Returns:
            torch.Tensor: audio embeddings of shape:
            - (n_sounds, embed_size) if `{max,avg}_time_pool` is used as reduction function,
            where embed_size=self.out_features
            - (n_sounds, n_timestamps) if `{max,avg}_channel_pool` is used as reduction function,
            where n_timestamps depends on the input length and is computed based on a window size
            of 1sec with the chosen hop size.
            - (n_sounds, embed_size * n_timestamps) if `flatten` is used as reduction function.
            - (n_sounds, embed_size, n_timestamps) if `identity` is used as reduction function.
        """
        # audio of shape (batch, time, channels) required
        audio = audio.swapdims(-1, -2)
        embeddings, _ = self.encoder(audio=audio, sr=self.sample_rate)
        embeddings.swapdims_(-1, -2)  # swap time and channel dims
        return self.reduce_fn(embeddings)


def openl3_mel256_music_6144_h05_center_max_time_pool():
    return TorchOpenL3Wrapper(
        input_repr="mel256",
        content_type="music",
        embedding_size=6144,
        hop_size=0.5,
        center=True,
        reduction="max_time_pool",
    )


# best resulting openl3 model (flatened omitted) from the sound attributes ranking evaluation
openl3_mel256_music_6144 = openl3_mel256_music_6144_h05_center_max_time_pool  # sound attr. eval mean: 0.940

# input_repr: str = "mel256",
# content_type: str = "music",
# embedding_size: int = 6144,
# hop_size: float = 0.5,
# center: bool = True,

# audio requirements:
# - sr= 48_000
# - num_channels = 1

# remarks:
# - has a frame length of 1 seconds (set hop-size to 1.0 to avoid redundancy and not skip samples)
# - the center argument allows to set the center of the first window to the beginning of the
#   signal (“zero centered”), and the returned timestamps correspond to the center of each window

##### feature maps sizes:
# x: torch.Size([4, 1, 256, 199])
# conv2d_1: torch.Size([4, 64, 256, 199])
# conv2d_2: torch.Size([4, 64, 256, 199])
# max_pooling2d_1: torch.Size([4, 64, 128, 99])
# conv2d_3: torch.Size([4, 128, 128, 99])
# conv2d_4: torch.Size([4, 128, 128, 99])
# max_pooling2d_2: torch.Size([4, 128, 64, 49])
# conv2d_5: torch.Size([4, 256, 64, 49])
# conv2d_6: torch.Size([4, 256, 64, 49])
# max_pooling2d_3: torch.Size([4, 256, 32, 24])
# conv2d_7: torch.Size([4, 512, 32, 24])
# audio_embedding_layer: torch.Size([4, 512, 32, 24])
# max_pooling2d_4: torch.Size([4, 512, 4, 3])
# flatten: torch.Size([4, 6144])

# stem: input-> BN
# block (x3): conv2D->BN->RelU->conv2D->BN->RelU->maxpool2D
# conv2D->BN->RelU->conv2D->maxpool2D->flatten
