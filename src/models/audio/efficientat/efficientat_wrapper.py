"""
Wrapper class around EfficientAT [1] models for integration into the current pipeline.

stripped down version of the EfficientAT repo available at
https://github.com/fschmid56/EfficientAT_HEAR/tree/main 

[1] Schmid, F., Koutini, K., & Widmer, G. (2022). 
Efficient Large-scale Audio Tagging via Transformer-to-CNN Knowledge Distillation.
http://arxiv.org/abs/2211.04772


"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

from models.audio.abstract_audio_model import AudioModel
from utils import reduce_fn
from . import hear_mn


load_dotenv()  # take environment variables from .env for checkpoints folder
# path to download/load checkpoints
torch.hub.set_dir(Path(os.environ["PROJECT_ROOT"]) / "checkpoints")


# Mono input sounds @32 kHz which get split based on a window size of 160ms with a hop size of 50ms.
# Then, for each of the resulting frames, a STFT (25 ms window and hop size of 10 ms) and a
# Log mel spectrograms with 128 frequency bands are computed and serve as input to the models.

##### Model terminology

# Clf 1 denotes the features resulting from global pooling,
# Clf 2 is the embedding from the penultimate linear layer and
# Clf 3 are the logits predicted for the 527 classes of AudioSet
# all_se denotes the output of all SE bottleneck layers
# b{i} denotes the features extracted from the i-th block using global avg pooling
# all_b denotes the features extracted from blocks b2, b11, b13, b15 using global avg pooling


class EfficientATWrapper(AudioModel):
    """
    Wrapper class around EfficientAT [1] models for integration into the current pipeline.

    stripped down version of the EfficientAT repo available at
    https://github.com/fschmid56/EfficientAT_HEAR/tree/main
    """

    def __init__(self, model_name: str, reduction: str) -> None:
        """
        Args:
            model_name (str): name of the model to load. See `./hear_mn/` for available models.
            reduction (str): the method used to reduce the model outputs to a single vector per
            audio input. Available reduction methods are available in utils/reduce_fn.py
        """
        super().__init__()
        self._module = getattr(hear_mn, model_name)
        self.model = self._module.load_model()
        self.model_name = model_name

        self.reduce_fn = getattr(reduce_fn, reduction)
        if reduction == "identity":
            raise NotImplementedError("Non-reduced outputs are not supported yet.")

    @property
    def sample_rate(self) -> int:
        """Return the required input audio signal's sample rate."""
        return 32_000

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return self.model_name

    @property
    def in_channels(self) -> int:
        """Return the required number of input audio channels."""
        return 1

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters of the model."""
        return sum(p.numel() for p in self.parameters())

    @property
    def out_features(self) -> int:
        """Return the number of output features of the model (assuming `{max, avg}_time_pool`)."""
        if self.name == "mn20_all_b_mel_avgs":
            return 1072
        elif self.name == "mn20_all_b":
            return 944  # excluding the 128 mel bins
        elif self.name == "mn04_all_b_mel_avgs":
            return 320
        elif self.name == "mn04_all_b":
            return 192  # excluding the 128 mel bins
        else:
            raise NotImplementedError()

    @property
    def includes_mel(self) -> bool:
        """Return whether the model concatenate the reduced mel spectrogram to its output."""
        if self.name.split("_")[-2] == "mel":
            return True
        return False

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio (torch.Tensor): mono input sounds @32khz of shape (n_sounds, n_channels=1, n_samples)
            in the range [-1, 1].

        Returns:
            torch.Tensor: audio embeddings of shape:
            - (n_sounds, embed_size) if `{max,avg}_time_pool` is used as reduction function,
            where embed_size=self.out_features.
            - (n_sounds, n_timestamps) if `{max,avg}_channel_pool` is used as reduction function.
            - (n_sounds, embed_size * n_timestamps) if `flatten` is used as reduction function.
            - (n_sounds, embed_size, n_timestamps) if `identity` is used as reduction function.
        """
        # passt requires mono input audio of shape (n_sounds, n_samples)
        audio = audio.squeeze(-2)
        embeddings, _ = self._module.get_timestamp_embeddings(audio, self.model)
        return self.reduce_fn(embeddings.swapdims_(-1, -2))  # swap time and channel dimsW


def mn20_all_b_mel_avgs_avg_time_pool():
    return EfficientATWrapper(model_name="mn20_all_b_mel_avgs", reduction="avg_time_pool")


def mn04_all_b_mel_avgs_avg_time_pool():
    return EfficientATWrapper(model_name="mn04_all_b_mel_avgs", reduction="avg_time_pool")


def mn20_all_b_avg_time_pool():
    return EfficientATWrapper(model_name="mn20_all_b", reduction="avg_time_pool")


def mn04_all_b_avg_time_pool():
    return EfficientATWrapper(model_name="mn04_all_b", reduction="avg_time_pool")


mn20_mel = mn20_all_b_mel_avgs_avg_time_pool  # sound attr. eval mean: 0.954
mn04_mel = mn04_all_b_mel_avgs_avg_time_pool  # sound attr. eval mean: 0.946
mn20 = mn20_all_b_avg_time_pool  # sound attr. eval mean: 0.931
mn04 = mn04_all_b_avg_time_pool  # sound attr. eval mean: 0.902
