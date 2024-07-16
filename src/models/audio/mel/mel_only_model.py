"""
Mel only model
"""

import torch
import torchaudio.functional as Fa
import torchaudio.transforms as T

from models.audio.abstract_audio_model import AudioModel
from utils import reduce_fn


class MelModel(AudioModel):
    """
    Log Mel Spectrogram or MFCC.
    """

    def __init__(
        self,
        n_mels: int,
        n_mfcc: int,
        min_db: float,
        reduction: str,
        sample_rate: int = 44_100,
    ) -> None:
        """
        Args:
            n_mels (int): Number of mel filters.
            n_mfcc (int): Number of mfcc filters to use. Pass 0 to generate mel spectrogram.
            min_db (int)): Minimum db value to use, i.e., magnitude falling below `min_db`
            will be clamped to 10^(min_db/10),
            reduction (str): the method used to reduce the model outputs to a single vector per
            audio input. Available reduction methods are available in utils/reduce_fn.py
            sample_rate (int): Sample rate of the audio signal.
        """
        super().__init__()
        self._sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.min_db = min_db

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            win_length=2048,
            hop_length=1024,
            f_min=20,
            f_max=16000,
            n_mels=self.n_mels,
            normalized="window",  # normalize the STFT by the window's L2 norm (for energy conservation)
            norm=None,
            mel_scale="htk",
        )
        if n_mfcc > 0:
            self.n_mfcc = n_mfcc
            # create DCT matrix
            self.register_buffer(
                "dct_mat",
                Fa.create_dct(n_mfcc=self.n_mfcc, n_mels=self.n_mels, norm="ortho"),
                persistent=False,
            )
        else:
            self.n_mfcc = None

        self.reduce_fn = getattr(reduce_fn, reduction)
        if reduction == "identity":
            raise NotImplementedError("Non-reduced outputs are not supported yet.")

        # normalization statistics (from AudioSet)
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974

        # # normalization statistics (from EfficientAT)
        # self.norm_mean = 4.5
        # self.norm_std = 5

        # normalization statistics (based on 2048 random preset from tal_noisemaker)
        # self.norm_mean = -48.586097717285156
        # self.norm_std = 18.549829483032227

    @property
    def sample_rate(self) -> int:
        """Return the required input audio signal's sample rate."""
        return self._sample_rate

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return f"mel{self.n_mels}_{self.n_mfcc}_{self.min_db}"

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
        return self.n_mfcc or self.n_mels

    @property
    def includes_mel(self) -> bool:
        """Return whether the model concatenate the reduced mel spectrogram to its output."""
        return True

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio (torch.Tensor): mono input sounds @self.sample_rate of shape
            (n_sounds, n_channels=1, n_samples) in the range [-1, 1].

        Returns:
            torch.Tensor: audio embeddings of shape:
            - (n_sounds, embed_size) if `{max,avg}_time_pool` is used as reduction function,
            where embed_size=self.out_features.
            - (n_sounds, n_timestamps) if `{max,avg}_channel_pool` is used as reduction function.
            - (n_sounds, embed_size * n_timestamps) if `flatten` is used as reduction function.
            - (n_sounds, embed_size, n_timestamps) if `identity` is used as reduction function.
        """
        if self.n_mfcc is not None:
            return self.reduce_fn(self._compute_mfcc(audio).squeeze_(1))

        return self.reduce_fn(self._compute_mel(audio).squeeze_(1))

    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        mel_specgram = self.mel_spectrogram(audio)  # mel scaled spectrogram
        # clamp magnitude by below to 10^(MIN_DB/10) (/10 and not /20 since squared spectrogram)
        mel_specgram = torch.maximum(mel_specgram, torch.tensor(10 ** (self.min_db / 10)))
        mel_specgram = 10 * torch.log10(mel_specgram)  # log magnitude spectrogram (in dB)
        mel_specgram = (mel_specgram - self.norm_mean) / self.norm_std
        return mel_specgram

    def _compute_mfcc(self, audio: torch.Tensor) -> torch.Tensor:
        mel_specgram = self._compute_mel(audio)  # mel spectrogram
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), self.dct_mat).transpose(-1, -2)  # compute MFCCs
        return mfcc


def mel128_mfcc0_min80_avg_time_pool() -> MelModel:
    return MelModel(n_mels=128, n_mfcc=0, min_db=-80.0, reduction="avg_time_pool")


# best resulting mel model from the sound attributes ranking evaluation
mel128 = mel128_mfcc0_min80_avg_time_pool  # sound attr. eval mean: 0.908
