# pylint: disable=E1102
"""
Audio helper functions
"""
from typing import Optional
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as Fa
from torch import nn


class MelSTFT(nn.Module):
    """
    Class to compute mel spectrogram
    Adapted from: https://github.com/fschmid56/EfficientAT_HEAR/commits/main/hear_mn/models/preprocess.py
    """

    def __init__(
        self,
        n_mels: int = 128,
        sr: int = 44_100,
        win_length: int = 1024,
        hop_length: int = 512,
        n_fft: int = 1024,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        eps: float = 1e-8,
        fast_normalization: bool = False,
    ):
        torch.nn.Module.__init__(self)

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.hopsize = hop_length
        self.eps = eps
        self.fast_normalization = fast_normalization
        self.register_buffer("window", torch.hann_window(win_length, periodic=False), persistent=False)

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-0.97, 1]]]), persistent=False)

    def forward(self, x):
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=False,
        )
        x = (x**2).sum(dim=-1)  # power mag

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sr,
            self.fmin,
            self.fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0), device=x.device
        )
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = 10 * torch.log10(melspec.clamp(min=self.eps))  # log magnitude spectrogram (in dB)

        if self.fast_normalization:
            melspec = (melspec + 4.5) / 5.0  # fast normalization

        return melspec


if __name__ == "__main__":
    import itertools

    MEL_CFG = {
        "num_samples": 30_000,
        "length": 5,
        "sr": 44_100,
        "num_bins": [64, 128, 256],
        "stft_win": [512, 1024, 2048],
    }

    cfg = itertools.product(MEL_CFG["num_bins"], MEL_CFG["stft_win"])

    print(f" \nResults for {MEL_CFG['num_samples']} samples of {MEL_CFG['length']} secs @ {MEL_CFG['sr']} Hz")
    for m, w in cfg:
        h = w // 2
        mel = MelSTFT(n_mels=m, win_length=w, hop_length=h, sr=MEL_CFG["sr"], fmax=MEL_CFG["sr"] // 2)
        x = torch.empty((1, MEL_CFG["sr"] * MEL_CFG["length"])).uniform_(-1, 1)
        try:
            specgram = mel(x)
        except RuntimeError:
            print(f"\nError with mel={m}, win={w}, hop={h}")
            continue

        size_in_gb = MEL_CFG["num_samples"] * specgram.nelement() * specgram.element_size() * 1e-9
        print(f"\nResults for mel={m}, win={w}, hop={h} ({size_in_gb:.2f} GB):")

        print(f"    mel shape: {specgram.shape}")
