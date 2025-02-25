# pylint: disable=E1102
"""
Audio helper functions
"""
from typing import Optional
import torch
import torch.nn.functional as F
import torchaudio.functional as Fa
from torch import nn

from utils.audio import MelSTFT


class MFCCDist(nn.Module):
    """
    mfcc distance
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_mfcc: int = 40,
        distance: str = "L1",
        sr: int = 44_100,
        win_length: int = 1024,
        hop_length: int = 512,
        n_fft: int = 1024,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        eps: float = 1e-8,
        fast_normalization: bool = False,
    ):
        super().__init__()
        self.mel = MelSTFT(
            n_mels=n_mels,
            sr=sr,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            fmin=fmin,
            fmax=fmax,
            eps=eps,
            fast_normalization=fast_normalization,
        )

        if distance == "L1":
            self.dist_fn = F.l1_loss
        elif distance == "L2":
            self.dist_fn = F.mse_loss
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

        self.n_mfcc = n_mfcc
        # create DCT matrix
        self.register_buffer(
            "dct_mat",
            Fa.create_dct(n_mfcc=self.n_mfcc, n_mels=n_mels, norm="ortho"),
            persistent=False,
        )

    def forward(self, x, y):
        x = self.mel(x)  # (b, n_mels, t)
        y = self.mel(y)  # (b, n_mels, t)
        # compute mfcc and remove first band (signal power)
        # (b, t, n_mels) @ (n_mels, n_mfcc) -> (b, t, n_mels -1) -> (b, n_mels-1, t)
        x = torch.matmul(x.transpose(-1, -2), self.dct_mat)[..., 1:].transpose(-1, -2)
        y = torch.matmul(y.transpose(-1, -2), self.dct_mat)[..., 1:].transpose(-1, -2)
        return self.dist_fn(x, y)


if __name__ == "__main__":
    from functools import partial

    preds = torch.empty((16, 1, 32_000 * 5)).uniform_(-1, 1)
    targets = torch.empty((16, 1, 2_000 * 5)).uniform_(-1, 1)
    mfccd = partial(MFCCDist, n_mels=128, n_mfcc=40, distance="L1")
    mfccd = MFCCDist(sr=32_000)
    metric = mfccd(preds, targets)
    print(f"MFCCD: {metric}")
