# pylint: disable=E1101:no-member,W1203
"""
Torch Dataset class for NSynth dataset
Adapted from: https://github.com/morris-frank/nsynth-pytorch/blob/master/nsynth/data.py 
"""

import json
import logging
from pathlib import Path
from typing import Union, List, Optional

import torch
import torch.utils
import torchaudio
from torch.utils.data import DataLoader, Dataset

from utils.audio import MelSTFT

# logger for this file
log = logging.getLogger(__name__)


class NSynthDataset(Dataset):
    """
    Dataset to handle the NSynth data in json/wav format.
    """

    def __init__(
        self,
        root: Union[Path, str],
        subset: str = "train",
        return_mel: bool = False,
        mel_kwargs: Optional[dict] = None,
        mel_stats: Optional[dict] = None,
        mel_norm: Optional[dict] = None,
        audio_length: float = 4.0,
        families: Optional[Union[str, List[str]]] = None,
        sources: Optional[Union[str, List[str]]] = None,
        pitch: Optional[Union[str, List[str]]] = None,
        velocities: Optional[Union[str, List[str]]] = None,
    ):
        """
        Args
            root (Union[Path, str]): The path to the dataset.
            Should contain the sub-folders for the splits as extracted from the .tar.gz.
            subset (str): The subset to use. Must be one of "valid", "test", "train".
            families (Optional[Union[str, List[str]]]): Only keep those Instrument families.
            Valid families: "bass", "brass", "flute", "guitar", "keyboard", "mallet",
            "organ", "reed", "string", "synth_lead", "vocal"
            sources (Optional[Union[str, List[str]]]): Only keep those instrument sources
            Valid sources: "acoustic", "electric", "synthetic".
            pitch (Optional[Union[str, List[int]]]): Only keep those pitch - should be in range [21,108].
            velocities (Optional[Union[str, List[int]]]): Only keep those velocities - should be in (25, 50, 75, 100, 127).
        """

        if return_mel:
            mel_kwargs = mel_kwargs or {}
            self.mel = MelSTFT(**mel_kwargs)
            assert mel_norm in ["min_max", "mean_std", None], f"Unknown mel normalization: {mel_norm}"
            self.mel_scaler = self._min_max_scaler if mel_norm == "min_max" else self._mean_std_scaler
            self.mel_norm = mel_norm
            self.mel_stats = mel_stats
            if self.mel.sr != self.sr:
                self.resampler = torchaudio.transforms.Resample(self.sr, self.mel.sr)
            else:
                self.resampler = None
        else:
            self.mel = None

        assert audio_length > 0
        self.audio_length = audio_length
        self._fadeout_length = int(self.sr * 0.1)
        self._fadeout = torch.linspace(1, 0, self._fadeout_length)

        self.subset = subset.lower()
        assert self.subset in ["valid", "test", "train"]

        root = Path(root) if isinstance(root, str) else root
        self.root = root / f"nsynth-{subset}"
        if not self.root.is_dir():
            raise ValueError("The given root path is not a directory." f"\nI got {self.root}")

        if not (self.root / "examples.json").is_file():
            raise ValueError("The given root path does not contain an `examples.json` file.")

        log.info(f"Loading NSynth data from split {self.subset} at {self.root}")

        with open(self.root / "examples.json", "r", encoding="utf-8") as f:
            self.attrs = json.load(f)

        families = [families] if isinstance(families, str) else families
        sources = [sources] if isinstance(sources, str) else sources
        pitch = [pitch] if isinstance(pitch, int) else pitch
        velocities = [velocities] if isinstance(velocities, int) else velocities

        if families:
            self.attrs = {k: a for k, a in self.attrs.items() if a["instrument_family_str"] in families}
        if sources:
            self.attrs = {k: a for k, a in self.attrs.items() if a["instrument_source_str"] in sources}
        if pitch:
            self.attrs = {k: a for k, a in self.attrs.items() if a["pitch"] in pitch}
        if velocities:
            self.attrs = {k: a for k, a in self.attrs.items() if a["velocity"] in velocities}

        log.info(f"\tFound {len(self)} samples.")

        files_on_disk = set(map(lambda x: x.stem, self.root.glob("audio/*.wav")))
        if not set(self.attrs) <= files_on_disk:
            raise FileNotFoundError

        self.names = list(self.attrs.keys())

    @property
    def sr(self):
        return 16_000

    def __len__(self):
        return len(self.attrs)

    def __str__(self):
        return f"NSynthDataset: {len(self):>7} samples in subset {self.subset}"

    def __getitem__(self, idx: int):
        name = self.names[idx]
        # returning attrs (for analysis) raised an error since the "quality_str"
        # key contains a list whose length vary from sample to sample
        # Hence, return the index and get the corresponding attrs using get_attrs()
        # attrs = self.attrs[name]
        with open(self.root / "audio" / f"{name}.wav", "rb") as f:
            audio, _ = torchaudio.load(f)

        if self.mel:
            audio = self._pad_or_truncate(audio)
            audio = audio if self.resampler is None else self.resampler(audio)
            specgram = self.mel(audio).squeeze(0)
            specgram = self.mel_scaler(specgram)
            return audio.squeeze(0), idx, specgram

        return audio.squeeze(0), idx

    def get_attrs(self, indices: torch.Tensor) -> Union[dict, List[dict]]:
        """
        Returns a list of attributes corresponding to the given indices.

        Args:
            indices (torch.Tensor): A 1D tensor or a scalar tensor of indices.

        Returns:
            Union[dict, List[dict]]: The attributes corresponding to the given indices.
        """
        if indices.ndim == 0:
            return self.attrs[self.names[indices]]

        attrs = [self.attrs[self.names[i]] for i in indices]
        return attrs

    def _min_max_scaler(self, x: torch.Tensor) -> torch.Tensor:
        """Min-max scaler in range [-1, 1] for the mel spectrograms."""
        return -1 + 2 * (x - self.mel_stats["min"]) / (self.mel_stats["max"] - self.mel_stats["min"])

    def _mean_std_scaler(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-std scaler for the mel spectrograms."""
        return (x - self.mel_stats["mean"]) / self.mel_stats["std"]

    def _pad_or_truncate(self, x: torch.Tensor) -> torch.Tensor:
        target_length = self.audio_length * self.sr
        if x.shape[-1] > target_length:  # fadeout
            x[..., -self._fadeout_len :] = x[..., -self._fadeout_len :] * self._fadeout
        elif x.shape[-1] < target_length:  # pad
            x = torch.nn.functional.pad(x, (0, int(target_length - x.shape[-1])), mode="constant", value=0)
        return x


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    PITCH = 60  # 60 samples with pitch 60 (C3)
    VELOCITY = None

    DATA_DIR = Path(os.environ["PROJECT_ROOT"]) / "data" / "datasets" / "eval"

    nsynth_dataset = NSynthDataset(
        root=DATA_DIR, subset="test", return_mel=False, pitch=PITCH, velocities=VELOCITY
    )
    nsynth_dataloader = DataLoader(nsynth_dataset, batch_size=8, shuffle=True)

    for i, batch in enumerate(nsynth_dataloader):
        audios, idxs, specs = batch
        attributes = nsynth_dataset.get_attrs(idxs)

        if i == 10:
            break

    print("✅")
