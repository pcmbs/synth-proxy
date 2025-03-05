"""
Module implementing a torch Dataset used to load audio embeddings (targets) and
synth parameters (input) from .pkl files.
"""

from pathlib import Path
from typing import Optional, Union
import torch
from torch.utils.data import Dataset


class SynthDatasetPkl(Dataset):
    """
    Dataset used to load synth parameters, audio embeddings, and optionally mel spectrograms
    from .pkl files.
    """

    def __init__(
        self,
        path_to_dataset: Union[str, Path],
        split: str = "all",
        has_mel: bool = False,
        mel_norm: Optional[str] = "min_max",
        mmap: bool = True,
    ):
        """
        Dataset used to load audio embeddings (targets) and synth parameters
        (features) from .pkl files.

        Args
            path_to_dataset (Union[str, Path]): path to the folder containing the pickled files
            split (str): name of the split to load. (Default: "all")
            has_mel (bool): whether the dataset contains spectrograms or not. (Default: False)
            mel_norm (str): type of normalization to apply to the mel spectrograms.
            Should be in ["min_max", "mean_std"]. (Default: "min_max")
            mmap (bool): whether the audio_embeddings.pkl and synth_parameters.pkl tensors should
            be mmaped rather than loading all the storages into memory.
            This can be advantageous for large datasets. (Default: True)
        """
        super().__init__()

        path_to_dataset = Path(path_to_dataset)
        if not path_to_dataset.is_dir():
            raise ValueError(f"{path_to_dataset} is not a directory.")

        self.path_to_dataset = path_to_dataset

        with open(self.path_to_dataset / "configs.pkl", "rb") as f:
            self.configs_dict = torch.load(f)

        assert split in ["train", "test", "all"], f"Unknown split: {split}"
        self.split = split

        self.has_mel = has_mel
        assert mel_norm in ["min_max", "mean_std", None], f"Unknown mel normalization: {mel_norm}"
        self.mel_norm = mel_norm

        # whether or not to mmap the dataset (pickled torch tensors)
        self.is_mmap = mmap

        # load the dataset in __getitem__() to avoid unexpected high memory usage when num_workers>0
        self.audio_embeddings = None
        self.synth_parameters = None
        self.mel = None
        self.mel_stats = None

    @property
    def audio_fe_name(self) -> str:
        "Audio model used as feature extractor."
        return self.configs_dict["audio_fe"]

    @property
    def embedding_dim(self) -> int:
        "Audio model's output dimension."
        return self.configs_dict["num_outputs"]

    @property
    def mel_cfg(self) -> int:
        "Mel spectrograms configs."
        if not self.has_mel:
            return None
        return self.configs_dict["mel_cfg"]

    @property
    def name(self) -> str:
        "Dataset name"
        return self.path_to_dataset.stem

    @property
    def synth_name(self) -> str:
        "Synthesizer name"
        return self.configs_dict["synth"]

    @property
    def num_used_synth_parameters(self) -> int:
        "Number of used synthesizer parameters used to generate the dataset."
        return self.configs_dict["num_used_params"]

    @property
    def embedding_size_in_mb(self) -> float:
        """Size of the audio embeddings tensor in MB."""
        return round(self.audio_embeddings.element_size() * self.audio_embeddings.nelement() * 1e-6, 2)

    @property
    def synth_parameters_size_in_mb(self) -> float:
        """Size of the synthesizer parameters tensor in MB."""
        return round(self.synth_parameters.element_size() * self.synth_parameters.nelement() * 1e-6, 2)

    @property
    def mel_size_in_mb(self) -> float:
        """Size of the mel specgrams tensor in MB."""
        if not self.has_mel:
            return None
        return round(self.mel.element_size() * self.mel.nelement() * 1e-6, 2)

    def __len__(self) -> int:
        if self.split == "all":
            return self.configs_dict["dataset_size"]
        if self.split == "train":
            return self.configs_dict["train_size"]
        if self.split == "test":
            return self.configs_dict["test_size"]
        return 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        if not self.has_mel:
            # load and mmap the dataset during the first call of __getitem__
            if self.audio_embeddings is None or self.synth_parameters is None:
                self._load_dataset()

            return self.synth_parameters[index], self.audio_embeddings[index]

        if self.audio_embeddings is None or self.synth_parameters is None or self.mel is None:
            self._load_dataset()

        return self.synth_parameters[index], self.audio_embeddings[index], self.mel[index]

    def _load_dataset(self) -> None:
        suffix = "_" + self.split if self.split != "all" else ""
        self.audio_embeddings = torch.load(
            str(self.path_to_dataset / f"audio_embeddings{suffix}.pkl"), map_location="cpu", mmap=self.is_mmap
        )
        self.synth_parameters = torch.load(
            str(self.path_to_dataset / f"synth_parameters{suffix}.pkl"), map_location="cpu", mmap=self.is_mmap
        )
        assert len(self.audio_embeddings) == len(self.synth_parameters)

        if self.has_mel:
            self.mel = torch.load(
                str(self.path_to_dataset / f"specgrams{suffix}.pkl"), map_location="cpu", mmap=self.is_mmap
            )
            suffix_stats = "_train" if self.split != "all" else ""
            self.mel_stats = torch.load(
                str(self.path_to_dataset / f"stats{suffix_stats}.pkl"),
                map_location="cpu",
            )
            if self.mel_norm == "min_max":
                self.mel = self._min_max_scaler(self.mel)
            elif self.mel_norm == "mean_std":
                self.mel = self._mean_std_scaler(self.mel)

            assert len(self.mel) == len(self.audio_embeddings)

    def _min_max_scaler(self, x: torch.Tensor) -> torch.Tensor:
        """Min-max scaler in range [-1, 1] for the mel spectrograms."""
        assert self.has_mel and self.mel_stats and self.mel_norm == "min_max"
        return -1 + 2 * (x - self.mel_stats["min"]) / (self.mel_stats["max"] - self.mel_stats["min"])

    def _mean_std_scaler(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-std scaler for the mel spectrograms."""
        assert self.has_mel and self.mel_stats and self.mel_norm == "mean_std"
        return (x - self.mel_stats["mean"]) / self.mel_stats["std"]


if __name__ == "__main__":
    import os
    from timeit import default_timer as timer
    from torch.utils.data import DataLoader

    NUM_EPOCH = 1
    PATH_TO_DATASET = Path(os.environ["PROJECT_ROOT"]) / "data" / "datasets" / "eval" / "diva_mn20_hc_v1"

    dataset = SynthDatasetPkl(
        PATH_TO_DATASET,
        split="train",
        has_mel=True,
        mel_norm="mean_std",
        mmap=True,
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    # add timer here
    start = timer()
    for e in range(NUM_EPOCH):
        print(f"Epoch {e}")
        for i, (params, audio, mel) in enumerate(loader):
            p, a, m = params, audio, mel
            if i % 100 == 0:
                print(f"{i} batch generated")
    print(f"Total time: {timer() - start}. Approximate time per epoch: {(timer() - start) / NUM_EPOCH}")
    print("mel min/max", dataset.mel.min(), dataset.mel.max())
    print("mel mean/std", dataset.mel.mean(), dataset.mel.std())
    print("")
