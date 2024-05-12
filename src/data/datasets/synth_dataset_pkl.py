"""
Module implementing a torch Dataset used to load audio embeddings (targets) and
synth parameters (input) from .pkl files.
"""

from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset


class SynthDatasetPkl(Dataset):
    """
    Dataset used to load audio embeddings (targets) and synth parameters
    (features) from .pkl files.
    """

    def __init__(self, path_to_dataset: Union[str, Path], mmap: bool = True):
        """
        Dataset used to load audio embeddings (targets) and synth parameters
        (features) from .pkl files.

        Args
            path_to_dataset (Union[str, Path]): path to the folder containing the pickled files
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

        # whether or not to mmap the dataset (pickled torch tensors)
        self.is_mmap = mmap

        # load the dataset in __getitem__() to avoid unexpected high memory usage when num_workers>0
        self.audio_embeddings = None
        self.synth_parameters = None

    @property
    def audio_fe_name(self) -> str:
        "Audio model used as feature extractor."
        return self.configs_dict["audio_fe"]

    @property
    def embedding_dim(self) -> int:
        "Audio model's output dimension."
        return self.configs_dict["num_outputs"]

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

    def __len__(self) -> int:
        return self.configs_dict["dataset_size"]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        # load and mmap the dataset during the first call of __getitem__
        if self.audio_embeddings is None or self.synth_parameters is None:
            self._load_dataset()

        return self.synth_parameters[index], self.audio_embeddings[index]

    def _load_dataset(self) -> None:
        self.audio_embeddings = torch.load(
            str(self.path_to_dataset / "audio_embeddings.pkl"), map_location="cpu", mmap=self.is_mmap
        )
        self.synth_parameters = torch.load(
            str(self.path_to_dataset / "synth_parameters.pkl"), map_location="cpu", mmap=self.is_mmap
        )
        assert len(self.audio_embeddings) == len(self.synth_parameters)


if __name__ == "__main__":
    import os
    from timeit import default_timer as timer
    from torch.utils.data import DataLoader

    NUM_EPOCH = 5
    PATH_TO_DATASET = (
        Path(os.environ["PROJECT_ROOT"]) / "data" / "datasets" / "talnm_mn04_size=10240000_seed=500_train_v1"
    )

    dataset = SynthDatasetPkl(PATH_TO_DATASET)

    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)

    # add timer here
    start = timer()
    for e in range(NUM_EPOCH):
        print(f"Epoch {e}")
        for i, (params, audio) in enumerate(loader):
            if i % 1000 == 0:
                print(f"{i} batch generated")
    print(f"Total time: {timer() - start}. Approximate time per epoch: {(timer() - start) / NUM_EPOCH}")

    print("")
