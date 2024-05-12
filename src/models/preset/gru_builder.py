from typing import Dict, Optional
import torch
from torch import nn

from models.preset.embedding_layers import OneHotEncoding
from utils.synth import PresetHelper


class GRUBuilder(nn.Module):
    def __init__(
        self,
        out_features: int,
        embedding_layer: nn.Module,
        hidden_features: int = 1024,
        num_layers: int = 2,
        dropout_p: float = 0.0,
        pre_norm: bool = False,
        embedding_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.embedding_layer = embedding_layer(**embedding_kwargs)
        self.in_features = self.embedding_layer.embedding_dim

        self.norm_pre = nn.LayerNorm(self.in_features) if pre_norm else nn.Identity()

        self.gru = nn.GRU(
            input_size=self.in_features,
            hidden_size=hidden_features // 2,  # since bi-GRU
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_p,
        )

        self.out_block = nn.Sequential(
            nn.Linear(
                in_features=hidden_features,
                out_features=hidden_features,
            ),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        )

        self.init_weights()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_weights(self) -> None:
        self.embedding_layer.init_weights()

        if isinstance(self.norm_pre, nn.LayerNorm):
            nn.init.ones_(self.norm_pre.weight)
            nn.init.zeros_(self.norm_pre.bias)

        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for name, param in self.out_block.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]  # batch size
        x = self.embedding_layer(x)
        # add input size dimension to input if needed (e.g., for 1D input such as OH encoding)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        x = self.norm_pre(x)
        _, x = self.gru(x)

        # get bi-GRU last layer's last hidden state and reshape for output block
        x = x[-2:].transpose(0, 1).reshape(n, -1)
        x = self.out_block(x)
        return x


def gru_oh(out_features: int, preset_helper: PresetHelper, **kwargs) -> nn.Module:
    """
    BiGRU with One-Hot encoded categorical synthesizer parameters.
    """
    return GRUBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        hidden_features=kwargs.get("hidden_features", 1024),
        num_layers=kwargs.get("num_layers", 2),
        dropout_p=kwargs.get("dropout_p", 0.0),
        embedding_kwargs={"preset_helper": preset_helper},
    )


if __name__ == "__main__":
    import os
    from pathlib import Path
    from torch.utils.data import DataLoader
    from data.datasets import SynthDatasetPkl

    DATASET_FOLDER = Path(os.environ["PROJECT_ROOT"]) / "data" / "datasets"
    DATASET_PATH = DATASET_FOLDER / "talnm_mn04_size=65536_seed=45858_dev_val_v1"

    BATCH_SIZE = 32
    OUT_FEATURES = 192

    PARAMETERS_TO_EXCLUDE_STR = (
        "master_volume",
        "voices",
        "lfo_1_sync",
        "lfo_1_keytrigger",
        "lfo_2_sync",
        "lfo_2_keytrigger",
        "envelope*",
        "portamento*",
        "pitchwheel*",
        "delay*",
    )

    p_helper = PresetHelper("talnm", PARAMETERS_TO_EXCLUDE_STR)

    dataset = SynthDatasetPkl(path_to_dataset=DATASET_PATH, mmap=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = gru_oh(
        out_features=OUT_FEATURES,
        preset_helper=p_helper,
        hidden_features=1536,
        num_layers=4,
    )
    print(model)
    print(f"num parameters: {model.num_parameters}")

    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    for x, _ in loader:
        out = model(x)
        print(out.shape)
        break
    print("")
