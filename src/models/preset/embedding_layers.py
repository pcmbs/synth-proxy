# pylint: disable=E1102
"""
Module implementing different embedding layers for the preset encoder.
"""
import math
from typing import Optional, Tuple
import torch
from torch import nn

from utils.synth import PresetHelper


class RawParameters(nn.Module):
    """Use raw parameter values in the range [0,1] given the category indices of categorical synth parameters."""

    def __init__(self, preset_helper: PresetHelper) -> None:
        """
        Use raw parameter values in the range [0,1] given the category indices of categorical synth parameters.

        Args:
            preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        """

        super().__init__()
        self._out_length = preset_helper.num_used_parameters
        self._grouped_used_parameters = preset_helper.grouped_used_parameters
        self.cat_parameters = self._group_cat_parameters_per_values()

    @property
    def out_length(self) -> int:
        return self._out_length

    @property
    def embedding_dim(self) -> int:
        return 1

    def init_weights(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # move instance attribute to device of input tensor (should only be done during the first forward pass)
        device = x.device
        if self.cat_parameters[0][0].device != device:
            self.cat_parameters = [(val.to(device), idx.to(device)) for (val, idx) in self.cat_parameters]

        for cat_values, indices in self.cat_parameters:
            x[..., indices] = cat_values[x[..., indices].to(torch.long)]
        return x

    def _group_cat_parameters_per_values(self):
        cat_parameters_val_dict = {}
        for (cat_values, _), indices in self._grouped_used_parameters["discrete"]["cat"].items():
            # create copy of the indices list to not modify the original ones
            indices = indices.copy()
            if cat_values in cat_parameters_val_dict:
                cat_parameters_val_dict[cat_values] += indices
            else:
                cat_parameters_val_dict[cat_values] = indices

        return [
            (torch.tensor(cat_values, dtype=torch.float32), torch.tensor(indices, dtype=torch.long))
            for cat_values, indices in cat_parameters_val_dict.items()
        ]


class OneHotEncoding(nn.Module):
    """One-hot encoding of categorical parameters."""

    def __init__(self, preset_helper: PresetHelper) -> None:
        """
        One-hot encoding of categorical parameters.

        Args:
            preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        """
        super().__init__()
        # used numerical and binary parameters indices
        self.noncat_idx = torch.tensor(preset_helper.used_noncat_parameters_idx, dtype=torch.long)
        self.num_noncat = len(self.noncat_idx)

        # used categorical parameters
        self.cat_idx = torch.tensor(preset_helper.used_cat_parameters_idx, dtype=torch.long)
        cat_offsets, self.total_num_cat = self._compute_cat_infos(preset_helper)
        self.register_buffer("cat_offsets", cat_offsets)
        self._out_length = self.total_num_cat + self.num_noncat

    @property
    def out_length(self) -> int:
        return self._out_length

    @property
    def embedding_dim(self) -> int:
        return 1

    def init_weights(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        oh_enc = torch.zeros(x.shape[0], self.out_length, device=x.device)

        # Assign noncat parameters
        oh_enc[:, : self.num_noncat] = x[:, self.noncat_idx]

        # Calculate and assign ones for categorical
        ones_idx = x[:, self.cat_idx].to(dtype=torch.long) + self.cat_offsets + self.num_noncat
        oh_enc.scatter_(1, ones_idx, 1)
        # oh_enc[torch.arange(x.shape[0]).unsqueeze(1), ones_idx] = 1

        return oh_enc

    def _compute_cat_infos(self, preset_helper: PresetHelper) -> Tuple[torch.Tensor, int]:
        """
        Compute the offsets for each categorical parameter and the total number of categories
        (i.e., sum over all categorical parameters' cardinality).

        Args:
            preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.

        Returns:
            cat_offsets (torch.Tensor): the offsets for each categorical parameter as a list cat_offsets[cat_param_idx] = offset.
            total_num_cat (int):  total number of categories.
        """
        cat_offsets = []
        offset = 0
        for i in preset_helper.used_cat_parameters_idx:
            cardinality = preset_helper.used_parameters[i].cardinality
            cat_offsets.append(offset)
            offset += cardinality
        total_num_cat = offset
        cat_offsets = torch.tensor(cat_offsets, dtype=torch.long)
        return cat_offsets, total_num_cat


class PresetTokenizer(nn.Module):
    """
    Synth Presets Tokenizer class.

    - Each non-categorical (numerical & binary) parameter is embedded using a learned linear projection.
    - Each categorical parameter is embedded using a nn.Embedding lookup table.

    Higly inspired from:
    https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/ft_transformer.py
    https://github.com/gwendal-lv/spinvae2/blob/main/model/presetmodel.py
    """

    def __init__(
        self,
        preset_helper: PresetHelper,
        token_dim: int,
        has_cls: bool,
        pe_type: Optional[str] = "absolute",
        pe_dropout_p: float = 0.0,
    ) -> None:
        """
        Args:
            preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
            token_dim (int): The token embedding dimension.
            cls_token (bool): Whether to add a class token.
            pe_type (Optional[str]): The type of positional encoding. Pass None to omit positional encoding.
            (Defaults: "absolute").
            pe_dropout_p (float): The dropout probability for the positional encoding. (Defaults: 0.0).
        """
        super().__init__()
        self._embedding_dim = token_dim
        self._out_length = preset_helper.num_used_parameters + has_cls

        if pe_type == "absolute":
            self.pos_encoding = PositionalEncoding(token_dim, dropout_p=pe_dropout_p)
        elif pe_type is None:
            self.pos_encoding = nn.Identity()
        else:
            raise NotImplementedError(f"Unknown positional encoding type: {pe_type}.")

        self.cls_token = nn.Parameter(torch.zeros(1, token_dim)) if has_cls else None

        self.noncat_idx = torch.tensor(preset_helper.used_noncat_parameters_idx, dtype=torch.long)
        self.noncat_tokenizer = nn.Parameter(torch.zeros(len(self.noncat_idx), token_dim))

        self.cat_idx = torch.tensor(preset_helper.used_cat_parameters_idx, dtype=torch.long)
        cat_offsets, total_num_cat = self._compute_cat_infos(preset_helper)
        self.cat_tokenizer = nn.Embedding(num_embeddings=total_num_cat, embedding_dim=token_dim)
        self.register_buffer("cat_offsets", cat_offsets)

    @property
    def out_length(self) -> int:
        return self._out_length

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def has_cls(self) -> bool:
        return self.cls_token is not None

    @property
    def pe_type(self) -> Optional[str]:
        if isinstance(self.pos_encoding, PositionalEncoding):
            return "absolute"
        return None

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_weights(self):
        nn.init.trunc_normal_(self.noncat_tokenizer, std=0.02)
        nn.init.trunc_normal_(self.cat_tokenizer.weight, std=0.02)
        if self.has_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.pe_type is not None:
            self.pos_encoding.init_weights()

    def forward(self, x):
        tokens = torch.zeros((*x.shape, self._embedding_dim), device=x.device)

        # Assign noncat embeddings
        noncat_tokens = self.noncat_tokenizer * x[:, self.noncat_idx, None]
        tokens[:, self.noncat_idx] = noncat_tokens

        # Assign cat embeddings
        cat_tokens = x[:, self.cat_idx].to(dtype=torch.long) + self.cat_offsets
        tokens[:, self.cat_idx] = self.cat_tokenizer(cat_tokens)

        if self.has_cls:
            tokens = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), tokens], dim=1)

        tokens = self.pos_encoding(tokens)

        return tokens

    def _compute_cat_infos(self, preset_helper: PresetHelper) -> Tuple[torch.Tensor, int]:
        """
        Compute the offsets for each categorical parameter and the total number of categories
        (i.e., sum over all categorical parameters' cardinality).

        Args:
            preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.

        Returns:
            cat_offsets (torch.Tensor): the offsets for each categorical parameter as a list cat_offsets[cat_param_idx] = offset.
            total_num_cat (int):  total number of categories.
        """
        cat_offsets = []
        offset = 0
        for i in preset_helper.used_cat_parameters_idx:
            cardinality = preset_helper.used_parameters[i].cardinality
            cat_offsets.append(offset)
            offset += cardinality
        total_num_cat = offset
        cat_offsets = torch.tensor(cat_offsets, dtype=torch.long)
        return cat_offsets, total_num_cat


class PositionalEncoding(nn.Module):
    """
    Absolute sinusoidal Positional Encoding.
    Adapted for batch-first inputs from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, embedding_dim: int, dropout_p: float = 0.0, max_len: int = 5000):
        """
        Absolute sinusoidal Positional Encoding.
        Adapted for batch-first inputs from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        Args:
            embedding_dim (int): embedding (i.e., model) dimension
            dropout_p (float): dropout probability
            max_len (int): maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        # equivalent to the original definition, exp for numerical stability
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(1, max_len, embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def init_weights(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, embedding_dim)
        """
        x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x)


class PresetTokenizerWithGRU(nn.Module):
    """
    Preset tokenizer with BiGRU.
    This module tokenize a batch of presets and then applies a BiGRU to the output.
    The output is a 2D context vector (batch_size, embedding_dim) made up of the bi-GRU's last hidden state,
    which can be used as input of a MLP based network.
    """

    def __init__(
        self,
        preset_helper: PresetHelper,
        token_dim: int,
        pre_norm: bool = False,
        gru_hidden_factor: int = 1,
        gru_num_layers: int = 1,
        gru_dropout_p: float = 0.0,
        pe_dropout_p: float = 0.0,
    ):
        """
        Preset tokenizer with BiGRU.
        This module tokenize a batch of presets and then applies a BiGRU to the output.
        The output is a 2D context vector (batch_size, embedding_dim) made up of the bi-GRU's last hidden state,
        which can be used as input of a MLP based network as alternative to flattening the sequence of tokens.

        Args:
            preset_helper (PresetHelper): preset helper
            token_dim (int): token dimension
            pre_norm (bool): whether to apply layer normalization on tokens, before the BiGRU.
            gru_hidden_factor (int): BiGRU hidden factor as a multiple of token_dim
            gru_num_layers (int): number of GRU layers
            gru_dropout_p (float): BiGRU dropout probability
            pe_dropout_p (float): positional encoding dropout probability
        """
        super().__init__()
        self._embedding_dim = gru_hidden_factor * token_dim

        self.tokenizer = PresetTokenizer(
            preset_helper=preset_helper,
            token_dim=token_dim,
            has_cls=False,
            pe_type="absolute",
            pe_dropout_p=pe_dropout_p,
        )

        self.pre_norm = nn.LayerNorm(token_dim) if pre_norm else nn.Identity()

        self.gru = nn.GRU(
            input_size=token_dim,
            hidden_size=int(gru_hidden_factor * token_dim // 2),  # // 2 since bi-GRU
            num_layers=gru_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=gru_dropout_p,
        )

        self.init_weights()

    @property
    def out_length(self) -> int:
        return 1

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def init_weights(self) -> None:
        self.tokenizer.init_weights()

        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len)
        """
        n = x.shape[0]  # batch size
        x = self.tokenizer(x)
        x = self.pre_norm(x)
        _, x = self.gru(x)
        # get BiGRU last layer's last hidden state and reshape
        x = x[-2:].transpose(0, 1).reshape(n, -1)
        return x


if __name__ == "__main__":
    import os
    from pathlib import Path
    from timeit import default_timer as timer
    from torch.utils.data import DataLoader

    from data.datasets import SynthDatasetPkl

    SYNTH = "diva"
    BATCH_SIZE = 512

    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"

    DATASET_FOLDER = Path(os.environ["PROJECT_ROOT"]) / "data" / "datasets"

    if SYNTH == "talnm":
        DATASET_PATH = DATASET_FOLDER / "talnm_mn04_size=65536_seed=45858_dev_val_v1"
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

    else:  # SYNTH == "diva":
        DATASET_PATH = DATASET_FOLDER / "diva_mn04_size=65536_seed=400_hpo_val_v1"
        PARAMETERS_TO_EXCLUDE_STR = (
            "main:output",
            "vcc:*",
            "opt:*",
            "scope1:*",
            "clk:*",
            "arp:*",
            "plate1:*",
            "delay1:*",
            "chrs2:*",
            "phase2:*",
            "rtary2:*",
            "*keyfollow",
            "*velocity",
            "env1:model",
            "env2:model",
            "*trigger",
            "*release_on",
            "env1:quantise",
            "env2:quantise",
            "env1:curve",
            "env2:curve",
            "lfo1:sync",
            "lfo2:sync",
            "lfo1:restart",
            "lfo2:restart",
            "mod:rectifysource",
            "mod:invertsource",
            "mod:addsource*",
            "*revision",
            "vca:pan",
            "vca:volume",
            "vca:vca",
            "vca:panmodulation",
            "vca:panmoddepth",
            "vca:mode",
            "vca:offset",
        )

    p_helper = PresetHelper(SYNTH, PARAMETERS_TO_EXCLUDE_STR)

    dataset = SynthDatasetPkl(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    oh = OneHotEncoding(p_helper)
    oh.to(DEVICE)

    # ft0 = FTTokenizer(p_helper, 128, False, None)
    # ft0.to(DEVICE)
    ft1 = PresetTokenizer(p_helper, 128, True, None)
    ft1.to(DEVICE)

    # with torch.no_grad():
    #     ft1.noncat_tokenizer.copy_(ft0.noncat_tokenizer)
    #     ft1.cat_tokenizer.weight.copy_(ft0.cat_tokenizer.weight)

    for params, _ in loader:
        # enc = oh(params.to(DEVICE))
        # tok0 = ft0(params.to(DEVICE))
        tok1 = ft1(params.to(DEVICE))
        break

    # print("")
