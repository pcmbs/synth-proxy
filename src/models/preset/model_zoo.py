"""
Script containing the constructors for all preset encoders, which includes:
- mlp_raw (not used)
- mlp_oh
- hn_oh
- rn_oh (not used)
- hn_pt
- hn_ptgru
- gru_oh (not used)
- tfm 
"""

from torch import nn

from models.preset.embedding_layers import (
    OneHotEncoding,
    PresetTokenizer,
    PresetTokenizerWithGRU,
    RawParameters,
)
from models.preset.gru_builder import GRUBuilder
from models.preset.mlp_builder import MlpBuilder, MLPBlock, HighwayBlock, ResNetBlock
from models.preset.tfm_builder import TfmBuilder
from utils.synth import PresetHelper

# TODO: Add support to load weights from a checkpoint


##############################################################################################################
#### MLP based models
##############################################################################################################
def mlp_raw(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 1,
    hidden_features: int = 2048,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
) -> nn.Module:
    """
    MLP with BatchNorm+ReLU blocks (by default) and raw parameter values in range [0,1].

    Args:
        out_features (int): number of output features. Should be the same as the used audio model.
        preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        num_blocks (int): Number of blocks in the network. (Default: 1).
        hidden_features (int): Hidden dimension per block. (Default: 2048).
        block_norm (str): Normalization layer type. (Default: 'BatchNorm1d').
        block_act_fn (str): Activation function. (Default: 'ReLU').
        block_dropout_p (float): Dropout probability. (Default: 0.0).

    Returns:
        nn.Module`: A Pytorch module representing the constructed preset encoder.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=RawParameters,
        block_layer=MLPBlock,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        block_kwargs={"norm": block_norm, "act_fn": block_act_fn, "dropout_p": block_dropout_p},
        embedding_kwargs={"preset_helper": preset_helper},
    )


def mlp_oh(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 1,
    hidden_features: int = 2048,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
) -> nn.Module:
    """
    MLP taking one-hot encoded categorical synthesizer parameters as input.

    Args:
        out_features (int): number of output features. Should be the same as the used audio model.
        preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        num_blocks (int): Number of blocks in the network. (Default: 1).
        hidden_features (int): Hidden dimension per block. (Default: 2048).
        block_norm (str): Normalization layer type. (Default: 'BatchNorm1d').
        block_act_fn (str): Activation function. (Default: 'ReLU').
        block_dropout_p (float): Dropout probability. (Default: 0.0).

    Returns:
        nn.Module`: A Pytorch module representing the constructed preset encoder.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=MLPBlock,
        num_blocks=num_blocks,
        hidden_features=hidden_features,
        block_kwargs={"norm": block_norm, "act_fn": block_act_fn, "dropout_p": block_dropout_p},
        embedding_kwargs={"preset_helper": preset_helper},
    )


def hn_oh(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 6,
    hidden_features: int = 768,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
) -> nn.Module:
    """
    MLP-based HighwayNet taking one-hot encoded categorical synthesizer parameters as input.

    Args:
        out_features (int): number of output features. Should be the same as the used audio model.
        preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        num_blocks (int): Number of blocks in the network. (Default: 6).
        hidden_features (int): Hidden dimension per block. (Default: 768).
        block_norm (str): Normalization layer type. (Default: 'BatchNorm1d').
        block_act_fn (str): Activation function. (Default: 'ReLU').
        block_dropout_p (float): Dropout probability. (Default: 0.0).

    Returns:
        nn.Module`: A Pytorch module representing the constructed preset encoder.
    """

    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=HighwayBlock,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        block_kwargs={"norm": block_norm, "act_fn": block_act_fn, "dropout_p": block_dropout_p},
        embedding_kwargs={"preset_helper": preset_helper},
    )


def rn_oh(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 4,
    hidden_features: int = 256,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
    block_residual_dropout_p: float = 0.0,
) -> nn.Module:
    """
    ResNet taking one-hot encoded categorical synthesizer parameters as input.

    Args:
        out_features (int): number of output features. Should be the same as the used audio model.
        preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        num_blocks (int): Number of blocks in the network. (Default: 4).
        hidden_features (int): Hidden dimension per block. (Default: 256).
        block_norm (str): Normalization layer type. (Default: 'BatchNorm1d').
        block_act_fn (str): Activation function. (Default: 'ReLU').
        block_dropout_p (float): Dropout probability. (Default: 0.0).
        block_residual_dropout_p (float): Dropout probability for residuals. (Default: 0.0).

    Returns:
        nn.Module`: A Pytorch module representing the constructed preset encoder.
    """

    return MlpBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        block_layer=ResNetBlock,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        block_kwargs={
            "norm": block_norm,
            "act_fn": block_act_fn,
            "dropout_p": block_dropout_p,
            "residual_dropout_p": block_residual_dropout_p,
        },
        embedding_kwargs={"preset_helper": preset_helper},
    )


def hn_pt(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 6,
    hidden_features: int = 512,
    token_dim: int = 64,
    pe_dropout_p: float = 0.0,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
) -> nn.Module:
    """
    MLP-based HighwayNet taking a flattened sequence of tokenized synthesizer parameter as input.

    Args:
        out_features (int): number of output features. Should be the same as the used audio model.
        preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        num_blocks (int): Number of blocks in the network. (Default: 6).
        hidden_features (int): Hidden dimension per block. (Default: 512).
        token_dim (int): Dimension of each token. (Default: 64).
        pe_dropout_p (float): Dropout probability for positional encoding. (Default: 0.0).
        block_norm (str): Normalization layer type. (Default: 'BatchNorm1d').
        block_act_fn (str): Activation function. (Default: 'ReLU').
        block_dropout_p (float): Dropout probability. (Default: 0.0).

    Returns:
        nn.Module`: A Pytorch module representing the constructed preset encoder.
    """

    return MlpBuilder(
        out_features=out_features,
        embedding_layer=PresetTokenizer,
        block_layer=HighwayBlock,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        block_kwargs={"norm": block_norm, "act_fn": block_act_fn, "dropout_p": block_dropout_p},
        embedding_kwargs={
            "preset_helper": preset_helper,
            "token_dim": token_dim,
            "pe_type": None,
            "has_cls": False,
            "pe_dropout_p": pe_dropout_p,
        },
    )


def hn_ptgru(
    out_features: int,
    preset_helper: PresetHelper,
    num_blocks: int = 6,
    hidden_features: int = 768,
    token_dim: int = 384,
    pe_dropout_p: float = 0.0,
    gru_hidden_factor: float = 1.0,
    gru_num_layers: int = 1,
    gru_dropout_p: float = 0.0,
    block_norm: str = "BatchNorm1d",
    block_act_fn: str = "ReLU",
    block_dropout_p: float = 0.0,
    pre_norm: bool = False,
) -> nn.Module:
    """
    MLP-based HighwayNet with PresetTokenizer+BiGRU preset encoding.

    Args:
        out_features (int): number of output features. Should be the same as the used audio model.
        preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        num_blocks (int): Number of blocks in the network. (Default: 6).
        hidden_features (int): Hidden dimension per block. (Default: 768).
        token_dim (int): Dimension of each token. (Default: 384).
        pe_dropout_p (float): Dropout probability for positional encoding. (Default: 0.0).
        gru_hidden_factor (float): BiGRU hidden factor as a multiple of `token_dim`. (Default: 1.0).
        gru_num_layers (int): Number of BiGRU layers. (Default: 1).
        gru_dropout_p (float): Dropout probability for BiGRU. (Default: 0.0).
        block_norm (str): Normalization layer type. (Default: 'BatchNorm1d').
        block_act_fn (str): Activation function. (Default: 'ReLU').
        block_dropout_p (float): Dropout probability. (Default: 0.0).
        pre_norm (bool): Whether to apply layer normalization on tokens, before the BiGRU. (Default: False).

    Returns:
        nn.Module`: A Pytorch module representing the constructed preset encoder.
    """
    return MlpBuilder(
        out_features=out_features,
        embedding_layer=PresetTokenizerWithGRU,
        block_layer=HighwayBlock,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        block_kwargs={"norm": block_norm, "act_fn": block_act_fn, "dropout_p": block_dropout_p},
        embedding_kwargs={
            "preset_helper": preset_helper,
            "token_dim": token_dim,
            "pre_norm": pre_norm,
            "gru_hidden_factor": gru_hidden_factor,
            "gru_num_layers": gru_num_layers,
            "gru_dropout_p": gru_dropout_p,
            "pe_dropout_p": pe_dropout_p,
        },
    )


##############################################################################################################
#### GRU based models
##############################################################################################################
def gru_oh(
    out_features: int,
    preset_helper: PresetHelper,
    num_layers: int = 1,
    hidden_features: int = 1024,
    dropout_p: float = 0.0,
) -> nn.Module:
    """
    BiGRU taking one-hot encoded categorical synthesizer parameters as input.

    Args:
        out_features (int): number of output features. Should be the same as the used audio model.
        preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        num_layers (int): Number of BiGRU layers. (Default: 1).
        hidden_features (int): Hidden dimension per layer (each GRU gets //2). (Default: 1024).
        dropout_p (float): Dropout probability. (Default: 0.0).

    Returns:
        nn.Module`: A Pytorch module representing the constructed preset encoder.
    """
    return GRUBuilder(
        out_features=out_features,
        embedding_layer=OneHotEncoding,
        hidden_features=hidden_features,
        num_layers=num_layers,
        dropout_p=dropout_p,
        embedding_kwargs={"preset_helper": preset_helper},
    )


##############################################################################################################
#### TFM based models
##############################################################################################################
def tfm(
    out_features: int,
    preset_helper: PresetHelper,
    pe_type: str = "absolute",
    num_blocks: int = 6,
    hidden_features: int = 256,
    num_heads: int = 8,
    mlp_factor: float = 4.0,
    pooling_type: str = "cls",
    last_activation: str = "ReLU",
    pe_dropout_p: float = 0.0,
    block_activation: str = "relu",
    block_dropout_p: float = 0.0,
) -> nn.Module:
    """
    Vanilla Transformer taking a sequence of tokenized synthesizer parameters as input.

    Args:
        out_features (int): number of output features. Should be the same as the used audio model.
        preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        pe_type (str): Type of positional encoding to use. Only implemented for 'absolute' for now. (Default: 'absolute').
        num_blocks (int): Number of transformer blocks. (Default: 6).
        hidden_features (int): Hidden dimension per layer. (Default: 256).
        num_heads (int): Number of attention heads. (Default: 8).
        mlp_factor (float): Factor by which to multiply the hidden dimension in the MLP. (Default: 4.0).
        pooling_type (str): Pooling type to use. Can either be 'cls' or 'avg'. If 'cls' is passed then the a special token
        in prepended to the sequence of tokens and will be used as the final representation (hence no pooling occurs).
        If 'avg'is passed, no special token is added, and the final representation is the average across the output tokens
        (Default: 'cls').
        last_activation (str): Activation function to use is the projection head. (Default: 'ReLU').
        pe_dropout_p (float): Dropout probability for the positional encoding. (Default: 0.0).
        block_activation (str): Activation function to use in the transformer blocks. (Default: 'relu').
        block_dropout_p (float): Dropout probability for the transformer blocks. (Default: 0.0).

    Returns:
        nn.Module`: A Pytorch module representing the constructed preset encoder.
    """
    return TfmBuilder(
        out_features=out_features,
        tokenizer=PresetTokenizer,
        pe_type=pe_type,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        num_heads=num_heads,
        mlp_factor=mlp_factor,
        pooling_type=pooling_type,
        last_activation=last_activation,
        tokenizer_kwargs={"preset_helper": preset_helper, "pe_dropout_p": pe_dropout_p},
        block_kwargs={"activation": block_activation, "dropout": block_dropout_p},
    )


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    from dotenv import load_dotenv
    from omegaconf import OmegaConf

    from models import audio as audio_models

    load_dotenv()
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

    SYNTH = "talnm"

    MODEL = "hn_oh"
    AUDIO_FE = "mn04"

    CONF = {
        "hn_pt": {"num_blocks": 6, "hidden_features": 512, "token_dim": 64},
        "hn_ptgru": {"num_blocks": 6, "hidden_features": 768, "token_dim": 384},
        "hn_oh": {"num_blocks": 6, "hidden_features": 768},
        "mlp_oh": {"num_blocks": 1, "hidden_features": 2048},
        "tfm": {"num_blocks": 6, "hidden_features": 256, "num_heads": 8, "mlp_factor": 4.0},
    }

    synth_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "export" / "synth" / f"{SYNTH}.yaml")[
        "parameters_to_exclude_str"
    ]

    p_helper = PresetHelper(SYNTH, synth_cfg)

    model = getattr(sys.modules[__name__], MODEL)(
        out_features=getattr(audio_models, AUDIO_FE)().out_features, preset_helper=p_helper, **CONF[MODEL]
    )
    print(f"\nNumber of model parameters: {model.num_parameters}")
    print(f"\nNumber of used synthesizer parameters: {p_helper.num_used_parameters}\n")
    print(model)
