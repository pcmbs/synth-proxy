import pytest
import torch
from torch import nn

from models.preset.mlp_builder import MLPBlock
from models.preset.model_zoo import mlp_raw, mlp_oh
from utils.synth import PresetHelper


@pytest.fixture
def tal_preset_helper():
    """Return a PresetHelper instance for the TAL-NoiseMaker synthesizer"""
    parameters_to_exclude = (
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

    return PresetHelper("talnm", parameters_to_exclude)


@pytest.mark.parametrize(
    "out_features, kwargs",
    [
        (768, {}),
        (
            768,
            {
                "hidden_features": 2048,
                "num_blocks": 3,
                "block_kwargs": {"norm": "BatchNorm1d", "act_fn": "ReLU", "dropout_p": 0.1},
            },
        ),
        (
            768,
            {
                "hidden_features": 1536,
                "num_blocks": 2,
                "block_kwargs": {"norm": "LayerNorm", "act_fn": "GELU", "dropout_p": 0.0},
            },
        ),
    ],
)
def test_mlp_raw(tal_preset_helper, out_features, kwargs):
    mlp = mlp_raw(out_features=out_features, preset_helper=tal_preset_helper, **kwargs)

    # assert that the layers are correct
    assert isinstance(mlp.norm_pre, nn.Identity)
    assert isinstance(mlp.blocks[0], MLPBlock)

    # assert that the number of blocks is correct
    assert len(mlp.blocks) == kwargs.get("num_blocks", 2)

    # assert that the dropout probability is correct
    assert mlp.blocks[0].block[3].p == kwargs.get("block_kwargs", {}).get("dropout_p", 0.0)

    # assert that the input size is correct
    assert mlp.embedding_layer.out_length == tal_preset_helper.num_used_parameters
    assert mlp.blocks[0].block[0].in_features == tal_preset_helper.num_used_parameters

    # assert that the hidden size is correct
    assert mlp.out_layer.in_features == kwargs.get("hidden_features", 1024)
    assert mlp.blocks[0].block[0].out_features == kwargs.get("hidden_features", 1024)

    # assert that the normalization layer and activation function are correct
    assert str(mlp.blocks[0].act_fn).split(".")[-1][:-2] == kwargs.get("block_kwargs", {}).get(
        "act_fn", "ReLU"
    )
    assert str(mlp.blocks[0].norm).split(".")[-1][:-2] == kwargs.get("block_kwargs", {}).get(
        "norm", "BatchNorm1d"
    )

    # assert that the output size is correct
    assert mlp.out_layer.out_features == out_features


@pytest.mark.parametrize(
    "out_features, kwargs",
    [
        (768, {}),
        (
            768,
            {
                "hidden_features": 2048,
                "num_blocks": 3,
                "block_kwargs": {"norm": "BatchNorm1d", "act_fn": "ReLU", "dropout_p": 0.1},
            },
        ),
        (
            768,
            {
                "hidden_features": 1536,
                "num_blocks": 2,
                "block_kwargs": {"norm": "LayerNorm", "act_fn": "GELU", "dropout_p": 0.0},
            },
        ),
    ],
)
def test_mlp_oh(tal_preset_helper, out_features, kwargs):
    mlp = mlp_oh(out_features=out_features, preset_helper=tal_preset_helper, **kwargs)

    # assert that the layers are correct
    assert isinstance(mlp.norm_pre, nn.Identity)
    assert isinstance(mlp.blocks[0], MLPBlock)

    # assert that the number of blocks is correct
    assert len(mlp.blocks) == kwargs.get("num_blocks", 2)

    # assert that the dropout probability is correct
    assert mlp.blocks[0].block[3].p == kwargs.get("block_kwargs", {}).get("dropout_p", 0.0)

    # assert that the input size is correct
    assert mlp.blocks[0].block[0].in_features == mlp.embedding_layer.out_length

    # assert that the hidden size is correct
    assert mlp.out_layer.in_features == kwargs.get("hidden_features", 1024)
    assert mlp.blocks[0].block[0].out_features == kwargs.get("hidden_features", 1024)

    # assert that the normalization layer and activation function are correct
    assert str(mlp.blocks[0].act_fn).split(".")[-1][:-2] == kwargs.get("block_kwargs", {}).get(
        "act_fn", "ReLU"
    )
    assert str(mlp.blocks[0].norm).split(".")[-1][:-2] == kwargs.get("block_kwargs", {}).get(
        "norm", "BatchNorm1d"
    )

    # assert that the output size is correct
    assert mlp.out_layer.out_features == out_features
