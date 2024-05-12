# pylint: disable=E1102
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.datasets import SynthDataset
from models.preset.embedding_layers import RawParameters, OneHotEncoding, PresetTokenizer
from utils.synth import PresetHelper

NUM_SAMPLES = 32

# TODO: SynthDatasetPkl instead of SynthDataset


@pytest.fixture
def talnm_dataset():
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

    p_helper = PresetHelper("talnm", parameters_to_exclude)

    dataset = SynthDataset(p_helper, NUM_SAMPLES, seed_offset=5423)

    return dataset


@pytest.fixture
def diva_dataset():
    """Return a PresetHelper instance for the TAL-NoiseMaker synthesizer"""
    parameters_to_exclude = (
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

    p_helper = PresetHelper("diva", parameters_to_exclude)

    dataset = SynthDataset(p_helper, NUM_SAMPLES, seed_offset=5423)

    return dataset


def test_raw_params_emb_layer_talnm(talnm_dataset):
    """
    Test that the RawParameters embedding layer returns the correct raw parameter values
    given the index of a category from a categorical synth parameters
    """

    loader = DataLoader(talnm_dataset, batch_size=NUM_SAMPLES, shuffle=False)

    emb_layer = RawParameters(preset_helper=talnm_dataset.preset_helper)

    params, _, _ = next(iter(loader))

    raw_params_emb = emb_layer(params.clone())
    for (cat_values, _), indices in talnm_dataset.preset_helper.grouped_used_parameters["discrete"][
        "cat"
    ].items():
        for i in indices:
            for sample, emb in zip(params, raw_params_emb):
                assert cat_values[int(sample[i])] == emb[i]


def test_raw_params_emb_layer_diva(diva_dataset):
    """
    Test that the RawParameters embedding layer returns the correct raw parameter values
    given the index of a category from a categorical synth parameters
    """

    loader = DataLoader(diva_dataset, batch_size=NUM_SAMPLES, shuffle=False)

    emb_layer = RawParameters(preset_helper=diva_dataset.preset_helper)

    params, _, _ = next(iter(loader))

    raw_params_emb = emb_layer(params.clone())

    for (cat_values, _), indices in diva_dataset.preset_helper.grouped_used_parameters["discrete"][
        "cat"
    ].items():
        for i in indices:
            for sample, emb in zip(params, raw_params_emb):
                assert cat_values[int(sample[i])] == emb[i]


def test_raw_params_emb_layer_out_shape(talnm_dataset):
    """
    Test that the RawParameters embedding layer returns the correct shape
    """

    loader = DataLoader(talnm_dataset, batch_size=NUM_SAMPLES, shuffle=False)

    emb_layer = RawParameters(preset_helper=talnm_dataset.preset_helper)

    params, _, _ = next(iter(loader))

    raw_params_emb = emb_layer(params.clone())

    assert len(raw_params_emb) == len(params)

    for sample, emb in zip(params, raw_params_emb):
        assert emb.shape == sample.shape
        assert emb.shape[0] == talnm_dataset.num_used_parameters


def test_onehot_params_emb_layer_talnm(talnm_dataset):
    """
    Test that the OneHotEncoding correctly embed the categorical parameters
    """

    loader = DataLoader(talnm_dataset, batch_size=NUM_SAMPLES, shuffle=False)

    emb_layer = OneHotEncoding(preset_helper=talnm_dataset.preset_helper)

    params, _, _ = next(iter(loader))

    onehot_params_emb = emb_layer(params.clone())

    num_noncat = emb_layer.num_noncat
    cat_offsets, _ = emb_layer._compute_cat_infos(talnm_dataset.preset_helper)

    for i, idx in enumerate(talnm_dataset.preset_helper.used_cat_parameters_idx):
        card = talnm_dataset.preset_helper.used_parameters[idx].cardinality
        for sample, emb in zip(params, onehot_params_emb):
            assert torch.all(
                F.one_hot(sample[idx].to(torch.long), card)
                == emb[num_noncat + cat_offsets[i] : num_noncat + cat_offsets[i] + card]
            )


def test_onehot_params_emb_layer_diva(diva_dataset):
    """
    Test that the OneHotEncoding correctly embed the categorical parameters
    """
    loader = DataLoader(diva_dataset, batch_size=NUM_SAMPLES, shuffle=False)

    emb_layer = OneHotEncoding(preset_helper=diva_dataset.preset_helper)

    params, _, _ = next(iter(loader))

    onehot_params_emb = emb_layer(params.clone())

    num_noncat = emb_layer.num_noncat
    cat_offsets, _ = emb_layer._compute_cat_infos(diva_dataset.preset_helper)

    for i, idx in enumerate(diva_dataset.preset_helper.used_cat_parameters_idx):
        card = diva_dataset.preset_helper.used_parameters[idx].cardinality
        for sample, emb in zip(params, onehot_params_emb):
            assert torch.all(
                F.one_hot(sample[idx].to(torch.long), card)
                == emb[num_noncat + cat_offsets[i] : num_noncat + cat_offsets[i] + card]
            )


def test_tokenizer_talnm(talnm_dataset):
    EMB_DIM = 128
    # one with cls, one without. Both without PE
    ft0 = PresetTokenizer(talnm_dataset.preset_helper, token_dim=EMB_DIM, has_cls=False, pe_type=None)
    ft1 = PresetTokenizer(talnm_dataset.preset_helper, token_dim=EMB_DIM, has_cls=True, pe_type=None)

    with torch.no_grad():
        ft1.noncat_tokenizer.copy_(ft0.noncat_tokenizer)
        ft1.cat_tokenizer.weight.copy_(ft0.cat_tokenizer.weight)

    loader = DataLoader(talnm_dataset, batch_size=16, shuffle=False)

    for params, _, _ in loader:
        tok0 = ft0(params)
        tok1 = ft1(params)
        break

    assert torch.unique(tok1[:, 0].mean(1)).shape[0] == 1
    assert torch.equal(tok0, tok1[:, 1:])


def test_tokenizer_diva(diva_dataset):
    EMB_DIM = 128
    # one with cls, one without. Both without PE
    ft0 = PresetTokenizer(diva_dataset.preset_helper, token_dim=EMB_DIM, has_cls=False, pe_type=None)
    ft1 = PresetTokenizer(diva_dataset.preset_helper, token_dim=EMB_DIM, has_cls=True, pe_type=None)

    with torch.no_grad():
        ft1.noncat_tokenizer.copy_(ft0.noncat_tokenizer)
        ft1.cat_tokenizer.weight.copy_(ft0.cat_tokenizer.weight)

    loader = DataLoader(diva_dataset, batch_size=16, shuffle=False)

    for params, _, _ in loader:
        tok0 = ft0(params)
        tok1 = ft1(params)
        break

    assert torch.unique(tok1[:, 0].mean(1)).shape[0] == 1
    assert torch.equal(tok0, tok1[:, 1:])
