import pytest
import torch
from torch.utils.data import DataLoader
from data.datasets.synth_dataset import SynthDataset
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


@pytest.mark.parametrize("seed_offset", [0, 5423, 10254])
def test_noisemaker_dataset_rnd(tal_preset_helper, seed_offset):
    """
    Test that each worker generate different random data.
    Additionally test that the midi parameters are not randomly sampled if a value is passed
    """

    NUM_WORKERS = 8

    dataset = SynthDataset(
        preset_helper=tal_preset_helper,
        dataset_size=NUM_WORKERS,
        seed_offset=seed_offset,
    )

    train_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=NUM_WORKERS)

    params_concat = torch.empty(NUM_WORKERS, dataset.num_used_parameters)

    for i, (params, _, _) in enumerate(train_loader):
        params_concat[i] = params[0]

    assert params_concat.std(0).count_nonzero() == dataset.num_used_parameters


@pytest.mark.parametrize("sample_rate,render_duration_in_sec", [(44_100, 4.0), (48_000, 3.0)])
def test_loader_output_shape(tal_preset_helper, sample_rate, render_duration_in_sec):
    dataset = SynthDataset(
        preset_helper=tal_preset_helper,
        dataset_size=32,
        seed_offset=5423,
        sample_rate=sample_rate,
        render_duration_in_sec=render_duration_in_sec,
    )

    train_loader = DataLoader(dataset=dataset, batch_size=32)

    params, audio, _ = next(iter(train_loader))

    assert params.shape == (32, dataset.num_used_parameters)
    assert audio.shape == (32, 1, int(sample_rate * render_duration_in_sec))
