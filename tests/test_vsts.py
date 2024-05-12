import os
import sys
import numpy as np
import pytest
from dotenv import load_dotenv
from utils.synth import PresetRenderer

load_dotenv()  # take environment variables from .env
TALNM_PATH = os.environ["TALNM_PATH"]
DIVA_PATH = os.environ["DIVA_PATH"]
DEXED_PATH = os.environ["DEXED_PATH"]


@pytest.fixture
def talnm_engine():
    renderer = PresetRenderer(
        sample_rate=44_100,
        convert_to_mono=False,
        render_duration_in_sec=4,
        synth_path=TALNM_PATH,
    )
    renderer.set_midi_parameters(60, 100, 4.0)
    return renderer


@pytest.fixture
def diva_engine():
    renderer = PresetRenderer(
        sample_rate=44_100,
        convert_to_mono=False,
        render_duration_in_sec=4,
        synth_path=DIVA_PATH,
    )
    renderer.set_midi_parameters(60, 100, 4.0)
    return renderer


@pytest.mark.skipif(sys.platform != "win32", reason="Test only applicable on Windows")
@pytest.fixture
def dexed_engine():
    renderer = PresetRenderer(
        sample_rate=44_100,
        convert_to_mono=False,
        render_duration_in_sec=4,
        synth_path=DEXED_PATH,
    )
    renderer.set_midi_parameters(60, 100, 4.0)
    return renderer


@pytest.mark.parametrize("num_presets", [20])
def test_tal_assign(talnm_engine, num_presets):
    rnd_presets = np.random.rand(num_presets, 86)
    diff_array = np.empty_like(rnd_presets)

    for i, p in enumerate(rnd_presets):
        talnm_engine.set_parameters(np.arange(86), p)
        diff_array[i] = np.array(
            [np.abs(p[i] - talnm_engine.synth.get_parameter(i)).round(5) for i in range(86)]
        )

    assert not np.any(diff_array)


@pytest.mark.parametrize("num_presets", [20])
def test_tal_render(talnm_engine, num_presets):
    std_list = []
    rms_list = []

    for _ in range(num_presets * 10):
        rnd_preset = np.random.rand(86)
        talnm_engine.set_parameters(np.arange(86), rnd_preset)
        out = talnm_engine.render_note()
        rms = np.sqrt(np.mean(out**2))
        if rms > 0.01:
            rms_list.append(rms)
            std_list.append(out.std())

        if len(rms_list) == 10:
            break

    _, val1 = np.unique(std_list, return_counts=True)
    _, val2 = np.unique(rms_list, return_counts=True)
    assert np.all(val1 == 1)
    assert np.all(val2 == 1)


@pytest.mark.parametrize("num_presets", [20])
def test_diva_assign(diva_engine, num_presets):
    rnd_presets = np.random.rand(num_presets, 155)
    diff_array = np.empty_like(rnd_presets)

    for i, p in enumerate(rnd_presets):
        diva_engine.set_parameters(np.arange(155), p)
        diff_array[i] = np.array(
            [np.abs(p[i] - diva_engine.synth.get_parameter(i)).round(5) for i in range(155)]
        )

    assert not np.any(diff_array)


@pytest.mark.parametrize("num_presets", [20])
def test_diva_render(diva_engine, num_presets):
    std_list = []
    rms_list = []

    for _ in range(num_presets * 10):
        rnd_preset = np.random.rand(155)
        diva_engine.set_parameters(np.arange(155), rnd_preset)
        out = diva_engine.render_note()
        rms = np.sqrt(np.mean(out**2))
        if rms > 0.01:
            rms_list.append(rms)
            std_list.append(out.std())

        if len(rms_list) == 10:
            break

    _, val1 = np.unique(std_list, return_counts=True)
    _, val2 = np.unique(rms_list, return_counts=True)
    assert np.all(val1 == 1)
    assert np.all(val2 == 1)


@pytest.mark.skipif(sys.platform != "win32", reason="Test only applicable on Windows")
@pytest.mark.parametrize("num_presets", [20])
def test_dexed_assign(dexed_engine, num_presets):
    rnd_presets = np.random.rand(num_presets, 155)
    diff_array = np.empty_like(rnd_presets)

    for i, p in enumerate(rnd_presets):
        dexed_engine.set_parameters(np.arange(155), p)
        diff_array[i] = np.array(
            [np.abs(p[i] - dexed_engine.synth.get_parameter(i)).round(5) for i in range(155)]
        )

    assert not np.any(diff_array)


@pytest.mark.skipif(sys.platform != "win32", reason="Test only applicable on Windows")
@pytest.mark.parametrize("num_presets", [20])
def test_dexed_render(dexed_engine, num_presets):
    std_list = []
    rms_list = []

    for _ in range(num_presets * 10):
        rnd_preset = np.random.rand(155)
        dexed_engine.set_parameters(np.arange(155), rnd_preset)
        out = dexed_engine.render_note()
        rms = np.sqrt(np.mean(out**2))
        if rms > 0.01:
            rms_list.append(rms)
            std_list.append(out.std())

        if len(rms_list) == 10:
            break

    _, val1 = np.unique(std_list, return_counts=True)
    _, val2 = np.unique(rms_list, return_counts=True)
    assert np.all(val1 == 1)
    assert np.all(val2 == 1)
