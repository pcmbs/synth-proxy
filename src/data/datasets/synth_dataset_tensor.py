import os
from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from utils.synth.preset_helper import PresetHelper
from utils.synth.preset_renderer import PresetRenderer


class SynthDatasetTensor(Dataset):
    """
    Dataset class used to render presets given a tensor of shape (num_presets, num_parameters).
    The dataset returns synth parameters and audio rendering of each preset.
    It is mostly used to export dataset of hand-crafted presets but should not be used for training,
    due to the slow rendering time - see SynthDatasetPkl instead.
    """

    def __init__(
        self,
        presets: torch.Tensor,
        preset_helper: PresetHelper,
        sample_rate: int = 44_100,
        render_duration_in_sec: float = 5.0,
        rms_range: Tuple[float, float] = (0.01, 1.0),
        midi_note: Optional[int] = 60,
        midi_velocity: Optional[int] = 100,
        midi_duration_in_sec: Optional[float] = 2,
        path_to_plugin: Optional[str | Path] = None,
    ):
        self.presets = presets
        self.preset_helper = preset_helper

        assert self.presets.shape[1] == self.preset_helper.num_used_parameters

        self.midi_note = midi_note
        self.midi_velocity = midi_velocity
        self.midi_duration_in_sec = midi_duration_in_sec
        self.render_duration_in_sec = render_duration_in_sec

        self.sample_rate = sample_rate
        self.rms_range = rms_range

        if path_to_plugin is None:
            if preset_helper.synth_name == "talnm":
                path_to_plugin = os.environ["TALNM_PATH"]
            elif preset_helper.synth_name == "dexed":
                path_to_plugin = os.environ["DEXED_PATH"]
            elif preset_helper.synth_name == "diva":
                path_to_plugin = os.environ["DIVA_PATH"]
            else:
                raise NotImplementedError()

        self.path_to_plugin = str(path_to_plugin)

        # instantiate renderer during first iteration to avoid possible pickle errors
        self.renderer = None

    def __len__(self):
        return self.presets.shape[0]

    def __getitem__(self, idx: int):
        # instantiate renderer during first iteration to avoid num_workers pickle error on windows
        if self.renderer is None:
            self._instantiate_renderer()

        synth_parameters = self.presets[idx]
        # set synth parameters
        self.renderer.set_parameters(self.preset_helper.used_parameters_absolute_idx, synth_parameters)
        # set midi parameters
        self.renderer.set_midi_parameters(self.midi_note, self.midi_velocity, self.midi_duration_in_sec)
        # render audio
        audio_out = torch.from_numpy(self.renderer.render_note()).squeeze(0)

        # Replace categorical parameter values by the category index
        for (values, _), indices in self.preset_helper.grouped_used_parameters["discrete"]["cat"].items():
            value_to_index = {val: idx for idx, val in enumerate(values)}
            synth_parameters[indices] = torch.tensor(
                [value_to_index[round(synth_parameters[i].item(), 3)] for i in indices], dtype=torch.float32
            )

        return synth_parameters, audio_out

    def _instantiate_renderer(self):
        # instantiate renderer during first iteration to avoid num_workers pickle error on windows
        self.renderer = PresetRenderer(
            synth_path=self.path_to_plugin,
            sample_rate=self.sample_rate,
            render_duration_in_sec=self.render_duration_in_sec,
            convert_to_mono=True,
            normalize_audio=False,
            fadeout_in_sec=0.1,
        )

        # set not used parameters to default values (for safety)
        self.renderer.set_parameters(
            self.preset_helper.excl_parameters_idx, self.preset_helper.excl_parameters_val
        )
