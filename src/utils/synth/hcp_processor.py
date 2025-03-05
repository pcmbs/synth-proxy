"""
Module implementing the classes used to export the hand-crafted preset datasets.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import torch
from tqdm import tqdm

from utils.synth import PresetHelper, PresetRenderer

load_dotenv()  # take environment variables from .env

PLUGINS_FOLDER = Path(os.environ["PROJECT_ROOT"]) / "data" / "synths"


class ProcessHCPresets:
    """
    Class used to process a dataset of hand-crafted presets stored as dictionary in a json file.
    {preset_id: {'meta': {...}, 'parameters': {'param_name': param_val}}. Calling an instance of this
    class on such a dict will return a tuple (presets, selected_presets, removed_presets)
    where:
    - `presets` is a torch.Tensor of shape (num_presets, num_used_parameters)
    -  `selected_presets` is a dictionary (formatted as the original dict) including the presets
    that will be included in the dataset.
    Note that the presets will be re-ordered to match the presets tensor allowing easy retrieval.
    - `removed_presets` is a dictionary of removed presets, containing duplicates and silent presets.
    """

    def __init__(
        self,
        preset_helper: PresetHelper,
        render_duration_in_sec: float = 5.0,
        midi_note: Optional[int] = 60,
        midi_velocity: Optional[int] = 100,
        midi_duration_in_sec: Optional[float] = 2,
        sample_rate: int = 44_100,
        rms_range: Tuple[float, float] = (0.01, 1.0),
        path_to_plugin: Optional[str | Path] = None,
    ):
        """
        Class used to process a dataset of hand-crafted presets stored as dictionary in a json file.

        Args:
            preset_helper (PresetHelper): Instance of the PresetHelper class, containing information about
            the parameters of a given synthesizer. It need to be the same used for training.
            render_duration_in_sec (float): Rendering duration in seconds to check for silent presets.
            Should match the value used during training. (default: 4.0)
            midi_note (Optional[int]): Midi note used to check for silent presets.
            Should match the value used during training. (default: 60)
            midi_velocity (Optional[int]): MIDI note velocity used to check for silent presets.
            Should match the value used during training. (default: 110)
            midi_duration_in_sec (Optional[float]): MIDI note duration in seconds used to check for silent presets.
            Should match the value used during training. (default: 2.0)
            sample_rate (int): Sample rate of the audio to generate. (default: 44_100)
            rms_range (Tuple[float, float]): acceptable audio RMS range. (default: (0.01, 1.0))
            path_to_plugin (Optional[str]): Path to the plugin. If None (default), it will look for it in
            <project-folder>/data based on preset_helper.synth_name.

        """
        self.preset_helper = preset_helper

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

        self.renderer = PresetRenderer(
            synth_path=path_to_plugin,
            sample_rate=sample_rate,
            render_duration_in_sec=render_duration_in_sec,
            convert_to_mono=True,
            normalize_audio=False,
            fadeout_in_sec=0.1,
        )

        # set not used parameters to default values (for safety)
        self.renderer.set_parameters(
            self.preset_helper.excl_parameters_idx, self.preset_helper.excl_parameters_val
        )

    def __call__(self, presets_dict: Dict) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Process the dict of presets. Silent and duplicates will be removed based on self.preset_helper.
        This function returns the processed presets along with dictionaries of selected and removed presets.

        Args:
            presets_dict (Union[str, Path]): Dict containing the hand-crafted presets.

        Returns:
            A tuple of
            - `presets` (torch.Tensor): Tensor of shape (num_selected_presets, num_used_parameters) containing the modified presets
            - `selected_presets` (Dict): Dictionary of selected presets
            - `removed_presets` (Dict): Dictionary of removed presets
        """

        indices_to_remove = []  # to be removed at the end

        #### Convert presets from dict format to a torch tensor of shape (num_presets, num_used_parameters)
        presets = self._convert_to_tensor(presets_dict)

        #### Clip continuous numerical parameters values in range used for training
        self._clip_continuous_parameters(presets)

        #### Round values for discrete parameters to the nearest value used during training
        # Note: if some categorical values are equal to a category excluded during trainng (e.g., S&H wave)
        #       they will be replaced by the nearest category. This is not ideal and synthesizer specific
        #       replacement (e.g., S&H->SQR) should be implemented.
        self._round_discrete_parameters(presets)

        #### Get indices from silent/loud presets considering excluded and previously modified parameters
        indices_to_remove.extend(self._get_indices_to_remove_from_rms(presets))

        #### Get indices from duplicate presets considering excluded and previously modified parameters
        indices_to_remove.extend(self._get_indices_from_duplicates(presets))

        #### Remove presets according to the list of indices
        # sort the list of indices to remove and remove duplicates
        indices_to_remove = list(dict.fromkeys(sorted(indices_to_remove)))
        # Create a mask to select presets that are not in the indices_to_remove list
        mask = torch.ones(presets.shape[0], dtype=torch.bool)
        mask[indices_to_remove] = False
        indices_to_keep = mask.nonzero().squeeze(1)
        # select presets to keep based on the mask
        presets = torch.index_select(presets, 0, indices_to_keep)

        # Create a new dictionary with the selected presets (re-indexed based on the reduced dataset)
        selected_presets_dict = {i: presets_dict[j] for i, j in enumerate(indices_to_keep.tolist())}
        # Create a new dictionary with the removed presets
        removed_presets_dict = {i: presets_dict[i] for i in indices_to_remove}

        return presets, selected_presets_dict, removed_presets_dict

    def _convert_to_tensor(self, dataset: dict) -> torch.Tensor:
        """Convert the dataset to a tensor of shape (num_presets, num_used_parameters)"""
        presets = []  # List to hold the tensor presets
        # iterate over all presets in the dataset and populate a fresh tensor with the
        # value of each synth parameter used during training
        for _, preset_dict in dataset.items():
            preset = torch.zeros(self.preset_helper.num_used_parameters)
            for param_name, param_val in preset_dict["parameters"].items():
                relative_idx = self.preset_helper.relative_idx_from_name(param_name)
                if relative_idx is not None:  # i.e., was used during training
                    preset[relative_idx] = param_val

            presets.append(preset)

        return torch.stack(presets)

    def _clip_continuous_parameters(self, presets) -> None:
        """Clip continuous numerical parameters values in the range defined by the preset helper"""
        intervals = self.preset_helper.grouped_used_parameters["continuous"]
        for interval, indices in intervals.items():
            if interval != (0.0, 1.0):
                presets[:, indices] = torch.clamp(presets[:, indices], min=interval[0], max=interval[1])

    def _round_discrete_parameters(self, presets) -> None:
        """Round discrete parameters (num, cat, bin) values to the nearest value used during training"""
        params_info = self.preset_helper.grouped_used_parameters["discrete"]
        for _, type_dict in params_info.items():

            for (candidates, _), indices in type_dict.items():
                candidates = torch.tensor(candidates, dtype=torch.float32)
                # Compute the pairwise absolute differences between the synth parameter values
                # and candidate values
                # Output shape: (num_presets, num_indices, num_candidates)
                #     i.e., abs_diff between each candidate and synth params for each presets
                abs_diff = torch.abs(presets[:, indices].unsqueeze(-1) - candidates.unsqueeze(0))
                # Find the index of the nearest candidate value for each synth parameter values
                # Output shape: (num_presets, num_indices)
                nearest_index = torch.argmin(abs_diff, dim=-1)
                # replace original synth parameter values by the nearest candidate values
                presets[:, indices] = candidates[nearest_index]

    def _get_indices_to_remove_from_rms(self, presets) -> List:
        """Return a list of indices of presets with RMS outside the accepted range"""
        indices_to_remove = []
        # Set excluded parameters to their default values
        self.renderer.set_parameters(
            self.preset_helper.excl_parameters_idx, self.preset_helper.excl_parameters_val
        )
        pbar = tqdm(
            presets,
            total=len(presets),
            dynamic_ncols=True,
        )
        # render all presets and compute their RMS. Presets with RMS outside the accepted range are removed
        for i, params in enumerate(pbar):
            self.renderer.set_parameters(self.preset_helper.used_parameters_absolute_idx, params)
            self.renderer.set_midi_parameters(self.midi_note, self.midi_velocity, self.midi_duration_in_sec)
            audio_out = torch.from_numpy(self.renderer.render_note())
            rms_out = torch.sqrt(torch.mean(torch.square(audio_out))).item()
            if not self.rms_range[0] < rms_out < self.rms_range[1]:
                indices_to_remove.append(i)
        return indices_to_remove

    def _get_indices_from_duplicates(self, presets) -> List:
        """Return a list of indices of presets that appear more than once in the input tensor"""
        indices_to_remove = []
        # return a tensor of shape (num_presets,) indicating where each preset can be found
        # in the tensor of unique presets (not used here)
        _, inverse_indices = torch.unique(presets, return_inverse=True, dim=0)
        # return a tensor of shape (num_unique_presets,) indicating how many times each unique preset
        # can be found in the input presets
        _, counts = torch.unique(inverse_indices, return_counts=True)
        # Get indices of unique presets that appear more than once in the input tensor
        duplicate_indices = (counts > 1).nonzero().flatten().tolist()
        # Interate over the indices of unique presets that appear more than once,
        # For each of them, retrieve the indices of the duplicate presets in the full presets tensor
        # and add them (excluding the first one) to the list of removed presets.
        for idx in duplicate_indices:
            current_duplicates = (inverse_indices == idx).nonzero().flatten().tolist()[1:]
            indices_to_remove.extend(current_duplicates)
        return indices_to_remove


if __name__ == "__main__":
    print("")
