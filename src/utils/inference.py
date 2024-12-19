"""
Utility function used for inference
"""

from pathlib import Path
import os

import torch
from omegaconf import OmegaConf

from utils.synth import PresetHelper
from models.preset import model_zoo

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
CKPT_DIR = PROJECT_ROOT / "checkpoints"


def get_preset_helper(synth: str) -> PresetHelper:
    """Get the preset helper object for a given synthesizer."""
    excluded_params = OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "configs" / "inference" / "synth" / f"{synth}.yaml")
    )
    return PresetHelper(synth, excluded_params)


def get_synth_proxy(
    arch: str, synth: str, preset_helper: PresetHelper, ckpt_path: str = None
) -> torch.nn.Module:
    """
    Load a pretrained synthesizer proxy.

    Args
        arch (str): Name of the proxy architecture.
        synth (str): Synthesizer name.
        preset_helper (PresetHelper): Preset helper object used to instantiate the proxy.
        ckpt_path (str, optional): Path to the checkpoint file (if custom path is required).
    """

    # Load the model configs
    model_cfg = OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "configs" / "inference" / "model" / f"{arch}.yaml")
    )

    # Load checkpoint (default location or specified path)
    if ckpt_path is None:
        ckpt_files = list(CKPT_DIR.glob(f"{synth}_{arch}_*.ckpt"))
        if not ckpt_files or len(ckpt_files) > 1:
            raise ValueError(
                f"No or multiple checkpoints found for synth '{synth}' and model '{arch}'."
                f"Please specify a checkpoint path."
            )
        ckpt_path = ckpt_files[0]

    ckpt = torch.load(ckpt_path)

    # Instantiate the preset encoder, get the number of output features from the checkpoint's projection head
    synth_proxy = getattr(model_zoo, arch)(
        out_features=ckpt["state_dict"]["preset_encoder.head.2.bias"].shape[0],
        preset_helper=preset_helper,
        **model_cfg,
    )

    # Load state dict - remove the "preset_encoder." prefix from the state dict's keys
    # since we load the nn.Module directly and not the LightningModule used during training
    synth_proxy.load_state_dict({k.replace("preset_encoder.", ""): v for k, v in ckpt["state_dict"].items()})
    synth_proxy.eval()

    return synth_proxy


def clip_continuous_params(presets: torch.Tensor, preset_helper: PresetHelper) -> torch.Tensor:
    """Clip continuous numerical parameters values in the range defined by the preset helper"""
    intervals = preset_helper.grouped_used_parameters["continuous"]
    for interval, indices in intervals.items():
        if interval != (0.0, 1.0):
            presets[:, indices] = torch.clamp(presets[:, indices], min=interval[0], max=interval[1])

    return presets


def round_discrete_params(presets: torch.Tensor, preset_helper: PresetHelper) -> torch.Tensor:
    """
    Round discrete parameters (num, cat, bin) values to the nearest value used during training.

    Note: if some categorical values are equal to a category excluded during trainng (e.g., S&H wave)
    they will be replaced by the nearest category. This is not ideal and synthesizer specific
    replacement (e.g., S&H->SQR) should be implemented.
    """
    params_info = preset_helper.grouped_used_parameters["discrete"]
    for type_info, type_dict in params_info.items():

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
            # for discrete numerical and bool parameters, and by the category index for categorical parameters
            if type_info == "cat":
                presets[:, indices] = nearest_index.to(dtype=torch.float32)
            else:
                presets[:, indices] = candidates[nearest_index]
    return presets
