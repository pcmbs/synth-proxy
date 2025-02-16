# pylint: disable=W0212,W1203
"""
Helper functions for instantiating lightning callbacks from DictConfig objects.

Source: https://github.com/ashleve/lightning-hydra-template
"""

from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.info("No callback configs found. Skipping...")
        return None

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def check_val_dataset(train_dataset: Dataset, val_dataset: Dataset) -> None:
    assert (
        val_dataset.audio_fe_name == train_dataset.audio_fe_name
    ), f"the audio model used for training and validation should be the same: {train_dataset.audio_fe_name} != {val_dataset.audio_fe_name}"
    assert (
        val_dataset.synth_name == train_dataset.synth_name
    ), f"the synthesizer used for training and validation should be the same: {train_dataset.synth_name} != {val_dataset.synth_name}"
    assert (
        val_dataset.configs_dict["params_to_exclude"] == train_dataset.configs_dict["params_to_exclude"]
    ), "the params_to_exclude used for training and validation should be the same."
    assert val_dataset.num_used_synth_parameters == train_dataset.num_used_synth_parameters
    for attr in ["render_duration_in_sec", "midi_note", "midi_velocity", "midi_duration_in_sec"]:
        assert (
            val_dataset.configs_dict[attr] == train_dataset.configs_dict[attr]
        ), f"the {attr} used for training and validation should be the same: {train_dataset.configs_dict[attr]} != {val_dataset.configs_dict[attr]}"
