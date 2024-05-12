"""
Module implementing a function to log hyperparameters.

Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/logging_utils.py 
"""

from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from .ranked_logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Args:
        object_dict (dict): A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"train_dataset"`: The torch dataset used for training.
        - `"val_dataset"`: The torch dataset used for validation.
        - `"m_preset"`: nn.Module for the preset encoder model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # general hyperparameters
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # training dataset related hyperparameters
    train_dataset = object_dict["train_dataset"]
    hparams["train_dataset/audio_fe"] = train_dataset.audio_fe_name
    hparams["train_dataset/embedding_dim"] = train_dataset.embedding_dim
    hparams["train_dataset/excl_synth_params"] = train_dataset.configs_dict["params_to_exclude"]
    hparams["train_dataset/name"] = train_dataset.name
    hparams["train_dataset/num_used_synth_params"] = train_dataset.num_used_synth_parameters
    hparams["train_dataset/midi_duration_in_sec"] = train_dataset.configs_dict["midi_duration_in_sec"]
    hparams["train_dataset/midi_note"] = train_dataset.configs_dict["midi_note"]
    hparams["train_dataset/midi_velocity"] = train_dataset.configs_dict["midi_velocity"]
    hparams["train_dataset/render_duration_in_sec"] = train_dataset.configs_dict["render_duration_in_sec"]
    hparams["train_dataset/sample_rate"] = train_dataset.configs_dict["sample_rate"]
    hparams["train_dataset/seed_offset"] = train_dataset.configs_dict["seed_offset"]
    hparams["train_dataset/size"] = len(train_dataset)
    hparams["train_dataset/synth_name"] = train_dataset.synth_name

    # validation dataset related hyperparameters
    val_dataset = object_dict["val_dataset"]
    hparams["val_dataset/name"] = val_dataset.name
    hparams["val_dataset/num_ranks"] = cfg["val_dataset"]["loader"]["num_ranks"]
    hparams["val_dataset/num_samples_per_rank"] = int(
        len(val_dataset) // cfg["val_dataset"]["loader"]["num_ranks"]
    )

    # preset encoder related hyperparameters
    preset = object_dict["m_preset"]
    for k, v in cfg["m_preset"]["cfg"].items():
        if k == "_target_":
            hparams["m_preset/name"] = v.split(".")[-1]
        elif k in ["embedding_kwargs", "block_kwargs"]:
            for kk, vv in v.items():
                hparams[f"m_preset/{k}/{kk}"] = vv
        else:
            hparams[f"m_preset/{k}"] = v
    hparams["m_preset/num_params"] = preset.num_parameters

    # solver related hyperparameters
    hparams["solver/loss"] = cfg["solver"]["loss"]["_target_"].split(".")[-1]

    hparams["solver/lr"] = cfg["solver"]["lr"]

    hparams["solver/optim/name"] = cfg["solver"]["optimizer"]["_target_"].split(".")[-1]
    for k, v in cfg["solver"]["optimizer"].items():
        if k not in ["_target_", "_partial_"]:
            hparams[f"solver/optim/{k}"] = v

    if cfg["solver"].get("scheduler"):
        hparams["solver/sched/name"] = cfg["solver"]["scheduler"]["_target_"].split(".")[-1]
        for k, v in cfg["solver"]["scheduler"].items():
            if k not in ["_target_", "_partial_"]:
                hparams[f"solver/sched/{k}"] = v

    if cfg["solver"].get("scheduler_config"):
        for k, v in cfg["solver"]["scheduler_config"].items():
            if k != "monitor":
                hparams[f"solver/sched/{k}"] = v

    # training related hyperparameters
    hparams["dataloader_train"] = cfg["train_dataset"]["loader"]
    for k, v in cfg["trainer"].items():
        if (k not in ["_target_", "default_root_dir"]) and (v is not None):
            hparams[f"trainer/{k}"] = v

    # send hparams to logger
    trainer.logger.log_hyperparams(hparams)
    # additional save the hydra config if using wandb a logger
    # (later in src/train.py since output_dir doesn't exist yet).
