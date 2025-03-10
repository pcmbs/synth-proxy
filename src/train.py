# pylint: disable=W0212:protected-access
# pylint: disable=W1203:logging-fstring-interpolation
"""
Training script.
Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py

See configs/train for more details.
"""
import os
from pathlib import Path
from typing import Any, Dict, List

import hydra
import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig
import wandb

from data.datasets import SynthDatasetPkl
from utils.logging import RankedLogger, log_hyperparameters
from utils.instantiators import instantiate_callbacks, check_val_dataset
from utils.synth.preset_helper import PresetHelper

# logger for this file
log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig) -> Dict[str, Any]:
    """
    Trains the mmodel

    Args:
        cfg (DictConfig): A DictConfig configuration composed by Hydra.

    Returns:
        A tuple with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    log.info(f"Instantiating training Dataset: {cfg.train_dataset.path}")
    train_dataset = SynthDatasetPkl(cfg.train_dataset.path)
    if cfg.train_dataset.dataset_size < 1:
        log.info(f"Using {cfg.train_dataset.dataset_size * 100}% of the dataset")
        dataset_size = int(cfg.train_dataset.dataset_size * len(train_dataset))
        rnd_indices = torch.multinomial(torch.ones(len(train_dataset)), dataset_size, replacement=False)
        train_dataset = Subset(train_dataset, rnd_indices)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train_dataset.loader.batch_size,
        shuffle=True,  # always shuffle the training dataset
        num_workers=cfg.train_dataset.loader.num_workers,
        drop_last=False,
    )

    log.info(f"Instantiating validation Dataset: {cfg.val_dataset.path}")
    val_dataset = SynthDatasetPkl(cfg.val_dataset.path)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.val_dataset.loader.num_ranks,
        shuffle=False,  # never shuffle the validation dataset
        num_workers=cfg.val_dataset.loader.num_workers,
        drop_last=True,  # drop last required for MRR
    )
    check_val_dataset(train_dataset, val_dataset)

    log.info(f"Instantiating Preset Helper for synth {train_dataset.synth_name} and excluded params:")
    log.info(f"{train_dataset.configs_dict['params_to_exclude']}")
    preset_helper = PresetHelper(
        synth_name=train_dataset.synth_name,
        parameters_to_exclude=train_dataset.configs_dict["params_to_exclude"],
    )

    log.info(f"Instantiating Preset Encoder <{cfg.m_preset.cfg._target_}>")
    m_preset: nn.Module = hydra.utils.instantiate(
        cfg.m_preset.cfg, out_features=train_dataset.embedding_dim, preset_helper=preset_helper
    )

    log.info(f"Instantiating Lightning Module <{cfg.solver._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.solver, preset_encoder=m_preset)

    log.info("Instantiating Callbacks...")
    callbacks: List[Callback] | None = instantiate_callbacks(cfg.get("callbacks"))

    logger: Logger | None = None
    if not cfg.get("logger"):
        log.warning("No logger configs found! Skipping...")
    elif cfg.logger.get("wandb"):
        log.info(f"Instantiating logger <{cfg.logger.wandb._target_}>")
        logger = hydra.utils.instantiate(cfg.logger.wandb)
    else:
        log.info(f"Logger {cfg.logger._target_} not supported! Skipping...")

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        log.info(
            f"Detected SLURM environment with SLURM job id {slurm_job_id}, activating Lightning SLURM plugin..."
        )
        slurm_plugin = SLURMEnvironment(auto_requeue=False)  # auto-requeue managed independently
        slurm_job_id = int(slurm_job_id)
    else:
        log.info("No SLURM environment detected, deactivating Lightning SLURM plugin...")
        slurm_plugin = None

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        plugins=slurm_plugin,
    )

    object_dict = {
        "cfg": cfg,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "m_preset": m_preset,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters...")
        log_hyperparameters(object_dict)

    # save hydra run dir and wandb run id to resume training on SLURM environment if requested.
    if slurm_job_id and logger and cfg.slurm.get("auto_requeue"):
        log.info(
            f"Job auto-requeue is enabled, writing environment variables to workspace/logs/{slurm_job_id}.sh to resume training..."
        )
        hydra_run_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        wandb_run_id = logger.version

        log.info(f"HYDRA_RUN_DIR={hydra_run_dir}")
        log.info(f"WANDB_RUN_ID={wandb_run_id}")

        env_file_path = Path(os.environ["PROJECT_ROOT"]) / "logs" / f"{slurm_job_id}.sh"
        with open(env_file_path, "w", encoding="utf-8") as env_file:
            env_file.write(f"export HYDRA_RUN_DIR={hydra_run_dir}\n")
            env_file.write(f"export WANDB_RUN_ID={wandb_run_id}\n")

    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        log.info(f"Resuming training from {ckpt_path}...")
    else:
        log.info("Starting training...")

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    # get metrics available to callbacks. This includes metrics logged via log().
    metrics_dict = trainer.callback_metrics

    if logger:
        if cfg.logger.get("wandb"):
            # additional save the hydra config
            # under <project_name>/Runs/<run_id>/Files/.hydra if using wandb a logger
            wandb.save(
                glob_str=os.path.join(cfg["paths"].get("output_dir"), ".hydra", "*.yaml"),
                base_path=cfg["paths"].get("output_dir"),
            )
            wandb.finish()

    return metrics_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs/train", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.
    """
    # train the model
    metrics_dict, _ = train(cfg)

    log.info(f"Metrics: {metrics_dict}")


if __name__ == "__main__":
    import sys

    # args = ["src/train.py", "experiment=debug_ckpt", "logger=none", "callbacks=no_lr_monitor"]
    # args = ["src/train.py", "experiment=diva_tfm_mn20_mel_mse", "debug=default"]
    args = ["src/train.py", "experiment=diva_tfm_mn20_mel_mse", "trainer.accelerator=cpu"]

    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        sys.argv = args

    main()  # pylint: disable=no-value-for-parameter
