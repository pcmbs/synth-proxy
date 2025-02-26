# pylint: disable=W0212:protected-access
# pylint: disable=W1203:logging-fstring-interpolation
"""
Training script.
Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py

See configs/ssm/train for more details.
"""
import os
from pathlib import Path
from typing import Any, Dict, List

import hydra
import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig
import wandb

from data.datasets import SynthDatasetPkl
from ssm.estimator_network import EstimatorNet
from ssm.litm_ssm import SSMLitModule
from utils.logging import RankedLogger
from utils.instantiators import instantiate_callbacks
from utils.synth.preset_helper import PresetHelper

# logger for this file
log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig) -> Dict[str, Any]:
    """
    Args:
        cfg (DictConfig): A DictConfig configuration composed by Hydra.

    Returns:
        A tuple with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating training Dataset: {cfg.dataset_name}")
    # instantiate train Dataset & DataLoader
    dataset = SynthDatasetPkl(**cfg.dataset)
    dataset_train, dataset_val = random_split(dataset, cfg.train_val_split)
    loader_train = DataLoader(
        dataset_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    loader_val = DataLoader(
        dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    log.info(f"Instantiating Preset Helper for synth {dataset.synth_name} and excluded params:")
    log.info(f"{dataset.configs_dict['params_to_exclude']}")
    preset_helper = PresetHelper(
        synth_name=dataset.synth_name,
        parameters_to_exclude=dataset.configs_dict["params_to_exclude"],
    )

    # instantiating estimator network
    log.info("Instantiating Estimator Network...")
    estimator = EstimatorNet(preset_helper)

    # instantiate synth_proxy
    log.info("Instantiating Synth Proxy...")
    synth_proxy: nn.Module = hydra.utils.instantiate(
        cfg.synth_proxy, out_features=dataset.embedding_dim, preset_helper=preset_helper
    )
    # get synth proxy state dict from lightning ckpt (need to rename keys to remove upper level prefix)
    ckpt = torch.load(cfg.path_to_ckpt)
    synth_proxy.load_state_dict({k.split(".", 1)[-1]: v for k, v in ckpt["state_dict"].items()})
    synth_proxy.eval()

    log.info("Instantiating SSM Lightning Module...")
    model = SSMLitModule(
        preset_helper=preset_helper,
        estimator=estimator,
        synth_proxy=synth_proxy,
        opt_cfg=cfg.optimizer,
        loss_sch_cfg=cfg.loss_sch,
        compute_a_metrics=cfg.compute_a_metrics,
        label_smoothing=cfg.label_smoothing,
        lw_cat=cfg.lw_cat,
        wandb_watch_args=cfg.wandb_watch_args,
    )

    log.info("Instantiating Callbacks...")
    callbacks: List[Callback] | None = instantiate_callbacks(cfg.get("callbacks"))

    logger: Logger | None = None
    if cfg.get("wandb"):
        log.info("Instantiating wandb logger...")
        logger = WandbLogger(**cfg.wandb)

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

    log.info("Instantiating Lightning Trainer...")
    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        plugins=slurm_plugin,
    )

    if logger:
        log.info("Logging hyperparameters...")
        object_dict = {
            "data": {
                "dataset": cfg.dataset_name,
                "train_val_split": cfg.train_val_split,
                "batch_size": cfg.batch_size,
            },
            "misc": {
                "seed": cfg.seed,
                "synth": cfg.synth,
            },
            "model": {
                "synth_proxy": cfg.synth_proxy._target_,
                "ckpt_name": cfg.path_to_ckpt.split("/")[-1],
            },
            "losses": {
                "lw_cat": cfg.lw_cat,
                "loss_sch": cfg.loss_sch.name,
                "start_ratio": cfg.start_ratio,
            },
            "opt": {
                "lr": cfg.optimizer.optimizer_kwargs.lr,
                "betas": cfg.optimizer.optimizer_kwargs.betas,
                "eps": cfg.optimizer.optimizer_kwargs.eps,
                "weight_decay": cfg.optimizer.optimizer_kwargs.weight_decay,
                "num_warmup_steps": cfg.optimizer.num_warmup_steps,
                "sch_factor": cfg.optimizer.scheduler_kwargs.factor,
                "sch_patience": cfg.optimizer.scheduler_kwargs.patience,
            },
            "trainer": {
                "max_epochs": cfg.trainer.max_epochs,
            },
        }
        trainer.logger.log_hyperparams(object_dict)

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
        train_dataloaders=loader_train,
        val_dataloaders=loader_val,
        ckpt_path=ckpt_path,
    )

    # get metrics available to callbacks. This includes metrics logged via log().
    metrics_dict = trainer.callback_metrics

    if logger:
        wandb.finish()

    return metrics_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs/ssm/train", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.
    """
    # train the model
    metrics_dict, _ = train(cfg)

    log.info(f"Metrics: {metrics_dict}")


if __name__ == "__main__":
    import sys

    args = ["src/ssm/train.py", "settings=debug", "loss_sch=switch_a_p"]

    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        sys.argv = args

    main()  # pylint: disable=no-value-for-parameter
