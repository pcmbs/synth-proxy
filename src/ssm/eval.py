# pylint: disable=W0212:protected-access
# pylint: disable=W1203:logging-fstring-interpolation
"""
Evaluation script.
Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py

See configs/ssm/train for more details.
"""
import os
from pathlib import Path
import pickle
from typing import Any, Dict

import hydra
import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig
import wandb

from data.datasets import SynthDatasetPkl, NSynthDataset
from ssm.estimator_network import EstimatorNet
from ssm.litm_ssm import SSMLitModule
from utils.logging import RankedLogger
from utils.synth.preset_helper import PresetHelper

# logger for this file
log = RankedLogger(__name__, rank_zero_only=True)


def evaluate(cfg: DictConfig) -> Dict[str, Any]:
    """
    Args:
        cfg (DictConfig): A DictConfig configuration composed by Hydra.

    Returns:
        A tuple with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    gen = torch.Generator().manual_seed(cfg.seed)

    log.info(f"Instantiating test Dataset: {cfg.dataset_name}")
    # instantiate test Dataset & DataLoader
    dataset = SynthDatasetPkl(path_to_dataset=cfg.dataset.path_to_dataset, has_mel=True, split="test")
    id_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, generator=gen)
    if cfg.ood_eval.run:
        log.info("Instantiating Out-Of-Domain NSynth dataset...")

        nsynth_dataset = NSynthDataset(
            root=Path(cfg.paths.root_dir) / "data" / "datasets" / "eval",
            subset=cfg.ood_eval.subset,
            return_mel=True,
            mel_kwargs={**dataset.mel_cfg, **{"sr": dataset.configs_dict["sample_rate"]}},
            mel_stats=torch.load(dataset.path_to_dataset / "stats_train.pkl"),
            mel_norm=dataset.mel_norm,
            audio_length=dataset.configs_dict["render_duration_in_sec"],
            pitch=cfg.ood_eval.pitch,
        )
        nsynth_loader = DataLoader(
            nsynth_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=0, generator=gen
        )
        loader = [id_loader, nsynth_loader]
    else:
        loader = id_loader

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
    if cfg.tag == "loss_p":
        synth_proxy = None
    else:
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
        test_batch_to_export=cfg.batch_to_export,
    )

    logger: Logger | None = None
    if cfg.get("wandb"):
        log.info("Instantiating wandb logger...")
        logger = WandbLogger(**cfg.wandb)

    log.info("Instantiating Lightning Trainer...")
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
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
                "synth_proxy_ckpt": cfg.path_to_ckpt.split("/")[-1],
                "estimator_ckpt": cfg.train_ckpt,
            },
            "losses": {
                "lw_cat": cfg.lw_cat,
                "loss_sch": cfg.loss_sch.name,
                "lw_a": cfg.lw_a,
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

    ckpt_path = cfg.get("path_to_train_ckpt")
    log.info(f"Starting evaluation from checkpoint: {ckpt_path}.")
    log.info(f"Audio files will be exported to {cfg.trainer.default_root_dir}.")
    if cfg.ood_eval.run:
        log.info("Running evaluation on both in-domain and out-of-domain datasets...")
    else:
        log.info("Running evaluation on in-domain dataset...")
    trainer.test(
        model=model,
        dataloaders=loader,
        ckpt_path=ckpt_path,
    )
    metrics_dict = trainer.callback_metrics

    if logger:
        wandb.finish()

    return metrics_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs/ssm/train", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.
    """
    # train the model
    metrics_dict, _ = evaluate(cfg)

    log.info(f"Metrics: {metrics_dict}")

    # dump metrics to pickle
    with open(os.path.join(cfg.paths.output_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics_dict, f)


if __name__ == "__main__":
    import sys

    args = ["src/ssm/eval.py", "ckpt=dexed_loss_p", "trainer.accelerator=cpu", "~wandb", "hydra.run.dir=."]

    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        sys.argv = args

    main()  # pylint: disable=no-value-for-parameter
