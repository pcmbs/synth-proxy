# pylint: disable=W0212,C0207
"""
Script to run HPO for the estimator network of the Synthesizer Sound Matching evaluation task

Usage example:
    python src/ssm/hpo_estimator_net.py synth=<synth> tag=<tag>

The results are exported to
`<project-root>/logs/optuna/ssm_<synth>_<dataset_name>_bs<batch_size>_<tag>` 

Remarks:
 - If a study with the same name already exists, it will be resumed.
 - Pressing Ctrl+Z will aborted the study at the end of the current trial.

The config is defined in `configs/hpo/hpo.yaml`.
"""
from pathlib import Path
import signal

import hydra
import lightning as L
from lightning import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import OmegaConf, DictConfig
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import wandb

from data.datasets import SynthDatasetPkl
from ssm.litm_ssm import SSMLitModule
from ssm.estimator_network import EstimatorNet
from utils.logging import RankedLogger
from utils.synth import PresetHelper

log = RankedLogger(__name__, rank_zero_only=True)

# For interrupting the study if a signal is received
is_interrupted = False  # pylint: disable=C0103


# Define a signal handler function for SIGSTP
def sigstp_handler(study: optuna.study.Study, signal, frame) -> None:
    """Handler for SIGSTP signal, aborts the current optuna study."""
    # Abort the study
    log.info(f"Signal {signal} detected, the study will be aborted at the end of the current trial...")
    study.stop()
    global is_interrupted  # pylint: disable=W0603
    is_interrupted = True


# Function to register signal handler with additional arguments
def register_signal_handler_with_args(signal_num, handler_func, *args):
    """Register a signal handler with additional arguments."""
    signal.signal(signal_num, lambda signal, frame: handler_func(*args, signal, frame))


def objective(trial: optuna.trial.Trial, cfg: DictConfig, is_startup: bool) -> float:
    """Objective function for optuna HPO."""
    # Init Flag for Pruning
    is_pruned = False

    # set RNG seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)

    # Sample hyperparameter values if required:
    hps = {}
    for k, v in cfg.search_space.items():
        hps[k] = getattr(trial, "suggest_" + v.type)(**v.kwargs) if isinstance(v, DictConfig) else v

    # instantiate train Dataset & DataLoader
    dataset = SynthDatasetPkl(**cfg.dataset)
    dataset_train, dataset_val = random_split(dataset, cfg.train_val_split)
    loader_train = DataLoader(
        dataset_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    loader_val = DataLoader(
        dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # instantiate PresetHelper
    preset_helper = PresetHelper(
        synth_name=dataset.synth_name,
        parameters_to_exclude=dataset.configs_dict["params_to_exclude"],
    )

    # instantiate estimator net
    estimator = EstimatorNet(preset_helper)

    # instantiate synth_proxy if required
    if cfg.has_perceptual_loss:
        synth_proxy: nn.Module = hydra.utils.instantiate(
            cfg.synth_proxy, out_features=dataset.embedding_dim, preset_helper=preset_helper
        )
        # loading ckpt
        ckpt = torch.load(cfg.path_to_ckpt)
        synth_proxy.load_state_dict(
            {k.replace("preset_encoder.", ""): v for k, v in ckpt["state_dict"].items()}
        )
        synth_proxy.eval()
    else:
        synth_proxy = None

    # instantiate optimizer, lr_scheduler, and scheduler_config
    lr = hps.get("lr", 1e-3)
    beta1 = hps.get("beta1", 0.9)
    beta2 = hps.get("beta2", 0.999)
    eps = hps.get("eps", 1e-8)
    weight_decay = hps.get("weight_decay", 0.0)

    opt_cfg = {
        "optimizer_kwargs": {"lr": lr, "betas": (beta1, beta2), "eps": eps, "weight_decay": weight_decay},
        "num_warmup_steps": hps.get("num_warmup_steps", 0),
    }

    # instantiate Lightning Module
    model = SSMLitModule(
        preset_helper=preset_helper,
        estimator=estimator,
        opt_cfg=opt_cfg,
        synth_proxy=synth_proxy,
        label_smoothing=hps.get("label_smoothing", 0.0),
        cat_loss_weight=cfg.get("cat_loss_weight", 1.0),
    )

    # instantiate logger
    logger = hydra.utils.instantiate(cfg.wandb, name=f"trial_{trial.number}") if cfg.get("wandb") else []

    # instantiate Lightning Trainer
    # don't auto detect SLURM environment
    SLURMEnvironment.detect = lambda: False
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=cfg.metric_to_optimize)],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    # Train for one epoch
    trainer.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_val)

    # get metric to optimize's value
    metric_value = trainer.callback_metrics[cfg.metric_to_optimize].item()

    # set user attributes
    trial.set_user_attr("seed", cfg.seed)
    trial.set_user_attr("dataset", dataset.path_to_dataset.stem)
    trial.set_user_attr("metric", cfg.metric_to_optimize)
    trial.set_user_attr("batch_size", cfg.batch_size)
    trial.set_user_attr(
        "sampler", cfg.sampler.get("name_startup_sampler") if is_startup else cfg.sampler.name
    )

    # Flag to abort the study if loss diverges
    if trainer.callback_metrics["train/loss"].item() > 1 or torch.isnan(
        trainer.callback_metrics["train/loss"]
    ):
        is_pruned = True

    if logger:
        # log hyperparameters, seed, and dataset refs
        # log sampler and pruner config
        sp_dict = {"sampler": {}, "pruner": {}}
        for name, d in sp_dict.items():
            tmp_dict = OmegaConf.to_container(cfg[name], resolve=True)
            d["name"] = (
                tmp_dict.get("name_startup_sampler")
                if is_startup and name == "sampler"
                else tmp_dict.get("name")
            )
            cfg_key = f"cfg{'_startup' if is_startup and name == 'sampler' else ''}"
            for param, value in tmp_dict[cfg_key].items():
                if param != "_target_":
                    d[f"{param}"] = value
        trainer.logger.log_hyperparams({"hps": hps, "misc": trial.user_attrs, "pruned": is_pruned, **sp_dict})
        # terminate wandb run
        wandb.finish()

    # Abort the study if loss diverges
    if is_pruned:
        log.info("Trial aborted due to high or NaN loss")
        raise optuna.TrialPruned

    return metric_value


@hydra.main(version_base="1.3", config_path="../../configs/ssm/hpo", config_name="estimator_net.yaml")
def hpo(cfg: DictConfig) -> None:
    """Main function for HPO. Config defined in `configs/hpo/hpo.yaml`."""
    # Add stream handler of stdout to show the messages
    optuna.logging.enable_propagation()  # send messages to the root logger
    optuna.logging.disable_default_handler()  # prevent showing messages twice

    study_name = cfg.study_name  # Unique identifier of the study.

    storage_name = f"sqlite:///{study_name}.db"

    # load sampler if exists else create
    if (Path.cwd() / "optuna_sampler.pkl").exists():
        log.info(f"Loading sampler from {Path.cwd() / 'optuna_sampler.pkl'}")
        sampler = torch.load(Path.cwd() / "optuna_sampler.pkl")
    else:
        if cfg.sampler.get("cfg_startup"):
            sampler = hydra.utils.instantiate(cfg.sampler.cfg_startup)
        else:
            sampler = hydra.utils.instantiate(cfg.sampler.cfg)

    pruner = hydra.utils.instantiate(cfg.pruner.cfg)

    study = optuna.create_study(
        study_name=str(study_name),
        storage=str(storage_name),
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
        direction=cfg.direction,
    )

    # Register the signal handler for different signals with additional study argument
    register_signal_handler_with_args(signal.SIGTSTP, sigstp_handler, study)
    register_signal_handler_with_args(signal.SIGUSR1, sigstp_handler, study)  # for HPC
    register_signal_handler_with_args(signal.SIGTERM, sigstp_handler, study)  # for HPC

    # Start startup trials if needed
    num_startup_trials = cfg.sampler.get("num_startup_trials", 0)

    if 0 <= len(study.trials) < num_startup_trials:
        log.info(f"Starting HPO startup with {study.sampler}...")
        log.info(f"Number of remaining startup trial: {num_startup_trials - len(study.trials)}")
        study.optimize(
            lambda trial: objective(trial, cfg, is_startup=True),
            n_trials=num_startup_trials - len(study.trials),
        )
        if not is_interrupted:
            log.info("Startup trials finished.")

    # Start non-startup trials except if study was aborted by user
    if not is_interrupted:

        # create sampler if the study was aborted exactly at the end of the startup trials
        # i.e., the pickled sampler is the one use during startup trials and need to be replaced
        if str(type(study.sampler)).split(".")[-1][:-2] != str(cfg.sampler.cfg._target_).split(".")[-1]:
            study.sampler = hydra.utils.instantiate(cfg.sampler.cfg)

        log.info(
            f"Starting HPO with {study.sampler}...\n"
            f"Number of remaining trials: {cfg.num_trials - len(study.trials)}"
        )

        study.optimize(
            lambda trial: objective(trial, cfg, is_startup=False),
            n_trials=cfg.num_trials - len(study.trials),
        )

    log.info(f"Number of finished trials: {len(study.trials)}")

    log.info("Best trial:")
    trial = study.best_trial
    log.info(f" Value: {trial.value}")

    log.info(" Params: ")
    for key, value in trial.params.items():
        log.info(f"  {key}: {value}")

    # Save the sampler to be loaded later if needed.
    log.info("Saving sampler...")
    with open("optuna_sampler.pkl", "wb") as f:
        torch.save(study.sampler, f)


if __name__ == "__main__":
    # import sys
    # args = ["src/hpo/run.py", <other args for debug>]
    # gettrace = getattr(sys, "gettrace", None)
    # if gettrace():
    #     sys.argv = args

    hpo()  # pylint: disable=no-value-for-parameter
