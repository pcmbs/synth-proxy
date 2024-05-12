"""
Module implementing a function to log hyperparameters.

Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/logging_utils.py 
"""

import logging
import os
from typing import Any, Dict
from omegaconf import OmegaConf
import wandb

log = logging.getLogger(__name__)


def eval_logger(object_dict: Dict[str, Any], run: wandb.sdk.wandb_run.Run) -> None:
    """Log infos & hps related to the evaluation.

    Args:
    object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"dataset_cfg"`: A DictConfig object containing the dataset config.
        - `"results"`: A dictionary containing the evaluation results.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    dataset_cfg = object_dict["dataset_cfg"]
    hc_results = object_dict["results"].get("hc", {})
    rnd_nol_results = object_dict["results"].get("rnd_nol", {})
    rnd_results = object_dict["results"].get("rnd", {})

    # General hyperparameters
    hparams["seed"] = cfg.get("seed")
    hparams["num_runs"] = cfg.get("num_runs")

    # Data related hyperparameters
    hparams["data/synth"] = cfg["synth"]["name"]
    hparams["data/excluded_params"] = dataset_cfg["params_to_exclude"]
    hparams["data/num_used_params"] = dataset_cfg["num_used_params"]
    hparams["data/render_duration_in_sec"] = dataset_cfg["render_duration_in_sec"]
    hparams["data/midi_note"] = dataset_cfg["midi_note"]
    hparams["data/midi_velocity"] = dataset_cfg["midi_velocity"]
    hparams["data/midi_duration_in_sec"] = dataset_cfg["midi_duration_in_sec"]
    hparams["data/audio_fe"] = dataset_cfg["audio_fe"]
    hparams["data/embedding_dim"] = dataset_cfg["num_outputs"]

    # Model hyperparameters
    hparams["model/type"] = cfg["model"]["type"]
    hparams["model/num_parameters"] = object_dict["model"].num_parameters
    hparams["model/ckpt_name"] = cfg["model"]["ckpt_name"]
    hparams["num_train_epochs"] = cfg["model"]["num_train_epochs"]

    for i, v in cfg["model"]["cfg"].items():
        if i == "_target_":
            hparams["model/name"] = v.split(".")[-1]
        elif i in ["embedding_kwargs", "block_kwargs"]:
            for kk, vv in v.items():
                hparams[f"model/{i}/{kk}"] = vv
        else:
            hparams[f"model/{i}"] = v

    # Results
    metrics_dict = {}
    # Results from training
    metrics_dict["val"] = {
        "mrr": cfg["model"]["val_mrr"],
        "loss": cfg["model"]["val_loss"],
        "epoch": cfg["model"]["epoch"],
    }
    # Results on hand-crafted presets
    if hc_results:
        hparams["data/num_hc_presets"] = hc_results["num_hc_presets"]
        metrics_dict["hc"] = {
            "mrr_mean": hc_results["mrr"]["mean"],
            "mrr_std": hc_results["mrr"]["std"],
            "loss_mean": hc_results["loss"]["mean"],
            "loss_std": hc_results["loss"]["std"],
        }
        for s in ["mean", "std"]:
            for i, result in enumerate(hc_results["top_k_mrr"][s]):
                metrics_dict["hc"][f"top_{i+1}_mrr_{s}"] = result

    # Results on random presets
    if rnd_nol_results:
        hparams["data/num_rnd_presets"] = rnd_nol_results["num_rnd_presets"]
        metrics_dict["rnd_nol"] = {"mrr": rnd_nol_results["mrr"], "loss": rnd_nol_results["loss"]}
        for i, mrr in enumerate(rnd_nol_results["top_k_mrr"]):
            metrics_dict["rnd_nol"][f"top_{i+1}_mrr"] = mrr
    # Results on random subsets of presets
    if rnd_results:
        metrics_dict["rnd"] = {
            "mrr_mean": rnd_results["mrr"]["mean"],
            "mrr_std": rnd_results["mrr"]["std"],
            "loss_mean": rnd_results["loss"]["mean"],
            "loss_std": rnd_results["loss"]["std"],
        }
        for s in ["mean", "std"]:
            for i, result in enumerate(rnd_results["top_k_mrr"][s]):
                metrics_dict["rnd"][f"top_{i+1}_mrr_{s}"] = result

    # log metrics
    run.log({"metrics": metrics_dict})  # log instead of summary to be able to create bar charts in wandb UI
    wandb.config.update(hparams)

    # hydra config is saved under <project_name>/Runs/<run_id>/Files/.hydra
    wandb.save(
        glob_str=os.path.join(cfg["paths"].get("output_dir"), ".hydra", "*.yaml"),
        base_path=cfg["paths"].get("output_dir"),
    )
