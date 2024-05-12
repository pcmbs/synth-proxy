# pylint: disable=W0212:protected-access
# pylint: disable=W1203:logging-fstring-interpolation
# pylint: disable=E1120:no-value-for-parameter
"""
Evaluation script.
Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py

See configs/eval/eval.yaml for more details on CLI arguments

Usage example using hydra multirun to evaluate all models on Tal-NoiseMaker:
    python src/eval.py -m model="glob(talnm*)"

"""
from pathlib import Path
import pickle
from typing import Dict

import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import numpy as np
from omegaconf import DictConfig
from torch import nn
import torch
from tqdm import tqdm
import wandb

from data.datasets import SynthDatasetPkl
from models.lit_module import PresetEmbeddingLitModule
from utils.evaluation import one_vs_all_eval, non_overlapping_eval, eval_logger
from utils.logging import RankedLogger
from utils.synth.preset_helper import PresetHelper

# logger for this file
log = RankedLogger(__name__, rank_zero_only=True)


def _log_results_to_console(results: Dict) -> None:
    for k, v in results.items():
        if k in ["mrr", "loss"]:
            log.info(f"{k}: mean: {v['mean']:.5f}; std.: {v['std']:.5f}")
        elif k == "top_k_mrr":
            for i, _ in enumerate(v["mean"]):
                log.info(f"mrr@{i+1}: mean: {v['mean'][i]:.5f}; std.: {v['std'][i]:.5f}")
        else:
            pass


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="eval.yaml")
def evaluate(cfg: DictConfig) -> None:
    """
    Evaluation pipeline for the preset embedding framework.

    Args:
        cfg (DictConfig): A DictConfig configuration composed by Hydra.
    """
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    L.seed_everything(cfg.seed)

    ##### Initializing datasets, models, and logger
    log.info(f"Instantiating hand-crafted presets Dataset: {cfg.synth.dataset_path}")
    hc_dataset = SynthDatasetPkl(cfg.synth.dataset_path)

    log.info(f"Instantiating synthetic presets Dataset: {cfg.synth.rnd_dataset_path}")
    rnd_dataset = SynthDatasetPkl(cfg.synth.rnd_dataset_path)

    assert hc_dataset.configs_dict["params_to_exclude"] == rnd_dataset.configs_dict["params_to_exclude"]

    log.info(f"Instantiating Preset Helper for synth {hc_dataset.synth_name} and excluded params:")
    log.info(f"{hc_dataset.configs_dict['params_to_exclude']}")
    preset_helper = PresetHelper(
        synth_name=hc_dataset.synth_name,
        parameters_to_exclude=hc_dataset.configs_dict["params_to_exclude"],
    )

    log.info(f"Instantiating Preset Encoder <{cfg.model.cfg._target_}>")
    m_preset: nn.Module = hydra.utils.instantiate(
        cfg.model.cfg, out_features=hc_dataset.embedding_dim, preset_helper=preset_helper
    )

    if cfg.get("wandb"):
        log.info("Instantiating wandb logger...")
        run = wandb.init(**cfg.wandb)

    log.info(f"Loading checkpoint from {cfg.ckpt_path}...")
    model = PresetEmbeddingLitModule.load_from_checkpoint(cfg.ckpt_path, preset_encoder=m_preset)
    model.to(device)
    model.freeze()

    ##### Computing Evaluation Metrics
    # the following statement should only take len(hc_dataset) for talnm, since there are only 404 presets..
    num_hc_presets = cfg.subset_size if 0 < cfg.subset_size < len(hc_dataset) else len(hc_dataset)
    log.info(f"Computing evaluation metrics on {num_hc_presets} hand-crafted presets (one-vs-all)...")
    hc_mrr = []
    hc_loss = []
    hc_top_k_mrr = []
    pbar = tqdm(range(cfg.num_runs), total=cfg.num_runs, dynamic_ncols=True)
    for i in pbar:
        mrr, top_k_mrr, hc_ranks_dict, loss = one_vs_all_eval(
            model=model,
            dataset=hc_dataset,
            subset_size=cfg.subset_size,
            device=device,
            seed=cfg.seed + i,
            log_results=cfg.num_runs == 1,
        )
        hc_mrr.append(mrr)
        hc_top_k_mrr.append(top_k_mrr)
        hc_loss.append(loss)

    hc_mrr = {"mean": np.mean(hc_mrr), "std": np.std(hc_mrr)}
    hc_top_k_mrr = np.row_stack(hc_top_k_mrr)
    hc_top_k_mrr = {
        "mean": np.mean(hc_top_k_mrr, axis=0),
        "std": np.std(hc_top_k_mrr, axis=0),
    }
    hc_loss = {"mean": np.mean(hc_loss), "std": np.std(hc_loss)}

    hc_results = {
        "mrr": hc_mrr,
        "top_k_mrr": hc_top_k_mrr,
        "ranks": hc_ranks_dict,
        "loss": hc_loss,
        "num_hc_presets": num_hc_presets,
    }

    _log_results_to_console(hc_results)

    log.info(f"Computing evaluation metrics on {cfg.subset_size} synthetic presets (one-vs-all)...")
    rnd_mrr = []
    rnd_loss = []
    rnd_top_k_mrr = []
    pbar = tqdm(range(cfg.num_runs), total=cfg.num_runs, dynamic_ncols=True)
    for i in pbar:
        mrr, top_k_mrr, _, loss = one_vs_all_eval(
            model=model,
            dataset=rnd_dataset,
            subset_size=cfg.subset_size,
            device=device,
            seed=cfg.seed + i,
            log_results=cfg.num_runs == 1,
        )
        rnd_mrr.append(mrr)
        rnd_top_k_mrr.append(top_k_mrr)
        rnd_loss.append(loss)

    rnd_mrr = {"mean": np.mean(rnd_mrr), "std": np.std(rnd_mrr)}
    rnd_top_k_mrr = np.row_stack(rnd_top_k_mrr)
    rnd_top_k_mrr = {"mean": np.mean(rnd_top_k_mrr, axis=0), "std": np.std(rnd_top_k_mrr, axis=0)}
    rnd_loss = {"mean": np.mean(rnd_loss), "std": np.std(rnd_loss)}

    rnd_results = {
        "mrr": rnd_mrr,
        "top_k_mrr": rnd_top_k_mrr,
        "loss": rnd_loss,
    }

    _log_results_to_console(rnd_results)

    rnd_incr_results = {}
    if cfg.random_incremential:
        log.info("Computing evaluation metrics on increasing number of synthetic presets (one-vs-all)...")
        for i in range(9, int(np.log2(len(rnd_dataset))) + 1):
            num_presets = int(2**i)
            log.info(f"{num_presets} synthetic presets...")
            mrr, _, _, loss = one_vs_all_eval(
                model=model,
                dataset=rnd_dataset,
                subset_size=num_presets,
                device=device,
                seed=cfg.seed,
                log_results=False,
            )
            rnd_incr_results[num_presets] = {"mrr": mrr, "loss": loss}

    if cfg.random_non_overlapping:
        log.info(f"Computing evaluation metrics on {len(rnd_dataset)} synthetic presets (non-overlapping)...")
        rnd_nol_mrr, rnd_nol_top_k_mrr, rnd_nol_ranks_dict, rnd_nol_loss = non_overlapping_eval(
            model=model, dataset=rnd_dataset, num_ranks=256, device=device, seed=cfg.seed, log_results=True
        )

        rnd_nol_results = {
            "mrr": rnd_nol_mrr,
            "top_k_mrr": rnd_nol_top_k_mrr,
            "ranks": rnd_nol_ranks_dict,
            "loss": rnd_nol_loss,
            "num_rnd_presets": len(rnd_dataset),
        }
        _log_results_to_console(rnd_nol_results)

    else:
        rnd_nol_results = {}

    results = {
        "hc": hc_results,
        "rnd": rnd_results,
        "rnd_nol": rnd_nol_results,
        "rnd_incr": rnd_incr_results,
    }

    ##### Logging results
    object_dict = {
        "cfg": cfg,
        "model": m_preset,
        "dataset_cfg": hc_dataset.configs_dict,
        "results": results,
    }

    if cfg.get("wandb"):
        log.info("Logging hyperparameters...")
        eval_logger(object_dict=object_dict, run=run)
        wandb.finish()  # required for hydra multirun

    with open(Path(HydraConfig.get().runtime.output_dir) / "results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    # import sys

    # args = ["src/eval.py", "model=talnm_highway_ft", "~wandb"]

    # gettrace = getattr(sys, "gettrace", None)
    # if gettrace():
    #     sys.argv = args
    evaluate()  # pylint: disable=no-value-for-parameter
