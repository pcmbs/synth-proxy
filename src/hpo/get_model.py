"""
Get a model to be used for hyperparameter optimization.

def get_<model>(
    out_features: int, preset_helper: PresetHelper, model_cfg: DictConfig, hps: DictConfig
) -> nn.Module:
    Return a <model> model instantiated with the parameters taken from (in order of precedence):
    (1): the model_cfg DictCongif (i.e., fixed hyperparameters), or
    (2): the hps DictConfig (i.e., tuned hyperparameters).

where <model> can be:
 - mlp_oh
 - gru_oh
 - resnet_oh
 - hn_oh
 - hn_pt
 - hn_ptgru
 - tfm
"""

import hydra
from omegaconf import DictConfig
from torch import nn

from utils.synth.preset_helper import PresetHelper


def get_hn_pt(
    out_features: int, preset_helper: PresetHelper, model_cfg: DictConfig, hps: DictConfig
) -> nn.Module:
    """
    Return a hn_pt model instantiated with the parameters taken from (in order of precedence):
    (1): the model_cfg DictCongif (i.e., fixed hyperparameters), or
    (2): the hps DictConfig (i.e., tuned hyperparameters).
    """
    return hydra.utils.instantiate(
        model_cfg,
        out_features=out_features,
        preset_helper=preset_helper,
        num_blocks=model_cfg.get("num_blocks") or hps.get("num_blocks"),
        hidden_features=model_cfg.get("hidden_features") or hps.get("hidden_features"),
        token_dim=model_cfg.get("token_dim") or hps.get("token_dim"),
        pe_dropout_p=model_cfg.get("pe_dropout_p") or hps.get("pe_dropout_p") or 0.0,
        block_norm=model_cfg.get("block_norm") or hps.get("block_norm"),
        block_act_fn=model_cfg.get("block_act_fn") or hps.get("block_act_fn"),
        block_dropout_p=model_cfg.get("block_dropout_p") or hps.get("block_dropout_p") or 0.0,
    )


def get_hn_ptgru(
    out_features: int, preset_helper: PresetHelper, model_cfg: DictConfig, hps: DictConfig
) -> nn.Module:
    """
    Return a hn_ptgru model instantiated with the parameters taken from (in order of precedence):
    (1): the model_cfg DictCongif (i.e., fixed hyperparameters), or
    (2): the hps DictConfig (i.e., tuned hyperparameters).
    """
    return hydra.utils.instantiate(
        model_cfg,
        out_features=out_features,
        preset_helper=preset_helper,
        num_blocks=model_cfg.get("num_blocks") or hps.get("num_blocks"),
        hidden_features=model_cfg.get("hidden_features") or hps.get("hidden_features"),
        token_dim=model_cfg.get("token_dim") or hps.get("token_dim"),
        pe_dropout_p=model_cfg.get("pe_dropout_p") or hps.get("pe_dropout_p") or 0.0,
        block_norm=model_cfg.get("block_norm") or hps.get("block_norm"),
        block_act_fn=model_cfg.get("block_act_fn") or hps.get("block_act_fn"),
        block_dropout_p=model_cfg.get("block_dropout_p") or hps.get("block_dropout_p") or 0.0,
    )


def get_mlp_oh(
    out_features: int, preset_helper: PresetHelper, model_cfg: DictConfig, hps: DictConfig
) -> nn.Module:
    """
    Return a mlp_oh model instantiated with the parameters taken from (in order of precedence):
    (1): the model_cfg DictCongif (i.e., fixed hyperparameters), or
    (2): the hps DictConfig (i.e., tuned hyperparameters).
    """
    return hydra.utils.instantiate(
        model_cfg,
        out_features=out_features,
        preset_helper=preset_helper,
        num_blocks=model_cfg.get("num_blocks") or hps.get("num_blocks"),
        hidden_features=model_cfg.get("hidden_features") or hps.get("hidden_features"),
        block_norm=model_cfg.get("block_norm") or hps.get("block_norm"),
        block_act_fn=model_cfg.get("block_act_fn") or hps.get("block_act_fn"),
        block_dropout_p=model_cfg.get("block_dropout_p") or hps.get("block_dropout_p") or 0.0,
    )


def get_hn_oh(
    out_features: int, preset_helper: PresetHelper, model_cfg: DictConfig, hps: DictConfig
) -> nn.Module:
    """
    Return a hn_oh model instantiated with the parameters taken from (in order of precedence):
    (1): the model_cfg DictCongif (i.e., fixed hyperparameters), or
    (2): the hps DictConfig (i.e., tuned hyperparameters).
    """
    return hydra.utils.instantiate(
        model_cfg,
        out_features=out_features,
        preset_helper=preset_helper,
        num_blocks=model_cfg.get("num_blocks") or hps.get("num_blocks"),
        hidden_features=model_cfg.get("hidden_features") or hps.get("hidden_features"),
        block_norm=model_cfg.get("block_norm") or hps.get("block_norm"),
        block_act_fn=model_cfg.get("block_act_fn") or hps.get("block_act_fn"),
        block_dropout_p=model_cfg.get("block_dropout_p") or hps.get("block_dropout_p") or 0.0,
    )


def get_resnet_oh(
    out_features: int, preset_helper: PresetHelper, model_cfg: DictConfig, hps: DictConfig
) -> nn.Module:
    """
    Return a resnet_oh model instantiated with the parameters taken from (in order of precedence):
    (1): the model_cfg DictCongif (i.e., fixed hyperparameters), or
    (2): the hps DictConfig (i.e., tuned hyperparameters).
    """
    return hydra.utils.instantiate(
        model_cfg,
        out_features=out_features,
        preset_helper=preset_helper,
        num_blocks=model_cfg.get("num_blocks") or hps.get("num_blocks"),
        hidden_features=model_cfg.get("hidden_features") or hps.get("hidden_features"),
        block_norm=model_cfg.get("block_norm") or hps.get("block_norm"),
        block_act_fn=model_cfg.get("block_act_fn") or hps.get("block_act_fn"),
        block_dropout_p=model_cfg.get("block_dropout_p") or hps.get("block_dropout_p") or 0.0,
        block_residual_dropout_p=model_cfg.get("block_residual_dropout_p")
        or hps.get("block_residual_dropout_p")
        or 0.0,
    )


def get_gru_oh(
    out_features: int, preset_helper: PresetHelper, model_cfg: DictConfig, hps: DictConfig
) -> nn.Module:
    """
    Return a gru_oh model instantiated with the parameters taken from (in order of precedence):
    (1): the model_cfg DictCongif (i.e., fixed hyperparameters), or
    (2): the hps DictConfig (i.e., tuned hyperparameters).
    """
    return hydra.utils.instantiate(
        model_cfg,
        out_features=out_features,
        preset_helper=preset_helper,
        num_layers=model_cfg.get("num_layers") or hps.get("num_layers"),
        hidden_features=model_cfg.get("hidden_features") or hps.get("hidden_features"),
        dropout_p=model_cfg.get("dropout_p") or hps.get("dropout_p") or 0.0,
    )


def get_tfm(
    out_features: int, preset_helper: PresetHelper, model_cfg: DictConfig, hps: DictConfig
) -> nn.Module:
    """
    Return a tfm model instantiated with the parameters taken from (in order of precedence):
    (1): the model_cfg DictCongif (i.e., fixed hyperparameters), or
    (2): the hps DictConfig (i.e., tuned hyperparameters).
    """
    return hydra.utils.instantiate(
        model_cfg,
        out_features=out_features,
        preset_helper=preset_helper,
        pe_type=model_cfg.get("pe_type") or hps.get("pe_type"),
        hidden_features=model_cfg.get("hidden_features") or hps.get("hidden_features"),
        num_blocks=model_cfg.get("num_blocks") or hps.get("num_blocks"),
        num_heads=model_cfg.get("num_heads") or hps.get("num_heads"),
        mlp_factor=model_cfg.get("mlp_factor") or hps.get("mlp_factor"),
        pooling_type=model_cfg.get("pooling_type") or hps.get("pooling_type"),
        last_activation=model_cfg.get("last_activation") or hps.get("last_activation"),
        pe_dropout_p=model_cfg.get("pe_dropout_p") or hps.get("pe_dropout_p") or 0.0,
        block_activation=model_cfg.get("block_activation") or hps.get("block_activation"),
        block_dropout_p=model_cfg.get("block_dropout_p") or hps.get("block_dropout_p") or 0.0,
    )
