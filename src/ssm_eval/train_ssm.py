import os
from typing import Tuple

from dotenv import load_dotenv
from omegaconf import OmegaConf
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data.datasets.synth_dataset_pkl import SynthDatasetPkl
from ssm_eval.estimator_network import ConvNet
from utils.synth import PresetHelper

load_dotenv()

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

PATH_TO_DATASET = PROJECT_ROOT / "data" / "datasets" / "eval" / "diva_mn20_hc_v1"

SYNTH = "diva"
 

class ParameterLoss(nn.Module):
    """
    Parameter loss for synthesizer sound matching task.
    """

    def __init__(
        self,
        preset_helper: PresetHelper,
        num_loss: "str" = "L1Loss",
        cat_loss: "str" = "CrossEntropyLoss",
        bin_loss: "str" = "BCEWithLogitsLoss",
    ) -> None:
        """
        Args:
            preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        """
        super().__init__()
        self.p_helper = preset_helper

        # losses
        self.num_loss_fn = getattr(nn, num_loss)()
        self.cat_loss_fn = getattr(nn, cat_loss)()
        self.bin_loss_fn = getattr(nn, bin_loss)()

        # index of parameters per type in the target vector
        self.num_idx_target = torch.tensor(preset_helper.used_num_parameters_idx, dtype=torch.long)
        self.cat_idx_target = torch.tensor(preset_helper.used_cat_parameters_idx, dtype=torch.long)
        self.bin_idx_target = torch.tensor(preset_helper.used_bin_parameters_idx, dtype=torch.long)

        # index of parameters per type in the prediction vector
        # (different since there are C outputs for a categorical parameter with C categories)
        num_idx_pred = []
        cat_idx_pred = []
        bin_idx_pred = []

        offset = 0
        for p in p_helper.used_parameters:
            if p.type == "num":
                num_idx_pred.append(offset)
                offset += 1
            elif p.type == "cat":
                cat_idx_pred.append(torch.arange(offset, offset + p.cardinality, dtype=torch.long))
                offset += p.cardinality
            else:  # bin
                bin_idx_pred.append(offset)
                offset += 1

        self.num_idx_pred = torch.tensor(num_idx_pred, dtype=torch.long)
        self.cat_idx_pred = cat_idx_pred
        self.bin_idx_pred = torch.tensor(bin_idx_pred, dtype=torch.long)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # regression loss for numerical parameters (pass through sigmoid first since in [0, 1])
        num_loss = self.num_loss_fn(F.sigmoid(pred[:, self.num_idx_pred]), target[:, self.num_idx_target])

        # binary loss for binary parameters
        bin_loss = self.bin_loss_fn(pred[:, self.bin_idx_pred], target[:, self.bin_idx_target])

        # categorical loss for categorical parameters
        # TODO: might need to scale down this loss...
        cat_loss = sum(
            [
                self.cat_loss_fn(pred[:, i], target[:, j].to(dtype=torch.long))
                for i, j in zip(self.cat_idx_pred, self.cat_idx_target)
            ]
        )
        return num_loss, bin_loss, cat_loss


class PresetDecoder(nn.Module):
    """
    Get predicted presets from model predictions.
    """

    def __init__(
        self,
        preset_helper: PresetHelper,
    ) -> None:
        """
        Args:
            preset_helper (PresetHelper): An instance of PresetHelper for a given synthesizer.
        """
        super().__init__()
        self.p_helper = preset_helper

        # index of parameters per type in the target vector
        self.num_idx_target = torch.tensor(preset_helper.used_num_parameters_idx, dtype=torch.long)
        self.cat_idx_target = torch.tensor(preset_helper.used_cat_parameters_idx, dtype=torch.long)
        self.bin_idx_target = torch.tensor(preset_helper.used_bin_parameters_idx, dtype=torch.long)

        # index of parameters per type in the prediction vector
        # (different since there are C outputs for a categorical parameter with C categories)
        num_idx_pred = []
        cat_idx_pred = []
        bin_idx_pred = []

        offset = 0
        for p in p_helper.used_parameters:
            if p.type == "num":
                num_idx_pred.append(offset)
                offset += 1
            elif p.type == "cat":
                cat_idx_pred.append(torch.arange(offset, offset + p.cardinality, dtype=torch.long))
                offset += p.cardinality
            else:  # bin
                bin_idx_pred.append(offset)
                offset += 1

        self.num_idx_pred = torch.tensor(num_idx_pred, dtype=torch.long)
        self.cat_idx_pred = cat_idx_pred
        self.bin_idx_pred = torch.tensor(bin_idx_pred, dtype=torch.long)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(pred.shape[0], self.p_helper.num_used_parameters, device=pred.device)
        # Pass the predictions through sigmoid for numerical parameters since in [0, 1]
        out[:, self.num_idx_target] = F.sigmoid(pred[:, self.num_idx_pred])
        # 0 or 1 for binary parameters (discrimination threshold at 0.5)
        out[:, self.bin_idx_target] = F.sigmoid(pred[:, self.bin_idx_pred]).round()
        # argmax for categorical parameters
        for i, j in zip(self.cat_idx_pred, self.cat_idx_target):
            # pred[:, i] = F.softmax(pred[:, i], dim=1)
            out[:, j] = torch.argmax(pred[:, i], dim=1)
        return out


if __name__ == "__main__":

    excluded_params = OmegaConf.load(PROJECT_ROOT / "configs" / "export" / "synth" / f"{SYNTH}.yaml")[
        "parameters_to_exclude_str"
    ]

    p_helper = PresetHelper(SYNTH, excluded_params)

    dataset = SynthDatasetPkl(
        PATH_TO_DATASET,
        split="test",
        has_mel=True,
        mel_norm="mean_std",
        mmap=True,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = ConvNet(preset_helper=p_helper)
    print(model)

    p_loss = ParameterLoss(p_helper)
    preset_decoder = PresetDecoder(p_helper)

    for i, (params, emb, mel) in enumerate(loader):
        y_hat = model(mel)
        y = params
        num_loss, bin_loss, cat_loss = p_loss(y_hat, y)
        loss = num_loss + bin_loss + cat_loss
        preset = preset_decoder(y_hat)
        break

    print("predicted logits shape:", y_hat.shape)
    print("num_loss:", num_loss.item())
    print("bin loss:", bin_loss.item())
    print("cat loss:", cat_loss.item())
    print("combined loss:", loss.item())
    print("target preset shape:", y.shape)
    print("predicted preset shape:", preset.shape)
    print(preset[0])
