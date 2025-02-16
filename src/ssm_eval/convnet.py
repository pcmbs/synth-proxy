# pylint: disable=E1102:not-callable
"""
Adapted from
https://github.com/inversynth/InverSynth2/blob/main/Python%20code/FM%20Synth/encdec.py 
"""

import torch
import torch.nn.functional as F
from torch import nn

from utils.synth import PresetHelper


class ConvNet(nn.Module):
    def __init__(self, preset_helper: PresetHelper):
        super().__init__()

        self.p_helper = preset_helper
        self.ch = 1

        self.out_dim = self._compute_out_dim(preset_helper)

        self.enc_nn = nn.Sequential(
            nn.Conv2d(self.ch, 8, (5, 5), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(8, 16, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(16, 32, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(128, 256, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 512, (4, 4), stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        )

        self.features_mixer_cnn = nn.Sequential(
            nn.Conv2d(512, 2048, (1, 1), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features=2048, out_features=self.out_dim),
            nn.BatchNorm1d(self.out_dim, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        )

        self.init_weights()

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters of the model."""
        return sum(p.numel() for p in self.parameters())

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.1)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze_(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze_(0).unsqueeze_(0)
        x = self.enc_nn(x)
        x = self.features_mixer_cnn(x)
        B, C, H, W = x.shape
        x = F.avg_pool2d(x, kernel_size=(H, W)).view(B, -1)
        x = self.mlp(x)
        return x

    def _compute_out_dim(self, p_helper: PresetHelper):
        """
        Compute the output dimension of the model.
        - Categorical parameters: one output for each class (for CE loss)
        - Binary parameters: a single output (for BCE loss)
        - Numerical parameters: a single output (for regression loss)
        """
        out_dim = 0
        for p in p_helper.used_parameters:
            if p.type in ["bin", "num"]:
                out_dim += 1
            else:  # cat
                out_dim += p.cardinality
        return out_dim



if __name__ == "__main__":
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from omegaconf import OmegaConf

    load_dotenv()
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

    SYNTH = "diva"

    excluded_params = OmegaConf.load(PROJECT_ROOT / "configs" / "export" / "synth" / f"{SYNTH}.yaml")[
        "parameters_to_exclude_str"
    ]

    p_helper = PresetHelper(SYNTH, excluded_params)

    model = ConvNet(p_helper)
    model.eval()
    print(model)

    x = torch.empty((2, 128, 431)).uniform_(-1, 1)
    # x = torch.empty((128, 431)).uniform_(-1, 1)

    with torch.no_grad():
        y = model(x)

    print("Number of parameters", model.num_parameters)
    print("Input shape", x.shape)
    print("Output shape", y.shape)

    print("✅")
