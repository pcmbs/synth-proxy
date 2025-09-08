# pylint: disable=W0212:protected-access
# pylint: disable=W1203:logging-fstring-interpolation
# pylint: disable=E1120:no-value-for-parameter
"""
Script to export audio samples from trained SSM models for qualitative
evaluation on both in-domain and the out-of-domain nsynth datasets.
"""

from datetime import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

import hydra
import torch
from torch.utils.data import DataLoader, Subset
from scipy.io import wavfile

from omegaconf import OmegaConf


from data.datasets import SynthDatasetPkl, NSynthDataset
from ssm.estimator_network import EstimatorNet

from ssm.litm_ssm import SSMLitModule
from utils.synth import PresetHelper

DEBUG = False

SEED = 999  # 875
BATCH_SIZE = 32

# subset of the NSynth dataset to use
NS_SUBSET = "valid"
RANGE_PITCH = [61 - 12, 61 + 12]

# number of in-domain predictions to render and export per checkpoints
NUM_ID_SAMPLES = 64

# number of nsynth prediction to render and export per checkpoints
# and each of the 11 instrument family
NUM_NS_SAMPLES = 4

# dict {ckpt_name: abbreviation} of the estimator net's checkpoints from which to predict presets
CKPTS = {
    "diva_p": "diva_ssm_mn20_hc_v1_loss_p_e259_loss2.0963.ckpt",
    "diva_map": "diva_ssm_mn20_hc_v1_mix_a_p_e529_loss1.8077.ckpt",
    "diva_spa": "diva_ssm_mn20_hc_v1_switch_p_a_e459_loss1.7167.ckpt",
    "dexed_p": "dexed_ssm_mn20_hc_v1_loss_p_e539_loss1.7647.ckpt",
    "dexed_map": "dexed_ssm_mn20_hc_v1_mix_a_p_e249_loss1.6251.ckpt",
    "dexed_spa": "dexed_ssm_mn20_hc_v1_switch_p_a_e229_loss1.6356.ckpt",
}

NS_FAMILIES = [
    "string",
    "bass",
    "brass",
    "organ",
    "reed",
    "vocal",
    "flute",
    "mallet",
    "guitar",
    "keyboard",
    "synth_lead",
]

if __name__ == "__main__":
    load_dotenv()

    flag_target = {"diva": False, "dexed": False}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(os.environ["PROJECT_ROOT"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_path = project_root / "results" / f"ssm_qual_eval_{timestamp}"
    export_path.mkdir(parents=True, exist_ok=True)

    ns_export_path = export_path / "nsynth"
    ns_export_path.mkdir(parents=True, exist_ok=True)

    for model_name, ckpt in CKPTS.items():
        synth = ckpt.split("_", 1)[0]
        id_export_path = export_path / synth
        id_export_path.mkdir(parents=True, exist_ok=True)

        # preset helper
        excluded_params = OmegaConf.load(project_root / "configs" / "export" / "synth" / f"{synth}.yaml")[
            "parameters_to_exclude_str"
        ]
        p_helper = PresetHelper(synth, excluded_params)

        # in-domain dataset and dataloader
        id_dataset = SynthDatasetPkl(
            project_root / "data" / "datasets" / "eval" / f"{synth}_mn20_hc_v1",
            split="test",
            has_mel=True,
            mel_norm="min_max",
        )
        data_cfg = id_dataset.configs_dict
        gen = torch.Generator().manual_seed(SEED)
        id_indices = torch.randperm(len(id_dataset), generator=gen)[:NUM_ID_SAMPLES]
        id_loader = DataLoader(
            Subset(id_dataset, id_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # model
        estimator = EstimatorNet(p_helper)
        synth_proxy_cfg = OmegaConf.load(
            project_root / "configs" / "ssm" / "train" / "settings" / "default.yaml"
        )["synth_proxy"]
        if model_name == f"{synth}_p":
            synth_proxy = None
        else:
            synth_proxy = hydra.utils.instantiate(  # for loading ckpt...
                synth_proxy_cfg, out_features=id_dataset.embedding_dim, preset_helper=p_helper
            )

        model = SSMLitModule.load_from_checkpoint(
            checkpoint_path=project_root / "checkpoints" / f"{ckpt}",
            preset_helper=p_helper,
            estimator=estimator,
            synth_proxy=synth_proxy,
            opt_cfg={"optimizer_kwargs": {"lr": None}, "num_warmup_steps": 0},
        )
        model.dataset_cfg = data_cfg
        model.eval()
        model.to(device)

        # in-domain predictions
        for i, batch in enumerate(id_loader):
            p, a, m = batch
            with torch.no_grad():
                p_hat = model(m.to(device))

            audio_hat = model._render_batch_audio(p_hat)
            audio_target = model._render_batch_audio(p)

            if not DEBUG:
                for j, (pred, target) in enumerate(zip(audio_hat, audio_target)):
                    wavfile.write(
                        id_export_path / f"{i * BATCH_SIZE + j}_{model_name}.wav",
                        data_cfg["sample_rate"],
                        pred.cpu().numpy().T,
                    )
                    if not flag_target[synth]:
                        wavfile.write(
                            id_export_path / f"{i * BATCH_SIZE + j}_0.wav",
                            data_cfg["sample_rate"],
                            target.cpu().numpy().T,
                        )

        # nsynth datasets
        for fam in NS_FAMILIES:
            ns_dataset = NSynthDataset(
                root=project_root / "data" / "datasets" / "eval",
                subset=NS_SUBSET,
                return_mel=True,
                mel_kwargs={**id_dataset.mel_cfg, **{"sr": data_cfg["sample_rate"]}},
                mel_stats=torch.load(id_dataset.path_to_dataset / "stats_train.pkl"),
                mel_norm=id_dataset.mel_norm,
                audio_length=data_cfg["render_duration_in_sec"],
                families=fam,
            )
            gen = torch.Generator().manual_seed(SEED + 1)
            id_indices = torch.randperm(len(ns_dataset), generator=gen)
            ns_loader = DataLoader(
                Subset(ns_dataset, id_indices),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )

            seen_samples = []

            # nsynth predictions
            for i, batch in enumerate(ns_loader):
                a, idx, m = batch
                attrs = ns_dataset.get_attrs(idx)

                for j, attr in enumerate(attrs):
                    if attr["instrument"] in seen_samples or not (
                        RANGE_PITCH[0] <= attr["pitch"] <= RANGE_PITCH[1]
                    ):
                        continue

                    with torch.no_grad():
                        p_hat = model(m[j].unsqueeze(0).to(device))

                    pred = model._render_batch_audio(p_hat).squeeze(0)

                    wavfile.write(
                        ns_export_path / f"{attr['note_str']}_{model_name}.wav",
                        data_cfg["sample_rate"],
                        pred.cpu().numpy().T,
                    )
                    if not flag_target[synth]:
                        wavfile.write(
                            ns_export_path / f"{attr['note_str']}_0.wav",
                            data_cfg["sample_rate"],
                            a[j].cpu().numpy().T,
                        )

                    seen_samples.append(attr["instrument"])
                    if len(seen_samples) >= NUM_NS_SAMPLES:
                        break

        flag_target[synth] = True

        print("")
