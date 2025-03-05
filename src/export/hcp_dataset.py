"""
Script to generate a dataset of hand-crafted presets given a existing dataset of presets stored in a json file 
Only the synthesizer parameters used during training (i.e., for generating the training dataset) are 
considered; the remaining parameters are set to their default values and excluded, and
silent and duplicate presets are removed.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from scipy.io import wavfile

from models import audio as audio_models
from data.datasets import SynthDatasetTensor
from utils.synth import PresetHelper, ProcessHCPresets
from utils.audio import MelSTFT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()
DATASETS_FOLDER = Path(os.environ["PROJECT_ROOT"]) / "data" / "datasets"

SEED = 42  # for reproducibilitys


def keystoint(x):
    try:
        return {int(k): v for k, v in x.items()}
    except ValueError:
        return x


def export_hc_dataset(
    path_to_json_dataset: Union[str, Path],
    path_to_render_cfg: Union[str, Path],
    export_path: Union[str, Path],
    train_test_split: Optional[List[float]] = None,
    export_mel: bool = False,
    mel_kwargs: Optional[dict] = None,
    export_audio: bool = True,
    batch_size: int = 128,
    num_workers: int = 8,
) -> None:
    """
    Function used to generate a dataset of hand-crafted presets
    given a existing dataset of hand-crafted presets stored in a json file. The dataset should be a
    a dictionary formatted as follows:
    {preset_id: {'meta': {...}, 'parameters': {'param_name': param_val}}.

    Only the synthesizer parameters that are not excluded in the render_cfg (meaning that were used to train the synth proxy) are
    considered; the remaining parameters are set to their default values and excluded, and
    silent and duplicate presets are removed.

    Args:
        path_to_json_dataset (Union[str, Path]): Path to the json file containing the presets to be evaluated.
        path_to_render_cfg (Union[str, Path]): Path to the configuration file to use for rendering.
        In most cases, this should be the same one as used to train the synth proxy.
        export_path (Union[str, Path]): Path where the modified dataset will be saved.
        train_test_split (Optional[List[float]]): Optional list of [train, test] split ratios. (Defaults: None).
        export_mel (bool): Whether or not to save the mel spectrogram of the rendered presets. (Defaults: False).
        mel_kwargs (dict): Configuration dictionary for the mel spectrogram. Required if export_mel is True. (Defaults: None).
        export_audio (bool): Whether or not to save the audio.
        Note that if train_test_split is not None, audio will be saved for the test set only. (Defaults: True).
        batch_size (int): Number of examples to process as once when creating the dataset. (Defaults: 128).
        num_workers (int): Number of workers to use when loading examples. (Defaults: 8).
    """
    print(f"Using device: {DEVICE}")
    ### Load the render config file and create export directory
    path_to_render_cfg = Path(path_to_render_cfg)
    if not path_to_render_cfg.exists():
        raise FileNotFoundError(f"Rendering config file at {path_to_render_cfg} not found.")
    print(f"Loading rendering config from {path_to_render_cfg}")
    with open(path_to_render_cfg, "rb") as f:
        render_cfg = torch.load(f)

    print(f"Rendering Config: {render_cfg}")

    required_keys = [
        "synth",
        "params_to_exclude",
        "num_used_params",
        "sample_rate",
        "render_duration_in_sec",
        "midi_note",
        "midi_velocity",
        "midi_duration_in_sec",
        "audio_fe",
    ]

    for k in required_keys:
        if k not in render_cfg:
            raise ValueError(f"Missing required key '{k}' in rendering config.")

    export_path = Path(export_path)
    export_path.mkdir(exist_ok=True, parents=True)

    if export_audio:
        (export_path / "audio").mkdir(exist_ok=True, parents=True)

    ### Process the json dict of hand-crafted presets used to generate the evaluation dataset
    preset_helper = PresetHelper(
        synth_name=render_cfg["synth"],
        parameters_to_exclude=render_cfg["params_to_exclude"],
    )

    processor = ProcessHCPresets(
        preset_helper=preset_helper,
        render_duration_in_sec=render_cfg["render_duration_in_sec"],
        midi_note=render_cfg["midi_note"],
        midi_velocity=render_cfg["midi_velocity"],
        midi_duration_in_sec=render_cfg["midi_duration_in_sec"],
        rms_range=(0.01, 1.0),
        sample_rate=render_cfg["sample_rate"],
    )

    print(f"Processing presets from {path_to_json_dataset}...")
    path_to_json_dataset = Path(path_to_json_dataset)
    if not path_to_json_dataset.exists():
        raise FileNotFoundError(f"Dataset file at {path_to_json_dataset} not found.")

    with open(path_to_json_dataset, "r", encoding="utf-8") as f:
        presets_dict = json.load(f, object_hook=keystoint)
        # # only keep 100 first for debug
        # presets_dict = {k: v for k, v in presets_dict.items() if k < 100}
    presets, selected_presets, removed_presets = processor(presets_dict)

    print(f"Number of remaining presets: {len(presets)}/{len(presets_dict)}")
    print(f"Number of removed presets: {len(removed_presets)}/{len(presets_dict)}")

    ### Generate the hand-crafted presets dataset
    print(f"Generating hand-crafted presets dataset at {export_path}")
    audio_fe = getattr(audio_models, render_cfg["audio_fe"])()
    audio_fe.to(DEVICE)
    audio_fe.eval()

    dataset = SynthDatasetTensor(
        presets=presets,
        preset_helper=preset_helper,
        render_duration_in_sec=render_cfg["render_duration_in_sec"],
        midi_note=render_cfg["midi_note"],
        midi_velocity=render_cfg["midi_velocity"],
        midi_duration_in_sec=render_cfg["midi_duration_in_sec"],
        sample_rate=render_cfg["sample_rate"],
        rms_range=(0.01, 1.0),
    )

    if train_test_split is not None:
        assert len(train_test_split) == 2
        print(f"Using train/test split ratios: {train_test_split}")
        gen = torch.Generator().manual_seed(SEED)
        datasets = random_split(dataset, train_test_split, generator=gen)
        splits = ["train", "test"]
    else:
        datasets = [dataset]
        splits = ["all"]

    for split, d in zip(splits, datasets):
        print(f"Processing {len(d)} presets...")
        d_size = len(d)
        loader = DataLoader(d, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        audio_embeddings = torch.empty((d_size, audio_fe.out_features), device="cpu")
        synth_parameters = torch.empty((d_size, preset_helper.num_used_parameters), device="cpu")

        if export_mel:
            assert mel_kwargs is not None, "mel_cfg must be provided if export_mel is True"
            mel = MelSTFT(sr=render_cfg["sample_rate"], **mel_kwargs).to(DEVICE)
            specgrams = []

        pbar = tqdm(
            loader,
            total=d_size // batch_size,
            dynamic_ncols=True,
        )

        for i, (params, audio) in enumerate(pbar):
            audio = audio.to(DEVICE)
            with torch.no_grad():
                audio_emb = audio_fe(audio)

            audio_embeddings[i * batch_size : (i + 1) * batch_size] = audio_emb.cpu()
            synth_parameters[i * batch_size : (i + 1) * batch_size] = params.cpu()

            if export_mel:
                specgrams.append(mel(audio).cpu())

            if export_audio and (train_test_split is None or split == "test"):
                for j, sample in enumerate(audio):
                    sample = sample.cpu().numpy()
                    wavfile.write(
                        export_path / "audio" / f"{i*batch_size+j}.wav",
                        render_cfg["sample_rate"],
                        sample.T,
                    )

        suffix = "_" + split if split != "all" else ""
        with open(export_path / f"synth_parameters{suffix}.pkl", "wb") as f:
            torch.save(synth_parameters, f)
        with open(export_path / f"audio_embeddings{suffix}.pkl", "wb") as f:
            torch.save(audio_embeddings, f)

        if export_mel:
            specgrams = torch.concat(specgrams)
            with open(export_path / f"specgrams{suffix}.pkl", "wb") as f:
                torch.save(specgrams, f)
            if splits != "test":  # don't get statistics from test set
                # compute statistics
                stats = {
                    "min": specgrams.min(),
                    "max": specgrams.max(),
                    "mean": specgrams.mean(),
                    "std": specgrams.std(),
                }
                with open(export_path / f"stats{suffix}.pkl", "wb") as f:
                    torch.save(stats, f)

    ### Dump eval dataset to disk
    print(f"Saving hand-crafted preset dataset to {export_path}...")
    configs_dict = {
        "synth": preset_helper.synth_name,
        "params_to_exclude": preset_helper.excl_parameters_str,
        "num_used_params": preset_helper.num_used_parameters,
        "dataset_size": len(presets),
        "train_test_split": train_test_split,
        "train_size": len(datasets[0]) if train_test_split is not None else None,
        "test_size": len(datasets[1]) if train_test_split is not None else None,
        "render_duration_in_sec": processor.renderer.render_duration_in_sec,
        "midi_note": processor.renderer.midi_note,
        "midi_velocity": processor.renderer.midi_velocity,
        "midi_duration_in_sec": processor.renderer.midi_duration_in_sec,
        "audio_fe": render_cfg["audio_fe"],
        "sample_rate": audio_fe.sample_rate,
        "num_outputs": audio_fe.out_features,
        "mel_cfg": mel_kwargs,
    }

    with open(export_path / "presets.json", "w", encoding="utf-8") as f:
        json.dump(selected_presets, f)
    with open(export_path / "removed_presets.json", "w", encoding="utf-8") as f:
        json.dump(removed_presets, f)
    with open(export_path / "configs.pkl", "wb") as f:
        torch.save(configs_dict, f)


if __name__ == "__main__":

    SYNTH = "diva"

    TAKE_RENDER_CFG_FROM = f"{SYNTH}_mn20_mel_size=2560000_seed=900_ssm_train_v1"
    EXPORT_PATH = f"{SYNTH}_mn20_hc_v1"

    MEL_KWARGS = {
        "n_mels": 128,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 512,
        "fast_normalization": False,
    }

    TRAIN_TEST_SPLIT = [0.9, 0.1]
    EXPORT_MEL = True
    EXPORT_AUDIO = True
    BATCH_SIZE = 128
    NUM_WORKERS = 8

    ##### Generate the evaluation dataset
    EXPORT_PATH = DATASETS_FOLDER / "eval" / EXPORT_PATH
    if not EXPORT_PATH.exists():
        EXPORT_PATH.mkdir(exist_ok=True, parents=True)

    export_hc_dataset(
        path_to_json_dataset=DATASETS_FOLDER / "eval" / "json_datasets" / f"{SYNTH}_dataset.json",
        path_to_render_cfg=DATASETS_FOLDER / TAKE_RENDER_CFG_FROM / "configs.pkl",
        export_path=EXPORT_PATH,
        train_test_split=TRAIN_TEST_SPLIT,
        export_mel=EXPORT_MEL,
        mel_kwargs=MEL_KWARGS,
        export_audio=EXPORT_AUDIO,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print("\nDone!")
