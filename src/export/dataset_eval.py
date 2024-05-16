"""
Script to generate a dataset of hand-crafted presets given a existing dataset of presets stored in a json file 
Only the synthesizer parameters used during training (i.e., for generating the training dataset) are 
considered; the remaining parameters are set to their default values and excluded, and
silent and duplicate presets are removed.
"""

import json
import os
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.io import wavfile

from models import audio as audio_models
from utils.evaluation import ProcessEvalPresets, RenderPreset
from utils.synth import PresetHelper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()
DATASETS_FOLDER = Path(os.environ["PROJECT_ROOT"]) / "data" / "datasets"


def keystoint(x):
    try:
        return {int(k): v for k, v in x.items()}
    except ValueError:
        return x


def generate_eval_dataset(
    path_to_json_dataset: Union[str, Path],
    path_to_train_cfg: Union[str, Path],
    export_path: Union[str, Path],
    export_audio: bool = True,
    batch_size: int = 128,
    num_workers: int = 8,
) -> None:
    """
    Function used to generate a dataset of hand-crafted presets
    given a existing dataset of hand-crafted presets stored in a json file. The dataset should be a
    a dictionary formatted as follows:
    {preset_id: {'meta': {...}, 'parameters': {'param_name': param_val}}.

    Only the synthesizer parameters used during training (i.e., for generating the training dataset) are
    considered; the remaining parameters are set to their default values and excluded, and
    silent and duplicate presets are removed.

    Args:
        path_to_json_dataset (Union[str, Path]): Path to the json file containing the presets to be evaluated.
        path_to_train_cfg (Union[str, Path]): Path to the configuration file used for training.
        export_path (Union[str, Path]): Path where the modified dataset will be saved.
        export_audio (bool): Whether or not to save the audio. (Defaults: True).
        batch_size (int): Number of examples to process as once when creating the dataset. (Defaults: 128).
        num_workers (int): Number of workers to use when loading examples. (Defaults: 8).
    """
    print(f"Using device: {DEVICE}")
    ### Load the train config file and create export directory
    path_to_train_cfg = Path(path_to_train_cfg)
    if not path_to_train_cfg.exists():
        raise FileNotFoundError(f"Train dataset config file at {path_to_train_cfg} not found.")
    print(f"Loading train dataset config from {path_to_train_cfg}")
    with open(path_to_train_cfg, "rb") as f:
        train_cfg = torch.load(f)

    print(f"Training Dataset Config: {train_cfg}")

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
        if k not in train_cfg:
            raise ValueError(f"Missing required key '{k}' in train config.")

    export_path = Path(export_path)
    export_path.mkdir(exist_ok=True, parents=True)

    if export_audio:
        (export_path / "audio").mkdir(exist_ok=True, parents=True)

    ### Process the json dict of hand-crafted presets used to generate the evaluation dataset
    preset_helper = PresetHelper(
        synth_name=train_cfg["synth"],
        parameters_to_exclude=train_cfg["params_to_exclude"],
    )

    processor = ProcessEvalPresets(
        preset_helper=preset_helper,
        render_duration_in_sec=train_cfg["render_duration_in_sec"],
        midi_note=train_cfg["midi_note"],
        midi_velocity=train_cfg["midi_velocity"],
        midi_duration_in_sec=train_cfg["midi_duration_in_sec"],
        rms_range=(0.01, 1.0),
        sample_rate=train_cfg["sample_rate"],
    )
    print(f"Processing presets from {path_to_json_dataset}...")
    path_to_json_dataset = Path(path_to_json_dataset)
    if not path_to_json_dataset.exists():
        raise FileNotFoundError(f"Dataset file at {path_to_json_dataset} not found.")

    with open(path_to_json_dataset, "r", encoding="utf-8") as f:
        presets_dict = json.load(f, object_hook=keystoint)
    presets, selected_presets, removed_presets = processor(presets_dict)

    print(f"Number of remaining presets: {len(presets)}/{len(presets_dict)}")
    print(f"Number of removed presets: {len(removed_presets)}/{len(presets_dict)}")

    ### Generate the evaluation dataset
    print(f"Generating evaluation dataset at {export_path}")
    audio_fe = getattr(audio_models, train_cfg["audio_fe"])()
    audio_fe.to(DEVICE)
    audio_fe.eval()

    dataset = RenderPreset(
        presets=presets,
        preset_helper=preset_helper,
        render_duration_in_sec=train_cfg["render_duration_in_sec"],
        midi_note=train_cfg["midi_note"],
        midi_velocity=train_cfg["midi_velocity"],
        midi_duration_in_sec=train_cfg["midi_duration_in_sec"],
        sample_rate=train_cfg["sample_rate"],
        rms_range=(0.01, 1.0),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    audio_embeddings = torch.empty((len(presets), audio_fe.out_features), device="cpu")
    synth_parameters = torch.empty((len(dataset), preset_helper.num_used_parameters), device="cpu")

    pbar = tqdm(
        loader,
        total=len(presets) // batch_size,
        dynamic_ncols=True,
    )

    for i, (params, audio) in enumerate(pbar):
        audio = audio.to(DEVICE)
        with torch.no_grad():
            audio_emb = audio_fe(audio)

        audio_embeddings[i * batch_size : (i + 1) * batch_size] = audio_emb.cpu()
        synth_parameters[i * batch_size : (i + 1) * batch_size] = params.cpu()

        if export_audio:
            for j, sample in enumerate(audio):
                sample = sample.cpu().numpy()
                wavfile.write(
                    export_path / "audio" / f"{i*batch_size+j}.wav",
                    train_cfg["sample_rate"],
                    sample.T,
                )

    ### Dump eval dataset to disk
    print(f"Saving evaluation dataset to {export_path}...")
    configs_dict = {
        "synth": preset_helper.synth_name,
        "params_to_exclude": preset_helper.excl_parameters_str,
        "num_used_params": preset_helper.num_used_parameters,
        "dataset_size": len(presets),
        "render_duration_in_sec": processor.renderer.render_duration_in_sec,
        "midi_note": processor.renderer.midi_note,
        "midi_velocity": processor.renderer.midi_velocity,
        "midi_duration_in_sec": processor.renderer.midi_duration_in_sec,
        "audio_fe": train_cfg["audio_fe"],
        "sample_rate": audio_fe.sample_rate,
        "num_outputs": audio_fe.out_features,
    }

    with open(export_path / "presets.json", "w", encoding="utf-8") as f:
        json.dump(selected_presets, f)
    with open(export_path / "removed_presets.json", "w", encoding="utf-8") as f:
        json.dump(removed_presets, f)
    with open(export_path / "synth_parameters.pkl", "wb") as f:
        torch.save(synth_parameters, f)
    with open(export_path / "audio_embeddings.pkl", "wb") as f:
        torch.save(audio_embeddings, f)
    with open(export_path / "configs.pkl", "wb") as f:
        torch.save(configs_dict, f)


if __name__ == "__main__":

    SYNTH = "diva"
    VERSION = 1  # Version number of the training dataset

    BATCH_SIZE = 128
    NUM_WORKERS = 8
    EXPORT_AUDIO = True

    # path to the dataset used for training and to the json dataset
    PATH_TO_TRAIN_DATASET = DATASETS_FOLDER / f"{SYNTH}_mn04_size=10240000_seed=500_train_v{VERSION}"
    PATH_TO_JSON_DATASET = DATASETS_FOLDER / "eval" / "json_datasets" / f"{SYNTH}_dataset.json"

    export_path = DATASETS_FOLDER / "eval" / f"{SYNTH}_mn04_eval_v{VERSION}"
    if not export_path.exists():
        export_path.mkdir(exist_ok=True, parents=True)

    generate_eval_dataset(
        path_to_json_dataset=PATH_TO_JSON_DATASET,
        path_to_train_cfg=PATH_TO_TRAIN_DATASET / "configs.pkl",
        export_path=export_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        export_audio=EXPORT_AUDIO,
    )

    print("Done!")
