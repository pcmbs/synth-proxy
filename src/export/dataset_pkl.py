# pylint: disable=W1203
"""
Module to be run from the command line (on Linux) used to generate audio embeddings
from a given audio model together with the corresponding synth parameters. 

See `configs/export/dataset_pkl.yaml` for more details.

The results are exported to
`<project-root>/<export-relative-path>/<synth>_<audio_fe>_<dataset-size>_<seed-offset>_pkl_<tag>`

It is possible to interrupt the process at the end of the current iteration by pressing Ctrl+Z (SIGSTP). 
Doing so will export a resume state file that will be used to resume the process. 

"""
import logging
from pathlib import Path
import signal
import sys

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from scipy.io import wavfile

from data.datasets import SynthDataset
from models import audio as audio_models
from utils.synth import PresetHelper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

RESUME_STATE_FILE = "resume_state.pkl"

# flag to track if keyboard interrupt has been detected
# which is used to stop export at the end of the current batch.
is_interrupted = False  # pylint: disable=C0103


def graceful_shutdown(signum, frame) -> None:
    """Handler for signal to interrupt the export process."""
    global is_interrupted  # pylint: disable=W0603
    print(f"Received signal {signum}, the export will be aborted at the end of the current iteration...")
    is_interrupted = True


log = logging.getLogger(__name__)

# Set the SIGSTP signal handler
signal.signal(signal.SIGTSTP, graceful_shutdown)
signal.signal(signal.SIGUSR1, graceful_shutdown)  # For HPC
signal.signal(signal.SIGTERM, graceful_shutdown)  # For HPC
signal.signal(signal.SIGINT, graceful_shutdown)  # For HPC


@hydra.main(version_base=None, config_path="../../configs/export", config_name="dataset_pkl")
def export_dataset_pkl(cfg: DictConfig) -> None:

    is_subset = False
    start_index = cfg.start_index
    end_index = cfg.end_index
    offset_index = 0  # for indexing loaded tensor when resuming

    # instantiate audio model, preset helper and dataset
    audio_fe = getattr(audio_models, cfg.audio_fe)()
    audio_fe.to(DEVICE)
    audio_fe.eval()

    p_helper = PresetHelper(
        synth_name=cfg.synth.name, parameters_to_exclude=cfg.synth.parameters_to_exclude_str
    )

    dataset = SynthDataset(
        preset_helper=p_helper,
        dataset_size=cfg.dataset_size,
        seed_offset=cfg.seed_offset,
        sample_rate=audio_fe.sample_rate,
        render_duration_in_sec=cfg.render_duration_in_sec,
        midi_note=cfg.midi_note,
        midi_velocity=cfg.midi_velocity,
        midi_duration_in_sec=cfg.midi_duration_in_sec,
    )

    if (Path.cwd() / "resume_state.pkl").exists():
        with open(Path.cwd() / "resume_state.pkl", "rb") as f:
            saved_data = torch.load(f)
            resume_index = saved_data["resume_index"]
            audio_embeddings = saved_data["audio_embeddings"]
            synth_parameters = saved_data["synth_parameters"]

        offset_index = resume_index - start_index  # update offset index
        start_index = resume_index  # overwrite start index to resume
        log.info(
            f"resume_state.pkl found, resuming from sample {start_index}.\n"
            f"Number of remaining samples to be "
            f"generated: {(cfg.end_index - start_index)}",
        )

    else:
        # don't put subset flag for interrupted export
        if start_index != 0 or end_index != cfg.dataset_size:
            is_subset = True

        if cfg.export_audio != 0:
            audio_path = Path.cwd() / "audio"
            audio_path.mkdir(parents=True, exist_ok=True)

        configs_dict = {
            "synth": cfg.synth.name,
            "params_to_exclude": cfg.synth.parameters_to_exclude_str,
            "num_used_params": dataset.num_used_parameters,
            "dataset_size": cfg.dataset_size,
            "seed_offset": cfg.seed_offset,
            "render_duration_in_sec": cfg.render_duration_in_sec,
            "midi_note": cfg.midi_note,
            "midi_velocity": cfg.midi_velocity,
            "midi_duration_in_sec": cfg.midi_duration_in_sec,
            "audio_fe": cfg.audio_fe,
            "sample_rate": audio_fe.sample_rate,
            "num_outputs": audio_fe.out_features,
        }

        log.info("Configs")
        for k, v in configs_dict.items():
            log.info(f"{k}: {v}")

        with open(Path.cwd() / "configs.pkl", "wb") as f:
            torch.save(configs_dict, f)

        log.info("\nSynth parameters description")
        for i, param in enumerate(p_helper.used_parameters):
            log.info(f"{str(i) + ':':<4} {param}")

        audio_embeddings = torch.empty((end_index - start_index, audio_fe.out_features), device="cpu")
        synth_parameters = torch.empty((end_index - start_index, dataset.num_used_parameters), device="cpu")

    if start_index != 0 or end_index != cfg.dataset_size:
        dataset = Subset(dataset, range(start_index, end_index))
        log.info(f"\nExporting samples {start_index} to {end_index-1}...")
    else:
        log.info(f"\nExporting all {cfg.dataset_size} samples in the dataset...")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    pbar = tqdm(
        dataloader,
        total=(end_index - start_index) // cfg.batch_size,
        dynamic_ncols=True,
    )

    for i, (params, audio, _) in enumerate(pbar):
        if cfg.synth.name in ["diva", "dexed"]:  # to keep track of progress since they spam the terminal
            log.info(f"Generating batch {i}/{(end_index - start_index) // cfg.batch_size - 1}")
        slice_start = i * cfg.batch_size + offset_index
        slice_end = slice_start + params.shape[0]  # instead of batch size since drop_last=False

        audio = audio.to(DEVICE)
        with torch.no_grad():
            audio_emb = audio_fe(audio)

        audio_embeddings[slice_start:slice_end] = audio_emb.cpu()
        synth_parameters[slice_start:slice_end] = params.cpu()

        if (cfg.export_audio > start_index + slice_start) or (cfg.export_audio == -1):
            for j, sample in enumerate(audio):
                if (cfg.export_audio > start_index + slice_start + j) or (cfg.export_audio == -1):
                    sample = sample.cpu().numpy()
                    wavfile.write(
                        audio_path / f"{start_index + slice_start+j}.wav",
                        audio_fe.sample_rate,
                        sample.T,
                    )

        if is_interrupted:
            new_starting_index = start_index + slice_end
            log.info(
                f"Finished generating samples for the current iteration "
                f"(samples {start_index + slice_start} to {start_index + slice_end - 1}), "
                f"interrupting generation..."
            )
            break

    if is_interrupted:
        log.info("Saving resume_state.pkl file...")
        with open(Path.cwd() / "resume_state.pkl", "wb") as f:
            saved_data = {
                "resume_index": new_starting_index,
                "audio_embeddings": audio_embeddings,
                "synth_parameters": synth_parameters,
            }
            torch.save(saved_data, f)
        log.info("Exiting Python program...")
        sys.exit(0)

    # Remove the resume state file if export completes successfully
    if (Path.cwd() / "resume_state.pkl").exists():
        (Path.cwd() / "resume_state.pkl").unlink()

    # Save the audio_embeddings and synth_parameters
    # take start_index from the configs since was changed if interrupted
    filename_suffix = f"_{cfg.start_index}_{cfg.end_index-1}" if is_subset else ""

    log.info(
        f"All samples generated, "
        f"saving audio_embeddings{filename_suffix}.pkl and synth_parameters{filename_suffix}.pkl..."
    )

    with open(Path.cwd() / f"audio_embeddings{filename_suffix}.pkl", "wb") as f:
        torch.save(audio_embeddings, f)

    with open(Path.cwd() / f"synth_parameters{filename_suffix}.pkl", "wb") as f:
        torch.save(synth_parameters, f)

    log.info("Export completed successfully!")


if __name__ == "__main__":
    export_dataset_pkl()  # pylint: disable=E1120:no-value-for-parameter̦
