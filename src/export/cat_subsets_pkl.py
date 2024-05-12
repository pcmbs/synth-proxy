# pylint: disable=W1203
"""
Module used to concatenate subsets of a dataset from several .pkl file
located in an unique folder `<export_relative_path>/<folder_name>` (see configs) in the project. 

The subsets are identified by their starting and ending indices, i.e., 
of the form `audio_embeddings_<start>_<end>.pkl` and `synth_parameters_<start>_<end>.pkl`,
which will be deleted after the concatenation if delete_subsets=True in the config. 

The concatenated dataset will be saved in the same folder as the original dataset
under the name `audio_embeddings.pkl` and `synth_parameters.pkl`.

A AssertionError will be raised if:
- the subsets do not cover the entire dataset (checked both for audio_embeddings and synth_parameters)
- the number of dimensions in the subsets does not match the number of dimensions in the original dataset
 
Corresponding config file can be found under `configs/export/cat_dataset_pkl.yaml`
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch

log = logging.getLogger(__name__)


def check_concatenated_slices_coverage(file_info_list, dataset_size):
    concatenated_slices = []
    for file_info in file_info_list:
        start, end = file_info["start"], file_info["end"]
        concatenated_slices.extend(range(start, end + 1))
    concatenated_slices = sorted(concatenated_slices)
    return concatenated_slices == list(range(dataset_size))


@hydra.main(version_base=None, config_path="../../configs/export", config_name="cat_dataset_pkl")
def cat_dataset_pkl(cfg: DictConfig) -> None:

    log.info(f"Concatenating subsets in {Path.cwd()}...")
    # open configs.pkl to get the dataset size and the embedding size (in features)
    with open(Path.cwd() / "configs.pkl", "rb") as f:
        configs = torch.load(f)

    dataset_size = configs["dataset_size"]
    out_features = configs["num_outputs"]
    in_features = configs["num_used_params"]

    # get the all filenames of the subsets (incl start and end indices)
    pkl_filename = {
        "audio_emb": [],
        "synth_parameters": [],
    }

    for pkl_file in sorted(Path.cwd().glob("*.pkl")):
        if pkl_file.stem.startswith("audio_embeddings_"):
            pkl_filename["audio_emb"].append(
                {
                    "filename": pkl_file.name,
                    "start": int(pkl_file.stem.split("_")[2]),
                    "end": int(pkl_file.stem.split("_")[3]),
                }
            )
        elif pkl_file.stem.startswith("synth_parameters_"):
            pkl_filename["synth_parameters"].append(
                {
                    "filename": pkl_file.name,
                    "start": int(pkl_file.stem.split("_")[2]),
                    "end": int(pkl_file.stem.split("_")[3]),
                }
            )

    # Check that the subsets cover the entire dataset
    if not check_concatenated_slices_coverage(pkl_filename["audio_emb"], dataset_size):
        raise AssertionError("Concatenated audio embedding slices are not the same as the dataset size.")

    if not check_concatenated_slices_coverage(pkl_filename["synth_parameters"], dataset_size):
        raise AssertionError(
            "Concatenated synthesizer parameters slices are not the same as the dataset size."
        )

    # Concatenate the subsets
    log.info(
        f"Found {len(pkl_filename['audio_emb'])} subsets: "
        f"\n {pkl_filename['audio_emb']}\n {pkl_filename['synth_parameters']}\n"
    )

    audio_emb = torch.cat(
        [torch.load(Path.cwd() / pkl_file["filename"]) for pkl_file in pkl_filename["audio_emb"]]
    )
    synth_parameters = torch.cat(
        [torch.load(Path.cwd() / pkl_file["filename"]) for pkl_file in pkl_filename["synth_parameters"]]
    )

    if audio_emb.shape[1] != out_features:
        raise AssertionError(
            f"Number of output features (enbedding size) does not match {out_features}, got {audio_emb.shape[1]}."
        )

    if synth_parameters.shape[1] != in_features:
        raise AssertionError(
            f"Number of input features (synth params) does not match {in_features}, got {synth_parameters.shape[1]}."
        )

    log.info(
        f"\nSuccessfully concatenated dataset with shape: {audio_emb.shape=} and {synth_parameters.shape=}"
    )

    log.info(f"Saving audio embeddings to {Path.cwd() / 'audio_embeddings.pkl'}...")
    torch.save(audio_emb, Path.cwd() / "audio_embeddings.pkl")

    log.info(f"Saving audio embeddings to {Path.cwd() / 'synth_parameters.pkl'}...")
    torch.save(synth_parameters, Path.cwd() / "synth_parameters.pkl")

    if cfg.delete_subsets:
        log.info("Deleting original subsets...")
        for file_info in pkl_filename["audio_emb"] + pkl_filename["synth_parameters"]:
            (Path.cwd() / file_info["filename"]).unlink()

    log.info("Concatenation completed successfully!")


if __name__ == "__main__":
    cat_dataset_pkl()  # pylint: disable=E1120:no-value-for-parameter
