"""
Script showing hos to load a synth proxy and use it for inference.

The example is based on Diva, and uses the "HS Eel Pi" preset.
Here, the L1 distances of monotonic variations of a single parameter are evaluated,
some of which actually affect the sound, while others do not.
"""

import json
from pathlib import Path
import os

import torch
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from utils.inference import get_synth_proxy, get_preset_helper, clip_continuous_params, round_discrete_params

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

SYNTH = "diva"
MODEL = "tfm"

# Preset to load (should be in data/datasets/eval/json_datasets/{SYNTH}_dataset.json)
EVAL_PRESET_NAME = "HS Eel Pi"

# preset mods to evaluate
PRESET_MODS = [
    # used parameter
    {"parameter_name": "vcf1:frequency", "values": torch.linspace(0, 1.0, 10)},
    {"parameter_name": "vcf1:resonance", "values": torch.linspace(0, 1.0, 10)},
    {"parameter_name": "osc:fmmoddepth", "values": torch.linspace(0, 1, 10)},
    {"parameter_name": "osc:tune3", "values": torch.linspace(0, 1, 10)},
    {"parameter_name": "osc:noisevol", "values": torch.linspace(0, 1, 10)},
    # not used parameter
    {"parameter_name": "chrs1:depth", "values": torch.linspace(0, 1, 10)},
    {"parameter_name": "vcf1:shapemix", "values": torch.linspace(0, 1, 10)},
    {"parameter_name": "osc:digitalshape2", "values": torch.linspace(0, 1, 10)},
]

if __name__ == "__main__":

    # Load the preset helper to the given synth which contains infos about the used parameters,
    # the excluded parameters values, and acceptable ranges used during training
    p_helper = get_preset_helper(SYNTH)
    # Load the synth proxy
    synth_proxy = get_synth_proxy(MODEL, SYNTH, p_helper).to(DEVICE)

    # Load the evaluation preset
    with open(
        PROJECT_ROOT / "data" / "datasets" / "eval" / "json_datasets" / f"{SYNTH}_dataset.json",
        "r",
        encoding="utf-8",
    ) as f:
        dataset = json.load(f)

    # Find the preset to use for evaluation
    eval_preset = [v["parameters"] for v in dataset.values() if v["meta"].get("name") == EVAL_PRESET_NAME]
    if eval_preset is None:
        raise ValueError(f"Preset '{EVAL_PRESET_NAME}' not found in dataset")
    eval_preset = eval_preset[0]  # if duplicates

    # Create a variation of this preset with increasing LPF frequency
    dists = []
    for mod in PRESET_MODS:
        preset_mods = []
        for freq in mod["values"]:
            preset = eval_preset.copy()
            preset[mod["parameter_name"]] = float(freq)
            preset_mods.append(preset)

        # In our case, the presets are stored as dicts {param_name: param_val}, so we need to convert
        # them to tensor of shape (num_used_parameters,)
        # where num_used_parameters is the number of parameters used during training
        presets = []  # List to hold the tensor presets
        # iterate over all presets in the dataset and populate a fresh tensor with the
        # value of each synth parameter used during training
        for preset_dict in preset_mods:
            p = torch.zeros(p_helper.num_used_parameters)
            for param_name, param_val in preset_dict.items():
                relative_idx = p_helper.relative_idx_from_name(param_name)
                if relative_idx is not None:  # i.e., was used during training
                    p[relative_idx] = param_val

            presets.append(p)

        presets = torch.stack(presets)

        # Clip continuous numerical parameters values in the range defined by the preset helper
        presets = clip_continuous_params(presets, p_helper)

        # Round discrete parameters (num, cat, bin) values to the nearest value used during training
        presets = round_discrete_params(presets, p_helper)

        # Generate a preset embedding
        with torch.no_grad():
            embeddings = synth_proxy(presets.to(DEVICE))

        # compute L1 distance from first preset embedding
        l1_dist = torch.abs(embeddings[0] - embeddings[1:]).mean(dim=-1).cpu().numpy()
        print(f"\nL1 distances for parameter {mod['parameter_name']}:\n{l1_dist}")
        dists.append(l1_dist)

        # plot the distances
        plt.plot(mod["values"][1:], l1_dist, label=mod["parameter_name"])

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    for i, dist in enumerate(dists):
        ax.plot(
            PRESET_MODS[i]["values"][1:],
            dist,
            label=PRESET_MODS[i]["parameter_name"],
        )

    ax.set_yscale("log")
    ax.set_xlim(0.1, 1)
    ax.grid(which="both", alpha=0.5)
    ax.set_xlabel("Difference in Parameter Value")
    ax.set_ylabel("L1 Distances")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title("L1 Distances to original preset embedding")
    plt.savefig("L1_distance.png")
