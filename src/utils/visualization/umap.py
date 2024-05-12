# pylint: disable=E1120:no-value-for-parameter
import os
from pathlib import Path
from typing import Dict, List, Sequence

from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

# Matplotlib stuff
COLORS = plt.cm.Paired.colors
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["font.sans-serif"] = ["cmr10"]
plt.rcParams["mathtext.fontset"] = "stix"

MODEL_NAME_FORMAT = {
    "hn_pt": "HN-PT",
    "hn_ptgru": "HN-PTGRU",
    "hn_oh": "HN-OH",
    "mlp_oh": "MLP-OH",
    "tfm": "TFM",
}

DEXED_LABEL_COLOR_MAP = {
    "harmonic": COLORS[1],
    "percussive": COLORS[3],
    "sfx": COLORS[7],
}

EXPORT_DIR = PROJECT_ROOT / "results" / "umap_projections"


def keystoint(x):
    try:
        return {int(k): v for k, v in x.items()}
    except ValueError:
        return x


def get_labels(dataset_json: Dict, synth: str) -> List:
    assert synth in ["diva", "dexed"]
    labels_key = "labels" if synth == "dexed" else "character"
    labels = []
    for _, v in dataset_json.items():
        labels.extend(v["meta"].get(labels_key, []))
    labels = list(set(labels))
    labels.sort()
    return labels


def get_preset_ids_per_label(dataset_json: Dict, labels: List[str], synth: str) -> Dict[str, List]:
    assert synth in ["diva", "dexed"]
    labels_key = "labels" if synth == "dexed" else "character"
    preset_id = {k: [] for k in labels}
    for k, v in dataset_json.items():
        current_labels = v["meta"].get(labels_key, [])
        if synth == "dexed":
            # only get presets with a single label for dexed...
            if len(current_labels) == 1:
                preset_id[current_labels[0]].append(k)
        else:  # ...but this doesn't matter for diva
            for c in current_labels:
                preset_id[c].append(k)
    return preset_id


def get_mutexcl_diva_labels(dataset_json: Dict, labels: List[str]) -> Dict[str, List]:
    label_dict = {k: set(labels) for k in labels}
    for _, v in dataset_json.items():
        current_labels = v["meta"].get("character", [])
        if current_labels:
            for c, s in label_dict.items():
                if c in current_labels:
                    s.difference_update(current_labels)

    return {k: list(v) for k, v in label_dict.items()}


def get_diva_categories(
    dataset_json: Dict, included_categories: Sequence[str] = ("Bass", "Drums", "Keys", "Pads")
) -> Dict[str, List]:
    # How preset categories are gather:
    # Bass: Bass + Basses,
    # Drums: Drums + Drums & Percussive,
    # FX: FX,
    # Keys: Keys,
    # Leads: Leads,
    # Others: Other,
    # Pads: Pads,
    # Seq & Arp: Seq & Arp + Sequences & Arps
    # Stabs: Stabs + Plucks & Stabs
    categories = {k: [] for k in included_categories}
    # iterate over presets
    for k, v in dataset_json.items():
        # get current preset categories
        preset_categories = v["meta"].get("categories", [])

        for c in preset_categories:
            c = c.split(":")[0]
            # ignored categories: ["Other", "Seq & Arp", "Sequences & Arps"]
            if c in ["Bass", "Basses"]:
                if "Bass" in included_categories:
                    categories["Bass"].append(k)
            if c in ["Drums", "Drums & Percussive"]:
                if "Drums" in included_categories:
                    categories["Drums"].append(k)
            if c == "FX":
                if "FX" in included_categories:
                    categories["FX"].append(k)
            if c == "Keys":
                if "Keys" in included_categories:
                    categories["Keys"].append(k)
            if c == "Leads":
                if "Leads" in included_categories:
                    categories["Leads"].append(k)
            if c == "Others":
                if "Others" in included_categories:
                    categories["Others"].append(k)
            if c == "Pads":
                if "Pads" in included_categories:
                    categories["Pads"].append(k)
            if c in ["Seq & Arp", "Sequences & Arps"]:
                if "Seq & Arp" in included_categories:
                    categories["Sequences & Arps"].append(k)
            if c in ["Stabs", "Stabs & Plucks"]:
                if "Stabs" in included_categories:
                    categories["Stabs"].append(k)

    categories = {k: list(set(v)) for k, v in categories.items()}
    return categories
