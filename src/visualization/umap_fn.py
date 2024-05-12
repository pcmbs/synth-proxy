"""
Module implementing the function to compute and plots the different UMAP projections.

See `umap_projections.py` for computing the UMAP projections and saving the figures. 
"""

# pylint: disable=E1120:no-value-for-parameter
import json
from typing import Any, Dict, Sequence, Tuple

from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import umap
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


from models.lit_module import PresetEmbeddingLitModule
from models.preset import model_zoo
from utils.synth import PresetHelper
from utils.visualization.umap import (
    COLORS,
    DEXED_LABEL_COLOR_MAP,
    EXPORT_DIR,
    MODEL_NAME_FORMAT,
    PROJECT_ROOT,
    get_labels,
    get_mutexcl_diva_labels,
    get_preset_ids_per_label,
    keystoint,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def umap_diva_labels(
    models: Sequence[str],
    umap_kwargs: Dict[str, Any] = {"n_neighbors": 75, "min_dist": 0.5, "metric": "euclidean", "init": "pca"},
    labels_to_plot: Sequence[Tuple[str]] = (("Aggressive", "Soft"), ("Bright", "Dark")),
    font_size: int = 30,
    seed: int = 42,
    dataset_version: int = 1,
    batch_size: int = 512,
) -> None:
    """
    Compute and plot the UMAP projections for the given pairs of mutually exclusive Diva labels
    for the given models (i.e., preset encoders).

    Args:
        models (Sequence[str]): list of model names (preset encoder names)
        umap_kwargs (Dict): keyword arguments passed to `umap.UMAP`
        labels+to_plot (Sequence[Tuple[str]]): list of tuples of mutually exclusive Diva labels to plot.
        font_size (int): font size for the plot
        seed (int): random seed used to sample the subset of presets labeled as harmonic and UMAP
        projection.
        dataset_version (int): version of the evaluation dataset to use.
        batch_size (int): batch size used to compute compute the preset embeddings if not found.
    """
    eval_dir = PROJECT_ROOT / "data" / "datasets" / "eval" / "diva_mn04_eval_v1"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    with open(eval_dir / "presets.json", "r", encoding="utf-8") as f:
        dataset_json = json.load(f, object_hook=keystoint)

    with open(eval_dir / "audio_embeddings.pkl", "rb") as f:
        audio_embeddings = torch.load(f)

    labels = get_labels(dataset_json, synth="diva")
    print("\nDiva labels:", labels)

    mutexcl_labels = get_mutexcl_diva_labels(dataset_json, labels)

    preset_ids = get_preset_ids_per_label(dataset_json, labels, synth="diva")

    print("\nNumber of presets per label:")
    for k, v in preset_ids.items():
        print(k, len(v))

    # mutually exclusive labels that are well discriminated by the audio model
    # labels_to_plot = [("Aggressive", "Soft"), ("Bright", "Dark")]
    for i, l in enumerate(labels_to_plot):
        if l[1] not in mutexcl_labels[l[0]]:
            print(f"\n{l} is not a mutual exclusive pair!")
            print("\nAvalaible mutually exclusive labels:")
            for k, v in mutexcl_labels.items():
                print(k, v)

    reducer = umap.UMAP(**umap_kwargs, random_state=seed)

    num_rows = len(labels_to_plot)
    num_cols = len(models) + 1
    fig, ax = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 4 * num_rows), layout="constrained"
    )
    # iterate over pairs of labels
    for i, (l1, l2) in enumerate(labels_to_plot):
        # dict to store the umap embeddings
        embeddings = {}

        # Extract embeddings and concatenate
        audio_emb_l1 = audio_embeddings[preset_ids[l1]].numpy()
        audio_emb_l2 = audio_embeddings[preset_ids[l2]].numpy()
        audio_emb = np.concatenate([audio_emb_l1, audio_emb_l2])
        # Compute the preset labels' start and end indices
        sep_idx = len(audio_emb_l1)
        # Fit and transform embeddings using UMAP
        u_audio = reducer.fit_transform(audio_emb)

        embeddings["ref"] = u_audio

        # iterate over the models to evaluate
        for j, m in enumerate(models):
            print(f"Computing UMAP projections for labels ({l1}, {l2}) using {MODEL_NAME_FORMAT[m]} model...")
            if not (eval_dir / "preset_embeddings" / f"{m}_embeddings.pkl").exists():
                filename = eval_dir / "preset_embeddings" / f"{m}_embeddings.pkl"
                print(f"{filename} does not exists, generating and exporting embeddings...")
                generate_preset_embeddings(
                    synth="diva", model=m, batch_size=batch_size, dataset_version=dataset_version
                )

            with open(eval_dir / "preset_embeddings" / f"{m}_embeddings.pkl", "rb") as f:
                preset_embeddings = torch.load(f)
            # Extract preset embeddings and concatenate
            preset_emb_l1 = preset_embeddings[preset_ids[l1]].numpy()
            preset_emb_l2 = preset_embeddings[preset_ids[l2]].numpy()
            preset_emb = np.concatenate([preset_emb_l1, preset_emb_l2])
            # Sanity check
            assert audio_emb.shape[0] == preset_emb.shape[0]
            assert len(audio_emb_l1) == len(preset_emb_l1)
            assert len(audio_emb_l2) == len(preset_emb_l2)

            # Fit and transform embeddings using UMAP
            u_preset = reducer.fit_transform(preset_emb)
            embeddings[m] = u_preset.copy()

        # Plot
        for a, (source, u_emb) in zip(ax[i], embeddings.items()):
            a.scatter(
                u_emb[:sep_idx, 0],
                u_emb[:sep_idx, 1],
                label=l1,
                s=30,
                alpha=0.5,
                color=COLORS[1],
                edgecolors="none",
            )
            a.scatter(
                u_emb[sep_idx:, 0],
                u_emb[sep_idx:, 1],
                label=l2,
                s=30,
                alpha=0.5,
                color=COLORS[7],
                edgecolors="none",
            )
            # only add legend and y labels on the reference (1st column)
            if source == "ref":
                a.set_ylabel(f"{l1} vs. {l2}", fontsize=font_size)

                legend_handles = [
                    plt.Rectangle((0, 0), 2, 1, fill=True, color=COLORS[1]),
                    plt.Rectangle((0, 0), 2, 1, fill=True, color=COLORS[7]),
                ]
                legend_labels = [l1, l2]
                ax_leg = plt.axes([0, 0.5 * (1 - i), 1, 0.5], facecolor=(1, 1, 1, 0))
                ax_leg.legend(
                    legend_handles,
                    legend_labels,
                    ncol=2,
                    frameon=False,
                    loc="lower left",
                    bbox_to_anchor=(0.02, -0.075 if i == 0 else -0.1),
                    fontsize="xx-large",
                )
                ax_leg.axis("off")

            # only add title on the first row
            if i == 0:
                if source == "ref":
                    a.set_title("Reference", fontsize=font_size)
                else:
                    a.set_title(f"{MODEL_NAME_FORMAT[source]}", fontsize=font_size)

            a.set_xticks([])
            a.set_yticks([])
            a.set_frame_on(False)

    # add line to separate the rows
    ax_outer = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    line = plt.Line2D([0.1, 0.9], [0.465, 0.465], color="black", linewidth=0.5)
    ax_outer.add_line(line)
    ax_outer.axis("off")

    # add a bit more space between the rows
    fig.get_layout_engine().set(hspace=0.18)

    plt.savefig(EXPORT_DIR / "umap_diva_labels.pdf", bbox_inches="tight")


def umap_dexed_labels(
    models: Sequence[str],
    umap_kwargs: Dict[str, Any] = {"n_neighbors": 75, "min_dist": 0.99, "metric": "euclidean", "init": "pca"},
    harmonic_subset_size: int = 2000,
    font_size: int = 30,
    seed: int = 42,
    dataset_version: int = 1,
    batch_size: int = 512,
) -> None:
    """
    Compute and plot the UMAP projections of the Dexed labels (harmonic, perc, sfx)
    for the given models (i.e., preset encoders).

    Args:
        models (Sequence[str]): list of model names (preset encoder names)
        umap_kwargs (Dict): keyword arguments passed to `umap.UMAP`
        harmonic_subset_size (int): number of presets labeled as harmonic to use.
        font_size (int): font size for the plot
        seed (int): random seed used to sample the subset of presets labeled as harmonic and UMAP
        projection.
        dataset_version (int): version of the evaluation dataset to use.
        batch_size (int): batch size used to compute compute the preset embeddings if not found.

    """
    EVAL_DIR = PROJECT_ROOT / "data" / "datasets" / "eval" / f"dexed_mn04_eval_v{dataset_version}"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    with open(EVAL_DIR / "presets.json", "r", encoding="utf-8") as f:
        dataset_json = json.load(f, object_hook=keystoint)

    with open(EVAL_DIR / "audio_embeddings.pkl", "rb") as f:
        audio_embeddings = torch.load(f)

    labels = get_labels(dataset_json, synth="dexed")
    print("Dexed labels:", labels)

    preset_id = get_preset_ids_per_label(dataset_json, labels, synth="dexed")
    print("Number of presets per label:")
    for k, v in preset_id.items():
        print(k, len(v))

    # take a random subset of the harmonic presets
    rnd_harm_idx = np.random.choice(len(preset_id["harmonic"]), harmonic_subset_size, replace=False)
    preset_id["harmonic"] = [preset_id["harmonic"][i] for i in rnd_harm_idx]

    # Initialize UMAP
    reducer = umap.UMAP(**umap_kwargs, random_state=seed)

    # dict to store the umap embeddings
    embeddings = {}

    # Extract and concatenate audio embeddings
    audio_emb = np.concatenate([audio_embeddings[v].numpy() for v in preset_id.values()])
    # Compute the preset labels' start and end indices
    sep_idx = np.cumsum([0] + [len(v) for v in preset_id.values()])
    # Fit and transform embeddings using UMAP
    u_audio = reducer.fit_transform(audio_emb)
    embeddings["ref"] = u_audio

    # iterate over the models to evaluate
    for model in models:
        print(f"Computing UMAP projections for the {MODEL_NAME_FORMAT[model]} model...")
        if not (EVAL_DIR / "preset_embeddings" / f"{model}_embeddings.pkl").exists():
            filename = EVAL_DIR / "preset_embeddings" / f"{model}_embeddings.pkl"
            print(f"{filename} does not exists, generating and exporting embeddings...")
            generate_preset_embeddings(
                synth="diva", model=model, batch_size=batch_size, dataset_version=dataset_version
            )

        with open(EVAL_DIR / "preset_embeddings" / f"{model}_embeddings.pkl", "rb") as f:
            preset_embeddings = torch.load(f)

        # Extract preset embeddings and concatenate
        preset_emb = np.concatenate([preset_embeddings[v].numpy() for v in preset_id.values()])
        # Sanity checks
        assert audio_emb.shape[0] == preset_emb.shape[0]
        for v in preset_id.values():
            assert len(audio_embeddings[v]) == len(preset_embeddings[v])

        # Fit and transform embeddings using UMAP
        u_preset = reducer.fit_transform(preset_emb)
        embeddings[model] = u_preset

    # Plot
    fig, ax = plt.subplots(
        nrows=1, ncols=len(embeddings), figsize=(4 * len(embeddings), 4.2), layout="constrained"
    )
    for a, (source, u_emb) in zip(ax, embeddings.items()):
        for i, c in enumerate(preset_id.keys()):
            a.scatter(
                u_emb[sep_idx[i] : sep_idx[i + 1], 0],
                u_emb[sep_idx[i] : sep_idx[i + 1], 1],
                label=c,
                s=15,
                alpha=0.5,
                color=DEXED_LABEL_COLOR_MAP[c],
                edgecolors="none",
            )
        if source == "ref":
            legend_handles = [
                plt.Rectangle((0, 0), 2, 1, fill=True, color=v) for v in DEXED_LABEL_COLOR_MAP.values()
            ]
            legend_labels = list(DEXED_LABEL_COLOR_MAP)
            fig.legend(
                legend_handles,
                legend_labels,
                ncol=3,
                frameon=False,
                loc="outside lower left",
                fontsize="x-large",
            )
            a.set_title("Reference", fontsize=font_size)
        else:
            a.set_title(f"{MODEL_NAME_FORMAT[source]}", fontsize=font_size)
        a.set_xticks([])
        a.set_yticks([])
        a.axis("off")

    plt.savefig(EXPORT_DIR / "umap_dexed_labels.pdf", bbox_inches="tight")


def generate_preset_embeddings(
    synth: str, model: str, batch_size: int = 512, dataset_version: int = 1
) -> None:
    # Load eval configs to retrieve model configs and ckpt name
    eval_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "eval" / "model" / f"{synth}_{model}.yaml")["model"]
    model_cfg = {k: v for k, v in eval_cfg["cfg"].items() if k != "_target_"}
    ckpt_name = eval_cfg["ckpt_name"]

    # path to the eval dataset
    data_dir = PROJECT_ROOT / "data" / "datasets" / "eval" / f"{synth}_mn04_eval_v{dataset_version}"

    with open(data_dir / "configs.pkl", "rb") as f:
        configs = torch.load(f)

    with open(data_dir / "synth_parameters.pkl", "rb") as f:
        synth_parameters = torch.load(f)

    preset_helper = PresetHelper(
        synth_name=synth,
        parameters_to_exclude=configs["params_to_exclude"],
    )

    m_preset = getattr(model_zoo, model)(
        **model_cfg, out_features=configs["num_outputs"], preset_helper=preset_helper
    )
    lit_model = PresetEmbeddingLitModule.load_from_checkpoint(
        str(PROJECT_ROOT / "checkpoints" / ckpt_name), preset_encoder=m_preset
    )
    lit_model.to(DEVICE)
    lit_model.freeze()

    preset_embeddings = torch.empty(len(synth_parameters), configs["num_outputs"])

    dataset = TensorDataset(synth_parameters)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        batch = batch[0].to(DEVICE)
        with torch.no_grad():
            preset_embeddings[i * batch_size : (i + 1) * batch_size] = lit_model(batch)

    (data_dir / "preset_embeddings").mkdir(parents=True, exist_ok=True)
    with open(data_dir / "preset_embeddings" / f"{model}_embeddings.pkl", "wb") as f:
        torch.save(preset_embeddings, f)


def umap_hc_vs_syn_presets(
    synths: Tuple[str] = ("dexed", "diva"),
    subset_size: Dict[str, int] = {"dexed": 30_000, "diva": 10_000},
    umap_kwargs: Dict[str, Any] = {"n_neighbors": 75, "min_dist": 0.99, "metric": "euclidean", "init": "pca"},
    seed: int = 42,
    font_size: int = 21,
    dataset_version: int = 1,
) -> None:
    """
    Compute and plot the UMAP projections of a subset of synthetic presets from the test dataset,
    and the available dataset of hand-crafted presets.

    Args:
       synths (Tuple[str], optional): Synthesizers from which to plot the presets. (Defaults: ("dexed", "diva")).
       subset_size (Dict[str, int], optional): Number of synthetic presets to use for each synthesizer.
       (Defaults: {"dexed": 30_000, "diva": 10_000}).
       seed (int, optional): Random seed for the subset of synthetic presets and to compute the UMAP. (Defaults: 42).
       font_size (int, optional): Font size for the plots. (Defaults: 21).
       dataset_version (int, optional): Version of the test dataset to use. (Defaults: 1).
    """
    synth_names_fmt = {"dexed": "Dexed", "diva": "Diva", "talnm": "TAL-NM."}

    eval_dir = PROJECT_ROOT / "data" / "datasets" / "eval"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    real_data_dict = {}
    synthetic_data_dict = {}
    for synth in synths:
        with open(eval_dir / f"{synth}_mn04_eval_v1" / "audio_embeddings.pkl", "rb") as f:
            real_data_dict[synth] = torch.load(f).numpy()

        with open(
            eval_dir / f"{synth}_mn04_size=131072_seed=600_test_v{dataset_version}" / "audio_embeddings.pkl",
            "rb",
        ) as f:
            synthetic_data_dict[synth] = torch.load(f).numpy()
            num_samples = subset_size[synth]
            rnd_indexes = np.random.choice(len(synthetic_data_dict[synth]), num_samples, replace=False)
            synthetic_data_dict[synth] = synthetic_data_dict[synth][rnd_indexes]

    reducer = umap.UMAP(**umap_kwargs, random_state=seed)

    fig, ax = plt.subplots(ncols=2, figsize=(8, 4.2), layout="constrained")

    for a, synth in zip(ax, synths):
        print(f"Computing UMAP projections for {synth}...")

        real_data = real_data_dict[synth]
        synthetic_data = synthetic_data_dict[synth]

        # Concatenate audio embeddings
        embeddings = np.concatenate([real_data, synthetic_data])
        sep_idx = len(real_data)

        # Fit and transform embeddings using UMAP
        u_embeddings = reducer.fit_transform(embeddings)

        a.scatter(
            u_embeddings[:sep_idx, 0],
            u_embeddings[:sep_idx, 1],
            label="hand-crafted",
            s=3,
            alpha=0.5,
            color=COLORS[1],
            edgecolors="none",
        )
        a.scatter(
            u_embeddings[sep_idx:, 0],
            u_embeddings[sep_idx:, 1],
            label="synthetic",
            s=3,
            alpha=0.5,
            color=COLORS[7],
            edgecolors="none",
        )
        if synth == "dexed":
            legend_handles = [
                plt.Rectangle((0, 0), 2, 1, fill=True, color=COLORS[1]),
                plt.Rectangle((0, 0), 2, 1, fill=True, color=COLORS[7]),
            ]
            legend_labels = ["hand-crafted", "synthetic"]
            fig.legend(
                legend_handles,
                legend_labels,
                frameon=False,
                loc="outside lower left",
                fontsize="x-large",
                ncol=2,
            )
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(synth_names_fmt[synth], fontsize=font_size)
        a.set_frame_on(False)

    plt.savefig(EXPORT_DIR / "umap_hc_vs_syn_presets.pdf", bbox_inches="tight")
