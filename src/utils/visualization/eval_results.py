"""
Utility functions for displaying the results presented in the paper.
"""

from pathlib import Path
import pickle
from typing import Dict, Sequence, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["font.sans-serif"] = ["cmr10"]
plt.rcParams["mathtext.fontset"] = "stix"
# YOLO
warnings.filterwarnings(
    "ignore",
    message="cmr10 font should ideally be used with mathtext, set axes.formatter.use_mathtext to True",
)

# DATA AND METRICS
_METRIC_FORMAT = {
    "mrr": "MRR",
    "loss": "L1 Error",
}
_METRIC_NAMES = tuple(_METRIC_FORMAT.values())

_VAL_FORMAT = {
    "rnd": "SYN.",
    "hc": "HC.",
    "percentage": "Diff. (%)",
}
_VAL_NAMES = tuple(_VAL_FORMAT.values())

_STATS_FORMAT = {"mean": "Mean", "std": "Std."}
_STATS_NAMES = tuple(_STATS_FORMAT.values())

# FORMAT AND PLOT THINGS
_MODEL_NAME_FORMAT = {
    "mlp_oh": "MLP-OH",
    "hn_oh": "HN-OH",
    "hn_pt": "HN-PT",
    "hn_ptgru": "HN-PTGRU",
    "tfm": "TFM",
}
_MODEL_NAMES = tuple(_MODEL_NAME_FORMAT.values())

_SYNTH_NAME_FORMAT = {
    "dexed": "Dexed",
    "diva": "Diva",
    "talnm": "TAL-NM.",
}
_SYNTH_NAMES = tuple(_SYNTH_NAME_FORMAT.values())

_COLORS = plt.cm.Paired.colors  # Get colors from colormap
# _COLOR_MAP = {"Dexed": _COLORS[6], "Diva": _COLORS[2], "TAL-NM.": _COLORS[0]} # paper color theme
_COLOR_MAP = {"Dexed": _COLORS[8], "Diva": _COLORS[4], "TAL-NM.": _COLORS[0]}  # presentation color theme
_FONT_SIZE = 20

# Rounding depending on the displayed value
_NUM_DECIMAL = {"mrr": 3, "loss": 4, "percentage": 1}


def load_results(path: Union[str, Path]) -> Dict:
    """Load the results for all model in `path`"""
    path = Path(path)

    # initialize results dict
    results = {}

    # iterate over the results of all models and synths
    # and retrieve the results
    for path in path.iterdir():
        if path.is_dir():
            synth, model_type = path.name.split("_", 1)
            if model_type not in results:
                results[model_type] = {}
            with open(path / "results.pkl", "rb") as f:
                results[model_type][synth] = pickle.load(f)
    return results


def get_results_df(results: Dict) -> pd.DataFrame:
    """Get a dataframe containing results given the dict of results"""
    col_synth_names = [
        s for s in _SYNTH_NAMES for _ in _METRIC_NAMES for _ in _VAL_NAMES for _ in _STATS_NAMES
    ]
    col_metric_names = [
        m for _ in _SYNTH_NAMES for m in _METRIC_NAMES for _ in _VAL_NAMES for _ in _STATS_NAMES
    ]
    col_val_names = [d for _ in _SYNTH_NAMES for _ in _METRIC_NAMES for d in _VAL_NAMES for _ in _STATS_NAMES]
    col_stat_names = [
        s for _ in _SYNTH_NAMES for _ in _METRIC_NAMES for _ in _VAL_NAMES for s in _STATS_NAMES
    ]

    num_val_per_synth = len(_METRIC_NAMES) * len(_VAL_NAMES) * len(_STATS_NAMES)

    data = np.empty((len(_MODEL_NAMES), len(col_val_names)))

    for i, model_type in enumerate(_MODEL_NAME_FORMAT.keys()):
        for j, s in enumerate(_SYNTH_NAME_FORMAT.keys()):
            # MRR score on random presets, hand-crafted presets, and difference in percentage
            rnd_preset_mrr = results[model_type][s]["rnd"]["mrr"]["mean"].round(_NUM_DECIMAL["mrr"])
            rnd_preset_mrr_std = results[model_type][s]["rnd"]["mrr"]["std"].round(_NUM_DECIMAL["mrr"])
            hc_preset_mrr = results[model_type][s]["hc"]["mrr"]["mean"].round(_NUM_DECIMAL["mrr"])
            hc_preset_mrr_std = results[model_type][s]["hc"]["mrr"]["std"].round(_NUM_DECIMAL["mrr"])
            percentage_decrease = -100 * (rnd_preset_mrr - hc_preset_mrr) / rnd_preset_mrr
            percentage_decrease = percentage_decrease.round(_NUM_DECIMAL["percentage"])

            # L1 Error on random presets, hand-crafted presets, and difference in percentage
            rnd_preset_loss = results[model_type][s]["rnd"]["loss"]["mean"].round(_NUM_DECIMAL["loss"])
            rnd_preset_loss_std = results[model_type][s]["rnd"]["loss"]["std"].round(_NUM_DECIMAL["loss"])
            hc_preset_loss = results[model_type][s]["hc"]["loss"]["mean"].round(_NUM_DECIMAL["loss"])
            hc_preset_loss_std = results[model_type][s]["hc"]["loss"]["std"].round(_NUM_DECIMAL["loss"])
            percentage_increase = 100 * (hc_preset_loss - rnd_preset_loss) / rnd_preset_loss
            percentage_increase = percentage_increase.round(_NUM_DECIMAL["percentage"])

            # store results for current model and synth
            data[i, j * num_val_per_synth : (j + 1) * num_val_per_synth] = np.array(
                [
                    rnd_preset_mrr,
                    rnd_preset_mrr_std,
                    hc_preset_mrr,
                    hc_preset_mrr_std,
                    percentage_decrease,
                    0.0,  # placeholder for diff std.
                    rnd_preset_loss,
                    rnd_preset_loss_std,
                    hc_preset_loss,
                    hc_preset_loss_std,
                    percentage_increase,
                    0.0,  # placeholder for diff std.
                ]
            )

    df = pd.DataFrame(
        data, index=_MODEL_NAMES, columns=[col_synth_names, col_metric_names, col_val_names, col_stat_names]
    )
    # Label the index header
    df.index.name = "Model"
    # remove the ("Diff. (%)", "Std.") columns for each synth and metric
    cols_to_drop = df.loc[:, (_SYNTH_NAMES, _METRIC_NAMES, _VAL_FORMAT["percentage"], _STATS_FORMAT["std"])]
    df = df.drop(cols_to_drop.columns, axis=1)
    # rename the columns "Mean" to "" at level 3 if the columns are level 2 are named "Diff. (%)"
    df.columns = df.columns.values
    df.columns = pd.MultiIndex.from_tuples(
        df.rename(
            columns={
                (s, m, _VAL_FORMAT["percentage"], _STATS_FORMAT["mean"]): (
                    s,
                    m,
                    _VAL_FORMAT["percentage"],
                    "",
                )
                for s in _SYNTH_NAMES
                for m in _METRIC_NAMES
            }
        )
    )
    return df


def synthetic_presets_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the results on synthetic presets from the main results DF."""
    results = df.loc[:, (_SYNTH_NAMES, _METRIC_NAMES, [_VAL_FORMAT["rnd"]])]
    results.columns = results.columns.droplevel(2)
    return results


def synthetic_presets_results_plot(
    df: pd.DataFrame, metric: str = "mrr", save_fig: bool = False, export_path: Union[str, Path] = None
) -> None:
    """
    Plot the results on synthetic presets (bar plot).

    Args:
       df (pandas.DataFrame): The DF of results on synthetic presets.
       metric (str): The metric to plot. Must be either 'mrr' or 'loss'. (Defaults: 'mrr')
       save_fig (bool): Whether to save the figure as a PDF. (Defaults: False)
       export_path (Union[str, Path]): Where to save the figure. Only used if `save_fig` is True.
       If None, it will be saved in the current directory. (Defaults: None)
    """
    metric = "MRR" if metric == "mrr" else "L1 Error"
    assert metric in _METRIC_NAMES, "metric must be either 'mrr' or 'loss'"

    # Set the width of each bar
    bar_width = 0.27

    # Calculate the width for each group of bars
    total_width = bar_width * len(_SYNTH_NAMES)
    index = np.arange(len(_MODEL_NAMES))

    # Initialize the plot
    # _, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    _, ax = plt.subplots(figsize=(6, 5), layout="constrained")

    # Iteratively plot each bar
    max_val = -np.inf  # initialize max value for ylim
    for i, m in enumerate(_MODEL_NAMES):
        for j, s in enumerate(_SYNTH_NAMES):
            val = df.loc[m, (s, metric, _STATS_FORMAT["mean"])]
            std_val = df.loc[m, (s, metric, _STATS_FORMAT["std"])]
            rect = ax.bar(
                i + j * bar_width,
                val,
                bar_width,
                label=s if i == 0 else "",
                color=_COLOR_MAP[s],
                edgecolor="black",
                linewidth=0.5,
            )
            # # Add mean labels
            # ax.bar_label(rect, padding=2)
            # Add mean and std labels
            # ax.bar_label(rect, labels=[f"{val}"], padding=12)
            # ax.bar_label(rect, labels=[r"$\pm$" + f"{std_val}"], padding=2, fontsize=8)

            # Add error bars for standard deviation
            ax.errorbar(
                i + j * bar_width,
                val,
                yerr=std_val,
                fmt="none",
                ecolor="black",
                elinewidth=0.5,
                capsize=6,
                capthick=0.5,
            )
            # Update max value if needed
            max_val = max(val, max_val)

    # Add labels and title
    # ax.set_xlabel("Model", labelpad=10, fontsize=_FONT_SIZE)
    ax.set_ylabel(r"L$^1$ Error" if metric == "L1 Error" else metric, fontsize=_FONT_SIZE)
    ax.set_ylim(0, max_val + 0.1 * max_val)
    ax.set_xticks(index + total_width / 3)
    ax.set_xticklabels(_MODEL_NAMES)
    ax.tick_params(axis="both", which="both", labelsize=_FONT_SIZE - 8)
    ax.legend(loc="lower right", ncols=3, bbox_to_anchor=(1.0, -0.20), fontsize=_FONT_SIZE - 8)

    if not save_fig:  # do not add plot title for the paper
        ax.set_title("Overall Results on Random Presets")

    if save_fig:
        suffix = "mrr" if metric == "MRR" else "loss"
        if export_path is None:
            plt.savefig(f"synthetic_presets_results_{suffix}.pdf", bbox_inches="tight")
        else:
            plt.savefig(Path(export_path) / f"synthetic_presets_results_{suffix}.pdf", bbox_inches="tight")


def handcrafted_presets_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with only the handcrafted preset results."""
    results = df.loc[:, (_SYNTH_NAMES, _METRIC_NAMES, [_VAL_FORMAT["hc"], _VAL_FORMAT["percentage"]])]
    return results


def handcrafted_presets_results_plot(
    df: pd.DataFrame,
    metric: str = "mrr",
    plot_percentage: bool = True,
    save_fig: bool = False,
    export_path: Union[str, Path] = None,
) -> None:
    """
    Plot the results of the handcrafted presets.

    Args:
        df (pandas.Dataframe): Dataframe containing the results to plot.
        metric (str, optional): Metric to plot. Must be one of "mrr" or "loss" (Defaults: "mrr").
        plot_percentage (bool, optional): Whether to also plot the difference with synthetic presets
        as percentage. (Defaults: True).
        save_fig (bool, optional): Whether to save the figure.
    """
    metric = "MRR" if metric == "mrr" else "L1 Error"
    assert metric in _METRIC_NAMES, "metric must be either 'mrr' or 'loss'"

    # _SYNTH_NAMES = ("Dexed", "Diva")

    # df = df.loc[:, (_SYNTH_NAMES, metric, _VAL_NAMES, (_STATS_FORMAT["mean"], ""))]
    df = df.loc[:, (_SYNTH_NAMES, metric, _VAL_NAMES)]
    df.columns = df.columns.droplevel(1)

    # Set the width of each bar
    bar_width = 0.153
    # bar_width = 0.18

    # Calculate the width for each group of bars
    total_width = bar_width * len(_SYNTH_NAMES) * 2  # * 2 since we have 2 bars per synth
    index = np.arange(len(_MODEL_NAMES))

    # Initialize the plot
    _, ax = plt.subplots(figsize=(15, 5), layout="constrained")

    # Iteratively plot each bar
    max_val = -np.inf  # initialize max value for ylim
    for i, m in enumerate(_MODEL_NAMES):
        for j, s in enumerate(_SYNTH_NAMES):
            for k, v in enumerate([_VAL_FORMAT["rnd"], _VAL_FORMAT["hc"]]):
                # plot results on `v` presets for `m` model and `s` synth
                val = df.loc[m, (s, v, _STATS_FORMAT["mean"])]
                std_val = df.loc[m, (s, v, _STATS_FORMAT["std"])]
                rect = ax.bar(
                    i + (j * 2 + k) * bar_width,
                    val,
                    bar_width,
                    color=_COLOR_MAP[s],
                    hatch="///" if k else "",
                    edgecolor="black",
                    linewidth=0.5,
                )
                # add value label
                # ax.bar_label(rect, padding=2)

                # Add error bars for standard deviation
                ax.errorbar(
                    i + (j * 2 + k) * bar_width,
                    val,
                    yerr=std_val,
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.5,
                    capsize=4,
                    capthick=0.5,
                )

                # Update max value if needed
                max_val = max(val, max_val)

                # add percentage decrease label aligned to hand-crafted bar...
            if plot_percentage:
                x_pos = i + (j * 2 + 1) * bar_width
                y_pos = val + (0.10 if metric == "MRR" else -0.025)
                label = (
                    ("+" if metric == "L1 Error" else "")
                    + str(df.loc[m, (s, _VAL_FORMAT["percentage"], "")])
                    + "%"
                )
                ax.text(
                    x_pos,
                    y_pos,
                    label,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontweight="semibold",
                    fontsize=_FONT_SIZE - 8,
                    bbox={
                        "facecolor": "white" if metric == "MRR" else _COLOR_MAP[s],
                        "edgecolor": "black",
                        "linewidth": 0.0 if metric == "MRR" else 0.5,
                        "mutation_aspect": 0.75,
                    },
                )

    # Plots settings
    ax.set_xlabel("Model", labelpad=10, fontsize=_FONT_SIZE)
    ax.set_ylabel(r"L$^1$ Error" if metric == "L1 Error" else metric, fontsize=_FONT_SIZE)
    ax.set_ylim(0, max_val + 0.1 * max_val)
    ax.set_xticks(index + total_width / 2.4)
    ax.set_xticklabels(_MODEL_NAMES)
    ax.tick_params(axis="both", which="both", labelsize=_FONT_SIZE - 4)

    # Add legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_COLOR_MAP["Dexed"], edgecolor="black", linewidth=0.5),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=0.5),
        plt.Rectangle((0, 0), 1, 1, facecolor=_COLOR_MAP["Diva"], edgecolor="black", linewidth=0.5),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=0.5, hatch="///"),
        plt.Rectangle((0, 0), 1, 1, facecolor=_COLOR_MAP["TAL-NM."], edgecolor="black", linewidth=0.5),
    ]
    legend_labels = ["Dexed", "Synthetic", "Diva", "Hand-Crafted", "TAL-NM."]
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        ncols=3,
        bbox_to_anchor=(1, -0.25),
        fontsize=_FONT_SIZE - 8,
    )

    if not save_fig:  # do not add title for the paper
        ax.set_title("Generalization Results: Hand-Crafted vs. Random Presets")

    if save_fig:
        suffix = "mrr" if metric == "MRR" else "loss"
        if export_path is None:
            plt.savefig(f"handcrafted_presets_results_{suffix}.pdf", bbox_inches="tight")
        else:
            plt.savefig(Path(export_path) / f"handcrafted_presets_results_{suffix}.pdf", bbox_inches="tight")

    return ax


def num_presets_vs_metric_plot(
    results: Dict,
    metric: str = "mrr",
    synths: Sequence[str] = ("dexed", "diva"),
    font_size: int = 12,
    save_fig: bool = False,
    export_path: Union[str, Path] = None,
) -> plt.Axes:
    """Plot a metric value vs. number of presets used to compute it (mainly for MRR)."""
    metric = "MRR" if metric == "mrr" else "L1 Error"
    assert metric in _METRIC_NAMES, "metric must be either 'mrr' or 'loss'"

    fig, ax = plt.subplots(
        ncols=len(synths), figsize=(3.1 * len(synths), 3.2), layout="constrained", sharey=True
    )

    for i, (a, synth) in enumerate(zip(ax, synths)):

        for model_name, model_results in results.items():
            result = model_results[synth]["rnd_incr"]
            x_values = list(result.keys())
            y_values = [data["mrr" if metric == "MRR" else "loss"] for data in result.values()]
            a.plot(x_values, y_values, label=_MODEL_NAME_FORMAT[model_name])

        # Add labels and title
        a.set_title(_SYNTH_NAME_FORMAT[synth], fontsize=font_size)
        if i == 0:
            a.set_xlabel(" ", fontsize=font_size)
            a.set_ylabel("MRR" if metric == "MRR" else r"$\mathrm{L_1}$ Error", fontsize=font_size)
            a.legend(loc="lower left", fontsize="small")
        a.set_ylim(0, 1 if metric == "MRR" else 0.10)
        a.set_xlim(min(x_values), max(x_values))
        a.grid(which="both", alpha=0.5)
        # log x axis
        a.set_xscale("log")
        # ticks font size (a bit smaller)
        a.tick_params(axis="both", which="both", labelsize=font_size - 4)
    # common x label
    fig.text(
        0.54,
        0.04,
        "Number of presets",
        ha="center",
        fontsize=font_size,
    )

    if not save_fig:  # do not add title for the paper
        fig.suptitle("MRR dependings on the number of non-matching presets")

    if save_fig:
        suffix = "mrr" if metric == "MRR" else "loss"
        if export_path is None:
            plt.savefig(f"num_presets_vs_{suffix}.pdf", bbox_inches="tight")
        else:
            plt.savefig(Path(export_path) / f"num_presets_vs_{suffix}.pdf", bbox_inches="tight")

    return ax
