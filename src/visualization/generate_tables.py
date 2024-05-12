"""
Module to display the evaluation results.

Table will be printed to the console and saved as a csv file in the output folder (along with the figures), 
which defaults to `PROJECT_ROOT/reports/eval_results`
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import matplotlib.pyplot as plt

from utils.visualization.eval_results import (
    load_results,
    get_results_df,
    synthetic_presets_results_df,
    synthetic_presets_results_plot,
    handcrafted_presets_results_df,
    handcrafted_presets_results_plot,
    num_presets_vs_metric_plot,
)

load_dotenv()
PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])


### Evaluation results to visualize
# Results on synthetic presets
DISPLAY_RESULTS_SYN = True
# Results on hand-crafted presets
DISPLAY_RESULTS_HC = True
# Number of presets vs metric
DISPLAY_NUM_PRESETS_VS_METRIC = False

### Visualization parameters
# metrics to plot
METRICS = ["mrr", "loss"]
# whether to save figures and results as CSV in EXPORT_DIR
EXPORT_RESULTS = True
# whether to plot the results
PLOT_RESULTS = False
# whether to print the results in latex format
PRINT_LATEX = False

### Paths
# Suffix appended to the results folder if the logs/eval directory contains several folder
# if a single folder is present, set it to None
SUFFIX = None
# Directory where to export the results
EXPORT_DIR = PROJECT_ROOT / "results" / "eval"

if __name__ == "__main__":

    EVAL_DIR = PROJECT_ROOT / "logs" / "eval"

    if SUFFIX is not None:
        results_path = EVAL_DIR / f"results{SUFFIX}"
    else:
        results_path = list(EVAL_DIR.glob("*"))[0]

    # Create export dir if it does not exist
    EXPORT_DIR.mkdir(exist_ok=True)

    results_dict = load_results(results_path)
    results_df = get_results_df(results_dict)

    if DISPLAY_RESULTS_SYN:
        syn_results_df = synthetic_presets_results_df(results_df)
        if EXPORT_RESULTS:
            syn_results_df.to_csv(EXPORT_DIR / "synthetic_presets_results.csv")
        for metric in METRICS:
            synthetic_presets_results_plot(
                syn_results_df, metric=metric, save_fig=EXPORT_RESULTS, export_path=EXPORT_DIR
            )
        print("\nResults on synthetic presets:\n")
        print(syn_results_df)

        if PRINT_LATEX:
            print("\nResults on synthetic presets (Latex):\n")
            print(
                syn_results_df.to_latex(
                    index=True,
                    float_format="{:.4f}".format,
                    multicolumn_format="c",
                ),
            )

    if DISPLAY_RESULTS_HC:
        hc_results_df = handcrafted_presets_results_df(results_df)
        if EXPORT_RESULTS:
            hc_results_df.to_csv(EXPORT_DIR / "handcrafted_presets_results.csv")
        for metric in METRICS:
            handcrafted_presets_results_plot(
                results_df, metric=metric, save_fig=EXPORT_RESULTS, export_path=EXPORT_DIR
            )
        print("\nResults on hand-crafted presets:\n")
        print(hc_results_df)

        if PRINT_LATEX:
            print("\nResults on hand-crafted presets (Latex):\n")
            print(
                hc_results_df.to_latex(
                    index=True,
                    float_format="{:.4f}".format,
                    multicolumn_format="c",
                )
            )

    if DISPLAY_NUM_PRESETS_VS_METRIC:
        for metric in METRICS:
            num_presets_vs_metric_plot(
                results_dict, metric=metric, font_size=16, save_fig=EXPORT_RESULTS, export_path=EXPORT_DIR
            )

    if PLOT_RESULTS:
        plt.show()
