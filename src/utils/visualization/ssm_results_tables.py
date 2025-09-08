"""
Script to build LaTeX tables from csv files exported from WandB.
"""

import pandas as pd

# path to the exported csv file
CSV_PATH = "wandb_export.csv"

COL_ID_FMT = {
    "misc.synth": "Synth.",
    "losses.loss_sch": "sched.",
    "test/id/metrics/num_mae/dataloader_idx_0": "Num. MAE",
    "test/id/metrics/bin_acc/dataloader_idx_0": "Bin. Acc.",
    "test/id/metrics/cat_acc/dataloader_idx_0": "Cat. Acc.",
    "test/id/metrics/stft/dataloader_idx_0": "STFT",
    "test/id/metrics/mstft/dataloader_idx_0": "mSTFT",
    "test/id/metrics/mel/dataloader_idx_0": "Mel",
    "test/id/metrics/mfccd/dataloader_idx_0": "MFCCD",
}

COL_NS_FMT = {
    "misc.synth": "Synth.",
    "losses.loss_sch": "sched.",
    "test/nsynth/metrics/stft/dataloader_idx_1": "STFT",
    "test/nsynth/metrics/mstft/dataloader_idx_1": "mSTFT",
    "test/nsynth/metrics/mel/dataloader_idx_1": "Mel",
    "test/nsynth/metrics/mfccd/dataloader_idx_1": "MFCCD",
}

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    # DF for in-domain results
    df_id = df.loc[:, COL_ID_FMT.keys()]
    df_id.columns = [COL_ID_FMT[c] for c in df_id.columns]
    print("\n\nIn-domain results:\n")
    print(df_id.to_latex(index=False, float_format="{:.3f}".format, multicolumn_format="c"))
    # transpose and print
    print("\n\nIn-domain results (transposed):\n")
    print(df_id.transpose().to_latex(index=False, float_format="{:.3f}".format, multicolumn_format="c"))

    # DF for out-of-domain results
    df_ns = df.loc[:, COL_NS_FMT.keys()]
    df_ns.columns = [COL_NS_FMT[c] for c in df_ns.columns]
    print("\n\nOut-Of-domain results:\n")
    print(df_ns.to_latex(index=False, float_format="{:.3f}".format, multicolumn_format="c"))
