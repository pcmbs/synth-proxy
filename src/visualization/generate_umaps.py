"""
Script to reproduce the UMAP projections figures from the paper.
Figures are saved under `PROJECT_ROOT/reports/umap_projections` 

The current configuration is the one used in the paper.
"""

from visualization.umap_fn import umap_hc_vs_syn_presets, umap_dexed_labels, umap_diva_labels

########## Common CFG
# for reproductibility (UMAP & Numpy)
RANDOM_SEED = 42

# avalaible models for Dexed and Diva labeled presets: ["mlp_oh", "hn_oh", "hn_pt", "hn_ptgru", "tfm"]
MODELS = ["tfm", "hn_ptgru", "mlp_oh"]

# for computing preset embeddings if required (Dexed and Diva labeled presets),
# feel free to reduce if needed
BATCH_SIZE = 512

########## UMAP projections of hand-crafted vs synthetic presets
# Synthesizers and subset sizes for the synthetic presets
SYNTHS = ["dexed", "diva"]
SUBSET_SIZE = {"dexed": 30_000, "diva": 10_000}

# Kwargs for umap.UMAP
UMAP_KWARGS_HC_VS_SYN = {"n_neighbors": 75, "min_dist": 0.99, "metric": "euclidean", "init": "pca"}


########## UMAP projections of labeled hand-crafted presets for Dexed
# we take 2k harmonic presets out of 25k using numpy random choice
# since there are only 800 percussive and 1500 sfx presets
HARMONIC_SUBSET_SIZE = 2000

# Kwargs for umap.UMAP
UMAP_KWARGS_DEXED = {"n_neighbors": 75, "min_dist": 0.99, "metric": "euclidean", "init": "pca"}


########## UMAP projections of labeled hand-crafted presets for Diva
# Mutually exclusive labels to plot
LABELS_TO_PLOT = (("Aggressive", "Soft"), ("Bright", "Dark"))

# Kwargs for umap.UMAP
UMAP_KWARGS_DIVA = {"n_neighbors": 75, "min_dist": 0.5, "metric": "euclidean", "init": "pca"}


if __name__ == "__main__":
    print("\nComputing and plotting UMAP projections of hand-crafted vs synthetic presets...")
    umap_hc_vs_syn_presets(
        synths=SYNTHS, subset_size=SUBSET_SIZE, umap_kwargs=UMAP_KWARGS_HC_VS_SYN, seed=RANDOM_SEED
    )

    print("\nComputing and plotting UMAP projections of labeled hand-crafted presets for Dexed...")
    umap_dexed_labels(
        models=MODELS,
        umap_kwargs=UMAP_KWARGS_DEXED,
        harmonic_subset_size=HARMONIC_SUBSET_SIZE,
        seed=RANDOM_SEED,
        batch_size=BATCH_SIZE,
    )

    print("\nComputing and plotting UMAP projections of labeled hand-crafted presets for Diva...")
    umap_diva_labels(
        models=MODELS,
        umap_kwargs=UMAP_KWARGS_DIVA,
        labels_to_plot=LABELS_TO_PLOT,
        seed=RANDOM_SEED,
        batch_size=BATCH_SIZE,
    )
