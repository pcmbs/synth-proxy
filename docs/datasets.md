## Available datasets
The datasets used for hyperparameter optimization, training, and evaluation (synthetic and hand-crafted) can be downloaded [here](https://e.pcloud.link/publink/show?code=kZUw9MZ60OxeaBeBPyr87PihHKfjSHC2qRk). 

## Synthetic data generation

The [./src/export/dataset_pkl.py](./src/export/dataset_pkl.py) script for Linux (also available for [Windows](./src/export/dataset_pkl_win.py)) can be used to generate synthetic presets. 
It uses the torch dataset class [SynthDataset](./src/data/datasets/synth_dataset.py) and [DawDreamer](https://github.com/DBraun/DawDreamer) under the hood. Note that it is currently only possible to generate synthetic presets by sampling their parameters as follows:
- Continuous numerical parameters are sampled from a uniform distribution between 0 and 1.
- Discrete parameters, i.e., categorical, binary, or discretized numerical, are sampled from a categorical distribution.

Example: 
```bash
python src/export/dataset_pkl.py synth=talnm audio_fe=mn04 dataset_size=65_536 seed_offset=10 batch_size=512 num_workers=8 tag=test
```

See the [export configuration](./configs/export/) for more details.

## Hand-crafted preset dataset
The script to generate the dataset of hand-crafted presets can be found in [./src/export/hcp_dataset.py](./src/export/hcp_dataset.py). It allows to generate a dataset of hand-crafted presets given an existing dataset of presets stored in a json file. 

→ Only the synthesizer parameters used during training (i.e., for generating the training dataset) are considered; the remaining parameters are set to their default values and excluded, and silent and duplicate presets are removed.