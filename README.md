<h1 align="center">
  Neural Proxies for Sound Synthesizers: <br>
  Learning Perceptually Informed Presets Representations

</h1>


## Overview
Code source repository for my master's thesis on Neural Proxies for Sound Synthesizers (available soon), in which I developed a method for approximating black-box sound synthesizers using neural networks. \
The proposed method relies on training a neural network capable of mapping synthesizer presets onto a perceptually informed embedding space defined by a pretrained audio model. More specifically, given a preset $\underline{x}$ from a synthesizer $s$, a preset encoder $f_{\underline{\theta}}$ learns to minimize the $L^1$ distance between its representation of $\underline{x}$ and that produced by a pretrained audio model $g_{\underline{\phi}}$, derived from the synthesized audio $s(\underline{x})$. This process effectively creates a neural proxy for a given synthesizer by leveraging the audio representations learned by the pretrained model via a cross-modal knowledge distillation task. 

The aim of this repository is to use various neural network architectures, including feedforward, recurrent, and transformer-based models, to encode presets from arbitrary software synthesizers, paving the way for the integration of non-differentiable, black-box synthesizers into end-to-end training pipeline relying on gradient descent.

<p align="center">
   <img src="https://i.imgur.com/MdOXU5R.png">  
</p>



## Installation

This project supports installation via pip and Docker. After cloning the repository, choose one of the following methods:

### Installation using pip

Create and activate a VE and install the required dependencies:
   ```bash
   $ python -m venv .venv 
   $ .venv/bin/activate # Use .venv\Scripts\activate on Windows
   $ pip install -r requirements.txt
   ```

### Installation using Docker

#### 1. Build the Docker image
Create a Docker image, specifying your user details to avoid permission issues with mounted volumes.
```bash
$ docker build --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g)  -t synth-proxy:local .
```

#### 2. Run the container
Run the container using the provided shell script
```bash
$ bash scripts/docker-run.sh
```
Note:  If you encounter permission errors related to file access, adjust the ownership of the project directory: 
```bash
$ sudo chown -R $(id -u):$(id -g) .
```


### Environment Configuration

Before running the project, ensure you have a `.env` file in the root directory with the necessary environment variables set:
```plaintext 
PROJECT_ROOT=/path/to/project/root/ 
WANDB_API_KEY=this-is-optional 
DEXED_PATH=/path/to/dexed/vst3
DIVA_PATH=/path/to/diva/vst3 
TALNM_PATH=/path/to/talnoisemaker/vst3 
```
Remarks: You do not need to install and link the synthesizers if you are not generating new datasets but are using the provided ones instead.

### Main dependencies
- [Pytorch](https://pytorch.org) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for defining the model and training.
- [DawDreamer](https://github.com/DBraun/DawDreamer) for rendering audio from VST plugins.
- [Wandb](https://wandb.ai) for logging.
- [Optuna](https://optuna.org) for hyperparameter optimization.
- [Hydra](https://hydra.cc) for configuration management.

The rest of the dependencies can be found in the [requirements.txt](./requirements.txt) file.



## Synthesizers

### Available Synthesizers
Currently, the project support the following synthesizers are supported: [Dexed](https://github.com/asb2m10/dexed), [Diva](https://u-he.com/products/diva/), and [TAL-NoiseMaker](https://tal-software.com/products/tal-noisemaker).

The [neural proxy](#available-preset-encoders-aka-neural-proxies)'s checkpoints for each synthesizer can be downloaded from the [link](https://e.pcloud.link/publink/show?code=kZkK9MZgyvowLICDzfmuQmiLltCgXiX31Ek).

### Adding Synthesizers
To add a new synthesizer, follow these steps:
1) Download and install the synthesizer (obviously).

2) Make sure the synthesizer is supported by [DawDreamer](https://github.com/DBraun/DawDreamer).

3) Add the path to the synthesizer as environment variable in the `.env` file, e.g., `NEW_SYNTH_PATH=/path/to/new/synth/vst3`.

4) Create a python file under [./src/data/synths/](./src/data/synths/) containing the internal representation of the synthesizer, which is implemented as a tuple of SynthParameter instances. You can use [./src/data/synths/another_synth.py](./src/data/synths/another_synth.py) as a template together with the helper script [./src/utils/synth/get_synth_parameters.py](./src/utils/synth/get_synth_parameters.py) to generate the representation of each synthesizer parameter (don't forget to manually double-check).

5) Add additional arguments for each SynthParameter instance if desired (see [./src/utils/synth/synth_parameter.py](./src/utils/synth/synth_parameter.py) and existing synthesizers for examples). These are used to constraint the sampling process used to generated synthetic presets.

6) Add the created python file for the new synthesizer to the package under [./src/data/synths/__init__.py](./src/data/synths/__init__.py).

7) Add the synthesizer name to the list of supported synthesizer names in the the PresetHelper class definition in [./src/utils/synth/preset_helper.py](./src/utils/synth/preset_helper.py) as well as in the SynthDataset class definition in [./src/data/datasets/synth_dataset.py](./src/data/datasets/synth_dataset.py).

8) Create a configuration file under [./configs/export/synth](./configs/export/synth) for the synthesizer specifying the parameters to exclude. The excluded parameters will be set to their default values during sampling and will not be fed to the preset encoder. 

9) Once the datasets have been generated, files in the following configuration folders need to be added depending on the need (see existing synthesizers for examples): [./configs/eval/synth](./configs/eval/synth), [./configs/hpo/synth](./configs/hpo/synth), [./configs/train/train_dataset](./configs/train/train_dataset), [./configs/train/val_dataset](./configs/train/val_dataset)

## Available Preset Encoders aka. Neural Proxies
An overview of the implemented neural proxies can be found under [./src/models/presets/model_zoo.py](./src/models/presets/model_zoo.py). The checkpoint of each pretrained model can be downloaded from this [link](https://e.pcloud.link/publink/show?code=kZkK9MZgyvowLICDzfmuQmiLltCgXiX31Ek) (Unzip and move all `.ckpt` files into the `checkpoints` folder).



## Audio Models

### Available Audio Models
Wrappers for the following audio models are available in the `./src/models/audio` directory: [EfficientAT](https://github.com/fschmid56/EfficientAT_HEAR/tree/main), [Torchopenl3](https://github.com/torchopenl3/torchopenl3), [PaSST](https://github.com/kkoutini/passt_hear21/tree/main), [Audio-MAE](https://github.com/facebookresearch/AudioMAE/tree/main), and Mel-only features.

Remarks:
- The aforementioned audio models were selected based on a custom evaluation pipeline built around the [TAL-NoiseMaker](https://tal-software.com/products/tal-noisemaker) Virtual Analog synthesizer. More details as well as the evaluation's results can be found in the dedicated [wandb report](https://wandb.ai/pcmbs/preset-embedding_audio-model-selection/reports/Neural-Proxies-for-Sound-Synthesizers-Audio-Model-Selection--Vmlldzo1MDYxNDUy).
- Only the `mn04_all_b` variation of the [EfficientAT](https://github.com/fschmid56/EfficientAT_HEAR/tree/main) models with average time pooling was used since the main objective of this work is to evaluate the feasibility of the proposed method rather than to find an optimal representation. 

### Adding Audio Models
1) Create a `./src/models/audio/<new_model>` directory and copy paste the model's source code into it.

2) Create a wrapper for the model inheriting from the `AudioModel` abstract class available in [./src/models/audio/abstract_audio_model.py](./src/models/audio/abstract_audio_model.py), and implement the necessary methods.

3) Instantiate the model below the wrapper class (see existing models for example). 

4) Add the instantiated model to the models.audio package by adding it in [./src/models/audio/__init__.py](./src/models/audio/__init__.py).

Once these steps are completed, the model can be used to generate a new dataset using the argument `audio_fe=<my_model>`. See [Data Generation](#data-generation) for more details.



## Data Generation

The [./src/data/datasets/synth_dataset_pkl.py](./src/data/datasets/synth_dataset_pkl.py) script for Linux (also available for [Windows](./src/data/datasets/synth_dataset_pkl_win.py)) can be used to generate synthetic presets. 
It uses the torch dataset class [SynthDataset](./src/data/datasets/synth_dataset.py) and [DawDreamer](https://github.com/DBraun/DawDreamer) under the hood. Note that it is currently only possible to generate synthetic presets by sampling their parameters as follows:
- Continuous numerical parameters are sampled from a uniform distribution between 0 and 1.
- Discrete parameters, i.e., categorical, binary, or discretized numerical, are sampled from a categorical distribution.

See the aformentioned files and the [export configuration](./configs/export/) for more details.

Example: 
```bash
$ python src/export/dataset_pkl.py synth=talnm audio_fe=mn04 dataset_size=65_536 seed_offset=10 batch_size=512 num_workers=8 tag=test
```

Remark: The datasets used for hyperparameter optimization, training, and evaluation (synthetic and hand-crafted) can be downloaded from the [link](https://e.pcloud.link/publink/show?code=kZUw9MZ60OxeaBeBPyr87PihHKfjSHC2qRk). 

## Hyperparameter Optimization
Details regarding the HPO can be found in [./configs/hpo/hpo.yaml](./configs/hpo/hpo.yaml) and [./src/hpo/run.py](./src/hpo/run.py).

Remark: the $L^1$ distance is used as the loss function, while the $L^1$ distance and mean reciprocal rank is used for validation. 
## Training
Details regarding training can be found in the [training script](./src/train.py) and [training configuration folder](./configs/train/).

Remarks:
- New experiments can be added to the existing ones in [./configs/train/experiments](./configs/train/experiments).
- Custom learning rate schedulers can be implemented in [./src/utils/lr_schedulers.py](./src/utils/lr_schedulers.py).
- training artifacts can be found in [./logs/train](./logs/train).
- the $L^1$ distance is used as the loss function, while the L1 distance and mean reciprocal rank is used for validation.

Example: 
```bash
$ python src/train.py experiment=dexed_tfm_b
```

## Evaluation
Compute the average $L^1$ distance and the mean reciprocal rank on a test set of synthetic and hand-crafted presets. More details can be found in the [evaluation script](./src/eval.py) and corresponding [configuration folder](./configs/eval/eval.yaml).

Example:
```bash
$ python src/eval.py model=dexed_tfm_b
```
Remarks:
- evaluation results will be saved under [./logs/eval](./logs/eval).
- to perform evaluation on a newly trained model, it is required to add its evaluation config file under [./configs/eval/model](./configs/eval/model) (see example there).

## Reproducibility

This section offers a detailed guide to replicate the results outlined in the paper. The project was developed on a Windows laptop equipped with an Intel i7 CPU and an RTX 2060 GPU (6GB VRAM), using Docker (through WSL2) with the provided Dockerfile. Most of the training was performed on a NVIDIA Tesla P100. Additional testing were conducted in a virtual environment using Python 3.10.11 and pip as dependencies manager.  

### Step-by-Step Guide

#### 1. Clone the Repository
Begin by cloning this repository to your local machine using the following command:
```bash 
$ git clone https://github.com/pcmbs/synth-proxy.git ; cd synth-proxy
```

#### 2. Download Datasets
Download the datasets used for testing the models from the provided [link](https://e.pcloud.link/publink/show?code=kZ4K9MZhrJlXX1OtNmVTYJiaGl7myPj0De7). Unzip the folder and move the most nested `eval` folder directly into the `data/datasets` directory.

#### 3. Download Model Checkpoints
Download the pretrained model checkpoints from this [link](https://e.pcloud.link/publink/show?code=kZkK9MZgyvowLICDzfmuQmiLltCgXiX31Ek). Unzip and move all `.ckpt` files into the `checkpoints` folder.

#### 4. Setup Environment
Install all necessary dependencies and configure the environment using pip or Docker by following the instructions from the [Installation section](#installation).

#### 5. Configure Environment Variables
Create a `.env` file in the project root directory with the following content:

```plaintext
PROJECT_ROOT=/path/to/project/root/
WANDB_API_KEY=this-is-optional
```
Note:
- Set `PROJECT_ROOT=/workspace` if you are using the provided Dockerfile.
- The `WANDB_API_KEY` is optional and can be omitted if you do not wish to log results to WandB.

#### 6. Evaluate Models
Run the evaluation script to test all models on the datasets of synthetic and hand-crafted presets:
```bash 
$ python src/eval.py -m model="glob(*)"
```
To run without logging the results, use:
```bash 
$ python src/eval.py -m model="glob(*)" ~wandb
```

#### 7. Generate Results
Generate the tables and figures from the paper using that summarize the evaluation results:

```bash 
$ python src/visualization/generate_tables.py ; python src/visualization/generate_umaps.py
```

The generated tables, along with additional figures, are saved under the `./results/eval` directory, while the figures of the UMAP projections are saved under the `./results/umap_projections` directory.



## Troubleshooting

If you encounter any issues, drop me an [email](mailto:paolocombes@gmail.com) or feel free to submit an issue.



## Thanks
Special shout out to [Joseph Turian](https://github.com/turian) for his initial guidance on the topic and overall methodology, and to [Gwendal le Vaillant](https://github.com/gwendal-lv) for the useful discussion on SPINVAE from which the transformer-based preset encoder is inspired.
