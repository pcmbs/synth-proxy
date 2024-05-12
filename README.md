# Neural Proxies for Sound Synthesizers: Learning Perceptually Informed Presets Representations


## Introduction
Code source repository for my master's thesis on Neural Proxies for Sound Synthesizers.

**Overview**
> This work presents a simple method for approximating black-box sound synthesizers and aims to overcome the limitations associated with parameter loss in neural-based methods for Automatic Synthesizer Programming (nASP) in handling the highly nonlinear relationship between synthesizer parameters and synthesized audio. \
The proposed method relies on training a neural network capable of mapping synthesizer presets onto a perceptually informed embedding space defined by a pretrained audio model. More specifically, given a preset $\underline{x} \in \mathcal{P}\_s$ from a synthesizer $s$ with parameter space $\mathcal{P}\_s$, a preset encoder $f_{\underline{\theta}}$ learns to minimize the distance between its representation of $\underline{x}$ and that produced by a pretrained audio model $g_{\underline{\phi}}$, derived from the synthesized audio $s(\underline{x}, \underline{\omega})=\underline{x}_a$ with MIDI parameters $\underline{\omega} \in \Omega$. This process effectively creates a neural proxy for a given synthesizer by leveraging the audio representations learned by the pretrained model via a cross-modal knowledge distillation task. \
The effectiveness of various neural network architectures, including feedforward, recurrent, and transformer-based models, in encoding synthesizer presets was evaluated using the mean reciprocal rank and averaged $L^1$ error on both synthetic and hand-crafted presets from three popular software synthesizers. Encouraging results were obtained for all synthesizers, paving the way for future research into the application of synthesizer proxies for nASP methods focusing on non-differentiable, black-box synthesizers

<p align="center">
   <img src="https://i.imgur.com/AlHSA4n.png">  
</p>

## Reproducibility

This section provides a comprehensive guide to reproduce the results presented in this project. The project has been tested on Windows using Python 3.10.11 and on WSL2 using a Docker image built with the provided Dockerfile.

### Step-by-Step Guide

#### 1. Clone the Repository
Begin by cloning this repository to your local machine using the following command:
```bash 
git clone https://github.com/pcmbs/synth-proxy.git ; cd synth-proxy
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
- Set `PROJECT_ROOT=/workspace` if you are using the provided Dockerfile
- The `WANDB_API_KEY` is optional and can be omitted if you do not wish to log results to WandB.

#### 6. Evaluate Models
Run the evaluation script to test all models and reproduce the results:
```bash 
python src/eval.py -m model="glob(*)"
```
To run without logging the results, use:
```bash 
python src/eval.py -m model="glob(*)" ~wandb
```

#### 7. Generate Results
Generate the tables and figures from the paper using that summarize the evaluation results:

```bash 
python src/visualization/generate_tables.py ; python src/visualization/generate_umaps.py
```

The generated tables, along with additional figures, are saved under the `./results/eval` directory, while the figures of the UMAP projections
are saved under the `./results/umap_projections` directory.

### Additional Notes

If you encounter any issues, drop me an [email](mailto:paolocombes@gmail.com) or feel free to submit an issue on the repository.

## Installation

This project supports installation via pip and Docker. After cloning the repository, choose one of the following methods:

### Installation using pip

#### 1. Create a VE and activate it
   ```bash
   python -m venv .venv ; .venv/bin/activate # Use .venv\Scripts\activate on Windows
   ```

#### 2. Install the required dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Installation using Docker

#### 1. Build the Docker image
Create a Docker image, specifying your user details to avoid permission issues with mounted volumes.
   ```bash
   docker build build --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g)  -t synth-proxy:local .
   ```

#### 2. Run the container
Run the container using the provided shell script
   ```bash
   bash scripts/docker-run.sh
   ```
   Note:  Note: If you encounter permission errors related to file access, adjust the ownership of the project directory: `$ sudo chown -R $(id -u):$(id -g) .`


### Environment Configuration

Before running the project, ensure you have a `.env` file in the root directory with the necessary environment variables set:
```plaintext 
PROJECT_ROOT=/path/to/project/root/ 
WANDB_API_KEY=this-is-optional 
DEXED_PATH=/path/to/dexed/vst3
DIVA_PATH=/path/to/diva/vst3 
TALNM_PATH=/path/to/talnoisemaker/vst3 
```
Note: You do not need to install and link the synthesizers if you are not generating new datasets but are using the provided ones instead.


## TODO: README
- [ ] Data generation
- [ ] Training 
- [ ] HPO
- [ ] Evaluation


