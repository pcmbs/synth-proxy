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


## Installation

This project can be installed using either pip, conda, or Docker. Once you cloned the repository, follow the instructions below based on your preferred method.

### Installation using pip

It is recommended to install the project within a virtual environment. Follow these steps to set up the project using pip:

1. Create a virtual environment and activate it:
```bash
python -m venv .venv source .venv/Scripts/activate
```

2. Install the required packages:
```bash
python -m pip install -r requirements.txt
```

### Installation using conda

You can set up the project environment directly from the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```

### Installation using Docker

You can build a Docker image using the provided Dockerfile:
```bash
docker build -t synth_proxy:<tag> .
```
when running the container, you will need to mount the following folders/files:
```bash
docker run -it -d --network=host --gpus all --shm-size=8g --name "$CONTAINER_NAME" \
        --env-file /path/to/.env \
        --mount type=bind,src=/path/to/logs,dst=/workspace/logs \
        --mount type=bind,src=/path/to/datasets,dst=/workspace/data/datasets \
        --mount type=bind,src=/path/to/checkpoints,dst=/workspace/checkpoints \
        --mount type=bind,src=path/to/configs,dst=/workspace/configs \
        --mount type=bind,src=path/to/src,dst=/workspace/src \
        --mount type=bind,src=path/to/tests,dst=/workspace/tests \
```

### Environment Configuration

Before running the project, ensure you have a `.env` file in the root directory with the necessary environment variables set:
```plaintext 
PROJECT_ROOT=/path/to/project/root/ 
WANDB_API_KEY=<can-be-omited> 
DEXED_PATH=/path/to/dexed/vst3
DIVA_PATH=/path/to/diva/vst3 
TALNM_PATH=/path/to/talnoisemaker/vst3 
```
You do not need to install and link the synthesizers if you are not generating new datasets but are using the provided ones instead.

## Reproducibility

This section provide a detailed guide to reproduce the results presented in this project.

### Step-by-Step Guide

1. **Clone the Repository:**
    Start by cloning this repository to your local machine:
   ```bash 
   git clone https://github.com/pcmbs/preset-embedding.git cd your-repository-directory
   ```
2. **Install Dependencies:**
   Install all required dependencies following the instructions from in the [Installation section](#installation).

3. **Download Datasets:**
   Download the evaluation datasets used for testing the models:
   [Download Evaluation Datasets](https://e.pcloud.link/publink/show?code=kZ4K9MZhrJlXX1OtNmVTYJiaGl7myPj0De7)

4. **Download Model Checkpoints:**
   Download the pretrained model checkpoints:
   [Download Model Checkpoints](https://e.pcloud.link/publink/show?code=kZkK9MZgyvowLICDzfmuQmiLltCgXiX31Ek)

5. **Evaluate Models:**
   Run the evaluation script to test all models and reproduce the results:
   ```bash 
   python src/eval.py -m model="glob(*)"
   ```

6. **Generate Tables:**
   Generate the tables that summarize the evaluation results:
   ```bash 
   python src/visualization/generate_tables.py
   ```
   The generated tables (along with additional figures) are saved under the `./results/eval` directory.

7. **UMAP Projections:**
   Produce the plots used in the research for a visual representation of the results:
   ```bash
   python src/visualization/generate_umaps.py
   ```

### Additional Notes

- Ensure that all paths and environment variables are correctly set as per the instructions in the `.env` file.
- If you encounter any issues, feel free to submit an issue on the repository.

## TODO: README
- [ ] Data generation
- [ ] Training 
- [ ] HPO
- [ ] Evaluation


