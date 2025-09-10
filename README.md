
<h1 align="center">
  Neural Proxies for Sound Synthesizers: <br>
  Perceptually Informed Preset Representations
</h1>

<p align="center">
  Official repository for the paper<br>
  <strong>"Neural Proxies for Sound Synthesizers: Learning Perceptually Informed Preset Representations"</strong><br>
  published in the <em>Journal of the Audio Engineering Society (JAES)</em>.
</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker"></a>
  <a href="https://wandb.ai/"><img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases&logoColor=black" alt="WandB"></a>
  <a href="https://pcmbs.github.io/synth-proxy/"><img src="https://img.shields.io/badge/Audio_Demos-Click_Here-blue" alt="Audio Demos"></a>
  <a href="https://doi.org/10.17743/jaes.2022.0219"><img src="https://img.shields.io/badge/DOI-10.17743%2Fjaes.2022.0219-blue" alt="DOI"></a>
  <a href="https://arxiv.org/abs/arxiv_id"><img src="https://img.shields.io/badge/arXiv-Postprint-B31B1B?logo=arxiv&logoColor=white" alt="arXiv Postprint"></a>
</p>


---

## Overview

This repository provides:
- Dataset generation for synthesizer presets  
- Training of neural proxies (preset encoders)  
- Evaluation on a sound-matching downstream task  

Audio examples are available on the [project website](https://pcmbs.github.io/synth-proxy/).  

The published version of the paper is available on JAES's website [here](https://doi.org/10.17743/jaes.2022.0219), while the Author's Accepted Manuscript (AAM) is available on [arXiv]() (todo).

## Main dependencies
* [PyTorch](https://pytorch.org) + [Lightning](https://lightning.ai/docs/pytorch/stable/)
* [DawDreamer](https://github.com/DBraun/DawDreamer) (VST rendering)
* [WandB](https://wandb.ai) (logging)
* [Optuna](https://optuna.org) (HPO)
* [Hydra](https://hydra.cc) (config management)

See [requirements.txt](./requirements.txt) for the full list.

---

## Installation
Clone the repo and install via **pip** or **Docker**.

→ See [Installation & environment setup](docs/reproducibility.md#installation--environment-setup) for details.

## Supported synthesizers
Currently, the following synthesizers are supported: 

* [Dexed](https://github.com/asb2m10/dexed)
* [Diva](https://u-he.com/products/diva/)
* [TAL-NoiseMaker](https://tal-software.com/products/tal-noisemaker).

→ See [Adding synthesizers](docs/adding_synth.md) for instructions on integrating new ones.

## Audio models
Wrappers for the following audio models are available in the [src/models/audio/](./src/models/audio) directory: 
* [EfficientAT](https://github.com/fschmid56/EfficientAT_HEAR/tree/main) (used in the paper)
* [Torchopenl3](https://github.com/torchopenl3/torchopenl3)
* [PaSST](https://github.com/kkoutini/passt_hear21/tree/main)
* [Audio-MAE](https://github.com/facebookresearch/AudioMAE/tree/main)
* Mel features.

→ See [Adding audio models](docs/adding_audio_model.md) for integration instructions.

→ The code for the evaluation of the pretrained audio models can be found in its corresponding [repository](https://github.com/pcmbs/synth-proxy_audio-model-selection).

## Preset Encoders (Neural Proxies)
An overview of the implemented neural proxies can be found in [src/models/preset/model\_zoo.py](./src/models/preset/model_zoo.py).

Download pretrained checkpoints [here](https://e.pcloud.link/publink/show?code=kZkK9MZgyvowLICDzfmuQmiLltCgXiX31Ek) and place them in `checkpoints/`.

## Datasets
See [Datasets](docs/datasets.md) for download links and generation instructions of **synthetic** and **hand-crafted** preset datasets.

## Experiments
This repository provides the following **experiments**:

* **Training** and **evaluation** of synthesizer proxies.
* **Hyperparameter optimization (HPO)** with Optuna.
* **Sound matching downstream tasks** (finetuning + estimator network).

→ See [Experiments](docs/experiments.md) for scripts, configs, and usage examples.


## Reproducibility
The detailed step-by-step instructions to replicate the results from the paper, including model evaluation and visualization scripts can be found in [Reproducibility](docs/reproducibility.md).


## Citation
```bibtex
@article{combes2025neural, 
  author={Combes, Paolo and Weinzierl, Stefan and Obermayer, Klaus}, 
  journal={Journal of the Audio Engineering Society}, 
  title={Neural Proxies for Sound Synthesizers: Learning Perceptually Informed Preset Representations}, 
  year={2025}, 
  volume={73}, 
  issue={9}, 
  pages={561-577}, 
  month={September},
} 
```

## Thanks
Special shout out to [Joseph Turian](https://github.com/turian) for his initial guidance on the topic and overall methodology, and to [Gwendal le Vaillant](https://github.com/gwendal-lv) for the useful discussion on SPINVAE from which the transformer-based preset encoder is inspired.