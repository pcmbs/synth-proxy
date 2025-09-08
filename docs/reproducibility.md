# Reproducibility

We provide step-by-step instructions to reproduce the results presented in the paper.

→ The project was developed on a Windows laptop equipped with an Intel i7 CPU and an RTX 2060 GPU (6GB VRAM), using Docker (through WSL2) with the provided Dockerfile. Most of the training was performed on a NVIDIA Tesla P100. Additional testing were conducted in a virtual environment using Python 3.10.11 and pip as dependencies manager. 

---

## Installation & environment setup

The project supports both **pip** and **Docker**.

### Cloning the repository
```bash
git clone https://github.com/pcmbs/synth-proxy.git
cd synth-proxy
```

### Using pip
```bash
python -m venv .venv
source .venv/bin/activate  # Use .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Using Docker

1. Build the image:

    ```bash
    docker build \
      --build-arg UNAME=$(whoami) \
      --build-arg UID=$(id -u) \
      --build-arg GID=$(id -g) \
      -t synth-proxy:local .
    ```
2. Run:

    ```bash
    bash scripts/docker-run.sh
    ```

    If you hit permission issues:

    ```bash
    sudo chown -R $(id -u):$(id -g) .
    ```

### Environment configuration

Before running the project, create a `.env` file in the project root:

```plaintext
PROJECT_ROOT=/path/to/project/root
WANDB_API_KEY=optional
DEXED_PATH=/path/to/dexed.vst3
DIVA_PATH=/path/to/diva.vst3
TALNM_PATH=/path/to/talnoisemaker.vst3
```

Note:
- Set `PROJECT_ROOT=/workspace` if you are using the provided Dockerfile.
- The `WANDB_API_KEY` is optional and can be omitted if you do not wish to log results to WandB.
- Synthesizers are only required if you plan to generate new datasets.

---

## Synth proxies benchmark repro

1. **Download datasets**
   [Download link](https://e.pcloud.link/publink/show?code=kZ4K9MZhrJlXX1OtNmVTYJiaGl7myPj0De7) → place `eval` into `data/datasets`.

2. **Download checkpoints**
   [Download link](https://e.pcloud.link/publink/show?code=kZkK9MZgyvowLICDzfmuQmiLltCgXiX31Ek) → place `.ckpt` into `checkpoints/`.

3. **Evaluate models**

   ```bash
   python src/eval.py -m model="glob(*)"
   ```

   To run without logging the results, use:

   ```bash
   python src/eval.py -m model="glob(*)" ~wandb
   ```

4. **Generate results (tables + UMAPs)**

   ```bash
   python src/visualization/generate_tables.py
   python src/visualization/generate_umaps.py
   ```

Results saved under `results/eval` and `results/umap_projections`.

---

## Sound matching repro
See [Sound Matching](experiments.md#sound-matching-task) to download the datasets and checkpoints. Then, as explained previously, the evaluation can be performed using the following command:
```bash
python src/ssm/eval.py -m ckpt="glob(*)"
```
Currently, the easiest way to display the results is to download the results on the generated wandb dashboard, and execute the script.

---
