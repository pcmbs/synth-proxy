# Experiments

Here you can find the scripts/configs for HPO, training, evaluation, and sound matching.

---
## Synthesizer proxies benchmark
### Hyperparameter Optimization (HPO)
Script: [src/hpo/run.py](./src/hpo/run.py), configs: [configs/hpo/](./configs/hpo/).

**Example**: 
```bash
python src/hpo/run.py synth=dexed m_preset=tfm_opt_b tag=dexed_tfm
```

### Training
Script: [src/train.py](./src/train.py), configs: [configs/train/](./configs/train/).

**Remarks**:
- New experiments can be added under [configs/train/experiments](./configs/train/experiments).
- Learning rate schedulers can be implemented in [src/utils/lr_schedulers.py](./src/utils/lr_schedulers.py).
- Training logs are generated in [logs/train](./logs/train).

**Example**: 
```bash
python src/train.py experiment=dexed_tfm_b
```

### Evaluation
Script: [src/eval.py](./src/eval.py), configs: [configs/eval/](./configs/eval/).

**Example**:
```bash
python src/eval.py model=dexed_tfm_b
```
**Remarks**:
- evaluation results will be saved under [logs/eval](./logs/eval).
- to perform evaluation on a newly trained model, it is required to add its evaluation config file under [configs/eval/model](./configs/eval/model) (see example there).

---

## Sound Matching Task
Scripts: [src/ssm](./src/ssm), configs: [configs/ssm](./configs/ssm).
Includes code for the finetuning the synth proxy on hand-crafted presets (HPO + training scripts), as well as the HPO, training, and evaluation of the estimator network using different loss schedulers.

### Datasets
The datasets of hand-crafted presets used to finetune the synth proxy and train the estimator network are named `dexed_mn20_hc_v1` and `diva_mn20_hc_v1` for Dexed and Diva, respectively, while the NSynth validation set is named `nsynth_valid`. All of them are available [here](https://e.pcloud.link/publink/show?code=kZUw9MZ60OxeaBeBPyr87PihHKfjSHC2qRk).

### Checkpoints
Checkpoints for 
- the finetuned synth proxy for Dexed can be found [here](https://e.pcloud.link/publink/show?code=XZpXfOZvJiNbn538dbvqN6KOxSdSfoC0mqV),
- the finetuned synth proxy for Diva can be found [here](https://e.pcloud.link/publink/show?code=XZjXfOZNxtaiylThi5iqdqMx41hOJEWyayk),
- the trained estimator networks can be found [here](http://e.pc.cd/mi9y6alK).

Downloaded checkpoints must be moved to the `checkpoints` folder.

### Evaluation
The trained estimator networks can be evaluated using the following command:
```bash
python src/ssm/eval.py -m ckpt="glob(*)"
```
The results will be saved under [logs/ssm/eval](./logs/ssm/eval).

Configs and code for exporting audio samples for qualitative 
evaluation on both in-domain and the out-of-domain nsynth datasets can be found in [src/ssm/export_audio_eval.py](./src/ssm/export_audio_eval.py).

### Training
Synth proxies can be finetuned on hand-crafted presets using the following command: 
```bash
python src/ssm/synth_proxy_ft.py synth=<synth>
```
while the estimator network can be trained using the following command:
```bash
python src/ssm/train.py synth=diva loss_sch=<loss scheduler>
```
The results will be saved under [logs/ssm/train](./logs/ssm/train).

### HPO
HPO for the optimizer hyperparameters can be performed using the following commands:
```bash
python src/ssm/hpo_synth_proxy_ft.py synth=<synth>
```
for synth proxies finetuning and
```bash
python src/ssm/hpo_estimator_net.py synth=<synth> has_perceptual_loss=false tag=loss_p
```
for the estimator network training. The results will be saved under [logs/optuna](./logs/optuna).
