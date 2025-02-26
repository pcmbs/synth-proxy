from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import os
import sys

from auraloss.freq import STFTLoss, MelSTFTLoss, MultiResolutionSTFTLoss
import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from optuna.trial import Trial
from scipy.io import wavfile

from ssm.loss_scheduler import LossScheduler, NopScheduler
from utils.logging import RankedLogger
from utils.synth import PresetHelper, PresetRenderer
from utils.evaluation import MFCCDist

log = RankedLogger(__name__, rank_zero_only=True)


class SSMLitModule(LightningModule):
    def __init__(
        self,
        preset_helper: PresetHelper,
        estimator: nn.Module,
        opt_cfg: Dict[str, Any],
        loss_sch_cfg: Optional[Dict[str, Any]] = None,
        synth_proxy: Optional[nn.Module] = None,
        compute_a_metrics: bool = True,
        label_smoothing: float = 0.0,
        lw_cat: float = 0.01,
        wandb_watch_args: Optional[Dict[str, Any]] = None,
        test_batch_to_export: int = 0,
        trial: Optional[Trial] = None,
    ):
        super().__init__()
        self.p_helper = preset_helper
        self.estimator = estimator
        self.synth_proxy = synth_proxy
        self.opt_cfg = opt_cfg
        self.lr = opt_cfg["optimizer_kwargs"]["lr"]
        self.num_warmup_steps = int(opt_cfg["num_warmup_steps"])
        self.wandb_watch_args = wandb_watch_args
        self.trial = trial  # for optuna
        self.test_batch_to_export = test_batch_to_export  # export to audio

        # activates manual optimization.
        self.automatic_optimization = False

        # parameter loss
        self.losses_fn = {
            "num": nn.L1Loss(),
            "cat": nn.CrossEntropyLoss(label_smoothing=label_smoothing),
            "bin": nn.BCEWithLogitsLoss(),
            "perc": nn.L1Loss(),
        }

        # scaling factors
        self.lw_cat = lw_cat  # for categorical loss
        self.lw_mfccd = 0.25  # for mfcc distance
        # self.lw_a_base = 10.0 # for perceptual loss (done in loss scheduler)

        # loss scheduler
        self.loss_sch = {}
        if loss_sch_cfg is not None:
            self.loss_sch["param"], self.loss_sch["perc"] = LossScheduler(loss_sch_cfg).get_schedules()
        else:
            self.loss_sch["param"], self.loss_sch["perc"] = NopScheduler(1.0), NopScheduler(1.0)

        # set synth proxy to None if only parameter loss
        if isinstance(self.loss_sch["perc"], NopScheduler):
            if self.loss_sch["perc"].value == 0.0:
                self.synth_proxy = None

        # synthesizer parameter indices
        pred_idx, target_idx = self._initialize_synth_parameters_idx()
        self.params_idx = {
            "pred": {"num": pred_idx[0], "cat": pred_idx[1], "bin": pred_idx[2]},
            "target": {"num": target_idx[0], "cat": target_idx[1], "bin": target_idx[2]},
        }

        # validation and test metrics
        self.p_metrics = {
            "num_mae": nn.L1Loss(),
            "bin_acc": BinaryAccuracy(),
            "cat_acc": MulticlassAccuracy(num_classes=max(len(t) for t in self.params_idx["pred"]["cat"])),
        }

        self.a_metrics = {
            "stft": STFTLoss(w_log_mag=1.0, w_sc=1.0, w_lin_mag=0.0, mag_distance="L1", output="loss"),
            "mstft": MultiResolutionSTFTLoss(
                w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0, mag_distance="L1", output="loss"
            ),
            "mel": partial(  # pass sample rate when getting dataset_cfg
                MelSTFTLoss,
                w_log_mag=1.0,
                w_sc=1.0,
                w_lin_mag=0.0,
                n_mels=128,
                mag_distance="L1",
                output="loss",
            ),
            "mfccd": partial(  # pass sample rate when getting dataset_cfg
                MFCCDist, n_mels=128, n_mfcc=40, distance="L1"
            ),
        }
        # whether or not to compute audio-based metrics during validation
        self.compute_a_metrics = compute_a_metrics

        # Preset renderer
        self.dataset_cfg = None
        self.renderer = None

    def _initialize_synth_parameters_idx(self):
        # index of parameters per type in the target vector
        num_idx_target = torch.tensor(self.p_helper.used_num_parameters_idx, dtype=torch.long)
        cat_idx_target = torch.tensor(self.p_helper.used_cat_parameters_idx, dtype=torch.long)
        bin_idx_target = torch.tensor(self.p_helper.used_bin_parameters_idx, dtype=torch.long)

        # index of parameters per type in the prediction vector
        # (different since there are C outputs for a categorical parameter with C categories)
        num_idx_pred = []
        cat_idx_pred = []
        bin_idx_pred = []

        offset = 0
        for p in self.p_helper.used_parameters:
            if p.type == "num":
                num_idx_pred.append(offset)
                offset += 1
            elif p.type == "cat":
                cat_idx_pred.append(torch.arange(offset, offset + p.cardinality, dtype=torch.long))
                offset += p.cardinality
            else:  # bin
                bin_idx_pred.append(offset)
                offset += 1

        num_idx_pred = torch.tensor(num_idx_pred, dtype=torch.long)
        bin_idx_pred = torch.tensor(bin_idx_pred, dtype=torch.long)
        return (num_idx_pred, cat_idx_pred, bin_idx_pred), (num_idx_target, cat_idx_target, bin_idx_target)

    def decode_presets(self, x: torch.Tensor):
        presets = torch.zeros(x.shape[0], self.p_helper.num_used_parameters, device=x.device)

        # Pass the predictions through sigmoid for numerical parameters since in [0, 1]
        if self.p_helper.has_num_parameters:
            presets[:, self.params_idx["target"]["num"]] = F.sigmoid(x[:, self.params_idx["pred"]["num"]])

        # 0 or 1 for binary parameters (discrimination threshold at 0.5)
        if self.p_helper.has_bin_parameters:
            presets[:, self.params_idx["target"]["bin"]] = F.sigmoid(
                x[:, self.params_idx["pred"]["bin"]]
            ).round()

        # argmax for categorical parameters
        if self.p_helper.has_cat_parameters:
            for i, j in zip(self.params_idx["pred"]["cat"], self.params_idx["target"]["cat"]):
                # pred[:, i] = F.softmax(pred[:, i], dim=1)
                presets[:, j] = torch.argmax(x[:, i], dim=1)

        return presets

    def parameter_loss_fn(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # compute the loss for each type of parameter or return 0
        # if the current synthesizer does not possess that parameter type

        # regression loss for numerical parameters (pass through sigmoid first since in [0, 1])
        if self.p_helper.has_num_parameters:
            loss_num = self.losses_fn["num"](
                F.sigmoid(pred[:, self.params_idx["pred"]["num"]]),
                target[:, self.params_idx["target"]["num"]],
            )
        else:
            loss_num = torch.tensor(0.0)

        # binary loss for binary parameters
        if self.p_helper.has_bin_parameters:
            loss_bin = self.losses_fn["bin"](
                pred[:, self.params_idx["pred"]["bin"]], target[:, self.params_idx["target"]["bin"]]
            )
        else:
            loss_bin = torch.tensor(0.0)

        # categorical loss for categorical parameters
        if self.p_helper.has_cat_parameters:
            loss_cat = sum(
                [
                    self.losses_fn["cat"](pred[:, i], target[:, j].to(dtype=torch.long))
                    for i, j in zip(self.params_idx["pred"]["cat"], self.params_idx["target"]["cat"])
                ]
            )
        else:
            loss_cat = torch.tensor(0.0)

        return loss_num, loss_bin, loss_cat

    def training_step(self, batch, batch_idx: int):
        opt = self.optimizers()

        p, a, m = batch  # synth parameters, audio embedding, mel spectrogram

        # parameter loss
        p_hat = self.estimator(m)
        loss_num, loss_bin, loss_cat = self.parameter_loss_fn(p_hat, p)
        loss_cat = loss_cat * self.lw_cat
        lw_p = self.loss_sch["param"](self.trainer.global_step)
        loss_p = loss_num + loss_bin + loss_cat

        # perceptual loss
        if callable(self.synth_proxy):
            presets_hat = self.decode_presets(p_hat)  # probabilities -> presets
            a_hat = self.synth_proxy(presets_hat)
            loss_a = self.losses_fn["perc"](a_hat, a)
        else:
            loss_a = 0.0
        lw_a = self.loss_sch["perc"](self.trainer.global_step)

        # composite loss
        loss = loss_p * lw_p + loss_a * lw_a

        # backward (only optimize estimator)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # linear LR warmup for synth proxy
        if self.trainer.global_step < self.num_warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.num_warmup_steps)
            for pg in opt.optimizer.param_groups:
                pg["lr"] = self.lr * lr_scale

        # logging
        if self.p_helper.has_num_parameters:
            self.log("train/loss/num", loss_num, on_step=True, on_epoch=True)
        if self.p_helper.has_bin_parameters:
            self.log("train/loss/bin", loss_bin, on_step=True, on_epoch=True)
        if self.p_helper.has_cat_parameters:
            self.log("train/loss/cat", loss_cat, on_step=True, on_epoch=True)
        self.log("train/loss/param", loss_p, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss/total", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lw/param", lw_p, on_step=True, on_epoch=True)
        if callable(self.synth_proxy):
            self.log("train/loss/perc", loss_a, prog_bar=True, on_step=True, on_epoch=True)
            self.log("lw/perc", lw_a, on_step=True, on_epoch=True)

    def on_train_start(self) -> None:
        if not isinstance(self.logger, WandbLogger) or self.wandb_watch_args is None:
            log.info("Skipping watching model.")
        else:
            self.logger.watch(
                self.estimator,
                log=self.wandb_watch_args["log"],
                log_freq=self.wandb_watch_args["log_freq"],
                log_graph=False,
            )
        # set the synth proxy to eval mode
        if callable(self.synth_proxy):
            self.synth_proxy.eval()

        # initialize the linear LR warmup
        opt = self.optimizers()
        for pg in opt.optimizer.param_groups:
            pg["lr"] = self.lr * 1.0 / self.num_warmup_steps

    def on_train_end(self) -> None:
        if isinstance(self.logger, WandbLogger) and self.wandb_watch_args is not None:
            self.logger.experiment.unwatch(self.estimator)

    def validation_step(self, batch, batch_idx: int):
        p, a, m = batch  # synth parameters, audio embedding, mel spectrogram

        # Here we do not want to weight the parameter and perceptual losses
        # to know how it evolves over time depending on the loss schedulers

        # parameter loss
        p_hat = self.estimator(m)
        loss_num, loss_bin, loss_cat = self.parameter_loss_fn(p_hat, p)
        loss_cat = loss_cat * self.lw_cat
        loss_p = loss_num + loss_bin + loss_cat

        # probabilities -> presets
        # for perceptual loss and accuracy metrics
        preset_pred = self.decode_presets(p_hat)

        # perceptual loss
        if callable(self.synth_proxy):
            a_hat = self.synth_proxy(preset_pred)
            loss_a = self.losses_fn["perc"](a_hat, a)
        else:
            loss_a = torch.tensor(0.0)

        loss = loss_p + loss_a

        metrics_dict = {
            "loss/total": loss.item(),
            "loss/param": loss_p.item(),
            "loss/num": loss_num.item(),
            "loss/perc": loss_a.item(),
            "loss/bin": loss_bin.item(),
            "loss/cat": loss_cat.item(),
        }

        # Compute validation metrics (don't return audio)
        metrics_dict.update(self._compute_metrics(preset_pred, p))

        # log all metrics
        for name, val in metrics_dict.items():
            self.log(f"val/{name}", val, on_step=False, on_epoch=True)

    def on_validation_start(self):
        if self.dataset_cfg is None:
            self.dataset_cfg = self._get_dataset_cfg()

        if sys.platform in ["linux", "linux2"] and self.dataset_cfg["synth"] == "dexed":
            log.info("Dexed not supported on linux, skipping audio-based metrics.")
            self.compute_a_metrics = False

        # instantiate renderer if required
        if self.compute_a_metrics and self.renderer is None:
            self._instantiate_renderer()

        # get sample rate to instantiate mel-based metrics
        if isinstance(self.a_metrics.get("mel"), partial):
            self.a_metrics["mel"] = self.a_metrics["mel"](sample_rate=self.dataset_cfg["sample_rate"])
        if isinstance(self.a_metrics.get("mfccd"), partial):
            self.a_metrics["mfccd"] = self.a_metrics["mfccd"](sr=self.dataset_cfg["sample_rate"])

    def on_validation_epoch_end(self):
        # only start LR scheduler after warmup
        sch = self.lr_schedulers()
        if sch is not None and self.trainer.global_step >= self.num_warmup_steps:
            sch.step(self.trainer.callback_metrics["val/metrics/a_mean"])

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            self._in_domain_test_step(batch, batch_idx)
        elif dataloader_idx == 1:
            self._nsynth_test_step(batch, batch_idx)
        else:
            raise NotImplementedError()

    def on_test_start(self):
        self.on_validation_start()

    def _in_domain_test_step(self, batch, batch_idx: int):
        p, a, m = batch  # synth parameters, audio embedding, mel spectrogram

        # Here we do not want to weight the parameter and perceptual losses
        # to know how it evolves over time depending on the loss schedulers

        # parameter loss
        p_hat = self.estimator(m)
        loss_num, loss_bin, loss_cat = self.parameter_loss_fn(p_hat, p)
        loss_cat = loss_cat * self.lw_cat
        loss_p = loss_num + loss_bin + loss_cat

        # probabilities -> presets
        # for perceptual loss and accuracy metrics
        preset_pred = self.decode_presets(p_hat)

        # perceptual loss
        if callable(self.synth_proxy):
            a_hat = self.synth_proxy(preset_pred)
            loss_a = self.losses_fn["perc"](a_hat, a)
        else:
            loss_a = torch.tensor(0.0)

        loss = loss_p + loss_a

        losses_dict = {
            "loss/total": loss.item(),
            "loss/param": loss_p.item(),
            "loss/num": loss_num.item(),
            "loss/perc": loss_a.item(),
            "loss/bin": loss_bin.item(),
            "loss/cat": loss_cat.item(),
        }

        # Compute validation metrics (return audio)
        metrics_dict, audio = self._compute_metrics(preset_pred, p, return_audio=True)
        logs_dict = {**losses_dict, **metrics_dict}

        if batch_idx < self.test_batch_to_export:
            self._export_audio(audio=audio["pred"], batch_idx=batch_idx, file_suffix="p", folder_suffix="id")
            self._export_audio(
                audio=audio["target"], batch_idx=batch_idx, file_suffix="t", folder_suffix="id"
            )

        # log all metrics
        for name, val in logs_dict.items():
            self.log(f"test/id/{name}", val, on_step=False, on_epoch=True)

    def _nsynth_test_step(self, batch, batch_idx: int):
        a, i, m = batch  # audio target, indexes, mel spectrogram
        device = a.device
        # Here we do not want to weight the parameter and perceptual losses
        # to know how it evolves over time depending on the loss schedulers

        # predict presets
        p_hat = self.estimator(m)
        preset_pred = self.decode_presets(p_hat)

        audio_pred = self._render_batch_audio(preset_pred).to(device)

        metrics_dict = self._compute_a_metrics(audio_pred, a, device)

        if batch_idx < self.test_batch_to_export:
            self._export_audio(audio_pred, batch_idx, file_suffix="p", folder_suffix="nsynth")
            self._export_audio(a, batch_idx, file_suffix="t", folder_suffix="nsynth")

        # log all metrics
        for name, val in metrics_dict.items():
            self.log(f"test/nsynth/{name}", val, on_step=False, on_epoch=True)

    def _compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, return_audio: bool = False
    ) -> Dict[str, float]:
        device = pred.device
        p_metrics_dict = self._compute_p_metrics(pred, target, device)

        if self.compute_a_metrics:
            audio_pred = self._render_batch_audio(pred).to(device)
            audio_target = self._render_batch_audio(target).to(device)

            a_metrics_dict = self._compute_a_metrics(audio_pred, audio_target, device)

        else:
            a_metrics_dict = {}

        metrics_dict = {**p_metrics_dict, **a_metrics_dict}

        if return_audio:
            audio = {"pred": audio_pred, "target": audio_target}
            return metrics_dict, audio

        return metrics_dict

    def _compute_p_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, device: torch.device
    ) -> Dict[str, float]:
        for _, fn in self.p_metrics.items():
            fn.to(device)

        idx = self.params_idx["target"]

        metrics_dict = {}
        if self.p_helper.has_num_parameters:  # same as the loss
            metrics_dict["metrics/num_mae"] = self.p_metrics["num_mae"](
                F.sigmoid(pred[:, idx["num"]]), target[:, idx["num"]]
            ).item()

        if self.p_helper.has_bin_parameters:
            metrics_dict["metrics/bin_acc"] = self.p_metrics["bin_acc"](
                pred[:, idx["bin"]], target[:, idx["bin"]]
            ).item()

        if self.p_helper.has_cat_parameters:
            metrics_dict["metrics/cat_acc"] = self.p_metrics["cat_acc"](
                pred[:, idx["cat"]], target[:, idx["cat"]]
            ).item()

        return metrics_dict

    def _compute_a_metrics(
        self,
        audio_pred: torch.Tensor,
        audio_target: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, float]:
        # move metrics to device if necessary
        for _, fn in self.a_metrics.items():
            fn.to(device)

        metrics_dict = {}

        for name, fn in self.a_metrics.items():
            if name in ["stft", "mstft", "mel"]:  # add channel dim for auraloss
                metrics_dict[f"metrics/{name}"] = fn(
                    audio_pred.unsqueeze(1), audio_target.unsqueeze(1)
                ).item()
            else:
                metrics_dict[f"metrics/{name}"] = fn(audio_pred, audio_target).item()

        metrics_dict["metrics/mfccd"] = metrics_dict["metrics/mfccd"] * self.lw_mfccd
        a_metrics_mean = (
            metrics_dict["metrics/stft"]
            + metrics_dict["metrics/mstft"]
            + metrics_dict["metrics/mel"]
            + metrics_dict["metrics/mfccd"]
        ) / 4
        metrics_dict["metrics/a_mean"] = a_metrics_mean

        return metrics_dict

    def _instantiate_renderer(self):
        if self.dataset_cfg["synth"] == "talnm":
            path_to_plugin = os.environ["TALNM_PATH"]
        elif self.dataset_cfg["synth"] == "dexed":
            path_to_plugin = os.environ["DEXED_PATH"]
        elif self.dataset_cfg["synth"] == "diva":
            path_to_plugin = os.environ["DIVA_PATH"]
        else:
            raise NotImplementedError()

        self.renderer = PresetRenderer(
            synth_path=path_to_plugin,
            sample_rate=self.dataset_cfg["sample_rate"],
            render_duration_in_sec=self.dataset_cfg["render_duration_in_sec"],
            convert_to_mono=True,
            normalize_audio=False,
            fadeout_in_sec=0.1,
        )

        # set not used parameters to default values
        self.renderer.set_parameters(self.p_helper.excl_parameters_idx, self.p_helper.excl_parameters_val)

    def _get_dataset_cfg(self):
        if self.trainer.val_dataloaders is not None:
            dataloaders = self.trainer.val_dataloaders
        elif self.trainer.test_dataloaders is not None:
            dataloaders = self.trainer.test_dataloaders
        else:
            dataloaders = self.trainer.train_dataloaders

        if isinstance(dataloaders, list):
            dataloader = dataloaders[0]
        else:
            dataloader = dataloaders

        if isinstance(dataloader.dataset, torch.utils.data.Subset):
            dataset = dataloader.dataset.dataset
        else:
            dataset = dataloader.dataset
        return dataset.configs_dict

    def _render_batch_audio(self, batch: torch.Tensor) -> torch.Tensor:
        audio = []
        for p in batch:
            audio.append(self._render_audio(p))
        return torch.stack(audio)

    def _render_audio(self, synth_parameters: torch.Tensor) -> torch.Tensor:
        if self.renderer is None:
            self._instantiate_renderer()
        # set synth parameters
        self.renderer.set_parameters(self.p_helper.used_parameters_absolute_idx, synth_parameters)
        # set midi parameters
        self.renderer.set_midi_parameters(
            self.dataset_cfg["midi_note"],
            self.dataset_cfg["midi_velocity"],
            self.dataset_cfg["midi_duration_in_sec"],
        )
        # render audio
        audio_out = torch.from_numpy(self.renderer.render_note()).squeeze(0)

        return audio_out

    def _export_audio(
        self, audio: Dict[str, torch.Tensor], batch_idx: int, file_suffix: str, folder_suffix: str
    ) -> None:
        export_path = Path(self.trainer.log_dir) / f"audio_{folder_suffix}"
        export_path.mkdir(exist_ok=True, parents=True)
        for i, a in enumerate(audio):
            wavfile.write(
                export_path / f"{batch_idx}_{i}_{file_suffix}.wav",
                self.dataset_cfg["sample_rate"],
                a.cpu().numpy().T,
            )

    # def _export_audio(self, audio: Dict[str, torch.Tensor], batch_idx: int, suffix: str) -> None:
    #     export_path = Path(self.trainer.log_dir) / "audio"
    #     export_path.mkdir(exist_ok=True, parents=True)
    #     for i, (pred, target) in enumerate(zip(audio["pred"], audio["target"])):
    #         wavfile.write(
    #             export_path / f"{batch_idx}_{i}_p.wav",
    #             self.dataset_cfg["sample_rate"],
    #             pred.cpu().numpy().T,
    #         )
    #         wavfile.write(
    #             export_path / f"{batch_idx}_{i}_t.wav",
    #             self.dataset_cfg["sample_rate"],
    #             target.cpu().numpy().T,
    #         )

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.estimator.parameters(),
            **self.opt_cfg["optimizer_kwargs"],
        )
        if self.opt_cfg.get("scheduler_kwargs"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                **self.opt_cfg["scheduler_kwargs"],
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return {"optimizer": optimizer}


if __name__ == "__main__":
    # for dev/debug
    from dotenv import load_dotenv
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, random_split
    from lightning import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor

    from data.datasets.synth_dataset_pkl import SynthDatasetPkl
    from ssm.estimator_network import EstimatorNet

    load_dotenv()

    SYNTH = "diva"
    DEBUG = True
    COMPUTE_A_METRICS = False
    DEVICE = "cpu"

    RANDOM_SPLIT = [0.89, 0.11]
    BATCH_SIZE = 64
    MAX_EPOCHS = 1
    NUM_WARMUP_EPOCH = 5

    # helpers / utils
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

    gettrace = getattr(sys, "gettrace", None)
    if gettrace():
        DEBUG = True

    class synth_proxy(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, 512)
            self.fc2 = nn.Linear(512, out_dim)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    excluded_params = OmegaConf.load(PROJECT_ROOT / "configs" / "export" / "synth" / f"{SYNTH}.yaml")[
        "parameters_to_exclude_str"
    ]

    p_helper = PresetHelper(SYNTH, excluded_params)

    # datasets
    dataset = SynthDatasetPkl(
        PROJECT_ROOT / "data" / "datasets" / "eval" / f"{SYNTH}_mn20_hc_v1",
        split="train",
        has_mel=True,
        mel_norm="min_max",
        mmap=True,
    )
    dataset_train, dataset_val = random_split(dataset, RANDOM_SPLIT)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE)

    # model
    estimator = EstimatorNet(p_helper)
    synth_proxy = None  # synth_proxy(in_dim=p_helper.num_used_parameters, out_dim=dataset.embedding_dim)

    model = SSMLitModule(
        preset_helper=p_helper,
        estimator=estimator,
        synth_proxy=synth_proxy,
        opt_cfg={
            "optimizer_kwargs": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": 0.2},
            "scheduler_kwargs": {"factor": 0.5, "patience": 10},
            "num_warmup_steps": (101 if SYNTH == "diva" else 345) * 5,
        },
        compute_a_metrics=COMPUTE_A_METRICS,
    )

    if DEBUG:
        trainer = Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator=DEVICE,
            deterministic=True,
            default_root_dir=None,
            enable_checkpointing=False,
            logger=False,
        )
    else:
        trainer = Trainer(
            max_epochs=MAX_EPOCHS,
            deterministic=True,
            default_root_dir=Path(__file__).parent,
            enable_checkpointing=False,
            callbacks=[LearningRateMonitor(logging_interval="step")],
            log_every_n_steps=50,
        )

    trainer.fit(model, dataloader_train, dataloader_val)
