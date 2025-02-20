from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from optuna.trial import Trial

from ssm.loss_scheduler import LossScheduler
from utils.logging import RankedLogger
from utils.synth import PresetHelper

log = RankedLogger(__name__, rank_zero_only=True)


class SSMLitModule(LightningModule):
    def __init__(
        self,
        preset_helper: PresetHelper,
        estimator: nn.Module,
        opt_cfg: Dict[str, Any],
        loss_sch_cfg: Optional[Dict[str, Any]] = None,
        synth_proxy: Optional[nn.Module] = None,
        label_smoothing: float = 0.0,
        cat_loss_weight: float = 1.0,
        wandb_watch_args: Optional[Dict[str, Any]] = None,
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
        self.trial = trial

        # activates manual optimization.
        self.automatic_optimization = False

        # parameter loss
        self.num_loss_fn = nn.L1Loss()
        self.cat_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bin_loss_fn = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing
        self.cat_loss_weight = cat_loss_weight

        # perceptual loss
        self.perceptual_loss_fn = nn.L1Loss()

        # loss scheduler
        self.loss_sch_cfg = loss_sch_cfg
        if loss_sch_cfg is not None:
            self.loss_scheduler = LossScheduler(**loss_sch_cfg)  # TODO: finish implementing that
            self.param_sch, self.perc_sch = self.loss_scheduler.get_schedules()
        else:
            self.param_sch, self.perc_sch = (1.0, 1.0)

        # synthesizer parameter indices
        pred_idx, target_idx = self._initialize_synth_parameters_idx()
        self.num_idx_pred, self.cat_idx_pred, self.bin_idx_pred = pred_idx
        self.num_idx_target, self.cat_idx_target, self.bin_idx_target = target_idx

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

    def decode_presets(self, pred: torch.Tensor):
        presets = torch.zeros(pred.shape[0], self.p_helper.num_used_parameters, device=pred.device)

        # Pass the predictions through sigmoid for numerical parameters since in [0, 1]
        presets[:, self.num_idx_target] = F.sigmoid(pred[:, self.num_idx_pred])

        # 0 or 1 for binary parameters (discrimination threshold at 0.5)
        presets[:, self.bin_idx_target] = F.sigmoid(pred[:, self.bin_idx_pred]).round()

        # argmax for categorical parameters
        for i, j in zip(self.cat_idx_pred, self.cat_idx_target):
            # pred[:, i] = F.softmax(pred[:, i], dim=1)
            presets[:, j] = torch.argmax(pred[:, i], dim=1)
        return presets

    def parameter_loss_fn(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # regression loss for numerical parameters (pass through sigmoid first since in [0, 1])
        loss_num = self.num_loss_fn(F.sigmoid(pred[:, self.num_idx_pred]), target[:, self.num_idx_target])

        # binary loss for binary parameters
        loss_bin = self.bin_loss_fn(pred[:, self.bin_idx_pred], target[:, self.bin_idx_target])

        # categorical loss for categorical parameters
        # TODO: might need to scale down this loss...
        loss_cat = sum(
            [
                self.cat_loss_fn(pred[:, i], target[:, j].to(dtype=torch.long))
                for i, j in zip(self.cat_idx_pred, self.cat_idx_target)
            ]
        )
        return loss_num, loss_bin, loss_cat

    def training_step(self, batch, batch_idx: int):
        opt = self.optimizers()

        p, a, m = batch  # synth parameters, audio embedding, mel spectrogram

        # parameter loss
        p_hat = self.estimator(m)
        loss_num, loss_bin, loss_cat = self.parameter_loss_fn(p_hat, p)
        loss_cat = loss_cat * self.cat_loss_weight
        loss_p = loss_num + loss_bin + loss_cat

        # perceptual loss
        if callable(self.synth_proxy):
            presets_hat = self.decode_presets(p_hat)  # probabilities -> presets
            a_hat = self.synth_proxy(presets_hat)
            loss_a = self.perceptual_loss_fn(a_hat, a)
            self.log("train/loss_a", loss_a, prog_bar=True, on_step=True, on_epoch=True)
        else:
            loss_a = 0

        loss = loss_p + loss_a

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
        self.log("train/loss_num", loss_num, on_step=True, on_epoch=True)
        self.log("train/loss_bin", loss_bin, on_step=True, on_epoch=True)
        self.log("train/loss_cat", loss_cat, on_step=True, on_epoch=True)
        self.log("train/loss_p", loss_p, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx: int):
        p, a, m = batch  # synth parameters, audio embedding, mel spectrogram

        # parameter loss
        p_hat = self.estimator(m)
        loss_num, loss_bin, loss_cat = self.parameter_loss_fn(p_hat, p)
        loss_cat = loss_cat * self.cat_loss_weight
        loss_p = loss_num + loss_bin + loss_cat

        # perceptual loss
        if callable(self.synth_proxy):
            presets_hat = self.decode_presets(p_hat)  # probabilities -> presets
            a_hat = self.synth_proxy(presets_hat)
            loss_a = self.perceptual_loss_fn(a_hat, a)
            self.log("val/loss_a", loss_a, on_step=False, on_epoch=True)
        else:
            loss_a = 0

        loss = loss_p + loss_a

        # logging
        self.log("val/loss_num", loss_num, on_step=False, on_epoch=True)
        self.log("val/loss_bin", loss_bin, on_step=False, on_epoch=True)
        self.log("val/loss_cat", loss_cat, on_step=False, on_epoch=True)
        self.log("val/loss_p", loss_p, on_step=False, on_epoch=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

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

    def on_train_epoch_end(self):
        # only start LR scheduler after warmup
        sch = self.lr_schedulers()
        if sch is not None and self.trainer.global_step >= self.num_warmup_steps:
            sch.step(self.trainer.callback_metrics["val/loss"])

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
    import os

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

    RANDOM_SPLIT = [0.89, 0.11]
    BATCH_SIZE = 64
    MAX_EPOCHS = 20
    NUM_WARMUP_EPOCH = 5

    # helpers / utils
    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

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
            "num_warmup_steps": (101 if SYNTH == "diva" else 101) * 5,
        },
    )

    if DEBUG:
        trainer = Trainer(
            max_epochs=MAX_EPOCHS,
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
            log_every_n_steps=1,
        )

    trainer.fit(model, dataloader_train, dataloader_val)
