"""
Lightning Module for training the preset encoder.
"""

from typing import Any, Dict, Optional

from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from optuna.trial import Trial
import torch
from torch import nn

from utils.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SynthProxyFT(LightningModule):
    """
    Lightning Module for fine-tuning the preset encoder on hand-crafted data.
    Basically the same as PresetEmbeddingLitModule but without the MRR evaluation,
    and with additional LR linear warmup.
    """

    def __init__(
        self,
        synth_proxy: nn.Module,
        loss: nn.Module,
        opt_cfg=Dict,
        wandb_watch_args: Optional[Dict[str, Any]] = None,
        trial: Optional[Trial] = None,
    ):
        super().__init__()
        self.synth_proxy = synth_proxy
        self.loss = loss
        self.opt_cfg = opt_cfg
        self.lr = opt_cfg["optimizer_kwargs"]["lr"]
        self.num_warmup_steps = int(opt_cfg["num_warmup_steps"])
        self.wandb_watch_args = wandb_watch_args
        self.trial = trial

        # activates manual optimization (to activate reduce_lr_on_plateau only after linear warmup).
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.synth_proxy(x)

    def on_train_start(self) -> None:
        if not isinstance(self.logger, WandbLogger) or self.wandb_watch_args is None:
            log.info("Skipping watching model.")
        else:
            self.logger.watch(
                self.synth_proxy,
                log=self.wandb_watch_args["log"],
                log_freq=self.wandb_watch_args["log_freq"],
                log_graph=False,
            )
        # initialize LR warmup
        opt = self.optimizers()
        for pg in opt.optimizer.param_groups:
            pg["lr"] = self.lr * 1.0 / self.num_warmup_steps

    def training_step(self, batch, batch_idx: int):
        opt = self.optimizers()

        preset, audio_embedding = batch
        preset_embedding = self(preset)
        loss = self.loss(preset_embedding, audio_embedding)

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # linear LR warm up lr without a scheduler
        if self.trainer.global_step < self.num_warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.num_warmup_steps)
            for pg in opt.optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # only start LR scheduler after warmup
        sch = self.lr_schedulers()
        if sch is not None and self.trainer.global_step >= self.num_warmup_steps:
            sch.step(self.trainer.callback_metrics["val/loss"])

    def on_train_end(self) -> None:
        if isinstance(self.logger, WandbLogger) and self.wandb_watch_args is not None:
            self.logger.experiment.unwatch(self.synth_proxy)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        preset, audio_embedding = batch
        preset_embedding = self(preset)
        # Val Loss
        loss = self.loss(preset_embedding, audio_embedding)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.synth_proxy.parameters(),
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
    from pathlib import Path
    from lightning import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split

    DEBUG = False

    RANDOM_SPLIT = [0.8, 0.2]
    BATCH_SIZE = 8
    MAX_EPOCHS = 200
    LOSS_FN = nn.L1Loss()
    OPT_CFG = {
        "optimizer_kwargs": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": 0.2},
        # "scheduler_kwargs": {"factor": 0.5, "patience": 5},
        "num_warmup_steps": 160 * 0.8 / 8 * 100,
    }

    # net and dataset
    class synth_proxy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 40)
            self.fc2 = nn.Linear(40, 2)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    class BoringDataset(Dataset):
        def __init__(self, num_samples, num_feats):
            self.len = num_samples
            self.stats = torch.randn(num_samples, 2) * 2
            self.stats[:, 1] = self.stats[:, 1].abs()
            # generate gaussian random with mean_var tensor
            self.data = []
            for i in range(num_feats):
                self.data.append(torch.normal(self.stats[:, 0], self.stats[:, 1]))
            self.data = torch.stack(self.data, dim=1)

        def __getitem__(self, index):
            return self.data[index], self.stats[index]

        def __len__(self):
            return self.len

    # dataloaders
    dataset = BoringDataset(160, 10)
    dataset_train, dataset_val = random_split(dataset, RANDOM_SPLIT)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE)

    # model and trainer
    synth_proxy = synth_proxy()
    model = SynthProxyFT(synth_proxy, LOSS_FN, OPT_CFG)
    if DEBUG is False:
        trainer = Trainer(
            max_epochs=MAX_EPOCHS,
            deterministic=True,
            default_root_dir=Path(__file__).parent,
            enable_checkpointing=False,
            callbacks=[LearningRateMonitor(logging_interval="step")],
            log_every_n_steps=1,
        )
    else:
        trainer = Trainer(
            max_epochs=MAX_EPOCHS,
            deterministic=True,
            default_root_dir=None,
            enable_checkpointing=False,
            logger=False,
        )

    # start training
    trainer.fit(model, dataloader_train, dataloader_val)
