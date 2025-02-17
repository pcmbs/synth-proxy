from pathlib import Path
from typing import Any, Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as S
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger

from utils.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SSMLitModule(LightningModule):
    def __init__(
        self,
        estimator: nn.Module,
        synth_proxy: nn.Module,
        opt_cfg: Dict[str, Any],
        wandb_watch_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.estimator = estimator
        self.synth_proxy = synth_proxy
        self.opt_cfg = opt_cfg

        self.wandb_watch_args = wandb_watch_args

        # activates manual optimization.
        self.automatic_optimization = False
        self.num_warmup_steps = int(opt_cfg["synth_proxy"]["scheduler"]["num_warmup_steps"])

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
            self.logger.watch(
                self.synth_proxy,
                log=self.wandb_watch_args["log"],
                log_freq=self.wandb_watch_args["log_freq"],
                log_graph=False,
            )

        # initialize synth proxy LR warmup
        _, proxy_opt = self.optimizers()
        for pg in proxy_opt.optimizer.param_groups:
            pg["lr"] = self.opt_cfg["synth_proxy"]["lr"] * 1.0 / self.num_warmup_steps

    def on_train_end(self) -> None:
        if isinstance(self.logger, WandbLogger) and self.wandb_watch_args is not None:
            self.logger.experiment.unwatch(self.estimator)
            self.logger.experiment.unwatch(self.synth_proxy)

    def training_step(self, batch, batch_idx: int):
        est_opt, proxy_opt = self.optimizers()

        x, z, y = batch  # mel, parameters, audio embedding

        # parameter loss
        z_hat = self.estimator(x)
        loss_z = F.l1_loss(z_hat, z)

        # perceptual loss
        y_hat = self.synth_proxy(z_hat)
        loss_y = F.l1_loss(y_hat, y)

        loss = loss_z + loss_y

        # backward
        est_opt.zero_grad()
        proxy_opt.zero_grad()
        self.manual_backward(loss_y, retain_graph=True)
        self.manual_backward(loss)
        est_opt.step()
        proxy_opt.step()

        # linear LR warmup for synth proxy
        if self.trainer.global_step < self.num_warmup_steps:
            # _, _, proxy_warmup_sch = self.lr_schedulers()
            # proxy_warmup_sch.step()

            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.num_warmup_steps)
            for pg in proxy_opt.optimizer.param_groups:
                pg["lr"] = self.opt_cfg["synth_proxy"]["lr"] * lr_scale

        self.log("train/loss_z", loss_z, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_y", loss_y, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx: int):
        x, z, y = batch  # mel, parameters, audio embedding

        # parameter loss
        z_hat = self.estimator(x)
        loss_z = F.l1_loss(z_hat, z)

        # perceptual loss
        y_hat = self.synth_proxy(z_hat)
        loss_y = F.l1_loss(y_hat, y)

        loss = loss_z + loss_y

        self.log("val/loss_z", loss_z, on_step=False, on_epoch=True)
        self.log("val/loss_y", loss_y, on_step=False, on_epoch=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        # est_sch, proxy_sch, _ = self.lr_schedulers()
        est_sch, proxy_sch = self.lr_schedulers()

        est_sch.step(self.trainer.callback_metrics["val/loss_z"])
        # only start proxy scheduler after warmup
        if self.trainer.global_step >= self.num_warmup_steps:
            proxy_sch.step(self.trainer.callback_metrics["val/loss_y"])

    def configure_optimizers(self) -> Any:
        est_opt = torch.optim.AdamW(
            self.estimator.parameters(),
            lr=self.opt_cfg["estimator"]["lr"],
            betas=self.opt_cfg["estimator"]["betas"],
            eps=self.opt_cfg["estimator"]["eps"],
            weight_decay=self.opt_cfg["estimator"]["weight_decay"],
        )
        proxy_opt = torch.optim.AdamW(
            self.synth_proxy.parameters(),
            lr=self.opt_cfg["synth_proxy"]["lr"],
            betas=self.opt_cfg["synth_proxy"]["betas"],
            eps=self.opt_cfg["synth_proxy"]["eps"],
            weight_decay=self.opt_cfg["synth_proxy"]["weight_decay"],
        )
        optimizers = [est_opt, proxy_opt]

        est_sch = S.ReduceLROnPlateau(
            est_opt,
            mode="min",
            factor=self.opt_cfg["estimator"]["scheduler"]["factor"],
            patience=self.opt_cfg["estimator"]["scheduler"]["patience"],
        )

        proxy_sch = S.ReduceLROnPlateau(
            proxy_opt,
            mode="min",
            factor=self.opt_cfg["synth_proxy"]["scheduler"]["factor"],
            patience=self.opt_cfg["synth_proxy"]["scheduler"]["patience"],
        )

        # proxy_warmup_sch = S.LinearLR(
        #     proxy_opt,
        #     start_factor=0.05,
        #     end_factor=1,
        #     total_iters=self.num_warmup_steps,
        # )

        # schedulers = [est_sch, proxy_sch, proxy_warmup_sch]
        schedulers = [est_sch, proxy_sch]

        return optimizers, schedulers


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader, random_split
    from lightning import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor

    class estimator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 40)
            self.fc2 = nn.Linear(40, 2)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    class synth_proxy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 40)
            self.fc2 = nn.Linear(40, 10)

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
            return self.data[index], self.stats[index], self.data[index]

        def __len__(self):
            return self.len

    DEBUG = True
    estimator = estimator()
    synth_proxy = synth_proxy()
    opt_cfg = {
        "estimator": {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0.0,
            "scheduler": {"factor": 0.1, "patience": 2},
        },
        "synth_proxy": {
            "lr": 1e-5,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0.0,
            "scheduler": {"factor": 0.1, "patience": 2, "num_warmup_steps": int(160 * 0.8 / 8 * 100)},
        },
    }
    model = SSMLitModule(
        estimator=estimator,
        synth_proxy=synth_proxy,
        opt_cfg=opt_cfg,
        wandb_watch_args=None,
    )

    dataset = BoringDataset(160, 10)
    dataset_train, dataset_val = random_split(dataset, [0.8, 0.2])
    dataloader_train = DataLoader(dataset_train, batch_size=8)
    dataloader_val = DataLoader(dataset_val, batch_size=8)

    if DEBUG is False:
        trainer = Trainer(
            max_epochs=200,
            deterministic=True,
            default_root_dir=Path(__file__).parent,
            enable_checkpointing=False,
            callbacks=[LearningRateMonitor(logging_interval="step")],
            log_every_n_steps=1,
        )
    else:
        trainer = Trainer(
            max_epochs=200,
            deterministic=True,
            default_root_dir=None,
            enable_checkpointing=False,
            logger=False,
        )

    trainer.fit(model, dataloader_train, dataloader_val)
