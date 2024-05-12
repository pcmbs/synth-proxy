"""
Lightning Module for training the preset encoder.
"""

from typing import Any, Dict, Optional, Tuple

from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn

from utils.evaluation import compute_mrr
from utils.logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PresetEmbeddingLitModule(LightningModule):
    """
    Lightning Module for training the preset encoder.
    """

    def __init__(
        self,
        preset_encoder: nn.Module,
        loss: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        lr: float = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        wandb_watch_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.preset_encoder = preset_encoder
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config

        self.wandb_watch_args = wandb_watch_args

        self.mrr_preds = []
        self.mrr_targets = None

        # self.save_hyperparameters("optimizer", "lr", "scheduler")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.preset_encoder(x)

    def _model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preset, audio_embedding = batch
        preset_embedding = self(preset)
        return preset_embedding, audio_embedding

    def on_train_start(self) -> None:
        if not isinstance(self.logger, WandbLogger) or self.wandb_watch_args is None:
            log.info("Skipping watching model.")
        else:
            self.logger.watch(
                self.preset_encoder,
                log=self.wandb_watch_args["log"],
                log_freq=self.wandb_watch_args["log_freq"],
                log_graph=False,
            )

    def training_step(self, batch, batch_idx: int):
        preset_embedding, audio_embedding = self._model_step(batch)
        loss = self.loss(preset_embedding, audio_embedding)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_end(self) -> None:
        if isinstance(self.logger, WandbLogger) and self.wandb_watch_args is not None:
            self.logger.experiment.unwatch(self.preset_encoder)

    def on_validation_epoch_start(self) -> None:
        self.mrr_preds.clear()
        self.mrr_targets = None

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        preset_embedding, audio_embedding = self._model_step(batch)
        # Val Loss
        loss = self.loss(preset_embedding, audio_embedding)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        # Get target and preds for MRR
        if batch_idx == 0:
            self.mrr_targets = audio_embedding
        self.mrr_preds.append(preset_embedding)

    def on_validation_epoch_end(self) -> None:
        # skip following if len(self.mrr_preds) == 1 since no ranking can be done, i.e.,
        # there is only a single prediction per ranking evaluation.
        # This can happen if the model is loaded from a checkpoint with
        # save_on_train_epoch_end=False (e.g., when monitoring a validation metric).
        if len(self.mrr_preds) <= 1:
            log.info("Number of predictions per ranking evaluation is less than 2. Skipping MRR evaluation.")
            # logging dummy value to avoid Lightning MisconfigurationException...
            # Note that this will overwrite the last logged value if resuming the same wandb run.
            # Hence a new wandb run should be created instead.
            self.log("val/mrr", -1, prog_bar=True, on_step=False, on_epoch=True)
            return

        num_eval, preds_dim = self.mrr_targets.shape
        # unsqueeze for torch.cdist (one target per eval) -> shape: (num_eval, 1, dim)
        targets = self.mrr_targets.unsqueeze_(1)
        # concatenate and reshape for torch.cdist-> shape (num_eval, num_preds_per_eval, dim)
        preds = torch.cat(self.mrr_preds, dim=1).view(num_eval, -1, preds_dim)
        mrr_score = compute_mrr(preds, targets, index=0, p=1)
        self.log("val/mrr", mrr_score, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(params=self.preset_encoder.parameters(), lr=self.lr)

        if self.scheduler is None:
            return {"optimizer": optimizer}

        scheduler = self.scheduler(optimizer=optimizer)

        if self.scheduler_config is None:
            scheduler_config = {"interval": "step", "frequency": 1}
        else:
            scheduler_config = self.scheduler_config

        scheduler_config["scheduler"] = scheduler

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
