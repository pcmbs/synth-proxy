"""
Test the MRR validation integration in Lightning module
"""

from typing import Any
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from utils.evaluation import compute_mrr


class DummyDataset(Dataset):
    def __init__(self, num_samples, dim, seed=0):
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        self.presets_emb = torch.randn(num_samples, dim, generator=rng)
        self.audio_emb = torch.randn(num_samples, dim, generator=rng)
        self.num_samples = num_samples
        self.dim = dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.presets_emb[idx], self.audio_emb[idx]


class DummyModel(LightningModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 4 * out_features),
            nn.ReLU(),
            nn.Linear(4 * out_features, 4 * out_features),
            nn.ReLU(),
            nn.Linear(4 * out_features, out_features),
        )
        self.loss = nn.L1Loss()

        self.mrr_preds = []
        self.mrr_targets = None

        self.first_mrr = 0  # only for test
        self.last_mrr = 0  # only for test

    def forward(self, presets_emb):
        return self.model(presets_emb)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        presets_emb, audio_emb = batch
        out = self(presets_emb)
        loss = self.loss(out, audio_emb)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.mrr_preds.clear()
        self.mrr_targets = None

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        presets_emb, audio_emb = batch
        if batch_idx == 0:
            self.mrr_targets = audio_emb
        self.mrr_preds.append(self(presets_emb))

    def on_validation_epoch_end(self) -> None:
        num_eval, preds_dim = self.mrr_targets.shape
        # unsqueeze for torch.cdist (one target per eval) -> shape: (num_eval, 1, dim)
        targets = self.mrr_targets.unsqueeze_(1)
        # concatenate and reshape for torch.cdist-> shape (num_eval, num_preds_per_eval, dim)
        preds = torch.cat(self.mrr_preds, dim=1).view(num_eval, -1, preds_dim)
        mrr_score = compute_mrr(preds, targets, index=0, p=1)

        # only for test
        if self.current_epoch == self.trainer.check_val_every_n_epoch - 1:
            self.first_mrr = mrr_score
        if self.current_epoch == self.trainer.max_epochs - 1:
            self.last_mrr = mrr_score

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)


def test_mrr_validation():
    seed_everything(42)

    NUM_EVAL = 4
    NUM_PRESETS_PER_EVAL = 32
    NUM_SAMPLES = NUM_EVAL * NUM_PRESETS_PER_EVAL
    DIM = 5

    dataset = DummyDataset(num_samples=NUM_SAMPLES, dim=DIM, seed=1)

    train_loader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=False, num_workers=0)
    val_loader = DataLoader(dataset, batch_size=NUM_EVAL, shuffle=False, num_workers=0)

    model = DummyModel(in_features=DIM, out_features=DIM)

    trainer = Trainer(
        max_epochs=4_000,
        check_val_every_n_epoch=100,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    assert model.last_mrr > model.first_mrr and model.last_mrr > 0.5
