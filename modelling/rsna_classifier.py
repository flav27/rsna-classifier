import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule

from utils.metrics import PFBeta
from modelling.models import RSNANet


class RsnaClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = RSNANet(device=self.device)
        self.train_pfbeta = PFBeta()
        self.val_pfbeta = PFBeta()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.loss_patches = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = x['image']
        x = self.model(x)
        return x

    def training_step(self, batch, batch_index):
        x, y = batch
        """
        RSNA output is : y, y_patches, salient_map
        """
        y_global, y_patches, _ = self.forward(x)
        loss_pred = self.loss(y_global, y)
        y_pred = torch.sigmoid(y_global)
        y_tgt = y
        self.train_pfbeta.update(y_pred, y_tgt)
        loss_patches = self.loss_patches(y_patches, y)
        loss = loss_pred + loss_patches
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        """
        RSNA output is : y_global, y_patches
        """
        y_global, y_patches, _ = self.forward(x)
        pred = torch.sigmoid(y_global)
        loss_pred = self.loss(y_global, y)
        self.val_pfbeta.update(pred, y)
        loss_patches = self.loss_patches(y_patches, y)
        loss = loss_pred + loss_patches
        return loss

    def training_epoch_end(self, training_step_outputs):
        pf_beta = self.train_pfbeta.compute()
        self.log("epoch_train_pf_beta", pf_beta, prog_bar=True, sync_dist=True)
        self.train_pfbeta.reset()

    def validation_epoch_end(self, validation_step_outputs):
        # compute metrics
        val_loss = torch.tensor(validation_step_outputs).mean()
        try:
            val_pf_beta = self.val_pfbeta.compute()
        except:
            val_pf_beta = 0.0
        # log metrics
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_pf_beta", val_pf_beta, prog_bar=True, sync_dist=True)

        self.val_pfbeta.reset()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        schedulers = [
            {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.1,
                    patience=1,
                    cooldown=0,
                    min_lr=1e-8,
                ),
                "monitor": "val_pf_beta",
                "interval": "epoch",
                "reduce_on_plateau": True,
                "frequency": 1,
            }
        ]
        return [optimizer], schedulers

    def size(self) -> int:
        """
        get number of parameters in model
        """
        return sum(p.numel() for p in self.parameters())
