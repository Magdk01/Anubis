import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from scaling_model.models.painn import PaiNN
from scaling_model.models.utils import (
    AtomwisePrediction,
    DipoleMomentPrediction,
    ElectronicSpatialExtentPrediction,
)


class PaiNNforQM9(pl.LightningModule):
    """ "
    Lightning wrapper for PaiNN for the QM9 dataset.
    """

    def __init__(
        self,
        ema_decay=0.9,
        painn_kwargs={},
        prediction_kwargs={},
        optimizer_kwargs={},
        lr_scheduler_kwargs={},
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.painn_kwargs = painn_kwargs
        self.prediction_kwargs = prediction_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.ema_val_loss = None
        self.example_input_array = (
            torch.tensor([8, 6, 6, 6, 6, 6, 6, 8, 6, 1, 1, 1, 1, 1, 1, 1, 1]),
            torch.tensor(
                [
                    [-1.9367, -1.9987, 0.1342],
                    [-1.7873, -0.8125, 0.3215],
                    [-0.5070, -0.1171, 0.2496],
                    [0.0975, 1.0183, 1.1337],
                    [1.4150, 0.2444, 1.2140],
                    [0.9229, -0.6731, 0.0458],
                    [1.3280, -0.0667, -1.2942],
                    [0.7616, 1.2588, -1.2733],
                    [-0.2487, 1.2681, -0.3002],
                    [-2.6507, -0.1535, 0.5738],
                    [-0.3977, 1.5453, 1.9420],
                    [2.3332, 0.8142, 1.0389],
                    [1.4985, -0.3062, 2.1528],
                    [1.0744, -1.7460, 0.1400],
                    [2.4107, 0.0298, -1.4204],
                    [0.9160, -0.6353, -2.1399],
                    [-1.0754, 1.9297, -0.5332],
                ]
            ),
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        )

        self.save_hyperparameters()

    def forward(self, atoms, atom_positions, graph_indexes):
        scalar_features, vector_features = self.painn(
            atoms, atom_positions, graph_indexes
        )
        pred_input = {
            "atoms": atoms,
            "atom_positions": atom_positions,
            "graph_indexes": graph_indexes,
            "scalar_features": scalar_features,
            "vector_features": vector_features,
        }
        preds = self.pred_module(**pred_input)
        return preds

    def setup(self, stage):
        target = self.trainer.datamodule.target
        target_type = self.trainer.datamodule.target_types[target]
        self.painn = PaiNN(**self.painn_kwargs)

        if target_type == "atomwise":
            y_mean, y_std, atom_refs = self.trainer.datamodule.get_target_stats(
                remove_atom_refs=True,
                divide_by_atoms=True,
            )
            self.pred_module = AtomwisePrediction(
                num_outputs=1,
                mean=y_mean,
                std=y_std,
                atom_refs=atom_refs,
                **self.prediction_kwargs,
            )
        elif target_type == "dipole_moment":
            self.pred_module = DipoleMomentPrediction(**self.prediction_kwargs)
        elif target_type == "electronic_spatial_extent":
            self.pred_module = ElectronicSpatialExtentPrediction(
                **self.prediction_kwargs
            )

    def _compute_loss(self, batch, reduction="mean"):
        input_ = {
            "atoms": batch.z,
            "atom_positions": batch.pos,
            "graph_indexes": batch.batch,
        }
        y_hat = self(**input_)
        loss = F.mse_loss(y_hat, batch.y, reduction=reduction)

        self.log(
            "predictions",
            y_hat.mean(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch.y.shape[0],
        )

        self.log(
            "targets",
            batch.y.mean(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=batch.y.shape[0],
        )

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, reduction="mean")
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.y.shape[0],
        )
        return loss

    def _shared_eval_step(self, batch, stage):
        loss = self._compute_loss(batch, reduction="none")  # [batch_size, num_targets]

        if not hasattr(self, f"{stage}_stats"):
            setattr(self, f"{stage}_stats", {"num_samples": [], "loss": []})

        eval_stats = getattr(self, f"{stage}_stats")
        eval_stats["num_samples"].append(loss.shape[0])
        eval_stats["loss"].append(loss.sum())

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, stage="test")

    def _shared_eval_epoch_end(self, stage):
        eval_stats = getattr(self, f"{stage}_stats")
        num_samples = sum(eval_stats["num_samples"])
        loss = sum(eval_stats["loss"]) / num_samples

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Clean up stats for this evaluation epoch
        delattr(self, f"{stage}_stats")

    def on_validation_epoch_end(self):
        self._shared_eval_epoch_end(stage="val")
        val_loss = self.trainer.logged_metrics["val_loss"]

        if self.ema_val_loss is None:
            self.ema_val_loss = val_loss
        else:
            self.ema_val_loss = (
                self.ema_decay * self.ema_val_loss + (1 - self.ema_decay) * val_loss
            )
        self.log(
            "ema_val_loss",
            self.ema_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log("hp/val_loss", val_loss)

    def on_test_epoch_end(self):
        self._shared_eval_epoch_end(stage="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ = {
            "atoms": batch.z,
            "atom_positions": batch.pos,
            "graph_indexes": batch.batch,
        }
        y_hat = self(**input_)

        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_kwargs)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **self.lr_scheduler_kwargs
            ),
            "monitor": "ema_val_loss",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
        return [optimizer], [scheduler]
