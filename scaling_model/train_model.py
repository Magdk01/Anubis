import hydra
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything
from scaling_model.models.utils import PredictionWriter
from scaling_model.models.painn_lightning import PaiNNforQM9
from scaling_model.data.data_module import QM9DataModule, TestDataModule

from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity


@hydra.main(
    config_path="configs",
    config_name="lightning_config",
    version_base=None,
)
def main(cfg):
    seed_everything(cfg.seed)
    # logger = loggers.TensorBoardLogger(**cfg.logger)
    # logger.log_hyperparams(cfg, {"hp/val_loss": float("inf")})
    cb = [
        callbacks.LearningRateMonitor(),
        callbacks.EarlyStopping(**cfg.early_stopping),
        callbacks.ModelCheckpoint(**cfg.model_checkpoint),
        PredictionWriter(dataloaders=["train", "val", "test"]),
    ]
    profiler = PyTorchProfiler(filename="profile_out", profile_memory=True)
    dm = TestDataModule(**cfg.data)
    model = PaiNNforQM9(**cfg.lightning_model)
    trainer = Trainer(callbacks=cb, profiler=profiler, **cfg.trainer)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")
    trainer.predict(
        model,
        dataloaders=[
            dm.train_dataloader(shuffle=False),
            dm.val_dataloader(),
            dm.test_dataloader(),
        ],
        return_predictions=False,
        ckpt_path="best",
    )


if __name__ == "__main__":
    main()
