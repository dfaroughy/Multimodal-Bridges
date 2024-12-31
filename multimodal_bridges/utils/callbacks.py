import os
import mlflow
from lightning.pytorch.callbacks import Callback


class MetricLoggerCallback(Callback):
    """
    Custom MLflow Callback to handle logging and saving checkpoints as MLflow artifacts.
    """

    def __init__(self):
        super().__init__()

    def setup(self, trainer, pl_module, stage=None):
        self.artifact_dir = mlflow.get_artifact_uri("checkpoints")
        os.makedirs(self.artifact_dir, exist_ok=True)

    def on_train_end(self, trainer, pl_module):
        """
        Log the checkpoint directory to MLflow at the end of training.
        """
        if hasattr(self, "artifact_dir") and os.path.exists(self.artifact_dir):
            mlflow.log_artifact(self.artifact_dir, artifact_path="checkpoints")
        print("Training finished!")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Logs metrics at the end of each training batch.
        """
        if outputs is not None:
            pl_module.log("train_loss", outputs["loss"], on_epoch=True)
            pl_module.log(
                "train_loss_continuous", outputs["loss_individual"][0], on_epoch=True
            )
            pl_module.log(
                "train_loss_discrete", outputs["loss_individual"][1], on_epoch=True
            )
            pl_module.log(
                "train_weights_continuous", outputs["weights"][0], on_epoch=True
            )
            pl_module.log(
                "train_weights_discrete", outputs["weights"][1], on_epoch=True
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Logs metrics at the end of each validation batch.
        """
        if outputs is not None:
            pl_module.log("val_loss", outputs["loss"], on_epoch=True)
            pl_module.log(
                "val_loss_continuous", outputs["loss_individual"][0], on_epoch=True
            )
            pl_module.log(
                "val_loss_discrete", outputs["loss_individual"][1], on_epoch=True
            )
            pl_module.log(
                "val_weights_continuous", outputs["weights"][0], on_epoch=True
            )
            pl_module.log("val_weights_discrete", outputs["weights"][1], on_epoch=True)
