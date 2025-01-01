import os
import tempfile
import mlflow
from lightning.pytorch.callbacks import Callback

from multimodal_bridge_matching import HybridState
from data.particle_clouds.particles import ParticleClouds
from data.particle_clouds.jets import JetClassHighLevelFeatures


class JetsGenerativeCallback(Callback):
    """Callback to save generated data as MLflow artifacts after prediction."""

    def __init__(self, config):
        super().__init__()
        self.batched_gen_states = []
        self.batched_source_states = []
        self.batched_target_states = []
        self.run_id = config.experiment.logger.run_id

    def on_predict_start(self, trainer, pl_module):
        if not mlflow.active_run():
            mlflow.start_run(run_id=self.run_id, nested=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is not None:
            assert isinstance(outputs, HybridState)
            self.batched_states.append(outputs[0])
            self.batched_source_states.append(outputs[1])
            self.batched_target_states.append(outputs[2])

    def on_predict_end(self, trainer, pl_module):
        gen_sample = HybridState.cat(self.batched_gen_states)
        source_sample = HybridState.cat(self.batched_source_states)
        target_sample = HybridState.cat(self.batched_target_states)
        # self.save_state_to(gen_sample, "generated_sample.h5")
        # self.save_state_to(source_sample, "source_sample.h5")
        # self.save_state_to(target_sample, "target_sample.h5")

        gen_sample = ParticleClouds(dataset=gen_sample)
        source_sample = ParticleClouds(dataset=source_sample)
        target_sample = ParticleClouds(dataset=target_sample)

        gen_sample.postprocess(
            input_continuous=config.data.target.preprocess.continuous,
            input_discrete=config.data.target.preprocess.discrete,
            stats=config.data.target.preprocess.stats,
        )




        mlflow.end_run()

    def save_state_to(self, sample, filename):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_path = os.path.join(tmp_dir, filename)
            sample.save_to(temp_file_path)
            mlflow.log_artifact(temp_file_path, artifact_path="data")


class MetricLoggerCallback(Callback):
    """
    Custom MLflow Callback to handle logging and saving checkpoints as MLflow artifacts.
    """

    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Logs metrics at the end of each training batch.
        """
        if outputs is not None:
            loss = outputs["loss"]
            loss_0 = outputs["loss_individual"][0]
            loss_1 = outputs["loss_individual"][1]
            weights_0 = outputs["weights"][0]
            weights_1 = outputs["weights"][1]
            pl_module.log("train_loss", loss, on_epoch=True)
            pl_module.log("train_loss_continuous", loss_0, on_epoch=True)
            pl_module.log("train_loss_discrete", loss_1, on_epoch=True)
            pl_module.log("train_weights_continuous", weights_0, on_epoch=True)
            pl_module.log("train_weights_discrete", weights_1, on_epoch=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Logs metrics at the end of each validation batch.
        """
        if outputs is not None:
            loss = outputs["loss"]
            loss_0 = outputs["loss_individual"][0]
            loss_1 = outputs["loss_individual"][1]
            weights_0 = outputs["weights"][0]
            weights_1 = outputs["weights"][1]
            pl_module.log("val_loss", loss, on_epoch=True)
            pl_module.log("val_loss_continuous", loss_0, on_epoch=True)
            pl_module.log("val_loss_discrete", loss_1, on_epoch=True)
            pl_module.log("val_weights_continuous", weights_0, on_epoch=True)
            pl_module.log("val_weights_discrete", weights_1, on_epoch=True)
