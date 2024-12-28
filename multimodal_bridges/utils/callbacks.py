import os
import mlflow
import pytorch_lightning as L



class MLflowCallback(L.callbacks.Callback):
    def __init__(
        self,
        experiment_name="default",
        tracking_uri="file:./mlruns",
        run_name=None,
        artifact_subdir="artifacts",
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.artifact_subdir = artifact_subdir

    def on_fit_start(self, trainer, pl_module):
        # Set up MLflow but do not handle metrics/logging here
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)

    def on_fit_end(self, trainer, pl_module):
        # Log artifacts like checkpoints
        if trainer.checkpoint_callback:
            checkpoint_path = trainer.checkpoint_callback.best_model_path
            artifact_path = os.path.join(self.artifact_subdir, "checkpoints")
            os.makedirs(artifact_path, exist_ok=True)
            mlflow.log_artifact(checkpoint_path, artifact_path)
        mlflow.end_run()

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass