from comet_ml import ExistingExperiment
import os
import json
import lightning.pytorch as L
from pytorch_lightning.loggers import CometLogger
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from utils.configs import ExperimentConfigs, progress_bar_theme
from utils.callbacks import (
    ModelCheckpointCallback,
    MetricLoggerCallback,
    JetsGenerativeCallback,
)
from utils.dataloader import DataloaderModule
from data.particle_clouds.jets import JetDataModule
from multimodal_bridge_matching import MultiModalBridgeMatching


def get_from_json(key, path, name='metadata.json'):
    path = os.path.join(path, name)
    with open(path, 'r') as f:
        file = json.load(f)
    return file[key]

class ExperimentPipeline:
    """
    A robust pipeline for configuring, training, and testing PyTorch Lightning models.
    """

    def __init__(
        self,
        config: str = None,
        experiment_path: str = None,
        load_checkpoint: str = "last.ckpt",
        accelerator: str = "gpu",
        strategy: str = "ddp",
        devices: str = "auto",
        num_nodes: int = 1,
    ):
        """
        Initialize the pipeline with configurations and components.

        Args:
            config_path (str): Path to the config file (used for training from scratch).
            experiment_path (str): Path to a saved experiment for resuming/inference training (optional).
            load_checkpoint (str): Name of the checkpoint file to load only if experiment_path is provided.
            accelerator (str): Type of accelerator to use (e.g., "gpu").
            strategy (str): Training strategy (e.g., "ddp").
            devices (str): Devices to use (e.g., "auto").
            num_nodes (int): Number of nodes for distributed training.
            log_every_n_steps (int): Logging frequency.
            val_check_interval (float): Validation check interval.
            config_update (dict): override model/data/train config matching keys.
        """

        self.checkpoint_path = None
        self.accelerator = accelerator
        self.strategy = strategy
        self.devices = devices
        self.num_nodes = num_nodes
        self.config_update = None

        if config and not experiment_path:
            print("INFO: starting new experiment")
            self.config = self._load_config(config)
            self.model = self._setup_model()
            self.logger = self._setup_logger()
        else:
            print(
                "INFO: resuming training or performing inference from existing experiment."
            )
            self.config_update = config
            assert experiment_path, "provide experiment path "
            assert load_checkpoint, "provide checkpoint name to load"
            self.experiment_path = experiment_path
            self.checkpoint_path = os.path.join(experiment_path, "checkpoints", load_checkpoint)
            self.model = MultiModalBridgeMatching.load_from_checkpoint(
                self.checkpoint_path
            )
            self.config = self.model.config
            self.config.path = experiment_path
            self.logger = self._setup_logger_from_experiment()

        self.callbacks = self._setup_callbacks()

    def train(self):
        """
        Only train the model using the configured Trainer and DataModule.
        """
        self.config.update(self.config_update, verbose=True)
        self.dataloader = self._setup_dataloader()
        self.trainer = self._setup_trainer()
        self.trainer.fit(
            model=self.model,
            train_dataloaders=self.dataloader.train,
            val_dataloaders=self.dataloader.valid,
            ckpt_path=self.checkpoint_path,
        )

    def generate(self):
        """
        Generate new target data from (pre) trained model using test source data.
        """
        self.config.data.remove("target_path")
        self.config.data.remove("target_preprocess_continuous")  # no need to preprocess test data
        self.config.data.remove("target_preprocess_discrete")  # no need to preprocess test data
        self.config.dataloader.data_split_frac = [0.0, 0.0, 1.0]  # test only dataloader
        self.config.update(self.config_update, verbose=True)
        assert (
            self.config.data.target_path
        ), "provide a valid target test data path in config_update!"

        self.dataloader = self._setup_dataloader()
        self.trainer = self._setup_trainer()
        self.trainer.predict(self.model, dataloaders=self.dataloader.test)

    # ...helper methods

    @staticmethod
    def _load_config(config_path: str) -> ExperimentConfigs:
        """
        Load experiment configurations from the given file path.
        """
        return ExperimentConfigs(config_path)

    def _setup_logger(self):
        """
        Set up the logger based on experiment configuration.
        """
        if hasattr(self.config, "comet_logger"):
            comet_config = self.config.comet_logger.to_dict()
            logger = CometLogger(**comet_config)
            self.config.comet_logger.experiment_key = (
                logger.experiment.get_key()
            )
            logger.experiment.log_parameters(self.config.to_dict())
            return logger
        return None

    def _setup_logger_from_experiment(self):
        """
        Set up the logger when resuming training from a checkpoint.
        """
        if hasattr(self.config, "comet_logger"):
            api_key = self.config.comet_logger.api_key
            exp_key = self.config.comet_logger.experiment_key
            self.config.checkpoints.dirpath = os.path.join(
                self.experiment_path, "checkpoints"
            )
            comet_config = self.config.comet_logger.to_dict()
            try:
                experiment = ExistingExperiment(api_key=api_key, experiment_key=exp_key)
                experiment.log_parameters(self.config.to_dict())
                return CometLogger(**comet_config)
            except Exception as e:
                print(f"Failed to reconnect to existing Comet experiment: {e}")
        elif hasattr(self.config, "mlflow_logger"):
            # TODO
            pass

        print("No logger configuration found in checkpoint; logger set to False.")
        return None

    def _setup_dataloader(self) -> DataloaderModule:
        """
        Prepare the data module for training and validation datasets.
        """
        jet_data = JetDataModule(config=self.config, preprocess=True)
        return DataloaderModule(config=self.config, datamodule=jet_data)

    def _setup_model(self) -> MultiModalBridgeMatching:
        """
        Set up the model using the loaded configurations.
        """
        return MultiModalBridgeMatching(self.config)

    def _setup_callbacks(self):
        """
        Configure and return the necessary callbacks for training.
        """
        callbacks = []
        callbacks.append(
            RichProgressBar(theme=RichProgressBarTheme(**progress_bar_theme))
        )
        callbacks.append(ModelCheckpointCallback(self.config.clone()))
        callbacks.append(MetricLoggerCallback(self.config.clone()))
        callbacks.append(JetsGenerativeCallback(self.config.clone()))
        return callbacks

    def _setup_trainer(
        self,
    ) -> L.Trainer:
        """
        Configure the PyTorch Lightning trainer dynamically.
        """
        trainer_config = {
            "max_epochs": self.config.trainer.max_epochs,
            "accelerator": self.accelerator,
            "strategy": self.strategy,
            "devices": self.devices,
            "gradient_clip_val": self.config.trainer.gradient_clip_val,
            "callbacks": self.callbacks,
            "logger": self.logger,
            "num_nodes": self.num_nodes,
        }

        return L.Trainer(**trainer_config)
