import yaml
import mlflow.pytorch
from pathlib import Path
from lightning.pytorch.loggers import MLFlowLogger


class ExperimentConfigs:
    def __init__(self, config_source, mlflow_logger=None):
        if isinstance(config_source, str):
            with open(config_source, "r") as f:
                config_dict = yaml.safe_load(f)

        elif isinstance(config_source, dict):
            config_dict = config_source
        else:
            raise ValueError("config_source must be a file path or a dictionary")

        self._set_attributes(config_dict)  # set attributes recursively

        # if mlflow_logger:
        #     self.mlflow_logger = MLFlowLogger(**self.experiment.logger.to_dict())
        #     self.mlflow_logger.log_hyperparams(self.flatten_dict())
        #     print('run id:', self.mlflow_logger.run_id)
        #     print('experiment id:',self.mlflow_logger.experiment_id)
        #     run_name, artifact_dir = get_run_info(
        #         run_id=self.mlflow_logger.run_id,
        #         experiment_id=self.mlflow_logger.experiment_id,
        #     )
        #     self.experiment.checkpoints.dirpath = artifact_dir / "checkpoints"
        #     self.experiment.logger.run_name = run_name
        #     self.experiment.logger.experiment_id = self.mlflow_logger.experiment_id
        #     self.experiment.logger.run_id = self.mlflow_logger.run_id
        #     self.experiment.logger.print()

    def _set_attributes(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):  # create a sub-config object
                sub_config = ExperimentConfigs(value)
                setattr(self, key, sub_config)
            else:
                setattr(self, key, value)

    def to_dict(self):
        """
        Recursively converts the Configs object into a dictionary.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ExperimentConfigs):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value

        return config_dict

    def delete(self, key):
        """
        Deletes a key from the configuration.
        """
        delattr(self, key)

    def print(self):
        """
        Prints the configuration parameters in a structured format.
        """
        config_dict = self.to_dict()
        self._print_dict(config_dict)

    def clone(self):
        """
        Clones the configuration.
        """
        return ExperimentConfigs(self.to_dict())
    
    def _print_dict(self, config_dict, indent=0):
        for key, value in config_dict.items():
            prefix = " " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict(value, indent + 4)
            else:
                print(f"{prefix}{key}: {value}")

    def flatten_dict(self):
        """
        Flattens the configuration into a single-level dictionary with dot-separated keys.
        """

        def _flatten_dict(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        return _flatten_dict(self.to_dict())


# def get_run_info(experiment_id=None, run_id=None, experiment_name=None, run_name=None):
#     """Returns `run_name` or `run_id` and articfact directory for a given run."""
#     client = mlflow.tracking.MlflowClient()
#     info = {}
#     if run_id and experiment_id:
#         for run in client.search_runs(experiment_ids=[experiment_id]):
#             info[run.info.run_id] = (run.info.run_name, Path(run.info.artifact_uri))
#         return info[run_id]
#     elif run_name and experiment_name:
#         experiment = client.get_experiment_by_name(experiment_name)
#         experiment_id = experiment.experiment_id
#         for run in client.search_runs(experiment_ids=[experiment_id]):
#             info[run.info.run_name] = (run.info.run_id, Path(run.info.artifact_uri))
#         return info[run_name]
#     else:
#         return None
