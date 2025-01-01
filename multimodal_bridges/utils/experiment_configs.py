import yaml
import mlflow.pytorch
from pathlib import Path


class Configs:
    def __init__(self, config_source):
        if isinstance(config_source, str):
            with open(config_source, "r") as f:
                config_dict = yaml.safe_load(f)

        elif isinstance(config_source, dict):
            config_dict = config_source
        else:
            raise ValueError("config_source must be a file path or a dictionary")

        self._set_attributes(config_dict)  # set attributes recursively

    def _set_attributes(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):  # create a sub-config object
                sub_config = Configs(value)
                setattr(self, key, sub_config)
            else:
                setattr(self, key, value)

    def to_dict(self):
        """
        Recursively converts the Configs object into a dictionary.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Configs):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value
        return config_dict

    def print(self):
        """
        Prints the configuration parameters in a structured format.
        """
        config_dict = self.to_dict()
        self._print_dict(config_dict)

    def _print_dict(self, config_dict, indent=0):
        """
        Helper method to recursively print the config dictionary.
        """
        for key, value in config_dict.items():
            prefix = " " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict(value, indent + 4)
            else:
                print(f"{prefix}{key}: {value}")

    def log_config(self, logger):
        """
        Logs the configuration parameters using the provided logger.
        """
        config_dict = self.to_dict()
        self._log_dict(config_dict, logger)

    def _log_dict(self, config_dict, logger, indent=0):
        """
        Helper method to recursively log the config dictionary.
        """
        for key, value in config_dict.items():
            prefix = " " * indent
            if isinstance(value, dict):
                logger.logfile.info(f"{prefix}{key}:")
                self._log_dict(value, logger, indent + 4)
            else:
                logger.logfile.info(f"{prefix}{key}: {value}")

    def save(self, path):
        """
        Saves the configuration parameters to a YAML file.
        """
        config_dict = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

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


def get_run_info(experiment_id=None, run_id=None, experiment_name=None, run_name=None):
    """Returns `run_name` or `run_id` and articfact directory for a given run."""
    client = mlflow.tracking.MlflowClient()
    info = {}
    if run_id and experiment_id:
        for run in client.search_runs(experiment_ids=[experiment_id]):
            info[run.info.run_id] = (run.info.run_name, Path(run.info.artifact_uri))
        return info[run_id]
    elif run_name and experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        for run in client.search_runs(experiment_ids=[experiment_id]):
            info[run.info.run_name] = (run.info.run_id, Path(run.info.artifact_uri))
        return info[run_name]
    else:
        return None
