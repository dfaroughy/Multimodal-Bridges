
import yaml
import random
from datetime import datetime


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

        if hasattr(self, "experiment"):
            if not hasattr(self.experiment, "type"):
                if hasattr(self.dynamics, "discrete") and not hasattr(
                    self.dynamics, "continuous"
                ):
                    self.experiment.type = "discrete"
                    assert self.data.dim.features_discrete > 0
                    self.data.dim.features_continuous = 0

                elif hasattr(self.dynamics, "continuous") and not hasattr(
                    self.dynamics, "discrete"
                ):
                    assert self.data.dim.features_continuous > 0
                    self.experiment.type = "continuous"
                    self.data.dim.features_discrete = 0

                else:
                    self.experiment.type = "multimodal"
                    assert (
                        self.data.dim.features_continuous > 0
                        and self.data.dim.features_discrete > 0
                    )

            if not hasattr(self.experiment, "name"):
                self.experiment.name = (
                    f"{self.data.source.name}_to_{self.data.target.name}"
                )

                if hasattr(self.dynamics, "continuous"):
                    self.experiment.name += f"_{self.dynamics.continuous.bridge}"

                if hasattr(self.dynamics, "discrete"):
                    self.experiment.name += f"_{self.dynamics.discrete.bridge}"

                self.experiment.name += f"_{self.model.name}"

                time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
                rnd = random.randint(0, 10000)
                self.experiment.name += f"_{time}_{rnd}"
                print(
                    "INFO: created experiment instance {}".format(self.experiment.name)
                )

            if self.experiment.type == "classifier":
                if len(self.data.train.path) > 1:
                    self.experiment.name = "multi_model"
                else:
                    self.experiment.name = "binary"

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
