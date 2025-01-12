import yaml
import os
from utils.misc import SimpleLogger as log


class ExperimentConfigs:
    def __init__(self, config_source):
        if isinstance(config_source, str):
            with open(config_source, "r") as f:
                config_dict = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            config_dict = config_source
        else:
            raise ValueError("config_source must be a file path or a dictionary")

        self._set_attributes(config_dict)

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

    def save(self, path, name="config.yaml"):
        """
        Saves the configuration parameters to a YAML file.
        """
        path = os.path.join(path, name)
        if not os.path.exists(path):
            log.info(f"Saving configuration to {path}")
            with open(path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)

    def remove(self, key):
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

    def update(self, config_new, verbose=False):
        """
        Updates the configuration with the given dictionary or ExperimentConfigs object.
        """
        if config_new is not None:
            if isinstance(config_new, ExperimentConfigs):
                config_new = config_new.to_dict()
            else:
                config_new = ExperimentConfigs(config_new).to_dict()

            def recursive_update(current, new):
                for key, value in new.items():
                    if isinstance(value, dict) and isinstance(current.get(key), dict):
                        recursive_update(current[key], value)
                    else:
                        current[key] = value
                        if verbose:
                            log.info(f"config update `{key}` -> {value}")

            current_config = self.to_dict()
            recursive_update(current_config, config_new)
            self._set_attributes(current_config)

    def _print_dict(self, config_dict, indent=0):
        for key, value in config_dict.items():
            prefix = " " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict(value, indent + 4)
            else:
                print(f"{prefix}{key}: {value}")


progress_bar = {
    "description": "green_yellow",
    "progress_bar": "green1",
    "progress_bar_finished": "green1",
    "progress_bar_pulse": "#6206E0",
    "batch_progress": "green_yellow",
    "time": "grey82",
    "processing_speed": "grey82",
    "metrics": "grey82",
    "metrics_text_delimiter": "\n",
    "metrics_format": ".3e",
}
