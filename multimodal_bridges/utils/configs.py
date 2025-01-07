import yaml
import os

class ExperimentConfigs:
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

    def save(self, path, name='config.yaml'):
        """
        Saves the configuration parameters to a YAML file.
        """
        path = os.join(path, name)
        config_dict = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

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